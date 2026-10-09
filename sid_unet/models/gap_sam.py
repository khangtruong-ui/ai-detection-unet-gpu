"""
GAP-SAM: A Global Artifact Prior for Generalizable AI-Generated Image Manipulation Localization.
Adapted for distilled foundation models from the sam-distil module (EfficientSAM3: TinyViT,
EfficientViT, and RepViT backbones) with PEFT LoRA parameter-efficient fine-tuning.

Reference:
    "GAP-SAM: A Global Artifact Prior for Generalizable AI-Generated Image Manipulation Localization"
    Haozhen Yan, Siyuan Shan, Zijian Yu, Youqi Wang, Yan Hong, Jun Lan, Jianfu Zhang (arXiv:2608.20929).

Architecture Overview:
    1. Frozen VAE Reconstruction:
       A frozen pretrained AutoencoderKL (e.g., Stable Diffusion 2.1 or 1.5) generates a reconstructed
       counterpart x_rec for each observed image x, exposing global reconstruction artifacts.
    2. Dual Image Encoder Branches (from sam-distil):
       - Adaptive Branch: Processes observed image x with trainable LoRA adapters, yielding multiscale
         feature pyramid F_o and final-layer feature map F_o^L.
       - Frozen Branch: Processes reconstructed image x_rec through a frozen configuration of the same
         EfficientSAM3 vision backbone, yielding F_r and final-layer map F_r^L.
    3. Paired Artifact Encoder:
       Applies Global Average Pooling (GAP) and branch-specific LayerNorms to F_o^L and F_r^L,
       concatenates the pooled descriptors [h_o; h_r], and fuses them via a 2-layer MLP with GELU
       and intermediate LayerNorm to produce the 256-dimensional Global Artifact Prior Token t_art.
    4. Zero-Gated FiLM Conditioning:
       Modulates each level l of the FPN multiscale feature pyramid channel-wise using zero-initialized
       scalar gate alpha_f and linear projections of t_art into scale gamma^l and shift beta^l:
           F_tilde^l = F_o^l + alpha_f * (gamma^l * F_o^l + beta^l)
       Guaranteeing exact identity mapping at initialization and preventing boundary adhesion.
    5. Mask Decoding:
       Passes the modulated multiscale feature pyramid to the EfficientSAM3 grounding/mask decoder.
    6. Artifact Classifier:
       Attaches a linear classifier c(h) to pooled descriptors h_o and h_r to anchor the paired
       latent space to the real-versus-synthetic forensic axis via binary cross-entropy.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.blocks import get_submodule_device_dtype
from sid_unet.models.sam3_distil import (
    DEFAULT_SAM3_DISTIL_CHECKPOINT,
    DEFAULT_TINYVIT_CHECKPOINT,
    DEFAULT_EFFICIENTVIT_CHECKPOINT,
    SAM3DistilLoRA,
    resolve_sam3_distil_checkpoint,
)

warnings.filterwarnings("ignore", message=".*memory_attention_rope_theta.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")

logger = logging.getLogger("sid_unet.models.gap_sam")

DEFAULT_GAP_SAM_VAE_CHECKPOINT = "stabilityai/sd-vae-ft-mse"


class PairedArtifactEncoder(nn.Module):
    """
    Paired Artifact Encoder from GAP-SAM (Section 4.1).

    Applies global average pooling (GAP) and branch-specific LayerNorm to the paired
    final-layer feature maps F_o^L (observed image) and F_r^L (VAE reconstruction):
        h_o = LayerNorm_o(GAP(F_o^L))
        h_r = LayerNorm_r(GAP(F_r^L))

    Then concatenates the descriptors and applies two linear layers with GELU and LayerNorm
    in between to produce the 256-dimensional Global Artifact Prior Token t_art:
        t_art = W_2 * LayerNorm(GELU(W_1 * [h_o; h_r] + b_1)) + b_2

    Args:
        feature_dim: Dimensionality of final-layer feature maps (default: 256).
        hidden_dim: Hidden dimension of the interaction MLP (default: 512).
        token_dim: Dimension of the output artifact token t_art (default: 256).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        token_dim: int = 256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim

        # Branch-specific LayerNorms for observed and reconstructed pooled features
        self.ln_o = nn.LayerNorm(feature_dim)
        self.ln_r = nn.LayerNorm(feature_dim)

        # 2-layer interaction MLP with GELU and intermediate LayerNorm
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.act = nn.GELU()
        self.ln_mid = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, token_dim)

    def forward(
        self,
        F_o: torch.Tensor,
        F_r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Paired Artifact Encoder.

        Args:
            F_o: Observed image final-layer feature map [B, C, H, W] or pooled [B, C].
            F_r: Reconstructed image final-layer feature map [B, C, H, W] or pooled [B, C].

        Returns:
            Tuple of:
                t_art: Global Artifact Prior Token [B, token_dim].
                h_o: Pooled and normalized observed feature descriptor [B, feature_dim].
                h_r: Pooled and normalized reconstructed feature descriptor [B, feature_dim].
        """
        # 1. Global Average Pooling over spatial dimensions
        if F_o.ndim == 4:
            gap_o = F_o.mean(dim=[-2, -1])
        else:
            gap_o = F_o

        if F_r.ndim == 4:
            gap_r = F_r.mean(dim=[-2, -1])
        else:
            gap_r = F_r

        # 2. Branch-specific LayerNorm
        h_o = self.ln_o(gap_o)
        h_r = self.ln_r(gap_r)

        # 3. Concatenate and project into interaction joint space
        cat = torch.cat([h_o, h_r], dim=-1)
        z = self.fc1(cat)
        z = self.act(z)
        z = self.ln_mid(z)
        t_art = self.fc2(z)

        return t_art, h_o, h_r


class ZeroGatedFiLM(nn.Module):
    """
    Zero-Gated Feature-wise Linear Modulation (FiLM) for FPN feature pyramids (Section 4.2).

    Projects the Global Artifact Prior Token t_art into channel-wise scale gamma^l
    and bias beta^l for each FPN level l:
        gamma^l, beta^l = split(W_f^FPN * t_art + b_f^FPN)
        F_tilde^l = F_o^l + alpha_f * (gamma^l * F_o^l + beta^l)

    The shared learnable scalar gate alpha_f is initialized to zero, ensuring that
    modulation is an exact identity transformation at the start of fine-tuning.

    Args:
        token_dim: Dimension of Global Artifact Prior Token t_art (default: 256).
        channels_per_level: Channel dimensions for each FPN level (int or list of ints, default: 256).
        num_levels: Number of multiscale pyramid levels (default: 3).
    """

    def __init__(
        self,
        token_dim: int = 256,
        channels_per_level: Union[int, List[int]] = 256,
        num_levels: int = 3,
    ):
        super().__init__()
        if isinstance(channels_per_level, int):
            self.channels = [channels_per_level] * num_levels
        else:
            self.channels = list(channels_per_level)
        self.num_levels = len(self.channels)

        # Linear projections from t_art to (gamma, beta) for each FPN level
        self.projs = nn.ModuleList([
            nn.Linear(token_dim, 2 * ch) for ch in self.channels
        ])

        # Shared learnable scalar gate initialized to 0.0
        self.alpha_f = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        fpn_features: List[torch.Tensor],
        t_art: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Apply zero-gated FiLM modulation channel-wise to all multiscale FPN feature levels.

        Args:
            fpn_features: List of multiscale FPN tensors [B, C_l, H_l, W_l].
            t_art: Global Artifact Prior Token [B, token_dim].

        Returns:
            List of modulated FPN feature tensors of matching shapes.
        """
        modulated = []
        b_sz = t_art.size(0)

        for l, feat in enumerate(fpn_features):
            proj = self.projs[l](t_art)  # [B, 2 * C_l]
            ch = self.channels[l]
            gamma, beta = proj.chunk(2, dim=-1)  # [B, C_l], [B, C_l]
            gamma = gamma.view(b_sz, ch, 1, 1)
            beta = beta.view(b_sz, ch, 1, 1)

            # Zero-gated modulation
            mod_feat = feat + self.alpha_f * (gamma * feat + beta)
            modulated.append(mod_feat)

        return modulated


class ArtifactClassifier(nn.Module):
    """
    Artifact Classifier attached to pooled descriptors h_o and h_r (Section 4.3).

    Constrains the paired feature space along the authentic-vs-reconstructed axis:
        c(h) = W_c * h + b_c

    Learning Objective:
        L_art = 1/|B_real| * sum_{i in B_real} [ BCE(c(h_o, i), 0) + BCE(c(h_r, i), 1) ]

    Args:
        in_features: Input feature dimension from PairedArtifactEncoder (default: 256).
    """

    def __init__(self, in_features: int = 256):
        super().__init__()
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute binary artifact logit [B, 1]."""
        return self.classifier(h)

    def compute_loss(
        self,
        h_o: torch.Tensor,
        h_r: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute real-versus-reconstructed binary cross entropy loss.

        Args:
            h_o: Pooled descriptors from observed image [B, C].
            h_r: Pooled descriptors from VAE reconstruction [B, C].
            labels: Optional ground-truth image class labels [B]. If present, label 0 is authentic.
            masks: Optional ground-truth manipulation masks [B, 1, H, W]. Authentic images have mask sum 0.

        Returns:
            Scalar artifact classification loss tensor.
        """
        logit_o = self.forward(h_o)  # [B, 1]
        logit_r = self.forward(h_r)  # [B, 1]
        target_r = torch.ones_like(logit_r)  # Reconstruction is always synthetic (class 1)

        # Identify authentic samples in batch if labels or masks are provided
        if labels is not None:
            is_real = (labels == 0)
        elif masks is not None:
            is_real = (masks.view(masks.size(0), -1).sum(dim=1) == 0)
        else:
            is_real = None

        if is_real is not None and is_real.any():
            # Real-only formulation from paper Eq. 6
            real_logit_o = logit_o[is_real]
            real_logit_r = logit_r[is_real]
            real_target_o = torch.zeros_like(real_logit_o)
            real_target_r = target_r[is_real]
            loss_o = F.binary_cross_entropy_with_logits(real_logit_o, real_target_o)
            loss_r = F.binary_cross_entropy_with_logits(real_logit_r, real_target_r)
            return loss_o + loss_r
        else:
            # Fallback across all batch items: original image vs synthetic reconstruction
            target_o = torch.zeros_like(logit_o)
            loss_o = F.binary_cross_entropy_with_logits(logit_o, target_o)
            loss_r = F.binary_cross_entropy_with_logits(logit_r, target_r)
            return loss_o + loss_r


class GAPSAM(SAM3DistilLoRA):
    """
    GAP-SAM: Global Artifact Prior with Distilled SAM3 (EfficientSAM3) models.

    Extends SAM3DistilLoRA with:
      - Frozen VAE Reconstruction generator (SD AutoencoderKL).
      - Paired Artifact Encoder with Global Average Pooling and learned MLP fusion.
      - Zero-gated FiLM conditioning on the multiscale FPN pyramid before mask decoding.
      - Artifact Classifier constraining the paired representation to the real-vs-synthetic axis.

    Args:
        checkpoint_path: Path or Hugging Face repo ID for EfficientSAM3 pre-distilled weights.
        backbone_type: Distilled backbone architecture ('tinyvit', 'efficientvit', 'repvit').
        model_name: Backbone model variant ('11m', 'b0', 'm1.1', etc.).
        text_encoder_type: Text encoder model (default: 'MobileCLIP-S0').
        text_encoder_context_length: Context token length (default: 16).
        load_in_4bit: Whether to enable 4-bit NF4 quantization for base model (default: False).
        load_in_8bit: Whether to enable 8-bit quantization for base model (default: False).
        lora_r: LoRA rank (default: 16).
        lora_alpha: LoRA scaling factor (default: 32).
        lora_dropout: Dropout probability for LoRA layers (default: 0.05).
        lora_target_modules: Names of module layers to attach LoRA adapters to.
        prompt_text: Text conditioning prompt (default: 'tampered region').
        aux_classifier: Whether to enable auxiliary classification head (default: True).
        num_classes: Number of target classes for auxiliary head (default: 3).
        in_channels: Input image channels (default: 3).
        out_channels: Output mask channels (default: 1).
        target_size: Native input resolution for EfficientSAM3 (default: (1008, 1008)).
        input_rescale: Whether to normalize [0, 1] inputs to [-1, 1] (default: True).
        vae_pretrained_model_name_or_path: Hugging Face model hub path or local path for VAE.
        vae_subfolder: Optional subfolder for VAE (default: None or 'vae').
        freeze_vae: Whether to freeze VAE weights (default: True).
        use_dummy_vae: If True, uses lightweight dummy AutoencoderKL for fast offline testing (default: False).
        dummy_vae_channels: Block channels for dummy VAE (default: (32, 64)).
        artifact_weight: Loss weight lambda_art for artifact classification objective (default: 1.0).
        enable_artifact_classifier: Whether to enable the auxiliary artifact classifier head (default: True).
        device: Target device (default: 'auto').
        cache_dir: Optional custom Hugging Face cache directory.
        token: Optional Hugging Face authentication token.
        force_download: Whether to force re-download from Hugging Face Hub.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = DEFAULT_SAM3_DISTIL_CHECKPOINT,
        backbone_type: str = "tinyvit",
        model_name: Optional[str] = None,
        text_encoder_type: Optional[str] = "MobileCLIP-S0",
        text_encoder_context_length: int = 16,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        prompt_text: str = "tampered region",
        aux_classifier: bool = True,
        num_classes: int = 3,
        in_channels: int = 3,
        out_channels: int = 1,
        target_size: Tuple[int, int] = (1008, 1008),
        input_rescale: bool = True,
        vae_pretrained_model_name_or_path: str = DEFAULT_GAP_SAM_VAE_CHECKPOINT,
        vae_subfolder: Optional[str] = None,
        freeze_vae: bool = True,
        use_dummy_vae: bool = False,
        dummy_vae_channels: Tuple[int, ...] = (32, 64),
        artifact_weight: float = 1.0,
        enable_artifact_classifier: bool = True,
        device: Optional[Union[str, torch.device]] = "auto",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ):
        # Resolve sensible default model_name based on backbone_type if omitted
        if model_name is None or str(model_name).strip() == "":
            bb_norm = str(backbone_type).lower()
            if "efficientvit" in bb_norm:
                model_name = "b0"
            elif "repvit" in bb_norm:
                model_name = "m1.1"
            else:
                model_name = "11m"

        super().__init__(
            checkpoint_path=checkpoint_path,
            backbone_type=backbone_type,
            model_name=model_name,
            text_encoder_type=text_encoder_type,
            text_encoder_context_length=text_encoder_context_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            prompt_text=prompt_text,
            aux_classifier=aux_classifier,
            num_classes=num_classes,
            in_channels=in_channels,
            out_channels=out_channels,
            target_size=target_size,
            input_rescale=input_rescale,
            device=device,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            **kwargs,
        )

        self.vae_pretrained_model_name_or_path = vae_pretrained_model_name_or_path
        self.vae_subfolder = vae_subfolder
        self.freeze_vae = freeze_vae
        self.use_dummy_vae = use_dummy_vae
        self.dummy_vae_channels = dummy_vae_channels
        self.artifact_weight = float(artifact_weight)
        self.enable_artifact_classifier = enable_artifact_classifier
        self.last_artifact_loss: Optional[torch.Tensor] = None

        # 1. Initialize frozen VAE
        self._init_vae()

        # 2. Initialize Paired Artifact Encoder (256-dim feature maps from EfficientSAM3)
        self.paired_artifact_encoder = PairedArtifactEncoder(
            feature_dim=256,
            hidden_dim=512,
            token_dim=256,
        ).to(self._target_device)

        # 3. Initialize Zero-Gated FiLM layer for FPN (3 levels with 256 channels each)
        self.zero_gated_film = ZeroGatedFiLM(
            token_dim=256,
            channels_per_level=256,
            num_levels=3,
        ).to(self._target_device)

        # 4. Initialize Artifact Classifier
        if self.enable_artifact_classifier:
            self.artifact_classifier = ArtifactClassifier(in_features=256).to(self._target_device)
        else:
            self.artifact_classifier = None

        logger.info(
            f"Initialized GAP-SAM ({self.backbone_type}-{self.model_name}) | "
            f"Artifact Weight: {self.artifact_weight} | "
            f"Frozen VAE: {self.vae_pretrained_model_name_or_path} (dummy={self.use_dummy_vae})"
        )

    def _init_vae(self) -> None:
        """Initialize frozen VAE (AutoencoderKL) for generating reconstructions."""
        from diffusers import AutoencoderKL

        if self.use_dummy_vae:
            logger.info("Initializing lightweight dummy AutoencoderKL for offline testing...")
            blocks = list(self.dummy_vae_channels)
            self.vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D"] * len(blocks),
                up_block_types=["UpDecoderBlock2D"] * len(blocks),
                block_out_channels=blocks,
                latent_channels=4,
                layers_per_block=1,
            )
        else:
            try:
                load_kwargs: Dict[str, Any] = {}
                if self.vae_subfolder:
                    load_kwargs["subfolder"] = self.vae_subfolder
                if self.cache_dir:
                    load_kwargs["cache_dir"] = self.cache_dir
                if self.token:
                    load_kwargs["token"] = self.token

                logger.info(
                    f"Loading pretrained AutoencoderKL from '{self.vae_pretrained_model_name_or_path}'..."
                )
                self.vae = AutoencoderKL.from_pretrained(
                    self.vae_pretrained_model_name_or_path, **load_kwargs
                )
            except Exception as exc:
                logger.warning(
                    f"Could not load pretrained AutoencoderKL from '{self.vae_pretrained_model_name_or_path}': {exc}. "
                    "Falling back to initialized lightweight dummy AutoencoderKL architecture..."
                )
                blocks = list(self.dummy_vae_channels)
                self.vae = AutoencoderKL(
                    in_channels=3,
                    out_channels=3,
                    down_block_types=["DownEncoderBlock2D"] * len(blocks),
                    up_block_types=["UpDecoderBlock2D"] * len(blocks),
                    block_out_channels=blocks,
                    latent_channels=4,
                    layers_per_block=1,
                )

        if self.freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()

        self.vae = self.vae.to(self._target_device)

    def reconstruct_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate deterministic VAE reconstruction of input images.

        Normalizes input tensor from [0, 1] to [-1, 1], encodes to latent distribution mode,
        decodes back to RGB, and unnormalizes to [0, 1].

        Args:
            x: Input images [B, 3, H, W] in [0, 1].

        Returns:
            Reconstructed images [B, 3, H, W] in [0, 1].
        """
        vae_device, vae_dtype = get_submodule_device_dtype(self.vae, x.device, torch.float32)
        _, _, h, w = x.shape

        # Ensure spatial dimensions are divisible by 8 for VAE latent downsampling
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_in = x

        # Scale from [0, 1] to [-1, 1]
        x_norm = (x_in * 2.0 - 1.0).to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            posterior = self.vae.encode(x_norm).latent_dist
            z = posterior.mode()
            x_rec_norm = self.vae.decode(z).sample

        # Unnormalize back to [0, 1]
        x_rec = torch.clamp((x_rec_norm + 1.0) / 2.0, 0.0, 1.0)

        # Crop padding if applied
        if pad_h > 0 or pad_w > 0:
            x_rec = x_rec[:, :, :h, :w]

        return x_rec.to(dtype=x.dtype, device=x.device)

    def compute_artifact_loss(
        self,
        h_o: torch.Tensor,
        h_r: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Compute artifact classification loss using the attached classifier head."""
        if not self.enable_artifact_classifier or self.artifact_classifier is None:
            return None
        return self.artifact_classifier.compute_loss(h_o, h_r, labels=labels, masks=masks)

    def forward(
        self,
        x: torch.Tensor,
        x_rec: Optional[torch.Tensor] = None,
        target_masks: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        return_artifact_loss: Optional[bool] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]:
        """
        Forward pass of GAP-SAM model.

        Pipeline:
            1. Preprocess observed image x and reconstructed image x_rec (auto-computed if None).
            2. Extract multiscale FPN features F_o via adaptive branch (with trainable LoRA).
            3. Extract multiscale FPN features F_r via frozen branch (with LoRA disabled).
            4. Encode final-layer features into Global Artifact Prior Token t_art via PairedArtifactEncoder.
            5. Modulate all FPN levels channel-wise using zero-gated FiLM layer.
            6. Decode modulated FPN through EfficientSAM3 mask decoder to produce mask_logits.
            7. Compute optional auxiliary classification logits and artifact loss.

        Args:
            x: Input images [B, 3, H, W] in [0, 1].
            x_rec: Optional pre-computed VAE reconstructions [B, 3, H, W].
            target_masks: Optional ground-truth masks for authentic sample masking in artifact loss.
            target_labels: Optional ground-truth labels for authentic sample masking.
            return_artifact_loss: If True, includes artifact loss in output. Defaults to self.training.
            return_dict: If True, returns dictionary with all intermediate representations.

        Returns:
            Depending on configuration:
                - If return_dict: Dict with keys 'mask_logits', 'class_logits', 'artifact_loss', 't_art', etc.
                - If return_artifact_loss is True (e.g. during training):
                    (mask_logits, class_logits, artifact_loss) if aux_classifier else (mask_logits, artifact_loss)
                - In standard evaluation mode:
                    (mask_logits, class_logits) if aux_classifier else mask_logits
        """
        from sam3.model.data_misc import FindStage

        if return_artifact_loss is None:
            return_artifact_loss = self.training and self.enable_artifact_classifier

        b_sz, _, orig_h, orig_w = x.shape
        model_device, param_dtype = get_submodule_device_dtype(
            self.peft_model, x.device, torch.float32
        )

        # 1. Generate or prepare VAE reconstruction
        if x_rec is None:
            x_rec = self.reconstruct_image(x)

        # 2. Resize to native EfficientSAM3 target resolution (1008, 1008)
        if (orig_h, orig_w) != self.target_size:
            x_proc = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
            x_rec_proc = F.interpolate(x_rec, size=self.target_size, mode="bilinear", align_corners=False)
        else:
            x_proc = x
            x_rec_proc = x_rec

        # 3. Rescale from [0, 1] to [-1, 1] if enabled
        if self.input_rescale:
            x_proc = (x_proc - 0.5) / 0.5
            x_rec_proc = (x_rec_proc - 0.5) / 0.5

        # 4. Cast to target device and precision
        if param_dtype in (torch.float16, torch.bfloat16):
            x_proc = x_proc.to(device=model_device, dtype=param_dtype)
            x_rec_proc = x_rec_proc.to(device=model_device, dtype=param_dtype)
        else:
            x_proc = x_proc.to(device=model_device)
            x_rec_proc = x_rec_proc.to(device=model_device)

        # 5. Adaptive Branch: visual features for observed image x (trainable via LoRA)
        backbone_out_o = self.base_model.backbone.forward_image(x_proc)

        # 6. Frozen Branch: visual features for reconstructed image x_rec (adapters disabled, no grad)
        with torch.no_grad():
            with self.peft_model.disable_adapter():
                backbone_out_r = self.base_model.backbone.forward_image(x_rec_proc)

        # 7. Paired Artifact Encoder: extract t_art from final-layer feature maps F_o^L and F_r^L
        F_o_L = backbone_out_o["backbone_fpn"][-1]
        F_r_L = backbone_out_r["backbone_fpn"][-1]

        t_art, h_o, h_r = self.paired_artifact_encoder(F_o_L, F_r_L)

        # 8. Zero-Gated FiLM: modulate observed multiscale FPN pyramid before mask decoding
        modulated_fpn = self.zero_gated_film(backbone_out_o["backbone_fpn"], t_art)
        backbone_out_o["backbone_fpn"] = modulated_fpn
        backbone_out_o["vision_features"] = modulated_fpn[-1]

        # 9. Extract text prompt embeddings (cached)
        cache_key = (self.prompt_text, str(model_device))
        if cache_key not in self._text_cache:
            with torch.no_grad():
                try:
                    out_t = self.base_model.backbone.forward_text(
                        [self.prompt_text], device=model_device
                    )
                except RuntimeError as exc:
                    if "FIND was unable to find an engine" in str(exc) or "cuDNN" in str(exc):
                        with torch.backends.cudnn.flags(enabled=False):
                            out_t = self.base_model.backbone.forward_text(
                                [self.prompt_text], device=model_device
                            )
                    else:
                        raise exc
                self._text_cache[cache_key] = {k: v.clone() for k, v in out_t.items()}
        text_outputs = {k: v.clone() for k, v in self._text_cache[cache_key].items()}

        # 10. Reusable find_stage
        find_stage = FindStage(
            img_ids=torch.tensor([0], device=model_device, dtype=torch.long),
            text_ids=torch.tensor([0], device=model_device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        semantic_segs = []
        presence_logits = []

        # 11. Grounding decoder per batch item
        for i in range(b_sz):
            item_backbone = {
                "vision_features": backbone_out_o["vision_features"][i:i+1],
                "vision_pos_enc": [p[i:i+1] for p in backbone_out_o["vision_pos_enc"]],
                "backbone_fpn": [f[i:i+1] for f in backbone_out_o["backbone_fpn"]],
            }
            if "sam2_backbone_out" in backbone_out_o and backbone_out_o["sam2_backbone_out"] is not None:
                s2 = backbone_out_o["sam2_backbone_out"]
                if isinstance(s2, dict):
                    item_s2 = {}
                    for k_s2, v_s2 in s2.items():
                        if isinstance(v_s2, torch.Tensor):
                            item_s2[k_s2] = v_s2[i:i+1]
                        elif isinstance(v_s2, (list, tuple)):
                            item_s2[k_s2] = [t[i:i+1] if isinstance(t, torch.Tensor) else t for t in v_s2]
                        else:
                            item_s2[k_s2] = v_s2
                    item_backbone["sam2_backbone_out"] = item_s2
                else:
                    item_backbone["sam2_backbone_out"] = s2

            item_backbone.update(text_outputs)

            out = self.base_model.forward_grounding(
                backbone_out=item_backbone,
                find_input=find_stage,
                geometric_prompt=self.dummy_prompt,
                find_target=None,
            )

            # Retrieve mask logits
            if "semantic_seg" in out and out["semantic_seg"] is not None:
                seg = out["semantic_seg"]
            elif "pred_masks" in out and out["pred_masks"] is not None:
                if "pred_logits" in out and out["pred_logits"] is not None:
                    q_weights = torch.sigmoid(out["pred_logits"])
                    seg = (torch.sigmoid(out["pred_masks"]) * q_weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
                else:
                    seg = out["pred_masks"].mean(dim=1, keepdim=True)
            else:
                raise RuntimeError("EfficientSAM3 output contains neither semantic_seg nor pred_masks.")

            # Retrieve presence score
            if "presence_logit_dec" in out and out["presence_logit_dec"] is not None:
                pres = out["presence_logit_dec"]
            elif "presence_logit" in out and out["presence_logit"] is not None:
                pres = out["presence_logit"]
            else:
                pres = seg.mean(dim=[-2, -1])

            if pres.ndim == 1:
                pres = pres.unsqueeze(-1)

            semantic_segs.append(seg)
            presence_logits.append(pres)

        # 12. Concatenate batch results
        batch_masks = torch.cat(semantic_segs, dim=0)
        batch_pres = torch.cat(presence_logits, dim=0)

        # 13. Resize mask logits back to input resolution [B, 1, orig_h, orig_w]
        mask_logits = F.interpolate(
            batch_masks.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        # 14. Auxiliary classification head
        cls_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            pres_input = batch_pres.to(
                device=self.classifier_head.weight.device,
                dtype=self.classifier_head.weight.dtype,
            )
            cls_logits = self.classifier_head(pres_input)

        # 15. Artifact classification loss
        art_loss = None
        if self.enable_artifact_classifier and self.artifact_classifier is not None:
            art_loss = self.compute_artifact_loss(
                h_o, h_r, labels=target_labels, masks=target_masks
            )
        self.last_artifact_loss = art_loss

        # 16. Return outputs
        if return_dict:
            return {
                "mask_logits": mask_logits,
                "class_logits": cls_logits,
                "artifact_loss": art_loss,
                "t_art": t_art,
                "h_o": h_o,
                "h_r": h_r,
                "alpha_f": self.zero_gated_film.alpha_f,
            }

        if return_artifact_loss:
            if self.aux_classifier and cls_logits is not None:
                return mask_logits, cls_logits, art_loss
            return mask_logits, art_loss

        if self.aux_classifier and cls_logits is not None:
            return mask_logits, cls_logits

        return mask_logits


# Aliases
GAPSAMDistil = GAPSAM
GAP_SAM = GAPSAM
