"""
Detective SAM: Adaptive AI-Image Forgery Localization model.
Integrates Meta's SAM2 foundation model with perturbation-driven forensic cues,
lightweight multi-scale Feature Adapters (delta correction), an automatic Mask Adapter
(with downscaled coarse transformer, fine conv refinement, and spatial gating),
and the frozen SAM2 mask decoder.

Based on:
    "Detective SAM: Adaptive AI-Image Forgery Localization" (ICLR 2026)
    Gert Lek, Nicolas Van Schaik, Chaoyi Zhu, Pin-Yu Chen, Robert Birke, Lydia Y. Chen.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.blocks import get_submodule_device_dtype
from sid_unet.models.forensic_memory import (
    ForensicVectorMemory,
    stop_gradient_query_replacement,
)
from sid_unet.models.sam2 import (
    DEFAULT_SAM2_SMALL_CHECKPOINT,
    DEFAULT_SAM2_LARGE_CHECKPOINT,
    FALLBACK_SAM2_CHECKPOINT,
    resolve_sam2_checkpoint,
)

logger = logging.getLogger("sid_unet.models.detective_sam")


class ImagePerturbation(nn.Module):
    """
    Image perturbation operator creating forensic clues for diffusion-based image forgery.
    Applies Gaussian Blur and additive Gaussian Noise (or optional JPEG compression)
    as specified in Detective SAM (Section 3.2, Table 16, Table 17).

    Args:
        perturbation_type (str): Type of perturbation ('blur_noise', 'blur', 'noise', 'none').
        noise_intensity (float): Intensity level controlling sigma_blur and sigma_noise (default: 0.75).
        sigma_blur (Optional[float]): Gaussian blur standard deviation. Overrides noise_intensity if given.
        sigma_noise (Optional[float]): Gaussian noise standard deviation. Overrides noise_intensity if given.
    """

    def __init__(
        self,
        perturbation_type: str = "blur_noise",
        noise_intensity: float = 0.75,
        sigma_blur: Optional[float] = None,
        sigma_noise: Optional[float] = None,
    ):
        super().__init__()
        self.perturbation_type = str(perturbation_type).lower()

        # Map noise intensity to sigma_blur and sigma_noise per Table 17
        if sigma_blur is not None:
            self.sigma_blur = float(sigma_blur)
        else:
            self.sigma_blur = float(noise_intensity)

        if sigma_noise is not None:
            self.sigma_noise = float(sigma_noise)
        else:
            # Linear scaling per Table 17: intensity 0.75 -> sigma 0.15
            self.sigma_noise = float(noise_intensity) * 0.20

        # Construct 5x5 Gaussian blur kernel for sigma_blur
        kernel_size = 5
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        s = max(self.sigma_blur, 1e-4)
        k1d = torch.exp(-0.5 * (x / s) ** 2)
        k1d = k1d / k1d.sum()
        k2d = torch.outer(k1d, k1d)
        kernel = k2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        self.register_buffer("blur_kernel", kernel)
        self.padding = radius

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply perturbation to input batch [B, 3, H, W].
        Returns perturbed image tensor [B, 3, H, W].
        """
        if self.perturbation_type in ("none", "identity"):
            return images

        out = images

        # 1. Apply Gaussian Blur
        if "blur" in self.perturbation_type:
            kernel = self.blur_kernel.to(dtype=images.dtype, device=images.device)
            out = F.conv2d(out, kernel, padding=self.padding, groups=3)

        # 2. Apply Gaussian Noise
        if "noise" in self.perturbation_type and self.sigma_noise > 0:
            noise = torch.randn_like(out) * self.sigma_noise
            # Maintain image bounds
            if images.min() >= 0.0 and images.max() <= 1.0:
                out = torch.clamp(out + noise, 0.0, 1.0)
            else:
                out = out + noise

        return out


class FeatureAdapter(nn.Module):
    """
    Lightweight 1x1 Convolutional Feature Adapter for a single scale s.
    Takes concatenated basic and perturbed feature maps [F_s^I, F_s^{I'}] (2*Cs channels)
    and predicts residual delta correction:
        Delta F_s = A_s([F_s^I, F_s^{I'}])
        F_tilde_s = F_s^I + Delta F_s
    """

    def __init__(self, in_channels: int, width: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        # Single 1x1 conv layer as described in Detective SAM Section 3.2
        if width is not None and width > 0:
            self.adapter = nn.Sequential(
                nn.Conv2d(2 * in_channels, width, kernel_size=1),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.Conv2d(width, in_channels, kernel_size=1),
            )
            # Initialize final conv with small weights so delta starts close to 0
            nn.init.zeros_(self.adapter[-1].weight)
            if self.adapter[-1].bias is not None:
                nn.init.zeros_(self.adapter[-1].bias)
        else:
            self.adapter = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
            nn.init.zeros_(self.adapter.weight)
            if self.adapter.bias is not None:
                nn.init.zeros_(self.adapter.bias)

    def forward(self, f_i: torch.Tensor, f_pert: torch.Tensor) -> torch.Tensor:
        """
        f_i: Unperturbed image features [B, C, H, W]
        f_pert: Perturbed image features [B, C, H, W]
        Returns adapted features F_tilde [B, C, H, W]
        """
        cat = torch.cat([f_i, f_pert], dim=1)
        delta = self.adapter(cat)
        return f_i + delta


class MaskAdapter(nn.Module):
    """
    Automatic Prompting Mask Adapter for Detective SAM.
    Maps all features {F_s^I, F_s^{I'}, F_tilde_s} into a dense heatmap prompt M.

    Architecture (Section 3.2, Page 4-5):
    1. Cross-scale, cross-stream convolutional fusion to F_fuse in R^{d x s_hat x s_hat} (d=64).
    2. Coarse Transformer branch operating on downscaled representation (16x downscale)
       producing coarse logits L_coarse and uncertainty map U in R^{s_hat x s_hat}.
    3. Fine Refinement branch: 2-layer conv network producing L_refine in R^{s_hat x s_hat}.
    4. Spatial Gating: Gate g = sigmoid(Conv1x1([L_coarse, U])), blending:
       M = g * L_refine + (1 - g) * L_coarse.
    """

    def __init__(
        self,
        in_channels_list: List[int] = [32, 64, 256],
        d: int = 64,
        downscale_factor: int = 16,
        transformer_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.d = d
        self.downscale_factor = downscale_factor
        self.in_channels_list = list(in_channels_list)

        # 3 streams (original, perturbed, adapted) across all scales
        total_in_channels = 3 * sum(self.in_channels_list)

        # Cross-scale, cross-stream shallow convolutional fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(total_in_channels, d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Conv2d(d, d, kernel_size=3, padding=1),
            nn.BatchNorm2d(d),
            nn.GELU(),
        )

        # Lightweight Transformer at coarse resolution
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.coarse_head = nn.Linear(d, 2)  # Predicts [L_coarse, U]

        # 2-layer convolutional network for fine refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Conv2d(d, 1, kernel_size=3, padding=1),
        )

        # Spatial gating conv 1x1 on [L_coarse, U] -> 1
        self.gate_conv = nn.Conv2d(2, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        f_i: List[torch.Tensor],
        f_pert: List[torch.Tensor],
        f_tilde: List[torch.Tensor],
        forensic_memory: Optional[ForensicVectorMemory] = None,
        random_retrieval: bool = False,
        return_features: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
    ]:
        """
        Inputs:
            f_i: List of unperturbed multi-scale feature tensors
            f_pert: List of perturbed multi-scale feature tensors
            f_tilde: List of adapted multi-scale feature tensors
            forensic_memory: Optional ForensicVectorMemory for similarity-based query replacement
            random_retrieval: If True, uses random retrieval for ablation (Baseline C)
            return_features: If True, also returns f_fuse and retrieval_latency
        Returns:
            Default: (m, l_coarse, l_refine, u)
            If return_features=True: (m, l_coarse, l_refine, u, f_fuse, retrieval_latency)
        """
        s_hat = f_i[0].shape[-2:]

        # 1. Bilinearly upsample all features to finest grid s_hat
        all_upsampled = []
        for stream in (f_i, f_pert, f_tilde):
            for feat in stream:
                if feat.shape[-2:] != s_hat:
                    feat = F.interpolate(feat, size=s_hat, mode="bilinear", align_corners=False)
                all_upsampled.append(feat)

        # Cross-scale, cross-stream fusion
        cat_features = torch.cat(all_upsampled, dim=1)
        f_fuse = self.fusion(cat_features)  # [B, d, s_hat, s_hat]

        # 2. Coarse Transformer Branch
        b, d, h, w = f_fuse.shape
        coarse_h = max(2, h // self.downscale_factor)
        coarse_w = max(2, w // self.downscale_factor)
        f_down = F.adaptive_avg_pool2d(f_fuse, (coarse_h, coarse_w))
        tokens = f_down.flatten(2).transpose(1, 2)  # [B, coarse_h * coarse_w, d]

        # Stop-gradient surrogate query replacement
        R = tokens  # [B, N_coarse, d]
        retrieval_latency = 0.0
        if (
            forensic_memory is not None
            and getattr(forensic_memory, "memory_enabled", False)
            and getattr(forensic_memory, "is_warmed_up", False)
        ):
            # Global image-level query representation across coarse tokens
            q = torch.mean(R, dim=1)  # [B, d]
            retrieved_vals, _, retrieval_latency = forensic_memory.retrieve(
                q,
                k=forensic_memory.memory_retrieval_k,
                random_retrieval=random_retrieval,
            )
            if retrieved_vals is not None:
                # Shape of retrieved_vals: [B, k, d]
                # Aggregate top-k retrieved vectors (mean pooling across k neighbors)
                v_agg = torch.mean(retrieved_vals, dim=1, keepdim=True)  # [B, 1, d]
                Rt = v_agg.expand_as(R).contiguous()  # [B, N_coarse, d]
                # Stop-gradient query replacement: Rx = R + (Rt - R).detach()
                R_x = stop_gradient_query_replacement(R, Rt)
            else:
                R_x = R
        else:
            R_x = R

        trans_out = self.transformer(R_x)
        coarse_pred = self.coarse_head(trans_out)  # [B, tokens, 2]
        coarse_pred = coarse_pred.transpose(1, 2).view(b, 2, coarse_h, coarse_w)

        # Upsample coarse predictions back to s_hat
        coarse_pred_up = F.interpolate(coarse_pred, size=s_hat, mode="bilinear", align_corners=False)
        l_coarse = coarse_pred_up[:, 0:1, :, :]
        u = coarse_pred_up[:, 1:2, :, :]

        # 3. Fine Refinement Branch
        l_refine = self.refine_conv(f_fuse)  # [B, 1, s_hat, s_hat]

        # 4. Linear Spatial Gating
        gate_input = torch.cat([l_coarse, u], dim=1)
        g = torch.sigmoid(self.gate_conv(gate_input))
        m = g * l_refine + (1.0 - g) * l_coarse

        if return_features:
            return m, l_coarse, l_refine, u, f_fuse, retrieval_latency
        return m, l_coarse, l_refine, u


class DetectiveSAM(nn.Module):
    """
    Detective SAM: Adaptive AI-Image Forgery Localization model.

    Integrates Meta's SAM2 foundation model with:
    - ImagePerturbation: perturbation-driven forensic clue generator.
    - FeatureAdapters: 1x1 conv residual delta correction per scale.
    - MaskAdapter: cross-scale cross-stream fusion + downscaled coarse transformer + spatial gating.
    - Frozen SAM2 Mask Decoder: decodes forensic heatmap prompt into precise forgery mask logits.

    Args:
        pretrained_model_name_or_path (str): SAM2 checkpoint path or HF repo ID (default: 'facebook/sam2-hiera-small').
        variant (str): SAM2 variant ('small', 'large', 'tiny').
        layer_width (int): Hidden dimension for fusion and mask adapter (default: 64).
        noise_intensity (float): Perturbation strength parameter (default: 0.75).
        perturbation_type (str): Perturbation method ('blur_noise', 'blur', 'noise', 'none').
        transformer_downscaling (int): Coarse transformer downsampling factor (default: 16).
        transformer_layers (int): Coarse transformer layers (default: 1).
        dropout (float): Dropout probability for mask adapter (default: 0.15).
        target_size (Tuple[int, int]): Native input resolution for SAM2 Hiera ViT (default: (512, 512)).
        freeze_sam2 (bool): Whether to freeze all base SAM2 parameters (default: True).
        freeze_decoder (bool): Whether to keep the mask decoder frozen (default: True per paper).
        multimask_output (bool): Whether mask decoder outputs candidate multimasks (default: False).
        aux_classifier (bool): Whether to expose auxiliary classification head (default: False).
        num_classes (int): Number of target classes for auxiliary head (default: 3).
        use_dummy (bool): If True, skips loading heavy pretrained weights for fast testing.
        device (str or torch.device): Target computing device.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        variant: Optional[str] = "small",
        layer_width: int = 64,
        noise_intensity: float = 0.75,
        perturbation_type: str = "blur_noise",
        transformer_downscaling: int = 16,
        transformer_layers: int = 1,
        dropout: float = 0.15,
        target_size: Tuple[int, int] = (512, 512),
        freeze_sam2: bool = True,
        freeze_decoder: bool = True,
        multimask_output: bool = False,
        aux_classifier: bool = False,
        num_classes: int = 3,
        in_channels: int = 3,
        out_channels: int = 1,
        use_dummy: bool = False,
        device: Optional[Union[str, torch.device]] = "auto",
        memory_enabled: bool = False,
        memory_max_entries: int = 5000,
        memory_key_dim: int = 64,
        memory_value_dim: int = 64,
        memory_vectors_per_image: int = 16,
        memory_retrieval_k: int = 8,
        memory_similarity_metric: str = "cosine",
        memory_dedup_threshold: float = 0.95,
        memory_consolidation_method: str = "dedup_greedy",
        memory_seed: int = 42,
        memory_path: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.variant = str(variant or "small").lower()
        self.pretrained_model_name_or_path = resolve_sam2_checkpoint(pretrained_model_name_or_path, variant=self.variant)
        self.layer_width = layer_width
        self.noise_intensity = noise_intensity
        self.perturbation_type = perturbation_type
        self.transformer_downscaling = transformer_downscaling
        self.transformer_layers = transformer_layers
        self.dropout = dropout
        self.target_size = tuple(target_size)
        self.freeze_sam2 = freeze_sam2
        self.freeze_decoder = freeze_decoder
        self.multimask_output = multimask_output
        self.aux_classifier = aux_classifier
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_dummy = use_dummy

        # Memory configuration
        self.memory_enabled = bool(memory_enabled)
        self.memory_max_entries = int(memory_max_entries)
        self.memory_key_dim = int(memory_key_dim)
        self.memory_value_dim = int(memory_value_dim)
        self.memory_vectors_per_image = int(memory_vectors_per_image)
        self.memory_retrieval_k = int(memory_retrieval_k)
        self.memory_similarity_metric = str(memory_similarity_metric).lower()
        self.memory_dedup_threshold = float(memory_dedup_threshold)
        self.memory_consolidation_method = str(memory_consolidation_method).lower()
        self.memory_seed = int(memory_seed)
        self.memory_path = memory_path

        # Resolve target device
        if device == "auto" or device is None:
            self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._target_device = torch.device(device)
        else:
            self._target_device = device

        # 1. Image Perturbation Operator
        self.perturbation = ImagePerturbation(
            perturbation_type=self.perturbation_type,
            noise_intensity=self.noise_intensity,
        )

        # 2. SAM2 Foundation Model
        self.sam2_model = None
        self.scale_channels = [32, 64, 256]  # Default Hiera small/tiny scales
        if not self.use_dummy:
            self._init_sam2_model()
        else:
            logger.info("Initializing DetectiveSAM in Dummy mode for fast unit tests.")

        # 3. Multi-Scale Feature Adapters (one per scale)
        self.feature_adapters = nn.ModuleList([
            FeatureAdapter(in_channels=c, width=self.layer_width)
            for c in self.scale_channels
        ])

        # 4. Mask Adapter (automatic prompting)
        self.mask_adapter = MaskAdapter(
            in_channels_list=self.scale_channels,
            d=self.layer_width,
            downscale_factor=self.transformer_downscaling,
            transformer_layers=self.transformer_layers,
            num_heads=4,
            dropout=self.dropout,
        )

        # 5. Optional Auxiliary Classifier Head
        if self.aux_classifier:
            self.classifier_head = nn.Linear(self.scale_channels[-1], self.num_classes)
        else:
            self.classifier_head = None

        # 6. Forensic Vector Memory (CPU-resident)
        self.forensic_memory = ForensicVectorMemory(
            memory_enabled=self.memory_enabled,
            memory_max_entries=self.memory_max_entries,
            memory_key_dim=self.memory_key_dim,
            memory_value_dim=self.memory_value_dim,
            memory_vectors_per_image=self.memory_vectors_per_image,
            memory_retrieval_k=self.memory_retrieval_k,
            memory_similarity_metric=self.memory_similarity_metric,
            memory_dedup_threshold=self.memory_dedup_threshold,
            memory_consolidation_method=self.memory_consolidation_method,
            memory_seed=self.memory_seed,
        )

        if self.memory_path:
            if os.path.exists(self.memory_path):
                self.forensic_memory.load(self.memory_path)
            elif self.memory_enabled:
                raise FileNotFoundError(
                    f"Forensic memory file not found at '{self.memory_path}' while memory_enabled is True."
                )

        logger.info(
            f"Initialized DetectiveSAM (variant='{self.variant}', width={self.layer_width}, "
            f"noise_intensity={self.noise_intensity}, freeze_sam2={self.freeze_sam2}, "
            f"freeze_decoder={self.freeze_decoder}, memory_enabled={self.memory_enabled})"
        )

    def _init_sam2_model(self):
        """Load and configure frozen SAM2 base model."""
        try:
            from transformers import Sam2Model, Sam2Config
        except ImportError as exc:
            logger.warning(f"transformers.Sam2Model not available ({exc}). Falling back to dummy mode.")
            self.use_dummy = True
            return

        try:
            logger.info(f"Loading SAM2 model from '{self.pretrained_model_name_or_path}'...")
            self.sam2_model = Sam2Model.from_pretrained(self.pretrained_model_name_or_path)
        except Exception as exc:
            logger.warning(f"Could not load SAM2 weights from '{self.pretrained_model_name_or_path}': {exc}. Using default config.")
            cfg = Sam2Config()
            self.sam2_model = Sam2Model(cfg)

        # Extract channel sizes from SAM2 config
        if hasattr(self.sam2_model.config, "vision_config"):
            vc = self.sam2_model.config.vision_config
            # Feature maps from Sam2VisionModel: level 0 (32), level 1 (64), level 2 (fpn_hidden_size=256)
            c0 = getattr(vc, "backbone_channel_list", [768, 384, 192, 96])[-1] // 3 if hasattr(vc, "backbone_channel_list") else 32
            # FPN hidden size is typically 256
            self.scale_channels = [32, 64, getattr(vc, "fpn_hidden_size", 256)]

        # Freeze SAM2 backbone
        if self.freeze_sam2:
            for p in self.sam2_model.parameters():
                p.requires_grad = False
            logger.info("Frozen base SAM2 vision encoder and prompt encoder.")

        # Handle mask decoder freezing
        if hasattr(self.sam2_model, "mask_decoder"):
            for p in self.sam2_model.mask_decoder.parameters():
                p.requires_grad = not self.freeze_decoder
            logger.info(f"SAM2 mask_decoder trainable: {not self.freeze_decoder}")

    def _extract_sam2_features(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale image embeddings from SAM2 vision encoder.
        Returns list of 3 feature tensors:
            [F_0 (1/4 scale, 32 ch), F_1 (1/8 scale, 64 ch), F_2 (1/16 scale, 256 ch)]
        """
        if self.use_dummy or self.sam2_model is None:
            # Dummy multi-scale features for testing
            b = pixel_values.shape[0]
            dev = pixel_values.device
            c0, c1, c2 = self.scale_channels
            h_t, w_t = self.target_size
            return [
                torch.zeros(b, c0, h_t // 4, w_t // 4, device=dev, dtype=pixel_values.dtype),
                torch.zeros(b, c1, h_t // 8, w_t // 8, device=dev, dtype=pixel_values.dtype),
                torch.zeros(b, c2, h_t // 16, w_t // 16, device=dev, dtype=pixel_values.dtype),
            ]

        # Use Sam2Model vision feature extraction
        b_sz = pixel_values.shape[0]
        image_outputs = self.sam2_model.get_image_features(pixel_values, return_dict=True)
        feature_maps = list(image_outputs.fpn_hidden_states)

        # Add no-memory embedding to the coarsest feature map
        if hasattr(self.sam2_model, "no_memory_embedding"):
            feature_maps[-1] = feature_maps[-1] + self.sam2_model.no_memory_embedding

        # Reshape to [B, C, H, W] dynamically matching spatial layout
        image_embeddings = []
        for feat in feature_maps:
            h_feat = int(math.isqrt(feat.shape[0]))
            w_feat = feat.shape[0] // h_feat
            image_embeddings.append(
                feat.permute(1, 2, 0).contiguous().reshape(b_sz, -1, h_feat, w_feat)
            )
        return image_embeddings

    def forward(
        self,
        images: torch.Tensor,
        return_dict: bool = False,
        gt_masks: Optional[torch.Tensor] = None,
        random_retrieval: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass of Detective SAM.

        Args:
            images (torch.Tensor): Input images of shape [B, 3, H, W].
            return_dict (bool): Whether to return complete dictionary of intermediate representations.
            gt_masks (Optional[torch.Tensor]): Ground-truth binary masks [B, 1, H, W] used strictly
                during training for forensic vector memory collection. Never used during evaluation/inference.
            random_retrieval (bool): Whether to use randomized retrieval for Baseline C ablation.

        Returns:
            - Default: Predicted mask logits of shape [B, 1, H, W]
              (or Tuple[mask_logits, cls_logits] if aux_classifier is True).
            - If return_dict is True: Dict containing 'mask_logits', 'heatmap_prompt',
              'l_coarse', 'l_refine', 'uncertainty', 'retrieval_latency', etc.
        """
        b_sz, _, orig_h, orig_w = images.shape
        model_device, param_dtype = get_submodule_device_dtype(
            self.mask_adapter, images.device, torch.float32
        )

        # 1. Resize input images to SAM2 target resolution (e.g. 1024x1024 or 512x512)
        if (orig_h, orig_w) != self.target_size:
            x_in = F.interpolate(images, size=self.target_size, mode="bilinear", align_corners=False)
        else:
            x_in = images

        # 2. Perturbation operator: generate perturbed image I'
        x_pert = self.perturbation(x_in)

        # 3. Multi-scale feature extraction for unperturbed and perturbed images
        # Batch together through frozen vision encoder for efficiency
        x_both = torch.cat([x_in, x_pert], dim=0)
        feats_both = self._extract_sam2_features(x_both)
        feats_i = [f[:b_sz] for f in feats_both]
        feats_pert = [f[b_sz:] for f in feats_both]

        # 4. Multi-Scale Feature Adapters: delta correction
        feats_adapted = []
        for s_idx, adapter in enumerate(self.feature_adapters):
            f_tilde = adapter(feats_i[s_idx], feats_pert[s_idx])
            feats_adapted.append(f_tilde)

        # 5. Mask Adapter: automatic prompting heatmap M with stop-gradient memory retrieval
        mask_adapter_out = self.mask_adapter(
            feats_i,
            feats_pert,
            feats_adapted,
            forensic_memory=self.forensic_memory if self.memory_enabled else None,
            random_retrieval=random_retrieval,
            return_features=True,
        )
        m_prompt, l_coarse, l_refine, u_map, f_fuse, retrieval_lat = mask_adapter_out

        # Collect ground-truth tampered features into memory strictly during training
        if self.training and self.memory_enabled and gt_masks is not None:
            self._collect_tampered_memory(f_fuse, gt_masks)

        # 6. Bilinearly upsample heatmap prompt and adapted features to 256x256 before decoding
        # (As specified in Section 3.2, Page 5: bilinearly upsample M and {~Fs} to 256x256)
        decode_res = (256, 256)
        m_256 = F.interpolate(m_prompt, size=decode_res, mode="bilinear", align_corners=False)

        # 7. SAM2 Mask Decoder execution
        if not self.use_dummy and self.sam2_model is not None:
            # Prepare dummy point prompt so sparse_embeddings are properly shaped
            input_points = torch.zeros(b_sz, 1, 1, 2, dtype=feats_adapted[-1].dtype, device=model_device)
            input_labels = -torch.ones(b_sz, 1, 1, dtype=torch.int32, device=model_device)

            sparse_emb, dense_emb = self.sam2_model.prompt_encoder(
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=None,
                input_masks=m_256,
            )

            # Get image wide positional embeddings and ensure spatial dimensions match coarsest feature map
            pos_emb = self.sam2_model.get_image_wide_positional_embeddings().repeat(b_sz, 1, 1, 1)
            target_feat_spatial = feats_adapted[-1].shape[-2:]
            if pos_emb.shape[-2:] != target_feat_spatial:
                pos_emb = F.interpolate(pos_emb, size=target_feat_spatial, mode="bilinear", align_corners=False)
            if dense_emb.shape[-2:] != target_feat_spatial:
                dense_emb = F.interpolate(dense_emb, size=target_feat_spatial, mode="bilinear", align_corners=False)

            # Mask decoder forward
            low_res_multimasks, iou_scores, _, _ = self.sam2_model.mask_decoder(
                image_embeddings=feats_adapted[-1],
                image_positional_embeddings=pos_emb,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=self.multimask_output,
                high_resolution_features=feats_adapted[:-1],
            )

            if self.multimask_output:
                # Softmax-weighted aggregation across candidate masks
                weights = torch.softmax(iou_scores, dim=-1).unsqueeze(-1)  # [B, 1, num_masks, 1, 1]
                mask_logits_raw = (low_res_multimasks * weights).sum(dim=2)  # [B, 1, 256, 256]
            else:
                mask_logits_raw = low_res_multimasks.squeeze(2) if low_res_multimasks.ndim == 5 else low_res_multimasks
                if mask_logits_raw.ndim == 3:
                    mask_logits_raw = mask_logits_raw.unsqueeze(1)
        else:
            # In dummy mode: route heatmap prompt directly
            mask_logits_raw = m_256

        # 8. Upsample predicted mask logits back to input image dimensions [B, 1, orig_h, orig_w]
        mask_logits = F.interpolate(
            mask_logits_raw.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        cls_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            # Global pooled representation from coarsest feature map
            cls_dev, cls_dtype = get_submodule_device_dtype(self.classifier_head, images.device, torch.float32)
            pooled = F.adaptive_avg_pool2d(feats_adapted[-1], (1, 1)).view(b_sz, -1).to(device=cls_dev, dtype=cls_dtype)
            cls_logits = self.classifier_head(pooled)

        if return_dict:
            res: Dict[str, Any] = {
                "mask_logits": mask_logits,
                "heatmap_prompt": m_prompt,
                "l_coarse": l_coarse,
                "l_refine": l_refine,
                "uncertainty": u_map,
                "perturbed_images": x_pert,
                "retrieval_latency": retrieval_lat,
                "memory_entries": self.forensic_memory.entry_count if self.forensic_memory is not None else 0,
                "is_warmed_up": self.forensic_memory.is_warmed_up if self.forensic_memory is not None else False,
            }
            if cls_logits is not None:
                res["cls_logits"] = cls_logits
            return res

        if self.aux_classifier and cls_logits is not None:
            return mask_logits, cls_logits

        return mask_logits

    def _collect_tampered_memory(self, f_fuse: torch.Tensor, gt_masks: torch.Tensor) -> int:
        """
        Extract features from ground-truth tampered regions and add to epoch collection buffer.
        Strictly excludes authentic/untampered images from positive forensic memory collection.

        Args:
            f_fuse (torch.Tensor): Fused feature map [B, d, s_hat, s_hat].
            gt_masks (torch.Tensor): Ground-truth masks [B, 1, H, W] in {0.0, 1.0}.

        Returns:
            int: Number of valid tampered vectors extracted and detached to CPU.
        """
        if self.forensic_memory is None or not self.memory_enabled:
            return 0
        if gt_masks is None or gt_masks.numel() == 0:
            return 0

        # Downscale ground-truth masks to feature map resolution s_hat using area interpolation
        s_hat = f_fuse.shape[-2:]
        if gt_masks.shape[-2:] != s_hat:
            gt_down = F.interpolate(gt_masks.float(), size=s_hat, mode="area") >= 0.5
        else:
            gt_down = (gt_masks >= 0.5)

        total_collected = 0
        b_sz = f_fuse.shape[0]

        for b in range(b_sz):
            tampered_mask_b = gt_down[b, 0]  # [s_hat_h, s_hat_w]
            num_tampered = int(tampered_mask_b.sum().item())
            if num_tampered == 0:
                # Authentic image: strictly exclude, do not collect any vectors
                continue

            # Feature map for image b: [s_hat_h, s_hat_w, d]
            feat_b = f_fuse[b].permute(1, 2, 0)
            tampered_vectors = feat_b[tampered_mask_b]  # [num_tampered, d]

            if num_tampered > self.memory_vectors_per_image:
                # Uniformly subsample across tampered region
                indices = torch.linspace(
                    0, num_tampered - 1, steps=self.memory_vectors_per_image, device=tampered_vectors.device
                ).long()
                selected_vectors = tampered_vectors[indices]
            else:
                selected_vectors = tampered_vectors

            # Add detached observations to CPU memory
            added = self.forensic_memory.add_observations(selected_vectors)
            total_collected += added

        return total_collected

    def consolidate_memory(self) -> Dict[str, Any]:
        """Consolidate accumulated epoch observations into authoritative database."""
        if self.forensic_memory is None:
            return {}
        return self.forensic_memory.consolidate()

    def save_memory(self, filepath: str) -> None:
        """Save authoritative memory database to disk."""
        if self.forensic_memory is not None:
            self.forensic_memory.save(filepath)

    def load_memory(self, filepath: str) -> None:
        """Load authoritative memory database from disk."""
        if self.forensic_memory is not None:
            self.forensic_memory.load(filepath)

    def get_memory_summary(self) -> Dict[str, Any]:
        """Return memory statistics and resource summary."""
        if self.forensic_memory is None:
            return {"memory_enabled": False}
        return {
            "memory_enabled": self.memory_enabled,
            "entry_count": self.forensic_memory.entry_count,
            "max_entries": self.forensic_memory.memory_max_entries,
            "is_warmed_up": self.forensic_memory.is_warmed_up,
            "key_dim": self.forensic_memory.memory_key_dim,
            "value_dim": self.forensic_memory.memory_value_dim,
            "memory_bytes": self.forensic_memory.get_memory_bytes(),
            "memory_mb": self.forensic_memory.get_memory_bytes() / (1024 * 1024),
            "similarity_metric": self.forensic_memory.memory_similarity_metric,
            "consolidation_method": self.forensic_memory.memory_consolidation_method,
        }

    @torch.no_grad()
    def predict_mask(
        self,
        images: torch.Tensor,
        threshold: float = 0.5,
        random_retrieval: bool = False,
    ) -> torch.Tensor:
        """
        Produce binary mask prediction B in {0, 1}^{H x W} given input images.
        Thresholds sigmoid(M_hat) >= 0.5 as defined in Section 3.2.
        """
        self.eval()
        outputs = self.forward(images, random_retrieval=random_retrieval)
        mask_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        probs = torch.sigmoid(mask_logits)
        return (probs >= threshold).float()


class DetectiveSAMLoss(nn.Module):
    """
    Combined objective for Detective SAM training:
        L = L_Dice + lambda_focal * L_focal + lambda_IoU * L_IoU
    Parameters from paper (Section 3.2, Table 16):
        lambda_focal = 20.0, lambda_IoU = 1.0, alpha = 0.6, gamma = 1.0.
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        alpha: float = 0.6,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        targets: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mask_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        targets = (targets >= 0.5).float()

        # 1. Binary Focal Loss
        probs = torch.sigmoid(mask_logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        focal_loss = -alpha_t * ((1.0 - pt) ** self.gamma) * torch.log(torch.clamp(pt, min=1e-7))
        l_focal = focal_loss.mean()

        # 2. Dice Loss
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.view(targets.size(0), -1)
        intersection = (p_flat * t_flat).sum(dim=1)
        denominator = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        l_dice = (1.0 - dice).mean()

        # 3. IoU Loss
        union = denominator - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        l_iou = (1.0 - iou).mean()

        total_loss = self.dice_weight * l_dice + self.focal_weight * l_focal + self.iou_weight * l_iou

        loss_dict = {
            "loss": float(total_loss.item()),
            "loss_focal": float(l_focal.item()),
            "loss_dice": float(l_dice.item()),
            "loss_iou": float(l_iou.item()),
            "iou": float(iou.mean().item()),
        }
        return total_loss, loss_dict
