"""
Finetuned Diffusion VAE model (SD1.5 AutoencoderKL) for AI-generated synthetic image segmentation.
Adapts pretrained latent diffusion VAE to decode directly into binary mask space.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.blocks import AuxiliaryClassifier

logger = logging.getLogger("sid_unet.models.vae_finetune")

DEFAULT_SD_VAE_CHECKPOINT = "runwayml/stable-diffusion-v1-5"


def _safe_checkpoint(func, *args):
    """Run function with torch activation checkpointing safely across PyTorch versions."""
    try:
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)
    except TypeError:
        return torch.utils.checkpoint.checkpoint(func, *args)


class SkipFusion(nn.Module):
    """
    UNet-style skip connection fusion block.
    Concatenates decoder features with encoder skip connection features,
    followed by convolution, normalization, and activation to project back
    to the target decoder feature channel dimension.
    """

    def __init__(
        self,
        dec_channels: int,
        skip_channels: int,
        out_channels: Optional[int] = None,
        norm_layer: str = "groupnorm",
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = dec_channels

        self.conv = nn.Conv2d(dec_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if norm_layer == "batchnorm":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_layer == "groupnorm":
            num_groups = min(32, out_channels)
            while out_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        else:
            self.norm = nn.Identity()

        if activation == "silu":
            self.act = nn.SiLU(inplace=True)
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, dec_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if dec_feat.shape[2:] != skip_feat.shape[2:]:
            skip_feat = F.interpolate(skip_feat, size=dec_feat.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([dec_feat, skip_feat], dim=1)
        return self.dropout(self.act(self.norm(self.conv(fused))))


class DiffusionVAEFinetune(nn.Module):
    """
    Finetuned VAE taken from a diffusion model (such as Stable Diffusion 1.5 AutoencoderKL)
    for AI-generated image mask segmentation with UNet-style skip connections.

    The encoder compresses the input image into latent distribution z while preserving
    multi-scale intermediate feature maps that are routed via skip connections to the decoder,
    and the decoder upsamples and projects back into the binary mask logit space.
    The entire VAE or only the decoder can be fine-tuned.

    Args:
        pretrained_model_name_or_path: Hugging Face model hub path or local path (default: 'runwayml/stable-diffusion-v1-5').
        subfolder: Subfolder in pretrained repository containing the VAE (default: 'vae').
        in_channels: Input image channels (default: 3 for RGB).
        out_channels: Output mask channels (default: 1 for binary mask logits).
        freeze_encoder: If True, freezes the VAE encoder and only finetunes the decoder.
        sample_mode: Latent sampling mode: 'sample' (sample with noise) or 'mode' (deterministic mean).
        scaling_factor: VAE latent scaling factor (SD1.5 standard: 0.18215).
        aux_classifier: Whether to enable 3-class auxiliary classification head.
        num_classes: Number of target classes for auxiliary head (default: 3).
        dropout: Dropout probability in auxiliary classification head.
        gradient_checkpointing: Whether to enable activation gradient checkpointing to save VRAM.
        use_dummy: If True, creates a lightweight dummy AutoencoderKL for fast offline testing.
        dummy_channels: Tuple of channel dimensions for dummy VAE blocks (default: (32, 64)).
        input_rescale: If True, automatically rescales inputs to [-1, 1] as expected by SD VAE.
        use_skip_connections: If True, connects multi-scale encoder layers to decoder layers (UNet-style skips).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = DEFAULT_SD_VAE_CHECKPOINT,
        subfolder: Optional[str] = "vae",
        in_channels: int = 3,
        out_channels: int = 1,
        freeze_encoder: bool = False,
        sample_mode: str = "sample",
        scaling_factor: float = 0.18215,
        aux_classifier: bool = True,
        num_classes: int = 3,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        use_dummy: bool = False,
        dummy_channels: Tuple[int, ...] = (32, 64),
        input_rescale: bool = True,
        use_skip_connections: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.subfolder = subfolder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze_encoder = freeze_encoder
        self.sample_mode = sample_mode
        self.scaling_factor = scaling_factor
        self.aux_classifier = aux_classifier
        self.num_classes = num_classes
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_dummy = use_dummy
        self.dummy_channels = dummy_channels
        self.input_rescale = input_rescale
        self.use_skip_connections = bool(kwargs.get("skip_connections", use_skip_connections))

        self._init_vae(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            use_dummy=use_dummy,
            dummy_channels=dummy_channels,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        # Latent channels from VAE
        latent_ch = getattr(self.vae.config, "latent_channels", 4)
        self.latent_channels = latent_ch

        # UNet-style skip connections from encoder to decoder
        if self.use_skip_connections:
            self._init_skip_fusions()
        else:
            self.skip_fusions = nn.ModuleList()
            self.skip_mapping = []

        # Auxiliary classifier head on pooled latent features
        if self.aux_classifier:
            self.classifier_head = AuxiliaryClassifier(
                in_channels=latent_ch,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            self.classifier_head = None

        if self.gradient_checkpointing:
            self.set_gradient_checkpointing(True)

    def _init_vae(
        self,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str],
        use_dummy: bool,
        dummy_channels: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
    ) -> None:
        from diffusers import AutoencoderKL

        if use_dummy:
            logger.info("Initializing lightweight dummy AutoencoderKL for offline testing...")
            blocks = list(dummy_channels)
            self.vae = AutoencoderKL(
                in_channels=in_channels,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D"] * len(blocks),
                up_block_types=["UpDecoderBlock2D"] * len(blocks),
                block_out_channels=blocks,
                latent_channels=4,
                layers_per_block=1,
            )
        else:
            try:
                logger.info(
                    f"Loading pretrained AutoencoderKL from '{pretrained_model_name_or_path}', subfolder='{subfolder}'..."
                )
                load_kwargs: Dict[str, Any] = {}
                if subfolder:
                    load_kwargs["subfolder"] = subfolder
                self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
            except Exception as e:
                logger.warning(
                    f"Could not load pretrained AutoencoderKL from '{pretrained_model_name_or_path}': {e}. "
                    f"Falling back to initialized AutoencoderKL architecture..."
                )
                blocks = list(dummy_channels)
                self.vae = AutoencoderKL(
                    in_channels=in_channels,
                    out_channels=3,
                    down_block_types=["DownEncoderBlock2D"] * len(blocks),
                    up_block_types=["UpDecoderBlock2D"] * len(blocks),
                    block_out_channels=blocks,
                    latent_channels=4,
                    layers_per_block=1,
                )

        # Adapt final decoder conv_out to output binary mask logits
        orig_conv_out = self.vae.decoder.conv_out
        conv_in_ch = orig_conv_out.in_channels
        self.vae.decoder.conv_out = nn.Conv2d(
            in_channels=conv_in_ch,
            out_channels=out_channels,
            kernel_size=orig_conv_out.kernel_size,
            stride=orig_conv_out.stride,
            padding=orig_conv_out.padding,
        )
        nn.init.kaiming_normal_(self.vae.decoder.conv_out.weight, mode="fan_out", nonlinearity="relu")
        if self.vae.decoder.conv_out.bias is not None:
            nn.init.zeros_(self.vae.decoder.conv_out.bias)

    def _init_skip_fusions(self) -> None:
        """Dynamically detect and initialize UNet-style skip connections from VAE encoder to decoder."""
        self.skip_fusions = nn.ModuleList()
        self.skip_mapping = []

        with torch.no_grad():
            dummy_x = torch.zeros((1, self.in_channels, 64, 64))
            h = self.vae.encoder.conv_in(dummy_x)
            enc_skips = [h]
            for b in self.vae.encoder.down_blocks:
                h = b(h)
                enc_skips.append(h)
            h = self.vae.encoder.mid_block(h)
            h = self.vae.encoder.conv_norm_out(h)
            h = self.vae.encoder.conv_act(h)
            h = self.vae.encoder.conv_out(h)
            if getattr(self.vae, "quant_conv", None) is not None:
                h = self.vae.quant_conv(h)

            z = h[:, :self.latent_channels]
            if getattr(self.vae, "post_quant_conv", None) is not None:
                z = self.vae.post_quant_conv(z)
            s = self.vae.decoder.conv_in(z)
            s = self.vae.decoder.mid_block(s)

            for i, up_b in enumerate(self.vae.decoder.up_blocks):
                s = up_b(s)
                # Find matching encoder skip by spatial resolution (closest matching down block)
                matching_idx = None
                for s_idx, skip in enumerate(enc_skips):
                    if skip.shape[2] == s.shape[2]:
                        matching_idx = s_idx
                if matching_idx is not None:
                    self.skip_mapping.append(matching_idx)
                    self.skip_fusions.append(
                        SkipFusion(
                            dec_channels=s.shape[1],
                            skip_channels=enc_skips[matching_idx].shape[1],
                            dropout=self.dropout,
                        )
                    )
                else:
                    self.skip_mapping.append(None)
                    self.skip_fusions.append(nn.Identity())
        logger.info(
            f"Initialized {len([m for m in self.skip_mapping if m is not None])} UNet-style skip connection(s) "
            f"between VAE encoder and decoder."
        )

    def _encode_with_skips(self, x: torch.Tensor) -> Tuple[Any, List[torch.Tensor]]:
        """Encode image into latent distribution while capturing intermediate encoder features for skip connections."""
        sample = self.vae.encoder.conv_in(x)
        skips = [sample]

        use_ckpt = self.gradient_checkpointing and self.training and sample.requires_grad

        for down_block in self.vae.encoder.down_blocks:
            if use_ckpt:
                sample = _safe_checkpoint(down_block, sample)
            else:
                sample = down_block(sample)
            skips.append(sample)

        if use_ckpt:
            sample = _safe_checkpoint(self.vae.encoder.mid_block, sample)
        else:
            sample = self.vae.encoder.mid_block(sample)

        sample = self.vae.encoder.conv_norm_out(sample)
        sample = self.vae.encoder.conv_act(sample)
        sample = self.vae.encoder.conv_out(sample)

        if getattr(self.vae, "quant_conv", None) is not None:
            sample = self.vae.quant_conv(sample)

        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        posterior = DiagonalGaussianDistribution(sample)
        return posterior, skips

    def _decode_with_skips(self, z: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Decode latent distribution back to binary mask space while fusing UNet encoder skip connections."""
        if getattr(self.vae, "post_quant_conv", None) is not None:
            z = self.vae.post_quant_conv(z)

        use_ckpt = self.gradient_checkpointing and self.training and z.requires_grad

        sample = self.vae.decoder.conv_in(z)
        if use_ckpt:
            sample = _safe_checkpoint(self.vae.decoder.mid_block, sample)
        else:
            sample = self.vae.decoder.mid_block(sample)

        for i, up_block in enumerate(self.vae.decoder.up_blocks):
            if use_ckpt:
                sample = _safe_checkpoint(up_block, sample)
            else:
                sample = up_block(sample)

            if i < len(self.skip_mapping) and i < len(self.skip_fusions):
                matching_skip_idx = self.skip_mapping[i]
                if matching_skip_idx is not None and matching_skip_idx < len(skips):
                    if use_ckpt:
                        sample = _safe_checkpoint(self.skip_fusions[i], sample, skips[matching_skip_idx])
                    else:
                        sample = self.skip_fusions[i](sample, skips[matching_skip_idx])

        sample = self.vae.decoder.conv_norm_out(sample)
        sample = self.vae.decoder.conv_act(sample)
        sample = self.vae.decoder.conv_out(sample)
        return sample

    def _freeze_encoder(self) -> None:
        """Freeze VAE encoder and quant_conv parameters."""
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
        if hasattr(self.vae, "quant_conv") and self.vae.quant_conv is not None:
            for param in self.vae.quant_conv.parameters():
                param.requires_grad = False
        logger.info("VAE encoder frozen for fine-tuning decoder only.")

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        """Dynamically enable or disable activation checkpointing."""
        self.gradient_checkpointing = bool(enable)
        if hasattr(self.vae, "enable_gradient_checkpointing") and enable:
            self.vae.enable_gradient_checkpointing()
        elif hasattr(self.vae, "disable_gradient_checkpointing") and not enable:
            self.vae.disable_gradient_checkpointing()

    def _preprocess_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Pad input tensor so height and width are divisible by 8 (VAE downsample factor),
        and rescale to [-1, 1] if input_rescale is True.
        """
        orig_h, orig_w = x.shape[2], x.shape[3]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        if self.input_rescale:
            # If tensor is roughly in [0, 1], map to [-1, 1]
            if x.min() >= -0.1 and x.max() <= 1.1:
                x = x * 2.0 - 1.0

        return x, orig_h, orig_w

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional UNet-style skip connections from encoder to decoder.

        Args:
            x: Input image tensor of shape [B, 3, H, W].

        Returns:
            If aux_classifier is True: Tuple of (mask_logits, class_logits)
            If aux_classifier is False: mask_logits of shape [B, out_channels, H, W]
        """
        x_proc, orig_h, orig_w = self._preprocess_input(x)

        # Encode image into latent space
        if self.use_skip_connections:
            if self.freeze_encoder:
                with torch.no_grad():
                    posterior, skips = self._encode_with_skips(x_proc)
                    skips = [s.detach() for s in skips]
            else:
                posterior, skips = self._encode_with_skips(x_proc)
        else:
            skips = None
            if self.freeze_encoder:
                with torch.no_grad():
                    posterior = self.vae.encode(x_proc).latent_dist
            else:
                posterior = self.vae.encode(x_proc).latent_dist

        if self.sample_mode == "sample" and self.training:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # Auxiliary classification on bottleneck latent features
        class_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            class_logits = self.classifier_head(z)

        # Scale latent for decoder
        z_scaled = z * self.scaling_factor

        # Decode latent back to binary mask space
        if self.use_skip_connections and skips is not None:
            mask_logits = self._decode_with_skips(z_scaled, skips)
        else:
            mask_logits = self.vae.decode(z_scaled).sample

        # Crop back to original dimensions if padded
        if mask_logits.shape[2] != orig_h or mask_logits.shape[3] != orig_w:
            mask_logits = mask_logits[:, :, :orig_h, :orig_w]

        if self.aux_classifier:
            return mask_logits, class_logits
        return mask_logits

    @torch.no_grad()
    def predict_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Produce binary mask prediction (0.0 or 1.0) given input image tensor."""
        self.eval()
        outputs = self.forward(x)
        mask_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.sigmoid(mask_logits)
        return (probs >= threshold).float()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        override_config: Optional[Union[Dict[str, Any], Any]] = None,
        strict: Optional[bool] = None,
        return_config: bool = False,
    ) -> Union[DiffusionVAEFinetune, Tuple[DiffusionVAEFinetune, Any]]:
        """Load DiffusionVAEFinetune model from checkpoint file (.pt)."""
        from sid_unet.models.unet import UNet
        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )


# Convenient aliases
VAEFinetune = DiffusionVAEFinetune
SDVAEFinetune = DiffusionVAEFinetune
