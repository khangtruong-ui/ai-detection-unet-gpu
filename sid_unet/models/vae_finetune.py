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


class DiffusionVAEFinetune(nn.Module):
    """
    Finetuned VAE taken from a diffusion model (such as Stable Diffusion 1.5 AutoencoderKL)
    for AI-generated image mask segmentation.

    The encoder compresses the input image into latent distribution z, and the decoder
    upsamples and projects back into the binary mask logit space.
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

        self._init_vae(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            use_dummy=use_dummy,
            dummy_channels=dummy_channels,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        if self.freeze_encoder:
            self._freeze_encoder()

        # Latent channels from VAE
        latent_ch = getattr(self.vae.config, "latent_channels", 4)
        self.latent_channels = latent_ch

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

    def _freeze_encoder(self) -> None:
        """Freeze VAE encoder and quant_conv parameters."""
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
        if hasattr(self.vae, "quant_conv"):
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
        Forward pass.

        Args:
            x: Input image tensor of shape [B, 3, H, W].

        Returns:
            If aux_classifier is True: Tuple of (mask_logits, class_logits)
            If aux_classifier is False: mask_logits of shape [B, out_channels, H, W]
        """
        x_proc, orig_h, orig_w = self._preprocess_input(x)

        # Encode image into latent space
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
