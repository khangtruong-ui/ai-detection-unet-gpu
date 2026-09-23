"""
Diffusion Multi-Noise Latent Feature Decoder (Diffusion-Diff) Model.
Extracts latent z0 from a frozen VAE encoder, perturbs z0 with noise at multiple timesteps,
computes diffuser-predicted noise, sinusoidal embeddings of timesteps and deviation (sigma),
and decodes the concatenated high-dimensional representation Z back to binary mask space
using a fully configurable trainable decoder.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.blocks import AuxiliaryClassifier

logger = logging.getLogger("sid_unet.models.diffusion_diff")

DEFAULT_DIFFUSION_CHECKPOINT = "runwayml/stable-diffusion-v1-5"


def sinusoidal_embedding(
    values: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """
    Compute sinusoidal positional embeddings for a 1D tensor of scalar values
    (e.g., timesteps or noise deviation sigma).

    Args:
        values: 1D tensor of scalar values of shape [B].
        dim: Embedding dimension.
        max_period: Maximum frequency period (default: 10000.0).

    Returns:
        Tensor of shape [B, dim].
    """
    if dim <= 0:
        return torch.empty((values.shape[0], 0), device=values.device, dtype=values.dtype)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=values.device)
        / half
    )
    args = values.unsqueeze(-1).float() * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def expand_to_spatial(emb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Expand [B, dim] embedding tensor across spatial dimensions to [B, dim, H, W]."""
    if emb.shape[1] == 0:
        return torch.empty((emb.shape[0], 0, height, width), device=emb.device, dtype=emb.dtype)
    return emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)


def get_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    """Create normalization layer based on configuration."""
    norm_type = norm_type.lower()
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "groupnorm":
        # Ensure num_groups divides num_channels
        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm_type in ("layernorm", "layer_norm"):
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    elif norm_type in ("none", "identity", ""):
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm_layer: {norm_type}")


def get_activation(activation: str) -> nn.Module:
    """Create activation layer based on configuration."""
    act = activation.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    elif act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif act == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with normalization, activation, and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: str = "batchnorm",
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = get_norm_layer(norm_layer, out_channels)
        self.act1 = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = get_norm_layer(norm_layer, out_channels)
        self.act2 = get_activation(activation)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm_layer(norm_layer, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act2(out + res)
        return out


class DecoderUpsampleStage(nn.Module):
    """Upsampling stage that increases spatial resolution by 2x."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_mode: str = "bilinear",
        norm_layer: str = "batchnorm",
        activation: str = "silu",
        dropout: float = 0.0,
        num_res_blocks: int = 1,
    ):
        super().__init__()
        self.upsample_mode = upsample_mode.lower()

        if self.upsample_mode == "transpose":
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        elif self.upsample_mode == "pixelshuffle":
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )
        else:  # bilinear
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            )

        self.norm = get_norm_layer(norm_layer, out_channels)
        self.act = get_activation(activation)

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(
                ResidualConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout=dropout,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.up(x)))
        x = self.res_blocks(x)
        return x


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
        norm_layer: str = "batchnorm",
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = dec_channels

        self.conv = nn.Conv2d(dec_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = get_norm_layer(norm_layer, out_channels)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, dec_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if dec_feat.shape[2:] != skip_feat.shape[2:]:
            skip_feat = F.interpolate(skip_feat, size=dec_feat.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([dec_feat, skip_feat], dim=1)
        return self.dropout(self.act(self.norm(self.conv(fused))))


class TrainableLatentDecoder(nn.Module):
    """
    Configurable trainable decoder that maps the high-dimensional latent representation Z
    at latent resolution (H/8, W/8) back to full binary image space (H, W), with optional
    UNet-style skip connections from the encoder.

    This module is entirely defined by config files and is the ONLY component with
    trainable parameters in the DiffusionDiffModel architecture.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Optional[List[int]] = None,
        out_channels: int = 1,
        upsample_mode: str = "bilinear",
        norm_layer: str = "batchnorm",
        activation: str = "silu",
        dropout: float = 0.0,
        num_res_blocks: int = 1,
        use_skip_connections: bool = True,
        skip_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        if channels is None:
            channels = [256, 128, 64, 32]
        self.channels = [int(c) for c in channels]
        self.out_channels = out_channels
        self.use_skip_connections = bool(use_skip_connections)

        # Initial projection to first channel dimension
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_layer, self.channels[0]),
            get_activation(activation),
        )

        # 3 Upsampling stages: H/8 -> H/4 -> H/2 -> H
        self.stages = nn.ModuleList()
        curr_ch = self.channels[0]
        for i in range(1, len(self.channels)):
            next_ch = self.channels[i]
            self.stages.append(
                DecoderUpsampleStage(
                    in_channels=curr_ch,
                    out_channels=next_ch,
                    upsample_mode=upsample_mode,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout=dropout,
                    num_res_blocks=num_res_blocks,
                )
            )
            curr_ch = next_ch

        # UNet-style skip connection fusions
        self.skip_fusions = nn.ModuleList()
        if self.use_skip_connections and skip_channels:
            for i in range(len(self.stages)):
                if i < len(skip_channels) and skip_channels[i] > 0:
                    stage_out_ch = self.channels[i + 1]
                    self.skip_fusions.append(
                        SkipFusion(
                            dec_channels=stage_out_ch,
                            skip_channels=skip_channels[i],
                            out_channels=stage_out_ch,
                            norm_layer=norm_layer,
                            activation=activation,
                            dropout=dropout,
                        )
                    )
                else:
                    self.skip_fusions.append(nn.Identity())

        # If fewer than 3 stages were defined in channels, pad upsampling to reach 8x
        remaining_upsample = 3 - (len(self.channels) - 1)
        self.extra_stages = nn.ModuleList()
        for _ in range(remaining_upsample):
            self.extra_stages.append(
                DecoderUpsampleStage(
                    in_channels=curr_ch,
                    out_channels=curr_ch,
                    upsample_mode=upsample_mode,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout=dropout,
                    num_res_blocks=num_res_blocks,
                )
            )

        # Final projection to binary mask logits
        self.out_conv = nn.Conv2d(curr_ch, out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.out_conv.weight, mode="fan_out", nonlinearity="linear")
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        feat = self.init_conv(z)
        for i, stage in enumerate(self.stages):
            feat = stage(feat)
            if skips is not None and self.use_skip_connections and i < len(self.skip_fusions):
                if i < len(skips) and skips[i] is not None:
                    feat = self.skip_fusions[i](feat, skips[i])
        for extra in self.extra_stages:
            feat = extra(feat)
        logits = self.out_conv(feat)
        return logits


class DiffusionDiffModel(nn.Module):
    """
    Diffusion Multi-Noise Latent Feature Model (Diffusion-Diff) for AI-Generated Synthetic Image Masking.

    Mechanism:
      1. Real image x [B, 3, H, W] is passed through VAE encoder (trainable = True by default) -> latent z0 [B, 4, H/8, W/8] and multi-scale skip features.
      2. For multiple configured timesteps (t_1, ..., t_K):
         - Add noise epsilon_k to z0 -> noisy versions z_{t_k}.
         - Pass noisy versions z_{t_k} through frozen diffuser UNet -> predicted noise eps_hat_k.
         - Compute sinusoidal embeddings of timestep t_k and noise deviation (sigma_k).
      3. Concatenate all:
         [z0, z_{t_k}, epsilon_k, eps_hat_k, sinusoidal_embeddings(t_k), sinusoidal_embeddings(sigma_k)]
         into high-dimensional representation Z [B, C_Z, H/8, W/8].
      4. Pass Z through a configurable trainable decoder with multi-scale encoder skip fusions.
      5. The VAE autoencoder and decoder (and optional auxiliary classifier) are trainable, while the diffuser UNet remains strictly frozen.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = DEFAULT_DIFFUSION_CHECKPOINT,
        vae_subfolder: Optional[str] = "vae",
        unet_subfolder: Optional[str] = "unet",
        timesteps: Optional[List[int]] = None,
        timestep_embed_dim: int = 32,
        sigma_embed_dim: int = 32,
        include_noisy_latents: bool = True,
        include_added_noise: bool = True,
        include_predicted_noise: bool = True,
        include_noise_diff: bool = True,
        include_z0: bool = True,
        decoder_config: Optional[Dict[str, Any]] = None,
        aux_classifier: bool = True,
        num_classes: int = 3,
        in_channels: int = 3,
        out_channels: int = 1,
        scaling_factor: float = 0.18215,
        use_dummy: bool = False,
        dummy_vae_channels: Tuple[int, ...] = (32, 64),
        dummy_unet_channels: Tuple[int, ...] = (32, 64),
        input_rescale: bool = True,
        use_skip_connections: bool = True,
        diffuser_fp16: bool = True,
        autoencoder_trainable: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.vae_subfolder = vae_subfolder
        self.unet_subfolder = unet_subfolder
        self.diffuser_fp16 = bool(kwargs.get("diffuser_fp16", diffuser_fp16))
        self.autoencoder_trainable = bool(kwargs.get("autoencoder_trainable", kwargs.get("trainable_autoencoder", autoencoder_trainable)))
        self.timesteps = [int(t) for t in (timesteps or [100, 250, 500])]
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.sigma_embed_dim = int(sigma_embed_dim)
        self.include_noisy_latents = bool(include_noisy_latents)
        self.include_added_noise = bool(include_added_noise)
        self.include_predicted_noise = bool(include_predicted_noise)
        self.include_noise_diff = bool(include_noise_diff)
        self.include_z0 = bool(include_z0)
        self.aux_classifier = bool(aux_classifier)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.scaling_factor = float(scaling_factor)
        self.input_rescale = bool(input_rescale)
        self.use_skip_connections = bool(kwargs.get("skip_connections", use_skip_connections))

        # 1. Initialize VAE encoder and freeze
        self._init_frozen_vae(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=vae_subfolder,
            use_dummy=use_dummy,
            dummy_channels=dummy_vae_channels,
            in_channels=in_channels,
        )

        # 2. Initialize Diffuser UNet and freeze
        self._init_frozen_diffuser(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=unet_subfolder,
            use_dummy=use_dummy,
            dummy_channels=dummy_unet_channels,
        )

        # 3. Setup Noise Schedule (alphas and sigmas)
        self._setup_noise_schedule()

        # 4. Calculate total channel dimension of concatenated high-dimensional Z
        self.total_z_channels = self._calculate_z_channels()
        logger.info(f"High-dimensional representation Z channel dimension: {self.total_z_channels}")

        # 5. Initialize Trainable Decoder (defined by config files)
        dec_cfg = dict(decoder_config or {})
        dec_channels = dec_cfg.get("channels", [256, 128, 64, 32])
        dec_upsample_mode = dec_cfg.get("upsample_mode", "bilinear")
        dec_norm = dec_cfg.get("norm_layer", "batchnorm")
        dec_act = dec_cfg.get("activation", "silu")
        dec_dropout = float(dec_cfg.get("dropout", 0.0))
        dec_res_blocks = int(dec_cfg.get("num_res_blocks", 1))

        # Normalization layer for concatenated high-dimensional Z representation
        z_norm_type = dec_cfg.get("z_norm", dec_cfg.get("norm_layer", "groupnorm"))
        self.z_norm = get_norm_layer(z_norm_type, self.total_z_channels)

        # Determine encoder skip channels
        if self.use_skip_connections:
            with torch.no_grad():
                dummy_x = torch.zeros((1, in_channels, 64, 64))
                _, dummy_skips = self._encode_with_skips(dummy_x)
                skip_channels = [s.shape[1] for s in dummy_skips]
        else:
            skip_channels = None

        self.decoder = TrainableLatentDecoder(
            in_channels=self.total_z_channels,
            channels=dec_channels,
            out_channels=out_channels,
            upsample_mode=dec_upsample_mode,
            norm_layer=dec_norm,
            activation=dec_act,
            dropout=dec_dropout,
            num_res_blocks=dec_res_blocks,
            use_skip_connections=self.use_skip_connections,
            skip_channels=skip_channels,
        )

        # 6. Auxiliary classifier head on Z representation
        if self.aux_classifier:
            self.classifier_head = AuxiliaryClassifier(
                in_channels=self.total_z_channels,
                num_classes=num_classes,
                dropout=dec_dropout,
            )
        else:
            self.classifier_head = None

        # Strictly freeze VAE and Diffuser parameters
        self._enforce_freeze()

    def _init_frozen_vae(
        self,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str],
        use_dummy: bool,
        dummy_channels: Tuple[int, ...],
        in_channels: int,
    ) -> None:
        from diffusers import AutoencoderKL

        if use_dummy:
            logger.info("Initializing lightweight dummy AutoencoderKL...")
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
                load_kwargs: Dict[str, Any] = {}
                if subfolder:
                    load_kwargs["subfolder"] = subfolder
                self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
            except Exception as e:
                logger.warning(f"Could not load pretrained VAE: {e}. Using dummy AutoencoderKL...")
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

        self.latent_channels = getattr(self.vae.config, "latent_channels", 4)

    def _init_frozen_diffuser(
        self,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str],
        use_dummy: bool,
        dummy_channels: Tuple[int, ...],
    ) -> None:
        from diffusers import UNet2DConditionModel

        if use_dummy:
            logger.info("Initializing lightweight dummy UNet2DConditionModel...")
            blocks = list(dummy_channels)
            down_types = ["DownBlock2D"] * len(blocks)
            up_types = ["UpBlock2D"] * len(blocks)
            self.diffuser = UNet2DConditionModel(
                sample_size=32,
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                layers_per_block=1,
                block_out_channels=tuple(blocks),
                down_block_types=tuple(down_types),
                up_block_types=tuple(up_types),
                cross_attention_dim=32,
            )
        else:
            try:
                load_kwargs: Dict[str, Any] = {}
                if subfolder:
                    load_kwargs["subfolder"] = subfolder
                if self.diffuser_fp16 and torch.cuda.is_available():
                    load_kwargs["torch_dtype"] = torch.float16
                self.diffuser = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
            except Exception as e:
                logger.warning(f"Could not load pretrained UNet: {e}. Using dummy UNet2DConditionModel...")
                blocks = list(dummy_channels)
                down_types = ["DownBlock2D"] * len(blocks)
                up_types = ["UpBlock2D"] * len(blocks)
                self.diffuser = UNet2DConditionModel(
                    sample_size=32,
                    in_channels=self.latent_channels,
                    out_channels=self.latent_channels,
                    layers_per_block=1,
                    block_out_channels=tuple(blocks),
                    down_block_types=tuple(down_types),
                    up_block_types=tuple(up_types),
                    cross_attention_dim=32,
                )

    def _setup_noise_schedule(self) -> None:
        """Setup standard 1000-step linear/scaled diffusion noise schedule."""
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

    def _calculate_z_channels(self) -> int:
        """Calculate total number of concatenated channels in representation Z."""
        c = 0
        if self.include_z0:
            c += self.latent_channels

        per_step_ch = 0
        if self.include_noisy_latents:
            per_step_ch += self.latent_channels
        if self.include_added_noise:
            per_step_ch += self.latent_channels
        if self.include_predicted_noise:
            per_step_ch += self.latent_channels
        if self.include_noise_diff:
            per_step_ch += self.latent_channels
        per_step_ch += self.timestep_embed_dim
        per_step_ch += self.sigma_embed_dim

        c += len(self.timesteps) * per_step_ch
        return c

    def _enforce_freeze(self) -> None:
        """Freeze parameters according to configuration."""
        if not self.autoencoder_trainable:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
        else:
            for param in self.vae.parameters():
                param.requires_grad = True
            self.vae.train()

        # Diffuser UNet is strictly frozen
        for param in self.diffuser.parameters():
            param.requires_grad = False
        self.diffuser.eval()

    def train(self, mode: bool = True):
        """Set training mode for trainable components while keeping frozen diffuser in eval."""
        super().train(mode)
        if not self.autoencoder_trainable:
            self.vae.eval()
        else:
            self.vae.train(mode)
        self.diffuser.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> "DiffusionDiffModel":
        """Move model to device and cast frozen diffuser to half precision if on CUDA."""
        res = super().to(*args, **kwargs)
        if getattr(self, "diffuser_fp16", False):
            try:
                device = next(self.parameters()).device
                if device.type == "cuda" and hasattr(self, "diffuser") and self.diffuser is not None:
                    self.diffuser = self.diffuser.to(device=device, dtype=torch.float16)
            except Exception:
                pass
        return res

    def _encode_with_skips(self, x: torch.Tensor) -> Tuple[Any, List[torch.Tensor]]:
        """Run VAE encoder to obtain latent distribution and multi-scale skip features."""
        grad_ctx = torch.enable_grad() if (self.training and self.autoencoder_trainable) else torch.no_grad()
        with grad_ctx:
            sample = self.vae.encoder.conv_in(x)
            skips = [sample]

            for down_block in self.vae.encoder.down_blocks:
                sample = down_block(sample)
                skips.append(sample)

            sample = self.vae.encoder.mid_block(sample)
            sample = self.vae.encoder.conv_norm_out(sample)
            sample = self.vae.encoder.conv_act(sample)
            sample = self.vae.encoder.conv_out(sample)

            if getattr(self.vae, "quant_conv", None) is not None:
                sample = self.vae.quant_conv(sample)

            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
            posterior = DiagonalGaussianDistribution(sample)

            # Map skips to the 3 decoder stages:
            # Stage 0: H/4 resolution
            # Stage 1: H/2 resolution
            # Stage 2: H resolution
            target_h = x.shape[2]
            h_4 = target_h // 4
            h_2 = target_h // 2

            skip_h4 = None
            skip_h2 = None
            skip_h = skips[0]

            for s in skips:
                if s.shape[2] == h_4:
                    skip_h4 = s
                elif s.shape[2] == h_2:
                    skip_h2 = s

            if skip_h4 is None:
                skip_h4 = skips[-1] if len(skips) > 2 else skips[-1]
            if skip_h2 is None:
                skip_h2 = skips[1] if len(skips) > 1 else skips[0]

            if self.training and self.autoencoder_trainable:
                ordered_skips = [skip_h4, skip_h2, skip_h]
            else:
                ordered_skips = [skip_h4.detach(), skip_h2.detach(), skip_h.detach()]
            return posterior, ordered_skips

    def _preprocess_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Pad input to be divisible by 8 and rescale to [-1, 1] if needed."""
        orig_h, orig_w = x.shape[2], x.shape[3]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        if self.input_rescale:
            if x.min() >= -0.1 and x.max() <= 1.1:
                x = x * 2.0 - 1.0

        return x, orig_h, orig_w

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Runs frozen VAE encoder, multi-step noise perturbation, frozen diffuser prediction,
        sinusoidal embeddings, concatenation to Z, and trainable decoder.
        """
        device = x.device
        dtype = x.dtype
        b_sz = x.shape[0]

        x_proc, orig_h, orig_w = self._preprocess_input(x)

        # 1. Real image passed through encoder to obtain latent z0 and encoder skips
        if self.use_skip_connections:
            posterior, enc_skips = self._encode_with_skips(x_proc)
            if self.training and self.autoencoder_trainable:
                z0 = posterior.mode() * self.scaling_factor
            else:
                with torch.no_grad():
                    z0 = posterior.mode() * self.scaling_factor
        else:
            enc_skips = None
            if self.training and self.autoencoder_trainable:
                posterior = self.vae.encode(x_proc).latent_dist
                z0 = posterior.mode() * self.scaling_factor
            else:
                with torch.no_grad():
                    posterior = self.vae.encode(x_proc).latent_dist
                    z0 = posterior.mode() * self.scaling_factor

        # Prevent runaway latent drift from destabilizing frozen diffuser UNet
        z0 = torch.clamp(z0, min=-15.0, max=15.0)

        _, _, h_z, w_z = z0.shape

        # Components to concatenate into high-dimensional Z
        z_components: List[torch.Tensor] = []
        if self.include_z0:
            z_components.append(z0 if (self.training and self.autoencoder_trainable) else z0.detach())

        # Prepare unconditioned text embeddings for conditional diffusers if needed
        cross_dim = getattr(self.diffuser.config, "cross_attention_dim", None)
        if cross_dim is not None:
            uncond_emb = torch.zeros((b_sz, 1, cross_dim), device=device, dtype=dtype)
        else:
            uncond_emb = None

        # 2. Add noise to z0 at different timesteps and get diffuser predictions
        for t_val in self.timesteps:
            t_clamped = max(0, min(t_val, len(self.alphas_cumprod) - 1))
            alpha_bar = self.alphas_cumprod[t_clamped].to(device=device, dtype=dtype)
            sigma_val = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
            sqrt_alpha_bar = torch.sqrt(alpha_bar)

            # Sample added noise epsilon_k
            eps_k = torch.randn_like(z0)

            # Noisy version z_{t_k}
            z_tk = sqrt_alpha_bar * z0 + sigma_val * eps_k

            # Diffuser predicted noise
            t_tensor = torch.full((b_sz,), t_clamped, device=device, dtype=torch.long)
            with torch.no_grad():
                diffuser_kwargs: Dict[str, Any] = {}
                diff_dtype = next(self.diffuser.parameters()).dtype if list(self.diffuser.parameters()) else dtype
                if uncond_emb is not None:
                    diffuser_kwargs["encoder_hidden_states"] = uncond_emb.to(diff_dtype)
                z_tk_input = z_tk.to(diff_dtype)
                pred_eps = self.diffuser(z_tk_input, t_tensor, **diffuser_kwargs).sample.detach().to(dtype)

            # Sinusoidal embeddings of timestep and deviation (sigma)
            t_emb = sinusoidal_embedding(t_tensor.float(), dim=self.timestep_embed_dim)
            t_spatial = expand_to_spatial(t_emb, h_z, w_z).to(dtype=dtype)

            sigma_tensor = torch.full((b_sz,), sigma_val.item(), device=device, dtype=torch.float32)
            sigma_emb = sinusoidal_embedding(sigma_tensor, dim=self.sigma_embed_dim)
            sigma_spatial = expand_to_spatial(sigma_emb, h_z, w_z).to(dtype=dtype)

            # Append components for this timestep
            if self.include_noisy_latents:
                z_components.append(z_tk if (self.training and self.autoencoder_trainable) else z_tk.detach())
            if self.include_added_noise:
                z_components.append(eps_k.detach())
            if self.include_predicted_noise:
                z_components.append(pred_eps)
            if self.include_noise_diff:
                z_components.append((pred_eps - eps_k).detach())
            if self.timestep_embed_dim > 0:
                z_components.append(t_spatial)
            if self.sigma_embed_dim > 0:
                z_components.append(sigma_spatial)

        # 3. Concatenate along channel dimension to get high-dimensional Z and normalize
        z_high_dim = torch.cat(z_components, dim=1)
        z_high_dim = self.z_norm(z_high_dim)

        # 4. Auxiliary classifier on Z representation if enabled
        class_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            class_logits = self.classifier_head(z_high_dim)

        # 5. Trainable Decoder decodes Z back to binary mask space (fusing encoder skips if enabled)
        mask_logits = self.decoder(z_high_dim, skips=enc_skips)

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
    ) -> Union[DiffusionDiffModel, Tuple[DiffusionDiffModel, Any]]:
        """Load DiffusionDiffModel from checkpoint file (.pt)."""
        from sid_unet.models.unet import UNet
        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )


# Convenient aliases
DiffusionDiff = DiffusionDiffModel
DiffuserNoiseDecoder = DiffusionDiffModel
DiffusionMultiNoiseDecoder = DiffusionDiffModel
