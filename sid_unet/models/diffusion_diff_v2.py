"""
Diffusion Multi-Noise Latent Feature Decoder V2 (Diffusion-Diff-V2) Model.

Key Architectural Advancements:
1. Frozen VAE Encoder by default: Real image x is passed through a frozen VAE encoder to obtain
   diffusion latent representation z0.
2. No Encoder-to-Decoder Skips by default (use_encoder_skips=False): Skips from encoder to the
   trainable decoder are disabled by default.
3. Parallel Pretrained Frozen Decoder: A parallel pretrained, frozen VAE decoder (from diffusion model)
   decodes the diffusion latent z0.
4. Perpendicular Skip Connection: Feature representations are extracted from intermediate layers of the
   parallel frozen decoder and injected perpendicularly into the corresponding layers of the trainable decoder.
   Flow: diffusion latent -> frozen decoder -> output extracted from layers of the frozen decoder -> inject to layers of trainable decoder.
5. High-Dimensional Representation Z: Formed by concatenating clean latent z0, perturbed latents z_{t_k},
   added noises eps_k, diffuser-predicted noises eps_hat_k, noise differences, and sinusoidal embeddings of
   timesteps and noise standard deviations across multiple diffusion steps.
6. Trainable Latent Decoder: Fuses representation Z with the perpendicular skips from the frozen decoder
   to predict the final binary tampering mask logits.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.blocks import AuxiliaryClassifier
from sid_unet.models.diffusion_diff import (
    DEFAULT_DIFFUSION_CHECKPOINT,
    DecoderUpsampleStage,
    ResidualConvBlock,
    expand_to_spatial,
    get_activation,
    get_norm_layer,
    sinusoidal_embedding,
)

logger = logging.getLogger("sid_unet.models.diffusion_diff_v2")


class PerpendicularSkipFusion(nn.Module):
    """
    Perpendicular skip connection fusion block.
    Fuses feature maps extracted from intermediate layers of the parallel frozen decoder
    into intermediate layers of the trainable decoder.

    Concatenates trainable decoder features with perpendicular frozen decoder features,
    followed by convolution, normalization, activation, and dropout to project back
    to the target decoder feature channel dimension.
    """

    def __init__(
        self,
        dec_channels: int,
        perp_channels: int,
        out_channels: Optional[int] = None,
        norm_layer: str = "batchnorm",
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = dec_channels

        # Normalize perpendicular features before concatenation to match decoder activation scale
        self.perp_norm = get_norm_layer(norm_layer, perp_channels)
        self.conv = nn.Conv2d(
            dec_channels + perp_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = get_norm_layer(norm_layer, out_channels)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, dec_feat: torch.Tensor, perp_feat: torch.Tensor) -> torch.Tensor:
        if dec_feat.shape[2:] != perp_feat.shape[2:]:
            perp_feat = F.interpolate(
                perp_feat,
                size=dec_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        perp_feat = self.perp_norm(perp_feat)
        fused = torch.cat([dec_feat, perp_feat], dim=1)
        out = self.dropout(self.act(self.norm(self.conv(fused))))
        if out.shape == dec_feat.shape:
            return dec_feat + out
        return out


class EncoderSkipFusion(nn.Module):
    """
    Optional UNet-style skip connection fusion block from encoder to trainable decoder.
    (Disabled by default in Diffusion-Diff-V2).
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

        # Normalize encoder skip features before concatenation
        self.skip_norm = get_norm_layer(norm_layer, skip_channels)
        self.conv = nn.Conv2d(
            dec_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = get_norm_layer(norm_layer, out_channels)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, dec_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if dec_feat.shape[2:] != skip_feat.shape[2:]:
            skip_feat = F.interpolate(
                skip_feat,
                size=dec_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        skip_feat = self.skip_norm(skip_feat)
        fused = torch.cat([dec_feat, skip_feat], dim=1)
        out = self.dropout(self.act(self.norm(self.conv(fused))))
        if out.shape == dec_feat.shape:
            return dec_feat + out
        return out


class TrainableLatentDecoderV2(nn.Module):
    """
    Trainable decoder for Diffusion-Diff-V2 that maps high-dimensional latent representation Z
    at latent resolution (H/8, W/8) back to binary mask space (H, W).

    Features:
    - Injects perpendicular skip connections from intermediate layers of the parallel frozen decoder.
    - Optional UNet-style encoder skip connections (default: disabled / no skip).
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
        use_perpendicular_skips: bool = True,
        perpendicular_skip_channels: Optional[List[int]] = None,
        use_encoder_skips: bool = False,
        encoder_skip_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        if channels is None:
            channels = [256, 128, 64, 32]
        self.channels = [int(c) for c in channels]
        self.out_channels = out_channels
        self.use_perpendicular_skips = bool(use_perpendicular_skips)
        self.use_encoder_skips = bool(use_encoder_skips)

        # Initial projection from total Z channels to first channel dimension
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_layer, self.channels[0]),
            get_activation(activation),
        )

        # Progressive upsampling stages: H/8 -> H/4 -> H/2 -> H
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

        # Perpendicular skip fusions from parallel frozen decoder
        self.perp_fusions = nn.ModuleList()
        if self.use_perpendicular_skips and perpendicular_skip_channels:
            for i in range(len(self.stages)):
                if i < len(perpendicular_skip_channels) and perpendicular_skip_channels[i] > 0:
                    stage_out_ch = self.channels[i + 1]
                    self.perp_fusions.append(
                        PerpendicularSkipFusion(
                            dec_channels=stage_out_ch,
                            perp_channels=perpendicular_skip_channels[i],
                            out_channels=stage_out_ch,
                            norm_layer=norm_layer,
                            activation=activation,
                            dropout=dropout,
                        )
                    )
                else:
                    self.perp_fusions.append(nn.Identity())

        # UNet-style encoder skip connections (default: disabled)
        self.encoder_skip_fusions = nn.ModuleList()
        if self.use_encoder_skips and encoder_skip_channels:
            for i in range(len(self.stages)):
                if i < len(encoder_skip_channels) and encoder_skip_channels[i] > 0:
                    stage_out_ch = self.channels[i + 1]
                    self.encoder_skip_fusions.append(
                        EncoderSkipFusion(
                            dec_channels=stage_out_ch,
                            skip_channels=encoder_skip_channels[i],
                            out_channels=stage_out_ch,
                            norm_layer=norm_layer,
                            activation=activation,
                            dropout=dropout,
                        )
                    )
                else:
                    self.encoder_skip_fusions.append(nn.Identity())

        # Extra padding stages if fewer than 3 stages defined
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

        # Final projection to mask logits
        self.out_conv = nn.Conv2d(curr_ch, out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.out_conv.weight, mode="fan_out", nonlinearity="linear")
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        z: torch.Tensor,
        perp_skips: Optional[List[torch.Tensor]] = None,
        encoder_skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        feat = self.init_conv(z)
        for i, stage in enumerate(self.stages):
            feat = stage(feat)
            # 1. Inject perpendicular skip from parallel frozen decoder
            if (
                perp_skips is not None
                and self.use_perpendicular_skips
                and i < len(self.perp_fusions)
            ):
                if i < len(perp_skips) and perp_skips[i] is not None:
                    feat = self.perp_fusions[i](feat, perp_skips[i])
            # 2. Inject encoder skip if enabled
            if (
                encoder_skips is not None
                and self.use_encoder_skips
                and i < len(self.encoder_skip_fusions)
            ):
                if i < len(encoder_skips) and encoder_skips[i] is not None:
                    feat = self.encoder_skip_fusions[i](feat, encoder_skips[i])
        for extra in self.extra_stages:
            feat = extra(feat)
        logits = self.out_conv(feat)
        return logits


class DiffusionDiffV2Model(nn.Module):
    """
    Diffusion Multi-Noise Latent Feature Decoder V2 (Diffusion-Diff-V2) Model.

    Key Architectural Highlights:
      1. VAE Encoder is frozen by default (`freeze_encoder = True`).
      2. Encoder-to-trainable-decoder skips are disabled by default (`use_encoder_skips = False`).
      3. A parallel pretrained frozen decoder (taken from diffusion model) runs in parallel on
         the diffusion latent z0.
      4. Perpendicular skip connection flow:
         diffusion latent z0 -> parallel frozen decoder -> outputs extracted from intermediate
         layers of frozen decoder -> injected to layers of trainable decoder.
      5. Diffuser UNet and frozen decoder remain strictly frozen throughout training.
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
        diffuser_fp16: bool = True,
        freeze_encoder: bool = True,
        use_encoder_skips: bool = False,
        use_perpendicular_skips: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.vae_subfolder = vae_subfolder
        self.unet_subfolder = unet_subfolder
        self.diffuser_fp16 = bool(kwargs.get("diffuser_fp16", diffuser_fp16))

        # Handle freeze_encoder / autoencoder_trainable configurations
        if "autoencoder_trainable" in kwargs:
            self.freeze_encoder = not bool(kwargs["autoencoder_trainable"])
        elif "trainable_autoencoder" in kwargs:
            self.freeze_encoder = not bool(kwargs["trainable_autoencoder"])
        else:
            self.freeze_encoder = bool(freeze_encoder)

        # Handle encoder skips (defaults to False in v2)
        if "use_skip_connections" in kwargs:
            self.use_encoder_skips = bool(kwargs["use_skip_connections"])
        elif "skip_connections" in kwargs:
            self.use_encoder_skips = bool(kwargs["skip_connections"])
        else:
            self.use_encoder_skips = bool(use_encoder_skips)

        # Perpendicular skips from parallel frozen decoder (defaults to True in v2)
        self.use_perpendicular_skips = bool(kwargs.get("perpendicular_skips", use_perpendicular_skips))

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

        # 1. Initialize VAE (contains frozen encoder and parallel frozen decoder)
        self._init_vae(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=vae_subfolder,
            use_dummy=use_dummy,
            dummy_channels=dummy_vae_channels,
            in_channels=in_channels,
        )

        # 2. Initialize Diffuser UNet and freeze
        self._init_diffuser(
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

        # 5. Determine layer dimensions for perpendicular skip connections
        dec_cfg = dict(decoder_config or {})
        dec_channels = dec_cfg.get("channels", [256, 128, 64, 32])
        dec_upsample_mode = dec_cfg.get("upsample_mode", "bilinear")
        dec_norm = dec_cfg.get("norm_layer", "batchnorm")
        dec_act = dec_cfg.get("activation", "silu")
        dec_dropout = float(dec_cfg.get("dropout", 0.0))
        dec_res_blocks = int(dec_cfg.get("num_res_blocks", 1))

        # Normalization layer for concatenated high-dimensional Z representation
        # Default to groupnorm to avoid zeroing out sinusoidal embeddings that have zero spatial variance
        z_norm_type = dec_cfg.get("z_norm", "groupnorm")
        self.z_norm = get_norm_layer(z_norm_type, self.total_z_channels)

        self.perp_mapping: List[Optional[int]] = []
        perp_skip_channels: Optional[List[int]] = None
        if self.use_perpendicular_skips:
            with torch.no_grad():
                dummy_z = torch.zeros((1, self.latent_channels, 32, 32))
                raw_frozen_feats = self._extract_frozen_decoder_features(dummy_z)
                frozen_shapes = [(f.shape[1], f.shape[2], f.shape[3]) for f in raw_frozen_feats]

                # Compute expected spatial output for each trainable stage
                trainable_stage_shapes = []
                cur_h, cur_w = 32, 32
                for i in range(1, len(dec_channels)):
                    cur_h, cur_w = cur_h * 2, cur_w * 2
                    trainable_stage_shapes.append((dec_channels[i], cur_h, cur_w))

                self.perp_mapping = self._match_stages(frozen_shapes, trainable_stage_shapes)
                perp_skip_channels = [
                    raw_frozen_feats[m].shape[1] if m is not None else 0
                    for m in self.perp_mapping
                ]
                logger.info(
                    f"Perpendicular skip mapping: {self.perp_mapping} with channels {perp_skip_channels}"
                )

        # Determine encoder skip channels if enabled
        if self.use_encoder_skips:
            with torch.no_grad():
                dummy_x = torch.zeros((1, in_channels, 64, 64))
                _, dummy_skips = self._encode_with_skips(dummy_x)
                encoder_skip_channels = [s.shape[1] for s in dummy_skips]
        else:
            encoder_skip_channels = None

        # 6. Initialize Trainable Decoder V2
        self.decoder = TrainableLatentDecoderV2(
            in_channels=self.total_z_channels,
            channels=dec_channels,
            out_channels=out_channels,
            upsample_mode=dec_upsample_mode,
            norm_layer=dec_norm,
            activation=dec_act,
            dropout=dec_dropout,
            num_res_blocks=dec_res_blocks,
            use_perpendicular_skips=self.use_perpendicular_skips,
            perpendicular_skip_channels=perp_skip_channels,
            use_encoder_skips=self.use_encoder_skips,
            encoder_skip_channels=encoder_skip_channels,
        )

        # 7. Auxiliary classifier head on Z representation
        if self.aux_classifier:
            self.classifier_head = AuxiliaryClassifier(
                in_channels=self.total_z_channels,
                num_classes=num_classes,
                dropout=dec_dropout,
            )
        else:
            self.classifier_head = None

        # 8. Strictly enforce freezing according to specifications
        self._enforce_freeze()

    @staticmethod
    def _match_stages(
        frozen_shapes: List[Tuple[int, int, int]],
        trainable_stage_shapes: List[Tuple[int, int, int]],
    ) -> List[Optional[int]]:
        """
        Match trainable decoder stages to frozen decoder layers by resolution proximity.
        """
        mapping: List[Optional[int]] = []
        for _, t_h, t_w in trainable_stage_shapes:
            if not frozen_shapes:
                mapping.append(None)
                continue
            exact_matches = [
                idx for idx, (_, f_h, f_w) in enumerate(frozen_shapes)
                if f_h == t_h and f_w == t_w
            ]
            if exact_matches:
                mapping.append(exact_matches[-1])
            else:
                diffs = [abs(f_h - t_h) for _, f_h, f_w in frozen_shapes]
                mapping.append(diffs.index(min(diffs)))
        return mapping

    def _init_vae(
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

    def _init_diffuser(
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
                self.diffuser = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path, **load_kwargs
                )
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
        """
        Freeze parameters according to architectural specifications:
        - Diffuser UNet is strictly frozen.
        - Parallel VAE decoder is strictly frozen.
        - VAE encoder is frozen by default (freeze_encoder=True).
        """
        # Diffuser UNet is strictly frozen
        for param in self.diffuser.parameters():
            param.requires_grad = False
        self.diffuser.eval()

        # Parallel pretrained VAE decoder is strictly frozen
        for param in self.vae.decoder.parameters():
            param.requires_grad = False
        self.vae.decoder.eval()
        if getattr(self.vae, "post_quant_conv", None) is not None:
            for param in self.vae.post_quant_conv.parameters():
                param.requires_grad = False
            self.vae.post_quant_conv.eval()

        # VAE Encoder
        if self.freeze_encoder:
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
            self.vae.encoder.eval()
            if getattr(self.vae, "quant_conv", None) is not None:
                for param in self.vae.quant_conv.parameters():
                    param.requires_grad = False
                self.vae.quant_conv.eval()
        else:
            for param in self.vae.encoder.parameters():
                param.requires_grad = True
            self.vae.encoder.train()
            if getattr(self.vae, "quant_conv", None) is not None:
                for param in self.vae.quant_conv.parameters():
                    param.requires_grad = True
                self.vae.quant_conv.train()

    def train(self, mode: bool = True):
        """Set training mode for trainable components while keeping frozen components in eval."""
        super().train(mode)
        self.diffuser.eval()
        self.vae.decoder.eval()
        if getattr(self.vae, "post_quant_conv", None) is not None:
            self.vae.post_quant_conv.eval()

        if self.freeze_encoder:
            self.vae.encoder.eval()
            if getattr(self.vae, "quant_conv", None) is not None:
                self.vae.quant_conv.eval()
        else:
            self.vae.encoder.train(mode)
            if getattr(self.vae, "quant_conv", None) is not None:
                self.vae.quant_conv.train(mode)
        return self

    def to(self, *args: Any, **kwargs: Any) -> "DiffusionDiffV2Model":
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

    def _extract_frozen_decoder_features(self, z0: torch.Tensor) -> List[torch.Tensor]:
        """
        Run the parallel pretrained, frozen decoder on diffusion latent z0 and extract
        intermediate layer feature maps for perpendicular skip injection.
        """
        z_in = (z0 / self.scaling_factor) if self.scaling_factor != 0 else z0
        if getattr(self.vae, "post_quant_conv", None) is not None:
            z_in = self.vae.post_quant_conv(z_in)

        sample = self.vae.decoder.conv_in(z_in)
        sample = self.vae.decoder.mid_block(sample)

        feats: List[torch.Tensor] = []
        for up_block in self.vae.decoder.up_blocks:
            sample = up_block(sample)
            feats.append(sample)
        return feats

    def _encode_with_skips(self, x: torch.Tensor) -> Tuple[Any, List[torch.Tensor]]:
        """Run VAE encoder to obtain latent distribution and multi-scale skip features."""
        grad_ctx = torch.enable_grad() if (self.training and not self.freeze_encoder) else torch.no_grad()
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

            if self.training and not self.freeze_encoder:
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
        Forward pass for Diffusion-Diff-V2.

        Flow:
          1. Preprocess input image x.
          2. Pass x through frozen VAE encoder -> latent z0.
          3. Pass z0 through parallel pretrained frozen decoder -> extract perpendicular skip features.
          4. Perturb z0 across timesteps, pass through frozen Diffuser UNet, compute sinusoidal embeddings.
          5. Concatenate components into high-dimensional representation Z.
          6. Decode Z using TrainableLatentDecoderV2 with perpendicular skips injected into its layers.
        """
        device = x.device
        dtype = x.dtype
        b_sz = x.shape[0]

        x_proc, orig_h, orig_w = self._preprocess_input(x)

        # 1. Real image passed through VAE encoder (frozen by default) to obtain diffusion latent z0
        if self.use_encoder_skips:
            posterior, enc_skips = self._encode_with_skips(x_proc)
            if self.training and not self.freeze_encoder:
                z0 = posterior.mode() * self.scaling_factor
            else:
                with torch.no_grad():
                    z0 = posterior.mode() * self.scaling_factor
        else:
            enc_skips = None
            if self.training and not self.freeze_encoder:
                posterior = self.vae.encode(x_proc).latent_dist
                z0 = posterior.mode() * self.scaling_factor
            else:
                with torch.no_grad():
                    posterior = self.vae.encode(x_proc).latent_dist
                    z0 = posterior.mode() * self.scaling_factor

        # Prevent runaway latent drift from destabilizing frozen diffuser UNet
        z0 = torch.clamp(z0, min=-15.0, max=15.0)

        _, _, h_z, w_z = z0.shape

        # 2. Parallel frozen decoder: extract perpendicular skip features from diffusion latent z0
        perp_skips: Optional[List[torch.Tensor]] = None
        if self.use_perpendicular_skips:
            with torch.no_grad():
                raw_frozen_feats = self._extract_frozen_decoder_features(z0.detach())
                perp_skips = [
                    raw_frozen_feats[m].detach() if (m is not None and m < len(raw_frozen_feats)) else None
                    for m in self.perp_mapping
                ]

        # 3. Multi-step perturbations and diffuser predictions
        z_components: List[torch.Tensor] = []
        if self.include_z0:
            z_components.append(z0 if (self.training and not self.freeze_encoder) else z0.detach())

        cross_dim = getattr(self.diffuser.config, "cross_attention_dim", None)
        if cross_dim is not None:
            uncond_emb = torch.zeros((b_sz, 1, cross_dim), device=device, dtype=dtype)
        else:
            uncond_emb = None

        for t_val in self.timesteps:
            t_clamped = max(0, min(t_val, len(self.alphas_cumprod) - 1))
            alpha_bar = self.alphas_cumprod[t_clamped].to(device=device, dtype=dtype)
            sigma_val = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
            sqrt_alpha_bar = torch.sqrt(alpha_bar)

            # Sample added noise epsilon_k
            if self.training:
                eps_k = torch.randn_like(z0)
            else:
                # Deterministic noise perturbation during evaluation for reproducible inference
                gen = torch.Generator(device=z0.device).manual_seed(t_clamped + 42)
                eps_k = torch.randn(z0.shape, generator=gen, device=z0.device, dtype=z0.dtype)

            z_tk = sqrt_alpha_bar * z0 + sigma_val * eps_k

            t_tensor = torch.full((b_sz,), t_clamped, device=device, dtype=torch.long)
            with torch.no_grad():
                diffuser_kwargs: Dict[str, Any] = {}
                diff_dtype = next(self.diffuser.parameters()).dtype if list(self.diffuser.parameters()) else dtype
                if uncond_emb is not None:
                    diffuser_kwargs["encoder_hidden_states"] = uncond_emb.to(diff_dtype)
                z_tk_input = z_tk.to(diff_dtype)
                pred_eps = self.diffuser(z_tk_input, t_tensor, **diffuser_kwargs).sample.detach().to(dtype)

            t_emb = sinusoidal_embedding(t_tensor.float(), dim=self.timestep_embed_dim)
            t_spatial = expand_to_spatial(t_emb, h_z, w_z).to(dtype=dtype)

            sigma_tensor = torch.full((b_sz,), sigma_val.item(), device=device, dtype=torch.float32)
            sigma_emb = sinusoidal_embedding(sigma_tensor, dim=self.sigma_embed_dim)
            sigma_spatial = expand_to_spatial(sigma_emb, h_z, w_z).to(dtype=dtype)

            if self.include_noisy_latents:
                z_components.append(z_tk if (self.training and not self.freeze_encoder) else z_tk.detach())
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

        # 4. Concatenate along channel dimension to form high-dimensional Z and normalize
        z_high_dim = torch.cat(z_components, dim=1)
        z_high_dim = self.z_norm(z_high_dim)

        # 5. Auxiliary classifier on Z representation if enabled
        class_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            class_logits = self.classifier_head(z_high_dim)

        # 6. Trainable Decoder: decodes Z, injecting perpendicular skips from the parallel frozen decoder
        mask_logits = self.decoder(z_high_dim, perp_skips=perp_skips, encoder_skips=enc_skips)

        # Crop back to original dimensions if padded and ensure contiguous layout
        if mask_logits.shape[2] != orig_h or mask_logits.shape[3] != orig_w:
            mask_logits = mask_logits[:, :, :orig_h, :orig_w].contiguous()
        else:
            mask_logits = mask_logits.contiguous()

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
    ) -> Union[DiffusionDiffV2Model, Tuple[DiffusionDiffV2Model, Any]]:
        """Load DiffusionDiffV2Model from checkpoint file (.pt)."""
        from sid_unet.models.unet import UNet
        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )


# Convenient aliases
DiffusionDiffV2 = DiffusionDiffV2Model
DiffuserNoiseDecoderV2 = DiffusionDiffV2Model
DiffusionMultiNoiseDecoderV2 = DiffusionDiffV2Model
