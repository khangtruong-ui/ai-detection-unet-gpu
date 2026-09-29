"""
Diffusion-Diff-Minimized: Ultra-Fast Latent Diffusion Tampering Forensic Model.

Key Architectural & Compute Optimizations:
1. Base Pretrained Model: Segmind Tiny-SD (segmind/tiny-sd) instead of SD 1.5,
   loaded via:
   DiffusionPipeline.from_pretrained("segmind/tiny-sd", dtype=torch.float16, device_map="cuda")
   yielding a compact ~400M parameter UNet with 3 stages (320, 640, 1280) vs SD 1.5's ~860M parameter UNet.
2. 2 Lines of Computation (vs 4 in Diffusion-Diff-V2):
   - Line 1: From the real image x: passes through frozen VAE encoder to obtain clean latent z0.
             Parallel frozen VAE decoder decodes z0 to provide multi-scale perpendicular skip features.
   - Line 2: Pick ONE noisy latent from the diffusion model: single chosen timestep t (default: 250),
             adding noise eps, predicting diffuser noise eps_hat, computing noise discrepancy (eps_hat - eps),
             and sinusoidal embeddings of timestep and noise deviation sigma.
3. High-Dimensional Representation Z:
   Formed by concatenating clean latent z0 (1 real) and the single chosen noisy latent representation (1 noisy),
   reducing Z channels from 244 in v2 down to 84 in minimized, slashing decoder compute and VRAM overhead.
4. Frozen Components:
   - Diffuser UNet is strictly frozen.
   - Parallel VAE decoder is strictly frozen.
   - VAE encoder is frozen by default (freeze_encoder=True).
5. Trainable Latent Decoder:
   Decodes representation Z back to full-resolution mask logits, fusing perpendicular skip connections
   from the parallel frozen decoder.
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
    DecoderUpsampleStage,
    ResidualConvBlock,
    expand_to_spatial,
    get_activation,
    get_norm_layer,
    sinusoidal_embedding,
)
from sid_unet.models.diffusion_diff_v2 import (
    PerpendicularSkipFusion,
    EncoderSkipFusion,
    TrainableLatentDecoderV2,
)

logger = logging.getLogger("sid_unet.models.diffusion_diff_minimized")

DEFAULT_DIFFUSION_CHECKPOINT = "segmind/tiny-sd"
TrainableLatentDecoderMinimized = TrainableLatentDecoderV2


class DiffusionDiffMinimizedModel(nn.Module):
    """
    Diffusion Multi-Noise Latent Feature Decoder Minimized (Diffusion-Diff-Minimized) Model.

    Designed for high-throughput, low-latency synthetic image forensics:
      1. Default diffusion model: `segmind/tiny-sd` loaded with half precision and GPU mapping:
         `DiffusionPipeline.from_pretrained("segmind/tiny-sd", dtype=torch.float16, device_map="cuda")`
      2. 2 Lines of Computation:
         - Line 1: Real clean latent z0 extracted from real image x via frozen VAE encoder.
         - Line 2: Pick ONE noisy latent z_t from the diffusion model at chosen timestep (default: 250).
      3. Parallel Pretrained Frozen Decoder: runs on z0 to extract intermediate features injected
         perpendicularly into the trainable decoder stages.
      4. Frozen Components: Diffuser UNet and parallel VAE decoder are strictly frozen. VAE encoder
         is frozen by default.
      5. Trainable Decoder: maps high-dimensional Z (84 channels by default) to binary mask logits.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = DEFAULT_DIFFUSION_CHECKPOINT,
        pipeline: Optional[Any] = None,
        vae_subfolder: Optional[str] = "vae",
        unet_subfolder: Optional[str] = "unet",
        timestep: Optional[int] = None,
        timesteps: Optional[Union[int, List[int]]] = None,
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
        device_map: Optional[str] = "cuda",
        enforce_single_noisy: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.vae_subfolder = vae_subfolder
        self.unet_subfolder = unet_subfolder
        self.diffuser_fp16 = bool(kwargs.get("diffuser_fp16", diffuser_fp16))
        self.device_map = device_map

        # Handle freeze_encoder / autoencoder_trainable configurations
        if "autoencoder_trainable" in kwargs:
            self.freeze_encoder = not bool(kwargs["autoencoder_trainable"])
        elif "trainable_autoencoder" in kwargs:
            self.freeze_encoder = not bool(kwargs["trainable_autoencoder"])
        else:
            self.freeze_encoder = bool(freeze_encoder)

        # Handle encoder skips (defaults to False in minimized)
        if "use_skip_connections" in kwargs:
            self.use_encoder_skips = bool(kwargs["use_skip_connections"])
        elif "skip_connections" in kwargs:
            self.use_encoder_skips = bool(kwargs["skip_connections"])
        else:
            self.use_encoder_skips = bool(use_encoder_skips)

        # Perpendicular skips from parallel frozen decoder (defaults to True)
        self.use_perpendicular_skips = bool(
            kwargs.get("perpendicular_skips", use_perpendicular_skips)
        )

        # 2 lines of computation: 1 real latent + 1 chosen noisy latent from the diffusion model
        if timestep is not None:
            self.timesteps = [int(timestep)]
        elif timesteps is not None:
            if isinstance(timesteps, int):
                self.timesteps = [int(timesteps)]
            elif len(timesteps) == 1:
                self.timesteps = [int(timesteps[0])]
            elif enforce_single_noisy:
                # Pick one representative noisy latent from the provided timesteps (middle index)
                chosen_idx = len(timesteps) // 2
                chosen_t = int(timesteps[chosen_idx])
                logger.info(
                    f"DiffusionDiffMinimizedModel enforces 2 lines of computation (1 real, 1 noisy). "
                    f"Picked noisy latent at timestep {chosen_t} from provided timesteps {timesteps}."
                )
                self.timesteps = [chosen_t]
            else:
                self.timesteps = [int(t) for t in timesteps]
        else:
            # Default to single chosen noisy latent at timestep 250
            self.timesteps = [250]

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

        # 1. Initialize Pipeline (or VAE & Diffuser UNet)
        self._init_pipeline_or_components(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pipeline=pipeline,
            subfolder_vae=vae_subfolder,
            subfolder_unet=unet_subfolder,
            use_dummy=use_dummy,
            dummy_vae_channels=dummy_vae_channels,
            dummy_unet_channels=dummy_unet_channels,
            in_channels=in_channels,
            device_map=device_map,
        )

        # 2. Setup Noise Schedule (alphas and sigmas)
        self._setup_noise_schedule()

        # 3. Calculate total channel dimension of concatenated high-dimensional Z (84 channels by default)
        self.total_z_channels = self._calculate_z_channels()
        logger.info(
            f"DiffusionDiffMinimizedModel representation Z channels: {self.total_z_channels} "
            f"(2 lines of computation: 1 real latent, {len(self.timesteps)} noisy latent)"
        )

        # 4. Determine layer dimensions for perpendicular skip connections
        dec_cfg = dict(decoder_config or {})
        dec_channels = dec_cfg.get("channels", [256, 128, 64, 32])
        dec_upsample_mode = dec_cfg.get("upsample_mode", "bilinear")
        dec_norm = dec_cfg.get("norm_layer", "batchnorm")
        dec_act = dec_cfg.get("activation", "silu")
        dec_dropout = float(dec_cfg.get("dropout", 0.1))
        dec_res_blocks = int(dec_cfg.get("num_res_blocks", 1))

        # Normalization layer for concatenated high-dimensional Z representation
        z_norm_type = dec_cfg.get("z_norm", "groupnorm")
        self.z_norm = get_norm_layer(z_norm_type, self.total_z_channels)

        # Determine reference device and dtype from VAE
        ref_device = next(self.vae.parameters()).device
        ref_dtype = next(self.vae.parameters()).dtype

        self.perp_mapping: List[Optional[int]] = []
        perp_skip_channels: Optional[List[int]] = None
        if self.use_perpendicular_skips:
            with torch.no_grad():
                dummy_z = torch.zeros(
                    (1, self.latent_channels, 32, 32),
                    device=ref_device,
                    dtype=ref_dtype,
                )
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
                dummy_x = torch.zeros(
                    (1, in_channels, 64, 64),
                    device=ref_device,
                    dtype=ref_dtype,
                )
                _, dummy_skips = self._encode_with_skips(dummy_x)
                encoder_skip_channels = [s.shape[1] for s in dummy_skips]
        else:
            encoder_skip_channels = None

        # 5. Initialize Trainable Latent Decoder
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

        # 6. Auxiliary classifier head on Z representation
        if self.aux_classifier:
            self.classifier_head = AuxiliaryClassifier(
                in_channels=self.total_z_channels,
                num_classes=num_classes,
                dropout=dec_dropout,
            )
        else:
            self.classifier_head = None

        # Synchronize devices if VAE was loaded directly onto a specific device (e.g. CUDA)
        if ref_device.type != "cpu":
            self.decoder.to(ref_device)
            self.z_norm.to(ref_device)
            if self.classifier_head is not None:
                self.classifier_head.to(ref_device)

        # 7. Strictly enforce freezing according to specifications
        self._enforce_freeze()

    @staticmethod
    def _match_stages(
        frozen_shapes: List[Tuple[int, int, int]],
        trainable_stage_shapes: List[Tuple[int, int, int]],
    ) -> List[Optional[int]]:
        """Match trainable decoder stages to frozen decoder layers by resolution proximity."""
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

    def _init_pipeline_or_components(
        self,
        pretrained_model_name_or_path: str,
        pipeline: Optional[Any],
        subfolder_vae: Optional[str],
        subfolder_unet: Optional[str],
        use_dummy: bool,
        dummy_vae_channels: Tuple[int, ...],
        dummy_unet_channels: Tuple[int, ...],
        in_channels: int,
        device_map: Optional[str],
    ) -> None:
        """
        Initialize VAE and Diffuser components via DiffusionPipeline:
        DiffusionPipeline.from_pretrained("segmind/tiny-sd", dtype=torch.float16, device_map="cuda")
        with robust fallbacks for dummy mode and offline/cpu environments.
        """
        from diffusers import AutoencoderKL, UNet2DConditionModel

        if pipeline is not None:
            logger.info("Using user-provided DiffusionPipeline...")
            self.vae = pipeline.vae
            self.diffuser = pipeline.unet
            self.latent_channels = getattr(self.vae.config, "latent_channels", 4)
            return

        if use_dummy:
            logger.info("Initializing lightweight dummy AutoencoderKL & UNet2DConditionModel...")
            v_blocks = list(dummy_vae_channels)
            self.vae = AutoencoderKL(
                in_channels=in_channels,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D"] * len(v_blocks),
                up_block_types=["UpDecoderBlock2D"] * len(v_blocks),
                block_out_channels=v_blocks,
                latent_channels=4,
                layers_per_block=1,
            )
            u_blocks = list(dummy_unet_channels)
            self.diffuser = UNet2DConditionModel(
                sample_size=32,
                in_channels=4,
                out_channels=4,
                layers_per_block=1,
                block_out_channels=tuple(u_blocks),
                down_block_types=tuple(["DownBlock2D"] * len(u_blocks)),
                up_block_types=tuple(["UpBlock2D"] * len(u_blocks)),
                cross_attention_dim=32,
            )
            self.latent_channels = 4
            return

        # Attempt loading via DiffusionPipeline as requested
        loaded_via_pipe = False
        try:
            from diffusers import DiffusionPipeline

            load_kwargs: Dict[str, Any] = {}
            if self.diffuser_fp16 and torch.cuda.is_available():
                load_kwargs["dtype"] = torch.float16
                if device_map:
                    load_kwargs["device_map"] = device_map
            elif self.diffuser_fp16:
                load_kwargs["dtype"] = torch.float32

            try:
                pipe = DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path, **load_kwargs
                )
            except TypeError:
                # Handle diffusers versions expecting torch_dtype
                if "dtype" in load_kwargs:
                    load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
                pipe = DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path, **load_kwargs
                )

            self.vae = pipe.vae
            self.diffuser = pipe.unet
            loaded_via_pipe = True
            logger.info(
                f"Successfully loaded {pretrained_model_name_or_path} via DiffusionPipeline"
            )
        except Exception as e:
            logger.warning(
                f"Could not load via DiffusionPipeline ({e}). Attempting subfolder loading..."
            )

        if not loaded_via_pipe:
            # Fallback to direct subfolder loading
            try:
                vae_kwargs: Dict[str, Any] = {}
                if subfolder_vae:
                    vae_kwargs["subfolder"] = subfolder_vae
                self.vae = AutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path, **vae_kwargs
                )
            except Exception as e:
                logger.warning(
                    f"Could not load AutoencoderKL ({e}). Falling back to dummy VAE..."
                )
                v_blocks = list(dummy_vae_channels)
                self.vae = AutoencoderKL(
                    in_channels=in_channels,
                    out_channels=3,
                    down_block_types=["DownEncoderBlock2D"] * len(v_blocks),
                    up_block_types=["UpDecoderBlock2D"] * len(v_blocks),
                    block_out_channels=v_blocks,
                    latent_channels=4,
                    layers_per_block=1,
                )

            try:
                unet_kwargs: Dict[str, Any] = {}
                if subfolder_unet:
                    unet_kwargs["subfolder"] = subfolder_unet
                if self.diffuser_fp16 and torch.cuda.is_available():
                    unet_kwargs["torch_dtype"] = torch.float16
                self.diffuser = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path, **unet_kwargs
                )
            except Exception as e:
                logger.warning(
                    f"Could not load UNet2DConditionModel ({e}). Falling back to dummy UNet..."
                )
                u_blocks = list(dummy_unet_channels)
                self.diffuser = UNet2DConditionModel(
                    sample_size=32,
                    in_channels=4,
                    out_channels=4,
                    layers_per_block=1,
                    block_out_channels=tuple(u_blocks),
                    down_block_types=tuple(["DownBlock2D"] * len(u_blocks)),
                    up_block_types=tuple(["UpBlock2D"] * len(u_blocks)),
                    cross_attention_dim=32,
                )

        self.latent_channels = getattr(self.vae.config, "latent_channels", 4)

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

    def to(self, *args: Any, **kwargs: Any) -> "DiffusionDiffMinimizedModel":
        """Move model to device and cast frozen diffuser to half precision if on CUDA."""
        res = super().to(*args, **kwargs)
        if getattr(self, "diffuser_fp16", False):
            try:
                device = next(self.parameters()).device
                if (
                    device.type == "cuda"
                    and hasattr(self, "diffuser")
                    and self.diffuser is not None
                ):
                    self.diffuser = self.diffuser.to(device=device, dtype=torch.float16)
            except Exception:
                pass
        return res

    def _extract_frozen_decoder_features(self, z0: torch.Tensor) -> List[torch.Tensor]:
        """
        Run parallel pretrained, frozen decoder on diffusion latent z0 and extract
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
        grad_ctx = (
            torch.enable_grad()
            if (self.training and not self.freeze_encoder)
            else torch.no_grad()
        )
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
        Forward pass for Diffusion-Diff-Minimized.

        Executes strictly 2 lines of computation:
          1. Line 1: From the real image:
             Real image x is encoded into latent z0 via frozen VAE encoder.
             Parallel frozen VAE decoder decodes z0 to extract perpendicular skip features.
          2. Line 2: Pick ONE noisy latent from the diffusion model:
             z0 is perturbed at chosen timestep t (default: 250) with noise eps_k -> z_tk.
             Frozen diffuser UNet predicts noise eps_hat.
             Noise difference and sinusoidal embeddings (t, sigma) are computed.
          3. High-dimensional representation Z is formed and decoded by TrainableLatentDecoderV2
             with perpendicular skip connection injection.
        """
        device = x.device
        dtype = x.dtype
        b_sz = x.shape[0]

        x_proc, orig_h, orig_w = self._preprocess_input(x)

        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device

        # Match device and dtype for VAE if needed
        x_vae = x_proc.to(device=vae_device, dtype=vae_dtype)

        # -------------------------------------------------------------
        # Line 1: From the real image (clean latent z0 & perpendicular skips)
        # -------------------------------------------------------------
        if self.use_encoder_skips:
            posterior, enc_skips = self._encode_with_skips(x_vae)
            if self.training and not self.freeze_encoder:
                z0 = posterior.mode() * self.scaling_factor
            else:
                with torch.no_grad():
                    z0 = posterior.mode() * self.scaling_factor
            enc_skips = [
                s.to(device=device, dtype=dtype) if s is not None else None
                for s in enc_skips
            ]
        else:
            enc_skips = None
            if self.training and not self.freeze_encoder:
                posterior = self.vae.encode(x_vae).latent_dist
                z0 = posterior.mode() * self.scaling_factor
            else:
                with torch.no_grad():
                    posterior = self.vae.encode(x_vae).latent_dist
                    z0 = posterior.mode() * self.scaling_factor

        # Prevent runaway latent drift
        z0 = torch.clamp(z0, min=-15.0, max=15.0)

        # Extract perpendicular skip features from parallel frozen decoder on clean latent z0
        perp_skips: Optional[List[torch.Tensor]] = None
        if self.use_perpendicular_skips:
            with torch.no_grad():
                raw_frozen_feats = self._extract_frozen_decoder_features(z0.detach())
                perp_skips = [
                    raw_frozen_feats[m].detach().to(device=device, dtype=dtype)
                    if (m is not None and m < len(raw_frozen_feats))
                    else None
                    for m in self.perp_mapping
                ]

        z0_out = z0.to(device=device, dtype=dtype)
        _, _, h_z, w_z = z0_out.shape

        z_components: List[torch.Tensor] = []
        if self.include_z0:
            z_components.append(
                z0_out if (self.training and not self.freeze_encoder) else z0_out.detach()
            )

        # -------------------------------------------------------------
        # Line 2: Pick ONE noisy latent from the diffusion model
        # -------------------------------------------------------------
        diff_dtype = (
            next(self.diffuser.parameters()).dtype
            if list(self.diffuser.parameters())
            else dtype
        )
        diff_device = (
            next(self.diffuser.parameters()).device
            if list(self.diffuser.parameters())
            else device
        )

        cross_dim = getattr(self.diffuser.config, "cross_attention_dim", None)
        if cross_dim is not None:
            uncond_emb = torch.zeros(
                (b_sz, 1, cross_dim), device=diff_device, dtype=diff_dtype
            )
        else:
            uncond_emb = None

        for t_val in self.timesteps:
            t_clamped = max(0, min(t_val, len(self.alphas_cumprod) - 1))
            alpha_bar = self.alphas_cumprod[t_clamped].to(device=device, dtype=dtype)
            sigma_val = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
            sqrt_alpha_bar = torch.sqrt(alpha_bar)

            # Sample added noise epsilon_k
            if self.training:
                eps_k = torch.randn_like(z0_out)
            else:
                # Deterministic noise perturbation during evaluation for reproducible inference
                gen = torch.Generator(device=z0_out.device).manual_seed(t_clamped + 42)
                eps_k = torch.randn(
                    z0_out.shape, generator=gen, device=z0_out.device, dtype=z0_out.dtype
                )

            z_tk = sqrt_alpha_bar * z0_out + sigma_val * eps_k

            t_tensor = torch.full((b_sz,), t_clamped, device=diff_device, dtype=torch.long)
            with torch.no_grad():
                diffuser_kwargs: Dict[str, Any] = {}
                if uncond_emb is not None:
                    diffuser_kwargs["encoder_hidden_states"] = uncond_emb
                z_tk_input = z_tk.to(device=diff_device, dtype=diff_dtype)
                pred_eps = (
                    self.diffuser(z_tk_input, t_tensor, **diffuser_kwargs)
                    .sample.detach()
                    .to(device=device, dtype=dtype)
                )

            t_tensor_dev = torch.full((b_sz,), t_clamped, device=device, dtype=torch.long)
            t_emb = sinusoidal_embedding(t_tensor_dev.float(), dim=self.timestep_embed_dim)
            t_spatial = expand_to_spatial(t_emb, h_z, w_z).to(dtype=dtype)

            sigma_tensor = torch.full(
                (b_sz,), sigma_val.item(), device=device, dtype=torch.float32
            )
            sigma_emb = sinusoidal_embedding(sigma_tensor, dim=self.sigma_embed_dim)
            sigma_spatial = expand_to_spatial(sigma_emb, h_z, w_z).to(dtype=dtype)

            if self.include_noisy_latents:
                z_components.append(
                    z_tk if (self.training and not self.freeze_encoder) else z_tk.detach()
                )
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

        # -------------------------------------------------------------
        # High-Dimensional Representation Z Concatenation & Trainable Decoding
        # -------------------------------------------------------------
        z_high_dim = torch.cat(z_components, dim=1)
        norm_dtype = (
            next(self.z_norm.parameters()).dtype
            if list(self.z_norm.parameters())
            else dtype
        )
        norm_device = (
            next(self.z_norm.parameters()).device
            if list(self.z_norm.parameters())
            else device
        )
        z_high_dim = z_high_dim.to(device=norm_device, dtype=norm_dtype)
        z_high_dim = self.z_norm(z_high_dim)

        class_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            class_logits = self.classifier_head(z_high_dim)

        mask_logits = self.decoder(
            z_high_dim, perp_skips=perp_skips, encoder_skips=enc_skips
        )

        # Crop back to original dimensions if padded and ensure contiguous layout
        if mask_logits.shape[2] != orig_h or mask_logits.shape[3] != orig_w:
            mask_logits = mask_logits[:, :, :orig_h, :orig_w].contiguous()
        else:
            mask_logits = mask_logits.contiguous()

        # Match return dtypes and devices to input tensor
        if mask_logits.device != device or mask_logits.dtype != dtype:
            mask_logits = mask_logits.to(device=device, dtype=dtype)
        if class_logits is not None and (
            class_logits.device != device or class_logits.dtype != dtype
        ):
            class_logits = class_logits.to(device=device, dtype=dtype)

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
    ) -> Union[DiffusionDiffMinimizedModel, Tuple[DiffusionDiffMinimizedModel, Any]]:
        """Load DiffusionDiffMinimizedModel from checkpoint file (.pt)."""
        from sid_unet.models.unet import UNet

        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args: Any,
        **kwargs: Any,
    ):
        """Load DiffusionDiffMinimizedModel from HF repository or local checkpoint."""
        from sid_unet.models.unet import UNet

        return UNet.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


# Convenient aliases
DiffusionDiffMinimized = DiffusionDiffMinimizedModel
DiffusionDiffMin = DiffusionDiffMinimizedModel
DiffusionMinimizedModel = DiffusionDiffMinimizedModel
