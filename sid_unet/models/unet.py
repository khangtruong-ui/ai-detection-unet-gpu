"""
UNet architecture with optional 3-class auxiliary classification head for AI image mask detection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from sid_unet.models.blocks import AuxiliaryClassifier, DoubleConv, Down, OutConv, Up

logger = logging.getLogger(__name__)


def _safe_checkpoint(func, *args):
    """Run function with torch activation checkpointing safely across PyTorch versions."""
    try:
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)
    except TypeError:
        return torch.utils.checkpoint.checkpoint(func, *args)


def _run_module(module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    return module(*inputs)


class UNet(nn.Module):
    """
    Standard UNet architecture with modular depth, configurable channel dimensions,
    bilinear/transposed upsampling, optional gradient checkpointing, and an optional 3-class auxiliary classification head.

    Args:
        in_channels (int): Input image channels (default: 3 for RGB).
        out_channels (int): Mask output channels (default: 1 for binary mask logits).
        features (List[int]): Number of feature channels at each encoder level.
        bilinear (bool): Whether to use bilinear interpolation for upsampling.
        dropout (float): Dropout probability.
        aux_classifier (bool): Whether to enable auxiliary 3-class classification head.
        num_classes (int): Number of target classes for auxiliary classification (default: 3).
        gradient_checkpointing (bool): Whether to enable activation checkpointing to save VRAM.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        bilinear: bool = True,
        dropout: float = 0.0,
        aux_classifier: bool = True,
        num_classes: int = 3,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.aux_classifier = aux_classifier
        self.gradient_checkpointing = gradient_checkpointing

        if features is None:
            features = [64, 128, 256, 512]
        self.features = [int(f) for f in features]

        # Initial convolution
        self.inc = DoubleConv(in_channels, self.features[0], dropout=dropout)

        # Encoder (Downsampling)
        self.downs = nn.ModuleList()
        for i in range(len(self.features) - 1):
            self.downs.append(Down(self.features[i], self.features[i + 1], dropout=dropout))

        # Bottleneck
        factor = 2 if bilinear else 1
        bottleneck_channels = self.features[-1] * 2 // factor
        self.bottleneck = Down(self.features[-1], bottleneck_channels, dropout=dropout)

        # Auxiliary classifier head on bottleneck features
        if self.aux_classifier:
            self.classifier_head = AuxiliaryClassifier(
                in_channels=bottleneck_channels,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            self.classifier_head = None

        # Decoder (Upsampling)
        self.ups = nn.ModuleList()
        curr_channels = bottleneck_channels
        rev_features = list(reversed(self.features))

        for i, skip_feat in enumerate(rev_features):
            is_last = (i == len(rev_features) - 1)
            next_out = self.features[0] if is_last else skip_feat // factor
            self.ups.append(
                Up(
                    in_channels_down=curr_channels,
                    in_channels_skip=skip_feat,
                    out_channels=next_out,
                    bilinear=bilinear,
                    dropout=dropout,
                )
            )
            curr_channels = next_out

        # Final 1x1 Convolution to binary mask logits
        self.outc = OutConv(self.features[0], out_channels)

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        """Dynamically enable or disable gradient checkpointing."""
        self.gradient_checkpointing = bool(enable)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Returns:
            If aux_classifier is True: (mask_logits, class_logits)
            If aux_classifier is False: mask_logits
        """
        use_ckpt = self.gradient_checkpointing and self.training and x.requires_grad

        # Encoder
        x1 = self.inc(x)
        encoder_skips = [x1]

        curr = x1
        for down_block in self.downs:
            if use_ckpt:
                curr = _safe_checkpoint(_run_module, down_block, curr)
            else:
                curr = down_block(curr)
            encoder_skips.append(curr)

        # Bottleneck
        if use_ckpt:
            bottleneck_feat = _safe_checkpoint(_run_module, self.bottleneck, curr)
        else:
            bottleneck_feat = self.bottleneck(curr)

        # Auxiliary Classification
        class_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            class_logits = self.classifier_head(bottleneck_feat)

        # Decoder
        d_curr = bottleneck_feat
        for i, up_block in enumerate(self.ups):
            skip = encoder_skips[-(i + 1)]
            if use_ckpt:
                d_curr = _safe_checkpoint(_run_module, up_block, d_curr, skip)
            else:
                d_curr = up_block(d_curr, skip)

        mask_logits = self.outc(d_curr)

        if self.aux_classifier:
            return mask_logits, class_logits
        return mask_logits

    @torch.no_grad()
    def predict_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Produce binary mask prediction (values 0.0 or 1.0) given input image tensor.
        """
        self.eval()
        outputs = self.forward(x)
        mask_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.sigmoid(mask_logits)
        return (probs >= threshold).float()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        device: Optional[Union[str, torch.device]] = None,
        override_config: Optional[Union[Dict[str, Any], Any]] = None,
        strict: Optional[bool] = None,
        return_config: bool = False,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> Union[UNet, Tuple[UNet, Any]]:
        """Hugging Face style from_pretrained loader for UNet and derivative architectures.

        Loads a checkpoint from local file path or Hugging Face repository ID/URI,
        automatically downloading into the HF cache directory ($HF_HOME/hub).
        """
        return cls.from_checkpoint(
            checkpoint_path=pretrained_model_name_or_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        override_config: Optional[Union[Dict[str, Any], Any]] = None,
        strict: Optional[bool] = None,
        return_config: bool = False,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> Union[UNet, Tuple[UNet, Any]]:
        """
        Load a trained UNet model from a checkpoint file (.pt).
        Automatically restores the model architecture hyperparameters
        saved within the checkpoint's embedded configuration.

        Args:
            checkpoint_path: Path to checkpoint .pt file or Hugging Face repository identifier/URI.
            device: Target device to move the model to (e.g. 'cpu', 'cuda', 'auto', or torch.device).
            override_config: Optional config dict or ConfigDict to override embedded config.
            strict: Whether to strictly enforce that the keys in state_dict match model keys.
            return_config: If True, returns a tuple (model, config_dict).
            cache_dir: Optional custom Hugging Face cache directory ($HF_HOME/hub).
            token: Optional Hugging Face authentication token.
            force_download: Whether to force re-download even if already cached.

        Returns:
            UNet instance (eval mode by default), or (UNet, ConfigDict) if return_config=True.
        """
        import os
        from sid_unet.utils.checkpoint import resolve_checkpoint_source

        resolved_path = resolve_checkpoint_source(
            checkpoint_path,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path} (resolved to {resolved_path})")
        checkpoint_path = resolved_path

        target_map = device if device not in (None, "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
        import warnings
        ckpt = None
        for wo in [True, False]:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
                    ckpt = torch.load(checkpoint_path, map_location=target_map, weights_only=wo)
                break
            except Exception:
                continue
        if ckpt is None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            saved_cfg = ckpt.get("config", {})
        elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
            saved_cfg = ckpt.get("config", {})
        elif isinstance(ckpt, dict) and (
            any(
                k.startswith((
                    "inc.", "downs.", "bottleneck.", "stage1.", "linear.", "stage5.",
                    "aggregator.", "prompt_learner.", "cls_aggregator.",
                    "net.", "module.net.", "backbone_swin.", "module.backbone_swin.",
                    "uperhead.", "module.uperhead.", "fc.", "module.fc.",
                    "backbone_net.", "mscc.", "bottleneck_fusion.", "classifer.",
                    "backbone.trunk.", "neck.module.", "head.decoder_modules.",
                    "expert_sparsevit.", "expert_trufor.", "expert_diffusionprint.", "gate.logit_gamma",
                    "encoder_net.", "lmu.", "dncnn.", "decode_head.",
                ))
                for k in ckpt.keys()
            ) or any("backbone_swin" in k or "image_encoder" in k for k in ckpt.keys())
        ):
            state_dict = ckpt
            saved_cfg = {}
        elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
            saved_cfg = ckpt.get("config", {})
        else:
            raise ValueError(f"Unrecognized checkpoint format in '{checkpoint_path}'")

        from sid_unet.utils.config import DEFAULT_CONFIG, ConfigDict, deep_merge
        if isinstance(saved_cfg, ConfigDict):
            saved_cfg_dict = saved_cfg.to_dict()
        elif isinstance(saved_cfg, dict):
            saved_cfg_dict = saved_cfg
        else:
            saved_cfg_dict = {}

        merged = deep_merge(DEFAULT_CONFIG, saved_cfg_dict)
        if isinstance(state_dict, dict):
            has_aux = any(k.startswith(("classifier_head.", "cls_aggregator.", "prompt_learner.")) for k in state_dict.keys())
            if not saved_cfg_dict or "aux_classifier" not in saved_cfg_dict.get("model", {}):
                merged["model"]["aux_classifier"] = has_aux
            saved_model_name = saved_cfg_dict.get("model", {}).get("name")
            if saved_model_name:
                merged["model"]["name"] = saved_model_name
                if saved_model_name == "efficientnet":
                    merged["model"]["sacrifice_of_pixel"] = any(k.startswith("linear.") for k in state_dict.keys())
            elif any(k.startswith(("aggregator.vit_model.", "prompt_learner.", "cls_aggregator.")) for k in state_dict.keys()):
                merged["model"]["name"] = "mask_clip"
                if "num_classes" not in saved_cfg_dict.get("model", {}):
                    merged["model"]["num_classes"] = 3 if any(k.startswith("class_head.") for k in state_dict.keys()) else 2
            elif any(k.startswith(("stage1.", "linear.")) for k in state_dict.keys()):
                merged["model"]["name"] = "efficientnet"
                merged["model"]["sacrifice_of_pixel"] = any(k.startswith("linear.") for k in state_dict.keys())
            elif any(k.startswith(("paired_artifact_encoder.", "zero_gated_film.")) for k in state_dict.keys()):
                merged["model"]["name"] = "gap_sam"
            elif any(k.startswith(("base_model.backbone.vision_backbone", "peft_model.base_model", "model.backbone.vision_backbone")) for k in state_dict.keys()):
                merged["model"]["name"] = "sam3_distil"
            elif any(k.startswith(("feature_adapters.", "mask_adapter.refine_conv", "perturbation.")) for k in state_dict.keys()):
                merged["model"]["name"] = "detective_sam"
            elif any("vision_encoder.backbone.blocks." in k for k in state_dict.keys()) or any("base_box" in k for k in state_dict.keys()):
                merged["model"]["name"] = "sam2"
            elif any(k.startswith(("base_model.", "model.vision_encoder", "model.detr_encoder", "model.mask_decoder")) for k in state_dict.keys()):
                merged["model"]["name"] = "sam3_qlora"
            elif any(k.startswith(("mscc.", "bottleneck_fusion.", "module.mscc.", "backbone_net.")) for k in state_dict.keys()):
                merged["model"]["name"] = "msccnet"
                merged["model"]["pretrained"] = False
                merged["model"]["backbone_pretrained"] = False
                merged["model"]["allow_random_init"] = True
                merged["model"]["aux_classifier"] = any(k.startswith(("classifer.", "module.classifer.")) for k in state_dict.keys())
                if any("backbone_net.layer3.22." in k for k in state_dict.keys()):
                    merged["model"].setdefault("backbone_depth", 101)
                elif any("backbone_net.layer3.5." in k for k in state_dict.keys()):
                    merged["model"].setdefault("backbone_depth", 50)
                if "classifer.fc.weight" in state_dict and hasattr(state_dict["classifer.fc.weight"], "shape"):
                    merged["model"].setdefault("num_classes", int(state_dict["classifer.fc.weight"].shape[0]))
                if "mscc.fuse.weight" in state_dict and hasattr(state_dict["mscc.fuse.weight"], "shape"):
                    n_groups = int(state_dict["mscc.fuse.weight"].shape[1])
                    k_val = int(round(n_groups ** 0.5))
                    if k_val * k_val == n_groups:
                        merged["model"].setdefault("spectral_k", k_val)
            elif any(k.startswith(("backbone.trunk.", "neck.module.", "head.decoder_modules.")) for k in state_dict.keys()):
                merged["model"]["name"] = "omnifd"
                merged["model"]["pretrained"] = False
                merged["model"]["backbone_pretrained"] = False
                merged["model"]["allow_random_init"] = True
                if any("backbone.trunk.layers.2.blocks.6." in k for k in state_dict.keys()):
                    merged["model"].setdefault("variant", "swin-s")
                else:
                    merged["model"].setdefault("variant", "swin-t")
                merged["model"]["has_image_head"] = any("head.decoder_modules.image." in k for k in state_dict.keys())
                merged["model"]["has_spatial_head"] = any("head.decoder_modules.spatial." in k for k in state_dict.keys())
                merged["model"]["aux_classifier"] = merged["model"]["has_image_head"]
            elif any(k.startswith("decoder.perp_fusions.") for k in state_dict.keys()):
                merged["model"]["name"] = "diffusion_diff_v2"
            elif any(k.startswith(("gate.fpn_out.", "gate.fpn_proj_32.", "gate.unc_gate.")) for k in state_dict.keys()):
                merged["model"]["name"] = "forensic_moe_v2"
                merged["model"]["pretrained"] = False
                merged["model"]["allow_random_init"] = True
            elif any(k.startswith(("expert_sparsevit.", "expert_trufor.", "expert_diffusionprint.", "gate.logit_gamma")) for k in state_dict.keys()):
                merged["model"]["name"] = "forensic_moe"
                merged["model"]["pretrained"] = False
                merged["model"]["allow_random_init"] = True
            elif any(k.startswith(("encoder_net.", "lmu.")) for k in state_dict.keys()):
                merged["model"]["name"] = "sparsevit"
                merged["model"]["pretrained"] = False
                merged["model"]["allow_random_init"] = True
            elif any(k.startswith(("dncnn.", "decode_head.", "decode_head_conf.")) for k in state_dict.keys()):
                merged["model"]["name"] = "trufor"
                merged["model"]["pretrained"] = False
                merged["model"]["allow_random_init"] = True
                if "np_adapter.weight" in state_dict and hasattr(state_dict["np_adapter.weight"], "shape"):
                    merged["model"]["np_out_channels"] = int(state_dict["np_adapter.weight"].shape[1])
                elif "dncnn.47.weight" in state_dict and hasattr(state_dict["dncnn.47.weight"], "shape"):
                    merged["model"]["np_out_channels"] = int(state_dict["dncnn.47.weight"].shape[0])
            elif any(k.startswith(("decoder.", "diffuser.")) for k in state_dict.keys()):
                merged["model"]["name"] = "diffusion_diff"
            elif any(k.startswith("vae.") for k in state_dict.keys()):
                merged["model"]["name"] = "vae_finetune"
            elif any(k.startswith(("noiseprint.", "conv_vit_x.", "conv_vit_n.")) for k in state_dict.keys()):
                merged["model"]["name"] = "noise-rag-prototype-v1"
            elif any(k.startswith(("pixel_module.", "transformer_decoder.")) for k in state_dict.keys()):
                merged["model"]["name"] = "iml"
            elif any("backbone_swin" in k or "uperhead" in k for k in state_dict.keys()):
                merged["model"]["name"] = "pim"
            elif any(
                k.startswith((
                    "net.stage1.", "net.stage2.", "net.stage3.", "net.stage4.", "net.image_encoder.",
                    "module.net.stage1.", "module.net.stage2.", "module.net.stage3.", "module.net.stage4.", "module.net.image_encoder."
                ))
                for k in state_dict.keys()
            ) or ("fc.weight" in state_dict or "module.fc.weight" in state_dict) or any("SAMBlock" in k or "HRNet" in k for k in state_dict.keys()):
                merged["model"]["name"] = "focal"
                if any("image_encoder" in k or "SAMBlock" in k for k in state_dict.keys()):
                    merged["model"]["backbone"] = "vit"
                else:
                    merged["model"]["backbone"] = "hrnet"
                if any(".lora_A" in k for k in state_dict.keys()):
                    merged["model"].setdefault("peft_enabled", True)
                    if not saved_cfg_dict.get("model", {}):
                        merged["model"]["peft_enabled"] = True
                        merged["model"]["use_dora"] = any(".dora_m" in k for k in state_dict.keys())
                        for k, v in state_dict.items():
                            if k.endswith(".lora_A") and hasattr(v, "shape"):
                                merged["model"]["lora_r"] = int(v.shape[0])
                                break
                        block_idxs = set()
                        targets = set()
                        for k in state_dict.keys():
                            if "blocks." in k and ".lora_A" in k:
                                parts = k.split(".")
                                for idx_i, part in enumerate(parts):
                                    if part == "blocks" and idx_i + 1 < len(parts) and parts[idx_i + 1].isdigit():
                                        block_idxs.add(int(parts[idx_i + 1]))
                                    if part in ("qkv", "proj", "lin1", "lin2"):
                                        targets.add(part)
                        if block_idxs:
                            merged["model"]["lora_block_scope"] = sorted(block_idxs)
                        if targets:
                            merged["model"]["lora_target_modules"] = sorted(targets)
                if any(k.startswith(("feature_adapter.", "module.feature_adapter.")) for k in state_dict.keys()):
                    merged["model"]["use_feature_adapter"] = True

        if override_config is not None:
            override_dict = override_config.to_dict() if isinstance(override_config, ConfigDict) else override_config
            merged = deep_merge(merged, override_dict)

        config = ConfigDict(merged)
        model = build_model(config)
        from sid_unet.training.callbacks import load_state_dict_compatible
        load_state_dict_compatible(model, state_dict, strict=strict, target_device=target_map)

        # Restore forensic vector memory if supported by the model architecture
        raw_model = getattr(model, "module", model)
        if hasattr(raw_model, "forensic_memory") and raw_model.forensic_memory is not None:
            loaded_mem = False
            # 1. From embedded checkpoint dictionary
            if isinstance(ckpt, dict) and "forensic_memory_state_dict" in ckpt and ckpt["forensic_memory_state_dict"] is not None:
                try:
                    raw_model.forensic_memory.load_state_dict(ckpt["forensic_memory_state_dict"])
                    logger.info(
                        f"Restored forensic vector memory ({raw_model.forensic_memory.entry_count} entries) "
                        f"from checkpoint dictionary."
                    )
                    loaded_mem = True
                except Exception as mem_err:
                    logger.warning(f"Could not restore forensic memory state dict from checkpoint ({mem_err}).")
            # 2. Or from sibling memory file in checkpoint directory
            if not loaded_mem:
                ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
                for cand in ["forensic_memory.pt", "forensic_memory_best.pt", "forensic_memory_latest.pt", "forensic_memory_periodic.pt"]:
                    cand_path = os.path.join(ckpt_dir, cand)
                    if os.path.isfile(cand_path):
                        try:
                            raw_model.forensic_memory.load(cand_path)
                            logger.info(
                                f"Loaded forensic vector memory ({raw_model.forensic_memory.entry_count} entries) "
                                f"from sibling file '{cand_path}'."
                            )
                            loaded_mem = True
                            break
                        except Exception as mem_err:
                            logger.warning(f"Could not load forensic memory from sibling file '{cand_path}': {mem_err}")
            # 3. Or from explicit configured memory_path
            if not loaded_mem and hasattr(raw_model, "memory_path") and raw_model.memory_path and os.path.isfile(raw_model.memory_path):
                try:
                    raw_model.forensic_memory.load(raw_model.memory_path)
                    logger.info(
                        f"Loaded forensic vector memory ({raw_model.forensic_memory.entry_count} entries) "
                        f"from memory_path '{raw_model.memory_path}'."
                    )
                    loaded_mem = True
                except Exception as mem_err:
                    logger.warning(f"Could not load forensic memory from memory_path '{raw_model.memory_path}': {mem_err}")

        # Restore vector DB retrieval for models with retrieval (e.g. NoiseRAGPrototypeV1)
        if hasattr(raw_model, "retrieval") and hasattr(raw_model.retrieval, "load_database") and not getattr(raw_model.retrieval, "has_database", False):
            loaded_db = False
            ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            for cand in ["mask_clip_db_mae.pt", "vector_db.pt", "forensic_memory.pt"]:
                cand_path = os.path.join(ckpt_dir, cand)
                if os.path.isfile(cand_path):
                    try:
                        raw_model.retrieval.load_database(cand_path)
                        logger.info(f"Loaded vector DB into retrieval module from sibling file '{cand_path}'.")
                        loaded_db = True
                        break
                    except Exception as db_err:
                        logger.warning(f"Could not load vector DB from sibling file '{cand_path}': {db_err}")
            if not loaded_db and hasattr(raw_model, "memory_path") and raw_model.memory_path and os.path.isfile(raw_model.memory_path):
                try:
                    raw_model.retrieval.load_database(raw_model.memory_path)
                    logger.info(f"Loaded vector DB into retrieval module from memory_path '{raw_model.memory_path}'.")
                except Exception as db_err:
                    logger.warning(f"Could not load vector DB from memory_path '{raw_model.memory_path}': {db_err}")

        model.eval()

        if device is not None:
            if isinstance(device, str):
                if device == "auto":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    device = torch.device(device)
            if not getattr(model, "load_in_4bit", False):
                model.to(device)

        model.config = config
        if return_config:
            return model, config
        return model


def build_model(config: Any) -> nn.Module:
    """Build model instance from configuration dict or model section."""
    if hasattr(config, "get") and "model" in config:
        model_cfg = config.get("model", {})
    elif hasattr(config, "model"):
        model_cfg = getattr(config, "model")
    elif isinstance(config, dict):
        model_cfg = config
    else:
        model_cfg = {}
    training_cfg = config.get("training", {}) if hasattr(config, "get") else {}
    loss_cfg = config.get("loss", {}) if hasattr(config, "get") else {}
    ckpt_flag = bool(
        model_cfg.get(
            "gradient_checkpointing",
            training_cfg.get("gradient_checkpointing", False) if hasattr(training_cfg, "get") else False,
        )
    )

    model_name = str(model_cfg.get("name", "unet")).lower()
    backbone = str(model_cfg.get("backbone", "")).lower()

    if any(k in model_name for k in ["noise-rag", "noise_rag", "noiserag"]) or any(
        k in backbone for k in ["noise-rag", "noise_rag", "noiserag"]
    ):
        from sid_unet.models.noise_rag import NoiseRAGPrototypeV1
        return NoiseRAGPrototypeV1(
            noiseprint_weights=model_cfg.get("noiseprint_weights", None),
            freeze_noiseprint=bool(model_cfg.get("freeze_noiseprint", True)),
            memory_path=model_cfg.get("memory_path", None),
            memory_dtype=str(model_cfg.get("memory_dtype", "float16")),
            embed_dim=int(model_cfg.get("embed_dim", 256)),
            conv_backbone=str(model_cfg.get("conv_backbone", "mobilenetv3_small_100")),
            num_vit_blocks=int(model_cfg.get("num_vit_blocks", 2)),
            num_cross_stages=int(model_cfg.get("num_cross_stages", 3)),
            out_stride=int(model_cfg.get("out_stride", 8)),
            retrieval_k=int(model_cfg.get("retrieval_k", 4)),
            align_loss_type=str(model_cfg.get("align_loss_type", "mse")),
            align_loss_weight=float(model_cfg.get("align_loss_weight", 0.2)),
            entropy_temperature=float(model_cfg.get("entropy_temperature", 1.0)),
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            cross_mode=str(model_cfg.get("cross_mode", "recurrent")),
        )

    if any(k in model_name for k in ["mask_clip", "mask-clip", "maskclip", "opensdi", "open_sdi", "sideclip"]) or any(
        k in backbone for k in ["mask_clip", "mask-clip", "maskclip", "opensdi"]
    ):
        from sid_unet.models.mask_clip import (
            MaskCLIP,
            DEFAULT_MASKCLIP_HF_REPO,
            DEFAULT_MASKCLIP_HF_FILENAME,
            DEFAULT_MAE_PRETRAIN_URL,
        )
        dev_cfg = (
            config.get("project", {}).get("device", "auto")
            if hasattr(config, "get") and hasattr(config.get("project", {}), "get")
            else "auto"
        )
        raw_fusion_map = model_cfg.get("fusion_map", None)
        parsed_fusion_map = (
            {int(k): int(v) for k, v in raw_fusion_map.items()}
            if isinstance(raw_fusion_map, dict)
            else None
        )
        return MaskCLIP(
            model_setting_name=str(model_cfg.get("model_setting_name", "ViTL")),
            pretrained=bool(model_cfg.get("pretrained", True)),
            pretrained_mode=str(model_cfg.get("pretrained_mode", "opensdi")),
            pretrained_model_name_or_path=str(
                model_cfg.get(
                    "pretrained_model_name_or_path",
                    model_cfg.get("checkpoint_path", DEFAULT_MASKCLIP_HF_REPO),
                )
            ),
            checkpoint_filename=model_cfg.get("checkpoint_filename", DEFAULT_MASKCLIP_HF_FILENAME),
            mae_pretrained_url=str(model_cfg.get("mae_pretrained_url", DEFAULT_MAE_PRETRAIN_URL)),
            clip_model_name=model_cfg.get("clip_model_name", None),
            allow_random_init=bool(model_cfg.get("allow_random_init", False)),
            resolution=int(model_cfg["resolution"]) if "resolution" in model_cfg and model_cfg["resolution"] is not None else None,
            selected_layers=model_cfg.get("selected_layers", None),
            fusion_map=parsed_fusion_map,
            language_ctx=int(model_cfg["language_ctx"]) if "language_ctx" in model_cfg and model_cfg["language_ctx"] is not None else None,
            language_depth=int(model_cfg["language_depth"]) if "language_depth" in model_cfg and model_cfg["language_depth"] is not None else None,
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            edge_lambda=float(model_cfg.get("edge_lambda", 20.0)),
            edge_mask_width=int(model_cfg.get("edge_mask_width", 7)),
            input_normalize=bool(model_cfg.get("input_normalize", True)),
            freeze_clip=bool(model_cfg.get("freeze_clip", True)),
            freeze_mae=bool(model_cfg.get("freeze_mae", False)),
            enable_clip_cache=bool(model_cfg.get("enable_clip_cache", False)),
            clip_cache_size=int(model_cfg.get("clip_cache_size", 10000)),
            lora_r=int(model_cfg.get("lora_r", 0)),
            lora_alpha=float(model_cfg.get("lora_alpha", 16.0)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            lora_target=str(model_cfg.get("lora_target", "mae")),
            use_dora=bool(model_cfg.get("use_dora", False)),
            cache_dir=model_cfg.get("cache_dir", None),
            token=model_cfg.get("token", None),
            force_download=bool(model_cfg.get("force_download", False)),
            device=dev_cfg,
        )

    if any(k in model_name for k in ["gap_sam", "gap-sam", "gapsam"]) or (
        ("gap" in model_name or "gap" in backbone) and any(b in backbone for b in ["tinyvit", "efficientvit", "repvit", "sam"])
    ):
        from sid_unet.models.gap_sam import GAPSAM
        ckpt_name = model_cfg.get(
            "checkpoint_path",
            model_cfg.get("pretrained_model_name_or_path", model_cfg.get("model_name", None))
        )
        dev_cfg = config.get("project", {}).get("device", "auto") if hasattr(config, "get") and hasattr(config.get("project", {}), "get") else "auto"
        bb_type = model_cfg.get("backbone", model_cfg.get("backbone_type", "tinyvit"))
        if str(bb_type).lower() in ["gap_sam", "gap-sam", "gapsam", "sam3"]:
            bb_type = model_cfg.get("sam_backbone", model_cfg.get("sam_backbone_type", "tinyvit"))

        # Sensible variant default based on backbone type
        default_variant = "11m" if "tinyvit" in str(bb_type).lower() else ("b0" if "efficientvit" in str(bb_type).lower() else "m1.1")
        variant_name = str(model_cfg.get("variant", model_cfg.get("model_variant", model_cfg.get("model_name", default_variant))))
        if variant_name in ["gap_sam", "gap-sam", "gapsam"]:
            variant_name = default_variant

        return GAPSAM(
            checkpoint_path=str(ckpt_name) if ckpt_name else None,
            backbone_type=str(bb_type),
            model_name=variant_name,
            text_encoder_type=model_cfg.get("text_encoder_type", "MobileCLIP-S0"),
            text_encoder_context_length=int(model_cfg.get("text_encoder_context_length", 16)),
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
            load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
            lora_r=int(model_cfg.get("lora_r", 16)),
            lora_alpha=int(model_cfg.get("lora_alpha", 32)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=model_cfg.get("lora_target_modules", None),
            prompt_text=str(model_cfg.get("prompt_text", "tampered region")),
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            target_size=tuple(model_cfg.get("target_size", [1008, 1008])),
            input_rescale=bool(model_cfg.get("input_rescale", True)),
            vae_pretrained_model_name_or_path=model_cfg.get("vae_pretrained_model_name_or_path", model_cfg.get("vae_checkpoint", "stabilityai/sd-vae-ft-mse")),
            vae_subfolder=model_cfg.get("vae_subfolder", None),
            freeze_vae=bool(model_cfg.get("freeze_vae", True)),
            use_dummy_vae=bool(model_cfg.get("use_dummy_vae", False)),
            dummy_vae_channels=tuple(model_cfg.get("dummy_vae_channels", [32, 64])),
            artifact_weight=float(model_cfg.get("artifact_weight", 1.0)),
            enable_artifact_classifier=bool(model_cfg.get("enable_artifact_classifier", True)),
            enable_artifact_cache=bool(model_cfg.get("enable_artifact_cache", True)),
            artifact_cache_size=int(model_cfg.get("artifact_cache_size", 50000)),
            device=dev_cfg,
            cache_dir=model_cfg.get("cache_dir", None),
            token=model_cfg.get("token", None),
            force_download=bool(model_cfg.get("force_download", False)),
        )

    if any(k in model_name for k in ["sam3_distil", "sam3-distil", "efficientsam3", "sam3distil"]) or (
        ("sam3" in model_name or "sam3" in backbone) and any(b in backbone for b in ["tinyvit", "efficientvit", "repvit"])
    ):
        from sid_unet.models.sam3_distil import SAM3DistilLoRA, DEFAULT_SAM3_DISTIL_CHECKPOINT
        ckpt_name = model_cfg.get(
            "checkpoint_path",
            model_cfg.get("pretrained_model_name_or_path", model_cfg.get("model_name", DEFAULT_SAM3_DISTIL_CHECKPOINT))
        )
        dev_cfg = config.get("project", {}).get("device", "auto") if hasattr(config, "get") and hasattr(config.get("project", {}), "get") else "auto"
        bb_type = model_cfg.get("backbone", model_cfg.get("backbone_type", "tinyvit"))
        if str(bb_type).startswith("sam3"):
            bb_type = "tinyvit"
        return SAM3DistilLoRA(
            checkpoint_path=str(ckpt_name) if ckpt_name else None,
            backbone_type=str(bb_type),
            model_name=str(model_cfg.get("variant", model_cfg.get("model_variant", model_cfg.get("model_name", "11m")))),
            text_encoder_type=model_cfg.get("text_encoder_type", "MobileCLIP-S0"),
            text_encoder_context_length=int(model_cfg.get("text_encoder_context_length", 16)),
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
            load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
            lora_r=int(model_cfg.get("lora_r", 16)),
            lora_alpha=int(model_cfg.get("lora_alpha", 32)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=model_cfg.get("lora_target_modules", None),
            prompt_text=str(model_cfg.get("prompt_text", "tampered region")),
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            target_size=tuple(model_cfg.get("target_size", [1008, 1008])),
            input_rescale=bool(model_cfg.get("input_rescale", True)),
            device=dev_cfg,
            cache_dir=model_cfg.get("cache_dir", None),
            token=model_cfg.get("token", None),
            force_download=bool(model_cfg.get("force_download", False)),
        )

    if any(k in model_name for k in ["iml", "refactor_maskformer", "refectormaskformer", "maskformer"]) or any(k in backbone for k in ["iml", "refactor_maskformer", "refectormaskformer", "maskformer"]):
        from sid_unet.models.iml_maskformer import IMLMaskformer
        dev_cfg = config.get("project", {}).get("device", "auto") if hasattr(config, "get") and hasattr(config.get("project", {}), "get") else "auto"
        return IMLMaskformer(
            variant=str(model_cfg.get("variant", model_cfg.get("model_variant", "large"))),
            sam2_repo=model_cfg.get("sam2_repo", None),
            config_path=model_cfg.get("config_path", None),
            checkpoint_path=model_cfg.get("checkpoint_path", model_cfg.get("pretrained_model_name_or_path", None)),
            embed_dim=int(model_cfg.get("embed_dim", 256)),
            num_queries=int(model_cfg.get("num_queries", 100)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            pixel_num_layers=int(model_cfg.get("pixel_num_layers", 6)),
            decoder_num_layers=int(model_cfg.get("decoder_num_layers", 9)),
            aux_classifier=bool(model_cfg.get("aux_classifier", False)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            target_size=tuple(model_cfg.get("target_size", [384, 384])),
            use_dummy=bool(model_cfg.get("use_dummy", False)),
            device=dev_cfg,
        )

    if any(k in model_name for k in ["detective_sam", "detective-sam", "detectivesam"]) or any(k in backbone for k in ["detective_sam", "detective-sam", "detectivesam"]):
        from sid_unet.models.detective_sam import DetectiveSAM
        dev_cfg = config.get("project", {}).get("device", "auto") if hasattr(config, "get") and hasattr(config.get("project", {}), "get") else "auto"
        return DetectiveSAM(
            pretrained_model_name_or_path=model_cfg.get("pretrained_model_name_or_path", model_cfg.get("checkpoint_path", None)),
            variant=str(model_cfg.get("variant", model_cfg.get("model_variant", "small"))).lower(),
            layer_width=int(model_cfg.get("layer_width", 64)),
            noise_intensity=float(model_cfg.get("noise_intensity", 0.75)),
            perturbation_type=str(model_cfg.get("perturbation_type", "blur_noise")),
            transformer_downscaling=int(model_cfg.get("transformer_downscaling", 16)),
            transformer_layers=int(model_cfg.get("transformer_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            target_size=tuple(model_cfg.get("target_size", [512, 512])),
            freeze_sam2=bool(model_cfg.get("freeze_sam2", True)),
            freeze_decoder=bool(model_cfg.get("freeze_decoder", True)),
            multimask_output=bool(model_cfg.get("multimask_output", False)),
            aux_classifier=bool(model_cfg.get("aux_classifier", False)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            use_dummy=bool(model_cfg.get("use_dummy", False)),
            device=dev_cfg,
            memory_enabled=bool(model_cfg.get("memory_enabled", False)),
            memory_max_entries=int(model_cfg.get("memory_max_entries", 5000)),
            memory_key_dim=int(model_cfg.get("memory_key_dim", 64)),
            memory_value_dim=int(model_cfg.get("memory_value_dim", 64)),
            memory_vectors_per_image=int(model_cfg.get("memory_vectors_per_image", 16)),
            memory_retrieval_k=int(model_cfg.get("memory_retrieval_k", 8)),
            memory_similarity_metric=str(model_cfg.get("memory_similarity_metric", "cosine")),
            memory_dedup_threshold=float(model_cfg.get("memory_dedup_threshold", 0.95)),
            memory_consolidation_method=str(model_cfg.get("memory_consolidation_method", "dedup_greedy")),
            memory_seed=int(model_cfg.get("memory_seed", 42)),
            memory_path=model_cfg.get("memory_path", None),
        )

    if "sam2" in model_name or "sam2" in backbone:
        from sid_unet.models.sam2 import (
            SAM2Segmenter,
            DEFAULT_SAM2_SMALL_CHECKPOINT,
            DEFAULT_SAM2_LARGE_CHECKPOINT,
        )
        variant = str(model_cfg.get("variant", model_cfg.get("model_variant", "small"))).lower()
        default_ckpt = (
            DEFAULT_SAM2_LARGE_CHECKPOINT
            if "large" in variant or "large" in model_name
            else DEFAULT_SAM2_SMALL_CHECKPOINT
        )
        ckpt_name = model_cfg.get(
            "pretrained_model_name_or_path",
            model_cfg.get("checkpoint_path", model_cfg.get("model_name", default_ckpt)),
        )
        dev_cfg = config.get("project", {}).get("device", "auto") if hasattr(config, "get") and hasattr(config.get("project", {}), "get") else "auto"
        return SAM2Segmenter(
            pretrained_model_name_or_path=str(ckpt_name),
            variant=str(variant) if variant else None,
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
            load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
            lora_r=int(model_cfg.get("lora_r", 8)),
            lora_alpha=int(model_cfg.get("lora_alpha", 16)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=model_cfg.get("lora_target_modules", None),
            aux_classifier=bool(model_cfg.get("aux_classifier", False)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            target_size=tuple(model_cfg.get("target_size", [1024, 1024])),
            use_mask_adapter=bool(model_cfg.get("use_mask_adapter", True)),
            freeze_decoder=bool(model_cfg.get("freeze_decoder", False)),
            reinit_decoder=bool(model_cfg.get("reinit_decoder", False)),
            device=dev_cfg,
        )

    if "sam3" in model_name or "sam3" in backbone:
        from sid_unet.models.sam3_qlora import SAM3QLoRA, DEFAULT_SAM3_CHECKPOINT
        ckpt_name = model_cfg.get("pretrained_model_name_or_path", model_cfg.get("model_name", DEFAULT_SAM3_CHECKPOINT))
        dev_cfg = config.get("project", {}).get("device", "auto") if hasattr(config, "get") and hasattr(config.get("project", {}), "get") else "auto"
        return SAM3QLoRA(
            pretrained_model_name_or_path=str(ckpt_name),
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
            load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
            lora_r=int(model_cfg.get("lora_r", 8)),
            lora_alpha=int(model_cfg.get("lora_alpha", 16)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=model_cfg.get("lora_target_modules", None),
            prompt_text=str(model_cfg.get("prompt_text", "tampered region")),
            use_learnable_prompt=bool(model_cfg.get("use_learnable_prompt", True)),
            num_learnable_tokens=int(model_cfg.get("num_learnable_tokens", 32)),
            aux_classifier=bool(model_cfg.get("aux_classifier", False)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            target_size=tuple(model_cfg.get("target_size", [1008, 1008])),
            device=dev_cfg,
        )

    if "efficientnet" in model_name or "efficientnet" in backbone:
        from sid_unet.models.efficientnet import EfficientNetSegmentation
        eff_backbone = backbone if backbone.startswith("efficientnet") else (
            model_name if model_name.startswith("efficientnet") and model_name != "efficientnet" else "efficientnet_b0"
        )
        return EfficientNetSegmentation(
            backbone=eff_backbone,
            pretrained=bool(model_cfg.get("pretrained", True)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            sacrifice_of_pixel=bool(model_cfg.get("sacrifice_of_pixel", False)),
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            bilinear=bool(model_cfg.get("bilinear", True)),
            gradient_checkpointing=ckpt_flag,
        )

    if any(k in model_name for k in ["minimized", "diffusion_diff_minimized", "diffusion-diff-minimized", "diff_minimized", "diffusion_minimized"]):
        from sid_unet.models.diffusion_diff_minimized import DiffusionDiffMinimizedModel, DEFAULT_DIFFUSION_CHECKPOINT
        ckpt_name = model_cfg.get("pretrained_model_name_or_path", model_cfg.get("model_name", DEFAULT_DIFFUSION_CHECKPOINT))
        dec_cfg = model_cfg.get("decoder", {})
        freeze_enc = model_cfg.get("freeze_encoder", True)
        if "autoencoder_trainable" in model_cfg:
            freeze_enc = not bool(model_cfg.get("autoencoder_trainable"))
        elif "trainable_autoencoder" in model_cfg:
            freeze_enc = not bool(model_cfg.get("trainable_autoencoder"))

        use_enc_skips = bool(
            model_cfg.get(
                "use_encoder_skips",
                model_cfg.get("use_skip_connections", model_cfg.get("skip_connections", False)),
            )
        )
        use_perp_skips = bool(model_cfg.get("use_perpendicular_skips", model_cfg.get("perpendicular_skips", True)))

        return DiffusionDiffMinimizedModel(
            pretrained_model_name_or_path=str(ckpt_name),
            pipeline=model_cfg.get("pipeline", None),
            vae_subfolder=model_cfg.get("vae_subfolder", "vae"),
            unet_subfolder=model_cfg.get("unet_subfolder", "unet"),
            timestep=model_cfg.get("timestep", None),
            timesteps=model_cfg.get("timesteps", [250]),
            timestep_embed_dim=int(model_cfg.get("timestep_embed_dim", 32)),
            sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 32)),
            include_noisy_latents=bool(model_cfg.get("include_noisy_latents", True)),
            include_added_noise=bool(model_cfg.get("include_added_noise", True)),
            include_predicted_noise=bool(model_cfg.get("include_predicted_noise", True)),
            include_noise_diff=bool(model_cfg.get("include_noise_diff", True)),
            include_z0=bool(model_cfg.get("include_z0", True)),
            decoder_config=dec_cfg,
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            scaling_factor=float(model_cfg.get("scaling_factor", 0.18215)),
            use_dummy=bool(model_cfg.get("use_dummy", False)),
            dummy_vae_channels=tuple(model_cfg.get("dummy_vae_channels", [32, 64])),
            dummy_unet_channels=tuple(model_cfg.get("dummy_unet_channels", [32, 64])),
            input_rescale=bool(model_cfg.get("input_rescale", True)),
            diffuser_fp16=bool(model_cfg.get("diffuser_fp16", training_cfg.get("amp", True))),
            freeze_encoder=bool(freeze_enc),
            use_encoder_skips=bool(use_enc_skips),
            use_perpendicular_skips=bool(use_perp_skips),
            device_map=model_cfg.get("device_map", "cuda" if torch.cuda.is_available() else None),
        )

    if any(k in model_name for k in ["diffusion_diff_v2", "diffusion-diff-v2", "diff_v2", "diffusion_diff2"]):
        from sid_unet.models.diffusion_diff_v2 import DiffusionDiffV2Model, DEFAULT_DIFFUSION_CHECKPOINT
        ckpt_name = model_cfg.get("pretrained_model_name_or_path", model_cfg.get("model_name", DEFAULT_DIFFUSION_CHECKPOINT))
        dec_cfg = model_cfg.get("decoder", {})
        freeze_enc = model_cfg.get("freeze_encoder", True)
        if "autoencoder_trainable" in model_cfg:
            freeze_enc = not bool(model_cfg.get("autoencoder_trainable"))
        elif "trainable_autoencoder" in model_cfg:
            freeze_enc = not bool(model_cfg.get("trainable_autoencoder"))

        use_enc_skips = bool(
            model_cfg.get(
                "use_encoder_skips",
                model_cfg.get("use_skip_connections", model_cfg.get("skip_connections", False)),
            )
        )
        use_perp_skips = bool(model_cfg.get("use_perpendicular_skips", model_cfg.get("perpendicular_skips", True)))

        return DiffusionDiffV2Model(
            pretrained_model_name_or_path=str(ckpt_name),
            vae_subfolder=model_cfg.get("vae_subfolder", "vae"),
            unet_subfolder=model_cfg.get("unet_subfolder", "unet"),
            timesteps=model_cfg.get("timesteps", [100, 250, 500]),
            timestep_embed_dim=int(model_cfg.get("timestep_embed_dim", 32)),
            sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 32)),
            include_noisy_latents=bool(model_cfg.get("include_noisy_latents", True)),
            include_added_noise=bool(model_cfg.get("include_added_noise", True)),
            include_predicted_noise=bool(model_cfg.get("include_predicted_noise", True)),
            include_noise_diff=bool(model_cfg.get("include_noise_diff", True)),
            include_z0=bool(model_cfg.get("include_z0", True)),
            decoder_config=dec_cfg,
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            scaling_factor=float(model_cfg.get("scaling_factor", 0.18215)),
            use_dummy=bool(model_cfg.get("use_dummy", False)),
            dummy_vae_channels=tuple(model_cfg.get("dummy_vae_channels", [32, 64])),
            dummy_unet_channels=tuple(model_cfg.get("dummy_unet_channels", [32, 64])),
            input_rescale=bool(model_cfg.get("input_rescale", True)),
            diffuser_fp16=bool(model_cfg.get("diffuser_fp16", training_cfg.get("amp", True))),
            freeze_encoder=bool(freeze_enc),
            use_encoder_skips=bool(use_enc_skips),
            use_perpendicular_skips=bool(use_perp_skips),
        )

    if any(k in model_name for k in ["diffusion_diff", "diffusion_noise", "diffuser_noise", "diffusion_multistep", "latent_noise"]):
        from sid_unet.models.diffusion_diff import DiffusionDiffModel, DEFAULT_DIFFUSION_CHECKPOINT
        ckpt_name = model_cfg.get("pretrained_model_name_or_path", model_cfg.get("model_name", DEFAULT_DIFFUSION_CHECKPOINT))
        dec_cfg = model_cfg.get("decoder", {})
        ae_cfg = model_cfg.get("autoencoder", {})
        ae_trainable = ae_cfg.get("trainable", True) if isinstance(ae_cfg, dict) else True
        ae_trainable = model_cfg.get("autoencoder_trainable", model_cfg.get("trainable_autoencoder", ae_trainable))
        return DiffusionDiffModel(
            pretrained_model_name_or_path=str(ckpt_name),
            vae_subfolder=model_cfg.get("vae_subfolder", "vae"),
            unet_subfolder=model_cfg.get("unet_subfolder", "unet"),
            timesteps=model_cfg.get("timesteps", [100, 250, 500]),
            timestep_embed_dim=int(model_cfg.get("timestep_embed_dim", 32)),
            sigma_embed_dim=int(model_cfg.get("sigma_embed_dim", 32)),
            include_noisy_latents=bool(model_cfg.get("include_noisy_latents", True)),
            include_added_noise=bool(model_cfg.get("include_added_noise", True)),
            include_predicted_noise=bool(model_cfg.get("include_predicted_noise", True)),
            include_noise_diff=bool(model_cfg.get("include_noise_diff", True)),
            include_z0=bool(model_cfg.get("include_z0", True)),
            decoder_config=dec_cfg,
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            scaling_factor=float(model_cfg.get("scaling_factor", 0.18215)),
            use_dummy=bool(model_cfg.get("use_dummy", False)),
            dummy_vae_channels=tuple(model_cfg.get("dummy_vae_channels", [32, 64])),
            dummy_unet_channels=tuple(model_cfg.get("dummy_unet_channels", [32, 64])),
            input_rescale=bool(model_cfg.get("input_rescale", True)),
            use_skip_connections=bool(model_cfg.get("use_skip_connections", model_cfg.get("skip_connections", True))),
            diffuser_fp16=bool(model_cfg.get("diffuser_fp16", training_cfg.get("amp", True))),
            autoencoder_trainable=bool(ae_trainable),
        )

    if any(k in model_name for k in ["vae_finetune", "diffusion_vae", "sd_vae"]) or (model_name == "vae"):
        from sid_unet.models.vae_finetune import DiffusionVAEFinetune, DEFAULT_SD_VAE_CHECKPOINT
        ckpt_name = model_cfg.get("pretrained_model_name_or_path", model_cfg.get("model_name", DEFAULT_SD_VAE_CHECKPOINT))
        return DiffusionVAEFinetune(
            pretrained_model_name_or_path=str(ckpt_name),
            subfolder=model_cfg.get("subfolder", "vae"),
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
            sample_mode=str(model_cfg.get("sample_mode", "sample")),
            scaling_factor=float(model_cfg.get("scaling_factor", 0.18215)),
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            gradient_checkpointing=ckpt_flag,
            use_dummy=bool(model_cfg.get("use_dummy", False)),
            dummy_channels=tuple(model_cfg.get("dummy_channels", [32, 64])),
            input_rescale=bool(model_cfg.get("input_rescale", True)),
            use_skip_connections=bool(model_cfg.get("use_skip_connections", model_cfg.get("skip_connections", True))),
        )

    if any(k in model_name for k in ["focal", "focal_hrnet", "focal_vit"]) or any(
        k in backbone for k in ["focal", "focal_hrnet", "focal_vit"]
    ):
        from sid_unet.models.focal import FOCAL
        focal_backbone = model_cfg.get("backbone", "hrnet")
        if "vit" in model_name or "vit" in backbone:
            focal_backbone = "vit"
        elif "hrnet" in model_name or "hrnet" in backbone:
            focal_backbone = "hrnet"

        ckpt_path = model_cfg.get(
            "checkpoint_path",
            model_cfg.get("pretrained_model_name_or_path", None)
        )
        return FOCAL(
            backbone=focal_backbone,
            pretrained=bool(model_cfg.get("pretrained", True)),
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            hf_repo=str(model_cfg.get("hf_repo", "KhangTruong/FOCAL")),
            cache_dir=model_cfg.get("cache_dir", None),
            aux_classifier=bool(model_cfg.get("aux_classifier", False)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            use_clustering=bool(model_cfg.get("use_clustering", True)),
            cluster_temperature=float(model_cfg.get("cluster_temperature", 0.1)),
            finetune_mode=model_cfg.get("finetune_mode", None),
            head_type=str(model_cfg.get("head_type", "auto")),
            force_vit_1024=bool(model_cfg.get("force_vit_1024", False)),
            cluster_separability_threshold=float(model_cfg.get("cluster_separability_threshold", 0.0015)),
            aux_pristine_gate=bool(model_cfg.get("aux_pristine_gate", False)),
            contrastive_sample_size=int(model_cfg.get("contrastive_sample_size", 1024)),
            peft_enabled=bool(model_cfg.get("peft_enabled", model_cfg.get("use_peft", False))),
            lora_r=int(model_cfg.get("lora_r", 8)),
            lora_alpha=float(model_cfg.get("lora_alpha", 16.0)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            use_dora=bool(model_cfg.get("use_dora", True)),
            lora_block_scope=model_cfg.get("lora_block_scope", "late_and_global"),
            lora_last_k_blocks=int(model_cfg.get("lora_last_k_blocks", 6)),
            lora_target_modules=model_cfg.get("lora_target_modules", None),
            train_neck=bool(model_cfg.get("train_neck", True)),
            train_vit_norms=bool(model_cfg.get("train_vit_norms", False)),
            hrnet_trainable_stages=model_cfg.get("hrnet_trainable_stages", None),
            freeze_backbone_bn=model_cfg.get("freeze_backbone_bn", None),
            freeze_stage4_bn=bool(model_cfg.get("freeze_stage4_bn", False)),
            use_feature_adapter=bool(model_cfg.get("use_feature_adapter", False)),
            prior_anchor_weight=float(model_cfg.get("prior_anchor_weight", loss_cfg.get("prior_anchor_weight", 0.0))),
            neck_l2_anchor_weight=float(model_cfg.get("neck_l2_anchor_weight", loss_cfg.get("neck_l2_anchor_weight", 0.0))),
            neck_lr_multiplier=float(model_cfg.get("neck_lr_multiplier", training_cfg.get("neck_lr_multiplier", 0.25))),
            lora_lr_multiplier=float(model_cfg.get("lora_lr_multiplier", training_cfg.get("lora_lr_multiplier", 1.0))),
            lora_weight_decay=float(model_cfg.get("lora_weight_decay", training_cfg.get("lora_weight_decay", 0.01))),
            # Domain-split 4-class / 4-cluster training with 2-class inference
            domain_split_mode=bool(model_cfg.get("domain_split_mode", False)),
            domain_loss_weight=float(model_cfg.get("domain_loss_weight", 1.0)),
            domain_proto_weight=float(model_cfg.get("domain_proto_weight", 0.5)),
            domain_supcon_weight=float(model_cfg.get("domain_supcon_weight", 0.5)),
            domain_clustering_mode=str(model_cfg.get("domain_clustering_mode", "prototype")),
            domain_names=model_cfg.get("domain_names", None),
            # Dual-backbone fusion (frozen primary + secondary backbone)
            secondary_backbone=model_cfg.get("secondary_backbone", model_cfg.get("fusion_backbone", None)),
            secondary_checkpoint_path=model_cfg.get("secondary_checkpoint_path", None),
            secondary_hf_repo=model_cfg.get("secondary_hf_repo", None),
            freeze_primary_backbone=model_cfg.get("freeze_primary_backbone", None),
            secondary_trainable=bool(model_cfg.get("secondary_trainable", True)),
            secondary_peft_enabled=bool(model_cfg.get("secondary_peft_enabled", False)),
            secondary_trainable_stages=model_cfg.get("secondary_trainable_stages", None),
            secondary_out_dim=int(model_cfg.get("secondary_out_dim", 128)),
            secondary_lr_multiplier=float(
                model_cfg.get("secondary_lr_multiplier", training_cfg.get("secondary_lr_multiplier", 0.25))
            ),
            fusion_mode=str(model_cfg.get("fusion_mode", "concat")),
            fusion_out_dim=model_cfg.get("fusion_out_dim", None),
            fusion_spatial_align=str(model_cfg.get("fusion_spatial_align", "primary")),
        )

    if any(k in model_name for k in ["pim", "seg_swin", "segswin"]) or any(
        k in backbone for k in ["pim", "seg_swin", "segswin"]
    ):
        from sid_unet.models.pim import PIM
        ckpt_path = model_cfg.get(
            "checkpoint_path",
            model_cfg.get("pretrained_model_name_or_path", None)
        )
        return PIM(
            pretrained=bool(model_cfg.get("pretrained", True)),
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            hf_repo=str(model_cfg.get("hf_repo", "KhangTruong/PIM")),
            cache_dir=model_cfg.get("cache_dir", None),
            num_classes=int(model_cfg.get("num_classes", 2)),
        )

    if any(k in model_name for k in ["msccnet", "mscc"]) or any(
        k in backbone for k in ["msccnet", "mscc"]
    ):
        from sid_unet.models.msccnet import MSCCNet
        ckpt_path = model_cfg.get(
            "checkpoint_path",
            model_cfg.get("pretrained_model_name_or_path", None),
        )
        bb_depth = model_cfg.get("backbone_depth", 50)
        if isinstance(backbone, str) and "101" in backbone:
            bb_depth = 101
        elif isinstance(backbone, str) and "50" in backbone:
            bb_depth = 50
        return MSCCNet(
            backbone_depth=int(bb_depth),
            backbone_outstride=int(model_cfg.get("backbone_outstride", 8)),
            use_stem=bool(model_cfg.get("use_stem", True)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            backbone_pretrained=model_cfg.get("backbone_pretrained", None),
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            hf_repo=model_cfg.get("hf_repo", None),
            allow_random_init=bool(model_cfg.get("allow_random_init", False)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            feats_channels=int(model_cfg.get("feats_channels", 512)),
            out_channels=int(model_cfg.get("out_channels", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            spectral_height=int(model_cfg.get("spectral_height", 64)),
            spectral_width=int(model_cfg.get("spectral_width", 64)),
            spectral_k=int(model_cfg.get("spectral_k", 2)),
            aux_classifier=bool(model_cfg.get("aux_classifier", True)),
            align_corners=bool(model_cfg.get("align_corners", False)),
            use_sync_bn=bool(model_cfg.get("use_sync_bn", False)),
            in_channels=int(model_cfg.get("in_channels", 3)),
        )

    if any(k in model_name for k in ["omnifd", "omni_fd", "omnivore"]) or any(
        k in backbone for k in ["omnifd", "omni_fd"]
    ):
        from sid_unet.models.omnifd import OmniFD
        ckpt_path = model_cfg.get(
            "checkpoint_path",
            model_cfg.get("pretrained_model_name_or_path", None),
        )
        variant = model_cfg.get("variant", model_cfg.get("backbone", "swin-t"))
        if str(variant).lower() in ("omnifd", "omni_fd", "default"):
            variant = "swin-t"
        return OmniFD(
            variant=str(variant),
            checkpoint_name=model_cfg.get("checkpoint_name", "omnifd_swin-t_50k_forgerynet"),
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            pretrained=bool(model_cfg.get("pretrained", True)),
            backbone_pretrained=bool(model_cfg.get("backbone_pretrained", False)),
            allow_random_init=bool(model_cfg.get("allow_random_init", False)),
            hf_repo=model_cfg.get("hf_repo", None),
            cache_dir=model_cfg.get("cache_dir", None),
            latent_channels=int(model_cfg.get("latent_channels", 512)),
            num_latents=int(model_cfg.get("num_latents", 64)),
            spatial_channels=int(model_cfg.get("spatial_channels", 256)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            aux_classifier=model_cfg.get("aux_classifier", None),
            has_image_head=model_cfg.get("has_image_head", None),
            has_spatial_head=model_cfg.get("has_spatial_head", None),
            with_cp=bool(ckpt_flag or model_cfg.get("with_cp", False)),
        )

    if any(k in model_name for k in ["forensic_moe", "forensic-moe", "moe"]) or any(
        k in backbone for k in ["forensic_moe", "forensic-moe"]
    ):
        from sid_unet.models.forensic_moe import ForensicMoE, ForensicMoEv2
        img_sz = model_cfg.get("img_size", None)
        if img_sz is None and hasattr(config, "data"):
            ds_sz = config.data.get("image_size", [512, 512])
            img_sz = ds_sz[0] if isinstance(ds_sz, (list, tuple)) else ds_sz
        is_v2 = (
            ("v2" in model_name)
            or ("v2" in backbone)
            or (str(model_cfg.get("version", "")).lower() == "v2")
            or (str(model_cfg.get("gate_version", "")).lower() == "v2")
        )
        common_kwargs = dict(
            pretrained=bool(model_cfg.get("pretrained", True)),
            allow_random_init=bool(model_cfg.get("allow_random_init", False)),
            sparsevit_checkpoint_path=model_cfg.get("sparsevit_checkpoint_path", None),
            sparsevit_hf_repo=model_cfg.get("sparsevit_hf_repo", None),
            trufor_checkpoint_path=model_cfg.get("trufor_checkpoint_path", None),
            trufor_hf_repo=model_cfg.get("trufor_hf_repo", None),
            diffusionprint_checkpoint_path=model_cfg.get("diffusionprint_checkpoint_path", None),
            diffusionprint_hf_repo=model_cfg.get("diffusionprint_hf_repo", None),
            cache_dir=model_cfg.get("cache_dir", None),
            img_size=int(img_sz or 512),
            gate_temperature=float(model_cfg.get("gate_temperature", 1.0)),
            balance_loss_weight=float(model_cfg.get("balance_loss_weight", loss_cfg.get("balance_loss_weight", 0.05))),
            aux_classifier=bool(model_cfg.get("aux_classifier", False)),
            num_classes=int(model_cfg.get("num_classes", 2)),
        )
        if is_v2:
            return ForensicMoEv2(
                **common_kwargs,
                router_proj_dim=int(model_cfg.get("router_proj_dim", 96)),
                router_hidden_dim=int(model_cfg.get("router_hidden_dim", 192)),
                unc_hidden_dim=int(model_cfg.get("unc_hidden_dim", 16)),
                unc_bias_init=float(model_cfg.get("unc_bias_init", -1.5)),
                oracle_loss_weight=float(model_cfg.get("oracle_loss_weight", loss_cfg.get("oracle_loss_weight", 0.02))),
                oracle_temperature=float(model_cfg.get("oracle_temperature", 0.5)),
                residual_l2_weight=float(model_cfg.get("residual_l2_weight", loss_cfg.get("residual_l2_weight", 1e-4))),
            )
        return ForensicMoE(
            **common_kwargs,
            spectral_hidden_dims=tuple(model_cfg.get("spectral_hidden_dims", [96, 192, 256])),
            spectral_embed_dim=int(model_cfg.get("spectral_embed_dim", 64)),
            spatial_hidden_dim=int(model_cfg.get("spatial_hidden_dim", 144)),
            spatial_num_blocks=int(model_cfg.get("spatial_num_blocks", 3)),
            oracle_loss_weight=float(model_cfg.get("oracle_loss_weight", loss_cfg.get("oracle_loss_weight", 0.15))),
            oracle_temperature=float(model_cfg.get("oracle_temperature", 0.15)),
        )

    if any(k in model_name for k in ["sparsevit", "sparse_vit"]) or any(
        k in backbone for k in ["sparsevit", "sparse_vit"]
    ):
        from sid_unet.models.sparsevit import SparseViT
        ckpt_path = model_cfg.get(
            "checkpoint_path",
            model_cfg.get("pretrained_model_name_or_path", None),
        )
        img_sz = model_cfg.get("img_size", None)
        if img_sz is None and hasattr(config, "data"):
            ds_sz = config.data.get("image_size", [512, 512])
            img_sz = ds_sz[0] if isinstance(ds_sz, (list, tuple)) else ds_sz
        return SparseViT(
            img_size=int(img_sz or 512),
            pretrained=bool(model_cfg.get("pretrained", True)),
            allow_random_init=bool(model_cfg.get("allow_random_init", False)),
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            hf_repo=model_cfg.get("hf_repo", None),
            cache_dir=model_cfg.get("cache_dir", None),
            freeze=bool(model_cfg.get("freeze", False)),
            auto_normalize=bool(model_cfg.get("auto_normalize", True)),
        )

    if any(k in model_name for k in ["trufor", "diffusionprint"]) or any(
        k in backbone for k in ["trufor", "diffusionprint"]
    ):
        from sid_unet.models.trufor import TruFor
        ckpt_path = model_cfg.get(
            "checkpoint_path",
            model_cfg.get("pretrained_model_name_or_path", None),
        )
        default_np_ch = 64 if "diffusionprint" in model_name or "diffusionprint" in backbone else 1
        return TruFor(
            np_out_channels=int(model_cfg.get("np_out_channels", default_np_ch)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            allow_random_init=bool(model_cfg.get("allow_random_init", False)),
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            hf_repo=model_cfg.get("hf_repo", None),
            hf_filename=model_cfg.get("hf_filename", None),
            cache_dir=model_cfg.get("cache_dir", None),
            freeze=bool(model_cfg.get("freeze", False)),
        )

    return UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        features=[int(f) for f in model_cfg.get("features", [64, 128, 256, 512])],
        bilinear=bool(model_cfg.get("bilinear", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        aux_classifier=bool(model_cfg.get("aux_classifier", True)),
        num_classes=int(model_cfg.get("num_classes", 3)),
        gradient_checkpointing=ckpt_flag,
    )
