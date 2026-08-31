"""
UNet architecture with optional 3-class auxiliary classification head for AI image mask detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from sid_unet.models.blocks import AuxiliaryClassifier, DoubleConv, Down, OutConv, Up


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
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        override_config: Optional[Union[Dict[str, Any], Any]] = None,
        strict: bool = True,
        return_config: bool = False,
    ) -> Union[UNet, Tuple[UNet, Any]]:
        """
        Load a trained UNet model from a checkpoint file (.pt).
        Automatically restores the model architecture hyperparameters
        saved within the checkpoint's embedded configuration.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
            device: Target device to move the model to (e.g. 'cpu', 'cuda', 'auto', or torch.device).
            override_config: Optional config dict or ConfigDict to override embedded config.
            strict: Whether to strictly enforce that the keys in state_dict match model keys.
            return_config: If True, returns a tuple (model, config_dict).

        Returns:
            UNet instance (eval mode by default), or (UNet, ConfigDict) if return_config=True.
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            saved_cfg = ckpt.get("config", {})
        elif isinstance(ckpt, dict) and any(k.startswith(("inc.", "downs.", "bottleneck.")) for k in ckpt.keys()):
            state_dict = ckpt
            saved_cfg = {}
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
        if override_config is not None:
            override_dict = override_config.to_dict() if isinstance(override_config, ConfigDict) else override_config
            merged = deep_merge(merged, override_dict)

        config = ConfigDict(merged)
        model = build_model(config)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        if device is not None:
            if isinstance(device, str):
                if device == "auto":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    device = torch.device(device)
            model.to(device)

        model.config = config
        if return_config:
            return model, config
        return model


def build_model(config: Any) -> UNet:
    """Build UNet model instance from configuration dict."""
    model_cfg = config.get("model", {}) if hasattr(config, "get") else {}
    training_cfg = config.get("training", {}) if hasattr(config, "get") else {}
    ckpt_flag = bool(
        model_cfg.get(
            "gradient_checkpointing",
            training_cfg.get("gradient_checkpointing", False) if hasattr(training_cfg, "get") else False,
        )
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
