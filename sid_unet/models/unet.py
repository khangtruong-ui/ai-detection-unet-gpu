"""
UNet architecture with optional 3-class auxiliary classification head for AI image mask detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from sid_unet.models.blocks import AuxiliaryClassifier, DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    """
    Standard UNet architecture with modular depth, configurable channel dimensions,
    bilinear/transposed upsampling, and an optional 3-class auxiliary classification head.

    Args:
        in_channels (int): Input image channels (default: 3 for RGB).
        out_channels (int): Mask output channels (default: 1 for binary mask logits).
        features (List[int]): Number of feature channels at each encoder level.
        bilinear (bool): Whether to use bilinear interpolation for upsampling.
        dropout (float): Dropout probability.
        aux_classifier (bool): Whether to enable auxiliary 3-class classification head.
        num_classes (int): Number of target classes for auxiliary classification (default: 3).
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.aux_classifier = aux_classifier

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

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Returns:
            If aux_classifier is True: (mask_logits, class_logits)
            If aux_classifier is False: mask_logits
        """
        # Encoder
        x1 = self.inc(x)
        encoder_skips = [x1]

        curr = x1
        for down_block in self.downs:
            curr = down_block(curr)
            encoder_skips.append(curr)

        # Bottleneck
        bottleneck_feat = self.bottleneck(curr)

        # Auxiliary Classification
        class_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            class_logits = self.classifier_head(bottleneck_feat)

        # Decoder
        d_curr = bottleneck_feat
        for i, up_block in enumerate(self.ups):
            skip = encoder_skips[-(i + 1)]
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


def build_model(config: Any) -> UNet:
    """Build UNet model instance from configuration dict."""
    model_cfg = config.get("model", {})
    return UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        features=[int(f) for f in model_cfg.get("features", [64, 128, 256, 512])],
        bilinear=bool(model_cfg.get("bilinear", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        aux_classifier=bool(model_cfg.get("aux_classifier", True)),
        num_classes=int(model_cfg.get("num_classes", 3)),
    )
