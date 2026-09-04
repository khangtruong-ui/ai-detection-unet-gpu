"""
EfficientNet-based segmentation models for AI image tampering and mask detection.
Supports:
1. Multi-scale feature mapping to a UNet decoder (default behaviour)
2. 'Sacrifice of Pixel' mode: uses only the final output feature map (8x8 or 7x7),
   projected through a linear layer, and zoomed out to match original image size.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sid_unet.models.blocks import AuxiliaryClassifier, DoubleConv, OutConv, Up


def _get_efficientnet_backbone(
    backbone_name: str = "efficientnet_b0",
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate EfficientNet backbone with optional pretrained weights."""
    model_fn = getattr(models, backbone_name, None)
    if model_fn is None:
        raise ValueError(
            f"Unsupported EfficientNet backbone: '{backbone_name}'. "
            f"Supported variants include: 'efficientnet_b0', 'efficientnet_b1', "
            f"'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', etc."
        )

    if pretrained:
        try:
            # Try torchvision modern weights enum first
            weights_enum_name = "".join([part.capitalize() for part in backbone_name.split("_")]) + "_Weights"
            weights_enum = getattr(models, weights_enum_name, None)
            if weights_enum is not None and hasattr(weights_enum, "DEFAULT"):
                return model_fn(weights=weights_enum.DEFAULT)
            return model_fn(weights="DEFAULT")
        except Exception:
            # Fall back to unweighted if internet is unavailable
            return model_fn(weights=None)
    else:
        return model_fn(weights=None)


class EfficientNetSegmentation(nn.Module):
    """
    EfficientNet-based segmentation network.

    Modes:
    1. Default (UNet structure): Maps intermediate feature maps from multiple stages
       (strides 2, 4, 8, 16, 32) into a progressive UNet decoder with skip connections.
    2. 'Sacrifice of Pixel' (sacrifice_of_pixel=True): Uses only the final bottleneck feature
       map (e.g. 8x8 or 7x7), feeds through a linear layer, and zooms out (bilinear interpolate)
       to match the full image resolution.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 1,
        sacrifice_of_pixel: bool = False,
        aux_classifier: bool = True,
        num_classes: int = 3,
        dropout: float = 0.1,
        bilinear: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sacrifice_of_pixel = sacrifice_of_pixel
        self.aux_classifier = aux_classifier
        self.num_classes = num_classes
        self.dropout = dropout
        self.bilinear = bilinear
        self.gradient_checkpointing = gradient_checkpointing

        # Load backbone
        raw_backbone = _get_efficientnet_backbone(backbone, pretrained=pretrained)
        features = raw_backbone.features

        # Adapt first conv if in_channels != 3
        if in_channels != 3:
            orig_conv = features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )
            features[0][0] = new_conv

        # Divide into stages:
        # stage 0..1: stride 2 (H/2)
        # stage 2:    stride 4 (H/4)
        # stage 3:    stride 8 (H/8)
        # stage 4..5: stride 16 (H/16)
        # stage 6..8: stride 32 (H/32 - Bottleneck)
        self.stage1 = nn.Sequential(features[0], features[1])
        self.stage2 = features[2]
        self.stage3 = features[3]
        self.stage4 = nn.Sequential(features[4], features[5])
        self.stage5 = nn.Sequential(features[6], features[7], features[8])

        # Infer channels dynamically with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            d1 = self.stage1(dummy)
            d2 = self.stage2(d1)
            d3 = self.stage3(d2)
            d4 = self.stage4(d3)
            d5 = self.stage5(d4)
            c1, c2, c3, c4, c5 = d1.shape[1], d2.shape[1], d3.shape[1], d4.shape[1], d5.shape[1]

        self.bottleneck_channels = c5

        # Auxiliary classifier head on bottleneck
        if self.aux_classifier:
            self.classifier_head = AuxiliaryClassifier(
                in_channels=self.bottleneck_channels,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            self.classifier_head = None

        if self.sacrifice_of_pixel:
            # Sacrifice of pixel mode:
            # Bottleneck feature map (8x8 or 7x7) -> Linear Layer -> Zoom out to pixel image
            self.linear = nn.Linear(self.bottleneck_channels, out_channels)
            self.up_sample = None
            self.outc = None
        else:
            # Default behaviour: UNet structure with skip connections
            factor = 2 if bilinear else 1
            dec4_out = 256 // factor
            dec3_out = 128 // factor
            dec2_out = 64 // factor
            dec1_out = 32

            self.up4 = Up(in_channels_down=c5, in_channels_skip=c4, out_channels=dec4_out, bilinear=bilinear, dropout=dropout)
            self.up3 = Up(in_channels_down=dec4_out, in_channels_skip=c3, out_channels=dec3_out, bilinear=bilinear, dropout=dropout)
            self.up2 = Up(in_channels_down=dec3_out, in_channels_skip=c2, out_channels=dec2_out, bilinear=bilinear, dropout=dropout)
            self.up1 = Up(in_channels_down=dec2_out, in_channels_skip=c1, out_channels=dec1_out, bilinear=bilinear, dropout=dropout)
            self.final_up = nn.Upsample(scale_factor=2, mode="bilinear" if bilinear else "nearest", align_corners=True if bilinear else None)
            self.outc = OutConv(dec1_out, out_channels)

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        """Dynamically toggle gradient checkpointing."""
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
        h_orig, w_orig = x.shape[2], x.shape[3]

        # Stage 1 to 5
        e1 = self.stage1(x)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        e5 = self.stage5(e4)  # Bottleneck feature map (e.g. 8x8 or 7x7)

        # Auxiliary classification logits
        class_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            class_logits = self.classifier_head(e5)

        if self.sacrifice_of_pixel:
            # 1. Feed bottleneck feature map through one layer of linear
            # Shape: (B, C, H_bot, W_bot) -> (B, H_bot, W_bot, C) -> Linear -> (B, H_bot, W_bot, out_channels)
            b, c, hb, wb = e5.shape
            feat_perm = e5.permute(0, 2, 3, 1)
            lin_out = self.linear(feat_perm)
            lin_perm = lin_out.permute(0, 3, 1, 2)  # (B, out_channels, hb, wb)

            # 2. Zoom out to match the size of the pixel image
            mask_logits = F.interpolate(
                lin_perm,
                size=(h_orig, w_orig),
                mode="bilinear" if self.bilinear else "nearest",
                align_corners=True if self.bilinear else None,
            )
        else:
            # Default UNet decoder with multi-scale skip connections
            d4 = self.up4(e5, e4)
            d3 = self.up3(d4, e3)
            d2 = self.up2(d3, e2)
            d1 = self.up1(d2, e1)
            d0 = self.final_up(d1)
            mask_logits = self.outc(d0)

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
        strict: bool = True,
        return_config: bool = False,
    ) -> Union[EfficientNetSegmentation, Tuple[EfficientNetSegmentation, Any]]:
        """Load trained EfficientNet model from checkpoint."""
        from sid_unet.models.unet import UNet
        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )
