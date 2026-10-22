"""
Feature Pyramid Network (FPN) and Multiscale Prediction Modules for AI Image Detection.

Provides:
- FeaturePyramidNetwork (FPNBlock): Classic and semantic FPN multi-scale feature
  fusion with top-down pathway, lateral connections, smoothing, and multi-scale
  prediction heads.
- MultiscalePredictionHead: Per-scale prediction heads for deep supervision.
- MultiscaleFusion: Unified multi-scale feature aggregator (concatenation / summation).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.blocks import OutConv

logger = logging.getLogger("sid_unet.models.fpn")


def get_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    """Create 2D normalization layer based on configuration string."""
    norm = norm_type.lower()
    if norm == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    elif norm == "groupnorm":
        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm in ("layernorm", "layer_norm"):
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    elif norm in ("instancenorm", "instance_norm"):
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm in ("none", "identity", ""):
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm_layer: {norm_type}")


def get_activation(activation: str) -> nn.Module:
    """Create activation layer based on configuration string."""
    act = activation.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    elif act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif act == "gelu":
        return nn.GELU()
    elif act in ("none", "identity", ""):
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class ConvBlock(nn.Module):
    """Convolution + Normalization + Activation building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_layer: str = "batchnorm",
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = get_norm_layer(norm_layer, out_channels)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.norm(self.conv(x))))


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) and Multiscale Feature Fusion Module.

    Architecture:
    1. Lateral Connections:
       1x1 convolutions project multi-scale feature maps from different stages
       (e.g., resolutions H/8, H/4, H/2, H) to a uniform feature dimension `fpn_channels`.
    2. Top-Down Pathway:
       Coarser semantic features (e.g. from lowest spatial resolution) are upsampled
       and added elementwise to the next finer spatial feature map.
    3. Anti-Aliasing Smoothing:
       3x3 convolutions with normalization and activation smooth each merged pyramid level.
    4. Multi-Scale Prediction Heads (Deep Supervision):
       Individual prediction heads at each pyramid level predict masks at native scales.
    5. Multi-Scale Feature Fusion (Semantic FPN):
       All smoothed pyramid levels are upsampled to the finest resolution (H, W),
       concatenated or summed, fused via convolution, and projected to the final
       high-resolution mask logits.

    Args:
        in_channels_list (List[int]): Channels for each feature map ordered from coarsest
            (lowest spatial resolution, e.g. H/8) to finest (highest spatial resolution, e.g. H).
        fpn_channels (int): Uniform channel dimension inside the feature pyramid (default: 128).
        out_channels (int): Output mask channels (default: 1 for binary mask logits).
        norm_layer (str): Normalization type ("batchnorm", "groupnorm", "layernorm", "none").
        activation (str): Activation function ("silu", "relu", "gelu").
        dropout (float): Dropout probability.
        fusion_mode (str): Multiscale aggregation mode ("concat", "sum").
        include_heads (bool): Whether to build per-scale prediction heads for multiscale output.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        fpn_channels: int = 128,
        out_channels: int = 1,
        norm_layer: str = "batchnorm",
        activation: str = "silu",
        dropout: float = 0.0,
        fusion_mode: str = "concat",
        include_heads: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels_list = [int(c) for c in in_channels_list]
        self.num_levels = len(self.in_channels_list)
        self.fpn_channels = int(fpn_channels)
        self.out_channels = int(out_channels)
        self.norm_layer = norm_layer
        self.activation = activation
        self.fusion_mode = fusion_mode.lower()
        self.include_heads = bool(include_heads)

        # 1. Lateral Connections: 1x1 conv projecting each stage to fpn_channels
        self.lateral_convs = nn.ModuleList()
        for in_c in self.in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_c, self.fpn_channels, kernel_size=1, bias=False),
                    get_norm_layer(norm_layer, self.fpn_channels),
                    get_activation(activation),
                )
            )

        # 2. Smooth Convolutions: 3x3 conv to reduce upsampling aliasing
        self.smooth_convs = nn.ModuleList()
        for _ in range(self.num_levels):
            self.smooth_convs.append(
                ConvBlock(
                    in_channels=self.fpn_channels,
                    out_channels=self.fpn_channels,
                    kernel_size=3,
                    padding=1,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout=dropout,
                )
            )

        # 3. Multi-Scale Prediction Heads (Deep Supervision)
        self.predict_heads = nn.ModuleList()
        if self.include_heads:
            for _ in range(self.num_levels):
                head = nn.Conv2d(self.fpn_channels, self.out_channels, kernel_size=3, padding=1)
                nn.init.kaiming_normal_(head.weight, mode="fan_out", nonlinearity="linear")
                if head.bias is not None:
                    nn.init.zeros_(head.bias)
                self.predict_heads.append(head)

        # 4. Multi-Scale Fusion Module
        if self.fusion_mode == "concat":
            fused_in_channels = self.fpn_channels * self.num_levels
        elif self.fusion_mode == "sum":
            fused_in_channels = self.fpn_channels
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}. Expected 'concat' or 'sum'.")

        self.fuse_conv = ConvBlock(
            in_channels=fused_in_channels,
            out_channels=self.fpn_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
            dropout=dropout,
        )

        # Final full-resolution prediction layer
        self.out_conv = nn.Conv2d(self.fpn_channels, self.out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.out_conv.weight, mode="fan_out", nonlinearity="linear")
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

        self.last_multiscale_logits: Optional[List[torch.Tensor]] = None

    def forward(
        self,
        features: List[torch.Tensor],
        return_multiscale: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through Feature Pyramid Network.

        Args:
            features: List of multi-scale feature maps [F_0, F_1, ..., F_{L-1}]
                ordered from coarsest (lowest spatial resolution) to finest (highest spatial resolution).
            return_multiscale: If True, returns `(fused_logits, multiscale_logits)`.
                Otherwise returns `fused_logits`.

        Returns:
            fused_logits: Full-resolution mask logits of shape [B, out_channels, H_finest, W_finest].
            multiscale_logits (optional): List of mask logits at each scale.
        """
        if len(features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature maps, got {len(features)}. "
                f"Configured in_channels_list: {self.in_channels_list}"
            )

        # 1. Lateral Projections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # 2. Top-Down Feature Merging (Coarsest -> Finest)
        pyramid: List[torch.Tensor] = [laterals[0]]
        for i in range(1, self.num_levels):
            prev_level = pyramid[i - 1]
            curr_lateral = laterals[i]
            target_size = curr_lateral.shape[2:]
            upsampled = F.interpolate(
                prev_level,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            merged = curr_lateral + upsampled
            pyramid.append(merged)

        # 3. Smoothing Convolutions
        smoothed = [smooth(p) for smooth, p in zip(self.smooth_convs, pyramid)]

        # 4. Multi-Scale Prediction Heads
        multiscale_logits: List[torch.Tensor] = []
        if self.include_heads:
            for head, p in zip(self.predict_heads, smoothed):
                multiscale_logits.append(head(p))
            self.last_multiscale_logits = multiscale_logits
        else:
            self.last_multiscale_logits = None

        # 5. Multi-Scale Feature Fusion to Finest Resolution
        finest_size = smoothed[-1].shape[2:]
        aligned_features: List[torch.Tensor] = []
        for p in smoothed:
            if p.shape[2:] != finest_size:
                aligned_p = F.interpolate(
                    p,
                    size=finest_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                aligned_p = p
            aligned_features.append(aligned_p)

        if self.fusion_mode == "concat":
            fused_rep = torch.cat(aligned_features, dim=1)
        else:  # sum
            fused_rep = torch.stack(aligned_features, dim=0).sum(dim=0)

        fused_features = self.fuse_conv(fused_rep)
        fused_logits = self.out_conv(fused_features)

        if return_multiscale:
            return fused_logits, multiscale_logits
        return fused_logits


FPNBlock = FeaturePyramidNetwork
