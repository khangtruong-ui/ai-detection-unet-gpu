"""
Neural network building blocks for UNet and auxiliary classification head.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2 with optional residual connection and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.residual = residual and (in_channels == out_channels)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual:
            out = out + identity
        out = self.relu2(out)
        return out


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    """Upscaling lower feature map, concatenating skip connection, and DoubleConv."""

    def __init__(
        self,
        in_channels_down: int,
        in_channels_skip: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            total_in = in_channels_down + in_channels_skip
        else:
            up_out = in_channels_down // 2
            self.up = nn.ConvTranspose2d(in_channels_down, up_out, kernel_size=2, stride=2)
            total_in = up_out + in_channels_skip

        self.conv = DoubleConv(total_in, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: Feature map from lower resolution (to upsample)
        x2: Feature map from encoder skip connection
        """
        x1 = self.up(x1)
        # Input shape is (B, C, H, W)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(
                x1,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 Convolution mapping feature maps to binary mask logits."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AuxiliaryClassifier(nn.Module):
    """Auxiliary classification head attached to the bottleneck features."""

    def __init__(self, in_channels: int, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pooled = self.pool(x)
        return self.fc(x_pooled)
