"""
Unified TruFor (`detconfcmx`) Module supporting both:
- **Expert 2 — TruFor (Noiseprint++)**: `np_out_channels=1` (1-channel DnCNN residual)
  from `grip-unina/TruFor` (`trufor.pth.tar`).
- **Expert 3 — TruFor-DiffusionPrint**: `np_out_channels=64` (64-channel MoCo DnCNN +
  1x1 `np_adapter` projection) from `giakoupg/diffusionprint`
  (`trufor_diffusionprint_tgif.pth`, CC-BY-4.0).

Architecture Summary:
- 17-layer DnCNN noise/diffusion-print extractor operating directly on raw `[0.0, 1.0]`
  float RGB inputs.
- Optional `np_adapter` (`Conv2d(64 -> 3, 1x1)`) when `np_out_channels > 3` (DiffusionPrint),
  or 3-channel tiling when `np_out_channels == 1` (Noiseprint++).
- Dual-path `SegFormer MiT-B2` + Cross-Modal Feature Rectify (`FRM`) & Feature Fusion
  (`FFM`) CMX backbone receiving ImageNet-normalized RGB + noise residual.
- Dual `MLPDecoder` heads (`decode_head` 2-class localization, `decode_head_conf` 1-ch
  pixel confidence) and `confpool` detection head (`detection`).
"""

from __future__ import annotations

import logging
import os
import urllib.request
import zipfile
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, to_2tuple
except ImportError:  # pragma: no cover
    from timm.models.layers import DropPath, to_2tuple

logger = logging.getLogger("sid_unet.models.trufor")

TRUFOR_WEIGHTS_ZIP_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
TRUFOR_DEFAULT_FILENAME = "trufor.pth.tar"
DIFFUSIONPRINT_DEFAULT_HF_REPO = "giakoupg/diffusionprint"
DIFFUSIONPRINT_DEFAULT_FILENAME = "trufor_diffusionprint_tgif.pth"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _make_dncnn(
    in_channels: int = 3,
    out_channels: int = 1,
    num_levels: int = 17,
    hidden_channels: int = 64,
    bn_momentum: float = 0.1,
) -> nn.Sequential:
    """Build 17-layer DnCNN matching TruFor / DiffusionPrint checkpoint keys (`dncnn.*`)."""
    features = [hidden_channels] * (num_levels - 1) + [out_channels]
    bns = [False] + [True] * (num_levels - 2) + [False]
    acts = ["relu"] * (num_levels - 1) + ["linear"]

    layers: List[nn.Module] = []
    curr_in = in_channels
    for i in range(num_levels):
        out_c = features[i]
        layers.append(nn.Conv2d(curr_in, out_c, kernel_size=3, padding=1, bias=(not bns[i])))
        if bns[i]:
            layers.append(nn.BatchNorm2d(out_c, momentum=bn_momentum))
        if acts[i] == "relu":
            layers.append(nn.ReLU(inplace=True))
        curr_in = out_c
    return nn.Sequential(*layers)


def _weighted_statistics_pooling(x: torch.Tensor, log_w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Confidence-weighted spatial statistics pooling used by TruFor's `confpool` detection head."""
    b, c = x.shape[0], x.shape[1]
    x_flat = x.reshape(b, c, -1)
    if log_w is None:
        log_w_flat = torch.zeros((b, 1, x_flat.shape[-1]), device=x.device, dtype=x.dtype)
    else:
        log_w_flat = log_w.reshape(b, 1, -1)

    log_w_norm = F.log_softmax(log_w_flat, dim=-1)
    x_min = -torch.logsumexp(log_w_norm - x_flat, dim=-1)
    x_max = torch.logsumexp(log_w_norm + x_flat, dim=-1)

    w = torch.exp(log_w_norm)
    x_avg = torch.sum(w * x_flat, dim=-1)
    x_msq = torch.sum(w * x_flat * x_flat, dim=-1)
    return torch.cat((x_min, x_max, x_avg, x_msq), dim=1)


# ------------------------------------------------------------------------------
# CMX Feature Rectify & Feature Fusion Modules (FRMs / FFMs)
# ------------------------------------------------------------------------------

class _ChannelWeights(nn.Module):
    def __init__(self, dim: int, reduction: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b = x1.shape[0]
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(b, self.dim * 2)
        mx = self.max_pool(x).view(b, self.dim * 2)
        y = self.mlp(torch.cat((avg, mx), dim=1)).view(b, self.dim * 2, 1)
        return y.reshape(b, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)


class _SpatialWeights(nn.Module):
    def __init__(self, dim: int, reduction: int = 1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x1.shape
        x = torch.cat((x1, x2), dim=1)
        return self.mlp(x).reshape(b, 2, 1, h, w).permute(1, 0, 2, 3, 4)


class _FeatureRectifyModule(nn.Module):
    def __init__(self, dim: int, reduction: int = 1, lambda_c: float = 0.5, lambda_s: float = 0.5) -> None:
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = _ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = _SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cw = self.channel_weights(x1, x2)
        sw = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * cw[1] * x2 + self.lambda_s * sw[1] * x2
        out_x2 = x2 + self.lambda_c * cw[0] * x1 + self.lambda_s * sw[0] * x1
        return out_x1, out_x2


class _CMXCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: Optional[float] = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x1.shape
        q1 = x1.reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        k1, v1 = self.kv1(x1).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = ((k1.transpose(-2, -1) @ v1) * self.scale).softmax(dim=-2)
        ctx2 = ((k2.transpose(-2, -1) @ v2) * self.scale).softmax(dim=-2)

        out1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(b, n, c).contiguous()
        out2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(b, n, c).contiguous()
        return out1, out2


class _CrossPath(nn.Module):
    def __init__(
        self,
        dim: int,
        reduction: int = 1,
        num_heads: int = 1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = _CMXCrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        return self.norm1(x1 + self.end_proj1(y1)), self.norm2(x2 + self.end_proj2(y2))


class _ChannelEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        b, _, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, H, W).contiguous()
        return self.norm(self.residual(x) + self.channel_embed(x))


class _FeatureFusionModule(nn.Module):
    def __init__(
        self,
        dim: int,
        reduction: int = 1,
        num_heads: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.cross = _CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = _ChannelEmbed(
            in_channels=dim * 2, out_channels=dim, reduction=reduction, norm_layer=norm_layer
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x1.shape
        t1 = x1.flatten(2).transpose(1, 2)
        t2 = x2.flatten(2).transpose(1, 2)
        t1, t2 = self.cross(t1, t2)
        return self.channel_emb(torch.cat((t1, t2), dim=-1), h, w)


# ------------------------------------------------------------------------------
# Dual SegFormer MiT-B2 Backbone & MLPDecoder
# ------------------------------------------------------------------------------

class _SegFormerDWConv(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        b, _, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, H, W).contiguous()
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class _SegFormerMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = _SegFormerDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.drop(self.act(self.dwconv(self.fc1(x), H, W)))
        return self.drop(self.fc2(x))


class _SegFormerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.norm(self.sr(x_).reshape(b, c, -1).permute(0, 2, 1))
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class _SegFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: Any = nn.LayerNorm,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _SegFormerAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _SegFormerMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class _OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        patch_size_t = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size_t,
            stride=stride,
            padding=(patch_size_t[0] // 2, patch_size_t[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, h, w


class DualSegformerMiTB2(nn.Module):
    """Dual-stream SegFormer MiT-B2 + CMX fusion matching `backbone.*` in TruFor checkpoints."""

    def __init__(
        self,
        img_size: int = 512,
        in_chans: int = 3,
        embed_dims: Sequence[int] = (64, 128, 320, 512),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        mlp_ratios: Sequence[float] = (4.0, 4.0, 4.0, 4.0),
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Any = partial(nn.LayerNorm, eps=1e-6),
        norm_fuse: type[nn.Module] = nn.BatchNorm2d,
        depths: Sequence[int] = (3, 4, 6, 3),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        stride0: int = 4,
    ) -> None:
        super().__init__()
        self.depths = list(depths)

        self.patch_embed1 = _OverlapPatchEmbed(img_size, 7, stride0, in_chans, embed_dims[0])
        self.patch_embed2 = _OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = _OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = _OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        self.extra_patch_embed1 = _OverlapPatchEmbed(img_size, 7, stride0, in_chans, embed_dims[0])
        self.extra_patch_embed2 = _OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.extra_patch_embed3 = _OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.extra_patch_embed4 = _OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            _SegFormerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])
        self.extra_block1 = nn.ModuleList([
            _SegFormerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            _SegFormerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])
        self.extra_block2 = nn.ModuleList([
            _SegFormerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.extra_norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            _SegFormerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])
        self.extra_block3 = nn.ModuleList([
            _SegFormerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.extra_norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            _SegFormerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])
        self.extra_block4 = nn.ModuleList([
            _SegFormerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.extra_norm4 = norm_layer(embed_dims[3])

        self.FRMs = nn.ModuleList([_FeatureRectifyModule(dim=d, reduction=1) for d in embed_dims])
        self.FFMs = nn.ModuleList([
            _FeatureFusionModule(dim=d, reduction=1, num_heads=nh, norm_layer=norm_fuse)
            for d, nh in zip(embed_dims, num_heads)
        ])

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor) -> List[torch.Tensor]:
        b = x_rgb.shape[0]
        outs: List[torch.Tensor] = []

        stages = [
            (self.patch_embed1, self.extra_patch_embed1, self.block1, self.extra_block1, self.norm1, self.extra_norm1),
            (self.patch_embed2, self.extra_patch_embed2, self.block2, self.extra_block2, self.norm2, self.extra_norm2),
            (self.patch_embed3, self.extra_patch_embed3, self.block3, self.extra_block3, self.norm3, self.extra_norm3),
            (self.patch_embed4, self.extra_patch_embed4, self.block4, self.extra_block4, self.norm4, self.extra_norm4),
        ]

        curr_rgb, curr_e = x_rgb, x_e
        for idx, (pe_r, pe_e, blks_r, blks_e, n_r, n_e) in enumerate(stages):
            t_r, h, w = pe_r(curr_rgb)
            t_e, _, _ = pe_e(curr_e)
            for blk in blks_r:
                t_r = blk(t_r, h, w)
            for blk in blks_e:
                t_e = blk(t_e, h, w)
            t_r = n_r(t_r).reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            t_e = n_e(t_e).reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            curr_rgb, curr_e = self.FRMs[idx](t_r, t_e)
            outs.append(self.FFMs[idx](curr_rgb, curr_e))

        return outs


class _DecoderMLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(2).transpose(1, 2))


class MLPDecoderHead(nn.Module):
    """SegFormer MLP decoder matching `decode_head.*` and `decode_head_conf.*`."""

    def __init__(
        self,
        in_channels: Sequence[int] = (64, 128, 320, 512),
        num_classes: int = 2,
        dropout_ratio: float = 0.1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        embed_dim: int = 512,
        align_corners: bool = False,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

        c1_c, c2_c, c3_c, c4_c = in_channels
        self.linear_c4 = _DecoderMLP(c4_c, embed_dim)
        self.linear_c3 = _DecoderMLP(c3_c, embed_dim)
        self.linear_c2 = _DecoderMLP(c2_c, embed_dim)
        self.linear_c1 = _DecoderMLP(c1_c, embed_dim)

        fuse_bn = norm_layer(embed_dim)
        if isinstance(fuse_bn, nn.BatchNorm2d):
            fuse_bn.eps = bn_eps
            fuse_bn.momentum = bn_momentum
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            fuse_bn,
            nn.ReLU(inplace=True),
        )
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        n = c4.shape[0]
        target_hw = c1.shape[2:]

        _c4 = F.interpolate(
            self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]),
            size=target_hw,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        _c3 = F.interpolate(
            self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]),
            size=target_hw,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        _c2 = F.interpolate(
            self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]),
            size=target_hw,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear_pred(x)


# ------------------------------------------------------------------------------
# Unified TruFor (`detconfcmx`) Builder
# ------------------------------------------------------------------------------

class TruFor(nn.Module):
    """
    Unified TruFor (`detconfcmx`) model supporting both:
    - `np_out_channels=1`: TruFor with Noiseprint++ (`trufor.pth.tar`)
    - `np_out_channels=64`: TruFor-DiffusionPrint (`trufor_diffusionprint_tgif.pth`)

    Expects raw `[0.0, 1.0]` float RGB inputs (with an automatic `_ensure_raw_rgb()`
    guard if ImageNet-standardized tensors are passed).
    """

    def __init__(
        self,
        np_out_channels: int = 1,
        pretrained: bool = True,
        allow_random_init: bool = False,
        checkpoint_path: Optional[str] = None,
        hf_repo: Optional[str] = None,
        hf_filename: Optional[str] = None,
        cache_dir: Optional[str] = None,
        freeze: bool = False,
        decoder_embed_dim: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.np_out_channels = int(np_out_channels)
        self.freeze = bool(freeze)
        self.hf_repo = hf_repo

        if self.np_out_channels not in (1, 3, 64):
            raise ValueError(f"Unsupported np_out_channels={self.np_out_channels}. Expected 1, 3, or 64.")

        self.channels = [64, 128, 320, 512]
        self.dncnn = _make_dncnn(
            in_channels=3,
            out_channels=self.np_out_channels,
            num_levels=17,
            hidden_channels=64,
            bn_momentum=0.1,
        )

        if self.np_out_channels > 3:
            self.np_adapter: Optional[nn.Module] = nn.Conv2d(self.np_out_channels, 3, kernel_size=1, bias=True)
        else:
            self.np_adapter = None

        self.backbone = DualSegformerMiTB2(
            img_size=512,
            in_chans=3,
            embed_dims=self.channels,
            num_heads=(1, 2, 5, 8),
            depths=(3, 4, 6, 3),
            sr_ratios=(8, 4, 2, 1),
        )
        self.decode_head = MLPDecoderHead(
            in_channels=self.channels,
            num_classes=2,
            dropout_ratio=0.1,
            embed_dim=decoder_embed_dim,
        )
        self.decode_head_conf = MLPDecoderHead(
            in_channels=self.channels,
            num_classes=1,
            dropout_ratio=0.1,
            embed_dim=decoder_embed_dim,
        )
        self.detection = nn.Sequential(
            nn.Linear(in_features=8, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1),
        )

        self.register_buffer(
            "_imagenet_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if not pretrained and not allow_random_init:
            raise ValueError(
                "TruFor requires pretrained=True (or explicit allow_random_init=True). "
                "Random weight initialization is strictly disabled."
            )

        if pretrained:
            self._load_pretrained(
                checkpoint_path=checkpoint_path,
                hf_repo=hf_repo,
                hf_filename=hf_filename,
                cache_dir=cache_dir,
            )

        if self.freeze:
            self.set_frozen(True)

    def set_frozen(self, frozen: bool = True) -> None:
        self.freeze = bool(frozen)
        for p in self.parameters():
            p.requires_grad_(not self.freeze)
        if self.freeze:
            self.eval()

    def train(self, mode: bool = True) -> "TruFor":
        super().train(mode)
        if self.freeze:
            super().train(False)
        return self

    def _resolve_checkpoint_path(
        self,
        checkpoint_path: Optional[str],
        hf_repo: Optional[str],
        hf_filename: Optional[str],
        cache_dir: Optional[str],
    ) -> str:
        if checkpoint_path and os.path.isfile(checkpoint_path):
            return os.path.abspath(checkpoint_path)

        home_cache = os.path.expanduser("~")
        if self.np_out_channels == 64:
            repo_id = hf_repo or DIFFUSIONPRINT_DEFAULT_HF_REPO
            fn = hf_filename or DIFFUSIONPRINT_DEFAULT_FILENAME
            candidates = [
                fn,
                os.path.join("pretrained", fn),
                os.path.join("weights", fn),
                os.path.join(home_cache, ".cache", "sid_unet_weights", fn),
            ]
            if cache_dir:
                candidates.insert(0, os.path.join(cache_dir, fn))
            for cand in candidates:
                if os.path.isfile(cand):
                    return os.path.abspath(cand)

            from huggingface_hub import hf_hub_download

            return hf_hub_download(repo_id=repo_id, filename=fn, cache_dir=cache_dir)

        # Noiseprint++ 1-ch TruFor weights
        target_fn = (
            os.path.basename(checkpoint_path)
            if checkpoint_path and checkpoint_path.endswith((".pth", ".pt", ".tar"))
            else TRUFOR_DEFAULT_FILENAME
        )
        candidates = [
            target_fn,
            os.path.join("pretrained", target_fn),
            os.path.join("weights", target_fn),
            os.path.join(home_cache, ".cache", "sid_unet_weights", target_fn),
            os.path.join(home_cache, ".cache", "sid_unet_weights", "trufor_unzipped", "weights", target_fn),
            os.path.join(home_cache, ".cache", "trufor", target_fn),
        ]
        if cache_dir:
            candidates.insert(0, os.path.join(cache_dir, target_fn))
        for cand in candidates:
            if os.path.isfile(cand):
                return os.path.abspath(cand)

        if hf_repo:
            try:
                from huggingface_hub import hf_hub_download

                return hf_hub_download(repo_id=hf_repo, filename=hf_filename or target_fn, cache_dir=cache_dir)
            except Exception as hf_err:
                logger.warning(f"HF Hub lookup failed for TruFor '{hf_repo}': {hf_err}")

        dest_dir = cache_dir or os.path.join(home_cache, ".cache", "sid_unet_weights")
        os.makedirs(dest_dir, exist_ok=True)
        zip_path = os.path.join(dest_dir, "TruFor_weights.zip")
        unzip_dir = os.path.join(dest_dir, "trufor_unzipped")
        try:
            urllib.request.urlretrieve(TRUFOR_WEIGHTS_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(unzip_dir)
            for root, _, files in os.walk(unzip_dir):
                for f in files:
                    if f == TRUFOR_DEFAULT_FILENAME or f.endswith(".pth.tar"):
                        return os.path.abspath(os.path.join(root, f))
        except Exception as dl_err:
            raise RuntimeError(
                f"Failed to download/extract TruFor weights from '{TRUFOR_WEIGHTS_ZIP_URL}': {dl_err}"
            ) from dl_err

        raise RuntimeError(f"Unable to locate '{TRUFOR_DEFAULT_FILENAME}' after extracting '{zip_path}'.")

    def _load_pretrained(
        self,
        checkpoint_path: Optional[str] = None,
        hf_repo: Optional[str] = None,
        hf_filename: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        resolved = self._resolve_checkpoint_path(checkpoint_path, hf_repo, hf_filename, cache_dir)
        ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid TruFor checkpoint payload at '{resolved}'")
        raw_sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        if not isinstance(raw_sd, dict):
            raise ValueError(f"Checkpoint '{resolved}' does not contain a state_dict dict.")

        clean_sd: Dict[str, Any] = {}
        for k, v in raw_sd.items():
            clean_k = k[len("module."):] if k.startswith("module.") else k
            clean_sd[clean_k] = v

        target_sd = self.state_dict()
        filtered_sd = {k: v for k, v in clean_sd.items() if k in target_sd}
        missing = sorted(set(target_sd.keys()) - set(filtered_sd.keys()))
        unexpected = sorted(set(clean_sd.keys()) - set(target_sd.keys()))
        if missing or unexpected:
            raise RuntimeError(
                f"TruFor (np_out_channels={self.np_out_channels}) state_dict mismatch from '{resolved}': "
                f"missing={missing[:10]} (total {len(missing)}), "
                f"unexpected={unexpected[:10]} (total {len(unexpected)})"
            )
        self.load_state_dict(filtered_sd, strict=True)
        logger.info(
            f"✅ Loaded genuine TruFor (np_out_channels={self.np_out_channels}) weights from: {resolved}"
        )

    def _ensure_raw_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is in raw `[0.0, 1.0]` range for DnCNN & internal normalization."""
        if float(x.min().detach()) < -0.05 or float(x.max().detach()) > 1.05:
            mean = self._imagenet_mean.to(device=x.device, dtype=x.dtype)
            std = self._imagenet_std.to(device=x.device, dtype=x.dtype)
            return torch.clamp(x * std + mean, 0.0, 1.0)
        return x

    def _prepro_imagenet(self, x_raw: torch.Tensor) -> torch.Tensor:
        mean = self._imagenet_mean.to(device=x_raw.device, dtype=x_raw.dtype)
        std = self._imagenet_std.to(device=x_raw.device, dtype=x_raw.dtype)
        return (x_raw - mean) / std

    def forward(
        self,
        rgb: torch.Tensor,
        return_dict: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        orig_h, orig_w = rgb.shape[2], rgb.shape[3]
        pad_h = (32 - orig_h % 32) % 32
        pad_w = (32 - orig_w % 32) % 32

        rgb_raw = self._ensure_raw_rgb(rgb)
        if pad_h > 0 or pad_w > 0:
            rgb_raw = F.pad(rgb_raw, (0, pad_w, 0, pad_h), mode="reflect")

        modal_x = self.dncnn(rgb_raw)
        if self.np_adapter is not None:
            modal_x = self.np_adapter(modal_x)
        elif modal_x.shape[1] == 1:
            modal_x = modal_x.repeat(1, 3, 1, 1)

        rgb_norm = self._prepro_imagenet(rgb_raw)
        feats = self.backbone(rgb_norm, modal_x)

        out = F.interpolate(
            self.decode_head(feats),
            size=rgb_raw.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        conf = F.interpolate(
            self.decode_head_conf(feats),
            size=rgb_raw.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :orig_h, :orig_w].contiguous()
            conf = conf[:, :, :orig_h, :orig_w].contiguous()

        mask_logits = out[:, 1:2, :, :] - out[:, 0:1, :, :]
        f1 = _weighted_statistics_pooling(conf).view(out.shape[0], -1)
        f2 = _weighted_statistics_pooling(mask_logits, F.logsigmoid(conf)).view(out.shape[0], -1)
        det_logit = self.detection(torch.cat((f1, f2), dim=-1))

        if return_dict:
            return {
                "mask_logits": mask_logits,
                "raw_logits": out,
                "conf_map": conf,
                "conf_prob": torch.sigmoid(conf),
                "det_logit": det_logit,
                "det_score": torch.sigmoid(det_logit),
                "pred_mask": torch.sigmoid(mask_logits),
            }
        return mask_logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        out = self.forward(x, return_dict=True)
        logits = out["mask_logits"] if isinstance(out, dict) else out
        return (torch.sigmoid(logits) >= threshold).float()
