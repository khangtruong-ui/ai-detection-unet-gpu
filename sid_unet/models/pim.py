"""
PIM: Plug-and-Play Image Inpainting Localization Network.
Official Paper: "A Plug-and-Play Image Inpainting Localization Network" (ICCV 2023)
Official Repository: https://github.com/ChenqiKONG/PIM
Pretrained Weights: KhangTruong/PIM (Hugging Face)

Strict zero-fallback policy:
- No local random initialization: `pretrained=False` raises an immediate ValueError.
- If weights cannot be downloaded/loaded, an immediate RuntimeError is raised.
- No dummy/fallback weights are permitted.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sid_unet.models.pim")

DEFAULT_PIM_HF_REPO = "KhangTruong/PIM"
DEFAULT_PIM_CKPT = "ckpt.pth"


# ------------------------------------------------------------------------------
# Helpers & Timm utilities
# ------------------------------------------------------------------------------

def to_2tuple(x: Any) -> Tuple[Any, Any]:
    if isinstance(x, (list, tuple)):
        return tuple(x[:2])
    return (x, x)


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    with torch.no_grad():
        import math
        l = (1.0 + math.erf((a - mean) / (std * math.sqrt(2.0)))) / 2.0
        u = (1.0 + math.erf((b - mean) / (std * math.sqrt(2.0)))) / 2.0
        tensor.uniform_(2.0 * l - 1.0, 2.0 * u - 1.0)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ------------------------------------------------------------------------------
# Swin Transformer Backbone Modules
# ------------------------------------------------------------------------------

class Mlp(nn.Module):
    """Multilayer Perceptron for Swin Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self Attention (W-MSA) with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionMask(nn.Module):
    """Window-based Attention with causal masking for the reconstruction branch."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        if self.training and training:
            shape = [n, n]
            c_mask = torch.triu(torch.ones(shape, device=x.device), diagonal=1).to(torch.uint8).t()
            start_mask = torch.ones(n, device=x.device, dtype=torch.float32)
            start_mask[0] = 0
            mask_pix = c_mask.unsqueeze(0).type_as(q)
            start_mask = start_mask.unsqueeze(1).type_as(q)
            attn = attn.masked_fill(mask_pix == 0, -1e4)
            attn = self.softmax(attn) * start_mask
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Any = nn.GELU,
        norm_layer: Any = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H: Optional[int] = None
        self.W: Optional[int] = None

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, l, c = x.shape
        h, w = self.H, self.W
        assert h is not None and w is not None
        assert l == h * w

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinTransformerBlockMask(nn.Module):
    """Swin Transformer Block with causal masking."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Any = nn.GELU,
        norm_layer: Any = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionMask(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H: Optional[int] = None
        self.W: Optional[int] = None

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        b, l, c = x.shape
        h, w = self.H, self.W
        assert h is not None and w is not None
        assert l == h * w

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows, mask=attn_mask, training=training)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, dim: int, norm_layer: Any = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, l, c = x.shape
        assert l == h * w
        x = x.view(b, h, w, c)

        if (h % 2 == 1) or (w % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(b, -1, 4 * c)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: Any = nn.LayerNorm,
        downsample: Optional[Any] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
        hp = int(np.ceil(h / self.window_size)) * self.window_size
        wp = int(np.ceil(w / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = h, w
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, h, w)
            wh, ww = (h + 1) // 2, (w + 1) // 2
            return x, h, w, x_down, wh, ww
        else:
            return x, h, w, x, h, w


class BasicLayerMask(nn.Module):
    """A basic Swin Transformer layer for the reconstruction branch."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: Any = nn.LayerNorm,
        downsample: Optional[Any] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlockMask(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, h: int, w: int, training: bool = True) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
        hp = int(np.ceil(h / self.window_size)) * self.window_size
        wp = int(np.ceil(w / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = h, w
            x = blk(x, attn_mask, training=training)

        if self.downsample is not None:
            x_down = self.downsample(x, h, w)
            wh, ww = (h + 1) // 2, (w + 1) // 2
            return x, h, w, x_down, wh, ww
        else:
            return x, h, w, x, h, w


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96, norm_layer: Optional[Any] = None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        if w % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
        if h % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            wh, ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer feature extractor."""

    def __init__(
        self,
        pretrain_img_size: int = 512,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
        norm_layer: Any = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            self.add_module(f"norm{i_layer}", layer)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.patch_embed(x)
        wh, ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, h, w, x, wh, ww = layer(x, wh, ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


class SwinTransformerMask(nn.Module):
    """Swin Transformer for reconstruction branch."""

    def __init__(
        self,
        pretrain_img_size: int = 512,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
        norm_layer: Any = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerMask(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            self.add_module(f"norm{i_layer}", layer)

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, ...]:
        x = self.patch_embed(x)
        wh, ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, h, w, x, wh, ww = layer(x, wh, ww, training=training)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


# ------------------------------------------------------------------------------
# Feature Local Enhancement & Feature Fusion
# ------------------------------------------------------------------------------

def create_conv_func():
    def func_cdc(x: torch.Tensor, weights: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1) -> torch.Tensor:
        weights_c = weights.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y - yc

    def func_rdc(x: torch.Tensor, weights: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> torch.Tensor:
        shape = weights.shape
        buffer = torch.zeros(shape[0], shape[1], 25, dtype=weights.dtype, device=weights.device)
        w = weights.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = w[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -w[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return F.conv2d(x, buffer, bias, stride=stride, padding=2 * dilation, dilation=dilation, groups=groups)

    def func_all_dc(x: torch.Tensor, weights_cd: torch.Tensor, weights_rd: torch.Tensor) -> torch.Tensor:
        return torch.cat([func_cdc(x, weights_cd), func_rdc(x, weights_rd)], dim=1)

    return func_all_dc


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.conv1(x))


class WeightBlock(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.fc = nn.Linear(channel, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(feature, (1, 1)).flatten(1)
        x = self.relu(self.fc(x))
        weight = torch.sigmoid(x)
        half_c = feature.size(1) // 2
        w0 = weight[:, 0:1].repeat(1, half_c)
        w1 = weight[:, 1:2].repeat(1, half_c)
        w = torch.cat([w0, w1], dim=1).view(-1, feature.size(1), 1, 1)
        return feature * w


class FeatureLocalConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.le_block = create_conv_func()
        self.weights_cd0 = nn.Parameter(torch.randn(96, 96, 3, 3))
        self.weights_cd1 = nn.Parameter(torch.randn(192, 192, 3, 3))
        self.weights_cd2 = nn.Parameter(torch.randn(384, 384, 3, 3))
        self.weights_cd3 = nn.Parameter(torch.randn(768, 768, 3, 3))

        self.weights_rd0 = nn.Parameter(torch.randn(96, 96, 3, 3))
        self.weights_rd1 = nn.Parameter(torch.randn(192, 192, 3, 3))
        self.weights_rd2 = nn.Parameter(torch.randn(384, 384, 3, 3))
        self.weights_rd3 = nn.Parameter(torch.randn(768, 768, 3, 3))

        self.weight_learn0 = WeightBlock(96 * 2)
        self.weight_learn1 = WeightBlock(192 * 2)
        self.weight_learn2 = WeightBlock(384 * 2)
        self.weight_learn3 = WeightBlock(768 * 2)

        self.norm0 = nn.BatchNorm2d(96 * 2)
        self.norm1 = nn.BatchNorm2d(192 * 2)
        self.norm2 = nn.BatchNorm2d(384 * 2)
        self.norm3 = nn.BatchNorm2d(768 * 2)

        self.conv_sep_all0 = SeparableConv2d(96 * 2, 96, 3, 1, 1)
        self.norm_all0 = nn.BatchNorm2d(96)

        self.conv_sep_all1 = SeparableConv2d(192 * 2, 192, 3, 1, 1)
        self.norm_all1 = nn.BatchNorm2d(192)

        self.conv_sep_all2 = SeparableConv2d(384 * 2, 384, 3, 1, 1)
        self.norm_all2 = nn.BatchNorm2d(384)

        self.conv_sep_all3 = SeparableConv2d(768 * 2, 768, 3, 1, 1)
        self.norm_all3 = nn.BatchNorm2d(768)

    def forward(self, feature: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        f0 = self.le_block(feature[0], self.weights_cd0, self.weights_rd0)
        f0 = self.norm_all0(self.conv_sep_all0(self.norm0(self.weight_learn0(f0)))) + feature[0]

        f1 = self.le_block(feature[1], self.weights_cd1, self.weights_rd1)
        f1 = self.norm_all1(self.conv_sep_all1(self.norm1(self.weight_learn1(f1)))) + feature[1]

        f2 = self.le_block(feature[2], self.weights_cd2, self.weights_rd2)
        f2 = self.norm_all2(self.conv_sep_all2(self.norm2(self.weight_learn2(f2)))) + feature[2]

        f3 = self.le_block(feature[3], self.weights_cd3, self.weights_rd3)
        f3 = self.norm_all3(self.conv_sep_all3(self.norm3(self.weight_learn3(f3)))) + feature[3]

        return [f0, f1, f2, f3]


class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_learn0 = WeightBlock(96 * 2)
        self.weight_learn1 = WeightBlock(192 * 2)
        self.weight_learn2 = WeightBlock(384 * 2)
        self.weight_learn3 = WeightBlock(768 * 2)

        self.norm0 = nn.BatchNorm2d(96 * 2)
        self.norm1 = nn.BatchNorm2d(192 * 2)
        self.norm2 = nn.BatchNorm2d(384 * 2)
        self.norm3 = nn.BatchNorm2d(768 * 2)

        self.conv_sep_all0 = SeparableConv2d(96 * 2, 96, 3, 1, 1)
        self.norm_all0 = nn.BatchNorm2d(96)

        self.conv_sep_all1 = SeparableConv2d(192 * 2, 192, 3, 1, 1)
        self.norm_all1 = nn.BatchNorm2d(192)

        self.conv_sep_all2 = SeparableConv2d(384 * 2, 384, 3, 1, 1)
        self.norm_all2 = nn.BatchNorm2d(384)

        self.conv_sep_all3 = SeparableConv2d(768 * 2, 768, 3, 1, 1)
        self.norm_all3 = nn.BatchNorm2d(768)

    def forward(self, feature_dc: Sequence[torch.Tensor], feature_rc: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        f0_cat = torch.cat([feature_dc[0], feature_rc[0]], dim=1)
        f0 = self.norm_all0(self.conv_sep_all0(self.norm0(self.weight_learn0(f0_cat)))) + feature_dc[0]

        f1_cat = torch.cat([feature_dc[1], feature_rc[1]], dim=1)
        f1 = self.norm_all1(self.conv_sep_all1(self.norm1(self.weight_learn1(f1_cat)))) + feature_dc[1]

        f2_cat = torch.cat([feature_dc[2], feature_rc[2]], dim=1)
        f2 = self.norm_all2(self.conv_sep_all2(self.norm2(self.weight_learn2(f2_cat)))) + feature_dc[2]

        f3_cat = torch.cat([feature_dc[3], feature_rc[3]], dim=1)
        f3 = self.norm_all3(self.conv_sep_all3(self.norm3(self.weight_learn3(f3_cat)))) + feature_dc[3]

        return [f0, f1, f2, f3]


# ------------------------------------------------------------------------------
# Decode Head (UPerHead)
# ------------------------------------------------------------------------------

class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, inplace: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class PPM(nn.ModuleList):
    def __init__(self, pool_scales: Sequence[int], in_channels: int, channels: int, align_corners: bool = False):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(in_channels, channels, 1, bias=False)
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled = F.interpolate(ppm_out, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners)
            ppm_outs.append(upsampled)
        return ppm_outs


class UPerHead(nn.Module):
    """UPerHead for multi-scale dense feature decoding."""

    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384, 768),
        channels: int = 512,
        num_classes: int = 2,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        align_corners: bool = False,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.align_corners = align_corners

        self.psp_modules = PPM(pool_scales, self.in_channels[-1], channels, align_corners=align_corners)
        self.bottleneck = ConvModule(self.in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1, bias=False)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for c in self.in_channels[:-1]:
            self.lateral_convs.append(ConvModule(c, channels, 1, bias=False))
            self.fpn_convs.append(ConvModule(channels, channels, 3, padding=1, bias=False))

        self.fpn_bottleneck = ConvModule(len(self.in_channels) * channels, channels, 3, padding=1, bias=False)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def psp_forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        return self.bottleneck(torch.cat(psp_outs, dim=1))

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        laterals = [lat(inputs[i]) for i, lat in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners)

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])

        target_size = fpn_outs[0].shape[2:]
        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size, mode="bilinear", align_corners=self.align_corners)

        out = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return self.conv_seg(out)


# ------------------------------------------------------------------------------
# Complete Seg_Swin Architecture
# ------------------------------------------------------------------------------

class Seg_Swin(nn.Module):
    """Complete dual-branch Swin Transformer architecture matching official PIM repo."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        self.backbone_swin = SwinTransformer(
            pretrain_img_size=512,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
        )

        self.decode_head = UPerHead(
            in_channels=[96, 192, 384, 768],
            channels=512,
            num_classes=2,
            pool_scales=(1, 2, 3, 6),
            align_corners=False,
        )

        self.boundary_head = UPerHead(
            in_channels=[96, 192, 384, 768],
            channels=512,
            num_classes=2,
            pool_scales=(1, 2, 3, 6),
            align_corners=False,
        )

        self.local_conv_enhancement = FeatureLocalConv()

        self.backbone_recon = SwinTransformerMask(
            pretrain_img_size=512,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
        )

        self.recon_head = UPerHead(
            in_channels=[96, 192, 384, 768],
            channels=512,
            num_classes=3,
            pool_scales=(1, 2, 3, 6),
            align_corners=False,
        )

        self.feature_fusion = FeatureFusion()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_dc = self.backbone_swin(x)
        feature_dc = self.local_conv_enhancement(feature_dc)
        feature_rc = self.backbone_recon(x)
        recon_img = self.recon_head(feature_rc)
        feature_fus = self.feature_fusion(feature_dc, feature_rc)
        out1 = self.decode_head(feature_fus)
        out2 = self.boundary_head(feature_fus)
        return out1, out2, recon_img


# ------------------------------------------------------------------------------
# Unified PIM Model Wrapper
# ------------------------------------------------------------------------------

class PIM(nn.Module):
    """
    PIM: Plug-and-Play Image Inpainting Localization Network.

    Features:
    - Dual-stream Swin-Transformer (deep feature stream + reconstruction feature stream).
    - Local frequency-sensitive convolution enhancement and dynamic feature fusion.
    - Multi-task dense heads for forgery segmentation, edge boundary localization, and image reconstruction.
    - Pretrained weights loaded strictly from Hugging Face `KhangTruong/PIM/ckpt.pth`.
    - Zero-fallback policy: `pretrained=False` or missing weights raises an immediate Exception.
    """

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        hf_repo: str = DEFAULT_PIM_HF_REPO,
        cache_dir: Optional[str] = None,
        num_classes: int = 2,
        **kwargs: Any,
    ):
        super().__init__()
        self.hf_repo = hf_repo
        self.num_classes = int(num_classes)

        if not pretrained:
            raise ValueError(
                "PIM requires pretrained=True. Fallback to random weights is strictly disabled."
            )

        self.net = Seg_Swin(num_classes=self.num_classes)

        # Load weights with strict zero-fallback policy
        self._load_pretrained(
            source=checkpoint_path or DEFAULT_PIM_CKPT,
            default_filename=DEFAULT_PIM_CKPT,
            cache_dir=cache_dir,
        )

    def _load_pretrained(
        self, source: str, default_filename: str, cache_dir: Optional[str] = None
    ) -> None:
        """Download or resolve genuine pretrained weights and load state_dict strictly."""
        resolved_path = None

        # 1. Direct local file
        if os.path.isfile(source):
            resolved_path = os.path.abspath(source)

        # 2. Local candidate directories
        if resolved_path is None:
            filename = os.path.basename(source) if source.endswith((".pth", ".pt")) else default_filename
            candidates = [
                filename,
                os.path.join("checkpoints", filename),
                os.path.join("weights", filename),
                os.path.join("pretrained_models", filename),
                os.path.join("/tmp/pim_repo", filename),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    resolved_path = os.path.abspath(cand)
                    break

        # 3. Download from Hugging Face Hub
        if resolved_path is None:
            repo_id = self.hf_repo
            filename = os.path.basename(source) if source.endswith((".pth", ".pt")) else default_filename
            # Priority candidates on Hugging Face Hub (handles both legacy and reformatted SID-UNet layouts)
            hf_candidates = [filename]
            for c in ["checkpoints/checkpoint_best.pt", "checkpoint_best.pt", "checkpoints/checkpoint_latest.pt", "ckpt.pth", "checkpoints/ckpt.pth"]:
                if c not in hf_candidates:
                    hf_candidates.append(c)

            from huggingface_hub import hf_hub_download
            last_err = None
            for c_fn in hf_candidates:
                try:
                    resolved_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=c_fn,
                        cache_dir=cache_dir,
                    )
                    if resolved_path and os.path.isfile(resolved_path):
                        logger.info(f"Downloaded PIM pretrained weights from HF '{repo_id}/{c_fn}': {resolved_path}")
                        break
                except Exception as hf_err:
                    last_err = hf_err
                    continue

            if resolved_path is None and last_err is not None:
                raise RuntimeError(
                    f"Failed to download/load PIM weights from Hugging Face repo '{repo_id}' (tried {hf_candidates}): {last_err}. "
                    f"Strict zero-fallback policy: aborting initialization."
                ) from last_err

        if resolved_path is None or not os.path.isfile(resolved_path):
            raise RuntimeError(
                f"Failed to resolve PIM weights at '{source}'. "
                f"Strict zero-fallback policy: local initialization is strictly disabled."
            )

        try:
            ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except Exception as load_err:
            raise RuntimeError(
                f"Corrupt or unreadable checkpoint at '{resolved_path}': {load_err}"
            ) from load_err

        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid PIM checkpoint payload from '{resolved_path}': expected state_dict dict.")

        if "model_state_dict" in ckpt:
            raw_sd = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            raw_sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            raw_sd = ckpt["state_dict"]
        else:
            raw_sd = ckpt

        # Strip module. and net. prefixes (e.g. from DataParallel save or top-level wrapper)
        clean_sd = {}
        for k, v in raw_sd.items():
            clean_k = k[7:] if k.startswith("module.") else k
            if clean_k.startswith("net."):
                clean_k = clean_k[4:]
            clean_sd[clean_k] = v

        try:
            # Check strictly that every expected parameter exists in clean_sd
            expected_keys = set(self.net.state_dict().keys())
            missing_keys = [k for k in expected_keys if k not in clean_sd]
            if missing_keys:
                raise RuntimeError(f"PIM missing required pretrained weights: {missing_keys[:10]}")

            missing, unexpected = self.net.load_state_dict(clean_sd, strict=False)
            logger.info(
                f"✅ Loaded genuine PIM pretrained weights from: {resolved_path} "
                f"(missing: {len(missing)}, unexpected: {len(unexpected)})"
            )
        except Exception as sd_err:
            raise RuntimeError(
                f"Failed strict load_state_dict for PIM: {sd_err}. "
                f"Zero-fallback policy enforced."
            ) from sd_err

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass of PIM.

        Args:
            x: Input RGB tensor `[B, 3, H, W]`.
            return_dict: If True, returns dictionary with standard keys.

        Returns:
            Dictionary with:
                - `mask_logits`: Binary forgery logit `[B, 1, H, W]`
                - `pred_mask`: Sigmoid probability `[B, 1, H, W]`
                - `boundary_logits`: Binary edge boundary logit `[B, 1, H, W]`
                - `recon_img`: Reconstructed RGB image `[B, 3, H, W]`
                - `raw_seg`: Raw 2-class logits `[B, 2, H, W]`
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        seg_small, boundary_small, recon_small = self.net(x)

        # Interpolate outputs to input resolution
        seg_full = F.interpolate(seg_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        boundary_full = F.interpolate(boundary_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        recon_full = F.interpolate(recon_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        # Single binary channel logit (forged class 1 minus pristine class 0)
        mask_logits = seg_full[:, 1:2] - seg_full[:, 0:1]
        boundary_logits = boundary_full[:, 1:2] - boundary_full[:, 0:1]

        if return_dict:
            return {
                "mask_logits": mask_logits,
                "pred_mask": torch.sigmoid(mask_logits),
                "boundary_logits": boundary_logits,
                "recon_img": recon_full,
                "raw_seg": seg_full,
                "raw_boundary": boundary_full,
            }

        return mask_logits, boundary_logits, recon_full

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Generate binary prediction mask `[B, 1, H, W]`."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_dict=True)
            prob = out["pred_mask"]
            return (prob > threshold).float()
