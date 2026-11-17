"""
OmniFD: A Unified Model for Versatile Face Forgery Detection.
Official Paper: Liu et al., "OmniFD: A Unified Model for Versatile Face Forgery Detection",
arXiv:2512.01128, 2025.
Official Repository: https://github.com/haotianll/OmniFD (MIT License).
Official Pretrained Weights (Google Drive):
    Folder: https://drive.google.com/drive/folders/1Bh8FbF-jiyPtgr7-URNuGAOJhi8kZSEC?usp=sharing

Architecture Summary:
- Omnivore SwinTransformer3D backbone (`SwinTransformerOmni`, Swin-T / Swin-S / Swin-B)
  operating on singleton-frame tensors `[B, 3, 1, H, W]` for images (or `[B, 3, T, H, W]`
  for video clips), extracting 4 hierarchical feature stages `[192, 384, 768, 768]`.
- Perceiver-style `Interaction` neck projecting all 4 backbone stages into a shared
  latent space (`512`-d) with learnable stage embeddings and cross-attending `64` global
  forgery latent tokens across 3 interaction layers.
- Unified multi-task heads:
  - `ImageClsHead` (`image`): pools latent tokens for 2-class real/fake prediction.
  - `SpatialHead` (`spatial`): fuses resized multi-scale spatial features (`2112 -> 256`)
    with projected latent queries via `einsum('bchw,bck->bkhw')` for pixel-level
    spatial forgery localization.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from operator import mul
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_

logger = logging.getLogger("sid_unet.models.omnifd")

OMNIFD_GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1Bh8FbF-jiyPtgr7-URNuGAOJhi8kZSEC?usp=sharing"
OMNIFD_ANNOTATIONS_GDRIVE_URL = "https://drive.google.com/drive/folders/1auf-c_82SpKP0XfJlx9vJdcz7BiwLOy_?usp=sharing"

OMNIFD_GDRIVE_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "omnifd_swin-t_50k_forgerynet": {
        "file_id": "1ZJAHtTL21qaCTOSacxeZ-MkGTnrEY0BL",
        "filename": "omnifd_swin-t_50k_forgerynet.pth",
        "backbone": "swin-t",
        "tasks": "unified (image, spatial, video, temporal)",
    },
    "omnifd_swin-s_50k_forgerynet": {
        "file_id": "14Y9hpsKlGlPLUh2dPFugssSXS4ftFXL2",
        "filename": "omnifd_swin-s_50k_forgerynet.pth",
        "backbone": "swin-s",
        "tasks": "unified (image, spatial, video, temporal)",
    },
    "omnifd_swin-s_20k_ff_image": {
        "file_id": "1RMvEDQD6Uzj2NBm9RgS6KA5bpblEXc41",
        "filename": "omnifd_swin-s_20k_ff_image.pth",
        "backbone": "swin-s",
        "tasks": "image",
    },
    "omnifd_swin-s_20k_ff_spatial": {
        "file_id": "1oKXW99ZyNGRwnxwbC22AwJoCoDkrM-Ih",
        "filename": "omnifd_swin-s_20k_ff_spatial.pth",
        "backbone": "swin-s",
        "tasks": "spatial",
    },
    "omnifd_swin-s_20k_fmld_spatial": {
        "file_id": "1jiHcVpHhcdSBb7eqnBs_89tZ3UzAPIIk",
        "filename": "omnifd_swin-s_20k_fmld_spatial.pth",
        "backbone": "swin-s",
        "tasks": "spatial",
    },
    "omnifd_swin-s_20k_wdf_video": {
        "file_id": "1jAS3WIBgQj_9sUby8xwQzB-ZzqoR3DVf",
        "filename": "omnifd_swin-s_20k_wdf_video.pth",
        "backbone": "swin-s",
        "tasks": "video",
    },
    "omnifd_swin-s_50k_ff_video": {
        "file_id": "18YEN4E0OgCYUHucIaEaFPTWHDjpChga9",
        "filename": "omnifd_swin-s_50k_ff_video.pth",
        "backbone": "swin-s",
        "tasks": "video",
    },
    "omnifd_swin-s_50k_lavdf_temporal": {
        "file_id": "1kevdiALBKD0JRigp3sPrAjA6h-rMBlId",
        "filename": "omnifd_swin-s_50k_lavdf_temporal.pth",
        "backbone": "swin-s",
        "tasks": "temporal",
    },
}

OMNIVORE_TRUNK_URLS: Dict[str, str] = {
    "omnivore_swinT": "https://dl.fbaipublicfiles.com/omnivore/models/swinT_checkpoint.torch",
    "omnivore_swinS": "https://dl.fbaipublicfiles.com/omnivore/models/swinS_checkpoint.torch",
    "omnivore_swinB": "https://dl.fbaipublicfiles.com/omnivore/models/swinB_checkpoint.torch",
}

SWIN_VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "swin-t": {
        "checkpoint_name": "omnivore_swinT",
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": (8, 7, 7),
        "drop_path_rate": 0.2,
        "feat_channels": [192, 384, 768, 768],
    },
    "swin-s": {
        "checkpoint_name": "omnivore_swinS",
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": (8, 7, 7),
        "drop_path_rate": 0.3,
        "feat_channels": [192, 384, 768, 768],
    },
    "swin-b": {
        "checkpoint_name": "omnivore_swinB",
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": (16, 7, 7),
        "drop_path_rate": 0.3,
        "feat_channels": [256, 512, 1024, 1024],
    },
}


def _normalize_variant(variant: str) -> str:
    v = str(variant).lower().strip().replace("_", "-")
    if v in ("swin-t", "swint", "tiny", "omnivore-swint"):
        return "swin-t"
    if v in ("swin-s", "swins", "small", "omnivore-swins"):
        return "swin-s"
    if v in ("swin-b", "swinb", "base", "omnivore-swinb"):
        return "swin-b"
    raise ValueError(f"Unsupported OmniFD backbone variant: '{variant}'. Choose from {list(SWIN_VARIANT_CONFIGS.keys())}.")


# ------------------------------------------------------------------------------
# Omnivore SwinTransformer3D Backbone
# ------------------------------------------------------------------------------

class _Im2Video(nn.Module):
    def __init__(self, time_dim: int = 2) -> None:
        super().__init__()
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.unsqueeze(self.time_dim)
        if x.ndim == 5:
            return x
        raise ValueError(f"Expected 4D or 5D tensor, got {tuple(x.shape)}")


class _SwinMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def _window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    b, d, h, w, c = x.shape
    x = x.view(
        b,
        d // window_size[0], window_size[0],
        h // window_size[1], window_size[1],
        w // window_size[2], window_size[2],
        c,
    )
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), c)


def _window_reverse(
    windows: torch.Tensor, window_size: Tuple[int, int, int], b: int, d: int, h: int, w: int
) -> torch.Tensor:
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)


def _get_window_size(
    x_size: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    shift_size: Optional[Tuple[int, int, int]] = None,
):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0
    if use_shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


@lru_cache(maxsize=64)
def _compute_3d_attn_mask(
    d: int, h: int, w: int,
    window_size: Tuple[int, int, int],
    shift_size: Tuple[int, int, int],
    device_str: str,
) -> torch.Tensor:
    device = torch.device(device_str)
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    cnt = 0
    for d_s in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h_s in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w_s in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d_s, h_s, w_s, :] = cnt
                cnt += 1
    mask_windows = _window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads,
            )
        )

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

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

        rel_bias = self.relative_position_bias_table[self.relative_position_index[:n, :n].reshape(-1)].reshape(
            n, n, -1
        )
        attn = attn + rel_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = self.softmax(attn.view(-1, self.num_heads, n, n))
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _SwinMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward_part1(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        b, d, h, w, c = x.shape
        window_size, shift_size = _get_window_size((d, h, w), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, dp, hp, wp, _ = x.shape

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = _window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask).view(-1, *(window_size + (c,)))
        shifted_x = _window_reverse(attn_windows, window_size, b, dp, hp, wp)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()
        return x

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        shortcut = x
        if use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if use_checkpoint and x.requires_grad:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class _PatchMerging3D(nn.Module):
    def __init__(self, dim: int, norm_layer: Any = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w, _ = x.shape
        if (h % 2 == 1) or (w % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class _BasicLayer3D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        downsample: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample(dim=dim) if downsample is not None else None

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        b, _, d, h, w = x.shape
        window_size, shift_size = _get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = _compute_3d_attn_mask(dp, hp, wp, window_size, shift_size, str(x.device))
        for blk in self.blocks:
            x = blk(x, attn_mask, use_checkpoint=use_checkpoint)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        return rearrange(x, "b d h w c -> b c d h w")


class _PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Any] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, d, h, w = x.size()
        if w % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if d % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            dd, hh, ww = x.size(2), x.size(3), x.size(4)
            x = self.norm(x.flatten(2).transpose(1, 2))
            x = x.transpose(1, 2).view(-1, self.embed_dim, dd, hh, ww)
        return x


class SwinTransformer3D(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.im2vid = _Im2Video()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList(
            [
                _BasicLayer3D(
                    dim=int(embed_dim * 2 ** i),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=_PatchMerging3D if i < self.num_layers - 1 else None,
                )
                for i in range(self.num_layers)
            ]
        )
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = nn.LayerNorm(self.num_features)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)
        return rearrange(x, "n d h w c -> n c d h w")

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> List[torch.Tensor]:
        x = self.im2vid(x)
        x = self.pos_drop(self.patch_embed(x))
        stage_outputs: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x.contiguous(), use_checkpoint=use_checkpoint)
            stage_outputs.append(x)
        stage_outputs[-1] = self._apply_norm(stage_outputs[-1])
        return stage_outputs


class SwinTransformerOmni(nn.Module):
    """Wrapper matching `backbone.trunk.*` naming in official OmniFD checkpoints."""

    def __init__(
        self,
        variant: str = "swin-t",
        with_cp: bool = False,
        drop_path_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        cfg = SWIN_VARIANT_CONFIGS[_normalize_variant(variant)]
        self.with_cp = bool(with_cp)
        self.trunk = SwinTransformer3D(
            patch_size=(2, 4, 4),
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            drop_path_rate=cfg["drop_path_rate"] if drop_path_rate is None else float(drop_path_rate),
            patch_norm=True,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.trunk(x, use_checkpoint=self.with_cp)


# ------------------------------------------------------------------------------
# Interaction Neck
# ------------------------------------------------------------------------------

def _feed_forward(dim: int, mult: int = 4) -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def _get_sinusoid_encoding(n_position: int, embed_dims: int) -> torch.Tensor:
    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(10000.0, -vec).view(1, -1)
    sinusoid_table = torch.arange(n_position, dtype=torch.float64).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()
    sinusoid_table[:, 1::2].cos_()
    return sinusoid_table.to(torch.float32).unsqueeze(0)


class InteractionAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        q = self.to_q(latents)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
        q = rearrange(q, "b t n (h d) -> b h t n d", h=self.heads)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=self.heads)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=self.heads)
        q = q * self.scale
        sim = torch.einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=self.heads)
        return self.to_out(out)


class InteractionModule(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 3,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        InteractionAttention(dim=dim, dim_head=dim_head, heads=heads),
                        _feed_forward(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        latents = repeat(self.latents, "n d -> b t n d", b=b, t=t)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class InteractionNeck(nn.Module):
    """Perceiver-style multi-stage Interaction neck matching `neck.*` in OmniFD."""

    def __init__(
        self,
        latent_channels: int = 512,
        num_latents: int = 64,
        feat_channels: Sequence[int] = (192, 384, 768, 768),
        depth: int = 3,
    ) -> None:
        super().__init__()
        dim = latent_channels
        rev_channels = list(reversed(list(feat_channels)))
        self.module = InteractionModule(dim=dim, depth=depth, num_latents=num_latents)
        self.stage_embed = nn.Parameter(torch.randn(len(rev_channels), 1, dim, 1, 1, 1)) if len(rev_channels) > 1 else None
        self.proj = nn.ModuleList(
            [
                nn.Identity() if (i == 0 and ch == dim) else nn.Sequential(nn.Linear(ch, dim), nn.LayerNorm(dim))
                for i, ch in enumerate(rev_channels)
            ]
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
        tokens: List[torch.Tensor] = []
        for i, (proj, feat) in enumerate(zip(self.proj, reversed(inputs))):
            feat = proj(feat.transpose(1, 4)).transpose(1, 4)
            b, c, t, _, _ = feat.shape
            if t > 1:
                with torch.no_grad():
                    t_emb = _get_sinusoid_encoding(t, c).transpose(1, 2).unsqueeze(-1).unsqueeze(-1).to(
                        device=feat.device, dtype=feat.dtype
                    )
                feat = feat + t_emb
            if self.stage_embed is not None:
                feat = feat + self.stage_embed[i]
            tokens.append(feat.contiguous().permute(0, 2, 1, 3, 4).reshape(b, 1, -1, c))
        x = self.module(torch.cat(tokens, dim=-2))
        latents = x.squeeze(1).contiguous().permute(0, 2, 1).contiguous()  # [B, dim, num_latents]
        return inputs, latents


# ------------------------------------------------------------------------------
# OmniFD Image & Spatial Heads
# ------------------------------------------------------------------------------

class _ConvReLUModule(nn.Module):
    """Matches `mmcv.cnn.ConvModule` attribute layout (`conv` + ReLU, no norm)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.conv(x))


class OmniFDImageClsHead(nn.Module):
    """Matches `head.decoder_modules.image.*` in OmniFD."""

    def __init__(self, in_channels: int = 512, num_classes: int = 2, dropout_ratio: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, feats_and_latents: Tuple[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        _, latents = feats_and_latents
        pooled = latents.mean(dim=2)
        return self.fc(self.dropout(pooled))


class OmniFDSpatialHead(nn.Module):
    """Matches `head.decoder_modules.spatial.*` in OmniFD."""

    def __init__(
        self,
        in_channels: Sequence[int] = (192, 384, 768, 768),
        channels: int = 256,
        latent_channels: int = 512,
        num_latents: int = 64,
        num_classes: int = 1,
        num_convs: int = 2,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.align_corners = align_corners
        total_in = sum(in_channels)
        convs = [_ConvReLUModule(total_in, channels, kernel_size=3, padding=1)]
        for _ in range(num_convs - 1):
            convs.append(_ConvReLUModule(channels, channels, kernel_size=3, padding=1))
        self.convs = nn.Sequential(*convs)
        self.proj = nn.Sequential(
            nn.Linear(latent_channels, channels),
            nn.LayerNorm(channels),
        )
        self.conv_seg = nn.Linear(num_latents, num_classes)

    def forward(self, feats_and_latents: Tuple[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        feats, latents = feats_and_latents
        if feats[0].ndim == 5:
            # For image inputs T=1 -> squeeze(2); for multi-frame videos take temporal mean
            feats_2d = [f.squeeze(2) if f.shape[2] == 1 else f.mean(dim=2) for f in feats]
        else:
            feats_2d = list(feats)
        target_hw = feats_2d[0].shape[2:]
        upsampled = [
            f if f.shape[2:] == target_hw else F.interpolate(
                f, size=target_hw, mode="bilinear", align_corners=self.align_corners
            )
            for f in feats_2d
        ]
        x = self.convs(torch.cat(upsampled, dim=1))
        lat = self.proj(latents.transpose(1, 2)).transpose(1, 2).contiguous()
        lat = self.conv_seg(lat)
        return torch.einsum("bchw,bck->bkhw", x, lat)


class OmniFDUnifiedHead(nn.Module):
    """Container matching `head.decoder_modules.*` in official OmniFD checkpoints."""

    def __init__(
        self,
        feat_channels: Sequence[int] = (192, 384, 768, 768),
        latent_channels: int = 512,
        num_latents: int = 64,
        spatial_channels: int = 256,
        num_classes: int = 2,
        has_image_head: bool = True,
        has_spatial_head: bool = True,
    ) -> None:
        super().__init__()
        modules: Dict[str, nn.Module] = {}
        if has_image_head:
            modules["image"] = OmniFDImageClsHead(in_channels=latent_channels, num_classes=num_classes)
        if has_spatial_head:
            modules["spatial"] = OmniFDSpatialHead(
                in_channels=feat_channels,
                channels=spatial_channels,
                latent_channels=latent_channels,
                num_latents=num_latents,
                num_classes=1,
            )
        self.decoder_modules = nn.ModuleDict(modules)


# ------------------------------------------------------------------------------
# Unified OmniFD Model Wrapper
# ------------------------------------------------------------------------------

class OmniFD(nn.Module):
    """
    OmniFD: Unified Model for Versatile Face Forgery Detection & Localization.

    Supports:
    - Image-level real/fake classification (`image` head -> `class_logits [B, 2]`).
    - Pixel-level spatial forgery localization (`spatial` head -> `mask_logits [B, 1, H, W]`).
    - Loading official Google Drive checkpoints (`omnifd_swin-t_50k_forgerynet`,
      `omnifd_swin-s_50k_forgerynet`, `omnifd_swin-s_20k_ff_spatial`, etc.), local `.pth`/`.pt`
      files, or Hugging Face Hub repositories.
    """

    def __init__(
        self,
        variant: str = "swin-t",
        checkpoint_name: Optional[str] = "omnifd_swin-t_50k_forgerynet",
        checkpoint_path: Optional[str] = None,
        pretrained: bool = True,
        backbone_pretrained: bool = False,
        allow_random_init: bool = False,
        hf_repo: Optional[str] = None,
        cache_dir: Optional[str] = None,
        latent_channels: int = 512,
        num_latents: int = 64,
        spatial_channels: int = 256,
        num_classes: int = 2,
        aux_classifier: Optional[bool] = None,
        has_image_head: Optional[bool] = None,
        has_spatial_head: Optional[bool] = None,
        with_cp: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.checkpoint_name = checkpoint_name
        self.hf_repo = hf_repo

        # Infer variant from checkpoint_name when not explicitly overridden
        if checkpoint_name and "swin-s" in checkpoint_name.lower() and variant == "swin-t":
            variant = "swin-s"
        self.variant = _normalize_variant(variant)
        vcfg = SWIN_VARIANT_CONFIGS[self.variant]
        feat_channels = vcfg["feat_channels"]

        # Determine active heads
        ckpt_meta = OMNIFD_GDRIVE_CHECKPOINTS.get(checkpoint_name or "", {})
        tasks_desc = ckpt_meta.get("tasks", "")
        if has_image_head is None:
            if aux_classifier is not None:
                has_image_head = bool(aux_classifier)
            elif tasks_desc == "spatial":
                has_image_head = False
            else:
                has_image_head = True
        if has_spatial_head is None:
            has_spatial_head = False if tasks_desc == "image" else True

        self.has_image_head = bool(has_image_head)
        self.has_spatial_head = bool(has_spatial_head)
        self.aux_classifier = self.has_image_head
        self.num_classes = int(num_classes)

        self.backbone = SwinTransformerOmni(variant=self.variant, with_cp=with_cp)
        self.neck = InteractionNeck(
            latent_channels=latent_channels,
            num_latents=num_latents,
            feat_channels=feat_channels,
        )
        self.head = OmniFDUnifiedHead(
            feat_channels=feat_channels,
            latent_channels=latent_channels,
            num_latents=num_latents,
            spatial_channels=spatial_channels,
            num_classes=self.num_classes,
            has_image_head=self.has_image_head,
            has_spatial_head=self.has_spatial_head,
        )

        if not pretrained and not allow_random_init:
            raise ValueError(
                "OmniFD requires pretrained=True (or explicit allow_random_init=True). "
                "Random weight initialization is strictly disabled."
            )

        if pretrained:
            self._load_pretrained(
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                cache_dir=cache_dir,
            )
        elif backbone_pretrained:
            self._load_omnivore_trunk(vcfg["checkpoint_name"])

    def _load_omnivore_trunk(self, omnivore_name: str) -> None:
        url = OMNIVORE_TRUNK_URLS.get(omnivore_name)
        if not url:
            return
        try:
            ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
            missing, unexpected = self.backbone.trunk.load_state_dict(ckpt["trunk"], strict=False)
            logger.info(
                f"Loaded Omnivore trunk '{omnivore_name}' (missing={len(missing)}, unexpected={len(unexpected)})"
            )
        except Exception as exc:
            logger.warning(f"Failed to download Omnivore trunk '{omnivore_name}': {exc}")

    def _resolve_checkpoint_path(
        self,
        checkpoint_path: Optional[str],
        checkpoint_name: Optional[str],
        cache_dir: Optional[str],
    ) -> str:
        # 1. Explicit file path
        if checkpoint_path and os.path.isfile(checkpoint_path):
            return os.path.abspath(checkpoint_path)

        # 2. Local search candidates
        target_fn = (
            os.path.basename(checkpoint_path)
            if checkpoint_path and checkpoint_path.endswith((".pth", ".pt"))
            else OMNIFD_GDRIVE_CHECKPOINTS.get(checkpoint_name or "", {}).get("filename", f"{checkpoint_name}.pth")
        )
        candidates = [
            target_fn,
            os.path.join("pretrained", target_fn),
            os.path.join("checkpoints", target_fn),
            os.path.join("weights", target_fn),
            os.path.join("/tmp/omnifd_weights", target_fn),
        ]
        if cache_dir:
            candidates.insert(0, os.path.join(cache_dir, target_fn))
        for cand in candidates:
            if os.path.isfile(cand):
                return os.path.abspath(cand)

        # 3. Hugging Face Hub if specified
        if self.hf_repo:
            try:
                from huggingface_hub import hf_hub_download
                for c_fn in [target_fn, "checkpoints/checkpoint_best.pt", "checkpoint_best.pt"]:
                    try:
                        dl = hf_hub_download(repo_id=self.hf_repo, filename=c_fn, cache_dir=cache_dir)
                        if dl and os.path.isfile(dl):
                            return dl
                    except Exception:
                        continue
            except Exception as hf_err:
                logger.warning(f"HF Hub lookup failed for '{self.hf_repo}': {hf_err}")

        # 4. Official Google Drive download via gdown
        meta = OMNIFD_GDRIVE_CHECKPOINTS.get(checkpoint_name or "")
        if meta is not None:
            dest_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "omnifd")
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, meta["filename"])
            if os.path.isfile(dest_path):
                return dest_path
            try:
                import gdown
                out = gdown.download(id=meta["file_id"], output=dest_path, quiet=False)
                if out and os.path.isfile(out):
                    return out
            except Exception as gd_err:
                raise RuntimeError(
                    f"Failed to download OmniFD checkpoint '{checkpoint_name}' from Google Drive "
                    f"(https://drive.google.com/file/d/{meta['file_id']}/view, folder: {OMNIFD_GDRIVE_FOLDER_URL}): "
                    f"{gd_err}"
                ) from gd_err

        raise RuntimeError(
            f"Unable to resolve OmniFD checkpoint (checkpoint_path={checkpoint_path}, "
            f"checkpoint_name={checkpoint_name}). Official weights folder: {OMNIFD_GDRIVE_FOLDER_URL}"
        )

    def _load_pretrained(
        self,
        checkpoint_path: Optional[str],
        checkpoint_name: Optional[str],
        cache_dir: Optional[str],
    ) -> None:
        resolved = self._resolve_checkpoint_path(checkpoint_path, checkpoint_name, cache_dir)
        ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid OmniFD checkpoint payload at '{resolved}'")
        raw_sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
        clean_sd: Dict[str, Any] = {}
        for k, v in raw_sd.items():
            ck = k[7:] if k.startswith("module.") else k
            # Drop video/temporal heads when only image/spatial modules are instantiated
            if ck.startswith(("head.decoder_modules.video.", "head.decoder_modules.temporal.")):
                continue
            clean_sd[ck] = v

        # Auto-align active heads with checkpoint contents when loading an external file
        expected = set(self.state_dict().keys())
        missing = [k for k in expected if k not in clean_sd and "relative_position_index" not in k]
        if missing:
            raise RuntimeError(f"OmniFD missing required weights from '{resolved}': {missing[:10]}")

        res_missing, res_unexpected = self.load_state_dict(clean_sd, strict=False)
        logger.info(
            f"Loaded OmniFD checkpoint '{resolved}' "
            f"(missing={len(res_missing)}, unexpected={len(res_unexpected)})"
        )

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        feats = self.backbone(x)
        neck_out = self.neck(feats)

        mask_logits: Optional[torch.Tensor] = None
        if "spatial" in self.head.decoder_modules:
            seg_small = self.head.decoder_modules["spatial"](neck_out)
            mask_logits = F.interpolate(
                seg_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
        else:
            # Image-only variant fallback: zero spatial map on matching device/dtype
            mask_logits = torch.zeros((x.shape[0], 1, orig_h, orig_w), device=x.device, dtype=x.dtype)

        class_logits: Optional[torch.Tensor] = None
        if "image" in self.head.decoder_modules:
            class_logits = self.head.decoder_modules["image"](neck_out)

        if return_dict:
            out: Dict[str, Any] = {
                "mask_logits": mask_logits,
                "pred_mask": torch.sigmoid(mask_logits),
                "latents": neck_out[1],
            }
            if class_logits is not None:
                out["class_logits"] = class_logits
                out["pred_label_probs"] = torch.softmax(class_logits, dim=-1)
            return out

        if class_logits is not None:
            return mask_logits, class_logits
        return mask_logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_dict=True)
            return (out["pred_mask"] >= threshold).float()
