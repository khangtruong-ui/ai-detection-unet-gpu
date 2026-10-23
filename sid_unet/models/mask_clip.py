"""
OpenSDI MaskCLIP Model Integration for SID-UNet.

Implements MaskCLIP based on the Synergizing Pretrained Models (SPM) framework from:
  "OpenSDI: Spotting Diffusion-Generated Images in the Open World" (CVPR 2025)
  Yabin Wang, Zhiwu Huang, Xiaopeng Hong (arXiv:2503.19653v3)
  Official Repository: https://github.com/iamwangyabin/OpenSDI
  Official Pretrained Weights: https://huggingface.co/nebula/MaskCLIP-weights

Key Architectural Components (SPM Framework):
  1. Frozen CLIP (ViT-L/14) Semantic Stream with multi-layer FeatureHooks at `ln_2`
     extracting intermediate CLS & spatial patch tokens from layers [2, 4, 8, 12, 16, 20, 23].
  2. PromptLearner (CoOp/MaPLe-style continuous context tokens `ctx_pos` and `ctx_neg`)
     encoded through the frozen CLIP text transformer.
  3. Pretrained MAE (Windowed ViT-B/16) Dense Spatial Stream with decomposed relative
     positional embeddings and windowed attention.
  4. Visual Cross-Attention (VCA / `AttentionFusion`) injecting multi-layer CLIP patch
     representations into MAE blocks {1..7}.
  5. Multi-Scale FPN + Textual-Visual Cross-Attention (TVCA / `MaskDecoder`) fusing
     4 spatial pyramid scales (4.0x, 2.0x, 1.0x, 0.5x) with fake text embeddings.
  6. Visual Self-Attention (VSA / `CLSAttentionAggregator`) aggregating multi-layer CLIP
     CLS tokens for image-level detection.
  7. Morphological EdgeMaskGenerator and edge-weighted BCE loss (`edge_lambda = 20.0`).

Pretrained Weights Policy:
  - Hardcoded online pretrained weight URLs and Hugging Face Hub identifiers are used by default.
  - Random initialization is strictly disallowed by default (`allow_random_init=False`).
  - Supports both full OpenSDI pretrained checkpoints (`nebula/MaskCLIP-weights`) and
    online CLIP (`ViT-L/14`) + MAE (`mae_pretrain_vit_base.pth`) foundation weights.

Model Surgery Readiness:
  - First-class APIs for spatial/semantic backbone replacement, VCA layer tap & fusion map
    rewiring, PEFT LoRA/DoRA injection, granular submodule freezing, forward intervention hooks,
    and WiSE-FT / Task Arithmetic weight-space merging.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from functools import partial
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, Mlp, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, Mlp, trunc_normal_  # type: ignore

logger = logging.getLogger(__name__)

# ============================================================================
# Hardcoded Online Pretrained Weight Sources (No Random Initialization)
# ============================================================================

DEFAULT_MASKCLIP_HF_REPO = "nebula/MaskCLIP-weights"
DEFAULT_MASKCLIP_HF_FILENAME = "MaskCLIP_sd15_20241108_09_14_22.pth"
ALT_MASKCLIP_HF_FILENAMES = (
    "MaskCLIP_sd15_20241108_09_14_22.pth",
    "MaskCLIP_sd15_20241109_08_53_19.pth",
    "MaskCLIP_sd15_20241103_17_45_16.pth",
    "MaskCLIP_sd15_20241103_17_45_19.pth",
)
SIDECLIP_1024_FILENAMES = (
    "MaskCLIP_sd15_20241103_17_45_16.pth",
    "MaskCLIP_sd15_20241103_17_45_19.pth",
)

DEFAULT_MAE_PRETRAIN_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
DEFAULT_CLIP_VIT_L14_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
)
DEFAULT_CLIP_MODEL_NAME = "ViT-L/14"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

MASKCLIP_SETTINGS: Dict[str, Dict[str, Any]] = {
    "ViTL": {
        "model_name": DEFAULT_CLIP_MODEL_NAME,
        "resolution": 512,
        "language_ctx": 10,
        "language_depth": 5,
        "adapter_length": 5,
        "selected_layers": [2, 4, 8, 12, 16, 20, 23],
        "fusion_map": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},
        "decoder_variant": "spm_fpn_tvca",
        "pretrain_img_size": 224,
        "pretrain_use_cls_token": True,
    },
    "ViTB": {
        "model_name": "ViT-B/16",
        "resolution": 512,
        "language_ctx": 10,
        "language_depth": 5,
        "adapter_length": 5,
        "selected_layers": [1, 3, 5, 7, 9, 11],
        "fusion_map": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
        "decoder_variant": "spm_fpn_tvca",
        "pretrain_img_size": 224,
        "pretrain_use_cls_token": True,
    },
    "ViTL_1024_SideCLIP": {
        "model_name": DEFAULT_CLIP_MODEL_NAME,
        "resolution": 1024,
        "language_ctx": 10,
        "language_depth": 5,
        "adapter_length": 5,
        "selected_layers": [5, 11, 17, 23],
        "fusion_map": {0: 0, 3: 1, 6: 2, 9: 3},
        "decoder_variant": "sideclip_mlp_refiner",
        "pretrain_img_size": 1024,
        "pretrain_use_cls_token": False,
    },
}


# ============================================================================
# Morphological Edge Mask Generator & Losses (OpenSDI / IMDLBenCo compatible)
# ============================================================================


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute soft Sørensen-Dice loss on probability predictions."""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class EdgeMaskGenerator(nn.Module):
    """Generate broadened morphological edge masks from binary ground-truth masks.

    Implements morphological dilation and erosion with a cross-shaped structuring kernel
    (default width 7 in OpenSDI) to emphasize manipulation boundaries during training.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"EdgeMaskGenerator kernel_size must be a positive odd integer, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        kernel = torch.zeros((1, 1, self.kernel_size, self.kernel_size), dtype=torch.float32)
        mid = self.kernel_size // 2
        kernel[0, 0, mid : mid + 1, :] = 1.0
        kernel[0, 0, :, mid : mid + 1] = 1.0
        self.register_buffer("kernel", kernel, persistent=False)

    def _dilate(self, mask: torch.Tensor) -> torch.Tensor:
        k = self.kernel.to(device=mask.device, dtype=mask.dtype)
        res = F.conv2d(mask, k, stride=1, padding=self.kernel_size // 2)
        return (res > 0).to(mask.dtype)

    @torch.no_grad()
    def forward(self, mask: torch.Tensor, return_all: bool = False):
        orig_ndim = mask.ndim
        if orig_ndim == 2:
            x = mask.unsqueeze(0).unsqueeze(0).float()
        elif orig_ndim == 3:
            x = mask.unsqueeze(1).float()
        else:
            x = mask.float()

        x_bin = (x > 0.5).float()
        dilated = self._dilate(x_bin)
        erosion = self._dilate(1.0 - x_bin)
        diff = (-torch.abs(erosion - dilated) + 1.0 > 0).float()

        if return_all:
            return diff, dilated, erosion
        return diff


# ============================================================================
# Windowed MAE Vision Transformer Backbone (model/mae.py)
# ============================================================================


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Extract relative positional embeddings according to query and key spatial sizes."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """Add decomposed Relative Positional Embeddings (MViTv2 / ViTDet style)."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn


class Attention(nn.Module):
    """Multi-head Attention block with decomposed relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("input_size must be provided when use_rel_pos=True")
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition [B, H, W, C] into non-overlapping windows with padding if needed."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Unpartition windows back into [B, H, W, C] sequences and remove padding."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Block(nn.Module):
    """Transformer block supporting windowed attention and relative positional embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding producing [B, H, W, C]."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)


def get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: Tuple[int, int]) -> torch.Tensor:
    """Interpolate absolute positional embeddings to target spatial grid (h, w)."""
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, h, w, -1)


class ViT(nn.Module):
    """Windowed Vision Transformer (MAE ViT-B/16) spatial encoder for MaskCLIP."""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        window_block_indexes: Sequence[int] = (),
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.output_size = (img_size // patch_size, img_size // patch_size)
        self.stride = patch_size
        self.channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2]))
        for blk in self.blocks:
            x = blk(x)
        return x.permute(0, 3, 1, 2)


# ============================================================================
# SPM Submodules: LayerNorm, VCA (AttentionFusion), FPN+TVCA (MaskDecoder),
# VSA (CLSAttentionAggregator), PromptLearner (model/clip_utils.py & prompt_learner.py)
# ============================================================================


class LayerNorm(nn.Module):
    """2D Channel-first LayerNorm over [B, C, H, W] tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskDecoder(nn.Module):
    """Multi-scale Feature Pyramid Network (FPN) + Textual-Visual Cross-Attention (TVCA) decoder."""

    def __init__(
        self,
        in_channels: int = 768,
        decoder_dim: int = 256,
        norm: str = "LN",
        num_heads: int = 8,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        backbone_dim = in_channels
        self.in_channels = in_channels
        self.decoder_dim = decoder_dim
        self.out_channels = out_channels
        scale_factors = (4.0, 2.0, 1.0, 0.5)

        # Textual-Visual Cross-Attention (TVCA) components
        self.num_heads = num_heads
        self.scale = (decoder_dim // num_heads) ** -0.5
        self.to_q = nn.Linear(decoder_dim, decoder_dim)
        self.to_kv = nn.Linear(decoder_dim, decoder_dim * 2)
        self.proj = nn.Linear(decoder_dim, decoder_dim)

        # Multi-scale FPN stages
        self.stages = nn.ModuleList()
        for scale in scale_factors:
            layers: List[nn.Module] = []
            if scale == 4.0:
                layers.extend([
                    nn.ConvTranspose2d(backbone_dim, backbone_dim // 2, kernel_size=2, stride=2),
                    LayerNorm(backbone_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(backbone_dim // 2, backbone_dim // 4, kernel_size=2, stride=2),
                ])
                cur_channels = backbone_dim // 4
            elif scale == 2.0:
                layers.append(nn.ConvTranspose2d(backbone_dim, backbone_dim // 2, kernel_size=2, stride=2))
                cur_channels = backbone_dim // 2
            elif scale == 1.0:
                cur_channels = backbone_dim
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                cur_channels = backbone_dim

            layers.extend([
                nn.Conv2d(cur_channels, decoder_dim, 1),
                LayerNorm(decoder_dim),
                nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
                LayerNorm(decoder_dim),
            ])
            self.stages.append(nn.Sequential(*layers))

        self.linear_fuse = nn.Conv2d(decoder_dim * len(scale_factors), decoder_dim, 1)

        if norm == "LN":
            self.norm: nn.Module = LayerNorm(decoder_dim)
        elif norm == "BN":
            self.norm = nn.BatchNorm2d(decoder_dim)
        else:
            self.norm = nn.InstanceNorm2d(decoder_dim, track_running_stats=True, affine=True)

        self.dropout = nn.Dropout(0.1)
        self.predict = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

    def cross_attention(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        query = query.unsqueeze(1)  # [B, 1, C]

        q = self.to_q(query)
        k, v = self.to_kv(x_flat).chunk(2, dim=-1)

        q = q.reshape(B, query.shape[1], self.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, H * W, self.num_heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2)
        out = out.reshape(B, query.shape[1], C)
        out = self.proj(out)

        return out.reshape(B, C, 1, 1).expand(B, C, H, W)

    def forward(self, query: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        features = [stage(x) for stage in self.stages]
        _, _, h, w = features[0].shape
        aligned_features = [
            F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
            for feat in features
        ]
        x = self.linear_fuse(torch.cat(aligned_features, dim=1))

        attn_out = self.cross_attention(x, query)
        x = x + attn_out

        x = self.norm(x)
        x = self.dropout(x)
        x = self.predict(x)
        return x


class _SideCLIPMLP(nn.Module):
    """MLP helper matching legacy 1024-resolution SideCLIP checkpoints."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3, conv: bool = False) -> None:
        super().__init__()
        self.conv = conv
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        if conv:
            self.layers = nn.ModuleList([nn.Conv2d(dims[i], dims[i + 1], kernel_size=1) for i in range(num_layers)])
        else:
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class SideCLIPMaskDecoder(nn.Module):
    """Legacy 1024-resolution SideCLIP mask decoder for `MaskCLIP_sd15_20241103_17_45_16.pth`."""

    def __init__(self, in_channels: int = 768, decoder_dim: int = 256) -> None:
        super().__init__()
        self.query_mlp = _SideCLIPMLP(in_channels, decoder_dim, decoder_dim, num_layers=3, conv=False)
        self.pix_mlp = _SideCLIPMLP(in_channels, decoder_dim, decoder_dim, num_layers=3, conv=True)
        self.refiner = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=2, stride=2),
            LayerNorm(decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=2, stride=2),
            LayerNorm(decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
        )

    def forward(self, query: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = self.query_mlp(query)  # [B, C]
        pix = self.pix_mlp(x)
        pix = self.refiner(pix)  # [B, C, 4H, 4W]
        return torch.einsum("bc,bchw->bhw", q, pix).unsqueeze(1)


class SideCLIPProjectionFusion(nn.Module):
    """Legacy 1024-resolution SideCLIP fusion block for `MaskCLIP_sd15_20241103_17_45_16.pth`."""

    def __init__(self, clip_dim: int = 1024, x_dim: int = 768) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            LayerNorm(clip_dim),
            nn.Conv2d(clip_dim, x_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, clip_features: torch.Tensor) -> torch.Tensor:
        target_h, target_w = x.shape[1:3]
        proj = self.input_proj(clip_features)
        resized = F.interpolate(proj, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1)


class AttentionFusion(nn.Module):
    """Visual Cross-Attention (VCA) fusing multi-layer CLIP patch features into MAE blocks."""

    def __init__(self, clip_dim: int = 1024, x_dim: int = 768, num_heads: int = 8) -> None:
        super().__init__()
        self.clip_dim = clip_dim
        self.x_dim = x_dim
        self.num_heads = num_heads
        self.head_dim = x_dim // num_heads
        if self.head_dim * num_heads != x_dim:
            raise ValueError(f"x_dim ({x_dim}) must be divisible by num_heads ({num_heads})")

        self.channel_adjust = nn.Sequential(
            nn.Conv2d(clip_dim, x_dim, kernel_size=1),
            LayerNorm(x_dim),
        )

        self.query_proj = nn.Linear(x_dim, x_dim)
        self.key_proj = nn.Linear(x_dim, x_dim)
        self.value_proj = nn.Linear(x_dim, x_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.LayerNorm(x_dim),
        )

        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, clip_features: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        target_h, target_w = x.shape[1:3]

        clip_resized = F.interpolate(
            clip_features,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        clip_adjusted = self.channel_adjust(clip_resized)
        clip_adjusted = clip_adjusted.permute(0, 2, 3, 1)  # [B, H, W, x_dim]

        seq_len = target_h * target_w
        query = clip_adjusted.reshape(batch_size, seq_len, -1)
        key = x.reshape(batch_size, seq_len, -1)
        value = x.reshape(batch_size, seq_len, -1)

        query = self.query_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key_proj(key).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value_proj(value).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(attention_weights, value)
        attended_features = attended_features.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        fused = self.output_proj(attended_features)
        fused = fused.reshape(batch_size, target_h, target_w, -1)
        return fused


def load_online_mae_weights(
    url_or_path: str = DEFAULT_MAE_PRETRAIN_URL,
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Download and load pretrained MAE ViT-Base weights online."""
    if os.path.isfile(url_or_path):
        ckpt = torch.load(url_or_path, map_location="cpu", weights_only=False)
    else:
        ckpt = torch.hub.load_state_dict_from_url(
            url_or_path,
            model_dir=cache_dir,
            map_location="cpu",
            check_hash=False,
        )
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


class SideAdapterNetwork(nn.Module):
    """SPM SideAdapterNetwork combining Windowed MAE ViT, VCA (AttentionFusion), and MaskDecoder."""

    def __init__(
        self,
        img_size: int = 512,
        align_channels: int = 768,
        clip_channels: int = 1024,
        fusion_map: Optional[Dict[int, int]] = None,
        mae_pretrained_url: Optional[str] = DEFAULT_MAE_PRETRAIN_URL,
        load_mae_online: bool = True,
        decoder_variant: str = "spm_fpn_tvca",
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        out_channels: int = 1,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.align_channels = align_channels
        self.clip_channels = clip_channels
        self.decoder_variant = decoder_variant
        self.fusion_map: Dict[int, int] = dict(fusion_map or {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6})

        use_rel_pos = decoder_variant != "sideclip_mlp_refiner"
        self.vit_model = ViT(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
            use_rel_pos=use_rel_pos,
            pretrain_img_size=pretrain_img_size,
            pretrain_use_cls_token=pretrain_use_cls_token,
        )

        if load_mae_online and mae_pretrained_url:
            mae_state = load_online_mae_weights(mae_pretrained_url, cache_dir=cache_dir)
            self.vit_model.load_state_dict(mae_state, strict=False)

        if decoder_variant == "sideclip_mlp_refiner":
            self.text_projection = nn.Linear(align_channels, 768)
            self.fusion_layers = nn.ModuleDict(
                {
                    f"layer_{tgt_idx}": SideCLIPProjectionFusion(clip_channels, 768)
                    for tgt_idx in self.fusion_map.keys()
                }
            )
            self.mask_decoder: nn.Module = SideCLIPMaskDecoder(in_channels=768, decoder_dim=256)
        else:
            self.text_projection = nn.Linear(align_channels, 256)
            self.fusion_layers = nn.ModuleDict(
                {
                    f"layer_{tgt_idx}": AttentionFusion(clip_channels, 768)
                    for tgt_idx in self.fusion_map.keys()
                }
            )
            self.mask_decoder = MaskDecoder(
                in_channels=768,
                decoder_dim=256,
                out_channels=out_channels,
            )

        # Optional custom spatial backbone adapter hook for model surgery
        self.custom_spatial_adapter: Optional[nn.Module] = None

    def forward(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
        clip_features: List[torch.Tensor],
        block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        decoder_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        image_features = self.forward_features(image, clip_features, block_hook=block_hook)
        if decoder_hook is not None:
            image_features = decoder_hook(image_features)
        # Select the fake class text token [:, 1, :]
        projected_text = self.text_projection(text_features.expand(image_features.shape[0], -1, -1))[:, 1, :]
        return self.mask_decoder(projected_text, image_features)

    def forward_features(
        self,
        image: torch.Tensor,
        clip_features: List[torch.Tensor],
        block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        if hasattr(self.vit_model, "patch_embed") and hasattr(self.vit_model, "blocks"):
            x = self.vit_model.patch_embed(image)
            if getattr(self.vit_model, "pos_embed", None) is not None:
                x = x + get_abs_pos(
                    self.vit_model.pos_embed,
                    getattr(self.vit_model, "pretrain_use_cls_token", True),
                    (x.shape[1], x.shape[2]),
                )
            for i, blk in enumerate(self.vit_model.blocks):
                x = blk(x)
                x = self.fuse(i, x, clip_features)
                if block_hook is not None:
                    x = block_hook(i, x)
            out = x.permute(0, 3, 1, 2)
        else:
            # Generic replaced spatial backbone (from model surgery)
            out = self.vit_model(image)
            if isinstance(out, (tuple, list)):
                out = out[-1]
            if out.ndim == 4 and out.shape[-1] == getattr(self, "replaced_backbone_channels", -1):
                out = out.permute(0, 3, 1, 2)
            if self.custom_spatial_adapter is not None:
                out = self.custom_spatial_adapter(out)
            # Apply VCA fusions over spatial feature map [B, C, H, W] -> [B, H, W, C]
            x_hwc = out.permute(0, 2, 3, 1)
            for i in sorted(self.fusion_map.keys()):
                x_hwc = self.fuse(i, x_hwc, clip_features)
                if block_hook is not None:
                    x_hwc = block_hook(i, x_hwc)
            out = x_hwc.permute(0, 3, 1, 2)
        return out

    def fuse(self, block_idx: int, x: torch.Tensor, clip_features: List[torch.Tensor]) -> torch.Tensor:
        if block_idx in self.fusion_map:
            src_idx = self.fusion_map[block_idx]
            if src_idx < len(clip_features):
                layer_key = f"layer_{block_idx}"
                if layer_key in self.fusion_layers:
                    x = self.fusion_layers[layer_key](x, clip_features[src_idx]) + x
        return x


class CLSAttentionAggregator(nn.Module):
    """Visual Self-Attention (VSA) aggregating multi-layer CLIP [CLS] tokens."""

    def __init__(
        self,
        feature_dim: int = 1024,
        out_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "multihead_attn": nn.MultiheadAttention(
                    embed_dim=feature_dim, num_heads=num_heads, batch_first=True
                ),
                "layer_norm": nn.LayerNorm(feature_dim),
                "dropout": nn.Dropout(dropout_rate),
            })
            for _ in range(num_layers)
        ])

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, num_layers_inp, feature_dim = x.size()
        x = x.view(bs, num_layers_inp, feature_dim)

        for layer in self.layers:
            attn_output, _ = layer["multihead_attn"](x, x, x)
            x = layer["layer_norm"](attn_output + x)
            x = layer["dropout"](x)

        x = torch.mean(x, dim=1)
        x = self.projector(x)
        return x


class FeatureHook:
    """Forward hook capturing intermediate outputs from CLIP visual transformer layers."""

    def __init__(self, name: str, module: nn.Module) -> None:
        self.name = name
        self.input: Any = None
        self.output: Any = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module: nn.Module, input: Any, output: Any) -> None:
        self.input = input
        self.output = output

    def close(self) -> None:
        self.hook.remove()


# ============================================================================
# PromptLearner & Tokenizer (model/prompt_learner.py)
# ============================================================================

_SIMPLE_TOKENIZER: Any = None


def _get_tokenizer() -> Any:
    global _SIMPLE_TOKENIZER
    if _SIMPLE_TOKENIZER is None:
        from clip.simple_tokenizer import SimpleTokenizer
        _SIMPLE_TOKENIZER = SimpleTokenizer()
    return _SIMPLE_TOKENIZER


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
) -> torch.IntTensor:
    """Tokenize input strings using OpenAI CLIP's SimpleTokenizer."""
    if isinstance(texts, str):
        texts = [texts]

    tokenizer = _get_tokenizer()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.int)
    return result  # type: ignore[return-value]


class PromptLearner(nn.Module):
    """Learnable positive (fake) and negative (real) continuous context prompt vectors."""

    def __init__(
        self,
        ctx_dim: int = 768,
        language_length: int = 10,
        language_depth: int = 5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.ctx_dim = ctx_dim
        self.n_ctx = language_length
        self.text_encoder_n_ctx = language_depth
        self.ctx_pos = nn.Parameter(torch.empty(1, self.n_ctx, ctx_dim))
        self.ctx_neg = nn.Parameter(torch.empty(1, self.n_ctx, ctx_dim))
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.ctx_neg, std=0.02)
        self.prompt_prefix = " ".join(["X"] * self.n_ctx)
        self.dtype = dtype

    def forward(
        self,
        clip_model: nn.Module,
        text_input: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompts = [self.prompt_prefix + " " + text + "." for text in text_input]
        tokenized_list = [tokenize(p) for p in prompts]
        tokenized_prompts = torch.cat(tokenized_list, dim=0).to(device)

        with torch.no_grad():
            embedding_text = clip_model.token_embedding(tokenized_prompts).to(self.dtype)

        token_prefix = embedding_text[:, :1, :]
        token_suffix = embedding_text[:, 1 + self.n_ctx :, :]

        ctx_pos = self.ctx_pos.to(device=device, dtype=embedding_text.dtype)
        ctx_neg = self.ctx_neg.to(device=device, dtype=embedding_text.dtype)

        prompts_pos = torch.cat([token_prefix, ctx_pos.expand(len(text_input), -1, -1), token_suffix], dim=1)
        prompts_neg = torch.cat([token_prefix, ctx_neg.expand(len(text_input), -1, -1), token_suffix], dim=1)

        full_prompts = torch.cat([prompts_neg, prompts_pos], dim=0)
        full_tokenized = torch.cat([tokenized_prompts, tokenized_prompts], dim=0)
        return full_prompts, full_tokenized


class TextEncoder(nn.Module):
    """Wrapper around CLIP text transformer for encoding continuous prompt embeddings."""

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.transformer = clip_model.transformer.to(clip_model.dtype)
        self.positional_embedding = clip_model.positional_embedding.to(clip_model.dtype)
        self.ln_final = clip_model.ln_final.to(clip_model.dtype)
        self.text_projection = clip_model.text_projection.to(clip_model.dtype)

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x)
        if isinstance(out, tuple):
            x = out[0]
        else:
            x = out
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


# ============================================================================
# Native Fallback LoRA Module (when PEFT or custom Conv/Linear surgery is used)
# ============================================================================


class SurgicalLoRALinear(nn.Module):
    """Lightweight Low-Rank / Weight-Decomposed Low-Rank (DoRA) Linear wrapper for Model Surgery."""

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_dora: bool = False,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.use_dora = use_dora

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=base_linear.weight.device, dtype=base_linear.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=base_linear.weight.device, dtype=base_linear.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        if self.use_dora:
            with torch.no_grad():
                weight_norm = base_linear.weight.norm(p=2, dim=1, keepdim=True)
            self.dora_m = nn.Parameter(weight_norm.clone())
        else:
            self.register_parameter("dora_m", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dora and self.dora_m is not None:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            combined_w = self.base_linear.weight + delta_w
            norm = combined_w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
            effective_w = (self.dora_m / norm) * combined_w
            return F.linear(x, effective_w, self.base_linear.bias)

        base_out = self.base_linear(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out


# ============================================================================
# Main MaskCLIP Model Class
# ============================================================================


_CLIP_PROTOTYPE_CACHE: Dict[str, nn.Module] = {}
_OPENSDI_STATE_DICT_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def _load_clip_backbone_online(
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
    download_root: Optional[str] = None,
) -> nn.Module:
    """Load OpenAI CLIP model online (hardcoded URL via `clip.load`)."""
    cache_key = f"{model_name}:{download_root or ''}"
    if cache_key in _CLIP_PROTOTYPE_CACHE:
        return copy.deepcopy(_CLIP_PROTOTYPE_CACHE[cache_key])

    import clip

    clip_model, _ = clip.load(model_name, device="cpu", download_root=download_root)
    clip_model.float()
    _CLIP_PROTOTYPE_CACHE[cache_key] = clip_model
    return copy.deepcopy(clip_model)


def _download_opensdi_checkpoint(
    repo_id_or_path: str = DEFAULT_MASKCLIP_HF_REPO,
    filename: Optional[str] = DEFAULT_MASKCLIP_HF_FILENAME,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """Resolve or download an OpenSDI MaskCLIP checkpoint from Hugging Face Hub or local path."""
    if os.path.isfile(repo_id_or_path):
        return os.path.abspath(repo_id_or_path)

    from sid_unet.utils.checkpoint import parse_hf_repo_uri

    repo_id, parsed_fname = parse_hf_repo_uri(repo_id_or_path)
    target_fname = parsed_fname or filename or DEFAULT_MASKCLIP_HF_FILENAME

    from huggingface_hub import hf_hub_download

    effective_cache = cache_dir or os.environ.get("HF_HUB_CACHE") or (
        os.path.join(os.environ["HF_HOME"], "hub") if "HF_HOME" in os.environ else None
    )
    return hf_hub_download(
        repo_id=repo_id,
        filename=target_fname,
        cache_dir=effective_cache,
        token=token,
        force_download=force_download,
    )


class MaskCLIP(nn.Module):
    """
    OpenSDI MaskCLIP Model (Synergizing Pretrained Models: CLIP + MAE) for
    AI-Generated / Diffusion-Manipulated Image Detection and Localization.

    Integrates natively with `sid_unet` training, evaluation, prediction, caching,
    and supports OpenSDI's `forward_opensdi` interface as well as rich Model Surgery APIs.
    """

    def __init__(
        self,
        model_setting_name: str = "ViTL",
        pretrained: bool = True,
        pretrained_mode: str = "opensdi",
        pretrained_model_name_or_path: Optional[str] = DEFAULT_MASKCLIP_HF_REPO,
        checkpoint_filename: Optional[str] = DEFAULT_MASKCLIP_HF_FILENAME,
        mae_pretrained_url: str = DEFAULT_MAE_PRETRAIN_URL,
        clip_model_name: Optional[str] = None,
        allow_random_init: bool = False,
        resolution: Optional[int] = None,
        selected_layers: Optional[Sequence[int]] = None,
        fusion_map: Optional[Dict[int, int]] = None,
        language_ctx: Optional[int] = None,
        language_depth: Optional[int] = None,
        aux_classifier: bool = True,
        num_classes: int = 2,
        in_channels: int = 3,
        out_channels: int = 1,
        edge_lambda: float = 20.0,
        edge_mask_width: int = 7,
        input_normalize: bool = True,
        freeze_clip: bool = True,
        freeze_mae: bool = False,
        enable_clip_cache: bool = False,
        clip_cache_size: int = 10000,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target: str = "mae",
        use_dora: bool = False,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if not pretrained and not allow_random_init:
            raise ValueError(
                "Random initialization is strictly disabled for MaskCLIP (allow_random_init=False). "
                "Pretrained weights must be loaded online (set pretrained=True)."
            )

        self.allow_random_init = bool(allow_random_init)
        self.pretrained = bool(pretrained)
        self.pretrained_mode = str(pretrained_mode).lower().strip()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path or DEFAULT_MASKCLIP_HF_REPO
        self.checkpoint_filename = checkpoint_filename or DEFAULT_MASKCLIP_HF_FILENAME
        self.mae_pretrained_url = mae_pretrained_url or DEFAULT_MAE_PRETRAIN_URL

        # Auto-detect 1024 SideCLIP setting if the user requested one of the Nov 3 checkpoints
        effective_setting_name = model_setting_name
        for legacy_name in SIDECLIP_1024_FILENAMES:
            if (self.checkpoint_filename and legacy_name in self.checkpoint_filename) or (
                self.pretrained_model_name_or_path and legacy_name in str(self.pretrained_model_name_or_path)
            ):
                if effective_setting_name == "ViTL":
                    effective_setting_name = "ViTL_1024_SideCLIP"
                break

        # Auto-detect ViTB profile if user specifies a ViT-B model while keeping default ViTL setting
        if (
            clip_model_name is not None
            and "vit-b" in clip_model_name.lower()
            and effective_setting_name == "ViTL"
        ):
            effective_setting_name = "ViTB"

        if effective_setting_name not in MASKCLIP_SETTINGS:
            effective_setting_name = "ViTL"
        self.model_setting_name = effective_setting_name
        settings = copy.deepcopy(MASKCLIP_SETTINGS[effective_setting_name])

        if clip_model_name is not None:
            settings["model_name"] = clip_model_name
        if resolution is not None:
            settings["resolution"] = int(resolution)
        if selected_layers is not None:
            settings["selected_layers"] = [int(i) for i in selected_layers]
        if fusion_map is not None:
            settings["fusion_map"] = {int(k): int(v) for k, v in fusion_map.items()}
        if language_ctx is not None:
            settings["language_ctx"] = int(language_ctx)
        if language_depth is not None:
            settings["language_depth"] = int(language_depth)

        self.settings = settings
        self.selected_layers: List[int] = list(settings["selected_layers"])
        self.resolution: int = int(settings["resolution"])
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.aux_classifier = bool(aux_classifier)
        self.num_classes = int(num_classes)
        self.edge_lambda = float(edge_lambda)
        self.edge_mask_width = int(edge_mask_width)
        self.input_normalize = bool(input_normalize)

        # Register CLIP normalization buffers
        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # 1. Load OpenAI CLIP Backbone (online hardcoded URL)
        self.clip = _load_clip_backbone_online(
            model_name=settings["model_name"],
            download_root=cache_dir,
        )
        self.clip.float()

        self.hooks: List[FeatureHook] = []
        self._register_clip_hooks()

        visemb_dim = int(self.clip.visual.conv1.out_channels)
        align_dim = int(self.clip.ln_final.weight.shape[0])
        self.visemb_dim = visemb_dim
        self.align_dim = align_dim

        # Determine whether to pre-load standalone MAE weights before loading full OpenSDI checkpoint
        load_full_opensdi = self.pretrained and self.pretrained_mode in (
            "opensdi",
            "full",
            "hf",
            "huggingface",
            "checkpoint",
        )
        load_mae_standalone = self.pretrained and (
            not load_full_opensdi or self.pretrained_mode in ("clip_mae", "backbones", "mae", "both")
        )

        # 2. Build SideAdapterNetwork (MAE ViT-B/16 + VCA AttentionFusion + FPN/TVCA MaskDecoder)
        self.aggregator = SideAdapterNetwork(
            img_size=self.resolution,
            align_channels=align_dim,
            clip_channels=visemb_dim,
            fusion_map=settings["fusion_map"],
            mae_pretrained_url=self.mae_pretrained_url,
            load_mae_online=load_mae_standalone,
            decoder_variant=settings.get("decoder_variant", "spm_fpn_tvca"),
            pretrain_img_size=settings.get("pretrain_img_size", 224),
            pretrain_use_cls_token=settings.get("pretrain_use_cls_token", True),
            out_channels=self.out_channels,
            cache_dir=cache_dir,
        )

        # 3. Build VSA (CLSAttentionAggregator) & PromptLearner
        self.cls_aggregator = CLSAttentionAggregator(
            feature_dim=visemb_dim,
            out_dim=align_dim,
            num_heads=8,
            num_layers=1,
            dropout_rate=0.1,
        )
        self.prompt_learner = PromptLearner(
            ctx_dim=align_dim,
            language_length=settings["language_ctx"],
            language_depth=settings["language_depth"],
            dtype=self.clip.dtype,
        )

        # Optional multi-class adapter if user configures num_classes > 2 (e.g. 3-class SID-UNet)
        if self.num_classes > 2:
            self.class_head: Optional[nn.Linear] = nn.Linear(2, self.num_classes)
            with torch.no_grad():
                self.class_head.weight.zero_()
                self.class_head.bias.zero_()
                # Map class 0 -> real (idx 0), class 1..K -> fake (idx 1)
                self.class_head.weight[0, 0] = 1.0
                for c_idx in range(1, self.num_classes):
                    self.class_head.weight[c_idx, 1] = 1.0
        else:
            self.class_head = None

        # 4. Load full OpenSDI trained checkpoint from Hugging Face if in "opensdi" mode
        self.loaded_pretrained_sources: Dict[str, str] = {
            "clip": f"openai-clip:{settings['model_name']} ({DEFAULT_CLIP_VIT_L14_URL})",
        }
        if load_mae_standalone:
            self.loaded_pretrained_sources["mae"] = self.mae_pretrained_url

        if load_full_opensdi:
            ckpt_file = _download_opensdi_checkpoint(
                repo_id_or_path=self.pretrained_model_name_or_path,
                filename=self.checkpoint_filename,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
            )
            self._load_opensdi_state_dict(ckpt_file, strict=False)
            self.loaded_pretrained_sources["opensdi_checkpoint"] = ckpt_file

        # 5. Criterion & EdgeMaskGenerator for OpenSDI compatibility
        self.ce_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCELoss()
        self.edge_mask_generator = EdgeMaskGenerator(kernel_size=self.edge_mask_width)

        # 6. Configure trainable parameters (default OpenSDI recipe: freeze CLIP, train MAE + SPM)
        self.set_trainable_submodules(
            clip_visual=not freeze_clip,
            clip_text=not freeze_clip,
            mae_backbone=not freeze_mae,
            vca_fusion=True,
            mask_decoder=True,
            cls_aggregator=True,
            prompt_learner=True,
        )

        # 7. Surgery hooks & in-memory CLIP feature cache
        self._surgery_hooks: Dict[str, List[Callable[..., Any]]] = {
            "clip_features": [],
            "text_features": [],
            "mae_block_post_fusion": [],
            "decoder_features": [],
            "cls_features": [],
        }
        self.enable_clip_cache = bool(enable_clip_cache)
        self.clip_cache_size = int(clip_cache_size)
        self._clip_feature_cache: "OrderedDict[str, Tuple[torch.Tensor, List[torch.Tensor]]]" = OrderedDict()

        # Total cached channels for sid-cache extractor:
        # 7 layers * 1024 patch channels (at 16x16) + 7 layers * 1024 CLS channels (broadcast to 16x16)
        self.total_z_channels = len(self.selected_layers) * self.visemb_dim * 2

        # 8. Optional immediate LoRA/DoRA injection if lora_r > 0
        if lora_r > 0:
            self.inject_peft_lora(
                target=lora_target,
                r=int(lora_r),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
                use_dora=bool(use_dora),
            )

        if device is not None and str(device) not in ("", "auto"):
            self.to(torch.device(device))

    # ------------------------------------------------------------------------
    # Internal Helpers & Pretrained Weight Loading
    # ------------------------------------------------------------------------

    def _register_clip_hooks(self) -> None:
        """Attach forward hooks to every `ln_2` layer in `self.clip.visual`."""
        for h in getattr(self, "hooks", []):
            h.close()
        self.hooks = [
            FeatureHook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

    def _load_opensdi_state_dict(self, checkpoint_path: str, strict: bool = False) -> Tuple[List[str], List[str]]:
        """Load an OpenSDI `.pth` or `sid_unet` `.pt` checkpoint into this MaskCLIP instance."""
        abs_path = os.path.abspath(checkpoint_path) if os.path.exists(checkpoint_path) else checkpoint_path
        mtime = os.path.getmtime(abs_path) if os.path.exists(abs_path) else 0.0
        cache_key = f"{abs_path}:{mtime}"
        if cache_key in _OPENSDI_STATE_DICT_CACHE:
            cleaned = _OPENSDI_STATE_DICT_CACHE[cache_key]
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "model" in ckpt and isinstance(ckpt["model"], dict):
                    state_dict = ckpt["model"]
                elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                    state_dict = ckpt["model_state_dict"]
                else:
                    state_dict = ckpt
            else:
                raise ValueError(f"Unsupported checkpoint payload type: {type(ckpt)}")

            cleaned = {}
            for k, v in state_dict.items():
                new_k = k[7:] if k.startswith("module.") else k
                cleaned[new_k] = v
            _OPENSDI_STATE_DICT_CACHE[cache_key] = cleaned

        # Filter out any shape-mismatched keys only if output channels/resolution were customized
        if self.visemb_dim != 1024:
            logger.warning(
                f"Loading OpenSDI checkpoint into model with visemb_dim={self.visemb_dim} "
                f"({self.model_setting_name}). Official OpenSDI checkpoints were trained "
                f"with ViT-L/14 (visemb_dim=1024, align_dim=768). Cross-attention adapter "
                f"layers and prompt learner have mismatched dimensions and will not transfer without fine-tuning."
            )

        model_sd = self.state_dict()
        compatible_sd: Dict[str, torch.Tensor] = {}
        for k, v in cleaned.items():
            if k in model_sd:
                if model_sd[k].shape == v.shape:
                    compatible_sd[k] = v
                elif k == "aggregator.vit_model.pos_embed":
                    # Interpolate pos_embed if resolution / pretrain_img_size differs
                    compatible_sd[k] = self._resize_pos_embed_tensor(v, model_sd[k])
                else:
                    logger.warning(
                        f"Skipping key '{k}' due to shape mismatch: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}"
                    )
            else:
                compatible_sd[k] = v

        missing, unexpected = self.load_state_dict(compatible_sd, strict=strict)
        return list(missing), list(unexpected)

    @staticmethod
    def _resize_pos_embed_tensor(src_pos: torch.Tensor, dst_pos: torch.Tensor) -> torch.Tensor:
        if src_pos.shape == dst_pos.shape:
            return src_pos
        dst_len = dst_pos.shape[1]
        src_len = src_pos.shape[1]
        dim = src_pos.shape[2]
        src_has_cls = int(math.sqrt(src_len)) ** 2 != src_len
        dst_has_cls = int(math.sqrt(dst_len)) ** 2 != dst_len

        cls_tok = src_pos[:, :1, :] if src_has_cls else torch.zeros((1, 1, dim), dtype=src_pos.dtype)
        src_grid = src_pos[:, 1:, :] if src_has_cls else src_pos
        s_size = int(math.sqrt(src_grid.shape[1]))
        d_num = (dst_len - 1) if dst_has_cls else dst_len
        d_size = int(math.sqrt(d_num))

        resized = F.interpolate(
            src_grid.reshape(1, s_size, s_size, dim).permute(0, 3, 1, 2).float(),
            size=(d_size, d_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, d_num, dim).to(src_pos.dtype)

        if dst_has_cls:
            return torch.cat([cls_tok, resized], dim=1)
        return resized

    def _preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input RGB image is normalized with CLIP mean/std when passed in [0, 1]."""
        if not self.input_normalize:
            return x
        # Heuristic: if tensor values lie within [-0.05, 1.05], apply CLIP normalization
        with torch.no_grad():
            x_min = float(x.min().item())
            x_max = float(x.max().item())
        if x_min >= -0.05 and x_max <= 1.05:
            mean = self.clip_mean.to(device=x.device, dtype=x.dtype)
            std = self.clip_std.to(device=x.device, dtype=x.dtype)
            return (x - mean) / std
        return x

    # ------------------------------------------------------------------------
    # Core Encoding & Forward Pass (SPM Pipeline)
    # ------------------------------------------------------------------------

    def encode_text(self, x: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """Encode continuous prompt tokens through the CLIP text transformer."""
        orig_dtype = x.dtype
        clip_dtype = self.clip.positional_embedding.dtype
        x = x.to(clip_dtype) + self.clip.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        if isinstance(x, tuple):
            x = x[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.clip.text_projection
        return x.to(orig_dtype)

    def compute_text_features(self, device: torch.device) -> torch.Tensor:
        """Compute normalized [real, fake] text embeddings of shape (2, align_dim)."""
        text = ["an image"] * 2
        prompts, tokenized_prompts = self.prompt_learner(self.clip, text, device)
        text_features = self.encode_text(prompts, tokenized_prompts)
        chunks = torch.chunk(text_features, dim=0, chunks=2)
        text_features_mean = torch.stack([chunks[0].mean(0), chunks[1].mean(0)], dim=0)
        text_features_mean = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        for hook_fn in self._surgery_hooks["text_features"]:
            text_features_mean = hook_fn(text_features_mean)
        return text_features_mean

    def extract_clip_features(
        self,
        norm_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract intermediate CLS and spatial patch features from CLIP visual encoder."""
        clip_image = F.interpolate(norm_image, size=(224, 224), mode="bilinear", align_corners=True)
        clip_trainable = any(p.requires_grad for p in self.clip.visual.parameters())

        context = torch.enable_grad() if (self.training and clip_trainable) else torch.no_grad()
        with context:
            self.clip.encode_image(clip_image.to(self.clip.dtype))
            hook_outputs = [h.output for h in self.hooks]
            # OpenAI CLIP ViT uses LND [257, B, 1024] inside resblocks; normalize if BLD [B, 257, 1024]
            if hook_outputs[0].shape[0] == clip_image.shape[0] and hook_outputs[0].shape[1] != clip_image.shape[0]:
                hook_outputs = [t.permute(1, 0, 2) for t in hook_outputs]
            features = torch.stack(hook_outputs, dim=2).float()
            selected_features = torch.stack([features[:, :, i, :] for i in self.selected_layers], dim=2)

        N, B, L, C = selected_features.shape
        cls_features = selected_features[0, :, :, :]  # [B, L, C]
        patch_tokens = selected_features[1:, :, :, :]  # [N-1, B, L, C]
        grid_size = int(math.sqrt(N - 1))
        patch_features = [
            patch_tokens[:, :, i, :].permute(1, 2, 0).reshape(B, C, grid_size, grid_size)
            for i in range(L)
        ]

        for hook_fn in self._surgery_hooks["clip_features"]:
            cls_features, patch_features = hook_fn(cls_features, patch_features)

        return cls_features, patch_features

    def _unpack_cached_clip_features(
        self,
        z_cached: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Unpack cached tensor [B, 2*L*C, 16, 16] into (cls_features, patch_features)."""
        B = z_cached.shape[0]
        L = len(self.selected_layers)
        C = self.visemb_dim
        z_float = z_cached.float()
        patch_part = z_float[:, : L * C, :, :]  # [B, L*C, H_p, W_p]
        cls_part = z_float[:, L * C : 2 * L * C, 0, 0]  # [B, L*C]
        cls_features = cls_part.view(B, L, C)
        patch_features = [patch_part[:, i * C : (i + 1) * C, :, :] for i in range(L)]
        return cls_features, patch_features

    @torch.no_grad()
    def extract_cache_tensors(self, images: torch.Tensor) -> torch.Tensor:
        """Extract frozen CLIP multi-layer patch & CLS features for `sid-cache`."""
        self.eval()
        norm_img = self._preprocess_image(images)
        cls_features, patch_features = self.extract_clip_features(norm_img)
        # patch_features: list of L tensors of shape [B, C, 16, 16]
        patch_stack = torch.cat(patch_features, dim=1)  # [B, L*C, 16, 16]
        H_p, W_p = patch_stack.shape[2], patch_stack.shape[3]
        cls_flat = cls_features.reshape(images.shape[0], -1, 1, 1).expand(-1, -1, H_p, W_p)
        return torch.cat([patch_stack, cls_flat], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        cached_clip_features: Optional[Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]] = None,
        return_dict: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Standard `sid_unet` forward pass.

        Args:
            x: Input RGB tensor `[B, 3, H, W]` (or cached tensor `[B, 2*L*C, 16, 16]` if `image` in kwargs).
            cached_clip_features: Optional pre-extracted CLIP features.
            return_dict: If True, returns a dictionary of logits and intermediate features.

        Returns:
            `(mask_logits, class_logits)` if `self.aux_classifier` is True, else `mask_logits`.
        """
        # Handle OpenSDI-style call `model(image, mask, label, edge_mask)` if 3+ positional args are given
        if len(args) >= 2 and isinstance(args[0], torch.Tensor):
            mask_arg = args[0]
            label_arg = args[1]
            edge_arg = args[2] if len(args) >= 3 else kwargs.get("edge_mask")
            return self.forward_opensdi(x, mask=mask_arg, label=label_arg, edge_mask=edge_arg)

        # Support cached training where x is RGB image and cached_clip_features is passed,
        # or x is z_cached and kwargs['image'] is passed
        if x.shape[1] == self.total_z_channels and "image" in kwargs and kwargs["image"] is not None:
            cached_clip_features = x
            x = kwargs["image"]

        orig_h, orig_w = x.shape[2], x.shape[3]
        norm_image = self._preprocess_image(x)

        if cached_clip_features is not None:
            if isinstance(cached_clip_features, (tuple, list)):
                cls_features, patch_features = cached_clip_features
            else:
                cls_features, patch_features = self._unpack_cached_clip_features(cached_clip_features)
        else:
            cls_features, patch_features = self.extract_clip_features(norm_image)

        text_features_mean = self.compute_text_features(norm_image.device)

        # Resize image for MAE spatial stream if input resolution is smaller than patch_size (16)
        mae_input = norm_image
        if orig_h < 32 or orig_w < 32:
            mae_input = F.interpolate(norm_image, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)

        def _block_hook(block_idx: int, feat: torch.Tensor) -> torch.Tensor:
            for fn in self._surgery_hooks["mae_block_post_fusion"]:
                feat = fn(block_idx, feat)
            return feat

        def _decoder_hook(feat: torch.Tensor) -> torch.Tensor:
            for fn in self._surgery_hooks["decoder_features"]:
                feat = fn(feat)
            return feat

        mask_logits = self.aggregator(
            mae_input,
            text_features_mean,
            patch_features,
            block_hook=_block_hook if self._surgery_hooks["mae_block_post_fusion"] else None,
            decoder_hook=_decoder_hook if self._surgery_hooks["decoder_features"] else None,
        )
        if mask_logits.shape[2:] != (orig_h, orig_w):
            mask_logits = F.interpolate(mask_logits, size=(orig_h, orig_w), mode="bilinear", align_corners=True)

        agg_cls = self.cls_aggregator(cls_features)
        for fn in self._surgery_hooks["cls_features"]:
            agg_cls = fn(agg_cls)

        binary_class_logits = agg_cls @ text_features_mean.t()  # [B, 2] (0: real, 1: fake)
        if self.class_head is not None and self.num_classes > 2:
            class_logits = self.class_head(binary_class_logits)
        else:
            class_logits = binary_class_logits

        if return_dict:
            return {
                "mask_logits": mask_logits,
                "class_logits": class_logits,
                "binary_class_logits": binary_class_logits,
                "pred_mask": torch.sigmoid(mask_logits),
                "pred_label": torch.argmax(binary_class_logits, dim=1),
                "text_features": text_features_mean,
                "cls_features": agg_cls,
            }

        if self.aux_classifier:
            return mask_logits, class_logits
        return mask_logits

    def forward_opensdi(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Official OpenSDI / IMDLBenCo compatible forward method returning `output_dict`.
        Computes `ce_loss + bce_loss + edge_loss` exactly as in `OpenSDI/model/MaskCLIP.py`.
        """
        out = self.forward(image, return_dict=True)
        assert isinstance(out, dict)
        mask_logits = out["mask_logits"]
        binary_logits = out["binary_class_logits"]
        mask_prob = torch.sigmoid(mask_logits)
        pred_label = torch.argmax(binary_logits, dim=1)

        if mask is None:
            return {
                "pred_mask": mask_prob,
                "pred_label": pred_label,
                "mask_logits": mask_logits,
                "class_logits": binary_logits,
            }

        target_mask = mask.float().to(mask_logits.device)
        if target_mask.ndim == 3:
            target_mask = target_mask.unsqueeze(1)

        if edge_mask is None:
            edge_mask_t = self.edge_mask_generator(target_mask)
        else:
            edge_mask_t = edge_mask.float().to(mask_logits.device)
            if edge_mask_t.ndim == 3:
                edge_mask_t = edge_mask_t.unsqueeze(1)

        edge_loss = (
            F.binary_cross_entropy_with_logits(
                input=mask_logits.float(),
                target=target_mask.float(),
                weight=edge_mask_t.float(),
            )
            * self.edge_lambda
        )
        bce_loss = F.binary_cross_entropy_with_logits(
            mask_logits.float().view(-1),
            target_mask.float().view(-1),
        )

        if label is not None:
            target_label = (label.to(binary_logits.device) > 0).long()
            ce_loss = self.ce_criterion(binary_logits.float(), target_label)
        else:
            ce_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)

        loss = ce_loss + bce_loss + edge_loss

        return {
            "backward_loss": loss,
            "pred_mask": mask_prob,
            "pred_label": pred_label,
            "visual_loss": {
                "loss_ce": ce_loss,
                "loss_bce": bce_loss,
                "loss_edge": edge_loss,
                "combined_loss": loss,
            },
            "visual_image": {
                "pred_mask": mask_prob,
            },
        }

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict thresholded binary manipulation mask `[B, 1, H, W]`."""
        self.eval()
        outputs = self.forward(x)
        mask_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        assert isinstance(mask_logits, torch.Tensor)
        return (torch.sigmoid(mask_logits) >= threshold).float()

    # ------------------------------------------------------------------------
    # Model Surgery APIs
    # ------------------------------------------------------------------------

    def set_trainable_submodules(
        self,
        clip_visual: bool = False,
        clip_text: bool = False,
        mae_backbone: bool = True,
        vca_fusion: bool = True,
        mask_decoder: bool = True,
        cls_aggregator: bool = True,
        prompt_learner: bool = True,
    ) -> Dict[str, int]:
        """
        Granular surgical control over which SPM submodules require gradients.

        Returns a summary dict with `total_params`, `trainable_params`, and `frozen_params`.
        """
        for name, param in self.clip.named_parameters():
            if name.startswith("visual."):
                param.requires_grad = bool(clip_visual)
            else:
                param.requires_grad = bool(clip_text)

        for param in self.aggregator.vit_model.parameters():
            param.requires_grad = bool(mae_backbone)
        if self.aggregator.custom_spatial_adapter is not None:
            for param in self.aggregator.custom_spatial_adapter.parameters():
                param.requires_grad = bool(mae_backbone)

        for param in self.aggregator.fusion_layers.parameters():
            param.requires_grad = bool(vca_fusion)

        for param in self.aggregator.text_projection.parameters():
            param.requires_grad = bool(mask_decoder)
        for param in self.aggregator.mask_decoder.parameters():
            param.requires_grad = bool(mask_decoder)

        for param in self.cls_aggregator.parameters():
            param.requires_grad = bool(cls_aggregator)

        for param in self.prompt_learner.parameters():
            param.requires_grad = bool(prompt_learner)

        if self.class_head is not None:
            for param in self.class_head.parameters():
                param.requires_grad = bool(cls_aggregator or prompt_learner)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        }

    def replace_spatial_backbone(
        self,
        new_backbone: nn.Module,
        backbone_out_channels: int = 768,
        fusion_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Model Surgery: Replace the MAE `vit_model` spatial backbone with a new dense encoder
        (e.g., SAM ViT, EfficientSAM3 TinyViT, DINOv2, or ConvNeXt).

        If `backbone_out_channels != 768`, automatically inserts a `1x1 Conv + LayerNorm`
        spatial adapter so VCA `AttentionFusion` and `MaskDecoder` weights remain compatible.
        """
        device = next(self.parameters()).device
        self.aggregator.vit_model = new_backbone.to(device)
        self.aggregator.replaced_backbone_channels = backbone_out_channels
        if backbone_out_channels != 768:
            self.aggregator.custom_spatial_adapter = nn.Sequential(
                nn.Conv2d(backbone_out_channels, 768, kernel_size=1),
                LayerNorm(768),
            ).to(device)
        else:
            self.aggregator.custom_spatial_adapter = None

        if fusion_map is not None:
            self.reconfigure_layer_tap_and_fusion(
                selected_layers=self.selected_layers,
                fusion_map=fusion_map,
                reuse_weights=True,
            )

    def replace_semantic_backbone(
        self,
        new_clip_model: nn.Module,
        selected_layers: Optional[Sequence[int]] = None,
        hook_module_substr: str = "ln_2",
    ) -> None:
        """
        Model Surgery: Replace the frozen CLIP backbone with another vision-language model
        and re-bind intermediate `FeatureHook`s.
        """
        device = next(self.parameters()).device
        for h in self.hooks:
            h.close()
        self.clip = new_clip_model.to(device).float()
        self.hooks = [
            FeatureHook(name, module)
            for name, module in self.clip.visual.named_modules()
            if hook_module_substr in name
        ]
        if selected_layers is not None:
            self.selected_layers = [int(i) for i in selected_layers]

    def replace_mask_decoder(self, new_decoder: nn.Module) -> None:
        """Model Surgery: Swap the FPN+TVCA `MaskDecoder` with a custom segmentation decoder."""
        device = next(self.parameters()).device
        self.aggregator.mask_decoder = new_decoder.to(device)

    def reconfigure_layer_tap_and_fusion(
        self,
        selected_layers: Sequence[int],
        fusion_map: Dict[int, int],
        reuse_weights: bool = True,
    ) -> None:
        """
        Model Surgery: Rewire which CLIP visual layers (`selected_layers`) are extracted and
        how they map into MAE blocks (`fusion_map: {mae_block_idx: clip_selected_idx}`).
        Preserves existing `AttentionFusion` weights for overlapping target blocks when `reuse_weights=True`.
        """
        device = next(self.parameters()).device
        self.selected_layers = [int(i) for i in selected_layers]
        new_map = {int(k): int(v) for k, v in fusion_map.items()}
        old_layers = self.aggregator.fusion_layers

        new_fusion_layers = nn.ModuleDict()
        for tgt_idx in new_map.keys():
            key = f"layer_{tgt_idx}"
            if reuse_weights and key in old_layers:
                new_fusion_layers[key] = old_layers[key]
            else:
                if self.aggregator.decoder_variant == "sideclip_mlp_refiner":
                    new_fusion_layers[key] = SideCLIPProjectionFusion(self.visemb_dim, 768).to(device)
                else:
                    new_fusion_layers[key] = AttentionFusion(self.visemb_dim, 768).to(device)

        self.aggregator.fusion_layers = new_fusion_layers
        self.aggregator.fusion_map = new_map
        self.total_z_channels = len(self.selected_layers) * self.visemb_dim * 2

    def inject_peft_lora(
        self,
        target: str = "mae",
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: Optional[Sequence[str]] = None,
        use_dora: bool = False,
    ) -> Dict[str, Any]:
        """
        Model Surgery: Inject Low-Rank Adaptation (LoRA) or Weight-Decomposed Low-Rank
        Adaptation (DoRA) into `mae`, `clip`, `vca`, or `all` linear attention projections.

        Returns metadata describing the modified modules and parameter counts.
        """
        target = target.lower().strip()
        if target_modules is None:
            target_modules = ("qkv", "proj", "fc1", "fc2", "out_proj", "query_proj", "key_proj", "value_proj")

        roots: List[Tuple[str, nn.Module]] = []
        if target in ("mae", "both", "all"):
            roots.append(("aggregator.vit_model", self.aggregator.vit_model))
        if target in ("clip", "clip_visual", "both", "all"):
            roots.append(("clip.visual", self.clip.visual))
        if target in ("vca", "fusion", "all"):
            roots.append(("aggregator.fusion_layers", self.aggregator.fusion_layers))

        replaced: List[str] = []
        for root_prefix, root_mod in roots:
            # First freeze all base parameters in the target root
            for p in root_mod.parameters():
                p.requires_grad = False

            for mod_name, mod in list(root_mod.named_modules()):
                if isinstance(mod, nn.Linear) and any( mod_name.endswith(t) for t in target_modules ):
                    parent_path = mod_name.rsplit(".", 1)
                    if len(parent_path) == 2:
                        parent = root_mod.get_submodule(parent_path[0])
                        attr = parent_path[1]
                    else:
                        parent = root_mod
                        attr = parent_path[0]

                    if isinstance(getattr(parent, attr, None), SurgicalLoRALinear):
                        continue
                    lora_layer = SurgicalLoRALinear(
                        base_linear=mod,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                        use_dora=use_dora,
                    )
                    setattr(parent, attr, lora_layer)
                    replaced.append(f"{root_prefix}.{mod_name}")

        total_p = sum(p.numel() for p in self.parameters())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "target": target,
            "r": r,
            "alpha": alpha,
            "use_dora": use_dora,
            "num_adapted_layers": len(replaced),
            "adapted_layers": replaced,
            "total_params": total_p,
            "trainable_params": train_p,
            "trainable_pct": round(100.0 * train_p / max(total_p, 1), 4),
        }

    def register_surgery_hook(self, stage: str, hook_fn: Callable[..., Any]) -> None:
        """
        Register a forward surgery intervention hook at one of:
        `clip_features`, `text_features`, `mae_block_post_fusion`, `decoder_features`, `cls_features`.
        """
        if stage not in self._surgery_hooks:
            raise ValueError(
                f"Unknown surgery hook stage '{stage}'. Valid stages: {list(self._surgery_hooks.keys())}"
            )
        self._surgery_hooks[stage].append(hook_fn)

    def clear_surgery_hooks(self, stage: Optional[str] = None) -> None:
        """Remove registered surgery hooks for a specific stage or all stages."""
        if stage is not None:
            if stage in self._surgery_hooks:
                self._surgery_hooks[stage].clear()
        else:
            for k in self._surgery_hooks:
                self._surgery_hooks[k].clear()

    @torch.no_grad()
    def task_arithmetic_merge(
        self,
        other: Union["MaskCLIP", Dict[str, torch.Tensor]],
        alpha: float = 0.5,
        submodules: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Model Surgery: WiSE-FT / Task Arithmetic weight-space interpolation:
          `W_merged = (1 - alpha) * W_self + alpha * W_other`
        Enables merging zero-shot/SD1.5 OpenSDI weights with fine-tuned checkpoints (e.g. Flux.1/SD3)
        to preserve open-world generalization while adapting to new generators.
        """
        other_sd = other.state_dict() if isinstance(other, nn.Module) else other
        merged_keys: List[str] = []
        self_sd = self.state_dict()

        for k, param in self_sd.items():
            if submodules and not any(k.startswith(prefix) for prefix in submodules):
                continue
            if k in other_sd and other_sd[k].shape == param.shape and param.is_floating_point():
                other_tensor = other_sd[k].to(device=param.device, dtype=param.dtype)
                param.copy_((1.0 - alpha) * param + alpha * other_tensor)
                merged_keys.append(k)
        return merged_keys

    def get_surgery_summary(self) -> Dict[str, Any]:
        """Return detailed parameter and architectural breakdown across all 7 SPM components."""
        groups = {
            "clip_visual": self.clip.visual,
            "clip_text": nn.ModuleList([self.clip.transformer, self.clip.token_embedding, self.clip.ln_final]),
            "mae_vit": self.aggregator.vit_model,
            "vca_fusion": self.aggregator.fusion_layers,
            "mask_decoder": self.aggregator.mask_decoder,
            "cls_aggregator": self.cls_aggregator,
            "prompt_learner": self.prompt_learner,
        }
        breakdown: Dict[str, Dict[str, int]] = {}
        for name, mod in groups.items():
            tot = sum(p.numel() for p in mod.parameters())
            trn = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            breakdown[name] = {
                "total": tot,
                "trainable": trn,
                "frozen": tot - trn,
            }
        total_all = sum(p.numel() for p in self.parameters())
        train_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_setting_name": self.model_setting_name,
            "resolution": self.resolution,
            "selected_layers": list(self.selected_layers),
            "fusion_map": dict(self.aggregator.fusion_map),
            "pretrained_sources": dict(self.loaded_pretrained_sources),
            "total_params": total_all,
            "trainable_params": train_all,
            "trainable_pct": round(100.0 * train_all / max(total_all, 1), 2),
            "submodules": breakdown,
        }

    # ------------------------------------------------------------------------
    # Factory Loaders (`from_pretrained` / `from_checkpoint`)
    # ------------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = DEFAULT_MASKCLIP_HF_REPO,
        *args: Any,
        filename: Optional[str] = DEFAULT_MASKCLIP_HF_FILENAME,
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = False,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> "MaskCLIP":
        """Load a pretrained MaskCLIP model from Hugging Face (`nebula/MaskCLIP-weights`) or local path."""
        return cls.from_checkpoint(
            checkpoint_path=pretrained_model_name_or_path,
            filename=filename,
            device=device,
            strict=strict,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = DEFAULT_MASKCLIP_HF_REPO,
        filename: Optional[str] = DEFAULT_MASKCLIP_HF_FILENAME,
        device: Optional[Union[str, torch.device]] = None,
        override_config: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = False,
        return_config: bool = False,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> Union["MaskCLIP", Tuple["MaskCLIP", Any]]:
        """Load MaskCLIP from either an OpenSDI `.pth` checkpoint or a `sid_unet` `.pt` checkpoint."""
        if os.path.isfile(checkpoint_path):
            resolved = os.path.abspath(checkpoint_path)
        else:
            try:
                resolved = _download_opensdi_checkpoint(
                    repo_id_or_path=checkpoint_path,
                    filename=filename,
                    cache_dir=cache_dir,
                    token=token,
                    force_download=force_download,
                )
            except Exception:
                from sid_unet.utils.checkpoint import resolve_checkpoint_source
                resolved = resolve_checkpoint_source(
                    checkpoint_path,
                    cache_dir=cache_dir,
                    token=token,
                    force_download=force_download,
                )

        model = cls(
            pretrained=True,
            pretrained_mode="opensdi",
            pretrained_model_name_or_path=resolved,
            checkpoint_filename=filename,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            device=device,
            **kwargs,
        )
        if strict:
            model._load_opensdi_state_dict(resolved, strict=True)
        model.eval()
        if return_config:
            from sid_unet.utils.config import ConfigDict, DEFAULT_CONFIG, deep_merge
            cfg = deep_merge(DEFAULT_CONFIG, {"model": {"name": "mask_clip"}})
            if override_config:
                cfg = deep_merge(cfg, override_config)
            return model, ConfigDict(cfg)
        return model


# Convenient aliases
OpenSDIMaskCLIP = MaskCLIP
MaskCLIPModel = MaskCLIP
