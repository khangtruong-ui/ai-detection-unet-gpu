"""
SparseViT: Nonsemantics-Centered, Parameter-Efficient Image Manipulation Localization
through Sparse-Coding Transformer (AAAI 2025).

Official Paper: Su et al., "Can We Get Rid of Handcrafted Feature Extractors?
SparseViT: Nonsemantics-Centered, Parameter-Efficient Image Manipulation Localization
through Spare-Coding Transformer", AAAI 2025 (arXiv:2412.14598).
Official Repository: https://github.com/scu-zjz/SparseViT

Architecture Summary:
- Hierarchical UniFormer-style backbone (`SparseViTEncoder`) with local convolutional
  blocks (`CBlock`) in stages 1-2 and sparse self-attention (`SABlock` via `alter_sparse`
  / `alter_unsparse`) in stages 3-4.
- Lightweight Feature Fusion (`MultipleLFFHead` / `lmu`) fusing 4 intermediate outputs
  from stage 3 and 2 outputs from stage 4 into single-channel binary manipulation logits.
- Expects ImageNet-standardized RGB inputs at 512x512 (auto-normalizes `[0, 1]` inputs
  when `auto_normalize=True` and resizes non-512 inputs gracefully).
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:  # pragma: no cover
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_

logger = logging.getLogger("sid_unet.models.sparsevit")

SPARSEVIT_GDRIVE_FILE_ID = "104BPPvLXkxuPu_NHaxjesdcdZ-ln92-G"
SPARSEVIT_GDRIVE_URL = f"https://drive.google.com/file/d/{SPARSEVIT_GDRIVE_FILE_ID}/view?usp=drive_link"
SPARSEVIT_DEFAULT_FILENAME = "sparsevit_mul.pth"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _DWConv(nn.Module):
    """Depthwise 3x3 convolution for token sequence MLP."""

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        b, _, c = x.shape
        # Ensure contiguous NCHW layout; transposed NHWC views fail cuDNN depthwise Conv2d engines under BF16 on SM 7.5.
        x = x.transpose(1, 2).reshape(b, c, H, W).contiguous()
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class _Mlp(nn.Module):
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
        self.dwconv = _DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class _CMlp(nn.Module):
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
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


def _block_partition(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int, int, int]:
    b, h, w, c = x.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.reshape(b, hp // block_size, block_size, wp // block_size, block_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, h, hp, c


def _unblock_partition(x: torch.Tensor, orig_h: int) -> torch.Tensor:
    b, h, w, win_h, win_w, c = x.shape
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(b, h * win_h, w * win_w, c)
    hp = h * win_h
    wp = w * win_w
    if hp > orig_h or wp > orig_h:
        x = x[:, :orig_h, :orig_h, :].contiguous()
    return x


def _alter_sparse(x: torch.Tensor, sparse_size: int = 8) -> Tuple[torch.Tensor, int, int, int]:
    x = x.permute(0, 2, 3, 1)
    if x.shape[1] % sparse_size != 0 or x.shape[2] % sparse_size != 0:
        raise ValueError(
            f"Spatial dimensions {tuple(x.shape[1:3])} must be divisible by sparse_size={sparse_size}."
        )
    grid_size = x.shape[1] // sparse_size
    out, h, hp, c = _block_partition(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, c)
    out = out.permute(0, 3, 1, 2)
    return out, h, hp, c


def _alter_unsparse(x: torch.Tensor, orig_h: int, hp: int, c: int, sparse_size: int = 8) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, hp // sparse_size, hp // sparse_size, sparse_size, sparse_size, c)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = _unblock_partition(x, orig_h)
    return out.permute(0, 3, 1, 2)


class CBlock(nn.Module):
    """Local convolutional UniFormer block used in SparseViT stages 1-2."""

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
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = _CMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    """Sparse self-attention block used in SparseViT stages 3-4."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sparse_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: Any = nn.LayerNorm,
        layer_scale: bool = True,
        init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls = bool(layer_scale)
        self.sparse_size = int(sparse_size)
        if self.ls:
            self.gamma_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_befor = x.flatten(2).transpose(1, 2)
        b, c_dim, h, w = x.shape

        x_sp, ho, hp, c = _alter_sparse(x, self.sparse_size)
        bf, nf, hf, wf = x_sp.shape
        x_sp = x_sp.flatten(2).transpose(1, 2)
        x_sp = self.attn(self.norm1(x_sp))
        x_sp = x_sp.transpose(1, 2).reshape(bf, nf, hf, wf)
        x_sp = _alter_unsparse(x_sp, ho, hp, c, self.sparse_size)
        x_sp = x_sp.flatten(2).transpose(1, 2)

        if self.ls:
            x_tok = x_befor + self.drop_path(self.gamma_1 * x_sp)
            x_tok = x_tok + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_tok), h, w))
        else:
            x_tok = x_befor + self.drop_path(x_sp)
            x_tok = x_tok + self.drop_path(self.mlp(self.norm2(x_tok), h, w))

        return x_tok.transpose(1, 2).reshape(b, c_dim, h, w).contiguous()


class _PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        img_size_t = to_2tuple(img_size)
        patch_size_t = to_2tuple(patch_size)
        self.img_size = img_size_t
        self.patch_size = patch_size_t
        self.num_patches = (img_size_t[1] // patch_size_t[1]) * (img_size_t[0] // patch_size_t[0])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size_t, stride=patch_size_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if h != self.img_size[0] or w != self.img_size[1]:
            raise ValueError(
                f"Input spatial size ({h}x{w}) does not match PatchEmbed size ({self.img_size[0]}x{self.img_size[1]})."
            )
        x = self.proj(x)
        b, _, hp, wp = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.reshape(b, hp, wp, -1).permute(0, 3, 1, 2).contiguous()


class SparseViTEncoder(nn.Module):
    """SparseViT hierarchical encoder matching `encoder_net.*` in official checkpoints."""

    def __init__(
        self,
        layers: Sequence[int] = (5, 8, 20, 7),
        img_size: int = 512,
        in_chans: int = 3,
        s_blocks3: Sequence[int] = (8, 4, 2, 1),
        s_blocks4: Sequence[int] = (2, 1),
        embed_dim: Sequence[int] = (64, 128, 320, 512),
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.num_features = self.embed_dim = list(embed_dim)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = _PatchEmbed(img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = _PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = _PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = _PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        num_heads = [dim // head_dim for dim in embed_dim]

        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(layers[0])
        ])
        self.norm1 = norm_layer(embed_dim[0])

        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + layers[0]],
                norm_layer=norm_layer,
            )
            for i in range(layers[1])
        ])
        self.norm2 = norm_layer(embed_dim[1])

        stage3_grid = max(1, img_size // 16)
        self.blocks3 = nn.ModuleList([
            SABlock(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                sparse_size=max(1, stage3_grid // s_blocks3[min(i // 5, len(s_blocks3) - 1)]),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + layers[0] + layers[1]],
                norm_layer=norm_layer,
            )
            for i in range(layers[2])
        ])
        self.norm3 = norm_layer(embed_dim[2])

        stage4_grid = max(1, img_size // 32)
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dim[3],
                num_heads=num_heads[3],
                sparse_size=max(1, stage4_grid // s_blocks4[min(i // 4, len(s_blocks4) - 1)]),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + layers[0] + layers[1] + layers[2]],
                norm_layer=norm_layer,
            )
            for i in range(layers[3])
        ])
        self.norm4 = norm_layer(embed_dim[3])
        self.pre_logits = nn.Identity()

    def forward_features(
        self,
        x: torch.Tensor,
        return_stage2: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        outputs: Dict[str, torch.Tensor] = {}
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        stage2_feat = x
        x = self.patch_embed3(x)
        for index, blk in enumerate(self.blocks3):
            x = blk(x)
            if (index + 1) % 5 == 0 and (index + 1) // 5 != 4:
                outputs[f"third{(index + 1) // 5}"] = x
        x_out3 = self.norm3(x.permute(0, 2, 3, 1))
        outputs["third"] = x_out3.permute(0, 3, 1, 2).contiguous()

        x = self.patch_embed4(x)
        for index, blk in enumerate(self.blocks4):
            x = blk(x)
            if (index + 1) % 4 == 0:
                outputs[f"last{(index + 1) // 4}"] = x
        x_out4 = self.norm4(x.permute(0, 2, 3, 1))
        outputs["last"] = x_out4.permute(0, 3, 1, 2).contiguous()
        if return_stage2:
            return outputs, stage2_feat
        return outputs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_features(x)


class MultipleLFFHead(nn.Module):
    """Lightweight Feature Fusion (LFF) decoder matching `lmu.*` in `SparseViT_Mul`."""

    def __init__(
        self,
        init_value: float = 1e-6,
        embed_dim: int = 512,
        stage3_dim: int = 320,
        predict_channels: int = 1,
        norm_layer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.gamma1 = nn.Parameter(init_value * torch.ones(embed_dim), requires_grad=True)
        self.gamma2 = nn.Parameter(init_value * torch.ones(embed_dim), requires_grad=True)
        self.gamma3 = nn.Parameter(init_value * torch.ones(embed_dim), requires_grad=True)
        self.gamma4 = nn.Parameter(init_value * torch.ones(embed_dim), requires_grad=True)
        self.gamma5 = nn.Parameter(init_value * torch.ones(embed_dim), requires_grad=True)
        self.gamma6 = nn.Parameter(init_value * torch.ones(embed_dim), requires_grad=True)
        self.norm = norm_layer(embed_dim)

        self.conv_layer1 = nn.Conv2d(stage3_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_layer2 = nn.Conv2d(stage3_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_layer3 = nn.Conv2d(stage3_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_layer4 = nn.Conv2d(stage3_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_last = nn.Conv2d(embed_dim, predict_channels, kernel_size=1)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        return_fused: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        c1, c2, c3, c4, c5, c6 = features
        c1 = self.conv_layer1(c1)
        c2 = self.conv_layer2(c2)
        c3 = self.conv_layer3(c3)
        c4 = self.conv_layer4(c4)

        b, c, h, w = c1.shape
        c5 = F.interpolate(c5, size=(h, w), mode="bilinear", align_corners=False)
        c6 = F.interpolate(c6, size=(h, w), mode="bilinear", align_corners=False)

        t1 = c1.flatten(2).transpose(1, 2)
        t2 = c2.flatten(2).transpose(1, 2)
        t3 = c3.flatten(2).transpose(1, 2)
        t4 = c4.flatten(2).transpose(1, 2)
        t5 = c5.flatten(2).transpose(1, 2)
        t6 = c6.flatten(2).transpose(1, 2)

        fused = (
            self.gamma1 * t1
            + self.gamma2 * t2
            + self.gamma3 * t3
            + self.gamma4 * t4
            + self.gamma5 * t5
            + self.gamma6 * t6
        )
        fused = fused.transpose(1, 2).reshape(b, c, h, w)
        fused = self.norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        logits = self.conv_last(fused)
        if return_fused:
            return logits, fused
        return logits


class SparseViT(nn.Module):
    """
    Pure-PyTorch SparseViT (`SparseViT_Mul`) for Image Manipulation Localization.

    Enforces `allow_random_init=False` by default so pretrained open weights must be
    loaded unless `allow_random_init=True` is explicitly provided for unit tests.
    """

    def __init__(
        self,
        depth: Sequence[int] = (5, 8, 20, 7),
        embed_dim: Sequence[int] = (64, 128, 320, 512),
        head_dim: int = 64,
        img_size: int = 512,
        s_blocks3: Sequence[int] = (8, 4, 2, 1),
        s_blocks4: Sequence[int] = (2, 1),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.2,
        pretrained: bool = True,
        allow_random_init: bool = False,
        checkpoint_path: Optional[str] = None,
        hf_repo: Optional[str] = None,
        cache_dir: Optional[str] = None,
        freeze: bool = False,
        auto_normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.freeze = bool(freeze)
        self.auto_normalize = bool(auto_normalize)
        self.hf_repo = hf_repo

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.encoder_net = SparseViTEncoder(
            layers=depth,
            img_size=self.img_size,
            in_chans=3,
            s_blocks3=s_blocks3,
            s_blocks4=s_blocks4,
            embed_dim=embed_dim,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.lmu = MultipleLFFHead(
            embed_dim=int(embed_dim[3]),
            stage3_dim=int(embed_dim[2]),
            predict_channels=1,
            norm_layer=norm_layer,
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
                "SparseViT requires pretrained=True (or explicit allow_random_init=True). "
                "Random weight initialization is strictly disabled."
            )

        if pretrained:
            self._load_pretrained(
                checkpoint_path=checkpoint_path,
                hf_repo=hf_repo,
                cache_dir=cache_dir,
            )

        if self.freeze:
            self.set_frozen(True)

    def set_frozen(self, frozen: bool = True) -> None:
        self.freeze = bool(frozen)
        for p in self.encoder_net.parameters():
            p.requires_grad_(not self.freeze)
        for p in self.lmu.parameters():
            p.requires_grad_(not self.freeze)
        if self.freeze:
            self.eval()

    def train(self, mode: bool = True) -> "SparseViT":
        super().train(mode)
        if self.freeze:
            self.encoder_net.eval()
            self.lmu.eval()
            super().train(False)
        return self

    def _resolve_checkpoint_path(
        self,
        checkpoint_path: Optional[str],
        hf_repo: Optional[str],
        cache_dir: Optional[str],
    ) -> str:
        if checkpoint_path and os.path.isfile(checkpoint_path):
            return os.path.abspath(checkpoint_path)

        target_fn = (
            os.path.basename(checkpoint_path)
            if checkpoint_path and checkpoint_path.endswith((".pth", ".pt"))
            else SPARSEVIT_DEFAULT_FILENAME
        )
        home_cache = os.path.expanduser("~")
        candidates = [
            target_fn,
            os.path.join("pretrained", target_fn),
            os.path.join("checkpoints", target_fn),
            os.path.join("weights", target_fn),
            os.path.join(home_cache, ".cache", "sid_unet_weights", target_fn),
            os.path.join(home_cache, ".cache", "sparsevit", target_fn),
        ]
        if cache_dir:
            candidates.insert(0, os.path.join(cache_dir, target_fn))
        for cand in candidates:
            if os.path.isfile(cand):
                return os.path.abspath(cand)

        if hf_repo:
            try:
                from huggingface_hub import hf_hub_download

                for fname in (target_fn, "sparsevit_mul.pth", "checkpoint_best.pt"):
                    try:
                        dl = hf_hub_download(repo_id=hf_repo, filename=fname, cache_dir=cache_dir)
                        if dl and os.path.isfile(dl):
                            return dl
                    except Exception:
                        continue
            except Exception as hf_err:
                logger.warning(f"HF Hub lookup failed for SparseViT '{hf_repo}': {hf_err}")

        dest_dir = cache_dir or os.path.join(home_cache, ".cache", "sid_unet_weights")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, SPARSEVIT_DEFAULT_FILENAME)
        if os.path.isfile(dest_path):
            return dest_path

        try:
            import gdown

            out = gdown.download(id=SPARSEVIT_GDRIVE_FILE_ID, output=dest_path, quiet=False)
            if out and os.path.isfile(out):
                return out
        except Exception as gd_err:
            raise RuntimeError(
                f"Failed to download SparseViT checkpoint from Google Drive ({SPARSEVIT_GDRIVE_URL}): {gd_err}"
            ) from gd_err

        raise RuntimeError(
            f"Unable to resolve SparseViT checkpoint (checkpoint_path={checkpoint_path}). "
            f"Official Google Drive link: {SPARSEVIT_GDRIVE_URL}"
        )

    def _load_pretrained(
        self,
        checkpoint_path: Optional[str] = None,
        hf_repo: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        resolved = self._resolve_checkpoint_path(checkpoint_path, hf_repo, cache_dir)
        ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid SparseViT checkpoint payload at '{resolved}'")
        raw_sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
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
                f"SparseViT state_dict mismatch from '{resolved}': "
                f"missing={missing[:10]} (total {len(missing)}), "
                f"unexpected={unexpected[:10]} (total {len(unexpected)})"
            )
        self.load_state_dict(filtered_sd, strict=True)
        logger.info(f"✅ Loaded genuine SparseViT pretrained weights from: {resolved}")

    def _prepare_input(self, x: torch.Tensor, input_is_normalized: Optional[bool] = None) -> torch.Tensor:
        if input_is_normalized is None:
            should_norm = self.auto_normalize and (float(x.min().detach()) >= -0.05 and float(x.max().detach()) <= 1.05)
        else:
            should_norm = not input_is_normalized

        if should_norm:
            mean = self._imagenet_mean.to(device=x.device, dtype=x.dtype)
            std = self._imagenet_std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std

        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        input_is_normalized: Optional[bool] = None,
        return_features: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        orig_h, orig_w = x.shape[2], x.shape[3]
        x_in = self._prepare_input(x, input_is_normalized=input_is_normalized)
        if return_features:
            feat_dict, s2_feat = self.encoder_net.forward_features(x_in, return_stage2=True)
            logits, lff_fused = self.lmu(list(feat_dict.values()), return_fused=True)
        else:
            feat_dict = self.encoder_net(x_in)
            logits = self.lmu(list(feat_dict.values()))
        mask_logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        if return_dict:
            res: Dict[str, Any] = {
                "mask_logits": mask_logits,
                "pred_mask": torch.sigmoid(mask_logits),
            }
            if return_features:
                res["features"] = {
                    "h8": s2_feat,
                    "lff": lff_fused,
                }
            return res
        return mask_logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        out = self.forward(x, return_dict=True)
        logits = out["mask_logits"] if isinstance(out, dict) else out
        return (torch.sigmoid(logits) >= threshold).float()
