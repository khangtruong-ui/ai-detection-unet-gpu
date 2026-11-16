"""
FOCAL: Forensic Contrastive Clustering for Image Forgery Detection and Localization.
Official Paper: "Rethinking Image Forgery Detection via Soft Contrastive Learning and Unsupervised Clustering"
Official Repository: https://github.com/HighwayWu/FOCAL
Pretrained Weights: KhangTruong/FOCAL (Hugging Face)

Strict zero-fallback policy:
- No local random initialization: `pretrained=False` raises an immediate ValueError.
- If weights cannot be downloaded/loaded, an immediate RuntimeError is raised.
- No dummy/fallback weights are permitted.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.mask_clip import SurgicalLoRALinear

logger = logging.getLogger("sid_unet.models.focal")

DEFAULT_FOCAL_HF_REPO = "KhangTruong/FOCAL"
DEFAULT_FOCAL_HRNET_CKPT = "FOCAL_HRNet_weights.pth"
DEFAULT_FOCAL_VIT_CKPT = "FOCAL_ViT_weights.pth"


# ------------------------------------------------------------------------------
# HRNet Backbone for FOCAL
# ------------------------------------------------------------------------------

BN_MOMENTUM = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        blocks: Any,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(
        self, branch_index: int, block: Any, num_blocks: List[int], num_channels: List[int], stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(
        self, num_branches: int, block: Any, num_blocks: List[int], num_channels: List[int]
    ) -> nn.ModuleList:
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> Optional[nn.ModuleList]:
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> List[int]:
        return self.num_inchannels

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Module):
    """HRNet-W32 architecture matching FOCAL HRNet backbone."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # Stage 2
        self.stage2_cfg = {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [32, 64], "BLOCK": "BASIC", "FUSE_METHOD": "SUM"}
        self.transition1 = self._make_transition_layer([256], [32, 64])
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, [32, 64])

        # Stage 3
        self.stage3_cfg = {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [32, 64, 128], "BLOCK": "BASIC", "FUSE_METHOD": "SUM"}
        self.transition2 = self._make_transition_layer(pre_stage_channels, [32, 64, 128])
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, [32, 64, 128])

        # Stage 4
        self.stage4_cfg = {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [32, 64, 128, 256], "BLOCK": "BASIC", "FUSE_METHOD": "SUM"}
        self.transition3 = self._make_transition_layer(pre_stage_channels, [32, 64, 128, 256])
        self.stage4, _ = self._make_stage(self.stage4_cfg, [32, 64, 128, 256], multi_scale_output=False)

    def _make_layer(self, block: Any, inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_inchannels: List[int], num_channels: List[int]) -> nn.ModuleList:
        num_branches_pre = len(num_inchannels)
        num_branches_cur = len(num_channels)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels[i] != num_inchannels[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[i], num_channels[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_inchannels[-1]
                    outchannels = num_channels[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_stage(
        self, layer_config: Dict[str, Any], num_inchannels: List[int], multi_scale_output: bool = True
    ) -> Tuple[nn.Sequential, List[int]]:
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = BasicBlock
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output if i == num_modules - 1 else True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        feats = self.stage4(x_list)[0]  # [B, 32, H // 4, W // 4]
        return feats


# ------------------------------------------------------------------------------
# ViT (SAM ViT-L Image Encoder) for FOCAL
# ------------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to Patch Embedding for ViT."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)  # [B, H, W, C]


def window_partition_sam(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (hp, wp)


def window_unpartition_sam(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)
    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


class SAMAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        input_size: Optional[Tuple[int, int]] = (64, 64),
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = self._add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (h, w), (h, w))

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, h, w, c)
        return self.proj(out)

    @staticmethod
    def _add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        qh, qw = q_size
        kh, kw = k_size
        rh = SAMAttention._get_rel_pos(qh, kh, rel_pos_h)
        rw = SAMAttention._get_rel_pos(qw, kw, rel_pos_w)
        b, _, dim = q.shape[0], q.shape[1], q.shape[-1]
        r_q = q.view(b, q.shape[1], qh, qw, dim)
        rel_h = torch.einsum("bhxyd,xkd->bhxyk", r_q, rh)
        rel_w = torch.einsum("bhxyd,ykd->bhxyk", r_q, rw)
        attn = (
            attn.view(b, q.shape[1], qh, qw, kh, kw)
            + rel_h.unsqueeze(-1)
            + rel_w.unsqueeze(-2)
        ).view(b, q.shape[1], qh * qw, kh * kw)
        return attn

    @staticmethod
    def _get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != max_rel_dist:
            # F.interpolate(..., mode="linear") requires a 3D [N, C, L] tensor
            rel_pos_resized = F.interpolate(
                rel_pos.t().unsqueeze(0).float(),
                size=max_rel_dist,
                mode="linear",
                align_corners=False,
            )
            rel_pos_resized = rel_pos_resized.squeeze(0).t().to(dtype=rel_pos.dtype)
        else:
            rel_pos_resized = rel_pos

        q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
        rel_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos_resized[rel_coords.long()]


class MLPBlock(nn.Module):
    """Feed-forward network used in SAM Transformer blocks."""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Any = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class SAMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 14,
        input_size: Optional[Tuple[int, int]] = (64, 64),
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SAMAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=mlp_dim)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition_sam(x, self.window_size)
            x = self.attn(x)
            x = window_unpartition_sam(x, self.window_size, pad_hw, (h, w))
        else:
            x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoderViT(nn.Module):
    """ViT-Large Image Encoder matching Segment Anything / FOCAL ViT."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (5, 11, 17, 23),
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList([
            SAMBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rel_pos=True,
                window_size=0 if i in global_attn_indexes else window_size,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            for i in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            nn.LayerNorm([out_chans, img_size // patch_size, img_size // patch_size]) if False else nn.Identity(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        )
        # Note: neck LayerNorm in original SAM is a custom LayerNorm2d:
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # Interpolate pos_embed if spatial dimensions differ
            if self.pos_embed.shape[1:3] != x.shape[1:3]:
                pe = self.pos_embed.permute(0, 3, 1, 2)
                pe = F.interpolate(pe, size=x.shape[1:3], mode="bicubic", align_corners=False)
                pe = pe.permute(0, 2, 3, 1)
                x = x + pe
            else:
                x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))  # [B, out_chans, H // 16, W // 16]
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first 2D tensors [B, C, H, W]."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class FOCALViTNet(nn.Module):
    """Wrapper around SAM ViT-Large image encoder matching FOCAL checkpoint hierarchy."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (5, 11, 17, 23),
    ):
        super().__init__()
        self.image_encoder = ImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(x)


# ------------------------------------------------------------------------------
# Lightweight Residual Upsampling Decoder Head (Strategy 1 Mode 1A)
# ------------------------------------------------------------------------------

def _select_num_groups(channels: int, max_groups: int = 16) -> int:
    """Return the largest group count <= max_groups that evenly divides channels."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _ResidualConvBlock(nn.Module):
    """Residual Conv -> GroupNorm -> GELU -> Conv -> GroupNorm refinement block."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _select_num_groups(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FOCALSupervisedHead(nn.Module):
    """
    Lightweight residual progressive upsampling decoder for Strategy 1 Mode 1A.

    Refines H/4 (HRNet) or H/16 (ViT) dense forensic features through residual blocks
    and progressive 2x bilinear upsampling stages before projecting to mask logits.
    """

    def __init__(self, in_dim: int, num_upsample_stages: int = 2, mid_dim: Optional[int] = None):
        super().__init__()
        base_dim = mid_dim if mid_dim is not None else (64 if in_dim <= 64 else 128)
        groups_base = _select_num_groups(base_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups_base, base_dim),
            nn.GELU(),
        )
        self.res_block = _ResidualConvBlock(base_dim)

        stages = []
        cur_dim = base_dim
        for _ in range(max(1, num_upsample_stages)):
            next_dim = max(32, cur_dim // 2)
            groups_next = _select_num_groups(next_dim)
            stages.append(
                nn.Sequential(
                    nn.Conv2d(cur_dim, next_dim, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(groups_next, next_dim),
                    nn.GELU(),
                )
            )
            cur_dim = next_dim
        self.upsample_stages = nn.ModuleList(stages)
        self.out_conv = nn.Conv2d(cur_dim, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
        x = self.res_block(self.stem(feats))
        for stage in self.upsample_stages:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = stage(x)
        logits = self.out_conv(x)
        if logits.shape[2] != orig_h or logits.shape[3] != orig_w:
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return logits


class FeatureAdapter(nn.Module):
    """
    Zero-initialized residual forensic feature adapter (Strategy B).

    Applies `1x1 Conv -> GroupNorm -> GELU -> 1x1 Conv` with zero-initialized final
    projection so `F + Adapter(F) == F` identically at initialization step 0.
    """

    def __init__(self, channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        mid = mid_channels if mid_channels is not None else max(16, channels // 2)
        groups = _select_num_groups(mid)
        self.proj_in = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, mid)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.act(self.norm(self.proj_in(x))))


# ------------------------------------------------------------------------------
# Unified FOCAL Model
# ------------------------------------------------------------------------------

class FOCAL(nn.Module):
    """
    FOCAL: Forensic Contrastive Clustering Model for Image Forgery Detection & Localization.

    Supports:
    - Backbone: "hrnet" (default, lightweight 32-dim features) or "vit" (SAM ViT-L, 256-dim features).
    - Pretrained weights loaded from Hugging Face repository `KhangTruong/FOCAL` with strict verification.
    - Zero fallback to random/dummy weights: missing weights raise an immediate RuntimeError.
    - Strategy 1 Finetuning Modes:
      - `"supervised"` (Mode 1A): Strict train/eval parity using a lightweight residual upsampling
        decoder (`FOCALSupervisedHead`) or legacy `fc` head (`use_clustering=False`).
      - `"contrastive"` (Mode 1B): Image-relative Soft Contrastive Learning (`MyInfoNCE`) during
        training with on-the-fly clustering (`_cluster_features`) and pristine-safety gating in eval.

    Args:
        backbone (str): "hrnet" or "vit" (default: "hrnet").
        pretrained (bool): Whether to load genuine pretrained weights (default: True, strictly required).
        checkpoint_path (Optional[str]): Local file path or HF repo path to weights.
        hf_repo (str): Hugging Face repository containing pretrained checkpoints.
        cache_dir (Optional[str]): Local cache directory for weights.
        aux_classifier (bool): Whether to enable auxiliary class logits (default: False).
        num_classes (int): Auxiliary class count (default: 2).
        use_clustering (bool): Enable unsupervised clustering inference (default: True).
        cluster_temperature (float): Temperature scaling for clustering and MyInfoNCE (default: 0.1).
        finetune_mode (Optional[str]): Strategy 1 mode: `"supervised"`, `"contrastive"`, or `None` (legacy).
        head_type (str): Supervised head variant: `"auto"`, `"decoder"`, or `"legacy"`.
        force_vit_1024 (bool): If True, upsamples ViT inputs to 1024x1024; otherwise uses native size.
        cluster_separability_threshold (float): Pristine-safety threshold on `1 - cos(c0, c1)`.
        aux_pristine_gate (bool): If True and `aux_classifier=True`, gates eval clustering by `class_head`.
        contrastive_sample_size (int): Max spatial tokens per image in `MyInfoNCE` (default: 1024).
    """

    def __init__(
        self,
        backbone: str = "hrnet",
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        hf_repo: str = DEFAULT_FOCAL_HF_REPO,
        cache_dir: Optional[str] = None,
        aux_classifier: bool = False,
        num_classes: int = 2,
        use_clustering: bool = True,
        cluster_temperature: float = 0.1,
        finetune_mode: Optional[str] = None,
        head_type: str = "auto",
        force_vit_1024: bool = False,
        cluster_separability_threshold: float = 0.0015,
        aux_pristine_gate: bool = False,
        contrastive_sample_size: int = 1024,
        # Strategy B: Prior-Preserving PEFT Surgery parameters
        peft_enabled: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        use_dora: bool = True,
        lora_block_scope: Union[str, List[int], Tuple[int, ...]] = "late_and_global",
        lora_last_k_blocks: int = 6,
        lora_target_modules: Optional[List[str]] = None,
        train_neck: bool = True,
        train_vit_norms: bool = False,
        hrnet_trainable_stages: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        freeze_backbone_bn: Optional[bool] = None,
        freeze_stage4_bn: bool = False,
        use_feature_adapter: bool = False,
        prior_anchor_weight: float = 0.0,
        neck_l2_anchor_weight: float = 0.0,
        neck_lr_multiplier: float = 0.25,
        lora_lr_multiplier: float = 1.0,
        lora_weight_decay: float = 0.01,
        **kwargs: Any,
    ):
        super().__init__()
        self.backbone_name = str(backbone).lower().strip()
        self.aux_classifier = bool(aux_classifier)
        self.num_classes = int(num_classes)
        self.hf_repo = hf_repo
        self.cluster_temperature = max(1e-4, float(cluster_temperature))
        self.force_vit_1024 = bool(force_vit_1024)
        self.cluster_separability_threshold = max(0.0, float(cluster_separability_threshold))
        self.aux_pristine_gate = bool(aux_pristine_gate)
        self.contrastive_sample_size = max(16, int(contrastive_sample_size))

        # Strategy B configuration
        if "use_peft" in kwargs and not peft_enabled:
            peft_enabled = bool(kwargs["use_peft"])
        self.peft_enabled = bool(peft_enabled)
        self.lora_r = max(1, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = max(0.0, float(lora_dropout))
        self.use_dora = bool(use_dora)
        self.lora_block_scope = lora_block_scope
        self.lora_last_k_blocks = max(1, int(lora_last_k_blocks))
        self.lora_target_modules: List[str] = (
            [str(m).strip() for m in lora_target_modules]
            if lora_target_modules is not None
            else ["qkv", "proj"]
        )
        self.train_neck = bool(train_neck)
        self.train_vit_norms = bool(train_vit_norms)
        self.hrnet_trainable_stages = hrnet_trainable_stages
        self.use_feature_adapter = bool(use_feature_adapter)
        self.prior_anchor_weight = max(0.0, float(prior_anchor_weight))
        self.neck_l2_anchor_weight = max(0.0, float(neck_l2_anchor_weight))
        self.neck_lr_multiplier = float(neck_lr_multiplier)
        self.lora_lr_multiplier = float(lora_lr_multiplier)
        self.lora_weight_decay = max(0.0, float(lora_weight_decay))
        self.freeze_stage4_bn = bool(freeze_stage4_bn)

        # Normalize Strategy 1 finetuning mode
        raw_mode = None if finetune_mode is None else str(finetune_mode).lower().strip()
        if raw_mode in (None, "", "legacy", "default", "none"):
            self.finetune_mode: Optional[str] = None
            self.use_clustering = bool(use_clustering)
        elif raw_mode in ("supervised", "mode_1a", "1a"):
            self.finetune_mode = "supervised"
            self.use_clustering = False
        elif raw_mode in ("contrastive", "mode_1b", "1b", "scl"):
            self.finetune_mode = "contrastive"
            self.use_clustering = True
        else:
            raise ValueError(
                f"Unsupported FOCAL finetune_mode '{finetune_mode}'. "
                "Supported: 'supervised', 'contrastive', or None."
            )

        raw_head = str(head_type).lower().strip()
        if raw_head in ("auto", ""):
            self.head_type = "decoder" if self.finetune_mode == "supervised" else "legacy"
        elif raw_head in ("decoder", "upsample", "residual", "residual_upsample"):
            self.head_type = "decoder"
        elif raw_head in ("legacy", "fc", "conv"):
            self.head_type = "legacy"
        else:
            raise ValueError(f"Unsupported FOCAL head_type '{head_type}'. Supported: 'auto', 'decoder', 'legacy'.")

        if not pretrained:
            raise ValueError(
                f"FOCAL requires pretrained=True. Fallback to random weights is strictly disabled."
            )

        # Build clean architecture BEFORE loading pretrained weights
        if self.backbone_name in ("hrnet", "hrnet32", "hrnet-w32"):
            self.net = HRNet()
            in_dim = 32
            num_upsample_stages = 2
            default_filename = DEFAULT_FOCAL_HRNET_CKPT
        elif self.backbone_name in ("vit", "vit-l", "sam_vit"):
            self.net = FOCALViTNet(
                img_size=1024,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                out_chans=256,
            )
            in_dim = 256
            num_upsample_stages = 4
            default_filename = DEFAULT_FOCAL_VIT_CKPT
        else:
            raise ValueError(f"Unsupported FOCAL backbone '{backbone}'. Supported: 'hrnet', 'vit'.")

        self.in_dim = in_dim

        # 2-class Convolutional classification head (as stored in genuine checkpoint)
        self.fc = nn.Conv2d(in_dim, 2, kernel_size=3, padding=1)

        # Optional upgraded supervised decoder (Mode 1A)
        self.decoder: Optional[FOCALSupervisedHead] = (
            FOCALSupervisedHead(in_dim=in_dim, num_upsample_stages=num_upsample_stages)
            if self.head_type == "decoder"
            else None
        )

        # Optional zero-initialized residual forensic adapter (Strategy B)
        self.feature_adapter: Optional[FeatureAdapter] = (
            FeatureAdapter(channels=in_dim) if self.use_feature_adapter else None
        )

        # Freeze unused self.fc so DDP never complains about unused parameters
        if self.head_type == "decoder" or self.finetune_mode == "contrastive":
            for p in self.fc.parameters():
                p.requires_grad_(False)

        # Optional auxiliary classifier
        if self.aux_classifier:
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_dim, 128),
                nn.GELU(),
                nn.Linear(128, num_classes),
            )
        else:
            self.class_head = None

        # Strategy B state tracking (populated by _apply_strategy_b_surgery)
        hrnet_stages_explicit = self.hrnet_trainable_stages is not None
        is_hrnet = self.backbone_name in ("hrnet", "hrnet32", "hrnet-w32")
        self._surgery_active = bool(
            self.peft_enabled
            or (is_hrnet and hrnet_stages_explicit)
            or self.use_feature_adapter
            or (freeze_backbone_bn is True)
        )
        if freeze_backbone_bn is None:
            self.freeze_backbone_bn = bool(is_hrnet and self._surgery_active)
        else:
            self.freeze_backbone_bn = bool(freeze_backbone_bn)

        self._surgical_backbone_param_names: Set[str] = set()
        self._frozen_bn_modules: List[nn.Module] = []
        self._adapted_block_indices: List[int] = []
        self._backbone_temporarily_frozen: bool = False
        self._surgery_applied: bool = False
        object.__setattr__(self, "_anchor_neck", None)
        object.__setattr__(self, "_anchor_hrnet_stage4", None)
        self._neck_init_params: Dict[str, torch.Tensor] = {}

        # Step 2 (strict init order): Load & verify pretrained base weights FIRST
        self._load_pretrained(
            source=checkpoint_path or default_filename,
            default_filename=default_filename,
            cache_dir=cache_dir,
        )

        # Step 3 (strict init order): Apply Strategy B surgery AFTER base weights are loaded
        if self._surgery_active and not self._surgery_applied:
            self._apply_strategy_b_surgery()
        elif not self._surgery_active:
            self._surgical_backbone_param_names = {n for n, _ in self.net.named_parameters()}

        if self._surgery_active:
            self.log_surgery_summary()

    @property
    def requires_gt_masks(self) -> bool:
        """True when Mode 1B contrastive training requires ground-truth masks in forward()."""
        return self.finetune_mode == "contrastive"

    def _resolve_vit_block_indices(self) -> List[int]:
        """Resolve which SAMBlock indices receive LoRA/DoRA surgery."""
        depth = len(self.net.image_encoder.blocks)
        global_idxs = {5, 11, 17, 23} & set(range(depth))
        scope = self.lora_block_scope

        if isinstance(scope, (list, tuple, set)):
            resolved = sorted({int(i) for i in scope})
            for i in resolved:
                if i < 0 or i >= depth:
                    raise ValueError(f"lora_block_scope index {i} out of range [0, {depth - 1}].")
            return resolved

        scope_str = str(scope).lower().strip()
        if scope_str in ("late_and_global", "default", ""):
            late_start = max(0, depth - self.lora_last_k_blocks)
            return sorted(global_idxs | set(range(late_start, depth)))
        if scope_str == "last_k":
            late_start = max(0, depth - self.lora_last_k_blocks)
            return list(range(late_start, depth))
        if scope_str == "global_only":
            return sorted(global_idxs)
        if scope_str == "all":
            return list(range(depth))
        if scope_str in ("none", "off"):
            return []
        raise ValueError(
            f"Unsupported lora_block_scope '{scope}'. "
            "Supported: 'late_and_global', 'last_k', 'global_only', 'all', 'none', or list of ints."
        )

    def _normalize_hrnet_stages(self) -> List[str]:
        """Normalize hrnet_trainable_stages config into canonical stage tokens."""
        raw = self.hrnet_trainable_stages
        if raw is None:
            return ["stage4"] if self.peft_enabled else ["all"]
        if isinstance(raw, str):
            tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
        else:
            tokens = [str(t).strip().lower() for t in raw if str(t).strip()]
        return tokens or ["none"]

    def inject_peft_lora(self) -> List[int]:
        """
        Apply Strategy B LoRA/DoRA surgery onto `FOCALViTNet` blocks and freeze all
        unadapted base parameters (`patch_embed`, `pos_embed`, `rel_pos_h/w`, early blocks).
        """
        if self.backbone_name not in ("vit", "vit-l", "sam_vit"):
            raise RuntimeError("inject_peft_lora() is only applicable to FOCAL ViT backbones.")

        # Freeze 100% of self.net first so patch_embed, pos_embed, rel_pos_h/w, and norms stay locked
        for p in self.net.parameters():
            p.requires_grad_(False)

        block_indices = self._resolve_vit_block_indices()
        valid_targets = {"qkv", "proj", "lin1", "lin2"}
        for t in self.lora_target_modules:
            if t not in valid_targets:
                raise ValueError(
                    f"Unsupported lora_target_modules entry '{t}'. Supported: {sorted(valid_targets)}."
                )

        encoder = self.net.image_encoder
        for idx in block_indices:
            blk = encoder.blocks[idx]
            if "qkv" in self.lora_target_modules and isinstance(blk.attn.qkv, nn.Linear):
                blk.attn.qkv = SurgicalLoRALinear(
                    blk.attn.qkv,
                    r=self.lora_r,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                    use_dora=self.use_dora,
                )
            if "proj" in self.lora_target_modules and isinstance(blk.attn.proj, nn.Linear):
                blk.attn.proj = SurgicalLoRALinear(
                    blk.attn.proj,
                    r=self.lora_r,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                    use_dora=self.use_dora,
                )
            if "lin1" in self.lora_target_modules and isinstance(blk.mlp.lin1, nn.Linear):
                blk.mlp.lin1 = SurgicalLoRALinear(
                    blk.mlp.lin1,
                    r=self.lora_r,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                    use_dora=self.use_dora,
                )
            if "lin2" in self.lora_target_modules and isinstance(blk.mlp.lin2, nn.Linear):
                blk.mlp.lin2 = SurgicalLoRALinear(
                    blk.mlp.lin2,
                    r=self.lora_r,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                    use_dora=self.use_dora,
                )
            if self.train_vit_norms:
                for p in blk.norm1.parameters():
                    p.requires_grad_(True)
                for p in blk.norm2.parameters():
                    p.requires_grad_(True)

        if self.train_neck:
            for p in encoder.neck.parameters():
                p.requires_grad_(True)

        self._adapted_block_indices = block_indices
        return block_indices

    def _apply_hrnet_surgery(self) -> None:
        """Apply stage-wise freezing and BatchNorm2d running-stat protection on HRNet."""
        stages = self._normalize_hrnet_stages()
        stage_modules: Dict[str, List[nn.Module]] = {
            "stem": [self.net.conv1, self.net.bn1, self.net.conv2, self.net.bn2],
            "layer1": [self.net.layer1],
            "transition1": [self.net.transition1],
            "stage2": [self.net.stage2],
            "transition2": [self.net.transition2],
            "stage3": [self.net.stage3],
            "transition3": [self.net.transition3],
            "stage4": [self.net.stage4],
            "stage4_last": [self.net.stage4[-1]],
        }

        if "all" not in stages:
            for p in self.net.parameters():
                p.requires_grad_(False)
            for s in stages:
                if s in ("none", "off"):
                    continue
                if s not in stage_modules:
                    raise ValueError(
                        f"Unsupported hrnet_trainable_stages entry '{s}'. "
                        f"Supported: {sorted(list(stage_modules.keys()) + ['all', 'none'])}."
                    )
                for mod in stage_modules[s]:
                    for p in mod.parameters():
                        p.requires_grad_(True)

        # Lock BatchNorm2d running stats & affine params for all frozen stages (and optionally stage4)
        self._frozen_bn_modules = []
        if self.freeze_backbone_bn:
            for mod_name, mod in self.net.named_modules():
                if isinstance(mod, nn.BatchNorm2d):
                    in_stage4 = mod_name.startswith("stage4")
                    has_trainable_param = any(p.requires_grad for p in mod.parameters())
                    if (not has_trainable_param) or (in_stage4 and self.freeze_stage4_bn):
                        for p in mod.parameters():
                            p.requires_grad_(False)
                        mod.eval()
                        self._frozen_bn_modules.append(mod)

    def _init_prior_anchors(self) -> None:
        """Snapshot minimal frozen anchor submodules/weights without duplicating the full backbone."""
        object.__setattr__(self, "_anchor_neck", None)
        object.__setattr__(self, "_anchor_hrnet_stage4", None)
        self._neck_init_params = {}

        if self.backbone_name in ("vit", "vit-l", "sam_vit"):
            if self.prior_anchor_weight > 0.0 and self.train_neck:
                anchor_neck = copy.deepcopy(self.net.image_encoder.neck)
                anchor_neck.eval()
                for p in anchor_neck.parameters():
                    p.requires_grad_(False)
                object.__setattr__(self, "_anchor_neck", anchor_neck)
            if self.neck_l2_anchor_weight > 0.0 and self.train_neck:
                self._neck_init_params = {
                    name: p.detach().clone()
                    for name, p in self.net.image_encoder.neck.named_parameters()
                }
        elif self.backbone_name in ("hrnet", "hrnet32", "hrnet-w32"):
            if self.prior_anchor_weight > 0.0 and self._surgical_backbone_param_names:
                anchor_s4 = copy.deepcopy(self.net.stage4)
                anchor_s4.eval()
                for p in anchor_s4.parameters():
                    p.requires_grad_(False)
                object.__setattr__(self, "_anchor_hrnet_stage4", anchor_s4)

    def _apply(self, fn: Any) -> "FOCAL":
        """Ensure non-registered anchor helper modules follow `.to(device)` / `.to(dtype)` calls."""
        super()._apply(fn)
        anchor_neck = getattr(self, "_anchor_neck", None)
        if anchor_neck is not None:
            anchor_neck._apply(fn)
        anchor_s4 = getattr(self, "_anchor_hrnet_stage4", None)
        if anchor_s4 is not None:
            anchor_s4._apply(fn)
        return self

    def _apply_strategy_b_surgery(self) -> None:
        """Apply configured Strategy B model surgery and record authoritative trainable mask."""
        if self._surgery_applied:
            return
        if self.backbone_name in ("vit", "vit-l", "sam_vit"):
            if self.peft_enabled:
                self.inject_peft_lora()
            elif not self.train_neck:
                for p in self.net.image_encoder.neck.parameters():
                    p.requires_grad_(False)
        elif self.backbone_name in ("hrnet", "hrnet32", "hrnet-w32"):
            self._apply_hrnet_surgery()

        self._surgical_backbone_param_names = {
            name for name, p in self.net.named_parameters() if p.requires_grad
        }
        self._init_prior_anchors()
        self._surgery_applied = True
        self._enforce_frozen_backbone_eval()

    def _enforce_frozen_backbone_eval(self) -> None:
        """Keep permanently frozen BatchNorm2d layers (or fully frozen backbone) in eval mode."""
        if self._backbone_temporarily_frozen:
            self.net.eval()
        elif self._surgery_active:
            if not self._surgical_backbone_param_names:
                self.net.eval()
            else:
                for bn in self._frozen_bn_modules:
                    bn.eval()
        if self._anchor_neck is not None:
            self._anchor_neck.eval()
        if self._anchor_hrnet_stage4 is not None:
            self._anchor_hrnet_stage4.eval()

    def train(self, mode: bool = True) -> "FOCAL":
        """Override train() so frozen backbone BatchNorm2d stats never leak during Trainer.train_epoch()."""
        super().train(mode)
        if mode:
            self._enforce_frozen_backbone_eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze backbone (`self.net`) respecting Strategy B surgical freeze masks."""
        self._backbone_temporarily_frozen = not bool(trainable)
        if not trainable:
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.net.eval()
            return

        if self._surgery_active:
            for name, p in self.net.named_parameters():
                p.requires_grad_(name in self._surgical_backbone_param_names)
            self.net.train(self.training)
            self._enforce_frozen_backbone_eval()
        else:
            for p in self.net.parameters():
                p.requires_grad_(True)
            self.net.train(self.training)

    def get_param_groups(
        self,
        lr: float,
        backbone_lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build optimizer parameter groups separating head, LoRA/DoRA adapters, and
        trainable neck / partial HRNet stages without exposing frozen base weights.
        """
        if not self._surgery_active:
            if backbone_lr is None:
                return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": float(lr)}]

            backbone_param_ids = {id(p) for p in self.net.parameters()}
            head_params = [p for p in self.parameters() if id(p) not in backbone_param_ids and p.requires_grad]
            backbone_params = [p for p in self.net.parameters()]

            groups: List[Dict[str, Any]] = []
            if head_params:
                groups.append({"params": head_params, "lr": float(lr), "name": "head"})
            if backbone_params:
                groups.append({"params": backbone_params, "lr": float(backbone_lr), "name": "backbone"})
            return groups

        # Strategy B surgical parameter groups
        backbone_param_ids = {id(p) for p in self.net.parameters()}
        anchor_param_ids: Set[int] = set()
        if self._anchor_neck is not None:
            anchor_param_ids.update(id(p) for p in self._anchor_neck.parameters())
        if self._anchor_hrnet_stage4 is not None:
            anchor_param_ids.update(id(p) for p in self._anchor_hrnet_stage4.parameters())

        head_params = [
            p for p in self.parameters()
            if id(p) not in backbone_param_ids and id(p) not in anchor_param_ids and p.requires_grad
        ]

        lora_params: List[nn.Parameter] = []
        neck_params: List[nn.Parameter] = []
        partial_bb_params: List[nn.Parameter] = []

        for name, p in self.net.named_parameters():
            if name not in self._surgical_backbone_param_names:
                continue
            if name.endswith((".lora_A", ".lora_B", ".dora_m")):
                lora_params.append(p)
            elif "image_encoder.neck." in name or name.startswith("neck."):
                neck_params.append(p)
            else:
                partial_bb_params.append(p)

        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": float(lr), "name": "head"})
        if lora_params:
            groups.append({
                "params": lora_params,
                "lr": float(lr) * self.lora_lr_multiplier,
                "weight_decay": float(self.lora_weight_decay),
                "name": "lora",
            })
        if neck_params:
            resolved_neck_lr = float(lr) * self.neck_lr_multiplier
            groups.append({"params": neck_params, "lr": resolved_neck_lr, "name": "neck"})
        if partial_bb_params:
            resolved_bb_lr = float(backbone_lr) if backbone_lr is not None else float(lr) * self.neck_lr_multiplier
            groups.append({"params": partial_bb_params, "lr": resolved_bb_lr, "name": "backbone_partial"})

        if not groups:
            trainable = [p for p in self.parameters() if p.requires_grad]
            if trainable:
                groups.append({"params": trainable, "lr": float(lr), "name": "head"})
        return groups

    def merge_peft_weights(self) -> "FOCAL":
        """
        Losslessly merge all `SurgicalLoRALinear` adapters (LoRA & DoRA) back into
        plain `nn.Linear` layers for zero-overhead inference and standard cross-eval.
        """
        merged_count = 0

        def _replace_recursive(parent: nn.Module) -> None:
            nonlocal merged_count
            for child_name, child in list(parent.named_children()):
                if isinstance(child, SurgicalLoRALinear):
                    setattr(parent, child_name, child.to_merged_linear())
                    merged_count += 1
                else:
                    _replace_recursive(child)

        _replace_recursive(self.net)
        self.peft_enabled = False
        self._surgical_backbone_param_names = {
            name for name, p in self.net.named_parameters() if p.requires_grad
        }
        if merged_count > 0:
            logger.info(f"🔀 Merged {merged_count} SurgicalLoRALinear adapter(s) into standard nn.Linear layers.")
        return self

    def get_surgery_summary(self) -> Dict[str, Any]:
        """Return parameter footprint and active Strategy B surgery metadata."""
        anchor_ids: Set[int] = set()
        if self._anchor_neck is not None:
            anchor_ids.update(id(p) for p in self._anchor_neck.parameters())
        if self._anchor_hrnet_stage4 is not None:
            anchor_ids.update(id(p) for p in self._anchor_hrnet_stage4.parameters())

        total_params = sum(p.numel() for p in self.parameters() if id(p) not in anchor_ids)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in anchor_ids)
        backbone_total = sum(p.numel() for p in self.net.parameters())
        backbone_trainable = sum(
            p.numel() for n, p in self.net.named_parameters() if n in self._surgical_backbone_param_names
        )
        return {
            "backbone": self.backbone_name,
            "finetune_mode": self.finetune_mode,
            "peft_enabled": self.peft_enabled,
            "use_dora": self.use_dora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "adapted_block_indices": list(self._adapted_block_indices),
            "lora_target_modules": list(self.lora_target_modules),
            "train_neck": self.train_neck,
            "train_vit_norms": self.train_vit_norms,
            "hrnet_trainable_stages": self.hrnet_trainable_stages,
            "freeze_backbone_bn": self.freeze_backbone_bn,
            "frozen_bn_count": len(self._frozen_bn_modules),
            "use_feature_adapter": self.use_feature_adapter,
            "prior_anchor_weight": self.prior_anchor_weight,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_pct": (100.0 * trainable_params / max(1, total_params)),
            "backbone_total_params": backbone_total,
            "backbone_trainable_params": backbone_trainable,
            "backbone_trainable_pct": (100.0 * backbone_trainable / max(1, backbone_total)),
        }

    def log_surgery_summary(self, custom_logger: Optional[logging.Logger] = None) -> None:
        """Log concise Strategy B startup surgery summary."""
        lg = custom_logger or logger
        s = self.get_surgery_summary()
        lg.info(
            f"🧬 [FOCAL Strategy B Surgery] backbone={s['backbone']} mode={s['finetune_mode']} | "
            f"trainable={s['trainable_params']:,}/{s['total_params']:,} ({s['trainable_pct']:.2f}%) | "
            f"backbone_trainable={s['backbone_trainable_params']:,}/{s['backbone_total_params']:,} ({s['backbone_trainable_pct']:.2f}%) | "
            f"peft={s['peft_enabled']} dora={s['use_dora']} blocks={s['adapted_block_indices']} "
            f"train_neck={s['train_neck']} frozen_bn={s['frozen_bn_count']} "
            f"adapter={s['use_feature_adapter']} prior_anchor={s['prior_anchor_weight']}"
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
                os.path.join("/tmp/focal_repo/weights", filename),
                os.path.join("pretrained_models", filename),
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
            if self.backbone_name == "hrnet":
                for c in ["checkpoints/checkpoint_best.pt", "checkpoint_best.pt", "FOCAL_HRNet_weights.pth", "checkpoints/FOCAL_HRNet_weights.pth"]:
                    if c not in hf_candidates:
                        hf_candidates.append(c)
            else:
                for c in ["checkpoints/focal_vit.pt", "focal_vit.pt", "checkpoints/checkpoint_best.pt", "FOCAL_ViT_weights.pth", "checkpoints/FOCAL_ViT_weights.pth"]:
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
                        logger.info(f"Downloaded FOCAL pretrained weights from HF '{repo_id}/{c_fn}': {resolved_path}")
                        break
                except Exception as hf_err:
                    last_err = hf_err
                    continue

            if resolved_path is None and last_err is not None:
                raise RuntimeError(
                    f"Failed to download/load FOCAL weights from Hugging Face repo '{repo_id}' (tried {hf_candidates}): {last_err}. "
                    f"Strict zero-fallback policy: aborting initialization."
                ) from last_err

        if resolved_path is None or not os.path.isfile(resolved_path):
            raise RuntimeError(
                f"Failed to resolve FOCAL weights at '{source}'. "
                f"Strict zero-fallback policy: local initialization is strictly disabled."
            )

        try:
            ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except Exception as load_err:
            raise RuntimeError(
                f"Corrupt or unreadable checkpoint at '{resolved_path}': {load_err}"
            ) from load_err

        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid FOCAL checkpoint payload from '{resolved_path}': expected state_dict dict.")

        if "model_state_dict" in ckpt:
            raw_sd = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            raw_sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            raw_sd = ckpt["state_dict"]
        else:
            raw_sd = ckpt

        # Strip module. prefix (e.g. from DataParallel save)
        clean_sd: Dict[str, Any] = {}
        for k, v in raw_sd.items():
            clean_k = k[7:] if k.startswith("module.") else k
            clean_sd[clean_k] = v

        is_peft_ckpt = any(
            ".lora_A" in k or ".lora_B" in k or ".dora_m" in k or ".base_linear." in k
            for k in clean_sd.keys()
        )
        if is_peft_ckpt and self._surgery_active and not self._surgery_applied:
            # Strategy B checkpoint reload: apply matching PEFT surgery before loading state_dict
            self._apply_strategy_b_surgery()

        expected_keys = set(self.state_dict().keys())
        if self.class_head is not None:
            expected_keys = {k for k in expected_keys if not k.startswith("class_head.")}
        if self.decoder is not None:
            expected_keys = {k for k in expected_keys if not k.startswith("decoder.")}
        if self.feature_adapter is not None:
            expected_keys = {k for k in expected_keys if not k.startswith("feature_adapter.")}
        expected_keys = {
            k for k in expected_keys
            if not k.startswith(("_anchor_neck.", "_anchor_hrnet_stage4."))
        }

        for k in list(clean_sd.keys()):
            # If standalone backbone was saved without 'net.' prefix, support matching to 'net.<k>'
            if not k.startswith("net.") and f"net.{k}" in expected_keys:
                clean_sd[f"net.{k}"] = clean_sd[k]

        try:
            missing_keys = [k for k in expected_keys if k not in clean_sd]
            if missing_keys:
                raise RuntimeError(f"FOCAL missing required pretrained weights: {missing_keys[:10]}")

            missing, unexpected = self.load_state_dict(clean_sd, strict=False)
            logger.info(
                f"✅ Loaded genuine FOCAL {self.backbone_name} pretrained weights from: {resolved_path} "
                f"(missing: {len(missing)}, unexpected: {len(unexpected)})"
            )
        except Exception as sd_err:
            raise RuntimeError(
                f"Failed strict load_state_dict for FOCAL ({self.backbone_name}): {sd_err}. "
                f"Zero-fallback policy enforced."
            ) from sd_err

    def _ensure_raw_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is in raw [0, 1] range as expected by FOCAL."""
        if x.min() < -0.05 or x.max() > 1.05:
            # Shifted / ImageNet standardized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            return torch.clamp(x * std + mean, 0.0, 1.0)
        return x

    def _prepare_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_raw_rgb(x)
        if self.backbone_name in ("vit", "vit-l", "sam_vit") and self.force_vit_1024:
            orig_h, orig_w = x.shape[2], x.shape[3]
            if orig_h != 1024 or orig_w != 1024:
                return F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial forensic features from input images (including optional FeatureAdapter)."""
        x_in = self._prepare_backbone_input(x)
        feats = self.net(x_in)
        if self.feature_adapter is not None:
            feats = self.feature_adapter(feats)
        return feats

    @torch.no_grad()
    def _extract_base_features_no_grad(self, x_in: torch.Tensor) -> torch.Tensor:
        """Compute unmodified pretrained backbone features without duplicating the 307M ViT in VRAM."""
        if self.backbone_name in ("vit", "vit-l", "sam_vit"):
            lora_mods = [m for m in self.net.modules() if isinstance(m, SurgicalLoRALinear)]
            for m in lora_mods:
                m.bypass = True
            try:
                encoder = self.net.image_encoder
                feat = encoder.patch_embed(x_in)
                if encoder.pos_embed is not None:
                    if encoder.pos_embed.shape[1:3] != feat.shape[1:3]:
                        pe = F.interpolate(
                            encoder.pos_embed.permute(0, 3, 1, 2),
                            size=feat.shape[1:3],
                            mode="bicubic",
                            align_corners=False,
                        ).permute(0, 2, 3, 1)
                        feat = feat + pe
                    else:
                        feat = feat + encoder.pos_embed
                for blk in encoder.blocks:
                    feat = blk(feat)
                neck_mod = self._anchor_neck if self._anchor_neck is not None else encoder.neck
                return neck_mod(feat.permute(0, 3, 1, 2))
            finally:
                for m in lora_mods:
                    m.bypass = False

        # HRNet base features
        if self._anchor_hrnet_stage4 is not None:
            h = self.net.relu(self.net.bn1(self.net.conv1(x_in)))
            h = self.net.relu(self.net.bn2(self.net.conv2(h)))
            h = self.net.layer1(h)
            x_list = [
                self.net.transition1[i](h) if self.net.transition1[i] is not None else h
                for i in range(self.net.stage2_cfg["NUM_BRANCHES"])
            ]
            y_list = self.net.stage2(x_list)
            x_list = [
                self.net.transition2[i](y_list[-1]) if self.net.transition2[i] is not None else y_list[i]
                for i in range(self.net.stage3_cfg["NUM_BRANCHES"])
            ]
            y_list = self.net.stage3(x_list)
            x_list = [
                self.net.transition3[i](y_list[-1]) if self.net.transition3[i] is not None else y_list[i]
                for i in range(self.net.stage4_cfg["NUM_BRANCHES"])
            ]
            return self._anchor_hrnet_stage4(x_list)[0]
        return self.net(x_in)

    def _compute_prior_anchor_terms(
        self, x_in: torch.Tensor, feats: torch.Tensor, feats_backbone: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute feature-space cosine prior-anchoring loss and optional neck L2 anchor penalty.
        Returns `(raw_cosine_drift, weighted_total_penalty)`.
        """
        raw_cos_loss: Optional[torch.Tensor] = None
        weighted_penalty: Optional[torch.Tensor] = None

        if self.prior_anchor_weight > 0.0:
            if not self._surgical_backbone_param_names and self.feature_adapter is not None:
                feats_base = feats_backbone.detach()
            else:
                feats_base = self._extract_base_features_no_grad(x_in)
            f_ad = F.normalize(feats.float(), p=2, dim=1)
            f_ba = F.normalize(feats_base.detach().float(), p=2, dim=1)
            raw_cos_loss = (1.0 - (f_ad * f_ba).sum(dim=1)).mean()
            weighted_penalty = self.prior_anchor_weight * raw_cos_loss

        if self.neck_l2_anchor_weight > 0.0 and self._neck_init_params:
            l2_terms = []
            for name, p in self.net.image_encoder.neck.named_parameters():
                if name in self._neck_init_params and p.requires_grad:
                    ref = self._neck_init_params[name].to(device=p.device, dtype=p.dtype)
                    l2_terms.append((p - ref).pow(2).mean())
            if l2_terms:
                neck_l2 = torch.stack(l2_terms).mean()
                weighted_neck = self.neck_l2_anchor_weight * neck_l2
                weighted_penalty = weighted_neck if weighted_penalty is None else (weighted_penalty + weighted_neck)

        return raw_cos_loss, weighted_penalty

    def _apply_pristine_gate(
        self,
        diff: torch.Tensor,
        pristine_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Shift clustering logits negative for images flagged as pristine/unimodal."""
        if not pristine_mask.any():
            return diff
        max_pos = diff.max(dim=-1, keepdim=True).values.clamp(min=0.0)
        shifted = diff - (max_pos + 2.0)
        return torch.where(pristine_mask.view(-1, 1), shifted, diff)

    def _cluster_features_with_stats(
        self,
        feats: torch.Tensor,
        orig_h: int,
        orig_w: int,
        aux_pristine_flags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        On-the-fly unsupervised k-means clustering mask and logit prediction as proposed in FOCAL.
        Clusters L2-normalized forensic embeddings into tampered and pristine groups per image,
        and returns `(logit_full, prob_full, cluster_separability)`.
        """
        b, c, h_f, w_f = feats.shape
        n_px = h_f * w_f
        feats_norm = F.normalize(feats.float(), p=2, dim=1)
        f_img = feats_norm.permute(0, 2, 3, 1).reshape(b, n_px, c)

        # Double farthest-point initialization (k-means++ style) across entire batch
        c1 = f_img[:, 0:1, :]
        dists1 = 1.0 - torch.bmm(f_img, c1.transpose(1, 2)).squeeze(-1)
        idx_max1 = torch.argmax(dists1, dim=-1, keepdim=True)
        c2 = torch.gather(f_img, dim=1, index=idx_max1.unsqueeze(-1).expand(-1, 1, c))

        dists2 = 1.0 - torch.bmm(f_img, c2.transpose(1, 2)).squeeze(-1)
        idx_max2 = torch.argmax(dists2, dim=-1, keepdim=True)
        c1 = torch.gather(f_img, dim=1, index=idx_max2.unsqueeze(-1).expand(-1, 1, c))

        centroids = torch.cat([c1, c2], dim=1)  # [B, 2, C]

        # EM iterations with cosine similarity (vectorized without CPU synchronization)
        for _ in range(8):
            sims = torch.bmm(f_img, centroids.transpose(1, 2))  # [B, N, 2]
            assign = torch.argmax(sims, dim=-1)  # [B, N]
            w0 = (assign == 0).unsqueeze(-1).to(f_img.dtype)
            w1 = (assign == 1).unsqueeze(-1).to(f_img.dtype)
            count0 = w0.sum(dim=1, keepdim=True)
            count1 = w1.sum(dim=1, keepdim=True)
            sum0 = torch.bmm(w0.transpose(1, 2), f_img)
            sum1 = torch.bmm(w1.transpose(1, 2), f_img)
            new_c0 = torch.where(count0 > 0, sum0 / count0.clamp(min=1.0), centroids[:, 0:1, :])
            new_c0 = F.normalize(new_c0, p=2, dim=-1)
            new_c1 = torch.where(count1 > 0, sum1 / count1.clamp(min=1.0), centroids[:, 1:2, :])
            new_c1 = F.normalize(new_c1, p=2, dim=-1)
            centroids = torch.cat([new_c0, new_c1], dim=1)

        cos_sim = (centroids[:, 0] * centroids[:, 1]).sum(dim=-1)  # [B]
        separability = (1.0 - cos_sim).clamp(min=0.0)

        sims = torch.bmm(f_img, centroids.transpose(1, 2))  # [B, N, 2]
        labels = torch.argmax(sims, dim=-1)  # [B, N]
        c1_count = labels.sum(dim=-1, keepdim=True)  # [B, 1]
        c0_count = n_px - c1_count  # [B, 1]

        # Tampered regions typically occupy the minority portion of an image.
        cond = (c1_count > c0_count)  # [B, 1]
        sim_tampered = torch.where(cond, sims[:, :, 0], sims[:, :, 1])  # [B, N]
        sim_pristine = torch.where(cond, sims[:, :, 1], sims[:, :, 0])  # [B, N]

        diff = (sim_tampered - sim_pristine) / self.cluster_temperature  # [B, N]

        pristine_mask = separability < self.cluster_separability_threshold
        if aux_pristine_flags is not None:
            pristine_mask = pristine_mask | aux_pristine_flags.to(device=separability.device, dtype=torch.bool)
        diff = self._apply_pristine_gate(diff, pristine_mask)

        logit_grid = diff.view(b, 1, h_f, w_f)
        logit_full = F.interpolate(logit_grid, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        prob_full = torch.sigmoid(logit_full)

        return logit_full, prob_full, separability

    def _cluster_features(
        self, feats: torch.Tensor, orig_h: int, orig_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        On-the-fly unsupervised k-means clustering mask and logit prediction as proposed in FOCAL.
        Clusters L2-normalized forensic embeddings into tampered and pristine groups per image.
        Fully vectorized across batch B without CPU-GPU synchronization barriers.

        Returns:
            mask_logits: [B, 1, H, W]
            pred_mask: [B, 1, H, W] in [0, 1]
        """
        logit_full, prob_full, _ = self._cluster_features_with_stats(feats, orig_h, orig_w)
        return logit_full, prob_full

    @staticmethod
    def _sample_contrastive_tokens(
        emb_flat: torch.Tensor,
        label_flat: torch.Tensor,
        max_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stratified-balanced spatial token subsampling to bound MyInfoNCE memory to O(K^2)."""
        n_total = emb_flat.shape[0]
        if n_total <= max_tokens:
            return emb_flat, label_flat

        pos_idx = torch.nonzero(label_flat == 1, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(label_flat == 0, as_tuple=False).squeeze(-1)
        half = max_tokens // 2
        n_pos = min(pos_idx.numel(), half)
        n_neg = min(neg_idx.numel(), max_tokens - n_pos)
        if n_pos + n_neg < max_tokens and pos_idx.numel() > n_pos:
            n_pos = min(pos_idx.numel(), max_tokens - n_neg)

        selected = []
        if n_pos > 0:
            perm_p = torch.randperm(pos_idx.numel(), device=emb_flat.device)[:n_pos]
            selected.append(pos_idx[perm_p])
        if n_neg > 0:
            perm_n = torch.randperm(neg_idx.numel(), device=emb_flat.device)[:n_neg]
            selected.append(neg_idx[perm_n])
        idx = torch.cat(selected, dim=0)
        return emb_flat[idx], label_flat[idx]

    def compute_contrastive_loss(
        self, feats: torch.Tensor, gt_masks: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute image-relative Soft Contrastive Learning loss (`MyInfoNCE`, FOCAL Strategy 1 Mode 1B).

        Contrasts L2-normalized dense embeddings strictly within each image against downsampled
        ground-truth masks. Pure-pristine and pure-tampered images safely contribute 0.0 without NaN/Inf.
        """
        zero_anchor = feats.sum() * 0.0
        if gt_masks is None:
            return zero_anchor

        b, c, h_f, w_f = feats.shape
        gt_f = gt_masks.float()
        if gt_f.ndim == 3:
            gt_f = gt_f.unsqueeze(1)
        if gt_f.shape[2] != h_f or gt_f.shape[3] != w_f:
            gt_small = F.interpolate(gt_f, size=(h_f, w_f), mode="area")
        else:
            gt_small = gt_f
        labels_grid = (gt_small >= 0.5).long().view(b, h_f * w_f)

        feats_norm = F.normalize(feats.float(), p=2, dim=1)
        emb_all = feats_norm.permute(0, 2, 3, 1).reshape(b, h_f * w_f, c)

        per_image_losses: List[torch.Tensor] = []
        for i in range(b):
            y_i = labels_grid[i]
            has_pos = bool((y_i == 1).any())
            has_neg = bool((y_i == 0).any())
            if not (has_pos and has_neg):
                continue

            z_i, y_sub = self._sample_contrastive_tokens(emb_all[i], y_i, self.contrastive_sample_size)
            k_i = z_i.shape[0]
            if k_i < 2 or not (y_sub == 1).any() or not (y_sub == 0).any():
                continue

            logits = torch.matmul(z_i, z_i.t()) / self.cluster_temperature  # [K, K]
            self_mask = torch.eye(k_i, dtype=torch.bool, device=z_i.device)
            same_label = (y_sub.unsqueeze(0) == y_sub.unsqueeze(1)) & (~self_mask)
            valid_anchors = same_label.any(dim=1)
            if not valid_anchors.any():
                continue

            logits_masked = logits.masked_fill(self_mask, float("-inf"))
            # Numerical stabilization per anchor row
            row_max = logits_masked.max(dim=-1, keepdim=True).values.detach()
            logits_stable = logits_masked - row_max
            log_denom = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
            log_prob = logits_stable - log_denom

            pos_counts = same_label.sum(dim=-1).clamp(min=1).to(log_prob.dtype)
            loss_per_anchor = -(log_prob.masked_fill(~same_label, 0.0)).sum(dim=-1) / pos_counts
            per_image_losses.append(loss_per_anchor[valid_anchors].mean())

        if not per_image_losses:
            return zero_anchor
        return torch.stack(per_image_losses).mean()

    def _run_supervised_head(
        self, feats: torch.Tensor, orig_h: int, orig_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute active supervised segmentation head (`decoder` or legacy `fc`)."""
        if self.decoder is not None:
            mask_logits = self.decoder(feats, orig_h, orig_w)
            return mask_logits, mask_logits
        logits_small = self.fc(feats)
        logits_full = F.interpolate(logits_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        mask_logits = logits_full[:, 1:2] - logits_full[:, 0:1]
        return mask_logits, logits_full

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        gt_masks: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]:
        """
        Forward pass of FOCAL.

        Args:
            x: Input RGB tensor `[B, 3, H, W]`.
            return_dict: If True, returns dictionary with standard keys.
            gt_masks: Optional ground-truth binary masks `[B, 1, H, W]` for Mode 1B (`contrastive`).

        Returns:
            Dictionary with:
                - `mask_logits`: `[B, 1, H, W]`
                - `pred_mask`: Sigmoid probability `[B, 1, H, W]`
                - `features`: Dense feature map `[B, C, H', W']`
                - `class_logits`: `[B, num_classes]` (if aux_classifier)
                - `align_loss`: Per-image Soft Contrastive Loss (if Mode 1B training)
                - `cluster_separability`: Inter-centroid distance `1 - cos(c0, c1)` (if clustering)
        """
        orig_h, orig_w = x.shape[2], x.shape[3]
        x_in = self._prepare_backbone_input(x)
        feats_backbone = self.net(x_in)
        feats = self.feature_adapter(feats_backbone) if self.feature_adapter is not None else feats_backbone

        prior_anchor_loss: Optional[torch.Tensor] = None
        prior_anchor_penalty: Optional[torch.Tensor] = None
        if self.training and (self.prior_anchor_weight > 0.0 or self.neck_l2_anchor_weight > 0.0):
            prior_anchor_loss, prior_anchor_penalty = self._compute_prior_anchor_terms(
                x_in=x_in, feats=feats, feats_backbone=feats_backbone
            )

        class_logits = self.class_head(feats) if self.class_head is not None else None
        aux_pristine_flags = (
            (class_logits[:, 1] < class_logits[:, 0])
            if (self.aux_pristine_gate and class_logits is not None and class_logits.shape[-1] >= 2)
            else None
        )

        align_loss: Optional[torch.Tensor] = None
        skip_mask_loss = False
        separability: Optional[torch.Tensor] = None

        if self.finetune_mode == "supervised":
            # Mode 1A: strict train/eval parity via supervised head
            mask_logits, logits_full = self._run_supervised_head(feats, orig_h, orig_w)
            pred_mask = torch.sigmoid(mask_logits)
        elif self.finetune_mode == "contrastive":
            # Mode 1B: image-relative contrastive training + clustering eval
            if self.training:
                align_loss = self.compute_contrastive_loss(feats, gt_masks)
                skip_mask_loss = True
                with torch.no_grad():
                    mask_logits, pred_mask, separability = self._cluster_features_with_stats(
                        feats.detach(), orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                    )
                logits_full = mask_logits
            else:
                mask_logits, pred_mask, separability = self._cluster_features_with_stats(
                    feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                )
                logits_full = mask_logits
        else:
            # Legacy mode: preserves backward compatibility with existing zero-shot / audit tests
            if not self.training and self.use_clustering:
                mask_logits, pred_mask, separability = self._cluster_features_with_stats(
                    feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                )
                logits_full = mask_logits
            else:
                mask_logits, logits_full = self._run_supervised_head(feats, orig_h, orig_w)
                pred_mask = torch.sigmoid(mask_logits)

        if return_dict:
            res: Dict[str, Any] = {
                "mask_logits": mask_logits,
                "raw_logits": logits_full,
                "pred_mask": pred_mask,
                "features": feats,
            }
            if class_logits is not None:
                res["class_logits"] = class_logits
                res["pred_label"] = torch.argmax(class_logits, dim=1)
            if align_loss is not None:
                res["align_loss"] = align_loss
                res["skip_mask_loss"] = skip_mask_loss
            if prior_anchor_loss is not None:
                res["prior_anchor_loss"] = prior_anchor_loss
            if prior_anchor_penalty is not None:
                res["prior_anchor_penalty"] = prior_anchor_penalty
            if separability is not None:
                res["cluster_separability"] = separability
            return res

        if align_loss is not None:
            if self.aux_classifier and class_logits is not None:
                return mask_logits, class_logits, align_loss
            return mask_logits, align_loss
        if self.aux_classifier and class_logits is not None:
            return mask_logits, class_logits
        return mask_logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Generate binary prediction mask `[B, 1, H, W]`."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_dict=True)
            prob = out["pred_mask"]
            return (prob > threshold).float()

    def predict_mask_clustering(self, x: torch.Tensor) -> torch.Tensor:
        """
        On-the-fly unsupervised k-means clustering mask prediction as proposed in FOCAL.

        Extracts L2-normalized forensic features and clusters pixels into 2 groups (tampered vs pristine).
        """
        self.eval()
        with torch.no_grad():
            orig_h, orig_w = x.shape[2], x.shape[3]
            feats = self.extract_features(x)
            _, pred_mask = self._cluster_features(feats, orig_h, orig_w)
            return (pred_mask > 0.5).float()


class FOCALHRNet(FOCAL):
    """Shortcut for FOCAL with HRNet backbone."""

    def __init__(self, **kwargs: Any):
        kwargs["backbone"] = "hrnet"
        super().__init__(**kwargs)


class FOCALViT(FOCAL):
    """Shortcut for FOCAL with ViT backbone."""

    def __init__(self, **kwargs: Any):
        kwargs["backbone"] = "vit"
        super().__init__(**kwargs)
