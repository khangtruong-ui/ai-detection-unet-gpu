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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, 1, -1, rel_pos.shape[-1]).permute(0, 3, 1, 2),
                size=(1, max_rel_dist),
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        q_coords = torch.arange(q_size)[:, None]
        k_coords = torch.arange(k_size)[None, :]
        rel_coords = (q_coords - k_coords) + (k_size - 1)
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
# Unified FOCAL Model
# ------------------------------------------------------------------------------

class FOCAL(nn.Module):
    """
    FOCAL: Forensic Contrastive Clustering Model for Image Forgery Detection & Localization.

    Supports:
    - Backbone: "hrnet" (default, lightweight 32-dim features) or "vit" (SAM ViT-L, 256-dim features).
    - Pretrained weights loaded from Hugging Face repository `KhangTruong/FOCAL` with strict verification.
    - Zero fallback to random/dummy weights: missing weights raise an immediate RuntimeError.

    Args:
        backbone (str): "hrnet" or "vit" (default: "hrnet").
        pretrained (bool): Whether to load genuine pretrained weights (default: True, strictly required).
        checkpoint_path (Optional[str]): Local file path or HF repo path to weights.
        hf_repo (str): Hugging Face repository containing pretrained checkpoints.
        cache_dir (Optional[str]): Local cache directory for weights.
        aux_classifier (bool): Whether to enable auxiliary class logits (default: False).
        num_classes (int): Auxiliary class count (default: 2).
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
        **kwargs: Any,
    ):
        super().__init__()
        self.backbone_name = str(backbone).lower().strip()
        self.aux_classifier = bool(aux_classifier)
        self.num_classes = int(num_classes)
        self.hf_repo = hf_repo

        if not pretrained:
            raise ValueError(
                f"FOCAL requires pretrained=True. Fallback to random weights is strictly disabled."
            )

        # Build architecture
        if self.backbone_name in ("hrnet", "hrnet32", "hrnet-w32"):
            self.net = HRNet()
            in_dim = 32
            default_filename = DEFAULT_FOCAL_HRNET_CKPT
        elif self.backbone_name in ("vit", "vit-l", "sam_vit"):
            # ImageEncoderViT matching SAM ViT-L in FOCAL
            self.net = FOCALViTNet(
                img_size=1024,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                out_chans=256,
            )
            in_dim = 256
            default_filename = DEFAULT_FOCAL_VIT_CKPT
        else:
            raise ValueError(f"Unsupported FOCAL backbone '{backbone}'. Supported: 'hrnet', 'vit'.")

        # 2-class Convolutional classification head (as stored in genuine checkpoint)
        self.fc = nn.Conv2d(in_dim, 2, kernel_size=3, padding=1)

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

        # Load weights with strict zero-fallback policy
        self._load_pretrained(
            source=checkpoint_path or default_filename,
            default_filename=default_filename,
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
        clean_sd = {}
        expected_keys = set(self.state_dict().keys())
        if self.class_head is not None:
            expected_keys = {k for k in expected_keys if not k.startswith("class_head.")}

        for k, v in raw_sd.items():
            clean_k = k[7:] if k.startswith("module.") else k
            clean_sd[clean_k] = v
            # If standalone backbone was saved without 'net.' prefix, support matching to 'net.<k>'
            if not clean_k.startswith("net.") and f"net.{clean_k}" in expected_keys:
                clean_sd[f"net.{clean_k}"] = v

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

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial forensic features from input images."""
        orig_h, orig_w = x.shape[2], x.shape[3]
        if self.backbone_name in ("vit", "vit-l", "sam_vit") and (orig_h != 1024 or orig_w != 1024):
            x_in = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
            return self.net(x_in)
        return self.net(x)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass of FOCAL.

        Args:
            x: Input RGB tensor `[B, 3, H, W]`.
            return_dict: If True, returns dictionary with standard keys.

        Returns:
            Dictionary with:
                - `mask_logits`: `[B, 1, H, W]`
                - `pred_mask`: Sigmoid probability `[B, 1, H, W]`
                - `features`: Dense feature map `[B, C, H', W']`
                - `class_logits`: `[B, num_classes]` (if aux_classifier)
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        feats = self.extract_features(x)  # [B, C, H', W']
        logits_small = self.fc(feats)  # [B, 2, H', W']

        # Interpolate to input resolution
        logits_full = F.interpolate(logits_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        # Single binary channel logit (forged class 1 minus pristine class 0)
        mask_logits = logits_full[:, 1:2] - logits_full[:, 0:1]

        class_logits = self.class_head(feats) if self.class_head is not None else None

        if return_dict:
            res: Dict[str, Any] = {
                "mask_logits": mask_logits,
                "raw_logits": logits_full,
                "pred_mask": torch.sigmoid(mask_logits),
                "features": feats,
            }
            if class_logits is not None:
                res["class_logits"] = class_logits
                res["pred_label"] = torch.argmax(class_logits, dim=1)
            return res

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
            feats = self.extract_features(x)  # [B, C, H', W']
            b, c, h_f, w_f = feats.shape

            # L2-normalize feature vectors across channel dimension
            feats_norm = F.normalize(feats, p=2, dim=1)

            masks = []
            for b_idx in range(b):
                f_img = feats_norm[b_idx].permute(1, 2, 0).reshape(-1, c)  # [HW, C]
                # Mini-batch k-means (k=2) via simple EM iterations
                # Initialize centroids from farthest points
                c1 = f_img[0]
                dists = 1.0 - torch.matmul(f_img, c1)
                c2 = f_img[torch.argmax(dists)]
                centroids = torch.stack([c1, c2], dim=0)  # [2, C]

                for _ in range(5):
                    # Cosine distance
                    sims = torch.matmul(f_img, centroids.t())  # [HW, 2]
                    assign = torch.argmax(sims, dim=-1)  # [HW]
                    m0 = f_img[assign == 0]
                    m1 = f_img[assign == 1]
                    if len(m0) > 0:
                        centroids[0] = F.normalize(m0.mean(dim=0), p=2, dim=-1)
                    if len(m1) > 0:
                        centroids[1] = F.normalize(m1.mean(dim=0), p=2, dim=-1)

                sims = torch.matmul(f_img, centroids.t())
                labels = torch.argmax(sims, dim=-1).float()  # [HW]
                # In forgery detection, tampered region is typically smaller than pristine
                if labels.sum() > (len(labels) / 2.0):
                    labels = 1.0 - labels

                mask_grid = labels.reshape(1, 1, h_f, w_f)
                mask_full = F.interpolate(mask_grid, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                masks.append(mask_full)

            return torch.cat(masks, dim=0)


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
