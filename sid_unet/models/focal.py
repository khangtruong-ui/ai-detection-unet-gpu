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
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.mask_clip import SurgicalLoRALinear

logger = logging.getLogger("sid_unet.models.focal")

DEFAULT_FOCAL_HF_REPO = "KhangTruong/FOCAL"
DEFAULT_FOCAL_HRNET_CKPT = "FOCAL_HRNet_weights.pth"
DEFAULT_FOCAL_VIT_CKPT = "FOCAL_ViT_weights.pth"

DOMAIN_SPLIT_LABELS: Tuple[str, ...] = (
    "casia_pristine",
    "casia_tampered",
    "imd_pristine",
    "imd_tampered",
)


def encode_domain_sources(
    sources: Union[Sequence[Any], torch.Tensor, str, None],
    device: Optional[torch.device] = None,
    domain_names: Optional[Sequence[str]] = None,
) -> Optional[torch.Tensor]:
    """
    Encode dataset source labels (e.g. 'casia', 'imd2020') into integer domain IDs.

    Returns a 1D ``torch.long`` tensor of shape ``[B]`` with:
      - ``0``: CASIA (e.g. ``casia``, ``casiav2``, ``casia_v2.0``)
      - ``1``: IMD2020 (e.g. ``imd2020``, ``imd``)
      - ``-1``: unknown / unmapped source
    """
    if sources is None:
        return None
    if isinstance(sources, torch.Tensor):
        out = sources.to(dtype=torch.long)
        return out.to(device=device) if device is not None else out

    items = [sources] if isinstance(sources, str) else list(sources)
    canonical = [str(d).lower().strip() for d in domain_names] if domain_names else ["casia", "imd"]

    ids: List[int] = []
    for item in items:
        if item is None:
            ids.append(-1)
            continue
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            val = int(item)
            ids.append(val if 0 <= val < len(canonical) else -1)
            continue
        s = str(item).lower().strip()
        if not s:
            ids.append(-1)
            continue
        matched = -1
        for dom_idx, token in enumerate(canonical):
            if token in s or (token == "imd2020" and "imd" in s):
                matched = dom_idx
                break
        ids.append(matched)
    return torch.tensor(ids, dtype=torch.long, device=device)


def _collapse_domain_4_to_2_mask_logits(logits_4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collapse 4 domain-split pixel logits [B, 4, H, W]
    (0: casia_pristine, 1: casia_tampered, 2: imd_pristine, 3: imd_tampered)
    into 2-class (pristine vs tampered) binary mask logits [B, 1, H, W] and
    2-channel [pristine, tampered] logits [B, 2, H, W] via exact log-sum-exp.
    """
    pristine_logit = torch.logsumexp(logits_4[:, [0, 2], :, :], dim=1, keepdim=True)
    tampered_logit = torch.logsumexp(logits_4[:, [1, 3], :, :], dim=1, keepdim=True)
    mask_logits = tampered_logit - pristine_logit
    logits_2 = torch.cat([pristine_logit, tampered_logit], dim=1)
    return mask_logits, logits_2


def _collapse_domain_4_to_2_class_logits(logits_4: torch.Tensor) -> torch.Tensor:
    """Collapse 4-class image logits [B, 4] into 2-class [pristine, tampered] logits [B, 2]."""
    pristine = torch.logsumexp(logits_4[:, [0, 2]], dim=1, keepdim=True)
    tampered = torch.logsumexp(logits_4[:, [1, 3]], dim=1, keepdim=True)
    return torch.cat([pristine, tampered], dim=1)


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

    def _forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            if self.pos_embed.shape[1:3] != x.shape[1:3]:
                pe = self.pos_embed.permute(0, 3, 1, 2)
                pe = F.interpolate(pe, size=x.shape[1:3], mode="bicubic", align_corners=False)
                pe = pe.permute(0, 2, 3, 1)
                x = x + pe
            else:
                x = x + self.pos_embed
        return x, len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._forward_tokens(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))  # [B, out_chans, H // 16, W // 16]
        return x

    def forward_with_taps(
        self, x: torch.Tensor, tap_indices: Sequence[int]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Run encoder once and return `(neck_out, {block_idx: [B, embed_dim, H/16, W/16]})`."""
        x, depth = self._forward_tokens(x)
        normalized_taps: Dict[int, int] = {}
        for raw_idx in tap_indices:
            idx = int(raw_idx) if int(raw_idx) >= 0 else depth + int(raw_idx)
            if idx < 0 or idx >= depth:
                raise ValueError(f"ViT tap index {raw_idx} out of range [0, {depth - 1}].")
            normalized_taps[idx] = int(raw_idx)

        taps: Dict[int, torch.Tensor] = {}
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in normalized_taps:
                taps[normalized_taps[idx]] = x.permute(0, 3, 1, 2).contiguous()

        neck_out = self.neck(x.permute(0, 3, 1, 2))
        return neck_out, taps


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

    def forward_with_taps(
        self, x: torch.Tensor, tap_indices: Sequence[int]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Forward pass returning both neck output and intermediate block taps."""
        return self.image_encoder.forward_with_taps(x, tap_indices)


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
    Supports ``out_channels=1`` (binary) or ``out_channels=4`` (domain-split 4-class).
    """

    def __init__(
        self,
        in_dim: int,
        num_upsample_stages: int = 2,
        mid_dim: Optional[int] = None,
        out_channels: int = 1,
    ):
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
        self.out_conv = nn.Conv2d(cur_dim, int(out_channels), kernel_size=1)

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


class SpatialCrossScaleAdapter(nn.Module):
    """
    Zero-initialized spatial cross-scale forensic feature adapter.

    Uses a bottleneck `1x1` projection followed by depthwise `3x3` (local) and
    dilated depthwise `3x3` (`dilation=2`, wider context) convolutions with a
    zero-initialized `1x1` output projection so step-0 features equal `x` identically
    while allowing high-res HRNet edges and global ViT semantics to interact spatially.
    """

    def __init__(self, channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        mid = mid_channels if mid_channels is not None else max(32, min(128, channels // 2))
        groups = _select_num_groups(mid)
        self.proj_in = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.GroupNorm(groups, mid),
            nn.GELU(),
        )
        self.dw_local = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.GroupNorm(groups, mid),
            nn.GELU(),
        )
        self.dw_dilated = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2, groups=mid, bias=False),
            nn.GroupNorm(groups, mid),
            nn.GELU(),
        )
        self.proj_out = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw_dilated(self.dw_local(self.proj_in(x)))
        return x + self.proj_out(h)


class BranchBalancedSpatialAdapter(nn.Module):
    """
    Branch-balanced zero-initialized spatial adapter for dual-backbone concatenation.

    Applies independent `SpatialCrossScaleAdapter` modules to primary (ViT) and
    secondary (HRNet) feature slices before concatenation so low-dimensional branches
    cannot suffer asymmetric perturbation from higher-dimensional branches.
    """

    def __init__(self, primary_dim: int, secondary_dim: int) -> None:
        super().__init__()
        self.primary_dim = int(primary_dim)
        self.secondary_dim = int(secondary_dim)
        self.primary_adapter = SpatialCrossScaleAdapter(channels=self.primary_dim)
        self.secondary_adapter = SpatialCrossScaleAdapter(channels=self.secondary_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_p = self.primary_adapter(x[:, : self.primary_dim])
        f_s = self.secondary_adapter(x[:, self.primary_dim : self.primary_dim + self.secondary_dim])
        return torch.cat([f_p, f_s], dim=1)


class ClusterPolarityHead(nn.Module):
    """
    Antisymmetric, rotation- & flip-invariant cluster polarity disambiguation head.

    Scores each 2-means cluster using spatial geometry and feature dispersion descriptors
    `[area, border_frac, corner_frac, center_dist, spatial_std, intra_disp]` via a shared
    zero-initialized MLP residual over a border-aware geometric prior.
    """

    NUM_FEATURES = 6

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        h_dim = max(16, int(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
        )
        last_linear = self.mlp[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    @staticmethod
    @torch.no_grad()
    def extract_cluster_descriptors(
        labels: torch.Tensor,
        f_img: torch.Tensor,
        centroids: torch.Tensor,
        h_f: int,
        w_f: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract `[B, 6]` flip/rotation-invariant descriptors `(g_0, g_1)` for clusters 0 and 1.
        """
        b, n_px, _ = f_img.shape
        device = f_img.device
        dtype = torch.float32

        grid = labels.view(b, h_f, w_f)
        bmask = torch.zeros((h_f, w_f), dtype=torch.bool, device=device)
        bmask[0, :] = True
        bmask[-1, :] = True
        bmask[:, 0] = True
        bmask[:, -1] = True
        b_flat = bmask.view(1, n_px).expand(b, -1)
        n_border = max(1.0, float(bmask.sum().item()))

        cw = max(1, min(h_f, w_f) // 16)
        cmask = torch.zeros((h_f, w_f), dtype=torch.bool, device=device)
        cmask[:cw, :cw] = True
        cmask[:cw, -cw:] = True
        cmask[-cw:, :cw] = True
        cmask[-cw:, -cw:] = True
        c_flat = cmask.view(1, n_px).expand(b, -1)
        n_corner = max(1.0, float(cmask.sum().item()))

        ys = torch.linspace(-1.0, 1.0, steps=h_f, device=device, dtype=dtype).view(1, h_f, 1).expand(b, h_f, w_f).reshape(b, n_px)
        xs = torch.linspace(-1.0, 1.0, steps=w_f, device=device, dtype=dtype).view(1, 1, w_f).expand(b, h_f, w_f).reshape(b, n_px)
        sqrt2 = math.sqrt(2.0)

        sim_0 = (f_img * centroids[:, 0:1, :]).sum(dim=-1)
        sim_1 = (f_img * centroids[:, 1:2, :]).sum(dim=-1)

        descs: List[torch.Tensor] = []
        for k, sim_k in ((0, sim_0), (1, sim_1)):
            mk = (labels == k).to(dtype)
            cnt = mk.sum(dim=1, keepdim=True).clamp(min=1.0)
            area = mk.mean(dim=1, keepdim=True)
            border_frac = ((labels == k) & b_flat).to(dtype).sum(dim=1, keepdim=True) / n_border
            corner_frac = ((labels == k) & c_flat).to(dtype).sum(dim=1, keepdim=True) / n_corner
            cy = (mk * ys).sum(dim=1, keepdim=True) / cnt
            cx = (mk * xs).sum(dim=1, keepdim=True) / cnt
            center_dist = torch.sqrt(cy.pow(2) + cx.pow(2) + 1e-8) / sqrt2
            var_y = (mk * (ys - cy).pow(2)).sum(dim=1, keepdim=True) / cnt
            var_x = (mk * (xs - cx).pow(2)).sum(dim=1, keepdim=True) / cnt
            spatial_std = torch.sqrt(var_y + var_x + 1e-8) / sqrt2
            intra_disp = (mk * (1.0 - sim_k)).sum(dim=1, keepdim=True) / cnt
            descs.append(torch.cat([area, border_frac, corner_frac, center_dist, spatial_std, intra_disp], dim=-1))

        return descs[0], descs[1]

    def forward(
        self,
        g0: torch.Tensor,
        g1: torch.Tensor,
        z_base: torch.Tensor,
        margin_scale: float = 5.0,
    ) -> torch.Tensor:
        """Return antisymmetric polarity logit (>0 means cluster 0 is tampered)."""
        phi0 = self.mlp(g0.float())
        phi1 = self.mlp(g1.float())
        return float(margin_scale) * z_base.float() + (phi1 - phi0)


class SecondaryTimmBackbone(nn.Module):
    """
    Pretrained ``timm`` secondary feature extractor with multi-scale FPN projection.

    Expects raw ``[0, 1]`` RGB tensors and applies ImageNet standardization internally
    before extracting and fusing the highest-resolution feature stages.
    """

    def __init__(self, model_name: str, out_dim: int = 128) -> None:
        super().__init__()
        clean_name = model_name.split(":", 1)[1] if model_name.startswith("timm:") else model_name
        try:
            import timm

            self.encoder = timm.create_model(
                clean_name,
                pretrained=True,
                features_only=True,
                out_indices=(-2, -1),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load genuine pretrained timm secondary backbone '{clean_name}': {exc}. "
                "Strict zero-fallback policy: aborting initialization."
            ) from exc

        in_channels_list = list(self.encoder.feature_info.channels())
        total_in = sum(in_channels_list)
        groups = _select_num_groups(out_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(total_in, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_dim),
            nn.GELU(),
        )
        self.out_dim = int(out_dim)
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x_norm = (x_raw - self._mean.to(device=x_raw.device, dtype=x_raw.dtype)) / self._std.to(
            device=x_raw.device, dtype=x_raw.dtype
        )
        feats = self.encoder(x_norm)
        target_hw = feats[0].shape[2:]
        aligned = [feats[0]]
        for f in feats[1:]:
            if f.shape[2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            aligned.append(f)
        return self.proj(torch.cat(aligned, dim=1))


class ClusterResidualRefiner(nn.Module):
    """
    Zero-initialized residual refinement head for clustering-anchored hybrid adaptation.

    Takes normalized forensic features and unsupervised clustering logits/probabilities,
    predicts a spatially smooth residual logit correction initialized identically to zero
    so step-0 predictions match pure unsupervised clustering exactly.
    """

    def __init__(
        self,
        in_dim: int,
        num_upsample_stages: int = 2,
        mid_dim: int = 64,
    ) -> None:
        super().__init__()
        mid = max(32, int(mid_dim))
        groups_mid = _select_num_groups(mid)
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim + 2, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups_mid, mid),
            nn.GELU(),
        )
        self.res_block = _ResidualConvBlock(mid)
        stages = []
        cur_dim = mid
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
        self.out_conv = nn.Conv2d(cur_dim, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        feats: torch.Tensor,
        cluster_logits: torch.Tensor,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        h_f, w_f = feats.shape[2], feats.shape[3]
        feats_norm = F.normalize(feats.float(), p=2, dim=1).to(dtype=feats.dtype)
        if cluster_logits.shape[2:] != (h_f, w_f):
            c_grid = F.interpolate(cluster_logits.float(), size=(h_f, w_f), mode="bilinear", align_corners=False)
        else:
            c_grid = cluster_logits.float()
        c_prob = torch.sigmoid(c_grid).to(dtype=feats.dtype)
        c_scaled = torch.tanh(c_grid / 5.0).to(dtype=feats.dtype)
        x = self.res_block(self.stem(torch.cat([feats_norm, c_prob, c_scaled], dim=1)))
        for stage in self.upsample_stages:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = stage(x)
        delta = self.out_conv(x)
        if delta.shape[2] != orig_h or delta.shape[3] != orig_w:
            delta = F.interpolate(delta, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return delta


class _DilatedResBlock(nn.Module):
    """Dilated residual conv block with GroupNorm and GELU."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        groups = _select_num_groups(channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PreNeckForensicProjector(nn.Module):
    """
    Zero-bias orthogonal subspace projector over pre-neck ViT intermediate taps.

    Compresses concatenated multi-tap ViT block features (`[B, C_in, H_t, W_t]`)
    into a compact `proj_dim`-channel L2-normalized forensic embedding without any
    static bias vector (preventing `cone_cos` collapse).
    """

    def __init__(self, in_dim: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim)
        self.proj = nn.Conv2d(self.in_dim, self.proj_dim, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x.float()), p=2, dim=1).to(dtype=x.dtype)


class ImageRelativeClusterRefiner(nn.Module):
    """
    Domain-invariant, rotation-invariant residual refiner for clustering-anchored FOCAL.

    Operates exclusively on image-relative geometric & clustering descriptors (never raw
    backbone feature channels) so adaptation on a single source domain (e.g. CASIA-v2.0)
    cannot memorize domain-specific semantic embeddings and cause OOD forgetting.
    Optionally augments descriptors with fixed SRM high-frequency residual maps (`+2` ch)
    and a zero-initialized non-local affinity-graph propagation block (`P @ h`).
    Outputs zero-initialized `(raw_bias, raw_scale)` maps at full image resolution.
    """

    BASE_DESC_CHANNELS = 10
    MULTI_BRANCH_DESC_CHANNELS = 24
    SRM_DESC_CHANNELS = 2
    BORDER_KNN_DESC_CHANNELS = 2

    def __init__(
        self,
        num_upsample_stages: int = 2,
        mid_dim: int = 48,
        multi_branch: bool = True,
        use_affinity_prop: bool = False,
        use_srm_desc: bool = False,
        use_border_knn_desc: bool = False,
    ) -> None:
        super().__init__()
        self.multi_branch = bool(multi_branch)
        self.use_affinity_prop = bool(use_affinity_prop)
        self.use_srm_desc = bool(use_srm_desc)
        self.use_border_knn_desc = bool(use_border_knn_desc)
        in_ch = self.MULTI_BRANCH_DESC_CHANNELS if self.multi_branch else self.BASE_DESC_CHANNELS
        if self.use_srm_desc:
            in_ch += self.SRM_DESC_CHANNELS
        if self.use_border_knn_desc:
            in_ch += self.BORDER_KNN_DESC_CHANNELS
        mid = max(32, int(mid_dim))
        groups_mid = _select_num_groups(mid)

        # Eagerly register fixed 3-filter SRM residual kernels for DDP buffer safety
        k1 = torch.tensor(
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            dtype=torch.float32,
        ) / 4.0
        k2 = torch.tensor(
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            dtype=torch.float32,
        ) / 12.0
        k3 = torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.float32,
        ) / 2.0
        self.register_buffer("_srm_kernels", torch.stack([k1, k2, k3], dim=0).unsqueeze(1), persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups_mid, mid),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            _DilatedResBlock(mid, dilation=1),
            _DilatedResBlock(mid, dilation=2),
            _DilatedResBlock(mid, dilation=4),
        )
        if self.use_affinity_prop:
            self.aff_norm = nn.GroupNorm(groups_mid, mid)
            self.aff_proj = nn.Conv2d(mid, mid, kernel_size=1, bias=True)
            nn.init.zeros_(self.aff_proj.weight)
            if self.aff_proj.bias is not None:
                nn.init.zeros_(self.aff_proj.bias)
        else:
            self.aff_norm = None
            self.aff_proj = None

        stages = []
        cur_dim = mid
        for _ in range(max(1, num_upsample_stages)):
            next_dim = max(24, cur_dim // 2)
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
        self.out_conv = nn.Conv2d(cur_dim, 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _branch_manifold_stats(feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute rotation-invariant global & multi-scale local cosine discrepancy maps."""
        f_hat = F.normalize(feats.float(), p=2, dim=1)
        mean_dir = F.normalize(f_hat.mean(dim=(2, 3), keepdim=True), p=2, dim=1)
        d_mean = (1.0 - (f_hat * mean_dir).sum(dim=1, keepdim=True)).clamp(0.0, 2.0)
        mu_d = d_mean.mean(dim=(2, 3), keepdim=True)
        std_d = d_mean.std(dim=(2, 3), keepdim=True, unbiased=False)
        d_mean_z = torch.tanh((d_mean - mu_d) / (std_d + 1e-5))

        ctx3 = F.normalize(F.avg_pool2d(f_hat, kernel_size=3, stride=1, padding=1), p=2, dim=1)
        d_loc3 = (1.0 - (f_hat * ctx3).sum(dim=1, keepdim=True)).clamp(0.0, 2.0)

        ctx9 = F.normalize(F.avg_pool2d(f_hat, kernel_size=9, stride=1, padding=4), p=2, dim=1)
        d_loc9 = (1.0 - (f_hat * ctx9).sum(dim=1, keepdim=True)).clamp(0.0, 2.0)

        ctx25 = F.normalize(F.avg_pool2d(f_hat, kernel_size=25, stride=1, padding=12), p=2, dim=1)
        d_loc25 = (1.0 - (f_hat * ctx25).sum(dim=1, keepdim=True)).clamp(0.0, 2.0)
        return d_mean_z, d_loc3, d_loc9, d_loc25

    @staticmethod
    @torch.no_grad()
    def _border_knn_descriptors(
        feats: torch.Tensor,
        grid_size: int = 48,
        k_local: int = 32,
        k_broad: int = 160,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-image rotation/flip-invariant Self-Border k-NN anomaly maps (`[B, 1, H_f, W_f]`)."""
        b, _, h_f, w_f = feats.shape
        g = min(int(grid_size), h_f, w_f)
        f_cur = feats.float()
        if (h_f, w_f) != (g, g):
            f_cur = F.interpolate(f_cur, size=(g, g), mode="bilinear", align_corners=False)
        h_g, w_g = f_cur.shape[2:]
        f_hat = F.normalize(f_cur, p=2, dim=1)

        bw = 2 if min(h_g, w_g) >= 12 else 1
        bmask = torch.zeros((h_g, w_g), dtype=torch.bool, device=feats.device)
        bmask[:bw, :] = True
        bmask[-bw:, :] = True
        bmask[:, :bw] = True
        bmask[:, -bw:] = True
        bidx = bmask.flatten().nonzero(as_tuple=False).squeeze(1)
        n_b = max(1, int(bidx.numel()))

        x_flat = f_hat.flatten(2)  # [B, C, N]
        x_border = x_flat[:, :, bidx]  # [B, C, n_b]
        sim = torch.bmm(x_flat.transpose(1, 2), x_border)  # [B, N, n_b]

        # Exact k=32 / k=160 on 48x48 (n_b=368), proportional scaling on smaller test grids
        if n_b >= 368:
            k1 = max(1, min(int(k_local), n_b))
            k2 = max(1, min(int(k_broad), n_b))
        else:
            k1 = max(1, min(n_b, int(round(n_b * float(k_local) / 368.0))))
            k2 = max(1, min(n_b, int(round(n_b * float(k_broad) / 368.0))))

        top_broad = torch.topk(sim, k=k2, dim=-1).values  # descending along border keys
        d_loc = (1.0 - top_broad[:, :, :k1].mean(dim=-1)).view(b, 1, h_g, w_g)
        d_brd = (1.0 - top_broad.mean(dim=-1)).view(b, 1, h_g, w_g)

        outs: List[torch.Tensor] = []
        for d_map in (d_loc, d_brd):
            mu = d_map.mean(dim=(2, 3), keepdim=True)
            sd = d_map.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-4)
            z_map = torch.tanh((d_map - mu) / sd)
            if (h_g, w_g) != (h_f, w_f):
                z_map = F.interpolate(z_map, size=(h_f, w_f), mode="bilinear", align_corners=False)
            outs.append(z_map)
        return outs[0], outs[1]

    @classmethod
    @torch.no_grad()
    def build_descriptors(
        cls,
        feats: torch.Tensor,
        c_fused_logits: torch.Tensor,
        sep_fused: torch.Tensor,
        area_fused: torch.Tensor,
        primary_branch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        secondary_branch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        multi_branch: bool = True,
        srm_kernels: Optional[torch.Tensor] = None,
        raw_rgb: Optional[torch.Tensor] = None,
        use_srm_desc: bool = False,
        use_border_knn_desc: bool = False,
    ) -> torch.Tensor:
        """Build `[B, 10|24(+2 SRM)(+2 Border-kNN), H_f, W_f]` rotation-invariant image-relative forensic descriptors."""
        b, _, h_f, w_f = feats.shape

        def _grid(t: torch.Tensor) -> torch.Tensor:
            t_f = t.float()
            if t_f.shape[2:] != (h_f, w_f):
                return F.interpolate(t_f, size=(h_f, w_f), mode="bilinear", align_corners=False)
            return t_f

        c_f = _grid(c_fused_logits)
        p_f = torch.sigmoid(c_f)
        t_f = torch.tanh(c_f / 5.0)
        p_clamp = p_f.clamp(1e-6, 1.0 - 1e-6)
        ent_f = -(p_clamp * torch.log(p_clamp) + (1.0 - p_clamp) * torch.log(1.0 - p_clamp)) / math.log(2.0)
        sep_f_map = sep_fused.float().view(b, 1, 1, 1).expand(-1, -1, h_f, w_f)
        area_f_map = area_fused.float().view(b, 1, 1, 1).expand(-1, -1, h_f, w_f)
        d_mean_z_f, d_loc3_f, d_loc9_f, d_loc25_f = cls._branch_manifold_stats(feats)

        desc_list = [
            p_f,
            t_f,
            ent_f,
            sep_f_map,
            area_f_map,
            d_mean_z_f,
            d_loc3_f,
            d_loc9_f,
            d_loc25_f,
        ]
        # Include centered probability contrast to make base block exactly 10 channels
        p_centered = (p_f - p_f.mean(dim=(2, 3), keepdim=True))
        desc_list.append(p_centered)

        if multi_branch:
            if primary_branch is not None and secondary_branch is not None:
                f_p, c_p_full, sep_p, area_p = primary_branch
                f_s, c_s_full, sep_s, area_s = secondary_branch
                c_p = _grid(c_p_full)
                c_s = _grid(c_s_full)
                p_p = torch.sigmoid(c_p)
                t_p = torch.tanh(c_p / 5.0)
                p_s = torch.sigmoid(c_s)
                t_s = torch.tanh(c_s / 5.0)
                agree = p_p * p_s
                disagree = torch.tanh((c_p - c_s) / 5.0)
                sep_p_map = sep_p.float().view(b, 1, 1, 1).expand(-1, -1, h_f, w_f)
                sep_s_map = sep_s.float().view(b, 1, 1, 1).expand(-1, -1, h_f, w_f)
                area_p_map = area_p.float().view(b, 1, 1, 1).expand(-1, -1, h_f, w_f)
                area_s_map = area_s.float().view(b, 1, 1, 1).expand(-1, -1, h_f, w_f)
                d_mean_z_p, _, d_loc9_p, _ = cls._branch_manifold_stats(f_p)
                d_mean_z_s, d_loc3_s, _, _ = cls._branch_manifold_stats(f_s)
                desc_list.extend(
                    [
                        p_p,
                        t_p,
                        p_s,
                        t_s,
                        agree,
                        disagree,
                        sep_p_map,
                        sep_s_map,
                        area_p_map,
                        area_s_map,
                        d_mean_z_p,
                        d_mean_z_s,
                        d_loc9_p,
                        d_loc3_s,
                    ]
                )
            else:
                zeros_mb = torch.zeros(b, cls.MULTI_BRANCH_DESC_CHANNELS - cls.BASE_DESC_CHANNELS, h_f, w_f, device=feats.device, dtype=torch.float32)
                desc_list.append(zeros_mb)

        if use_srm_desc:
            if raw_rgb is not None and srm_kernels is not None:
                gray = raw_rgb.float().mean(dim=1, keepdim=True)
                res = F.conv2d(gray, srm_kernels.to(device=gray.device, dtype=torch.float32), padding=2).clamp(-2.0, 2.0)
                e_map = F.interpolate(res.pow(2).mean(dim=1, keepdim=True), size=(h_f, w_f), mode="area")
                e_log = torch.log1p(e_map * 100.0)
                mu_e = e_log.mean(dim=(2, 3), keepdim=True)
                sd_e = e_log.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-4)
                srm_z = torch.tanh((e_log - mu_e) / sd_e)
                srm_loc = torch.tanh((e_log - F.avg_pool2d(e_log, kernel_size=9, stride=1, padding=4)) / sd_e)
                desc_list.extend([srm_z, srm_loc])
            else:
                desc_list.append(torch.zeros(b, cls.SRM_DESC_CHANNELS, h_f, w_f, device=feats.device, dtype=torch.float32))

        if use_border_knn_desc:
            b_loc_f, b_brd_f = cls._border_knn_descriptors(feats)
            if secondary_branch is not None:
                sec_f = _grid(secondary_branch[0])
                b_loc_s, b_brd_s = cls._border_knn_descriptors(sec_f)
                b_loc = 0.5 * (b_loc_f + b_loc_s)
                b_brd = 0.5 * (b_brd_f + b_brd_s)
            else:
                b_loc, b_brd = b_loc_f, b_brd_f
            desc_list.extend([b_loc, b_brd])

        return torch.cat(desc_list, dim=1).to(dtype=feats.dtype)

    def forward(
        self,
        descriptors: torch.Tensor,
        orig_h: int,
        orig_w: int,
        affinity_P: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.blocks(self.stem(descriptors))
        if self.use_affinity_prop and self.aff_proj is not None and affinity_P is not None:
            b, c, h_x, w_x = x.shape
            n_aff = affinity_P.shape[1]
            aff_side = int(round(math.sqrt(n_aff)))
            if aff_side * aff_side == n_aff and aff_side > 0:
                x_g = (
                    F.interpolate(x.float(), size=(aff_side, aff_side), mode="bilinear", align_corners=False)
                    if (h_x, w_x) != (aff_side, aff_side)
                    else x.float()
                )
                x_flat = x_g.permute(0, 2, 3, 1).reshape(b, n_aff, c)
                x_prop = torch.bmm(affinity_P.float(), x_flat).reshape(b, aff_side, aff_side, c).permute(0, 3, 1, 2)
                if (h_x, w_x) != (aff_side, aff_side):
                    x_prop = F.interpolate(x_prop, size=(h_x, w_x), mode="bilinear", align_corners=False)
                x = x + self.aff_proj(self.aff_norm(x_prop.to(dtype=x.dtype)))
            else:
                x = x + self.aff_proj(self.aff_norm(x)) * 0.0
        elif self.use_affinity_prop and self.aff_proj is not None:
            x = x + self.aff_proj(self.aff_norm(x)) * 0.0

        for stage in self.upsample_stages:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = stage(x)
        out = self.out_conv(x)
        if out.shape[2] != orig_h or out.shape[3] != orig_w:
            out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return out[:, 0:1], out[:, 1:2]


class DualBackboneFusionModule(nn.Module):
    """
    Fuses features from a frozen primary FOCAL backbone and a secondary backbone.

    Supported ``fusion_mode`` options:
      - ``"concat"``: L2-normalized channel concatenation ``[F_norm(f1), F_norm(f2)]`` (FOCAL paper style).
      - ``"proj_concat"``: L2-normalized concatenation projected via ``1x1 Conv -> GroupNorm -> GELU -> 1x1 Conv``.
      - ``"zero_gated"``: Zero-initialized residual fusion ``f1 + zero_proj(f2)`` preserving ``f1`` at step 0.
    """

    def __init__(
        self,
        primary_dim: int,
        secondary_dim: int,
        fusion_mode: str = "concat",
        out_dim: Optional[int] = None,
        learnable_fusion_scale: bool = False,
    ) -> None:
        super().__init__()
        self.fusion_mode = str(fusion_mode).lower().strip()
        self.primary_dim = int(primary_dim)
        self.secondary_dim = int(secondary_dim)
        self.learnable_fusion_scale = bool(learnable_fusion_scale) and self.fusion_mode == "concat"
        self.fusion_scale_logits: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(2)) if self.learnable_fusion_scale else None
        )

        if self.fusion_mode == "concat":
            self.out_dim = self.primary_dim + self.secondary_dim
            self.proj = None
        elif self.fusion_mode == "proj_concat":
            resolved_out = int(out_dim) if out_dim is not None else self.primary_dim
            mid = max(32, resolved_out)
            groups = _select_num_groups(mid)
            self.proj = nn.Sequential(
                nn.Conv2d(self.primary_dim + self.secondary_dim, mid, kernel_size=1, bias=False),
                nn.GroupNorm(groups, mid),
                nn.GELU(),
                nn.Conv2d(mid, resolved_out, kernel_size=1, bias=True),
            )
            self.out_dim = resolved_out
        elif self.fusion_mode == "zero_gated":
            self.out_dim = self.primary_dim
            mid = max(32, min(self.primary_dim, self.secondary_dim))
            groups = _select_num_groups(mid)
            self.proj_in = nn.Sequential(
                nn.Conv2d(self.secondary_dim, mid, kernel_size=1, bias=False),
                nn.GroupNorm(groups, mid),
                nn.GELU(),
            )
            self.proj_out = nn.Conv2d(mid, self.primary_dim, kernel_size=1, bias=True)
            nn.init.zeros_(self.proj_out.weight)
            if self.proj_out.bias is not None:
                nn.init.zeros_(self.proj_out.bias)
            self.proj = None
        else:
            raise ValueError(
                f"Unsupported FOCAL fusion_mode '{fusion_mode}'. "
                "Supported: 'concat', 'proj_concat', 'zero_gated'."
            )

    def forward(self, f_primary: torch.Tensor, f_secondary: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "concat":
            p_norm = F.normalize(f_primary.float(), p=2, dim=1).to(dtype=f_primary.dtype)
            s_norm = F.normalize(f_secondary.float(), p=2, dim=1).to(dtype=f_secondary.dtype)
            if self.fusion_scale_logits is not None:
                weights = 2.0 * torch.softmax(self.fusion_scale_logits.float(), dim=0)
                p_norm = p_norm * torch.sqrt(weights[0]).to(dtype=p_norm.dtype)
                s_norm = s_norm * torch.sqrt(weights[1]).to(dtype=s_norm.dtype)
            return torch.cat([p_norm, s_norm], dim=1)
        if self.fusion_mode == "proj_concat":
            p_norm = F.normalize(f_primary.float(), p=2, dim=1).to(dtype=f_primary.dtype)
            s_norm = F.normalize(f_secondary.float(), p=2, dim=1).to(dtype=f_secondary.dtype)
            return self.proj(torch.cat([p_norm, s_norm], dim=1))
        # zero_gated: preserves f_primary identically at initialization step 0
        s_norm = F.normalize(f_secondary.float(), p=2, dim=1).to(dtype=f_secondary.dtype)
        return f_primary + self.proj_out(self.proj_in(s_norm))


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
        # Task 1: Domain-split 4-class / 4-cluster training with 2-class inference
        domain_split_mode: bool = False,
        domain_loss_weight: float = 1.0,
        domain_proto_weight: float = 0.5,
        domain_supcon_weight: float = 0.5,
        domain_clustering_mode: str = "prototype",
        domain_names: Optional[Sequence[str]] = None,
        # Task 2: Dual-backbone fusion (frozen primary + secondary backbone)
        secondary_backbone: Optional[str] = None,
        secondary_checkpoint_path: Optional[str] = None,
        secondary_hf_repo: Optional[str] = None,
        freeze_primary_backbone: Optional[bool] = None,
        secondary_trainable: bool = True,
        secondary_peft_enabled: bool = False,
        secondary_trainable_stages: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        secondary_out_dim: int = 128,
        secondary_lr_multiplier: float = 0.25,
        fusion_mode: str = "concat",
        fusion_out_dim: Optional[int] = None,
        fusion_spatial_align: str = "primary",
        learnable_fusion_scale: bool = False,
        # Clustering-anchored hybrid residual calibration
        cluster_residual_max_scale: float = 2.5,
        cluster_refiner_dim: int = 64,
        cluster_refiner_l2_weight: float = 0.0,
        detach_refiner_feats: bool = True,
        # FOCAL v2 upgrades (domain-invariant image-relative refiner, soft-EM, spatial adapter, ViT res & TTA)
        clustering_mode: str = "kmeans",
        vit_input_size: Optional[int] = None,
        tta_hflip: bool = False,
        feature_adapter_type: str = "mlp",
        cluster_refiner_type: str = "legacy",
        cluster_residual_scale_gain: float = 0.5,
        multi_branch_logit_fusion: bool = False,
        contrastive_boundary_ratio: float = 0.0,
        # FOCAL v3 upgrades (multi-tap pre-neck ViT, flip-equivariant k-means, polarity & gated refiner)
        vit_feature_taps: Optional[Sequence[int]] = None,
        vit_tap_fusion_mode: str = "dual_logit",
        vit_tap_cluster_size: Optional[int] = 48,
        learnable_tap_weights: bool = False,
        feature_centering_alpha: float = 0.0,
        cluster_polarity_mode: str = "minority",
        polarity_border_weight: float = 0.75,
        polarity_area_gate: float = 0.25,
        polarity_margin_scale: float = 5.0,
        polarity_loss_weight: float = 0.0,
        refiner_sep_gate_tau: float = 0.0,
        refiner_zero_mean_bias_weight: float = 0.0,
        refiner_ortho_weight: float = 0.0,
        # FOCAL v4 upgrades (HRNet AdaBN-Mix, graph diffusion, Otsu calibration, pre-neck projector & manifold refiner)
        hrnet_adabn_lambda: float = 0.0,
        hrnet_adabn_stages: Sequence[str] = ("stage4",),
        hrnet_adabn_shift_floor: float = 0.0,
        hrnet_adabn_shift_scale: float = 0.35,
        affinity_diff_beta: float = 0.0,
        affinity_diff_tau: float = 0.08,
        affinity_diff_steps: int = 2,
        feat_smooth_gamma: float = 0.0,
        otsu_cal_alpha: float = 0.0,
        use_preneck_projector: bool = False,
        preneck_proj_dim: int = 256,
        preneck_ortho_weight: float = 0.1,
        preneck_contrastive_weight: float = 0.5,
        refiner_use_affinity_prop: bool = False,
        refiner_use_srm_desc: bool = False,
        refiner_use_border_knn_desc: bool = False,
        adapter_ortho_weight: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.backbone_name = str(backbone).lower().strip()
        self.aux_classifier = bool(aux_classifier)
        self.num_classes = int(num_classes)
        self.hf_repo = hf_repo
        self.cluster_temperature = max(1e-4, float(cluster_temperature))
        self.force_vit_1024 = bool(force_vit_1024)
        self.vit_input_size: Optional[int] = (
            int(vit_input_size)
            if vit_input_size not in (None, "", "none", "null", 0)
            else (1024 if self.force_vit_1024 else None)
        )
        self.tta_hflip = bool(tta_hflip)
        self.clustering_mode = str(clustering_mode).lower().strip()
        if self.clustering_mode not in ("kmeans", "kmeans_robust", "soft_kmeans"):
            raise ValueError(
                f"Unsupported clustering_mode '{clustering_mode}'. "
                "Supported: 'kmeans', 'kmeans_robust', 'soft_kmeans'."
            )
        self.cluster_separability_threshold = max(0.0, float(cluster_separability_threshold))
        self.aux_pristine_gate = bool(aux_pristine_gate)
        self.contrastive_sample_size = max(16, int(contrastive_sample_size))
        self.contrastive_boundary_ratio = min(0.95, max(0.0, float(contrastive_boundary_ratio)))

        # Task 1: Domain-split 4-class / 4-cluster training configuration
        self.domain_split_mode = bool(domain_split_mode)
        self.domain_loss_weight = max(0.0, float(domain_loss_weight))
        self.domain_proto_weight = max(0.0, float(domain_proto_weight))
        self.domain_supcon_weight = max(0.0, float(domain_supcon_weight))
        self.domain_clustering_mode = str(domain_clustering_mode).lower().strip()
        if self.domain_clustering_mode not in ("prototype", "hybrid", "kmeans"):
            raise ValueError(
                f"Unsupported domain_clustering_mode '{domain_clustering_mode}'. "
                "Supported: 'prototype', 'hybrid', 'kmeans'."
            )
        self.domain_names: Tuple[str, ...] = (
            tuple(str(d).lower().strip() for d in domain_names)
            if domain_names is not None
            else ("casia", "imd")
        )

        # Task 2: Dual-backbone fusion configuration
        if secondary_backbone is None and "fusion_backbone" in kwargs:
            secondary_backbone = kwargs["fusion_backbone"]
        self.secondary_backbone_name: Optional[str] = (
            str(secondary_backbone).lower().strip()
            if secondary_backbone not in (None, "", "none", "null")
            else None
        )
        self.secondary_checkpoint_path = secondary_checkpoint_path
        self.secondary_hf_repo = secondary_hf_repo or hf_repo
        self.freeze_primary_backbone = (
            bool(freeze_primary_backbone)
            if freeze_primary_backbone is not None
            else (self.secondary_backbone_name is not None)
        )
        self.secondary_trainable = bool(secondary_trainable)
        self.secondary_peft_enabled = bool(secondary_peft_enabled)
        self.secondary_trainable_stages = secondary_trainable_stages
        self.secondary_out_dim = max(16, int(secondary_out_dim))
        self.secondary_lr_multiplier = float(secondary_lr_multiplier)
        self.fusion_mode = str(fusion_mode).lower().strip()
        self.fusion_out_dim = int(fusion_out_dim) if fusion_out_dim is not None else None
        self.fusion_spatial_align = str(fusion_spatial_align).lower().strip()
        if self.fusion_spatial_align not in ("primary", "max"):
            raise ValueError(
                f"Unsupported fusion_spatial_align '{fusion_spatial_align}'. Supported: 'primary', 'max'."
            )
        self.learnable_fusion_scale = bool(learnable_fusion_scale)
        self.cluster_residual_max_scale = max(0.0, float(cluster_residual_max_scale))
        self.cluster_residual_scale_gain = max(0.0, float(cluster_residual_scale_gain))
        self.cluster_refiner_dim = max(32, int(cluster_refiner_dim))
        self.cluster_refiner_l2_weight = max(0.0, float(cluster_refiner_l2_weight))
        self.detach_refiner_feats = bool(detach_refiner_feats)
        self.feature_adapter_type = str(feature_adapter_type).lower().strip()
        if self.feature_adapter_type not in ("mlp", "spatial", "branch_balanced"):
            raise ValueError(
                f"Unsupported feature_adapter_type '{feature_adapter_type}'. "
                "Supported: 'mlp', 'spatial', 'branch_balanced'."
            )
        self.cluster_refiner_type = str(cluster_refiner_type).lower().strip()
        if self.cluster_refiner_type not in ("legacy", "image_relative"):
            raise ValueError(
                f"Unsupported cluster_refiner_type '{cluster_refiner_type}'. Supported: 'legacy', 'image_relative'."
            )
        self.multi_branch_logit_fusion = bool(multi_branch_logit_fusion)

        # FOCAL v3 multi-tap, polarity disambiguation, and gated refiner configuration
        self.vit_feature_taps: List[int] = (
            [int(t) for t in vit_feature_taps]
            if vit_feature_taps not in (None, "", "none", "null")
            else []
        )
        self.vit_tap_fusion_mode = str(vit_tap_fusion_mode).lower().strip()
        if self.vit_tap_fusion_mode not in ("dual_logit", "logit"):
            raise ValueError(
                f"Unsupported vit_tap_fusion_mode '{vit_tap_fusion_mode}'. Supported: 'dual_logit', 'logit'."
            )
        self.vit_tap_cluster_size: Optional[int] = (
            int(vit_tap_cluster_size)
            if vit_tap_cluster_size not in (None, "", "none", "null", 0)
            else None
        )
        self.learnable_tap_weights = bool(learnable_tap_weights)
        self.feature_centering_alpha = max(0.0, min(1.0, float(feature_centering_alpha)))
        self.cluster_polarity_mode = str(cluster_polarity_mode).lower().strip()
        if self.cluster_polarity_mode not in ("minority", "border_aware", "learned"):
            raise ValueError(
                f"Unsupported cluster_polarity_mode '{cluster_polarity_mode}'. "
                "Supported: 'minority', 'border_aware', 'learned'."
            )
        self.polarity_border_weight = float(polarity_border_weight)
        self.polarity_area_gate = max(0.0, min(0.5, float(polarity_area_gate)))
        self.polarity_margin_scale = float(polarity_margin_scale)
        self.polarity_loss_weight = max(0.0, float(polarity_loss_weight))
        self.refiner_sep_gate_tau = max(0.0, float(refiner_sep_gate_tau))
        self.refiner_zero_mean_bias_weight = max(0.0, float(refiner_zero_mean_bias_weight))
        self.refiner_ortho_weight = max(0.0, float(refiner_ortho_weight))

        # FOCAL v4 configuration
        self.hrnet_adabn_lambda = max(0.0, min(1.0, float(hrnet_adabn_lambda)))
        self.hrnet_adabn_shift_floor = max(0.0, float(hrnet_adabn_shift_floor))
        self.hrnet_adabn_shift_scale = max(1e-4, float(hrnet_adabn_shift_scale))
        if isinstance(hrnet_adabn_stages, str):
            self.hrnet_adabn_stages: Tuple[str, ...] = tuple(
                s.strip() for s in hrnet_adabn_stages.split(",") if s.strip()
            ) or ("stage4",)
        else:
            self.hrnet_adabn_stages = tuple(str(s).strip() for s in hrnet_adabn_stages if str(s).strip()) or ("stage4",)
        self.affinity_diff_beta = max(0.0, min(1.0, float(affinity_diff_beta)))
        self.affinity_diff_tau = max(1e-3, float(affinity_diff_tau))
        self.affinity_diff_steps = max(1, int(affinity_diff_steps))
        self.feat_smooth_gamma = max(0.0, min(1.0, float(feat_smooth_gamma)))
        self.otsu_cal_alpha = max(0.0, min(2.0, float(otsu_cal_alpha)))
        self.use_preneck_projector = bool(use_preneck_projector)
        self.preneck_proj_dim = max(32, int(preneck_proj_dim))
        self.preneck_ortho_weight = max(0.0, float(preneck_ortho_weight))
        self.preneck_contrastive_weight = max(0.0, float(preneck_contrastive_weight))
        self.refiner_use_affinity_prop = bool(refiner_use_affinity_prop)
        self.refiner_use_srm_desc = bool(refiner_use_srm_desc)
        self.refiner_use_border_knn_desc = bool(refiner_use_border_knn_desc)
        self.adapter_ortho_weight = max(0.0, float(adapter_ortho_weight))

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
        elif raw_mode in ("hybrid", "clustering_hybrid", "cluster_hybrid", "residual_hybrid"):
            self.finetune_mode = "hybrid"
            self.use_clustering = True
        else:
            raise ValueError(
                f"Unsupported FOCAL finetune_mode '{finetune_mode}'. "
                "Supported: 'supervised', 'contrastive', 'hybrid', or None."
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

        # Build clean primary architecture BEFORE loading pretrained weights
        self.net, primary_in_dim, primary_upsample_stages, default_filename = self._build_backbone_net(
            self.backbone_name
        )
        self.primary_in_dim = primary_in_dim

        # Optional secondary backbone & fusion module (Task 2)
        self.secondary_net: Optional[nn.Module] = None
        self.secondary_kind: Optional[str] = None  # "hrnet", "vit", or "timm"
        self.fusion_module: Optional[DualBackboneFusionModule] = None
        secondary_default_ckpt: Optional[str] = None

        if self.secondary_backbone_name is not None:
            (
                self.secondary_net,
                secondary_in_dim,
                secondary_upsample_stages,
                secondary_default_ckpt,
                self.secondary_kind,
            ) = self._build_secondary_backbone_net(self.secondary_backbone_name)
            self.fusion_module = DualBackboneFusionModule(
                primary_dim=primary_in_dim,
                secondary_dim=secondary_in_dim,
                fusion_mode=self.fusion_mode,
                out_dim=self.fusion_out_dim,
                learnable_fusion_scale=self.learnable_fusion_scale,
            )
            in_dim = self.fusion_module.out_dim
            if self.fusion_spatial_align == "max":
                num_upsample_stages = min(primary_upsample_stages, secondary_upsample_stages)
            else:
                num_upsample_stages = primary_upsample_stages
        else:
            in_dim = primary_in_dim
            num_upsample_stages = primary_upsample_stages

        self.in_dim = in_dim

        # 2-class Convolutional classification head (matches upstream checkpoint shape `primary_in_dim -> 2`)
        self.fc = nn.Conv2d(primary_in_dim, 2, kernel_size=3, padding=1)
        use_legacy_supervised_conv = (
            self.head_type == "legacy" and self.finetune_mode != "contrastive"
        )
        self.fused_fc: Optional[nn.Conv2d] = (
            nn.Conv2d(in_dim, 2, kernel_size=3, padding=1)
            if (self.secondary_net is not None and use_legacy_supervised_conv and in_dim != primary_in_dim)
            else None
        )
        self.domain_fc: Optional[nn.Conv2d] = (
            nn.Conv2d(in_dim, 4, kernel_size=3, padding=1)
            if (self.domain_split_mode and use_legacy_supervised_conv)
            else None
        )

        # Optional upgraded supervised decoder (Mode 1A; 4 channels when domain_split_mode=True)
        decoder_out_channels = 4 if self.domain_split_mode else 1
        self.decoder: Optional[FOCALSupervisedHead] = (
            FOCALSupervisedHead(
                in_dim=in_dim,
                num_upsample_stages=num_upsample_stages,
                out_channels=decoder_out_channels,
            )
            if self.head_type == "decoder"
            else None
        )

        # Optional zero-initialized residual forensic adapter (Strategy B / SpatialCrossScaleAdapter / BranchBalanced)
        if self.use_feature_adapter:
            if self.feature_adapter_type == "branch_balanced" and self.secondary_net is not None and self.fusion_mode == "concat":
                self.feature_adapter: Optional[nn.Module] = BranchBalancedSpatialAdapter(
                    primary_dim=primary_in_dim,
                    secondary_dim=secondary_in_dim,
                )
            elif self.feature_adapter_type in ("spatial", "branch_balanced"):
                self.feature_adapter = SpatialCrossScaleAdapter(channels=in_dim)
            else:
                self.feature_adapter = FeatureAdapter(channels=in_dim)
        else:
            self.feature_adapter = None

        # Optional clustering-anchored residual calibration refiner (hybrid mode)
        if self.finetune_mode == "hybrid":
            if self.cluster_refiner_type == "image_relative":
                self.cluster_refiner: Optional[nn.Module] = ImageRelativeClusterRefiner(
                    num_upsample_stages=num_upsample_stages,
                    mid_dim=self.cluster_refiner_dim,
                    multi_branch=(self.secondary_net is not None),
                    use_affinity_prop=self.refiner_use_affinity_prop,
                    use_srm_desc=self.refiner_use_srm_desc,
                    use_border_knn_desc=self.refiner_use_border_knn_desc,
                )
            else:
                self.cluster_refiner = ClusterResidualRefiner(
                    in_dim=in_dim,
                    num_upsample_stages=num_upsample_stages,
                    mid_dim=self.cluster_refiner_dim,
                )
        else:
            self.cluster_refiner = None

        self.branch_logit_weights: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(2))
            if (self.finetune_mode == "hybrid" and self.multi_branch_logit_fusion and self.secondary_net is not None)
            else None
        )

        # Optional Orthogonal Zero-Bias Pre-Neck ViT Subspace Projector (FOCAL v4)
        vit_embed_dim = 1024
        if self._canonical_backbone_kind(self.backbone_name) == "vit":
            vit_embed_dim = int(self.net.image_encoder.patch_embed.proj.out_channels)
        if self.use_preneck_projector and self.vit_feature_taps and self._canonical_backbone_kind(self.backbone_name) == "vit":
            self.preneck_projector: Optional[PreNeckForensicProjector] = PreNeckForensicProjector(
                in_dim=vit_embed_dim * len(self.vit_feature_taps),
                proj_dim=self.preneck_proj_dim,
            )
            if self.finetune_mode != "hybrid":
                for p in self.preneck_projector.parameters():
                    p.requires_grad_(False)
        else:
            self.preneck_projector = None

        # Optional learnable multi-tap logit blend & learned cluster polarity head (FOCAL v3/v4)
        if self.learnable_tap_weights and self.vit_feature_taps and self._canonical_backbone_kind(self.backbone_name) == "vit":
            n_taps = 2 if self.vit_tap_fusion_mode == "dual_logit" else (1 + len(self.vit_feature_taps))
            if self.preneck_projector is not None:
                n_taps += 1
            init_tw = torch.zeros(n_taps)
            if self.preneck_projector is not None:
                # Start preneck projector softly gated so step-0 preserves zero-shot v4 floor
                init_tw[-1] = -2.0
            self.tap_logit_weights: Optional[nn.Parameter] = nn.Parameter(init_tw)
        else:
            self.tap_logit_weights = None

        self.polarity_head: Optional[ClusterPolarityHead] = (
            ClusterPolarityHead(hidden_dim=32) if self.cluster_polarity_mode == "learned" else None
        )
        if self.polarity_head is not None and self.finetune_mode != "hybrid":
            for p in self.polarity_head.parameters():
                p.requires_grad_(False)

        # Freeze unused legacy conv heads so DDP never complains about unused parameters
        if (
            self.head_type == "decoder"
            or self.finetune_mode in ("contrastive", "hybrid")
            or self.fused_fc is not None
            or self.domain_fc is not None
        ):
            for p in self.fc.parameters():
                p.requires_grad_(False)
        if self.head_type == "decoder" or self.finetune_mode in ("contrastive", "hybrid"):
            if self.fused_fc is not None:
                for p in self.fused_fc.parameters():
                    p.requires_grad_(False)
            if self.domain_fc is not None:
                for p in self.domain_fc.parameters():
                    p.requires_grad_(False)

        # Optional auxiliary classifier
        if self.aux_classifier:
            eff_aux_classes = 4 if self.domain_split_mode else num_classes
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_dim, 128),
                nn.GELU(),
                nn.Linear(128, eff_aux_classes),
            )
        else:
            self.class_head = None

        # Domain-split 4-cluster prototypes (0: casia_p, 1: casia_t, 2: imd_p, 3: imd_t)
        if self.domain_split_mode and self.finetune_mode == "contrastive":
            proto_init = F.normalize(torch.randn(4, in_dim), p=2, dim=-1)
            self.domain_prototypes: Optional[nn.Parameter] = nn.Parameter(proto_init)
            self.register_buffer("domain_prototypes_initialized", torch.zeros(4, dtype=torch.bool))
        else:
            self.domain_prototypes = None
            self.register_buffer("domain_prototypes_initialized", torch.zeros(0, dtype=torch.bool), persistent=False)

        # Strategy B & Dual-Backbone state tracking
        hrnet_stages_explicit = self.hrnet_trainable_stages is not None
        is_hrnet = self.backbone_name in ("hrnet", "hrnet32", "hrnet-w32")
        self._surgery_active = bool(
            self.peft_enabled
            or (is_hrnet and hrnet_stages_explicit)
            or self.use_feature_adapter
            or (freeze_backbone_bn is True)
            or self.freeze_primary_backbone
            or (self.secondary_net is not None)
        )
        if freeze_backbone_bn is None:
            self.freeze_backbone_bn = bool(
                (is_hrnet and self._surgery_active)
                or self.freeze_primary_backbone
                or (self.secondary_kind == "hrnet" and self.secondary_trainable_stages is not None)
            )
        else:
            self.freeze_backbone_bn = bool(freeze_backbone_bn)

        self._surgical_backbone_param_names: Set[str] = set()
        self._surgical_secondary_param_names: Set[str] = set()
        self._frozen_bn_modules: List[nn.Module] = []
        self._adapted_block_indices: List[int] = []
        self._secondary_adapted_block_indices: List[int] = []
        self._backbone_temporarily_frozen: bool = False
        self._surgery_applied: bool = False
        object.__setattr__(self, "_anchor_neck", None)
        object.__setattr__(self, "_anchor_hrnet_stage4", None)
        object.__setattr__(self, "_anchor_secondary_neck", None)
        object.__setattr__(self, "_anchor_secondary_hrnet_stage4", None)
        self._neck_init_params: Dict[str, torch.Tensor] = {}

        # Step 2 (strict init order): Load & verify pretrained base weights FIRST
        secondary_loaded_from_primary_ckpt = self._load_pretrained(
            source=checkpoint_path or default_filename,
            default_filename=default_filename,
            cache_dir=cache_dir,
        )
        if (
            self.secondary_net is not None
            and not secondary_loaded_from_primary_ckpt
            and self.secondary_kind in ("hrnet", "vit")
            and secondary_default_ckpt is not None
        ):
            self._load_secondary_focal_weights(
                source=self.secondary_checkpoint_path or secondary_default_ckpt,
                default_filename=secondary_default_ckpt,
                cache_dir=cache_dir,
            )

        # Step 3 (strict init order): Apply Strategy B / dual-backbone surgery AFTER base weights are loaded
        if self._surgery_active and not self._surgery_applied:
            self._apply_strategy_b_surgery()
        elif not self._surgery_active:
            self._surgical_backbone_param_names = {n for n, _ in self.net.named_parameters()}

        if self._surgery_active:
            self.log_surgery_summary()

    @staticmethod
    def _canonical_backbone_kind(name: str) -> Optional[str]:
        norm = str(name).lower().strip()
        if norm in ("hrnet", "hrnet32", "hrnet-w32"):
            return "hrnet"
        if norm in ("vit", "vit-l", "sam_vit"):
            return "vit"
        return None

    @classmethod
    def _build_backbone_net(cls, backbone_name: str) -> Tuple[nn.Module, int, int, str]:
        kind = cls._canonical_backbone_kind(backbone_name)
        if kind == "hrnet":
            return HRNet(), 32, 2, DEFAULT_FOCAL_HRNET_CKPT
        if kind == "vit":
            net = FOCALViTNet(
                img_size=1024,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                out_chans=256,
            )
            return net, 256, 4, DEFAULT_FOCAL_VIT_CKPT
        raise ValueError(f"Unsupported FOCAL backbone '{backbone_name}'. Supported: 'hrnet', 'vit'.")

    def _build_secondary_backbone_net(
        self, backbone_name: str
    ) -> Tuple[nn.Module, int, int, Optional[str], str]:
        kind = self._canonical_backbone_kind(backbone_name)
        if kind is not None:
            net, dim, upsample, ckpt = self._build_backbone_net(kind)
            return net, dim, upsample, ckpt, kind
        # Pretrained timm backbone
        sec_net = SecondaryTimmBackbone(model_name=backbone_name, out_dim=self.secondary_out_dim)
        return sec_net, self.secondary_out_dim, 3, None, "timm"

    @property
    def requires_gt_masks(self) -> bool:
        """True when Mode 1B contrastive, hybrid, or domain-split 4-class training needs ground-truth masks."""
        return self.finetune_mode in ("contrastive", "hybrid") or self.domain_split_mode

    @property
    def requires_domain_info(self) -> bool:
        """True when domain-split training uses dataset source metadata."""
        return self.domain_split_mode

    def _resolve_vit_block_indices(self, encoder: Optional[ImageEncoderViT] = None) -> List[int]:
        """Resolve which SAMBlock indices receive LoRA/DoRA surgery."""
        enc = encoder if encoder is not None else self.net.image_encoder
        depth = len(enc.blocks)
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

    @staticmethod
    def _normalize_hrnet_stages_from(raw: Any, default_peft: bool) -> List[str]:
        """Normalize hrnet_trainable_stages config into canonical stage tokens."""
        if raw is None:
            return ["stage4"] if default_peft else ["all"]
        if isinstance(raw, str):
            tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
        else:
            tokens = [str(t).strip().lower() for t in raw if str(t).strip()]
        return tokens or ["none"]

    def _normalize_hrnet_stages(self) -> List[str]:
        return self._normalize_hrnet_stages_from(self.hrnet_trainable_stages, self.peft_enabled)

    def _apply_vit_peft_to_net(self, vit_net: FOCALViTNet) -> List[int]:
        """Inject LoRA/DoRA adapters onto a `FOCALViTNet` and freeze all unadapted base weights."""
        for p in vit_net.parameters():
            p.requires_grad_(False)

        encoder = vit_net.image_encoder
        block_indices = self._resolve_vit_block_indices(encoder)
        valid_targets = {"qkv", "proj", "lin1", "lin2"}
        for t in self.lora_target_modules:
            if t not in valid_targets:
                raise ValueError(
                    f"Unsupported lora_target_modules entry '{t}'. Supported: {sorted(valid_targets)}."
                )

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

        return block_indices

    def inject_peft_lora(self) -> List[int]:
        """
        Apply Strategy B LoRA/DoRA surgery onto `FOCALViTNet` blocks and freeze all
        unadapted base parameters (`patch_embed`, `pos_embed`, `rel_pos_h/w`, early blocks).
        """
        if self._canonical_backbone_kind(self.backbone_name) != "vit":
            raise RuntimeError("inject_peft_lora() is only applicable to FOCAL ViT backbones.")
        block_indices = self._apply_vit_peft_to_net(self.net)
        self._adapted_block_indices = block_indices
        return block_indices

    def _apply_hrnet_stage_surgery(
        self,
        hrnet_net: HRNet,
        stages: List[str],
        freeze_bn: bool,
    ) -> List[nn.Module]:
        """Apply stage-wise freezing and BatchNorm2d running-stat protection on an HRNet module."""
        stage_modules: Dict[str, List[nn.Module]] = {
            "stem": [hrnet_net.conv1, hrnet_net.bn1, hrnet_net.conv2, hrnet_net.bn2],
            "layer1": [hrnet_net.layer1],
            "transition1": [hrnet_net.transition1],
            "stage2": [hrnet_net.stage2],
            "transition2": [hrnet_net.transition2],
            "stage3": [hrnet_net.stage3],
            "transition3": [hrnet_net.transition3],
            "stage4": [hrnet_net.stage4],
            "stage4_last": [hrnet_net.stage4[-1]],
        }

        if "all" not in stages:
            for p in hrnet_net.parameters():
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

        frozen_bns: List[nn.Module] = []
        if freeze_bn:
            for mod_name, mod in hrnet_net.named_modules():
                if isinstance(mod, nn.BatchNorm2d):
                    in_stage4 = mod_name.startswith("stage4")
                    has_trainable_param = any(p.requires_grad for p in mod.parameters())
                    if (not has_trainable_param) or (in_stage4 and self.freeze_stage4_bn):
                        for p in mod.parameters():
                            p.requires_grad_(False)
                        mod.eval()
                        frozen_bns.append(mod)
        return frozen_bns

    def _apply_hrnet_surgery(self) -> None:
        """Apply stage-wise freezing and BatchNorm2d running-stat protection on primary HRNet."""
        stages = self._normalize_hrnet_stages()
        self._frozen_bn_modules = self._apply_hrnet_stage_surgery(
            self.net, stages=stages, freeze_bn=self.freeze_backbone_bn
        )

    def _init_prior_anchors(self) -> None:
        """Snapshot minimal frozen anchor submodules/weights without duplicating the full backbone."""
        object.__setattr__(self, "_anchor_neck", None)
        object.__setattr__(self, "_anchor_hrnet_stage4", None)
        object.__setattr__(self, "_anchor_secondary_neck", None)
        object.__setattr__(self, "_anchor_secondary_hrnet_stage4", None)
        self._neck_init_params = {}

        if not self.freeze_primary_backbone:
            if self._canonical_backbone_kind(self.backbone_name) == "vit":
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
            elif self._canonical_backbone_kind(self.backbone_name) == "hrnet":
                if self.prior_anchor_weight > 0.0 and self._surgical_backbone_param_names:
                    anchor_s4 = copy.deepcopy(self.net.stage4)
                    anchor_s4.eval()
                    for p in anchor_s4.parameters():
                        p.requires_grad_(False)
                    object.__setattr__(self, "_anchor_hrnet_stage4", anchor_s4)

        if self.secondary_net is not None and self.prior_anchor_weight > 0.0 and self._surgical_secondary_param_names:
            if self.secondary_kind == "vit" and self.train_neck:
                anchor_sec_neck = copy.deepcopy(self.secondary_net.image_encoder.neck)
                anchor_sec_neck.eval()
                for p in anchor_sec_neck.parameters():
                    p.requires_grad_(False)
                object.__setattr__(self, "_anchor_secondary_neck", anchor_sec_neck)
            elif self.secondary_kind == "hrnet":
                anchor_sec_s4 = copy.deepcopy(self.secondary_net.stage4)
                anchor_sec_s4.eval()
                for p in anchor_sec_s4.parameters():
                    p.requires_grad_(False)
                object.__setattr__(self, "_anchor_secondary_hrnet_stage4", anchor_sec_s4)

    def _apply(self, fn: Any) -> "FOCAL":
        """Ensure non-registered anchor helper modules follow `.to(device)` / `.to(dtype)` calls."""
        super()._apply(fn)
        for attr in ("_anchor_neck", "_anchor_hrnet_stage4", "_anchor_secondary_neck", "_anchor_secondary_hrnet_stage4"):
            mod = getattr(self, attr, None)
            if mod is not None:
                mod._apply(fn)
        return self

    def _apply_strategy_b_surgery(self) -> None:
        """Apply configured Strategy B and dual-backbone surgery and record trainable masks."""
        if self._surgery_applied:
            return

        primary_kind = self._canonical_backbone_kind(self.backbone_name)
        if self.freeze_primary_backbone:
            for p in self.net.parameters():
                p.requires_grad_(False)
            self._frozen_bn_modules = [
                m for m in self.net.modules() if isinstance(m, nn.BatchNorm2d)
            ]
            for bn in self._frozen_bn_modules:
                bn.eval()
        elif primary_kind == "vit":
            if self.peft_enabled:
                self.inject_peft_lora()
            elif not self.train_neck:
                for p in self.net.image_encoder.neck.parameters():
                    p.requires_grad_(False)
        elif primary_kind == "hrnet":
            self._apply_hrnet_surgery()

        self._surgical_backbone_param_names = {
            name for name, p in self.net.named_parameters() if p.requires_grad
        }

        # Apply secondary backbone freezing / PEFT surgery if present
        self._surgical_secondary_param_names = set()
        if self.secondary_net is not None:
            if not self.secondary_trainable:
                for p in self.secondary_net.parameters():
                    p.requires_grad_(False)
                for m in self.secondary_net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        self._frozen_bn_modules.append(m)
            elif self.secondary_kind == "vit":
                if self.secondary_peft_enabled or self.peft_enabled:
                    self._secondary_adapted_block_indices = self._apply_vit_peft_to_net(self.secondary_net)
                elif not self.train_neck:
                    for p in self.secondary_net.image_encoder.neck.parameters():
                        p.requires_grad_(False)
            elif self.secondary_kind == "hrnet":
                if self.secondary_trainable_stages is not None or self.freeze_backbone_bn:
                    sec_stages = self._normalize_hrnet_stages_from(
                        self.secondary_trainable_stages, default_peft=False
                    )
                    sec_bns = self._apply_hrnet_stage_surgery(
                        self.secondary_net, stages=sec_stages, freeze_bn=self.freeze_backbone_bn
                    )
                    self._frozen_bn_modules.extend(sec_bns)
            self._surgical_secondary_param_names = {
                name for name, p in self.secondary_net.named_parameters() if p.requires_grad
            }

        self._init_prior_anchors()
        self._surgery_applied = True
        self._enforce_frozen_backbone_eval()

    def _enforce_frozen_backbone_eval(self) -> None:
        """Keep permanently frozen BatchNorm2d layers (or fully frozen backbone) in eval mode."""
        if self.freeze_primary_backbone or self._backbone_temporarily_frozen or not self._surgical_backbone_param_names:
            self.net.eval()
        if self.secondary_net is not None:
            if self._backbone_temporarily_frozen or not self._surgical_secondary_param_names:
                self.secondary_net.eval()
        if self._surgery_active:
            for bn in self._frozen_bn_modules:
                bn.eval()
        for attr in ("_anchor_neck", "_anchor_hrnet_stage4", "_anchor_secondary_neck", "_anchor_secondary_hrnet_stage4"):
            mod = getattr(self, attr, None)
            if mod is not None:
                mod.eval()

    def train(self, mode: bool = True) -> "FOCAL":
        """Override train() so frozen backbone BatchNorm2d stats never leak during Trainer.train_epoch()."""
        super().train(mode)
        if mode:
            self._enforce_frozen_backbone_eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze backbones respecting Strategy B & frozen-primary surgical masks."""
        self._backbone_temporarily_frozen = not bool(trainable)
        if not trainable:
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.net.eval()
            if self.secondary_net is not None:
                for p in self.secondary_net.parameters():
                    p.requires_grad_(False)
                self.secondary_net.eval()
            return

        if self._surgery_active:
            for name, p in self.net.named_parameters():
                p.requires_grad_(name in self._surgical_backbone_param_names)
            self.net.train(self.training and not self.freeze_primary_backbone)
            if self.secondary_net is not None:
                for name, p in self.secondary_net.named_parameters():
                    p.requires_grad_(name in self._surgical_secondary_param_names)
                self.secondary_net.train(self.training)
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
        Build optimizer parameter groups separating head/fusion, LoRA/DoRA adapters,
        trainable neck / partial HRNet stages, and optional secondary backbone.
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

        backbone_param_ids = {id(p) for p in self.net.parameters()}
        secondary_param_ids = (
            {id(p) for p in self.secondary_net.parameters()} if self.secondary_net is not None else set()
        )
        anchor_param_ids: Set[int] = set()
        for attr in ("_anchor_neck", "_anchor_hrnet_stage4", "_anchor_secondary_neck", "_anchor_secondary_hrnet_stage4"):
            mod = getattr(self, attr, None)
            if mod is not None:
                anchor_param_ids.update(id(p) for p in mod.parameters())

        head_params = [
            p
            for p in self.parameters()
            if id(p) not in backbone_param_ids
            and id(p) not in secondary_param_ids
            and id(p) not in anchor_param_ids
            and p.requires_grad
        ]

        lora_params: List[nn.Parameter] = []
        neck_params: List[nn.Parameter] = []
        partial_bb_params: List[nn.Parameter] = []
        secondary_params: List[nn.Parameter] = []

        for name, p in self.net.named_parameters():
            if name not in self._surgical_backbone_param_names:
                continue
            if name.endswith((".lora_A", ".lora_B", ".dora_m")):
                lora_params.append(p)
            elif "image_encoder.neck." in name or name.startswith("neck."):
                neck_params.append(p)
            else:
                partial_bb_params.append(p)

        if self.secondary_net is not None:
            for name, p in self.secondary_net.named_parameters():
                if name not in self._surgical_secondary_param_names:
                    continue
                if name.endswith((".lora_A", ".lora_B", ".dora_m")):
                    lora_params.append(p)
                elif "image_encoder.neck." in name or name.startswith("neck."):
                    neck_params.append(p)
                else:
                    secondary_params.append(p)

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
        if secondary_params:
            resolved_sec_lr = float(lr) * self.secondary_lr_multiplier
            groups.append({"params": secondary_params, "lr": resolved_sec_lr, "name": "secondary_backbone"})

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
        if self.secondary_net is not None:
            _replace_recursive(self.secondary_net)
            self._surgical_secondary_param_names = {
                name for name, p in self.secondary_net.named_parameters() if p.requires_grad
            }
        self.peft_enabled = False
        self.secondary_peft_enabled = False
        self._surgical_backbone_param_names = {
            name for name, p in self.net.named_parameters() if p.requires_grad
        }
        if merged_count > 0:
            logger.info(f"🔀 Merged {merged_count} SurgicalLoRALinear adapter(s) into standard nn.Linear layers.")
        return self

    def get_surgery_summary(self) -> Dict[str, Any]:
        """Return parameter footprint and active Strategy B / fusion / domain-split metadata."""
        anchor_ids: Set[int] = set()
        for attr in ("_anchor_neck", "_anchor_hrnet_stage4", "_anchor_secondary_neck", "_anchor_secondary_hrnet_stage4"):
            mod = getattr(self, attr, None)
            if mod is not None:
                anchor_ids.update(id(p) for p in mod.parameters())

        total_params = sum(p.numel() for p in self.parameters() if id(p) not in anchor_ids)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in anchor_ids)
        backbone_total = sum(p.numel() for p in self.net.parameters())
        backbone_trainable = sum(
            p.numel() for n, p in self.net.named_parameters() if n in self._surgical_backbone_param_names
        )
        secondary_total = sum(p.numel() for p in self.secondary_net.parameters()) if self.secondary_net is not None else 0
        secondary_trainable = (
            sum(
                p.numel()
                for n, p in self.secondary_net.named_parameters()
                if n in self._surgical_secondary_param_names
            )
            if self.secondary_net is not None
            else 0
        )
        return {
            "backbone": self.backbone_name,
            "secondary_backbone": self.secondary_backbone_name,
            "freeze_primary_backbone": self.freeze_primary_backbone,
            "fusion_mode": self.fusion_mode if self.secondary_net is not None else None,
            "domain_split_mode": self.domain_split_mode,
            "finetune_mode": self.finetune_mode,
            "peft_enabled": self.peft_enabled,
            "use_dora": self.use_dora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "adapted_block_indices": list(self._adapted_block_indices),
            "secondary_adapted_block_indices": list(self._secondary_adapted_block_indices),
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
            "secondary_total_params": secondary_total,
            "secondary_trainable_params": secondary_trainable,
        }

    def log_surgery_summary(self, custom_logger: Optional[logging.Logger] = None) -> None:
        """Log concise Strategy B / dual-backbone startup summary."""
        lg = custom_logger or logger
        s = self.get_surgery_summary()
        lg.info(
            f"🧬 [FOCAL Surgery] backbone={s['backbone']} secondary={s['secondary_backbone']} "
            f"freeze_primary={s['freeze_primary_backbone']} fusion={s['fusion_mode']} "
            f"domain_split={s['domain_split_mode']} mode={s['finetune_mode']} | "
            f"trainable={s['trainable_params']:,}/{s['total_params']:,} ({s['trainable_pct']:.2f}%) | "
            f"backbone_trainable={s['backbone_trainable_params']:,}/{s['backbone_total_params']:,} ({s['backbone_trainable_pct']:.2f}%) | "
            f"peft={s['peft_enabled']} dora={s['use_dora']} blocks={s['adapted_block_indices']} "
            f"train_neck={s['train_neck']} frozen_bn={s['frozen_bn_count']} "
            f"adapter={s['use_feature_adapter']} prior_anchor={s['prior_anchor_weight']}"
        )

    @staticmethod
    def _resolve_checkpoint_path(
        source: str,
        default_filename: str,
        backbone_kind: Optional[str],
        hf_repo: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Resolve local or Hugging Face checkpoint path under strict zero-fallback policy."""
        resolved_path = None
        if os.path.isfile(source):
            resolved_path = os.path.abspath(source)

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

        if resolved_path is None:
            filename = os.path.basename(source) if source.endswith((".pth", ".pt")) else default_filename
            hf_candidates = [filename]
            if backbone_kind == "hrnet":
                for c in [
                    "checkpoints/checkpoint_best.pt",
                    "checkpoint_best.pt",
                    "FOCAL_HRNet_weights.pth",
                    "checkpoints/FOCAL_HRNet_weights.pth",
                ]:
                    if c not in hf_candidates:
                        hf_candidates.append(c)
            else:
                for c in [
                    "checkpoints/focal_vit.pt",
                    "focal_vit.pt",
                    "checkpoints/checkpoint_best.pt",
                    "FOCAL_ViT_weights.pth",
                    "checkpoints/FOCAL_ViT_weights.pth",
                ]:
                    if c not in hf_candidates:
                        hf_candidates.append(c)

            from huggingface_hub import hf_hub_download

            last_err = None
            for c_fn in hf_candidates:
                try:
                    resolved_path = hf_hub_download(
                        repo_id=hf_repo,
                        filename=c_fn,
                        cache_dir=cache_dir,
                    )
                    if resolved_path and os.path.isfile(resolved_path):
                        logger.info(f"Downloaded FOCAL pretrained weights from HF '{hf_repo}/{c_fn}': {resolved_path}")
                        break
                except Exception as hf_err:
                    last_err = hf_err
                    continue

            if resolved_path is None and last_err is not None:
                raise RuntimeError(
                    f"Failed to download/load FOCAL weights from Hugging Face repo '{hf_repo}' (tried {hf_candidates}): {last_err}. "
                    f"Strict zero-fallback policy: aborting initialization."
                ) from last_err

        if resolved_path is None or not os.path.isfile(resolved_path):
            raise RuntimeError(
                f"Failed to resolve FOCAL weights at '{source}'. "
                f"Strict zero-fallback policy: local initialization is strictly disabled."
            )
        return resolved_path

    @staticmethod
    def _extract_clean_state_dict(resolved_path: str) -> Dict[str, Any]:
        """Load checkpoint dict and strip DataParallel ``module.`` prefixes."""
        try:
            ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except Exception as load_err:
            raise RuntimeError(f"Corrupt or unreadable checkpoint at '{resolved_path}': {load_err}") from load_err

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

        return {(k[7:] if k.startswith("module.") else k): v for k, v in raw_sd.items()}

    def _load_pretrained(
        self, source: str, default_filename: str, cache_dir: Optional[str] = None
    ) -> bool:
        """
        Download or resolve genuine pretrained weights and load state_dict strictly.
        Returns True if the checkpoint already supplied ``secondary_net.*`` weights.
        """
        resolved_path = self._resolve_checkpoint_path(
            source=source,
            default_filename=default_filename,
            backbone_kind=self._canonical_backbone_kind(self.backbone_name),
            hf_repo=self.hf_repo,
            cache_dir=cache_dir,
        )
        clean_sd = self._extract_clean_state_dict(resolved_path)

        is_peft_ckpt = any(
            ".lora_A" in k or ".lora_B" in k or ".dora_m" in k or ".base_linear." in k
            for k in clean_sd.keys()
        )
        if is_peft_ckpt and self._surgery_active and not self._surgery_applied:
            # Strategy B checkpoint reload: apply matching PEFT surgery before loading state_dict
            self._apply_strategy_b_surgery()

        has_secondary_weights = self.secondary_net is not None and any(
            k.startswith("secondary_net.") for k in clean_sd.keys()
        )

        exempt_prefixes = [
            "_anchor_neck.",
            "_anchor_hrnet_stage4.",
            "_anchor_secondary_neck.",
            "_anchor_secondary_hrnet_stage4.",
        ]
        if self.class_head is not None:
            exempt_prefixes.append("class_head.")
        if self.decoder is not None:
            exempt_prefixes.append("decoder.")
        if self.feature_adapter is not None:
            exempt_prefixes.append("feature_adapter.")
        if self.cluster_refiner is not None:
            exempt_prefixes.append("cluster_refiner.")
        if self.branch_logit_weights is not None:
            exempt_prefixes.append("branch_logit_weights")
        if self.tap_logit_weights is not None:
            exempt_prefixes.append("tap_logit_weights")
        if self.preneck_projector is not None:
            exempt_prefixes.append("preneck_projector.")
        if self.polarity_head is not None:
            exempt_prefixes.append("polarity_head.")
        if self.domain_split_mode:
            exempt_prefixes.extend(["domain_prototypes", "domain_fc."])
        if self.secondary_net is not None:
            if not has_secondary_weights:
                exempt_prefixes.append("secondary_net.")
            exempt_prefixes.extend(["fusion_module.", "fused_fc."])

        expected_keys = {
            k for k in self.state_dict().keys() if not k.startswith(tuple(exempt_prefixes))
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
        return has_secondary_weights

    def _load_secondary_focal_weights(
        self, source: str, default_filename: str, cache_dir: Optional[str] = None
    ) -> None:
        """Load genuine pretrained FOCAL weights for a secondary HRNet or ViT backbone."""
        assert self.secondary_net is not None
        resolved_path = self._resolve_checkpoint_path(
            source=source,
            default_filename=default_filename,
            backbone_kind=self.secondary_kind,
            hf_repo=self.secondary_hf_repo,
            cache_dir=cache_dir,
        )
        clean_sd = self._extract_clean_state_dict(resolved_path)

        sec_sd: Dict[str, Any] = {}
        expected_sec_keys = set(self.secondary_net.state_dict().keys())
        for k, v in clean_sd.items():
            if k in expected_sec_keys:
                sec_sd[k] = v
            elif k.startswith("net.") and k[4:] in expected_sec_keys:
                sec_sd[k[4:]] = v
            elif k.startswith("secondary_net.") and k[len("secondary_net."):] in expected_sec_keys:
                sec_sd[k[len("secondary_net."):]] = v

        missing_keys = [k for k in expected_sec_keys if k not in sec_sd]
        if missing_keys:
            raise RuntimeError(
                f"FOCAL secondary backbone ({self.secondary_backbone_name}) missing pretrained weights: "
                f"{missing_keys[:10]}. Zero-fallback policy enforced."
            )
        self.secondary_net.load_state_dict(sec_sd, strict=True)
        logger.info(
            f"✅ Loaded genuine FOCAL secondary '{self.secondary_backbone_name}' weights from: {resolved_path}"
        )

    def _ensure_raw_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is in raw [0, 1] range per sample as expected by FOCAL."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        if x.ndim >= 4:
            x_min = x.amin(dim=(1, 2, 3), keepdim=True)
            x_max = x.amax(dim=(1, 2, 3), keepdim=True)
            needs_denorm = (x_min < -0.05) | (x_max > 1.05)
            denorm = torch.clamp(x * std + mean, 0.0, 1.0)
            return torch.where(needs_denorm, denorm, x)
        if x.min() < -0.05 or x.max() > 1.05:
            return torch.clamp(x.unsqueeze(0) * std + mean, 0.0, 1.0).squeeze(0)
        return x

    def _prepare_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_raw_rgb(x)
        if self._canonical_backbone_kind(self.backbone_name) == "vit" and self.vit_input_size is not None:
            orig_h, orig_w = x.shape[2], x.shape[3]
            if orig_h != self.vit_input_size or orig_w != self.vit_input_size:
                return F.interpolate(
                    x,
                    size=(self.vit_input_size, self.vit_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
        return x

    def _forward_hrnet_maybe_adabn(self, hrnet_net: nn.Module, x_in: torch.Tensor) -> torch.Tensor:
        """
        Run HRNet with optional per-image dead-channel-guarded AdaBN-Mix (`hrnet_adabn_lambda > 0`).
        When `hrnet_adabn_shift_floor > 0`, blending is gated smoothly per image by mean active
        log-variance shift from running stats so low-shift domains keep clean stats while shifted
        OOD domains receive up to `hrnet_adabn_lambda` recalibration.
        Strictly non-destructive to running_mean/running_var buffers and independent per image.
        """
        lam = self.hrnet_adabn_lambda
        if lam <= 0.0:
            return hrnet_net(x_in)

        shift_floor = self.hrnet_adabn_shift_floor
        shift_scale = self.hrnet_adabn_shift_scale
        stages = self.hrnet_adabn_stages
        hooks = []

        def _make_adabn_hook(blend_lam: float, floor: float, scale: float):
            def _hook(mod: nn.BatchNorm2d, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
                x = inp[0]
                if x.ndim != 4 or mod.running_mean is None or mod.running_var is None:
                    return out
                rm = mod.running_mean.float()
                rv = mod.running_var.float()
                active = rv > 1e-5
                if not active.any():
                    return out
                xf = x.float()
                img_mean = xf.mean(dim=(2, 3), keepdim=True)
                img_var = xf.var(dim=(2, 3), keepdim=True, unbiased=False)
                rm_v = rm.view(1, -1, 1, 1)
                rv_v = rv.view(1, -1, 1, 1)
                act_v = active.view(1, -1, 1, 1)
                if floor > 0.0:
                    img_shift = (
                        torch.log((img_var + mod.eps) / (rv_v + mod.eps)).abs() * act_v
                    ).sum(dim=1, keepdim=True) / act_v.sum(dim=1, keepdim=True).clamp(min=1.0)
                    eff_lam = blend_lam * ((img_shift - floor) / scale).clamp(0.0, 1.0)
                else:
                    eff_lam = blend_lam
                mix_mean = torch.where(act_v, (1.0 - eff_lam) * rm_v + eff_lam * img_mean, rm_v)
                mix_var = torch.where(act_v, (1.0 - eff_lam) * rv_v + eff_lam * img_var, rv_v)
                w = mod.weight.float().view(1, -1, 1, 1) if mod.weight is not None else 1.0
                b = mod.bias.float().view(1, -1, 1, 1) if mod.bias is not None else 0.0
                normed = (xf - mix_mean) / torch.sqrt(mix_var + mod.eps)
                return (normed * w + b).to(dtype=out.dtype)
            return _hook

        for mod_name, mod in hrnet_net.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and any(mod_name.startswith(st) for st in stages):
                hooks.append(mod.register_forward_hook(_make_adabn_hook(lam, shift_floor, shift_scale)))
        try:
            return hrnet_net(x_in)
        finally:
            for h in hooks:
                h.remove()

    def _extract_secondary_features(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Extract features from the optional secondary backbone."""
        assert self.secondary_net is not None
        x_sec = x_raw
        if self.secondary_kind == "vit" and self.vit_input_size is not None:
            if x_sec.shape[2] != self.vit_input_size or x_sec.shape[3] != self.vit_input_size:
                x_sec = F.interpolate(
                    x_sec,
                    size=(self.vit_input_size, self.vit_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
        if not self.secondary_trainable and not any(p.requires_grad for p in self.secondary_net.parameters()):
            with torch.no_grad():
                if self.secondary_kind == "hrnet":
                    return self._forward_hrnet_maybe_adabn(self.secondary_net, x_sec)
                return self.secondary_net(x_sec)
        if self.secondary_kind == "hrnet":
            return self._forward_hrnet_maybe_adabn(self.secondary_net, x_sec)
        return self.secondary_net(x_sec)

    def _extract_all_branch_features_with_taps_single(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Dict[int, torch.Tensor],
    ]:
        """Run primary (with optional ViT taps), optional secondary, fusion, and FeatureAdapter."""
        x_raw = self._ensure_raw_rgb(x)
        x_in = self._prepare_backbone_input(x_raw)

        vit_taps: Dict[int, torch.Tensor] = {}
        use_taps = bool(self.vit_feature_taps) and self._canonical_backbone_kind(self.backbone_name) == "vit"
        is_primary_hrnet = self._canonical_backbone_kind(self.backbone_name) == "hrnet"
        primary_frozen = self.freeze_primary_backbone and not any(
            p.requires_grad for p in self.net.parameters()
        )
        if primary_frozen:
            with torch.no_grad():
                if use_taps:
                    f_primary, vit_taps = self.net.forward_with_taps(x_in, self.vit_feature_taps)
                elif is_primary_hrnet:
                    f_primary = self._forward_hrnet_maybe_adabn(self.net, x_in)
                else:
                    f_primary = self.net(x_in)
        else:
            if use_taps:
                f_primary, vit_taps = self.net.forward_with_taps(x_in, self.vit_feature_taps)
            elif is_primary_hrnet:
                f_primary = self._forward_hrnet_maybe_adabn(self.net, x_in)
            else:
                f_primary = self.net(x_in)

        f_secondary: Optional[torch.Tensor] = None
        if self.secondary_net is not None and self.fusion_module is not None:
            f_secondary = self._extract_secondary_features(x_raw)
            if self.fusion_spatial_align == "max":
                target_h = max(f_primary.shape[2], f_secondary.shape[2])
                target_w = max(f_primary.shape[3], f_secondary.shape[3])
                if f_primary.shape[2] != target_h or f_primary.shape[3] != target_w:
                    f_primary = F.interpolate(
                        f_primary, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )
                if f_secondary.shape[2] != target_h or f_secondary.shape[3] != target_w:
                    f_secondary = F.interpolate(
                        f_secondary, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )
            elif f_secondary.shape[2:] != f_primary.shape[2:]:
                f_secondary = F.interpolate(
                    f_secondary, size=f_primary.shape[2:], mode="bilinear", align_corners=False
                )
            feats_backbone = self.fusion_module(f_primary, f_secondary)
        else:
            feats_backbone = f_primary

        feats = self.feature_adapter(feats_backbone) if self.feature_adapter is not None else feats_backbone
        return x_in, f_primary, f_secondary, feats_backbone, feats, vit_taps

    def _extract_all_branch_features_with_taps(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Dict[int, torch.Tensor],
    ]:
        """Extract branch, fused, and ViT tap features with optional horizontal-flip eval TTA."""
        x_in, f_p, f_s, feats_bb, feats, vit_taps = self._extract_all_branch_features_with_taps_single(x)
        if not self.training and self.tta_hflip:
            _, f_p2, f_s2, feats_bb2, feats2, vit_taps2 = self._extract_all_branch_features_with_taps_single(
                torch.flip(x, dims=[-1])
            )

            def _avg_l2(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if a is None or b is None:
                    return a
                b_flip = torch.flip(b, dims=[-1])
                a_n = F.normalize(a.float(), p=2, dim=1)
                b_n = F.normalize(b_flip.float(), p=2, dim=1)
                return F.normalize(0.5 * (a_n + b_n), p=2, dim=1).to(dtype=a.dtype)

            f_p = _avg_l2(f_p, f_p2)  # type: ignore[assignment]
            f_s = _avg_l2(f_s, f_s2)
            feats_bb = _avg_l2(feats_bb, feats_bb2)  # type: ignore[assignment]
            feats = _avg_l2(feats, feats2)  # type: ignore[assignment]
            for k in list(vit_taps.keys()):
                if k in vit_taps2:
                    vit_taps[k] = _avg_l2(vit_taps[k], vit_taps2[k])  # type: ignore[assignment]
        return x_in, f_p, f_s, feats_bb, feats, vit_taps

    def _extract_all_branch_features_single(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run primary (and optional secondary + fusion) backbones and optional FeatureAdapter."""
        x_in, f_p, f_s, feats_bb, feats, _ = self._extract_all_branch_features_with_taps_single(x)
        return x_in, f_p, f_s, feats_bb, feats

    def _extract_all_branch_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Extract branch and fused features, applying horizontal-flip eval TTA when enabled."""
        x_in, f_p, f_s, feats_bb, feats, _ = self._extract_all_branch_features_with_taps(x)
        return x_in, f_p, f_s, feats_bb, feats

    def _extract_primary_and_fused_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run primary (and optional secondary + fusion) backbones and optional FeatureAdapter."""
        x_in, f_primary, _, feats_backbone, feats = self._extract_all_branch_features(x)
        return x_in, f_primary, feats_backbone, feats

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial forensic features from input images (including fusion & FeatureAdapter)."""
        _, _, _, feats = self._extract_primary_and_fused_features(x)
        return feats

    @staticmethod
    @torch.no_grad()
    def _forward_vit_base_no_grad(
        vit_net: FOCALViTNet,
        x_in: torch.Tensor,
        anchor_neck: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Run frozen base ViT forward pass with LoRA/DoRA bypassed and optional anchored neck."""
        lora_mods = [m for m in vit_net.modules() if isinstance(m, SurgicalLoRALinear)]
        for m in lora_mods:
            m.bypass = True
        try:
            encoder = vit_net.image_encoder
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
            neck_mod = anchor_neck if anchor_neck is not None else encoder.neck
            neck_in = feat.permute(0, 3, 1, 2)
            if anchor_neck is not None:
                try:
                    anchor_dev = next(anchor_neck.parameters()).device
                    if neck_in.device != anchor_dev:
                        return anchor_neck(neck_in.to(anchor_dev)).to(neck_in.device)
                except StopIteration:
                    pass
            return neck_mod(neck_in)
        finally:
            for m in lora_mods:
                m.bypass = False

    @staticmethod
    @torch.no_grad()
    def _forward_hrnet_with_stage4_no_grad(
        hrnet_net: HRNet,
        x_in: torch.Tensor,
        anchor_stage4: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Run frozen base HRNet forward pass using optional anchored stage4."""
        if anchor_stage4 is None:
            return hrnet_net(x_in)
        h = hrnet_net.relu(hrnet_net.bn1(hrnet_net.conv1(x_in)))
        h = hrnet_net.relu(hrnet_net.bn2(hrnet_net.conv2(h)))
        h = hrnet_net.layer1(h)
        x_list = [
            hrnet_net.transition1[i](h) if hrnet_net.transition1[i] is not None else h
            for i in range(hrnet_net.stage2_cfg["NUM_BRANCHES"])
        ]
        y_list = hrnet_net.stage2(x_list)
        x_list = [
            hrnet_net.transition2[i](y_list[-1]) if hrnet_net.transition2[i] is not None else y_list[i]
            for i in range(hrnet_net.stage3_cfg["NUM_BRANCHES"])
        ]
        y_list = hrnet_net.stage3(x_list)
        x_list = [
            hrnet_net.transition3[i](y_list[-1]) if hrnet_net.transition3[i] is not None else y_list[i]
            for i in range(hrnet_net.stage4_cfg["NUM_BRANCHES"])
        ]
        try:
            anchor_dev = next(anchor_stage4.parameters()).device
            if x_in.device != anchor_dev:
                x_list_dev = [t.to(anchor_dev) for t in x_list]
                return anchor_stage4(x_list_dev)[0].to(x_in.device)
        except StopIteration:
            pass
        return anchor_stage4(x_list)[0]

    @torch.no_grad()
    def _extract_base_features_no_grad(self, x_in: torch.Tensor) -> torch.Tensor:
        """Compute unmodified pretrained primary backbone features without duplicating the 307M ViT in VRAM."""
        if self._canonical_backbone_kind(self.backbone_name) == "vit":
            return self._forward_vit_base_no_grad(self.net, x_in, anchor_neck=self._anchor_neck)
        return self._forward_hrnet_with_stage4_no_grad(
            self.net, x_in, anchor_stage4=self._anchor_hrnet_stage4
        )

    @torch.no_grad()
    def _extract_secondary_base_features_no_grad(self, x_raw: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute unmodified pretrained secondary backbone features for dual-backbone prior anchoring."""
        if self.secondary_net is None:
            return None
        x_sec = x_raw
        if self.secondary_kind == "vit":
            if self.vit_input_size is not None and (x_sec.shape[2] != self.vit_input_size or x_sec.shape[3] != self.vit_input_size):
                x_sec = F.interpolate(x_sec, size=(self.vit_input_size, self.vit_input_size), mode="bilinear", align_corners=False)
            return self._forward_vit_base_no_grad(
                self.secondary_net, x_sec, anchor_neck=self._anchor_secondary_neck
            )
        if self.secondary_kind == "hrnet":
            return self._forward_hrnet_with_stage4_no_grad(
                self.secondary_net, x_sec, anchor_stage4=self._anchor_secondary_hrnet_stage4
            )
        return self.secondary_net(x_sec)

    def _compute_prior_anchor_terms(
        self,
        x_in: torch.Tensor,
        feats: torch.Tensor,
        feats_backbone: torch.Tensor,
        f_primary: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute feature-space cosine prior-anchoring loss, optional neck L2 anchor penalty,
        and optional residual refiner L2 regularization.
        Returns `(raw_cosine_drift, weighted_total_penalty)`.
        """
        raw_cos_loss: Optional[torch.Tensor] = None
        weighted_penalty: Optional[torch.Tensor] = None

        if self.prior_anchor_weight > 0.0:
            feats_base: Optional[torch.Tensor] = None
            if self.secondary_net is not None:
                if (
                    self.freeze_primary_backbone
                    and not self._surgical_secondary_param_names
                    and self.feature_adapter is not None
                    and feats_backbone.shape[1] == feats.shape[1]
                ):
                    feats_base = feats_backbone.detach()
                elif self.freeze_primary_backbone and f_primary is not None and f_primary.shape[1] == feats.shape[1]:
                    feats_base = f_primary.detach()
                elif self.fusion_mode == "concat":
                    x_raw = self._ensure_raw_rgb(x_in)
                    f_p_base = (
                        f_primary.detach()
                        if (self.freeze_primary_backbone and f_primary is not None)
                        else self._extract_base_features_no_grad(x_in)
                    )
                    f_s_base = self._extract_secondary_base_features_no_grad(x_raw)
                    if f_p_base is not None and f_s_base is not None:
                        target_hw = feats.shape[2:]
                        if f_p_base.shape[2:] != target_hw:
                            f_p_base = F.interpolate(f_p_base, size=target_hw, mode="bilinear", align_corners=False)
                        if f_s_base.shape[2:] != target_hw:
                            f_s_base = F.interpolate(f_s_base, size=target_hw, mode="bilinear", align_corners=False)
                        p_n = F.normalize(f_p_base.float(), p=2, dim=1)
                        s_n = F.normalize(f_s_base.float(), p=2, dim=1)
                        feats_base = torch.cat([p_n, s_n], dim=1)
            elif not self._surgical_backbone_param_names and self.feature_adapter is not None:
                feats_base = feats_backbone.detach()
            else:
                feats_base = self._extract_base_features_no_grad(x_in)

            if feats_base is not None:
                if feats_base.shape[2:] != feats.shape[2:]:
                    feats_base = F.interpolate(
                        feats_base, size=feats.shape[2:], mode="bilinear", align_corners=False
                    )
                if isinstance(self.feature_adapter, BranchBalancedSpatialAdapter):
                    p_dim = self.feature_adapter.primary_dim
                    f_p_ad = F.normalize(feats[:, :p_dim].float(), p=2, dim=1)
                    f_p_ba = F.normalize(feats_base[:, :p_dim].detach().float(), p=2, dim=1)
                    f_s_ad = F.normalize(feats[:, p_dim:].float(), p=2, dim=1)
                    f_s_ba = F.normalize(feats_base[:, p_dim:].detach().float(), p=2, dim=1)
                    drift_p = (1.0 - (f_p_ad * f_p_ba).sum(dim=1)).mean()
                    drift_s = (1.0 - (f_s_ad * f_s_ba).sum(dim=1)).mean()
                    raw_cos_loss = 0.5 * (drift_p + drift_s)
                else:
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

        if self.cluster_refiner_l2_weight > 0.0 and self.cluster_refiner is not None:
            norm_weight_ids = {
                id(m.weight)
                for m in self.cluster_refiner.modules()
                if isinstance(
                    m,
                    (
                        nn.GroupNorm,
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.LayerNorm,
                        nn.InstanceNorm2d,
                    ),
                )
                and getattr(m, "weight", None) is not None
            }
            ref_l2_terms = [
                (p.float() - 1.0).pow(2).mean() if id(p) in norm_weight_ids else p.float().pow(2).mean()
                for p in self.cluster_refiner.parameters()
                if p.requires_grad
            ]
            if ref_l2_terms:
                ref_l2 = torch.stack(ref_l2_terms).mean()
                weighted_ref = self.cluster_refiner_l2_weight * ref_l2
                weighted_penalty = weighted_ref if weighted_penalty is None else (weighted_penalty + weighted_ref)

        if self.refiner_ortho_weight > 0.0 and isinstance(self.cluster_refiner, ImageRelativeClusterRefiner):
            w_out = self.cluster_refiner.out_conv.weight.float().view(2, -1)
            n0 = w_out[0].norm()
            n1 = w_out[1].norm()
            if float(n0.item()) > 1e-6 and float(n1.item()) > 1e-6:
                cos_01 = F.cosine_similarity(w_out[0], w_out[1], dim=0)
                ortho_pen = self.refiner_ortho_weight * cos_01.pow(2)
            else:
                ortho_pen = w_out.sum() * 0.0
            weighted_penalty = ortho_pen if weighted_penalty is None else (weighted_penalty + ortho_pen)

        if self.adapter_ortho_weight > 0.0 and isinstance(self.feature_adapter, BranchBalancedSpatialAdapter):
            w_sec = self.feature_adapter.secondary_adapter.proj_out.weight.float()
            c_sec = w_sec.shape[0]
            w_flat = w_sec.view(c_sec, -1)
            norms = w_flat.norm(dim=1, keepdim=True)
            if float(norms.max().item()) > 1e-6:
                w_norm = w_flat / norms.clamp(min=1e-6)
                gram = torch.matmul(w_norm, w_norm.t())
                off_diag = gram - torch.eye(c_sec, device=gram.device, dtype=gram.dtype)
                ad_ortho = self.adapter_ortho_weight * off_diag.pow(2).mean()
            else:
                ad_ortho = w_sec.sum() * 0.0
            weighted_penalty = ad_ortho if weighted_penalty is None else (weighted_penalty + ad_ortho)

        if self.preneck_ortho_weight > 0.0 and self.preneck_projector is not None:
            w_pn = self.preneck_projector.proj.weight.float()
            c_pn = w_pn.shape[0]
            w_pn_flat = w_pn.view(c_pn, -1)
            w_pn_norm = F.normalize(w_pn_flat, p=2, dim=1)
            gram_pn = torch.matmul(w_pn_norm, w_pn_norm.t())
            off_pn = gram_pn - torch.eye(c_pn, device=gram_pn.device, dtype=gram_pn.dtype)
            pn_ortho = self.preneck_ortho_weight * off_pn.pow(2).mean()
            weighted_penalty = pn_ortho if weighted_penalty is None else (weighted_penalty + pn_ortho)

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

    def _resolve_cluster_polarity(
        self,
        labels: torch.Tensor,
        f_img: torch.Tensor,
        centroids: torch.Tensor,
        c0_count: torch.Tensor,
        c1_count: torch.Tensor,
        h_f: int,
        w_f: int,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Determine which cluster index (`0` vs `1`) represents tampered pixels.

        Returns `(cond, polarity_loss)` where `cond` (`[B, 1]` bool) is True when cluster 0 is tampered.
        """
        n_px = h_f * w_f
        cond_min = c1_count > c0_count
        if self.cluster_polarity_mode == "minority":
            return cond_min, None

        b = f_img.shape[0]
        with torch.no_grad():
            area0 = c0_count / float(max(1, n_px))
            area1 = c1_count / float(max(1, n_px))
            grid = labels.view(b, h_f, w_f)
            bmask = torch.zeros((h_f, w_f), dtype=torch.bool, device=f_img.device)
            bmask[0, :] = True
            bmask[-1, :] = True
            bmask[:, 0] = True
            bmask[:, -1] = True
            b_flat = bmask.view(1, n_px).expand(b, -1)
            n_border = max(1.0, float(bmask.sum().item()))
            bf1 = ((labels == 1) & b_flat).float().sum(dim=-1, keepdim=True) / n_border
            bf0 = 1.0 - bf1
            z_border = (area1 + self.polarity_border_weight * bf1) - (area0 + self.polarity_border_weight * bf0)
            z_min = area1 - area0
            if self.polarity_area_gate > 0.0:
                min_area = torch.minimum(area0, area1)
                z_base = torch.where(min_area > self.polarity_area_gate, z_border, z_min)
            else:
                z_base = z_border

        if self.cluster_polarity_mode == "border_aware" or self.polarity_head is None:
            return (z_base > 0.0), None

        with torch.no_grad():
            g0, g1 = ClusterPolarityHead.extract_cluster_descriptors(labels, f_img, centroids, h_f, w_f)
        polarity_logit = self.polarity_head(g0, g1, z_base, margin_scale=self.polarity_margin_scale)
        cond = polarity_logit.detach() > 0.0

        pol_loss: Optional[torch.Tensor] = None
        if self.training:
            # Always keep polarity_head connected to autograd for DDP unused-parameter safety.
            pol_anchor = polarity_logit.sum() * 0.0
            if gt_masks is not None:
                with torch.no_grad():
                    gt_f = gt_masks.float()
                    if gt_f.ndim == 3:
                        gt_f = gt_f.unsqueeze(1)
                    if gt_f.shape[2:] != (h_f, w_f):
                        gt_f = F.interpolate(gt_f, size=(h_f, w_f), mode="area")
                    gt_bin = (gt_f[:, 0].reshape(b, n_px) >= 0.5).float()
                    m0 = (labels == 0).float()
                    m1 = (labels == 1).float()
                    iou0 = (m0 * gt_bin).sum(dim=1) / ((m0 + gt_bin - m0 * gt_bin).sum(dim=1) + 1e-6)
                    iou1 = (m1 * gt_bin).sum(dim=1) / ((m1 + gt_bin - m1 * gt_bin).sum(dim=1) + 1e-6)
                    valid = (gt_bin.sum(dim=1) > 0.0) & ((iou0 - iou1).abs() > 0.02)
                    y_pol = (iou0 > iou1).float()
                if valid.any():
                    pol_loss = F.binary_cross_entropy_with_logits(
                        polarity_logit.squeeze(-1)[valid], y_pol[valid]
                    ) + pol_anchor
                else:
                    pol_loss = pol_anchor
            else:
                pol_loss = pol_anchor

        return cond, pol_loss

    def _cluster_features_with_polarity(
        self,
        feats: torch.Tensor,
        orig_h: int,
        orig_w: int,
        aux_pristine_flags: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        On-the-fly unsupervised clustering (`"kmeans"`, `"kmeans_robust"`, or `"soft_kmeans"`)
        with configurable cluster polarity resolution and optional supervised polarity loss.
        """
        b, c, h_f, w_f = feats.shape
        n_px = h_f * w_f
        with torch.no_grad():
            feats_norm = F.normalize(feats.float(), p=2, dim=1)
            if self.feature_centering_alpha > 0.0:
                mu_f = feats_norm.mean(dim=(2, 3), keepdim=True)
                feats_norm = F.normalize(feats_norm - self.feature_centering_alpha * mu_f, p=2, dim=1)
            f_img = feats_norm.permute(0, 2, 3, 1).reshape(b, n_px, c)

            if self.clustering_mode == "soft_kmeans":
                # Global mean anchor + top-2% farthest-point initialization (immune to corner/outlier pixels)
                c1 = F.normalize(f_img.mean(dim=1, keepdim=True), p=2, dim=-1)
                dists1 = 1.0 - torch.bmm(f_img, c1.transpose(1, 2)).squeeze(-1)
                k_far = max(1, n_px // 50)
                far_idx = torch.topk(dists1, k=k_far, dim=-1).indices
                far_vecs = torch.gather(f_img, dim=1, index=far_idx.unsqueeze(-1).expand(-1, -1, c))
                c2 = F.normalize(far_vecs.mean(dim=1, keepdim=True), p=2, dim=-1)
                centroids = torch.cat([c1, c2], dim=1)  # [B, 2, C]

                for step in range(10):
                    em_temp = 0.15 if step < 5 else 0.08
                    sims = torch.bmm(f_img, centroids.transpose(1, 2))
                    probs = torch.softmax(sims / em_temp, dim=-1)
                    w0 = probs[:, :, 0:1]
                    w1 = probs[:, :, 1:2]
                    mass0 = w0.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    mass1 = w1.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    new_c0 = F.normalize(torch.bmm(w0.transpose(1, 2), f_img) / mass0, p=2, dim=-1)
                    new_c1 = F.normalize(torch.bmm(w1.transpose(1, 2), f_img) / mass1, p=2, dim=-1)
                    centroids = torch.cat([new_c0, new_c1], dim=1)

                sims = torch.bmm(f_img, centroids.transpose(1, 2))
                probs = torch.softmax(sims / 0.08, dim=-1)
                labels = torch.argmax(probs, dim=-1)
                c0_count = probs[:, :, 0].sum(dim=-1, keepdim=True)
                c1_count = probs[:, :, 1].sum(dim=-1, keepdim=True)
            else:
                if self.clustering_mode == "kmeans_robust":
                    # Flip/rotation-equivariant global mean + top-2% farthest-point initialization
                    c_mean = F.normalize(f_img.mean(dim=1, keepdim=True), p=2, dim=-1)
                    k_far = max(1, n_px // 50)
                    dists1 = 1.0 - torch.bmm(f_img, c_mean.transpose(1, 2)).squeeze(-1)
                    far1 = torch.topk(dists1, k=k_far, dim=-1).indices
                    c2 = F.normalize(
                        torch.gather(f_img, 1, far1.unsqueeze(-1).expand(-1, -1, c)).mean(dim=1, keepdim=True),
                        p=2,
                        dim=-1,
                    )
                    dists2 = 1.0 - torch.bmm(f_img, c2.transpose(1, 2)).squeeze(-1)
                    far2 = torch.topk(dists2, k=k_far, dim=-1).indices
                    c1 = F.normalize(
                        torch.gather(f_img, 1, far2.unsqueeze(-1).expand(-1, -1, c)).mean(dim=1, keepdim=True),
                        p=2,
                        dim=-1,
                    )
                else:
                    # Legacy double farthest-point initialization (k-means++ style)
                    c1 = f_img[:, 0:1, :]
                    dists1 = 1.0 - torch.bmm(f_img, c1.transpose(1, 2)).squeeze(-1)
                    idx_max1 = torch.argmax(dists1, dim=-1, keepdim=True)
                    c2 = torch.gather(f_img, dim=1, index=idx_max1.unsqueeze(-1).expand(-1, 1, c))

                    dists2 = 1.0 - torch.bmm(f_img, c2.transpose(1, 2)).squeeze(-1)
                    idx_max2 = torch.argmax(dists2, dim=-1, keepdim=True)
                    c1 = torch.gather(f_img, dim=1, index=idx_max2.unsqueeze(-1).expand(-1, 1, c))

                centroids = torch.cat([c1, c2], dim=1)  # [B, 2, C]

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

                sims = torch.bmm(f_img, centroids.transpose(1, 2))  # [B, N, 2]
                labels = torch.argmax(sims, dim=-1)  # [B, N]
                c1_count = labels.sum(dim=-1, keepdim=True).to(f_img.dtype)  # [B, 1]
                c0_count = float(n_px) - c1_count  # [B, 1]

            cos_sim = (centroids[:, 0] * centroids[:, 1]).sum(dim=-1)  # [B]
            separability = (1.0 - cos_sim).clamp(min=0.0)

        cond, pol_loss = self._resolve_cluster_polarity(
            labels=labels,
            f_img=f_img,
            centroids=centroids,
            c0_count=c0_count,
            c1_count=c1_count,
            h_f=h_f,
            w_f=w_f,
            gt_masks=gt_masks,
        )

        with torch.no_grad():
            sim_tampered = torch.where(cond, sims[:, :, 0], sims[:, :, 1])  # [B, N]
            sim_pristine = torch.where(cond, sims[:, :, 1], sims[:, :, 0])  # [B, N]
            tampered_ratio = (torch.where(cond, c0_count, c1_count).squeeze(-1) / float(max(1, n_px))).clamp(0.0, 1.0)

            diff = (sim_tampered - sim_pristine) / self.cluster_temperature  # [B, N]

            pristine_mask = separability < self.cluster_separability_threshold
            if aux_pristine_flags is not None:
                pristine_mask = pristine_mask | aux_pristine_flags.to(device=separability.device, dtype=torch.bool)
            diff = self._apply_pristine_gate(diff, pristine_mask)

            logit_grid = diff.view(b, 1, h_f, w_f)
            logit_full = F.interpolate(logit_grid, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            prob_full = torch.sigmoid(logit_full)

        return logit_full, prob_full, separability, tampered_ratio, pol_loss

    def _cluster_features_with_full_stats(
        self,
        feats: torch.Tensor,
        orig_h: int,
        orig_w: int,
        aux_pristine_flags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        On-the-fly unsupervised clustering mask and logit prediction.
        Returns `(logit_full, prob_full, cluster_separability, minority_ratio)`.
        """
        logit_full, prob_full, separability, tampered_ratio, _ = self._cluster_features_with_polarity(
            feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags, gt_masks=None
        )
        return logit_full, prob_full, separability, tampered_ratio

    @staticmethod
    @torch.no_grad()
    def _build_affinity_matrix(feats: torch.Tensor, tau: float = 0.08, grid_size: Optional[int] = 48) -> torch.Tensor:
        """Build per-image row-stochastic forensic affinity matrix `P` (`[B, N, N]`) at compact grid size."""
        f_cur = feats.float()
        if grid_size is not None and f_cur.shape[2:] != (grid_size, grid_size):
            f_cur = F.interpolate(f_cur, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
        b, c, h_g, w_g = f_cur.shape
        f_img = F.normalize(f_cur, p=2, dim=1).permute(0, 2, 3, 1).reshape(b, h_g * w_g, c)
        sim = torch.bmm(f_img, f_img.transpose(1, 2))
        w_aff = torch.exp(((sim - 1.0) / max(1e-3, float(tau))).clamp(min=-40.0, max=0.0))
        return w_aff / w_aff.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    @staticmethod
    @torch.no_grad()
    def _smooth_features_on_graph(feats: torch.Tensor, affinity_P: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply one-step spherical manifold feature smoothing `(1-g)*F + g*(P@F)` at `affinity_P`'s grid."""
        b, c, h_f, w_f = feats.shape
        n_aff = affinity_P.shape[1]
        side = int(round(math.sqrt(n_aff)))
        f_cur = feats.float()
        if (h_f, w_f) != (side, side):
            f_cur = F.interpolate(f_cur, size=(side, side), mode="bilinear", align_corners=False)
        f_flat = F.normalize(f_cur, p=2, dim=1).permute(0, 2, 3, 1).reshape(b, n_aff, c)
        f_smooth = F.normalize((1.0 - gamma) * f_flat + gamma * torch.bmm(affinity_P, f_flat), p=2, dim=-1)
        out = f_smooth.reshape(b, side, side, c).permute(0, 3, 1, 2)
        if (h_f, w_f) != (side, side):
            out = F.interpolate(out, size=(h_f, w_f), mode="bilinear", align_corners=False)
        return out.to(dtype=feats.dtype)

    @staticmethod
    def _diffuse_logits_on_graph(
        logits_full: torch.Tensor,
        affinity_P: torch.Tensor,
        beta: float,
        steps: int = 2,
    ) -> torch.Tensor:
        """Apply differentiable Markov affinity diffusion over detached `affinity_P` and upsample back."""
        b, _, orig_h, orig_w = logits_full.shape
        n_aff = affinity_P.shape[1]
        side = int(round(math.sqrt(n_aff)))
        aff_det = affinity_P.detach().float()
        l0 = F.interpolate(logits_full.float(), size=(side, side), mode="bilinear", align_corners=False).reshape(b, n_aff, 1)
        l_cur = l0
        for _ in range(max(1, int(steps))):
            l_cur = (1.0 - beta) * l0 + beta * torch.bmm(aff_det, l_cur)
        l_grid = l_cur.reshape(b, 1, side, side)
        return F.interpolate(l_grid, size=(orig_h, orig_w), mode="bilinear", align_corners=False).to(dtype=logits_full.dtype)

    @staticmethod
    def _apply_otsu_calibration(logits: torch.Tensor, alpha: float, n_bins: int = 64) -> torch.Tensor:
        """Recenter logits per image toward the detached 1D Otsu valley threshold `t_otsu`."""
        if alpha <= 0.0:
            return logits
        b = logits.shape[0]
        with torch.no_grad():
            probs = torch.sigmoid(logits.float())
            p_flat = probs.view(b, -1).clamp(1e-4, 1.0 - 1e-4)
            centers = (torch.arange(n_bins, device=logits.device, dtype=torch.float32) + 0.5) / float(n_bins)
            idx = torch.clamp((p_flat * n_bins).long(), 0, n_bins - 1)
            hist = torch.zeros((b, n_bins), device=logits.device, dtype=torch.float32)
            hist.scatter_add_(1, idx, torch.ones_like(p_flat))
            w1 = torch.cumsum(hist, dim=1)
            w2 = hist.sum(dim=1, keepdim=True) - w1
            m_cum = torch.cumsum(hist * centers.unsqueeze(0), dim=1)
            m_tot = m_cum[:, -1:]
            valid = (w1 > 0) & (w2 > 0)
            mu1 = m_cum / w1.clamp(min=1.0)
            mu2 = (m_tot - m_cum) / w2.clamp(min=1.0)
            var_between = torch.where(valid, w1 * w2 * (mu1 - mu2).pow(2), torch.zeros_like(w1))
            best_bin = torch.argmax(var_between, dim=1)
            t_ot = centers[best_bin].clamp(0.05, 0.95).view(b, 1, 1, 1)
            logit_ot = torch.log(t_ot / (1.0 - t_ot))
        return (logits.float() - float(alpha) * logit_ot.detach()).to(dtype=logits.dtype)

    def _build_preneck_tap_tensor(self, vit_taps: Dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
        """Concatenate L2-normalized pre-neck ViT taps at compact `vit_tap_cluster_size`."""
        ordered_taps = [vit_taps[k] for k in self.vit_feature_taps if k in vit_taps]
        if not ordered_taps:
            return None
        target_hw = (
            (self.vit_tap_cluster_size, self.vit_tap_cluster_size)
            if self.vit_tap_cluster_size is not None
            else ordered_taps[0].shape[2:]
        )
        parts: List[torch.Tensor] = []
        for t in ordered_taps:
            t_cur = t
            if t_cur.shape[2:] != target_hw:
                t_cur = F.interpolate(t_cur, size=target_hw, mode="bilinear", align_corners=False)
            parts.append(F.normalize(t_cur.float(), p=2, dim=1))
        return torch.cat(parts, dim=1)

    def _build_multitap_cluster_tensor(
        self,
        f_primary: torch.Tensor,
        f_secondary: Optional[torch.Tensor],
        vit_taps: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Build L2-normalized concatenated multi-tap feature map at compact `vit_tap_cluster_size`."""
        ordered_taps = [vit_taps[k] for k in self.vit_feature_taps if k in vit_taps]
        first_ref = ordered_taps[0] if ordered_taps else f_primary
        if self.vit_tap_cluster_size is not None:
            target_hw = (self.vit_tap_cluster_size, self.vit_tap_cluster_size)
        else:
            target_hw = first_ref.shape[2:]

        parts: List[torch.Tensor] = []
        for t in [f_primary] + ordered_taps + ([f_secondary] if f_secondary is not None else []):
            t_cur = t
            if t_cur.shape[2:] != target_hw:
                t_cur = F.interpolate(t_cur, size=target_hw, mode="bilinear", align_corners=False)
            parts.append(F.normalize(t_cur.float(), p=2, dim=1))
        return torch.cat(parts, dim=1)

    def _fuse_multitap_clustering(
        self,
        base_logits: torch.Tensor,
        base_sep: torch.Tensor,
        base_area: torch.Tensor,
        f_primary: torch.Tensor,
        f_secondary: Optional[torch.Tensor],
        vit_taps: Dict[int, torch.Tensor],
        orig_h: int,
        orig_w: int,
        aux_pristine_flags: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        preneck_feats: Optional[torch.Tensor] = None,
        affinity_P: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Combine base fused-feature cluster logits with pre-neck ViT multi-tap cluster logits."""
        if not vit_taps:
            return base_logits, base_sep, base_area, None, affinity_P

        branch_logits: List[torch.Tensor] = [base_logits]
        branch_seps: List[torch.Tensor] = [base_sep]
        branch_areas: List[torch.Tensor] = [base_area]
        pol_losses: List[torch.Tensor] = []

        if self.vit_tap_fusion_mode == "dual_logit":
            mt_feats = self._build_multitap_cluster_tensor(
                f_primary.detach(),
                f_secondary.detach() if f_secondary is not None else None,
                {k: v.detach() for k, v in vit_taps.items()},
            )
            if affinity_P is None and (self.feat_smooth_gamma > 0.0 or self.affinity_diff_beta > 0.0 or self.refiner_use_affinity_prop):
                affinity_P = self._build_affinity_matrix(
                    mt_feats,
                    tau=self.affinity_diff_tau,
                    grid_size=self.vit_tap_cluster_size or 48,
                )
            if self.feat_smooth_gamma > 0.0 and affinity_P is not None:
                mt_feats = self._smooth_features_on_graph(mt_feats, affinity_P, self.feat_smooth_gamma)

            mt_l, _, mt_sep, mt_area, mt_pl = self._cluster_features_with_polarity(
                mt_feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags, gt_masks=gt_masks
            )
            branch_logits.append(mt_l)
            branch_seps.append(mt_sep)
            branch_areas.append(mt_area)
            if mt_pl is not None:
                pol_losses.append(mt_pl)
        else:
            for k in self.vit_feature_taps:
                if k not in vit_taps:
                    continue
                t_feats = vit_taps[k].detach()
                if self.vit_tap_cluster_size is not None and t_feats.shape[2:] != (
                    self.vit_tap_cluster_size,
                    self.vit_tap_cluster_size,
                ):
                    t_feats = F.interpolate(
                        t_feats,
                        size=(self.vit_tap_cluster_size, self.vit_tap_cluster_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                tl, _, ts, ta, tpl = self._cluster_features_with_polarity(
                    t_feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags, gt_masks=gt_masks
                )
                branch_logits.append(tl)
                branch_seps.append(ts)
                branch_areas.append(ta)
                if tpl is not None:
                    pol_losses.append(tpl)

        if preneck_feats is not None:
            pn_l, _, pn_sep, pn_area, pn_pl = self._cluster_features_with_polarity(
                preneck_feats.detach(), orig_h, orig_w, aux_pristine_flags=aux_pristine_flags, gt_masks=gt_masks
            )
            branch_logits.append(pn_l)
            branch_seps.append(pn_sep)
            branch_areas.append(pn_area)
            if pn_pl is not None:
                pol_losses.append(pn_pl)

        n_b = len(branch_logits)
        if self.tap_logit_weights is not None and self.tap_logit_weights.numel() == n_b:
            weights = F.softmax(self.tap_logit_weights.float(), dim=0)
        else:
            weights = torch.full((n_b,), 1.0 / float(n_b), device=base_logits.device, dtype=torch.float32)

        fused_logits = sum(weights[i] * branch_logits[i].float() for i in range(n_b))
        w_det = weights.detach()
        fused_sep = sum(w_det[i] * branch_seps[i] for i in range(n_b))
        fused_area = sum(w_det[i] * branch_areas[i] for i in range(n_b))
        pol_loss = torch.stack(pol_losses).mean() if pol_losses else None
        return fused_logits, fused_sep, fused_area, pol_loss, affinity_P

    def _cluster_features_with_stats(
        self,
        feats: torch.Tensor,
        orig_h: int,
        orig_w: int,
        aux_pristine_flags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        On-the-fly unsupervised clustering mask and logit prediction as proposed in FOCAL.
        Clusters L2-normalized forensic embeddings into tampered and pristine groups per image,
        and returns `(logit_full, prob_full, cluster_separability)`.
        """
        logit_full, prob_full, separability, _ = self._cluster_features_with_full_stats(
            feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
        )
        return logit_full, prob_full, separability

    def _cluster_domain_features(
        self,
        feats: torch.Tensor,
        orig_h: int,
        orig_w: int,
        aux_pristine_flags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Domain-split 4-cluster inference collapsed strictly into 2 classes (pristine vs tampered).

        Uses the 4 learned domain-split prototypes (casia_pristine, casia_tampered,
        imd_pristine, imd_tampered) when initialized, collapsing {0, 2} -> pristine
        and {1, 3} -> tampered via exact log-sum-exp, and returns
        `(mask_logits [B,1,H,W], pred_mask [B,1,H,W], separability [B], logits_2 [B,2,H,W])`.
        """
        kmeans_logit, kmeans_prob, separability = self._cluster_features_with_stats(
            feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
        )
        if (
            self.domain_prototypes is None
            or self.domain_clustering_mode == "kmeans"
            or not bool(self.domain_prototypes_initialized.all().item())
        ):
            logits_2_fallback = torch.cat([-0.5 * kmeans_logit, 0.5 * kmeans_logit], dim=1)
            return kmeans_logit, kmeans_prob, separability, logits_2_fallback

        feats_norm = F.normalize(feats.float(), p=2, dim=1)
        proto_norm = F.normalize(self.domain_prototypes.float(), p=2, dim=-1)  # [4, C]
        sim_4 = torch.einsum("bchw,kc->bkhw", feats_norm, proto_norm) / self.cluster_temperature

        pristine_grid = torch.logsumexp(sim_4[:, [0, 2], :, :], dim=1, keepdim=True)
        tampered_grid = torch.logsumexp(sim_4[:, [1, 3], :, :], dim=1, keepdim=True)
        proto_diff_grid = tampered_grid - pristine_grid

        if aux_pristine_flags is not None and aux_pristine_flags.any():
            b, _, h_f, w_f = proto_diff_grid.shape
            gated = self._apply_pristine_gate(
                proto_diff_grid.view(b, h_f * w_f),
                aux_pristine_flags.to(device=proto_diff_grid.device, dtype=torch.bool),
            )
            proto_diff_grid = gated.view(b, 1, h_f, w_f)

        proto_logit_full = F.interpolate(
            proto_diff_grid, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        if self.domain_clustering_mode == "hybrid":
            final_logit = 0.5 * proto_logit_full + 0.5 * kmeans_logit
            logits_2 = torch.cat([-0.5 * final_logit, 0.5 * final_logit], dim=1)
        else:
            final_logit = proto_logit_full
            logits_2_grid = torch.cat([pristine_grid, tampered_grid], dim=1)
            logits_2 = F.interpolate(
                logits_2_grid, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )

        return final_logit, torch.sigmoid(final_logit), separability, logits_2

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

    @staticmethod
    def _sample_multiclass_tokens(
        emb_flat: torch.Tensor,
        label_flat: torch.Tensor,
        max_tokens: int,
        num_classes: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stratified-balanced sampling across up to ``num_classes`` domain-split classes."""
        valid_mask = (label_flat >= 0) & (label_flat < num_classes)
        emb_valid = emb_flat[valid_mask]
        lbl_valid = label_flat[valid_mask]
        n_total = emb_valid.shape[0]
        if n_total <= max_tokens or n_total == 0:
            return emb_valid, lbl_valid

        class_indices = [
            torch.nonzero(lbl_valid == c, as_tuple=False).squeeze(-1)
            for c in range(num_classes)
        ]
        present = [idxs for idxs in class_indices if idxs.numel() > 0]
        if not present:
            return emb_valid[:0], lbl_valid[:0]

        per_cls = max(1, max_tokens // len(present))
        selected = []
        leftover_pools = []
        taken = 0
        for idxs in present:
            k = min(idxs.numel(), per_cls)
            perm = torch.randperm(idxs.numel(), device=emb_valid.device)
            selected.append(idxs[perm[:k]])
            taken += k
            if idxs.numel() > k:
                leftover_pools.append(idxs[perm[k:]])

        if taken < max_tokens and leftover_pools:
            rem = torch.cat(leftover_pools, dim=0)
            extra = min(rem.numel(), max_tokens - taken)
            perm_r = torch.randperm(rem.numel(), device=emb_valid.device)[:extra]
            selected.append(rem[perm_r])

        idx = torch.cat(selected, dim=0)
        return emb_valid[idx], lbl_valid[idx]

    @staticmethod
    def _build_domain_4class_mask_targets(
        gt_masks: torch.Tensor,
        domain_ids: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
        mode: str = "area",
    ) -> torch.Tensor:
        """
        Build 4-class pixel targets (0: casia_pristine, 1: casia_tampered,
        2: imd_pristine, 3: imd_tampered) with ``-100`` for unknown domain samples.
        """
        b = gt_masks.shape[0]
        gt_f = gt_masks.float()
        if gt_f.ndim == 3:
            gt_f = gt_f.unsqueeze(1)
        if gt_f.shape[2] != target_hw[0] or gt_f.shape[3] != target_hw[1]:
            gt_resized = F.interpolate(gt_f, size=target_hw, mode=mode)
        else:
            gt_resized = gt_f
        is_tamp = (gt_resized[:, 0] >= 0.5).long()  # [B, H, W]

        if domain_ids is None:
            return torch.full_like(is_tamp, -100)

        dom = domain_ids.to(device=gt_masks.device, dtype=torch.long).view(b, 1, 1)
        valid = (dom >= 0) & (dom <= 1)
        targets = dom.clamp(min=0, max=1) * 2 + is_tamp
        return torch.where(valid, targets, torch.full_like(targets, -100))

    @staticmethod
    def _info_nce_on_tokens(
        z: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> Optional[torch.Tensor]:
        """Compute InfoNCE across arbitrary integer labels on normalized tokens `[K, C]`."""
        k_n = z.shape[0]
        if k_n < 2:
            return None
        logits = torch.matmul(z, z.t()) / temperature
        self_mask = torch.eye(k_n, dtype=torch.bool, device=z.device)
        same_label = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)
        valid_anchors = same_label.any(dim=1)
        if not valid_anchors.any():
            return None

        logits_masked = logits.masked_fill(self_mask, float("-inf"))
        row_max = logits_masked.max(dim=-1, keepdim=True).values.detach()
        logits_stable = logits_masked - row_max
        log_denom = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
        log_prob = logits_stable - log_denom
        pos_counts = same_label.sum(dim=-1).clamp(min=1).to(log_prob.dtype)
        loss_per_anchor = -(log_prob.masked_fill(~same_label, 0.0)).sum(dim=-1) / pos_counts
        return loss_per_anchor[valid_anchors].mean()

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
        mask_bin = (gt_small >= 0.5).float()
        labels_grid = (mask_bin >= 0.5).long().view(b, h_f * w_f)
        boundary_grid: Optional[torch.Tensor] = None
        if self.contrastive_boundary_ratio > 0.0:
            dilated = F.max_pool2d(mask_bin, kernel_size=3, stride=1, padding=1)
            eroded = -F.max_pool2d(-mask_bin, kernel_size=3, stride=1, padding=1)
            boundary_grid = ((dilated - eroded) > 0.5).view(b, h_f * w_f)

        feats_norm = F.normalize(feats.float(), p=2, dim=1)
        emb_all = feats_norm.permute(0, 2, 3, 1).reshape(b, h_f * w_f, c)

        per_image_losses: List[torch.Tensor] = []
        for i in range(b):
            y_i = labels_grid[i]
            has_pos = bool((y_i == 1).any())
            has_neg = bool((y_i == 0).any())
            if not (has_pos and has_neg):
                continue

            if boundary_grid is not None and boundary_grid[i].any():
                b_mask = boundary_grid[i]
                b_budget = max(2, int(round(self.contrastive_sample_size * self.contrastive_boundary_ratio)))
                z_b, y_b = self._sample_contrastive_tokens(emb_all[i][b_mask], y_i[b_mask], b_budget)
                rem_budget = max(0, self.contrastive_sample_size - z_b.shape[0])
                if rem_budget > 0 and (~b_mask).any():
                    z_r, y_r = self._sample_contrastive_tokens(emb_all[i][~b_mask], y_i[~b_mask], rem_budget)
                    z_i = torch.cat([z_b, z_r], dim=0)
                    y_sub = torch.cat([y_b, y_r], dim=0)
                else:
                    z_i, y_sub = z_b, y_b
            else:
                z_i, y_sub = self._sample_contrastive_tokens(emb_all[i], y_i, self.contrastive_sample_size)
            if z_i.shape[0] < 2 or not (y_sub == 1).any() or not (y_sub == 0).any():
                continue

            loss_i = self._info_nce_on_tokens(z_i, y_sub, self.cluster_temperature)
            if loss_i is not None:
                per_image_losses.append(loss_i)

        if not per_image_losses:
            return zero_anchor
        return torch.stack(per_image_losses).mean()

    def compute_domain_contrastive_loss(
        self,
        feats: torch.Tensor,
        gt_masks: Optional[torch.Tensor],
        domain_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute 4-cluster domain-split contrastive loss:
          1. Image-relative ``MyInfoNCE`` (preserving sharp intra-image boundary separation),
          2. 4-class Prototype InfoNCE against ``domain_prototypes``
             (0: casia_pristine, 1: casia_tampered, 2: imd_pristine, 3: imd_tampered),
          3. Cross-image 4-class SupCon keeping CASIA and IMD clusters separate.
        """
        l_intra = self.compute_contrastive_loss(feats, gt_masks)
        proto_zero = (
            self.domain_prototypes.sum() * 0.0 if self.domain_prototypes is not None else feats.sum() * 0.0
        )
        if gt_masks is None or self.domain_prototypes is None or domain_ids is None:
            return l_intra + proto_zero

        b, c, h_f, w_f = feats.shape
        targets_4 = self._build_domain_4class_mask_targets(
            gt_masks, domain_ids, target_hw=(h_f, w_f), mode="area"
        ).view(-1)

        feats_norm = F.normalize(feats.float(), p=2, dim=1)
        emb_flat = feats_norm.permute(0, 2, 3, 1).reshape(b * h_f * w_f, c)

        # Initialize unseen domain prototypes directly from actual batch centroids on first sight
        with torch.no_grad():
            for k in range(4):
                if not bool(self.domain_prototypes_initialized[k].item()):
                    k_mask = targets_4 == k
                    if k_mask.any():
                        mean_vec = F.normalize(emb_flat[k_mask].mean(dim=0), p=2, dim=0)
                        self.domain_prototypes.data[k].copy_(
                            mean_vec.to(dtype=self.domain_prototypes.dtype, device=self.domain_prototypes.device)
                        )
                        self.domain_prototypes_initialized[k] = True

        z_sub, y_sub = self._sample_multiclass_tokens(
            emb_flat, targets_4, self.contrastive_sample_size, num_classes=4
        )
        if z_sub.shape[0] == 0:
            return l_intra + proto_zero

        total = l_intra
        if self.domain_proto_weight > 0.0:
            proto_norm = F.normalize(self.domain_prototypes.float(), p=2, dim=-1)
            proto_logits = torch.matmul(z_sub, proto_norm.t()) / self.cluster_temperature
            l_proto = F.cross_entropy(proto_logits, y_sub)
            total = total + self.domain_proto_weight * l_proto
        else:
            total = total + proto_zero

        if self.domain_supcon_weight > 0.0 and torch.unique(y_sub).numel() >= 2:
            l_supcon = self._info_nce_on_tokens(z_sub, y_sub, self.cluster_temperature)
            if l_supcon is not None:
                total = total + self.domain_supcon_weight * l_supcon

        return total

    def _run_supervised_head(
        self, feats: torch.Tensor, orig_h: int, orig_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute active supervised segmentation head (`decoder`, `domain_fc`, `fused_fc`, or `fc`).
        When ``domain_split_mode=True``, returns 2-class collapsed `(mask_logits, logits_2, logits_4)`.
        """
        if self.domain_split_mode:
            if self.decoder is not None:
                logits_4 = self.decoder(feats, orig_h, orig_w)
            elif self.domain_fc is not None:
                logits_small = self.domain_fc(feats)
                logits_4 = F.interpolate(
                    logits_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                )
            else:
                raise RuntimeError("domain_split_mode requires 4-channel decoder or domain_fc.")
            mask_logits, logits_2 = _collapse_domain_4_to_2_mask_logits(logits_4)
            return mask_logits, logits_2, logits_4

        if self.decoder is not None:
            mask_logits = self.decoder(feats, orig_h, orig_w)
            return mask_logits, mask_logits, None
        head_conv = self.fused_fc if self.fused_fc is not None else self.fc
        logits_small = head_conv(feats)
        logits_full = F.interpolate(logits_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        mask_logits = logits_full[:, 1:2] - logits_full[:, 0:1]
        return mask_logits, logits_full, None

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        gt_masks: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        dataset_sources: Optional[Sequence[Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]:
        """
        Forward pass of FOCAL.

        Args:
            x: Input RGB tensor `[B, 3, H, W]`.
            return_dict: If True, returns dictionary with standard keys.
            gt_masks: Optional ground-truth binary masks `[B, 1, H, W]` for Mode 1B (`contrastive`)
                and `domain_split_mode` training.
            domain_ids: Optional `[B]` tensor of domain indices (`0`: casia, `1`: imd2020) during
                `domain_split_mode` training (never required at inference).
            dataset_sources: Optional sequence of dataset source strings (`"casia"`, `"imd2020"`).

        Returns:
            Dictionary with strictly 2-class (pristine vs tampered) inference outputs:
                - `mask_logits`: `[B, 1, H, W]`
                - `pred_mask`: Sigmoid probability `[B, 1, H, W]`
                - `features`: Dense feature map `[B, C, H', W']`
                - `class_logits`: `[B, 2]` (or `[B, num_classes]` when not domain-split)
                - `align_loss`: Contrastive / domain-split training loss
                - `cluster_separability`: Inter-centroid distance `1 - cos(c0, c1)` (if clustering)
        """
        orig_h, orig_w = x.shape[2], x.shape[3]
        x_in, f_primary, f_secondary, feats_backbone, feats, vit_taps = self._extract_all_branch_features_with_taps(x)

        preneck_feats: Optional[torch.Tensor] = None
        if self.preneck_projector is not None and vit_taps:
            taps_cat = self._build_preneck_tap_tensor({k: v.detach() for k, v in vit_taps.items()})
            if taps_cat is not None:
                pn_emb = self.preneck_projector(taps_cat)
                if f_secondary is not None:
                    fs_pn = (
                        F.interpolate(f_secondary.detach(), size=pn_emb.shape[2:], mode="bilinear", align_corners=False)
                        if f_secondary.shape[2:] != pn_emb.shape[2:]
                        else f_secondary.detach()
                    )
                    preneck_feats = torch.cat([pn_emb, F.normalize(fs_pn.float(), p=2, dim=1).to(dtype=pn_emb.dtype)], dim=1)
                else:
                    preneck_feats = pn_emb

        resolved_domain_ids = domain_ids
        if resolved_domain_ids is None and dataset_sources is not None:
            resolved_domain_ids = encode_domain_sources(
                dataset_sources, device=x.device, domain_names=self.domain_names
            )
        elif resolved_domain_ids is None and "dataset" in kwargs:
            resolved_domain_ids = encode_domain_sources(
                kwargs["dataset"], device=x.device, domain_names=self.domain_names
            )

        prior_anchor_loss: Optional[torch.Tensor] = None
        prior_anchor_penalty: Optional[torch.Tensor] = None
        if self.training and (
            self.prior_anchor_weight > 0.0
            or self.neck_l2_anchor_weight > 0.0
            or self.cluster_refiner_l2_weight > 0.0
            or self.refiner_ortho_weight > 0.0
            or self.adapter_ortho_weight > 0.0
            or self.preneck_ortho_weight > 0.0
        ):
            prior_anchor_loss, prior_anchor_penalty = self._compute_prior_anchor_terms(
                x_in=x_in, feats=feats, feats_backbone=feats_backbone, f_primary=f_primary
            )

        class_logits_raw = self.class_head(feats) if self.class_head is not None else None
        if self.domain_split_mode and class_logits_raw is not None and class_logits_raw.shape[-1] == 4:
            class_logits = _collapse_domain_4_to_2_class_logits(class_logits_raw)
        else:
            class_logits = class_logits_raw

        aux_pristine_flags = (
            (class_logits[:, 1] < class_logits[:, 0])
            if (self.aux_pristine_gate and class_logits is not None and class_logits.shape[-1] >= 2)
            else None
        )

        align_loss: Optional[torch.Tensor] = None
        skip_mask_loss = False
        separability: Optional[torch.Tensor] = None

        if self.finetune_mode == "supervised":
            # Mode 1A: strict train/eval parity via supervised head (collapsed 4->2 if domain_split_mode)
            mask_logits, logits_full, logits_4 = self._run_supervised_head(feats, orig_h, orig_w)
            pred_mask = torch.sigmoid(mask_logits)
            if self.training and self.domain_split_mode and logits_4 is not None and gt_masks is not None:
                target_4 = self._build_domain_4class_mask_targets(
                    gt_masks, resolved_domain_ids, target_hw=(orig_h, orig_w), mode="nearest"
                )
                if (target_4 != -100).any():
                    domain_ce = F.cross_entropy(logits_4.float(), target_4, ignore_index=-100)
                else:
                    domain_ce = logits_4.sum() * 0.0
                if (
                    class_logits_raw is not None
                    and class_logits_raw.shape[-1] == 4
                    and resolved_domain_ids is not None
                ):
                    gt_f = gt_masks.float()
                    if gt_f.ndim == 3:
                        gt_f = gt_f.unsqueeze(1)
                    img_tamp = (gt_f.amax(dim=(1, 2, 3)) >= 0.5).long()
                    dom = resolved_domain_ids.to(device=gt_masks.device, dtype=torch.long)
                    valid_img = (dom >= 0) & (dom <= 1)
                    img_y4 = torch.where(valid_img, dom.clamp(0, 1) * 2 + img_tamp, torch.full_like(dom, -100))
                    if (img_y4 != -100).any():
                        domain_ce = domain_ce + 0.2 * F.cross_entropy(
                            class_logits_raw.float(), img_y4, ignore_index=-100
                        )
                align_loss = self.domain_loss_weight * domain_ce
                skip_mask_loss = False
        elif self.finetune_mode == "contrastive":
            # Mode 1B: image-relative (or 4-cluster domain-split) contrastive training + 2-class clustering eval
            if self.training:
                if self.domain_split_mode:
                    align_loss = self.compute_domain_contrastive_loss(feats, gt_masks, resolved_domain_ids)
                else:
                    align_loss = self.compute_contrastive_loss(feats, gt_masks)
                skip_mask_loss = True
                with torch.no_grad():
                    if self.domain_split_mode:
                        mask_logits, pred_mask, separability, logits_full = self._cluster_domain_features(
                            feats.detach(), orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                        )
                    else:
                        mask_logits, pred_mask, separability = self._cluster_features_with_stats(
                            feats.detach(), orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                        )
                        logits_full = mask_logits
            else:
                if self.domain_split_mode:
                    mask_logits, pred_mask, separability, logits_full = self._cluster_domain_features(
                        feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                    )
                else:
                    mask_logits, pred_mask, separability = self._cluster_features_with_stats(
                        feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                    )
                    logits_full = mask_logits
        elif self.finetune_mode == "hybrid":
            # Clustering-anchored hybrid: image-relative contrastive features + bounded zero-init residual refiner
            if self.training:
                if self.domain_split_mode:
                    align_loss = self.compute_domain_contrastive_loss(feats, gt_masks, resolved_domain_ids)
                else:
                    align_loss = self.compute_contrastive_loss(feats, gt_masks)
                if preneck_feats is not None and self.preneck_contrastive_weight > 0.0:
                    pn_nce = self.compute_contrastive_loss(preneck_feats, gt_masks)
                    weighted_pn = self.preneck_contrastive_weight * pn_nce
                    align_loss = weighted_pn if align_loss is None else (align_loss + weighted_pn)

            pol_loss_terms: List[torch.Tensor] = []
            affinity_P: Optional[torch.Tensor] = None
            need_affinity = (
                self.feat_smooth_gamma > 0.0
                or self.affinity_diff_beta > 0.0
                or self.refiner_use_affinity_prop
            )
            if need_affinity:
                aff_src = (
                    self._build_multitap_cluster_tensor(
                        f_primary.detach(),
                        f_secondary.detach() if f_secondary is not None else None,
                        {k: v.detach() for k, v in vit_taps.items()},
                    )
                    if vit_taps
                    else feats.detach()
                )
                affinity_P = self._build_affinity_matrix(
                    aff_src,
                    tau=self.affinity_diff_tau,
                    grid_size=self.vit_tap_cluster_size or 48,
                )

            if self.domain_split_mode:
                with torch.no_grad():
                    cluster_logits, _, separability, _ = self._cluster_domain_features(
                        feats.detach(), orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                    )
                    area_fused = (cluster_logits > 0).float().mean(dim=(1, 2, 3))
            else:
                base_cluster_feats = (
                    self._smooth_features_on_graph(feats.detach(), affinity_P, self.feat_smooth_gamma)
                    if (self.feat_smooth_gamma > 0.0 and affinity_P is not None)
                    else feats.detach()
                )
                cluster_logits, _, separability, area_fused, pol_loss_main = self._cluster_features_with_polarity(
                    base_cluster_feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags, gt_masks=gt_masks
                )
                if pol_loss_main is not None:
                    pol_loss_terms.append(pol_loss_main)
                if vit_taps:
                    cluster_logits, separability, area_fused, pol_loss_mt, affinity_P = self._fuse_multitap_clustering(
                        base_logits=cluster_logits,
                        base_sep=separability,
                        base_area=area_fused,
                        f_primary=f_primary,
                        f_secondary=f_secondary,
                        vit_taps=vit_taps,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        aux_pristine_flags=aux_pristine_flags,
                        gt_masks=gt_masks,
                        preneck_feats=preneck_feats,
                        affinity_P=affinity_P,
                    )
                    if pol_loss_mt is not None:
                        pol_loss_terms.append(pol_loss_mt)

            if self.affinity_diff_beta > 0.0 and affinity_P is not None:
                cluster_logits = self._diffuse_logits_on_graph(
                    cluster_logits, affinity_P, beta=self.affinity_diff_beta, steps=self.affinity_diff_steps
                )
            if self.otsu_cal_alpha > 0.0:
                cluster_logits = self._apply_otsu_calibration(cluster_logits, self.otsu_cal_alpha)

            if self.training and pol_loss_terms:
                mean_pol_loss = torch.stack(pol_loss_terms).mean()
                pol_weight = self.polarity_loss_weight if self.polarity_loss_weight > 0.0 else 0.0
                weighted_pol = pol_weight * mean_pol_loss
                align_loss = weighted_pol if align_loss is None else (align_loss + weighted_pol)

            with torch.no_grad():
                primary_branch_stats = None
                secondary_branch_stats = None
                if (
                    self.secondary_net is not None
                    and f_primary is not None
                    and f_secondary is not None
                    and (self.cluster_refiner_type == "image_relative" or self.branch_logit_weights is not None)
                ):
                    c_p_full, _, sep_p, area_p = self._cluster_features_with_full_stats(
                        f_primary.detach(), orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                    )
                    c_s_full, _, sep_s, area_s = self._cluster_features_with_full_stats(
                        f_secondary.detach(), orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                    )
                    primary_branch_stats = (f_primary.detach(), c_p_full, sep_p, area_p)
                    secondary_branch_stats = (f_secondary.detach(), c_s_full, sep_s, area_s)

            base_logits = cluster_logits
            if self.branch_logit_weights is not None and primary_branch_stats is not None and secondary_branch_stats is not None:
                w_b = 0.5 * torch.tanh(self.branch_logit_weights.float())
                base_logits = (
                    cluster_logits.float()
                    + w_b[0] * primary_branch_stats[1].float()
                    + w_b[1] * secondary_branch_stats[1].float()
                )

            if self.cluster_refiner is not None:
                if isinstance(self.cluster_refiner, ImageRelativeClusterRefiner):
                    x_raw_img = self._ensure_raw_rgb(x) if self.cluster_refiner.use_srm_desc else None
                    desc = ImageRelativeClusterRefiner.build_descriptors(
                        feats=feats.detach(),
                        c_fused_logits=cluster_logits.detach(),
                        sep_fused=separability.detach(),
                        area_fused=area_fused.detach(),
                        primary_branch=primary_branch_stats,
                        secondary_branch=secondary_branch_stats,
                        multi_branch=self.cluster_refiner.multi_branch,
                        srm_kernels=getattr(self.cluster_refiner, "_srm_kernels", None),
                        raw_rgb=x_raw_img,
                        use_srm_desc=self.cluster_refiner.use_srm_desc,
                        use_border_knn_desc=self.cluster_refiner.use_border_knn_desc,
                    )
                    raw_bias, raw_scale = self.cluster_refiner(desc, orig_h, orig_w, affinity_P=affinity_P)
                    sep_gate: Union[float, torch.Tensor] = 1.0
                    if self.refiner_sep_gate_tau > 0.0:
                        sep_gate = (
                            (separability.detach() / self.refiner_sep_gate_tau)
                            .clamp(0.0, 1.0)
                            .view(-1, 1, 1, 1)
                            .to(dtype=raw_bias.dtype)
                        )
                    scale = 1.0 + sep_gate * self.cluster_residual_scale_gain * torch.tanh(raw_scale)
                    if self.cluster_residual_max_scale > 0.0:
                        bias = sep_gate * self.cluster_residual_max_scale * torch.tanh(raw_bias)
                    else:
                        bias = sep_gate * raw_bias
                    if self.training and self.refiner_zero_mean_bias_weight > 0.0:
                        dc_bias_pen = self.refiner_zero_mean_bias_weight * bias.float().mean(dim=(2, 3)).pow(2).mean()
                        prior_anchor_penalty = (
                            dc_bias_pen if prior_anchor_penalty is None else (prior_anchor_penalty + dc_bias_pen)
                        )
                    mask_logits = scale * base_logits.to(dtype=bias.dtype) + bias
                else:
                    ref_feats = feats.detach() if self.detach_refiner_feats else feats
                    raw_delta = self.cluster_refiner(ref_feats, cluster_logits.detach(), orig_h, orig_w)
                    if self.cluster_residual_max_scale > 0.0:
                        delta = self.cluster_residual_max_scale * torch.tanh(raw_delta)
                    else:
                        delta = raw_delta
                    mask_logits = base_logits.to(dtype=delta.dtype) + delta
            else:
                mask_logits = base_logits.to(dtype=feats.dtype)
            logits_full = mask_logits
            pred_mask = torch.sigmoid(mask_logits)
            skip_mask_loss = False
        else:
            # Legacy mode: preserves backward compatibility with existing zero-shot / audit tests
            if not self.training and self.use_clustering:
                if self.domain_split_mode:
                    mask_logits, pred_mask, separability, logits_full = self._cluster_domain_features(
                        feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                    )
                else:
                    affinity_P = None
                    need_affinity = self.feat_smooth_gamma > 0.0 or self.affinity_diff_beta > 0.0
                    if need_affinity:
                        aff_src = (
                            self._build_multitap_cluster_tensor(
                                f_primary,
                                f_secondary,
                                vit_taps,
                            )
                            if vit_taps
                            else feats
                        )
                        affinity_P = self._build_affinity_matrix(
                            aff_src,
                            tau=self.affinity_diff_tau,
                            grid_size=self.vit_tap_cluster_size or 48,
                        )
                    base_cluster_feats = (
                        self._smooth_features_on_graph(feats, affinity_P, self.feat_smooth_gamma)
                        if (self.feat_smooth_gamma > 0.0 and affinity_P is not None)
                        else feats
                    )
                    mask_logits, pred_mask, separability, area_fused, _ = self._cluster_features_with_polarity(
                        base_cluster_feats, orig_h, orig_w, aux_pristine_flags=aux_pristine_flags
                    )
                    if vit_taps:
                        mask_logits, separability, _, _, affinity_P = self._fuse_multitap_clustering(
                            base_logits=mask_logits,
                            base_sep=separability,
                            base_area=area_fused,
                            f_primary=f_primary,
                            f_secondary=f_secondary,
                            vit_taps=vit_taps,
                            orig_h=orig_h,
                            orig_w=orig_w,
                            aux_pristine_flags=aux_pristine_flags,
                            preneck_feats=preneck_feats,
                            affinity_P=affinity_P,
                        )
                    if self.affinity_diff_beta > 0.0 and affinity_P is not None:
                        mask_logits = self._diffuse_logits_on_graph(
                            mask_logits, affinity_P, beta=self.affinity_diff_beta, steps=self.affinity_diff_steps
                        )
                    if self.otsu_cal_alpha > 0.0:
                        mask_logits = self._apply_otsu_calibration(mask_logits, self.otsu_cal_alpha)
                    pred_mask = torch.sigmoid(mask_logits)
                    logits_full = mask_logits
            else:
                mask_logits, logits_full, _ = self._run_supervised_head(feats, orig_h, orig_w)
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
