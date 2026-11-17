"""
MSCCNet: Multi-spectral Class Center Network for Face Manipulation Localization.
Official Paper: Miao et al., "Multi-spectral Class Center Network for Face Manipulation
Localization", ACM Transactions on Multimedia Computing, Communications, and Applications
(ACM TOMM), 2025 (arXiv:2305.10794, DOI: 10.1145/3747296).
Official Repository: https://github.com/miaoct/MSCCNet (MIT License).

Architecture Summary:
- Multi-scale ResNet backbone (outstride=8 with 3x3 stem by default) extracting stages 2, 3, 4.
- Image-level classification branch (`classifer`) predicting real/fake manipulation logits.
- Multi-level feature fusion bottlenecks (`bottleneck1`, `bottleneck2`, `bottleneck3`,
  `bottleneck_fusion`).
- Multi-Spectral Class Center (`MSCC`) module combining:
  - Fixed 2D DCT basis frequency decomposition (`MultiSpectralDCTLayer`),
  - Frequency-residual sub-decoder (`decoder_map`) & spatial fusion (`fuse`),
  - Graph Convolutional Network (`GCN`) across spectral class centers and channels,
  - Soft-attention reconstruction over spectral class centers + residual fusion.
- Pixel-level manipulation localization decoder (`decoder`).

Note on pretrained weights:
The official `miaoct/MSCCNet` repository does not publish task-trained MSCCNet checkpoints;
training uses ImageNet-pretrained ResNet backbones (`resnet50_v1c` / `resnet101_v1c` or
`resnet50` / `resnet101`). This integration supports loading an explicit MSCCNet checkpoint
(`checkpoint_path`) when available as well as ImageNet-pretrained backbone initialization.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

logger = logging.getLogger("sid_unet.models.msccnet")

MSCC_BACKBONE_URLS: Dict[str, str] = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnet18stem": "https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth",
    "resnet50stem": "https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth",
    "resnet101stem": "https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth",
}


def _make_norm2d(channels: int, use_sync_bn: bool = False) -> nn.Module:
    if use_sync_bn and torch.distributed.is_available() and torch.distributed.is_initialized():
        return nn.SyncBatchNorm(channels)
    return nn.BatchNorm2d(channels)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        use_sync_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = _make_norm2d(planes, use_sync_bn=use_sync_bn)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _make_norm2d(planes, use_sync_bn=use_sync_bn)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        use_sync_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = _make_norm2d(planes, use_sync_bn=use_sync_bn)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = _make_norm2d(planes, use_sync_bn=use_sync_bn)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = _make_norm2d(planes * self.expansion, use_sync_bn=use_sync_bn)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class MSCCResNet(nn.Module):
    """Dilated ResNet backbone with optional 3x3 deep stem matching miaoct/MSCCNet."""

    arch_settings = {
        18: (_BasicBlock, (2, 2, 2, 2)),
        34: (_BasicBlock, (3, 4, 6, 3)),
        50: (_Bottleneck, (3, 4, 6, 3)),
        101: (_Bottleneck, (3, 4, 23, 3)),
        152: (_Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        stem_channels: int = 64,
        depth: int = 50,
        outstride: int = 8,
        contract_dilation: bool = True,
        use_stem: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        use_avg_for_downsample: bool = False,
        use_sync_bn: bool = False,
    ) -> None:
        super().__init__()
        if depth not in self.arch_settings:
            raise ValueError(f"Unsupported ResNet depth: {depth}. Supported: {sorted(self.arch_settings.keys())}")
        outstride_map = {
            8: ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        if outstride not in outstride_map:
            raise ValueError(f"Unsupported outstride: {outstride}. Supported: {sorted(outstride_map.keys())}")

        self.inplanes = stem_channels
        self.out_indices = tuple(out_indices)
        self.use_stem = use_stem
        self.depth = depth
        self.use_sync_bn = use_sync_bn

        block, num_blocks_list = self.arch_settings[depth]
        stride_list, dilation_list = outstride_map[outstride]

        if use_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
                _make_norm2d(stem_channels // 2, use_sync_bn=use_sync_bn),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_channels // 2, stem_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
                _make_norm2d(stem_channels // 2, use_sync_bn=use_sync_bn),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                _make_norm2d(stem_channels, use_sync_bn=use_sync_bn),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = _make_norm2d(stem_channels, use_sync_bn=use_sync_bn)
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        mult = 4 if depth >= 50 else 1
        self.layer1 = self._make_layer(
            block, stem_channels, base_channels, num_blocks_list[0],
            stride_list[0], dilation_list[0], contract_dilation, use_avg_for_downsample,
        )
        self.layer2 = self._make_layer(
            block, base_channels * mult, base_channels * 2, num_blocks_list[1],
            stride_list[1], dilation_list[1], contract_dilation, use_avg_for_downsample,
        )
        self.layer3 = self._make_layer(
            block, base_channels * 2 * mult, base_channels * 4, num_blocks_list[2],
            stride_list[2], dilation_list[2], contract_dilation, use_avg_for_downsample,
        )
        self.layer4 = self._make_layer(
            block, base_channels * 4 * mult, base_channels * 8, num_blocks_list[3],
            stride_list[3], dilation_list[3], contract_dilation, use_avg_for_downsample,
        )

    def _make_layer(
        self,
        block: Any,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        contract_dilation: bool,
        use_avg_for_downsample: bool,
    ) -> nn.Sequential:
        downsample = None
        dilations = [dilation] * num_blocks
        if contract_dilation and dilation > 1:
            dilations[0] = dilation // 2
        if stride != 1 or inplanes != planes * block.expansion:
            if use_avg_for_downsample:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                    _make_norm2d(planes * block.expansion, use_sync_bn=self.use_sync_bn),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                    _make_norm2d(planes * block.expansion, use_sync_bn=self.use_sync_bn),
                )
        layers = [
            block(inplanes, planes, stride=stride, dilation=dilations[0], downsample=downsample, use_sync_bn=self.use_sync_bn)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(planes * block.expansion, planes, stride=1, dilation=dilations[i], use_sync_bn=self.use_sync_bn)
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.use_stem:
            x = self.stem(x)
        else:
            x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        all_outs = (x1, x2, x3, x4)
        return tuple(all_outs[i] for i in self.out_indices)


class MSCCClassifier(nn.Module):
    """Image-level real/fake classification branch from MSCCNet."""

    def __init__(self, inplanes: int, num_classes: int = 2) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _get_freq_indices(k: int = 8) -> List[Tuple[int, int]]:
    assert k % 2 == 0, "k must be an even number"
    return [(x, y) for x in range(k) for y in range(k)]


def _build_dct_filter_bank(tile_h: int, tile_w: int, uv_xy: List[Tuple[int, int]], channel: int) -> torch.Tensor:
    """Vectorized construction of 2D DCT basis filter bank [channel, tile_h, tile_w]."""
    c_part = channel // len(uv_xy)
    dct_filter = torch.zeros(channel, tile_h, tile_w, dtype=torch.float32)

    tx = torch.arange(tile_h, dtype=torch.float32)
    ty = torch.arange(tile_w, dtype=torch.float32)

    for i, (u_x, v_y) in enumerate(uv_xy):
        bx = torch.cos(math.pi * u_x * (tx + 0.5) / tile_h) / math.sqrt(tile_h)
        if u_x != 0:
            bx = bx * math.sqrt(2.0)
        by = torch.cos(math.pi * v_y * (ty + 0.5) / tile_w) / math.sqrt(tile_w)
        if v_y != 0:
            by = by * math.sqrt(2.0)
        basis = torch.outer(bx, by)
        dct_filter[i * c_part : (i + 1) * c_part] = basis.unsqueeze(0)

    return dct_filter


class MultiSpectralDCTLayer(nn.Module):
    """2D Discrete Cosine Transform frequency decomposition layer."""

    def __init__(self, height: int, width: int, channel: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.channel = channel
        uv_xy = _get_freq_indices(k=k)
        self.register_buffer("weight", _build_dct_filter_bank(height, width, uv_xy, channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {tuple(x.shape)}")
        h, w = x.shape[2], x.shape[3]
        weight = self.weight
        if weight.shape[1] != h or weight.shape[2] != w:
            weight = F.interpolate(
                weight.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
        return x * weight.to(dtype=x.dtype, device=x.device)


class MSCCGCN(nn.Module):
    """Graph Convolutional Network over spectral group class centers."""

    def __init__(self, num_node: int, num_channel: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out + x)
        return self.conv2(out)


def _channel_split(x: torch.Tensor, num: int) -> torch.Tensor:
    b, c, h, w = x.size()
    c_per = c // num
    out = x.view(b, num, c_per, h, w)
    return out.permute(0, 1, 3, 4, 2).contiguous()


def _channel_recover(x: torch.Tensor, num: int) -> torch.Tensor:
    b, n, h, w, c_per = x.size()
    out = x.permute(0, 1, 4, 2, 3).contiguous()
    return out.view(b, num * c_per, h, w)


class MSCCModule(nn.Module):
    """Multi-Spectral Class Center (MSCC) attention and GCN refinement module."""

    def __init__(
        self,
        num_classes: int,
        feats_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        height: int = 64,
        width: int = 64,
        k: int = 2,
        dct_k: int = 8,
        use_sync_bn: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.out_channels = out_channels
        self.num = k * k
        if feats_channels % self.num != 0:
            raise ValueError(f"feats_channels ({feats_channels}) must be divisible by k*k ({self.num})")

        c_per = feats_channels // self.num
        self.frequency = MultiSpectralDCTLayer(height, width, feats_channels, k=dct_k)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm2d(out_channels, use_sync_bn=use_sync_bn),
            nn.ReLU(inplace=True),
        )
        self.decoder_map = nn.Sequential(
            nn.Conv2d(c_per, c_per, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm2d(c_per, use_sync_bn=use_sync_bn),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_per, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.fuse = nn.Conv2d(self.num, 1, kernel_size=1)
        self.gcn = MSCCGCN(self.num, c_per)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        b, _, h, w = x.size()
        fres = self.frequency(x)

        x_split = _channel_split(x, num=self.num)
        fres_split = _channel_split(fres, num=self.num)
        n = x_split.shape[1]
        c_per = x_split.shape[-1]

        fres_flat = fres_split.view(b * n, h, w, c_per).permute(0, 3, 1, 2).contiguous()
        fres_map = self.decoder_map(fres_flat)

        fres_map_output = self.fuse(fres_map.view(b, n, self.num_classes, h * w))
        fres_map_output = fres_map_output.squeeze(1).view(b, self.num_classes, h, w)

        x_tok = x_split.view(b, n, h * w, c_per)
        fres_tok = fres_split.view(b, n, h * w, c_per)
        fres_map_tok = fres_map.view(b, n, self.num_classes, h * w)

        local_fre = torch.matmul(fres_map_tok, fres_tok)
        local_fre = self.gcn(local_fre)

        weight_map = F.softmax(torch.matmul(x_tok, local_fre.transpose(2, 3)), dim=-1)
        out = torch.matmul(weight_map, local_fre).view(b, n, h, w, c_per)
        out = _channel_recover(out, num=self.num)
        out = self.bottleneck(torch.cat([residual, out], dim=1))
        return out, fres_map_output


class MSCCNet(nn.Module):
    """
    MSCCNet: Multi-spectral Class Center Network for Face Manipulation Detection and Localization.

    Supports both:
    - Image-level binary/multi-class manipulation classification (`class_logits`).
    - Pixel-level manipulation localization (`mask_logits` + auxiliary spectral map `fres_map_logits`).

    Conforms to SID-UNet's unified `(mask_logits, class_logits)` and `return_dict=True` contracts.
    """

    def __init__(
        self,
        backbone_depth: int = 50,
        backbone_outstride: int = 8,
        use_stem: bool = True,
        pretrained: bool = True,
        backbone_pretrained: Optional[bool] = None,
        checkpoint_path: Optional[str] = None,
        hf_repo: Optional[str] = None,
        allow_random_init: bool = False,
        num_classes: int = 2,
        feats_channels: int = 512,
        out_channels: int = 512,
        dropout: float = 0.1,
        spectral_height: int = 64,
        spectral_width: int = 64,
        spectral_k: int = 2,
        aux_classifier: bool = True,
        align_corners: bool = False,
        use_sync_bn: bool = False,
        in_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone_depth = int(backbone_depth)
        self.num_classes = int(num_classes)
        self.aux_classifier = bool(aux_classifier)
        self.align_corners = bool(align_corners)
        self.use_stem = bool(use_stem)

        self.backbone_net = MSCCResNet(
            in_channels=int(in_channels),
            depth=self.backbone_depth,
            outstride=int(backbone_outstride),
            use_stem=self.use_stem,
            out_indices=(0, 1, 2, 3),
            use_sync_bn=use_sync_bn,
        )

        # Stage channels for ResNet50/101/152: [256, 512, 1024, 2048]; for ResNet18/34: [64, 128, 256, 512]
        stage_mult = 4 if self.backbone_depth >= 50 else 1
        c2, c3, c4 = 128 * stage_mult, 256 * stage_mult, 512 * stage_mult

        # Attribute name `classifer` matches upstream miaoct/MSCCNet checkpoints.
        self.classifer = MSCCClassifier(c4, num_classes=self.num_classes)

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(c2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm2d(feats_channels, use_sync_bn=use_sync_bn),
            nn.ReLU(inplace=True),
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(c3, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm2d(feats_channels, use_sync_bn=use_sync_bn),
            nn.ReLU(inplace=True),
        )
        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(c4, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm2d(feats_channels, use_sync_bn=use_sync_bn),
            nn.ReLU(inplace=True),
        )
        self.mscc = MSCCModule(
            num_classes=self.num_classes,
            feats_channels=feats_channels,
            out_channels=out_channels,
            dropout=dropout,
            height=spectral_height,
            width=spectral_width,
            k=spectral_k,
            use_sync_bn=use_sync_bn,
        )
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(feats_channels * 3, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm2d(feats_channels, use_sync_bn=use_sync_bn),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm2d(out_channels, use_sync_bn=use_sync_bn),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, self.num_classes, kernel_size=1, stride=1, padding=0),
        )

        if not pretrained and not allow_random_init:
            raise ValueError(
                "MSCCNet requires pretrained=True (or explicit allow_random_init=True for training from scratch). "
                "Random weight initialization is strictly disabled."
            )

        if checkpoint_path:
            self.load_msccnet_checkpoint(checkpoint_path)
        elif hf_repo:
            self._load_hf_checkpoint(hf_repo)
        elif pretrained and not allow_random_init:
            raise RuntimeError(
                "MSCCNet strict zero-random-init policy: official upstream miaoct/MSCCNet only provides "
                "ImageNet ResNet backbones and did not release task-trained MSCCNet weights (epoch_best.pth). "
                "Provide a full MSCCNet checkpoint via `checkpoint_path` or `hf_repo`, or pass "
                "`allow_random_init=True` explicitly when training from an ImageNet-pretrained backbone."
            )
        else:
            load_bb = pretrained if backbone_pretrained is None else bool(backbone_pretrained)
            if load_bb:
                self._load_imagenet_backbone()

    def _load_hf_checkpoint(self, hf_repo: str) -> None:
        from huggingface_hub import hf_hub_download
        last_err = None
        for fname in ["checkpoints/checkpoint_best.pt", "checkpoint_best.pt", "epoch_best.pth", "checkpoints/epoch_best.pth"]:
            try:
                p = hf_hub_download(repo_id=hf_repo, filename=fname)
                if p and os.path.isfile(p):
                    self.load_msccnet_checkpoint(p)
                    return
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"Failed to download MSCCNet checkpoint from HF repo '{hf_repo}': {last_err}") from last_err

    @property
    def classifier(self) -> MSCCClassifier:
        return self.classifer

    def _load_imagenet_backbone(self) -> None:
        key = f"resnet{self.backbone_depth}" + ("stem" if self.use_stem else "")
        url = MSCC_BACKBONE_URLS.get(key)
        if not url:
            logger.warning(f"No ImageNet URL registered for MSCCNet backbone '{key}'; skipping backbone init.")
            return
        try:
            ckpt = model_zoo.load_url(url, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self.backbone_net.load_state_dict(sd, strict=False)
            logger.info(
                f"Loaded ImageNet-pretrained MSCCNet backbone '{key}' from {url} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        except Exception as exc:
            logger.warning(f"Could not download ImageNet backbone '{key}' ({url}): {exc}")

    def load_msccnet_checkpoint(self, checkpoint_path: str) -> None:
        """Load full MSCCNet checkpoint (supports SID-UNet and upstream miaoct/MSCCNet formats)."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"MSCCNet checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            raw_sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt.get("state_dict") or ckpt
        else:
            raise ValueError(f"Unsupported MSCCNet checkpoint payload type: {type(ckpt)}")
        clean_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in raw_sd.items()}
        missing, unexpected = self.load_state_dict(clean_sd, strict=False)
        logger.info(
            f"Loaded MSCCNet checkpoint from {checkpoint_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    @staticmethod
    def _to_binary_mask_logit(seg_logits: torch.Tensor) -> torch.Tensor:
        if seg_logits.shape[1] == 1:
            return seg_logits
        if seg_logits.shape[1] == 2:
            return seg_logits[:, 1:2] - seg_logits[:, 0:1]
        # Multi-class: foreground (non-zero classes) vs background class 0
        fg = torch.logsumexp(seg_logits[:, 1:], dim=1, keepdim=True)
        return fg - seg_logits[:, 0:1]

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        orig_h, orig_w = x.shape[2], x.shape[3]
        backbone_outputs = self.backbone_net(x)
        predictions_img = self.classifer(backbone_outputs[-1])

        x1 = self.bottleneck1(backbone_outputs[1])
        x2 = self.bottleneck2(backbone_outputs[2])
        x3 = self.bottleneck3(backbone_outputs[3])

        # Align spatial sizes if outstride != 8
        if x2.shape[2:] != x1.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=self.align_corners)
        if x3.shape[2:] != x1.shape[2:]:
            x3 = F.interpolate(x3, size=x1.shape[2:], mode="bilinear", align_corners=self.align_corners)

        x123 = self.bottleneck_fusion(torch.cat((x1, x2, x3), dim=1))
        mscc_feat, fres_map_small = self.mscc(x123)
        seg_small = self.decoder(mscc_feat)

        seg_full = F.interpolate(seg_small, size=(orig_h, orig_w), mode="bilinear", align_corners=self.align_corners)
        fres_full = F.interpolate(
            fres_map_small, size=(orig_h, orig_w), mode="bilinear", align_corners=self.align_corners
        )

        mask_logits = self._to_binary_mask_logit(seg_full)
        fres_map_logits = self._to_binary_mask_logit(fres_full)

        if return_dict:
            return {
                "mask_logits": mask_logits,
                "pred_mask": torch.sigmoid(mask_logits),
                "class_logits": predictions_img,
                "pred_label_probs": torch.softmax(predictions_img, dim=-1),
                "fres_map_logits": fres_map_logits,
                "fres_map_prob": torch.sigmoid(fres_map_logits),
                "raw_seg": seg_full,
                "raw_fres_map": fres_full,
            }

        if self.aux_classifier:
            return mask_logits, predictions_img
        return mask_logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_dict=True)
            return (out["pred_mask"] >= threshold).float()
