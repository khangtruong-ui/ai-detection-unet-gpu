"""
DetectiveSAMv2: Evidence-Driven Spatial Cross-Attention Forgery Localization model.

Integrates the DetectiveSAMv2 architecture and pretrained weights from
`Gertlek/DetectiveSAMv2` (built on SAM2.1 Hiera Base+) into `sid_unet`:
  - Three perturbation streams by default: Gaussian blur, JPEG compression, Gaussian noise.
  - StreamEvidenceBuilder: multi-scale residual & absolute difference evidence projector.
  - SpatialCrossAttentionSharedAdapter: global low-res and windowed mid-res cross-attention
    adapters with gated residual deltas.
  - TransformerEvidenceMaskAdapter: hierarchical 2D transformer prompt generator.
  - Frozen SAM2.1 Hiera Base+ encoder, prompt encoder, and mask decoder.

Supports optional L2-SP starting-point anchor regularization and selective adapter
freezing for low-forgetting domain adaptation (e.g. CASIA v2.0).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sid_unet.models.blocks import get_submodule_device_dtype

logger = logging.getLogger("sid_unet.models.detective_sam_v2")

DEFAULT_DETECTIVE_SAM_V2_REPO = "Gertlek/DetectiveSAMv2"
DEFAULT_V2_PERTURBATIONS = "gaussian_blur+jpeg_compression+gaussian_noise"
MODEL_CHANNELS = [256, 32, 64]
MASK_ADAPTER_RESOLUTION = (128, 128)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def parse_perturbation_types(perturbation_type: str) -> List[str]:
    """Parse `+` or `/` separated perturbation stream names."""
    p = str(perturbation_type or "").strip()
    if not p or p.lower() == "none":
        return []
    if "+" in p:
        return [item.strip() for item in p.split("+") if item.strip()]
    if "/" in p:
        return [item.strip() for item in p.split("/") if item.strip()]
    return [p]


def compute_perturbation_params(perturbation_intensity: float) -> Dict[str, Union[float, int]]:
    """Map perturbation intensity to blur sigma, JPEG quality, and Gaussian noise std."""
    intensity = float(perturbation_intensity)
    return {
        "blur_sigma": intensity * 2.0,
        "jpeg_quality": max(10, int(95 - (intensity * 56.67))),
        "noise_std": intensity * 0.2,
    }


def _create_spatial_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    x_pos, y_pos = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(x_pos**2 + y_pos**2) / (2.0 * max(sigma, 1e-6) ** 2))
    gaussian = gaussian / gaussian.sum()
    return gaussian.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1).to(dtype=dtype)


def _apply_gaussian_blur_batch(x_01: torch.Tensor, sigma: float, kernel_size: int = 5) -> torch.Tensor:
    if sigma <= 0:
        return x_01.clone()
    channels = x_01.shape[1]
    kernel = _create_spatial_gaussian_kernel(kernel_size, sigma, channels, x_01.device, x_01.dtype)
    return F.conv2d(x_01, kernel, padding=kernel_size // 2, groups=channels)


def _apply_jpeg_compression_batch(x_01: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply JPEG recompression to [B, 3, H, W] tensor in [0, 1], preserving device/dtype."""
    device = x_01.device
    dtype = x_01.dtype
    x_cpu = x_01.detach().clamp(0.0, 1.0)
    use_tvio = False
    try:
        import torchvision.io as tvio
        use_tvio = hasattr(tvio, "encode_jpeg") and hasattr(tvio, "decode_jpeg")
    except Exception:
        tvio = None  # type: ignore

    out_list: List[torch.Tensor] = []
    if use_tvio and tvio is not None:
        u8_batch = (x_cpu * 255.0).round().to(torch.uint8).cpu()
        for idx in range(u8_batch.shape[0]):
            encoded = tvio.encode_jpeg(u8_batch[idx], quality=int(quality))
            decoded = tvio.decode_jpeg(encoded).float() / 255.0
            out_list.append(decoded)
    else:
        import torchvision.transforms.functional as TF
        x_cpu_f = x_cpu.float().cpu()
        for idx in range(x_cpu_f.shape[0]):
            pil_img = TF.to_pil_image(x_cpu_f[idx])
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=int(quality))
            buf.seek(0)
            with Image.open(buf) as dec:
                out_list.append(TF.to_tensor(dec.convert("RGB")))

    return torch.stack(out_list, dim=0).to(device=device, dtype=dtype)


def _apply_gaussian_noise_batch(
    x_01: torch.Tensor,
    std: float,
    deterministic: bool = False,
    seed: int = 42,
) -> torch.Tensor:
    if std <= 0:
        return x_01.clone()
    if deterministic:
        gen = torch.Generator(device=x_01.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(x_01.shape, generator=gen, device=x_01.device, dtype=x_01.dtype) * std
    else:
        noise = torch.randn_like(x_01) * std
    return torch.clamp(x_01 + noise, 0.0, 1.0)


class FlexibleEvidenceBlock(nn.Module):
    """Residual depthwise-separable evidence refinement block with channel gating."""

    def __init__(self, channels: int, dropout_rate: float = 0.1, expansion: int = 2) -> None:
        super().__init__()
        expanded = channels * expansion
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.expand = nn.Conv2d(channels, expanded, kernel_size=1)
        self.depthwise = nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded)
        self.act = nn.GELU()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded, max(channels // 4, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 8), expanded, kernel_size=1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(expanded, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.norm(x)
        hidden = self.expand(hidden)
        hidden = self.depthwise(hidden)
        hidden = self.act(hidden)
        hidden = hidden * self.channel_gate(hidden)
        hidden = self.project(hidden)
        hidden = self.dropout(hidden)
        return residual + hidden


class FlexibleEvidenceProjector(nn.Module):
    """Channel and local-spatial evidence projector used by DetectiveSAMv2 adapters."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
        )
        self.blocks = nn.Sequential(
            *[
                FlexibleEvidenceBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input_proj(x))


class StreamEvidenceBuilder(nn.Module):
    """Build multi-scale forensic evidence from target and perturbation-stream features."""

    def __init__(
        self,
        in_channels_list: List[int],
        hidden_dim: int,
        dropout_rate: float = 0.1,
        max_streams: int = 3,
    ) -> None:
        super().__init__()
        self.max_streams = max(1, max_streams)
        self.projections = nn.ModuleList(
            [
                FlexibleEvidenceProjector(
                    in_channels=channels * (1 + 3 * self.max_streams),
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    depth=3,
                )
                for channels in in_channels_list
            ]
        )

    def forward(
        self,
        unadapted: List[torch.Tensor],
        streams_unadapted: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        evidence = []
        for scale_idx, (target, streams) in enumerate(zip(unadapted, streams_unadapted)):
            fixed_streams = list(streams[: self.max_streams])
            while len(fixed_streams) < self.max_streams:
                fixed_streams.append(torch.zeros_like(target))

            abs_diffs = [(target - stream).abs() for stream in fixed_streams]
            signed_diffs = [target - stream for stream in fixed_streams]
            raw = torch.cat([target, *fixed_streams, *abs_diffs, *signed_diffs], dim=1)
            evidence.append(self.projections[scale_idx](raw))
        return evidence


class CrossAttentionSharedAdapter(nn.Module):
    """Per-scale cross-attention adapter over perturbation streams."""

    def __init__(
        self,
        in_channels_list: List[int],
        hidden_dim: int,
        dropout_rate: float = 0.1,
        max_streams: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_scales = len(in_channels_list)
        self.max_streams = max(1, max_streams)
        self.num_heads = max(1, num_heads)
        if hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({self.num_heads})")

        self.head_dim = hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.ModuleList(nn.Conv2d(channels, hidden_dim, kernel_size=1) for channels in in_channels_list)
        self.k_proj = nn.ModuleList(nn.Conv2d(channels, hidden_dim, kernel_size=1) for channels in in_channels_list)
        self.v_proj = nn.ModuleList(nn.Conv2d(channels, hidden_dim, kernel_size=1) for channels in in_channels_list)
        self.out_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout2d(p=dropout_rate),
                    nn.Conv2d(hidden_dim, channels, kernel_size=1),
                )
                for channels in in_channels_list
            ]
        )
        self.evidence_q_proj = nn.ModuleList(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            for _ in in_channels_list
        )
        self.evidence_context_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout2d(p=dropout_rate),
                )
                for _ in in_channels_list
            ]
        )
        self.residual_gate = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels + hidden_dim, channels, kernel_size=1),
                    nn.Sigmoid(),
                )
                for channels in in_channels_list
            ]
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(
        self,
        stream_features: List[torch.Tensor],
        unadapted: torch.Tensor,
        scale_idx: int,
        return_delta: bool = False,
        evidence: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not stream_features:
            delta = torch.zeros_like(unadapted)
            if return_delta:
                return unadapted, delta
            return unadapted

        batch_size, _, height, width = unadapted.shape
        stream_count = len(stream_features)
        query = self.q_proj[scale_idx](unadapted)
        if evidence is not None:
            if evidence.shape[-2:] != query.shape[-2:]:
                evidence = F.interpolate(evidence, size=query.shape[-2:], mode="bilinear", align_corners=False)
            query = query + self.evidence_q_proj[scale_idx](evidence)
        query = query.view(batch_size, self.num_heads, self.head_dim, 1, height, width)

        keys = [self.k_proj[scale_idx](feature) for feature in stream_features]
        values = [self.v_proj[scale_idx](feature) for feature in stream_features]
        key = torch.stack(keys, dim=1).view(batch_size, stream_count, self.num_heads, self.head_dim, height, width)
        value = torch.stack(values, dim=1).view(batch_size, stream_count, self.num_heads, self.head_dim, height, width)
        key = key.permute(0, 2, 3, 1, 4, 5)
        value = value.permute(0, 2, 3, 1, 4, 5)

        attention_logits = (query * key).sum(dim=2) * self.scale
        attention = F.softmax(attention_logits, dim=2)
        context = (attention.unsqueeze(2) * value).sum(dim=3)
        context = context.reshape(batch_size, self.num_heads * self.head_dim, height, width)
        if evidence is not None:
            context = context + self.evidence_context_proj[scale_idx](evidence)
        context = self.dropout(context)

        raw_delta = self.out_proj[scale_idx](context)
        gate = self.residual_gate[scale_idx](torch.cat([unadapted, context], dim=1))
        delta = gate * raw_delta
        adapted = unadapted + delta
        if return_delta:
            return adapted, delta
        return adapted


class SpatialCrossAttentionSharedAdapter(CrossAttentionSharedAdapter):
    """Cross-attention adapter with global low-res and windowed mid-res spatial mixing."""

    def __init__(
        self,
        in_channels_list: List[int],
        hidden_dim: int,
        dropout_rate: float = 0.1,
        max_streams: int = 2,
        num_heads: int = 4,
        global_spatial_scales: Tuple[int, ...] = (0,),
        windowed_spatial_scales: Tuple[int, ...] = (2,),
        window_size: int = 8,
    ) -> None:
        super().__init__(
            in_channels_list=in_channels_list,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_streams=max_streams,
            num_heads=num_heads,
        )
        self.global_spatial_scales = set(global_spatial_scales)
        self.windowed_spatial_scales = set(windowed_spatial_scales)
        self.window_size = window_size

    def _apply_residual(
        self,
        context: torch.Tensor,
        unadapted: torch.Tensor,
        scale_idx: int,
        return_delta: bool,
        evidence: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if evidence is not None:
            if evidence.shape[-2:] != context.shape[-2:]:
                evidence = F.interpolate(evidence, size=context.shape[-2:], mode="bilinear", align_corners=False)
            context = context + self.evidence_context_proj[scale_idx](evidence)
        context = self.dropout(context)
        raw_delta = self.out_proj[scale_idx](context)
        gate = self.residual_gate[scale_idx](torch.cat([unadapted, context], dim=1))
        delta = gate * raw_delta
        adapted = unadapted + delta
        if return_delta:
            return adapted, delta
        return adapted

    def _global_spatial_context(
        self,
        stream_features: List[torch.Tensor],
        unadapted: torch.Tensor,
        scale_idx: int,
        evidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = unadapted.shape
        stream_count = len(stream_features)
        num_tokens = height * width
        query = self.q_proj[scale_idx](unadapted)
        if evidence is not None:
            if evidence.shape[-2:] != query.shape[-2:]:
                evidence = F.interpolate(evidence, size=query.shape[-2:], mode="bilinear", align_corners=False)
            query = query + self.evidence_q_proj[scale_idx](evidence)
        query = query.view(batch_size, self.num_heads, self.head_dim, num_tokens).transpose(2, 3)

        keys = [self.k_proj[scale_idx](feature) for feature in stream_features]
        values = [self.v_proj[scale_idx](feature) for feature in stream_features]
        key = torch.stack(keys, dim=1).view(batch_size, stream_count, self.num_heads, self.head_dim, num_tokens)
        value = torch.stack(values, dim=1).view(batch_size, stream_count, self.num_heads, self.head_dim, num_tokens)
        key = key.permute(0, 2, 1, 4, 3).reshape(batch_size, self.num_heads, stream_count * num_tokens, self.head_dim)
        value = value.permute(0, 2, 1, 4, 3).reshape(batch_size, self.num_heads, stream_count * num_tokens, self.head_dim)

        attention = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, value)
        return context.transpose(2, 3).reshape(batch_size, self.num_heads * self.head_dim, height, width)

    def _window_partition_q(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        batch_size, heads, head_dim, height, width = x.shape
        return (
            x.permute(0, 3, 4, 1, 2)
            .reshape(batch_size, height // window_size, window_size, width // window_size, window_size, heads, head_dim)
            .permute(0, 1, 3, 5, 2, 4, 6)
            .reshape(batch_size * (height // window_size) * (width // window_size), heads, window_size * window_size, head_dim)
        )

    def _window_partition_kv(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        batch_size, streams, heads, head_dim, height, width = x.shape
        return (
            x.permute(0, 4, 5, 1, 2, 3)
            .reshape(
                batch_size,
                height // window_size,
                window_size,
                width // window_size,
                window_size,
                streams,
                heads,
                head_dim,
            )
            .permute(0, 1, 3, 6, 2, 4, 5, 7)
            .reshape(
                batch_size * (height // window_size) * (width // window_size),
                heads,
                window_size * window_size * streams,
                head_dim,
            )
        )

    def _window_unpartition(
        self,
        x: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        window_size: int,
    ) -> torch.Tensor:
        heads = x.shape[1]
        head_dim = x.shape[-1]
        windows_h = height // window_size
        windows_w = width // window_size
        return (
            x.reshape(batch_size, windows_h, windows_w, heads, window_size, window_size, head_dim)
            .permute(0, 3, 6, 1, 4, 2, 5)
            .reshape(batch_size, heads, head_dim, height, width)
        )

    def _windowed_spatial_context(
        self,
        stream_features: List[torch.Tensor],
        unadapted: torch.Tensor,
        scale_idx: int,
        evidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = unadapted.shape
        window_size = self.window_size
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        padded_h = height + pad_h
        padded_w = width + pad_w

        query = self.q_proj[scale_idx](unadapted)
        if evidence is not None:
            if evidence.shape[-2:] != query.shape[-2:]:
                evidence = F.interpolate(evidence, size=query.shape[-2:], mode="bilinear", align_corners=False)
            query = query + self.evidence_q_proj[scale_idx](evidence)
        query = query.view(batch_size, self.num_heads, self.head_dim, height, width)
        keys = [
            self.k_proj[scale_idx](feature).view(batch_size, self.num_heads, self.head_dim, height, width)
            for feature in stream_features
        ]
        values = [
            self.v_proj[scale_idx](feature).view(batch_size, self.num_heads, self.head_dim, height, width)
            for feature in stream_features
        ]
        if pad_h or pad_w:
            query = F.pad(query, (0, pad_w, 0, pad_h))
            keys = [F.pad(key, (0, pad_w, 0, pad_h)) for key in keys]
            values = [F.pad(value, (0, pad_w, 0, pad_h)) for value in values]

        key = torch.stack(keys, dim=1)
        value = torch.stack(values, dim=1)
        query_windows = self._window_partition_q(query, window_size)
        key_windows = self._window_partition_kv(key, window_size)
        value_windows = self._window_partition_kv(value, window_size)

        attention = torch.matmul(query_windows, key_windows.transpose(-1, -2)) * self.scale
        attention = F.softmax(attention, dim=-1)
        context_windows = torch.matmul(attention, value_windows)
        context = self._window_unpartition(context_windows, batch_size, padded_h, padded_w, window_size)
        return context[:, :, :, :height, :width].reshape(batch_size, self.num_heads * self.head_dim, height, width)

    def forward(
        self,
        stream_features: List[torch.Tensor],
        unadapted: torch.Tensor,
        scale_idx: int,
        return_delta: bool = False,
        evidence: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not stream_features:
            delta = torch.zeros_like(unadapted)
            if return_delta:
                return unadapted, delta
            return unadapted
        if scale_idx in self.global_spatial_scales:
            context = self._global_spatial_context(stream_features, unadapted, scale_idx, evidence)
            return self._apply_residual(context, unadapted, scale_idx, return_delta, evidence)
        if scale_idx in self.windowed_spatial_scales:
            context = self._windowed_spatial_context(stream_features, unadapted, scale_idx, evidence)
            return self._apply_residual(context, unadapted, scale_idx, return_delta, evidence)
        return super().forward(stream_features, unadapted, scale_idx, return_delta, evidence)


class Transformer2DBlock(nn.Module):
    """2D self-attention (global or windowed) + depthwise Conv-MLP block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        window_size: Optional[int] = None,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm2 = nn.GroupNorm(_group_count(hidden_dim), hidden_dim)
        expanded = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_dim, expanded, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(expanded, hidden_dim, kernel_size=1),
            nn.Dropout2d(p=dropout_rate),
        )

    def _global_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        sequence = x.flatten(2).transpose(1, 2)
        sequence_norm = self.norm1(sequence)
        attention_out, _ = self.attn(sequence_norm, sequence_norm, sequence_norm, need_weights=False)
        return attention_out.transpose(1, 2).reshape(batch_size, channels, height, width)

    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        return (
            x.reshape(batch_size, channels, height // window_size, window_size, width // window_size, window_size)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch_size * (height // window_size) * (width // window_size), window_size * window_size, channels)
        )

    def _window_unpartition(
        self,
        windows: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        window_size: int,
    ) -> torch.Tensor:
        channels = windows.shape[-1]
        return (
            windows.reshape(
                batch_size,
                height // window_size,
                width // window_size,
                window_size,
                window_size,
                channels,
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch_size, channels, height, width)
        )

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        window_size = int(self.window_size)
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x
        _, _, padded_h, padded_w = x_padded.shape
        windows = self._window_partition(x_padded, window_size)
        windows_norm = self.norm1(windows)
        attention_windows, _ = self.attn(windows_norm, windows_norm, windows_norm, need_weights=False)
        out = self._window_unpartition(attention_windows, batch_size, padded_h, padded_w, window_size)
        return out[:, :, :height, :width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size is None:
            x = x + self._global_attention(x)
        else:
            x = x + self._window_attention(x)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEvidenceMaskAdapter(nn.Module):
    """DetectiveSAMv2 hierarchical evidence mask adapter."""

    def __init__(
        self,
        hidden_dim: int = 96,
        out_channels: int = 1,
        output_resolution: Tuple[int, int] = (128, 128),
        in_channels_list: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        num_heads: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        channels = in_channels_list or list(MODEL_CHANNELS)
        self.output_resolution = tuple(output_resolution)
        self.hidden_dim = hidden_dim
        self.mask_evidence_proj = nn.ModuleList(
            [
                FlexibleEvidenceProjector(
                    in_channels=hidden_dim + (2 * in_channels),
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    depth=2,
                )
                for in_channels in channels
            ]
        )
        self.low_blocks = nn.Sequential(
            *[
                Transformer2DBlock(hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate, window_size=None)
                for _ in range(2)
            ]
        )
        self.mid_fuse = FlexibleEvidenceProjector(
            in_channels=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            depth=1,
        )
        self.mid_blocks = nn.Sequential(
            *[
                Transformer2DBlock(hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate, window_size=window_size)
                for _ in range(2)
            ]
        )
        self.high_fuse = FlexibleEvidenceProjector(
            in_channels=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            depth=1,
        )
        self.high_blocks = nn.Sequential(
            *[
                Transformer2DBlock(hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate, window_size=window_size)
                for _ in range(2)
            ]
        )
        self.mask_head = nn.Sequential(
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )
        self.aux_head = nn.Sequential(
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def _scale_evidence(
        self,
        adapted: List[torch.Tensor],
        unadapted: List[torch.Tensor],
        evidence: Optional[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        projected = []
        for scale_idx, projection in enumerate(self.mask_evidence_proj):
            if evidence is None:
                evidence_i = torch.zeros(
                    adapted[scale_idx].shape[0],
                    self.hidden_dim,
                    adapted[scale_idx].shape[-2],
                    adapted[scale_idx].shape[-1],
                    dtype=adapted[scale_idx].dtype,
                    device=adapted[scale_idx].device,
                )
            else:
                evidence_i = evidence[scale_idx]
                if evidence_i.shape[-2:] != adapted[scale_idx].shape[-2:]:
                    evidence_i = F.interpolate(
                        evidence_i,
                        size=adapted[scale_idx].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            adapter_delta = adapted[scale_idx] - unadapted[scale_idx]
            projected.append(
                projection(torch.cat([evidence_i, adapted[scale_idx], adapter_delta.abs()], dim=1))
            )
        return projected

    def forward(
        self,
        adapted: List[torch.Tensor],
        streams_unadapted: List[List[torch.Tensor]],
        unadapted: List[torch.Tensor],
        evidence: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        del streams_unadapted
        high_target = self.output_resolution
        scale_features = self._scale_evidence(adapted, unadapted, evidence)

        low = self.low_blocks(scale_features[0])
        mid = scale_features[2] + F.interpolate(
            low,
            size=scale_features[2].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mid = self.mid_fuse(torch.cat([mid, scale_features[2]], dim=1))
        mid = self.mid_blocks(mid)

        high = scale_features[1] + F.interpolate(
            mid,
            size=scale_features[1].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high = self.high_fuse(torch.cat([high, scale_features[1]], dim=1))
        high = self.high_blocks(high)

        if high.shape[-2:] != high_target:
            high = F.interpolate(high, size=high_target, mode="bilinear", align_corners=False)
        mask_logits = self.mask_head(high)
        aux_logits = self.aux_head(mid)
        return mask_logits, aux_logits, None


def _initialize_sam2_hydra(config_dir: Union[str, Path]) -> None:
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir_str = str(Path(config_dir).resolve())
    hydra = GlobalHydra.instance()
    current_dir = getattr(_initialize_sam2_hydra, "_current_dir", None)
    if hydra.is_initialized():
        if current_dir == config_dir_str:
            return
        hydra.clear()
    initialize_config_dir(config_dir=config_dir_str, version_base=None)
    _initialize_sam2_hydra._current_dir = config_dir_str


def _resolve_v2_bundle_paths(
    pretrained_model_name_or_path: Optional[str],
    sam_config_file: Optional[str] = None,
    sam_checkpoint: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
) -> Dict[str, str]:
    """Resolve local paths for SAM2 config, SAM2 weights, and DetectiveSAMv2 checkpoint."""
    source = str(pretrained_model_name_or_path or DEFAULT_DETECTIVE_SAM_V2_REPO).strip()
    if source.startswith("hf://"):
        source = source[5:]

    if os.path.isdir(source):
        root = Path(source)
        ckpt_path = root / "checkpoints" / "detective_sam_v2.pth"
        cfg_file = Path(sam_config_file) if sam_config_file else root / "sam2configs" / "sam2.1_hiera_b+.yaml"
        if not cfg_file.is_absolute():
            cand = root / cfg_file
            if not cand.exists():
                cand = root / "sam2configs" / cfg_file.name
            cfg_file = cand
        sam_ckpt = Path(sam_checkpoint) if sam_checkpoint else root / "sam2configs" / "sam2.1_hiera_base_plus.pt"
        if not sam_ckpt.is_absolute():
            sam_ckpt = root / sam_ckpt
        return {
            "v2_checkpoint": str(ckpt_path),
            "sam_config_dir": str(cfg_file.parent),
            "sam_config_name": cfg_file.name,
            "sam_checkpoint": str(sam_ckpt),
        }

    if os.path.isfile(source):
        ckpt_path = Path(source)
        root = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
        cfg_file = Path(sam_config_file) if sam_config_file else root / "sam2configs" / "sam2.1_hiera_b+.yaml"
        if not cfg_file.is_absolute():
            cand = root / cfg_file
            if not cand.exists():
                cand = root / "sam2configs" / cfg_file.name
            cfg_file = cand
        sam_ckpt = Path(sam_checkpoint) if sam_checkpoint else root / "sam2configs" / "sam2.1_hiera_base_plus.pt"
        if not sam_ckpt.is_absolute():
            sam_ckpt = root / sam_ckpt
        if cfg_file.exists() and sam_ckpt.exists():
            return {
                "v2_checkpoint": str(ckpt_path),
                "sam_config_dir": str(cfg_file.parent),
                "sam_config_name": cfg_file.name,
                "sam_checkpoint": str(sam_ckpt),
            }
        # Fall back to downloading SAM2 base config/weights from default HF bundle
        repo_id = DEFAULT_DETECTIVE_SAM_V2_REPO
    else:
        repo_id = source

    from huggingface_hub import hf_hub_download

    sam_cfg_rel = "sam2configs/sam2.1_hiera_b+.yaml"
    if sam_config_file:
        sam_cfg_name = os.path.basename(sam_config_file)
        sam_cfg_rel = f"sam2configs/{sam_cfg_name}" if not str(sam_config_file).startswith("sam2configs/") else str(sam_config_file)
    sam_ckpt_rel = str(sam_checkpoint or "sam2configs/sam2.1_hiera_base_plus.pt")

    cfg_local = hf_hub_download(
        repo_id=repo_id,
        filename=sam_cfg_rel,
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
    )
    sam_local = hf_hub_download(
        repo_id=repo_id,
        filename=sam_ckpt_rel,
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
    )

    v2_local = source if os.path.isfile(source) else hf_hub_download(
        repo_id=repo_id,
        filename="checkpoints/detective_sam_v2.pth",
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
    )
    return {
        "v2_checkpoint": str(v2_local),
        "sam_config_dir": str(Path(cfg_local).parent),
        "sam_config_name": Path(cfg_local).name,
        "sam_checkpoint": str(sam_local),
    }


class DetectiveSAMv2(nn.Module):
    """
    DetectiveSAMv2 model integrated into `sid_unet`.

    Args:
        pretrained_model_name_or_path: HF repo ID (`Gertlek/DetectiveSAMv2`), local bundle dir,
            or `.pth` checkpoint path.
        load_pretrained_v2: If True, loads weights from `detective_sam_v2.pth`.
        prompt_dim: Adapter & mask adapter hidden dimension (default: 96).
        dropout_rate: Dropout probability for adapters (default: 0.1).
        perturbation_type: Perturbation stream spec (default: `'gaussian_blur+jpeg_compression+gaussian_noise'`).
        perturbation_intensity: Perturbation intensity in [0, 1] (default: 0.5).
        target_size: Spatial input resolution for SAM2.1 encoder (default: (512, 512)).
        freeze_sam2: Keep SAM2.1 image encoder and prompt encoder frozen (default: True).
        freeze_decoder: Keep SAM2.1 mask decoder frozen (default: True).
        trainable_modules: Which v2 modules to keep trainable (`'all'`, `'mask_adapter'`,
            `'adapters+mask_adapter'`, `'mask_head'`, or comma-separated list of module names).
        input_normalized: Whether inputs passed to `forward` are already ImageNet-normalized
            (True for `sid_unet` dataloaders).
        l2sp_weight: Optional L2-SP starting-point anchor weight against pretrained weights
            to prevent catastrophic forgetting during domain adaptation.
        multiscale_supervision: If True during training, exposes `[coarse_mask]` in `multiscale_logits`.
        aux_classifier: Optional auxiliary classifier head over coarsest adapted features.
        num_classes: Number of classes when `aux_classifier=True`.
        use_dummy: Lightweight dummy mode for fast unit tests without downloading weights.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = DEFAULT_DETECTIVE_SAM_V2_REPO,
        load_pretrained_v2: bool = True,
        sam_config_file: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        prompt_dim: int = 96,
        dropout_rate: float = 0.1,
        perturbation_type: str = DEFAULT_V2_PERTURBATIONS,
        perturbation_intensity: float = 0.5,
        target_size: Tuple[int, int] = (512, 512),
        freeze_sam2: bool = True,
        freeze_decoder: bool = True,
        trainable_modules: str = "all",
        input_normalized: bool = True,
        l2sp_weight: float = 0.0,
        multiscale_supervision: bool = False,
        aux_classifier: bool = False,
        num_classes: int = 3,
        in_channels: int = 3,
        out_channels: int = 1,
        use_dummy: bool = False,
        device: Optional[Union[str, torch.device]] = "auto",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.pretrained_model_name_or_path = pretrained_model_name_or_path or DEFAULT_DETECTIVE_SAM_V2_REPO
        self.load_pretrained_v2 = bool(load_pretrained_v2)
        self.prompt_dim = int(prompt_dim)
        self.dropout_rate = float(dropout_rate)
        self.perturbation_type = str(perturbation_type)
        self.perturbation_intensity = float(perturbation_intensity)
        self.perturbations = parse_perturbation_types(self.perturbation_type)
        self.max_streams = max(1, len(self.perturbations))
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.freeze_sam2 = bool(freeze_sam2)
        self.freeze_decoder = bool(freeze_decoder)
        self.trainable_modules = str(trainable_modules or "all")
        self.input_normalized = bool(input_normalized)
        self.l2sp_weight = float(l2sp_weight)
        self.multiscale_supervision = bool(multiscale_supervision)
        self.aux_classifier = bool(aux_classifier)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_dummy = bool(use_dummy)

        if device == "auto" or device is None:
            self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._target_device = torch.device(device)
        else:
            self._target_device = device

        self.register_buffer(
            "_norm_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_norm_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.sam_prompt_encoder: Optional[nn.Module] = None
        self.transforms: Optional[Any] = None
        self.no_mem_embed: Optional[nn.Parameter] = None

        v2_ckpt_path: Optional[str] = None
        if not self.use_dummy:
            bundle = _resolve_v2_bundle_paths(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                sam_config_file=sam_config_file,
                sam_checkpoint=sam_checkpoint,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
            )
            v2_ckpt_path = bundle["v2_checkpoint"]
            self._init_sam2_backbone(
                sam_config_dir=bundle["sam_config_dir"],
                sam_config_name=bundle["sam_config_name"],
                sam_checkpoint=bundle["sam_checkpoint"],
            )
        else:
            self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 256), requires_grad=False)

        self.adapters = SpatialCrossAttentionSharedAdapter(
            in_channels_list=list(MODEL_CHANNELS),
            hidden_dim=self.prompt_dim,
            dropout_rate=self.dropout_rate,
            max_streams=self.max_streams,
        )
        self.evidence_builder = StreamEvidenceBuilder(
            in_channels_list=list(MODEL_CHANNELS),
            hidden_dim=self.prompt_dim,
            dropout_rate=self.dropout_rate,
            max_streams=self.max_streams,
        )
        self.mask_adapter = TransformerEvidenceMaskAdapter(
            hidden_dim=self.prompt_dim,
            out_channels=self.out_channels,
            output_resolution=MASK_ADAPTER_RESOLUTION,
            in_channels_list=list(MODEL_CHANNELS),
            dropout_rate=self.dropout_rate,
        )
        if self.aux_classifier:
            self.classifier_head = nn.Linear(MODEL_CHANNELS[0], self.num_classes)
        else:
            self.classifier_head = None

        if not self.use_dummy and self.load_pretrained_v2 and v2_ckpt_path and os.path.isfile(v2_ckpt_path):
            self._load_pretrained_v2_weights(v2_ckpt_path)

        self._configure_trainable_params()
        self._anchor_params: Dict[str, torch.Tensor] = {}
        if self.l2sp_weight > 0.0:
            self.capture_anchor_weights()

    def _init_sam2_backbone(
        self,
        sam_config_dir: str,
        sam_config_name: str,
        sam_checkpoint: str,
    ) -> None:
        from sam2.build_sam import build_sam2
        from sam2.utils.transforms import SAM2Transforms

        _initialize_sam2_hydra(sam_config_dir)
        sam = build_sam2(sam_config_name, sam_checkpoint, device="cpu")
        sam.image_size = self.target_size[0]
        self.no_mem_embed = sam.no_mem_embed if hasattr(sam, "no_mem_embed") else None
        self.encoder = sam.image_encoder
        self.decoder = sam.sam_mask_decoder
        self.sam_prompt_encoder = sam.sam_prompt_encoder
        self.sam_prompt_encoder.image_embedding_size = MASK_ADAPTER_RESOLUTION
        self.transforms = SAM2Transforms(resolution=sam.image_size, mask_threshold=0.0)

    def _load_pretrained_v2_weights(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            state_dict = None
            for key in ("model", "ema", "model_state_dict", "state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if state_dict is None:
                state_dict = ckpt
        else:
            raise ValueError(f"Unexpected DetectiveSAMv2 checkpoint type at '{ckpt_path}': {type(ckpt)}")

        cleaned = {
            k.removeprefix("module.").removeprefix("_orig_mod."): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        non_trivial_missing = [k for k in missing if not k.startswith("classifier_head.")]
        if non_trivial_missing:
            logger.warning(f"DetectiveSAMv2 missing keys when loading '{ckpt_path}': {non_trivial_missing[:10]}")
        if unexpected:
            logger.warning(f"DetectiveSAMv2 unexpected keys when loading '{ckpt_path}': {unexpected[:10]}")
        logger.info(f"Loaded DetectiveSAMv2 pretrained weights from '{ckpt_path}' (loaded {len(cleaned)} tensors).")

    def _configure_trainable_params(self) -> None:
        # 1. SAM2 backbone, prompt encoder, and mask decoder
        if self.freeze_sam2:
            if self.no_mem_embed is not None:
                self.no_mem_embed.requires_grad_(False)
            for mod in (self.encoder, self.sam_prompt_encoder):
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = False
        if self.decoder is not None:
            for p in self.decoder.parameters():
                p.requires_grad = not self.freeze_decoder

        # 2. DetectiveSAMv2 trainable adapter modules
        spec = self.trainable_modules.strip().lower()
        for mod in (self.evidence_builder, self.adapters, self.mask_adapter):
            for p in mod.parameters():
                p.requires_grad = False

        if spec in ("all", "all_adapters", "default"):
            for mod in (self.evidence_builder, self.adapters, self.mask_adapter):
                for p in mod.parameters():
                    p.requires_grad = True
        elif spec in ("mask_adapter", "mask_adapter_only"):
            for p in self.mask_adapter.parameters():
                p.requires_grad = True
        elif spec in ("adapters+mask_adapter", "adapters_and_mask_adapter"):
            for mod in (self.adapters, self.mask_adapter):
                for p in mod.parameters():
                    p.requires_grad = True
        elif spec in ("mask_head", "head_only"):
            for p in self.mask_adapter.mask_head.parameters():
                p.requires_grad = True
            for p in self.mask_adapter.aux_head.parameters():
                p.requires_grad = True
        else:
            parts = [s.strip() for s in spec.split(",") if s.strip()]
            name_map = {
                "evidence_builder": self.evidence_builder,
                "adapters": self.adapters,
                "mask_adapter": self.mask_adapter,
                "mask_head": self.mask_adapter.mask_head,
                "aux_head": self.mask_adapter.aux_head,
                "decoder": self.decoder,
            }
            for part in parts:
                mod = name_map.get(part)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True

        if self.classifier_head is not None:
            for p in self.classifier_head.parameters():
                p.requires_grad = True

    def capture_anchor_weights(self) -> None:
        """Snapshot current trainable parameters as L2-SP anchor targets."""
        self._anchor_params = {
            name: p.detach().clone()
            for name, p in self.named_parameters()
            if p.requires_grad
        }

    def compute_l2sp_loss(self, device: torch.device) -> torch.Tensor:
        """Compute L2-SP distance 0.5 * sum(||theta - theta_0||^2) across trainable parameters."""
        if not self._anchor_params:
            return torch.zeros((), device=device, dtype=torch.float32)
        total = torch.zeros((), device=device, dtype=torch.float32)
        for name, p in self.named_parameters():
            if p.requires_grad and name in self._anchor_params:
                anchor = self._anchor_params[name].to(device=p.device, dtype=torch.float32)
                diff = p.float() - anchor
                total = total + 0.5 * (diff * diff).sum()
        return total

    def _build_perturbation_streams(
        self,
        images_norm: torch.Tensor,
        x_01: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Generate normalized perturbation streams matching DetectiveSAMv2."""
        if not self.perturbations:
            return []

        params = compute_perturbation_params(self.perturbation_intensity)
        mean = self._norm_mean.to(device=images_norm.device, dtype=images_norm.dtype)
        std = self._norm_std.to(device=images_norm.device, dtype=images_norm.dtype)

        streams: List[torch.Tensor] = []
        for idx, p_name in enumerate(self.perturbations):
            if p_name == "gaussian_blur":
                s_01 = _apply_gaussian_blur_batch(x_01, sigma=float(params["blur_sigma"]))
            elif p_name == "jpeg_compression":
                s_01 = _apply_jpeg_compression_batch(x_01, quality=int(params["jpeg_quality"]))
            elif p_name == "gaussian_noise":
                # Deterministic noise during eval for reproducibility
                seed = 42 + idx
                if not self.training:
                    digest = hashlib.sha256(f"{x_01.shape}|{idx}".encode("utf-8")).digest()
                    seed = int.from_bytes(digest[:4], "big")
                s_01 = _apply_gaussian_noise_batch(
                    x_01,
                    std=float(params["noise_std"]),
                    deterministic=not self.training,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unsupported DetectiveSAMv2 perturbation type: '{p_name}'")
            streams.append((s_01 - mean) / std)
        return streams

    def _project_sam_features(
        self,
        backbone_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        image_embeddings = backbone_features[-1]
        if self.no_mem_embed is not None:
            image_embeddings = image_embeddings + self.no_mem_embed.reshape(1, 256, 1, 1).detach()
        assert self.decoder is not None
        high_res_features = [
            self.decoder.conv_s0(backbone_features[0]),
            self.decoder.conv_s1(backbone_features[1]),
        ]
        return image_embeddings, high_res_features

    def _encode_original_and_streams(
        self,
        orig: torch.Tensor,
        streams: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        if self.use_dummy or self.encoder is None or self.decoder is None:
            b = orig.shape[0]
            h, w = self.target_size
            dev, dt = orig.device, orig.dtype
            unadapted = [
                torch.zeros(b, 256, h // 16, w // 16, device=dev, dtype=dt),
                torch.zeros(b, 32, h // 4, w // 4, device=dev, dtype=dt),
                torch.zeros(b, 64, h // 8, w // 8, device=dev, dtype=dt),
            ]
            streams_unadapted = [
                [torch.zeros_like(u) for _ in streams]
                for u in unadapted
            ]
            return unadapted, streams_unadapted

        def _encode_one(x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            backbone_features = self.encoder(x)["backbone_fpn"]
            return self._project_sam_features(backbone_features)

        if self.freeze_sam2 and self.freeze_decoder:
            with torch.no_grad():
                orig_emb, orig_hi = _encode_one(orig)
                s_emb, s_hi0, s_hi1 = [], [], []
                for stream in streams:
                    emb, hi = _encode_one(stream)
                    s_emb.append(emb)
                    s_hi0.append(hi[0])
                    s_hi1.append(hi[1])
        else:
            orig_emb, orig_hi = _encode_one(orig)
            s_emb, s_hi0, s_hi1 = [], [], []
            for stream in streams:
                emb, hi = _encode_one(stream)
                s_emb.append(emb)
                s_hi0.append(hi[0])
                s_hi1.append(hi[1])

        unadapted = [orig_emb, orig_hi[0], orig_hi[1]]
        streams_unadapted = [s_emb, s_hi0, s_hi1]
        return unadapted, streams_unadapted

    def forward(
        self,
        images: torch.Tensor,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        del kwargs
        orig_h, orig_w = images.shape[-2:]
        model_device, param_dtype = get_submodule_device_dtype(
            self.mask_adapter, images.device, torch.float32
        )

        x_in = images.to(device=model_device)
        if (orig_h, orig_w) != self.target_size:
            x_in = F.interpolate(x_in, size=self.target_size, mode="bilinear", align_corners=False)

        mean = self._norm_mean.to(device=x_in.device, dtype=x_in.dtype)
        std = self._norm_std.to(device=x_in.device, dtype=x_in.dtype)
        if self.input_normalized:
            x_01 = torch.clamp(x_in * std + mean, 0.0, 1.0)
            orig_norm = x_in
        else:
            x_01 = torch.clamp(x_in, 0.0, 1.0)
            orig_norm = (x_01 - mean) / std

        streams = self._build_perturbation_streams(orig_norm, x_01)
        unadapted, streams_unadapted = self._encode_original_and_streams(orig_norm, streams)

        evidence = self.evidence_builder(unadapted, streams_unadapted)
        adapted = [
            self.adapters(
                streams_unadapted[scale_idx],
                unadapted[scale_idx],
                scale_idx,
                evidence=evidence[scale_idx],
            )
            for scale_idx in range(len(unadapted))
        ]

        mask_prompt, coarse_mask, _ = self.mask_adapter(
            adapted,
            streams_unadapted,
            unadapted,
            evidence=evidence,
        )

        if not self.use_dummy and self.decoder is not None and self.sam_prompt_encoder is not None:
            image_embeddings = F.interpolate(
                adapted[0],
                size=adapted[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            high_res_features = [
                F.interpolate(feat, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                for feat in adapted[1:]
            ]
            mask_prompt_in = F.interpolate(mask_prompt, size=mask_prompt.shape[-2:], mode="nearest")

            self.sam_prompt_encoder.image_embedding_size = image_embeddings.shape[-2:]
            sparse_prompt_embeddings, dense_prompt_embeddings = self.sam_prompt_encoder(
                points=None,
                boxes=None,
                masks=mask_prompt_in,
            )
            mask_logits_raw, iou_pred, _, _ = self.decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            mask_logits = F.interpolate(
                mask_logits_raw.float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            iou_pred = None
            mask_logits = F.interpolate(
                mask_prompt.float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )

        cls_logits = None
        if self.aux_classifier and self.classifier_head is not None:
            cls_dev, cls_dtype = get_submodule_device_dtype(self.classifier_head, model_device, param_dtype)
            pooled = F.adaptive_avg_pool2d(adapted[0], (1, 1)).view(adapted[0].shape[0], -1).to(
                device=cls_dev, dtype=cls_dtype
            )
            cls_logits = self.classifier_head(pooled)

        need_train_dict = self.training and (self.l2sp_weight > 0.0 or self.multiscale_supervision)
        if return_dict or need_train_dict:
            res: Dict[str, Any] = {
                "mask_logits": mask_logits,
                "mask_prompt": mask_prompt,
                "coarse_mask": coarse_mask,
                "iou_pred": iou_pred,
            }
            if cls_logits is not None:
                res["class_logits"] = cls_logits
            if self.multiscale_supervision:
                res["multiscale_logits"] = [coarse_mask]
            if self.l2sp_weight > 0.0 and self.training:
                raw_l2sp = self.compute_l2sp_loss(mask_logits.device)
                res["prior_anchor_loss"] = raw_l2sp
                res["prior_anchor_penalty"] = self.l2sp_weight * raw_l2sp
            return res

        if self.aux_classifier and cls_logits is not None:
            return mask_logits, cls_logits

        return mask_logits

    @torch.no_grad()
    def predict_mask(self, images: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        outputs = self.forward(images)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return (torch.sigmoid(logits) >= threshold).float()
