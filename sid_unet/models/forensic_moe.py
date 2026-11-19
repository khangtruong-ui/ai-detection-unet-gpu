"""
ForensicMoE Prototype v1: Lightweight 3-Expert Mixture-of-Experts for Image Manipulation
Localization without Catastrophic Forgetting.

Combines three strictly frozen, open-weight forensic specialists (~187.7M total):
  1. Expert 1B — SparseViT (AAAI 2025, ~50.3M params) -> ImageNet-standardized RGB
  2. Expert 2  — TruFor (Noiseprint++, CVPR 2023, ~68.7M params) -> Raw [0, 1] RGB
  3. Expert 3  — TruFor-DiffusionPrint (CVPRW 2026, ~68.7M params) -> Raw [0, 1] RGB

And trains only a lightweight two-level forensic-signal-only calibration & gating head
(~0.45M–0.6M trainable parameters, <0.4% of ensemble size):
  - Per-expert affine logit calibration (6 scalars)
  - Image-level domain prior router (global forensic summary + SRM/DCT spectral stem)
  - Pixel-level spatial residual gate at H/4 x W/4 strictly on 11 forensic channels
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.sparsevit import SparseViT
from sid_unet.models.trufor import TruFor

logger = logging.getLogger("sid_unet.models.forensic_moe")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_srm_dct_kernels() -> torch.Tensor:
    """
    Build 3 fixed 5x5 high-pass forensic residual filters (zero-sum):
      0: SRM 2nd-order edge filter
      1: SRM 5x5 square KV residual filter
      2: High-frequency 2D DCT basis kernel
    Returns tensor of shape [3, 1, 5, 5].
    """
    k0 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 4.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ) / 4.0

    k1 = torch.tensor(
        [
            [-1.0, 2.0, -2.0, 2.0, -1.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-2.0, 8.0, -12.0, 8.0, -2.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-1.0, 2.0, -2.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
    ) / 12.0

    # 2D DCT high-frequency (u=3, v=3) zero-mean filter
    coords = torch.arange(5, dtype=torch.float32) + 0.5
    basis_u = torch.cos(math.pi * 3.0 * coords / 5.0)
    basis_v = torch.cos(math.pi * 3.0 * coords / 5.0)
    k2 = torch.outer(basis_u, basis_v)
    k2 = k2 - k2.mean()
    k2 = k2 / k2.abs().sum().clamp(min=1e-6)

    return torch.stack([k0, k1, k2], dim=0).unsqueeze(1)


def _make_group_norm(num_channels: int, max_groups: int = 16) -> nn.GroupNorm:
    """Create GroupNorm choosing the largest valid divisor <= max_groups."""
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class _ConvGNAct(nn.Module):
    """Conv2d + GroupNorm + GELU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        max_groups: int = 16,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm = _make_group_norm(out_channels, max_groups=max_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _ResConvBlock(nn.Module):
    """Two-layer 3x3 residual Conv+GroupNorm+GELU block."""

    def __init__(self, channels: int, max_groups: int = 16) -> None:
        super().__init__()
        self.conv1 = _ConvGNAct(channels, channels, kernel_size=3, padding=1, max_groups=max_groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_group_norm(channels, max_groups=max_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm2(self.conv2(self.conv1(x)))
        return self.act(x + h)


class _CrossExpertAttentionBlock(nn.Module):
    """Cross-expert channel (SE) + spatial attention + residual refinement at H/16."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(16, channels // reduction)
        self.res = _ResConvBlock(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.res(x)
        h = h * self.se(h)
        s_in = torch.cat([h.mean(dim=1, keepdim=True), h.max(dim=1, keepdim=True).values], dim=1)
        return h * self.spatial_attn(s_in)


class _DepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable Conv2d block with GroupNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups_norm: int = 8,
    ) -> None:
        super().__init__()
        gn_groups = min(groups_norm, out_channels)
        while out_channels % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(gn_groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class _SpatialResidualBlock(nn.Module):
    """Residual depthwise-separable inverted-bottleneck block for the spatial gate."""

    def __init__(self, channels: int, expand_ratio: int = 2) -> None:
        super().__init__()
        mid = channels * expand_ratio
        gn_groups = min(8, channels)
        while channels % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.norm1 = nn.GroupNorm(gn_groups, channels)
        self.pw1 = nn.Conv2d(channels, mid, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(gn_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.dw(x))
        h = self.pw2(self.act(self.pw1(h)))
        return x + self.norm2(h)


class ForensicMoEGate(nn.Module):
    """
    Two-level forensic-signal-only calibration and gating head (~0.45M-0.55M params).

    Level 1: Per-expert affine logit calibration (6 scalars)
    Level 2a: Image-level domain prior router alpha_img(x) in Delta^2
    Level 2b: Pixel-level spatial residual gate w(x, y) in Delta^2 at H/4 x W/4
    """

    def __init__(
        self,
        num_experts: int = 3,
        spectral_hidden_dims: Tuple[int, int, int] = (96, 192, 256),
        spectral_embed_dim: int = 64,
        spatial_hidden_dim: int = 144,
        spatial_num_blocks: int = 3,
        gate_temperature: float = 1.0,
        spatial_size: Tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.gate_temperature = max(0.05, float(gate_temperature))
        self.spatial_size = tuple(int(v) for v in spatial_size)

        # 1. Per-expert affine logit calibration (6 trainable scalars for 3 experts)
        self.logit_gamma = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))
        self.logit_beta = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))

        # 2. Fixed SRM + DCT high-pass stem -> 3-layer depthwise-separable Conv2d -> 64-dim
        self.register_buffer("srm_dct_kernels", _build_srm_dct_kernels(), persistent=True)
        c1, c2, c3 = spectral_hidden_dims
        self.spectral_stem = nn.Sequential(
            _DepthwiseSeparableBlock(3, c1, stride=2),
            _DepthwiseSeparableBlock(c1, c2, stride=2),
            _DepthwiseSeparableBlock(c2, c3, stride=2),
            nn.Conv2d(c3, spectral_embed_dim, kernel_size=1, bias=True),
            nn.GELU(),
        )

        # Summary vector: 2 detection logits + 5 stats x 5 forensic maps = 27 dims
        self.summary_dim = 2 + 5 * 5
        self.summary_norm = nn.LayerNorm(self.summary_dim)
        router_in_dim = self.summary_dim + spectral_embed_dim
        self.image_router_mlp = nn.Sequential(
            nn.Linear(router_in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_experts),
        )

        # 3. Pixel-level spatial residual gate on strictly 11 forensic channels:
        # [z'_1, z'_2, z'_3] (3) + [conf_2, conf_3] (2) + [H(z'_k)] (3) + [alpha_img] (3) = 11
        self.spatial_in_channels = 11
        self.spatial_in_proj = nn.Sequential(
            nn.Conv2d(self.spatial_in_channels, spatial_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, spatial_hidden_dim),
            nn.GELU(),
        )
        self.spatial_blocks = nn.Sequential(
            *[_SpatialResidualBlock(spatial_hidden_dim, expand_ratio=2) for _ in range(spatial_num_blocks)]
        )
        self.spatial_out = nn.Conv2d(spatial_hidden_dim, self.num_experts, kernel_size=1, bias=True)
        # Zero-initialize spatial residual conv so initial gate equals alpha_img(x)
        nn.init.zeros_(self.spatial_out.weight)
        nn.init.zeros_(self.spatial_out.bias)

    def _extract_spectral_features(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Apply fixed SRM+DCT kernels on luminance of raw [0, 1] RGB and pool to 64-dim."""
        # Convert to luminance [B, 1, H, W]
        luma = (
            0.299 * x_raw[:, 0:1] + 0.587 * x_raw[:, 1:2] + 0.114 * x_raw[:, 2:3]
        ).float()
        kernels = self.srm_dct_kernels.to(device=luma.device, dtype=luma.dtype)
        hp = F.conv2d(luma, kernels, padding=2)
        feat = self.spectral_stem(hp)
        return feat.mean(dim=(2, 3))

    @staticmethod
    def _compute_map_stats(maps_small: torch.Tensor, is_logit: bool) -> torch.Tensor:
        """
        Compute [mean, std, max, p95, frac(>0.5)] per channel on [B, C, H', W'].
        Returns [B, C * 5].
        """
        b, c, _, _ = maps_small.shape
        flat = maps_small.reshape(b, c, -1).float()
        mean_v = flat.mean(dim=-1)
        std_v = flat.std(dim=-1, unbiased=False)
        max_v = flat.max(dim=-1).values
        p95_v = torch.quantile(flat, 0.95, dim=-1)
        if is_logit:
            frac_v = (flat > 0.0).float().mean(dim=-1)
        else:
            frac_v = (flat > 0.5).float().mean(dim=-1)
        return torch.stack([mean_v, std_v, max_v, p95_v, frac_v], dim=-1).reshape(b, c * 5)

    @staticmethod
    def _binary_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Element-wise binary entropy H(sigmoid(logits)) in nats."""
        p = torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)
        return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

    def forward(
        self,
        x_raw: torch.Tensor,
        expert_logits: torch.Tensor,
        conf_probs: torch.Tensor,
        det_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_raw: Raw `[0.0, 1.0]` RGB tensor `[B, 3, H, W]`.
            expert_logits: Stacked raw mask logits `[B, 3, H, W]` for Experts (1B, 2, 3).
            conf_probs: Stacked TruFor confidence probs `[B, 2, H, W]` in `[0, 1]`.
            det_logits: Stacked TruFor detection logits `[B, 2]`.
        """
        orig_h, orig_w = expert_logits.shape[2], expert_logits.shape[3]
        target_sh = min(self.spatial_size[0], max(8, orig_h // 4))
        target_sw = min(self.spatial_size[1], max(8, orig_w // 4))

        # 1. Affine calibration in float32
        gamma = self.logit_gamma.view(1, self.num_experts, 1, 1)
        beta = self.logit_beta.view(1, self.num_experts, 1, 1)
        cal_logits = expert_logits.float() * gamma + beta

        # Downsample forensic signals to compact spatial grid (128x128 at 512 input)
        cal_small = F.interpolate(cal_logits, size=(target_sh, target_sw), mode="bilinear", align_corners=False)
        conf_small = F.interpolate(conf_probs.float(), size=(target_sh, target_sw), mode="bilinear", align_corners=False)

        # 2. Image-level domain prior router alpha_img
        logit_stats = self._compute_map_stats(cal_small, is_logit=True)
        conf_stats = self._compute_map_stats(conf_small, is_logit=False)
        det_feats = torch.clamp(det_logits.float(), -15.0, 15.0)
        summary_vec = self.summary_norm(torch.cat([det_feats, logit_stats, conf_stats], dim=-1))

        spectral_vec = self._extract_spectral_features(x_raw)
        g_img = self.image_router_mlp(torch.cat([summary_vec, spectral_vec], dim=-1))
        alpha_img = F.softmax(g_img / self.gate_temperature, dim=-1)  # [B, 3]

        # 3. Pixel-level spatial residual gate strictly on 11 forensic channels
        entropy_small = self._binary_entropy(cal_small)
        alpha_bcast = alpha_img.view(-1, self.num_experts, 1, 1).expand(-1, self.num_experts, target_sh, target_sw)
        forensic_in = torch.cat([cal_small, conf_small, entropy_small, alpha_bcast], dim=1)  # [B, 11, Hs, Ws]

        delta_g_small = self.spatial_out(self.spatial_blocks(self.spatial_in_proj(forensic_in)))
        log_alpha = torch.log(alpha_img.clamp(min=1e-6)).view(-1, self.num_experts, 1, 1)
        w_small = F.softmax(log_alpha + delta_g_small, dim=1)
        w_full = F.interpolate(w_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        # Re-normalize after bilinear interpolation to preserve exact simplex sum = 1
        w_full = w_full / w_full.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Mixture in calibrated log-odds space
        moe_logits = (w_full * cal_logits).sum(dim=1, keepdim=True)

        return {
            "mask_logits": moe_logits,
            "calibrated_logits": cal_logits,
            "alpha_img": alpha_img,
            "spatial_weights": w_full,
            "delta_g": delta_g_small,
        }


class ForensicMoE(nn.Module):
    """
    ForensicMoE Prototype v1 orchestrating 3 frozen forensic experts + 1 tiny trainable gate.
    """

    requires_gt_masks = True

    def __init__(
        self,
        pretrained: bool = True,
        allow_random_init: bool = False,
        sparsevit_checkpoint_path: Optional[str] = None,
        sparsevit_hf_repo: Optional[str] = None,
        trufor_checkpoint_path: Optional[str] = None,
        trufor_hf_repo: Optional[str] = None,
        diffusionprint_checkpoint_path: Optional[str] = None,
        diffusionprint_hf_repo: Optional[str] = None,
        cache_dir: Optional[str] = None,
        img_size: int = 512,
        gate_temperature: float = 1.0,
        spectral_hidden_dims: Tuple[int, int, int] = (96, 192, 256),
        spectral_embed_dim: int = 64,
        spatial_hidden_dim: int = 144,
        spatial_num_blocks: int = 3,
        balance_loss_weight: float = 0.05,
        oracle_loss_weight: float = 0.15,
        oracle_temperature: float = 0.15,
        aux_classifier: bool = False,
        num_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.balance_loss_weight = float(balance_loss_weight)
        self.oracle_loss_weight = float(oracle_loss_weight)
        self.oracle_temperature = max(0.01, float(oracle_temperature))
        self.aux_classifier = bool(aux_classifier)
        self.num_classes = int(num_classes)

        if not pretrained and not allow_random_init:
            raise ValueError(
                "ForensicMoE requires pretrained=True (or explicit allow_random_init=True for unit tests). "
                "Random expert initialization is strictly disabled."
            )

        # 1. Strictly frozen experts (all locked in .eval() with requires_grad=False)
        self.expert_sparsevit = SparseViT(
            img_size=self.img_size,
            pretrained=pretrained,
            allow_random_init=allow_random_init,
            checkpoint_path=sparsevit_checkpoint_path,
            hf_repo=sparsevit_hf_repo,
            cache_dir=cache_dir,
            freeze=True,
            auto_normalize=False,
        )
        self.expert_trufor = TruFor(
            np_out_channels=1,
            pretrained=pretrained,
            allow_random_init=allow_random_init,
            checkpoint_path=trufor_checkpoint_path,
            hf_repo=trufor_hf_repo,
            cache_dir=cache_dir,
            freeze=True,
        )
        self.expert_diffusionprint = TruFor(
            np_out_channels=64,
            pretrained=pretrained,
            allow_random_init=allow_random_init,
            checkpoint_path=diffusionprint_checkpoint_path,
            hf_repo=diffusionprint_hf_repo,
            cache_dir=cache_dir,
            freeze=True,
        )

        # 2. Trainable calibration & gating head
        self.gate = self._build_gate(
            gate_temperature=gate_temperature,
            spectral_hidden_dims=tuple(int(v) for v in spectral_hidden_dims),
            spectral_embed_dim=int(spectral_embed_dim),
            spatial_hidden_dim=int(spatial_hidden_dim),
            spatial_num_blocks=int(spatial_num_blocks),
            **kwargs,
        )

        self.class_head: Optional[nn.Module] = None
        if self.aux_classifier:
            self.class_head = nn.Linear(5, self.num_classes)

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

        # Enforce frozen state immediately
        self._lock_frozen_experts()

    def _build_gate(
        self,
        gate_temperature: float,
        spectral_hidden_dims: Tuple[int, int, int],
        spectral_embed_dim: int,
        spatial_hidden_dim: int,
        spatial_num_blocks: int,
        **kwargs: Any,
    ) -> nn.Module:
        return ForensicMoEGate(
            num_experts=3,
            spectral_hidden_dims=spectral_hidden_dims,
            spectral_embed_dim=spectral_embed_dim,
            spatial_hidden_dim=spatial_hidden_dim,
            spatial_num_blocks=spatial_num_blocks,
            gate_temperature=gate_temperature,
        )

    def _lock_frozen_experts(self) -> None:
        for expert in (self.expert_sparsevit, self.expert_trufor, self.expert_diffusionprint):
            expert.set_frozen(True)
            expert.eval()

    def train(self, mode: bool = True) -> "ForensicMoE":
        super().train(mode)
        # Never allow Trainer.train_epoch() -> model.train() to unfreeze or flip BN in experts
        self._lock_frozen_experts()
        return self

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _ensure_raw_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is in raw `[0.0, 1.0]` RGB scale.
        Reverses ImageNet standardization if detected.
        """
        if float(x.min().detach()) < -0.05 or float(x.max().detach()) > 1.05:
            mean = self._imagenet_mean.to(device=x.device, dtype=x.dtype)
            std = self._imagenet_std.to(device=x.device, dtype=x.dtype)
            return torch.clamp(x * std + mean, 0.0, 1.0)
        return x

    def _to_imagenet_rgb(self, x_raw: torch.Tensor) -> torch.Tensor:
        mean = self._imagenet_mean.to(device=x_raw.device, dtype=x_raw.dtype)
        std = self._imagenet_std.to(device=x_raw.device, dtype=x_raw.dtype)
        return (x_raw - mean) / std

    def _compute_gate_regularization(
        self,
        alpha_img: torch.Tensor,
        calibrated_logits: torch.Tensor,
        gt_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute anti-collapse load-balancing regularizer + optional soft per-sample
        oracle supervision toward argmax_k quality_k(z'_k, gt_mask).
        """
        eps = 1e-8
        mean_alpha = alpha_img.mean(dim=0).clamp(min=eps)
        k = float(alpha_img.shape[1])
        # KL(mean_alpha || Uniform(1/K)) >= 0
        balance_loss = torch.sum(mean_alpha * (torch.log(mean_alpha) + math.log(k)))
        reg_loss = self.balance_loss_weight * balance_loss

        metrics: Dict[str, float] = {
            "moe_balance_loss": float(balance_loss.detach().item()),
        }

        if gt_masks is not None and self.oracle_loss_weight > 0.0:
            with torch.no_grad():
                gt = gt_masks.float()
                if gt.ndim == 3:
                    gt = gt.unsqueeze(1)
                if gt.shape[2:] != calibrated_logits.shape[2:]:
                    gt = F.interpolate(gt, size=calibrated_logits.shape[2:], mode="nearest")
                gt_bin = (gt >= 0.5).float()  # [B, 1, H, W]

                # Per-sample negative BCE + soft Dice score across all 3 experts [B, 3]
                probs = torch.sigmoid(calibrated_logits.detach().float())
                gt_exp = gt_bin.expand_as(probs)
                bce_per_sample = F.binary_cross_entropy_with_logits(
                    calibrated_logits.detach().float(),
                    gt_exp,
                    reduction="none",
                ).mean(dim=(2, 3))
                inter = (probs * gt_exp).sum(dim=(2, 3))
                denom = probs.sum(dim=(2, 3)) + gt_exp.sum(dim=(2, 3))
                dice_score = (2.0 * inter + 1.0) / (denom + 1.0)
                oracle_score = dice_score - bce_per_sample
                q_oracle = F.softmax(oracle_score / self.oracle_temperature, dim=-1)

            oracle_kl = F.kl_div(
                torch.log(alpha_img.clamp(min=eps)),
                q_oracle,
                reduction="batchmean",
            )
            reg_loss = reg_loss + self.oracle_loss_weight * oracle_kl
            metrics["moe_oracle_kl"] = float(oracle_kl.detach().item())

        return reg_loss, metrics

    def forward(
        self,
        x: torch.Tensor,
        gt_masks: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        x_raw = self._ensure_raw_rgb(x)
        x_imagenet = self._to_imagenet_rgb(x_raw)

        # All 3 experts run strictly frozen in eval + no_grad
        with torch.no_grad():
            self._lock_frozen_experts()
            out_1 = self.expert_sparsevit(x_imagenet, return_dict=True, input_is_normalized=True)
            out_2 = self.expert_trufor(x_raw, return_dict=True)
            out_3 = self.expert_diffusionprint(x_raw, return_dict=True)

            z1 = out_1["mask_logits"].detach()
            z2 = out_2["mask_logits"].detach()
            z3 = out_3["mask_logits"].detach()
            conf2 = out_2["conf_prob"].detach()
            conf3 = out_3["conf_prob"].detach()
            det2 = out_2["det_logit"].detach()
            det3 = out_3["det_logit"].detach()

        expert_logits = torch.cat([z1, z2, z3], dim=1)  # [B, 3, H, W]
        conf_probs = torch.cat([conf2, conf3], dim=1)   # [B, 2, H, W]
        det_logits = torch.cat([det2, det3], dim=1)     # [B, 2]

        gate_out = self.gate(
            x_raw=x_raw,
            expert_logits=expert_logits,
            conf_probs=conf_probs,
            det_logits=det_logits,
        )
        mask_logits = gate_out["mask_logits"]

        reg_loss, reg_metrics = self._compute_gate_regularization(
            alpha_img=gate_out["alpha_img"],
            calibrated_logits=gate_out["calibrated_logits"],
            gt_masks=gt_masks if self.training else None,
        )

        class_logits = None
        if self.class_head is not None:
            cls_in = torch.cat([gate_out["alpha_img"], torch.sigmoid(det_logits)], dim=-1)
            class_logits = self.class_head(cls_in)

        if return_dict:
            res: Dict[str, Any] = {
                "mask_logits": mask_logits,
                "pred_mask": torch.sigmoid(mask_logits),
                "expert_logits": expert_logits,
                "calibrated_logits": gate_out["calibrated_logits"],
                "alpha_img": gate_out["alpha_img"],
                "spatial_weights": gate_out["spatial_weights"],
                "conf_probs": conf_probs,
                "det_logits": det_logits,
                "artifact_loss": reg_loss,
                "align_loss": reg_loss,
                **reg_metrics,
            }
            if class_logits is not None:
                res["class_logits"] = class_logits
                res["pred_label"] = torch.argmax(class_logits, dim=1)
            return res

        if self.aux_classifier and class_logits is not None:
            return mask_logits, class_logits
        return mask_logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        out = self.forward(x, return_dict=True)
        logits = out["mask_logits"] if isinstance(out, dict) else out
        return (torch.sigmoid(logits) >= threshold).float()


class ForensicMoEv2Gate(nn.Module):
    """
    ForensicMoE v2 Multi-Scale Cross-Expert Feature Router + Uncertainty-Gated
    Top-Down Residual FPN Decoder (~6.6M trainable params).

    Formulation:
        z_MoE-v2(x, y) = sum_k w_k(x, y) * z'_k(x, y) + g_unc(x, y) * delta_z_FPN(F_1, F_2, F_3)
    """

    def __init__(
        self,
        num_experts: int = 3,
        gate_temperature: float = 1.0,
        router_proj_dim: int = 96,
        router_hidden_dim: int = 192,
        unc_hidden_dim: int = 16,
        unc_bias_init: float = -1.5,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.gate_temperature = max(0.05, float(gate_temperature))

        # 1. Per-expert affine logit calibration (z'_k = gamma_k * z_k + beta_k)
        self.logit_gamma = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))
        self.logit_beta = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))

        # 2. Multi-Scale Cross-Expert Attention Router (~1.7M params) -> w_k(x, y)
        self.router_proj_1 = _ConvGNAct(512, router_proj_dim, kernel_size=1, padding=0)
        self.router_proj_2 = _ConvGNAct(320, router_proj_dim, kernel_size=1, padding=0)
        self.router_proj_3 = _ConvGNAct(320, router_proj_dim, kernel_size=1, padding=0)

        # 3 * 96 + (3 cal_logits + 2 conf_probs + 3 entropy) = 296 channels
        router_fuse_in = router_proj_dim * 3 + 8
        self.router_fuse = _ConvGNAct(router_fuse_in, router_hidden_dim, kernel_size=3, padding=1)
        self.router_attn = _CrossExpertAttentionBlock(router_hidden_dim)

        self.router_img_head = nn.Sequential(
            nn.Linear(router_hidden_dim + 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, self.num_experts),
        )
        self.router_up16_to_8 = _ConvGNAct(router_hidden_dim, 128, kernel_size=3, padding=1)
        self.router_up8_to_4 = _ConvGNAct(128, 96, kernel_size=3, padding=1)
        self.router_spatial_out = nn.Conv2d(96, self.num_experts, kernel_size=1, bias=True)
        # Zero-init spatial residual routing so initial spatial weights equal alpha_img
        nn.init.zeros_(self.router_spatial_out.weight)
        nn.init.zeros_(self.router_spatial_out.bias)

        # 3. Top-Down Multi-Scale Residual FPN Decoder (~5.0M params) -> delta_z_FPN
        # Strictly forensic representations (zero raw RGB semantics)
        # H/32: [s2_h32 (512), s3_h32 (512)] -> 1024 -> 192
        self.fpn_proj_32 = _ConvGNAct(1024, 192, kernel_size=1, padding=0)
        self.fpn_refine_32 = nn.Sequential(_ResConvBlock(192), _ResConvBlock(192))

        # H/16: [s1_lff (512), s2_h16 (320), s3_h16 (320)] -> 1152 -> 128
        self.fpn_proj_16 = _ConvGNAct(1152, 128, kernel_size=1, padding=0)
        self.fpn_smooth_16 = _ConvGNAct(128 + 192, 192, kernel_size=3, padding=1)
        self.fpn_refine_16 = nn.Sequential(_ResConvBlock(192), _ResConvBlock(192))

        # H/8: [s1_h8 (128), s2_h8 (128), s3_h8 (128)] -> 384 -> 96
        self.fpn_proj_8 = _ConvGNAct(384, 96, kernel_size=1, padding=0)
        self.fpn_smooth_8 = _ConvGNAct(96 + 192, 128, kernel_size=3, padding=1)
        self.fpn_refine_8 = nn.Sequential(_ResConvBlock(128), _ResConvBlock(128))

        # H/4: [s2_h4 (512), s3_h4 (512), dp_h4 (64), cal_logits (3), conf_probs (2), entropy (3)] -> 1096 -> 64
        self.fpn_proj_4 = _ConvGNAct(1096, 64, kernel_size=1, padding=0)
        self.fpn_smooth_4 = _ConvGNAct(64 + 128, 96, kernel_size=3, padding=1)
        self.fpn_refine_4 = _ResConvBlock(96)

        # Full-res progressive refinement (H/4 -> H/2 -> H)
        self.fpn_up_2 = _ConvGNAct(96, 64, kernel_size=3, padding=1)
        self.fpn_up_1 = _ConvGNAct(64, 32, kernel_size=3, padding=1)
        self.fpn_out = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        # Zero-init final residual conv so delta_z_FPN == 0 at step 0
        nn.init.zeros_(self.fpn_out.weight)
        nn.init.zeros_(self.fpn_out.bias)

        # 4. Uncertainty / Disagreement Gate g_unc(x, y) in (0, 1)
        # Conditioned on [mean_entropy(z'_1..3), std_k(z'_k), 1 - max(conf_2, conf_3)]
        self.unc_gate = nn.Sequential(
            nn.Conv2d(3, unc_hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(unc_hidden_dim, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.unc_gate[-1].bias, float(unc_bias_init))

    @staticmethod
    def _binary_entropy(logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)
        return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

    @staticmethod
    def _align_to(
        tensor: torch.Tensor,
        size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if tensor.shape[2:] != size:
            tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def forward(
        self,
        expert_logits: torch.Tensor,
        conf_probs: torch.Tensor,
        det_logits: torch.Tensor,
        features_1: Dict[str, torch.Tensor],
        features_2: Dict[str, torch.Tensor],
        features_3: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        orig_h, orig_w = expert_logits.shape[2], expert_logits.shape[3]

        # 1. Per-expert affine calibration in float32
        gamma = self.logit_gamma.view(1, self.num_experts, 1, 1)
        beta = self.logit_beta.view(1, self.num_experts, 1, 1)
        cal_logits = expert_logits.float() * gamma + beta  # [B, 3, H, W]
        entropy_full = self._binary_entropy(cal_logits)    # [B, 3, H, W]
        conf_f = conf_probs.float()                        # [B, 2, H, W]

        s1_h8, s1_lff = features_1["h8"], features_1["lff"]
        s2_h4, s2_h8, s2_h16, s2_h32 = (
            features_2["h4"],
            features_2["h8"],
            features_2["h16"],
            features_2["h32"],
        )
        s3_h4, s3_h8, s3_h16, s3_h32, dp_raw = (
            features_3["h4"],
            features_3["h8"],
            features_3["h16"],
            features_3["h32"],
            features_3["dncnn"],
        )

        feat_dtype = s2_h16.dtype
        sz32 = (s2_h32.shape[2], s2_h32.shape[3])
        sz16 = (s2_h16.shape[2], s2_h16.shape[3])
        sz8 = (s2_h8.shape[2], s2_h8.shape[3])
        sz4 = (s2_h4.shape[2], s2_h4.shape[3])
        sz2 = (max(1, orig_h // 2), max(1, orig_w // 2))

        # 2. Multi-Scale Cross-Expert Attention Router at H/16 -> H/4
        r1 = self.router_proj_1(self._align_to(s1_lff, sz16, feat_dtype))
        r2 = self.router_proj_2(self._align_to(s2_h16, sz16, feat_dtype))
        r3 = self.router_proj_3(self._align_to(s3_h16, sz16, feat_dtype))
        cal_h16 = self._align_to(cal_logits, sz16, feat_dtype)
        conf_h16 = self._align_to(conf_f, sz16, feat_dtype)
        ent_h16 = self._align_to(entropy_full, sz16, feat_dtype)

        router_h16 = self.router_attn(
            self.router_fuse(torch.cat([r1, r2, r3, cal_h16, conf_h16, ent_h16], dim=1))
        )

        # Global image-level domain prior alpha_img from GAP(router_h16) + det_logits
        pooled = router_h16.float().mean(dim=(2, 3))
        det_feats = torch.clamp(det_logits.float(), -15.0, 15.0)
        g_img = self.router_img_head(torch.cat([pooled, det_feats], dim=-1))
        alpha_img = F.softmax(g_img / self.gate_temperature, dim=-1)  # [B, 3]

        # Smooth spatial routing weights H/16 -> H/8 -> H/4
        r_up8 = self.router_up16_to_8(self._align_to(router_h16, sz8, feat_dtype))
        r_up4 = self.router_up8_to_4(self._align_to(r_up8, sz4, feat_dtype))
        delta_w_h4 = self.router_spatial_out(r_up4).float()
        log_alpha = torch.log(alpha_img.clamp(min=1e-6)).view(-1, self.num_experts, 1, 1)
        w_h4 = F.softmax(log_alpha + delta_w_h4, dim=1)
        w_full = F.interpolate(w_h4, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        w_full = w_full / w_full.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # 3. Top-Down Multi-Scale Residual FPN Decoder -> delta_z_FPN
        p32 = self.fpn_refine_32(
            self.fpn_proj_32(
                torch.cat([s2_h32.to(feat_dtype), self._align_to(s3_h32, sz32, feat_dtype)], dim=1)
            )
        )

        lat16 = self.fpn_proj_16(
            torch.cat(
                [
                    self._align_to(s1_lff, sz16, feat_dtype),
                    s2_h16.to(feat_dtype),
                    self._align_to(s3_h16, sz16, feat_dtype),
                ],
                dim=1,
            )
        )
        p16 = self.fpn_refine_16(
            self.fpn_smooth_16(torch.cat([lat16, self._align_to(p32, sz16, feat_dtype)], dim=1))
        )

        lat8 = self.fpn_proj_8(
            torch.cat(
                [
                    self._align_to(s1_h8, sz8, feat_dtype),
                    s2_h8.to(feat_dtype),
                    self._align_to(s3_h8, sz8, feat_dtype),
                ],
                dim=1,
            )
        )
        p8 = self.fpn_refine_8(
            self.fpn_smooth_8(torch.cat([lat8, self._align_to(p16, sz8, feat_dtype)], dim=1))
        )

        if dp_raw.shape[2] == sz4[0] * 4 and dp_raw.shape[3] == sz4[1] * 4:
            dp_h4 = F.avg_pool2d(dp_raw.to(feat_dtype), kernel_size=4, stride=4)
        else:
            dp_h4 = F.adaptive_avg_pool2d(dp_raw.to(feat_dtype), output_size=sz4)

        cal_h4 = self._align_to(cal_logits, sz4, feat_dtype)
        conf_h4 = self._align_to(conf_f, sz4, feat_dtype)
        ent_h4 = self._align_to(entropy_full, sz4, feat_dtype)
        lat4 = self.fpn_proj_4(
            torch.cat(
                [
                    s2_h4.to(feat_dtype),
                    self._align_to(s3_h4, sz4, feat_dtype),
                    dp_h4,
                    cal_h4,
                    conf_h4,
                    ent_h4,
                ],
                dim=1,
            )
        )
        p4 = self.fpn_refine_4(
            self.fpn_smooth_4(torch.cat([lat4, self._align_to(p8, sz4, feat_dtype)], dim=1))
        )

        p2 = self.fpn_up_2(self._align_to(p4, sz2, feat_dtype))
        p1 = self.fpn_up_1(self._align_to(p2, (orig_h, orig_w), feat_dtype))
        delta_z_fpn = self.fpn_out(p1).float()  # [B, 1, H, W]

        # 4. Uncertainty / Disagreement Gate g_unc(x, y)
        mean_entropy = entropy_full.mean(dim=1, keepdim=True)
        std_logits = cal_logits.std(dim=1, unbiased=False, keepdim=True)
        disagreement_conf = 1.0 - conf_f.max(dim=1, keepdim=True).values
        unc_in = torch.cat([mean_entropy, std_logits, disagreement_conf], dim=1).to(feat_dtype)
        g_unc = torch.sigmoid(self.unc_gate(unc_in).float())  # [B, 1, H, W]

        # 5. Anchored Mixture + Uncertainty-Gated Multi-Scale FPN Residual
        anchored_mixture = (w_full * cal_logits).sum(dim=1, keepdim=True)
        moe_logits = anchored_mixture + g_unc * delta_z_fpn

        return {
            "mask_logits": moe_logits,
            "anchored_logits": anchored_mixture,
            "delta_z_fpn": delta_z_fpn,
            "g_unc": g_unc,
            "calibrated_logits": cal_logits,
            "alpha_img": alpha_img,
            "spatial_weights": w_full,
            "delta_g": delta_w_h4,
        }


class ForensicMoEv2(ForensicMoE):
    """
    ForensicMoE v2 orchestrating the same 3 strictly frozen forensic experts with a
    Multi-Scale Cross-Expert Feature Router + Uncertainty-Gated Residual FPN Decoder
    (~6.6M trainable params, <4.5% of the ~194M ensemble).
    """

    def __init__(
        self,
        pretrained: bool = True,
        allow_random_init: bool = False,
        sparsevit_checkpoint_path: Optional[str] = None,
        sparsevit_hf_repo: Optional[str] = None,
        trufor_checkpoint_path: Optional[str] = None,
        trufor_hf_repo: Optional[str] = None,
        diffusionprint_checkpoint_path: Optional[str] = None,
        diffusionprint_hf_repo: Optional[str] = None,
        cache_dir: Optional[str] = None,
        img_size: int = 512,
        gate_temperature: float = 1.0,
        router_proj_dim: int = 96,
        router_hidden_dim: int = 192,
        unc_hidden_dim: int = 16,
        unc_bias_init: float = -1.5,
        balance_loss_weight: float = 0.05,
        oracle_loss_weight: float = 0.02,
        oracle_temperature: float = 0.5,
        residual_l2_weight: float = 1e-4,
        aux_classifier: bool = False,
        num_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        self.residual_l2_weight = float(residual_l2_weight)
        super().__init__(
            pretrained=pretrained,
            allow_random_init=allow_random_init,
            sparsevit_checkpoint_path=sparsevit_checkpoint_path,
            sparsevit_hf_repo=sparsevit_hf_repo,
            trufor_checkpoint_path=trufor_checkpoint_path,
            trufor_hf_repo=trufor_hf_repo,
            diffusionprint_checkpoint_path=diffusionprint_checkpoint_path,
            diffusionprint_hf_repo=diffusionprint_hf_repo,
            cache_dir=cache_dir,
            img_size=img_size,
            gate_temperature=gate_temperature,
            balance_loss_weight=balance_loss_weight,
            oracle_loss_weight=oracle_loss_weight,
            oracle_temperature=oracle_temperature,
            aux_classifier=aux_classifier,
            num_classes=num_classes,
            router_proj_dim=router_proj_dim,
            router_hidden_dim=router_hidden_dim,
            unc_hidden_dim=unc_hidden_dim,
            unc_bias_init=unc_bias_init,
            **kwargs,
        )

    def _build_gate(
        self,
        gate_temperature: float,
        spectral_hidden_dims: Tuple[int, int, int],
        spectral_embed_dim: int,
        spatial_hidden_dim: int,
        spatial_num_blocks: int,
        **kwargs: Any,
    ) -> nn.Module:
        return ForensicMoEv2Gate(
            num_experts=3,
            gate_temperature=gate_temperature,
            router_proj_dim=int(kwargs.get("router_proj_dim", 96)),
            router_hidden_dim=int(kwargs.get("router_hidden_dim", 192)),
            unc_hidden_dim=int(kwargs.get("unc_hidden_dim", 16)),
            unc_bias_init=float(kwargs.get("unc_bias_init", -1.5)),
        )

    def forward(
        self,
        x: torch.Tensor,
        gt_masks: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        x_raw = self._ensure_raw_rgb(x)
        x_imagenet = self._to_imagenet_rgb(x_raw)

        # All 3 experts run strictly frozen in eval + no_grad, returning multi-scale taps
        with torch.no_grad():
            self._lock_frozen_experts()
            out_1 = self.expert_sparsevit(
                x_imagenet, return_dict=True, input_is_normalized=True, return_features=True
            )
            out_2 = self.expert_trufor(x_raw, return_dict=True, return_features=True)
            out_3 = self.expert_diffusionprint(x_raw, return_dict=True, return_features=True)

            z1 = out_1["mask_logits"].detach()
            z2 = out_2["mask_logits"].detach()
            z3 = out_3["mask_logits"].detach()
            conf2 = out_2["conf_prob"].detach()
            conf3 = out_3["conf_prob"].detach()
            det2 = out_2["det_logit"].detach()
            det3 = out_3["det_logit"].detach()
            feats_1 = {k: v.detach() for k, v in out_1["features"].items()}
            feats_2 = {k: v.detach() for k, v in out_2["features"].items()}
            feats_3 = {k: v.detach() for k, v in out_3["features"].items()}

        expert_logits = torch.cat([z1, z2, z3], dim=1)
        conf_probs = torch.cat([conf2, conf3], dim=1)
        det_logits = torch.cat([det2, det3], dim=1)

        gate_out = self.gate(
            expert_logits=expert_logits,
            conf_probs=conf_probs,
            det_logits=det_logits,
            features_1=feats_1,
            features_2=feats_2,
            features_3=feats_3,
        )
        mask_logits = gate_out["mask_logits"]

        reg_loss, reg_metrics = self._compute_gate_regularization(
            alpha_img=gate_out["alpha_img"],
            calibrated_logits=gate_out["calibrated_logits"],
            gt_masks=gt_masks if self.training else None,
        )
        if self.residual_l2_weight > 0.0:
            res_l2 = torch.mean(gate_out["delta_z_fpn"].float() ** 2)
            reg_loss = reg_loss + self.residual_l2_weight * res_l2
            reg_metrics["moe_residual_l2"] = float(res_l2.detach().item())

        class_logits = None
        if self.class_head is not None:
            cls_in = torch.cat([gate_out["alpha_img"], torch.sigmoid(det_logits)], dim=-1)
            class_logits = self.class_head(cls_in)

        if return_dict:
            res: Dict[str, Any] = {
                "mask_logits": mask_logits,
                "pred_mask": torch.sigmoid(mask_logits),
                "anchored_logits": gate_out["anchored_logits"],
                "delta_z_fpn": gate_out["delta_z_fpn"],
                "g_unc": gate_out["g_unc"],
                "expert_logits": expert_logits,
                "calibrated_logits": gate_out["calibrated_logits"],
                "alpha_img": gate_out["alpha_img"],
                "spatial_weights": gate_out["spatial_weights"],
                "conf_probs": conf_probs,
                "det_logits": det_logits,
                "artifact_loss": reg_loss,
                "align_loss": reg_loss,
                **reg_metrics,
            }
            if class_logits is not None:
                res["class_logits"] = class_logits
                res["pred_label"] = torch.argmax(class_logits, dim=1)
            return res

        if self.aux_classifier and class_logits is not None:
            return mask_logits, class_logits
        return mask_logits
