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

        # 2. Tiny trainable calibration & two-level gate (< 1.5M params)
        self.gate = ForensicMoEGate(
            num_experts=3,
            spectral_hidden_dims=tuple(int(v) for v in spectral_hidden_dims),
            spectral_embed_dim=int(spectral_embed_dim),
            spatial_hidden_dim=int(spatial_hidden_dim),
            spatial_num_blocks=int(spatial_num_blocks),
            gate_temperature=gate_temperature,
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
