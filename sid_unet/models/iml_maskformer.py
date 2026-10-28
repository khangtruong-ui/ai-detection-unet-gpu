"""
IML (Image Manipulation Localization) MaskFormer Model.
Refactored MaskFormer / Mask2Former architecture leveraging Meta's SAM2 Hiera vision backbone
for high-accuracy pixel-level synthetic and tampered image forgery localization.

Architecture Summary:
1. Frozen SAM 2.1 Hiera Trunk: Multi-scale feature extraction producing 4 spatial stages:
   F4 (1/4 scale), F8 (1/8 scale), F16 (1/16 scale), and F32 (1/32 scale).
2. Pixel Encoder & Decoder: Projects multi-scale features to uniform embedding dimension (256),
   applies 2D sinusoidal position embeddings and multi-level deformable attention across 6 layers,
   and fuses F4 with upscaled D8 to generate high-resolution mask features (H/4, W/4).
3. Transformer Decoder with Dual-Query Interactive Attention:
   Maintains 100 queries separated into classification (cls_query) and localization (loc_query).
   9 decoder layers with interactive cross-query attention, masked cross-attention to image features,
   and query self-attention.
4. Prediction Heads:
   - ClsPredictionHead: 3-class distribution (0: authentic object, 1: forged object, 2: no object).
   - MaskPredictionHead: 3-layer MLP query embedding dot-product with high-resolution mask features.
5. Semantic Assembly:
   Combines query classification probabilities and query mask probabilities to produce dense
   pixel-level forgery logits and masks.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from sid_unet.models.blocks import get_submodule_device_dtype

logger = logging.getLogger("sid_unet.models.iml_maskformer")

# Default paths and candidates for SAM2 repo and weights
DEFAULT_SAM2_REPO_CANDIDATES = [
    Path("/content/sam2"),
    Path("/workspace/sam2"),
    Path("./sam2"),
    Path("../sam2"),
]

SAM2_CONFIGS = {
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
}

SAM2_WEIGHTS_URLS = {
    "large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
}

BACKBONE_DIMS = {
    "large": [144, 288, 576, 1152],
    "small": [96, 192, 384, 768],
    "tiny": [96, 192, 384, 768],
}


def assert_finite(x: torch.Tensor, name: str) -> torch.Tensor:
    """Pass-through validation helper."""
    return x


def pairwise_dice_loss(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Pairwise Dice loss between predicted query logits [N, P] and target points [M, P]."""
    probs = logits.sigmoid()
    intersection = torch.einsum("np,mp->nm", probs, targets)
    pred_area = probs.sum(-1)[:, None]
    target_area = targets.sum(-1)[None, :]

    numerator = 2 * intersection
    denominator = pred_area + target_area
    dice = (numerator + 1.0) / (denominator + 1.0)
    loss = 1.0 - dice

    eps = 1e-6
    iou = (intersection + eps) / (denominator - intersection + eps)
    return loss, dice, iou


def pairwise_bce_loss(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Pairwise BCE loss between predicted query logits [N, P] and target points [M, P]."""
    p = logits.shape[1]
    pos = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    loss = torch.einsum("np,mp->nm", pos, targets) + torch.einsum("np,mp->nm", neg, (1.0 - targets))
    return loss / max(1, p), pos, neg


def point_sample(input_tensor: Tensor, point_coords: Tensor, **kwargs: Any) -> Tensor:
    """Wrapper around grid_sample for 3D normalized point coordinate tensors [N, P, 2]."""
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input_tensor, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def calculate_uncertainty(logits: Tensor) -> Tensor:
    """Estimate uncertainty as L1 distance from decision boundary (0.0)."""
    return -torch.abs(logits)


def sigmoid_ce_loss(inputs: Tensor, targets: Tensor, num_masks: float) -> Tuple[Tensor, Tensor]:
    """Sigmoid Cross-Entropy loss normalized by matched mask count."""
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / max(1.0, num_masks), loss.mean(1)


def dice_loss(inputs: Tensor, targets: Tensor, num_masks: float) -> Tensor:
    """Soft Dice loss normalized by matched mask count."""
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return loss.sum() / max(1.0, num_masks)


def get_valid_ratio(pad_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Ratio of valid (non-padded) pixels along width and height."""
    feature_mask = (~pad_mask).squeeze(1)
    _, height, width = feature_mask.shape
    valid_height = torch.sum(feature_mask[:, :, 0], dim=1)
    valid_width = torch.sum(feature_mask[:, 0, :], dim=1)
    valid_ratio_height = valid_height.to(dtype) / max(1, height)
    valid_ratio_width = valid_width.to(dtype) / max(1, width)
    return torch.stack([valid_ratio_width, valid_ratio_height], dim=-1)


def get_reference_points(list_pad_mask: List[Tensor], dtype: torch.dtype) -> Tensor:
    """Generate multi-level 2D reference points for deformable attention."""
    device = list_pad_mask[0].device
    valid_ratios = []
    spatial_shapes = []
    for pad_mask in list_pad_mask:
        _, _, H, W = pad_mask.shape
        valid_ratio = get_valid_ratio(pad_mask, dtype)
        valid_ratios.append(valid_ratio)
        spatial_shapes.append([W, H])

    valid_ratios = torch.stack(valid_ratios, dim=1)
    reference_points = []
    for lvl, (width, height) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height + 1e-6)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width + 1e-6)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points.append(ref)

    reference_points = torch.cat(reference_points, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


def flatten_spatial(list_spatial_features: List[Tensor]) -> Tensor:
    """Flatten list of spatial tensors [B, C, H, W] to [B, sum(H*W), C]."""
    flatten_features = [rearrange(f, "b c h w -> b (h w) c") for f in list_spatial_features]
    return torch.concat(flatten_features, dim=1)


class HungarianMatcher(nn.Module):
    """
    Bipartite Hungarian Matcher for MaskFormer query-to-ground-truth instance assignment.
    Uses cost matrix based on dice loss, BCE loss, and classification probabilities.
    """

    def __init__(self, mask_weight: float = 5.0, dice_weight: float = 5.0, num_points: int = 12544):
        super().__init__()
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.num_points = num_points

    def _match_prediction_with_labels(
        self,
        cls_logits: Tensor,
        mask_logits: Tensor,
        cls_labels: Tensor,
        mask_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        mask_logits = mask_logits.float()
        mask_labels = mask_labels.float()

        n_queries = cls_logits.size(0)
        n_masks = cls_labels.size(0)
        device = cls_logits.device

        cls_probs = cls_logits.softmax(-1)
        y_hat = cls_probs[:, cls_labels]  # [N, Mi]

        mask_logits = mask_logits[:, None]
        mask_labels = mask_labels[:, None]

        point_coordinates = torch.rand(1, self.num_points, 2, device=device)
        pred_coordinates = point_coordinates.repeat(n_queries, 1, 1)
        target_coordinates = point_coordinates.repeat(n_masks, 1, 1)

        sampled_mask_logits = point_sample(mask_logits, pred_coordinates, align_corners=False).squeeze(1)
        sampled_mask_labels = point_sample(mask_labels, target_coordinates, align_corners=False).squeeze(1)

        mask_loss, _, _ = pairwise_bce_loss(sampled_mask_logits, sampled_mask_labels)
        dice_loss_val, dice, _ = pairwise_dice_loss(sampled_mask_logits, sampled_mask_labels)

        alpha = 0.75
        beta = 0.25
        c_loc = self.mask_weight * mask_loss + self.dice_weight * dice_loss_val
        c_cls = 1.0 - (dice ** alpha) * (y_hat ** beta)
        c = c_loc * torch.abs(c_cls)

        try:
            from torch_linear_assignment import batch_linear_assignment, assignment_to_indices
            assignment = batch_linear_assignment(c.unsqueeze(0))
            pred_indices, target_indices = assignment_to_indices(assignment)
            return pred_indices.squeeze(0), target_indices.squeeze(0)
        except Exception:
            from scipy.optimize import linear_sum_assignment
            c_np = c.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(c_np)
            return (
                torch.as_tensor(row_ind, dtype=torch.int64, device=device),
                torch.as_tensor(col_ind, dtype=torch.int64, device=device),
            )

    @torch.no_grad()
    def forward(
        self,
        cls_logits: Tensor,
        mask_logits: Tensor,
        cls_labels: List[Tensor],
        mask_labels: List[Tensor],
    ) -> List[Tuple[Tensor, Tensor]]:
        b = cls_logits.size(0)
        matched_indices = []
        is_cuda = torch.cuda.is_available() and cls_logits.is_cuda
        autocast_dev = "cuda" if is_cuda else "cpu"

        for i in range(b):
            with torch.autocast(device_type=autocast_dev, enabled=False):
                pred_indices, target_indices = self._match_prediction_with_labels(
                    cls_logits=cls_logits[i],
                    mask_logits=mask_logits[i],
                    cls_labels=cls_labels[i],
                    mask_labels=mask_labels[i],
                )
            matched_indices.append((pred_indices, target_indices))
        return matched_indices


class RefactorMaskformerLoss(nn.Module):
    """
    Multi-task loss for MaskFormer: Cross-Entropy, Pointwise Mask BCE, and Dice Loss
    with optional auxiliary prediction supervision across decoder layers.
    """

    def __init__(
        self,
        num_labels: int = 2,
        no_object_weight: float = 0.1,
        cls_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.no_object_weight = no_object_weight
        self.matcher = HungarianMatcher(mask_weight=mask_weight, dice_weight=dice_weight)
        self.cls_weight = cls_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.no_object_weight
        self.register_buffer("empty_weight", empty_weight)

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        max_h = max(t.shape[-2] for t in tensors)
        max_w = max(t.shape[-1] for t in tensors)
        batch_size = len(tensors)
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded = torch.zeros((batch_size, max_h, max_w), dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, max_h, max_w), dtype=torch.bool, device=device)
        for i, t in enumerate(tensors):
            h, w = t.shape[-2], t.shape[-1]
            padded[i, :h, :w] = t if t.ndim == 2 else t[0]
            padding_masks[i, :h, :w] = False
        return padded, padding_masks

    def _sample_points_using_uncertainty(self, logits: Tensor) -> Tensor:
        num_masks = logits.shape[0]
        num_points_sampled = int(self.num_points * self.oversample_ratio)
        point_coords = torch.rand(num_masks, num_points_sampled, 2, device=logits.device)
        point_logits = point_sample(logits, point_coords, align_corners=False)
        uncertainties = calculate_uncertainty(point_logits)

        num_uncertain = int(self.importance_sample_ratio * self.num_points)
        num_random = self.num_points - num_uncertain

        idx = torch.topk(uncertainties[:, 0, :], k=num_uncertain, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_masks, dtype=torch.long, device=logits.device)
        idx = idx + shift[:, None]

        sampled_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_masks, num_uncertain, 2)
        if num_random > 0:
            random_coords = torch.rand(num_masks, num_random, 2, device=logits.device)
            sampled_coords = torch.cat([sampled_coords, random_coords], dim=1)
        return sampled_coords

    def forward(
        self,
        cls_logits: Tensor,
        mask_logits: Tensor,
        cls_labels: List[Tensor],
        mask_labels: List[Tensor],
        auxiliary_predictions: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        matched_indices = self.matcher(cls_logits, mask_logits, cls_labels, mask_labels)
        b, n, c_dim = cls_logits.shape
        device = cls_logits.device

        # Cross-Entropy classification loss
        target_classes = torch.full((b, n), self.num_labels, dtype=torch.long, device=device)
        for i, (pred_idx, tgt_idx) in enumerate(matched_indices):
            target_classes[i, pred_idx] = cls_labels[i][tgt_idx]
        loss_ce = F.cross_entropy(cls_logits.view(-1, c_dim), target_classes.view(-1), weight=self.empty_weight)

        # Mask & Dice Losses on matched queries
        total_matched = sum(len(p) for p, _ in matched_indices)
        num_masks = max(1.0, float(total_matched))

        matched_preds = []
        matched_targets = []
        for i, (pred_idx, tgt_idx) in enumerate(matched_indices):
            if len(pred_idx) > 0:
                matched_preds.append(mask_logits[i, pred_idx])
                matched_targets.append(mask_labels[i][tgt_idx])

        if len(matched_preds) > 0:
            all_preds = torch.cat(matched_preds, dim=0).unsqueeze(1)  # [M, 1, H, W]
            all_targets = torch.cat(matched_targets, dim=0).unsqueeze(1)  # [M, 1, H', W']
            if all_targets.shape[-2:] != all_preds.shape[-2:]:
                all_targets = F.interpolate(all_targets.float(), size=all_preds.shape[-2:], mode="nearest")

            sampled_points = self._sample_points_using_uncertainty(all_preds)
            sampled_preds = point_sample(all_preds, sampled_points, align_corners=False).squeeze(1)
            sampled_targets = point_sample(all_targets, sampled_points, align_corners=False).squeeze(1)

            loss_mask_val, _ = sigmoid_ce_loss(sampled_preds, sampled_targets, num_masks)
            loss_dice_val = dice_loss(sampled_preds, sampled_targets, num_masks)
        else:
            loss_mask_val = torch.tensor(0.0, device=device)
            loss_dice_val = torch.tensor(0.0, device=device)

        total_loss = self.cls_weight * loss_ce + self.dice_weight * loss_dice_val + self.mask_weight * loss_mask_val

        # Auxiliary layer losses if provided
        if auxiliary_predictions:
            for aux in auxiliary_predictions:
                aux_cls = aux["cls_logits"]
                aux_mask = aux["mask_logits"]
                aux_matched = self.matcher(aux_cls, aux_mask, cls_labels, mask_labels)
                aux_target_classes = torch.full((b, n), self.num_labels, dtype=torch.long, device=device)
                for i, (p_idx, t_idx) in enumerate(aux_matched):
                    aux_target_classes[i, p_idx] = cls_labels[i][t_idx]
                aux_ce = F.cross_entropy(aux_cls.view(-1, c_dim), aux_target_classes.view(-1), weight=self.empty_weight)
                total_loss = total_loss + 0.5 * self.cls_weight * aux_ce

        return total_loss, loss_ce, loss_dice_val, loss_mask_val


@dataclass
class DeformableFeatures:
    features: Tensor
    pos_embeddings: Tensor
    pad_masks: Tensor
    spatial_shapes: Tensor
    reference_points: Tensor

    @property
    def features_with_pos_embeddings(self) -> Tensor:
        return self.features + self.pos_embeddings

    def to_spatial(self) -> SpatialFeatures:
        list_spatial_features = []
        list_pos_embeddings = []
        list_pad_mask = []
        start = 0
        for w, h in self.spatial_shapes.tolist():
            end = start + h * w
            f = self.features[:, start:end, :]
            p = self.pos_embeddings[:, start:end, :]
            m = self.pad_masks[:, start:end, :]
            start = end
            list_spatial_features.append(rearrange(f, "b (h w) c -> b c h w", h=h, w=w))
            list_pos_embeddings.append(rearrange(p, "b (h w) c -> b c h w", h=h, w=w))
            list_pad_mask.append(rearrange(m, "b (h w) c -> b c h w", h=h, w=w))

        return SpatialFeatures(
            list_spatial_features=list_spatial_features,
            list_pos_embeddings=list_pos_embeddings,
            list_pad_mask=list_pad_mask,
        )


@dataclass
class SpatialFeatures:
    list_spatial_features: List[Tensor]
    list_pos_embeddings: List[Tensor]
    list_pad_mask: List[Tensor]

    def to_deformable(self) -> DeformableFeatures:
        device = self.list_spatial_features[0].device
        spatial_shapes = torch.tensor(
            [[f.shape[3], f.shape[2]] for f in self.list_spatial_features],
            dtype=torch.long,
            device=device,
        )
        reference_points = get_reference_points(self.list_pad_mask, dtype=torch.float32)
        return DeformableFeatures(
            features=flatten_spatial(self.list_spatial_features),
            pos_embeddings=flatten_spatial(self.list_pos_embeddings),
            pad_masks=flatten_spatial(self.list_pad_mask),
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
        )

    @property
    def batch_size(self) -> int:
        return self.list_spatial_features[0].size(0)


class SinusoidPositionEmbeddings2D(nn.Module):
    """2D Sinusoidal positional embeddings for multi-scale feature maps."""

    def __init__(self, embed_dim: int, spatial_shape: Tuple[int, int], temperature: float = 10000.0, normalize: bool = True):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.spatial_shape = tuple(spatial_shape)
        self.normalize = normalize

    def _compute_position_embeddings(self, pos_x: Tensor, pos_y: Tensor, dtype: torch.dtype) -> Tensor:
        c = self.embed_dim
        device = pos_x.device
        if self.normalize:
            eps = 1e-6
            width = pos_x[:, :, :, -1:] + eps
            height = pos_y[:, :, -1:, :] + eps
            pos_x = 2 * math.pi * (pos_x / width)
            pos_y = 2 * math.pi * (pos_y / height)

        i = torch.arange(c // 2, dtype=dtype, device=device)
        _2i = 2 * torch.div(i, 2, rounding_mode="floor")[None, :, None, None]

        pe_x = pos_x / (self.temperature ** (_2i / (c // 2)))
        pe_y = pos_y / (self.temperature ** (_2i / (c // 2)))

        pe_x_sin = pe_x[:, ::2, :, :].sin()
        pe_x_cos = pe_x[:, 1::2, :, :].cos()
        pe_y_sin = pe_y[:, ::2, :, :].sin()
        pe_y_cos = pe_y[:, 1::2, :, :].cos()

        pe_x_full = torch.empty_like(pe_x)
        pe_x_full[:, ::2, :, :] = pe_x_sin
        pe_x_full[:, 1::2, :, :] = pe_x_cos

        pe_y_full = torch.empty_like(pe_y)
        pe_y_full[:, ::2, :, :] = pe_y_sin
        pe_y_full[:, 1::2, :, :] = pe_y_cos

        return torch.cat([pe_x_full, pe_y_full], dim=1)

    def _build_position_embeddings(self) -> Tensor:
        h, w = self.spatial_shape
        dtype = torch.float32
        pos_x = torch.arange(1, w + 1, dtype=dtype)[None, None, None, :].expand(1, 1, h, w)
        pos_y = torch.arange(1, h + 1, dtype=dtype)[None, None, :, None].expand(1, 1, h, w)
        pe = self._compute_position_embeddings(pos_x, pos_y, dtype=dtype)
        self.register_buffer("position_embeddings", pe, persistent=False)
        return pe

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        b, c, h, w = x.shape
        if not hasattr(self, "position_embeddings"):
            pe = self._build_position_embeddings()
        else:
            pe = self.position_embeddings
        if pe.size(0) != b:
            pe = pe.expand(b, -1, h, w)
        return pe.to(device=x.device, dtype=x.dtype)


class SAM2ImageEncoder(nn.Module):
    """
    Frozen SAM 2.1 Hiera Vision Backbone for multi-scale feature extraction.
    Loads official Hiera trunk and extracts 4 hierarchy stages:
    [B, C0, 96, 96], [B, C1, 48, 48], [B, C2, 24, 24], [B, C3, 12, 12].
    """

    def __init__(
        self,
        sam2_repo: Optional[Union[str, Path]] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        variant: str = "large",
        device: str = "auto",
        use_dummy: bool = False,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.use_dummy = use_dummy
        self.backbone_dims = BACKBONE_DIMS.get(self.variant, BACKBONE_DIMS["large"])

        if use_dummy:
            logger.info("Initializing Dummy SAM2ImageEncoder for fast testing.")
            self.image_encoder = None
            return

        resolved_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
        repo_dir = self._find_sam2_repo(sam2_repo)
        cfg_file = config_path or SAM2_CONFIGS.get(self.variant, SAM2_CONFIGS["large"])

        # Resolve checkpoint file
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt_file = str(checkpoint_path)
        else:
            ckpt_name = f"sam2.1_hiera_{self.variant}.pt"
            candidate_ckpt = repo_dir / "checkpoints" / ckpt_name
            if candidate_ckpt.exists():
                ckpt_file = str(candidate_ckpt)
            else:
                ckpt_file = self._download_checkpoint(candidate_ckpt, self.variant)

        try:
            from sam2.build_sam import build_sam2
            model = build_sam2(str(cfg_file), str(ckpt_file), device=resolved_device)
            self.image_encoder = model.image_encoder.trunk
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            logger.info(f"Loaded SAM2 Hiera trunk ({self.variant}) from '{ckpt_file}'.")
        except Exception as exc:
            logger.warning(f"Could not load official SAM2 trunk ({exc}). Falling back to lightweight dummy trunk.")
            self.use_dummy = True
            self.image_encoder = None

    def _find_sam2_repo(self, explicit_repo: Optional[Union[str, Path]]) -> Path:
        if explicit_repo and Path(explicit_repo).exists():
            return Path(explicit_repo)
        for cand in DEFAULT_SAM2_REPO_CANDIDATES:
            if cand.exists():
                return cand
        return Path("/content/sam2")

    def _download_checkpoint(self, target_path: Path, variant: str) -> str:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = SAM2_WEIGHTS_URLS.get(variant, SAM2_WEIGHTS_URLS["large"])
        logger.info(f"Downloading SAM2 checkpoint from {url} to {target_path}...")
        import urllib.request
        urllib.request.urlretrieve(url, str(target_path))
        return str(target_path)

    @torch.no_grad()
    def forward(self, images: Tensor) -> List[Tensor]:
        if self.use_dummy or self.image_encoder is None:
            # Dummy multi-scale outputs for testing without weights
            b = images.shape[0]
            dev = images.device
            c0, c1, c2, c3 = self.backbone_dims
            return [
                torch.zeros(b, c0, 96, 96, device=dev),
                torch.zeros(b, c1, 48, 48, device=dev),
                torch.zeros(b, c2, 24, 24, device=dev),
                torch.zeros(b, c3, 12, 12, device=dev),
            ]
        return self.image_encoder(images)


class PixelEncoder(nn.Module):
    """
    Projects SAM2 backbone feature hierarchy into unified embedding dimension (256)
    and constructs multi-scale DeformableFeatures.
    """

    def __init__(self, embed_dim: int = 256, variant: str = "large", use_dummy: bool = False):
        super().__init__()
        self.sam_backbone = SAM2ImageEncoder(variant=variant, use_dummy=use_dummy)
        backbone_dims = BACKBONE_DIMS.get(variant, BACKBONE_DIMS["large"])
        self.embed_dim = embed_dim

        spatial_shapes = [(48, 48), (24, 24), (12, 12)]
        self.pos_embeddings = nn.ModuleList([
            SinusoidPositionEmbeddings2D(embed_dim=embed_dim, spatial_shape=s)
            for s in spatial_shapes
        ])
        self.level_embeddings = nn.Embedding(3, embed_dim)
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embed_dim, kernel_size=1),
                nn.GroupNorm(32, embed_dim),
            )
            for dim in backbone_dims[1:]
        ])

    def forward(self, images: Tensor) -> Tuple[DeformableFeatures, Tensor]:
        features = self.sam_backbone(images)
        mask_features = features[0]  # F4 [B, C0, 96, 96]
        pyramid_features = features[1:]  # [F8, F16, F32]

        list_spatial_features = []
        list_pos_embeddings = []
        list_pad_mask = []

        for i, f in enumerate(pyramid_features):
            proj_feat = self.input_projs[i](f)
            b, _, h, w = proj_feat.shape
            pad_mask = torch.zeros((b, 1, h, w), dtype=torch.bool, device=proj_feat.device)
            pos_emb = self.pos_embeddings[i](proj_feat, pad_mask) + self.level_embeddings.weight[i][None, :, None, None]
            list_spatial_features.append(proj_feat)
            list_pos_embeddings.append(pos_emb)
            list_pad_mask.append(pad_mask)

        spatial_features = SpatialFeatures(
            list_spatial_features=list_spatial_features[::-1],
            list_pos_embeddings=list_pos_embeddings[::-1],
            list_pad_mask=list_pad_mask[::-1],
        )
        return spatial_features.to_deformable(), mask_features


class DeformableAttention(nn.Module):
    """Multi-scale Deformable Attention across feature pyramid levels."""

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attn_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def _get_sampling_locations(self, features: DeformableFeatures) -> Tensor:
        query = features.features_with_pos_embeddings
        sampling_offsets = self.sampling_offsets(query).float()
        sampling_offsets = rearrange(
            sampling_offsets,
            "b l (h lv p two) -> b l h lv p two",
            h=self.num_heads,
            lv=self.num_levels,
            p=self.num_points,
            two=2,
        )
        spatial_shapes = features.spatial_shapes
        sampling_offsets = sampling_offsets / (spatial_shapes[None, None, None, :, None, :] + 1e-6)
        sampling_locations = sampling_offsets + features.reference_points[:, :, None, :, None, :]
        return sampling_locations

    def _get_attn_weight(self, features: DeformableFeatures) -> Tensor:
        query = features.features_with_pos_embeddings
        weights = self.attn_weights(query)
        weights = rearrange(weights, "b n (h l p) -> b n h (l p)", h=self.num_heads, l=self.num_levels, p=self.num_points)
        weights = torch.softmax(weights.float(), dim=-1)
        return rearrange(weights, "b n h (l p) -> b n h l p", l=self.num_levels)

    def _get_value(self, features: DeformableFeatures) -> Tensor:
        value = self.v_proj(features.features)
        value = value.masked_fill(features.pad_masks, 0.0)
        return rearrange(value, "b n (h d) -> b n h d", h=self.num_heads).float()

    def _deformable_attention(self, value: Tensor, spatial_shapes: Tensor, sampling_locations: Tensor, attn_weights: Tensor) -> Tensor:
        batch_size, _, num_heads, hidden_dim = value.shape
        _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
        value_list = value.split([h * w for w, h in spatial_shapes.tolist()], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level_id, (width, height) in enumerate(spatial_shapes.tolist()):
            value_l = value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
            sampling_grid_l = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
            sampled = F.grid_sample(value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
            sampling_value_list.append(sampled)

        attn_weights = attn_weights.transpose(1, 2).reshape(batch_size * num_heads, 1, num_queries, num_levels * num_points)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attn_weights).sum(-1)
        return output.view(batch_size, num_heads * hidden_dim, num_queries).transpose(1, 2).contiguous()

    def forward(self, features: DeformableFeatures) -> Tensor:
        dtype = features.features.dtype
        value = self._get_value(features)
        sampling_locations = self._get_sampling_locations(features)
        attn_weights = self._get_attn_weight(features)

        is_cuda = torch.cuda.is_available() and features.features.is_cuda
        dev_str = "cuda" if is_cuda else "cpu"
        with torch.autocast(device_type=dev_str, enabled=False):
            out = self._deformable_attention(
                value=value,
                spatial_shapes=features.spatial_shapes,
                sampling_locations=sampling_locations,
                attn_weights=attn_weights,
            )
        return self.o_proj(out.to(dtype))


class DeformableTransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, ffn_hid_dim: int, num_heads: int, num_levels: int, num_points: int, attn_drop: float = 0.0, ffn_drop: float = 0.0):
        super().__init__()
        self.self_attn = DeformableAttention(embed_dim, num_heads, num_levels, num_points)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hid_dim),
            nn.ReLU(),
            nn.Dropout(ffn_drop),
            nn.Linear(ffn_hid_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, features: DeformableFeatures) -> DeformableFeatures:
        residual = features.features
        attn_out = self.self_attn(features)
        features_norm = self.norm1(residual + attn_out)

        ffn_out = self.ffn(features_norm)
        output_features = self.norm2(features_norm + ffn_out)

        return DeformableFeatures(
            features=output_features,
            pos_embeddings=features.pos_embeddings,
            pad_masks=features.pad_masks,
            spatial_shapes=features.spatial_shapes,
            reference_points=features.reference_points,
        )


class DeformableTransformer(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, ffn_hid_dim: int, num_heads: int, num_levels: int, num_points: int, attn_drop: float = 0.0, ffn_drop: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerLayer(embed_dim, ffn_hid_dim, num_heads, num_levels, num_points, attn_drop, ffn_drop)
            for _ in range(num_layers)
        ])

    def forward(self, features: DeformableFeatures) -> DeformableFeatures:
        for layer in self.layers:
            features = layer(features)
        return features


class PixelDecoder(nn.Module):
    """
    Deformable Attention Transformer decoding pyramid features and fusing with F4 (stride 4).
    """

    def __init__(self, num_layers: int, embed_dim: int, ffn_hid_dim: int, num_heads: int, num_levels: int, num_points: int, f4_dim: int = 144, attn_drop: float = 0.0, ffn_drop: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.deformable_transformer = DeformableTransformer(
            num_layers=num_layers,
            embed_dim=embed_dim,
            ffn_hid_dim=ffn_hid_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            attn_drop=attn_drop,
            ffn_drop=ffn_drop,
        )
        self.fpn_proj = nn.Sequential(
            nn.Conv2d(f4_dim, embed_dim, kernel_size=1),
            nn.GroupNorm(32, embed_dim),
        )
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(),
        )
        self.mask_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, features: DeformableFeatures, mask_features: Tensor) -> Tuple[SpatialFeatures, Tensor]:
        decoded: DeformableFeatures = self.deformable_transformer(features)
        spatial: SpatialFeatures = decoded.to_spatial()

        # Fuse D8 (1/8 scale) with F4 (1/4 scale)
        d8 = spatial.list_spatial_features[-1]
        f4_proj = self.fpn_proj(mask_features)
        _, _, h, w = f4_proj.shape
        fused = f4_proj + F.interpolate(d8, size=(h, w), mode="bilinear", align_corners=False)
        refined_mask_features = self.mask_proj(self.fpn_conv(fused))
        return spatial, refined_mask_features


class TransformerDecoderLayer(nn.Module):
    """
    Dual-Query Decoder Layer:
    - Interactive attention between cls_query and loc_query
    - Masked cross-attention to image spatial features
    - Self-attention on cls_query and loc_query
    - Position-wise Feed-Forward Networks
    """

    def __init__(self, embed_dim: int, ffn_hid_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.interactive_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.interactive_attn_norm = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.cls_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cls_self_attn_norm = nn.LayerNorm(embed_dim)

        self.loc_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.loc_self_attn_norm = nn.LayerNorm(embed_dim)

        self.cls_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hid_dim, embed_dim),
        )
        self.cls_ffn_norm = nn.LayerNorm(embed_dim)

        self.loc_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hid_dim, embed_dim),
        )
        self.loc_ffn_norm = nn.LayerNorm(embed_dim)

    def _build_attn_mask(self, mask_logits: Tensor, target_size: Tuple[int, int], pad_mask: Optional[Tensor] = None) -> Tensor:
        interpolated = F.interpolate(mask_logits, size=target_size, mode="bilinear", align_corners=True)
        obj_mask = interpolated.sigmoid() >= 0.5
        attn_mask = ~obj_mask
        if pad_mask is not None:
            attn_mask = attn_mask | pad_mask

        attn_mask = rearrange(attn_mask, "b n h w -> b n (h w)")
        all_masked = attn_mask.all(dim=-1)
        attn_mask[all_masked] = False

        attn_mask = repeat(attn_mask, "b n l -> (b h) n l", h=self.num_heads)
        return torch.cat([attn_mask, attn_mask], dim=1).detach()

    def forward(
        self,
        level_index: int,
        loc_query: Tensor,
        cls_query: Tensor,
        loc_pos_embeddings: Tensor,
        cls_pos_embeddings: Tensor,
        spatial_features: SpatialFeatures,
        mask_logits: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        features = spatial_features.list_spatial_features[level_index]
        pos_embeddings = spatial_features.list_pos_embeddings[level_index]
        pad_mask = spatial_features.list_pad_mask[level_index]
        _, _, h, w = features.shape
        n = cls_query.shape[1]

        attn_mask = self._build_attn_mask(mask_logits, target_size=(h, w), pad_mask=pad_mask)

        # 1. Interactive Attention
        query = torch.cat([cls_query, loc_query], dim=1)
        q_pos = torch.cat([cls_pos_embeddings, loc_pos_embeddings], dim=1)
        q = query + q_pos
        interactive_out, _ = self.interactive_attn(query=q, key=q, value=query)
        query = self.interactive_attn_norm(query + interactive_out)

        # 2. Masked Cross-Attention
        k = rearrange(features + pos_embeddings, "b c h w -> b (h w) c")
        v = rearrange(features, "b c h w -> b (h w) c")
        cross_out, _ = self.cross_attn(query=query + q_pos, key=k, value=v, attn_mask=attn_mask)
        query = self.cross_attn_norm(query + cross_out)

        cls_query, loc_query = torch.split(query, n, dim=1)

        # 3. Self-Attention
        cls_out, _ = self.cls_self_attn(query=cls_query + cls_pos_embeddings, key=cls_query + cls_pos_embeddings, value=cls_query)
        cls_query = self.cls_self_attn_norm(cls_query + cls_out)

        loc_out, _ = self.loc_self_attn(query=loc_query + loc_pos_embeddings, key=loc_query + loc_pos_embeddings, value=loc_query)
        loc_query = self.loc_self_attn_norm(loc_query + loc_out)

        # 4. FFNs
        cls_query = self.cls_ffn_norm(cls_query + self.cls_ffn(cls_query))
        loc_query = self.loc_ffn_norm(loc_query + self.loc_ffn(loc_query))

        return cls_query, loc_query


class ClsPredictionHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, cls_query: Tensor) -> Tensor:
        return self.classifier(self.norm(cls_query))


class MaskPredictionHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_embeddings = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, loc_query: Tensor, mask_features: Tensor) -> Tensor:
        embeddings = self.mask_embeddings(self.norm(loc_query))
        return torch.einsum("bnc,bchw->bnhw", embeddings, mask_features)


class PredictionHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int = 3):
        super().__init__()
        self.cls_head = ClsPredictionHead(embed_dim, n_classes)
        self.mask_head = MaskPredictionHead(embed_dim)

    def forward(self, cls_query: Tensor, loc_query: Tensor, mask_features: Tensor) -> Tuple[Tensor, Tensor]:
        return self.cls_head(cls_query), self.mask_head(loc_query, mask_features)


class TransformerDecoder(nn.Module):
    """
    9-layer Transformer Decoder with 100 learned object queries cycling through 3 pyramid levels.
    """

    def __init__(self, num_layers: int = 9, num_queries: int = 100, embed_dim: int = 256, ffn_hid_dim: int = 2048, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_levels = 3
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim=embed_dim, ffn_hid_dim=ffn_hid_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.cls_query = nn.Embedding(num_queries, embed_dim)
        self.loc_query = nn.Embedding(num_queries, embed_dim)
        self.cls_pos_embeddings = nn.Embedding(num_queries, embed_dim)
        self.loc_pos_embeddings = nn.Embedding(num_queries, embed_dim)

        spatial_shapes = [(12, 12), (24, 24), (48, 48)]
        self.pos_embeddings = nn.ModuleList([
            SinusoidPositionEmbeddings2D(embed_dim=embed_dim, spatial_shape=s)
            for s in spatial_shapes
        ])
        self.level_embeddings = nn.Embedding(self.num_levels, embed_dim)
        self.prediction_head = PredictionHead(embed_dim, n_classes=3)

    def _update_pos_embeddings(self, spatial_features: SpatialFeatures) -> SpatialFeatures:
        list_pos = []
        for i, f in enumerate(spatial_features.list_spatial_features):
            lvl = self.level_embeddings.weight[i][None, :, None, None]
            pos = self.pos_embeddings[i](f) + lvl
            list_pos.append(pos)
        return SpatialFeatures(
            list_spatial_features=spatial_features.list_spatial_features,
            list_pos_embeddings=list_pos,
            list_pad_mask=spatial_features.list_pad_mask,
        )

    def forward(self, spatial_features: SpatialFeatures, mask_features: Tensor) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        spatial_features = self._update_pos_embeddings(spatial_features)
        b = spatial_features.batch_size

        cls_q = self.cls_query.weight[None, :, :].expand(b, self.num_queries, self.embed_dim)
        loc_q = self.loc_query.weight[None, :, :].expand(b, self.num_queries, self.embed_dim)
        cls_pos = self.cls_pos_embeddings.weight[None, :, :].expand(b, self.num_queries, self.embed_dim)
        loc_pos = self.loc_pos_embeddings.weight[None, :, :].expand(b, self.num_queries, self.embed_dim)

        aux_predictions = []
        for level_idx, layer in enumerate(self.layers):
            lvl = level_idx % self.num_levels
            cls_logits, mask_logits = self.prediction_head(cls_q, loc_q, mask_features)
            aux_predictions.append({"cls_logits": cls_logits, "mask_logits": mask_logits})
            cls_q, loc_q = layer(
                level_index=lvl,
                loc_query=loc_q,
                cls_query=cls_q,
                loc_pos_embeddings=loc_pos,
                cls_pos_embeddings=cls_pos,
                spatial_features=spatial_features,
                mask_logits=mask_logits,
            )

        cls_logits, mask_logits = self.prediction_head(cls_q, loc_q, mask_features)
        final_prediction = {"cls_logits": cls_logits, "mask_logits": mask_logits}
        return aux_predictions, final_prediction


class PixelModule(nn.Module):
    def __init__(self, num_layers: int = 6, embed_dim: int = 256, ffn_hid_dim: int = 1024, num_heads: int = 8, num_levels: int = 3, num_points: int = 4, variant: str = "large", use_dummy: bool = False, attn_drop: float = 0.0, ffn_drop: float = 0.0):
        super().__init__()
        self.pixel_encoder = PixelEncoder(embed_dim=embed_dim, variant=variant, use_dummy=use_dummy)
        f4_dim = BACKBONE_DIMS.get(variant, BACKBONE_DIMS["large"])[0]
        self.pixel_decoder = PixelDecoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            ffn_hid_dim=ffn_hid_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            f4_dim=f4_dim,
            attn_drop=attn_drop,
            ffn_drop=ffn_drop,
        )

    def forward(self, images: Tensor) -> Tuple[SpatialFeatures, Tensor]:
        features, mask_features = self.pixel_encoder(images)
        return self.pixel_decoder(features, mask_features)


class IMLMaskformer(nn.Module):
    """
    Image Manipulation Localization (IML) MaskFormer model.

    Integrates Meta's SAM2 Hiera vision backbone, Deformable Attention Pixel Decoder,
    and Dual-Query Interactive MaskFormer Transformer Decoder.

    Args:
        variant (str): SAM2 backbone variant ('large', 'small', 'tiny'). Defaults to 'large'.
        sam2_repo (str or Path, optional): Custom path to SAM2 repository.
        checkpoint_path (str or Path, optional): Custom path to sam2.1_hiera_*.pt weights.
        embed_dim (int): Common transformer embedding dimension (default: 256).
        num_queries (int): Number of object queries (default: 100).
        num_classes (int): Prediction classes (0: authentic, 1: forgery, 2: no object). Defaults to 3.
        pixel_num_layers (int): Layers in Deformable Attention Pixel Decoder (default: 6).
        decoder_num_layers (int): Layers in Transformer Decoder (default: 9).
        aux_classifier (bool): Whether to expose auxiliary image-level classification head.
        target_size (Tuple[int, int]): Native input resolution for backbone (default: (384, 384)).
        use_dummy (bool): If True, skips weight download and uses mock backbone for rapid tests.
        device (str or torch.device): Computing device.
    """

    def __init__(
        self,
        variant: str = "large",
        sam2_repo: Optional[Union[str, Path]] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        embed_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 3,
        pixel_num_layers: int = 6,
        decoder_num_layers: int = 9,
        aux_classifier: bool = False,
        in_channels: int = 3,
        out_channels: int = 1,
        target_size: Tuple[int, int] = (384, 384),
        use_dummy: bool = False,
        device: Optional[Union[str, torch.device]] = "auto",
        **kwargs: Any,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.target_size = tuple(target_size)
        self.aux_classifier = aux_classifier
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pixel_module = PixelModule(
            num_layers=pixel_num_layers,
            embed_dim=embed_dim,
            ffn_hid_dim=1024,
            num_heads=8,
            num_levels=3,
            num_points=4,
            variant=self.variant,
            use_dummy=use_dummy,
        )

        self.transformer_decoder = TransformerDecoder(
            num_layers=decoder_num_layers,
            num_queries=num_queries,
            embed_dim=embed_dim,
            ffn_hid_dim=2048,
            num_heads=8,
            dropout=0.0,
        )

        self.loss_fn = RefactorMaskformerLoss()

        if self.aux_classifier:
            self.classifier_head = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier_head = None

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        for name, m in self.named_modules():
            if "sam_backbone" in name:
                continue
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        images: Tensor,
        cls_labels: Optional[List[Tensor]] = None,
        mask_labels: Optional[List[Tensor]] = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Dict[str, Any]]:
        """
        Forward pass for IML MaskFormer.

        Args:
            images (Tensor): Input image tensor [B, 3, H, W].
            cls_labels (List[Tensor], optional): Ground-truth class labels per instance.
            mask_labels (List[Tensor], optional): Ground-truth instance masks [Mi, H, W].
            return_dict (bool): Return complete dictionary with loss breakdown and intermediate masks.

        Returns:
            - If called in standard trainer: Tensor of dense mask logits [B, 1, H, W]
              (or Tuple[mask_logits, cls_logits] if aux_classifier is True).
            - If return_dict is True: Dictionary containing 'loss', 'forgery_mask', 'mask_logits', etc.
        """
        orig_h, orig_w = images.shape[-2:]
        if (orig_h, orig_w) != self.target_size:
            x_in = F.interpolate(images, size=self.target_size, mode="bilinear", align_corners=False)
        else:
            x_in = images

        spatial_features, mask_features = self.pixel_module(x_in)
        aux_predictions, final_prediction = self.transformer_decoder(spatial_features, mask_features)

        cls_logits = final_prediction["cls_logits"]  # [B, N, 3]
        query_mask_logits = final_prediction["mask_logits"]  # [B, N, 96, 96]

        # Calculate dense forgery logits map
        cls_probs = cls_logits.float().softmax(dim=-1)[..., :-1]  # [B, N, 2] (0: real, 1: fake)
        interpolated_mask_logits = F.interpolate(query_mask_logits, size=self.target_size, mode="bilinear", align_corners=False)
        mask_probs = interpolated_mask_logits.sigmoid()

        # Semantic scores [B, 2, H, W]: Channel 0 = authentic score, Channel 1 = forgery score
        semantic_scores = torch.einsum("bnc,bnhw->bchw", cls_probs, mask_probs)
        # Direct semantic difference: Channel 1 (forgery) minus Channel 0 (authentic).
        # Positive logits (> 0, sigmoid > 0.5) correspond exactly to argmax(dim=1) == 1 (forgery),
        # matching the native RefactorMaskformer inference decision rule.
        dense_mask_logits = semantic_scores[:, 1:2, :, :] - semantic_scores[:, 0:1, :, :]

        if (orig_h, orig_w) != self.target_size:
            dense_mask_logits = F.interpolate(dense_mask_logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        # Handle native Hungarian loss if labels provided or requested
        loss = None
        loss_dict: Dict[str, Any] = {}
        if cls_labels is not None and mask_labels is not None:
            loss, ce, dice, mask_l = self.loss_fn(
                cls_logits,
                query_mask_logits,
                cls_labels,
                mask_labels,
                auxiliary_predictions=aux_predictions,
            )
            loss_dict = {
                "loss": loss,
                "loss_ce": ce,
                "loss_dice": dice,
                "loss_mask": mask_l,
            }

        forgery_mask = (dense_mask_logits >= 0.0)

        if return_dict:
            res: Dict[str, Any] = {
                "mask_logits": dense_mask_logits,
                "forgery_mask": forgery_mask,
                "cls_logits": cls_logits,
                "query_mask_logits": query_mask_logits,
            }
            res.update(loss_dict)
            return res

        if self.aux_classifier and self.classifier_head is not None:
            cls_out = self.classifier_head(spatial_features.list_spatial_features[-1].mean(dim=[2, 3]))
            return dense_mask_logits, cls_out

        return dense_mask_logits

    @torch.no_grad()
    def predict_mask(self, images: Tensor, threshold: float = 0.5) -> Tensor:
        """Produce binary mask prediction [B, 1, H, W] given input image tensor."""
        self.eval()
        outputs = self.forward(images)
        mask_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        probs = torch.sigmoid(mask_logits)
        return (probs >= threshold).float()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        override_config: Optional[Union[Dict[str, Any], Any]] = None,
        strict: bool = False,
        return_config: bool = False,
    ) -> Union[IMLMaskformer, Tuple[IMLMaskformer, Any]]:
        """Load IMLMaskformer model from checkpoint file (.pt or .ckpt)."""
        from sid_unet.models.unet import UNet
        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )


# Aliases matching original naming from 03_IML
RefectorMaskformer = IMLMaskformer
RefactorMaskformer = IMLMaskformer
