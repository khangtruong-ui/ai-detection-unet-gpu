"""
Segmentation metrics: IoU, Dice / Pixel F1, AUROC, Pixel Accuracy, Precision, Recall, and Specificity.
Includes robust per-sample edge case handling and running accumulators.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_binary_auroc(
    pred_probs: Union[torch.Tensor, np.ndarray],
    target_mask: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic (AUROC) curve on continuous pixel probabilities.
    Robustly handles single-class masks (all-authentic or all-tampered).
    """
    if isinstance(pred_probs, torch.Tensor):
        p = pred_probs.detach().cpu().numpy().astype(np.float64)
    else:
        p = np.asarray(pred_probs, dtype=np.float64)

    if isinstance(target_mask, torch.Tensor):
        t = target_mask.detach().cpu().numpy().astype(np.int32)
    else:
        t = np.asarray(target_mask, dtype=np.int32)

    p_flat = p.flatten()
    t_flat = (t >= 0.5).flatten().astype(np.int32)

    n_pos = int(np.sum(t_flat))
    n_neg = len(t_flat) - n_pos

    # If all pixels are negative (authentic image)
    if n_pos == 0:
        mean_p = float(np.mean(p_flat))
        return 1.0 if mean_p < 0.5 else max(0.0, 1.0 - mean_p)

    # If all pixels are positive (fully synthetic image)
    if n_neg == 0:
        mean_p = float(np.mean(p_flat))
        return 1.0 if mean_p >= 0.5 else max(0.0, mean_p)

    try:
        return float(roc_auc_score(t_flat, p_flat))
    except Exception:
        # Fallback to Mann-Whitney U calculation
        order = np.argsort(p_flat)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(p_flat)) + 1
        pos_rank_sum = np.sum(ranks[t_flat == 1])
        u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))


def compute_binary_metrics(
    pred_mask: Union[torch.Tensor, np.ndarray],
    target_mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute binary segmentation metrics for a single sample or batch,
    including Pixel F1 score and AUROC.
    pred_mask: Float tensor/array with probabilities or binary values.
    target_mask: Float/Int tensor/array in {0, 1}.
    """
    if isinstance(pred_mask, torch.Tensor):
        p_raw = pred_mask.detach().cpu().numpy()
        p = (pred_mask >= threshold).cpu().numpy().astype(np.bool_)
    else:
        p_raw = np.asarray(pred_mask)
        p = (pred_mask >= threshold).astype(np.bool_)

    if isinstance(target_mask, torch.Tensor):
        t_raw = target_mask.detach().cpu().numpy()
        t = (target_mask >= threshold).cpu().numpy().astype(np.bool_)
    else:
        t_raw = np.asarray(target_mask)
        t = (target_mask >= threshold).astype(np.bool_)

    p_flat = p.flatten()
    t_flat = t.flatten()

    tp = np.sum(p_flat & t_flat)
    fp = np.sum(p_flat & (~t_flat))
    fn = np.sum((~p_flat) & t_flat)
    tn = np.sum((~p_flat) & (~t_flat))
    total_pixels = float(len(t_flat))

    # Pixel Accuracy
    pixel_acc = (tp + tn) / max(total_pixels, 1.0)

    # Intersection over Union (IoU) & Dice / Pixel F1
    union = tp + fp + fn
    if union == 0:
        # Both prediction and target are entirely background (e.g. authentic image with perfect prediction)
        iou = 1.0
        dice = 1.0
    else:
        iou = float(tp) / float(union)
        dice = (2.0 * float(tp)) / float(2 * tp + fp + fn)

    pixel_f1 = dice

    # Precision, Recall, Specificity
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 1.0

    # AUROC
    auroc = compute_binary_auroc(p_raw, t_raw)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(pixel_f1),
        "pixel_f1": float(pixel_f1),
        "auroc": float(auroc),
        "pixel_auroc": float(auroc),
        "pixel_acc": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


class SegmentationMetricTracker:
    """Accumulates and computes dataset-wide aggregated and per-label segmentation metrics including F1 and AUROC."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.metrics_sum: Dict[str, float] = {
            "iou": 0.0,
            "dice": 0.0,
            "f1": 0.0,
            "pixel_f1": 0.0,
            "auroc": 0.0,
            "pixel_auroc": 0.0,
            "pixel_acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
        }
        # Per-label accumulator
        self.per_label_stats: Dict[int, Dict[str, Union[float, int]]] = {
            0: {"iou_sum": 0.0, "dice_sum": 0.0, "f1_sum": 0.0, "auroc_sum": 0.0, "pixel_acc_sum": 0.0, "count": 0},
            1: {"iou_sum": 0.0, "dice_sum": 0.0, "f1_sum": 0.0, "auroc_sum": 0.0, "pixel_acc_sum": 0.0, "count": 0},
            2: {"iou_sum": 0.0, "dice_sum": 0.0, "f1_sum": 0.0, "auroc_sum": 0.0, "pixel_acc_sum": 0.0, "count": 0},
        }

    def update(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Update tracker with batch predictions and ground truth.
        pred_masks: (B, 1, H, W) or (B, H, W)
        target_masks: (B, 1, H, W) or (B, H, W)
        labels: (B,)
        """
        if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
            pred_masks = pred_masks.squeeze(1)
        if target_masks.dim() == 4 and target_masks.size(1) == 1:
            target_masks = target_masks.squeeze(1)

        b = pred_masks.size(0)
        probs = torch.sigmoid(pred_masks) if pred_masks.dtype.is_floating_point else pred_masks

        for i in range(b):
            m = compute_binary_metrics(
                probs[i],
                target_masks[i],
                threshold=self.threshold,
            )
            self.total_samples += 1
            for k in self.metrics_sum:
                self.metrics_sum[k] += m.get(k, 0.0)

            if labels is not None:
                lbl = int(labels[i].item())
                if lbl in self.per_label_stats:
                    self.per_label_stats[lbl]["iou_sum"] += m["iou"]
                    self.per_label_stats[lbl]["dice_sum"] += m["dice"]
                    self.per_label_stats[lbl]["f1_sum"] += m["f1"]
                    self.per_label_stats[lbl]["auroc_sum"] += m["auroc"]
                    self.per_label_stats[lbl]["pixel_acc_sum"] += m["pixel_acc"]
                    self.per_label_stats[lbl]["count"] += 1

    def compute(self) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        """
        Returns:
            overall_metrics: Dict[str, float]
            per_label_metrics: Dict[int, Dict[str, float]]
        """
        if self.total_samples == 0:
            return {k: 0.0 for k in self.metrics_sum}, {}

        overall = {k: self.metrics_sum[k] / self.total_samples for k in self.metrics_sum}

        per_label = {}
        for lbl, stats in self.per_label_stats.items():
            cnt = int(stats["count"])
            if cnt > 0:
                per_label[lbl] = {
                    "iou": stats["iou_sum"] / cnt,
                    "dice": stats["dice_sum"] / cnt,
                    "f1": stats["f1_sum"] / cnt,
                    "pixel_f1": stats["f1_sum"] / cnt,
                    "auroc": stats["auroc_sum"] / cnt,
                    "pixel_auroc": stats["auroc_sum"] / cnt,
                    "pixel_acc": stats["pixel_acc_sum"] / cnt,
                    "samples": cnt,
                }
            else:
                per_label[lbl] = {
                    "iou": 0.0,
                    "dice": 0.0,
                    "f1": 0.0,
                    "pixel_f1": 0.0,
                    "auroc": 0.0,
                    "pixel_auroc": 0.0,
                    "pixel_acc": 0.0,
                    "samples": 0,
                }

        return overall, per_label
