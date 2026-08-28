"""
Segmentation metrics: IoU, Dice / F1, Pixel Accuracy, Precision, Recall, and Specificity.
Includes robust per-sample edge case handling and running accumulators.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


def compute_binary_metrics(
    pred_mask: Union[torch.Tensor, np.ndarray],
    target_mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute binary segmentation metrics for a single sample or batch.
    pred_mask: Float tensor/array with probabilities or binary values.
    target_mask: Float/Int tensor/array in {0, 1}.
    """
    if isinstance(pred_mask, torch.Tensor):
        p = (pred_mask >= threshold).cpu().numpy().astype(np.bool_)
    else:
        p = (pred_mask >= threshold).astype(np.bool_)

    if isinstance(target_mask, torch.Tensor):
        t = (target_mask >= threshold).cpu().numpy().astype(np.bool_)
    else:
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

    # Intersection over Union (IoU)
    union = tp + fp + fn
    if union == 0:
        # Both prediction and target are entirely background (e.g. authentic image with perfect prediction)
        iou = 1.0
        dice = 1.0
    else:
        iou = float(tp) / float(union)
        dice = (2.0 * float(tp)) / float(2 * tp + fp + fn)

    # Precision, Recall, Specificity
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 1.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_acc": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


class SegmentationMetricTracker:
    """Accumulates and computes dataset-wide aggregated and per-label segmentation metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.metrics_sum: Dict[str, float] = {
            "iou": 0.0,
            "dice": 0.0,
            "pixel_acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
        }
        # Per-label accumulator
        self.per_label_stats: Dict[int, Dict[str, Union[float, int]]] = {
            0: {"iou_sum": 0.0, "dice_sum": 0.0, "pixel_acc_sum": 0.0, "count": 0},
            1: {"iou_sum": 0.0, "dice_sum": 0.0, "pixel_acc_sum": 0.0, "count": 0},
            2: {"iou_sum": 0.0, "dice_sum": 0.0, "pixel_acc_sum": 0.0, "count": 0},
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
                self.metrics_sum[k] += m[k]

            if labels is not None:
                lbl = int(labels[i].item())
                if lbl in self.per_label_stats:
                    self.per_label_stats[lbl]["iou_sum"] += m["iou"]
                    self.per_label_stats[lbl]["dice_sum"] += m["dice"]
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
                    "pixel_acc": stats["pixel_acc_sum"] / cnt,
                    "samples": cnt,
                }
            else:
                per_label[lbl] = {"iou": 0.0, "dice": 0.0, "pixel_acc": 0.0, "samples": 0}

        return overall, per_label
