"""
Segmentation metrics: IoU, Dice / Pixel F1, AUROC, Pixel Accuracy, Precision, Recall, and Specificity.
Includes robust per-sample edge case handling and running accumulators.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


from scipy.stats import rankdata


def compute_binary_auroc(
    pred_probs: Union[torch.Tensor, np.ndarray],
    target_mask: Union[torch.Tensor, np.ndarray],
    is_logit: Optional[bool] = None,
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic (AUROC) curve on continuous pixel probabilities.
    Robustly handles single-class masks (all-authentic or all-tampered), soft labels, ties, and raw logits.
    """
    if isinstance(pred_probs, torch.Tensor):
        p = pred_probs.detach().cpu().float().numpy().astype(np.float64)
    else:
        p = np.asarray(pred_probs, dtype=np.float64)

    if isinstance(target_mask, torch.Tensor):
        t = target_mask.detach().cpu().float().numpy().astype(np.float64)
    else:
        t = np.asarray(target_mask, dtype=np.float64)

    # Sanitize NaNs and Infs if any
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert to probabilities via sigmoid if logits
    if is_logit is True:
        p = 1.0 / (1.0 + np.exp(-np.clip(p, -500.0, 500.0)))
    elif is_logit is False:
        pass
    elif np.any(p < 0.0) or np.any(p > 1.0):
        p = 1.0 / (1.0 + np.exp(-np.clip(p, -500.0, 500.0)))

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
        # Fallback to Mann-Whitney U calculation using midranks for tied values
        ranks = rankdata(p_flat)
        pos_rank_sum = float(np.sum(ranks[t_flat == 1]))
        u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))


def compute_binary_metrics(
    pred_mask: Union[torch.Tensor, np.ndarray],
    target_mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7,
    is_logit: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Compute binary segmentation metrics for a single sample or batch,
    including Pixel F1 score and AUROC.
    pred_mask: Float tensor/array with probabilities, logits, or binary values.
    target_mask: Float/Int tensor/array in {0, 1} or continuous [0, 1].
    """
    if isinstance(pred_mask, torch.Tensor):
        p_raw = pred_mask.detach().cpu().float().numpy().astype(np.float64)
    else:
        p_raw = np.asarray(pred_mask, dtype=np.float64)

    if isinstance(target_mask, torch.Tensor):
        t_raw = target_mask.detach().cpu().float().numpy().astype(np.float64)
    else:
        t_raw = np.asarray(target_mask, dtype=np.float64)

    p_raw = np.nan_to_num(p_raw, nan=0.0, posinf=1.0, neginf=0.0)
    t_raw = np.nan_to_num(t_raw, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert to probabilities via sigmoid if logits
    if is_logit is True:
        p_probs = 1.0 / (1.0 + np.exp(-np.clip(p_raw, -500.0, 500.0)))
    elif is_logit is False:
        p_probs = p_raw
    elif np.any(p_raw < 0.0) or np.any(p_raw > 1.0):
        p_probs = 1.0 / (1.0 + np.exp(-np.clip(p_raw, -500.0, 500.0)))
    else:
        p_probs = p_raw

    p = (p_probs >= threshold).astype(np.bool_)
    t = (t_raw >= threshold).astype(np.bool_)

    p_flat = p.flatten()
    t_flat = t.flatten()

    tp = int(np.sum(p_flat & t_flat))
    fp = int(np.sum(p_flat & (~t_flat)))
    fn = int(np.sum((~p_flat) & t_flat))
    tn = int(np.sum((~p_flat) & (~t_flat)))
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
    auroc = compute_binary_auroc(p_probs, t_raw, is_logit=False)

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
        "num_positive_pixels": int(tp + fn),
        "has_positive": bool((tp + fn) > 0),
    }


class SegmentationMetricTracker:
    """Accumulates and computes dataset-wide aggregated and per-label segmentation metrics including F1 and AUROC.
    
    Args:
        threshold (float): Probability threshold for binarizing predictions (default: 0.5).
        filter_zero_positive (bool): When True, filters out images with zero ground-truth positive pixels
            when calculating average IoU and Dice, preventing unmanipulated images from falsely pulling
            down the segmentation metric (default: False).
    """

    def __init__(self, threshold: float = 0.5, filter_zero_positive: bool = False):
        self.threshold = threshold
        self.filter_zero_positive = filter_zero_positive
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.positive_samples = 0
        self.zero_positive_samples = 0
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
        self.positive_metrics_sum: Dict[str, float] = {
            "iou": 0.0,
            "dice": 0.0,
            "f1": 0.0,
            "pixel_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
        # Per-label accumulator (0: Real, 1: Fully AI, 2: Partially AI, plus dynamic labels)
        self.per_label_stats: Dict[int, Dict[str, Union[float, int]]] = {
            0: {"iou_sum": 0.0, "dice_sum": 0.0, "f1_sum": 0.0, "auroc_sum": 0.0, "pixel_acc_sum": 0.0, "precision_sum": 0.0, "recall_sum": 0.0, "specificity_sum": 0.0, "count": 0, "pos_count": 0, "iou_pos_sum": 0.0, "dice_pos_sum": 0.0, "f1_pos_sum": 0.0, "precision_pos_sum": 0.0, "recall_pos_sum": 0.0},
            1: {"iou_sum": 0.0, "dice_sum": 0.0, "f1_sum": 0.0, "auroc_sum": 0.0, "pixel_acc_sum": 0.0, "precision_sum": 0.0, "recall_sum": 0.0, "specificity_sum": 0.0, "count": 0, "pos_count": 0, "iou_pos_sum": 0.0, "dice_pos_sum": 0.0, "f1_pos_sum": 0.0, "precision_pos_sum": 0.0, "recall_pos_sum": 0.0},
            2: {"iou_sum": 0.0, "dice_sum": 0.0, "f1_sum": 0.0, "auroc_sum": 0.0, "pixel_acc_sum": 0.0, "precision_sum": 0.0, "recall_sum": 0.0, "specificity_sum": 0.0, "count": 0, "pos_count": 0, "iou_pos_sum": 0.0, "dice_pos_sum": 0.0, "f1_pos_sum": 0.0, "precision_pos_sum": 0.0, "recall_pos_sum": 0.0},
        }

    def update(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        is_logit: Optional[bool] = None,
    ):
        """
        Update tracker with batch predictions and ground truth.
        pred_masks: (B, 1, H, W) or (B, H, W)
        target_masks: (B, 1, H, W) or (B, H, W)
        labels: (B,) or (B, 1)
        is_logit: Optional bool. If True, applies sigmoid. If False, treats as probabilities/binary.
                  If None, automatically detects based on value range (values outside [0, 1] indicate logits).
        """
        if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
            pred_masks = pred_masks.squeeze(1)
        if target_masks.dim() == 4 and target_masks.size(1) == 1:
            target_masks = target_masks.squeeze(1)
        target_masks = target_masks.float()

        b = pred_masks.size(0)

        # Distinguish between raw logits and already probabilities/binary masks
        pred_float = pred_masks.float()
        if is_logit is True:
            probs = torch.sigmoid(pred_float)
        elif is_logit is False:
            probs = pred_float
        else:
            # Auto-detect: if values fall outside [0, 1], they are raw logits requiring sigmoid
            if (pred_float < 0.0).any() or (pred_float > 1.0).any():
                probs = torch.sigmoid(pred_float)
            else:
                probs = pred_float

        if labels is not None:
            labels_flat = labels.view(-1)
        else:
            labels_flat = None

        for i in range(b):
            m = compute_binary_metrics(
                probs[i],
                target_masks[i],
                threshold=self.threshold,
                is_logit=False,
            )
            self.total_samples += 1
            has_pos = m.get("has_positive", True)
            if has_pos:
                self.positive_samples += 1
                for k in self.positive_metrics_sum:
                    self.positive_metrics_sum[k] += m.get(k, 0.0)
            else:
                self.zero_positive_samples += 1

            for k in self.metrics_sum:
                self.metrics_sum[k] += m.get(k, 0.0)

            if labels_flat is not None and i < len(labels_flat):
                lbl = int(labels_flat[i].item())
                if lbl not in self.per_label_stats:
                    self.per_label_stats[lbl] = {
                        "iou_sum": 0.0,
                        "dice_sum": 0.0,
                        "f1_sum": 0.0,
                        "auroc_sum": 0.0,
                        "pixel_acc_sum": 0.0,
                        "precision_sum": 0.0,
                        "recall_sum": 0.0,
                        "specificity_sum": 0.0,
                        "count": 0,
                        "pos_count": 0,
                        "iou_pos_sum": 0.0,
                        "dice_pos_sum": 0.0,
                        "f1_pos_sum": 0.0,
                        "precision_pos_sum": 0.0,
                        "recall_pos_sum": 0.0,
                    }
                self.per_label_stats[lbl]["iou_sum"] += m["iou"]
                self.per_label_stats[lbl]["dice_sum"] += m["dice"]
                self.per_label_stats[lbl]["f1_sum"] += m["f1"]
                self.per_label_stats[lbl]["auroc_sum"] += m["auroc"]
                self.per_label_stats[lbl]["pixel_acc_sum"] += m["pixel_acc"]
                self.per_label_stats[lbl]["precision_sum"] += m["precision"]
                self.per_label_stats[lbl]["recall_sum"] += m["recall"]
                self.per_label_stats[lbl]["specificity_sum"] += m["specificity"]
                self.per_label_stats[lbl]["count"] += 1
                if has_pos:
                    self.per_label_stats[lbl]["pos_count"] += 1
                    self.per_label_stats[lbl]["iou_pos_sum"] += m["iou"]
                    self.per_label_stats[lbl]["dice_pos_sum"] += m["dice"]
                    self.per_label_stats[lbl]["f1_pos_sum"] += m["f1"]
                    self.per_label_stats[lbl]["precision_pos_sum"] += m["precision"]
                    self.per_label_stats[lbl]["recall_pos_sum"] += m["recall"]

    def compute(self) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        """
        Returns:
            overall_metrics: Dict[str, float]
            per_label_metrics: Dict[int, Dict[str, float]]
        """
        if self.total_samples == 0:
            return {k: 0.0 for k in self.metrics_sum}, {}

        overall = {k: self.metrics_sum[k] / self.total_samples for k in self.metrics_sum}

        overall["iou_all"] = self.metrics_sum["iou"] / self.total_samples
        overall["dice_all"] = self.metrics_sum["dice"] / self.total_samples
        overall["f1_all"] = self.metrics_sum["f1"] / self.total_samples
        overall["positive_samples"] = self.positive_samples
        overall["zero_positive_samples"] = self.zero_positive_samples

        if self.positive_samples > 0:
            overall["iou_positive"] = self.positive_metrics_sum["iou"] / self.positive_samples
            overall["dice_positive"] = self.positive_metrics_sum["dice"] / self.positive_samples
            overall["f1_positive"] = self.positive_metrics_sum["f1"] / self.positive_samples
            overall["precision_positive"] = self.positive_metrics_sum["precision"] / self.positive_samples
            overall["recall_positive"] = self.positive_metrics_sum["recall"] / self.positive_samples
        else:
            overall["iou_positive"] = overall["iou_all"]
            overall["dice_positive"] = overall["dice_all"]
            overall["f1_positive"] = overall["f1_all"]
            overall["precision_positive"] = overall.get("precision", 0.0)
            overall["recall_positive"] = overall.get("recall", 0.0)

        if self.filter_zero_positive and self.positive_samples > 0:
            overall["iou"] = overall["iou_positive"]
            overall["dice"] = overall["dice_positive"]
            overall["f1"] = overall["f1_positive"]
            overall["pixel_f1"] = overall["f1_positive"]
            overall["precision"] = overall["precision_positive"]
            overall["recall"] = overall["recall_positive"]

        per_label = {}
        for lbl, stats in self.per_label_stats.items():
            cnt = int(stats["count"])
            pos_cnt = int(stats.get("pos_count", cnt))
            if cnt > 0:
                if self.filter_zero_positive and pos_cnt > 0:
                    lbl_iou = stats["iou_pos_sum"] / pos_cnt
                    lbl_dice = stats.get("dice_pos_sum", stats["dice_sum"]) / pos_cnt
                    lbl_f1 = stats.get("f1_pos_sum", stats["f1_sum"]) / pos_cnt
                    lbl_prec = stats.get("precision_pos_sum", stats.get("precision_sum", 0.0)) / pos_cnt
                    lbl_rec = stats.get("recall_pos_sum", stats.get("recall_sum", 0.0)) / pos_cnt
                else:
                    lbl_iou = stats["iou_sum"] / cnt
                    lbl_dice = stats["dice_sum"] / cnt
                    lbl_f1 = stats["f1_sum"] / cnt
                    lbl_prec = stats.get("precision_sum", 0.0) / cnt
                    lbl_rec = stats.get("recall_sum", 0.0) / cnt

                per_label[lbl] = {
                    "iou": lbl_iou,
                    "iou_all": stats["iou_sum"] / cnt,
                    "iou_positive": stats["iou_pos_sum"] / pos_cnt if pos_cnt > 0 else (stats["iou_sum"] / cnt),
                    "dice": lbl_dice,
                    "dice_all": stats["dice_sum"] / cnt,
                    "dice_positive": stats.get("dice_pos_sum", stats["dice_sum"]) / pos_cnt if pos_cnt > 0 else (stats["dice_sum"] / cnt),
                    "f1": lbl_f1,
                    "f1_all": stats["f1_sum"] / cnt,
                    "f1_positive": stats.get("f1_pos_sum", stats["f1_sum"]) / pos_cnt if pos_cnt > 0 else (stats["f1_sum"] / cnt),
                    "pixel_f1": lbl_f1,
                    "auroc": stats["auroc_sum"] / cnt,
                    "pixel_auroc": stats["auroc_sum"] / cnt,
                    "pixel_acc": stats["pixel_acc_sum"] / cnt,
                    "precision": lbl_prec,
                    "precision_all": stats.get("precision_sum", 0.0) / cnt,
                    "precision_positive": stats.get("precision_pos_sum", stats.get("precision_sum", 0.0)) / pos_cnt if pos_cnt > 0 else (stats.get("precision_sum", 0.0) / cnt),
                    "recall": lbl_rec,
                    "recall_all": stats.get("recall_sum", 0.0) / cnt,
                    "recall_positive": stats.get("recall_pos_sum", stats.get("recall_sum", 0.0)) / pos_cnt if pos_cnt > 0 else (stats.get("recall_sum", 0.0) / cnt),
                    "specificity": stats.get("specificity_sum", 0.0) / cnt,
                    "samples": cnt,
                    "positive_samples": pos_cnt,
                }
            else:
                per_label[lbl] = {
                    "iou": 0.0,
                    "iou_all": 0.0,
                    "iou_positive": 0.0,
                    "dice": 0.0,
                    "dice_all": 0.0,
                    "dice_positive": 0.0,
                    "f1": 0.0,
                    "f1_all": 0.0,
                    "f1_positive": 0.0,
                    "pixel_f1": 0.0,
                    "auroc": 0.0,
                    "pixel_auroc": 0.0,
                    "pixel_acc": 0.0,
                    "precision": 0.0,
                    "precision_all": 0.0,
                    "precision_positive": 0.0,
                    "recall": 0.0,
                    "recall_all": 0.0,
                    "recall_positive": 0.0,
                    "specificity": 0.0,
                    "samples": 0,
                    "positive_samples": 0,
                }

        return overall, per_label
