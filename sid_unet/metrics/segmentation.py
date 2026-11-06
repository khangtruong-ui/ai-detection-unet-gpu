"""
Segmentation metrics: IoU, Dice / Pixel F1, AUROC, Pixel Accuracy, Precision, Recall, and Specificity.
Includes robust per-sample edge case handling and running accumulators.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


from scipy.stats import rankdata

__all__ = [
    "compute_binary_auroc",
    "compute_binary_auroc_batch",
    "compute_binary_metrics",
    "compute_binary_metrics_batch",
    "SegmentationMetricTracker",
]


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


def compute_binary_auroc_batch(
    pred_probs: torch.Tensor,
    target_masks: torch.Tensor,
    is_logit: Optional[bool] = None,
) -> torch.Tensor:
    """
    Compute AUROC curve on continuous pixel probabilities across a batch of masks on device.
    Robustly handles single-class masks (all-authentic or all-tampered), soft labels, ties, and raw logits.

    Args:
        pred_probs: Tensor of shape (B, 1, H, W), (B, H, W), or (B, N).
        target_masks: Tensor of same shape with ground truth values.
        is_logit: Optional bool indicating if pred_probs are logits.

    Returns:
        torch.Tensor of shape (B,) containing AUROC scores in [0.0, 1.0].
    """
    if pred_probs.dim() > 2:
        p = pred_probs.reshape(pred_probs.size(0), -1).float()
    else:
        p = pred_probs.float()

    if target_masks.dim() > 2:
        t = target_masks.reshape(target_masks.size(0), -1).float()
    else:
        t = target_masks.float()

    p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)

    if is_logit is True:
        p = torch.sigmoid(p.clamp(-500.0, 500.0))
    elif is_logit is False:
        pass
    elif (p < 0.0).any() or (p > 1.0).any():
        p = torch.sigmoid(p.clamp(-500.0, 500.0))

    t_bin = (t >= 0.5).long()
    b, n = p.shape
    device = p.device

    n_pos = t_bin.sum(dim=-1).double()
    n_neg = (n - n_pos).double()

    # Fast path: sort along last dim
    sorted_p, sort_idx = torch.sort(p, dim=-1)
    sorted_t = torch.gather(t_bin, dim=-1, index=sort_idx)

    diff = sorted_p[:, 1:] != sorted_p[:, :-1]
    has_ties = diff.sum(dim=-1) < (n - 1)

    ranks = torch.arange(1, n + 1, device=device, dtype=torch.float64).unsqueeze(0).expand(b, n)
    pos_rank_sum = (ranks * sorted_t.double()).sum(dim=-1)
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1.0)) / 2.0
    auroc_no_ties = u_stat / (n_pos * n_neg).clamp(min=1e-7)

    aurocs = torch.empty(b, dtype=torch.float32, device=device)
    for i in range(b):
        np_i = int(n_pos[i].item())
        nn_i = int(n_neg[i].item())
        if np_i == 0:
            mp = float(p[i].mean().item())
            aurocs[i] = 1.0 if mp < 0.5 else max(0.0, 1.0 - mp)
        elif nn_i == 0:
            mp = float(p[i].mean().item())
            aurocs[i] = 1.0 if mp >= 0.5 else max(0.0, mp)
        elif not has_ties[i].item():
            aurocs[i] = float(auroc_no_ties[i].item())
        else:
            diff_i = diff[i]
            if not diff_i.any():
                aurocs[i] = 0.5
            else:
                split_idx = torch.nonzero(diff_i, as_tuple=False).flatten() + 1
                starts = torch.cat([torch.tensor([0], device=device), split_idx])
                ends = torch.cat([split_idx, torch.tensor([n], device=device)])
                midranks = (starts + ends + 1).to(torch.float64) / 2.0
                repeats = ends - starts
                sorted_ranks = torch.repeat_interleave(midranks, repeats)
                pos_r_sum = (sorted_ranks * sorted_t[i].to(torch.float64)).sum().item()
                u_i = pos_r_sum - (np_i * (np_i + 1)) / 2.0
                aurocs[i] = float(u_i / (np_i * nn_i))

    return aurocs


def compute_binary_metrics_batch(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    threshold: float = 0.5,
    is_logit: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute binary segmentation metrics (IoU, Dice/F1, AUROC, Accuracy, Precision, Recall, Specificity)
    in parallel across a batch on the tensor's native device.

    Args:
        pred_masks: Tensor of shape (B, 1, H, W) or (B, H, W) with logits or probabilities.
        target_masks: Tensor of shape (B, 1, H, W) or (B, H, W) with ground truth masks.
        threshold: Binarization probability threshold.
        is_logit: Optional bool indicating if pred_masks are raw logits.

    Returns:
        Dict mapping metric name to 1D Tensor of shape (B,) on same device.
    """
    if pred_masks.dim() > 2:
        p_raw = pred_masks.reshape(pred_masks.size(0), -1).float()
    else:
        p_raw = pred_masks.float()

    if target_masks.dim() > 2:
        t_raw = target_masks.reshape(target_masks.size(0), -1).float()
    else:
        t_raw = target_masks.float()

    p_raw = torch.nan_to_num(p_raw, nan=0.0, posinf=1.0, neginf=0.0)
    t_raw = torch.nan_to_num(t_raw, nan=0.0, posinf=1.0, neginf=0.0)

    if is_logit is True:
        p_probs = torch.sigmoid(p_raw.clamp(-500.0, 500.0))
    elif is_logit is False:
        p_probs = p_raw
    elif (p_raw < 0.0).any() or (p_raw > 1.0).any():
        p_probs = torch.sigmoid(p_raw.clamp(-500.0, 500.0))
    else:
        p_probs = p_raw

    p_bin = p_probs >= threshold
    t_bin = t_raw >= threshold

    tp = (p_bin & t_bin).sum(dim=-1).float()
    fp = (p_bin & (~t_bin)).sum(dim=-1).float()
    fn = ((~p_bin) & t_bin).sum(dim=-1).float()
    tn = ((~p_bin) & (~t_bin)).sum(dim=-1).float()
    total_pixels = float(p_raw.size(-1))

    pixel_acc = (tp + tn) / max(total_pixels, 1.0)
    union = tp + fp + fn
    iou = torch.where(union == 0, torch.ones_like(union), tp / union.clamp(min=1e-7))
    dice = torch.where(union == 0, torch.ones_like(union), (2.0 * tp) / (2.0 * tp + fp + fn).clamp(min=1e-7))
    pixel_f1 = dice

    prec_denom = tp + fp
    precision = torch.where(prec_denom > 0, tp / prec_denom.clamp(min=1e-7), torch.where(fn == 0, torch.ones_like(tp), torch.zeros_like(tp)))

    rec_denom = tp + fn
    recall = torch.where(rec_denom > 0, tp / rec_denom.clamp(min=1e-7), torch.where(fp == 0, torch.ones_like(tp), torch.zeros_like(tp)))

    spec_denom = tn + fp
    specificity = torch.where(spec_denom > 0, tn / spec_denom.clamp(min=1e-7), torch.ones_like(tn))

    auroc = compute_binary_auroc_batch(p_probs, t_raw, is_logit=False)

    num_pos = (tp + fn).long()
    has_pos = (tp + fn) > 0

    return {
        "iou": iou,
        "dice": dice,
        "f1": pixel_f1,
        "pixel_f1": pixel_f1,
        "auroc": auroc,
        "pixel_auroc": auroc,
        "pixel_acc": pixel_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "num_positive_pixels": num_pos,
        "has_positive": has_pos,
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
        self.per_dataset_stats: Dict[str, Dict[str, Union[float, int]]] = {}

    def update(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        is_logit: Optional[bool] = None,
        datasets: Optional[Union[List[str], Sequence[str]]] = None,
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
        if b == 0:
            return

        batch_metrics = compute_binary_metrics_batch(
            pred_masks,
            target_masks,
            threshold=self.threshold,
            is_logit=is_logit,
        )

        has_pos = batch_metrics["has_positive"]
        num_pos = int(has_pos.sum().item())

        self.total_samples += b
        self.positive_samples += num_pos
        self.zero_positive_samples += (b - num_pos)

        # Global sums
        for k in self.metrics_sum:
            self.metrics_sum[k] += float(batch_metrics[k].sum().item())

        if num_pos > 0:
            for k in self.positive_metrics_sum:
                self.positive_metrics_sum[k] += float(batch_metrics[k][has_pos].sum().item())

        # Per-label stats
        if labels is not None:
            labels_flat = labels.view(-1)
            unique_labels = torch.unique(labels_flat)
            for lbl_t in unique_labels:
                lbl = int(lbl_t.item())
                if lbl not in self.per_label_stats:
                    self.per_label_stats[lbl] = {
                        "iou_sum": 0.0, "dice_sum": 0.0, "f1_sum": 0.0, "auroc_sum": 0.0,
                        "pixel_acc_sum": 0.0, "precision_sum": 0.0, "recall_sum": 0.0,
                        "specificity_sum": 0.0, "count": 0, "pos_count": 0,
                        "iou_pos_sum": 0.0, "dice_pos_sum": 0.0, "f1_pos_sum": 0.0,
                        "precision_pos_sum": 0.0, "recall_pos_sum": 0.0,
                    }
                mask_lbl = (labels_flat == lbl_t)
                cnt_lbl = int(mask_lbl.sum().item())
                pos_mask_lbl = mask_lbl & has_pos
                pos_cnt_lbl = int(pos_mask_lbl.sum().item())

                l_stat = self.per_label_stats[lbl]
                l_stat["count"] += cnt_lbl
                l_stat["pos_count"] += pos_cnt_lbl
                l_stat["iou_sum"] += float(batch_metrics["iou"][mask_lbl].sum().item())
                l_stat["dice_sum"] += float(batch_metrics["dice"][mask_lbl].sum().item())
                l_stat["f1_sum"] += float(batch_metrics["f1"][mask_lbl].sum().item())
                l_stat["auroc_sum"] += float(batch_metrics["auroc"][mask_lbl].sum().item())
                l_stat["pixel_acc_sum"] += float(batch_metrics["pixel_acc"][mask_lbl].sum().item())
                l_stat["precision_sum"] += float(batch_metrics["precision"][mask_lbl].sum().item())
                l_stat["recall_sum"] += float(batch_metrics["recall"][mask_lbl].sum().item())
                l_stat["specificity_sum"] += float(batch_metrics["specificity"][mask_lbl].sum().item())

                if pos_cnt_lbl > 0:
                    l_stat["iou_pos_sum"] += float(batch_metrics["iou"][pos_mask_lbl].sum().item())
                    l_stat["dice_pos_sum"] += float(batch_metrics["dice"][pos_mask_lbl].sum().item())
                    l_stat["f1_pos_sum"] += float(batch_metrics["f1"][pos_mask_lbl].sum().item())
                    l_stat["precision_pos_sum"] += float(batch_metrics["precision"][pos_mask_lbl].sum().item())
                    l_stat["recall_pos_sum"] += float(batch_metrics["recall"][pos_mask_lbl].sum().item())

        # Per-dataset stats
        if datasets is not None:
            for i, raw_ds in enumerate(datasets):
                if i >= b:
                    break
                if raw_ds is None or not str(raw_ds).strip():
                    continue
                ds_name = str(raw_ds).strip()
                if ds_name not in self.per_dataset_stats:
                    self.per_dataset_stats[ds_name] = {
                        "iou_sum": 0.0, "dice_sum": 0.0, "f1_sum": 0.0, "auroc_sum": 0.0,
                        "pixel_acc_sum": 0.0, "precision_sum": 0.0, "recall_sum": 0.0,
                        "specificity_sum": 0.0, "count": 0, "pos_count": 0,
                        "iou_pos_sum": 0.0, "dice_pos_sum": 0.0, "f1_pos_sum": 0.0,
                        "precision_pos_sum": 0.0, "recall_pos_sum": 0.0,
                    }
                ds_stat = self.per_dataset_stats[ds_name]
                ds_stat["count"] += 1
                ds_stat["iou_sum"] += float(batch_metrics["iou"][i].item())
                ds_stat["dice_sum"] += float(batch_metrics["dice"][i].item())
                ds_stat["f1_sum"] += float(batch_metrics["f1"][i].item())
                ds_stat["auroc_sum"] += float(batch_metrics["auroc"][i].item())
                ds_stat["pixel_acc_sum"] += float(batch_metrics["pixel_acc"][i].item())
                ds_stat["precision_sum"] += float(batch_metrics["precision"][i].item())
                ds_stat["recall_sum"] += float(batch_metrics["recall"][i].item())
                ds_stat["specificity_sum"] += float(batch_metrics["specificity"][i].item())

                if bool(has_pos[i].item()):
                    ds_stat["pos_count"] += 1
                    ds_stat["iou_pos_sum"] += float(batch_metrics["iou"][i].item())
                    ds_stat["dice_pos_sum"] += float(batch_metrics["dice"][i].item())
                    ds_stat["f1_pos_sum"] += float(batch_metrics["f1"][i].item())
                    ds_stat["precision_pos_sum"] += float(batch_metrics["precision"][i].item())
                    ds_stat["recall_pos_sum"] += float(batch_metrics["recall"][i].item())

    def compute_per_dataset(self) -> Dict[str, Dict[str, float]]:
        """
        Compute averaged metrics broken down by sub-dataset.
        Returns:
            Dict[dataset_name, Dict[metric_name, float]]
        """
        per_dataset: Dict[str, Dict[str, float]] = {}
        for ds_name, stats in self.per_dataset_stats.items():
            cnt = int(stats["count"])
            pos_cnt = int(stats.get("pos_count", cnt))
            if cnt == 0:
                continue

            ds_m: Dict[str, float] = {
                "count": float(cnt),
                "samples": float(cnt),
                "pos_count": float(pos_cnt),
                "iou_all": stats["iou_sum"] / cnt,
                "dice_all": stats["dice_sum"] / cnt,
                "f1_all": stats["f1_sum"] / cnt,
                "pixel_acc": stats["pixel_acc_sum"] / cnt,
                "auroc": stats["auroc_sum"] / cnt,
                "pixel_auroc": stats["auroc_sum"] / cnt,
                "precision_all": stats["precision_sum"] / cnt,
                "recall_all": stats["recall_sum"] / cnt,
                "specificity": stats["specificity_sum"] / cnt,
            }

            if pos_cnt > 0:
                ds_m["iou_positive"] = stats["iou_pos_sum"] / pos_cnt
                ds_m["dice_positive"] = stats["dice_pos_sum"] / pos_cnt
                ds_m["f1_positive"] = stats["f1_pos_sum"] / pos_cnt
                ds_m["precision_positive"] = stats["precision_pos_sum"] / pos_cnt
                ds_m["recall_positive"] = stats["recall_pos_sum"] / pos_cnt
            else:
                ds_m["iou_positive"] = ds_m["iou_all"]
                ds_m["dice_positive"] = ds_m["dice_all"]
                ds_m["f1_positive"] = ds_m["f1_all"]
                ds_m["precision_positive"] = ds_m["precision_all"]
                ds_m["recall_positive"] = ds_m["recall_all"]

            if self.filter_zero_positive and pos_cnt > 0:
                ds_m["iou"] = ds_m["iou_positive"]
                ds_m["dice"] = ds_m["dice_positive"]
                ds_m["f1"] = ds_m["f1_positive"]
                ds_m["pixel_f1"] = ds_m["f1_positive"]
                ds_m["precision"] = ds_m["precision_positive"]
                ds_m["recall"] = ds_m["recall_positive"]
            else:
                ds_m["iou"] = ds_m["iou_all"]
                ds_m["dice"] = ds_m["dice_all"]
                ds_m["f1"] = ds_m["f1_all"]
                ds_m["pixel_f1"] = ds_m["f1_all"]
                ds_m["precision"] = ds_m["precision_all"]
                ds_m["recall"] = ds_m["recall_all"]

            per_dataset[ds_name] = ds_m

        return per_dataset

    @property
    def per_dataset_metrics(self) -> Dict[str, Dict[str, float]]:
        return self.compute_per_dataset()

    def compute(self, return_per_dataset: bool = False) -> Union[
        Tuple[Dict[str, float], Dict[int, Dict[str, float]]],
        Tuple[Dict[str, float], Dict[int, Dict[str, float]], Dict[str, Dict[str, float]]]
    ]:
        """
        Returns:
            overall_metrics: Dict[str, float]
            per_label_metrics: Dict[int, Dict[str, float]]
        """
        if self.total_samples == 0:
            if return_per_dataset:
                return {k: 0.0 for k in self.metrics_sum}, {}, {}
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

        if return_per_dataset:
            return overall, per_label, self.compute_per_dataset()
        return overall, per_label
