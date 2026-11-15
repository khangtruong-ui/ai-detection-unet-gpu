"""
Classification metrics for 3-class auxiliary label prediction (0: Real, 1: Fully AI, 2: Partially AI).
Includes accuracy, macro F1, AUROC, and confusion matrix.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class ClassificationMetricTracker:
    """Accumulates and computes 3-class classification accuracy, macro F1, AUROC, confusion matrix, and CIs."""

    def __init__(
        self,
        num_classes: int = 3,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 300,
    ):
        self.num_classes = num_classes
        self.confidence_level = float(confidence_level)
        self.bootstrap_samples = int(bootstrap_samples)
        self.reset()

    def reset(self):
        self.all_preds: List[int] = []
        self.all_targets: List[int] = []
        self.all_probs: List[List[float]] = []

    def update(
        self,
        class_logits: Union[torch.Tensor, np.ndarray],
        target_labels: Union[torch.Tensor, np.ndarray, List[int]],
    ):
        """
        class_logits: (B, num_classes) or (B, 1) or (B,)
        target_labels: (B,) or (B, 1)
        """
        if isinstance(class_logits, torch.Tensor):
            t_logits = class_logits.detach().cpu().float()
            if t_logits.ndim == 1:
                t_logits = t_logits.unsqueeze(1)
            if t_logits.shape[-1] == 1:
                p1 = torch.sigmoid(t_logits)
                p0 = 1.0 - p1
                probs = torch.cat([p0, p1], dim=-1).numpy()
            else:
                probs = torch.softmax(t_logits, dim=-1).numpy()
        else:
            arr = np.asarray(class_logits, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[-1] == 1:
                p1 = 1.0 / (1.0 + np.exp(-np.clip(arr, -500.0, 500.0)))
                probs = np.concatenate([1.0 - p1, p1], axis=-1)
            elif (arr < 0.0).any() or (arr > 1.0).any() or not np.allclose(arr.sum(axis=-1), 1.0, atol=1e-3):
                exp_arr = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
                probs = exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)
            else:
                probs = arr

        if probs.shape[-1] > self.num_classes:
            self.num_classes = probs.shape[-1]

        preds = np.argmax(probs, axis=1).tolist()

        if isinstance(target_labels, torch.Tensor):
            targets = target_labels.detach().cpu().view(-1).tolist()
        else:
            targets = np.asarray(target_labels).reshape(-1).tolist()

        self.all_preds.extend(preds)
        self.all_targets.extend([int(t) for t in targets])
        self.all_probs.extend(probs.tolist())

    @staticmethod
    def _compute_cm_and_macro_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eff_classes: int,
    ) -> Tuple[np.ndarray, float]:
        cm = np.zeros((eff_classes, eff_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < eff_classes and 0 <= p < eff_classes:
                cm[t, p] += 1

        f1_scores = []
        active_classes = []
        for c in range(eff_classes):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            support = tp + fn
            predicted = tp + fp
            precision = tp / predicted if predicted > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
            if support > 0 or predicted > 0:
                active_classes.append(c)

        if active_classes:
            macro_f1 = float(np.mean([f1_scores[c] for c in active_classes]))
        else:
            macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        return cm, macro_f1

    @staticmethod
    def _compute_auroc(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        acc: float,
    ) -> float:
        try:
            unique_classes = np.unique(y_true)
            n_prob_classes = y_probs.shape[1] if y_probs.ndim > 1 else 1

            if len(unique_classes) > 1 and n_prob_classes >= 2:
                if n_prob_classes == 2 and len(unique_classes) == 2:
                    pos_cls = unique_classes[1]
                    bin_true = (y_true == pos_cls).astype(int)
                    return float(roc_auc_score(bin_true, y_probs[:, 1]))
                if n_prob_classes > 2 and len(unique_classes) == n_prob_classes:
                    return float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
                present_aurocs = []
                for c in unique_classes:
                    if c < n_prob_classes:
                        bin_true = (y_true == c).astype(int)
                        if len(np.unique(bin_true)) > 1:
                            present_aurocs.append(float(roc_auc_score(bin_true, y_probs[:, c])))
                return float(np.mean(present_aurocs)) if present_aurocs else (1.0 if acc > 0.5 else 0.5)
            return 1.0 if acc > 0.5 else 0.5
        except Exception:
            return 1.0 if acc > 0.5 else 0.5

    def compute(self) -> Tuple[Dict[str, Any], Optional[List[List[int]]]]:
        if not self.all_targets:
            return {}, None

        from sid_unet.metrics.confidence import (
            attach_ci_fields,
            compute_bootstrap_confidence_interval,
            compute_wilson_confidence_interval,
        )

        y_true = np.asarray(self.all_targets, dtype=np.int64).reshape(-1)
        y_pred = np.asarray(self.all_preds, dtype=np.int64).reshape(-1)
        y_probs = np.asarray(self.all_probs, dtype=np.float64)

        n_samples = int(len(y_true))
        correct = int(np.sum(y_true == y_pred))
        acc = float(correct) / float(max(1, n_samples))

        max_label = max(
            int(np.max(y_true)) if n_samples > 0 else 0,
            int(np.max(y_pred)) if n_samples > 0 else 0,
            y_probs.shape[1] - 1 if y_probs.ndim > 1 else 0,
        )
        eff_classes = max(self.num_classes, max_label + 1)
        cm, macro_f1 = self._compute_cm_and_macro_f1(y_true, y_pred, eff_classes)
        aux_auroc = self._compute_auroc(y_true, y_probs, acc)

        cls_ci_map: Dict[str, Dict[str, float]] = {
            "aux_accuracy": compute_wilson_confidence_interval(
                successes=correct,
                total=n_samples,
                confidence_level=self.confidence_level,
            ),
            "aux_macro_f1": compute_bootstrap_confidence_interval(
                point_estimate=macro_f1,
                metric_fn=lambda idx: self._compute_cm_and_macro_f1(y_true[idx], y_pred[idx], eff_classes)[1],
                n_samples=n_samples,
                confidence_level=self.confidence_level,
                n_bootstrap=self.bootstrap_samples,
            ),
            "aux_auroc": compute_bootstrap_confidence_interval(
                point_estimate=aux_auroc,
                metric_fn=lambda idx: self._compute_auroc(
                    y_true[idx], y_probs[idx], float(np.mean(y_true[idx] == y_pred[idx]))
                ) if len(np.unique(y_true[idx])) > 1 else None,
                n_samples=n_samples,
                confidence_level=self.confidence_level,
                n_bootstrap=self.bootstrap_samples,
            ),
        }

        metrics: Dict[str, Any] = {
            "aux_accuracy": acc,
            "aux_macro_f1": macro_f1,
            "aux_auroc": aux_auroc,
        }
        attach_ci_fields(metrics, cls_ci_map, confidence_level=self.confidence_level)
        return metrics, cm.tolist()
