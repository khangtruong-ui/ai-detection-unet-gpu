"""
Classification metrics for 3-class auxiliary label prediction (0: Real, 1: Fully AI, 2: Partially AI).
Includes accuracy, macro F1, AUROC, and confusion matrix.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class ClassificationMetricTracker:
    """Accumulates and computes 3-class classification accuracy, macro F1, AUROC, and confusion matrix."""

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
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
        class_logits: (B, num_classes)
        target_labels: (B,) or (B, 1)
        """
        if isinstance(class_logits, torch.Tensor):
            probs = torch.softmax(class_logits, dim=1).detach().cpu().numpy()
        else:
            arr = np.asarray(class_logits, dtype=np.float64)
            if (arr < 0.0).any() or (arr > 1.0).any() or not np.allclose(arr.sum(axis=-1), 1.0, atol=1e-3):
                exp_arr = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
                probs = exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)
            else:
                probs = arr

        preds = np.argmax(probs, axis=1).tolist()

        if isinstance(target_labels, torch.Tensor):
            targets = target_labels.detach().cpu().view(-1).tolist()
        else:
            targets = np.asarray(target_labels).reshape(-1).tolist()

        self.all_preds.extend(preds)
        self.all_targets.extend([int(t) for t in targets])
        self.all_probs.extend(probs.tolist())

    def compute(self) -> Tuple[Dict[str, float], Optional[List[List[int]]]]:
        if not self.all_targets:
            return {}, None

        y_true = np.asarray(self.all_targets, dtype=np.int64).reshape(-1)
        y_pred = np.asarray(self.all_preds, dtype=np.int64).reshape(-1)
        y_probs = np.asarray(self.all_probs, dtype=np.float64)

        # Accuracy
        acc = float(np.mean(y_true == y_pred))

        # Confusion Matrix
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                cm[t, p] += 1

        # Macro F1 (computed across active classes: present in ground truth or predicted)
        f1_scores = []
        active_classes = []
        for c in range(self.num_classes):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            support = tp + fn
            predicted = tp + fp

            precision = tp / predicted if predicted > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

            # Class is active if present in ground truth or falsely predicted by model
            if support > 0 or predicted > 0:
                active_classes.append(c)

        if active_classes:
            macro_f1 = float(np.mean([f1_scores[c] for c in active_classes]))
        else:
            macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        # Multi-class AUROC
        aux_auroc = 0.0
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1 and y_probs.shape[1] == self.num_classes:
                if len(unique_classes) == self.num_classes:
                    aux_auroc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
                else:
                    # One-vs-rest on present classes
                    present_aurocs = []
                    for c in unique_classes:
                        bin_true = (y_true == c).astype(int)
                        if len(np.unique(bin_true)) > 1:
                            present_aurocs.append(float(roc_auc_score(bin_true, y_probs[:, c])))
                    aux_auroc = float(np.mean(present_aurocs)) if present_aurocs else 1.0
            else:
                aux_auroc = 1.0 if acc > 0.5 else 0.5
        except Exception:
            aux_auroc = 1.0 if acc > 0.5 else 0.5

        metrics = {
            "aux_accuracy": acc,
            "aux_macro_f1": macro_f1,
            "aux_auroc": aux_auroc,
        }
        return metrics, cm.tolist()
