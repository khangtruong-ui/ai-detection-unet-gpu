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

    def update(self, class_logits: torch.Tensor, target_labels: torch.Tensor):
        """
        class_logits: (B, num_classes)
        target_labels: (B,)
        """
        probs = torch.softmax(class_logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1).tolist()
        targets = target_labels.detach().cpu().tolist()

        self.all_preds.extend(preds)
        self.all_targets.extend(targets)
        self.all_probs.extend(probs.tolist())

    def compute(self) -> Tuple[Dict[str, float], Optional[List[List[int]]]]:
        if not self.all_targets:
            return {}, None

        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)
        y_probs = np.array(self.all_probs)

        # Accuracy
        acc = float(np.mean(y_true == y_pred))

        # Confusion Matrix
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                cm[t, p] += 1

        # Macro F1
        f1_scores = []
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores))

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
