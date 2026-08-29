"""
Classification metrics for 3-class auxiliary label prediction (0: Real, 1: Fully AI, 2: Partially AI).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import torch


class ClassificationMetricTracker:
    """Accumulates and computes 3-class classification accuracy, macro F1, and confusion matrix."""

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.all_preds: List[int] = []
        self.all_targets: List[int] = []

    def update(self, class_logits: torch.Tensor, target_labels: torch.Tensor):
        """
        class_logits: (B, num_classes)
        target_labels: (B,)
        """
        preds = torch.argmax(class_logits, dim=1).detach().cpu().tolist()
        targets = target_labels.detach().cpu().tolist()

        self.all_preds.extend(preds)
        self.all_targets.extend(targets)

    def compute(self) -> Tuple[Dict[str, float], Optional[List[List[int]]]]:
        if not self.all_targets:
            return {}, None

        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)

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

        metrics = {
            "aux_accuracy": acc,
            "aux_macro_f1": macro_f1,
        }
        return metrics, cm.tolist()
