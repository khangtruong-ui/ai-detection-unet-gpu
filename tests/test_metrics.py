import pytest
import numpy as np
import torch
from sid_unet.metrics.segmentation import compute_binary_metrics, SegmentationMetricTracker
from sid_unet.metrics.classification import ClassificationMetricTracker


def test_perfect_mask_metrics():
    target = np.ones((100, 100), dtype=np.float32)
    pred = np.ones((100, 100), dtype=np.float32)

    m = compute_binary_metrics(pred, target)
    assert m["iou"] == 1.0
    assert m["dice"] == 1.0
    assert m["pixel_acc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_empty_mask_metrics():
    target = np.zeros((100, 100), dtype=np.float32)
    pred = np.zeros((100, 100), dtype=np.float32)

    m = compute_binary_metrics(pred, target)
    assert m["iou"] == 1.0
    assert m["pixel_acc"] == 1.0


def test_segmentation_tracker_per_label():
    tracker = SegmentationMetricTracker(threshold=0.5)

    # Batch of 3 samples: labels 0, 1, 2
    # Start with strongly negative logits (-10.0 => prob ~ 0.0)
    preds = torch.full((3, 1, 32, 32), -10.0)
    targets = torch.zeros(3, 1, 32, 32)
    labels = torch.tensor([0, 1, 2])

    # Sample 0 (Real): target 0, pred -10.0 (0) -> perfect
    # Sample 1 (Synthetic): target 1, pred -10.0 (0) -> total mismatch
    targets[1, 0, :, :] = 1.0
    # Sample 2 (Tampered): half target 1, half pred 1
    targets[2, 0, :16, :] = 1.0
    preds[2, 0, :16, :] = 10.0  # High positive logit -> prob ~ 1.0

    tracker.update(preds, targets, labels)
    overall, per_label = tracker.compute()

    assert 0 in per_label
    assert 1 in per_label
    assert 2 in per_label
    assert per_label[0]["iou"] == 1.0
    assert per_label[1]["iou"] == 0.0
    assert per_label[2]["iou"] == 1.0


def test_classification_tracker():
    tracker = ClassificationMetricTracker(num_classes=3)
    logits = torch.tensor([
        [10.0, 0.0, 0.0],  # Pred 0, Target 0
        [0.0, 10.0, 0.0],  # Pred 1, Target 1
        [0.0, 0.0, 10.0],  # Pred 2, Target 2
        [10.0, 0.0, 0.0],  # Pred 0, Target 1 (wrong)
    ])
    targets = torch.tensor([0, 1, 2, 1])

    tracker.update(logits, targets)
    metrics, cm = tracker.compute()

    assert metrics["aux_accuracy"] == 0.75
    assert len(cm) == 3
    assert cm[0][0] == 1 # Target 0, Pred 0
    assert cm[1][1] == 1 # Target 1, Pred 1
    assert cm[1][0] == 1 # Target 1, Pred 0
    assert cm[2][2] == 1 # Target 2, Pred 2
