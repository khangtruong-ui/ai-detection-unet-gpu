"""Tests for 95% confidence interval metrics across segmentation, classification, and reporting."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from sid_unet.metrics import (
    ClassificationMetricTracker,
    OnlineVarianceTracker,
    SegmentationMetricTracker,
    attach_ci_fields,
    compute_bootstrap_confidence_interval,
    compute_mean_confidence_interval,
    compute_wilson_confidence_interval,
    critical_t_value,
)
from sid_unet.utils.report import (
    format_metrics_table,
    generate_checkpoint_cross_eval_report,
    generate_evaluation_report,
    generate_master_cross_evaluation_report,
)


def test_critical_t_value_matches_scipy():
    from scipy.stats import t as scipy_t

    for df in (1, 5, 29, 120):
        expected = float(scipy_t.ppf(0.975, df=df))
        actual = critical_t_value(df=df, confidence_level=0.95)
        assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_mean_confidence_interval_exactness_and_clamping():
    rng = np.random.default_rng(7)
    samples = np.clip(0.85 + 0.25 * rng.normal(size=40), 0.0, 1.0)
    tracker = OnlineVarianceTracker(["iou"])
    for v in samples:
        tracker.update_single({"iou": float(v)})

    ci = tracker.ci_for("iou", confidence_level=0.95, bounds=(0.0, 1.0))
    sample_mean = float(np.mean(samples))
    sample_std = float(np.std(samples, ddof=1))
    expected_margin = critical_t_value(len(samples) - 1, 0.95) * (sample_std / math.sqrt(len(samples)))

    assert math.isclose(ci["mean"], sample_mean, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(ci["std"], sample_std, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(ci["margin"], expected_margin, rel_tol=1e-6, abs_tol=1e-6)
    assert 0.0 <= ci["low"] <= ci["mean"] <= ci["high"] <= 1.0

    # Degenerate n=0 and n=1
    ci0 = compute_mean_confidence_interval(0.5, 0.0, 0, confidence_level=0.95)
    assert ci0["margin"] == 0.0 and ci0["low"] == 0.5 and ci0["high"] == 0.5
    ci1 = compute_mean_confidence_interval(0.75, 0.0, 1, confidence_level=0.95)
    assert ci1["margin"] == 0.0 and ci1["low"] == 0.75 and ci1["high"] == 0.75


def test_online_variance_tracker_chan_merge_equals_sample_by_sample():
    rng = np.random.default_rng(123)
    values = rng.uniform(0.1, 0.95, size=53)

    single_tracker = OnlineVarianceTracker(["dice"])
    for x in values:
        single_tracker.update_single({"dice": float(x)})

    batch_tracker = OnlineVarianceTracker(["dice"])
    for chunk in (values[:10], values[10:27], values[27:]):
        b_count = len(chunk)
        b_sum = float(np.sum(chunk))
        b_mean = b_sum / b_count
        b_m2 = float(np.sum((chunk - b_mean) ** 2))
        batch_tracker.merge_batch(b_count, {"dice": b_sum}, {"dice": b_m2})

    ci_single = single_tracker.ci_for("dice", confidence_level=0.95)
    ci_batch = batch_tracker.ci_for("dice", confidence_level=0.95)
    for k in ("mean", "low", "high", "margin", "std", "sem"):
        assert math.isclose(ci_single[k], ci_batch[k], rel_tol=1e-9, abs_tol=1e-9)


def test_wilson_and_bootstrap_ci():
    wilson = compute_wilson_confidence_interval(successes=80, total=100, confidence_level=0.95)
    assert math.isclose(wilson["mean"], 0.80, abs_tol=1e-9)
    assert 0.70 < wilson["low"] < 0.80 < wilson["high"] < 0.88
    assert wilson["margin"] > 0.0

    preds = np.array([0.1, 0.2, 0.4, 0.8, 0.9, 0.75, 0.3, 0.85, 0.2, 0.92])
    targets = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])

    def acc_fn(idx: np.ndarray) -> float:
        return float(((preds[idx] > 0.5) == targets[idx]).mean())

    boot_a = compute_bootstrap_confidence_interval(acc_fn(np.arange(len(targets))), acc_fn, len(targets), n_bootstrap=200, seed=42)
    boot_b = compute_bootstrap_confidence_interval(acc_fn(np.arange(len(targets))), acc_fn, len(targets), n_bootstrap=200, seed=42)
    assert boot_a == boot_b
    assert 0.0 <= boot_a["low"] <= boot_a["high"] <= 1.0


def test_segmentation_metric_tracker_emits_ci95_across_subsets():
    tracker = SegmentationMetricTracker(confidence_level=0.95, filter_zero_positive=True)
    torch.manual_seed(0)

    for b_idx in range(4):
        masks = torch.zeros((4, 1, 16, 16), dtype=torch.float32)
        logits = torch.full((4, 1, 16, 16), -4.0, dtype=torch.float32)
        labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        ds_names = ["CASIA", "CASIA", "CocoGlide", "CocoGlide"]

        # Positive masks on samples 1..3 with varying quality
        masks[1:, :, 2:10, 2:10] = 1.0
        logits[1, :, 2:10, 2:10] = 4.0
        logits[2, :, 3:10, 3:10] = 3.5
        logits[3, :, 4:9, 4:9] = 2.0 + 0.5 * b_idx

        tracker.update(pred_masks=logits, target_masks=masks, labels=labels, is_logit=True, datasets=ds_names)

    summary, per_label, per_dataset = tracker.compute(return_per_dataset=True)
    assert summary["confidence_level"] == 0.95
    assert "confidence_intervals" in summary

    for metric_key in ("iou", "dice", "f1", "auroc", "pixel_acc", "precision", "recall", "specificity"):
        assert f"{metric_key}_ci_low" in summary
        assert f"{metric_key}_ci_high" in summary
        assert f"{metric_key}_ci_margin" in summary
        assert f"{metric_key}_ci95_low" in summary
        assert f"{metric_key}_ci95_high" in summary
        assert 0.0 <= summary[f"{metric_key}_ci95_low"] <= summary[f"{metric_key}_ci95_high"] <= 1.0

    # Check per_label and per_dataset CIs
    assert 2 in per_label
    assert "iou_ci95_low" in per_label[2]
    assert "iou_ci95_high" in per_label[2]
    assert per_label[2]["iou_ci95_low"] <= per_label[2]["iou"] <= per_label[2]["iou_ci95_high"]

    assert "CocoGlide" in per_dataset
    assert "iou_ci95_low" in per_dataset["CocoGlide"]
    assert "auroc_ci95_high" in per_dataset["CocoGlide"]


def test_classification_metric_tracker_emits_ci95():
    # Binary classification (2 logits)
    cls_tracker = ClassificationMetricTracker(num_classes=2, confidence_level=0.95, bootstrap_samples=150)
    logits = torch.tensor(
        [
            [3.0, -2.0],
            [2.5, -1.0],
            [-1.5, 2.5],
            [-2.0, 3.0],
            [1.0, 1.5],  # misclassified if target=0
            [-0.5, 1.8],
            [2.2, -1.2],
            [-2.5, 2.9],
        ]
        * 3,
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1] * 3, dtype=torch.long)
    cls_tracker.update(logits, targets)

    res, cm = cls_tracker.compute()
    assert res["confidence_level"] == 0.95
    assert cm is not None
    for k in ("aux_accuracy", "aux_macro_f1", "aux_auroc"):
        assert f"{k}_ci95_low" in res
        assert f"{k}_ci95_high" in res
        assert f"{k}_ci_margin" in res
        assert 0.0 <= res[f"{k}_ci95_low"] <= res[k] <= res[f"{k}_ci95_high"] <= 1.0


def test_report_formatting_renders_ci95_columns_and_preserves_ranking(tmp_path):
    overall = {
        "iou": 0.72,
        "dice": 0.81,
        "f1": 0.81,
        "auroc": 0.90,
        "pixel_acc": 0.95,
        "precision": 0.79,
        "recall": 0.84,
        "specificity": 0.96,
        "total_evaluated_samples": 100,
    }
    attach_ci_fields(
        overall,
        {
            "iou": {"mean": 0.72, "low": 0.68, "high": 0.76, "margin": 0.04, "std": 0.20, "sem": 0.02, "n": 100.0, "confidence_level": 0.95},
            "dice": {"mean": 0.81, "low": 0.77, "high": 0.85, "margin": 0.04, "std": 0.20, "sem": 0.02, "n": 100.0, "confidence_level": 0.95},
            "f1": {"mean": 0.81, "low": 0.77, "high": 0.85, "margin": 0.04, "std": 0.20, "sem": 0.02, "n": 100.0, "confidence_level": 0.95},
            "auroc": {"mean": 0.90, "low": 0.87, "high": 0.93, "margin": 0.03, "std": 0.15, "sem": 0.015, "n": 100.0, "confidence_level": 0.95},
        },
        confidence_level=0.95,
    )

    table_md = format_metrics_table(overall, title="CI Table")
    assert "95% Confidence Interval" in table_md
    assert "[0.6800, 0.7600]" in table_md
    # Raw CI suffix keys should not clutter rows as separate metrics
    assert "Iou Ci95 Low" not in table_md

    per_label = {2: dict(overall, samples=50)}
    per_ds = {"CASIA": dict(overall, samples=50)}
    eval_rep = generate_evaluation_report(
        overall_metrics=overall,
        per_label_metrics=per_label,
        per_dataset_metrics=per_ds,
        config={"training": {"mixed_precision": "auto", "amp": True}},
        output_dir=str(tmp_path),
    )
    assert "IoU 95% CI" in eval_rep["markdown"]
    assert "[0.6800, 0.7600]" in eval_rep["markdown"]

    ckpt_res = generate_checkpoint_cross_eval_report(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        results=[{
            "config_name": "cfgA",
            "dataset_name": "DS_A",
            "checkpoint_name": "ckpt_a",
            "metrics": overall,
            "per_label_metrics": per_label,
            "per_dataset_metrics": per_ds,
        }],
        output_dir=str(tmp_path),
    )
    assert "IoU 95% CI" in ckpt_res["markdown"]

    low_metrics = dict(overall, iou=0.45)
    attach_ci_fields(
        low_metrics,
        {"iou": {"mean": 0.45, "low": 0.40, "high": 0.50, "margin": 0.05, "std": 0.25, "sem": 0.025, "n": 100.0, "confidence_level": 0.95}},
        confidence_level=0.95,
    )
    master_res = generate_master_cross_evaluation_report(
        cross_results=[
            {"checkpoint_name": "ckpt_low", "config_name": "c1", "dataset_name": "d1", "metrics": low_metrics},
            {"checkpoint_name": "ckpt_high", "config_name": "c1", "dataset_name": "d1", "metrics": overall},
        ],
        output_dir=str(tmp_path),
    )
    md = master_res["markdown"]
    assert "IoU 95% CI" in md
    assert md.index("ckpt_high") < md.index("ckpt_low")
