"""
Tests for nn-toolbox runtime learnability diagnostics in SID-UNet.

Focuses on validating learnability diagnostics at runtime:
1. Signal propagation and gradient flow across trainable layers.
2. Parameter update dynamics (||Δθ|| / ||θ|| update-to-weight ratio).
3. Reporting of verified healthy learnability dimensions alongside actionable warnings.
4. Detection of runtime unlearnability pathologies (e.g. frozen weights, zero displacement).
5. Deep-mode active experiments (tiny-dataset memorization capacity, train/eval consistency).
"""

import json
import os
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sid_unet.train import parse_args
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config


class LearnableSegmentationLoss(nn.Module):
    """Simple MSE loss for segmentation learnability testing."""

    def forward(self, outputs, masks, labels=None):
        out_t = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        loss = nn.functional.mse_loss(torch.sigmoid(out_t), masks.float())
        return loss, {"total_loss": float(loss.item()), "iou": 0.5}


def _create_synthetic_loader(num_samples: int = 4, image_size: int = 64, batch_size: int = 2):
    images = torch.randn(num_samples, 3, image_size, image_size)
    masks = (torch.rand(num_samples, 1, image_size, image_size) > 0.5).float()
    dataset = [{"image": images[i], "mask": masks[i], "label": torch.tensor(0)} for i in range(num_samples)]
    return DataLoader(dataset, batch_size=batch_size)


def test_cli_debug_flag_parsing(monkeypatch):
    """Verify CLI argument parsing for debug mode."""
    import sys

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "configs/default.yaml"])
    args = parse_args()
    assert args.debug is False
    assert args.debug_mode is None

    monkeypatch.setattr(sys, "argv", ["train.py", "--debug", "--config", "configs/default.yaml"])
    args = parse_args()
    assert args.debug is True

    monkeypatch.setattr(sys, "argv", ["train.py", "--debug-mode", "deep", "--config", "configs/default.yaml"])
    args = parse_args()
    assert args.debug_mode == "deep"


def test_trainer_runtime_learnability_healthy_flow():
    """Verify that a standard learnable pipeline reports healthy signals and updates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("configs/test_smoke.yaml")
        config.project.output_dir = tmpdir
        config.training.epochs = 1
        config.training.debug_mode = "light"
        config.training.auto_batch_size = False
        config.data.batch_size = 2

        train_loader = _create_synthetic_loader(num_samples=4)
        val_loader = _create_synthetic_loader(num_samples=4)

        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=LearnableSegmentationLoss(),
        )

        assert trainer.debug_mode == "light"
        results = trainer.train()
        assert results is not None

        # Verify diagnostic report was generated and attached to training output
        diag_report = results.get("diagnostic_report")
        assert diag_report is not None, "Expected diagnostic_report to be attached to trainer output"

        # Check learnability dimensions:
        # 1. Forward signal propagation was measured across network depth
        assert "forward_analysis" in diag_report.metrics
        fwd_metrics = diag_report.metrics["forward_analysis"]
        assert len(fwd_metrics.get("layers", {})) > 0, "Expected monitored layers in forward analysis"

        # 2. Backward gradient flow reached trainable parameters
        assert "backward_analysis" in diag_report.metrics
        bwd_metrics = diag_report.metrics["backward_analysis"]
        assert bwd_metrics.get("params_with_gradients", 0) > 0, "Trainable parameters should receive gradients"

        # 3. Parameter displacement was tracked (||Δθ|| / ||θ||)
        assert "update_stats" in diag_report.metrics
        upd_metrics = diag_report.metrics["update_stats"]
        assert "global_update_ratio" in upd_metrics
        assert upd_metrics["global_update_ratio"] > 0, "Learnable model should have non-zero update ratio"

        # 4. Verified healthy dimensions are briefly reported
        healthy_findings = diag_report.healthy_findings
        assert len(healthy_findings) > 0, "Healthy learnability checks should produce positive confirmations"
        healthy_categories = {f.category for f in healthy_findings}
        assert ("forward" in healthy_categories) or ("backward" in healthy_categories) or ("data" in healthy_categories)

        # 5. Output files exist and are valid
        diag_dir = os.path.join(tmpdir, "reports", "diagnostics")
        json_path = os.path.join(diag_dir, "diagnostic_report.json")
        html_path = os.path.join(diag_dir, "diagnostic_report.html")
        assert os.path.exists(json_path)
        assert os.path.exists(html_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["mode"] == "light"
            assert "summary" in data
            assert data["summary"]["healthy_count"] > 0


def test_trainer_runtime_learnability_unlearnable_frozen_network():
    """Verify that an unlearnable model (all weights frozen) is detected at runtime."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("configs/test_smoke.yaml")
        config.project.output_dir = tmpdir
        config.training.epochs = 1
        config.training.debug_mode = "light"
        config.training.auto_batch_size = False
        config.data.batch_size = 2

        train_loader = _create_synthetic_loader(num_samples=4)
        val_loader = _create_synthetic_loader(num_samples=4)

        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=LearnableSegmentationLoss(),
        )

        # Freeze all model parameters to simulate runtime unlearnability
        for p in trainer.model.parameters():
            p.requires_grad = False

        # Run diagnostics directly
        diag_report = trainer._run_debug_diagnostics()
        assert diag_report is not None

        # Verify that the unlearnable condition is flagged
        actionables = diag_report.actionable_findings
        assert len(actionables) > 0, "Frozen network should produce actionable learnability warnings"

        # Check for critical parameter freezing finding
        opt_findings = [f for f in actionables if f.category == "optimization"]
        assert len(opt_findings) > 0
        assert any("frozen" in f.observation.lower() for f in opt_findings)
        assert any(f.severity == "critical" for f in opt_findings)


def test_trainer_runtime_learnability_unlearnable_zero_lr():
    """Verify that an optimizer configured with lr=0 produces zero displacement warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("configs/test_smoke.yaml")
        config.project.output_dir = tmpdir
        config.training.epochs = 1
        config.training.debug_mode = "light"
        config.training.auto_batch_size = False
        config.data.batch_size = 2

        train_loader = _create_synthetic_loader(num_samples=4)
        val_loader = _create_synthetic_loader(num_samples=4)

        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=LearnableSegmentationLoss(),
        )

        # Zero out learning rate in optimizer to simulate zero parameter displacement
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = 0.0

        diag_report = trainer._run_debug_diagnostics()
        assert diag_report is not None

        # Check update stats show zero displacement
        upd = diag_report.metrics.get("update_stats", {})
        assert upd.get("global_update_ratio", 0.0) == 0.0

        # Actionable optimization finding should be triggered
        actionables = diag_report.actionable_findings
        opt_issues = [f for f in actionables if f.category == "optimization"]
        assert len(opt_issues) > 0
        assert any("did not change" in f.observation.lower() or "ratio" in f.observation.lower() for f in opt_issues)


def test_trainer_runtime_learnability_deep_mode_memorization():
    """Verify that deep mode runs active memorization tests and train/eval checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("configs/test_smoke.yaml")
        config.project.output_dir = tmpdir
        config.training.epochs = 1
        config.training.debug_mode = "deep"
        config.training.auto_batch_size = False
        config.data.batch_size = 2

        train_loader = _create_synthetic_loader(num_samples=4)
        val_loader = _create_synthetic_loader(num_samples=4)

        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=LearnableSegmentationLoss(),
        )

        results = trainer.train()
        assert results is not None
        diag_report = results["diagnostic_report"]
        assert diag_report is not None
        assert diag_report.mode == "deep"

        # Verify active experiments ran
        assert "overfit_test" in diag_report.metrics
        of_metrics = diag_report.metrics["overfit_test"]
        assert "results" in of_metrics
        assert len(of_metrics["results"]) > 0

        # Verify train/eval consistency test ran
        assert "train_eval" in diag_report.metrics
        te_metrics = diag_report.metrics["train_eval"]
        assert "relative_difference" in te_metrics
