"""
Tests for nn-toolbox debug mode integration in SID-UNet.
Verifies CLI flags, diagnostic execution, and report generation during training.
"""

import os
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sid_unet.train import parse_args, train_single_run
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config


class TinyMockLoss(nn.Module):
    def forward(self, outputs, masks, labels=None):
        out_t = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        loss = nn.functional.mse_loss(torch.sigmoid(out_t), masks.float())
        return loss, {"total_loss": float(loss.item()), "iou": 0.5}


def test_cli_debug_flag_parsing(monkeypatch):
    import sys

    # Test default
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "configs/default.yaml"])
    args = parse_args()
    assert args.debug is False
    assert args.debug_mode is None

    # Test --debug
    monkeypatch.setattr(sys, "argv", ["train.py", "--debug", "--config", "configs/default.yaml"])
    args = parse_args()
    assert args.debug is True

    # Test --debug-mode deep
    monkeypatch.setattr(sys, "argv", ["train.py", "--debug-mode", "deep", "--config", "configs/default.yaml"])
    args = parse_args()
    assert args.debug_mode == "deep"


def test_trainer_debug_mode_light_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("configs/test_smoke.yaml")
        config.project.output_dir = tmpdir
        config.training.epochs = 1
        config.training.debug_mode = "light"
        config.training.auto_batch_size = False
        config.data.batch_size = 2

        # Create tiny synthetic dataset
        images = torch.randn(4, 3, 64, 64)
        masks = (torch.rand(4, 1, 64, 64) > 0.5).float()
        dataset = [{"image": images[i], "mask": masks[i], "label": torch.tensor(0)} for i in range(4)]
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=TinyMockLoss(),
        )

        assert trainer.debug_mode == "light"

        results = trainer.train()
        assert results is not None

        # Verify diagnostic report artifacts were generated
        diag_dir = os.path.join(tmpdir, "reports", "diagnostics")
        json_path = os.path.join(diag_dir, "diagnostic_report.json")
        html_path = os.path.join(diag_dir, "diagnostic_report.html")

        assert os.path.exists(json_path), f"Expected {json_path} to exist"
        assert os.path.exists(html_path), f"Expected {html_path} to exist"

        # Verify JSON report content
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["mode"] == "light"
            assert "findings" in data
            assert "investigation_targets" in data


def test_trainer_debug_mode_deep_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("configs/test_smoke.yaml")
        config.project.output_dir = tmpdir
        config.training.epochs = 1
        config.training.debug_mode = "deep"
        config.training.auto_batch_size = False
        config.data.batch_size = 2

        images = torch.randn(4, 3, 64, 64)
        masks = (torch.rand(4, 1, 64, 64) > 0.5).float()
        dataset = [{"image": images[i], "mask": masks[i], "label": torch.tensor(0)} for i in range(4)]
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=TinyMockLoss(),
        )

        assert trainer.debug_mode == "deep"

        results = trainer.train()
        assert results is not None

        diag_dir = os.path.join(tmpdir, "reports", "diagnostics")
        json_path = os.path.join(diag_dir, "diagnostic_report.json")
        assert os.path.exists(json_path)

        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["mode"] == "deep"
