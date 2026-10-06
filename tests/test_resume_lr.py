"""
Tests for learning rate and scheduler behavior upon resuming training.
Verifies that resuming does not get locked at min_lr (1e-6) and that
cosine scheduler correctly reschedules or restarts over extended epochs.
"""

import math
import os
import tempfile
import torch
import pytest

from sid_unet.models.unet import UNet
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import ConfigDict


def _create_mock_checkpoint(ckpt_path: str, epoch: int = 10, step: int = 100, lr: float = 1e-6):
    dummy_model = UNet(in_channels=3, out_channels=1, features=[4, 8], bilinear=True)
    dummy_opt = torch.optim.AdamW(dummy_model.parameters(), lr=lr)
    dummy_sched = torch.optim.lr_scheduler.CosineAnnealingLR(dummy_opt, T_max=10, eta_min=1e-6)
    for _ in range(10):
        dummy_sched.step()

    torch.save({
        "epoch": epoch,
        "step": step,
        "best_score": 0.5,
        "best_epoch": epoch,
        "model_state_dict": dummy_model.state_dict(),
        "optimizer_state_dict": dummy_opt.state_dict(),
        "scheduler_state_dict": dummy_sched.state_dict(),
        "history": [{"epoch": i, "val_iou": 0.5} for i in range(1, epoch + 1)],
    }, ckpt_path)


def test_resume_lr_auto_reschedules_cosine():
    """Verify that when training resumes from a completed run (lr=1e-6), the learning rate is NOT stuck at 1e-6."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint_latest.pt")
        _create_mock_checkpoint(ckpt_path, epoch=10, lr=1e-6)

        cfg = ConfigDict({
            "project": {"name": "test_lr", "device": "cpu", "output_dir": tmpdir},
            "data": {"dataset_name": "dummy"},
            "model": {"name": "unet", "features": [4, 8], "bilinear": True},
            "loss": {"mask_loss_type": "combined"},
            "training": {
                "epochs": 40,
                "learning_rate": 0.0003,
                "min_lr": 1e-6,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "resume_lr_mode": "auto",
            },
            "logging": {"log_interval": 10},
        })

        trainer = Trainer(config=cfg)
        assert trainer.epochs == 40
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0003)

        # Resume from checkpoint that had decayed to 1e-6
        trainer.resume_from_checkpoint(ckpt_path)

        resumed_lr = trainer.optimizer.param_groups[0]["lr"]
        # Resumed LR must NOT be 1e-6! It should be scaled according to cosine at epoch 11/40
        assert resumed_lr > 1e-5
        expected_lr = 1e-6 + 0.5 * (0.0003 - 1e-6) * (1.0 + math.cos(math.pi * 11 / 40))
        assert resumed_lr == pytest.approx(expected_lr, rel=1e-3)
        assert getattr(trainer.scheduler, "T_max", None) == 40


def test_resume_lr_restart_mode():
    """Verify that 'restart' mode starts a new cosine cycle over remaining epochs from base_lr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint_latest.pt")
        _create_mock_checkpoint(ckpt_path, epoch=10, lr=1e-6)

        cfg = ConfigDict({
            "project": {"name": "test_restart", "device": "cpu", "output_dir": tmpdir},
            "data": {"dataset_name": "dummy"},
            "model": {"name": "unet", "features": [4, 8], "bilinear": True},
            "loss": {"mask_loss_type": "combined"},
            "training": {
                "epochs": 40,
                "learning_rate": 0.0003,
                "min_lr": 1e-6,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "resume_lr_mode": "restart",
            },
            "logging": {"log_interval": 10},
        })

        trainer = Trainer(config=cfg)
        trainer.resume_from_checkpoint(ckpt_path)

        resumed_lr = trainer.optimizer.param_groups[0]["lr"]
        expected_restart_lr = 1e-6 + 0.5 * (0.0003 - 1e-6) * (1.0 + math.cos(math.pi * 1 / 30))
        assert resumed_lr == pytest.approx(expected_restart_lr, rel=1e-3)

        # Stepping through remaining epochs reaches min_lr at epoch 40
        for ep in range(11, 41):
            trainer.scheduler.step()
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-6, rel=1e-3)


def test_resume_lr_custom_explicit_rate():
    """Verify that training.resume_lr or --resume-lr sets custom starting learning rate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint_latest.pt")
        _create_mock_checkpoint(ckpt_path, epoch=10, lr=1e-6)

        cfg = ConfigDict({
            "project": {"name": "test_custom_lr", "device": "cpu", "output_dir": tmpdir},
            "data": {"dataset_name": "dummy"},
            "model": {"name": "unet", "features": [4, 8], "bilinear": True},
            "loss": {"mask_loss_type": "combined"},
            "training": {
                "epochs": 40,
                "learning_rate": 0.0003,
                "resume_lr": 0.0005,
                "min_lr": 1e-6,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "resume_lr_mode": "restart",
            },
            "logging": {"log_interval": 10},
        })

        trainer = Trainer(config=cfg)
        trainer.resume_from_checkpoint(ckpt_path)

        expected_custom_lr = 1e-6 + 0.5 * (0.0005 - 1e-6) * (1.0 + math.cos(math.pi * 1 / 30))
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected_custom_lr, rel=1e-3)


def test_resume_lr_keep_mode():
    """Verify that 'keep' mode strictly keeps the checkpoint's lr untouched."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint_latest.pt")
        _create_mock_checkpoint(ckpt_path, epoch=10, lr=1e-6)

        cfg = ConfigDict({
            "project": {"name": "test_keep", "device": "cpu", "output_dir": tmpdir},
            "data": {"dataset_name": "dummy"},
            "model": {"name": "unet", "features": [4, 8], "bilinear": True},
            "loss": {"mask_loss_type": "combined"},
            "training": {
                "epochs": 40,
                "learning_rate": 0.0003,
                "min_lr": 1e-6,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "resume_lr_mode": "keep",
            },
            "logging": {"log_interval": 10},
        })

        trainer = Trainer(config=cfg)
        trainer.resume_from_checkpoint(ckpt_path)

        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
