"""
Unit and integration tests for automatic checkpoint resumption from repository
and Hugging Face Hub model repositories.
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sid_unet.models.unet import UNet
from sid_unet.training.callbacks import (
    CheckpointManager,
    download_hf_checkpoint,
    find_auto_resume_checkpoint,
    format_no_resume_notification,
    format_resume_notification,
    inspect_checkpoint,
    is_hf_repo_id,
    parse_hf_repo_uri,
)
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.train import train_single_run, main as train_main


class DummyDataset(Dataset):
    def __init__(self, size=4, img_size=(32, 32)):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, *self.img_size),
            "mask": torch.ones(1, *self.img_size),
            "label": torch.tensor(idx % 3, dtype=torch.long),
            "img_id": f"dummy_{idx}",
        }


class MockHFDataset:
    def __init__(self, count=10):
        from PIL import Image
        self.samples = [
            {
                "image": Image.new("RGB", (32, 32), color=(i * 20, 100, 100)),
                "label": i % 3,
                "mask": Image.new("L", (32, 32), color=255 if i % 3 == 2 else 0),
                "img_id": f"mock_{i}",
            }
            for i in range(count)
        ]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def shuffle(self, seed=None, buffer_size=None):
        return self

    def select(self, indices):
        return [self.samples[i] for i in indices]


def test_is_hf_repo_id():
    # Valid HF repo identifiers
    assert is_hf_repo_id("KhangTruong/sid-unet") is True
    assert is_hf_repo_id("hf://KhangTruong/sid-unet") is True
    assert is_hf_repo_id("https://huggingface.co/KhangTruong/sid-unet") is True
    assert is_hf_repo_id("KhangTruong/sid-unet:checkpoint_best.pt") is True

    # Invalid / Local paths
    assert is_hf_repo_id("") is False
    assert is_hf_repo_id(None) is False
    assert is_hf_repo_id("/absolute/path/to/model.pt") is False
    assert is_hf_repo_id("./relative/path.pt") is False
    assert is_hf_repo_id("../parent/path.pt") is False
    assert is_hf_repo_id("outputs/checkpoints.pt") is False


def test_parse_hf_repo_uri():
    repo, fname = parse_hf_repo_uri("hf://KhangTruong/sid-unet:checkpoint_best.pt")
    assert repo == "KhangTruong/sid-unet"
    assert fname == "checkpoint_best.pt"

    repo, fname = parse_hf_repo_uri("https://huggingface.co/KhangTruong/sid-unet")
    assert repo == "KhangTruong/sid-unet"
    assert fname is None

    repo, fname = parse_hf_repo_uri("KhangTruong/sid-unet:checkpoint_latest.pt")
    assert repo == "KhangTruong/sid-unet"
    assert fname == "checkpoint_latest.pt"

    repo, fname = parse_hf_repo_uri("KhangTruong/sid-unet/")
    assert repo == "KhangTruong/sid-unet"
    assert fname is None


def test_download_hf_checkpoint_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_ckpt_file = os.path.join(tmpdir, "downloaded_ckpt.pt")
        torch.save({"epoch": 5, "step": 500, "model_state_dict": {}}, fake_ckpt_file)

        mock_download = MagicMock(return_value=fake_ckpt_file)
        with patch("huggingface_hub.hf_hub_download", mock_download):
            res = download_hf_checkpoint("KhangTruong/sid-unet", filename="checkpoint_latest.pt")
            assert res["checkpoint_path"] == fake_ckpt_file
            assert res["repo_id"] == "KhangTruong/sid-unet"
            assert res["filename"] == "checkpoint_latest.pt"
            assert res["source"] == "huggingface_repo"
            mock_download.assert_called_once()


def test_inspect_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")
        state = {
            "epoch": 4,
            "step": 420,
            "best_score": 0.875,
            "best_epoch": 4,
            "metrics": {"val_iou": 0.875},
            "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}, {"epoch": 4}],
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
        }
        torch.save(state, ckpt_path)

        meta = inspect_checkpoint(ckpt_path)
        assert meta["epoch"] == 4
        assert meta["step"] == 420
        assert meta["best_score"] == 0.875
        assert meta["best_epoch"] == 4
        assert meta["has_optimizer"] is True
        assert meta["has_scheduler"] is True
        assert meta["has_scaler"] is True
        assert meta["has_history"] is True
        assert len(meta["history"]) == 4


def test_find_auto_resume_checkpoint_priority():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        latest_path = os.path.join(ckpt_dir, "checkpoint_latest.pt")
        best_path = os.path.join(ckpt_dir, "checkpoint_best.pt")
        periodic_path = os.path.join(ckpt_dir, "checkpoint_periodic.pt")

        torch.save({"epoch": 3, "step": 300, "model_state_dict": {}}, best_path)
        torch.save({"epoch": 4, "step": 400, "model_state_dict": {}}, periodic_path)
        torch.save({"epoch": 5, "step": 500, "model_state_dict": {}}, latest_path)

        # 1. When all exist, checkpoint_latest.pt has top priority
        found = find_auto_resume_checkpoint(output_dir=tmpdir)
        assert found is not None
        assert found["filename"] == "checkpoint_latest.pt"
        assert found["epoch"] == 5

        # 2. When latest removed, periodic has priority over best
        os.remove(latest_path)
        found = find_auto_resume_checkpoint(output_dir=tmpdir)
        assert found is not None
        assert found["filename"] == "checkpoint_periodic.pt"
        assert found["epoch"] == 4

        # 3. When periodic removed, best is picked
        os.remove(periodic_path)
        found = find_auto_resume_checkpoint(output_dir=tmpdir)
        assert found is not None
        assert found["filename"] == "checkpoint_best.pt"
        assert found["epoch"] == 3

        # 4. When empty, returns None
        os.remove(best_path)
        found = find_auto_resume_checkpoint(output_dir=tmpdir)
        assert found is None


def test_resume_notifications_formatting():
    info = {
        "checkpoint_path": "outputs/RUN/test_exp/checkpoints/checkpoint_latest.pt",
        "epoch": 5,
        "step": 1250,
        "best_score": 0.884,
        "source": "local_repo",
    }
    banner = format_resume_notification(info)
    assert "[AUTO-RESUME]" in banner
    assert "Epoch 6" in banner
    assert "1250" in banner
    assert "0.8840" in banner

    no_banner = format_no_resume_notification("outputs/my_run")
    assert "[AUTO-RESUME]" in no_banner
    assert "outputs/my_run" in no_banner
    assert "Epoch 1" in no_banner


def test_trainer_resume_from_checkpoint_and_continue_training():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=3",
            "training.batch_size=2",
            "training.save_latest=true",
            "model.features=[8, 16]",
            "data.image_size=[32, 32]",
            "logging.log_interval=1",
            "logging.save_sample_images=false",
            "training.amp=false",
        ])

        train_loader = DataLoader(DummyDataset(size=4), batch_size=2)
        val_loader = DataLoader(DummyDataset(size=2), batch_size=2)

        # 1. Run first training session for 2 epochs
        cfg.training.epochs = 2
        trainer1 = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader)
        res1 = trainer1.train()
        assert len(res1["history"]) == 2
        latest_ckpt = os.path.join(tmpdir, "checkpoints", "checkpoint_latest.pt")
        assert os.path.exists(latest_ckpt)

        # 2. Resume in a new Trainer instance with target of 4 epochs
        cfg.training.epochs = 4
        trainer2 = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader)
        resume_meta = trainer2.resume_from_checkpoint(latest_ckpt)

        assert resume_meta["epoch"] == 2
        assert trainer2.start_epoch == 2
        assert len(trainer2.history) == 2

        # 3. Train the remaining epochs (epochs 3 and 4)
        res2 = trainer2.train()
        assert len(res2["history"]) == 4
        assert res2["history"][0]["epoch"] == 1
        assert res2["history"][1]["epoch"] == 2
        assert res2["history"][2]["epoch"] == 3
        assert res2["history"][3]["epoch"] == 4


def test_train_single_run_auto_resume(monkeypatch, capsys):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(10))
    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase 1: Train 1 epoch with save_latest enabled
        res1 = train_single_run(
            config_path="configs/test_smoke.yaml",
            overrides=[
                f"project.output_dir={tmpdir}",
                "project.device=cpu",
                "training.epochs=1",
                "training.batch_size=2",
                "training.save_latest=true",
                "data.num_workers=0",
                "data.train_samples_per_epoch=2",
                "data.val_samples=2",
                "model.features=[8, 16]",
                "data.image_size=[32, 32]",
                "training.amp=false",
            ],
            auto_resume=False,
            skip_collision=False,
        )
        assert res1["best_epoch"] == 1

        # Phase 2: Run train_single_run with auto_resume=True targeting 2 epochs
        res2 = train_single_run(
            config_path="configs/test_smoke.yaml",
            overrides=[
                f"project.output_dir={tmpdir}",
                "project.device=cpu",
                "training.epochs=2",
                "training.batch_size=2",
                "training.save_latest=true",
                "data.num_workers=0",
                "data.train_samples_per_epoch=2",
                "data.val_samples=2",
                "model.features=[8, 16]",
                "data.image_size=[32, 32]",
                "training.amp=false",
            ],
            auto_resume=True,
            skip_collision=False,
        )

        captured = capsys.readouterr()
        assert "[AUTO-RESUME]" in captured.out
        assert "Found existing checkpoint in repository!" in captured.out
        assert len(res2["history"]) == 2
        assert res2["history"][1]["epoch"] == 2


def test_cli_auto_resume_and_no_auto_resume(monkeypatch, capsys):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(10))
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. First run
        run_args_1 = [
            "sid-train",
            "--config", "configs/test_smoke.yaml",
            "--output_dir", tmpdir,
            "--override",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "training.save_latest=true",
            "data.num_workers=0",
            "data.train_samples_per_epoch=2",
            "data.val_samples=2",
            "model.features=[8, 16]",
            "data.image_size=[32, 32]",
            "training.amp=false",
        ]
        monkeypatch.setattr(sys, "argv", run_args_1)
        res1 = train_main()
        assert res1["best_epoch"] == 1

        # 2. Second run with --no-auto-resume (should train from scratch and NOT resume)
        run_args_no_resume = [
            "sid-train",
            "--config", "configs/test_smoke.yaml",
            "--output_dir", tmpdir,
            "--no-auto-resume",
            "--no-skip-collision",
            "--override",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "training.save_latest=true",
            "data.num_workers=0",
            "data.train_samples_per_epoch=2",
            "data.val_samples=2",
            "model.features=[8, 16]",
            "data.image_size=[32, 32]",
            "training.amp=false",
        ]
        monkeypatch.setattr(sys, "argv", run_args_no_resume)
        capsys.readouterr()  # Clear buffer
        res2 = train_main()
        captured = capsys.readouterr()
        # Should not have resumed
        assert "Found existing checkpoint in repository!" not in captured.out


def test_cli_resume_repo_flag(monkeypatch, capsys):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(10))
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock downloaded checkpoint
        fake_ckpt = os.path.join(tmpdir, "hf_downloaded.pt")
        torch.save({
            "epoch": 2,
            "step": 20,
            "best_score": 0.9,
            "best_epoch": 2,
            "model_state_dict": {},
            "history": [{"epoch": 1}, {"epoch": 2}],
        }, fake_ckpt)

        mock_download = MagicMock(return_value={"checkpoint_path": fake_ckpt, "repo_id": "test/model", "source": "huggingface_repo"})
        monkeypatch.setattr("sid_unet.train.download_hf_checkpoint", mock_download)

        run_args = [
            "sid-train",
            "--config", "configs/test_smoke.yaml",
            "--output_dir", os.path.join(tmpdir, "out"),
            "--resume-repo", "test/model",
            "--no-skip-collision",
            "--override",
            "project.device=cpu",
            "training.epochs=3",
            "training.batch_size=2",
            "training.save_latest=true",
            "data.num_workers=0",
            "data.train_samples_per_epoch=2",
            "data.val_samples=2",
            "model.features=[8, 16]",
            "data.image_size=[32, 32]",
            "training.amp=false",
        ]
        monkeypatch.setattr(sys, "argv", run_args)
        capsys.readouterr()
        res = train_main()
        captured = capsys.readouterr()
        assert "[AUTO-RESUME]" in captured.out
        assert "Hugging Face Hub (test/model)" in captured.out
        assert len(res["history"]) == 3
        assert res["history"][2]["epoch"] == 3
