import glob
import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sid_unet.training.hard_mining import HardMiner, HardMiningBatchFilter
from sid_unet.utils.config import load_config
from sid_unet.training.trainer import Trainer


class DummyMapDataset(Dataset):
    def __init__(self, size=20, in_channels=3, img_size=(64, 64)):
        self.size = size
        self.in_channels = in_channels
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate predictable data
        x = torch.full((self.in_channels, *self.img_size), float(idx) / float(self.size))
        y = torch.zeros((1, *self.img_size), dtype=torch.float32)
        if idx % 2 == 0:
            y[:, :32, :32] = 1.0
        return {
            "image": x,
            "mask": y,
            "label": torch.tensor(1 if idx % 2 == 0 else 0, dtype=torch.long),
            "sample_idx": idx,
        }


def test_hard_mining_config_defaults():
    # Test default config
    cfg = load_config("configs/default.yaml")
    assert "hard_mining" in cfg
    assert cfg.hard_mining.enabled is False
    assert cfg.hard_mining.metric == "median"
    assert cfg.hard_mining.reset_epochs == 5

    # Test all diffusion-diff configs have hard mining enabled
    diff_configs = glob.glob("configs/experiments/diffusion_diff*/**/*.yaml", recursive=True)
    assert len(diff_configs) == 6, f"Expected 6 diffusion configs, found {len(diff_configs)}"
    for cfg_path in diff_configs:
        c = load_config(cfg_path)
        assert c.hard_mining.enabled is True, f"{cfg_path} should have hard_mining.enabled=True"
        assert c.hard_mining.metric == "median", f"{cfg_path} should use median metric"
        assert c.hard_mining.reset_epochs == 5, f"{cfg_path} should have reset_epochs=5"
        assert c.data.num_workers == -1, f"{cfg_path} should have num_workers=-1"


def test_hard_miner_unit_filtering():
    cfg = {
        "hard_mining": {
            "enabled": True,
            "metric": "median",
            "reset_epochs": 5,
        }
    }
    miner = HardMiner(cfg)
    assert miner.enabled is True
    assert miner.metric == "median"
    assert miner.reset_epochs == 5

    # Epoch 1: full training (initial epoch of cycle 1)
    assert miner.is_hard_mining_active(epoch=1) is False

    # Simulate 6 batches with losses [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # Median is 3.5. Batches >= 3.5 are indices 3, 4, 5.
    for idx, loss_val in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        miner.record_batch_loss(idx, loss_val)

    miner.on_epoch_end(epoch=1)
    assert miner.threshold == pytest.approx(3.5)
    assert miner.hard_batch_indices == {3, 4, 5}

    # Epoch 2: hard mining active
    assert miner.is_hard_mining_active(epoch=2) is True

    # Test batch filtering on a dummy DataLoader
    dataset = list(range(6))
    loader = DataLoader(dataset, batch_size=1)
    filtered_loader = miner.get_train_dataloader(loader, epoch=2)
    assert isinstance(filtered_loader, HardMiningBatchFilter)
    assert len(filtered_loader) == 3

    yielded_items = [b.item() for b in filtered_loader]
    assert yielded_items == [3, 4, 5]


def test_hard_miner_mean_metric():
    cfg = {
        "hard_mining": {
            "enabled": True,
            "metric": "mean",
            "reset_epochs": 5,
        }
    }
    miner = HardMiner(cfg)
    assert miner.metric == "mean"

    # Losses [1.0, 1.0, 1.0, 7.0]. Mean is 2.5.
    for idx, val in enumerate([1.0, 1.0, 1.0, 7.0]):
        miner.record_batch_loss(idx, val)

    miner.on_epoch_end(epoch=1)
    assert miner.threshold == pytest.approx(2.5)
    assert miner.hard_batch_indices == {3}


def test_hard_miner_cycle_reset_after_5_epochs():
    # If reset_epochs = 5:
    # Epoch 1: full (1 mod 6 = 1) -> active: False
    # Epoch 2: HM 1 (2 mod 6 = 2) -> active: True
    # Epoch 3: HM 2 (3 mod 6 = 3) -> active: True
    # Epoch 4: HM 3 (4 mod 6 = 4) -> active: True
    # Epoch 5: HM 4 (5 mod 6 = 5) -> active: True
    # Epoch 6: HM 5 (6 mod 6 = 0) -> active: True
    # Epoch 7: Reset to full (7 mod 6 = 1) -> active: False
    # Epoch 8: HM 1 (8 mod 6 = 2) -> active: True
    cfg = {
        "hard_mining": {
            "enabled": True,
            "metric": "median",
            "reset_epochs": 5,
        }
    }
    miner = HardMiner(cfg)

    expected_active = {
        1: False,  # First full training
        2: True,   # HM 1
        3: True,   # HM 2
        4: True,   # HM 3
        5: True,   # HM 4
        6: True,   # HM 5
        7: False,  # Reset: Full training repeated
        8: True,   # HM 1 of cycle 2
        9: True,   # HM 2 of cycle 2
        10: True,  # HM 3 of cycle 2
        11: True,  # HM 4 of cycle 2
        12: True,  # HM 5 of cycle 2
        13: False, # Reset again
    }

    for ep, should_be_active in expected_active.items():
        assert miner.is_hard_mining_active(epoch=ep) == should_be_active, (
            f"Epoch {ep} expected active={should_be_active}, got {miner.is_hard_mining_active(epoch=ep)}"
        )
        # Record some dummy losses and advance epoch
        miner.record_batch_loss(0, 1.0)
        miner.record_batch_loss(1, 2.0)
        miner.on_epoch_end(epoch=ep)


def test_hard_miner_state_dict_and_load():
    cfg = {"hard_mining": {"enabled": True, "metric": "median", "reset_epochs": 5}}
    miner = HardMiner(cfg)

    miner.record_batch_loss(0, 10.0)
    miner.record_batch_loss(1, 20.0)
    miner.record_batch_loss(2, 30.0)
    miner.on_epoch_end(epoch=1)

    state = miner.state_dict()
    assert state["enabled"] is True
    assert state["threshold"] == pytest.approx(20.0)
    assert set(state["hard_batch_indices"]) == {1, 2}

    # Load into fresh miner
    new_miner = HardMiner(cfg)
    new_miner.load_state_dict(state)

    assert new_miner.threshold == pytest.approx(20.0)
    assert new_miner.hard_batch_indices == {1, 2}
    assert new_miner.is_hard_mining_active(epoch=2) is True


def test_hard_mining_trainer_7_epochs():
    """End-to-end integration test of Trainer with hard mining over 7 epochs.
    Verifies that:
    - Epoch 1 runs for N batches (full dataset)
    - Epochs 2..6 run for K batches (where K is batches >= median)
    - Epoch 7 runs for N batches (cycle reset)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_size = 2
        num_samples = 16  # 8 batches per full epoch
        dataset = DummyMapDataset(size=num_samples, in_channels=3, img_size=(32, 32))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        cfg = load_config("configs/test_smoke.yaml")
        cfg.project.output_dir = tmpdir
        cfg.data.num_workers = 0
        cfg.data.batch_size = batch_size
        cfg.training.epochs = 7
        cfg.training.data_parallel = False
        cfg.training.save_best = False
        cfg.training.save_latest = True
        cfg.training.eval_interval = 10
        cfg.training.early_stopping_patience = 50
        cfg.hard_mining = {
            "enabled": True,
            "metric": "median",
            "reset_epochs": 5,
        }
        cfg.training.use_hard_mining = True

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        batches_per_epoch = []
        original_train_epoch = trainer.train_epoch

        def tracked_train_epoch(epoch):
            active_loader = trainer.hard_miner.get_train_dataloader(trainer.train_loader, epoch=epoch)
            is_active = trainer.hard_miner.is_hard_mining_active(epoch)
            batches_per_epoch.append((epoch, len(active_loader), is_active))
            return original_train_epoch(epoch)

        trainer.train_epoch = tracked_train_epoch

        # Run training through train()
        metrics = trainer.train()

        # Check recorded batch counts per epoch
        # Total samples = 16, batch_size = 2 -> 8 batches full epoch
        assert len(batches_per_epoch) == 7
        assert batches_per_epoch[0] == (1, 8, False)  # Epoch 1: full (longest)
        assert batches_per_epoch[1] == (2, 4, True)   # Epoch 2: hard mining (half iterations)
        assert batches_per_epoch[2] == (3, 4, True)   # Epoch 3: hard mining
        assert batches_per_epoch[3] == (4, 4, True)   # Epoch 4: hard mining
        assert batches_per_epoch[4] == (5, 4, True)   # Epoch 5: hard mining
        assert batches_per_epoch[5] == (6, 4, True)   # Epoch 6: hard mining (5th HM epoch)
        assert batches_per_epoch[6] == (7, 8, False)  # Epoch 7: cycle reset to full training!

        # Let's inspect history and checkpoint
        ckpt_path = os.path.join(tmpdir, "checkpoints", "checkpoint_latest.pt")
        assert os.path.exists(ckpt_path), "Latest checkpoint should exist"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "hard_mining" in ckpt, "Checkpoint should contain hard_mining state"
        assert ckpt["hard_mining"]["enabled"] is True

        # Verify trainer's hard miner status
        assert trainer.hard_miner.enabled is True


def test_resume_preserves_hard_mining_state():
    """Verify that resuming training from checkpoint restores HardMiner state exactly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_size = 2
        dataset = DummyMapDataset(size=8, in_channels=3, img_size=(32, 32))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        cfg = load_config("configs/test_smoke.yaml")
        cfg.project.output_dir = tmpdir
        cfg.data.num_workers = 0
        cfg.data.batch_size = batch_size
        cfg.training.epochs = 2
        cfg.training.data_parallel = False
        cfg.training.save_latest = True
        cfg.training.save_best = False
        cfg.hard_mining = {
            "enabled": True,
            "metric": "median",
            "reset_epochs": 5,
        }
        cfg.training.use_hard_mining = True

        trainer = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader)
        trainer.train()

        # Hard miner should have threshold and hard batch indices from epoch 1
        assert len(trainer.hard_miner.hard_batch_indices) > 0
        orig_indices = set(trainer.hard_miner.hard_batch_indices)
        orig_threshold = trainer.hard_miner.threshold

        # Now create a second trainer configured to resume
        cfg_resume = load_config("configs/test_smoke.yaml")
        cfg_resume.project.output_dir = tmpdir
        cfg_resume.data.num_workers = 0
        cfg_resume.data.batch_size = batch_size
        cfg_resume.training.epochs = 4
        cfg_resume.training.data_parallel = False
        cfg_resume.training.resume = os.path.join(tmpdir, "checkpoints", "checkpoint_latest.pt")
        cfg_resume.hard_mining = {
            "enabled": True,
            "metric": "median",
            "reset_epochs": 5,
        }
        cfg_resume.training.use_hard_mining = True

        trainer_resume = Trainer(config=cfg_resume, train_loader=train_loader, val_loader=val_loader)
        trainer_resume.resume_from_checkpoint(cfg_resume.training.resume)

        assert trainer_resume.start_epoch == 2
        assert trainer_resume.hard_miner.enabled is True
        assert trainer_resume.hard_miner.hard_batch_indices == orig_indices
        assert trainer_resume.hard_miner.threshold == pytest.approx(orig_threshold)
        assert trainer_resume.hard_miner.is_hard_mining_active(epoch=3) is True

