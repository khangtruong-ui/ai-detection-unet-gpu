"""
Unit tests for validation GPU efficiency, bottleneck warning start timing,
periodic checkpoint timer reset, and safe batch size probing.
"""

from __future__ import annotations

import time
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from unittest.mock import MagicMock

from sid_unet.dataset.loader import CUDAPrefetcher
from sid_unet.postprocessing import MaskPostProcessor
from sid_unet.training.callbacks import CheckpointManager
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.utils.network import BottleneckDetector


class DummyDataset(Dataset):
    def __init__(self, size: int = 8, img_size: tuple = (32, 32)):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return {
            "image": torch.randn(3, *self.img_size),
            "mask": torch.zeros(1, *self.img_size),
            "label": torch.tensor(0, dtype=torch.long),
            "img_id": f"dummy_{idx}",
        }


def test_periodic_checkpoint_timer_resets():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            checkpoint_dir=tmpdir,
            checkpoint_period=3600.0,  # 1 hour period
        )

        # Simulate 30 minutes elapsed in epoch 1
        mgr.last_periodic_save_time = time.time() - 1800.0
        assert not mgr.should_save_periodic(step=10)

        # Start of epoch 2 resets timer
        mgr.reset_periodic_timer()
        assert (time.time() - mgr.last_periodic_save_time) < 1.0
        assert not mgr.should_save_periodic(step=20)

        # If period has elapsed (e.g. 3601s in a long 2-hour epoch), it triggers
        mgr.last_periodic_save_time = time.time() - 3601.0
        assert mgr.should_save_periodic(step=30)


def test_bottleneck_warning_timing_in_trainer(monkeypatch):
    """Test that BottleneckDetector.record_step is only called after step 2 of the first epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=2",
            "training.batch_size=2",
            "model.features=[16, 32]",
            "data.image_size=[32, 32]",
            "logging.log_interval=1",
            "training.amp=false",
            "training.detect_bottlenecks=true",
        ])

        train_ds = DummyDataset(size=8, img_size=(32, 32))  # 4 steps per epoch
        val_ds = DummyDataset(size=2, img_size=(32, 32))

        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(val_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        mock_detector = MagicMock(spec=BottleneckDetector)
        trainer.bottleneck_detector = mock_detector

        recorded_calls = []
        def mock_record_step(data_time, compute_time):
            recorded_calls.append((trainer._current_epoch if hasattr(trainer, "_current_epoch") else 1, len(recorded_calls) + 1))

        mock_detector.record_step.side_effect = mock_record_step

        # Train epoch 1 (first epoch)
        trainer.start_epoch = 1
        trainer.train_epoch(epoch=1)

        # In epoch 1 with 4 steps: steps 1 and 2 are skipped, steps 3 and 4 are recorded (2 calls)
        assert mock_detector.record_step.call_count == 2

        # Train epoch 2 (subsequent epoch)
        mock_detector.reset_mock()
        trainer.train_epoch(epoch=2)

        # In epoch 2 with 4 steps: all 4 steps are recorded (4 calls)
        assert mock_detector.record_step.call_count == 4


def test_cuda_prefetcher_iteration():
    ds = DummyDataset(size=6, img_size=(16, 16))
    loader = DataLoader(ds, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefetcher = CUDAPrefetcher(loader, device=device)

    assert len(prefetcher) == 3
    batches = list(prefetcher)
    assert len(batches) == 3

    for b in batches:
        assert b["image"].shape == (2, 3, 16, 16)
        assert b["image"].device.type == device.type


def test_mask_postprocessor_batch_execution():
    post = MaskPostProcessor(enabled=True, min_area=10, fill_holes=True)
    masks = torch.zeros((4, 1, 32, 32))
    # Add small blob (area = 2) in sample 0
    masks[0, 0, 0:2, 0:1] = 1.0
    # Add large blob (area = 16) in sample 1
    masks[1, 0, 10:14, 10:14] = 1.0

    proc_batch, stats = post.process_batch(masks, is_logit=False)
    assert proc_batch.shape == (4, 1, 32, 32)
    assert len(stats) == 4
    # Small component should be removed from sample 0
    assert proc_batch[0].sum().item() == 0.0
    # Large component preserved in sample 1
    assert proc_batch[1].sum().item() > 0.0

    # Test disabled mode fast path
    post_disabled = MaskPostProcessor(enabled=False)
    proc_dis, stats_dis = post_disabled.process_batch(masks, is_logit=False)
    assert proc_dis.shape == (4, 1, 32, 32)
    assert len(stats_dis) == 4
    assert stats_dis[0]["enabled"] is False
