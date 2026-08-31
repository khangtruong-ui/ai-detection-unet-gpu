"""
Tests for memory management, OOM error handling, gradient accumulation,
gradient checkpointing, and auto batch sizing in SID-UNet.
"""

from __future__ import annotations

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sid_unet.models.unet import UNet, build_model
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.utils.memory import (
    auto_scale_batch_size_and_grad_accum,
    clear_memory_cache,
    find_optimal_batch_size,
    format_memory_summary,
    get_memory_summary,
    is_oom_error,
    split_batch,
)


class DummyDataset(Dataset):
    def __init__(self, size: int = 8, img_size: tuple = (64, 64)):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        lbl = idx % 3
        img = torch.randn(3, *self.img_size)
        mask = torch.zeros(1, *self.img_size)
        if lbl == 1:
            mask.fill_(1.0)
        elif lbl == 2:
            mask[:, : self.img_size[0] // 2, :] = 1.0

        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(lbl, dtype=torch.long),
            "img_id": f"dummy_{idx}",
        }


def test_is_oom_error_detection():
    # Standard Python MemoryError
    assert is_oom_error(MemoryError("Cannot allocate memory")) is True

    # RuntimeError with OOM patterns
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")) is True
    assert is_oom_error(RuntimeError("cuda error: out of memory")) is True
    assert is_oom_error(RuntimeError("CUDA error: out of memory during alloc")) is True
    assert is_oom_error(RuntimeError("allocate memory failed")) is True

    # PyTorch OutOfMemoryError if present
    if hasattr(torch.cuda, "OutOfMemoryError"):
        assert is_oom_error(torch.cuda.OutOfMemoryError("CUDA OOM")) is True

    # Non-OOM errors
    assert is_oom_error(ValueError("Invalid argument")) is False
    assert is_oom_error(RuntimeError("Shape mismatch")) is False
    assert is_oom_error(KeyError("image")) is False


def test_clear_memory_cache_execution():
    clear_memory_cache("cpu")
    if torch.cuda.is_available():
        clear_memory_cache("cuda")


def test_memory_summary_and_formatting():
    summary = get_memory_summary("cpu")
    assert isinstance(summary, dict)
    assert "allocated_mb" in summary
    assert "device_type" in summary

    fmt = format_memory_summary("cpu")
    assert isinstance(fmt, str)
    assert len(fmt) > 0


def test_split_batch_functionality():
    batch = {
        "image": torch.randn(6, 3, 32, 32),
        "mask": torch.zeros(6, 1, 32, 32),
        "label": torch.tensor([0, 1, 2, 0, 1, 2]),
        "img_id": ["id_0", "id_1", "id_2", "id_3", "id_4", "id_5"],
        "scalar_meta": "constant_value",
    }

    # Split into chunks of 2
    sub_batches = split_batch(batch, micro_batch_size=2)
    assert len(sub_batches) == 3

    for sub in sub_batches:
        assert sub["image"].shape == (2, 3, 32, 32)
        assert sub["mask"].shape == (2, 1, 32, 32)
        assert len(sub["label"]) == 2
        assert len(sub["img_id"]) == 2
        assert sub["scalar_meta"] == "constant_value"

    # Concatenate back and verify equality
    reconstructed_img = torch.cat([s["image"] for s in sub_batches], dim=0)
    assert torch.equal(reconstructed_img, batch["image"])

    # Split with uneven size (e.g., chunk size 4 on batch of 6 -> chunks of 4 and 2)
    uneven_splits = split_batch(batch, micro_batch_size=4)
    assert len(uneven_splits) == 2
    assert uneven_splits[0]["image"].shape[0] == 4
    assert uneven_splits[1]["image"].shape[0] == 2


def test_auto_scale_batch_size_and_grad_accum():
    # 64 requested with safe 16 -> batch_size=16, grad_accum=4 (effective 64)
    bs, accum = auto_scale_batch_size_and_grad_accum(requested_batch_size=64, safe_batch_size=16, current_grad_accum=1)
    assert bs == 16
    assert accum == 4
    assert bs * accum == 64

    # 32 requested with safe 32 -> batch_size=32, grad_accum=1
    bs2, accum2 = auto_scale_batch_size_and_grad_accum(requested_batch_size=32, safe_batch_size=32, current_grad_accum=1)
    assert bs2 == 32
    assert accum2 == 1

    # 32 requested with existing grad_accum=2 (effective 64) and safe 16 -> bs=16, grad_accum=4 (effective 64)
    bs3, accum3 = auto_scale_batch_size_and_grad_accum(requested_batch_size=32, safe_batch_size=16, current_grad_accum=2)
    assert bs3 == 16
    assert accum3 == 4


def test_find_optimal_batch_size():
    model = UNet(features=[16, 32], dropout=0.0, aux_classifier=True)
    optimal_bs = find_optimal_batch_size(
        model=model,
        sample_shape=(3, 64, 64),
        device=torch.device("cpu"),
        max_batch_size=16,
        min_batch_size=1,
    )
    assert optimal_bs >= 1


def test_gradient_checkpointing_forward_backward():
    model = UNet(
        features=[16, 32],
        dropout=0.0,
        aux_classifier=True,
        gradient_checkpointing=True,
    )
    model.train()

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    mask_out, cls_out = model(x)
    assert mask_out.shape == (2, 1, 64, 64)
    assert cls_out.shape == (2, 3)

    loss = mask_out.mean() + cls_out.mean()
    loss.backward()

    # Ensure all convolutional weights received gradients
    assert model.inc.conv1.weight.grad is not None
    assert model.bottleneck.mpconv[1].conv1.weight.grad is not None
    assert model.outc.conv.weight.grad is not None


def test_gradient_accumulation_trainer():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "training.gradient_accumulation_steps=2",
            "model.features=[16, 32]",
            "data.image_size=[64, 64]",
            "logging.log_interval=1",
            "training.amp=false",
        ])

        train_ds = DummyDataset(size=8, img_size=(64, 64))
        val_ds = DummyDataset(size=4, img_size=(64, 64))

        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(val_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        results = trainer.train()
        assert "best_score" in results
        assert results["history"][0]["epoch"] == 1


def test_oom_recovery_sub_batching_in_trainer(monkeypatch):
    """Test that when a large batch triggers OOM, Trainer catches and recovers via micro-batching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=4",
            "model.features=[16, 32]",
            "data.image_size=[64, 64]",
            "logging.log_interval=1",
            "training.amp=false",
        ])

        train_ds = DummyDataset(size=4, img_size=(64, 64))
        val_ds = DummyDataset(size=2, img_size=(64, 64))

        train_loader = DataLoader(train_ds, batch_size=4)
        val_loader = DataLoader(val_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        original_step = trainer._step_batch_train
        oom_triggered = {"count": 0}

        def mock_step_batch_train(batch, loss_divisor=1.0):
            # Trigger simulated OOM only if batch size is 4
            if len(batch["image"]) == 4:
                oom_triggered["count"] += 1
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.0 GiB")
            return original_step(batch, loss_divisor)

        monkeypatch.setattr(trainer, "_step_batch_train", mock_step_batch_train)

        # Training should not crash, but should recover and complete!
        results = trainer.train()
        assert oom_triggered["count"] >= 1
        assert "best_score" in results
