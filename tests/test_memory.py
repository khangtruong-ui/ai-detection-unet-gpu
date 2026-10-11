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
    MemoryGuard,
    auto_scale_batch_size_and_grad_accum,
    clear_memory_cache,
    estimate_memory_profile,
    find_optimal_batch_size,
    format_memory_summary,
    get_memory_summary,
    is_oom_error,
    predict_safe_batch_size,
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


def test_find_optimal_batch_size_latent_with_mask_shape():
    """Verify that find_optimal_batch_size reconciles differing input and mask spatial resolutions."""
    class MockLatentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(84, 1, kernel_size=1)
            self.aux_classifier = False

        def forward(self, x):
            # Input is (B, 84, 32, 32), output is upsampled to (B, 1, 256, 256)
            feat = self.conv(x)
            return nn.functional.interpolate(feat, size=(256, 256), mode="bilinear")

    model = MockLatentModel()
    from sid_unet.losses.combined import CombinedMaskLoss
    loss_fn = CombinedMaskLoss()

    optimal_bs = find_optimal_batch_size(
        model=model,
        loss_fn=loss_fn,
        sample_shape=(84, 32, 32),
        mask_shape=(256, 256),
        device=torch.device("cpu"),
        max_batch_size=4,
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


def test_oom_raises_in_train_epoch(monkeypatch):
    """Test that when a batch triggers OOM in train_epoch, it raises immediately without micro-batch retry."""
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

        def mock_step_batch_train(batch, loss_divisor=1.0):
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.0 GiB")

        monkeypatch.setattr(trainer, "_step_batch_train", mock_step_batch_train)

        # Training should raise the OOM error directly
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            trainer.train()


def test_memory_summary_fractions():
    """Verify usage_fraction and peak_fraction fields in memory summary."""
    summary_cpu = get_memory_summary("cpu")
    assert "usage_fraction" in summary_cpu
    assert "peak_fraction" in summary_cpu
    assert 0.0 <= summary_cpu["usage_fraction"] <= 1.0

    if torch.cuda.is_available():
        summary_cuda = get_memory_summary("cuda")
        assert "usage_fraction" in summary_cuda
        assert "peak_fraction" in summary_cuda
        assert 0.0 <= summary_cuda["usage_fraction"] <= 1.0
        assert summary_cuda["total_mb"] > 0


def test_memory_guard_core_functionality():
    """Verify MemoryGuard lifecycle: status, threshold checks, recovery, and batch scaling."""
    guard = MemoryGuard(device="cpu", max_memory_fraction=0.80, warning_threshold=0.75)
    assert guard.max_memory_fraction == 0.80
    assert guard.warning_threshold == 0.75

    status = guard.get_memory_status()
    assert isinstance(status, dict)
    assert "usage_fraction" in status

    # Verify batch scaling recommendation preserves effective batch size
    scaled_bs, scaled_accum = guard.recommend_batch_scaling(current_batch_size=16, current_grad_accum=1)
    assert scaled_bs == 8
    assert scaled_accum == 2
    assert scaled_bs * scaled_accum == 16

    # Verify recover_memory calls cache clear without exception
    new_frac = guard.recover_memory()
    assert isinstance(new_frac, float)

    # Verify format_status returns string representation
    fmt = guard.format_status()
    assert isinstance(fmt, str)
    assert "%" in fmt


def test_memory_guard_threshold_detection(monkeypatch):
    """Verify MemoryGuard correctly identifies when memory limits are exceeded."""
    guard = MemoryGuard(device="cpu", max_memory_fraction=0.80)

    # Mock normal memory state (60% usage)
    monkeypatch.setattr(
        guard,
        "get_memory_status",
        lambda: {"usage_fraction": 0.60, "peak_fraction": 0.60, "used_mb": 6000.0, "total_mb": 10000.0},
    )
    is_exceeded, frac = guard.check_memory(step=1)
    assert is_exceeded is False
    assert frac == 0.60
    assert guard.is_memory_exceeded() is False

    # Mock exceeded memory state (85% usage)
    monkeypatch.setattr(
        guard,
        "get_memory_status",
        lambda: {"usage_fraction": 0.85, "peak_fraction": 0.85, "used_mb": 8500.0, "total_mb": 10000.0},
    )
    is_exceeded, frac = guard.check_memory(step=2)
    assert is_exceeded is True
    assert frac == 0.85
    assert guard.is_memory_exceeded() is True


def test_estimate_memory_profile_and_predict_safe_batch_size():
    """Verify linear memory profiling and predictive batch size calculation."""
    model = UNet(features=[16, 32], dropout=0.0, aux_classifier=True)
    profile = estimate_memory_profile(
        model=model,
        sample_shape=(3, 64, 64),
        device=torch.device("cpu"),
    )
    assert isinstance(profile, dict)
    assert "baseline_mb" in profile
    assert "per_sample_mb" in profile
    assert "total_mb" in profile
    assert profile["baseline_mb"] > 0
    assert profile["per_sample_mb"] > 0

    predicted_bs = predict_safe_batch_size(
        model=model,
        sample_shape=(3, 64, 64),
        max_memory_fraction=0.80,
        device=torch.device("cpu"),
        max_batch_size=32,
        min_batch_size=1,
    )
    assert isinstance(predicted_bs, int)
    assert 1 <= predicted_bs <= 32


def test_find_optimal_batch_size_enforces_memory_limit(monkeypatch):
    """Verify find_optimal_batch_size rejects candidates that exceed max_memory_fraction."""
    model = UNet(features=[16, 32], dropout=0.0, aux_classifier=True)

    # When running on CPU with force_probe=True and mocked memory returning high memory for large batch
    class MemoryStateTracker:
        def __init__(self):
            self.probed_sizes = []

    tracker = MemoryStateTracker()

    import sid_unet.utils.memory as mem_mod
    orig_get_summary = mem_mod.get_memory_summary

    def mock_get_memory_summary(device=None):
        base = orig_get_summary(device)
        # Pretend host RAM is 85% full
        base["usage_fraction"] = 0.85
        return base

    monkeypatch.setattr(mem_mod, "get_memory_summary", mock_get_memory_summary)

    # With max_memory_fraction=0.80 and current usage 85%, should fall back to min_batch_size
    optimal_bs = find_optimal_batch_size(
        model=model,
        sample_shape=(3, 64, 64),
        device=torch.device("cpu"),
        max_batch_size=16,
        min_batch_size=1,
        max_memory_fraction=0.80,
    )
    assert optimal_bs == 1


def test_trainer_with_max_memory_fraction():
    """Verify Trainer correctly stores max_memory_fraction and configures MemoryGuard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "training.max_memory_fraction=0.75",
            "model.features=[16, 32]",
            "data.image_size=[64, 64]",
            "logging.log_interval=1",
            "training.amp=false",
        ])

        train_ds = DummyDataset(size=4, img_size=(64, 64))
        val_ds = DummyDataset(size=2, img_size=(64, 64))

        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(val_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        assert trainer.max_memory_fraction == 0.75
        assert trainer.memory_guard.max_memory_fraction == 0.75
        results = trainer.train()
        assert "best_score" in results


def test_is_oom_error_recognizes_cublas_and_cudnn_status_errors():
    """Verify is_oom_error correctly identifies cuBLAS and cuDNN allocation / workspace failures from .logs.txt."""
    cublas_sgemm_err = RuntimeError(
        "CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)"
    )
    cublas_lt_err = RuntimeError(
        "gemm_and_bias error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasLtMatmul with transpose_mat1 1 transpose_mat2 0 m 256 n 3208 k 2048"
    )
    cublas_alloc_err = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
    cudnn_alloc_err = RuntimeError("CUDNN_STATUS_ALLOC_FAILED")
    cudnn_internal_err = RuntimeError("CUDNN_STATUS_INTERNAL_ERROR")
    cuda_mem_err = RuntimeError("cudaErrorMemoryAllocation")

    assert is_oom_error(cublas_sgemm_err) is True
    assert is_oom_error(cublas_lt_err) is True
    assert is_oom_error(cublas_alloc_err) is True
    assert is_oom_error(cudnn_alloc_err) is True
    assert is_oom_error(cudnn_internal_err) is True
    assert is_oom_error(cuda_mem_err) is True


def test_find_optimal_batch_size_recovers_from_cublas_internal_error(monkeypatch):
    """
    Verify that when candidate batch size 16 fails with CUBLAS_STATUS_INTERNAL_ERROR
    (as seen in .logs.txt), find_optimal_batch_size cleans up and recovers with a smaller batch size.
    """
    class MockModelWithCublasFailure(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
            self.aux_classifier = False

        def forward(self, x):
            if x.shape[0] > 4:
                raise RuntimeError(
                    "CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)"
                )
            return self.linear(x.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)

    model = MockModelWithCublasFailure()
    safe_bs = find_optimal_batch_size(
        model=model,
        sample_shape=(3, 32, 32),
        device=torch.device("cpu"),
        max_batch_size=16,
        min_batch_size=1,
        force_probe=True,
    )
    # Batch size 16 and 8 should fail with CUBLAS error, recovering at batch size <= 4
    assert safe_bs <= 4
    assert safe_bs >= 1


def test_trainer_auto_scale_fallback_on_probe_exception(monkeypatch):
    """
    Verify Trainer._check_and_auto_scale_batch_size falls back to safe batch size
    if find_optimal_batch_size raises an unexpected exception, preventing OOM crash on step 0.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=16",
            "training.auto_batch_size=true",
            "training.gradient_accumulation_steps=1",
            "model.features=[16, 32]",
            "data.image_size=[32, 32]",
            "training.amp=false",
        ])

        train_ds = DummyDataset(size=16, img_size=(32, 32))
        val_ds = DummyDataset(size=4, img_size=(32, 32))

        train_loader = DataLoader(train_ds, batch_size=16)
        val_loader = DataLoader(val_ds, batch_size=4)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        import sid_unet.training.trainer as tr_mod
        def mock_failing_probe(*args, **kwargs):
            raise RuntimeError("CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasSgemm")

        monkeypatch.setattr(tr_mod, "find_optimal_batch_size", mock_failing_probe)

        # Force device type check bypass for test on CPU
        from unittest.mock import MagicMock
        mock_dev = MagicMock()
        mock_dev.type = "cuda"
        monkeypatch.setattr(trainer, "device", mock_dev)

        # Execute auto scale probe with failing probe
        trainer._check_and_auto_scale_batch_size()

        # Should NOT remain at unsafe batch_size=16; should fall back to safe batch size <= 4
        assert trainer.config.data.batch_size <= 4
        assert trainer.gradient_accumulation_steps >= 4
        assert trainer.config.data.batch_size * trainer.gradient_accumulation_steps == 16

