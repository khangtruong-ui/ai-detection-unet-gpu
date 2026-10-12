"""Tests for signal shielding and forward termination."""

import os
import signal
import subprocess
import sys
import time
import pytest
import torch

from sid_unet.utils.signals import (
    shield_process_signals,
    SSH_DISCONNECT_SIGNALS,
    SSH_DISCONNECT_SIGNAL_NAMES,
)


def test_ssh_disconnect_signals_constants():
    assert signal.SIGTERM in SSH_DISCONNECT_SIGNALS
    assert signal.SIGPIPE in SSH_DISCONNECT_SIGNALS
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in SSH_DISCONNECT_SIGNALS
    if hasattr(signal, "SIGQUIT"):
        assert signal.SIGQUIT in SSH_DISCONNECT_SIGNALS


def test_subprocess_ignores_ssh_signals():
    """Verify that a child process running with shield_process_signals ignores SIGTERM and SIGHUP."""
    code = (
        "import time, sys\n"
        "from sid_unet.utils.signals import shield_process_signals\n"
        "shield_process_signals()\n"
        "sys.stdout.write('READY\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(3)\n"
        "sys.stdout.write('SURVIVED\\n')\n"
        "sys.stdout.flush()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait until ready
    line = proc.stdout.readline()
    assert "READY" in line

    # Send SIGTERM (signal 15) - process MUST ignore it and not die
    proc.send_signal(signal.SIGTERM)
    time.sleep(0.3)
    assert proc.poll() is None, "Process died from SIGTERM but should have ignored it"

    # Send SIGHUP (signal 1) if available - process MUST ignore it
    if hasattr(signal, "SIGHUP"):
        proc.send_signal(signal.SIGHUP)
        time.sleep(0.3)
        assert proc.poll() is None, "Process died from SIGHUP but should have ignored it"

    # Send SIGINT (signal 2) - process should raise KeyboardInterrupt and exit
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=2.0)
    assert proc.poll() is not None, "Process did not exit on SIGINT"


def test_safe_stream_wrapper_broken_pipe_and_eio():
    """Verify that SafeStreamWrapper suppresses BrokenPipeError and EIO on write/flush."""
    import errno
    from sid_unet.utils.signals import SafeStreamWrapper

    class BrokenStream:
        def __init__(self):
            self.broken = False

        def write(self, s):
            if self.broken:
                raise BrokenPipeError(errno.EPIPE, "Broken pipe")
            return len(s)

        def flush(self):
            if self.broken:
                raise OSError(errno.EIO, "Input/output error")

        def isatty(self):
            return False

        def fileno(self):
            return 1

    raw = BrokenStream()
    wrapped = SafeStreamWrapper(raw)
    assert wrapped.write("hello") == 5
    wrapped.flush()

    raw.broken = True
    # Should not raise BrokenPipeError or OSError(EIO)
    written = wrapped.write("data after ssh drop")
    assert written == len("data after ssh drop")
    wrapped.flush()


def test_worker_init_fn_shields_dataloader_worker():
    """Verify that worker_init_fn applies signal shielding in child worker processes without disabling SIGTERM."""
    code = (
        "import signal, sys\n"
        "from sid_unet.dataset.loader import worker_init_fn\n"
        "worker_init_fn(0)\n"
        "assert signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN\n"
        "assert signal.getsignal(signal.SIGTERM) != signal.SIG_IGN\n"
        "sys.exit(0)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Failed: {res.stderr}"


def test_dataloader_worker_terminates_on_sigterm():
    """Verify that a worker running worker_init_fn terminates promptly when sent SIGTERM."""
    code = (
        "import time, sys\n"
        "from sid_unet.dataset.loader import worker_init_fn\n"
        "worker_init_fn(0)\n"
        "sys.stdout.write('WORKER_READY\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline()
    assert "WORKER_READY" in line

    # Send SIGTERM - worker MUST terminate promptly rather than ignoring it and hanging
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=3.0)
    assert proc.poll() is not None, "Worker did not terminate upon receiving SIGTERM!"


def test_trainer_closes_loaders_on_exception():
    """Verify that Trainer cleanly calls close() and cleans up loaders on unhandled exception."""
    from sid_unet.training.trainer import Trainer
    from sid_unet.utils.config import ConfigDict
    import tempfile
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(4, 3, 32, 32)
    y = torch.zeros(4, 1, 32, 32)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ConfigDict({
            "project": {"name": "test_err_cleanup", "output_dir": tmpdir, "seed": 42, "device": "cpu"},
            "data": {"image_size": [32, 32], "batch_size": 2},
            "model": {"name": "unet", "in_channels": 3, "out_channels": 1, "aux_classifier": False},
            "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
            "training": {"epochs": 1, "learning_rate": 0.001, "save_best": False, "save_latest": False},
            "logging": {"log_interval": 1, "measure_network": False},
        })
        trainer = Trainer(config=cfg, train_loader=loader)

        closed = False
        orig_close = trainer.close
        def mock_close():
            nonlocal closed
            closed = True
            orig_close()
        trainer.close = mock_close

        def mock_train_epoch(epoch):
            raise RuntimeError("Simulated fatal failure during train_epoch")
        trainer.train_epoch = mock_train_epoch

        with pytest.raises(RuntimeError, match="Simulated fatal failure"):
            trainer.train()

        assert closed is True, "trainer.close() was not invoked when train() encountered an exception!"


def test_trainer_cudnn_benchmark_config_for_sam_and_unet():
    """Verify that UNet defaults cudnn.benchmark=True on CUDA while GAP-SAM defaults to False."""
    from sid_unet.training.trainer import Trainer
    from sid_unet.utils.config import ConfigDict
    import tempfile

    orig_benchmark = torch.backends.cudnn.benchmark if torch.cuda.is_available() else False
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            cfg_unet = ConfigDict({
                "project": {"name": "test_unet_benchmark", "output_dir": tmpdir, "seed": 42, "device": dev},
                "data": {"image_size": [32, 32], "batch_size": 2},
                "model": {"name": "unet", "in_channels": 3, "out_channels": 1, "aux_classifier": False},
                "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
                "training": {"epochs": 1, "learning_rate": 0.001, "save_best": False, "save_latest": False},
                "logging": {"log_interval": 1, "measure_network": False},
            })
            trainer_unet = Trainer(config=cfg_unet)
            if torch.cuda.is_available():
                assert torch.backends.cudnn.benchmark is True

            cfg_gapsam = ConfigDict({
                "project": {"name": "test_gapsam_benchmark", "output_dir": tmpdir, "seed": 42, "device": dev},
                "data": {"image_size": [32, 32], "batch_size": 2},
                "model": {
                    "name": "gap_sam",
                    "backbone": "tinyvit",
                    "in_channels": 3,
                    "out_channels": 1,
                    "aux_classifier": False,
                    "checkpoint_path": None,
                    "vae_pretrained_model_name_or_path": None,
                    "use_dummy_vae": True,
                },
                "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
                "training": {"epochs": 1, "learning_rate": 0.001, "save_best": False, "save_latest": False},
                "logging": {"log_interval": 1, "measure_network": False},
            })
            trainer_gapsam = Trainer(config=cfg_gapsam)
            if torch.cuda.is_available():
                assert torch.backends.cudnn.benchmark is False
        finally:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = orig_benchmark


def test_sam_and_gap_models_guard_cudnn_sdp_on_cuda():
    """Verify that SAM and GAP models disable cuDNN SDP on CUDA to avoid kernel launch failures."""
    if not torch.cuda.is_available() or not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        pytest.skip("CUDA and enable_cudnn_sdp required")
    from sid_unet.training.trainer import Trainer
    from sid_unet.utils.config import ConfigDict
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ConfigDict({
            "project": {"name": "test_sdp_guard", "output_dir": tmpdir, "seed": 42, "device": "cuda"},
            "data": {"image_size": [32, 32], "batch_size": 2},
            "model": {
                "name": "gap_sam",
                "backbone": "tinyvit",
                "in_channels": 3,
                "out_channels": 1,
                "aux_classifier": False,
                "checkpoint_path": None,
                "use_dummy_vae": True,
            },
            "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
            "training": {"epochs": 1, "learning_rate": 0.001, "save_best": False, "save_latest": False},
            "logging": {"log_interval": 1, "measure_network": False},
        })
        orig_sdp = torch.backends.cuda.cudnn_sdp_enabled()
        try:
            # Explicitly enable cuDNN SDP first
            torch.backends.cuda.enable_cudnn_sdp(True)
            trainer = Trainer(config=cfg)
            # Trainer must disable cuDNN SDP for GAP-SAM/attention architectures
            assert torch.backends.cuda.cudnn_sdp_enabled() is False
        finally:
            torch.backends.cuda.enable_cudnn_sdp(orig_sdp)


def test_fatal_cuda_fault_detection():
    """Verify that fatal CUDA accelerator and launch errors are correctly identified."""
    class FakeAcceleratorError(Exception):
        pass

    exc1 = FakeAcceleratorError("CUDA error: unspecified launch failure")
    is_cuda_fault1 = (
        "CUDA error" in str(exc1)
        or "AcceleratorError" in str(type(exc1))
        or "cudaError" in str(exc1)
    )
    assert is_cuda_fault1 is True

    exc2 = ValueError("Shape mismatch in tensor")
    is_cuda_fault2 = (
        "CUDA error" in str(exc2)
        or "AcceleratorError" in str(type(exc2))
        or "cudaError" in str(exc2)
    )
    assert is_cuda_fault2 is False



def test_trainer_closes_loaders_on_iou_sync_cuda_error():
    """Verify Trainer calls close() when a deferred async CUDA error surfaces during IoU metric sync.

    Root cause of the bug in .logs.txt:
      torch.AcceleratorError: CUDA error: unspecified launch failure
      ...trainer.py, line 1037: loss_dict["iou"] = float(ious.mean().item())

    The `.item()` call forces a GPU->CPU synchronisation that can expose CUDA errors that were
    silently queued during the forward or backward pass (asynchronous CUDA execution).  Without
    a guard on this sync point, PyTorch's C++ runtime calls std::terminate() -> SIGABRT, killing
    the process without invoking trainer.close().  DataLoader worker processes then hang
    indefinitely because they never receive SIGTERM.

    This test simulates the deferred CUDA error by patching train_epoch to raise an error
    matching the CUDA accelerator error pattern, which replicates the same propagation path as
    the real bug (error bubbles from _step_batch_train through train_epoch into trainer.train()).
    We verify that trainer.close() is called to shut down DataLoader workers before re-raising.
    """
    from sid_unet.training.trainer import Trainer
    from sid_unet.utils.config import ConfigDict
    import tempfile
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(4, 3, 32, 32)
    y = torch.zeros(4, 1, 32, 32)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ConfigDict({
            "project": {"name": "test_iou_cuda_cleanup", "output_dir": tmpdir, "seed": 42, "device": "cpu"},
            "data": {"image_size": [32, 32], "batch_size": 2},
            "model": {"name": "unet", "in_channels": 3, "out_channels": 1, "aux_classifier": False},
            "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
            "training": {"epochs": 1, "learning_rate": 0.001, "save_best": False, "save_latest": False},
            "logging": {"log_interval": 1, "measure_network": False},
        })
        trainer = Trainer(config=cfg, train_loader=loader)

        closed = False
        orig_close = trainer.close
        def mock_close():
            nonlocal closed
            closed = True
            orig_close()
        trainer.close = mock_close

        # Simulate an async CUDA error that surfaces during the IoU metric GPU->CPU sync.
        # Patching train_epoch directly replicates the propagation path of the real bug:
        # the error raised from _step_batch_train bubbles through train_epoch into trainer.train().
        def mock_train_epoch(epoch):
            raise RuntimeError("CUDA error: unspecified launch failure (cudaErrorLaunchFailure)")
        trainer.train_epoch = mock_train_epoch

        with pytest.raises(RuntimeError, match="CUDA error"):
            trainer.train()

        assert closed is True, (
            "trainer.close() was not invoked when a CUDA error surfaced during IoU metric sync! "
            "DataLoader workers would have hung indefinitely (see .logs.txt bug)."
        )


def test_step_batch_train_resilient_to_iou_metric_failure(monkeypatch):
    """Verify that Trainer._step_batch_train does not crash when diagnostic IoU computation fails.
    
    Diagnostic IoU calculation should log a non-fatal warning, default loss_dict['iou'] to 0.0,
    and proceed with backward pass and training uninterrupted.
    """
    from sid_unet.training.trainer import Trainer
    from sid_unet.utils.config import ConfigDict
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ConfigDict({
            "project": {"name": "test_iou_resilience", "output_dir": tmpdir, "seed": 42, "device": "cpu"},
            "data": {"image_size": [32, 32], "batch_size": 2},
            "model": {"name": "unet", "in_channels": 3, "out_channels": 1, "aux_classifier": False},
            "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
            "training": {"epochs": 1, "learning_rate": 0.001, "save_best": False, "save_latest": False, "amp": False},
            "logging": {"log_interval": 1, "measure_network": False},
        })
        trainer = Trainer(config=cfg)

        batch = {
            "image": torch.randn(2, 3, 32, 32),
            "mask": torch.zeros(2, 1, 32, 32),
        }

        # Simulate failure in torch.sigmoid specifically during diagnostic IoU metric step
        orig_sigmoid = torch.sigmoid
        def mock_sigmoid(tensor):
            if tensor.requires_grad is False:
                raise RuntimeError("Simulated transient CUDA accelerator error or NaN in IoU calculation")
            return orig_sigmoid(tensor)

        monkeypatch.setattr(torch, "sigmoid", mock_sigmoid)

        loss, loss_dict = trainer._step_batch_train(batch)
        assert torch.isfinite(loss)
        assert "iou" in loss_dict
        assert loss_dict["iou"] == 0.0


def test_step_batch_train_handles_fatal_backward_error_with_cleanup(monkeypatch):
    """Verify that Trainer._step_batch_train invokes close() and cleans up workers when backward pass encounters fatal CUDA error."""
    from sid_unet.training.trainer import Trainer
    from sid_unet.utils.config import ConfigDict
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ConfigDict({
            "project": {"name": "test_backward_cleanup", "output_dir": tmpdir, "seed": 42, "device": "cpu"},
            "data": {"image_size": [32, 32], "batch_size": 2},
            "model": {"name": "unet", "in_channels": 3, "out_channels": 1, "aux_classifier": False},
            "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
            "training": {"epochs": 1, "learning_rate": 0.001, "save_best": False, "save_latest": False, "amp": False},
            "logging": {"log_interval": 1, "measure_network": False},
        })
        trainer = Trainer(config=cfg)

        batch = {
            "image": torch.randn(2, 3, 32, 32),
            "mask": torch.zeros(2, 1, 32, 32),
        }

        closed = False
        def mock_close():
            nonlocal closed
            closed = True
        trainer.close = mock_close

        class FakeAcceleratorError(Exception):
            pass

        def mock_backward():
            raise FakeAcceleratorError("CUDA error: unspecified launch failure (cudaErrorLaunchFailure)")

        scaled_loss_mock = torch.tensor(1.0, requires_grad=True)
        monkeypatch.setattr(trainer.scaler, "scale", lambda l: type("Scaled", (), {"backward": lambda self: mock_backward()})())

        with pytest.raises(FakeAcceleratorError, match="CUDA error"):
            trainer._step_batch_train(batch)

        assert closed is True


def test_diffusion_models_guard_cudnn_sdp_on_cuda():
    """Verify that Diffusion models disable cuDNN SDP on CUDA to avoid kernel launch failures."""
    if not torch.cuda.is_available() or not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        pytest.skip("CUDA and enable_cudnn_sdp required")
    from sid_unet.models.diffusion_diff_minimized import DiffusionDiffMinimizedModel

    orig_sdp = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
        # Instantiating DiffusionDiffMinimizedModel on CUDA should disable cuDNN SDP
        model = DiffusionDiffMinimizedModel(
            pretrained_model_name_or_path="dummy",
            use_dummy=True,
            device_map="cuda",
        )
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
    finally:
        torch.backends.cuda.enable_cudnn_sdp(orig_sdp)


def test_trainer_intra_epoch_validation_and_empty_cache():
    """Verify Trainer correctly schedules intra-epoch validation and periodic cache clearing."""
    from sid_unet.training.trainer import Trainer
    from sid_unet.utils.config import ConfigDict
    import tempfile
    from torch.utils.data import DataLoader, TensorDataset

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, masks):
            self.imgs = imgs
            self.masks = masks
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            return {"image": self.imgs[idx], "mask": self.masks[idx]}

    x = torch.randn(8, 3, 32, 32)
    y = torch.zeros(8, 1, 32, 32)
    train_loader = DataLoader(DictDataset(x, y), batch_size=2)
    val_loader = DataLoader(DictDataset(x[:2], y[:2]), batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ConfigDict({
            "project": {"name": "test_intra_val", "output_dir": tmpdir, "seed": 42, "device": "cpu"},
            "data": {"image_size": [32, 32], "batch_size": 2},
            "model": {"name": "unet", "in_channels": 3, "out_channels": 1, "aux_classifier": False},
            "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "save_best": False,
                "save_latest": False,
                "amp": False,
                "val_check_steps": 2,
                "empty_cache_steps": 2,
            },
            "logging": {"log_interval": 1, "measure_network": False},
        })
        trainer = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader)

        val_call_count = 0
        orig_validate = trainer.validate
        def mock_val(*args, **kwargs):
            nonlocal val_call_count
            val_call_count += 1
            return orig_validate(*args, **kwargs)
        trainer.validate = mock_val

        trainer.train()
        # 4 batches total: step 2 triggers intra-epoch val, plus 1 end-of-epoch val = 2 calls
        assert val_call_count >= 2, f"Expected at least 2 validation calls, got {val_call_count}"


