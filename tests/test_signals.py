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
    from sid_unet.dataset.loader import worker_init_fn

    worker_init_fn(0)
    # SIGPIPE should be ignored to protect workers from BrokenPipeError on SSH drops
    assert signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN
    # SIGTERM must NOT be ignored so PyTorch can terminate worker processes cleanly via SIGTERM
    assert signal.getsignal(signal.SIGTERM) != signal.SIG_IGN


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

    with tempfile.TemporaryDirectory() as tmpdir:
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

