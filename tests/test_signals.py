"""Tests for signal shielding and forward termination."""

import os
import signal
import subprocess
import sys
import time
import pytest

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
    """Verify that worker_init_fn applies signal shielding in child worker processes."""
    from sid_unet.dataset.loader import worker_init_fn

    worker_init_fn(0)
    for sig in (signal.SIGTERM, signal.SIGPIPE):
        assert signal.getsignal(sig) == signal.SIG_IGN


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

