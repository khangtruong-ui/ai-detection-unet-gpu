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
