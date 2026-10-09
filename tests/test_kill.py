"""Tests for sid-kill background process cleanup and CLI."""

import os
import subprocess
import sys
import time
import pytest

from sid_unet.kill import (
    find_unet_processes,
    get_ancestor_pids,
    is_unet_process,
    kill_all_background_tasks,
    cli_main,
)


def test_ancestor_pids_contains_current():
    ancestors = get_ancestor_pids()
    assert os.getpid() in ancestors
    assert 1 not in ancestors or len(ancestors) >= 1


def test_is_unet_process():
    assert is_unet_process(1234, "sid-train --config default.yaml", "", "") is True
    assert is_unet_process(1234, "python -m sid_unet.train", "", "") is True
    assert is_unet_process(1234, "torchrun -m sid_unet.train", "", "") is True
    assert is_unet_process(1234, "python -u -m sid_unet.cache.cli", "", "") is True
    assert is_unet_process(1234, "python dummy.py", "", "/workspace/ai-detection-unet-gpu") is True
    assert is_unet_process(1234, "bash", "/bin/bash", "/workspace") is False
    assert is_unet_process(1234, "pytest", "/venv/main/bin/pytest", "/workspace") is False


def test_find_unet_processes_excludes_self():
    procs = find_unet_processes(include_children=True)
    pids = [p["pid"] for p in procs]
    assert os.getpid() not in pids


def test_kill_dummy_background_unet_process():
    # Spawn a dummy python process whose command line mimics a sid_unet background task
    dummy_proc = subprocess.Popen(
        [sys.executable, "-c", "import time; # sid_unet dummy background task\ntime.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)
    try:
        assert dummy_proc.poll() is None

        # Verify discovery
        procs = find_unet_processes(include_children=True)
        matching_pids = [p["pid"] for p in procs]
        assert dummy_proc.pid in matching_pids

        # Test dry-run
        dry_run_count = kill_all_background_tasks(dry_run=True, quiet=True)
        assert dry_run_count >= 1
        assert dummy_proc.poll() is None  # Still alive

        # Test actual kill
        killed = kill_all_background_tasks(force=True, quiet=True)
        assert killed >= 1
        dummy_proc.wait(timeout=2.0)
        assert dummy_proc.poll() is not None
    finally:
        if dummy_proc.poll() is None:
            dummy_proc.kill()


def test_cli_main_dry_run():
    ret = cli_main(["--dry-run", "--quiet"])
    assert ret == 0


def test_shield_process_signals():
    import signal
    from sid_unet.utils.signals import shield_process_signals, SSH_DISCONNECT_SIGNALS

    ignored = shield_process_signals()
    assert signal.SIGTERM in ignored
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in ignored

    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    if hasattr(signal, "SIGHUP"):
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN


def test_kill_shielded_background_process_gracefully():
    """Verify that a background task that shields itself from SIGTERM (signal 15)
    is still gracefully terminated by sid-kill without force=True."""
    # Process ignores SIGTERM and SIGHUP, but responds to SIGINT
    code = (
        "import time, signal\n"
        "from sid_unet.utils.signals import shield_process_signals\n"
        "shield_process_signals()\n"
        "# sid_unet dummy background task\n"
        "time.sleep(60)\n"
    )
    dummy_proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)
    try:
        assert dummy_proc.poll() is None

        # Verify it was found
        procs = find_unet_processes(include_children=True)
        assert dummy_proc.pid in [p["pid"] for p in procs]

        # Graceful kill (force=False) must terminate the shielded process promptly via SIGINT
        t0 = time.time()
        killed = kill_all_background_tasks(force=False, timeout=3.0, quiet=True)
        elapsed = time.time() - t0

        assert killed >= 1
        dummy_proc.wait(timeout=2.0)
        assert dummy_proc.poll() is not None
        # Should have terminated quickly without stalling on the timeout
        assert elapsed < 2.5
    finally:
        if dummy_proc.poll() is None:
            dummy_proc.kill()

