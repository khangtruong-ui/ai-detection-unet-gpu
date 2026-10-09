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
