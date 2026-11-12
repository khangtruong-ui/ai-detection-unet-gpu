"""
Tests for ProcessManager, ProcessRegistry, and SIDTerminalUI.
Verifies tracking of running, completed, killed, and exception-terminated processes,
live log inspection, exception extraction, and CLI/TUI operations.
"""

import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from sid_unet.process_manager import (
    STATUS_COMPLETED,
    STATUS_EXCEPTION,
    STATUS_KILLED,
    STATUS_RUNNING,
    STATUS_UNKNOWN,
    ProcessRecord,
    ProcessRegistry,
    inspect_log_for_termination,
    terminate_process,
)
from sid_unet.tui import SIDTerminalUI, cli_main, print_process_table, read_process_logs


def test_process_record_lifecycle():
    """Verify ProcessRecord dataclass properties, serialization, and duration calculations."""
    rec = ProcessRecord(
        pid=1234,
        name="sid-train",
        cmdline="python -m sid_unet.train --config configs/base.yaml",
        cwd="/workspace",
        log_file="/workspace/.logs.txt",
        start_time="2026-11-12T08:00:00",
        status=STATUS_RUNNING,
    )
    assert rec.is_running is True
    assert rec.duration_seconds >= 0.0
    assert "s" in rec.duration_str or "m" in rec.duration_str

    d = rec.to_dict()
    assert d["pid"] == 1234
    assert d["name"] == "sid-train"

    rec2 = ProcessRecord.from_dict(d)
    assert rec2.pid == rec.pid
    assert rec2.name == rec.name


def test_inspect_log_for_clean_completion(tmp_path):
    """Verify detection of cleanly finished tasks."""
    log_file = tmp_path / "complete.log"
    log_file.write_text(
        "Starting training epoch 10/10\n"
        "Step 1000: loss=0.123, iou=0.85\n"
        "✅ Training completed successfully! Checkpoint saved.\n"
    )

    status, reason, err = inspect_log_for_termination(str(log_file))
    assert status == STATUS_COMPLETED
    assert "completed successfully" in reason.lower()
    assert err is None


def test_inspect_log_for_unhandled_exception(tmp_path):
    """Verify extraction and classification of Python exceptions and stack traces."""
    log_file = tmp_path / "error.log"
    log_file.write_text(
        "Loading dataset shard 4...\n"
        "Traceback (most recent call last):\n"
        '  File "sid_unet/train.py", line 450, in train_epoch\n'
        '    loss = criterion(outputs, targets)\n'
        '  File "sid_unet/losses/combined.py", line 82, in forward\n'
        "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB\n"
    )

    status, reason, err = inspect_log_for_termination(str(log_file))
    assert status == STATUS_EXCEPTION
    assert "CUDA out of memory" in reason
    assert "RuntimeError: CUDA out of memory" in err


def test_inspect_log_for_keyboard_interrupt(tmp_path):
    """Verify extraction of KeyboardInterrupt as killed/cancelled status."""
    log_file = tmp_path / "interrupt.log"
    log_file.write_text(
        "Evaluating batch 42...\n"
        "Traceback (most recent call last):\n"
        '  File "sid_unet/cross_eval.py", line 720, in evaluate\n'
        "    time.sleep(1)\n"
        "KeyboardInterrupt\n"
    )

    status, reason, err = inspect_log_for_termination(str(log_file))
    assert status == STATUS_KILLED
    assert "KeyboardInterrupt" in reason or "interrupted" in reason.lower()


def test_inspect_log_abrupt_termination(tmp_path):
    """Verify classification of abrupt terminations with no exit marker as KILLED."""
    log_file = tmp_path / "killed.log"
    log_file.write_text(
        "Starting worker rank 0\n"
        "Step 45: loss=0.552\n"
        "Killed\n"
    )

    status, reason, err = inspect_log_for_termination(str(log_file))
    assert status == STATUS_KILLED
    assert "Killed" in reason or "Killed" in str(err)


def test_process_registry_crud(tmp_path):
    """Verify register, update, retrieve, list, and clear in ProcessRegistry."""
    reg = ProcessRegistry(str(tmp_path))

    # 1. Register running process
    r1 = reg.register_process(
        pid=9001,
        name="sid-train",
        cmdline="sid-train --epochs 5",
        log_file=str(tmp_path / "p1.log"),
        ppid=100,
    )
    assert r1.pid == 9001
    assert r1.status == STATUS_RUNNING

    # 2. Retrieve
    retrieved = reg.get_process(9001)
    assert retrieved is not None
    assert retrieved.name == "sid-train"

    # 3. Update process
    reg.update_process(9001, rss_mb=512.5)
    assert reg.get_process(9001).rss_mb == 512.5

    # 4. Mark terminated
    reg.mark_process_terminated(
        9001,
        status=STATUS_COMPLETED,
        reason="Finished 5 epochs",
        exit_code=0,
    )
    r1_term = reg.get_process(9001)
    assert r1_term.status == STATUS_COMPLETED
    assert r1_term.exit_code == 0
    assert r1_term.termination_reason == "Finished 5 epochs"

    # 5. Register second process with exception
    reg.register_process(
        pid=9002,
        name="sid-cross-eval",
        cmdline="sid-cross-eval",
        log_file=str(tmp_path / "p2.log"),
    )
    reg.mark_process_terminated(
        9002,
        status=STATUS_EXCEPTION,
        reason="Exception: ValueError: Invalid config",
        error_summary="ValueError: Invalid config",
    )

    # 6. List and filter
    all_procs = reg.list_processes()
    assert len(all_procs) == 2

    exc_procs = reg.list_processes(status_filter="EXCEPTION")
    assert len(exc_procs) == 1
    assert exc_procs[0].pid == 9002

    comp_procs = reg.list_processes(status_filter="COMPLETED")
    assert len(comp_procs) == 1
    assert comp_procs[0].pid == 9001

    # 7. Clear history
    removed = reg.clear_history(keep_running=True)
    assert removed == 2
    assert len(reg.list_processes()) == 0


def test_process_registry_reconcile_with_dead_process(tmp_path):
    """Verify that reconcile() marks inactive PIDs and inspects their logs."""
    reg = ProcessRegistry(str(tmp_path))

    log_file = tmp_path / "crashed_run.log"
    log_file.write_text(
        "Traceback (most recent call last):\n"
        '  File "sid_unet/train.py", line 12, in <module>\n'
        "KeyError: 'loss'\n"
    )

    # Register an already dead PID (e.g. 9999999)
    reg.register_process(
        pid=9999999,
        name="sid-train",
        cmdline="sid-train",
        log_file=str(log_file),
    )

    # Before reconcile: RUNNING
    assert reg.get_process(9999999).status == STATUS_RUNNING

    # Reconcile detects the process is dead and checks the log
    updated = reg.reconcile()
    p = reg.get_process(9999999)
    assert p.status == STATUS_EXCEPTION
    assert "KeyError: 'loss'" in p.error_summary


def test_terminate_process_flow(tmp_path):
    """Verify graceful termination and ledger status update."""
    reg = ProcessRegistry(str(tmp_path))

    # Spawn an actual mock background process
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid = proc.pid

    reg.register_process(
        pid=pid,
        name="sid-sleep",
        cmdline="sleep test",
        log_file=str(tmp_path / "sleep.log"),
    )

    try:
        terminated = terminate_process(pid, force=False, timeout=1.0, registry=reg)
        assert terminated is True
        rec = reg.get_process(pid)
        assert rec.status == STATUS_KILLED
    finally:
        try:
            proc.kill()
            proc.wait(timeout=0.5)
        except Exception:
            pass


def test_read_process_logs(tmp_path):
    """Verify log reader respecting start offset."""
    log_file = tmp_path / "multiproc.log"
    # First process section
    log_file.write_text("Old process line 1\nOld process line 2\n")
    offset = log_file.stat().st_size

    # Second process appended
    with open(log_file, "a") as f:
        f.write("New process line 1\nNew process line 2\n")

    rec = ProcessRecord(
        pid=123,
        name="sid-test",
        cmdline="sid-test",
        cwd=str(tmp_path),
        log_file=str(log_file),
        start_time="2026-11-12T00:00:00",
        log_offset_start=offset,
    )

    lines = read_process_logs(rec)
    assert len(lines) == 2
    assert lines[0] == "New process line 1"
    assert lines[1] == "New process line 2"


def test_tui_keyboard_navigation_and_actions(tmp_path):
    """Unit test SIDTerminalUI logic without requiring an interactive tty."""
    reg = ProcessRegistry(str(tmp_path))
    reg.register_process(
        pid=1001,
        name="sid-train",
        cmdline="sid-train",
        log_file=str(tmp_path / "t1.log"),
    )
    reg.register_process(
        pid=1002,
        name="sid-cross-eval",
        cmdline="sid-cross-eval",
        log_file=str(tmp_path / "t2.log"),
    )

    ui = SIDTerminalUI(registry=reg)
    with patch("sid_unet.kill._is_pid_alive", return_value=True):
        ui.refresh_process_data(force=True)
        assert len(ui.processes) == 2

        # Test filtering
        ui.filter_mode = "RUNNING"
        assert len(ui.get_filtered_processes()) == 2
        ui.filter_mode = "EXCEPTION"
        assert len(ui.get_filtered_processes()) == 0

        # Test tab cycle
        ui._handle_key(ord("\t"))
        assert ui.filter_mode in ("ALL", "RUNNING", "EXCEPTION", "TERMINATED")

        # Test view mode toggle
        assert ui.view_mode == "SPLIT"
        ui._handle_key(10)  # Enter
        assert ui.view_mode == "FULL_LOG"
        ui._handle_key(10)  # Enter back
        assert ui.view_mode == "SPLIT"

        # Test auto follow toggle
        ui._handle_key(ord("f"))
        assert ui.log_auto_follow is False
        ui._handle_key(ord("f"))
        assert ui.log_auto_follow is True

        # Test quit
        ui._handle_key(ord("q"))
        assert ui.should_exit is True


def test_cli_main_commands(tmp_path, capsys):
    """Verify headless CLI options: --list, --status, --clear-history."""
    reg = ProcessRegistry(str(tmp_path))
    reg.register_process(
        pid=5555,
        name="sid-train",
        cmdline="sid-train --config test",
        log_file=str(tmp_path / "run.log"),
    )
    reg.mark_process_terminated(
        5555,
        status=STATUS_EXCEPTION,
        reason="RuntimeError: CUDA out of memory",
        error_summary="RuntimeError: CUDA out of memory",
    )

    with patch("sid_unet.tui.get_process_registry", return_value=reg):
        # 1. Test --list
        code = cli_main(["--list"])
        assert code == 0
        captured = capsys.readouterr()
        assert "5555" in captured.out
        assert "EXCEPTION" in captured.out

        # 2. Test --status
        code = cli_main(["--status"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Exception Terminations: 1" in captured.out

        # 3. Test --clear-history
        code = cli_main(["--clear-history"])
        assert code == 0
        assert len(reg.list_processes()) == 0
