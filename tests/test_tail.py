"""
Unit and integration tests for sid-tail (sid_unet.tail).
"""

import os
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import pytest

from sid_unet.process_manager import (
    ProcessRegistry,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_EXCEPTION,
    STATUS_KILLED,
)
from sid_unet.tail import (
    cli_main,
    print_initial_lines,
    resolve_tail_target,
    tail_log_stream,
)


@pytest.fixture
def temp_env():
    """Create a temporary directory for isolated registry and log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_dir = os.path.join(tmpdir, ".sid_processes")
        os.makedirs(reg_dir, exist_ok=True)
        log_file = os.path.join(tmpdir, "test.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("Line 1: init\nLine 2: setup\nLine 3: ready\n")
        yield {
            "tmpdir": tmpdir,
            "reg_dir": reg_dir,
            "log_file": log_file,
        }


def test_resolve_tail_target_fallback(temp_env):
    """When no target or active processes exist, fallback to .logs.txt."""
    log_path, pid, offset = resolve_tail_target(
        target=None,
        pid_override=None,
        registry_dir=temp_env["reg_dir"],
    )
    assert log_path.endswith(".logs.txt")
    assert pid is None
    assert offset == 0


def test_resolve_tail_target_explicit_pid(temp_env):
    """Resolving an explicit PID registered in ledger."""
    reg = ProcessRegistry(registry_dir=temp_env["reg_dir"])
    rec = reg.register_process(
        pid=999999,
        name="test-task",
        cmdline="python -m test",
        log_file=temp_env["log_file"],
        log_offset_start=42,
    )

    log_path, pid, offset = resolve_tail_target(
        pid_override=999999,
        registry_dir=temp_env["reg_dir"],
    )
    assert log_path == temp_env["log_file"]
    assert pid == 999999
    assert offset == 42


def test_resolve_tail_target_positional_pid(temp_env):
    """Positional numeric argument should be recognized as PID."""
    reg = ProcessRegistry(registry_dir=temp_env["reg_dir"])
    reg.register_process(
        pid=888888,
        name="test-proc",
        cmdline="sid-eval",
        log_file=temp_env["log_file"],
        log_offset_start=10,
    )

    log_path, pid, offset = resolve_tail_target(
        target="888888",
        registry_dir=temp_env["reg_dir"],
    )
    assert log_path == temp_env["log_file"]
    assert pid == 888888
    assert offset == 10


def test_resolve_tail_target_file_path(temp_env):
    """Specifying an explicit file path should resolve that file and link to running PID if present."""
    reg = ProcessRegistry(registry_dir=temp_env["reg_dir"])
    current_pid = os.getpid()
    reg.register_process(
        pid=current_pid,
        name="active-task",
        cmdline="sid-train",
        log_file=temp_env["log_file"],
        log_offset_start=15,
    )

    log_path, pid, offset = resolve_tail_target(
        target=temp_env["log_file"],
        registry_dir=temp_env["reg_dir"],
    )
    assert log_path == temp_env["log_file"]
    assert pid == current_pid
    assert offset == 15


def test_print_initial_lines(temp_env, capsys):
    """Verify last N lines printing."""
    log_file = temp_env["log_file"]
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("Line 4: run\nLine 5: finished\n")

    size = print_initial_lines(log_file, lines_count=2, start_offset=0)
    captured = capsys.readouterr()
    assert "Line 4: run\n" in captured.out
    assert "Line 5: finished\n" in captured.out
    assert "Line 1: init\n" not in captured.out
    assert size > 0


def test_tail_log_stream_clean_completion(temp_env, capsys):
    """Tail stream halts automatically when task completes cleanly."""
    log_file = temp_env["log_file"]

    # Start a background child process that writes to log_file and exits with completion marker
    cmd = (
        f"{sys.executable} -c \""
        f"import time; "
        f"f = open('{log_file}', 'a'); "
        f"time.sleep(0.2); "
        f"f.write('Step 100/100 completed\\n'); f.flush(); "
        f"time.sleep(0.2); "
        f"f.write('✅ Training finished completed successfully\\n'); f.flush(); "
        f"f.close()\""
    )
    proc = subprocess.Popen(cmd, shell=True)

    exit_code = tail_log_stream(
        log_path=log_file,
        monitored_pid=proc.pid,
        lines_count=0,
        poll_interval=0.1,
        timeout=10.0,
    )
    proc.wait()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Step 100/100 completed" in captured.out
    assert "completed successfully" in captured.out
    assert "[sid-tail] Task completed successfully" in captured.out


def test_tail_log_stream_exception_termination(temp_env, capsys):
    """Tail stream halts automatically and reports error when unhandled exception occurs."""
    log_file = temp_env["log_file"]

    cmd = (
        f"{sys.executable} -c \""
        f"import time; "
        f"f = open('{log_file}', 'a'); "
        f"time.sleep(0.2); "
        f"f.write('Traceback (most recent call last):\\n  File app.py\\nValueError: Invalid tensor dimension\\n'); f.flush(); "
        f"f.close()\""
    )
    proc = subprocess.Popen(cmd, shell=True)

    exit_code = tail_log_stream(
        log_path=log_file,
        monitored_pid=proc.pid,
        lines_count=0,
        poll_interval=0.1,
        timeout=10.0,
    )
    proc.wait()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "ValueError: Invalid tensor dimension" in captured.out
    assert "[sid-tail] Task terminated with unhandled exception" in captured.out


def test_tail_log_stream_process_death_without_marker(temp_env, capsys):
    """Tail stream halts automatically when monitored PID dies even without explicit marker."""
    log_file = temp_env["log_file"]

    cmd = (
        f"{sys.executable} -c \""
        f"import time; "
        f"f = open('{log_file}', 'a'); "
        f"time.sleep(0.2); "
        f"f.write('Unexpected process exit midway\\n'); f.flush(); "
        f"f.close()\""
    )
    proc = subprocess.Popen(cmd, shell=True)

    exit_code = tail_log_stream(
        log_path=log_file,
        monitored_pid=proc.pid,
        lines_count=0,
        poll_interval=0.1,
        timeout=10.0,
    )
    proc.wait()
    captured = capsys.readouterr()

    assert exit_code == 1  # Not clean completion
    assert "Unexpected process exit midway" in captured.out
    assert "[sid-tail]" in captured.out


def test_tail_log_stream_timeout(temp_env):
    """Tail stream respects timeout."""
    log_file = temp_env["log_file"]
    # Monitoring PID of a long sleeper
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        t0 = time.time()
        exit_code = tail_log_stream(
            log_path=log_file,
            monitored_pid=proc.pid,
            lines_count=0,
            poll_interval=0.1,
            timeout=0.6,
            quiet=True,
        )
        elapsed = time.time() - t0
        assert exit_code == 1
        assert elapsed < 3.0
    finally:
        proc.kill()
        proc.wait()


def test_cli_main_invocation(temp_env):
    """Test calling cli_main programmatically."""
    log_file = temp_env["log_file"]
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("Finished all runs completed successfully\n")

    exit_code = cli_main([log_file, "-n", "5", "--quiet"])
    assert exit_code == 0
