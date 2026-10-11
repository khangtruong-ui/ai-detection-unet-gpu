"""
Tests for logging utilities in sid_unet.utils.logger.
Verifies RotatingFileHandler, SafeStreamHandler, default log directories,
uncaught exception logging hooks, MetricLogger, and gitignore patterns.
"""

import logging
import os
import subprocess
import sys
from unittest.mock import MagicMock
import pytest

from sid_unet.utils.logger import (
    DEFAULT_LOG_DIR,
    MetricLogger,
    SafeStreamHandler,
    flush_loggers,
    get_logger,
    install_exception_logger,
    setup_logger,
)
from logging.handlers import RotatingFileHandler


def test_setup_logger_with_rotating_file_handler(tmp_path):
    """Verify setup_logger configures a RotatingFileHandler and SafeStreamHandler."""
    log_file = tmp_path / "sub" / "test_app.log"
    logger = setup_logger(
        name="test_app",
        log_file=str(log_file),
        level=logging.DEBUG,
        max_bytes=1024 * 1024,
        backup_count=3,
        install_excepthook=False,
    )

    handlers = logger.handlers
    assert any(isinstance(h, RotatingFileHandler) for h in handlers)
    assert any(isinstance(h, SafeStreamHandler) for h in handlers)

    rf_handler = [h for h in handlers if isinstance(h, RotatingFileHandler)][0]
    assert rf_handler.maxBytes == 1024 * 1024
    assert rf_handler.backupCount == 3

    logger.info("Test log line 1")
    logger.debug("Test debug line")
    flush_loggers(logger)

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log line 1" in content
    assert "Test debug line" in content


def test_rotating_file_handler_rotation(tmp_path):
    """Verify log files rotate when exceeding max_bytes."""
    log_file = tmp_path / "rotating.log"
    logger = setup_logger(
        name="test_rotator",
        log_file=str(log_file),
        max_bytes=200,
        backup_count=2,
        install_excepthook=False,
    )

    for i in range(25):
        logger.info(f"Log message iteration {i:03d} to exceed byte limit")
    flush_loggers(logger)

    assert log_file.exists()
    rotated_file_1 = tmp_path / "rotating.log.1"
    assert rotated_file_1.exists(), "Expected rotated backup file .1 to exist"


def test_default_log_dir_creation(tmp_path, monkeypatch):
    """Verify setup_logger defaults to creating a log/ folder when log_file is omitted."""
    monkeypatch.chdir(tmp_path)
    logger = setup_logger(
        name="my_service",
        log_file=None,
        install_excepthook=False,
    )
    logger.info("Service started successfully")
    flush_loggers(logger)

    expected_log = tmp_path / DEFAULT_LOG_DIR / "my_service.log"
    assert expected_log.exists()
    content = expected_log.read_text(encoding="utf-8")
    assert "Service started successfully" in content


def test_safe_stream_handler_broken_pipe_suppression():
    """Verify SafeStreamHandler silently suppresses BrokenPipeError and IOError."""
    mock_stream = MagicMock()
    mock_stream.write.side_effect = BrokenPipeError("Broken pipe simulated on closed PTY")
    mock_stream.flush.side_effect = OSError("Simulated EIO")

    handler = SafeStreamHandler(mock_stream)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Terminal disconnect simulation",
        args=(),
        exc_info=None,
    )

    # Neither emit nor flush should raise BrokenPipeError / OSError
    handler.emit(record)
    handler.flush()


def test_install_exception_logger_intercepts_unhandled_exception(tmp_path):
    """Verify install_exception_logger logs uncaught exceptions to the rotating log file."""
    log_file = tmp_path / "crash.log"
    logger = setup_logger(
        name="crash_logger",
        log_file=str(log_file),
        install_excepthook=False,
    )

    original_excepthook = sys.excepthook
    try:
        install_exception_logger(logger)
        assert sys.excepthook != original_excepthook

        # Simulate uncaught runtime exception
        try:
            raise RuntimeError("Simulated unhandled training crash")
        except RuntimeError:
            exc_type, exc_val, exc_tb = sys.exc_info()
            sys.excepthook(exc_type, exc_val, exc_tb)

        flush_loggers(logger)
        assert log_file.exists()
        log_content = log_file.read_text(encoding="utf-8")
        assert "💥 Uncaught fatal exception encountered:" in log_content
        assert "Simulated unhandled training crash" in log_content
        assert "RuntimeError" in log_content
    finally:
        sys.excepthook = original_excepthook


def test_install_exception_logger_keyboard_interrupt_passthrough():
    """Verify KeyboardInterrupt is passed to original excepthook without logging crash."""
    original_excepthook = sys.excepthook
    mock_orig = MagicMock()
    sys.__excepthook__ = mock_orig

    mock_logger = MagicMock()
    try:
        install_exception_logger(mock_logger)
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        assert not mock_logger.critical.called
        assert mock_orig.called
    finally:
        sys.excepthook = original_excepthook


def test_get_logger_and_flush_loggers(tmp_path):
    """Verify get_logger retrieves existing or sets up new logger, and flush_loggers flushes."""
    logger = get_logger("sid_custom_component")
    assert isinstance(logger, logging.Logger)
    logger.info("Custom log message")
    flush_loggers(logger)
    flush_loggers()


def test_metric_logger():
    """Verify MetricLogger tracks values, running sums, counts, and averages."""
    ml = MetricLogger()
    ml.update("loss", 2.0, n=2)
    ml.update("loss", 4.0, n=2)
    assert ml.val("loss") == 4.0
    assert ml.avg("loss") == 3.0

    ml.update_dict({"acc": 0.8, "iou": 0.6}, n=1)
    avgs = ml.averages()
    assert avgs["loss"] == 3.0
    assert avgs["acc"] == 0.8
    assert avgs["iou"] == 0.6

    ml.reset()
    assert ml.avg("loss") == 0.0
    assert ml.val("loss") == 0.0


def test_gitignore_ignores_logs():
    """Verify that .gitignore properly ignores log/, logs/, *.log, *.logs.txt, and .logs.txt."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        "log/",
        "logs/",
        "train.log",
        "log/run.log",
        "logs/run.log",
        ".logs.txt",
        "experiment.logs.txt",
    ]

    for item in candidates:
        res = subprocess.run(
            ["git", "check-ignore", item],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Expected '{item}' to be ignored by git in .gitignore"
