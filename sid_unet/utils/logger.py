"""
Logging utilities for SID-UNet.
Supports resilient console output, rotating file logs, and uncaught exception handling.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import threading
from typing import Any, Dict, Optional, Union
import numpy as np
import torch

DEFAULT_LOG_DIR = "log"
DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20 MB per log file
DEFAULT_BACKUP_COUNT = 5
DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class SafeStreamHandler(logging.StreamHandler):
    """
    Stream handler resilient to terminal disconnection (SSH logout, closed pts, broken pipes).
    Suppresses BrokenPipeError, IOError, and OSError to ensure that background or headless
    processes do not crash when their controlling terminal or standard stream detaches.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
            self.flush()
        except (BrokenPipeError, IOError, OSError, ValueError):
            pass

    def flush(self) -> None:
        try:
            super().flush()
        except (BrokenPipeError, IOError, OSError, ValueError):
            pass


class MetricLogger:
    """Tracks running averages and current values for training/validation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.meters: Dict[str, Dict[str, float]] = {}

    def update(self, name: str, val: float, n: int = 1):
        if name not in self.meters:
            self.meters[name] = {"sum": 0.0, "count": 0, "val": 0.0}
        self.meters[name]["val"] = val
        self.meters[name]["sum"] += val * n
        self.meters[name]["count"] += n

    def update_dict(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number, torch.Tensor)):
                val = float(v.item() if isinstance(v, torch.Tensor) else v)
                self.update(k, val, n)

    def avg(self, name: str) -> float:
        if name not in self.meters or self.meters[name]["count"] == 0:
            return 0.0
        return self.meters[name]["sum"] / self.meters[name]["count"]

    def val(self, name: str) -> float:
        if name not in self.meters:
            return 0.0
        return self.meters[name]["val"]

    def averages(self) -> Dict[str, float]:
        return {k: self.avg(k) for k in self.meters}


def install_exception_logger(logger: Optional[logging.Logger] = None) -> None:
    """
    Install a global exception hook (sys.excepthook and threading.excepthook)
    that logs any uncaught fatal exceptions to the logger with full traceback and timestamp.
    """
    target_logger = logger or logging.getLogger("sid_unet")

    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow clean keyboard interrupt handling without stack trace spam
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        target_logger.critical(
            "💥 Uncaught fatal exception encountered:",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        flush_loggers(target_logger)

    sys.excepthook = handle_uncaught_exception

    if hasattr(threading, "excepthook"):
        def handle_thread_exception(args):
            if issubclass(args.exc_type, KeyboardInterrupt):
                return
            target_logger.critical(
                f"💥 Uncaught fatal thread exception in thread '{args.thread.name}':",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            flush_loggers(target_logger)

        threading.excepthook = handle_thread_exception


def flush_loggers(logger: Optional[logging.Logger] = None) -> None:
    """Flush all active stream and file handlers on the logger and package logger."""
    loggers_to_flush = [logger] if logger else []
    for name in ("sid_unet", ""):
        lg = logging.getLogger(name)
        if lg not in loggers_to_flush:
            loggers_to_flush.append(lg)

    for lg in loggers_to_flush:
        if lg:
            for h in lg.handlers:
                try:
                    h.flush()
                except Exception:
                    pass


def setup_logger(
    name: str = "SID_UNet",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    enable_console: bool = True,
    enable_file: bool = True,
    install_excepthook: bool = True,
) -> logging.Logger:
    """
    Set up and configure a resilient console and rotating file logger.

    Args:
        name: Name of the logger.
        log_file: Optional path to the log file. If omitted and enable_file=True,
            defaults to {log_dir or DEFAULT_LOG_DIR}/{sanitized_name}.log.
        log_dir: Optional base directory for log files if log_file is not fully specified.
        level: Logging level (e.g. logging.INFO).
        max_bytes: Maximum size per log file before rotation (default: 20MB).
        backup_count: Number of rotated backup files to retain (default: 5).
        enable_console: Whether to attach console StreamHandler.
        enable_file: Whether to attach rotating file handler.
        install_excepthook: Whether to register the global uncaught exception hook.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicate messages on re-configuration
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    console_handler = None
    if enable_console:
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_handler = None
    if enable_file:
        resolved_file = log_file
        if not resolved_file:
            base_dir = log_dir or DEFAULT_LOG_DIR
            sanitized_name = name.lower().replace(" ", "_").replace("-", "_")
            resolved_file = os.path.join(base_dir, f"{sanitized_name}.log")

        resolved_dir = os.path.dirname(os.path.abspath(resolved_file))
        os.makedirs(resolved_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            resolved_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Attach handlers to the package logger ("sid_unet") to ensure submodule logs are captured
    if name != "sid_unet":
        pkg_logger = logging.getLogger("sid_unet")
        pkg_logger.setLevel(level)
        for h in logger.handlers:
            if h not in pkg_logger.handlers:
                pkg_logger.addHandler(h)

    if install_excepthook:
        install_exception_logger(logger)

    return logger


def get_logger(name: str = "sid_unet") -> logging.Logger:
    """Retrieve or initialize the logger for the specified component."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
