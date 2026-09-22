"""
Logging utilities for SID-UNet.
Supports console output, rotating file logs, and smart progress logging
that automatically optimizes output for both local interactive terminals
and headless cloud logging environments (such as Modal) without TQDM newline spam.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def is_modal_environment() -> bool:
    """
    Detect if code is executing inside a Modal container or headless/non-interactive cloud environment.
    """
    if os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_LOG_FORMAT"):
        return True
    try:
        import modal
        if hasattr(modal, "is_local") and not modal.is_local():
            return True
    except Exception:
        pass
    mode = os.environ.get("SID_PROGRESS_MODE", "").lower()
    if mode == "clean":
        return True
    if mode == "tqdm":
        return False
    # Non-TTY output (redirected to file or piped stream)
    return not (sys.stdout.isatty() or sys.stderr.isatty())


def _format_time(seconds: float) -> str:
    """Format duration in seconds as mm:ss or hh:mm:ss."""
    mins, s = divmod(int(seconds), 60)
    h, m = divmod(mins, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class SmartProgressBar:
    """
    Progress bar optimized for both local interactive terminals and cloud logging (Modal).

    In Modal / non-TTY cloud environments:
      Eliminates carriage-return spam and outputs clean, periodic, single-line log messages
      at fixed step intervals or time intervals, showing percentage, elapsed time, ETA,
      iterations/second, and dynamic metric postfixes.

    In local interactive terminal:
      Wraps standard tqdm for smooth, real-time animation.
    """

    def __init__(
        self,
        iterable: Optional[Any] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        leave: bool = True,
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10,
        min_interval: float = 3.0,
        mode: str = "auto",
        **kwargs,
    ):
        self.iterable = iterable
        self.desc = desc or "Progress"
        if total is not None and total > 0:
            self.total = total
        elif iterable is not None:
            try:
                cand = len(iterable)
                self.total = cand if cand > 0 else None
            except (TypeError, NotImplementedError, AttributeError):
                self.total = None
        else:
            self.total = None
        self.leave = leave
        self.logger = logger or logging.getLogger("SID_UNet")
        self.log_interval = max(1, log_interval)
        self.min_interval = min_interval
        self.step = 0
        self.start_time: Optional[float] = None
        self.last_log_time: float = 0.0
        self.last_logged_step: int = -1
        self.postfix_data: Dict[str, Any] = {}

        if mode == "auto":
            self.use_clean_logging = is_modal_environment()
        elif mode == "clean":
            self.use_clean_logging = True
        else:
            self.use_clean_logging = False

        self._tqdm_instance = None
        if not self.use_clean_logging:
            try:
                from tqdm import tqdm
                self._tqdm_instance = tqdm(
                    iterable=iterable,
                    desc=desc,
                    total=self.total,
                    leave=leave,
                    **kwargs,
                )
            except ImportError:
                self.use_clean_logging = True

    def set_postfix(self, ordered_dict: Optional[Dict[str, Any]] = None, refresh: bool = True, **kwargs):
        """Update postfix metrics display."""
        if ordered_dict:
            self.postfix_data.update(ordered_dict)
        if kwargs:
            self.postfix_data.update(kwargs)

        if self._tqdm_instance is not None:
            self._tqdm_instance.set_postfix(ordered_dict, refresh=refresh, **kwargs)

    def set_description(self, desc: Optional[str] = None, refresh: bool = True):
        """Update description string."""
        if desc:
            self.desc = desc
        if self._tqdm_instance is not None:
            self._tqdm_instance.set_description(desc, refresh=refresh)

    def update(self, n: int = 1):
        """Manually advance progress by n steps."""
        self.step += n
        if self._tqdm_instance is not None:
            self._tqdm_instance.update(n)
        elif self.use_clean_logging:
            self._maybe_log_progress()

    def __iter__(self):
        if self._tqdm_instance is not None:
            yield from self._tqdm_instance
            return

        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time

        if self.iterable is not None:
            for item in self.iterable:
                yield item
                self.step += 1
                self._maybe_log_progress()

            self.close()

    def _maybe_log_progress(self, force: bool = False):
        if self.start_time is None:
            self.start_time = time.perf_counter()
            self.last_log_time = self.start_time

        now = time.perf_counter()
        is_first = (self.step == 1)
        is_last = (self.total is not None and self.step >= self.total)
        is_interval = (self.step % self.log_interval == 0)
        is_time_interval = (now - self.last_log_time >= self.min_interval)

        if force or is_first or is_last or is_interval or is_time_interval:
            if self.step != self.last_logged_step:
                self._log_progress(now)

    def _log_progress(self, now: float):
        elapsed = max(0.001, now - self.start_time)
        rate = self.step / elapsed if elapsed > 0 else 0.0

        metrics_part = ""
        if self.postfix_data:
            metrics_part = " | " + ", ".join(f"{k}: {v}" for k, v in self.postfix_data.items())

        if self.total and self.total > 0:
            pct = (self.step / self.total) * 100.0
            eta = (self.total - self.step) / rate if rate > 0 else 0.0
            msg = (
                f"{self.desc} [Step {self.step}/{self.total} ({pct:.1f}%)]"
                f"{metrics_part} "
                f"[{_format_time(elapsed)}<{_format_time(eta)}, {rate:.2f} it/s]"
            )
        else:
            msg = (
                f"{self.desc} [Step {self.step}]"
                f"{metrics_part} "
                f"[{_format_time(elapsed)}, {rate:.2f} it/s]"
            )

        self.logger.info(msg)
        self.last_log_time = now
        self.last_logged_step = self.step

    def close(self):
        """Close progress bar and output summary if needed."""
        if self._tqdm_instance is not None:
            self._tqdm_instance.close()
            self._tqdm_instance = None
        elif self.use_clean_logging and self.start_time is not None:
            if self.step != self.last_logged_step and self.step > 0:
                self._log_progress(time.perf_counter())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_progress_bar(
    iterable: Optional[Any] = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    leave: bool = True,
    logger: Optional[logging.Logger] = None,
    log_interval: int = 10,
    min_interval: float = 3.0,
    mode: str = "auto",
    **kwargs,
) -> SmartProgressBar:
    """
    Factory function returning a SmartProgressBar instance.
    Drop-in replacement for tqdm(...).
    """
    return SmartProgressBar(
        iterable=iterable,
        desc=desc,
        total=total,
        leave=leave,
        logger=logger,
        log_interval=log_interval,
        min_interval=min_interval,
        mode=mode,
        **kwargs,
    )


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
            if torch is not None and isinstance(v, torch.Tensor):
                val = float(v.item())
            elif np is not None and isinstance(v, np.number):
                val = float(v)
            elif isinstance(v, (int, float)):
                val = float(v)
            else:
                try:
                    val = float(v)
                except (ValueError, TypeError):
                    continue
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


def setup_logger(
    name: str = "SID_UNet",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up and configure console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
