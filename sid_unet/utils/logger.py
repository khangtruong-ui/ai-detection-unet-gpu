"""
Logging utilities for SID-UNet.
Supports console output and rotating file logs.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional, Union
import numpy as np
import torch


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

