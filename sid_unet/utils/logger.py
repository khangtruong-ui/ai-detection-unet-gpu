"""
Logging utilities for SID-UNet.
Supports console output, rotating file logs, and TensorBoard metric/image tracking.
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


class TensorboardLogger:
    """TensorBoard summary writer wrapper with helper methods for scalars and segmentation masks."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.writer = None
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("Warning: torch.utils.tensorboard not available. Tensorboard logging disabled.")
                self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int):
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image_comparison(
        self,
        tag: str,
        images: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        step: int,
        max_samples: int = 4,
    ):
        """
        Log side-by-side comparison of Input Image | Ground Truth Mask | Predicted Mask.
        images: (B, 3, H, W)
        targets: (B, 1, H, W) or (B, H, W)
        predictions: (B, 1, H, W) or (B, H, W)
        """
        if not self.enabled or self.writer is None:
            return

        import torchvision.utils as vutils

        b = min(images.size(0), max_samples)
        imgs = images[:b].detach().cpu()
        # Unnormalize images if standard ImageNet norm was applied, else clamp to [0, 1]
        imgs = torch.clamp(imgs, 0.0, 1.0)

        targs = targets[:b].detach().cpu()
        if targs.dim() == 3:
            targs = targs.unsqueeze(1)
        targs = targs.repeat(1, 3, 1, 1) if targs.size(1) == 1 else targs

        preds = predictions[:b].detach().cpu()
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        preds = preds.repeat(1, 3, 1, 1) if preds.size(1) == 1 else preds

        # Stack triplets: for each sample, [image, target, prediction]
        triplets = []
        for i in range(b):
            triplets.extend([imgs[i], targs[i], preds[i]])

        grid = vutils.make_grid(torch.stack(triplets), nrow=3, padding=4, normalize=False)
        self.writer.add_image(tag, grid, step)

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()
