"""
Callbacks for training: checkpoint management and early stopping.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from sid_unet.utils.config import save_config


class CheckpointManager:
    """Manages saving and loading model checkpoints (best, latest, and periodic)."""

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = "val_iou",
        mode: str = "max",
        save_best: bool = True,
        save_latest: bool = False,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.mode = mode
        self.save_best = save_best
        self.save_latest = save_latest

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1

    def is_better(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False,
    ) -> Dict[str, str]:
        """Save checkpoints to disk."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "config": config,
        }

        saved_paths = {}

        if self.save_latest:
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
            torch.save(state, latest_path)
            latest_cfg_path = os.path.join(self.checkpoint_dir, "checkpoint_latest_config.yaml")
            save_config(config, latest_cfg_path)
            saved_paths["latest"] = latest_path

        current_score = metrics.get(self.metric_name, None)
        if current_score is not None and self.is_better(current_score):
            self.best_score = current_score
            self.best_epoch = epoch
            if self.save_best:
                best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
                torch.save(state, best_path)
                best_cfg_path = os.path.join(self.checkpoint_dir, "checkpoint_best_config.yaml")
                save_config(config, best_cfg_path)
                saved_paths["best"] = best_path

        return saved_paths

    def load_latest(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None) -> Optional[int]:
        """Load latest checkpoint if available. Returns resumed epoch."""
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        if not os.path.exists(latest_path):
            return None
        return self.load_checkpoint(latest_path, model, optimizer, scheduler)

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None) -> int:
        """Load specific checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", 0)


class EarlyStopping:
    """Early stops training when monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.patience <= 0:
            return False

        if self.mode == "max":
            improved = (current_score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - current_score) > self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
