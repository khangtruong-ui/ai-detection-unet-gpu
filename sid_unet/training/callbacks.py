"""
Callbacks for training: checkpoint management and early stopping.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

logger = logging.getLogger("sid_unet.training.callbacks")

from sid_unet.utils.config import save_config
from sid_unet.utils.checkpoint import (
    find_auto_resume_checkpoint,
    download_hf_checkpoint,
    is_hf_repo_id,
    inspect_checkpoint,
    format_resume_notification,
    format_no_resume_notification,
    parse_hf_repo_uri,
)


class CheckpointManager:
    """Manages saving and loading model checkpoints (best, latest, and periodic)."""

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = "val_iou",
        mode: str = "max",
        save_best: bool = True,
        save_latest: bool = False,
        checkpoint_period: Optional[float] = 3600.0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.mode = mode
        self.save_best = save_best
        self.save_latest = save_latest
        self.checkpoint_period = checkpoint_period
        self.last_periodic_save_time = time.time()

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.last_loaded_checkpoint_info: Dict[str, Any] = {}

    def is_better(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def should_save_periodic(self) -> bool:
        """Check whether the configured checkpoint period has elapsed."""
        if self.checkpoint_period is None or self.checkpoint_period <= 0:
            return False
        return (time.time() - self.last_periodic_save_time) >= self.checkpoint_period

    def save_periodic(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        scaler: Optional[Any] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Save a periodic checkpoint based on elapsed time."""
        self.last_periodic_save_time = time.time()
        cfg_dict = config or {}
        state = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None,
            "metrics": metrics or {},
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "history": history or [],
            "config": cfg_dict,
        }

        periodic_path = os.path.join(self.checkpoint_dir, "checkpoint_periodic.pt")
        torch.save(state, periodic_path)
        periodic_cfg_path = os.path.join(self.checkpoint_dir, "checkpoint_periodic_config.yaml")
        save_config(cfg_dict, periodic_cfg_path)

        # Also update latest checkpoint for seamless resumption
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(state, latest_path)
        latest_cfg_path = os.path.join(self.checkpoint_dir, "checkpoint_latest_config.yaml")
        save_config(cfg_dict, latest_cfg_path)

        return {"periodic": periodic_path, "latest": latest_path}

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False,
        step: Optional[int] = None,
        scaler: Optional[Any] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Save checkpoints to disk."""
        current_score = metrics.get(self.metric_name, None)
        if current_score is not None and self.is_better(current_score):
            self.best_score = current_score
            self.best_epoch = epoch

        state = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None,
            "metrics": metrics,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "history": history or [],
            "config": config,
        }

        saved_paths = {}

        if self.save_latest:
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
            torch.save(state, latest_path)
            latest_cfg_path = os.path.join(self.checkpoint_dir, "checkpoint_latest_config.yaml")
            save_config(config, latest_cfg_path)
            saved_paths["latest"] = latest_path

        if current_score is not None and current_score == self.best_score:
            if self.save_best:
                best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
                torch.save(state, best_path)
                best_cfg_path = os.path.join(self.checkpoint_dir, "checkpoint_best_config.yaml")
                save_config(config, best_cfg_path)
                saved_paths["best"] = best_path

        return saved_paths

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        strict: Optional[bool] = None,
    ) -> Optional[int]:
        """Load latest checkpoint if available. Returns resumed epoch."""
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        if not os.path.exists(latest_path):
            return None
        return self.load_checkpoint(latest_path, model, optimizer, scheduler, scaler=scaler, strict=strict)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        strict: Optional[bool] = None,
    ) -> int:
        """Load specific checkpoint with robust strict/non-strict fallback for quantized/LoRA models."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        is_sam3 = "sam3" in getattr(model, "__class__", type(model)).__name__.lower() or getattr(model, "load_in_4bit", False) or hasattr(model, "pretrained_model_name_or_path")
        effective_strict = (not is_sam3) if strict is None else strict

        model_sd = model.state_dict()

        # Handle 'module.' prefix differences (e.g. from DDP)
        if not any(k in model_sd for k in state_dict.keys()):
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            elif any(f"module.{k}" in model_sd for k in state_dict.keys()):
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}

        # Filter out shape/size mismatches (e.g. 4-bit packed params loaded into float32 CPU model or vice versa)
        compatible_sd = {}
        mismatched_keys = []
        for k, v in state_dict.items():
            if k in model_sd:
                if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and v.shape != model_sd[k].shape:
                    mismatched_keys.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                    continue
            compatible_sd[k] = v

        if mismatched_keys:
            if strict is True:
                raise RuntimeError(
                    f"Size mismatch when loading checkpoint in strict mode for {len(mismatched_keys)} keys: "
                    f"{mismatched_keys[:3]}"
                )
            logger.warning(
                f"⚠️ [CHECKPOINT COMPATIBILITY] Skipped {len(mismatched_keys)} parameter(s) with shape/quantization "
                f"mismatches (e.g. 4-bit quantized vs float32). Successfully loaded {len(compatible_sd)} matching parameter(s)."
            )

        try:
            model.load_state_dict(compatible_sd, strict=effective_strict and not mismatched_keys)
        except RuntimeError as err:
            if not (strict is True):
                model.load_state_dict(compatible_sd, strict=False)
            else:
                raise err

        if optimizer is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            try:
                opt_sd = checkpoint["optimizer_state_dict"]
                try:
                    target_device = next(model.parameters()).device
                    if "state" in opt_sd:
                        for p_state in opt_sd["state"].values():
                            if isinstance(p_state, dict):
                                for sk, sv in p_state.items():
                                    if isinstance(sv, torch.Tensor) and sv.device != target_device:
                                        p_state[sk] = sv.to(target_device)
                except Exception:
                    pass
                optimizer.load_state_dict(opt_sd)
            except Exception as opt_err:
                logger.warning(f"Could not restore optimizer state ({opt_err}); continuing with initialized optimizer.")

        if scheduler is not None and isinstance(checkpoint, dict) and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as sched_err:
                logger.warning(f"Could not restore scheduler state ({sched_err}); continuing.")

        if scaler is not None and isinstance(checkpoint, dict) and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as scaler_err:
                logger.warning(f"Could not restore scaler state ({scaler_err}); continuing.")

        if isinstance(checkpoint, dict):
            if "best_score" in checkpoint and checkpoint["best_score"] is not None:
                self.best_score = float(checkpoint["best_score"])
            elif "metrics" in checkpoint and isinstance(checkpoint["metrics"], dict):
                m = checkpoint["metrics"]
                for candidate in [self.metric_name, f"val_{self.metric_name}", "val_iou", "iou", "val_dice", "val_f1"]:
                    if candidate in m and m[candidate] is not None:
                        self.best_score = float(m[candidate])
                        break
            if "best_epoch" in checkpoint and checkpoint["best_epoch"] is not None:
                self.best_epoch = int(checkpoint["best_epoch"])
            elif "epoch" in checkpoint and checkpoint["epoch"] is not None:
                self.best_epoch = int(checkpoint["epoch"])

        epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
        step = int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) and checkpoint.get("step") is not None else None

        self.last_loaded_checkpoint_info = {
            "epoch": epoch,
            "step": step,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "history": checkpoint.get("history", []) if isinstance(checkpoint, dict) else [],
            "metrics": checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {},
            "checkpoint_path": checkpoint_path,
        }
        return epoch


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
