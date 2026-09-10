"""
Callbacks for training: checkpoint management and early stopping.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
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


def load_state_dict_compatible(
    model: nn.Module,
    state_dict: Dict[str, Any],
    strict: Optional[bool] = None,
    target_device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load state_dict into model with dynamic adaptation for quantized (4-bit/8-bit) and LoRA parameters.

    Features:
      - Automatically maps weights to the target model device.
      - Detects bitsandbytes 4-bit packed parameters and converts floating-point checkpoint weights
        directly into NormalFloat4 (NF4) Params4bit layers.
      - Dequantizes 4-bit weights into float/half if restoring into an unquantized model on CUDA.
      - Handles LoRA / PEFT models by ensuring all trained adapters are strictly restored while
        preserving pre-initialized base foundation weights.
      - Tolerates and corrects DDP 'module.' prefix mismatches.
    """
    if target_device is None:
        try:
            target_device = next(model.parameters()).device
        except Exception:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(target_device, str):
        if target_device == "auto":
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            target_device = torch.device(target_device)

    model_sd = model.state_dict()

    # Handle 'module.' prefix differences (e.g. from DDP)
    if not any(k in model_sd for k in state_dict.keys()):
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        elif any(f"module.{k}" in model_sd for k in state_dict.keys()):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    compatible_sd = {}
    adapted_keys = []
    preserved_keys = []
    mismatched_keys = []

    for k, v in state_dict.items():
        if k not in model_sd:
            compatible_sd[k] = v
            continue

        target_tensor = model_sd[k]
        if hasattr(v, "shape") and hasattr(target_tensor, "shape") and v.shape != target_tensor.shape:
            # Check for parent module and param name
            parts = k.split(".")
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p, None)
                if mod is None:
                    break
            param_name = parts[-1]

            adapted = False
            if mod is not None and hasattr(mod, param_name):
                target_param = getattr(mod, param_name)

                # Case 1: Model parameter is 4-bit (Params4bit / Linear4bit) and incoming tensor is float/half
                is_target_4bit = (
                    hasattr(target_param, "quant_state")
                    or type(target_param).__name__ == "Params4bit"
                    or (hasattr(mod, "weight") and type(mod.weight).__name__ == "Params4bit")
                )
                if is_target_4bit and hasattr(v, "dtype") and v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    try:
                        from bitsandbytes.nn.modules import Params4bit
                        from bitsandbytes.functional import quantize_4bit

                        q_type = getattr(target_param, "quant_type", "nf4")
                        dev = (
                            target_param.device
                            if (hasattr(target_param, "device") and target_param.device.type == "cuda")
                            else (target_device if target_device.type == "cuda" else torch.device("cuda"))
                        )
                        q_data, q_state = quantize_4bit(v.to(dev), quant_type=q_type)
                        new_param = Params4bit(q_data, requires_grad=False, quant_state=q_state, quant_type=q_type)
                        setattr(mod, param_name, new_param)
                        compatible_sd[k] = q_data
                        adapted = True
                        adapted_keys.append(k)
                    except Exception as e:
                        logger.debug(f"Dynamic 4-bit quantization failed for {k}: {e}")

                # Case 2: Model parameter is float and incoming tensor is 4-bit uint8
                elif hasattr(v, "dtype") and v.dtype == torch.uint8 and target_tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    try:
                        from bitsandbytes.functional import dequantize_4bit

                        qs = getattr(v, "quant_state", None)
                        if qs is None:
                            qs_key = f"{k}.quant_state.bitsandbytes__nf4"
                            qs = state_dict.get(qs_key, None)
                        if qs is not None:
                            dev = target_device if target_device.type == "cuda" else torch.device("cuda")
                            deq = dequantize_4bit(v.to(dev), quant_state=qs, quant_type="nf4")
                            compatible_sd[k] = deq.to(device=target_device, dtype=target_tensor.dtype)
                            adapted = True
                            adapted_keys.append(k)
                    except Exception as e:
                        logger.debug(f"Dynamic dequantization failed for {k}: {e}")

            if not adapted:
                # Check if this is a frozen base model parameter
                is_trainable = False
                if mod is not None and hasattr(mod, param_name):
                    p_obj = getattr(mod, param_name)
                    if isinstance(p_obj, nn.Parameter) and p_obj.requires_grad:
                        is_trainable = True
                if not is_trainable and "lora_" not in k and "classifier_head" not in k:
                    # Model already has pre-initialized foundation base weights
                    preserved_keys.append(k)
                else:
                    mismatched_keys.append((k, tuple(v.shape), tuple(target_tensor.shape)))
        else:
            compatible_sd[k] = v

    is_sam3 = "sam3" in getattr(model, "__class__", type(model)).__name__.lower() or getattr(model, "load_in_4bit", False) or hasattr(model, "pretrained_model_name_or_path")
    effective_strict = (not is_sam3) if strict is None else strict

    if mismatched_keys:
        if strict is True:
            raise RuntimeError(
                f"Size mismatch when loading checkpoint in strict mode for {len(mismatched_keys)} keys: "
                f"{mismatched_keys[:3]}"
            )
        logger.warning(
            f"⚠️ [CHECKPOINT COMPATIBILITY] Skipped {len(mismatched_keys)} incompatible parameter(s): {mismatched_keys[:3]}"
        )

    if adapted_keys or preserved_keys:
        logger.info(
            f"✅ [CHECKPOINT COMPATIBILITY] Successfully loaded {len(compatible_sd) + len(preserved_keys)}/{len(state_dict)} parameters "
            f"({len(compatible_sd) - len(adapted_keys)} exact match, {len(adapted_keys)} dynamically quantized/adapted, "
            f"{len(preserved_keys)} base foundation parameters preserved)."
        )

    try:
        model.load_state_dict(compatible_sd, strict=effective_strict and not (mismatched_keys or preserved_keys))
    except RuntimeError as err:
        if not (strict is True):
            model.load_state_dict(compatible_sd, strict=False)
        else:
            raise err

    return compatible_sd


class CheckpointManager:
    """Manages saving and loading model checkpoints (best, latest, and periodic)."""

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = "val_iou",
        mode: str = "max",
        save_best: bool = True,
        save_latest: bool = True,
        checkpoint_period: Optional[float] = 3600.0,
        checkpoint_steps: Optional[int] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.mode = mode
        self.save_best = save_best
        self.save_latest = save_latest
        self.checkpoint_period = checkpoint_period
        self.checkpoint_steps = checkpoint_steps
        self.last_periodic_save_time = time.time()
        self.last_periodic_step = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.last_loaded_checkpoint_info: Dict[str, Any] = {}

    def is_better(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def should_save_periodic(self, step: Optional[int] = None) -> bool:
        """Check whether the configured checkpoint period or step interval has elapsed."""
        now = time.time()
        if self.checkpoint_steps is not None and self.checkpoint_steps > 0 and step is not None:
            if step > 0 and (step - self.last_periodic_step) >= self.checkpoint_steps:
                return True
        if self.checkpoint_period is not None and self.checkpoint_period > 0:
            if (now - self.last_periodic_save_time) >= self.checkpoint_period:
                return True
        return False

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
        """Save a periodic checkpoint based on elapsed time or step count."""
        self.last_periodic_save_time = time.time()
        if step is not None:
            self.last_periodic_step = step
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
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Optional[int]:
        """Load latest checkpoint if available. Returns resumed epoch."""
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        if not os.path.exists(latest_path):
            return None
        return self.load_checkpoint(
            latest_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            strict=strict,
            map_location=map_location,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        strict: Optional[bool] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> int:
        """Load specific checkpoint with robust strict/non-strict fallback and dynamic quantization adaptation."""
        # 1. Resolve map_location
        if map_location is None:
            try:
                map_location = next(model.parameters()).device
            except Exception:
                map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(map_location, str):
            if map_location == "auto":
                map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                map_location = torch.device(map_location)

        # 2. Load checkpoint safely (weights_only=True first, falling back to weights_only=False)
        checkpoint = None
        for weights_only_flag in [True, False]:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
                    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only_flag)
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                # Fall back to CPU if target device runs out of memory or device mapping fails
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only_flag)
                    break
                except Exception:
                    continue
            except Exception:
                continue

        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        # 3. Use load_state_dict_compatible to handle LoRA, 4-bit/8-bit, and shape differences
        load_state_dict_compatible(
            model=model,
            state_dict=state_dict,
            strict=strict,
            target_device=map_location,
        )

        if optimizer is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            try:
                opt_sd = checkpoint["optimizer_state_dict"]
                try:
                    target_device = map_location
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
