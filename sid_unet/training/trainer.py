"""
Trainer for SID-UNet segmentation and classification.
Supports mixed precision (AMP), step-based streaming epochs, learning rate scheduling,
and evaluation report generation.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sid_unet.dataset.loader import safe_dataloader_len
from sid_unet.losses.auxiliary import SIDTotalLoss, build_loss
from sid_unet.metrics.classification import ClassificationMetricTracker
from sid_unet.metrics.segmentation import SegmentationMetricTracker
from sid_unet.models.unet import UNet, build_model
from sid_unet.training.callbacks import CheckpointManager, EarlyStopping
from sid_unet.utils.logger import MetricLogger, setup_logger
from sid_unet.utils.memory import (
    auto_scale_batch_size_and_grad_accum,
    clear_memory_cache,
    find_optimal_batch_size,
    format_memory_summary,
    get_memory_summary,
    is_oom_error,
    split_batch,
)
from sid_unet.utils.plotting import plot_training_curves, save_history_data
from sid_unet.utils.report import format_metrics_table, generate_evaluation_report
from sid_unet.utils.network import NetworkSpeedMonitor, BottleneckDetector
from sid_unet.utils.compatibility import check_8bit_compatibility

def parse_checkpoint_period(training_cfg: Any) -> float:
    """Parse checkpoint period from config, defaulting to 3600 seconds (1 hour)."""
    if not hasattr(training_cfg, "get"):
        return 3600.0

    raw = training_cfg.get(
        "checkpoint_period",
        training_cfg.get(
            "checkpoint_period_hours",
            training_cfg.get(
                "checkpoint_period_seconds",
                training_cfg.get("checkpoint_interval", 3600.0),
            ),
        ),
    )
    if raw is None:
        return 3600.0

    if isinstance(raw, (int, float)):
        if raw <= 0:
            return 0.0
        if "checkpoint_period_hours" in training_cfg or raw <= 24:
            return float(raw) * 3600.0
        return float(raw)

    if isinstance(raw, str):
        s = raw.strip().lower()
        if s.endswith("h"):
            return float(s[:-1]) * 3600.0
        elif s.endswith("m"):
            return float(s[:-1]) * 60.0
        elif s.endswith("s"):
            return float(s[:-1])
        try:
            val = float(s)
            return (val * 3600.0) if val <= 24 else val
        except ValueError:
            return 3600.0

    return 3600.0


class Trainer:
    """End-to-end training and validation loop manager with OOM safety and gradient accumulation."""

    def __init__(
        self,
        config: Any,
        model: Optional[UNet] = None,
        loss_fn: Optional[SIDTotalLoss] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        custom_logger: Optional[Any] = None,
    ):
        self.config = config

        # 1. Device configuration
        dev_cfg = config.project.get("device", "auto")
        if dev_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_cfg)

        # 2. Output and logging setup
        self.output_dir = config.project.get("output_dir", "outputs")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.report_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

        self.logger = custom_logger or setup_logger(
            name="SID_Trainer",
            log_file=os.path.join(self.log_dir, "training.log"),
        )

        # 3. Model, Loss, DataLoaders
        loaded_model = model or build_model(config)
        is_quantized = getattr(loaded_model, "load_in_4bit", False) or getattr(loaded_model, "load_in_8bit", False)
        if not is_quantized:
            self.model = loaded_model.to(self.device)
        else:
            self.model = loaded_model

        self.loss_fn = (loss_fn or build_loss(config)).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 3.1 Automatic 8-Bit Compatibility Verification at Training Time
        auto_check_8bit = bool(config.training.get("check_8bit_compatibility", True))
        if auto_check_8bit:
            self.is_8bit_compatible, self.compatibility_8bit_details = check_8bit_compatibility(
                device=self.device,
                verbose=False,
                custom_logger=self.logger,
            )
            if self.device.type == "cuda":
                if self.is_8bit_compatible:
                    self.logger.info(
                        f"💻 8-Bit Hardware & Library Capability: Supported "
                        f"({self.compatibility_8bit_details.get('device_name')}, "
                        f"{self.compatibility_8bit_details.get('compute_capability_str')}, "
                        f"bitsandbytes v{self.compatibility_8bit_details.get('bitsandbytes_version')})"
                    )
                else:
                    self.logger.debug(
                        f"💻 8-bit capability check result: {self.compatibility_8bit_details.get('message')}"
                    )
        else:
            self.is_8bit_compatible = False
            self.compatibility_8bit_details = {}

        # 3.2 Precision mode and GPU 16-bit configuration
        self.lr = float(config.training.get("learning_rate", 1e-3))
        self.weight_decay = float(config.training.get("weight_decay", 1e-4))
        self.opt_name = str(config.training.get("optimizer", "adamw")).lower()
        use_8bit_opt = bool(config.training.get("use_8bit_optimizer", False))

        # Determine native GPU 16-bit AMP dtype (bfloat16 if supported, else float16)
        if self.device.type == "cuda" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.amp_dtype_str = "bfloat16"
            else:
                self.amp_dtype = torch.float16
                self.amp_dtype_str = "float16"
        else:
            self.amp_dtype = torch.float32
            self.amp_dtype_str = "float32"

        # Default precision mode
        self.precision_mode = "8bit" if (use_8bit_opt or "8bit" in self.opt_name) else "16bit"

        # Validate 8-bit model quantization if requested
        if getattr(loaded_model, "load_in_8bit", False):
            if self.is_8bit_compatible:
                self.logger.info("⚡ 8-bit Model Quantization Active (bitsandbytes load_in_8bit).")
            else:
                fallback = bool(config.training.get("fallback_on_unsupported_8bit", True)) or bool(
                    config.training.get("fallback_to_16bit", True)
                )
                if fallback:
                    self.precision_mode = "gpu_16bit" if self.device.type == "cuda" else "cpu_fp32"
                    self.logger.warning(
                        f"⚠️ 8-Bit Model Quantization Fallback: load_in_8bit is not supported "
                        f"({self.compatibility_8bit_details.get('message')}). "
                        f"Falling back to GPU 16-bit mode ({self.amp_dtype_str} on {self.device})."
                    )
                    if self.device.type == "cuda":
                        self.model = loaded_model.to(device=self.device, dtype=self.amp_dtype)
                else:
                    raise RuntimeError(
                        f"Model configured with load_in_8bit=True, but 8-bit mode is not supported: "
                        f"{self.compatibility_8bit_details.get('message')}"
                    )

        # 4. Optimizer & Scheduler
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = list(self.model.parameters())

        # Support explicit 8-bit optimizer names or use_8bit_optimizer flag
        is_8bit_opt_requested = use_8bit_opt or self.opt_name in [
            "adamw8bit", "adamw_8bit", "8bit_adamw", "8bit_adam",
            "adam8bit", "adam_8bit", "paged_adamw8bit", "paged_adamw_8bit",
            "paged_adam8bit", "paged_adam_8bit", "8bit",
        ]

        if is_8bit_opt_requested:
            if self.is_8bit_compatible:
                try:
                    import bitsandbytes as bnb
                    if "paged" in self.opt_name:
                        self.optimizer = bnb.optim.PagedAdamW8bit(
                            trainable_params, lr=self.lr, weight_decay=self.weight_decay
                        )
                        opt_label = "PagedAdamW8bit (with CPU paging)"
                    else:
                        self.optimizer = bnb.optim.AdamW8bit(
                            trainable_params, lr=self.lr, weight_decay=self.weight_decay
                        )
                        opt_label = "AdamW8bit"
                    self.precision_mode = "8bit"
                    self.logger.info(
                        f"⚡ 8-Bit Optimizer Active: Initialized {opt_label} "
                        f"(75% optimizer VRAM savings on {self.compatibility_8bit_details.get('device_name', self.device)})."
                    )
                except Exception as opt_err:
                    fallback = bool(config.training.get("fallback_on_unsupported_8bit", True)) or bool(
                        config.training.get("fallback_to_16bit", True)
                    )
                    if fallback:
                        self.precision_mode = "gpu_16bit" if self.device.type == "cuda" else "cpu_fp32"
                        self.logger.warning(
                            f"⚠️ 8-Bit Optimizer initialization failed ({opt_err}). "
                            f"Falling back to GPU 16-bit mode (AMP {self.amp_dtype_str} + torch.optim.AdamW on {self.device})."
                        )
                        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
                    else:
                        raise opt_err
            else:
                fallback = bool(config.training.get("fallback_on_unsupported_8bit", True)) or bool(
                    config.training.get("fallback_to_16bit", True)
                )
                if fallback:
                    self.precision_mode = "gpu_16bit" if self.device.type == "cuda" else "cpu_fp32"
                    self.logger.warning(
                        f"⚠️ 8-Bit Training Mode Fallback: 8-bit optimizer requested ('{self.opt_name}'), but "
                        f"{self.compatibility_8bit_details.get('message', 'environment is incompatible')}. "
                        f"Falling back to GPU 16-bit mode (AMP {self.amp_dtype_str} + torch.optim.AdamW on {self.device})."
                    )
                    self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
                else:
                    raise RuntimeError(
                        f"8-bit optimizer '{self.opt_name}' requested but environment is incompatible: "
                        f"{self.compatibility_8bit_details.get('message')}"
                    )
        elif self.opt_name == "adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt_name == "sgd":
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        self.epochs = int(config.training.get("epochs", 10))
        self.scheduler_name = config.training.get("scheduler", "cosine").lower()
        self.scheduler = self._build_scheduler()

        # 5. Mixed Precision, Gradient Accumulation & Clipping
        # When falling back to GPU 16-bit mode or amp is enabled, ensure AMP is active on CUDA
        configured_amp = bool(config.training.get("amp", True))
        is_fallback_16bit = (self.precision_mode in ["gpu_16bit", "16bit"])
        self.use_amp = (configured_amp or is_fallback_16bit) and self.device.type == "cuda"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip = float(config.training.get("grad_clip_norm", 1.0))
        self.gradient_accumulation_steps = max(1, int(config.training.get("gradient_accumulation_steps", 1)))
        self.auto_batch_size = bool(config.training.get("auto_batch_size", True))
        self.empty_cache_per_epoch = bool(config.training.get("empty_cache_per_epoch", True))
        self.log_memory = bool(config.logging.get("log_memory", True))

        # 5.1 Debug Mode Configuration (nn-toolbox)
        raw_debug = config.training.get("debug_mode", config.training.get("debug", False))
        if isinstance(raw_debug, bool):
            self.debug_mode = "light" if raw_debug else False
        elif isinstance(raw_debug, str) and raw_debug.lower() in ("true", "1", "yes", "light"):
            self.debug_mode = "light"
        elif isinstance(raw_debug, str) and raw_debug.lower() == "deep":
            self.debug_mode = "deep"
        else:
            self.debug_mode = False

        # 6. Callbacks
        checkpoint_period = parse_checkpoint_period(config.training)
        checkpoint_steps = config.training.get("checkpoint_steps", config.training.get("checkpoint_interval_steps", None))
        if checkpoint_steps is not None:
            checkpoint_steps = int(checkpoint_steps)
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            metric_name=config.training.get("early_stopping_metric", "val_iou"),
            mode=config.training.get("early_stopping_mode", "max"),
            save_best=bool(config.training.get("save_best", True)),
            save_latest=bool(config.training.get("save_latest", True)),
            checkpoint_period=checkpoint_period,
            checkpoint_steps=checkpoint_steps,
        )
        self.early_stopping = EarlyStopping(
            patience=int(config.training.get("early_stopping_patience", 5)),
            mode=config.training.get("early_stopping_mode", "max"),
        )

        self.global_step = 0
        self.start_epoch = 0
        self.history: list = []
        self.log_interval = int(config.logging.get("log_interval", 20))

        # 5. Network Speed & Pipeline Bottleneck Monitoring
        logging_cfg = config.get("logging", {}) if hasattr(config, "get") else getattr(config, "logging", {})
        measure_net = logging_cfg.get("measure_network", True) if hasattr(logging_cfg, "get") else getattr(logging_cfg, "measure_network", True)
        if measure_net:
            method = str(logging_cfg.get("network_measure_method", "io_counters") if hasattr(logging_cfg, "get") else "io_counters")
            active_url = str(logging_cfg.get("network_probe_url", "https://huggingface.co") if hasattr(logging_cfg, "get") else "https://huggingface.co")
            check_interval = float(logging_cfg.get("bottleneck_check_interval", 30.0) if hasattr(logging_cfg, "get") else 30.0)
            self.network_monitor = NetworkSpeedMonitor(
                interval=1.0,
                method=method,
                active_url=active_url,
            )
            self.bottleneck_detector = BottleneckDetector(
                network_monitor=self.network_monitor,
                logger=self.logger,
                check_interval=check_interval,
            )
            self.network_monitor.start()
        else:
            self.network_monitor = None
            self.bottleneck_detector = None

    def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint state into model, optimizer, scheduler, scaler, and restore epoch/step/history."""
        self.logger.info(f"Loading checkpoint state from '{checkpoint_path}'...")
        resumed_epoch = self.ckpt_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=getattr(self, "scaler", None),
            strict=strict,
            map_location=self.device,
        )
        self.start_epoch = resumed_epoch
        if self.epochs <= self.start_epoch:
            orig_epochs = self.epochs
            self.epochs = self.start_epoch + orig_epochs
            self.logger.info(
                f"Resumed checkpoint completed epoch {self.start_epoch}, while configured epochs is {orig_epochs}. "
                f"Extending total epochs to {self.epochs} (will execute epochs {self.start_epoch + 1} to {self.epochs})."
            )
            if self.scheduler is not None and hasattr(self.scheduler, "T_max"):
                self.scheduler.T_max = self.epochs

        ckpt_meta = getattr(self.ckpt_manager, "last_loaded_checkpoint_info", {})
        if ckpt_meta.get("step") is not None:
            self.global_step = int(ckpt_meta["step"])

        # Restore best score in early stopping
        if self.ckpt_manager.best_score not in [float("-inf"), float("inf")]:
            self.early_stopping.best_score = self.ckpt_manager.best_score
            self.early_stopping.counter = 0

        # Restore history if present in checkpoint
        ckpt_history = ckpt_meta.get("history", [])
        if ckpt_history:
            self.history = list(ckpt_history)
        elif not self.history:
            history_json = os.path.join(self.report_dir, "training_history.json")
            if os.path.exists(history_json):
                try:
                    import json
                    with open(history_json, "r", encoding="utf-8") as f:
                        loaded_h = json.load(f)
                        if isinstance(loaded_h, list):
                            self.history = loaded_h
                except Exception:
                    pass

        self.logger.info(
            f"✅ Checkpoint state loaded successfully: Resumed Epoch {self.start_epoch}, "
            f"Step {self.global_step}, Best Score: {self.ckpt_manager.best_score}"
        )

        # Synchronize checkpoint_latest.pt with current model format so subsequent runs resume cleanly
        if self.ckpt_manager.save_latest:
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
            if not os.path.exists(latest_path) or os.path.abspath(checkpoint_path) != os.path.abspath(latest_path):
                try:
                    self.ckpt_manager.save_periodic(
                        epoch=self.start_epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metrics={"resumed_from": str(checkpoint_path), "step": self.global_step},
                        config=self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
                        step=self.global_step,
                        scaler=self.scaler,
                        history=self.history,
                    )
                except Exception as sync_err:
                    self.logger.warning(f"Could not write initial synced latest checkpoint: {sync_err}")

        return {
            "epoch": self.start_epoch,
            "step": self.global_step,
            "best_score": self.ckpt_manager.best_score,
            "best_epoch": self.ckpt_manager.best_epoch,
            "history": self.history,
        }

    def close(self) -> None:
        """Close and clean up data loaders, background prefetcher threads, and network monitor."""
        if hasattr(self, "network_monitor") and self.network_monitor is not None:
            try:
                self.network_monitor.stop()
            except Exception:
                pass
        for loader in (self.train_loader, self.val_loader, self.test_loader):
            if loader is not None and hasattr(loader, "close") and callable(loader.close):
                try:
                    loader.close()
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _build_scheduler(self):
        if self.scheduler_name == "cosine":
            min_lr = float(self.config.training.get("min_lr", 1e-6))
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=min_lr)
        elif self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, self.epochs // 3), gamma=0.5)
        elif self.scheduler_name == "plateau":
            mode = "max" if self.config.training.get("early_stopping_mode", "max") == "max" else "min"
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, factor=0.5, patience=2)
    def _run_debug_diagnostics(self) -> Optional[Any]:
        """Run automated diagnostic session using nn-toolbox."""
        mode_str = self.debug_mode if isinstance(self.debug_mode, str) else "light"
        self.logger.info(f"🔬 [DEBUG MODE] Initializing nn-toolbox diagnostic laboratory (Mode: {mode_str.upper()})...")
        try:
            from nn_toolbox import diagnose

            def _adapted_loss_fn(model_out, targets=None):
                if targets is not None:
                    loss, _ = self.loss_fn(model_out, targets)
                else:
                    out_t = model_out[0] if isinstance(model_out, (tuple, list)) else model_out
                    loss = out_t.float().sum()
                return loss

            diag_report = diagnose(
                model=self.model,
                dataloader=self.train_loader,
                loss_fn=_adapted_loss_fn,
                optimizer=self.optimizer,
                mode=mode_str,
                device=self.device,
                verbose=True,
            )

            diag_dir = os.path.join(self.report_dir, "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            json_path = os.path.join(diag_dir, "diagnostic_report.json")
            html_path = os.path.join(diag_dir, "diagnostic_report.html")

            diag_report.save_json(json_path)
            diag_report.save_html(html_path)

            self.logger.info(f"🔬 [DEBUG MODE] Diagnostic report saved to: {json_path} and {html_path}")

            # 1. Report what was analyzed and verified healthy (briefly)
            healthy = getattr(diag_report, "healthy_findings", [f for f in diag_report.findings if not f.is_actionable()])
            if healthy:
                self.logger.info(f"✅ [DEBUG MODE] Verified healthy learnability dimensions ({len(healthy)}):")
                for f in healthy:
                    mod_info = f" [{f.module}]" if f.module else ""
                    self.logger.info(f"   ✓ [HEALTHY] {f.category.upper()}{mod_info}: {f.observation}")

            # 2. Report actionable findings (warnings / critical anomalies)
            actionables = diag_report.actionable_findings
            if actionables:
                self.logger.warning(f"⚠️ [DEBUG MODE] {len(actionables)} actionable learnability issue(s) detected:")
                for f in actionables:
                    mod_info = f" [{f.module}]" if f.module else ""
                    self.logger.warning(f"   ! [{f.severity.upper()}] {f.category.upper()}{mod_info}: {f.observation}")
            else:
                self.logger.info("🎉 [DEBUG MODE] All learnability diagnostic checks passed cleanly with zero warnings.")

            # 3. Report prioritized investigation targets if any
            targets = diag_report.get_investigation_targets()
            if targets and actionables:
                self.logger.info("🎯 [DEBUG MODE] Prioritized investigation targets:")
                for idx, t in enumerate(targets[:3], 1):
                    hypos = f" (Hypotheses: {', '.join(t['hypotheses'][:2])})" if t['hypotheses'] else ""
                    self.logger.info(f"   {idx}. [{t['max_severity'].upper()}] {t['target']}{hypos}")

            self.diagnostic_report = diag_report
            return diag_report
        except ImportError:
            self.logger.warning("⚠️ [DEBUG MODE] nn-toolbox package is not installed; skipping diagnostics.")
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ [DEBUG MODE] Diagnostic session encountered an error: {e}")
            return None

    def _check_and_auto_scale_batch_size(self) -> None:
        """Probe GPU memory and scale down batch size / increase gradient accumulation steps if memory is tight."""
        if not self.auto_batch_size or self.train_loader is None or self.device.type != "cuda":
            return

        current_bs = int(self.config.data.get("batch_size", 16))
        img_size = tuple(self.config.data.get("image_size", [256, 256]))

        try:
            safe_bs = find_optimal_batch_size(
                model=self.model,
                loss_fn=self.loss_fn,
                sample_shape=(3, img_size[0], img_size[1]),
                device=self.device,
                max_batch_size=current_bs,
                min_batch_size=1,
                use_amp=self.use_amp,
                aux_classifier=getattr(self.model, "aux_classifier", True),
                num_classes=int(self.config.model.get("num_classes", 3)),
                logger=self.logger,
            )

            if safe_bs < current_bs:
                adjusted_bs, adjusted_grad_accum = auto_scale_batch_size_and_grad_accum(
                    requested_batch_size=current_bs,
                    safe_batch_size=safe_bs,
                    current_grad_accum=self.gradient_accumulation_steps,
                )
                self.logger.warning(
                    f"⚡ Auto Batch Sizing: Scaling batch size {current_bs} -> {adjusted_bs} "
                    f"with gradient_accumulation_steps={adjusted_grad_accum} "
                    f"(Preserving effective batch size: {current_bs * self.gradient_accumulation_steps})."
                )
                self.config.data.batch_size = adjusted_bs
                self.config.training.gradient_accumulation_steps = adjusted_grad_accum
                self.gradient_accumulation_steps = adjusted_grad_accum

                from sid_unet.dataset.loader import create_dataloaders
                eval_test = bool(self.config.data.get("evaluate_on_test", False))
                loaders = create_dataloaders(self.config, include_test=eval_test)
                if eval_test:
                    self.train_loader, self.val_loader, self.test_loader = loaders
                else:
                    self.train_loader, self.val_loader = loaders
        except Exception as e:
            self.logger.warning(f"Auto batch sizing probe encountered warning: {e}")
        finally:
            clear_memory_cache(self.device)
            gc.collect()

    def _step_batch_train(self, batch: Dict[str, Any], loss_divisor: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute forward pass and scaled backward pass for a single training batch/sub-batch."""
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(images)
            loss, loss_dict = self.loss_fn(outputs, masks, labels)

        scaled_loss = loss / max(1.0, float(loss_divisor))
        self.scaler.scale(scaled_loss).backward()

        # Compute training IoU metric on batch
        with torch.no_grad():
            mask_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if mask_logits is not None:
                preds = (torch.sigmoid(mask_logits.detach()) >= 0.5)
                targets = (masks >= 0.5)
                preds_flat = preds.view(preds.size(0), -1)
                targets_flat = targets.view(targets.size(0), -1)
                intersection = (preds_flat & targets_flat).sum(dim=1).float()
                union = (preds_flat | targets_flat).sum(dim=1).float()
                ious = torch.where(union == 0, torch.ones_like(intersection), intersection / torch.clamp(union, min=1.0))
                loss_dict["iou"] = float(ious.mean().item())

        return loss, loss_dict

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch with gradient accumulation and Out-Of-Memory (OOM) recovery."""
        if self.empty_cache_per_epoch:
            clear_memory_cache(self.device)

        self.model.train()
        metric_logger = MetricLogger()
        total_batches = safe_dataloader_len(self.train_loader)
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [Train]",
            total=total_batches,
            leave=False,
        )

        self.optimizer.zero_grad()
        accum_steps = self.gradient_accumulation_steps
        step_in_epoch = 0
        has_pending_grads = False

        prev_batch_end_time = time.perf_counter()
        try:
            for batch in pbar:
                data_time = time.perf_counter() - prev_batch_end_time
                compute_start_time = time.perf_counter()

                step_in_epoch += 1
                batch_size = len(batch["image"])
                loss_dict_batch = {}
                loss_val = 0.0

                try:
                    loss_t, loss_dict_batch = self._step_batch_train(batch, loss_divisor=accum_steps)
                    loss_val = loss_t.item()
                    has_pending_grads = True
                except Exception as exc:
                    if is_oom_error(exc):
                        self.logger.warning(
                            f"⚠️ Out-of-Memory (OOM) on batch size {batch_size} (Epoch {epoch}, Step {step_in_epoch})! "
                            f"Clearing cache and recovering via micro-batching..."
                        )
                        clear_memory_cache(self.device)
                        self.optimizer.zero_grad()
                        has_pending_grads = False

                        # Split batch into smaller micro-batches
                        micro_bs = max(1, batch_size // 2)
                        sub_batches = split_batch(batch, micro_batch_size=micro_bs)
                        sub_divisor = len(sub_batches) * accum_steps

                        for sub_b in sub_batches:
                            try:
                                _, s_dict = self._step_batch_train(sub_b, loss_divisor=sub_divisor)
                                for k, v in s_dict.items():
                                    loss_dict_batch[k] = loss_dict_batch.get(k, 0.0) + (v / len(sub_batches))
                                has_pending_grads = True
                            except Exception as sub_exc:
                                if is_oom_error(sub_exc):
                                    # Fallback to single sample micro-batching
                                    clear_memory_cache(self.device)
                                    nano_batches = split_batch(sub_b, micro_batch_size=1)
                                    nano_divisor = len(nano_batches) * sub_divisor
                                    for nano_b in nano_batches:
                                        _, n_dict = self._step_batch_train(nano_b, loss_divisor=nano_divisor)
                                        for k, v in n_dict.items():
                                            loss_dict_batch[k] = loss_dict_batch.get(k, 0.0) + (v / (len(sub_batches) * len(nano_batches)))
                                        has_pending_grads = True
                                else:
                                    raise sub_exc
                        loss_val = loss_dict_batch.get("total_loss", 0.0)
                    else:
                        raise exc

                metric_logger.update_dict(loss_dict_batch, n=batch_size)
                self.global_step += 1

                # Step optimizer on gradient accumulation boundary
                is_accum_boundary = (step_in_epoch % accum_steps == 0) or (
                    total_batches is not None and step_in_epoch == total_batches
                )
                if is_accum_boundary and has_pending_grads:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    has_pending_grads = False

                if self.ckpt_manager.should_save_periodic(step=self.global_step):
                    p_paths = self.ckpt_manager.save_periodic(
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metrics={"train_loss": float(loss_val), "step": self.global_step},
                        config=self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
                        step=self.global_step,
                        scaler=self.scaler,
                        history=self.history,
                    )
                    self.logger.info(f"⏱️ Periodic checkpoint saved to {p_paths['periodic']} (Step {self.global_step}, Epoch {epoch})")

                compute_time = time.perf_counter() - compute_start_time
                prev_batch_end_time = time.perf_counter()

                if self.bottleneck_detector is not None:
                    self.bottleneck_detector.record_step(data_time=data_time, compute_time=compute_time)

                net_str = self.network_monitor.get_speed_str() if self.network_monitor is not None else "0.0 MB/s"
                postfix_dict = {
                    "loss": f"{loss_val:.4f}",
                    "net": net_str,
                    "iou": f"{loss_dict_batch.get('iou', 0.0):.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                pbar.set_postfix(postfix_dict)

            # Flush pending accumulated gradients if last step did not land on accumulation boundary
            if has_pending_grads:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                has_pending_grads = False
        finally:
            pbar.close()
            del pbar
            if self.empty_cache_per_epoch:
                clear_memory_cache(self.device)

        averages = metric_logger.averages()
        return averages

    def _eval_batch_step(
        self,
        batch: Dict[str, Any],
        seg_tracker: SegmentationMetricTracker,
        cls_tracker: ClassificationMetricTracker,
        metric_logger: MetricLogger,
    ) -> None:
        """Execute single forward evaluation step with tracker updates."""
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(images)
            loss, loss_dict = self.loss_fn(outputs, masks, labels)

        metric_logger.update_dict(loss_dict, n=images.size(0))

        if isinstance(outputs, tuple):
            mask_logits, class_logits = outputs
        else:
            mask_logits, class_logits = outputs, None

        seg_tracker.update(mask_logits, masks, labels, is_logit=True)
        if class_logits is not None and labels is not None:
            cls_tracker.update(class_logits, labels)

    @torch.no_grad()
    def validate(
        self,
        epoch: Optional[int] = None,
        loader: Optional[DataLoader] = None,
        split_name: str = "val",
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], Optional[list]]:
        """Run validation or evaluation loop and compute comprehensive metrics with OOM safety."""
        self.model.eval()
        target_loader = loader if loader is not None else self.val_loader
        if target_loader is None:
            return {}, {}, None

        if self.empty_cache_per_epoch:
            clear_memory_cache(self.device)

        seg_tracker = SegmentationMetricTracker(threshold=0.5)
        cls_tracker = ClassificationMetricTracker(num_classes=self.config.model.get("num_classes", 3))
        metric_logger = MetricLogger()

        desc_str = f"Epoch {epoch}/{self.epochs} [{split_name.title()}]" if epoch is not None else f"Evaluating [{split_name.title()}]"
        val_total = safe_dataloader_len(target_loader)
        pbar = tqdm(target_loader, desc=desc_str, total=val_total, leave=False)

        try:
            for batch in pbar:
                batch_size = len(batch["image"])
                try:
                    self._eval_batch_step(batch, seg_tracker, cls_tracker, metric_logger)
                except Exception as exc:
                    if is_oom_error(exc):
                        self.logger.warning(
                            f"⚠️ OOM during validation on batch size {batch_size}! Recovering with chunked evaluation..."
                        )
                        clear_memory_cache(self.device)
                        sub_batches = split_batch(batch, micro_batch_size=max(1, batch_size // 2))
                        for sub_b in sub_batches:
                            try:
                                self._eval_batch_step(sub_b, seg_tracker, cls_tracker, metric_logger)
                            except Exception as sub_exc:
                                if is_oom_error(sub_exc):
                                    clear_memory_cache(self.device)
                                    nano_batches = split_batch(sub_b, micro_batch_size=1)
                                    for nano_b in nano_batches:
                                        self._eval_batch_step(nano_b, seg_tracker, cls_tracker, metric_logger)
                                else:
                                    raise sub_exc
                    else:
                        raise exc
        finally:
            pbar.close()
            del pbar
            if self.empty_cache_per_epoch:
                clear_memory_cache(self.device)

        # Compute metrics
        overall_seg_metrics, per_label_seg_metrics = seg_tracker.compute()
        cls_metrics, confusion_mat = cls_tracker.compute()

        # Combine metrics
        prefix = f"{split_name}_" if split_name else ""
        summary = {}
        for k, v in metric_logger.averages().items():
            summary[f"{prefix}{k}"] = v
        for k, v in overall_seg_metrics.items():
            summary[f"{prefix}{k}"] = v
        for k, v in cls_metrics.items():
            summary[f"{prefix}{k}"] = v

        return summary, per_label_seg_metrics, confusion_mat

    def evaluate(
        self,
        loader: Optional[DataLoader] = None,
        split_name: str = "test",
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], Optional[list]]:
        if loader is not None:
            target_loader = loader
        elif self.test_loader is not None:
            target_loader = self.test_loader
        else:
            target_loader = self.val_loader
        if target_loader is None:
            raise ValueError("No DataLoader provided for evaluation.")
        return self.validate(epoch=None, loader=target_loader, split_name=split_name)

    def train(self) -> Dict[str, Any]:
        """Execute full training loop across all epochs with OOM resilience."""
        self.logger.info(f"Starting training on device '{self.device}' for {self.epochs} epochs.")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        if self.device.type == "cuda":
            self.logger.info(f"Initial GPU memory: {format_memory_summary(self.device)}")

        # Run Debug Diagnostics via nn-toolbox if enabled
        if getattr(self, "debug_mode", False):
            self._run_debug_diagnostics()

        # Auto-tune batch size if enabled and on GPU
        self._check_and_auto_scale_batch_size()

        history = list(self.history)
        best_val_score = self.ckpt_manager.best_score
        start_time = time.time()
        self._current_epoch = self.start_epoch

        if self.start_epoch >= self.epochs:
            self.logger.info(
                f"Training already completed up to epoch {self.start_epoch} (configured epochs: {self.epochs}). "
                "Skipping training loop and generating final evaluation report."
            )
        else:
            try:
                for epoch in range(self.start_epoch + 1, self.epochs + 1):
                    self._current_epoch = epoch
                    epoch_start = time.time()
                    train_metrics = self.train_epoch(epoch)

                    # Run validation
                    val_summary, per_label_metrics, confusion_mat = self.validate(epoch)

                    # Record per-epoch history
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    epoch_record = {
                        "epoch": epoch,
                        "lr": current_lr,
                        "train_loss": float(train_metrics.get("total_loss", 0.0)),
                        "train_mask_loss": float(train_metrics.get("mask_loss", 0.0)),
                        "train_aux_loss": float(train_metrics.get("aux_loss", 0.0)),
                        "train_iou": float(train_metrics.get("iou", 0.0)),
                        "val_loss": float(val_summary.get("val_total_loss", val_summary.get("val_loss", 0.0))),
                        "val_mask_loss": float(val_summary.get("val_mask_loss", 0.0)),
                        "val_aux_loss": float(val_summary.get("val_aux_loss", 0.0)),
                        "val_iou": float(val_summary.get("val_iou", 0.0)),
                        "val_dice": float(val_summary.get("val_dice", 0.0)),
                        "val_f1": float(val_summary.get("val_f1", val_summary.get("val_dice", 0.0))),
                        "val_pixel_f1": float(val_summary.get("val_pixel_f1", val_summary.get("val_dice", 0.0))),
                        "val_auroc": float(val_summary.get("val_auroc", val_summary.get("val_pixel_auroc", 0.0))),
                        "val_pixel_auroc": float(val_summary.get("val_pixel_auroc", val_summary.get("val_auroc", 0.0))),
                        "val_pixel_acc": float(val_summary.get("val_pixel_acc", 0.0)),
                        "val_precision": float(val_summary.get("val_precision", 0.0)),
                        "val_recall": float(val_summary.get("val_recall", 0.0)),
                        "val_aux_accuracy": float(val_summary.get("val_aux_accuracy", val_summary.get("val_accuracy", 0.0))),
                        "train_metrics": train_metrics,
                        "val_metrics": val_summary,
                    }
                    history.append(epoch_record)
                    self.history = history

                    # Step scheduler
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            monitored_val = val_summary.get(self.ckpt_manager.metric_name, 0.0)
                            self.scheduler.step(monitored_val)
                        else:
                            self.scheduler.step()

                    # Save checkpoint
                    saved_paths = self.ckpt_manager.save(
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metrics=val_summary,
                        config=self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
                        step=self.global_step,
                        scaler=self.scaler,
                        history=history,
                    )

                    epoch_time = time.time() - epoch_start
                    self.logger.info(
                        f"Epoch {epoch:02d}/{self.epochs:02d} [{epoch_time:.1f}s] - "
                        f"Train Loss: {train_metrics.get('total_loss', 0.0):.4f} - "
                        f"Train IoU: {train_metrics.get('iou', 0.0):.4f} - "
                        f"Val Loss: {val_summary.get('val_total_loss', 0.0):.4f} - "
                        f"Val IoU: {val_summary.get('val_iou', 0.0):.4f} - "
                        f"Val F1: {val_summary.get('val_f1', val_summary.get('val_dice', 0.0)):.4f} - "
                        f"Val AUROC: {val_summary.get('val_auroc', 0.0):.4f}"
                    )

                    if "best" in saved_paths:
                        self.logger.info(f"⭐ New best model saved to {saved_paths['best']} (score: {self.ckpt_manager.best_score:.4f})")

                    # Check periodic checkpointing
                    if self.ckpt_manager.should_save_periodic(step=self.global_step):
                        p_paths = self.ckpt_manager.save_periodic(
                            epoch=epoch,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            metrics=val_summary,
                            config=self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
                            step=self.global_step,
                            scaler=self.scaler,
                            history=history,
                        )
                        self.logger.info(f"⏱️ Periodic checkpoint saved to {p_paths['periodic']} (Epoch {epoch})")

                    # Check early stopping
                    monitored_score = val_summary.get(self.ckpt_manager.metric_name, 0.0)
                    if self.early_stopping(monitored_score):
                        self.logger.info(f"Early stopping triggered at epoch {epoch}!")
                        break
            except KeyboardInterrupt:
                self.logger.warning(
                    "⚠️ Training interrupted by user (KeyboardInterrupt). Saving emergency checkpoint..."
                )
                try:
                    curr_ep = getattr(self, "_current_epoch", self.start_epoch)
                    self.ckpt_manager.save_periodic(
                        epoch=curr_ep,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metrics={"interrupted": True},
                        config=self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
                        step=self.global_step,
                        scaler=self.scaler,
                        history=history,
                    )
                    self.logger.info("Emergency checkpoint successfully saved to checkpoint_latest.pt and checkpoint_periodic.pt.")
                except Exception as save_err:
                    self.logger.error(f"Failed to save emergency checkpoint: {save_err}")
                self.close()
                raise

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes.")

        # Save raw history data (JSON and CSV)
        history_paths = save_history_data(history, output_dir=self.report_dir, prefix="training_history")
        self.logger.info(f"Training history saved to {history_paths.get('json')} and {history_paths.get('csv')}")

        # Generate publication-quality training curves graph (PNG, JPG, PDF)
        curves_png_path = os.path.join(self.report_dir, "training_curves.png")
        saved_curves = plot_training_curves(
            history=history,
            output_path=curves_png_path,
            title_suffix=f"({self.config.project.get('name', 'UNet')})",
            formats=["png", "pdf", "jpg"],
        )
        primary_curves_path = saved_curves[0] if saved_curves else curves_png_path
        self.logger.info(f"📈 Training curves plotted and saved to: {primary_curves_path}")

        # Generate final validation evaluation report
        final_val_summary, final_per_label, final_cm = self.validate(epoch=self.epochs)
        report_data = generate_evaluation_report(
            overall_metrics=final_val_summary,
            per_label_metrics=final_per_label,
            confusion_matrix=final_cm,
            config=self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
            output_dir=self.report_dir,
            report_name="training_final_report",
            history=history,
            curves_path=primary_curves_path,
        )
        self.logger.info(f"\n{report_data['markdown']}")

        # Optional test evaluation on best checkpoint
        test_results = None
        test_report_path = None
        if self.test_loader is not None:
            best_ckpt = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            if os.path.exists(best_ckpt):
                self.logger.info(f"Loading best checkpoint from '{best_ckpt}' for test set evaluation...")
                self.ckpt_manager.load_checkpoint(best_ckpt, self.model)

            self.logger.info("Running evaluation on test split...")
            test_summary, test_per_label, test_cm = self.evaluate(loader=self.test_loader, split_name="test")
            test_report_data = generate_evaluation_report(
                overall_metrics=test_summary,
                per_label_metrics=test_per_label,
                confusion_matrix=test_cm,
                config=self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
                output_dir=self.report_dir,
                report_name="test_evaluation_report",
            )
            test_report_path = os.path.join(self.report_dir, "test_evaluation_report.md")
            self.logger.info(f"🧪 Test Set Evaluation Report:\n{test_report_data['markdown']}")
            test_results = {
                "metrics": test_summary,
                "per_label_metrics": test_per_label,
                "confusion_matrix": test_cm,
                "report_path": test_report_path,
            }

        import gc
        gc.collect()
        self.close()

        return {
            "run_name": self.config.project.get("name", "sid_unet"),
            "best_score": self.ckpt_manager.best_score,
            "best_epoch": self.ckpt_manager.best_epoch,
            "final_metrics": final_val_summary,
            "per_label_metrics": final_per_label,
            "confusion_matrix": final_cm,
            "test_results": test_results,
            "test_report_path": test_report_path,
            "history": history,
            "curves_plot_path": primary_curves_path,
            "all_curves_paths": saved_curves,
            "history_json_path": history_paths.get("json"),
            "history_csv_path": history_paths.get("csv"),
            "report_path": os.path.join(self.report_dir, "training_final_report.md"),
            "report_json_path": os.path.join(self.report_dir, "training_final_report.json"),
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
            "report_data": report_data,
            "diagnostic_report": getattr(self, "diagnostic_report", None),
        }

