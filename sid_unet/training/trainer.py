"""
Trainer for SID-UNet segmentation and classification.
Supports mixed precision (AMP), step-based streaming epochs, learning rate scheduling,
and evaluation report generation.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        self.model = (model or build_model(config)).to(self.device)
        self.loss_fn = (loss_fn or build_loss(config)).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 4. Optimizer & Scheduler
        self.lr = float(config.training.get("learning_rate", 1e-3))
        self.weight_decay = float(config.training.get("weight_decay", 1e-4))
        self.opt_name = config.training.get("optimizer", "adamw").lower()

        if self.opt_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.epochs = int(config.training.get("epochs", 10))
        self.scheduler_name = config.training.get("scheduler", "cosine").lower()
        self.scheduler = self._build_scheduler()

        # 5. Mixed Precision, Gradient Accumulation & Clipping
        self.use_amp = bool(config.training.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = float(config.training.get("grad_clip_norm", 1.0))
        self.gradient_accumulation_steps = max(1, int(config.training.get("gradient_accumulation_steps", 1)))
        self.auto_batch_size = bool(config.training.get("auto_batch_size", False))
        self.empty_cache_per_epoch = bool(config.training.get("empty_cache_per_epoch", True))
        self.log_memory = bool(config.logging.get("log_memory", True))

        # 6. Callbacks
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            metric_name=config.training.get("early_stopping_metric", "val_iou"),
            mode=config.training.get("early_stopping_mode", "max"),
            save_best=bool(config.training.get("save_best", True)),
            save_latest=bool(config.training.get("save_latest", True)),
        )
        self.early_stopping = EarlyStopping(
            patience=int(config.training.get("early_stopping_patience", 5)),
            mode=config.training.get("early_stopping_mode", "max"),
        )

        self.global_step = 0
        self.log_interval = int(config.logging.get("log_interval", 20))

    def _build_scheduler(self):
        if self.scheduler_name == "cosine":
            min_lr = float(self.config.training.get("min_lr", 1e-6))
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=min_lr)
        elif self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, self.epochs // 3), gamma=0.5)
        elif self.scheduler_name == "plateau":
            mode = "max" if self.config.training.get("early_stopping_mode", "max") == "max" else "min"
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, factor=0.5, patience=2)
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

    def _step_batch_train(self, batch: Dict[str, Any], loss_divisor: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute forward pass and scaled backward pass for a single training batch/sub-batch."""
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            outputs = self.model(images)
            loss, loss_dict = self.loss_fn(outputs, masks, labels)

        scaled_loss = loss / max(1.0, float(loss_divisor))
        self.scaler.scale(scaled_loss).backward()
        return loss, loss_dict

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch with gradient accumulation and Out-Of-Memory (OOM) recovery."""
        self.model.train()
        metric_logger = MetricLogger()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)

        if self.empty_cache_per_epoch:
            clear_memory_cache(self.device)

        self.optimizer.zero_grad()
        accum_steps = self.gradient_accumulation_steps
        step_in_epoch = 0
        total_batches = len(self.train_loader) if hasattr(self.train_loader, "__len__") else None

        try:
            for batch in pbar:
                step_in_epoch += 1
                batch_size = len(batch["image"])
                loss_dict_batch = {}
                loss_val = 0.0

                try:
                    loss_t, loss_dict_batch = self._step_batch_train(batch, loss_divisor=accum_steps)
                    loss_val = loss_t.item()
                except Exception as exc:
                    if is_oom_error(exc):
                        self.logger.warning(
                            f"⚠️ Out-of-Memory (OOM) on batch size {batch_size} (Epoch {epoch}, Step {step_in_epoch})! "
                            f"Clearing cache and recovering via micro-batching..."
                        )
                        clear_memory_cache(self.device)
                        self.optimizer.zero_grad()

                        # Split batch into smaller micro-batches
                        micro_bs = max(1, batch_size // 2)
                        sub_batches = split_batch(batch, micro_batch_size=micro_bs)
                        sub_divisor = len(sub_batches) * accum_steps

                        for sub_b in sub_batches:
                            try:
                                _, s_dict = self._step_batch_train(sub_b, loss_divisor=sub_divisor)
                                for k, v in s_dict.items():
                                    loss_dict_batch[k] = loss_dict_batch.get(k, 0.0) + (v / len(sub_batches))
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
                if is_accum_boundary:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                postfix_dict = {
                    "loss": f"{loss_val:.4f}",
                    "mask_loss": f"{loss_dict_batch.get('mask_loss', 0.0):.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                if self.device.type == "cuda" and self.log_memory:
                    vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    postfix_dict["vram"] = f"{vram_mb:.0f}MB"
                pbar.set_postfix(postfix_dict)
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

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            outputs = self.model(images)
            loss, loss_dict = self.loss_fn(outputs, masks, labels)

        metric_logger.update_dict(loss_dict, n=images.size(0))

        if isinstance(outputs, tuple):
            mask_logits, class_logits = outputs
        else:
            mask_logits, class_logits = outputs, None

        seg_tracker.update(mask_logits, masks, labels)
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
        pbar = tqdm(target_loader, desc=desc_str, leave=False)

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
        """Evaluate model on a specific DataLoader."""
        target_loader = loader or self.test_loader or self.val_loader
        if target_loader is None:
            raise ValueError("No DataLoader provided for evaluation.")
        return self.validate(epoch=None, loader=target_loader, split_name=split_name)

    def train(self) -> Dict[str, Any]:
        """Execute full training loop across all epochs with OOM resilience."""
        self.logger.info(f"Starting training on device '{self.device}' for {self.epochs} epochs.")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        if self.device.type == "cuda":
            self.logger.info(f"Initial GPU memory: {format_memory_summary(self.device)}")

        # Auto-tune batch size if enabled and on GPU
        self._check_and_auto_scale_batch_size()

        history = []
        best_val_score = float("-inf")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
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
                "val_loss": float(val_summary.get("val_total_loss", val_summary.get("val_loss", 0.0))),
                "val_mask_loss": float(val_summary.get("val_mask_loss", 0.0)),
                "val_aux_loss": float(val_summary.get("val_aux_loss", 0.0)),
                "val_iou": float(val_summary.get("val_iou", 0.0)),
                "val_dice": float(val_summary.get("val_dice", 0.0)),
                "val_pixel_acc": float(val_summary.get("val_pixel_acc", 0.0)),
                "val_precision": float(val_summary.get("val_precision", 0.0)),
                "val_recall": float(val_summary.get("val_recall", 0.0)),
                "val_aux_accuracy": float(val_summary.get("val_aux_accuracy", val_summary.get("val_accuracy", 0.0))),
                "train_metrics": train_metrics,
                "val_metrics": val_summary,
            }
            history.append(epoch_record)

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
            )

            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} [{epoch_time:.1f}s] - "
                f"Train Loss: {train_metrics.get('total_loss', 0.0):.4f} - "
                f"Val Loss: {val_summary.get('val_total_loss', 0.0):.4f} - "
                f"Val IoU: {val_summary.get('val_iou', 0.0):.4f} - "
                f"Val Dice: {val_summary.get('val_dice', 0.0):.4f}"
            )

            if "best" in saved_paths:
                self.logger.info(f"⭐ New best model saved to {saved_paths['best']} (score: {self.ckpt_manager.best_score:.4f})")

            # Check early stopping
            monitored_score = val_summary.get(self.ckpt_manager.metric_name, 0.0)
            if self.early_stopping(monitored_score):
                self.logger.info(f"Early stopping triggered at epoch {epoch}!")
                break

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
        }

