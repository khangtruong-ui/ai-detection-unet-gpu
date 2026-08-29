"""
Trainer for SID-UNet segmentation and classification.
Supports mixed precision (AMP), step-based streaming epochs, learning rate scheduling,
evaluation report generation, and TensorBoard metric logging.
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
from sid_unet.utils.logger import MetricLogger, TensorboardLogger, setup_logger
from sid_unet.utils.report import format_metrics_table, generate_evaluation_report


class Trainer:
    """End-to-end training and validation loop manager."""

    def __init__(
        self,
        config: Any,
        model: Optional[UNet] = None,
        loss_fn: Optional[SIDTotalLoss] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
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
        self.tb_logger = TensorboardLogger(
            log_dir=config.logging.get("tensorboard_dir", os.path.join(self.output_dir, "runs")),
            enabled=config.logging.get("use_tensorboard", True),
        )

        # 3. Model, Loss, DataLoaders
        self.model = (model or build_model(config)).to(self.device)
        self.loss_fn = (loss_fn or build_loss(config)).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

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

        # 5. Mixed Precision & Gradient Clipping
        self.use_amp = bool(config.training.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = float(config.training.get("grad_clip_norm", 1.0))

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

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        metric_logger = MetricLogger()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)

        try:
            for batch in pbar:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                labels = batch.get("label")
                if labels is not None:
                    labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss, loss_dict = self.loss_fn(outputs, masks, labels)

                self.scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_size = images.size(0)
                metric_logger.update_dict(loss_dict, n=batch_size)
                self.global_step += 1

                if self.global_step % self.log_interval == 0:
                    for k, v in loss_dict.items():
                        self.tb_logger.log_scalar(f"Train/{k}", v, self.global_step)
                    self.tb_logger.log_scalar("Train/LearningRate", self.optimizer.param_groups[0]["lr"], self.global_step)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mask_loss": f"{loss_dict.get('mask_loss', 0.0):.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })
        finally:
            pbar.close()
            del pbar

        averages = metric_logger.averages()
        return averages

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], Optional[list]]:
        """Run validation loop and compute comprehensive metrics."""
        self.model.eval()
        seg_tracker = SegmentationMetricTracker(threshold=0.5)
        cls_tracker = ClassificationMetricTracker(num_classes=self.config.model.get("num_classes", 3))
        metric_logger = MetricLogger()

        first_batch_images = None
        first_batch_targets = None
        first_batch_preds = None

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]", leave=False)

        try:
            for i, batch in enumerate(pbar):
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

                # Update segmentation tracker
                seg_tracker.update(mask_logits, masks, labels)

                # Update classification tracker
                if class_logits is not None and labels is not None:
                    cls_tracker.update(class_logits, labels)

                # Save first batch for visualization
                if i == 0:
                    first_batch_images = images
                    first_batch_targets = masks
                    first_batch_preds = torch.sigmoid(mask_logits)
        finally:
            pbar.close()
            del pbar

        # Compute metrics
        overall_seg_metrics, per_label_seg_metrics = seg_tracker.compute()
        cls_metrics, confusion_mat = cls_tracker.compute()

        # Combine metrics
        val_summary = {}
        for k, v in metric_logger.averages().items():
            val_summary[f"val_{k}"] = v
        for k, v in overall_seg_metrics.items():
            val_summary[f"val_{k}"] = v
        for k, v in cls_metrics.items():
            val_summary[f"val_{k}"] = v

        # TensorBoard logging
        for k, v in val_summary.items():
            self.tb_logger.log_scalar(f"Validation/{k}", v, epoch)

        if first_batch_images is not None and self.config.logging.get("save_sample_images", True):
            self.tb_logger.log_image_comparison(
                tag="Validation/Sample_Masks",
                images=first_batch_images,
                targets=first_batch_targets,
                predictions=first_batch_preds,
                step=epoch,
                max_samples=int(self.config.logging.get("num_sample_images", 4)),
            )

        return val_summary, per_label_seg_metrics, confusion_mat

    def train(self) -> Dict[str, Any]:
        """Execute full training loop across all epochs."""
        self.logger.info(f"Starting training on device '{self.device}' for {self.epochs} epochs.")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        history = []
        best_val_score = float("-inf")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)

            # Run validation
            val_summary, per_label_metrics, confusion_mat = self.validate(epoch)

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

        # Generate final evaluation report
        final_val_summary, final_per_label, final_cm = self.validate(epoch=self.epochs)
        report_data = generate_evaluation_report(
            overall_metrics=final_val_summary,
            per_label_metrics=final_per_label,
            confusion_matrix=final_cm,
            config=self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
            output_dir=self.report_dir,
            report_name="training_final_report",
        )
        self.logger.info(f"\n{report_data['markdown']}")
        self.tb_logger.close()
        import gc
        gc.collect()

        return {
            "run_name": self.config.project.get("name", "sid_unet"),
            "best_score": self.ckpt_manager.best_score,
            "best_epoch": self.ckpt_manager.best_epoch,
            "final_metrics": final_val_summary,
            "per_label_metrics": final_per_label,
            "confusion_matrix": final_cm,
            "report_path": os.path.join(self.report_dir, "training_final_report.md"),
            "report_json_path": os.path.join(self.report_dir, "training_final_report.json"),
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
            "report_data": report_data,
        }
