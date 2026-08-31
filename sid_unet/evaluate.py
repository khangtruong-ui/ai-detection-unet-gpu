"""
CLI evaluation and benchmarking entrypoint for SID-UNet.
Usage:
    python -m sid_unet.evaluate --checkpoint outputs/checkpoints/checkpoint_best.pt --config configs/evaluate.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import torch
from tqdm import tqdm

from sid_unet.dataset.loader import create_eval_dataloader, create_dataloaders
from sid_unet.losses.auxiliary import build_loss
from sid_unet.metrics.classification import ClassificationMetricTracker
from sid_unet.metrics.segmentation import SegmentationMetricTracker
from sid_unet.models.unet import UNet, build_model
from sid_unet.utils.config import load_config
from sid_unet.utils.logger import setup_logger
from sid_unet.utils.memory import clear_memory_cache, is_oom_error, split_batch, format_memory_summary
from sid_unet.utils.report import generate_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SID-UNet checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional; defaults to config embedded in checkpoint or evaluate.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate ('test', 'validation', 'val', 'train'). Defaults to test (with auto fallback to val).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to evaluate on (overrides config sample limits, e.g. --samples 500)",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config data.batch_size)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides (e.g. data.val_samples=1000 data.streaming=false)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports (defaults to outputs/eval_reports)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for predicted mask probabilities",
    )
    return parser.parse_args()


def eval_single_batch(batch, model, loss_fn, device, seg_tracker, cls_tracker):
    """Execute forward evaluation for a single batch and return (loss_val * batch_size, batch_size)."""
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    labels = batch.get("label")
    if labels is not None:
        labels = labels.to(device, non_blocking=True)

    outputs = model(images)
    loss, _ = loss_fn(outputs, masks, labels)

    b_size = images.size(0)
    if isinstance(outputs, tuple):
        mask_logits, class_logits = outputs
    else:
        mask_logits, class_logits = outputs, None

    seg_tracker.update(mask_logits, masks, labels)
    if class_logits is not None and labels is not None:
        cls_tracker.update(class_logits, labels)

    return loss.item() * b_size, b_size


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    overrides = list(args.override)
    if args.batch_size is not None:
        overrides.append(f"data.batch_size={args.batch_size}")

    # Load model and config from checkpoint with optional overrides
    if args.config:
        override_cfg = load_config(args.config, overrides=overrides)
    elif overrides:
        from sid_unet.utils.config import apply_overrides, ConfigDict
        override_cfg = ConfigDict(apply_overrides({}, overrides))
    else:
        override_cfg = None

    model, config = UNet.from_checkpoint(
        args.checkpoint,
        override_config=override_cfg,
        return_config=True,
    )

    output_dir = args.output_dir or os.path.join(config.project.get("output_dir", "outputs"), "eval_reports")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(
        name="SID_Eval",
        log_file=os.path.join(output_dir, "evaluate.log"),
    )
    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    logger.info(f"Threshold: {args.threshold}")

    device_str = config.project.get("device", "auto")
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")
    logger.info(f"Evaluation device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU Memory: {format_memory_summary(device)}")

    model.to(device)
    model.eval()

    loss_fn = build_loss(config).to(device)

    # Build DataLoader for specified evaluation split (test, validation, etc.)
    eval_loader = create_eval_dataloader(config, split=args.split, max_samples=args.samples)
    resolved_split = getattr(eval_loader.dataset, "resolved_split", args.split or config.data.get("eval_split", "test"))
    requested_split = args.split or config.data.get("eval_split", config.data.get("test_split", "test"))
    logger.info(f"Evaluating dataset '{config.data.get('dataset_name')}' on split: '{resolved_split}' (requested: '{requested_split}')")

    seg_tracker = SegmentationMetricTracker(threshold=args.threshold)
    cls_tracker = ClassificationMetricTracker(num_classes=config.model.get("num_classes", 3))

    total_loss_sum = 0.0
    total_samples = 0

    clear_memory_cache(device)
    pbar = tqdm(eval_loader, desc=f"Evaluating ({resolved_split})", leave=True)
    with torch.no_grad():
        for batch in pbar:
            batch_size = len(batch["image"])
            try:
                loss_contrib, count = eval_single_batch(batch, model, loss_fn, device, seg_tracker, cls_tracker)
                total_loss_sum += loss_contrib
                total_samples += count
            except Exception as exc:
                if is_oom_error(exc):
                    logger.warning(
                        f"⚠️ OOM on evaluation batch size {batch_size}! Clearing cache and recovering with sub-batching..."
                    )
                    clear_memory_cache(device)
                    sub_batches = split_batch(batch, micro_batch_size=max(1, batch_size // 2))
                    for sub_b in sub_batches:
                        try:
                            loss_contrib, count = eval_single_batch(sub_b, model, loss_fn, device, seg_tracker, cls_tracker)
                            total_loss_sum += loss_contrib
                            total_samples += count
                        except Exception as sub_exc:
                            if is_oom_error(sub_exc):
                                clear_memory_cache(device)
                                nano_batches = split_batch(sub_b, micro_batch_size=1)
                                for nano_b in nano_batches:
                                    loss_contrib, count = eval_single_batch(nano_b, model, loss_fn, device, seg_tracker, cls_tracker)
                                    total_loss_sum += loss_contrib
                                    total_samples += count
                            else:
                                raise sub_exc
                else:
                    raise exc

    clear_memory_cache(device)

    overall_seg, per_label_seg = seg_tracker.compute()
    cls_metrics, confusion_mat = cls_tracker.compute()

    overall_metrics = {
        "eval_split": resolved_split,
        "eval_total_loss": total_loss_sum / max(1, total_samples),
        **overall_seg,
        **cls_metrics,
        "total_evaluated_samples": total_samples,
    }

    report_result = generate_evaluation_report(
        overall_metrics=overall_metrics,
        per_label_metrics=per_label_seg,
        confusion_matrix=confusion_mat,
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        output_dir=output_dir,
        report_name="evaluation_report",
    )

    logger.info("\n" + report_result["markdown"])
    logger.info(f"Reports saved to: {output_dir}")
    return report_result


def cli_main():
    import sys
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli_main()
