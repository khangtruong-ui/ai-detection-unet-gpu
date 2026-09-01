"""
CLI evaluation and benchmarking entrypoint for SID-UNet.
Supports evaluating single or multiple checkpoints concurrently with local report overriding
and consolidated multi-checkpoint comparative benchmarking.

Usage:
    # Single checkpoint (saves/overwrites report in checkpoint's local directory):
    python -m sid_unet.evaluate --checkpoint outputs/RUN001/checkpoints/checkpoint_best.pt

    # Multiple checkpoints at once:
    python -m sid_unet.evaluate --checkpoints outputs/RUN001/checkpoints/checkpoint_best.pt outputs/RUN002/checkpoints/checkpoint_best.pt
    sid-eval --checkpoints outputs/*/checkpoints/checkpoint_best.pt --split test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional
import torch
from tqdm import tqdm

from sid_unet.dataset.loader import create_eval_dataloader
from sid_unet.losses.auxiliary import build_loss
from sid_unet.metrics.classification import ClassificationMetricTracker
from sid_unet.metrics.segmentation import SegmentationMetricTracker
from sid_unet.models.sam3_refiner import get_sam_refiner
from sid_unet.models.unet import UNet
from sid_unet.utils.config import load_config, apply_overrides, ConfigDict
from sid_unet.utils.logger import setup_logger
from sid_unet.utils.memory import clear_memory_cache, is_oom_error, split_batch, format_memory_summary
from sid_unet.utils.report import generate_evaluation_report, generate_multi_experiment_report
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SID-UNet checkpoint(s)")
    parser.add_argument(
        "--checkpoint",
        "--checkpoints",
        nargs="+",
        dest="checkpoint",
        required=True,
        help="Path(s) to checkpoint .pt file(s) or glob pattern(s). Pass multiple files to evaluate multiple checkpoints at once.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional; defaults to config embedded in checkpoint)",
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
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports. If omitted, reports override in-place in each checkpoint's local run directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for predicted mask probabilities",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default=None,
        help="Optional segment model for mask refinement (e.g. 'facebook/sam3'). Contrasts UNet mask areas with SAM segments via joins.",
    )
    return parser.parse_args()



def eval_single_batch(batch, model, loss_fn, device, seg_tracker, cls_tracker, refiner=None):
    """Execute forward evaluation for a single batch and return (loss_val * batch_size, batch_size, change_metrics_list)."""
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

    change_metrics_list = []
    if refiner is not None:
        refined_masks, change_metrics_list = refiner.refine_batch(images, mask_logits)
        eval_mask_input = refined_masks
    else:
        eval_mask_input = mask_logits

    seg_tracker.update(eval_mask_input, masks, labels)
    if class_logits is not None and labels is not None:
        cls_tracker.update(class_logits, labels)

    return loss.item() * b_size, b_size, change_metrics_list


def resolve_checkpoint_output_dir(checkpoint_path: str, specified_output_dir: Optional[str] = None, run_name: Optional[str] = None, multi_run: bool = False) -> str:
    """
    Resolve the target directory where the evaluation report will be written and override old reports.
    If specified_output_dir is given, use it (or subfolder in multi_run). Otherwise, place reports inside the checkpoint's local run directory.
    """
    if specified_output_dir:
        if multi_run and run_name:
            return os.path.join(specified_output_dir, run_name, "eval_reports")
        return specified_output_dir

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(ckpt_dir) == "checkpoints":
        run_dir = os.path.dirname(ckpt_dir)
        return os.path.join(run_dir, "eval_reports")

    return os.path.join(ckpt_dir, "eval_reports")


def evaluate_single_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    split: Optional[str] = None,
    samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    threshold: float = 0.5,
    segment: Optional[str] = None,
    multi_run: bool = False,
) -> Dict[str, Any]:
    """Run full evaluation on a single checkpoint and override report in its local directory."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    overrides = list(overrides or [])
    if batch_size is not None:
        overrides.append(f"data.batch_size={batch_size}")

    # Load model and config from checkpoint with optional overrides
    if config_path:
        override_cfg = load_config(config_path, overrides=overrides)
    elif overrides:
        override_cfg = ConfigDict(apply_overrides({}, overrides))
    else:
        override_cfg = None

    model, config = UNet.from_checkpoint(
        checkpoint_path,
        override_config=override_cfg,
        return_config=True,
    )

    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    run_name = config.project.get("name", ckpt_stem)

    resolved_out_dir = resolve_checkpoint_output_dir(checkpoint_path, output_dir, run_name=run_name, multi_run=multi_run)
    os.makedirs(resolved_out_dir, exist_ok=True)

    # Local checkpoint run directory for guaranteed in-place update
    local_out_dir = resolve_checkpoint_output_dir(checkpoint_path, None)
    os.makedirs(local_out_dir, exist_ok=True)

    logger = setup_logger(
        name=f"SID_Eval_{run_name}",
        log_file=os.path.join(resolved_out_dir, "evaluate.log"),
    )
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"Threshold: {threshold}")

    device_str = config.project.get("device", "auto")
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")
    logger.info(f"Evaluation device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU Memory: {format_memory_summary(device)}")

    refiner = None
    if segment:
        logger.info(f"Initializing Segment Mask Refiner with model: {segment}")
        refiner = get_sam_refiner(segment, device=device, threshold=threshold)
        logger.info(f"Segment Mask Refinement active ({segment})")

    model.to(device)
    model.eval()

    loss_fn = build_loss(config).to(device)

    # Build DataLoader for specified evaluation split (test, validation, etc.)
    eval_loader = create_eval_dataloader(config, split=split, max_samples=samples)
    resolved_split = getattr(eval_loader.dataset, "resolved_split", split or config.data.get("eval_split", "test"))
    requested_split = split or config.data.get("eval_split", config.data.get("test_split", "test"))
    logger.info(f"Evaluating dataset '{config.data.get('dataset_name')}' on split: '{resolved_split}' (requested: '{requested_split}')")

    seg_tracker = SegmentationMetricTracker(threshold=threshold)
    cls_tracker = ClassificationMetricTracker(num_classes=config.model.get("num_classes", 3))

    total_loss_sum = 0.0
    total_samples = 0
    all_change_metrics: List[Dict[str, Any]] = []

    clear_memory_cache(device)
    pbar = tqdm(eval_loader, desc=f"Evaluating [{run_name}] ({resolved_split})", leave=True)
    with torch.no_grad():
        for batch in pbar:
            b_sz = len(batch["image"])
            try:
                loss_contrib, count, change_met = eval_single_batch(batch, model, loss_fn, device, seg_tracker, cls_tracker, refiner=refiner)
                total_loss_sum += loss_contrib
                total_samples += count
                if change_met:
                    all_change_metrics.extend(change_met)
            except Exception as exc:
                if is_oom_error(exc):
                    logger.warning(
                        f"⚠️ OOM on evaluation batch size {b_sz}! Clearing cache and recovering with sub-batching..."
                    )
                    clear_memory_cache(device)
                    sub_batches = split_batch(batch, micro_batch_size=max(1, b_sz // 2))
                    for sub_b in sub_batches:
                        try:
                            loss_contrib, count, change_met = eval_single_batch(sub_b, model, loss_fn, device, seg_tracker, cls_tracker, refiner=refiner)
                            total_loss_sum += loss_contrib
                            total_samples += count
                            if change_met:
                                all_change_metrics.extend(change_met)
                        except Exception as sub_exc:
                            if is_oom_error(sub_exc):
                                clear_memory_cache(device)
                                nano_batches = split_batch(sub_b, micro_batch_size=1)
                                for nano_b in nano_batches:
                                    loss_contrib, count, change_met = eval_single_batch(nano_b, model, loss_fn, device, seg_tracker, cls_tracker, refiner=refiner)
                                    total_loss_sum += loss_contrib
                                    total_samples += count
                                    if change_met:
                                        all_change_metrics.extend(change_met)
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

    if refiner is not None and all_change_metrics:
        change_ratios = [m.get("pixel_change_ratio", 0.0) for m in all_change_metrics]
        pixels_changed = [m.get("pixels_changed", 0) for m in all_change_metrics]
        masks_refined = sum(1 for p in pixels_changed if p > 0)
        overall_metrics["segment_model"] = segment
        overall_metrics["segment_refinement"] = True
        overall_metrics["mean_pixel_change_ratio"] = float(np.mean(change_ratios))
        overall_metrics["total_masks_refined"] = masks_refined
        overall_metrics["refined_sample_count"] = len(all_change_metrics)
        overall_metrics["sam_refinement_stats"] = {
            "segment_model": segment,
            "mean_pixel_change_ratio": float(np.mean(change_ratios)),
            "total_pixels_changed": int(sum(pixels_changed)),
            "total_masks_refined": masks_refined,
            "total_samples": len(all_change_metrics),
        }
        logger.info(
            f"SAM Refinement Summary: Model='{segment}' | "
            f"Mean Pixel Change: {np.mean(change_ratios):.2%} | "
            f"Refined Masks: {masks_refined}/{len(all_change_metrics)}"
        )

    # Generate and override the report in the target output directory
    report_result = generate_evaluation_report(
        overall_metrics=overall_metrics,
        per_label_metrics=per_label_seg,
        confusion_matrix=confusion_mat,
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        output_dir=resolved_out_dir,
        report_name="evaluation_report",
    )

    # Also ensure local run directory report is overwritten
    if local_out_dir != resolved_out_dir:
        generate_evaluation_report(
            overall_metrics=overall_metrics,
            per_label_metrics=per_label_seg,
            confusion_matrix=confusion_mat,
            config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
            output_dir=local_out_dir,
            report_name="evaluation_report",
        )

    logger.info("\n" + report_result["markdown"])
    logger.info(f"Evaluation report overridden at: {os.path.join(resolved_out_dir, 'evaluation_report.md')}")

    return {
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "overall_metrics": overall_metrics,
        "final_metrics": overall_metrics,
        "per_label_metrics": per_label_seg,
        "confusion_matrix": confusion_mat,
        "report_path": os.path.join(resolved_out_dir, "evaluation_report.md"),
        "report_result": report_result,
        "output_dir": resolved_out_dir,
    }



def expand_checkpoint_patterns(patterns: List[str]) -> List[str]:
    """Expand list of checkpoint file paths or glob patterns into sorted unique file paths."""
    resolved_paths: List[str] = []
    for pattern in patterns:
        if any(char in pattern for char in ["*", "?", "["]):
            matches = glob.glob(pattern, recursive=True)
            resolved_paths.extend([m for m in matches if os.path.isfile(m)])
        elif os.path.isdir(pattern):
            candidates = [
                os.path.join(pattern, "checkpoints", "checkpoint_best.pt"),
                os.path.join(pattern, "checkpoint_best.pt"),
            ]
            found = False
            for c in candidates:
                if os.path.isfile(c):
                    resolved_paths.append(c)
                    found = True
                    break
            if not found:
                resolved_paths.extend(glob.glob(os.path.join(pattern, "**", "*.pt"), recursive=True))
        elif os.path.isfile(pattern):
            resolved_paths.append(pattern)
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {pattern}")

    seen = set()
    unique_paths = []
    for p in resolved_paths:
        abs_p = os.path.abspath(p)
        if abs_p not in seen:
            seen.add(abs_p)
            unique_paths.append(p)

    return unique_paths


def main():
    args = parse_args()
    raw_patterns = args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
    checkpoint_paths = expand_checkpoint_patterns(raw_patterns)

    if not checkpoint_paths:
        raise FileNotFoundError(f"No valid checkpoints found for pattern(s): {raw_patterns}")

    overrides = list(args.override)

    # Single Checkpoint Evaluation
    if len(checkpoint_paths) == 1:
        res = evaluate_single_checkpoint(
            checkpoint_path=checkpoint_paths[0],
            config_path=args.config,
            split=args.split,
            samples=args.samples,
            batch_size=args.batch_size,
            overrides=overrides,
            output_dir=args.output_dir,
            threshold=args.threshold,
            segment=args.segment,
            multi_run=False,
        )
        return res["report_result"]

    # Multiple Checkpoints Evaluation Suite
    print("\n" + "=" * 70)
    print(f"📊 Evaluating Multiple Checkpoints ({len(checkpoint_paths)} models)")
    print("=" * 70 + "\n")

    all_results: List[Dict[str, Any]] = []
    for i, ckpt_path in enumerate(checkpoint_paths, 1):
        print(f"\n>>> [{i}/{len(checkpoint_paths)}] Evaluating Checkpoint: {ckpt_path}")
        print("-" * 70)
        res = evaluate_single_checkpoint(
            checkpoint_path=ckpt_path,
            config_path=args.config,
            split=args.split,
            samples=args.samples,
            batch_size=args.batch_size,
            overrides=overrides,
            output_dir=args.output_dir,
            threshold=args.threshold,
            segment=args.segment,
            multi_run=True,
        )
        all_results.append(res)


    # Generate comparative benchmarking report across all evaluated checkpoints
    suite_output_dir = args.output_dir or "outputs"
    multi_report = generate_multi_experiment_report(
        experiment_results=all_results,
        output_dir=suite_output_dir,
        report_name="multi_checkpoint_evaluation",
    )

    print("\n" + "=" * 70)
    print("⭐ MULTI-CHECKPOINT EVALUATION SUMMARY")
    print("=" * 70)
    print(multi_report["summary_table"])
    print(f"\nDetailed Markdown comparison: {os.path.join(suite_output_dir, 'multi_checkpoint_evaluation.md')}")
    print(f"Detailed JSON comparison: {os.path.join(suite_output_dir, 'multi_checkpoint_evaluation.json')}\n")

    return all_results


def cli_main():
    import sys
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli_main()
