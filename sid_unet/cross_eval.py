"""
CLI and execution engine for cross-evaluation of SID-UNet models.
Evaluates multiple checkpoints against multiple dataset/experiment configuration files (cross-evaluation matrix),
generates neighbor reports adjacent to checkpoint folders, and compiles a comprehensive master report with 2D matrices.

Usage:
    sid-cross-eval --cross-configs conf1.yaml conf2.yaml --checkpoints ckpt1.pt ckpt2.pt
    python -m sid_unet.cross_eval --cross-configs configs/experiments/*.yaml --checkpoints outputs/*/checkpoints/checkpoint_best.pt --split test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
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
from sid_unet.utils.report import (
    extract_key_hyperparameters,
    generate_evaluation_report,
    generate_checkpoint_cross_eval_report,
    generate_master_cross_evaluation_report,
)
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-evaluate SID-UNet checkpoints across multiple dataset/experiment configurations"
    )
    parser.add_argument(
        "--cross-configs",
        "--cross_configs",
        "--configs",
        "--config",
        nargs="+",
        dest="cross_configs",
        required=True,
        help="Path(s) to YAML configuration file(s) or glob pattern(s) representing target evaluation datasets/settings.",
    )
    parser.add_argument(
        "--checkpoints",
        "--checkpoint",
        nargs="+",
        dest="checkpoints",
        required=True,
        help="Path(s) to checkpoint .pt file(s), run directories, or glob pattern(s).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split override ('test', 'validation', 'val', 'train'). If omitted, uses split configured in each cross-config.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to evaluate per configuration (overrides config limits, e.g. --samples 500, -1 for full).",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config data.batch_size).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides applied to all evaluations (e.g. data.num_workers=0 project.device=cpu).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default=None,
        help="Master directory to store consolidated cross-evaluation report and matrices. Defaults to 'outputs/cross_evaluation'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for predicted mask probabilities (default: 0.5).",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default=None,
        help="Optional segment model for mask refinement (e.g. 'facebook/sam3'). Contrasts UNet mask areas with SAM segments via joins.",
    )
    return parser.parse_args()



def expand_config_patterns(patterns: List[str]) -> List[str]:
    """Expand list of config file paths, directory paths, or glob patterns into sorted unique file paths."""
    resolved: List[str] = []
    for pat in patterns:
        if any(c in pat for c in ["*", "?", "["]):
            matches = glob.glob(pat, recursive=True)
            resolved.extend([m for m in matches if os.path.isfile(m) and m.endswith((".yaml", ".yml"))])
        elif os.path.isdir(pat):
            matches = glob.glob(os.path.join(pat, "**", "*.yaml"), recursive=True) + \
                      glob.glob(os.path.join(pat, "**", "*.yml"), recursive=True)
            resolved.extend([m for m in matches if os.path.isfile(m)])
        elif os.path.isfile(pat):
            resolved.append(pat)
        else:
            raise FileNotFoundError(f"Configuration path not found: {pat}")

    seen = set()
    unique = []
    for p in resolved:
        abs_p = os.path.abspath(p)
        if abs_p not in seen:
            seen.add(abs_p)
            unique.append(p)
    return unique


def expand_checkpoint_patterns(patterns: List[str]) -> List[str]:
    """Expand list of checkpoint file paths, directories, or glob patterns into sorted unique file paths."""
    resolved: List[str] = []
    for pat in patterns:
        if any(c in pat for c in ["*", "?", "["]):
            matches = glob.glob(pat, recursive=True)
            resolved.extend([m for m in matches if os.path.isfile(m) and m.endswith(".pt")])
        elif os.path.isdir(pat):
            candidates = [
                os.path.join(pat, "checkpoints", "checkpoint_best.pt"),
                os.path.join(pat, "checkpoints", "checkpoint_latest.pt"),
                os.path.join(pat, "checkpoint_best.pt"),
            ]
            found = False
            for c in candidates:
                if os.path.isfile(c):
                    resolved.append(c)
                    found = True
                    break
            if not found:
                resolved.extend(glob.glob(os.path.join(pat, "**", "*.pt"), recursive=True))
        elif os.path.isfile(pat):
            resolved.append(pat)
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {pat}")

    seen = set()
    unique = []
    for p in resolved:
        abs_p = os.path.abspath(p)
        if abs_p not in seen:
            seen.add(abs_p)
            unique.append(p)
    return unique


def resolve_checkpoint_neighbor_dir(checkpoint_path: str) -> str:
    """
    Resolve directory adjacent / neighbor to the checkpoint folder.
    e.g. outputs/RUN/exp_01/checkpoints/checkpoint_best.pt -> outputs/RUN/exp_01/cross_eval_reports
    """
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(ckpt_dir) == "checkpoints":
        run_dir = os.path.dirname(ckpt_dir)
        return os.path.join(run_dir, "cross_eval_reports")
    return os.path.join(ckpt_dir, "cross_eval_reports")


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


def evaluate_checkpoint_on_config(
    checkpoint_path: str,
    config_path: str,
    split: Optional[str] = None,
    samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    overrides: Optional[List[str]] = None,
    threshold: float = 0.5,
    segment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a specific checkpoint against a specific dataset / experiment cross-configuration.
    Preserves model weights/architecture from the checkpoint while applying dataset, loss,
    and evaluation settings from the cross-config.
    """
    overrides = list(overrides or [])
    if batch_size is not None:
        overrides.append(f"data.batch_size={batch_size}")

    eval_config = load_config(config_path, overrides=overrides)

    # Load model from checkpoint
    model, ckpt_config = UNet.from_checkpoint(
        checkpoint_path,
        return_config=True,
    )

    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    ckpt_run_name = ckpt_config.project.get("name", ckpt_stem) if hasattr(ckpt_config, "project") else ckpt_stem
    cfg_stem = os.path.splitext(os.path.basename(config_path))[0]
    cfg_proj_name = eval_config.project.get("name", cfg_stem)

    device_str = eval_config.project.get("device", ckpt_config.project.get("device", "auto"))
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")

    refiner = None
    if segment:
        refiner = get_sam_refiner(segment, device=device, threshold=threshold)

    model.to(device)
    model.eval()

    loss_fn = build_loss(eval_config).to(device)

    eval_loader = create_eval_dataloader(eval_config, split=split, max_samples=samples)
    resolved_split = getattr(eval_loader.dataset, "resolved_split", split or eval_config.data.get("eval_split", "test"))
    dataset_name = eval_config.data.get("dataset_name", "KhangTruong/IMD2020")

    seg_tracker = SegmentationMetricTracker(threshold=threshold)
    cls_tracker = ClassificationMetricTracker(num_classes=ckpt_config.model.get("num_classes", 3))

    total_loss_sum = 0.0
    total_samples = 0
    all_change_metrics: List[Dict[str, Any]] = []

    clear_memory_cache(device)
    pbar = tqdm(
        eval_loader,
        desc=f"Cross-Eval [{ckpt_run_name} x {cfg_proj_name}] ({resolved_split})",
        leave=False,
    )

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

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": ckpt_run_name,
        "config_path": config_path,
        "config_name": cfg_proj_name,
        "dataset_name": dataset_name,
        "eval_split": resolved_split,
        "total_evaluated_samples": total_samples,
        "metrics": overall_metrics,
        "overall_metrics": overall_metrics,
        "per_label_metrics": per_label_seg,
        "confusion_matrix": confusion_mat,
        "checkpoint_config": ckpt_config.to_dict() if hasattr(ckpt_config, "to_dict") else dict(ckpt_config),
        "eval_config": eval_config.to_dict() if hasattr(eval_config, "to_dict") else dict(eval_config),
        "checkpoint_hyperparameters": extract_key_hyperparameters(ckpt_config),
        "eval_hyperparameters": extract_key_hyperparameters(eval_config),
    }


def run_cross_evaluation(
    checkpoint_paths: List[str],
    config_paths: List[str],
    split: Optional[str] = None,
    samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    threshold: float = 0.5,
    segment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full cross-evaluation for all (checkpoint, config) pairs.
    Generates neighbor reports for each checkpoint and consolidated master report.
    """
    overrides = list(overrides or [])
    master_output_dir = output_dir or "outputs/cross_evaluation"
    os.makedirs(master_output_dir, exist_ok=True)

    logger = setup_logger(
        name="SID_CrossEval",
        log_file=os.path.join(master_output_dir, "cross_evaluation.log"),
    )

    logger.info("=" * 75)
    logger.info(f"🚀 Starting Cross-Evaluation Matrix")
    logger.info(f"   Checkpoints ({len(checkpoint_paths)}): {checkpoint_paths}")
    logger.info(f"   Configurations ({len(config_paths)}): {config_paths}")
    logger.info(f"   Total Evaluation Runs: {len(checkpoint_paths) * len(config_paths)}")
    if segment:
        logger.info(f"   Segment Mask Refinement Model: {segment}")
    logger.info(f"   Master Output Directory: {master_output_dir}")
    logger.info("=" * 75)

    all_cross_results: List[Dict[str, Any]] = []
    checkpoint_to_results: Dict[str, List[Dict[str, Any]]] = {}

    run_idx = 0
    total_runs = len(checkpoint_paths) * len(config_paths)

    for ckpt_path in checkpoint_paths:
        checkpoint_to_results[ckpt_path] = []
        for cfg_path in config_paths:
            run_idx += 1
            logger.info(f"\n[{run_idx}/{total_runs}] Evaluating Checkpoint '{os.path.basename(ckpt_path)}' with Config '{os.path.basename(cfg_path)}'...")

            res = evaluate_checkpoint_on_config(
                checkpoint_path=ckpt_path,
                config_path=cfg_path,
                split=split,
                samples=samples,
                batch_size=batch_size,
                overrides=overrides,
                threshold=threshold,
                segment=segment,
            )

            all_cross_results.append(res)
            checkpoint_to_results[ckpt_path].append(res)

            m = res["metrics"]
            logger.info(
                f"   -> Dataset: '{res['dataset_name']}' ({res['eval_split']}) | "
                f"Loss: {m.get('eval_total_loss', 0.0):.4f} | "
                f"IoU: {m.get('iou', 0.0):.4f} | "
                f"F1: {m.get('f1', 0.0):.4f} | "
                f"AUROC: {m.get('auroc', 0.0):.4f} | "
                f"PixelAcc: {m.get('pixel_acc', 0.0):.4f}"
            )

    # 1. Generate neighbor report for each checkpoint folder
    for ckpt_path, results in checkpoint_to_results.items():
        neighbor_dir = resolve_checkpoint_neighbor_dir(ckpt_path)
        os.makedirs(neighbor_dir, exist_ok=True)
        generate_checkpoint_cross_eval_report(
            checkpoint_path=ckpt_path,
            results=results,
            output_dir=neighbor_dir,
            report_name="cross_evaluation_report",
        )
        # Also write per-config reports in neighbor dir
        for r in results:
            cfg_stem = r.get("config_name", "config").replace(" ", "_")
            generate_evaluation_report(
                overall_metrics=r["metrics"],
                per_label_metrics=r["per_label_metrics"],
                confusion_matrix=r["confusion_matrix"],
                config=r["eval_config"],
                output_dir=neighbor_dir,
                report_name=f"eval_{cfg_stem}_report",
            )
        logger.info(f"✅ Checkpoint neighbor report saved at: {os.path.join(neighbor_dir, 'cross_evaluation_report.md')}")

    # 2. Generate Master Cross-Evaluation Report
    master_report = generate_master_cross_evaluation_report(
        cross_results=all_cross_results,
        output_dir=master_output_dir,
        report_name="master_cross_evaluation_report",
    )

    print("\n" + "=" * 75)
    print("⭐ MASTER CROSS-EVALUATION REPORT")
    print("=" * 75)
    print(master_report["iou_matrix_table"])
    print("\n" + master_report["summary_table"])
    print(f"\n📄 Master Markdown Report: {os.path.join(master_output_dir, 'master_cross_evaluation_report.md')}")
    print(f"📊 Master JSON Report: {os.path.join(master_output_dir, 'master_cross_evaluation_report.json')}")
    print(f"📈 Cross Matrices: {os.path.join(master_output_dir, 'cross_eval_matrix.json')}\n")

    return {
        "master_report": master_report,
        "cross_results": all_cross_results,
        "checkpoint_results": checkpoint_to_results,
        "output_dir": master_output_dir,
    }


def main():
    args = parse_args()
    raw_configs = args.cross_configs if isinstance(args.cross_configs, list) else [args.cross_configs]
    config_paths = expand_config_patterns(raw_configs)

    raw_ckpts = args.checkpoints if isinstance(args.checkpoints, list) else [args.checkpoints]
    checkpoint_paths = expand_checkpoint_patterns(raw_ckpts)

    if not config_paths:
        raise FileNotFoundError(f"No valid configuration files found matching: {raw_configs}")
    if not checkpoint_paths:
        raise FileNotFoundError(f"No valid checkpoints found matching: {raw_ckpts}")

    results = run_cross_evaluation(
        checkpoint_paths=checkpoint_paths,
        config_paths=config_paths,
        split=args.split,
        samples=args.samples,
        batch_size=args.batch_size,
        overrides=args.override,
        output_dir=args.output_dir,
        threshold=args.threshold,
        segment=args.segment,
    )
    return results


def cli_main():
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli_main()

