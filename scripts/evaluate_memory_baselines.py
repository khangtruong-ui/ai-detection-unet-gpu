"""
Benchmark script comparing Detective SAM under 3 conditions:
A. Original Detective SAM without memory (memory_enabled=False).
B. Detective SAM with nearest-neighbor forensic vector memory retrieval.
C. Detective SAM with randomized retrieval of the same number of memory entries.

Evaluates on identical streaming batches from COCO-inpainted.
Measures IoU, F1 score, end-to-end latency, retrieval latency, peak GPU memory,
database entry count, and CPU memory usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from sid_unet.dataset.loader import create_dataloaders
from sid_unet.models.detective_sam import DetectiveSAM
from sid_unet.utils.config import ConfigDict
from sid_unet.utils.memory import clear_memory_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_memory_baselines")


def compute_metrics(
    preds_binary: torch.Tensor, targets_binary: torch.Tensor
) -> Tuple[float, float]:
    """Compute mean IoU and F1 (Dice) score across a batch."""
    p_flat = preds_binary.view(preds_binary.size(0), -1).float()
    t_flat = targets_binary.view(targets_binary.size(0), -1).float()

    intersection = (p_flat * t_flat).sum(dim=1)
    union = (p_flat + t_flat).clamp(max=1.0).sum(dim=1)
    total_pixels = p_flat.sum(dim=1) + t_flat.sum(dim=1)

    iou = torch.where(
        union == 0,
        torch.ones_like(intersection),
        intersection / torch.clamp(union, min=1.0),
    )
    dice = torch.where(
        total_pixels == 0,
        torch.ones_like(intersection),
        (2.0 * intersection) / torch.clamp(total_pixels, min=1.0),
    )

    return float(iou.mean().item()), float(dice.mean().item())


def load_cached_batches(
    num_batches: int = 10,
    batch_size: int = 2,
    image_size: int = 256,
    dataset_name: str = "KhangTruong/COCO-inpainted",
    split: str = "validation",
) -> List[Dict[str, torch.Tensor]]:
    """Load and cache identical evaluation batches in memory."""
    logger.info(
        f"Loading {num_batches} fixed '{split}' batches from {dataset_name} "
        f"(batch_size={batch_size}, size={image_size})..."
    )
    cfg = ConfigDict({
        "project": {"seed": 42},
        "data": {
            "dataset_name": dataset_name,
            "streaming": True,
            "image_size": [image_size, image_size],
            "batch_size": batch_size,
            "num_workers": 0,
            "train_split": "train",
            "val_split": split,
            "val_samples": num_batches * batch_size,
            "augmentations": {
                "horizontal_flip": 0.0,
                "vertical_flip": 0.0,
                "random_rotate90": 0.0,
                "color_jitter": 0.0,
            },
        },
    })
    _, val_loader = create_dataloaders(cfg)
    batches = []
    for i, b in enumerate(val_loader):
        if i >= num_batches:
            break
        batches.append({
            "image": b["image"].clone(),
            "mask": (b["mask"] >= 0.5).float().clone(),
        })
    logger.info(f"Cached {len(batches)} evaluation batches in host RAM.")
    return batches


def run_condition_eval(
    model: DetectiveSAM,
    batches: List[Dict[str, torch.Tensor]],
    condition_name: str,
    memory_enabled: bool,
    random_retrieval: bool,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run evaluation for a single condition across all cached batches."""
    logger.info(
        f"\n=======================================================\n"
        f"  Evaluating Condition: {condition_name}\n"
        f"  (memory_enabled={memory_enabled}, random_retrieval={random_retrieval})\n"
        f"======================================================="
    )

    model.memory_enabled = memory_enabled
    if model.forensic_memory is not None:
        model.forensic_memory.memory_enabled = memory_enabled

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    ious: List[float] = []
    f1s: List[float] = []
    latencies_ms: List[float] = []
    retrieval_latencies_ms: List[float] = []

    with torch.no_grad():
        for idx, batch in enumerate(batches):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_start = time.perf_counter()

            # Forward pass with return_dict to capture retrieval latency
            res = model(
                images,
                return_dict=True,
                random_retrieval=random_retrieval,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_end = time.perf_counter()

            batch_lat_ms = (t_end - t_start) * 1000.0
            latencies_ms.append(batch_lat_ms)

            ret_lat_ms = float(res.get("retrieval_latency", 0.0)) * 1000.0
            retrieval_latencies_ms.append(ret_lat_ms)

            mask_logits = res["mask_logits"]
            preds = (torch.sigmoid(mask_logits) >= threshold).float()

            batch_iou, batch_f1 = compute_metrics(preds, targets)
            ious.append(batch_iou)
            f1s.append(batch_f1)

    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    db_bytes = model.forensic_memory.get_memory_bytes() if model.forensic_memory else 0
    db_entries = model.forensic_memory.entry_count if model.forensic_memory else 0

    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    mean_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    mean_lat_ms = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
    mean_ret_ms = float(sum(retrieval_latencies_ms) / len(retrieval_latencies_ms)) if retrieval_latencies_ms else 0.0

    logger.info(
        f"Result [{condition_name}]: IoU={mean_iou:.4f}, F1={mean_f1:.4f}, "
        f"Latency={mean_lat_ms:.2f}ms/batch, Retrieval={mean_ret_ms:.3f}ms, "
        f"Peak GPU={peak_gpu_mb:.1f}MB, DB Entries={db_entries}"
    )

    return {
        "condition": condition_name,
        "memory_enabled": memory_enabled,
        "random_retrieval": random_retrieval,
        "iou": round(mean_iou, 4),
        "f1": round(mean_f1, 4),
        "latency_ms_per_batch": round(mean_lat_ms, 2),
        "retrieval_latency_ms": round(mean_ret_ms, 3),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 2),
        "cpu_memory_mb": round(db_bytes / (1024 ** 2), 4),
        "database_entry_count": db_entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Detective SAM with/without Forensic Vector Memory")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of eval batches (default: 10)")
    parser.add_argument("--batch-size", type=int, default=2, help="Eval batch size (default: 2)")
    parser.add_argument("--image-size", type=int, default=256, help="Evaluation image resolution (default: 256)")
    parser.add_argument("--dataset-name", type=str, default="KhangTruong/COCO-inpainted")
    parser.add_argument("--output-json", type=str, default="benchmark_memory_baselines_results.json")
    parser.add_argument("--use-dummy", action="store_true", help="Use dummy backbone instead of real SAM2")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    logger.info(f"Running evaluation benchmark on device: {device}")

    # 1. Instantiate Detective SAM with memory capability enabled
    logger.info("Initializing DetectiveSAM...")
    model = DetectiveSAM(
        variant="small",
        layer_width=64,
        target_size=(512, 512),
        use_dummy=args.use_dummy or not torch.cuda.is_available(),
        memory_enabled=True,
        memory_key_dim=64,
        memory_value_dim=64,
        memory_retrieval_k=8,
        memory_similarity_metric="cosine",
        device=device,
    ).to(device)

    # 2. Pre-cache fixed evaluation batches from COCO-inpainted
    val_batches = load_cached_batches(
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        image_size=args.image_size,
        dataset_name=args.dataset_name,
        split="validation",
    )

    # 3. Collect genuine training observations to populate authoritative memory
    logger.info("Collecting forensic memory observations from training split...")
    train_cfg = ConfigDict({
        "project": {"seed": 42},
        "data": {
            "dataset_name": args.dataset_name,
            "streaming": True,
            "image_size": [args.image_size, args.image_size],
            "batch_size": args.batch_size,
            "num_workers": 0,
            "train_split": "train",
            "val_split": "validation",
            "train_samples_per_epoch": 20 * args.batch_size,
        },
    })
    train_loader, _ = create_dataloaders(train_cfg)
    model.train()
    model.memory_enabled = True

    obs_collected = 0
    t_coll_start = time.perf_counter()
    for step_i, b in enumerate(train_loader):
        if step_i >= 20:
            break
        imgs = b["image"].to(device)
        gt = (b["mask"] >= 0.5).float().to(device)
        with torch.no_grad():
            # Training forward pass collects tampered features
            _ = model(imgs, gt_masks=gt)
        obs_collected += 1

    # Consolidate collected memory observations
    t_cons_start = time.perf_counter()
    cons_stats = model.consolidate_memory()
    t_cons_end = time.perf_counter()
    cons_duration = t_cons_end - t_cons_start

    logger.info(
        f"Memory populated: {cons_stats.get('final_entry_count')} entries consolidated "
        f"in {cons_duration:.3f}s (from {cons_stats.get('new_observations')} raw observations)"
    )

    # 4. Run comparative evaluations
    results = []

    # Condition A: Original Detective SAM without memory
    res_a = run_condition_eval(
        model=model,
        batches=val_batches,
        condition_name="A: Original Detective SAM (No Memory)",
        memory_enabled=False,
        random_retrieval=False,
        device=device,
    )
    res_a["memory_consolidation_duration_sec"] = 0.0
    results.append(res_a)

    # Condition B: Detective SAM with Nearest-Neighbor Memory Retrieval
    res_b = run_condition_eval(
        model=model,
        batches=val_batches,
        condition_name="B: Detective SAM with NN Memory Retrieval",
        memory_enabled=True,
        random_retrieval=False,
        device=device,
    )
    res_b["memory_consolidation_duration_sec"] = round(cons_duration, 4)
    results.append(res_b)

    # Condition C: Detective SAM with Randomized Memory Retrieval
    res_c = run_condition_eval(
        model=model,
        batches=val_batches,
        condition_name="C: Detective SAM with Randomized Retrieval",
        memory_enabled=True,
        random_retrieval=True,
        device=device,
    )
    res_c["memory_consolidation_duration_sec"] = round(cons_duration, 4)
    results.append(res_c)

    # 5. Output Summary Table
    print("\n" + "=" * 90)
    print("           DETECTIVE SAM FORENSIC VECTOR MEMORY BENCHMARK REPORT")
    print("=" * 90)
    header = (
        f"{'Condition':<40} | {'IoU':<7} | {'F1':<7} | {'Latency (ms)':<12} | "
        f"{'Retr (ms)':<10} | {'Peak GPU':<9} | {'DB Entries'}"
    )
    print(header)
    print("-" * 90)
    for r in results:
        row = (
            f"{r['condition']:<40} | "
            f"{r['iou']:<7.4f} | "
            f"{r['f1']:<7.4f} | "
            f"{r['latency_ms_per_batch']:<12.2f} | "
            f"{r['retrieval_latency_ms']:<10.3f} | "
            f"{r['peak_gpu_memory_mb']:<9.1f} | "
            f"{r['database_entry_count']}"
        )
        print(row)
    print("=" * 90)

    # Save to JSON
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(device),
                "num_batches": len(val_batches),
                "batch_size": args.batch_size,
                "consolidation_stats": cons_stats,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved benchmark results to '{args.output_json}'.")


if __name__ == "__main__":
    main()
