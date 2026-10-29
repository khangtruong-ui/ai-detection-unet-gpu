"""
Evaluation script to benchmark the training IoU trajectory of the first 300 steps across models:
1. SAM2 Segmenter (Meta SAM2 Small + QLoRA + Mask Adapter + Unfrozen Decoder)
2. Detective SAM (ICLR 2026: Frozen SAM2 + Feature Adapters + Mask Adapter at 512x512 resolution)
3. Senior's IML MaskFormer (RefactorMaskformer with SAM 2.1 Hiera Vision Trunk + Dual Queries)

All models are trained on the exact same 300 batches from COCO-inpainted.
Logs step-by-step IoU and loss, outputs summary comparison table and trends,
and saves complete trajectory data to benchmark_300_steps_results.json.
"""

import json
import logging
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.dataset.loader import create_dataloaders
from sid_unet.losses.combined import CombinedMaskLoss
from sid_unet.models.detective_sam import DetectiveSAM
from sid_unet.models.iml_maskformer import IMLMaskformer
from sid_unet.models.sam2 import SAM2Segmenter
from sid_unet.utils.config import ConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_300_steps")


def compute_iou(preds_binary: torch.Tensor, targets_binary: torch.Tensor) -> float:
    """Compute mean Intersection over Union across a batch."""
    preds_flat = preds_binary.view(preds_binary.size(0), -1)
    targets_flat = targets_binary.view(targets_binary.size(0), -1)
    intersection = (preds_flat & targets_flat).sum(dim=1).float()
    union = (preds_flat | targets_flat).sum(dim=1).float()
    ious = torch.where(union == 0, torch.ones_like(intersection), intersection / torch.clamp(union, min=1.0))
    return float(ious.mean().item())


def load_benchmark_batches(num_batches: int = 300, batch_size: int = 2, image_size: int = 256):
    """Pre-load fixed benchmark batches from COCO-inpainted for identical comparison."""
    logger.info(f"Loading {num_batches} fixed batches from COCO-inpainted (batch_size={batch_size}, size={image_size})...")
    cfg = ConfigDict({
        "project": {"seed": 42},
        "data": {
            "dataset_name": "KhangTruong/COCO-inpainted",
            "streaming": True,
            "image_size": [image_size, image_size],
            "batch_size": batch_size,
            "num_workers": 0,
            "train_split": "train",
            "val_split": "validation",
            "train_samples_per_epoch": num_batches * batch_size,
            "val_samples": 4,
            "augmentations": {
                "horizontal_flip": 0.0,
                "vertical_flip": 0.0,
                "random_rotate90": 0.0,
                "color_jitter": 0.0,
            },
        }
    })
    train_loader, _ = create_dataloaders(cfg)
    batches = []
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        batches.append({
            "image": batch["image"].clone(),
            "mask": batch["mask"].clone(),
        })
    logger.info(f"Successfully cached {len(batches)} batches in memory.")
    return batches


def run_300_step_training(
    model_name: str,
    model: nn.Module,
    batches: list,
    lr: float,
    device: torch.device,
    use_amp: bool = True,
):
    """Train a model for 300 steps and record step-by-step metrics."""
    logger.info(f"Starting 300-step training for [{model_name}] on {device} (lr={lr}, amp={use_amp})...")
    model = model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"[{model_name}] Trainable params: {param_count:,} / Total params: {total_param_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    loss_fn = CombinedMaskLoss(bce_weight=0.5, dice_weight=0.5).to(device)

    records = []
    start_time = time.time()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = use_amp and (device.type == "cuda")

    for step, batch in enumerate(batches, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=torch.float16):
            outputs = model(images)
            mask_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss, loss_metrics = loss_fn(mask_logits, masks)

        with torch.no_grad():
            preds = (torch.sigmoid(mask_logits.float()) >= 0.5)
            targets = (masks >= 0.5)
            step_iou = compute_iou(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        rec = {
            "step": step,
            "loss": float(loss.item()),
            "iou": step_iou,
        }
        records.append(rec)

        if step in (1, 10, 25, 50, 75, 100, 150, 200, 250, 300):
            elapsed = time.time() - start_time
            logger.info(f"[{model_name}] Step {step:3d}/300 | Loss: {loss.item():.4f} | IoU: {step_iou:.4f} | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    logger.info(f"[{model_name}] 300 steps completed in {total_time:.2f}s ({total_time / len(batches):.3f}s/step).")

    # Segment metrics
    iou_1_50 = sum(r["iou"] for r in records[:50]) / 50
    iou_51_100 = sum(r["iou"] for r in records[50:100]) / 50
    iou_101_200 = sum(r["iou"] for r in records[100:200]) / 100
    iou_201_300 = sum(r["iou"] for r in records[200:]) / len(records[200:])

    return records, {
        "model_name": model_name,
        "trainable_params": param_count,
        "total_params": total_param_count,
        "total_time_s": total_time,
        "step_1_iou": records[0]["iou"],
        "step_10_iou": records[9]["iou"],
        "step_50_iou": records[49]["iou"],
        "step_100_iou": records[99]["iou"],
        "step_200_iou": records[199]["iou"],
        "step_300_iou": records[299]["iou"],
        "max_iou": max(r["iou"] for r in records),
        "mean_iou_1_50": iou_1_50,
        "mean_iou_51_100": iou_51_100,
        "mean_iou_101_200": iou_101_200,
        "mean_iou_201_300": iou_201_300,
        "mean_final_20_iou": sum(r["iou"] for r in records[-20:]) / 20,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load 300 identical batches
    batches = load_benchmark_batches(num_batches=300, batch_size=2, image_size=256)

    results = {}
    summary_table = []

    # 1. Model 1: SAM2 Baseline (SAM2 Small + LoRA + Unfrozen Decoder + Balanced Adapter)
    logger.info("\n" + "=" * 70 + "\n--- Instantiating Method 1: SAM2 Small (LoRA + Unfrozen Decoder) ---\n" + "=" * 70)
    sam2_model = SAM2Segmenter(
        pretrained_model_name_or_path="facebook/sam2-hiera-small",
        load_in_4bit=False,
        lora_r=8,
        lora_alpha=16,
        use_mask_adapter=True,
        freeze_decoder=False,
        reinit_decoder=False,
        target_size=(1024, 1024),
        device=device,
    )
    rec1, summary1 = run_300_step_training("SAM2 Small (Unfrozen Decoder)", sam2_model, batches, lr=2e-4, device=device)
    results["sam2"] = {"records": rec1, "summary": summary1}
    summary_table.append(summary1)
    del sam2_model
    torch.cuda.empty_cache()

    # 2. Model 2: Detective SAM (ICLR 2026 at 512x512 resolution)
    logger.info("\n" + "=" * 70 + "\n--- Instantiating Method 2: Detective SAM (ICLR 2026 @ 512x512) ---\n" + "=" * 70)
    detective_sam = DetectiveSAM(
        variant="small",
        layer_width=64,
        noise_intensity=0.75,
        perturbation_type="blur_noise",
        transformer_downscaling=16,
        transformer_layers=1,
        dropout=0.15,
        target_size=(512, 512),
        freeze_sam2=True,
        freeze_decoder=True,
        device=device,
    )
    rec2, summary2 = run_300_step_training("Detective SAM (512x512)", detective_sam, batches, lr=1e-3, device=device)
    results["detective_sam"] = {"records": rec2, "summary": summary2}
    summary_table.append(summary2)
    del detective_sam
    torch.cuda.empty_cache()

    # 3. Model 3: Senior's IML MaskFormer (RefactorMaskformer)
    logger.info("\n" + "=" * 70 + "\n--- Instantiating Method 3: Senior's IML MaskFormer (RefactorMaskformer) ---\n" + "=" * 70)
    iml_model = IMLMaskformer(
        variant="small",
        embed_dim=256,
        num_queries=100,
        pixel_num_layers=4,
        decoder_num_layers=6,
        target_size=(384, 384),
        device=device,
    )
    rec3, summary3 = run_300_step_training("Senior's IML MaskFormer", iml_model, batches, lr=1e-4, device=device)
    results["iml"] = {"records": rec3, "summary": summary3}
    summary_table.append(summary3)
    del iml_model
    torch.cuda.empty_cache()

    # Save complete benchmark results
    out_path = "benchmark_300_steps_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved complete benchmark results to '{out_path}'.")

    # Print summary
    print("\n" + "=" * 105)
    print("300-STEP TRAINING TRAJECTORY BENCHMARK COMPARISON")
    print("=" * 105)
    header = f"{'Model':<28} | {'Params':<16} | {'Step 1':<8} | {'Step 50':<8} | {'Step 100':<8} | {'Step 200':<8} | {'Step 300':<8} | {'Peak IoU':<8}"
    print(header)
    print("-" * 105)
    for s in summary_table:
        params_str = f"{s['trainable_params']/1e6:.2f}M / {s['total_params']/1e6:.1f}M"
        line = f"{s['model_name']:<28} | {params_str:<16} | {s['step_1_iou']:<8.4f} | {s['step_50_iou']:<8.4f} | {s['step_100_iou']:<8.4f} | {s['step_200_iou']:<8.4f} | {s['step_300_iou']:<8.4f} | {s['max_iou']:<8.4f}"
        print(line)
    print("=" * 105)


if __name__ == "__main__":
    main()
