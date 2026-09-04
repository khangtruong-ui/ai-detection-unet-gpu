"""
CLI and engine for visualizing model predictions on random dataset samples across multiple checkpoints.
Usage:
    sid-illu --model-ckpts ckpt1.pt ckpt2.pt --dataset-configs configs1.yaml configs2.yaml
    sid-illu --model-ckpts "outputs/*/checkpoints/checkpoint_best.pt" --dataset-configs configs/cross-eval/*.yaml --samples 5
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import torch
from torchvision import transforms

from sid_unet.dataset.loader import create_eval_dataloader
from sid_unet.metrics.segmentation import compute_binary_metrics
from sid_unet.models.sam3_refiner import get_sam_refiner
from sid_unet.models.unet import UNet
from sid_unet.postprocessing import get_postprocessor_from_config
from sid_unet.utils.config import load_config, apply_overrides, ConfigDict
from sid_unet.utils.logger import setup_logger
from sid_unet.utils.plotting import create_mask_overlay, create_error_map, _to_rgb_array, _to_mask_2d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Illustrate predictions on random dataset samples across multiple model checkpoints."
    )
    parser.add_argument(
        "--model-ckpts",
        "--model_ckpts",
        "--checkpoints",
        "--checkpoint",
        nargs="+",
        dest="model_ckpts",
        required=True,
        help="Path(s) to model checkpoint .pt file(s) or glob pattern(s).",
    )
    parser.add_argument(
        "--dataset-configs",
        "--dataset_configs",
        "--configs",
        "--config",
        nargs="+",
        dest="dataset_configs",
        required=True,
        help="Path(s) to dataset YAML configuration file(s) or glob pattern(s).",
    )
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        "--samples",
        type=int,
        default=4,
        dest="num_samples",
        help="Number of random samples to illustrate per dataset (default: 4).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to sample from ('test', 'validation', 'train'). Defaults to config test_split/val_split.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=str,
        default="outputs/illustrations",
        dest="output_dir",
        help="Directory to save generated visual illustrations and summary report (default: outputs/illustrations).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for predicted mask probabilities (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42).",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default=None,
        help="Optional SAM model for mask refinement (e.g. 'facebook/sam3').",
    )
    parser.add_argument(
        "--post-process",
        "--postprocess",
        dest="post_process",
        action="store_true",
        default=True,
        help="Enable post-processing for predicted masks (default: True).",
    )
    parser.add_argument(
        "--no-post-process",
        "--no-postprocess",
        dest="post_process",
        action="store_false",
        help="Disable post-processing.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides applied to dataset loading (e.g. data.streaming=false).",
    )
    return parser.parse_args()


def expand_patterns(patterns: List[str], extensions: Tuple[str, ...]) -> List[str]:
    """Expand list of file paths or glob patterns into sorted unique paths."""
    resolved = []
    for pat in patterns:
        if any(c in pat for c in ["*", "?", "["]):
            matches = glob.glob(pat, recursive=True)
            resolved.extend([m for m in matches if os.path.isfile(m) and m.endswith(extensions)])
        elif os.path.isdir(pat):
            for ext in extensions:
                resolved.extend(glob.glob(os.path.join(pat, f"**/*{ext}"), recursive=True))
        elif os.path.isfile(pat):
            resolved.append(pat)
        else:
            # Check if it was a relative path or direct file
            if os.path.exists(pat):
                resolved.append(pat)
            else:
                raise FileNotFoundError(f"Path not found: '{pat}'")

    seen = set()
    unique = []
    for p in resolved:
        abs_p = os.path.abspath(p)
        if abs_p not in seen:
            seen.add(abs_p)
            unique.append(p)
    return unique


def extract_random_dataset_samples(
    config: Any,
    split: Optional[str] = None,
    num_samples: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Extract a list of random (image, gt_mask, label, id) samples from the dataset."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_split = split or config.data.get("test_split", config.data.get("val_split", "test"))

    # Create evaluation dataloader with a small batch size
    # We fetch a buffer of samples to randomly pick from
    buffer_size = max(50, num_samples * 10)
    eval_loader, resolved_split = create_eval_dataloader(
        config=config,
        split=target_split,
        samples_override=buffer_size,
    )

    collected_samples: List[Dict[str, Any]] = []
    for batch in eval_loader:
        images = batch["image"]
        masks = batch["mask"]
        labels = batch.get("label")
        img_ids = batch.get("image_id", [f"img_{i}" for i in range(len(images))])

        for i in range(len(images)):
            collected_samples.append({
                "image": images[i],
                "gt_mask": masks[i],
                "label": int(labels[i]) if labels is not None else None,
                "image_id": str(img_ids[i]),
            })
            if len(collected_samples) >= buffer_size:
                break
        if len(collected_samples) >= buffer_size:
            break

    if not collected_samples:
        return []

    # Pick random samples
    if len(collected_samples) <= num_samples:
        return collected_samples

    return random.sample(collected_samples, num_samples)


def plot_multi_model_comparison_grid(
    samples_data: List[Dict[str, Any]],
    model_names: List[str],
    output_path: str,
    dataset_name: str,
    dpi: int = 200,
) -> str:
    """
    Generate a comparative side-by-side visualization grid across multiple models.
    Columns:
      - Input Image
      - Ground Truth Mask
      - For each model:
          - Model Mask Prediction
          - Overlay (mask on original image)
          - Error Map (TP=Green, FP=Red, FN=Blue)
    """
    if not samples_data or not model_names:
        return ""

    num_samples = len(samples_data)
    num_models = len(model_names)

    # Columns: [Image, GT] + [Model_i Pred, Model_i Overlay, Model_i Error]*num_models
    cols_per_model = 3
    total_cols = 2 + (num_models * cols_per_model)

    fig_w = total_cols * 2.8
    fig_h = max(3.5, num_samples * 2.6) + 1.0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(num_samples, total_cols, figsize=(fig_w, fig_h), dpi=dpi)

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        f"Multi-Model Tampering Detection Comparison on Dataset: {dataset_name}",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    for r_idx, s in enumerate(samples_data):
        img_rgb = _to_rgb_array(s["image"])
        gt_m = _to_mask_2d(s["gt_mask"])

        # Col 0: Original Image
        ax0 = axes[r_idx, 0]
        ax0.imshow(img_rgb)
        if r_idx == 0:
            ax0.set_title("Input Image", fontsize=10, fontweight="bold")
        lbl_str = f"Label: {s.get('label')}" if s.get("label") is not None else ""
        ax0.set_xlabel(f"{s.get('image_id', '')} {lbl_str}".strip(), fontsize=8)
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Col 1: Ground Truth
        ax1 = axes[r_idx, 1]
        ax1.imshow(gt_m, cmap="gray", vmin=0, vmax=1)
        if r_idx == 0:
            ax1.set_title("Ground Truth", fontsize=10, fontweight="bold")
        gt_area = float((gt_m > 0.5).sum()) / max(1, (gt_m.shape[0] * gt_m.shape[1]))
        ax1.set_xlabel(f"Tampered: {gt_area:.1%}", fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Model Columns
        c_offset = 2
        for m_idx, m_name in enumerate(model_names):
            pred_m = _to_mask_2d(s["models"][m_name]["pred_mask"])
            iou_score = s["models"][m_name].get("iou", 0.0)
            dice_score = s["models"][m_name].get("dice", 0.0)

            overlay = create_mask_overlay(img_rgb, pred_m)
            error_map = create_error_map(pred_m, gt_m)

            # Prediction Mask
            ax_pred = axes[r_idx, c_offset]
            ax_pred.imshow(pred_m, cmap="gray", vmin=0, vmax=1)
            if r_idx == 0:
                ax_pred.set_title(f"{m_name}\nPred Mask", fontsize=9, fontweight="bold")
            ax_pred.set_xlabel(f"IoU: {iou_score:.3f}", fontsize=8)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            # Overlay
            ax_ov = axes[r_idx, c_offset + 1]
            ax_ov.imshow(overlay)
            if r_idx == 0:
                ax_ov.set_title(f"{m_name}\nOverlay", fontsize=9, fontweight="bold")
            ax_ov.set_xlabel(f"Dice: {dice_score:.3f}", fontsize=8)
            ax_ov.set_xticks([])
            ax_ov.set_yticks([])

            # Error Map
            ax_err = axes[r_idx, c_offset + 2]
            ax_err.imshow(error_map)
            if r_idx == 0:
                ax_err.set_title(f"{m_name}\nError Map (TP/FP/FN)", fontsize=8, fontweight="bold")
            ax_err.set_xticks([])
            ax_err.set_yticks([])

            c_offset += cols_per_model

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_illustration(
    model_ckpts: List[str],
    dataset_configs: List[str],
    num_samples: int = 4,
    split: Optional[str] = None,
    output_dir: str = "outputs/illustrations",
    threshold: float = 0.5,
    seed: int = 42,
    segment: Optional[str] = None,
    post_process: bool = True,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute multi-model illustration pipeline across dataset configs."""
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(
        name="SID_Illu",
        log_file=os.path.join(output_dir, "illustration.log"),
    )

    logger.info("=" * 70)
    logger.info("🎨 SID-Illu: Random Dataset Sample Model Prediction Visualizer")
    logger.info(f"   Model Checkpoints ({len(model_ckpts)}): {model_ckpts}")
    logger.info(f"   Dataset Configs ({len(dataset_configs)}): {dataset_configs}")
    logger.info(f"   Samples per Dataset: {num_samples}")
    logger.info(f"   Output Directory: {output_dir}")
    logger.info("=" * 70)

    # 1. Load models
    models_dict: Dict[str, Tuple[torch.nn.Module, Any]] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference Device: {device}")

    # SAM refiner if requested
    sam_refiner = None
    if segment:
        logger.info(f"Loading SAM refiner model: {segment}")
        sam_refiner = get_sam_refiner(segment, device=device, threshold=threshold)

    for ckpt_path in model_ckpts:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        # If duplicated ckpt_name, append parent dir
        if ckpt_name in models_dict:
            parent_name = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
            ckpt_name = f"{parent_name}_{ckpt_name}"

        logger.info(f"Loading model checkpoint '{ckpt_name}' from: {ckpt_path}")
        model, cfg = UNet.from_checkpoint(ckpt_path, device=device, return_config=True)
        model.eval()
        models_dict[ckpt_name] = (model, cfg)

    # 2. Iterate through each dataset config
    summary_report_lines = [
        "# 🎨 SID-Illu Qualitative Visual Report",
        "",
        f"- **Total Checkpoints Evaluated**: {len(model_ckpts)}",
        f"- **Total Datasets Illustrated**: {len(dataset_configs)}",
        f"- **Samples Per Dataset**: {num_samples}",
        f"- **Post-Processing**: {post_process}",
        f"- **SAM Refinement**: {segment if segment else 'Disabled'}",
        "",
        "---",
        "",
    ]

    all_dataset_results: Dict[str, Any] = {}

    for cfg_idx, cfg_path in enumerate(dataset_configs, 1):
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        dataset_cfg = load_config(cfg_path, overrides=overrides or [])
        ds_name = dataset_cfg.data.get("dataset_name", cfg_name)

        logger.info(f"\n[{cfg_idx}/{len(dataset_configs)}] Extracting {num_samples} random samples from '{ds_name}'...")
        samples = extract_random_dataset_samples(
            config=dataset_cfg,
            split=split,
            num_samples=num_samples,
            seed=seed + cfg_idx,
        )

        if not samples:
            logger.warning(f"No samples extracted for dataset '{ds_name}'. Skipping...")
            continue

        logger.info(f"Loaded {len(samples)} samples. Running model predictions...")

        # Initialize postprocessor
        postprocessor = get_postprocessor_from_config(dataset_cfg, threshold=threshold, override_enabled=post_process)

        # Run inference for each sample across each model
        samples_with_predictions: List[Dict[str, Any]] = []

        for s in samples:
            img_tensor = s["image"]
            gt_mask_tensor = s["gt_mask"]
            if img_tensor.ndim == 3:
                img_batch = img_tensor.unsqueeze(0).to(device)
            else:
                img_batch = img_tensor.to(device)

            gt_mask_np = gt_mask_tensor.squeeze().cpu().numpy()

            sample_entry = {
                "image": img_tensor.cpu(),
                "gt_mask": gt_mask_tensor.cpu(),
                "label": s.get("label"),
                "image_id": s.get("image_id", "sample"),
                "models": {},
            }

            for m_name, (model, m_cfg) in models_dict.items():
                with torch.no_grad():
                    out = model(img_batch)
                    mask_logits = out[0] if isinstance(out, tuple) else out
                    prob_map = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
                    bin_mask = (prob_map >= threshold).astype(np.float32)

                # Post-processing
                if postprocessor and postprocessor.enabled:
                    proc_res = postprocessor.process_single(bin_mask)
                    bin_mask = proc_res[0] if isinstance(proc_res, tuple) else proc_res

                # SAM Refinement
                if sam_refiner is not None:
                    # Input image array in uint8 [0..255]
                    img_np = _to_rgb_array(img_tensor)
                    ref_res = sam_refiner.refine_single_sample(img_np, bin_mask)
                    bin_mask = ref_res[0] if isinstance(ref_res, tuple) else ref_res

                # Compute sample metrics
                sample_metrics = compute_binary_metrics(bin_mask, gt_mask_np, threshold=threshold)
                iou = sample_metrics.get("iou", 0.0)
                dice = sample_metrics.get("dice", sample_metrics.get("f1", 0.0))

                sample_entry["models"][m_name] = {
                    "pred_mask": bin_mask,
                    "prob_map": prob_map,
                    "iou": iou,
                    "dice": dice,
                }

            samples_with_predictions.append(sample_entry)

        # 3. Generate Visual Grid Figure
        fig_name = f"illustration_{cfg_name}.png"
        fig_path = os.path.join(output_dir, fig_name)
        plot_multi_model_comparison_grid(
            samples_data=samples_with_predictions,
            model_names=list(models_dict.keys()),
            output_path=fig_path,
            dataset_name=ds_name,
        )
        logger.info(f"Saved visual comparison grid: {fig_path}")

        # Tabulate sample metrics
        table_rows = []
        for s_idx, s in enumerate(samples_with_predictions, 1):
            for m_name in models_dict.keys():
                m_data = s["models"][m_name]
                table_rows.append([
                    f"Sample {s_idx} ({s['image_id']})",
                    m_name,
                    f"{m_data['iou']:.4f}",
                    f"{m_data['dice']:.4f}",
                    str(s.get("label", "-")),
                ])

        table_md = tabulate(
            table_rows,
            headers=["Sample", "Model Checkpoint", "IoU", "Dice / F1", "Class Label"],
            tablefmt="github",
        )

        summary_report_lines.extend([
            f"## Dataset: {ds_name} (`{cfg_name}`)",
            "",
            f"![Comparison Grid]({os.path.basename(fig_path)})",
            "",
            "### Sample Quantitative Metrics",
            "",
            table_md,
            "",
            "---",
            "",
        ])

        all_dataset_results[cfg_name] = {
            "dataset_name": ds_name,
            "figure_path": fig_path,
            "samples": samples_with_predictions,
        }

    # Write summary report markdown
    report_md_path = os.path.join(output_dir, "illustration_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_report_lines))

    print("\n" + "=" * 70)
    print("🎨 ALL ILLUSTRATIONS COMPLETED")
    print("=" * 70)
    print(f"Summary Report: {report_md_path}")
    print(f"Output Directory: {output_dir}\n")

    return {
        "output_dir": output_dir,
        "report_path": report_md_path,
        "dataset_results": all_dataset_results,
    }


def main():
    args = parse_args()
    raw_ckpts = args.model_ckpts if isinstance(args.model_ckpts, list) else [args.model_ckpts]
    ckpt_paths = expand_patterns(raw_ckpts, (".pt", ".pth"))

    raw_cfgs = args.dataset_configs if isinstance(args.dataset_configs, list) else [args.dataset_configs]
    cfg_paths = expand_patterns(raw_cfgs, (".yaml", ".yml"))

    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found matching: {raw_ckpts}")
    if not cfg_paths:
        raise FileNotFoundError(f"No dataset configuration files found matching: {raw_cfgs}")

    results = run_illustration(
        model_ckpts=ckpt_paths,
        dataset_configs=cfg_paths,
        num_samples=args.num_samples,
        split=args.split,
        output_dir=args.output_dir,
        threshold=args.threshold,
        seed=args.seed,
        segment=args.segment,
        post_process=args.post_process,
        overrides=args.override,
    )
    return results


def cli_main():
    import sys
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli_main()
