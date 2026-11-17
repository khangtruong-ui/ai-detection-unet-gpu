"""
Plotting and graphing utilities for SID-UNet training curves and metrics visualization.
Generates publication-quality figures (PNG, JPG, PDF) for loss curves, segmentation metrics,
classification metrics, learning rate schedules, and multi-experiment comparisons without TensorBoard.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Use headless backend for server/Docker environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _normalize_history(history: Union[List[Dict[str, Any]], Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
    """Normalize history from either a list of per-epoch dicts or a dict of metric lists."""
    if isinstance(history, dict):
        return history

    if not history:
        return {}

    normalized: Dict[str, List[Any]] = {}
    for entry in history:
        seen_keys = set()
        # 1. First process top-level numeric fields
        for k, v in entry.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                normalized.setdefault(k, []).append(v)
                seen_keys.add(k)

        # 2. Check train_metrics & val_metrics sub-dictionaries for any additional metrics
        for sub_dict_name, prefix in [("train_metrics", "train_"), ("val_metrics", "val_")]:
            sub_dict = entry.get(sub_dict_name)
            if isinstance(sub_dict, dict):
                for sub_k, sub_v in sub_dict.items():
                    if isinstance(sub_v, (int, float)) and not isinstance(sub_v, bool):
                        full_k = sub_k if sub_k.startswith(prefix) else f"{prefix}{sub_k}"
                        if full_k not in seen_keys:
                            normalized.setdefault(full_k, []).append(sub_v)
                            seen_keys.add(full_k)

    return normalized


def plot_training_curves(
    history: Union[List[Dict[str, Any]], Dict[str, List[Any]]],
    output_path: str,
    title_suffix: str = "",
    formats: Optional[List[str]] = None,
    dpi: int = 300,
) -> List[str]:
    """
    Generate and save a 4-panel publication-quality training curve figure.

    Panels:
        1. Training & Validation Loss (Total Loss & Mask Loss)
        2. Validation Segmentation Metrics (IoU, Dice/F1, Pixel Accuracy)
        3. Auxiliary & Detection Metrics (Precision, Recall, Classification Accuracy)
        4. Learning Rate Schedule

    Args:
        history: List of per-epoch metric dictionaries or dict of metric arrays.
        output_path: Base path to save the generated plot (e.g. 'reports/training_curves.png').
        title_suffix: Optional text appended to the figure main title.
        formats: Optional list of additional formats to save (e.g. ['png', 'pdf', 'jpg']).
        dpi: Dots per inch for raster export.

    Returns:
        List of saved plot filepaths.
    """
    data = _normalize_history(history)
    if not data or "epoch" not in data or len(data["epoch"]) == 0:
        return []

    epochs = data["epoch"]
    num_epochs = len(epochs)

    # Styling setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=dpi)
    fig.suptitle(
        f"SID-UNet Training Performance Curves {title_suffix}".strip(),
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    x = np.array(epochs)
    marker = "o" if num_epochs <= 25 else None
    markersize = 5

    # -------------------------------------------------------------
    # Panel 1: Loss Curves
    # -------------------------------------------------------------
    ax_loss = axes[0, 0]
    has_loss = False

    # Train total loss
    train_loss = data.get("train_loss", data.get("total_loss"))
    if train_loss and len(train_loss) == num_epochs:
        ax_loss.plot(x, train_loss, label="Train Total Loss", color="#1f77b4", linewidth=2.2, marker=marker, markersize=markersize)
        has_loss = True

    # Val total loss
    val_loss = data.get("val_loss", data.get("val_total_loss"))
    if val_loss and len(val_loss) == num_epochs:
        ax_loss.plot(x, val_loss, label="Val Total Loss", color="#ff7f0e", linewidth=2.2, linestyle="--", marker=marker, markersize=markersize)
        has_loss = True
        # Mark minimum validation loss
        min_idx = int(np.argmin(val_loss))
        ax_loss.scatter([x[min_idx]], [val_loss[min_idx]], color="#d62728", s=100, zorder=5, marker="*", label=f"Min Val Loss ({val_loss[min_idx]:.4f} @ Ep {x[min_idx]})")

    # Train & Val Mask Loss if distinct
    train_mask_loss = data.get("train_mask_loss", data.get("mask_loss"))
    val_mask_loss = data.get("val_mask_loss")
    if train_mask_loss and len(train_mask_loss) == num_epochs and train_loss and train_mask_loss != train_loss:
        ax_loss.plot(x, train_mask_loss, label="Train Mask Loss", color="#aec7e8", linewidth=1.5, linestyle=":")
    if val_mask_loss and len(val_mask_loss) == num_epochs and val_loss and val_mask_loss != val_loss:
        ax_loss.plot(x, val_mask_loss, label="Val Mask Loss", color="#ffbb78", linewidth=1.5, linestyle=":")

    ax_loss.set_title("Training & Validation Loss", fontsize=13, fontweight="semibold")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    if has_loss:
        ax_loss.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    if num_epochs == 1:
        ax_loss.set_xlim([0.5, 1.5])

    # -------------------------------------------------------------
    # Panel 2: Validation Segmentation Metrics
    # -------------------------------------------------------------
    ax_seg = axes[0, 1]
    has_seg = False

    val_iou = data.get("val_iou", data.get("iou"))
    if val_iou and len(val_iou) == num_epochs:
        ax_seg.plot(x, val_iou, label="Val Mean IoU", color="#2ca02c", linewidth=2.2, marker=marker, markersize=markersize)
        best_iou_idx = int(np.argmax(val_iou))
        ax_seg.scatter([x[best_iou_idx]], [val_iou[best_iou_idx]], color="#2ca02c", s=110, zorder=5, marker="*", label=f"Best IoU ({val_iou[best_iou_idx]:.4f} @ Ep {x[best_iou_idx]})")
        has_seg = True

    val_dice = data.get("val_dice", data.get("dice"))
    if val_dice and len(val_dice) == num_epochs:
        ax_seg.plot(x, val_dice, label="Val Dice / F1", color="#9467bd", linewidth=2.0, marker=marker, markersize=markersize)
        has_seg = True

    val_pixel_acc = data.get("val_pixel_acc", data.get("pixel_acc"))
    if val_pixel_acc and len(val_pixel_acc) == num_epochs:
        ax_seg.plot(x, val_pixel_acc, label="Val Pixel Accuracy", color="#17becf", linewidth=1.8, linestyle="-.", marker=marker, markersize=markersize)
        has_seg = True

    ax_seg.set_title("Validation Segmentation Metrics", fontsize=13, fontweight="semibold")
    ax_seg.set_xlabel("Epoch", fontsize=11)
    ax_seg.set_ylabel("Score [0.0 - 1.0]", fontsize=11)
    ax_seg.set_ylim([-0.05, 1.05])
    ax_seg.grid(True, linestyle="--", alpha=0.5)
    if has_seg:
        ax_seg.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    if num_epochs == 1:
        ax_seg.set_xlim([0.5, 1.5])

    # -------------------------------------------------------------
    # Panel 3: Classification & Detection Metrics
    # -------------------------------------------------------------
    ax_cls = axes[1, 0]
    has_cls = False

    val_aux_acc = data.get("val_aux_accuracy", data.get("val_accuracy", data.get("aux_accuracy")))
    if val_aux_acc and len(val_aux_acc) == num_epochs:
        ax_cls.plot(x, val_aux_acc, label="Val Aux Class Accuracy", color="#e377c2", linewidth=2.2, marker=marker, markersize=markersize)
        has_cls = True

    val_prec = data.get("val_precision", data.get("precision"))
    if val_prec and len(val_prec) == num_epochs:
        ax_cls.plot(x, val_prec, label="Val Precision", color="#8c564b", linewidth=1.8, linestyle="--", marker=marker, markersize=markersize)
        has_cls = True

    val_rec = data.get("val_recall", data.get("recall"))
    if val_rec and len(val_rec) == num_epochs:
        ax_cls.plot(x, val_rec, label="Val Recall", color="#bcbd22", linewidth=1.8, linestyle=":", marker=marker, markersize=markersize)
        has_cls = True

    val_aux_f1 = data.get("val_aux_f1_macro")
    if val_aux_f1 and len(val_aux_f1) == num_epochs:
        ax_cls.plot(x, val_aux_f1, label="Val Aux F1 (Macro)", color="#7f7f7f", linewidth=1.6, linestyle="-.", marker=marker, markersize=markersize)
        has_cls = True

    # Fallback if no specific auxiliary metrics were logged
    if not has_cls and val_iou and val_dice and len(val_iou) == num_epochs and len(val_dice) == num_epochs:
        ax_cls.plot(x, val_iou, label="Val IoU (Ref)", color="#2ca02c", linestyle="--")
        ax_cls.plot(x, val_dice, label="Val Dice (Ref)", color="#9467bd", linestyle="--")
        has_cls = True

    ax_cls.set_title("Classification & Boundary Detection Metrics", fontsize=13, fontweight="semibold")
    ax_cls.set_xlabel("Epoch", fontsize=11)
    ax_cls.set_ylabel("Score [0.0 - 1.0]", fontsize=11)
    ax_cls.set_ylim([-0.05, 1.05])
    ax_cls.grid(True, linestyle="--", alpha=0.5)
    if has_cls:
        ax_cls.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    if num_epochs == 1:
        ax_cls.set_xlim([0.5, 1.5])

    # -------------------------------------------------------------
    # Panel 4: Learning Rate Schedule
    # -------------------------------------------------------------
    ax_lr = axes[1, 1]
    has_lr = False

    lr_vals = data.get("lr", data.get("learning_rate"))
    if lr_vals and len(lr_vals) == num_epochs:
        ax_lr.plot(x, lr_vals, label="Learning Rate", color="#d62728", linewidth=2.0, marker=marker, markersize=markersize)
        ax_lr.ticklabel_format(style="scientific", scilimits=(0, 0), axis="y")
        has_lr = True

    ax_lr.set_title("Learning Rate Schedule", fontsize=13, fontweight="semibold")
    ax_lr.set_xlabel("Epoch", fontsize=11)
    ax_lr.set_ylabel("Learning Rate", fontsize=11)
    ax_lr.grid(True, linestyle="--", alpha=0.5)
    if has_lr:
        ax_lr.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    if num_epochs == 1:
        ax_lr.set_xlim([0.5, 1.5])


    # Format epoch axis with integer ticks when epoch count is reasonable
    for ax in axes.flat:
        if num_epochs <= 30:
            ax.set_xticks(epochs)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save to disk in primary format and requested secondary formats
    saved_paths: List[str] = []
    base_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(base_dir, exist_ok=True)

    base_root, base_ext = os.path.splitext(output_path)
    primary_ext = base_ext.lstrip(".").lower() if base_ext else "png"

    all_formats: List[str] = [primary_ext]
    if formats:
        for fmt in formats:
            clean_fmt = fmt.lstrip(".").lower()
            if clean_fmt not in all_formats:
                all_formats.append(clean_fmt)

    for fmt in all_formats:
        target_file = f"{base_root}.{fmt}"
        plt.savefig(target_file, dpi=dpi, bbox_inches="tight")
        saved_paths.append(target_file)

    plt.close(fig)
    return saved_paths



def plot_multi_experiment_curves(
    experiment_histories: Dict[str, Union[List[Dict[str, Any]], Dict[str, List[Any]]]],
    output_path: str,
    title: str = "Multi-Experiment Comparative Training Curves",
    dpi: int = 300,
) -> str:
    """
    Generate comparative multi-run overlay curves across multiple experiments.

    Panels:
        1. Validation Loss across all experiments
        2. Validation Mean IoU across all experiments
        3. Validation Dice / F1 across all experiments
        4. Training Loss across all experiments
    """
    if not experiment_histories:
        return ""

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=dpi)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, (exp_name, raw_hist) in enumerate(experiment_histories.items()):
        color = palette[idx % len(palette)]
        data = _normalize_history(raw_hist)
        if not data or "epoch" not in data:
            continue

        x = np.array(data["epoch"])
        marker = "o" if len(x) <= 20 else None

        # 1. Val Loss
        val_loss = data.get("val_loss", data.get("val_total_loss"))
        if val_loss:
            axes[0, 0].plot(x, val_loss, label=exp_name, color=color, linewidth=2.0, marker=marker, markersize=4)

        # 2. Val IoU
        val_iou = data.get("val_iou", data.get("iou"))
        if val_iou:
            axes[0, 1].plot(x, val_iou, label=exp_name, color=color, linewidth=2.0, marker=marker, markersize=4)

        # 3. Val Dice
        val_dice = data.get("val_dice", data.get("dice"))
        if val_dice:
            axes[1, 0].plot(x, val_dice, label=exp_name, color=color, linewidth=2.0, marker=marker, markersize=4)

        # 4. Train Loss
        train_loss = data.get("train_loss", data.get("total_loss"))
        if train_loss:
            axes[1, 1].plot(x, train_loss, label=exp_name, color=color, linewidth=2.0, marker=marker, markersize=4)

    axes[0, 0].set_title("Validation Loss Comparison", fontsize=13, fontweight="semibold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Val Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(fontsize=9, frameon=True)

    axes[0, 1].set_title("Validation Mean IoU Comparison", fontsize=13, fontweight="semibold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Mean IoU")
    axes[0, 1].set_ylim([-0.05, 1.05])
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(fontsize=9, frameon=True)

    axes[1, 0].set_title("Validation Dice / F1 Comparison", fontsize=13, fontweight="semibold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Dice Score")
    axes[1, 0].set_ylim([-0.05, 1.05])
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(fontsize=9, frameon=True)

    axes[1, 1].set_title("Training Loss Comparison", fontsize=13, fontweight="semibold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Train Loss")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_history_data(
    history: Union[List[Dict[str, Any]], Dict[str, List[Any]]],
    output_dir: str,
    prefix: str = "training_history",
) -> Dict[str, str]:
    """
    Save training history to structured JSON and CSV formats.

    Args:
        history: List of per-epoch dicts or dict of metric arrays.
        output_dir: Directory where files should be saved.
        prefix: Base filename without extension.

    Returns:
        Dict mapping 'json' and 'csv' to their respective saved paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}.json")
    csv_path = os.path.join(output_dir, f"{prefix}.csv")

    # 1. Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 2. Save CSV
    if isinstance(history, list) and len(history) > 0:
        # Flatten dictionary keys for CSV
        rows: List[Dict[str, Any]] = []
        all_keys = set()
        for item in history:
            flat_item: Dict[str, Any] = {}
            for k, v in item.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float, str, bool)):
                            col_name = f"{k}_{sub_k}"
                            flat_item[col_name] = sub_v
                            all_keys.add(col_name)
                elif isinstance(v, (int, float, str, bool)):
                    flat_item[k] = v
                    all_keys.add(k)
            rows.append(flat_item)

        fieldnames = ["epoch"] + [k for k in sorted(all_keys) if k != "epoch"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    elif isinstance(history, dict) and len(history) > 0:
        keys = list(history.keys())
        num_rows = len(history[keys[0]])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(num_rows):
                writer.writerow([history[k][i] if i < len(history[k]) else "" for k in keys])

    return {"json": json_path, "csv": csv_path}


def _to_rgb_array(img: Union[torch.Tensor, np.ndarray, Any]) -> np.ndarray:
    """Convert input image tensor, PIL image, or ndarray into RGB uint8 numpy array [H, W, 3]."""
    from PIL import Image

    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float().clone()
        if t.ndim == 4 and t.size(0) == 1:
            t = t.squeeze(0)
        if t.ndim == 3 and t.size(0) == 3:
            # Denormalize assuming ImageNet mean and std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = t * std + mean
            t = torch.clamp(t, 0.0, 1.0)
            arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            return arr
        elif t.ndim == 3 and t.size(0) == 1:
            t = torch.clamp(t.squeeze(0), 0.0, 1.0)
            arr = (t.numpy() * 255.0).astype(np.uint8)
            return np.stack([arr, arr, arr], axis=-1)
        elif t.ndim == 2:
            t = torch.clamp(t, 0.0, 1.0)
            arr = (t.numpy() * 255.0).astype(np.uint8)
            return np.stack([arr, arr, arr], axis=-1)

    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))

    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.max() <= 1.0 and arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr.astype(np.uint8)


def _to_mask_2d(mask: Union[torch.Tensor, np.ndarray, Any], threshold: float = 0.5) -> np.ndarray:
    """Convert input mask tensor or array into 2D float32 binary array [H, W] in {0.0, 1.0}."""
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().float().numpy()
    else:
        arr = np.asarray(mask, dtype=np.float32)

    while arr.ndim > 2:
        arr = arr[0]

    return (arr >= threshold).astype(np.float32)


def create_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Generate an RGB error map highlighting:
      - Green: True Positive (TP)
      - Red: False Positive (FP)
      - Blue: False Negative (FN)
      - Dark Gray: True Negative (TN)
    """
    p = pred_mask > 0.5
    g = gt_mask > 0.5
    h, w = p.shape
    error_rgb = np.full((h, w, 3), 35, dtype=np.uint8)  # Background dark gray

    tp = p & g
    fp = p & (~g)
    fn = (~p) & g

    error_rgb[tp] = [50, 205, 50]    # Lime green
    error_rgb[fp] = [230, 40, 40]    # Vibrant red
    error_rgb[fn] = [30, 144, 255]   # Dodger blue
    return error_rgb


def create_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Semi-transparent red overlay over original image for detected positive regions."""
    h, w, _ = image_rgb.shape
    m_bool = mask > 0.5
    overlay = image_rgb.astype(np.float32).copy()
    red_color = np.array([255, 45, 45], dtype=np.float32)
    overlay[m_bool] = (1.0 - alpha) * overlay[m_bool] + alpha * red_color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_eval_sample_predictions(
    samples: List[Dict[str, Any]],
    output_path: str,
    title: str = "SID-UNet Evaluation Sample Mask Predictions & Overlays",
    max_samples: int = 8,
    dpi: int = 200,
) -> str:
    """
    Generate a publication-grade qualitative visual illustration grid for evaluation samples.

    Shows for each sample:
      1. Original Input Image
      2. Ground Truth Mask
      3. UNet Raw Prediction
      4. [Optional] Post-Processed Mask (if present)
      5. [Optional] SAM Refined Mask (if present)
      6. Final Overlay on Original Image
      7. Error Map (TP=Green, FP=Red, FN=Blue)

    Args:
        samples: List of sample dicts containing:
                 'image', 'gt_mask', 'unet_mask', and optionally 'post_mask', 'sam_mask', 'final_mask', 'label', 'img_id'
        output_path: Path to save the illustration image.
        title: Main title of the figure.
        max_samples: Maximum number of samples to plot (default: 8).
        dpi: Output resolution.

    Returns:
        Path to the saved figure file.
    """
    if not samples:
        return ""

    num_samples = min(len(samples), max_samples)
    plot_samples = samples[:num_samples]

    has_post = any("post_mask" in s and s["post_mask"] is not None for s in plot_samples)
    has_sam = any("sam_mask" in s and s["sam_mask"] is not None for s in plot_samples)

    # Determine columns
    columns = [("Input Image", "image"), ("Ground Truth", "gt_mask"), ("UNet Raw", "unet_mask")]
    if has_post:
        columns.append(("Post-Processed", "post_mask"))
    if has_sam:
        columns.append(("SAM Refined", "sam_mask"))
    columns.append(("Final Overlay", "overlay"))
    columns.append(("Error Map (TP=G, FP=R, FN=B)", "error_map"))

    num_cols = len(columns)
    fig_w = num_cols * 3.2
    fig_h = max(3.5, num_samples * 2.8) + 0.8

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_w, fig_h), dpi=dpi)

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    label_names = {0: "Real", 1: "Synthetic", 2: "Tampered"}

    for r_idx, s in enumerate(plot_samples):
        img_rgb = _to_rgb_array(s["image"])
        gt_m = _to_mask_2d(s.get("gt_mask", np.zeros((img_rgb.shape[0], img_rgb.shape[1]))))
        unet_m = _to_mask_2d(s.get("unet_mask", np.zeros_like(gt_m)))

        final_m = _to_mask_2d(s.get("final_mask", s.get("post_mask", unet_m)))
        overlay_rgb = create_mask_overlay(img_rgb, final_m)
        error_rgb = create_error_map(final_m, gt_m)

        col_data = {
            "image": (img_rgb, None),
            "gt_mask": (gt_m, "gray"),
            "unet_mask": (unet_m, "gray"),
            "post_mask": (_to_mask_2d(s["post_mask"]) if "post_mask" in s and s["post_mask"] is not None else None, "gray"),
            "sam_mask": (_to_mask_2d(s["sam_mask"]) if "sam_mask" in s and s["sam_mask"] is not None else None, "gray"),
            "overlay": (overlay_rgb, None),
            "error_map": (error_rgb, None),
        }

        for c_idx, (col_title, col_key) in enumerate(columns):
            ax = axes[r_idx, c_idx]
            data, cmap = col_data.get(col_key, (None, None))

            if data is not None:
                if cmap:
                    ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
                else:
                    ax.imshow(data)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", color="gray")

            if r_idx == 0:
                ax.set_title(col_title, fontsize=11, fontweight="semibold", pad=6)

            # Add row label on first column
            if c_idx == 0:
                lbl = s.get("label")
                lbl_str = f" [{label_names.get(int(lbl), str(lbl))}]" if lbl is not None else ""
                sample_id = s.get("img_id", f"Sample {r_idx + 1}")
                ax.set_ylabel(f"{sample_id}{lbl_str}", fontsize=9, fontweight="medium")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_eval_ablation_bar_chart(
    ablation_data: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Pipeline Ablation: Mask Refinement & Post-Processing Impact",
    dpi: int = 250,
) -> str:
    """
    Generate a grouped bar chart comparing key metrics across pipeline ablation variants
    (e.g. Raw UNet, +Post-Processing, +SAM Refinement, +SAM & Post-Processing).

    Args:
        ablation_data: Dict mapping variant name to metrics dict containing iou, dice/f1, pixel_acc, etc.
        output_path: Save destination path.
        title: Chart title.
        dpi: Output resolution.

    Returns:
        Path to saved chart file.
    """
    if not ablation_data:
        return ""

    variant_names = list(ablation_data.keys())
    metric_keys = [
        ("Mean IoU", ["iou", "val_iou"]),
        ("Dice / F1", ["f1", "dice", "pixel_f1", "val_f1"]),
        ("Pixel Acc", ["pixel_acc", "val_pixel_acc"]),
        ("Precision", ["precision", "val_precision"]),
        ("Recall", ["recall", "val_recall"]),
    ]

    metric_labels = [m[0] for m in metric_keys]
    num_metrics = len(metric_labels)
    num_variants = len(variant_names)

    palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(max(9, num_metrics * 2.2), 6), dpi=dpi)

    x = np.arange(num_metrics)
    total_width = 0.75
    bar_width = total_width / num_variants

    for v_idx, v_name in enumerate(variant_names):
        v_met = ablation_data[v_name]
        vals = []
        for _, candidate_keys in metric_keys:
            val = None
            for ck in candidate_keys:
                if ck in v_met and isinstance(v_met[ck], (int, float)):
                    val = float(v_met[ck])
                    break
            vals.append(val if val is not None else 0.0)

        offset = (v_idx - num_variants / 2.0 + 0.5) * bar_width
        color = palette[v_idx % len(palette)]
        bars = ax.bar(x + offset, vals, width=bar_width, label=v_name, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        # Annotate score above each bar
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="semibold",
                )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="medium")
    ax.set_ylabel("Metric Score [0.0 - 1.0]", fontsize=11)
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cross_eval_heatmaps(
    matrices: Dict[str, Dict[Tuple[str, str], float]],
    checkpoints: List[str],
    configs: List[str],
    output_dir: str,
    dpi: int = 250,
) -> List[str]:
    """
    Generate publication-quality 2D heatmaps with numerical annotations for cross-evaluation matrices.

    Args:
        matrices: Dict mapping metric name ('iou', 'dice_f1', 'pixel_acc', 'auroc') to
                  dict mapping (ckpt_name, config_name) tuple -> float score.
        checkpoints: List of unique checkpoint names (rows).
        configs: List of unique config names (columns).
        output_dir: Directory where heatmaps will be saved.
        dpi: Dots per inch.

    Returns:
        List of generated heatmap filepaths.
    """
    if not checkpoints or not configs:
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    metric_meta = [
        ("iou", "Mean IoU (Intersection over Union)", "viridis", "cross_eval_iou_heatmap.png"),
        ("dice_f1", "Dice / Pixel F1 Score", "magma", "cross_eval_f1_heatmap.png"),
        ("pixel_acc", "Pixel Accuracy", "plasma", "cross_eval_pixel_acc_heatmap.png"),
        ("auroc", "AUROC (ROC-AUC)", "cividis", "cross_eval_auroc_heatmap.png"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for met_key, met_title, colormap, filename in metric_meta:
        matrix_dict = matrices.get(met_key)
        if not matrix_dict:
            continue

        n_rows = len(checkpoints)
        n_cols = len(configs)
        grid = np.zeros((n_rows, n_cols), dtype=np.float32)

        for r_i, ckpt in enumerate(checkpoints):
            for c_j, cfg in enumerate(configs):
                val = matrix_dict.get((ckpt, cfg))
                if val is None:
                    # Also try string key format 'ckpt:cfg'
                    val = matrix_dict.get(f"{ckpt}:{cfg}")
                grid[r_i, c_j] = float(val) if val is not None and isinstance(val, (int, float)) else 0.0

        fig_w = max(7.0, n_cols * 2.2 + 2.5)
        fig_h = max(5.0, n_rows * 1.5 + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        im = ax.imshow(grid, cmap=colormap, vmin=max(0.0, float(np.min(grid)) - 0.05), vmax=min(1.0, float(np.max(grid)) + 0.05))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        ax.set_title(f"Cross-Evaluation Matrix: {met_title}", fontsize=13, fontweight="bold", pad=14)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=10, fontweight="medium")
        ax.set_yticklabels(checkpoints, fontsize=10, fontweight="medium")

        # Annotate text in each cell
        thresh = (grid.max() + grid.min()) / 2.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = grid[i, j]
                text_color = "black" if val > thresh else "white"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

        ax.grid(False)
        plt.tight_layout()

        out_file = os.path.join(output_dir, filename)
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_file)

    return saved_paths


def plot_detailed_inference_pdf(
    samples_data: List[Dict[str, Any]],
    model_names: List[str],
    output_path: str,
    title: str = "Detailed Multi-Column Forensic Inference Report",
    model_summaries: Optional[List[Dict[str, Any]]] = None,
    rows_per_page: int = 3,
    dpi: int = 200,
) -> str:
    """
    Generate a multi-page, 8-column detailed PDF inference report across samples and models.

    Page 1 renders an executive summary (architecture/weights metadata table, aggregate
    quantitative metrics table, and comparative bar chart). Subsequent pages render an
    8-column per-(sample, model) visual & quantitative breakdown:
      1. Input Image
      2. Ground Truth Mask
      3. Continuous Probability Heatmap [0, 1]
      4. Thresholded Binary Prediction Mask
      5. Auxiliary / Spectral Residual / Post-Processed Map
      6. Red Mask Overlay on Input Image
      7. Pixel Error Map (TP=Green, FP=Red, FN=Blue)
      8. Image-Level Real/Fake Detection & Pixel Probability Distribution
    """
    if not samples_data or not model_names:
        return ""

    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    label_names = {0: "Real", 1: "Synthetic", 2: "Tampered"}

    # Build flat list of (sample_dict, model_name) entries
    entries: List[Tuple[Dict[str, Any], str]] = []
    for s in samples_data:
        for m_name in model_names:
            if m_name in s.get("models", {}):
                entries.append((s, m_name))

    if not entries:
        return ""

    # Compute aggregate stats per model
    agg_rows: List[List[str]] = []
    agg_chart_data: Dict[str, Dict[str, float]] = {}
    for m_name in model_names:
        m_entries = [s["models"][m_name] for s in samples_data if m_name in s.get("models", {})]
        if not m_entries:
            continue
        ious = [float(e.get("iou", 0.0)) for e in m_entries]
        dices = [float(e.get("dice", 0.0)) for e in m_entries]
        paccs = [float(e.get("pixel_acc", 0.0)) for e in m_entries]
        precs = [float(e.get("precision", 0.0)) for e in m_entries]
        recs = [float(e.get("recall", 0.0)) for e in m_entries]
        p_fakes = [float(e["cls_probs"][1]) for e in m_entries if e.get("cls_probs") is not None and len(e["cls_probs"]) >= 2]
        lats = [float(e["inference_ms"]) for e in m_entries if e.get("inference_ms") is not None]

        mean_iou = float(np.mean(ious))
        mean_dice = float(np.mean(dices))
        mean_pacc = float(np.mean(paccs))
        mean_prec = float(np.mean(precs))
        mean_rec = float(np.mean(recs))
        mean_pfake = f"{np.mean(p_fakes):.3f}" if p_fakes else "N/A (spatial)"
        mean_lat = f"{np.mean(lats):.1f} ms" if lats else "—"

        agg_rows.append([
            m_name,
            f"{len(m_entries)}",
            f"{mean_iou:.4f}",
            f"{mean_dice:.4f}",
            f"{mean_pacc:.4f}",
            f"{mean_prec:.4f}",
            f"{mean_rec:.4f}",
            mean_pfake,
            mean_lat,
        ])
        agg_chart_data[m_name] = {
            "Mean IoU": mean_iou,
            "Dice / F1": mean_dice,
            "Pixel Acc": mean_pacc,
            "Precision": mean_prec,
            "Recall": mean_rec,
        }

    # Optional per-domain breakdown (when samples include 'domain_split')
    domain_order: List[str] = []
    for s in samples_data:
        ds_sp = s.get("domain_split")
        if ds_sp and ds_sp not in domain_order:
            domain_order.append(ds_sp)
    domain_rows: List[List[str]] = []
    if domain_order:
        for dom in domain_order:
            dom_samples = [s for s in samples_data if s.get("domain_split") == dom]
            for m_name in model_names:
                d_entries = [(s, s["models"][m_name]) for s in dom_samples if m_name in s.get("models", {})]
                if not d_entries:
                    continue
                d_ious = [float(e.get("iou", 0.0)) for _, e in d_entries]
                d_dices = [float(e.get("dice", 0.0)) for _, e in d_entries]
                d_paccs = [float(e.get("pixel_acc", 0.0)) for _, e in d_entries]
                fake_dices = [
                    float(e.get("dice", 0.0))
                    for s, e in d_entries
                    if float((_to_mask_2d(s["gt_mask"]) > 0.5).mean()) > 0.0 or int(s.get("label", 0)) > 0
                ]
                fake_f1_str = f"{np.mean(fake_dices):.4f}" if fake_dices else "N/A"
                cls_hits = []
                for s, e in d_entries:
                    cp = e.get("cls_probs")
                    if cp is not None and len(cp) >= 2:
                        pred_fake = int(float(cp[1]) >= 0.5)
                        gt_fake = int(float((_to_mask_2d(s["gt_mask"]) > 0.5).mean()) > 0.0 or int(s.get("label", 0)) > 0)
                        cls_hits.append(float(pred_fake == gt_fake))
                cls_acc_str = f"{np.mean(cls_hits):.3f}" if cls_hits else "N/A"
                domain_rows.append([
                    str(dom),
                    m_name,
                    str(len(d_entries)),
                    f"{np.mean(d_ious):.4f}",
                    f"{np.mean(d_dices):.4f}",
                    fake_f1_str,
                    f"{np.mean(d_paccs):.4f}",
                    cls_acc_str,
                ])

    with PdfPages(output_path) as pdf:
        # ------------------------------------------------------------------
        # Page 1: Executive Summary, Weights/Architecture & Aggregate Metrics
        # ------------------------------------------------------------------
        has_domains = bool(domain_rows)
        fig_sum = plt.figure(figsize=(18, 11), dpi=dpi)
        gs = fig_sum.add_gridspec(3, 1, height_ratios=[1.15, 0.95, 1.4], hspace=0.35)
        fig_sum.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

        # Panel A: Model & Pretrained Weights Metadata Table
        ax_meta = fig_sum.add_subplot(gs[0])
        ax_meta.axis("off")
        ax_meta.set_title("1. Integrated Models, Branches & Pretrained Weights Summary", fontsize=11, fontweight="bold", loc="left", pad=6)
        if model_summaries:
            meta_headers = ["Model", "Paper / Repo", "Backbone", "Detection", "Localization", "Weights / Status"]
            meta_rows = [
                [
                    str(ms.get("model", "")),
                    str(ms.get("paper", "")),
                    str(ms.get("backbone", "")),
                    str(ms.get("detection", "")),
                    str(ms.get("localization", "")),
                    str(ms.get("weights", "")),
                ]
                for ms in model_summaries
            ]
        else:
            meta_headers = ["Model", "Evaluated Samples"]
            meta_rows = [[m, str(len(samples_data))] for m in model_names]
        tbl_meta = ax_meta.table(cellText=meta_rows, colLabels=meta_headers, loc="center", cellLoc="left")
        tbl_meta.auto_set_font_size(False)
        tbl_meta.set_fontsize(7.5)
        tbl_meta.auto_set_column_width(col=list(range(len(meta_headers))))
        tbl_meta.scale(1.0, 1.55)
        for (r, c), cell in tbl_meta.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 1:
                cell.set_facecolor("#f7f9fb")

        # Panel B: Aggregate Quantitative Metrics Table
        ax_tbl = fig_sum.add_subplot(gs[1])
        ax_tbl.axis("off")
        ax_tbl.set_title("2. Aggregate Quantitative Inference Metrics", fontsize=11, fontweight="bold", loc="left", pad=6)
        agg_headers = ["Model", "N", "Mean IoU", "Dice / F1", "Pixel Acc", "Precision", "Recall", "Mean P(Fake)", "Avg Latency"]
        tbl_agg = ax_tbl.table(cellText=agg_rows, colLabels=agg_headers, loc="center", cellLoc="center")
        tbl_agg.auto_set_font_size(False)
        tbl_agg.set_fontsize(8.0)
        tbl_agg.auto_set_column_width(col=list(range(len(agg_headers))))
        tbl_agg.scale(1.0, 1.55)
        for (r, c), cell in tbl_agg.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1f4e78")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 1:
                cell.set_facecolor("#f4f7fa")

        # Panel C: Comparative Metric Bar Chart
        ax_bar = fig_sum.add_subplot(gs[2])
        met_names = ["Mean IoU", "Dice / F1", "Pixel Acc", "Precision", "Recall"]
        x_pos = np.arange(len(met_names))
        n_m = max(1, len(agg_chart_data))
        bar_w = 0.72 / n_m
        palette = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
        for m_i, (m_name, m_vals) in enumerate(agg_chart_data.items()):
            vals = [m_vals[k] for k in met_names]
            offset = (m_i - n_m / 2.0 + 0.5) * bar_w
            bars = ax_bar.bar(x_pos + offset, vals, width=bar_w, label=m_name, color=palette[m_i % len(palette)], alpha=0.9, edgecolor="black", linewidth=0.5)
            for b_rect, v in zip(bars, vals):
                ax_bar.annotate(
                    f"{v:.3f}",
                    xy=(b_rect.get_x() + b_rect.get_width() / 2.0, b_rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="semibold",
                )
        ax_bar.set_title("3. Per-Model Quantitative Comparison Across Sampled Images", fontsize=11, fontweight="bold", loc="left")
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(met_names, fontsize=10, fontweight="medium")
        ax_bar.set_ylim([0.0, 1.15])
        ax_bar.set_ylabel("Score [0.0 - 1.0]")
        ax_bar.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax_bar.grid(True, linestyle="--", alpha=0.45, axis="y")

        fig_sum.text(
            0.5, 0.01,
            "Note: Each subsequent page shows an 8-column forensic inspection grid (Input, GT, Prob Heatmap, Binary Pred, Aux/Spectral/Post Map, Overlay, Error Map, Detection & Calibration).",
            ha="center", fontsize=8.5, style="italic", color="#444444",
        )
        pdf.savefig(fig_sum, dpi=dpi, bbox_inches="tight")
        plt.close(fig_sum)

        # ------------------------------------------------------------------
        # Optional Page 2: Domain-Aligned Quantitative Breakdown
        # ------------------------------------------------------------------
        if has_domains:
            fig_dom = plt.figure(figsize=(18, max(7.0, 2.2 + len(domain_rows) * 0.42)), dpi=dpi)
            ax_dom = fig_dom.add_subplot(111)
            ax_dom.axis("off")
            ax_dom.set_title(
                f"{title}\n4. Domain Breakdown (Model 1 In-Domain vs Model 2 In-Domain vs Out-of-Domain Faces)",
                fontsize=12.5,
                fontweight="bold",
                loc="left",
                pad=14,
            )
            dom_headers = ["Domain Split", "Model", "N", "Mean IoU", "Dice / F1", "Tampered F1", "Pixel Acc", "Image Cls ACC"]
            tbl_dom = ax_dom.table(cellText=domain_rows, colLabels=dom_headers, loc="center", cellLoc="center")
            tbl_dom.auto_set_font_size(False)
            tbl_dom.set_fontsize(8.5)
            tbl_dom.auto_set_column_width(col=list(range(len(dom_headers))))
            tbl_dom.scale(1.0, 1.55)
            for (r, c), cell in tbl_dom.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#274e13")
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 1:
                    cell.set_facecolor("#f3f8f1")
            pdf.savefig(fig_dom, dpi=dpi, bbox_inches="tight")
            plt.close(fig_dom)

        # ------------------------------------------------------------------
        # Subsequent Pages: 8-Column Detailed Visual & Quantitative Rows
        # ------------------------------------------------------------------
        col_titles = [
            "1. Input Image",
            "2. Ground Truth",
            "3. Prob Heatmap [0..1]",
            "4. Binary Pred (≥0.5)",
            "5. Aux / Spectral / Post",
            "6. Mask Overlay",
            "7. Error Map\n(G=TP, R=FP, B=FN)",
            "8. Detection Score &\nProb Histogram",
        ]
        num_cols = len(col_titles)

        for start_idx in range(0, len(entries), rows_per_page):
            chunk = entries[start_idx : start_idx + rows_per_page]
            n_rows = len(chunk)
            fig_w = num_cols * 2.85
            fig_h = max(4.2, n_rows * 3.15 + 1.0)
            fig, axes = plt.subplots(n_rows, num_cols, figsize=(fig_w, fig_h), dpi=dpi)
            if n_rows == 1:
                axes = np.expand_dims(axes, axis=0)

            page_num = (3 if has_domains else 2) + (start_idx // rows_per_page)
            fig.suptitle(f"{title} — Detailed Visual Breakdown (Page {page_num})", fontsize=13, fontweight="bold", y=0.995)

            for r_idx, (s, m_name) in enumerate(chunk):
                m_res = s["models"][m_name]
                img_rgb = _to_rgb_array(s["image"])
                gt_m = _to_mask_2d(s["gt_mask"])
                pred_m = _to_mask_2d(m_res["pred_mask"])

                prob_raw = m_res.get("prob_map")
                if prob_raw is not None:
                    prob_m = np.asarray(prob_raw, dtype=np.float32)
                    if prob_m.ndim == 3:
                        prob_m = prob_m.squeeze()
                    prob_m = np.clip(prob_m, 0.0, 1.0)
                else:
                    prob_m = pred_m

                aux_raw = m_res.get("aux_map")
                if aux_raw is not None:
                    aux_m = np.asarray(aux_raw, dtype=np.float32)
                    if aux_m.ndim == 3:
                        aux_m = aux_m.squeeze()
                    aux_m = np.clip(aux_m, 0.0, 1.0)
                    aux_cmap = m_res.get("aux_cmap", "viridis")
                    aux_title = m_res.get("aux_label", "Aux / Spectral Map")
                else:
                    aux_m = pred_m
                    aux_cmap = "gray"
                    aux_title = "Thresholded Mask"

                overlay = create_mask_overlay(img_rgb, pred_m)
                error_map = create_error_map(pred_m, gt_m)

                iou = float(m_res.get("iou", 0.0))
                dice = float(m_res.get("dice", 0.0))
                pacc = float(m_res.get("pixel_acc", 0.0))
                prec = float(m_res.get("precision", 0.0))
                rec = float(m_res.get("recall", 0.0))
                gt_area = float((gt_m > 0.5).mean())
                pred_area = float((pred_m > 0.5).mean())

                lbl = s.get("label")
                lbl_str = label_names.get(int(lbl), str(lbl)) if lbl is not None else ("Tampered" if gt_area > 0 else "Real")
                ds_tag = s.get("dataset", "")
                row_tag = f"{s.get('image_id', 'sample')}\n[{m_name}]"

                # Col 0: Input Image
                ax = axes[r_idx, 0]
                ax.imshow(img_rgb)
                ax.set_ylabel(row_tag, fontsize=8, fontweight="bold")
                ax.set_xlabel(f"{ds_tag} | {img_rgb.shape[1]}x{img_rgb.shape[0]} | GT:{lbl_str}".strip(" |"), fontsize=7.5)

                # Col 1: Ground Truth
                ax = axes[r_idx, 1]
                ax.imshow(gt_m, cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_xlabel(f"GT Tampered: {gt_area:.1%}", fontsize=7.5)

                # Col 2: Prob Heatmap
                ax = axes[r_idx, 2]
                ax.imshow(prob_m, cmap="inferno", vmin=0.0, vmax=1.0)
                ax.set_xlabel(f"min:{prob_m.min():.2f} μ:{prob_m.mean():.2f} max:{prob_m.max():.2f}", fontsize=7.5)

                # Col 3: Binary Pred
                ax = axes[r_idx, 3]
                ax.imshow(pred_m, cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_xlabel(f"IoU:{iou:.3f} | F1:{dice:.3f} | Area:{pred_area:.1%}", fontsize=7.5)

                # Col 4: Aux / Spectral / Post
                ax = axes[r_idx, 4]
                ax.imshow(aux_m, cmap=aux_cmap, vmin=0.0, vmax=1.0)
                aux_extra = m_res.get("aux_metric_str", "")
                ax.set_xlabel(f"{aux_title}" + (f"\n{aux_extra}" if aux_extra else ""), fontsize=7.5)

                # Col 5: Overlay
                ax = axes[r_idx, 5]
                ax.imshow(overlay)
                lat_ms = m_res.get("inference_ms")
                ax.set_xlabel(f"Latency: {lat_ms:.1f} ms" if lat_ms is not None else f"{m_name}", fontsize=7.5)

                # Col 6: Error Map
                ax = axes[r_idx, 6]
                ax.imshow(error_map)
                ax.set_xlabel(f"Acc:{pacc:.3f} P:{prec:.2f} R:{rec:.2f}", fontsize=7.5)

                for c_i in range(7):
                    axes[r_idx, c_i].set_xticks([])
                    axes[r_idx, c_i].set_yticks([])
                    if r_idx == 0:
                        axes[r_idx, c_i].set_title(col_titles[c_i], fontsize=9, fontweight="bold", pad=6)

                # Col 7: Detection Score & Pixel Probability Histogram
                ax7 = axes[r_idx, 7]
                if r_idx == 0:
                    ax7.set_title(col_titles[7], fontsize=9, fontweight="bold", pad=6)
                cls_probs = m_res.get("cls_probs")
                if cls_probs is not None and len(cls_probs) >= 2:
                    p_real = float(cls_probs[0])
                    p_fake = float(1.0 - p_real) if len(cls_probs) == 2 else float(sum(cls_probs[1:]))
                    cls_summary = f"Img Cls: P(Real)={p_real:.2f}, P(Fake)={p_fake:.2f}"
                else:
                    p_fake = float(np.percentile(prob_m, 99))
                    p_real = 1.0 - p_fake
                    cls_summary = f"Spatial p99={p_fake:.2f} (no img head)"

                hist_vals, bin_edges = np.histogram(prob_m.ravel(), bins=20, range=(0.0, 1.0))
                hist_frac = hist_vals / max(1, hist_vals.sum())
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax7.bar(centers, hist_frac, width=0.045, color="#3b75af", alpha=0.85, edgecolor="#1c3b57", linewidth=0.4)
                ax7.axvline(0.5, color="#d62728", linestyle="--", linewidth=1.1, label="thr=0.5")
                # Overlay top bar badge for Real vs Fake prob
                ax7.barh([0.92], [p_real], height=0.08, left=0.0, color="#2ca02c", alpha=0.85, transform=ax7.get_xaxis_transform())
                ax7.barh([0.92], [p_fake], height=0.08, left=p_real, color="#d62728", alpha=0.85, transform=ax7.get_xaxis_transform())
                ax7.set_xlim([0.0, 1.0])
                ax7.set_ylim([0.0, 1.05])
                ax7.tick_params(axis="both", labelsize=7)
                ax7.set_xlabel(cls_summary, fontsize=7.2)

            plt.tight_layout(rect=[0, 0.01, 1, 0.97])
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    return output_path
