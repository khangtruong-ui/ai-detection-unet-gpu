"""
Plotting and graphing utilities for SID-UNet training curves and metrics visualization.
Generates publication-quality figures (PNG, JPG, PDF) for loss curves, segmentation metrics,
classification metrics, learning rate schedules, and multi-experiment comparisons without TensorBoard.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Union

# Use headless backend for server/Docker environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
