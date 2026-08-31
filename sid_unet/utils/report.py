"""
Report generation and formatting utilities for SID-UNet evaluation and benchmarking.
Produces Markdown, JSON, and rich formatted terminal tables for single and multi-experiment runs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from tabulate import tabulate


def extract_key_hyperparameters(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract a flattened, human-readable dictionary of key hyperparameters from config."""
    if not config:
        return {}

    project_cfg = config.get("project", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("training", {})

    # Steps / samples representation
    train_samples = data_cfg.get("train_samples_per_epoch")
    steps_per_epoch = data_cfg.get("steps_per_epoch", train_cfg.get("steps_per_epoch"))
    if steps_per_epoch is not None and int(steps_per_epoch) <= 0:
        steps_repr = "Full Dataset (until depletion)"
    elif steps_per_epoch is not None:
        steps_repr = f"{steps_per_epoch} steps/epoch"
    elif train_samples is not None and int(train_samples) <= 0:
        steps_repr = "Full Dataset (until depletion)"
    elif train_samples is not None:
        steps_repr = f"{train_samples} samples/epoch"
    else:
        steps_repr = "Full Dataset"

    loss_desc = loss_cfg.get("mask_loss_type", "combined")
    if loss_desc == "combined":
        loss_desc = f"Combined (BCE: {loss_cfg.get('bce_weight', 0.5)}, Dice: {loss_cfg.get('dice_weight', 0.5)})"

    return {
        "Project Name": project_cfg.get("name", "N/A"),
        "Dataset": data_cfg.get("dataset_name", "N/A"),
        "Splits": f"Train: '{data_cfg.get('train_split', 'train')}' | Val: '{data_cfg.get('val_split', 'validation')}' | Test: '{data_cfg.get('test_split', 'test')}'",
        "Streaming": data_cfg.get("streaming", True),
        "Image Size": str(data_cfg.get("image_size", [256, 256])),
        "Batch Size": data_cfg.get("batch_size", 16),
        "Epoch Limit / Steps": steps_repr,
        "Model": f"{model_cfg.get('name', 'unet')} (features={model_cfg.get('features')}, bilinear={model_cfg.get('bilinear', True)})",
        "Aux Classifier": f"{model_cfg.get('aux_classifier', True)} (classes={model_cfg.get('num_classes', 3)})",
        "Mask Loss": loss_desc,
        "Aux Loss": f"{loss_cfg.get('aux_loss_type', 'cross_entropy')} (wt={loss_cfg.get('aux_weight', 0.2)})",
        "Optimizer": f"{train_cfg.get('optimizer', 'adamw')} (lr={train_cfg.get('learning_rate', 1e-3)}, wd={train_cfg.get('weight_decay', 1e-4)})",
        "Scheduler": train_cfg.get("scheduler", "cosine"),
        "Epochs": train_cfg.get("epochs", 10),
        "AMP": train_cfg.get("amp", True),
    }


def format_config_table(config: Dict[str, Any], title: str = "Configuration & Hyperparameters") -> str:
    """Format configuration dictionary into a clean tabulated markdown table."""
    params = extract_key_hyperparameters(config)
    if not params:
        return ""
    headers = ["Hyperparameter / Setting", "Configured Value"]
    rows = [[k, str(v)] for k, v in params.items()]
    table_str = tabulate(rows, headers=headers, tablefmt="github")
    return f"### {title}\n\n{table_str}\n"


def format_metrics_table(metrics: Dict[str, Any], title: str = "Evaluation Metrics") -> str:
    """Format dictionary metrics into a clean tabulated ASCII/Markdown table."""
    headers = ["Metric", "Value"]
    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            formatted_val = f"{v:.4f}"
        elif isinstance(v, (int, str)):
            formatted_val = str(v)
        else:
            continue
        # Convert snake_case or prefix to readable title
        metric_name = k.replace("_", " ").title()
        rows.append([metric_name, formatted_val])

    table_str = tabulate(rows, headers=headers, tablefmt="github")
    title_str = f"### {title}\n\n" if title else ""
    return f"{title_str}{table_str}\n"


def format_history_table(history: List[Dict[str, Any]]) -> str:
    """Format epoch-by-epoch training and validation metrics into a clean tabulated table."""
    if not history:
        return ""
    headers = ["Epoch", "Train Loss", "Val Loss", "Val IoU", "Val F1", "Val AUROC", "Val Pixel Acc", "Learning Rate"]
    rows = []
    for h in history:
        ep = h.get("epoch", "-")
        tr_loss = h.get("train_loss", h.get("total_loss"))
        v_loss = h.get("val_loss", h.get("val_total_loss"))
        v_iou = h.get("val_iou", h.get("iou"))
        v_f1 = h.get("val_f1", h.get("val_pixel_f1", h.get("val_dice", h.get("dice"))))
        v_auroc = h.get("val_auroc", h.get("val_pixel_auroc", h.get("auroc")))
        v_pacc = h.get("val_pixel_acc", h.get("pixel_acc"))
        lr = h.get("lr", h.get("learning_rate"))

        rows.append([
            ep,
            f"{tr_loss:.4f}" if isinstance(tr_loss, (int, float)) else "-",
            f"{v_loss:.4f}" if isinstance(v_loss, (int, float)) else "-",
            f"{v_iou:.4f}" if isinstance(v_iou, (int, float)) else "-",
            f"{v_f1:.4f}" if isinstance(v_f1, (int, float)) else "-",
            f"{v_auroc:.4f}" if isinstance(v_auroc, (int, float)) else "-",
            f"{v_pacc:.4f}" if isinstance(v_pacc, (int, float)) else "-",
            f"{lr:.2e}" if isinstance(lr, (int, float)) else "-",
        ])

    table_str = tabulate(rows, headers=headers, tablefmt="github")
    return f"### Epoch-by-Epoch Training & Validation Progression\n\n{table_str}\n"


def generate_evaluation_report(
    overall_metrics: Dict[str, float],
    per_label_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    confusion_matrix: Optional[list[list[int]]] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    report_name: str = "evaluation_report",
    history: Optional[List[Dict[str, Any]]] = None,
    curves_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report with configuration details, overall metrics,
    per-label stats, training curves plot links, and auxiliary classification details. Saves both JSON and Markdown formats.
    """
    label_names = {
        0: "Label 0 (Real / Black Mask)",
        1: "Label 1 (Synthetic / White Mask)",
        2: "Label 2 (Tampered / Partial Mask)",
    }

    report_dict: Dict[str, Any] = {
        "overall_metrics": overall_metrics,
        "per_label_metrics": {},
    }

    if curves_path:
        report_dict["curves_plot_path"] = curves_path

    if history:
        report_dict["training_history"] = history

    if per_label_metrics:
        for lbl, stats in per_label_metrics.items():
            lbl_key = label_names.get(lbl, f"Label {lbl}")
            report_dict["per_label_metrics"][lbl_key] = stats

    if confusion_matrix is not None and len(confusion_matrix) > 0:
        report_dict["auxiliary_classification_confusion_matrix"] = confusion_matrix

    if config:
        report_dict["configuration"] = config
        report_dict["hyperparameters"] = extract_key_hyperparameters(config)

    # Generate Markdown text
    md_lines = ["# SID-UNet Evaluation Report\n"]

    if config:
        md_lines.append(format_config_table(config, title="Run Configuration & Hyperparameters"))

    md_lines.append(format_metrics_table(overall_metrics, title="Overall Segmentation & Classification Metrics"))

    if curves_path:
        rel_curve_name = os.path.basename(curves_path)
        md_lines.append("### Training & Validation Curves\n")
        md_lines.append(f"![Training Curves]({rel_curve_name})\n")

    if history:
        md_lines.append(format_history_table(history))

    if per_label_metrics:
        md_lines.append("### Per-Subset Metrics Breakdown\n")
        per_label_rows = []
        for lbl, stats in per_label_metrics.items():
            name = label_names.get(lbl, f"Label {lbl}")
            row = [
                name,
                f"{stats.get('iou', 0.0):.4f}",
                f"{stats.get('dice', 0.0):.4f}",
                f"{stats.get('pixel_acc', 0.0):.4f}",
                f"{stats.get('samples', 0)}",
            ]
            per_label_rows.append(row)
        per_label_headers = ["Subset", "Mean IoU", "Dice / F1", "Pixel Accuracy", "Sample Count"]
        md_lines.append(tabulate(per_label_rows, headers=per_label_headers, tablefmt="github"))
        md_lines.append("\n")

    if confusion_matrix is not None and len(confusion_matrix) > 0:
        md_lines.append("### Auxiliary Classification Confusion Matrix (Rows: Ground Truth, Cols: Predicted)\n")
        num_classes = len(confusion_matrix)
        default_class_names = ["Class 0 (Real)", "Class 1 (Synthetic)", "Class 2 (Tampered)"]
        cm_headers = ["Actual \\ Pred"] + [
            default_class_names[i] if i < len(default_class_names) else f"Class {i}"
            for i in range(num_classes)
        ]
        cm_rows = []
        for i, row in enumerate(confusion_matrix):
            row_label = default_class_names[i] if i < len(default_class_names) else f"Class {i}"
            cm_rows.append([row_label, *row])
        md_lines.append(tabulate(cm_rows, headers=cm_headers, tablefmt="github"))
        md_lines.append("\n")

    markdown_content = "\n".join(md_lines)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{report_name}.json")
        md_path = os.path.join(output_dir, f"{report_name}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    return {
        "report_dict": report_dict,
        "markdown": markdown_content,
    }



def generate_multi_experiment_report(
    experiment_results: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    report_name: str = "experiment_comparison",
    multi_curves_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate comparison report for multiple experiment runs.
    Presents side-by-side config hyperparameters and validation/test results.
    """
    comparison_data: List[Dict[str, Any]] = []
    summary_rows = []

    for exp in experiment_results:
        exp_name = exp.get("run_name") or exp.get("name", "Unknown Run")
        config = exp.get("config", {})
        metrics = exp.get("final_metrics", {})
        best_epoch = exp.get("best_epoch", "N/A")
        best_score = exp.get("best_score")

        hp = extract_key_hyperparameters(config)

        # Pull key metric scores (supports validation and evaluation metrics)
        val_loss = metrics.get("val_total_loss", metrics.get("val_loss", metrics.get("eval_total_loss")))
        val_iou = metrics.get("val_iou", metrics.get("iou"))
        val_f1 = metrics.get("val_f1", metrics.get("val_pixel_f1", metrics.get("val_dice", metrics.get("f1", metrics.get("pixel_f1", metrics.get("dice"))))))
        val_auroc = metrics.get("val_auroc", metrics.get("val_pixel_auroc", metrics.get("auroc", metrics.get("pixel_auroc"))))
        val_pixel_acc = metrics.get("val_pixel_acc", metrics.get("pixel_acc"))
        val_acc = metrics.get("val_aux_accuracy", metrics.get("val_accuracy", metrics.get("aux_accuracy", metrics.get("accuracy"))))

        row = [
            exp_name,
            hp.get("Model", "UNet"),
            hp.get("Mask Loss", "Combined"),
            f"{config.get('training', {}).get('learning_rate', 'N/A')}",
            f"{config.get('data', {}).get('batch_size', 'N/A')}",
            f"{best_epoch}",
            f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "N/A",
            f"{val_iou:.4f}" if isinstance(val_iou, (int, float)) else "N/A",
            f"{val_f1:.4f}" if isinstance(val_f1, (int, float)) else "N/A",
            f"{val_auroc:.4f}" if isinstance(val_auroc, (int, float)) else "N/A",
            f"{val_pixel_acc:.4f}" if isinstance(val_pixel_acc, (int, float)) else "N/A",
            f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "N/A",
        ]
        summary_rows.append(row)

        comparison_data.append({
            "run_name": exp_name,
            "config_path": exp.get("config_path"),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "hyperparameters": hp,
            "metrics": metrics,
            "report_path": exp.get("report_path"),
            "curves_plot_path": exp.get("curves_plot_path"),
        })

    summary_headers = [
        "Experiment",
        "Model",
        "Loss",
        "LR",
        "Batch Size",
        "Best Epoch",
        "Val Loss",
        "Val IoU",
        "Val F1",
        "Val AUROC",
        "Pixel Acc",
        "Class Acc",
    ]
    summary_table_md = tabulate(summary_rows, headers=summary_headers, tablefmt="github")

    md_lines = [
        "# Multi-Experiment Benchmarking & Comparison Report\n",
        "## Summary Comparison\n",
        summary_table_md,
        "\n",
    ]

    if multi_curves_path:
        rel_curve_name = os.path.basename(multi_curves_path)
        md_lines.append("### Comparative Training & Validation Curves\n")
        md_lines.append(f"![Multi-Experiment Curves]({rel_curve_name})\n\n")

    md_lines.append("---\n\n## Detailed Experiment Breakdown\n")

    for item in comparison_data:
        md_lines.append(f"### Experiment: {item['run_name']}\n")
        if item.get("config_path"):
            md_lines.append(f"- **Config File**: `{item['config_path']}`")
        if item.get("report_path"):
            md_lines.append(f"- **Detailed Report**: `{item['report_path']}`")
        if item.get("curves_plot_path"):
            md_lines.append(f"- **Curves Plot**: `{item['curves_plot_path']}`")
        if item.get("best_score") is not None:
            md_lines.append(f"- **Best Epoch**: {item['best_epoch']} (Score: {item['best_score']:.4f})\n")
        else:
            md_lines.append(f"- **Best Epoch**: {item['best_epoch']}\n")

        # Hyperparameters table
        if item.get("hyperparameters"):
            hp_rows = [[k, str(v)] for k, v in item["hyperparameters"].items()]
            md_lines.append("#### Configuration Parameters\n")
            md_lines.append(tabulate(hp_rows, headers=["Parameter", "Value"], tablefmt="github"))
            md_lines.append("\n")

        # Metrics table
        if item.get("metrics"):
            md_lines.append("#### Final Results\n")
            md_lines.append(format_metrics_table(item["metrics"], title=""))
            md_lines.append("\n")

    markdown_content = "\n".join(md_lines)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{report_name}.json")
        md_path = os.path.join(output_dir, f"{report_name}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    return {
        "comparison_data": comparison_data,
        "markdown": markdown_content,
        "summary_table": summary_table_md,
    }


def generate_checkpoint_cross_eval_report(
    checkpoint_path: str,
    results: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    report_name: str = "cross_evaluation_report",
) -> Dict[str, Any]:
    """
    Generate a cross-evaluation report for a single checkpoint evaluated across multiple configs.
    Saved neighbor to the checkpoint folder.
    """
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    ckpt_name = results[0].get("checkpoint_name", ckpt_stem) if results else ckpt_stem

    summary_rows = []
    for res in results:
        cfg_name = res.get("config_name", "default")
        ds_name = res.get("dataset_name", res.get("config", {}).get("data", {}).get("dataset_name", "N/A"))
        m = res.get("metrics", res.get("overall_metrics", {}))
        split = res.get("eval_split", m.get("eval_split", "test"))
        samples = res.get("total_evaluated_samples", m.get("total_evaluated_samples", 0))
        loss = m.get("eval_total_loss", m.get("total_loss", m.get("loss")))
        iou = m.get("iou", 0.0)
        f1 = m.get("f1", m.get("pixel_f1", m.get("dice", 0.0)))
        auroc = m.get("auroc", m.get("pixel_auroc", 0.0))
        pacc = m.get("pixel_acc", 0.0)
        cacc = m.get("aux_accuracy", m.get("accuracy"))

        summary_rows.append([
            cfg_name,
            ds_name,
            str(split),
            str(samples),
            f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A",
            f"{iou:.4f}" if isinstance(iou, (int, float)) else "N/A",
            f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A",
            f"{auroc:.4f}" if isinstance(auroc, (int, float)) else "N/A",
            f"{pacc:.4f}" if isinstance(pacc, (int, float)) else "N/A",
            f"{cacc:.4f}" if isinstance(cacc, (int, float)) else "N/A",
        ])

    headers = [
        "Config",
        "Dataset",
        "Split",
        "Samples",
        "Loss",
        "IoU",
        "Dice / F1",
        "AUROC",
        "Pixel Acc",
        "Class Acc",
    ]
    summary_table_md = tabulate(summary_rows, headers=headers, tablefmt="github", disable_numparse=True)

    md_lines = [
        f"# Cross-Evaluation Report: {ckpt_name}\n",
        f"- **Checkpoint Path**: `{checkpoint_path}`",
        f"- **Evaluated Configurations**: {len(results)}\n",
        "## Cross-Dataset Performance Summary\n",
        summary_table_md,
        "\n",
    ]

    # Detailed per-config sections
    md_lines.append("---\n\n## Per-Configuration Breakdown\n")
    for res in results:
        cfg_name = res.get("config_name", "default")
        ds_name = res.get("dataset_name", "N/A")
        m = res.get("metrics", res.get("overall_metrics", {}))
        per_label = res.get("per_label_metrics", {})
        cm = res.get("confusion_matrix")

        md_lines.append(f"### Configuration: {cfg_name} (Dataset: {ds_name})\n")
        if res.get("config_path"):
            md_lines.append(f"- **Config File**: `{res['config_path']}`")

        md_lines.append(format_metrics_table(m, title="Evaluation Metrics"))

        if per_label:
            label_names = {0: "Label 0 (Real)", 1: "Label 1 (Synthetic)", 2: "Label 2 (Tampered)"}
            pl_rows = []
            for lbl, stats in per_label.items():
                name = label_names.get(lbl, f"Label {lbl}")
                pl_rows.append([
                    name,
                    f"{stats.get('iou', 0.0):.4f}",
                    f"{stats.get('dice', stats.get('f1', 0.0)):.4f}",
                    f"{stats.get('pixel_acc', 0.0):.4f}",
                    f"{stats.get('samples', 0)}",
                ])
            md_lines.append("#### Per-Subset Breakdown\n")
            md_lines.append(tabulate(pl_rows, headers=["Subset", "IoU", "Dice / F1", "Pixel Acc", "Samples"], tablefmt="github", disable_numparse=True))
            md_lines.append("\n")

        if cm and len(cm) > 0:
            md_lines.append("#### Confusion Matrix\n")
            cm_headers = ["Actual \\ Pred"] + [f"Class {i}" for i in range(len(cm))]
            cm_rows = [[f"Class {i}", *row] for i, row in enumerate(cm)]
            md_lines.append(tabulate(cm_rows, headers=cm_headers, tablefmt="github"))
            md_lines.append("\n")

    markdown_content = "\n".join(md_lines)

    report_dict = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": ckpt_name,
        "total_configs": len(results),
        "results": results,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{report_name}.json")
        md_path = os.path.join(output_dir, f"{report_name}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    return {
        "report_dict": report_dict,
        "markdown": markdown_content,
        "summary_table": summary_table_md,
    }


def _build_matrix_table(
    checkpoints: List[str],
    configs: List[str],
    matrix_data: Dict[Tuple[str, str], float],
    metric_name: str,
) -> Tuple[str, List[List[Any]]]:
    """Helper to build 2D tabulated cross-evaluation matrix."""
    headers = ["Checkpoint \\ Config"] + configs
    rows = []
    for ckpt in checkpoints:
        row = [ckpt]
        for cfg in configs:
            val = matrix_data.get((ckpt, cfg))
            if val is not None and isinstance(val, (int, float)):
                row.append(f"{val:.4f}")
            else:
                row.append("N/A")
        rows.append(row)
    table_str = tabulate(rows, headers=headers, tablefmt="github", disable_numparse=True)
    return f"### {metric_name} Cross-Evaluation Matrix\n\n{table_str}\n", rows


def generate_master_cross_evaluation_report(
    cross_results: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    report_name: str = "master_cross_evaluation_report",
) -> Dict[str, Any]:
    """
    Generate master cross-evaluation report containing all cross-evaluation information,
    matrices (IoU, F1, AUROC, Loss), rankings, and checkpoint/config breakdowns.
    """
    checkpoints_seen: Dict[str, str] = {}  # ckpt_path -> ckpt_name
    configs_seen: Dict[str, str] = {}      # cfg_path -> cfg_name

    for res in cross_results:
        cp = res.get("checkpoint_path", "unknown")
        cn = res.get("checkpoint_name", os.path.splitext(os.path.basename(cp))[0])
        kp = res.get("config_path", "unknown")
        kn = res.get("config_name", os.path.splitext(os.path.basename(kp))[0])
        checkpoints_seen[cp] = cn
        configs_seen[kp] = kn

    unique_ckpt_names = list(checkpoints_seen.values())
    unique_config_names = list(configs_seen.values())

    iou_matrix: Dict[Tuple[str, str], float] = {}
    f1_matrix: Dict[Tuple[str, str], float] = {}
    auroc_matrix: Dict[Tuple[str, str], float] = {}
    pacc_matrix: Dict[Tuple[str, str], float] = {}
    loss_matrix: Dict[Tuple[str, str], float] = {}

    master_summary_rows = []

    for res in cross_results:
        cp = res.get("checkpoint_path", "")
        cn = res.get("checkpoint_name", checkpoints_seen.get(cp, cp))
        kp = res.get("config_path", "")
        kn = res.get("config_name", configs_seen.get(kp, kp))
        ds_name = res.get("dataset_name", res.get("config", {}).get("data", {}).get("dataset_name", "N/A"))
        m = res.get("metrics", res.get("overall_metrics", {}))
        split = res.get("eval_split", m.get("eval_split", "test"))
        samples = res.get("total_evaluated_samples", m.get("total_evaluated_samples", 0))

        loss = m.get("eval_total_loss", m.get("total_loss", m.get("loss")))
        iou = m.get("iou", 0.0)
        f1 = m.get("f1", m.get("pixel_f1", m.get("dice", 0.0)))
        auroc = m.get("auroc", m.get("pixel_auroc", 0.0))
        pacc = m.get("pixel_acc", 0.0)
        prec = m.get("precision", 0.0)
        rec = m.get("recall", 0.0)
        spec = m.get("specificity", 0.0)
        cacc = m.get("aux_accuracy", m.get("accuracy"))

        iou_matrix[(cn, kn)] = iou
        f1_matrix[(cn, kn)] = f1
        auroc_matrix[(cn, kn)] = auroc
        pacc_matrix[(cn, kn)] = pacc
        if isinstance(loss, (int, float)):
            loss_matrix[(cn, kn)] = loss

        master_summary_rows.append([
            cn,
            f"{kn} ({ds_name})",
            str(split),
            str(samples),
            f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A",
            f"{iou:.4f}" if isinstance(iou, (int, float)) else "N/A",
            f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A",
            f"{auroc:.4f}" if isinstance(auroc, (int, float)) else "N/A",
            f"{pacc:.4f}" if isinstance(pacc, (int, float)) else "N/A",
            f"{prec:.4f}" if isinstance(prec, (int, float)) else "N/A",
            f"{rec:.4f}" if isinstance(rec, (int, float)) else "N/A",
            f"{spec:.4f}" if isinstance(spec, (int, float)) else "N/A",
            f"{cacc:.4f}" if isinstance(cacc, (int, float)) else "N/A",
        ])

    # Sort master summary rows by IoU descending
    def _safe_float(val):
        try:
            return float(val)
        except Exception:
            return -1.0

    master_summary_rows.sort(key=lambda r: _safe_float(r[5]), reverse=True)

    master_headers = [
        "Checkpoint",
        "Config (Dataset)",
        "Split",
        "Samples",
        "Loss",
        "IoU",
        "Dice / F1",
        "AUROC",
        "Pixel Acc",
        "Precision",
        "Recall",
        "Specificity",
        "Class Acc",
    ]
    master_table_md = tabulate(master_summary_rows, headers=master_headers, tablefmt="github", disable_numparse=True)

    iou_table_md, iou_rows = _build_matrix_table(unique_ckpt_names, unique_config_names, iou_matrix, "Mean IoU (Intersection over Union)")
    f1_table_md, f1_rows = _build_matrix_table(unique_ckpt_names, unique_config_names, f1_matrix, "Dice / Pixel F1 Score")
    auroc_table_md, auroc_rows = _build_matrix_table(unique_ckpt_names, unique_config_names, auroc_matrix, "AUROC (Pixel-level ROC-AUC)")
    pacc_table_md, pacc_rows = _build_matrix_table(unique_ckpt_names, unique_config_names, pacc_matrix, "Pixel Accuracy")
    loss_table_md, loss_rows = _build_matrix_table(unique_ckpt_names, unique_config_names, loss_matrix, "Evaluation Total Loss")

    md_lines = [
        "# SID-UNet Master Cross-Evaluation Report\n",
        "## Overview\n",
        f"- **Evaluated Checkpoints**: {len(unique_ckpt_names)}",
        f"- **Evaluated Configurations**: {len(unique_config_names)}",
        f"- **Total Cross-Evaluation Runs**: {len(cross_results)}\n",
        "## Cross-Evaluation Matrices\n",
        iou_table_md,
        f1_table_md,
        auroc_table_md,
        pacc_table_md,
        loss_table_md,
        "## Master Consolidated Ranking\n",
        master_table_md,
        "\n",
    ]

    # Checkpoint-centric breakdown
    md_lines.append("---\n\n## Per-Checkpoint Deep-Dive\n")
    for cp, cn in checkpoints_seen.items():
        ckpt_runs = [r for r in cross_results if r.get("checkpoint_path") == cp]
        md_lines.append(f"### Checkpoint: {cn}\n")
        md_lines.append(f"- **Path**: `{cp}`\n")
        ckpt_rows = []
        for cr in ckpt_runs:
            kn = cr.get("config_name", "N/A")
            ds = cr.get("dataset_name", "N/A")
            m = cr.get("metrics", cr.get("overall_metrics", {}))
            ckpt_rows.append([
                kn,
                ds,
                f"{m.get('eval_total_loss', 0.0):.4f}",
                f"{m.get('iou', 0.0):.4f}",
                f"{m.get('f1', m.get('pixel_f1', 0.0)):.4f}",
                f"{m.get('auroc', m.get('pixel_auroc', 0.0)):.4f}",
                f"{m.get('pixel_acc', 0.0):.4f}",
                f"{cr.get('total_evaluated_samples', 0)}",
            ])
        md_lines.append(tabulate(ckpt_rows, headers=["Config", "Dataset", "Loss", "IoU", "Dice / F1", "AUROC", "Pixel Acc", "Samples"], tablefmt="github"))
        md_lines.append("\n")

    # Config-centric breakdown
    md_lines.append("---\n\n## Per-Configuration / Dataset Deep-Dive\n")
    for kp, kn in configs_seen.items():
        cfg_runs = [r for r in cross_results if r.get("config_path") == kp]
        first_ds = cfg_runs[0].get("dataset_name", "N/A") if cfg_runs else "N/A"
        md_lines.append(f"### Configuration: {kn} (Dataset: {first_ds})\n")
        md_lines.append(f"- **Config Path**: `{kp}`\n")
        cfg_rows = []
        for cr in cfg_runs:
            cn = cr.get("checkpoint_name", "N/A")
            m = cr.get("metrics", cr.get("overall_metrics", {}))
            cfg_rows.append([
                cn,
                f"{m.get('eval_total_loss', 0.0):.4f}",
                f"{m.get('iou', 0.0):.4f}",
                f"{m.get('f1', m.get('pixel_f1', 0.0)):.4f}",
                f"{m.get('auroc', m.get('pixel_auroc', 0.0)):.4f}",
                f"{m.get('pixel_acc', 0.0):.4f}",
                f"{cr.get('total_evaluated_samples', 0)}",
            ])
        md_lines.append(tabulate(cfg_rows, headers=["Checkpoint", "Loss", "IoU", "Dice / F1", "AUROC", "Pixel Acc", "Samples"], tablefmt="github"))
        md_lines.append("\n")

    markdown_content = "\n".join(md_lines)

    matrices_dict = {
        "iou": {f"{c}:{k}": v for (c, k), v in iou_matrix.items()},
        "dice_f1": {f"{c}:{k}": v for (c, k), v in f1_matrix.items()},
        "auroc": {f"{c}:{k}": v for (c, k), v in auroc_matrix.items()},
        "pixel_acc": {f"{c}:{k}": v for (c, k), v in pacc_matrix.items()},
        "loss": {f"{c}:{k}": v for (c, k), v in loss_matrix.items()},
    }

    report_dict = {
        "total_checkpoints": len(unique_ckpt_names),
        "total_configs": len(unique_config_names),
        "total_runs": len(cross_results),
        "checkpoints": unique_ckpt_names,
        "configs": unique_config_names,
        "matrices": matrices_dict,
        "cross_results": cross_results,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{report_name}.json")
        md_path = os.path.join(output_dir, f"{report_name}.md")
        matrix_path = os.path.join(output_dir, "cross_eval_matrix.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        with open(matrix_path, "w", encoding="utf-8") as f:
            json.dump(matrices_dict, f, indent=2)

    return {
        "report_dict": report_dict,
        "markdown": markdown_content,
        "summary_table": master_table_md,
        "iou_matrix_table": iou_table_md,
        "matrices": matrices_dict,
    }

