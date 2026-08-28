"""
Report generation and formatting utilities for SID-UNet evaluation and benchmarking.
Produces Markdown, JSON, and rich formatted terminal tables.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from tabulate import tabulate


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
    return f"### {title}\n\n{table_str}\n"


def generate_evaluation_report(
    overall_metrics: Dict[str, float],
    per_label_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    confusion_matrix: Optional[list[list[int]]] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    report_name: str = "evaluation_report",
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report with overall metrics, per-label stats,
    and auxiliary classification details. Saves both JSON and Markdown formats.
    """
    label_names = {0: "Label 0 (Real / Black Mask)", 1: "Label 1 (Synthetic / White Mask)", 2: "Label 2 (Tampered / Partial Mask)"}

    report_dict: Dict[str, Any] = {
        "overall_metrics": overall_metrics,
        "per_label_metrics": {},
    }

    if per_label_metrics:
        for lbl, stats in per_label_metrics.items():
            lbl_key = label_names.get(lbl, f"Label {lbl}")
            report_dict["per_label_metrics"][lbl_key] = stats

    if confusion_matrix is not None:
        report_dict["auxiliary_classification_confusion_matrix"] = confusion_matrix

    if config:
        report_dict["config_summary"] = {
            "dataset": config.get("data", {}).get("dataset_name"),
            "image_size": config.get("data", {}).get("image_size"),
            "model": config.get("model", {}).get("name"),
            "aux_classifier": config.get("model", {}).get("aux_classifier"),
        }

    # Generate Markdown text
    md_lines = ["# SID-UNet Evaluation Report\n"]
    md_lines.append(format_metrics_table(overall_metrics, title="Overall Segmentation & Classification Metrics"))

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

    if confusion_matrix is not None:
        md_lines.append("### Auxiliary Classification Confusion Matrix (Rows: Ground Truth, Cols: Predicted)\n")
        cm_headers = ["Actual \\ Pred", "Class 0 (Real)", "Class 1 (Synthetic)", "Class 2 (Tampered)"]
        cm_rows = []
        for i, row in enumerate(confusion_matrix):
            cm_rows.append([f"Class {i}", *row])
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
