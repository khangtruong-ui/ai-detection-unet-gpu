"""
CLI training entrypoint for SID-UNet.
Supports single-config training and multi-experiment execution across multiple configs.

Usage:
    # Single experiment:
    python -m sid_unet.train --config configs/train_streaming.yaml
    python -m sid_unet.train --config configs/default.yaml --override training.batch_size=8 training.epochs=5

    # Multi-experiment suite (runs sequentially and generates comparative report):
    python -m sid_unet.train --configs configs/test_smoke.yaml configs/test_quick.yaml
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from sid_unet.dataset.loader import create_dataloaders
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config, save_config
from sid_unet.utils.logger import setup_logger
from sid_unet.utils.plotting import plot_multi_experiment_curves
from sid_unet.utils.report import generate_multi_experiment_report



def set_seed(seed: int = 42):
    """Set deterministic seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for AI Generated Image Masking on SID_Set")
    parser.add_argument(
        "--config",
        "--configs",
        nargs="+",
        dest="config",
        default=["configs/train_streaming.yaml"],
        help="Path(s) to YAML configuration file(s). Pass multiple files to run multiple experiments sequentially.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save experiment outputs/checkpoints/reports",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per training/validation step (overrides config data.batch_size)",
    )
    parser.add_argument(
        "--auto_batch_size",
        "--auto-batch-size",
        dest="auto_batch_size",
        action="store_true",
        default=None,
        help="Automatically probe GPU memory and scale down batch size / increase gradient accumulation to avoid OOM (default: enabled)",
    )
    parser.add_argument(
        "--no_auto_batch_size",
        "--no-auto-batch-size",
        "--disable-auto-batch-size",
        dest="auto_batch_size",
        action="store_false",
        help="Disable automatic batch size scaling",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Number of micro-batches to accumulate before optimizer step",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable activation gradient checkpointing in UNet to save VRAM",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key.nested=value format (e.g., --override training.batch_size=16 data.streaming=false)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from",
    )
    # Collision checking flags
    parser.add_argument(
        "--skip-collision",
        "--skip_collision",
        dest="skip_collision",
        action="store_true",
        default=True,
        help="Check for collision against previously trained/evaluated models and skip with notification (default: True).",
    )
    parser.add_argument(
        "--no-skip-collision",
        "--force",
        dest="skip_collision",
        action="store_false",
        help="Disable collision checking and force retraining/evaluation.",
    )
    return parser.parse_args()


def train_single_run(
    config_path: str,
    overrides: Optional[List[str]] = None,
    resume: Optional[str] = None,
    run_idx: int = 1,
    total_runs: int = 1,
    base_output_dir: Optional[str] = None,
    skip_collision: bool = True,
) -> Dict[str, Any]:
    """Execute a single training experiment with its given config."""
    config = load_config(config_path, overrides=overrides or [])

    # Set random seed
    seed = int(config.project.get("seed", 42))
    set_seed(seed)

    # In multi-run mode, name each run folder cleanly as RUN001, RUN002, etc.
    cfg_stem = os.path.splitext(os.path.basename(config_path))[0]
    if total_runs > 1:
        run_name = f"RUN{run_idx:03d}"
        root_dir = base_output_dir if base_output_dir else "outputs"
        output_dir = os.path.join(root_dir, run_name)
        config.project.output_dir = output_dir
        config.project.name = run_name
    else:
        if base_output_dir:
            output_dir = base_output_dir
            config.project.output_dir = output_dir
        else:
            output_dir = config.project.get("output_dir", "outputs")
        run_name = config.project.get("name", cfg_stem)

    os.makedirs(output_dir, exist_ok=True)

    # Save copy of effective config in output directory
    config_save_path = os.path.join(output_dir, "effective_config.yaml")
    save_config(config, config_save_path)

    logger = setup_logger(
        name=run_name,
        log_file=os.path.join(output_dir, "logs", "train_run.log"),
    )
    logger.info(f"[{run_idx}/{total_runs}] Loaded configuration from '{config_path}' (Run: {run_name})")
    logger.info(f"Effective configuration saved to '{config_save_path}'")
    logger.info(f"Dataset: {config.data.dataset_name} | Streaming: {config.data.streaming}")

    # Check for existing checkpoint (continue from checkpoint as default behaviour)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    latest_ckpt = os.path.join(ckpt_dir, "checkpoint_latest.pt")
    best_ckpt = os.path.join(ckpt_dir, "checkpoint_best.pt")

    if resume is None:
        if os.path.exists(latest_ckpt):
            resume = latest_ckpt
            logger.info(f"Found existing latest checkpoint '{resume}' - continuing from checkpoint by default.")
        elif os.path.exists(best_ckpt):
            resume = best_ckpt
            logger.info(f"Found existing best checkpoint '{resume}' - continuing from checkpoint by default.")

    # Collision check: has this combination already been trained and evaluated?
    eval_rep_json = os.path.join(output_dir, "eval_reports", "evaluation_report.json")
    if not os.path.exists(eval_rep_json):
        eval_rep_json = os.path.join(output_dir, "evaluation_report.json")

    if skip_collision and os.path.exists(eval_rep_json) and (resume or os.path.exists(best_ckpt) or os.path.exists(latest_ckpt)):
        try:
            with open(eval_rep_json, "r", encoding="utf-8") as f:
                import json
                saved_eval = json.load(f)
                saved_cfg = saved_eval.get("config", {})
                m_name = saved_cfg.get("model", {}).get("name", config.model.name)
                d_name = saved_cfg.get("data", {}).get("dataset_name", config.data.dataset_name)
                om = saved_eval.get("overall_metrics", {})
                score = om.get("val_iou", om.get("iou", 0.0))
                logger.info(
                    f"\n⚡ [COLLISION DETECTED - SKIPPED] Model config '{m_name}' and dataset config '{d_name}' "
                    f"in '{output_dir}' has already been evaluated. Skipping run..."
                )
                return {
                    "config_path": config_path,
                    "run_name": run_name,
                    "best_score": score,
                    "best_epoch": saved_eval.get("best_epoch", -1),
                    "best_checkpoint_path": best_ckpt if os.path.exists(best_ckpt) else (resume or ""),
                    "report_path": eval_rep_json.replace(".json", ".md"),
                    "history": saved_eval.get("history", []),
                    "overall_metrics": om,
                    "final_metrics": om,
                    "output_dir": output_dir,
                }
        except Exception as e:
            logger.warning(f"Could not read previous evaluation report '{eval_rep_json}': {e}")

    # Build DataLoaders
    logger.info("Initializing DataLoaders...")
    evaluate_on_test = bool(config.data.get("evaluate_on_test", False))
    if evaluate_on_test:
        train_loader, val_loader, test_loader = create_dataloaders(config, include_test=True)
    else:
        train_loader, val_loader = create_dataloaders(config, include_test=False)
        test_loader = None

    # Build Trainer
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        custom_logger=logger,
    )

    # Optional resume
    if resume:
        logger.info(f"Resuming training from checkpoint '{resume}'...")
        trainer.ckpt_manager.load_checkpoint(resume, trainer.model, trainer.optimizer, trainer.scheduler)

    # Run training
    results = trainer.train()
    results["config_path"] = config_path
    results["run_name"] = run_name

    logger.info(f"Experiment '{results['run_name']}' finished successfully!")
    logger.info(f"Best Score: {results['best_score']:.4f} (Epoch {results['best_epoch']})")
    logger.info(f"Evaluation report: {results['report_path']}")

    return results


def main():
    args = parse_args()
    config_paths = args.config if isinstance(args.config, list) else [args.config]

    overrides = list(args.override)
    if args.batch_size is not None:
        overrides.append(f"data.batch_size={args.batch_size}")
    if args.auto_batch_size is True:
        overrides.append("training.auto_batch_size=true")
    elif args.auto_batch_size is False:
        overrides.append("training.auto_batch_size=false")
    if args.gradient_accumulation_steps is not None:
        overrides.append(f"training.gradient_accumulation_steps={args.gradient_accumulation_steps}")
    if args.gradient_checkpointing:
        overrides.append("model.gradient_checkpointing=true")
        overrides.append("training.gradient_checkpointing=true")

    if len(config_paths) == 1:
        if args.output_dir:
            overrides.append(f"project.output_dir={args.output_dir}")
        results = train_single_run(
            config_path=config_paths[0],
            overrides=overrides,
            resume=args.resume,
            run_idx=1,
            total_runs=1,
            base_output_dir=args.output_dir,
            skip_collision=args.skip_collision,
        )
        return results

    # Multi-experiment suite
    parent_output_dir = args.output_dir
    if not parent_output_dir:
        for ov in overrides:
            if ov.startswith("project.output_dir="):
                parent_output_dir = ov.split("=", 1)[1].strip()
                break
    if not parent_output_dir:
        parent_output_dir = "outputs"

    print("\n" + "=" * 70)
    print(f"🚀 Launching Multi-Experiment Suite ({len(config_paths)} experiments)")
    print(f"📁 Suite Output Directory: {parent_output_dir}")
    print(f"⚡ Collision Detection / Skip: {args.skip_collision}")
    print("=" * 70 + "\n")

    all_results: List[Dict[str, Any]] = []

    for i, cfg_path in enumerate(config_paths, 1):
        print(f"\n>>> Running Experiment [{i}/{len(config_paths)}]: {cfg_path} (RUN{i:03d})")
        print("-" * 70)
        res = train_single_run(
            config_path=cfg_path,
            overrides=overrides,
            resume=args.resume if i == 1 else None,
            run_idx=i,
            total_runs=len(config_paths),
            base_output_dir=parent_output_dir,
            skip_collision=args.skip_collision,
        )
        all_results.append(res)

    # Collect experiment histories and plot multi-run comparison curves
    histories_dict = {}
    for r in all_results:
        exp_name = r.get("run_name", "Run")
        if r.get("history"):
            histories_dict[exp_name] = r["history"]

    multi_curves_path = None
    if histories_dict:
        multi_curves_path = os.path.join(parent_output_dir, "multi_experiment_curves.png")
        plot_multi_experiment_curves(
            experiment_histories=histories_dict,
            output_path=multi_curves_path,
        )

    # Generate and display continuous multi-experiment comparison report
    combined_results = list(all_results)
    multi_json_path = os.path.join(parent_output_dir, "multi_experiment_comparison.json")
    if os.path.exists(multi_json_path):
        try:
            with open(multi_json_path, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
                if isinstance(data, dict) and "experiments" in data:
                    existing_runs = data["experiments"]
                    curr_runs = {r.get("run_name") for r in all_results if r.get("run_name")}
                    for er in existing_runs:
                        if er.get("run_name") not in curr_runs:
                            combined_results.append(er)
        except Exception:
            pass

    multi_report = generate_multi_experiment_report(
        experiment_results=combined_results,
        output_dir=parent_output_dir,
        report_name="multi_experiment_comparison",
        multi_curves_path=multi_curves_path,
    )

    print("\n" + "=" * 70)
    print("⭐ ALL EXPERIMENTS COMPLETED - SUMMARY REPORT")
    print("=" * 70)
    print(multi_report["summary_table"])
    if multi_curves_path and os.path.exists(multi_curves_path):
        print(f"Multi-Experiment comparison curves plot: {multi_curves_path}")
    print(f"\nDetailed Markdown comparison: {os.path.join(parent_output_dir, 'multi_experiment_comparison.md')}")
    print(f"Detailed JSON comparison: {os.path.join(parent_output_dir, 'multi_experiment_comparison.json')}\n")

    return all_results



def cli_main():
    import sys
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli_main()
