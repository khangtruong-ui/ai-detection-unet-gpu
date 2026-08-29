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
    return parser.parse_args()


def train_single_run(
    config_path: str,
    overrides: Optional[List[str]] = None,
    resume: Optional[str] = None,
    run_idx: int = 1,
    total_runs: int = 1,
    base_output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Execute a single training experiment with its given config."""
    config = load_config(config_path, overrides=overrides or [])

    # Set random seed
    seed = int(config.project.get("seed", 42))
    set_seed(seed)

    # In multi-run mode, give each run an isolated output subdirectory under base_output_dir
    cfg_stem = os.path.splitext(os.path.basename(config_path))[0]
    if total_runs > 1:
        output_dir = os.path.join(base_output_dir, f"exp_{run_idx:02d}_{cfg_stem}")
        config.project.output_dir = output_dir
    else:
        output_dir = config.project.get("output_dir", "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # Save copy of effective config in output directory
    config_save_path = os.path.join(output_dir, "effective_config.yaml")
    save_config(config, config_save_path)

    logger = setup_logger(
        name=f"SID_UNet_Run_{run_idx}" if total_runs > 1 else "SID_UNet",
        log_file=os.path.join(output_dir, "logs", "train_run.log"),
    )
    logger.info(f"[{run_idx}/{total_runs}] Loaded configuration from '{config_path}'")
    logger.info(f"Effective configuration saved to '{config_save_path}'")
    logger.info(f"Dataset: {config.data.dataset_name} | Streaming: {config.data.streaming}")

    # Build DataLoaders
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader = create_dataloaders(config)

    # Build Trainer
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        custom_logger=logger,
    )

    # Optional resume
    if resume:
        logger.info(f"Resuming training from checkpoint '{resume}'...")
        trainer.ckpt_manager.load_checkpoint(resume, trainer.model, trainer.optimizer, trainer.scheduler)

    # Run training
    results = trainer.train()
    results["config_path"] = config_path
    results["run_name"] = config.project.get("name", cfg_stem)

    logger.info(f"Experiment '{results['run_name']}' finished successfully!")
    logger.info(f"Best Score: {results['best_score']:.4f} (Epoch {results['best_epoch']})")
    logger.info(f"Evaluation report: {results['report_path']}")

    return results


def main():
    args = parse_args()
    config_paths = args.config if isinstance(args.config, list) else [args.config]

    if len(config_paths) == 1:
        results = train_single_run(
            config_path=config_paths[0],
            overrides=args.override,
            resume=args.resume,
            run_idx=1,
            total_runs=1,
        )
        return results

    # Multi-experiment suite
    print("\n" + "=" * 70)
    print(f"🚀 Launching Multi-Experiment Suite ({len(config_paths)} experiments)")
    print("=" * 70 + "\n")

    all_results: List[Dict[str, Any]] = []
    first_cfg = load_config(config_paths[0], overrides=args.override)
    parent_output_dir = first_cfg.project.get("output_dir", "outputs")

    for i, cfg_path in enumerate(config_paths, 1):
        print(f"\n>>> Running Experiment [{i}/{len(config_paths)}]: {cfg_path}")
        print("-" * 70)
        res = train_single_run(
            config_path=cfg_path,
            overrides=args.override,
            resume=args.resume if i == 1 else None,
            run_idx=i,
            total_runs=len(config_paths),
            base_output_dir=parent_output_dir,
        )
        all_results.append(res)

    # Generate and display multi-experiment comparison report
    multi_report = generate_multi_experiment_report(
        experiment_results=all_results,
        output_dir=parent_output_dir,
        report_name="multi_experiment_comparison",
    )

    print("\n" + "=" * 70)
    print("⭐ ALL EXPERIMENTS COMPLETED - SUMMARY REPORT")
    print("=" * 70)
    print(multi_report["summary_table"])
    print(f"\nDetailed Markdown comparison: {os.path.join(parent_output_dir, 'multi_experiment_comparison.md')}")
    print(f"Detailed JSON comparison: {os.path.join(parent_output_dir, 'multi_experiment_comparison.json')}\n")

    return all_results


if __name__ == "__main__":
    main()
