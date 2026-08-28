"""
CLI training entrypoint for SID-UNet.
Usage:
    python -m sid_unet.train --config configs/train_streaming.yaml
    python -m sid_unet.train --config configs/default.yaml --override training.batch_size=8 training.epochs=5
"""

from __future__ import annotations

import argparse
import os
import random
import numpy as np
import torch

from sid_unet.dataset.loader import create_dataloaders
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config, save_config
from sid_unet.utils.logger import setup_logger


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
        type=str,
        default="configs/train_streaming.yaml",
        help="Path to YAML configuration file",
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


def main():
    args = parse_args()
    config = load_config(args.config, overrides=args.override)

    # Set random seed
    seed = int(config.project.get("seed", 42))
    set_seed(seed)

    output_dir = config.project.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save copy of effective config in output directory
    config_save_path = os.path.join(output_dir, "effective_config.yaml")
    save_config(config, config_save_path)

    logger = setup_logger(
        name="SID_UNet",
        log_file=os.path.join(output_dir, "logs", "train_run.log"),
    )
    logger.info(f"Loaded configuration from '{args.config}'")
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
    if args.resume:
        logger.info(f"Resuming training from checkpoint '{args.resume}'...")
        trainer.ckpt_manager.load_checkpoint(args.resume, trainer.model, trainer.optimizer, trainer.scheduler)

    # Run training
    results = trainer.train()
    logger.info("Training pipeline finished successfully!")
    logger.info(f"Best Score: {results['best_score']:.4f} (Epoch {results['best_epoch']})")
    logger.info(f"Evaluation report: {results['report_path']}")


if __name__ == "__main__":
    main()
