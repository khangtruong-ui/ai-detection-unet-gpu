"""
CLI entrypoint for dataset tensor caching (sid-cache / sid-dataset-cache).
Extracts frozen model latents and syncs to local disk and Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import torch

from sid_unet.models.unet import build_model
from sid_unet.utils.config import load_config
from sid_unet.utils.logger import setup_logger
from sid_unet.cache.manager import DatasetCacheManager


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and cache high-dimensional diffusion forensics latents (sid-cache)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/diffusion_diff_minimized/default.yaml",
        help="Path to YAML model configuration file (default: configs/experiments/diffusion_diff_minimized/default.yaml)",
    )
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        type=str,
        default=None,
        dest="dataset_name",
        help="Source dataset name or path on Hugging Face (default: data.dataset_name from config)",
    )
    parser.add_argument(
        "--hf-repo",
        "--cached-hf-repo",
        "--target-repo",
        type=str,
        default=None,
        dest="hf_repo",
        help="Target Hugging Face Hub repository ID to upload cached dataset to (e.g. 'KhangTruong/COCO-inpainted-cache')",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=str,
        default="outputs/dataset_cache",
        help="Local directory to store cached Parquet shards (default: 'outputs/dataset_cache')",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for feature extraction (default: 8)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Dataset splits to cache (default: ['train', 'validation'])",
    )
    parser.add_argument(
        "--max-samples",
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to extract per split (useful for testing and fast runs)",
    )
    parser.add_argument(
        "--samples-per-shard",
        "--samples_per_shard",
        type=int,
        default=5000,
        help="Number of samples to pack into each Parquet shard file (default: 5000)",
    )
    parser.add_argument(
        "--push-to-hub",
        "--push_to_hub",
        dest="push_to_hub",
        action="store_true",
        default=None,
        help="Upload cached shards to Hugging Face Hub dataset repo upon completion (default: true if --hf-repo is specified)",
    )
    parser.add_argument(
        "--no-push-to-hub",
        "--no_push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not upload to Hugging Face Hub even if --hf-repo is provided",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Save cached tensors in float16 precision (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token for repository write access (optional if logged in via huggingface-cli)",
    )
    return parser.parse_args(args)


def cli_main(args: Optional[list] = None) -> None:
    parsed_args = parse_args(args)
    logger = setup_logger(name="sid_cache")

    logger.info("🔧 Loading configuration...")
    if not os.path.exists(parsed_args.config):
        logger.error(f"Config file not found: {parsed_args.config}")
        sys.exit(1)

    cfg = load_config(parsed_args.config)
    source_dataset = parsed_args.dataset_name or cfg.data.get("dataset_name", "KhangTruong/COCO-inpainted")
    hf_repo = parsed_args.hf_repo or cfg.data.get("cached_hf_repo", None)

    # Determine push_to_hub default
    push_hub = parsed_args.push_to_hub
    if push_hub is None:
        push_hub = bool(hf_repo)

    device_str = parsed_args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    logger.info(f"🏗️ Building model from config: {parsed_args.config} on device {device}...")
    model = build_model(cfg).to(device)

    img_size = tuple(cfg.data.get("image_size", [256, 256]))

    manager = DatasetCacheManager(
        model=model,
        source_dataset_name=source_dataset,
        output_dir=parsed_args.output_dir,
        hf_repo_id=hf_repo,
        batch_size=parsed_args.batch_size,
        samples_per_shard=parsed_args.samples_per_shard,
        image_size=img_size,
        device=device,
        fp16=parsed_args.fp16,
        hf_token=parsed_args.token,
    )

    logger.info(
        f"🚀 Running cache extraction: source='{source_dataset}', splits={parsed_args.splits}, "
        f"max_samples={parsed_args.max_samples}, output_dir='{parsed_args.output_dir}'"
    )

    all_shards = manager.cache_all_splits(
        splits=parsed_args.splits,
        max_samples_per_split=parsed_args.max_samples,
        push_to_hub=push_hub,
    )

    logger.info(f"🎉 Dataset caching finished! Total shards generated: {sum(len(v) for v in all_shards.values())}")
    if push_hub and hf_repo:
        logger.info(f"🔗 Hugging Face Hub dataset URL: https://huggingface.co/datasets/{hf_repo}")


if __name__ == "__main__":
    cli_main()
