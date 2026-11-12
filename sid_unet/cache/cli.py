"""
CLI entrypoint for dataset tensor caching (sid-cache / sid-dataset-cache).
Extracts frozen model latents and progressively syncs to local disk and Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Optional

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
        "--detach",
        dest="detach",
        action="store_true",
        default=True,
        help="Detach from terminal and run dataset caching in background logging to .logs.txt (default: True)",
    )
    parser.add_argument(
        "--no-detach",
        "--no_detach",
        "--foreground",
        "-f",
        dest="detach",
        action="store_false",
        help="Run dataset caching in foreground attached to terminal (do not detach)",
    )
    parser.add_argument(
        "--log-file",
        "--logfile",
        "--logs-file",
        dest="log_file",
        type=str,
        default=".logs.txt",
        help="File path to redirect standard output and error when detached (default: .logs.txt)",
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
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key.nested=value format (e.g., --override data.image_size=[512,512])",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for feature extraction (default: 8 or from config)",
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
        default=2000,
        help="Number of samples to pack into each Parquet shard file (default: 2000)",
    )
    parser.add_argument(
        "--push-to-hub",
        "--push_to_hub",
        dest="push_to_hub",
        action="store_true",
        default=None,
        help="Progressively upload cached shards to Hugging Face Hub dataset repo (default: true if --hf-repo is specified)",
    )
    parser.add_argument(
        "--no-push-to-hub",
        "--no_push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not upload to Hugging Face Hub even if --hf-repo is provided",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume caching from existing remote/local shards, skipping already processed samples (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume; restart caching from shard index 0",
    )
    parser.add_argument(
        "--delete-on-update",
        "--delete-local-on-upload",
        "--delete-on-upload",
        dest="delete_on_update",
        action="store_true",
        default=True,
        help="Remove local Parquet files once uploaded to Hugging Face Hub to save disk space (default: True)",
    )
    parser.add_argument(
        "--no-delete-on-update",
        "--no-delete-local-on-upload",
        "--no-delete-on-upload",
        "--keep-local",
        dest="delete_on_update",
        action="store_false",
        help="Keep local Parquet files after uploading to Hugging Face Hub",
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
    parser.add_argument(
        "--store-images",
        dest="store_images",
        action="store_true",
        default=None,
        help="Store RGB images in Parquet shards alongside features (default: True for descriptor models like gap_sam)",
    )
    parser.add_argument(
        "--no-store-images",
        dest="store_images",
        action="store_false",
        help="Do not store RGB images in Parquet shards",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        "-j",
        type=int,
        default=None,
        dest="num_workers",
        help="Number of background CPU worker threads for image loading and preprocessing (default: 4)",
    )
    parser.add_argument(
        "--prefetch-batches",
        "--prefetch_batches",
        type=int,
        default=None,
        dest="prefetch_batches",
        help="Number of batches to prefetch ahead in RAM for non-blocking GPU execution (default: 16 or from config)",
    )
    parsed = parser.parse_args(args)
    parsed.delete_local_on_upload = getattr(parsed, "delete_on_update", True)
    return parsed


def cli_main(args: Optional[list] = None) -> Any:
    from sid_unet.utils.signals import shield_process_signals
    shield_process_signals()

    parsed_args = parse_args(args)

    # Check if we should detach from controlling terminal and run in background logging to .logs.txt
    in_dist = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    in_pytest = ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)
    already_detached = bool(os.environ.get("SID_UNET_DETACHED", False))
    no_detach_env = bool(os.environ.get("SID_UNET_NO_DETACH", False)) or bool(os.environ.get("SID_UNET_FOREGROUND", False))

    if bool(getattr(parsed_args, "detach", True)) and not in_dist and not in_pytest and not already_detached and not no_detach_env:
        import subprocess
        from datetime import datetime

        log_path = os.path.abspath(getattr(parsed_args, "log_file", None) or ".logs.txt")
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        start_offset = os.path.getsize(log_path) if os.path.isfile(log_path) else 0

        log_fp = open(log_path, "a", encoding="utf-8")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_fp.write(
            f"\n{'='*70}\n"
            f"🚀 [sid-cache detached session started at {now_str}]\n"
            f"Command: {' '.join(sys.argv if args is None else [sys.executable, '-m', 'sid_unet.cache.cli'] + list(args))}\n"
            f"{'='*70}\n\n"
        )
        log_fp.flush()

        child_env = os.environ.copy()
        child_env["SID_UNET_DETACHED"] = "1"
        child_env["PYTHONUNBUFFERED"] = "1"

        sys_args = list(args) if args is not None else sys.argv[1:]
        child_argv = [a for a in sys_args if a != "--detach"]
        if not any(a in ("--no-detach", "--no_detach", "--foreground", "-f") for a in child_argv):
            child_argv.insert(0, "--no-detach")

        cmd = [sys.executable, "-m", "sid_unet.cache.cli"] + child_argv

        sub_kwargs = {"start_new_session": True} if os.name != "nt" else {}
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=child_env,
            **sub_kwargs,
        )
        log_fp.close()

        # Register in process ledger
        try:
            from sid_unet.process_manager import get_process_registry
            get_process_registry().register_process(
                pid=proc.pid,
                name="sid-cache",
                cmdline=" ".join(cmd),
                log_file=log_path,
                ppid=os.getpid(),
                log_offset_start=start_offset,
            )
        except Exception:
            pass

        print("\n" + "=" * 70)
        print(f"🚀 [DETACHED] sid-cache running in background (PID: {proc.pid})")
        print(f"📄 Logging output to: {log_path}")
        print(f"🖥️ To manage processes & view live output: sid-tui")
        print(f"🔍 Or tail logs directly: tail -f {getattr(parsed_args, 'log_file', '.logs.txt')}")
        print("🛑 To stop caching: sid-kill (or 'k' in sid-tui)")
        print("=" * 70 + "\n")
        sys.stdout.flush()
        return {"status": "detached", "pid": proc.pid, "log_file": log_path}

    logger = setup_logger(name="sid_cache")

    logger.info("🔧 Loading configuration...")
    if not os.path.exists(parsed_args.config):
        logger.error(f"Config file not found: {parsed_args.config}")
        sys.exit(1)

    overrides = list(parsed_args.override or [])
    cfg = load_config(parsed_args.config, overrides=overrides)
    source_dataset = parsed_args.dataset_name or cfg.data.get("dataset_name", "KhangTruong/DEAL300K-30K")
    hf_repo = parsed_args.hf_repo or cfg.data.get("cached_hf_repo", None)

    # Determine push_to_hub default
    push_hub = parsed_args.push_to_hub
    if push_hub is None:
        push_hub = bool(hf_repo)

    device_str = parsed_args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if device.type == "cuda" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass

    logger.info(f"🏗️ Building model from config: {parsed_args.config} on device {device}...")
    model = build_model(cfg).to(device)

    img_size = tuple(cfg.data.get("image_size", [256, 256]))
    batch_size = parsed_args.batch_size if parsed_args.batch_size is not None else cfg.data.get("batch_size", 8)
    num_workers = parsed_args.num_workers
    if num_workers is None:
        cfg_workers = cfg.data.get("num_workers", None)
        if cfg_workers is not None and cfg_workers > 0:
            num_workers = int(cfg_workers)
        elif cfg_workers is not None and cfg_workers < 0:
            num_workers = min(16, max(4, os.cpu_count() or 4))
        else:
            num_workers = min(4, os.cpu_count() or 4)
    elif num_workers < 0:
        num_workers = min(16, max(4, os.cpu_count() or 4))

    prefetch_batches = parsed_args.prefetch_batches
    if prefetch_batches is None:
        prefetch_batches = int(cfg.data.get("prefetch_batches", 16))

    manager = DatasetCacheManager(
        model=model,
        source_dataset_name=source_dataset,
        output_dir=parsed_args.output_dir,
        hf_repo_id=hf_repo,
        batch_size=batch_size,
        samples_per_shard=parsed_args.samples_per_shard,
        image_size=img_size,
        device=device,
        fp16=parsed_args.fp16,
        hf_token=parsed_args.token,
        resume=parsed_args.resume,
        delete_local_on_upload=parsed_args.delete_local_on_upload,
        push_to_hub=push_hub,
        store_images=parsed_args.store_images,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
    )

    is_vector_db_mode = (
        cfg.get("cache", {}).get("mode") == "vector_db"
        or "vector_db" in getattr(parsed_args, "config", "")
        or "mask-clip-db-mae" in getattr(parsed_args, "config", "")
    )

    if is_vector_db_mode:
        from sid_unet.cache.vector_db import MaskCLIPVectorDBBuilder
        from sid_unet.cache.manager import PrefetchedBatchLoader
        from sid_unet.dataset.loader import load_parquet_or_hf_dataset
        from sid_unet.dataset.mask_utils import decode_mask_bytes
        from tqdm import tqdm

        cache_cfg = cfg.get("cache", {})
        target_entries = int(cache_cfg.get("target_entries", 100000))
        forged_ratio = float(cache_cfg.get("forged_ratio", 0.9))
        authentic_ratio = float(cache_cfg.get("authentic_ratio", 0.1))
        db_filename = cache_cfg.get("vector_db_filename", "mask_clip_db_mae.pt")
        output_db_path = os.path.join(parsed_args.output_dir, db_filename)
        dtype_str = "float16" if parsed_args.fp16 else cache_cfg.get("dtype", "float16")

        builder = MaskCLIPVectorDBBuilder(
            model=model,
            target_entries=target_entries,
            forged_ratio=forged_ratio,
            authentic_ratio=authentic_ratio,
            dtype=dtype_str,
            max_patches_per_image_forged=int(cache_cfg.get("max_patches_per_image_forged", 32)),
            max_patches_per_image_authentic=int(cache_cfg.get("max_patches_per_image_authentic", 4)),
            seed=int(cfg.get("project", {}).get("seed", 42)),
        )

        split_to_use = parsed_args.splits[0] if parsed_args.splits else "train"
        ds, _ = load_parquet_or_hf_dataset(source_dataset, requested_split=split_to_use, streaming=True)

        loader = PrefetchedBatchLoader(
            dataset=ds,
            process_sample_fn=manager._process_sample_to_tensors,
            batch_size=batch_size,
            max_samples=parsed_args.max_samples,
            store_images=False,
            num_workers=num_workers,
        )

        logger.info(
            f"🚀 [VECTOR DB] Extracting MaskCLIP Pre-TVCA Vector Database: "
            f"target={target_entries} (90% forged / 10% authentic), output='{output_db_path}'"
        )
        pbar = tqdm(total=target_entries, desc="MaskCLIP Vector DB")
        for batch in loader:
            images = batch["images"]
            masks_raw = batch["masks"]
            decoded_masks = []
            for m in masks_raw:
                if isinstance(m, bytes):
                    decoded_masks.append(torch.from_numpy(decode_mask_bytes(m)))
                elif isinstance(m, torch.Tensor):
                    decoded_masks.append(m)
                else:
                    decoded_masks.append(torch.zeros(images.shape[2:], dtype=torch.float32))
            masks = torch.stack(decoded_masks, dim=0)

            builder.process_batch(images, masks)
            total_curr = builder.num_forged + builder.num_authentic
            pbar.n = min(total_curr, target_entries)
            pbar.set_postfix({
                "forged": f"{builder.num_forged}/{builder.target_forged}",
                "auth": f"{builder.num_authentic}/{builder.target_authentic}",
            })
            pbar.refresh()
            if builder.is_complete:
                break

        pbar.close()
        loader.close()
        saved_file = builder.save(output_db_path)
        logger.info(f"🎉 Vector database successfully saved to: {saved_file}")

        if manager.uploader is not None:
            logger.info(f"📡 Uploading vector database to Hugging Face Hub: '{hf_repo}'...")
            try:
                manager.uploader.upload_file_sync(
                    local_path=saved_file,
                    rel_repo_path=os.path.basename(saved_file),
                    commit_message=f"Upload MaskCLIP Pre-TVCA Vector Database ({builder.target_entries} entries)",
                )
                logger.info(f"🎉 Vector database successfully uploaded to Hugging Face Hub: '{hf_repo}'")
            except Exception as e:
                logger.warning(f"Failed to upload vector database to Hugging Face Hub: {e}")
            finally:
                manager.close()
        else:
            manager.close()

        in_pytest = ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)
        if args is None and not in_pytest:
            sys.exit(0)
        return {"vector_db": [saved_file]}

    logger.info(
        f"🚀 Running progressive cache extraction: source='{source_dataset}', splits={parsed_args.splits}, "
        f"max_samples={parsed_args.max_samples}, output_dir='{parsed_args.output_dir}', "
        f"push_to_hub={push_hub}, resume={parsed_args.resume}"
    )

    all_shards = manager.cache_all_splits(
        splits=parsed_args.splits,
        max_samples_per_split=parsed_args.max_samples,
        push_to_hub=push_hub,
        resume=parsed_args.resume,
    )

    logger.info(f"🎉 Dataset caching finished! Total shards generated: {sum(len(v) for v in all_shards.values())}")
    if push_hub and hf_repo:
        logger.info(f"🔗 Hugging Face Hub dataset URL: https://huggingface.co/datasets/{hf_repo}")

    in_pytest = ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)
    if args is None and not in_pytest:
        sys.exit(0)
    return all_shards


if __name__ == "__main__":
    cli_main()
