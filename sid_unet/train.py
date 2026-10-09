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
import sys
import warnings
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import ImageFile
import torch

# Ensure PIL handles truncated images during dataset loading and training
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sid_unet.dataset.loader import create_dataloaders
from sid_unet.training.trainer import Trainer
from sid_unet.utils.checkpoint import (
    find_auto_resume_checkpoint,
    download_hf_checkpoint,
    is_hf_repo_id,
    inspect_checkpoint,
    format_resume_notification,
    format_no_resume_notification,
)
from sid_unet.utils.config import load_config, save_config
from sid_unet.utils.logger import setup_logger
from sid_unet.utils.plotting import plot_multi_experiment_curves
from sid_unet.utils.report import generate_multi_experiment_report
from sid_unet.utils.distributed import (
    init_distributed_mode,
    cleanup_distributed,
    is_dist_avail_and_initialized,
    is_main_process,
    get_rank,
    get_world_size,
    find_free_port,
)



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
        "--use-8bit-optimizer",
        "--use_8bit_optimizer",
        "--8bit-optimizer",
        "--8bit",
        dest="use_8bit_optimizer",
        action="store_true",
        default=False,
        help="Enable 8-bit AdamW optimizer via bitsandbytes (saves 75%% optimizer VRAM)",
    )
    parser.add_argument(
        "--check-8bit",
        "--check_8bit",
        dest="check_8bit",
        action="store_true",
        default=False,
        help="Check 8-bit hardware and library compatibility and print diagnostic report before training",
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
        help="Path to checkpoint .pt file or Hugging Face model repo (e.g. 'hf://owner/repo') to resume training from",
    )
    parser.add_argument(
        "--resume-repo",
        "--resume_repo",
        type=str,
        default=None,
        help="Explicit Hugging Face model repository ID or URI (e.g. 'hf://KhangTruong/sid-unet:checkpoint_best.pt') to resume from.",
    )
    parser.add_argument(
        "--hf-repo",
        "--hf_repo",
        "--hub-repo",
        "--hub_repo",
        type=str,
        default=None,
        help="Hugging Face model repository ID to checkpoint on (and auto-resume if existing checkpoint found), e.g. 'KhangTruong/Testing-model'.",
    )
    parser.add_argument(
        "--cached-hf-repo",
        "--cached_hf_repo",
        "--cached-repo",
        type=str,
        default=None,
        dest="cached_hf_repo",
        help="Hugging Face dataset repository ID or local path containing pre-computed cached forensic latents (e.g. 'KhangTruong/COCO-inpainted-cache').",
    )
    parser.add_argument(
        "--push-to-hub",
        "--push_to_hub",
        "--hf-checkpoint",
        "--hf_checkpoint",
        dest="push_to_hub",
        action="store_true",
        default=None,
        help="Enable checkpointing directly to Hugging Face Hub during training (warning emitted if flag is not on).",
    )
    parser.add_argument(
        "--no-push-to-hub",
        "--no_push_to_hub",
        "--no-hf-checkpoint",
        "--no_hf_checkpoint",
        dest="push_to_hub",
        action="store_false",
        help="Disable checkpointing to Hugging Face Hub even if repository is configured.",
    )
    parser.add_argument(
        "--hf-version",
        "--hf_version",
        "--hub-version",
        "--hub_version",
        type=str,
        default="v1",
        help="Version identifier for Hugging Face Hub checkpoints (default: 'v1').",
    )
    parser.add_argument(
        "--auto-resume",
        "--auto_resume",
        dest="auto_resume",
        action="store_true",
        default=True,
        help="Automatically search the repository and output directories for existing checkpoints to resume from (default: True).",
    )
    parser.add_argument(
        "--no-auto-resume",
        "--no_auto_resume",
        "--no-resume",
        dest="auto_resume",
        action="store_false",
        help="Disable automatic checkpoint resumption.",
    )
    parser.add_argument(
        "--resume-lr-mode",
        "--resume_lr_mode",
        "--lr-resume-mode",
        type=str,
        default=None,
        choices=["auto", "reschedule", "restart", "cycle", "reset", "keep"],
        help="Strategy for learning rate and scheduler when resuming training (default: 'auto').",
    )
    parser.add_argument(
        "--resume-lr",
        "--resume_lr",
        type=float,
        default=None,
        help="Explicit starting/base learning rate when resuming training.",
    )
    parser.add_argument(
        "--resume-epoch",
        "--resume_epoch",
        "--start-epoch",
        "--start_epoch",
        type=int,
        default=None,
        help="Explicit epoch number to resume training from.",
    )
    parser.add_argument(
        "--resume-epoch-offset",
        "--resume_epoch_offset",
        type=int,
        default=None,
        help="Offset to add to checkpoint epoch when resuming (default: 0 to resume at checkpoint epoch).",
    )
    parser.add_argument(
        "--val-samples-per-epoch",
        "--val_samples_per_epoch",
        "--val-samples",
        "--val_samples",
        type=int,
        default=None,
        help="Limit number of validation samples evaluated per epoch (-1 for entire split)",
    )
    parser.add_argument(
        "--checkpoint-period",
        "--checkpoint_period",
        "--checkpoint-interval",
        "--checkpoint_interval",
        type=float,
        default=None,
        help="Interval in seconds (or hours if <= 24) to save periodic checkpoints (default: 3600s / 1 hour)",
    )
    parser.add_argument(
        "--checkpoint-steps",
        "--checkpoint_steps",
        "--checkpoint-interval-steps",
        "--checkpoint_interval_steps",
        type=int,
        default=None,
        help="Interval in training steps to save periodic checkpoints (e.g. 100 steps)",
    )
    parser.add_argument(
        "--save-latest",
        "--save_latest",
        dest="save_latest",
        action="store_true",
        default=None,
        help="Continuously save and maintain checkpoint_latest.pt (default: True)",
    )
    parser.add_argument(
        "--no-save-latest",
        "--no_save_latest",
        dest="save_latest",
        action="store_false",
        help="Disable maintaining checkpoint_latest.pt",
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
    # Debug mode flags (nn-toolbox)
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Enable debug mode using nn-toolbox to automatically diagnose model, signals, and optimizer.",
    )
    parser.add_argument(
        "--debug-mode",
        "--debug_mode",
        dest="debug_mode",
        type=str,
        choices=["light", "deep"],
        default=None,
        help="Diagnostic mode for nn-toolbox ('light' or 'deep'). Default: 'light' when --debug is passed.",
    )
    # Bootstrapping v1.0 flags
    parser.add_argument(
        "--run-bootstrap",
        "--run_bootstrap",
        "--bootstrap",
        dest="run_bootstrap",
        action="store_true",
        default=None,
        help="Enable Bootstrapping v1.0 kickstarting phase before normal training.",
    )
    parser.add_argument(
        "--no-run-bootstrap",
        "--no-bootstrap",
        dest="run_bootstrap",
        action="store_false",
        help="Disable Bootstrapping v1.0 kickstarting phase.",
    )
    parser.add_argument(
        "--bootstrap-epochs",
        dest="bootstrap_epochs",
        type=int,
        default=None,
        help="Number of epochs to run during Bootstrapping v1.0 kickstarting (e.g. 5).",
    )
    parser.add_argument(
        "--bootstrap-examples",
        "--bootstrap-samples",
        dest="bootstrap_examples",
        type=int,
        default=None,
        help="Number of samples to train on during Bootstrapping v1.0 (e.g. 512 or 2048).",
    )
    parser.add_argument(
        "--bootstrap-lr",
        "--bootstrap-learning-rate",
        dest="bootstrap_lr",
        type=float,
        default=None,
        help="Learning rate for Bootstrapping v1.0 optimizer.",
    )
    parser.add_argument(
        "--bootstrap-strategy",
        dest="bootstrap_strategy",
        type=str,
        choices=["channel_stream", "dimension_stream", "whole_layer", "auto", "backbone", "encoder", "except_head", "custom"],
        default=None,
        help="Freezing strategy during Bootstrapping v1.0 ('channel_stream', 'dimension_stream', 'whole_layer', 'auto', 'custom').",
    )
    parser.add_argument(
        "--bootstrap-stream-ratio",
        dest="bootstrap_stream_ratio",
        type=float,
        default=None,
        help="Active channel stream ratio during Bootstrapping v1.0 (e.g. 0.5 for first 50%% channels).",
    )
    parser.add_argument(
        "--bootstrap-init",
        "--bootstrap-initialization",
        dest="bootstrap_init",
        type=str,
        default=None,
        help="Initialization method for unfrozen components ('kaiming_normal', 'xavier_normal', 'none').",
    )
    parser.add_argument(
        "--bootstrap-target-score",
        dest="bootstrap_target_score",
        type=float,
        default=None,
        help="Target score (IoU) to achieve before early releasing frozen parameters.",
    )
    # Hard Mining flags
    parser.add_argument(
        "--hard-mining",
        "--use-hard-mining",
        "--hard_mining",
        dest="hard_mining",
        action="store_true",
        default=None,
        help="Enable hard example mining from epoch 2 based on loss >= median (default: False)",
    )
    parser.add_argument(
        "--no-hard-mining",
        dest="hard_mining",
        action="store_false",
        help="Disable hard example mining",
    )
    parser.add_argument(
        "--hard-mining-epochs",
        "--hard-mining-reset-epochs",
        dest="hard_mining_reset_epochs",
        type=int,
        default=None,
        help="Number of epochs of hard mining before forgetting and repeating full epoch (default: 5)",
    )
    parser.add_argument(
        "--hard-mining-metric",
        dest="hard_mining_metric",
        type=str,
        default=None,
        help="Criterion for hard mining ('median' or 'mean', default: 'median')",
    )
    # Data Parallelism flags
    parser.add_argument(
        "--data-parallel",
        "--data_parallel",
        dest="data_parallel",
        action="store_true",
        default=None,
        help="Enable multi-GPU DataParallel across all available GPUs (default: enabled if multiple GPUs exist)",
    )
    parser.add_argument(
        "--no-data-parallel",
        "--no_data_parallel",
        dest="data_parallel",
        action="store_false",
        help="Disable multi-GPU DataParallel and train on single device",
    )
    return parser.parse_args()


def train_single_run(
    config_path: str,
    overrides: Optional[List[str]] = None,
    resume: Optional[str] = None,
    resume_repo: Optional[str] = None,
    auto_resume: bool = True,
    run_idx: int = 1,
    total_runs: int = 1,
    base_output_dir: Optional[str] = None,
    skip_collision: bool = True,
    cached_hf_repo: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single training experiment with its given config."""
    config = load_config(config_path, overrides=overrides or [])
    if cached_hf_repo:
        config.data.cached_hf_repo = cached_hf_repo

    # Set random seed
    seed = int(config.project.get("seed", 42))
    set_seed(seed)

    # Unification folder 'RUN': name each subfolder after its config (no numbering)
    cfg_stem = os.path.splitext(os.path.basename(config_path))[0]
    run_name = cfg_stem

    if base_output_dir:
        norm = os.path.normpath(base_output_dir)
        base_name = os.path.basename(norm)
        if base_name == "RUN":
            run_root = base_output_dir
            output_dir = os.path.join(run_root, cfg_stem)
        elif base_name == cfg_stem:
            output_dir = base_output_dir
        else:
            run_root = os.path.join(base_output_dir, "RUN")
            output_dir = os.path.join(run_root, cfg_stem)
    else:
        # Check if project.output_dir was explicitly provided via overrides (e.g. in targeted tests)
        override_output_dir = None
        if overrides:
            for ov in overrides:
                if ov.startswith("project.output_dir="):
                    override_output_dir = ov.split("=", 1)[1].strip()
                    break
        if override_output_dir:
            output_dir = override_output_dir
        else:
            run_root = os.path.join("outputs", "RUN")
            output_dir = os.path.join(run_root, cfg_stem)

    config.project.output_dir = output_dir
    config.project.name = run_name

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

    # Hugging Face checkpointing configuration & upfront verification
    hf_repo = (
        config.training.get("hub_repo")
        or config.training.get("hf_repo")
        or config.project.get("hub_repo")
        or config.project.get("hf_repo")
    )
    push_to_hub = bool(
        config.training.get("push_to_hub", False)
        or (hf_repo is not None and config.training.get("push_to_hub") is not False)
    )
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not push_to_hub:
        warn_msg = (
            "⚠️ [HF CHECKPOINT] Hugging Face model checkpointing is not enabled. "
            "Checkpoints will only be saved locally. "
            "To enable Hugging Face checkpointing, pass --hf-repo <repo_id> or --push-to-hub."
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=2)
        logger.warning(warn_msg)
    else:
        if not hf_repo:
            warn_msg = (
                "⚠️ [HF CHECKPOINT] Hugging Face checkpointing flag is enabled, but no Hugging Face repository was specified. "
                "Checkpoints will only be saved locally. Please provide --hf-repo <repo_id>."
            )
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
            logger.warning(warn_msg)
        else:
            # Verify if the repo is checkpointable upfront ("on the first hand")
            logger.info(f"Verifying if Hugging Face repository '{hf_repo}' is checkpointable...")
            from sid_unet.checkpoint_sync import verify_hf_repo_checkpointable
            verify_hf_repo_checkpointable(hf_repo, token=hf_token, create_if_missing=True)
            logger.info(f"✅ Hugging Face repository '{hf_repo}' is verified checkpointable.")

    # Checkpoint resolution:
    # 1. Explicit Hugging Face repo or URI (--resume-repo or --resume hf://...)
    # 2. Explicit local path (--resume path/to/ckpt.pt)
    # 3. Config project.resume_repo or training.resume_repo
    # 4. Hub repository (--hf-repo) if existing checkpoint exists
    # 5. Automatic search in repo and output directories (if auto_resume=True)
    resume_target = resume or resume_repo or config.project.get("resume_repo") or config.training.get("resume_repo")
    if not resume_target and auto_resume and hf_repo:
        resume_target = hf_repo

    resume_info = None

    if resume_target:
        if is_hf_repo_id(str(resume_target)):
            logger.info(f"Resolving checkpoint from Hugging Face model repository '{resume_target}'...")
            try:
                hf_data = download_hf_checkpoint(str(resume_target))
                resume = hf_data["checkpoint_path"]
                resume_info = {**inspect_checkpoint(resume), **hf_data}
            except FileNotFoundError as fnf_err:
                if (resume and is_hf_repo_id(str(resume))) or resume_repo:
                    logger.error(f"Failed to download checkpoint from Hugging Face repo '{resume_target}': {fnf_err}")
                    raise fnf_err
                else:
                    logger.info(
                        f"ℹ️ [AUTO-RESUME] No existing checkpoint found in Hugging Face repository '{resume_target}'. "
                        "Starting fresh training from epoch 1 and checkpointing to this repository."
                    )
                    resume = None
                    resume_info = None
            except Exception as e:
                logger.error(f"Failed to download checkpoint from Hugging Face repo '{resume_target}': {e}")
                raise e
        elif os.path.exists(str(resume_target)):
            resume = str(resume_target)
            resume_info = {"checkpoint_path": resume, "source": "local_repo", **inspect_checkpoint(resume)}
        else:
            raise FileNotFoundError(f"Specified checkpoint or repo not found: '{resume_target}'")
    elif auto_resume:
        found = find_auto_resume_checkpoint(
            output_dir=output_dir,
            config_stem=cfg_stem,
            repo_root=".",
        )
        if found:
            resume = found["checkpoint_path"]
            resume_info = found

    # On-screen and logger notification
    if is_main_process():
        if resume_info:
            msg = format_resume_notification(resume_info)
            print(msg)
            logger.info(msg)
        elif auto_resume:
            msg = format_no_resume_notification(output_dir)
            print(msg)
            logger.info(msg)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    latest_ckpt = os.path.join(ckpt_dir, "checkpoint_latest.pt")
    best_ckpt = os.path.join(ckpt_dir, "checkpoint_best.pt")

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

    # If training with cached representations, bypass heavy diffuser UNet to save VRAM and latency
    if getattr(config.data, "cached_hf_repo", None):
        raw_m = getattr(trainer, "raw_model", getattr(trainer, "model", None))
        if hasattr(raw_m, "bypass_diffuser_for_cached_training"):
            raw_m.bypass_diffuser_for_cached_training()
            if is_main_process():
                logger.info(
                    f"⚡ [DATASET CACHE] High-dimensional latent training active with repository '{config.data.cached_hf_repo}'. "
                    "Diffuser UNet bypassed for extreme throughput!"
                )

    # Resume training state if checkpoint found
    if resume:
        trainer.resume_from_checkpoint(resume)

    # Run training
    results = trainer.train()
    results["config_path"] = config_path
    results["run_name"] = run_name

    if is_main_process():
        logger.info(f"Experiment '{results['run_name']}' finished successfully!")
        logger.info(f"Best Score: {results['best_score']:.4f} (Epoch {results['best_epoch']})")
        logger.info(f"Evaluation report: {results['report_path']}")

    return results


def main():
    import signal
    if hasattr(signal, "SIGHUP"):
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except Exception:
            pass

    args = parse_args()

    if getattr(args, "check_8bit", False):
        from sid_unet.utils.compatibility import check_8bit_compatibility, format_compatibility_table
        _, details = check_8bit_compatibility(verbose=False)
        print("\n" + format_compatibility_table(details) + "\n")
        return

    config_paths = args.config if isinstance(args.config, list) else [args.config]

    overrides = list(args.override)
    if getattr(args, "use_8bit_optimizer", False):
        overrides.append("training.use_8bit_optimizer=true")
        overrides.append("training.optimizer=adamw8bit")
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
    if args.val_samples_per_epoch is not None:
        overrides.append(f"data.val_samples_per_epoch={args.val_samples_per_epoch}")
        overrides.append(f"data.val_samples={args.val_samples_per_epoch}")
    if args.checkpoint_period is not None:
        overrides.append(f"training.checkpoint_period={args.checkpoint_period}")
    if args.checkpoint_steps is not None:
        overrides.append(f"training.checkpoint_steps={args.checkpoint_steps}")
    if args.save_latest is True:
        overrides.append("training.save_latest=true")
    elif args.save_latest is False:
        overrides.append("training.save_latest=false")
    if getattr(args, "resume_epoch", None) is not None:
        overrides.append(f"training.resume_epoch={args.resume_epoch}")
    if getattr(args, "resume_epoch_offset", None) is not None:
        overrides.append(f"training.resume_epoch_offset={args.resume_epoch_offset}")
    if getattr(args, "debug", False) or getattr(args, "debug_mode", None):
        dbg_mode = args.debug_mode or "light"
        overrides.append(f"training.debug_mode={dbg_mode}")
    if getattr(args, "run_bootstrap", None) is True:
        overrides.append("bootstrapping.enabled=true")
        overrides.append("bootstrapping.run_bootstrap=true")
    elif getattr(args, "run_bootstrap", None) is False:
        overrides.append("bootstrapping.enabled=false")
        overrides.append("bootstrapping.run_bootstrap=false")
    if getattr(args, "bootstrap_epochs", None) is not None:
        overrides.append(f"bootstrapping.epochs={args.bootstrap_epochs}")
    if getattr(args, "bootstrap_examples", None) is not None:
        overrides.append(f"bootstrapping.num_samples={args.bootstrap_examples}")
    if getattr(args, "bootstrap_lr", None) is not None:
        overrides.append(f"bootstrapping.learning_rate={args.bootstrap_lr}")
    if getattr(args, "bootstrap_strategy", None) is not None:
        overrides.append(f"bootstrapping.freeze_strategy={args.bootstrap_strategy}")
    if getattr(args, "bootstrap_stream_ratio", None) is not None:
        overrides.append(f"bootstrapping.stream_ratio={args.bootstrap_stream_ratio}")
    if getattr(args, "bootstrap_init", None) is not None:
        overrides.append(f"bootstrapping.initialization={args.bootstrap_init}")
    if getattr(args, "bootstrap_target_score", None) is not None:
        overrides.append(f"bootstrapping.target_score={args.bootstrap_target_score}")
    if getattr(args, "hard_mining", None) is True:
        overrides.append("hard_mining.enabled=true")
        overrides.append("training.use_hard_mining=true")
    elif getattr(args, "hard_mining", None) is False:
        overrides.append("hard_mining.enabled=false")
        overrides.append("training.use_hard_mining=false")
    if getattr(args, "hard_mining_reset_epochs", None) is not None:
        overrides.append(f"hard_mining.reset_epochs={args.hard_mining_reset_epochs}")
    if getattr(args, "hard_mining_metric", None) is not None:
        overrides.append(f"hard_mining.metric={args.hard_mining_metric}")
    if getattr(args, "data_parallel", None) is True:
        overrides.append("training.data_parallel=true")
    elif getattr(args, "data_parallel", None) is False:
        overrides.append("training.data_parallel=false")
    if getattr(args, "resume_lr_mode", None) is not None:
        overrides.append(f"training.resume_lr_mode={args.resume_lr_mode}")
    if getattr(args, "resume_lr", None) is not None:
        overrides.append(f"training.resume_lr={args.resume_lr}")
    if getattr(args, "hf_repo", None) is not None:
        overrides.append(f"training.hub_repo={args.hf_repo}")
        overrides.append(f"training.hf_repo={args.hf_repo}")
    if getattr(args, "push_to_hub", None) is True:
        overrides.append("training.push_to_hub=true")
    elif getattr(args, "push_to_hub", None) is False:
        overrides.append("training.push_to_hub=false")
    elif getattr(args, "hf_repo", None) is not None:
        overrides.append("training.push_to_hub=true")
    if getattr(args, "hf_version", None) is not None:
        overrides.append(f"training.hub_version={args.hf_version}")

    # Check if we should auto-launch multi-process DDP via torchrun
    dp_disabled = (getattr(args, "data_parallel", None) is False) or any(
        ov.startswith("training.data_parallel=false") or ov.startswith("project.device=cpu") or ov.startswith("training.data_parallel=False") for ov in overrides
    )
    in_dist = ("RANK" in os.environ and "WORLD_SIZE" in os.environ) or is_dist_avail_and_initialized()
    in_pytest = ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)
    no_autospawn = bool(os.environ.get("SID_UNET_NO_AUTOSPAWN", False))

    # Pre-flight check: warn if other sid_unet processes are currently running
    if torch.cuda.is_available() and not in_dist and not in_pytest:
        try:
            from sid_unet.kill import find_unet_processes
            other_tasks = [p for p in find_unet_processes(include_children=False) if p["pid"] != os.getpid()]
            if other_tasks:
                pids_str = ", ".join(str(p["pid"]) for p in other_tasks[:3])
                print(
                    f"⚠️ WARNING: Found {len(other_tasks)} other active/background sid_unet process(es) "
                    f"(PID {pids_str}). If you encounter CUDA Out of Memory, run 'sid-kill' to terminate background tasks."
                )
                sys.stdout.flush()
        except Exception:
            pass

    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and not dp_disabled
        and not in_dist
        and not in_pytest
        and not no_autospawn
    ):
        import signal
        import subprocess
        num_gpus = torch.cuda.device_count()
        port = find_free_port()
        cmd = [
            sys.executable,
            "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port", str(port),
            "-m", "sid_unet.train",
        ] + sys.argv[1:]
        print(f"🚀 Auto-launching High-Performance Multi-GPU DistributedDataParallel (DDP) across {num_gpus} GPUs on port {port}...")
        sys.stdout.flush()
        sub_kwargs = {"start_new_session": True} if os.name != "nt" else {}
        proc = subprocess.Popen(cmd, **sub_kwargs)

        def _forward_kill(signum=None, frame=None):
            if proc.poll() is None:
                try:
                    if os.name != "nt":
                        os.killpg(proc.pid, signal.SIGTERM)
                        time.sleep(0.5)
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.terminate()
                except OSError:
                    pass
            if signum is not None:
                sys.exit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, _forward_kill)
            except (ValueError, OSError):
                pass

        try:
            ret = proc.wait()
            sys.exit(ret)
        finally:
            _forward_kill()

    # Initialize distributed mode if running under torchrun / distributed launcher
    init_distributed_mode()

    try:
        if len(config_paths) == 1:
            if args.output_dir:
                overrides.append(f"project.output_dir={args.output_dir}")
            results = train_single_run(
                config_path=config_paths[0],
                overrides=overrides,
                resume=args.resume,
                resume_repo=args.resume_repo,
                auto_resume=args.auto_resume,
                run_idx=1,
                total_runs=1,
                base_output_dir=args.output_dir,
                skip_collision=args.skip_collision,
                cached_hf_repo=getattr(args, "cached_hf_repo", None),
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

        norm = os.path.normpath(parent_output_dir)
        suite_run_dir = parent_output_dir if os.path.basename(norm) == "RUN" else os.path.join(parent_output_dir, "RUN")

        if is_main_process():
            print("\n" + "=" * 70)
            print(f"🚀 Launching Multi-Experiment Suite ({len(config_paths)} experiments)")
            print(f"📁 Suite Output Directory: {suite_run_dir}")
            print(f"⚡ Collision Detection / Skip: {args.skip_collision}")
            print("=" * 70 + "\n")

        all_results: List[Dict[str, Any]] = []

        for i, cfg_path in enumerate(config_paths, 1):
            cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
            if is_main_process():
                print(f"\n>>> Running Experiment [{i}/{len(config_paths)}]: {cfg_path} ({cfg_name})")
                print("-" * 70)
            res = train_single_run(
                config_path=cfg_path,
                overrides=overrides,
                resume=args.resume if i == 1 else None,
                resume_repo=args.resume_repo if i == 1 else None,
                auto_resume=args.auto_resume,
                run_idx=i,
                total_runs=len(config_paths),
                base_output_dir=suite_run_dir,
                skip_collision=args.skip_collision,
                cached_hf_repo=getattr(args, "cached_hf_repo", None),
            )
            all_results.append(res)

        multi_curves_path = None
        multi_report = None
        if is_main_process():
            # Collect experiment histories and plot multi-run comparison curves
            histories_dict = {}
            for r in all_results:
                exp_name = r.get("run_name", "Run")
                if r.get("history"):
                    histories_dict[exp_name] = r["history"]

            if histories_dict:
                multi_curves_path = os.path.join(suite_run_dir, "multi_experiment_curves.png")
                plot_multi_experiment_curves(
                    experiment_histories=histories_dict,
                    output_path=multi_curves_path,
                )

            # Generate and display continuous multi-experiment comparison report
            combined_results = list(all_results)
            multi_json_path = os.path.join(suite_run_dir, "multi_experiment_comparison.json")
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
                output_dir=suite_run_dir,
                report_name="multi_experiment_comparison",
                multi_curves_path=multi_curves_path,
            )

            print("\n" + "=" * 70)
            print("⭐ ALL EXPERIMENTS COMPLETED - SUMMARY REPORT")
            print("=" * 70)
            print(multi_report["summary_table"])
            if multi_curves_path and os.path.exists(multi_curves_path):
                print(f"Multi-Experiment comparison curves plot: {multi_curves_path}")
            print(f"\nDetailed Markdown comparison: {os.path.join(suite_run_dir, 'multi_experiment_comparison.md')}")
            print(f"Detailed JSON comparison: {os.path.join(suite_run_dir, 'multi_experiment_comparison.json')}\n")

        return all_results
    finally:
        cleanup_distributed()



def cli_main():
    import sys
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli_main()
