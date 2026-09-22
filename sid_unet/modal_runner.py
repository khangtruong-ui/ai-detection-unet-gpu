"""
Modal integration and remote execution engine for SID-UNet.

Features:
- Modal authentication detection (exits execution if unauthenticated)
- Automated Modal Volume provisioning (creates volume if not already existing)
- Default execution on Modal when no local CUDA GPU is available
- Cheap GPU allocation for testing and mock tests (default: T4)
- Structured output directory organization inside persistent Modal volume (/vol)
- CLI and programmatic entrypoints for training, evaluation, and test suites
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import modal

# Constants
DEFAULT_VOLUME_NAME = "sid-unet-data"
DEFAULT_MOUNT_PATH = "/vol"
DEFAULT_OUTPUT_DIR = f"{DEFAULT_MOUNT_PATH}/outputs"
DEFAULT_RUNS_DIR = f"{DEFAULT_MOUNT_PATH}/outputs/RUN"
DEFAULT_TEST_DIR = f"{DEFAULT_MOUNT_PATH}/test_outputs"

# Cost-effective GPUs on Modal:
# - L40S: ~$1.95/hr, 733 BF16 TFLOPS ($0.00266/TFLOP - cheapest price over TFLOPS under $2/hr)
# - L4:   ~$0.80/hr, 242 BF16 TFLOPS ($0.00331/TFLOP - ultra low cost Ada Lovelace)
# - T4:   ~$0.59/hr, cheap testing / mock test GPU
DEFAULT_TRAIN_GPU = "L40S"
DEFAULT_TEST_GPU = "T4"
DEFAULT_HF_SECRET_NAME = "huggingface"

# Repo root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_hf_token_status() -> Dict[str, Any]:
    """Inspect Hugging Face authentication availability locally and on Modal."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    source = "environment (HF_TOKEN)" if token else None

    if not token:
        cache_token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(cache_token_path):
            try:
                with open(cache_token_path, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                    if t:
                        token = t
                        source = f"cache ({cache_token_path})"
            except Exception:
                pass

    return {
        "authenticated": bool(token),
        "source": source or f"Modal Secret ('{DEFAULT_HF_SECRET_NAME}')",
        "secret_name": DEFAULT_HF_SECRET_NAME,
        "token_preview": f"{token[:6]}...{token[-4:]}" if token and len(token) > 10 else ("***" if token else None),
    }


def get_hf_secret(secret_name: str = DEFAULT_HF_SECRET_NAME) -> modal.Secret:
    """Return Modal Secret reference for Hugging Face authentication."""
    return modal.Secret.from_name(secret_name)


def is_modal_authenticated() -> bool:
    """
    Check if Modal credentials are valid and configured.
    Checks modal.config, active profile, and environment variables.
    """
    try:
        import modal.config

        cfg = getattr(modal.config, "config", None)
        token_id = cfg.get("token_id") if cfg is not None else None
        token_secret = cfg.get("token_secret") if cfg is not None else None

        # Fallback to standard Modal environment variables
        token_id = token_id or os.environ.get("MODAL_TOKEN_ID")
        token_secret = token_secret or os.environ.get("MODAL_TOKEN_SECRET")

        if not (token_id and token_secret):
            return False

        # Check active profile if available
        try:
            profile = modal.config._profile.get_active()
            if not profile:
                return False
        except Exception:
            pass

        return True
    except Exception:
        return False


def ensure_modal_authenticated(exit_on_failure: bool = True) -> bool:
    """
    Detect if Modal is authenticated.
    If not authenticated, prints actionable instructions and exits execution.

    Args:
        exit_on_failure: If True, calls sys.exit(1) on failure.

    Returns:
        True if authenticated, False otherwise.
    """
    if not is_modal_authenticated():
        msg = (
            "\n"
            + "=" * 76 + "\n"
            + "❌ ERROR: Modal authentication required but not detected!\n"
            + "=" * 76 + "\n"
            + "To authenticate Modal, please run:\n"
            + "    modal setup\n\n"
            + "Or configure the environment variables:\n"
            + "    export MODAL_TOKEN_ID='ak-...'\n"
            + "    export MODAL_TOKEN_SECRET='as-...'\n"
            + "=" * 76 + "\n"
        )
        sys.stderr.write(msg)
        sys.stderr.flush()
        if exit_on_failure:
            sys.exit(1)
        return False
    return True


def has_local_gpu() -> bool:
    """
    Detect if a usable local CUDA GPU is available.
    Returns False if torch is uninstalled or CUDA is unavailable.
    """
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def get_or_create_volume(
    volume_name: str = DEFAULT_VOLUME_NAME,
    client: Optional[Any] = None,
) -> modal.Volume:
    """
    Reference a Modal Volume, creating it on Modal if it does not already exist.

    Args:
        volume_name: Name of the volume (default: 'sid-unet-data').
        client: Optional Modal client instance.

    Returns:
        modal.Volume handle.
    """
    ensure_modal_authenticated(exit_on_failure=True)
    try:
        modal.Volume.objects.create(volume_name, allow_existing=True, client=client)
    except Exception:
        pass
    return modal.Volume.from_name(volume_name, create_if_missing=True, client=client)


def get_volume_output_dir(
    subpath: Optional[str] = None,
    volume_mount_path: str = DEFAULT_MOUNT_PATH,
    default_subpath: str = "outputs/RUN",
) -> str:
    """
    Organize and resolve the default output directory within the Modal volume.

    Volume organization structure:
      /vol/
        outputs/
          RUN/
            {config_stem}/
              checkpoints/
              reports/
              illustrations/
              logs/
            multi_experiment_comparison.md
            multi_experiment_comparison.json
        test_outputs/
          mock_run/
          pytest_reports/

    Args:
        subpath: Optional user-specified output path.
        volume_mount_path: Base mount point of the volume inside container (default: /vol).
        default_subpath: Default relative subpath if none specified.

    Returns:
        Absolute normalized container path within the volume mount.
    """
    if not subpath:
        return os.path.normpath(f"{volume_mount_path}/{default_subpath}")

    norm = os.path.normpath(subpath)
    mount_norm = os.path.normpath(volume_mount_path)
    if norm.startswith(mount_norm):
        return norm

    # Clean leading slashes and nest within volume mount
    clean_sub = subpath.lstrip("/").lstrip("\\")
    return os.path.normpath(f"{mount_norm}/{clean_sub}")


# Define Modal container image with full PyTorch/CUDA environment
def create_modal_image() -> modal.Image:
    """Construct the Debian-based Modal container image with CUDA dependencies."""
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libgl1", "libglib2.0-0")
        .pip_install(
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "datasets>=2.14.0",
            "huggingface_hub>=0.20.0",
            "numpy>=1.22.0",
            "pillow>=9.0.0",
            "pyyaml>=6.0",
            "tqdm>=4.65.0",
            "scikit-learn>=1.0.0",
            "tabulate>=0.9.0",
            "matplotlib>=3.5.0",
            "diffusers>=0.25.0",
            "accelerate>=0.25.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "bitsandbytes>=0.41.0",
        )
        .env({
            "MODAL_LOG_FORMAT": "PLAIN",
            "SID_PROGRESS_MODE": "clean",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/vol/cache/huggingface",
            "HF_DATASETS_CACHE": "/vol/cache/huggingface/datasets",
        })
    )
    # Add source code and configs
    if (PROJECT_ROOT / "sid_unet").is_dir():
        image = image.add_local_python_source("sid_unet")
    if (PROJECT_ROOT / "configs").is_dir():
        image = image.add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/root/configs")
    if (PROJECT_ROOT / "tests").is_dir():
        image = image.add_local_dir(str(PROJECT_ROOT / "tests"), remote_path="/root/tests")
    return image


# Modal App definition
app = modal.App("sid-unet")
image = create_modal_image()
volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)
hf_secret = get_hf_secret(DEFAULT_HF_SECRET_NAME)


# ==============================================================================
# Modal Remote Functions
# ==============================================================================

@app.function(
    image=image,
    gpu="L40S",  # Cheapest price over TFLOPS under $2/hr: $1.95/hr for 733 BF16 TFLOPS ($0.00266/TFLOP)
    volumes={DEFAULT_MOUNT_PATH: volume},
    secrets=[hf_secret],
    timeout=86400,
)
def train_remote_l40s(
    config_paths: List[str],
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    resume: Optional[str] = None,
    resume_repo: Optional[str] = None,
    auto_resume: bool = True,
    skip_collision: bool = True,
    save_latest: Optional[bool] = None,
    batch_size: Optional[int] = None,
    auto_batch_size: Optional[bool] = None,
    val_samples_per_epoch: Optional[int] = None,
    checkpoint_period: Optional[float] = None,
    checkpoint_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Execute training on Modal with an L40S GPU (cheapest price/TFLOPS, ~$1.95/hr)."""
    return _execute_train_remote(
        config_paths=config_paths,
        overrides=overrides,
        output_dir=output_dir,
        resume=resume,
        resume_repo=resume_repo,
        auto_resume=auto_resume,
        skip_collision=skip_collision,
        save_latest=save_latest,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
        val_samples_per_epoch=val_samples_per_epoch,
        checkpoint_period=checkpoint_period,
        checkpoint_steps=checkpoint_steps,
    )


@app.function(
    image=image,
    gpu="L4",  # High cost-efficiency Ada Lovelace GPU ($0.80/hr for 242 BF16 TFLOPS = $0.00331/TFLOP)
    volumes={DEFAULT_MOUNT_PATH: volume},
    secrets=[hf_secret],
    timeout=86400,
)
def train_remote_l4(
    config_paths: List[str],
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    resume: Optional[str] = None,
    resume_repo: Optional[str] = None,
    auto_resume: bool = True,
    skip_collision: bool = True,
    save_latest: Optional[bool] = None,
    batch_size: Optional[int] = None,
    auto_batch_size: Optional[bool] = None,
    val_samples_per_epoch: Optional[int] = None,
    checkpoint_period: Optional[float] = None,
    checkpoint_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Execute training on Modal with an L4 GPU (~$0.80/hr)."""
    return _execute_train_remote(
        config_paths=config_paths,
        overrides=overrides,
        output_dir=output_dir,
        resume=resume,
        resume_repo=resume_repo,
        auto_resume=auto_resume,
        skip_collision=skip_collision,
        save_latest=save_latest,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
        val_samples_per_epoch=val_samples_per_epoch,
        checkpoint_period=checkpoint_period,
        checkpoint_steps=checkpoint_steps,
    )


@app.function(
    image=image,
    gpu="A10G",
    volumes={DEFAULT_MOUNT_PATH: volume},
    secrets=[hf_secret],
    timeout=86400,
)
def train_remote_a10g(
    config_paths: List[str],
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    resume: Optional[str] = None,
    resume_repo: Optional[str] = None,
    auto_resume: bool = True,
    skip_collision: bool = True,
    save_latest: Optional[bool] = None,
    batch_size: Optional[int] = None,
    auto_batch_size: Optional[bool] = None,
    val_samples_per_epoch: Optional[int] = None,
    checkpoint_period: Optional[float] = None,
    checkpoint_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Execute training on Modal with an A10G GPU."""
    return _execute_train_remote(
        config_paths=config_paths,
        overrides=overrides,
        output_dir=output_dir,
        resume=resume,
        resume_repo=resume_repo,
        auto_resume=auto_resume,
        skip_collision=skip_collision,
        save_latest=save_latest,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
        val_samples_per_epoch=val_samples_per_epoch,
        checkpoint_period=checkpoint_period,
        checkpoint_steps=checkpoint_steps,
    )


@app.function(
    image=image,
    gpu=DEFAULT_TEST_GPU,  # Cheap T4 GPU for testing/smoke runs
    volumes={DEFAULT_MOUNT_PATH: volume},
    secrets=[hf_secret],
    timeout=43200,
)
def train_remote_t4(
    config_paths: List[str],
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    resume: Optional[str] = None,
    resume_repo: Optional[str] = None,
    auto_resume: bool = True,
    skip_collision: bool = True,
    save_latest: Optional[bool] = None,
    batch_size: Optional[int] = None,
    auto_batch_size: Optional[bool] = None,
    val_samples_per_epoch: Optional[int] = None,
    checkpoint_period: Optional[float] = None,
    checkpoint_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Execute training on Modal with a cheap T4 GPU."""
    return _execute_train_remote(
        config_paths=config_paths,
        overrides=overrides,
        output_dir=output_dir,
        resume=resume,
        resume_repo=resume_repo,
        auto_resume=auto_resume,
        skip_collision=skip_collision,
        save_latest=save_latest,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
        val_samples_per_epoch=val_samples_per_epoch,
        checkpoint_period=checkpoint_period,
        checkpoint_steps=checkpoint_steps,
    )


def _execute_train_remote(
    config_paths: List[str],
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    resume: Optional[str] = None,
    resume_repo: Optional[str] = None,
    auto_resume: bool = True,
    skip_collision: bool = True,
    save_latest: Optional[bool] = None,
    batch_size: Optional[int] = None,
    auto_batch_size: Optional[bool] = None,
    val_samples_per_epoch: Optional[int] = None,
    checkpoint_period: Optional[float] = None,
    checkpoint_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Internal implementation of remote training execution in Modal container."""
    from sid_unet.train import train_single_run
    from sid_unet.utils.plotting import plot_multi_experiment_curves
    from sid_unet.utils.report import generate_multi_experiment_report

    # Resolve output directory inside Modal volume
    suite_run_dir = get_volume_output_dir(output_dir, volume_mount_path=DEFAULT_MOUNT_PATH, default_subpath="outputs/RUN")
    os.makedirs(suite_run_dir, exist_ok=True)

    # Configure Hugging Face authentication and persistent cache
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_cache = os.environ.get("HF_HOME", "/vol/cache/huggingface")
    os.makedirs(hf_cache, exist_ok=True)
    if hf_token:
        print(f"🔑 [Modal Container] Hugging Face authenticated via HF_TOKEN (cache: {hf_cache})")
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass
    else:
        print(f"ℹ️ [Modal Container] Running without HF_TOKEN (cache: {hf_cache})")

    # Normalize config paths (check local or /root/configs)
    resolved_configs = []
    for cp in config_paths:
        if os.path.exists(cp):
            resolved_configs.append(cp)
        elif os.path.exists(f"/root/{cp}"):
            resolved_configs.append(f"/root/{cp}")
        elif os.path.exists(f"/root/configs/{os.path.basename(cp)}"):
            resolved_configs.append(f"/root/configs/{os.path.basename(cp)}")
        else:
            resolved_configs.append(cp)

    overrides_list = list(overrides or [])
    if batch_size is not None:
        overrides_list.append(f"data.batch_size={batch_size}")
    if auto_batch_size is True:
        overrides_list.append("training.auto_batch_size=true")
    elif auto_batch_size is False:
        overrides_list.append("training.auto_batch_size=false")
    if val_samples_per_epoch is not None:
        overrides_list.append(f"data.val_samples_per_epoch={val_samples_per_epoch}")
    if checkpoint_period is not None:
        overrides_list.append(f"training.checkpoint_period={checkpoint_period}")
    if checkpoint_steps is not None:
        overrides_list.append(f"training.checkpoint_steps={checkpoint_steps}")
    if save_latest is True:
        overrides_list.append("training.save_latest=true")
    elif save_latest is False:
        overrides_list.append("training.save_latest=false")

    all_results = []
    if len(resolved_configs) == 1:
        cfg = resolved_configs[0]
        cfg_stem = os.path.splitext(os.path.basename(cfg))[0]
        single_out = os.path.join(suite_run_dir, cfg_stem) if not suite_run_dir.endswith(cfg_stem) else suite_run_dir
        res = train_single_run(
            config_path=cfg,
            overrides=overrides_list,
            resume=resume,
            resume_repo=resume_repo,
            auto_resume=auto_resume,
            run_idx=1,
            total_runs=1,
            base_output_dir=single_out,
            skip_collision=skip_collision,
        )
        all_results.append(res)
    else:
        for i, cfg in enumerate(resolved_configs, 1):
            res = train_single_run(
                config_path=cfg,
                overrides=overrides_list,
                resume=resume if i == 1 else None,
                resume_repo=resume_repo if i == 1 else None,
                auto_resume=auto_resume,
                run_idx=i,
                total_runs=len(resolved_configs),
                base_output_dir=suite_run_dir,
                skip_collision=skip_collision,
            )
            all_results.append(res)

        # Multi-experiment summary
        histories_dict = {r["run_name"]: r["history"] for r in all_results if r.get("history") and r.get("run_name")}
        multi_curves_path = None
        if histories_dict:
            multi_curves_path = os.path.join(suite_run_dir, "multi_experiment_curves.png")
            plot_multi_experiment_curves(experiment_histories=histories_dict, output_path=multi_curves_path)

        generate_multi_experiment_report(
            experiment_results=all_results,
            output_dir=suite_run_dir,
            report_name="multi_experiment_comparison",
            multi_curves_path=multi_curves_path,
        )

    # Persist volume changes immediately
    try:
        vol = modal.Volume.from_name(DEFAULT_VOLUME_NAME)
        vol.commit()
    except Exception:
        pass

    return all_results


@app.function(
    image=image,
    gpu=DEFAULT_TEST_GPU,  # Cheap T4 GPU for evaluation / testing
    volumes={DEFAULT_MOUNT_PATH: volume},
    secrets=[hf_secret],
    timeout=7200,
)
def eval_remote_t4(
    checkpoint_paths: List[str],
    config_path: Optional[str] = None,
    split: Optional[str] = None,
    samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[str] = None,
    threshold: float = 0.5,
    min_area: int = 0,
    morphology: str = "none",
    segment: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Execute model evaluation on Modal with cheap T4 GPU."""
    from sid_unet.evaluate import evaluate_single_checkpoint

    # Configure Hugging Face authentication and persistent cache
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_cache = os.environ.get("HF_HOME", "/vol/cache/huggingface")
    os.makedirs(hf_cache, exist_ok=True)
    if hf_token:
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    # Resolve output directory inside Modal volume
    eval_out_dir = get_volume_output_dir(
        output_dir,
        volume_mount_path=DEFAULT_MOUNT_PATH,
        default_subpath="outputs/evaluations",
    )
    os.makedirs(eval_out_dir, exist_ok=True)

    results = []
    for ckpt in checkpoint_paths:
        # Check volume path vs container path
        actual_ckpt = ckpt
        if not os.path.exists(actual_ckpt):
            vol_candidate = get_volume_output_dir(ckpt, volume_mount_path=DEFAULT_MOUNT_PATH)
            if os.path.exists(vol_candidate):
                actual_ckpt = vol_candidate

        res = evaluate_single_checkpoint(
            checkpoint_path=actual_ckpt,
            config_path=config_path,
            split=split,
            samples=samples,
            batch_size=batch_size,
            threshold=threshold,
            min_area=min_area,
            morphology=morphology,
            segment=segment,
            overrides=overrides or [],
            output_dir=eval_out_dir,
        )
        results.append(res)

    try:
        vol = modal.Volume.from_name(DEFAULT_VOLUME_NAME)
        vol.commit()
    except Exception:
        pass

    return results


@app.function(
    image=image,
    gpu=DEFAULT_TEST_GPU,  # Cheap T4 GPU for testing
    volumes={DEFAULT_MOUNT_PATH: volume},
    secrets=[hf_secret],
    timeout=3600,
)
def run_pytest_remote_t4(
    pytest_args: Optional[List[str]] = None,
    output_subdir: str = "pytest_reports",
) -> Dict[str, Any]:
    """Execute pytest test suite on Modal using a cheap T4 GPU."""
    import subprocess

    test_out_dir = get_volume_output_dir(
        f"test_outputs/{output_subdir}",
        volume_mount_path=DEFAULT_MOUNT_PATH,
        default_subpath="test_outputs",
    )
    os.makedirs(test_out_dir, exist_ok=True)

    args = list(pytest_args or ["tests/test_config.py", "-v"])
    cmd = [sys.executable, "-m", "pytest"] + args

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Persist log to volume
    log_file = os.path.join(test_out_dir, "pytest_output.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Returncode: {result.returncode}\n\n")
        f.write("--- STDOUT ---\n")
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)

    try:
        vol = modal.Volume.from_name(DEFAULT_VOLUME_NAME)
        vol.commit()
    except Exception:
        pass

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "log_path": log_file,
    }


@app.function(
    image=image,
    gpu=DEFAULT_TEST_GPU,  # Cheap T4 GPU for mock tests
    volumes={DEFAULT_MOUNT_PATH: volume},
    secrets=[hf_secret],
    timeout=600,
)
def mock_test_remote_t4() -> Dict[str, Any]:
    """
    Execute a synthetic mock training and testing cycle on a cheap T4 GPU.
    Validates model instantiation, forward/backward pass, metric computation,
    and volume persistence.
    """
    import torch
    from sid_unet.models.unet import UNet
    from sid_unet.losses.combined import CombinedLoss
    from sid_unet.metrics.segmentation import SegmentationMetricTracker

    test_out_dir = get_volume_output_dir(
        "test_outputs/mock_test",
        volume_mount_path=DEFAULT_MOUNT_PATH,
        default_subpath="test_outputs/mock_test",
    )
    os.makedirs(test_out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, features=[16, 32]).to(device)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5, focal_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metric_tracker = SegmentationMetricTracker(threshold=0.5)

    # Run 2 synthetic steps
    model.train()
    loss_val = 0.0
    for _ in range(2):
        x = torch.randn(2, 3, 64, 64, device=device)
        y = (torch.rand(2, 1, 64, 64, device=device) > 0.5).float()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())

    # Evaluation step
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 64, 64, device=device)
        y = (torch.rand(2, 1, 64, 64, device=device) > 0.5).float()
        preds = torch.sigmoid(model(x))
        metric_tracker.update(preds, y)

    metrics = metric_tracker.compute()

    # Save mock checkpoint in volume
    ckpt_path = os.path.join(test_out_dir, "mock_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "loss": loss_val,
        },
        ckpt_path,
    )

    try:
        vol = modal.Volume.from_name(DEFAULT_VOLUME_NAME)
        vol.commit()
    except Exception:
        pass

    return {
        "status": "success",
        "device": str(device),
        "loss": loss_val,
        "metrics": metrics,
        "saved_checkpoint": ckpt_path,
    }


# ==============================================================================
# Client Launchers & Orchestration
# ==============================================================================

def run_train_on_modal(
    args: Any,
    volume_name: str = DEFAULT_VOLUME_NAME,
    gpu: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Launch training on Modal.
    Verifies authentication, provisions volume, and routes to appropriate GPU.
    """
    ensure_modal_authenticated(exit_on_failure=True)
    get_or_create_volume(volume_name=getattr(args, "modal_volume", None) or volume_name)

    # Propagate explicit CLI HF token to environment if provided
    cli_token = getattr(args, "hf_token", None)
    if cli_token:
        os.environ["HF_TOKEN"] = cli_token

    chosen_gpu = (getattr(args, "modal_gpu", None) or gpu or DEFAULT_TRAIN_GPU).upper()
    config_paths = args.config if isinstance(args.config, list) else [args.config]
    hf_info = check_hf_token_status()

    print(f"📦 Modal Volume: {getattr(args, 'modal_volume', None) or volume_name} (mount: {DEFAULT_MOUNT_PATH})")
    print(f"🖥️ Modal GPU: {chosen_gpu}")
    print(f"🔑 Hugging Face Secret: attached '{DEFAULT_HF_SECRET_NAME}' (HF_TOKEN)")
    if hf_info["authenticated"]:
        print(f"   Local HF token detected: yes ({hf_info['source']})")
    else:
        print(f"   Runtime HF token: loaded via Modal Secret '{DEFAULT_HF_SECRET_NAME}'")
    print(f"📋 Configs: {config_paths}")

    remote_map = {
        "L40S": train_remote_l40s,
        "L40": train_remote_l40s,
        "L4": train_remote_l4,
        "A10G": train_remote_a10g,
        "T4": train_remote_t4,
    }
    target_fn = remote_map.get(chosen_gpu, train_remote_l40s)

    with modal.enable_output():
        with app.run():
            res = target_fn.remote(
                config_paths=config_paths,
                overrides=getattr(args, "override", []),
                output_dir=getattr(args, "output_dir", None),
                resume=getattr(args, "resume", None),
                resume_repo=getattr(args, "resume_repo", None),
                auto_resume=getattr(args, "auto_resume", True),
                skip_collision=getattr(args, "skip_collision", True),
                save_latest=getattr(args, "save_latest", None),
                batch_size=getattr(args, "batch_size", None),
                auto_batch_size=getattr(args, "auto_batch_size", None),
                val_samples_per_epoch=getattr(args, "val_samples_per_epoch", None),
                checkpoint_period=getattr(args, "checkpoint_period", None),
                checkpoint_steps=getattr(args, "checkpoint_steps", None),
            )
    return res


def run_eval_on_modal(
    args: Any,
    volume_name: str = DEFAULT_VOLUME_NAME,
    gpu: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Launch evaluation/testing on Modal using a cheap GPU (T4).
    """
    ensure_modal_authenticated(exit_on_failure=True)
    get_or_create_volume(volume_name=getattr(args, "modal_volume", None) or volume_name)

    cli_token = getattr(args, "hf_token", None)
    if cli_token:
        os.environ["HF_TOKEN"] = cli_token

    chosen_gpu = (getattr(args, "modal_gpu", None) or gpu or DEFAULT_TEST_GPU).upper()
    ckpts = args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
    hf_info = check_hf_token_status()

    print(f"📦 Modal Volume: {getattr(args, 'modal_volume', None) or volume_name} (mount: {DEFAULT_MOUNT_PATH})")
    print(f"🖥️ Modal GPU (cheap for testing): {chosen_gpu}")
    print(f"🔑 Hugging Face Secret: attached '{DEFAULT_HF_SECRET_NAME}' (HF_TOKEN)")
    if hf_info["authenticated"]:
        print(f"   Local HF token detected: yes ({hf_info['source']})")
    else:
        print(f"   Runtime HF token: loaded via Modal Secret '{DEFAULT_HF_SECRET_NAME}'")
    print(f"🎯 Checkpoints: {ckpts}")

    with modal.enable_output():
        with app.run():
            res = eval_remote_t4.remote(
                checkpoint_paths=ckpts,
                config_path=getattr(args, "config", None),
                split=getattr(args, "split", None),
                samples=getattr(args, "samples", None),
                batch_size=getattr(args, "batch_size", None),
                output_dir=getattr(args, "output_dir", None),
                threshold=getattr(args, "threshold", 0.5),
                min_area=getattr(args, "min_area", 0),
                morphology=getattr(args, "morphology", "none"),
                segment=getattr(args, "segment", None),
                overrides=getattr(args, "override", []),
            )
    return res


def run_tests_on_modal(
    pytest_args: Optional[List[str]] = None,
    volume_name: str = DEFAULT_VOLUME_NAME,
    gpu: str = DEFAULT_TEST_GPU,
) -> Dict[str, Any]:
    """Launch pytest test suite on Modal using a cheap T4 GPU."""
    ensure_modal_authenticated(exit_on_failure=True)
    get_or_create_volume(volume_name=volume_name)

    print(f"📦 Modal Volume: {volume_name} (mount: {DEFAULT_MOUNT_PATH})")
    print(f"🖥️ Running pytest on cheap GPU: {gpu}")

    with modal.enable_output():
        with app.run():
            res = run_pytest_remote_t4.remote(pytest_args=pytest_args)
    return res


def run_mock_test_on_modal(
    volume_name: str = DEFAULT_VOLUME_NAME,
    gpu: str = DEFAULT_TEST_GPU,
) -> Dict[str, Any]:
    """Launch mock training/testing cycle on Modal with a cheap T4 GPU."""
    ensure_modal_authenticated(exit_on_failure=True)
    get_or_create_volume(volume_name=volume_name)

    print(f"📦 Modal Volume: {volume_name} (mount: {DEFAULT_MOUNT_PATH})")
    print(f"🖥️ Running synthetic mock test on cheap GPU: {gpu}")

    with modal.enable_output():
        with app.run():
            res = mock_test_remote_t4.remote()
    return res


# ==============================================================================
# CLI Entrypoints
# ==============================================================================

@app.local_entrypoint()
def local_entrypoint(
    train: bool = False,
    test: bool = False,
    mock: bool = False,
    config: str = "configs/test_smoke.yaml",
):
    """Direct local entrypoint when executed via `modal run sid_unet.modal_runner`."""
    if mock:
        print("Running mock test on Modal...")
        print(mock_test_remote_t4.remote())
    elif test:
        print("Running test suite on Modal...")
        print(run_pytest_remote_t4.remote())
    else:
        print(f"Running training on Modal with config: {config}...")
        print(train_remote_t4.remote(config_paths=[config]))


def cli_main():
    """Universal CLI dispatcher for Modal commands (`sid-modal`)."""
    parser = argparse.ArgumentParser(
        prog="sid-modal",
        description="Launch and manage SID-UNet training, testing, and mock tests on Modal.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Modal subcommand to execute")

    # Subcommand: auth-check
    subparsers.add_parser("auth-check", help="Verify if Modal is authenticated")

    # Subcommand: volume-info
    vol_parser = subparsers.add_parser("volume-info", help="Inspect or create Modal Volume")
    vol_parser.add_argument("--volume", type=str, default=DEFAULT_VOLUME_NAME, help="Modal volume name")

    # Subcommand: train
    train_p = subparsers.add_parser("train", help="Run training on Modal")
    train_p.add_argument("--config", "--configs", nargs="+", default=["configs/train_streaming.yaml"], help="Config file(s)")
    train_p.add_argument("--output-dir", "--output_dir", type=str, default=None, help="Output directory")
    train_p.add_argument("--gpu", type=str, default=DEFAULT_TRAIN_GPU, help="GPU type (e.g. A10G, T4)")
    train_p.add_argument("--volume", type=str, default=DEFAULT_VOLUME_NAME, help="Modal volume name")
    train_p.add_argument("--override", nargs="*", default=[], help="Config overrides")
    train_p.add_argument("--hf-token", "--hf_token", type=str, default=None, help="Hugging Face API token")

    # Subcommand: eval
    eval_p = subparsers.add_parser("eval", help="Run evaluation/testing on Modal (cheap T4 GPU)")
    eval_p.add_argument("--checkpoint", nargs="+", required=True, help="Checkpoint file(s) or glob")
    eval_p.add_argument("--config", type=str, default=None, help="Config YAML file")
    eval_p.add_argument("--split", type=str, default=None, help="Dataset split")
    eval_p.add_argument("--samples", type=int, default=None, help="Max evaluation samples")
    eval_p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    eval_p.add_argument("--gpu", type=str, default=DEFAULT_TEST_GPU, help="GPU type (default: cheap T4)")
    eval_p.add_argument("--volume", type=str, default=DEFAULT_VOLUME_NAME, help="Modal volume name")
    eval_p.add_argument("--hf-token", "--hf_token", type=str, default=None, help="Hugging Face API token")

    # Subcommand: test
    test_p = subparsers.add_parser("test", help="Run pytest test suite on Modal (cheap T4 GPU)")
    test_p.add_argument("pytest_args", nargs="*", default=["tests/test_config.py"], help="Pytest test targets/flags")
    test_p.add_argument("--gpu", type=str, default=DEFAULT_TEST_GPU, help="GPU type (default: cheap T4)")
    test_p.add_argument("--volume", type=str, default=DEFAULT_VOLUME_NAME, help="Modal volume name")

    # Subcommand: mock-test
    mock_p = subparsers.add_parser("mock-test", help="Run mock synthetic train/test cycle on Modal (cheap T4 GPU)")
    mock_p.add_argument("--gpu", type=str, default=DEFAULT_TEST_GPU, help="GPU type (default: cheap T4)")
    mock_p.add_argument("--volume", type=str, default=DEFAULT_VOLUME_NAME, help="Modal volume name")

    args = parser.parse_args()

    if args.command == "auth-check":
        is_auth = is_modal_authenticated()
        if is_auth:
            print("✅ Modal is authenticated and ready to run jobs.")
            hf_info = check_hf_token_status()
            print(f"🔑 Hugging Face Secret: attached '{DEFAULT_HF_SECRET_NAME}' (HF_TOKEN)")
            if hf_info["authenticated"]:
                print(f"   Local HF token detected: yes ({hf_info['source']})")
            else:
                print(f"   Runtime HF token: loaded via Modal Secret '{DEFAULT_HF_SECRET_NAME}'")
            sys.exit(0)
        else:
            ensure_modal_authenticated(exit_on_failure=True)

    elif args.command == "volume-info":
        ensure_modal_authenticated(exit_on_failure=True)
        vol = get_or_create_volume(args.volume)
        print(f"✅ Volume '{args.volume}' is provisioned and ready. Handle: {vol}")
        sys.exit(0)

    elif args.command == "train":
        args.modal_volume = args.volume
        args.modal_gpu = args.gpu
        res = run_train_on_modal(args, volume_name=args.volume, gpu=args.gpu)
        print("Training completed successfully on Modal.")

    elif args.command == "eval":
        args.modal_volume = args.volume
        args.modal_gpu = args.gpu
        res = run_eval_on_modal(args, volume_name=args.volume, gpu=args.gpu)
        print("Evaluation completed successfully on Modal.")

    elif args.command == "test":
        res = run_tests_on_modal(pytest_args=args.pytest_args, volume_name=args.volume, gpu=args.gpu)
        print(f"Remote pytest exited with returncode {res['returncode']}")
        print(res["stdout"])
        if res["stderr"]:
            print("STDERR:", res["stderr"])
        sys.exit(res["returncode"])

    elif args.command == "mock-test":
        res = run_mock_test_on_modal(volume_name=args.volume, gpu=args.gpu)
        print(f"Mock test completed: {res}")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
