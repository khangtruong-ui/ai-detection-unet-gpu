"""
Checkpoint utilities for SID-UNet.
Supports discovering local repository checkpoints, downloading from Hugging Face Hub,
inspecting checkpoint metadata, and formatting on-screen resume notifications.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import torch


HF_CHECKPOINT_CANDIDATES = [
    "checkpoint_latest.pt",
    "checkpoint_periodic.pt",
    "checkpoint_best.pt",
    "pytorch_model.bin",
    "model.pt",
]

LOCAL_CHECKPOINT_PRIORITY = [
    "checkpoint_latest.pt",
    "checkpoint_periodic.pt",
    "checkpoint_best.pt",
]


def is_hf_repo_id(identifier: str) -> bool:
    """Determine if a given string represents a Hugging Face repository identifier.

    Recognizes:
      - 'username/repo-name' (when not an existing local directory or file)
      - 'hf://username/repo-name'
      - 'https://huggingface.co/username/repo-name'
    """
    if not identifier or not isinstance(identifier, str):
        return False

    raw = identifier.strip()
    if raw.startswith("hf://") or raw.startswith("https://huggingface.co/"):
        return True

    # If the path exists as a local file or directory, prioritize local path
    if os.path.exists(raw):
        return False

    # Check for username/repo pattern with optional subfolder or tag
    # Exclude Windows absolute paths (e.g. C:\...) and unix paths starting with / or ./ or ../
    if raw.startswith("/") or raw.startswith("./") or raw.startswith("../") or "\\" in raw:
        return False

    # Strip subfolder/filename if syntax like 'owner/repo:checkpoint.pt'
    repo_part = raw.split(":", 1)[0]
    parts = repo_part.split("/")
    if len(parts) == 2:
        owner, repo = parts
        if re.match(r"^[a-zA-Z0-9_\-\.]+$", owner) and re.match(r"^[a-zA-Z0-9_\-\.]+$", repo):
            # Don't treat files like 'foo/bar.pt' as repo IDs if extension indicates file
            ext = os.path.splitext(repo)[1].lower()
            if ext in [".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".yaml", ".yml", ".json"]:
                return False
            return True

    return False


def parse_hf_repo_uri(uri: str) -> Tuple[str, Optional[str]]:
    """Parse a Hugging Face URI/identifier into (repo_id, optional_filename).

    Examples:
      - 'hf://KhangTruong/sid-unet:checkpoint_best.pt' -> ('KhangTruong/sid-unet', 'checkpoint_best.pt')
      - 'https://huggingface.co/KhangTruong/sid-unet' -> ('KhangTruong/sid-unet', None)
      - 'KhangTruong/sid-unet' -> ('KhangTruong/sid-unet', None)
      - 'KhangTruong/sid-unet:checkpoint_latest.pt' -> ('KhangTruong/sid-unet', 'checkpoint_latest.pt')
    """
    cleaned = uri.strip()
    if cleaned.startswith("hf://"):
        cleaned = cleaned[5:]
    elif cleaned.startswith("https://huggingface.co/"):
        cleaned = cleaned[23:]

    filename = None
    if ":" in cleaned:
        repo_id, filename = cleaned.split(":", 1)
        repo_id = repo_id.strip()
        filename = filename.strip()
    else:
        repo_id = cleaned

    # Strip trailing slashes
    repo_id = repo_id.rstrip("/")
    return repo_id, filename


def download_hf_checkpoint(
    repo_id_or_uri: str,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Download a checkpoint file from a Hugging Face Model Repository.

    Args:
        repo_id_or_uri: HF repository id (e.g. 'KhangTruong/sid-unet') or URI ('hf://...').
        filename: Checkpoint filename in the repo. If None, tries candidates:
                  checkpoint_latest.pt, checkpoint_periodic.pt, checkpoint_best.pt, model.pt, pytorch_model.bin.
        cache_dir: Optional local directory to store downloaded checkpoint.
        token: Optional HF authentication token.

    Returns:
        Dict containing downloaded 'checkpoint_path', 'repo_id', 'filename', and 'source'.
    """
    repo_id, uri_filename = parse_hf_repo_uri(repo_id_or_uri)
    target_filename = filename or uri_filename

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("huggingface_hub is required to resume from Hugging Face repository: " + str(e))

    filenames_to_try = [target_filename] if target_filename else HF_CHECKPOINT_CANDIDATES

    last_error = None
    for fname in filenames_to_try:
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                cache_dir=cache_dir,
                token=token,
            )
            return {
                "checkpoint_path": downloaded_path,
                "repo_id": repo_id,
                "filename": fname,
                "source": "huggingface_repo",
            }
        except Exception as exc:
            last_error = exc
            if target_filename:
                # If specific file was requested, don't try other candidates
                break

    raise FileNotFoundError(
        f"Failed to locate or download any checkpoint from Hugging Face repository '{repo_id}'. "
        f"Tried files: {filenames_to_try}. Last error: {last_error}"
    )


def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Safely inspect a PyTorch checkpoint file to extract metadata without full model instantiation."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: '{checkpoint_path}'")

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        return {
            "epoch": 0,
            "step": 0,
            "metrics": {},
            "best_score": None,
            "best_epoch": -1,
            "is_pure_weights": True,
        }

    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0)) if ckpt.get("step") is not None else None
    metrics = ckpt.get("metrics", {})
    best_score = ckpt.get("best_score", None)
    best_epoch = int(ckpt.get("best_epoch", -1)) if ckpt.get("best_epoch") is not None else -1

    return {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "has_optimizer": "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None,
        "has_scheduler": "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None,
        "has_scaler": "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None,
        "has_history": "history" in ckpt and bool(ckpt["history"]),
        "history": ckpt.get("history", []),
        "is_pure_weights": "model_state_dict" not in ckpt,
    }


def find_auto_resume_checkpoint(
    output_dir: Optional[str] = None,
    config_stem: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Automatically search the local repository and output directories for existing checkpoints to resume from.

    Search locations in order of priority:
      1. output_dir/checkpoints/ (latest, periodic, best, or newest *.pt)
      2. output_dir/ (latest, periodic, best, or newest *.pt)
      3. repo_root/outputs/RUN/<config_stem>/checkpoints/
      4. repo_root/outputs/<config_stem>/checkpoints/
      5. repo_root/outputs/checkpoints/
      6. repo_root/checkpoints/

    Returns:
        Dict containing checkpoint path, inspected metadata, and source, or None if no checkpoint is found.
    """
    root = repo_root or os.getcwd()
    search_dirs: List[str] = []

    if output_dir:
        norm_out = os.path.abspath(output_dir)
        search_dirs.append(os.path.join(norm_out, "checkpoints"))
        search_dirs.append(norm_out)

    if config_stem:
        search_dirs.append(os.path.join(root, "outputs", "RUN", config_stem, "checkpoints"))
        search_dirs.append(os.path.join(root, "outputs", "RUN", config_stem))
        search_dirs.append(os.path.join(root, "outputs", config_stem, "checkpoints"))
        search_dirs.append(os.path.join(root, "outputs", config_stem))

    search_dirs.append(os.path.join(root, "outputs", "checkpoints"))
    search_dirs.append(os.path.join(root, "checkpoints"))

    # De-duplicate directories while preserving order
    seen_dirs = set()
    unique_dirs = []
    for d in search_dirs:
        norm_d = os.path.normpath(d)
        if norm_d not in seen_dirs:
            seen_dirs.add(norm_d)
            unique_dirs.append(norm_d)

    for directory in unique_dirs:
        if not os.path.isdir(directory):
            continue

        # 1. Check priority named checkpoint files
        for priority_name in LOCAL_CHECKPOINT_PRIORITY:
            candidate = os.path.join(directory, priority_name)
            if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                try:
                    meta = inspect_checkpoint(candidate)
                    return {
                        "checkpoint_path": os.path.abspath(candidate),
                        "filename": priority_name,
                        "source": "local_repo",
                        "directory": directory,
                        **meta,
                    }
                except Exception:
                    continue

        # 2. Check any other *.pt files in directory sorted by modification time (newest first)
        other_pts = glob.glob(os.path.join(directory, "*.pt"))
        other_pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for candidate in other_pts:
            if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                try:
                    meta = inspect_checkpoint(candidate)
                    return {
                        "checkpoint_path": os.path.abspath(candidate),
                        "filename": os.path.basename(candidate),
                        "source": "local_repo",
                        "directory": directory,
                        **meta,
                    }
                except Exception:
                    continue

    return None


def format_resume_notification(info: Dict[str, Any]) -> str:
    """Format an informative, highlighted on-screen notification when a checkpoint is found."""
    source_label = "Local Repository" if info.get("source") == "local_repo" else f"Hugging Face Hub ({info.get('repo_id', 'remote')})"
    path = info.get("checkpoint_path", "")
    epoch = info.get("epoch", 0)
    step = info.get("step")
    best_score = info.get("best_score")

    lines = [
        "",
        "================================================================================",
        "🔄 [AUTO-RESUME] Found existing checkpoint in repository!",
        f"   📁 Source:           {source_label}",
        f"   💾 Checkpoint File:  {path}",
        f"   📊 Resume Training:  Starting from Epoch {epoch + 1} (Completed Epoch {epoch})",
    ]
    if step is not None:
        lines.append(f"   ⏱️ Global Step:      {step}")
    if best_score is not None:
        lines.append(f"   ⭐ Best Score:       {best_score:.4f}")
    lines.append("================================================================================")
    lines.append("")
    return "\n".join(lines)


def format_no_resume_notification(output_dir: Optional[str] = None) -> str:
    """Format an on-screen notice indicating no existing checkpoint was found in repository."""
    loc_str = f" in '{output_dir}' or repository search paths" if output_dir else " in repository search paths"
    lines = [
        "",
        "--------------------------------------------------------------------------------",
        f"ℹ️ [AUTO-RESUME] No existing checkpoint found{loc_str}.",
        "   Starting fresh training run from Epoch 1.",
        "--------------------------------------------------------------------------------",
        "",
    ]
    return "\n".join(lines)
