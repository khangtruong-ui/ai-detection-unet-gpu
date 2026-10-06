"""
Hugging Face Hub Checkpoint Synchronization and Version Management.

Provides CLI commands (`sid-push`, `sid-pull`, `sid-checkpoint`) and Python APIs
for pushing, pulling, versioning, and inspecting model checkpoints on Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

try:
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationDelete,
        HfApi,
        get_token,
        hf_hub_download,
        snapshot_download,
    )
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


def _get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Retrieve Hugging Face token from parameter, env vars, or local hub cache."""
    if explicit_token:
        return explicit_token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    if _HF_AVAILABLE:
        try:
            return get_token()
        except Exception:
            pass
    return None


def _get_hf_username(api: HfApi, token: Optional[str] = None) -> Optional[str]:
    """Attempt to get authenticated username from token."""
    try:
        user_info = api.whoami(token=token)
        return user_info.get("name")
    except Exception:
        return None


def verify_hf_repo_checkpointable(
    repo_id_or_uri: str,
    token: Optional[str] = None,
    create_if_missing: bool = True,
    private: bool = False,
    raise_on_error: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Verify if a Hugging Face model repository is accessible, writable, and checkpointable.

    Performs comprehensive upfront ('first hand') validation:
      1. Validates repository format and parses repo ID.
      2. Ensures `huggingface_hub` package is installed.
      3. Verifies Hugging Face API token is present and valid (`whoami`).
      4. Checks whether repository exists and is writable (`auth_check(write=True)`).
      5. If repository does not exist and `create_if_missing=True`, creates repository
         and validates write authorization.

    Returns:
        (True, None) if repo is verified checkpointable.
        (False, error_message) or raises exception if not checkpointable.
    """
    if not _HF_AVAILABLE:
        msg = "huggingface_hub is required for checkpointing to Hugging Face: pip install huggingface_hub"
        if raise_on_error:
            raise ImportError(msg)
        return False, msg

    hf_token = _get_hf_token(token)
    if not hf_token:
        msg = (
            f"No Hugging Face authentication token found to checkpoint to repository '{repo_id_or_uri}'. "
            "Please set HF_TOKEN environment variable or run `huggingface-cli login`."
        )
        if raise_on_error:
            raise PermissionError(msg)
        return False, msg

    api = HfApi(token=hf_token)

    try:
        whoami = api.whoami(token=hf_token)
        username = whoami.get("name")
    except Exception as exc:
        msg = f"Invalid or expired Hugging Face token when authenticating for '{repo_id_or_uri}': {exc}"
        if raise_on_error:
            raise PermissionError(msg) from exc
        return False, msg

    # Resolve repo ID
    repo_id = repo_id_or_uri.strip()
    if repo_id.startswith("hf://"):
        repo_id = repo_id[5:]
    elif repo_id.startswith("https://huggingface.co/"):
        repo_id = repo_id[len("https://huggingface.co/"):]
    if ":" in repo_id:
        repo_id = repo_id.split(":", 1)[0]
    repo_id = repo_id.strip("/")

    if "/" not in repo_id and username:
        repo_id = f"{username}/{repo_id}"

    try:
        exists = api.repo_exists(repo_id=repo_id, repo_type="model", token=hf_token)
    except Exception as exc:
        msg = f"Could not query Hugging Face repository '{repo_id}': {exc}"
        if raise_on_error:
            raise RuntimeError(msg) from exc
        return False, msg

    if exists:
        try:
            api.auth_check(repo_id=repo_id, repo_type="model", write=True, token=hf_token)
        except Exception as exc:
            msg = f"Hugging Face repository '{repo_id}' exists but write access was denied: {exc}"
            if raise_on_error:
                raise PermissionError(msg) from exc
            return False, msg
    else:
        if create_if_missing:
            try:
                api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=hf_token)
                api.auth_check(repo_id=repo_id, repo_type="model", write=True, token=hf_token)
            except Exception as exc:
                msg = f"Failed to create or verify write access for Hugging Face repository '{repo_id}': {exc}"
                if raise_on_error:
                    raise PermissionError(msg) from exc
                return False, msg
        else:
            msg = f"Hugging Face repository '{repo_id}' does not exist and create_if_missing is False."
            if raise_on_error:
                raise FileNotFoundError(msg)
            return False, msg

    return True, None



def resolve_repo_and_version(
    name: Optional[str] = None,
    repo: Optional[str] = None,
    version: Optional[str] = None,
    config: Optional[Any] = None,
    token: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve the target Hugging Face repo_id and version identifier.

    Handles flexible user inputs:
      - --name KhangTruong/diffusion-diff-minimized -> repo='KhangTruong/diffusion-diff-minimized', version='v1'
      - --name v2 --repo KhangTruong/diffusion-diff-minimized -> repo='...', version='v2'
      - --name diffusion-diff-minimized -> resolves to username/diffusion-diff-minimized
    """
    if not _HF_AVAILABLE:
        raise ImportError("huggingface_hub is required for checkpoint sync: pip install huggingface_hub")

    api = HfApi(token=token)
    username = _get_hf_username(api, token=token)

    resolved_repo = repo
    resolved_version = version

    # 1. Resolve repo from --repo flag
    if not resolved_repo:
        # Check if name contains '/' (e.g. 'KhangTruong/diffusion-diff-minimized')
        if name and "/" in name:
            resolved_repo = name
            # If name was the repo, version should come from --version or default
            if not resolved_version:
                resolved_version = "v1"
        elif name:
            # Check if name is a version tag (e.g. 'v1', 'v2', 'latest')
            is_version_tag = bool(re.match(r"^v\d+(\.\d+)*$", name, re.IGNORECASE)) or name.lower() in ("latest", "best", "main")
            if is_version_tag:
                resolved_version = name
            else:
                # Name is a model / experiment name (e.g. 'diffusion-diff-minimized')
                # Check if username exists, repo could be f"{username}/{name}"
                if username:
                    candidate = f"{username}/{name}"
                    resolved_repo = candidate
                else:
                    resolved_repo = name

    # Fallback to config if repo is still not resolved
    if not resolved_repo and config:
        project_cfg = getattr(config, "project", {}) if not hasattr(config, "get") else config.get("project", {})
        training_cfg = getattr(config, "training", {}) if not hasattr(config, "get") else config.get("training", {})
        cfg_repo = (
            project_cfg.get("hub_repo")
            or project_cfg.get("resume_repo")
            or training_cfg.get("hub_repo")
            or training_cfg.get("resume_repo")
        )
        if cfg_repo and not cfg_repo.endswith(".pt"):
            resolved_repo = str(cfg_repo)

    if not resolved_repo:
        if name:
            if username and "/" not in name:
                resolved_repo = f"{username}/{name}"
            else:
                resolved_repo = name
        else:
            raise ValueError(
                "Unable to determine Hugging Face repository. Please provide --name or --repo "
                "(e.g. --name KhangTruong/diffusion-diff-minimized or --repo KhangTruong/sid-unet --name v1)."
            )

    if not resolved_version:
        if name and name != resolved_repo and not ("/" in name) and not resolved_repo.endswith(f"/{name}"):
            resolved_version = name
        else:
            resolved_version = "v1"

    # Normalize version string (e.g. '1' -> 'v1')
    if resolved_version.isdigit():
        resolved_version = f"v{resolved_version}"

    return resolved_repo, resolved_version


def find_local_checkpoint_artifacts(
    source_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover local checkpoint files, configs, and reports to synchronize."""
    target_dir = None
    if source_dir and os.path.isdir(source_dir):
        target_dir = os.path.abspath(source_dir)
    elif config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
            out = raw_cfg.get("project", {}).get("output_dir")
            if out and os.path.isdir(out):
                target_dir = os.path.abspath(out)
        except Exception:
            pass

    # Search common directories if not found
    if not target_dir:
        candidates = [
            "outputs/RUN/default",
            "outputs/experiments/diffusion_diff_minimized",
            "outputs",
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                target_dir = os.path.abspath(cand)
                break

    if not target_dir or not os.path.exists(target_dir):
        raise FileNotFoundError(f"Local output directory not found: '{source_dir or target_dir}'")

    # Locate checkpoints
    ckpt_dir = os.path.join(target_dir, "checkpoints") if os.path.isdir(os.path.join(target_dir, "checkpoints")) else target_dir
    checkpoints: Dict[str, str] = {}
    
    if checkpoint_name:
        candidate_path = os.path.join(ckpt_dir, checkpoint_name)
        if os.path.isfile(candidate_path):
            checkpoints[checkpoint_name] = candidate_path
        elif os.path.isfile(checkpoint_name):
            checkpoints[os.path.basename(checkpoint_name)] = os.path.abspath(checkpoint_name)
        else:
            raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_name}")
    else:
        for p in ["checkpoint_best.pt", "checkpoint_latest.pt", "checkpoint_periodic.pt"]:
            full_p = os.path.join(ckpt_dir, p)
            if os.path.isfile(full_p):
                checkpoints[p] = full_p
        # If none of the standard names exist, find any *.pt
        if not checkpoints:
            for pt_file in glob.glob(os.path.join(ckpt_dir, "*.pt")):
                checkpoints[os.path.basename(pt_file)] = pt_file

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint (.pt) files found in '{ckpt_dir}'.")

    # Locate config files
    configs: Dict[str, str] = {}
    for cfg_cand in [
        os.path.join(target_dir, "effective_config.yaml"),
        os.path.join(ckpt_dir, "checkpoint_best_config.yaml"),
        os.path.join(ckpt_dir, "checkpoint_latest_config.yaml"),
        os.path.join(ckpt_dir, "checkpoint_periodic_config.yaml"),
    ]:
        if os.path.isfile(cfg_cand):
            configs[os.path.basename(cfg_cand)] = cfg_cand

    # Locate reports and curves
    reports: Dict[str, str] = {}
    report_dir = os.path.join(target_dir, "reports") if os.path.isdir(os.path.join(target_dir, "reports")) else target_dir
    for rep in [
        "training_curves.png", "training_curves.jpg", "training_curves.pdf",
        "test_evaluation_report.json", "test_evaluation_report.md",
        "training_final_report.json", "training_final_report.md",
        "training_history.json", "training_history.csv",
    ]:
        full_rep = os.path.join(report_dir, rep)
        if os.path.isfile(full_rep):
            reports[rep] = full_rep

    return {
        "base_dir": target_dir,
        "checkpoint_dir": ckpt_dir,
        "checkpoints": checkpoints,
        "configs": configs,
        "reports": reports,
    }


def generate_model_card_content(
    repo_id: str,
    version: str,
    manifest: Dict[str, Any],
    config_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a clean, structured Hugging Face README.md Model Card."""
    model_name = repo_id.split("/")[-1]
    tags = [
        "pytorch",
        "sid-unet",
        "image-segmentation",
        "synthetic-image-detection",
        "diffusion-detection",
    ]
    
    versions_list = manifest.get("versions", [])
    current_meta = manifest.get("versions_meta", {}).get(version, {})
    best_score = current_meta.get("best_score", "N/A")
    best_epoch = current_meta.get("best_epoch", "N/A")
    total_epochs = current_meta.get("total_epochs", "N/A")

    # Build versions table
    table_rows = []
    for v in sorted(versions_list):
        v_meta = manifest.get("versions_meta", {}).get(v, {})
        v_score = v_meta.get("best_score", "N/A")
        v_ep = v_meta.get("best_epoch", "N/A")
        v_date = v_meta.get("updated_at", "")[:10]
        table_rows.append(f"| `{v}` | {v_score} | {v_ep} | {v_date} |")
    versions_table = "\n".join(table_rows) if table_rows else f"| `{version}` | {best_score} | {best_epoch} | {datetime.now(timezone.utc).strftime('%Y-%m-%d')} |"

    card = f"""---
license: apache-2.0
tags:
{yaml.dump(tags, default_flow_style=False).strip()}
pipeline_tag: image-segmentation
library_name: sid-unet
---

# {model_name}

Official repository for **{model_name}** checkpoints, evaluated on AI-generated image detection and segmentation tasks.

## 📌 Available Versions

| Version | Best Metric Score | Best Epoch | Updated |
| :--- | :--- | :--- | :--- |
{versions_table}

- **Active Version:** `{version}`
- **Best Validation Score:** `{best_score}` (Epoch `{best_epoch}`)
- **Total Training Epochs:** `{total_epochs}`

---

## 🚀 Quickstart & Checkpoint Usage

### 1. Pull checkpoints using `sid-pull`
To download the latest checkpoint directly to your local workspace:
```bash
sid-pull --name {repo_id}
```

To pull a specific version:
```bash
sid-pull --name {repo_id} --version {version}
```

### 2. Resume or Fine-tune with `sid-train`
Once downloaded, resume training seamlessly:
```bash
sid-train --config configs/experiments/diffusion_diff_minimized/default.yaml --override training.epochs=40
```
Or specify explicit checkpoint path:
```bash
sid-train --resume checkpoints/checkpoint_latest.pt
```

### 3. Push new checkpoints using `sid-push`
```bash
sid-push --name {repo_id} --version v2
```

---

## 📂 Repository Layout

```
{repo_id}/
├── README.md               # Model Card & Documentation
├── config.yaml             # Model & Training Configuration
├── manifest.json           # Catalog of all versions & metrics
├── checkpoints/            # Active / Latest Checkpoints
│   ├── checkpoint_best.pt
│   ├── checkpoint_latest.pt
│   └── checkpoint_periodic.pt
├── reports/                # Evaluation reports & plots
│   ├── training_curves.png
│   └── test_evaluation_report.json
└── versions/               # Version Archive
    └── {version}/
        ├── checkpoints/
        └── reports/
```
"""
    return card


def push_checkpoint(
    name: Optional[str] = None,
    repo: Optional[str] = None,
    version: Optional[str] = None,
    source_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    message: Optional[str] = None,
    private: bool = False,
    include_reports: bool = True,
    set_latest: bool = True,
    tag: Optional[str] = None,
    reformat: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Push checkpoints to Hugging Face Hub with versioning and clean multi-version structure."""
    if not _HF_AVAILABLE:
        raise ImportError("huggingface_hub is required for checkpoint sync: pip install huggingface_hub")

    hf_token = _get_hf_token(token)
    api = HfApi(token=hf_token)

    # 1. Resolve repository and version
    repo_id, ver = resolve_repo_and_version(
        name=name,
        repo=repo,
        version=version,
        token=hf_token,
    )

    # Verify repository is checkpointable upfront ("on the first hand")
    verify_hf_repo_checkpointable(
        repo_id_or_uri=repo_id,
        token=hf_token,
        create_if_missing=True,
        private=private,
        raise_on_error=True,
    )

    # 2. Find local artifacts
    artifacts = find_local_checkpoint_artifacts(
        source_dir=source_dir,
        config_path=config_path,
        checkpoint_name=checkpoint_name,
    )

    # Inspect checkpoint metadata
    best_score = "N/A"
    best_epoch = "N/A"
    step = "N/A"
    total_epochs = "N/A"

    primary_pt = artifacts["checkpoints"].get("checkpoint_best.pt") or artifacts["checkpoints"].get("checkpoint_latest.pt") or next(iter(artifacts["checkpoints"].values()))
    try:
        import torch
        ckpt_data = torch.load(primary_pt, map_location="cpu", weights_only=False)
        if isinstance(ckpt_data, dict):
            best_score = ckpt_data.get("best_score", ckpt_data.get("metrics", {}).get("val_iou", "N/A"))
            best_epoch = ckpt_data.get("best_epoch", ckpt_data.get("epoch", "N/A"))
            step = ckpt_data.get("step", "N/A")
            total_epochs = ckpt_data.get("epoch", "N/A")
    except Exception:
        pass

    # 3. Fetch or initialize remote manifest.json
    manifest: Dict[str, Any] = {"versions": [], "versions_meta": {}, "current_version": ver}
    try:
        manifest_path = hf_hub_download(repo_id=repo_id, filename="manifest.json", token=hf_token)
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded_m = json.load(f)
            if isinstance(loaded_m, dict):
                manifest = loaded_m
    except Exception:
        pass

    if ver not in manifest.get("versions", []):
        manifest.setdefault("versions", []).append(ver)

    manifest.setdefault("versions_meta", {})[ver] = {
        "version": ver,
        "best_score": str(best_score),
        "best_epoch": str(best_epoch),
        "total_epochs": str(total_epochs),
        "step": str(step),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": list(artifacts["checkpoints"].keys()),
    }
    manifest["current_version"] = ver

    # 4. Prepare commit operations
    operations = []

    # Upload versioned checkpoints: versions/<version>/checkpoints/
    for fname, fpath in artifacts["checkpoints"].items():
        operations.append(CommitOperationAdd(path_in_repo=f"versions/{ver}/checkpoints/{fname}", path_or_fileobj=fpath))
        if set_latest:
            operations.append(CommitOperationAdd(path_in_repo=f"checkpoints/{fname}", path_or_fileobj=fpath))

    # Upload configs
    main_cfg_path = None
    for fname, fpath in artifacts["configs"].items():
        operations.append(CommitOperationAdd(path_in_repo=f"versions/{ver}/{fname}", path_or_fileobj=fpath))
        if fname in ("effective_config.yaml", "checkpoint_best_config.yaml", "checkpoint_latest_config.yaml"):
            main_cfg_path = fpath
        if set_latest:
            operations.append(CommitOperationAdd(path_in_repo=f"checkpoints/{fname}", path_or_fileobj=fpath))

    if main_cfg_path and set_latest:
        operations.append(CommitOperationAdd(path_in_repo="config.yaml", path_or_fileobj=main_cfg_path))

    # Upload reports if enabled
    if include_reports:
        for fname, fpath in artifacts["reports"].items():
            operations.append(CommitOperationAdd(path_in_repo=f"versions/{ver}/reports/{fname}", path_or_fileobj=fpath))
            if set_latest:
                operations.append(CommitOperationAdd(path_in_repo=f"reports/{fname}", path_or_fileobj=fpath))

    # Upload manifest.json and README.md
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    operations.append(CommitOperationAdd(path_in_repo="manifest.json", path_or_fileobj=manifest_bytes))
    operations.append(CommitOperationAdd(path_in_repo=f"versions/{ver}/manifest.json", path_or_fileobj=manifest_bytes))

    card_text = generate_model_card_content(repo_id=repo_id, version=ver, manifest=manifest)
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_text.encode("utf-8")))

    # Reformat / Clean up legacy ugly structure (e.g. outputs/RUN/default/...)
    if reformat:
        try:
            remote_files = api.list_repo_files(repo_id=repo_id)
            legacy_files = [f for f in remote_files if f.startswith("outputs/")]
            for leg in legacy_files:
                operations.append(CommitOperationDelete(path_in_repo=leg))
            if legacy_files:
                print(f"🧹 Reformatting repository: removing {len(legacy_files)} legacy path(s) under outputs/...")
        except Exception as e:
            print(f"Warning during legacy scan: {e}")

    commit_msg = message or f"feat(checkpoint): push {repo_id} version {ver} (score: {best_score})"
    print(f"🚀 Uploading checkpoint artifacts to Hugging Face: '{repo_id}' (version: '{ver}')...")
    
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_msg,
    )

    # Create Git tag if requested
    if tag:
        try:
            api.create_tag(repo_id=repo_id, tag=tag, tag_message=f"Release {tag} - Version {ver}")
            print(f"🏷️ Created Hugging Face Git tag: '{tag}'")
        except Exception as tag_err:
            print(f"Note on tag creation: {tag_err}")

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"✅ Successfully pushed checkpoints to: {repo_url}")
    print(f"   Version: {ver} | Commit: {commit_info.oid[:8] if hasattr(commit_info, 'oid') else 'done'}")
    print(f"   Active Checkpoints: checkpoints/checkpoint_best.pt, checkpoints/checkpoint_latest.pt")
    print(f"   Archived Version: versions/{ver}/checkpoints/")

    return {
        "repo_id": repo_id,
        "version": ver,
        "repo_url": repo_url,
        "commit": getattr(commit_info, "oid", None),
        "manifest": manifest,
    }


def pull_checkpoint(
    name: Optional[str] = None,
    repo: Optional[str] = None,
    version: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    force: bool = False,
    list_only: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Pull checkpoints from Hugging Face Hub, placing them in target output directory."""
    if not _HF_AVAILABLE:
        raise ImportError("huggingface_hub is required for checkpoint sync: pip install huggingface_hub")

    hf_token = _get_hf_token(token)
    api = HfApi(token=hf_token)

    # 1. Resolve repo ID
    repo_id, ver = resolve_repo_and_version(
        name=name,
        repo=repo,
        version=version,
        token=hf_token,
    )

    # 2. Query remote files
    try:
        remote_files = api.list_repo_files(repo_id=repo_id)
    except Exception as e:
        raise FileNotFoundError(f"Could not access Hugging Face repository '{repo_id}': {e}")

    # Inspect manifest if present
    manifest = {}
    if "manifest.json" in remote_files:
        try:
            m_path = hf_hub_download(repo_id=repo_id, filename="manifest.json", token=hf_token)
            with open(m_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    available_versions = manifest.get("versions", [])
    if not available_versions:
        # Detect from directories
        for f in remote_files:
            m = re.match(r"^versions/([^/]+)/", f)
            if m and m.group(1) not in available_versions:
                available_versions.append(m.group(1))

    if list_only:
        print(f"\n📦 Checkpoints & Versions for Hugging Face Repository: {repo_id}")
        print("=" * 65)
        if available_versions:
            print("Available Versions:")
            for v in available_versions:
                meta = manifest.get("versions_meta", {}).get(v, {})
                sc = meta.get("best_score", "N/A")
                ep = meta.get("best_epoch", "N/A")
                print(f"  • {v:12s} (Best Score: {sc}, Best Epoch: {ep})")
        else:
            print("No version subdirectories found; repository uses root/legacy layout.")

        pt_files = [f for f in remote_files if f.endswith(".pt")]
        print(f"\nAvailable Checkpoint Files ({len(pt_files)}):")
        for f in pt_files:
            print(f"  - {f}")
        return {"repo_id": repo_id, "versions": available_versions, "files": pt_files}

    # 3. Determine target local directory
    target_dir = None
    if output_dir:
        target_dir = os.path.abspath(output_dir)
    elif config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
            out = raw_cfg.get("project", {}).get("output_dir")
            if out:
                target_dir = os.path.abspath(out)
        except Exception:
            pass

    if not target_dir:
        # Check standard destinations
        short_name = repo_id.split("/")[-1]
        matching_config = f"configs/experiments/{short_name.replace('-', '_')}/default.yaml"
        if os.path.isfile(matching_config):
            try:
                with open(matching_config, "r", encoding="utf-8") as f:
                    raw_cfg = yaml.safe_load(f) or {}
                out = raw_cfg.get("project", {}).get("output_dir")
                if out:
                    target_dir = os.path.abspath(out)
            except Exception:
                pass

    if not target_dir:
        target_dir = os.path.abspath(os.path.join("outputs", "RUN", "default"))

    ckpt_target_dir = os.path.join(target_dir, "checkpoints")
    os.makedirs(ckpt_target_dir, exist_ok=True)

    # 4. Determine which files to download
    # Check if a specific version is requested
    prefix = ""
    if version and version.lower() not in ("latest", "default"):
        prefix = f"versions/{version}/"

    files_to_download = []
    if checkpoint_name:
        candidate_names = [checkpoint_name]
        if not checkpoint_name.endswith(".pt"):
            candidate_names.append(f"{checkpoint_name}.pt")
        found_target = False
        for cname in candidate_names:
            candidates = [
                f"{prefix}checkpoints/{cname}",
                f"{prefix}{cname}",
                f"checkpoints/{cname}",
                cname,
            ]
            for c in candidates:
                if c in remote_files:
                    files_to_download.append(c)
                    found_target = True
                    break
            if found_target:
                break
        if not found_target:
            raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found in remote repository '{repo_id}'.")
    else:
        # Download key checkpoints: latest and best (and periodic if present)
        for cand in ["checkpoint_latest.pt", "checkpoint_best.pt", "checkpoint_periodic.pt"]:
            found = False
            for p in [f"{prefix}checkpoints/{cand}", f"{prefix}{cand}", f"checkpoints/{cand}", cand]:
                if p in remote_files:
                    files_to_download.append(p)
                    found = True
                    break
            # Fallback to legacy path if not found in root/version
            if not found:
                for leg in [
                    f"outputs/RUN/default/checkpoints/{cand}",
                    f"outputs/RUN/default/{cand}",
                    f"outputs/checkpoints/{cand}",
                ]:
                    if leg in remote_files:
                        files_to_download.append(leg)
                        break

        # Also download config and reports if present
        for cfg_cand in [f"{prefix}config.yaml", "config.yaml", "effective_config.yaml", "outputs/RUN/default/effective_config.yaml"]:
            if cfg_cand in remote_files:
                files_to_download.append(cfg_cand)
                break

    if not files_to_download:
        # Fall back to downloading any *.pt file
        pts = [f for f in remote_files if f.endswith(".pt")]
        if pts:
            files_to_download.append(pts[0])
        else:
            raise FileNotFoundError(f"No checkpoint files found in Hugging Face repository '{repo_id}'.")

    # 5. Download files into local workspace
    downloaded_paths = {}
    print(f"📥 Pulling checkpoints from Hugging Face '{repo_id}' (target: {ckpt_target_dir})...")
    for rpath in files_to_download:
        cached_file = hf_hub_download(
            repo_id=repo_id,
            filename=rpath,
            token=hf_token,
            force_download=force,
        )
        base_name = os.path.basename(rpath)
        dest_path = os.path.join(ckpt_target_dir, base_name)
        shutil.copy2(cached_file, dest_path)
        downloaded_paths[base_name] = dest_path

        # If it's a config, also save to parent target_dir
        if base_name in ("config.yaml", "effective_config.yaml"):
            shutil.copy2(cached_file, os.path.join(target_dir, base_name))

    # Also synchronize convenience directory outputs/RUN/default so auto_resume finds it everywhere
    legacy_run_dir = os.path.abspath(os.path.join("outputs", "RUN", "default", "checkpoints"))
    if os.path.abspath(ckpt_target_dir) != legacy_run_dir:
        os.makedirs(legacy_run_dir, exist_ok=True)
        for fname, fpath in downloaded_paths.items():
            if fname.endswith(".pt"):
                shutil.copy2(fpath, os.path.join(legacy_run_dir, fname))

    print(f"✅ Successfully pulled {len(downloaded_paths)} file(s) into: {ckpt_target_dir}")
    for fname, fpath in downloaded_paths.items():
        print(f"   ✓ {fname} ({os.path.getsize(fpath) / (1024 * 1024):.1f} MB)")

    # Suggest resume command
    resume_cmd = (
        f"sid-train --config configs/experiments/diffusion_diff_minimized/default.yaml"
        f" --override training.epochs=40"
    )
    print(f"\n💡 Resume training with:\n   {resume_cmd}\n")

    return {
        "repo_id": repo_id,
        "target_dir": target_dir,
        "checkpoint_dir": ckpt_target_dir,
        "files": downloaded_paths,
    }


def cli_push(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for sid-push."""
    parser = argparse.ArgumentParser(
        prog="sid-push",
        description="Push local checkpoints to Hugging Face Hub with versioning and clean multi-version structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", "-n", type=str, default=None, help="Name of repo, model, or version to push (e.g. 'v1', 'diffusion-diff-minimized', or 'KhangTruong/diffusion-diff-minimized')")
    parser.add_argument("--repo", "-r", type=str, default=None, help="Hugging Face repository ID (e.g. 'KhangTruong/diffusion-diff-minimized')")
    parser.add_argument("--version", "-v", type=str, default=None, help="Version tag (e.g. 'v1', 'v2'). Defaults to --name if valid version, or 'v1'")
    parser.add_argument("--dir", "-d", "--output-dir", dest="output_dir", type=str, default=None, help="Local directory containing checkpoints to push")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file path to resolve local output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint filename to push (default: best, latest, and periodic)")
    parser.add_argument("--message", "-m", type=str, default=None, help="Commit message for Hugging Face upload")
    parser.add_argument("--private", action="store_true", help="Create repo as private if it does not already exist")
    parser.add_argument("--no-reports", dest="include_reports", action="store_false", default=True, help="Exclude evaluation reports and training plots")
    parser.add_argument("--no-latest", dest="set_latest", action="store_false", default=True, help="Do not update root default/latest checkpoints")
    parser.add_argument("--tag", type=str, default=None, help="Create a git tag / revision on Hugging Face (e.g. 'v1.0')")
    parser.add_argument("--reformat", action="store_true", default=False, help="Reformat repo to clean multi-version structure, removing legacy paths")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")

    args = parser.parse_args(argv)
    try:
        push_checkpoint(
            name=args.name,
            repo=args.repo,
            version=args.version,
            source_dir=args.output_dir,
            config_path=args.config,
            checkpoint_name=args.checkpoint,
            message=args.message,
            private=args.private,
            include_reports=args.include_reports,
            set_latest=args.set_latest,
            tag=args.tag,
            reformat=args.reformat,
            token=args.token,
        )
        return 0
    except Exception as exc:
        print(f"❌ Error during sid-push: {exc}", file=sys.stderr)
        return 1


def cli_pull(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for sid-pull."""
    parser = argparse.ArgumentParser(
        prog="sid-pull",
        description="Pull model checkpoints from Hugging Face Hub with version selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", "-n", type=str, default=None, help="Name of repo, model, or version to pull (e.g. 'KhangTruong/diffusion-diff-minimized' or 'diffusion-diff-minimized')")
    parser.add_argument("--repo", "-r", type=str, default=None, help="Hugging Face repository ID (e.g. 'KhangTruong/diffusion-diff-minimized')")
    parser.add_argument("--version", "-v", type=str, default="latest", help="Version to pull (e.g. 'v1', 'v2', 'latest')")
    parser.add_argument("--dir", "-d", "--output-dir", dest="output_dir", type=str, default=None, help="Local directory to store pulled checkpoints")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file path to resolve target output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file to pull (default: both best and latest)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if already present locally")
    parser.add_argument("--list", "-l", action="store_true", help="List available versions and checkpoints in remote repository without downloading")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")

    args = parser.parse_args(argv)
    try:
        pull_checkpoint(
            name=args.name,
            repo=args.repo,
            version=args.version,
            output_dir=args.output_dir,
            config_path=args.config,
            checkpoint_name=args.checkpoint,
            force=args.force,
            list_only=args.list,
            token=args.token,
        )
        return 0
    except Exception as exc:
        print(f"❌ Error during sid-pull: {exc}", file=sys.stderr)
        return 1


def cli_main(argv: Optional[List[str]] = None) -> int:
    """Unified entrypoint for `sid-checkpoint` with `push`, `pull`, and `list` subcommands."""
    parser = argparse.ArgumentParser(
        prog="sid-checkpoint",
        description="Hugging Face Hub Checkpoint and Version Management CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")
    
    # Subcommand push
    push_p = subparsers.add_parser("push", help="Push local checkpoints to Hugging Face Hub")
    push_p.add_argument("--name", "-n", type=str, default=None)
    push_p.add_argument("--repo", "-r", type=str, default=None)
    push_p.add_argument("--version", "-v", type=str, default=None)
    push_p.add_argument("--dir", "-d", dest="output_dir", type=str, default=None)
    push_p.add_argument("--config", "-c", type=str, default=None)
    push_p.add_argument("--checkpoint", type=str, default=None)
    push_p.add_argument("--message", "-m", type=str, default=None)
    push_p.add_argument("--private", action="store_true")
    push_p.add_argument("--reformat", action="store_true", default=False)
    push_p.add_argument("--token", type=str, default=None)

    # Subcommand pull
    pull_p = subparsers.add_parser("pull", help="Pull checkpoints from Hugging Face Hub")
    pull_p.add_argument("--name", "-n", type=str, default=None)
    pull_p.add_argument("--repo", "-r", type=str, default=None)
    pull_p.add_argument("--version", "-v", type=str, default="latest")
    pull_p.add_argument("--dir", "-d", dest="output_dir", type=str, default=None)
    pull_p.add_argument("--config", "-c", type=str, default=None)
    pull_p.add_argument("--checkpoint", type=str, default=None)
    pull_p.add_argument("--force", action="store_true")
    pull_p.add_argument("--token", type=str, default=None)

    # Subcommand list
    list_p = subparsers.add_parser("list", help="List versions and checkpoints in remote repository")
    list_p.add_argument("--name", "-n", type=str, default=None)
    list_p.add_argument("--repo", "-r", type=str, default=None)
    list_p.add_argument("--token", type=str, default=None)

    args, unknown = parser.parse_known_args(argv)
    if args.command == "push":
        return cli_push(argv[1:] if argv else None)
    elif args.command == "pull":
        return cli_pull(argv[1:] if argv else None)
    elif args.command == "list":
        try:
            pull_checkpoint(name=args.name, repo=args.repo, list_only=True, token=args.token)
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(cli_main())
