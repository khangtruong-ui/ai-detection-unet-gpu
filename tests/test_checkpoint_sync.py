"""
Unit tests for Hugging Face checkpoint synchronization and version management (sid-push, sid-pull).
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import torch

from sid_unet.checkpoint_sync import (
    cli_main,
    cli_pull,
    cli_push,
    find_local_checkpoint_artifacts,
    generate_model_card_content,
    pull_checkpoint,
    push_checkpoint,
    resolve_repo_and_version,
)
from sid_unet.utils.config import ConfigDict


def test_resolve_repo_and_version():
    with patch("sid_unet.checkpoint_sync._get_hf_username", return_value="TestUser"):
        # Explicit repo with version in name
        repo, ver = resolve_repo_and_version(name="v2", repo="KhangTruong/sid-unet")
        assert repo == "KhangTruong/sid-unet"
        assert ver == "v2"

        # Name is full repo
        repo, ver = resolve_repo_and_version(name="KhangTruong/diffusion-diff-minimized")
        assert repo == "KhangTruong/diffusion-diff-minimized"
        assert ver == "v1"

        # Name is model name (auto-prefixes username)
        repo, ver = resolve_repo_and_version(name="my-model")
        assert repo == "TestUser/my-model"
        assert ver == "v1"

        # Explicit version
        repo, ver = resolve_repo_and_version(name="my-model", version="v3.1")
        assert repo == "TestUser/my-model"
        assert ver == "v3.1"


def test_find_local_checkpoint_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        rep_dir = os.path.join(tmpdir, "reports")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(rep_dir, exist_ok=True)

        best_pt = os.path.join(ckpt_dir, "checkpoint_best.pt")
        latest_pt = os.path.join(ckpt_dir, "checkpoint_latest.pt")
        torch.save({"epoch": 5, "best_score": 0.8}, best_pt)
        torch.save({"epoch": 5, "best_score": 0.8}, latest_pt)

        curve_png = os.path.join(rep_dir, "training_curves.png")
        with open(curve_png, "wb") as f:
            f.write(b"dummy_png")

        artifacts = find_local_checkpoint_artifacts(source_dir=tmpdir)
        assert "checkpoint_best.pt" in artifacts["checkpoints"]
        assert "checkpoint_latest.pt" in artifacts["checkpoints"]
        assert "training_curves.png" in artifacts["reports"]


def test_generate_model_card_content():
    manifest = {
        "versions": ["v1", "v2"],
        "versions_meta": {
            "v1": {"best_score": "0.75", "best_epoch": "5", "updated_at": "2026-01-01T00:00:00Z"},
            "v2": {"best_score": "0.85", "best_epoch": "10", "updated_at": "2026-01-02T00:00:00Z"},
        },
    }
    card = generate_model_card_content("KhangTruong/my-model", "v2", manifest)
    assert "# my-model" in card
    assert "`v1`" in card
    assert "`v2`" in card
    assert "sid-pull" in card
    assert "sid-train" in card


@patch("sid_unet.checkpoint_sync.HfApi")
def test_push_checkpoint_mocked(mock_api_cls):
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.repo_info.return_value = {"id": "TestUser/test-repo"}
    mock_commit = MagicMock()
    mock_commit.oid = "abc123456789"
    mock_api.create_commit.return_value = mock_commit

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"epoch": 3, "best_score": 0.6}, os.path.join(ckpt_dir, "checkpoint_best.pt"))

        res = push_checkpoint(
            name="TestUser/test-repo",
            version="v1",
            source_dir=tmpdir,
            message="Test commit",
        )
        assert res["repo_id"] == "TestUser/test-repo"
        assert res["version"] == "v1"
        assert mock_api.create_commit.called


@patch("sid_unet.checkpoint_sync.hf_hub_download")
@patch("sid_unet.checkpoint_sync.HfApi")
def test_pull_checkpoint_mocked(mock_api_cls, mock_download):
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.list_repo_files.return_value = [
        "checkpoints/checkpoint_best.pt",
        "checkpoints/checkpoint_latest.pt",
        "config.yaml",
    ]

    with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as dest_dir:
        dummy_file = os.path.join(cache_dir, "file.pt")
        torch.save({"dummy": 1}, dummy_file)
        mock_download.return_value = dummy_file

        res = pull_checkpoint(
            name="TestUser/test-repo",
            output_dir=dest_dir,
        )
        assert res["repo_id"] == "TestUser/test-repo"
        assert os.path.exists(os.path.join(dest_dir, "checkpoints", "checkpoint_best.pt"))


def test_cli_parsing_no_crash():
    with patch("sid_unet.checkpoint_sync.push_checkpoint") as mock_push:
        mock_push.return_value = {}
        ret = cli_push(["--name", "TestUser/test-repo", "--version", "v1"])
        assert ret == 0

    with patch("sid_unet.checkpoint_sync.pull_checkpoint") as mock_pull:
        mock_pull.return_value = {}
        ret = cli_pull(["--name", "TestUser/test-repo", "--version", "v1"])
        assert ret == 0

    with patch("sid_unet.checkpoint_sync.cli_push") as mock_sub_push:
        mock_sub_push.return_value = 0
        ret = cli_main(["push", "--name", "TestUser/test-repo"])
        assert ret == 0
