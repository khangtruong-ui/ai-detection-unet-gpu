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


@patch("sid_unet.checkpoint_sync._get_hf_token", return_value="fake_token")
@patch("sid_unet.checkpoint_sync.HfApi")
def test_push_checkpoint_mocked(mock_api_cls, mock_token):
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.whoami.return_value = {"name": "TestUser"}
    mock_api.repo_exists.return_value = True
    mock_api.auth_check.return_value = None
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


# --- New Verification & Tying Tests ---

from sid_unet.checkpoint_sync import verify_hf_repo_checkpointable
from sid_unet.training.callbacks import CheckpointManager
from sid_unet.models.unet import UNet


@patch("sid_unet.checkpoint_sync._get_hf_token", return_value="fake_token")
@patch("sid_unet.checkpoint_sync.HfApi")
def test_verify_hf_repo_checkpointable_success(mock_api_cls, mock_token):
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.whoami.return_value = {"name": "KhangTruong"}
    mock_api.repo_exists.return_value = True
    mock_api.auth_check.return_value = None

    ok, err = verify_hf_repo_checkpointable("KhangTruong/Testing-model")
    assert ok is True
    assert err is None
    mock_api.auth_check.assert_called_with(
        repo_id="KhangTruong/Testing-model", repo_type="model", write=True, token="fake_token"
    )


@patch("sid_unet.checkpoint_sync._get_hf_token", return_value=None)
def test_verify_hf_repo_checkpointable_no_token(mock_token):
    with pytest.raises(PermissionError) as exc_info:
        verify_hf_repo_checkpointable("KhangTruong/Testing-model", token=None, raise_on_error=True)
    assert "No Hugging Face authentication token found" in str(exc_info.value)

    ok, err = verify_hf_repo_checkpointable("KhangTruong/Testing-model", token=None, raise_on_error=False)
    assert ok is False
    assert "No Hugging Face authentication token found" in err


@patch("sid_unet.checkpoint_sync._get_hf_token", return_value="fake_token")
@patch("sid_unet.checkpoint_sync.HfApi")
def test_verify_hf_repo_checkpointable_forbidden(mock_api_cls, mock_token):
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.whoami.return_value = {"name": "KhangTruong"}
    mock_api.repo_exists.return_value = True
    mock_api.auth_check.side_effect = RuntimeError("403 Forbidden: write access denied")

    with pytest.raises(PermissionError) as exc_info:
        verify_hf_repo_checkpointable("openai/clip-vit", raise_on_error=True)
    assert "write access was denied" in str(exc_info.value)

    ok, err = verify_hf_repo_checkpointable("openai/clip-vit", raise_on_error=False)
    assert ok is False
    assert "write access was denied" in err


@patch("sid_unet.checkpoint_sync._get_hf_token", return_value="fake_token")
@patch("sid_unet.checkpoint_sync.HfApi")
def test_verify_hf_repo_checkpointable_creates_missing(mock_api_cls, mock_token):
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.whoami.return_value = {"name": "KhangTruong"}
    mock_api.repo_exists.return_value = False
    mock_api.create_repo.return_value = "https://huggingface.co/KhangTruong/new-model"
    mock_api.auth_check.return_value = None

    ok, err = verify_hf_repo_checkpointable("KhangTruong/new-model", create_if_missing=True)
    assert ok is True
    assert err is None
    assert mock_api.create_repo.called


def test_checkpoint_manager_hf_warning_when_flag_not_on():
    """Verify that CheckpointManager emits a warning when HF checkpointing flag is not on."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.warns(UserWarning, match="Hugging Face model checkpointing is not enabled"):
            mgr = CheckpointManager(
                checkpoint_dir=tmpdir,
                push_to_hub=False,
            )
            assert mgr.push_to_hub is False


@patch("sid_unet.checkpoint_sync.verify_hf_repo_checkpointable", return_value=(True, None))
@patch("sid_unet.checkpoint_sync.push_checkpoint")
def test_checkpoint_manager_tied_hf_checkpointing(mock_push, mock_verify):
    """Verify that regular checkpoint saving triggers Hugging Face push when push_to_hub is enabled."""
    mock_push.return_value = {"repo_id": "KhangTruong/Testing-model", "version": "v1"}

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            checkpoint_dir=tmpdir,
            metric_name="val_iou",
            mode="max",
            push_to_hub=True,
            hf_repo="KhangTruong/Testing-model",
            hf_version="v1",
        )
        assert mgr.push_to_hub is True
        assert mgr.hf_repo == "KhangTruong/Testing-model"
        assert mock_verify.called

        model = UNet(in_channels=3, out_channels=1, features=[4, 8], bilinear=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 1. Regular save
        saved = mgr.save(
            epoch=1,
            model=model,
            optimizer=opt,
            scheduler=None,
            metrics={"val_iou": 0.75},
            config={},
            is_best=True,
            step=10,
        )
        assert "latest" in saved
        assert "best" in saved
        assert mock_push.called
        assert mock_push.call_args[1]["repo"] == "KhangTruong/Testing-model"

        # 2. Periodic save
        mock_push.reset_mock()
        p_saved = mgr.save_periodic(
            epoch=2,
            model=model,
            optimizer=opt,
            step=20,
        )
        assert "periodic" in p_saved
        assert mock_push.called
        assert mock_push.call_args[1]["repo"] == "KhangTruong/Testing-model"


def test_verify_hf_repo_checkpointable_live_testing_model():
    """Live verification against KhangTruong/Testing-model playground repo using environment HF_TOKEN."""
    ok, err = verify_hf_repo_checkpointable("KhangTruong/Testing-model", raise_on_error=False)
    assert ok is True, f"Live verification failed: {err}"

