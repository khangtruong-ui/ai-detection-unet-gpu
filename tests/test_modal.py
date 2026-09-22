"""
Tests for Modal cloud execution engine:
- Authentication verification & early termination
- Volume provisioning & check
- Output directory organization within volumes
- Cheap GPU selection for testing & mock tests
- Automatic defaulting to Modal when local GPU is unavailable
- CLI commands for sid-modal
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from unittest.mock import MagicMock, patch
import pytest

from sid_unet.modal_runner import (
    DEFAULT_MOUNT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_TEST_DIR,
    DEFAULT_TEST_GPU,
    DEFAULT_TRAIN_GPU,
    DEFAULT_VOLUME_NAME,
    DEFAULT_HF_SECRET_NAME,
    check_hf_token_status,
    ensure_modal_authenticated,
    get_hf_secret,
    get_or_create_volume,
    get_volume_output_dir,
    has_local_gpu,
    is_modal_authenticated,
    run_eval_on_modal,
    run_mock_test_on_modal,
    run_tests_on_modal,
    run_train_on_modal,
)
from sid_unet.train import parse_args as train_parse_args
from sid_unet.evaluate import parse_args as eval_parse_args
from sid_unet.utils.logger import (
    SmartProgressBar,
    create_progress_bar,
    is_modal_environment,
)


# ==============================================================================
# Authentication Detection Tests
# ==============================================================================

def test_modal_authenticated_when_token_present():
    """Verify that is_modal_authenticated returns True when valid credentials are found."""
    mock_cfg = {"token_id": "ak-test12345", "token_secret": "as-test67890"}
    with patch("modal.config.config.get", side_effect=lambda k: mock_cfg.get(k)):
        assert is_modal_authenticated() is True


def test_modal_unauthenticated_when_token_missing():
    """Verify that is_modal_authenticated returns False when token is missing or empty."""
    with patch("modal.config.config.get", return_value=None):
        with patch.dict(os.environ, {"MODAL_TOKEN_ID": "", "MODAL_TOKEN_SECRET": ""}, clear=True):
            assert is_modal_authenticated() is False


def test_ensure_modal_authenticated_exits_on_failure(capsys):
    """Verify that ensure_modal_authenticated terminates execution when unauthenticated."""
    with patch("sid_unet.modal_runner.is_modal_authenticated", return_value=False):
        with pytest.raises(SystemExit) as exc_info:
            ensure_modal_authenticated(exit_on_failure=True)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Modal authentication required" in captured.err
        assert "modal setup" in captured.err


def test_ensure_modal_authenticated_returns_false_when_not_exiting():
    """Verify ensure_modal_authenticated returns False without raising when exit_on_failure=False."""
    with patch("sid_unet.modal_runner.is_modal_authenticated", return_value=False):
        assert ensure_modal_authenticated(exit_on_failure=False) is False


def test_ensure_modal_authenticated_succeeds_when_authenticated():
    """Verify ensure_modal_authenticated returns True when credentials exist."""
    with patch("sid_unet.modal_runner.is_modal_authenticated", return_value=True):
        assert ensure_modal_authenticated(exit_on_failure=True) is True


# ==============================================================================
# Local GPU Detection Tests
# ==============================================================================

def test_has_local_gpu_returns_false_when_no_cuda():
    """Verify has_local_gpu returns False when CUDA is unavailable."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        assert has_local_gpu() is False


def test_has_local_gpu_returns_true_when_cuda_available():
    """Verify has_local_gpu returns True when CUDA device is present."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1

    with patch.dict("sys.modules", {"torch": mock_torch}):
        assert has_local_gpu() is True


# ==============================================================================
# Volume Provisioning & Check Tests
# ==============================================================================

def test_get_or_create_volume_creates_if_missing():
    """Verify get_or_create_volume provisions volume if not existing and returns handle."""
    from modal.volume import VolumeManager

    mock_handle = MagicMock()

    with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
        with patch.object(VolumeManager, "create") as mock_create:
            with patch("modal.Volume.from_name", return_value=mock_handle) as mock_from_name:
                handle = get_or_create_volume("custom-vol")

                mock_create.assert_called_once_with("custom-vol", allow_existing=True, client=None)
                mock_from_name.assert_called_once_with("custom-vol", create_if_missing=True, client=None)
                assert handle == mock_handle


def test_get_or_create_volume_handles_existing_cleanly():
    """Verify get_or_create_volume handles already existing volume without error."""
    from modal.volume import VolumeManager

    mock_handle = MagicMock()
    with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
        with patch.object(VolumeManager, "create", side_effect=Exception("Already exists")):
            with patch("modal.Volume.from_name", return_value=mock_handle):
                handle = get_or_create_volume("existing-vol")
                assert handle == mock_handle


# ==============================================================================
# Volume Output Directory Organization Tests
# ==============================================================================

def test_get_volume_output_dir_default():
    """Verify default output directory in volume is /vol/outputs/RUN."""
    out = get_volume_output_dir()
    assert out == "/vol/outputs/RUN"


def test_get_volume_output_dir_custom_relative():
    """Verify relative output paths are structured under /vol."""
    out = get_volume_output_dir("outputs/RUN/unet_wide_b32")
    assert out == "/vol/outputs/RUN/unet_wide_b32"


def test_get_volume_output_dir_preserves_vol_prefix():
    """Verify paths already beginning with /vol are not doubly prefixed."""
    out = get_volume_output_dir("/vol/custom_experiments")
    assert out == "/vol/custom_experiments"


def test_get_volume_output_dir_test_outputs():
    """Verify test outputs directory organization in volume."""
    out = get_volume_output_dir(subpath="test_outputs/smoke", default_subpath="test_outputs")
    assert out == "/vol/test_outputs/smoke"


# ==============================================================================
# Cheap GPU Selection for Testing & Mock Tests
# ==============================================================================

def test_cheap_gpu_configuration():
    """Verify default GPU allocation distinguishes cheap test GPU (T4) vs high-efficiency train GPU (L40S)."""
    assert DEFAULT_TEST_GPU == "T4"
    assert DEFAULT_TRAIN_GPU == "L40S"


def test_eval_uses_cheap_gpu_by_default():
    """Verify evaluation defaults to cheap T4 GPU."""
    test_args = ["sid-eval", "--checkpoint", "checkpoint.pt"]
    with patch("sys.argv", test_args):
        args = eval_parse_args()
        assert args.modal_gpu == "T4"


def test_train_uses_l40s_by_default():
    """Verify training defaults to L40S GPU (cheapest price over TFLOPS under $2/hr)."""
    test_args = ["sid-train", "--config", "configs/train_streaming.yaml"]
    with patch("sys.argv", test_args):
        args = train_parse_args()
        assert args.modal_gpu == "L40S"


# ==============================================================================
# Defaulting to Modal when No Local GPU Exists
# ==============================================================================

def test_train_defaults_to_modal_when_no_gpu():
    """Verify sid-train defaults to Modal execution when local GPU is absent."""
    test_args = ["sid-train", "--config", "configs/test_smoke.yaml"]
    with patch("sys.argv", test_args):
        with patch("sid_unet.modal_runner.has_local_gpu", return_value=False):
            with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
                with patch("sid_unet.modal_runner.run_train_on_modal", return_value=[{"run_name": "test"}]) as mock_run:
                    from sid_unet.train import main as train_main
                    result = train_main()
                    assert mock_run.called
                    assert result == [{"run_name": "test"}]


def test_train_respects_force_local_flag():
    """Verify --local disables defaulting to Modal even without local GPU."""
    test_args = ["sid-train", "--config", "configs/test_smoke.yaml", "--local"]
    with patch("sys.argv", test_args):
        with patch("sid_unet.modal_runner.has_local_gpu", return_value=False):
            with patch("sid_unet.modal_runner.run_train_on_modal") as mock_modal:
                from sid_unet.train import main as train_main
                with pytest.raises(SystemExit):  # Exits because torch is not installed locally
                    train_main()
                assert not mock_modal.called


def test_train_unauthenticated_modal_ends_execution(capsys):
    """Verify unauthenticated Modal detection terminates execution immediately."""
    test_args = ["sid-train", "--config", "configs/test_smoke.yaml"]
    with patch("sys.argv", test_args):
        with patch("sid_unet.modal_runner.has_local_gpu", return_value=False):
            with patch("sid_unet.modal_runner.is_modal_authenticated", return_value=False):
                from sid_unet.train import main as train_main
                with pytest.raises(SystemExit) as exc:
                    train_main()
                assert exc.value.code == 1

                captured = capsys.readouterr()
                assert "Modal authentication required" in captured.err


def test_eval_defaults_to_modal_when_no_gpu():
    """Verify sid-eval defaults to Modal execution with cheap GPU when local GPU is absent."""
    test_args = ["sid-eval", "--checkpoint", "checkpoint_best.pt"]
    with patch("sys.argv", test_args):
        with patch("sid_unet.modal_runner.has_local_gpu", return_value=False):
            with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
                with patch("sid_unet.modal_runner.run_eval_on_modal", return_value=[{"miou": 0.85}]) as mock_eval:
                    from sid_unet.evaluate import main as eval_main
                    result = eval_main()
                    assert mock_eval.called
                    assert result == [{"miou": 0.85}]


# ==============================================================================
# Mock Launchers & Execution Tests
# ==============================================================================

def test_run_train_on_modal_invokes_remote():
    """Verify run_train_on_modal resolves configs and calls remote function."""
    mock_args = argparse.Namespace(
        config=["configs/test_smoke.yaml"],
        modal_volume=DEFAULT_VOLUME_NAME,
        modal_gpu="T4",
        override=[],
        output_dir="outputs/RUN/smoke",
        resume=None,
        resume_repo=None,
        auto_resume=True,
        skip_collision=True,
        save_latest=None,
        batch_size=None,
        auto_batch_size=None,
        val_samples_per_epoch=None,
        checkpoint_period=None,
        checkpoint_steps=None,
    )
    with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
        with patch("sid_unet.modal_runner.get_or_create_volume"):
            with patch("sid_unet.modal_runner.app.run") as mock_app_run:
                with patch("sid_unet.modal_runner.train_remote_t4.remote", return_value=[{"score": 0.9}]) as mock_remote:
                    res = run_train_on_modal(mock_args, gpu="T4")
                    assert mock_remote.called
                    assert res == [{"score": 0.9}]


def test_run_eval_on_modal_invokes_remote():
    """Verify run_eval_on_modal routes to eval_remote_t4."""
    mock_args = argparse.Namespace(
        checkpoint=["checkpoint.pt"],
        config="configs/evaluate.yaml",
        modal_volume=DEFAULT_VOLUME_NAME,
        modal_gpu="T4",
        split="test",
        samples=10,
        batch_size=2,
        output_dir="outputs/eval",
        threshold=0.5,
        min_area=0,
        morphology="none",
        segment=None,
        override=[],
    )
    with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
        with patch("sid_unet.modal_runner.get_or_create_volume"):
            with patch("sid_unet.modal_runner.app.run") as mock_app_run:
                with patch("sid_unet.modal_runner.eval_remote_t4.remote", return_value=[{"miou": 0.88}]) as mock_remote:
                    res = run_eval_on_modal(mock_args)
                    assert mock_remote.called
                    assert res == [{"miou": 0.88}]


def test_run_mock_test_on_modal_invokes_remote():
    """Verify run_mock_test_on_modal triggers mock_test_remote_t4."""
    with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
        with patch("sid_unet.modal_runner.get_or_create_volume"):
            with patch("sid_unet.modal_runner.app.run"):
                with patch("sid_unet.modal_runner.mock_test_remote_t4.remote", return_value={"status": "success"}) as mock_remote:
                    res = run_mock_test_on_modal()
                    assert mock_remote.called
                    assert res == {"status": "success"}


def test_run_tests_on_modal_invokes_remote():
    """Verify run_tests_on_modal triggers run_pytest_remote_t4."""
    with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
        with patch("sid_unet.modal_runner.get_or_create_volume"):
            with patch("sid_unet.modal_runner.app.run"):
                with patch("sid_unet.modal_runner.run_pytest_remote_t4.remote", return_value={"returncode": 0, "stdout": "PASSED"}) as mock_remote:
                    res = run_tests_on_modal(pytest_args=["tests/test_config.py"])
                    assert mock_remote.called
                    assert res["returncode"] == 0


def test_run_train_on_modal_defaults_to_l40s():
    """Verify run_train_on_modal defaults to train_remote_l40s."""
    mock_args = argparse.Namespace(
        config=["configs/test_smoke.yaml"],
        modal_volume=DEFAULT_VOLUME_NAME,
        modal_gpu="L40S",
        override=[],
        output_dir="outputs/RUN/smoke",
        resume=None,
        resume_repo=None,
        auto_resume=True,
        skip_collision=True,
        save_latest=None,
        batch_size=None,
        auto_batch_size=None,
        val_samples_per_epoch=None,
        checkpoint_period=None,
        checkpoint_steps=None,
    )
    with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
        with patch("sid_unet.modal_runner.get_or_create_volume"):
            with patch("sid_unet.modal_runner.app.run"):
                with patch("sid_unet.modal_runner.train_remote_l40s.remote", return_value=[{"score": 0.95}]) as mock_remote:
                    res = run_train_on_modal(mock_args)
                    assert mock_remote.called
                    assert res == [{"score": 0.95}]


def test_run_train_on_modal_routes_to_l4():
    """Verify run_train_on_modal routes to train_remote_l4 when requested."""
    mock_args = argparse.Namespace(
        config=["configs/test_smoke.yaml"],
        modal_volume=DEFAULT_VOLUME_NAME,
        modal_gpu="L4",
        override=[],
        output_dir="outputs/RUN/smoke",
        resume=None,
        resume_repo=None,
        auto_resume=True,
        skip_collision=True,
        save_latest=None,
        batch_size=None,
        auto_batch_size=None,
        val_samples_per_epoch=None,
        checkpoint_period=None,
        checkpoint_steps=None,
    )
    with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
        with patch("sid_unet.modal_runner.get_or_create_volume"):
            with patch("sid_unet.modal_runner.app.run"):
                with patch("sid_unet.modal_runner.train_remote_l4.remote", return_value=[{"score": 0.93}]) as mock_remote:
                    res = run_train_on_modal(mock_args, gpu="L4")
                    assert mock_remote.called
                    assert res == [{"score": 0.93}]


# ==============================================================================
# Progress Bar & Modal Environment Logging Tests
# ==============================================================================

def test_is_modal_environment_detection():
    """Verify is_modal_environment correctly detects Modal env vars and modes."""
    with patch.dict(os.environ, {"MODAL_TASK_ID": "task-12345"}, clear=True):
        assert is_modal_environment() is True

    with patch.dict(os.environ, {"MODAL_LOG_FORMAT": "PLAIN"}, clear=True):
        assert is_modal_environment() is True

    with patch.dict(os.environ, {"SID_PROGRESS_MODE": "clean"}, clear=True):
        assert is_modal_environment() is True

    with patch.dict(os.environ, {"SID_PROGRESS_MODE": "tqdm"}, clear=True):
        assert is_modal_environment() is False


def test_smart_progress_bar_clean_mode():
    """Verify SmartProgressBar clean logging mode outputs periodic structured logs without \\r spam."""
    test_logger = logging.getLogger("test_clean_logger")
    test_logger.setLevel(logging.INFO)
    records = []

    class TestHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = TestHandler()
    test_logger.addHandler(handler)

    try:
        items = list(range(10))
        pbar = create_progress_bar(
            items,
            desc="CleanTest",
            total=10,
            log_interval=2,
            min_interval=0.0,
            mode="clean",
            logger=test_logger,
        )
        for i in pbar:
            pbar.set_postfix({"loss": f"{1.0 / (i + 1):.4f}"})

        pbar.close()

        assert len(records) > 0
        # Check formatted message contents
        first_log = records[0]
        assert "CleanTest [Step" in first_log
        assert "loss:" in first_log
        assert "\r" not in first_log
    finally:
        test_logger.removeHandler(handler)


def test_smart_progress_bar_manual_update_and_context_manager():
    """Verify SmartProgressBar works as context manager with manual updates."""
    test_logger = logging.getLogger("test_cm_logger")
    test_logger.setLevel(logging.INFO)
    records = []

    class TestHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = TestHandler()
    test_logger.addHandler(handler)

    try:
        with create_progress_bar(total=5, desc="ManualTest", log_interval=1, min_interval=0.0, mode="clean", logger=test_logger) as pbar:
            pbar.set_description("UpdatedDesc")
            for _ in range(5):
                pbar.update(1)
        assert len(records) >= 5
        assert "UpdatedDesc [Step 5/5 (100.0%)]" in records[-1]
    finally:
        test_logger.removeHandler(handler)


# ==============================================================================
# Hugging Face Authentication & Secret Tests
# ==============================================================================

def test_huggingface_secret_configuration():
    """Verify default HF secret name is 'huggingface' and get_hf_secret returns valid Modal Secret."""
    assert DEFAULT_HF_SECRET_NAME == "huggingface"
    sec = get_hf_secret()
    assert sec is not None


def test_check_hf_token_status_from_env():
    """Verify check_hf_token_status detects HF_TOKEN from environment."""
    with patch.dict(os.environ, {"HF_TOKEN": "hf_dummytesttoken123456789"}, clear=True):
        status = check_hf_token_status()
        assert status["authenticated"] is True
        assert "HF_TOKEN" in status["source"]
        assert status["token_preview"] is not None


def test_check_hf_token_status_without_token():
    """Verify check_hf_token_status indicates Modal Secret fallback when no local token exists."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isfile", return_value=False):
            status = check_hf_token_status()
            assert status["authenticated"] is False
            assert "Modal Secret" in status["source"]
            assert status["secret_name"] == "huggingface"


def test_cli_hf_token_argument_parsing():
    """Verify --hf-token argument is parsed correctly by train and eval CLIs."""
    test_train_args = ["sid-train", "--config", "configs/test_smoke.yaml", "--hf-token", "hf_mycustomtoken"]
    with patch("sys.argv", test_train_args):
        parsed_train = train_parse_args()
        assert parsed_train.hf_token == "hf_mycustomtoken"

    test_eval_args = ["sid-eval", "--checkpoint", "checkpoint.pt", "--hf-token", "hf_mycustomtoken"]
    with patch("sys.argv", test_eval_args):
        parsed_eval = eval_parse_args()
        assert parsed_eval.hf_token == "hf_mycustomtoken"


def test_run_train_on_modal_propagates_hf_token():
    """Verify run_train_on_modal sets HF_TOKEN in environment when passed via args."""
    mock_args = argparse.Namespace(
        config=["configs/test_smoke.yaml"],
        modal_volume=DEFAULT_VOLUME_NAME,
        modal_gpu="L40S",
        hf_token="hf_test_token_propagation",
        override=[],
        output_dir="outputs/RUN/smoke",
        resume=None,
        resume_repo=None,
        auto_resume=True,
        skip_collision=True,
        save_latest=None,
        batch_size=None,
        auto_batch_size=None,
        val_samples_per_epoch=None,
        checkpoint_period=None,
        checkpoint_steps=None,
    )
    with patch.dict(os.environ, {}, clear=True):
        with patch("sid_unet.modal_runner.ensure_modal_authenticated", return_value=True):
            with patch("sid_unet.modal_runner.get_or_create_volume"):
                with patch("sid_unet.modal_runner.app.run"):
                    with patch("sid_unet.modal_runner.train_remote_l40s.remote", return_value=[{"score": 0.9}]) as mock_remote:
                        res = run_train_on_modal(mock_args)
                        assert mock_remote.called
                        assert os.environ.get("HF_TOKEN") == "hf_test_token_propagation"


