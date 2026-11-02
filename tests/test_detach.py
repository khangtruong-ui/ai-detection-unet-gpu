import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from sid_unet.train import parse_args, main as train_main
from sid_unet.cross_eval import parse_args as cross_eval_parse_args, main as cross_eval_main
from sid_unet.cache.cli import parse_args as cache_parse_args, cli_main as cache_cli_main
from sid_unet.cache.manager import AsyncHubUploader, DatasetCacheManager


def test_parse_args_detach_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sid-train", "--config", "configs/default.yaml"])
    args = parse_args()
    assert args.detach is True
    assert args.log_file == ".logs.txt"


@pytest.mark.parametrize("flag", ["--no-detach", "--no_detach", "--foreground", "-f"])
def test_parse_args_foreground_flags(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["sid-train", "--config", "configs/default.yaml", flag])
    args = parse_args()
    assert args.detach is False


def test_parse_args_custom_log_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sid-train", "--config", "configs/default.yaml", "--log-file", "my_custom_log.txt"])
    args = parse_args()
    assert args.detach is True
    assert args.log_file == "my_custom_log.txt"


def test_train_main_detaches_when_enabled(monkeypatch):
    """Verify that when running outside pytest and in_dist, main() detaches via subprocess.Popen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_log_file = os.path.join(tmpdir, "test_output.logs.txt")
        monkeypatch.setattr(
            sys,
            "argv",
            ["sid-train", "--config", "configs/default.yaml", "--log-file", test_log_file],
        )

        # Clear pytest indicators to simulate normal terminal invocation
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr("sid_unet.train.is_dist_avail_and_initialized", lambda: False)

        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("sys.modules", {k: v for k, v in sys.modules.items() if k != "pytest"}):
            result = train_main()

            assert result is not None
            assert result.get("status") == "detached"
            assert result.get("pid") == 99999
            assert result.get("log_file") == os.path.abspath(test_log_file)

            # Check subprocess call arguments
            mock_popen.assert_called_once()
            call_args, call_kwargs = mock_popen.call_args
            cmd = call_args[0]
            assert cmd[0] == sys.executable
            assert cmd[1] == "-m"
            assert cmd[2] == "sid_unet.train"
            assert "--no-detach" in cmd
            assert "--config" in cmd
            assert call_kwargs.get("start_new_session") is True
            assert call_kwargs.get("env", {}).get("SID_UNET_DETACHED") == "1"

        # Check that header was written into test_log_file
        assert os.path.exists(test_log_file)
        with open(test_log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "sid-train detached session started at" in content


def test_train_main_runs_foreground_when_no_detach(monkeypatch):
    """Verify that with --no-detach, main() does NOT detach via subprocess.Popen."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["sid-train", "--config", "configs/default.yaml", "--no-detach"],
    )
    with patch("subprocess.Popen") as mock_popen, \
         patch("sid_unet.train.train_single_run", return_value={"run_name": "mock_run", "best_score": 0.85, "best_epoch": 1, "report_path": "report.json"}):
        result = train_main()
        mock_popen.assert_not_called()
        assert result.get("run_name") == "mock_run"


# =============================================================================
# sid-cross-eval background detachment tests
# =============================================================================

def test_cross_eval_parse_args_detach_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sid-cross-eval", "--cross-configs", "configs/cross-eval/coco-inpainted.yaml", "--checkpoints", "ckpt.pt"],
    )
    args = cross_eval_parse_args()
    assert args.detach is True
    assert args.log_file == ".logs.txt"


@pytest.mark.parametrize("flag", ["--no-detach", "--no_detach", "--foreground", "-f"])
def test_cross_eval_parse_args_foreground_flags(monkeypatch, flag):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sid-cross-eval", "--cross-configs", "configs/cross-eval/coco-inpainted.yaml", "--checkpoints", "ckpt.pt", flag],
    )
    args = cross_eval_parse_args()
    assert args.detach is False


def test_cross_eval_parse_args_custom_log_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sid-cross-eval",
            "--cross-configs", "configs/cross-eval/coco-inpainted.yaml",
            "--checkpoints", "ckpt.pt",
            "--log-file", "eval_output.log",
        ],
    )
    args = cross_eval_parse_args()
    assert args.detach is True
    assert args.log_file == "eval_output.log"


def test_cross_eval_main_detaches_when_enabled(monkeypatch):
    """Verify that when running outside pytest and in_dist, cross_eval main() detaches via subprocess.Popen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_log_file = os.path.join(tmpdir, "test_eval_output.logs.txt")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sid-cross-eval",
                "--cross-configs", "configs/cross-eval/coco-inpainted.yaml",
                "--checkpoints", "ckpt.pt",
                "--log-file", test_log_file,
            ],
        )

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        mock_proc = MagicMock()
        mock_proc.pid = 88888

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("sys.modules", {k: v for k, v in sys.modules.items() if k != "pytest"}):
            result = cross_eval_main()

            assert result is not None
            assert result.get("status") == "detached"
            assert result.get("pid") == 88888
            assert result.get("log_file") == os.path.abspath(test_log_file)

            mock_popen.assert_called_once()
            call_args, call_kwargs = mock_popen.call_args
            cmd = call_args[0]
            assert cmd[0] == sys.executable
            assert cmd[1] == "-m"
            assert cmd[2] == "sid_unet.cross_eval"
            assert "--no-detach" in cmd
            assert "--cross-configs" in cmd
            assert call_kwargs.get("start_new_session") is True
            assert call_kwargs.get("env", {}).get("SID_UNET_DETACHED") == "1"

        assert os.path.exists(test_log_file)
        with open(test_log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "sid-cross-eval detached session started at" in content


def test_cross_eval_main_runs_foreground_when_no_detach(monkeypatch):
    """Verify that with --no-detach, cross_eval main() does NOT detach via subprocess.Popen."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sid-cross-eval",
            "--cross-configs", "configs/cross-eval/coco-inpainted.yaml",
            "--checkpoints", "ckpt.pt",
            "--no-detach",
        ],
    )
    with patch("subprocess.Popen") as mock_popen, \
         patch("sid_unet.cross_eval.expand_checkpoint_patterns", return_value=["/mock/ckpt.pt"]), \
         patch("sid_unet.cross_eval.run_cross_evaluation", return_value={"status": "mock_completed"}):
        result = cross_eval_main()
        mock_popen.assert_not_called()
        assert result == {"status": "mock_completed"}


# =============================================================================
# sid-cache background detachment & --delete-on-update tests
# =============================================================================

def test_cache_parse_args_detach_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sid-cache", "--config", "configs/experiments/diffusion_diff_minimized/default.yaml"],
    )
    args = cache_parse_args()
    assert args.detach is True
    assert args.log_file == ".logs.txt"


@pytest.mark.parametrize("flag", ["--no-detach", "--no_detach", "--foreground", "-f"])
def test_cache_parse_args_foreground_flags(monkeypatch, flag):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sid-cache", "--config", "configs/experiments/diffusion_diff_minimized/default.yaml", flag],
    )
    args = cache_parse_args()
    assert args.detach is False


def test_cache_parse_args_custom_log_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sid-cache",
            "--config", "configs/experiments/diffusion_diff_minimized/default.yaml",
            "--log-file", "cache_custom.log",
        ],
    )
    args = cache_parse_args()
    assert args.detach is True
    assert args.log_file == "cache_custom.log"


def test_cache_cli_main_detaches_when_enabled(monkeypatch):
    """Verify that when running outside pytest and in_dist, cache cli_main() detaches via subprocess.Popen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_log_file = os.path.join(tmpdir, "test_cache_output.logs.txt")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sid-cache",
                "--config", "configs/experiments/diffusion_diff_minimized/default.yaml",
                "--log-file", test_log_file,
            ],
        )

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        mock_proc = MagicMock()
        mock_proc.pid = 77777

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("sys.modules", {k: v for k, v in sys.modules.items() if k != "pytest"}):
            result = cache_cli_main()

            assert result is not None
            assert result.get("status") == "detached"
            assert result.get("pid") == 77777
            assert result.get("log_file") == os.path.abspath(test_log_file)

            mock_popen.assert_called_once()
            call_args, call_kwargs = mock_popen.call_args
            cmd = call_args[0]
            assert cmd[0] == sys.executable
            assert cmd[1] == "-m"
            assert cmd[2] == "sid_unet.cache.cli"
            assert "--no-detach" in cmd
            assert "--config" in cmd
            assert call_kwargs.get("start_new_session") is True
            assert call_kwargs.get("env", {}).get("SID_UNET_DETACHED") == "1"

        assert os.path.exists(test_log_file)
        with open(test_log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "sid-cache detached session started at" in content


def test_cache_cli_main_runs_foreground_when_no_detach(monkeypatch):
    """Verify that with --no-detach, cache cli_main() does NOT detach via subprocess.Popen."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sid-cache",
            "--config", "configs/experiments/diffusion_diff_minimized/default.yaml",
            "--no-detach",
        ],
    )
    with patch("subprocess.Popen") as mock_popen, \
         patch("sid_unet.cache.cli.build_model", return_value=MagicMock()), \
         patch("sid_unet.cache.cli.DatasetCacheManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.cache_all_splits.return_value = {"train": ["s1.parquet"]}
        mock_mgr_cls.return_value = mock_instance

        result = cache_cli_main()
        mock_popen.assert_not_called()
        assert result == {"train": ["s1.parquet"]}


def test_cache_parse_args_delete_on_update_defaults(monkeypatch):
    """Verify that --delete-on-update is True by default in sid-cache."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["sid-cache", "--config", "configs/experiments/diffusion_diff_minimized/default.yaml"],
    )
    args = cache_parse_args()
    assert args.delete_on_update is True
    assert args.delete_local_on_upload is True


@pytest.mark.parametrize("flag", ["--delete-on-update", "--delete-local-on-upload", "--delete-on-upload"])
def test_cache_parse_args_delete_on_update_explicit_flags(monkeypatch, flag):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sid-cache", "--config", "configs/experiments/diffusion_diff_minimized/default.yaml", flag],
    )
    args = cache_parse_args()
    assert args.delete_on_update is True
    assert args.delete_local_on_upload is True


@pytest.mark.parametrize("flag", ["--no-delete-on-update", "--no-delete-local-on-upload", "--no-delete-on-upload", "--keep-local"])
def test_cache_parse_args_no_delete_on_update_flags(monkeypatch, flag):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sid-cache", "--config", "configs/experiments/diffusion_diff_minimized/default.yaml", flag],
    )
    args = cache_parse_args()
    assert args.delete_on_update is False
    assert args.delete_local_on_upload is False


def test_cache_manager_delete_on_update_defaults():
    """Verify that AsyncHubUploader and DatasetCacheManager default to delete_local_on_upload=True."""
    with patch("huggingface_hub.HfApi"):
        uploader = AsyncHubUploader(repo_id="test/repo")
        assert uploader.delete_local_on_upload is True
        assert uploader.delete_on_update is True

        uploader_false = AsyncHubUploader(repo_id="test/repo", delete_local_on_upload=False)
        assert uploader_false.delete_local_on_upload is False
        assert uploader_false.delete_on_update is False

        uploader_alias = AsyncHubUploader(repo_id="test/repo", delete_on_update=False)
        assert uploader_alias.delete_local_on_upload is False
        assert uploader_alias.delete_on_update is False

