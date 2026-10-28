import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from sid_unet.train import parse_args, main as train_main


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
