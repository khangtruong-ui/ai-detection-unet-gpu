"""
Unit and integration tests for sid-copy-config CLI and copy_config utility.
Verifies config duplication, nested overrides, CLI flags, and default merging.
"""

import os
import sys
import tempfile
import yaml
import pytest

from sid_unet.utils.config import copy_config, load_config
from sid_unet.config_copy import cli_main, parse_args


def test_copy_config_basic():
    """Verify basic duplication without overrides preserves YAML keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "source.yaml")
        dst = os.path.join(tmpdir, "dest.yaml")
        content = {
            "project": {"name": "test_exp", "seed": 123},
            "data": {"image_size": [256, 256], "batch_size": 16},
        }
        with open(src, "w") as f:
            yaml.safe_dump(content, f)

        res = copy_config(source_path=src, dest_path=dst)
        assert os.path.exists(dst)
        assert res["project"]["name"] == "test_exp"
        assert res["data"]["image_size"] == [256, 256]

        with open(dst, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded == content


def test_copy_config_with_overrides():
    """Verify dot-notation overrides correctly modify and inject keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "source.yaml")
        dst = os.path.join(tmpdir, "dest.yaml")
        content = {
            "project": {"name": "orig_name", "output_dir": "outputs/orig"},
            "data": {"image_size": [256, 256], "batch_size": 8},
        }
        with open(src, "w") as f:
            yaml.safe_dump(content, f)

        overrides = [
            "data.image_size=[512,512]",
            "project.name=new_name_512",
            "data.batch_size=4",
            "training.epochs=25",
        ]
        res = copy_config(source_path=src, dest_path=dst, overrides=overrides)

        assert res["data"]["image_size"] == [512, 512]
        assert res["project"]["name"] == "new_name_512"
        assert res["data"]["batch_size"] == 4
        assert res["training"]["epochs"] == 25

        with open(dst, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded["data"]["image_size"] == [512, 512]
        assert loaded["project"]["name"] == "new_name_512"
        assert loaded["training"]["epochs"] == 25


def test_copy_config_directory_dest():
    """Verify that specifying a directory as dest appends the source basename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "custom_experiment.yaml")
        dest_dir = os.path.join(tmpdir, "subfolder")
        os.makedirs(dest_dir, exist_ok=True)

        content = {"project": {"name": "dir_test"}}
        with open(src, "w") as f:
            yaml.safe_dump(content, f)

        res = copy_config(source_path=src, dest_path=dest_dir, overrides=["project.name=dir_test_copied"])
        expected_dest = os.path.join(dest_dir, "custom_experiment.yaml")
        assert os.path.exists(expected_dest)
        with open(expected_dest, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded["project"]["name"] == "dir_test_copied"


def test_copy_config_merge_defaults():
    """Verify merge_defaults=True populates DEFAULT_CONFIG fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "minimal.yaml")
        dst = os.path.join(tmpdir, "with_defaults.yaml")
        content = {"project": {"name": "minimal_exp"}}
        with open(src, "w") as f:
            yaml.safe_dump(content, f)

        # merge_defaults=False should preserve only the source keys
        res_no_defaults = copy_config(source_path=src, dest_path=None, merge_defaults=False)
        assert "model" not in res_no_defaults

        # merge_defaults=True should include standard baseline defaults like model, loss, training
        res_with_defaults = copy_config(source_path=src, dest_path=dst, merge_defaults=True)
        assert "model" in res_with_defaults
        assert res_with_defaults["model"]["in_channels"] == 3


def test_copy_config_missing_source():
    """Verify FileNotFoundError is raised when source config does not exist."""
    with pytest.raises(FileNotFoundError):
        copy_config("non_existent_file.yaml", "dest.yaml")


def test_cli_copy_config_positional(capsys):
    """Test sid-copy-config CLI execution using positional arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.yaml")
        dst = os.path.join(tmpdir, "dst.yaml")
        with open(src, "w") as f:
            yaml.safe_dump({"data": {"image_size": [256, 256]}}, f)

        cli_main([src, dst, "--override", "data.image_size=[512,512]"])
        assert os.path.exists(dst)
        with open(dst, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded["data"]["image_size"] == [512, 512]


def test_cli_copy_config_flags():
    """Test sid-copy-config CLI execution using --src and --dst flags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.yaml")
        dst = os.path.join(tmpdir, "dst.yaml")
        with open(src, "w") as f:
            yaml.safe_dump({"training": {"epochs": 10}}, f)

        cli_main(["--src", src, "--dst", dst, "--override", "training.epochs=20"])
        assert os.path.exists(dst)
        with open(dst, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded["training"]["epochs"] == 20


def test_cli_copy_config_print(capsys):
    """Test sid-copy-config CLI with --print flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.yaml")
        with open(src, "w") as f:
            yaml.safe_dump({"project": {"name": "to_print"}}, f)

        cli_main([src, "--print", "--override", "project.name=printed_ok"])
        captured = capsys.readouterr()
        assert "printed_ok" in captured.out


def test_cli_copy_config_missing_dest_exits():
    """Test sid-copy-config CLI exits cleanly when destination is omitted without --print."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.yaml")
        with open(src, "w") as f:
            yaml.safe_dump({"project": {"name": "no_dest"}}, f)

        with pytest.raises(SystemExit) as exc_info:
            cli_main([src])
        assert exc_info.value.code == 1


def test_copied_diffusion_diff_minimized_512_validity():
    """Verify that generated 512x512 config is valid and loads correctly."""
    config_path = "configs/experiments/diffusion_diff_minimized/diffusion_diff_minimized_512.yaml"
    assert os.path.exists(config_path), f"Expected {config_path} to exist"

    cfg = load_config(config_path)
    assert cfg.data.image_size == [512, 512]
    assert cfg.model.name == "diffusion_diff_minimized"
    assert cfg.project.name == "diffusion_diff_minimized_512"
