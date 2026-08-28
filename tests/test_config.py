import os
import tempfile
import pytest
from sid_unet.utils.config import load_config, save_config, apply_overrides, ConfigDict, DEFAULT_CONFIG


def test_default_config_loading():
    cfg = load_config()
    assert isinstance(cfg, ConfigDict)
    assert cfg.model.in_channels == 3
    assert cfg.model.out_channels == 1
    assert cfg.data.streaming is True
    assert cfg.model.aux_classifier is True


def test_config_overrides():
    cfg = load_config(overrides=["training.batch_size=32", "data.streaming=false", "model.dropout=0.25"])
    assert cfg.training.batch_size == 32
    assert cfg.data.streaming is False
    assert cfg.model.dropout == 0.25


def test_save_and_load_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_cfg.yaml")
        cfg = load_config(overrides=["project.name=custom_experiment"])
        save_config(cfg, config_path)

        loaded_cfg = load_config(config_path)
        assert loaded_cfg.project.name == "custom_experiment"
        assert loaded_cfg.model.name == "unet"
