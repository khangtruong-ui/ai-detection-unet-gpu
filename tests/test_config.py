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


def test_test_configs_loading():
    smoke_cfg = load_config("configs/test_smoke.yaml")
    assert smoke_cfg.project.name == "sid_unet_smoke_test"
    assert smoke_cfg.data.train_samples_per_epoch == 4
    assert smoke_cfg.data.val_samples == 2

    quick_cfg = load_config("configs/test_quick.yaml")
    assert quick_cfg.project.name == "sid_unet_quick_test"
    assert quick_cfg.loss.mask_loss_type == "dice"
    assert quick_cfg.model.aux_classifier is False


def test_all_experiment_configs_validity():
    import glob
    import torch
    from sid_unet.models.unet import build_model
    from sid_unet.losses.auxiliary import build_loss

    exp_configs = glob.glob("configs/experiments/*.yaml")
    assert len(exp_configs) >= 8, f"Expected at least 8 experiment configs, found {len(exp_configs)}"

    for cfg_file in exp_configs:
        cfg = load_config(cfg_file)
        assert cfg.data.batch_size >= 16
        # Ensure sample budgets are strictly bounded (not full dataset passes)
        assert cfg.data.train_samples_per_epoch > 0
        assert cfg.data.val_samples > 0

        # Verify model and loss build cleanly
        model = build_model(cfg)
        loss_fn = build_loss(cfg)

        # Test forward pass with small batch
        h, w = cfg.data.image_size
        x = torch.randn(2, 3, h, w)
        out = model(x)
        if cfg.model.aux_classifier:
            assert isinstance(out, tuple)
            mask_out, cls_out = out
            assert mask_out.shape == (2, 1, h, w)
            assert cls_out.shape == (2, cfg.model.num_classes)
        else:
            assert out.shape == (2, 1, h, w)


