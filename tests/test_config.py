import os
import tempfile
import pytest
from sid_unet.utils.config import load_config, save_config, apply_overrides, ConfigDict, DEFAULT_CONFIG


def test_default_config_loading():
    cfg = load_config()
    assert isinstance(cfg, ConfigDict)
    assert cfg.model.in_channels == 3
    assert cfg.model.out_channels == 1
    assert cfg.data.dataset_name == "KhangTruong/IMD2020"
    assert cfg.data.streaming is False
    assert cfg.data.batch_size == 16
    assert cfg.data.train_samples_per_epoch == -1
    assert cfg.data.val_samples == -1
    assert cfg.model.aux_classifier is False
    assert cfg.training.auto_batch_size is True
    assert cfg.training.save_latest is False


def test_config_overrides():
    cfg = load_config(overrides=["training.batch_size=32", "data.streaming=true", "model.dropout=0.25"])
    assert cfg.training.batch_size == 32
    assert cfg.data.streaming is True
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
    assert smoke_cfg.data.dataset_name == "KhangTruong/IMD2020"
    assert smoke_cfg.data.train_samples_per_epoch == 4
    assert smoke_cfg.data.val_samples == 2

    quick_cfg = load_config("configs/test_quick.yaml")
    assert quick_cfg.project.name == "sid_unet_quick_test"
    assert quick_cfg.data.dataset_name == "KhangTruong/IMD2020"
    assert quick_cfg.loss.mask_loss_type == "dice"
    assert quick_cfg.model.aux_classifier is False


@pytest.mark.timeout(180)
def test_all_experiment_configs_validity():
    import glob
    import torch
    from sid_unet.models.unet import build_model
    from sid_unet.losses.auxiliary import build_loss

    exp_configs = glob.glob("configs/experiments/**/*.yaml", recursive=True)
    assert len(exp_configs) >= 12, f"Expected at least 12 experiment configs, found {len(exp_configs)}"

    for cfg_file in exp_configs:
        cfg = load_config(cfg_file)
        assert cfg.data.dataset_name in ["KhangTruong/IMD2020", "KhangTruong/BeyondTheBrush", "saberzl/SID_Set", "KhangTruong/COCO-inpainted"]
        assert cfg.data.streaming in [True, False]
        assert cfg.data.batch_size in [1, 2, 4, 8, 16, 32, 64, 128]
        # Ensure sample budgets are -1 for running all dataset
        assert cfg.data.train_samples_per_epoch == -1
        assert cfg.data.val_samples == -1

        cfg.project.device = "cpu"
        # Use fast tiny model for sam3 / gap-sam config loop testing
        if "gap_sam" in str(cfg.model.name).lower() or "gap-sam" in str(cfg.model.name).lower():
            cfg.model.checkpoint_path = None
            cfg.model.load_in_4bit = False
            cfg.model.use_dummy_vae = True
        elif "sam3_distil" in str(cfg.model.name).lower():
            cfg.model.checkpoint_path = None
            cfg.model.load_in_4bit = False
        elif "sam3" in str(cfg.model.name).lower():
            cfg.model.pretrained_model_name_or_path = "yujiepan/sam3-tiny-random"
            cfg.model.load_in_4bit = False
        elif "diffusion" in str(cfg.model.name).lower() or "vae" in str(cfg.model.name).lower():
            cfg.model.use_dummy = True

        # Verify model and loss build cleanly
        try:
            model = build_model(cfg)
        except ImportError as e:
            if "sam" in str(cfg.model.name).lower() or "gap" in str(cfg.model.name).lower():
                continue
            raise e
        loss_fn = build_loss(cfg)

        # Test forward pass with small batch and no grad
        model.eval()
        h, w = cfg.data.image_size
        test_h, test_w = min(h, 64), min(w, 64)
        dev = next(model.parameters()).device
        with torch.no_grad():
            x = torch.randn(1, 3, test_h, test_w, device=dev)
            out = model(x)
        if cfg.model.aux_classifier:
            assert isinstance(out, tuple)
            mask_out = out[0]
            cls_out = out[1]
            assert mask_out.shape == (1, 1, test_h, test_w)
            assert cls_out.shape == (1, cfg.model.num_classes)
        else:
            mask_out = out[0] if isinstance(out, tuple) else out
            assert mask_out.shape == (1, 1, test_h, test_w)

        del model, loss_fn, out
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


