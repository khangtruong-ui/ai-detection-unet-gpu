"""
Comprehensive test suite for Bootstrapping v1.0 kickstarting in SID-UNet.

Tests:
1. Configuration loading and CLI argument parsing for Bootstrapping v1.0.
2. Parameter initialization schemes (Kaiming normal/uniform, Xavier, zero-bias).
3. Selective freezing mechanisms and clean parameter release across architectures (UNet, DiffusionDiff, DiffusionDiffV2).
4. Bootstrap subset DataLoader generation from map-style and iterable datasets.
5. End-to-end kickstart training execution in Trainer with nn-toolbox diagnostic verification.
6. Validation of dedicated DiffusionDiff v1 and v2 bootstrap configuration files.
"""

import copy
import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sid_unet.models.unet import build_model
from sid_unet.train import parse_args
from sid_unet.training.bootstrapping import (
    apply_bootstrap_freeze,
    create_bootstrap_loader,
    initialize_bootstrap_parameters,
    release_bootstrap_freeze,
    run_bootstrapping_phase,
)
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import DEFAULT_CONFIG, load_config


class SyntheticSegmentationLoss(nn.Module):
    """Simple segmentation loss for testing."""

    def forward(self, outputs, masks, labels=None):
        out_t = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        loss = nn.functional.mse_loss(torch.sigmoid(out_t), masks.float())
        pred_binary = (torch.sigmoid(out_t) > 0.5).float()
        intersection = (pred_binary * masks).sum().item()
        union = (pred_binary + masks).clamp(0, 1).sum().item()
        iou = (intersection + 1e-6) / (union + 1e-6)
        return loss, {"total_loss": float(loss.item()), "iou": float(iou)}


def _create_synthetic_loader(num_samples: int = 16, image_size: int = 64, batch_size: int = 4):
    images = torch.randn(num_samples, 3, image_size, image_size)
    masks = (torch.rand(num_samples, 1, image_size, image_size) > 0.5).float()
    dataset = [{"image": images[i], "mask": masks[i], "label": torch.tensor(0)} for i in range(num_samples)]
    return DataLoader(dataset, batch_size=batch_size)


def test_bootstrap_default_config():
    """Verify that bootstrapping is disabled by default in DEFAULT_CONFIG and loads cleanly."""
    assert "bootstrapping" in DEFAULT_CONFIG
    boot_cfg = DEFAULT_CONFIG["bootstrapping"]
    assert boot_cfg["enabled"] is False
    assert boot_cfg["run_bootstrap"] is False
    assert boot_cfg["epochs"] == 5
    assert boot_cfg["num_samples"] == 512
    assert boot_cfg["freeze_strategy"] == "auto"
    assert boot_cfg["initialization"] == "kaiming_normal"


def test_cli_bootstrap_argument_parsing(monkeypatch):
    """Verify CLI argument parsing for Bootstrapping v1.0 flags."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/default.yaml",
            "--run-bootstrap",
            "--bootstrap-epochs",
            "7",
            "--bootstrap-examples",
            "1024",
            "--bootstrap-lr",
            "0.0005",
            "--bootstrap-strategy",
            "encoder",
            "--bootstrap-init",
            "xavier_normal",
            "--bootstrap-target-score",
            "0.65",
        ],
    )
    args = parse_args()
    assert args.run_bootstrap is True
    assert args.bootstrap_epochs == 7
    assert args.bootstrap_examples == 1024
    assert args.bootstrap_lr == 0.0005
    assert args.bootstrap_strategy == "encoder"
    assert args.bootstrap_init == "xavier_normal"
    assert args.bootstrap_target_score == 0.65

    # Test explicit disable flag
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/default.yaml", "--no-run-bootstrap"],
    )
    args2 = parse_args()
    assert args2.run_bootstrap is False


def test_initialize_bootstrap_parameters():
    """Verify that parameter initialization updates weights and zeroes biases."""
    conv = nn.Conv2d(4, 8, kernel_size=3, bias=True)
    linear = nn.Linear(8, 4, bias=True)
    norm = nn.BatchNorm2d(8)

    model = nn.Sequential(conv, norm, nn.ReLU(), nn.Flatten(), linear)

    # Initialize with kaiming_normal
    initialize_bootstrap_parameters(model, initialization="kaiming_normal", unfrozen_only=False)
    assert torch.all(conv.bias == 0)
    assert torch.all(linear.bias == 0)
    assert torch.all(norm.weight == 1)
    assert torch.all(norm.bias == 0)

    # Initialize with xavier_uniform
    initialize_bootstrap_parameters(model, initialization="xavier_uniform", unfrozen_only=False)
    assert torch.all(conv.bias == 0)
    assert torch.all(linear.bias == 0)


def test_apply_and_release_bootstrap_freeze_unet():
    """Verify selective freezing and full release in UNet."""
    model_cfg = {
        "name": "unet",
        "in_channels": 3,
        "out_channels": 1,
        "features": [16, 32],
        "bilinear": True,
        "aux_classifier": True,
        "num_classes": 3,
    }
    model = build_model(model_cfg)

    # Initially, all parameters should be trainable
    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable

    # Apply auto bootstrap freeze: encoder frozen, decoder and classifier trainable
    saved_states = apply_bootstrap_freeze(model, strategy="auto")

    frozen_params = [name for name, p in model.named_parameters() if not p.requires_grad]
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]

    assert len(frozen_params) > 0, "Encoder layers should be frozen"
    assert len(trainable_params) > 0, "Decoder/classifier layers should remain trainable"
    assert any("inc" in name or "down" in name for name in frozen_params)
    assert any("up" in name or "outc" in name or "classifier" in name for name in trainable_params)

    # Release bootstrap freeze: restore original trainable status
    release_bootstrap_freeze(model, saved_states)
    restored_all_trainable = all(p.requires_grad for p in model.parameters())
    assert restored_all_trainable, "All parameters should be released back to trainable"


def test_apply_and_release_bootstrap_freeze_diffusion_diff():
    """Verify selective freezing and release in DiffusionDiff."""
    model_cfg = {
        "name": "diffusion_diff",
        "use_dummy": True,
        "dummy_vae_channels": [32, 64],
        "dummy_unet_channels": [32, 64],
        "autoencoder_trainable": True,
        "aux_classifier": True,
        "decoder": {"channels": [64, 32, 16]},
    }
    model = build_model(model_cfg)

    # In model, vae was trainable, diffuser was frozen
    assert any("vae" in name and p.requires_grad for name, p in model.named_parameters())
    assert all(not p.requires_grad for name, p in model.named_parameters() if "diffuser" in name)

    saved_states = apply_bootstrap_freeze(model, strategy="auto")

    # During bootstrap: vae and diffuser frozen, decoder and classifier trainable
    vae_frozen = all(not p.requires_grad for name, p in model.named_parameters() if "vae" in name)
    diffuser_frozen = all(not p.requires_grad for name, p in model.named_parameters() if "diffuser" in name)
    decoder_trainable = any("decoder" in name and p.requires_grad for name, p in model.named_parameters())
    assert vae_frozen, "VAE must be frozen during kickstarting"
    assert diffuser_frozen, "Diffuser must be frozen during kickstarting"
    assert decoder_trainable, "Decoder must be trainable during kickstarting"

    # Release: vae restored to trainable, diffuser remains frozen
    release_bootstrap_freeze(model, saved_states)
    assert any("vae" in name and p.requires_grad for name, p in model.named_parameters())
    assert all(not p.requires_grad for name, p in model.named_parameters() if "diffuser" in name)


def test_apply_and_release_bootstrap_freeze_diffusion_diff_v2():
    """Verify selective freezing and release in DiffusionDiffV2."""
    model_cfg = {
        "name": "diffusion_diff_v2",
        "use_dummy": True,
        "dummy_vae_channels": [32, 64],
        "dummy_unet_channels": [32, 64],
        "freeze_encoder": False,  # VAE trainable
        "use_perpendicular_skips": True,
        "decoder": {"channels": [64, 32, 16], "z_norm": "groupnorm"},
    }
    model = build_model(model_cfg)

    saved_states = apply_bootstrap_freeze(model, strategy="auto")

    # During bootstrap: vae frozen, decoder trainable
    vae_frozen = all(not p.requires_grad for name, p in model.named_parameters() if "vae" in name)
    decoder_trainable = any("decoder" in name and p.requires_grad for name, p in model.named_parameters())
    assert vae_frozen
    assert decoder_trainable

    # Release
    release_bootstrap_freeze(model, saved_states)
    assert any("vae" in name and p.requires_grad for name, p in model.named_parameters())


def test_create_bootstrap_loader():
    """Verify subset DataLoader creation for both map-style and streaming loaders."""
    # 1. Map-style loader
    x = torch.randn(50, 3, 32, 32)
    y = torch.randint(0, 2, (50, 1, 32, 32)).float()
    map_loader = DataLoader(TensorDataset(x, y), batch_size=8)

    boot_loader = create_bootstrap_loader(map_loader, num_samples=16, batch_size=4)
    total_samples = len(boot_loader.dataset)
    assert total_samples == 16
    assert boot_loader.batch_size == 4

    # 2. Iterable / synthetic dictionary loader
    synth_loader = _create_synthetic_loader(num_samples=20, image_size=32, batch_size=5)
    boot_loader2 = create_bootstrap_loader(synth_loader, num_samples=10, batch_size=5)
    batches = list(boot_loader2)
    collected_count = sum(b["image"].shape[0] for b in batches)
    assert collected_count == 10


def test_trainer_bootstrapping_e2e_with_nn_toolbox():
    """Verify end-to-end kickstart training in Trainer with nn-toolbox verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("configs/test_smoke.yaml")
        config.project.output_dir = tmpdir
        config.training.epochs = 1
        config.training.debug_mode = "light"
        config.training.auto_batch_size = False
        config.data.batch_size = 2

        # Configure Bootstrapping v1.0
        config.bootstrapping = {
            "enabled": True,
            "run_bootstrap": True,
            "epochs": 3,
            "num_samples": 8,
            "freeze_strategy": "auto",
            "initialization": "kaiming_normal",
            "target_score": 0.30,
            "min_loss_drop": 0.05,
            "early_stopping": True,
        }

        train_loader = _create_synthetic_loader(num_samples=8, image_size=32, batch_size=2)
        val_loader = _create_synthetic_loader(num_samples=8, image_size=32, batch_size=2)

        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=SyntheticSegmentationLoss(),
        )

        assert trainer.bootstrap_cfg.get("enabled", False) is True

        results = trainer.train()
        assert results is not None
        assert "bootstrapping" in results
        boot_res = results["bootstrapping"]
        assert boot_res is not None
        assert boot_res["enabled"] is True
        assert boot_res["epochs_trained"] >= 1
        assert "initial_loss" in boot_res
        assert "final_loss" in boot_res
        assert "loss_drop" in boot_res
        assert boot_res["frozen_param_grad_leak"] is False

        # nn-toolbox verification integration
        assert "nn_toolbox_verified" in boot_res
        assert "diagnostic_findings" in boot_res
        findings = boot_res["diagnostic_findings"]
        assert len(findings) > 0
        categories = {f["category"] for f in findings}
        assert "bootstrap" in categories

        # Check that after bootstrapping completes, full model trained normally
        assert results["best_epoch"] >= 1
        assert os.path.exists(results["report_path"])


def test_dedicated_diffusion_diff_bootstrap_configs():
    """Verify that both dedicated bootstrap configs exist and load cleanly."""
    v1_cfg_path = "configs/experiments/diffusion_diff/diffusion_diff_bootstrap.yaml"
    v2_cfg_path = "configs/experiments/diffusion_diff_v2/diffusion_diff_v2_bootstrap.yaml"

    assert os.path.exists(v1_cfg_path), f"Missing {v1_cfg_path}"
    assert os.path.exists(v2_cfg_path), f"Missing {v2_cfg_path}"

    cfg1 = load_config(v1_cfg_path)
    assert cfg1.model.name == "diffusion_diff"
    assert cfg1.bootstrapping.enabled is True
    assert cfg1.bootstrapping.run_bootstrap is True
    assert cfg1.bootstrapping.num_samples == 512
    assert cfg1.bootstrapping.epochs == 5

    cfg2 = load_config(v2_cfg_path)
    assert cfg2.model.name == "diffusion_diff_v2"
    assert cfg2.bootstrapping.enabled is True
    assert cfg2.bootstrapping.run_bootstrap is True
    assert cfg2.bootstrapping.num_samples == 512
    assert cfg2.bootstrapping.epochs == 5
