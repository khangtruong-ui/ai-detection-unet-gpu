"""
Unit tests for 8-bit hardware and library compatibility checking and 8-bit optimizer training.
"""

from unittest.mock import patch
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sid_unet.training.trainer import Trainer
from sid_unet.utils.compatibility import (
    check_8bit_compatibility,
    format_compatibility_table,
    installation_check_8bit,
    validate_8bit_environment,
)
from sid_unet.utils.config import load_config


class TinyDataset(Dataset):
    def __init__(self, size=4, img_size=(32, 32)):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        lbl = idx % 3
        img = torch.randn(3, *self.img_size)
        mask = torch.zeros(1, *self.img_size)
        if lbl == 1:
            mask = torch.ones(1, *self.img_size)
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(lbl, dtype=torch.long),
            "img_id": f"syn_{idx}",
        }


def test_check_8bit_compatibility_structure():
    """Verify that check_8bit_compatibility returns all expected diagnostic fields."""
    compatible, details = check_8bit_compatibility(verbose=False)
    assert isinstance(compatible, bool)
    assert isinstance(details, dict)

    required_keys = [
        "compatible",
        "cuda_available",
        "device_name",
        "device_index",
        "compute_capability",
        "compute_capability_str",
        "bitsandbytes_installed",
        "bitsandbytes_version",
        "bitsandbytes_functional",
        "optimizer_8bit_supported",
        "model_8bit_supported",
        "fp8_hardware_supported",
        "fp8_torch_supported",
        "fp8_native_supported",
        "supported_optimizers",
        "message",
    ]
    for key in required_keys:
        assert key in details, f"Missing key in 8-bit details: {key}"

    table = format_compatibility_table(details)
    assert isinstance(table, str)
    assert "SID-UNet 8-Bit Training Environment Compatibility" in table


def test_installation_check_8bit_runs():
    """Verify installation check hook executes without exceptions."""
    result = installation_check_8bit()
    assert isinstance(result, bool)


def test_validate_8bit_environment_behavior():
    """Test validation behavior with and without error raising."""
    is_ok = validate_8bit_environment(raise_error=False)
    assert isinstance(is_ok, bool)

    if not is_ok:
        with pytest.raises(RuntimeError, match="8-bit training mode requested"):
            validate_8bit_environment(raise_error=True)


def test_check_8bit_mocked_cpu():
    """Test 8-bit compatibility behavior when CUDA is unavailable (e.g. CPU environment)."""
    with patch("torch.cuda.is_available", return_value=False):
        compatible, details = check_8bit_compatibility(verbose=False)
        assert compatible is False
        assert details["cuda_available"] is False
        assert details["optimizer_8bit_supported"] is False
        assert "CUDA is not available" in details["message"]


def test_trainer_with_8bit_optimizer(tmp_path):
    """Test Trainer initialization and step with 8-bit AdamW optimizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(overrides=[
        f"project.output_dir={tmp_path}",
        f"project.device={device}",
        "training.epochs=1",
        "training.batch_size=2",
        "training.optimizer=adamw8bit",
        "training.fallback_on_unsupported_8bit=true",
        "model.features=[8, 16]",
        "data.image_size=[32, 32]",
        "logging.log_interval=1",
        "training.amp=false",
        "logging.measure_network=false",
    ])

    ds = TinyDataset(size=2, img_size=(32, 32))
    loader = DataLoader(ds, batch_size=2)

    trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)

    if torch.cuda.is_available() and trainer.is_8bit_compatible:
        import bitsandbytes as bnb
        assert isinstance(trainer.optimizer, bnb.optim.AdamW8bit)
    else:
        # Gracefully falls back to AdamW on CPU
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    # Verify forward and loss computation
    batch = next(iter(loader))
    loss, loss_dict = trainer._step_batch_train(batch)
    assert loss.item() > 0
    assert "total_loss" in loss_dict


def test_trainer_with_paged_8bit_optimizer(tmp_path):
    """Test Trainer initialization with PagedAdamW8bit optimizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(overrides=[
        f"project.output_dir={tmp_path}",
        f"project.device={device}",
        "training.epochs=1",
        "training.batch_size=2",
        "training.optimizer=paged_adamw8bit",
        "training.fallback_on_unsupported_8bit=true",
        "model.features=[8, 16]",
        "data.image_size=[32, 32]",
        "logging.log_interval=1",
        "training.amp=false",
        "logging.measure_network=false",
    ])

    ds = TinyDataset(size=2, img_size=(32, 32))
    loader = DataLoader(ds, batch_size=2)

    trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)

    if torch.cuda.is_available() and trainer.is_8bit_compatible:
        import bitsandbytes as bnb
        assert isinstance(trainer.optimizer, bnb.optim.PagedAdamW8bit)
    else:
        assert isinstance(trainer.optimizer, torch.optim.AdamW)


def test_trainer_with_use_8bit_flag(tmp_path):
    """Test Trainer with use_8bit_optimizer config flag."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(overrides=[
        f"project.output_dir={tmp_path}",
        f"project.device={device}",
        "training.epochs=1",
        "training.batch_size=2",
        "training.use_8bit_optimizer=true",
        "training.fallback_on_unsupported_8bit=true",
        "model.features=[8, 16]",
        "data.image_size=[32, 32]",
        "logging.measure_network=false",
    ])

    ds = TinyDataset(size=2, img_size=(32, 32))
    loader = DataLoader(ds, batch_size=2)

    trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)
    if torch.cuda.is_available() and trainer.is_8bit_compatible:
        import bitsandbytes as bnb
        assert isinstance(trainer.optimizer, bnb.optim.AdamW8bit)
