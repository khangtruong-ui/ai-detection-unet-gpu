import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader
from sid_unet.dataset.loader import process_raw_sample
from sid_unet.dataset.transforms import get_transforms


def test_process_raw_sample_labels():
    transform = get_transforms(image_size=(128, 128), is_train=False)

    # Label 0: Real
    raw_sample_0 = {
        "img_id": "test_real_001",
        "image": Image.new("RGB", (200, 200), color=(120, 120, 120)),
        "mask": None,
        "label": 0,
    }
    processed_0 = process_raw_sample(raw_sample_0, transform=transform, target_image_size=(128, 128))
    assert processed_0["image"].shape == (3, 128, 128)
    assert processed_0["mask"].shape == (1, 128, 128)
    assert torch.all(processed_0["mask"] == 0.0)
    assert processed_0["label"].item() == 0
    assert processed_0["img_id"] == "test_real_001"

    # Label 1: Full synthetic
    raw_sample_1 = {
        "img_id": "test_syn_001",
        "image": Image.new("RGB", (200, 200), color=(200, 50, 50)),
        "mask": None,
        "label": 1,
    }
    processed_1 = process_raw_sample(raw_sample_1, transform=transform, target_image_size=(128, 128))
    assert torch.all(processed_1["mask"] == 1.0)
    assert processed_1["label"].item() == 1

    # Label 2: Tampered with mask
    mask_img = Image.new("L", (200, 200), color=0)
    mask_img.paste(255, (50, 50, 150, 150))
    raw_sample_2 = {
        "img_id": "test_tamp_001",
        "image": Image.new("RGB", (200, 200), color=(50, 200, 50)),
        "mask": mask_img,
        "label": 2,
    }
    processed_2 = process_raw_sample(raw_sample_2, transform=transform, target_image_size=(128, 128))
    assert processed_2["label"].item() == 2
    assert processed_2["mask"].sum() > 0
    assert (processed_2["mask"] == 0.0).sum() > 0


def test_resolve_sample_limit():
    from sid_unet.dataset.loader import resolve_sample_limit

    # Positive steps
    assert resolve_sample_limit(samples_val=2000, steps_val=10, batch_size=4) == 40
    # Negative steps -> None (deplete dataset)
    assert resolve_sample_limit(samples_val=2000, steps_val=-1, batch_size=4) is None
    assert resolve_sample_limit(samples_val=2000, steps_val=0, batch_size=4) is None

    # No steps, positive samples
    assert resolve_sample_limit(samples_val=500, steps_val=None, batch_size=4) == 500
    # No steps, negative samples -> None (deplete dataset)
    assert resolve_sample_limit(samples_val=-1, steps_val=None, batch_size=4) is None
    assert resolve_sample_limit(samples_val=0, steps_val=None, batch_size=4) is None

    # Fallback default
    assert resolve_sample_limit(samples_val=None, steps_val=None, batch_size=4, default_samples=100) == 100


def test_two_column_dataset_processing():
    """Test 2-column image/mask datasets such as KhangTruong/IMD2020."""
    transform = get_transforms(image_size=(128, 128), is_train=False)

    # 1. Authentic / Pristine image with black mask (no 'label' or 'img_id' column)
    raw_sample_real = {
        "image": Image.new("RGB", (200, 200), color=(100, 100, 100)),
        "mask": Image.new("L", (200, 200), color=0),
    }
    processed_real = process_raw_sample(raw_sample_real, transform=transform, target_image_size=(128, 128))
    assert processed_real["image"].shape == (3, 128, 128)
    assert processed_real["mask"].shape == (1, 128, 128)
    assert torch.all(processed_real["mask"] == 0.0)
    assert processed_real["label"].item() == 0  # Inferred real (0)

    # 2. Tampered / Inpainted image with region mask
    mask_tampered = Image.new("L", (200, 200), color=0)
    mask_tampered.paste(255, (40, 40, 160, 160))
    raw_sample_tampered = {
        "image": Image.new("RGB", (200, 200), color=(50, 150, 200)),
        "mask": mask_tampered,
    }
    processed_tampered = process_raw_sample(raw_sample_tampered, transform=transform, target_image_size=(128, 128))
    assert processed_tampered["mask"].shape == (1, 128, 128)
    assert processed_tampered["label"].item() == 2  # Inferred partially tampered (2)
    assert processed_tampered["mask"].sum() > 0
    assert (processed_tampered["mask"] == 0.0).sum() > 0

    # 3. Fully synthetic image with all-white mask
    raw_sample_syn = {
        "image": Image.new("RGB", (200, 200), color=(220, 80, 80)),
        "mask": Image.new("L", (200, 200), color=255),
    }
    processed_syn = process_raw_sample(raw_sample_syn, transform=transform, target_image_size=(128, 128))
    assert processed_syn["mask"].shape == (1, 128, 128)
    assert torch.all(processed_syn["mask"] == 1.0)
    assert processed_syn["label"].item() == 1  # Inferred fully synthetic (1)


