import io
import numpy as np
import pytest
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader

from sid_unet.dataset.loader import (
    process_raw_sample,
    worker_init_fn,
    SIDStreamingDataset,
)
from sid_unet.dataset.mask_utils import ensure_rgb_image, process_sample_mask
from sid_unet.dataset.transforms import get_transforms


def _generate_truncated_png_bytes(size=(200, 200)) -> bytes:
    """Generate truncated PNG bytes that trigger struct.error/OSError in PIL when LOAD_TRUNCATED_IMAGES=False."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    full_bytes = buf.getvalue()

    # Find a truncation point that causes OSError: image file is truncated
    for cut in range(10, 40):
        trunc = full_bytes[:-cut]
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            t = Image.open(io.BytesIO(trunc))
            t.load()
        except OSError:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            return trunc

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    return full_bytes[:-21]


def test_pil_load_truncated_image_flag():
    """Verify that ImageFile.LOAD_TRUNCATED_IMAGES suppresses OSError on truncated images."""
    trunc_bytes = _generate_truncated_png_bytes()

    # When False, opening and loading truncated image must raise OSError
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with pytest.raises(OSError, match="image file is truncated"):
        img = Image.open(io.BytesIO(trunc_bytes))
        img.load()

    # When True, it succeeds without error
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img_ok = Image.open(io.BytesIO(trunc_bytes))
    img_ok.load()
    assert img_ok.size == (200, 200)


def test_process_raw_sample_with_truncated_image():
    """Verify process_raw_sample correctly processes a truncated image."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    trunc_bytes = _generate_truncated_png_bytes()
    img = Image.open(io.BytesIO(trunc_bytes))
    mask = Image.new("L", (200, 200), color=0)

    sample = {"image": img, "mask": mask, "label": 0, "img_id": "trunc_001"}
    transform = get_transforms(image_size=(128, 128), is_train=False)
    processed = process_raw_sample(sample, transform=transform, target_image_size=(128, 128))

    assert processed["image"].shape == (3, 128, 128)
    assert processed["mask"].shape == (1, 128, 128)
    assert processed["label"].item() == 0
    assert processed["img_id"] == "trunc_001"


def test_process_raw_sample_with_byte_and_dict_inputs():
    """Verify process_raw_sample handles raw bytes and HuggingFace dict format."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    trunc_bytes = _generate_truncated_png_bytes()
    transform = get_transforms(image_size=(64, 64), is_train=False)

    # 1. Image as raw bytes
    sample_bytes = {"image": trunc_bytes, "mask": None, "label": 1}
    processed_bytes = process_raw_sample(sample_bytes, transform=transform, target_image_size=(64, 64))
    assert processed_bytes["image"].shape == (3, 64, 64)
    assert processed_bytes["mask"].shape == (1, 64, 64)
    assert torch.all(processed_bytes["mask"] == 1.0)

    # 2. Image as HuggingFace dict format {'bytes': ...}
    sample_dict = {"image": {"bytes": trunc_bytes, "path": None}, "mask": None, "label": 0}
    processed_dict = process_raw_sample(sample_dict, transform=transform, target_image_size=(64, 64))
    assert processed_dict["image"].shape == (3, 64, 64)
    assert processed_dict["label"].item() == 0


def test_ensure_rgb_image_various_types():
    """Test ensure_rgb_image with PIL Image, ndarray, bytes, BytesIO, and dict."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    arr = np.ones((50, 50, 3), dtype=np.uint8) * 128

    # ndarray
    img_from_arr = ensure_rgb_image(arr)
    assert isinstance(img_from_arr, Image.Image)
    assert img_from_arr.mode == "RGB"

    # PIL Image
    pil_im = Image.new("RGBA", (50, 50), color=(100, 150, 200, 255))
    img_from_pil = ensure_rgb_image(pil_im)
    assert img_from_pil.mode == "RGB"

    # Truncated bytes
    trunc_bytes = _generate_truncated_png_bytes((50, 50))
    img_from_bytes = ensure_rgb_image(trunc_bytes)
    assert img_from_bytes.mode == "RGB"

    # BytesIO
    img_from_bio = ensure_rgb_image(io.BytesIO(trunc_bytes))
    assert img_from_bio.mode == "RGB"

    # Dict
    img_from_dict = ensure_rgb_image({"bytes": trunc_bytes, "path": None})
    assert img_from_dict.mode == "RGB"


def test_process_sample_mask_with_bytes():
    """Test process_sample_mask decoding mask from bytes and dict."""
    mask_im = Image.new("L", (100, 100), color=255)
    buf = io.BytesIO()
    mask_im.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    # From bytes
    mask_arr = process_sample_mask(mask_bytes, label=2, image_size=(100, 100))
    assert mask_arr.shape == (100, 100)
    assert np.all(mask_arr == 1.0)

    # From dict
    mask_arr_dict = process_sample_mask({"bytes": mask_bytes}, label=2, image_size=(100, 100))
    assert mask_arr_dict.shape == (100, 100)
    assert np.all(mask_arr_dict == 1.0)


def test_worker_init_fn():
    """Verify worker_init_fn sets LOAD_TRUNCATED_IMAGES = True."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    worker_init_fn(0)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True


def test_streaming_dataset_handles_truncated_and_corrupt(monkeypatch):
    """
    Verify SIDStreamingDataset yields truncated samples cleanly
    and skips completely corrupt samples without halting.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    trunc_bytes = _generate_truncated_png_bytes((64, 64))

    # Mock generator stream with good, truncated, completely corrupt, and good samples
    samples = [
        {"image": Image.new("RGB", (64, 64), color="blue"), "mask": None, "label": 0},
        {"image": trunc_bytes, "mask": None, "label": 1},
        {"image": b"corrupt_invalid_non_image_garbage_bytes", "mask": None, "label": 0},
        {"image": Image.new("RGB", (64, 64), color="red"), "mask": None, "label": 0},
    ]

    class MockStreamDataset:
        def __iter__(self):
            return iter(samples)

    monkeypatch.setattr(
        "sid_unet.dataset.loader.load_hf_dataset_robust",
        lambda name, requested_split, streaming: (MockStreamDataset(), "train")
    )

    ds = SIDStreamingDataset(
        dataset_name="mock_dataset",
        split="train",
        shuffle_buffer_size=0,
        target_image_size=(32, 32),
    )

    results = list(ds)
    # The corrupt sample should have been skipped, leaving 3 valid results
    assert len(results) == 3
    assert results[0]["image"].shape == (3, 32, 32)
    assert results[1]["image"].shape == (3, 32, 32)
    assert results[2]["image"].shape == (3, 32, 32)


def test_dataloader_batching_with_truncated_images():
    """Verify DataLoader properly batches samples containing truncated images."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    trunc_bytes = _generate_truncated_png_bytes((64, 64))

    class TruncatedMapDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            sample = {"image": trunc_bytes, "mask": None, "label": idx % 3, "img_id": f"trunc_{idx}"}
            transform = get_transforms(image_size=(32, 32), is_train=False)
            return process_raw_sample(sample, transform=transform, target_image_size=(32, 32))

    ds = TruncatedMapDataset()
    loader = DataLoader(ds, batch_size=2, shuffle=False, worker_init_fn=worker_init_fn)

    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["image"].shape == (2, 3, 32, 32)
    assert batches[0]["mask"].shape == (2, 1, 32, 32)
    assert batches[0]["label"].shape == (2,)


def test_trainer_handles_truncated_dataset(tmp_path):
    """Verify Trainer can successfully execute training and validation on datasets with truncated images."""
    from sid_unet.models.unet import UNet
    from sid_unet.training.trainer import Trainer
    from sid_unet.utils.config import load_config

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    trunc_bytes = _generate_truncated_png_bytes((64, 64))

    class TruncatedDataset(torch.utils.data.Dataset):
        def __init__(self, size=4):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            sample = {"image": trunc_bytes, "mask": None, "label": idx % 3, "img_id": f"trunc_{idx}"}
            transform = get_transforms(image_size=(32, 32), is_train=False)
            return process_raw_sample(sample, transform=transform, target_image_size=(32, 32))

    cfg = load_config(overrides=[
        f"project.output_dir={str(tmp_path)}",
        "project.device=cpu",
        "training.epochs=1",
        "training.batch_size=2",
        "model.features=[16, 32]",
        "data.image_size=[32, 32]",
        "logging.log_interval=1",
        "logging.save_sample_images=false",
        "training.amp=false",
    ])

    train_loader = DataLoader(TruncatedDataset(4), batch_size=2, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(TruncatedDataset(2), batch_size=2, worker_init_fn=worker_init_fn)

    trainer = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader)
    results = trainer.train()

    assert "best_score" in results
    assert len(results["history"]) == 1
    assert results["best_epoch"] == 1
