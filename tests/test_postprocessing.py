import numpy as np
import pytest
import torch

from sid_unet.postprocessing import (
    MaskPostProcessor,
    apply_morphology,
    fill_mask_holes,
    get_postprocessor_from_config,
    remove_small_components,
)


def test_remove_small_components():
    # 64x64 mask with one large component (10x10 = 100 px) and one small speckle (2x2 = 4 px)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[5:15, 5:15] = 1.0  # 100 pixels
    mask[40:42, 40:42] = 1.0  # 4 pixels

    filtered, num_removed = remove_small_components(mask, min_area=20)
    assert num_removed == 1
    assert np.sum(filtered[40:42, 40:42]) == 0.0
    assert np.sum(filtered[5:15, 5:15]) == 100.0


def test_fill_mask_holes():
    # 64x64 mask with a square containing a 2x2 hole inside
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    mask[18:20, 18:20] = 0.0  # 4-pixel hole

    filled, filled_pixels = fill_mask_holes(mask, max_hole_size=50)
    assert filled_pixels == 4
    assert np.all(filled[18:20, 18:20] == 1.0)
    assert np.sum(filled) == 400.0


def test_apply_morphology():
    # Test open, close, open_close
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[5:15, 5:15] = 1.0
    mask[25, 25] = 1.0  # single pixel speckle

    opened = apply_morphology(mask, operation="open", kernel_size=3)
    assert opened[25, 25] == 0.0
    assert opened[10, 10] == 1.0

    closed = apply_morphology(mask, operation="close", kernel_size=3)
    assert closed.shape == (32, 32)

    open_close = apply_morphology(mask, operation="open_close", kernel_size=3)
    assert open_close.shape == (32, 32)

    none_op = apply_morphology(mask, operation="none")
    assert np.array_equal(none_op, mask)


def test_postprocessor_process_single():
    proc = MaskPostProcessor(enabled=True, min_area=16, fill_holes=True, morphology="open_close")

    # Empty mask
    empty = np.zeros((32, 32), dtype=np.float32)
    res, stats = proc.process_single(empty)
    assert np.sum(res) == 0.0
    assert stats["pixels_changed"] == 0

    # Mask with small noise
    noisy = np.zeros((32, 32), dtype=np.float32)
    noisy[2:10, 2:10] = 1.0  # 64 px
    noisy[20, 20] = 1.0  # 1 px noise
    res, stats = proc.process_single(noisy)
    assert stats["enabled"] is True
    assert stats["pixels_changed"] > 0
    assert res[20, 20] == 0.0


def test_postprocessor_process_batch():
    proc = MaskPostProcessor(enabled=True, min_area=10, fill_holes=True)

    # Batch of shape [2, 1, 32, 32] with negative logits for background
    batch = torch.full((2, 1, 32, 32), -5.0)
    batch[0, 0, 5:15, 5:15] = 5.0  # Positive logits (100 px)
    batch[1, 0, 20, 20] = 5.0      # Isolated positive logit (1 px)

    out, stats = proc.process_batch(batch)
    assert out.shape == (2, 1, 32, 32)
    assert len(stats) == 2
    # Sample 0 has large component preserved
    assert torch.sum(out[0]) > 0
    # Sample 1 had only 1 pixel, removed by min_area=10
    assert torch.sum(out[1]) == 0


def test_get_postprocessor_from_config():
    config = {
        "post_processing": {
            "enabled": True,
            "min_area": 128,
            "fill_holes": True,
            "max_hole_size": 512,
            "morphology": "close",
            "morph_kernel_size": 5,
        }
    }
    proc = get_postprocessor_from_config(config)
    assert proc.enabled is True
    assert proc.min_area == 128
    assert proc.morphology == "close"
    assert proc.morph_kernel_size == 5

    # Override enabled
    proc_disabled = get_postprocessor_from_config(config, override_enabled=False)
    assert proc_disabled.enabled is False
