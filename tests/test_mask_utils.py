import numpy as np
import pytest
from PIL import Image
import torch
from sid_unet.dataset.mask_utils import process_sample_mask, ensure_rgb_image


def test_label_0_full_black_mask():
    # Label 0: Real image -> all zeros
    mask = process_sample_mask(mask_input=None, label=0, image_size=(256, 256))
    assert mask.shape == (256, 256)
    assert np.all(mask == 0.0)
    assert mask.dtype == np.float32


def test_label_1_full_white_mask():
    # Label 1: Fully AI generated -> all ones
    mask = process_sample_mask(mask_input=None, label=1, image_size=(256, 256))
    assert mask.shape == (256, 256)
    assert np.all(mask == 1.0)
    assert mask.dtype == np.float32


def test_label_2_pil_mask():
    # Label 2: Inpainting/tampered -> ground truth mask
    raw_mask = Image.new("L", (100, 100), color=0)
    # Paint top half white
    for x in range(100):
        for y in range(50):
            raw_mask.putpixel((x, y), 255)

    processed = process_sample_mask(mask_input=raw_mask, label=2, image_size=(256, 256))
    assert processed.shape == (256, 256)
    # Top half should be 1.0, bottom half 0.0
    assert np.all(processed[:128, :] == 1.0)
    assert np.all(processed[128:, :] == 0.0)


def test_ensure_rgb_image():
    # Grayscale image
    gray_img = Image.new("L", (64, 64), color=128)
    rgb_img = ensure_rgb_image(gray_img)
    assert rgb_img.mode == "RGB"

    # RGBA image
    rgba_img = Image.new("RGBA", (64, 64), color=(100, 150, 200, 255))
    rgb_img = ensure_rgb_image(rgba_img)
    assert rgb_img.mode == "RGB"

    # Numpy array
    np_img = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb_img = ensure_rgb_image(np_img)
    assert rgb_img.mode == "RGB"
