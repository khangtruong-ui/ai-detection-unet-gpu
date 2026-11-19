import numpy as np
import pytest
from PIL import Image
import torch
from sid_unet.dataset.transforms import (
    get_transforms,
    JointCompose,
    JointRandomJPEGCompression,
    JointResize,
)


def test_joint_resize_and_tensor_shapes():
    tf = get_transforms(image_size=(256, 256), is_train=False)
    img = Image.new("RGB", (512, 384), color=(100, 150, 200))
    mask = np.zeros((384, 512), dtype=np.float32)
    mask[:100, :100] = 1.0

    img_t, mask_t = tf(img, mask)

    assert isinstance(img_t, torch.Tensor)
    assert isinstance(mask_t, torch.Tensor)
    assert img_t.shape == (3, 256, 256)
    assert mask_t.shape == (1, 256, 256)
    # Mask should remain binary
    unique_vals = torch.unique(mask_t).tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0)


def test_joint_augmentations():
    tf = get_transforms(
        image_size=(128, 128),
        is_train=True,
        augment_config={"horizontal_flip": 1.0, "vertical_flip": 1.0, "random_rotate90": 1.0},
    )
    img = Image.new("RGB", (100, 100), color=(50, 50, 50))
    mask = np.zeros((100, 100), dtype=np.float32)

    img_t, mask_t = tf(img, mask)
    assert img_t.shape == (3, 128, 128)
    assert mask_t.shape == (1, 128, 128)


def test_joint_random_jpeg_compression():
    rng = np.random.default_rng(123)
    arr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[20:60, 20:60] = 1.0
    orig_mask = mask.copy()

    jpeg_tf = JointRandomJPEGCompression(p=1.0, quality_range=(65, 85))
    img_out, mask_out = jpeg_tf(img, mask)
    assert img_out.size == (128, 128)
    assert img_out.mode == "RGB"
    assert np.array_equal(mask_out, orig_mask)
    assert not np.array_equal(np.asarray(img_out), arr)

    tf = get_transforms(
        image_size=(128, 128),
        is_train=True,
        augment_config={
            "horizontal_flip": 0.5,
            "vertical_flip": 0.2,
            "random_jpeg": 1.0,
            "jpeg_qf_min": 65,
            "jpeg_qf_max": 100,
        },
        normalize=False,
    )
    img_t, mask_t = tf(img, mask)
    assert img_t.shape == (3, 128, 128)
    assert mask_t.shape == (1, 128, 128)
