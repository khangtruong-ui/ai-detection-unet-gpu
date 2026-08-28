import numpy as np
import pytest
from PIL import Image
import torch
from sid_unet.dataset.transforms import get_transforms, JointCompose, JointResize


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
