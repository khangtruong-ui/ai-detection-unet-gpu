"""
Data augmentation and preprocessing transforms for image-mask pairs.
Ensures joint transformations for segmentation consistency.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


class JointCompose:
    """Compose joint transformations on PIL Image and Mask numpy array."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class JointResize:
    """Resize image with bilinear interpolation and mask with nearest neighbor."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = (size[0], size[1])  # (H, W)

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        # PIL resize takes (width, height)
        target_w, target_h = self.size[1], self.size[0]
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), resample=Image.BILINEAR)

        if mask.shape != (target_h, target_w):
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((target_w, target_h), resample=Image.NEAREST)
            mask = (np.array(mask_pil, dtype=np.float32) >= 128.0).astype(np.float32)

        return image, mask


class JointRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()
        return image, mask


class JointRandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = np.flipud(mask).copy()
        return image, mask


class JointRandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            rot_map = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}
            image = image.transpose(rot_map[k])
            mask = np.rot90(mask, k=k).copy()
        return image, mask


class JointToTensorAndNormalize:
    """Convert PIL image to normalized FloatTensor and mask to (1, H, W) FloatTensor."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t = TF.to_tensor(image)  # Converts to [0, 1], shape (3, H, W)
        if self.normalize:
            img_t = TF.normalize(img_t, mean=self.mean, std=self.std)

        # Mask: convert 2D (H, W) array to (1, H, W) float tensor
        mask_t = torch.from_numpy(mask).float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        return img_t, mask_t


def get_transforms(
    image_size: Tuple[int, int] = (256, 256),
    is_train: bool = True,
    augment_config: Optional[Dict[str, Any]] = None,
    normalize: bool = True,
) -> JointCompose:
    """Build preprocessing and augmentation pipeline."""
    transforms_list = [JointResize(image_size)]

    if is_train and augment_config:
        hflip = augment_config.get("horizontal_flip", 0.5)
        if hflip > 0:
            transforms_list.append(JointRandomHorizontalFlip(p=hflip))

        vflip = augment_config.get("vertical_flip", 0.0)
        if vflip > 0:
            transforms_list.append(JointRandomVerticalFlip(p=vflip))

        rot90 = augment_config.get("random_rotate90", 0.0)
        if rot90 > 0:
            transforms_list.append(JointRandomRotate90(p=rot90))

    transforms_list.append(JointToTensorAndNormalize(normalize=normalize))
    return JointCompose(transforms_list)
