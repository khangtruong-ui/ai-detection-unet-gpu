"""
Data augmentation and preprocessing transforms for image-mask pairs.
Ensures joint transformations for segmentation consistency.
"""

from __future__ import annotations

import io
import random
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFilter
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


class JointRandomJPEGCompression:
    """Apply random JPEG recompression at QF in [min_qf, max_qf] without altering mask."""

    def __init__(self, p: float = 0.5, quality_range: Tuple[int, int] = (65, 100)):
        self.p = float(p)
        q_min, q_max = int(quality_range[0]), int(quality_range[1])
        self.quality_range = (max(1, min(q_min, q_max)), min(100, max(q_min, q_max)))

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < self.p:
            qf = random.randint(self.quality_range[0], self.quality_range[1])
            if qf < 100:
                rgb_img = image.convert("RGB") if image.mode != "RGB" else image
                buf = io.BytesIO()
                rgb_img.save(buf, format="JPEG", quality=qf)
                buf.seek(0)
                with Image.open(buf) as decoded:
                    image = decoded.convert("RGB")
                buf.close()
        return image, mask


class JointMaskZoomCrop:
    """
    Mask-aware random zoom/crop augmentation exposing training to varied tampered area ratios
    (including >50% interior crops and shifted border configurations).
    """

    def __init__(
        self,
        p: float = 0.0,
        zoom_prob: float = 0.65,
        min_crop_scale: float = 0.35,
        max_crop_scale: float = 0.85,
    ):
        self.p = float(p)
        self.zoom_prob = float(zoom_prob)
        self.min_crop_scale = max(0.15, min(0.95, float(min_crop_scale)))
        self.max_crop_scale = max(self.min_crop_scale, min(1.0, float(max_crop_scale)))

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() >= self.p:
            return image, mask
        ys, xs = np.where(mask >= 0.5)
        if len(ys) == 0:
            return image, mask

        img_w, img_h = image.size
        mask_h, mask_w = mask.shape[:2]
        if img_w < 16 or img_h < 16 or mask_h != img_h or mask_w != img_w:
            return image, mask

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        box_h, box_w = y1 - y0, x1 - x0
        cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)

        scale = random.uniform(self.min_crop_scale, self.max_crop_scale)
        if random.random() < self.zoom_prob:
            # Tighter crop around tampered region to expose >40% area ratios
            base_side = max(box_h, box_w, int(round(min(img_h, img_w) * self.min_crop_scale)))
            crop_side = int(round(min(min(img_h, img_w), max(base_side * 1.15, min(img_h, img_w) * scale))))
        else:
            crop_side = int(round(min(img_h, img_w) * scale))
        crop_side = max(32, min(img_h, img_w, crop_side))

        # Randomize center offset while keeping tampered centroid inside crop
        half = crop_side / 2.0
        jitter = crop_side * 0.25
        c_y = cy + random.uniform(-jitter, jitter)
        c_x = cx + random.uniform(-jitter, jitter)
        top = int(round(max(0.0, min(float(img_h - crop_side), c_y - half))))
        left = int(round(max(0.0, min(float(img_w - crop_side), c_x - half))))
        bottom = top + crop_side
        right = left + crop_side

        cropped_mask = mask[top:bottom, left:right]
        if cropped_mask.max() < 0.5:
            return image, mask

        cropped_img = image.crop((left, top, right, bottom))
        return cropped_img, cropped_mask.copy()


class JointSoftBoundaryFeather:
    """
    Subtle boundary-band Gaussian smoothing/feathering along tampered edges to simulate
    low-separability seamless manipulations without changing ground-truth mask alignment.
    """

    def __init__(
        self,
        p: float = 0.0,
        radius_range: Tuple[float, float] = (1.5, 4.0),
        blend_alpha_range: Tuple[float, float] = (0.25, 0.55),
    ):
        self.p = float(p)
        self.radius_range = (float(radius_range[0]), float(radius_range[1]))
        self.blend_alpha_range = (float(blend_alpha_range[0]), float(blend_alpha_range[1]))

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() >= self.p or mask.max() < 0.5:
            return image, mask

        rgb_img = image.convert("RGB") if image.mode != "RGB" else image
        radius = random.uniform(self.radius_range[0], self.radius_range[1])
        alpha = random.uniform(self.blend_alpha_range[0], self.blend_alpha_range[1])

        blurred = rgb_img.filter(ImageFilter.GaussianBlur(radius=radius))
        arr_orig = np.asarray(rgb_img, dtype=np.float32)
        arr_blur = np.asarray(blurred, dtype=np.float32)

        # Build soft spatial weight concentrated around tampered mask boundary
        m_u8 = (mask >= 0.5).astype(np.uint8) * 255
        m_pil = Image.fromarray(m_u8, mode="L")
        band = np.asarray(m_pil.filter(ImageFilter.GaussianBlur(radius=max(2.0, radius * 1.5))), dtype=np.float32) / 255.0
        edge_band = np.clip(4.0 * band * (1.0 - band), 0.0, 1.0)[..., None]

        # Also slightly blend interior contrast toward local blurred image
        blend_w = np.clip(alpha * (0.35 * (mask[..., None] >= 0.5).astype(np.float32) + 0.65 * edge_band), 0.0, 0.85)
        mixed = np.clip(arr_orig * (1.0 - blend_w) + arr_blur * blend_w, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(mixed, mode="RGB"), mask


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
    pre_resize_transforms = []
    if is_train and augment_config:
        zoom_p = float(augment_config.get("mask_zoom_crop_prob", 0.0))
        if zoom_p > 0:
            pre_resize_transforms.append(
                JointMaskZoomCrop(
                    p=zoom_p,
                    zoom_prob=float(augment_config.get("mask_zoom_focus_prob", 0.65)),
                    min_crop_scale=float(augment_config.get("mask_zoom_min_scale", 0.35)),
                    max_crop_scale=float(augment_config.get("mask_zoom_max_scale", 0.85)),
                )
            )

    transforms_list = pre_resize_transforms + [JointResize(image_size)]

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

        feather_p = float(augment_config.get("soft_boundary_feather_prob", 0.0))
        if feather_p > 0:
            transforms_list.append(
                JointSoftBoundaryFeather(
                    p=feather_p,
                    radius_range=tuple(augment_config.get("soft_boundary_radius", (1.5, 4.0))),
                    blend_alpha_range=tuple(augment_config.get("soft_boundary_alpha", (0.25, 0.55))),
                )
            )

        jpeg_p = float(augment_config.get("random_jpeg", augment_config.get("jpeg_compression", 0.0)))
        if jpeg_p > 0:
            jpeg_q = augment_config.get("jpeg_quality", None)
            if isinstance(jpeg_q, (list, tuple)) and len(jpeg_q) >= 2:
                qf_min, qf_max = int(jpeg_q[0]), int(jpeg_q[1])
            else:
                qf_min = int(augment_config.get("jpeg_qf_min", 65))
                qf_max = int(augment_config.get("jpeg_qf_max", 100))
            transforms_list.append(JointRandomJPEGCompression(p=jpeg_p, quality_range=(qf_min, qf_max)))

    transforms_list.append(JointToTensorAndNormalize(normalize=normalize))
    return JointCompose(transforms_list)
