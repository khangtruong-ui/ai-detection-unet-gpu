"""
Mask generation and normalization utilities for SID dataset.
Handles logic for label 0 (full black / 0), label 1 (full white / 1), and label 2 (ground truth mask).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch


def process_sample_mask(
    mask_input: Optional[Union[Image.Image, np.ndarray, torch.Tensor]],
    label: int,
    image_size: Tuple[int, int],  # (width, height) in PIL convention or (H, W)
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Process or synthesize binary mask based on dataset label and provided mask.

    - Label 0 (Real): Full black mask (all 0s).
    - Label 1 (Fully Synthetic): Full white mask (all 1s).
    - Label 2 (Partially Synthetic / Inpainting): Ground truth binary mask from `mask_input`.

    Returns:
        np.ndarray: 2D float32 numpy array with values in {0.0, 1.0} and shape (H, W).
    """
    w, h = image_size if len(image_size) == 2 else (image_size[1], image_size[0])

    if label == 0:
        # Full black mask (0 = real background)
        return np.zeros((h, w), dtype=np.float32)

    if label == 1:
        # Full white mask (1 = fully synthetic)
        return np.ones((h, w), dtype=np.float32)

    if label == 2:
        if mask_input is None:
            # Fallback if somehow None, default to zeros
            return np.zeros((h, w), dtype=np.float32)

        if isinstance(mask_input, Image.Image):
            # Ensure single channel grayscale
            mask_gray = mask_input.convert("L")
            if mask_gray.size != (w, h):
                mask_gray = mask_gray.resize((w, h), resample=Image.NEAREST)
            mask_arr = np.array(mask_gray, dtype=np.float32) / 255.0
        elif isinstance(mask_input, np.ndarray):
            mask_arr = mask_input.astype(np.float32)
            if mask_arr.ndim == 3:
                # If RGB/RGBA, take first channel or mean across channels
                mask_arr = mask_arr.mean(axis=2) if mask_arr.shape[2] in (3, 4) else mask_arr[0]
            if mask_arr.max() > 1.0:
                mask_arr = mask_arr / 255.0
            if mask_arr.shape != (h, w):
                pil_mask = Image.fromarray((mask_arr * 255).astype(np.uint8))
                pil_mask = pil_mask.resize((w, h), resample=Image.NEAREST)
                mask_arr = np.array(pil_mask, dtype=np.float32) / 255.0
        elif isinstance(mask_input, torch.Tensor):
            mask_t = mask_input.float()
            if mask_t.ndim == 3 and mask_t.size(0) in (1, 3, 4):
                mask_t = mask_t.mean(dim=0)
            mask_arr = mask_t.cpu().numpy()
            if mask_arr.max() > 1.0:
                mask_arr = mask_arr / 255.0
        else:
            return np.zeros((h, w), dtype=np.float32)

        # Binarize with threshold
        return (mask_arr >= threshold).astype(np.float32)

    # For unknown label, default to zeros
    return np.zeros((h, w), dtype=np.float32)


def ensure_rgb_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Ensure input image is a PIL Image in RGB format."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 4:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
