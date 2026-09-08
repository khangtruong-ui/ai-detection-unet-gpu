"""
Mask generation and normalization utilities for SID and general image manipulation datasets.
Handles logic for:
- Standard 2-column datasets (image, mask) such as KhangTruong/IMD2020.
- Explicit label datasets (saberzl/SID_Set): label 0 (full black / 0), label 1 (full white / 1), and label 2 (ground truth mask).
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFile
import torch

# Ensure PIL handles truncated images without raising OSError
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _extract_and_binarize_mask(
    mask_input: Optional[Union[Image.Image, np.ndarray, torch.Tensor, Dict[str, Any], bytes, bytearray, io.BytesIO]],
    image_size: Tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Extract, resize, normalize, and binarize mask_input to shape (h, w)."""
    w, h = image_size if len(image_size) == 2 else (image_size[1], image_size[0])
    if mask_input is None:
        return np.zeros((h, w), dtype=np.float32)

    if isinstance(mask_input, dict):
        if "bytes" in mask_input and mask_input["bytes"] is not None:
            mask_input = Image.open(io.BytesIO(mask_input["bytes"]))
        elif "path" in mask_input and mask_input["path"] is not None:
            mask_input = Image.open(mask_input["path"])
    elif isinstance(mask_input, (bytes, bytearray)):
        mask_input = Image.open(io.BytesIO(mask_input))
    elif isinstance(mask_input, io.BytesIO):
        mask_input = Image.open(mask_input)

    if isinstance(mask_input, Image.Image):
        try:
            mask_input.load()
        except Exception:
            pass
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
        mask_arr = mask_t.detach().cpu().numpy()
        if mask_arr.max() > 1.0:
            mask_arr = mask_arr / 255.0
        if mask_arr.shape != (h, w):
            pil_mask = Image.fromarray((mask_arr * 255).astype(np.uint8))
            pil_mask = pil_mask.resize((w, h), resample=Image.NEAREST)
            mask_arr = np.array(pil_mask, dtype=np.float32) / 255.0
    else:
        return np.zeros((h, w), dtype=np.float32)

    # Binarize with threshold
    return (mask_arr >= threshold).astype(np.float32)


def process_sample_mask(
    mask_input: Optional[Union[Image.Image, np.ndarray, torch.Tensor]],
    label: Optional[int] = None,
    image_size: Tuple[int, int] = (256, 256),  # (width, height) in PIL convention or (H, W)
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Process or synthesize binary mask based on dataset label and provided mask.

    - Label 0 (Real): Full black mask (all 0s).
    - Label 1 (Fully Synthetic): Full white mask (all 1s).
    - Label 2 (Partially Synthetic / Inpainting): Ground truth binary mask from `mask_input`.
    - Label is None (Standard 2-column image + mask dataset, e.g. KhangTruong/IMD2020):
      Directly processes and binarizes `mask_input`.

    Returns:
        np.ndarray: 2D float32 numpy array with values in {0.0, 1.0} and shape (H, W).
    """
    w, h = image_size if len(image_size) == 2 else (image_size[1], image_size[0])

    if label is not None:
        if label == 0:
            # Full black mask (0 = real background)
            return np.zeros((h, w), dtype=np.float32)
        elif label == 1:
            # Full white mask (1 = fully synthetic)
            return np.ones((h, w), dtype=np.float32)
        elif label == 2:
            return _extract_and_binarize_mask(mask_input, image_size=(w, h), threshold=threshold)
        else:
            if mask_input is not None:
                return _extract_and_binarize_mask(mask_input, image_size=(w, h), threshold=threshold)
            return np.zeros((h, w), dtype=np.float32)

    # If label is None (standard 2-column image and mask dataset format, e.g. KhangTruong/IMD2020):
    return _extract_and_binarize_mask(mask_input, image_size=(w, h), threshold=threshold)


def check_image_mask_mismatch(
    image: Optional[Union[Image.Image, np.ndarray, torch.Tensor]],
    mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
    raise_on_mismatch: bool = False,
) -> Dict[str, Any]:
    """
    Check if image and mask pair has any shape, dimension, channel, or missing value mismatches.

    Specifically validates:
    - Missing image or missing mask
    - Spatial resolution mismatch (width, height of image vs mask)
    - Channel mismatch / unexpected dimensions
    - Data type consistency

    Returns:
        Dict with keys:
            - 'mismatch': bool (True if mismatch detected, False otherwise)
            - 'image_size': Optional[Tuple[int, int]] (width, height)
            - 'mask_size': Optional[Tuple[int, int]] (width, height)
            - 'issues': List[str] describing any detected discrepancies.
    """
    issues: List[str] = []
    img_size: Optional[Tuple[int, int]] = None
    mask_size: Optional[Tuple[int, int]] = None

    if image is None:
        issues.append("Image is None or missing")
    else:
        if isinstance(image, Image.Image):
            img_size = (image.width, image.height)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img_size = (image.shape[1], image.shape[0])
            elif image.ndim == 3:
                if image.shape[2] in (1, 3, 4):
                    img_size = (image.shape[1], image.shape[0])
                elif image.shape[0] in (1, 3, 4):
                    img_size = (image.shape[2], image.shape[1])
                else:
                    img_size = (image.shape[1], image.shape[0])
            else:
                issues.append(f"Image has unexpected ndarray dimension {image.ndim}")
        elif isinstance(image, torch.Tensor):
            if image.ndim == 2:
                img_size = (image.shape[1], image.shape[0])
            elif image.ndim == 3:
                img_size = (image.shape[2], image.shape[1])
            elif image.ndim == 4:
                img_size = (image.shape[3], image.shape[2])
            else:
                issues.append(f"Image has unexpected tensor dimension {image.ndim}")
        elif isinstance(image, (bytes, bytearray, io.BytesIO, dict)):
            try:
                temp_img = ensure_rgb_image(image)
                img_size = (temp_img.width, temp_img.height)
            except Exception as e:
                issues.append(f"Failed to decode image from bytes/dict: {e}")
        else:
            issues.append(f"Image has unsupported type: {type(image)}")

    if mask is not None:
        if isinstance(mask, Image.Image):
            mask_size = (mask.width, mask.height)
        elif isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                mask_size = (mask.shape[1], mask.shape[0])
            elif mask.ndim == 3:
                if mask.shape[2] in (1, 3, 4):
                    mask_size = (mask.shape[1], mask.shape[0])
                elif mask.shape[0] in (1, 3, 4):
                    mask_size = (mask.shape[2], mask.shape[1])
                else:
                    mask_size = (mask.shape[1], mask.shape[0])
            else:
                issues.append(f"Mask has unexpected ndarray dimension {mask.ndim}")
        elif isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask_size = (mask.shape[1], mask.shape[0])
            elif mask.ndim == 3:
                mask_size = (mask.shape[2], mask.shape[1])
            elif mask.ndim == 4:
                mask_size = (mask.shape[3], mask.shape[2])
            else:
                issues.append(f"Mask has unexpected tensor dimension {mask.ndim}")
        elif isinstance(mask, (bytes, bytearray, io.BytesIO, dict)):
            try:
                if isinstance(mask, dict):
                    raw_b = mask.get("bytes")
                    p = mask.get("path")
                    t_m = Image.open(io.BytesIO(raw_b)) if raw_b else (Image.open(p) if p else None)
                elif isinstance(mask, (bytes, bytearray)):
                    t_m = Image.open(io.BytesIO(mask))
                elif isinstance(mask, io.BytesIO):
                    t_m = Image.open(mask)
                else:
                    t_m = None
                if t_m is not None:
                    mask_size = (t_m.width, t_m.height)
            except Exception as e:
                issues.append(f"Failed to decode mask from bytes/dict: {e}")
        else:
            issues.append(f"Mask has unsupported type: {type(mask)}")

    if img_size is not None and mask_size is not None:
        if img_size != mask_size:
            issues.append(
                f"Spatial size mismatch: Image is {img_size[0]}x{img_size[1]} (W x H) "
                f"but mask is {mask_size[0]}x{mask_size[1]} (W x H)"
            )

    has_mismatch = len(issues) > 0

    if has_mismatch and raise_on_mismatch:
        raise ValueError(f"Image-mask mismatch detected: {'; '.join(issues)}")

    return {
        "mismatch": has_mismatch,
        "image_size": img_size,
        "mask_size": mask_size,
        "issues": issues,
    }


def ensure_rgb_image(
    image: Union[Image.Image, np.ndarray, Dict[str, Any], bytes, bytearray, io.BytesIO]
) -> Image.Image:
    """
    Ensure input image is a PIL Image in RGB format.
    Supports PIL.Image, numpy arrays, raw bytes, io.BytesIO, and HuggingFace dict format (e.g. {'bytes': ...}).
    """
    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            image = Image.open(io.BytesIO(image["bytes"]))
        elif "path" in image and image["path"] is not None:
            image = Image.open(image["path"])
        else:
            raise ValueError(f"Dictionary sample image has neither 'bytes' nor 'path': {list(image.keys())}")
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, io.BytesIO):
        image = Image.open(image)

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 4:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        # Force load image to verify and handle truncated image streams
        try:
            image.load()
        except Exception:
            pass
        return image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
