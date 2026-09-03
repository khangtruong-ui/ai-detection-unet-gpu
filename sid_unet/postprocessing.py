"""
Post-processing module for SID-UNet generated masks.
Provides algorithms to clean, regularize, and refine raw UNet predicted segmentation masks:
  1. Small Area Filtering (removes isolated false-positive noise blobs below min_area).
  2. Hole Filling (fills small enclosed cavities inside detected tampered regions).
  3. Morphological Operations (opening to eliminate thin protrusions, closing to bridge gaps).
  4. Unified Pipeline & Batch Processing with detailed change tracking statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    generate_binary_structure,
    label,
    sum_labels,
)

logger = logging.getLogger("sid_unet.postprocessing")


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 64,
    connectivity: int = 2,
) -> Tuple[np.ndarray, int]:
    """
    Remove connected positive components in a 2D binary mask smaller than `min_area` pixels.

    Args:
        mask: 2D binary numpy array (0 or 1, or boolean).
        min_area: Minimum number of pixels required to retain a connected component.
        connectivity: Connectivity for neighborhood (1: 4-connected, 2: 8-connected for 2D).

    Returns:
        Tuple of (filtered_binary_mask, number_of_components_removed).
    """
    binary = (mask > 0.5).astype(np.uint8)
    if min_area <= 1 or np.sum(binary) == 0:
        return binary.astype(np.float32), 0

    structure = generate_binary_structure(2, connectivity)
    labeled, num_features = label(binary, structure=structure)
    if num_features == 0:
        return binary.astype(np.float32), 0

    # Calculate area of each component
    component_sizes = sum_labels(np.ones_like(binary), labeled, range(1, num_features + 1))

    # Identify components that meet or exceed min_area
    keep_indices = np.where(component_sizes >= min_area)[0] + 1
    removed_count = num_features - len(keep_indices)

    if removed_count == 0:
        return binary.astype(np.float32), 0

    filtered = np.isin(labeled, keep_indices).astype(np.float32)
    return filtered, int(removed_count)


def fill_mask_holes(
    mask: np.ndarray,
    max_hole_size: Optional[int] = 256,
    connectivity: int = 1,
) -> Tuple[np.ndarray, int]:
    """
    Fill enclosed holes within positive regions of a 2D binary mask.

    Args:
        mask: 2D binary numpy array.
        max_hole_size: Maximum size of holes (in pixels) to fill. If None, fills all enclosed holes.
        connectivity: Connectivity for hole components (1: 4-connected, 2: 8-connected).

    Returns:
        Tuple of (hole_filled_binary_mask, number_of_pixels_filled).
    """
    binary = (mask > 0.5).astype(bool)
    if not np.any(binary) or np.all(binary):
        return binary.astype(np.float32), 0

    structure = generate_binary_structure(2, connectivity)
    filled = binary_fill_holes(binary, structure=structure)

    if max_hole_size is None:
        diff_pixels = int(np.sum(filled.astype(np.int32) - binary.astype(np.int32)))
        return filled.astype(np.float32), diff_pixels

    # If max_hole_size is set, only fill holes smaller than or equal to max_hole_size
    added_mask = filled & (~binary)
    if not np.any(added_mask):
        return binary.astype(np.float32), 0

    labeled_holes, num_holes = label(added_mask, structure=structure)
    if num_holes == 0:
        return binary.astype(np.float32), 0

    hole_sizes = sum_labels(np.ones_like(added_mask, dtype=int), labeled_holes, range(1, num_holes + 1))
    valid_hole_indices = np.where(hole_sizes <= max_hole_size)[0] + 1

    valid_holes = np.isin(labeled_holes, valid_hole_indices)
    result = binary | valid_holes
    diff_pixels = int(np.sum(valid_holes))
    return result.astype(np.float32), diff_pixels


def apply_morphology(
    mask: np.ndarray,
    operation: str = "open_close",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply morphological filtering to smooth boundaries, bridge micro-fractures, and remove thin spurs.

    Args:
        mask: 2D binary numpy array.
        operation: Morphological operation:
            - 'open': opening (erosion then dilation, removes small foreground specks)
            - 'close': closing (dilation then erosion, fills narrow background gaps)
            - 'open_close': opening followed by closing
            - 'close_open': closing followed by opening
            - 'none': no operation
        kernel_size: Size of square structuring element (odd integer, e.g. 3).

    Returns:
        Morphologically filtered binary mask as np.float32.
    """
    if operation is None or operation.lower() in ("none", "null", ""):
        return (mask > 0.5).astype(np.float32)

    binary = (mask > 0.5).astype(bool)
    if not np.any(binary) or np.all(binary):
        return binary.astype(np.float32)

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    op = operation.lower().strip()

    if op == "open":
        processed = binary_opening(binary, structure=structure)
    elif op == "close":
        processed = binary_closing(binary, structure=structure)
    elif op in ("open_close", "open-close"):
        opened = binary_opening(binary, structure=structure)
        processed = binary_closing(opened, structure=structure)
    elif op in ("close_open", "close-open"):
        closed = binary_closing(binary, structure=structure)
        processed = binary_opening(closed, structure=structure)
    else:
        logger.warning(f"Unknown morphology operation '{operation}', skipping.")
        processed = binary

    return processed.astype(np.float32)


class MaskPostProcessor:
    """
    Configurable post-processing pipeline for SID-UNet generated masks.
    Executes small component removal, hole filling, and morphological boundary smoothing.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_area: int = 64,
        fill_holes: bool = True,
        max_hole_size: Optional[int] = 256,
        morphology: Optional[str] = "open_close",
        morph_kernel_size: int = 3,
        threshold: float = 0.5,
    ):
        self.enabled = enabled
        self.min_area = min_area
        self.fill_holes_enabled = fill_holes
        self.max_hole_size = max_hole_size
        self.morphology = morphology
        self.morph_kernel_size = morph_kernel_size
        self.threshold = threshold

    def process_single(
        self,
        mask: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single 2D mask.

        Args:
            mask: 2D numpy array or PyTorch tensor of shape [H, W], [1, H, W],
                  containing probabilities or binary values.

        Returns:
            Tuple of (processed_binary_mask_np [H, W], stats_dict).
        """
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().float().numpy()
        else:
            arr = np.asarray(mask, dtype=np.float32)

        while arr.ndim > 2:
            arr = arr[0]

        h, w = arr.shape
        orig_binary = (arr >= self.threshold).astype(np.float32)
        orig_area = float(np.sum(orig_binary))

        if not self.enabled:
            return orig_binary, {
                "enabled": False,
                "original_area": orig_area,
                "final_area": orig_area,
                "pixels_changed": 0,
                "pixel_change_ratio": 0.0,
                "components_removed": 0,
                "hole_pixels_filled": 0,
            }

        curr = orig_binary
        components_removed = 0
        hole_pixels_filled = 0

        # Step 1: Small area filtering
        if self.min_area > 1:
            curr, components_removed = remove_small_components(curr, min_area=self.min_area)

        # Step 2: Hole filling
        if self.fill_holes_enabled:
            curr, hole_pixels_filled = fill_mask_holes(curr, max_hole_size=self.max_hole_size)

        # Step 3: Morphological smoothing
        if self.morphology and self.morphology.lower() not in ("none", "null", ""):
            curr = apply_morphology(curr, operation=self.morphology, kernel_size=self.morph_kernel_size)

        final_area = float(np.sum(curr))
        diff = np.abs(curr - orig_binary)
        pixels_changed = int(np.sum(diff > 0.5))
        pixel_change_ratio = float(pixels_changed / max(1, h * w))

        stats = {
            "enabled": True,
            "original_area": orig_area,
            "final_area": final_area,
            "pixels_changed": pixels_changed,
            "pixel_change_ratio": pixel_change_ratio,
            "components_removed": components_removed,
            "hole_pixels_filled": hole_pixels_filled,
        }

        return curr, stats

    def process_batch(
        self,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Process a batch of predicted masks.

        Args:
            masks: PyTorch tensor of shape [B, 1, H, W] or [B, H, W] containing logits or probabilities.

        Returns:
            Tuple of (processed_binary_tensor [B, 1, H, W] on same device/dtype, list of stats_dicts).
        """
        device = masks.device
        dtype = masks.dtype
        b_sz = masks.size(0)

        # Sigmoid if values appear to be unnormalized logits
        if masks.min() < 0.0 or masks.max() > 1.0:
            probs = torch.sigmoid(masks)
        else:
            probs = masks

        processed_tensors = []
        stats_list = []

        for i in range(b_sz):
            sample_m = probs[i]
            proc_np, stats = self.process_single(sample_m)
            stats_list.append(stats)

            proc_t = torch.from_numpy(proc_np).to(device=device, dtype=dtype)
            if proc_t.ndim == 2:
                proc_t = proc_t.unsqueeze(0)
            elif proc_t.ndim == 3 and proc_t.size(0) != 1:
                proc_t = proc_t[0:1]
            processed_tensors.append(proc_t)

        processed_batch = torch.stack(processed_tensors, dim=0)
        return processed_batch, stats_list


def get_postprocessor_from_config(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    threshold: float = 0.5,
    override_enabled: Optional[bool] = None,
) -> MaskPostProcessor:
    """
    Factory helper to instantiate a MaskPostProcessor from configuration dictionary or ConfigDict.

    Args:
        config: Configuration dictionary (can contain 'post_processing' section).
        threshold: Default binarization threshold.
        override_enabled: Explicit boolean override for enabling/disabling post-processing.
    """
    post_cfg: Dict[str, Any] = {}
    if config is not None:
        if hasattr(config, "get"):
            post_cfg = config.get("post_processing", {}) or {}
        elif isinstance(config, dict):
            post_cfg = config.get("post_processing", {}) or {}

    enabled = post_cfg.get("enabled", True)
    if override_enabled is not None:
        enabled = override_enabled

    min_area = int(post_cfg.get("min_area", 64))
    fill_holes = bool(post_cfg.get("fill_holes", True))
    max_hole_size = post_cfg.get("max_hole_size", 256)
    if max_hole_size is not None:
        max_hole_size = int(max_hole_size)

    morphology = post_cfg.get("morphology", "open_close")
    morph_kernel_size = int(post_cfg.get("morph_kernel_size", 3))

    return MaskPostProcessor(
        enabled=enabled,
        min_area=min_area,
        fill_holes=fill_holes,
        max_hole_size=max_hole_size,
        morphology=morphology,
        morph_kernel_size=morph_kernel_size,
        threshold=threshold,
    )
