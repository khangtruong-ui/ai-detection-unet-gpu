"""
Post-processing module for SID-UNet generated masks.
Provides algorithms to clean, regularize, and refine raw UNet predicted segmentation masks:
  1. Small Area Filtering (removes isolated false-positive noise blobs below min_area).
  2. Hole Filling (fills small enclosed cavities inside detected tampered regions).
  3. Morphological Operations (opening to eliminate thin protrusions, closing to bridge gaps).
  4. Unified Pipeline & Batch Processing with detailed change tracking statistics.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
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

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

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
    if min_area <= 1 or not np.any(binary):
        return binary.astype(np.float32), 0

    if _HAS_CV2:
        cv_conn = 8 if connectivity == 2 else 4
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=cv_conn)
        if num_labels <= 1:
            return binary.astype(np.float32), 0
        areas = stats[:, cv2.CC_STAT_AREA]
        keep_indices = np.where(areas[1:] >= min_area)[0] + 1
        removed_count = (num_labels - 1) - len(keep_indices)
        if removed_count == 0:
            return binary.astype(np.float32), 0
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[keep_indices] = 1
        filtered = lut[labels].astype(np.float32)
        return filtered, int(removed_count)

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
    binary = (mask > 0.5).astype(np.uint8)
    if not np.any(binary) or np.all(binary):
        return binary.astype(np.float32), 0

    if _HAS_CV2:
        h, w = binary.shape
        padded = np.pad(binary, 1, mode="constant", constant_values=0)
        ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
        cv2.floodFill(padded, ff_mask, (0, 0), 2)
        all_holes = (padded[1:-1, 1:-1] == 0)
        if not np.any(all_holes):
            return binary.astype(np.float32), 0
        if max_hole_size is None:
            filled = binary | all_holes.astype(np.uint8)
            diff_pixels = int(np.sum(all_holes))
            return filled.astype(np.float32), diff_pixels

        cv_conn = 8 if connectivity == 2 else 4
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(all_holes.astype(np.uint8), connectivity=cv_conn)
        if num_labels <= 1:
            return binary.astype(np.float32), 0
        areas = stats[:, cv2.CC_STAT_AREA]
        valid_hole_idx = np.where(areas[1:] <= max_hole_size)[0] + 1
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[valid_hole_idx] = 1
        valid_holes = lut[labels]
        result = binary | valid_holes
        diff_pixels = int(np.sum(valid_holes))
        return result.astype(np.float32), diff_pixels

    structure = generate_binary_structure(2, connectivity)
    filled = binary_fill_holes(binary.astype(bool), structure=structure)

    if max_hole_size is None:
        diff_pixels = int(np.sum(filled.astype(np.int32) - binary.astype(np.int32)))
        return filled.astype(np.float32), diff_pixels

    # If max_hole_size is set, only fill holes smaller than or equal to max_hole_size
    added_mask = filled & (~binary.astype(bool))
    if not np.any(added_mask):
        return binary.astype(np.float32), 0

    labeled_holes, num_holes = label(added_mask, structure=structure)
    if num_holes == 0:
        return binary.astype(np.float32), 0

    hole_sizes = sum_labels(np.ones_like(added_mask, dtype=int), labeled_holes, range(1, num_holes + 1))
    valid_hole_indices = np.where(hole_sizes <= max_hole_size)[0] + 1

    valid_holes = np.isin(labeled_holes, valid_hole_indices)
    result = binary.astype(bool) | valid_holes
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

    binary = (mask > 0.5).astype(np.uint8)
    if not np.any(binary) or np.all(binary):
        return binary.astype(np.float32)

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    op = operation.lower().strip()

    if _HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        b_type = cv2.BORDER_CONSTANT
        if op == "open":
            res = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, borderType=b_type, borderValue=0)
        elif op == "close":
            res = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, borderType=b_type, borderValue=0)
        elif op in ("open_close", "open-close"):
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, borderType=b_type, borderValue=0)
            res = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, borderType=b_type, borderValue=0)
        elif op in ("close_open", "close-open"):
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, borderType=b_type, borderValue=0)
            res = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, borderType=b_type, borderValue=0)
        else:
            logger.warning(f"Unknown morphology operation '{operation}', skipping.")
            res = binary
        return res.astype(np.float32)

    structure = np.ones((kernel_size, kernel_size), dtype=bool)

    if op == "open":
        processed = binary_opening(binary.astype(bool), structure=structure)
    elif op == "close":
        processed = binary_closing(binary.astype(bool), structure=structure)
    elif op in ("open_close", "open-close"):
        opened = binary_opening(binary.astype(bool), structure=structure)
        processed = binary_closing(opened, structure=structure)
    elif op in ("close_open", "close-open"):
        closed = binary_closing(binary.astype(bool), structure=structure)
        processed = binary_opening(closed, structure=structure)
    else:
        logger.warning(f"Unknown morphology operation '{operation}', skipping.")
        processed = binary.astype(bool)

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
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _get_executor(self, max_workers: int = 4) -> concurrent.futures.ThreadPoolExecutor:
        """Lazily initialize and reuse thread pool for parallel mask post-processing across batches."""
        if self._executor is None:
            workers = min(max(1, max_workers), os.cpu_count() or 4)
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="mask_postproc",
            )
        return self._executor

    def close(self) -> None:
        """Shutdown background worker pool if allocated."""
        if getattr(self, "_executor", None) is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
            self._executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
        is_logit: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Process a batch of predicted masks.

        Args:
            masks: PyTorch tensor of shape [B, 1, H, W] or [B, H, W] containing logits or probabilities.
            is_logit: Optional bool. If True, applies sigmoid. If False, treats as probabilities/binary.
                      If None, auto-detects based on whether values fall outside [0, 1].

        Returns:
            Tuple of (processed_binary_tensor [B, 1, H, W] on same device/dtype, list of stats_dicts).
        """
        device = masks.device
        dtype = masks.dtype
        b_sz = masks.size(0)

        # Sigmoid if values appear to be unnormalized logits
        if is_logit is True:
            probs = torch.sigmoid(masks)
        elif is_logit is False:
            probs = masks
        elif masks.min() < 0.0 or masks.max() > 1.0:
            probs = torch.sigmoid(masks)
        else:
            probs = masks

        if not self.enabled:
            # Fast tensor binarization directly on device
            bin_tensor = (probs >= self.threshold).to(dtype=dtype)
            if bin_tensor.ndim == 3:
                bin_tensor = bin_tensor.unsqueeze(1)
            stats_list = [
                {
                    "enabled": False,
                    "original_area": float(bin_tensor[i].sum().item()),
                    "final_area": float(bin_tensor[i].sum().item()),
                    "pixels_changed": 0,
                    "pixel_change_ratio": 0.0,
                    "components_removed": 0,
                    "hole_pixels_filled": 0,
                }
                for i in range(b_sz)
            ]
            return bin_tensor, stats_list

        processed_tensors = []
        stats_list = []

        if b_sz > 1:
            # Multi-threaded execution across CPU cores for morphological operations with pooled executor
            prob_slices = [probs[i].detach().cpu().float().numpy() for i in range(b_sz)]
            executor = self._get_executor(max_workers=min(max(4, os.cpu_count() or 4), b_sz))
            results = list(executor.map(self.process_single, prob_slices))
            for proc_np, stats in results:
                stats_list.append(stats)
                proc_t = torch.from_numpy(proc_np).to(device=device, dtype=dtype)
                if proc_t.ndim == 2:
                    proc_t = proc_t.unsqueeze(0)
                elif proc_t.ndim == 3 and proc_t.size(0) != 1:
                    proc_t = proc_t[0:1]
                processed_tensors.append(proc_t)
        else:
            sample_m = probs[0]
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
