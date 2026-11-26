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
import queue
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
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


def _to_binary_u8(mask: np.ndarray) -> np.ndarray:
    """Convert an arbitrary 2D numpy mask array to a contiguous uint8 {0, 1} array."""
    if isinstance(mask, np.ndarray) and mask.dtype == np.uint8:
        # Fast-path when already uint8 {0, 1}; normalize >0 values cleanly if e.g. 255
        if mask.size == 0 or mask.max() <= 1:
            return np.ascontiguousarray(mask)
        return np.ascontiguousarray((mask > 0).astype(np.uint8))
    return np.ascontiguousarray((mask > 0.5).astype(np.uint8))


def _remove_small_components_u8(
    binary: np.ndarray,
    min_area: int = 64,
    connectivity: int = 2,
) -> Tuple[np.ndarray, int]:
    """Internal uint8 implementation of remove_small_components returning (uint8 {0,1}, removed_count)."""
    if min_area <= 1 or not np.any(binary):
        return binary, 0

    if _HAS_CV2:
        cv_conn = 8 if connectivity == 2 else 4
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=cv_conn)
        if num_labels <= 1:
            return binary, 0
        areas = stats[:, cv2.CC_STAT_AREA]
        keep_indices = np.where(areas[1:] >= min_area)[0] + 1
        removed_count = (num_labels - 1) - len(keep_indices)
        if removed_count == 0:
            return binary, 0
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[keep_indices] = 1
        return lut[labels], int(removed_count)

    structure = generate_binary_structure(2, connectivity)
    labeled, num_features = label(binary, structure=structure)
    if num_features == 0:
        return binary, 0

    component_sizes = sum_labels(np.ones_like(binary), labeled, range(1, num_features + 1))
    keep_indices = np.where(component_sizes >= min_area)[0] + 1
    removed_count = num_features - len(keep_indices)
    if removed_count == 0:
        return binary, 0

    return np.isin(labeled, keep_indices).astype(np.uint8), int(removed_count)


def _fill_mask_holes_u8(
    binary: np.ndarray,
    max_hole_size: Optional[int] = 256,
    connectivity: int = 1,
) -> Tuple[np.ndarray, int]:
    """Internal uint8 implementation of fill_mask_holes returning (uint8 {0,1}, pixels_filled)."""
    if not np.any(binary) or np.all(binary):
        return binary, 0

    if _HAS_CV2:
        h, w = binary.shape
        padded = np.pad(binary, 1, mode="constant", constant_values=0)
        ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
        cv2.floodFill(padded, ff_mask, (0, 0), 2)
        all_holes = padded[1:-1, 1:-1] == 0
        if not np.any(all_holes):
            return binary, 0
        if max_hole_size is None:
            filled = binary | all_holes.astype(np.uint8)
            return filled, int(np.sum(all_holes))

        cv_conn = 8 if connectivity == 2 else 4
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(all_holes.astype(np.uint8), connectivity=cv_conn)
        if num_labels <= 1:
            return binary, 0
        areas = stats[:, cv2.CC_STAT_AREA]
        valid_hole_idx = np.where(areas[1:] <= max_hole_size)[0] + 1
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[valid_hole_idx] = 1
        valid_holes = lut[labels]
        return (binary | valid_holes), int(np.sum(valid_holes))

    structure = generate_binary_structure(2, connectivity)
    filled = binary_fill_holes(binary.astype(bool), structure=structure)
    if max_hole_size is None:
        diff_pixels = int(np.sum(filled.astype(np.int32) - binary.astype(np.int32)))
        return filled.astype(np.uint8), diff_pixels

    added_mask = filled & (~binary.astype(bool))
    if not np.any(added_mask):
        return binary, 0

    labeled_holes, num_holes = label(added_mask, structure=structure)
    if num_holes == 0:
        return binary, 0

    hole_sizes = sum_labels(np.ones_like(added_mask, dtype=int), labeled_holes, range(1, num_holes + 1))
    valid_hole_indices = np.where(hole_sizes <= max_hole_size)[0] + 1
    valid_holes = np.isin(labeled_holes, valid_hole_indices)
    return (binary.astype(bool) | valid_holes).astype(np.uint8), int(np.sum(valid_holes))


def _apply_morphology_u8(
    binary: np.ndarray,
    operation: str = "open_close",
    kernel_size: int = 3,
) -> np.ndarray:
    """Internal uint8 implementation of apply_morphology returning uint8 {0, 1}."""
    if operation is None or operation.lower() in ("none", "null", ""):
        return binary
    if not np.any(binary) or np.all(binary):
        return binary

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    op = operation.lower().strip()

    if _HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        b_type = cv2.BORDER_CONSTANT
        if op == "open":
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, borderType=b_type, borderValue=0)
        if op == "close":
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, borderType=b_type, borderValue=0)
        if op in ("open_close", "open-close"):
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, borderType=b_type, borderValue=0)
            return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, borderType=b_type, borderValue=0)
        if op in ("close_open", "close-open"):
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, borderType=b_type, borderValue=0)
            return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, borderType=b_type, borderValue=0)
        logger.warning(f"Unknown morphology operation '{operation}', skipping.")
        return binary

    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    b_bool = binary.astype(bool)
    if op == "open":
        processed = binary_opening(b_bool, structure=structure)
    elif op == "close":
        processed = binary_closing(b_bool, structure=structure)
    elif op in ("open_close", "open-close"):
        processed = binary_closing(binary_opening(b_bool, structure=structure), structure=structure)
    elif op in ("close_open", "close-open"):
        processed = binary_opening(binary_closing(b_bool, structure=structure), structure=structure)
    else:
        logger.warning(f"Unknown morphology operation '{operation}', skipping.")
        processed = b_bool
    return processed.astype(np.uint8)


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
    filtered_u8, removed = _remove_small_components_u8(
        _to_binary_u8(mask), min_area=min_area, connectivity=connectivity
    )
    return filtered_u8.astype(np.float32), removed


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
    filled_u8, diff_pixels = _fill_mask_holes_u8(
        _to_binary_u8(mask), max_hole_size=max_hole_size, connectivity=connectivity
    )
    return filled_u8.astype(np.float32), diff_pixels


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
        return _to_binary_u8(mask).astype(np.float32)
    return _apply_morphology_u8(_to_binary_u8(mask), operation=operation, kernel_size=kernel_size).astype(np.float32)


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

    def process_single_binary_u8(
        self,
        binary_u8: np.ndarray,
        collect_stats: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a pre-binarized 2D ``uint8`` mask in ``{0, 1}`` without float32 conversions."""
        orig_binary = _to_binary_u8(binary_u8)
        while orig_binary.ndim > 2:
            orig_binary = orig_binary[0]

        h, w = orig_binary.shape
        if not self.enabled:
            if not collect_stats:
                return orig_binary, {}
            orig_area = float(np.count_nonzero(orig_binary))
            return orig_binary, {
                "enabled": False,
                "original_area": orig_area,
                "final_area": orig_area,
                "pixels_changed": 0,
                "pixel_change_ratio": 0.0,
                "components_removed": 0,
                "hole_pixels_filled": 0,
            }

        # All morphological steps are identity no-ops on completely empty masks
        if not np.any(orig_binary):
            if not collect_stats:
                return orig_binary, {}
            return orig_binary, {
                "enabled": True,
                "original_area": 0.0,
                "final_area": 0.0,
                "pixels_changed": 0,
                "pixel_change_ratio": 0.0,
                "components_removed": 0,
                "hole_pixels_filled": 0,
            }

        curr = orig_binary
        components_removed = 0
        hole_pixels_filled = 0

        if self.min_area > 1:
            curr, components_removed = _remove_small_components_u8(curr, min_area=self.min_area)

        if self.fill_holes_enabled:
            curr, hole_pixels_filled = _fill_mask_holes_u8(curr, max_hole_size=self.max_hole_size)

        if self.morphology and self.morphology.lower() not in ("none", "null", ""):
            curr = _apply_morphology_u8(curr, operation=self.morphology, kernel_size=self.morph_kernel_size)

        if not collect_stats:
            return curr, {}

        orig_area = float(np.count_nonzero(orig_binary))
        final_area = float(np.count_nonzero(curr))
        pixels_changed = int(np.count_nonzero(curr != orig_binary))
        pixel_change_ratio = float(pixels_changed / max(1, h * w))

        return curr, {
            "enabled": True,
            "original_area": orig_area,
            "final_area": final_area,
            "pixels_changed": pixels_changed,
            "pixel_change_ratio": pixel_change_ratio,
            "components_removed": components_removed,
            "hole_pixels_filled": hole_pixels_filled,
        }

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

        orig_binary = np.ascontiguousarray((arr >= self.threshold).astype(np.uint8))
        out_u8, stats = self.process_single_binary_u8(orig_binary, collect_stats=True)
        return out_u8.astype(np.float32), stats

    def process_batch_binary_np(
        self,
        bin_np: np.ndarray,
        collect_stats: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a batch of pre-binarized uint8 masks ``(B, H, W)`` or ``(B, 1, H, W)``.
        Returns ``(processed_u8 [B, 1, H, W], stats_list)``.
        """
        arr = np.asarray(bin_np, dtype=np.uint8)
        if arr.ndim == 4 and arr.size > 0:
            arr = arr[:, 0]
        elif arr.ndim == 2:
            arr = arr[None, ...]

        b_sz = arr.shape[0]
        if b_sz == 0:
            h = arr.shape[-2] if arr.ndim >= 2 else 0
            w = arr.shape[-1] if arr.ndim >= 1 else 0
            return np.empty((0, 1, h, w), dtype=np.uint8), []

        arr = _to_binary_u8(arr) if arr.max() > 1 else np.ascontiguousarray(arr)
        if not self.enabled:
            out_4d = arr[:, None, :, :]
            if not collect_stats:
                return out_4d, [{} for _ in range(b_sz)]
            stats_list = []
            for i in range(b_sz):
                area = float(np.count_nonzero(arr[i]))
                stats_list.append({
                    "enabled": False,
                    "original_area": area,
                    "final_area": area,
                    "pixels_changed": 0,
                    "pixel_change_ratio": 0.0,
                    "components_removed": 0,
                    "hole_pixels_filled": 0,
                })
            return out_4d, stats_list

        out_batch = arr.copy()
        stats_list: List[Dict[str, Any]] = [{} for _ in range(b_sz)]
        non_empty_indices: List[int] = []
        for i in range(b_sz):
            if np.any(arr[i]):
                non_empty_indices.append(i)
            elif collect_stats:
                stats_list[i] = {
                    "enabled": True,
                    "original_area": 0.0,
                    "final_area": 0.0,
                    "pixels_changed": 0,
                    "pixel_change_ratio": 0.0,
                    "components_removed": 0,
                    "hole_pixels_filled": 0,
                }

        if len(non_empty_indices) > 1:
            executor = self._get_executor(max_workers=min(max(4, os.cpu_count() or 4), len(non_empty_indices)))
            slices = [arr[idx] for idx in non_empty_indices]
            flags = [collect_stats] * len(non_empty_indices)
            for idx, (proc_u8, st) in zip(non_empty_indices, executor.map(self.process_single_binary_u8, slices, flags)):
                out_batch[idx] = proc_u8
                stats_list[idx] = st
        elif len(non_empty_indices) == 1:
            idx = non_empty_indices[0]
            proc_u8, st = self.process_single_binary_u8(arr[idx], collect_stats=collect_stats)
            out_batch[idx] = proc_u8
            stats_list[idx] = st

        return np.ascontiguousarray(out_batch[:, None, :, :]), stats_list

    def process_batch(
        self,
        masks: torch.Tensor,
        is_logit: Optional[bool] = None,
        output_device: Optional[torch.device] = None,
        collect_stats: bool = True,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Process a batch of predicted masks.

        Args:
            masks: PyTorch tensor of shape [B, 1, H, W] or [B, H, W] containing logits or probabilities.
            is_logit: Optional bool. If True, applies sigmoid. If False, treats as probabilities/binary.
                      If None, auto-detects based on whether values fall outside [0, 1].
            output_device: Optional target device for returned processed tensor (defaults to ``masks.device``).
            collect_stats: Whether to compute per-sample pixel change statistics.

        Returns:
            Tuple of (processed_binary_tensor [B, 1, H, W] on target device, list of stats_dicts).
        """
        if masks.ndim == 2:
            masks = masks.unsqueeze(0).unsqueeze(0)
        elif masks.ndim == 3:
            masks = masks.unsqueeze(1)
        elif masks.ndim == 4 and masks.size(1) != 1:
            masks = masks[:, 0:1]

        target_device = output_device if output_device is not None else masks.device
        dtype = masks.dtype
        b_sz = masks.size(0)
        if b_sz == 0:
            return torch.empty((0, 1, masks.size(-2), masks.size(-1)), device=target_device, dtype=dtype), []

        # Binarize directly on masks.device before D2H transfer (4x smaller uint8 copy vs float32)
        if is_logit is True:
            bin_tensor = torch.sigmoid(masks) >= self.threshold
        elif is_logit is False:
            bin_tensor = masks >= self.threshold
        elif masks.min() < 0.0 or masks.max() > 1.0:
            bin_tensor = torch.sigmoid(masks) >= self.threshold
        else:
            bin_tensor = masks >= self.threshold

        if not self.enabled:
            out_t = bin_tensor.to(device=target_device, dtype=dtype)
            if not collect_stats:
                return out_t, [{} for _ in range(b_sz)]
            areas = bin_tensor.reshape(b_sz, -1).sum(dim=-1).detach().cpu().tolist()
            stats_list = [
                {
                    "enabled": False,
                    "original_area": float(area),
                    "final_area": float(area),
                    "pixels_changed": 0,
                    "pixel_change_ratio": 0.0,
                    "components_removed": 0,
                    "hole_pixels_filled": 0,
                }
                for area in areas
            ]
            return out_t, stats_list

        bin_np = bin_tensor[:, 0].to(torch.uint8).detach().cpu().numpy()
        out_u8, stats_list = self.process_batch_binary_np(bin_np, collect_stats=collect_stats)
        processed_batch = torch.from_numpy(out_u8).to(device=target_device, dtype=dtype)
        return processed_batch, stats_list


class AsyncPostProcessPipeline:
    """
    Bounded background pipeline overlapping CPU OpenCV post-processing & post-processed metric updates
    with GPU forward passes in cross-evaluation and standalone evaluation.
    """

    def __init__(
        self,
        postprocessor: MaskPostProcessor,
        post_seg_tracker: Any,
        collect_samples: Optional[List[Dict[str, Any]]] = None,
        max_collect: int = 6,
        max_queue_size: int = 4,
    ):
        self.postprocessor = postprocessor
        self.post_seg_tracker = post_seg_tracker
        self.collect_samples = collect_samples
        self.max_collect = int(max_collect)
        self._submitted_collect = 0
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self._error: Optional[BaseException] = None
        self._finished = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="async_postproc_pipeline",
            daemon=True,
        )
        self._worker.start()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _worker_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                try:
                    bin_u8_cpu, masks_u8_cpu, labels_cpu, datasets, raw_auroc_cpu, sample_payload = item
                    post_u8, _ = self.postprocessor.process_batch_binary_np(bin_u8_cpu, collect_stats=False)

                    self.post_seg_tracker.update(
                        torch.from_numpy(post_u8),
                        torch.from_numpy(masks_u8_cpu),
                        labels_cpu,
                        is_logit=False,
                        datasets=datasets,
                        precomputed_auroc=raw_auroc_cpu,
                    )

                    if sample_payload is not None and self.collect_samples is not None:
                        images_k, gt_k, u_k, labels_k, img_ids_k, start_idx = sample_payload
                        for i in range(images_k.size(0)):
                            p_item = post_u8[i, 0].astype(np.float32)
                            lbl_val = int(labels_k[i].item()) if labels_k is not None else None
                            sample_id = (
                                img_ids_k[i]
                                if (isinstance(img_ids_k, list) and i < len(img_ids_k))
                                else f"Sample {start_idx + i + 1}"
                            )
                            self.collect_samples.append({
                                "image": images_k[i],
                                "gt_mask": gt_k[i],
                                "unet_mask": u_k[i],
                                "post_mask": p_item,
                                "sam_mask": None,
                                "final_mask": p_item,
                                "label": lbl_val,
                                "img_id": sample_id,
                            })
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._error = exc
            # Drain remaining queued items cleanly so main thread put()/join() never deadlocks
            while True:
                try:
                    leftover = self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
                    if leftover is None:
                        break

    def submit(
        self,
        mask_logits: torch.Tensor,
        masks: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        datasets: Optional[Sequence[Any]] = None,
        raw_auroc: Optional[torch.Tensor] = None,
        is_logit: bool = True,
        images: Optional[torch.Tensor] = None,
        img_ids: Optional[Sequence[Any]] = None,
    ) -> None:
        """
        Snapshot minimal uint8 CPU arrays on the caller thread and enqueue post-processing
        and tracker updates for asynchronous CPU execution.
        """
        self._raise_if_failed()
        if self._finished:
            raise RuntimeError("Cannot submit to a finished AsyncPostProcessPipeline.")

        if mask_logits.ndim == 2:
            logits_4d = mask_logits.unsqueeze(0).unsqueeze(0)
        elif mask_logits.ndim == 3:
            logits_4d = mask_logits.unsqueeze(1)
        else:
            logits_4d = mask_logits[:, 0:1]

        b_sz = logits_4d.size(0)
        if b_sz == 0:
            return

        if is_logit:
            bin_gpu = torch.sigmoid(logits_4d) >= self.postprocessor.threshold
        else:
            bin_gpu = logits_4d >= self.postprocessor.threshold

        tracker_thresh = float(getattr(self.post_seg_tracker, "threshold", 0.5))
        masks_4d = masks.unsqueeze(1) if masks.ndim == 3 else masks
        target_bin_gpu = torch.nan_to_num(masks_4d.float(), nan=0.0, posinf=1.0, neginf=0.0) >= tracker_thresh

        bin_u8_cpu = bin_gpu[:, 0].to(torch.uint8).detach().cpu().numpy()
        masks_u8_cpu = target_bin_gpu.to(torch.uint8).detach().cpu().numpy()
        labels_cpu = labels.detach().cpu() if labels is not None else None
        raw_auroc_cpu = raw_auroc.detach().cpu() if raw_auroc is not None else None
        datasets_copy = list(datasets) if datasets is not None else None

        sample_payload = None
        if self.collect_samples is not None and images is not None and self._submitted_collect < self.max_collect:
            k = min(b_sz, self.max_collect - self._submitted_collect)
            raw_probs_k = (torch.sigmoid(logits_4d[:k].float()) if is_logit else logits_4d[:k].float())[:, 0].detach().cpu().numpy()
            gt_k = masks_4d[:k, 0].detach().cpu().float().numpy()
            images_k = images[:k].detach().cpu()
            labels_k = labels_cpu[:k] if labels_cpu is not None else None
            img_ids_k = list(img_ids[:k]) if isinstance(img_ids, (list, tuple)) else None
            sample_payload = (images_k, gt_k, raw_probs_k, labels_k, img_ids_k, self._submitted_collect)
            self._submitted_collect += k

        while True:
            self._raise_if_failed()
            if not self._worker.is_alive():
                raise RuntimeError("AsyncPostProcessPipeline worker stopped unexpectedly.")
            try:
                self._queue.put(
                    (bin_u8_cpu, masks_u8_cpu, labels_cpu, datasets_copy, raw_auroc_cpu, sample_payload),
                    timeout=0.5,
                )
                break
            except queue.Full:
                continue

    def finish(self) -> None:
        """Flush all pending work, join worker thread, and re-raise any worker exception."""
        if not self._finished:
            self._finished = True
            if self._worker.is_alive():
                while True:
                    if not self._worker.is_alive():
                        break
                    try:
                        self._queue.put(None, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            self._worker.join()
        self._raise_if_failed()

    def __enter__(self) -> "AsyncPostProcessPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            if not self._finished:
                self._finished = True
                if self._worker.is_alive():
                    try:
                        self._queue.put(None, timeout=0.5)
                    except queue.Full:
                        pass
                    self._worker.join(timeout=5.0)
            return
        self.finish()


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
