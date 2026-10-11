"""
Memory management and Out-of-Memory (OOM) handling utilities for SID-UNet.
Provides OOM error detection, GPU memory tracking, automatic memory cache clearing,
batch splitting for micro-batching, and automatic batch size finding.
"""

from __future__ import annotations

import copy
import gc
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def is_oom_error(exc: BaseException) -> bool:
    """
    Check whether an exception was caused by CUDA or host memory exhaustion (Out of Memory).
    Handles torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, MemoryError, and
    RuntimeError with CUDA OOM message patterns.
    """
    if isinstance(exc, (MemoryError, getattr(torch, "OutOfMemoryError", ()))) or (
        hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError)
    ):
        return True

    exc_msg = str(exc).lower()
    oom_patterns = [
        "out of memory",
        "cuda out of memory",
        "cuda error: out of memory",
        "allocate memory",
        "all_memory",
        "allocated memory",
        "trying to allocate",
        "cudamalloc",
        "hip out of memory",
    ]
    return any(pattern in exc_msg for pattern in oom_patterns)


def clear_memory_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Clear Python garbage collection and release cached allocator memory in PyTorch CUDA/accelerators.
    Safe to call on any device (CPU or GPU).
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass


def get_memory_summary(device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """
    Get detailed memory metrics for the specified device (or current CUDA device / host RAM if on CPU).
    Returns dictionary with allocated, reserved, peak allocated, total memory, used memory, and
    memory usage and peak fractions (0.0 to 1.0).
    """
    summary = {
        "device_type": "cpu",
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
        "total_mb": 0.0,
        "free_mb": 0.0,
        "used_mb": 0.0,
        "usage_fraction": 0.0,
        "peak_fraction": 0.0,
    }

    dev = device if isinstance(device, torch.device) else torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
            max_allocated = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2)
            max_reserved = torch.cuda.max_memory_reserved(dev_idx) / (1024 ** 2)
            props = torch.cuda.get_device_properties(dev_idx)
            total = props.total_memory / (1024 ** 2)

            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
                free = free_bytes / (1024 ** 2)
                total = total_bytes / (1024 ** 2)
            except Exception:
                free = max(0.0, total - reserved)

            used = max(0.0, total - free)
            usage_fraction = min(1.0, max(0.0, used / total)) if total > 0 else 0.0
            peak_used = max(used, max_reserved, max_allocated)
            peak_fraction = min(1.0, max(0.0, peak_used / total)) if total > 0 else 0.0

            summary.update({
                "device_type": "cuda",
                "device_index": dev_idx,
                "device_name": props.name,
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "max_allocated_mb": round(max_allocated, 2),
                "max_reserved_mb": round(max_reserved, 2),
                "total_mb": round(total, 2),
                "free_mb": round(free, 2),
                "used_mb": round(used, 2),
                "usage_fraction": round(usage_fraction, 4),
                "peak_fraction": round(peak_fraction, 4),
            })
            return summary
        except Exception:
            pass

    # CPU / System RAM tracking
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_mb = vm.total / (1024 ** 2)
        free_mb = vm.available / (1024 ** 2)
        used_mb = vm.used / (1024 ** 2)
        usage_frac = vm.percent / 100.0
        summary.update({
            "device_type": "cpu",
            "device_name": "System RAM",
            "total_mb": round(total_mb, 2),
            "free_mb": round(free_mb, 2),
            "used_mb": round(used_mb, 2),
            "allocated_mb": round(used_mb, 2),
            "reserved_mb": round(used_mb, 2),
            "max_allocated_mb": round(used_mb, 2),
            "max_reserved_mb": round(used_mb, 2),
            "usage_fraction": round(usage_frac, 4),
            "peak_fraction": round(usage_frac, 4),
        })
    except Exception:
        pass

    return summary


def format_memory_summary(device: Optional[Union[str, torch.device]] = None) -> str:
    """
    Return a formatted human-readable summary of current GPU or system memory usage.
    """
    stats = get_memory_summary(device)
    if stats["device_type"] != "cuda":
        total_mb = stats.get("total_mb", 0.0)
        used_mb = stats.get("used_mb", 0.0)
        frac = stats.get("usage_fraction", 0.0)
        return f"RAM: {used_mb:.1f}/{total_mb:.0f} MB used ({frac * 100:.1f}%)"
    used_mb = stats.get("used_mb", stats["total_mb"] - stats["free_mb"])
    usage_pct = stats.get("usage_fraction", used_mb / max(1.0, stats["total_mb"])) * 100.0
    peak_pct = stats.get("peak_fraction", 0.0) * 100.0
    return (
        f"VRAM: {used_mb:.1f}/{stats['total_mb']:.0f} MB used ({usage_pct:.1f}%) | "
        f"Peak: {stats['max_reserved_mb']:.1f} MB ({peak_pct:.1f}%) | "
        f"PyTorch: {stats['allocated_mb']:.1f} MB allocated, {stats['reserved_mb']:.1f} MB reserved"
    )



def split_batch(batch: Dict[str, Any], micro_batch_size: int) -> List[Dict[str, Any]]:
    """
    Split a batch dictionary of tensors/lists into smaller micro-batches of size `micro_batch_size`.
    Used for gradient accumulation and sub-batch recovery on Out-Of-Memory events.
    """
    if not batch:
        return []

    # Determine total batch size from the first tensor/list value
    total_size = None
    for v in batch.values():
        if isinstance(v, torch.Tensor) or isinstance(v, (list, tuple)):
            total_size = len(v)
            break

    if total_size is None or total_size <= micro_batch_size or micro_batch_size <= 0:
        return [batch]

    num_splits = math.ceil(total_size / micro_batch_size)
    splits: List[Dict[str, Any]] = [{} for _ in range(num_splits)]

    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            chunks = torch.split(val, micro_batch_size, dim=0)
            for i, chunk in enumerate(chunks):
                splits[i][key] = chunk
        elif isinstance(val, (list, tuple)):
            for i in range(num_splits):
                start_idx = i * micro_batch_size
                end_idx = min((i + 1) * micro_batch_size, total_size)
                sub_list = val[start_idx:end_idx]
                splits[i][key] = type(val)(sub_list)
        else:
            # Replicate non-indexable values across all sub-batches
            for i in range(num_splits):
                splits[i][key] = val

    return splits


def auto_scale_batch_size_and_grad_accum(
    requested_batch_size: int,
    safe_batch_size: int,
    current_grad_accum: int = 1,
) -> Tuple[int, int]:
    """
    Calculate adjusted batch size and gradient accumulation steps to maintain the desired effective batch size.
    effective_batch_size = requested_batch_size * current_grad_accum
    """
    effective_batch_size = max(1, requested_batch_size * max(1, current_grad_accum))
    adjusted_bs = max(1, min(safe_batch_size, requested_batch_size))
    adjusted_grad_accum = max(1, math.ceil(effective_batch_size / adjusted_bs))
    return adjusted_bs, adjusted_grad_accum


def estimate_memory_profile(
    model: nn.Module,
    sample_shape: Tuple[int, int, int] = (3, 256, 256),
    device: Optional[Union[str, torch.device]] = None,
    use_amp: bool = True,
    aux_classifier: bool = True,
    num_classes: int = 3,
    mask_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Empirically estimate the baseline memory footprint (static model weights, optimizer states,
    framework context) and the dynamic per-sample memory slope during forward+backward passes.
    Returns:
      {
        'baseline_mb': float,
        'per_sample_mb': float,
        'total_mb': float,
      }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    summary = get_memory_summary(device)
    total_mb = summary.get("total_mb", 1.0)
    c, h, w = sample_shape
    mask_h, mask_w = mask_shape if mask_shape is not None else (h, w)

    # Estimate theoretical model static footprint
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Optimizer states (AdamW momentum + variance in fp32 = 2x fp32 size)
    opt_bytes = sum(p.numel() * 4 * 2 for p in model.parameters() if p.requires_grad)
    static_model_mb = (param_bytes + opt_bytes) / (1024 ** 2)

    if device.type != "cuda" or not torch.cuda.is_available():
        # CPU approximation based on tensor activation heuristics
        input_mb = (c * h * w * 4) / (1024 ** 2)
        # Activation estimate ~ 20x input size for UNet backprop
        per_sample_mb = max(1.0, input_mb * 20.0)
        baseline_mb = max(10.0, static_model_mb)
        return {
            "baseline_mb": round(baseline_mb, 2),
            "per_sample_mb": round(per_sample_mb, 2),
            "total_mb": round(total_mb, 2),
        }

    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    orig_training = model.training
    orig_state = copy.deepcopy(model.state_dict())
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    try:
        measurements: List[Tuple[int, float]] = []
        for test_bs in [1, 2]:
            clear_memory_cache(device)
            torch.cuda.reset_peak_memory_stats(dev_idx)
            free_start, t_bytes = torch.cuda.mem_get_info(dev_idx)

            images = torch.randn(test_bs, c, h, w, device=device)
            masks = torch.zeros(test_bs, 1, mask_h, mask_w, device=device)
            labels = torch.randint(0, num_classes, (test_bs,), device=device) if aux_classifier else None
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = outputs[0].mean() if isinstance(outputs, (tuple, list)) else outputs.mean()

            scaler.scale(loss).backward()
            peak_alloc = torch.cuda.max_memory_allocated(dev_idx)
            peak_res = torch.cuda.max_memory_reserved(dev_idx)
            free_end, _ = torch.cuda.mem_get_info(dev_idx)
            used_driver = t_bytes - free_end
            step_peak_bytes = max(used_driver, peak_res, (t_bytes - free_start) + peak_alloc)
            step_peak_mb = step_peak_bytes / (1024 ** 2)

            del images, masks, labels, outputs, loss
            optimizer.zero_grad(set_to_none=True)
            clear_memory_cache(device)
            measurements.append((test_bs, step_peak_mb))

        # Slope: delta_M / delta_B
        bs1, m1 = measurements[0]
        bs2, m2 = measurements[1]
        per_sample_mb = max(0.1, (m2 - m1) / max(1, bs2 - bs1))
        baseline_mb = max(static_model_mb, m1 - (per_sample_mb * bs1))

        return {
            "baseline_mb": round(baseline_mb, 2),
            "per_sample_mb": round(per_sample_mb, 2),
            "total_mb": round(total_mb, 2),
        }
    except Exception:
        # Fallback on theoretical calculation
        input_mb = (c * h * w * 2 if use_amp else c * h * w * 4) / (1024 ** 2)
        per_sample_mb = max(5.0, input_mb * 15.0)
        baseline_mb = max(300.0, static_model_mb + 300.0) # CUDA context baseline
        return {
            "baseline_mb": round(baseline_mb, 2),
            "per_sample_mb": round(per_sample_mb, 2),
            "total_mb": round(total_mb, 2),
        }
    finally:
        model.load_state_dict(orig_state)
        if not orig_training:
            model.eval()
        del optimizer, scaler
        clear_memory_cache(device)


def predict_safe_batch_size(
    model: nn.Module,
    sample_shape: Tuple[int, int, int] = (3, 256, 256),
    max_memory_fraction: float = 0.80,
    device: Optional[Union[str, torch.device]] = None,
    max_batch_size: int = 64,
    min_batch_size: int = 1,
    safety_margin: float = 0.05,
    **kwargs: Any,
) -> int:
    """
    Predict the largest safe batch size that guarantees memory usage stays below
    `max_memory_fraction` (e.g. 0.80 for 80% memory limit) using a calibrated linear memory model.
    """
    profile = estimate_memory_profile(model, sample_shape=sample_shape, device=device, **kwargs)
    total_mb = profile["total_mb"]
    baseline_mb = profile["baseline_mb"]
    per_sample_mb = profile["per_sample_mb"]

    target_mb = max(0.0, (max_memory_fraction - safety_margin) * total_mb)
    avail_mb = max(0.0, target_mb - baseline_mb)

    if per_sample_mb <= 0:
        return min_batch_size

    predicted_bs = int(math.floor(avail_mb / per_sample_mb))
    return max(min_batch_size, min(max_batch_size, predicted_bs))


def find_optimal_batch_size(
    model: nn.Module,
    loss_fn: Optional[nn.Module] = None,
    sample_shape: Tuple[int, int, int] = (3, 256, 256),
    device: Optional[Union[str, torch.device]] = None,
    max_batch_size: int = 64,
    min_batch_size: int = 1,
    use_amp: bool = True,
    aux_classifier: bool = True,
    num_classes: int = 3,
    logger: Optional[Any] = None,
    mask_shape: Optional[Tuple[int, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    in_channels: Optional[int] = None,
    force_probe: bool = False,
    max_memory_fraction: float = 0.80,
    target_memory_fraction: Optional[float] = None,
    safety_margin: float = 0.02,
    **kwargs: Any,
) -> int:
    """
    Automatically probe and determine the largest safe batch size that fits in memory
    for the given model, image dimensions, and device without triggering an Out-of-Memory (OOM) error
    and strictly ensuring peak memory usage stays below `max_memory_fraction` (e.g. 0.80 for 80%).
    Performs test forward, loss calculation, backward pass, and optimizer zeroing on dummy tensors.
    """
    if image_size is not None:
        c = in_channels if in_channels is not None else sample_shape[0]
        sample_shape = (c, image_size[0], image_size[1])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    target_fraction = float(target_memory_fraction) if target_memory_fraction is not None else float(max_memory_fraction)
    target_fraction = max(0.05, min(1.0, target_fraction))

    # If running on CPU without strict memory pressure, check host memory summary
    if device.type == "cpu" and not force_probe:
        summary = get_memory_summary("cpu")
        if summary.get("usage_fraction", 0.0) > target_fraction:
            if logger:
                logger.warning(
                    f"⚠️ CPU host RAM ({summary.get('usage_fraction', 0.0)*100:.1f}%) exceeds "
                    f"target threshold ({target_fraction*100:.1f}%). Falling back to min_batch_size={min_batch_size}."
                )
            return min_batch_size
        return max(min_batch_size, max_batch_size)

    # Candidate batch sizes to test in descending order (powers of 2, plus target max)
    candidates = []
    curr = max_batch_size
    while curr >= min_batch_size:
        if curr not in candidates:
            candidates.append(curr)
        if curr % 2 != 0 and curr > 1:
            curr = curr - 1
        else:
            curr = curr // 2
    if min_batch_size not in candidates:
        candidates.append(min_batch_size)
    candidates = sorted(list(set(candidates)), reverse=True)

    if logger:
        logger.info(
            f"🔍 Probing safe batch sizes for memory constraints (target <= {target_fraction * 100:.1f}%): "
            f"candidates = {candidates}"
        )

    # Put model in train mode on target device
    orig_device = next(model.parameters()).device if list(model.parameters()) else device
    orig_training = model.training
    orig_state = copy.deepcopy(model.state_dict())
    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    c, h, w = sample_shape
    safe_bs = min_batch_size
    found_safe_candidate = False
    mask_h, mask_w = mask_shape if mask_shape is not None else (h, w)
    dev_idx = device.index if (device.type == "cuda" and device.index is not None) else (
        torch.cuda.current_device() if device.type == "cuda" and torch.cuda.is_available() else None
    )

    try:
        # Pre-warmup step with min_batch_size on CUDA to initialize CUDA runtime, cuDNN algorithms and autograd engine
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                clear_memory_cache(device)
                w_img = torch.randn(min_batch_size, c, h, w, device=device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    w_out = model(w_img)
                    w_loss = w_out[0].mean() if isinstance(w_out, (tuple, list)) else w_out.mean()
                scaler.scale(w_loss).backward()
                optimizer.zero_grad(set_to_none=True)
                del w_img, w_out, w_loss
                clear_memory_cache(device)
            except Exception:
                pass

        for bs in candidates:
            clear_memory_cache(device)
            free_start = 0
            total_bytes = 1
            if device.type == "cuda" and dev_idx is not None:
                torch.cuda.reset_peak_memory_stats(dev_idx)
                try:
                    free_start, total_bytes = torch.cuda.mem_get_info(dev_idx)
                except Exception:
                    pass

            try:
                # Generate dummy batch
                images = torch.randn(bs, c, h, w, device=device)
                masks = torch.zeros(bs, 1, mask_h, mask_w, device=device)
                labels = torch.randint(0, num_classes, (bs,), device=device) if aux_classifier else None

                optimizer.zero_grad(set_to_none=True)

                probe_kwargs: Dict[str, Any] = {}
                is_gap_sam = (
                    getattr(model, "model_name", "") in ("gap_sam", "gap-sam", "gapsam")
                    or getattr(model, "is_gap_sam", False)
                    or hasattr(model, "bypass_vae_for_cached_training")
                    or getattr(model, "vae_bypassed", False)
                )
                if is_gap_sam or hasattr(model, "forward_cached"):
                    probe_kwargs["f_r"] = torch.zeros(bs, 256, device=device)

                with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    outputs = model(images, **probe_kwargs) if probe_kwargs else model(images)
                    # Dynamically reconcile mask spatial dimensions if model outputs a different size
                    pred_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    if (
                        pred_tensor is not None
                        and hasattr(pred_tensor, "shape")
                        and pred_tensor.dim() >= 4
                        and pred_tensor.shape[-2:] != masks.shape[-2:]
                    ):
                        masks = torch.zeros(bs, 1, pred_tensor.shape[-2], pred_tensor.shape[-1], device=device)

                    if loss_fn is not None:
                        loss, _ = loss_fn(outputs, masks, labels)
                    else:
                        if isinstance(outputs, (tuple, list)):
                            mask_out = outputs[0]
                            cls_out = outputs[1] if len(outputs) > 1 else None
                            loss = mask_out.mean() + (cls_out.mean() if cls_out is not None else 0.0)
                        else:
                            loss = outputs.mean()

                scaler.scale(loss).backward()
                optimizer.zero_grad(set_to_none=True)

                # Measure peak memory consumption during this pass
                if device.type == "cuda" and dev_idx is not None:
                    peak_alloc = torch.cuda.max_memory_allocated(dev_idx)
                    peak_res = torch.cuda.max_memory_reserved(dev_idx)
                    try:
                        free_end, total_bytes = torch.cuda.mem_get_info(dev_idx)
                        used_driver = total_bytes - free_end
                    except Exception:
                        used_driver = peak_res
                    peak_used = max(used_driver, peak_res, (total_bytes - free_start) + peak_alloc)
                    peak_fraction = peak_used / max(1.0, float(total_bytes))
                else:
                    summary = get_memory_summary("cpu")
                    peak_fraction = summary.get("usage_fraction", 0.0)

                # Cleanup dummy tensors
                del images, masks, labels, outputs, loss
                clear_memory_cache(device)

                # Evaluate against memory threshold
                if peak_fraction > target_fraction:
                    if logger:
                        logger.warning(
                            f"⚠️ Batch size {bs} consumed {peak_fraction * 100:.1f}% memory "
                            f"(exceeds target limit {target_fraction * 100:.1f}%). Trying smaller candidate..."
                        )
                    continue

                safe_bs = bs
                found_safe_candidate = True
                if logger:
                    logger.info(
                        f"✅ Batch size {bs} succeeded within memory limit: "
                        f"peak memory {peak_fraction * 100:.1f}% <= {target_fraction * 100:.1f}% "
                        f"({format_memory_summary(device)})."
                    )
                break

            except Exception as exc:
                if is_oom_error(exc):
                    if logger:
                        logger.warning(f"⚠️ Batch size {bs} triggered OOM on {device}. Trying smaller candidate...")
                    clear_memory_cache(device)
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # If error is not memory related, raise it
                    clear_memory_cache(device)
                    raise exc

        if not found_safe_candidate and logger:
            logger.warning(
                f"⚠️ None of the tested batch sizes satisfied the strict {target_fraction * 100:.1f}% "
                f"memory threshold. Falling back to min_batch_size={min_batch_size}."
            )
    finally:
        # Restore original model weights, running stats, and training mode
        model.load_state_dict(orig_state)
        if not orig_training:
            model.eval()
        else:
            model.train()

        try:
            del optimizer, scaler
        except Exception:
            pass

        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)

        clear_memory_cache(device)
        gc.collect()

    return max(min_batch_size, safe_bs)


class MemoryGuard:
    """
    Real-time memory monitor and dynamic memory enforcer for SID-UNet.
    Ensures memory usage (VRAM on GPU, RAM on CPU) stays below a specified threshold (e.g. 80%).
    Provides threshold monitoring, cache flushing, dynamic batch size scaling recommendations,
    and formatted logging for progress bars and metric tracking.
    """
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        max_memory_fraction: float = 0.80,
        warning_threshold: Optional[float] = None,
        logger: Optional[Any] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.max_memory_fraction = max(0.05, min(1.0, float(max_memory_fraction)))
        self.warning_threshold = float(warning_threshold) if warning_threshold is not None else max(
            0.05, self.max_memory_fraction - 0.05
        )
        self.logger = logger
        self.peak_fraction_history: List[float] = []
        self._last_warn_step: int = -1

    def reset_peak_stats(self) -> None:
        """Reset peak memory tracking counters for the device."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                dev_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
                torch.cuda.reset_peak_memory_stats(dev_idx)
            except Exception:
                pass

    def get_memory_status(self) -> Dict[str, Any]:
        """Query current memory summary and fractions."""
        return get_memory_summary(self.device)

    def is_memory_exceeded(self, threshold: Optional[float] = None) -> bool:
        """Check whether current or peak memory usage exceeds the threshold."""
        target = float(threshold) if threshold is not None else self.max_memory_fraction
        status = self.get_memory_status()
        peak_frac = status.get("peak_fraction", status.get("usage_fraction", 0.0))
        return peak_frac > target

    def check_memory(self, step: Optional[int] = None) -> Tuple[bool, float]:
        """
        Check memory status against maximum allowed memory fraction.
        Returns (is_exceeded, peak_fraction).
        Logs warning if warning threshold is reached.
        """
        status = self.get_memory_status()
        peak_frac = float(status.get("peak_fraction", status.get("usage_fraction", 0.0)))
        self.peak_fraction_history.append(peak_frac)

        is_exceeded = peak_frac > self.max_memory_fraction
        is_warning = peak_frac > self.warning_threshold

        if (is_exceeded or is_warning) and self.logger is not None:
            should_log = (step is None) or (step - self._last_warn_step >= 10)
            if should_log:
                if step is not None:
                    self._last_warn_step = step
                level_str = "EXCEEDED LIMIT" if is_exceeded else "Approaching Limit"
                self.logger.warning(
                    f"⚠️ [MemoryGuard] Memory usage {level_str}: {peak_frac * 100:.1f}% "
                    f"(Target limit: {self.max_memory_fraction * 100:.1f}%, "
                    f"Warning threshold: {self.warning_threshold * 100:.1f}%). "
                    f"[{format_memory_summary(self.device)}]"
                )

        return is_exceeded, peak_frac

    def recover_memory(self) -> float:
        """
        Trigger memory reclamation (garbage collection and PyTorch cache clearing).
        Returns new memory usage fraction after recovery.
        """
        clear_memory_cache(self.device)
        status = self.get_memory_status()
        return float(status.get("usage_fraction", 0.0))

    def recommend_batch_scaling(
        self,
        current_batch_size: int,
        current_grad_accum: int = 1,
        reduction_factor: float = 0.5,
    ) -> Tuple[int, int]:
        """
        Recommend scaled-down batch size and adjusted gradient accumulation steps
        to bring memory usage below max_memory_fraction while maintaining effective batch size.
        """
        new_bs = max(1, int(math.floor(current_batch_size * reduction_factor)))
        return auto_scale_batch_size_and_grad_accum(
            requested_batch_size=current_batch_size,
            safe_batch_size=new_bs,
            current_grad_accum=current_grad_accum,
        )

    def format_status(self) -> str:
        """Return compact string for progress bars or metric logs."""
        status = self.get_memory_status()
        used = status.get("used_mb", 0.0)
        total = status.get("total_mb", 1.0)
        frac = status.get("peak_fraction", status.get("usage_fraction", 0.0))
        return f"{frac * 100:.1f}% ({used:.0f}/{total:.0f}MB)"
