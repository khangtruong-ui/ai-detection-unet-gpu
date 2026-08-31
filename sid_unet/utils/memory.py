"""
Memory management and Out-of-Memory (OOM) handling utilities for SID-UNet.
Provides OOM error detection, GPU memory tracking, automatic memory cache clearing,
batch splitting for micro-batching, and automatic batch size finding.
"""

from __future__ import annotations

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
    Get detailed memory metrics for the specified device (or current CUDA device if available).
    Returns dictionary with allocated, reserved, peak allocated, and total memory in megabytes (MB).
    """
    summary = {
        "device_type": "cpu",
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
        "total_mb": 0.0,
        "free_mb": 0.0,
    }

    if not torch.cuda.is_available():
        return summary

    dev = device if isinstance(device, torch.device) else torch.device(device or "cuda")
    if dev.type != "cuda":
        return summary

    try:
        dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(dev_idx) / (1024 ** 2)
        props = torch.cuda.get_device_properties(dev_idx)
        total = props.total_memory / (1024 ** 2)
        free = total - reserved

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
        })
    except Exception:
        pass

    return summary


def format_memory_summary(device: Optional[Union[str, torch.device]] = None) -> str:
    """
    Return a formatted human-readable summary of current GPU memory usage.
    """
    stats = get_memory_summary(device)
    if stats["device_type"] != "cuda":
        return "Memory: CPU"
    return (
        f"VRAM: {stats['allocated_mb']:.1f} MB allocated | "
        f"{stats['reserved_mb']:.1f} MB reserved | "
        f"Peak: {stats['max_allocated_mb']:.1f} MB (Total: {stats['total_mb']:.0f} MB)"
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
) -> int:
    """
    Automatically probe and determine the largest safe batch size that fits in memory
    for the given model, image dimensions, and device without triggering an Out-of-Memory (OOM) error.
    Performs test forward, loss calculation, backward pass, and optimizer zeroing on dummy tensors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # If running on CPU without strict memory pressure, default to requested max_batch_size
    if device.type == "cpu":
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
        logger.info(f"🔍 Probing safe batch sizes for memory constraints: candidates = {candidates}")

    # Put model in train mode on target device
    orig_device = next(model.parameters()).device if list(model.parameters()) else device
    orig_training = model.training
    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    c, h, w = sample_shape
    safe_bs = min_batch_size

    for bs in candidates:
        clear_memory_cache(device)
        try:
            # Generate dummy batch
            images = torch.randn(bs, c, h, w, device=device)
            masks = torch.zeros(bs, 1, h, w, device=device)
            labels = torch.randint(0, num_classes, (bs,), device=device) if aux_classifier else None

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                outputs = model(images)
                if loss_fn is not None:
                    loss, _ = loss_fn(outputs, masks, labels)
                else:
                    if isinstance(outputs, tuple):
                        mask_out, cls_out = outputs
                        loss = mask_out.mean() + (cls_out.mean() if cls_out is not None else 0.0)
                    else:
                        loss = outputs.mean()

            scaler.scale(loss).backward()
            optimizer.zero_grad(set_to_none=True)

            # Cleanup dummy tensors
            del images, masks, labels, outputs, loss
            clear_memory_cache(device)

            safe_bs = bs
            if logger:
                logger.info(f"✅ Batch size {bs} succeeded within device memory ({format_memory_summary(device)}).")
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

    # Restore original model mode
    if not orig_training:
        model.eval()

    clear_memory_cache(device)
    return max(min_batch_size, safe_bs)
