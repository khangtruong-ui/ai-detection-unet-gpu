"""
Memory management and Out-of-Memory (OOM) handling utilities for SID-UNet.
Provides OOM error detection, GPU memory tracking, automatic memory cache clearing,
batch splitting for micro-batching, and automatic batch size finding.
"""

from __future__ import annotations

import copy
import ctypes
import gc
import inspect
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from sid_unet.utils.precision import is_bf16_supported


def is_oom_error(exc: BaseException) -> bool:
    """
    Check whether an exception was caused by CUDA or host memory exhaustion (Out of Memory).
    Handles torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, MemoryError,
    torch.AcceleratorError, and RuntimeError with CUDA memory/launch limit patterns.
    """
    accelerator_err_cls = getattr(torch, "AcceleratorError", ())
    if isinstance(exc, (MemoryError, getattr(torch, "OutOfMemoryError", ()))) or (
        hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError)
    ) or (
        accelerator_err_cls and isinstance(exc, accelerator_err_cls)
    ):
        return True

    exc_name = type(exc).__name__.lower()
    if "acceleratorerror" in exc_name:
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
        "acceleratorerror",
        "too many resources requested for launch",
    ]
    return any(pattern in exc_msg for pattern in oom_patterns)


def clear_memory_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Clear Python garbage collection and release cached allocator memory in PyTorch CUDA/accelerators.
    Safe to call on any device (CPU or GPU).
    """
    gc.collect()
    if "pyarrow" in sys.modules:
        try:
            sys.modules["pyarrow"].default_memory_pool().release_unused()
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
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

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
            free = free_bytes / (1024 ** 2)
            total = total_bytes / (1024 ** 2)
        except Exception:
            free = max(0.0, total - reserved)

        used = max(0.0, total - free)

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
    used_mb = stats.get("used_mb", stats["total_mb"] - stats["free_mb"])
    return (
        f"VRAM: {used_mb:.1f}/{stats['total_mb']:.0f} MB used "
        f"({stats['free_mb']:.1f} MB free) | "
        f"PyTorch: {stats['allocated_mb']:.1f} MB allocated, "
        f"{stats['reserved_mb']:.1f} MB reserved | "
        f"Peak: {stats['max_allocated_mb']:.1f} MB"
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


def is_auto_batch_size(value: Any) -> bool:
    """Return True when a configured batch_size requests automatic GPU-saturating inference (-1 or 'auto')."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == -1
    if isinstance(value, str):
        return value.strip().lower() in ("-1", "auto")
    return False


def is_auto_batch_size_config(config: Any) -> bool:
    """Return True when config specifies batch_size = -1 (or 'auto') in data or training sections."""
    if config is None:
        return False
    data_cfg = getattr(config, "data", None)
    if data_cfg is None and hasattr(config, "get"):
        data_cfg = config.get("data", None)
    if data_cfg is not None and hasattr(data_cfg, "get"):
        if bool(data_cfg.get("_auto_infer_batch_size", False)):
            return True
        if is_auto_batch_size(data_cfg.get("batch_size", None)):
            return True
        if is_auto_batch_size(data_cfg.get("base_batch_size", None)):
            return True
    training_cfg = getattr(config, "training", None)
    if training_cfg is None and hasattr(config, "get"):
        training_cfg = config.get("training", None)
    if training_cfg is not None and hasattr(training_cfg, "get"):
        if is_auto_batch_size(training_cfg.get("batch_size", None)):
            return True
    return False


def _prepare_probe_inputs(
    model: nn.Module,
    batch_size: int,
    sample_shape: Tuple[int, int, int],
    mask_shape: Tuple[int, int],
    device: torch.device,
    aux_classifier: bool,
    num_classes: int,
    should_pass_f_r: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Build dummy probe tensors and model forward kwargs for a candidate batch size."""
    c, h, w = sample_shape
    mask_h, mask_w = mask_shape
    raw_model = getattr(model, "module", model)

    images = torch.randn(batch_size, c, h, w, device=device)
    masks = torch.zeros(batch_size, 1, mask_h, mask_w, device=device)
    if getattr(raw_model, "requires_gt_masks", False):
        masks[:, :, : max(1, mask_h // 2), :] = 1.0
    labels = torch.randint(0, num_classes, (batch_size,), device=device) if aux_classifier else None

    probe_kwargs: Dict[str, Any] = {}
    if should_pass_f_r:
        probe_kwargs["f_r"] = torch.zeros(batch_size, 256, device=device)
    if getattr(raw_model, "requires_gt_masks", False):
        probe_kwargs["gt_masks"] = masks
    return images, masks, labels, probe_kwargs


def _extract_mask_logits(outputs: Any) -> Optional[torch.Tensor]:
    """Extract primary mask logits tensor from Tensor, tuple/list, or dict model outputs."""
    if isinstance(outputs, (tuple, list)):
        return outputs[0] if outputs else None
    if isinstance(outputs, dict):
        for k in ("mask_logits", "pred_masks", "masks", "logits"):
            if k in outputs and outputs[k] is not None:
                return outputs[k]
        return None
    return outputs


def _get_available_host_ram_bytes() -> Optional[int]:
    """Return currently available host RAM in bytes, or None if unavailable."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass
    return None


def _is_probe_memory_failure(exc: BaseException) -> bool:
    """Return True if exception during memory probing indicates VRAM or hardware launch exhaustion."""
    exc_str = str(exc).lower()
    return (
        is_oom_error(exc)
        or "acceleratorerror" in type(exc).__name__.lower()
        or "cuda error" in exc_str
        or "cudamalloc" in exc_str
        or "invalid argument" in exc_str
        or "device-side assert" in exc_str
    )


def infer_optimal_batch_size(
    model: nn.Module,
    loss_fn: Optional[nn.Module] = None,
    sample_shape: Tuple[int, int, int] = (3, 256, 256),
    device: Optional[Union[str, torch.device]] = None,
    mode: str = "train",
    min_batch_size: int = 1,
    max_batch_size: int = 256,
    target_memory_fraction: Optional[float] = None,
    use_amp: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
    aux_classifier: bool = False,
    num_classes: int = 3,
    mask_shape: Optional[Tuple[int, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    in_channels: Optional[int] = None,
    cpu_fallback_batch_size: int = 16,
    logger: Optional[Any] = None,
    force_probe: bool = False,
    refiner: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    prefetch_batches: int = 3,
    host_prefetch_batches: int = 10,
    max_host_memory_fraction: float = 0.30,
    timeout_seconds: float = 60.0,
) -> int:
    """
    Automatically infer the largest safe batch size that saturates available GPU memory
    up to `target_memory_fraction` without triggering Out-of-Memory (OOM) errors.

    Uses exponential doubling followed by bounded binary search refinement.
    Supports both `mode="train"` (forward + backward + optimizer reserve) and
    `mode="eval"` (inference_mode forward + optional loss/SAM refiner).
    """
    norm_mode = "eval" if str(mode).strip().lower() in ("eval", "evaluate", "inference", "test", "val") else "train"
    if image_size is not None:
        c = in_channels if in_channels is not None else sample_shape[0]
        sample_shape = (c, int(image_size[0]), int(image_size[1]))

    min_batch_size = max(1, int(min_batch_size))
    max_batch_size = max(min_batch_size, int(max_batch_size))

    if target_memory_fraction is None:
        target_frac = 0.78 if norm_mode == "eval" else 0.75
    else:
        target_frac = max(0.1, min(0.98, float(target_memory_fraction)))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda" and not force_probe:
        fallback = max(min_batch_size, min(int(cpu_fallback_batch_size), max_batch_size))
        if logger:
            logger.info(f"ℹ️ Auto batch size inference on {device.type.upper()}: using fallback batch_size={fallback}.")
        return fallback

    is_dp = isinstance(model, nn.DataParallel)
    dp_devices = list(model.device_ids) if (is_dp and getattr(model, "device_ids", None)) else []
    step_unit = len(dp_devices) if len(dp_devices) > 1 else 1
    raw_model = getattr(model, "module", model)

    c, h, w = sample_shape
    mask_h, mask_w = mask_shape if mask_shape is not None else (h, w)

    # Cap max_batch_size against available host RAM so DataLoader worker queues + Arrow row-group decode cannot trigger Linux OOM (SIGKILL 137)
    if device.type == "cuda" and not force_probe:
        avail_ram = _get_available_host_ram_bytes()
        if avail_ram is not None and avail_ram > 0:
            per_sample_host_bytes = max(1, (c * h * w + mask_h * mask_w) * 4 * 2)
            host_baseline_bytes = int(3.5 * (1024 ** 3)) if avail_ram > int(4.5 * (1024 ** 3)) else int(avail_ram * 0.35)
            usable_ram = max(int(avail_ram * 0.25), avail_ram - host_baseline_bytes)
            host_budget = int(usable_ram * max(0.15, min(0.75, float(max_host_memory_fraction))))
            ram_safe_bs = max(min_batch_size, host_budget // (max(1, int(host_prefetch_batches)) * per_sample_host_bytes))
            if ram_safe_bs < max_batch_size:
                if logger:
                    logger.info(
                        f"ℹ️ Capping auto batch_size probe upper bound {max_batch_size} -> {ram_safe_bs} "
                        f"for host RAM safety (avail={avail_ram / (1024 ** 3):.2f} GiB)."
                    )
                max_batch_size = max(min_batch_size, ram_safe_bs)

    if amp_dtype is None:
        if use_amp and device.type == "cuda" and torch.cuda.is_available():
            amp_dtype = torch.bfloat16 if is_bf16_supported(device=device) else torch.float16
        else:
            amp_dtype = torch.float32
    amp_active = bool(use_amp and device.type == "cuda" and torch.cuda.is_available() and amp_dtype != torch.float32)

    orig_training = raw_model.training
    bn_stats = {}
    if norm_mode == "train":
        for m in raw_model.modules():
            if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,)):
                bn_stats[m] = (
                    m.running_mean.clone() if m.running_mean is not None else None,
                    m.running_var.clone() if m.running_var is not None else None,
                    m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None,
                )

    if not is_dp:
        model = model.to(device)
    if norm_mode == "eval":
        model.eval()
    else:
        model.train()

    try:
        forward_params = inspect.signature(raw_model.forward).parameters
        accepts_f_r = "f_r" in forward_params
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values())
    except Exception:
        accepts_f_r = False
        accepts_kwargs = False

    is_gap_sam = (
        getattr(raw_model, "model_name", "") in ("gap_sam", "gap-sam", "gapsam")
        or getattr(raw_model, "is_gap_sam", False)
        or hasattr(raw_model, "bypass_vae_for_cached_training")
        or getattr(raw_model, "vae_bypassed", False)
    )
    should_pass_f_r = accepts_f_r or (is_gap_sam and accepts_kwargs)

    # Estimate optimizer state reserve if training optimizer hasn't materialized AdamW moments yet
    opt_reserve_bytes = 0
    if norm_mode == "train" and device.type == "cuda":
        has_opt_state = optimizer is not None and len(getattr(optimizer, "state", {})) > 0
        if not has_opt_state:
            trainable_elems = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            opt_reserve_bytes = int(trainable_elems * 8)

    use_scaler = bool(norm_mode == "train" and amp_active and amp_dtype == torch.float16)
    probe_opt = None
    scaler = None
    if norm_mode == "train":
        trainable_params = [p for p in raw_model.parameters() if p.requires_grad] or list(raw_model.parameters())
        if trainable_params:
            probe_opt = torch.optim.SGD(trainable_params, lr=1e-3)
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    dev_idx = (device.index if device.index is not None else torch.cuda.current_device()) if (device.type == "cuda" and torch.cuda.is_available()) else 0

    if logger:
        logger.info(
            f"🔍 Auto-inferring GPU-saturating batch size (mode={norm_mode}, shape={sample_shape}, "
            f"range=[{min_batch_size}, {max_batch_size}], target_vram={target_frac * 100:.0f}%)..."
        )
        sys.stdout.flush()

    def _run_trial(bs: int) -> Tuple[bool, str]:
        """Run a single trial batch size; returns (passed, reason)."""
        clear_memory_cache(device)
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(dev_idx)
            except Exception:
                pass

        images = masks = labels = outputs = loss = None
        try:
            images, masks, labels, probe_kwargs = _prepare_probe_inputs(
                model=model,
                batch_size=bs,
                sample_shape=sample_shape,
                mask_shape=(mask_h, mask_w),
                device=device,
                aux_classifier=aux_classifier,
                num_classes=num_classes,
                should_pass_f_r=should_pass_f_r,
            )

            # DataParallel safety for small batches or non-tensor kwargs
            fwd_model = getattr(model, "module", model) if (is_dp and (bs < step_unit or bs <= 1)) else model
            if isinstance(fwd_model, nn.DataParallel):
                fwd_kwargs = {k: v for k, v in probe_kwargs.items() if isinstance(v, torch.Tensor)}
            else:
                fwd_kwargs = probe_kwargs

            if norm_mode == "eval":
                with torch.inference_mode():
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_active):
                        outputs = fwd_model(images, **fwd_kwargs) if fwd_kwargs else fwd_model(images)
                        pred_tensor = _extract_mask_logits(outputs)
                        if (
                            pred_tensor is not None
                            and hasattr(pred_tensor, "shape")
                            and pred_tensor.dim() >= 4
                            and pred_tensor.shape[-2:] != masks.shape[-2:]
                        ):
                            masks = torch.zeros(bs, 1, pred_tensor.shape[-2], pred_tensor.shape[-1], device=device)
                        if loss_fn is not None:
                            loss, _ = loss_fn(outputs, masks, labels)
                        if refiner is not None and pred_tensor is not None:
                            try:
                                refiner.refine_batch(images, pred_tensor, is_logit=True)
                            except TypeError:
                                refiner.refine_batch(images, pred_tensor)
            else:
                if probe_opt is not None:
                    probe_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_active):
                    outputs = fwd_model(images, **fwd_kwargs) if fwd_kwargs else fwd_model(images)
                    pred_tensor = _extract_mask_logits(outputs)
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
                        elif isinstance(outputs, dict):
                            loss = pred_tensor.mean() if pred_tensor is not None else next(iter(outputs.values())).mean()
                        else:
                            loss = outputs.mean()
                if loss is not None and getattr(loss, "requires_grad", False):
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                if probe_opt is not None:
                    probe_opt.zero_grad(set_to_none=True)

            # Check GPU VRAM budget while trial tensors / activations are still live
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
                live_used = max(0, total_bytes - free_bytes)
                peak_alloc = torch.cuda.max_memory_allocated(dev_idx)
                prefetch_reserve = int((prefetch_batches + 1) * bs * (c * h * w + mask_h * mask_w) * 4)
                projected_used = max(live_used, peak_alloc) + prefetch_reserve + opt_reserve_bytes
                budget_bytes = int(total_bytes * target_frac)
                if bs > min_batch_size and projected_used > budget_bytes:
                    return False, (
                        f"projected VRAM {projected_used / (1024 ** 2):.1f} MB > "
                        f"budget {budget_bytes / (1024 ** 2):.1f} MB ({target_frac * 100:.0f}%)"
                    )

            return True, "ok"

        except Exception as exc:
            if _is_probe_memory_failure(exc):
                return False, f"OOM/memory limit: {exc}"
            if norm_mode == "train" and bs == 1 and "expected more than 1 value per channel" in str(exc).lower():
                return False, "bn_single_sample"
            raise
        finally:
            try:
                del images, masks, labels, outputs, loss
            except Exception:
                pass
            if probe_opt is not None:
                probe_opt.zero_grad(set_to_none=True)
            clear_memory_cache(device)

    t0 = time.monotonic()
    last_ok: Optional[int] = None
    first_fail: Optional[int] = None
    tested: Dict[int, bool] = {}

    try:
        start_bs = max(min_batch_size, step_unit)
        if start_bs > max_batch_size:
            start_bs = min_batch_size

        # Phase 1: Exponential doubling
        bs = start_bs
        while bs <= max_batch_size:
            ok, reason = _run_trial(bs)
            tested[bs] = ok
            if not ok and reason == "bn_single_sample" and bs == 1 and max_batch_size >= 2:
                # BatchNorm requires >= 2 samples in training mode
                bs = 2
                continue
            if ok:
                last_ok = bs
                if bs == max_batch_size:
                    break
                if (time.monotonic() - t0) > timeout_seconds:
                    break
                next_bs = min(max_batch_size, bs * 2)
                if step_unit > 1 and next_bs < max_batch_size:
                    next_bs = max(bs + step_unit, (next_bs // step_unit) * step_unit)
                if next_bs == bs:
                    break
                bs = next_bs
            else:
                first_fail = bs
                if bs == start_bs and start_bs > min_batch_size:
                    # Fallback if step_unit start failed
                    ok_min, _ = _run_trial(min_batch_size)
                    tested[min_batch_size] = ok_min
                    if ok_min:
                        last_ok = min_batch_size
                break

        # Phase 2: Binary search refinement in (last_ok, first_fail)
        if (
            last_ok is not None
            and first_fail is not None
            and (first_fail - last_ok) > step_unit
            and (time.monotonic() - t0) <= timeout_seconds
        ):
            low = last_ok
            high = first_fail
            for _ in range(5):
                if (high - low) <= max(step_unit, max(1, last_ok // 8)):
                    break
                if (time.monotonic() - t0) > timeout_seconds:
                    break
                mid = (low + high) // 2
                if step_unit > 1:
                    mid = (mid // step_unit) * step_unit
                elif mid > 4 and (mid % 2 != 0):
                    mid = mid - 1
                if mid <= low:
                    mid = low + step_unit
                if mid >= high:
                    break
                if mid in tested:
                    ok_mid = tested[mid]
                else:
                    ok_mid, _ = _run_trial(mid)
                    tested[mid] = ok_mid
                if ok_mid:
                    low = mid
                    last_ok = mid
                else:
                    high = mid

        inferred_bs = max(min_batch_size, last_ok if last_ok is not None else min_batch_size)
    finally:
        for m, (rm, rv, nbt) in bn_stats.items():
            if rm is not None and m.running_mean is not None:
                m.running_mean.copy_(rm)
            if rv is not None and m.running_var is not None:
                m.running_var.copy_(rv)
            if nbt is not None and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(nbt)

        if not orig_training:
            raw_model.eval()
        else:
            raw_model.train()

        try:
            del probe_opt, scaler
        except Exception:
            pass

        if hasattr(raw_model, "zero_grad"):
            raw_model.zero_grad(set_to_none=True)

        clear_memory_cache(device)
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(dev_idx)
            except Exception:
                pass
        gc.collect()

    if logger:
        logger.info(
            f"✅ Inferred optimal batch_size={inferred_bs} (mode={norm_mode}, {format_memory_summary(device)})."
        )
        sys.stdout.flush()

    return inferred_bs


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
    **kwargs: Any,
) -> int:
    """
    Automatically probe and determine the largest safe batch size that fits in memory
    for the given model, image dimensions, and device without triggering an Out-of-Memory (OOM) error.
    Performs test forward, loss calculation, backward pass, and optimizer zeroing on dummy tensors.
    """
    if max_batch_size == -1 or kwargs.get("saturate", False) or str(kwargs.get("mode", "train")).lower() in ("eval", "evaluate", "inference"):
        return infer_optimal_batch_size(
            model=model,
            loss_fn=loss_fn,
            sample_shape=sample_shape,
            device=device,
            mode=kwargs.get("mode", "train"),
            min_batch_size=min_batch_size,
            max_batch_size=256 if max_batch_size == -1 else max_batch_size,
            target_memory_fraction=kwargs.get("target_memory_fraction", None),
            use_amp=use_amp,
            amp_dtype=kwargs.get("amp_dtype", None),
            aux_classifier=aux_classifier,
            num_classes=num_classes,
            mask_shape=mask_shape,
            image_size=image_size,
            in_channels=in_channels,
            logger=logger,
            force_probe=force_probe,
            refiner=kwargs.get("refiner", None),
            optimizer=kwargs.get("optimizer", None),
        )

    if image_size is not None:
        c = in_channels if in_channels is not None else sample_shape[0]
        sample_shape = (c, image_size[0], image_size[1])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # If running on CPU without strict memory pressure, default to requested max_batch_size unless force_probe
    if device.type == "cpu" and not force_probe:
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
        sys.stdout.flush()

    # Put model in train mode on target device
    orig_device = next(model.parameters()).device if list(model.parameters()) else device
    orig_training = model.training

    # Preserve BatchNorm running stats without deepcopying entire model weights
    bn_stats = {}
    for m in model.modules():
        if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,)):
            bn_stats[m] = (
                m.running_mean.clone() if m.running_mean is not None else None,
                m.running_var.clone() if m.running_var is not None else None,
                m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None,
            )

    model = model.to(device)
    model.train()

    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_available() and is_bf16_supported(device=device))
        else torch.float16
    )
    use_scaler = bool(use_amp and device.type == "cuda" and amp_dtype == torch.float16)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad] or list(model.parameters()), lr=1e-3)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    c, h, w = sample_shape
    safe_bs = min_batch_size
    mask_h, mask_w = mask_shape if mask_shape is not None else (h, w)

    # Determine whether model accepts 'f_r' (e.g. GAP-SAM) or expects only input tensor x
    try:
        forward_params = inspect.signature(model.forward).parameters
        accepts_f_r = "f_r" in forward_params
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values())
    except Exception:
        accepts_f_r = False
        accepts_kwargs = False

    is_gap_sam = (
        getattr(model, "model_name", "") in ("gap_sam", "gap-sam", "gapsam")
        or getattr(model, "is_gap_sam", False)
        or hasattr(model, "bypass_vae_for_cached_training")
        or getattr(model, "vae_bypassed", False)
    )
    should_pass_f_r = accepts_f_r or (is_gap_sam and accepts_kwargs)

    try:
        for bs in candidates:
            clear_memory_cache(device)
            if logger:
                logger.info(f"   Testing candidate batch size {bs}...")
                sys.stdout.flush()
            try:
                # Generate dummy batch
                images = torch.randn(bs, c, h, w, device=device)
                masks = torch.zeros(bs, 1, mask_h, mask_w, device=device)
                if getattr(model, "requires_gt_masks", False):
                    masks[:, :, : max(1, mask_h // 2), :] = 1.0
                labels = torch.randint(0, num_classes, (bs,), device=device) if aux_classifier else None

                optimizer.zero_grad(set_to_none=True)

                probe_kwargs: Dict[str, Any] = {}
                if should_pass_f_r:
                    probe_kwargs["f_r"] = torch.zeros(bs, 256, device=device)
                if getattr(model, "requires_gt_masks", False):
                    probe_kwargs["gt_masks"] = masks

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
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

                # Cleanup dummy tensors
                del images, masks, labels, outputs, loss
                clear_memory_cache(device)

                safe_bs = bs
                if logger:
                    logger.info(f"✅ Batch size {bs} succeeded within device memory ({format_memory_summary(device)}).")
                    sys.stdout.flush()
                break

            except Exception as exc:
                exc_str = str(exc).lower()
                is_probe_failure = (
                    is_oom_error(exc)
                    or "acceleratorerror" in type(exc).__name__.lower()
                    or "cuda error" in exc_str
                    or "cudamalloc" in exc_str
                    or "invalid argument" in exc_str
                    or "device-side assert" in exc_str
                )
                if is_probe_failure:
                    if logger:
                        logger.warning(f"⚠️ Batch size {bs} triggered hardware/memory limit ({exc}) on {device}. Trying smaller candidate...")
                        sys.stdout.flush()
                    clear_memory_cache(device)
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # If error is not memory related, raise it
                    clear_memory_cache(device)
                    raise exc
    finally:
        # Restore original running stats and training mode without costly full state_dict copies
        for m, (rm, rv, nbt) in bn_stats.items():
            if rm is not None and m.running_mean is not None:
                m.running_mean.copy_(rm)
            if rv is not None and m.running_var is not None:
                m.running_var.copy_(rv)
            if nbt is not None and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(nbt)

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
