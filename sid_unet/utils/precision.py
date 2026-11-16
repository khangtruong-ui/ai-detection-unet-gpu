"""
FP8 / BF8 (float8_e4m3fn) and Mixed Precision utilities for SID-UNet.

Provides hardware capability checks, unified precision resolution across
training, evaluation, cross-evaluation, and prediction, and safe model
preparation using PyTorch + TorchAO FP8 kernels on supported GPUs
(NVIDIA Ada Lovelace SM 8.9+, Hopper SM 9.0+, Blackwell SM 10.0+).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("sid_unet.precision")

_FP8_ALIASES = frozenset({"fp8", "bf8", "e4m3", "e4m3fn", "float8", "float8_e4m3", "float8_e4m3fn"})
_BF16_ALIASES = frozenset({"bf16", "bfloat16"})
_FP16_ALIASES = frozenset({"fp16", "float16", "half", "16", "16bit", "gpu_16bit"})
_FP32_ALIASES = frozenset({"fp32", "float32", "32", "full", "none", "off", "false"})
_AUTO_ALIASES = frozenset({"auto", "", "default", "true"})


@dataclass(frozen=True)
class PrecisionSpec:
    """Resolved execution precision configuration for training or inference."""

    mode: str                     # Normalized active mode: "fp8", "bf16", "fp16", "fp32"
    requested_mode: str           # Original normalized request ("auto", "fp8", "bf16", "fp16", "fp32")
    fp8_enabled: bool             # Whether FP8 e4m3 conversion is active
    fp8_dtype: Optional[torch.dtype]  # torch.float8_e4m3fn when fp8_enabled else None
    amp_enabled: bool             # Whether torch.amp.autocast is active
    amp_dtype: torch.dtype        # Fallback / non-FP8 op autocast dtype (bfloat16, float16, float32)
    amp_dtype_str: str            # String representation ("fp8_e4m3fn", "bfloat16", "float16", "float32")
    use_grad_scaler: bool         # True only for float16 AMP
    fp8_all_parameters: bool = True
    fallback_reason: Optional[str] = None


def normalize_precision_mode(value: Any) -> str:
    """Normalize user or config precision strings into canonical tokens."""
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "auto" if value else "fp32"
    raw = str(value).strip().lower()
    if raw in _AUTO_ALIASES:
        return "auto"
    if raw in _FP8_ALIASES:
        return "fp8"
    if raw in _BF16_ALIASES:
        return "bf16"
    if raw in _FP16_ALIASES:
        return "fp16"
    if raw in _FP32_ALIASES:
        return "fp32"
    return raw


def check_fp8_compatibility(device: Optional[torch.device] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Check hardware, PyTorch, and TorchAO support for native FP8 (float8_e4m3fn).

    Returns:
        (is_supported, diagnostics_dict)
    """
    cuda_available = torch.cuda.is_available()
    target_idx = 0
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return False, {
                "fp8_supported": False,
                "cuda_available": cuda_available,
                "device_type": device.type,
                "device_name": str(device),
                "compute_capability": None,
                "fp8_hardware_supported": False,
                "fp8_torch_supported": hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm"),
                "torchao_training_supported": False,
                "torchao_inference_supported": False,
                "reason": f"Target device '{device}' is not CUDA.",
            }
        if device.index is not None:
            target_idx = device.index

    if not cuda_available:
        return False, {
            "fp8_supported": False,
            "cuda_available": False,
            "device_type": "cpu",
            "device_name": "CPU",
            "compute_capability": None,
            "fp8_hardware_supported": False,
            "fp8_torch_supported": hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm"),
            "torchao_training_supported": False,
            "torchao_inference_supported": False,
            "reason": "CUDA is not available.",
        }

    try:
        dev_name = torch.cuda.get_device_name(target_idx)
        cc = torch.cuda.get_device_capability(target_idx)
    except Exception as exc:
        return False, {
            "fp8_supported": False,
            "cuda_available": True,
            "device_name": "Unknown CUDA Device",
            "compute_capability": None,
            "fp8_hardware_supported": False,
            "fp8_torch_supported": False,
            "torchao_training_supported": False,
            "torchao_inference_supported": False,
            "reason": f"Failed to query CUDA capability: {exc}",
        }

    fp8_hardware_supported = cc >= (8, 9)
    fp8_torch_supported = (
        hasattr(torch, "float8_e4m3fn")
        and hasattr(torch, "float8_e5m2")
        and hasattr(torch, "_scaled_mm")
    )

    torchao_training_supported = False
    torchao_inference_supported = False
    torchao_version = None
    try:
        import torchao
        torchao_version = getattr(torchao, "__version__", "unknown")
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig  # noqa: F401
        torchao_training_supported = True
    except Exception:
        pass
    try:
        from torchao.quantization import (  # noqa: F401
            quantize_,
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
        )
        torchao_inference_supported = True
    except Exception:
        pass

    fp8_supported = bool(fp8_hardware_supported and fp8_torch_supported)
    reason = None
    if not fp8_hardware_supported:
        reason = f"GPU '{dev_name}' (SM {cc[0]}.{cc[1]}) does not support native FP8 E4M3 (requires SM >= 8.9)."
    elif not fp8_torch_supported:
        reason = f"PyTorch {torch.__version__} does not expose float8_e4m3fn / _scaled_mm."

    return fp8_supported, {
        "fp8_supported": fp8_supported,
        "cuda_available": True,
        "device_type": "cuda",
        "device_name": dev_name,
        "compute_capability": cc,
        "fp8_hardware_supported": fp8_hardware_supported,
        "fp8_torch_supported": fp8_torch_supported,
        "torchao_version": torchao_version,
        "torchao_training_supported": torchao_training_supported,
        "torchao_inference_supported": torchao_inference_supported,
        "reason": reason,
    }


def is_fp8_supported(device: Optional[torch.device] = None) -> bool:
    """Return True when the active/target device natively supports FP8 E4M3."""
    supported, _ = check_fp8_compatibility(device=device)
    return supported


def _get_nested(cfg: Any, section: str, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    sec = cfg.get(section, None) if isinstance(cfg, dict) else getattr(cfg, section, None)
    if sec is None:
        return default
    if isinstance(sec, dict):
        return sec.get(key, default)
    return getattr(sec, key, default) if hasattr(sec, key) else sec.get(key, default)


def resolve_precision_spec(
    config: Optional[Any] = None,
    mode: Optional[str] = None,
    device: Optional[torch.device] = None,
    fp8_all_parameters: Optional[bool] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> PrecisionSpec:
    """
    Resolve runtime precision from explicit overrides, config settings, and GPU capabilities.

    When set to 'auto' (or left unset), defaults to 'fp8' (float8_e4m3fn) whenever
    the target GPU supports SM >= 8.9 and PyTorch/TorchAO FP8 ops are available.
    """
    log = logger_instance or logger

    cfg_mode = (
        mode
        if mode is not None
        else (
            _get_nested(config, "training", "mixed_precision")
            or _get_nested(config, "training", "precision")
            or _get_nested(config, "inference", "precision")
            or "auto"
        )
    )
    requested = normalize_precision_mode(cfg_mode)
    amp_cfg = _get_nested(config, "training", "amp", True)

    if fp8_all_parameters is None:
        fp8_all_cfg = _get_nested(config, "training", "fp8_all_parameters", None)
        if fp8_all_cfg is None:
            fp8_all_cfg = _get_nested(config, "model", "fp8_all_parameters", True)
        fp8_all_parameters = bool(fp8_all_cfg)

    # Explicit amp=False with auto precision means fp32
    if requested == "auto" and amp_cfg is False:
        requested = "fp32"

    is_cuda = (device is None and torch.cuda.is_available()) or (
        isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available()
    )
    if not is_cuda:
        fallback_reason = "CPU execution: FP8/AMP disabled." if requested in ("fp8", "bf16", "fp16") else None
        if requested == "fp8":
            log.warning("FP8 precision requested, but CUDA device is not active; using float32.")
        return PrecisionSpec(
            mode="fp32",
            requested_mode=requested,
            fp8_enabled=False,
            fp8_dtype=None,
            amp_enabled=False,
            amp_dtype=torch.float32,
            amp_dtype_str="float32",
            use_grad_scaler=False,
            fp8_all_parameters=bool(fp8_all_parameters),
            fallback_reason=fallback_reason,
        )

    bf16_ok = torch.cuda.is_bf16_supported()
    fp8_ok, fp8_info = check_fp8_compatibility(device=device)

    if requested == "fp32":
        return PrecisionSpec(
            mode="fp32",
            requested_mode="fp32",
            fp8_enabled=False,
            fp8_dtype=None,
            amp_enabled=False,
            amp_dtype=torch.float32,
            amp_dtype_str="float32",
            use_grad_scaler=False,
            fp8_all_parameters=bool(fp8_all_parameters),
        )

    if requested == "bf16":
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        mode_out = "bf16" if bf16_ok else "fp16"
        return PrecisionSpec(
            mode=mode_out,
            requested_mode="bf16",
            fp8_enabled=False,
            fp8_dtype=None,
            amp_enabled=True,
            amp_dtype=amp_dtype,
            amp_dtype_str="bfloat16" if amp_dtype == torch.bfloat16 else "float16",
            use_grad_scaler=(amp_dtype == torch.float16),
            fp8_all_parameters=bool(fp8_all_parameters),
            fallback_reason=None if bf16_ok else "bfloat16 not supported; fell back to float16.",
        )

    if requested == "fp16":
        return PrecisionSpec(
            mode="fp16",
            requested_mode="fp16",
            fp8_enabled=False,
            fp8_dtype=None,
            amp_enabled=True,
            amp_dtype=torch.float16,
            amp_dtype_str="float16",
            use_grad_scaler=True,
            fp8_all_parameters=bool(fp8_all_parameters),
        )

    # requested is "auto" or "fp8"
    if fp8_ok:
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        return PrecisionSpec(
            mode="fp8",
            requested_mode=requested,
            fp8_enabled=True,
            fp8_dtype=torch.float8_e4m3fn,
            amp_enabled=True,
            amp_dtype=amp_dtype,
            amp_dtype_str="fp8_e4m3fn",
            use_grad_scaler=(amp_dtype == torch.float16),
            fp8_all_parameters=bool(fp8_all_parameters),
        )

    # Fallback when FP8 is unsupported
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    fallback_mode = "bf16" if bf16_ok else "fp16"
    reason = fp8_info.get("reason") or "FP8 not supported on this GPU."
    if requested == "fp8":
        log.warning(f"FP8 requested but unsupported ({reason}); falling back to {fallback_mode}.")
    return PrecisionSpec(
        mode=fallback_mode,
        requested_mode=requested,
        fp8_enabled=False,
        fp8_dtype=None,
        amp_enabled=True,
        amp_dtype=amp_dtype,
        amp_dtype_str="bfloat16" if amp_dtype == torch.bfloat16 else "float16",
        use_grad_scaler=(amp_dtype == torch.float16),
        fp8_all_parameters=bool(fp8_all_parameters),
        fallback_reason=reason,
    )


def _is_peft_wrapped_base_linear(fqn: str) -> bool:
    """Return True if fqn refers to a frozen `base_linear` inside a LoRA/DoRA wrapper."""
    if not fqn:
        return False
    return fqn == "base_linear" or fqn.endswith(".base_linear") or ".base_linear." in fqn


def fp8_linear_module_filter(mod: nn.Module, fqn: str = "") -> bool:
    """
    Match nn.Linear modules whose weight dimensions satisfy CUDA _scaled_mm
    16-byte alignment constraints (in_features % 16 == 0 and out_features % 16 == 0),
    excluding frozen `base_linear` children inside PEFT/DoRA wrappers so
    `base_linear.weight + delta_w` never mixes Float8Tensor subclasses with float deltas.
    """
    if _is_peft_wrapped_base_linear(fqn):
        return False
    return (
        isinstance(mod, nn.Linear)
        and (mod.in_features % 16 == 0)
        and (mod.out_features % 16 == 0)
    )


def _enable_fp8_retrieval_modules(model: nn.Module, enabled: bool) -> int:
    """Toggle FP8 retrieval mode on any module implementing set_fp8_retrieval."""
    count = 0
    for mod in model.modules():
        fn = getattr(mod, "set_fp8_retrieval", None)
        if callable(fn):
            fn(enabled)
            count += 1
    return count


def prepare_model_for_fp8_training(
    model: nn.Module,
    precision_spec: PrecisionSpec,
    pad_inner_dim: bool = True,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Convert all eligible model parameters/layers to native FP8 (float8_e4m3fn) training
    when precision_spec.fp8_enabled is True.
    """
    log = logger_instance or logger
    target = getattr(model, "module", model)

    if not precision_spec.fp8_enabled:
        _enable_fp8_retrieval_modules(target, False)
        return {
            "enabled": False,
            "mode": precision_spec.mode,
            "converted_linear_layers": 0,
            "skipped_linear_layers": 0,
            "fp8_vectordb_modules": 0,
        }

    total_linear = sum(1 for _, m in target.named_modules() if isinstance(m, nn.Linear))
    eligible_linear = sum(1 for n, m in target.named_modules() if fp8_linear_module_filter(m, n))
    already_fp8 = sum(1 for _, m in target.named_modules() if "Float8Linear" in type(m).__name__)

    converted_count = already_fp8
    if eligible_linear > 0 and already_fp8 == 0:
        try:
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig

            f8_cfg = Float8LinearConfig(pad_inner_dim=pad_inner_dim)
            convert_to_float8_training(
                target,
                module_filter_fn=fp8_linear_module_filter,
                config=f8_cfg,
            )
            converted_count = sum(1 for _, m in target.named_modules() if "Float8Linear" in type(m).__name__)
        except Exception as exc:
            log.warning(f"TorchAO convert_to_float8_training skipped due to error: {exc}")

    vectordb_count = 0
    if precision_spec.fp8_all_parameters:
        vectordb_count = _enable_fp8_retrieval_modules(target, True)

    skipped_count = max(0, total_linear - eligible_linear)
    log.info(
        f"⚡ [FP8 E4M3 TRAINING] Prepared model: {converted_count} Float8Linear layer(s) active "
        f"({skipped_count} non-16-aligned Linear layer(s) kept in {precision_spec.amp_dtype_str}), "
        f"{vectordb_count} VectorDB module(s) using FP8 retrieval."
    )
    return {
        "enabled": True,
        "mode": precision_spec.mode,
        "converted_linear_layers": converted_count,
        "skipped_linear_layers": skipped_count,
        "fp8_vectordb_modules": vectordb_count,
    }


def _has_quantized_or_lowbit_weight(module: nn.Module) -> bool:
    for param in module.parameters(recurse=False):
        cls_name = type(param).__name__
        data_cls_name = type(param.data).__name__
        if any(tag in cls_name or tag in data_cls_name for tag in ("Params4bit", "Int8Params", "Float8")):
            return True
        if param.dtype in (torch.uint8, torch.int8, getattr(torch, "float8_e4m3fn", torch.uint8)):
            return True
    return False


def _cast_floating_tensors_for_fp8_inference(model: nn.Module, compute_dtype: torch.dtype) -> None:
    """
    Safely align Linear and MultiheadAttention floating parameters to compute_dtype
    before TorchAO Float8 quantization so F.multi_head_attention_forward never
    triggers Float8Tensor.to(dtype) inside torch.amp.autocast, while keeping
    Conv2d/BatchNorm/residual filters untouched in FP32.
    """
    for mod in model.modules():
        if not isinstance(mod, (nn.Linear, nn.MultiheadAttention)):
            continue
        if getattr(mod, "_keep_fp32", False) or _has_quantized_or_lowbit_weight(mod):
            continue
        for param_name, param in list(mod._parameters.items()):
            if param is not None and param.is_floating_point() and param.dtype != compute_dtype:
                param.data = param.data.to(dtype=compute_dtype)


def prepare_model_for_fp8_inference(
    model: nn.Module,
    precision_spec: PrecisionSpec,
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Quantize all eligible model parameters to FP8 (float8_e4m3fn) for inference
    (evaluate, cross_eval, predict). Call AFTER loading checkpoint state_dict.
    """
    log = logger_instance or logger
    target = getattr(model, "module", model)

    if not precision_spec.fp8_enabled:
        _enable_fp8_retrieval_modules(target, False)
        return model, {
            "enabled": False,
            "mode": precision_spec.mode,
            "dynamic_fp8_layers": 0,
            "weight_only_fp8_layers": 0,
            "fp8_vectordb_modules": 0,
        }

    already_quantized = any(
        type(p.data).__name__ in ("Float8Tensor", "Float8TrainingTensor")
        or p.dtype == torch.float8_e4m3fn
        for p in target.parameters()
    )

    compute_dtype = precision_spec.amp_dtype
    dyn_count = 0
    wo_count = 0

    if not already_quantized:
        # Align parameters cleanly to compute_dtype so Float8Tensor + MultiheadAttention works cleanly under autocast
        _cast_floating_tensors_for_fp8_inference(target, compute_dtype)

        try:
            from torchao.quantization import (
                quantize_,
                Float8DynamicActivationFloat8WeightConfig,
                Float8WeightOnlyConfig,
            )

            def _is_standard_unquantized_linear(m: nn.Module, fqn: str = "") -> bool:
                return (
                    isinstance(m, nn.Linear)
                    and not _is_peft_wrapped_base_linear(fqn)
                    and not _has_quantized_or_lowbit_weight(m)
                    and type(m.weight.data).__name__ not in ("Float8Tensor", "Float8TrainingTensor")
                )

            eligible_dyn = [
                n for n, m in target.named_modules()
                if _is_standard_unquantized_linear(m, n) and fp8_linear_module_filter(m, n)
            ]
            if eligible_dyn:
                quantize_(
                    target,
                    Float8DynamicActivationFloat8WeightConfig(),
                    filter_fn=lambda m, fqn: _is_standard_unquantized_linear(m, fqn) and fp8_linear_module_filter(m, fqn),
                )
                dyn_count = len(eligible_dyn)

            if precision_spec.fp8_all_parameters:
                eligible_wo = [
                    n for n, m in target.named_modules()
                    if _is_standard_unquantized_linear(m, n) and not fp8_linear_module_filter(m, n)
                ]
                if eligible_wo:
                    quantize_(
                        target,
                        Float8WeightOnlyConfig(),
                        filter_fn=lambda m, fqn: _is_standard_unquantized_linear(m, fqn) and not fp8_linear_module_filter(m, fqn),
                    )
                    wo_count = len(eligible_wo)
        except Exception as exc:
            log.warning(f"TorchAO FP8 inference quantization skipped due to error: {exc}")

    vectordb_count = 0
    if precision_spec.fp8_all_parameters:
        vectordb_count = _enable_fp8_retrieval_modules(target, True)

    log.info(
        f"⚡ [FP8 E4M3 INFERENCE] Quantized model weights: {dyn_count} Linear (dynamic E4M3 act+wt), "
        f"{wo_count} Linear (E4M3 weight-only for non-16-aligned shapes), "
        f"{vectordb_count} VectorDB module(s) using FP8 retrieval."
    )
    return model, {
        "enabled": True,
        "mode": precision_spec.mode,
        "dynamic_fp8_layers": dyn_count,
        "weight_only_fp8_layers": wo_count,
        "fp8_vectordb_modules": vectordb_count,
    }
