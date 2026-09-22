"""
8-bit hardware and library compatibility checker for SID-UNet.
Verifies CUDA availability, GPU Compute Capability, bitsandbytes installation,
native FP8 support, and 8-bit optimizer availability for training and installation.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional, Tuple, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger("sid_unet.utils.compatibility")


def check_8bit_compatibility(
    device: Optional[Union[str, Any]] = None,
    verbose: bool = False,
    custom_logger: Optional[Any] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if the current hardware and software environment supports 8-bit training.

    Evaluates:
      1. CUDA availability.
      2. GPU architecture and Compute Capability (SM >= 7.0 for bitsandbytes, SM >= 8.9 for FP8).
      3. bitsandbytes installation and functional CUDA kernel execution.
      4. Support for 8-bit optimizers (AdamW8bit, PagedAdamW8bit).
      5. Support for 8-bit base model quantization (bitsandbytes load_in_8bit).

    Args:
        device: Target device to check (default: first available CUDA device or 'cpu').
        verbose: If True, logs the compatibility diagnostic report.
        custom_logger: Logger instance to use (defaults to module logger).

    Returns:
        Tuple of (is_compatible: bool, details_dict: dict)
    """
    log = custom_logger or logger
    cuda_available = torch.cuda.is_available() if torch is not None else False
    device_name = "None"
    compute_capability = (0, 0)
    device_index = 0
    fp8_hardware_supported = False
    fp8_torch_supported = (hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")) if torch is not None else False

    if cuda_available and torch is not None:
        try:
            if device is not None:
                dev = torch.device(device)
                device_index = dev.index if dev.index is not None else 0
            else:
                device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            compute_capability = torch.cuda.get_device_capability(device_index)
            # Ada Lovelace (SM 8.9), Hopper (SM 9.0), Blackwell (SM 10.0/12.0) support FP8 natively
            fp8_hardware_supported = compute_capability >= (8, 9)
        except Exception as e:
            log.debug(f"Failed to query CUDA device capabilities: {e}")

    # Check bitsandbytes library
    bnb_installed = False
    bnb_version = "None"
    bnb_functional = False
    bnb_error_msg = None

    try:
        import bitsandbytes as bnb
        bnb_installed = True
        bnb_version = getattr(bnb, "__version__", "unknown")

        if cuda_available:
            # Functional test: verify that bitsandbytes CUDA kernel runs without failure
            try:
                dummy_param = torch.nn.Parameter(torch.randn(2, 2, device=f"cuda:{device_index}"))
                test_opt = bnb.optim.AdamW8bit([dummy_param], lr=1e-3)
                dummy_param.grad = torch.ones_like(dummy_param)
                test_opt.step()
                del test_opt, dummy_param
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bnb_functional = True
            except Exception as fn_err:
                bnb_functional = False
                bnb_error_msg = str(fn_err)
        else:
            bnb_functional = False
            bnb_error_msg = "CUDA is not available; bitsandbytes requires an NVIDIA GPU for 8-bit operations."
    except ImportError as imp_err:
        bnb_installed = False
        bnb_error_msg = (
            "bitsandbytes is not installed. Install via `pip install 'sid-unet[8bit]'` or `pip install bitsandbytes`."
        )
    except Exception as e:
        bnb_installed = False
        bnb_error_msg = str(e)

    # Determine overall 8-bit compatibility
    # Modern bitsandbytes requires compute capability >= 7.0 for optimal performance
    cc_ok = compute_capability >= (7, 0)
    optimizer_8bit_supported = cuda_available and bnb_installed and bnb_functional and cc_ok
    model_8bit_supported = cuda_available and bnb_installed and bnb_functional
    fp8_native_supported = cuda_available and fp8_hardware_supported and fp8_torch_supported

    # Determine fallback target mode (explicitly GPU 16-bit mode when on CUDA)
    bf16_ok = cuda_available and torch.cuda.is_bf16_supported()
    fallback_precision = "BF16" if bf16_ok else "FP16"
    fallback_target = f"GPU 16-Bit Mode (AMP {fallback_precision} + AdamW)" if cuda_available else "CPU 32-Bit Mode"
    overall_compatible = optimizer_8bit_supported

    # Formulate human-readable message
    if overall_compatible:
        msg = (
            f"8-bit training is fully supported on {device_name} (SM {compute_capability[0]}.{compute_capability[1]}) "
            f"with bitsandbytes {bnb_version}."
        )
        if fp8_native_supported:
            msg += " Hardware native FP8 compute (Ada/Hopper) is also available."
    else:
        reasons = []
        if not cuda_available:
            reasons.append("CUDA is not available (CPU-only detected)")
        elif not cc_ok:
            reasons.append(
                f"GPU Compute Capability {compute_capability[0]}.{compute_capability[1]} is below minimum recommended SM 7.0"
            )
        if not bnb_installed:
            reasons.append("bitsandbytes is not installed (`pip install 'sid-unet[8bit]'`)")
        elif not bnb_functional:
            reasons.append(f"bitsandbytes CUDA kernel initialization failed: {bnb_error_msg}")
        msg = f"8-bit training mode is unavailable: {'; '.join(reasons)}. Fallback will use {fallback_target}."

    details: Dict[str, Any] = {
        "compatible": overall_compatible,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "device_index": device_index,
        "compute_capability": compute_capability,
        "compute_capability_str": f"SM {compute_capability[0]}.{compute_capability[1]}",
        "bitsandbytes_installed": bnb_installed,
        "bitsandbytes_version": bnb_version,
        "bitsandbytes_functional": bnb_functional,
        "bitsandbytes_error": bnb_error_msg,
        "optimizer_8bit_supported": optimizer_8bit_supported,
        "model_8bit_supported": model_8bit_supported,
        "fp8_hardware_supported": fp8_hardware_supported,
        "fp8_torch_supported": fp8_torch_supported,
        "fp8_native_supported": fp8_native_supported,
        "gpu_16bit_supported": cuda_available,
        "gpu_16bit_bf16": bf16_ok,
        "fallback_target": fallback_target,
        "supported_optimizers": (
            ["adamw8bit", "paged_adamw8bit", "adam8bit", "paged_adam8bit"] if optimizer_8bit_supported else []
        ),
        "message": msg,
    }

    if verbose:
        table_str = format_compatibility_table(details)
        log.info(table_str)

    return overall_compatible, details


def format_compatibility_table(details: Dict[str, Any]) -> str:
    """Format compatibility check results into an ASCII status table."""
    status_icon = "✅" if details.get("compatible", False) else "⚠️"
    cuda_status = "Available" if details.get("cuda_available") else "Not Available"
    bnb_status = f"Installed (v{details.get('bitsandbytes_version')})" if details.get("bitsandbytes_installed") else "Not Installed"
    if details.get("bitsandbytes_installed") and not details.get("bitsandbytes_functional"):
        bnb_status += " [CUDA Init Failed]"
    fp8_status = "Supported" if details.get("fp8_native_supported") else "Not Supported"
    opt_status = "Supported (AdamW8bit, PagedAdamW8bit)" if details.get("optimizer_8bit_supported") else "Not Supported"
    fallback_str = details.get("fallback_target", "GPU 16-Bit Mode (AMP FP16/BF16)")

    lines = [
        "================================================================================",
        f" {status_icon} SID-UNet 8-Bit Training Environment Compatibility",
        "--------------------------------------------------------------------------------",
        f"  CUDA GPU Hardware   : {cuda_status} ({details.get('device_name')})",
        f"  Compute Capability  : {details.get('compute_capability_str')}",
        f"  bitsandbytes Library: {bnb_status}",
        f"  8-Bit Optimizers    : {opt_status}",
        f"  FP8 Native Compute  : {fp8_status}",
        f"  Fallback Mode       : {fallback_str}",
        "--------------------------------------------------------------------------------",
        f"  Summary: {details.get('message')}",
        "================================================================================",
    ]
    return "\n".join(lines)


def validate_8bit_environment(
    device: Optional[Union[str, Any]] = None,
    raise_error: bool = False,
    custom_logger: Optional[Any] = None,
) -> bool:
    """
    Validate that the environment can execute 8-bit training.

    Args:
        device: Device to check.
        raise_error: If True, raises RuntimeError when 8-bit is incompatible.
        custom_logger: Logger for warnings/errors.

    Returns:
        True if 8-bit training is supported, False otherwise.
    """
    is_ok, details = check_8bit_compatibility(device=device, verbose=False, custom_logger=custom_logger)
    if not is_ok and raise_error:
        raise RuntimeError(
            f"8-bit training mode requested but the environment is not compatible: {details.get('message')}"
        )
    return is_ok


def installation_check_8bit() -> bool:
    """
    Non-destructive compatibility check intended for package installation time.
    Prints diagnostics to stdout and returns compatibility status.
    """
    try:
        is_compatible, details = check_8bit_compatibility(verbose=False)
        print("\n" + format_compatibility_table(details) + "\n")
        return is_compatible
    except Exception as exc:
        print(f"\n[sid-unet] Notice: 8-bit compatibility check encountered non-fatal note: {exc}\n")
        return False


def cli_check_8bit():
    """CLI entrypoint for checking 8-bit hardware and library compatibility."""
    import argparse

    parser = argparse.ArgumentParser(description="SID-UNet 8-Bit Compatibility Checker")
    parser.add_argument("--device", type=str, default=None, help="Target device (e.g. 'cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if 8-bit is unsupported")
    args = parser.parse_args()

    compatible, details = check_8bit_compatibility(device=args.device, verbose=False)
    print(format_compatibility_table(details))

    if args.strict and not compatible:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    cli_check_8bit()
