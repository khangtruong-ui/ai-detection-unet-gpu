"""
Distributed training utilities for SID-UNet.
Supports high-efficiency DistributedDataParallel (DDP) across multiple GPUs
with zero GIL contention, NCCL collective communications, and rank-0 coordination.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Dict, Optional, Tuple
import torch
import torch.distributed as dist


def find_free_port() -> int:
    """Find and return an available free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def is_dist_avail_and_initialized() -> bool:
    """Check if PyTorch distributed package is available and process group is initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Return the total number of processes in the distributed process group."""
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Return the global rank of the current process."""
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Return the local device rank of the current process within the current node."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Return True if current process is rank 0 (the primary coordination process)."""
    return get_rank() == 0


def init_distributed_mode(backend: str = "nccl") -> Tuple[int, int, int]:
    """
    Initialize distributed process group if environment variables are detected.
    Supports torchrun and multi-process launchers setting RANK, LOCAL_RANK, WORLD_SIZE.
    Returns (rank, local_rank, world_size).
    """
    if is_dist_avail_and_initialized():
        return get_rank(), get_local_rank(), get_world_size()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        return 0, 0, 1

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if local_rank >= num_devices:
            local_rank = local_rank % num_devices
        torch.cuda.set_device(local_rank)
        chosen_backend = backend if dist.is_nccl_available() else "gloo"
    else:
        chosen_backend = "gloo"

    if not is_dist_avail_and_initialized():
        dist.init_process_group(
            backend=chosen_backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Cleanly destroy distributed process group if active."""
    if is_dist_avail_and_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all distributed processes using all_reduce.
    If average is True, divides by world_size.
    """
    if not is_dist_avail_and_initialized():
        return tensor

    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if average:
        reduced = reduced / get_world_size()
    return reduced


def reduce_dict(metrics: Dict[str, float], average: bool = True) -> Dict[str, float]:
    """
    Synchronize and reduce a dictionary of float scalar metrics across all distributed ranks.
    """
    if not is_dist_avail_and_initialized() or not metrics:
        return metrics

    keys = sorted(metrics.keys())
    local_vals = [float(metrics[k]) for k in keys]
    device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    t = torch.tensor(local_vals, dtype=torch.float64, device=device)

    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if average:
        t = t / get_world_size()

    out_vals = t.cpu().tolist()
    return {k: float(v) for k, v in zip(keys, out_vals)}


def broadcast_scalar(val: float, src: int = 0) -> float:
    """Broadcast a float scalar from src rank to all ranks."""
    if not is_dist_avail_and_initialized():
        return val

    device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([val], dtype=torch.float64, device=device)
    dist.broadcast(t, src=src)
    return float(t.item())


def sync_scalar_min(val: int | float) -> int | float:
    """Synchronize a scalar across all ranks taking the minimum."""
    if not is_dist_avail_and_initialized():
        return val

    device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    is_int = isinstance(val, int)
    dtype = torch.long if is_int else torch.float64
    t = torch.tensor([val], dtype=dtype, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item()) if is_int else float(t.item())


def sync_scalar_max(val: int | float) -> int | float:
    """Synchronize a scalar across all ranks taking the maximum."""
    if not is_dist_avail_and_initialized():
        return val

    device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    is_int = isinstance(val, int)
    dtype = torch.long if is_int else torch.float64
    t = torch.tensor([val], dtype=dtype, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item()) if is_int else float(t.item())

