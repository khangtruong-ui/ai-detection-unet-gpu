"""Process cleanup and termination utility for background sid_unet and parquet tasks.

Provides:
- find_unet_processes(): Discovers all background, orphaned, or running unet tasks.
- kill_all_background_tasks(): Gracefully terminates and cleans up unet and dataset processes.
- cli_main(): CLI entrypoint for 'sid-kill'.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sid_unet.kill")

# Exact or prefix binary / script names that indicate sid_unet tasks
UNET_CLI_NAMES = {
    "sid-train",
    "sid-eval",
    "sid-cross-eval",
    "sid-predict",
    "sid-illu",
    "sid-check-8bit",
    "sid-push",
    "sid-pull",
    "sid-checkpoint",
    "sid-cache",
    "sid-dataset-cache",
    "pdl-kill",
}

# Substrings in cmdline indicating sid_unet or related tasks
UNET_CMDLINE_SUBSTRINGS = [
    "sid_unet",
    "torch.distributed.run",
    "torchrun",
    "parquet_dataset_loader",
]

# System or critical process names that should NEVER be touched
PROTECTED_PROCESS_NAMES = {
    "systemd",
    "init",
    "sshd",
    "bash",
    "sh",
    "zsh",
    "tmux",
    "screen",
    "agy",
    "portal",
    "caddy",
    "syncthing",
    "cron",
    "tclsh",
    "tclsh8.6",
    "unbuffer",
    "supervisord",
    "supervisorctl",
}


def get_ancestor_pids() -> Set[int]:
    """Collect all ancestor PIDs of the current process up to root."""
    ancestors: Set[int] = set()
    current = os.getpid()
    ancestors.add(current)
    try:
        import psutil
        p = psutil.Process(current)
        for parent in p.parents():
            ancestors.add(parent.pid)
    except Exception:
        # Fallback reading /proc/<pid>/stat
        pid = current
        while pid > 1:
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat_parts = f.read().split()
                    ppid = int(stat_parts[3])
                    if ppid <= 1 or ppid in ancestors:
                        break
                    ancestors.add(ppid)
                    pid = ppid
            except Exception:
                break
    return ancestors


def is_unet_process(pid: int, cmdline_str: str, exe_str: str, cwd_str: str) -> bool:
    """Determine whether a process is related to sid_unet or parquet_dataset_loader."""
    cmd_lower = cmdline_str.lower()
    exe_name = os.path.basename(exe_str).lower() if exe_str else ""

    # Check for direct CLI names
    for cli_name in UNET_CLI_NAMES:
        if cli_name in cmd_lower or exe_name == cli_name:
            return True

    # Check for sid_unet module or script execution
    for sub in UNET_CMDLINE_SUBSTRINGS:
        if sub in cmd_lower:
            return True

    # Check if process is running python within the repo cwd
    if "python" in exe_name or "python" in cmd_lower:
        if "/workspace/ai-detection-unet-gpu" in cwd_str:
            # Exclude current script or pytest if invoked from test runner
            if "sid-kill" not in cmd_lower and "pytest" not in cmd_lower:
                return True

    return False


def find_unet_processes(
    include_children: bool = True,
) -> List[Dict[str, Any]]:
    """Scan the system for running or background processes associated with sid_unet.

    Returns a list of dicts with process information (pid, ppid, name, cmdline, rss_mb).
    """
    ancestors = get_ancestor_pids()
    target_procs: Dict[int, Dict[str, Any]] = {}
    all_children_map: Dict[int, List[int]] = {}

    try:
        import psutil

        for proc in psutil.process_iter(
            ["pid", "ppid", "name", "cmdline", "exe", "cwd", "memory_info"]
        ):
            try:
                pid = proc.info["pid"]
                if pid in ancestors or pid <= 1:
                    continue

                name = proc.info.get("name") or ""
                if name.lower() in PROTECTED_PROCESS_NAMES:
                    continue

                cmd_list = proc.info.get("cmdline") or []
                cmdline = " ".join(cmd_list)
                exe = proc.info.get("exe") or ""
                cwd = proc.info.get("cwd") or ""
                ppid = proc.info.get("ppid") or 0

                # Track parent-child tree
                if ppid:
                    all_children_map.setdefault(ppid, []).append(pid)

                # Check if it matches unet tasks
                if is_unet_process(pid, cmdline, exe, cwd):
                    mem_info = proc.info.get("memory_info")
                    rss_mb = (mem_info.rss / (1024 * 1024)) if mem_info else 0.0
                    target_procs[pid] = {
                        "pid": pid,
                        "ppid": ppid,
                        "name": name,
                        "cmdline": cmdline,
                        "rss_mb": rss_mb,
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        # Stdlib /proc fallback
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in ancestors or pid <= 1:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd_bytes = f.read()
                    cmdline = cmd_bytes.replace(b"\0", b" ").decode(
                        "utf-8", errors="replace"
                    ).strip()
                exe = ""
                try:
                    exe = os.readlink(f"/proc/{pid}/exe")
                except OSError:
                    pass
                cwd = ""
                try:
                    cwd = os.readlink(f"/proc/{pid}/cwd")
                except OSError:
                    pass
                ppid = 0
                try:
                    with open(f"/proc/{pid}/stat", "r") as f:
                        ppid = int(f.read().split()[3])
                except Exception:
                    pass

                if ppid:
                    all_children_map.setdefault(ppid, []).append(pid)

                if is_unet_process(pid, cmdline, exe, cwd):
                    target_procs[pid] = {
                        "pid": pid,
                        "ppid": ppid,
                        "name": os.path.basename(exe) if exe else "python",
                        "cmdline": cmdline,
                        "rss_mb": 0.0,
                    }
            except (OSError, IOError):
                continue

    # Include descendants (DataLoader workers, etc.)
    if include_children:
        to_expand = list(target_procs.keys())
        visited: Set[int] = set(to_expand)
        while to_expand:
            curr = to_expand.pop()
            for child_pid in all_children_map.get(curr, []):
                if child_pid not in visited and child_pid not in ancestors:
                    visited.add(child_pid)
                    to_expand.append(child_pid)
                    if child_pid not in target_procs:
                        try:
                            import psutil
                            cp = psutil.Process(child_pid)
                            cmdline = " ".join(cp.cmdline())
                            mem = cp.memory_info()
                            target_procs[child_pid] = {
                                "pid": child_pid,
                                "ppid": curr,
                                "name": cp.name(),
                                "cmdline": cmdline,
                                "rss_mb": mem.rss / (1024 * 1024) if mem else 0.0,
                            }
                        except Exception:
                            target_procs[child_pid] = {
                                "pid": child_pid,
                                "ppid": curr,
                                "name": "worker",
                                "cmdline": "<child worker>",
                                "rss_mb": 0.0,
                            }

    return sorted(target_procs.values(), key=lambda x: x["pid"])


def get_gpu_memory_summary() -> List[Dict[str, Any]]:
    """Retrieve current GPU VRAM utilization if CUDA is available."""
    gpus: List[Dict[str, Any]] = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    free_mb = free / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    used_mb = total_mb - free_mb
                    name = torch.cuda.get_device_name(i)
                    gpus.append({
                        "index": i,
                        "name": name,
                        "used_mb": used_mb,
                        "total_mb": total_mb,
                        "free_mb": free_mb,
                    })
                except Exception:
                    pass
    except Exception:
        pass
    return gpus


def kill_all_background_tasks(
    force: bool = False,
    timeout: float = 2.0,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> int:
    """Find and kill all background and active tasks related to sid_unet.

    Args:
        force: If True, immediately send SIGKILL without waiting for SIGTERM.
        timeout: Grace period in seconds to wait for SIGTERM before sending SIGKILL.
        dry_run: If True, only discover and report tasks without terminating them.
        verbose: If True, output full command lines and extra diagnostics.
        quiet: If True, suppress console output.

    Returns:
        Number of terminated processes.
    """
    if not quiet:
        print("🔍 Searching for active and background tasks related to sid_unet...")

    procs = find_unet_processes(include_children=True)

    if not procs:
        if not quiet:
            print("✨ No background or running sid_unet tasks found.")
        # Still attempt pdl cleanup and GPU empty cache
        _cleanup_libraries(quiet=quiet)
        return 0

    if not quiet:
        print(f"🎯 Found {len(procs)} matching process(es):")
        for p in procs:
            cmd_preview = p["cmdline"]
            if not verbose and len(cmd_preview) > 90:
                cmd_preview = cmd_preview[:87] + "..."
            rss_str = f" ({p['rss_mb']:.1f} MB RAM)" if p.get("rss_mb") else ""
            print(f"   • PID {p['pid']:<6} [PPID {p['ppid']:<6}] {p['name']}{rss_str}: {cmd_preview}")

    if dry_run:
        if not quiet:
            print("🧪 Dry-run mode: no processes were killed.")
        return len(procs)

    pids_to_kill = [p["pid"] for p in procs]

    # Step 1: Send initial signal (SIGKILL if force, else graceful SIGINT + SIGTERM)
    if force:
        if not quiet:
            print(f"🛑 Sending SIGKILL to {len(pids_to_kill)} process(es)...")
        for pid in pids_to_kill:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as e:
                if not quiet:
                    print(f"   ⚠️ Permission denied killing PID {pid}: {e}")
    else:
        if not quiet:
            print(f"🛑 Sending graceful termination signal (SIGINT/SIGTERM) to {len(pids_to_kill)} process(es)...")
        for pid in pids_to_kill:
            # Send SIGINT first: tasks shielded against SSH disconnect (like sid-train) ignore SIGTERM
            # but gracefully terminate on SIGINT. Also send SIGTERM for processes listening only to SIGTERM.
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    break
                except PermissionError as e:
                    if not quiet:
                        print(f"   ⚠️ Permission denied killing PID {pid}: {e}")
                    break

    # Step 2: Wait for processes to exit if graceful signals were sent
    if not force:
        start_t = time.time()
        while time.time() - start_t < timeout:
            alive = [p for p in pids_to_kill if _is_pid_alive(p)]
            if not alive:
                break
            time.sleep(0.1)

        # Step 3: Escalate stubborn processes to SIGKILL
        still_alive = [p for p in pids_to_kill if _is_pid_alive(p)]
        if still_alive:
            if not quiet:
                print(f"⚡ Force killing {len(still_alive)} remaining process(es) with SIGKILL...")
            for pid in still_alive:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    # Step 4: Cleanup library memory & dataset threads
    _cleanup_libraries(quiet=quiet)

    # Step 5: Report status and GPU memory
    time.sleep(0.2)
    final_alive = [p for p in pids_to_kill if _is_pid_alive(p)]
    killed_count = len(pids_to_kill) - len(final_alive)

    if not quiet:
        if final_alive:
            print(f"⚠️ Successfully killed {killed_count} process(es); {len(final_alive)} still alive: {final_alive}")
        else:
            print(f"✅ Successfully killed all {killed_count} background task(s).")

        gpu_info = get_gpu_memory_summary()
        if gpu_info:
            print("🖥️ Current GPU Memory Status:")
            for g in gpu_info:
                print(
                    f"   • GPU {g['index']} ({g['name']}): "
                    f"{g['used_mb']:.1f} MB used / {g['total_mb']:.1f} MB total ({g['free_mb']:.1f} MB free)"
                )

    return killed_count


def _is_pid_alive(pid: int) -> bool:
    """Check if a process PID is currently alive and running (not a zombie)."""
    try:
        import psutil
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" not in line
        return True
    except (ProcessLookupError, FileNotFoundError):
        return False
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _cleanup_libraries(quiet: bool = False) -> None:
    """Invoke parquet-dataset-loader cleanup and PyTorch CUDA cache clearing."""
    # 1. Parquet Dataset Loader background worker and manager cleanup
    try:
        import parquet_dataset_loader as pdl
        if hasattr(pdl, "cleanup_background_tasks"):
            pdl.cleanup_background_tasks()
        elif hasattr(pdl, "close_all_datasets"):
            pdl.close_all_datasets()
    except Exception:
        pass

    # 2. PyTorch CUDA cache flush
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def cli_main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for sid-kill."""
    parser = argparse.ArgumentParser(
        prog="sid-kill",
        description="Terminate all background, orphaned, or active processes related to sid_unet.",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Immediately terminate processes with SIGKILL without waiting for graceful shutdown.",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Display matching processes without terminating them.",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=2.0,
        help="Grace period in seconds to wait for SIGTERM before sending SIGKILL (default: 2.0).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose diagnostics including full command lines.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output.",
    )
    args = parser.parse_args(argv)

    kill_all_background_tasks(
        force=args.force,
        timeout=args.timeout,
        dry_run=args.dry_run,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
