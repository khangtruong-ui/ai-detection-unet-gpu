"""
Process management, tracking, and lifecycle registry for sid_unet background tasks.

Provides:
- ProcessRecord: Structured record of an active or terminated sid_unet task.
- ProcessRegistry: Thread-safe, process-safe ledger for discovering, tracking, and updating tasks.
- inspect_log_for_termination(): Analyzes output logs to identify clean completion, exceptions, or kill events.
- terminate_process(): Gracefully or forcefully terminates a single registered task.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from filelock import FileLock

logger = logging.getLogger("sid_unet.process_manager")

DEFAULT_REGISTRY_DIR = ".sid_processes"
DEFAULT_REGISTRY_FILE = "registry.json"
DEFAULT_LOCK_FILE = "registry.lock"

# Status constants
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_KILLED = "KILLED"
STATUS_EXCEPTION = "EXCEPTION"
STATUS_UNKNOWN = "UNKNOWN"

# Regex patterns for Python stack traces and common termination errors
TRACEBACK_PATTERN = re.compile(
    r"(Traceback \(most recent call last\):[\s\S]*?(?=\n\S|\Z))",
    re.MULTILINE,
)
EXCEPTION_LINE_PATTERN = re.compile(
    r"^([A-Za-z0-9_.]+(?:Error|Exception|Interrupt|Exit|Failure|Warning)):\s*(.*)$",
    re.MULTILINE,
)
COMPLETION_PATTERNS = [
    re.compile(r"✅.*completed successfully", re.IGNORECASE),
    re.compile(r"✅.*Cross-evaluation matrix completed", re.IGNORECASE),
    re.compile(r"Finished all runs", re.IGNORECASE),
    re.compile(r"Training finished", re.IGNORECASE),
    re.compile(r"All shards processed", re.IGNORECASE),
]


@dataclass
class ProcessRecord:
    """Represents a tracked sid_unet process, whether active or terminated."""

    pid: int
    name: str
    cmdline: str
    cwd: str
    log_file: str
    start_time: str
    end_time: Optional[str] = None
    status: str = STATUS_RUNNING
    exit_code: Optional[int] = None
    termination_reason: Optional[str] = None
    error_summary: Optional[str] = None
    log_offset_start: int = 0
    rss_mb: float = 0.0
    gpu_info: Optional[str] = None
    ppid: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessRecord":
        fields_set = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in fields_set}
        return cls(**filtered)

    @property
    def is_running(self) -> bool:
        return self.status == STATUS_RUNNING

    @property
    def duration_seconds(self) -> float:
        """Calculate elapsed or total runtime in seconds."""
        try:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time) if self.end_time else datetime.now()
            return max(0.0, (end_dt - start_dt).total_seconds())
        except Exception:
            return 0.0

    @property
    def duration_str(self) -> str:
        """Formatted duration string (e.g. '01m 23s' or '02h 15m')."""
        secs = int(self.duration_seconds)
        if secs < 60:
            return f"{secs}s"
        mins = secs // 60
        rem_secs = secs % 60
        if mins < 60:
            return f"{mins:02d}m {rem_secs:02d}s"
        hours = mins // 60
        rem_mins = mins % 60
        return f"{hours:02d}h {rem_mins:02d}m"


def inspect_log_for_termination(
    log_path: str,
    start_offset: int = 0,
    max_scan_bytes: int = 256 * 1024,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Analyze a process log file to infer termination status and error details.

    Args:
        log_path: Path to the log file.
        start_offset: Byte offset where the target session started in the file.
        max_scan_bytes: Maximum number of bytes to read from the end of the file.

    Returns:
        Tuple of (status, termination_reason, error_summary).
    """
    if not os.path.isfile(log_path):
        return STATUS_UNKNOWN, "Log file not found", None

    try:
        file_size = os.path.getsize(log_path)
        if file_size <= start_offset:
            return STATUS_KILLED, "Log terminated abruptly before writing output", None

        # Read relevant section
        with open(log_path, "rb") as f:
            read_start = max(start_offset, file_size - max_scan_bytes)
            f.seek(read_start)
            raw_bytes = f.read()

        text = raw_bytes.decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # 1. Check for unhandled exceptions and tracebacks
        tb_idx = text.rfind("Traceback (most recent call last):")
        if tb_idx != -1:
            tb_text = text[tb_idx:]
            tb_lines = [l.strip() for l in tb_text.splitlines() if l.strip()]
            # Scan from bottom for the exception line
            exc_line = tb_lines[-1] if len(tb_lines) > 1 else "Unknown Exception"
            for line in reversed(tb_lines[1:]):
                if re.match(r"^[A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit|Failure|Warning):?.*", line):
                    exc_line = line
                    break

            if "KeyboardInterrupt" in exc_line:
                return (
                    STATUS_KILLED,
                    "Process interrupted by user (KeyboardInterrupt / SIGINT)",
                    exc_line,
                )
            if "OutOfMemoryError" in exc_line or "CUDA out of memory" in exc_line:
                return (
                    STATUS_EXCEPTION,
                    f"CUDA Out Of Memory: {exc_line}",
                    exc_line,
                )

            return STATUS_EXCEPTION, f"Exception raised: {exc_line}", exc_line

        # 2. Check for explicit completion markers
        for line in reversed(lines[-25:]):
            for pattern in COMPLETION_PATTERNS:
                if pattern.search(line):
                    return STATUS_COMPLETED, f"Completed successfully: {line}", None

        # 3. Check for SIGKILL, Segmentation fault, or abrupt termination
        for line in reversed(lines[-15:]):
            if "Segmentation fault" in line:
                return STATUS_EXCEPTION, "Segmentation fault (core dumped)", line
            if "Killed" in line:
                return STATUS_KILLED, "Killed by OS (OOM Killer / SIGKILL)", line

        # If ended without completion marker or exception, classify as killed/interrupted
        return (
            STATUS_KILLED,
            "Terminated abruptly or killed without clean exit marker",
            None,
        )

    except Exception as e:
        logger.debug(f"Error inspecting log {log_path}: {e}")
        return STATUS_UNKNOWN, f"Could not inspect log: {e}", None


class ProcessRegistry:
    """Thread-safe and process-safe registry managing background sid_unet processes."""

    def __init__(self, registry_dir: Optional[str] = None):
        self.registry_dir = os.path.abspath(registry_dir or DEFAULT_REGISTRY_DIR)
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_file = os.path.join(self.registry_dir, DEFAULT_REGISTRY_FILE)
        self.lock_file = os.path.join(self.registry_dir, DEFAULT_LOCK_FILE)

    def _get_lock(self, timeout: float = 5.0) -> FileLock:
        return FileLock(self.lock_file, timeout=timeout)

    def _read_records_unlocked(self) -> Dict[str, ProcessRecord]:
        if not os.path.isfile(self.registry_file):
            return {}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            records = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    try:
                        records[k] = ProcessRecord.from_dict(v)
                    except Exception:
                        pass
            return records
        except Exception as e:
            logger.debug(f"Failed to read registry: {e}")
            return {}

    def _write_records_unlocked(self, records: Dict[str, ProcessRecord]) -> None:
        data = {k: v.to_dict() for k, v in records.items()}
        temp_file = f"{self.registry_file}.tmp.{os.getpid()}"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, self.registry_file)

    def register_process(
        self,
        pid: int,
        name: str,
        cmdline: str,
        log_file: str,
        cwd: Optional[str] = None,
        ppid: Optional[int] = None,
        log_offset_start: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessRecord:
        """Register a newly launched sid process into the ledger."""
        record = ProcessRecord(
            pid=int(pid),
            name=str(name),
            cmdline=str(cmdline),
            cwd=os.path.abspath(cwd or os.getcwd()),
            log_file=os.path.abspath(log_file),
            start_time=datetime.now().isoformat(),
            status=STATUS_RUNNING,
            ppid=int(ppid or os.getpid()),
            log_offset_start=int(log_offset_start),
            metadata=dict(metadata or {}),
        )

        with self._get_lock():
            records = self._read_records_unlocked()
            records[str(pid)] = record
            self._write_records_unlocked(records)

        logger.info(f"Registered sid process {record.name} (PID: {record.pid}) in ledger.")
        return record

    def update_process(
        self,
        pid: int,
        **kwargs: Any,
    ) -> Optional[ProcessRecord]:
        """Update fields on an existing process record."""
        key = str(pid)
        with self._get_lock():
            records = self._read_records_unlocked()
            if key not in records:
                return None
            record = records[key]
            for field_name, value in kwargs.items():
                if hasattr(record, field_name):
                    setattr(record, field_name, value)
            records[key] = record
            self._write_records_unlocked(records)
            return record

    def mark_process_terminated(
        self,
        pid: int,
        status: str,
        reason: Optional[str] = None,
        exit_code: Optional[int] = None,
        error_summary: Optional[str] = None,
    ) -> Optional[ProcessRecord]:
        """Mark a process as terminated with full status metadata."""
        now_str = datetime.now().isoformat()
        return self.update_process(
            pid,
            status=status,
            end_time=now_str,
            termination_reason=reason,
            exit_code=exit_code,
            error_summary=error_summary,
        )

    def get_process(self, pid: int) -> Optional[ProcessRecord]:
        """Retrieve record for a specific PID."""
        with self._get_lock():
            records = self._read_records_unlocked()
            return records.get(str(pid))

    def list_processes(
        self,
        status_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProcessRecord]:
        """List all tracked processes, optionally filtered by status, sorted latest first."""
        with self._get_lock():
            records = self._read_records_unlocked()

        result = list(records.values())
        if status_filter:
            sf_upper = status_filter.upper()
            result = [r for r in result if r.status.upper() == sf_upper]

        result.sort(key=lambda r: r.start_time, reverse=True)
        if limit and limit > 0:
            result = result[:limit]
        return result

    def clear_history(self, keep_running: bool = True) -> int:
        """Remove terminated records from ledger. Returns count of deleted entries."""
        with self._get_lock():
            records = self._read_records_unlocked()
            to_delete = [
                k for k, r in records.items()
                if not (keep_running and r.status == STATUS_RUNNING)
            ]
            for k in to_delete:
                del records[k]
            self._write_records_unlocked(records)
            return len(to_delete)

    def reconcile(self) -> List[ProcessRecord]:
        """Reconcile ledger against live operating system processes.

        - Verifies if running processes are still alive.
        - Discovers external or unrecorded sid_unet background tasks.
        - Automatically detects exceptions, terminations, or completions for dead tasks.
        - Refreshes RSS memory and GPU metrics.

        Returns:
            List of all updated ProcessRecord objects.
        """
        from sid_unet.kill import _is_pid_alive, find_unet_processes

        # 1. Discover currently alive sid processes from system
        live_os_procs = {p["pid"]: p for p in find_unet_processes(include_children=False)}

        with self._get_lock():
            records = self._read_records_unlocked()
            changed = False

            # Check existing records
            for key, rec in list(records.items()):
                pid = rec.pid
                if rec.status == STATUS_RUNNING:
                    is_alive = _is_pid_alive(pid)
                    if is_alive:
                        # Update live RAM stats if available
                        if pid in live_os_procs:
                            os_p = live_os_procs[pid]
                            new_rss = float(os_p.get("rss_mb", 0.0))
                            if abs(new_rss - rec.rss_mb) > 0.5:
                                rec.rss_mb = new_rss
                                changed = True
                    else:
                        # Process has died: inspect logs to diagnose termination
                        status, reason, err_sum = inspect_log_for_termination(
                            rec.log_file,
                            start_offset=rec.log_offset_start,
                        )
                        rec.status = status
                        rec.termination_reason = rec.termination_reason or reason
                        rec.error_summary = rec.error_summary or err_sum
                        rec.end_time = rec.end_time or datetime.now().isoformat()
                        changed = True

            # Discover unrecorded live sid processes
            for pid, os_p in live_os_procs.items():
                if str(pid) not in records:
                    cmd_line = os_p.get("cmdline", "")
                    name = os_p.get("name", "sid-task")
                    # Match name against UNET commands
                    for part in cmd_line.split():
                        base = os.path.basename(part)
                        if base.startswith("sid-") or "sid_unet" in base:
                            name = base
                            break

                    new_rec = ProcessRecord(
                        pid=pid,
                        name=name,
                        cmdline=cmd_line,
                        cwd=os.getcwd(),
                        log_file=os.path.abspath(".logs.txt"),
                        start_time=datetime.now().isoformat(),
                        status=STATUS_RUNNING,
                        rss_mb=float(os_p.get("rss_mb", 0.0)),
                        ppid=int(os_p.get("ppid", 0)),
                        metadata={"auto_discovered": True},
                    )
                    records[str(pid)] = new_rec
                    changed = True

            if changed:
                self._write_records_unlocked(records)

            all_records = list(records.values())

        all_records.sort(key=lambda r: r.start_time, reverse=True)
        return all_records


# Global default registry instance
_default_registry: Optional[ProcessRegistry] = None


def get_process_registry(registry_dir: Optional[str] = None) -> ProcessRegistry:
    """Get or create singleton ProcessRegistry instance."""
    global _default_registry
    if _default_registry is None or registry_dir is not None:
        _default_registry = ProcessRegistry(registry_dir)
    return _default_registry


def terminate_process(
    pid: int,
    force: bool = False,
    timeout: float = 2.0,
    registry: Optional[ProcessRegistry] = None,
) -> bool:
    """Terminate a specific process PID gracefully or with SIGKILL and update ledger.

    Returns:
        True if the process is confirmed terminated, False otherwise.
    """
    from sid_unet.kill import _is_pid_alive

    reg = registry or get_process_registry()

    if not _is_pid_alive(pid):
        reg.mark_process_terminated(
            pid,
            status=STATUS_KILLED,
            reason="Process was already inactive",
        )
        return True

    # 1. Send signal
    if force:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            except Exception as e:
                logger.warning(f"Failed to send signal {sig} to PID {pid}: {e}")

        # Wait for graceful exit
        start_t = time.time()
        while time.time() - start_t < timeout:
            if not _is_pid_alive(pid):
                break
            time.sleep(0.1)

        # Escalate to SIGKILL if still alive
        if _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    time.sleep(0.1)
    is_dead = not _is_pid_alive(pid)

    if is_dead:
        reg.mark_process_terminated(
            pid,
            status=STATUS_KILLED,
            reason="Terminated by user request via process manager",
        )
    return is_dead
