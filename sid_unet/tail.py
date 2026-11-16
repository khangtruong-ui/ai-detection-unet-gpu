"""
Log tailing utility with automated termination detection for sid_unet tasks.

Unlike `tail -f`, `sid-tail` actively monitors the running task's process lifecycle
and output log markers, streaming live output and halting automatically as soon as
the task completes, crashes, or is terminated.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import List, Optional, Tuple

from sid_unet.kill import _is_pid_alive
from sid_unet.process_manager import (
    COMPLETION_PATTERNS,
    STATUS_COMPLETED,
    STATUS_EXCEPTION,
    STATUS_KILLED,
    STATUS_RUNNING,
    STATUS_UNKNOWN,
    ProcessRecord,
    collapse_line_carriage_returns,
    get_process_registry,
    inspect_log_for_termination,
)


def resolve_tail_target(
    target: Optional[str] = None,
    pid_override: Optional[int] = None,
    file_override: Optional[str] = None,
    registry_dir: Optional[str] = None,
) -> Tuple[str, Optional[int], int]:
    """
    Resolve target log file path, monitored PID, and starting byte offset.

    Returns:
        Tuple of (resolved_log_path, monitored_pid, start_offset)
    """
    reg = get_process_registry(registry_dir=registry_dir)
    reg.reconcile()

    # 1. Explicit PID override
    if pid_override is not None:
        rec = reg.get_process(pid_override)
        if rec is not None:
            return rec.log_file, rec.pid, rec.log_offset_start
        log_file = file_override or target or ".logs.txt"
        return os.path.abspath(log_file), pid_override, 0

    # 2. Positional target specified
    if target is not None:
        # Check if target is a numeric PID
        if target.isdigit():
            target_pid = int(target)
            rec = reg.get_process(target_pid)
            if rec is not None:
                return rec.log_file, rec.pid, rec.log_offset_start
            return os.path.abspath(".logs.txt"), target_pid, 0

        # Target is a file path
        target_path = os.path.abspath(target)
        # Check if any active process is logging to this file
        running_records = [r for r in reg.list_processes() if r.status == STATUS_RUNNING and _is_pid_alive(r.pid)]
        for r in running_records:
            if os.path.abspath(r.log_file) == target_path:
                return target_path, r.pid, r.log_offset_start
        return target_path, None, 0

    # 3. Explicit file override
    if file_override is not None:
        file_path = os.path.abspath(file_override)
        running_records = [r for r in reg.list_processes() if r.status == STATUS_RUNNING and _is_pid_alive(r.pid)]
        for r in running_records:
            if os.path.abspath(r.log_file) == file_path:
                return file_path, r.pid, r.log_offset_start
        return file_path, None, 0

    # 4. Default: Find latest running process in registry
    running_records = [r for r in reg.list_processes() if r.status == STATUS_RUNNING and _is_pid_alive(r.pid)]
    if running_records:
        latest = sorted(running_records, key=lambda r: r.start_time, reverse=True)[0]
        return latest.log_file, latest.pid, latest.log_offset_start

    # 5. Fallback to .logs.txt
    return os.path.abspath(".logs.txt"), None, 0


class _StreamLineRenderer:
    """Renders streamed log chunks while collapsing TQDM carriage-return updates cleanly."""

    def __init__(self, is_tty: Optional[bool] = None, allow_live_cr: Optional[bool] = None) -> None:
        if is_tty is None:
            try:
                is_tty = bool(sys.stdout.isatty())
            except Exception:
                is_tty = False
        self.is_tty = bool(is_tty)
        if allow_live_cr is None:
            allow_live_cr = self.is_tty or ("COLAB_RELEASE_TAG" in os.environ)
        self.allow_live_cr = bool(allow_live_cr)
        self.active_line = ""
        self.line_has_cr = False
        self._emitted_active_len = 0
        self._pending_cr = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._pending_cr:
            chunk = "\r" + chunk
            self._pending_cr = False
        if chunk.endswith("\r"):
            self._pending_cr = True
            chunk = chunk[:-1]
        if not chunk:
            return ""

        text = chunk.replace("\r\n", "\n")
        segments = text.split("\n")
        out_parts: List[str] = []

        for seg in segments[:-1]:
            had_cr = self.line_has_cr or ("\r" in seg)
            combined = collapse_line_carriage_returns(self.active_line + seg)
            if self.is_tty and had_cr:
                out_parts.append(f"\r\x1b[K{combined}\n")
            elif self.allow_live_cr and self._emitted_active_len > 0:
                out_parts.append(f"\r{combined}\n")
            elif not had_cr and self._emitted_active_len > 0 and self._emitted_active_len == len(self.active_line):
                out_parts.append(f"{seg}\n")
            else:
                out_parts.append(f"{combined}\n")
            self.active_line = ""
            self.line_has_cr = False
            self._emitted_active_len = 0

        tail = segments[-1]
        if tail:
            had_cr_in_tail = "\r" in tail
            self.active_line = collapse_line_carriage_returns(self.active_line + tail)
            if had_cr_in_tail:
                self.line_has_cr = True
            if self.is_tty:
                if self.line_has_cr:
                    out_parts.append(f"\r\x1b[K{self.active_line}")
                else:
                    out_parts.append(tail)
                self._emitted_active_len = len(self.active_line)
            elif self.allow_live_cr and self.line_has_cr:
                out_parts.append(f"\r{self.active_line}")
                self._emitted_active_len = len(self.active_line)
            elif not self.line_has_cr:
                out_parts.append(tail)
                self._emitted_active_len = len(self.active_line)

        return "".join(out_parts)

    def flush(self) -> str:
        if self._pending_cr:
            self._pending_cr = False
        if not self.active_line:
            return ""
        if self._emitted_active_len > 0:
            self.active_line = ""
            self.line_has_cr = False
            self._emitted_active_len = 0
            return "\n"
        line = self.active_line
        self.active_line = ""
        self.line_has_cr = False
        self._emitted_active_len = 0
        return f"{line}\n"


def _collapse_log_lines(content: str, lines_count: int) -> Tuple[List[str], Optional[str]]:
    """
    Split log text on newlines (preserving raw '\\r' for per-line TQDM collapsing)
    and return `(completed_lines, trailing_incomplete_raw_line)`.
    """
    normalized = content.replace("\r\n", "\n")
    ended_with_newline = normalized.endswith("\n")
    raw_lines = normalized.split("\n")
    if ended_with_newline and raw_lines and raw_lines[-1] == "":
        raw_lines.pop()

    trailing_raw: Optional[str] = None
    if not ended_with_newline and raw_lines:
        trailing_raw = raw_lines.pop()

    collapsed = [collapse_line_carriage_returns(line) for line in raw_lines]
    if trailing_raw is not None:
        max_completed = max(0, lines_count - 1)
        completed = collapsed[-max_completed:] if (max_completed > 0 and len(collapsed) > max_completed) else ([] if max_completed == 0 else collapsed)
    else:
        completed = collapsed[-lines_count:] if len(collapsed) > lines_count else collapsed
    return completed, trailing_raw


def print_initial_lines(
    file_path: str,
    lines_count: int,
    start_offset: int = 0,
) -> int:
    """
    Print the last `lines_count` lines from the file, starting at or after start_offset.
    Collapses TQDM carriage-return updates within each line.
    Returns the file byte offset after reading.
    """
    if not os.path.isfile(file_path) or lines_count <= 0:
        return 0

    try:
        file_size = os.path.getsize(file_path)
        if file_size <= start_offset:
            return file_size

        with open(file_path, "r", newline="", encoding="utf-8", errors="replace") as f:
            f.seek(start_offset)
            content = f.read()

        completed, trailing_raw = _collapse_log_lines(content, lines_count)
        for line in completed:
            sys.stdout.write(f"{line}\n")
        if trailing_raw:
            sys.stdout.write(f"{collapse_line_carriage_returns(trailing_raw)}\n")
        sys.stdout.flush()
        return file_size
    except Exception:
        return 0


def tail_log_stream(
    log_path: str,
    monitored_pid: Optional[int] = None,
    start_offset: int = 0,
    lines_count: int = 20,
    poll_interval: float = 0.2,
    timeout: Optional[float] = None,
    wait_for_file: float = 3.0,
    quiet: bool = False,
) -> int:
    """
    Tail log output continuously and halt automatically upon task completion or termination.

    Args:
        log_path: Path to the log file.
        monitored_pid: Optional PID to monitor for process death.
        start_offset: Byte offset where the relevant run begins.
        lines_count: Number of existing lines to print before following (default 20; 0 for none).
        poll_interval: Sleep seconds between log checks.
        timeout: Maximum seconds to tail before returning.
        wait_for_file: Maximum seconds to wait for log file creation if missing.
        quiet: If True, suppress banner and status messages.

    Returns:
        0 on clean completion, 1 on failure/exception/kill/timeout.
    """
    # Wait for log file creation if needed
    start_wait = time.time()
    while not os.path.isfile(log_path):
        if monitored_pid is not None and not _is_pid_alive(monitored_pid):
            if not quiet:
                print(f"[sid-tail] Process PID {monitored_pid} exited before log was created at: {log_path}", file=sys.stderr)
            return 1
        if (time.time() - start_wait) >= wait_for_file:
            if not quiet:
                print(f"[sid-tail] Log file not found after {wait_for_file:.1f}s: {log_path}", file=sys.stderr)
            return 1
        time.sleep(0.1)

    if not quiet:
        pid_info = f" (Monitoring PID: {monitored_pid})" if monitored_pid else ""
        print(f"📄 [sid-tail] Following {log_path}{pid_info} (halts upon task done)...")
        print("-" * 70)
        sys.stdout.flush()

    # Print initial lines if requested
    try:
        f = open(log_path, "r", newline="", encoding="utf-8", errors="replace")
    except Exception as e:
        if not quiet:
            print(f"[sid-tail] Failed to open log file: {e}", file=sys.stderr)
        return 1

    renderer = _StreamLineRenderer()

    try:
        content = ""
        if lines_count > 0:
            read_start = max(0, start_offset)
            f.seek(read_start)
            content = f.read()
            completed, trailing_raw = _collapse_log_lines(content, lines_count)
            for line in completed:
                sys.stdout.write(f"{line}\n")
            if trailing_raw is not None:
                rendered_tail = renderer.feed(trailing_raw)
                if rendered_tail:
                    sys.stdout.write(rendered_tail)
            sys.stdout.flush()
        else:
            f.seek(0, os.SEEK_END)

        start_ts = time.time()
        recent_chunks: List[str] = [content[-4096:]] if (lines_count > 0 and content) else []
        consecutive_empty = 0

        def _emit_chunk(text: str) -> None:
            rendered = renderer.feed(text)
            if rendered:
                sys.stdout.write(rendered)
                sys.stdout.flush()

        while True:
            if timeout is not None and (time.time() - start_ts) >= timeout:
                flushed = renderer.flush()
                if flushed:
                    sys.stdout.write(flushed)
                    sys.stdout.flush()
                if not quiet:
                    print(f"\n⚠️ [sid-tail] Timeout ({timeout:.1f}s) reached while following task.", file=sys.stderr)
                return 1

            chunk = f.read(8192)
            if chunk:
                _emit_chunk(chunk)
                recent_chunks.append(chunk)
                if len(recent_chunks) > 20:
                    recent_chunks.pop(0)
                consecutive_empty = 0
            else:
                consecutive_empty += 1

                # Check if monitored PID exited
                if monitored_pid is not None and not _is_pid_alive(monitored_pid):
                    # Drain any remaining writes
                    time.sleep(0.3)
                    final_chunk = f.read()
                    if final_chunk:
                        _emit_chunk(final_chunk)
                        recent_chunks.append(final_chunk)
                    break

                # If no PID is monitored and no data is arriving, check if log is already terminated
                if monitored_pid is None and consecutive_empty >= 2:
                    term_status, _, _ = inspect_log_for_termination(log_path, start_offset=start_offset)
                    if term_status in (STATUS_COMPLETED, STATUS_EXCEPTION, STATUS_KILLED):
                        break

                # Check for completion or termination patterns in recent output
                if consecutive_empty >= 2:
                    joined_recent = "".join(recent_chunks)
                    if any(pat.search(joined_recent) for pat in COMPLETION_PATTERNS):
                        time.sleep(0.2)
                        final_chunk = f.read()
                        if final_chunk:
                            _emit_chunk(final_chunk)
                        break

                    if (
                        "Traceback (most recent call last):" in joined_recent
                        or "KeyboardInterrupt" in joined_recent
                        or "Fatal unhandled error" in joined_recent
                    ):
                        time.sleep(0.3)
                        final_chunk = f.read()
                        if final_chunk:
                            _emit_chunk(final_chunk)
                        break

                time.sleep(poll_interval)

        flushed = renderer.flush()
        if flushed:
            sys.stdout.write(flushed)
            sys.stdout.flush()

    except KeyboardInterrupt:
        if not quiet:
            print("\n[sid-tail] Stream halted by user.")
        return 130
    finally:
        try:
            f.close()
        except Exception:
            pass

    # Inspect final status
    status, reason, err = inspect_log_for_termination(log_path, start_offset=start_offset)

    if not quiet:
        print("\n" + "-" * 70)
        if status == STATUS_COMPLETED:
            print(f"✅ [sid-tail] Task completed successfully: {reason or 'Done'}")
        elif status == STATUS_EXCEPTION:
            print(f"🛑 [sid-tail] Task terminated with unhandled exception: {err or reason}")
        elif status == STATUS_KILLED:
            print(f"⚠️ [sid-tail] Task was interrupted or killed: {reason}")
        else:
            print(f"ℹ️ [sid-tail] Task halted. Final status: {status} ({reason or 'N/A'})")
        sys.stdout.flush()

    return 0 if status == STATUS_COMPLETED else 1


def cli_main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for sid-tail."""
    parser = argparse.ArgumentParser(
        prog="sid-tail",
        description="Tail task logs in real time and automatically halt upon completion or error (unlike tail -f).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Log file path (e.g. .logs.txt) or active PID to monitor. If omitted, tails latest active task.",
    )
    parser.add_argument(
        "-p", "--pid",
        type=int,
        default=None,
        help="Process ID to monitor for process termination.",
    )
    parser.add_argument(
        "-f", "--file",
        dest="file_override",
        type=str,
        default=None,
        help="Log file path to tail.",
    )
    parser.add_argument(
        "-n", "--lines",
        type=int,
        default=20,
        help="Number of initial lines from log file to display before following.",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=None,
        help="Maximum duration in seconds to tail before exiting.",
    )
    parser.add_argument(
        "-i", "--poll-interval",
        type=float,
        default=0.2,
        help="Polling interval in seconds between checks.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress banner and termination summary messages.",
    )

    args = parser.parse_args(argv)

    log_path, monitored_pid, start_offset = resolve_tail_target(
        target=args.target,
        pid_override=args.pid,
        file_override=args.file_override,
    )

    return tail_log_stream(
        log_path=log_path,
        monitored_pid=monitored_pid,
        start_offset=start_offset,
        lines_count=args.lines,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(cli_main())
