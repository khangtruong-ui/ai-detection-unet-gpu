"""
Terminal User Interface (TUI) for managing running and terminated sid_unet processes.

Provides:
- Interactive full-screen curses TUI:
  * Browse running, completed, killed, and exception-halted sid processes.
  * Real-time live output viewing (replacing raw `tail -f .logs.txt`).
  * Process termination (single or all).
  * Direct exception inspector jumping to stack traces.
- Headless and non-interactive CLI commands (--list, --logs, --tail, --status, --kill, --clear).
- CLI entry points for `sid-tui` and `sid-top`.
"""

from __future__ import annotations

import argparse
import curses
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sid_unet.kill import get_gpu_memory_summary, kill_all_background_tasks
from sid_unet.process_manager import (
    STATUS_COMPLETED,
    STATUS_EXCEPTION,
    STATUS_KILLED,
    STATUS_RUNNING,
    ProcessRecord,
    ProcessRegistry,
    get_process_registry,
    terminate_process,
)

# Color pair identifiers for curses
CP_HEADER = 1
CP_RUNNING = 2
CP_COMPLETED = 3
CP_KILLED = 4
CP_EXCEPTION = 5
CP_SELECTED = 6
CP_DIM = 7
CP_STATUS_BAR = 8
CP_ERROR_LINE = 9


def format_bytes(num_bytes: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def collapse_line_carriage_returns(text: str) -> str:
    """
    Simulate terminal cursor carriage return ('\\r') overwriting within a single physical line.
    Collapses multiple TQDM progress updates (or carriage-return-separated chunks)
    into the final visible text on the row, preventing multiple screen lines per update.
    """
    if "\r" not in text:
        return text
    parts = text.split("\r")
    buf: List[str] = []
    pos = 0
    for i, part in enumerate(parts):
        if i == 0:
            buf.extend(list(part))
            pos = len(buf)
        else:
            pos = 0
            for ch in part:
                if pos < len(buf):
                    buf[pos] = ch
                else:
                    buf.append(ch)
                pos += 1
    return "".join(buf)


def read_process_logs(
    record: ProcessRecord,
    max_lines: int = 5000,
) -> List[str]:
    """Read log file lines belonging to a specific process record with TQDM carriage return collapsing."""
    if not os.path.isfile(record.log_file):
        return [f"[No log file found at {record.log_file}]"]

    try:
        with open(record.log_file, "r", newline="", encoding="utf-8", errors="replace") as f:
            if record.log_offset_start > 0:
                f.seek(record.log_offset_start)
            raw_text = f.read()

        raw_lines = raw_text.replace("\r\n", "\n").split("\n")
        if raw_lines and raw_lines[-1] == "" and raw_text.endswith(("\n", "\r\n")):
            raw_lines.pop()

        lines = [collapse_line_carriage_returns(line) for line in raw_lines]
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return lines
    except Exception as e:
        return [f"[Error reading log file: {e}]"]


class SIDTerminalUI:
    """Curses-based terminal UI to manage and inspect active and terminated sid processes."""

    def __init__(
        self,
        registry: Optional[ProcessRegistry] = None,
        refresh_interval: float = 1.0,
    ):
        self.registry = registry or get_process_registry()
        self.refresh_interval = refresh_interval
        self.processes: List[ProcessRecord] = []
        self.selected_index: int = 0
        self.filter_mode: str = "ALL"  # ALL, RUNNING, EXCEPTION, TERMINATED
        self.view_mode: str = "SPLIT"  # SPLIT or FULL_LOG
        self.log_scroll_offset: int = 0
        self.log_auto_follow: bool = True
        self.cached_log_lines: List[str] = []
        self.last_log_pid: Optional[int] = None
        self.last_reconcile_time: float = 0.0
        self.status_message: str = "Ready. Press [?] for help, [q] to exit."
        self.status_message_time: float = time.time()
        self.should_exit: bool = False

    def get_filtered_processes(self) -> List[ProcessRecord]:
        if self.filter_mode == "RUNNING":
            return [p for p in self.processes if p.status == STATUS_RUNNING]
        elif self.filter_mode == "EXCEPTION":
            return [p for p in self.processes if p.status == STATUS_EXCEPTION]
        elif self.filter_mode == "TERMINATED":
            return [p for p in self.processes if p.status != STATUS_RUNNING]
        return self.processes

    def refresh_process_data(self, force: bool = False) -> None:
        now = time.time()
        if force or (now - self.last_reconcile_time >= self.refresh_interval):
            self.processes = self.registry.reconcile()
            self.last_reconcile_time = now

            filtered = self.get_filtered_processes()
            if self.selected_index >= len(filtered):
                self.selected_index = max(0, len(filtered) - 1)

    def set_status(self, msg: str) -> None:
        self.status_message = msg
        self.status_message_time = time.time()

    def run(self, stdscr: Any) -> None:
        """Main curses execution loop."""
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(self.refresh_interval * 1000))

        # Setup colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(CP_HEADER, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(CP_RUNNING, curses.COLOR_GREEN, -1)
        curses.init_pair(CP_COMPLETED, curses.COLOR_CYAN, -1)
        curses.init_pair(CP_KILLED, curses.COLOR_YELLOW, -1)
        curses.init_pair(CP_EXCEPTION, curses.COLOR_RED, -1)
        curses.init_pair(CP_SELECTED, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(CP_DIM, curses.COLOR_WHITE, -1)
        curses.init_pair(CP_STATUS_BAR, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(CP_ERROR_LINE, curses.COLOR_RED, -1)

        self.refresh_process_data(force=True)

        while not self.should_exit:
            self.refresh_process_data()
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            if h < 10 or w < 50:
                stdscr.addstr(0, 0, "Terminal window too small for SID-TUI.")
                stdscr.refresh()
                time.sleep(0.2)
                continue

            self._draw_header(stdscr, w)
            if self.view_mode == "FULL_LOG":
                self._draw_full_log(stdscr, h, w)
            else:
                self._draw_split_view(stdscr, h, w)

            self._draw_status_bar(stdscr, h, w)
            stdscr.refresh()

            # Handle user keyboard input
            try:
                ch = stdscr.getch()
                if ch != -1:
                    self._handle_key(ch)
            except Exception as e:
                self.set_status(f"Input error: {e}")

    def _draw_header(self, win: Any, width: int) -> None:
        """Render top summary banner."""
        filtered = self.get_filtered_processes()
        n_running = sum(1 for p in self.processes if p.status == STATUS_RUNNING)
        n_exc = sum(1 for p in self.processes if p.status == STATUS_EXCEPTION)
        n_term = sum(1 for p in self.processes if p.status != STATUS_RUNNING)

        time_str = datetime.now().strftime("%H:%M:%S")
        title = f" 🖥️  SID-UNet Process Manager [Filter: {self.filter_mode}] ({len(filtered)} shown) "
        stats = f"🟢 Running: {n_running} | 🔴 Exceptions: {n_exc} | ⚪ Terminated: {n_term} | 🕒 {time_str} "

        pad_len = max(0, width - len(title) - len(stats))
        header_text = title + (" " * pad_len) + stats
        header_text = header_text[:width]

        try:
            win.attron(curses.color_pair(CP_HEADER) | curses.A_BOLD)
            win.addstr(0, 0, header_text)
            win.attroff(curses.color_pair(CP_HEADER) | curses.A_BOLD)
        except curses.error:
            pass

    def _draw_split_view(self, win: Any, height: int, width: int) -> None:
        """Draw upper process table and lower log output viewer."""
        table_height = max(5, height // 2 - 1)
        log_height = height - table_height - 2

        self._draw_process_table(win, 1, table_height, width)
        self._draw_log_pane(win, 1 + table_height, log_height, width)

    def _draw_process_table(self, win: Any, start_y: int, max_h: int, width: int) -> None:
        """Render list of tracked processes."""
        filtered = self.get_filtered_processes()

        col_header = (
            f" {'STATUS':<11} {'PID':<7} {'NAME':<16} {'DURATION':<9} {'RAM':<9} {'COMMAND / REASON'}"
        )
        try:
            win.attron(curses.A_UNDERLINE | curses.A_BOLD)
            win.addstr(start_y, 0, col_header[:width].ljust(width))
            win.attroff(curses.A_UNDERLINE | curses.A_BOLD)
        except curses.error:
            pass

        content_h = max_h - 1
        if not filtered:
            msg = " [No processes found for current filter. Run sid-train or sid-cross-eval to begin.]"
            try:
                win.addstr(start_y + 1, 0, msg[:width])
            except curses.error:
                pass
            return

        # Viewport scrolling for process table
        top_offset = 0
        if self.selected_index >= content_h:
            top_offset = self.selected_index - content_h + 1

        for row_idx in range(content_h):
            p_idx = top_offset + row_idx
            if p_idx >= len(filtered):
                break

            proc = filtered[p_idx]
            is_selected = (p_idx == self.selected_index)

            # Badge styling
            badge = f"[{proc.status[:8]}]"
            color_pair = CP_DIM
            if proc.status == STATUS_RUNNING:
                badge = "[RUNNING]  "
                color_pair = CP_RUNNING
            elif proc.status == STATUS_EXCEPTION:
                badge = "[EXCEPTION]"
                color_pair = CP_EXCEPTION
            elif proc.status == STATUS_KILLED:
                badge = "[KILLED]   "
                color_pair = CP_KILLED
            elif proc.status == STATUS_COMPLETED:
                badge = "[COMPLETED]"
                color_pair = CP_COMPLETED

            ram_str = f"{proc.rss_mb:.1f} MB" if proc.rss_mb > 0 else "-"
            extra = proc.error_summary or proc.termination_reason or proc.cmdline
            extra = extra.replace("\n", " ").strip()

            line_text = (
                f" {badge:<11} {proc.pid:<7} {proc.name[:15]:<16} "
                f"{proc.duration_str:<9} {ram_str:<9} {extra}"
            )
            line_text = line_text[:width].ljust(width)

            y = start_y + 1 + row_idx
            try:
                if is_selected:
                    win.attron(curses.color_pair(CP_SELECTED) | curses.A_BOLD)
                    win.addstr(y, 0, line_text)
                    win.attroff(curses.color_pair(CP_SELECTED) | curses.A_BOLD)
                else:
                    win.attron(curses.color_pair(color_pair))
                    win.addstr(y, 0, line_text)
                    win.attroff(curses.color_pair(color_pair))
            except curses.error:
                pass

    def _draw_log_pane(self, win: Any, start_y: int, max_h: int, width: int) -> None:
        """Render bottom real-time log output viewer."""
        filtered = self.get_filtered_processes()
        if not filtered or self.selected_index >= len(filtered):
            return

        proc = filtered[self.selected_index]
        follow_str = "ON (tail -f)" if self.log_auto_follow else "PAUSED"
        title = (
            f" 📄 Log Output: {proc.name} (PID: {proc.pid}) | File: {os.path.basename(proc.log_file)} "
            f"| Follow: [{follow_str}] (toggle 'f') | Fullscreen: 'Enter' "
        )

        try:
            win.attron(curses.A_REVERSE)
            win.addstr(start_y, 0, title[:width].ljust(width))
            win.attroff(curses.A_REVERSE)
        except curses.error:
            pass

        # Update cached log content
        lines = read_process_logs(proc)
        self.cached_log_lines = lines

        log_display_h = max_h - 1
        if not lines:
            try:
                win.addstr(start_y + 1, 0, " [Empty log file]")
            except curses.error:
                pass
            return

        if self.log_auto_follow:
            scroll_pos = max(0, len(lines) - log_display_h)
        else:
            scroll_pos = max(0, min(self.log_scroll_offset, len(lines) - log_display_h))

        for i in range(log_display_h):
            line_idx = scroll_pos + i
            if line_idx >= len(lines):
                break
            raw_line = lines[line_idx]
            display_line = raw_line[:width].ljust(width)

            y = start_y + 1 + i
            try:
                if "Traceback" in raw_line or "Error" in raw_line or "Exception" in raw_line:
                    win.attron(curses.color_pair(CP_ERROR_LINE) | curses.A_BOLD)
                    win.addstr(y, 0, display_line)
                    win.attroff(curses.color_pair(CP_ERROR_LINE) | curses.A_BOLD)
                else:
                    win.addstr(y, 0, display_line)
            except curses.error:
                pass

    def _draw_full_log(self, win: Any, height: int, width: int) -> None:
        """Render fullscreen log viewer."""
        filtered = self.get_filtered_processes()
        if not filtered or self.selected_index >= len(filtered):
            self.view_mode = "SPLIT"
            return

        proc = filtered[self.selected_index]
        follow_str = "ON (Live Tail)" if self.log_auto_follow else "PAUSED (Browse)"
        banner = (
            f" 🔍 [FULLSCREEN LOG VIEWER] PID {proc.pid} ({proc.name}) | Status: {proc.status} "
            f"| Follow: [{follow_str}] | [f]: Toggle Tail | [e]: Jump Exception | [Esc/q/Enter]: Back "
        )

        try:
            win.attron(curses.A_REVERSE | curses.A_BOLD)
            win.addstr(1, 0, banner[:width].ljust(width))
            win.attroff(curses.A_REVERSE | curses.A_BOLD)
        except curses.error:
            pass

        lines = read_process_logs(proc)
        self.cached_log_lines = lines
        display_h = height - 3

        if self.log_auto_follow:
            scroll_pos = max(0, len(lines) - display_h)
        else:
            scroll_pos = max(0, min(self.log_scroll_offset, len(lines) - display_h))

        for i in range(display_h):
            line_idx = scroll_pos + i
            if line_idx >= len(lines):
                break
            raw_line = lines[line_idx]
            display_line = raw_line[:width].ljust(width)

            y = 2 + i
            try:
                if any(w in raw_line for w in ["Traceback", "Error", "Exception", "RuntimeError", "OOM"]):
                    win.attron(curses.color_pair(CP_ERROR_LINE) | curses.A_BOLD)
                    win.addstr(y, 0, display_line)
                    win.attroff(curses.color_pair(CP_ERROR_LINE) | curses.A_BOLD)
                else:
                    win.addstr(y, 0, display_line)
            except curses.error:
                pass

    def _draw_status_bar(self, win: Any, height: int, width: int) -> None:
        """Render bottom interactive help bar."""
        filtered = self.get_filtered_processes()
        curr_proc = filtered[self.selected_index] if (filtered and self.selected_index < len(filtered)) else None

        if time.time() - self.status_message_time < 5.0:
            msg = f" 💬 {self.status_message}"
        else:
            msg = " [↑/↓]: Navigate | [Enter]: View Log | [f]: Follow Log | [k]: Kill Task | [K]: Kill All | [e]: Jump Error | [Tab]: Filter | [c]: Clear Hist | [q]: Quit"

        bar_text = msg[:width].ljust(width)
        try:
            win.attron(curses.color_pair(CP_STATUS_BAR))
            win.addstr(height - 1, 0, bar_text)
            win.attroff(curses.color_pair(CP_STATUS_BAR))
        except curses.error:
            pass

    def _handle_key(self, ch: int) -> None:
        """Process keyboard shortcut."""
        filtered = self.get_filtered_processes()
        num_procs = len(filtered)

        # Navigation
        if ch in (curses.KEY_UP, ord("k")):
            if self.view_mode == "FULL_LOG" or not self.log_auto_follow:
                self.log_scroll_offset = max(0, self.log_scroll_offset - 1)
            else:
                self.selected_index = max(0, self.selected_index - 1)
        elif ch in (curses.KEY_DOWN, ord("j")):
            if self.view_mode == "FULL_LOG" or not self.log_auto_follow:
                self.log_scroll_offset = self.log_scroll_offset + 1
            else:
                self.selected_index = min(max(0, num_procs - 1), self.selected_index + 1)
        elif ch in (curses.KEY_PPAGE, ord("b")):  # Page up
            if self.view_mode == "FULL_LOG" or not self.log_auto_follow:
                self.log_scroll_offset = max(0, self.log_scroll_offset - 20)
        elif ch in (curses.KEY_NPAGE, ord(" ")):  # Page down
            if self.view_mode == "FULL_LOG" or not self.log_auto_follow:
                self.log_scroll_offset = self.log_scroll_offset + 20

        # Toggle view mode
        elif ch in (10, 13, ord("l")):  # Enter or 'l'
            if self.view_mode == "FULL_LOG":
                self.view_mode = "SPLIT"
            else:
                self.view_mode = "FULL_LOG"
                self.log_auto_follow = True

        # Toggle log auto-follow
        elif ch == ord("f"):
            self.log_auto_follow = not self.log_auto_follow
            state = "ON" if self.log_auto_follow else "PAUSED"
            self.set_status(f"Live log auto-follow set to: {state}")
            if not self.log_auto_follow and self.cached_log_lines:
                self.log_scroll_offset = max(0, len(self.cached_log_lines) - 20)

        # Jump to exception
        elif ch == ord("e"):
            if self.cached_log_lines:
                found_idx = None
                for idx, line in enumerate(self.cached_log_lines):
                    if "Traceback (most recent call last)" in line:
                        found_idx = idx
                        break
                if found_idx is not None:
                    self.log_auto_follow = False
                    self.log_scroll_offset = max(0, found_idx - 1)
                    self.view_mode = "FULL_LOG"
                    self.set_status(f"Jumped to Traceback at line {found_idx + 1}")
                else:
                    self.set_status("No Exception / Traceback found in log file.")

        # Filter mode
        elif ch == ord("\t"):
            filters = ["ALL", "RUNNING", "EXCEPTION", "TERMINATED"]
            curr_idx = filters.index(self.filter_mode) if self.filter_mode in filters else 0
            self.filter_mode = filters[(curr_idx + 1) % len(filters)]
            self.selected_index = 0
            self.set_status(f"Active filter: {self.filter_mode}")

        # Kill single process
        elif ch == ord("k") and self.view_mode != "FULL_LOG":
            if filtered and self.selected_index < len(filtered):
                target = filtered[self.selected_index]
                if target.status == STATUS_RUNNING:
                    self.set_status(f"Terminating PID {target.pid} ({target.name})...")
                    terminated = terminate_process(target.pid, force=False, registry=self.registry)
                    if terminated:
                        self.set_status(f"✅ PID {target.pid} successfully terminated.")
                    else:
                        self.set_status(f"⚠️ Failed to terminate PID {target.pid}.")
                    self.refresh_process_data(force=True)
                else:
                    self.set_status(f"Process PID {target.pid} is already {target.status}.")

        # Kill all processes
        elif ch == ord("K"):
            self.set_status("🛑 Terminating all background sid_unet processes...")
            count = kill_all_background_tasks(force=False, quiet=True)
            self.set_status(f"✅ Successfully killed {count} task(s).")
            self.refresh_process_data(force=True)

        # Clear terminated history
        elif ch == ord("c"):
            removed = self.registry.clear_history(keep_running=True)
            self.set_status(f"Cleared {removed} terminated process record(s) from history.")
            self.refresh_process_data(force=True)

        # Refresh
        elif ch in (ord("r"), ord("R")):
            self.refresh_process_data(force=True)
            self.set_status("Refreshed process list and live diagnostics.")

        # Exit
        elif ch in (ord("q"), ord("Q"), 27):  # 'q' or Esc
            if self.view_mode == "FULL_LOG":
                self.view_mode = "SPLIT"
            else:
                self.should_exit = True


# ------------------------------------------------------------------------------
# Headless & Non-Interactive CLI Renderers
# ------------------------------------------------------------------------------

def print_process_table(
    processes: List[ProcessRecord],
    filter_mode: Optional[str] = None,
) -> None:
    """Print clean, formatted table of processes for non-interactive shells."""
    from tabulate import tabulate

    if filter_mode and filter_mode.upper() != "ALL":
        fm_upper = filter_mode.upper()
        if fm_upper == "TERMINATED":
            processes = [p for p in processes if p.status != STATUS_RUNNING]
        else:
            processes = [p for p in processes if p.status.upper() == fm_upper]

    if not processes:
        print(f"✨ No processes found (Filter: {filter_mode or 'ALL'}).")
        return

    table_data = []
    for p in processes:
        status_icon = "🟢" if p.status == STATUS_RUNNING else ("🔴" if p.status == STATUS_EXCEPTION else ("🟡" if p.status == STATUS_KILLED else "⚪"))
        status_str = f"{status_icon} {p.status}"
        ram_str = f"{p.rss_mb:.1f} MB" if p.rss_mb > 0 else "-"
        reason = p.error_summary or p.termination_reason or "-"
        if len(reason) > 50:
            reason = reason[:47] + "..."
        table_data.append([
            status_str,
            p.pid,
            p.name,
            p.duration_str,
            ram_str,
            reason,
        ])

    headers = ["Status", "PID", "Name", "Duration", "RAM", "Termination Reason / Error"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="fancy_grid") + "\n")


def print_system_status(registry: Optional[ProcessRegistry] = None) -> None:
    """Print summary overview of running tasks and GPU usage."""
    reg = registry or get_process_registry()
    procs = reg.reconcile()
    running = [p for p in procs if p.status == STATUS_RUNNING]
    exceptions = [p for p in procs if p.status == STATUS_EXCEPTION]
    killed = [p for p in procs if p.status == STATUS_KILLED]
    completed = [p for p in procs if p.status == STATUS_COMPLETED]

    print("\n" + "=" * 65)
    print("📊 SID-UNET PROCESS AND SYSTEM MONITOR")
    print("=" * 65)
    print(f"  • Active Running Tasks : {len(running)}")
    print(f"  • Exception Terminations: {len(exceptions)}")
    print(f"  • Killed / Aborted Tasks: {len(killed)}")
    print(f"  • Successfully Completed: {len(completed)}")
    print(f"  • Total Tracked in History: {len(procs)}")

    if running:
        print("\n🚀 Active Processes:")
        for r in running:
            print(f"   - PID {r.pid:<6} | {r.name:<15} | Running for {r.duration_str:<8} | RAM: {r.rss_mb:.1f} MB | {r.cmdline[:60]}")

    if exceptions:
        print("\n🔴 Recent Exceptions:")
        for e in exceptions[:5]:
            print(f"   - PID {e.pid:<6} | {e.name:<15} | {e.error_summary or e.termination_reason}")

    gpu_info = get_gpu_memory_summary()
    if gpu_info:
        print("\n🖥️ GPU Utilization:")
        for g in gpu_info:
            print(f"   - GPU {g['index']} ({g['name']}): {g['used_mb']:.1f} MB used / {g['total_mb']:.1f} MB total ({g['free_mb']:.1f} MB free)")
    print("=" * 65 + "\n")


def print_process_logs(
    pid: int,
    tail: bool = False,
    lines_count: int = 100,
    registry: Optional[ProcessRegistry] = None,
) -> None:
    """Stream or print log output for a specific PID."""
    reg = registry or get_process_registry()
    proc = reg.get_process(pid)

    if not proc:
        # Check if active process with this PID exists
        reg.reconcile()
        proc = reg.get_process(pid)

    if not proc:
        print(f"❌ Error: Process PID {pid} is not registered in ledger.")
        return

    log_file = proc.log_file
    if not os.path.isfile(log_file):
        print(f"❌ Error: Log file for PID {pid} not found at: {log_file}")
        return

    print(f"📄 Displaying output for PID {pid} ({proc.name}) [File: {log_file}]:\n" + "-" * 70)
    all_lines = read_process_logs(proc, max_lines=max(lines_count, 1000))
    to_show = all_lines[-lines_count:] if len(all_lines) > lines_count else all_lines
    for line in to_show:
        print(line)

    if tail and proc.status == STATUS_RUNNING:
        print("\n🔄 Following log output in real-time (Press Ctrl+C to stop)...")
        try:
            with open(log_file, "r", newline="", encoding="utf-8", errors="replace") as f:
                f.seek(0, os.SEEK_END)
                while True:
                    chunk = f.read(4096)
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                    else:
                        time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped log stream.")


def cli_main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for sid-tui and sid-top."""
    parser = argparse.ArgumentParser(
        prog="sid-tui",
        description="Terminal User Interface and process monitor for sid_unet running and terminated tasks.",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all tracked processes in a non-interactive table.",
    )
    parser.add_argument(
        "-s", "--status",
        action="store_true",
        help="Print summary system overview of processes and GPU status.",
    )
    parser.add_argument(
        "--filter",
        choices=["all", "running", "exception", "terminated"],
        default="all",
        help="Filter process list by status.",
    )
    parser.add_argument(
        "--logs",
        type=int,
        metavar="PID",
        help="Display log output for the specified process PID.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        metavar="PID",
        help="Stream live log output for the specified running PID.",
    )
    parser.add_argument(
        "--kill",
        type=int,
        metavar="PID",
        help="Terminate a specific process PID.",
    )
    parser.add_argument(
        "--kill-all",
        action="store_true",
        help="Terminate all running sid_unet background tasks.",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear terminated process records from history.",
    )
    parser.add_argument(
        "-n", "--lines",
        type=int,
        default=50,
        help="Number of log lines to show with --logs (default: 50).",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Force non-interactive snapshot mode even when running in a terminal.",
    )
    args = parser.parse_args(argv)

    registry = get_process_registry()

    # Action dispatch
    if args.kill_all:
        count = kill_all_background_tasks(force=False)
        print(f"✅ Terminated {count} process(es).")
        return 0

    if args.kill is not None:
        success = terminate_process(args.kill, force=False, registry=registry)
        if success:
            print(f"✅ PID {args.kill} successfully terminated.")
        else:
            print(f"⚠️ PID {args.kill} could not be terminated.")
        return 0

    if args.clear_history:
        removed = registry.clear_history(keep_running=True)
        print(f"✅ Cleared {removed} terminated process record(s) from ledger.")
        return 0

    if args.logs is not None:
        print_process_logs(args.logs, tail=False, lines_count=args.lines, registry=registry)
        return 0

    if args.tail is not None:
        print_process_logs(args.tail, tail=True, lines_count=args.lines, registry=registry)
        return 0

    if args.status:
        print_system_status(registry=registry)
        return 0

    if args.list:
        procs = registry.reconcile()
        print_process_table(procs, filter_mode=args.filter)
        return 0

    # If stdout or stdin is not a TTY (or --no-tui was passed), fallback cleanly to table view
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if not is_tty or args.no_tui or os.environ.get("SID_TUI_NO_INTERACTIVE"):
        procs = registry.reconcile()
        print_process_table(procs, filter_mode=args.filter)
        return 0

    # Run curses interactive TUI
    ui = SIDTerminalUI(registry=registry)
    try:
        curses.wrapper(ui.run)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
