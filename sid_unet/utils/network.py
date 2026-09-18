"""
Network speed measurement and performance bottleneck detection utilities.
Provides real-time network throughput tracking (passive I/O counters and active probes)
and bottleneck profiling (GPU-bound vs. CPU-bound vs. Internet-bound).
"""

from __future__ import annotations

from enum import Enum
import logging
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.request

try:
    import psutil
except ImportError:
    psutil = None


def get_network_rx_bytes() -> int:
    """
    Get total received bytes across all network interfaces (excluding loopback).
    Tries psutil first, falls back to parsing /proc/net/dev on Linux.
    """
    if psutil is not None:
        try:
            counters = psutil.net_io_counters(pernic=True)
            total_rx = sum(
                c.bytes_recv for name, c in counters.items() if not name.startswith("lo")
            )
            return total_rx
        except Exception:
            pass

    try:
        with open("/proc/net/dev", "r") as f:
            lines = f.readlines()[2:]
        total_rx = 0
        for line in lines:
            parts = line.strip().split()
            iface = parts[0].rstrip(":")
            if iface != "lo":
                total_rx += int(parts[1])
        return total_rx
    except Exception:
        return 0


def format_network_speed(bytes_per_sec: float) -> str:
    """Format bytes per second into human-readable transfer rate (e.g. 14.5 MB/s)."""
    if bytes_per_sec < 0:
        bytes_per_sec = 0.0

    mb_per_sec = bytes_per_sec / (1024 * 1024)
    if mb_per_sec >= 1000:
        return f"{mb_per_sec / 1024:.2f} GB/s"
    elif mb_per_sec >= 1.0:
        return f"{mb_per_sec:.2f} MB/s"
    elif bytes_per_sec >= 1024:
        return f"{bytes_per_sec / 1024:.1f} KB/s"
    else:
        return f"{bytes_per_sec:.0f} B/s"


def measure_active_download_speed(
    url: str = "https://huggingface.co",
    timeout: float = 3.0,
    chunk_size: int = 256 * 1024,
) -> Optional[float]:
    """
    Active speed measurement probe: downloads a small payload and computes transfer rate in MB/s.
    Returns None if probe fails or is unreachable.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SID-UNet-SpeedProbe/1.0"},
        )
        t_start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(chunk_size)
        t_elapsed = time.perf_counter() - t_start
        if t_elapsed > 0 and len(data) > 0:
            mb = len(data) / (1024 * 1024)
            return mb / t_elapsed
    except Exception:
        return None
    return None


def get_gpu_utilization() -> Optional[float]:
    """Query GPU utilization percentage via nvidia-smi if available."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.0,
            check=False,
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            if lines:
                return float(lines[0].strip())
    except Exception:
        pass
    return None


class NetworkSpeedMonitor:
    """
    Background daemon thread monitoring network throughput.
    Supports passive I/O monitoring (measuring actual streaming traffic),
    active probes, or hybrid modes.
    """

    def __init__(
        self,
        interval: float = 1.0,
        method: str = "io_counters",  # 'io_counters', 'active_probe', 'hybrid'
        active_url: str = "https://huggingface.co",
        active_interval: float = 60.0,
    ):
        self.interval = max(0.1, interval)
        self.method = method.lower()
        self.active_url = active_url
        self.active_interval = max(5.0, active_interval)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._current_speed_bps = 0.0
        self._current_speed_str = "0.0 MB/s"
        self._active_speed_mbps: Optional[float] = None
        self._last_rx_bytes = get_network_rx_bytes()
        self._last_time = time.perf_counter()

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._last_rx_bytes = get_network_rx_bytes()
        self._last_time = time.perf_counter()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="NetworkSpeedMonitor",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal monitor thread to stop and wait for termination."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        last_active_time = 0.0
        while not self._stop_event.wait(timeout=self.interval):
            now = time.perf_counter()

            if self.method in ("io_counters", "hybrid"):
                rx = get_network_rx_bytes()
                dt = now - self._last_time
                if dt > 0:
                    drx = rx - self._last_rx_bytes
                    if drx < 0:
                        drx = 0  # Counter wrap/reset
                    inst_bps = drx / dt

                    with self._lock:
                        # Exponential moving average for stability
                        self._current_speed_bps = 0.7 * inst_bps + 0.3 * self._current_speed_bps
                        self._current_speed_str = format_network_speed(self._current_speed_bps)

                    self._last_rx_bytes = rx
                    self._last_time = now

            if self.method in ("active_probe", "hybrid"):
                if now - last_active_time >= self.active_interval:
                    probe_mbps = measure_active_download_speed(url=self.active_url)
                    if probe_mbps is not None:
                        with self._lock:
                            self._active_speed_mbps = probe_mbps
                            if self.method == "active_probe":
                                self._current_speed_bps = probe_mbps * 1024 * 1024
                                self._current_speed_str = f"{probe_mbps:.2f} MB/s"
                    last_active_time = now

    def get_speed_bps(self) -> float:
        """Return current estimated download speed in bytes per second."""
        with self._lock:
            return self._current_speed_bps

    def get_speed_mbps(self) -> float:
        """Return current estimated download speed in MB/s."""
        with self._lock:
            return self._current_speed_bps / (1024 * 1024)

    def get_speed_str(self) -> str:
        """Return human-readable download speed string (e.g. '12.4 MB/s')."""
        with self._lock:
            return self._current_speed_str

    def get_active_probe_speed_mbps(self) -> Optional[float]:
        """Return most recent active probe speed result if available."""
        with self._lock:
            return self._active_speed_mbps


class BottleneckStatus(str, Enum):
    GPU_BOUND = "GPU_BOUND"
    CPU_BOUND = "CPU_BOUND"
    INTERNET_BOUND = "INTERNET_BOUND"
    BALANCED = "BALANCED"


class BottleneckDetector:
    """
    Monitors data loading time vs GPU compute time and system utilization
    to diagnose pipeline bottlenecks (Internet vs. CPU vs. GPU-bound).
    """

    def __init__(
        self,
        network_monitor: Optional[NetworkSpeedMonitor] = None,
        logger: Optional[logging.Logger] = None,
        check_interval: float = 30.0,
        step_interval: Optional[int] = None,
        window_size: int = 20,
        cooldown: float = 30.0,
    ):
        self.network_monitor = network_monitor
        self.logger = logger or logging.getLogger(__name__)
        self.check_interval = check_interval
        self.step_interval = step_interval
        self.window_size = window_size
        self.cooldown = cooldown

        self._data_times: List[float] = []
        self._compute_times: List[float] = []
        self._step_count = 0
        self._last_check_time = time.perf_counter()
        self._last_warn_time = 0.0
        self._last_status: BottleneckStatus = BottleneckStatus.BALANCED

    def record_step(self, data_time: float, compute_time: float) -> Optional[BottleneckStatus]:
        """
        Record a single training step's data wait time and compute time.
        Periodically triggers bottleneck analysis and warning logs.
        """
        self._data_times.append(max(0.0, data_time))
        self._compute_times.append(max(0.0, compute_time))
        self._step_count += 1

        if len(self._data_times) > self.window_size:
            self._data_times.pop(0)
            self._compute_times.pop(0)

        now = time.perf_counter()
        time_due = (self.check_interval <= 0) or (now - self._last_check_time >= self.check_interval)
        step_due = (self.step_interval is not None and self._step_count % self.step_interval == 0)
        has_min_samples = len(self._data_times) >= min(5, self.window_size)

        if (time_due or step_due) and has_min_samples:
            self._last_check_time = now
            status = self.check_and_report()
            return status

        return None

    def check_and_report(self) -> BottleneckStatus:
        """Evaluate recent metrics and emit warnings or status logs."""
        if not self._data_times:
            return BottleneckStatus.BALANCED

        avg_data = sum(self._data_times) / len(self._data_times)
        avg_compute = sum(self._compute_times) / len(self._compute_times)

        cpu_percent = psutil.cpu_percent(interval=None) if psutil is not None else 0.0
        gpu_util = get_gpu_utilization()
        net_speed_mbps = self.network_monitor.get_speed_mbps() if self.network_monitor else 0.0
        net_str = self.network_monitor.get_speed_str() if self.network_monitor else "0.0 MB/s"

        now = time.perf_counter()

        # Decision logic:
        # If compute time dominates data wait time or GPU utilization is high -> GPU BOUND (Desirable!)
        is_gpu_busy = (gpu_util is not None and gpu_util >= 60.0) or (avg_compute >= avg_data * 0.9)
        if is_gpu_busy:
            self._last_status = BottleneckStatus.GPU_BOUND
            self.logger.debug(
                f"✅ Training is GPU BOUND: Compute={avg_compute*1000:.1f}ms, Data wait={avg_data*1000:.1f}ms "
                f"(GPU: {gpu_util if gpu_util is not None else 'N/A'}%, CPU: {cpu_percent:.1f}%, Net: {net_str})."
            )
            return self._last_status

        # If data wait time > compute time, pipeline is starved of data
        # Check whether Internet or CPU is the culprit
        if avg_data > avg_compute:
            # If network speed is actively transferring or very low/throttling during streaming
            if net_speed_mbps > 0.5 or (net_speed_mbps > 0.05 and cpu_percent < 75.0):
                self._last_status = BottleneckStatus.INTERNET_BOUND
                if now - self._last_warn_time >= self.cooldown:
                    self._last_warn_time = now
                    self.logger.warning(
                        f"⚠️ Bottleneck Warning: Training is INTERNET BOUND! "
                        f"DataLoader wait time ({avg_data:.2f}s) dominates GPU compute ({avg_compute:.2f}s). "
                        f"Network download speed: {net_str}, CPU: {cpu_percent:.1f}%. "
                        f"The GPU is starving for data while waiting for remote streaming. "
                        f"Tip: Increase prefetch_batches, optimize network connection, or cache dataset locally."
                    )
                return self._last_status
            elif cpu_percent >= 75.0:
                self._last_status = BottleneckStatus.CPU_BOUND
                if now - self._last_warn_time >= self.cooldown:
                    self._last_warn_time = now
                    self.logger.warning(
                        f"⚠️ Bottleneck Warning: Training is CPU BOUND! "
                        f"DataLoader wait time ({avg_data:.2f}s) dominates GPU compute ({avg_compute:.2f}s) "
                        f"with high CPU usage ({cpu_percent:.1f}%). "
                        f"CPU worker threads are bottlenecking data augmentation or image decoding. "
                        f"Tip: Increase num_workers, enable pin_memory, or simplify augmentations."
                    )
                return self._last_status
            else:
                self._last_status = BottleneckStatus.CPU_BOUND
                if now - self._last_warn_time >= self.cooldown:
                    self._last_warn_time = now
                    self.logger.warning(
                        f"⚠️ Bottleneck Warning: Training is DATA-BOUND (Data wait: {avg_data:.2f}s vs GPU compute: {avg_compute:.2f}s). "
                        f"GPU is idling for {avg_data / max(avg_data + avg_compute, 1e-6) * 100:.0f}% of batch cycle. "
                        f"(Net: {net_str}, CPU: {cpu_percent:.1f}%)."
                    )
                return self._last_status

        self._last_status = BottleneckStatus.BALANCED
        return self._last_status

    @property
    def current_status(self) -> BottleneckStatus:
        return self._last_status
