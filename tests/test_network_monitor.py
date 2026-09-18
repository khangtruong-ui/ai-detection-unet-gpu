import time
import logging
import pytest
from sid_unet.utils.network import (
    get_network_rx_bytes,
    format_network_speed,
    measure_active_download_speed,
    NetworkSpeedMonitor,
    BottleneckDetector,
    BottleneckStatus,
)


def test_get_network_rx_bytes():
    rx = get_network_rx_bytes()
    assert isinstance(rx, int)
    assert rx >= 0


def test_format_network_speed():
    assert format_network_speed(0) == "0 B/s" or "0.0" in format_network_speed(0)
    assert format_network_speed(500) == "500 B/s"
    assert "KB/s" in format_network_speed(2048)
    assert "1.00 MB/s" == format_network_speed(1024 * 1024)
    assert "10.00 MB/s" == format_network_speed(10 * 1024 * 1024)
    assert "1.50 GB/s" == format_network_speed(1.5 * 1024 * 1024 * 1024)


def test_network_speed_monitor_lifecycle():
    monitor = NetworkSpeedMonitor(interval=0.1, method="io_counters")
    assert monitor.get_speed_bps() == 0.0
    assert "MB/s" in monitor.get_speed_str() or "B/s" in monitor.get_speed_str()

    monitor.start()
    time.sleep(0.3)
    speed_bps = monitor.get_speed_bps()
    speed_mbps = monitor.get_speed_mbps()
    speed_str = monitor.get_speed_str()

    assert isinstance(speed_bps, float)
    assert isinstance(speed_mbps, float)
    assert isinstance(speed_str, str)
    monitor.stop()


def test_bottleneck_detector_gpu_bound():
    logger = logging.getLogger("test_gpu_bound")
    detector = BottleneckDetector(
        network_monitor=None,
        logger=logger,
        check_interval=0.0,
        window_size=5,
        cooldown=0.0,
    )

    # Simulate fast data loading and slow compute -> GPU bound (optimal)
    status = None
    for _ in range(6):
        status = detector.record_step(data_time=0.01, compute_time=0.20)

    assert status == BottleneckStatus.GPU_BOUND or detector.current_status == BottleneckStatus.GPU_BOUND


def test_bottleneck_detector_cpu_bound(monkeypatch):
    logger = logging.getLogger("test_cpu_bound")
    detector = BottleneckDetector(
        network_monitor=None,
        logger=logger,
        check_interval=0.0,
        window_size=5,
        cooldown=0.0,
    )

    # Mock high CPU utilization and GPU idle
    import sid_unet.utils.network as net_mod
    monkeypatch.setattr(net_mod, "get_gpu_utilization", lambda: 5.0)
    if net_mod.psutil is not None:
        monkeypatch.setattr(net_mod.psutil, "cpu_percent", lambda interval=None: 95.0)

    # Data time dominates compute time
    for _ in range(6):
        detector.record_step(data_time=0.50, compute_time=0.02)

    assert detector.current_status == BottleneckStatus.CPU_BOUND


def test_bottleneck_detector_internet_bound(monkeypatch):
    logger = logging.getLogger("test_internet_bound")

    class MockMonitor:
        def get_speed_mbps(self):
            return 8.5
        def get_speed_str(self):
            return "8.5 MB/s"

    detector = BottleneckDetector(
        network_monitor=MockMonitor(),
        logger=logger,
        check_interval=0.0,
        window_size=5,
        cooldown=0.0,
    )

    import sid_unet.utils.network as net_mod
    monkeypatch.setattr(net_mod, "get_gpu_utilization", lambda: 2.0)
    if net_mod.psutil is not None:
        monkeypatch.setattr(net_mod.psutil, "cpu_percent", lambda interval=None: 25.0)

    # Slow data wait while network is streaming heavily -> internet bound
    for _ in range(6):
        status = detector.record_step(data_time=0.80, compute_time=0.02)

    assert detector.current_status == BottleneckStatus.INTERNET_BOUND
