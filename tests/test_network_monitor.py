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
    get_cpu_utilization,
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


def test_get_cpu_utilization_multicore(monkeypatch):
    import sid_unet.utils.network as net_mod

    if net_mod.psutil is not None:
        # 128 cores: 1 saturated at 96%, others 0%
        per_core = [96.0] + [0.0] * 127
        monkeypatch.setattr(net_mod.psutil, "cpu_percent", lambda interval=None, percpu=False: per_core if percpu else 0.75)

        avg_cpu, max_cpu = get_cpu_utilization()
        assert max_cpu == 96.0
        assert avg_cpu == pytest.approx(96.0 / 128, 0.01)


def test_bottleneck_detector_multicore_cpu_bound(monkeypatch):
    """Ensure a single pinned worker on a high core count machine is identified as CPU bound."""
    logger = logging.getLogger("test_multicore_cpu_bound")
    detector = BottleneckDetector(
        network_monitor=None,
        logger=logger,
        check_interval=0.0,
        window_size=5,
        cooldown=0.0,
    )

    import sid_unet.utils.network as net_mod
    monkeypatch.setattr(net_mod, "get_gpu_utilization", lambda: 5.0)
    if net_mod.psutil is not None:
        # Single worker core at 95%, average across 128 cores is under 1%
        per_core = [95.0] + [0.0] * 127
        monkeypatch.setattr(net_mod.psutil, "cpu_percent", lambda interval=None, percpu=False: per_core if percpu else 0.74)

    for _ in range(6):
        detector.record_step(data_time=0.50, compute_time=0.02)

    assert detector.current_status == BottleneckStatus.CPU_BOUND


def test_bottleneck_detector_acute_stall(monkeypatch):
    """Acute stall (> 5.0s) should trigger immediately on step 1 without waiting for window_size."""
    logger = logging.getLogger("test_acute_stall")
    class MockMonitor:
        def get_speed_mbps(self):
            return 25.0
        def get_speed_str(self):
            return "25.0 MB/s"

    detector = BottleneckDetector(
        network_monitor=MockMonitor(),
        logger=logger,
        check_interval=30.0,  # normally would wait 30s
        window_size=20,       # normally would wait for 5 samples
        cooldown=0.0,
        acute_stall_threshold=5.0,
    )

    import sid_unet.utils.network as net_mod
    monkeypatch.setattr(net_mod, "get_gpu_utilization", lambda: 0.0)
    if net_mod.psutil is not None:
        monkeypatch.setattr(net_mod.psutil, "cpu_percent", lambda interval=None, percpu=False: [20.0] * 4 if percpu else 20.0)

    # First step experiences a 12.0s acute stall
    status = detector.record_step(data_time=12.0, compute_time=1.0)
    assert status == BottleneckStatus.INTERNET_BOUND
    assert detector.current_status == BottleneckStatus.INTERNET_BOUND


def test_bottleneck_detector_prevents_false_gpu_bound(monkeypatch):
    """Momentary GPU spikes during brief compute phase must not mask data starvation."""
    logger = logging.getLogger("test_gpu_burst_filter")
    class MockMonitor:
        def get_speed_mbps(self):
            return 10.0
        def get_speed_str(self):
            return "10.0 MB/s"

    detector = BottleneckDetector(
        network_monitor=MockMonitor(),
        logger=logger,
        check_interval=0.0,
        window_size=5,
        cooldown=0.0,
    )

    import sid_unet.utils.network as net_mod
    # GPU utilization query reports 99% because it was sampled during the 0.05s compute slice
    monkeypatch.setattr(net_mod, "get_gpu_utilization", lambda: 99.0)
    if net_mod.psutil is not None:
        monkeypatch.setattr(net_mod.psutil, "cpu_percent", lambda interval=None, percpu=False: [20.0] * 4 if percpu else 20.0)

    for _ in range(6):
        # 10s data wait vs 0.05s compute -> GPU is starved 99.5% of the time!
        detector.record_step(data_time=10.0, compute_time=0.05)

    assert detector.current_status == BottleneckStatus.INTERNET_BOUND


def test_bottleneck_detector_step_interval():
    """Verify step_interval triggers evaluation on step boundaries."""
    logger = logging.getLogger("test_step_interval")
    detector = BottleneckDetector(
        network_monitor=None,
        logger=logger,
        check_interval=3600.0,  # large time interval so only step_interval triggers
        step_interval=3,
        window_size=5,
        cooldown=0.0,
    )

    # Steps 1 and 2: no check
    s1 = detector.record_step(data_time=0.01, compute_time=0.20)
    assert s1 is None
    s2 = detector.record_step(data_time=0.01, compute_time=0.20)
    assert s2 is None

    # Step 3: step_interval=3 met, but len(data_times)=3 < min(5, 5), so no check yet
    s3 = detector.record_step(data_time=0.01, compute_time=0.20)
    assert s3 is None

    # Step 4, 5
    s4 = detector.record_step(data_time=0.01, compute_time=0.20)
    s5 = detector.record_step(data_time=0.01, compute_time=0.20)

    # Step 6: step_interval=3 met and len >= 5 -> triggers check!
    s6 = detector.record_step(data_time=0.01, compute_time=0.20)
    assert s6 == BottleneckStatus.GPU_BOUND


def test_bottleneck_detector_high_speed_diagnosed_data_bound(monkeypatch):
    """When download speed is very high (e.g. 77 MB/s) but data wait stalls, diagnose DATA_BOUND."""
    logger = logging.getLogger("test_high_speed_data_bound")

    class MockHighSpeedMonitor:
        def get_speed_mbps(self):
            return 77.03

        def get_speed_str(self):
            return "77.03 MB/s"

    detector = BottleneckDetector(
        network_monitor=MockHighSpeedMonitor(),
        logger=logger,
        check_interval=0.0,
        window_size=5,
        cooldown=0.0,
        acute_stall_threshold=5.0,
    )

    import sid_unet.utils.network as net_mod
    monkeypatch.setattr(net_mod, "get_gpu_utilization", lambda: 0.0)
    if net_mod.psutil is not None:
        monkeypatch.setattr(net_mod.psutil, "cpu_percent", lambda interval=None, percpu=False: [25.0] * 4 if percpu else 25.0)

    # 17s stall despite 77 MB/s download speed -> data-bound / cache-thrashing
    status = detector.record_step(data_time=17.85, compute_time=2.64)
    assert status == BottleneckStatus.DATA_BOUND
    assert detector.current_status == BottleneckStatus.DATA_BOUND


