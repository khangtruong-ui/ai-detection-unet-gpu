"""
Pytest configuration for SID-UNet test suite.
Dynamically handles environments without local PyTorch/GPU by running
configuration, reporting, network monitoring, and Modal cloud integration tests.
"""

from __future__ import annotations

import os
from pathlib import Path


def pytest_ignore_collect(collection_path, config):
    """
    If PyTorch or NumPy is not installed locally (e.g. lightweight Modal-only install),
    skip collecting test files that require local PyTorch/CUDA execution.
    When running in full GPU environment (or on Modal), all tests are collected.
    """
    try:
        import torch
        import numpy
        return False
    except ImportError:
        filename = os.path.basename(str(collection_path))
        # Tests that can run without PyTorch/NumPy
        standalone_tests = {
            "test_modal.py",
            "test_config.py",
            "test_network_monitor.py",
            "test_report.py",
        }
        if filename.startswith("test_") and filename not in standalone_tests:
            return True
        return False
