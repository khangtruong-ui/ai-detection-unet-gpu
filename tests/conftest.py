import gc
import os
import time
import pytest

@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Ensure GPU memory and garbage collector are flushed before and after each test."""
    yield
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Safeguard test execution against intermittent GPU memory fragmentation."""
    original_runtest = item.runtest

    def runtest_with_retry():
        # Default to 1 (no retries) to prevent hanging and infinite loops.
        # Users can override via MAX_TEST_RETRIES environment variable if desired.
        max_retries = int(os.environ.get("MAX_TEST_RETRIES", 1))
        for i in range(max_retries):
            try:
                return original_runtest()
            except Exception as e:
                err_str = str(e).lower()
                if ("out of memory" in err_str or "oom" in err_str) and i < max_retries - 1:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    gc.collect()
                    time.sleep(1.0)
                    continue
                raise

    item.runtest = runtest_with_retry
