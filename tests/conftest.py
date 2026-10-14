import pytest
import time
import gc
import os

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    original_runtest = item.runtest

    def runtest_with_retry():
        max_retries = int(os.environ.get("MAX_TEST_RETRIES", 10))
        for i in range(max_retries):
            try:
                return original_runtest()
            except Exception as e:
                err_str = str(e).lower()
                if "out of memory" in err_str or "oom" in err_str:
                    if i < max_retries - 1:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                        gc.collect()
                        # Exponential backoff + fixed wait
                        time.sleep(2 * (i + 1))
                        continue
                raise

    item.runtest = runtest_with_retry
