import os

# Disable numba JIT to avoid sparse/numba compatibility issues
os.environ["NUMBA_DISABLE_JIT"] = "1"

def pytest_collection_modifyitems(items, config):
    """
    Test that take longer than ~10s to run are marked as slow, tests that take more than one or two minutes to run are marked as veryslow.
    By default don't run either of these tests. You can call `pytest -m 'slow'` or `pytest -m 'slow and veryslow'` to run them
    """
    # Disable slow tests by default
    markexpr = config.getoption("markexpr", 'False') or 'False'
    config.option.markexpr = f"((not slow) and (not veryslow)) or ({markexpr})"