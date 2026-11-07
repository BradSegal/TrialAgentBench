from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest


def pytest_configure() -> None:
    """Expose shared test fixtures as importable test-only modules."""

    tests_root = Path(__file__).resolve().parent
    if str(tests_root) not in sys.path:
        sys.path.insert(0, str(tests_root))


@pytest.fixture(autouse=True)
def _fresh_executor_probe() -> Iterator[None]:
    """Probe each test's executor image anew; the cache is per process in production."""
    from trialagentbench_verifier.executor import _probe_executor_image

    _probe_executor_image.cache_clear()
    yield
    _probe_executor_image.cache_clear()
