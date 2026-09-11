"""Pytest configuration shared by the whole suite.

The marker is registered here rather than in `pyproject.toml` because that file
is not tracked (see `.gitignore`), so anything put there would be registered on
one machine and produce an unknown-marker warning everywhere else.
"""

from __future__ import annotations

import jax
import pytest


@pytest.fixture(scope="module", autouse=True)
def _restore_x64():
    """Undo any `jax_enable_x64` a test module switches on.

    Several modules enable x64 inside their fixtures and never switch it back,
    which leaks float64 into every module that runs after them.
    """
    before = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", before)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: trains a network to convergence to check it against ground truth "
        '(minutes, not seconds; deselect with -m "not slow")',
    )
