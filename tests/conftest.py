"""Pytest configuration shared by the whole suite.

The marker is registered here rather than in `pyproject.toml` because that file
is not tracked (see `.gitignore`), so anything put there would be registered on
one machine and produce an unknown-marker warning everywhere else.
"""

from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: trains a network to convergence to check it against ground truth "
        '(minutes, not seconds; deselect with -m "not slow")',
    )
