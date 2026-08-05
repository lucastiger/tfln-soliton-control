"""Repo-root pytest configuration.

Adds the ``--runslow`` option and the ``slow`` marker. Tests marked ``slow`` are
SKIPPED unless ``--runslow`` is given, so the default suite stays fast enough to
run on every edit while the expensive large-grid cases remain one flag away.

Nothing here changes the behaviour of any pre-existing test: at the time this
file was added no test in the repository carried a ``slow`` marker, so the skip
rule applies only to tests that opt in by marking themselves.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (skipped by default)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: expensive test, skipped unless --runslow is passed",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
