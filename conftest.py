"""Repo-root pytest configuration.

Adds the ``--runslow`` option and the ``slow`` marker. Tests marked ``slow`` are
SKIPPED unless ``--runslow`` is given, so the default suite stays fast enough to
run on every edit while the expensive large-grid cases remain one flag away.

Nothing here changes the behaviour of any pre-existing test: at the time this
file was added no test in the repository carried a ``slow`` marker, so the skip
rule applies only to tests that opt in by marking themselves.

Also registers ``pylle_full``, for tests whose subject is the OUTCOME of a
completed pyLLE ladder run rather than an invariant of the code. They are cheap
to execute, but a failure there is fixed by re-running the cross-check (which
needs Julia and a pyLLE environment), not by editing code, so they are skipped by
default and run with ``pytest -m pylle_full``. Every cheap invariant -- frozen
artifact hashes, convention-function identity, the dispersion mirror, the H5
gate, the tolerance fingerprint, schema validation -- deliberately stays in the
default suite. See docs/VALIDATION_STATUS.md section 3.
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
    config.addinivalue_line(
        "markers",
        "pylle_full: asserts the shape of a completed pyLLE ladder run; skipped "
        "unless selected with -m pylle_full",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # ``pylle_full`` is skipped unless it was explicitly selected with -m. The
    # tests are still COLLECTED either way, which is what
    # tests/test_validation_freeze.py::test_marked_tests_still_exist relies on to
    # prove the split hid nothing.
    selected = config.getoption("-m", default="") or ""
    if "pylle_full" not in selected:
        skip_pylle = pytest.mark.skip(
            reason="needs a completed pyLLE ladder run; select with -m pylle_full")
        for item in items:
            if "pylle_full" in item.keywords:
                item.add_marker(skip_pylle)

    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
