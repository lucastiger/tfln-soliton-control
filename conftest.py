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

Two further things live here, both added with the packaging work and both inert
on a fully-provisioned developer machine:

* a toolchain banner (``pytest_sessionstart``) naming the jax/jaxlib/numpy
  build, so a CI log is self-documenting when a ULP comparison drifts;
* ``collect_ignore`` for the test modules that hard-import an OPTIONAL extra,
  so the benchmark's own suite still runs in an environment that installed only
  the base dependencies (see ``_OPTIONAL_MODULE_REQUIREMENTS``).
"""

from __future__ import annotations

import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Optional-extra test modules
# ---------------------------------------------------------------------------
# ``torch`` moved out of the base dependency set and into the ``ml`` extra: the
# stochastic-LLE benchmark is a solver, and a user running it should not have to
# install a ~1 GB deep-learning framework to do so. These two modules import
# torch at MODULE scope, so in a base-only environment they fail at COLLECTION
# (an error, not a skip) and take the whole run down with them.
#
# Rather than edit the tests -- they are correct as written, and rewriting a
# test to accommodate packaging is backwards -- the modules are dropped from
# collection only when their extra is genuinely absent. With the ``ml`` extra
# installed this mapping resolves to an empty ignore list and collection is
# exactly what it has always been.
#
# tests/test_optional_extras.py holds this honest: it re-derives the set of test
# modules that hard-import an optional package and fails if this mapping has
# drifted, so a newly added torch test cannot silently vanish from CI.
_OPTIONAL_MODULE_REQUIREMENTS: dict[str, str] = {
    "tests/test_ablation_encoders.py": "torch",
    "tests/test_model.py": "torch",
}


def _module_available(name: str) -> bool:
    """True if ``name`` can be imported WITHOUT importing it.

    ``find_spec`` is deliberate. Importing jax here would create the process's
    jax module object before ``simulator.lle_solver`` gets to set
    ``jax_enable_x64`` (lle_solver.py:29), and jax bakes that flag in at
    array-creation time. Nothing in this file may perturb that ordering.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


collect_ignore = [
    path
    for path, requirement in _OPTIONAL_MODULE_REQUIREMENTS.items()
    if not _module_available(requirement)
]


def _toolchain_versions() -> dict[str, str]:
    """Versions of the libraries that decide bit-level reproducibility.

    Read from distribution metadata rather than by importing the packages, for
    the ``_module_available`` reason above: this runs at session start, before
    any test has imported the solver.
    """
    import platform
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version()}
    for dist in ("jax", "jaxlib", "numpy", "scipy", "h5py"):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            versions[dist] = "<not installed>"
    return versions


def pytest_sessionstart(session: pytest.Session) -> None:
    """Print the toolchain once per session, at the top of every log.

    Written straight to the terminal reporter rather than via
    ``pytest_report_header`` or a print inside a fixture, because neither of
    those is actually visible where it matters. CI runs ``pytest -q``, and under
    ``-q`` pytest does not call the report-header hook at all, while fixture
    output goes through capture. Only this survives, and at every verbosity.

    Worth the trouble because when ``test_all_off_bit_identical_to_golden``
    drifts, the first question is always "against which jaxlib?" -- and the
    answer should already be in the log rather than needing a re-run to obtain.
    """
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # e.g. -p no:terminal
        return

    versions = _toolchain_versions()
    reporter.write_line(
        "toolchain: " + "  ".join(f"{k}={v}" for k, v in versions.items())
    )
    if collect_ignore:
        missing = ", ".join(
            f"{path} (needs {_OPTIONAL_MODULE_REQUIREMENTS[path]})"
            for path in collect_ignore
        )
        reporter.write_line(f"not collected, optional extra absent: {missing}")


@pytest.fixture(scope="session")
def toolchain_versions() -> dict[str, str]:
    """The same mapping ``pytest_report_header`` prints, for tests that assert on it."""
    return _toolchain_versions()


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
