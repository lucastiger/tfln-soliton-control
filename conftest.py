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


# ---------------------------------------------------------------------------
# Hardware-locked byte comparisons
# ---------------------------------------------------------------------------
# These tests assert that solver output is BYTE-IDENTICAL to an artifact
# committed to the repository. They are correct and they are worth having. They
# are also only satisfiable on the machine that produced the artifact.
#
# The evidence, from the first CI run of this workflow (PR #83):
#
#   * On python 3.11 with jax 0.10.2 / jaxlib 0.10.2 / numpy 2.4.6 -- the exact
#     toolchain recorded in tests/data/golden/*.provenance.json, reported by the
#     suite itself as `version_mismatch=None` --
#     test_euler_default_bit_identical still failed, 266155 elements differing.
#   * The `identity` job, on that same pinned toolchain, failed at 0 ULP with
#     max_abs_diff = 6.2e-19 and max_rel_diff = 1.8e-13.
#   * The same tests pass at 0 ULP on the machine the goldens came from.
#
# Same versions, same source, different bytes: XLA vectorises and reassociates
# reductions according to the CPU it finds, so a runner with different SIMD
# width produces a different -- equally valid -- rounding of the same
# computation. Bit-identity is a property of (source x toolchain x HARDWARE),
# and the third factor is not something a shared CI runner supplies.
#
# The differences are ~6 orders of magnitude below the repo's own tolerance
# (observed 6.2e-19 against ATOL 1e-13), so the physics is untouched; what fails
# is only the byte-level claim.
#
# Hence this flag, used by the `fast` and `slow` CI jobs. It changes nothing by
# default: locally, on the reference machine, these tests run and must pass.
#
# test_all_off_bit_identical_to_golden is listed for ONE parametrisation only,
# and the reason is physics rather than packaging -- see the entry below.
#
# Node IDs are matched by prefix, so an unparametrised entry covers every case
# of that test and a bracketed entry covers exactly one.
# tests/test_packaging_metadata.py checks that every entry still matches a test
# pytest actually collects, so neither a rename nor a dropped parameter can
# silently empty this list.
HARDWARE_LOCKED_NODE_IDS: tuple[str, ...] = (
    # vs tests/data/golden/*.npz, compare_to_golden(..., strict=True)
    "tests/test_thermal_integrator.py::test_euler_default_bit_identical",
    # vs tests/data/lle_singlestep_legacy_128.npy, np.array_equal
    "tests/test_dispersion_grid.py::test_n_substeps_1_matches_legacy_single_step",
    "tests/test_dispersion_grid.py::test_antialiasing_toggles_off_are_bit_identical",
    "tests/test_dispersion_grid.py::test_dispersion_validity_mask_off_identical_on_changes",
    "tests/test_dispersion_grid.py::test_fine_cadence_M1_identical_M_gt1_changes",
    # sha256 of a full solver trajectory
    "tests/test_regression_figures.py::test_default_path_golden_hash",
    # vs tests/data/golden/s1024_near.npz, allclose(atol=1e-13) -- and this one
    # is hardware-locked for a DIFFERENT reason from the five above. They fail
    # off-reference by ~1e-19, a flat rounding difference. This case fails by
    # ~1e-6 (max_rel_diff of order 1), because it is the only parameter set that
    # runs in a CHAOTIC regime, where rounding differences do not stay small.
    #
    # s1024_near is the Delta_omega = 2*kappa case. Labelling its own golden
    # trajectory with simulator.state_labeler.label_trajectory gives, over the
    # 200 snapshots: CW for the first 40, modulation instability for the next
    # 60, then chaotic (class 3) for the last 100. The other three sets are CW
    # for every snapshot of every run.
    #
    # That difference is measurable rather than asserted. Perturbing delta_omega
    # by one ULP (relative 2**-52) and re-running, max|dE| over the snapshots:
    #
    #   s256_short   3.4e-21 -> 7.2e-20     bounded, no growth
    #   s512_prod    5.0e-22 -> 2.8e-20     bounded, no growth
    #   s512_sub4    8.5e-22 -> 6.2e-20     bounded, no growth
    #   s1024_near   3.4e-20 -> 1.3e-08     11.6 decades, monotonic
    #
    # The last row is exponential. A log-linear fit over the clean band between
    # the float64 noise floor and saturation gives lambda = 1.29e-2 per round
    # trip, an e-folding every 77 of them; the curve crosses ATOL=1e-13 at
    # snapshot 85, which is where the labeller has MI giving way to chaos. On a
    # CW attractor the dynamics contract and the last bits stay put; on a
    # chaotic one they cannot.
    #
    # So for this parametrisation ATOL is not a tolerance that a different CPU
    # can be brought inside of. Any sub-ULP difference in XLA's reduction order
    # is amplified past ATOL well before the run ends, and then saturates at the
    # attractor's own scale -- the first weekly run measured max_abs_diff
    # 1.2e-06 and max_rel_diff 1.06 on E_snapshots, i.e. fully decorrelated,
    # which is what chaos does and not what a regression looks like. Widening
    # ATOL enough to pass here would have to reach order-1 relative error and
    # would assert nothing at all.
    #
    # Scoped to the one case on purpose: the other three parametrisations of
    # this test keep asserting the goldens at ATOL in `fast`, `identity` and
    # `slow`, and on the reference machine a plain `pytest --runslow` still
    # asserts all four, at 0 ULP under SOLITON_STRICT_ULP=1.
    "tests/test_noise_off_identity.py::test_all_off_bit_identical_to_golden[s1024_near]",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (skipped by default)",
    )
    parser.addoption(
        "--skip-hardware-locked",
        action="store_true",
        default=False,
        help=(
            "skip the byte-comparisons against committed artifacts "
            "(HARDWARE_LOCKED_NODE_IDS). For CI on shared runners, whose CPU is "
            "not the one the artifacts were generated on. Off by default: on the "
            "reference machine these tests run and must pass."
        ),
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

    # Byte-comparisons against committed artifacts, skipped only when asked.
    # See HARDWARE_LOCKED_NODE_IDS above for why a shared runner cannot satisfy
    # them and why that is not a physics failure.
    if config.getoption("--skip-hardware-locked"):
        skip_hw = pytest.mark.skip(
            reason="byte-identical to a committed artifact; only reproducible on "
                   "the hardware that produced it (--skip-hardware-locked)"
        )
        for item in items:
            if item.nodeid.startswith(HARDWARE_LOCKED_NODE_IDS):
                item.add_marker(skip_hw)

    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
