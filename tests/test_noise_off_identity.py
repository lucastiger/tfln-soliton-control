"""Regression suite for the deterministic (noise-off) LLE solver path.

Guards four independent failure modes of the stochastic-LLE benchmark, all of
which invalidate the noise attribution if they go unnoticed:

1. ``test_all_off_bit_identical_to_golden`` -- the integrator still reproduces
   the committed golden trajectories.
2. ``test_all_off_equals_every_channel_individually_off`` -- naming a channel
   explicitly off changes nothing relative to ``all_off()``.
3. ``test_enabling_then_disabling_is_idempotent`` -- running the noisy
   configuration in between does not contaminate the deterministic one through
   process-global state (jax config mutation, a cache keyed on the wrong thing).
4. ``test_golden_provenance_matches_current_config`` -- the goldens still
   describe the config they were produced from.

Strictness of (1) is controlled by the environment:

* ``SOLITON_STRICT_ULP=1`` -> raw-byte equality, 0 ULP;
* otherwise               -> ``np.allclose(atol=1e-13, rtol=0)``.

The loose default is deliberate. Bit-identity is a property of the *toolchain*
(jax/jaxlib/XLA are free to change reduction order and kernel fusion between
releases), so demanding it unconditionally would turn every dependency bump into
a red suite that looks like a physics regression. Set ``SOLITON_STRICT_ULP=1``
on the pinned reference environment, where 0 ULP is the real contract.

(2) and (3) are always compared bit-for-bit with no tolerance knob: they run in
the same process on the same machine within seconds of each other, so anything
other than identical bytes is a genuine defect, not toolchain noise.

Failure messages carry the array name, max abs diff, max rel diff, the first
differing FLAT index, and the jax/jaxlib/numpy versions of both sides. A bare
"arrays not equal" is worthless when this fires unattended.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from simulator.lle_solver import (
    _load_config,
    d2_to_beta2_lle,
    load_dint_grid,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
)
from simulator.noise_config import SWITCH_FIELDS, NoiseConfig
from validation.noise_off_identity import (
    ARRAY_NAMES,
    ATOL,
    CONFIG_PATH,
    DEFAULT_GOLDEN_DIR,
    DISPERSION_CSV,
    PARAM_SETS,
    REPO_ROOT,
    RTOL,
    SNAPSHOT_INTERVAL,
    compare_to_golden,
    run_deterministic,
)

GOLDEN_DIR = REPO_ROOT / DEFAULT_GOLDEN_DIR

#: Grid size above which a parameter set is considered expensive enough to hide
#: behind ``--runslow``.
SLOW_N_TAU = 1024


def _cases() -> list[pytest.param]:
    """PARAM_SETS as pytest params, with the 1024-mode case marked slow."""
    return [
        pytest.param(
            ps,
            id=str(ps["name"]),
            marks=[pytest.mark.slow] if int(ps["n_tau"]) >= SLOW_N_TAU else [],
        )
        for ps in PARAM_SETS
    ]


CASES = _cases()


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _flat_bit_diff_mask(current: np.ndarray, golden: np.ndarray) -> np.ndarray:
    """Flat elementwise mask of raw-byte differences.

    Byte-level rather than value-level so ``NaN`` payloads and the
    ``+0.0``/``-0.0`` distinction both count as differences: the contract is
    about the bits the solver emitted, not about IEEE equality.
    """
    itemsize = current.dtype.itemsize
    cur = np.ascontiguousarray(current).reshape(-1).view(np.uint8)
    gold = np.ascontiguousarray(golden).reshape(-1).view(np.uint8)
    cur = cur.reshape(current.size, itemsize)
    gold = gold.reshape(golden.size, itemsize)
    return np.any(cur != gold, axis=1)


def _diff_stats(current: np.ndarray, golden: np.ndarray) -> tuple[float, float]:
    """(max abs diff, max rel diff); NaN in both positions counts as agreement."""
    if current.size == 0:
        return 0.0, 0.0
    both_nan = np.isnan(current) & np.isnan(golden)
    abs_diff = np.where(both_nan, 0.0, np.abs(current - golden))
    denom = np.abs(golden)
    nonzero = denom > 0.0
    rel_diff = np.where(
        nonzero,
        abs_diff / np.where(nonzero, denom, 1.0),
        np.where(abs_diff > 0.0, np.inf, 0.0),
    )
    return float(np.max(abs_diff)), float(np.max(rel_diff))


def _describe_array_diff(
    name: str, current: np.ndarray, reference: np.ndarray, strict: bool
) -> str | None:
    """One diagnostic line for ``name``, or ``None`` when the arrays agree."""
    if current.shape != reference.shape or current.dtype != reference.dtype:
        return (
            f"  {name}: shape/dtype mismatch — "
            f"current {current.shape} {current.dtype} vs "
            f"reference {reference.shape} {reference.dtype}"
        )

    flat_bit_mask = _flat_bit_diff_mask(current, reference)
    if strict:
        mask = flat_bit_mask
    else:
        mask = ~np.isclose(
            current, reference, atol=ATOL, rtol=RTOL, equal_nan=True
        ).reshape(-1)
    n_diff = int(np.count_nonzero(mask))
    if n_diff == 0:
        return None

    max_abs, max_rel = _diff_stats(current, reference)
    first_flat = int(np.argmax(mask))
    first_multi = tuple(int(i) for i in np.unravel_index(first_flat, current.shape))
    return (
        f"  {name}: {n_diff}/{current.size} elements differ  "
        f"shape={current.shape}\n"
        f"      max_abs_diff      = {max_abs:.6e}\n"
        f"      max_rel_diff      = {max_rel:.6e}\n"
        f"      first_diff_flat   = {first_flat}  (multi-index {first_multi})\n"
        f"      current[first]    = {current.reshape(-1)[first_flat]!r}\n"
        f"      reference[first]  = {reference.reshape(-1)[first_flat]!r}\n"
        f"      bitwise_n_diff    = {int(np.count_nonzero(flat_bit_mask))}"
    )


def _versions_block(golden: dict[str, Any] | None, current: dict[str, Any]) -> str:
    """Both sides' jax/jaxlib/numpy versions — the #1 cause of ULP drift."""
    golden = golden or {}
    lines = ["  library versions (golden vs current):"]
    for key in ("jax", "jaxlib", "numpy", "python"):
        g = golden.get(key, "<absent>")
        c = current.get(key, "<absent>")
        flag = "" if g == c else "   <-- MISMATCH"
        lines.append(f"      {key:<7} golden={g!s:<16} current={c!s:<16}{flag}")
    return "\n".join(lines)


def _current_versions() -> dict[str, str]:
    import jaxlib  # local import: only needed to render a failure message

    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
    }


def _assert_bit_identical(
    current: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    current_label: str,
    reference_label: str,
    context: str,
) -> None:
    """Assert every array in ARRAY_NAMES is byte-for-byte equal.

    Used for same-process comparisons (tests 2 and 3), where any difference at
    all is a real defect rather than toolchain drift, so no tolerance is offered.
    """
    problems = [
        line
        for line in (
            _describe_array_diff(name, current[name], reference[name], strict=True)
            for name in ARRAY_NAMES
        )
        if line is not None
    ]
    if not problems:
        return
    versions = _current_versions()
    pytest.fail(
        f"{context}\n"
        f"  {current_label}  !=  {reference_label}   (raw-byte comparison)\n"
        + "\n".join(problems)
        + "\n  running versions: "
        + ", ".join(f"{k}={v}" for k, v in versions.items()),
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# Solver driver for an arbitrary NoiseConfig
# ---------------------------------------------------------------------------

def _solve_with(
    param_set: dict[str, Any], noise_config: NoiseConfig, seed: int = 0
) -> dict[str, np.ndarray]:
    """Run one trajectory under ``noise_config``; return the four golden arrays.

    Mirrors :func:`validation.noise_off_identity.run_deterministic` exactly,
    differing only in that the noise configuration is a parameter — that
    function hard-codes ``NoiseConfig.all_off()``, which is precisely the thing
    tests 2 and 3 need to vary.

    The duplication is pinned rather than trusted:
    :func:`test_all_off_equals_every_channel_individually_off` asserts that this
    helper called with ``all_off()`` is byte-identical to ``run_deterministic``,
    so any drift between the two setups fails loudly instead of quietly making
    the comparisons vacuous.

    Nothing is cached: memoising runs would be exactly the kind of hidden shared
    state test 3 exists to detect.
    """
    n_tau = int(param_set["n_tau"])
    physical = _load_config(str(CONFIG_PATH))
    _kappa_i, kappa_c, kappa = resolve_cavity_rates(str(CONFIG_PATH))
    beta2 = float(d2_to_beta2_lle(physical["d2_rad_per_s2"], physical["fsr_hz"]))

    solution = solve_lle_ssfm_jax(
        pin=float(physical["pin_w"]),
        # delta_omega = omega_res - omega_pump; positive = red-detuned = soliton side.
        delta_omega=float(param_set["dw_over_kappa"]) * float(kappa),
        t_slow=int(param_set["t_slow"]),
        beta=[beta2],
        kappa=float(kappa),
        kappa_c=float(kappa_c),
        rng_key=jax.random.PRNGKey(seed),
        n_tau=n_tau,
        config_path=str(CONFIG_PATH),
        snapshot_interval=SNAPSHOT_INTERVAL,
        d_int_grid=load_dint_grid(n_tau, csv_path=DISPERSION_CSV).grid,
        n_substeps=int(param_set["n_substeps"]),
        noise_config=noise_config,
    )
    return {
        "U_int_history": np.asarray(solution["U_int_history"]),
        "E_final": np.asarray(solution["e_final"]),
        "E_snapshots": np.asarray(solution["E_snapshots"]),
        "delta_T_history": np.asarray(solution["DeltaT_history"]),
    }


def _read_provenance(name: str) -> dict[str, Any]:
    path = GOLDEN_DIR / f"{name}.provenance.json"
    if not path.exists():
        pytest.fail(
            f"missing golden provenance sidecar {path}. Run "
            f"`python -m validation.noise_off_identity --write-golden` first.",
            pytrace=False,
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256_of(path: Path) -> str:
    """Independent digest, so a bug in the writer's hashing cannot self-agree."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1. Golden bit-identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param_set", CASES)
def test_all_off_bit_identical_to_golden(param_set: dict[str, Any]) -> None:
    """The noise-off trajectory still reproduces its committed golden file.

    Strict (0 ULP) when ``SOLITON_STRICT_ULP=1``, otherwise
    ``allclose(atol=1e-13, rtol=0)``.
    """
    strict = os.environ.get("SOLITON_STRICT_ULP") == "1"
    report = compare_to_golden(param_set, strict=strict, golden_dir=GOLDEN_DIR)
    if report["ok"]:
        return

    golden_versions = report["provenance_golden"].get("versions", {})
    current_versions = report["provenance_current"].get("versions", {})

    lines = [
        f"noise-off trajectory {report['name']!r} no longer matches its golden file.",
        f"  mode      : {report['mode']}"
        + ("  (SOLITON_STRICT_ULP=1)" if strict else "  (set SOLITON_STRICT_ULP=1 for 0 ULP)"),
        f"  golden    : {report['golden_npz']}",
        f"  total diff: {report['n_differences']} elements",
    ]
    for name in ARRAY_NAMES:
        rep = report["arrays"][name]
        if rep["equal"]:
            continue
        multi = rep["first_diff_index"]
        flat = (
            int(np.ravel_multi_index(tuple(multi), tuple(rep["shape"])))
            if multi is not None
            else None
        )
        lines.append(
            f"  {name}: {rep['n_diff']}/{int(np.prod(rep['shape']))} elements differ  "
            f"shape={tuple(rep['shape'])}\n"
            f"      max_abs_diff    = {rep['max_abs_diff']:.6e}\n"
            f"      max_rel_diff    = {rep['max_rel_diff']:.6e}\n"
            f"      first_diff_flat = {flat}  (multi-index {multi})\n"
            f"      bitwise_equal   = {rep['bitwise_equal']}"
        )
        if rep.get("note"):
            lines.append(f"      note: {rep['note']}")
    lines.append(_versions_block(golden_versions, current_versions))
    if report["version_mismatch"]:
        lines.append(
            "  NOTE: the toolchain moved under this golden file. XLA may change "
            "reduction order and kernel fusion between releases, so this is very "
            "likely environmental rather than a solver regression. Reproduce on "
            "the recorded versions before concluding the physics changed."
        )
    pytest.fail("\n".join(lines), pytrace=False)


# ---------------------------------------------------------------------------
# 2. Naming a channel off changes nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param_set", CASES)
def test_all_off_equals_every_channel_individually_off(
    param_set: dict[str, Any]
) -> None:
    """Explicitly switching one channel off reproduces ``all_off()`` exactly.

    Eight configurations, one per field of ``SWITCH_FIELDS`` (the seven
    stochastic channels plus ``thermal_feedback``), each with that one field
    passed as ``False`` while the rest are already ``False``.

    Also pins the local ``_solve_with`` driver to
    :func:`validation.noise_off_identity.run_deterministic`, so the two setups
    cannot silently diverge and make this comparison vacuous.
    """
    name = str(param_set["name"])
    assert len(SWITCH_FIELDS) == 8, (
        f"expected 8 switch fields, got {len(SWITCH_FIELDS)}: {SWITCH_FIELDS}. "
        f"Update this test deliberately if a channel was added or removed."
    )

    baseline = _solve_with(param_set, NoiseConfig.all_off())

    production = run_deterministic(param_set)
    _assert_bit_identical(
        baseline,
        {key: production[key] for key in ARRAY_NAMES},
        "_solve_with(all_off())",
        "run_deterministic()",
        context=(
            f"[{name}] the test's solver driver has drifted from "
            f"validation.noise_off_identity.run_deterministic; every comparison "
            f"in this test would otherwise be measured against the wrong baseline."
        ),
    )

    for field in SWITCH_FIELDS:
        config = NoiseConfig.all_off(**{field: False})
        assert config.to_dict() == NoiseConfig.all_off().to_dict(), (
            f"all_off({field}=False) should be semantically identical to all_off()"
        )
        _assert_bit_identical(
            _solve_with(param_set, config),
            baseline,
            f"all_off({field}=False)",
            "all_off()",
            context=(
                f"[{name}] naming {field!r} explicitly off perturbed the "
                f"deterministic trajectory."
            ),
        )


# ---------------------------------------------------------------------------
# 3. No contamination from a noisy run in between
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param_set", CASES)
def test_enabling_then_disabling_is_idempotent(param_set: dict[str, Any]) -> None:
    """all_off -> all_on -> all_off in one process: runs 1 and 3 must be identical.

    Catches process-global contamination that a single run can never expose:
    mutated jax configuration, a compilation or labeler cache keyed on something
    that does not actually distinguish the two configurations, host-side RNG
    state leaking between runs.
    """
    name = str(param_set["name"])
    first = _solve_with(param_set, NoiseConfig.all_off())
    noisy = _solve_with(param_set, NoiseConfig.all_on())
    third = _solve_with(param_set, NoiseConfig.all_off())

    # Guard the test's own validity: if the noisy run were somehow inert, the
    # idempotence assertion below would pass without testing anything.
    assert not np.array_equal(noisy["U_int_history"], first["U_int_history"]), (
        f"[{name}] NoiseConfig.all_on() produced a U_int_history identical to "
        f"all_off(); the intervening noisy run did nothing, so this test would "
        f"not be exercising contamination at all."
    )

    _assert_bit_identical(
        third,
        first,
        "all_off() after all_on()",
        "all_off() before all_on()",
        context=(
            f"[{name}] a noisy run in the same process changed the deterministic "
            f"result — process-global state is leaking between solver calls."
        ),
    )


# ---------------------------------------------------------------------------
# 4. Goldens still describe the config they came from
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param_set", CASES)
def test_golden_provenance_matches_current_config(param_set: dict[str, Any]) -> None:
    """The golden's recorded input digests still match the committed inputs.

    A config edit invalidates every golden trajectory. Caught here it reads as
    "the config changed"; caught only by the numeric comparison it reads as a
    mystery ULP failure and costs hours.
    """
    name = str(param_set["name"])
    provenance = _read_provenance(name)
    inputs = provenance.get("inputs", {})

    for label, path, key in (
        ("config", CONFIG_PATH, "config_sha256"),
        ("dispersion CSV", DISPERSION_CSV, "dispersion_csv_sha256"),
    ):
        recorded = inputs.get(key)
        assert recorded is not None, (
            f"[{name}] golden provenance is missing {key!r}; it cannot be "
            f"verified against the current {label}."
        )
        current = _sha256_of(path)
        assert recorded == current, (
            f"[{name}] the {label} has changed since the goldens were written.\n"
            f"  file     : {path.relative_to(REPO_ROOT)}\n"
            f"  golden   : {recorded}\n"
            f"  current  : {current}\n"
            f"Every golden trajectory is invalid until regenerated with\n"
            f"  python -m validation.noise_off_identity --write-golden\n"
            f"Regenerate deliberately — confirm the config change was intended "
            f"first, because it moves the benchmark's reference physics."
        )

    recorded_set = provenance.get("param_set", {})
    for key in ("n_tau", "t_slow", "dw_over_kappa", "n_substeps"):
        assert recorded_set.get(key) == param_set[key], (
            f"[{name}] golden was written for {key}={recorded_set.get(key)!r} but "
            f"PARAM_SETS now says {param_set[key]!r}; regenerate the goldens."
        )
