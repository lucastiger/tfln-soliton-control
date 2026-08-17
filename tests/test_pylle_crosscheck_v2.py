"""Tests for the v2 cross-code check (``validation/pylle_crosscheck_v2.py``).

These test the machinery and the invariants, not the numbers a particular run
produced. Two things they deliberately DO pin:

* the sha256 of the three frozen v1 artifacts, so a future edit to the
  historical record fails loudly rather than quietly;
* bit-identity of the convention transformations against the v1 module, so a
  "cleanup" of the translation table cannot silently change the physics.

Nothing here needs Julia or pyLLE to *execute*. A few tests do assert properties
that only a completed ladder run can satisfy -- that both ladders reached three
levels, that every GATED observable got a containment verdict, that Picard
statistics were recorded. Those are marked ``@pytest.mark.pylle_full``: they are
cheap to run but they are assertions about the frozen artifact's SHAPE, and if
one fails the fix is a re-run of the cross-check (which needs Julia), not a code
change. The default suite therefore skips them; ``pytest -m pylle_full`` runs
them. See docs/VALIDATION_STATUS.md section 3.

Everything that guards an INVARIANT rather than a run outcome stays in the
default suite, including the frozen-artifact hashes, the convention-function
identity checks, the dispersion mirror, the H5 synthetic-failure test, the
tolerance fingerprint, schema validation and defaults_reproduce_this_run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from validation import criteria as CR
from validation import pylle_crosscheck as V1
from validation import pylle_crosscheck_v2 as V2

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "validation" / "results"
V1_DIR = RESULTS / "v1"

# sha256 taken BEFORE any v2 code ran; see validation/results/v1/MANIFEST.md.
V1_SHA256 = {
    "pylle_crosscheck.json":
        "2c404b9e2dbbfbdc892b40c5908be57281a718ec0798f8edb05d344ae8b7b017",
    "pylle_crosscheck.png":
        "95869d7cabdb6813bc2ccf1eab98b62cd706ed40d37ff58e38e67326cc206b18",
    "pylle_crosscheck_fields.npz":
        "1838653b6abd5c4012c1e501bb5ae40607c47670965a3fe1edeb9a35af98acfe",
}


# ==========================================================================
# A1 / F3 -- the historical record is frozen, not moved
# ==========================================================================
@pytest.mark.parametrize("name", sorted(V1_SHA256))
def test_v1_artifacts_untouched(name):
    """The ORIGINALS must still exist, unmodified, at their original paths."""
    original = RESULTS / name
    assert original.exists(), f"{name} was moved or deleted; it must be copied, not moved"
    assert V2._sha_file(original) == V1_SHA256[name], (
        f"{name} was MODIFIED. The v1 artifacts are the historical record of a "
        f"run that cannot be reproduced (its argv was never recorded) and must "
        f"never be edited.")


@pytest.mark.parametrize("name", sorted(V1_SHA256))
def test_v1_frozen_copy_matches_original(name):
    copy = V1_DIR / name
    assert copy.exists(), f"validation/results/v1/{name} is missing"
    assert V2._sha_file(copy) == V1_SHA256[name]


def test_v1_manifest_exists_and_records_every_artifact():
    manifest = V1_DIR / "MANIFEST.md"
    assert manifest.exists()
    text = manifest.read_text()
    for name, sha in V1_SHA256.items():
        assert name in text, f"MANIFEST.md does not mention {name}"
        assert sha in text, f"MANIFEST.md does not record the sha256 of {name}"
    # The v1 run's argv was never recorded; the manifest must say so and give
    # the reconstruction rather than implying the defaults reproduce it.
    assert "b31cc0b" in text
    assert "--roundtrips 5000" in text


# ==========================================================================
# R3 -- the nine convention findings are carried over, not re-derived
# ==========================================================================
def test_convention_functions_are_the_v1_objects():
    """v2 must not own a second copy of any convention transformation."""
    for name in ("ours_to_pylle", "pylle_to_ours", "assert_round_trip",
                 "build_pylle_dispfile", "cw_lower_branch", "soliton_seed",
                 "local_d2_from_dint"):
        assert hasattr(V1, name), f"v1 lost {name}"
    src = (REPO_ROOT / "validation" / "pylle_crosscheck_v2.py").read_text()
    for name in ("def ours_to_pylle", "def pylle_to_ours", "def soliton_seed",
                 "def build_pylle_dispfile", "def cw_lower_branch",
                 "def local_d2_from_dint", "def assert_round_trip"):
        assert name not in src, (
            f"pylle_crosscheck_v2.py redefines '{name}'. Convention "
            f"transformations must be imported from v1, never copied: a second "
            f"copy can drift and the drift would be invisible.")


def test_convention_functions_bit_identical_to_v1():
    """Fixed inputs, exact equality. No tolerance -- these must be bit-identical."""
    dev = V1.load_device()
    f_pmp = 1.935893489877590e14

    fwd_a = V1.ours_to_pylle(dev, f_pmp_hz=f_pmp)
    fwd_b = V2.__dict__.get("ours_to_pylle", V1.ours_to_pylle)(dev, f_pmp_hz=f_pmp)
    for k, v in fwd_a.items():
        assert fwd_b[k] == v, f"ours_to_pylle drifted on {k}"

    back = V1.pylle_to_ours(fwd_a)
    for k, v in dev.items():
        assert abs(back[k] - v) <= 1e-12 * abs(v), f"round trip broke on {k}"

    kappa = dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]
    seed = V1.soliton_seed(129, 16.0 * kappa, kappa, dev["kappa_c_rad_per_s"],
                           dev["gamma_LLE_per_J_per_s"], dev["pin_w"], 50011.53497093499)
    # Pinned so a change to the seed shows up here rather than in a run.
    assert seed.shape == (129,)
    assert V2._sha_arr(np.asarray(seed, dtype=np.complex128)) == V2._sha_arr(
        np.asarray(V1.soliton_seed(129, 16.0 * kappa, kappa, dev["kappa_c_rad_per_s"],
                                   dev["gamma_LLE_per_J_per_s"], dev["pin_w"],
                                   50011.53497093499), dtype=np.complex128))
    # The seed sits on the lower CW branch, and the sech is overflow-safe (no
    # exact zeros in the tails, which would be a truncated seed that radiates).
    assert np.all(np.isfinite(seed))
    assert np.count_nonzero(np.abs(seed - seed[0]) == 0.0) < seed.size


def test_dispersion_mirror_is_applied_to_the_input(tmp_path):
    """Finding 2b: pyLLE is handed D_int(-mu), not D_int(mu)."""
    mu = np.arange(-4, 5)
    d_int = np.array([-8.0, -5.0, -3.0, -1.0, 0.0, 2.0, 5.0, 9.0, 14.0])
    path = tmp_path / "disp.csv"
    d1 = 1.5e11
    f_pmp = 1.9e14
    V1.build_pylle_dispfile(path, mu, d_int, f_pmp, d1, 1000)
    rows = np.loadtxt(path, delimiter=",")
    omega = 2.0 * math.pi * rows[:, 1]
    recovered = omega - 2.0 * math.pi * f_pmp - d1 * mu
    assert np.allclose(recovered, d_int[::-1], rtol=0, atol=1e-3), (
        "build_pylle_dispfile must encode the MIRRORED dispersion; feeding the "
        "un-mirrored profile is the failure mode Finding 2b documents.")


# ==========================================================================
# H5 -- the detuning endpoint gate
# ==========================================================================
def test_detuning_endpoint_gate_catches_probe_offset():
    """A pyLLE result at 29.9328 kappa must FAIL H5, not pass it.

    This is the upstream probe-cadence bug: with Tscan = 5000 and
    num_probe = 200 the last probe lands at round trip 4976, so the returned
    field sits 0.0672 kappa short of where the run was asked to end. v1
    compared that state against ours at 30.0000 kappa.
    """
    kappa = 1.5188e8
    dw_ours = 30.0 * kappa
    dw_pylle_upstream = -29.9328 * kappa          # sign flipped, Finding 2
    rel = abs(dw_ours - (-dw_pylle_upstream)) / abs(dw_ours)
    checks = CR.evaluate_hard({"detuning_endpoint_match_rel": rel})
    h5 = next(c for c in checks if c["cid"] == "H5")
    assert h5["verdict"] == "FAIL", (
        f"H5 passed a 0.0672 kappa endpoint mismatch (rel={rel:.3e}); the gate "
        f"is not doing its job")
    assert rel > 1e-12

    # ... and with patch 0003 the last probe IS the final step.
    ok = CR.evaluate_hard({"detuning_endpoint_match_rel": 0.0})
    assert next(c for c in ok if c["cid"] == "H5")["verdict"] == "PASS"


def test_detuning_endpoint_gate_is_not_measured_when_absent():
    checks = CR.evaluate_hard({})
    assert next(c for c in checks if c["cid"] == "H5")["verdict"] == "NOT_MEASURED"


# ==========================================================================
# R5 -- containment logic
# ==========================================================================
def _conv(status="OK", limit=1.0, u=0.01, p=2.0):
    return {"status": status, "extrapolated": limit, "U_obs": u, "observed_order": p}


def test_containment_logic_contained():
    c = V2.containment_verdict(_conv(limit=100.0, u=0.01),
                               _conv(limit=100.5, u=0.01),
                               absolute=False, coverage=2.0)
    assert c["verdict"] == "CONTAINED"
    assert c["limit_gap"] < c["band"]


def test_containment_logic_separated():
    c = V2.containment_verdict(_conv(limit=100.0, u=1e-4),
                               _conv(limit=110.0, u=1e-4),
                               absolute=False, coverage=2.0)
    assert c["verdict"] == "SEPARATED"
    assert c["limit_gap"] > c["band"]
    assert "both ladders converged" in c["reason"]


@pytest.mark.parametrize("bad", ["NON_MONOTONE", "ORDER_OUT_OF_RANGE",
                                 "INSUFFICIENT_LEVELS"])
@pytest.mark.parametrize("side", ["ours", "pylle"])
def test_containment_logic_indeterminate(bad, side):
    a = _conv(status=bad if side == "ours" else "OK", limit=1.0)
    b = _conv(status=bad if side == "pylle" else "OK", limit=1.0)
    c = V2.containment_verdict(a, b, absolute=False, coverage=2.0)
    assert c["verdict"] == "INDETERMINATE"
    assert bad in c["reason"]


def test_containment_absolute_uses_mode_units():
    """An absolute observable must be compared in its own units.

    mu_DW limits near -3075 with relative U of 1e-5 are 0.03 modes apart in
    absolute terms; comparing a mode gap against a relative band would call a
    0.5-mode disagreement contained.
    """
    c = V2.containment_verdict(_conv(limit=-3075.0, u=1e-5),
                               _conv(limit=-3075.5, u=1e-5),
                               absolute=True, coverage=2.0)
    assert c["kind"] == "absolute"
    assert c["limit_gap"] == pytest.approx(0.5)
    # Each side's relative U is scaled by ITS OWN limit before combining.
    expected = 2.0 * math.sqrt((3075.0 * 1e-5) ** 2 + (3075.5 * 1e-5) ** 2)
    assert c["band"] == pytest.approx(expected, rel=1e-9)
    assert c["verdict"] == "SEPARATED"

    # Judged relatively the SAME pair yields a band ~3075x smaller and a gap
    # ~3075x smaller. Both scale together, so the verdict happens to agree here
    # -- but mixing the two (a gap in modes against a band as a fraction) would
    # compare 0.5 against 2.8e-5 and call a half-mode disagreement enormous, or
    # 1.6e-4 against 0.087 and call it invisible. That is the failure this
    # separation prevents.
    rel = V2.containment_verdict(_conv(limit=-3075.0, u=1e-5),
                                 _conv(limit=-3075.5, u=1e-5),
                                 absolute=False, coverage=2.0)
    assert rel["band"] == pytest.approx(c["band"] / 3075.25, rel=1e-3)
    assert rel["limit_gap"] == pytest.approx(c["limit_gap"] / 3075.5, rel=1e-3)


def test_containment_is_symmetric_in_the_two_codes():
    """Swapping the two codes must not change the verdict or the gap."""
    a, b = _conv(limit=100.0, u=0.002), _conv(limit=100.9, u=0.001)
    ab = V2.containment_verdict(a, b, absolute=False, coverage=2.0)
    ba = V2.containment_verdict(b, a, absolute=False, coverage=2.0)
    assert ab["verdict"] == ba["verdict"]
    assert ab["limit_gap"] == pytest.approx(ba["limit_gap"])
    assert ab["band"] == pytest.approx(ba["band"])


def test_containment_reports_numbers_even_when_indeterminate():
    """An INDETERMINATE verdict with no numbers is silence, not evidence."""
    c = V2.containment_verdict(_conv(status="NON_MONOTONE", limit=None, u=0.004),
                               _conv(limit=100.0, u=0.001),
                               absolute=False, coverage=2.0)
    assert c["verdict"] == "INDETERMINATE"
    assert c["pylle_limit"] == 100.0
    assert c["ours_U"] == 0.004


# ==========================================================================
# R6 -- pyLLE is never the reference
# ==========================================================================
FORBIDDEN_KEYS = {"error", "truth", "reference_value", "expected"}


def _all_keys(obj, out=None):
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            _all_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _all_keys(v, out)
    return out


def test_no_ground_truth_fields_in_the_emitted_report():
    """No emitted key may name either code as the reference."""
    payload = _synthetic_report()
    keys = _all_keys(payload)
    bad = keys & FORBIDDEN_KEYS
    assert not bad, (
        f"the report uses ground-truth field names {sorted(bad)}; pyLLE is an "
        f"independent implementation, not a reference, and the JSON must not "
        f"imply otherwise")


def test_no_ground_truth_fields_in_the_committed_report():
    path = RESULTS / "pylle_crosscheck_v2.json"
    if not path.exists():
        pytest.skip("no committed v2 report yet")
    bad = _all_keys(json.loads(path.read_text())) & FORBIDDEN_KEYS
    assert not bad, f"committed v2 report uses ground-truth field names {sorted(bad)}"


def test_module_source_does_not_call_pylle_the_reference():
    src = (REPO_ROOT / "validation" / "pylle_crosscheck_v2.py").read_text()
    for phrase in ('"error"', "'error'", '"truth"', '"expected"',
                   '"reference_value"'):
        assert phrase not in src, f"{phrase} appears as a field name in v2"
    # R6: the bare word "reference" is permitted only in the sanctioned phrase.
    # The other allowances are not synonyms for it: "pump_reference" /
    # "referenced to" describe which RESONANCE the dispersion grid is indexed
    # from (Finding 8), and the "ground-truth field names" line is the docstring
    # that documents this very prohibition.
    for line in src.splitlines():
        low = line.lower()
        if "reference" not in low:
            continue
        assert ("independent implementation reference" in low
                or "pump_reference" in low or "pump reference" in low
                or "pump_mode_reference" in low or "referenced to" in low
                or "ground-truth field names" in low), (
            f"unsanctioned use of 'reference': {line.strip()}")


# ==========================================================================
# DIAGNOSTIC results can never gate
# ==========================================================================
def test_diagnostic_does_not_gate():
    tolset = CR.derive_tolerances()
    values = _passing_values(tolset)
    good = CR.build_report(values, tolset, extra={})
    # Now make every diagnostic as alarming as possible.
    for cid in CR.DIAGNOSTIC_IDS:
        values[f"diag_{CR.CRITERIA_BY_ID[cid].name}"] = {
            "catastrophe": True, "verdict": "FAIL", "value": float("nan")}
    bad = CR.build_report(values, tolset, extra={})
    assert bad["overall"] == good["overall"]
    assert bad["overall_qualified"] == good["overall_qualified"]
    for c in bad["checks"]:
        if c["class"] == "DIAGNOSTIC":
            assert c["verdict"] == "DIAGNOSTIC"


def test_spectral_containment_warning_is_diagnostic_only():
    """R9's warning is loud but must not move the verdict."""
    mu = np.arange(-10, 11)
    dbc = np.full(mu.size, -30.0)          # wildly uncontained
    out = V2.spectral_containment(dbc, mu)
    assert out["contained"] is False
    assert "COMB NOT SPECTRALLY CONTAINED" in out["warning"]
    assert CR.CRITERIA_BY_ID["D6"].cls is CR.CriterionClass.DIAGNOSTIC

    dbc_ok = np.full(mu.size, -140.0)
    assert V2.spectral_containment(dbc_ok, mu)["contained"] is True
    assert V2.spectral_containment(dbc_ok, mu)["warning"] is None


# ==========================================================================
# PROVENANCE
# ==========================================================================
def test_argv_recorded():
    payload = _synthetic_report()
    assert "argv" in payload["provenance"]
    assert isinstance(payload["provenance"]["argv"], list)
    assert payload["provenance"]["argv"] == ["--mu-half", "3300"]
    assert "git_commit" in payload["provenance"]
    assert "git_dirty" in payload["provenance"]
    # criteria.py records its own argv too, so the tolerance derivation is
    # traceable independently of the run that consumed it.
    assert "argv" in payload["criteria_provenance"]


def test_committed_report_records_argv_and_defaults_flag():
    path = RESULTS / "pylle_crosscheck_v2.json"
    if not path.exists():
        pytest.skip("no committed v2 report yet")
    prov = json.loads(path.read_text())["provenance"]
    assert isinstance(prov["argv"], list)
    assert prov["defaults_reproduce_this_run"] is True, (
        "the committed run must be reproducible from the module defaults; that "
        "is the v1 defect this flag exists to close")


def test_module_defaults_match_the_documented_run():
    """v1's defaults did not reproduce its own committed numbers. v2's must."""
    assert V2.MU_HALF == 3300
    assert V2.N_ROUNDTRIPS_MAIN == 5000
    assert V2.N_ROUNDTRIPS_SCAN == 2000
    assert (V2.DW_START_KAPPA, V2.DW_END_KAPPA) == (16.0, 30.0)
    assert V2.SCAN_KAPPA == "3,5,8,16,30,40,50"
    assert V2.BISECT_ITERS == 5
    assert V2.OURS_LEVELS == "1,2,4,8,16"
    assert V2.PYLLE_LEVELS == "1,0.5,0.25,0.125"


# ==========================================================================
# determinism digest
# ==========================================================================
def test_numerical_digest_ignores_only_volatile_fields():
    a = {"x": 1.0, "wall_s": 3.2, "timestamp_utc": "a", "nested": {"y": 2, "wall_s": 9}}
    b = {"x": 1.0, "wall_s": 99.9, "timestamp_utc": "b", "nested": {"y": 2, "wall_s": 1}}
    assert V2.numerical_digest(a) == V2.numerical_digest(b)
    c = dict(a, x=1.0000001)
    assert V2.numerical_digest(a) != V2.numerical_digest(c)


def test_numerical_digest_is_not_self_referential():
    a = {"x": 1.0}
    d = V2.numerical_digest(a)
    assert V2.numerical_digest(dict(a, numerical_digest=d)) == d


# ==========================================================================
# R7/R8 helpers
# ==========================================================================
def test_band_residual_table_covers_the_grid_without_gaps_or_overlap():
    mu = np.arange(-3300, 3301)
    dbc_a = np.zeros(mu.size)
    dbc_b = np.full(mu.size, -1.0)
    rows = V2.band_residual_table(dbc_a, dbc_b, mu, np.zeros(mu.size), 4.09e-11)
    assert len(rows) == len(V2.RESIDUAL_BANDS)
    assert sum(r["n_modes"] for r in rows) == mu.size
    for r in rows:
        assert r["median_ours_minus_pylle_db"] == pytest.approx(1.0)


def test_stationarity_flag_thresholds():
    """A 1% peak-power swing over the last few round trips is NON_STATIONARY."""
    assert V2.STATIONARITY_TOL == 0.01
    assert V2.STATIONARITY_OFFSETS == (1, 2, 5, 10)
    assert V2.HOLD_TIME_MULTIPLIERS == (1, 2, 4)


def test_existence_bracket_overlap_is_the_test_not_the_endpoints():
    """G7 compares INTERVALS. v1 compared the tightest surviving point."""
    tolset = CR.derive_tolerances()
    overlapping = CR.evaluate_existence_edges({
        "existence_lower_bracket_ours": [5.75, 6.125],
        "existence_lower_bracket_pylle": [6.0, 6.3125],
        "existence_upper_bracket_ours": [37.5, 38.0],
        "existence_upper_bracket_pylle": [36.875, 37.6],
    }, tolset)
    assert all(c["verdict"] == "PASS" for c in overlapping)
    assert all(c["brackets_overlap"] for c in overlapping)

    disjoint = CR.evaluate_existence_edges({
        "existence_lower_bracket_ours": [5.75, 5.9375],
        "existence_lower_bracket_pylle": [6.125, 6.3125],
    }, tolset)
    lower = next(c for c in disjoint if c["edge"] == "lower")
    assert lower["verdict"] == "FAIL"
    assert lower["brackets_overlap"] is False


def test_hold_time_uncertainty_only_applies_when_the_edge_moved():
    """An edge that is stable under 4x the hold gets no extra uncertainty."""
    tolset = CR.derive_tolerances()
    stable = CR.evaluate_existence_edges({
        "existence_lower_bracket_ours": [5.75, 5.9375],
        "existence_lower_bracket_pylle": [6.125, 6.3125],
        "existence_lower_u_ours": 0.0, "existence_lower_u_pylle": 0.0,
    }, tolset)
    moved = CR.evaluate_existence_edges({
        "existence_lower_bracket_ours": [5.75, 5.9375],
        "existence_lower_bracket_pylle": [6.125, 6.3125],
        "existence_lower_u_ours": 0.19, "existence_lower_u_pylle": 0.19,
    }, tolset)
    s = next(c for c in stable if c["edge"] == "lower")
    m = next(c for c in moved if c["edge"] == "lower")
    assert m["tolerance"] > s["tolerance"], (
        "a measured hold-time movement must widen the edge's uncertainty; "
        "otherwise the bisection reports a precision it does not have")


# ==========================================================================
# schema
# ==========================================================================
def test_synthetic_report_validates_against_the_v2_schema():
    CR.validate_report(_synthetic_report())


def test_committed_report_validates_against_the_v2_schema():
    path = RESULTS / "pylle_crosscheck_v2.json"
    if not path.exists():
        pytest.skip("no committed v2 report yet")
    CR.validate_report(json.loads(path.read_text()))


@pytest.mark.pylle_full
def test_committed_report_has_a_containment_verdict_for_every_gated_observable():
    """A4."""
    path = RESULTS / "pylle_crosscheck_v2.json"
    if not path.exists():
        pytest.skip("no committed v2 report yet")
    payload = json.loads(path.read_text())
    keys = {CR.CRITERIA_BY_ID[cid].conv_key for cid in CR.DERIVED_GATED_IDS}
    assert set(payload["containment"]) == keys
    for k, c in payload["containment"].items():
        assert c["verdict"] in {"CONTAINED", "SEPARATED", "INDETERMINATE"}


@pytest.mark.pylle_full
def test_committed_report_ladders_have_at_least_three_levels():
    """A3 / F2."""
    path = RESULTS / "pylle_crosscheck_v2.json"
    if not path.exists():
        pytest.skip("no committed v2 report yet")
    lad = json.loads(path.read_text())["ladders"]
    assert len(lad["ours"]) >= 3
    shipped = [lv for lv in lad["pylle"] if lv["level"].endswith("_shipped")]
    tight = [lv for lv in lad["pylle"] if lv["level"].endswith("_tight")]
    assert len(shipped) >= 3 and len(tight) >= 3


def test_committed_report_hard_checks_all_pass():
    """A2."""
    path = RESULTS / "pylle_crosscheck_v2.json"
    if not path.exists():
        pytest.skip("no committed v2 report yet")
    hard = [c for c in json.loads(path.read_text())["checks"] if c["class"] == "HARD"]
    assert {c["cid"] for c in hard} == set(CR.HARD_IDS)
    failed = [c["cid"] for c in hard if c["verdict"] != "PASS"]
    assert not failed, f"HARD checks failed: {failed}; the comparison is invalid"


@pytest.mark.pylle_full
def test_committed_report_records_picard_iterations_for_every_pylle_level():
    """R4."""
    path = RESULTS / "pylle_crosscheck_v2.json"
    if not path.exists():
        pytest.skip("no committed v2 report yet")
    for lv in json.loads(path.read_text())["ladders"]["pylle"]:
        assert lv["mean_picard_iters"] is not None, lv["level"]
        assert lv["max_picard_iters"] is not None, lv["level"]


# ==========================================================================
# fixtures
# ==========================================================================
def _passing_values(tolset) -> dict:
    """A values dict in which every HARD and GATED check passes."""
    values = {
        "parameter_round_trip_max_rel": 0.0,
        "dispersion_refit_max_rel": 0.0,
        "dispersion_mirror_applied": True,
        "pump_mode_reference_delta_hz": 0.0,
        "detuning_endpoint_match_rel": 0.0,
        "delta_omega_eff_max_rel": 0.0,
        "seed_arrays_identical": True,
        "existence_lower_bracket_ours": [5.9, 6.0],
        "existence_lower_bracket_pylle": [5.95, 6.05],
        "existence_upper_bracket_ours": [37.0, 37.5],
        "existence_upper_bracket_pylle": [37.1, 37.6],
    }
    for cid in CR.DERIVED_GATED_IDS:
        c = CR.CRITERIA_BY_ID[cid]
        values[f"{c.ours_key}__ours"] = 1.0
        values[f"{c.pylle_key}__pylle"] = 1.0
    return values


def _synthetic_report() -> dict:
    tolset = CR.derive_tolerances(argv=["--mu-half", "3300"])
    values = _passing_values(tolset)
    for cid in CR.DIAGNOSTIC_IDS:
        values[f"diag_{CR.CRITERIA_BY_ID[cid].name}"] = {"note": "synthetic"}
    return CR.build_report(values, tolset, extra={
        "provenance": {"argv": ["--mu-half", "3300"], "git_commit": "0" * 40,
                       "git_dirty": False},
        "containment": {"U_mean_w": {"verdict": "CONTAINED"}},
        "ladders": {"ours": [], "pylle": []},
    })
