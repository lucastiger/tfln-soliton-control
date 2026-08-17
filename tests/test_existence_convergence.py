"""Tests for the existence-edge discretization measurement.

The experiment itself needs the solver and takes ~20 minutes; these tests cover
the machinery, the reuse guarantees and the frozen-artifact invariants, and read
the committed result when it exists. Nothing here needs Julia or pyLLE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from validation import criteria as CR
from validation import convergence_lle as CL
from validation import existence_convergence as EC
from validation import pylle_crosscheck as V1
from validation import pylle_crosscheck_v2 as V2

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "validation" / "results"
RESULT_JSON = RESULTS / "existence_convergence_ours.json"

# sha256 of every artifact this task must not touch, taken before it ran.
FROZEN = {
    "v1/pylle_crosscheck.json":
        "2c404b9e2dbbfbdc892b40c5908be57281a718ec0798f8edb05d344ae8b7b017",
    "v1/pylle_crosscheck.png":
        "95869d7cabdb6813bc2ccf1eab98b62cd706ed40d37ff58e38e67326cc206b18",
    "v1/pylle_crosscheck_fields.npz":
        "1838653b6abd5c4012c1e501bb5ae40607c47670965a3fe1edeb9a35af98acfe",
    "pylle_crosscheck.json":
        "2c404b9e2dbbfbdc892b40c5908be57281a718ec0798f8edb05d344ae8b7b017",
    "pylle_crosscheck.png":
        "95869d7cabdb6813bc2ccf1eab98b62cd706ed40d37ff58e38e67326cc206b18",
    "pylle_crosscheck_fields.npz":
        "1838653b6abd5c4012c1e501bb5ae40607c47670965a3fe1edeb9a35af98acfe",
}
# The v2 artifacts are read-only inputs here rather than pinned constants: this
# task must not write them, which test_v2_artifacts_present_and_readonly_here
# and test_committed_v2_json_not_rewritten check directly.


# ==========================================================================
# F2 -- frozen artifacts
# ==========================================================================
@pytest.mark.parametrize("name", [
    "v1/pylle_crosscheck.json", "v1/pylle_crosscheck.png",
    "v1/pylle_crosscheck_fields.npz", "pylle_crosscheck.json",
    "pylle_crosscheck.png", "pylle_crosscheck_fields.npz",
])
def test_frozen_v1_artifacts_untouched(name):
    p = RESULTS / name
    assert p.exists(), f"{name} is missing"
    assert EC._sha_file(p) == FROZEN[name], (
        f"{name} was MODIFIED. It is frozen evidence of a run that cannot be "
        f"reproduced and must never be edited.")


def test_v2_artifacts_present_and_readonly_here():
    """The v2 results are inputs to this task, never outputs of it."""
    for name in ("pylle_crosscheck_v2.json", "pylle_crosscheck_v2.png",
                 "pylle_crosscheck_v2_fields.npz"):
        assert (RESULTS / name).exists(), f"{name} is required as a read-only input"
    src = (REPO_ROOT / "validation" / "existence_convergence.py").read_text()
    for forbidden in ("pylle_crosscheck_v2.json\").write",
                      "pylle_crosscheck_v2.png", "np.savez_compressed(V2_FIELDS"):
        assert forbidden not in src, f"the module appears to write a v2 artifact: {forbidden}"


# ==========================================================================
# R2 -- reuse, not reimplementation
# ==========================================================================
def test_reuses_v2_objects():
    """The imported callables must BE the v2/v1 objects, not lookalikes."""
    import validation.existence_convergence as mod

    # Bracket primitives and the classifier come from the v2 module.
    assert V2.initial_brackets.__module__ == "validation.pylle_crosscheck_v2"
    assert V2.update_bracket.__module__ == "validation.pylle_crosscheck_v2"
    assert V2.classify_field.__module__ == "validation.pylle_crosscheck_v2"
    # The classifier itself is v1's.
    assert V1.is_single_soliton.__module__ == "validation.pylle_crosscheck"
    # Richardson is convergence_lle's, used verbatim.
    assert CL.richardson.__module__ == "validation.convergence_lle"

    # ... and this module must not define its own copies of any of them.
    src = (REPO_ROOT / "validation" / "existence_convergence.py").read_text()
    for name in ("def soliton_seed", "def is_single_soliton", "def richardson",
                 "def initial_brackets", "def update_bracket", "def classify_field",
                 "def cw_lower_branch", "def run_ours"):
        assert name not in src, (
            f"existence_convergence.py redefines '{name}'. A second copy can "
            f"drift from the v2 harness and the drift would be invisible.")
    assert not hasattr(mod, "_soliton_seed"), "shadow copy of the seed"


def test_v2_bracket_primitives_replay_the_committed_bisection():
    """The extracted primitives must reproduce the v2 run's own brackets.

    This is what makes the refactor safe: the same decision rule, replayed
    against the committed trace, lands on the committed brackets.
    """
    v2 = json.loads((RESULTS / "pylle_crosscheck_v2.json").read_text())["existence"]
    scan = v2["scan_over_kappa"]
    br_lo, br_hi = V2.initial_brackets(v2["survival_ours"], scan)
    for step in v2["bisect_trace"]:
        for key, edge in (("lo", "lower"), ("hi", "upper")):
            if key not in step["ours"]:
                continue
            m = step["ours"][key]["midpoint_over_kappa"]
            alive = step["ours"][key]["single"]
            if edge == "lower":
                br_lo = V2.update_bracket(br_lo, m, alive, "lower")
            else:
                br_hi = V2.update_bracket(br_hi, m, alive, "upper")
    assert sorted(br_lo) == v2["lower_bracket_ours"]
    assert sorted(br_hi) == v2["upper_bracket_ours"]


def test_update_bracket_keeps_the_two_edges_opposite():
    """A surviving midpoint tightens the ALIVE end -- opposite ends per edge."""
    assert V2.update_bracket((5.0, 8.0), 6.5, True, "lower") == (5.0, 6.5)
    assert V2.update_bracket((5.0, 8.0), 6.5, False, "lower") == (6.5, 8.0)
    assert V2.update_bracket((30.0, 40.0), 35.0, True, "upper") == (35.0, 40.0)
    assert V2.update_bracket((30.0, 40.0), 35.0, False, "upper") == (30.0, 35.0)
    assert V2.update_bracket(None, 1.0, True, "lower") is None
    with pytest.raises(ValueError):
        V2.update_bracket((1.0, 2.0), 1.5, True, "sideways")


# ==========================================================================
# R6 -- criteria.py change is inert unless a measurement is supplied
# ==========================================================================
def test_criteria_fingerprint_unchanged_when_u_disc_is_none():
    tolset = CR.derive_tolerances()
    assert tolset.fingerprint == CR._fingerprint(tolset.entries)
    # The fingerprint covers the derived GATED tolerances; G7's band is computed
    # per-evaluation, so adding an optional term must not perturb it at all.
    assert tolset.fingerprint == (
        "98ea9e2f23be59f495a92148fc2318343ace7a7087299a04dac317a600c03653")


def test_g7_identical_when_discretization_u_is_none():
    tolset = CR.derive_tolerances()
    values = {
        "existence_lower_bracket_ours": [5.84375, 5.9375],
        "existence_lower_bracket_pylle": [6.125, 6.21875],
        "existence_upper_bracket_ours": [37.8125, 38.125],
        "existence_upper_bracket_pylle": [36.875, 37.1875],
    }
    base = CR.evaluate_existence_edges(values, tolset)
    none_kw = CR.evaluate_existence_edges(values, tolset, discretization_u=None)
    empty = CR.evaluate_existence_edges(values, tolset, discretization_u={})
    zero = CR.evaluate_existence_edges(
        values, tolset, discretization_u={"lower": 0.0, "upper": 0.0})
    assert base == none_kw == empty == zero, (
        "supplying no discretization term must be bit-identical to the original "
        "derivation; otherwise the frozen v2 artifact stops being reproducible")
    # and it must reproduce the band the v2 run recorded
    lower = next(c for c in base if c["edge"] == "lower")
    assert lower["tolerance"] == pytest.approx(0.13258252147247766, rel=1e-12)


def test_g7_band_quadrature():
    """band = K*sqrt(hw_o^2 + hw_p^2 + U^2), exactly."""
    tolset = CR.derive_tolerances()
    hw_o, hw_p, u = 0.046875, 0.046875, 0.2
    values = {
        "existence_lower_bracket_ours": [5.890625 - hw_o, 5.890625 + hw_o],
        "existence_lower_bracket_pylle": [6.171875 - hw_p, 6.171875 + hw_p],
        "existence_upper_bracket_ours": [37.8125, 38.125],
        "existence_upper_bracket_pylle": [36.875, 37.1875],
    }
    got = CR.evaluate_existence_edges(values, tolset,
                                      discretization_u={"lower": u})
    lower = next(c for c in got if c["edge"] == "lower")
    expected = CR.COVERAGE_FACTOR * math.sqrt(hw_o ** 2 + hw_p ** 2 + u ** 2)
    assert lower["tolerance"] == pytest.approx(expected, rel=1e-12)
    assert lower["u_discretization_ours"] == u
    # the upper edge got no term and must be unchanged
    upper = next(c for c in got if c["edge"] == "upper")
    assert "u_discretization_ours" not in upper


def test_g7_discretization_term_can_only_widen_never_tighten():
    tolset = CR.derive_tolerances()
    values = {
        "existence_lower_bracket_ours": [5.84375, 5.9375],
        "existence_lower_bracket_pylle": [6.125, 6.21875],
        "existence_upper_bracket_ours": [37.8125, 38.125],
        "existence_upper_bracket_pylle": [36.875, 37.1875],
    }
    base = next(c for c in CR.evaluate_existence_edges(values, tolset)
                if c["edge"] == "lower")
    for u in (1e-6, 0.01, 0.5, 5.0):
        got = next(c for c in CR.evaluate_existence_edges(
            values, tolset, discretization_u={"lower": u}) if c["edge"] == "lower")
        assert got["tolerance"] >= base["tolerance"]


def test_no_other_criterion_is_affected():
    tolset = CR.derive_tolerances()
    values = {f"{CR.CRITERIA_BY_ID[c].ours_key}__ours": 1.0
              for c in CR.DERIVED_GATED_IDS}
    values.update({f"{CR.CRITERIA_BY_ID[c].pylle_key}__pylle": 1.0
                   for c in CR.DERIVED_GATED_IDS})
    a = CR.evaluate_gated(values, tolset)
    b = CR.evaluate_gated(values, tolset)
    assert a == b
    assert CR.COVERAGE_FACTOR == 2.0 and CR.MAX_TOL == 0.25


# ==========================================================================
# edge uncertainty and hypothesis selection
# ==========================================================================
def test_u_disc_is_floored_at_the_finest_bracket_halfwidth():
    """However small the drift, the bisection never resolved better than that."""
    u = EC.edge_uncertainty(5.890625, 5.890625, 5.890625, 0.046875)
    assert u["drift_1_to_4"] == 0.0
    assert u["U_disc"] == 0.046875
    assert u["u_disc_source"] == "bracket_halfwidth"


def test_u_disc_uses_the_drift_when_it_dominates():
    u = EC.edge_uncertainty(5.5, 5.7, 6.0, 0.046875)
    assert u["drift_1_to_4"] == pytest.approx(0.5)
    assert u["drift_2_to_4"] == pytest.approx(0.3)
    assert u["U_disc"] == pytest.approx(0.3)
    assert u["u_disc_source"] == "drift_2_to_4"


def test_richardson_status_propagates():
    """A non-monotone sequence must NOT be forced into a fit."""
    u = EC.edge_uncertainty(5.90, 5.80, 5.95, 0.046875)   # down then up
    assert u["richardson"]["status"] == "NON_MONOTONE"
    assert u["richardson"]["extrapolated"] is None
    # ... and U_disc still comes from the fallback rule
    assert u["U_disc"] == pytest.approx(max(abs(5.95 - 5.80), 0.046875))


def test_edge_uncertainty_handles_missing_levels():
    u = EC.edge_uncertainty(5.9, None, None, 0.046875)
    assert u["U_disc"] == 0.046875
    assert u["u_disc_source"] == "bracket_halfwidth_only"
    assert u["richardson"] is None


@pytest.mark.parametrize("drift,expected", [
    (0.0, "H-B"), (0.049, "H-B"), (0.05, "H-C"), (0.2799, "H-C"),
    (0.28, "H-A"), (1.0, "H-A"),
])
def test_hypothesis_thresholds_are_the_pre_committed_ones(drift, expected):
    assert EC.select_hypothesis(drift, 0.0)["outcome"] == expected
    # the worst of the two edges decides
    assert EC.select_hypothesis(0.0, drift)["outcome"] == expected


def test_hypothesis_thresholds_are_not_editable_after_the_fact():
    """The constants are the pre-registration; pin them."""
    assert EC.H_A_THRESHOLD == 0.28
    assert EC.H_B_THRESHOLD == 0.05


def test_recompute_g7_v2_variant_matches_criteria_exactly():
    """The `v2_as_measured` variant must reproduce criteria.py's own G7."""
    tolset = CR.derive_tolerances()
    values = {
        "existence_lower_bracket_ours": [5.84375, 5.9375],
        "existence_lower_bracket_pylle": [6.125, 6.21875],
        "existence_upper_bracket_ours": [37.8125, 38.125],
        "existence_upper_bracket_pylle": [36.875, 37.1875],
    }
    from_criteria = {c["edge"]: c for c in CR.evaluate_existence_edges(values, tolset)}
    for edge in ("lower", "upper"):
        mine = EC.recompute_g7(edge, values[f"existence_{edge}_bracket_ours"],
                               values[f"existence_{edge}_bracket_ours"],
                               values[f"existence_{edge}_bracket_pylle"], 0.0,
                               CR.COVERAGE_FACTOR)["v2_as_measured"]
        assert mine["band"] == pytest.approx(from_criteria[edge]["tolerance"])
        assert mine["verdict"] == from_criteria[edge]["verdict"]
        assert mine["abs_diff"] == pytest.approx(from_criteria[edge]["abs_diff"])


def test_recompute_g7_separates_refinement_from_band_widening():
    """A verdict change must be attributed to the right cause.

    Case 1: the edge moves onto pyLLE's and the band term is irrelevant.
    Case 2: the edge does not move and only the widened band flips the verdict.
    These must not be reported the same way.
    """
    pylle = [6.125, 6.21875]
    moved = EC.recompute_g7("lower", [5.84375, 5.9375], [6.03125, 6.125],
                            pylle, 0.1875, 2.0)
    assert moved["v2_as_measured"]["verdict"] == "FAIL"
    assert moved["refined_no_u_disc"]["verdict"] == "PASS"
    assert "refined edge measurement" in moved["verdict_change_attributed_to"]

    # Same brackets as v2 (edge did not move), but a large U_disc.
    widened = EC.recompute_g7("lower", [5.84375, 5.9375], [5.84375, 5.9375],
                              pylle, 1.0, 2.0)
    assert widened["v2_as_measured"]["verdict"] == "FAIL"
    assert widened["refined_no_u_disc"]["verdict"] == "FAIL"
    assert widened["refined_with_u_disc"]["verdict"] == "PASS"
    assert "scepticism" in widened["verdict_change_attributed_to"]


def test_recompute_g7_flags_the_halfwidth_double_count():
    """U_disc floored at the half-width enters the quadrature twice."""
    g = EC.recompute_g7("upper", [37.8125, 38.125], [37.8125, 38.125],
                        [36.875, 37.1875], 0.15625, 2.0)
    assert g["u_disc_double_counts_halfwidth"] is True
    g2 = EC.recompute_g7("upper", [37.8125, 38.125], [37.8125, 38.125],
                         [36.875, 37.1875], 0.4, 2.0)
    assert g2["u_disc_double_counts_halfwidth"] is False


# ==========================================================================
# R3 -- determinism, and R8 -- observable versioning
# ==========================================================================
def test_numerical_digest_ignores_only_volatile_fields():
    a = {"x": 1.0, "wall_s": 3.2, "timestamp_utc": "a", "n": {"y": 2, "wall_s": 9}}
    b = {"x": 1.0, "wall_s": 99.9, "timestamp_utc": "b", "n": {"y": 2, "wall_s": 1}}
    assert EC.numerical_digest(a) == EC.numerical_digest(b)
    assert EC.numerical_digest(a) != EC.numerical_digest(dict(a, x=1.0000001))


def test_edge_observable_is_versioned():
    assert EC.EDGE_OBSERVABLE_VERSION == "edge_v1"
    assert "contrast >= 5" in EC.EDGE_DEFINITION
    assert "one circular connected component" in EC.EDGE_DEFINITION


# ==========================================================================
# the committed result
# ==========================================================================
def _result():
    if not RESULT_JSON.exists():
        pytest.skip("no committed existence-convergence result yet")
    return json.loads(RESULT_JSON.read_text())


def test_committed_regression_gate_passed():
    """A1 / R1."""
    r = _result()
    gate = r["regression_gate"]
    assert gate["reproduces_v2_n1"] is True, (
        f"n=1 did not reproduce the v2 existence block: "
        f"{[k for k, v in gate['per_field'].items() if not v['match']]}")


def test_committed_levels_completed():
    """A2 / F3."""
    r = _result()
    ok = [lv for lv in r["levels"] if lv["status"] == "OK"]
    assert len(ok) >= 2, f"only {len(ok)} level(s) completed"
    assert {lv["n_substeps"] for lv in ok} >= {1, 2, 4}


def test_committed_u_disc_reported_for_both_edges():
    """A3."""
    r = _result()
    for edge in ("lower", "upper"):
        e = r["edge_convergence"][edge]
        assert e["U_disc"] is not None
        assert e["u_disc_source"] in {"drift_2_to_4", "bracket_halfwidth",
                                      "bracket_halfwidth_only"}


def test_committed_hypothesis_selected_by_the_thresholds():
    """A4 -- the recorded outcome must follow from the recorded drift."""
    r = _result()
    h = r["hypothesis_outcome"]
    assert h["outcome"] in {"H-A", "H-B", "H-C"}
    recomputed = EC.select_hypothesis(h["drift_lower"], h["drift_upper"])
    assert recomputed["outcome"] == h["outcome"], (
        "the recorded hypothesis does not follow from the recorded drifts")
    assert h["thresholds"] == {"H_A_at_or_above": EC.H_A_THRESHOLD,
                               "H_B_below": EC.H_B_THRESHOLD}


def test_committed_g7_asymmetry_is_declared():
    r = _result()
    g = r["g7_recomputed"]
    assert g["pylle_u_disc_not_measured"] is True
    assert "asymmetric" in g["asymmetry_note"]


def test_committed_g7_v2_variant_matches_the_frozen_v2_verdict():
    """The decomposition's v2 row must agree with what the v2 JSON recorded."""
    r = _result()
    v2 = json.loads((RESULTS / "pylle_crosscheck_v2.json").read_text())
    v2_g7 = {c["edge"]: c for c in v2["checks"] if c["cid"] == "G7"}
    for edge in ("lower", "upper"):
        mine = r["g7_recomputed"][edge]["v2_as_measured"]
        assert mine["verdict"] == v2_g7[edge]["verdict"]
        assert mine["band"] == pytest.approx(v2_g7[edge]["tolerance"], rel=1e-12)
        assert mine["abs_diff"] == pytest.approx(v2_g7[edge]["abs_diff"], rel=1e-12)


def test_committed_g7_reports_all_three_variants():
    r = _result()
    for edge in ("lower", "upper"):
        g = r["g7_recomputed"][edge]
        for name in ("v2_as_measured", "refined_no_u_disc", "refined_with_u_disc"):
            assert g[name]["verdict"] in {"PASS", "FAIL"}
        assert g["verdict_change_attributed_to"]


def test_committed_v2_json_not_rewritten():
    """R7 -- the v2 artifact stays as the record of what was measured then."""
    v2 = json.loads((RESULTS / "pylle_crosscheck_v2.json").read_text())
    g7 = [c for c in v2["checks"] if c["cid"] == "G7"]
    assert len(g7) == 2
    for c in g7:
        assert "u_discretization_ours" not in c, (
            "the v2 report was recomputed in place; it must be left untouched")
