"""Tests for the convergence-aware acceptance criteria.

Each test pins one property that the v1 criteria lacked, so a regression back to
the old behaviour fails here rather than silently returning a meaningless
verdict.
"""

from __future__ import annotations

import json
import math

import pytest

from validation.criteria import (
    COVERAGE_FACTOR,
    DERIVED_GATED_IDS,
    HARD_IDS,
    CriteriaError,
    CriterionClass,
    ToleranceSet,
    Verdict,
    build_report,
    combine,
    derive_tolerances,
    evaluate_existence_edges,
    evaluate_gated,
    overall_verdict,
    rel,
    validate_report,
)


# ------------------------------------------------------------------ fixtures
def _conv_ours(tmp_path, u_map, status="OK", extrap=None):
    """Synthetic Prompt-B-shaped convergence study."""
    payload = {
        "schema_version": "1.0",
        "convergence": {k: {"status": status, "U_obs": v,
                            "extrapolated": (extrap or {}).get(k, 1.0),
                            "observed_order": 2.0, "levels": [1, 2, 3], "h": [1, .5, .25]}
                        for k, v in u_map.items()},
        "numerical_uncertainty_at_n1": dict(u_map),
        "numerical_uncertainty_at_n8": dict(u_map),
    }
    p = tmp_path / "conv_ours.json"
    p.write_text(json.dumps(payload))
    return p


def _conv_pylle(tmp_path, u_map, tag="tight", status="OK", extrap=None):
    """Synthetic Prompt-D-shaped refinement study."""
    payload = {
        "schema_version": "1.0",
        "convergence": {f"{tag}:{k}": {"status": status, "U_obs": v,
                                       "extrapolated": (extrap or {}).get(k, 1.0),
                                       "observed_order": 2.0}
                        for k, v in u_map.items()},
    }
    p = tmp_path / "conv_pylle.json"
    p.write_text(json.dumps(payload))
    return p


ALL_CONV_KEYS = ("U_mean_w", "comb_frac", "P_peak_w", "S3_modes", "mu_DW",
                 "dw_power_dbc")


def _full_tolset(tmp_path, u_o=1e-3, u_p=2e-3, **kw):
    o = _conv_ours(tmp_path, {k: u_o for k in ALL_CONV_KEYS}, **kw)
    p = _conv_pylle(tmp_path, {k: u_p for k in ALL_CONV_KEYS})
    return derive_tolerances(o, p, argv=["test"])


def _passing_values(tolset):
    """Values that make every HARD and GATED check pass."""
    v = {
        "parameter_round_trip_max_rel": 0.0,
        "dispersion_refit_max_rel": 0.0,
        "dispersion_mirror_applied": True,
        "pump_mode_reference_delta_hz": 0.0,
        "detuning_endpoint_match_rel": 0.0,
        "delta_omega_eff_max_rel": 0.0,
        "seed_arrays_identical": True,
        "existence_lower_bracket_ours": [5.0, 5.5],
        "existence_lower_bracket_pylle": [5.1, 5.6],
        "existence_upper_bracket_ours": [30.0, 31.0],
        "existence_upper_bracket_pylle": [30.2, 31.2],
    }
    for cid in DERIVED_GATED_IDS:
        from validation.criteria import CRITERIA_BY_ID
        c = CRITERIA_BY_ID[cid]
        v[f"{c.ours_key}__ours"] = 1.0
        v[f"{c.pylle_key}__pylle"] = 1.0
    return v


# ------------------------------------------------------------------ helpers
def test_rel_handles_double_zero():
    assert rel(0.0, 0.0) == 0.0
    assert rel(1.0, 1.0) == 0.0
    assert rel(1.0, 0.0) == 1.0


def test_combine_is_quadrature_with_coverage():
    assert combine(3.0, 4.0, k=1.0) == pytest.approx(5.0)
    assert combine(3.0, 4.0) == pytest.approx(2.0 * 5.0)


# ------------------------------------------------------- tolerance derivation
def test_tolerance_derived_from_convergence(tmp_path):
    """The derived tolerance must be exactly K*sqrt(U1^2 + U2^2)."""
    u1, u2 = 3e-3, 4e-3
    ts = _full_tolset(tmp_path, u_o=u1, u_p=u2)
    ent = ts.entries["G1"]            # relative, floor 1e-6, well above it
    assert ent.status == "DERIVED"
    assert ent.tolerance == pytest.approx(COVERAGE_FACTOR * math.sqrt(u1**2 + u2**2))
    assert ent.u_ours == pytest.approx(u1)
    assert ent.u_pylle == pytest.approx(u2)
    assert ent.source_ours["file"].endswith(".json")
    assert ent.source_pylle["key"].startswith("tight:")


def test_gated_tolerances_track_the_input_data(tmp_path):
    """A2, behaviourally: change the convergence input, the tolerance must move.

    This is the property that matters -- a tolerance written into the source
    could not respond to the data. Doubling both uncertainties must exactly
    double every derived (non-floored) tolerance.
    """
    a = _full_tolset(tmp_path, u_o=1e-2, u_p=2e-2)
    b = _full_tolset(tmp_path, u_o=2e-2, u_p=4e-2)
    moved = 0
    for cid in DERIVED_GATED_IDS:
        ea, eb = a.entries[cid], b.entries[cid]
        ta, tb = ea.tolerance, eb.tolerance
        if ta is None or tb is None:
            continue
        # A clipped tolerance cannot scale: at the ceiling it is capped, and at
        # the floor it is held. Both are correct behaviour, so exclude them.
        clipped = (ta >= 0.25 or tb >= 0.25
                   or ta == pytest.approx(ea.floor) or tb == pytest.approx(eb.floor))
        if clipped:
            continue
        assert tb == pytest.approx(2.0 * ta), cid
        moved += 1
    assert moved >= 3, "no derived tolerance responded to the input data"
    assert a.fingerprint != b.fingerprint


def test_no_gated_tolerance_literal_in_derivation_code():
    """A2, structurally: derive_tolerances contains no bare tolerance literal.

    Only the named constants (COVERAGE_FACTOR, the floor tables, MAX_TOL) and
    trivial structural numbers may appear; anything else would be a hand-picked
    tolerance smuggled into the derivation.
    """
    import ast
    import inspect
    import textwrap

    import validation.criteria as crit

    src = textwrap.dedent(inspect.getsource(crit.derive_tolerances))
    tree = ast.parse(src)
    allowed = {0, 1, 2, 0.0, 1.0, 2.0}     # indices, exponents, halves
    bad = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool) and n.value not in allowed]
    assert not bad, f"bare numeric literals in derive_tolerances: {bad}"


def test_v1_tolerances_preserved_but_unused():
    """The superseded dict stays in the repo, and nothing consumes it."""
    import ast
    import inspect

    import validation.criteria as crit

    assert crit.TOLERANCES_V1_HISTORICAL["spectral_span_3db"] == 0.02
    assert crit.TOLERANCES_V1_HISTORICAL["dw_peak_mode_indices"] == 0.0
    tree = ast.parse(inspect.getsource(crit))
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
            and n.id == "TOLERANCES_V1_HISTORICAL" and isinstance(n.ctx, ast.Load)]
    assert not uses, "the superseded v1 tolerances must not be read by any code path"


def test_floor_applies_when_uncertainties_are_tiny(tmp_path):
    ts = _full_tolset(tmp_path, u_o=1e-15, u_p=1e-15)
    assert ts.entries["G1"].tolerance == pytest.approx(1e-6)     # relative floor
    assert ts.entries["G3"].tolerance == pytest.approx(1e-4)


def test_underived_tolerance_demotes_and_warns(tmp_path):
    """A NON_MONOTONE study yields UNDERIVED, never a fallback number."""
    o = _conv_ours(tmp_path, {k: 1e-3 for k in ALL_CONV_KEYS}, status="NON_MONOTONE")
    p = _conv_pylle(tmp_path, {k: 1e-3 for k in ALL_CONV_KEYS})
    ts = derive_tolerances(o, p, argv=["test"])
    ent = ts.entries["G1"]
    assert ent.status == "UNDERIVED"
    assert ent.tolerance is None
    assert "NON_MONOTONE" in ent.reason

    values = _passing_values(ts)
    values["U_mean_w__ours"], values["U_mean_w__pylle"] = 1.0, 99.0   # wildly off
    res = evaluate_gated(values, ts)
    g1 = next(r for r in res if r["cid"] == "G1")
    assert g1["verdict"] == Verdict.UNDERIVED.value
    assert g1["demoted_to"] == "DIAGNOSTIC"

    report = build_report(values, ts)
    assert report["overall_qualified"] is True
    assert any("UNDERIVED" in r for r in report["overall_qualification_reasons"])


# --------------------------------------------------------------- verdict logic
def test_hard_failure_fails_overall(tmp_path):
    ts = _full_tolset(tmp_path)
    values = _passing_values(ts)
    values["parameter_round_trip_max_rel"] = 1e-3        # way above 1e-12
    report = build_report(values, ts)
    assert report["overall"] == "FAIL"
    assert any(c["cid"] == "H1" and c["verdict"] == "FAIL" for c in report["checks"])


def test_not_measured_on_gated_is_failure(tmp_path):
    ts = _full_tolset(tmp_path)
    values = _passing_values(ts)
    values["U_mean_w__ours"] = None
    report = build_report(values, ts)
    assert report["overall"] == "FAIL"
    g1 = next(c for c in report["checks"] if c["cid"] == "G1")
    assert g1["verdict"] == "NOT_MEASURED"


def test_not_measured_on_hard_is_failure(tmp_path):
    ts = _full_tolset(tmp_path)
    values = _passing_values(ts)
    values["dispersion_mirror_applied"] = None
    report = build_report(values, ts)
    assert report["overall"] == "FAIL"


def test_diagnostic_cannot_change_overall(tmp_path):
    """Every DIAGNOSTIC wildly wrong, every HARD/GATED fine -> still PASS."""
    ts = _full_tolset(tmp_path)
    values = _passing_values(ts)
    for key in ("comb_line_count_60dbc", "integer_3db_span", "integer_dw_argmax",
                "raw_peak_power", "spectral_residual_by_band", "spectral_edge_dbc",
                "dint_phase_budget"):
        values[f"diag_{key}"] = {"ours": 1, "pylle": 10**9, "absurd": True}
    report = build_report(values, ts)
    assert report["overall"] == "PASS"
    diags = [c for c in report["checks"]
             if c["class"] == CriterionClass.DIAGNOSTIC.value]
    assert len(diags) >= 7
    assert all(c["verdict"] == "DIAGNOSTIC" for c in diags)


# ------------------------------------------------------------------ fingerprint
def test_fingerprint_guard(tmp_path):
    ts = _full_tolset(tmp_path)
    values = _passing_values(ts)
    build_report(values, ts)                       # fine before mutation
    ts.entries["G1"].tolerance = 0.99              # post-hoc loosening
    with pytest.raises(CriteriaError, match="fingerprint"):
        build_report(values, ts)


def test_fingerprint_is_stable_across_identical_derivations(tmp_path):
    a = _full_tolset(tmp_path)
    b = _full_tolset(tmp_path)
    assert a.fingerprint == b.fingerprint


# ----------------------------------------------------------------------- G5
def test_dw_centroid_criterion_is_continuous(tmp_path):
    """0.49 vs 0.51 modes against a 0.50 tolerance -> PASS then FAIL."""
    from validation.criteria import CRITERIA_BY_ID
    ts = _full_tolset(tmp_path)
    ts.entries["G5"].tolerance = 0.50
    ts.entries["G5"].status = "DERIVED"
    c = CRITERIA_BY_ID["G5"]

    def diff_verdict(d):
        v = {f"{c.ours_key}__ours": -3075.0, f"{c.pylle_key}__pylle": -3075.0 + d}
        for cid in DERIVED_GATED_IDS:
            cc = CRITERIA_BY_ID[cid]
            v.setdefault(f"{cc.ours_key}__ours", 1.0)
            v.setdefault(f"{cc.pylle_key}__pylle", 1.0)
        r = next(x for x in evaluate_gated(v, ts) if x["cid"] == "G5")
        return r

    assert diff_verdict(0.49)["verdict"] == "PASS"
    assert diff_verdict(0.51)["verdict"] == "FAIL"

    # the committed case: -3074.55 vs -3073.90 differ by 0.65 modes and must be
    # judged on 0.65, NOT on int(-3075) vs int(-3074)
    v = {"mu_DW__ours": -3074.5562, "mu_DW__pylle": -3073.9043}
    for cid in DERIVED_GATED_IDS:
        cc = CRITERIA_BY_ID[cid]
        v.setdefault(f"{cc.ours_key}__ours", 1.0)
        v.setdefault(f"{cc.pylle_key}__pylle", 1.0)
    r = next(x for x in evaluate_gated(v, ts) if x["cid"] == "G5")
    assert r["diff"] == pytest.approx(0.6519, abs=1e-3)
    assert r["verdict"] == "FAIL"          # 0.65 > 0.50, on the CONTINUOUS value


def test_dw_criterion_uses_absolute_modes(tmp_path):
    ts = _full_tolset(tmp_path)
    assert ts.entries["G5"].kind == "absolute"
    assert ts.entries["G5"].floor == pytest.approx(0.05)


# ----------------------------------------------------------------------- G7
def test_existence_bracket_overlap_logic():
    ts = ToleranceSet({}, "x" * 64, {})
    overlapping = {
        "existence_lower_bracket_ours": [5.75, 6.20],
        "existence_lower_bracket_pylle": [6.10, 6.40],
        "existence_upper_bracket_ours": [37.5, 38.1],
        "existence_upper_bracket_pylle": [36.9, 37.6],
    }
    res = evaluate_existence_edges(overlapping, ts)
    assert all(r["verdict"] == "PASS" for r in res)
    assert all(r["brackets_overlap"] for r in res)

    # the committed, genuinely-resolved disagreement
    committed = {
        "existence_lower_bracket_ours": [5.75, 5.9375],
        "existence_lower_bracket_pylle": [6.125, 6.3125],
        "existence_upper_bracket_ours": [37.5, 38.125],
        "existence_upper_bracket_pylle": [36.875, 37.5],
    }
    res = evaluate_existence_edges(committed, ts)
    lower = next(r for r in res if r["edge"] == "lower")
    assert lower["brackets_overlap"] is False
    assert lower["verdict"] == "FAIL", (
        "the committed lower-edge brackets do not overlap; this disagreement is "
        "genuinely resolved and must be reported as such")
    # the upper brackets touch at 37.5 -> overlap -> PASS
    upper = next(r for r in res if r["edge"] == "upper")
    assert upper["brackets_overlap"] is True
    assert upper["verdict"] == "PASS"


def test_existence_missing_bracket_is_not_measured():
    ts = ToleranceSet({}, "x" * 64, {})
    res = evaluate_existence_edges({}, ts)
    assert all(r["verdict"] == "NOT_MEASURED" for r in res)


# --------------------------------------------------------------------- schema
def test_schema_roundtrip(tmp_path):
    ts = _full_tolset(tmp_path)
    report = build_report(_passing_values(ts), ts)
    validate_report(report)                       # must not raise
    assert report["schema_version"] == "2.0"


def test_schema_rejects_missing_hard_check(tmp_path):
    ts = _full_tolset(tmp_path)
    report = build_report(_passing_values(ts), ts)
    report["checks"] = [c for c in report["checks"] if c["cid"] != HARD_IDS[0]]
    with pytest.raises(CriteriaError, match="missing HARD"):
        validate_report(report)


def test_schema_rejects_gated_without_uncertainties(tmp_path):
    ts = _full_tolset(tmp_path)
    report = build_report(_passing_values(ts), ts)
    for c in report["checks"]:
        if c["cid"] == "G1":
            del c["u_ours"]
    with pytest.raises(CriteriaError, match="u_ours"):
        validate_report(report)


def test_schema_rejects_fingerprint_mismatch(tmp_path):
    ts = _full_tolset(tmp_path)
    report = build_report(_passing_values(ts), ts)
    report["tolerances"]["G1"]["tolerance"] = 0.42
    with pytest.raises(CriteriaError, match="fingerprint"):
        validate_report(report)


def test_schema_rejects_unknown_verdict(tmp_path):
    ts = _full_tolset(tmp_path)
    report = build_report(_passing_values(ts), ts)
    report["checks"][0]["verdict"] = "PROBABLY_FINE"
    with pytest.raises(CriteriaError):
        validate_report(report)


def test_schema_rejects_overall_disagreeing_with_checks(tmp_path):
    ts = _full_tolset(tmp_path)
    report = build_report(_passing_values(ts), ts)
    report["overall"] = "FAIL"
    with pytest.raises(CriteriaError, match="disagrees"):
        validate_report(report)


# ------------------------------------------------------------------ real data
def test_derives_against_the_committed_convergence_studies():
    """Derivation must work on the real Prompt B / Prompt D artifacts."""
    ts = derive_tolerances(argv=["test"])
    assert len(ts.entries) == len(DERIVED_GATED_IDS)
    assert len(ts.fingerprint) == 64
    # at least one must derive, and every UNDERIVED one must say why
    assert any(e.status == "DERIVED" for e in ts.entries.values())
    for e in ts.entries.values():
        if e.status == "UNDERIVED":
            assert e.tolerance is None and e.reason
        else:
            assert e.tolerance is not None and e.u_ours is not None \
                and e.u_pylle is not None


def test_overall_verdict_ignores_diagnostics_structurally():
    checks = [
        {"cid": "H1", "name": "h", "class": "HARD", "verdict": "PASS"},
        {"cid": "D1", "name": "d", "class": "DIAGNOSTIC", "verdict": "DIAGNOSTIC"},
    ]
    assert overall_verdict(checks)[0] == "PASS"
    checks.append({"cid": "D2", "name": "d2", "class": "DIAGNOSTIC",
                   "verdict": "DIAGNOSTIC"})
    assert overall_verdict(checks)[0] == "PASS"
