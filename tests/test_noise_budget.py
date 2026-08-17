"""Unit tests for the noise-budget engine (``analysis/noise_budget.py``).

These are the invariants that decide whether a budget TABLE means anything,
tested without running the solver:

* the channel sets are the specified ones, and the grouped delta_T channels
  carry ``trn`` (without which they are identically zero);
* ``thermal_feedback`` is pinned across every set, so OAT/LOO differences are
  not measuring the deterministic thermo-optic ODE;
* a sidecar written for a channel set actually RESOLVES back to that channel set
  through the solver's own precedence chain -- the failure this guards against
  is silent, and produces a complete, plausible table for the wrong physics;
* the Welch segmentation refuses to report a frequency the record cannot
  observe, rather than clamping an interpolation to the nearest observed bin;
* the derived OAT / LOO / interaction-residual arithmetic.

The end-to-end run is exercised by ``python analysis/noise_budget.py --quick``
(gate G4), not here.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from analysis.noise_budget import (
    CHANNEL_SETS,
    OAT_GROUPING,
    OBSERVABLES,
    PROBE_MUS,
    S_REP_FREQS_HZ,
    BudgetError,
    RecordContext,
    _bootstrap_ci,
    _derive,
    _psd_at,
    _summarise,
    budget_sidecar,
    channel_sets,
    obs_step_jitter,
    obs_u_int_rms_frac,
    welch_segmentation,
)
from simulator.lle_solver import _resolve_noise_flags
from simulator.noise_config import NoiseConfig

T_R = 4.065040650406504e-11
FS = 1.0 / T_R


# ---------------------------------------------------------------------------
# Channel sets
# ---------------------------------------------------------------------------
def test_channel_set_names_are_the_specified_thirteen():
    expected = {
        "all_off", "quantum", "trn", "pyro_eo", "fsr", "dT_family",
        "pump_fn", "pump_rin", "all_on",
        "loo_quantum", "loo_dT_family", "loo_pump_fn", "loo_pump_rin",
    }
    assert set(CHANNEL_SETS) == expected


def test_all_off_has_no_stochastic_channel():
    assert CHANNEL_SETS["all_off"].enabled_channels == ()


def test_all_on_has_every_stochastic_channel():
    nc = CHANNEL_SETS["all_on"]
    assert set(nc.enabled_channels) == set(NoiseConfig.all_on().enabled_channels)


@pytest.mark.parametrize("name", ["pyro_eo", "fsr", "dT_family"])
def test_delta_t_channels_carry_trn(name):
    """pyro_eo and fsr are driven by the trn delta_T realization.

    Without ``trn`` they are identically zero -- ``TotalNoise.__init__`` warns
    about exactly this -- so a row that ran them "alone" would be measuring
    nothing while looking like a measurement.
    """
    assert CHANNEL_SETS[name].trn is True


def test_running_a_delta_t_channel_without_trn_would_warn():
    """Pins the upstream behaviour the grouping exists to avoid."""
    from simulator.noise_models import TotalNoise, _load_config

    cfg = _load_config()
    with pytest.warns(UserWarning, match="identically zero FSR noise"):
        TotalNoise(cfg, noise_config=NoiseConfig.all_off(fsr=True, trn=False))


def test_thermal_feedback_is_pinned_on_in_every_set():
    """Otherwise all_on - all_off would include the deterministic thermal ODE."""
    assert all(nc.thermal_feedback for nc in CHANNEL_SETS.values())
    # And the bare constructors this guards against really do disagree:
    assert NoiseConfig.all_off().thermal_feedback is False
    assert NoiseConfig.all_on().thermal_feedback is True


def test_loo_sets_are_all_on_minus_exactly_one_group():
    from analysis.noise_budget import _OAT_SPEC

    on = set(CHANNEL_SETS["all_on"].enabled_channels)
    for target in OAT_GROUPING:
        missing = on - set(CHANNEL_SETS[f"loo_{target}"].enabled_channels)
        assert missing == set(_OAT_SPEC[target]), target


def test_oat_grouping_is_non_overlapping():
    from analysis.noise_budget import _OAT_SPEC

    seen: set[str] = set()
    for g in OAT_GROUPING:
        channels = set(_OAT_SPEC[g])
        assert not (channels & seen), f"{g} overlaps an earlier group"
        seen |= channels


def test_every_set_has_a_distinct_hash():
    hashes = {name: nc.sha256() for name, nc in CHANNEL_SETS.items()}
    # pyro_eo and trn differ as CONFIGS even though this device makes them
    # produce identical output; the budget relies on that to prove CRN works.
    assert len(set(hashes.values())) == len(hashes)


def test_channel_sets_is_a_fresh_mapping_each_call():
    a, b = channel_sets(), channel_sets()
    assert a == b and a is not b


# ---------------------------------------------------------------------------
# Sidecars: the precedence trap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CHANNEL_SETS))
def test_sidecar_resolves_back_to_its_channel_set(tmp_path, name):
    """Every field of the written sidecar resolves to the requested config.

    ``_resolve_noise_flags`` ranks an explicit legacy switch in
    ``physical_parameters`` ABOVE the top-level ``noise:`` block, so a sidecar
    that left ``pump_noise_enabled`` in place would silently overrule the block
    for both pump channels at once.
    """
    import yaml

    nc = CHANNEL_SETS[name]
    path = budget_sidecar(nc, tmp_path / f"{name}.yaml")
    physical = (yaml.safe_load(path.read_text(encoding="utf-8"))
                or {})["physical_parameters"]
    resolved = _resolve_noise_flags(physical, path, None)
    assert resolved == nc


def test_sidecar_drops_the_conflicting_legacy_switches(tmp_path):
    import yaml

    path = budget_sidecar(CHANNEL_SETS["pump_fn"], tmp_path / "p.yaml")
    physical = yaml.safe_load(path.read_text(encoding="utf-8"))["physical_parameters"]
    for key in ("quantum_noise_enabled", "pump_noise_enabled", "fsr_noise_enabled"):
        assert key not in physical


def test_sidecar_can_express_pump_fn_without_pump_rin(tmp_path):
    """The legacy schema cannot; the whole point of dropping those keys."""
    import yaml

    path = budget_sidecar(CHANNEL_SETS["pump_fn"], tmp_path / "p.yaml")
    physical = yaml.safe_load(path.read_text(encoding="utf-8"))["physical_parameters"]
    resolved = _resolve_noise_flags(physical, path, None)
    assert resolved.pump_freq_noise is True
    assert resolved.pump_rin is False


def test_sidecar_carries_the_ecdl_amplitudes(tmp_path):
    """Zero amplitudes would make pump_fn/pump_rin structural zeros."""
    import yaml

    from analysis.noise_validation_campaign import (
        ECDL_H0, ECDL_HM1, ECDL_RIN_FLOOR_DBC, KG_GEOMETRY,
    )

    path = budget_sidecar(CHANNEL_SETS["all_on"], tmp_path / "a.yaml")
    pp = yaml.safe_load(path.read_text(encoding="utf-8"))["physical_parameters"]
    assert pp["pump_freq_noise_h0_hz2_per_hz"] == pytest.approx(ECDL_H0)
    assert pp["pump_freq_noise_hm1_hz3_per_hz"] == pytest.approx(ECDL_HM1)
    assert pp["pump_rin_floor_dbc_per_hz"] == pytest.approx(ECDL_RIN_FLOOR_DBC)
    for key, value in KG_GEOMETRY.items():
        assert pp[key] == pytest.approx(value)


def test_sidecar_sets_trn_psd_model_in_physical_parameters(tmp_path):
    """noise_models reads it from there, not from the noise: block."""
    import yaml

    from analysis.noise_budget import TRN_PSD_MODEL

    path = budget_sidecar(CHANNEL_SETS["trn"], tmp_path / "t.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["physical_parameters"]["trn_psd_model"] == TRN_PSD_MODEL
    assert cfg["noise"]["trn_psd_model"] == TRN_PSD_MODEL


def test_sidecar_raises_when_the_block_is_overruled(tmp_path, monkeypatch):
    """A precedence trap must be a hard error, not a wrong table."""
    import analysis.noise_budget as nb

    monkeypatch.setattr(nb, "_LEGACY_SWITCHES_TO_DROP", ())
    with pytest.raises(BudgetError, match="does not resolve"):
        nb.budget_sidecar(CHANNEL_SETS["pump_fn"], tmp_path / "bad.yaml")


# ---------------------------------------------------------------------------
# Welch segmentation
# ---------------------------------------------------------------------------
def test_target_below_the_fourier_floor_is_refused():
    """1 kHz on an 8.13 us record is not a measurement, it is an extrapolation."""
    seg = welch_segmentation(1.0e3, 200_000, FS)
    assert seg["nperseg"] is None
    assert seg["resolved"] is False
    assert "below the record's Fourier floor" in seg["reason"]


def test_segmentation_reports_the_record_length_that_would_work():
    seg = welch_segmentation(1.0e3, 200_000, FS)
    needed = int(seg["reason"].split("t_slow >= ")[1].split(" ")[0])
    assert needed >= FS / 1.0e3


def test_resolved_target_has_margin_and_segments():
    from analysis.noise_budget import WELCH_BIN_MARGIN, WELCH_MIN_SEGMENTS

    seg = welch_segmentation(1.0e6, 20_000_000, FS)
    assert seg["resolved"] is True
    assert seg["bins_below_target"] >= WELCH_BIN_MARGIN
    assert seg["n_segments"] >= WELCH_MIN_SEGMENTS


def test_underaveraged_target_is_computed_but_flagged():
    """A number with an honest chi^2 bar beats a null, as long as it says so."""
    seg = welch_segmentation(1.0e6, 200_000, FS)
    assert seg["nperseg"] is not None
    assert seg["resolved"] is False
    assert seg["reason"]


def test_segmentation_never_exceeds_the_record():
    for n in (1_000, 20_000, 200_000, 20_000_000):
        for f_t in S_REP_FREQS_HZ:
            seg = welch_segmentation(f_t, n, FS)
            if seg["nperseg"] is not None:
                assert seg["nperseg"] <= n


def test_psd_at_refuses_to_extrapolate():
    """psd_at_offset would clamp here and return a confident wrong number."""
    from analysis.noise_metrology import psd_at_offset

    f = np.array([1e5, 1e6, 1e7])
    s = np.array([1e-3, 1e-4, 1e-5])
    assert math.isnan(_psd_at(f, s, 1e3))
    assert math.isnan(_psd_at(f, s, 1e9))
    assert _psd_at(f, s, 1e6) == pytest.approx(1e-4, rel=1e-9)
    # The upstream helper clamps instead -- this is what is being avoided.
    assert psd_at_offset(f, s, 1e3) == pytest.approx(1e-3, rel=1e-9)


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
def _ctx(**kw):
    base = dict(record="fast", t_r=T_R, kappa=1.5188e8, t_slow=1000,
                snapshot_interval=1, n_tau=256, probe_mus=PROBE_MUS,
                channel_set="trn", seed=100)
    base.update(kw)
    return RecordContext(**base)


def test_u_int_rms_frac_matches_the_definition():
    rng = np.random.default_rng(0)
    u = 1.0 + 0.01 * rng.standard_normal(4096)
    ctx = _ctx(t_slow=4096)
    values, unit, unc = obs_u_int_rms_frac({"U_int_history": u[None, :]}, ctx)
    assert unit == "dimensionless"
    assert values["value"] == pytest.approx(float(np.std(u) / np.mean(u)))
    assert unc["value"] > 0
    assert ctx.notes["value"]["resolved"] is True


def test_u_int_rms_frac_is_zero_for_a_constant_record():
    ctx = _ctx(t_slow=1024)
    values, _u, _e = obs_u_int_rms_frac(
        {"U_int_history": np.full((1, 1024), 3.0)}, ctx)
    assert values["value"] == pytest.approx(0.0)


def test_u_int_rms_frac_refuses_a_zero_mean_record():
    ctx = _ctx(t_slow=64)
    values, _u, _e = obs_u_int_rms_frac({"U_int_history": np.zeros((1, 64))}, ctx)
    assert values["value"] is None
    assert ctx.notes["value"]["resolved"] is False


def test_step_jitter_returns_one_crossing_per_level():
    det_k = np.linspace(11.0, 4.5, 14)
    count = np.array([3] * 6 + [2] * 4 + [1] * 3 + [0])
    ctx = _ctx(record="staircase", det_k=det_k, n_solitons=3)
    values, unit, unc = obs_step_jitter({"soliton_count": count}, ctx)
    assert unit == "kappa"
    assert set(values) == {"N<=2", "N<=1", "N<=0"}
    assert values["N<=2"] == pytest.approx(det_k[6])
    assert values["N<=1"] == pytest.approx(det_k[10])
    assert values["N<=0"] == pytest.approx(det_k[13])
    assert all(v > 0 for v in unc.values())


def test_step_jitter_reports_none_for_a_level_never_reached():
    """A realization that never annihilated must be DROPPED, not clamped."""
    det_k = np.linspace(11.0, 4.5, 10)
    ctx = _ctx(record="staircase", det_k=det_k, n_solitons=3)
    values, _u, _e = obs_step_jitter({"soliton_count": np.full(10, 3)}, ctx)
    assert values == {"N<=2": None, "N<=1": None, "N<=0": None}
    assert "never dropped" in ctx.notes["N<=2"]["reason"]


def test_step_jitter_uses_the_imported_w2_estimator():
    """The estimator is shared with W2, not re-derived here."""
    import analysis.noise_budget as nb
    from analysis.noise_validation_campaign import level_crossing

    assert nb.level_crossing is level_crossing


def test_step_jitter_budget_statistic_is_std():
    """The mean is the step LOCATION; the spread is the jitter."""
    assert OBSERVABLES["step_jitter"]["statistic"] == "std"
    assert all(OBSERVABLES[n]["statistic"] == "mean"
               for n in OBSERVABLES if n != "step_jitter")


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def test_error_is_formatted_on_its_own_scale():
    """Regression: scaling the error to the value's exponent gave 5067.79e-6."""
    from analysis.noise_budget import _fmt

    # Value and error differ by three decades; each must keep its own scale.
    out = _fmt(6.875e-6, 5.068e-3, tex=True)
    assert r"6.875\times10^{-6}" in out
    assert "0.0051" in out
    assert "5067" not in out  # the old bug: error scaled to the value's exponent
    # And where both are large, neither is dragged into scientific notation.
    assert _fmt(0.02881, 0.00345, tex=True) == r"$0.02881\,{\pm}\,0.0034$"


def test_fmt_renders_an_absent_value_as_a_dash():
    from analysis.noise_budget import _fmt

    assert _fmt(None, None, tex=True) == "---"
    assert _fmt(float("nan"), None, tex=False) == "—"


def test_fmt_omits_a_zero_uncertainty():
    """The deterministic all_off reference has exactly zero spread."""
    from analysis.noise_budget import _fmt

    assert _fmt(0.0, 0.0, tex=True) == "$0$"


def test_tex_braces_are_balanced_and_escaped():
    from analysis.noise_budget import _tex_set

    out = _tex_set(("quantum", "dT_family"))
    assert out.count(r"\{") == 1 and out.count(r"\}") == 1
    assert r"dT\_family" in out
    # An unescaped closing brace would end the math group early.
    assert not out.replace(r"\{", "").replace(r"\}", "").count("}")


def test_rendered_tex_has_balanced_braces():
    """A cheap structural check that the generated table would compile."""
    from analysis.noise_budget import _render_tex

    report = {
        "provenance": {"git_commit": "abc123", "generated_utc": "2026-01-01T00:00:00Z"},
        "run": {"seeds": 2},
        "cells": {"all_off": {"obs": _cell(1.0)}, "all_on": {"obs": _cell(3.0)}},
        "derived": {},
    }
    tex = _render_tex(report)
    assert tex.count("{") == tex.count("}")
    assert r"\begin{tabular}" in tex and r"\bottomrule" in tex


# ---------------------------------------------------------------------------
# Resume guard
# ---------------------------------------------------------------------------
def test_fingerprint_is_stable_for_identical_settings():
    from analysis.noise_budget import QUICK_RECORDS, QUICK_STAIRCASE, _config_fingerprint

    sets = sorted(CHANNEL_SETS)
    a = _config_fingerprint(QUICK_RECORDS, QUICK_STAIRCASE, sets)
    b = _config_fingerprint(QUICK_RECORDS, QUICK_STAIRCASE, sets)
    assert a == b and len(a) == 64


def test_fingerprint_changes_with_record_length():
    """The failure this guards: resuming across a record-length change."""
    from analysis.noise_budget import (
        QUICK_RECORDS, QUICK_STAIRCASE, RecordSpec, _config_fingerprint,
    )

    sets = sorted(CHANNEL_SETS)
    base = _config_fingerprint(QUICK_RECORDS, QUICK_STAIRCASE, sets)
    altered = dict(QUICK_RECORDS)
    altered["slow"] = RecordSpec("slow", t_slow=999_999, settle_rt=3_000,
                                 snapshot_interval=8, n_tau=256)
    assert _config_fingerprint(altered, QUICK_STAIRCASE, sets) != base


def test_fingerprint_changes_with_the_staircase_grid():
    from analysis.noise_budget import QUICK_RECORDS, QUICK_STAIRCASE, _config_fingerprint

    sets = sorted(CHANNEL_SETS)
    base = _config_fingerprint(QUICK_RECORDS, QUICK_STAIRCASE, sets)
    altered = {**QUICK_STAIRCASE, "n_steps": QUICK_STAIRCASE["n_steps"] + 2}
    assert _config_fingerprint(QUICK_RECORDS, altered, sets) != base


def test_quick_and_production_records_differ():
    from analysis.noise_budget import (
        QUICK_RECORDS, QUICK_STAIRCASE, RECORDS, STAIRCASE, _config_fingerprint,
    )

    sets = sorted(CHANNEL_SETS)
    assert (_config_fingerprint(QUICK_RECORDS, QUICK_STAIRCASE, sets)
            != _config_fingerprint(RECORDS, STAIRCASE, sets))


def test_quick_staircase_resolves_annihilation():
    """n_tau=1024 does not annihilate a 3-soliton state on this ramp.

    Measured: the count stays at 3 across the whole 11 -> 5.0 kappa range at
    n_tau = 1024, leaving the step-jitter observable with no level crossing to
    measure. The quick preset must stay at 2048.
    """
    from analysis.noise_budget import QUICK_STAIRCASE

    assert QUICK_STAIRCASE["n_tau"] >= 2048


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_summarise_reports_mean_sem_std_and_both_cis():
    sample = [1.0, 2.0, 3.0, 4.0]
    out = _summarise(sample, [100, 101, 102, 103])
    assert out["mean"] == pytest.approx(2.5)
    assert out["std"] == pytest.approx(np.std(sample, ddof=1))
    assert out["sem"] == pytest.approx(np.std(sample, ddof=1) / 2.0)
    assert out["ci95_mean"][0] <= out["mean"] <= out["ci95_mean"][1]
    assert out["n_seeds_used"] == 4
    assert out["bootstrap_resamples"] == 2000


def test_summarise_drops_non_finite_seeds_but_counts_them():
    out = _summarise([1.0, None, 3.0], [100, 101, 102])
    assert out["n_seeds_requested"] == 3
    assert out["n_seeds_used"] == 2
    assert out["seeds_used"] == [100, 102]
    assert out["mean"] == pytest.approx(2.0)
    assert out["per_seed"] == [1.0, None, 3.0]


def test_summarise_of_a_degenerate_sample_has_zero_spread():
    """The deterministic all_off reference: identical seeds, exactly no jitter."""
    out = _summarise([2.0, 2.0, 2.0], [100, 101, 102])
    assert out["std"] == pytest.approx(0.0)
    assert out["sem"] == pytest.approx(0.0)


def test_summarise_of_an_empty_sample_is_null_not_nan():
    out = _summarise([None, None], [100, 101])
    assert out["mean"] is None and out["std"] is None
    assert json.dumps(out)  # must stay JSON-serialisable


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    rng_sample = np.random.default_rng(1).normal(5.0, 1.0, 64)
    a = _bootstrap_ci(rng_sample, np.mean, np.random.default_rng(7))
    b = _bootstrap_ci(rng_sample, np.mean, np.random.default_rng(7))
    assert a == b
    assert a[0] < float(np.mean(rng_sample)) < a[1]


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
def _cell(value, sem=0.1):
    return {
        "points": {"value": {"statistics": {
            "mean": value, "sem": sem, "std": sem,
            "ci95_std": [sem * 0.5, sem * 1.5],
        }}},
    }


def _cells(**by_set):
    return {name: {"obs": _cell(v)} for name, v in by_set.items()}


def test_oat_and_loo_contributions_are_the_specified_differences():
    cells = _cells(all_off=1.0, quantum=1.5, dT_family=2.0, pump_fn=1.2,
                   pump_rin=1.1, all_on=3.0, loo_quantum=2.6)
    d = _derive(cells, "obs", "mean")
    assert d["oat_contribution"]["quantum"]["contribution"]["value"] == pytest.approx(0.5)
    assert d["oat_contribution"]["dT_family"]["contribution"]["value"] == pytest.approx(1.0)
    # LOO = X[all_on] - X[all_on minus set]
    assert d["loo_contribution"]["quantum"]["contribution"]["value"] == pytest.approx(0.4)


def test_total_contribution_is_a_per_point_mapping():
    """Regression: a scalar rebind inside the residual loop clobbered this dict."""
    cells = _cells(all_off=1.0, quantum=1.5, dT_family=2.0, pump_fn=1.2,
                   pump_rin=1.1, all_on=3.0)
    d = _derive(cells, "obs", "mean")
    assert isinstance(d["total_contribution"], dict)
    assert d["total_contribution"] == {"value": pytest.approx(2.0)}


def test_all_on_is_not_reported_as_an_oat_row():
    """X[all_on] - X[all_off] is the total, not a one-at-a-time contribution."""
    cells = _cells(all_off=1.0, quantum=1.5, dT_family=2.0, pump_fn=1.2,
                   pump_rin=1.1, all_on=3.0)
    d = _derive(cells, "obs", "mean")
    assert "all_on" not in d["oat_contribution"]
    assert "all_off" not in d["oat_contribution"]


def test_interaction_residual_matches_its_definition():
    cells = _cells(all_off=1.0, quantum=1.5, dT_family=2.0, pump_fn=1.2,
                   pump_rin=1.1, all_on=3.0)
    d = _derive(cells, "obs", "mean")
    # total 2.0; OAT parts 0.5 + 1.0 + 0.2 + 0.1 = 1.8; residual 0.2
    assert d["interaction_residual"]["value"] == pytest.approx(0.2)


def test_interaction_residual_is_zero_for_perfectly_additive_channels():
    cells = _cells(all_off=0.0, quantum=1.0, dT_family=2.0, pump_fn=3.0,
                   pump_rin=4.0, all_on=10.0)
    d = _derive(cells, "obs", "mean")
    assert d["interaction_residual"]["value"] == pytest.approx(0.0)


def test_interaction_residual_is_null_when_a_group_is_missing():
    """Better no residual than one silently computed over a partial grouping."""
    cells = _cells(all_off=1.0, quantum=1.5, pump_fn=1.2, all_on=3.0)
    d = _derive(cells, "obs", "mean")
    assert d["interaction_residual"]["value"] is None


def test_commentary_explains_the_residual_and_names_tccr():
    cells = _cells(all_off=1.0, quantum=1.5, dT_family=2.0, pump_fn=1.2,
                   pump_rin=1.1, all_on=3.0)
    d = _derive(cells, "obs", "mean")
    text = d["commentary"]
    assert "delta_T" in text
    assert "tccr" in text
    assert "expected" in text.lower()


def test_derived_uncertainties_combine_in_quadrature():
    cells = _cells(all_off=1.0, quantum=1.5, dT_family=2.0, pump_fn=1.2,
                   pump_rin=1.1, all_on=3.0)
    d = _derive(cells, "obs", "mean")
    err = d["oat_contribution"]["quantum"]["uncertainty"]["value"]
    # each cell carries sem = 0.1 (see _cell), and a difference of two.
    assert err == pytest.approx(math.sqrt(0.1 ** 2 + 0.1 ** 2))


def test_std_statistic_uses_the_bootstrap_ci_not_the_sem():
    """The SEM is the error of the MEAN; quoting it on a jitter is wrong."""
    from analysis.noise_budget import _err_of_point

    cell = _cell(2.0, sem=0.1)
    assert _err_of_point(cell, "value", "mean") == pytest.approx(0.1)
    # ci95_std spans [0.05, 0.15] in _cell -> half-width 0.05.
    assert _err_of_point(cell, "value", "std") == pytest.approx(0.05)
