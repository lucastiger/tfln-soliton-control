"""The next-phase scaffold must not drift away from production, and must not run.

Two failure modes this guards:

* the scaffold quietly characterizing a configuration nobody uses, because
  production moved and the scaffold restated its flags as literals;
* someone wiring it into CI and burning real compute on a study that was never
  agreed to.
"""

from __future__ import annotations

import pytest

from analysis.dks_access import N_TAU, OPERATING_DW_KAPPA, PRODUCTION_NUMERICS
from validation import production_config_convergence as PCC


def test_module_imports():
    assert PCC.PRODUCTION_OPERATING_POINT is not None


def test_main_raises_systemexit_and_does_not_run_anything():
    with pytest.raises(SystemExit) as exc:
        PCC.main()
    msg = str(exc.value)
    assert "SCAFFOLD" in msg and "NOT" in msg
    assert "docs/VALIDATION_STATUS.md" in msg


def test_declared_flags_match_production_exactly():
    """R6: the scaffold cannot drift away from production silently."""
    assert PCC.PRODUCTION_OPERATING_POINT.solver_flags == PRODUCTION_NUMERICS, (
        "the scaffold's solver flags have diverged from "
        "analysis.dks_access.PRODUCTION_NUMERICS; it would characterize a "
        "configuration nobody runs")


def test_declared_operating_point_matches_production():
    op = PCC.PRODUCTION_OPERATING_POINT
    assert op.delta_omega_over_kappa == float(OPERATING_DW_KAPPA)
    assert N_TAU in op.n_tau_ladder, (
        "the production n_tau must be the first rung of the ladder, otherwise "
        "the study never measures the configuration in use")


def test_ladders_start_at_production_settings():
    """Refining below production would characterize something nobody runs."""
    op = PCC.PRODUCTION_OPERATING_POINT
    assert min(op.n_substeps_ladder) == PRODUCTION_NUMERICS["n_substeps"]
    assert min(op.n_tau_ladder) == N_TAU
    assert len(op.n_substeps_ladder) >= 3, "a Richardson fit needs three levels"
    assert len(op.n_tau_ladder) >= 3


def test_observables_are_the_frozen_ones():
    """A new observable here would make the study incomparable with the frozen studies."""
    op = PCC.PRODUCTION_OPERATING_POINT
    assert op.observable_source == "validation.convergence_lle.observables_v2"
    from validation.convergence_lle import observables_v2
    assert callable(observables_v2)


def test_the_two_questions_are_stated():
    op = PCC.PRODUCTION_OPERATING_POINT
    assert len(op.questions) == 2
    joined = " ".join(op.questions).lower()
    assert "edge absorber" in joined
    assert "n_tau" in joined and "dealias" in joined


def test_dealias_cutoff_puts_both_dws_outside_the_scan_grid():
    """The measured fact that motivates question 2.

    At the production SCAN grid the 2/3 dealias zeroes both dispersive-wave
    crossings. If this ever stops being true the question is answered and the
    study's motivation changes -- so it is asserted rather than asserted-in-prose.
    """
    dw_red, dw_blue = 3038, 3239
    assert PCC.dealias_cutoff(8192) == 2730
    assert PCC.dealias_cutoff(8192) < dw_red < dw_blue, (
        "both DW crossings should sit above the 8192-grid dealias cutoff")
    # the settle grid does resolve them
    assert PCC.dealias_cutoff(16384) == 5461
    assert PCC.dealias_cutoff(16384) > dw_blue


def test_scaffold_produced_no_artifact():
    """Nothing under validation/results/ may have come from this module."""
    from pathlib import Path
    results = Path(__file__).resolve().parents[1] / "validation" / "results"
    for p in results.rglob("*.json"):
        text = p.read_text()
        assert "production_config_convergence" not in text, (
            f"{p.name} references the scaffold; it must not have been run")
