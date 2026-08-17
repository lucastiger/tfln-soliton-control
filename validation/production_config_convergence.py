"""SCAFFOLD ONLY -- the production-configuration convergence study. NOT RUN.

This module declares a study and refuses to execute it. ``main()`` raises
``SystemExit``. Nothing here has produced a number, and no artifact in
``validation/results/`` came from it.

WHY THIS STUDY IS THE NEXT PRIORITY
-----------------------------------
Four layers of validation exist (``docs/VALIDATION_STATUS.md`` section 1) and all
four sit at the SAME configuration: ``delta_omega = 30 kappa``, ``n_tau = 6601``,
every numerical mask off, ``n_substeps`` 1..16. **No production artifact uses
that configuration.** Production runs at ``delta_omega = 10 kappa``,
``n_tau = 8192`` or ``16384``, with the 2/3 dealias and the edge absorber ON, at
``n_substeps = 4``.

Turning the masks off was the right call for the work that has been done -- pyLLE
has neither a dealias nor an absorber, so leaving ours on would have compared two
different problems, and the order-of-accuracy study had to measure the shipped
scheme rather than a masked variant. But the consequence is that the quantities
the production artifacts quote carry **no uncertainty budget at all**.

THE TWO QUESTIONS THIS STUDY MUST ANSWER
----------------------------------------
1. **Does the edge absorber measurably alter DW power at production settings,
   and by how much?**
   The absorber is ON in production and absent from every validated
   configuration. It acts in the spectral wings, which is exactly where the
   dispersive waves live. The answer must be a number in dB with an uncertainty,
   not a reassurance. Method: run the ladder twice, once with
   ``edge_absorber=True`` and once ``False``, everything else identical, and
   report the difference in ``dw_power_dbc`` against the discretization
   uncertainty of each.

2. **Are the artifacts that quote DW physics converged in ``n_tau``?**
   At ``n_tau = 8192`` the 2/3 dealias zeroes every mode with ``|mu| > 2730``.
   The dispersive-wave phase-matching crossings for this device sit at
   ``|mu| ~ 3038`` (red) and ``|mu| ~ 3239`` (blue) -- measured, see
   ``pylle_crosscheck_v2.json::spectral_containment.blue_dw_phase_match_mu``.
   **Both DWs lie inside the region the production scan configuration zeroes.**
   On its face ``SCAN_N_TAU = 8192`` cannot represent the dispersive waves at
   all. ``SETTLE_N_TAU = 16384`` moves the cutoff to 5461 and does resolve them,
   which is presumably why it exists -- but the difference has never been
   measured, and "presumably" is not a validation status.

THE STUDY
---------
Operating point: ``PRODUCTION_OPERATING_POINT`` below, read from
``analysis.dks_access`` rather than restated, so it cannot drift away from
production silently (``tests/test_production_config_scaffold.py`` asserts the
flags match ``PRODUCTION_NUMERICS`` exactly).

    delta_omega   OPERATING_DW_KAPPA (10 kappa) -- the production operating point
    solver flags  PRODUCTION_NUMERICS: dealias_two_thirds=True,
                  edge_absorber=True, dispersion_validity_mask=False
    n_substeps    4, 8, 16, 32       (production is 4; the ladder starts there)
    n_tau         8192, 16384, 32768 (production scan / settle / one beyond)

Observables: ``validation.convergence_lle.observables_v2`` -- the same
definitions the frozen studies used, so the numbers are comparable -- plus the
quantities the noise campaign actually quotes: DW position, DW power, the 3 dB
span and the comb energy fraction. Reusing ``observables_v2`` is not optional: a
new observable defined here would make this study incomparable with the frozen
ones, which is most of its value.

Uncertainty: ``validation.convergence_lle.richardson`` on each ladder, with the
same NON_MONOTONE / ORDER_OUT_OF_RANGE guards. A ladder that does not converge
must be reported as not converging, exactly as in the frozen studies.

WHAT THIS STUDY IS NOT
----------------------
It is not a cross-code check and must not become one. pyLLE has neither mask, so
there is nothing to compare against at this configuration; the pyLLE work is
FROZEN (``docs/VALIDATION_STATUS.md`` section 5) and none of its re-run triggers
is fired by running this. This study is single-code: it measures how far the
production configuration is from ITS OWN converged limit.

COST
----
The n_tau = 32768 levels are ~5x the grid of the frozen studies and the ladder is
two-dimensional. Expect this to be substantially more expensive than
``convergence_lle.py``; budget accordingly, and consider whether the n_tau and
n_substeps ladders can be run as a cross rather than a full grid.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProductionOperatingPoint:
    """The configuration this study will characterize.

    Every field is READ from the production modules rather than restated, so a
    change to production shows up here as a test failure rather than as a study
    that silently characterizes something nobody runs.
    """

    name: str
    delta_omega_over_kappa: float
    solver_flags: dict
    n_substeps_ladder: tuple[int, ...]
    n_tau_ladder: tuple[int, ...]
    observable_source: str
    extra_observables: tuple[str, ...]
    questions: tuple[str, ...] = field(default_factory=tuple)


def _production_operating_point() -> ProductionOperatingPoint:
    from analysis.dks_access import (N_TAU, OPERATING_DW_KAPPA,
                                     PRODUCTION_NUMERICS)
    from scripts.regenerate_dks_artifacts import SCAN_N_TAU, SETTLE_N_TAU

    assert N_TAU == SCAN_N_TAU, (
        "analysis.dks_access.N_TAU and regenerate_dks_artifacts.SCAN_N_TAU have "
        "diverged; the study would characterize neither")

    return ProductionOperatingPoint(
        name="production_dw10k",
        delta_omega_over_kappa=float(OPERATING_DW_KAPPA),
        solver_flags=dict(PRODUCTION_NUMERICS),
        # Production is n_substeps = 4, so the ladder starts there: refining
        # below production would measure a configuration nobody runs.
        n_substeps_ladder=(4, 8, 16, 32),
        n_tau_ladder=(SCAN_N_TAU, SETTLE_N_TAU, 2 * SETTLE_N_TAU),
        observable_source="validation.convergence_lle.observables_v2",
        extra_observables=("dw_position_mu", "dw_power_dbc", "span_3db_modes",
                           "comb_energy_fraction"),
        questions=(
            "Does the edge absorber measurably alter DW power at production "
            "settings, and by how much (in dB, against each ladder's own "
            "discretization uncertainty)?",
            "Are the artifacts that quote DW physics converged in n_tau, given "
            "that 2/3 dealias at n_tau = 8192 zeroes |mu| > 2730 while the DW "
            "crossings sit at |mu| ~ 3038 and 3239?",
        ),
    )


PRODUCTION_OPERATING_POINT = _production_operating_point()


def dealias_cutoff(n_tau: int) -> int:
    """Highest |mu| SURVIVING the 2/3 dealias at this grid size.

    The whole point of question 2: compare this against the DW crossings at
    |mu| ~ 3038 and 3239.
    """
    return int(n_tau // 3)


def main(argv: list[str] | None = None) -> int:
    op = PRODUCTION_OPERATING_POINT
    cutoffs = {n: dealias_cutoff(n) for n in op.n_tau_ladder}
    raise SystemExit(
        "validation/production_config_convergence.py is a SCAFFOLD and has NOT "
        "been run.\n\n"
        "It declares the next validation study; it does not perform it. Running "
        "it is a\ndeliberate act that costs real compute, so it is not something "
        "to trigger by\naccident.\n\n"
        f"  operating point   delta_omega = {op.delta_omega_over_kappa:g} kappa "
        f"(production)\n"
        f"  solver flags      {op.solver_flags}\n"
        f"  n_substeps ladder {list(op.n_substeps_ladder)}\n"
        f"  n_tau ladder      {list(op.n_tau_ladder)}\n"
        f"  dealias cutoffs   {cutoffs}   (DW crossings at |mu| ~ 3038, 3239)\n"
        f"  observables       {op.observable_source} + {list(op.extra_observables)}\n\n"
        "Questions it must answer:\n"
        + "".join(f"  {i + 1}. {q}\n" for i, q in enumerate(op.questions))
        + "\nWhy it matters: every existing validation layer sits at "
        "delta_omega = 30 kappa,\nn_tau = 6601, masks OFF. Production uses none "
        "of those. See docs/VALIDATION_STATUS.md\nsection 2 for the "
        "configuration-coverage table and section 6 for why this is next.\n\n"
        "To implement it: follow validation/convergence_lle.py, which is the "
        "same shape of\nstudy at the validated operating point, and reuse its "
        "observables and its\nRichardson routine verbatim so the results are "
        "comparable with the frozen ones."
    )


if __name__ == "__main__":
    sys.exit(main())
