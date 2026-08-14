"""Figures for the validation harness.

Additive package: every module here is a standalone script that imports from
``validation`` and ``simulator`` and writes an image. Nothing here is imported
by the solver, the models, or the control stack, so importing it cannot perturb
any existing numerical output.

Modules
-------
fig_analytic_cw
    The two-panel companion to :mod:`validation.analytic_cw`: the bistable
    CW S-curve (analytic vs solver) and the relative-residual spectrum against
    both exact references.
fig_convergence
    The two-panel companion to :mod:`validation.convergence`: the deterministic
    dt-refinement log-log with reference slopes (production vs fixed scheme),
    and the weak-order fit.
"""

from __future__ import annotations

__all__ = ["fig_analytic_cw", "fig_convergence"]
