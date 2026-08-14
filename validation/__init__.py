"""Validation harness for the stochastic-LLE benchmark.

Additive package: nothing here is imported by the solver, the models, or the
control stack, so importing it cannot perturb any existing numerical output.

Modules
-------
noise_off_identity
    Bit-identity golden regression for the *deterministic* solver path
    (``NoiseConfig.all_off()``). Establishes that the SSFM integrator itself is
    reproducible to 0 ULP across runs/machines/library versions, which is the
    precondition for attributing any observed spread in the stochastic runs to
    the noise channels rather than to the integrator.
analytic_cw
    Machine-precision verification of the homogeneous (CW) steady state against
    exact mathematics, replacing any cross-code comparison. Two references: the
    continuum LLE cubic ``P*[(kappa/2)^2 + (delta_omega - gamma*P)^2] =
    kappa_c*P_in`` (the physics target), and the exact fixed point of the
    solver's own discrete map (the integrator target, reproduced to ~1e-14).
    The gap between them — a first-order-in-dt mean-field truncation of size
    ~kappa*t_r/2 = 3.1e-3 — is measured, not assumed; see its module docstring.
mms
    Method of Manufactured Solutions: a smooth field is chosen first, the
    forcing that makes it an exact solution is derived symbolically, and the
    solver is run with that forcing. Unlike self-convergence this pins the
    scheme's order AND that it converges to the right PDE.
convergence
    Order-of-accuracy verification built on ``mms``: temporal (three thermal
    configurations), spatial (spectral), and weak (noise-on) studies.
    FINDING: the shipping scheme is FIRST order, for two independent reasons --
    the drive kick sits outside the linear half-step (non-palindromic), and the
    field<->thermal coupling is lagged. Both fixes are opt-in, default-off flags
    on ``solve_lle_ssfm_jax``; see that module's docstring.
figures
    Standalone figure scripts for the modules above.

Where ``noise_off_identity`` shows the integrator is reproducible, ``analytic_cw``
shows it is *correct*: reproducibility and exactness are independent claims and
each has its own harness.
"""

from __future__ import annotations

__all__ = ["analytic_cw", "convergence", "figures", "mms", "noise_off_identity"]
