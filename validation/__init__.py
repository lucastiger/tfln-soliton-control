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
"""

from __future__ import annotations

__all__ = ["noise_off_identity"]
