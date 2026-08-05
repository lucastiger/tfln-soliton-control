"""DEPRECATED shim -- the provenance helpers now live in :mod:`simulator.provenance`.

They were promoted out of ``analysis/`` so the SOLVER can stamp its own output
(``solve_lle_ssfm_jax(..., provenance=True)``) instead of every analysis script
having to remember to do it. This module re-exports the promoted API unchanged
-- in particular ``provenance_stamp(..., legacy=True)`` still produces the exact
quantum-noise-report stamp, so ``analysis/results/quantum_noise_report.json``'s
committed content is untouched.

Import from ``simulator.provenance`` instead.
"""

from __future__ import annotations

import warnings

from simulator.provenance import *  # noqa: F401,F403
from simulator.provenance import __all__ as _PROMOTED_ALL

__all__ = list(_PROMOTED_ALL)

warnings.warn(
    "analysis._provenance is deprecated; import from simulator.provenance instead.",
    DeprecationWarning,
    stacklevel=2,
)
