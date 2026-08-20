"""Simulation package for TFLN soliton dynamics and state characterization.

The package is the path from a YAML configuration to a benchmark number: it
integrates the stochastic Lugiato--Lefever equation (LLE), synthesizes the
per-channel noise that drives it, classifies the resulting intracavity field,
and stamps the run so the number is attributable a year later.

Modules
-------
lle_solver
    Split-step Fourier (SSFM) integrator for the generalized LLE plus the
    thermo-optic ODE, and the unit-conversion helpers that map published
    microresonator quantities (:math:`\\gamma_{\\rm NLSE}`, :math:`D_2`,
    :math:`D_3`) onto the solver's energy normalization. Entry point
    :func:`~simulator.lle_solver.solve_lle_ssfm_jax`.
noise_config
    Immutable, hashable declaration of which stochastic channels are active
    (:class:`~simulator.noise_config.NoiseConfig`). Pure configuration: it
    imports nothing from the solver.
noise_models
    The channel implementations -- thermorefractive
    (:class:`~simulator.noise_models.TRNoise`), pyro-electric/electro-optic
    (:class:`~simulator.noise_models.PyroEONoise`), thermal-carrier
    (:class:`~simulator.noise_models.TCCRNoise`), their shared-temperature sum
    (:class:`~simulator.noise_models.TotalNoise`) and pump-laser frequency
    noise/RIN (:class:`~simulator.noise_models.PumpNoise`).
colored_noise
    PSD-specified colored-noise engine
    (:func:`~simulator.colored_noise.synthesize_from_psd`) and the target
    spectra the channels are built from.
state_labeler
    Seven-class classification of an intracavity field snapshot, in a
    ``jax.lax.scan``-traceable form and a NumPy form that must agree with it.
equation_map
    Machine-readable map from each noise channel to the equation of
    arXiv:2604.05897v1 it implements, rendered as Markdown or LaTeX.
provenance
    Run stamping and the atomic artifact writer/reader that refuses to hand
    back an unstamped array file.

Notes
-----
Importing this package imports none of the submodules, so ``import simulator``
does not pull in JAX. That matters because
:mod:`simulator.lle_solver` enables 64-bit JAX process-wide at import time
(``jax.config.update("jax_enable_x64", True)``), and JAX bakes that flag in at
array-creation time -- code that must control the ordering imports the
submodule explicitly.

``__version__`` is read from the installed distribution metadata of the
``stochastic-lle`` project rather than duplicated here, so it cannot drift
from ``pyproject.toml``. A source tree that was never installed (no
distribution metadata) reports ``"0.0.0+unknown"``.

Examples
--------
>>> import simulator
>>> isinstance(simulator.__version__, str)
True
>>> "lle_solver" in simulator.__all__
True
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _distribution_version

__all__ = [
    "colored_noise",
    "equation_map",
    "lle_solver",
    "noise_config",
    "noise_models",
    "provenance",
    "state_labeler",
]

try:
    #: Version of the installed ``stochastic-lle`` distribution.
    __version__: str = _distribution_version("stochastic-lle")
except PackageNotFoundError:  # pragma: no cover - source tree, never installed
    __version__ = "0.0.0+unknown"
