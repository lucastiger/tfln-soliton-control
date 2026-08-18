"""Machine-precision verification of the solver's homogeneous (CW) steady state.

Why this exists
---------------
Cross-code comparison against another LLE integrator can only ever agree to the
level at which the two codes *discretize* the same PDE — typically 1e-3 or
worse. It cannot certify anything to 1e-12. Exact mathematics can: the
homogeneous steady state of the equation this repo actually implements is the
root of a cubic, and the fixed point of the *discrete map* this repo actually
iterates is the root of one scalar transcendental equation. Both can be
evaluated to full double precision, so both give a real machine-precision
target. This module builds both and compares the solver against them at 405
points.

The two reference solutions, and why there are two
--------------------------------------------------
**(1) The continuum LLE.** ``simulator/lle_solver.py`` composes, per sub-step of
length ``dt``: a pump kick ``E -> E + F*dt`` with ``F = sqrt(kappa_c*pin)``, a
half linear step ``exp((-kappa/2 - i*D_int - i*delta_omega)*dt/2)``, a Kerr
kick ``exp(+i*gamma*|E|^2*dt)``, and the second half linear step (see
``_fine_step``). Its continuum limit is

    dE/dt = -(kappa/2)*E - i*delta_omega*E + i*gamma*|E|^2*E + F        (zero D_int)

so the Kerr term shifts the effective detuning to ``delta_omega - gamma*P`` and
the homogeneous steady state obeys

    P * [ (kappa/2)^2 + (delta_omega - gamma*P)^2 ] = kappa_c * P_in    (P = |E|^2, J)

which is what :func:`analytic_cw_roots` solves. This is the *physics* target.

**(2) The discrete map.** The solver does not integrate the ODE above; it
iterates a map. For a flat field with zero dispersion the whole round trip
collapses to one complex scalar recursion, whose fixed point
(:func:`discrete_map_cw_fixed_point`) is exact, not asymptotic. This is the
*integrator* target.

Reference (1) is the physically meaningful one but is NOT reachable to 1e-12,
and that is a real, reproducible property of the solver rather than a defect of
this harness — see the next section. Reference (2) is reachable to ~1e-14 and
is therefore what the CLI exit code gates on. Both are always reported.

FINDING: the mean-field (continuum) residual is 3e-3 and first order in dt
-------------------------------------------------------------------------
Measured over the full sweep at the committed SiN parameters
(kappa*t_r = 6.174e-3):

  * the solver's CW state is *bit-stationary* — the last 10% of
    ``U_int_history`` is flat to between 0.0e+00 and 4.7e-14 relative — yet it
    sits 2.37e-03 to 3.08e-03 away from the cubic root, everywhere;
  * the residual shrinks as 1/n_substeps: 3.070e-03, 7.707e-04, 1.929e-04 for
    n_substeps = 1, 4, 16 (ratios 3.98, 4.00), i.e. clean FIRST ORDER in dt;
  * its magnitude is ~kappa*t_r/2 = 3.087e-03;
  * it PEAKS AT RESONANCE (delta_omega = +1*kappa, where the intracavity power
    is largest), not at the bistability turning points. At 10*P_th all 13
    bistable grid points sit at 3.082e-03 with tail flatness exactly 0.0e+00.

So this is not slow convergence near a singular Jacobian, and extending the
settle time cannot touch it. The cause is the operator splitting: the pump is
injected as ``E + F*dt`` OUTSIDE the linear exponential, where the exact
integrating factor over the step would contribute ``F*(exp(z)-1)/z`` with
``z = (-kappa/2 - i*delta_omega)*dt``. The two differ at relative order
``|z|/2``, which is first order in dt and is exactly what is observed.
Reaching 1e-12 through ``n_substeps`` alone would require n_substeps ~ 1e11.

This is the standard mean-field (Ikeda-map -> LLE) truncation, and 3e-3 is a
perfectly ordinary size for it. It is quantified here rather than hidden:
:func:`mean_field_order_check` re-measures the 1/n_substeps law on every CLI
run, so a change in the splitting scheme shows up as a change in the ORDER, not
merely in a number.

Isolating the CW problem from the rest of the solver
----------------------------------------------------
Three things are switched off, all of them opt-in and local to this module:

* **Stochastic channels** — ``NoiseConfig.all_off()``, exactly as
  ``validation/noise_off_identity.py`` does.
* **Dispersion** — ``beta = [0.0, 0.0]`` makes ``build_dispersion`` return
  identically zero, so ``D_int(mu) = 0`` for every mode and a flat field stays
  flat under the linear step (the FFT of a constant populates bin 0 only).
* **Thermo-optic feedback** — this one needs care. ``NoiseConfig.all_off()``
  leaves ``thermal_feedback`` at its field default, but that switch is
  declarative: no solver code path reads it, so the deterministic thermo-optic
  ODE runs regardless (see the note in ``validation/noise_off_identity.py``).
  A live thermal shift would move the *effective* detuning away from the
  programmed one, and the cubic would then simply be evaluated at the wrong
  delta_omega. :func:`thermal_off_config` therefore derives a temporary config
  from the committed one with the single leaf ``dn_dT_per_k`` set to 0.0, which
  makes ``thermal_shift = -(omega0/n0)*dn_dT*DeltaT`` identically zero for
  every DeltaT. Nothing else is altered, and the helper asserts leaf-by-leaf
  that nothing else changed. ``config/sin_params.yaml`` itself is never
  touched, and every leaf stays numeric (tests/test_config.py).

Which branch the solver is compared against
-------------------------------------------
In the bistable region the cubic has three real positive roots. The physically
selected one depends on history, so :func:`stable_branch` encodes the branch a
slow UPWARD detuning scan reaches (the largest root — see its docstring for the
derivation), and :func:`solver_cw_steady_state` seeds the solver from a flat
field DELIBERATELY OFFSET from that root (2% in amplitude by default) so the
solver has to relax to its own fixed point rather than being handed it. The
last-10% flatness assertion is what makes that meaningful: a seed that fell
outside the basin, or landed on the unstable middle branch, shows up as a
non-stationary tail rather than as a silent pass.

Reference: Herr, Tikan & Kippenberg, arXiv:2604.05897v1 (7 Apr 2026).

CLI
---
``python -m validation.analytic_cw`` runs the full 81 x 5 sweep, prints the
table, and exits nonzero if the discrete-map residual exceeds ``--rtol``
(default 1e-12). ``--full`` prints every point; ``--quick`` runs a reduced grid.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:                    # allow `python validation/analytic_cw.py`
    sys.path.insert(0, str(_REPO_ROOT))

from simulator.lle_solver import (                     # noqa: E402
    _load_config,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
)
from simulator.noise_config import NoiseConfig         # noqa: E402

__all__ = [
    "CavityParams",
    "DEFAULT_DETUNING_GRID",
    "DEFAULT_PIN_MULTIPLIERS",
    "analytic_cw_roots",
    "discrete_map_cw_fixed_point",
    "load_cavity_params",
    "mean_field_order_check",
    "p_threshold",
    "solver_cw_steady_state",
    "stable_branch",
    "thermal_off_config",
    "verify",
]

#: The detuning sweep of the acceptance criterion, in units of kappa.
DEFAULT_DETUNING_GRID = np.linspace(-5.0, 15.0, 81)

#: The pump sweep of the acceptance criterion, in units of P_th.
DEFAULT_PIN_MULTIPLIERS = (0.5, 1.0, 2.0, 5.0, 10.0)

#: Round trips per solver run. The CW relaxation time is 2/kappa = 324*t_r, so
#: 60000 round trips is ~185 relaxation times — measured tail flatness is
#: <= 4.8e-14 relative at every point of the sweep.
DEFAULT_T_SLOW = 60_000

#: Fast-time samples. The CW problem is spatially uniform and dispersion-free,
#: so n_tau only sets the size of the (exactly redundant) grid; 16 keeps the
#: 405-point sweep at ~80 s while leaving room for a non-flat state to show up
#: in the flatness assertion if the branch under test were unstable.
DEFAULT_N_TAU = 16

#: Safety factor on the derived stationarity floor (see :func:`flatness_floor`).
FLATNESS_SAFETY = 8.0


def flatness_floor(kappa: float, dt: float, safety: float = FLATNESS_SAFETY) -> float:
    """Smallest tail peak-to-peak a converged float64 CW run can achieve.

    Parameters
    ----------
    kappa : float
        Total cavity loss rate [rad/s].
    dt : float
        Field sub-step [s], i.e. ``t_r/n_substeps``.
    safety : float, optional
        Multiplier on the derived floor [dimensionless], default
        :data:`FLATNESS_SAFETY` (8.0).

    Returns
    -------
    float
        Relative peak-to-peak [dimensionless] below which a stationarity test
        would be testing floating-point noise rather than convergence.

    Raises
    ------
    ValueError
        If ``kappa*dt`` is not positive, which would make the map
        non-contracting and the geometric sum divergent.

    Notes
    -----
    A converged fixed point does not sit still in floating point: it executes a
    small limit cycle driven by rounding. Per sub-step the map contracts a
    perturbation by ``|L| = exp(-kappa*dt/2)`` and rounding re-injects a
    relative perturbation of order ``eps``, so the stationary jitter amplitude
    is the geometric sum

        eps * (1 + |L| + |L|^2 + ...) = eps / (1 - exp(-kappa*dt/2)).

    At the committed SiN parameters ``kappa*dt/2 = 3.087e-03``, giving
    ``2.22e-16 / 3.086e-03 = 7.2e-14`` — so a 1e-14 stationarity bound is
    BELOW the noise floor and cannot be met by any settle time. Measured worst
    tail flatness over the full 405-point sweep is 4.7e-14 (at 10*P_th, where
    the fixed point is nearest a fold and the true Jacobian eigenvalue is
    closer to 1 than the linear-loss estimate), consistent with the bound.

    ``safety`` covers that Jacobian tightening and cross-machine reduction-order
    differences without loosening the check to the point of meaninglessness:
    the default floor, ~5.8e-13, is still four orders below the 3e-3 mean-field
    residual and would catch any genuinely drifting or oscillating state.

    Examples
    --------
    >>> params = load_cavity_params()
    >>> print(f"{flatness_floor(params.kappa, params.t_r):.4e}")
    5.7632e-13

    Refining the sub-step contracts less per step, so the achievable floor
    RISES -- which is why the bound must be derived rather than fixed:

    >>> bool(flatness_floor(params.kappa, params.t_r / 8)
    ...      > flatness_floor(params.kappa, params.t_r))
    True

    >>> flatness_floor(0.0, 1e-11)
    Traceback (most recent call last):
        ...
    ValueError: kappa*dt must be positive (got 0.0).
    """
    contraction = -math.expm1(-float(kappa) * float(dt) / 2.0)   # 1 - |L|
    if contraction <= 0.0:
        raise ValueError(f"kappa*dt must be positive (got {kappa * dt}).")
    return float(safety) * float(np.finfo(np.float64).eps) / contraction


# ---------------------------------------------------------------------------
# Cavity parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CavityParams:
    """Resolved cavity constants plus the derived thermo-optic-free config.

    Parameters
    ----------
    kappa : float
        Total loss rate [rad/s].
    kappa_c : float
        Coupling rate [rad/s].
    kappa_i : float
        Intrinsic loss rate [rad/s].
    gamma : float
        ``gamma_LLE`` [J^-1 s^-1].
    t_r : float
        Round-trip time [s].
    p_th : float
        Modulation-instability threshold pump power [W]; see
        :func:`p_threshold`.
    config_path : str
        Path of the DERIVED (``dn_dT_per_k = 0``) config handed to the solver.
    source_config : str
        Path of the committed config it was derived from.

    Raises
    ------
    dataclasses.FrozenInstanceError
        On any attempt to mutate an instance.

    Notes
    -----
    ``config_path`` is the path handed to
    :func:`~simulator.lle_solver.solve_lle_ssfm_jax` -- the derived config of
    :func:`thermal_off_config`, NOT the committed one. Every other field is
    read from the committed config unchanged, and ``source_config`` records
    which one, so a result can be traced back to its inputs.

    Examples
    --------
    >>> params = load_cavity_params()
    >>> bool(abs(params.kappa - (params.kappa_i + params.kappa_c)) < 1.0)
    True
    >>> params.source_config.endswith("sin_params.yaml")
    True
    >>> params.config_path != params.source_config
    True
    """

    kappa: float          # total loss rate, rad/s
    kappa_c: float        # coupling rate, rad/s
    kappa_i: float        # intrinsic loss rate, rad/s
    gamma: float          # gamma_LLE, J^-1 s^-1
    t_r: float            # round-trip time, s
    p_th: float           # MI threshold pump power, W
    config_path: str      # derived (dn_dT = 0) config used for the solver runs
    source_config: str    # the committed config it was derived from


def p_threshold(kappa: float, kappa_c: float, gamma: float) -> float:
    """Return the modulation-instability onset pump power.

    Parameters
    ----------
    kappa : float
        Total cavity loss rate [rad/s].
    kappa_c : float
        Coupling rate [rad/s].
    gamma : float
        ``gamma_LLE`` [J^-1 s^-1].

    Returns
    -------
    float
        ``P_th = kappa**3 / (8*gamma*kappa_c)`` [W].

    Raises
    ------
    ZeroDivisionError
        If ``gamma`` or ``kappa_c`` is zero.

    Notes
    -----
    The same expression the solver's own pre-flight check uses: setting
    ``gamma*U_cw = kappa/2`` at ``delta_omega = 0`` with
    ``U_cw = kappa_c*pin/(kappa/2)**2``. Because ``|E|**2`` is already in
    joules in this normalization there is no stray ``1/t_r`` -- the factor that
    makes this expression disagree with the power-normalized form quoted in
    most of the literature.

    This is the unit every pump power in the study is expressed in, so that a
    sweep reads the same whatever cavity it is run on.

    Examples
    --------
    >>> print(f"{p_threshold(1e9, 5e8, 1e18):.4e} W")
    2.5000e-01 W

    Threshold falls as the nonlinearity rises:

    >>> p_threshold(1e9, 5e8, 2e18) == p_threshold(1e9, 5e8, 1e18) / 2
    True
    """
    return kappa ** 3 / (8.0 * gamma * kappa_c)


def thermal_off_config(
    source: str | Path | None = None, *, out_dir: str | Path | None = None
) -> str:
    """Derive a config identical to ``source`` except ``dn_dT_per_k = 0.0``.

    Parameters
    ----------
    source : str or pathlib.Path or None, optional
        Committed config to derive from. ``None`` (default) uses
        ``config/sin_params.yaml``.
    out_dir : str or pathlib.Path or None, optional
        Directory for the derived file. ``None`` (default) writes a temp file.
        Keyword-only.

    Returns
    -------
    str
        Path of the derived config, suitable as ``config_path`` for
        :func:`~simulator.lle_solver.solve_lle_ssfm_jax`.

    Raises
    ------
    ValueError
        If ``source`` has no ``physical_parameters`` block.
    AssertionError
        If the YAML round trip changed the key set, perturbed any numeric leaf
        other than ``dn_dT_per_k``, or failed to zero ``dn_dT_per_k``.
    OSError
        If ``source`` cannot be read or the destination cannot be written.

    Notes
    -----
    The thermo-optic detuning shift in the solver is
    ``-(omega0/n0)*dn_dT*DeltaT``; zeroing ``dn_dT`` makes it identically zero
    for every DeltaT, which decouples the (still-running, still-harmless)
    thermal ODE from the field and leaves ``delta_omega_eff == delta_omega``
    exactly. That is the precondition for comparing against a cubic written in
    the PROGRAMMED detuning.

    The committed config is never modified. The result is written to a temp
    file (or ``out_dir``); every numeric leaf is asserted to survive the YAML
    round trip bit-for-bit, so the derived config differs from the source in
    exactly one leaf.

    Checking every leaf rather than trusting the round trip is the point: a
    float that did not survive ``safe_dump`` -> ``safe_load`` unchanged would
    silently perturb the physics of a comparison whose whole claim is
    machine precision.

    Examples
    --------
    >>> import yaml
    >>> from pathlib import Path
    >>> derived = thermal_off_config()
    >>> block = yaml.safe_load(Path(derived).read_text())["physical_parameters"]
    >>> block["dn_dT_per_k"]
    0.0

    Every other leaf is unchanged, and the committed config is untouched:

    >>> source = yaml.safe_load(
    ...     Path("config/sin_params.yaml").read_text())["physical_parameters"]
    >>> sorted(k for k in source if source[k] != block[k])
    ['dn_dT_per_k']
    """
    src = Path(source) if source is not None else _REPO_ROOT / "config" / "sin_params.yaml"
    with src.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    physical = cfg.get("physical_parameters")
    if not isinstance(physical, dict):
        raise ValueError(f"{src} has no physical_parameters block.")
    original = dict(physical)
    physical["dn_dT_per_k"] = 0.0

    if out_dir is None:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", prefix="analytic_cw_", delete=False, encoding="utf-8"
        )
        dest = Path(handle.name)
        yaml.safe_dump(cfg, handle, sort_keys=False)
        handle.close()
    else:
        dest = Path(out_dir) / "analytic_cw_thermal_off.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)

    # The derived config must differ in EXACTLY one leaf. A float that does not
    # survive yaml.safe_dump -> yaml.safe_load unchanged would silently perturb
    # the physics of the comparison, so check every leaf rather than trusting it.
    reloaded = _load_config(dest)
    if set(reloaded) != set(original):
        raise AssertionError(
            f"derived config key set changed: "
            f"{set(original) ^ set(reloaded)}"
        )
    for key, value in original.items():
        if key == "dn_dT_per_k":
            continue
        if reloaded[key] != value:
            raise AssertionError(
                f"YAML round trip perturbed physical_parameters[{key!r}]: "
                f"{value!r} -> {reloaded[key]!r}"
            )
    if reloaded["dn_dT_per_k"] != 0.0:
        raise AssertionError("dn_dT_per_k was not zeroed in the derived config.")
    return str(dest)


def load_cavity_params(
    config_path: str | Path | None = None, *, out_dir: str | Path | None = None
) -> CavityParams:
    """Resolve the cavity constants and derive the thermo-optic-free config.

    Parameters
    ----------
    config_path : str or pathlib.Path or None, optional
        Committed config to read. ``None`` (default) uses
        ``config/sin_params.yaml``.
    out_dir : str or pathlib.Path or None, optional
        Directory for the derived config; ``None`` (default) writes a temp
        file. Keyword-only.

    Returns
    -------
    CavityParams
        ``kappa``, ``kappa_c``, ``kappa_i`` [rad/s], ``gamma`` [J^-1 s^-1],
        ``t_r`` [s], ``p_th`` [W], plus the derived and source config paths.

    Raises
    ------
    ValueError
        Propagated from :func:`~simulator.lle_solver.resolve_cavity_rates` if
        the intrinsic loss cannot be resolved, or from
        :func:`thermal_off_config`.
    KeyError
        If ``gamma_LLE_per_J_per_s`` or ``fsr_hz`` is absent.
    OSError
        If the config cannot be read or the derived config cannot be written.

    Notes
    -----
    The single entry point for every module in the validation harness that
    needs cavity constants, so all of them agree by construction. Deriving the
    thermo-optic-free config HERE rather than at each call site is what
    guarantees no study accidentally runs against the committed config and
    picks up a thermal shift the analytic reference does not model.

    Examples
    --------
    >>> params = load_cavity_params()
    >>> bool(params.kappa > 0 and params.gamma > 0 and params.t_r > 0)
    True
    >>> from validation.analytic_cw import p_threshold
    >>> bool(abs(params.p_th - p_threshold(params.kappa, params.kappa_c,
    ...                                    params.gamma)) < 1e-18)
    True
    """
    src = Path(config_path) if config_path is not None else _REPO_ROOT / "config" / "sin_params.yaml"
    kappa_i, kappa_c, kappa = resolve_cavity_rates(src)
    physical = _load_config(src)
    gamma = float(physical["gamma_LLE_per_J_per_s"])
    t_r = 1.0 / float(physical["fsr_hz"])
    return CavityParams(
        kappa=float(kappa),
        kappa_c=float(kappa_c),
        kappa_i=float(kappa_i),
        gamma=gamma,
        t_r=t_r,
        p_th=p_threshold(float(kappa), float(kappa_c), gamma),
        config_path=thermal_off_config(src, out_dir=out_dir),
        source_config=str(src),
    )


# ---------------------------------------------------------------------------
# Reference 1: the continuum LLE cubic
# ---------------------------------------------------------------------------
def _normalized_cubic_roots(d: float, x_drive: float) -> np.ndarray:
    """Real positive roots of ``x*(1 + (d-x)^2) = x_drive``, machine precision.

    The normalization is what makes this well conditioned. With
    ``x = 2*gamma*P/kappa`` and ``d = 2*delta_omega/kappa`` the steady-state
    equation ``P*[(kappa/2)^2 + (delta_omega - gamma*P)^2] = kappa_c*P_in``
    becomes, dividing through by ``(kappa/2)^2 * kappa/(2*gamma)``,

        x * [1 + (d - x)^2] = 8*gamma*kappa_c*P_in/kappa^3 = P_in/P_th

    so every coefficient is O(1..d^2) instead of spanning the 1e36 dynamic
    range of the dimensional form (gamma^2 ~ 1e36 against kappa_c*P_in ~ 1e-3).

    Solved in DEPRESSED form. Writing the monic cubic as
    ``x^3 - 2d*x^2 + (1+d^2)*x - X = 0`` and substituting ``x = y + 2d/3``
    eliminates the quadratic term exactly:

        y^3 + p*y + q = 0,    p = 1 - d^2/3,    q = 2*d^3/27 + 2*d/3 - X

    ``numpy.roots`` on ``[1, 0, p, q]`` is the companion-matrix eigensolve; its
    ~1e-15 relative accuracy is then polished to full precision by Newton on
    the ORIGINAL residual (the depressed shift is exact in floating point only
    up to rounding, and the polish removes that too).
    """
    p = 1.0 - d * d / 3.0
    q = 2.0 * d ** 3 / 27.0 + 2.0 * d / 3.0 - x_drive
    shift = 2.0 * d / 3.0
    raw = np.roots(np.array([1.0, 0.0, p, q], dtype=np.float64)) + shift

    # A real cubic has real coefficients, so genuinely complex roots come in
    # conjugate pairs with an imaginary part of the same order as the root
    # itself; numerical noise on a real root is ~1e-15 of it. Split on a
    # generous relative threshold, then let the Newton polish reject anything
    # that was not actually a root.
    scale = max(1.0, float(np.max(np.abs(raw))))
    candidates = [float(z.real) for z in raw if abs(z.imag) <= 1e-7 * scale]

    def f(x: float) -> float:
        return x * (1.0 + (d - x) ** 2) - x_drive

    def df(x: float) -> float:
        return 3.0 * x * x - 4.0 * d * x + (1.0 + d * d)

    polished: list[float] = []
    for x0 in candidates:
        x = x0
        for _ in range(100):
            slope = df(x)
            if slope == 0.0:
                break
            step = f(x) / slope
            x -= step
            if abs(step) <= 4.0 * np.finfo(float).eps * max(abs(x), 1e-300):
                break
        if x <= 0.0:
            continue
        # Reject a candidate that did not converge: the residual must be at the
        # rounding level of the individual terms of f.
        term_scale = max(abs(x) * (1.0 + (d - x) ** 2), abs(x_drive))
        if abs(f(x)) > 1e-11 * term_scale:
            continue
        polished.append(x)

    polished.sort()
    unique: list[float] = []
    for x in polished:
        if not unique or abs(x - unique[-1]) > 1e-10 * max(abs(x), 1.0):
            unique.append(x)
    return np.asarray(unique, dtype=np.float64)


def analytic_cw_roots(
    pin: float,
    delta_omega: float,
    kappa: float,
    kappa_c: float,
    gamma: float,
) -> np.ndarray:
    """Solve the continuum CW steady state for its real positive roots.

    Parameters
    ----------
    pin : float
        Pump power [W].
    delta_omega : float
        Detuning ``omega_res - omega_pump`` [rad/s]; positive is red-detuned,
        the soliton side.
    kappa : float
        Total loss rate [rad/s].
    kappa_c : float
        Coupling rate [rad/s].
    gamma : float
        ``gamma_LLE`` [J^-1 s^-1].

    Returns
    -------
    numpy.ndarray
        Real positive roots ``P = |E|**2`` [J], dtype ``float64``, sorted
        ascending. Shape ``(1,)`` outside the bistable region and ``(3,)``
        inside it, with the middle root the unstable branch. Empty only for
        ``pin <= 0``.

    Raises
    ------
    ValueError
        If ``kappa`` or ``gamma`` is not positive.

    Notes
    -----
    Solves the homogeneous steady state of the LLE,

        P * [(kappa/2)**2 + (delta_omega - gamma*P)**2] = kappa_c * pin,

    which is the PHYSICS target of the whole module: the continuum reference
    the solver is compared against.

    The cubic is solved in NORMALIZED, DEPRESSED form. With
    ``x = 2*gamma*P/kappa`` and ``d = 2*delta_omega/kappa`` it becomes
    ``x*[1 + (d - x)**2] = pin/P_th``, whose coefficients are O(1) instead of
    spanning the 1e36 dynamic range of the dimensional form
    (``gamma**2 ~ 1e36`` against ``kappa_c*pin ~ 1e-3``). Conditioning is the
    whole reason for the change of variables: a reference that loses digits is
    not a reference at machine precision.

    Examples
    --------
    >>> params = load_cavity_params()
    >>> args = (params.kappa, params.kappa_c, params.gamma)

    On resonance at threshold the response is single-valued:

    >>> analytic_cw_roots(params.p_th, 0.0, *args).size
    1

    Far red-detuned and well above threshold it is bistable:

    >>> roots = analytic_cw_roots(10 * params.p_th, 5 * params.kappa, *args)
    >>> roots.size
    3
    >>> bool(roots[0] < roots[1] < roots[2])
    True

    Every root satisfies the steady-state equation to machine precision:

    >>> import numpy as np
    >>> lhs = roots * ((params.kappa / 2) ** 2
    ...                + (5 * params.kappa - params.gamma * roots) ** 2)
    >>> bool(np.all(np.abs(lhs / (params.kappa_c * 10 * params.p_th) - 1) < 1e-9))
    True

    >>> analytic_cw_roots(0.0, 0.0, *args).size
    0
    """
    pin = float(pin)
    kappa = float(kappa)
    kappa_c = float(kappa_c)
    gamma = float(gamma)
    if kappa <= 0.0 or gamma <= 0.0:
        raise ValueError(f"kappa and gamma must be positive (got {kappa}, {gamma}).")
    if pin <= 0.0:
        return np.zeros(0, dtype=np.float64)

    d = 2.0 * float(delta_omega) / kappa
    x_drive = pin / p_threshold(kappa, kappa_c, gamma)
    return _normalized_cubic_roots(d, x_drive) * (kappa / (2.0 * gamma))


def stable_branch(
    roots: Sequence[float] | np.ndarray, scan_direction: str = "up"
) -> float:
    """The branch a slow monotonic detuning scan reaches.

    Derivation for ``scan_direction="up"`` (delta_omega increasing, the
    blue-to-red scan used for soliton access). Solving the steady state for the
    detuning instead of the power gives two branches,

        delta_omega(P) = gamma*P -/+ sqrt(kappa_c*P_in/P - (kappa/2)^2),

    defined for ``P <= P_max = kappa_c*P_in/(kappa/2)^2``. On the minus branch

        d(delta_omega)/dP = gamma + (kappa_c*P_in/P^2) / (2*sqrt(...)) > 0

    for every P, so that branch is single-valued and rises monotonically from
    ``delta_omega -> -inf`` at ``P -> 0`` to ``delta_omega = gamma*P_max`` at
    ``P = P_max``. A scan starting far blue therefore begins on it and climbs
    it without ever meeting a fold. Continuing through ``P_max`` onto the plus
    branch, ``d(delta_omega)/dP -> -inf`` there, so delta_omega keeps
    increasing while P falls, until the fold where the derivative changes sign.
    That fold is the UPPER turning point, and everything traversed up to it is
    the high-power branch. Hence: inside the bistable window an upward scan
    sits on the LARGEST root, and it stays there until the window closes.

    A downward scan is the mirror image and sits on the smallest root.

    Parameters
    ----------
    roots : sequence of float or numpy.ndarray
        Ascending roots [J], as returned by :func:`analytic_cw_roots`.
    scan_direction : {'up', 'down'}, optional
        Direction of the slow detuning scan; ``'up'`` (default) is the
        blue-to-red scan used for soliton access.

    Returns
    -------
    float
        The selected root ``|E|**2`` [J].

    Raises
    ------
    ValueError
        If ``roots`` is empty, or if ``scan_direction`` is neither ``'up'`` nor
        ``'down'``.

    Examples
    --------
    >>> params = load_cavity_params()
    >>> roots = analytic_cw_roots(10 * params.p_th, 5 * params.kappa,
    ...                           params.kappa, params.kappa_c, params.gamma)
    >>> stable_branch(roots, "up") == float(roots[-1])
    True
    >>> stable_branch(roots, "down") == float(roots[0])
    True

    Outside the bistable window both directions agree, since there is only one
    branch to be on:

    >>> single = analytic_cw_roots(params.p_th, 0.0, params.kappa,
    ...                            params.kappa_c, params.gamma)
    >>> stable_branch(single, "up") == stable_branch(single, "down")
    True

    >>> stable_branch(roots, "sideways")
    Traceback (most recent call last):
        ...
    ValueError: scan_direction must be 'up' or 'down', got 'sideways'.
    """
    arr = np.asarray(roots, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("stable_branch() needs at least one root.")
    if scan_direction == "up":
        return float(arr[-1])
    if scan_direction == "down":
        return float(arr[0])
    raise ValueError(f"scan_direction must be 'up' or 'down', got {scan_direction!r}.")


# ---------------------------------------------------------------------------
# Reference 2: the exact fixed point of the solver's own discrete map
# ---------------------------------------------------------------------------
def discrete_map_cw_fixed_point(
    pin: float,
    delta_omega: float,
    kappa: float,
    kappa_c: float,
    gamma: float,
    dt: float,
    *,
    p_guess: float | None = None,
) -> float:
    """Exact CW fixed point of the SSFM map, to full double precision [J].

    For a flat field with zero dispersion the FFTs in ``_fine_step`` are the
    identity up to rounding (a constant populates FFT bin 0 only, and the
    linear operator is then a single scalar), so one sub-step of the solver is
    exactly the scalar recursion

        A     = E + F*dt                                (pump kick, F = sqrt(kappa_c*pin))
        E_out = A * L,   L = r * exp(i*(phi - delta_omega*dt))
        r     = exp(-kappa*dt/2),   phi = gamma * r * |A|^2 * dt

    ``phi`` carries the factor ``r`` because the Kerr kick sees the field
    AFTER the first half linear step, where ``|E|^2 = r*|A|^2``.

    At the fixed point ``E_out = E``, so ``A*(1 - L) = F*dt`` and therefore

        Q * |1 - L(Q)|^2 = (F*dt)^2,      Q = |A|^2,      P = |E|^2 = r^2 * Q.

    ``|1 - L|^2`` is evaluated as

        |1 - L|^2 = (1 - r)^2 + 4*r*sin^2(psi/2),        psi = phi - delta_omega*dt

    NOT as ``1 - 2*r*cos(psi) + r^2``. Near resonance psi is small and both
    ``1 - r`` and ``1 - cos(psi)`` are ~1e-3..1e-6, so the direct form loses
    about five digits to cancellation — enough to put the measured residual at
    6.4e-12 and fail the 1e-12 gate for a reason that lives entirely in this
    reference rather than in the solver. With ``1 - r`` computed as
    ``-expm1(-kappa*dt/2)`` and the half-angle identity, the same points come
    out at ~1e-15.

    Newton on ``g(Q) = Q*|1-L(Q)|^2 - (F*dt)^2`` converges in a handful of
    iterations from the continuum root, which is within ~0.3% of the answer.

    Parameters
    ----------
    pin : float
        Pump power [W].
    delta_omega : float
        Detuning ``omega_res - omega_pump`` [rad/s].
    kappa : float
        Total loss rate [rad/s].
    kappa_c : float
        Coupling rate [rad/s].
    gamma : float
        ``gamma_LLE`` [J^-1 s^-1].
    dt : float
        The solver's field sub-step [s],
        ``t_r/(fine_cadence_M * n_substeps)``.
    p_guess : float or None, optional
        Starting ``|E|**2`` [J]. ``None`` (default) uses the upward-scan
        continuum root. Keyword-only.

    Returns
    -------
    float
        ``|E|**2`` [J] at the exact fixed point of the discrete map. Zero when
        the continuum problem has no positive root (``pin <= 0``).

    Raises
    ------
    RuntimeError
        If Newton did not converge to a relative residual below 1e-11, naming
        the operating point. A reference that quietly returned a
        half-converged value would be worse than none.
    ValueError
        Propagated from :func:`analytic_cw_roots` for a non-positive ``kappa``
        or ``gamma``.

    Notes
    -----
    This is the INTEGRATOR target, as distinct from the physics target of
    :func:`analytic_cw_roots`. The gap between the two is the mean-field
    (Ikeda-map to LLE) truncation of the splitting -- first order in dt and of
    size ``~kappa*t_r/2 = 3.1e-3`` at the committed parameters -- so it is
    measured, not assumed, and only this reference can be gated at 1e-12.

    Examples
    --------
    >>> params = load_cavity_params()
    >>> args = (params.kappa, params.kappa_c, params.gamma)
    >>> pin, dw = 10 * params.p_th, 5 * params.kappa
    >>> continuum = stable_branch(analytic_cw_roots(pin, dw, *args), "up")
    >>> discrete = discrete_map_cw_fixed_point(pin, dw, *args, params.t_r)
    >>> print(f"{abs(discrete - continuum) / continuum:.2e}")   # ~kappa*t_r/2
    3.08e-03

    Refining the sub-step drives the two together at first order:

    >>> refined = discrete_map_cw_fixed_point(pin, dw, *args, params.t_r / 8)
    >>> gap = abs(discrete - continuum) / abs(refined - continuum)
    >>> bool(7.0 < gap < 9.0)
    True
    """
    pin = float(pin)
    dt = float(dt)
    kappa = float(kappa)
    gamma = float(gamma)
    delta_omega = float(delta_omega)

    if p_guess is None:
        roots = analytic_cw_roots(pin, delta_omega, kappa, kappa_c, gamma)
        if roots.size == 0:
            return 0.0
        p_guess = stable_branch(roots, "up")

    drive = math.sqrt(float(kappa_c) * pin) * dt          # F*dt
    drive_sq = drive * drive
    r = math.exp(-kappa * dt / 2.0)
    one_minus_r = -math.expm1(-kappa * dt / 2.0)          # cancellation-free
    one_minus_r_sq = one_minus_r * one_minus_r

    def modulus_sq(q: float) -> tuple[float, float]:
        """(|1 - L|^2, psi) at Q = q."""
        psi = gamma * r * q * dt - delta_omega * dt
        return one_minus_r_sq + 4.0 * r * math.sin(psi / 2.0) ** 2, psi

    q = float(p_guess) / (r * r)
    for _ in range(200):
        s, psi = modulus_sq(q)
        g = q * s - drive_sq
        # d/dQ [Q*|1-L|^2] = |1-L|^2 + Q * 2*r*sin(psi) * dpsi/dQ,
        # with dpsi/dQ = gamma*r*dt.
        dg = s + q * 2.0 * r * math.sin(psi) * gamma * r * dt
        if dg == 0.0:
            break
        step = g / dg
        q -= step
        if abs(step) <= 4.0 * np.finfo(float).eps * abs(q):
            break

    s, _ = modulus_sq(q)
    residual = abs(q * s - drive_sq)
    if residual > 1e-11 * max(q * s, drive_sq):
        raise RuntimeError(
            f"discrete_map_cw_fixed_point did not converge at "
            f"delta_omega={delta_omega:.6e}, pin={pin:.6e}: "
            f"relative residual {residual / max(q * s, drive_sq):.3e}."
        )
    return r * r * q


# ---------------------------------------------------------------------------
# The solver's own answer
# ---------------------------------------------------------------------------
def solver_cw_steady_state(
    pin: float,
    delta_omega: float | Sequence[float] | np.ndarray,
    params: CavityParams,
    *,
    t_slow: int = DEFAULT_T_SLOW,
    n_tau: int = DEFAULT_N_TAU,
    n_substeps: int = 1,
    scan_direction: str = "up",
    seed_perturbation: float = 1.02,
    flatness_rtol: float | None = None,
    seed_power: Sequence[float] | np.ndarray | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Converge the solver to its CW fixed point and return the field power.

    Parameters
    ----------
    pin : float
        Pump power [W].
    delta_omega : float or sequence of float or numpy.ndarray
        Scalar or 1-D array of detunings [rad/s].
    params : CavityParams
        Resolved cavity constants; supplies the derived thermo-optic-free
        config.
    t_slow : int, optional
        Round trips [dimensionless count], default :data:`DEFAULT_T_SLOW`.
        Keyword-only.
    n_tau : int, optional
        Fast-time grid points, default :data:`DEFAULT_N_TAU`. Keyword-only.
    n_substeps : int, optional
        Field sub-steps per round trip, default 1 (the production path).
        Keyword-only.
    scan_direction : {'up', 'down'}, optional
        Which analytic branch to seed from; see :func:`stable_branch`.
        Keyword-only.
    seed_perturbation : float, optional
        Multiplicative offset [dimensionless] applied to the seed amplitude,
        default 1.02. Keyword-only.
    flatness_rtol : float or None, optional
        Required relative peak-to-peak [dimensionless] of the last 10% of
        ``U_int_history``. ``None`` (default) uses :func:`flatness_floor`.
        Keyword-only.
    seed_power : sequence of float or numpy.ndarray or None, optional
        Explicit seed powers [J], one per detuning, overriding the analytic
        branch. Used to probe a chosen branch. Keyword-only.
    return_diagnostics : bool, optional
        Also return per-detuning ``tail_flatness`` [dimensionless] and the
        final ``u_int`` [J*s]. Keyword-only.

    Returns
    -------
    numpy.ndarray or tuple
        ``|E|**2`` [J], shape ``(n_detuning,)``; or that array together with a
        diagnostics dict when ``return_diagnostics`` is set.

    Raises
    ------
    ValueError
        If ``delta_omega`` is not scalar or 1-D, or if ``seed_power`` does not
        match its shape.
    AssertionError
        If the run did not settle -- the last 10% of ``U_int_history`` has a
        relative peak-to-peak above ``flatness_rtol`` -- naming the worst
        detuning. Either the settle time is too short or the seeded branch is
        not stable.

    Notes
    -----
    All stochastic channels off (``NoiseConfig.all_off()``), zero dispersion
    (``beta = [0.0, 0.0]``), thermo-optic feedback neutralized via
    ``params.config_path``, flat initial condition.

    The initial field is flat and complex, built from the analytic branch under
    test but deliberately OFFSET by ``seed_perturbation`` in amplitude, so the
    solver must relax to its own fixed point rather than being handed it. The
    tail-flatness assertion below is what turns that into a real test: a seed
    outside the basin of attraction, or an unstable branch, produces a
    non-stationary tail and raises.

    All detunings are solved in ONE batched (vmapped) call, which is what keeps
    the 405-point sweep at ~80 s.

    ``U_int`` is the intracavity energy in J*s, so the flat-state field power
    is recovered as ``U_int/t_r`` [J].

    No ``Examples`` section: a single call runs 60000 round trips to reach the
    stationarity floor, which is exactly the long solve a doctest must not
    contain. :func:`verify` drives it over the acceptance grid, and
    ``tests/test_analytic_cw.py`` exercises it end to end.
    """
    import jax                                          # local: heavy import

    dws = np.atleast_1d(np.asarray(delta_omega, dtype=np.float64))
    if dws.ndim != 1:
        raise ValueError(f"delta_omega must be scalar or 1-D, got ndim={dws.ndim}.")
    if flatness_rtol is None:
        flatness_rtol = flatness_floor(params.kappa, params.t_r / int(n_substeps))

    if seed_power is None:
        seeds = np.array(
            [
                stable_branch(
                    analytic_cw_roots(pin, dw, params.kappa, params.kappa_c, params.gamma),
                    scan_direction,
                )
                for dw in dws
            ],
            dtype=np.float64,
        )
    else:
        seeds = np.asarray(seed_power, dtype=np.float64)
        if seeds.shape != dws.shape:
            raise ValueError(
                f"seed_power shape {seeds.shape} must match delta_omega shape {dws.shape}."
            )

    # Flat complex seed E = F / (kappa/2 + i*(delta_omega - gamma*P)), i.e. the
    # steady state's own amplitude AND phase, scaled off the fixed point.
    drive = math.sqrt(params.kappa_c * float(pin))
    e_seed = seed_perturbation * drive / (
        params.kappa / 2.0 + 1j * (dws - params.gamma * seeds)
    )
    e0 = np.repeat(e_seed[:, None], n_tau, axis=1).astype(np.complex128)
    dw_sweep = np.repeat(dws[:, None], int(t_slow), axis=1)

    out = solve_lle_ssfm_jax(
        pin=float(pin),
        delta_omega=dw_sweep,
        t_slow=int(t_slow),
        beta=[0.0, 0.0],                       # D_int == 0 identically
        kappa=params.kappa,
        kappa_c=params.kappa_c,
        rng_key=jax.random.PRNGKey(0),         # unused: no live noise channel
        n_tau=int(n_tau),
        config_path=params.config_path,
        snapshot_interval=int(t_slow),         # one snapshot: keep memory flat
        e0_override=e0,
        n_substeps=int(n_substeps),
        noise_config=NoiseConfig.all_off(),
    )

    u_int = np.asarray(out["U_int_history"], dtype=np.float64)   # (n_traj, t_slow), J*s
    tail = u_int[:, -max(1, int(0.1 * int(t_slow))):]
    tail_mean = np.abs(tail.mean(axis=1))
    flatness = np.where(
        tail_mean > 0.0,
        (tail.max(axis=1) - tail.min(axis=1)) / np.where(tail_mean > 0.0, tail_mean, 1.0),
        np.inf,
    )
    worst = int(np.argmax(flatness))
    if not np.all(flatness <= flatness_rtol):
        raise AssertionError(
            f"CW run did not settle: last 10% of U_int_history has relative "
            f"peak-to-peak {flatness[worst]:.3e} > {flatness_rtol:.1e} at "
            f"delta_omega/kappa = {dws[worst] / params.kappa:.4f} "
            f"(pin = {pin:.6e} W, t_slow = {t_slow}). Either the settle time is "
            f"too short or the seeded branch is not stable."
        )

    # U_int is the intracavity energy in J*s; the field power |E|^2 [J] of a
    # flat state is U_int / t_r (E_total = U_int/t_r).
    power = u_int[:, -1] / params.t_r
    if return_diagnostics:
        return power, {"tail_flatness": flatness, "u_int_final": u_int[:, -1]}
    return power


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def verify(
    rtol: float = 1e-12,
    *,
    params: CavityParams | None = None,
    detuning_grid: Sequence[float] | np.ndarray = DEFAULT_DETUNING_GRID,
    pin_multipliers: Sequence[float] = DEFAULT_PIN_MULTIPLIERS,
    t_slow: int = DEFAULT_T_SLOW,
    n_tau: int = DEFAULT_N_TAU,
    n_substeps: int = 1,
    scan_direction: str = "up",
) -> dict[str, Any]:
    """Sweep the CW state and compare against both exact references.

    The grid is ``detuning_grid`` (in units of kappa) crossed with
    ``pin_multipliers`` (in units of P_th) — 81 x 5 = 405 points by default.

    Parameters
    ----------
    rtol : float, optional
        Gate on the DISCRETE-MAP residual [dimensionless], default 1e-12.
    params : CavityParams or None, optional
        Resolved cavity constants; ``None`` (default) calls
        :func:`load_cavity_params`. Keyword-only.
    detuning_grid : sequence of float or numpy.ndarray, optional
        Detunings in units of ``kappa`` [dimensionless], default
        :data:`DEFAULT_DETUNING_GRID`. Keyword-only.
    pin_multipliers : sequence of float, optional
        Pump powers in units of ``P_th`` [dimensionless], default
        :data:`DEFAULT_PIN_MULTIPLIERS`. Keyword-only.
    t_slow : int, optional
        Round trips per solver run, default :data:`DEFAULT_T_SLOW`.
        Keyword-only.
    n_tau : int, optional
        Fast-time grid points, default :data:`DEFAULT_N_TAU`. Keyword-only.
    n_substeps : int, optional
        Field sub-steps per round trip, default 1. Keyword-only.
    scan_direction : {'up', 'down'}, optional
        Branch to seed from, default ``'up'``. Keyword-only.

    Returns
    -------
    dict
        Per point (all shaped ``(n_pin, n_detuning)``): ``p_analytic`` [J]
        (continuum branch), ``p_discrete`` [J] (exact map fixed point),
        ``p_solver`` [J], ``n_roots``, ``tail_flatness`` [dimensionless] and
        the residual arrays ``rel_continuum`` / ``rel_discrete``
        [dimensionless]. Plus the axes ``delta_over_kappa``, ``pin_over_pth``,
        ``delta_omega`` [rad/s]; the scalar summaries ``max_rel_discrete`` /
        ``max_rel_continuum`` with their argmax coordinates
        ``(pin_over_pth, delta_over_kappa)``; ``median_rel_discrete``,
        ``max_tail_flatness``, ``flatness_floor``, ``n_bistable``,
        ``n_points``; the echoed settings; and ``passed``.

    Raises
    ------
    AssertionError
        Propagated from :func:`solver_cw_steady_state` if any point failed to
        settle.
    RuntimeError
        Propagated from :func:`discrete_map_cw_fixed_point` if Newton failed at
        any point.

    Notes
    -----
    The grid is ``detuning_grid`` (in units of kappa) crossed with
    ``pin_multipliers`` (in units of P_th) -- 81 x 5 = 405 points by default.

    ``rtol`` gates the DISCRETE-MAP residual, which is the machine-precision
    claim: the continuum residual is a genuine first-order-in-dt mean-field
    error of ~3e-3 and cannot be gated at 1e-12. The continuum residual is
    measured and returned regardless, because characterizing it is the point --
    see :func:`mean_field_order_check`.

    No ``Examples`` section: the default grid runs 405 converged solves.
    """
    if params is None:
        params = load_cavity_params()

    grid = np.asarray(detuning_grid, dtype=np.float64)
    mults = np.asarray(pin_multipliers, dtype=np.float64)
    dws = grid * params.kappa
    dt = params.t_r / int(n_substeps)

    shape = (mults.size, grid.size)
    p_analytic = np.empty(shape, dtype=np.float64)
    p_discrete = np.empty(shape, dtype=np.float64)
    p_solver = np.empty(shape, dtype=np.float64)
    n_roots = np.empty(shape, dtype=np.int64)
    tail_flat = np.empty(shape, dtype=np.float64)

    for i, mult in enumerate(mults):
        pin = float(mult) * params.p_th
        for j, dw in enumerate(dws):
            roots = analytic_cw_roots(pin, dw, params.kappa, params.kappa_c, params.gamma)
            n_roots[i, j] = roots.size
            p_analytic[i, j] = stable_branch(roots, scan_direction)
            p_discrete[i, j] = discrete_map_cw_fixed_point(
                pin, dw, params.kappa, params.kappa_c, params.gamma, dt,
                p_guess=p_analytic[i, j],
            )
        solved, diag = solver_cw_steady_state(
            pin, dws, params,
            t_slow=t_slow, n_tau=n_tau, n_substeps=n_substeps,
            scan_direction=scan_direction, return_diagnostics=True,
        )
        p_solver[i] = solved
        tail_flat[i] = diag["tail_flatness"]

    rel_continuum = np.abs(p_solver - p_analytic) / p_analytic
    rel_discrete = np.abs(p_solver - p_discrete) / p_discrete

    def _argmax2d(arr: np.ndarray) -> tuple[float, float, float]:
        i, j = np.unravel_index(int(np.argmax(arr)), arr.shape)
        return float(arr[i, j]), float(mults[i]), float(grid[j])

    max_disc, disc_pin, disc_dw = _argmax2d(rel_discrete)
    max_cont, cont_pin, cont_dw = _argmax2d(rel_continuum)

    return {
        "params": params,
        "rtol": float(rtol),
        "scan_direction": scan_direction,
        "n_substeps": int(n_substeps),
        "t_slow": int(t_slow),
        "n_tau": int(n_tau),
        "delta_over_kappa": grid,
        "pin_over_pth": mults,
        "delta_omega": dws,
        "p_analytic": p_analytic,
        "p_discrete": p_discrete,
        "p_solver": p_solver,
        "n_roots": n_roots,
        "tail_flatness": tail_flat,
        "rel_continuum": rel_continuum,
        "rel_discrete": rel_discrete,
        "max_rel_discrete": max_disc,
        "max_rel_discrete_at": (disc_pin, disc_dw),
        "max_rel_continuum": max_cont,
        "max_rel_continuum_at": (cont_pin, cont_dw),
        "median_rel_discrete": float(np.median(rel_discrete)),
        "max_tail_flatness": float(np.max(tail_flat)),
        "flatness_floor": flatness_floor(params.kappa, dt),
        "n_bistable": int(np.sum(n_roots == 3)),
        "n_points": int(rel_discrete.size),
        "passed": bool(max_disc <= float(rtol)),
    }


def mean_field_order_check(
    params: CavityParams | None = None,
    *,
    delta_over_kappa: float = 2.0,
    pin_over_pth: float = 1.0,
    substeps: Sequence[int] = (1, 2, 4, 8),
    t_slow: int = 40_000,
    n_tau: int = DEFAULT_N_TAU,
) -> dict[str, Any]:
    """Measure the ORDER of the continuum residual in dt.

    The continuum residual is a splitting error, so it must fall like
    ``1/n_substeps``. Measuring the exponent (rather than pinning a magnitude)
    is what makes the reported 3e-3 a characterization instead of a magic
    number: a change to the splitting scheme moves the ORDER, which this
    catches, while an unrelated parameter change moves only the prefactor.

    Parameters
    ----------
    params : CavityParams or None, optional
        Resolved cavity constants; ``None`` (default) calls
        :func:`load_cavity_params`.
    delta_over_kappa : float, optional
        Operating detuning in units of ``kappa`` [dimensionless], default 2.0.
        Keyword-only.
    pin_over_pth : float, optional
        Operating pump power in units of ``P_th`` [dimensionless], default 1.0.
        Keyword-only.
    substeps : sequence of int, optional
        Refinement ladder ``n_substeps`` [dimensionless], default
        ``(1, 2, 4, 8)``. Keyword-only.
    t_slow : int, optional
        Round trips per run, default 40000. Keyword-only.
    n_tau : int, optional
        Fast-time grid points, default :data:`DEFAULT_N_TAU`. Keyword-only.

    Returns
    -------
    dict
        ``substeps`` (int array), ``rel_continuum`` (residual per level,
        dimensionless), ``ratios`` (successive residual ratios -- ~2 for first
        order at doubling), ``order`` (fitted from a log-log least squares),
        and the echoed operating point.

    Raises
    ------
    AssertionError
        Propagated from :func:`solver_cw_steady_state` if a level did not
        settle.
    numpy.linalg.LinAlgError
        Propagated from ``numpy.polyfit`` for a degenerate ladder.

    Notes
    -----
    The continuum residual is a SPLITTING error, so it must fall like
    ``1/n_substeps``. Measuring the exponent rather than pinning a magnitude is
    what makes the reported 3e-3 a characterization instead of a magic number:
    a change to the splitting scheme moves the ORDER, which this catches, while
    an unrelated parameter change moves only the prefactor.

    No ``Examples`` section: the ladder runs four 40000-round-trip solves.
    """
    if params is None:
        params = load_cavity_params()
    dw = float(delta_over_kappa) * params.kappa
    pin = float(pin_over_pth) * params.p_th
    p_an = stable_branch(
        analytic_cw_roots(pin, dw, params.kappa, params.kappa_c, params.gamma), "up"
    )
    residuals = []
    for n in substeps:
        solved = solver_cw_steady_state(
            pin, dw, params, t_slow=t_slow, n_tau=n_tau, n_substeps=int(n),
        )
        residuals.append(abs(float(solved[0]) - p_an) / p_an)
    residuals_arr = np.asarray(residuals, dtype=np.float64)
    subs = np.asarray(substeps, dtype=np.float64)
    slope = float(np.polyfit(np.log(subs), np.log(residuals_arr), 1)[0])
    return {
        "delta_over_kappa": float(delta_over_kappa),
        "pin_over_pth": float(pin_over_pth),
        "substeps": subs.astype(int),
        "rel_continuum": residuals_arr,
        "ratios": residuals_arr[:-1] / residuals_arr[1:],
        "order": -slope,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_report(result: dict[str, Any], *, full: bool) -> None:
    params: CavityParams = result["params"]
    grid = result["delta_over_kappa"]
    mults = result["pin_over_pth"]

    print()
    print("=" * 78)
    print("ANALYTIC CW VERIFICATION  (validation/analytic_cw.py)")
    print("=" * 78)
    print(f"  source config : {params.source_config}")
    print(f"  derived config: {params.config_path}  (dn_dT_per_k = 0)")
    print(
        f"  kappa = {params.kappa:.6e} rad/s   kappa_c = {params.kappa_c:.6e} rad/s   "
        f"kappa_i = {params.kappa_i:.6e} rad/s"
    )
    print(
        f"  gamma = {params.gamma:.6e} J^-1 s^-1   t_r = {params.t_r:.6e} s   "
        f"kappa*t_r = {params.kappa * params.t_r:.6e}"
    )
    print(f"  P_th  = {params.p_th * 1e3:.6f} mW")
    print(
        f"  grid  : {grid.size} detunings in [{grid[0]:g}, {grid[-1]:g}]*kappa  x  "
        f"{mults.size} pumps {tuple(float(m) for m in mults)}*P_th "
        f"= {result['n_points']} points"
    )
    print(
        f"  solver: t_slow = {result['t_slow']}, n_tau = {result['n_tau']}, "
        f"n_substeps = {result['n_substeps']}, branch = {result['scan_direction']}-scan, "
        f"all noise off, beta = [0, 0]"
    )
    print(f"  bistable points on the grid: {result['n_bistable']}")
    print(
        f"  worst tail flatness (last 10% of U_int_history): "
        f"{result['max_tail_flatness']:.3e}  "
        f"(float64 stationarity floor {result['flatness_floor']:.3e} = "
        f"{FLATNESS_SAFETY:g}*eps/(1-exp(-kappa*dt/2)))"
    )

    print()
    print("Per-pump summary (|E|^2 relative residual)")
    print("-" * 78)
    print(
        f"{'pin/P_th':>9} {'#bist':>6} | {'discrete map':>34} | {'continuum LLE':>20}"
    )
    print(
        f"{'':>9} {'':>6} | {'max':>12} {'@dw/k':>8} {'median':>12} | "
        f"{'max':>12} {'@dw/k':>7}"
    )
    print("-" * 78)
    for i, mult in enumerate(mults):
        rd = result["rel_discrete"][i]
        rc = result["rel_continuum"][i]
        nb = int(np.sum(result["n_roots"][i] == 3))
        print(
            f"{mult:>9.1f} {nb:>6d} | {rd.max():>12.4e} {grid[int(rd.argmax())]:>8.2f} "
            f"{np.median(rd):>12.4e} | {rc.max():>12.4e} "
            f"{grid[int(rc.argmax())]:>7.2f}"
        )
    print("-" * 78)

    print()
    print("Point table (every 5th detuning; --full for all)" if not full
          else "Point table (all points)")
    print("-" * 78)
    print(
        f"{'pin/P_th':>8} {'dw/kappa':>9} {'#roots':>6} {'P_analytic [J]':>16} "
        f"{'P_solver [J]':>16} {'rel_disc':>10} {'rel_cont':>10}"
    )
    print("-" * 78)
    stride = 1 if full else 5
    for i, mult in enumerate(mults):
        for j in range(0, grid.size, stride):
            print(
                f"{mult:>8.1f} {grid[j]:>9.2f} {result['n_roots'][i, j]:>6d} "
                f"{result['p_analytic'][i, j]:>16.9e} {result['p_solver'][i, j]:>16.9e} "
                f"{result['rel_discrete'][i, j]:>10.3e} "
                f"{result['rel_continuum'][i, j]:>10.3e}"
            )
    print("-" * 78)

    print()
    print("RESULT")
    print("-" * 78)
    dp, dd = result["max_rel_discrete_at"]
    cp, cd = result["max_rel_continuum_at"]
    print(
        f"  max |rel residual| vs EXACT DISCRETE MAP : "
        f"{result['max_rel_discrete']:.6e}  at pin = {dp:g}*P_th, "
        f"dw = {dd:+g}*kappa   (median {result['median_rel_discrete']:.3e})"
    )
    print(
        f"  max |rel residual| vs CONTINUUM LLE CUBIC: "
        f"{result['max_rel_continuum']:.6e}  at pin = {cp:g}*P_th, "
        f"dw = {cd:+g}*kappa"
    )
    print(f"  gate: discrete-map residual <= rtol = {result['rtol']:.1e}")
    print(f"  {'PASS' if result['passed'] else 'FAIL'}")


def _print_order_check(order: dict[str, Any]) -> None:
    print()
    print("Mean-field (continuum) residual vs n_substeps  "
          f"[dw = {order['delta_over_kappa']:+g}*kappa, "
          f"pin = {order['pin_over_pth']:g}*P_th]")
    print("-" * 78)
    print(f"{'n_substeps':>11} {'rel_continuum':>16} {'ratio to previous':>19}")
    for k, (n, r) in enumerate(zip(order["substeps"], order["rel_continuum"])):
        ratio = "" if k == 0 else f"{order['ratios'][k - 1]:>19.4f}"
        print(f"{int(n):>11d} {r:>16.6e} {ratio}")
    print("-" * 78)
    print(
        f"  fitted order in dt: {order['order']:.4f}  "
        f"(1 = first-order splitting error; the pump kick is applied outside "
        f"the linear exponential)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CW verification from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments; ``None`` (default) reads ``sys.argv[1:]``.
        Supports ``--rtol``, ``--t-slow``, ``--n-tau``, ``--n-substeps``,
        ``--config``, ``--full``, ``--quick`` and ``--no-order-check``.

    Returns
    -------
    int
        0 if the discrete-map residual is within ``--rtol`` at every grid
        point, 1 otherwise -- the process exit status.

    Raises
    ------
    SystemExit
        From ``argparse`` on a malformed command line or ``--help``.
    AssertionError
        Propagated from :func:`verify` if a point failed to settle.

    Notes
    -----
    Prints the report to stdout and returns a status rather than asserting, so
    the module doubles as a CI gate (``python -m validation.analytic_cw``) and
    as an interactive tool. ``--quick`` reduces the grid to 21 x 2 points at
    ``t_slow = 20000``.

    No ``Examples`` section: every path runs the full sweep.
    """
    ap = argparse.ArgumentParser(
        prog="python -m validation.analytic_cw",
        description=(
            "Verify the solver's homogeneous CW steady state against exact "
            "mathematics: the continuum LLE cubic and the exact fixed point of "
            "the solver's own discrete map."
        ),
    )
    ap.add_argument("--rtol", type=float, default=1e-12,
                    help="gate on the discrete-map residual (default 1e-12)")
    ap.add_argument("--t-slow", type=int, default=DEFAULT_T_SLOW)
    ap.add_argument("--n-tau", type=int, default=DEFAULT_N_TAU)
    ap.add_argument("--n-substeps", type=int, default=1)
    ap.add_argument("--config", type=str, default=None,
                    help="source config (default config/sin_params.yaml)")
    ap.add_argument("--full", action="store_true", help="print every grid point")
    ap.add_argument("--quick", action="store_true",
                    help="reduced grid (21 detunings x 2 pumps, t_slow=20000)")
    ap.add_argument("--no-order-check", action="store_true",
                    help="skip the n_substeps convergence-order measurement")
    args = ap.parse_args(argv)

    params = load_cavity_params(args.config)
    grid = np.linspace(-5.0, 15.0, 21) if args.quick else DEFAULT_DETUNING_GRID
    mults = (1.0, 10.0) if args.quick else DEFAULT_PIN_MULTIPLIERS
    t_slow = 20_000 if args.quick else args.t_slow

    result = verify(
        args.rtol,
        params=params,
        detuning_grid=grid,
        pin_multipliers=mults,
        t_slow=t_slow,
        n_tau=args.n_tau,
        n_substeps=args.n_substeps,
    )
    _print_report(result, full=args.full)

    if not args.no_order_check:
        _print_order_check(mean_field_order_check(params, n_tau=args.n_tau))

    print()
    if result["passed"]:
        print(
            f"OK: the solver reproduces the exact fixed point of its own discrete "
            f"map to {result['max_rel_discrete']:.3e} at every one of "
            f"{result['n_points']} points."
        )
        print(
            f"NOTE: the residual against the CONTINUUM LLE cubic is "
            f"{result['max_rel_continuum']:.3e}. That is the mean-field "
            f"(Ikeda-map -> LLE) truncation of the splitting, first order in dt "
            f"and of size ~kappa*t_r/2 = "
            f"{params.kappa * params.t_r / 2.0:.3e}; it is NOT a convergence "
            f"artifact (see the module docstring)."
        )
    else:
        print(
            f"FAILED: max discrete-map residual {result['max_rel_discrete']:.6e} "
            f"exceeds rtol {result['rtol']:.1e}."
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
