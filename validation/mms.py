"""Method of Manufactured Solutions for the implemented LLE.

What MMS buys that a self-convergence study does not
---------------------------------------------------
Self-convergence (comparing dt against dt/2) measures the *rate* at which a
scheme converges to its own limit — it cannot tell you the limit is the right
PDE. MMS pins both: a smooth field ``E_mms(tau, t)`` is chosen first, the
forcing ``S = dE/dt - RHS[E]`` that makes it an exact solution is derived
symbolically, and the solver is then run with that forcing. Any discrepancy
between the solver's answer and ``E_mms`` at the final time is discretization
error and nothing else. A sign error in the dispersion operator, a missing
factor in the Kerr term, or a mis-scaled pump would all show up as a solution
that fails to converge to ``E_mms`` at all, rather than as a clean but wrong
convergence rate.

The manufactured solution
-------------------------
    E_mms(tau, t) = A * exp(-((tau - tau0)/w)^2) * exp(i*(k*tau + omega_m*t))

with ``k = 2*pi*m_k/t_r`` for an integer ``m_k``, so the phase factor is
EXACTLY periodic on [0, t_r).

The Gaussian envelope is not exactly periodic, but it is periodic to far below
any error this study resolves. At the defaults (``w = t_r/16``,
``tau0 = t_r/2``) the envelope at the domain edge is
``exp(-(t_r/2 / (t_r/16))^2) = exp(-64) = 1.6e-28``, and its Fourier content at
the Nyquist mode of a 128-point grid is ``exp(-(omega_max*w/2)^2) ~ 1e-69``. So
the field is periodic to ~1e-28 and spectrally resolved to ~1e-69, both
astronomically below the ~1e-6..1e-3 discretization errors being measured.
:func:`periodicity_defect` reports both numbers so this stays checkable rather
than asserted.

Deriving the forcing
--------------------
The continuum limit of what ``simulator/lle_solver.py`` integrates (see
``_fine_step``) is

    dE/dt = -(kappa/2)*E - i*D_hat*E - i*delta_omega*E + i*gamma*|E|^2*E + F + S

where ``F = sqrt(kappa_c*pin)`` and ``D_hat`` is the dispersion operator. The
solver applies ``exp(-1j*disp(omega)*dt)`` to ``fft(E)`` with
``disp(omega) = sum_k beta_k/k! * omega^k`` and ``omega = 2*pi*fftfreq(...)``.
Under numpy's FFT sign convention a bin at ``omega`` corresponds to the real-
space mode ``exp(+i*omega*tau)``, so ``omega -> -i*d/dtau`` and

    -i*D_hat*E = -i * sum_k (beta_k/k!) * (-i*d/dtau)^k E

which for k=2 is ``+i*(beta_2/2)*d2E/dtau2`` (the usual anomalous-dispersion
term for beta_2 > 0, matching the repo's sign convention) and for k=3 is
``+(beta_3/6)*d3E/dtau3``. Getting this wrong is the single most likely way to
build a plausible-but-wrong MMS, so :func:`residual_check` re-derives the same
residual SPECTRALLY — applying ``disp(omega)`` in Fourier space exactly as the
solver does — and compares. The two agree to ~1e-14 relative or the test fails.

Why the forcing is cheap at run time
------------------------------------
Every term of ``S`` except the constant pump ``-F`` is linear in ``E`` or
involves ``|E|^2`` (which is ``t``-independent), so the whole ``t`` dependence
factors out:

    S(tau, t) = Psi(tau) * exp(i*omega_m*t) - F

``Psi`` is derived once with sympy, evaluated on the tau grid host-side in
float64, and the traced forcing is then two array ops per sub-step. Nothing is
lambdified into the JAX trace. :func:`manufactured_source` asserts the
factorization symbolically (``Psi`` must contain no ``t``) rather than assuming
it, so a change to the manufactured solution that breaks separability fails
loudly instead of silently dropping a term.

How the forcing enters the solver
---------------------------------
``source_fn`` is added as ``S(t)*dt_sub`` in the SAME place and the SAME way as
the pump kick. That is deliberate: MMS then measures the order of the scheme as
it actually runs. With the legacy drive treatment the measured order is 1, not
2 — see ``validation/convergence.py`` and the ``symmetric_drive`` note in
``simulator/lle_solver.py``.

sympy is imported lazily, inside the derivation, so importing ``validation`` or
running any other validation module does not require it.

Reference: Herr, Tikan & Kippenberg, arXiv:2604.05897v1 (7 Apr 2026).
"""

from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.analytic_cw import CavityParams, load_cavity_params     # noqa: E402

__all__ = [
    "MMSSolution",
    "DEFAULT_BETA",
    "DEFAULT_N_TAU",
    "DEFAULT_T_FINAL_ROUND_TRIPS",
    "manufactured_solution",
    "manufactured_source",
    "make_source_fn",
    "mms_error",
    "periodicity_defect",
    "residual_check",
]

#: beta_2 [s] from the committed config's D2 (d2_to_beta2_lle(3.76991e4, 2.46e10)).
#: Non-zero so the MMS exercises the dispersion operator rather than skipping it.
DEFAULT_BETA: tuple[float, float] = (1.578e-18, 0.0)

DEFAULT_N_TAU = 128

#: Final time in round trips. 200*t_r = 8.13 ns ~ 1.2 cavity lifetimes (2/kappa),
#: long enough for the transient to matter and short enough to refine 6 times.
DEFAULT_T_FINAL_ROUND_TRIPS = 200


@dataclass(frozen=True)
class MMSSolution:
    """The manufactured solution and everything derived from it."""

    amplitude: float          # A, sqrt(J)
    width: float              # w, s
    center: float             # tau0, s
    mode_number: int          # m_k, so k = 2*pi*m_k/t_r
    omega_m: float            # temporal angular frequency, rad/s
    pin: float                # pump power [W] the run is driven with
    delta_omega: float        # detuning [rad/s]
    beta: tuple[float, ...]   # dispersion coefficients handed to the solver
    params: CavityParams

    @property
    def k(self) -> float:
        """Fast-time wavenumber [rad/s], an exact integer multiple of 2*pi/t_r."""
        return 2.0 * math.pi * self.mode_number / self.params.t_r

    def tau_grid(self, n_tau: int) -> np.ndarray:
        return np.arange(n_tau) * (self.params.t_r / n_tau)

    def field(self, tau: np.ndarray, t: float) -> np.ndarray:
        """``E_mms(tau, t)`` [sqrt(J)]."""
        env = self.amplitude * np.exp(-(((tau - self.center) / self.width) ** 2))
        return (env * np.exp(1j * (self.k * tau + self.omega_m * t))).astype(np.complex128)


def manufactured_solution(
    params: CavityParams | None = None,
    *,
    pin_over_pth: float = 2.0,
    delta_over_kappa: float = 2.0,
    beta: Sequence[float] = DEFAULT_BETA,
    mode_number: int = 3,
    width_fraction: float = 16.0,
    omega_m_over_kappa: float = 2.0,
    kerr_over_kappa: float = 0.68,
) -> MMSSolution:
    """Build the manufactured solution at physically meaningful scales.

    ``kerr_over_kappa`` sets the amplitude through ``gamma*A^2 = f*kappa``, so
    the Kerr term is a real fraction of the linear rates rather than a
    perturbation the study would be blind to. The default 0.68 puts
    ``gamma*|E|^2`` at the same order as ``kappa/2`` and ``delta_omega``.
    """
    if params is None:
        params = load_cavity_params()
    amplitude = math.sqrt(kerr_over_kappa * params.kappa / params.gamma)
    return MMSSolution(
        amplitude=amplitude,
        width=params.t_r / float(width_fraction),
        center=params.t_r / 2.0,
        mode_number=int(mode_number),
        omega_m=float(omega_m_over_kappa) * params.kappa,
        pin=float(pin_over_pth) * params.p_th,
        delta_omega=float(delta_over_kappa) * params.kappa,
        beta=tuple(float(b) for b in beta),
        params=params,
    )


def periodicity_defect(sol: MMSSolution, n_tau: int = DEFAULT_N_TAU) -> dict[str, float]:
    """Quantify the two approximations in the manufactured solution.

    Returns ``envelope_at_edge`` (the Gaussian's value at the domain boundary
    relative to its peak — the periodicity defect) and ``spectral_tail`` (its
    Fourier amplitude at the Nyquist mode relative to the peak — the resolution
    defect). Both must sit far below the discretization errors being measured.
    """
    half = sol.params.t_r / 2.0
    envelope_at_edge = math.exp(-((half / sol.width) ** 2))
    omega_max = math.pi * n_tau / sol.params.t_r
    spectral_tail = math.exp(-((omega_max * sol.width / 2.0) ** 2))
    return {
        "envelope_at_edge": envelope_at_edge,
        "spectral_tail": spectral_tail,
        "samples_per_width": n_tau * sol.width / sol.params.t_r,
    }


def manufactured_source(
    sol: MMSSolution, n_tau: int = DEFAULT_N_TAU
) -> tuple[np.ndarray, float]:
    """Derive ``Psi(tau)`` with sympy, so that ``S(tau,t) = Psi*exp(i*w_m*t) - F``.

    Returns ``(Psi, F)`` with ``Psi`` complex128 of shape ``(n_tau,)`` and
    ``F = sqrt(kappa_c*pin)``.

    The residual is assembled term by term from

        S = dE/dt + (kappa/2)*E + i*D_applied + i*delta_omega*E
            - i*gamma*|E|^2*E - F,
        D_applied = sum_k (beta_k/k!) * (-i*d/dtau)^k E

    and the ``exp(i*omega_m*t)`` factorization is CHECKED symbolically rather
    than assumed.
    """
    import sympy as sp                                   # lazy: see module docstring

    tau, t = sp.symbols("tau t", real=True)
    p = sol.params

    A = sp.Float(sol.amplitude, 30)
    w = sp.Float(sol.width, 30)
    tau0 = sp.Float(sol.center, 30)
    k = 2 * sp.pi * sp.Integer(sol.mode_number) / sp.Float(p.t_r, 30)
    wm = sp.Float(sol.omega_m, 30)
    kappa = sp.Float(p.kappa, 30)
    gamma = sp.Float(p.gamma, 30)
    dw = sp.Float(sol.delta_omega, 30)

    envelope = A * sp.exp(-(((tau - tau0) / w) ** 2))    # real and positive
    phase = k * tau + wm * t
    E = envelope * sp.exp(sp.I * phase)
    mod_sq = envelope ** 2                               # |E|^2, exactly real

    # Dispersion: omega -> -i d/dtau under numpy's FFT sign convention.
    d_applied = sp.Integer(0)
    for i, b in enumerate(sol.beta):
        order = i + 2
        if b == 0.0:
            continue
        coeff = sp.Float(b, 30) / sp.factorial(order)
        d_applied += coeff * (-sp.I) ** order * sp.diff(E, tau, order)

    drive = sp.Float(math.sqrt(p.kappa_c * sol.pin), 30)

    # The envelope is factored out BY CONSTRUCTION rather than left for
    # simplify() to find. Every term except the Kerr one is linear in E, hence
    # of the form envelope*polynomial(tau)*exp(i*phase); dividing the linear
    # part by envelope*exp(i*phase) cancels the Gaussian exactly and leaves a
    # polynomial. Doing it the other way round -- expanding the whole residual
    # and simplifying -- produces terms like exp(+a*tau^2)*exp(-b*tau^2) with
    # a < b, which is mathematically decaying but OVERFLOWS to inf*0 = nan when
    # lambdify evaluates the two factors separately in float64.
    linear_part = (
        sp.diff(E, t) + (kappa / 2) * E + sp.I * d_applied + sp.I * dw * E
    )
    linear_bracket = sp.simplify(
        sp.expand(linear_part * sp.exp(-sp.I * phase) / envelope)
    )
    if linear_bracket.has(sp.exp):
        linear_bracket = sp.powsimp(linear_bracket, force=True)
    if linear_bracket.has(t):
        raise AssertionError(
            "the manufactured source did not factor as Psi(tau)*exp(i*omega_m*t): "
            "the chosen solution is not separable, so make_source_fn's two-op "
            "runtime form would silently drop the residual t dependence."
        )

    # The Kerr term: -i*gamma*|E|^2*E / (envelope*exp(i*phase)) = -i*gamma*envelope^2,
    # a DECAYING Gaussian, so it is safe to keep in exponential form.
    kerr_bracket = -sp.I * gamma * mod_sq

    # Psi(tau) = envelope * exp(i*k*tau) * (linear + kerr). The residual's
    # constant -F term carries no exp(i*omega_m*t) and is applied at run time.
    psi_expr = envelope * sp.exp(sp.I * k * tau) * (linear_bracket + kerr_bracket)

    psi_fn = sp.lambdify(tau, psi_expr, modules="numpy")
    grid = sol.tau_grid(n_tau)
    psi = np.asarray(psi_fn(grid), dtype=np.complex128)
    if psi.shape != (n_tau,):                            # constant-folded expression
        psi = np.broadcast_to(psi, (n_tau,)).astype(np.complex128).copy()
    if not np.all(np.isfinite(psi)):
        raise AssertionError("manufactured source contains non-finite values.")
    return psi, float(drive)


def residual_check(
    sol: MMSSolution, n_tau: int = DEFAULT_N_TAU, t_eval: float = 0.0
) -> float:
    """Independent SPECTRAL re-derivation of the source; returns the relative diff.

    Applies ``disp(omega) = sum_k beta_k/k! * omega^k`` in Fourier space exactly
    as ``_fine_step`` does, instead of differentiating symbolically in tau. If
    the sign convention assumed in :func:`manufactured_source` were wrong, the
    two would disagree at O(1). Returns ``||S_sym - S_spec|| / ||S_spec||``.
    """
    p = sol.params
    grid = sol.tau_grid(n_tau)
    psi, drive = manufactured_source(sol, n_tau)
    s_sym = psi * np.exp(1j * sol.omega_m * t_eval) - drive

    e = sol.field(grid, t_eval)
    omega = 2.0 * np.pi * np.fft.fftfreq(n_tau, d=p.t_r / n_tau)
    disp = np.zeros_like(omega)
    for i, b in enumerate(sol.beta):
        disp = disp + float(b) / math.factorial(i + 2) * omega ** (i + 2)

    de_dt = 1j * sol.omega_m * e
    rhs = (
        -(p.kappa / 2.0) * e
        - 1j * np.fft.ifft(disp * np.fft.fft(e))
        - 1j * sol.delta_omega * e
        + 1j * p.gamma * np.abs(e) ** 2 * e
        + drive
    )
    s_spec = de_dt - rhs
    return float(np.linalg.norm(s_sym - s_spec) / np.linalg.norm(s_spec))


@functools.lru_cache(maxsize=None)
def _cached_source_fn(psi_bytes: bytes, n_tau: int, omega_m: float, drive: float):
    """Build the traced forcing closure once per (Psi, omega_m, F).

    Keyed on the raw bytes of ``Psi`` so repeated calls with the same
    manufactured solution return the SAME callable object. That matters: the
    solver's ``_per_traj_variant`` caches its jit on ``source_fn`` identity, so
    a fresh closure per call would recompile on every dt of the study.
    """
    import jax.numpy as jnp

    psi = jnp.asarray(np.frombuffer(psi_bytes, dtype=np.complex128).reshape(n_tau))
    om = float(omega_m)
    f = float(drive)

    def source_fn(t):
        return psi * jnp.exp(1j * om * t) - f

    return source_fn


def make_source_fn(sol: MMSSolution, n_tau: int = DEFAULT_N_TAU):
    """The jax-traceable ``source_fn`` for ``solve_lle_ssfm_jax``."""
    psi, drive = manufactured_source(sol, n_tau)
    return _cached_source_fn(
        np.ascontiguousarray(psi).tobytes(), int(n_tau), sol.omega_m, drive
    )


def mms_error(
    dt: float,
    n_tau: int = DEFAULT_N_TAU,
    *,
    sol: MMSSolution | None = None,
    t_final_round_trips: int = DEFAULT_T_FINAL_ROUND_TRIPS,
    symmetric_drive: bool = False,
    thermal_coupling: str = "lagged",
    thermal_integrator: str = "euler",
) -> float:
    """Relative L2 error of the solver against ``E_mms`` at the final time.

    ``dt`` is the field sub-step in seconds. It must divide ``t_r`` an integer
    number of times: the refinement knob is ``fine_cadence_M = t_r/dt`` with
    ``n_substeps = 1``, which refines the field step, the thermal step, the
    detuning and the energy balance together, so a single dt controls the whole
    scheme.

    The run uses the derived thermo-optic-free config
    (``CavityParams.config_path``, ``dn_dT_per_k = 0``) so the manufactured
    solution is not perturbed by thermal feedback, and ``NoiseConfig.all_off()``.
    """
    import jax
    from simulator.lle_solver import solve_lle_ssfm_jax
    from simulator.noise_config import NoiseConfig

    if sol is None:
        sol = manufactured_solution()
    p = sol.params

    ratio = p.t_r / float(dt)
    fine_cadence_M = int(round(ratio))
    if fine_cadence_M < 1 or abs(ratio - fine_cadence_M) > 1e-9 * fine_cadence_M:
        raise ValueError(
            f"dt = {dt:.6e} s must divide t_r = {p.t_r:.6e} s an integer number "
            f"of times (got t_r/dt = {ratio:.9f})."
        )

    grid = sol.tau_grid(n_tau)
    e0 = sol.field(grid, 0.0)
    t_slow = int(t_final_round_trips)

    out = solve_lle_ssfm_jax(
        pin=sol.pin,
        delta_omega=sol.delta_omega,
        t_slow=t_slow,
        beta=list(sol.beta),
        kappa=p.kappa,
        kappa_c=p.kappa_c,
        rng_key=jax.random.PRNGKey(0),
        n_tau=int(n_tau),
        config_path=p.config_path,
        snapshot_interval=t_slow,
        e0_override=e0,
        fine_cadence_M=fine_cadence_M,
        n_substeps=1,
        source_fn=make_source_fn(sol, n_tau),
        symmetric_drive=symmetric_drive,
        thermal_coupling=thermal_coupling,
        noise_config=NoiseConfig.all_off(thermal_integrator=thermal_integrator),
    )

    e_num = np.asarray(out["e_final"])[0]
    e_ref = sol.field(grid, t_slow * p.t_r)
    return float(np.linalg.norm(e_num - e_ref) / np.linalg.norm(e_ref))
