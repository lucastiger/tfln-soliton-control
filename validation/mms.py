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
    """The manufactured solution and everything derived from it.

    Parameters
    ----------
    amplitude : float
        Peak field amplitude A [sqrt(J)].
    width : float
        Gaussian envelope 1/e half-width w [s] in fast time.
    center : float
        Envelope centre tau0 [s] within the round trip.
    mode_number : int
        Integer mode index m_k [dimensionless] of the carrier, so that
        ``k = 2*pi*m_k/t_r``.
    omega_m : float
        Temporal angular frequency [rad/s] of the slow-time phase rotation.
    pin : float
        Pump power [W] the run is driven with.
    delta_omega : float
        Detuning ``omega_res - omega_pump`` [rad/s].
    beta : tuple of float
        Dispersion coefficients handed to the solver: beta2 [s], beta3
        [s**2], ...
    params : CavityParams
        Cavity rates, nonlinearity and derived config the run uses.

    Raises
    ------
    dataclasses.FrozenInstanceError
        On any attempt to mutate an instance; the solution is the ground truth
        of the study and must not drift between refinement levels.

    Notes
    -----
    The manufactured field is

        E_mms(tau, t) = A * exp(-((tau - tau0)/w)**2)
                          * exp(1j*(k*tau + omega_m*t))     [sqrt(J)],

    a Gaussian envelope carried by an exact grid mode. The carrier mode number
    is an INTEGER so that ``k`` is an exact multiple of ``2*pi/t_r`` and the
    carrier is periodic on the domain to machine precision; only the Gaussian
    envelope is non-periodic, and :func:`periodicity_defect` bounds that defect.

    Examples
    --------
    >>> sol = manufactured_solution()
    >>> sol.mode_number
    3
    >>> abs(sol.k * sol.params.t_r / (2 * 3.141592653589793) - 3) < 1e-9
    True
    >>> grid = sol.tau_grid(8)
    >>> grid.shape, float(grid[0])
    ((8,), 0.0)
    >>> field = sol.field(grid, 0.0)
    >>> print(field.shape, field.dtype)
    (8,) complex128
    """

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
        """Fast-time wavenumber of the carrier.

        Returns
        -------
        float
            ``2*pi*mode_number/t_r`` [rad/s], an exact integer multiple of
            ``2*pi/t_r`` and therefore exactly representable on the FFT grid.

        Notes
        -----
        Using an integer mode number is what keeps the carrier periodic on the
        domain: a non-integer ``k`` would leak a discontinuity into the spectrum
        and pollute the very truncation error the study measures.

        Examples
        --------
        >>> sol = manufactured_solution(mode_number=3)
        >>> abs(sol.k - 2 * 3.141592653589793 * 3 / sol.params.t_r) < 1.0
        True
        """
        return 2.0 * math.pi * self.mode_number / self.params.t_r

    def tau_grid(self, n_tau: int) -> np.ndarray:
        """Fast-time sample points of one round trip.

        Parameters
        ----------
        n_tau : int
            Number of fast-time grid points.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_tau,)``, dtype ``float64``, units s: the uniform grid
            ``[0, t_r)`` with spacing ``t_r/n_tau``.

        Raises
        ------
        ValueError
            Propagated from ``numpy.arange`` for a negative ``n_tau``.

        Notes
        -----
        Half-open, matching the solver's FFT convention: the endpoint ``t_r``
        is the same physical point as 0 and must not be sampled twice.

        Examples
        --------
        >>> sol = manufactured_solution()
        >>> grid = sol.tau_grid(4)
        >>> grid.shape
        (4,)
        >>> bool(abs(grid[1] - sol.params.t_r / 4) < 1e-24)
        True
        """
        return np.arange(n_tau) * (self.params.t_r / n_tau)

    def field(self, tau: np.ndarray, t: float) -> np.ndarray:
        """Evaluate the manufactured field ``E_mms(tau, t)``.

        Parameters
        ----------
        tau : numpy.ndarray
            Fast-time coordinates [s], normally from :meth:`tau_grid`.
        t : float
            Slow time [s] since the start of the run.

        Returns
        -------
        numpy.ndarray
            Same shape as ``tau``, dtype ``complex128``, units sqrt(J).

        Notes
        -----
        This is the EXACT solution the solver is scored against -- both as the
        initial condition (``t = 0``) and as the reference at the final time.
        Its slow-time dependence is a pure phase rotation, so ``|E_mms|`` is
        stationary and any amplitude error the solver shows is entirely
        numerical.

        Examples
        --------
        >>> import numpy as np
        >>> sol = manufactured_solution()
        >>> e0 = sol.field(sol.tau_grid(64), 0.0)
        >>> print(e0.shape, e0.dtype)
        (64,) complex128
        >>> et = sol.field(sol.tau_grid(64), 1e-9)
        >>> bool(np.allclose(np.abs(e0), np.abs(et)))   # phase rotation only
        True
        """
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

    Parameters
    ----------
    params : CavityParams or None, optional
        Cavity rates and nonlinearity. ``None`` (default) calls
        :func:`~validation.analytic_cw.load_cavity_params`, which also derives
        the thermo-optic-free config the study runs against.
    pin_over_pth : float, optional
        Pump power in units of the MI threshold ``p_th`` [dimensionless],
        default 2.0. Keyword-only.
    delta_over_kappa : float, optional
        Detuning in units of ``kappa`` [dimensionless], default 2.0 -- inside
        the soliton existence window. Keyword-only.
    beta : sequence of float, optional
        Dispersion coefficients starting at beta2 [s], default
        :data:`DEFAULT_BETA`. Non-zero so the study exercises the dispersion
        operator instead of skipping it. Keyword-only.
    mode_number : int, optional
        Integer carrier mode index [dimensionless], default 3. Keyword-only.
    width_fraction : float, optional
        Envelope width as ``t_r/width_fraction`` [dimensionless], default 16.0.
        Keyword-only.
    omega_m_over_kappa : float, optional
        Slow-time phase rotation rate in units of ``kappa`` [dimensionless],
        default 2.0. Keyword-only.
    kerr_over_kappa : float, optional
        Kerr shift in units of ``kappa`` [dimensionless], default 0.68.
        Keyword-only.

    Returns
    -------
    MMSSolution
        The frozen solution record, with ``amplitude`` [sqrt(J)] derived from
        ``kerr_over_kappa`` and ``pin`` [W] from ``pin_over_pth``.

    Raises
    ------
    ZeroDivisionError
        If ``params.gamma`` or ``width_fraction`` is zero.
    OSError
        Propagated from :func:`~validation.analytic_cw.load_cavity_params` when
        ``params`` is ``None`` and the config cannot be read or the derived
        config cannot be written.

    Notes
    -----
    Every knob is expressed in units of a physical rate rather than as a bare
    number, so the manufactured problem sits at the same scales as a real run.
    ``kerr_over_kappa`` sets the amplitude through ``gamma*A**2 = f*kappa``, so
    the Kerr term is a real FRACTION of the linear rates rather than a
    perturbation the study would be blind to; the default 0.68 puts
    ``gamma*|E|**2`` at the same order as ``kappa/2`` and ``delta_omega``.

    A manufactured solution chosen at a convenient but unphysical amplitude
    would measure the order of the LINEAR part of the scheme only, and would
    report second order for a scheme whose nonlinear coupling is first order.

    Examples
    --------
    >>> sol = manufactured_solution()
    >>> p = sol.params
    >>> bool(abs(sol.pin / p.p_th - 2.0) < 1e-12)
    True
    >>> bool(abs(sol.delta_omega / p.kappa - 2.0) < 1e-12)
    True
    >>> bool(abs(p.gamma * sol.amplitude ** 2 / p.kappa - 0.68) < 1e-9)
    True
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

    Parameters
    ----------
    sol : MMSSolution
        The manufactured solution to characterize.
    n_tau : int, optional
        Fast-time grid size the study will run at, default
        :data:`DEFAULT_N_TAU`.

    Returns
    -------
    dict of str to float
        ``envelope_at_edge`` -- the Gaussian's value at the domain boundary
        relative to its peak [dimensionless], i.e. the PERIODICITY defect;
        ``spectral_tail`` -- its Fourier amplitude at the Nyquist mode relative
        to the peak [dimensionless], i.e. the RESOLUTION defect;
        ``samples_per_width`` -- grid points per envelope width
        [dimensionless].

    Raises
    ------
    ZeroDivisionError
        If ``sol.width`` or ``sol.params.t_r`` is zero.

    Notes
    -----
    The manufactured field is not exactly periodic (a Gaussian has infinite
    support) and not exactly band-limited. Both defects are exponentially
    small, but "exponentially small" is not an argument -- it is a number, and
    a convergence study is only meaningful while both defects sit FAR below the
    discretization error being measured. At the defaults they are of order
    1e-28 and 1e-69, against measured errors of order 1e-3, so the study has
    roughly twenty-five decades of headroom before either becomes the limiting
    term.

    Examples
    --------
    >>> defect = periodicity_defect(manufactured_solution())
    >>> sorted(defect)
    ['envelope_at_edge', 'samples_per_width', 'spectral_tail']
    >>> bool(defect["envelope_at_edge"] < 1e-20)
    True
    >>> bool(defect["spectral_tail"] < 1e-20)
    True
    >>> print(f"{defect['samples_per_width']:.1f}")
    8.0
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
    """Derive the forcing that makes the manufactured field an exact solution.

    Parameters
    ----------
    sol : MMSSolution
        The manufactured solution.
    n_tau : int, optional
        Fast-time grid size, default :data:`DEFAULT_N_TAU`.

    Returns
    -------
    psi : numpy.ndarray
        Shape ``(n_tau,)``, dtype ``complex128``, units sqrt(J)/s: the
        tau-dependent factor of the source.
    drive : float
        ``F = sqrt(kappa_c*pin)`` [sqrt(J)/s], the constant pump term, returned
        separately because it carries no ``exp(1j*omega_m*t)`` factor.

    Raises
    ------
    ImportError
        If sympy is not installed. The import is lazy so that a base install
        can import this module.
    AssertionError
        If the residual does not factor as ``Psi(tau)*exp(1j*omega_m*t)``, or
        if the evaluated source contains non-finite values.

    Notes
    -----
    The whole point of the method of manufactured solutions: rather than
    comparing the solver against itself at finer resolution, choose the
    solution FIRST and derive the forcing that makes it exact. The residual is
    assembled term by term from

        S = dE/dt + (kappa/2)*E + 1j*D_applied + 1j*delta_omega*E
            - 1j*gamma*|E|**2*E - F,
        D_applied = sum_k (beta_k/k!) * (-1j*d/dtau)**k E

    so ``S(tau, t) = Psi(tau)*exp(1j*omega_m*t) - F``. The
    ``exp(1j*omega_m*t)`` factorization is CHECKED symbolically rather than
    assumed: if the chosen solution were not separable, the two-operation
    runtime form built by :func:`make_source_fn` would silently drop the
    residual's t dependence and the study would converge to the wrong PDE
    while looking healthy.

    The Gaussian envelope is factored out BY CONSTRUCTION rather than left for
    ``simplify`` to find. Expanding the whole residual first produces terms like
    ``exp(+a*tau**2)*exp(-b*tau**2)`` with ``a < b`` -- mathematically decaying,
    but ``inf * 0 = nan`` when ``lambdify`` evaluates the two factors separately
    in float64.

    Examples
    --------
    >>> import pytest; _ = pytest.importorskip("sympy")
    >>> import numpy as np
    >>> sol = manufactured_solution()
    >>> psi, drive = manufactured_source(sol, n_tau=32)
    >>> print(psi.shape, psi.dtype)
    (32,) complex128
    >>> bool(np.all(np.isfinite(psi)))
    True
    >>> p = sol.params
    >>> bool(abs(drive - (p.kappa_c * sol.pin) ** 0.5) < 1e-9 * drive)
    True
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
    """Re-derive the source spectrally and report the relative disagreement.

    Parameters
    ----------
    sol : MMSSolution
        The manufactured solution.
    n_tau : int, optional
        Fast-time grid size, default :data:`DEFAULT_N_TAU`.
    t_eval : float, optional
        Slow time [s] at which to compare, default 0.0. The comparison holds at
        any time; 0.0 simply avoids a needless phase.

    Returns
    -------
    float
        ``||S_sym - S_spec|| / ||S_spec||`` [dimensionless]. Of order 1e-7 at
        the defaults -- the residual of the symbolic-versus-spectral
        derivative, not a physical error.

    Raises
    ------
    ImportError
        If sympy is not installed (via :func:`manufactured_source`).
    AssertionError
        Propagated from :func:`manufactured_source`.

    Notes
    -----
    An INDEPENDENT check of the source, not a restatement of it. This applies
    ``disp(omega) = sum_k beta_k/k! * omega**k`` in Fourier space exactly as the
    solver's fine step does, instead of differentiating symbolically in tau. If
    the sign convention assumed in :func:`manufactured_source` were wrong --
    ``omega -> -1j*d/dtau`` under numpy's FFT sign convention -- the two would
    disagree at O(1) rather than at 1e-7.

    This is the check that makes the whole MMS study trustworthy: a
    manufactured source derived with a sign error would still produce a clean
    convergence ladder, converging at the right ORDER to the wrong EQUATION.

    Examples
    --------
    >>> import pytest; _ = pytest.importorskip("sympy")
    >>> bool(residual_check(manufactured_solution(), n_tau=32) < 1e-5)
    True
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
    """Build the jax-traceable ``source_fn`` for :func:`~simulator.lle_solver.solve_lle_ssfm_jax`.

    Parameters
    ----------
    sol : MMSSolution
        The manufactured solution.
    n_tau : int, optional
        Fast-time grid size, default :data:`DEFAULT_N_TAU`.

    Returns
    -------
    callable
        ``source_fn(t)`` taking the scalar sub-step time [s] and returning an
        ``(n_tau,)`` complex array [sqrt(J)/s], suitable as the solver's
        ``source_fn`` argument.

    Raises
    ------
    ImportError
        If sympy (for the derivation) or JAX (for the closure) is unavailable.
    AssertionError
        Propagated from :func:`manufactured_source`.

    Notes
    -----
    The returned closure is CACHED on the raw bytes of ``Psi`` together with
    ``n_tau``, ``omega_m`` and ``F``, so repeated calls for the same
    manufactured solution return the SAME callable object. That identity
    matters: the solver caches its jit on ``source_fn`` identity, so a fresh
    closure per call would trigger a recompilation at every dt of the
    refinement ladder and turn a minute-long study into an hour-long one.

    Examples
    --------
    >>> import pytest; _ = pytest.importorskip("sympy")
    >>> sol = manufactured_solution()
    >>> fn = make_source_fn(sol, n_tau=32)
    >>> make_source_fn(sol, n_tau=32) is fn        # cached: jit-stable identity
    True
    >>> print(fn(0.0).shape)
    (32,)
    """
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
    """Measure the solver's relative L2 error against the manufactured solution.

    Parameters
    ----------
    dt : float
        Field sub-step [s]. Must divide ``t_r`` an integer number of times.
    n_tau : int, optional
        Fast-time grid size, default :data:`DEFAULT_N_TAU`.
    sol : MMSSolution or None, optional
        The manufactured solution; ``None`` (default) builds the standard one.
        Keyword-only.
    t_final_round_trips : int, optional
        Run length in round trips [dimensionless count], default
        :data:`DEFAULT_T_FINAL_ROUND_TRIPS`. Keyword-only.
    symmetric_drive : bool, optional
        Solver flag: split the drive kick to make the sub-step palindromic.
        Default ``False`` (the shipping first-order scheme). Keyword-only.
    thermal_coupling : {'lagged', 'strang'}, optional
        Solver flag, default ``'lagged'`` (the shipping scheme). Keyword-only.
    thermal_integrator : {'euler', 'exponential'}, optional
        Solver flag, default ``'euler'`` (the shipping scheme). Keyword-only.

    Returns
    -------
    float
        ``||E_num - E_mms|| / ||E_mms||`` [dimensionless] at the final time.

    Raises
    ------
    ValueError
        If ``dt`` does not divide ``t_r`` an integer number of times, naming
        the non-integer ratio.
    ImportError
        If sympy or JAX is unavailable.

    Notes
    -----
    The refinement knob is ``fine_cadence_M = t_r/dt`` with ``n_substeps = 1``.
    That single knob refines the field step, the thermal step, the detuning and
    the energy balance TOGETHER, so one ``dt`` controls the whole scheme --
    refining only the field step would measure the order of the field solver
    while the lagged thermal coupling silently capped the coupled order at 1.

    The run uses the derived thermo-optic-free config
    (``CavityParams.config_path``, with ``dn_dT_per_k = 0``) so the
    manufactured solution is not perturbed by thermal feedback, and
    ``NoiseConfig.all_off()`` so the error is deterministic.

    No ``Examples`` section: each call runs a 200-round-trip solve, which is
    exactly the "long solve" a doctest should not contain. The refinement
    ladder built on this function lives in :mod:`validation.convergence` and is
    exercised by ``tests/test_mms_convergence.py``.
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
