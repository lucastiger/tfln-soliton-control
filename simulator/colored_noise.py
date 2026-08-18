"""General PSD-specified colored-noise engine and target PSD models.

This module generalizes the repository's AR(1)-only noise machinery: any
stationary Gaussian noise channel can now be synthesized from an arbitrary
target one-sided power spectral density S(f), supplied as

* a Python callable ``S(f) -> array`` (one-sided, units X**2/Hz for a
  process of units X, e.g. (rad/s)**2/Hz for a detuning noise),
* a named analytic model built by the factories below
  (:func:`single_pole_psd`, :func:`kondratiev_gorodetsky_psd`), or
* a two-column CSV (f [Hz], S) loaded with :func:`csv_psd`
  (log-log interpolated, clamped flat outside the tabulated span).

Everything here is HOST-SIDE numpy in float64: FFT synthesis of long
sequences is cheap on the host and must not depend on the JAX x64 flag.
Determinism is anchored to JAX PRNG keys via :func:`np_generator_from_key`
(the full key data is folded into a ``numpy.random.SeedSequence``), so
distinct JAX keys give independent, fully reproducible numpy streams.

No matplotlib import (this module is loaded by the solver hot path).

Synthesis recipe (exact; see :func:`synthesize_from_psd`)
---------------------------------------------------------
For a length-``N`` real sequence at sample rate ``f_s``: on the rfft bins
``k = 0..N/2`` with ``f_k = k*f_s/N``, draw ``zeta_k`` as a standard complex
normal (E|zeta_k|**2 = 1) for ``0 < k < N/2`` and a standard real normal at
``k = 0`` and (even ``N``) ``k = N/2``; set

    c_k = zeta_k * sqrt(S(f_k) * f_s * N / 2)

and ``x = irfft(c, n=N)``. Then ``Var(x) = sum_k S(f_k)*Delta_f ~
integral_0^{f_s/2} S(f) df`` (the one-sided convention) and the Welch PSD of
``x`` reproduces ``S(f_k)`` bin by bin. The DC bin is clamped,
``S(0) := S(f_1)``, so 1/f-type spectra cannot inject a divergent DC power;
DC and Nyquist each carry ~1/N of the variance.

Aliasing note: the channels built from this engine are sampled once per
round trip at ``f_s = 1/t_r ~ 24.6 GHz``, far above any thermal band
(the thermorefractive spectrum has decayed by many orders of magnitude at
f_s/2), so synthesis-band truncation/aliasing of the physical spectra is
negligible.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

_K_B = 1.380649e-23  # Boltzmann constant [J/K]


# ---------------------------------------------------------------------------
# Determinism: JAX key -> numpy Generator
# ---------------------------------------------------------------------------
def np_generator_from_key(key: Any) -> np.random.Generator:
    """Derive a deterministic host-side numpy ``Generator`` from a JAX PRNG key.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key (typed key, or the legacy ``uint32`` key array returned by
        ``jax.random.PRNGKey``). Dimensionless.

    Returns
    -------
    numpy.random.Generator
        A ``PCG64``-backed generator seeded from the FULL key data, so distinct
        JAX keys give independent, reproducible numpy streams regardless of the
        JAX backend or of the ``jax_enable_x64`` flag.

    Raises
    ------
    ImportError
        If JAX cannot be imported. The import is deliberately function-local:
        this module must stay importable (and must not create a JAX module
        object) on the solver's hot path.
    TypeError
        Propagated from ``jax.random.key_data`` if ``key`` is not a PRNG key.

    Notes
    -----
    The full key data is folded into a ``numpy.random.SeedSequence`` as a
    little-endian byte concatenation of the ``uint32`` key words. This is the
    SINGLE seeding convention used by every host-side noise synthesis in the
    repository -- pump noise adopted it first, the colored TRN and
    segment-continuity paths reuse it -- and it is pinned by a determinism test
    in ``tests/test_colored_noise.py``.

    No physical units are involved: this function is pure bookkeeping between
    two PRNG conventions.

    Examples
    --------
    The same key always yields the same stream, and different keys do not:

    >>> import jax
    >>> a = np_generator_from_key(jax.random.PRNGKey(0)).standard_normal(3)
    >>> b = np_generator_from_key(jax.random.PRNGKey(0)).standard_normal(3)
    >>> c = np_generator_from_key(jax.random.PRNGKey(1)).standard_normal(3)
    >>> bool((a == b).all()), bool((a == c).all())
    (True, False)
    """
    import jax  # local import: keep this module importable without tracing

    data = np.asarray(jax.random.key_data(key), dtype=np.uint32).ravel()
    entropy = int.from_bytes(data.tobytes(), "little")
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def synthesize_from_psd(rng: np.random.Generator, n: int,
                        psd: Callable[[np.ndarray], np.ndarray], f_s: float,
                        clamp_dc: bool = True) -> np.ndarray:
    """Synthesize a real sequence whose one-sided PSD matches ``psd``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Host-side generator, normally built by :func:`np_generator_from_key`.
        The draw order -- all real parts first, then all imaginary parts -- is
        part of the determinism contract and must not be reordered.
    n : int
        Sequence length [samples]. ``n < 2`` short-circuits to zeros.
    psd : callable
        One-sided target PSD ``S(f)`` [X**2/Hz] for a process carrying units
        X, evaluated on the non-negative rfft frequency grid [Hz]. Negative
        returns are clipped to 0.
    f_s : float
        Sample rate [Hz].
    clamp_dc : bool, optional
        If ``True`` (default) clamp ``S(0) := S(f_1)`` so 1/f-type spectra stay
        finite at DC. ``False`` preserves whatever the callable returns at
        ``f = 0`` -- the legacy pump-noise behaviour, where the callables clamp
        their own DC bin.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``, dtype ``float64``, units X (the square root of the
        units of ``psd``), with
        ``Var(x) ~ integral_0^{f_s/2} S(f) df``.

    Raises
    ------
    ValueError
        Propagated from ``numpy.broadcast_to`` if ``psd(f)`` returns an array
        that is neither scalar nor of the rfft grid shape ``(n//2 + 1,)``.

    Notes
    -----
    Implements the module-docstring recipe exactly. On the rfft bins
    ``f_k = k*f_s/n``, draw ``zeta_k`` as a standard complex normal
    (``E|zeta_k|**2 = 1``) for ``0 < k < n/2`` and as a standard real normal at
    ``k = 0`` and (for even ``n``) ``k = n/2``; set
    ``c_k = zeta_k*sqrt(S(f_k)*f_s*n/2)`` and ``x = irfft(c, n=n)``.

    The result is stationary from sample 0 -- there is no AR(1)-style start-up
    transient -- which is what the segment-continuity path
    (``legacy_segment_noise = 0``) relies on when it slices one long
    realization into per-segment pieces.

    Examples
    --------
    A Lorentzian target whose total variance is 4 (units X**2):

    >>> import numpy as np
    >>> x = synthesize_from_psd(np.random.default_rng(0), 65536,
    ...                         single_pole_psd(4.0, 1e-4), f_s=1e5)
    >>> print(x.shape, x.dtype)
    (65536,) float64
    >>> bool(0.8 < x.var() / 4.0 < 1.2)
    True

    Degenerate lengths return zeros without touching ``rng``:

    >>> synthesize_from_psd(np.random.default_rng(0), 1,
    ...                     single_pole_psd(1.0, 1e-3), f_s=1e5)
    array([0.])
    """
    n = int(n)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    f = np.fft.rfftfreq(n, d=1.0 / float(f_s))               # (n//2 + 1,)
    s = np.asarray(psd(f), dtype=np.float64)
    if s.shape != f.shape:
        s = np.broadcast_to(s, f.shape).astype(np.float64)
    if clamp_dc:
        s = s.copy()
        s[0] = s[1]                                          # S(0) := S(f_1)
    amp = np.sqrt(np.maximum(s, 0.0) * float(f_s) * n / 2.0)
    zr = rng.standard_normal(f.size)
    zi = rng.standard_normal(f.size)
    z = (zr + 1j * zi) / math.sqrt(2.0)                      # E|z|^2 = 1
    z[0] = zr[0]                                             # DC bin real
    if n % 2 == 0:
        z[-1] = zi[-1]                                       # Nyquist bin real
    return np.fft.irfft(z * amp, n=n).astype(np.float64)


def integrate_psd(psd: Callable[[np.ndarray], np.ndarray], f_lo: float,
                  f_hi: float, n_grid: int = 20_000) -> float:
    """Integrate a one-sided PSD over a band on a log-spaced trapezoid grid.

    Parameters
    ----------
    psd : callable
        One-sided PSD ``S(f)`` [X**2/Hz] as a function of frequency [Hz].
    f_lo : float
        Lower integration limit [Hz]. Must be strictly positive (the grid is
        logarithmic).
    f_hi : float
        Upper integration limit [Hz]. Must exceed ``f_lo``.
    n_grid : int, optional
        Number of log-spaced quadrature points (default 20000). Dimensionless.

    Returns
    -------
    float
        ``integral_{f_lo}^{f_hi} S(f) df`` [X**2].

    Raises
    ------
    ValueError
        If ``f_lo <= 0`` or ``f_hi <= f_lo``.

    Notes
    -----
    Two callers rely on this: it renormalizes the Kondratiev--Gorodetsky
    asymptotic PSD (arXiv:2604.05897v1 Eq. 130) to the thermodynamic variance
    of Eq. 129, and it supplies the variance target of the engine unit tests.

    Choose ``f_lo`` low enough that the omitted band ``[0, f_lo]`` is negligible
    for the model at hand. The Kondratiev--Gorodetsky spectrum behaves as
    ``f**-1/2`` at low frequency, so its ``[0, f_lo]`` contribution scales as
    ``sqrt(f_lo)`` and vanishes as ``f_lo -> 0``.

    Examples
    --------
    A Lorentzian of unit-consistent total variance 2 integrates back to 2 once
    the band covers its corner at ``1/(2*pi*tau)``:

    >>> total = integrate_psd(single_pole_psd(2.0, 1e-6), f_lo=1.0, f_hi=1e9)
    >>> print(f"{total:.4f}")
    1.9998

    >>> integrate_psd(single_pole_psd(2.0, 1e-6), f_lo=0.0, f_hi=1e9)
    Traceback (most recent call last):
        ...
    ValueError: need 0 < f_lo < f_hi, got (0.0, 1000000000.0).
    """
    if not (f_lo > 0.0 and f_hi > f_lo):
        raise ValueError(f"need 0 < f_lo < f_hi, got ({f_lo!r}, {f_hi!r}).")
    f = np.logspace(math.log10(f_lo), math.log10(f_hi), int(n_grid))
    s = np.asarray(psd(f), dtype=np.float64)
    return float(np.trapezoid(s, f))


# ---------------------------------------------------------------------------
# Named PSD models
# ---------------------------------------------------------------------------
def single_pole_psd(variance: float,
                    tau: float) -> Callable[[np.ndarray], np.ndarray]:
    """Build a one-sided single-pole (Lorentzian) PSD of given total variance.

    Parameters
    ----------
    variance : float
        Total variance [X**2] of the process, i.e. ``integral_0^inf S df``.
        Units follow the process: K**2 for a temperature fluctuation,
        (rad/s)**2 for a detuning noise.
    tau : float
        Correlation time [s]. The half-power corner sits at ``1/(2*pi*tau)``
        [Hz].

    Returns
    -------
    callable
        ``S(f)`` mapping frequency [Hz] (scalar or array) to the one-sided PSD
        [X**2/Hz], of the same shape as its argument.

    Raises
    ------
    TypeError
        If ``variance`` or ``tau`` cannot be coerced to ``float`` -- both are
        converted eagerly, so a bad argument fails at build time rather than on
        first evaluation.

    Notes
    -----
    The spectrum is

    .. math:: S(f) = \\frac{4\\,\\sigma^2 \\tau}{1 + (2\\pi f \\tau)^2}

    which integrates to ``variance`` over ``[0, inf)`` in the one-sided
    convention. It is the spectral twin of the repository's AR(1) generator: an
    AR(1) process with correlation time ``tau``, sampled at ``dt << tau``, has
    exactly this spectrum. That equivalence is what lets
    :meth:`simulator.noise_models.TotalNoise.sample_full_with_delta_t`
    reproduce the ``single_pole`` channel through the PSD engine.

    Examples
    --------
    >>> psd = single_pole_psd(variance=2.0, tau=1e-6)
    >>> float(psd(0.0))                       # S(0) = 4*variance*tau
    8e-06
    >>> float(psd(1.0 / (2 * 3.141592653589793 * 1e-6)))   # half power
    4e-06
    """
    variance = float(variance)
    tau = float(tau)

    def _psd(f):
        f = np.asarray(f, dtype=np.float64)
        return (4.0 * variance * tau) / (1.0 + (2.0 * np.pi * f * tau) ** 2)

    return _psd


def kondratiev_gorodetsky_psd(
    T_k: float,
    kappa_th: float,
    rho: float,
    cp: float,
    R: float,
    d_a: float,
    d_b: float,
    mode_volume: float,
    f_max: float,
    f_lo: float = 1.0,
) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Build the analytic WGM thermorefractive temperature PSD (Eq. 130).

    Parameters
    ----------
    T_k : float
        Ambient temperature [K]. ``T_k = 0`` is the repository's deterministic
        "noise-off" convention and returns an identically zero spectrum.
    kappa_th : float
        Thermal conductivity [W/(m K)].
    rho : float
        Density [kg/m**3].
    cp : float
        Specific heat capacity [J/(kg K)].
    R : float
        Ring / whispering-gallery radius [m].
    d_a : float
        Gaussian mode half-dimension along the MAJOR axis [m].
    d_b : float
        Gaussian mode half-dimension along the MINOR axis [m].
    mode_volume : float
        Optical mode volume V [m**3], used only for the Eq. 129 variance pin.
    f_max : float
        Upper limit [Hz] of the band the variance is pinned over. Should be the
        synthesis Nyquist ``f_s/2``.
    f_lo : float, optional
        Lower limit [Hz] of the renormalization integral (default 1.0). The
        shape goes as ``f**-1/2`` at low frequency, so the omitted ``[0, f_lo]``
        band carries ``O(sqrt(f_lo))`` of the norm -- negligible at 1 Hz against
        an ``f_max`` of order 1e10 Hz.

    Returns
    -------
    psd : callable
        ``S_dT(f)`` mapping frequency [Hz] to the one-sided temperature PSD
        [K**2/Hz], same shape as its argument.
    var_eq129 : float
        The thermodynamic variance ``k_B*T**2/(rho*C*V)`` [K**2] that the
        returned spectrum integrates to over ``[f_lo, f_max]``.

    Raises
    ------
    ValueError
        If any of ``R``, ``d_a``, ``d_b`` is non-positive, or if
        ``d_a < 1.2*d_b`` -- the Eq. 130 prefactor
        ``[R*sqrt(d_a**2 - d_b**2)]**-1`` degenerates for a near-circular mode,
        and the paper prescribes a rescaling for that limit rather than this
        formula.

    Notes
    -----
    One-sided ``S_dT(omega)`` of the mode-averaged temperature fluctuation of a
    whispering-gallery / ring mode, arXiv:2604.05897v1 Eq. 130::

        S_dT(omega) = [k_B*T**2 / sqrt(pi**3 * kappa_th * rho * C * omega)]
                      * [R * sqrt(d_a**2 - d_b**2)]**-1
                      * [1 + (omega*tau_d)**(3/4)]**-2,
        tau_d = (pi/4)**(1/3) * (rho*C/kappa_th) * d_b**2,

    with ``k_B`` the Boltzmann constant [J/K] and ``omega = 2*pi*f`` [rad/s].

    The analytic form is an ASYMPTOTIC matching of the low- and
    high-frequency limits, so its absolute integral does not exactly reproduce
    the thermodynamic variance. The returned PSD is therefore rescaled by a
    single constant such that

        integral_{f_lo}^{f_max} S_dT(f) df == k_B*T**2/(rho*C*V)   (Eq. 129)

    with ``V = mode_volume``: the total variance is pinned to thermodynamics and
    the analytic curve supplies only the SHAPE. This renormalization is
    deliberate documented behaviour, not a fudge factor -- it is what makes the
    ``kondratiev_gorodetsky`` and ``single_pole`` TRN models carry the same
    total power and differ only in spectral shape.

    Examples
    --------
    >>> psd, var = kondratiev_gorodetsky_psd(
    ...     T_k=300.0, kappa_th=4.6, rho=4.64e3, cp=700.0,
    ...     R=1e-4, d_a=2e-6, d_b=1e-6, mode_volume=1e-15, f_max=1e10)
    >>> print(f"{var:.4e}")            # k_B*T^2/(rho*C*V), Eq. 129 [K^2]
    3.8257e-10
    >>> bool(psd(1e3) > psd(1e6))      # red-tilted: falls with frequency
    True

    ``T_k = 0`` is the noise-off convention:

    >>> psd0, var0 = kondratiev_gorodetsky_psd(
    ...     T_k=0.0, kappa_th=4.6, rho=4.64e3, cp=700.0,
    ...     R=1e-4, d_a=2e-6, d_b=1e-6, mode_volume=1e-15, f_max=1e10)
    >>> var0, float(psd0(1e3))
    (0.0, 0.0)

    A near-circular mode is rejected rather than silently mis-normalized:

    >>> kondratiev_gorodetsky_psd(
    ...     T_k=300.0, kappa_th=4.6, rho=4.64e3, cp=700.0,
    ...     R=1e-4, d_a=1.0e-6, d_b=1.0e-6, mode_volume=1e-15, f_max=1e10)
    Traceback (most recent call last):
        ...
    ValueError: Kondratiev-Gorodetsky PSD requires d_a >= 1.2*d_b ...
    """
    T_k = float(T_k)
    if not (d_a > 0.0 and d_b > 0.0 and R > 0.0):
        raise ValueError(
            f"Kondratiev-Gorodetsky geometry must be positive: "
            f"R={R!r} m, d_a={d_a!r} m, d_b={d_b!r} m."
        )
    if not (d_a >= 1.2 * d_b):
        raise ValueError(
            f"Kondratiev-Gorodetsky PSD requires d_a >= 1.2*d_b "
            f"(got d_a={d_a:.3e} m, d_b={d_b:.3e} m): the Eq. 130 prefactor "
            f"[R*sqrt(d_a^2-d_b^2)]^-1 degenerates for d_a ~ d_b. See the "
            f"paper's rescaling remark for near-circular modes "
            f"(arXiv:2604.05897, discussion around Eq. 130)."
        )
    var_eq129 = _K_B * T_k**2 / (float(rho) * float(cp) * float(mode_volume))
    if var_eq129 == 0.0:                    # T_k = 0: noise-off convention
        return (lambda f: np.zeros_like(np.asarray(f, dtype=np.float64)),
                0.0)

    tau_d = (math.pi / 4.0) ** (1.0 / 3.0) * (
        float(rho) * float(cp) / float(kappa_th)
    ) * float(d_b) ** 2
    pref = (_K_B * T_k**2) / (
        float(R) * math.sqrt(float(d_a) ** 2 - float(d_b) ** 2)
    )

    def _shape(f):
        f = np.asarray(f, dtype=np.float64)
        omega = 2.0 * np.pi * np.maximum(f, 1e-30)
        s = (
            pref
            / np.sqrt(math.pi**3 * float(kappa_th) * float(rho) * float(cp)
                      * omega)
            / (1.0 + (omega * tau_d) ** 0.75) ** 2
        )
        return s

    # Pin integral_0^{f_max} S df to the Eq. 129 thermodynamic variance. The
    # shape ~ f**-1/2 below 1/tau_d, so the omitted [0, f_lo] band carries
    # O(sqrt(f_lo)) of the norm — negligible at the default f_lo = 1 Hz
    # against f_max ~ 1e10 Hz.
    norm = integrate_psd(_shape, f_lo=float(f_lo), f_hi=float(f_max))
    scale = var_eq129 / norm

    def _psd(f):
        return scale * _shape(f)

    return _psd, var_eq129


def csv_psd(csv_path: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """Build a one-sided PSD from a two-column ``f [Hz], S`` CSV.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to a comma-separated two-column file: frequency [Hz] and
        one-sided spectral density S. Typically a measured or FEM-computed
        thermorefractive spectrum.

    Returns
    -------
    callable
        ``S(f)`` mapping frequency [Hz] to the tabulated one-sided PSD, same
        shape as its argument. Units are whatever the file tabulates -- the
        caller selects the interpretation (``S_dT`` [K**2/Hz] versus
        ``S_domega`` [(rad/s)**2/Hz]) through the ``trn_csv_units`` config key.

    Raises
    ------
    ValueError
        If the file has fewer than two columns, or if fewer than two rows
        survive the ``f > 0 and S > 0`` filter that log-log interpolation
        requires.
    OSError
        Propagated from ``numpy.loadtxt`` if the file cannot be read.

    Notes
    -----
    Follows the Huang et al. (2019) style of tabulated thermorefractive
    spectra: values are interpolated LINEARLY IN LOG-LOG space, so tabulated
    power laws become straight lines, and clamped FLAT outside the tabulated
    span (``S(f < f_min) = S(f_min)``, ``S(f > f_max) = S(f_max)``). ``f = 0``
    is guarded by the low-side clamp, and :func:`synthesize_from_psd`
    additionally clamps its own DC bin. Rows with non-positive ``f`` or ``S``
    are dropped, since neither has a logarithm.

    Examples
    --------
    A two-point table is a single power law between the tabulated points and
    flat outside them:

    >>> import pathlib, tempfile
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "psd.csv"
    >>> _ = path.write_text("1.0,1e-06" + chr(10) + "1000.0,1e-12" + chr(10))
    >>> psd = csv_psd(path)
    >>> [f"{float(psd(f)):.3e}" for f in (0.1, 1.0, 10.0, 1000.0, 1e5)]
    ['1.000e-06', '1.000e-06', '1.000e-08', '1.000e-12', '1.000e-12']
    """
    path = Path(csv_path)
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"{path}: expected two columns (f_hz, S), got shape "
                         f"{data.shape}.")
    f_tab, s_tab = data[:, 0], data[:, 1]
    keep = (f_tab > 0.0) & (s_tab > 0.0) & np.isfinite(f_tab) & np.isfinite(s_tab)
    f_tab, s_tab = f_tab[keep], s_tab[keep]
    if f_tab.size < 2:
        raise ValueError(
            f"{path}: need >= 2 rows with f > 0 and S > 0 for log-log "
            f"interpolation, got {f_tab.size}."
        )
    order = np.argsort(f_tab)
    log_f, log_s = np.log10(f_tab[order]), np.log10(s_tab[order])

    def _psd(f):
        f = np.asarray(f, dtype=np.float64)
        # Guard f <= 0 before the log; those bins land on the flat low clamp.
        lf = np.log10(np.maximum(f, 10.0 ** log_f[0]))
        return 10.0 ** np.interp(lf, log_f, log_s)   # np.interp clamps flat

    return _psd
