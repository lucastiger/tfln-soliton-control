"""Spectral metrics for the validated single-DKS comb.

This module operates on the optical power spectrum ``|a_mu|**2`` (LINEAR power,
one value per resonator mode ``mu``) of the dissipative-Kerr-soliton (DKS) state
produced by ``analysis.dks_access``.  At the PRODUCTION operating point
(delta_omega = 10 kappa, past the ~9.3-9.4 kappa Hopf boundary) the attractor is
a STATIONARY single soliton, so the committed spectrum artifacts are single
snapshots -- a reproducible observable in their own right, not cycle-averages.

At the historically A/B-validated point delta_omega = 8 kappa the attractor is
instead a deterministic BREATHER (period T_b ~ 152-153 RT, dU/U ~4.1%), so there
a single-snapshot spectrum is breathing-phase dependent and NOT a reproducible
observable; a deliberate 8-kappa run must instead supply the breathing-cycle-
averaged spectrum from ``analysis.dks_access.cycle_averaged_spectrum`` (that
function, and ``breathing_metrics`` / ``breathing_scan``, remain valid for
characterising the breather sub-band).  All public metrics take such a
precomputed spectrum (single snapshot or cycle-average) as their primary input;
the sole convenience wrapper that starts from a solver trajectory
(:func:`average_power_spectrum`) performs its averaging in linear power, never
in dB and never on complex fields.

Feature 1 -- 3 dB spectral span
-------------------------------
The 3 dB spectral span is the full width (at the half-power / -3 dB level) of
the soliton comb envelope, reported both in relative mode number
``Delta_mu`` and in Hz (``Delta_mu * FSR``).  The envelope is extracted from
the discrete comb-line powers with the pump line excluded (it sits on a strong
CW background well above the sech^2 soliton envelope), a numerical-floor mask
applied, and light dB-domain median smoothing followed by monotone-cubic
(PCHIP) interpolation.  See :func:`spectral_envelope_db` and
:func:`three_db_span` for the full method and its justification.

Feature 2 -- conversion efficiency
----------------------------------
Two clearly-separated efficiency metrics answer two different questions:
:func:`intracavity_comb_fraction` (``eta_intra``, always computable, scale-
invariant) is the fraction of intracavity power carried by non-pump comb lines,
and :func:`pump_to_comb_efficiency` (``eta``, the physically standard
bus-waveguide metric) is ``P_comb,out / P_in``, computable only when the
out-coupling rate ``kappa_c`` and pump power ``P_in`` are in the run config AND
the input spectrum carries an absolute power scale.  Both sum the TOTAL non-pump
power (the only excluded line is the pump itself -- no envelope/threshold/dB
cutoff).  See those functions for the exact intracavity->bus derivation in the
repo's LLE normalization.

Feature 3 -- soliton steps (power vs detuning)
----------------------------------------------
:func:`hold_window_average` reduces the per-round-trip power history of a
constant-detuning hold to one reproducible value by averaging the FINAL fraction
of the hold in LINEAR power (discarding the re-settling transient and cycle-
averaging the breather).  :func:`detect_power_steps` finds the soliton-step
discontinuities in the assembled power-vs-detuning trace via a robust MAD test on
its first differences, and :func:`plot_soliton_steps` renders the publication
staircase figure.  :func:`soliton_count_transitions` lists every edge where the
MEASURED soliton number changes, and :func:`match_steps_to_transitions` aligns
the detected power steps with those transitions -- a matched step is a
VALIDATED soliton step (a power discontinuity proven to coincide with a
soliton-number change), and only matched steps may be labelled as soliton steps
downstream.  The solver-driving sweep lives in the analysis-layer driver
``analysis/run_detuning_sweep.py``; these functions consume its output and never
import the solver.

Conventions
-----------
* Analysis-layer arrays are NumPy float64 (JAX arrays are converted with
  ``np.asarray``); nothing here imports the solver.
* ``mu`` is the relative cavity-mode number with ``mu = 0`` the pump line,
  matching the FFT-bin indexing used by ``analysis.dks_access`` (one bin per
  FSR).
* Powers are LINEAR (``|a_mu|**2``); the repo normalises the committed spectra
  so the pump line has unit power, but every metric here is invariant to that
  overall scale (it works from relative dB).
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# Speed of light [m/s], for deriving kappa_c from a coupling-Q when the explicit
# rate is absent (matches simulator.lle_solver.resolve_cavity_rates).
_C_LIGHT = 299_792_458.0

# -3 dB (half-power) level.  "3 dB span" and "FWHM" denote the same thing here:
# the full width where the envelope power falls to half its maximum, i.e. a
# 10*log10(2) = 3.0103 dB drop.  ``level_db`` defaults to exactly 3.0 dB (the
# literal "-3 dB" of the spec); the two differ by 0.04 %, far below the metric's
# few-percent robustness budget.
HALF_POWER_DB = 10.0 * math.log10(2.0)          # 3.0103 dB

DEFAULT_LEVEL_DB = 3.0
DEFAULT_FLOOR_DB = 60.0        # discard lines > this far below the strongest
                               # non-pump line before smoothing
DEFAULT_SMOOTH_MODES = 5       # light median window (single-line-spur removal)
DEFAULT_INTERP_FACTOR = 10     # PCHIP fine-grid density (>= 10x mode density)
AUTO_SMOOTH_W_MIN = 5          # smallest odd median window the auto-search tries
AUTO_SMOOTH_W_MAX = 101        # cap on the auto-search median window

# arccosh(sqrt(2)): sech^2(x) = 1/2 at x = arccosh(sqrt(2)) = 0.881373587...
_ARCCOSH_SQRT2 = math.acosh(math.sqrt(2.0))


# ---------------------------------------------------------------------------
# 1. Input normalisation
# ---------------------------------------------------------------------------
def comb_line_powers(spectrum, mode_index=None, *, pump_mu: int = 0):
    """Normalise a spectrum to per-comb-line powers ``(mu, P_mu)``.

    ``spectrum`` is either an already per-resonator-mode power array or a dense
    (fftshifted) FFT-grid power spectrum -- in this repo these are the same
    object, one FFT bin per cavity mode (one per FSR), so this is a validating
    pass-through.  ``mode_index`` is the matching relative mode number ``mu``
    (``mu = pump_mu`` is the pump line); when omitted the standard fftshifted
    FFT-grid indexing ``mu = arange(n) - n // 2`` is assumed.

    The pump index is taken from the mode indexing (``mu == pump_mu``), NEVER
    inferred as "the strongest line": for a DKS the pump sits on a bright CW
    background and IS the strongest line, but that coincidence must not define
    the pump -- other operating points (or a pump-suppressed comb) would break
    it.  Argmax is used only as a validated fallback, with a warning, when the
    supplied ``mode_index`` does not contain ``pump_mu``.

    Returns ``(mu, P_mu)`` sorted by ascending ``mu``, with ``mu`` int64 and
    ``P_mu`` float64.  Validates that powers are finite and non-negative.
    """
    P = np.asarray(spectrum, dtype=np.float64).ravel()
    n = P.size
    if mode_index is None:
        mu = np.arange(n, dtype=np.int64) - n // 2
    else:
        mu = np.asarray(mode_index).ravel().astype(np.int64)
        if mu.size != n:
            raise ValueError(
                f"mode_index length {mu.size} != spectrum length {n}"
            )
    if not np.all(np.isfinite(P)):
        raise ValueError("spectrum contains non-finite (NaN/Inf) powers")
    if np.any(P < 0.0):
        raise ValueError("spectrum contains negative powers (expected |a_mu|^2)")

    order = np.argsort(mu, kind="stable")
    mu, P = mu[order], P[order]

    pump_hits = np.nonzero(mu == pump_mu)[0]
    if pump_hits.size == 0:
        i = int(np.argmax(P))
        warnings.warn(
            f"pump mode mu={pump_mu} absent from mode_index; falling back to "
            f"argmax at mu={int(mu[i])} (validated fallback only)",
            RuntimeWarning,
            stacklevel=2,
        )
    return mu, P


def normal_ordered_spectrum(e_field, n_tau, hbar_omega0, *, clip=True):
    """Convert a symmetric-ordered (truncated-Wigner) spectrum to the normally-ordered
    spectrum an OSA or photodiode measures, by subtracting the half-photon-per-mode
    vacuum bias:  ``S_normal = |FFT(E)|^2 - n_tau^2 * hbar_omega0 / 2``.

    With ``clip=False`` the (possibly negative) signed residual is returned, which is what
    you want when propagating error bars; with ``clip=True`` it is floored at 0.

    REQUIRED FOR ANY COMPARISON TO MEASUREMENT
    ------------------------------------------
    The solver integrates the LLE in the truncated-Wigner c-number limit
    (``simulator/lle_solver.py::_qnoise_increment``, arXiv:2604.05897v1 Sec. V.B.2,
    Eq. 126). Wigner-representation moments are SYMMETRICALLY ordered, so a mode in the
    vacuum state carries ``<a a* + a* a>/2 = 1/2`` of a photon and the simulated spectrum
    sits on a flat, state-independent pedestal of exactly ``n_tau^2 * hbar_omega0 / 2``
    per mode (the ``n_tau^2`` follows from the repo's FFT convention,
    ``Etilde_mu = a_mu * n_tau * sqrt(hbar_omega0)``).

    A photodetector or an optical spectrum analyser measures ``<a^dagger a>`` -- NORMALLY
    ordered -- and therefore reads exactly zero for the vacuum. Comparing a raw simulated
    ``|FFT(E)|^2`` against a measured spectrum silently adds that half-photon pedestal to
    the simulation only. At the committed operating point the pedestal sits about
    -82 dB relative to the pump, so it dominates every comb feature weaker than that and
    makes the simulation look like it has a noise floor the instrument does not report.
    Subtract it with this function before any such comparison.

    THE SYMMETRIC-ORDERED FORM REMAINS CORRECT FOR INTERNAL ENERGY BOOKKEEPING
    -------------------------------------------------------------------------
    Do NOT use this function for energy accounting. ``U_int``, ``P_abs = kappa_i*<|E|^2>``,
    the thermo-optic drive, the intracavity-energy budget and the vacuum-floor
    diagnostics themselves are all statements about the symmetric-ordered field the
    solver actually integrates; subtracting the pedestal there would double-count the
    vacuum and break the energy balance. The rule of thumb: subtract when you are asking
    "what would the instrument read?", keep the pedestal when you are asking "how much
    energy is in the cavity?".

    STATISTICS
    ----------
    Subtraction removes the vacuum's BIAS, not its VARIANCE. A single-snapshot vacuum mode
    is exponentially distributed with mean and standard deviation both equal to the
    pedestal, so the residual of a pure-vacuum mode is a zero-mean random variable of
    order the pedestal itself -- individual residuals are routinely NEGATIVE. Averaging
    ``N`` independent snapshots reduces the residual's standard error to
    ``pedestal / sqrt(N)`` but never to zero, so a feature is only resolvable once it
    exceeds that standard error. Use ``clip=False`` whenever you intend to average,
    fit, or attach an error bar: clipping at zero turns a zero-mean residual into a
    positively biased one and destroys exactly the cancellation the averaging relies on.
    ``clip=True`` is for display only (a dB axis cannot render a negative power).

    Args:
        e_field: Either the complex fast-time field ``E(tau)`` (any shape, transformed
            along the LAST axis) or an already-formed ``|FFT(E)|^2`` power spectrum. A
            complex input is FFT'd (NOT fftshifted -- the caller owns the mode ordering);
            a real input is treated as an already-computed power spectrum and validated
            to be non-negative.
        n_tau: Number of fast-time grid points, i.e. the FFT length that set the
            normalization. Taken as an argument rather than inferred from the array
            shape, so a truncated or masked spectrum still subtracts the right pedestal.
        hbar_omega0: Photon energy ``hbar*omega0`` [J] (see
            :func:`simulator.lle_solver.hbar_omega0_from_config`).
        clip: ``True`` (default) floors the result at 0; ``False`` returns the signed
            residual.

    Returns:
        float64 array of the same shape as the input spectrum, in the same units as
        ``|FFT(E)|^2``.
    """
    n_tau = int(n_tau)
    if n_tau <= 0:
        raise ValueError(f"n_tau must be a positive integer, got {n_tau}")
    hbar_omega0 = float(hbar_omega0)
    if not (hbar_omega0 > 0.0) or not math.isfinite(hbar_omega0):
        raise ValueError(
            f"hbar_omega0 must be a positive, finite photon energy [J], "
            f"got {hbar_omega0!r}"
        )

    arr = np.asarray(e_field)
    if np.iscomplexobj(arr):
        power = np.abs(np.fft.fft(arr.astype(np.complex128), axis=-1)) ** 2
    else:
        power = np.asarray(arr, dtype=np.float64)
        if np.any(power < 0.0):
            raise ValueError(
                "real-valued input is treated as |FFT(E)|^2 and must be "
                "non-negative; pass the complex field to have the FFT taken here"
            )
    if not np.all(np.isfinite(power)):
        raise ValueError("spectrum contains non-finite (NaN/Inf) values")

    vacuum_bias = float(n_tau) ** 2 * hbar_omega0 / 2.0
    residual = power - vacuum_bias
    return np.maximum(residual, 0.0) if clip else residual


def average_power_spectrum(snapshots, *, is_power: bool = False,
                           mode_index=None):
    """Cycle/slow-time-average |a_mu|^2 over a stack of snapshots (LINEAR power).

    Convenience wrapper for going from a solver trajectory to an averaged
    spectrum (mirrors ``analysis.dks_access.cycle_averaged_spectrum``, which is
    the production path).  ``snapshots`` has the slow-time / round-trip axis
    first:

    * ``is_power=False`` (default): ``snapshots`` are complex intracavity fast-
      time fields, shape ``(n_avg, n_tau)``.  The per-snapshot mode power is
      ``|fftshift(fft(field))|**2`` and these are averaged.
    * ``is_power=True``: ``snapshots`` are already per-mode LINEAR power spectra,
      shape ``(n_avg, n_modes)`` (fftshifted), and are averaged directly.

    Averaging is done in LINEAR power -- never on dB values (which would bias
    the mean toward the peak) and never on the complex fields (breathing phase
    drift would cancel real power).  Returns ``(mu, P_mu)`` via
    :func:`comb_line_powers`.
    """
    arr = np.asarray(snapshots)
    if arr.ndim != 2:
        raise ValueError(f"snapshots must be 2D (n_avg, n); got shape {arr.shape}")
    if is_power:
        power = np.asarray(arr, dtype=np.float64)
        if np.iscomplexobj(arr):
            raise ValueError("is_power=True but snapshots are complex")
    else:
        fld = np.asarray(arr, dtype=np.complex128)
        power = np.abs(np.fft.fftshift(np.fft.fft(fld, axis=-1), axes=-1)) ** 2
    mean_power = np.mean(power, axis=0)
    return comb_line_powers(mean_power, mode_index)


# ---------------------------------------------------------------------------
# 2. Envelope extraction
# ---------------------------------------------------------------------------
def _build_envelope(mu, P_mu, *, exclude_pump=True,
                    smooth_modes=DEFAULT_SMOOTH_MODES, floor_db=DEFAULT_FLOOR_DB,
                    interp_factor=DEFAULT_INTERP_FACTOR, pump_mu=0):
    """Shared envelope construction; returns every intermediate as a dict.

    Steps (see :func:`spectral_envelope_db` for the physical justification):
    exclude pump -> numerical-floor mask -> dB relative to the strongest
    retained (non-pump) line -> odd-window median smoothing -> PCHIP onto a
    fine mu grid.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import medfilt

    mu = np.asarray(mu).astype(np.int64)
    P = np.asarray(P_mu, dtype=np.float64)
    if mu.shape != P.shape:
        raise ValueError("mu and P_mu must have the same shape")

    keep = np.isfinite(P) & (P > 0.0)
    if exclude_pump:
        keep &= mu != pump_mu
    m, p = mu[keep], P[keep]
    if m.size < 4:
        raise ValueError(
            f"too few usable comb lines ({m.size}) to build an envelope"
        )

    ref_lin = float(p.max())                      # strongest non-pump line
    floor_lin = ref_lin * 10.0 ** (-floor_db / 10.0)
    above = p >= floor_lin
    m, p = m[above], p[above]

    order = np.argsort(m, kind="stable")
    m, p = m[order], p[order]
    s_db_raw = 10.0 * np.log10(p / ref_lin)       # <= 0 dB, peak at 0

    w = int(smooth_modes)
    if w > 1:
        if w % 2 == 0:                            # medfilt requires odd windows
            w += 1
        s_db_sm = medfilt(s_db_raw, kernel_size=min(w, _odd_le(m.size)))
    else:
        s_db_sm = s_db_raw.copy()

    step = 1.0 / float(interp_factor)
    mu_env = np.arange(float(m.min()), float(m.max()) + 0.5 * step, step)
    s_db_env = PchipInterpolator(m.astype(np.float64), s_db_sm)(mu_env)

    return {
        "mu_ret": m,
        "P_ret": p,
        "s_db_raw": s_db_raw,
        "s_db_sm": s_db_sm,
        "mu_env": mu_env,
        "s_db_env": s_db_env,
        "ref_lin": ref_lin,
        "floor_lin": floor_lin,
        "smooth_modes": w,
        "floor_db": float(floor_db),
        "interp_factor": int(interp_factor),
    }


def _odd_le(n: int) -> int:
    """Largest odd integer <= n (>= 1)."""
    n = int(n)
    return n if n % 2 == 1 else max(n - 1, 1)


def spectral_envelope_db(mu, P_mu, *, exclude_pump=True,
                         smooth_modes=DEFAULT_SMOOTH_MODES,
                         floor_db=DEFAULT_FLOOR_DB,
                         interp_factor=DEFAULT_INTERP_FACTOR, pump_mu=0):
    """Comb-envelope in dB on a fine ``mu`` grid: ``(mu_env, S_db)``.

    Method and justification (implemented exactly as below):

    * **Comb-line peak powers only.**  The discrete ``P_mu`` ARE the comb
      lines; there is no inter-line noise floor to reject at the analysis
      layer, only a numerical floor (below).

    * **Exclude the pump line (mu = 0).**  For a DKS the pump line sits on a
      strong intracavity CW background component that lies well above the
      sech^2 soliton envelope, so including it biases both the envelope peak
      and the half-max level.

    * **Numerical-floor mask.**  Lines more than ``floor_db`` (default 60 dB)
      below the strongest non-pump line are discarded before smoothing, so the
      far wings sitting at the float64/dealias numerical floor cannot distort a
      smoother or fit.

    * **dB-domain smoothing.**  ``S_db = 10*log10(P_mu / max(P_mu, non-pump))``
      and all smoothing/interpolation happens in dB.  The sech^2 envelope is
      near-parabolic in dB over its core, so dB-domain smoothing preserves the
      -3 dB crossing shape with minimal bias; linear-domain smoothing would
      over-weight the peak and bias the width low.  (This is orthogonal to the
      upstream breathing-phase averaging, which is done in LINEAR power.)

    * **Smoothing = median + PCHIP.**  A light odd-window median filter
      (``smooth_modes``, default 5) suppresses single-line outliers
      (mode-crossing / dispersive-wave spurs), followed by monotone-cubic
      (PCHIP) interpolation onto a mu grid ``interp_factor`` times denser than
      the modes, for sub-mode crossing resolution.  A global sech^2 model is
      deliberately NOT fit here -- the lab-relevant spectra contain dispersive-
      wave features a global fit cannot represent (see :func:`sech2_core_fwhm`
      for the optional core cross-check).

    ``S_db`` is referenced to the strongest retained non-pump line (its peak is
    ~0 dB).  Returns float64 ``(mu_env, S_db)``.
    """
    env = _build_envelope(mu, P_mu, exclude_pump=exclude_pump,
                          smooth_modes=smooth_modes, floor_db=floor_db,
                          interp_factor=interp_factor, pump_mu=pump_mu)
    return env["mu_env"], env["s_db_env"]


# ---------------------------------------------------------------------------
# 3. 3 dB span
# ---------------------------------------------------------------------------
def _level_crossings(x, y, level):
    """Interpolated ``x`` values where ``y`` crosses ``level`` (linear interp)."""
    d = np.asarray(y, dtype=np.float64) - float(level)
    x = np.asarray(x, dtype=np.float64)
    out = []
    for i in range(d.size - 1):
        a, b = d[i], d[i + 1]
        if a == 0.0:
            out.append(x[i])
            continue
        if (a < 0.0) != (b < 0.0):
            t = a / (a - b)
            out.append(x[i] + t * (x[i + 1] - x[i]))
    return np.asarray(out, dtype=np.float64)


def _span_from_envelope(mu_env, s_db_env, level_db):
    """Outermost -level_db crossings of a prepared dB envelope.

    Returns ``(left, right, peak_mu, reference_db, n_crossings)`` with ``left``/
    ``right`` NaN where the retained band ends before a crossing.
    """
    reference_db = float(np.max(s_db_env))
    peak_mu = float(mu_env[int(np.argmax(s_db_env))])
    level = reference_db - level_db
    crossings = _level_crossings(mu_env, s_db_env, level)
    left_side = crossings[crossings <= peak_mu]
    right_side = crossings[crossings >= peak_mu]
    left = float(left_side.min()) if left_side.size else math.nan     # outermost
    right = float(right_side.max()) if right_side.size else math.nan  # outermost
    return left, right, peak_mu, reference_db, int(crossings.size)


def _auto_smooth_window(mu, P_mu, *, exclude_pump, floor_db, interp_factor,
                        level_db, pump_mu, w_min=AUTO_SMOOTH_W_MIN,
                        w_max=AUTO_SMOOTH_W_MAX):
    """Data-derived median window: smallest odd ``w >= w_min`` giving a clean,
    single-lobe -level_db crossing on each side.

    The deterministic cycle-averaged breather spectrum carries a strong quasi-
    periodic interference modulation (soliton/CW beating and breathing
    sidebands) whose depth exceeds ``level_db``; a light median cannot remove
    it, so the -3 dB crossings would be modulation-dominated and unstable.
    Rather than hardcode a window for one dataset, grow the odd median window
    until the smoothed envelope descends monotonically to exactly ONE crossing
    per side (``n_crossings == 2`` with both sides bracketed) and stays that way
    at the next window (hysteresis, so a fragile single-window dip is not
    picked).  For a clean sech^2 comb this returns ``w_min`` immediately.

    Returns ``(window, converged)``; ``converged`` is False if no window up to
    ``w_max`` produced a clean pair of crossings (then ``w_max`` is returned).
    """
    w = w_min if w_min % 2 == 1 else w_min + 1
    prev_clean = False
    prev_w = None
    while w <= w_max:
        mu_env, s_db_env = spectral_envelope_db(
            mu, P_mu, exclude_pump=exclude_pump, smooth_modes=w,
            floor_db=floor_db, interp_factor=interp_factor, pump_mu=pump_mu)
        left, right, _, _, ncross = _span_from_envelope(mu_env, s_db_env, level_db)
        clean = ncross == 2 and math.isfinite(left) and math.isfinite(right)
        if clean and prev_clean:
            return prev_w, True
        prev_clean, prev_w = clean, w
        w += 2
    # fell through: either never clean, or only became clean at the last window
    if prev_clean:
        return prev_w, True
    return _odd_le(w_max), False


def three_db_span(mu, P_mu, metadata, *, exclude_pump=True, smooth_modes="auto",
                  floor_db=DEFAULT_FLOOR_DB, interp_factor=DEFAULT_INTERP_FACTOR,
                  level_db=DEFAULT_LEVEL_DB, pump_mu=0,
                  auto_w_min=AUTO_SMOOTH_W_MIN, auto_w_max=AUTO_SMOOTH_W_MAX):
    """3 dB spectral span (FWHM) of the soliton comb envelope.

    Builds the envelope with :func:`spectral_envelope_db` and measures its full
    width at ``level_db`` (default -3 dB) below the envelope maximum.

    * **Reference level** is the MAXIMUM of the smoothed envelope over the
      retained (non-pump, above-floor) lines -- not the pump power and not any
      single raw line.  Localised dispersive-wave peaks are suppressed by the
      median step and, at the default ``floor_db``, fall below the floor mask
      entirely; if a DW nonetheless dominates the smoothed envelope (the -3 dB
      band fails to straddle the comb centre ``mu = 0``) a warning is emitted
      naming the peak location, because the "3 dB span" of a DW-dominated
      envelope is not the sech^2 bandwidth.

    * **Outermost crossings.**  On each side the span uses the OUTERMOST
      crossing of ``reference - level_db`` (the last down-crossing before the
      retained-data boundary), which makes the metric robust to small ripple
      near the peak.

    * **Smoothing window.**  ``smooth_modes='auto'`` (default) selects the
      window from the data via :func:`_auto_smooth_window` -- the smallest odd
      window whose envelope has a single clean crossing per side -- so no
      dataset-specific window is hardcoded; pass an int to force a fixed window
      (used by the synthetic unit tests).

    Edge cases return NaN (never raise): if the envelope never rises
    ``level_db`` above its wings, both crossings are NaN with a warning; if only
    one side is bracketed before the retained band ends, that side is NaN and
    ``one_sided`` is set.

    ``metadata`` supplies the mode spacing for the Hz conversion via key
    ``fsr_hz`` (pulled from config/run metadata, never hardcoded); ``pump_mu``
    may also be supplied there.  Returns a dict with the span in mode-number and
    Hz units, crossing positions, reference level, every parameter used, flags,
    warnings, units, and the metric definition string.
    """
    metadata = dict(metadata or {})
    pump_mu = int(metadata.get("pump_mu", pump_mu))
    fsr_hz = metadata.get("fsr_hz", None)

    # Normalise/validate inputs and guarantee ascending-mu ordering.
    mu_arr, P_arr = comb_line_powers(P_mu, mu, pump_mu=pump_mu)

    warnings_list: list[str] = []

    if smooth_modes == "auto":
        window, converged = _auto_smooth_window(
            mu_arr, P_arr, exclude_pump=exclude_pump, floor_db=floor_db,
            interp_factor=interp_factor, level_db=level_db, pump_mu=pump_mu,
            w_min=auto_w_min, w_max=auto_w_max)
        if not converged:
            warnings_list.append(
                f"auto smoothing did not converge to a single-lobe envelope "
                f"below window {auto_w_max}; the comb-core interference "
                f"modulation is deeper than {level_db} dB -- the 3 dB span is "
                f"fringe-sensitive; reporting the window-{window} result")
    else:
        window = int(smooth_modes)

    env = _build_envelope(mu_arr, P_arr, exclude_pump=exclude_pump,
                          smooth_modes=window, floor_db=floor_db,
                          interp_factor=interp_factor, pump_mu=pump_mu)
    mu_env, s_db_env = env["mu_env"], env["s_db_env"]

    left, right, peak_mu, reference_db, ncross = _span_from_envelope(
        mu_env, s_db_env, level_db)

    one_sided = False
    span_modes = math.nan
    dynamic_range = float(np.max(s_db_env) - np.min(s_db_env))
    if dynamic_range < level_db:
        warnings_list.append(
            f"envelope dynamic range {dynamic_range:.2f} dB < {level_db} dB: "
            f"it never falls {level_db} dB below its maximum; span is undefined")
    elif math.isnan(left) != math.isnan(right):
        one_sided = True
        missing = "left" if math.isnan(left) else "right"
        warnings_list.append(
            f"3 dB crossing not bracketed on the {missing} side (retained band "
            f"ends first); reporting a one-sided span flag, span = NaN")
    elif math.isnan(left) and math.isnan(right):
        warnings_list.append(
            "no 3 dB crossing on either side within the retained band")
    else:
        span_modes = right - left
        # DW-dominated check: a soliton comb is centred on the pump, so its
        # 3 dB band must straddle mu = 0.  If it does not, the envelope peak is
        # a dispersive wave / far shoulder, not the sech^2 core.
        if not (left <= pump_mu <= right):
            warnings_list.append(
                f"3 dB band [{left:.1f}, {right:.1f}] does not straddle the "
                f"comb centre mu={pump_mu}: the smoothed-envelope peak at "
                f"mu={peak_mu:.1f} is a dispersive-wave/shoulder feature, so "
                f"this span is NOT the sech^2 comb bandwidth")

    span_hz = span_modes * float(fsr_hz) if fsr_hz is not None else math.nan
    if fsr_hz is None:
        warnings_list.append(
            "metadata has no 'fsr_hz'; span_hz/THz/GHz reported as NaN")

    for w in warnings_list:
        warnings.warn(w, RuntimeWarning, stacklevel=2)

    return {
        "metric": "three_db_span",
        "metric_definition": (
            "Full width, in relative cavity-mode number Delta_mu, of the "
            "pump-excluded, numerical-floor-masked, dB-domain median-smoothed, "
            "PCHIP-interpolated comb-line power envelope, measured between the "
            "OUTERMOST points where the envelope falls level_db below its "
            f"maximum (level_db = {level_db} dB, i.e. the -3 dB / half-power "
            "level). Converted to frequency via Delta_mu * FSR."),
        "span_modes": float(span_modes),
        "span_hz": float(span_hz),
        "span_thz": float(span_hz / 1e12) if math.isfinite(span_hz) else math.nan,
        "span_ghz": float(span_hz / 1e9) if math.isfinite(span_hz) else math.nan,
        "left_crossing_mu": float(left),
        "right_crossing_mu": float(right),
        "envelope_peak_mu": float(peak_mu),
        "reference_level_db": float(reference_db),
        "n_crossings": int(ncross),
        "one_sided": bool(one_sided),
        "fsr_hz": float(fsr_hz) if fsr_hz is not None else math.nan,
        "params": {
            "exclude_pump": bool(exclude_pump),
            "smooth_modes_requested": smooth_modes,
            "smooth_modes_used": int(env["smooth_modes"]),
            "floor_db": float(floor_db),
            "interp_factor": int(interp_factor),
            "level_db": float(level_db),
            "pump_mu": int(pump_mu),
        },
        "warnings": warnings_list,
        "units": {
            "span_modes": "cavity modes (relative mode number Delta_mu)",
            "span_hz": "Hz",
            "span_thz": "THz",
            "span_ghz": "GHz",
            "left_crossing_mu": "cavity mode number (mu)",
            "right_crossing_mu": "cavity mode number (mu)",
            "envelope_peak_mu": "cavity mode number (mu)",
            "reference_level_db": "dB, relative to the strongest non-pump line",
            "fsr_hz": "Hz",
            "floor_db": "dB",
            "level_db": "dB",
        },
    }


# ---------------------------------------------------------------------------
# Optional cross-check: dB-domain sech^2 core fit (NOT the primary method)
# ---------------------------------------------------------------------------
def sech2_core_fwhm(mu, P_mu, *, core_mu=200, exclude_pump=True, pump_mu=0,
                    level_db=DEFAULT_LEVEL_DB, fsr_hz=None):
    """Optional cross-check: FWHM from a dB-domain sech^2 fit of the comb CORE.

    Fits ``10*log10(sech^2((mu - mu_c)/w)) + c`` to the retained comb lines with
    ``|mu| <= core_mu`` and reports the closed-form ``level_db`` full width,
    ``FWHM = 2 * w * arccosh(sqrt(10**(level_db/10) - 1 + ... ))`` (for the
    -3 dB / half-power level this is ``2 * w * arccosh(sqrt(2))``).  This is a
    CROSS-CHECK only -- the primary metric (:func:`three_db_span`) never fits a
    global sech^2 because the wings carry dispersive-wave features a single
    sech^2 cannot represent -- but over the core the sech^2 fit is faithful and
    provides an independent width estimate.  Returns a dict (NaN width on fit
    failure, never raises).
    """
    from scipy.optimize import curve_fit

    mu = np.asarray(mu).astype(np.float64)
    P = np.asarray(P_mu, dtype=np.float64)
    keep = np.isfinite(P) & (P > 0.0) & (np.abs(mu) <= core_mu)
    if exclude_pump:
        keep &= mu != pump_mu
    x, p = mu[keep], P[keep]
    if x.size < 5:
        return {"fwhm_modes": math.nan, "reason": "too few core lines"}
    ref = float(p.max())
    y = 10.0 * np.log10(p / ref)

    def sech2_db(m, w, c, mc):
        a = np.abs(m - mc) / w
        log_cosh = a + np.log1p(np.exp(-2.0 * a)) - math.log(2.0)
        return c - 20.0 / math.log(10.0) * log_cosh

    try:
        popt, _ = curve_fit(sech2_db, x, y, p0=[max(x.max() / 3.0, 5.0), 0.0, 0.0],
                            maxfev=40000)
    except Exception as exc:  # pragma: no cover - fit robustness guard
        return {"fwhm_modes": math.nan, "reason": f"fit failed: {exc}"}
    w = abs(float(popt[0]))
    ratio = 10.0 ** (level_db / 10.0)             # power drop factor (2 at 3 dB)
    half = w * math.acosh(math.sqrt(ratio))
    fwhm = 2.0 * half
    rms = float(np.sqrt(np.mean((sech2_db(x, *popt) - y) ** 2)))
    span_hz = fwhm * float(fsr_hz) if fsr_hz is not None else math.nan
    return {
        "fwhm_modes": float(fwhm),
        "fwhm_hz": float(span_hz),
        "fwhm_thz": float(span_hz / 1e12) if math.isfinite(span_hz) else math.nan,
        "width_w_modes": float(w),
        "center_mu": float(popt[2]),
        "peak_db": float(popt[1]),
        "fit_rms_db": rms,
        "core_mu": int(core_mu),
        "note": "cross-check only (dB-domain sech^2 core fit); not the primary "
                "3 dB span metric",
    }


# ---------------------------------------------------------------------------
# 4. Conversion efficiency (Feature 2)
# ---------------------------------------------------------------------------
# Metric-definition strings (also written verbatim into the JSON output).
ETA_INTRA_DEFINITION = (
    "Intracavity comb fraction eta_intra = sum_{mu != 0} P_mu / sum_mu P_mu: the "
    "fraction of the CYCLE-AVERAGED intracavity power carried by comb lines other "
    "than the pump mode (mu = 0). The pump line is identified from the "
    "simulation's mode indexing (mu = 0), never as 'the strongest line'. Input "
    "P_mu is the cycle-averaged LINEAR power spectrum <|a_mu|^2>: the average is "
    "taken on |a_mu|^2 per breather-phase snapshot BEFORE this ratio is formed, "
    "so eta_intra is a RATIO OF AVERAGES, never an average of per-snapshot ratios. "
    "The only excluded line is the pump itself -- no envelope/threshold/dB-cutoff "
    "mask is applied (numerical-floor lines contribute negligibly to the linear "
    "power sum). LIMITATIONS: single-point estimate at one detuning, "
    "breather-cycle-averaged; the entire pump line (residual pump + the soliton's "
    "own mu = 0 component) is conservatively assigned to 'pump', so the soliton's "
    "share is undercounted; eta_intra is an INTRACAVITY quantity and is NOT "
    "comparable to experimentally quoted (bus-waveguide) conversion efficiencies."
)

ETA_DEFINITION = (
    "Pump-to-comb conversion efficiency eta = P_comb,out / P_in: the fraction of "
    "on-chip pump power P_in that emerges in the bus waveguide as non-pump comb "
    "light. Intracavity->bus derivation in the repo's LLE normalization "
    "(simulator/lle_solver.py): the fast-time field E(tau) has intracavity energy "
    "U_int = sum_tau |E|^2 (t_r/n_tau) [J] and mean <|E|^2> = U_int/t_r = "
    "(1/n_tau^2) sum_mu P_mu for P_mu = |fftshift(fft(E))_mu|^2 (Parseval, numpy "
    "FFT). The solver's intrinsic-loss power is P_abs = kappa_i <|E|^2>, so by the "
    "same rate*energy relation each mode radiates P_out,mu = kappa_c W_mu into the "
    "bus, with W_mu = P_mu / n_tau^2 the per-mode share of <|E|^2> and kappa_c = "
    "kappa_ext the external out-coupling rate (this device is over-coupled, "
    "kappa_c ~ 4 kappa_i). A non-pump mode has no input field to interfere with, "
    "so P_comb,out = kappa_c sum_{mu != 0} W_mu = (kappa_c/n_tau^2) sum_{mu != 0} "
    "P_mu and eta = kappa_c sum_{mu != 0} P_mu / (n_tau^2 P_in) = kappa_c "
    "eta_intra <|E|^2> / P_in. Requires an ABSOLUTE (not pump-normalized) "
    "cycle-averaged spectrum plus kappa_c and P_in from the run config; returns "
    "None with a logged explanation otherwise (never guesses). LIMITATIONS: "
    "single-point estimate at one detuning, breather-cycle-averaged; the pump line "
    "(residual pump + soliton mu = 0 component) is conservatively assigned "
    "entirely to 'pump' and excluded from P_comb,out. Unlike eta_intra, eta is "
    "(approximately) comparable to experimentally quoted conversion efficiencies."
)


def _lookup(config, *keys):
    """First non-None value among ``keys`` in a dict-like OR attribute-holding config."""
    if config is None:
        return None
    for k in keys:
        if isinstance(config, dict):
            v = config.get(k)
        else:
            v = getattr(config, k, None)
        if v is not None:
            return v
    return None


def _resolve_kappa_c(config):
    """Absolute external out-coupling rate kappa_c [rad/s] from config, else None.

    Prefers an explicit rate (``kappa_c_rad_per_s`` / ``kappa_c``); falls back to
    ``omega0 / coupling_q`` with ``omega0 = 2*pi*c / pump_wavelength_m`` -- the same
    resolution order as :func:`simulator.lle_solver.resolve_cavity_rates`. A bare
    coupling *ratio* is insufficient (eta needs the absolute rate), so it is not
    accepted here.
    """
    kc = _lookup(config, "kappa_c_rad_per_s", "kappa_c")
    if kc is not None:
        return float(kc)
    q_c = _lookup(config, "coupling_q")
    lam = _lookup(config, "pump_wavelength_m")
    if q_c is not None and lam is not None and float(q_c) > 0:
        return 2.0 * math.pi * _C_LIGHT / float(lam) / float(q_c)
    return None


def _resolve_pin(config):
    """On-chip pump power P_in [W] from config, else None."""
    pin = _lookup(config, "pin_w", "pin", "pin_watts", "on_chip_pump_w")
    return float(pin) if pin is not None else None


def _split_pump(mu, P_mu, pump_mu):
    """Return ``(pump_power, comb_sum, total)`` with the pump at ``mu == pump_mu``.

    Uses :func:`comb_line_powers` (validates finiteness/non-negativity and warns
    with an argmax fallback if ``pump_mu`` is absent from the index).
    """
    m, P = comb_line_powers(P_mu, mu, pump_mu=pump_mu)
    total = float(np.sum(P))
    hits = np.nonzero(m == pump_mu)[0]
    i_pump = int(hits[0]) if hits.size else int(np.argmax(P))
    p_pump = float(P[i_pump])
    return p_pump, total - p_pump, total


def intracavity_comb_fraction(mu, P_mu, *, pump_mu: int = 0) -> float:
    """Intracavity comb fraction ``eta_intra`` (always computable).

    ``eta_intra = sum_{mu != 0} P_mu / sum_mu P_mu`` -- the fraction of
    intracavity power in comb lines other than the pump.  This is NOT the
    experimental conversion efficiency, but it is unambiguous, dataset-
    independent, scale-invariant (a pump-normalised and an absolute spectrum give
    the identical value), and monotonically related to comb strength.

    See :data:`ETA_INTRA_DEFINITION` for the full definition and limitations.
    Key points enforced here:

    * **Pump = mu == pump_mu** from the mode indexing, never the strongest line
      (via :func:`comb_line_powers`; argmax is only a warned fallback when the
      index lacks the pump).
    * **Total non-pump power** is the numerator -- no floor/threshold/dB-cutoff
      mask; the only excluded line is the pump.  Numerical-floor lines add
      negligibly to a linear-power sum, so no floor mask is needed.
    * **Ratio of averages, not average of ratios.**  ``P_mu`` must already be the
      cycle-averaged ``<|a_mu|^2>`` (averaged in LINEAR power per snapshot); this
      function sums those averages and THEN divides, so it forms a ratio of
      averages.  It never averages per-snapshot ratios (which would be biased).

    Returns the float fraction in ``[0, 1)``.
    """
    _, comb_sum, total = _split_pump(mu, P_mu, pump_mu)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("total intracavity power must be finite and > 0")
    return comb_sum / total


def conversion_efficiency_report(mu, P_mu, config, *, pump_mu: int = 0) -> dict:
    """Both efficiency metrics + all inputs in one dict (for JSON / plotting).

    The JSON-ready aggregator behind :func:`intracavity_comb_fraction` and
    :func:`pump_to_comb_efficiency`: computes ``eta_intra`` (always) and attempts
    ``eta`` (the bus-waveguide metric).  ``eta`` is left ``None`` -- with a
    human-readable ``eta_reason`` -- when either the config lacks ``kappa_c`` /
    ``P_in`` or the spectrum has no absolute power scale (a pump-normalised
    spectrum, unless an explicit ``mean_intracavity_energy_j`` anchor is
    supplied).  Never guesses.

    Returns a dict with ``eta_intra``, ``eta``, ``eta_reason``, and every input
    used (pump/comb/total power sums, ``kappa_c``, ``P_in``, ``n_tau_fft``,
    ``mean_intracavity_energy_j``, and whether the input was pump-normalised).
    See :data:`ETA_INTRA_DEFINITION` / :data:`ETA_DEFINITION` for the metric
    definitions and limitations.
    """
    p_pump, comb_sum, total = _split_pump(mu, P_mu, pump_mu)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("total intracavity power must be finite and > 0")
    eta_intra = comb_sum / total
    n_tau = int(_lookup(config, "n_tau_fft") or np.asarray(P_mu).size)

    kappa_c = _resolve_kappa_c(config)
    pin = _resolve_pin(config)
    w_total = _lookup(config, "mean_intracavity_energy_j")
    normalized = bool(_lookup(config, "spectrum_pump_normalized"))

    eta = None
    eta_reason = ""
    w_used = None
    if kappa_c is None or pin is None or not (pin > 0):
        eta_reason = (
            "kappa_c (external out-coupling rate) and/or on-chip pump power P_in "
            "is not unambiguously available in the run config; eta not computed "
            "(not guessing).")
    elif w_total is None and normalized:
        eta_reason = (
            "the input spectrum is pump-normalised (|a_0|^2 == 1), so the absolute "
            "intracavity power scale <|E|^2> was normalised away; the absolute "
            "out-coupled comb power (hence eta) cannot be recovered without an "
            "absolute-power anchor (an un-normalised spectrum, or "
            "mean_intracavity_energy_j). eta_intra is still reported.")
    else:
        # <|E|^2> [repo J]: an explicit anchor if given, else recovered from the
        # ABSOLUTE FFT powers via Parseval (<|E|^2> = sum_mu P_mu / n_tau^2).
        w_used = float(w_total) if w_total is not None else total / float(n_tau) ** 2
        # eta = kappa_c * (comb share of <|E|^2>) / P_in. The comb share uses the
        # eta_intra RATIO of the already-averaged powers, so this stays a ratio of
        # averages (never an average of per-snapshot ratios).
        eta = float(kappa_c * (eta_intra * w_used) / pin)

    return {
        "eta_intra": float(eta_intra),
        "eta": eta,
        "eta_reason": eta_reason,
        "pump_mu": int(pump_mu),
        "pump_power": p_pump,
        "comb_power_sum": comb_sum,
        "total_power_sum": total,
        "kappa_c_rad_per_s": kappa_c,
        "pin_w": pin,
        "n_tau_fft": n_tau,
        "mean_intracavity_energy_j": w_used,
        "spectrum_pump_normalized": normalized,
    }


def pump_to_comb_efficiency(mu, P_mu, config, *, pump_mu: int = 0):
    """Pump-to-comb conversion efficiency ``eta = P_comb,out / P_in`` (or ``None``).

    The physically standard metric.  Returns the float efficiency when the run
    config supplies the out-coupling rate ``kappa_c`` (``kappa_c_rad_per_s`` /
    ``kappa_c``, or ``coupling_q`` + ``pump_wavelength_m``) and the on-chip pump
    power ``P_in`` (``pin_w``), AND the spectrum carries an absolute power scale;
    otherwise returns ``None`` after logging why (it never guesses a missing
    coupling rate, pump power, or absolute scale).

    ``config`` is a dict-like (or attribute-holding) object.  Recognised keys:
    ``kappa_c_rad_per_s`` / ``kappa_c`` / (``coupling_q`` + ``pump_wavelength_m``),
    ``pin_w``, optional ``n_tau_fft`` (defaults to ``len(P_mu)`` -- the full FFT
    grid), optional ``mean_intracavity_energy_j`` (an explicit ``<|E|^2>`` anchor,
    which bypasses the ``n_tau^2`` FFT factor), and ``spectrum_pump_normalized``
    (set True when ``P_mu`` is pump-normalised, which forces ``None`` unless the
    energy anchor is given).

    See :data:`ETA_DEFINITION` for the exact intracavity->bus derivation in the
    repo's LLE normalization (``P_out,mu = kappa_c * P_mu / n_tau^2``) and the
    limitations.  For the JSON-ready bundle (both metrics, params, reason) call
    :func:`conversion_efficiency_report`.
    """
    detail = conversion_efficiency_report(mu, P_mu, config, pump_mu=pump_mu)
    if detail["eta"] is None and detail["eta_reason"]:
        logger.info("pump_to_comb_efficiency: eta unavailable -- %s",
                    detail["eta_reason"])
    return detail["eta"]


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------
def plot_spectrum_with_span(mu, P_mu, span_result, path, *, metadata=None,
                            title_extra="", dpi=300, also_pdf=True,
                            extra_metrics_lines=None):
    """Annotated comb-spectrum plot showing the 3 dB span overlay.

    Draws the comb lines (dB relative to the strongest non-pump line), the
    smoothed envelope curve, the horizontal ``reference - level_db`` line, the
    shaded 3 dB span region, and a text annotation of the span in both GHz/THz
    and mode count.  Matches the repo's matplotlib conventions and saves both a
    300-dpi PNG and a PDF (per shared analysis conventions).  ``path`` is the
    PNG path; the PDF sits beside it.

    ``extra_metrics_lines`` (Feature 2): optional list of strings appended to the
    same annotation box so the figure carries a COMBINED metrics readout (3 dB
    span + conversion efficiency) rather than a near-duplicate second figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    metadata = dict(metadata or {})
    p = span_result["params"]
    env = _build_envelope(mu, P_mu, exclude_pump=p["exclude_pump"],
                          smooth_modes=p["smooth_modes_used"],
                          floor_db=p["floor_db"], interp_factor=p["interp_factor"],
                          pump_mu=p["pump_mu"])
    mu_env, s_db_env = env["mu_env"], env["s_db_env"]
    m_ret, s_ret = env["mu_ret"], env["s_db_raw"]

    ref = span_result["reference_level_db"]
    level_db = p["level_db"]
    left = span_result["left_crossing_mu"]
    right = span_result["right_crossing_mu"]
    span_modes = span_result["span_modes"]
    span_thz = span_result["span_thz"]
    span_ghz = span_result["span_ghz"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    # retained comb lines (stems + markers), pump excluded from the envelope
    ax.vlines(m_ret, np.minimum(s_ret, ref - 1.5 * max(level_db, 1.0) - 40),
              s_ret, color="tab:blue", lw=0.4, alpha=0.35)
    ax.plot(m_ret, s_ret, ".", ms=2.2, color="tab:blue",
            label=f"comb lines (pump excluded), n_tau={np.asarray(mu).size}")
    # smoothed envelope
    ax.plot(mu_env, s_db_env, "-", lw=1.6, color="tab:orange",
            label=f"envelope (median w={p['smooth_modes_used']} + PCHIP, dB)")

    metrics_lines = []
    if math.isfinite(span_modes):
        ax.axhline(ref - level_db, color="tab:red", ls="--", lw=1.0,
                   label=f"reference $-${level_db:g} dB")
        ax.axvspan(left, right, color="tab:red", alpha=0.12,
                   label="3 dB span")
        for xc in (left, right):
            ax.axvline(xc, color="tab:red", ls=":", lw=0.9, alpha=0.8)
        metrics_lines += [
            f"3 dB span = {span_modes:.1f} modes",
            f"= {span_ghz:.1f} GHz ({span_thz:.2f} THz)",
            f"[{left:.1f}, {right:.1f}] $\\mu$",
        ]
        pad = 0.9 * span_modes
        ax.set_xlim(left - pad, right + pad)
    else:
        ax.set_xlim(m_ret.min(), m_ret.max())

    # Combined metrics box: 3 dB span (above) + any extra lines (e.g. Feature 2
    # conversion efficiency), kept in ONE annotation rather than a second figure.
    if extra_metrics_lines:
        if metrics_lines:
            metrics_lines.append("-" * 18)                       # thin separator
        metrics_lines += [str(s) for s in extra_metrics_lines]
    if metrics_lines:
        ax.annotate(
            "\n".join(metrics_lines),
            xy=(0.0, 0.0), xytext=(0.02, 0.06), textcoords="axes fraction",
            ha="left", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="tab:red",
                      alpha=0.9))

    ax.axhline(ref, color="0.5", ls="-", lw=0.6, alpha=0.6)
    ax.set_xlabel(r"cavity mode index $\mu$  ($\mu=0$ = pump)")
    ax.set_ylabel(r"power (dB, rel. strongest non-pump line)")
    dw_txt = metadata.get("delta_omega_over_kappa", None)
    dw_str = f" @ $\\delta\\omega$ = {dw_txt:.1f}$\\kappa$" if dw_txt else ""
    ttl = (f"Single-DKS 3 dB spectral span{dw_str} "
           f"(cycle-averaged, {metadata.get('n_rt_averaged', '?')} RT)")
    if title_extra:
        ttl += f"\n{title_extra.lstrip(' -')}"
    ax.set_title(ttl, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = Path(path)
    fig.savefig(path, dpi=dpi)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. Soliton steps: power-vs-detuning staircase (Feature 3)
# ---------------------------------------------------------------------------
# The physically meaningful per-detuning observable of a detuning sweep is the
# power averaged over the LAST portion of a long hold at that detuning: at the
# validated operating point the attractor breathes (limit cycle), so the
# instantaneous power oscillates and only a slow-time / cycle average is a
# reproducible observable.  :func:`hold_window_average` performs that averaging
# in LINEAR power (never dB, never on complex fields); the driver
# ``analysis/run_detuning_sweep.py`` feeds it the per-round-trip power history of
# each hold.  :func:`detect_power_steps` then finds the soliton-step
# discontinuities in the assembled trace, and :func:`plot_soliton_steps` renders
# the publication figure.
SOLITON_STEP_DEFINITION = (
    "Soliton step: a discontinuity in the per-detuning averaged cavity power "
    "P(delta_omega) as the pump-cavity detuning is swept quasi-statically through "
    "the soliton existence range. For each detuning the observable is the power "
    "averaged over the FINAL avg_frac of a long constant-detuning hold (LINEAR "
    "power, never dB / never complex), which simultaneously discards the post-step "
    "transient and cycle-averages the deterministic breathing (the attractor is a "
    "limit cycle, so the instantaneous power oscillates). Step edges are the "
    "first-difference outliers of that trace: |diff(P)[i] - median(diff P)| > "
    "k * 1.4826 * MAD(diff P) (robust scale; conservative k). A step marks a "
    "soliton nucleation/annihilation boundary of the branch, at which the "
    "intracavity power (and hence the through-port transmission via P_trans = "
    "P_in - kappa_i * <|E|^2>) jumps discretely."
)

DEFAULT_STEP_K = 6.0        # conservative first-difference MAD multiple for a step
_MAD_TO_SIGMA = 1.4826      # scale a MAD to a Gaussian-sigma estimate


def hold_window_average(series, *, avg_frac: float = 0.25):
    """Average the FINAL ``avg_frac`` of a per-hold power series in LINEAR power.

    ``series`` is the per-round-trip (or per-snapshot) sequence of a POSITIVE,
    LINEAR power/energy observable recorded while the field is held at one
    detuning (e.g. the solver's ``U_int_history`` slice, or a per-round-trip
    ``sum_mu |a_mu|^2``).  The window is the last ``avg_frac`` of the hold, whose
    mean is the plotted per-detuning value and whose std is the breathing-
    amplitude indicator (an error bar).

    The averaging is done in LINEAR power -- never on dB values (which bias the
    mean toward the peak) and never on complex fields (breathing-phase drift
    cancels real power) -- so passing a dB or complex ``series`` is rejected.
    Selecting the LAST fraction discards the per-step re-settling transient and,
    because the hold spans many breathing periods, cycle-averages the limit-cycle
    oscillation.

    The window starts at index ``floor(n * (1 - avg_frac))`` and always contains
    at least one sample.  Returns a dict with ``mean``, ``std`` (population),
    ``i_start``, ``n_window``, ``n_total`` and ``avg_frac``.
    """
    x = np.asarray(series)
    if np.iscomplexobj(x):
        raise ValueError(
            "series is complex; average LINEAR power |a|^2, never the field")
    x = x.astype(np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("series is empty")
    if not (0.0 < avg_frac <= 1.0):
        raise ValueError(f"avg_frac must be in (0, 1], got {avg_frac}")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains non-finite values")
    i_start = int(math.floor(n * (1.0 - avg_frac)))
    i_start = min(max(i_start, 0), n - 1)          # >= 1 sample in the window
    w = x[i_start:]
    return {
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "i_start": int(i_start),
        "n_window": int(w.size),
        "n_total": int(n),
        "avg_frac": float(avg_frac),
    }


def detect_power_steps(x, y, *, k: float = DEFAULT_STEP_K):
    """Robust step-edge detection on a 1-D averaged trace ``y(x)``.

    A soliton step is a DISCONTINUITY in ``y``: a first difference
    ``dy[i] = y[i+1] - y[i]`` that is an outlier relative to the bulk of the
    differences.  Robustly (no fit, no per-dataset threshold):

    * ``med = median(dy)``, ``sigma = 1.4826 * MAD(dy)`` (a Gaussian-sigma
      estimate from the median absolute deviation about the median);
    * edge ``i`` is a step where ``|dy[i] - med| > k * sigma`` (default ``k = 6``
      -- conservative, only large jumps flag; ``k`` is configurable).

    The scaled MAD keys the threshold to the trace's own smooth-region ripple, so
    a gentle branch (small, uniform ``dy``) does not trip while a genuine jump
    does.  When ``sigma == 0`` (a perfectly linear ramp plus isolated jumps) a
    tiny relative floor derived from the data range is used instead of zero.

    Consecutive flagged edges are one step; the un-flagged edges partition the
    samples into plateaus.  Returns a dict with the step ``edges`` (index ``i``,
    the step lying between ``x[i]`` and ``x[i+1]``), ``step_x`` (edge midpoints),
    ``step_dy`` (signed magnitudes), ``plateaus`` (inclusive index ranges) and
    ``plateau_bounds_x`` (their ``x`` spans), plus the robust scale used.  Never
    raises on short input: fewer than 3 samples yields no steps.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must match; got {x.shape} vs {y.shape}")
    n = y.size
    empty = {
        "edges": [], "step_x": [], "step_dy": [],
        "plateaus": [[0, n - 1]] if n else [],
        "plateau_bounds_x": [[float(x[0]), float(x[-1])]] if n else [],
        "n_steps": 0, "k": float(k), "sigma": 0.0, "median_dy": 0.0,
        "units": {"step_x": "same as x (detuning)", "step_dy": "same as y (power)"},
    }
    if n < 3:
        return empty

    dy = np.diff(y)
    med = float(np.median(dy))
    mad = float(np.median(np.abs(dy - med)))
    sigma = _MAD_TO_SIGMA * mad
    if sigma <= 0.0:
        span = float(np.max(y) - np.min(y))
        sigma = max(1e-12, 1e-6 * (span if span > 0 else 1.0))
    thr = float(k) * sigma

    flagged = np.abs(dy - med) > thr            # length n-1, edge i joins i, i+1

    # Plateaus = connected runs of nodes joined by UN-flagged edges.
    plateaus = []
    start = 0
    for i in range(n - 1):
        if flagged[i]:
            plateaus.append([start, i])
            start = i + 1
    plateaus.append([start, n - 1])

    edges = [int(i) for i in np.nonzero(flagged)[0]]
    step_x = [0.5 * (float(x[i]) + float(x[i + 1])) for i in edges]
    step_dy = [float(dy[i]) for i in edges]
    plateau_bounds_x = [[float(x[a]), float(x[b])] for a, b in plateaus]

    return {
        "edges": edges,
        "step_x": step_x,
        "step_dy": step_dy,
        "plateaus": [[int(a), int(b)] for a, b in plateaus],
        "plateau_bounds_x": plateau_bounds_x,
        "n_steps": len(edges),
        "k": float(k),
        "sigma": float(sigma),
        "median_dy": med,
        "units": {"step_x": "same as x (detuning)", "step_dy": "same as y (power)"},
    }


def single_dks_region(detuning, is_single):
    """Longest contiguous single-DKS detuning span and its annihilation edge.

    ``detuning`` and ``is_single`` are the per-step detuning and single-soliton
    flag of a sweep (any order).  Returns ``(lo, hi, annihilation)`` where the
    span is the longest run of ``is_single`` True (in ascending detuning) and
    ``annihilation`` is the midpoint between the lowest-detuning soliton point and
    the next-lower non-single point -- the soliton step (the branch's lower edge).
    Any element is ``None`` when there is no single-DKS point, or the branch
    reaches the sweep edge with no lower non-single neighbour.
    """
    dw = np.asarray(detuning, dtype=np.float64).ravel()
    flag = np.asarray(is_single, dtype=bool).ravel()
    if dw.shape != flag.shape:
        raise ValueError("detuning and is_single must have the same shape")
    order = np.argsort(dw)
    dw, flag = dw[order], flag[order]
    best_len, best = 0, None
    i = 0
    while i < flag.size:
        if flag[i]:
            j = i
            while j < flag.size and flag[j]:
                j += 1
            if (j - i) > best_len:
                best_len, best = j - i, (i, j - 1)
            i = j
        else:
            i += 1
    if best is None:
        return None, None, None
    lo_i, hi_i = best
    lo_k, hi_k = float(dw[lo_i]), float(dw[hi_i])
    annih = float(0.5 * (dw[lo_i] + dw[lo_i - 1])) if lo_i > 0 else None
    return lo_k, hi_k, annih


def soliton_count_transitions(detuning, soliton_count):
    """Every soliton-number transition edge of a sweep, in ascending detuning.

    ``detuning`` and ``soliton_count`` are the per-step detuning and measured
    soliton count of a sweep (any order; sorting/validation mirrors
    :func:`single_dks_region`).  An edge ``i`` joins the adjacent ascending-
    detuning samples ``i`` and ``i + 1`` (the :func:`detect_power_steps` edge
    convention), and a transition is any edge where the count changes.  Returns
    a list of dicts, one per transition, in ascending edge order::

        {"edge_index":  i,
         "dw_mid":      0.5 * (x[i] + x[i + 1]),   # edge midpoint (detuning)
         "n_high_side": count[i + 1],              # high-detuning side
         "n_low_side":  count[i],                  # low-detuning side
         "delta_n":     count[i + 1] - count[i]}

    ``delta_n`` is the number of solitons LOST crossing the edge downward in
    detuning (the sweep direction): ``+1`` is a single N -> N-1 annihilation
    (one staircase step), larger values are merged annihilations, and a
    NEGATIVE value means the count rose going down -- on a clean staircase that
    never happens (it is what the driver's monotonicity gate rejects).

    Counts must be integer-valued and non-negative (a fractional or negative
    soliton number is always an upstream bug, never data).  Fewer than two
    samples yield no transitions.
    """
    dw = np.asarray(detuning, dtype=np.float64).ravel()
    c_raw = np.asarray(soliton_count).ravel()
    if dw.shape != c_raw.shape:
        raise ValueError("detuning and soliton_count must have the same shape")
    if np.iscomplexobj(c_raw):
        raise ValueError("soliton_count must be integer-valued, got complex")
    c_f = c_raw.astype(np.float64)
    if not np.all(np.isfinite(c_f)):
        raise ValueError("soliton_count contains non-finite values")
    if np.any(c_f != np.round(c_f)):
        raise ValueError("soliton_count must be integer-valued")
    if np.any(c_f < 0):
        raise ValueError("soliton_count must be non-negative")
    c = c_f.astype(np.int64)

    order = np.argsort(dw)
    dw, c = dw[order], c[order]
    return [
        {
            "edge_index": int(i),
            "dw_mid": float(0.5 * (dw[i] + dw[i + 1])),
            "n_high_side": int(c[i + 1]),
            "n_low_side": int(c[i]),
            "delta_n": int(c[i + 1] - c[i]),
        }
        for i in range(c.size - 1)
        if c[i + 1] != c[i]
    ]


def match_steps_to_transitions(steps, transitions, tol_samples: int = 1):
    """Align detected power steps with measured soliton-number transitions.

    ``steps`` is a :func:`detect_power_steps` result (uses its ``edges``,
    ``step_x`` and ``step_dy``) and ``transitions`` a
    :func:`soliton_count_transitions` list, both on the SAME ascending-detuning
    sample grid so their edge indices are comparable.  A detected power step and
    a soliton-count transition match when their edge indices differ by at most
    ``tol_samples`` (default 1: the hold that straddles a switching event may
    register the power drop one sample away from where the end-of-hold count
    flips).  Matching is one-to-one: candidate pairs are taken closest-first,
    and each step / transition is used at most once.

    A matched step is a VALIDATED soliton step -- a power discontinuity PROVEN
    to coincide with a change in the measured soliton number.  Only matched
    steps may be labelled as soliton steps anywhere downstream (figure legends,
    JSON blocks, prose).  The two unmatched populations carry their own honest
    physics:

    * ``unmatched_steps`` -- power discontinuities with NO state change.  At
      this device these are the near-resonance MI/CW power rise, and they must
      keep their existing "power-trace discontinuity" label; they are never
      soliton steps.
    * ``unmatched_transitions`` -- state changes with NO power step.  A
      power-muted transition (e.g. a final 1 -> 0 annihilation whose soliton
      is replaced by a comparable-energy background) lands here; forcing such
      an edge to register as a power step is forbidden.  Whether the final
      edge is matched or unmatched is DECIDED BY THE DATA on the trace the
      detector ran on -- on the 2026 regenerated staircase the 1 -> 0 edge
      resolves naturally on the pump-excluded comb power (which collapses
      there), so it is matched; on a total-power trace it may not be.

    Returns ``{"matched": [...], "unmatched_steps": [...],
    "unmatched_transitions": [...], "tol_samples": tol}``.  Each ``matched``
    entry combines both sides: ``step_edge_index``, ``transition_edge_index``,
    ``edge_distance``, ``step_x``, ``step_dy``, ``dw_mid``, ``n_high_side``,
    ``n_low_side``, ``delta_n``.  Each ``unmatched_steps`` entry has
    ``edge_index``, ``step_x``, ``step_dy``; ``unmatched_transitions`` entries
    are the transition dicts unchanged.  Pure bookkeeping -- no thresholds, no
    smoothing, no re-detection.
    """
    edges = [int(e) for e in steps["edges"]]
    step_x = [float(v) for v in steps["step_x"]]
    step_dy = [float(v) for v in steps["step_dy"]]
    if not (len(edges) == len(step_x) == len(step_dy)):
        raise ValueError("steps dict has inconsistent edges/step_x/step_dy")
    tol = int(tol_samples)
    if tol < 0:
        raise ValueError(f"tol_samples must be >= 0, got {tol_samples}")

    candidates = sorted(
        (abs(e - int(tr["edge_index"])), si, ti)
        for si, e in enumerate(edges)
        for ti, tr in enumerate(transitions)
        if abs(e - int(tr["edge_index"])) <= tol
    )
    used_steps, used_transitions = set(), set()
    matched = []
    for dist, si, ti in candidates:
        if si in used_steps or ti in used_transitions:
            continue
        used_steps.add(si)
        used_transitions.add(ti)
        tr = transitions[ti]
        matched.append({
            "step_edge_index": edges[si],
            "transition_edge_index": int(tr["edge_index"]),
            "edge_distance": int(dist),
            "step_x": step_x[si],
            "step_dy": step_dy[si],
            "dw_mid": float(tr["dw_mid"]),
            "n_high_side": int(tr["n_high_side"]),
            "n_low_side": int(tr["n_low_side"]),
            "delta_n": int(tr["delta_n"]),
        })
    matched.sort(key=lambda m: m["step_edge_index"])
    return {
        "matched": matched,
        "unmatched_steps": [
            {"edge_index": edges[si], "step_x": step_x[si],
             "step_dy": step_dy[si]}
            for si in range(len(edges)) if si not in used_steps
        ],
        "unmatched_transitions": [
            dict(transitions[ti]) for ti in range(len(transitions))
            if ti not in used_transitions
        ],
        "tol_samples": tol,
    }


def plateau_transition_energies(primary, soliton_count, transitions):
    """Plateau-level transition energies -- the observable robust to the
    count/energy one-hold offset at a mid-hold annihilation.

    A single-edge ``step_dy`` mis-measures a soliton's energy quantum when the
    integer, hold-quantized ``soliton_count`` decrements one hold before (or
    after) the continuous comb-energy drop completes: the drop is then split
    across the count-flip hold and an adjacent in-plateau hold, so no single
    edge carries the whole quantum (see ``analysis/results/
    staircase_forensics.md`` -- count/energy lag, verdict INTRINSIC LAG).  The
    PLATEAU-MEAN difference is insensitive to that offset: it compares the
    settled energy of the count-N branch with the settled energy of the
    count-(N-1) branch, ignoring the one or two transitional holds where the
    count and energy are mid-flip.

    For every transition (a :func:`soliton_count_transitions` dict, edge ``e``
    joining ascending samples ``e`` and ``e+1``) this computes::

        plateau_energy = mean(primary over the STABLE count-(n_high) plateau)
                       - mean(primary over the STABLE count-(n_low)  plateau)

    where each stable plateau is the maximal constant-count run adjacent to the
    edge with its boundary holds that touch a count transition removed (the
    transitional holds), always keeping at least one hold (a length-1 or -2
    plateau falls back to its full extent).  ``primary`` and ``soliton_count``
    are per-hold arrays in ASCENDING detuning order (the
    :func:`detect_power_steps` / :func:`soliton_count_transitions` convention);
    ``primary`` may be raw or normalised (the difference scales with it).

    Returns one dict per transition (input order) with ``edge_index``,
    ``n_high_side``, ``n_low_side``, ``delta_n``, the ``high_plateau`` /
    ``low_plateau`` full index ranges, the ``high_interior`` / ``low_interior``
    stable index ranges actually averaged, ``mean_high`` / ``mean_low``,
    ``plateau_energy`` (signed: high minus low) and ``per_quantum``
    (``|plateau_energy| / delta_n`` -- the energy of ONE annihilated soliton,
    so a merged N -> N-2 edge is divided by two).
    """
    y = np.asarray(primary, dtype=np.float64).ravel()
    c = np.asarray(soliton_count).ravel()
    if y.shape != c.shape:
        raise ValueError(
            f"primary and soliton_count must match; got {y.shape} vs {c.shape}")
    n = c.size
    c_int = c.astype(np.int64)

    # every count-change edge (transition boundary), from the counts alone.
    change_edges = {int(i) for i in range(n - 1) if c_int[i + 1] != c_int[i]}

    def _plateau_of(idx):
        """Maximal constant-count run [lo, hi] containing sample ``idx``."""
        lo = hi = int(idx)
        while lo - 1 >= 0 and c_int[lo - 1] == c_int[idx]:
            lo -= 1
        while hi + 1 < n and c_int[hi + 1] == c_int[idx]:
            hi += 1
        return lo, hi

    def _stable_interior(lo, hi):
        """Drop the boundary holds adjacent to a count transition (>= 1 kept)."""
        a, b = int(lo), int(hi)
        # a touches a transition below iff edge a-1 is a change edge
        if (a - 1) in change_edges:
            a += 1
        # b touches a transition above iff edge b is a change edge
        if b in change_edges:
            b -= 1
        if a > b:                       # length-1/-2 plateau: use its full extent
            return int(lo), int(hi)
        return a, b

    out = []
    for tr in transitions:
        e = int(tr["edge_index"])
        hi_lo, hi_hi = _plateau_of(e + 1)          # high-detuning (n_high) side
        lo_lo, lo_hi = _plateau_of(e)              # low-detuning (n_low) side
        hi_a, hi_b = _stable_interior(hi_lo, hi_hi)
        lo_a, lo_b = _stable_interior(lo_lo, lo_hi)
        mean_high = float(np.mean(y[hi_a:hi_b + 1]))
        mean_low = float(np.mean(y[lo_a:lo_b + 1]))
        plateau_energy = mean_high - mean_low
        delta_n = max(int(tr["delta_n"]), 1)
        out.append({
            "edge_index": e,
            "n_high_side": int(tr["n_high_side"]),
            "n_low_side": int(tr["n_low_side"]),
            "delta_n": int(tr["delta_n"]),
            "high_plateau": [int(hi_lo), int(hi_hi)],
            "low_plateau": [int(lo_lo), int(lo_hi)],
            "high_interior": [int(hi_a), int(hi_b)],
            "low_interior": [int(lo_a), int(lo_b)],
            "mean_high": mean_high,
            "mean_low": mean_low,
            "plateau_energy": float(plateau_energy),
            "per_quantum": float(abs(plateau_energy) / delta_n),
        })
    return out


def _moving_average(y, w):
    """Odd-window centred moving average (edge-replicated); display smoothing only."""
    y = np.asarray(y, dtype=np.float64)
    w = int(w)
    if w <= 1 or y.size < 2:
        return y.copy()
    if w % 2 == 0:
        w += 1
    w = min(w, y.size if y.size % 2 == 1 else y.size - 1)
    pad = w // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kern = np.ones(w) / w
    return np.convolve(ypad, kern, mode="valid")


def _draw_regions(ax, soliton_region, annihilation_kappa, steps, *, label=True,
                  any_soliton_region=None, matched_step_x=None):
    """Shade the single-DKS existence region, mark the soliton step, and (lightly)
    the power-trace discontinuities.  Shared by both panels; ``label=False``
    suppresses the legend entries (used on the secondary panel to avoid repeats).

    ``any_soliton_region`` (optional) additionally shades the any-soliton
    (``soliton_count >= 1``) existence span behind the single-DKS one.
    ``matched_step_x`` (optional) is the detuning midpoints of the VALIDATED
    soliton steps -- detected power discontinuities whose edge coincides with a
    measured soliton-number transition (:func:`match_steps_to_transitions`);
    they are drawn solid and legend-labelled as state-verified soliton steps,
    while every step in ``steps`` NOT in that set keeps the existing dotted
    "power-trace discontinuity" style (at this device those are the
    near-resonance MI/CW rise, not soliton steps)."""
    def _lbl(text):
        return text if label else None

    if any_soliton_region is not None:
        lo, hi = float(any_soliton_region[0]), float(any_soliton_region[1])
        ax.axvspan(lo, hi, color="tab:olive", alpha=0.10, zorder=0,
                   label=_lbl(r"any-soliton existence ($N \geq 1$)"))
    if soliton_region is not None:
        lo, hi = float(soliton_region[0]), float(soliton_region[1])
        ax.axvspan(lo, hi, color="tab:green", alpha=0.12, zorder=0,
                   label=_lbl("single-DKS existence"))
    if annihilation_kappa is not None:
        ax.axvline(float(annihilation_kappa), color="tab:red", ls="-", lw=1.6,
                   zorder=3, label=_lbl("soliton annihilation (step)"))
    matched_set = {float(x) for x in (matched_step_x or [])}
    # State-verified soliton steps: solid black, distinct from the tab:red
    # annihilation line and the gray dotted discontinuities.
    for j, xs in enumerate(sorted(matched_set)):
        ax.axvline(xs, color="black", ls="-", lw=1.4, zorder=3,
                   label=_lbl(r"soliton step ($N \to N{-}1$, state-verified)")
                   if j == 0 else None)
    # Power-trace discontinuities (from detect_power_steps) are a SECONDARY,
    # honestly-labelled overlay: at this operating point they key on the
    # near-resonance MI/CW power rise, NOT the soliton step (see the driver).
    unmatched = [xs for xs in (steps.get("step_x") if steps else [])
                 if float(xs) not in matched_set]
    for j, xs in enumerate(unmatched):
        ax.axvline(xs, color="0.45", ls=":", lw=1.0, zorder=2,
                   label=_lbl("power-trace discontinuity") if j == 0 else None)


def plot_soliton_steps(detuning_kappa, power, path, *, power_std=None,
                       transmission=None, soliton_region=None,
                       any_soliton_region=None,
                       annihilation_kappa=None, steps=None, metadata=None,
                       state_counts=None, match_tol_samples: int = 1,
                       observable_label=r"intracavity power  $\sum_\mu |a_\mu|^2$ "
                                         r"(norm.)",
                       second_panel_ylabel="norm. transmission",
                       second_panel_legend=("through-port power "
                                            "$P_\\mathrm{trans}/P_\\mathrm{in}$"),
                       smooth_window: int = 0, dpi: int = 300, also_pdf: bool = True):
    """Publication figure of the soliton-branch power vs pump-cavity detuning.

    ``detuning_kappa`` is the swept detuning in units of kappa (the repo's native
    plotting unit) and ``power`` is the per-detuning averaged cavity power -- the
    primary trace, plotted RAW.  ``power_std`` (optional) draws the per-step
    breathing-amplitude error bars (they are large on the breather sub-band of the
    soliton branch and collapse when the soliton annihilates, a visual marker of
    the step).  ``transmission`` (optional) adds a second panel; by default it is
    labelled as the normalised through-port power, but callers may relabel it via
    ``second_panel_ylabel`` / ``second_panel_legend`` to plot any secondary
    observable there (e.g. the pump-excluded comb power of the multi-soliton
    staircase driver).

    Region markers:

    * ``soliton_region`` -- an ``(lo_kappa, hi_kappa)`` span of the single-DKS
      existence region (from the per-step state flag), lightly shaded;
    * ``any_soliton_region`` -- an ``(lo_kappa, hi_kappa)`` span of the
      any-soliton (``soliton_count >= 1``) existence region, shaded behind it;
    * ``annihilation_kappa`` -- the soliton step (the branch's lower edge), drawn
      as a solid vertical line;
    * ``steps`` -- a :func:`detect_power_steps` result, drawn as light dotted
      lines and labelled honestly as *power-trace discontinuities* (which, for
      this operating point, mark the near-resonance MI/CW power rise rather than
      the soliton annihilation -- see the driver's note).

    ``state_counts`` (optional) is the per-detuning MEASURED soliton count of
    the sweep (same sample order as ``detuning_kappa``).  When given it is
    drawn as data, not decoration: a thin step-plot on a twin axis, with each
    plateau annotated "N = k" (plateaus narrower than 3 samples stay unlabelled
    to avoid overplotting -- the step trace itself still shows them).  When
    ``steps`` is also given, the detected power steps are aligned with the
    soliton-count transitions (:func:`match_steps_to_transitions`, tolerance
    ``match_tol_samples``; ``steps`` must have been computed on the
    ascending-detuning trace so the edge indices are comparable): matched steps
    -- the VALIDATED soliton steps -- get a distinct solid line + marker and
    the legend entry "soliton step (N -> N-1, state-verified)", while unmatched
    power discontinuities keep the existing dotted "power-trace discontinuity"
    style plus a note that they carry no state change (the near-resonance MI/CW
    rise at this device).  Without ``state_counts`` the drawing semantics are
    exactly the pre-existing ones.

    ``smooth_window`` enables a DISPLAY-ONLY moving-average overlay (default 0 =
    off); the raw trace always stays visible underneath and the smoothed curve is
    never saved as data.  ``metadata`` may carry ``kappa_rad_s`` (adds a secondary
    top axis in frequency-detuning MHz), ``caption`` and the thermal/noise
    settings.  Saves a 300-dpi PNG and a PDF (per repo conventions); ``path`` is
    the PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    metadata = dict(metadata or {})
    dwk = np.asarray(detuning_kappa, dtype=np.float64).ravel()
    P = np.asarray(power, dtype=np.float64).ravel()
    order = np.argsort(dwk)
    dwk, P = dwk[order], P[order]
    std = (np.asarray(power_std, dtype=np.float64).ravel()[order]
           if power_std is not None else None)
    T = (np.asarray(transmission, dtype=np.float64).ravel()[order]
         if transmission is not None else None)
    counts = (np.asarray(state_counts).ravel().astype(np.int64)[order]
              if state_counts is not None else None)

    # Align the detected power steps with the measured soliton-count
    # transitions; only the matched ones may be styled as soliton steps.
    align = None
    if counts is not None and steps is not None:
        align = match_steps_to_transitions(
            steps, soliton_count_transitions(dwk, counts),
            tol_samples=match_tol_samples)
    matched_x = [m["step_x"] for m in align["matched"]] if align else []

    two_panel = T is not None
    if two_panel:
        fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.0]})
        ax, axt = axes
    else:
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        axt = None

    _draw_regions(ax, soliton_region, annihilation_kappa, steps,
                  any_soliton_region=any_soliton_region,
                  matched_step_x=matched_x)

    # Raw primary trace (+ breathing error bars).
    if std is not None:
        ax.errorbar(dwk, P, yerr=std, fmt="o-", ms=3.4, lw=1.0, color="tab:blue",
                    ecolor="tab:blue", elinewidth=0.8, capsize=1.8, alpha=0.9,
                    zorder=4,
                    label="per-step avg (final 25% of hold); bars = breathing std")
    else:
        ax.plot(dwk, P, "o-", ms=3.4, lw=1.0, color="tab:blue", zorder=4,
                label="per-step avg (final 25% of hold)")

    if smooth_window and smooth_window > 1:
        ax.plot(dwk, _moving_average(P, smooth_window), "-", lw=1.6,
                color="tab:orange", alpha=0.85, zorder=5,
                label=f"display smoothing (w={int(smooth_window)}, not saved)")

    if steps is not None and not steps.get("step_x"):
        ax.annotate("no power-trace discontinuity detected", xy=(0.02, 0.04),
                    xycoords="axes fraction", ha="left", va="bottom", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5",
                              alpha=0.85))

    # Validated soliton steps: solid marker on the trace at the step midpoint
    # (the vertical lines are drawn by _draw_regions).
    if align is not None:
        for m in align["matched"]:
            i = m["step_edge_index"]
            ax.plot(m["step_x"], 0.5 * (P[i] + P[i + 1]), "D", color="black",
                    ms=6.5, mec="white", mew=0.6, zorder=6)
        if align["unmatched_steps"]:
            ax.annotate(
                "dotted = power discontinuity with NO soliton-count change\n"
                "(near-resonance MI/CW rise at this device -- not a soliton "
                "step)",
                xy=(0.02, 0.04), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.45",
                          alpha=0.85))

    ax.set_ylabel(observable_label)
    ax.grid(alpha=0.25)

    # Measured soliton count: a thin step-plot on a twin axis (data, not
    # decoration), with "N = k" plateau annotations.
    if counts is not None:
        axn = ax.twinx()
        axn.step(dwk, counts, where="mid", color="0.35", lw=0.9, alpha=0.9,
                 label="soliton count $N$ (measured)")
        axn.set_ylabel("soliton number $N$", fontsize=9, color="0.25")
        n_max = int(counts.max()) if counts.size else 0
        axn.set_ylim(-0.4, n_max + 2.6)          # headroom for the main legend
        axn.set_yticks(range(0, n_max + 1))
        axn.tick_params(axis="y", labelsize=8, colors="0.25")
        # Maximal constant-count runs; runs narrower than 3 samples stay
        # unlabelled (the step trace still shows them).
        run_start = 0
        for i in range(1, counts.size + 1):
            if i == counts.size or counts[i] != counts[run_start]:
                if i - run_start >= 3:
                    axn.annotate(f"N = {int(counts[run_start])}",
                                 xy=(0.5 * (dwk[run_start] + dwk[i - 1]),
                                     counts[run_start] + 0.18),
                                 ha="center", va="bottom", fontsize=7.5,
                                 color="0.25")
                run_start = i
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axn.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=8, loc="upper right")

    # Secondary top axis in frequency-detuning MHz (delta_omega / 2*pi).
    kappa = metadata.get("kappa_rad_s")
    if kappa:
        def _k2mhz(v):
            return np.asarray(v) * float(kappa) / (2.0 * math.pi) / 1e6

        def _mhz2k(v):
            return np.asarray(v) * 1e6 * 2.0 * math.pi / float(kappa)

        secax = ax.secondary_xaxis("top", functions=(_k2mhz, _mhz2k))
        secax.set_xlabel("pump-cavity detuning  $\\delta\\omega/2\\pi$  (MHz)",
                         fontsize=9)

    if two_panel:
        _draw_regions(axt, soliton_region, annihilation_kappa, steps,
                      label=False, any_soliton_region=any_soliton_region,
                      matched_step_x=matched_x)
        axt.plot(dwk, T, "s-", ms=3.0, lw=1.0, color="tab:purple", zorder=4,
                 label=second_panel_legend)
        axt.set_ylabel(second_panel_ylabel)
        axt.grid(alpha=0.25)
        axt.legend(fontsize=8, loc="lower left")
        axt.set_xlabel(r"pump-cavity detuning  $\delta\omega/\kappa$")
    else:
        ax.set_xlabel(r"pump-cavity detuning  $\delta\omega/\kappa$")

    caption = metadata.get("caption")
    if caption:
        fig.text(0.5, -0.02 if not two_panel else -0.01, caption, ha="center",
                 va="top", fontsize=7.5, wrap=True)

    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path
