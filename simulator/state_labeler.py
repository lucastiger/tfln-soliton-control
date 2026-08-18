"""Seven-class classification of an intracavity-field snapshot.

Two labelers live here and they must agree:

* :func:`make_state_labeler` builds a ``jax.lax.scan``-traceable closure with
  every threshold baked in as a Python float, so it can run INSIDE the solver's
  scan and label a snapshot without leaving the device;
* :func:`label_soliton_state` is the NumPy path, free to use SciPy peak finding
  and curve fitting, used for offline labeling of stored trajectories.

Both are driven by the SAME threshold dict, built once by
:func:`make_threshold_params` from the physical configuration, and
:func:`assert_labelers_consistent` checks a field through both.

Classes
-------
0
    Off / below threshold -- mean ``|E|**2`` below the physical CW floor.
1
    CW -- flat field, low contrast.
2
    Modulation instability -- periodic structure, moderate contrast.
3
    Chaotic -- high contrast, high spectral entropy.
4
    Multi-soliton -- high contrast, low entropy, more than one peak.
5
    Soliton crystal -- as class 4, but with evenly spaced peaks.
6
    Single soliton -- high contrast, low entropy, sech**2 comb.

Notes
-----
Fields are stored as physical energies: ``|E|**2`` is in joules, with
``mean|E|**2`` of order 1e-11 to 1e-9 J across a detuning sweep against an
empty-cavity numerical floor around 1e-16 J. Every power threshold here is
therefore derived from the configuration rather than hard-coded; see
:func:`physical_off_floor`.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Physically-scaled OFF threshold
# ---------------------------------------------------------------------------
# Intracavity fields are stored as physical energies: |E|² is in Joules, with
# mean|E|² ~ 1e-11 … 1e-9 J across a detuning sweep and an empty-cavity numerical
# floor ~1e-16 J. A field is "OFF" when essentially no coherent CW power has
# built up. Rather than a magic constant, we tie the floor to the smallest CW
# energy the cavity can support over the sweep:
#
#     U_cw(δω) = κ_c · pin / ((κ/2)² + δω²)          [Joules]
#
# U_cw is monotonically decreasing in |δω|, so its minimum over the sweep is at
# the largest |δω|:
#
#     U_cw,min = κ_c · pin / ((κ/2)² + δω_max²)
#
# OFF is then declared when mean(|E|²) < f · U_cw,min, i.e. the field sits a
# factor f below even the dimmest CW state. f ≈ 1e-3 … 1e-2 leaves a wide margin
# above the ~1e-16 J empty-cavity floor while staying far below any real CW field.

def physical_off_floor(
    kappa: float,
    kappa_c: float,
    pin: float,
    delta_omega_max: float,
    off_fraction: float = 1e-3,
) -> float:
    """Return the OFF energy floor: the dimmest CW state, scaled down.

    Parameters
    ----------
    kappa : float
        Total cavity loss rate ``kappa`` [rad/s].
    kappa_c : float
        Coupling rate ``kappa_c`` [rad/s].
    pin : float
        Pump power [W].
    delta_omega_max : float
        Largest ``|delta_omega|`` reached in the detuning sweep [rad/s], where
        ``delta_omega = omega_res - omega_pump``.
    off_fraction : float, optional
        Fraction ``f`` in (0, 1) of ``U_cw,min`` below which a field counts as
        OFF (default 1e-3). Dimensionless.

    Returns
    -------
    float
        The OFF power floor [J], i.e. ``off_fraction * U_cw,min``.

    Raises
    ------
    ZeroDivisionError
        If ``kappa`` and ``delta_omega_max`` are both zero, which would make the
        CW denominator vanish. Any physical configuration has ``kappa > 0``.

    Notes
    -----
    The homogeneous (CW) intracavity energy of the LLE is

        U_cw(delta_omega) = kappa_c * pin / ((kappa/2)**2 + delta_omega**2)  [J]

    which decreases monotonically in ``|delta_omega|``, so its minimum over a
    sweep is reached at the largest ``|delta_omega|``:

        U_cw,min = kappa_c * pin / ((kappa/2)**2 + delta_omega_max**2).

    A field is declared OFF when ``mean(|E|**2) < off_fraction * U_cw,min``,
    i.e. when it sits a factor ``f`` below even the dimmest CW state the cavity
    can support. Tying the floor to the physics rather than to a magic constant
    is what keeps the labeler correct when the pump power or the sweep range
    changes: ``f`` of 1e-3 to 1e-2 leaves a wide margin above the ~1e-16 J
    empty-cavity numerical floor while staying far below any real CW field.

    Examples
    --------
    >>> print(f"{physical_off_floor(1e9, 5e8, 0.1, 5e9):.4e} J")
    1.9802e-15 J

    Pushing the sweep further off resonance lowers the dimmest CW state, and
    with it the floor:

    >>> bool(physical_off_floor(1e9, 5e8, 0.1, 1e10)
    ...      < physical_off_floor(1e9, 5e8, 0.1, 5e9))
    True
    """
    k = float(kappa)
    u_cw_min = float(kappa_c) * float(pin) / ((k / 2.0) ** 2 + float(delta_omega_max) ** 2)
    return float(off_fraction) * u_cw_min


def make_threshold_params(
    kappa: float,
    kappa_c: float,
    pin: float,
    delta_omega_max: float,
    *,
    off_fraction: float = 1e-3,
    vacuum_floor_level: float = 0.0,
    envelope_smooth_modes: int = 1,
    vacuum_off_floor: float = 0.0,
    overrides: dict | None = None,
) -> dict:
    """Build the shared threshold dict from physical config -- single source of truth.

    Parameters
    ----------
    kappa : float
        Total cavity loss rate [rad/s].
    kappa_c : float
        Coupling rate [rad/s].
    pin : float
        Pump power [W].
    delta_omega_max : float
        Largest ``|delta_omega|`` in the sweep [rad/s].
    off_fraction : float, optional
        Fraction of ``U_cw,min`` defining the OFF floor (default 1e-3),
        dimensionless. Keyword-only.
    vacuum_floor_level : float, optional
        Absolute clip applied to the (smoothed) spectral envelope before the
        single-DKS monotonicity gate, in raw ``|FFT(E)|**2`` units -- i.e.
        ``n_tau**2 * hbar*omega0/2`` times a margin. Default 0.0, which clips
        nothing and reproduces the exact historical arithmetic. Keyword-only.
    envelope_smooth_modes : int, optional
        Circular moving-average width [modes] for that same envelope,
        odd-adjusted; 1 (default) is the identity. Keyword-only.
    vacuum_off_floor : float, optional
        Lower bound [J] on the OFF power floor -- ``n_tau * hbar*omega0/2``
        times a margin. Default 0.0, leaving the legacy floor unchanged.
        Keyword-only.
    overrides : dict or None, optional
        Applied LAST, over everything above. Escape hatch for tests and for
        studies that need a non-physical threshold. Keyword-only.

    Returns
    -------
    dict
        A copy of ``_DEFAULT_THRESHOLD_PARAMS`` with ``power_floor`` [J],
        ``vacuum_floor_level`` and ``envelope_smooth_modes`` replaced, then
        ``overrides`` applied.

    Raises
    ------
    TypeError
        If ``envelope_smooth_modes`` cannot be coerced to ``int``.
    ZeroDivisionError
        Propagated from :func:`physical_off_floor` for a degenerate cavity.

    Notes
    -----
    The OFF floor is derived from ``(kappa, kappa_c, pin, delta_omega_max)``
    through :func:`physical_off_floor`; all other (geometric and spectral)
    thresholds come from ``_DEFAULT_THRESHOLD_PARAMS``. The SAME dict feeds both
    the JAX labeler (:func:`make_state_labeler`) and the NumPy labeler
    (:func:`label_soliton_state`), which is what keeps the two consistent.

    The three quantum-vacuum-floor parameters are inactive by default. The
    solver computes them from ``hbar*omega0``, ``n_tau`` and the config margins
    ONLY when the quantum-noise channel is enabled -- the labeler never derives
    physics itself. Smoothing over ``w`` modes reduces the exponential-statistics
    wing fluctuation of a single snapshot from ~5.6 dB/mode to ~5.6/sqrt(w) dB,
    and ``vacuum_off_floor`` stops a pure vacuum-filled cavity (``pin = 0`` with
    the Langevin drive on) from being promoted out of OFF by its own half-photon
    background.

    Examples
    --------
    >>> params = make_threshold_params(1e9, 5e8, 0.1, 5e9)
    >>> print(f"{params['power_floor']:.4e} J")
    1.9802e-15 J
    >>> params["contrast_cw"], params["envelope_smooth_modes"]
    (2.0, 1)

    ``vacuum_off_floor`` can only RAISE the floor, never lower it:

    >>> lifted = make_threshold_params(1e9, 5e8, 0.1, 5e9, vacuum_off_floor=1e-12)
    >>> print(f"{lifted['power_floor']:.4e} J")
    1.0000e-12 J

    ``overrides`` wins over everything:

    >>> make_threshold_params(1e9, 5e8, 0.1, 5e9,
    ...                       overrides={"contrast_cw": 3.0})["contrast_cw"]
    3.0
    """
    params = dict(_DEFAULT_THRESHOLD_PARAMS)
    params["power_floor"] = max(
        physical_off_floor(kappa, kappa_c, pin, delta_omega_max, off_fraction),
        float(vacuum_off_floor),
    )
    params["vacuum_floor_level"] = float(vacuum_floor_level)
    params["envelope_smooth_modes"] = int(envelope_smooth_modes)
    if overrides:
        params.update(overrides)
    return params


def make_state_labeler(threshold_params: dict | None = None):
    """Return a JAX-traceable 7-class state labeler for use inside ``jax.lax.scan``.

    Parameters
    ----------
    threshold_params : dict or None, optional
        Thresholds, normally the dict produced by
        :func:`make_threshold_params`. Any subset of
        ``_DEFAULT_THRESHOLD_PARAMS`` is accepted and merged over the defaults;
        ``None`` (default) uses the defaults unchanged. Units follow the keys:
        ``power_floor`` [J], ``vacuum_floor_level`` [raw ``|FFT(E)|**2``],
        ``comb_structure_min_db`` [dB], ``envelope_smooth_modes`` [modes], the
        rest dimensionless.

    Returns
    -------
    callable
        ``state_labeler(e_t)`` mapping an ``(n_tau,)`` complex field [sqrt(J)]
        to a scalar ``jnp.int32`` class in 0--6.

    Raises
    ------
    KeyError
        If a required threshold key is missing from the merged dict -- possible
        only if ``_DEFAULT_THRESHOLD_PARAMS`` and this function drift apart.

    Notes
    -----
    Every threshold is baked into a Python-float constant at BUILD time, so the
    returned closure contains no Python branching on traced values and is fully
    ``jax.lax.scan``-traceable. The two vacuum-floor knobs are likewise resolved
    at build time, so at their inactive defaults (``0.0`` and ``1``) the traced
    graph is EXACTLY the historical arithmetic rather than a no-op clip and a
    width-1 convolution.

    Passing the same dict to this function and to :func:`label_soliton_state` is
    what makes the two labelers comparable; a disagreement is then a genuine
    reduction or mechanism drift rather than a config mismatch.

    Classes
    -------
    0
        Off / below threshold -- ``mean|E|**2`` below the physical CW floor.
    1
        CW -- flat field, low contrast.
    2
        Modulation instability -- periodic structure, moderate contrast.
    3
        Chaotic -- high contrast, high spectral entropy.
    4
        Multi-soliton -- high contrast, low entropy, more than one peak.
    5
        Soliton crystal -- high contrast, low entropy, evenly spaced peaks.
    6
        Single soliton -- high contrast, low entropy, sech**2 comb.

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> labeler = make_state_labeler()
    >>> n = 1024
    >>> tau = np.arange(n) - n // 2
    >>> soliton = (1e-4 / np.cosh(tau / 16.0)).astype(np.complex128)
    >>> int(labeler(jnp.asarray(soliton, dtype=jnp.complex64)))
    6
    >>> int(labeler(jnp.full(n, 1e-5 + 0j, dtype=jnp.complex64)))
    1
    >>> int(labeler(jnp.full(n, 1e-9 + 0j, dtype=jnp.complex64)))
    0
    """

    # --- bake all thresholds into float constants (no traced Python branching) ---
    _params = {**_DEFAULT_THRESHOLD_PARAMS, **(threshold_params or {})}
    POWER_FLOOR        = float(_params["power_floor"])
    CONTRAST_CW        = float(_params["contrast_cw"])
    CONTRAST_HIGH      = float(_params["contrast_high"])
    ENTROPY_CHAOTIC    = float(_params["entropy_chaotic"])
    CRYSTAL_CV         = float(_params["crystal_cv"])
    PEAK_AMP_THRESHOLD = float(_params["peak_prominence"])  # fraction of p_max
    SECH2_ENV_MONO_MIN = float(_params["sech2_env_mono_min"])  # single-DKS envelope test
    SECH2_ENV_TOL      = float(_params["sech2_env_tol"])       # per-step log tolerance
    COMB_MIN_DB        = float(_params["comb_structure_min_db"])  # comb-vs-flat-floor gate
    # Quantum-vacuum-floor robustness of the envelope gate (see the docstring
    # of make_threshold_params). Both are build-time STATIC no-ops at their
    # inactive defaults (0.0 / 1): the branches below are Python-level, so the
    # inactive path traces EXACTLY the historical arithmetic.
    VACUUM_FLOOR_LEVEL = float(_params["vacuum_floor_level"])  # raw |FFT(E)|² units
    _w = int(_params["envelope_smooth_modes"])
    ENV_SMOOTH_W       = 2 * (_w // 2) + 1 if _w > 1 else 1    # odd-adjusted width

    def state_labeler(e_t: jnp.ndarray) -> jnp.int32:
        n_tau = e_t.shape[0]
        p = jnp.abs(e_t) ** 2

        # --- spectral features ---
        spec = jnp.abs(jnp.fft.fft(e_t)) ** 2
        spec_norm = spec / jnp.maximum(jnp.sum(spec), 1e-20)
        # spectral entropy: low = ordered comb, high = chaotic
        entropy = -jnp.sum(spec_norm * jnp.log(jnp.maximum(spec_norm, 1e-20)))
        entropy_max = jnp.log(jnp.array(e_t.shape[0], dtype=jnp.float32))
        norm_entropy = entropy / entropy_max   # in [0, 1]

        # --- temporal features ---
        p_mean = jnp.mean(p)
        p_max  = jnp.max(p)
        contrast = p_max / jnp.maximum(p_mean, 1e-20)

        # number of peaks: count points above 50% of max with positive->negative
        # zero-crossings of the gradient (proxy for peak count, JAX-traceable)
        # Only count peaks whose amplitude exceeds PEAK_AMP_THRESHOLD·p_max (sourced
        # from params["peak_prominence"], identical to the NumPy labeler's prominence).
        # Without this threshold, dispersive-wave tails and FFT ringing on the 512-point
        # grid produce O(10–50) spurious peaks in single-soliton states, making
        # sign_changes >> 1 and systematically mislabeling single solitons as multi-soliton.
        grad = jnp.diff(p, append=p[:1])          # circular gradient, length n_tau
        _is_local_max = (grad > 0) & (jnp.roll(grad, -1) <= 0)
        peak_mask = _is_local_max & (p > PEAK_AMP_THRESHOLD * p_max)
        sign_changes = jnp.sum(peak_mask).astype(jnp.float32)

        # Smooth-sech²-envelope test (single-DKS vs chaos), JAX-traceable analogue
        # of the NumPy path's temporal sech² goodness-of-fit.
        #
        # A single dissipative Kerr soliton is a strong pump (DC) line plus a comb of
        # sidebands whose power envelope is sech² — i.e. it decreases MONOTONICALLY
        # outward from the pump on BOTH sides. Chaos/MI spectra are jagged and
        # non-monotonic. We measure the fraction of outward mode-steps whose (log)
        # power does not increase (within SECH2_ENV_TOL). This REPLACES the old
        # top-N-points "sharpness" proxy, which silently mislabels a genuine single
        # DKS as chaotic: the soliton sits on a bright CW background that carries most
        # of the total energy, so the fraction of power in the top ~32 points is only
        # ~0.2 (< the 0.75 sharpness gate), and the state fell through to class 3.
        # The envelope monotonicity is ~1.0 for a single DKS at any resolution
        # (the comb may be broad, but it is smooth), and ~0.5–0.75 for MI/chaos.
        spec_shift = jnp.fft.fftshift(spec)
        # Vacuum-floor-robust envelope for the monotonicity gate ONLY (entropy,
        # contrast and the inner/outer band ratio keep the raw spectrum).
        # Single-snapshot modal power of a vacuum-filled mode is exponentially
        # distributed => log10-power std = (pi/sqrt(6))/ln 10 ~ 0.56 decades,
        # which the raw-envelope step test reads as non-monotonic structure.
        # (i) circular moving average over ENV_SMOOTH_W modes (linear power,
        # fftshifted ordering; the wrap joins the +/-Nyquist edges) reduces the
        # fluctuation to ~5.6/sqrt(w) dB; (ii) absolute clip at
        # VACUUM_FLOOR_LEVEL (raw units, BEFORE peak normalization, so the
        # floor is state-independent): modes at the clip form an exactly flat
        # plateau whose steps are 0 <= tol — "envelope terminated", trivially
        # monotone, never new structure. Both branches are Python-static; at
        # the inactive defaults (w=1, level=0.0) the traced arithmetic is
        # bit-for-bit the historical one.
        env_lin = spec_shift
        if ENV_SMOOTH_W > 1:
            _h = ENV_SMOOTH_W // 2
            env_lin = sum(
                jnp.roll(env_lin, k) for k in range(-_h, _h + 1)
            ) / float(ENV_SMOOTH_W)
        if VACUUM_FLOOR_LEVEL > 0.0:
            env_lin = jnp.maximum(env_lin, VACUUM_FLOOR_LEVEL)
        log_env = jnp.log10(
            jnp.maximum(env_lin / jnp.maximum(jnp.max(env_lin), 1e-30), 1e-12)
        )
        c_idx = n_tau // 2                                   # DC / pump line index
        right_steps = log_env[c_idx + 1:] - log_env[c_idx:-1]   # outward, i > center
        left_steps = log_env[:c_idx] - log_env[1:c_idx + 1]     # outward, i < center
        mono_frac = 0.5 * (
            jnp.mean((right_steps <= SECH2_ENV_TOL).astype(jnp.float32))
            + jnp.mean((left_steps <= SECH2_ENV_TOL).astype(jnp.float32))
        )

        # Comb-structure ("central bulge") test: a real soliton comb concentrates
        # sideband power NEAR the pump (inner half-band) and decays outward, so the
        # inner-band mean is well above the outer-band mean. A CW field carrying a
        # single-sample numerical spike has a FLAT sideband floor (the spike spreads
        # equally over all modes), giving inner ≈ outer ≈ 0 dB. mono_frac alone
        # cannot separate these — a flat floor is trivially "monotonic" — so we also
        # require a minimum inner/outer sideband ratio. This is the JAX analogue of
        # "the spectrum is a comb, not a pump line on a flat floor".
        q_idx = n_tau // 4
        inner_band = jnp.concatenate(
            [spec_shift[c_idx + 1:c_idx + q_idx], spec_shift[c_idx - q_idx + 1:c_idx]]
        )
        outer_band = jnp.concatenate(
            [spec_shift[c_idx + q_idx:], spec_shift[:c_idx - q_idx + 1]]
        )
        inner_outer_db = 10.0 * jnp.log10(
            jnp.mean(inner_band) / jnp.maximum(jnp.mean(outer_band), 1e-30)
        )


        # --- decision tree (all jnp.where for JAX traceability) ---
        # Physical fields are in Joules (mean|E|² ~ 1e-11–1e-9). Use mean(|E|²)
        # (matches the NumPy labeler) compared against the physically-scaled OFF
        # floor (POWER_FLOOR = f·U_cw,min, baked from config at build time).
        is_off     = p_mean < POWER_FLOOR
        is_cw      = contrast < CONTRAST_CW
        is_mi      = (contrast >= CONTRAST_CW) & (contrast < CONTRAST_HIGH)
        is_chaotic = (contrast >= CONTRAST_HIGH) & (norm_entropy > ENTROPY_CHAOTIC)


        # ---- crystal detection: peak spacing coefficient of variation ----
        # Extract peak positions as a sorted array of length n_tau,
        # with non-peak slots filled by n_tau (a sentinel beyond all valid indices).
        # After sorting, the first sign_changes entries are the true peak positions.
        sentinel = jnp.float32(n_tau)
        peak_locs = jnp.where(peak_mask, jnp.arange(n_tau, dtype=jnp.float32), sentinel)
        peak_locs_sorted = jnp.sort(peak_locs)          # real peaks first, sentinels at end
        
        # Spacings between consecutive real peaks.
        # diff of sorted locs: entry i = peak_locs_sorted[i] - peak_locs_sorted[i-1]
        # The last entry wraps to peak_locs_sorted[0]+n_tau-peak_locs_sorted[-1] (circular),
        # but we only use the first (sign_changes - 1) entries, so the wrap-around
        # and sentinel-to-sentinel diffs don't matter if we mask them.
        locs_shifted = jnp.roll(peak_locs_sorted, 1)
        raw_spacings = peak_locs_sorted - locs_shifted   # (n_tau,); first entry is garbage
        
        # Valid entries: indices 1 .. sign_changes-1 (between real peaks)
        # Build a validity mask: entry i is valid if i >= 1 and i < sign_changes
        valid_idx = jnp.arange(n_tau, dtype=jnp.float32)
        spacing_valid = (valid_idx >= 1.0) & (valid_idx < sign_changes)
        
        n_valid = jnp.maximum(sign_changes - 1.0, 1.0)
        sp_mean = jnp.sum(jnp.where(spacing_valid, raw_spacings, 0.0)) / n_valid
        sp_sq   = jnp.sum(jnp.where(spacing_valid, (raw_spacings - sp_mean)**2, 0.0)) / n_valid
        spacing_cv = jnp.sqrt(sp_sq) / jnp.maximum(sp_mean, 1.0)
        
        is_crystal = (
            (contrast >= CONTRAST_HIGH) & (norm_entropy <= ENTROPY_CHAOTIC)
            & (sign_changes > 2.5)                       # ← require ≥ 3 peaks, not ≥ 2
            & (spacing_cv < CRYSTAL_CV)
        )
        is_multi = (
            (contrast >= CONTRAST_HIGH) & (norm_entropy <= ENTROPY_CHAOTIC)
            & (sign_changes > 1.5)
            & ~is_crystal                                 # anything multi that isn't crystal
        )

        # CW-dominated with a single-sample numerical spike: high contrast and ONE
        # peak, but the sideband spectrum is FLAT (no comb — inner ≈ outer). This is
        # physically a CW state whose lone hot sample fakes a high peak-to-mean; it
        # must NOT be read as a soliton. Route it to CW (1). (A genuine soliton has a
        # comb: inner_outer_db well above COMB_MIN_DB.)
        is_flat_spike = (
            (contrast >= CONTRAST_HIGH)
            & (sign_changes <= 1.5)
            & (inner_outer_db < COMB_MIN_DB)
        )

        # single soliton: high contrast, ordered spectrum, ONE temporal peak, a smooth
        # (monotonic) sech² spectral envelope, AND real comb structure (sidebands
        # concentrated near the pump, not a flat floor). Keying on single-peak +
        # smooth sech² comb (the features the NumPy sech²-fit path uses) makes this
        # robust to a soliton on a bright CW background, while chaos (jagged, low
        # mono_frac), multi/MI (multiple peaks), and CW+spike (flat, is_flat_spike)
        # are all excluded.
        is_single = (
            (contrast >= CONTRAST_HIGH) & (norm_entropy <= ENTROPY_CHAOTIC)
            & (sign_changes <= 1.5)
            & (mono_frac >= SECH2_ENV_MONO_MIN)
            & (inner_outer_db >= COMB_MIN_DB)
        )

        label = jnp.where(is_off,        0,
                jnp.where(is_cw,         1,
                jnp.where(is_mi,         2,
                jnp.where(is_chaotic,    3,
                jnp.where(is_flat_spike, 1,   # CW + single-sample spike -> CW
                jnp.where(is_multi,      4,
                jnp.where(is_crystal,    5,
                jnp.where(is_single,     6,
                                         3))))))))

        return label.astype(jnp.int32)

    return state_labeler

_DEFAULT_THRESHOLD_PARAMS: dict = {
    # power_floor is a conservative fallback only. The canonical OFF floor is
    # derived from physical config via make_threshold_params() / physical_off_floor();
    # callers generating real (Joule-scale) data should ALWAYS pass that floor in.
    "power_floor": 1e-13,
    "contrast_cw": 2.0,
    "contrast_high": 8.0,
    "entropy_chaotic": 0.5,
    "crystal_cv": 0.1,
    "sech2_r2": 0.95,        # NumPy single-soliton sech² goodness-of-fit (class 6)
    # JAX single-soliton test: fraction of outward spectral-envelope steps that are
    # monotonically non-increasing (within sech2_env_tol, in log10 units). A single
    # DKS comb is ~1.0; MI/chaos are ~0.5-0.75. Replaces the old "sharpness_min"
    # top-N-power-fraction proxy, which mislabeled a DKS on a bright CW background.
    "sech2_env_mono_min": 0.9,
    "sech2_env_tol": 0.05,
    # Minimum inner/outer sideband power ratio (dB) for a real comb. A single DKS is
    # >= ~2.5 dB (sidebands bunch near the pump); a CW field with a single-sample
    # numerical spike has a FLAT sideband floor (~0 dB) and is routed to CW. Used by
    # both is_single (require a comb) and is_flat_spike (flat floor -> CW).
    "comb_structure_min_db": 1.5,
    "sharpness_min": 0.75,   # DEPRECATED: no longer used by the JAX labeler
    "peak_prominence": 0.3,
    "peak_width": 2.0,
    # Quantum-vacuum-floor robustness (see make_threshold_params). Inactive by
    # default: 0.0 clips nothing and width 1 is the identity, so the JAX
    # envelope gate traces the exact historical arithmetic. Set by the solver
    # (physics-anchored: n_tau²·ħω₀/2 × margin, and an odd-adjusted smoothing
    # width) only when the quantum noise channel is enabled. The NumPy labeler
    # carries the keys for parity but has no spectral monotonicity gate — its
    # single-DKS discriminator is the TEMPORAL sech² fit, which is robust to
    # the vacuum floor (the floor perturbs |E(τ)|² at the ~1e-5 relative
    # level); the shared OFF floor lift enters via power_floor for both paths.
    "vacuum_floor_level": 0.0,
    "envelope_smooth_modes": 1,
}

def label_soliton_state(E_tau, threshold_params) -> int:
    """Label one intracavity-field snapshot with the 7-class soliton scheme (NumPy path).

    Parameters
    ----------
    E_tau : numpy.ndarray
        Intracavity field E(tau), shape ``(n_tau,)``, complex, units sqrt(J);
        ``|E|**2`` is in joules.
    threshold_params : dict or None
        Thresholds, normally from :func:`make_threshold_params`. Merged over
        ``_DEFAULT_THRESHOLD_PARAMS``, so a partial dict (or ``None``) is
        accepted.

    Returns
    -------
    int
        Class index 0--6; see :func:`make_state_labeler` for the class list.

    Raises
    ------
    IndexError
        If ``E_tau`` is not 1-D -- ``E_tau.shape[0]`` is read directly as
        ``n_tau``.
    ValueError
        Propagated from ``numpy.fft`` for an empty field.

    Notes
    -----
    The SciPy counterpart of :func:`make_state_labeler`: same thresholds, same
    class definitions, but free to use ``find_peaks`` and ``curve_fit`` because
    it never runs inside a traced scan.

    The decision order is power floor, then contrast, then -- for high-contrast
    fields -- spectral entropy, peak count and peak-spacing regularity. The
    single-soliton branch is guarded twice. First by comb structure: a CW field
    carrying a single-sample numerical spike also has high contrast and one
    peak, but its sideband floor is FLAT (inner/outer ratio near 0 dB) whereas a
    real comb bunches sideband power near the pump, so a flat-floored spike is
    routed to CW rather than read as a soliton. Then by a sech**2 fit of the
    temporal profile: a failed fit or a poor R**2 lands in chaotic, never in
    single-soliton.

    Examples
    --------
    >>> import numpy as np
    >>> n = 1024
    >>> tau = np.arange(n) - n // 2
    >>> label_soliton_state((1e-4 / np.cosh(tau / 16.0)).astype(complex), None)
    6
    >>> label_soliton_state(np.full(n, 1e-5 + 0j), None)
    1
    >>> label_soliton_state(np.full(n, 1e-9 + 0j), None)
    0
    """
    
    params = {**_DEFAULT_THRESHOLD_PARAMS, **(threshold_params or {})}

    p = np.abs(E_tau) ** 2
    p_mean = float(np.mean(p))
    if p_mean < params["power_floor"]:
        return 0

    p_max = float(np.max(p))
    contrast = p_max / p_mean
    if contrast < params["contrast_cw"]:
        return 1

    n_tau = E_tau.shape[0]
    spec = np.abs(np.fft.fft(E_tau)) ** 2
    spec_norm = spec / max(float(np.sum(spec)), 1e-20)
    entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-20))
    norm_entropy = float(entropy / np.log(n_tau))

    peaks, _ = find_peaks(
        p,
        prominence=params["peak_prominence"] * p_max,
        width=params["peak_width"],
    )
    n_peaks = int(peaks.size)

    # Comb-structure ("central bulge") metric: inner/outer sideband power ratio (dB).
    # A real soliton comb bunches sideband power near the pump (inner >> outer); a CW
    # field carrying a single-sample numerical spike has a FLAT sideband floor
    # (inner ≈ outer ≈ 0 dB). Mirrors the JAX labeler's inner_outer_db.
    spec_shift = np.fft.fftshift(spec)
    c_idx = n_tau // 2
    q_idx = n_tau // 4
    inner_band = np.concatenate(
        [spec_shift[c_idx + 1:c_idx + q_idx], spec_shift[c_idx - q_idx + 1:c_idx]]
    )
    outer_band = np.concatenate(
        [spec_shift[c_idx + q_idx:], spec_shift[:c_idx - q_idx + 1]]
    )
    inner_outer_db = 10.0 * np.log10(
        float(np.mean(inner_band)) / max(float(np.mean(outer_band)), 1e-300)
    )

    if contrast >= params["contrast_high"]:
        if norm_entropy > params["entropy_chaotic"]:
            return 3
        if n_peaks >= 3:
            spacings = np.diff(np.sort(peaks))
            spacing_cv = float(spacings.std() / max(float(spacings.mean()), 1.0))
            if spacing_cv < params["crystal_cv"]:
                return 5
            return 4
        if n_peaks == 2:
            return 4
        if n_peaks <= 1:
            # CW + single-sample numerical spike: high contrast, <=1 wide peak, but a
            # FLAT sideband spectrum (no comb). Physically a CW state; route to CW (1)
            # so it is never read as a soliton. A genuine soliton has real comb
            # structure (inner_outer_db well above the flat floor).
            if inner_outer_db < params["comb_structure_min_db"]:
                return 1

            x = np.arange(n_tau, dtype=float)

            def sech2_model(x_vals, A, x0, w, B):
                return A / np.cosh((x_vals - x0) / w) ** 2 + B

            p0 = [p_max, float(np.argmax(p)), n_tau / 20.0, float(np.min(p))]
            try:
                popt, _ = curve_fit(sech2_model, x, p, p0=p0, maxfev=10000)
            except Exception:
                return 3

            p_fit = sech2_model(x, *popt)
            ss_res = float(np.sum((p - p_fit) ** 2))
            ss_tot = float(np.sum((p - p_mean) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-20)
            if r2 >= params["sech2_r2"]:
                return 6
            return 3

    if contrast < params["contrast_high"] and contrast >= params["contrast_cw"]:
        return 2

    return 0


def label_trajectory(E_history, threshold_params=None) -> np.ndarray:
    """Label every snapshot in a trajectory with the 7-class soliton scheme.

    Parameters
    ----------
    E_history : numpy.ndarray
        Snapshot history, shape ``(n_snapshots, n_tau)``, complex, units
        sqrt(J).
    threshold_params : dict or None, optional
        Thresholds, normally from :func:`make_threshold_params`. Merged over
        the defaults ONCE here and passed down, so every snapshot in a
        trajectory is labeled against identical thresholds.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_snapshots,)``, dtype ``int32``, values in 0--6.

    Raises
    ------
    IndexError
        If ``E_history`` is not 2-D.
    ValueError
        Propagated from :func:`label_soliton_state` for a degenerate snapshot.

    Notes
    -----
    A plain Python loop over :func:`label_soliton_state`, not a vectorized
    reduction: the single-soliton branch runs a Levenberg--Marquardt sech**2 fit
    per snapshot, which has no array form. Labeling inside a solve uses the JAX
    labeler instead; this one is for stored trajectories.

    Examples
    --------
    >>> import numpy as np
    >>> n = 1024
    >>> tau = np.arange(n) - n // 2
    >>> history = np.stack([np.full(n, 1e-9 + 0j),
    ...                     np.full(n, 1e-5 + 0j),
    ...                     (1e-4 / np.cosh(tau / 16.0)).astype(complex)])
    >>> labels = label_trajectory(history)
    >>> print(labels, labels.dtype)
    [0 1 6] int32
    """
    params = {**_DEFAULT_THRESHOLD_PARAMS, **(threshold_params or {})}

    n_snapshots = E_history.shape[0]
    labels = np.zeros((n_snapshots,), dtype=np.int32)
    for i in range(n_snapshots):
        labels[i] = label_soliton_state(E_history[i], params)
    return labels


def sech2_envelope_correlation(e_field: np.ndarray) -> tuple[float, float, float]:
    """Fit a sech**2 envelope to the comb spectrum and report how well it matches.

    Parameters
    ----------
    e_field : numpy.ndarray
        Intracavity field E(tau), shape ``(n_tau,)``, complex, units sqrt(J).

    Returns
    -------
    pearson_corr : float
        Pearson correlation [dimensionless] between the measured log-spectrum
        and the fitted sech**2 envelope, over the sidebands only. NaN on fit
        failure.
    r2 : float
        Coefficient of determination [dimensionless] of the same fit. NaN on
        fit failure.
    fitted_mode_width : float
        Fitted envelope half-width [modes]. NaN on fit failure.

    Raises
    ------
    IndexError
        If ``e_field`` is not 1-D. Fit failures do NOT raise -- they return
        three NaNs, because this runs over long trajectories where one
        non-convergent snapshot must not abort the sweep.

    Notes
    -----
    The quantitative, fit-based counterpart of the single-soliton discriminator
    the labelers use. A dissipative Kerr soliton spectrum is a strong pump (DC)
    line plus a sech**2 comb of sidebands, since ``|FT{sech}|**2 = sech**2``. A
    width-matched sech**2 (plus a constant floor) is fitted to the fftshifted
    sideband envelope in log space -- where the comb spans many decades -- with
    the pump line itself EXCLUDED, because it is not part of the envelope.

    A single dissipative Kerr soliton scores above 0.99; modulation-instability
    and chaotic combs score near zero or negative.

    This lives in the simulator layer alongside the labeler so that ``analysis``
    code can import it from here: nothing in ``simulator`` may import from
    ``analysis``.

    Examples
    --------
    >>> import numpy as np
    >>> n = 1024
    >>> tau = np.arange(n) - n // 2
    >>> corr, r2, width = sech2_envelope_correlation(
    ...     (1.0 / np.cosh(tau / 16.0)).astype(complex))
    >>> print(f"{corr:.4f} {r2:.4f} {width:.4f}")
    1.0000 0.9999 6.4458

    A modulation-instability comb, with its sidebands spread rather than
    bunched, scores far below that:

    >>> mi = 1e-5 * (1.0 + 0.5 * np.cos(2 * np.pi * 8 * np.arange(n) / n))
    >>> corr, _, _ = sech2_envelope_correlation(mi.astype(complex))
    >>> print(f"{corr:.4f}")
    0.2484
    """
    n = e_field.shape[0]
    spec = np.abs(np.fft.fftshift(np.fft.fft(e_field))) ** 2
    spec_n = spec / max(spec.max(), 1e-300)
    mu = np.arange(n) - n // 2
    y = np.log10(np.maximum(spec_n, 1e-12))

    mask = np.ones(n, dtype=bool)
    mask[n // 2] = False  # drop the pump (DC) line

    def model(m, log_a, mode_w, log_floor):
        return np.log10(10.0 ** log_a / np.cosh(m / mode_w) ** 2 + 10.0 ** log_floor)

    try:
        popt, _ = curve_fit(
            model, mu[mask], y[mask], p0=[0.0, 60.0, -4.0], maxfev=40000
        )
        fit = model(mu, *popt)
        corr = float(np.corrcoef(y[mask], fit[mask])[0, 1])
        ss_res = float(np.sum((y[mask] - fit[mask]) ** 2))
        ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        return corr, r2, abs(float(popt[1]))
    except Exception:
        return float("nan"), float("nan"), float("nan")


def assert_labelers_consistent(
    e_field: np.ndarray,
    atol: float = 0.0,
    threshold_params: dict | None = None,
) -> None:
    """Verify that the JAX and NumPy labelers agree on a test field.

    Parameters
    ----------
    e_field : numpy.ndarray
        Intracavity field E(tau), shape ``(n_tau,)``, complex, units sqrt(J).
    atol : float, optional
        Accepted for API stability and currently unused: the labels are
        integers, so the comparison is exact. Default 0.0.
    threshold_params : dict or None, optional
        Thresholds, merged over the defaults and passed to BOTH labelers.

    Returns
    -------
    None
        Returns normally iff the two labelers agree.

    Raises
    ------
    AssertionError
        If the labels differ, reporting both labels together with the field's
        peak power [J] and contrast [dimensionless] so the disagreeing regime is
        identifiable from the message alone.

    Notes
    -----
    Both labelers are driven by the SAME threshold dict, so any disagreement is
    a genuine reduction or mechanism drift rather than a config mismatch. The
    JAX side is evaluated in ``complex64``, matching how the labeler is called
    inside the solver's scan.

    Run this during dataset generation to catch labeler drift early: a
    trajectory labeled on-device and re-labeled offline must carry the same
    classes, or the dataset and its analysis disagree about what they contain.

    Examples
    --------
    >>> import numpy as np
    >>> n = 1024
    >>> tau = np.arange(n) - n // 2
    >>> assert_labelers_consistent((1e-4 / np.cosh(tau / 16.0)).astype(complex))
    >>> assert_labelers_consistent(np.full(n, 1e-5 + 0j))
    """
    params = {**_DEFAULT_THRESHOLD_PARAMS, **(threshold_params or {})}
    jax_labeler = make_state_labeler(params)
    jax_label = int(jax_labeler(jnp.array(e_field, dtype=jnp.complex64)))
    np_label = int(label_soliton_state(e_field, threshold_params=params))
    assert jax_label == np_label, (
        f"Labeler inconsistency: JAX={jax_label}, NumPy={np_label} for field with "
        f"max_power={float(np.max(np.abs(e_field)**2)):.3e}, "
        f"contrast={float(np.max(np.abs(e_field)**2)/np.mean(np.abs(e_field)**2)):.1f}"
    )
