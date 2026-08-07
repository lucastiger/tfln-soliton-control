"""Colored-noise engine + TRN spectral-model tests (categories 1-6).

Covers the Q3 test plan:
  (1) engine variance = integral of S df within 2%;
  (2) Welch-vs-target within 3 dB per octave for white, 1/f, single-pole and
      Kondratiev-Gorodetsky shapes;
  (3) K-G total variance = the Eq. 129 thermodynamic value after
      renormalization (analytic vs numeric integral AND sampled variance,
      2%)  -> STOP-AND-REPORT GATE 1;
  (4) trn_psd_model = single_pole bit-identical to the pre-change AR(1)
      machinery for a fixed key (inline legacy reimplementation, plus the
      solver-level _detuning_noise_sequences surface);
  (5) CSV round-trip (write a PSD, read it back through the csv model,
      regenerate, compare), both unit conventions;
  (6) segment continuity: the full-length-then-slice path has no boundary
      variance dip (boundary/interior mean-square ratio in [0.9, 1.1]) while
      legacy_segment_noise keeps the old per-segment restart stats
      -> STOP-AND-REPORT GATE 2.

Categories 7-9 (FSR exactness, probe exactness, synthetic-comb metrology)
live in tests/test_noise_metrology.py; category 10 is the full suite.
"""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from simulator.colored_noise import (
    csv_psd,
    integrate_psd,
    kondratiev_gorodetsky_psd,
    np_generator_from_key,
    single_pole_psd,
    synthesize_from_psd,
)
from simulator.noise_config import NoiseConfig
from simulator.noise_models import (
    PyroEONoise,
    TCCRNoise,
    TotalNoise,
    TRNoise,
    _ar1_samples,
    _load_config as nm_load_cfg,
)

K_B = 1.380649e-23


def _welch(x, f_s, nperseg=4096):
    from scipy.signal import welch

    f, s = welch(np.asarray(x), fs=f_s, nperseg=nperseg)
    return f[1:], s[1:]


def _octave_band_ratio_db(f, s_meas, psd_target, f_lo, f_hi):
    """Max |band-averaged measured/target| in dB over octave bands."""
    worst = 0.0
    rows = []
    lo = f_lo
    while lo * 2.0 <= f_hi:
        hi = lo * 2.0
        m = (f >= lo) & (f < hi)
        if m.sum() >= 2:
            meas = float(np.mean(s_meas[m]))
            tgt = float(np.mean(np.asarray(psd_target(f[m]))))
            if tgt > 0:
                db = abs(10.0 * math.log10(meas / tgt))
                rows.append((lo, hi, db))
                worst = max(worst, db)
        lo = hi
    assert rows, "no octave bands resolved — bad test configuration"
    return worst, rows


# ---------------------------------------------------------------------------
# (1) Engine variance = integral of the PSD
# ---------------------------------------------------------------------------
def _expected_mean_square(psd, n, f_s):
    """E[mean(x^2)] of the synthesis: Sum_k S(f_k) df - (df/2)(S_DC + S_Nyq).

    The interior rfft bins represent BOTH +-k modes (weight 2 in Parseval)
    while DC and Nyquist appear once — so relative to the naive one-sided sum
    the two edge bins carry half weight. This is exactly what the engine
    injects; it converges to integral_0^{f_s/2} S df as n grows (the edge
    correction is O(1/n) for resolved spectra).
    """
    f = np.fft.rfftfreq(n, d=1.0 / f_s)
    s = np.asarray(psd(f), dtype=np.float64)
    s[0] = s[1]                                   # engine DC clamp
    df = f_s / n
    return float(np.sum(s) * df - 0.5 * df * (s[0] + s[-1]))


def test_engine_variance_matches_psd_integral():
    f_s = 1.0e6
    n = 1 << 16
    # Realization counts sized so the chi^2 estimator noise of the ensemble
    # mean is well below the 2% band (the 1/f case concentrates ~13% of its
    # power in the k = 1 bin, so it needs the deepest ensemble); the fixed
    # keys make the measured values deterministic.
    cases = {
        "white": (lambda f: np.full_like(np.asarray(f, float), 3.7e-4), 24),
        "single_pole": (single_pole_psd(2.5, 2.0e-5), 100),
        "one_over_f": (lambda f: 1.0e-2 / np.maximum(np.asarray(f, float),
                                                     f_s / n), 400),
    }
    for name, (psd, reps) in cases.items():
        ms = np.mean([
            np.mean(synthesize_from_psd(
                np_generator_from_key(jax.random.PRNGKey(100 + i)), n, psd, f_s
            ) ** 2)
            for i in range(reps)
        ])
        target = _expected_mean_square(psd, n, f_s)
        rel = abs(ms - target) / target
        assert rel < 0.02, (name, ms, target, rel)


def test_engine_dc_clamp_and_determinism():
    f_s, n = 1.0e6, 4096
    psd = lambda f: 1.0 / np.maximum(np.asarray(f, float), 1e-12)  # noqa: E731
    key = jax.random.PRNGKey(7)
    x1 = synthesize_from_psd(np_generator_from_key(key), n, psd, f_s)
    x2 = synthesize_from_psd(np_generator_from_key(key), n, psd, f_s)
    assert np.array_equal(x1, x2), "same key must give bit-identical noise"
    assert np.isfinite(x1).all(), "DC clamp must keep 1/f finite"
    x3 = synthesize_from_psd(
        np_generator_from_key(jax.random.PRNGKey(8)), n, psd, f_s
    )
    assert not np.array_equal(x1, x3), "distinct keys must differ"


# ---------------------------------------------------------------------------
# (2) Welch PSD fidelity per octave: white, 1/f, single-pole, K-G
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["white", "one_over_f", "single_pole", "kg"])
def test_welch_matches_target_within_3db_per_octave(name):
    f_s = 2.46e10                      # the physical per-round-trip rate
    n = 1 << 18
    f1 = f_s / n
    if name == "white":
        psd = lambda f: np.full_like(np.asarray(f, float), 1.0e-9)  # noqa: E731
    elif name == "one_over_f":
        psd = lambda f: 1.0 / np.maximum(np.asarray(f, float), f1)  # noqa: E731
    elif name == "single_pole":
        psd = single_pole_psd(4.0e2, 5.0e-6)
    else:
        cfg = nm_load_cfg(None)
        psd, _ = kondratiev_gorodetsky_psd(
            T_k=300.0, kappa_th=float(cfg["kappa_th_w_per_m_k"]),
            rho=float(cfg["rho_kg_per_m3"]), cp=float(cfg["Cp_j_per_kg_k"]),
            R=100e-6, d_a=2.0e-6, d_b=8.0e-7,
            mode_volume=float(cfg["mode_volume_m3"]), f_max=f_s / 2.0,
        )
    xs = np.stack([
        synthesize_from_psd(
            np_generator_from_key(jax.random.PRNGKey(300 + i)), n, psd, f_s
        )
        for i in range(6)
    ])
    f, s = _welch(xs[0], f_s, nperseg=1 << 13)
    s = np.mean(np.stack([_welch(x, f_s, nperseg=1 << 13)[1] for x in xs]),
                axis=0)
    # Resolved octaves: from a few Welch bins up to Nyquist/2 (skip the very
    # last octave where the taper bias grows).
    worst, rows = _octave_band_ratio_db(f, s, psd, f[4], f_s / 4.0)
    assert worst < 3.0, (name, worst, rows[-3:])


# ---------------------------------------------------------------------------
# (3) K-G variance pinned to Eq. 129  -> GATE 1
# ---------------------------------------------------------------------------
def test_kg_variance_renormalized_to_eq129_gate1():
    cfg = nm_load_cfg(None)
    rho = float(cfg["rho_kg_per_m3"])
    cp = float(cfg["Cp_j_per_kg_k"])
    v = float(cfg["mode_volume_m3"])
    t_k = 300.0
    f_s = 2.46e10
    psd, var_129 = kondratiev_gorodetsky_psd(
        T_k=t_k, kappa_th=float(cfg["kappa_th_w_per_m_k"]), rho=rho, cp=cp,
        R=100e-6, d_a=2.0e-6, d_b=8.0e-7, mode_volume=v, f_max=f_s / 2.0,
    )
    var_analytic = K_B * t_k**2 / (rho * cp * v)
    assert var_129 == pytest.approx(var_analytic, rel=1e-12)

    # Numeric integral of the RENORMALIZED PSD over the synthesis band.
    var_numeric = integrate_psd(psd, f_lo=1.0, f_hi=f_s / 2.0)
    rel_int = abs(var_numeric - var_analytic) / var_analytic
    assert rel_int < 0.02, (var_numeric, var_analytic)

    # Sampled mean-square of the synthesized sequences vs the exact injected
    # target (edge-corrected discrete sum; the f^-1/2 low end concentrates
    # ~13% of the power in the k = 1 bin, so the ensemble is deep — 400
    # realizations put the chi^2 noise of the mean near 1%).
    n = 1 << 17
    var_band = _expected_mean_square(psd, n, f_s)
    var_samp = np.mean([
        np.mean(synthesize_from_psd(
            np_generator_from_key(jax.random.PRNGKey(500 + i)), n, psd, f_s
        ) ** 2)
        for i in range(400)
    ])
    rel_samp = abs(var_samp - var_band) / var_band
    assert rel_samp < 0.02, (var_samp, var_band)

    print("\n" + "=" * 72)
    print("GATE 1 (after test 3): Kondratiev-Gorodetsky variance vs Eq. 129")
    print(f"{'quantity':<38}{'measured':>16}{'target':>14}")
    print(f"{'Eq.129 var k_B T^2/(rho Cp V) [K^2]':<38}"
          f"{var_129:>16.6e}{var_analytic:>14.6e}")
    print(f"{'renormalized integral [K^2]':<38}"
          f"{var_numeric:>16.6e}{var_analytic:>14.6e}  "
          f"(rel err {rel_int:.3%} < 2%)")
    print(f"{'sampled variance (resolved band) [K^2]':<38}"
          f"{var_samp:>16.6e}{var_band:>14.6e}  "
          f"(rel err {rel_samp:.3%} < 2%)")
    print("=" * 72)


def test_kg_geometry_validation():
    cfg = nm_load_cfg(None)
    kw = dict(T_k=300.0, kappa_th=30.0, rho=float(cfg["rho_kg_per_m3"]),
              cp=700.0, mode_volume=1e-15, f_max=1e10)
    with pytest.raises(ValueError, match="d_a >= 1.2"):
        kondratiev_gorodetsky_psd(R=100e-6, d_a=1.0e-6, d_b=0.99e-6, **kw)
    with pytest.raises(ValueError, match="positive"):
        kondratiev_gorodetsky_psd(R=0.0, d_a=2e-6, d_b=1e-6, **kw)
    # T_k = 0 -> identically zero PSD and variance (noise-off convention)
    psd0, var0 = kondratiev_gorodetsky_psd(
        R=100e-6, d_a=2e-6, d_b=1e-6, **{**kw, "T_k": 0.0})
    assert var0 == 0.0 and np.all(psd0(np.logspace(0, 9, 50)) == 0.0)


def test_kg_config_validation_requires_geometry():
    cfg = dict(nm_load_cfg(None))
    cfg["trn_psd_model"] = "kondratiev_gorodetsky"
    with pytest.raises(ValueError, match="trn_R_m/trn_da_m/trn_db_m"):
        TRNoise(cfg)
    cfg.update(trn_R_m=100e-6, trn_da_m=2.0e-6, trn_db_m=8.0e-7)
    trn = TRNoise(cfg)          # now valid
    assert trn.is_colored


# ---------------------------------------------------------------------------
# (4) single_pole bit-identical to the pre-change AR(1)
# ---------------------------------------------------------------------------
def _legacy_ar1(key, n, tau_corr, sigma, t_r):
    """Inline reimplementation of the PRE-CHANGE _ar1_samples (float32 scan)."""
    alpha = jnp.exp(-t_r / tau_corr)
    sigma_step = sigma * jnp.sqrt(1 - alpha**2)
    xi = jax.random.normal(key, shape=(n,), dtype=jnp.float32)

    def scan_fn(x_prev, xi_n):
        x_next = alpha * x_prev + sigma_step * xi_n
        return x_next, x_next

    _, samples = jax.lax.scan(scan_fn, jnp.zeros((), dtype=jnp.float32), xi)
    return samples


def test_single_pole_bit_identical_to_legacy_ar1():
    cfg = nm_load_cfg(None)
    assert (cfg.get("trn_psd_model", "single_pole") or "single_pole") == \
        "single_pole", "repo default must remain single_pole"
    tn = TotalNoise(cfg)
    assert not tn.is_colored
    key = jax.random.PRNGKey(20260720)
    n = 4096

    # Legacy TotalNoise.sample: split(key,2) -> thermal AR(1), tccr AR(1),
    # combined trn - pyro + tccr in float32.
    key_thermal, key_tccr = jax.random.split(key, 2)
    temp = _legacy_ar1(key_thermal, n, tn.tau_th,
                       math.sqrt(tn.var_delta_t), tn.t_r)
    trn = (tn.omega_0 / tn.n0 * tn.dn_dT) * temp
    pyro = (tn.omega_0 * tn.n0**2 * tn.r33 * tn.p
            / (2.0 * tn.eps0 * tn.eps_r_eff)) * temp
    tccr = _legacy_ar1(key_tccr, n, tn.tccr.tau_carrier,
                       tn.tccr.sigma_tccr, tn.t_r)
    legacy = np.asarray((trn - pyro + tccr).astype(jnp.float32))

    assert np.array_equal(np.asarray(tn.sample(key, n)), legacy)

    # TRNoise standalone: AR(1) keyed directly.
    trn_legacy = np.asarray(_legacy_ar1(key, n, tn.tau_th,
                                        (tn.omega_0 / tn.n0) * tn.dn_dT
                                        * math.sqrt(tn.var_delta_t), tn.t_r))
    assert np.array_equal(np.asarray(TRNoise(cfg).sample(key, n)), trn_legacy)

    # Solver surface: _detuning_noise_sequences (the sequences handed to the
    # scan) must equal the vmapped legacy stream key-for-key.
    from simulator.lle_solver import _detuning_noise_sequences

    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    got = np.asarray(_detuning_noise_sequences(keys, 512, None))
    want = np.stack([
        np.asarray((lambda k: (
            (tn.omega_0 / tn.n0 * tn.dn_dT)
            * _legacy_ar1(jax.random.split(k, 2)[0], 512, tn.tau_th,
                          math.sqrt(tn.var_delta_t), tn.t_r)
            - (tn.omega_0 * tn.n0**2 * tn.r33 * tn.p
               / (2.0 * tn.eps0 * tn.eps_r_eff))
            * _legacy_ar1(jax.random.split(k, 2)[0], 512, tn.tau_th,
                          math.sqrt(tn.var_delta_t), tn.t_r)
            + _legacy_ar1(jax.random.split(k, 2)[1], 512,
                          tn.tccr.tau_carrier, tn.tccr.sigma_tccr, tn.t_r)
        ).astype(jnp.float32))(k)).astype(np.float64)
        for k in keys
    ])
    assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# (4b) AR(1) start-up (burn-in) policy: trn_ar1_stationary_init
#
# The legacy generator starts its lax.scan carry at exactly 0.0, so the record
# is not stationary: Var(x_n) = sigma^2 * (1 - alpha^(2n)), with a burn-in of
# n_burn = tau_corr/(2*t_r) round trips. See analysis/trn_burnin_study.py.
# ---------------------------------------------------------------------------


def _pooled_mean_square(tau_corr, t_r, n, n_real, seed, stationary_init):
    """Pooled E[x^2]/sigma^2 estimate over ``n_real`` vmapped realizations.

    RAW second moment, not ``np.var``: for tau_corr >> n*t_r a whole record is
    one slow excursion, so subtracting the sample mean would remove exactly the
    low-frequency power the burn-in suppresses. The process is zero-mean by
    construction, so this is the right variance estimator. sigma_physical = 1,
    hence the returned number IS the ratio to target.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), int(n_real))
    x = np.asarray(
        jax.vmap(
            lambda k: _ar1_samples(k, int(n), tau_corr, 1.0, t_r,
                                   stationary_init)
        )(keys),
        dtype=np.float64,
    )
    return float(np.mean(x**2))


def test_ar1_legacy_init_bit_identical():
    """stationary_init=False reproduces the PRE-CHANGE _ar1_samples bit-for-bit.

    Compared against ``_legacy_ar1`` above — the inline reimplementation of the
    generator as it stood before the flag existed — for several
    (N, tau, sigma) points, and at the two public surfaces that consume it
    (``TRNoise.sample`` and ``TotalNoise.sample_with_delta_t``). The default
    argument value is exercised as well as the explicit ``False``, so a future
    change of the default cannot slip through.
    """
    cfg = nm_load_cfg(None)
    tn = TotalNoise(cfg)
    key = jax.random.PRNGKey(20260807)

    cases = [
        (1, tn.tau_th, 1.0),
        (2, tn.tau_th, 1.0),
        (4096, tn.tau_th, math.sqrt(tn.var_delta_t)),
        (4096, 100.0 * tn.t_r, 3.7),          # short tau: burn-in fully resolved
        (10_000, 1.0e-7, 1.0),                # the TCCR correlation time
    ]
    for n, tau, sigma in cases:
        want = np.asarray(_legacy_ar1(key, n, tau, sigma, tn.t_r))
        got_default = np.asarray(_ar1_samples(key, n, tau, sigma, tn.t_r))
        got_explicit = np.asarray(
            _ar1_samples(key, n, tau, sigma, tn.t_r, False)
        )
        assert np.array_equal(got_default, want), (n, tau, sigma)
        assert np.array_equal(got_explicit, want), (n, tau, sigma)
        assert got_default.dtype == want.dtype == np.float32

        # ...and the stationary path must genuinely differ (so the equality
        # above is not vacuously true for both branches).
        got_stat = np.asarray(_ar1_samples(key, n, tau, sigma, tn.t_r, True))
        assert got_stat.shape == want.shape and got_stat.dtype == want.dtype
        if sigma > 0.0:
            assert not np.array_equal(got_stat, want), (n, tau, sigma)

    # Public surfaces: default construction must be untouched.
    n = 4096
    assert np.array_equal(
        np.asarray(TRNoise(cfg).sample(key, n)),
        np.asarray(_legacy_ar1(key, n, tn.tau_th,
                               (tn.omega_0 / tn.n0) * tn.dn_dT
                               * math.sqrt(tn.var_delta_t), tn.t_r)),
    )
    assert TRNoise(cfg).ar1_stationary_init is False
    assert TotalNoise(cfg).trn_ar1_stationary_init is False

    legacy_dt = np.asarray(
        _legacy_ar1(jax.random.split(key, 2)[0], n, tn.tau_th,
                    math.sqrt(tn.var_delta_t), tn.t_r)
    )
    for noise_config in (
        None,
        NoiseConfig(trn=True, pyro_eo=True),                       # field default
        NoiseConfig(trn=True, pyro_eo=True, trn_ar1_stationary_init=False),
    ):
        tn_i = TotalNoise(cfg, noise_config=noise_config)
        _combined, dt = tn_i.sample_with_delta_t(key, n)
        assert np.array_equal(np.asarray(dt, dtype=np.float32), legacy_dt), (
            f"noise_config={noise_config!r} perturbed the legacy dT stream"
        )

    # And the flag ON does change the solver-facing stream.
    tn_on = TotalNoise(cfg, noise_config=NoiseConfig(
        trn=True, pyro_eo=True, trn_ar1_stationary_init=True))
    assert tn_on.trn_ar1_stationary_init is True
    assert tn_on.trn.ar1_stationary_init is True
    _c_on, dt_on = tn_on.sample_with_delta_t(key, n)
    assert not np.array_equal(np.asarray(dt_on, dtype=np.float32), legacy_dt)


def test_ar1_stationary_variance_within_2pct_at_short_N():
    """N = 1000, 512 realizations: the stationary path realizes its target.

    Two regimes, both at N = 1000 with 512 realizations:

    (a) tau_corr = 8*t_r (record spans ~125 correlation times). Here the
        estimator is precise — the effective dof is
        R*N/L with L = (1+a^2)/(1-a^2) ~ 8, i.e. ~6.4e4, so its relative
        1-sigma spread is sqrt(2/6.4e4) ~ 0.6% and a 2% band is a ~3.5-sigma
        statement. This pins the AMPLITUDE CALIBRATION of the stationary draw:
        x0 ~ Normal(0, sigma_physical) and the innovations together must land
        on sigma_physical^2 exactly.

    (b) the production tau_th (~1.23e5 round trips, so N = 1000 sits deep
        inside the burn-in). Here a whole record is essentially ONE correlated
        excursion, so the estimator has only ~R degrees of freedom and a
        relative 1-sigma spread of sqrt(2/512) ~ 6.3%; a 2% band would be a
        0.3-sigma statement, i.e. meaningless. The assertion is therefore
        20% (~3.2 sigma) for the stationary path, plus the headline burn-in
        fact — the legacy path realizes < 5% of its target variance, against
        an analytic prediction of ~0.8%.

    Splitting the tolerance this way is deliberate: the tight 2% number is
    claimed only where the statistics support it.
    """
    cfg = nm_load_cfg(None)
    t_r = 1.0 / float(cfg["fsr_hz"])
    tau_th = float(cfg["tau_th_s"])
    n, n_real, seed = 1000, 512, 20260807

    def analytic_legacy(tau):
        alpha = math.exp(-t_r / tau)
        a2 = alpha * alpha
        return 1.0 - a2 * (-math.expm1(2 * n * math.log(alpha))) / (
            n * (1.0 - a2)
        )

    # --- (a) resolved-correlation regime: the 2% claim ---------------------
    tau_short = 8.0 * t_r
    stat_short = _pooled_mean_square(tau_short, t_r, n, n_real, seed, True)
    assert abs(stat_short - 1.0) < 0.02, (
        f"stationary_init=True realized {stat_short:.4f} of the target "
        f"variance at N={n}, tau={tau_short:.3e} s (expected 1.000 +- 0.02)"
    )
    # sanity: at 125 correlation times the legacy path is nearly stationary
    # too, so (a) alone could not have detected the burn-in — hence (b).
    leg_short = _pooled_mean_square(tau_short, t_r, n, n_real, seed, False)
    assert abs(leg_short - analytic_legacy(tau_short)) < 0.03, leg_short

    # --- (b) production regime: the burn-in is severe -----------------------
    stat_prod = _pooled_mean_square(tau_th, t_r, n, n_real, seed, True)
    leg_prod = _pooled_mean_square(tau_th, t_r, n, n_real, seed, False)
    predicted = analytic_legacy(tau_th)

    assert abs(stat_prod - 1.0) < 0.20, (
        f"stationary_init=True realized {stat_prod:.4f} of the target variance "
        f"at N={n}, tau_th={tau_th:.3e} s (expected 1.00 +- 0.20 at "
        f"{n_real} realizations)"
    )
    assert leg_prod < 0.05, (
        f"legacy AR(1) realized {leg_prod:.5f} of its target variance; the "
        f"burn-in bias is supposed to be severe here (analytic {predicted:.5f})"
    )
    assert abs(leg_prod - predicted) < 0.5 * predicted, (leg_prod, predicted)
    assert stat_prod / leg_prod > 20.0, (stat_prod, leg_prod)

    print("\n" + "=" * 72)
    print(f"AR(1) burn-in at N = {n}, {n_real} realizations "
          f"(realized/target variance)")
    print(f"{'regime':<34}{'legacy':>12}{'analytic':>12}{'stationary':>12}")
    print(f"{'tau = 8*t_r (resolved)':<34}{leg_short:>12.5f}"
          f"{analytic_legacy(tau_short):>12.5f}{stat_short:>12.5f}")
    print(f"{'tau = tau_th (production)':<34}{leg_prod:>12.5f}"
          f"{predicted:>12.5f}{stat_prod:>12.5f}")
    print("=" * 72)


@pytest.mark.parametrize("stationary_init", [False, True])
@pytest.mark.parametrize("tau_over_tr", [200.0, None])
def test_ar1_lag1_autocorrelation_matches_alpha(stationary_init, tau_over_tr):
    """Long N: the lag-1 sample autocorrelation equals alpha = exp(-t_r/tau).

    Both flag states, because changing the start-up policy must not touch the
    RECURSION — only its initial condition. ``tau_over_tr=None`` uses the
    production ``tau_th`` (alpha = 1 - 8.1e-6), which additionally checks that
    the float32 recursion still resolves a correlation time four orders of
    magnitude longer than the sampling interval.

    Tolerance is derived from the statistics rather than hand-tuned: 5x the
    large-sample standard error of a lag-1 correlation estimate,
    sqrt((1 - alpha^2)/N), PLUS the estimator's known finite-sample downward
    bias ~ (1 + 3*alpha)/N (which dominates the error budget once alpha
    approaches 1, as it does at the production tau_th). N is large enough that
    both terms are small; at N = 2e5 the bias term alone already exceeds the
    standard error for the production tau.
    """
    cfg = nm_load_cfg(None)
    t_r = 1.0 / float(cfg["fsr_hz"])
    tau = (tau_over_tr * t_r) if tau_over_tr is not None \
        else float(cfg["tau_th_s"])
    n = 1_000_000
    alpha = math.exp(-t_r / tau)

    x = np.asarray(
        _ar1_samples(jax.random.PRNGKey(4242), n, tau, 1.0, t_r,
                     stationary_init),
        dtype=np.float64,
    )
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    tol = 5.0 * math.sqrt((1.0 - alpha**2) / n) + (1.0 + 3.0 * alpha) / n
    assert abs(r1 - alpha) < tol, (
        f"lag-1 autocorrelation {r1:.9f} != alpha {alpha:.9f} "
        f"(|diff| {abs(r1 - alpha):.3e} > tol {tol:.3e}); "
        f"stationary_init={stationary_init}, tau={tau:.3e} s"
    )


def test_sample_with_delta_t_consistency():
    """combined == sample() bit-for-bit; dT drives both thermal channels."""
    cfg = nm_load_cfg(None)
    tn = TotalNoise(cfg)
    key = jax.random.PRNGKey(11)
    comb, dt = tn.sample_with_delta_t(key, 2048)
    assert np.array_equal(np.asarray(comb), np.asarray(tn.sample(key, 2048)))
    # SiN: TCCR is zero, so combined = (c_pull - pyro_coeff)*dT exactly
    # (up to the float32 cast of the combined output).
    recon = (tn.c_pull - tn.pyro_coeff) * np.asarray(dt, dtype=np.float64)
    assert np.allclose(np.asarray(comb, dtype=np.float64), recon, rtol=2e-6)


def test_alpha_l_zero_is_bitwise_neutral_and_scales_pull():
    cfg = dict(nm_load_cfg(None))
    trn0 = TRNoise(cfg)
    cfg_a = dict(cfg)
    cfg_a["alpha_L_per_k"] = 0.0
    assert TRNoise(cfg_a).c_pull == trn0.c_pull
    cfg_a["alpha_L_per_k"] = 1.0e-5    # fold in thermal expansion
    trn_a = TRNoise(cfg_a)
    want = (trn0.omega_0 / trn0.n0) * (trn0.dn_dT + trn0.n0 * 1.0e-5)
    assert trn_a.c_pull == pytest.approx(want, rel=1e-14)
    assert trn_a.c_pull > trn0.c_pull


# ---------------------------------------------------------------------------
# (5) CSV round trip
# ---------------------------------------------------------------------------
def test_csv_roundtrip_regenerates_matching_psd(tmp_path):
    cfg = nm_load_cfg(None)
    f_s = 1.0 / TotalNoise(cfg).t_r
    # Write the single-pole S_dT to CSV, read it back through the csv model,
    # regenerate, and compare the Welch PSD of the two generators.
    sp = single_pole_psd(TRNoise(cfg).var_delta_t, TRNoise(cfg).tau_th)
    f_tab = np.logspace(2, math.log10(f_s / 2.0), 400)
    csv_file = tmp_path / "s_delta_t.csv"
    np.savetxt(csv_file, np.column_stack([f_tab, sp(f_tab)]), delimiter=",")

    cfg_csv = dict(cfg)
    cfg_csv.update(trn_psd_model="csv", trn_psd_csv_path=str(csv_file),
                   trn_csv_units="S_delta_T")
    trn_csv = TRNoise(cfg_csv)
    assert trn_csv.is_colored

    # PSD agreement of the interpolated model vs the analytic source over
    # the tabulated span (log-log interpolation of a smooth curve).
    f_chk = np.logspace(2.2, math.log10(f_s / 2.0) - 0.1, 60)
    ratio = np.asarray(trn_csv.delta_t_psd(f_chk)) / sp(f_chk)
    assert np.max(np.abs(10.0 * np.log10(ratio))) < 0.1   # < 0.1 dB

    # Flat clamp outside the span + f = 0 guard.
    low = np.asarray(trn_csv.delta_t_psd(np.array([0.0, 1.0, 50.0])))
    assert np.allclose(low, low[0])
    hi = np.asarray(trn_csv.delta_t_psd(np.array([f_s, 10 * f_s])))
    assert np.allclose(hi, hi[0])

    # Regenerated samples: Welch within 3 dB/octave of the analytic target
    # in delta-omega units.
    n = 1 << 16
    x = np.stack([
        np.asarray(trn_csv.sample(jax.random.PRNGKey(700 + i), n))
        for i in range(6)
    ])
    from scipy.signal import welch as _w

    f_e, s_e = _w(x, fs=f_s, nperseg=1 << 12, axis=-1)
    s_e = s_e.mean(axis=0)[1:]
    f_e = f_e[1:]
    target = lambda f: trn_csv.c_pull**2 * sp(f)          # noqa: E731
    worst, _rows = _octave_band_ratio_db(f_e, s_e, target, f_e[4], f_s / 4.0)
    assert worst < 3.0, worst


def test_csv_delta_omega_units_share_delta_t_with_pyroeo(tmp_path):
    """S_delta_omega CSVs map to K^2/Hz via C_pull^2; Pyro-EO shares the dT."""
    cfg = dict(nm_load_cfg(None))
    # A chi2-active config so the Pyro-EO channel is non-zero.
    cfg.update(eo_r33_m_per_v=3.1e-11, pyroelectric_coeff_c_per_m2_k=9.6e-2)
    trn_ref = TRNoise(cfg)
    f_tab = np.logspace(3, 9, 200)
    s_domega = trn_ref.c_pull**2 * single_pole_psd(
        trn_ref.var_delta_t, trn_ref.tau_th)(f_tab)
    csv_file = tmp_path / "s_delta_omega.csv"
    np.savetxt(csv_file, np.column_stack([f_tab, s_domega]), delimiter=",")

    cfg.update(trn_psd_model="csv", trn_psd_csv_path=str(csv_file),
               trn_csv_units="S_delta_omega")
    tn = TotalNoise(cfg)
    # The dT PSD equals the S_delta_T that generated the file (conversion by
    # 1/C_pull^2), up to interpolation error.
    f_chk = np.logspace(3.2, 8.8, 40)
    want = single_pole_psd(trn_ref.var_delta_t, trn_ref.tau_th)(f_chk)
    got = np.asarray(tn.delta_t_psd(f_chk))
    assert np.max(np.abs(10 * np.log10(got / want))) < 0.1

    # Correlated channels: combined = (c_pull - pyro)*dT (TCCR active only
    # for surface-carrier configs; here it just adds an independent term).
    comb, dt = tn.sample_with_delta_t(jax.random.PRNGKey(5), 4096)
    comb64 = np.asarray(comb, dtype=np.float64)
    tccr = np.asarray(tn.tccr.sample(
        jax.random.split(jax.random.PRNGKey(5), 2)[1], 4096))
    recon = (tn.c_pull - tn.pyro_coeff) * np.asarray(dt) + tccr
    assert np.allclose(comb64, recon, rtol=2e-5, atol=1e-3 * np.std(recon))


def test_csv_validation_errors(tmp_path):
    cfg = dict(nm_load_cfg(None))
    cfg["trn_psd_model"] = "csv"
    with pytest.raises(ValueError, match="trn_psd_csv_path"):
        TRNoise(cfg)
    bad = tmp_path / "bad_units.csv"
    np.savetxt(bad, np.column_stack([[1e3, 1e6], [1.0, 1.0]]), delimiter=",")
    cfg.update(trn_psd_csv_path=str(bad), trn_csv_units="nonsense")
    with pytest.raises(ValueError, match="trn_csv_units"):
        TRNoise(cfg)
    cfg["trn_csv_units"] = "S_delta_T"
    TRNoise(cfg)   # valid two-row file parses
    with pytest.raises(ValueError, match="trn_psd_model"):
        TRNoise({**cfg, "trn_psd_model": "not_a_model"})


# ---------------------------------------------------------------------------
# (6) Segment continuity  -> GATE 2
# ---------------------------------------------------------------------------
def _short_tau_cfg():
    """Config variant with tau_th = 100 RT so a segment resolves the noise.

    The production tau_th (~1.2e5 RT) hides the boundary statistics inside
    the correlation time; shortening it makes the restart transient AND the
    stationarity of the fixed path measurable within a few hundred RT.
    """
    cfg = dict(nm_load_cfg(None))
    cfg["tau_th_s"] = 100.0 / float(cfg["fsr_hz"])
    return cfg


def test_segment_continuity_gate2():
    cfg = _short_tau_cfg()
    tn = TotalNoise(cfg)
    n_traj = 400
    seg = 500
    n_seg = 4
    t_total = seg * n_seg
    keys = jax.random.split(jax.random.PRNGKey(99), n_traj)

    # --- legacy per-segment path: regenerate per segment from x0 = 0 -------
    _fold = jax.vmap(jax.random.fold_in, in_axes=(0, None))
    legacy = np.concatenate([
        np.asarray(jax.vmap(lambda k: tn.sample(k, seg))(_fold(keys, s)))
        for s in range(n_seg)
    ], axis=1)

    # --- continuity path: full length once, sliced per segment -------------
    full = np.stack([
        tn.sample_full_with_delta_t(k, t_total)[0] for k in keys
    ])

    w = 50                       # window half..: [boundary, boundary+w)
    boundary_ms_l, interior_ms_l = [], []
    boundary_ms_f, interior_ms_f = [], []
    for s in range(1, n_seg):
        b = s * seg
        boundary_ms_l.append(np.mean(legacy[:, b:b + w] ** 2))
        interior_ms_l.append(np.mean(legacy[:, b + seg - 2 * w:b + seg - w] ** 2))
        boundary_ms_f.append(np.mean(full[:, b:b + w] ** 2))
        interior_ms_f.append(np.mean(full[:, b + seg - 2 * w:b + seg - w] ** 2))
    ratio_legacy = float(np.mean(boundary_ms_l) / np.mean(interior_ms_l))
    ratio_fixed = float(np.mean(boundary_ms_f) / np.mean(interior_ms_f))

    # Legacy restarts at zero: the boundary window sits deep in the AR(1)
    # transient (ratio << 1 — the OLD per-segment stats, pinned).
    assert ratio_legacy < 0.9, ratio_legacy
    # Continuity path: stationary across the boundary.
    assert 0.9 < ratio_fixed < 1.1, ratio_fixed

    print("\n" + "=" * 72)
    print("GATE 2 (after test 6): segment-boundary variance (tau_th = 100 RT)")
    print(f"{'path':<34}{'boundary/interior MS':>22}{'band':>14}")
    print(f"{'legacy per-segment (restart flaw)':<34}"
          f"{ratio_legacy:>22.4f}{'< 0.9':>14}")
    print(f"{'full-length-then-slice (fixed)':<34}"
          f"{ratio_fixed:>22.4f}{'[0.9, 1.1]':>14}")
    print("=" * 72)


def test_legacy_segment_noise_default_and_batch_paths(tmp_path):
    """Default config stays legacy; the two modes produce different streams
    but identical shapes, and the continuity mode is finite/stationary."""
    from data.dataset_generator import DatasetGenerator

    def make(config_path=None):
        gen = DatasetGenerator(
            param_grid={"pin": [1e-3], "sweep_rate": [1.0],
                        "Gamma_th": [4.72e-3], "noise_scale": [1.0]},
            config_path=config_path, output_dir=tmp_path, n_tau=64,
            snapshot_interval=10, seed=42)
        gen.SEGMENT_RT = 60
        gen.HOLD_RT = 30
        return gen

    gen_l = make()
    assert gen_l.legacy_segment_noise is True   # default preserved

    cfg = yaml.safe_load(open("config/sin_params.yaml"))
    cfg["physical_parameters"]["legacy_segment_noise"] = 0
    cont_yaml = tmp_path / "cont.yaml"
    yaml.safe_dump(cfg, open(cont_yaml, "w"), sort_keys=False)
    gen_c = make(str(cont_yaml))
    assert gen_c.legacy_segment_noise is False

    sr = 8.0 * gen_l.kappa / (2.0 * 60)
    params = [dict(pin=1e-3, sweep_rate=sr, Gamma_th=4.72e-3, noise_scale=1.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_l = gen_l.simulate_batch(params, batch_global_idx=0)
        res_c = gen_c.simulate_batch(params, batch_global_idx=0)
    for k in res_l:
        assert res_l[k].shape == res_c[k].shape, k
    assert np.isfinite(res_c["U_int"]).all()
    assert not np.array_equal(res_l["U_int"], res_c["U_int"])


# ---------------------------------------------------------------------------
# (7) Explicit per-channel enables (TRN / pyro-EO / TCCR)
#
# The channels used to be silenced only implicitly — TRN via T_k = 0 ("0 kelvin
# ambient"), pyro-EO/TCCR via r33 = 0. These pin the explicit switches AND the
# invariant that flipping one channel never disturbs another channel's stream.
# ---------------------------------------------------------------------------

# A chi2-like (TFLN) config: r33 != 0 makes pyro-EO and TCCR genuinely non-zero,
# so the cross-channel tests below are not vacuously true (on the committed SiN
# config sigma_tccr == 0 and every TCCR sample is zero whatever the switch says).
_CHI2_OVERRIDES = {
    "eo_r33_m_per_v": 3.1e-11,
    "pyroelectric_coeff_c_per_m2_k": 9.6e-2,
}


def _chi2_cfg():
    return {**nm_load_cfg(None), **_CHI2_OVERRIDES}


def test_trn_disabled_returns_exact_zeros():
    """enabled=False => exact zeros from every sampler, for every PSD model."""
    base = nm_load_cfg(None)
    geo = {"trn_R_m": 100e-6, "trn_da_m": 2.0e-6, "trn_db_m": 8.0e-7}
    models = (
        {"trn_psd_model": "single_pole"},
        {"trn_psd_model": "kondratiev_gorodetsky", **geo},
    )
    key = jax.random.PRNGKey(4242)
    f = np.logspace(3, 9, 128)

    for model_kw in models:
        cfg = {**base, **model_kw}
        on = TRNoise(cfg)                      # legacy rule: T_k = 300 -> enabled
        off = TRNoise(cfg, enabled=False)
        assert on.enabled is True and off.enabled is False

        for n in (1, 2, 1024):
            s = off.sample(key, n)
            assert s.shape == (n,)
            assert np.all(np.asarray(s) == 0.0), model_kw
            # dtype must match the enabled path exactly (float32 AR(1) / float64 colored)
            assert s.dtype == on.sample(key, n).dtype, model_kw

            dt = off.sample_delta_t(key, n)
            assert dt.shape == (n,) and dt.dtype == np.float64
            assert np.all(dt == 0.0), model_kw

        assert np.all(np.asarray(off.psd(f)) == 0.0), model_kw
        # and the enabled channel is genuinely non-zero, so the above is meaningful
        assert float(np.std(np.asarray(on.sample(key, 1024)))) > 0.0, model_kw

    # same contract for the other two channels (chi2 config so they are live)
    cfg2 = _chi2_cfg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cls in (PyroEONoise, TCCRNoise):
            on, off = cls(cfg2), cls(cfg2, enabled=False)
            assert on.enabled is True and off.enabled is False
            assert np.all(np.asarray(off.sample(key, 512)) == 0.0), cls.__name__
            assert off.sample(key, 512).dtype == on.sample(key, 512).dtype
            assert np.all(np.asarray(off.psd(f)) == 0.0), cls.__name__
            assert float(np.std(np.asarray(on.sample(key, 512)))) > 0.0, cls.__name__


def test_disabling_trn_does_not_shift_tccr_stream():
    """Toggling TRN must leave the TCCR stream bit-identical for the same key.

    This is the key-chain invariant: the split in sample_with_delta_t is
    unconditional, so the TCCR subkey does not depend on whether the thermal
    branch consumed a draw.
    """
    cfg = _chi2_cfg()
    key = jax.random.PRNGKey(31337)
    n = 2048

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trn_on = TotalNoise(cfg, noise_config=NoiseConfig(
            trn=True, pyro_eo=False, tccr=True))
        trn_off = TotalNoise(cfg, noise_config=NoiseConfig(
            trn=False, pyro_eo=False, tccr=True))
        comb_on, dt_on = trn_on.sample_with_delta_t(key, n)
        comb_off, dt_off = trn_off.sample_with_delta_t(key, n)

    # TRN off => dT identically zero; TRN on => genuinely fluctuating
    assert np.all(np.asarray(dt_off) == 0.0)
    assert float(np.std(np.asarray(dt_on))) > 0.0

    # With pyro off and TRN off, the combined sequence IS the TCCR stream; with TRN
    # on it is TCCR + c_pull*dT. Recover TCCR from the TRN-on run and compare.
    tccr_from_off = np.asarray(comb_off, dtype=np.float64)
    tccr_direct = np.asarray(
        trn_on.tccr.sample(jax.random.split(key, 2)[1], n), dtype=np.float64
    )
    assert np.array_equal(tccr_from_off, tccr_direct), (
        "TCCR stream changed when TRN was disabled"
    )
    assert float(np.std(tccr_direct)) > 0.0, "TCCR must be non-trivial here"

    # the TCCR subkey itself is independent of the TRN switch
    assert np.array_equal(
        np.asarray(trn_on.tccr.sample(jax.random.split(key, 2)[1], n)),
        np.asarray(trn_off.tccr.sample(jax.random.split(key, 2)[1], n)),
    )


def test_fsr_without_trn_warns_and_is_zero():
    """fsr=True with trn=False is almost certainly a user error: warn, and dT == 0."""
    cfg = nm_load_cfg(None)
    with pytest.warns(UserWarning, match="identically zero FSR noise"):
        tn = TotalNoise(cfg, noise_config=NoiseConfig(trn=False, fsr=True))

    _comb, dt = tn.sample_with_delta_t(jax.random.PRNGKey(5), 1024)
    # FSR is dD1(t) = (D1/omega0)*C_pull*dT(t): a zero dT makes it identically zero.
    assert np.all(np.asarray(dt) == 0.0)

    full, dt_full = tn.sample_full_with_delta_t(jax.random.PRNGKey(5), 1024)
    assert np.all(dt_full == 0.0) and np.all(full == 0.0)

    # the sane combinations do NOT warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        TotalNoise(cfg, noise_config=NoiseConfig(trn=True, fsr=True))
        TotalNoise(cfg, noise_config=NoiseConfig(trn=False, fsr=False))


@pytest.mark.parametrize("t_k", [300.0, 0.0])
def test_enabled_none_reproduces_legacy_bitwise(t_k):
    """enabled=None reproduces the pre-change implicit gating BIT-FOR-BIT.

    Rebuilt from the inline legacy AR(1) (``_legacy_ar1``), the same reference the
    pre-existing single-pole bit-identity test uses. Covers both sides of the old
    implicit rule: T_k = 300 (channel live) and T_k = 0 (channel silent).
    """
    cfg = {**nm_load_cfg(None), "T_k": t_k}
    key = jax.random.PRNGKey(20260805)
    n = 2048

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tn = TotalNoise(cfg)                      # noise_config=None => legacy rules

    # the legacy rule is exactly "T_k > 0" for the two temperature-driven channels
    assert tn.trn.enabled is (t_k > 0.0)
    assert tn.pyroeo.enabled is (t_k > 0.0)
    assert tn.tccr.enabled is False               # SiN: r33 = 0 => sigma_tccr = 0

    key_thermal, key_tccr = jax.random.split(key, 2)
    temp = _legacy_ar1(key_thermal, n, tn.tau_th,
                       math.sqrt(tn.var_delta_t), tn.t_r)
    trn = (tn.omega_0 / tn.n0 * tn.dn_dT) * temp
    pyro = (tn.omega_0 * tn.n0**2 * tn.r33 * tn.p
            / (2.0 * tn.eps0 * tn.eps_r_eff)) * temp
    tccr = _legacy_ar1(key_tccr, n, tn.tccr.tau_carrier,
                       tn.tccr.sigma_tccr, tn.t_r)
    legacy_combined = np.asarray((trn - pyro + tccr).astype(jnp.float32))

    combined, delta_t = tn.sample_with_delta_t(key, n)
    assert np.array_equal(np.asarray(combined), legacy_combined)
    assert np.array_equal(np.asarray(tn.sample(key, n)), legacy_combined)
    assert np.asarray(combined).dtype == legacy_combined.dtype

    # dT: the legacy stream upcast to float64 under the solver's x64 mode
    assert np.array_equal(
        np.asarray(delta_t), np.asarray(temp, dtype=np.float64)
    )

    if t_k == 0.0:
        assert np.all(legacy_combined == 0.0)     # old code already returned zeros
    else:
        assert float(np.std(legacy_combined)) > 0.0
