"""Noise model implementations for TFLN simulation."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import welch

from simulator.colored_noise import (
    csv_psd,
    kondratiev_gorodetsky_psd,
    np_generator_from_key,
    single_pole_psd,
    synthesize_from_psd,
)


_DEF_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

# Valid values of the ``trn_psd_model`` config key. ``single_pole`` is the
# historical AR(1)/Lorentzian model and stays BIT-IDENTICAL to the
# pre-colored-noise code; the other two produce host-side float64 sequences
# via simulator.colored_noise (FFT synthesis from the target PSD).
_TRN_PSD_MODELS = ("single_pole", "kondratiev_gorodetsky", "csv")
_TRN_CSV_UNITS = ("S_delta_T", "S_delta_omega")


def _resolve_enabled(enabled, legacy_rule: bool, name: str) -> bool:
    """Resolve a channel's ``enabled`` switch, defaulting to the legacy rule.

    ``enabled=None`` (the default everywhere) reproduces the historical, IMPLICIT
    gating exactly — the channel is on whenever the old code would have produced a
    non-zero sequence, and off precisely where the old code already produced exact
    zeros. An explicit ``True``/``False`` (or 0/1, matching the repository's numeric
    config convention) overrides it.

    Making the switch explicit is a readability fix, not a physics change: "T_k = 0"
    literally means "0 kelvin ambient", which is a nonsense way to spell "no
    thermorefractive noise".
    """
    if enabled is None:
        return bool(legacy_rule)
    if not isinstance(enabled, (bool, int, np.integer)) or int(enabled) not in (0, 1):
        raise ValueError(
            f"{name}: enabled must be boolean-valued (bool or 0/1), got {enabled!r}."
        )
    return bool(int(enabled))


def _zeros(n, dtype) -> jnp.ndarray:
    """Exact zeros of the shape/dtype a disabled sampler must return.

    A disabled channel returns these WITHOUT touching any RNG. That is safe for the
    key chain because JAX PRNG is functional: consuming a key produces no side effect,
    so skipping a draw cannot shift any other channel's stream. The split ladder
    itself is what must stay fixed — see ``TotalNoise.sample_with_delta_t``.

    The dtype is canonicalized so that with jax_enable_x64 OFF a requested float64
    degrades to float32 exactly as the ENABLED path's own arrays do — same dtype,
    no spurious truncation warning.
    """
    return jnp.zeros((int(n),), dtype=jax.dtypes.canonicalize_dtype(dtype))


def _zeros_psd(f) -> jnp.ndarray:
    """Zero PSD matching the shape/dtype the enabled ``psd()`` would return."""
    return jnp.zeros_like(
        jnp.asarray(f, dtype=jax.dtypes.canonicalize_dtype(np.float64))
    )


def _resolve_trn_psd_model(cfg) -> str:
    """Validate and return the ``trn_psd_model`` config value."""
    model = cfg.get("trn_psd_model", "single_pole")
    if model is None:
        model = "single_pole"
    model = str(model)
    if model not in _TRN_PSD_MODELS:
        raise ValueError(
            f"trn_psd_model must be one of {_TRN_PSD_MODELS}, got {model!r}."
        )
    return model


def _build_delta_t_psd(cfg, model: str, c_pull: float, f_s: float,
                       var_delta_t: float, tau_th: float):
    """Return ``(S_dT(f) callable [K^2/Hz], variance_target [K^2])``.

    * ``single_pole``: the Lorentzian spectral twin of the AR(1) generator,
      total variance = the Eq. 129 thermodynamic value ``var_delta_t``.
    * ``kondratiev_gorodetsky``: paper Eq. 130 shape renormalized so its
      integral over [0, f_s/2] equals the Eq. 129 variance (see
      simulator.colored_noise.kondratiev_gorodetsky_psd). Requires the
      geometry keys ``trn_R_m``/``trn_da_m``/``trn_db_m`` (validated here).
    * ``csv``: user-tabulated S_dT(f) or S_domega(f) selected by
      ``trn_csv_units``; S_domega is mapped to temperature units via
      S_dT = S_domega / C_pull^2 so the Pyro-EO channel can share the SAME
      dT sequence. T_k = 0 forces a zero PSD for EVERY model (the
      repository's deterministic noise-off convention — the CSV tabulation
      does not itself scale with T_k, so the switch is applied explicitly).
    """
    t_k = float(cfg.get("T_k", 300.0))
    if t_k == 0.0:
        zero = lambda f: np.zeros_like(np.asarray(f, dtype=np.float64))  # noqa: E731
        return zero, 0.0

    if model == "single_pole":
        return single_pole_psd(var_delta_t, tau_th), var_delta_t

    if model == "kondratiev_gorodetsky":
        missing = [k for k in ("trn_R_m", "trn_da_m", "trn_db_m")
                   if not (cfg.get(k) or 0.0) > 0.0]
        if missing:
            raise ValueError(
                f"trn_psd_model = 'kondratiev_gorodetsky' requires positive "
                f"geometry keys trn_R_m/trn_da_m/trn_db_m [m]; "
                f"missing/invalid: {missing}."
            )
        psd, _var = kondratiev_gorodetsky_psd(
            T_k=t_k,
            kappa_th=float(cfg.get("kappa_th_w_per_m_k", 4.6)),
            rho=float(cfg.get("rho_kg_per_m3", 4.64e3)),
            cp=float(cfg.get("Cp_j_per_kg_k", 700.0)),
            R=float(cfg["trn_R_m"]),
            d_a=float(cfg["trn_da_m"]),
            d_b=float(cfg["trn_db_m"]),
            mode_volume=float(cfg.get("mode_volume_m3", 1.0e-15)),
            f_max=f_s / 2.0,
        )
        return psd, _var

    # model == "csv"
    path = cfg.get("trn_psd_csv_path")
    if not path:
        raise ValueError(
            "trn_psd_model = 'csv' requires trn_psd_csv_path (two-column "
            "CSV: f [Hz], S)."
        )
    units = cfg.get("trn_csv_units", "S_delta_T") or "S_delta_T"
    if units not in _TRN_CSV_UNITS:
        raise ValueError(
            f"trn_csv_units must be one of {_TRN_CSV_UNITS}, got {units!r}."
        )
    raw = csv_psd(path)
    if units == "S_delta_omega":
        if c_pull == 0.0:
            raise ValueError(
                "trn_csv_units = 'S_delta_omega' needs a non-zero frequency "
                "pull C_pull = (omega0/n0)*(dn_dT + n0*alpha_L_per_k) to map "
                "the PSD into temperature units."
            )
        inv_c2 = 1.0 / c_pull**2
        psd = lambda f, _raw=raw, _s=inv_c2: _s * _raw(f)   # noqa: E731
    else:
        psd = raw
    # Variance target: numeric integral over the synthesis band (measured
    # PSDs have no closed form).
    from simulator.colored_noise import integrate_psd

    var = integrate_psd(psd, f_lo=max(1.0, 1e-6 * f_s), f_hi=f_s / 2.0)
    return psd, var


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and return physical parameters dict."""
    cfg_path = Path(config_path) if config_path is not None else _DEF_CFG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("physical_parameters", {})


def _ar1_samples(key, N, tau_corr, sigma_physical, t_r,
                 stationary_init: bool = False):
    """Generate AR(1) samples with target stationary variance ``sigma_physical**2``.

    The recursion is ``x_n = alpha*x_{n-1} + sigma_step*xi_n`` with
    ``alpha = exp(-t_r/tau_corr)`` and
    ``sigma_step = sigma_physical*sqrt(1 - alpha**2)``, so the STATIONARY
    variance is exactly ``sigma_physical**2``. Which variance the returned
    record actually realizes depends on how the scan carry is initialised:

    ``stationary_init=False`` (default -- the historical behaviour)
        The carry starts at exactly ``0.0``. The record is therefore NOT
        stationary; its variance ramps as

            Var(x_n) = sigma_physical**2 * (1 - alpha**(2n)),

        i.e. it reaches a fraction ``1 - exp(-2*n*t_r/tau_corr)`` of the target
        after ``n`` steps. The burn-in scale is ``n_burn = tau_corr/(2*t_r)``
        round trips (the 1/e point of ``alpha**(2n)``). Pooled over a record of
        length ``N`` the mean square is suppressed to

            E[mean(x**2)] / sigma_physical**2
                = 1 - alpha**2 * (1 - alpha**(2N)) / (N*(1 - alpha**2))
                ~ N*t_r/tau_corr        for  N*t_r << tau_corr,

        so a SHORT run has systematically suppressed amplitude, worst at the
        start of the record. With the repository's TRN numbers
        (``t_r = 40.65 ps``, ``tau_th = 5 us``) ``n_burn ~ 6.1e4`` round trips,
        which is longer than most trajectories -- see
        ``analysis/trn_burnin_study.py`` for the measured bias curve.

        This path consumes the key EXACTLY as the pre-change implementation
        did (one ``jax.random.normal`` draw of shape ``(N,)`` keyed directly on
        ``key``, no split), so it is bit-identical to it.

    ``stationary_init=True``
        ``key`` is split ONCE into ``(key_x0, key_xi)``; the initial carry is
        drawn as ``x0 ~ Normal(0, sigma_physical)`` from the FIRST subkey and
        the ``N`` innovations from the SECOND. The record is then stationary
        from sample 0: ``Var(x_n) = sigma_physical**2`` for every ``n``, and
        ``Corr(x_n, x_{n+1}) = alpha`` including at ``n = 0``.

        Drawing ``x0`` from a dedicated subkey (rather than, say, taking the
        first element of an ``N+1``-long draw) is what keeps the innovation
        stream independent of ``N``: ``key_xi`` is a fixed function of ``key``
        alone, so lengthening a run extends its innovation sequence instead of
        re-deriving it.

    Args:
        key: JAX PRNG key.
        N: number of samples to return.
        tau_corr: correlation time [s].
        sigma_physical: target stationary standard deviation.
        t_r: sampling interval (round-trip time) [s].
        stationary_init: see above. ``False`` (default) preserves the legacy
            zero-start behaviour bit-for-bit.

    Returns:
        ``(N,)`` float32 array (the scan carry dtype pins the whole recursion
        to float32 -- see ``docs/NOISE_CHANNEL_INVENTORY.md`` G3).
    """
    alpha = jnp.exp(-t_r / tau_corr)
    sigma_step = sigma_physical * jnp.sqrt(1 - alpha**2)
    if stationary_init:
        key_x0, key_xi = jax.random.split(key, 2)
        x0 = (
            sigma_physical
            * jax.random.normal(key_x0, shape=(), dtype=jnp.float32)
        ).astype(jnp.float32)
    else:
        # Legacy: no split at all, so the key is consumed exactly as before.
        key_xi = key
        x0 = jnp.zeros((), dtype=jnp.float32)
    xi = jax.random.normal(key_xi, shape=(N,), dtype=jnp.float32)

    def scan_fn(x_prev, xi_n):
        x_next = alpha * x_prev + sigma_step * xi_n
        return x_next, x_next

    _, samples = jax.lax.scan(scan_fn, x0, xi)
    return samples


class TRNoise:
    """Thermorefractive (TRN) detuning noise with a pluggable PSD strategy.

    ``trn_psd_model`` (config) selects the temperature-fluctuation spectrum:

    * ``single_pole`` (default): the historical AR(1) generator — the
      sampled stream is BYTE-COMPATIBLE with the pre-colored-noise code.
    * ``kondratiev_gorodetsky``: analytic WGM PSD, arXiv:2604.05897 Eq. 130,
      variance renormalized to Eq. 129 (see simulator.colored_noise).
    * ``csv``: measured/FEM tabulated PSD (Huang et al. 2019 style).

    In every model the frequency-pull map is ``domega(t) = C_pull*dT(t)``
    with ``C_pull = (omega0/n0)*(dn_dT + n0*alpha_L_per_k)`` — the optional
    ``alpha_L_per_k`` (thermal-expansion coefficient, 1/K, default 0) folds
    the paper's "dimensional fluctuation" companion of TRN into the pull
    coefficient. Colored models synthesize HOST-SIDE (float64, numpy rng
    derived deterministically from the JAX key); ``single_pole`` keeps the
    traced AR(1) path.
    """

    def __init__(self, cfg, enabled: bool | None = None,
                 ar1_stationary_init: bool = False):
        self.cfg = cfg
        # Start the legacy AR(1) generator from its stationary distribution
        # instead of from x0 = 0. False (default) is bit-identical to the
        # historical behaviour; see _ar1_samples for the burn-in formula.
        # Only meaningful for trn_psd_model = single_pole -- the colored models
        # synthesize from the PSD and are stationary from sample 0 already.
        self.ar1_stationary_init = bool(ar1_stationary_init)
        self.t_r = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.omega_0 = 2.0 * math.pi * 299_792_458.0 / float(cfg.get("pump_wavelength_m", 1.55e-6))
        self.n0 = float(cfg.get("n0", 2.2))
        self.dn_dT = float(cfg.get("dn_dT_per_k", 4.0e-5))
        self.tau_th = float(cfg.get("tau_th_s", 5.0e-6))
        self.rho = float(cfg.get("rho_kg_per_m3", 4.64e3))
        self.cp = float(cfg.get("Cp_j_per_kg_k", 700.0))
        self.v = float(cfg.get("mode_volume_m3", 1.0e-15))
        self.kappa_th = float(cfg.get("kappa_th_w_per_m_k", 4.6))
        self.T_k = float(cfg.get("T_k", 300.0))
        self.k_b = 1.380649e-23
        self.var_delta_t = self.k_b * self.T_k**2 / (self.rho * self.cp * self.v)
        # Frequency pull dω/dT [rad/s/K]: thermo-optic + (optional) thermal
        # expansion. With alpha_L_per_k = 0 this is bit-identical to the
        # historical (omega0/n0)*dn_dT (x + n0*0.0 == x in IEEE arithmetic).
        self.alpha_L = float(cfg.get("alpha_L_per_k", 0.0) or 0.0)
        self.c_pull = (self.omega_0 / self.n0) * (
            self.dn_dT + self.n0 * self.alpha_L
        )
        self.sigma_trn = self.c_pull * math.sqrt(self.var_delta_t)
        self.psd_model = _resolve_trn_psd_model(cfg)
        self.f_s = 1.0 / self.t_r
        # (S_dT(f) [K^2/Hz], variance target [K^2]) for the selected model —
        # built eagerly so config errors surface at construction time.
        self.delta_t_psd, self.var_delta_t_target = _build_delta_t_psd(
            cfg, self.psd_model, self.c_pull, self.f_s,
            self.var_delta_t, self.tau_th,
        )
        # Explicit switch. Legacy rule: the channel was live iff the thermodynamic
        # variance k_B*T_k^2/(rho*Cp*V) was non-zero, i.e. iff T_k > 0. At T_k = 0 the
        # old code still ran the generator but every sample was exactly 0.0, so
        # enabled=None is bit-identical to the old behaviour on BOTH sides of the rule.
        self.enabled = _resolve_enabled(enabled, self.T_k > 0.0, "TRNoise")

    @property
    def is_colored(self) -> bool:
        return self.psd_model != "single_pole"

    @property
    def _sample_dtype(self):
        """dtype of :meth:`sample` — float32 on the legacy AR(1) path, else float64."""
        return jnp.float64 if self.is_colored else jnp.float32

    def sample_delta_t(self, key, N) -> np.ndarray:
        """Host-side float64 dT(t) sequence, (N,), from the selected PSD.

        Available for EVERY model (single_pole uses its Lorentzian spectral
        twin), stationary from sample 0 — this is the sequence the
        segment-continuity path slices. NOT the byte-compatible AR(1) stream
        (that one lives in :meth:`sample` for ``single_pole``).

        Disabled: exact float64 zeros, no RNG draw.
        """
        if not self.enabled:
            return np.zeros(int(N), dtype=np.float64)
        rng = np_generator_from_key(key)
        return synthesize_from_psd(rng, int(N), self.delta_t_psd, self.f_s)

    def sample(self, key, N) -> jnp.ndarray:
        if not self.enabled:
            return _zeros(N, self._sample_dtype)
        if not self.is_colored:
            return _ar1_samples(key, N, self.tau_th, self.sigma_trn, self.t_r,
                                self.ar1_stationary_init)
        return jnp.asarray(self.c_pull * self.sample_delta_t(key, N),
                           dtype=jnp.float64)

    def psd(self, f) -> jnp.ndarray:
        """One-sided S_domega(f) [(rad/s)^2/Hz] of the selected model."""
        if not self.enabled:
            return _zeros_psd(f)
        if not self.is_colored:
            s_delta_t = (
                (4.0 * self.k_b * self.T_k**2 * self.tau_th) / (self.rho * self.cp * self.v)
            ) / (1.0 + (2.0 * jnp.pi * f * self.tau_th) ** 2)
            # c_pull == (omega0/n0)*dn_dT when alpha_L_per_k = 0 (the
            # historical value); with expansion enabled the pull follows suit.
            return self.c_pull**2 * s_delta_t
        return jnp.asarray(
            self.c_pull**2 * self.delta_t_psd(np.asarray(f, dtype=np.float64))
        )


class PyroEONoise:
    def __init__(self, cfg, enabled: bool | None = None):
        self.cfg = cfg
        self.t_r = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.omega_0 = 2.0 * math.pi * 299_792_458.0 / float(cfg.get("pump_wavelength_m", 1.55e-6))
        self.n0 = float(cfg.get("n0", 2.2))
        self.r33 = float(cfg.get("eo_r33_m_per_v", 3.1e-11))
        self.p = float(cfg.get("pyroelectric_coeff_c_per_m2_k", 9.6e-2))
        self.tau_th = float(cfg.get("tau_th_s", 5.0e-6))
        self.rho = float(cfg.get("rho_kg_per_m3", 4.64e3))
        self.cp = float(cfg.get("Cp_j_per_kg_k", 700.0))
        self.v = float(cfg.get("mode_volume_m3", 1.0e-15))
        self.kappa_th = float(cfg.get("kappa_th_w_per_m_k", 4.6))
        self.T_k = float(cfg.get("T_k", 300.0))
        self.eps0 = 8.8541878128e-12
        self.k_b = 1.380649e-23
        self.var_delta_t = self.k_b * self.T_k**2 / (self.rho * self.cp * self.v)

        self.eps_r_z = float(cfg.get("eps_r_z", 28.0))
        # Geometric screening from dielectric boundary conditions in thin-film stack.
        # 1D approximation: E_LN = P / (ε₀ * ε_r_eff) where ε_r_eff accounts for
        # the fraction of field extending into cladding layers.
        _t_ln  = float(cfg.get("t_ln_m", 4.0e-7))
        _t_top = float(cfg.get("t_clad_top_m", 1.0e-6))
        _t_bot = float(cfg.get("t_clad_bot_m", 2.0e-6))
        _er_top = float(cfg.get("eps_r_clad_top", 1.0))
        _er_bot = float(cfg.get("eps_r_clad_bot", 3.9))
        self.eps_r_eff = (
            self.eps_r_z
            + _er_top * (_t_top / _t_ln)
            + _er_bot * (_t_bot / _t_ln)
        )
        # η_geom = eps_r_z / eps_r_eff  (implicitly encoded by using eps_r_eff)
        # Pyro-EO frequency pull dω/dT [rad/s/K]; DRIVEN BY THE SAME dT(t) as
        # TRN (see TotalNoise), whatever PSD model generates that dT.
        self.pyro_coeff = (
            self.omega_0 * self.n0**2 * self.r33 * self.p
            / (2.0 * self.eps0 * self.eps_r_eff)
        )
        self.sigma_pyroeo = self.pyro_coeff * math.sqrt(self.var_delta_t)
        # Same PSD strategy as TRNoise (the pyro-EO channel is temperature-
        # driven, so it inherits the trn_psd_model selection).
        self.psd_model = _resolve_trn_psd_model(cfg)
        self.f_s = 1.0 / self.t_r
        _alpha = float(cfg.get("alpha_L_per_k", 0.0) or 0.0)
        _dn_dT = float(cfg.get("dn_dT_per_k", 4.0e-5))
        _c_pull = (self.omega_0 / self.n0) * (_dn_dT + self.n0 * _alpha)
        self.delta_t_psd, self.var_delta_t_target = _build_delta_t_psd(
            cfg, self.psd_model, _c_pull, self.f_s,
            self.var_delta_t, self.tau_th,
        )
        # Explicit switch. Legacy rule: this channel is temperature-driven, so it was
        # live iff T_k > 0 — the SAME rule as TRNoise. (A zero pyro/EO coefficient,
        # e.g. r33 = 0 on centrosymmetric SiN, independently makes every sample 0.0;
        # that is a material fact, not a switch, so it is deliberately NOT folded in
        # here. enabled=None therefore stays bit-identical either way.)
        self.enabled = _resolve_enabled(enabled, self.T_k > 0.0, "PyroEONoise")

    @property
    def is_colored(self) -> bool:
        return self.psd_model != "single_pole"

    @property
    def _sample_dtype(self):
        """dtype of :meth:`sample` — float32 on the legacy AR(1) path, else float64."""
        return jnp.float64 if self.is_colored else jnp.float32

    def sample(self, key, N) -> jnp.ndarray:
        if not self.enabled:
            return _zeros(N, self._sample_dtype)
        if not self.is_colored:
            return _ar1_samples(key, N, self.tau_th, self.sigma_pyroeo, self.t_r)
        rng = np_generator_from_key(key)
        delta_t = synthesize_from_psd(rng, int(N), self.delta_t_psd, self.f_s)
        return jnp.asarray(self.pyro_coeff * delta_t, dtype=jnp.float64)

    def psd(self, f) -> jnp.ndarray:
        if not self.enabled:
            return _zeros_psd(f)
        if not self.is_colored:
            s_delta_t = (
                (4.0 * self.k_b * self.T_k**2 * self.tau_th) / (self.rho * self.cp * self.v)
            ) / (1.0 + (2.0 * jnp.pi * f * self.tau_th) ** 2)
            return self.pyro_coeff**2 * s_delta_t
        return jnp.asarray(
            self.pyro_coeff**2 * self.delta_t_psd(np.asarray(f, dtype=np.float64))
        )


class TCCRNoise:
    def __init__(self, cfg, enabled: bool | None = None):
        self.t_r         = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.omega_0     = 2.0 * math.pi * 299_792_458.0 / float(cfg.get("pump_wavelength_m", 1.55e-6))
        self.tau_carrier = float(cfg.get("tau_carrier_s", 1.0e-7))
        self.k_b         = 1.380649e-23
        self.T_k         = float(cfg.get("T_k", 300.0))

        # Physical path: surface carrier shot noise → EO frequency shift
        n_s       = float(cfg.get("surface_state_density_per_m2", 1.0e16))   # m⁻²
        r33       = float(cfg.get("eo_r33_m_per_v",  3.1e-11))               # m/V
        n0        = float(cfg.get("n0", 2.2))
        eps0      = 8.8541878128e-12
        eps_r_eff = float(cfg.get("eps_r_z", 28.0))   # simplified; use PyroEO value for full model
        A_eff     = float(cfg.get("effective_mode_area_m2", 1.0e-12))         # m²
        t_ln      = float(cfg.get("t_ln_m", 4.0e-7))                         # m
        e_charge  = 1.602176634e-19                                            # C

        # Equilibrium surface carrier number within mode footprint
        N_s_eq = n_s * A_eff                                                   # dimensionless

        # EO frequency shift per carrier [rad/s per carrier]
        # Derivation: delta_n = -n0^3 * r33 * E / 2  =>  delta_omega = omega_0 * delta_n / n0
        #                     = -omega_0 * n0 ^ 2 * r33 * E / 2
        # E_per_carrier is already in V/m; t_ln does NOT appear here.
        E_per_carrier = e_charge / (eps0 * eps_r_eff * A_eff)   # V/m per carrier
        dw_dNs = -self.omega_0 * n0**2 * r33 * E_per_carrier / 2.0   # rad/s per carrier  ← NO t_ln

        # Two-sided TCCR PSD at f=0: S0 = (dω/dNs)² · N_s_eq · 2·τ_carrier
        self.s0_tccr    = dw_dNs**2 * N_s_eq * 2.0 * self.tau_carrier        # (rad/s)²/Hz ✓
        self.var_tccr   = self.s0_tccr / (2.0 * self.tau_carrier)             # stationary variance
        self.sigma_tccr = math.sqrt(max(self.var_tccr, 0.0))

        # Sanity: for chi2 platforms (e.g. TFLN) sigma_tccr ~ [1e4, 1e11] rad/s.
        # sigma_tccr == 0 is the expected SiN case (r33 = 0): skip the warning.
        if self.sigma_tccr > 0.0 and not (1e4 < self.sigma_tccr < 1e11):
            import warnings
            warnings.warn(
                f"TCCRNoise.sigma_tccr = {self.sigma_tccr:.2e} rad/s is outside the "
                f"expected physical range [1e4, 1e11] rad/s. "
                f"Check surface_state_density_per_m2 and eo_r33_m_per_v in config.",
                stacklevel=2,
            )

        kappa_estimate = 2.0 * 2.0 * math.pi * 299_792_458.0 / (
            float(cfg.get("pump_wavelength_m", 1.55e-6)) * float(cfg.get("intrinsic_q", 2e6))
        )
        if self.sigma_tccr > kappa_estimate:
            import warnings
            warnings.warn(
                f"sigma_tccr ({self.sigma_tccr:.2e} rad/s) > kappa ({kappa_estimate:.2e} rad/s). "
                f"TCCR noise is non-perturbative and will destabilize all solitons. "
                f"Reduce surface_state_density_per_m2 (currently {n_s:.1e} m^-2) or calibrate "
                f"against the Yu lab's measured noise floor before generating the training dataset.",
                stacklevel=2,
            )

        # Explicit switch. Legacy rule: TCCR was silenced ONLY by a zero EO
        # coefficient (r33 = 0 => dw_dNs = 0 => s0_tccr = 0 => sigma_tccr = 0), in
        # which case the old AR(1) already returned exact zeros. Note this channel is
        # NOT gated by T_k — its variance carries no T_k factor — which is exactly why
        # the T_k=0 "noise-off" convention never silenced it on chi2 platforms.
        self.enabled = _resolve_enabled(
            enabled, self.sigma_tccr > 0.0, "TCCRNoise"
        )

    def sample(self, key, N) -> jnp.ndarray:
        if not self.enabled:
            return _zeros(N, jnp.float32)
        return _ar1_samples(key, N, self.tau_carrier, self.sigma_tccr, self.t_r)

    def psd(self, f) -> jnp.ndarray:
        if not self.enabled:
            return _zeros_psd(f)
        return self.s0_tccr / (1.0 + (2.0 * jnp.pi * f * self.tau_carrier) ** 2)


class TotalNoise:
    """Combined TRN + Pyro-EO + TCCR detuning noise.

    The TRN and Pyro-EO channels share ONE temperature sequence dT(t) — they
    are the same thermodynamic fluctuation seen through two different pull
    coefficients — whatever ``trn_psd_model`` generates that sequence. TCCR
    (carrier noise, zero for SiN) keeps its independent AR(1) stream.

    Sampling surfaces:
      * :meth:`sample` — the historical (N,) float32 combined sequence;
        BYTE-COMPATIBLE with the pre-colored-noise code for
        ``trn_psd_model = single_pole``.
      * :meth:`sample_with_delta_t` — same combined sequence PLUS the
        underlying dT(t) (float64), so the FSR-noise channel
        dD1(t) = (D1/omega0)*C_pull*dT(t) can reuse the identical sequence.
      * :meth:`sample_full_with_delta_t` — host-side float64 PSD-synthesized
        path for ALL models (single_pole uses its Lorentzian spectral twin);
        stationary from sample 0, used by the ``legacy_segment_noise = 0``
        full-trajectory-then-slice mode of the dataset generator.
    """

    def __init__(self, cfg, noise_config=None):
        """Build the combined channel.

        Args:
            cfg: the ``physical_parameters`` mapping.
            noise_config: optional :class:`simulator.noise_config.NoiseConfig`. When
                ``None`` (the default) every sub-channel resolves its own switch from
                the legacy rule, so behaviour is EXACTLY as before this parameter
                existed. When supplied, each sub-channel's ``enabled`` comes from the
                corresponding field (``trn``, ``pyro_eo``, ``tccr``), and
                ``trn_ar1_stationary_init`` selects the start-up policy of the
                shared thermal AR(1) stream (see :func:`_ar1_samples`). That
                field defaults to ``False``, so a supplied config that leaves it
                alone is still bit-identical to the historical stream.
        """
        self.cfg = cfg
        self.noise_config = noise_config
        # AR(1) start-up policy for the SHARED thermal stream (single_pole only).
        # Default False => bit-identical to the historical zero-start behaviour.
        self.trn_ar1_stationary_init = bool(
            getattr(noise_config, "trn_ar1_stationary_init", False)
        )
        if noise_config is None:
            trn_enabled = pyro_enabled = tccr_enabled = None
        else:
            trn_enabled = bool(noise_config.trn)
            pyro_enabled = bool(noise_config.pyro_eo)
            tccr_enabled = bool(noise_config.tccr)
            if noise_config.fsr and not noise_config.trn:
                warnings.warn(
                    "fsr=True with trn=False produces identically zero FSR noise "
                    "because both are driven by the same delta_T realization.",
                    stacklevel=2,
                )
        self.trn = TRNoise(cfg, enabled=trn_enabled,
                           ar1_stationary_init=self.trn_ar1_stationary_init)
        self.pyroeo = PyroEONoise(cfg, enabled=pyro_enabled)
        self.tccr = TCCRNoise(cfg, enabled=tccr_enabled)
        self.t_r = self.trn.t_r
        self.omega_0 = self.trn.omega_0
        self.n0 = self.trn.n0
        self.dn_dT = self.trn.dn_dT
        self.r33 = self.pyroeo.r33
        self.p = self.pyroeo.p

        self.eps0 = self.pyroeo.eps0
        self.eps_r_z = self.pyroeo.eps_r_z
        self.eps_r_eff = self.pyroeo.eps_r_eff

        self.tau_th = self.trn.tau_th
        self.var_delta_t = self.trn.var_delta_t
        self.tau_carrier = self.tccr.tau_carrier
        self.psd_model = self.trn.psd_model
        self.c_pull = self.trn.c_pull
        self.pyro_coeff = self.pyroeo.pyro_coeff
        self.delta_t_psd = self.trn.delta_t_psd
        self.var_delta_t_target = self.trn.var_delta_t_target
        self.f_s = 1.0 / self.t_r

    @property
    def is_colored(self) -> bool:
        return self.psd_model != "single_pole"

    def sample_with_delta_t(self, key, N):
        """(combined detuning noise, dT sequence): shapes (N,), (N,).

        The combined sequence is float32 and — for ``single_pole`` —
        bit-identical to the historical :meth:`sample` (identical key split,
        identical arithmetic; dT is merely also returned). dT is float64
        (colored) / the float32 AR(1) stream upcast (legacy). The legacy
        branch is fully traceable (vmap-safe); colored models synthesize on
        the HOST and must be looped, not vmapped.
        """
        # CRITICAL: this split is UNCONDITIONAL. It must happen regardless of which
        # channels are enabled, so that enabling or disabling one channel never
        # shifts another channel's random stream. Never move it inside a branch,
        # never change its arity, and never reorder the subkeys.
        key_thermal, key_tccr = jax.random.split(key, 2)
        if not self.trn.enabled:
            # dT is the SHARED realization behind TRN, pyro-EO and FSR: with the
            # thermorefractive channel off there is no temperature fluctuation at all,
            # so all three collapse to zero. No RNG is drawn (safe: JAX PRNG is
            # functional, so a skipped draw cannot perturb key_tccr above).
            temp_noise = _zeros(N, jnp.float64 if self.is_colored else jnp.float32)
        elif not self.is_colored:
            # trn_ar1_stationary_init=False (default) reproduces the historical
            # zero-start stream bit-for-bit, including its ~tau_th/(2*t_r)
            # round-trip burn-in; True starts from the stationary distribution.
            temp_noise = _ar1_samples(
                key_thermal, N, self.tau_th, math.sqrt(self.var_delta_t),
                self.t_r, self.trn_ar1_stationary_init,
            )
        else:
            rng = np_generator_from_key(key_thermal)
            temp_noise = jnp.asarray(
                synthesize_from_psd(rng, int(N), self.delta_t_psd, self.f_s),
                dtype=jnp.float64,
            )
        # With both channels enabled these are exactly the historical expressions.
        trn_noise = (self.c_pull if self.trn.enabled else 0.0) * temp_noise
        pyroeo_noise = (self.pyro_coeff if self.pyroeo.enabled else 0.0) * temp_noise
        tccr_noise = self.tccr.sample(key_tccr, N)

        # Sign convention: PyroEO *partially cancels* TRN for z-cut TFLN with
        # air top-cladding (Yu lab geometry).  For SiO₂-clad or flipped substrate,
        # the sign of pyroeo_noise may need to flip.  Verify against Fig. 2 of the
        # TCCR paper (DOI to be added) before generating the training dataset.
        combined = (trn_noise - pyroeo_noise + tccr_noise).astype(jnp.float32)
        # dT stays float64 where available (colored path is float64 already;
        # the legacy AR(1) stream upcasts only under the solver's x64 mode —
        # avoids a spurious truncation warning in standalone float32 use).
        if jax.config.read("jax_enable_x64") and temp_noise.dtype != jnp.float64:
            temp_noise = temp_noise.astype(jnp.float64)
        return combined, temp_noise

    def sample(self, key, N) -> jnp.ndarray:
        return self.sample_with_delta_t(key, N)[0]

    def sample_full_with_delta_t(self, key, N):
        """Host float64 (combined, dT) pair for the segment-continuity path.

        EVERY model synthesizes from its PSD here (single_pole from the
        Lorentzian twin of the AR(1)), so the sequence is stationary from
        sample 0 — a full trajectory generated once up front and sliced per
        segment has no boundary decorrelation transient. TCCR also
        synthesizes from its single-pole PSD (independent stream from the
        second subkey). Returns numpy float64 arrays, shape (N,) each.
        """
        # CRITICAL: unconditional split — see sample_with_delta_t. Toggling a channel
        # must never shift another channel's stream.
        key_thermal, key_tccr = jax.random.split(key, 2)
        if self.trn.enabled:
            rng_t = np_generator_from_key(key_thermal)
            delta_t = synthesize_from_psd(rng_t, int(N), self.delta_t_psd, self.f_s)
        else:
            delta_t = np.zeros(int(N), dtype=np.float64)
        combined = (
            (self.c_pull if self.trn.enabled else 0.0)
            - (self.pyro_coeff if self.pyroeo.enabled else 0.0)
        ) * delta_t
        if self.tccr.enabled and self.tccr.sigma_tccr > 0.0:
            rng_c = np_generator_from_key(key_tccr)
            tccr_psd = single_pole_psd(self.tccr.var_tccr, self.tau_carrier)
            combined = combined + synthesize_from_psd(
                rng_c, int(N), tccr_psd, self.f_s
            )
        return combined.astype(np.float64), delta_t.astype(np.float64)


def _np_generator_from_key(key) -> np.random.Generator:
    """Deterministic host-side numpy Generator derived from a JAX PRNG key.

    Thin alias of :func:`simulator.colored_noise.np_generator_from_key` (the
    single seeding convention for every host-side noise synthesis); kept
    under its historical name for the pump-noise call sites and tests.
    """
    return np_generator_from_key(key)


def _synthesize_from_onesided_psd(rng: np.random.Generator, n: int, psd_fn,
                                  f_s: float) -> np.ndarray:
    """Real sequence x (n,) float64 with one-sided target PSD ``psd_fn(f)``.

    Thin alias of :func:`simulator.colored_noise.synthesize_from_psd` with
    ``clamp_dc=False`` -- the LEGACY pump-noise semantics, where the
    callables clamp their own DC bin (S(f_0) := S(f_1) for the 1/f parts).
    Same recipe, same draw order, bit-identical output.
    """
    return synthesize_from_psd(rng, n, psd_fn, f_s, clamp_dc=False)


class PumpNoise:
    """Pump-laser frequency noise and RIN (arXiv:2604.05897 Secs. V.B.4–V.B.5).

    Two channels, both sampled once per round trip at f_s = 1/t_r and both
    synthesized HOST-SIDE in float64 (deterministic per JAX key, independent
    of the jax x64 flag):

    Frequency noise (Sec. V.B.4)
        One-sided PSD of the instantaneous laser-frequency deviation δν_p(t):
            S_δν(f) = h₀ + h₋₁/f   [Hz²/Hz]   on f ∈ [1/(N·t_r), 1/(2·t_r)].
        The white plateau h₀ carries the intrinsic Lorentzian linewidth via
        the standard identity Δν_L = π·h₀ (exposed as
        ``lorentzian_linewidth_hz``); h₋₁ is the flicker (1/f) coefficient
        [Hz³/Hz]. Generation: the white part is i.i.d. Gaussian per round
        trip with variance h₀·f_s/2 (one-sided convention:
        var = ∫₀^{f_s/2} S df); the flicker part is FFT-synthesized with
        S_flicker(f_k) = h₋₁ / max(f_k, f₁) — the DC bin is clamped to the
        first bin f₁ = f_s/N, so the (single-bin) DC variance is
        h₋₁/f₁·Δf = h₋₁/N·f_s/N·(N/f_s) = h₋₁ instead of diverging.
        ``sample_freq`` returns 2π·δν_p(t) in rad/s; the SOLVER subtracts it
        from the detuning (δω ≡ ω_res − ω_p, so a positive laser-frequency
        excursion reduces δω).

    RIN (Sec. V.B.5)
        P_in(t) = P̄_in·(1 + ε(t)) with one-sided PSD
            S_ε(f) = 10^(floor_dBc/10) + 10^(excess_dBc/10)·(f_c/f)  (f < f_c)
                   = 10^(floor_dBc/10)                               (f ≥ f_c)
        [1/Hz]. The floor is i.i.d. Gaussian per round trip (variance
        floor·f_s/2) and the excess is FFT-synthesized exactly like the
        flicker part (same DC clamp, zero above the corner). ε is clipped so
        1 + ε ≥ 0; if more than 0.01% of samples clip, a warning reports the
        clipped fraction.

    ``pump_noise_enabled`` = 0/False forces BOTH channels inert regardless of
    the numeric values (samples are exactly zero, PSDs return zero); the value
    ranges are validated only when enabled. Representative values —
    ECDL: h₀ ≈ 3e3 Hz²/Hz (Δν_L ≈ 10 kHz), h₋₁ ≈ 1e10 Hz³/Hz;
    fiber laser: h₀ ≈ 30 Hz²/Hz (Δν_L ≈ 100 Hz).
    """

    def __init__(self, cfg, enabled: bool | None = None):
        self.cfg = cfg
        self.t_r = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.f_s = 1.0 / self.t_r
        if enabled is None:
            enabled = cfg.get("pump_noise_enabled", 0)
        if not (isinstance(enabled, (bool, int, np.integer)) and int(enabled) in (0, 1)):
            raise ValueError(
                f"pump_noise_enabled must be boolean-valued (bool or 0/1), got {enabled!r}."
            )
        self.enabled = bool(int(enabled))

        self.h0 = float(cfg.get("pump_freq_noise_h0_hz2_per_hz", 0.0))
        self.hm1 = float(cfg.get("pump_freq_noise_hm1_hz3_per_hz", 0.0))
        self.rin_floor_dbc = float(cfg.get("pump_rin_floor_dbc_per_hz", -300.0))
        self.rin_excess_dbc = float(cfg.get("pump_rin_excess_dbc_per_hz", -300.0))
        self.rin_corner_hz = float(cfg.get("pump_rin_corner_hz", 1.0e4))

        if self.enabled:
            if self.h0 < 0.0 or self.hm1 < 0.0:
                raise ValueError(
                    f"pump frequency-noise coefficients must be >= 0: "
                    f"h0 = {self.h0!r} Hz²/Hz, h-1 = {self.hm1!r} Hz³/Hz."
                )
            for name, val in (
                ("pump_rin_floor_dbc_per_hz", self.rin_floor_dbc),
                ("pump_rin_excess_dbc_per_hz", self.rin_excess_dbc),
            ):
                if val > -80.0:
                    raise ValueError(
                        f"{name} = {val!r} exceeds -80 dBc/Hz. RIN levels are "
                        f"dB quantities; a value this large is almost "
                        f"certainly a LINEAR spectral density entered where "
                        f"dBc/Hz is expected (physical lasers sit below "
                        f"-80 dBc/Hz)."
                    )
            if self.rin_corner_hz <= 0.0:
                raise ValueError(
                    f"pump_rin_corner_hz must be > 0, got {self.rin_corner_hz!r}."
                )

        # Effective (inert-when-disabled) parameters used by sample_*/psd_*.
        _on = 1.0 if self.enabled else 0.0
        self._h0 = self.h0 * _on
        self._hm1 = self.hm1 * _on
        self._rin_floor_lin = 10.0 ** (self.rin_floor_dbc / 10.0) * _on   # 1/Hz
        self._rin_excess_lin = 10.0 ** (self.rin_excess_dbc / 10.0) * _on  # 1/Hz

        # Intrinsic Lorentzian linewidth from the white plateau: Δν_L = π·h₀.
        self.lorentzian_linewidth_hz = math.pi * self._h0

    # -- closed-form one-sided PSDs (validation targets) ---------------------
    def psd_freq(self, f) -> np.ndarray:
        """One-sided S_δν(f) [Hz²/Hz] of the laser-frequency deviation δν_p."""
        f = np.asarray(f, dtype=np.float64)
        return self._h0 + self._hm1 / np.maximum(f, np.finfo(np.float64).tiny)

    def psd_rin(self, f) -> np.ndarray:
        """One-sided S_ε(f) [1/Hz] of the relative intensity fluctuation ε."""
        f = np.asarray(f, dtype=np.float64)
        excess = np.where(
            f < self.rin_corner_hz,
            self._rin_excess_lin
            * self.rin_corner_hz
            / np.maximum(f, np.finfo(np.float64).tiny),
            0.0,
        )
        return self._rin_floor_lin + excess

    # -- samplers ------------------------------------------------------------
    def sample_freq(self, key, N: int) -> np.ndarray:
        """2π·δν_p(t) [rad/s], shape (N,), float64, one sample per round trip.

        The caller (solver) applies the sign: δω-noise contribution is
        −2π·δν_p because δω ≡ ω_res − ω_p.
        """
        n = int(N)
        if not self.enabled or (self._h0 == 0.0 and self._hm1 == 0.0):
            return np.zeros(n, dtype=np.float64)
        rng = _np_generator_from_key(key)
        dnu = np.zeros(n, dtype=np.float64)
        if self._h0 > 0.0:  # white: var = h0*f_s/2 (one-sided convention)
            dnu += rng.standard_normal(n) * math.sqrt(self._h0 * self.f_s / 2.0)
        if self._hm1 > 0.0 and n >= 2:  # flicker via FFT synthesis
            f1 = self.f_s / n
            dnu += _synthesize_from_onesided_psd(
                rng, n, lambda f: self._hm1 / np.maximum(f, f1), self.f_s
            )
        return 2.0 * math.pi * dnu

    def sample_rin(self, key, N: int) -> np.ndarray:
        """ε(t) (dimensionless), shape (N,), float64, one sample per round trip.

        Clipped so 1 + ε ≥ 0; warns if more than 0.01% of samples clip.
        """
        n = int(N)
        if not self.enabled:
            return np.zeros(n, dtype=np.float64)
        rng = _np_generator_from_key(key)
        eps = np.zeros(n, dtype=np.float64)
        if self._rin_floor_lin > 0.0:
            eps += rng.standard_normal(n) * math.sqrt(
                self._rin_floor_lin * self.f_s / 2.0
            )
        if self._rin_excess_lin > 0.0 and n >= 2:
            f1 = self.f_s / n
            f_c = self.rin_corner_hz
            eps += _synthesize_from_onesided_psd(
                rng,
                n,
                lambda f: np.where(
                    f < f_c,
                    self._rin_excess_lin * f_c / np.maximum(f, f1),
                    0.0,
                ),
                self.f_s,
            )
        n_clip = int(np.count_nonzero(eps < -1.0))
        if n_clip > 1e-4 * n:
            import warnings

            warnings.warn(
                f"PumpNoise.sample_rin: {n_clip}/{n} samples "
                f"({100.0 * n_clip / n:.3f}%) clipped at ε = -1 (P_in >= 0). "
                f"The configured RIN is so large that the Gaussian model is "
                f"physically strained; the clipped sequence is returned.",
                stacklevel=2,
            )
        return np.maximum(eps, -1.0)


def plot_noise_psd() -> None:
    cfg = _load_config()
    total = TotalNoise(cfg)
    trn = total.trn
    pyro = total.pyroeo
    tccr = total.tccr

    N = 100_000
    key = jax.random.PRNGKey(0)
    samples = np.asarray(total.sample(key, N), dtype=np.float32)
    f_emp, p_emp = welch(samples, fs=1.0 / total.t_r, nperseg=1024)

    f = np.logspace(3, 9, 2000)
    s_trn = np.asarray(trn.psd(f))
    s_pyro = np.asarray(pyro.psd(f))
    s_tccr = np.asarray(tccr.psd(f))

    k_b = 1.380649e-23
    c = 299_792_458.0
    eps0 = 8.8541878128e-12
    n0_si3n4 = 2.0
    dn_dt_si3n4 = 2.45e-5
    rho_si3n4 = 3.17e3
    cp_si3n4 = 700.0
    kappa_si3n4 = 3.0
    v_si3n4 = 1e-15
    tau_si3n4 = 5e-6
    t_k = float(cfg.get("T_k", 300.0))
    omega_0 = 2.0 * math.pi * c / float(cfg.get("pump_wavelength_m", 1.55e-6))
    s_delta_t_si3n4 = (
        (4.0 * k_b * t_k**2 * tau_si3n4) / (rho_si3n4 * cp_si3n4 * v_si3n4)
    ) / (1.0 + (2.0 * np.pi * f * tau_si3n4) ** 2)
    s_si3n4 = ((omega_0 / n0_si3n4) * dn_dt_si3n4) ** 2 * s_delta_t_si3n4

    out = Path("analysis/figures")
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.loglog(f, s_trn, label="TFLN TRN", lw=1.8)
    plt.loglog(f, s_pyro, label="TFLN Pyro-EO", lw=1.8)
    plt.loglog(f, s_tccr, label="TFLN TCCR", lw=1.8)
    plt.loglog(f_emp[1:], p_emp[1:], "k:", lw=2.2, label="Empirical total (Welch)")
    plt.loglog(f, s_si3n4, color="gray", lw=1.6, label="Si₃N₄ TRN reference")
    plt.xlim(1e3, 1e9)
    plt.ylim(None, None)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("S_δω(f)  [(rad/s)²/Hz]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "noise_psd_comparison.pdf")
    plt.close()


def validate_noise_models() -> None:
    cfg = _load_config()
    total = TotalNoise(cfg)
    key = jax.random.PRNGKey(0)

    s10k = total.sample(key, 10_000)
    assert s10k.shape == (10_000,)
    assert s10k.dtype == jnp.float32

    s100k = total.sample(key, 100_000)
    std_total = float(jnp.std(s100k))
    sigma_thermal_combined = abs(total.trn.sigma_trn - total.pyroeo.sigma_pyroeo)  # correlated, opposite sign
    expected_sigma = math.sqrt(sigma_thermal_combined**2 + total.tccr.var_tccr)
    assert 0.1 * expected_sigma < std_total < 10.0 * expected_sigma, (
        f"Total noise std {std_total:.3e} outside expected range "
        f"[{0.1*expected_sigma:.3e}, {10.0*expected_sigma:.3e}]"
    )

    trn_std = float(jnp.std(total.trn.sample(jax.random.PRNGKey(1), 100_000)))
    tccr_std = float(jnp.std(total.tccr.sample(jax.random.PRNGKey(2), 100_000)))
    
    if 0.0 < tccr_std <= trn_std:   # SiN has no TCCR (tccr_std == 0); not a warning condition
        import warnings
        warnings.warn(
            f"TRN ({trn_std:.3e}) >= TCCR ({tccr_std:.3e}) for current config. "
            f"TCCR is not the dominant noise source. For TFLN devices where TCCR should "
            f"dominate, increase surface_state_density_per_m2 or verify A_eff.",
            stacklevel=2,
        )

    tccr_samples = np.asarray(total.tccr.sample(jax.random.PRNGKey(3), 200_000), dtype=np.float64)
    if float(np.std(tccr_samples)) > 0.0:   # only when TCCR is active (chi2 platforms); SiN -> skip
        r1 = np.corrcoef(tccr_samples[:-1], tccr_samples[1:])[0, 1]
        tau_est = -total.t_r / np.log(r1)
        tau_target = float(cfg.get("tau_carrier_s", 1.0e-7))
        assert abs(tau_est - tau_target) / tau_target < 0.5


if __name__ == "__main__":
    validate_noise_models()
    plot_noise_psd()
