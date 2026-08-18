"""Noise-channel implementations for the stochastic-LLE benchmark.

Each class here turns a ``physical_parameters`` mapping into (a) a closed-form
one-sided power spectral density and (b) a sampler that draws a realization of
that spectrum, one sample per cavity round trip at ``f_s = 1/t_r``. The solver
consumes the realizations; the PSDs are the validation targets.

Channels
--------
:class:`TRNoise`
    Thermorefractive noise: a thermodynamic temperature fluctuation dT(t) [K]
    pulled onto the resonance through
    ``C_pull = (omega0/n0)*(dn_dT + n0*alpha_L)`` [rad/(s K)].
:class:`PyroEONoise`
    Pyro-electric charge release converted to a resonance shift by the
    electro-optic effect, driven by the SAME dT(t) as :class:`TRNoise`.
:class:`TCCRNoise`
    Thermal carrier / surface-state noise: surface-charge shot noise seen
    through the same electro-optic coefficient, on an INDEPENDENT stream.
:class:`TotalNoise`
    Their sum, with the shared-temperature correlation between TRN and Pyro-EO
    handled explicitly rather than by adding independent draws.
:class:`PumpNoise`
    Pump-laser frequency noise [Hz**2/Hz] and relative intensity noise [1/Hz],
    arXiv:2604.05897v1 Secs. V.B.4--V.B.5.

Conventions
-----------
Every detuning-noise channel returns a contribution to ``delta_omega``
[rad/s] under the repository's sign convention
``delta_omega = omega_res - omega_pump``. Every sampler is deterministic in its
JAX PRNG key: colored models synthesize HOST-SIDE in float64 through
:mod:`simulator.colored_noise`, while the historical ``single_pole`` path keeps
the traced AR(1) recursion and its float32 output, which is BIT-IDENTICAL to
the pre-colored-noise code.

A disabled channel returns exact zeros WITHOUT drawing from its key. That is
safe for every other channel because JAX PRNG is functional -- consuming a key
has no side effect -- but it makes the split ladders in
:meth:`TotalNoise.sample_with_delta_t` load-bearing: they are unconditional, so
toggling one channel can never shift another channel's stream.
"""

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

    Parameters
    ----------
    cfg : dict
        The ``physical_parameters`` mapping. Keys read (with units and
        defaults): ``fsr_hz`` [Hz, 2e11], ``pump_wavelength_m`` [m, 1.55e-6],
        ``n0`` [dimensionless, 2.2], ``dn_dT_per_k`` [1/K, 4e-5],
        ``alpha_L_per_k`` [1/K, 0.0], ``tau_th_s`` [s, 5e-6],
        ``rho_kg_per_m3`` [kg/m**3, 4640], ``Cp_j_per_kg_k`` [J/(kg K), 700],
        ``mode_volume_m3`` [m**3, 1e-15], ``kappa_th_w_per_m_k`` [W/(m K), 4.6],
        ``T_k`` [K, 300], ``trn_psd_model`` [str, ``'single_pole'``] and, for
        the ``kondratiev_gorodetsky`` model, ``trn_R_m`` / ``trn_da_m`` /
        ``trn_db_m`` [m].
    enabled : bool or None, optional
        Explicit channel switch. ``None`` (default) reproduces the historical
        IMPLICIT gating exactly: the channel is live iff the thermodynamic
        variance is non-zero, i.e. iff ``T_k > 0``.
    ar1_stationary_init : bool, optional
        Start the legacy AR(1) generator from its stationary distribution
        instead of from ``x0 = 0``. Default ``False``, which is bit-identical
        to the historical behaviour. Meaningful only for
        ``trn_psd_model = 'single_pole'`` -- the colored models are stationary
        from sample 0 already.

    Attributes
    ----------
    c_pull : float
        Frequency pull ``domega/dT`` [rad/(s K)].
    var_delta_t : float
        Thermodynamic temperature variance ``k_B*T**2/(rho*C*V)`` [K**2]
        (arXiv:2604.05897v1 Eq. 129).
    sigma_trn : float
        Standard deviation of the detuning noise, ``c_pull*sqrt(var_delta_t)``
        [rad/s].
    delta_t_psd : callable
        One-sided ``S_dT(f)`` [K**2/Hz] of the selected model.
    var_delta_t_target : float
        Variance [K**2] the selected model's spectrum integrates to over the
        synthesis band.
    f_s : float
        Sample rate ``1/t_r`` [Hz].
    enabled : bool
        Resolved channel switch.

    Raises
    ------
    ValueError
        If ``trn_psd_model`` is not one of ``single_pole``,
        ``kondratiev_gorodetsky``, ``csv``; if ``enabled`` is not
        boolean-valued; if the ``kondratiev_gorodetsky`` geometry keys are
        missing or non-positive; or if ``trn_psd_model = 'csv'`` without a
        usable ``trn_psd_csv_path`` / ``trn_csv_units``. The PSD is built
        EAGERLY in ``__init__`` so a config error surfaces at construction
        rather than at first sample.

    Notes
    -----
    In every model the frequency-pull map is

        domega(t) = C_pull * dT(t),
        C_pull    = (omega0/n0) * (dn_dT + n0*alpha_L_per_k),

    where the optional ``alpha_L_per_k`` folds the paper's "dimensional
    fluctuation" companion of thermorefractive noise into the pull coefficient.
    With ``alpha_L_per_k = 0`` this is bit-identical to the historical
    ``(omega0/n0)*dn_dT`` (``x + n0*0.0 == x`` in IEEE arithmetic).

    ``trn_psd_model`` selects the temperature-fluctuation spectrum:

    * ``single_pole`` (default) -- the historical AR(1) generator; the sampled
      stream is BYTE-COMPATIBLE with the pre-colored-noise code;
    * ``kondratiev_gorodetsky`` -- analytic whispering-gallery PSD,
      arXiv:2604.05897v1 Eq. 130, variance renormalized to Eq. 129 (see
      :func:`simulator.colored_noise.kondratiev_gorodetsky_psd`);
    * ``csv`` -- measured or FEM-tabulated PSD, Huang et al. (2019) style.

    Examples
    --------
    >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6}
    >>> trn = TRNoise(cfg)
    >>> trn.enabled, trn.is_colored
    (True, False)
    >>> print(f"{trn.c_pull:.4e} rad/(s K)")
    2.2096e+10 rad/(s K)
    >>> print(f"{trn.var_delta_t:.4e} K^2")
    3.8257e-10 K^2

    ``T_k = 0`` is the repository's deterministic noise-off convention:

    >>> TRNoise({**cfg, "T_k": 0.0}).enabled
    False
    """

    def __init__(self, cfg, enabled: bool | None = None,
                 ar1_stationary_init: bool = False):
        """Resolve the pull coefficient and build the temperature PSD eagerly.

        See :class:`TRNoise` for the parameter table, the units of every config
        key it reads, and the attributes this sets.

        Raises
        ------
        ValueError
            If ``enabled`` is not boolean-valued, or if the selected
            ``trn_psd_model`` is unknown or its required keys are missing.

        Notes
        -----
        The PSD is built here rather than lazily so that a configuration error
        surfaces at construction time, where the traceback names the config,
        instead of at the first sample deep inside a solve.
        """
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
        """Whether this instance uses the PSD engine rather than the AR(1) path.

        Returns
        -------
        bool
            ``True`` for ``trn_psd_model`` in
            ``{'kondratiev_gorodetsky', 'csv'}`` -- the host-side float64 FFT
            synthesis; ``False`` for ``'single_pole'``, the traced float32 AR(1)
            recursion that is bit-identical to the pre-colored-noise code.

        Notes
        -----
        This is the single predicate that decides both the sampler branch and
        the output dtype, so it is a property rather than a stored flag: it
        cannot drift from ``psd_model``.

        Examples
        --------
        >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6}
        >>> TRNoise(cfg).is_colored
        False
        >>> TRNoise({**cfg, "trn_psd_model": "kondratiev_gorodetsky",
        ...          "trn_R_m": 1e-4, "trn_da_m": 2e-6,
        ...          "trn_db_m": 1e-6}).is_colored
        True
        """
        return self.psd_model != "single_pole"

    @property
    def _sample_dtype(self):
        """dtype of :meth:`sample` — float32 on the legacy AR(1) path, else float64."""
        return jnp.float64 if self.is_colored else jnp.float32

    def sample_delta_t(self, key, N) -> np.ndarray:
        """Draw the underlying temperature-fluctuation sequence dT(t).

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key. Folded into a numpy generator by
            :func:`simulator.colored_noise.np_generator_from_key`.
        N : int
            Number of round trips [samples], sampled at ``f_s = 1/t_r``.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)``, dtype ``float64``, units K. A disabled channel
            returns exact zeros WITHOUT drawing from ``key``.

        Raises
        ------
        ImportError
            Propagated from :func:`~simulator.colored_noise.np_generator_from_key`
            if JAX is unavailable.

        Notes
        -----
        Available for EVERY model -- ``single_pole`` uses the Lorentzian
        spectral twin of its AR(1) recursion -- and stationary from sample 0.
        This is the sequence the segment-continuity path
        (``legacy_segment_noise = 0``) slices, and it is NOT the
        byte-compatible AR(1) stream; that one lives in :meth:`sample`.

        Examples
        --------
        >>> import jax
        >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6}
        >>> dT = TRNoise(cfg).sample_delta_t(jax.random.PRNGKey(0), 1024)
        >>> print(dT.shape, dT.dtype)
        (1024,) float64
        >>> TRNoise({**cfg, "T_k": 0.0}).sample_delta_t(jax.random.PRNGKey(0), 3)
        array([0., 0., 0.])
        """
        if not self.enabled:
            return np.zeros(int(N), dtype=np.float64)
        rng = np_generator_from_key(key)
        return synthesize_from_psd(rng, int(N), self.delta_t_psd, self.f_s)

    def sample(self, key, N) -> jnp.ndarray:
        """Draw the thermorefractive detuning-noise sequence domega(t).

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        N : int
            Number of round trips [samples].

        Returns
        -------
        jax.Array
            Shape ``(N,)``, units rad/s. dtype ``float32`` on the
            ``single_pole`` AR(1) path (the historical stream) and ``float64``
            on the colored paths. A disabled channel returns exact zeros
            WITHOUT drawing from ``key``.

        Raises
        ------
        ImportError
            Propagated from the host-side generator on the colored paths.

        Notes
        -----
        ``domega(t) = C_pull * dT(t)``. The ``single_pole`` branch runs the
        traced AR(1) recursion, so it is vmap-safe and bit-identical to the
        pre-colored-noise code; the colored branches synthesize on the HOST and
        must be looped over trajectories rather than vmapped.

        Examples
        --------
        >>> import jax
        >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6}
        >>> x = TRNoise(cfg).sample(jax.random.PRNGKey(0), 8)
        >>> print(x.shape, x.dtype)
        (8,) float32
        """
        if not self.enabled:
            return _zeros(N, self._sample_dtype)
        if not self.is_colored:
            return _ar1_samples(key, N, self.tau_th, self.sigma_trn, self.t_r,
                                self.ar1_stationary_init)
        return jnp.asarray(self.c_pull * self.sample_delta_t(key, N),
                           dtype=jnp.float64)

    def psd(self, f) -> jnp.ndarray:
        """Evaluate the closed-form one-sided detuning PSD.

        Parameters
        ----------
        f : array_like
            Frequency [Hz], scalar or array.

        Returns
        -------
        jax.Array
            ``S_domega(f)`` [(rad/s)**2/Hz], same shape as ``f``. A disabled
            channel returns exact zeros.

        Raises
        ------
        ValueError
            Propagated from the CSV interpolator for a malformed table.

        Notes
        -----
        This is the VALIDATION TARGET the sampled stream is checked against; it
        is a closed form, not an estimate. On the ``single_pole`` path

            S_domega(f) = C_pull**2 * 4*k_B*T**2*tau_th/(rho*C*V)
                          / (1 + (2*pi*f*tau_th)**2),

        the Lorentzian whose integral is the Eq. 129 thermodynamic variance
        (arXiv:2604.05897v1). On the colored paths it is ``C_pull**2`` times the
        selected ``S_dT(f)``.

        Examples
        --------
        >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6}
        >>> trn = TRNoise(cfg)
        >>> print(f"{float(trn.psd(0.0)):.4e}")       # (rad/s)^2/Hz at DC
        3.7355e+06
        >>> bool(trn.psd(0.0) > trn.psd(1e6))         # Lorentzian roll-off
        True
        >>> float(TRNoise({**cfg, "T_k": 0.0}).psd(1e3))
        0.0
        """
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
    """Pyro-electric / electro-optic detuning noise, driven by the TRN temperature.

    Parameters
    ----------
    cfg : dict
        The ``physical_parameters`` mapping. Beyond the thermodynamic keys
        :class:`TRNoise` reads, this channel uses ``eo_r33_m_per_v`` [m/V,
        3.1e-11], ``pyroelectric_coeff_c_per_m2_k`` [C/(m**2 K), 9.6e-2],
        ``eps_r_z`` [dimensionless, 28.0] and the thin-film stack geometry
        ``t_ln_m`` [m, 4e-7], ``t_clad_top_m`` [m, 1e-6], ``t_clad_bot_m``
        [m, 2e-6], ``eps_r_clad_top`` [dimensionless, 1.0], ``eps_r_clad_bot``
        [dimensionless, 3.9].
    enabled : bool or None, optional
        Explicit channel switch. ``None`` (default) reproduces the legacy rule,
        which is the SAME as :class:`TRNoise`: live iff ``T_k > 0``.

    Attributes
    ----------
    eps_r_eff : float
        Effective relative permittivity [dimensionless] after the 1-D
        dielectric-boundary screening of the thin-film stack.
    pyro_coeff : float
        Frequency pull ``domega/dT`` [rad/(s K)] of this channel.
    sigma_pyroeo : float
        Standard deviation of the resulting detuning noise [rad/s].
    delta_t_psd : callable
        The SAME temperature spectrum family :class:`TRNoise` uses, selected by
        ``trn_psd_model``.
    enabled : bool
        Resolved channel switch.

    Raises
    ------
    ValueError
        If ``enabled`` is not boolean-valued, or from the shared PSD builder for
        a malformed ``trn_psd_model`` configuration.

    Notes
    -----
    A temperature fluctuation releases pyro-electric surface charge, whose field
    shifts the resonance through the electro-optic coefficient:

        domega(t) = [omega0 * n0**2 * r33 * p / (2*eps0*eps_r_eff)] * dT(t).

    The screening factor ``eps_r_eff`` is a 1-D approximation of the dielectric
    boundary conditions in the thin-film stack: the field extending into the
    claddings dilutes ``eps_r_z``, so ``eps_r_eff = eps_r_z +
    eps_r_top*(t_top/t_ln) + eps_r_bot*(t_bot/t_ln)``.

    This channel is driven by the SAME dT(t) realization as :class:`TRNoise` --
    they are one thermodynamic fluctuation seen through two pull coefficients,
    which is why :class:`TotalNoise` adds them coherently (and, for z-cut TFLN
    with an air top cladding, with opposite sign) rather than in quadrature.

    A zero pyro or electro-optic coefficient -- ``r33 = 0`` on centrosymmetric
    SiN -- independently makes every sample exactly 0.0. That is a MATERIAL
    fact, not a switch, so it is deliberately not folded into ``enabled``.

    Examples
    --------
    >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6}
    >>> pyro = PyroEONoise(cfg)
    >>> print(f"{pyro.eps_r_eff:.1f}")           # 28 + 1.0*2.5 + 3.9*5.0
    50.0
    >>> print(f"{pyro.pyro_coeff:.4e} rad/(s K)")
    1.9770e+13 rad/(s K)
    >>> PyroEONoise({**cfg, "T_k": 0.0}).enabled
    False
    """

    def __init__(self, cfg, enabled: bool | None = None):
        """Resolve the electro-optic pull coefficient and the stack screening.

        See :class:`PyroEONoise` for the parameter table, the units of every
        config key it reads, and the attributes this sets.

        Raises
        ------
        ValueError
            If ``enabled`` is not boolean-valued, or from the shared PSD
            builder for a malformed ``trn_psd_model`` configuration.

        Notes
        -----
        Reads the SAME ``trn_psd_model`` selection as :class:`TRNoise`, because
        the channel is driven by the same temperature fluctuation and must not
        be able to disagree with it about that fluctuation's spectrum.
        """
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
        """Whether this instance uses the PSD engine rather than the AR(1) path.

        Returns
        -------
        bool
            ``True`` unless ``trn_psd_model`` is ``'single_pole'``. The channel
            inherits the TRN spectrum selection because it is driven by the same
            temperature fluctuation.

        Examples
        --------
        >>> PyroEONoise({"fsr_hz": 2.0e11, "T_k": 300.0}).is_colored
        False
        """
        return self.psd_model != "single_pole"

    @property
    def _sample_dtype(self):
        """dtype of :meth:`sample` — float32 on the legacy AR(1) path, else float64."""
        return jnp.float64 if self.is_colored else jnp.float32

    def sample(self, key, N) -> jnp.ndarray:
        """Draw the pyro-electric/electro-optic detuning-noise sequence.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        N : int
            Number of round trips [samples].

        Returns
        -------
        jax.Array
            Shape ``(N,)``, units rad/s; ``float32`` on the ``single_pole``
            path and ``float64`` on the colored paths. A disabled channel
            returns exact zeros WITHOUT drawing from ``key``.

        Raises
        ------
        ImportError
            Propagated from the host-side generator on the colored paths.

        Notes
        -----
        Sampling this class STANDALONE draws an independent temperature
        realization, which is correct only for a per-channel PSD check. In a
        solve, the physically correct coupling -- one shared dT(t) behind both
        pull coefficients -- is provided by
        :meth:`TotalNoise.sample_with_delta_t`.

        Examples
        --------
        >>> import jax
        >>> x = PyroEONoise({"fsr_hz": 2.0e11, "T_k": 300.0}).sample(
        ...     jax.random.PRNGKey(0), 8)
        >>> print(x.shape, x.dtype)
        (8,) float32
        """
        if not self.enabled:
            return _zeros(N, self._sample_dtype)
        if not self.is_colored:
            return _ar1_samples(key, N, self.tau_th, self.sigma_pyroeo, self.t_r)
        rng = np_generator_from_key(key)
        delta_t = synthesize_from_psd(rng, int(N), self.delta_t_psd, self.f_s)
        return jnp.asarray(self.pyro_coeff * delta_t, dtype=jnp.float64)

    def psd(self, f) -> jnp.ndarray:
        """Evaluate the closed-form one-sided pyro-electric/EO detuning PSD.

        Parameters
        ----------
        f : array_like
            Frequency [Hz], scalar or array.

        Returns
        -------
        jax.Array
            ``S_domega(f)`` [(rad/s)**2/Hz], same shape as ``f``; exact zeros
            when the channel is disabled.

        Raises
        ------
        ValueError
            Propagated from the CSV interpolator for a malformed table.

        Notes
        -----
        ``pyro_coeff**2`` times the same ``S_dT(f)`` [K**2/Hz] that
        :meth:`TRNoise.psd` scales by ``c_pull**2`` -- the two channels differ
        only in their pull coefficient.

        Examples
        --------
        >>> pyro = PyroEONoise({"fsr_hz": 2.0e11, "T_k": 300.0})
        >>> bool(pyro.psd(0.0) > pyro.psd(1e6))
        True
        >>> float(PyroEONoise({"T_k": 0.0}).psd(1e3))
        0.0
        """
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
    """Thermal-carrier / surface-state detuning noise (independent stream).

    Parameters
    ----------
    cfg : dict
        The ``physical_parameters`` mapping. Keys read (with units and
        defaults): ``fsr_hz`` [Hz, 2e11], ``pump_wavelength_m`` [m, 1.55e-6],
        ``tau_carrier_s`` [s, 1e-7], ``T_k`` [K, 300],
        ``surface_state_density_per_m2`` [1/m**2, 1e16], ``eo_r33_m_per_v``
        [m/V, 3.1e-11], ``n0`` [dimensionless, 2.2], ``eps_r_z``
        [dimensionless, 28.0], ``effective_mode_area_m2`` [m**2, 1e-12],
        ``t_ln_m`` [m, 4e-7] and ``intrinsic_q`` [dimensionless, 2e6].
    enabled : bool or None, optional
        Explicit channel switch. ``None`` (default) reproduces the legacy rule:
        live iff ``sigma_tccr > 0``, i.e. iff the electro-optic coefficient is
        non-zero.

    Attributes
    ----------
    s0_tccr : float
        Zero-frequency PSD ``S_domega(0)`` [(rad/s)**2/Hz].
    var_tccr : float
        Stationary variance ``s0_tccr/(2*tau_carrier)`` [(rad/s)**2].
    sigma_tccr : float
        Standard deviation [rad/s].
    enabled : bool
        Resolved channel switch.

    Raises
    ------
    ValueError
        If ``enabled`` is not boolean-valued.

    Warns
    -----
    UserWarning
        If ``sigma_tccr`` is non-zero but outside the physically plausible
        ``[1e4, 1e11]`` rad/s window for a chi(2) platform, or if it exceeds the
        cavity linewidth estimate ``kappa ~ 2*omega0/Q_i`` -- in which case the
        channel is non-perturbative and would destabilize every soliton, so the
        configuration should be calibrated against a measured noise floor before
        it is used to generate data.

    Notes
    -----
    Surface-state carriers fluctuate by shot noise about their equilibrium
    number ``N_s_eq = n_s*A_eff``; each carrier contributes a field
    ``E = e/(eps0*eps_r_eff*A_eff)`` [V/m], which the electro-optic effect turns
    into a resonance shift

        domega/dN_s = -omega0 * n0**2 * r33 * E_per_carrier / 2   [rad/s].

    The film thickness ``t_ln`` deliberately does NOT enter that expression --
    ``E_per_carrier`` is already a field. With an exponential carrier
    autocorrelation of time ``tau_carrier`` the spectrum is the Lorentzian

        S_domega(f) = s0 / (1 + (2*pi*f*tau_carrier)**2),
        s0 = (domega/dN_s)**2 * N_s_eq * 2*tau_carrier.

    This channel is NOT gated by ``T_k`` -- its variance carries no ``T_k``
    factor -- which is exactly why the ``T_k = 0`` "noise-off" convention never
    silenced it on chi(2) platforms, and why the explicit switch exists.

    Examples
    --------
    A centrosymmetric platform (SiN, ``r33 = 0``) has no such channel at all:

    >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6}
    >>> sin = TCCRNoise({**cfg, "eo_r33_m_per_v": 0.0})
    >>> sin.sigma_tccr, sin.enabled
    (0.0, False)

    A chi(2) platform does:

    >>> tccr = TCCRNoise({**cfg, "surface_state_density_per_m2": 1e14})
    >>> print(f"{tccr.sigma_tccr:.4e} rad/s")
    5.8918e+08 rad/s
    """

    def __init__(self, cfg, enabled: bool | None = None):
        """Derive the per-carrier frequency shift and the resulting Lorentzian.

        See :class:`TCCRNoise` for the parameter table, the units of every
        config key it reads, and the attributes this sets.

        Raises
        ------
        ValueError
            If ``enabled`` is not boolean-valued.

        Warns
        -----
        UserWarning
            If the resulting ``sigma_tccr`` is non-zero but implausible, or
            exceeds the cavity linewidth. Both checks run HERE, at
            construction, so a mis-set surface-state density is caught before a
            campaign runs rather than after it.
        """
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
        """Draw the thermal-carrier detuning-noise sequence.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key. This channel's stream is INDEPENDENT of the shared
            thermal stream, so in :class:`TotalNoise` it is fed the second
            subkey of an unconditional split.
        N : int
            Number of round trips [samples].

        Returns
        -------
        jax.Array
            Shape ``(N,)``, dtype ``float32``, units rad/s. A disabled channel
            returns exact zeros WITHOUT drawing from ``key``.

        Notes
        -----
        Always the traced AR(1) recursion with correlation time
        ``tau_carrier``: unlike the thermal channels, this one has no
        pluggable-PSD variant, because its spectrum is a genuine Lorentzian
        rather than an asymptotic fit.

        Examples
        --------
        >>> import jax
        >>> cfg = {"fsr_hz": 2.0e11, "surface_state_density_per_m2": 1e14}
        >>> x = TCCRNoise(cfg).sample(jax.random.PRNGKey(0), 8)
        >>> print(x.shape, x.dtype)
        (8,) float32
        """
        if not self.enabled:
            return _zeros(N, jnp.float32)
        return _ar1_samples(key, N, self.tau_carrier, self.sigma_tccr, self.t_r)

    def psd(self, f) -> jnp.ndarray:
        """Evaluate the closed-form one-sided thermal-carrier detuning PSD.

        Parameters
        ----------
        f : array_like
            Frequency [Hz], scalar or array.

        Returns
        -------
        jax.Array
            ``S_domega(f) = s0/(1 + (2*pi*f*tau_carrier)**2)``
            [(rad/s)**2/Hz], same shape as ``f``; exact zeros when disabled.

        Notes
        -----
        The corner sits at ``1/(2*pi*tau_carrier)`` [Hz] -- with the default
        ``tau_carrier = 100 ns`` that is ~1.6 MHz, three decades above the
        thermorefractive corner, which is what makes the two channels separable
        in a measured spectrum.

        Examples
        --------
        >>> cfg = {"fsr_hz": 2.0e11, "surface_state_density_per_m2": 1e14}
        >>> tccr = TCCRNoise(cfg)
        >>> print(f"{float(tccr.psd(0.0)):.4e}")
        6.9427e+10
        >>> bool(tccr.psd(0.0) > tccr.psd(1e9))
        True
        """
        if not self.enabled:
            return _zeros_psd(f)
        return self.s0_tccr / (1.0 + (2.0 * jnp.pi * f * self.tau_carrier) ** 2)


class TotalNoise:
    """Combined TRN + Pyro-EO + TCCR detuning noise with correct correlations.

    Parameters
    ----------
    cfg : dict
        The ``physical_parameters`` mapping, passed through to each
        sub-channel; see :class:`TRNoise`, :class:`PyroEONoise` and
        :class:`TCCRNoise` for the keys and units each one reads.
    noise_config : NoiseConfig or None, optional
        Optional :class:`simulator.noise_config.NoiseConfig`. ``None`` (the
        default) leaves every sub-channel to resolve its own switch from the
        legacy rule, so behaviour is EXACTLY as before this parameter existed.
        When supplied, each sub-channel's ``enabled`` comes from the matching
        field (``trn``, ``pyro_eo``, ``tccr``) and ``trn_ar1_stationary_init``
        selects the start-up policy of the shared thermal AR(1) stream.

    Attributes
    ----------
    trn, pyroeo, tccr : TRNoise, PyroEONoise, TCCRNoise
        The sub-channels.
    c_pull : float
        Thermorefractive frequency pull [rad/(s K)].
    pyro_coeff : float
        Pyro-electric/EO frequency pull [rad/(s K)].
    delta_t_psd : callable
        The shared temperature spectrum ``S_dT(f)`` [K**2/Hz].
    f_s : float
        Sample rate ``1/t_r`` [Hz].

    Raises
    ------
    ValueError
        Propagated from any sub-channel's construction.

    Warns
    -----
    UserWarning
        If ``noise_config.fsr`` is on while ``noise_config.trn`` is off: the FSR
        channel is driven by the same dT(t) realization, so that combination
        produces identically zero FSR noise rather than the intended channel.

    Notes
    -----
    The thermorefractive and pyro-electric/EO channels share ONE temperature
    sequence dT(t) [K] -- they are the same thermodynamic fluctuation seen
    through two different pull coefficients -- whatever ``trn_psd_model``
    generates that sequence. Adding two independent draws instead would get the
    total variance wrong, because for z-cut TFLN with an air top cladding the
    pyro-electric pull PARTIALLY CANCELS the thermorefractive one:

        domega(t) = (C_pull - pyro_coeff) * dT(t) + domega_TCCR(t).

    The thermal-carrier channel (zero for SiN) keeps its own independent AR(1)
    stream, fed by the second subkey of an unconditional two-way split.

    Three sampling surfaces, deliberately kept distinct:

    * :meth:`sample` -- the historical ``(N,)`` float32 combined sequence,
      BYTE-COMPATIBLE with the pre-colored-noise code for
      ``trn_psd_model = 'single_pole'``;
    * :meth:`sample_with_delta_t` -- the same combined sequence PLUS the
      underlying dT(t), so the FSR-noise channel
      ``dD1(t) = (D1/omega0)*C_pull*dT(t)`` can reuse the identical realization;
    * :meth:`sample_full_with_delta_t` -- the host-side float64 PSD-synthesized
      path for ALL models, stationary from sample 0, used by the
      ``legacy_segment_noise = 0`` full-trajectory-then-slice mode.

    Examples
    --------
    >>> import jax
    >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6,
    ...        "eo_r33_m_per_v": 0.0}
    >>> total = TotalNoise(cfg)
    >>> x = total.sample(jax.random.PRNGKey(0), 8)
    >>> print(x.shape, x.dtype)
    (8,) float32
    >>> total.tccr.enabled                       # r33 = 0 -> no carrier channel
    False
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
        """Whether the shared thermal stream comes from the PSD engine.

        Returns
        -------
        bool
            ``True`` unless ``trn_psd_model`` is ``'single_pole'``. Mirrors
            :attr:`TRNoise.is_colored`, from which it is derived.

        Examples
        --------
        >>> TotalNoise({"fsr_hz": 2.0e11, "T_k": 300.0,
        ...             "surface_state_density_per_m2": 1e14}).is_colored
        False
        """
        return self.psd_model != "single_pole"

    def sample_with_delta_t(self, key, N):
        """Draw the combined detuning noise together with the temperature that drove it.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key, split UNCONDITIONALLY into ``(key_thermal,
            key_tccr)``.
        N : int
            Number of round trips [samples].

        Returns
        -------
        combined : jax.Array
            Shape ``(N,)``, dtype ``float32``, units rad/s: the total detuning
            noise ``(C_pull - pyro_coeff)*dT(t) + domega_TCCR(t)``, with each
            coefficient zeroed if its channel is disabled.
        delta_t : jax.Array
            Shape ``(N,)``, units K: the shared temperature realization.
            ``float64`` on the colored path, and on the legacy path the float32
            AR(1) stream upcast to float64 whenever ``jax_enable_x64`` is on.

        Raises
        ------
        ImportError
            Propagated from the host-side generator on the colored paths.

        Notes
        -----
        The two-way split at the top of this method is load-bearing and must
        stay UNCONDITIONAL: it happens regardless of which channels are enabled,
        so toggling one channel can never shift another channel's random stream.
        Never move it inside a branch, never change its arity, and never reorder
        the subkeys.

        For ``single_pole`` the combined sequence is bit-identical to
        :meth:`sample` -- identical key split, identical arithmetic; dT is
        merely also returned. The legacy branch is fully traceable and
        vmap-safe; the colored models synthesize on the HOST and must be looped
        over trajectories.

        Returning dT alongside the detuning noise is what lets the FSR channel
        use the SAME realization, as it physically must: the repetition-rate
        fluctuation and the resonance shift are two consequences of one
        temperature excursion.

        Examples
        --------
        >>> import jax
        >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6,
        ...        "surface_state_density_per_m2": 1e14}
        >>> combined, dT = TotalNoise(cfg).sample_with_delta_t(
        ...     jax.random.PRNGKey(0), 16)
        >>> print(combined.shape, combined.dtype, dT.shape)
        (16,) float32 (16,)
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
        """Draw the combined detuning-noise sequence.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        N : int
            Number of round trips [samples].

        Returns
        -------
        jax.Array
            Shape ``(N,)``, dtype ``float32``, units rad/s. For
            ``trn_psd_model = 'single_pole'`` this stream is BYTE-COMPATIBLE
            with the pre-colored-noise code.

        Raises
        ------
        ImportError
            Propagated from the host-side generator on the colored paths.

        Notes
        -----
        A thin projection of :meth:`sample_with_delta_t`, kept so that callers
        that do not need the temperature realization are not tempted to build
        their own combination and get the TRN/Pyro-EO correlation wrong.

        Examples
        --------
        >>> import jax
        >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6,
        ...        "surface_state_density_per_m2": 1e14}
        >>> total = TotalNoise(cfg)
        >>> key = jax.random.PRNGKey(0)
        >>> bool((total.sample(key, 8) == total.sample_with_delta_t(key, 8)[0]).all())
        True
        """
        return self.sample_with_delta_t(key, N)[0]

    def sample_full_with_delta_t(self, key, N):
        """Draw the host-side float64 (combined, dT) pair for the segment-continuity path.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key, split UNCONDITIONALLY into ``(key_thermal,
            key_tccr)`` exactly as in :meth:`sample_with_delta_t`.
        N : int
            Length of the FULL trajectory [samples], not of one segment.

        Returns
        -------
        combined : numpy.ndarray
            Shape ``(N,)``, dtype ``float64``, units rad/s.
        delta_t : numpy.ndarray
            Shape ``(N,)``, dtype ``float64``, units K.

        Raises
        ------
        ImportError
            Propagated from :func:`~simulator.colored_noise.np_generator_from_key`.

        Notes
        -----
        EVERY model synthesizes from its PSD here -- ``single_pole`` from the
        Lorentzian twin of its AR(1) recursion -- so the sequence is stationary
        from sample 0. A full trajectory generated once up front and then sliced
        per segment therefore has no boundary decorrelation transient, which is
        the whole point of the ``legacy_segment_noise = 0`` mode: with the
        legacy per-segment regeneration, every segment boundary restarts the
        AR(1) burn-in and injects a spurious transient into the statistics.

        The thermal-carrier channel also synthesizes from its single-pole PSD
        here, on an independent stream from the second subkey.

        Examples
        --------
        >>> import jax
        >>> cfg = {"fsr_hz": 2.0e11, "T_k": 300.0, "pump_wavelength_m": 1.55e-6,
        ...        "surface_state_density_per_m2": 1e14}
        >>> combined, dT = TotalNoise(cfg).sample_full_with_delta_t(
        ...     jax.random.PRNGKey(0), 64)
        >>> print(combined.shape, combined.dtype, dT.dtype)
        (64,) float64 float64
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

    Parameters
    ----------
    cfg : dict
        The ``physical_parameters`` mapping. Keys read (with units and
        defaults): ``fsr_hz`` [Hz, 2e11], ``pump_noise_enabled``
        [0/1, 0], ``pump_freq_noise_h0_hz2_per_hz`` [Hz**2/Hz, 0.0],
        ``pump_freq_noise_hm1_hz3_per_hz`` [Hz**3/Hz, 0.0],
        ``pump_rin_floor_dbc_per_hz`` [dBc/Hz, -300],
        ``pump_rin_excess_dbc_per_hz`` [dBc/Hz, -300] and
        ``pump_rin_corner_hz`` [Hz, 1e4].
    enabled : bool or None, optional
        Explicit master switch for BOTH channels. ``None`` (default) reads
        ``pump_noise_enabled`` from ``cfg``.

    Attributes
    ----------
    t_r : float
        Round-trip time [s].
    f_s : float
        Sample rate ``1/t_r`` [Hz].
    lorentzian_linewidth_hz : float
        Intrinsic linewidth ``pi*h0`` [Hz] implied by the white plateau; 0 when
        the channel is disabled.
    enabled : bool
        Resolved master switch.

    Raises
    ------
    ValueError
        If ``pump_noise_enabled`` is not boolean-valued; or, when enabled, if
        either frequency-noise coefficient is negative, if
        ``pump_rin_corner_hz <= 0``, or if either RIN level exceeds
        -80 dBc/Hz -- a value that large is almost certainly a LINEAR spectral
        density entered where a dB quantity is expected, since physical lasers
        sit below -80 dBc/Hz.

    Notes
    -----
    Disabling the channel does not merely skip the samplers: the effective
    coefficients are multiplied by zero at construction, so the PSD accessors
    return exact zeros too and ``lorentzian_linewidth_hz`` reports 0 Hz. That
    keeps "what spectrum did this run have?" answerable from the same object
    whether or not the channel was on.

    Both channels are sampled once per round trip and synthesized HOST-SIDE in
    float64, so they are deterministic per JAX key and independent of the
    ``jax_enable_x64`` flag. The solver threads them into machinery it already
    has -- the frequency channel is summed into the detuning sequence and the
    RIN channel scales the pump kick -- so the cavity transfer function and the
    thermal transduction of RIN emerge from the equations of motion rather than
    being hand-implemented.

    Examples
    --------
    >>> cfg = {"fsr_hz": 2.0e11, "pump_noise_enabled": 1,
    ...        "pump_freq_noise_h0_hz2_per_hz": 3e3,
    ...        "pump_freq_noise_hm1_hz3_per_hz": 1e10,
    ...        "pump_rin_floor_dbc_per_hz": -160.0,
    ...        "pump_rin_excess_dbc_per_hz": -140.0}
    >>> pump = PumpNoise(cfg)
    >>> print(f"{pump.lorentzian_linewidth_hz:.4e} Hz")   # pi*h0
    9.4248e+03 Hz

    Disabled is inert, whatever the numbers say:

    >>> off = PumpNoise({**cfg, "pump_noise_enabled": 0})
    >>> off.enabled, off.lorentzian_linewidth_hz, float(off.psd_freq(1e6))
    (False, 0.0, 0.0)

    A linear density entered where dBc/Hz belongs is rejected:

    >>> PumpNoise({**cfg, "pump_rin_floor_dbc_per_hz": 1e-16})
    Traceback (most recent call last):
        ...
    ValueError: pump_rin_floor_dbc_per_hz = 1e-16 exceeds -80 dBc/Hz...
    """

    def __init__(self, cfg, enabled: bool | None = None):
        """Validate the laser-noise coefficients and pre-scale them by the switch.

        See :class:`PumpNoise` for the parameter table, the units of every
        config key it reads, and the attributes this sets.

        Raises
        ------
        ValueError
            If ``pump_noise_enabled`` is not boolean-valued or, when enabled,
            if a frequency-noise coefficient is negative, the RIN corner is
            non-positive, or a RIN level exceeds -80 dBc/Hz.

        Notes
        -----
        The effective coefficients are multiplied by the switch here, so a
        disabled channel yields exact zeros from the PSD accessors as well as
        from the samplers -- "what spectrum did this run have?" stays
        answerable either way. The value RANGES are validated only when the
        channel is enabled, so a config carrying placeholder numbers for an
        unused channel is not rejected.
        """
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
        """Evaluate the one-sided frequency-noise PSD of the laser.

        Parameters
        ----------
        f : array_like
            Frequency [Hz], scalar or array.

        Returns
        -------
        numpy.ndarray
            ``S_dnu(f) = h0 + h_-1/f`` [Hz**2/Hz], same shape as ``f``. Exact
            zeros when the channel is disabled.

        Notes
        -----
        The closed-form validation target for :meth:`sample_freq`, per
        arXiv:2604.05897v1 Sec. V.B.4. The white plateau ``h0`` carries the
        intrinsic Lorentzian linewidth through ``dnu_L = pi*h0``; ``h_-1`` is
        the flicker coefficient [Hz**3/Hz]. The ``f = 0`` bin is guarded by
        clamping the divisor at the smallest positive float rather than by
        special-casing, so the callable stays vectorized.

        Examples
        --------
        >>> pump = PumpNoise({"fsr_hz": 2.0e11, "pump_noise_enabled": 1,
        ...                   "pump_freq_noise_h0_hz2_per_hz": 3e3,
        ...                   "pump_freq_noise_hm1_hz3_per_hz": 1e10})
        >>> print(f"{float(pump.psd_freq(1e6)):.4e}")    # h0 + h_-1/f
        1.3000e+04
        >>> bool(pump.psd_freq(1e3) > pump.psd_freq(1e6))
        True
        """
        f = np.asarray(f, dtype=np.float64)
        return self._h0 + self._hm1 / np.maximum(f, np.finfo(np.float64).tiny)

    def psd_rin(self, f) -> np.ndarray:
        """Evaluate the one-sided relative-intensity-noise PSD.

        Parameters
        ----------
        f : array_like
            Frequency [Hz], scalar or array.

        Returns
        -------
        numpy.ndarray
            ``S_eps(f)`` [1/Hz], same shape as ``f``: a flat floor
            ``10**(floor_dBc/10)`` plus, below the corner ``f_c``, an excess
            ``10**(excess_dBc/10)*(f_c/f)``. Exact zeros when disabled.

        Notes
        -----
        The closed-form validation target for :meth:`sample_rin`, per
        arXiv:2604.05897v1 Sec. V.B.5. ``eps`` is dimensionless:
        ``P_in(t) = Pbar_in*(1 + eps(t))``.

        Examples
        --------
        >>> pump = PumpNoise({"fsr_hz": 2.0e11, "pump_noise_enabled": 1,
        ...                   "pump_rin_floor_dbc_per_hz": -160.0,
        ...                   "pump_rin_excess_dbc_per_hz": -140.0,
        ...                   "pump_rin_corner_hz": 1e4})
        >>> print(f"{float(pump.psd_rin(1e6)):.4e}")   # above the corner: floor
        1.0000e-16
        >>> print(f"{float(pump.psd_rin(1e3)):.4e}")   # below it: 1/f excess
        1.0010e-13
        """
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
        """Draw the pump-laser frequency-noise sequence.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        N : int
            Number of round trips [samples], sampled at ``f_s = 1/t_r``.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)``, dtype ``float64``, units rad/s: ``2*pi*dnu_p(t)``.
            Exact zeros when the channel is disabled or both coefficients are
            zero.

        Raises
        ------
        ImportError
            Propagated from the host-side generator.

        Notes
        -----
        SIGN CONVENTION: this returns ``+2*pi*dnu_p``. The caller applies the
        sign -- the contribution to the detuning is ``-2*pi*dnu_p``, because
        ``delta_omega = omega_res - omega_p``, so a positive laser-frequency
        excursion LOWERS the detuning.

        The white part is drawn i.i.d. per round trip with variance
        ``h0*f_s/2`` (the one-sided convention ``var = integral_0^{f_s/2} S df``);
        the flicker part is FFT-synthesized with ``S(f_k) = h_-1/max(f_k, f_1)``,
        so the single DC bin carries ``h_-1`` rather than diverging.

        Examples
        --------
        >>> import jax
        >>> pump = PumpNoise({"fsr_hz": 2.0e11, "pump_noise_enabled": 1,
        ...                   "pump_freq_noise_h0_hz2_per_hz": 3e3})
        >>> x = pump.sample_freq(jax.random.PRNGKey(0), 8)
        >>> print(x.shape, x.dtype)
        (8,) float64
        >>> off = PumpNoise({"fsr_hz": 2.0e11})
        >>> off.sample_freq(jax.random.PRNGKey(0), 3)
        array([0., 0., 0.])
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
        """Draw the pump relative-intensity-noise sequence.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        N : int
            Number of round trips [samples], sampled at ``f_s = 1/t_r``.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)``, dtype ``float64``, dimensionless: ``eps(t)`` in
            ``P_in(t) = Pbar_in*(1 + eps(t))``, clipped so ``1 + eps >= 0``.
            Exact zeros when the channel is disabled.

        Raises
        ------
        ImportError
            Propagated from the host-side generator.

        Warns
        -----
        UserWarning
            If more than 0.01% of the samples clip at ``eps = -1``, reporting
            the clipped fraction. The configured RIN is then so large that the
            Gaussian model is physically strained; the clipped sequence is
            returned rather than silently renormalized.

        Notes
        -----
        The floor is i.i.d. Gaussian per round trip with variance
        ``floor*f_s/2``; the excess is FFT-synthesized exactly like the flicker
        part of :meth:`sample_freq` -- same DC clamp -- and is zero above the
        corner. The clip enforces the physical constraint that the pump power
        cannot be negative.

        Examples
        --------
        >>> import jax
        >>> pump = PumpNoise({"fsr_hz": 2.0e11, "pump_noise_enabled": 1,
        ...                   "pump_rin_floor_dbc_per_hz": -160.0})
        >>> eps = pump.sample_rin(jax.random.PRNGKey(0), 8)
        >>> print(eps.shape, eps.dtype)
        (8,) float64
        >>> bool((eps >= -1.0).all())
        True
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
    """Write the noise-PSD comparison figure to ``analysis/figures``.

    Returns
    -------
    None
        The figure is written to
        ``analysis/figures/noise_psd_comparison.pdf``; the directory is created
        if needed.

    Raises
    ------
    FileNotFoundError
        If the default config ``config/sin_params.yaml`` is missing.
    OSError
        If ``analysis/figures`` cannot be created or written.

    Notes
    -----
    Overlays the three closed-form detuning PSDs [(rad/s)**2/Hz] of the
    configured platform -- thermorefractive, pyro-electric/EO and
    thermal-carrier -- on the Welch estimate of a 100000-sample realization of
    their sum, plus a Si3N4 thermorefractive reference curve computed from
    literature material constants. Agreement between the closed forms and the
    Welch estimate is the visual form of the check
    :func:`validate_noise_models` makes numerically.

    No ``Examples`` section: this function writes a file and needs the
    repository's config, so it is exercised by
    ``python -m simulator.noise_models`` rather than by a doctest.
    """
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
    """Assert the sampled noise streams match their closed-form statistics.

    Returns
    -------
    None
        Returns normally iff every check passes.

    Raises
    ------
    AssertionError
        If :meth:`TotalNoise.sample` does not return the expected shape/dtype;
        if the realized standard deviation is outside a decade of
        ``sqrt((sigma_trn - sigma_pyroeo)**2 + var_tccr)`` [rad/s], the
        prediction that follows from TRN and Pyro-EO sharing one temperature
        realization with opposite signs; or if the thermal-carrier
        autocorrelation time estimated from lag-1 correlation differs from
        ``tau_carrier_s`` by more than 50%.
    FileNotFoundError
        If the default config ``config/sin_params.yaml`` is missing.

    Warns
    -----
    UserWarning
        If the thermal-carrier channel is active but not dominant over the
        thermorefractive one, which for a chi(2) device suggests
        ``surface_state_density_per_m2`` or the effective mode area needs
        checking.

    Notes
    -----
    The correlated-sum check is the one that matters: adding the channels in
    quadrature would predict a LARGER total than is observed, because the
    pyro-electric pull partially cancels the thermorefractive one. The
    SiN-configured defaults leave the carrier channel identically zero, and both
    the correlation-time check and the dominance warning are skipped in that
    case rather than asserting on an all-zero stream.

    No ``Examples`` section: this is the module's self-test entry point
    (``python -m simulator.noise_models``) and reads the repository config.
    """
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
