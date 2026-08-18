"""Declarative, hashable noise-channel configuration for the stochastic-LLE benchmark.

This module is PURE CONFIGURATION. It is deliberately **not wired into the solver**:
nothing here imports :mod:`simulator.lle_solver` or :mod:`simulator.noise_models`, and
no existing code path reads :class:`NoiseConfig`. It exists so that a benchmark run can
declare *which stochastic channels are active* in one place, hash that declaration into
a run manifest, and later prove that two runs used the same noise configuration.

Why a dedicated object
----------------------
The channel switches are currently spread across the physical-parameter YAML in three
different idioms (see ``docs/NOISE_CHANNEL_INVENTORY.md``):

* explicit 0/1 integer flags — ``quantum_noise_enabled``, ``pump_noise_enabled``,
  ``fsr_noise_enabled``;
* *implicit* gating by a physical constant — the thermorefractive / pyro-electric /
  thermal-carrier channels are silenced only by ``T_k = 0`` or by material coefficients
  (``eo_r33_m_per_v``, ``pyroelectric_coeff_c_per_m2_k``) being zero;
* no switch at all for the thermal-expansion pull (gated by ``alpha_L_per_k = 0.0``).

``NoiseConfig`` gives every channel a first-class boolean so an ablation can be stated
declaratively instead of by nudging physical constants.

Switches vs parameters
----------------------
The eight ``bool`` fields listed in :data:`SWITCH_FIELDS` are **switches**: each one
turns a mechanism on or off, and every one of them defaults to ``False``, so a
default-constructed ``NoiseConfig`` is fully deterministic.

Every other field is a **parameter**: it shapes a channel that is already on and has no
"off" meaning of its own. Three parameters happen to be booleans
(:data:`PARAMETER_BOOL_DEFAULTS`) and two of those default to ``True``; they are *not*
switches, and "all channels off" says nothing about them. Keeping this distinction
explicit is what lets :func:`tests.test_noise_config` fail loudly when a newly added
boolean field is neither a declared switch nor a declared parameter.

``thermal_feedback`` is a switch but **not a noise channel**: see the field docs and
:meth:`NoiseConfig.all_off`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "NoiseConfig",
    "NOISE_CHANNELS",
    "SWITCH_FIELDS",
    "PARAMETER_BOOL_DEFAULTS",
    "TRN_PSD_MODELS",
    "THERMAL_INTEGRATORS",
    "NOISE_DTYPES",
    "QUANTUM_INJECTION_CADENCES",
    "LEGACY_SWITCH_KEYS",
    "LEGACY_PARAMETER_KEYS",
]


#: The seven genuinely *stochastic* channels. ``thermal_feedback`` is deliberately
#: absent — it is deterministic dynamics, not noise.
NOISE_CHANNELS: tuple[str, ...] = (
    "quantum_vacuum",
    "trn",
    "pyro_eo",
    "tccr",
    "pump_freq_noise",
    "pump_rin",
    "fsr",
)

#: Every boolean field that acts as an on/off switch. All of these MUST default to
#: ``False`` — :func:`tests.test_noise_config.test_every_switch_defaults_to_false`
#: enumerates the dataclass to enforce it.
SWITCH_FIELDS: tuple[str, ...] = NOISE_CHANNELS + ("thermal_feedback",)

#: Boolean fields that are parameters rather than switches, with their expected
#: defaults pinned. A boolean field that is in neither this mapping nor
#: :data:`SWITCH_FIELDS` is a classification error and fails the test suite.
PARAMETER_BOOL_DEFAULTS: dict[str, bool] = {
    "trn_ar1_stationary_init": False,
    "quantum_seed_vacuum_init": True,
    "legacy_segment_noise": True,
}

TRN_PSD_MODELS: tuple[str, ...] = ("single_pole", "kondratiev_gorodetsky", "csv")
THERMAL_INTEGRATORS: tuple[str, ...] = ("euler", "exponential")
NOISE_DTYPES: tuple[str, ...] = ("float32", "float64")
QUANTUM_INJECTION_CADENCES: tuple[int, ...] = (0, 1)

#: Legacy ``physical_parameters`` switch keys → the :class:`NoiseConfig` field(s) they
#: map onto. Note ``pump_noise_enabled`` is a SINGLE legacy flag that gates BOTH pump
#: channels, so the legacy schema cannot express "frequency noise on, RIN off".
LEGACY_SWITCH_KEYS: dict[str, tuple[str, ...]] = {
    "quantum_noise_enabled": ("quantum_vacuum",),
    "pump_noise_enabled": ("pump_freq_noise", "pump_rin"),
    "fsr_noise_enabled": ("fsr",),
}

#: Legacy ``physical_parameters`` parameter keys → :class:`NoiseConfig` field.
LEGACY_PARAMETER_KEYS: dict[str, str] = {
    "trn_psd_model": "trn_psd_model",
    "legacy_segment_noise": "legacy_segment_noise",
    "quantum_noise_injection_cadence": "quantum_injection_cadence",
    "quantum_noise_seed_vacuum_init": "quantum_seed_vacuum_init",
}


def _as_bool(value: Any, key: str) -> bool:
    """Coerce a legacy 0/1/bool config leaf to ``bool``, rejecting anything else.

    The repository encodes booleans as 0/1 integers under ``physical_parameters``
    because every leaf there must parse as a plain number (locked in by
    ``tests/test_config.py``), so both spellings must be accepted here.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(
        f"legacy key {key!r} must be boolean-valued (bool or 0/1), got {value!r}."
    )


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    """Immutable declaration of which noise channels are active, plus their parameters.

    All eight switches default to ``False``, so ``NoiseConfig()`` is the fully
    deterministic configuration.

    Parameters
    ----------
    quantum_vacuum : bool, optional
        SWITCH. Quantum-vacuum Langevin drive, additive on the field
        [sqrt(J) per fine step]. arXiv:2604.05897v1 Sec. V.B.2, Eq. 126.
        Default ``False``.
    trn : bool, optional
        SWITCH. Thermorefractive detuning noise [rad/s]. Default ``False``.
    pyro_eo : bool, optional
        SWITCH. Pyro-electric / electro-optic detuning noise [rad/s], driven by
        the SAME temperature realization dT(t) [K] as ``trn``. Default ``False``.
    tccr : bool, optional
        SWITCH. Thermal-carrier / surface-state detuning noise [rad/s], an
        independent stream. Default ``False``.
    pump_freq_noise : bool, optional
        SWITCH. Pump-laser frequency noise, entering as a detuning [rad/s].
        arXiv:2604.05897v1 Sec. V.B.4. Default ``False``.
    pump_rin : bool, optional
        SWITCH. Pump relative-intensity noise (dimensionless eps, modulating the
        drive amplitude). arXiv:2604.05897v1 Sec. V.B.5. Default ``False``.
    fsr : bool, optional
        SWITCH. Thermorefractive-driven FSR / repetition-rate noise
        dD1(t) [rad/s], also driven by the ``trn`` temperature realization.
        Default ``False``.
    thermal_feedback : bool, optional
        SWITCH, but **NOT a noise channel**. Controls the DETERMINISTIC
        thermo-optic ODE -- the single-pole ``dDeltaT/dt`` feedback [K/s] that
        shifts the detuning with absorbed power [W]. It is a *dynamics* toggle,
        carried here only because a benchmark run must record it alongside the
        stochastic channels to be reproducible. A run with every stochastic
        channel off may legitimately still want thermo-optic dynamics on, which
        is why :meth:`all_off` does not force it off. Default ``False``.
    trn_psd_model : {'single_pole', 'kondratiev_gorodetsky', 'csv'}, optional
        PARAMETER. Spectrum of the temperature fluctuation [K**2/Hz] driving
        ``trn`` / ``pyro_eo`` / ``fsr``. Default ``'single_pole'``.
    trn_ar1_stationary_init : bool, optional
        PARAMETER. Start the AR(1) generator from its stationary distribution
        instead of from zero, removing the ``~tau_th/(2*t_r)`` round-trip
        start-up transient. Default ``False`` (bit-identical to the historical
        stream).
    thermal_integrator : {'euler', 'exponential'}, optional
        PARAMETER. Integrator for the thermo-optic ODE. Default ``'euler'``,
        which is what every committed golden trajectory was produced with.
    quantum_injection_cadence : {0, 1}, optional
        PARAMETER. ``0`` injects the vacuum increment once per fine step
        (dt_fine = t_r/M [s]; the exact prescription); ``1`` once per round trip
        (dt = t_r [s]). Default ``0``.
    quantum_seed_vacuum_init : bool, optional
        PARAMETER. Seed a cold start at the half-photon-per-mode vacuum level
        [J] rather than with the legacy ``1e-3*|e_cw|`` ad-hoc seed. Default
        ``True``.
    legacy_segment_noise : bool, optional
        PARAMETER. ``True`` regenerates segment noise from zero per segment (the
        historical behaviour); ``False`` slices one stationary full-trajectory
        realization, removing the per-segment decorrelation transient. Default
        ``True``.
    noise_dtype : {'float32', 'float64'}, optional
        PARAMETER. Working precision of the generated classical noise
        sequences. Default ``'float32'``.
    seed : int or None, optional
        PARAMETER. Master seed for the run (dimensionless). ``None`` means the
        caller supplies the JAX PRNG key. Default ``None``.

    Raises
    ------
    ValueError
        From :meth:`__post_init__` if ``trn_psd_model``, ``thermal_integrator``
        or ``noise_dtype`` is outside its allowed set, or if
        ``quantum_injection_cadence`` is not the ``int`` 0 or 1 (a ``bool`` is
        rejected explicitly: ``True`` and ``1`` would serialise differently and
        so produce two digests for one configuration).

    Notes
    -----
    The eight ``bool`` fields listed in :data:`SWITCH_FIELDS` are SWITCHES: each
    turns a mechanism on or off and every one defaults to ``False``. Every other
    field is a PARAMETER -- it shapes a channel that is already on and has no
    "off" meaning of its own. Three parameters happen to be booleans
    (:data:`PARAMETER_BOOL_DEFAULTS`) and two of those default to ``True``; they
    are NOT switches, and "all channels off" says nothing about them.

    The class is a frozen dataclass, so an instance is hashable and safe to
    share across trajectories, and :meth:`sha256` gives it a stable content
    identity for a run manifest.

    Examples
    --------
    The default is fully deterministic:

    >>> NoiseConfig().enabled_channels
    ()

    Leave-one-out ablation off the full-physics configuration:

    >>> NoiseConfig.all_on(tccr=False).enabled_channels
    ('quantum_vacuum', 'trn', 'pyro_eo', 'pump_freq_noise', 'pump_rin', 'fsr')

    Validation happens at construction, not at first use:

    >>> NoiseConfig(thermal_integrator="rk4")
    Traceback (most recent call last):
        ...
    ValueError: thermal_integrator must be one of ('euler', 'exponential'), got 'rk4'.
    """

    # --- switches (all default False) ---------------------------------------
    quantum_vacuum: bool = False
    trn: bool = False
    pyro_eo: bool = False
    tccr: bool = False
    pump_freq_noise: bool = False
    pump_rin: bool = False
    fsr: bool = False
    thermal_feedback: bool = False

    # --- parameters ---------------------------------------------------------
    trn_psd_model: str = "single_pole"
    trn_ar1_stationary_init: bool = False
    thermal_integrator: str = "euler"          # "euler" | "exponential"
    quantum_injection_cadence: int = 0          # 0 = per fine step, 1 = per round trip
    quantum_seed_vacuum_init: bool = True
    legacy_segment_noise: bool = True
    noise_dtype: str = "float32"                # "float32" | "float64"
    seed: int | None = None

    # -----------------------------------------------------------------------
    # validation
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate every enumerated field at construction time.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``trn_psd_model`` is not in :data:`TRN_PSD_MODELS`, if
            ``thermal_integrator`` is not in :data:`THERMAL_INTEGRATORS`, if
            ``noise_dtype`` is not in :data:`NOISE_DTYPES`, or if
            ``quantum_injection_cadence`` is not an ``int`` in
            :data:`QUANTUM_INJECTION_CADENCES`.

        Notes
        -----
        ``bool`` is a subclass of ``int``, so ``True in (0, 1)`` is ``True``.
        The cadence check rejects ``bool`` explicitly because ``True`` and ``1``
        serialise differently (``"true"`` versus ``"1"``) and would therefore
        give two different :meth:`sha256` digests for one semantically identical
        configuration.

        Examples
        --------
        >>> NoiseConfig(quantum_injection_cadence=True)
        Traceback (most recent call last):
            ...
        ValueError: quantum_injection_cadence must be one of (0, 1) (int, not bool), got True.
        """
        if self.trn_psd_model not in TRN_PSD_MODELS:
            raise ValueError(
                f"trn_psd_model must be one of {TRN_PSD_MODELS}, "
                f"got {self.trn_psd_model!r}."
            )
        if self.thermal_integrator not in THERMAL_INTEGRATORS:
            raise ValueError(
                f"thermal_integrator must be one of {THERMAL_INTEGRATORS}, "
                f"got {self.thermal_integrator!r}."
            )
        if self.noise_dtype not in NOISE_DTYPES:
            raise ValueError(
                f"noise_dtype must be one of {NOISE_DTYPES}, "
                f"got {self.noise_dtype!r}."
            )
        # bool is a subclass of int, so `True in (0, 1)` is True. Reject it explicitly:
        # True and 1 would serialise differently ("true" vs "1") and therefore produce
        # two different sha256 digests for a semantically identical configuration.
        if isinstance(self.quantum_injection_cadence, bool) or (
            self.quantum_injection_cadence not in QUANTUM_INJECTION_CADENCES
        ):
            raise ValueError(
                f"quantum_injection_cadence must be one of "
                f"{QUANTUM_INJECTION_CADENCES} (int, not bool), "
                f"got {self.quantum_injection_cadence!r}."
            )

    # -----------------------------------------------------------------------
    # constructors
    # -----------------------------------------------------------------------
    @classmethod
    def all_off(cls, **overrides: Any) -> "NoiseConfig":
        """Construct a configuration with every *stochastic* channel off.

        Parameters
        ----------
        **overrides
            Any :class:`NoiseConfig` field, applied AFTER the seven stochastic
            channels are forced ``False``. This is what makes the constructor
            double as "one channel only".

        Returns
        -------
        NoiseConfig
            A frozen instance with every field in :data:`NOISE_CHANNELS` set to
            ``False`` except those named in ``overrides``.

        Raises
        ------
        TypeError
            If ``overrides`` names a field that does not exist.
        ValueError
            From :meth:`__post_init__` for an out-of-range enumerated value.

        Notes
        -----
        ``thermal_feedback`` is intentionally NOT forced off. It controls the
        deterministic thermo-optic ODE, which is dynamics rather than noise, and
        a deterministic run may legitimately want it active. It is left at
        whatever is passed in -- ``False`` by default, because that is the field
        default -- so pass ``all_off(thermal_feedback=True)`` for a
        deterministic run WITH thermo-optic dynamics.

        Examples
        --------
        >>> NoiseConfig.all_off().enabled_channels
        ()
        >>> NoiseConfig.all_off(trn=True).enabled_channels
        ('trn',)
        >>> NoiseConfig.all_off(thermal_feedback=True).thermal_feedback
        True
        """
        values: dict[str, Any] = {channel: False for channel in NOISE_CHANNELS}
        values.update(overrides)
        return cls(**values)

    @classmethod
    def all_on(cls, **overrides: Any) -> "NoiseConfig":
        """Construct the full-physics configuration: every switch on.

        Parameters
        ----------
        **overrides
            Any :class:`NoiseConfig` field, applied AFTER every field in
            :data:`SWITCH_FIELDS` is forced ``True``.

        Returns
        -------
        NoiseConfig
            All seven stochastic channels plus the deterministic thermo-optic
            feedback enabled, modified by ``overrides``.

        Raises
        ------
        TypeError
            If ``overrides`` names a field that does not exist.
        ValueError
            From :meth:`__post_init__` for an out-of-range enumerated value.

        Notes
        -----
        Overriding one channel to ``False`` is the leave-one-out ablation the
        benchmark's per-channel noise attribution is built on.

        Examples
        --------
        >>> NoiseConfig.all_on().thermal_feedback
        True
        >>> len(NoiseConfig.all_on().enabled_channels)
        7
        >>> len(NoiseConfig.all_on(tccr=False).enabled_channels)
        6
        """
        values: dict[str, Any] = {switch: True for switch in SWITCH_FIELDS}
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NoiseConfig":
        """Build a config from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML file carrying either a top-level ``noise:`` mapping (preferred)
            or, failing that, the legacy switches inside
            ``physical_parameters``.

        Returns
        -------
        NoiseConfig
            The declared configuration. Fields absent from the file keep their
            dataclass defaults.

        Raises
        ------
        ValueError
            If the document is not a mapping, if ``noise`` or
            ``physical_parameters`` is present but is not a mapping, if the
            ``noise`` block carries an unknown key, or -- via :func:`_as_bool`
            -- if a legacy switch is not boolean-valued.
        OSError
            If ``path`` cannot be opened.
        yaml.YAMLError
            If the file is not valid YAML.

        Warns
        -----
        DeprecationWarning
            When the legacy fallback consumed at least one
            ``physical_parameters`` key, naming every key it used.

        Notes
        -----
        Preferred form -- a top-level ``noise:`` block whose keys are
        :class:`NoiseConfig` field names::

            noise:
              trn: true
              quantum_vacuum: true
              trn_psd_model: kondratiev_gorodetsky

        Unknown keys inside that block raise :class:`ValueError` rather than being
        ignored, so a typo'd channel name cannot silently leave a channel off.

        Legacy fallback — when no ``noise:`` block is present the legacy switches
        inside ``physical_parameters`` are consulted (:data:`LEGACY_SWITCH_KEYS` and
        :data:`LEGACY_PARAMETER_KEYS`), with the thermorefractive channel inferred from
        ``T_k > 0``. Each legacy key actually consumed is named in a
        :class:`DeprecationWarning`.

        .. warning::
           The legacy schema has **no switch at all** for ``pyro_eo`` and ``tccr`` —
           in the current code those channels are gated only by material constants
           (``eo_r33_m_per_v``, ``pyroelectric_coeff_c_per_m2_k``). The fallback
           therefore reports them as ``False``, which is correct for a centrosymmetric
           (SiN, ``r33 = 0``) config but **understates a χ²/TFLN config**, where they
           are physically active. Likewise ``thermal_feedback`` has no legacy key and
           is reported ``False`` even though the legacy solver always integrates the
           thermo-optic ODE. Configs that need those channels stated truthfully should
           carry an explicit ``noise:`` block.

        Examples
        --------
        >>> import pathlib, tempfile
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "cfg.yaml"
        >>> _ = path.write_text(chr(10).join(
        ...     ["noise:", "  trn: true", "  quantum_vacuum: true"]))
        >>> NoiseConfig.from_yaml(path).enabled_channels
        ('quantum_vacuum', 'trn')

        A typo'd channel name is an error, never a silently-off channel:

        >>> _ = path.write_text(chr(10).join(["noise:", "  trnn: true"]))
        >>> NoiseConfig.from_yaml(path)
        Traceback (most recent call last):
            ...
        ValueError: ...unknown key(s) in the 'noise' block: ['trnn']...
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level.")

        block = raw.get("noise")
        if block is not None:
            if not isinstance(block, dict):
                raise ValueError(
                    f"{path}: top-level 'noise' must be a mapping, "
                    f"got {type(block).__name__}."
                )
            known = {f.name for f in dataclasses.fields(cls)}
            unknown = sorted(set(block) - known)
            if unknown:
                raise ValueError(
                    f"{path}: unknown key(s) in the 'noise' block: {unknown}. "
                    f"Valid fields: {sorted(known)}."
                )
            return cls(**dict(block))

        # --- legacy fallback -------------------------------------------------
        physical = raw.get("physical_parameters") or {}
        if not isinstance(physical, dict):
            raise ValueError(
                f"{path}: 'physical_parameters' must be a mapping, "
                f"got {type(physical).__name__}."
            )

        values: dict[str, Any] = {}
        consumed: list[str] = []

        for legacy_key, targets in LEGACY_SWITCH_KEYS.items():
            if legacy_key in physical:
                enabled = _as_bool(physical[legacy_key], legacy_key)
                for target in targets:
                    values[target] = enabled
                consumed.append(legacy_key)

        # The thermorefractive channel has no legacy switch: it is active whenever the
        # thermodynamic temperature-fluctuation variance is non-zero, i.e. T_k > 0.
        if "T_k" in physical:
            values["trn"] = float(physical["T_k"]) > 0.0
            consumed.append("T_k")

        for legacy_key, target in LEGACY_PARAMETER_KEYS.items():
            if legacy_key in physical:
                value = physical[legacy_key]
                if target in ("legacy_segment_noise", "quantum_seed_vacuum_init"):
                    value = _as_bool(value, legacy_key)
                elif target == "quantum_injection_cadence":
                    value = int(value)
                values[target] = value
                consumed.append(legacy_key)

        if consumed:
            warnings.warn(
                f"{path}: no top-level 'noise:' block; built NoiseConfig from the "
                f"legacy physical_parameters key(s) {sorted(consumed)}. The "
                f"thermorefractive channel was inferred from T_k > 0, and the legacy "
                f"schema cannot express pyro_eo/tccr/thermal_feedback at all (they are "
                f"reported False). Add an explicit 'noise:' block.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(**values)

    # -----------------------------------------------------------------------
    # serialisation / identity
    # -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable, key-sorted mapping of every field.

        Returns
        -------
        dict
            One entry per dataclass field, ordered by key. Values are the plain
            ``bool`` / ``int`` / ``str`` / ``None`` leaves the dataclass holds,
            so the mapping round-trips through :func:`json.dumps` unchanged.

        Raises
        ------
        TypeError
            Only if a future field were to hold a non-copyable value;
            ``dataclasses.asdict`` deep-copies its input.

        Notes
        -----
        Key-sorting here (rather than only inside :meth:`sha256`) means the
        mapping written into a run manifest has the same ordering as the bytes
        that were hashed, so a manifest can be diffed line-by-line against
        another run.

        Examples
        --------
        >>> d = NoiseConfig(trn=True, seed=7).to_dict()
        >>> list(d)[:3]
        ['fsr', 'legacy_segment_noise', 'noise_dtype']
        >>> d["trn"], d["seed"], d["thermal_integrator"]
        (True, 7, 'euler')
        """
        raw = dataclasses.asdict(self)
        return {key: raw[key] for key in sorted(raw)}

    def sha256(self) -> str:
        """Return the stable content hash of the configuration.

        Returns
        -------
        str
            64 lowercase hex characters:
            ``sha256(json.dumps(self.to_dict(), sort_keys=True))``.

        Raises
        ------
        TypeError
            If a future field were not JSON-serialisable.

        Notes
        -----
        Stable across processes and across field REORDERING in the dataclass,
        and sensitive to any field VALUE change -- which is what makes it usable
        as a run-manifest identity. :func:`simulator.provenance.noise_config_sha256`
        delegates here rather than re-canonicalizing, so the digest recorded in
        the noise-off identity manifests cannot drift from this one.

        Examples
        --------
        >>> len(NoiseConfig().sha256())
        64
        >>> NoiseConfig().sha256() == NoiseConfig().sha256()
        True
        >>> NoiseConfig().sha256() == NoiseConfig(trn=True).sha256()
        False
        """
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -----------------------------------------------------------------------
    # reporting
    # -----------------------------------------------------------------------
    @property
    def enabled_channels(self) -> tuple[str, ...]:
        """Return the stochastic channels that are on, in declaration order.

        Returns
        -------
        tuple of str
            A subset of :data:`NOISE_CHANNELS`, in that tuple's order.
            ``thermal_feedback`` never appears -- it is deterministic dynamics,
            not a noise channel.

        Raises
        ------
        AttributeError
            Only if :data:`NOISE_CHANNELS` were to name a field the dataclass
            does not have; ``tests/test_noise_config.py`` guards against that.

        Notes
        -----
        Declaration order rather than alphabetical order, so a run log reads in
        the same sequence as the channel inventory in
        ``docs/NOISE_CHANNEL_INVENTORY.md``.

        Examples
        --------
        >>> NoiseConfig(trn=True, quantum_vacuum=True).enabled_channels
        ('quantum_vacuum', 'trn')
        >>> NoiseConfig(thermal_feedback=True).enabled_channels
        ()
        """
        return tuple(c for c in NOISE_CHANNELS if getattr(self, c))

    def describe(self) -> str:
        """Render a human-readable summary for run logs.

        Returns
        -------
        str
            A multi-line block: a header line carrying the truncated
            :meth:`sha256` and the seed, one ``[on ]`` line per enabled
            stochastic channel with the parameters that shape it, a
            ``thermal_feedback`` line flagged as NOT a noise channel, and a
            trailing line with the working dtype and the segment-noise policy.

        Raises
        ------
        KeyError
            Only if :data:`NOISE_CHANNELS` gained a channel without a matching
            entry in the local ``detail`` mapping.

        Notes
        -----
        ``thermal_feedback`` is always printed, on or off, precisely because it
        is easy to forget that a "deterministic" run may still be integrating
        the thermo-optic ODE.

        Examples
        --------
        >>> lines = NoiseConfig.all_off(trn=True).describe().splitlines()
        >>> lines[0].startswith("NoiseConfig sha256=")
        True
        >>> print(lines[1])
          [on ] trn              psd_model=single_pole, ar1_stationary_init=False
        >>> print(NoiseConfig().describe().splitlines()[1])
          [off] no stochastic channels enabled (deterministic run)
        """
        lines = [f"NoiseConfig sha256={self.sha256()[:16]}… seed={self.seed!r}"]

        detail: dict[str, str] = {
            "quantum_vacuum": (
                f"cadence={self.quantum_injection_cadence} "
                f"({'per fine step' if self.quantum_injection_cadence == 0 else 'per round trip'}), "
                f"seed_vacuum_init={self.quantum_seed_vacuum_init}"
            ),
            "trn": (
                f"psd_model={self.trn_psd_model}, "
                f"ar1_stationary_init={self.trn_ar1_stationary_init}"
            ),
            "pyro_eo": "shares the TRN temperature realization",
            "tccr": "independent stream",
            "pump_freq_noise": "enters as a detuning",
            "pump_rin": "modulates the drive amplitude",
            "fsr": "shares the TRN temperature realization",
        }

        enabled = self.enabled_channels
        if enabled:
            for channel in enabled:
                lines.append(f"  [on ] {channel:<16} {detail[channel]}")
        else:
            lines.append("  [off] no stochastic channels enabled (deterministic run)")

        lines.append(
            f"  [{'on ' if self.thermal_feedback else 'off'}] "
            f"{'thermal_feedback':<16} deterministic thermo-optic ODE "
            f"(integrator={self.thermal_integrator}) — NOT a noise channel"
        )
        lines.append(
            f"  noise_dtype={self.noise_dtype}, "
            f"legacy_segment_noise={self.legacy_segment_noise}"
        )
        return "\n".join(lines)
