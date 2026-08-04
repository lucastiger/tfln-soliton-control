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

    Switch fields
    -------------
    quantum_vacuum:
        Quantum-vacuum Langevin drive (additive on the field).
    trn:
        Thermorefractive detuning noise.
    pyro_eo:
        Pyro-electric / electro-optic detuning noise. Driven by the SAME temperature
        realization as ``trn``.
    tccr:
        Thermal carrier / surface-state detuning noise (independent stream).
    pump_freq_noise:
        Pump-laser frequency noise (enters as a detuning).
    pump_rin:
        Pump relative-intensity noise (modulates the drive amplitude).
    fsr:
        Thermorefractive-driven FSR / repetition-rate noise. Also driven by the ``trn``
        temperature realization.
    thermal_feedback:
        **NOT a noise channel.** This switch controls the DETERMINISTIC thermo-optic
        ODE — the single-pole ``dΔT/dt`` feedback that shifts the detuning with
        absorbed power. It is a *dynamics* toggle, listed here only because a benchmark
        run must record it alongside the stochastic channels to be reproducible. A run
        with every stochastic channel off may legitimately still want thermo-optic
        dynamics on, which is why :meth:`all_off` does not force it off.

    Parameter fields
    ----------------
    trn_psd_model:
        Spectrum used for the temperature fluctuation driving ``trn``/``pyro_eo``/``fsr``.
    trn_ar1_stationary_init:
        Start the AR(1) generator from its stationary distribution instead of from
        zero, removing the start-up transient.
    thermal_integrator:
        Integrator for the thermo-optic ODE.
    quantum_injection_cadence:
        ``0`` = inject once per fine step (exact); ``1`` = once per round trip.
    quantum_seed_vacuum_init:
        Seed a cold start at the vacuum level rather than with the legacy ad-hoc seed.
    legacy_segment_noise:
        ``True`` regenerates segment noise from zero per segment (historical
        behaviour); ``False`` slices one stationary full-trajectory realization.
    noise_dtype:
        Working precision of the generated classical noise sequences.
    seed:
        Master seed for the run; ``None`` means the caller supplies the key.
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
        """Every *stochastic* channel off.

        ``thermal_feedback`` is intentionally NOT forced off. It controls the
        deterministic thermo-optic ODE, which is dynamics rather than noise: a
        deterministic run may legitimately want the thermal feedback active. It is
        therefore left at whatever value is passed in, which is ``False`` by default
        because that is the field default — pass ``all_off(thermal_feedback=True)`` for
        a deterministic run *with* thermo-optic dynamics.

        Any other field may also be overridden, so this doubles as "one channel only":
        ``NoiseConfig.all_off(trn=True)``.
        """
        values: dict[str, Any] = {channel: False for channel in NOISE_CHANNELS}
        values.update(overrides)
        return cls(**values)

    @classmethod
    def all_on(cls, **overrides: Any) -> "NoiseConfig":
        """Every switch on, including ``thermal_feedback``.

        This is the "full physics" configuration: all seven stochastic channels plus
        the deterministic thermo-optic feedback. Individual fields may be overridden,
        e.g. ``NoiseConfig.all_on(tccr=False)`` for a leave-one-out ablation.
        """
        values: dict[str, Any] = {switch: True for switch in SWITCH_FIELDS}
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NoiseConfig":
        """Build a config from a YAML file.

        Preferred form — a top-level ``noise:`` block whose keys are
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
        """JSON-serialisable mapping of every field, key-sorted."""
        raw = dataclasses.asdict(self)
        return {key: raw[key] for key in sorted(raw)}

    def sha256(self) -> str:
        """Stable content hash of the configuration.

        ``sha256(json.dumps(self.to_dict(), sort_keys=True))``. Stable across
        processes and across field *reordering* in the dataclass, and sensitive to any
        field value change — suitable as a run-manifest identity.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -----------------------------------------------------------------------
    # reporting
    # -----------------------------------------------------------------------
    @property
    def enabled_channels(self) -> tuple[str, ...]:
        """The stochastic channels that are on, in declaration order."""
        return tuple(c for c in NOISE_CHANNELS if getattr(self, c))

    def describe(self) -> str:
        """Human-readable summary for run logs: one line per enabled channel."""
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
