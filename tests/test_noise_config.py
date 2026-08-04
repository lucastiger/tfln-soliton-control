"""Tests for :mod:`simulator.noise_config`.

These pin the properties the benchmark relies on:

* a default-constructed config is fully deterministic (every SWITCH off), and the
  switch/parameter classification is exhaustive — a newly added boolean field that is
  neither cannot slip through with a silent ``True`` default;
* ``sha256()`` is a stable, collision-free run identity;
* the legacy ``physical_parameters`` fallback still reads the committed config, loudly.

Nothing here imports the solver: ``simulator.noise_config`` is not wired in yet.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import pytest

from simulator.noise_config import (
    NOISE_CHANNELS,
    PARAMETER_BOOL_DEFAULTS,
    QUANTUM_INJECTION_CADENCES,
    SWITCH_FIELDS,
    THERMAL_INTEGRATORS,
    TRN_PSD_MODELS,
    NoiseConfig,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

# Pinned digest of NoiseConfig.all_off(). Recomputing this by hand is the point: if a
# field is added, removed, renamed, or its default changes, the run identity of every
# historical benchmark changes with it, and that must never happen silently.
ALL_OFF_SHA256 = "48261c63e68c1287a282981387256c593e4d5a6ad91a3ad11ef35da20d425b4f"


def _bool_fields() -> list[str]:
    return [
        f.name
        for f in dataclasses.fields(NoiseConfig)
        if f.type in ("bool", bool)
    ]


# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------
def test_every_switch_defaults_to_false():
    """Every SWITCH boolean defaults to False, enumerated from the dataclass itself."""
    defaults = {f.name: f.default for f in dataclasses.fields(NoiseConfig)}
    for name in SWITCH_FIELDS:
        assert name in defaults, f"declared switch {name!r} is not a dataclass field"
        assert defaults[name] is False, (
            f"switch {name!r} must default to False, got {defaults[name]!r}"
        )


def test_boolean_field_classification_is_exhaustive():
    """No boolean field escapes classification as either a switch or a parameter.

    This is the drift guard: a newly added boolean is a hard failure until it is
    explicitly declared a switch (default False, enforced above) or a parameter with a
    pinned default. It is what makes "every boolean defaults to False" checkable
    despite two parameter booleans legitimately defaulting to True.
    """
    declared = set(SWITCH_FIELDS) | set(PARAMETER_BOOL_DEFAULTS)
    actual = set(_bool_fields())
    assert actual == declared, (
        f"unclassified boolean field(s): {sorted(actual - declared)}; "
        f"declared-but-missing: {sorted(declared - actual)}"
    )
    # switches and parameters must not overlap
    assert not (set(SWITCH_FIELDS) & set(PARAMETER_BOOL_DEFAULTS))


def test_parameter_bool_defaults_are_pinned():
    """The non-switch booleans keep exactly the documented defaults."""
    defaults = {f.name: f.default for f in dataclasses.fields(NoiseConfig)}
    for name, expected in PARAMETER_BOOL_DEFAULTS.items():
        assert defaults[name] is expected, (
            f"parameter {name!r} default changed: {defaults[name]!r} != {expected!r}"
        )


def test_default_construction_is_fully_deterministic():
    """A bare NoiseConfig() has no stochastic channel enabled."""
    assert NoiseConfig().enabled_channels == ()


# ---------------------------------------------------------------------------
# 2. all_off / all_on
# ---------------------------------------------------------------------------
def test_all_off_has_no_enabled_switch():
    off = NoiseConfig.all_off()
    for name in SWITCH_FIELDS:
        assert getattr(off, name) is False, f"{name} should be off"
    assert off.enabled_channels == ()
    assert off == NoiseConfig()


def test_all_off_leaves_thermal_feedback_at_its_passed_value():
    """thermal_feedback is deterministic dynamics, not noise: all_off() must not force it."""
    assert NoiseConfig.all_off().thermal_feedback is False        # field default
    warm = NoiseConfig.all_off(thermal_feedback=True)
    assert warm.thermal_feedback is True
    assert warm.enabled_channels == ()          # still no NOISE, despite the feedback


def test_all_on_enables_every_switch():
    on = NoiseConfig.all_on()
    for name in SWITCH_FIELDS:
        assert getattr(on, name) is True, f"{name} should be on"
    assert on.enabled_channels == NOISE_CHANNELS


def test_all_on_supports_leave_one_out_overrides():
    ablated = NoiseConfig.all_on(tccr=False)
    assert ablated.tccr is False
    assert "tccr" not in ablated.enabled_channels
    assert len(ablated.enabled_channels) == len(NOISE_CHANNELS) - 1


# ---------------------------------------------------------------------------
# 3. sha256 stability
# ---------------------------------------------------------------------------
def test_all_off_sha256_matches_pinned_digest():
    assert NoiseConfig.all_off().sha256() == ALL_OFF_SHA256


def test_sha256_is_deterministic_within_and_across_processes():
    import subprocess
    import sys

    a = NoiseConfig.all_off().sha256()
    b = NoiseConfig.all_off().sha256()
    assert a == b                                  # stable within a process

    repo_root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from simulator.noise_config import NoiseConfig;"
            "print(NoiseConfig.all_off().sha256())",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == a                 # stable across a fresh interpreter


def test_to_dict_is_sorted_and_json_serialisable():
    d = NoiseConfig.all_on(seed=7).to_dict()
    assert list(d) == sorted(d)
    assert json.loads(json.dumps(d)) == d
    assert set(d) == {f.name for f in dataclasses.fields(NoiseConfig)}


# ---------------------------------------------------------------------------
# 4. sha256 sensitivity — every single field must move the digest
# ---------------------------------------------------------------------------
def _alternate_value(name: str, current):
    """A valid value for ``name`` that differs from ``current``."""
    if name in PARAMETER_BOOL_DEFAULTS or name in SWITCH_FIELDS:
        return not current
    if name == "trn_psd_model":
        return next(v for v in TRN_PSD_MODELS if v != current)
    if name == "thermal_integrator":
        return next(v for v in THERMAL_INTEGRATORS if v != current)
    if name == "noise_dtype":
        return "float64" if current == "float32" else "float32"
    if name == "quantum_injection_cadence":
        return next(v for v in QUANTUM_INJECTION_CADENCES if v != current)
    if name == "seed":
        return 12345 if current != 12345 else 54321
    raise AssertionError(f"no alternate value defined for field {name!r}")


@pytest.mark.parametrize(
    "field_name", [f.name for f in dataclasses.fields(NoiseConfig)]
)
def test_sha256_changes_when_any_single_field_changes(field_name):
    base = NoiseConfig.all_off()
    current = getattr(base, field_name)
    changed = dataclasses.replace(
        base, **{field_name: _alternate_value(field_name, current)}
    )
    assert changed != base
    assert changed.sha256() != base.sha256(), (
        f"sha256 is blind to field {field_name!r}"
    )


def test_all_field_digests_are_mutually_distinct():
    """No two single-field mutations collide — the digest separates every field."""
    base = NoiseConfig.all_off()
    digests = {}
    for f in dataclasses.fields(NoiseConfig):
        mutated = dataclasses.replace(
            base, **{f.name: _alternate_value(f.name, getattr(base, f.name))}
        )
        digests[f.name] = mutated.sha256()
    assert len(set(digests.values())) == len(digests), digests


# ---------------------------------------------------------------------------
# 5. Legacy YAML fallback
# ---------------------------------------------------------------------------
def test_from_yaml_committed_config_is_legacy_trn_only():
    """The committed config has no 'noise:' block: T_k=300 => trn only, and it warns."""
    with pytest.warns(DeprecationWarning) as record:
        cfg = NoiseConfig.from_yaml(CONFIG_PATH)

    assert cfg.trn is True
    for name in SWITCH_FIELDS:
        if name != "trn":
            assert getattr(cfg, name) is False, f"{name} should be off"
    assert cfg.enabled_channels == ("trn",)

    message = str(record[0].message)
    for legacy_key in ("T_k", "quantum_noise_enabled", "pump_noise_enabled",
                       "fsr_noise_enabled"):
        assert legacy_key in message, f"warning does not name {legacy_key!r}"


def test_from_yaml_noise_block_takes_precedence(tmp_path):
    path = tmp_path / "with_noise.yaml"
    path.write_text(
        "noise:\n"
        "  trn: true\n"
        "  quantum_vacuum: true\n"
        "  trn_psd_model: kondratiev_gorodetsky\n"
        "physical_parameters:\n"
        "  T_k: 0.0\n"
        "  quantum_noise_enabled: 0\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)   # must NOT warn
        cfg = NoiseConfig.from_yaml(path)
    assert cfg.enabled_channels == ("quantum_vacuum", "trn")
    assert cfg.trn_psd_model == "kondratiev_gorodetsky"


def test_from_yaml_legacy_switches_map_through(tmp_path):
    """One legacy pump flag gates BOTH pump channels."""
    path = tmp_path / "legacy_on.yaml"
    path.write_text(
        "physical_parameters:\n"
        "  T_k: 0.0\n"
        "  quantum_noise_enabled: 1\n"
        "  pump_noise_enabled: 1\n"
        "  fsr_noise_enabled: 1\n",
        encoding="utf-8",
    )
    with pytest.warns(DeprecationWarning):
        cfg = NoiseConfig.from_yaml(path)
    assert cfg.quantum_vacuum is True
    assert cfg.pump_freq_noise is True and cfg.pump_rin is True
    assert cfg.fsr is True
    assert cfg.trn is False                       # T_k = 0


def test_from_yaml_rejects_unknown_noise_key(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("noise:\n  trn_noise: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        NoiseConfig.from_yaml(path)


def test_from_yaml_rejects_non_boolean_legacy_switch(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "physical_parameters:\n  quantum_noise_enabled: 7\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="boolean-valued"):
        NoiseConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# 6. Enum validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"trn_psd_model": "lorentzian"}, "trn_psd_model"),
        ({"thermal_integrator": "rk4"}, "thermal_integrator"),
        ({"noise_dtype": "float16"}, "noise_dtype"),
        ({"quantum_injection_cadence": 2}, "quantum_injection_cadence"),
        ({"quantum_injection_cadence": -1}, "quantum_injection_cadence"),
    ],
)
def test_invalid_enum_values_raise_valueerror(kwargs, match):
    with pytest.raises(ValueError, match=match) as exc:
        NoiseConfig(**kwargs)
    offending = repr(next(iter(kwargs.values())))
    assert offending in str(exc.value), (
        f"error message must contain the offending value {offending}"
    )


def test_quantum_injection_cadence_rejects_bool():
    """True == 1 numerically but serialises differently, so it must not be accepted."""
    with pytest.raises(ValueError, match="quantum_injection_cadence"):
        NoiseConfig(quantum_injection_cadence=True)


def test_valid_enum_values_all_accepted():
    for value in TRN_PSD_MODELS:
        assert NoiseConfig(trn_psd_model=value).trn_psd_model == value
    for value in THERMAL_INTEGRATORS:
        assert NoiseConfig(thermal_integrator=value).thermal_integrator == value
    for value in QUANTUM_INJECTION_CADENCES:
        assert NoiseConfig(quantum_injection_cadence=value).quantum_injection_cadence == value


# ---------------------------------------------------------------------------
# 7. Misc surface
# ---------------------------------------------------------------------------
def test_config_is_frozen():
    cfg = NoiseConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.trn = True                                   # type: ignore[misc]


def test_describe_lists_one_line_per_enabled_channel():
    cfg = NoiseConfig.all_off(trn=True, pump_rin=True)
    text = cfg.describe()
    on_lines = [ln for ln in text.splitlines() if ln.strip().startswith("[on ]")]
    # two stochastic channels; thermal_feedback is off so it renders as [off]
    assert len(on_lines) == 2
    assert any("trn" in ln for ln in on_lines)
    assert any("pump_rin" in ln for ln in on_lines)
    assert "NOT a noise channel" in text


def test_describe_deterministic_run_says_so():
    assert "no stochastic channels enabled" in NoiseConfig.all_off().describe()
