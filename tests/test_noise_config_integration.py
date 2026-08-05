"""Integration tests: NoiseConfig wired into solve_lle_ssfm_jax.

Pins the precedence chain resolved by ``simulator.lle_solver._resolve_noise_flags``
(one test per level), the equivalence of the new and legacy "noise off" spellings,
and — most importantly — that the default committed config still reproduces
today's behaviour bit-for-bit now that a top-level ``noise:`` block exists.

Precedence, highest first:
  1. explicit solve_lle_ssfm_jax keyword argument
  2. a NoiseConfig passed as noise_config=...
  3. an EXPLICIT deprecated switch in physical_parameters
  4. the top-level `noise:` block
  5. an IMPLICIT physical gate (T_k > 0, eo_r33_m_per_v != 0)
  6. the NoiseConfig field default
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from simulator.lle_solver import (
    _detuning_noise_sequences,
    _resolve_noise_flags,
    solve_lle_ssfm_jax,
)
from simulator.noise_config import NoiseConfig
from simulator.noise_models import TotalNoise, _load_config as nm_load_cfg

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

_KAPPA = 1.519e8
_KAPPA_C = 1.215e8
_BETA2 = 1.578e-18


def _write_cfg(tmp_path, name, *, noise=None, physical_updates=None, drop_noise=False):
    """Sidecar built from the committed config, with targeted edits."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if drop_noise:
        cfg.pop("noise", None)
    if noise is not None:
        cfg.setdefault("noise", {}).update(noise)
    if physical_updates:
        pp = cfg.setdefault("physical_parameters", {})
        for k, v in physical_updates.items():
            if v is None:
                pp.pop(k, None)
            else:
                pp[k] = v
    out = tmp_path / name
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _resolve(path, **kw):
    with open(path, "r", encoding="utf-8") as f:
        physical = (yaml.safe_load(f) or {}).get("physical_parameters", {})
    return _resolve_noise_flags(physical, path, kw.pop("noise_config", None), **kw)


# ---------------------------------------------------------------------------
# Precedence — one test per level
# ---------------------------------------------------------------------------
def test_level1_kwarg_beats_noise_config(tmp_path):
    """An explicit solver kwarg outranks even an explicit NoiseConfig object."""
    path = _write_cfg(tmp_path, "l1.yaml", noise={"quantum_vacuum": False})
    resolved = _resolve(
        path,
        noise_config=NoiseConfig.all_off(quantum_vacuum=False),
        quantum_noise_enabled=True,
    )
    assert resolved.quantum_vacuum is True

    # and the kwarg can force a channel OFF against a NoiseConfig that wants it on
    resolved_off = _resolve(
        path,
        noise_config=NoiseConfig.all_off(quantum_vacuum=True),
        quantum_noise_enabled=False,
    )
    assert resolved_off.quantum_vacuum is False


def test_level2_noise_config_beats_explicit_legacy_key(tmp_path):
    """A NoiseConfig object outranks an explicit legacy physical_parameters key."""
    path = _write_cfg(
        tmp_path, "l2.yaml", physical_updates={"quantum_noise_enabled": 1}
    )
    assert _resolve(path).quantum_vacuum is True          # legacy alone wins
    resolved = _resolve(path, noise_config=NoiseConfig.all_off())
    assert resolved.quantum_vacuum is False               # object overrides it


def test_level3_explicit_legacy_key_beats_noise_block(tmp_path):
    """An explicitly-written legacy key outranks the `noise:` block.

    This is what keeps sidecar-based callers working: many build a config by
    loading the committed file, editing physical_parameters, and dumping it back,
    which carries the block along. A key the user actually wrote must not be
    silently overruled by that passenger block.
    """
    path = _write_cfg(
        tmp_path,
        "l3.yaml",
        noise={"quantum_vacuum": False, "fsr": False},
        physical_updates={"quantum_noise_enabled": 1, "fsr_noise_enabled": 1},
    )
    resolved = _resolve(path)
    assert resolved.quantum_vacuum is True
    assert resolved.fsr is True


def test_level4_noise_block_beats_implicit_physical_gate(tmp_path):
    """The block outranks the implicit T_k / r33 gates.

    This is the point of the whole exercise: trn/pyro_eo/tccr had NO explicit
    switch and could only be silenced by setting an ambient temperature to
    absolute zero or zeroing a material constant.
    """
    # T_k = 300 (implicit gate says TRN on) but the block says off -> block wins
    path_off = _write_cfg(tmp_path, "l4a.yaml", noise={"trn": False, "pyro_eo": False})
    resolved_off = _resolve(path_off)
    assert resolved_off.trn is False and resolved_off.pyro_eo is False

    # T_k = 0 (implicit gate says TRN off) but the block says on -> block wins.
    # The PHYSICS still gates it: zero variance => exact zeros (asserted below).
    path_on = _write_cfg(
        tmp_path, "l4b.yaml", noise={"trn": True}, physical_updates={"T_k": 0.0}
    )
    assert _resolve(path_on).trn is True


def test_level5_implicit_gate_used_when_no_block(tmp_path):
    """With the block removed, the legacy implicit gates still decide."""
    warm = _write_cfg(tmp_path, "l5warm.yaml", drop_noise=True)
    cold = _write_cfg(
        tmp_path, "l5cold.yaml", drop_noise=True, physical_updates={"T_k": 0.0}
    )
    assert _resolve(warm).trn is True and _resolve(warm).pyro_eo is True
    assert _resolve(cold).trn is False and _resolve(cold).pyro_eo is False


def test_level6_field_defaults_when_nothing_present(tmp_path):
    """No block and no legacy keys -> NoiseConfig defaults (everything off)."""
    path = tmp_path / "l6.yaml"
    path.write_text("physical_parameters: {}\n", encoding="utf-8")
    resolved = _resolve(path)
    assert resolved == NoiseConfig()
    assert resolved.enabled_channels == ()


def test_block_override_does_not_defeat_the_physics(tmp_path):
    """`trn: true` with T_k = 0 resolves ON but still produces exact zeros."""
    path = _write_cfg(
        tmp_path, "phys.yaml", noise={"trn": True}, physical_updates={"T_k": 0.0}
    )
    with open(path, "r", encoding="utf-8") as f:
        physical = (yaml.safe_load(f) or {}).get("physical_parameters", {})
    model = TotalNoise(physical, noise_config=_resolve(path))
    assert model.trn.enabled is True                      # the switch says on
    combined, delta_t = model.sample_with_delta_t(jax.random.PRNGKey(0), 512)
    assert np.all(np.asarray(combined) == 0.0)            # physics says zero
    assert np.all(np.asarray(delta_t) == 0.0)


# ---------------------------------------------------------------------------
# The default config still reproduces today's behaviour bit-for-bit
# ---------------------------------------------------------------------------
def test_default_config_resolves_to_todays_effective_state():
    resolved = _resolve(CONFIG_PATH)
    assert resolved.trn is True            # T_k = 300 K
    assert resolved.pyro_eo is True        # temperature-driven (r33=0 zeroes it)
    assert resolved.tccr is False          # sigma_tccr = 0 on SiN
    assert resolved.quantum_vacuum is False
    assert resolved.pump_freq_noise is False and resolved.pump_rin is False
    assert resolved.fsr is False


def test_default_detuning_sequences_bit_identical_to_legacy():
    """The block-resolved path equals the legacy per-channel rules, bit for bit."""
    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    via_solver = np.asarray(_detuning_noise_sequences(keys, 512, None))

    legacy_model = TotalNoise(nm_load_cfg(None))          # noise_config=None
    legacy = np.stack(
        [np.asarray(legacy_model.sample(k, 512), dtype=np.float64) for k in keys]
    )
    assert np.array_equal(via_solver, legacy)
    assert float(np.std(legacy)) > 0.0                    # non-trivial comparison


def test_default_solver_run_unchanged_by_noise_config_plumbing():
    """A default solve equals one with noise_config explicitly set to the resolved state."""
    kw = dict(
        pin=0.214, delta_omega=3.0 * _KAPPA, t_slow=30, beta=[_BETA2],
        kappa=_KAPPA, kappa_c=_KAPPA_C, n_tau=128, snapshot_interval=10,
        rng_key=jax.random.PRNGKey(11),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = solve_lle_ssfm_jax(**kw)
        explicit = solve_lle_ssfm_jax(noise_config=_resolve(CONFIG_PATH), **kw)
    assert np.array_equal(default["E_snapshots"], explicit["E_snapshots"])
    assert np.array_equal(default["U_int_history"], explicit["U_int_history"])


# ---------------------------------------------------------------------------
# New and legacy "noise off" spellings agree
# ---------------------------------------------------------------------------
def test_all_off_matches_write_noise_off_config_bitwise(tmp_path):
    """noise_config=all_off() == config_path=write_noise_off_config(), bit for bit."""
    from analysis.run_detuning_sweep import write_noise_off_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sidecar = str(write_noise_off_config(out_path=tmp_path / "noiseoff.yaml"))

    kw = dict(
        pin=0.214, delta_omega=3.0 * _KAPPA, t_slow=30, beta=[_BETA2],
        kappa=_KAPPA, kappa_c=_KAPPA_C, n_tau=128, snapshot_interval=10,
        rng_key=jax.random.PRNGKey(5),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via_object = solve_lle_ssfm_jax(noise_config=NoiseConfig.all_off(), **kw)
        via_sidecar = solve_lle_ssfm_jax(config_path=sidecar, **kw)

    for key in ("E_snapshots", "U_int_history", "P_trans_history",
                "delta_omega_eff_history"):
        assert np.array_equal(via_object[key], via_sidecar[key]), key


def test_write_noise_off_config_writes_both_representations(tmp_path):
    from analysis.run_detuning_sweep import write_noise_off_config

    with pytest.warns(DeprecationWarning, match="NoiseConfig.all_off"):
        path = write_noise_off_config(out_path=tmp_path / "off.yaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    # authoritative block
    assert cfg["noise"] == NoiseConfig.all_off().to_dict()
    # legacy keys retained for backward compatibility
    assert cfg["physical_parameters"]["T_k"] == 0.0
    assert cfg["physical_parameters"]["quantum_noise_enabled"] == 0
    assert cfg["physical_parameters"]["pump_noise_enabled"] == 0


def test_unknown_key_in_noise_block_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "noise:\n  trn_noise: true\nphysical_parameters: {}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown key"):
        _resolve(path)
