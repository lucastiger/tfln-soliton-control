"""Tests for :mod:`simulator.provenance` and the solver's opt-in stamp.

Covers the five properties a provenance record has to have before a benchmark
table can lean on it:

1. ``solve_lle_ssfm_jax(..., provenance=True)`` returns a stamp with every
   required key -- and ``provenance=False`` (the default) leaves the result
   dict unchanged key-for-key, so the golden whole-dict hash in
   ``tests/test_regression_figures.py`` is untouched.
2. ``noise_config_sha256`` changes iff the :class:`NoiseConfig` changes.
3. ``physical_parameters_sha256`` pins the resolved VALUES, so it survives YAML
   key reordering and comment edits.
4. ``save_artifact``/``load_artifact`` round-trip, and ``load_artifact``
   refuses an artifact whose sidecar is missing.
5. ``provenance_stamp(legacy=True)`` still reproduces the committed
   ``quantum_noise_report.json`` stamp shape.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from simulator.noise_config import NoiseConfig
from simulator.provenance import (
    ARXIV_REF,
    config_block_sha256,
    env_fingerprint,
    load_artifact,
    noise_config_sha256,
    provenance_stamp,
    save_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"
QUANTUM_REPORT_JSON = REPO_ROOT / "analysis" / "results" / "quantum_noise_report.json"

#: Keys the solver stamp must carry. Anything missing here makes a run
#: unreproducible in practice.
REQUIRED_STAMP_KEYS = {
    "script",
    "caller",
    "git_commit",
    "physical_parameters_sha256",
    "noise_config_sha256",
    "seed",
    "env_fingerprint",
    "arxiv_ref",
    "generated_utc",
}

#: A cheap-but-real solve. Small n_tau/t_slow: these tests are about the stamp,
#: not the physics, and the default flags keep every noise channel off.
_SOLVE_KWARGS = dict(
    pin=0.214,
    delta_omega=10.0 * 1.519e8,
    t_slow=12,
    beta=[1.578e-18],
    kappa=1.519e8,
    kappa_c=1.215e8,
    n_tau=64,
    snapshot_interval=6,
)


def _solve(**overrides):
    from simulator.lle_solver import solve_lle_ssfm_jax

    kwargs = dict(_SOLVE_KWARGS)
    kwargs.update(overrides)
    kwargs.setdefault("rng_key", jax.random.PRNGKey(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_lle_ssfm_jax(**kwargs)


# ---------------------------------------------------------------------------
# 1. solver stamp
# ---------------------------------------------------------------------------
def test_solver_provenance_has_every_required_key() -> None:
    sol = _solve(provenance=True)

    assert "provenance" in sol, "provenance=True must attach the stamp."
    stamp = sol["provenance"]
    assert isinstance(stamp, dict), f"stamp must be a dict, got {type(stamp)}."
    missing = REQUIRED_STAMP_KEYS - set(stamp)
    assert not missing, f"stamp is missing required key(s): {sorted(missing)}."

    # The stamp must be JSON-serialisable -- an artifact sidecar is its whole
    # point, and a numpy scalar or a Path would break json.dump.
    json.dumps(stamp)

    assert stamp["arxiv_ref"]["id"] == "2604.05897"
    assert stamp["arxiv_ref"]["version"] == "v1", (
        "the arXiv version pin is mandatory: equation/section numbers cited "
        "throughout the solver are v1 numbers."
    )
    assert len(stamp["physical_parameters_sha256"]) == 64
    assert len(stamp["noise_config_sha256"]) == 64
    assert stamp["seed"] == 0, "seed must be recovered from the PRNGKey."
    for key in ("python", "numpy", "jax", "backend"):
        assert key in stamp["env_fingerprint"]


def test_provenance_default_off_leaves_result_keys_unchanged() -> None:
    """The default must reproduce the legacy dict EXACTLY.

    ``tests/test_regression_figures.py`` hashes every key of this dict against
    a golden, so a new key on the default path would break an existing golden
    that must keep matching main.
    """
    plain = _solve()
    stamped = _solve(provenance=True)

    assert "provenance" not in plain
    assert set(stamped) - set(plain) == {"provenance"}, (
        "provenance=True must add exactly one key and remove none."
    )
    # ... and the physics must be untouched by the flag.
    for key in plain:
        np.testing.assert_array_equal(
            np.asarray(plain[key]),
            np.asarray(stamped[key]),
            err_msg=f"provenance flag perturbed {key!r} -- it must be pure bookkeeping.",
        )


def test_env_fingerprint_never_raises() -> None:
    fp = env_fingerprint()
    assert isinstance(fp, dict)
    for key in ("python", "platform", "numpy", "jax", "jaxlib", "backend",
                "device_kind", "blas"):
        assert key in fp, f"env_fingerprint missing {key!r}."
        assert isinstance(fp[key], str)


# ---------------------------------------------------------------------------
# 2. noise_config_sha256
# ---------------------------------------------------------------------------
def test_noise_config_sha256_stable_for_equal_configs() -> None:
    assert noise_config_sha256(NoiseConfig()) == noise_config_sha256(NoiseConfig())
    # Field ORDER at the call site is irrelevant; the value set is what counts.
    a = NoiseConfig(trn=True, seed=3)
    b = NoiseConfig(seed=3, trn=True)
    assert noise_config_sha256(a) == noise_config_sha256(b)
    # And it agrees with the manifest hash used by validation/noise_off_identity.py.
    assert noise_config_sha256(a) == a.sha256()


@pytest.mark.parametrize(
    "overrides",
    [
        {"quantum_vacuum": True},
        {"trn": True},
        {"pyro_eo": True},
        {"tccr": True},
        {"pump_freq_noise": True},
        {"pump_rin": True},
        {"fsr": True},
        {"thermal_feedback": True},
        {"trn_psd_model": "kondratiev_gorodetsky"},
        {"trn_ar1_stationary_init": True},
        {"thermal_integrator": "exponential"},
        {"quantum_injection_cadence": 1},
        {"quantum_seed_vacuum_init": False},
        {"legacy_segment_noise": False},
        {"noise_dtype": "float64"},
        {"seed": 1},
    ],
)
def test_noise_config_sha256_changes_when_any_field_changes(overrides: dict) -> None:
    """"iff" in both directions: EVERY field must move the digest."""
    base = noise_config_sha256(NoiseConfig())
    assert noise_config_sha256(NoiseConfig(**overrides)) != base, (
        f"changing {overrides} left the digest unchanged -- an ablation that "
        f"differs only in that field would be indistinguishable in a manifest."
    )


# ---------------------------------------------------------------------------
# 3. physical_parameters_sha256 canonicalisation
# ---------------------------------------------------------------------------
_YAML_A = """\
# Base config for the ordering-invariance test.
physical_parameters:
  gamma_LLE_per_J_per_s: 1.234e+19   # nonlinearity
  pump_wavelength_m: 1.55e-06
  Gamma_th: 12345.0
  intrinsic_q: 1000000.0
"""

_YAML_B = """\
# Same block, different key ORDER and different comments entirely.
physical_parameters:
  intrinsic_q: 1000000.0
  # a comment that only exists in this file
  Gamma_th: 12345.0
  pump_wavelength_m: 1.55e-06

  gamma_LLE_per_J_per_s: 1.234e+19
"""


def _physical_block(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("physical_parameters", {})


def test_physical_parameters_sha256_invariant_to_yaml_order_and_comments(
    tmp_path: Path,
) -> None:
    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    path_a.write_text(_YAML_A, encoding="utf-8")
    path_b.write_text(_YAML_B, encoding="utf-8")

    # The two files differ as BYTES -- otherwise this test proves nothing.
    assert path_a.read_bytes() != path_b.read_bytes()

    assert config_block_sha256(_physical_block(path_a)) == config_block_sha256(
        _physical_block(path_b)
    ), "the hash must pin the resolved VALUES, not the YAML byte layout."


def test_physical_parameters_sha256_changes_when_a_value_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(_YAML_A.replace("12345.0", "12346.0"), encoding="utf-8")
    base = config_block_sha256(yaml.safe_load(_YAML_A)["physical_parameters"])
    assert config_block_sha256(_physical_block(path)) != base


# ---------------------------------------------------------------------------
# 4. save_artifact / load_artifact
# ---------------------------------------------------------------------------
def test_save_load_artifact_round_trip(tmp_path: Path) -> None:
    arrays = {
        "u_int": np.linspace(0.0, 1.0, 17),
        "field": np.arange(8, dtype=np.complex128) * (1 + 2j),
    }
    meta = {"t_slow": 12, "n_tau": 64, "note": "round-trip"}
    stamp = provenance_stamp(
        "tests/test_provenance.py",
        7,
        physical_params={"gamma_LLE_per_J_per_s": 1.234e19},
    )

    npz_path = save_artifact(tmp_path / "run", arrays, meta, provenance=stamp)

    assert npz_path == tmp_path / "run.npz"
    assert npz_path.exists()
    assert (tmp_path / "run.provenance.json").exists()
    # No .tmp debris left behind.
    assert not list(tmp_path.glob("*.tmp"))

    loaded_arrays, sidecar = load_artifact(tmp_path / "run")
    assert set(loaded_arrays) == set(arrays)
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded_arrays[key], value)
    assert sidecar["provenance"] == stamp
    assert sidecar["meta"] == meta
    assert sidecar["arrays"] == sorted(arrays)

    # The stem may be given with or without the .npz suffix.
    again, _ = load_artifact(tmp_path / "run.npz")
    assert set(again) == set(arrays)


def test_dotted_stem_is_not_truncated(tmp_path: Path) -> None:
    """``sweep_v1.2`` must not collapse onto ``sweep_v1.npz``.

    ``Path.with_suffix`` would substitute ``.2`` away and make two distinct
    runs collide on one filename -- an artifact silently overwritten by a
    different run is the exact failure provenance exists to prevent.
    """
    stamp = provenance_stamp("s", 0, physical_params={"a": 1.0})
    p1 = save_artifact(tmp_path / "sweep_v1.2", {"x": np.zeros(2)}, {}, provenance=stamp)
    p2 = save_artifact(tmp_path / "sweep_v1.3", {"x": np.ones(2)}, {}, provenance=stamp)

    assert p1.name == "sweep_v1.2.npz"
    assert p2.name == "sweep_v1.3.npz"
    np.testing.assert_array_equal(load_artifact(p1)[0]["x"], np.zeros(2))
    np.testing.assert_array_equal(load_artifact(p2)[0]["x"], np.ones(2))


def test_load_artifact_raises_without_sidecar(tmp_path: Path) -> None:
    """An unstamped .npz must NOT load.

    This is the failure that silently poisons a benchmark table: arrays land,
    the process dies before the stamp is written, and a later run reads them
    back as if they were attributable.
    """
    stamp = provenance_stamp(
        "tests/test_provenance.py", 0, physical_params={"a": 1.0}
    )
    save_artifact(tmp_path / "run", {"x": np.zeros(3)}, {}, provenance=stamp)
    (tmp_path / "run.provenance.json").unlink()

    with pytest.raises(FileNotFoundError, match="provenance sidecar missing"):
        load_artifact(tmp_path / "run")


def test_load_artifact_raises_without_arrays(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="artifact arrays missing"):
        load_artifact(tmp_path / "never-written")


def test_save_artifact_overwrites_atomically(tmp_path: Path) -> None:
    """A second save replaces BOTH files -- no stale sidecar survives."""
    stamp_1 = provenance_stamp("s", 1, physical_params={"a": 1.0})
    stamp_2 = provenance_stamp("s", 2, physical_params={"a": 2.0})
    save_artifact(tmp_path / "run", {"x": np.zeros(3)}, {"v": 1}, provenance=stamp_1)
    save_artifact(tmp_path / "run", {"x": np.ones(5)}, {"v": 2}, provenance=stamp_2)

    arrays, sidecar = load_artifact(tmp_path / "run")
    np.testing.assert_array_equal(arrays["x"], np.ones(5))
    assert sidecar["provenance"]["seed"] == 2
    assert sidecar["meta"]["v"] == 2


# ---------------------------------------------------------------------------
# 5. legacy stamp still matches the committed report
# ---------------------------------------------------------------------------
def test_legacy_stamp_reproduces_committed_quantum_noise_report_shape() -> None:
    committed = json.loads(QUANTUM_REPORT_JSON.read_text())["provenance"]

    stamp = provenance_stamp(
        "analysis/quantum_noise_report.py", 0, legacy=True, config_path=CONFIG_PATH
    )

    assert list(stamp) == list(committed), (
        "legacy stamp key set/order drifted from the committed report."
    )
    assert stamp["script"] == committed["script"]
    assert stamp["seed"] == committed["seed"]
    # Full 40-hex commit hash, not the 12-char short form the new mode uses.
    assert len(stamp["git_commit"]) == len(committed["git_commit"]) == 40
    assert len(stamp["base_config_sha256"]) == len(committed["base_config_sha256"]) == 64
    # Legacy mode hashes the config's whole-file BYTES (not the resolved block).
    # It is deliberately NOT compared against the committed digest: sin_params.yaml
    # has been edited since that report was generated, so the two differ by
    # design -- what must not drift is the hashing RULE.
    import hashlib

    assert stamp["base_config_sha256"] == hashlib.sha256(
        CONFIG_PATH.read_bytes()
    ).hexdigest()
    assert stamp["base_config_sha256"] != config_block_sha256(
        _physical_block(CONFIG_PATH)
    ), "legacy (whole-file) and new (resolved-block) hashes must not be conflated."

    # Legacy mode carries none of the new-mode fields.
    for absent in ("physical_parameters_sha256", "quick", "generated_utc"):
        assert absent not in stamp


def test_legacy_stamp_requires_config_path() -> None:
    with pytest.raises(ValueError, match="config_path"):
        provenance_stamp("s", 0, legacy=True)


def test_new_stamp_requires_physical_params() -> None:
    with pytest.raises(ValueError, match="physical_params"):
        provenance_stamp("s", 0)


def test_analysis_shim_still_exports_provenance_stamp() -> None:
    """The old import path keeps working, but warns."""
    import importlib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.reload(importlib.import_module("analysis._provenance"))

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "analysis._provenance must emit a DeprecationWarning on import."
    )
    from simulator.provenance import provenance_stamp as promoted

    assert module.provenance_stamp is promoted
    assert module.config_block_sha256 is config_block_sha256
    assert module.ARXIV_REF is ARXIV_REF
