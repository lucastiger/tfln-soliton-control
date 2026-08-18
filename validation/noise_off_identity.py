"""Bit-identity golden regression for the deterministic (noise-off) LLE solver.

Why this exists
---------------
The stochastic-LLE benchmark attributes trajectory-to-trajectory spread to the
physical noise channels. That attribution is only meaningful if the underlying
SSFM integrator is itself *exactly* reproducible: with every stochastic channel
switched off and the RNG key pinned, two runs of ``solve_lle_ssfm_jax`` must
agree to 0 ULP. This module pins that down as a golden-file regression.

What is compared, and why ``U_int_history`` is the canary
--------------------------------------------------------
Four arrays per parameter set:

``E_final``          the exact complex field at round trip ``t_slow - 1``;
``E_snapshots``      the strided snapshot history (the state the labeler sees);
``U_int_history``    per-round-trip intracavity energy [J], recorded EVERY step;
``delta_T_history``  the thermo-optic temperature history [K].

``U_int_history`` is the early-warning canary. It is a reduction over the whole
fast-time grid recorded at full cadence, so it accumulates the least-significant
bits of every mode on every round trip — it drifts first, long before the
snapshot grid or the final field visibly move. It is also the *input* to the
thermo-optic ODE (``P_abs = kappa_i * <|E|^2>`` drives ``delta_T``, which feeds
back into the detuning), so a 1 ULP wobble there is amplified by the feedback
loop rather than averaged away. Never drop it from the comparison.

Determinism caveats this harness makes visible rather than hides
---------------------------------------------------------------
Bit-identity of a floating-point reduction is a property of the *toolchain*, not
only of the source. XLA is free to pick different reduction orders and different
fused kernels across jax/jaxlib versions, across backends (CPU vs GPU), and
across CPU microarchitectures. The #1 observed cause of ULP drift is a
jax/jaxlib/numpy version change, so the exact versions are recorded in a
sidecar next to every golden file and :func:`compare_to_golden` WARNS LOUDLY on
a mismatch instead of failing: a version-induced difference is a fact about the
environment, and silently failing the test would blame the physics for it.

The solver is called exactly as an ordinary user would call it — no
pre-compilation, no shared jitted wrapper, no ``jax.jit`` reuse engineered
across parameter sets — so nothing here can reorder operations relative to
production use. (``load_dint_grid`` and ``_physical_state_labeler`` carry their
own caches inside the solver module; those are part of the ordinary user path
and are deliberately left alone.)

Configuration
-------------
Committed ``config/sin_params.yaml`` and the measured integrated-dispersion grid
from ``config/pyLLE_dispersion_w4400_h800.csv``. Detuning follows the repo
convention ``delta_omega = omega_res - omega_pump`` (positive = red-detuned =
soliton side), specified per parameter set as a multiple of the resolved total
``kappa``. Noise: ``NoiseConfig.all_off()`` verbatim.

Note on ``all_off()`` and the thermal ODE: ``all_off()`` leaves
``thermal_feedback`` at its field default (``False``), but that switch is
declarative — no solver code path reads it — so the deterministic thermo-optic
ODE runs regardless and ``delta_T_history`` is a live, non-trivial array here.

CLI
---
::

    python -m validation.noise_off_identity --write-golden
    python -m validation.noise_off_identity --check --strict
    python -m validation.noise_off_identity --check --device cpu

Reference: Herr, Tikan & Kippenberg, arXiv:2604.05897v1 (7 Apr 2026).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "PARAM_SETS",
    "ARRAY_NAMES",
    "DEFAULT_GOLDEN_DIR",
    "run_deterministic",
    "write_golden",
    "compare_to_golden",
    "main",
]

# ---------------------------------------------------------------------------
# Paths (repo-relative, resolved from this file so cwd never matters)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"
DISPERSION_CSV = REPO_ROOT / "config" / "pyLLE_dispersion_w4400_h800.csv"
DEFAULT_GOLDEN_DIR = "tests/data/golden"

#: Arrays stored in every golden ``.npz``, in comparison order. ``U_int_history``
#: is listed first on purpose: it is the array that drifts first, so a report
#: read top-down surfaces the earliest evidence of a toolchain change.
ARRAY_NAMES = ("U_int_history", "E_final", "E_snapshots", "delta_T_history")

#: Expected dtype of each golden array. Checked, never coerced — a dtype change
#: is itself a regression and must not be papered over by a silent cast.
ARRAY_DTYPES = {
    "U_int_history": np.dtype(np.float64),
    "E_final": np.dtype(np.complex128),
    "E_snapshots": np.dtype(np.complex128),
    "delta_T_history": np.dtype(np.float64),
}

#: Non-strict tolerance. Absolute only (``rtol=0``): the four arrays span many
#: orders of magnitude (|E|^2 ~ 1e-10 J down to delta_T ~ 1e-6 K), and a relative
#: tolerance would silently license large absolute excursions on the big entries.
ATOL = 1e-13
RTOL = 0.0

#: Solver knobs held fixed across every parameter set, so the only things that
#: vary are the four dimensions the sets actually sweep.
SNAPSHOT_INTERVAL = 10

PARAM_SETS = [
    dict(name="s256_short",  n_tau=256,  t_slow=2000, dw_over_kappa=4.0,  n_substeps=1),
    dict(name="s512_prod",   n_tau=512,  t_slow=5000, dw_over_kappa=10.0, n_substeps=1),
    dict(name="s512_sub4",   n_tau=512,  t_slow=5000, dw_over_kappa=10.0, n_substeps=4),
    dict(name="s1024_near",  n_tau=1024, t_slow=2000, dw_over_kappa=2.0,  n_substeps=1),
]


# ---------------------------------------------------------------------------
# Device selection — must happen BEFORE jax is imported
# ---------------------------------------------------------------------------

def _select_device(device: str | None) -> None:
    """Pin the JAX backend via ``JAX_PLATFORMS``, before the first jax import.

    ``simulator.lle_solver`` forces ``jax_enable_x64`` at import time, so the
    backend has to be chosen before any solver import happens; that is why every
    jax/simulator import in this module is deliberately lazy.

    ``gpu`` maps to ``cuda`` and is *not* given a CPU fallback: silently running
    a "GPU" identity check on CPU would make a backend-portability failure look
    like a pass.
    """
    if device is None:
        return
    if "jax" in sys.modules:
        warnings.warn(
            "jax was already imported before --device was applied; the backend "
            f"cannot be changed to {device!r} in this process. Re-run with "
            "--device as the first thing the process does.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    os.environ["JAX_PLATFORMS"] = {"cpu": "cpu", "gpu": "cuda"}[device]


def _import_solver():
    """Import the solver lazily and return the handful of names we need."""
    import jax  # noqa: PLC0415  (deliberately deferred; see _select_device)

    from simulator.lle_solver import (  # noqa: PLC0415
        _load_config,
        d2_to_beta2_lle,
        load_dint_grid,
        resolve_cavity_rates,
        solve_lle_ssfm_jax,
    )
    from simulator.noise_config import NoiseConfig  # noqa: PLC0415

    return (
        jax,
        solve_lle_ssfm_jax,
        load_dint_grid,
        resolve_cavity_rates,
        d2_to_beta2_lle,
        _load_config,
        NoiseConfig,
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _library_versions() -> dict[str, str]:
    """Exact versions of the libraries that can move the last bits.

    jax/jaxlib/numpy are the ones that matter — an XLA update can change the
    reduction order of an FFT or a sum without any source change on our side.
    """
    import jax  # noqa: PLC0415
    import jaxlib  # noqa: PLC0415

    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


#: Version keys whose mismatch is the documented #1 cause of ULP drift.
_CRITICAL_VERSION_KEYS = ("jax", "jaxlib", "numpy")


def _find_param_set(name: str) -> dict[str, Any]:
    for candidate in PARAM_SETS:
        if candidate["name"] == name:
            return candidate
    raise KeyError(f"unknown param set {name!r}; known: {[p['name'] for p in PARAM_SETS]}")


def _normalise_param_set(param_set: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(param_set, str):
        return _find_param_set(param_set)
    missing = {"name", "n_tau", "t_slow", "dw_over_kappa", "n_substeps"} - set(param_set)
    if missing:
        raise KeyError(f"param_set is missing key(s): {sorted(missing)}")
    return param_set


# ---------------------------------------------------------------------------
# The deterministic run
# ---------------------------------------------------------------------------

def run_deterministic(param_set: dict[str, Any] | str, seed: int = 0) -> dict[str, Any]:
    """Run one noise-off trajectory and return its arrays plus provenance.

    Calls :func:`simulator.lle_solver.solve_lle_ssfm_jax` exactly as a user
    would: committed ``config/sin_params.yaml``, the measured ``d_int_grid``
    interpolated from ``config/pyLLE_dispersion_w4400_h800.csv``, a fixed
    ``rng_key = jax.random.PRNGKey(seed)``, and ``NoiseConfig.all_off()``.

    Parameters
    ----------
    param_set : dict or str
        One entry of :data:`PARAM_SETS`, or its ``name``. Must carry ``name``,
        ``n_tau``, ``t_slow``, ``dw_over_kappa`` [dimensionless, in units of
        kappa] and ``n_substeps``.
    seed : int, optional
        Seed for the fixed ``jax.random.PRNGKey``, default 0. Pinned, not
        swept: with every stochastic channel off it only selects the tiny
        ``1e-3*|e_cw|`` cold-start seed noise, which is nevertheless part of
        the deterministic trajectory and so must stay fixed.

    Returns
    -------
    dict
        ``U_int_history`` (n_traj, t_slow) float64 [J*s], ``E_final``
        (n_traj, n_tau) complex128 [sqrt(J)], ``E_snapshots``
        (n_traj, n_snapshots, n_tau) complex128 [sqrt(J)], ``delta_T_history``
        (n_traj, t_slow) float64 [K], plus a ``provenance`` mapping recording
        the parameter set, seed, config hashes and library versions.

    Raises
    ------
    KeyError
        If ``param_set`` names an unknown set, or a supplied dict is missing a
        required key.
    OSError
        If the committed config or the dispersion CSV cannot be read.

    Notes
    -----
    The four arrays keep the solver's leading TRAJECTORY axis (``n_traj = 1``).
    Reshaping here would be a silent transformation of the golden bytes, and
    the point of this module is that the bytes are what is compared.

    No ``Examples`` section: the smallest parameter set runs 2000 round trips
    on a 256-point grid.
    """
    param_set = _normalise_param_set(param_set)
    (
        jax,
        solve_lle_ssfm_jax,
        load_dint_grid,
        resolve_cavity_rates,
        d2_to_beta2_lle,
        _load_config,
        NoiseConfig,
    ) = _import_solver()

    n_tau = int(param_set["n_tau"])
    t_slow = int(param_set["t_slow"])
    n_substeps = int(param_set["n_substeps"])
    dw_over_kappa = float(param_set["dw_over_kappa"])

    physical = _load_config(str(CONFIG_PATH))
    _kappa_i, kappa_c, kappa = resolve_cavity_rates(str(CONFIG_PATH))
    beta2 = float(d2_to_beta2_lle(physical["d2_rad_per_s2"], physical["fsr_hz"]))
    pin = float(physical["pin_w"])

    # delta_omega = omega_res - omega_pump; positive = red-detuned = soliton side.
    delta_omega = dw_over_kappa * float(kappa)

    d_int_grid = load_dint_grid(n_tau, csv_path=DISPERSION_CSV).grid
    noise_config = NoiseConfig.all_off()

    solution = solve_lle_ssfm_jax(
        pin=pin,
        delta_omega=delta_omega,
        t_slow=t_slow,
        beta=[beta2],
        kappa=float(kappa),
        kappa_c=float(kappa_c),
        rng_key=jax.random.PRNGKey(seed),
        n_tau=n_tau,
        config_path=str(CONFIG_PATH),
        snapshot_interval=SNAPSHOT_INTERVAL,
        d_int_grid=d_int_grid,
        n_substeps=n_substeps,
        noise_config=noise_config,
    )

    arrays = {
        "U_int_history": np.asarray(solution["U_int_history"]),
        "E_final": np.asarray(solution["e_final"]),
        "E_snapshots": np.asarray(solution["E_snapshots"]),
        "delta_T_history": np.asarray(solution["DeltaT_history"]),
    }
    for name, arr in arrays.items():
        expected = ARRAY_DTYPES[name]
        if arr.dtype != expected:
            raise TypeError(
                f"{param_set['name']}/{name}: solver returned dtype {arr.dtype}, "
                f"expected {expected}. A dtype change is itself a regression — "
                f"investigate rather than casting."
            )

    provenance = {
        "param_set": {
            "name": str(param_set["name"]),
            "n_tau": n_tau,
            "t_slow": t_slow,
            "dw_over_kappa": dw_over_kappa,
            "n_substeps": n_substeps,
        },
        "seed": int(seed),
        "solver": {
            "entry_point": "simulator.lle_solver.solve_lle_ssfm_jax",
            "pin_w": pin,
            "delta_omega_rad_per_s": delta_omega,
            "kappa_rad_per_s": float(kappa),
            "kappa_c_rad_per_s": float(kappa_c),
            "beta2_s": beta2,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "detuning_convention": "delta_omega = omega_res - omega_pump",
        },
        "noise_config": noise_config.to_dict(),
        "noise_config_sha256": noise_config.sha256(),
        "inputs": {
            "config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "config_sha256": _sha256(CONFIG_PATH),
            "dispersion_csv": str(DISPERSION_CSV.relative_to(REPO_ROOT)),
            "dispersion_csv_sha256": _sha256(DISPERSION_CSV),
        },
        # The #1 cause of ULP drift. compare_to_golden warns loudly, never fails,
        # when any of these move.
        "versions": _library_versions(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
            "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        },
        "git_commit": _git_commit(),
        "arrays": {
            name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for name, arr in arrays.items()
        },
    }

    result: dict[str, Any] = dict(arrays)
    result["provenance"] = provenance
    return result


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------

def _golden_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"{name}.npz", out_dir / f"{name}.provenance.json"


def write_golden(out_dir: str | Path = DEFAULT_GOLDEN_DIR, seed: int = 0) -> None:
    """Write ``<name>.npz`` + ``<name>.provenance.json`` for every parameter set.

    Uses :func:`numpy.savez`, **never** ``savez_compressed``: DEFLATE output is
    not bit-stable across zlib versions and build flags, so a compressed golden
    file would compare unequal byte-for-byte on a machine whose zlib merely
    chose different match lengths — a false alarm indistinguishable from a real
    solver regression.

    Parameters
    ----------
    out_dir : str or pathlib.Path, optional
        Destination directory; a relative path is resolved against the
        repository root. Default :data:`DEFAULT_GOLDEN_DIR`.
    seed : int, optional
        Seed for the fixed PRNG key, default 0. Recorded in each sidecar.

    Returns
    -------
    None
        Writes ``<name>.npz`` and ``<name>.provenance.json`` per parameter set
        and prints one progress line each.

    Raises
    ------
    OSError
        If the directory cannot be created or a file cannot be written.
    KeyError
        Propagated from :func:`run_deterministic`.

    Notes
    -----
    Regenerating goldens is a deliberate, reviewable act: it invalidates every
    bit-identity claim made against the previous files, so it belongs in its own
    commit with the toolchain change that motivated it.

    No ``Examples`` section: it runs all four parameter sets and writes into the
    repository.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for param_set in PARAM_SETS:
        name = str(param_set["name"])
        print(f"[golden] running {name} ...", flush=True)
        result = run_deterministic(param_set, seed=seed)
        npz_path, json_path = _golden_paths(out_dir, name)

        # np.savez, NOT savez_compressed — compression is not bit-stable.
        np.savez(npz_path, **{key: result[key] for key in ARRAY_NAMES})
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result["provenance"], fh, indent=2, sort_keys=True)
            fh.write("\n")

        shapes = ", ".join(f"{k}{result[k].shape}" for k in ARRAY_NAMES)
        print(f"[golden] wrote {npz_path.name}, {json_path.name}  ({shapes})", flush=True)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _bitwise_diff_mask(current: np.ndarray, golden: np.ndarray) -> np.ndarray:
    """Elementwise mask of raw-byte differences (0 ULP means identical bytes).

    Byte-level rather than value-level so that NaN payloads and the ``+0.0`` /
    ``-0.0`` distinction are both caught: a golden regression is about the bits
    the solver emitted, not about IEEE equality.
    """
    itemsize = current.dtype.itemsize
    cur_bytes = np.ascontiguousarray(current).reshape(-1).view(np.uint8)
    gold_bytes = np.ascontiguousarray(golden).reshape(-1).view(np.uint8)
    cur_bytes = cur_bytes.reshape(current.size, itemsize)
    gold_bytes = gold_bytes.reshape(golden.size, itemsize)
    return np.any(cur_bytes != gold_bytes, axis=1).reshape(current.shape)


def _diff_stats(current: np.ndarray, golden: np.ndarray) -> tuple[float, float]:
    """(max abs diff, max rel diff), NaN-in-both counted as agreement."""
    if current.size == 0:
        return 0.0, 0.0
    both_nan = np.isnan(current) & np.isnan(golden)
    abs_diff = np.where(both_nan, 0.0, np.abs(current - golden))
    denom = np.abs(golden)
    nonzero = denom > 0.0
    rel_diff = np.where(
        nonzero,
        abs_diff / np.where(nonzero, denom, 1.0),
        np.where(abs_diff > 0.0, np.inf, 0.0),
    )
    return float(np.max(abs_diff)), float(np.max(rel_diff))


def _array_report(
    name: str, current: np.ndarray, golden: np.ndarray, strict: bool
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "name": name,
        "shape": list(current.shape),
        "golden_shape": list(golden.shape),
        "dtype": str(current.dtype),
        "golden_dtype": str(golden.dtype),
    }
    if current.shape != golden.shape or current.dtype != golden.dtype:
        report.update(
            equal=False,
            bitwise_equal=False,
            value_equal=False,
            allclose=False,
            max_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            first_diff_index=None,
            n_diff=int(max(current.size, golden.size)),
            note="shape/dtype mismatch — elementwise comparison not possible",
        )
        return report

    bit_mask = _bitwise_diff_mask(current, golden)
    close = np.isclose(current, golden, atol=ATOL, rtol=RTOL, equal_nan=True)
    tol_mask = ~close
    max_abs, max_rel = _diff_stats(current, golden)

    mask = bit_mask if strict else tol_mask
    n_diff = int(np.count_nonzero(mask))
    first_index = (
        tuple(int(i) for i in np.unravel_index(int(np.argmax(mask)), mask.shape))
        if n_diff
        else None
    )

    report.update(
        equal=bool(n_diff == 0),
        bitwise_equal=bool(not bit_mask.any()),
        value_equal=bool(np.array_equal(current, golden, equal_nan=True)),
        allclose=bool(not tol_mask.any()),
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        first_diff_index=first_index,
        n_diff=n_diff,
    )
    return report


def _check_versions(golden_provenance: dict[str, Any]) -> dict[str, Any] | None:
    """Warn loudly (never fail) when the toolchain moved under the golden file."""
    golden_versions = dict(golden_provenance.get("versions", {}))
    current_versions = _library_versions()
    mismatched = {
        key: {"golden": golden_versions.get(key), "current": current_versions.get(key)}
        for key in _CRITICAL_VERSION_KEYS
        if golden_versions.get(key) != current_versions.get(key)
    }
    if not mismatched:
        return None
    detail = ", ".join(
        f"{key}: golden={vals['golden']} current={vals['current']}"
        for key, vals in mismatched.items()
    )
    warnings.warn(
        "\n"
        "!!! TOOLCHAIN VERSION MISMATCH vs the golden sidecar !!!\n"
        f"    {detail}\n"
        "    This is the #1 cause of ULP-level drift in these arrays: XLA is free "
        "to change reduction order and kernel fusion between releases. Any "
        "difference reported below is very likely environmental, NOT a solver "
        "regression. Reproduce on the recorded versions before drawing a physics "
        "conclusion; regenerate the goldens only once you have.",
        RuntimeWarning,
        stacklevel=3,
    )
    return mismatched


def compare_to_golden(
    param_set: dict[str, Any] | str,
    strict: bool,
    golden_dir: str | Path = DEFAULT_GOLDEN_DIR,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-run a parameter set and compare it against its golden file.

    Parameters
    ----------
    param_set : dict or str
        One entry of :data:`PARAM_SETS`, or its ``name``.
    strict : bool
        ``True`` requires raw-BYTE equality of every array (0 ULP); ``False``
        uses ``numpy.allclose(atol=1e-13, rtol=0)``.
    golden_dir : str or pathlib.Path, optional
        Directory holding ``<name>.npz`` and ``<name>.provenance.json``,
        default :data:`DEFAULT_GOLDEN_DIR`.
    seed : int, optional
        Seed for the fixed PRNG key, default 0. Must match the golden's.

    Returns
    -------
    dict
        ``ok``, ``name``, ``mode``, ``n_differences``, per-array reports under
        ``arrays`` (shape, dtype, ``n_diff``, ``max_abs_diff``,
        ``max_rel_diff``, ``first_diff_index``) and ``version_mismatch``.

    Raises
    ------
    FileNotFoundError
        If the golden ``.npz`` or its provenance sidecar is missing.
    KeyError
        If ``param_set`` is unknown, or a golden file lacks one of
        :data:`ARRAY_NAMES`.
    AssertionError
        If a golden array's dtype differs from :data:`ARRAY_DTYPES`. Dtypes are
        checked, never coerced -- a dtype change is itself a regression and must
        not be papered over by a silent cast.

    Warns
    -----
    RuntimeWarning
        When the recorded jax/jaxlib/numpy versions differ from the running
        ones. XLA is free to change reduction order and kernel fusion between
        releases, so a difference under a version mismatch is very likely
        environmental. The warning never by itself fails the comparison --
        turning an environment change into a physics verdict would be worse
        than reporting both and letting a human decide.

    Notes
    -----
    The tolerance is ABSOLUTE only (``rtol = 0``). The four arrays span many
    orders of magnitude -- ``|E|**2`` of order 1e-10 J down to ``delta_T`` of
    order 1e-6 K -- and a relative tolerance would silently license large
    absolute excursions on the big entries.

    No ``Examples`` section: each call re-runs the full parameter set.
    """
    param_set = _normalise_param_set(param_set)
    name = str(param_set["name"])

    golden_dir = Path(golden_dir)
    if not golden_dir.is_absolute():
        golden_dir = REPO_ROOT / golden_dir
    npz_path, json_path = _golden_paths(golden_dir, name)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"missing golden {npz_path}. Run "
            f"`python -m validation.noise_off_identity --write-golden` first."
        )
    if not json_path.exists():
        raise FileNotFoundError(
            f"missing provenance sidecar {json_path}; the golden .npz cannot be "
            f"interpreted without the toolchain versions it was produced with."
        )

    with open(json_path, "r", encoding="utf-8") as fh:
        golden_provenance = json.load(fh)
    version_mismatch = _check_versions(golden_provenance)

    golden_seed = golden_provenance.get("seed")
    if golden_seed is not None and int(golden_seed) != int(seed):
        raise ValueError(
            f"{name}: golden was written with seed={golden_seed} but seed={seed} "
            f"was requested; the trajectories are not comparable."
        )

    result = run_deterministic(param_set, seed=seed)

    arrays: dict[str, dict[str, Any]] = {}
    with np.load(npz_path) as golden_npz:
        missing = [key for key in ARRAY_NAMES if key not in golden_npz.files]
        if missing:
            raise KeyError(f"{npz_path} is missing array(s): {missing}")
        for key in ARRAY_NAMES:
            arrays[key] = _array_report(key, result[key], golden_npz[key], strict)

    n_differences = sum(rep["n_diff"] for rep in arrays.values())
    return {
        "name": name,
        "strict": bool(strict),
        "mode": "bitwise (0 ULP)" if strict else f"allclose(atol={ATOL:g}, rtol={RTOL:g})",
        "golden_npz": str(npz_path),
        "golden_provenance": str(json_path),
        "ok": all(rep["equal"] for rep in arrays.values()),
        "n_differences": int(n_differences),
        "version_mismatch": version_mismatch,
        "arrays": arrays,
        "provenance_golden": golden_provenance,
        "provenance_current": result["provenance"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report: dict[str, Any]) -> None:
    status = "OK" if report["ok"] else "DIFF"
    print(f"\n[{status}] {report['name']}  ({report['mode']})")
    if report["version_mismatch"]:
        for key, vals in report["version_mismatch"].items():
            print(f"    WARNING version {key}: golden={vals['golden']} current={vals['current']}")
    for key in ARRAY_NAMES:
        rep = report["arrays"][key]
        flag = "ok  " if rep["equal"] else "DIFF"
        print(
            f"    {flag} {key:<16} shape={tuple(rep['shape'])!s:<18} "
            f"n_diff={rep['n_diff']:<10} max_abs={rep['max_abs_diff']:.3e} "
            f"max_rel={rep['max_rel_diff']:.3e} first={rep['first_diff_index']}"
        )
        if rep.get("note"):
            print(f"         note: {rep['note']}")


def main(argv: list[str] | None = None) -> int:
    """Write or check the golden bit-identity artifacts from the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments; ``None`` (default) reads ``sys.argv[1:]``.
        Supports ``--write-golden``, ``--check``, ``--strict``, ``--device``,
        ``--golden-dir`` and ``--seed``.

    Returns
    -------
    int
        The process exit status: 0 when every checked parameter set matched (or
        when only ``--write-golden`` was requested), 1 otherwise.

    Raises
    ------
    SystemExit
        From ``argparse`` on a malformed command line, on ``--help``, or when
        neither ``--write-golden`` nor ``--check`` was given.
    FileNotFoundError
        Propagated from :func:`compare_to_golden` for a missing golden file.

    Notes
    -----
    ``--device`` is applied BEFORE jax is imported, which is why every
    jax/simulator import in this module is deliberately lazy:
    :mod:`simulator.lle_solver` forces ``jax_enable_x64`` at import time, so the
    backend must be chosen first.

    No ``Examples`` section: every path runs all four parameter sets.
    """
    parser = argparse.ArgumentParser(
        prog="python -m validation.noise_off_identity",
        description=(
            "Golden bit-identity regression for the deterministic (noise-off) "
            "LLE solver path."
        ),
    )
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="regenerate the golden .npz + .provenance.json for every param set",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-run every param set and compare against its golden file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="compare raw bytes (0 ULP) instead of allclose(atol=1e-13, rtol=0)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=None,
        help="pin the JAX backend (applied before jax is imported)",
    )
    parser.add_argument(
        "--golden-dir",
        default=DEFAULT_GOLDEN_DIR,
        help=f"golden directory (default: {DEFAULT_GOLDEN_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the fixed jax.random.PRNGKey (default: 0)",
    )
    args = parser.parse_args(argv)

    if not args.write_golden and not args.check:
        parser.error("nothing to do: pass --write-golden and/or --check")

    _select_device(args.device)

    if args.write_golden:
        write_golden(args.golden_dir, seed=args.seed)

    if not args.check:
        return 0

    reports = [
        compare_to_golden(ps, strict=args.strict, golden_dir=args.golden_dir, seed=args.seed)
        for ps in PARAM_SETS
    ]
    for report in reports:
        _print_report(report)

    total = sum(rep["n_differences"] for rep in reports)
    failed = [rep["name"] for rep in reports if not rep["ok"]]
    mode = "strict (0 ULP)" if args.strict else f"allclose(atol={ATOL:g}, rtol={RTOL:g})"
    print(f"\n{len(reports)} param sets, mode={mode}: {total} differences")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all param sets reproduce the golden files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
