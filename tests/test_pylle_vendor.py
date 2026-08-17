"""Tests for the vendored, refinable pyLLE Julia kernel.

The hash and patch tests are pure-Python and always run. The three tests that
need Julia are skipped when the pyLLE environment is not available (it lives
outside the repo, is several hundred MB, and cannot share this interpreter
because pyLLE requires numpy<2). Point them at an environment with

    PYLLE_PYTHON=/path/to/pylle-env/bin/python
    JULIA_BIN=/path/to/pylle-env/bin/julia
    JULIA_DEPOT_PATH=/path/to/juliadepot

and they run. They are NOT xfail and NOT deleted: with the environment present
they must pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VEND = REPO / "third_party" / "pylle"
sys.path.insert(0, str(VEND))

from verify_vendor import (  # noqa: E402
    PATCH_DIR,
    PATCH_ORDER,
    PRISTINE,
    UPSTREAM_SHA256,
    VENDORED,
    VENDORED_SHA256,
    apply_patches,
    assert_vendor_integrity,
    sha256_file,
    sha256_text,
)

PYLLE_PYTHON = os.environ.get("PYLLE_PYTHON")
JULIA_BIN = os.environ.get("JULIA_BIN")
_HAVE_JULIA = bool(PYLLE_PYTHON and JULIA_BIN
                   and Path(PYLLE_PYTHON).exists() and Path(JULIA_BIN).exists())
needs_julia = pytest.mark.skipif(
    not _HAVE_JULIA,
    reason="set PYLLE_PYTHON, JULIA_BIN (and JULIA_DEPOT_PATH) to run the Julia tests")


# --------------------------------------------------------------- pure Python
def test_upstream_hash_pinned():
    """The pristine copy must be the exact upstream file the patches assume."""
    assert sha256_file(PRISTINE) == UPSTREAM_SHA256


def test_patches_reproduce_vendor():
    """Vendored == upstream + the three patches, byte for byte."""
    rebuilt = apply_patches(PRISTINE.read_text())
    assert sha256_text(rebuilt) == sha256_file(VENDORED) == VENDORED_SHA256


def test_vendor_integrity_entrypoint():
    info = assert_vendor_integrity(check_installed=False)
    assert info["upstream_sha256"] == UPSTREAM_SHA256
    assert len(info["patches"]) == 3


def test_patches_are_minimal():
    """Each patch touches only a handful of lines; a bloated patch is a red flag."""
    limits = {"0001-restore-dt-cli.patch": 15,
              "0002-restore-tol-maxiter-cli.patch": 40,
              "0003-probe-final-step.patch": 20}
    for name in PATCH_ORDER:
        text = (PATCH_DIR / name).read_text()
        changed = sum(1 for ln in text.splitlines()
                      if (ln.startswith("+") or ln.startswith("-"))
                      and not ln.startswith(("+++", "---")))
        assert changed <= limits[name], f"{name} changed {changed} lines"


def test_dt_is_dimensionally_consistent():
    """Every live use of dt must be consistent with a step of dt round trips.

    This is the premise of the whole vendoring exercise (task F3): if any use of
    dt were inconsistent, the kernel would not be refinable and no amount of CLI
    plumbing would make it so.
    """
    src = PRISTINE.read_text().splitlines()
    uses = [ln.strip() for ln in src
            if "dt" in ln.split("#")[0].split()
            or any(tok in ln.split("#")[0] for tok in ("dt/2", "* dt", ".* dt", "/dt"))]
    live = [u for u in uses if not u.startswith("#")]
    # the four documented live uses, plus the definition
    assert any("Nt = round(t_ramp/tR/dt)" in u for u in live)
    assert any("sqrt(κext) .* dt" in u for u in live)
    assert any("FFT_Lin(Int(it)) .* dt/2" in u for u in live)
    assert any(".* dt/2" in u for u in live)


def test_upstream_clobbers_cli_tolerance():
    """Document the defect patch 0002 fixes: :46-47 parse ARGS, :278-279 overwrite."""
    src = PRISTINE.read_text()
    assert "tol = parse(Float64,ARGS[2])" in src
    assert "maxiter = parse(Int,ARGS[3])" in src
    assert "\ntol = 1e-2\nmaxiter = 10\n" in src          # the clobber
    vend = VENDORED.read_text()
    assert "tol = parse(Float64,ARGS[2])" in vend         # CLI parse retained
    assert "\ntol = 1e-2\nmaxiter = 10\n" not in vend     # clobber removed


def test_vendored_dt_defaults_to_one():
    """With ARGS[5] absent the vendored kernel must use dt = 1.0."""
    assert "dt = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0" in VENDORED.read_text()


# ------------------------------------------------------------- needs Julia
def _run_pair(tmp_path, runs):
    """Drive the worker in the pyLLE env for a list of run specs."""
    sys.path.insert(0, str(REPO))
    from simulator.lle_solver import load_dint_grid
    from validation.convergence_lle import D2_LOCAL_COMMITTED, _device_and_frame
    from validation.pylle_crosscheck import (build_pylle_dispfile, ours_to_pylle,
                                             soliton_seed)

    dev, kappa, d1, fsr_used, t_r = _device_and_frame()
    csv = np.loadtxt(REPO / "config" / "pyLLE_dispersion_w4400_h800.csv", delimiter=",")
    mu_csv = csv[:, 0].astype(np.int64)
    f_pmp = float(csv[int(np.where(mu_csv == 0)[0][0]), 1])
    fwd = ours_to_pylle(dev, f_pmp_hz=f_pmp)

    mu_half = 512
    n_modes = 2 * mu_half + 1
    mu = np.arange(-mu_half, mu_half + 1)
    grid = load_dint_grid(n_modes, csv_path=REPO / "config" / "pyLLE_dispersion_w4400_h800.csv")
    d_int_nat = np.fft.fftshift(np.asarray(grid.grid, dtype=float))
    dispfile = tmp_path / "disp.csv"
    m0 = int(round(dev["n0"] * fwd["L_cav_m"] / dev["pump_wavelength_m"]))
    build_pylle_dispfile(dispfile, mu, d_int_nat, f_pmp, d1, m0)
    seed = soliton_seed(n_modes, 16.0 * kappa, kappa, dev["kappa_c_rad_per_s"],
                        dev["gamma_LLE_per_J_per_s"], dev["pin_w"], D2_LOCAL_COMMITTED)

    spec = {
        "mu_half": mu_half, "t_r_s": t_r, "num_probe": 200,
        "mu_fit": [-mu_half, mu_half],
        "res": {"R": fwd["R_m"], "Qi": fwd["Qi"], "Qc": fwd["Qc"],
                "gamma": fwd["gamma_nlse_per_w_per_m"], "dispfile": str(dispfile)},
        "Pin_w": fwd["Pin_w"], "f_pmp_hz": fwd["f_pmp_hz"],
        "phi_pmp_rad": fwd["phi_pmp_rad"], "D1_manual_rad_per_s": d1,
        "seed_real": seed.real.tolist(), "seed_imag": seed.imag.tolist(),
        "runs": runs,
    }
    sp, op = tmp_path / "spec.json", tmp_path / "out.json"
    sp.write_text(json.dumps(spec))
    proc = subprocess.run(
        [PYLLE_PYTHON, str(VEND / "run_refinement.py"), "--worker",
         "--spec", str(sp), "--out", str(op), "--julia-bin", JULIA_BIN],
        capture_output=True, text=True, env=dict(os.environ, MPLBACKEND="Agg"))
    assert "WORKER_OK" in (proc.stdout or ""), (proc.stdout or "")[-3000:] + (proc.stderr or "")[-2000:]
    return json.loads(op.read_text())["runs"]


@needs_julia
def test_dt_default_is_one(tmp_path):
    """Vendored kernel with NO dt argument == vendored kernel with dt=1.0."""
    kappa_dw = 20.0 * (3.038e7 + 1.215e8)
    base = {"Tscan": 100, "dw_start": kappa_dw, "dw_end": kappa_dw,
            "tol": 1e-2, "maxiter": 10, "jl_path": str(VENDORED)}
    runs = [dict(base, name="nodt", dt=1.0, pass_dt=False),
            dict(base, name="dt1", dt=1.0, pass_dt=True)]
    out = _run_pair(tmp_path, runs)
    a = np.asarray(out["nodt"]["field_real"]) + 1j * np.asarray(out["nodt"]["field_imag"])
    b = np.asarray(out["dt1"]["field_real"]) + 1j * np.asarray(out["dt1"]["field_imag"])
    assert np.linalg.norm(a - b) == 0.0


@needs_julia
def test_equivalence_short_run(tmp_path):
    """R3: patches 0001+0002 (0003 reverted) must reproduce upstream to <= 1e-12."""
    sys.path.insert(0, str(VEND))
    from run_refinement import build_kernel_variant

    k12 = build_kernel_variant(["0001-restore-dt-cli.patch",
                                "0002-restore-tol-maxiter-cli.patch"],
                               tmp_path / "k12.jl")
    kappa_dw = 20.0 * (3.038e7 + 1.215e8)
    base = {"Tscan": 200, "dw_start": kappa_dw, "dw_end": kappa_dw,
            "tol": 1e-2, "maxiter": 10}
    runs = [dict(base, name="upstream", dt=1.0, jl_path=str(PRISTINE), pass_dt=False),
            dict(base, name="vendored12", dt=1.0, jl_path=str(k12), pass_dt=True)]
    out = _run_pair(tmp_path, runs)
    a = np.asarray(out["upstream"]["field_real"]) + 1j * np.asarray(out["upstream"]["field_imag"])
    b = np.asarray(out["vendored12"]["field_real"]) + 1j * np.asarray(out["vendored12"]["field_imag"])
    rel = np.linalg.norm(b - a) / np.linalg.norm(a)
    assert rel <= 1e-12, f"equivalence gate failed: rel L2 = {rel:.3e}"


@needs_julia
def test_tol_is_honoured(tmp_path):
    """The direct test that the CLI tolerance is live again.

    At the upstream default (1e-2) the Picard loop exits after exactly 2
    iterations; at 1e-8 it must take strictly more, or the knob is still dead.
    """
    kappa_dw = 20.0 * (3.038e7 + 1.215e8)
    base = {"Tscan": 100, "dw_start": kappa_dw, "dw_end": kappa_dw,
            "dt": 1.0, "jl_path": str(VENDORED)}
    runs = [dict(base, name="loose", tol=1e-2, maxiter=10),
            dict(base, name="tight", tol=1e-8, maxiter=60)]
    out = _run_pair(tmp_path, runs)
    loose = out["loose"]["mean_picard_iters"]
    tight = out["tight"]["mean_picard_iters"]
    assert loose is not None and tight is not None, "picard stats were not recorded"
    assert tight > 2.0, f"tight tolerance still exits at {tight} iterations; knob is dead"
    assert tight > loose, f"tight ({tight}) must exceed loose ({loose})"
