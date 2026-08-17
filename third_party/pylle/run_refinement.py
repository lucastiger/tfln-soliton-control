"""Driver for the vendored, refinable pyLLE kernel.

Lives beside the vendored kernel because it is the only thing that knows how to
invoke it: upstream's ``LLEsolver.SolveTemporal`` builds its Julia command line
inline with a fixed four-argument list and offers no hook for a fifth, so this
driver uses pyLLE for ``Analyze``/``Setup`` (which write ``ParamLLEJulia.h5``),
then invokes Julia itself with ``ARGS[5] = dt`` and reads ``ResultsJulia.h5``
directly. **Nothing in site-packages is modified.**

Reading the results file directly also means the extra ``picard_stats`` key that
patch 0002 writes is available; ``RetrieveData`` would delete both HDF5 files
before we could look at it.

Two modes, so the numpy-2 project environment and the numpy<2 pyLLE environment
never have to be the same interpreter:

* ``--worker`` runs inside the pyLLE env and does the Julia runs.
* the default (orchestrator) mode runs in the project env, builds the seed and
  the spec, shells out to the worker, and computes observables/Richardson using
  ``validation.convergence_lle``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS_DIR = REPO_ROOT / "validation" / "results"
DISP_CSV = REPO_ROOT / "config" / "pyLLE_dispersion_w4400_h800.csv"

VENDORED_JL = HERE / "ComputeLLE.jl"
PRISTINE_JL = HERE / "ComputeLLE.jl.orig"


# ===========================================================================
# worker: runs inside the pyLLE environment
# ===========================================================================
def _worker(argv) -> int:
    import h5py
    import pyLLE

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--julia-bin", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())

    mu_half = int(spec["mu_half"])
    n_modes = 2 * mu_half + 1
    t_r = float(spec["t_r_s"])
    res = spec["res"]
    seed = (np.asarray(spec["seed_real"], float) + 1j * np.asarray(spec["seed_imag"], float))

    out_runs, err = {}, None
    for run in spec["runs"]:
        sim = {
            "Pin": [spec["Pin_w"]], "f_pmp": [spec["f_pmp_hz"]],
            "phi_pmp": [spec["phi_pmp_rad"]],
            "Tscan": float(run["Tscan"]),
            "mu_sim": [-mu_half, mu_half],
            "mu_fit": spec["mu_fit"],
            "D1_manual": spec["D1_manual_rad_per_s"],
            "domega_init": -float(run["dw_start"]),
            "domega_end": -float(run["dw_end"]),
            "num_probe": int(spec["num_probe"]),
        }
        solver = pyLLE.LLEsolver(sim=sim, res=res)
        solver.Analyze(plot=False)
        # same conjugate mapping the cross-check uses: A = conj(E)/sqrt(t_r)
        solver._sim["DKS_init"] = np.conj(seed) / math.sqrt(t_r)
        solver.Setup(verbose=False)

        jl = run["jl_path"]
        cmd = [a.julia_bin, jl, solver.tmp_dir, str(run["tol"]), str(run["maxiter"]),
               str(run.get("step_factor", 0.1))]
        if run.get("pass_dt", True):
            cmd.append(str(run["dt"]))       # ARGS[5]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=run.get("timeout_s", 100000))
        wall = time.time() - t0
        stdout = proc.stdout or ""
        conv_err = "Convergence Error" in stdout       # F4: never swallowed

        rec = {"wall_s": wall, "returncode": proc.returncode,
               "convergence_error_printed": conv_err,
               "stderr_tail": (proc.stderr or "")[-800:]}
        # pyLLE's tmp_dir is a filename PREFIX, not a directory: it builds paths
        # by plain concatenation (tmp_dir + "ResultsJulia.h5"), and so must we.
        rf = Path(str(solver.tmp_dir) + "ResultsJulia.h5")
        if proc.returncode != 0 or not rf.exists():
            rec["status"] = f"ERROR rc={proc.returncode}"
            out_runs[run["name"]] = rec
            err = rec["status"]
            continue

        with h5py.File(rf, "r") as S:
            u = S["Results/u_probeReal"][:] + 1j * S["Results/u_probeImag"][:]
            det = S["Results/detuningReal"][:]
            stats = (S["Results/picard_statsReal"][:]
                     if "Results/picard_statsReal" in S else None)
        final = np.asarray(u[:, -1], dtype=np.complex128)
        rec.update({
            "status": "OK",
            "field_real": final.real.tolist(), "field_imag": final.imag.tolist(),
            "detuning_pylle_final": float(det[-1]),
            "mean_picard_iters": float(stats[0]) if stats is not None else None,
            "max_picard_iters": float(stats[1]) if stats is not None else None,
            "picard_calls": float(stats[2]) if stats is not None else None,
            "picard_fail_count": float(stats[3]) if stats is not None else None,
        })
        out_runs[run["name"]] = rec
        print(f"  [{run['name']}] dt={run['dt']} tol={run['tol']} "
              f"{wall:7.1f}s  mean_picard={rec['mean_picard_iters']}  "
              f"max={rec['max_picard_iters']}  fails={rec['picard_fail_count']}", flush=True)
        # clean up so the next run in this batch starts fresh
        for f in ("ParamLLEJulia.h5", "ResultsJulia.h5"):
            p = Path(str(solver.tmp_dir) + f)
            if p.exists():
                p.unlink()

    import scipy
    Path(a.out).write_text(json.dumps({
        "runs": out_runs, "error": err,
        "versions": {"pyLLE": getattr(pyLLE, "__version__", "4.1.2"),
                     "python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__},
    }))
    print("WORKER_OK")
    return 0


# ===========================================================================
# orchestrator helpers
# ===========================================================================
def _sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


def build_kernel_variant(patches, dest: Path) -> Path:
    """Materialize a kernel with only ``patches`` applied (for the R3 gate)."""
    sys.path.insert(0, str(HERE))
    from verify_vendor import PATCH_DIR, PRISTINE

    dest.write_text(PRISTINE.read_text())
    for name in patches:
        proc = subprocess.run(["patch", "--silent", "-p0", str(dest)],
                              input=(PATCH_DIR / name).read_text(),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"could not apply {name}: {proc.stdout}{proc.stderr}")
    return dest


def julia_env() -> dict:
    env = dict(os.environ, MPLBACKEND="Agg")
    return env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pylle-python", default=os.environ.get("PYLLE_PYTHON"))
    ap.add_argument("--julia-bin", default=os.environ.get("JULIA_BIN"))
    ap.add_argument("--mu-half", type=int, default=3300)
    ap.add_argument("--tscan", type=int, default=5000)
    ap.add_argument("--dw-start", type=float, default=16.0)
    ap.add_argument("--dw-end", type=float, default=30.0)
    ap.add_argument("--dts", default="1.0,0.5,0.25,0.125")
    ap.add_argument("--tight-tol", type=float, default=1e-8)
    ap.add_argument("--tight-maxiter", type=int, default=60)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--num-probe", type=int, default=200)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--out-tag", default="dw30k")
    args = ap.parse_args(argv)
    if not args.pylle_python or not args.julia_bin:
        print("ERROR: --pylle-python and --julia-bin are required", file=sys.stderr)
        return 2

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HERE))
    from verify_vendor import assert_vendor_integrity, unified_diff  # noqa: F401

    from validation.convergence_lle import (D2_LOCAL_COMMITTED, observables_v2,
                                            richardson, _device_and_frame)
    from validation.pylle_crosscheck import (build_pylle_dispfile, load_device,
                                             ours_to_pylle, soliton_seed)
    from simulator.lle_solver import load_dint_grid

    t_start = time.time()
    integrity = assert_vendor_integrity()
    print("vendor integrity OK:", integrity["vendored_sha256"][:16])

    dev, kappa, d1, fsr_used, t_r = _device_and_frame()
    C0 = 299_792_458.0
    csv = np.loadtxt(DISP_CSV, delimiter=",")
    mu_csv = csv[:, 0].astype(np.int64)
    f_pmp = float(csv[int(np.where(mu_csv == 0)[0][0]), 1])
    fwd = ours_to_pylle(dev, f_pmp_hz=f_pmp)
    fwd["D1_manual_rad_per_s"] = d1

    work = Path(tempfile.mkdtemp(prefix="pylle_refine_"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- R3 equivalence gate ----------------
    gate = {"skipped": bool(args.skip_gate)}
    if not args.skip_gate:
        gate.update(_equivalence_gate(args, work, dev, kappa, d1, fsr_used, t_r,
                                      fwd, f_pmp, mu_csv))
        print(f"[gate] upstream vs vendored(0001+0002, dt=1): rel_L2 = "
              f"{gate['rel_l2']:.3e}  -> {'PASS' if gate['passed'] else 'FAIL'}")
        if not gate["passed"]:
            print("FAILED (R3/F2): a patch is not minimal.", file=sys.stderr)
            return 1

    # ---------------- R4 refinement ladder ----------------
    mu_half = args.mu_half
    n_modes = 2 * mu_half + 1
    mu = np.arange(-mu_half, mu_half + 1)
    grid = load_dint_grid(n_modes, csv_path=DISP_CSV)
    d_int_nat = np.fft.fftshift(np.asarray(grid.grid, dtype=float))
    dispfile = work / "dispfile_mirrored.csv"
    m0 = int(round(dev["n0"] * fwd["L_cav_m"] / dev["pump_wavelength_m"]))
    build_pylle_dispfile(dispfile, mu, d_int_nat, f_pmp, d1, m0)

    seed = soliton_seed(n_modes, args.dw_start * kappa, kappa, dev["kappa_c_rad_per_s"],
                        dev["gamma_LLE_per_J_per_s"], dev["pin_w"], D2_LOCAL_COMMITTED)

    dts = [float(x) for x in args.dts.split(",")]
    runs = []
    for dt in dts:
        runs.append({"name": f"dt{dt}_loose", "dt": dt, "tol": 1e-2, "maxiter": 10,
                     "Tscan": args.tscan, "dw_start": args.dw_start * kappa,
                     "dw_end": args.dw_end * kappa, "jl_path": str(VENDORED_JL),
                     "timeout_s": args.max_seconds})
    for dt in dts:
        runs.append({"name": f"dt{dt}_tight", "dt": dt, "tol": args.tight_tol,
                     "maxiter": args.tight_maxiter, "Tscan": args.tscan,
                     "dw_start": args.dw_start * kappa, "dw_end": args.dw_end * kappa,
                     "jl_path": str(VENDORED_JL), "timeout_s": args.max_seconds})

    spec = {
        "mu_half": mu_half, "t_r_s": t_r, "num_probe": args.num_probe,
        "mu_fit": [-mu_half, mu_half],
        "res": {"R": fwd["R_m"], "Qi": fwd["Qi"], "Qc": fwd["Qc"],
                "gamma": fwd["gamma_nlse_per_w_per_m"], "dispfile": str(dispfile)},
        "Pin_w": fwd["Pin_w"], "f_pmp_hz": fwd["f_pmp_hz"],
        "phi_pmp_rad": fwd["phi_pmp_rad"],
        "D1_manual_rad_per_s": d1,
        "seed_real": seed.real.tolist(), "seed_imag": seed.imag.tolist(),
        "runs": runs,
    }
    pl = _call_worker(args, work, spec, "refine")

    # ---------------- observables + Richardson ----------------
    levels, fields = [], {}
    for r in runs:
        rec = pl["runs"].get(r["name"], {})
        entry = {"name": r["name"], "dt": r["dt"], "Nt": int(round(r["Tscan"] / r["dt"])),
                 "tol": r["tol"], "maxiter": r["maxiter"],
                 "wall_s": rec.get("wall_s"), "status": rec.get("status", "MISSING"),
                 "mean_picard_iters": rec.get("mean_picard_iters"),
                 "max_picard_iters": rec.get("max_picard_iters"),
                 "picard_fail_count": rec.get("picard_fail_count"),
                 "convergence_error_printed": rec.get("convergence_error_printed"),
                 "detuning_pylle_final": rec.get("detuning_pylle_final")}
        if rec.get("status") == "OK":
            if rec["wall_s"] > args.max_seconds:
                entry["status"] = "TIMEOUT"
            a = (np.asarray(rec["field_real"], float)
                 + 1j * np.asarray(rec["field_imag"], float))
            ours_conv = np.conj(a) * math.sqrt(t_r)      # into OUR convention
            entry["observables"] = observables_v2(ours_conv, t_r, mu)
            fields[f"field_{r['name']}"] = ours_conv
        levels.append(entry)
        print(f"  {entry['name']:16s} Nt={entry['Nt']:6d} "
              f"picard(mean/max)={entry['mean_picard_iters']}/{entry['max_picard_iters']} "
              f"P_peak={entry.get('observables', {}).get('P_peak_w')}")

    conv = {}
    for tag in ("loose", "tight"):
        sel = [lv for lv in levels
               if lv["name"].endswith(tag) and lv["status"] == "OK" and lv.get("observables")]
        sel.sort(key=lambda x: -x["dt"])           # coarse -> fine
        if len(sel) < 3:
            continue
        for key in ("P_peak_w", "S3_modes", "mu_DW", "dw_power_dbc", "U_mean_w", "comb_frac"):
            conv[f"{tag}:{key}"] = richardson([s["observables"][key] for s in sel],
                                              [s["dt"] for s in sel])

    payload = {
        "schema_version": "1.0",
        "provenance": _provenance(args, integrity, dev, fwd, d1, fsr_used, t_r, seed,
                                  dispfile, spec, pl, time.time() - t_start),
        "upstream": {"pylle_version": integrity["pylle_version"],
                     "julia_version": subprocess.run([args.julia_bin, "--version"],
                                                     capture_output=True, text=True).stdout.strip(),
                     "computelle_sha256": integrity["upstream_sha256"]},
        "patches": _patch_meta(),
        "equivalence_gate": gate,
        "levels": levels,
        "convergence": conv,
    }
    jp = RESULTS_DIR / f"pylle_refinement_{args.out_tag}.json"
    jp.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(RESULTS_DIR / f"pylle_refinement_{args.out_tag}_fields.npz",
                        mu=mu, seed=seed, d_int_rad_per_s=d_int_nat, **fields)
    print(f"\nwrote {jp}")
    print(f"wrote {RESULTS_DIR / f'pylle_refinement_{args.out_tag}_fields.npz'}")
    return 0


def _call_worker(args, work: Path, spec: dict, tag: str) -> dict:
    sp, op = work / f"spec_{tag}.json", work / f"out_{tag}.json"
    sp.write_text(json.dumps(spec))
    proc = subprocess.run(
        [args.pylle_python, str(Path(__file__).resolve()), "--worker",
         "--spec", str(sp), "--out", str(op), "--julia-bin", args.julia_bin],
        env=julia_env(), capture_output=True, text=True)
    tail = "\n".join((proc.stdout or "").splitlines()[-30:])
    print(tail)
    if proc.returncode != 0 or "WORKER_OK" not in (proc.stdout or ""):
        print((proc.stderr or "")[-3000:], file=sys.stderr)
        raise RuntimeError(f"pyLLE worker failed rc={proc.returncode}")
    return json.loads(op.read_text())


def _equivalence_gate(args, work, dev, kappa, d1, fsr_used, t_r, fwd, f_pmp, mu_csv) -> dict:
    """R3: vendored(0001+0002) with dt=1 must equal upstream to <= 1e-12."""
    from validation.pylle_crosscheck import build_pylle_dispfile, soliton_seed
    from validation.convergence_lle import D2_LOCAL_COMMITTED
    from simulator.lle_solver import load_dint_grid

    mu_half, tscan = 512, 200
    n_modes = 2 * mu_half + 1
    mu = np.arange(-mu_half, mu_half + 1)
    grid = load_dint_grid(n_modes, csv_path=DISP_CSV)
    d_int_nat = np.fft.fftshift(np.asarray(grid.grid, dtype=float))
    dispfile = work / "gate_dispfile.csv"
    m0 = int(round(dev["n0"] * fwd["L_cav_m"] / dev["pump_wavelength_m"]))
    build_pylle_dispfile(dispfile, mu, d_int_nat, f_pmp, d1, m0)
    seed = soliton_seed(n_modes, 16.0 * kappa, kappa, dev["kappa_c_rad_per_s"],
                        dev["gamma_LLE_per_J_per_s"], dev["pin_w"], D2_LOCAL_COMMITTED)

    # kernel with ONLY 0001+0002 -- patch 0003 reverted, per R3
    k12 = build_kernel_variant(["0001-restore-dt-cli.patch",
                                "0002-restore-tol-maxiter-cli.patch"],
                               work / "ComputeLLE_0001_0002.jl")
    dw = 20.0 * kappa
    base = {"Tscan": tscan, "dw_start": dw, "dw_end": dw, "tol": 1e-2, "maxiter": 10}
    runs = [
        dict(base, name="gate_upstream", dt=1.0, jl_path=str(PRISTINE_JL), pass_dt=False),
        dict(base, name="gate_vendored12", dt=1.0, jl_path=str(k12), pass_dt=True),
    ]
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
    out = _call_worker(args, work, spec, "gate")
    ru, rv = out["runs"]["gate_upstream"], out["runs"]["gate_vendored12"]
    if ru.get("status") != "OK" or rv.get("status") != "OK":
        return {"rel_l2": None, "passed": False,
                "note": f"gate run failed: {ru.get('status')} / {rv.get('status')}"}
    fu = np.asarray(ru["field_real"], float) + 1j * np.asarray(ru["field_imag"], float)
    fv = np.asarray(rv["field_real"], float) + 1j * np.asarray(rv["field_imag"], float)
    rel = float(np.linalg.norm(fv - fu) / np.linalg.norm(fu))
    return {"rel_l2": rel, "passed": bool(rel <= 1e-12), "tolerance": 1e-12,
            "mu_half": mu_half, "Tscan": tscan, "dt": 1.0, "tol": 1e-2, "maxiter": 10,
            "note": "vendored with 0001+0002 only (0003 reverted) vs pristine upstream",
            "upstream_mean_picard": ru.get("mean_picard_iters"),
            "vendored_mean_picard": rv.get("mean_picard_iters")}


def _patch_meta() -> list:
    sys.path.insert(0, str(HERE))
    from verify_vendor import PATCH_DIR, PATCH_ORDER
    rationale = {
        "0001-restore-dt-cli.patch":
            "dt was pinned at 1 round trip; every live use is already consistent with "
            "a step of dt round trips, so exposing it as ARGS[5] makes the kernel "
            "refinable without touching the scheme.",
        "0002-restore-tol-maxiter-cli.patch":
            "lines 278-279 unconditionally clobbered the CLI tol/maxiter parsed at "
            "46-47, making the knob dead; deleting them revives it. Picard counters "
            "added so the iteration count is observable.",
        "0003-probe-final-step.patch":
            "the probe cadence never fires on the last step, so the returned field was "
            "at 29.9328 kappa instead of 30 kappa; the final slot is now overwritten "
            "with the true end-of-run state.",
    }
    out = []
    for n in PATCH_ORDER:
        text = (PATCH_DIR / n).read_text()
        out.append({
            "name": n,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "lines_changed": sum(1 for ln in text.splitlines()
                                 if (ln.startswith("+") or ln.startswith("-"))
                                 and not ln.startswith(("+++", "---"))),
            "rationale": rationale[n],
        })
    return out


def _provenance(args, integrity, dev, fwd, d1, fsr_used, t_r, seed, dispfile,
                spec, pl, wall) -> dict:
    depot = os.environ.get("JULIA_DEPOT_PATH", "")
    jlpkg = {}
    try:
        proc = subprocess.run(
            [args.julia_bin, "-e",
             'using Pkg; for (n,v) in Pkg.dependencies(); if v.name in ("FFTW","HDF5"); '
             'println(v.name, "=", v.version); end; end'],
            capture_output=True, text=True, timeout=180, env=julia_env())
        for line in (proc.stdout or "").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                jlpkg[k.strip()] = v.strip()
    except Exception:
        pass
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "argv": list(sys.argv),
        "vendor_integrity": integrity,
        "julia_depot_path": depot,
        "julia_packages": jlpkg,
        "pylle_env_versions": pl.get("versions"),
        "res_dict": spec["res"],
        "sim_template": {k: spec[k] for k in
                         ("Pin_w", "f_pmp_hz", "phi_pmp_rad", "D1_manual_rad_per_s",
                          "mu_half", "mu_fit", "num_probe", "t_r_s")},
        "dispfile_sha256": hashlib.sha256(Path(dispfile).read_bytes()).hexdigest(),
        "seed_sha256": _sha_arr(seed),
        "physical_parameters": dev,
        "fsr_hz_used": fsr_used, "d1_rad_per_s": d1, "t_r_s": t_r,
        "Tscan": args.tscan, "dts": args.dts,
        "tight_tol": args.tight_tol, "tight_maxiter": args.tight_maxiter,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "platform": platform.platform(), "hostname": platform.node()},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_total_s": wall,
    }


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker([a for a in sys.argv[1:] if a != "--worker"]))
    sys.exit(main())
