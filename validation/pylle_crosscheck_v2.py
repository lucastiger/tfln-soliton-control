"""Cross-code check against pyLLE, v2: convention audit + convergence-envelope containment.

WHAT THIS IS, AND WHAT IT CANNOT ESTABLISH
------------------------------------------
A cross-code check cannot establish that either code is right. Two independent
implementations of the same PDE agree only to the level at which they discretize
it, and agreement cannot distinguish "both right" from "both wrong in the same
way". ``validation/analytic_cw.py`` (exact mathematics to 1e-12) and
``validation/mms.py`` / ``validation/convergence.py`` (observed order of
accuracy) are strictly stronger claims and are where quantitative verification
of this solver actually rests.

What this check *can* establish is two things exact mathematics cannot:

1. **Convention and bookkeeping.** The sign of the detuning, the direction of
   the dispersion, factors of 2*pi, the mode-index origin, which resonance the
   pump sits on. Those are the errors that silently produce plausible-looking
   solitons. The nine findings in ``validation/pylle_crosscheck.py`` are that
   audit; they are carried over here **verbatim, by import**, not re-derived.

2. **Envelope containment.** Whether the two codes, each refined along its own
   ladder, converge to the *same limit*. That is a question about the continuum
   solution both codes claim to approximate, and unlike a single-point
   comparison it can be answered without either code standing in for the truth.

WHY v1 IS SUPERSEDED
--------------------
v1 compared ours(n_substeps=1) against pyLLE(dt=1) and read the residual gaps as
cross-code disagreement. That comparison is structurally unable to distinguish
"both codes right, both coarse" from "one code wrong": at one step per round trip
neither code is near its own limit, and our own refinement moves the peak power
by more than the cross-code gap. v1 also recorded ``pylle_refinable: false``,
which is false -- ``dt`` is a plumbed parameter of the Julia kernel and the
``tol``/``maxiter`` CLI plumbing exists at ``ComputeLLE.jl:46-47`` and is
clobbered at ``:278-279`` (see ``third_party/pylle/``). And the two codes were
not compared at the same operating point: upstream's probe cadence returned
pyLLE's field at round trip 4976 of 5000, i.e. 29.9328 kappa against our
30.0000 kappa. See ``docs/PYLLE_STATUS_V2.md`` for each correction with its
evidence, and ``validation/results/v1/`` for the frozen v1 artifacts.

THE METHOD
----------
Both codes integrate the *same* problem: the same D_int array (pyLLE's own
refit, mirrored for it and un-mirrored for us), the same deterministic sech
seed, the same detuning trajectory ending at the same final detuning, the same
thermo-optic-off derived config. Then each is refined along its own ladder::

    ours   n_substeps = 1, 2, 4, 8, 16          (h = 1/n_substeps)
    pyLLE  dt = 1, 1/2, 1/4, 1/8                (h = dt, round trips)

and, separately, ``dt1_tight`` -- dt = 1 with the Picard tolerance at 1e-8 and
maxiter 60 instead of the shipped 1e-2/10 -- which separates the step-size
effect from the nonlinear-solve effect. At the shipped tolerance the iteration
runs exactly 2 passes at every step regardless of dt, so it is an activity test,
not a convergence test.

Per observable, each ladder is Richardson-extrapolated
(``validation.convergence_lle.richardson``) to a limit with an uncertainty band,
and the two limits are compared::

    CONTAINED     |limit_o - limit_p| <= K*sqrt(U_o^2 + U_p^2)
    SEPARATED     the gap exceeds that band and BOTH ladders converged
    INDETERMINATE either ladder is non-monotone, out of order range, or short

SEPARATED is not a failure of this task. It is the most informative outcome
available and is reported at the top of the report.

pyLLE is reported **independently and symmetrically**, never as ground truth:
every observable carries ours(each level), pyLLE(each level), both limits, both
uncertainties, and the containment verdict. None of the ground-truth field names
(the four listed in ``tests/test_pylle_crosscheck_v2.py::FORBIDDEN_KEYS``)
appears anywhere in the emitted JSON, and a test asserts that.

Acceptance criteria, classification and tolerances come from
``validation/criteria.py`` and are fingerprinted before any comparison value is
read. Observables come from ``validation/convergence_lle.py``. **This module
defines no observable and no tolerance of its own.**

RUNNING IT
----------
pyLLE needs numpy<2 and a Julia backend, so it cannot share this repo's
environment and runs out-of-process (this same file, re-executed under the pyLLE
interpreter with ``--worker``; every heavy import is function-local for that
reason)::

    python -m validation.pylle_crosscheck_v2 \\
        --pylle-python /path/to/pylle-env/bin/python \\
        --julia-bin    /path/to/pylle-env/bin/julia \\
        --vendored-kernel third_party/pylle/ComputeLLE.jl

Unlike v1, **every default here is the value the committed run used**, so a bare
re-run reproduces the committed artifacts. The argv is recorded in the JSON
regardless.

Outputs ``validation/results/pylle_crosscheck_v2.{json,png}`` and
``pylle_crosscheck_v2_fields.npz``. Exit status is nonzero if any HARD or GATED
check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

C0 = 299_792_458.0
REPO_ROOT = Path(__file__).resolve().parents[1]
DISP_CSV = REPO_ROOT / "config" / "pyLLE_dispersion_w4400_h800.csv"
RESULTS_DIR = REPO_ROOT / "validation" / "results"
THIRD_PARTY = REPO_ROOT / "third_party" / "pylle"

SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Protocol constants. Every one of these is the value the committed run used;
# v1's defaults did NOT reproduce its own committed numbers, which is the defect
# this file's defaults exist to close.
# ---------------------------------------------------------------------------
MU_HALF = 3300
N_ROUNDTRIPS_MAIN = 5000
N_ROUNDTRIPS_SCAN = 2000
DW_START_KAPPA = 16.0
DW_END_KAPPA = 30.0
SCAN_KAPPA = "3,5,8,16,30,40,50"
BISECT_ITERS = 5
OURS_LEVELS = "1,2,4,8,16"
PYLLE_LEVELS = "1,0.5,0.25,0.125"

# pyLLE's shipped Picard controls, and the tight pair used to separate the
# nonlinear-solve effect from the step-size effect (R4).
PICARD_SHIPPED = (1e-2, 10)
PICARD_TIGHT = (1e-8, 60)
NUM_PROBE = 200

# R8: a snapshot comparison of a breathing state is not a comparison of a fixed
# point. Classify at the final round trip and at these offsets before it.
STATIONARITY_OFFSETS = (1, 2, 5, 10)
STATIONARITY_TOL = 0.01          # 1% in peak power, or any label change
# R8: settle time is the leading untested explanation for the lower-edge gap.
HOLD_TIME_MULTIPLIERS = (1, 2, 4)

# R9: a comb whose edge sits above this is not spectrally contained, and any
# observable that integrates the wings is then grid-limited rather than physical.
SPECTRAL_CONTAINMENT_DBC = -100.0

# R7 / D5: fixed |mu| bands for the spectral residual table.
RESIDUAL_BANDS = ((0, 50), (50, 222), (222, 500), (500, 1000), (1000, 1500),
                  (1500, 2000), (2000, 2500), (2500, 3000), (3000, 3300))

# Fields that legitimately differ between two identical runs and are therefore
# excluded from the numerical determinism digest (T3/A7).
VOLATILE_KEYS = frozenset({
    "wall_s", "wall_clock_total_s", "timestamp_utc", "hostname", "platform",
    "work_dir", "stderr_tail", "julia_depot_path", "python", "jl_path",
})


# ==========================================================================
# pyLLE worker -- runs INSIDE the pyLLE interpreter (numpy<2, no jax)
# ==========================================================================
def _worker_main(argv: list[str]) -> int:
    """Drive the vendored kernel once per run and return raw fields.

    Modelled on ``third_party/pylle/run_refinement.py::_worker``: pyLLE is used
    for ``Analyze``/``Setup`` (which write ``ParamLLEJulia.h5``) and Julia is
    then invoked directly so ``ARGS[5] = dt`` can be passed and the
    ``picard_stats`` written by patch 0002 can be read before ``RetrieveData``
    deletes the file. **Nothing in site-packages is modified.**

    Unlike that driver, the seed is PER RUN: the existence scan needs a seed
    whose width follows its own detuning.
    """
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

    out_runs: dict = {}
    d_int_saved = None
    failure = None

    for run in spec["runs"]:
        sim = {
            "Pin": [spec["Pin_w"]],
            "f_pmp": [spec["f_pmp_hz"]],
            "phi_pmp": [spec["phi_pmp_rad"]],
            "Tscan": float(run["Tscan"]),
            "mu_sim": [-mu_half, mu_half],
            # mu_fit is PUMP-RELATIVE and pyLLE indexes its CSV as
            # ind_pmp + mu_fit[0] >= 0 (_analyzedisp.py:127).
            "mu_fit": spec["mu_fit"],
            "D1_manual": spec["D1_manual_rad_per_s"],
            # Finding 2: pyLLE's detuning is the NEGATIVE of ours.
            "domega_init": -float(run["dw_start"]),
            "domega_end": -float(run["dw_end"]),
            "num_probe": int(spec["num_probe"]),
        }
        solver = pyLLE.LLEsolver(sim=sim, res=res)
        solver.Analyze(plot=False)

        d_int = np.asarray(solver._sim["Dint"], dtype=np.float64).ravel()
        if d_int.size != n_modes:
            raise RuntimeError(f"pyLLE Dint size {d_int.size} != expected {n_modes}")
        mu0 = (n_modes - 1) // 2
        # Finding 6: the kernel re-zeros D_int at the DOMAIN CENTRE. With the
        # pump centred that is a no-op; verify rather than assume.
        if abs(float(d_int[mu0])) > 1e-6 * max(1.0, float(np.max(np.abs(d_int)))):
            raise RuntimeError(
                f"pyLLE D_int is not zero at the domain centre (D_int[mu0]="
                f"{d_int[mu0]:.6e}); the pump is not centred and Finding 6 would bite.")
        d_int_saved = d_int

        # Finding 8: pyLLE locates the pump by SEARCHING its dispersion file for
        # the mode nearest sim['f_pmp'], so the resonance it actually locked
        # onto is a property of pyLLE's own analysis, not of what we asked for.
        # Reporting it back is what makes H4 a check rather than a restatement.
        an = solver._analyze
        pump_freq = float(np.asarray(an.freq_fit).ravel()[
            int(np.ravel(an.pmp_ind_fit)[0])])
        pump_mu = int(np.ravel(an.pmp_ind)[0])

        # Findings 2, 3 and 5: the SAME deterministic seed, mapped into pyLLE's
        # convention A = conj(E)/sqrt(t_r).
        seed_ours = (np.asarray(run["seed_real"], dtype=np.float64)
                     + 1j * np.asarray(run["seed_imag"], dtype=np.float64))
        a_init = np.conj(seed_ours) / math.sqrt(t_r)
        solver._sim["DKS_init"] = a_init
        # H7: hash the array pyLLE actually holds. The orchestrator computes the
        # SAME forward map on its own seed and compares hashes, so this checks
        # the JSON transport and the conjugate map bit-for-bit. (Hashing a
        # round trip back through the inverse map would not: dividing and then
        # multiplying by sqrt(t_r) is not the identity in floating point, so
        # that comparison would fail on rounding rather than on physics.)
        seed_sha = hashlib.sha256(
            np.ascontiguousarray(a_init, dtype=np.complex128).tobytes()).hexdigest()

        solver.Setup(verbose=False)

        cmd = [a.julia_bin, run["jl_path"], solver.tmp_dir,
               str(run["tol"]), str(run["maxiter"]), str(run.get("step_factor", 0.1)),
               str(run["dt"])]                     # ARGS[5] = dt (patch 0001)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=run.get("timeout_s", 100000))
        wall = time.time() - t0
        stdout = proc.stdout or ""

        rec = {
            "wall_s": wall,
            "returncode": proc.returncode,
            # F4 of the vendoring task: a Julia-side convergence failure is
            # never swallowed.
            "convergence_failure_printed": "Convergence Error" in stdout,
            "stderr_tail": (proc.stderr or "")[-800:],
            "seed_sha256_pylle_convention": seed_sha,
            "pump_freq_hz_located_by_pylle": pump_freq,
            "pump_mu_located_by_pylle": pump_mu,
        }

        # pyLLE's tmp_dir is a filename PREFIX, not a directory: it builds paths
        # by plain concatenation, and so must we.
        rf = Path(str(solver.tmp_dir) + "ResultsJulia.h5")
        if proc.returncode != 0 or not rf.exists():
            rec["status"] = f"FAILED rc={proc.returncode}"
            out_runs[run["name"]] = rec
            failure = rec["status"]
            continue

        with h5py.File(rf, "r") as S:
            u = S["Results/u_probeReal"][:] + 1j * S["Results/u_probeImag"][:]
            det = S["Results/detuningReal"][:]
            stats = (S["Results/picard_statsReal"][:]
                     if "Results/picard_statsReal" in S else None)
        final = np.asarray(u[:, -1], dtype=np.complex128)
        if not np.all(np.isfinite(final)):
            rec["status"] = "FAILED non-finite field"
            out_runs[run["name"]] = rec
            failure = rec["status"]
            continue

        rec.update({
            "status": "OK",
            "field_real": final.real.tolist(),
            "field_imag": final.imag.tolist(),
            # patch 0003 makes the LAST probe the true end-of-run state, so this
            # is the detuning the returned field actually sits at.
            "detuning_pylle_final": float(det[-1]),
            "mean_picard_iters": float(stats[0]) if stats is not None else None,
            "max_picard_iters": float(stats[1]) if stats is not None else None,
            "picard_calls": float(stats[2]) if stats is not None else None,
            "picard_failure_count": float(stats[3]) if stats is not None else None,
        })
        out_runs[run["name"]] = rec
        print(f"  [{run['name']}] dt={run['dt']} tol={run['tol']} Tscan={run['Tscan']}"
              f"  {wall:7.1f}s  picard mean={rec['mean_picard_iters']} "
              f"max={rec['max_picard_iters']}", flush=True)

        for f in ("ParamLLEJulia.h5", "ResultsJulia.h5"):
            p = Path(str(solver.tmp_dir) + f)
            if p.exists():
                p.unlink()

    import scipy
    Path(a.out).write_text(json.dumps({
        "runs": out_runs,
        "d_int_rad_per_s": None if d_int_saved is None else np.asarray(d_int_saved).tolist(),
        "worker_failure": failure,
        "versions": {"pyLLE": getattr(pyLLE, "__version__", "4.1.2"),
                     "python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "h5py": h5py.__version__},
    }))
    print("WORKER_OK")
    return 0


# ==========================================================================
# small helpers
# ==========================================================================
def _sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _strip_volatile(obj):
    """Recursively drop the fields that legitimately differ between two runs."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def numerical_digest(payload: dict) -> str:
    """Hash every non-volatile field of the report, so two runs can be compared.

    Parameters
    ----------
    payload : dict
        The report. Its own ``numerical_digest`` key, if present, is excluded.

    Returns
    -------
    str
        64 lowercase hex characters over the payload with every key in
        :data:`VOLATILE_KEYS` removed recursively.

    Raises
    ------
    TypeError
        If a value is neither JSON-serialisable nor ``str``-representable.

    Notes
    -----
    Acceptance criterion A7: two identical runs must produce the same digest.
    Timestamps, wall-clock times, hostname and paths are stripped recursively,
    so the digest answers "did the numbers change?" and not "was this the same
    invocation?" -- the second question has an uninteresting answer.

    Examples
    --------
    >>> len(numerical_digest({"a": 1}))
    64
    >>> numerical_digest({"a": 1}) == numerical_digest({"a": 1, "wall_s": 3})
    True
    >>> numerical_digest({"a": 1}) == numerical_digest({"a": 2})
    False
    """
    stripped = _strip_volatile({k: v for k, v in payload.items()
                                if k != "numerical_digest"})
    return hashlib.sha256(
        json.dumps(stripped, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


def _fmt(v, spec="", dash="-"):
    """Format a value, rendering a missing one as a dash padded to the same width.

    Parameters
    ----------
    v : object
        The value. ``None`` and non-finite floats are treated as missing.
    spec : str, optional
        A ``format`` spec, e.g. ``"8.3f"``. Default ``""`` (plain ``str``).
    dash : str, optional
        The placeholder for a missing value, default ``"-"``.

    Returns
    -------
    str
        ``format(v, spec)`` when the value is present, otherwise ``dash``
        right-justified to the width parsed out of ``spec``.

    Raises
    ------
    ValueError
        Propagated from ``format`` for a spec the value's type does not accept.

    Notes
    -----
    A dash that ignores the field width silently shifts every later column of a
    table, which makes an unmeasured row look like a different row -- the kind
    of misreading that costs an afternoon.

    Examples
    --------
    >>> _fmt(1.5, "8.3f")
    '   1.500'
    >>> _fmt(None, "8.3f")            # same width, so the columns still line up
    '       -'
    >>> _fmt(float("nan"), "8.3f") == _fmt(None, "8.3f")
    True
    """
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        width = "".join(c for c in spec.split(".")[0] if c.isdigit())
        return f"{dash:>{width}}" if width else dash
    return format(v, spec) if spec else str(v)


# ==========================================================================
# R5 -- envelope containment
# ==========================================================================
def containment_verdict(ours_conv: dict, pylle_conv: dict, *, absolute: bool,
                        coverage: float) -> dict:
    """Do the two refinement ladders converge to the SAME limit?

    Symmetric by construction: neither code supplies the band the other is
    measured against; each contributes its own extrapolated limit and its own
    uncertainty, and the verdict is a statement about the pair.

    ``absolute`` selects the units the comparison is made in. The Richardson
    uncertainties are relative, so for an absolute observable (a mode index)
    they are scaled by that code's own limit before combining -- otherwise a
    band in "fraction of 3075" would be compared against a gap in modes.

    Parameters
    ----------
    ours_conv : dict
        Our refinement ladder's Richardson result: the extrapolated limit (in
        the observable's units) and the relative uncertainty.
    pylle_conv : dict
        pyLLE's, from its own refinement study.
    absolute : bool
        ``True`` when the observable is absolute (a mode index), so each
        relative uncertainty is scaled by that code's own limit before
        combining. Keyword-only.
    coverage : float
        Coverage factor K [dimensionless]. Keyword-only.

    Returns
    -------
    dict
        Both limits, both uncertainties, the combined band, the gap between the
        limits and the verdict.

    Raises
    ------
    KeyError
        If either ladder result is missing its limit or uncertainty.

    Notes
    -----
    Symmetric by construction: NEITHER code supplies the band the other is
    measured against. Each contributes its own extrapolated limit and its own
    uncertainty, and the verdict is a statement about the pair. An asymmetric
    test -- treating one code's answer as ground truth and the other's
    departure from it as the thing being measured -- would be a different and
    much weaker claim, and this module deliberately does not make it.

    No ``Examples`` section: a meaningful call needs two completed refinement
    ladders.
    """
    out = {
        "ours_limit": None, "ours_U": None, "ours_p": None,
        "ours_status": ours_conv.get("status") if ours_conv else "MISSING",
        "pylle_limit": None, "pylle_U": None, "pylle_p": None,
        "pylle_status": pylle_conv.get("status") if pylle_conv else "MISSING",
        "limit_gap": None, "band": None, "coverage_factor": coverage,
        "kind": "absolute" if absolute else "relative",
        "verdict": "INDETERMINATE", "reason": None,
    }
    if not ours_conv or not pylle_conv:
        out["reason"] = "a ladder is missing this observable"
        return out

    out["ours_p"] = ours_conv.get("observed_order")
    out["pylle_p"] = pylle_conv.get("observed_order")

    bad = [f"ours: {ours_conv.get('status')}" if ours_conv.get("status") != "OK" else None,
           f"pylle: {pylle_conv.get('status')}" if pylle_conv.get("status") != "OK" else None]
    bad = [b for b in bad if b]
    if bad:
        # Still report whatever each side did produce -- an INDETERMINATE
        # verdict with no numbers is not evidence, it is silence.
        out["ours_limit"] = ours_conv.get("extrapolated")
        out["pylle_limit"] = pylle_conv.get("extrapolated")
        out["ours_U"] = ours_conv.get("U_obs")
        out["pylle_U"] = pylle_conv.get("U_obs")
        out["reason"] = "; ".join(bad)
        return out

    lo, lp = float(ours_conv["extrapolated"]), float(pylle_conv["extrapolated"])
    uo, up = float(ours_conv["U_obs"]), float(pylle_conv["U_obs"])
    out.update({"ours_limit": lo, "pylle_limit": lp, "ours_U": uo, "pylle_U": up})

    if absolute:
        uo_c, up_c = uo * abs(lo), up * abs(lp)
        gap = abs(lo - lp)
    else:
        uo_c, up_c = uo, up
        denom = max(abs(lo), abs(lp))
        gap = abs(lo - lp) / denom if denom else 0.0
    band = coverage * math.sqrt(uo_c ** 2 + up_c ** 2)
    out.update({"limit_gap": gap, "band": band,
                "ours_U_in_comparison_units": uo_c,
                "pylle_U_in_comparison_units": up_c,
                "verdict": "CONTAINED" if gap <= band else "SEPARATED"})
    if out["verdict"] == "SEPARATED":
        out["reason"] = (f"both ladders converged (p_ours={out['ours_p']:.3f}, "
                         f"p_pylle={out['pylle_p']:.3f}) but the limits differ by "
                         f"{gap:.4g} against a combined band of {band:.4g}")
    return out


# ==========================================================================
# our solver
# ==========================================================================
def run_ours(ramp: np.ndarray, seed: np.ndarray, d_int_fftorder: np.ndarray,
             dev: dict, config_path: str, n_substeps: int = 1) -> dict:
    """Run this repository's solver over a detuning ramp and report the residual.

    Parameters
    ----------
    ramp : numpy.ndarray
        Detuning per round trip [rad/s], shape ``(t_slow,)``.
    seed : numpy.ndarray
        Initial field, shape ``(n_modes,)``, complex, units sqrt(J).
    d_int_fftorder : numpy.ndarray
        ``D_int(mu)`` [rad/s], shape ``(n_modes,)``, in FFT-bin order.
    dev : dict
        Device parameters from
        :func:`~validation.pylle_crosscheck.load_device`.
    config_path : str
        The derived config both codes are driven from.
    n_substeps : int, optional
        Strang sub-steps per round trip [dimensionless], default 1 -- matching
        pyLLE's one step per round trip.

    Returns
    -------
    dict
        ``field`` (n_modes,) complex128 [sqrt(J)], the wall-clock time [s], and
        the thermo-optic residual: the largest relative deviation of
        ``delta_omega_eff`` from the programmed ramp [dimensionless].

    Raises
    ------
    ValueError
        Propagated from the solver for a shape mismatch.
    ImportError
        If JAX is unavailable.

    Notes
    -----
    The same call as :func:`~validation.pylle_crosscheck.run_ours` and
    :func:`~validation.convergence_lle.run_level`, except that it RETURNS the
    thermo-optic residual instead of only asserting on it: in v2 that residual
    is criterion H6, a reported quantity with a threshold, rather than an
    internal sanity check.

    No ``Examples`` section: it runs a multi-thousand-round-trip solve on a
    6601-mode grid.
    """
    import jax
    from simulator.lle_solver import solve_lle_ssfm_jax
    from simulator.noise_config import NoiseConfig

    kappa = dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]
    n_steps = int(np.asarray(ramp).size)
    t0 = time.time()
    out = solve_lle_ssfm_jax(
        pin=dev["pin_w"],
        delta_omega=np.asarray(ramp, dtype=float),
        t_slow=n_steps,
        beta=[0.0, 0.0],                      # unused: d_int_grid drives dispersion
        kappa=kappa,
        kappa_c=dev["kappa_c_rad_per_s"],
        rng_key=jax.random.PRNGKey(0),        # unused: every noise channel is off
        n_tau=int(seed.size),
        config_path=config_path,
        snapshot_interval=n_steps,
        e0_override=np.asarray(seed, dtype=np.complex128),
        d_int_grid=np.asarray(d_int_fftorder, dtype=float),
        n_substeps=int(n_substeps),
        fine_cadence_M=1,
        noise_config=NoiseConfig.all_off(),
    )
    wall = time.time() - t0
    eff = np.asarray(out["delta_omega_eff_history"], dtype=float).ravel()
    programmed = np.asarray(ramp, dtype=float).ravel()
    scale = max(1.0, float(np.max(np.abs(programmed))))
    eff_rel = float(np.max(np.abs(eff - programmed)) / scale)
    field = np.asarray(out["e_final"], dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(field)):
        raise AssertionError(f"our solver returned a non-finite field at n_substeps={n_substeps}")
    return {"field": field, "wall_s": wall, "delta_omega_eff_max_rel": eff_rel,
            "n_substeps": int(n_substeps), "n_roundtrips": n_steps}


# ==========================================================================
# diagnostics
# ==========================================================================
def band_residual_table(dbc_ours: np.ndarray, dbc_pylle: np.ndarray,
                        mu: np.ndarray, d_int_nat: np.ndarray, t_r: float) -> list:
    """D5. Median (ours - pyLLE) dB per fixed |mu| band, with the phase budget.

    The phase columns are there to test the standing hypothesis that the wing
    gap tracks the under-resolved per-round-trip phase |D_int|*t_r. Reporting
    the band gap without them would leave that untestable.

    Parameters
    ----------
    dbc_ours, dbc_pylle : numpy.ndarray
        Per-mode spectra [dBc], shape ``(n_modes,)``, from the same estimator.
    mu : numpy.ndarray
        Mode numbers [dimensionless], shape ``(n_modes,)``.
    d_int_nat : numpy.ndarray
        ``D_int(mu)`` [rad/s], mu-centred, same shape.
    t_r : float
        Round-trip time [s].

    Returns
    -------
    list of dict
        One row per band in :data:`RESIDUAL_BANDS`: the ``mu_band``, the mode
        count, the median and 90th-percentile ``ours - pyLLE`` difference [dB],
        and the median and maximum ``|D_int|*t_r`` [rad] over that band.

    Raises
    ------
    IndexError
        If the arrays have different lengths.

    Notes
    -----
    Median rather than mean, because the wing residual is heavy-tailed: a
    handful of modes near a null dominate a mean and say nothing about the
    band.

    No ``Examples`` section: it needs both codes' spectra on a common grid.
    """
    resid = dbc_ours - dbc_pylle
    rows = []
    for lo, hi in RESIDUAL_BANDS:
        sel = (np.abs(mu) >= lo) & (np.abs(mu) < hi) if hi < RESIDUAL_BANDS[-1][1] \
            else (np.abs(mu) >= lo) & (np.abs(mu) <= hi)
        if not np.any(sel):
            continue
        phase = np.abs(d_int_nat[sel]) * t_r
        rows.append({
            "mu_band": [int(lo), int(hi)],
            "n_modes": int(np.count_nonzero(sel)),
            "median_ours_minus_pylle_db": float(np.median(resid[sel])),
            "p90_abs_db": float(np.percentile(np.abs(resid[sel]), 90)),
            "median_abs_Dint_t_r": float(np.median(phase)),
            "max_abs_Dint_t_r": float(np.max(phase)),
        })
    return rows


def spectral_containment(dbc: np.ndarray, mu: np.ndarray) -> dict:
    """Check whether the comb is contained by the grid or is hitting its edge.

    Parameters
    ----------
    dbc : numpy.ndarray
        Per-mode spectrum [dBc], shape ``(n_modes,)``, mu-centred.
    mu : numpy.ndarray
        Mode numbers [dimensionless], same shape.

    Returns
    -------
    dict
        ``mu_minus_half_dbc`` and ``mu_plus_half_dbc`` [dBc] -- the two grid
        edges; ``threshold_dbc``; ``contained`` (bool); and ``warning`` --
        ``None`` when contained, otherwise a message naming the worst edge.

    Raises
    ------
    IndexError
        If ``mu`` and ``dbc`` have different lengths.

    Notes
    -----
    Criterion R9. When the comb is NOT contained, every observable that
    integrates the wings -- line counts, band powers -- is grid-limited, and a
    cross-code agreement on such a quantity would be an agreement about two
    grids rather than about two physical solutions. Hence a warning that names
    the consequence rather than a bare boolean.

    Examples
    --------
    >>> import numpy as np
    >>> n = 401
    >>> mu = np.arange(n) - n // 2
    >>> quiet = np.full(n, -200.0)
    >>> quiet[n // 2] = 0.0
    >>> result = spectral_containment(quiet, mu)
    >>> result["contained"], result["warning"] is None
    (True, True)

    A comb reaching the edge is flagged, with the reason:

    >>> loud = np.full(n, -50.0)
    >>> loud[n // 2] = 0.0
    >>> spectral_containment(loud, mu)["contained"]
    False
    """
    edge_lo = float(dbc[mu == mu.min()][0])
    edge_hi = float(dbc[mu == mu.max()][0])
    worst = max(edge_lo, edge_hi)
    return {
        "mu_minus_half_dbc": edge_lo,
        "mu_plus_half_dbc": edge_hi,
        "threshold_dbc": SPECTRAL_CONTAINMENT_DBC,
        "contained": bool(worst <= SPECTRAL_CONTAINMENT_DBC),
        "warning": (None if worst <= SPECTRAL_CONTAINMENT_DBC else
                    f"COMB NOT SPECTRALLY CONTAINED: grid edge at {worst:.1f} dBc, "
                    f"above the {SPECTRAL_CONTAINMENT_DBC:.0f} dBc containment "
                    f"threshold. Any observable that integrates the wings "
                    f"(line counts, band powers) is grid-limited here."),
    }


def classify_field(field: np.ndarray, t_r: float, kappa: float,
                   gamma: float) -> dict:
    """Classify one final field, and record what the classifier saw.

    ``is_single_soliton`` is a THRESHOLD test (contrast >= 5, exactly one
    circular connected component above the half-way level), so near an existence
    boundary it can flip on a small difference in a genuinely marginal state.
    Returning the contrast and the peak count alongside the boolean is what makes
    such a flip visible in the trace rather than invisible in a verdict.

    Parameters
    ----------
    field : numpy.ndarray
        Final field in OUR convention, shape ``(n_modes,)``, complex, units
        sqrt(J).
    t_r : float
        Round-trip time [s].
    kappa : float
        Total loss rate [rad/s].
    gamma : float
        ``gamma_LLE`` [J^-1 s^-1].

    Returns
    -------
    dict
        ``single`` (bool), ``n_peaks`` [count] and ``contrast``
        [dimensionless], plus whatever else the classifier recorded.

    Raises
    ------
    ValueError
        Propagated from
        :func:`~validation.pylle_crosscheck.is_single_soliton` for an empty
        field.

    Notes
    -----
    The classifier is a THRESHOLD test -- contrast at or above 5, exactly one
    circular connected component above the half-way level -- so near an
    existence boundary it can flip on a small difference in a genuinely
    marginal state. Recording what it saw is what makes such a flip visible in
    the bisection trace instead of invisible inside a verdict.

    No ``Examples`` section: it delegates to the v1 classifier, whose behaviour
    is documented and exercised there.
    """
    from validation.pylle_crosscheck import is_single_soliton

    ok, npk, contrast = is_single_soliton(field, t_r, kappa, gamma)
    return {"single": bool(ok), "n_peaks": int(npk), "contrast": float(contrast),
            "peak_power_w": float(np.max(np.abs(field) ** 2) / t_r)}


def initial_brackets(survival, scan_values):
    """Find the coarse-grid brackets straddling each existence edge.

    Parameters
    ----------
    survival : sequence of bool
        Whether a soliton survived at each scanned detuning, in scan order.
    scan_values : sequence of float
        The scanned detunings [rad/s, or units of kappa -- whatever the caller
        works in], same length and order.

    Returns
    -------
    lower : tuple of float or None
        ``(dead, alive)`` straddling the lower edge, or ``None`` when the scan
        does not straddle it.
    upper : tuple of float or None
        ``(alive, dead)`` straddling the upper edge, or ``None`` likewise.

    Raises
    ------
    IndexError
        If ``scan_values`` is shorter than ``survival``.

    Notes
    -----
    A pure function of the survival vector and the grid -- no solver, no state
    -- which is what lets both this module and
    :mod:`validation.existence_convergence` share it and so be certain they
    bisect identically.

    Returning ``None`` rather than clamping to the grid edge is deliberate: a
    scan that never died has not located an edge, and reporting its endpoint as
    one would invent a measurement.

    Examples
    --------
    >>> survival = [False, False, True, True, True, False]
    >>> values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> initial_brackets(survival, values)
    ((2.0, 3.0), (5.0, 6.0))

    A scan with no survivors, or one that never dies, brackets nothing:

    >>> initial_brackets([False] * 6, values)
    (None, None)
    >>> initial_brackets([True] * 6, values)
    (None, None)
    """
    idx = [i for i, s in enumerate(survival) if s]
    if not idx:
        return None, None
    i0, i1 = idx[0], idx[-1]
    low = (scan_values[i0 - 1], scan_values[i0]) if i0 > 0 else None
    high = (scan_values[i1], scan_values[i1 + 1]) if i1 + 1 < len(scan_values) else None
    return low, high


def update_bracket(bracket, midpoint: float, alive: bool, edge: str):
    """One bisection step. ``edge`` is ``"lower"``/``"lo"`` or ``"upper"``/``"hi"``.

    The lower bracket is ordered ``(dead, alive)`` and the upper ``(alive, dead)``,
    so a surviving midpoint tightens the *alive* end in both cases -- which is
    opposite ends of the tuple. Keeping that asymmetry in ONE function is why
    this is shared rather than reimplemented per caller.

    Parameters
    ----------
    bracket : tuple of float or None
        The current bracket, or ``None`` (returned unchanged).
    midpoint : float
        The detuning just tested, in the same units as the bracket.
    alive : bool
        Whether a soliton survived there.
    edge : {'lower', 'lo', 'upper', 'hi'}
        Which edge is being bisected.

    Returns
    -------
    tuple of float or None
        The tightened bracket, or ``None`` if the input was ``None``.

    Raises
    ------
    ValueError
        If ``edge`` is not one of the four accepted spellings. Guessing would
        silently bisect the wrong end.

    Examples
    --------
    The lower bracket is ``(dead, alive)``, so a surviving midpoint tightens
    its RIGHT end:

    >>> update_bracket((2.0, 3.0), 2.5, alive=True, edge="lower")
    (2.0, 2.5)
    >>> update_bracket((2.0, 3.0), 2.5, alive=False, edge="lower")
    (2.5, 3.0)

    The upper bracket is ``(alive, dead)``, so the same survival tightens its
    LEFT end:

    >>> update_bracket((5.0, 6.0), 5.5, alive=True, edge="upper")
    (5.5, 6.0)
    >>> update_bracket((5.0, 6.0), 5.5, alive=False, edge="upper")
    (5.0, 5.5)

    >>> update_bracket((1.0, 2.0), 1.5, True, "sideways")
    Traceback (most recent call last):
        ...
    ValueError: edge must be lower/upper, got 'sideways'
    """
    if bracket is None:
        return None
    if edge in ("lower", "lo"):
        return (bracket[0], midpoint) if alive else (midpoint, bracket[1])
    if edge in ("upper", "hi"):
        return (midpoint, bracket[1]) if alive else (bracket[0], midpoint)
    raise ValueError(f"edge must be lower/upper, got {edge!r}")


def bracket_midpoint(bracket):
    """Return the midpoint of a bracket, or ``None`` for a missing one.

    Parameters
    ----------
    bracket : sequence of float or None
        A two-element bracket, in whatever units the caller works in.

    Returns
    -------
    float or None
        ``0.5*(lo + hi)``, or ``None`` when ``bracket`` is ``None``.

    Raises
    ------
    IndexError
        If ``bracket`` has fewer than two elements.

    Notes
    -----
    The midpoint is the reported EDGE ESTIMATE, and
    :func:`bracket_halfwidth` is its uncertainty. Reporting the tightest
    surviving point instead would be a biased estimator -- see criterion G7.

    Examples
    --------
    >>> bracket_midpoint((2.0, 3.0))
    2.5
    >>> bracket_midpoint(None) is None
    True
    """
    return None if bracket is None else 0.5 * (bracket[0] + bracket[1])


def bracket_halfwidth(bracket):
    """Return the half-width of a bracket, or ``None`` for a missing one.

    Parameters
    ----------
    bracket : sequence of float or None
        A two-element bracket, in whatever units the caller works in.

    Returns
    -------
    float or None
        ``0.5*|hi - lo|``, or ``None`` when ``bracket`` is ``None``.

    Raises
    ------
    IndexError
        If ``bracket`` has fewer than two elements.

    Notes
    -----
    This is the resolution the bisection actually achieved, and it floors every
    uncertainty claim about the edge: no analysis downstream may report the
    edge more precisely than this.

    Examples
    --------
    >>> bracket_halfwidth((2.0, 3.0))
    0.5
    >>> bracket_halfwidth(None) is None
    True
    """
    return None if bracket is None else 0.5 * abs(bracket[1] - bracket[0])


def _brackets_overlap(a, b) -> bool:
    """Do two closed intervals share any point? (Figure labelling only; the
    verdict itself is computed by ``criteria.evaluate_existence_edges``.)"""
    if not a or not b:
        return False
    lo_a, hi_a = sorted(float(x) for x in a)
    lo_b, hi_b = sorted(float(x) for x in b)
    return lo_a <= hi_b and lo_b <= hi_a


def dw_phase_matching_roots(mu: np.ndarray, d_int_nat: np.ndarray,
                            dw_final: float) -> list:
    """Find the modes where a dispersive wave is phase-matched.

    Parameters
    ----------
    mu : numpy.ndarray
        Mode numbers [dimensionless], mu-centred.
    d_int_nat : numpy.ndarray
        ``D_int(mu)`` [rad/s], same shape and ordering.
    dw_final : float
        Detuning at the operating point [rad/s].

    Returns
    -------
    list of int
        Mode numbers where ``D_int(mu) - delta_omega`` changes sign.

    Raises
    ------
    IndexError
        If ``mu`` and ``d_int_nat`` have different lengths.

    Notes
    -----
    ``D_int(mu) = delta_omega`` is the phase-matching condition for a
    dispersive wave. Computing it from the dispersion ALONE, before looking at
    any spectrum, is what makes a bump found there a confirmed prediction
    rather than a feature noticed after the fact and explained afterwards.

    Examples
    --------
    A parabolic ``D_int`` crosses a positive detuning at two symmetric pairs of
    adjacent modes:

    >>> import numpy as np
    >>> mu = np.arange(-100, 101)
    >>> dw_phase_matching_roots(mu, 0.5 * mu ** 2, 200.0)
    [-21, -20, 19, 20]
    """
    s = np.sign(d_int_nat - dw_final)
    return [int(mu[i]) for i in np.flatnonzero(np.diff(s) != 0)]


# ==========================================================================
# figure
# ==========================================================================
def make_figure(path: Path, *, mu, dbc_ours_fine, dbc_pylle_fine, bands,
                d_int_nat, t_r, ladders, containment, existence, mu_half,
                spec_contain, ours_label, pylle_label) -> str:
    """Draw the four-panel comparison figure in-run and return its hash.

    Parameters
    ----------
    path : pathlib.Path
        Destination image.
    mu : numpy.ndarray
        Mode numbers [dimensionless]. Keyword-only.
    dbc_ours_fine, dbc_pylle_fine : numpy.ndarray
        Per-mode spectra [dBc] at the finest level of each ladder.
        Keyword-only.
    bands : list of dict
        The band-residual table from :func:`band_residual_table`. Keyword-only.
    d_int_nat : numpy.ndarray
        ``D_int(mu)`` [rad/s], mu-centred. Keyword-only.
    t_r : float
        Round-trip time [s]. Keyword-only.
    ladders : dict
        Both refinement ladders, for the convergence panel. Keyword-only.
    containment : dict
        Per-observable containment verdicts from :func:`containment_verdict`.
        Keyword-only.
    existence : dict
        The existence-edge brackets [units of kappa]. Keyword-only.
    mu_half : int
        Half-width of the mode grid [mode number]. Keyword-only.
    spec_contain : dict
        The result of :func:`spectral_containment`. Keyword-only.
    ours_label, pylle_label : str
        Legend labels naming each code and its refinement level. Keyword-only.

    Returns
    -------
    str
        ``sha256`` of the written image, recorded in the report JSON.

    Raises
    ------
    OSError
        If ``path`` cannot be written.
    ImportError
        If matplotlib is unavailable.

    Notes
    -----
    Generated IN-RUN and hashed. The v1 figure was regenerated after the fact
    from the saved archive with edited rendering code, which means the figure a
    reader saw was not provably the figure the numbers came from. This one is
    written by the same process that produced them, and its hash goes into the
    JSON, so the two cannot drift apart unnoticed.

    Uses the ``Agg`` backend explicitly so the figure renders identically
    headless.

    No ``Examples`` section: it needs both completed ladders and writes a file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(12, 17))

    # ---- (a) spectra, finest level of each ladder --------------------------
    ax[0].plot(mu, dbc_ours_fine, lw=0.7, label=f"ours ({ours_label})")
    ax[0].plot(mu, dbc_pylle_fine, lw=0.7, ls="--", label=f"pyLLE ({pylle_label})")
    ax[0].axhline(-60, color="grey", lw=0.6, ls=":", label="-60 dBc")
    for x in (mu.min(), mu.max()):
        ax[0].axvline(x, color="k", lw=1.0, alpha=0.6)
    ax[0].set_ylim(-140, 5)
    ax[0].set_xlabel("mode index mu  (ours convention; pyLLE mirrored mu -> -mu)")
    ax[0].set_ylabel("power (dBc)")
    edge = max(spec_contain["mu_minus_half_dbc"], spec_contain["mu_plus_half_dbc"])
    ax[0].set_title(
        f"(a) Spectra at each ladder's finest level. Grid edge at {edge:.1f} dBc"
        + ("  --  COMB NOT SPECTRALLY CONTAINED" if not spec_contain["contained"] else ""),
        fontsize=10, loc="left")
    ax[0].legend(loc="upper right", fontsize=9)

    # ---- (b) per-band residual with the phase boundaries -------------------
    centres = [0.5 * (b["mu_band"][0] + b["mu_band"][1]) for b in bands]
    ax[1].bar(range(len(bands)), [b["median_ours_minus_pylle_db"] for b in bands],
              color="crimson", alpha=0.75)
    ax[1].set_xticks(range(len(bands)))
    ax[1].set_xticklabels([f"{b['mu_band'][0]}-{b['mu_band'][1]}" for b in bands],
                          rotation=45, fontsize=8)
    ax[1].axhline(0, color="k", lw=0.6)
    # |D_int|*t_r = 1 and = pi boundaries, located on the band axis
    phase = np.abs(d_int_nat) * t_r
    for level, style, lab in ((1.0, "--", r"|D_int|*t_r = 1"),
                             (math.pi, ":", r"|D_int|*t_r = pi")):
        idx = np.flatnonzero(phase >= level)
        if idx.size:
            mu_cross = float(np.min(np.abs(mu[idx])))
            pos = float(np.interp(mu_cross, centres, np.arange(len(bands))))
            ax[1].axvline(pos, color="navy", ls=style, lw=1.2,
                          label=f"{lab}  (|mu| = {mu_cross:.0f})")
    ax[1].set_ylabel("median (ours - pyLLE), dB")
    ax[1].set_xlabel("|mu| band")
    ax[1].set_title("(b) Per-band spectral residual. A gap that tracked the "
                    "under-resolved phase would switch sign at these lines",
                    fontsize=10, loc="left")
    ax[1].legend(fontsize=8)

    # Lay the four host panels out BEFORE the nested sub-axes of (c) and (d) are
    # created: tight_layout moves the parent gridspec, and sub-axes positioned
    # against its pre-layout geometry would land in the wrong place.
    for a in (ax[0], ax[1]):
        a.grid(alpha=0.3)
    fig.suptitle("soliton-control vs pyLLE (independent implementation reference)\n"
                 "convention audit + convergence-envelope containment", fontsize=12)
    fig.tight_layout(rect=(0, 0.015, 1, 0.975))

    # ---- (c) convergence, one axis per GATED observable --------------------
    # The panel title goes on the figure, not on the host axes: a title drawn on
    # ax[2] lands underneath the sub-axes' own titles and both become unreadable.
    ax[2].axis("off")
    pos = ax[2].get_position()
    # The title sits INSIDE the host panel's band, and the sub-axes are pushed
    # into the lower rows of a nested grid to make room. Placing it above the
    # host panel instead would collide with the rotated tick labels of (b).
    fig.text(pos.x0, pos.y1 - 0.002,
             "(c) Refinement ladders per GATED observable, with each code's "
             "extrapolated limit and uncertainty band",
             fontsize=10, ha="left", va="top")
    keys = [k for k in containment]
    n = max(1, len(keys))
    sub = ax[2].get_subplotspec().subgridspec(
        2, n, wspace=0.55, height_ratios=[0.30, 1.0], hspace=0.0)
    for j, key in enumerate(keys):
        a = fig.add_subplot(sub[1, j])
        c = containment[key]
        for side, marker, colour in (("ours", "o", "tab:blue"),
                                     ("pylle", "s", "tab:orange")):
            lad = ladders[side]
            hs = [lv["h"] for lv in lad if lv.get("obs")]
            vs = [lv["obs"].get(key) for lv in lad if lv.get("obs")]
            hs = [h for h, v in zip(hs, vs) if v is not None]
            vs = [v for v in vs if v is not None]
            if hs:
                a.plot(hs, vs, marker=marker, ms=4, lw=0.9, color=colour, label=side)
            lim, u = c.get(f"{side}_limit"), c.get(f"{side}_U")
            if lim is not None:
                a.axhline(lim, color=colour, lw=0.8, ls="--")
                if u is not None:
                    half = abs(u * lim) if c["kind"] == "relative" else abs(u * lim)
                    a.axhspan(lim - half, lim + half, color=colour, alpha=0.13)
        a.set_xscale("log")
        a.set_xlabel("h")
        a.set_title(f"{key}\n{c['verdict']}", fontsize=8)
        a.tick_params(labelsize=7)
        if j == 0:
            a.legend(fontsize=7)
    # ---- (d) existence brackets as INTERVALS -------------------------------
    # One zoomed axis per edge. On a shared 5-38 kappa axis a 0.09-wide bracket
    # is a hairline, and the whole point of the panel -- that these are resolved
    # intervals which either overlap or do not -- becomes invisible.
    ax[3].axis("off")
    pos3 = ax[3].get_position()
    fig.text(pos3.x0, pos3.y1 - 0.002,
             "(d) Single-soliton existence edges as resolved INTERVALS, not "
             "points. Overlap is the test; a bracket end is not an edge",
             fontsize=10, ha="left", va="top")
    sub3 = ax[3].get_subplotspec().subgridspec(
        2, 2, wspace=0.22, height_ratios=[0.16, 1.0], hspace=0.0)
    for j, edge in enumerate(("lower", "upper")):
        a = fig.add_subplot(sub3[1, j])
        spans = []
        for i, side in enumerate(("ours", "pylle")):
            br = existence.get(f"{edge}_bracket_{side}")
            if not br:
                continue
            lo, hi = sorted(float(x) for x in br)
            spans += [lo, hi]
            colour = "tab:blue" if side == "ours" else "tab:orange"
            a.plot([lo, hi], [i, i], lw=14, alpha=0.55, color=colour,
                   solid_capstyle="butt")
            a.plot([0.5 * (lo + hi)], [i], marker="|", ms=18, color="k")
            a.annotate(f"[{lo:.4f}, {hi:.4f}]", xy=(0.5 * (lo + hi), i),
                       xytext=(0, 13), textcoords="offset points",
                       ha="center", fontsize=8)
        a.set_yticks([0, 1])
        a.set_yticklabels(["ours", "pyLLE"], fontsize=9)
        a.set_ylim(-0.7, 1.7)
        if spans:
            pad = 0.25 * (max(spans) - min(spans)) + 1e-3
            a.set_xlim(min(spans) - pad, max(spans) + pad)
        overlap = "OVERLAP" if _brackets_overlap(
            existence.get(f"{edge}_bracket_ours"),
            existence.get(f"{edge}_bracket_pylle")) else "DISJOINT"
        a.set_title(f"{edge} edge  --  {overlap}", fontsize=9)
        a.set_xlabel("delta_omega / kappa")
        a.grid(alpha=0.3, axis="x")

    # Suppress the PNG creation-time chunk so the file is byte-deterministic and
    # its sha256 can be part of the determinism digest (A7).
    fig.savefig(path, dpi=130, metadata={"Software": "validation.pylle_crosscheck_v2",
                                         "Creation Time": None})
    plt.close(fig)
    return _sha_file(path)


# ==========================================================================
# orchestration
# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    """Run the v2 pyLLE cross-check from the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments; ``None`` (default) reads ``sys.argv[1:]``.
        ``--pylle-python`` and ``--julia-bin`` locate the pyLLE
        environment and default to the ``PYLLE_PYTHON`` and ``JULIA_BIN``
        environment variables.

    Returns
    -------
    int
        The process exit status: 0 when the assembled report's ``overall`` is
        ``PASS``, non-zero otherwise.

    Raises
    ------
    SystemExit
        From ``argparse`` on a malformed command line or ``--help``.
    CriteriaError
        From :mod:`validation.criteria` if a tolerance changed between
        derivation and evaluation, or the report fails its own schema.
    FileNotFoundError
        If the pyLLE interpreter, the Julia binary or the dispersion CSV
        cannot be found.

    Notes
    -----
    Tolerances are DERIVED first, from the two measured uncertainty studies,
    and fingerprinted before any comparison value is read; the report cannot be
    built if one moved afterwards. pyLLE runs out of process under its own
    interpreter, which pins ``numpy < 2`` and needs a Julia toolchain.

    Writes ``validation/results/pylle_crosscheck_v2.json``, its fields archive
    and the in-run figure. This cross-check is FROZEN: re-running it is one of
    the five named triggers in ``docs/VALIDATION_STATUS.md`` section 5, not
    routine maintenance.

    No ``Examples`` section: it requires a provisioned pyLLE environment.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pylle-python", default=os.environ.get("PYLLE_PYTHON"))
    ap.add_argument("--julia-bin", default=os.environ.get("JULIA_BIN"))
    ap.add_argument("--vendored-kernel", default=str(THIRD_PARTY / "ComputeLLE.jl"))
    ap.add_argument("--mu-half", type=int, default=MU_HALF)
    ap.add_argument("--roundtrips", type=int, default=N_ROUNDTRIPS_MAIN)
    ap.add_argument("--scan-roundtrips", type=int, default=N_ROUNDTRIPS_SCAN)
    ap.add_argument("--dw-start", type=float, default=DW_START_KAPPA)
    ap.add_argument("--dw-end", type=float, default=DW_END_KAPPA)
    ap.add_argument("--scan", default=SCAN_KAPPA)
    ap.add_argument("--bisect-iters", type=int, default=BISECT_ITERS)
    ap.add_argument("--ours-levels", default=OURS_LEVELS)
    ap.add_argument("--pylle-levels", default=PYLLE_LEVELS)
    ap.add_argument("--num-probe", type=int, default=NUM_PROBE)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--skip-vendor-gate", action="store_true",
                    help="skip the vendored-kernel integrity assertion (never for a "
                         "committed run; recorded in provenance when used)")
    args, _unknown = ap.parse_known_args(raw_argv)

    if not args.pylle_python or not args.julia_bin:
        print("ERROR: --pylle-python and --julia-bin (or PYLLE_PYTHON / JULIA_BIN) are "
              "required. pyLLE needs numpy<2 and a Julia backend and cannot share this "
              "repo's environment.", file=sys.stderr)
        return 2

    import tempfile
    from validation import criteria as CR
    from validation.convergence_lle import (CONVERGED_OBSERVABLES, OBSERVABLE_CONSTANTS,
                                            OBSERVABLE_DEFINITIONS, observables_v2,
                                            richardson, spectrum_dbc)
    from validation.pylle_crosscheck import (assert_round_trip, build_pylle_dispfile,
                                             derived_config, load_device,
                                             local_d2_from_dint, ours_to_pylle)

    t_start = time.time()
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="xcheck2_"))
    work.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- vendored kernel integrity (Prompt D) ------------------
    sys.path.insert(0, str(THIRD_PARTY))
    integrity = {"skipped": bool(args.skip_vendor_gate)}
    if not args.skip_vendor_gate:
        from verify_vendor import assert_vendor_integrity
        integrity.update(assert_vendor_integrity(check_installed=False))
        print(f"vendored kernel integrity OK: {integrity['vendored_sha256'][:16]}")

    # ---------------- PART 1: translation and the round trip ----------------
    dev = load_device()
    csv = np.loadtxt(DISP_CSV, delimiter=",")
    mu_csv = csv[:, 0].astype(np.int64)
    f_pmp = float(csv[int(np.where(mu_csv == 0)[0][0]), 1])       # Finding 8
    pump_offset_fsr = (f_pmp - C0 / dev["pump_wavelength_m"]) / dev["fsr_hz"]
    fwd = ours_to_pylle(dev, f_pmp_hz=f_pmp)
    rt = assert_round_trip(dev, f_pmp_hz=f_pmp)

    kappa = dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]
    n_modes = 2 * args.mu_half + 1
    mu = np.arange(-args.mu_half, args.mu_half + 1)

    from simulator.lle_solver import load_dint_grid
    ours_grid = load_dint_grid(n_modes, csv_path=DISP_CSV)
    d_int_ours_natural = np.fft.fftshift(np.asarray(ours_grid.grid, dtype=float))
    d2_local = local_d2_from_dint(mu, d_int_ours_natural)
    d1_csv = float(ours_grid.d1)                                   # Finding 9
    fwd["D1_manual_rad_per_s"] = d1_csv
    fsr_used = d1_csv / (2.0 * math.pi)
    t_r = 1.0 / fsr_used
    fwd["t_r_s"] = t_r

    dw_start, dw_end = args.dw_start * kappa, args.dw_end * kappa
    scan_dws = [float(x) * kappa for x in args.scan.split(",")]

    from validation.pylle_crosscheck import soliton_seed
    seed_main = soliton_seed(n_modes, dw_start, kappa, dev["kappa_c_rad_per_s"],
                             dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)
    seed_main_sha = _sha_arr(np.asarray(seed_main, dtype=np.complex128))
    # The SAME forward map the worker applies (Findings 2 and 3). H7 compares
    # these two hashes, so it proves both codes hold the identical array.
    seed_pylle_sha = _sha_arr(np.conj(np.asarray(seed_main, dtype=np.complex128))
                              / math.sqrt(t_r))

    dispfile = work / "pylle_dispfile_mirrored.csv"                 # Finding 2b
    m0_abs = int(round(dev["n0"] * fwd["L_cav_m"] / dev["pump_wavelength_m"]))
    build_pylle_dispfile(dispfile, mu, d_int_ours_natural, f_pmp, d1_csv, m0_abs)

    print("=" * 96)
    print("CROSS-CODE CHECK v2  --  convention audit + convergence-envelope containment")
    print("=" * 96)
    print(f"pump pinned to CSV mu=0 resonance f = {f_pmp:.6e} Hz "
          f"({pump_offset_fsr:+.2f} FSR from nominal c/lambda)   [Finding 8]")
    print(f"D1 (CSV fit) = {d1_csv:.9e} rad/s -> FSR {fsr_used / 1e9:.6f} GHz, "
          f"t_r = {t_r:.9e} s   [Finding 9]")
    print(f"round-trip worst rel diff = {rt['_worst_rel_diff']:.3e}   [H1]")
    print(f"grid {n_modes} modes, {args.roundtrips} round trips, "
          f"delta_omega {args.dw_start:g}k -> {args.dw_end:g}k")

    # ---------------- pyLLE plumbing ---------------------------------------
    jl_path = str(Path(args.vendored_kernel).resolve())
    spec_base = {
        "mu_half": args.mu_half, "t_r_s": t_r, "num_probe": args.num_probe,
        "mu_fit": [int(mu.min()), int(mu.max())],
        "res": {"R": fwd["R_m"], "Qi": fwd["Qi"], "Qc": fwd["Qc"],
                "gamma": fwd["gamma_nlse_per_w_per_m"], "dispfile": str(dispfile)},
        "Pin_w": fwd["Pin_w"], "f_pmp_hz": f_pmp,
        "phi_pmp_rad": fwd["phi_pmp_rad"],
        "D1_manual_rad_per_s": d1_csv,
    }
    env = dict(os.environ, MPLBACKEND="Agg")
    env.setdefault("JULIA_DEPOT_PATH", os.environ.get("JULIA_DEPOT_PATH", ""))

    def call_pylle(run_list: list[dict], tag: str) -> dict:
        spec_local = dict(spec_base, runs=run_list)
        sp, op = work / f"spec_{tag}.json", work / f"out_{tag}.json"
        sp.write_text(json.dumps(spec_local))
        print(f"[pyLLE:{tag}] {len(run_list)} run(s)", flush=True)
        proc = subprocess.run(
            [args.pylle_python, str(Path(__file__).resolve()), "--worker",
             "--spec", str(sp), "--out", str(op), "--julia-bin", args.julia_bin],
            env=env, capture_output=True, text=True)
        if proc.returncode != 0 or "WORKER_OK" not in (proc.stdout or ""):
            print("\n".join((proc.stdout or "").splitlines()[-30:]))
            print((proc.stderr or "")[-4000:], file=sys.stderr)
            raise RuntimeError(f"pyLLE worker failed (returncode {proc.returncode})")
        print((proc.stdout or "").rstrip())
        got = json.loads(op.read_text())
        if got.get("worker_failure"):
            raise RuntimeError(f"pyLLE worker reported: {got['worker_failure']}")
        return got

    def prun(name, tscan, dw_a, dw_b, dt, tol, maxiter, seed):
        return {"name": name, "Tscan": float(tscan), "dw_start": float(dw_a),
                "dw_end": float(dw_b), "dt": float(dt), "tol": float(tol),
                "maxiter": int(maxiter), "jl_path": jl_path,
                "seed_real": np.asarray(seed).real.tolist(),
                "seed_imag": np.asarray(seed).imag.tolist()}

    def to_ours(rec: dict) -> np.ndarray:
        """pyLLE field -> our convention. Findings 2 and 3, applied once, here."""
        a = (np.asarray(rec["field_real"], float) + 1j * np.asarray(rec["field_imag"], float))
        return np.conj(a) * math.sqrt(t_r)

    # ---------------- pyLLE main ladders -----------------------------------
    # Two ladders. The dt ladder at the SHIPPED Picard tolerance is what R4
    # specifies; at tol=1e-2 the iteration runs exactly 2 passes at every step
    # regardless of dt, so refining dt there refines the linear step while
    # leaving the nonlinear solve inexact. The tight ladder repeats it with
    # tol=1e-8/maxiter=60. Running both is what makes "step size" and "Picard
    # tolerance" separable rather than confounded.
    dts = [float(x) for x in args.pylle_levels.split(",")]
    pl_runs = []
    for dt in dts:
        pl_runs.append(prun(f"dt{dt:g}_shipped", args.roundtrips, dw_start, dw_end,
                            dt, PICARD_SHIPPED[0], PICARD_SHIPPED[1], seed_main))
    for dt in dts:
        pl_runs.append(prun(f"dt{dt:g}_tight", args.roundtrips, dw_start, dw_end,
                            dt, PICARD_TIGHT[0], PICARD_TIGHT[1], seed_main))
    pl_main = call_pylle(pl_runs, "ladders")

    # ---------------- dispersion: pyLLE's own refit, on BOTH sides ----------
    d_int_pylle_natural = np.asarray(pl_main["d_int_rad_per_s"], dtype=float)
    d_int_unmirrored = d_int_pylle_natural[::-1].copy()
    scale = float(np.max(np.abs(d_int_ours_natural)))
    refit_rel = float(np.max(np.abs(d_int_unmirrored - d_int_ours_natural)) / scale)
    d_int_fftorder = np.fft.ifftshift(d_int_unmirrored)
    max_dint_tr = float(np.max(np.abs(d_int_pylle_natural)) * t_r)
    print(f"[dispersion] pyLLE refit (un-mirrored) vs our loader: {refit_rel:.3e}  [H2]")

    cfg = derived_config(fsr_used, work)
    ramp = np.linspace(dw_start, dw_end, args.roundtrips)

    # ---------------- our ladder -------------------------------------------
    ours_levels = [int(x) for x in args.ours_levels.split(",")]
    ladder_ours, eff_rel_worst = [], 0.0
    for nsub in ours_levels:
        r = run_ours(ramp, seed_main, d_int_fftorder, dev, cfg, n_substeps=nsub)
        obs = observables_v2(r["field"], t_r, mu)
        eff_rel_worst = max(eff_rel_worst, r["delta_omega_eff_max_rel"])
        ladder_ours.append({
            "level": f"n{nsub}", "n_substeps": nsub, "h": 1.0 / nsub,
            "wall_s": r["wall_s"], "obs": obs,
            "delta_omega_eff_max_rel": r["delta_omega_eff_max_rel"],
            "_field": r["field"],
        })
        print(f"[ours n={nsub:<3d}] {r['wall_s']:6.1f}s  "
              f"U_mean={_fmt(obs['U_mean_w'], '.6f')} W  "
              f"P_peak={_fmt(obs['P_peak_w'], '.4f')} W  "
              f"mu_DW={_fmt(obs['mu_DW'], '.4f')}", flush=True)

    # ---------------- pyLLE ladders -> observables -------------------------
    def pylle_ladder(suffix: str) -> list:
        out = []
        for dt in dts:
            rec = pl_main["runs"][f"dt{dt:g}_{suffix}"]
            field = to_ours(rec)
            out.append({
                "level": f"dt{dt:g}_{suffix}", "dt": dt, "h": dt,
                "tol": (PICARD_SHIPPED if suffix == "shipped" else PICARD_TIGHT)[0],
                "maxiter": (PICARD_SHIPPED if suffix == "shipped" else PICARD_TIGHT)[1],
                "wall_s": rec["wall_s"],
                "mean_picard_iters": rec.get("mean_picard_iters"),
                "max_picard_iters": rec.get("max_picard_iters"),
                "picard_failure_count": rec.get("picard_failure_count"),
                "detuning_pylle_final": rec["detuning_pylle_final"],
                "seed_sha256_pylle_convention": rec["seed_sha256_pylle_convention"],
                "pump_freq_hz_located_by_pylle": rec["pump_freq_hz_located_by_pylle"],
                "pump_mu_located_by_pylle": rec["pump_mu_located_by_pylle"],
                "obs": observables_v2(field, t_r, mu),
                "_field": field,
            })
            o = out[-1]
            print(f"[pyLLE {o['level']:<16}] {o['wall_s']:6.1f}s  "
                  f"U_mean={_fmt(o['obs']['U_mean_w'], '.6f')} W  "
                  f"P_peak={_fmt(o['obs']['P_peak_w'], '.4f')} W  "
                  f"mu_DW={_fmt(o['obs']['mu_DW'], '.4f')}  "
                  f"picard={o['mean_picard_iters']}", flush=True)
        return out

    ladder_pylle_shipped = pylle_ladder("shipped")
    ladder_pylle_tight = pylle_ladder("tight")

    # R4's named separate level: dt = 1 with the tight Picard pair.
    dt1_tight = next(lv for lv in ladder_pylle_tight if lv["dt"] == 1.0)
    dt1_shipped = next(lv for lv in ladder_pylle_shipped if lv["dt"] == 1.0)

    # ---------------- H5: are the two codes at the SAME final detuning? -----
    # Finding 2: pyLLE's detuning is the negative of ours. Patch 0003 makes the
    # last probe the true end-of-run state, so this compares like with like.
    dw_p_final = float(dt1_shipped["detuning_pylle_final"])
    detuning_match_rel = abs(dw_end - (-dw_p_final)) / abs(dw_end)
    print(f"[detuning] ours {dw_end:.6e}  pyLLE {-dw_p_final:.6e} (sign-flipped)  "
          f"rel {detuning_match_rel:.3e}   [H5]")

    # ---------------- Richardson on both ladders ---------------------------
    def ladder_convergence(ladder: list) -> dict:
        out = {}
        for key in CONVERGED_OBSERVABLES:
            vals = [lv["obs"].get(key) for lv in ladder]
            hs = [lv["h"] for lv in ladder]
            order = np.argsort(hs)[::-1]              # coarse -> fine
            out[key] = richardson([vals[i] for i in order], [hs[i] for i in order])
        return out

    conv_ours = ladder_convergence(ladder_ours)
    conv_pylle_shipped = ladder_convergence(ladder_pylle_shipped)
    conv_pylle_tight = ladder_convergence(ladder_pylle_tight)

    # ---------------- R5: envelope containment ------------------------------
    # The headline pairing is ours vs pyLLE's TIGHT ladder. That is not chosen
    # from the results: Prompt A derived every GATED tolerance from the tight
    # ladder's uncertainty (pylle_tag="tight"), and the fingerprint over those
    # tolerances is frozen. The shipped ladder is reported alongside it in full.
    gated_keys = [CR.CRITERIA_BY_ID[cid].conv_key for cid in CR.DERIVED_GATED_IDS]
    absolute_keys = {CR.CRITERIA_BY_ID[cid].conv_key for cid in CR.DERIVED_GATED_IDS
                     if CR.CRITERIA_BY_ID[cid].comparison == "absolute"}
    containment = {}
    containment_shipped = {}
    for key in gated_keys:
        containment[key] = containment_verdict(
            conv_ours.get(key), conv_pylle_tight.get(key),
            absolute=key in absolute_keys, coverage=CR.COVERAGE_FACTOR)
        containment_shipped[key] = containment_verdict(
            conv_ours.get(key), conv_pylle_shipped.get(key),
            absolute=key in absolute_keys, coverage=CR.COVERAGE_FACTOR)

    separated = [k for k, v in containment.items() if v["verdict"] == "SEPARATED"]

    # ---------------- R8: existence scan ------------------------------------
    print(f"\n[existence] {len(scan_dws)} detunings x {args.scan_roundtrips} round trips, "
          f"both codes", flush=True)

    def classify(field: np.ndarray) -> dict:
        return classify_field(field, t_r, kappa, dev["gamma_LLE_per_J_per_s"])

    def seed_at(dw: float) -> np.ndarray:
        return soliton_seed(n_modes, dw, kappa, dev["kappa_c_rad_per_s"],
                            dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)

    def ours_hold(dw: float, tscan: int) -> np.ndarray:
        return run_ours(np.full(int(tscan), dw), seed_at(dw), d_int_fftorder,
                        dev, cfg)["field"]

    # Stationarity is measured by TRUNCATING the hold, which at constant
    # detuning is exactly the state at that round trip: the trajectory is
    # deterministic and the detuning array is constant, so run(N-k) is the state
    # at round trip N-k of run(N). This keeps the protocol identical on both
    # sides rather than reading our snapshots against pyLLE's probe cadence.
    scan_points = []
    pl_scan_runs = []
    for dw in scan_dws:
        for off in (0,) + STATIONARITY_OFFSETS:
            pl_scan_runs.append(prun(f"scan{dw / kappa:.3f}_m{off}",
                                     args.scan_roundtrips - off, dw, dw, 1.0,
                                     PICARD_SHIPPED[0], PICARD_SHIPPED[1], seed_at(dw)))
    pl_scan = call_pylle(pl_scan_runs, "scan")

    for dw in scan_dws:
        k = dw / kappa
        rec = {"delta_omega_over_kappa": k,
               "kerr_phase_per_roundtrip_rad": 2.0 * dw * t_r,
               "roundtrips": args.scan_roundtrips}
        for side in ("ours", "pylle"):
            evals = {}
            for off in (0,) + STATIONARITY_OFFSETS:
                if side == "ours":
                    f = ours_hold(dw, args.scan_roundtrips - off)
                else:
                    f = to_ours(pl_scan["runs"][f"scan{k:.3f}_m{off}"])
                evals[f"minus_{off}"] = classify(f)
            base = evals["minus_0"]
            peaks = [e["peak_power_w"] for e in evals.values()]
            labels = {e["single"] for e in evals.values()}
            spread = (max(peaks) - min(peaks)) / max(peaks) if max(peaks) > 0 else 0.0
            rec[side] = {
                "single": base["single"], "n_peaks": base["n_peaks"],
                "contrast": base["contrast"], "peak_power_w": base["peak_power_w"],
                "stationarity": {
                    "evaluations": evals,
                    "peak_power_spread_rel": spread,
                    "label_changes": len(labels) > 1,
                    "flag": ("NON_STATIONARY" if (spread > STATIONARITY_TOL
                                                  or len(labels) > 1) else "STATIONARY"),
                },
            }
        scan_points.append(rec)
        print(f"   dw/k={k:6.3f}  ours single={rec['ours']['single']} "
              f"peaks={rec['ours']['n_peaks']} contrast={rec['ours']['contrast']:9.1f} "
              f"{rec['ours']['stationarity']['flag']:<14} | "
              f"pylle single={rec['pylle']['single']} peaks={rec['pylle']['n_peaks']} "
              f"contrast={rec['pylle']['contrast']:9.1f} "
              f"{rec['pylle']['stationarity']['flag']}", flush=True)

    surv_ours = [p["ours"]["single"] for p in scan_points]
    surv_pylle = [p["pylle"]["single"] for p in scan_points]

    br_lo_o, br_hi_o = initial_brackets(surv_ours, scan_dws)
    br_lo_p, br_hi_p = initial_brackets(surv_pylle, scan_dws)
    bisect_trace = []
    for it in range(args.bisect_iters):
        mids_p = {}
        batch = []
        for key, br in (("lo", br_lo_p), ("hi", br_hi_p)):
            if br is None:
                continue
            m = 0.5 * (br[0] + br[1])
            mids_p[key] = m
            batch.append(prun(f"bis{it}_{key}", args.scan_roundtrips, m, m, 1.0,
                              PICARD_SHIPPED[0], PICARD_SHIPPED[1], seed_at(m)))
        step = {"iteration": it, "ours": {}, "pylle": {}}
        if batch:
            got = call_pylle(batch, f"bisect{it}")
            for key, m in mids_p.items():
                c = classify(to_ours(got["runs"][f"bis{it}_{key}"]))
                step["pylle"][key] = {"midpoint_over_kappa": m / kappa, **c}
                if key == "lo":
                    br_lo_p = update_bracket(br_lo_p, m, c["single"], "lo")
                else:
                    br_hi_p = update_bracket(br_hi_p, m, c["single"], "hi")
        for key, br in (("lo", br_lo_o), ("hi", br_hi_o)):
            if br is None:
                continue
            m = bracket_midpoint(br)
            c = classify(ours_hold(m, args.scan_roundtrips))
            step["ours"][key] = {"midpoint_over_kappa": m / kappa, **c}
            if key == "lo":
                br_lo_o = update_bracket(br_lo_o, m, c["single"], "lo")
            else:
                br_hi_o = update_bracket(br_hi_o, m, c["single"], "hi")
        bisect_trace.append(step)
        print(f"   [bisect {it}] ours lo={_br_str(br_lo_o, kappa)} hi={_br_str(br_hi_o, kappa)}"
              f" | pylle lo={_br_str(br_lo_p, kappa)} hi={_br_str(br_hi_p, kappa)}", flush=True)

    def br_k(br):
        return None if br is None else sorted([br[0] / kappa, br[1] / kappa])

    # ---------------- R8: hold-time sensitivity at each edge ----------------
    # The lower-edge disagreement sits at a Kerr phase where the step-size
    # explanation cannot apply, so settle time is the leading alternative. If
    # the edge moves with hold time, the edge is not resolved to better than
    # that movement, and that movement is its uncertainty.
    print("\n[hold-time] re-classifying each edge at 1x / 2x / 4x the hold", flush=True)
    hold = {}
    hold_batch = []
    for edge, br_o, br_p in (("lower", br_lo_o, br_lo_p), ("upper", br_hi_o, br_hi_p)):
        for side, br in (("ours", br_o), ("pylle", br_p)):
            if br is None:
                continue
            m = 0.5 * (br[0] + br[1])
            for mult in HOLD_TIME_MULTIPLIERS:
                if side == "pylle":
                    hold_batch.append(prun(f"hold_{edge}_{mult}x", args.scan_roundtrips * mult,
                                           m, m, 1.0, PICARD_SHIPPED[0], PICARD_SHIPPED[1],
                                           seed_at(m)))
    pl_hold = call_pylle(hold_batch, "holdtime") if hold_batch else {"runs": {}}

    for edge, br_o, br_p in (("lower", br_lo_o, br_lo_p), ("upper", br_hi_o, br_hi_p)):
        for side, br in (("ours", br_o), ("pylle", br_p)):
            key = f"{edge}_{side}"
            if br is None:
                hold[key] = None
                continue
            m = 0.5 * (br[0] + br[1])
            per = {}
            for mult in HOLD_TIME_MULTIPLIERS:
                f = (ours_hold(m, args.scan_roundtrips * mult) if side == "ours"
                     else to_ours(pl_hold["runs"][f"hold_{edge}_{mult}x"]))
                per[f"{mult}x"] = classify(f)
            labels = {v["single"] for v in per.values()}
            peaks = [v["peak_power_w"] for v in per.values()]
            hold[key] = {
                "midpoint_over_kappa": m / kappa,
                "evaluations": per,
                "classification_changed_with_hold": len(labels) > 1,
                "peak_power_spread_rel": ((max(peaks) - min(peaks)) / max(peaks)
                                          if max(peaks) > 0 else 0.0),
            }
            print(f"   {key:<14} mid={m / kappa:7.4f}k  "
                  + "  ".join(f"{k}:single={v['single']}" for k, v in per.items())
                  + f"   moved={hold[key]['classification_changed_with_hold']}", flush=True)

    # An edge whose classification flips with hold time is not resolved to
    # better than one bisection step; that measured movement becomes its
    # uncertainty rather than being asserted away.
    def edge_u(edge: str, side: str, br) -> float:
        h = hold.get(f"{edge}_{side}")
        if not h or br is None:
            return 0.0
        return abs(br[1] - br[0]) / kappa if h["classification_changed_with_hold"] else 0.0

    existence = {
        "scan_over_kappa": [d / kappa for d in scan_dws],
        "points": scan_points,
        "survival_ours": surv_ours,
        "survival_pylle": surv_pylle,
        "bisect_iters": args.bisect_iters,
        "bisect_trace": bisect_trace,
        "hold_time_sensitivity": hold,
        "hold_time_multipliers": list(HOLD_TIME_MULTIPLIERS),
    }
    for edge, br_o, br_p in (("lower", br_lo_o, br_lo_p), ("upper", br_hi_o, br_hi_p)):
        bo, bp = br_k(br_o), br_k(br_p)
        existence[f"{edge}_bracket_ours"] = bo
        existence[f"{edge}_bracket_pylle"] = bp
        for side, b in (("ours", bo), ("pylle", bp)):
            existence[f"{edge}_midpoint_{side}"] = None if b is None else 0.5 * (b[0] + b[1])
            existence[f"{edge}_halfwidth_{side}"] = None if b is None else 0.5 * (b[1] - b[0])
        existence[f"{edge}_u_ours"] = edge_u(edge, "ours", br_o)
        existence[f"{edge}_u_pylle"] = edge_u(edge, "pylle", br_p)

    # ---------------- diagnostics ------------------------------------------
    fine_ours = ladder_ours[int(np.argmin([lv["h"] for lv in ladder_ours]))]
    fine_pylle = ladder_pylle_tight[int(np.argmin([lv["h"] for lv in ladder_pylle_tight]))]
    coarse_ours = next(lv for lv in ladder_ours if lv["n_substeps"] == 1)

    dbc_o_fine = spectrum_dbc(fine_ours["_field"])
    dbc_p_fine = spectrum_dbc(fine_pylle["_field"])
    dbc_o_n1 = spectrum_dbc(coarse_ours["_field"])
    dbc_p_dt1 = spectrum_dbc(dt1_shipped["_field"])
    dbc_p_dt1_tight = spectrum_dbc(dt1_tight["_field"])

    bands_finest = band_residual_table(dbc_o_fine, dbc_p_fine, mu, d_int_unmirrored, t_r)
    bands_v1_like = band_residual_table(dbc_o_n1, dbc_p_dt1, mu, d_int_unmirrored, t_r)
    bands_dt1_tight = band_residual_table(dbc_o_n1, dbc_p_dt1_tight, mu, d_int_unmirrored, t_r)

    # R7's designated falsification test. The standing hypothesis in
    # docs/PYLLE_STATUS.md is that the wing gap is pyLLE's 1e-2 nonlinear
    # tolerance leaving a broadband floor. If that is right, tightening the
    # tolerance at the SAME dt must move the wings toward ours.
    wing = [b for b in bands_v1_like if b["mu_band"][0] >= 2000]
    wing_t = [b for b in bands_dt1_tight if b["mu_band"][0] >= 2000]
    gap_shipped = (float(np.median([b["median_ours_minus_pylle_db"] for b in wing]))
                   if wing else None)
    gap_tight = (float(np.median([b["median_ours_minus_pylle_db"] for b in wing_t]))
                 if wing_t else None)
    picard_test = {
        "hypothesis": "the wing gap is pyLLE's shipped 1e-2 Picard tolerance leaving "
                      "a broadband numerical floor above the true comb wings",
        "test": "hold dt = 1 and tighten only the Picard pair to 1e-8 / 60; if the "
                "hypothesis holds, the |mu| >= 2000 wing gap must shrink toward 0",
        "wing_median_gap_db_dt1_shipped_tol": gap_shipped,
        "wing_median_gap_db_dt1_tight_tol": gap_tight,
        "change_db": (None if gap_tight is None or gap_shipped is None
                      else gap_tight - gap_shipped),
        "mean_picard_iters_shipped": dt1_shipped["mean_picard_iters"],
        "mean_picard_iters_tight": dt1_tight["mean_picard_iters"],
        "verdict": ("NOT_APPLICABLE (no |mu| >= 2000 band on this grid)"
                    if gap_tight is None or gap_shipped is None
                    else "SUPPORTED" if abs(gap_tight) < 0.5 * abs(gap_shipped)
                    else "NOT_SUPPORTED"),
    }

    spec_contain_ours = spectral_containment(dbc_o_fine, mu)
    spec_contain_pylle = spectral_containment(dbc_p_fine, mu)
    spec_contain = {
        "ours": spec_contain_ours, "pylle": spec_contain_pylle,
        "mu_half": args.mu_half,
        "warning": spec_contain_ours["warning"] or spec_contain_pylle["warning"],
        "blue_dw_phase_match_mu": dw_phase_matching_roots(mu, d_int_unmirrored, dw_end),
    }
    blue_roots = [r for r in spec_contain["blue_dw_phase_match_mu"] if r > 0]
    spec_contain["blue_dw_margin_modes"] = (
        int(args.mu_half - max(blue_roots)) if blue_roots else None)
    spec_contain["blue_dw_resolved"] = bool(
        blue_roots and (args.mu_half - max(blue_roots)) > 150)

    phase = np.abs(d_int_unmirrored) * t_r
    over = phase > math.pi
    diag = {
        "comb_line_count_60dbc": {
            "ours": {k: fine_ours["obs"][k] for k in
                     ("N60", "dN60_ddB", "N60_core", "N60_mid", "N60_edge")},
            "pylle": {k: fine_pylle["obs"][k] for k in
                      ("N60", "dN60_ddB", "N60_core", "N60_mid", "N60_edge")},
            "note": "conditioning dN60/ddB is the number of lines a 1 dB threshold "
                    "shift moves; the core counts are what a physical comb-width "
                    "claim would rest on",
        },
        "integer_3db_span": {"ours": fine_ours["obs"]["S3_int_modes"],
                             "pylle": fine_pylle["obs"]["S3_int_modes"],
                             "quantum_modes": 1,
                             "note": "one mode is the smallest resolvable difference; "
                                     "v1's 0.23% agreement was exactly one quantum"},
        "integer_dw_argmax": {"ours": fine_ours["obs"].get("mu_DW"),
                              "pylle": fine_pylle["obs"].get("mu_DW"),
                              "note": "the continuous centroid is the gated quantity; "
                                      "the rounded index is reported only for continuity "
                                      "with v1"},
        "raw_peak_power": {"ours": fine_ours["obs"]["P_peak_raw_w"],
                           "pylle": fine_pylle["obs"]["P_peak_raw_w"],
                           "ours_band_limited": fine_ours["obs"]["P_peak_w"],
                           "pylle_band_limited": fine_pylle["obs"]["P_peak_w"],
                           "ours_subsample_offset": fine_ours["obs"]["peak_subsample_offset"],
                           "pylle_subsample_offset": fine_pylle["obs"]["peak_subsample_offset"],
                           "note": "the raw grid peak carries a sampling-phase bias; "
                                   "the band-limited peak is the gated quantity"},
        "spectral_residual_by_band": {
            "finest_levels": bands_finest,
            "matched_coarse_n1_vs_dt1_shipped": bands_v1_like,
            "dt1_tight_picard": bands_dt1_tight,
            "picard_floor_hypothesis_test": picard_test,
        },
        "spectral_edge_dbc": spec_contain,
        "dint_phase_budget": {
            "n_modes_over_pi": int(np.count_nonzero(over)),
            "smallest_abs_mu_over_pi": (int(np.min(np.abs(mu[over]))) if np.any(over)
                                        else None),
            "max_abs_Dint_t_r": max_dint_tr,
            "note": "a per-round-trip phase above pi is under-resolved at dt = 1 in "
                    "BOTH codes; it is a shared property of the discretization",
        },
    }

    # ---------------- criteria ---------------------------------------------
    # Tolerances are DERIVED from the two committed convergence studies and
    # fingerprinted before any comparison value below is read.
    tolset = CR.derive_tolerances(argv=raw_argv)

    # The GATED point comparison pairs each side's value with the state whose
    # uncertainty the tolerance was derived from: ours at n_substeps = 1
    # (Prompt A's ours_level="numerical_uncertainty_at_n1") against pyLLE's
    # tight ladder at its finest level (pylle_tag="tight", whose U_obs is the
    # Richardson band at that level). Pairing a value with someone else's
    # uncertainty is what would make the tolerance meaningless.
    gated_ours_level = coarse_ours
    gated_pylle_level = fine_pylle

    values = {
        "parameter_round_trip_max_rel": float(rt["_worst_rel_diff"]),
        "dispersion_refit_max_rel": refit_rel,
        "dispersion_mirror_applied": True,
        # Finding 8: the resonance pyLLE's own dispersion search locked onto,
        # against the CSV mu = 0 row our D_int is referenced to. Comparing the
        # frequency we ASKED for would be a tautology; this is not.
        "pump_mode_reference_delta_hz": max(
            abs(float(lv["pump_freq_hz_located_by_pylle"]) - f_pmp)
            for lv in ladder_pylle_shipped + ladder_pylle_tight),
        "detuning_endpoint_match_rel": detuning_match_rel,
        "delta_omega_eff_max_rel": eff_rel_worst,
        "seed_arrays_identical": all(
            lv["seed_sha256_pylle_convention"] == seed_pylle_sha
            for lv in ladder_pylle_shipped + ladder_pylle_tight),
    }
    for cid in CR.DERIVED_GATED_IDS:
        c = CR.CRITERIA_BY_ID[cid]
        values[f"{c.ours_key}__ours"] = gated_ours_level["obs"].get(c.ours_key)
        values[f"{c.pylle_key}__pylle"] = gated_pylle_level["obs"].get(c.pylle_key)
    for edge in ("lower", "upper"):
        for side in ("ours", "pylle"):
            values[f"existence_{edge}_bracket_{side}"] = existence[f"{edge}_bracket_{side}"]
            values[f"existence_{edge}_u_{side}"] = existence[f"{edge}_u_{side}"]
    for cid in CR.DIAGNOSTIC_IDS:
        values[f"diag_{CR.CRITERIA_BY_ID[cid].name}"] = diag.get(CR.CRITERIA_BY_ID[cid].name)

    # ---------------- provenance -------------------------------------------
    import platform as _plat
    import jax
    import scipy
    julia_ver = subprocess.run([args.julia_bin, "--version"], capture_output=True,
                               text=True).stdout.strip()
    jl_pkgs = subprocess.run(
        [args.julia_bin, "-e",
         'using Pkg; for (k,v) in Pkg.dependencies(); if v.name in ("FFTW","HDF5"); '
         'println(v.name, " ", v.version); end; end'],
        capture_output=True, text=True, env=env).stdout.strip()

    patch_files = sorted((THIRD_PARTY / "patches").glob("*.patch"))
    provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": raw_argv,
        "command": "python -m validation.pylle_crosscheck_v2 " + " ".join(raw_argv),
        "defaults_reproduce_this_run": (
            args.mu_half == MU_HALF and args.roundtrips == N_ROUNDTRIPS_MAIN
            and args.scan_roundtrips == N_ROUNDTRIPS_SCAN
            and args.dw_start == DW_START_KAPPA and args.dw_end == DW_END_KAPPA
            and args.scan == SCAN_KAPPA and args.bisect_iters == BISECT_ITERS
            and args.ours_levels == OURS_LEVELS and args.pylle_levels == PYLLE_LEVELS),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": _plat.node(),
        "platform": _plat.platform(),
        "work_dir": str(work),
        "versions": {
            "ours": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "jax": jax.__version__},
            "pylle_env": pl_main["versions"],
            "julia": julia_ver,
            "julia_packages": jl_pkgs,
            "julia_depot_path": env.get("JULIA_DEPOT_PATH", ""),
        },
        "vendored_kernel": {
            "path": str(Path(args.vendored_kernel)),
            "sha256": _sha_file(Path(args.vendored_kernel)),
            "integrity": integrity,
            "patches": [{"name": p.name, "sha256": _sha_file(p),
                         "lines": len(p.read_text().splitlines())}
                        for p in patch_files],
        },
        "physical_parameters": dev,
        "derived_config": {
            # Content-addressed, not path-addressed: the config lives in an
            # ephemeral work directory whose name changes every run, and a
            # changing path would make the determinism digest report a
            # difference where none exists. The full text is in the npz.
            "filename": Path(cfg).name,
            "sha256": _sha_file(Path(cfg)),
            "overrides": {"dn_dT_per_k": 0.0, "fsr_hz": fsr_used},
            "override_reasons": {
                "dn_dT_per_k": "pyLLE has no thermo-optic ODE; switching it off is how "
                               "analytic_cw.py does it. delta_omega_eff == delta_omega "
                               "exactly, which H6 measures.",
                "fsr_hz": "Finding 9: pyLLE hardwires t_r = 2*pi/D1, so one D1 must "
                          "serve both the co-moving frame and the timebase. The "
                          "committed leaf is 24.6 GHz; the CSV fit gives "
                          f"{fsr_used / 1e9:.6f} GHz. config/sin_params.yaml is not "
                          "modified.",
            },
            "fsr_hz_config_leaf": dev["fsr_hz"],
            "fsr_hz_used": fsr_used,
            "fsr_rel_diff": abs(fsr_used - dev["fsr_hz"]) / dev["fsr_hz"],
        },
        "solver_flags": {
            "symmetric_drive": False, "edge_absorber": False,
            "dealias_two_thirds": False, "dispersion_validity_mask": False,
            "fine_cadence_M": 1, "noise_config": "NoiseConfig.all_off()",
            "thermal_coupling": "lagged (decoupled: dn_dT_per_k = 0)",
            "rng_key": "jax.random.PRNGKey(0), unused -- every noise channel is off",
        },
        "dispersion_csv": {
            "path": str(DISP_CSV), "sha256": _sha_file(DISP_CSV),
            "mu_min": int(mu_csv.min()), "mu_max": int(mu_csv.max()),
            "n_rows": int(mu_csv.size),
        },
        "observable_definitions": OBSERVABLE_DEFINITIONS,
        "observable_constants": OBSERVABLE_CONSTANTS,
        "observable_source": "validation.convergence_lle.observables_v2",
        "criteria_source": "validation.criteria",
    }

    extra = {
        "provenance": provenance,
        "conventions": {
            "source": "validation/pylle_crosscheck.py module docstring, Findings 1-9, "
                      "established from pyLLE 4.1.2 source and carried over verbatim "
                      "by import; not re-derived here",
            "finding_1_d_int_units": "rad/s on both sides, no 2*pi anywhere "
                                     "(ComputeLLE.jl:338, :88)",
            "finding_2_field_map": "E_ours = conj(A_pyLLE)*sqrt(t_r)",
            "finding_2_detuning_map": "delta_omega_pylle = -delta_omega_ours",
            "finding_2b_dispersion_map": "D_int_pyLLE(mu) = D_int_ours(-mu); the "
                                         "conjugation forces the mirror on the INPUT",
            "finding_3_power_map": "|A|^2 = |E|^2 / t_r",
            "finding_4_drive_phase": "phi_pmp = -pi/2 makes pyLLE's drive real positive",
            "finding_5_determinism": "pyLLE's Noise() call site is commented out; both "
                                     "codes get the same explicit sech seed",
            "finding_6_rezero": "pyLLE re-zeros D_int at the domain centre; verified a "
                                "no-op with the pump centred",
            "finding_7_refit": "pyLLE spline-refits its dispersion CSV; its own D_int is "
                               "fed back to BOTH codes so the refit cancels",
            "finding_8_pump_reference": "sim['f_pmp'] pinned to the CSV mu=0 resonance, "
                                        f"{pump_offset_fsr:+.2f} FSR from c/lambda_nominal",
            "finding_9_d1_timebase": "D1 sets the frame AND pyLLE's t_r; D1_csv is used "
                                     "and our fsr_hz follows it",
            "dispersion_mirror_applied": True,
            "dispersion_refit_max_rel_diff_vs_our_loader": refit_rel,
            "f_pmp_hz_used": f_pmp,
            "f_pmp_hz_nominal_c_over_lambda": C0 / dev["pump_wavelength_m"],
            "pump_offset_from_nominal_fsr": pump_offset_fsr,
        },
        "translation": dict(fwd),
        "translation_round_trip": rt,
        "frame_and_timebase": {
            "D1_rad_per_s": d1_csv, "fsr_hz_used": fsr_used,
            "fsr_hz_config_leaf": dev["fsr_hz"], "t_r_s": t_r,
            "kappa_rad_per_s": kappa, "kappa_t_r": kappa * t_r,
        },
        "discretization": {
            "n_modes": n_modes, "mu_min": int(mu[0]), "mu_max": int(mu[-1]),
            "n_roundtrips_main": args.roundtrips,
            "n_roundtrips_scan": args.scan_roundtrips,
            "ours_levels_n_substeps": ours_levels,
            "pylle_levels_dt": dts,
            "pylle_picard_shipped": {"tol": PICARD_SHIPPED[0], "maxiter": PICARD_SHIPPED[1]},
            "pylle_picard_tight": {"tol": PICARD_TIGHT[0], "maxiter": PICARD_TIGHT[1]},
            "num_probe": args.num_probe,
            "max_abs_Dint_t_r_rad": max_dint_tr,
        },
        "initialization": {
            "seed_kind": "deterministic sech DKS ansatz on the lower CW branch",
            "seed_amplitude": "A^2 = 2*delta_omega/gamma_LLE",
            "seed_width_theta": "w = sqrt(|D2|/(2*delta_omega))",
            "local_D2_rad_per_s2": d2_local,
            "seed_delta_omega_rad_per_s": dw_start,
            "seed_sha256": seed_main_sha,
            "randomness": "none on either side (Finding 5)",
        },
        "detuning": {
            "ramp_start_over_kappa": args.dw_start,
            "ramp_end_over_kappa": args.dw_end,
            "rule": "numpy.linspace(dw_start, dw_end, n_roundtrips), one value per "
                    "round trip; pyLLE is given domega_init/-end with the sign flipped "
                    "and builds the same linear ramp internally",
            "final_delta_omega_ours_rad_per_s": dw_end,
            "final_delta_omega_pylle_rad_per_s": dw_p_final,
            "final_delta_omega_pylle_sign_flipped": -dw_p_final,
            "endpoint_match_rel": detuning_match_rel,
            "endpoint_note": "patch 0003 makes pyLLE's last probe the true end-of-run "
                             "state; upstream returned round trip 4976 of 5000",
        },
        "ladders": {
            "ours": [{k: v for k, v in lv.items() if not k.startswith("_")}
                     for lv in ladder_ours],
            "pylle": [{k: v for k, v in lv.items() if not k.startswith("_")}
                      for lv in ladder_pylle_shipped + ladder_pylle_tight],
        },
        "convergence": {
            "ours": conv_ours,
            "pylle_shipped_picard": conv_pylle_shipped,
            "pylle_tight_picard": conv_pylle_tight,
        },
        "containment": containment,
        "containment_vs_shipped_picard_ladder": containment_shipped,
        "containment_separated": separated,
        "gated_comparison_levels": {
            "ours": gated_ours_level["level"], "pylle": gated_pylle_level["level"],
            "rationale": "each side's value is paired with the state whose measured "
                         "uncertainty Prompt A derived its tolerance from: ours at "
                         "n_substeps=1 (ours_level='numerical_uncertainty_at_n1') and "
                         "pyLLE at the finest level of the tight ladder "
                         "(pylle_tag='tight'). Chosen by the frozen derivation, not "
                         "from these results.",
        },
        "existence": existence,
        "spectral_containment": spec_contain,
        "diagnostics_detail": diag,
    }

    report = CR.build_report(values, tolset, extra=extra)

    # ---------------- artifacts ---------------------------------------------
    npz_path = RESULTS_DIR / "pylle_crosscheck_v2_fields.npz"
    arrays = {
        "mu": mu,
        "d_int_rad_per_s": d_int_unmirrored,
        "d_int_pylle_mirrored_rad_per_s": d_int_pylle_natural,
        "seed_ours": np.asarray(seed_main, dtype=np.complex128),
        "derived_config_text": np.frombuffer(Path(cfg).read_bytes(), dtype=np.uint8),
        "t_r_s": np.array([t_r]),
        "kappa_rad_per_s": np.array([kappa]),
        "delta_omega_ramp": ramp,
    }
    for lv in ladder_ours:
        arrays[f"field_ours_{lv['level']}"] = lv["_field"]
    for lv in ladder_pylle_shipped + ladder_pylle_tight:
        arrays[f"field_pylle_{lv['level']}"] = lv["_field"]
    np.savez_compressed(npz_path, **arrays)
    print(f"\nwrote {npz_path}")

    png_path = RESULTS_DIR / "pylle_crosscheck_v2.png"
    fig_sha = make_figure(
        png_path, mu=mu, dbc_ours_fine=dbc_o_fine, dbc_pylle_fine=dbc_p_fine,
        bands=bands_finest, d_int_nat=d_int_unmirrored, t_r=t_r,
        ladders={"ours": ladder_ours, "pylle": ladder_pylle_tight},
        containment=containment, existence=existence, mu_half=args.mu_half,
        spec_contain=spec_contain_ours, ours_label=fine_ours["level"],
        pylle_label=fine_pylle["level"])
    report["provenance"]["figure"] = {"path": str(png_path), "sha256": fig_sha,
                                      "generated": "in-run by this process"}
    report["provenance"]["fields_npz"] = {"path": str(npz_path),
                                          "sha256": _sha_file(npz_path)}
    report["provenance"]["wall_clock_total_s"] = time.time() - t_start
    report["numerical_digest"] = numerical_digest(report)

    CR.validate_report(report)
    json_path = RESULTS_DIR / "pylle_crosscheck_v2.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {png_path}\nwrote {json_path}")

    _print_report(report, containment, containment_shipped, ladder_ours,
                  ladder_pylle_shipped + ladder_pylle_tight, existence,
                  spec_contain, picard_test)

    return 0 if report["overall"] == "PASS" else 1


def _br_str(br, kappa: float) -> str:
    """Format a bracket in units of kappa, or a dash when it is missing.

    Parameters
    ----------
    br : sequence of float or None
        A two-element bracket [rad/s].
    kappa : float
        Total loss rate [rad/s], the unit the bracket is reported in.

    Returns
    -------
    str
        ``"[lo,hi]"`` in units of kappa to four decimals, or ``"-"``.

    Raises
    ------
    ZeroDivisionError
        If ``kappa`` is zero.

    Examples
    --------
    >>> _br_str((3.0e8, 4.5e8), kappa=1.5e8)
    '[2.0000,3.0000]'
    >>> _br_str(None, kappa=1.5e8)
    '-'
    """
    return "-" if br is None else f"[{br[0] / kappa:.4f},{br[1] / kappa:.4f}]"


def _print_report(report, containment, containment_shipped, lad_o, lad_p,
                  existence, spec_contain, picard_test) -> None:
    """Print the full v2 comparison report to stdout.

    Parameters
    ----------
    report : dict
        The assembled report from :mod:`validation.criteria`.
    containment : dict
        Per-observable containment verdicts at the TIGHT solver settings.
    containment_shipped : dict
        The same at the SHIPPED settings, so the two can be read side by side.
    lad_o, lad_p : dict
        Each code's refinement ladder.
    existence : dict
        The existence-edge brackets [units of kappa].
    spec_contain : dict
        The result of :func:`spectral_containment`.
    picard_test : dict
        pyLLE's Picard-tolerance sensitivity check.

    Returns
    -------
    None
        Writes to stdout.

    Raises
    ------
    KeyError
        If any input is missing a field the report displays.

    Notes
    -----
    Both the shipped and the tight settings are printed, deliberately: a
    verdict that holds only at one of them is a different claim from one that
    holds at both, and collapsing them to a single column would hide which is
    which.

    No ``Examples`` section: it prints a wide multi-section report.
    """
    sep = [k for k, v in containment.items() if v["verdict"] == "SEPARATED"]
    print("\n" + "=" * 104)
    if sep:
        print("SEPARATED CONTAINMENT VERDICTS  --  the two codes converge to "
              "DIFFERENT limits for:")
        for k in sep:
            c = containment[k]
            print(f"   {k}: ours {c['ours_limit']:.6g} +/- {c['ours_U_in_comparison_units']:.3g}"
                  f"   pyLLE {c['pylle_limit']:.6g} +/- {c['pylle_U_in_comparison_units']:.3g}"
                  f"   gap {c['limit_gap']:.4g} vs band {c['band']:.4g}")
    else:
        print("No SEPARATED containment verdict: no observable shows the two codes "
              "converging to demonstrably different limits.")
    print("=" * 104)

    print("\nENVELOPE CONTAINMENT  (ours ladder vs pyLLE tight-Picard ladder)")
    print("-" * 104)
    print(f"{'observable':<16}{'ours limit':>15}{'U_o':>11}{'pyLLE limit':>15}{'U_p':>11}"
          f"{'gap':>11}{'band':>11}  verdict")
    for k, c in containment.items():
        print(f"{k:<16}{_fmt(c['ours_limit'], '>15.6g')}"
              f"{_fmt(c.get('ours_U_in_comparison_units'), '>11.3g')}"
              f"{_fmt(c['pylle_limit'], '>15.6g')}"
              f"{_fmt(c.get('pylle_U_in_comparison_units'), '>11.3g')}"
              f"{_fmt(c['limit_gap'], '>11.4g')}{_fmt(c['band'], '>11.4g')}  "
              f"{c['verdict']}"
              + (f"  ({c['reason']})" if c["verdict"] == "INDETERMINATE" and c["reason"]
                 else ""))

    print("\nCHECKS")
    print("-" * 104)
    print(f"{'cid':<5}{'class':<12}{'name':<34}{'ours':>13}{'pyLLE':>13}"
          f"{'diff':>11}{'tol':>10}  verdict")
    def cell(v, width=13):
        if isinstance(v, bool) or v is None:
            return f"{str(v):>{width}}"
        if isinstance(v, (int, float)):
            return f"{v:>{width}.6g}"
        return f"{str(v):>{width}}"

    for c in report["checks"]:
        if c["class"] == "DIAGNOSTIC":
            continue
        left = c["ours"] if "ours" in c else c.get("value", c.get("estimate_ours"))
        right = c["pylle"] if "pylle" in c else c.get("estimate_pylle")
        diff = c.get("diff", c.get("abs_diff"))
        tol = c.get("tolerance", c.get("threshold"))
        # G7 emits one row per edge; without the edge in the label the two rows
        # are indistinguishable in the table.
        label = c["name"] + (f" [{c['edge']}]" if "edge" in c else "")
        print(f"{c['cid']:<5}{c['class']:<12}{label:<34}"
              f"{cell(left)}{cell(right)}{cell(diff, 11)}{cell(tol, 10)}  {c['verdict']}")

    print(f"\nPicard-floor hypothesis test: wing gap "
          f"{_fmt(picard_test['wing_median_gap_db_dt1_shipped_tol'], '+.2f')} dB "
          f"(shipped tol) -> "
          f"{_fmt(picard_test['wing_median_gap_db_dt1_tight_tol'], '+.2f')} dB "
          f"(tight tol)   {picard_test['verdict']}")
    if spec_contain["warning"]:
        print(f"\n{spec_contain['warning']}")
    print(f"\nOVERALL: {report['overall']}"
          + ("  (QUALIFIED)" if report["overall_qualified"] else ""))
    for r in report["overall_qualification_reasons"]:
        print(f"   - {r}")
    print("=" * 104)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker_main([a for a in sys.argv[1:] if a != "--worker"]))
    sys.exit(main())
