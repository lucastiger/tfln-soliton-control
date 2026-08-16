"""Discretization-uncertainty harness for the LLE solver at a physical operating point.

WHAT THIS IS FOR, AND HOW IT DIFFERS FROM THE EXISTING CONVERGENCE MODULES
--------------------------------------------------------------------------
``validation/convergence.py`` and ``validation/mms.py`` verify that the SCHEME
converges to the PDE it claims to solve, against manufactured solutions. That is
a statement about the discretization in the abstract.

This module answers a different, narrower question: **at one specific, difficult
physical operating point, how far is each observable from its h -> 0 limit, and
what numerical uncertainty are we entitled to claim for it?** The answer is a
prerequisite for any cross-code tolerance: a cross-code threshold tighter than a
code's own demonstrated discretization uncertainty is not a physics test, it is
a coin flip. The ``U_obs`` values produced here are meant to be consumed as
tolerances downstream.

Nothing here runs, imports, or refers to pyLLE as a reference. No other code is
treated as ground truth.

THE OPERATING POINT, AND WHY IT IS THE HARD ONE
-----------------------------------------------
Reproduced from ``validation/results/pylle_crosscheck.json``: measured SiN
dispersion, mu in [-3300, 3300] (6601 modes), 5000 round trips, delta_omega
ramped 16 kappa -> 30 kappa, deterministic sech seed on the lower CW branch,
one step per round trip. At that point:

* the Kerr phase per round trip at the soliton peak is ``2*delta_omega*t_r``
  = 0.373 rad -- not a small angle for a first-order splitting;
* the soliton width parameter ``w = sqrt(D2/(2*delta_omega))`` is 2.46 SAMPLES
  (FWHM 4.34 samples), so ``max|E_j|^2`` is sampling-phase noise as much as
  physics;
* 439 modes carry ``|D_int|*t_r > pi``;
* the spectrum is still at about -52 dBc at the grid edge, i.e. the comb is not
  spectrally contained.

WHY THE OBSERVABLE DEFINITIONS HERE DIFFER FROM ``pylle_crosscheck.observables``
--------------------------------------------------------------------------------
Each redefinition below fixes a specific ill-conditioning, and each is stated
with its rationale so it is not "simplified" back later:

* **Peak power** (:func:`band_limited_peak`) is evaluated on a 32x zero-padded
  FFT. The field is band-limited and periodic, so its true maximum is not
  ``max|E_j|^2``; with FWHM = 4.34 samples the sampling-phase error bound is
  ``1 - sech^2(dtheta/(2w))`` = 4.02%. The legacy value is reported alongside as
  ``P_peak_raw`` together with the sub-sample offset, so the contamination is
  visible rather than inferred.
* **3 dB span** (:func:`subbin_span_3db`) crosses the -3 dB level by linear
  interpolation in dB. The integer version has an intrinsic quantum of
  ``1/S3``: at this operating point S3 = 443 so one quantum is 0.226%, and the
  previously reported "0.23% agreement" is exactly one quantum -- the smallest
  non-zero value the metric can express. At delta_omega = 8 kappa, S3 = 64 and
  the same quantum is 1.6%, so identical solution quality would "fail" a 2%
  test purely from quantization.
* **DW position** (:func:`dw_centroid`) is a pedestal-subtracted power centroid
  over a FIXED, code-independent mu band -- not an argmax, and not a
  boxcar-smoothed argmax. The wing carries soliton/DW walk-off fringes with a
  period of order 10-13 modes, comparable to the 25-point boxcar previously
  used, so the smoothed argmax moves with the fringe phase.
* **Line count** ``N60`` is retained as a DIAGNOSTIC ONLY and is explicitly
  disqualified as an acceptance criterion; its conditioning ``dN60/ddB`` and its
  core/mid/edge decomposition are recorded so the reader can see where any
  disagreement lives.
* **Integral observables** (:func:`integral_observables`) -- mean intracavity
  power (Parseval-exact) and the comb fraction -- are translation-invariant and
  well-conditioned, and are the ones fit for cross-code use.

Reference for the physics: Herr, Tikan & Kippenberg, arXiv:2604.05897v1
(7 Apr 2026).

CLI
---
``python -m validation.convergence_lle --operating-point dw30k --levels 1,2,4,8,16``
writes ``validation/results/convergence_lle_dw30k.json`` and the companion
``..._fields.npz``. Exit status is nonzero only on a HARNESS failure (F1-F4); an
observable that fails to converge is a scientific result, is reported with a
conservative uncertainty band, and does not fail the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DISP_CSV = REPO_ROOT / "config" / "pyLLE_dispersion_w4400_h800.csv"
RESULTS_DIR = REPO_ROOT / "validation" / "results"
COMMITTED_FIELDS = RESULTS_DIR / "pylle_crosscheck_fields.npz"

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Binding numerical constants. Fixed BEFORE any run; changing one is a
# provenance-recorded event, never a post-hoc adjustment.
# ---------------------------------------------------------------------------
UPSAMPLE = 32                      # D1 zero-pad factor for the band-limited peak
DW_BAND_DEFAULT = (-3200, -2950)   # D3 fixed red-DW band at the 30 kappa point
DBC_THRESHOLD = -60.0              # D4 line-count threshold
PEDESTAL_PCT = 10.0                # D3 pedestal percentile within the DW band
GCI_FS = 1.25                      # D6 safety factor, >=3 levels with observed p
ORDER_MIN, ORDER_MAX = 0.5, 4.0    # D6 acceptable observed-order window
D2_LOCAL_COMMITTED = 50011.53497093499   # from pylle_crosscheck.json initialization

OBSERVABLE_DEFINITIONS = {
    "P_peak_w": "max|E_up|^2/t_r on a 32x zero-padded FFT (band-limited peak)",
    "P_peak_raw_w": "max|E_j|^2/t_r on the native grid (legacy; sampling-phase biased)",
    "peak_subsample_offset": "argmax position on the upsampled grid, in native samples, minus native argmax",
    "S3_modes": "sub-bin 3 dB span: dB-linear interpolation of env(mu) across max(env)-3, pump bin excluded",
    "S3_int_modes": "legacy integer 3 dB extent (mu of last index >= level minus first)",
    "S3_n_runs": "number of disjoint index runs above the -3 dB level",
    "mu_DW": "pedestal-subtracted power centroid over the fixed band DW_BAND",
    "dw_power_dbc": "10*log10(sum p over DW band / max p over grid)",
    "N60": "count of modes with spectrum_dbc >= -60 (DIAGNOSTIC ONLY, never a gate)",
    "dN60_ddB": "N60(-60.5) - N60(-59.5), lines per dB (conditioning of N60)",
    "N60_core": "N60 restricted to |mu| <= 1500",
    "N60_mid": "N60 restricted to 1500 < |mu| <= 3000",
    "N60_edge": "N60 restricted to |mu| > 3000",
    "U_mean_w": "sum(|E_j|^2)/N/t_r  (Parseval-exact mean intracavity power)",
    "comb_frac": "(sum|E_mu|^2 - |E_0|^2)/sum|E_mu|^2",
}
OBSERVABLE_CONSTANTS = {
    "upsample_factor": UPSAMPLE,
    "dw_band": list(DW_BAND_DEFAULT),
    "dbc_threshold": DBC_THRESHOLD,
    "pedestal_percentile": PEDESTAL_PCT,
    "gci_safety_factor": GCI_FS,
    "order_window": [ORDER_MIN, ORDER_MAX],
}

# Observables that carry a Richardson study. N60 and the legacy/quantized
# variants are deliberately excluded from convergence gating (D4).
CONVERGED_OBSERVABLES = (
    "P_peak_w", "P_peak_raw_w", "S3_modes", "mu_DW",
    "dw_power_dbc", "U_mean_w", "comb_frac",
)


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    mu_half: int
    n_roundtrips: int
    dw_start_over_kappa: float
    dw_end_over_kappa: float
    config_path: str          # derived config actually used
    seed_kind: str            # "sech_lower_branch"


# ===========================================================================
# NUMERICAL DEFINITIONS  (D1-D5)
# ===========================================================================
def _spectrum_lines(field: np.ndarray) -> np.ndarray:
    """fftshift(fft(E))/N -- per-mode complex amplitude, mu-centred."""
    return np.fft.fftshift(np.fft.fft(field)) / field.size


def spectrum_dbc(field: np.ndarray) -> np.ndarray:
    """Per-mode power in dB relative to the strongest line."""
    power = np.abs(_spectrum_lines(field)) ** 2
    return 10.0 * np.log10(np.maximum(power, 1e-300) / power.max())


def band_limited_peak(field: np.ndarray, t_r: float, upsample: int = UPSAMPLE) -> dict:
    """D1. Peak intracavity power from a zero-padded (band-limited) reconstruction.

    The field is a band-limited periodic function sampled at N points, so its
    true maximum is not ``max|E_j|^2``. Zero-padding the spectrum to ``K = 32N``
    evaluates the same trigonometric interpolant on a 32x finer grid, which is
    exact reconstruction, not smoothing.

    Hermitian bin layout for ODD N: bins ``0..h`` are the non-negative
    frequencies and the top ``h`` bins are the negatives, ``h = (N-1)//2``. The
    ``K/N`` factor keeps ``ifft`` normalization consistent.
    """
    n = field.size
    if n % 2 == 0:
        raise ValueError(f"band_limited_peak assumes an ODD mode count, got {n}")
    h = (n - 1) // 2
    k = int(upsample) * n
    x = np.fft.fft(field)
    y = np.zeros(k, dtype=np.complex128)
    y[: h + 1] = x[: h + 1]
    y[k - h:] = x[h + 1:]
    up = np.fft.ifft(y) * (k / n)

    p_up = np.abs(up) ** 2
    i_up = int(np.argmax(p_up))
    p_raw = np.abs(field) ** 2
    i_raw = int(np.argmax(p_raw))
    # position of the upsampled argmax expressed in native samples, wrapped to
    # the shortest signed distance from the native argmax
    pos_native = i_up / float(upsample)
    offset = (pos_native - i_raw + n / 2.0) % n - n / 2.0
    return {
        "P_peak_w": float(p_up.max() / t_r),
        "P_peak_raw_w": float(p_raw.max() / t_r),
        "peak_subsample_offset": float(offset),
    }


def subbin_span_3db(field: np.ndarray, mu: np.ndarray) -> dict:
    """D2. 3 dB spectral span with sub-bin (dB-linear) level crossing.

    The pump bin carries the CW background and stands far above every comb line,
    so it is excluded; what is measured is the soliton's own spectral envelope.
    """
    env = spectrum_dbc(field).copy()
    env[mu == 0] = -np.inf
    level = float(env.max()) - 3.0
    above = env >= level
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return {"S3_modes": 0.0, "S3_int_modes": 0, "S3_n_runs": 0}
    i_l, i_r = int(idx[0]), int(idx[-1])

    def _cross(i_in: int, i_out: int) -> float:
        """Fractional mu where the dB-linear segment i_out->i_in crosses level."""
        if i_out < 0 or i_out >= env.size or not np.isfinite(env[i_out]):
            return float(mu[i_in])
        y0, y1 = env[i_out], env[i_in]
        if y1 == y0:
            return float(mu[i_in])
        frac = (level - y0) / (y1 - y0)
        return float(mu[i_out] + frac * (mu[i_in] - mu[i_out]))

    mu_left = _cross(i_l, i_l - 1)
    mu_right = _cross(i_r, i_r + 1)
    # number of disjoint runs above the level: the metric is an EXTENT, and if
    # this exceeds 1 it is not a contiguous span
    n_runs = int(np.count_nonzero(above & ~np.roll(above, 1)))
    if above[0] and above[-1] and n_runs > 0:
        n_runs -= 0  # circular wrap is not meaningful on a centred mu axis
    return {
        "S3_modes": float(mu_right - mu_left),
        "S3_int_modes": int(mu[i_r] - mu[i_l]),
        "S3_n_runs": n_runs,
    }


def dw_centroid(field: np.ndarray, mu: np.ndarray, band=DW_BAND_DEFAULT) -> dict:
    """D3. Pedestal-subtracted power centroid of the DW over a FIXED mu band.

    Fixed band + centroid, rather than argmax of a smoothed spectrum: the wing
    carries soliton/DW walk-off fringes of period ~10-13 modes, so a boxcar of
    comparable width displaces the smoothed argmax with the fringe phase, and
    differently for any two solutions.
    """
    p = np.abs(_spectrum_lines(field)) ** 2
    sel = (mu >= band[0]) & (mu <= band[1])
    if not np.any(sel):
        return {"mu_DW": None, "dw_power_dbc": None}
    pb = p[sel]
    pedestal = float(np.percentile(pb, PEDESTAL_PCT))
    q = np.maximum(pb - pedestal, 0.0)
    total = float(q.sum())
    centroid = float(np.sum(mu[sel] * q) / total) if total > 0 else None
    dw_dbc = 10.0 * math.log10(max(float(pb.sum()), 1e-300) / float(p.max()))
    return {"mu_DW": centroid, "dw_power_dbc": float(dw_dbc)}


def line_count_diagnostics(field: np.ndarray, mu: np.ndarray) -> dict:
    """D4. N60 and its conditioning. DIAGNOSTIC ONLY -- never an acceptance gate.

    ``dN60_ddB`` is the slope of the count against the threshold. A large value
    means the count is set by where a nearly-flat wing crosses an arbitrary
    line, not by the solution.
    """
    dbc = spectrum_dbc(field)
    n60 = int(np.count_nonzero(dbc >= DBC_THRESHOLD))
    n_lo = int(np.count_nonzero(dbc >= DBC_THRESHOLD - 0.5))
    n_hi = int(np.count_nonzero(dbc >= DBC_THRESHOLD + 0.5))
    a = np.abs(mu)
    return {
        "N60": n60,
        "dN60_ddB": int(n_lo - n_hi),
        "N60_core": int(np.count_nonzero((dbc >= DBC_THRESHOLD) & (a <= 1500))),
        "N60_mid": int(np.count_nonzero((dbc >= DBC_THRESHOLD) & (a > 1500) & (a <= 3000))),
        "N60_edge": int(np.count_nonzero((dbc >= DBC_THRESHOLD) & (a > 3000))),
    }


def integral_observables(field: np.ndarray, t_r: float, mu: np.ndarray) -> dict:
    """D5. Translation-invariant integrals -- the well-posed observables."""
    p_time = np.abs(field) ** 2
    u_mean = float(p_time.sum() / field.size / t_r)
    p_mode = np.abs(_spectrum_lines(field)) ** 2
    tot = float(p_mode.sum())
    pump = float(p_mode[mu == 0].sum())
    return {
        "U_mean_w": u_mean,
        "comb_frac": float((tot - pump) / tot) if tot > 0 else 0.0,
    }


def observables_v2(field: np.ndarray, t_r: float, mu: np.ndarray,
                   band=DW_BAND_DEFAULT) -> dict:
    """All of D1-D5 for one field."""
    out = {}
    out.update(band_limited_peak(field, t_r))
    out.update(subbin_span_3db(field, mu))
    out.update(dw_centroid(field, mu, band))
    out.update(line_count_diagnostics(field, mu))
    out.update(integral_observables(field, t_r, mu))
    return out


# ===========================================================================
# D6 -- Richardson extrapolation and numerical uncertainty
# ===========================================================================
def richardson(values, h_values) -> dict:
    """Observed order, extrapolated limit and GCI from >=3 levels (finest last).

    ``values`` and ``h_values`` are ordered coarse -> fine. Uses the three
    finest levels. Every guard demanded by D6 is enforced, and the raw level
    values are always returned alongside whatever status is produced.
    """
    vals = [None if v is None else float(v) for v in values]
    hs = [float(h) for h in h_values]
    base = {"levels": vals, "h": hs, "observed_order": None,
            "extrapolated": None, "gci": None, "U_obs": None}

    if any(v is None or not np.isfinite(v) for v in vals) or len(vals) < 3:
        base["status"] = "INSUFFICIENT_LEVELS"
        return base

    f3, f2, f1 = vals[-3], vals[-2], vals[-1]      # coarse, mid, fine
    h3, h2, h1 = hs[-3], hs[-2], hs[-1]
    r = h2 / h1
    d_coarse, d_fine = (f2 - f3), (f1 - f2)
    scale = abs(f1) if f1 != 0 else 1.0
    conservative = max(abs(d_fine), abs(d_coarse)) / scale

    if d_coarse == 0.0 or d_fine == 0.0 or (d_coarse > 0) != (d_fine > 0):
        base["status"] = "NON_MONOTONE"
        base["U_obs"] = float(conservative)
        return base

    p = math.log(abs(d_coarse / d_fine)) / math.log(r)
    base["observed_order"] = float(p)
    if not (ORDER_MIN <= p <= ORDER_MAX):
        base["status"] = "ORDER_OUT_OF_RANGE"
        base["U_obs"] = float(conservative)
        return base

    q_star = f1 + d_fine / (r ** p - 1.0)
    e_fine = abs(q_star - f1) / (abs(q_star) if q_star != 0 else 1.0)
    gci = GCI_FS * e_fine
    base.update({
        "extrapolated": float(q_star),
        "gci": float(gci),
        "status": "OK",
        # U_obs is the max of the GCI and the raw finest-gap, so a suspiciously
        # small GCI from a fortuitous fit can never shrink the claimed band
        # below the difference actually observed between the two finest levels.
        "U_obs": float(max(gci, abs(d_fine) / scale)),
    })
    return base


# ===========================================================================
# Solver driving
# ===========================================================================
def _device_and_frame():
    """Device parameters, kappa, the CSV-fitted D1 and the resulting t_r."""
    from simulator.lle_solver import load_dint_grid
    from validation.pylle_crosscheck import load_device

    dev = load_device()
    kappa = dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]
    d1 = float(load_dint_grid(6601, csv_path=DISP_CSV).d1)
    fsr_used = d1 / (2.0 * math.pi)
    return dev, kappa, d1, fsr_used, 1.0 / fsr_used


def committed_d_int_fftorder() -> np.ndarray:
    """D_int in FFT-bin order, taken from the committed cross-check artifact.

    A1 requires the substep ladder to drive exactly the committed problem, so
    the dispersion array comes from the committed npz (stored mu-centred) rather
    than being re-derived.
    """
    with np.load(COMMITTED_FIELDS) as z:
        return np.fft.ifftshift(np.asarray(z["d_int_rad_per_s"], dtype=float))


def build_seed(n_modes: int, dw: float, dev: dict, kappa: float,
               d2_local: float) -> np.ndarray:
    """Deterministic sech-on-lower-CW-branch seed (reused verbatim from the
    cross-check module so the two harnesses cannot drift apart)."""
    from validation.pylle_crosscheck import soliton_seed
    return soliton_seed(n_modes, dw, kappa, dev["kappa_c_rad_per_s"],
                        dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)


def run_level(op: OperatingPoint, n_substeps: int, d_int_fftorder, dev, *,
              fine_cadence_M: int = 1) -> dict:
    """Run the solver at one refinement level; return the final field and timing."""
    import jax
    from simulator.lle_solver import solve_lle_ssfm_jax
    from simulator.noise_config import NoiseConfig

    kappa = dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]
    n_modes = 2 * op.mu_half + 1
    d2_local = dev.get("_d2_local", D2_LOCAL_COMMITTED)
    seed = build_seed(n_modes, op.dw_start_over_kappa * kappa, dev, kappa, d2_local)
    ramp = np.linspace(op.dw_start_over_kappa * kappa,
                       op.dw_end_over_kappa * kappa, op.n_roundtrips)

    t0 = time.time()
    out = solve_lle_ssfm_jax(
        pin=dev["pin_w"],
        delta_omega=ramp,
        t_slow=int(op.n_roundtrips),
        beta=[0.0, 0.0],                       # unused: d_int_grid drives dispersion
        kappa=kappa,
        kappa_c=dev["kappa_c_rad_per_s"],
        rng_key=jax.random.PRNGKey(0),         # unused: every channel off
        n_tau=n_modes,
        config_path=op.config_path,
        snapshot_interval=int(op.n_roundtrips),
        e0_override=np.asarray(seed, dtype=np.complex128),
        d_int_grid=np.asarray(d_int_fftorder, dtype=float),
        n_substeps=int(n_substeps),
        fine_cadence_M=int(fine_cadence_M),
        noise_config=NoiseConfig.all_off(),
    )
    wall = time.time() - t0

    # F4: the thermo-optic shift must be off, so the effective detuning must BE
    # the programmed ramp. Same assertion as pylle_crosscheck.run_ours.
    eff = np.asarray(out["delta_omega_eff_history"], dtype=float).ravel()
    worst = float(np.max(np.abs(eff - ramp.ravel())))
    if worst > 1e-6 * max(1.0, float(np.max(np.abs(ramp)))):
        raise AssertionError(
            f"delta_omega_eff deviates from the programmed ramp by {worst:.3e} rad/s "
            "(F4): the thermo-optic shift is not off."
        )
    field = np.asarray(out["e_final"], dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(field)):
        raise AssertionError(f"level n_substeps={n_substeps} returned a non-finite field (F3)")
    return {"field": field, "n_substeps": int(n_substeps),
            "h": 1.0 / float(n_substeps), "wall_s": wall, "seed": seed}


# ===========================================================================
# Context diagnostics (D7, R3, R4)
# ===========================================================================
def dispersion_context(d_int_fftorder: np.ndarray, mu: np.ndarray, t_r: float,
                       dw_final: float) -> dict:
    """D7 phase budget, R4 provenance mask, and the DW phase-matching roots."""
    d_nat = np.fft.fftshift(d_int_fftorder)
    phase = np.abs(d_nat) * t_r
    over = phase > math.pi
    csv = np.loadtxt(DISP_CSV, delimiter=",")
    mu_csv = csv[:, 0].astype(np.int64)
    lo, hi = int(mu_csv.min()), int(mu_csv.max())
    extrap = (mu < lo) | (mu > hi)

    # phase matching D_int(mu) = delta_omega
    s = np.sign(d_nat - dw_final)
    roots = [int(mu[i]) for i in np.flatnonzero(np.diff(s) != 0)]
    far = [m for m in roots if abs(m) > 1500]
    return {
        "max_abs_Dint_t_r_rad": float(phase.max()),
        "n_modes_Dint_tr_over_pi": int(np.count_nonzero(over)),
        "min_abs_mu_Dint_tr_over_pi": int(np.min(np.abs(mu[over]))) if np.any(over) else None,
        "dispersion_fully_measured": bool(not np.any(extrap)),
        "modes_extrapolated": {
            "count": int(np.count_nonzero(extrap)),
            "mu_min": int(mu[extrap].min()) if np.any(extrap) else None,
            "mu_max": int(mu[extrap].max()) if np.any(extrap) else None,
            "csv_mu_range": [lo, hi],
        },
        "phase_matching_roots_far": far,
    }


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return "unknown"


def build_provenance(argv, op: OperatingPoint, dev, d1, fsr_used, t_r, seed,
                     overrides: dict, wall_total: float) -> dict:
    import jax
    import scipy
    import yaml

    with open(op.config_path, "r", encoding="utf-8") as fh:
        resolved = (yaml.safe_load(fh) or {})["physical_parameters"]
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "argv": list(argv),
        "resolved_config": {
            "physical_parameters": resolved,
            "derived_config_path": op.config_path,
            "overridden_leaves": overrides,
            "source_config": str(REPO_ROOT / "config" / "sin_params.yaml"),
        },
        "solver_flags": {
            "symmetric_drive": False, "edge_absorber": False,
            "dealias_two_thirds": False, "dispersion_validity_mask": False,
            "fine_cadence_M": 1, "noise": "all_off", "dn_dT_per_k": 0.0,
        },
        "dispersion_source": {
            "csv": str(DISP_CSV),
            "sha256": hashlib.sha256(DISP_CSV.read_bytes()).hexdigest(),
            "d1_rad_per_s": d1,
            "csv_mu_range": [-3261, 5813],
            "d_int_array_source": f"{COMMITTED_FIELDS.name}::d_int_rad_per_s (ifftshifted)",
        },
        "seed": {
            "kind": op.seed_kind,
            "delta_omega_rad_per_s": op.dw_start_over_kappa
            * (dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]),
            "d2_rad_per_s2": D2_LOCAL_COMMITTED,
            "sha256": _sha(seed),
        },
        "detuning_trajectory": {
            "start_over_kappa": op.dw_start_over_kappa,
            "end_over_kappa": op.dw_end_over_kappa,
            "n_points": op.n_roundtrips,
            "rule": "np.linspace over t_slow",
        },
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "jax": jax.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "hostname": platform.node(),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_total_s": wall_total,
        "observable_definitions": OBSERVABLE_DEFINITIONS,
        "observable_constants": OBSERVABLE_CONSTANTS,
    }


# ===========================================================================
# The study
# ===========================================================================
def study(op: OperatingPoint, levels=(1, 2, 4, 8, 16, 32), *,
          grid_levels=(3300, 4400, 5500), grid_nsub: int = 8,
          band=DW_BAND_DEFAULT, max_seconds: float = 1800.0,
          argv=None, check_regression: bool = True) -> dict:
    """Run the substep ladder and the grid ladder; assemble the full result."""
    from simulator.lle_solver import load_dint_grid

    t_start = time.time()
    dev, kappa, d1, fsr_used, t_r = _device_and_frame()
    n_modes = 2 * op.mu_half + 1
    mu = np.arange(-op.mu_half, op.mu_half + 1)
    d_int_fft = committed_d_int_fftorder()
    dw_final = op.dw_end_over_kappa * kappa
    seed = build_seed(n_modes, op.dw_start_over_kappa * kappa, dev, kappa,
                      D2_LOCAL_COMMITTED)

    ctx = dispersion_context(d_int_fft, mu, t_r, dw_final)
    arrays = {"mu": mu, "d_int_rad_per_s": np.fft.fftshift(d_int_fft), "seed": seed}

    # ---- substep ladder ----
    level_records, fields = [], {}
    for n_sub in levels:
        try:
            res = run_level(op, n_sub, d_int_fft, dev)
        except AssertionError:
            raise
        except Exception as exc:                    # pragma: no cover - defensive
            level_records.append({"n_substeps": int(n_sub), "h": 1.0 / n_sub,
                                  "wall_s": None, "status": f"ERROR: {exc}",
                                  "observables": {}})
            continue
        status = "TIMEOUT" if res["wall_s"] > max_seconds else "OK"
        obs = observables_v2(res["field"], t_r, mu, band)
        fields[f"field_nsub{n_sub}"] = res["field"]
        level_records.append({"n_substeps": int(n_sub), "h": res["h"],
                              "wall_s": res["wall_s"], "status": status,
                              "observables": obs})
        print(f"  [nsub={n_sub:3d}] {res['wall_s']:7.1f}s  P_peak={obs['P_peak_w']:10.4f} W "
              f"(raw {obs['P_peak_raw_w']:10.4f})  S3={obs['S3_modes']:8.3f}  "
              f"mu_DW={obs['mu_DW']}  N60={obs['N60']}", flush=True)

    ok_levels = [r for r in level_records if r["status"] in ("OK", "TIMEOUT")]
    if len(ok_levels) < 3:
        raise RuntimeError(f"only {len(ok_levels)} levels completed; need >=3 (F2)")

    # ---- A1 regression gate ----
    regression = {"checked": False}
    if check_regression and "field_nsub1" in fields and COMMITTED_FIELDS.exists():
        with np.load(COMMITTED_FIELDS) as z:
            ref = np.asarray(z["field_ours"], dtype=np.complex128).reshape(-1)
        got = fields["field_nsub1"]
        rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref)) if got.size == ref.size \
            else float("inf")
        regression = {"checked": True, "relative_l2": rel, "tolerance": 1e-10,
                      "passed": bool(rel <= 1e-10)}

    # ---- grid ladder (R3) at fixed n_substeps ----
    grid_records = []
    for mh in grid_levels:
        nm = 2 * mh + 1
        op_g = OperatingPoint(op.name, mh, op.n_roundtrips, op.dw_start_over_kappa,
                              op.dw_end_over_kappa, op.config_path, op.seed_kind)
        g_int = np.asarray(load_dint_grid(nm, csv_path=DISP_CSV).grid, dtype=float)
        try:
            res = run_level(op_g, grid_nsub, g_int, dev)
        except Exception as exc:                    # pragma: no cover - defensive
            grid_records.append({"mu_half": mh, "n_modes": nm,
                                 "status": f"ERROR: {exc}"})
            continue
        mu_g = np.arange(-mh, mh + 1)
        dbc = spectrum_dbc(res["field"])
        obs = observables_v2(res["field"], t_r, mu_g, band)
        # blue DW: is the phase-matching root inside the grid, and is there a
        # bump there standing above the local wing?
        d_nat = np.fft.fftshift(g_int)
        s = np.sign(d_nat - dw_final)
        roots = [int(mu_g[i]) for i in np.flatnonzero(np.diff(s) != 0)]
        blue = [m for m in roots if m > 1500]
        resolved, blue_root = False, (blue[0] if blue else None)
        if blue_root is not None and blue_root + 120 < mh:
            w = (mu_g >= blue_root - 120) & (mu_g <= blue_root + 120)
            near = (np.abs(mu_g - blue_root) > 200) & (np.abs(mu_g - blue_root) < 500)
            if np.any(w) and np.any(near):
                resolved = bool(dbc[w].max() - float(np.median(dbc[near])) >= 3.0)
        grid_records.append({
            "mu_half": mh, "n_modes": nm, "n_substeps": grid_nsub, "status": "OK",
            "wall_s": res["wall_s"],
            "edge_dbc_red": float(dbc[0]), "edge_dbc_blue": float(dbc[-1]),
            "blue_root_mu": blue_root, "blue_dw_resolved": resolved,
            "dispersion_fully_measured": bool(mu_g.min() >= -3261),
            "observables": obs,
        })
        fields[f"field_muhalf{mh}"] = res["field"]
        arrays[f"mu_muhalf{mh}"] = mu_g
        print(f"  [mu_half={mh}] edge red {dbc[0]:7.2f} dBc  blue {dbc[-1]:7.2f} dBc  "
              f"blue_root={blue_root}  resolved={resolved}", flush=True)

    # ---- convergence per observable ----
    conv = {}
    usable = [r for r in level_records if r["status"] == "OK" and r["observables"]]
    for key in CONVERGED_OBSERVABLES:
        vals = [r["observables"].get(key) for r in usable]
        hs = [r["h"] for r in usable]
        if len(vals) < 3:
            conv[key] = {"levels": vals, "h": hs, "observed_order": None,
                         "extrapolated": None, "gci": None, "U_obs": None,
                         "status": "INSUFFICIENT_LEVELS"}
        else:
            conv[key] = richardson(vals, hs)

    def _u_at(n_sub: int) -> dict:
        """Relative distance from each observable's Richardson limit at one level."""
        rec = next((r for r in usable if r["n_substeps"] == n_sub), None)
        if rec is None:
            return {}
        out = {}
        for key, c in conv.items():
            v = rec["observables"].get(key)
            q = c.get("extrapolated")
            if v is None:
                continue
            if q is None or q == 0:
                out[key] = c.get("U_obs")
            else:
                out[key] = float(abs(v - q) / abs(q))
        return out

    overrides = {"dn_dT_per_k": 0.0, "fsr_hz": fsr_used}
    prov = build_provenance(argv or sys.argv, op, dev, d1, fsr_used, t_r, seed,
                            overrides, time.time() - t_start)
    prov["a1_regression"] = regression

    result = {
        "schema_version": SCHEMA_VERSION,
        "study": "lle_substep_convergence",
        "provenance": prov,
        "operating_point": {
            "name": op.name, "mu_half": op.mu_half, "n_modes": n_modes,
            "n_roundtrips": op.n_roundtrips,
            "dw_start_over_kappa": op.dw_start_over_kappa,
            "dw_end_over_kappa": op.dw_end_over_kappa,
            "kappa_rad_per_s": kappa, "t_r_s": t_r, "fsr_hz_used": fsr_used,
            "kerr_phase_per_roundtrip_rad": 2.0 * dw_final * t_r,
            **{k: ctx[k] for k in ("max_abs_Dint_t_r_rad", "n_modes_Dint_tr_over_pi",
                                   "min_abs_mu_Dint_tr_over_pi",
                                   "dispersion_fully_measured", "modes_extrapolated")},
            "phase_matching_roots_far": ctx["phase_matching_roots_far"],
            "soliton_w_samples": float(
                math.sqrt(D2_LOCAL_COMMITTED / (2.0 * dw_final)) / (2 * math.pi / n_modes)),
        },
        "solver_flags": prov["solver_flags"],
        "levels": level_records,
        "grid_levels": grid_records,
        "convergence": conv,
        "numerical_uncertainty_at_n1": _u_at(1),
        "numerical_uncertainty_at_n8": _u_at(8),
    }
    return result, fields, arrays


# ===========================================================================
# CLI
# ===========================================================================
OPERATING_POINTS = {
    "dw30k": dict(mu_half=3300, n_roundtrips=5000,
                  dw_start_over_kappa=16.0, dw_end_over_kappa=30.0),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--operating-point", default="dw30k", choices=sorted(OPERATING_POINTS))
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--grid-levels", default="3300,4400,5500")
    ap.add_argument("--grid-nsub", type=int, default=8)
    ap.add_argument("--dw-band", default=f"{DW_BAND_DEFAULT[0]},{DW_BAND_DEFAULT[1]}")
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    ap.add_argument("--no-regression-check", action="store_true")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args(argv)
    argv_full = list(sys.argv)

    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    glevels = tuple(int(x) for x in args.grid_levels.split(",") if x.strip())
    band = tuple(int(x) for x in args.dw_band.split(","))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or args.operating_point
    json_path = RESULTS_DIR / f"convergence_lle_{tag}.json"
    npz_path = RESULTS_DIR / f"convergence_lle_{tag}_fields.npz"

    dev, kappa, d1, fsr_used, t_r = _device_and_frame()
    import tempfile
    from validation.pylle_crosscheck import derived_config
    work = Path(tempfile.mkdtemp(prefix="conv_lle_"))
    cfg = derived_config(fsr_used, work)

    spec = OPERATING_POINTS[args.operating_point]
    op = OperatingPoint(name=args.operating_point, config_path=cfg,
                        seed_kind="sech_lower_branch", **spec)

    print(f"operating point {op.name}: mu_half={op.mu_half} "
          f"({2 * op.mu_half + 1} modes), {op.n_roundtrips} round trips, "
          f"delta_omega {op.dw_start_over_kappa}k -> {op.dw_end_over_kappa}k")
    print(f"levels={levels}  grid_levels={glevels}@nsub{args.grid_nsub}  band={band}")

    try:
        result, fields, arrays = study(
            op, levels, grid_levels=glevels, grid_nsub=args.grid_nsub, band=band,
            max_seconds=args.max_seconds, argv=argv_full,
            check_regression=not args.no_regression_check)
    except Exception as exc:
        # Provenance must be written even on failure.
        seed = build_seed(2 * op.mu_half + 1, op.dw_start_over_kappa * kappa, dev,
                          kappa, D2_LOCAL_COMMITTED)
        prov = build_provenance(argv_full, op, dev, d1, fsr_used, t_r, seed,
                                {"dn_dT_per_k": 0.0, "fsr_hz": fsr_used}, 0.0)
        json_path.write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "study": "lle_substep_convergence",
             "provenance": prov, "status": "FAILED", "error": str(exc)}, indent=2))
        print(f"FAILED: {exc}", file=sys.stderr)
        print(f"wrote {json_path} (failure provenance)", file=sys.stderr)
        return 1

    reg = result["provenance"]["a1_regression"]
    if reg.get("checked") and not reg.get("passed"):
        json_path.write_text(json.dumps(result, indent=2))
        print(f"FAILED (A1): n_substeps=1 does not reproduce the committed field; "
              f"relative L2 = {reg['relative_l2']:.3e} > 1e-10", file=sys.stderr)
        return 1

    json_path.write_text(json.dumps(result, indent=2))
    np.savez_compressed(npz_path, **fields, **arrays)
    print(f"\nwrote {json_path}")
    print(f"wrote {npz_path}")

    print("\nconvergence summary")
    print(f"  {'observable':<18}{'p':>8}{'q*':>16}{'U_obs':>12}  status")
    for k, c in result["convergence"].items():
        p = "-" if c["observed_order"] is None else f"{c['observed_order']:.3f}"
        q = "-" if c["extrapolated"] is None else f"{c['extrapolated']:.6g}"
        u = "-" if c["U_obs"] is None else f"{c['U_obs'] * 100:.4f}%"
        print(f"  {k:<18}{p:>8}{q:>16}{u:>12}  {c['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
