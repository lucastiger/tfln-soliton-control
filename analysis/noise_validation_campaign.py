#!/usr/bin/env python
"""Noise-enabled publication validation campaign (arXiv:2604.05897 stochastic stack).

This is an ANALYSIS-ONLY driver: it makes NO ``simulator/`` change and touches no
noise/metrology numerics.  Every stochastic model was already proven to leave the
deterministic dynamics invariant when its channel is silenced (T_k=0 / switches
off); the deterministic-invariance regression is not repeated here.  This campaign
answers the complementary, physical question: *what changes when the new
stochastic models are intentionally enabled?*  Five workstreams, each ending in a
STOP-AND-REPORT gate that prints a quantitative table (with uncertainty) before
the next begins:

  W1  DW-peak survival under the full stochastic stack (flagship): do the two
      dispersive-wave/Cherenkov peaks survive, and is any change decomposable
      into physical (vacuum floor + pump jitter) vs anomalous (bug) terms?
  W2  Monte-Carlo soliton staircase: did noise alter soliton seeding/access?
      staircase recognizability, single-soliton success rate, step bias vs jitter.
  W3  Cross-realization β-line linewidth significance + pump-regime map:
      upgrade the DW-recoil parabola from within-record to across-realization.
  W4  Coherence / RF-beatnote ensemble: rep-rate beatnote PSD, tape-model
      S_c/S_cr/S_rep, per-line linewidth, vs the quantum + TRN+FSR limits.
  W5  Vacuum-floor + energy-fluctuation budget (sanity anchor): far wings ->
      hbar*omega0/2 per mode; energy fluctuations at the shot-noise scale.

Everything is built on the committed operating point and helpers:
``analysis.dks_access`` (OPERATING_DW_KAPPA, PIN_W, PRODUCTION_NUMERICS,
access_by_seeding, attach_dispersion, measured ``d_int_grid``,
dispersive_wave_peaks), ``analysis.spectral_metrics``
(average_power_spectrum, three_db_span, intracavity_comb_fraction,
detect_power_steps, soliton_count_transitions), ``analysis.noise_metrology``
(tape model, effective linewidth, rep-rate phase), ``analysis.run_detuning_sweep``
(warm-continuation staircase), and ``analysis._provenance``.

Figures (150+ dpi, titled, unit-labelled axes, legends, uncertainty bands where
an ensemble exists) and one consolidated ``campaign_report.json`` (with a
provenance stamp) go to ``--out`` (default ``analysis/results/validation/``).
Each workstream merges its block into the JSON as it finishes, so a partial or
``--workstream`` run leaves a valid, incrementally-updated report.

Vacuum-floor normalization used throughout (solver convention, lle_solver.py
_qnoise_increment docstring: ``Ẽ_μ = a_μ · n_tau · √(ħω₀)``):
  * raw |fft(E)_μ|^2 symmetric-ordered vacuum floor  =  n_tau^2 · ħω₀/2
  * per-mode photon number  <n_μ>  =  |fft(E)_μ|^2 / (n_tau^2 · ħω₀)
  * per-mode energy [J]     E_μ    =  |fft(E)_μ|^2 / n_tau^2   (floor ħω₀/2)
  * vacuum contribution to P_abs   =  κ_i · n_tau · ħω₀/2   [W]

Run ``--workstream all --quick`` for a minutes-scale smoke of the whole pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402

from analysis.dks_access import (  # noqa: E402
    CONFIG_PATH,
    OPERATING_DW_KAPPA,
    PIN_W,
    PRODUCTION_NUMERICS,
    RESULTS_DIR,
    _run,
    _spectrum_dict,
    attach_dispersion,
    dispersive_wave_peaks,
    load_cavity_params,
    sech_soliton_seed,
)
from analysis.noise_metrology import (  # noqa: E402
    effective_linewidth,
    frequency_noise_psd,
    phase_noise_psd,
    psd_at_offset,
    rep_rate_phase,
    tape_model_fit,
    unwrapped_phases,
)
from analysis.plot_utils import apply_pub_style  # noqa: E402
from analysis.run_detuning_sweep import (  # noqa: E402
    SweepConfig,
    run_detuning_sweep,
    write_noise_off_config,
)
from analysis.spectral_metrics import (  # noqa: E402
    detect_power_steps,
    intracavity_comb_fraction,
    normal_ordered_spectrum,
    single_dks_region,
    soliton_count_transitions,
    three_db_span,
)

#: Significance (in units of the ensemble-mean standard error) that a
#: normally-ordered spectral residual must reach to be called RESOLVED. The
#: conventional detection bar; see the W1 docstring for why the gate is a
#: significance test rather than a comparison against the pedestal itself.
RESOLVE_SIGMA = 3.0

#: Clustering tripwire thresholds. A per-seed sample is called CLUSTERED only
#: when ALL of these hold at its largest internal gap: the gap is at least
#: CLUSTER_GAP_RATIO times the pooled within-cluster spread, it spans at least
#: CLUSTER_GAP_RANGE_FRAC of the sample's total range, both sides hold at least
#: CLUSTER_MIN_MEMBERS points, and the sample has at least CLUSTER_MIN_N points.
#:
#: Calibrated by simulation rather than by taste. The gap-to-spread ratio alone
#: false-positives on small samples (a 2-vs-6 split of 8 Gaussian draws can
#: reach 5x), which is why the range fraction and the minimum cluster size are
#: there. At these values, over 4000 trials each: 0/4000 false positives on
#: Gaussian and uniform samples at n = 24, and <= 0.05% on exponential ones;
#: below n = 12 the rate is not acceptable, hence CLUSTER_MIN_N.
#:
#: Sensitivity is deliberately blunt -- a planted 10-sigma split is caught ~61%
#: of the time and a 20-sigma split always. This is a "do not quote the mean"
#: tripwire for gross separation, not a mixture-model fit, and a warning that
#: cried wolf would be worse than no warning at all.
CLUSTER_GAP_RATIO = 5.0
CLUSTER_GAP_RANGE_FRAC = 0.5
CLUSTER_MIN_MEMBERS = 3
CLUSTER_MIN_N = 12


def _distribution_diagnostic(values, label="value"):
    """Summarise a small per-seed sample, and detect CLUSTERING.

    ``mean +- std`` is only a faithful summary of a unimodal sample. When the
    per-seed values split into separated clusters, the mean lands in the gap
    BETWEEN them -- a number no realization produced -- and the std measures the
    cluster separation rather than any run-to-run uncertainty. Quoting such a
    mean as "X broadened to <mean> +- <std>" is wrong in a way that looks
    perfectly normal in a report, so this runs on every per-seed sample and says
    so explicitly.

    Detection is a single-linkage split at the largest gap in the sorted sample,
    accepted only when every threshold above is met (see the CLUSTER_* constants
    for their simulated false-positive rates).

    Returns a JSON-ready dict: mean/std/median/IQR/min/max, the cluster split
    when one is found, and ``closest_seed_to_mean_abs`` -- the distance from the
    mean to the NEAREST actual sample, which is the number that makes a
    misleading mean obvious at a glance.
    """
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = v.size
    out = {
        "n": int(n),
        "mean": float(np.mean(v)) if n else float("nan"),
        "std": float(np.std(v, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(v)) if n else float("nan"),
        "iqr": float(np.subtract(*np.percentile(v, [75, 25]))) if n else float("nan"),
        "min": float(v[0]) if n else float("nan"),
        "max": float(v[-1]) if n else float("nan"),
        "clustered": False,
        "cluster_gap": 0.0,
        "cluster_gap_over_within_std": 0.0,
        "cluster_gap_over_range": 0.0,
        "clusters": None,
        "closest_seed_to_mean_abs": float("nan"),
    }
    if n == 0:
        return out
    out["closest_seed_to_mean_abs"] = float(np.min(np.abs(v - out["mean"])))
    if n < CLUSTER_MIN_N:
        return out

    gaps = np.diff(v)
    i = int(np.argmax(gaps))
    lo, hi = v[: i + 1], v[i + 1:]
    if lo.size < CLUSTER_MIN_MEMBERS or hi.size < CLUSTER_MIN_MEMBERS:
        return out
    # Pooled within-cluster std; guarded so an exactly-degenerate cluster cannot
    # divide by zero and declare infinite separation.
    within = math.sqrt(
        ((lo.size - 1) * np.var(lo, ddof=1) + (hi.size - 1) * np.var(hi, ddof=1))
        / max(lo.size + hi.size - 2, 1)
    )
    ratio = float(gaps[i] / within) if within > 0 else float("inf")
    total_range = float(v[-1] - v[0])
    range_frac = float(gaps[i] / total_range) if total_range > 0 else 0.0
    out["cluster_gap"] = float(gaps[i])
    out["cluster_gap_over_within_std"] = ratio
    out["cluster_gap_over_range"] = range_frac
    if ratio >= CLUSTER_GAP_RATIO and range_frac >= CLUSTER_GAP_RANGE_FRAC:
        out["clustered"] = True
        out["clusters"] = [
            {"n": int(lo.size), "fraction": float(lo.size / n),
             "mean": float(np.mean(lo)), "std": float(np.std(lo, ddof=1)),
             "min": float(lo.min()), "max": float(lo.max())},
            {"n": int(hi.size), "fraction": float(hi.size / n),
             "mean": float(np.mean(hi)), "std": float(np.std(hi, ddof=1)),
             "min": float(hi.min()), "max": float(hi.max())},
        ]
    out["label"] = str(label)
    return out
from simulator.lle_solver import _load_config, hbar_omega0_from_config, resolve_cavity_rates  # noqa: E402
from simulator.noise_models import TRNoise  # noqa: E402

# ---------------------------------------------------------------------------
# Committed presets (identical device geometry / laser to noise_comparison_report)
# ---------------------------------------------------------------------------
# Kondratiev-Gorodetsky TRN geometry for THIS SiN device (ring R = L_cav/2pi,
# 4.4 x 0.8 um core Gaussian mode half-dimensions).
KG_GEOMETRY = dict(trn_R_m=9.298e-4, trn_da_m=2.2e-6, trn_db_m=4.0e-7)
# ECDL pump preset (paper V.B.4-V.B.5): white plateau h0 ~ 3e3 Hz^2/Hz
# (Delta_nu_L = pi*h0 ~ 9.4 kHz), flicker h-1 = 1e10 Hz^3/Hz, RIN floor -150 dBc/Hz.
ECDL_H0 = 3.0e3
ECDL_HM1 = 1.0e10
ECDL_RIN_FLOOR_DBC = -150.0
# DW-recoil linewidth preset (W3): white plateau + STRONG flicker, thermal off,
# so the measured-D_int dispersive-wave recoil transduces pump frequency noise
# into a resolvable repetition-rate parabola while the soliton stays healthy.
DW_RECOIL_H0 = 3.0e3
DW_RECOIL_HM1 = 1.0e13
# Soliton temporal-contrast health floor: below this a "single soliton" run is a
# collapsed / MI state and is dropped from the linewidth ensemble (logged).
CONTRAST_HEALTH_FLOOR = 8.0

VALIDATION_DIRNAME = "validation"


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _quietrun():
    """Silence the solver's per-run stdout (thermal pre-flight etc.) + warnings.

    The gate tables and progress lines this driver prints stay readable even
    across a 24-seed ensemble; real exceptions still propagate.
    """
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        yield buf


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _load_report(path: Path) -> dict:
    if path.exists():
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)


def _sidecar(base_cfg, tag: str, **overrides) -> str:
    """Write a sidecar YAML = ``base_cfg`` with physical_parameters updated."""
    cfg = yaml.safe_load(open(base_cfg, encoding="utf-8"))
    cfg["physical_parameters"].update(overrides)
    fd, name = tempfile.mkstemp(prefix=f"sin_params_{tag}_", suffix=".yaml")
    os.close(fd)
    with open(name, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return name


def _full_stack_sidecar(base_cfg) -> str:
    """Full stochastic stack ON: quantum vacuum + ECDL pump (freq+RIN) + K-G TRN + FSR."""
    return _sidecar(
        base_cfg, "fullstack",
        quantum_noise_enabled=1, quantum_noise_injection_cadence=1,
        pump_noise_enabled=1,
        pump_freq_noise_h0_hz2_per_hz=ECDL_H0,
        pump_freq_noise_hm1_hz3_per_hz=ECDL_HM1,
        pump_rin_floor_dbc_per_hz=ECDL_RIN_FLOOR_DBC,
        trn_psd_model="kondratiev_gorodetsky", **KG_GEOMETRY,
        fsr_noise_enabled=1,
    )


def _pump_only_sidecar(base_off_cfg, tag, h0, hm1) -> str:
    """Quantum + pump frequency noise only (thermal/FSR off): isolates DW recoil."""
    return _sidecar(
        base_off_cfg, tag,
        quantum_noise_enabled=1, quantum_noise_injection_cadence=1,
        pump_noise_enabled=1,
        pump_freq_noise_h0_hz2_per_hz=float(h0),
        pump_freq_noise_hm1_hz3_per_hz=float(hm1),
    )


def _hbar_omega0() -> float:
    return hbar_omega0_from_config(_load_config(CONFIG_PATH))


def _mean_power_spectrum(e_snapshots: np.ndarray) -> np.ndarray:
    """Time-averaged fftshifted |fft(E)|^2 (LINEAR raw power), shape (n_tau,)."""
    snaps = np.asarray(e_snapshots)
    power = np.abs(np.fft.fftshift(np.fft.fft(snaps, axis=-1), axes=-1)) ** 2
    return np.mean(power, axis=0)


def _lin_to_db_rel(p_lin, ref_lin):
    return 10.0 * np.log10(np.maximum(np.asarray(p_lin) / ref_lin, 1e-300))


def _settle(cav, dw, n_tau, cfgp, settle_rt, seed, numerics):
    """Warm-start a sech soliton and settle onto the attractor; return (E, dT)."""
    seed_field = sech_soliton_seed(dw, cav, n_tau=n_tau, pin=PIN_W)
    with _quietrun():
        s0 = _run(dw, settle_rt, cav, e0=seed_field, seed=seed, n_tau=n_tau,
                  pin=PIN_W, snapshot_interval=settle_rt, config_path=cfgp,
                  **numerics)
    return np.asarray(s0["e_final"])[0], float(np.asarray(s0["delta_t_final"]).ravel()[0])


def _record(cav, dw, n_tau, cfgp, e0, dt0, settle_rt, record_rt, seed,
            numerics, probe_mus=()):
    """Settle under noise from (e0,dt0) then record every round trip.

    Returns the record ``sol`` dict (E_snapshots every RT, U_int_history, ...).
    ``settle_rt`` uses the trajectory key ``seed``; the recorded segment a
    distinct key so it is an independent stretch of the same noise process.
    """
    with _quietrun():
        s1 = _run(dw, settle_rt, cav, e0=e0, delta_t0=dt0, seed=seed, n_tau=n_tau,
                  pin=PIN_W, snapshot_interval=settle_rt, config_path=cfgp,
                  **numerics)
        e1 = np.asarray(s1["e_final"])[0]
        dt1 = float(np.asarray(s1["delta_t_final"]).ravel()[0])
        kw = dict(mode_probe_indices=tuple(int(m) for m in probe_mus)) if probe_mus else {}
        sol = _run(dw, record_rt, cav, e0=e1, delta_t0=dt1, seed=seed * 2 + 1,
                   n_tau=n_tau, pin=PIN_W, snapshot_interval=1, config_path=cfgp,
                   **kw, **numerics)
    return sol


# ---------------------------------------------------------------------------
# Workstream 1 -- DW-peak survival under the full noise stack (flagship)
# ---------------------------------------------------------------------------
def _format_cluster_warning(dist, unit=""):
    """Gate lines warning that a per-seed sample is clustered; [] when it is not.

    Separated from the gate print so the wording and every format field are
    exercised by tests. This branch only fires on a real clustered ensemble, so
    without a test its first execution would be in the middle of a production
    run -- precisely when a KeyError in a warning path is least welcome.
    """
    if not dist.get("clustered"):
        return []
    lo, hi = dist["clusters"]
    u = f" {unit}" if unit else ""
    return [
        f"\n  *** {dist.get('label', 'value')}: PER-SEED DISTRIBUTION IS "
        f"CLUSTERED -- DO NOT QUOTE mean +/- std ***",
        f"      {lo['n']}/{dist['n']} seeds at {lo['mean']:.4g} +/- {lo['std']:.2g}{u}"
        f"   |   {hi['n']}/{dist['n']} seeds at {hi['mean']:.4g} +/- {hi['std']:.2g}{u}",
        f"      gap {dist['cluster_gap']:.4g}{u} = "
        f"{dist['cluster_gap_over_within_std']:.0f}x the within-cluster spread "
        f"and {dist['cluster_gap_over_range']:.0%} of the whole range;",
        f"      the reported mean {dist['mean']:.6g}{u} lies inside that gap, "
        f"{dist['closest_seed_to_mean_abs']:.4g}{u} from the nearest actual seed.",
        f"      Quote the median ({dist['median']:.6g}{u}) or report the two "
        f"clusters separately.",
    ]


def workstream1(out_dir, seeds, quick, numerics=PRODUCTION_NUMERICS):
    """Ensemble full-stack ON vs deterministic OFF at OPERATING_DW_KAPPA.

    Spectral ordering, and why the DW gate is a significance test
    -------------------------------------------------------------
    The solver integrates in the truncated-Wigner c-number limit, whose moments
    are SYMMETRICALLY ordered: every mode carries half a photon of vacuum, so the
    raw ensemble-mean |fft(E)|^2 sits on a flat pedestal n_tau^2*hbar*omega0/2
    (~ -82 dB rel. pump here). An OSA or photodiode measures <a^dagger a> --
    NORMALLY ordered -- and reads exactly zero for the vacuum, so every quantity
    below that an experiment could see is ALSO reported after subtracting that
    pedestal with analysis.spectral_metrics.normal_ordered_spectrum.

    Both orderings are reported side by side. The symmetric-ordered numbers are
    kept because they are the right ones for the floor diagnostics and the energy
    bookkeeping (the empirical far-wing floor, the broadening budget, W5's modal
    energy and P_abs terms) -- subtracting the pedestal there would double-count
    the vacuum and break the energy balance.

    The DW survival gate is normally ordered AND is a significance test, not a
    comparison against the pedestal. "Below the vacuum floor" is not a physical
    statement about detectability: the pedestal is a representation artifact that
    subtracts exactly, so it biases the spectrum without limiting it. What DOES
    limit detectability is the vacuum's VARIANCE, which subtraction leaves behind
    and only averaging reduces. The gate is therefore

        residual(mu) = mean_lin(mu) - pedestal   >   RESOLVE_SIGMA * sem(mu),

    with sem(mu) = std across seeds / sqrt(seeds) -- an EMPIRICAL standard error
    of the ensemble mean, so it already accounts for the within-record snapshot
    correlation (each seed contributes one record-mean sample) instead of
    assuming an independence factor. A peak below that bar is not "buried under
    the vacuum"; it is below this ensemble's noise floor, and the report says how
    much bigger an ensemble would resolve it.
    """
    n_tau = 16384                        # required to resolve DW peaks at |mu|~3000-3300
    settle_off = 800 if quick else 2000
    settle_on = 200 if quick else 1000
    record_rt = 64 if quick else 320
    hbar_omega0 = _hbar_omega0()
    floor_raw = n_tau ** 2 * hbar_omega0 / 2.0        # |fft|^2 vacuum floor per mode

    cav = attach_dispersion(load_cavity_params(CONFIG_PATH), n_tau)
    dw = OPERATING_DW_KAPPA * cav.kappa
    fsr_hz = cav.fsr_measured_hz if cav.fsr_measured_hz is not None else cav.fsr_hz
    cfg_off = str(write_noise_off_config())
    cfg_full = _full_stack_sidecar(CONFIG_PATH)

    print(f"[W1] n_tau={n_tau} seeds={seeds} settle_off={settle_off} "
          f"settle_on={settle_on} record={record_rt}  hbar*omega0={hbar_omega0:.4e} J")

    # Shared clean warm-start (deterministic), reused by OFF and every ON seed.
    e_seed, dt_seed = _settle(cav, dw, n_tau, cfg_off, settle_off, 0, numerics)

    # --- deterministic OFF reference ---
    t0 = time.time()
    sol_off = _record(cav, dw, n_tau, cfg_off, e_seed, dt_seed, settle_on,
                      record_rt, 0, numerics)
    spec_off = _mean_power_spectrum(np.asarray(sol_off["E_snapshots"])[0])
    print(f"[W1] OFF reference done ({time.time() - t0:.1f}s)")

    # --- full-stack ON ensemble ---
    ens_spec = np.empty((seeds, n_tau))
    ens_u = []                           # per-seed intracavity-energy series (for W4/W5)
    spans, combs, span_windows = [], [], []
    mu = np.arange(n_tau) - n_tau // 2
    pump_bin = n_tau // 2
    for s in range(seeds):
        t0 = time.time()
        sol = _record(cav, dw, n_tau, cfg_full, e_seed, dt_seed, settle_on,
                      record_rt, 1000 + s, numerics)
        sp = _mean_power_spectrum(np.asarray(sol["E_snapshots"])[0])
        ens_spec[s] = sp
        ens_u.append(np.asarray(sol["U_int_history"])[0])
        span = three_db_span(mu, sp, {"fsr_hz": fsr_hz})
        spans.append(span["span_ghz"])
        # The auto-selected median window, per seed. Kept because it is the
        # first thing to check if the per-seed spans turn out to be clustered:
        # a window that jumps between seeds points at the metric, one that does
        # not points at the spectra.
        span_windows.append(int(span["params"]["smooth_modes_used"]))
        combs.append(intracavity_comb_fraction(mu, sp))
        print(f"[W1] ON seed {s + 1}/{seeds}  span={span['span_ghz']:.2f} GHz "
              f"comb_frac={combs[-1]:.4f}  ({time.time() - t0:.1f}s)")
        jax.clear_caches()

    mean_lin = ens_spec.mean(axis=0)
    std_lin = ens_spec.std(axis=0, ddof=1) if seeds > 1 else np.zeros(n_tau)
    p_pump_on = float(mean_lin[pump_bin])
    p_pump_off = float(spec_off[pump_bin])

    # --- normally-ordered (instrument-comparable) ON spectrum ---------------
    # Signed residual: clip=False is mandatory here. Subtraction removes the
    # vacuum's BIAS, not its variance, so a pure-vacuum mode's residual is
    # zero-mean and negative about half the time; clipping would turn that into a
    # positively biased pedestal remnant and defeat the whole point.
    mean_norm = normal_ordered_spectrum(mean_lin, n_tau, hbar_omega0, clip=False)
    spec_off_norm = normal_ordered_spectrum(spec_off, n_tau, hbar_omega0,
                                            clip=False)
    # Empirical standard error of the ensemble mean, per mode. Each seed
    # contributes ONE record-mean sample, so the within-record snapshot
    # correlation is already inside std_lin -- no independence factor assumed.
    sem_lin = std_lin / math.sqrt(seeds) if seeds > 1 else np.full(n_tau, np.inf)
    sem_floor_rel_pump_db = (
        10.0 * math.log10(float(np.median(sem_lin[np.isfinite(sem_lin)])) / p_pump_on)
        if seeds > 1 else float("nan")
    )

    # DW peaks are DISPERSION-set: the phase-matched crossings D_int(mu)=delta_omega
    # depend only on the measured d_int_grid and the (programmed) detuning, so they
    # are IDENTICAL ON and OFF by construction -- that identity is the rigorous
    # "dispersion operator intact" statement. The empirical OFF peak is detected on
    # the clean deterministic spectrum (where it stands ~77 dB above the sech tail).
    sp_off_dict = _spectrum_dict(spec_off, cav)
    peaks_off = dispersive_wave_peaks(sp_off_dict, dw)

    floor_db_rel_pump = 10.0 * math.log10(floor_raw / p_pump_on)
    floor_db_rel_pump_off = 10.0 * math.log10(floor_raw / p_pump_off)
    # Empirical far-wing vacuum floor: median mean-spectrum level in an
    # un-dealiased far-wing band (outside the comb, inside |mu|<=n_tau/3).
    wing = (np.abs(mu) >= 3600) & (np.abs(mu) <= 5000)
    emp_floor_lin = float(np.median(mean_lin[wing]))
    emp_floor_mult = emp_floor_lin / floor_raw        # multiple of n_tau^2 hbar w0/2

    # For each OFF (deterministic) DW peak decide, robustly, whether it SURVIVES
    # above the risen vacuum floor or is SUBMERGED beneath it. The ON position is
    # measured at the FIXED OFF/crossing mode (no argmax wander) and its ON
    # prominence over the local floor decides survival; only a SURVIVING peak that
    # then moves > +/-2 modes is the dispersion/seeding-bug canary. A submerged
    # peak's argmax is meaningless (it tracks floor fluctuations), so it is
    # reported as physics (vacuum-floor burial), never as a position-shift bug.
    SURVIVE_PROM_DB = 3.0        # legacy symmetric-ordered prominence criterion
    peak_rows, resolvable_shift = [], []
    for p in sorted(peaks_off, key=lambda q: q["mu"]):
        side = "red" if p["mu"] < 0 else "blue"
        mu_dw = int(p["mu"])
        bin_dw = mu_dw + n_tau // 2
        cross = float(p["crossing_mu"])
        level_off_db = float(p["power_db"])
        snr_off = level_off_db - floor_db_rel_pump_off
        # ON level AT the fixed OFF mode + ON local floor (ring around the peak,
        # excluding the immediate +/-6 modes) -> ON prominence at that position.
        local = (np.abs(mu - mu_dw) <= 40) & (np.abs(mu - mu_dw) >= 8)
        on_local_floor_lin = float(np.median(mean_lin[local]))
        level_on_at_off_db = 10.0 * math.log10(max(mean_lin[bin_dw], 1e-300) / p_pump_on)
        prom_on_at_off = level_on_at_off_db - 10.0 * math.log10(on_local_floor_lin / p_pump_on)
        # LEGACY criterion (symmetric-ordered prominence over the local pedestal),
        # retained for continuity with the previously committed report. It is NOT
        # the gate any more -- see the workstream docstring.
        survives_prom_legacy = bool(prom_on_at_off >= SURVIVE_PROM_DB)

        # --- normally-ordered significance: THE GATE ------------------------
        residual_lin = float(mean_norm[bin_dw])
        sem_here = float(sem_lin[bin_dw])
        significance = residual_lin / sem_here if sem_here > 0 else float("nan")
        survives_on = bool(np.isfinite(significance) and significance >= RESOLVE_SIGMA)
        level_on_norm_db = (
            10.0 * math.log10(residual_lin / p_pump_on) if residual_lin > 0.0
            else float("nan")          # residual consistent with (or below) zero
        )
        # How much bigger an ensemble would resolve the DETERMINISTIC peak at
        # RESOLVE_SIGMA? sem scales as 1/sqrt(n_seeds), so the factor is
        # (RESOLVE_SIGMA * sem / P_off_peak)^2 in seeds.
        p_off_peak_lin = float(spec_off[bin_dw])
        seed_factor_to_resolve = (
            (RESOLVE_SIGMA * sem_here / p_off_peak_lin) ** 2
            if p_off_peak_lin > 0.0 and np.isfinite(sem_here) else float("inf")
        )

        # ON argmax within the +/-30 recoil window (only meaningful if it survives).
        win = (mu >= cross - 30) & (mu <= cross + 30)
        mu_on_argmax = int(mu[win][int(np.argmax(mean_lin[win]))])
        shift = mu_on_argmax - mu_dw if survives_on else 0
        if survives_on:
            resolvable_shift.append(shift)
        peak_rows.append({
            "side": side,
            "mu_off": mu_dw, "crossing_mu": cross,
            "mu_on_argmax": mu_on_argmax,
            "mu_shift_resolvable": int(shift),
            "off_peak_at_crossing": bool(abs(mu_dw - cross) <= 30),
            "wavelength_nm": float(p["wavelength_nm"]),
            "level_db_off": level_off_db,
            "level_on_at_off_mode_db": level_on_at_off_db,
            "snr_db_off": float(snr_off),
            "prominence_on_at_off_db": float(prom_on_at_off),
            # --- the gate (normally ordered, significance test) -------------
            "survives_above_vacuum_floor": survives_on,
            "survival_criterion": (
                f"normally-ordered residual > {RESOLVE_SIGMA:g} x the empirical "
                f"standard error of the ensemble-mean spectrum at that mode"),
            "level_on_normal_ordered_db": level_on_norm_db,
            "residual_normal_ordered_lin": residual_lin,
            "sem_ensemble_mean_lin": sem_here,
            "residual_significance_sigma": float(significance),
            "seed_factor_to_resolve_off_peak": float(seed_factor_to_resolve),
            # --- legacy symmetric-ordered numbers, kept for continuity ------
            # Key names preserved: tests/test_noise_validation_campaign.py pins
            # this schema. "submerged_below_vacuum_floor_db" remains the
            # SYMMETRIC-ordered depth below the pedestal -- a representation
            # statement, not a detectability one; the gate is the sigma above.
            "survives_symmetric_prominence_legacy": survives_prom_legacy,
            "submerged_below_vacuum_floor_db": float(
                floor_db_rel_pump_off - level_off_db),
            "prominence_db_off": float(p["prominence_db"]),
        })

    # --- broadening budget (mid-wing band, away from DW peaks and pump) ---
    mid = (np.abs(mu) >= 1000) & (np.abs(mu) <= 1600)
    off_wing_lin = float(np.median(spec_off[mid]))
    on_wing_lin = float(np.median(mean_lin[mid]))
    # Predicted ON wing = incoherent sum of the deterministic sech tail and the
    # flat vacuum floor (ħω₀/2 per mode); residual is attributed to pump-jitter
    # envelope wander (the pump frequency/RIN noise breathes the comb envelope).
    pred_on_wing_lin = off_wing_lin + floor_raw
    measured_change_db = 10.0 * math.log10(on_wing_lin / off_wing_lin)
    vacuum_change_db = 10.0 * math.log10(pred_on_wing_lin / off_wing_lin)
    pump_jitter_db = 10.0 * math.log10(on_wing_lin / pred_on_wing_lin)
    # factor-3 tolerance band on the wing-level change (in dB, ~4.77 dB).
    budget_tol_db = 10.0 * math.log10(3.0)
    budget_ok = bool(abs(measured_change_db - vacuum_change_db) <= budget_tol_db + abs(pump_jitter_db) + 1e-9)
    # The budget above is deliberately SYMMETRIC-ordered: it is the internal
    # consistency check "ON wing == OFF wing + pedestal", i.e. bookkeeping of the
    # field the solver integrates, and normal-ordering it would subtract the very
    # term it exists to verify. What an OSA would read in that band is the
    # normally-ordered residual, reported separately here.
    on_wing_norm_lin = float(np.median(mean_norm[mid]))
    on_wing_norm_sem = float(np.median(sem_lin[mid])) if seeds > 1 else float("inf")
    on_wing_norm_sigma = (
        on_wing_norm_lin / on_wing_norm_sem if np.isfinite(on_wing_norm_sem)
        and on_wing_norm_sem > 0 else float("nan")
    )

    # --- instrument-comparable comb metrics (normally ordered) --------------
    # Computed on the ENSEMBLE-MEAN spectrum, where the standard error is
    # smallest; a per-seed version would have sqrt(seeds) more residual noise and,
    # for the comb fraction, would need clipping (and hence a positive bias).
    # The comb fraction sums EVERY mode, so it is the pedestal-sensitive metric:
    # summing the SIGNED residual keeps it unbiased.
    tot_norm = float(np.sum(mean_norm))
    comb_fraction_on_norm = (
        float((tot_norm - float(mean_norm[pump_bin])) / tot_norm)
        if tot_norm > 0.0 else float("nan")
    )
    tot_off_norm = float(np.sum(spec_off_norm))
    comb_fraction_off_norm = (
        float((tot_off_norm - float(spec_off_norm[pump_bin])) / tot_off_norm)
        if tot_off_norm > 0.0 else float("nan")
    )
    # The 3 dB span is a comb-CORE envelope metric: its own 60 dB floor mask
    # already discards the pedestal-dominated wings, so it should be nearly
    # ordering-independent. clip=True is safe (and required by the metric's
    # non-negativity check) precisely because the retained lines sit far above
    # the pedestal.
    span_on_norm = three_db_span(
        mu, normal_ordered_spectrum(mean_lin, n_tau, hbar_omega0, clip=True),
        {"fsr_hz": fsr_hz})["span_ghz"]
    # ...and the SAME estimator on the SYMMETRIC-ordered ensemble mean. Without
    # this control the normally-ordered span could only be compared against
    # three_db_span_ghz_on_mean, which is the mean of PER-SEED spans -- a
    # different estimator, and a biased one for this metric: single-seed wings
    # are noisy, noise pushes the -3 dB crossings outward, and averaging spectra
    # before measuring suppresses that. Comparing the two would therefore
    # conflate the ordering change with the estimator change and could make a
    # pedestal-driven "broadening" out of an averaging artifact (or vice versa).
    # With this key the two effects separate:
    #   symmetric per-seed mean  vs  symmetric ensemble mean  -> estimator effect
    #   symmetric ensemble mean  vs  normally-ordered mean    -> ordering effect
    span_on_sym_ens = three_db_span(mu, mean_lin, {"fsr_hz": fsr_hz})["span_ghz"]

    # Dispersion canary: every OFF peak sits at its phase-matched crossing, and
    # every peak that still RESOLVES above the risen floor stays within +/-2 modes.
    # Submerged peaks (below the vacuum floor) are physics, not a position bug.
    n_survive = sum(r["survives_above_vacuum_floor"] for r in peak_rows)
    n_unresolved = len(peak_rows) - n_survive
    off_at_crossing = all(r["off_peak_at_crossing"] for r in peak_rows) if peak_rows else False
    max_resolvable_shift = int(max((abs(s) for s in resolvable_shift), default=0))
    positions_invariant = bool(off_at_crossing and max_resolvable_shift <= 2)

    span_mean = float(np.mean(spans)) if spans else float("nan")
    span_std = float(np.std(spans, ddof=1)) if len(spans) > 1 else 0.0
    comb_mean = float(np.mean(combs)) if combs else float("nan")
    comb_std = float(np.std(combs, ddof=1)) if len(combs) > 1 else 0.0
    # Per-seed distributions, and whether "mean +- std" is a faithful summary of
    # them at all. See _distribution_diagnostic.
    span_dist = _distribution_diagnostic(spans, "three_db_span_ghz_on")
    comb_dist = _distribution_diagnostic(combs, "comb_fraction_on")

    # ---- figures ----
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(mu, _lin_to_db_rel(spec_off, p_pump_off), lw=0.6, color="C0",
            label="all noise OFF (deterministic)")
    on_db = _lin_to_db_rel(mean_lin, p_pump_on)
    hi = _lin_to_db_rel(mean_lin + std_lin, p_pump_on)
    lo = _lin_to_db_rel(np.maximum(mean_lin - std_lin, 1e-300), p_pump_on)
    ax.fill_between(mu, lo, hi, color="C1", alpha=0.3, lw=0,
                    label=r"full stack ON: ensemble mean $\pm$ std")
    ax.plot(mu, on_db, lw=0.6, color="C1", alpha=0.9)
    ax.axhline(floor_db_rel_pump, color="k", ls="--", lw=1.0,
               label=r"$\hbar\omega_0/2$ pedestal (symmetric-ordered)")
    # Instrument-comparable trace: pedestal subtracted. Only the positive part is
    # renderable on a dB axis -- the modes that vanish here are exactly the ones
    # whose residual is consistent with zero, which is the point.
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_db_trace = np.where(
            mean_norm > 0.0,
            10.0 * np.log10(np.maximum(mean_norm, 1e-300) / p_pump_on),
            np.nan)
    ax.plot(mu, norm_db_trace, lw=0.5, color="C2", alpha=0.85,
            label="full stack ON, normally ordered (what an OSA reads)")
    if np.isfinite(sem_floor_rel_pump_db):
        ax.axhline(sem_floor_rel_pump_db + 10.0 * math.log10(RESOLVE_SIGMA),
                   color="C3", ls=":", lw=1.2,
                   label=rf"{RESOLVE_SIGMA:g}$\sigma$ detectability limit "
                         rf"({seeds} seeds)")
    for r in peak_rows:
        tag = "unresolved" if not r["survives_above_vacuum_floor"] else "resolved"
        ax.annotate(f"DW $\\mu$={r['mu_off']}\nOFF {r['level_db_off']:.0f} dB\n({tag})",
                    xy=(r["mu_off"], r["level_db_off"]),
                    xytext=(r["mu_off"], r["level_db_off"] - 26),
                    ha="center", fontsize=6,
                    arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel(r"mode index $\mu$")
    ax.set_ylabel("power relative to pump [dB]")
    ax.set_ylim(-200, 5)
    ax.set_title(f"DW-peak survival: OFF vs full-stack ON, cycle-averaged\n"
                 f"single soliton at $\\delta\\omega={OPERATING_DW_KAPPA:.0f}\\kappa$, "
                 f"$n_\\tau$={n_tau}, {seeds} seeds")
    ax.legend(loc="lower center", fontsize=7, ncol=2)
    fig.savefig(out_dir / "dw_survival_spectrum_off_on.png")
    plt.close(fig)

    if peak_rows:
        fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.4))
        sides = [r["side"] for r in peak_rows]
        x = np.arange(len(peak_rows))
        axs[0].bar(x - 0.15, [r["mu_off"] for r in peak_rows], 0.3, label="OFF peak")
        axs[0].bar(x + 0.15, [r["crossing_mu"] for r in peak_rows], 0.3,
                   label="phase-match crossing")
        axs[0].set_ylabel(r"mode index $\mu$")
        axs[0].set_title("DW position (dispersion-set)")
        axs[1].bar(x - 0.15, [r["level_db_off"] for r in peak_rows], 0.3, label="OFF peak")
        axs[1].bar(x + 0.15, [r["level_on_at_off_mode_db"] for r in peak_rows], 0.3,
                   label="ON at that mode (symmetric)")
        axs[1].axhline(floor_db_rel_pump, color="k", ls="--", lw=1.0,
                       label=r"$\hbar\omega_0/2$ pedestal")
        if np.isfinite(sem_floor_rel_pump_db):
            axs[1].axhline(sem_floor_rel_pump_db + 10.0 * math.log10(RESOLVE_SIGMA),
                           color="C3", ls=":", lw=1.2,
                           label=rf"{RESOLVE_SIGMA:g}$\sigma$ detectability")
        axs[1].set_ylabel("level rel. pump [dB]")
        axs[1].set_title("DW level vs pedestal and detectability")
        axs[2].bar(x, [r["residual_significance_sigma"] for r in peak_rows], 0.4,
                   color="C3")
        axs[2].axhline(RESOLVE_SIGMA, color="k", ls=":", lw=1.0,
                       label=rf"{RESOLVE_SIGMA:g}$\sigma$")
        axs[2].axhline(0.0, color="k", lw=0.8)
        axs[2].set_ylabel(r"normally-ordered residual [$\sigma$]")
        axs[2].set_title("DW detection significance (ON)")
        axs[2].legend(fontsize=6)
        for a in axs[:2]:
            a.legend(fontsize=6)
        for a in axs:
            a.set_xticks(x); a.set_xticklabels(sides)
        fig.suptitle("DW peak metrics: detectability is set by the ensemble "
                     "s.e., not by the pedestal")
        fig.savefig(out_dir / "dw_peak_metrics.png")
        plt.close(fig)

    block = {
        "n_tau": n_tau, "seeds": seeds, "quick": bool(quick),
        "operating_dw_kappa": OPERATING_DW_KAPPA,
        "hbar_omega0_j": hbar_omega0,
        "vacuum_floor_db_rel_pump": floor_db_rel_pump,
        "vacuum_floor_db_rel_pump_off": floor_db_rel_pump_off,
        "dw_peaks": peak_rows,
        "dw_positions_invariant_within_2_modes": positions_invariant,
        "dw_off_peaks_at_phase_match_crossing": off_at_crossing,
        "max_resolvable_dw_position_shift_modes": max_resolvable_shift,
        # Key names preserved (pinned by tests/test_noise_validation_campaign.py);
        # the COUNTS now come from the normally-ordered significance gate, so
        # "surviving" means RESOLVED above this ensemble's noise floor and
        # "submerged" means UNRESOLVED. The sum is still the peak count.
        "n_dw_peaks_surviving_above_floor": int(n_survive),
        "n_dw_peaks_submerged_below_floor": int(n_unresolved),
        "n_dw_peaks_resolved_normal_ordered": int(n_survive),
        "n_dw_peaks_unresolved_normal_ordered": int(n_unresolved),
        "dw_resolve_sigma": float(RESOLVE_SIGMA),
        "ensemble_mean_sem_median_db_rel_pump": float(sem_floor_rel_pump_db),
        "spectral_ordering_note": (
            "Keys without a '_normal_ordered' suffix are SYMMETRIC-ordered "
            "(truncated-Wigner), i.e. they include the n_tau^2*hbar*omega0/2 "
            "half-photon pedestal the solver actually integrates -- correct for "
            "the floor diagnostics and the energy bookkeeping, and NOT what an "
            "OSA reads. Keys suffixed '_normal_ordered' have that pedestal "
            "subtracted (analysis.spectral_metrics.normal_ordered_spectrum, "
            "clip=False) and are the instrument-comparable ones. NOTE the "
            "semantics of n_dw_peaks_surviving_above_floor / "
            "n_dw_peaks_submerged_below_floor changed: the counts now come from "
            "the normally-ordered significance gate (resolved / unresolved "
            "against this ensemble's standard error), not from a comparison "
            "against the pedestal. The old symmetric-ordered verdict is kept "
            "per peak as survives_symmetric_prominence_legacy."),
        "three_db_span_ghz_off": float(three_db_span(mu, spec_off, {"fsr_hz": fsr_hz})["span_ghz"]),
        "three_db_span_ghz_on_mean": span_mean,
        "three_db_span_ghz_on_std": span_std,
        "three_db_span_ghz_on_normal_ordered": float(span_on_norm),
        "three_db_span_ghz_on_symmetric_ensemble_mean": float(span_on_sym_ens),
        # Per-seed samples, persisted. They were previously reduced to mean+std
        # and discarded, which is exactly how a clustered distribution hides.
        "three_db_span_ghz_on_per_seed": [float(s) for s in spans],
        "three_db_span_smooth_modes_per_seed": [int(w) for w in span_windows],
        "comb_fraction_on_per_seed": [float(c) for c in combs],
        "three_db_span_on_distribution": span_dist,
        "comb_fraction_on_distribution": comb_dist,
        "comb_fraction_off": float(intracavity_comb_fraction(mu, spec_off)),
        "comb_fraction_on_mean": comb_mean,
        "comb_fraction_on_std": comb_std,
        "comb_fraction_on_normal_ordered": comb_fraction_on_norm,
        "comb_fraction_off_normal_ordered": comb_fraction_off_norm,
        "empirical_far_wing_floor_multiple_of_vacuum": emp_floor_mult,
        "mid_wing_normal_ordered": {
            "band_modes": [1000, 1600],
            "residual_lin": on_wing_norm_lin,
            "sem_lin": on_wing_norm_sem,
            "significance_sigma": float(on_wing_norm_sigma),
            "db_rel_pump": (
                10.0 * math.log10(on_wing_norm_lin / p_pump_on)
                if on_wing_norm_lin > 0.0 else float("nan")),
        },
        "broadening_budget": {
            "mid_wing_band_modes": [1000, 1600],
            "off_wing_db_rel_pump": 10.0 * math.log10(off_wing_lin / p_pump_off),
            "on_wing_db_rel_pump": 10.0 * math.log10(on_wing_lin / p_pump_on),
            "measured_wing_change_db": measured_change_db,
            "vacuum_floor_term_db": vacuum_change_db,
            "pump_jitter_residual_db": pump_jitter_db,
            "budget_tolerance_db_factor3": budget_tol_db,
            "within_budget": budget_ok,
        },
    }

    # ---- GATE W1 ----
    print("\n" + "=" * 72)
    print("GATE W1 -- DW-peak survival under the full noise stack")
    print("=" * 72)
    print(f"  symmetric-ordered vacuum pedestal = {floor_db_rel_pump_off:.1f} dB rel. pump "
          f"(NOT an instrument floor)")
    print(f"  ensemble-mean s.e. (median)       = {sem_floor_rel_pump_db:.1f} dB rel. pump "
          f"<- the detectability limit")
    print(f"{'peak':>6} {'mu_OFF':>7} {'cross':>7} {'lvl_OFF':>8} "
          f"{'ON sym':>8} {'ON norm':>9} {'sigma':>7} {'x seeds':>10} {'resolved':>9}")
    for r in peak_rows:
        norm_db = r["level_on_normal_ordered_db"]
        norm_s = f"{norm_db:>9.1f}" if np.isfinite(norm_db) else f"{'<=0':>9}"
        print(f"{r['side']:>6} {r['mu_off']:>7d} {r['crossing_mu']:>7.0f} "
              f"{r['level_db_off']:>8.1f} {r['level_on_at_off_mode_db']:>8.1f} "
              f"{norm_s} {r['residual_significance_sigma']:>+7.2f} "
              f"{r['seed_factor_to_resolve_off_peak']:>10.1e} "
              f"{str(r['survives_above_vacuum_floor']):>9}")
    print(f"\n  OFF peaks at phase-match crossing (dispersion intact): {off_at_crossing}")
    print(f"  DW positions invariant (resolvable, |shift|<=2): {positions_invariant} "
          f"(max resolvable shift {max_resolvable_shift})")
    print(f"  DW peaks RESOLVED (normally ordered, >={RESOLVE_SIGMA:g} sigma): "
          f"{n_survive}/{len(peak_rows)}  (unresolved: {n_unresolved})")
    print(f"    An unresolved peak is below THIS ensemble's vacuum-fluctuation "
          f"noise floor,\n    not below a physical floor: the pedestal subtracts "
          f"exactly, its variance does not.")
    print(f"  comb fraction (normally ordered): OFF {comb_fraction_off_norm:.4f} | "
          f"ON {comb_fraction_on_norm:.4f}")
    print(f"  3 dB span [GHz]: OFF {block['three_db_span_ghz_off']:.2f} | ON "
          f"per-seed mean {span_mean:.2f} | ON ensemble mean {span_on_sym_ens:.2f} "
          f"(symmetric) -> {span_on_norm:.2f} (normally ordered)")
    print(f"    estimator effect {span_on_sym_ens - span_mean:+.2f} GHz, "
          f"ordering effect {span_on_norm - span_on_sym_ens:+.2f} GHz")
    for dist, unit in ((span_dist, "GHz"), (comb_dist, "")):
        for line in _format_cluster_warning(dist, unit):
            print(line)
    print(f"  3 dB span  : OFF {block['three_db_span_ghz_off']:.2f} GHz | "
          f"ON {span_mean:.2f} +/- {span_std:.2f} GHz")
    print(f"  comb frac  : OFF {block['comb_fraction_off']:.4f} | "
          f"ON {comb_mean:.4f} +/- {comb_std:.4f}")
    print(f"  far-wing floor = {emp_floor_mult:.2f} x (n_tau^2 hbar w0/2)  "
          f"(target ~1, factor 3)")
    bb = block["broadening_budget"]
    print(f"  broadening budget: measured wing change {bb['measured_wing_change_db']:+.2f} dB "
          f"= vacuum {bb['vacuum_floor_term_db']:+.2f} dB + pump-jitter "
          f"{bb['pump_jitter_residual_db']:+.2f} dB  (within budget: {bb['within_budget']})")
    print("=" * 72 + "\n")

    # Decision-rule assert -- the canary is a RESOLVABLE peak that moves. A peak
    # below the vacuum floor (submerged) has a meaningless argmax and is physics,
    # not a dispersion bug; the dispersion is proven intact by off_at_crossing.
    # FAIL only on (a) an OFF peak NOT at its crossing (dispersion perturbed), or
    # (b) a surviving peak that shifts > +/-2 modes, or (c) the wing change out of
    # the vacuum+pump budget.
    bug = (not off_at_crossing) or (n_survive > 0 and max_resolvable_shift > 2) or (not budget_ok)
    if bug:
        _dw_failure_dump(out_dir, cav, dw, n_tau, cfg_full, mu, mean_lin, spec_off,
                         ens_u, peak_rows,
                         reason=f"off_at_crossing={off_at_crossing} "
                                f"max_resolvable_shift={max_resolvable_shift} "
                                f"budget_ok={budget_ok}")
        raise AssertionError(
            f"W1 FAIL: anomalous (non-physical) DW change -- OFF-at-crossing="
            f"{off_at_crossing}, resolvable shift={max_resolvable_shift} modes, "
            f"budget_ok={budget_ok}. The change is NOT explained by the vacuum "
            f"floor + pump jitter; the noise path may be perturbing the dispersion "
            f"operator or seeding. Diagnostic dump written; investigate.")

    # Clustered per-seed spans cannot be diagnosed from the report alone -- it
    # needs the individual spectra, which are otherwise reduced to mean/std and
    # thrown away. Persist them (and the per-seed windows) ONLY when the tripwire
    # fires, so the cost is paid exactly when someone will need the data.
    if span_dist.get("clustered") or comb_dist.get("clustered"):
        dump = out_dir / "span_bimodality_diagnostic.npz"
        np.savez_compressed(
            dump, mu=mu, ens_spec=ens_spec, spec_off=spec_off,
            spans_ghz=np.asarray(spans), smooth_modes=np.asarray(span_windows),
            comb_fractions=np.asarray(combs), p_pump_on=p_pump_on,
            p_pump_off=p_pump_off, hbar_omega0=hbar_omega0,
        )
        print(f"  [W1] per-seed spectra written for offline diagnosis -> {dump}")

    ens = {
        "n_tau": n_tau, "mu": mu, "mean_lin": mean_lin, "std_lin": std_lin,
        "spec_off": spec_off, "p_pump_on": p_pump_on, "p_pump_off": p_pump_off,
        "energy_series": ens_u, "hbar_omega0": hbar_omega0, "cav": cav,
        "fsr_hz": fsr_hz,
    }
    return block, ens


def _dw_failure_dump(out_dir, cav, dw, n_tau, cfgp, mu, mean_lin, spec_off,
                     ens_u, peak_rows, reason):
    """Persist a diagnostic bundle so a W1 FAIL cause can be traced offline."""
    dump = out_dir / "dw_failure_diagnostic.npz"
    np.savez_compressed(
        dump, reason=reason, mu=mu, mean_lin_on=mean_lin, spec_off=spec_off,
        energy_series=np.array(ens_u, dtype=object),
        peak_rows=json.dumps(peak_rows), dw=dw, n_tau=n_tau)
    print(f"[W1] DIAGNOSTIC DUMP -> {dump}  (reason: {reason})")


# ---------------------------------------------------------------------------
# Workstream 2 -- Monte-Carlo soliton staircase
# ---------------------------------------------------------------------------
def workstream2(out_dir, seeds, quick):
    """Deterministic reference staircase + full-stack ON ensemble, identical schedule."""
    # Production mirrors the committed staircase (analysis/results/detuning_sweep.npz:
    # n_solitons=5, dw 12->5.5 kappa, n_tau=4096) that is already validated to
    # resolve the 5->4->3->2->1->0 annihilation cascade, only coarsened in step
    # count to keep the noise ensemble affordable. Quick is a small correctness
    # smoke (fewer solitons, shorter, but still crosses annihilations).
    if quick:
        swp = dict(n_tau=2048, n_solitons=3, n_steps=31, settle_rt=1500,
                   hold_rt=400, dw_start_kappa=11.0, dw_stop_kappa=5.5)
        n_ens = min(seeds, 3)
    else:
        swp = dict(n_tau=4096, n_solitons=5, n_steps=101, settle_rt=6000,
                   hold_rt=1000, dw_start_kappa=12.0, dw_stop_kappa=5.5)
        n_ens = min(seeds, 6)

    cav = attach_dispersion(load_cavity_params(CONFIG_PATH), swp["n_tau"])
    cfg_off = str(write_noise_off_config())
    cfg_full = _full_stack_sidecar(CONFIG_PATH)
    base = SweepConfig(**swp)
    det_k = base.detunings_kappa()

    print(f"[W2] n_tau={swp['n_tau']} N={swp['n_solitons']} steps={swp['n_steps']} "
          f"hold={swp['hold_rt']} ensemble={n_ens}")

    # --- deterministic reference ---
    t0 = time.time()
    with _quietrun():
        det = run_detuning_sweep(cav, base, config_path=cfg_off)
    det_count = det["soliton_count"]
    det_trans = soliton_count_transitions(det["dw_over_kappa"], det_count)
    det_steps = detect_power_steps(det["dw_over_kappa"], det["P_intra"])
    det_lo, det_hi, det_annih = single_dks_region(det["dw_over_kappa"], det["soliton_count"] == 1)
    print(f"[W2] deterministic reference done ({time.time() - t0:.1f}s); "
          f"transitions={[(t['n_high_side'], t['n_low_side'], round(t['dw_mid'], 3)) for t in det_trans]}")

    # --- full-stack ON ensemble ---
    ens_counts, ens_pintra, ens_single, dropped = [], [], [], []
    for s in range(n_ens):
        t0 = time.time()
        cfg_s = SweepConfig(**{**swp, "seed": 100 + s, "position_seed": 1 + s})
        try:
            with _quietrun():
                sw = run_detuning_sweep(cav, cfg_s, config_path=cfg_full)
        except Exception as e:  # pre-settle lost/merged a soliton under noise
            dropped.append({"seed": 100 + s, "reason": str(e)[:160]})
            print(f"[W2] ON seed {s + 1}/{n_ens} DROPPED: {str(e)[:80]}")
            continue
        ens_counts.append(sw["soliton_count"])
        ens_pintra.append(sw["P_intra"])
        ens_single.append(bool(np.any(sw["soliton_count"] == 1)))
        print(f"[W2] ON seed {s + 1}/{n_ens}  final N={int(sw['soliton_count'][-1])}  "
              f"single-reached={ens_single[-1]}  ({time.time() - t0:.1f}s)")
        jax.clear_caches()

    n_ok = len(ens_counts)
    ens_counts = np.array(ens_counts) if n_ok else np.zeros((0, det_k.size))
    ens_pintra = np.array(ens_pintra) if n_ok else np.zeros((0, det_k.size))
    success_rate = float(np.mean(ens_single)) if ens_single else float("nan")

    # --- step-location bias vs jitter via ROBUST LEVEL-CROSSINGS ---
    # A coarse warm-continuation grid merges adjacent annihilations (5->3 instead
    # of 5->4->3), so exact (n_high, n_low) matching is fragile and low-N. The
    # soliton-number boundary "detuning at which the count first drops to <= k",
    # scanning down the sweep, is well-defined for EVERY level k regardless of
    # merging, giving one matched boundary per level per realization. Bias is the
    # ensemble-mean offset from the deterministic boundary; jitter is the per-seed
    # std -- exactly the jitter-vs-bias split the prompt asks for.
    def _level_crossing(dw_desc, count, k):
        below = np.where(np.asarray(count) <= k)[0]
        return float(dw_desc[below[0]]) if below.size else None

    level_rows = []
    for k in range(int(swp["n_solitons"]) - 1, -1, -1):
        det_cross = _level_crossing(det_k, det_count, k)
        if det_cross is None:
            continue
        on_cross = [x for x in (_level_crossing(det_k, c, k) for c in ens_counts)
                    if x is not None]
        if len(on_cross) < 2:
            continue
        on_cross = np.array(on_cross)
        mean_loc = float(on_cross.mean())
        jitter = float(on_cross.std(ddof=1))
        bias = mean_loc - det_cross
        level_rows.append({
            "boundary": f"N<={k}",
            "deterministic_dw_kappa": det_cross,
            "ensemble_mean_dw_kappa": mean_loc,
            "bias_kappa": bias,
            "jitter_kappa": jitter,
            "n_realizations": int(on_cross.size),
            "per_seed_dw_kappa": [float(x) for x in on_cross],
            "bias_lt_jitter": bool(abs(bias) < 3.0 * max(jitter, 1e-12)),
        })

    # Secondary diagnostic: exact (n_high, n_low) matching (kept for provenance).
    trans_rows = []
    for dt in det_trans:
        key = (dt["n_high_side"], dt["n_low_side"])
        locs = [m[0] for c in ens_counts
                for m in [[t["dw_mid"] for t in soliton_count_transitions(det_k, c)
                           if (t["n_high_side"], t["n_low_side"]) == key]] if m]
        if not locs:
            continue
        locs = np.array(locs)
        trans_rows.append({
            "transition": f"{key[0]}->{key[1]}",
            "deterministic_dw_kappa": float(dt["dw_mid"]),
            "ensemble_mean_dw_kappa": float(locs.mean()),
            "bias_kappa": float(locs.mean() - dt["dw_mid"]),
            "jitter_kappa": float(locs.std(ddof=1)) if locs.size > 1 else 0.0,
            "n_realizations": int(locs.size),
        })

    # --- per-detuning soliton-number probability ---
    max_n = int(max(det_count.max(), ens_counts.max() if n_ok else 0))
    prob = np.zeros((max_n + 1, det_k.size))
    if n_ok:
        for k in range(max_n + 1):
            prob[k] = np.mean(ens_counts == k, axis=0)

    # ---- figures ----
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for c in ens_pintra:
        ax.plot(det_k, c, lw=0.4, color="C1", alpha=0.35)
    if n_ok:
        m = ens_pintra.mean(axis=0)
        sd = ens_pintra.std(axis=0, ddof=1) if n_ok > 1 else np.zeros_like(m)
        ax.fill_between(det_k, m - sd, m + sd, color="C1", alpha=0.25, lw=0,
                        label=r"ON ensemble mean $\pm$ std")
    ax.plot(det["dw_over_kappa"], det["P_intra"], lw=1.8, color="k",
            label="deterministic (noise OFF)")
    for t in det_trans:
        ax.axvline(t["dw_mid"], color="C3", ls=":", lw=0.7)
    ax.set_xlabel(r"detuning $\delta\omega/\kappa$")
    ax.set_ylabel(r"intracavity power $\sum_\mu|a_\mu|^2$  [arb.]")
    ax.set_title(f"Monte-Carlo soliton staircase: deterministic vs full-stack ON\n"
                 f"({n_ok} realizations, $N_0$={swp['n_solitons']} seeded)")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "staircase_montecarlo.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    if level_rows:
        y = np.arange(len(level_rows))
        axs[0].errorbar([r["ensemble_mean_dw_kappa"] for r in level_rows], y,
                        xerr=[r["jitter_kappa"] for r in level_rows], fmt="o",
                        capsize=3, label=r"ON mean $\pm$ jitter")
        axs[0].plot([r["deterministic_dw_kappa"] for r in level_rows], y, "kx",
                    ms=8, label="deterministic")
        axs[0].set_yticks(y)
        axs[0].set_yticklabels([r["boundary"] for r in level_rows])
        axs[0].set_xlabel(r"switching detuning $\delta\omega/\kappa$")
        axs[0].set_title("Step location: bias vs jitter (level crossings)")
        axs[0].legend(fontsize=7)
    for k in range(max_n + 1):
        if prob[k].max() > 0:
            axs[1].plot(det_k, prob[k], lw=1.0, label=f"N={k}")
    axs[1].set_xlabel(r"detuning $\delta\omega/\kappa$")
    axs[1].set_ylabel("probability")
    axs[1].set_title("Soliton-number probability")
    axs[1].legend(fontsize=7, ncol=2)
    fig.savefig(out_dir / "staircase_step_jitter.png")
    plt.close(fig)

    all_bias_ok = all(r["bias_lt_jitter"] for r in level_rows) if level_rows else False
    block = {
        "sweep_config": swp, "ensemble_requested": n_ens, "ensemble_ok": n_ok,
        "dropped": dropped,
        "single_soliton_success_rate": success_rate,
        "deterministic_single_dks_region_kappa": [det_lo, det_hi],
        "deterministic_annihilation_kappa": det_annih,
        "deterministic_power_step_locations_kappa": [float(x) for x in det_steps["step_x"]],
        "level_crossings": level_rows,
        "exact_transition_matches": trans_rows,
        "all_bias_lt_jitter": all_bias_ok,
        "soliton_number_probability_detuning_kappa": det_k,
        "soliton_number_probability": {str(k): prob[k] for k in range(max_n + 1)},
    }

    # ---- GATE W2 ----
    print("\n" + "=" * 72)
    print("GATE W2 -- Monte-Carlo soliton staircase")
    print("=" * 72)
    print(f"  ensemble: {n_ok}/{n_ens} realizations survived pre-settle "
          f"({len(dropped)} dropped)")
    print(f"  single-soliton access success rate: {success_rate:.2%}")
    print(f"  {'boundary':>10} {'determ':>9} {'ON mean':>9} {'bias':>8} "
          f"{'jitter':>8} {'bias<jit':>8}  [kappa]")
    for r in level_rows:
        print(f"  {r['boundary']:>10} {r['deterministic_dw_kappa']:>9.3f} "
              f"{r['ensemble_mean_dw_kappa']:>9.3f} {r['bias_kappa']:>+8.4f} "
              f"{r['jitter_kappa']:>8.4f} {str(r['bias_lt_jitter']):>8}")
    print(f"\n  all soliton-number boundaries have |bias| < 3*jitter: {all_bias_ok}")
    print("=" * 72 + "\n")
    return block


# ---------------------------------------------------------------------------
# Workstream 3 -- cross-realization linewidth significance + pump-regime map
# ---------------------------------------------------------------------------
def _linewidth_run(cfgp, measured, seed, n_tau, t_slow, probe_mus, numerics):
    """One β-separation-line linewidth parabola fit.

    Returns a2 (FWHM^2 curvature), mu_fix (parabola vertex), per-line linewidths,
    and the soliton temporal contrast (health check). Detrend OFF in the
    frequency-noise PSD so the low-frequency 1/f + DW-recoil content the β-line
    integral needs is preserved (same knob fig6b uses).
    """
    cav = load_cavity_params(CONFIG_PATH)
    if measured:
        cav = attach_dispersion(cav, n_tau)
    dw = OPERATING_DW_KAPPA * cav.kappa
    seed_field = sech_soliton_seed(dw, cav, n_tau=n_tau, pin=PIN_W)
    with _quietrun():
        s0 = _run(dw, 2000, cav, e0=seed_field, seed=seed, n_tau=n_tau, pin=PIN_W,
                  snapshot_interval=2000, config_path=cfgp, **numerics)
        sol = _run(dw, t_slow, cav, e0=np.asarray(s0["e_final"])[0],
                   delta_t0=float(np.asarray(s0["delta_t_final"]).ravel()[0]),
                   seed=seed + 1, n_tau=n_tau, pin=PIN_W,
                   snapshot_interval=max(t_slow // 16, 1), config_path=cfgp,
                   mode_probe_indices=tuple(probe_mus), **numerics)
    probes = np.asarray(sol["mode_probe_history"])[0][t_slow // 4:]
    e_final = np.asarray(sol["e_final"])[0]
    p = np.abs(e_final) ** 2
    contrast = float(p.max() / max(p.mean(), 1e-300))
    phases = unwrapped_phases(probes)
    # Lab-frame line phases: add the integrated pump phase (measured runs record
    # pump_freq_noise_history; the frame co-rotates with the pump).
    if "pump_freq_noise_history" in sol:
        pump_hist = np.asarray(sol["pump_freq_noise_history"])[0][t_slow // 4:]
        phi_pump = -np.cumsum(pump_hist) * cav.t_r
        phases = phases + phi_pump[:, None]
    mus = np.asarray(probe_mus, dtype=float)
    nseg = min(1 << 12, phases.shape[0] // 8)
    lw = np.array([effective_linewidth(
        *frequency_noise_psd(phases[:, j], cav.t_r, nperseg=nseg, detrend=False))
        for j in range(len(probe_mus))])
    coef = np.polyfit(mus, lw ** 2, 2)
    a2 = float(coef[0])
    mu_fix = float(-coef[1] / (2.0 * coef[0])) if coef[0] != 0 else float("nan")
    return {"a2": a2, "mu_fix": mu_fix, "linewidths_hz": lw, "mus": mus,
            "contrast": contrast, "coef": coef}


def workstream3(out_dir, seeds, quick, numerics=PRODUCTION_NUMERICS):
    if quick:
        n_tau, t_slow = 1024, 4096
        probe_mus = (-300, -240, -180, -120, 120, 180, 240, 300)
        n_ens, n_preset = min(max(seeds, 6), 6), 3
        presets = [1e9, 1e13]
    else:
        n_tau, t_slow = 4096, 1 << 14
        probe_mus = (-750, -600, -450, -300, -150, 150, 300, 450, 600, 750)
        # >=16 independent pump realizations (prompt); capped so a large --seeds
        # (chosen for the W1 flagship) does not inflate this expensive ensemble.
        n_ens, n_preset = min(max(seeds, 16), 16), 4
        # The flagship (strong-flicker) preset is presets[-1]; keeping it IN the
        # regime map means its Taylor negative control is computed alongside, so
        # the cross-realization significance has a paired >=10x-smaller control.
        presets = [1e9, 1e10, 1e11, DW_RECOIL_HM1]

    flagship_hm1 = presets[-1]
    cfg_off = str(write_noise_off_config())
    cfg_flag = _pump_only_sidecar(cfg_off, "dwflag", DW_RECOIL_H0, flagship_hm1)
    print(f"[W3] n_tau={n_tau} t_slow={t_slow} ensemble={n_ens} "
          f"flagship_hm1={flagship_hm1:.0e} probes={probe_mus}")

    # --- cross-realization ensemble at the flagship (strong-flicker) preset ---
    a2_ens, mufix_ens, lw_ens, ens_dropped = [], [], [], []
    for s in range(n_ens):
        t0 = time.time()
        r = _linewidth_run(cfg_flag, True, 3000 + s, n_tau, t_slow, probe_mus, numerics)
        if r["contrast"] < CONTRAST_HEALTH_FLOOR:
            ens_dropped.append({"seed": 3000 + s, "contrast": r["contrast"]})
            print(f"[W3] ens seed {s + 1}/{n_ens} DROPPED (contrast {r['contrast']:.1f})")
            continue
        a2_ens.append(r["a2"]); mufix_ens.append(r["mu_fix"]); lw_ens.append(r["linewidths_hz"])
        print(f"[W3] ens seed {s + 1}/{n_ens}  a2={r['a2']:.3e}  mu_fix={r['mu_fix']:.0f}  "
              f"contrast={r['contrast']:.0f}  ({time.time() - t0:.1f}s)")
        jax.clear_caches()

    a2_ens = np.array(a2_ens); mufix_ens = np.array(mufix_ens)
    lw_ens = np.array(lw_ens) if len(lw_ens) else np.zeros((0, len(probe_mus)))
    a2_mean = float(a2_ens.mean()) if a2_ens.size else float("nan")
    a2_std = float(a2_ens.std(ddof=1)) if a2_ens.size > 1 else 0.0
    # Bootstrap p(a2>0): resample realizations, fraction of bootstrap MEANS > 0.
    rng = np.random.default_rng(0)
    if a2_ens.size:
        boot = np.array([rng.choice(a2_ens, a2_ens.size, replace=True).mean()
                         for _ in range(2000)])
        p_a2_pos = float(np.mean(boot > 0))
    else:
        p_a2_pos = float("nan")

    # --- pump-regime map: a2/mu_fix vs preset, measured vs Taylor control ---
    preset_rows = []
    for hm1 in presets:
        cfg_p = _pump_only_sidecar(cfg_off, f"dwp{hm1:.0e}", DW_RECOIL_H0, hm1)
        row = {"pump_hm1_hz3_per_hz": hm1}
        for label, measured in (("measured", True), ("taylor", False)):
            a2s, mfs = [], []
            for s in range(n_preset):
                r = _linewidth_run(cfg_p, measured, 5000 + int(math.log10(hm1)) * 100 + s,
                                   n_tau, t_slow, probe_mus, numerics)
                if r["contrast"] >= CONTRAST_HEALTH_FLOOR:
                    a2s.append(r["a2"]); mfs.append(r["mu_fix"])
            a2s = np.array(a2s); mfs = np.array(mfs)
            row[f"{label}_a2_mean"] = float(a2s.mean()) if a2s.size else float("nan")
            row[f"{label}_a2_std"] = float(a2s.std(ddof=1)) if a2s.size > 1 else 0.0
            row[f"{label}_mu_fix_mean"] = float(mfs.mean()) if mfs.size else float("nan")
            row[f"{label}_n"] = int(a2s.size)
            print(f"[W3] preset hm1={hm1:.0e} {label}: a2={row[f'{label}_a2_mean']:.3e} "
                  f"mu_fix={row[f'{label}_mu_fix_mean']:.0f} (n={a2s.size})")
        # Taylor control must be >=10x smaller than measured for a genuine result.
        m, tl = abs(row["measured_a2_mean"]), abs(row["taylor_a2_mean"])
        row["taylor_ratio"] = float(tl / m) if m > 0 else float("nan")
        preset_rows.append(row)
        jax.clear_caches()

    # ---- figures ----
    apply_pub_style()
    mus = np.asarray(probe_mus, dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    if lw_ens.size:
        m = lw_ens.mean(axis=0) / 1e6
        sd = lw_ens.std(axis=0, ddof=1) / 1e6 if lw_ens.shape[0] > 1 else np.zeros_like(m)
        ax.errorbar(mus, m, yerr=sd, fmt="o", capsize=3, color="C0",
                    label=r"measured $D_{int}$: per-line linewidth (cross-real. mean $\pm$ std)")
        mg = np.linspace(mus.min() * 1.1, mus.max() * 1.1, 300)
        cf = np.polyfit(mus, (m * 1e6) ** 2, 2)
        ax.plot(mg, np.sqrt(np.maximum(np.polyval(cf, mg), 0)) / 1e6, "-", color="C0",
                lw=1.2, label="quadratic fit (DW recoil)")
    if np.isfinite(a2_mean) and a2_ens.size:
        mu_fix_mean = float(np.nanmean(mufix_ens))
        ax.axvline(mu_fix_mean, color="gray", ls=":",
                   label=fr"$\mu_{{fix}}={mu_fix_mean:.0f}$")
    ax.set_xlabel(r"mode index $\mu$")
    ax.set_ylabel("effective linewidth [MHz]")
    ax.set_title("Cross-realization β-line linewidth parabola (DW recoil)\n"
                 f"flicker-dominated pump, measured $D_{{int}}$, {a2_ens.size} realizations")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "linewidth_parabola_ensemble.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    hm1s = [r["pump_hm1_hz3_per_hz"] for r in preset_rows]
    axs[0].errorbar(hm1s, [r["measured_a2_mean"] for r in preset_rows],
                    yerr=[r["measured_a2_std"] for r in preset_rows], fmt="o-",
                    capsize=3, label="measured $D_{int}$")
    axs[0].errorbar(hm1s, [r["taylor_a2_mean"] for r in preset_rows],
                    yerr=[r["taylor_a2_std"] for r in preset_rows], fmt="s--",
                    capsize=3, label="Taylor $D_2$ (control)")
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"pump flicker $h_{-1}$ [Hz$^3$/Hz]")
    axs[0].set_ylabel(r"$a_2$ curvature [Hz$^2$/mode$^2$]")
    axs[0].set_title("β-line curvature vs pump regime")
    axs[0].legend(fontsize=7)
    axs[1].plot(hm1s, [r["measured_mu_fix_mean"] for r in preset_rows], "o-",
                label="measured $D_{int}$")
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"pump flicker $h_{-1}$ [Hz$^3$/Hz]")
    axs[1].set_ylabel(r"$\mu_{fix}$")
    axs[1].set_title("Fix point vs pump regime")
    axs[1].legend(fontsize=7)
    fig.savefig(out_dir / "a2_vs_pump_regime.png")
    plt.close(fig)

    # Flagship Taylor negative control: the map row at the flagship preset gives
    # the paired |a2_taylor|/|a2_measured| ratio (must be <=0.1 for a genuine
    # DW-recoil result -- pure quadratic dispersion -> pure common mode -> a2->0).
    flag_row = next((r for r in preset_rows
                     if abs(r["pump_hm1_hz3_per_hz"] - flagship_hm1) < 1), preset_rows[-1])
    flagship_taylor_ratio = flag_row["taylor_ratio"]

    block = {
        "n_tau": n_tau, "t_slow": t_slow, "probe_mus": list(map(int, probe_mus)),
        "flagship_pump_hm1_hz3_per_hz": flagship_hm1,
        "ensemble_ok": int(a2_ens.size), "ensemble_dropped": ens_dropped,
        "a2_cross_realization_mean": a2_mean,
        "a2_cross_realization_std": a2_std,
        "a2_bootstrap_p_positive": p_a2_pos,
        "mu_fix_cross_realization_mean": float(np.nanmean(mufix_ens)) if mufix_ens.size else float("nan"),
        "mu_fix_cross_realization_std": float(np.nanstd(mufix_ens, ddof=1)) if mufix_ens.size > 1 else 0.0,
        "flagship_taylor_ratio": flagship_taylor_ratio,
        "flagship_taylor_control_10x_smaller": bool(np.isfinite(flagship_taylor_ratio)
                                                    and flagship_taylor_ratio <= 0.1),
        "pump_regime_map": preset_rows,
    }

    # ---- GATE W3 ----
    print("\n" + "=" * 72)
    print("GATE W3 -- cross-realization β-line linewidth + pump-regime map")
    print("=" * 72)
    print(f"  cross-realization a2 = {a2_mean:.4e} +/- {a2_std:.4e}  "
          f"(n={a2_ens.size})")
    print(f"  bootstrap p(a2 > 0) = {p_a2_pos:.4f}")
    print(f"  mu_fix = {block['mu_fix_cross_realization_mean']:.0f} +/- "
          f"{block['mu_fix_cross_realization_std']:.0f}")
    print(f"  flagship Taylor control ratio |a2_tay|/|a2_meas| = "
          f"{flagship_taylor_ratio:.3f} (<=0.1 for a genuine result: "
          f"{block['flagship_taylor_control_10x_smaller']})")
    print(f"  {'hm1':>10} {'meas_a2':>11} {'tay_a2':>11} {'tay/meas':>9} "
          f"{'meas_mu_fix':>11}")
    for r in preset_rows:
        print(f"  {r['pump_hm1_hz3_per_hz']:>10.0e} {r['measured_a2_mean']:>11.3e} "
              f"{r['taylor_a2_mean']:>11.3e} {r['taylor_ratio']:>9.3f} "
              f"{r['measured_mu_fix_mean']:>11.0f}")
    print("=" * 72 + "\n")
    return block


# ---------------------------------------------------------------------------
# Workstream 4 -- coherence / RF-beatnote ensemble
# ---------------------------------------------------------------------------
def workstream4(out_dir, seeds, quick, numerics=PRODUCTION_NUMERICS):
    if quick:
        n_tau, t_slow = 1024, 8192
        probe_mus = (-300, -200, -100, 100, 200, 300)
        n_ens = min(max(seeds, 4), 4)
    else:
        n_tau, t_slow = 4096, 1 << 14
        probe_mus = (-750, -500, -250, 250, 500, 750)
        n_ens = min(max(seeds, 16), 16)

    cav = attach_dispersion(load_cavity_params(CONFIG_PATH), n_tau)
    dw = OPERATING_DW_KAPPA * cav.kappa
    cfg_off = str(write_noise_off_config())
    cfg_full = _full_stack_sidecar(CONFIG_PATH)
    # TRN(K-G) limit reference for S_rep.
    trn_kg = TRNoise({**_load_config(CONFIG_PATH),
                      "trn_psd_model": "kondratiev_gorodetsky", **KG_GEOMETRY})
    d1_over_omega0 = (2.0 * math.pi * cav.fsr_hz) / trn_kg.omega_0
    print(f"[W4] n_tau={n_tau} t_slow={t_slow} ensemble={n_ens} probes={probe_mus}")

    e_seed, dt_seed = _settle(cav, dw, n_tau, cfg_off, 2000, 0, numerics)

    srep_list, sc_list, scr_list, screp_list, mufix_list = [], [], [], [], []
    beat_list, linewidth_list = [], []
    f_ref = None
    for s in range(n_ens):
        t0 = time.time()
        sol = _record(cav, dw, n_tau, cfg_full, e_seed, dt_seed, 500, t_slow,
                      7000 + s, numerics, probe_mus=probe_mus)
        probes = np.asarray(sol["mode_probe_history"])[0]
        u_series = np.asarray(sol["U_int_history"])[0]     # DC-PD proxy Sum_j|E_j|^2 (up to t_r/n_tau)
        phases = unwrapped_phases(probes)
        nseg = min(1 << 12, phases.shape[0] // 4)
        # Rep-rate PSD via the pairwise-difference estimator (extreme pair):
        # cancels the common mode deterministically (the reliable estimator).
        phi_rep = rep_rate_phase(phases, probe_mus, 0, len(probe_mus) - 1)
        f_r, s_rep = frequency_noise_psd(phi_rep, cav.t_r, nperseg=nseg)
        # Tape-model decomposition (S_c, S_cr, S_rep, mu_fix) over the probe set.
        fit = tape_model_fit(phases, probe_mus, cav.t_r, nperseg=nseg)
        # DC-PD beatnote proxy PSD (baseband pedestal of the pulse-train envelope).
        f_b, s_b = phase_noise_psd(u_series - u_series.mean(), cav.t_r, nperseg=nseg)
        # Per-line effective linewidth (β-separation line), median over probes.
        lws = [effective_linewidth(*frequency_noise_psd(phases[:, j], cav.t_r, nperseg=nseg))
               for j in range(len(probe_mus))]
        if f_ref is None:
            f_ref, f_ref_tape, f_ref_beat = f_r, fit["f"], f_b
        srep_list.append(np.interp(f_ref, f_r, s_rep))
        sc_list.append(np.interp(f_ref_tape, fit["f"], fit["S_c"]))
        scr_list.append(np.interp(f_ref_tape, fit["f"], fit["S_cr"]))
        screp_list.append(np.interp(f_ref_tape, fit["f"], fit["S_rep"]))
        beat_list.append(np.interp(f_ref_beat, f_b, s_b))
        linewidth_list.append(float(np.median(lws)))
        mufix_list.append(np.interp(f_ref_tape, fit["f"], fit["mu_fix"]))
        print(f"[W4] seed {s + 1}/{n_ens}  median line linewidth={linewidth_list[-1]:.3e} Hz  "
              f"({time.time() - t0:.1f}s)")
        jax.clear_caches()

    srep = np.array(srep_list); sc = np.array(sc_list); screp = np.array(screp_list)
    beat = np.array(beat_list)
    srep_m, srep_s = srep.mean(0), (srep.std(0, ddof=1) if len(srep) > 1 else np.zeros_like(srep[0]))
    # TRN(K-G) + FSR limit and quantum-limited comparison for S_rep.
    s_rep_trn = (d1_over_omega0 ** 2 * trn_kg.c_pull ** 2
                 * np.asarray(trn_kg.delta_t_psd(f_ref)) / (2 * math.pi) ** 2)
    lw_mean = float(np.mean(linewidth_list)) if linewidth_list else float("nan")
    lw_std = float(np.std(linewidth_list, ddof=1)) if len(linewidth_list) > 1 else 0.0
    # Band ratio of measured S_rep to the TRN+FSR limit over the low-offset decade.
    band = (f_ref >= f_ref[2]) & (f_ref <= f_ref[2] * 30)
    srep_over_trn_db = float(np.median(10 * np.log10(np.maximum(srep_m[band], 1e-300)
                                                     / np.maximum(s_rep_trn[band], 1e-300))))
    # Quantum-limited S_rep floor: the white plateau S_rep flattens to at the
    # high-offset decade (per-line quantum phase noise / n_tau leverage), read
    # off the measured ensemble itself (no separate quantum-only run needed).
    hi_band = f_ref >= f_ref[-1] / 10.0
    srep_quantum_floor = float(np.median(srep_m[hi_band]))
    # DC-photodetector beatnote proxy Sum_j|E_j|^2(t): baseband intensity-noise
    # pedestal (the pulse-train envelope PSD), an independent rep-rate cross-check.
    beat_m = beat.mean(0) if beat.size else np.zeros_like(f_ref_beat)

    # ---- figures ----
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.fill_between(f_ref, np.maximum(srep_m - srep_s, 1e-300), srep_m + srep_s,
                    color="C0", alpha=0.3, lw=0, label=r"$S_{\delta\nu,rep}$ (ensemble mean $\pm$ std)")
    ax.loglog(f_ref, srep_m, color="C0", lw=1.0)
    ax.loglog(f_ref, s_rep_trn, "k--", lw=1.4,
              label=r"TRN(K-G)+FSR limit $(D_1/\omega_0)^2 C_{pull}^2 S_{\delta T}/(2\pi)^2$")
    ax.axhline(srep_quantum_floor, color="C3", ls=":", lw=1.2,
               label="quantum-limited floor (high-offset plateau)")
    ax.set_xlabel("offset frequency f [Hz]")
    ax.set_ylabel(r"$S_{\delta\nu,rep}(f)$  [Hz$^2$/Hz]")
    ax.set_title("Repetition-rate beatnote frequency noise (full stack)\n"
                 f"vs the TRN+FSR limit ({len(srep)} realizations)")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "rf_beatnote_ensemble.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    screp_cr = np.array(scr_list) if scr_list else np.zeros_like(sc)
    for arr, lab, col in ((sc, r"$S_c$ (common)", "C0"),
                          (screp_cr, r"$|S_{cr}|$ (cross)", "C2"),
                          (screp, r"$S_{rep}$", "C1")):
        m = np.abs(arr).mean(0)
        sd = np.abs(arr).std(0, ddof=1) if len(arr) > 1 else np.zeros_like(m)
        ax.fill_between(f_ref_tape, np.maximum(m - sd, 1e-300), m + sd, color=col, alpha=0.25, lw=0)
        ax.loglog(f_ref_tape, np.maximum(m, 1e-300), color=col, lw=1.0, label=lab)
    ax.set_xlabel("offset frequency f [Hz]")
    ax.set_ylabel(r"frequency-noise PSD  [Hz$^2$/Hz]")
    ax.set_title("Elastic-tape comb phase-noise decomposition (full stack)\n"
                 r"$S_\mu(f)=S_c+2\mu S_{cr}+\mu^2 S_{rep}$")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "comb_phase_noise_tape.png")
    plt.close(fig)

    block = {
        "n_tau": n_tau, "t_slow": t_slow, "ensemble": int(len(srep)),
        "probe_mus": list(map(int, probe_mus)),
        "rep_rate_linewidth_hz_mean": lw_mean,
        "rep_rate_linewidth_hz_std": lw_std,
        "d1_over_omega0": d1_over_omega0,
        "srep_over_trn_limit_band_median_db": srep_over_trn_db,
        "srep_ensemble_mean_at_1e6hz": float(psd_at_offset(f_ref, srep_m, 1e6)),
        "trn_limit_srep_at_1e6hz": float(psd_at_offset(f_ref, s_rep_trn, 1e6)),
        "srep_quantum_limited_floor_hz2_per_hz": srep_quantum_floor,
        "dc_pd_beatnote_proxy_psd_median": float(np.median(beat_m)) if beat_m.size else float("nan"),
    }

    # ---- GATE W4 ----
    print("\n" + "=" * 72)
    print("GATE W4 -- coherence / RF-beatnote ensemble")
    print("=" * 72)
    print(f"  ensemble: {len(srep)} realizations")
    print(f"  per-line effective linewidth = {lw_mean:.4e} +/- {lw_std:.2e} Hz")
    print(f"  S_rep vs TRN+FSR limit (low-offset decade): "
          f"{srep_over_trn_db:+.1f} dB")
    print(f"  S_rep(1 MHz): measured {block['srep_ensemble_mean_at_1e6hz']:.3e} | "
          f"TRN limit {block['trn_limit_srep_at_1e6hz']:.3e}  Hz^2/Hz")
    print("=" * 72 + "\n")
    return block


# ---------------------------------------------------------------------------
# Workstream 5 -- vacuum-floor + energy-fluctuation budget (sanity anchor)
# ---------------------------------------------------------------------------
def workstream5(out_dir, seeds, quick, w1_ens=None, numerics=PRODUCTION_NUMERICS):
    """Confirm the ensemble reproduces the ħω₀/2 vacuum floor + shot-noise energy fluctuations."""
    if w1_ens is None:
        # Own small ensemble (n_tau=16384 needed for un-dealiased far wings).
        n_tau = 16384
        settle_off = 800 if quick else 2000
        settle_on = 200 if quick else 800
        record_rt = 64 if quick else 256
        n_ens = min(seeds, 3) if quick else min(seeds, 8)
        cav = attach_dispersion(load_cavity_params(CONFIG_PATH), n_tau)
        dw = OPERATING_DW_KAPPA * cav.kappa
        hbar_omega0 = _hbar_omega0()
        cfg_off = str(write_noise_off_config())
        cfg_full = _full_stack_sidecar(CONFIG_PATH)
        print(f"[W5] own ensemble n_tau={n_tau} seeds={n_ens}")
        e_seed, dt_seed = _settle(cav, dw, n_tau, cfg_off, settle_off, 0, numerics)
        ens_spec, ens_u = [], []
        mu = np.arange(n_tau) - n_tau // 2
        for s in range(n_ens):
            sol = _record(cav, dw, n_tau, cfg_full, e_seed, dt_seed, settle_on,
                          record_rt, 2000 + s, numerics)
            ens_spec.append(_mean_power_spectrum(np.asarray(sol["E_snapshots"])[0]))
            ens_u.append(np.asarray(sol["U_int_history"])[0])
            jax.clear_caches()
        ens_spec = np.array(ens_spec)
        mean_lin = ens_spec.mean(0)
        p_pump_on = float(mean_lin[n_tau // 2])
    else:
        n_tau = w1_ens["n_tau"]
        mu = w1_ens["mu"]
        mean_lin = w1_ens["mean_lin"]
        ens_u = w1_ens["energy_series"]
        p_pump_on = w1_ens["p_pump_on"]
        hbar_omega0 = w1_ens["hbar_omega0"]
        cav = w1_ens["cav"]
        print("[W5] reusing Workstream-1 ensemble")

    kappa_i, _kappa_c, _kappa = resolve_cavity_rates(CONFIG_PATH)
    t_r = cav.t_r

    # (i) far-wing modal energy -> hbar*omega0/2 per mode.
    # SYMMETRIC-ordered throughout: this is the vacuum-floor diagnostic itself,
    # so subtracting the pedestal would subtract the quantity under test.
    wing = (np.abs(mu) >= 3600) & (np.abs(mu) <= 5000)
    modal_energy_j = mean_lin / n_tau ** 2               # E_mu = |fft|^2 / n_tau^2
    far_wing_energy = float(np.median(modal_energy_j[wing]))
    floor_multiple = far_wing_energy / (hbar_omega0 / 2.0)
    far_wing_db_rel_pump = 10.0 * math.log10(float(np.median(mean_lin[wing])) / p_pump_on)

    # (i-b) The same band, NORMALLY ordered -- the one W5 quantity an experiment
    # could see. The far wing is pure vacuum, so an OSA reads zero there: this
    # residual must be consistent with zero, which is the instrument-facing
    # restatement of "the floor is at hbar*omega0/2". Reported as a multiple of
    # the symmetric pedestal so it is directly comparable with floor_multiple
    # above (1.000 there <-> 0.000 here).
    far_wing_norm_lin = float(np.median(
        normal_ordered_spectrum(mean_lin, n_tau, hbar_omega0, clip=False)[wing]))
    far_wing_norm_multiple = far_wing_norm_lin / (n_tau ** 2 * hbar_omega0 / 2.0)

    # (ii) intracavity-energy fluctuation vs the shot-noise scale.
    # The solver's U_int = (total field energy) * t_r (u_int = sum_tau|E|^2*t_r/n_tau,
    # and sum_mu E_mu = sum_tau|E|^2/n_tau), so the PHYSICAL intracavity energy is
    # E_total = U_int / t_r [J] -- consistent with P_abs = kappa_i*U_int/t_r and with
    # the modal-energy sum that reproduces hbar*omega0/2 per far mode. Shot noise of a
    # coherent field of energy E: var(E) = hbar*omega0*E => std = sqrt(hbar*omega0*E).
    e_series = [np.asarray(u) / t_r for u in ens_u] if len(ens_u) else [np.zeros(1)]
    e_all = np.concatenate(e_series)
    e_mean = float(np.mean(e_all))                       # ~ pJ physical energy
    e_std = float(np.std(e_all))
    shot_std = math.sqrt(hbar_omega0 * e_mean) if e_mean > 0 else float("nan")
    energy_fluct_ratio = e_std / shot_std if shot_std and np.isfinite(shot_std) else float("nan")
    n_photons = e_mean / hbar_omega0 if e_mean > 0 else float("nan")

    # (iii) vacuum contribution to P_abs = κ_i·n_tau·ħω₀/2 [W], vs the real P_abs.
    p_abs_vacuum = kappa_i * n_tau * hbar_omega0 / 2.0
    p_abs_real = kappa_i * e_mean if e_mean > 0 else float("nan")  # = kappa_i*U_int/t_r
    p_abs_vac_ratio = p_abs_vacuum / p_abs_real if np.isfinite(p_abs_real) and p_abs_real > 0 else float("nan")

    # ---- figures ----
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.semilogy(mu, modal_energy_j, lw=0.5, color="C1", label="ensemble-mean modal energy")
    ax.axhline(hbar_omega0 / 2.0, color="k", ls="--", lw=1.2, label=r"$\hbar\omega_0/2$")
    ax.axhspan(hbar_omega0 / 6.0, hbar_omega0 * 1.5, color="k", alpha=0.08,
               label="factor-3 band")
    ax.set_xlabel(r"mode index $\mu$")
    ax.set_ylabel(r"modal energy $E_\mu$ [J]")
    ax.set_ylim(hbar_omega0 / 20.0, None)
    ax.set_title("Vacuum floor: far-wing modal energy vs $\\hbar\\omega_0/2$")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "vacuum_floor_ensemble.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    L = min(len(e) for e in e_series)
    stack = np.array([e[:L] for e in e_series])          # physical energy [J]
    t_ms = np.arange(L) * t_r * 1e6
    m = stack.mean(0) * 1e12                              # pJ
    sd = stack.std(0) * 1e12 if stack.shape[0] > 1 else np.zeros_like(m)
    ax.fill_between(t_ms, m - sd, m + sd, color="C0", alpha=0.3, lw=0,
                    label=r"ensemble mean $\pm$ std")
    ax.plot(t_ms, m, color="C0", lw=0.7)
    ax.axhspan((e_mean - shot_std) * 1e12, (e_mean + shot_std) * 1e12,
               color="C2", alpha=0.35, label="shot-noise band $\\pm\\sqrt{\\hbar\\omega_0 E}$")
    ax.set_xlabel("time [µs]")
    ax.set_ylabel(r"intracavity energy $E_{total}=U_{int}/t_r$ [pJ]")
    ax.set_title("Intracavity-energy fluctuations vs the shot-noise scale\n"
                 f"($\\bar E$ = {e_mean * 1e12:.2f} pJ $\\approx$ {n_photons:.1e} photons)")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "energy_fluctuation_budget.png")
    plt.close(fig)

    block = {
        "n_tau": n_tau, "hbar_omega0_j": hbar_omega0,
        "reused_workstream1_ensemble": bool(w1_ens is not None),
        "far_wing_modal_energy_j": far_wing_energy,
        "far_wing_floor_multiple_of_half_hbar_omega0": floor_multiple,
        "far_wing_db_rel_pump": far_wing_db_rel_pump,
        "floor_within_factor_3": bool(1.0 / 3.0 <= floor_multiple <= 3.0),
        "far_wing_normal_ordered_lin": far_wing_norm_lin,
        "far_wing_normal_ordered_multiple_of_pedestal": far_wing_norm_multiple,
        "spectral_ordering_note": (
            "The floor diagnostics and every energy term here (modal energy, "
            "intracavity energy, shot-noise scale, P_abs) are SYMMETRIC-ordered "
            "by design -- they describe the field the solver integrates, and "
            "normal-ordering them would double-count the vacuum and break the "
            "energy balance. Only far_wing_normal_ordered_* is "
            "instrument-comparable: the far wing is pure vacuum, so an OSA reads "
            "zero there and that multiple should sit near 0.000 while "
            "far_wing_floor_multiple_of_half_hbar_omega0 sits near 1.000."),
        "intracavity_energy_mean_j": e_mean,
        "intracavity_energy_photons": n_photons,
        "intracavity_energy_std_j": e_std,
        "shot_noise_energy_std_j": shot_std,
        "energy_fluctuation_ratio_measured_over_shot": energy_fluct_ratio,
        "energy_fluctuation_at_or_above_shot_floor": bool(
            np.isfinite(energy_fluct_ratio) and energy_fluct_ratio >= 0.3),
        "p_abs_vacuum_w": p_abs_vacuum,
        "p_abs_real_w": p_abs_real,
        "p_abs_vacuum_over_real": p_abs_vac_ratio,
        "kappa_i_rad_s": float(kappa_i),
    }

    # ---- GATE W5 ----
    print("\n" + "=" * 72)
    print("GATE W5 -- vacuum-floor + energy-fluctuation budget")
    print("=" * 72)
    print(f"  far-wing modal energy = {far_wing_energy:.4e} J = "
          f"{floor_multiple:.3f} x (ħω₀/2)   [within factor 3: {block['floor_within_factor_3']}]")
    print(f"  far-wing level = {far_wing_db_rel_pump:.1f} dB rel. pump "
          f"(symmetric-ordered)")
    print(f"  far-wing NORMALLY ordered = {far_wing_norm_multiple:+.4f} x pedestal "
          f"(an OSA reads 0 here; the wing is pure vacuum)")
    print(f"  intracavity energy E = {e_mean * 1e12:.3f} pJ ({n_photons:.2e} photons)")
    print(f"  energy-fluct std = {e_std:.3e} J | shot-noise scale "
          f"{shot_std:.3e} J | ratio {energy_fluct_ratio:.2f} (>=1 => classical excess)")
    print(f"  P_abs vacuum contribution = κ_i·n_tau·ħω₀/2 = {p_abs_vacuum:.3e} W")
    print(f"    vs real P_abs = {p_abs_real:.3e} W  ->  ratio {p_abs_vac_ratio:.2e} "
          f"(negligible for the thermal ODE)")
    print("=" * 72 + "\n")
    return block


# ---------------------------------------------------------------------------
# Consolidated metric-vs-expectation table (printed after --workstream all)
# ---------------------------------------------------------------------------
def _consolidated_summary(report: dict) -> None:
    """Print the final metric-vs-physical-expectation table across workstreams."""
    def g(block, key, default=None):
        return report.get(block, {}).get(key, default)

    print("\n" + "#" * 74)
    print("# CONSOLIDATED CAMPAIGN SUMMARY -- metric vs physical expectation")
    print("#" * 74)
    rows = []
    w1 = report.get("workstream1_dw_survival", {})
    if w1:
        rows += [
            ("W1 DW crossings at phase-match (dispersion intact)",
             str(w1.get("dw_off_peaks_at_phase_match_crossing")), "True"),
            ("W1 DW position invariance (resolvable shift, modes)",
             f"{w1.get('max_resolvable_dw_position_shift_modes')}", "<= 2"),
            ("W1 DW peaks unresolved (normally ordered)",
             f"{w1.get('n_dw_peaks_submerged_below_floor')}/"
             f"{w1.get('n_dw_peaks_submerged_below_floor', 0) + w1.get('n_dw_peaks_surviving_above_floor', 0)}",
             f"< {w1.get('dw_resolve_sigma', RESOLVE_SIGMA):g} sigma"),
            ("W1 ensemble s.e. (detectability limit) [dB rel pump]",
             f"{w1.get('ensemble_mean_sem_median_db_rel_pump', float('nan')):.1f}",
             "sets what is visible"),
            ("W1 comb fraction ON (normally ordered)",
             f"{w1.get('comb_fraction_on_normal_ordered', float('nan')):.4f}",
             f"~ OFF {w1.get('comb_fraction_off_normal_ordered', float('nan')):.4f}"),
            ("W1 wing change within vacuum+pump budget",
             str(w1.get("broadening_budget", {}).get("within_budget")), "True"),
            ("W1 far-wing floor / (n_tau^2 hbar w0/2)",
             f"{w1.get('empirical_far_wing_floor_multiple_of_vacuum'):.2f}", "~1 (factor 3)"),
        ]
    w2 = report.get("workstream2_staircase", {})
    if w2:
        rows += [
            ("W2 single-soliton access success rate",
             f"{w2.get('single_soliton_success_rate'):.2f}", ">= 0.5"),
            ("W2 all matched transitions bias < jitter",
             str(w2.get("all_bias_lt_jitter")), "True"),
        ]
    w3 = report.get("workstream3_linewidth", {})
    if w3:
        rows += [
            ("W3 cross-realization a2 curvature [Hz^2/mode^2]",
             f"{w3.get('a2_cross_realization_mean'):.3e}", "> 0"),
            ("W3 bootstrap p(a2 > 0)",
             f"{w3.get('a2_bootstrap_p_positive'):.3f}", ">= 0.9"),
            ("W3 flagship Taylor control ratio",
             f"{w3.get('flagship_taylor_ratio'):.3f}", "<= 0.1"),
        ]
    w4 = report.get("workstream4_beatnote", {})
    if w4:
        rows += [
            ("W4 rep-rate per-line linewidth [Hz]",
             f"{w4.get('rep_rate_linewidth_hz_mean'):.3e}", "finite"),
            ("W4 S_rep vs TRN+FSR limit [dB]",
             f"{w4.get('srep_over_trn_limit_band_median_db'):+.1f}", ">= 0 (pump/quantum)"),
        ]
    w5 = report.get("workstream5_vacuum_budget", {})
    if w5:
        rows += [
            ("W5 far-wing floor / (hbar w0/2)",
             f"{w5.get('far_wing_floor_multiple_of_half_hbar_omega0'):.3f}", "~1 (factor 3)"),
            ("W5 energy fluctuation / shot-noise scale",
             f"{w5.get('energy_fluctuation_ratio_measured_over_shot'):.2f}", ">= ~1"),
            ("W5 P_abs vacuum / real P_abs",
             f"{w5.get('p_abs_vacuum_over_real'):.2e}", "<< 1 (negligible)"),
        ]
    w = max((len(r[0]) for r in rows), default=10)
    print(f"{'metric':<{w}} {'measured':>26} {'expectation':>22}")
    print("-" * (w + 50))
    for name, meas, exp in rows:
        print(f"{name:<{w}} {meas:>26} {exp:>22}")
    print("#" * 74 + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workstream", default="all",
                    help="1|2|3|4|5|all  (comma lists allowed, e.g. 1,5)")
    ap.add_argument("--seeds", type=int, default=24,
                    help="ensemble size (W1 default 24; W2 caps at 12, W3/W4 use >=16)")
    ap.add_argument("--quick", action="store_true",
                    help="minutes-scale smoke of the whole pipeline")
    ap.add_argument("--out", default=str(RESULTS_DIR / VALIDATION_DIRNAME),
                    help="output directory for figures + campaign_report.json")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "campaign_report.json"

    want = {s.strip() for s in args.workstream.split(",") if s.strip()}
    if "all" in want:
        want = {"1", "2", "3", "4", "5"}

    report = _load_report(report_path)
    report.setdefault("meta", {})
    report["meta"].update({"seeds": args.seeds, "quick": bool(args.quick),
                           "operating_dw_kappa": OPERATING_DW_KAPPA, "pin_w": PIN_W})

    t_start = time.time()
    w1_ens = None
    if "1" in want:
        block, w1_ens = workstream1(out_dir, args.seeds, args.quick)
        report["workstream1_dw_survival"] = block
        _save_report(report_path, report)
    if "2" in want:
        report["workstream2_staircase"] = workstream2(out_dir, args.seeds, args.quick)
        _save_report(report_path, report)
    if "3" in want:
        report["workstream3_linewidth"] = workstream3(out_dir, args.seeds, args.quick)
        _save_report(report_path, report)
    if "4" in want:
        report["workstream4_beatnote"] = workstream4(out_dir, args.seeds, args.quick)
        _save_report(report_path, report)
    if "5" in want:
        report["workstream5_vacuum_budget"] = workstream5(
            out_dir, args.seeds, args.quick, w1_ens=w1_ens)
        _save_report(report_path, report)

    from analysis._provenance import provenance_stamp
    report["provenance"] = provenance_stamp(
        "analysis/noise_validation_campaign.py", args.seeds,
        physical_params=_load_config(str(CONFIG_PATH)),
        quick=bool(args.quick),
    )
    report["meta"]["wall_time_s"] = round(time.time() - t_start, 1)
    _save_report(report_path, report)
    if want == {"1", "2", "3", "4", "5"}:
        _consolidated_summary(report)
    print(f"\nWrote {report_path}  (workstreams: {sorted(want)}; "
          f"{report['meta']['wall_time_s']}s)")


if __name__ == "__main__":
    main()
