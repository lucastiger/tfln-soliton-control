#!/usr/bin/env python
"""Stochastic-LLE NOISE BUDGET: which channel costs how much, with error bars.

This is the benchmark's central table. For a single dissipative Kerr soliton at
the committed operating point it answers, per observable, *how much of the
measured fluctuation each noise channel is responsible for* -- one-at-a-time
(OAT), leave-one-out (LOO), and the interaction residual between them.

Design
------
CHANNEL SETS (13 rows). Nine named sets plus four generated LOO rows::

    all_off    every stochastic channel off (the deterministic reference)
    quantum    quantum_vacuum only
    trn        trn only
    pyro_eo    pyro_eo + trn          <- see "grouped channels" below
    fsr        fsr + trn              <- same reason
    dT_family  trn + pyro_eo + fsr    <- the physically correct grouped channel
    pump_fn    pump_freq_noise only
    pump_rin   pump_rin only
    all_on     every switch on
    loo_<X>    all_on minus X, for X in {quantum, dT_family, pump_fn, pump_rin}

GROUPED CHANNELS -- why pyro_eo and fsr are never run alone. The
thermorefractive (``trn``), pyro-electric/electro-optic (``pyro_eo``) and
repetition-rate (``fsr``) channels are not three independent noise sources:
they are ONE temperature fluctuation delta_T(t) seen through three different
coupling coefficients. ``simulator.noise_models.TotalNoise.sample_with_delta_t``
draws a single delta_T realization and forms

    trn_noise    = c_pull    * delta_T(t)
    pyroeo_noise = pyro_coeff* delta_T(t)
    dD1(t)       = (D1/omega0) * c_pull * delta_T(t)      [the FSR channel]

so ``pyro_eo=True, trn=False`` yields IDENTICALLY ZERO (the solver warns about
exactly this), and the same holds for ``fsr``. Running them "alone" would
measure nothing. The rows named ``pyro_eo`` and ``fsr`` therefore carry ``trn``
with them, and ``dT_family`` is the grouped channel that the interaction
accounting actually uses. This is also why a large interaction residual is
EXPECTED rather than alarming -- see :func:`interaction_commentary`.

THIS DEVICE (config/sin_params.yaml, SiN): ``eo_r33_m_per_v = 0`` and
``pyroelectric_coeff_c_per_m2_k = 0`` because SiN is centrosymmetric, so
``pyro_coeff = 0`` and ``sigma_tccr = 0``. Two consequences the table will show
and the run asserts:

  * the ``pyro_eo`` row must come out BIT-IDENTICAL to the ``trn`` row (adding a
    zero-coefficient channel to the same delta_T realization changes nothing);
  * ``tccr``, though switched on inside ``all_on``, contributes exactly zero, so
    its absence from the non-overlapping OAT grouping costs the residual nothing.

The first of those is checked at run time (``--quick`` included) and is the
engine's live proof that common random numbers are actually working.

COMMON RANDOM NUMBERS. Every channel set is run on the SAME seed list. This is
not cosmetic: OAT and LOO contributions are DIFFERENCES between channel sets,
and with independent seeds those differences are swamped by sampling noise --
the single most common way a budget table comes out meaningless. The solver's
key derivation makes CRN exact here rather than approximate: the PRNG splits in
``solve_lle_ssfm_jax`` and in ``TotalNoise.sample_with_delta_t`` are structural
and UNCONDITIONAL (each channel's subkey is derived by position, never by
consumption order), so toggling one channel does not shift any other channel's
random stream. Given a seed, the delta_T realization behind the ``trn`` row is
the very same realization as behind the ``all_on`` row.

THERMAL FEEDBACK IS PINNED ON in every cell. ``NoiseConfig.all_off()`` leaves
``thermal_feedback`` at ``False`` while ``NoiseConfig.all_on()`` sets it
``True``; differencing those two directly would fold the deterministic
thermo-optic ODE -- which is *dynamics*, not noise -- into every OAT and LOO
contribution, and the budget would partly be measuring the thermal ODE. Every
cell here therefore uses ``all_off(thermal_feedback=True)`` and ``all_on()``,
matching the committed device config (``noise.thermal_feedback: true``). This is
recorded as ``thermal_feedback_pinned`` in the JSON.

AMPLITUDE PRESET. The committed config carries zero pump-noise PSD amplitudes
(``h0 = 0``, ``h-1 = 0``, RIN floor ``-300 dBc/Hz``), so a budget run against it
verbatim would report structural zeros for ``pump_fn``, ``pump_rin`` and their
LOO rows. Every cell is therefore run against the campaign's committed ECDL
flagship preset -- ``ECDL_H0``, ``ECDL_HM1``, ``ECDL_RIN_FLOOR_DBC`` and the
Kondratiev-Gorodetsky TRN geometry ``KG_GEOMETRY``, imported from
``analysis.noise_validation_campaign`` -- so the budget is directly comparable
with that campaign's W1-W5 results on the same device.

RECORD LENGTHS. The lowest Fourier frequency a record can carry is
1/(t_slow * t_r), with t_r = 40.65 ps. Two record lengths are run:

    "fast"  t_slow = 2e5   (8.13 us)   Fourier floor 123 kHz
    "slow"  t_slow = 2e7   (813 us)    Fourier floor 1.23 kHz

but the Fourier floor is NOT the frequency a Welch estimate reaches: Welch
resolves fs/nperseg, and a segment length short enough to average over
``WELCH_MIN_SEGMENTS`` segments sits well above the record floor. Every spectral
cell therefore records the segment length, the segment count and the achieved
bin spacing that produced it, and is marked ``resolved`` only when the target
frequency clears ``WELCH_BIN_MARGIN`` bins with at least ``WELCH_MIN_SEGMENTS``
segments behind it. A target BELOW the record's Fourier floor yields ``null``,
never an extrapolation -- ``noise_metrology.psd_at_offset`` clamps at the ends of
its interpolation table, which would otherwise return a confident-looking number
for a frequency the record never observed. Which record produced each cell is
written into the JSON alongside the value, because a budget number without its
observation time is not a number.

At the production record lengths above, the requested S_rep targets resolve as:
1 MHz on both records; 100 kHz and 10 kHz on the slow record only; and 1 kHz on
NEITHER (it needs nperseg >= 4*fs/1e3 ~ 1e8 samples, i.e. t_slow ~ 2e8, an 8.1 ms
record). The engine reports that rather than papering over it.

OBSERVABLES (5). Each is a function ``(solution, ctx) -> (values, unit,
uncertainty)`` where ``values`` and ``uncertainty`` are dicts keyed by a point
label (``"value"`` for scalar observables), and per-point diagnostics are pushed
into ``ctx.notes``:

  1. ``u_int_rms_frac``  std(U_int)/mean(U_int) over the record [dimensionless]
  2. ``s_rep``           S_rep(f) from ``noise_metrology.tape_model_fit`` at
                         f in {1e3, 1e4, 1e5, 1e6} Hz [Hz^2/Hz]
  3. ``line_linewidth``  ``noise_metrology.effective_linewidth`` at
                         mu in {-100, -10, 0, 10, 100} [Hz]
  4. ``timing_jitter``   ``noise_metrology.timing_jitter`` integrated over
                         1 kHz - 1 MHz [fs]
  5. ``step_jitter``     soliton-step switching detuning from the staircase
                         protocol [kappa], via
                         ``noise_validation_campaign.level_crossing``

S_REP ESTIMATOR CAVEAT. Observable 2 is the tape-model fit, as specified. That
fit's own docstring documents where it fails: when the common mode dominates
(S_c >> mu_max^2 * S_rep, which is the thermorefractive regime for this device,
common/rep ratio ~ (omega0/D1)^2 ~ 6e7) the fitted mu^2 curvature drowns in the
per-bin chi^2 noise of the Welch estimates and the fitted S_rep is not
meaningful. Every s_rep cell therefore also carries ``s_rep_pairwise`` -- the
same quantity from the pairwise DIFFERENCE estimator ``rep_rate_phase``, which
cancels the common mode deterministically before any spectral estimate -- and
``common_mode_dominated``, true when the cell sits in the documented failure
regime. Read the pairwise number there; that is what the campaign's quiet-point
sweep and TRN-limit validation use.

STATISTICS. ``--seeds N`` (default 24) realizations per cell; mean, SEM, std and
a ``BOOTSTRAP_RESAMPLES``-resample percentile 95% CI on both the mean and the
std. Observables 1-4 report the MEAN as the budget statistic; observable 5
reports the STD, because the switching-detuning *jitter* is by definition the
seed-to-seed spread of a per-seed switching detuning (the mean of that sample is
the step LOCATION, and location-minus-deterministic is a bias, not a jitter).
Each cell names its own ``budget_statistic``.

DERIVED QUANTITIES (all three reported, per observable point):

  * OAT contribution:  X[set]    - X[all_off]
  * LOO contribution:  X[all_on] - X[all_on minus set]
  * interaction residual:
        X[all_on] - X[all_off] - sum_{S in OAT_GROUPING} (X[S] - X[all_off])
    over the NON-OVERLAPPING grouping {quantum, dT_family, pump_fn, pump_rin}.

OUTPUT. ``<out>/budget.json`` (plus ``budget.npz`` + ``budget.provenance.json``
via ``simulator.provenance.save_artifact``), ``budget_table.tex`` (booktabs,
ready to ``\\input``) and ``budget_table.md``. The per-seed raw store
``budget_raw.json`` is rewritten atomically after EVERY solver unit, so a crash
at cell 47 costs one unit rather than 46 cells; ``--resume`` picks it back up.

Reference: Herr, Tikan & Kippenberg, arXiv:2604.05897v1 (7 Apr 2026) -- noise
metrology Sec. V.B.1, quantum-vacuum drive Eq. 126, TRN Eqs. 129-130, pump
frequency noise / RIN Sec. V.B.4-V.B.5. Equation numbers are quoted for v1.

Usage::

    python analysis/noise_budget.py --quick
    python analysis/noise_budget.py --seeds 24
    python analysis/noise_budget.py --resume --channel-set trn --channel-set all_on
    python analysis/noise_budget.py --observable u_int_rms_frac --observable step_jitter
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402

from analysis.dks_access import (  # noqa: E402
    CONFIG_PATH,
    OPERATING_DW_KAPPA,
    PIN_W,
    PRODUCTION_NUMERICS,
    RESULTS_DIR,
    _run,
    attach_dispersion,
    load_cavity_params,
    sech_soliton_seed,
)
from analysis.noise_metrology import (  # noqa: E402
    effective_linewidth,
    frequency_noise_psd,
    rep_rate_phase,
    tape_model_fit,
    timing_jitter as metrology_timing_jitter,
    unwrapped_phases,
)

# The W2 staircase machinery, IMPORTED rather than reimplemented: the sweep
# driver, its config object, and the level-crossing switching-detuning estimator
# that W2 uses for its bias-vs-jitter split.
from analysis.noise_validation_campaign import (  # noqa: E402
    ECDL_H0,
    ECDL_HM1,
    ECDL_RIN_FLOOR_DBC,
    KG_GEOMETRY,
    level_crossing,
)
from analysis.run_detuning_sweep import SweepConfig, run_detuning_sweep  # noqa: E402
from simulator.lle_solver import _load_config, _resolve_noise_flags  # noqa: E402
from simulator.noise_config import NoiseConfig  # noqa: E402
from simulator.provenance import provenance_stamp, save_artifact  # noqa: E402

__all__ = [
    "BUDGET_DIRNAME",
    "CHANNEL_SETS",
    "OAT_GROUPING",
    "OBSERVABLES",
    "RECORDS",
    "BudgetError",
    "RecordContext",
    "RecordSpec",
    "budget_sidecar",
    "channel_sets",
    "interaction_commentary",
    "main",
    "run_budget",
    "welch_segmentation",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUDGET_DIRNAME = "budget"

#: Probe comb lines. Five distinct mu are the MINIMUM ``tape_model_fit``
#: accepts: the per-bin fit has three free parameters (S_c, S_cr, S_rep) and
#: must be over-determined. These double as the linewidth report points, so one
#: probe set serves observables 2 and 3.
PROBE_MUS: tuple[int, ...] = (-100, -10, 0, 10, 100)

#: S_rep report frequencies [Hz].
S_REP_FREQS_HZ: tuple[float, ...] = (1.0e3, 1.0e4, 1.0e5, 1.0e6)

#: Timing-jitter integration band [Hz]. The achieved band is the intersection of
#: this with what the record actually resolves, and is reported per cell.
JITTER_BAND_HZ: tuple[float, float] = (1.0e3, 1.0e6)

#: A Welch estimate is called RESOLVED at f only when f sits at least this many
#: bins above DC (bin spacing fs/nperseg) AND at least WELCH_MIN_SEGMENTS
#: segments were averaged. Below either bar the value is still computed and
#: reported -- with its (large) chi^2 uncertainty -- but flagged unresolved.
WELCH_BIN_MARGIN = 4.0
WELCH_MIN_SEGMENTS = 8

#: Blocks used for the within-record uncertainty of time-domain scalars. The
#: record is split into this many contiguous blocks, the statistic is evaluated
#: per block, and the uncertainty is the block-to-block SEM -- which, unlike a
#: naive iid formula, does not pretend that a correlated record carries N
#: independent samples.
UNCERTAINTY_BLOCKS = 8

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
#: Fixed so the published CI is reproducible from the same per-seed sample.
BOOTSTRAP_SEED = 20260407

DEFAULT_SEEDS = 24
#: Base of the seed list. Shared by every channel set -- that IS the common
#: random numbers guarantee.
SEED_BASE = 100

#: The soliton-step staircase is expensive, and its per-seed statistic is a
#: switching detuning rather than a spectral estimate, so it caps its ensemble
#: independently of ``--seeds``.
STEP_JITTER_MAX_SEEDS = 6

#: Temperature-fluctuation spectrum for every cell. The ECDL flagship preset
#: pairs the pump numbers above with the analytic Kondratiev-Gorodetsky WGM PSD
#: (arXiv:2604.05897v1 Eq. 130, variance renormalized to the Eq. 129
#: thermodynamic value) on the ``KG_GEOMETRY`` mode geometry, which is what the
#: W1-W5 campaign's full-stack runs use. It must be set BOTH on the NoiseConfig
#: and in ``physical_parameters`` -- see :func:`budget_sidecar`.
TRN_PSD_MODEL = "kondratiev_gorodetsky"

OAT_GROUPING: tuple[str, ...] = ("quantum", "dT_family", "pump_fn", "pump_rin")
LOO_TARGETS: tuple[str, ...] = OAT_GROUPING

#: Channel-set name -> the NoiseConfig switch fields it turns on. ``pyro_eo`` and
#: ``fsr`` carry ``trn`` because they are driven by the same delta_T realization
#: (module docstring, "grouped channels").
_OAT_SPEC: dict[str, tuple[str, ...]] = {
    "quantum": ("quantum_vacuum",),
    "trn": ("trn",),
    "pyro_eo": ("pyro_eo", "trn"),
    "fsr": ("fsr", "trn"),
    "dT_family": ("trn", "pyro_eo", "fsr"),
    "pump_fn": ("pump_freq_noise",),
    "pump_rin": ("pump_rin",),
}


def _freq_label(f: float) -> str:
    """Point label for a report frequency, e.g. ``1e6 Hz``."""
    return f"{f:.0e} Hz".replace("e+0", "e").replace("e+", "e")


class BudgetError(RuntimeError):
    """A budget run cannot proceed without producing a misleading table."""


# ---------------------------------------------------------------------------
# Channel sets
# ---------------------------------------------------------------------------
def channel_sets() -> dict[str, NoiseConfig]:
    """The 13 channel sets, in table order.

    ``thermal_feedback`` is pinned ON in every set (module docstring): it is
    deterministic thermo-optic dynamics, not a noise channel, and letting it
    differ between ``all_off`` and ``all_on`` would contaminate every OAT and
    LOO contribution with the thermal ODE.
    """
    common = dict(thermal_feedback=True, trn_psd_model=TRN_PSD_MODEL)
    sets: dict[str, NoiseConfig] = {
        "all_off": NoiseConfig.all_off(**common),
    }
    for name, channels in _OAT_SPEC.items():
        sets[name] = NoiseConfig.all_off(
            **common, **{c: True for c in channels}
        )
    sets["all_on"] = NoiseConfig.all_on(**common)
    for target in LOO_TARGETS:
        off = _OAT_SPEC[target]
        sets[f"loo_{target}"] = NoiseConfig.all_on(
            **common, **{c: False for c in off})
    return sets


CHANNEL_SETS: dict[str, NoiseConfig] = channel_sets()


# ---------------------------------------------------------------------------
# Sidecar configs
# ---------------------------------------------------------------------------
#: Legacy ``physical_parameters`` switches that OUTRANK the top-level ``noise:``
#: block (``_resolve_noise_flags`` level 3 beats level 4). They are removed from
#: every budget sidecar so the block governs. ``pump_noise_enabled`` in
#: particular is a SINGLE flag gating BOTH pump channels, so leaving it in place
#: would make the ``pump_fn`` and ``pump_rin`` rows inexpressible.
_LEGACY_SWITCHES_TO_DROP = (
    "quantum_noise_enabled",
    "pump_noise_enabled",
    "fsr_noise_enabled",
)


def budget_sidecar(nc: NoiseConfig, out_path: Path, base_config=CONFIG_PATH) -> Path:
    """Write the sidecar YAML for one channel set and VERIFY it resolves to ``nc``.

    The sidecar is the committed config with three edits:

    1. the ECDL flagship amplitude preset written into ``physical_parameters``
       (pump frequency-noise plateau + flicker, RIN floor, Kondratiev-Gorodetsky
       TRN model and its mode geometry). ``trn_psd_model`` must live in
       ``physical_parameters`` and not only in the ``noise:`` block, because
       ``simulator.noise_models._resolve_trn_psd_model`` reads it from there;
    2. the legacy switch keys removed (see ``_LEGACY_SWITCHES_TO_DROP``);
    3. ``noise:`` set to ``nc``.

    The write is then read back through ``_resolve_noise_flags`` -- the solver's
    own precedence chain, not a reimplementation of it -- and compared field by
    field against ``nc``. A silent precedence trap here would not crash: it
    would produce a complete, plausible budget table for the wrong channel sets,
    so it is made a hard error.
    """
    with open(base_config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    pp = cfg.setdefault("physical_parameters", {})
    for key in _LEGACY_SWITCHES_TO_DROP:
        pp.pop(key, None)
    pp["pump_freq_noise_h0_hz2_per_hz"] = float(ECDL_H0)
    pp["pump_freq_noise_hm1_hz3_per_hz"] = float(ECDL_HM1)
    pp["pump_rin_floor_dbc_per_hz"] = float(ECDL_RIN_FLOOR_DBC)
    pp["trn_psd_model"] = nc.trn_psd_model
    pp.update({k: float(v) for k, v in KG_GEOMETRY.items()})

    cfg["noise"] = nc.to_dict()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    resolved = _resolve_noise_flags(pp, out_path, None)
    mismatched = {
        f.name: (getattr(nc, f.name), getattr(resolved, f.name))
        for f in dataclasses.fields(NoiseConfig)
        if getattr(nc, f.name) != getattr(resolved, f.name)
    }
    if mismatched:
        raise BudgetError(
            f"sidecar {out_path} does not resolve to the requested NoiseConfig; "
            f"field(s) requested vs resolved: {mismatched}. The precedence chain "
            f"(_resolve_noise_flags) overruled the 'noise:' block -- refusing to "
            f"run a budget whose channel sets are not what the table will claim."
        )
    return out_path


@contextlib.contextmanager
def _quiet():
    """Silence the solver's per-run stdout and warnings; exceptions still fly."""
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        yield buf


# ---------------------------------------------------------------------------
# Record specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecordSpec:
    """One record length, with everything needed to reproduce it.

    ``t_slow`` round trips are recorded after ``settle_rt`` round trips of
    settling under the SAME noise configuration (a record started straight from
    the analytic seed would carry the settling transient into every PSD).
    """

    name: str
    t_slow: int
    settle_rt: int
    snapshot_interval: int
    n_tau: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


#: Production records. "fast" carries the time-domain observables and the top
#: spectral decade; "slow" reaches down to the low-frequency S_rep points and
#: the linewidths. See the module docstring for what each actually resolves.
RECORDS: dict[str, RecordSpec] = {
    "fast": RecordSpec("fast", t_slow=200_000, settle_rt=20_000,
                       snapshot_interval=1, n_tau=512),
    "slow": RecordSpec("slow", t_slow=20_000_000, settle_rt=20_000,
                       snapshot_interval=2_000, n_tau=512),
}

#: Quick records: a plumbing smoke, not a measurement. At fs = 24.6 GHz no
#: affordable record reaches 1 kHz, so the low S_rep points come back null with
#: a stated reason -- which is the correct answer, and exercises exactly the
#: reporting path a production run needs.
QUICK_RECORDS: dict[str, RecordSpec] = {
    "fast": RecordSpec("fast", t_slow=16_000, settle_rt=2_500,
                       snapshot_interval=1, n_tau=256),
    # 45k round trips keeps the record's Fourier floor (547 kHz) below the top
    # S_rep target, so the 1 MHz point is still a measurement rather than a
    # null -- the smallest slow record for which --quick exercises the spectral
    # path at all.
    "slow": RecordSpec("slow", t_slow=45_000, settle_rt=2_500,
                       snapshot_interval=8, n_tau=256),
}

#: Staircase (observable 5) sweep settings. Production mirrors the W2 ensemble
#: shape; quick still crosses annihilations, just coarsely.
STAIRCASE = dict(n_tau=4096, n_solitons=5, n_steps=101, settle_rt=6_000,
                 hold_rt=1_000, dw_start_kappa=12.0, dw_stop_kappa=5.5)
#: n_tau = 2048 is not negotiable even in quick mode: at n_tau = 1024 the
#: 3-soliton state survives the whole 11 -> 5.5 kappa ramp without annihilating,
#: so no level crossing exists and the step-jitter observable has nothing to
#: measure. The stop detuning runs to 4.5 kappa so the crossing lands INSIDE the
#: grid rather than on its last step, where it would be an edge artefact.
QUICK_STAIRCASE = dict(n_tau=2048, n_solitons=3, n_steps=11, settle_rt=1200,
                       hold_rt=450, dw_start_kappa=11.0, dw_stop_kappa=4.5)


@dataclass
class RecordContext:
    """What an observable needs besides the solver output.

    ``notes`` is the diagnostics channel: observables keep the specified
    ``(values, unit, uncertainty)`` return signature and push per-point
    metadata (segment lengths, achieved bands, resolution flags) in here for the
    runner to store next to the value.
    """

    record: str
    t_r: float
    kappa: float
    t_slow: int
    snapshot_interval: int
    n_tau: int
    probe_mus: tuple[int, ...]
    channel_set: str
    seed: int
    notes: dict = field(default_factory=dict)
    # Staircase-only context (observable 5).
    det_k: np.ndarray | None = None
    n_solitons: int = 0

    @property
    def duration_s(self) -> float:
        return float(self.t_slow) * float(self.t_r)

    @property
    def fourier_floor_hz(self) -> float:
        """1/(t_slow*t_r): the lowest frequency the RECORD can carry at all."""
        return 1.0 / self.duration_s


# ---------------------------------------------------------------------------
# Welch segmentation
# ---------------------------------------------------------------------------
def welch_segmentation(f_target: float, n_samples: int, fs: float) -> dict:
    """Pick a Welch segment length for estimating a PSD AT ``f_target``.

    Welch resolves ``fs/nperseg``, not ``fs/n_samples``: the record's Fourier
    floor is reached only by a single-segment periodogram, which has 100%
    relative error. This picks the SHORTEST segment (hence the most averaging)
    whose bin spacing puts ``f_target`` at least ``WELCH_BIN_MARGIN`` bins above
    DC, falling back to progressively longer segments and finally to the whole
    record.

    Returns a dict with ``nperseg``, ``n_segments`` (50% overlap),
    ``f_bin_hz`` = fs/nperseg, ``bins_below_target``, ``resolved``, and
    ``reason`` when unresolved. ``nperseg`` is ``None`` when ``f_target`` lies
    below the record's Fourier floor entirely -- there the honest answer is no
    answer, since interpolating there would clamp to the lowest observed bin and
    return a confident number for a frequency never observed.
    """
    n_samples = int(n_samples)
    fs = float(fs)
    f_target = float(f_target)
    floor = fs / max(n_samples, 1)
    if not (f_target > 0.0) or n_samples < 4:
        return {"nperseg": None, "n_segments": 0, "f_bin_hz": float("nan"),
                "bins_below_target": 0.0, "resolved": False,
                "reason": "degenerate record"}
    if f_target < floor:
        return {
            "nperseg": None, "n_segments": 0, "f_bin_hz": floor,
            "bins_below_target": f_target / floor, "resolved": False,
            "reason": (
                f"f={f_target:.3g} Hz is below the record's Fourier floor "
                f"{floor:.3g} Hz (= fs/N, fs={fs:.4g} Hz, N={n_samples}); a "
                f"record of t_slow >= {math.ceil(fs / f_target):d} samples is "
                f"needed to observe it at all"
            ),
        }

    # Candidate segment lengths: powers of two up to the record, plus the record.
    candidates: list[int] = []
    p = 256
    while p <= n_samples:
        candidates.append(p)
        p *= 2
    if not candidates or candidates[-1] != n_samples:
        candidates.append(n_samples)

    def _segments(nperseg: int) -> int:
        # scipy.signal.welch default noverlap = nperseg//2.
        return max(int(2 * n_samples / nperseg) - 1, 1)

    best = None
    for nperseg in candidates:
        f_bin = fs / nperseg
        bins = f_target / f_bin
        n_seg = _segments(nperseg)
        if bins >= WELCH_BIN_MARGIN and n_seg >= WELCH_MIN_SEGMENTS:
            best = (nperseg, f_bin, bins, n_seg, True, "")
            break
    if best is None:
        # Nothing clears both bars. Take the shortest segment that at least puts
        # the target above its own bin spacing, so the number is a measurement
        # of the requested frequency even though it is under-averaged.
        for nperseg in candidates:
            f_bin = fs / nperseg
            bins = f_target / f_bin
            n_seg = _segments(nperseg)
            if bins >= 1.0:
                reason = []
                if bins < WELCH_BIN_MARGIN:
                    reason.append(
                        f"target sits {bins:.2f} bins above DC "
                        f"(< {WELCH_BIN_MARGIN:g} required)")
                if n_seg < WELCH_MIN_SEGMENTS:
                    reason.append(
                        f"only {n_seg} Welch segment(s) "
                        f"(< {WELCH_MIN_SEGMENTS} required)")
                best = (nperseg, f_bin, bins, n_seg, False, "; ".join(reason))
                break
    if best is None:
        return {"nperseg": None, "n_segments": 0, "f_bin_hz": floor,
                "bins_below_target": f_target / floor, "resolved": False,
                "reason": "no segment length places the target above DC"}

    nperseg, f_bin, bins, n_seg, resolved, reason = best
    return {"nperseg": int(nperseg), "n_segments": int(n_seg),
            "f_bin_hz": float(f_bin), "bins_below_target": float(bins),
            "resolved": bool(resolved), "reason": reason}


def _psd_at(f: np.ndarray, s: np.ndarray, f_target: float) -> float:
    """Log-log interpolated PSD at ``f_target``, NaN outside the observed span.

    ``noise_metrology.psd_at_offset`` uses ``np.interp``, which CLAMPS to the
    edge value outside its table -- so asking it for 1 kHz on a record whose
    lowest bin is 123 kHz returns the 123 kHz value with no indication that
    anything went wrong. This refuses instead.
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    keep = (f > 0) & np.isfinite(s) & (s > 0)
    if keep.sum() < 2:
        return float("nan")
    fk, sk = f[keep], s[keep]
    if f_target < fk.min() or f_target > fk.max():
        return float("nan")
    return float(10.0 ** np.interp(math.log10(f_target),
                                   np.log10(fk), np.log10(sk)))


def _band_rms(f: np.ndarray, s: np.ndarray, lo: float, hi: float) -> float:
    """sqrt(integral of one-sided S over [lo, hi]); NaN if the band is empty."""
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    mask = (f >= lo) & (f <= hi) & np.isfinite(s)
    if mask.sum() < 2:
        return float("nan")
    return float(math.sqrt(max(float(np.trapezoid(s[mask], f[mask])), 0.0)))


def _blocked_sem(values: np.ndarray, stat, n_blocks: int = UNCERTAINTY_BLOCKS) -> float:
    """Within-record uncertainty of ``stat`` via contiguous block splitting.

    A correlated record does not carry N independent samples, so the iid SEM
    would understate the uncertainty. Splitting into contiguous blocks and
    taking the block-to-block SEM keeps whatever correlation is shorter than a
    block inside it.
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n < 2 * n_blocks:
        return float("nan")
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    per = np.array([stat(v[a:b]) for a, b in zip(edges[:-1], edges[1:])],
                   dtype=np.float64)
    per = per[np.isfinite(per)]
    if per.size < 2:
        return float("nan")
    return float(np.std(per, ddof=1) / math.sqrt(per.size))


# ---------------------------------------------------------------------------
# Observables -- each is (solution, ctx) -> (values, unit, uncertainty)
# ---------------------------------------------------------------------------
def obs_u_int_rms_frac(solution, ctx: RecordContext):
    """std(U_int)/mean(U_int) over the record [dimensionless]."""
    u = np.asarray(solution["U_int_history"], dtype=np.float64).ravel()
    mean = float(np.mean(u))
    if not np.isfinite(mean) or mean == 0.0:
        ctx.notes["value"] = {"resolved": False,
                              "reason": "mean(U_int) is zero or non-finite"}
        return {"value": None}, "dimensionless", {"value": None}

    def _frac(x):
        m = float(np.mean(x))
        return float(np.std(x)) / m if m else float("nan")

    value = _frac(u)
    ctx.notes["value"] = {
        "resolved": True,
        "n_samples": int(u.size),
        "mean_U_int_J": mean,
        "record_duration_s": ctx.duration_s,
    }
    return ({"value": value}, "dimensionless",
            {"value": _blocked_sem(u, _frac)})


def obs_s_rep(solution, ctx: RecordContext):
    """S_rep(f) at the report frequencies [Hz^2/Hz], tape-model fit.

    Reports the tape-model number as specified, and alongside it (in
    ``ctx.notes``) the pairwise-difference estimate and whether the cell sits in
    the tape fit's documented common-mode-dominated failure regime -- see the
    module docstring, "S_rep estimator caveat".
    """
    probes = np.asarray(solution["mode_probe_history"])
    if probes.ndim == 3:
        probes = probes[0]
    phases = unwrapped_phases(probes)
    mus = np.asarray(ctx.probe_mus, dtype=np.float64)
    fs = 1.0 / ctx.t_r
    # instantaneous_frequency differences the phase, costing one sample.
    n_freq = phases.shape[0] - 1

    # Different targets often land on the same segment length; a tape fit over a
    # 2e7-sample record is not something to redo four times for nothing.
    fit_cache: dict[int, tuple] = {}

    def _estimates(nperseg: int):
        if nperseg not in fit_cache:
            fit = tape_model_fit(phases, mus, ctx.t_r, nperseg=nperseg)
            # Pairwise difference estimator: the common mode cancels
            # DETERMINISTICALLY before any spectral estimate is formed.
            phi_rep = rep_rate_phase(phases, ctx.probe_mus, 0,
                                     len(ctx.probe_mus) - 1)
            f_p, s_p = frequency_noise_psd(phi_rep, ctx.t_r, nperseg=nperseg)
            fit_cache[nperseg] = (fit, f_p, s_p)
        return fit_cache[nperseg]

    values: dict[str, float | None] = {}
    uncert: dict[str, float | None] = {}
    for f_t in S_REP_FREQS_HZ:
        label = _freq_label(f_t)
        seg = welch_segmentation(f_t, n_freq, fs)
        note = {"target_hz": f_t, **seg}
        if seg["nperseg"] is None:
            values[label] = None
            uncert[label] = None
            ctx.notes[label] = {**note, "estimator": "tape_model_fit"}
            continue

        fit, f_p, s_p = _estimates(int(seg["nperseg"]))
        s_rep = _psd_at(fit["f"], fit["S_rep"], f_t)
        s_c = _psd_at(fit["f"], np.abs(fit["S_c"]), f_t)
        s_rep_pair = _psd_at(f_p, s_p, f_t)

        mu_max2 = float(np.max(mus ** 2))
        ref = s_rep_pair if np.isfinite(s_rep_pair) and s_rep_pair > 0 else s_rep
        dominated = bool(
            np.isfinite(s_c) and np.isfinite(ref) and ref > 0
            and s_c > mu_max2 * ref * math.sqrt(max(seg["n_segments"], 1))
        )
        # Welch chi^2 relative error of a PSD averaged over n_seg segments.
        rel = 1.0 / math.sqrt(max(seg["n_segments"], 1))
        values[label] = None if not np.isfinite(s_rep) else float(s_rep)
        uncert[label] = (None if not np.isfinite(s_rep)
                         else float(abs(s_rep) * rel))
        ctx.notes[label] = {
            **note,
            "estimator": "tape_model_fit",
            "s_rep_pairwise": (None if not np.isfinite(s_rep_pair)
                               else float(s_rep_pair)),
            "s_rep_pairwise_uncertainty": (None if not np.isfinite(s_rep_pair)
                                           else float(abs(s_rep_pair) * rel)),
            "S_c": None if not np.isfinite(s_c) else float(s_c),
            "common_mode_dominated": dominated,
            "caveat": (
                "tape_model_fit S_rep is not meaningful here: the fitted mu^2 "
                "curvature is below the per-bin chi^2 noise of the common mode. "
                "Use s_rep_pairwise." if dominated else ""
            ),
        }
    return values, "Hz^2/Hz", uncert


def obs_line_linewidth(solution, ctx: RecordContext):
    """Beta-separation-line effective FWHM linewidth per probe mode [Hz]."""
    probes = np.asarray(solution["mode_probe_history"])
    if probes.ndim == 3:
        probes = probes[0]
    phases = unwrapped_phases(probes)
    # The beta-line integral wants the widest band the record supports, so the
    # segmentation is chosen for averaging quality (WELCH_MIN_SEGMENTS
    # segments), not for any single target frequency.
    n_freq = phases.shape[0] - 1
    nperseg = max(min(n_freq, int(2 * n_freq / (WELCH_MIN_SEGMENTS + 1))), 8)
    n_seg = max(int(2 * n_freq / nperseg) - 1, 1)
    f, s_mu = frequency_noise_psd(phases, ctx.t_r, nperseg=nperseg)

    values: dict[str, float | None] = {}
    uncert: dict[str, float | None] = {}
    for j, mu in enumerate(ctx.probe_mus):
        label = f"mu={int(mu)}"
        s_j = np.asarray(s_mu[:, j], dtype=np.float64)
        fwhm = effective_linewidth(f, s_j)
        beta = 8.0 * math.log(2.0) * f / math.pi ** 2
        crossed = bool(np.any(s_j > beta))
        values[label] = float(fwhm)
        # FWHM = sqrt(8 ln2 * A) with A linear in the PSD, so a relative PSD
        # error eps propagates to eps/2 on the linewidth.
        uncert[label] = float(0.5 * fwhm / math.sqrt(max(n_seg, 1)))
        ctx.notes[label] = {
            "resolved": crossed,
            "band_hz": [float(f.min()), float(f.max())],
            "nperseg": int(nperseg),
            "n_segments": int(n_seg),
            "record_fourier_floor_hz": ctx.fourier_floor_hz,
            "reason": "" if crossed else (
                "S_dnu never crosses the beta-separation line on the observed "
                "band, so the linewidth is set by the (unresolved) Lorentzian "
                "wings and effective_linewidth returns 0.0 by convention"),
        }
    return values, "Hz", uncert


def obs_timing_jitter(solution, ctx: RecordContext):
    """Pulse timing jitter integrated over the requested band [fs].

    Integrates the one-sided S_dt(f) from ``noise_metrology.timing_jitter`` over
    the intersection of ``JITTER_BAND_HZ`` with the band the snapshot record
    actually resolves, and reports the ACHIEVED band. At the production fast
    record (t_slow = 2e5) the resolved band starts near 123 kHz, so the 1 kHz
    end of the requested band is not covered and the cell says so rather than
    quietly integrating a narrower band under the requested label.
    """
    snaps = np.asarray(solution["E_snapshots"])
    if snaps.ndim == 3:
        snaps = snaps[0]
    if snaps.shape[0] < 16:
        ctx.notes["value"] = {
            "resolved": False,
            "reason": f"only {snaps.shape[0]} snapshots; need >= 16 for a PSD",
        }
        return {"value": None}, "fs", {"value": None}

    tj = metrology_timing_jitter(snaps, ctx.snapshot_interval, ctx.t_r)
    f, s_dt = np.asarray(tj["f"]), np.asarray(tj["S_dt"])
    lo_req, hi_req = JITTER_BAND_HZ
    lo = max(lo_req, float(f.min()))
    hi = min(hi_req, float(f.max()))
    if not (hi > lo):
        ctx.notes["value"] = {
            "resolved": False,
            "band_hz_requested": [lo_req, hi_req],
            "band_hz_achieved": None,
            "reason": (
                f"requested band [{lo_req:.3g}, {hi_req:.3g}] Hz does not "
                f"intersect the resolved band [{f.min():.3g}, {f.max():.3g}] Hz"),
        }
        return {"value": None}, "fs", {"value": None}

    rms_s = _band_rms(f, s_dt, lo, hi)
    n_in = int(((f >= lo) & (f <= hi)).sum())
    covers = bool(lo <= lo_req * 1.001 and hi >= hi_req * 0.999)
    ctx.notes["value"] = {
        "resolved": covers,
        "band_hz_requested": [lo_req, hi_req],
        "band_hz_achieved": [lo, hi],
        "n_bins_in_band": n_in,
        "snapshot_rate_hz": 1.0 / (ctx.snapshot_interval * ctx.t_r),
        "jitter_rms_full_record_s": float(tj["jitter_rms_s"]),
        "reason": "" if covers else (
            f"achieved band [{lo:.4g}, {hi:.4g}] Hz does not cover the "
            f"requested [{lo_req:.3g}, {hi_req:.3g}] Hz; the low end is limited "
            f"by the record duration ({ctx.duration_s * 1e6:.3g} us, Fourier "
            f"floor {ctx.fourier_floor_hz:.4g} Hz) and the high end by the "
            f"snapshot rate"),
    }
    value = None if not np.isfinite(rms_s) else float(rms_s * 1e15)
    unc = (None if value is None
           else float(value * 0.5 / math.sqrt(max(n_in, 1))))
    return {"value": value}, "fs", {"value": unc}


def obs_step_jitter(solution, ctx: RecordContext):
    """Per-seed soliton-step switching detuning per level [kappa].

    ``solution`` is a ``run_detuning_sweep`` result. The per-seed value is the
    switching detuning itself, from the imported W2 estimator
    ``noise_validation_campaign.level_crossing``; the JITTER is the seed-to-seed
    STD of that sample, which is why this observable's budget statistic is
    ``std`` and not ``mean`` (the mean is the step LOCATION, and
    location-minus-deterministic is a bias).
    """
    count = np.asarray(solution["soliton_count"])
    det_k = np.asarray(ctx.det_k if ctx.det_k is not None
                       else solution["dw_over_kappa"], dtype=np.float64)
    values: dict[str, float | None] = {}
    uncert: dict[str, float | None] = {}
    for k in range(int(ctx.n_solitons) - 1, -1, -1):
        label = f"N<={k}"
        cross = level_crossing(det_k, count, k)
        values[label] = cross
        # Within-record uncertainty of a level crossing is the grid spacing:
        # the crossing is localised only to the step that produced it.
        spacing = (float(abs(det_k[1] - det_k[0])) if det_k.size > 1
                   else float("nan"))
        uncert[label] = None if cross is None else spacing / math.sqrt(12.0)
        ctx.notes[label] = {
            "resolved": cross is not None,
            "grid_spacing_kappa": spacing,
            "final_soliton_count": int(count[-1]),
            "reason": "" if cross is not None else (
                f"soliton count never dropped to <= {k} within the swept range "
                f"[{det_k.min():.3g}, {det_k.max():.3g}] kappa"),
        }
    return values, "kappa", uncert


#: name -> (function, which record feeds it, which statistic is the budget
#: number). The record assignment is the one specified for the benchmark; what
#: each record can actually RESOLVE is decided per cell by
#: :func:`welch_segmentation` and recorded in the JSON.
OBSERVABLES: dict[str, dict] = {
    "u_int_rms_frac": {"fn": obs_u_int_rms_frac, "record": "fast",
                       "statistic": "mean", "unit": "dimensionless",
                       "preferred_point": "value"},
    "s_rep": {"fn": obs_s_rep, "record": "both",
              "statistic": "mean", "unit": "Hz^2/Hz",
              "preferred_point": _freq_label(S_REP_FREQS_HZ[-1])},
    "line_linewidth": {"fn": obs_line_linewidth, "record": "slow",
                       "statistic": "mean", "unit": "Hz",
                       "preferred_point": "mu=0"},
    # Evaluated on BOTH records and resolved per cell, not because it is
    # convenient but because it is the only way the number means anything: the
    # requested band starts at 1 kHz, and the fast record's Fourier floor is
    # 123 kHz, so the fast record contributes at most a sliver of the top
    # decade. The slow record (snapshot_interval 2000 -> 12.3 MHz snapshot rate,
    # 10^4 snapshots) covers ~10 kHz - 1 MHz. Whichever record covered more of
    # the requested band is named in the cell.
    "timing_jitter": {"fn": obs_timing_jitter, "record": "both",
                      "statistic": "mean", "unit": "fs",
                      "preferred_point": "value"},
    "step_jitter": {"fn": obs_step_jitter, "record": "staircase",
                    "statistic": "std", "unit": "kappa"},
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _bootstrap_ci(sample: np.ndarray, stat, rng) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of ``stat`` over ``BOOTSTRAP_RESAMPLES``."""
    v = np.asarray(sample, dtype=np.float64)
    if v.size < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, v.size, size=(BOOTSTRAP_RESAMPLES, v.size))
    draws = np.array([stat(v[row]) for row in idx], dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    if draws.size < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * BOOTSTRAP_ALPHA / 2,
                                   100 * (1 - BOOTSTRAP_ALPHA / 2)])
    return float(lo), float(hi)


def _summarise(per_seed: list, seeds: list[int]) -> dict:
    """mean / SEM / std of a per-seed sample, with bootstrap CIs on both.

    Non-finite entries are dropped and counted rather than propagated: one
    dropped realization must not turn a whole budget cell into NaN, but the
    reader has to be able to see that it happened.
    """
    raw = np.array([np.nan if x is None else float(x) for x in per_seed],
                   dtype=np.float64)
    good = np.isfinite(raw)
    v = raw[good]
    used_seeds = [int(s) for s, g in zip(seeds, good) if g]
    out = {
        "n_seeds_requested": int(raw.size),
        "n_seeds_used": int(v.size),
        "seeds_used": used_seeds,
        "per_seed": [None if not np.isfinite(x) else float(x) for x in raw],
        "mean": float(np.mean(v)) if v.size else None,
        "sem": (float(np.std(v, ddof=1) / math.sqrt(v.size))
                if v.size > 1 else (0.0 if v.size == 1 else None)),
        "std": (float(np.std(v, ddof=1)) if v.size > 1
                else (0.0 if v.size == 1 else None)),
        "median": float(np.median(v)) if v.size else None,
        "min": float(v.min()) if v.size else None,
        "max": float(v.max()) if v.size else None,
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lo, hi = _bootstrap_ci(v, lambda x: float(np.mean(x)), rng)
    out["ci95_mean"] = [None if not np.isfinite(lo) else lo,
                        None if not np.isfinite(hi) else hi]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    lo, hi = _bootstrap_ci(v, lambda x: float(np.std(x, ddof=1)), rng)
    out["ci95_std"] = [None if not np.isfinite(lo) else lo,
                       None if not np.isfinite(hi) else hi]
    out["bootstrap_resamples"] = BOOTSTRAP_RESAMPLES
    return out


# ---------------------------------------------------------------------------
# Solver units
# ---------------------------------------------------------------------------
def _metrology_unit(spec: RecordSpec, cav, cfg_path: Path, seed: int,
                    probe_mus: tuple[int, ...]):
    """Settle a single DKS under the channel set, then record it.

    Settling runs under the SAME noise configuration as the record: settling
    noise-free and only then switching the channels on would put a transient
    into the first decade of every PSD. The record uses a distinct trajectory
    key so it is an independent stretch of the same noise process.
    """
    dw = OPERATING_DW_KAPPA * cav.kappa
    with _quiet():
        # Inside _quiet: the analytic sech ansatz overflows cosh in its far
        # tails by construction (dks_access.py), and that RuntimeWarning is not
        # this driver's news to report once per unit.
        seed_field = sech_soliton_seed(dw, cav, n_tau=spec.n_tau, pin=PIN_W)
        s0 = _run(dw, int(spec.settle_rt), cav, e0=seed_field, seed=int(seed),
                  n_tau=spec.n_tau, pin=PIN_W,
                  snapshot_interval=int(spec.settle_rt),
                  config_path=str(cfg_path), **PRODUCTION_NUMERICS)
        e0 = np.asarray(s0["e_final"])[0]
        dt0 = float(np.asarray(s0["delta_t_final"]).ravel()[0])
        sol = _run(dw, int(spec.t_slow), cav, e0=e0, delta_t0=dt0,
                   seed=int(seed) * 2 + 1, n_tau=spec.n_tau, pin=PIN_W,
                   snapshot_interval=int(spec.snapshot_interval),
                   config_path=str(cfg_path),
                   mode_probe_indices=tuple(int(m) for m in probe_mus),
                   **PRODUCTION_NUMERICS)
    return sol


#: Symmetry-breaking soliton placement for the staircase. HELD FIXED across the
#: ensemble, unlike the W2 campaign which varies it for ensemble diversity. The
#: placement is a deterministic initial condition, so varying it makes the
#: seed-to-seed spread of the switching detuning a mixture of noise-induced
#: jitter and initial-condition spread -- and the deterministic ``all_off``
#: reference then shows a non-zero "jitter" of its own (measured: 0.33 kappa),
#: which every OAT contribution would silently subtract. With it fixed, the only
#: thing varying between seeds is the noise realization, ``all_off`` has exactly
#: zero spread by construction, and the step jitter is purely noise-induced.
STAIRCASE_POSITION_SEED = 1


def _staircase_unit(swp: dict, cav, cfg_path: Path, seed_index: int):
    """One warm-continuation staircase realization under the channel set.

    Only the noise trajectory key varies with ``seed_index``; the soliton
    placement is pinned (see ``STAIRCASE_POSITION_SEED``). A given index
    therefore reuses one initial condition across every channel set -- common
    random numbers for the staircase too.
    """
    cfg = SweepConfig(**{**swp, "seed": SEED_BASE + seed_index,
                         "position_seed": STAIRCASE_POSITION_SEED})
    with _quiet():
        return run_detuning_sweep(cav, cfg, config_path=str(cfg_path))


# ---------------------------------------------------------------------------
# Incremental store
# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _clean(obj):
    """Replace non-finite floats with None so the JSON is valid everywhere."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


def _atomic_json(path: Path, payload: dict) -> None:
    """Write JSON via a tmp sibling + os.replace, so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_clean(payload), fh, indent=2, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
def _combine_err(*errs):
    """Quadrature-combined uncertainty of a difference, None-safe."""
    vals = [float(x) for x in errs
            if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return float(math.sqrt(sum(x * x for x in vals)))


def interaction_commentary(residual: dict, observable: str) -> str:
    """Generated prose explaining what a large interaction residual means here.

    A residual is not a bug report. The OAT grouping is non-overlapping in
    SWITCHES but not in RANDOM VARIABLES: trn, pyro_eo and fsr share one
    delta_T(t) realization, so ``dT_family`` is deliberately the grouped row --
    and even so, the soliton is a nonlinear system in which channels do not
    superpose. This says that in each table.
    """
    parts = [
        f"Interaction residual for '{observable}': "
        f"X[all_on] - X[all_off] - sum over {{{', '.join(OAT_GROUPING)}}} of "
        f"(X[set] - X[all_off]).",
        "A LARGE residual is expected, not anomalous. Three reasons (their "
        "relative size is not asserted here -- compare the OAT rows to see it):",
        "(1) The channels do not superpose. The LLE is nonlinear and the "
        "soliton is an attractor: two channels driving it together move the "
        "operating point in a way neither does alone, so the observables are "
        "not additive functionals of the drives.",
        "(2) trn, pyro_eo and fsr are ONE temperature realization delta_T(t) "
        "seen through three coupling coefficients, not three independent "
        "sources. That is exactly why the grouping uses the dT_family row "
        "rather than summing trn, pyro_eo and fsr separately -- summing them "
        "would triple-count one random variable.",
        "(3) all_on additionally carries tccr, which is NOT in the grouping. "
        "On this SiN device eo_r33_m_per_v = 0 so sigma_tccr = 0 and tccr "
        "contributes exactly zero, making this term vanish here -- but on a "
        "chi(2) platform (TFLN) it would not, and the grouping would need a "
        "tccr row before the residual could be read as pure interaction.",
    ]
    finite = [v for v in residual.values()
              if isinstance(v, (int, float)) and math.isfinite(v)]
    if finite:
        parts.append(
            f"Residual magnitude across the {len(finite)} resolved point(s) of "
            f"this observable: min {min(finite):.4g}, max {max(finite):.4g}.")
    return " ".join(parts)


def _derive(cells: dict, observable: str, statistic: str) -> dict:
    """OAT contributions, LOO contributions and the interaction residual."""
    def cell(set_name):
        return cells.get(set_name, {}).get(observable)

    points: list[str] = []
    for set_name in CHANNEL_SETS:
        c = cell(set_name)
        if c:
            for p in c.get("points", {}):
                if p not in points:
                    points.append(p)

    off = cell("all_off")
    on = cell("all_on")

    oat: dict[str, dict] = {}
    loo: dict[str, dict] = {}
    residual: dict[str, float | None] = {}
    residual_err: dict[str, float | None] = {}

    # X[all_on] - X[all_off] is the TOTAL, not a one-at-a-time contribution;
    # it is reported separately so the OAT block holds only real OAT rows.
    total: dict[str, float | None] = {}
    for p in points:
        a, b = _stat_of_point(on, p, statistic), _stat_of_point(off, p, statistic)
        total[p] = None if (a is None or b is None) else a - b

    for set_name in CHANNEL_SETS:
        if set_name in ("all_off", "all_on"):
            continue
        c = cell(set_name)
        if not c:
            continue
        if not set_name.startswith("loo_"):
            row, err = {}, {}
            for p in points:
                a = _stat_of_point(c, p, statistic)
                b = _stat_of_point(off, p, statistic)
                row[p] = None if (a is None or b is None) else a - b
                err[p] = _combine_err(_err_of_point(c, p, statistic),
                                      _err_of_point(off, p, statistic))
            oat[set_name] = {"contribution": row, "uncertainty": err}
        else:
            target = set_name[len("loo_"):]
            row, err = {}, {}
            for p in points:
                a = _stat_of_point(on, p, statistic)
                b = _stat_of_point(c, p, statistic)
                row[p] = None if (a is None or b is None) else a - b
                err[p] = _combine_err(_err_of_point(on, p, statistic),
                                      _err_of_point(c, p, statistic))
            loo[target] = {"contribution": row, "uncertainty": err,
                           "channel_set": set_name}

    for p in points:
        total_a = _stat_of_point(on, p, statistic)
        total_b = _stat_of_point(off, p, statistic)
        if total_a is None or total_b is None:
            residual[p] = None
            residual_err[p] = None
            continue
        total_value = total_a - total_b
        parts, errs, missing = [], [], []
        for g in OAT_GROUPING:
            contrib = oat.get(g, {}).get("contribution", {}).get(p)
            if contrib is None:
                missing.append(g)
            else:
                parts.append(contrib)
                errs.append(oat[g]["uncertainty"].get(p))
        if missing:
            residual[p] = None
            residual_err[p] = None
            continue
        residual[p] = float(total_value - sum(parts))
        residual_err[p] = _combine_err(
            _err_of_point(on, p, statistic),
            _err_of_point(off, p, statistic),
            *errs,
        )

    return {
        "budget_statistic": statistic,
        "points": points,
        "total_contribution": total,
        "oat_contribution": oat,
        "loo_contribution": loo,
        "interaction_residual": residual,
        "interaction_residual_uncertainty": residual_err,
        "interaction_grouping": list(OAT_GROUPING),
        "commentary": interaction_commentary(residual, observable),
    }


def _stat_of_point(cell, point, statistic):
    if not cell:
        return None
    stats = cell.get("points", {}).get(point, {}).get("statistics")
    if not stats:
        return None
    v = stats.get(statistic)
    return None if v is None or not math.isfinite(float(v)) else float(v)


def _err_of_point(cell, point, statistic):
    """Uncertainty attached to the budget statistic of one point.

    For a ``mean`` statistic that is the SEM. For a ``std`` statistic (the
    step-jitter observable) the SEM is the wrong bar entirely -- it is the
    uncertainty of the mean, not of the spread -- so the bootstrap CI on the std
    is used, halved to a one-sigma-like number so it combines in quadrature with
    the others.
    """
    if not cell:
        return None
    stats = cell.get("points", {}).get(point, {}).get("statistics")
    if not stats:
        return None
    if statistic == "std":
        lo, hi = (stats.get("ci95_std") or [None, None])
        if lo is None or hi is None:
            return None
        return float(abs(float(hi) - float(lo)) / 2.0)
    v = stats.get("sem")
    return None if v is None or not math.isfinite(float(v)) else float(v)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def _fmt_num(x: float, tex: bool, sig: int = 4) -> str:
    """One number, in scientific form only where it earns it.

    The exponent is chosen from the number ITSELF. Rendering an uncertainty on
    the value's exponent -- which is the tempting shortcut -- produces mantissas
    like ``5067.79e-6`` the moment the error and the value differ by a few
    decades, which is exactly what happens when a contribution is consistent
    with zero.
    """
    a = abs(x)
    if a != 0.0 and (a < 1e-3 or a >= 1e4):
        mant, exp = f"{x:.{sig - 1}e}".split("e")
        exp = int(exp)
        return rf"{mant}\times10^{{{exp}}}" if tex else f"{mant}e{exp}"
    return f"{x:.{sig}g}"


def _fmt(value, err=None, tex=False) -> str:
    """``value ± err`` for a table cell; an em dash when there is no number."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return r"---" if tex else "—"
    body = _fmt_num(float(value), tex)
    if err is not None and math.isfinite(float(err)) and float(err) != 0.0:
        e = _fmt_num(float(err), tex, sig=2)
        body += (rf"\,{{\pm}}\," + e) if tex else f" ± {e}"
    return f"${body}$" if tex else body


def _pick_points(cells: dict, observable: str) -> str | None:
    """The headline point of a multi-point observable.

    The point with the most channel sets carrying a finite value wins, so the
    summary table shows something real rather than the first declared point.
    Ties -- including the all-null tie a short record produces -- go to the
    observable's ``preferred_point`` (the top S_rep decade, the mu=0 comb line),
    which is the one a reader expects to see quoted.
    """
    counts: dict[str, int] = {}
    for set_name in CHANNEL_SETS:
        c = cells.get(set_name, {}).get(observable)
        if not c:
            continue
        for p, blob in c.get("points", {}).items():
            stats = blob.get("statistics") or {}
            counts[p] = counts.get(p, 0) + (1 if stats.get("mean") is not None
                                            else 0)
    if not counts:
        return None
    preferred = OBSERVABLES.get(observable, {}).get("preferred_point")
    best_n = max(counts.values())
    tied = [p for p, n in counts.items() if n == best_n]
    if preferred in tied:
        return preferred
    return tied[0]


def _table_columns(cells: dict) -> list[tuple[str, str, str, str]]:
    """(observable, point, header, unit) for the headline table."""
    cols = []
    for name, spec in OBSERVABLES.items():
        point = _pick_points(cells, name)
        if point is None:
            continue
        header = name if point == "value" else f"{name} ({point})"
        cols.append((name, point, header, spec["unit"]))
    return cols


def _render_markdown(report: dict) -> str:
    cells = report["cells"]
    cols = _table_columns(cells)
    lines = [
        "# Stochastic-LLE noise budget",
        "",
        f"- generated: `{report['provenance']['generated_utc']}`",
        f"- git commit: `{report['provenance']['git_commit']}`",
        f"- seeds per cell: {report['run']['seeds']} "
        f"(common random numbers: {report['run']['seed_list']})",
        f"- quick mode: {report['run']['quick']}",
        f"- thermal_feedback pinned ON in every cell: "
        f"{report['run']['thermal_feedback_pinned']}",
        f"- amplitude preset: {report['run']['amplitude_preset']}",
        "",
        "## Budget table",
        "",
        "Cell = mean ± SEM over seeds, except `step_jitter`, whose budget "
        "statistic is the seed-to-seed **std** (the jitter itself).",
        "",
    ]
    head = "| channel set | " + " | ".join(
        f"{h} [{u}]" for _, _, h, u in cols) + " |"
    lines += [head, "|" + "---|" * (len(cols) + 1)]
    for set_name in CHANNEL_SETS:
        row = [f"`{set_name}`"]
        for obs, point, _h, _u in cols:
            stat = OBSERVABLES[obs]["statistic"]
            c = cells.get(set_name, {}).get(obs)
            v = _stat_of_point(c, point, stat)
            e = _err_of_point(c, point, stat)
            row.append(_fmt(v, e))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Record length per cell", "",
              "| observable | point | record | t_slow [RT] | duration | "
              "Fourier floor |", "|---|---|---|---|---|---|"]
    for obs, point, _h, _u in cols:
        for set_name in CHANNEL_SETS:
            c = cells.get(set_name, {}).get(obs)
            if not c:
                continue
            blob = c.get("points", {}).get(point, {})
            floor = blob.get("record_fourier_floor_hz")
            lines.append(
                f"| {obs} | {point} | {blob.get('record')} | "
                f"{blob.get('t_slow')} | "
                f"{(blob.get('record_duration_s') or 0) * 1e6:.4g} µs | "
                f"{'n/a' if floor is None else f'{floor:.4g} Hz'} |")
            break

    for obs in OBSERVABLES:
        d = report["derived"].get(obs)
        if not d:
            continue
        stat = d["budget_statistic"]
        lines += ["", f"## {obs} — contributions ({stat})", "",
                  "| set | OAT: X[set]−X[all_off] | LOO: X[all_on]−X[all_on∖set] |",
                  "|---|---|---|"]
        for set_name in CHANNEL_SETS:
            if set_name in ("all_off", "all_on") or set_name.startswith("loo_"):
                continue
            point = _pick_points(cells, obs)
            o = d["oat_contribution"].get(set_name, {})
            lo = d["loo_contribution"].get(set_name, {})
            lines.append(
                f"| `{set_name}` | "
                f"{_fmt(o.get('contribution', {}).get(point), o.get('uncertainty', {}).get(point))} | "
                f"{_fmt(lo.get('contribution', {}).get(point), lo.get('uncertainty', {}).get(point)) if lo else '—'} |")
        point = _pick_points(cells, obs)
        lines += ["",
                  f"**Interaction residual** ({point}): "
                  f"{_fmt(d['interaction_residual'].get(point), d['interaction_residual_uncertainty'].get(point))}",
                  "", d["commentary"]]
    return "\n".join(lines) + "\n"


def _render_tex(report: dict) -> str:
    cells = report["cells"]
    cols = _table_columns(cells)
    spec = "l" + "r" * len(cols)
    out = [
        "% Stochastic-LLE noise budget -- generated by analysis/noise_budget.py",
        f"% git {report['provenance']['git_commit']} "
        f"{report['provenance']['generated_utc']}",
        "% Requires \\usepackage{booktabs}. Ready to \\input.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Noise budget of the stochastic Lugiato--Lefever model at "
        f"$\\delta\\omega = {OPERATING_DW_KAPPA:g}\\kappa$. "
        f"Cells are the mean over {report['run']['seeds']} realizations "
        "$\\pm$ the standard error, sharing one seed list across channel sets "
        "(common random numbers); \\texttt{step\\_jitter} reports the "
        "seed-to-seed standard deviation, which is the jitter itself. "
        "Channels \\texttt{pyro\\_eo} and \\texttt{fsr} are driven by the same "
        "$\\delta T(t)$ realization as \\texttt{trn} and are therefore run "
        "with it.}",
        "  \\label{tab:noise-budget}",
        f"  \\begin{{tabular}}{{{spec}}}",
        "    \\toprule",
        "    channel set & " + " & ".join(
            _tex_header(h, u) for _, _, h, u in cols) + r" \\",
        "    \\midrule",
    ]
    for set_name in CHANNEL_SETS:
        if set_name == "all_on":
            out.append("    \\midrule")
        row = [_tt(set_name)]
        for obs, point, _h, _u in cols:
            stat = OBSERVABLES[obs]["statistic"]
            c = cells.get(set_name, {}).get(obs)
            row.append(_fmt(_stat_of_point(c, point, stat),
                            _err_of_point(c, point, stat), tex=True))
        out.append("    " + " & ".join(row) + r" \\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]

    out += [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{One-at-a-time (OAT) and leave-one-out (LOO) contributions "
        "and the interaction residual, over the non-overlapping grouping "
        + _tex_set(OAT_GROUPING)
        + ". A large residual is expected: the channels do not superpose in a "
        "nonlinear attractor, and the $\\delta T$ family shares one "
        "realization.}",
        "  \\label{tab:noise-budget-contributions}",
        "  \\begin{tabular}{llrr}",
        "    \\toprule",
        "    observable & set & OAT & LOO \\\\",
        "    \\midrule",
    ]
    for obs in OBSERVABLES:
        d = report["derived"].get(obs)
        if not d:
            continue
        point = _pick_points(cells, obs)
        for set_name in _OAT_SPEC:
            o = d["oat_contribution"].get(set_name, {})
            lo = d["loo_contribution"].get(set_name, {})
            out.append(
                "    " + _tt(obs) + " & " + _tt(set_name) + " & "
                + _fmt(o.get("contribution", {}).get(point),
                       o.get("uncertainty", {}).get(point), tex=True) + " & "
                + (_fmt(lo.get("contribution", {}).get(point),
                        lo.get("uncertainty", {}).get(point), tex=True)
                   if lo else "---")
                + r" \\")
        out.append(
            "    " + _tt(obs) + " & "
            + r"\emph{residual} & \multicolumn{2}{r}{"
            + _fmt(d["interaction_residual"].get(point),
                   d["interaction_residual_uncertainty"].get(point), tex=True)
            + r"} \\")
        out.append("    \\midrule")
    if out[-1] == "    \\midrule":
        out.pop()
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def _tt(name: str) -> str:
    r"""``\texttt{...}`` with underscores escaped.

    A helper rather than an inline expression because an f-string expression
    cannot contain a backslash before Python 3.12.
    """
    escaped = str(name).replace("_", r"\_")
    return r"\texttt{" + escaped + "}"


def _tex_set(names) -> str:
    r"""``$\{a,\ b\}$`` -- note BOTH braces escaped; ``\{x}`` does not compile."""
    body = r",\ ".join(str(n).replace("_", r"\_") for n in names)
    return r"$\{" + body + r"\}$"


def _tex_header(header: str, unit: str) -> str:
    h = header.replace("_", r"\_").replace("<=", r"$\leq$")
    u = (unit.replace("^2", r"$^2$").replace("_", r"\_")
         .replace("kappa", r"$\kappa$"))
    return rf"\shortstack{{\texttt{{{h}}}\\{{[{u}]}}}}"


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def _unit_key(set_name: str, record: str, seed: int) -> str:
    return f"{set_name}|{record}|{seed}"


def _config_fingerprint(records: dict, staircase: dict,
                        set_names: list[str]) -> str:
    """Content hash of everything a stored solver unit depends on.

    ``--resume`` matches units by ``set|record|seed``, which says nothing about
    the record LENGTH, the staircase grid or the channel-set definitions behind
    them. Resuming a run whose settings changed would therefore silently blend
    two different experiments into one table -- every cell present, every number
    plausible, the budget meaningless. The fingerprint makes that a loud restart
    instead.
    """
    payload = json.dumps({
        "records": {k: v.as_dict() for k, v in sorted(records.items())},
        "staircase": dict(sorted(staircase.items())),
        "staircase_position_seed": STAIRCASE_POSITION_SEED,
        "channel_sets": {k: CHANNEL_SETS[k].to_dict() for k in sorted(set_names)},
        "probe_mus": list(PROBE_MUS),
        "s_rep_freqs_hz": list(S_REP_FREQS_HZ),
        "jitter_band_hz": list(JITTER_BAND_HZ),
        "numerics": dict(sorted(PRODUCTION_NUMERICS.items())),
        "operating_dw_over_kappa": float(OPERATING_DW_KAPPA),
        "pin_w": float(PIN_W),
        "seed_base": SEED_BASE,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_budget(out_dir: Path, *, seeds: int, quick: bool, resume: bool,
               only_observables: list[str] | None,
               only_channel_sets: list[str] | None) -> dict:
    """Run the whole budget and write every artifact. Returns the report dict."""
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "budget_raw.json"

    records = QUICK_RECORDS if quick else RECORDS
    staircase = QUICK_STAIRCASE if quick else STAIRCASE

    obs_names = list(only_observables or OBSERVABLES)
    unknown = [o for o in obs_names if o not in OBSERVABLES]
    if unknown:
        raise BudgetError(f"unknown --observable {unknown}; "
                          f"choose from {sorted(OBSERVABLES)}")
    set_names = list(only_channel_sets or CHANNEL_SETS)
    unknown = [s for s in set_names if s not in CHANNEL_SETS]
    if unknown:
        raise BudgetError(f"unknown --channel-set {unknown}; "
                          f"choose from {sorted(CHANNEL_SETS)}")

    # Which solver units the selected observables actually require.
    needed_records: set[str] = set()
    for name in obs_names:
        which = OBSERVABLES[name]["record"]
        if which == "both":
            needed_records |= {"fast", "slow"}
        elif which in records:
            needed_records.add(which)
    need_staircase = any(OBSERVABLES[n]["record"] == "staircase"
                         for n in obs_names)

    seed_list = [SEED_BASE + i for i in range(int(seeds))]
    stair_seed_idx = list(range(min(int(seeds), STEP_JITTER_MAX_SEEDS)))

    cav_base = load_cavity_params(CONFIG_PATH)
    t_r, kappa = cav_base.t_r, cav_base.kappa
    cav_cache: dict[int, object] = {}

    def cav_for(n_tau: int):
        if n_tau not in cav_cache:
            with _quiet():
                cav_cache[n_tau] = attach_dispersion(cav_base, int(n_tau))
        return cav_cache[n_tau]

    sidecar_dir = Path(tempfile.mkdtemp(prefix="noise_budget_cfgs_"))
    sidecars = {name: budget_sidecar(CHANNEL_SETS[name],
                                     sidecar_dir / f"{name}.yaml")
                for name in set_names}

    fingerprint = _config_fingerprint(records, staircase, set_names)
    store: dict = {}
    if resume and raw_path.exists():
        try:
            stored = json.load(open(raw_path, encoding="utf-8"))
            previous = (stored.get("meta") or {}).get("config_fingerprint")
            if previous is not None and previous != fingerprint:
                print(f"[budget] --resume: REFUSING to reuse {raw_path}: it was "
                      f"written with different run settings "
                      f"(fingerprint {previous[:12]} vs {fingerprint[:12]}). "
                      f"Record lengths, the staircase grid or the channel-set "
                      f"definitions changed, and mixing them would produce a "
                      f"complete but meaningless table. Starting fresh.")
            else:
                store = stored.get("units", {})
                print(f"[budget] --resume: {len(store)} completed unit(s) "
                      f"loaded from {raw_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[budget] --resume: could not read {raw_path} ({exc}); "
                  f"starting fresh")
            store = {}

    total_units = (len(set_names) * len(needed_records) * len(seed_list)
                   + (len(set_names) * len(stair_seed_idx) if need_staircase else 0))
    rt_per_unit = {r: records[r].t_slow + records[r].settle_rt
                   for r in needed_records}
    est_rt = (len(set_names) * len(seed_list) * sum(rt_per_unit.values())
              + (len(set_names) * len(stair_seed_idx)
                 * (staircase["settle_rt"] + staircase["n_steps"] * staircase["hold_rt"])
                 if need_staircase else 0))
    print("=" * 78)
    print("STOCHASTIC-LLE NOISE BUDGET")
    print("=" * 78)
    print(f"  channel sets : {len(set_names)}  {set_names}")
    print(f"  observables  : {obs_names}")
    print(f"  seeds        : {len(seed_list)} (common random numbers) {seed_list}")
    print(f"  records      : " + ", ".join(
        f"{r}(t_slow={records[r].t_slow}, {records[r].t_slow * t_r * 1e6:.3g} us, "
        f"floor {1.0 / (records[r].t_slow * t_r):.4g} Hz)"
        for r in sorted(needed_records)))
    if need_staircase:
        print(f"  staircase    : {stair_seed_idx and len(stair_seed_idx)} seeds "
              f"x {staircase['n_steps']} steps @ n_tau={staircase['n_tau']}")
    print(f"  solver units : {total_units}  (~{est_rt:.3g} round trips total)")
    print(f"  thermal_feedback pinned ON in every cell (see module docstring)")
    print("=" * 78)

    def _persist():
        _atomic_json(raw_path, {
            "units": store,
            "meta": {
                "config_fingerprint": fingerprint,
                "seeds": seed_list, "quick": quick,
                "records": {k: v.as_dict() for k, v in records.items()},
                "staircase": staircase,
                "channel_sets": {k: CHANNEL_SETS[k].to_dict() for k in set_names},
            },
        })

    done = 0
    for set_name in set_names:
        nc = CHANNEL_SETS[set_name]
        cfg_path = sidecars[set_name]

        for record in sorted(needed_records):
            spec = records[record]
            cav = cav_for(spec.n_tau)
            for seed in seed_list:
                key = _unit_key(set_name, record, seed)
                if key in store:
                    done += 1
                    continue
                t0 = time.time()
                ctx = RecordContext(
                    record=record, t_r=t_r, kappa=kappa, t_slow=spec.t_slow,
                    snapshot_interval=spec.snapshot_interval, n_tau=spec.n_tau,
                    probe_mus=PROBE_MUS, channel_set=set_name, seed=seed)
                try:
                    sol = _metrology_unit(spec, cav, cfg_path, seed, PROBE_MUS)
                except Exception as exc:  # noqa: BLE001 - realization dropped
                    store[key] = {"failed": True, "reason": str(exc)[:400],
                                  "channel_set": set_name, "record": record,
                                  "seed": seed}
                    _persist()
                    done += 1
                    print(f"[budget] {done}/{total_units} {key} DROPPED: "
                          f"{str(exc)[:90]}")
                    continue

                unit = {"failed": False, "channel_set": set_name,
                        "record": record, "seed": seed,
                        "t_slow": spec.t_slow,
                        "duration_s": spec.t_slow * t_r,
                        "fourier_floor_hz": 1.0 / (spec.t_slow * t_r),
                        "n_tau": spec.n_tau,
                        "snapshot_interval": spec.snapshot_interval,
                        "observables": {}}
                for name in obs_names:
                    which = OBSERVABLES[name]["record"]
                    if which != record and which != "both":
                        continue
                    ctx.notes = {}
                    values, unit_str, uncert = OBSERVABLES[name]["fn"](sol, ctx)
                    unit["observables"][name] = {
                        "unit": unit_str,
                        "values": _clean(values),
                        "uncertainty": _clean(uncert),
                        "notes": _clean(ctx.notes),
                    }
                del sol
                store[key] = unit
                _persist()
                done += 1
                print(f"[budget] {done}/{total_units} {key} "
                      f"({time.time() - t0:.1f}s)")

        if need_staircase:
            stair_cav = cav_for(staircase["n_tau"])
            for i in stair_seed_idx:
                key = _unit_key(set_name, "staircase", SEED_BASE + i)
                if key in store:
                    done += 1
                    continue
                t0 = time.time()
                try:
                    sw = _staircase_unit(staircase, stair_cav, cfg_path, i)
                except Exception as exc:  # noqa: BLE001
                    store[key] = {"failed": True, "reason": str(exc)[:400],
                                  "channel_set": set_name, "record": "staircase",
                                  "seed": SEED_BASE + i}
                    _persist()
                    done += 1
                    print(f"[budget] {done}/{total_units} {key} DROPPED: "
                          f"{str(exc)[:90]}")
                    continue
                ctx = RecordContext(
                    record="staircase", t_r=t_r, kappa=kappa,
                    t_slow=staircase["hold_rt"] * staircase["n_steps"],
                    snapshot_interval=1, n_tau=staircase["n_tau"],
                    probe_mus=PROBE_MUS, channel_set=set_name,
                    seed=SEED_BASE + i,
                    det_k=np.asarray(sw["dw_over_kappa"]),
                    n_solitons=int(staircase["n_solitons"]))
                values, unit_str, uncert = obs_step_jitter(sw, ctx)
                store[key] = {
                    "failed": False, "channel_set": set_name,
                    "record": "staircase", "seed": SEED_BASE + i,
                    "t_slow": staircase["hold_rt"] * staircase["n_steps"],
                    "duration_s": staircase["hold_rt"] * staircase["n_steps"] * t_r,
                    "fourier_floor_hz": None,
                    "n_tau": staircase["n_tau"], "snapshot_interval": 1,
                    "observables": {"step_jitter": {
                        "unit": unit_str, "values": _clean(values),
                        "uncertainty": _clean(uncert),
                        "notes": _clean(ctx.notes)}},
                }
                _persist()
                done += 1
                print(f"[budget] {done}/{total_units} {key} "
                      f"({time.time() - t0:.1f}s)")

    # NB: jax.clear_caches() is deliberately NOT called between units. Every
    # unit of a given record reuses the same traced shapes (t_slow, n_tau,
    # snapshot cadence are fixed per record), so the compilation cache holds a
    # handful of entries for the whole run -- while clearing it forces a ~20 s
    # recompile PER SWEEP, which on the staircase alone costs more wall clock
    # than every solver step in this driver put together.

    # --- aggregate into cells ------------------------------------------------
    cells: dict[str, dict] = {}
    for set_name in set_names:
        cells[set_name] = {}
        for name in obs_names:
            which = OBSERVABLES[name]["record"]
            rec_names = (["fast", "slow"] if which == "both"
                         else [which])
            rec_names = [r for r in rec_names
                         if r in needed_records or r == "staircase"]
            if which == "staircase" and not need_staircase:
                continue
            # For an observable evaluated on several records, prefer per POINT
            # the record that resolved it (and, among those, the longest one --
            # more averaging). Which record won is recorded with the value.
            point_blobs: dict[str, dict] = {}
            cell_meta: dict = {}
            for rec in rec_names:
                seeds_here = ([SEED_BASE + i for i in stair_seed_idx]
                              if rec == "staircase" else seed_list)
                units = [store.get(_unit_key(set_name, rec, s))
                         for s in seeds_here]
                live = [(s, u) for s, u in zip(seeds_here, units)
                        if u and not u.get("failed")
                        and name in u.get("observables", {})]
                if not live:
                    continue
                cell_meta = {
                    "t_slow": live[0][1]["t_slow"],
                    "duration_s": live[0][1]["duration_s"],
                    "fourier_floor_hz": live[0][1]["fourier_floor_hz"],
                    "n_tau": live[0][1]["n_tau"],
                } if not cell_meta else cell_meta
                unit_str = live[0][1]["observables"][name]["unit"]
                all_points: list[str] = []
                for _s, u in live:
                    for p in u["observables"][name]["values"]:
                        if p not in all_points:
                            all_points.append(p)
                for p in all_points:
                    per_seed = [u["observables"][name]["values"].get(p)
                                for _s, u in live]
                    within = [u["observables"][name]["uncertainty"].get(p)
                              for _s, u in live]
                    notes = [u["observables"][name]["notes"].get(p, {})
                             for _s, u in live]
                    resolved = [bool(n.get("resolved")) for n in notes]
                    stats = _summarise(per_seed, [s for s, _u in live])
                    n_res = sum(resolved)
                    n_fin = int(stats["n_seeds_used"])
                    # An observable evaluated on several records keeps, per
                    # point, the record that did best on it: fully resolved
                    # beats partly resolved, a finite value beats a null, and
                    # among equals the longer record wins (more averaging).
                    prev = point_blobs.get(p)
                    rank = (n_res, n_fin, live[0][1]["t_slow"] or 0)
                    if prev is not None and rank <= prev["_rank"]:
                        continue
                    wr = [x for x in within
                          if x is not None and math.isfinite(float(x))]
                    point_blobs[p] = {
                        "_rank": rank,
                        "record": rec,
                        "t_slow": live[0][1]["t_slow"],
                        "record_duration_s": live[0][1]["duration_s"],
                        "record_fourier_floor_hz": live[0][1]["fourier_floor_hz"],
                        "unit": unit_str,
                        "n_seeds_resolved": int(n_res),
                        "resolved": bool(n_res == len(live)),
                        "statistics": stats,
                        "within_record_uncertainty_mean": (
                            float(np.mean(wr)) if wr else None),
                        "notes": notes[0] if notes else {},
                    }
            for blob in point_blobs.values():
                blob.pop("_rank", None)
            if not point_blobs:
                continue
            cells[set_name][name] = {
                "observable": name,
                "unit": OBSERVABLES[name]["unit"],
                "budget_statistic": OBSERVABLES[name]["statistic"],
                "record": rec_names[0] if len(rec_names) == 1 else "per-point",
                "seed_list": ([SEED_BASE + i for i in stair_seed_idx]
                              if which == "staircase" else seed_list),
                **cell_meta,
                "points": point_blobs,
            }

    # --- CRN self-check ------------------------------------------------------
    crn = _crn_check(cells)

    derived = {name: _derive(cells, name, OBSERVABLES[name]["statistic"])
               for name in obs_names if any(name in c for c in cells.values())}

    physical = _load_config(CONFIG_PATH)
    report = {
        "provenance": provenance_stamp(
            "analysis/noise_budget.py", seed=seed_list[0],
            physical_params=physical, quick=quick),
        "run": {
            "seeds": len(seed_list),
            "seed_list": seed_list,
            "staircase_seed_list": [SEED_BASE + i for i in stair_seed_idx],
            "quick": quick,
            "common_random_numbers": True,
            "thermal_feedback_pinned": True,
            "amplitude_preset": {
                "name": "ECDL flagship (analysis.noise_validation_campaign)",
                "pump_freq_noise_h0_hz2_per_hz": float(ECDL_H0),
                "pump_freq_noise_hm1_hz3_per_hz": float(ECDL_HM1),
                "pump_rin_floor_dbc_per_hz": float(ECDL_RIN_FLOOR_DBC),
                "trn_psd_model": TRN_PSD_MODEL,
                **{k: float(v) for k, v in KG_GEOMETRY.items()},
            },
            "operating_dw_over_kappa": float(OPERATING_DW_KAPPA),
            "pin_w": float(PIN_W),
            "t_r_s": float(t_r),
            "kappa_rad_per_s": float(kappa),
            "probe_mus": list(PROBE_MUS),
            "records": {k: v.as_dict() for k, v in records.items()},
            "staircase": staircase,
            "numerics": dict(PRODUCTION_NUMERICS),
            "config_fingerprint": fingerprint,
            "welch_bin_margin": WELCH_BIN_MARGIN,
            "welch_min_segments": WELCH_MIN_SEGMENTS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "wall_clock_s": None,
            "reference": "arXiv:2604.05897v1 (Herr, Tikan, Kippenberg, 7 Apr 2026)",
        },
        "channel_sets": {k: CHANNEL_SETS[k].to_dict() for k in set_names},
        "channel_set_sha256": {k: CHANNEL_SETS[k].sha256() for k in set_names},
        "cells": cells,
        "derived": derived,
        "crn_check": crn,
        "units_failed": {k: v for k, v in store.items() if v.get("failed")},
    }
    report["run"]["wall_clock_s"] = float(time.time() - t_start)

    _atomic_json(out_dir / "budget.json", report)
    (out_dir / "budget_table.tex").write_text(_render_tex(report),
                                              encoding="utf-8")
    (out_dir / "budget_table.md").write_text(_render_markdown(report),
                                             encoding="utf-8")

    arrays: dict[str, np.ndarray] = {}
    for set_name, obs_map in cells.items():
        for name, cell in obs_map.items():
            for p, blob in cell["points"].items():
                tag = f"{set_name}__{name}__{p}".replace(" ", "").replace("=", "")
                arrays[tag] = np.array(
                    [np.nan if x is None else x
                     for x in blob["statistics"]["per_seed"]], dtype=np.float64)
    if not arrays:
        arrays["empty"] = np.zeros(0)
    save_artifact(out_dir / "budget", arrays,
                  meta={"run": report["run"], "crn_check": crn,
                        "channel_sets": report["channel_sets"]},
                  provenance=report["provenance"])

    _print_gate(report)
    return report


def _crn_check(cells: dict) -> dict:
    """Assert the pyro_eo row reproduces the trn row -- live proof CRN works.

    On this SiN device ``pyro_coeff = 0``, and ``pyro_eo`` is driven by the very
    same delta_T realization as ``trn``, so under common random numbers the two
    rows must agree to the last bit. If they disagree, the seed list is not
    actually shared (or a channel is perturbing another's random stream) and
    every OAT/LOO difference in the table is contaminated by sampling noise.
    """
    out = {
        "description": (
            "pyro_eo must reproduce trn bit-for-bit on this device "
            "(pyroelectric_coeff = 0 and eo_r33 = 0 => pyro_coeff = 0, and "
            "pyro_eo shares the trn delta_T realization). A mismatch means "
            "common random numbers are NOT in effect."),
        "checked": 0, "identical": 0, "mismatches": [], "passed": None,
    }
    a, b = cells.get("trn"), cells.get("pyro_eo")
    if not a or not b:
        out["passed"] = None
        out["reason"] = "trn and/or pyro_eo not in this run; check skipped"
        return out
    for name in a:
        if name not in b:
            continue
        for p, blob in a[name]["points"].items():
            if p not in b[name]["points"]:
                continue
            xa = blob["statistics"]["per_seed"]
            xb = b[name]["points"][p]["statistics"]["per_seed"]
            out["checked"] += 1
            if xa == xb:
                out["identical"] += 1
            else:
                out["mismatches"].append(
                    {"observable": name, "point": p, "trn": xa, "pyro_eo": xb})
    out["passed"] = bool(out["checked"] and not out["mismatches"])
    return out


def _print_gate(report: dict) -> None:
    cells = report["cells"]
    cols = _table_columns(cells)
    print("\n" + "=" * 78)
    print("GATE G4 -- noise budget")
    print("=" * 78)
    n_cells = sum(len(v) for v in cells.values())
    n_points = sum(len(c["points"]) for v in cells.values() for c in v.values())
    complete = 0
    incomplete = []
    for set_name, obs_map in cells.items():
        for name, cell in obs_map.items():
            for p, blob in cell["points"].items():
                ok = (blob["statistics"]["mean"] is not None
                      and blob["unit"]
                      and blob["statistics"]["sem"] is not None
                      and blob["t_slow"] is not None
                      and cell["seed_list"])
                if ok:
                    complete += 1
                else:
                    incomplete.append(f"{set_name}/{name}/{p}")
    print(f"  cells: {n_cells}  points: {n_points}  "
          f"with value+unit+uncertainty+record+seeds: {complete}")
    if incomplete:
        print(f"  points without a numeric value ({len(incomplete)}): "
              f"{incomplete[:8]}{' ...' if len(incomplete) > 8 else ''}")
        print("    (each carries an explicit 'reason' in budget.json -- these "
              "are frequencies the record cannot observe, not missing work)")
    crn = report["crn_check"]
    print(f"  CRN self-check (pyro_eo == trn): {crn['passed']} "
          f"({crn['identical']}/{crn['checked']} identical)")
    if crn.get("mismatches"):
        print("    *** COMMON RANDOM NUMBERS ARE NOT IN EFFECT -- every OAT/LOO "
              "difference in this table is contaminated ***")
    if report["units_failed"]:
        print(f"  dropped solver units: {len(report['units_failed'])}")
    header = "  " + f"{'channel set':<14}" + "".join(
        f"{h[:22]:>24}" for _, _, h, _ in cols)
    print("\n" + header)
    for set_name in CHANNEL_SETS:
        if set_name not in cells:
            continue
        row = f"  {set_name:<14}"
        for obs, point, _h, _u in cols:
            stat = OBSERVABLES[obs]["statistic"]
            c = cells[set_name].get(obs)
            v = _stat_of_point(c, point, stat)
            row += f"{(_fmt(v) if v is not None else '—'):>24}"
        print(row)
    print(f"\n  wall clock: {report['run']['wall_clock_s']:.1f} s")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stochastic-LLE noise budget (OAT + LOO + interaction).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds and short records: a plumbing smoke, not a "
                         "measurement (the low S_rep points come back null "
                         "with a stated reason)")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                    help=f"realizations per cell (default {DEFAULT_SEEDS}); "
                         f"the staircase caps at {STEP_JITTER_MAX_SEEDS}")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed solver units from budget_raw.json")
    ap.add_argument("--observable", action="append", default=None,
                    metavar="NAME", choices=sorted(OBSERVABLES),
                    help="restrict to this observable (repeatable)")
    ap.add_argument("--channel-set", action="append", default=None,
                    metavar="NAME", choices=sorted(CHANNEL_SETS),
                    help="restrict to this channel set (repeatable)")
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / BUDGET_DIRNAME,
                    help="output directory (default analysis/results/budget)")
    args = ap.parse_args(argv)

    seeds = 2 if args.quick else int(args.seeds)
    if seeds < 1:
        ap.error("--seeds must be >= 1")

    report = run_budget(
        args.out, seeds=seeds, quick=args.quick, resume=args.resume,
        only_observables=args.observable, only_channel_sets=args.channel_set)
    ok = report["crn_check"]["passed"]
    return 0 if ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
