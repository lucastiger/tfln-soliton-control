#!/usr/bin/env python
"""Runtime and memory scaling of the stochastic-LLE solver.

Sweeps :func:`simulator.lle_solver.solve_lle_ssfm_jax` over the FFT grid
``n_tau``, the record length ``t_slow``, the trajectory batch ``n_traj``, the
noise configuration and every available JAX backend, and reports per-point
compile time, wall time, normalized cost, host memory and device memory.

    n_tau  in {256, 512, 1024, 2048, 4096, 8192}
    t_slow in {1e3, 1e4, 1e5}
    n_traj in {1, 8, 64}
    noise  in {all_off, all_on}
    device in the available backends (cpu, plus gpu when jax.devices('gpu') is
            non-empty)

MEASUREMENT CONTRACT -- what the clock actually covers
------------------------------------------------------
The classic JAX benchmarking error is timing an asynchronous dispatch and
reporting it as compute: JAX returns un-awaited device buffers, so a naive timer
measures how fast Python enqueued the work. This benchmark is immune to that,
and PROVES it rather than asserting it:

* ``solve_lle_ssfm_jax`` ends with ``{k: np.asarray(v) for k, v in out.items()}``.
  ``np.asarray`` on a device buffer is a blocking device-to-host transfer, so
  every output is fully materialized before the function returns.
* On top of that, :func:`_block_until_ready` walks the returned dict and calls
  ``jax.block_until_ready`` on anything that is still a ``jax.Array``. Today it
  finds none, and the count it finds is recorded per point as
  ``jax_arrays_blocked`` -- ``0`` there is the evidence that the numpy
  conversion had already synchronized, not a claim that no synchronization was
  needed.

Consequence worth stating plainly: the reported ``wall_s`` is the END-TO-END
cost of calling the solver, which INCLUDES the device-to-host transfer of every
returned history. That is what the function costs a caller, so it is the honest
number, but it means ``wall_s`` is not pure FLOPs. ``output_bytes`` is recorded
per point so the transfer share is visible; at the largest batches the per-round-
trip histories are hundreds of MB and the transfer is a real fraction of the
time. Snapshots are deliberately minimized (``snapshot_interval = t_slow``, one
snapshot) so ``E_snapshots`` does not dominate that transfer.

WHAT ``jit_compile_s`` MEANS
---------------------------
The first call necessarily traces, compiles AND executes -- there is no way to
compile without running. So the first call is timed separately as
``first_call_s``, and

    jit_compile_s = max(first_call_s - median(wall_s), 0)

i.e. compile+trace overhead inferred by difference against the steady-state
cost. Both numbers are reported; the raw ``first_call_s`` is the one to trust if
the difference ever goes negative (recorded as ``0`` with the raw value intact).
The three timed runs happen AFTER a separate warmup call, so no compilation
lands inside the timed window.

MEMORY GUARDRAIL
----------------
A configuration is skipped when its PROJECTED peak exceeds ``--max-mem-gb``
(default 8), recorded as ``skipped: projected N GB`` rather than discovered by
OOM-ing the machine. The projection is an explicit, auditable byte model (see
:func:`project_memory_bytes`) and its full breakdown is stored for every point,
run or skipped -- a guardrail whose arithmetic you cannot inspect is not a
guardrail. The model counts the device result, the host copy that
``np.asarray`` makes of it, the SSFM working set and the noise sequences.

RUNTIME IS THE BINDING CONSTRAINT, NOT MEMORY
---------------------------------------------
Worth knowing before launching a full sweep: on this hardware the largest
configuration (n_tau 8192, t_slow 1e5, n_traj 64) projects to well under 1 GB
and so passes the memory guardrail comfortably -- while costing hours per timed
run. The memory budget does not protect against it. The sweep therefore prints
a calibrated wall-clock projection before it starts, and ``--max-seconds``
(default: disabled, so the specified behaviour is unchanged) can skip points
whose projected single-run cost exceeds a budget.

SCALING EXPONENT -- on which quantity
-------------------------------------
An FFT-limited step costs O(n log n) in the grid, so TOTAL time obeys
``T ~ n_tau^alpha`` with ``alpha = 1 + 1/ln(n_tau) ~ 1.1`` over this range: the
"1 + log-correction" the FFT predicts. The normalized cost
``ns_per_roundtrip_per_mode = T/(t_slow*n_tau*n_traj)`` divides one factor of
n_tau back out, so ITS exponent is ``alpha - 1 ~ 0.1``. Both are reported and
each is labelled with the quantity it belongs to, because an exponent quoted
without its ordinate is unreadable. A pure-power-law fit is also compared
against an explicit ``c*n*log2(n)`` reference model, since over six points a
weak logarithm and a small power are hard to tell apart by eye.

NUMERICS
--------
The solver's DEFAULT numerics are used (``n_substeps=1``, no dealiasing, no edge
absorber): this is a scaling measurement of the solver's base path, least
confounded by optional work. Note that ``analysis.dks_access.PRODUCTION_NUMERICS``
(``n_substeps=4`` + 2/3 dealias + edge absorber) is what the physics drivers
actually run and costs roughly 4x more per round trip. The numerics in force are
recorded in the JSON so the two are never confused.

``all_on`` also enables ``thermal_feedback``, the deterministic thermo-optic ODE
(``NoiseConfig.all_on()`` sets every switch, and that one is dynamics rather
than noise). The reported noise overhead is therefore "all_on vs all_off" as
specified, which includes that ODE.

Usage::

    python benchmarks/runtime_scaling.py --quick
    python benchmarks/runtime_scaling.py --max-mem-gb 16
    python benchmarks/runtime_scaling.py --max-seconds 300 --out benchmarks
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import platform
import re
import resource
import statistics
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402

from simulator.lle_solver import _load_config, solve_lle_ssfm_jax  # noqa: E402
from simulator.noise_config import NoiseConfig  # noqa: E402
from simulator.provenance import provenance_stamp  # noqa: E402

__all__ = [
    "BENCH_DIR",
    "N_TAU_GRID",
    "T_SLOW_GRID",
    "N_TRAJ_GRID",
    "NOISE_SETS",
    "BenchmarkError",
    "available_devices",
    "cpu_model",
    "fit_scaling_exponent",
    "main",
    "noise_overhead_pct",
    "project_memory_bytes",
    "run_sweep",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"

N_TAU_GRID: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)
T_SLOW_GRID: tuple[int, ...] = (1_000, 10_000, 100_000)
N_TRAJ_GRID: tuple[int, ...] = (1, 8, 64)
NOISE_SETS: tuple[str, ...] = ("all_off", "all_on")

QUICK_N_TAU: tuple[int, ...] = (256, 512)
QUICK_T_SLOW: tuple[int, ...] = (1_000,)
QUICK_N_TRAJ: tuple[int, ...] = (1,)

#: Timed repetitions after a warmup call. Three is what the spec asks for; the
#: median is reported with min and max so a single scheduler hiccup is visible
#: rather than averaged in.
N_REPEATS = 3

DEFAULT_MAX_MEM_GB = 8.0

#: Physical operating point. Held FIXED across the sweep -- this measures cost,
#: not physics, and a detuning that changed with the grid would confound the
#: scaling with a change of dynamical regime.
PIN_W = 0.214
KAPPA = 1.519e8
KAPPA_C = 1.215e8
BETA2 = 1.578e-18
DW_OVER_KAPPA = 10.0
RNG_SEED = 0

#: Live field-sized temporaries carried by one SSFM round trip (the field, its
#: FFT, the linear-operator product, the nonlinear kick, the dealias/absorber
#: scratch and the XLA fusion buffers). Deliberately generous: a memory
#: guardrail that under-projects is worse than one that skips a point it could
#: have run, because the failure mode of under-projecting is an OOM.
WORKING_FIELD_COPIES = 16

#: Per-round-trip float64 histories always returned: P_trans, U_int, DeltaT,
#: delta_omega_eff.
N_RT_HISTORIES = 4

#: Independent per-round-trip noise sequences materialized when noise is on
#: (combined detuning noise, delta_T, pump frequency, pump RIN, FSR dD1).
N_NOISE_SEQUENCES = 5

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


class BenchmarkError(RuntimeError):
    """A benchmark cannot proceed without reporting a misleading number."""


# ---------------------------------------------------------------------------
# Hardware identification
# ---------------------------------------------------------------------------
def cpu_model() -> str:
    """Exact CPU model string, for the table caption.

    A scaling table without the part number it was measured on is not
    reproducible, so this falls back through several sources rather than
    returning something vague.
    """
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def cpu_count_logical() -> int:
    return int(os.cpu_count() or 1)


def device_label(device) -> str:
    """Exact device model, e.g. ``NVIDIA A100-SXM4-40GB`` or the CPU part.

    The MODEL only. Core count is reported separately so the caption can state
    it once instead of twice.
    """
    if device.platform == "cpu":
        return cpu_model()
    kind = getattr(device, "device_kind", None) or "unknown"
    return str(kind)


def available_devices(requested: list[str] | None = None) -> list:
    """Backends to sweep: always cpu, plus gpu when one is actually present.

    ``jax.devices('gpu')`` RAISES when no GPU backend is linked in -- it does not
    return an empty list -- so the "non-empty" test has to be written as a
    try/except or the benchmark dies on a CPU-only machine.

    ``requested is None`` means "every backend present" (what argparse hands
    over when ``--device`` is absent). An explicit empty list means exactly
    that -- no backends -- and raises, rather than being quietly reinterpreted
    as the default.
    """
    found = []
    for name in (["cpu", "gpu"] if requested is None else requested):
        try:
            devices = jax.devices(name)
        except RuntimeError:
            continue
        if devices:
            found.append(devices[0])
    if not found:
        raise BenchmarkError("no JAX backend available, not even cpu.")
    return found


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def project_memory_bytes(n_tau: int, t_slow: int, n_traj: int,
                         noise_on: bool) -> dict:
    """Projected peak bytes for one configuration, with its full breakdown.

    The model, term by term (``snapshot_interval = t_slow``, so exactly one
    snapshot is retained):

    * ``result_device``  -- E_snapshots (n_traj, 1, n_tau) complex128, the four
      (n_traj, t_slow) float64 round-trip histories, label_history, e_final
      (n_traj, n_tau) complex128 and delta_t_final;
    * ``result_host``    -- the copy ``np.asarray`` makes of all of that on the
      way out. Both live at once, which is why it is counted twice;
    * ``working``        -- ``WORKING_FIELD_COPIES`` field-sized complex128
      temporaries carried through one SSFM round trip;
    * ``noise``          -- ``N_NOISE_SEQUENCES`` (n_traj, t_slow) float64
      sequences, materialized only when a channel is on;
    * ``detuning_input`` -- the (n_traj, t_slow) float32 drive array, host and
      device.

    Returned in bytes, with ``total`` and a human-readable ``formula``. The
    breakdown is stored for every point so the guardrail's arithmetic can be
    audited instead of trusted.
    """
    n_tau, t_slow, n_traj = int(n_tau), int(t_slow), int(n_traj)
    complex_b, float_b = 16, 8

    snapshots = n_traj * 1 * n_tau * complex_b
    histories = n_traj * t_slow * float_b * N_RT_HISTORIES
    labels = n_traj * 1 * 4
    e_final = n_traj * n_tau * complex_b
    result_device = snapshots + histories + labels + e_final + n_traj * float_b
    result_host = result_device

    working = n_traj * n_tau * complex_b * WORKING_FIELD_COPIES
    noise = (n_traj * t_slow * float_b * N_NOISE_SEQUENCES) if noise_on else 0
    detuning_input = 2 * n_traj * t_slow * 4

    total = result_device + result_host + working + noise + detuning_input
    return {
        "result_device": result_device,
        "result_host": result_host,
        "working": working,
        "noise": noise,
        "detuning_input": detuning_input,
        "total": total,
        "total_gb": total / 1024 ** 3,
        "formula": (
            f"2*(snapshots {snapshots} + histories {histories} + labels {labels} "
            f"+ e_final {e_final}) + working {working} + noise {noise} "
            f"+ detuning_input {detuning_input}"
        ),
    }


class _RSSSampler:
    """Sample this process's RSS on a thread to get a TRUE per-config peak.

    ``resource.getrusage().ru_maxrss`` is a process-lifetime high-water mark: it
    never goes down, so after the first big configuration every later one would
    inherit that peak and report it as its own. Polling ``/proc/self/statm``
    around each timed call gives the peak that configuration actually reached.
    ``ru_maxrss`` is recorded alongside it as a cross-check.
    """

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = float(interval_s)
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _rss_bytes() -> int:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as fh:
                return int(fh.read().split()[1]) * _PAGE_SIZE
        except (OSError, IndexError, ValueError):
            # Non-Linux: fall back to the high-water mark (KiB on Linux/BSD,
            # bytes on macOS -- only used where statm is absent).
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._rss_bytes())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "_RSSSampler":
        self.peak_bytes = self._rss_bytes()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_bytes = max(self.peak_bytes, self._rss_bytes())


def device_memory_mb(device) -> float | None:
    """Peak device bytes in MB, or ``None`` where the backend has no stats.

    The CPU backend returns ``None`` from ``memory_stats()`` (there is no
    separate device allocator to report on), which is exactly the "else null"
    case the spec calls for -- not an error.
    """
    try:
        stats = device.memory_stats()
    except Exception:  # noqa: BLE001 - backend-specific, never fatal here
        return None
    if not stats:
        return None
    for key in ("peak_bytes_in_use", "peak_memory_in_use", "bytes_in_use"):
        if key in stats:
            return float(stats[key]) / 1024 ** 2
    return None


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet():
    """Silence the solver's per-run stdout and warnings; exceptions still fly."""
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        yield buf


def _block_until_ready(result: dict) -> int:
    """Force every still-async output; return how many needed it.

    Returns 0 today because the solver already converts each output with
    ``np.asarray`` (a blocking transfer). Recording the count rather than
    assuming it is what makes "we measured compute, not dispatch" checkable
    from the artifact instead of taken on faith.
    """
    pending = [v for v in result.values() if isinstance(v, jax.Array)]
    if pending:
        jax.block_until_ready(pending)
    return len(pending)


def _output_bytes(result: dict) -> int:
    return int(sum(v.nbytes for v in result.values()
                   if isinstance(v, np.ndarray)))


def _call_solver(n_tau: int, t_slow: int, n_traj: int, nc: NoiseConfig) -> dict:
    dw = np.full((n_traj, t_slow), DW_OVER_KAPPA * KAPPA, dtype=np.float32)
    return solve_lle_ssfm_jax(
        pin=PIN_W,
        delta_omega=dw,
        t_slow=int(t_slow),
        beta=[BETA2],
        kappa=KAPPA,
        kappa_c=KAPPA_C,
        rng_key=jax.random.PRNGKey(RNG_SEED),
        n_tau=int(n_tau),
        # One snapshot only: E_snapshots would otherwise dominate the
        # device-to-host transfer and turn a compute benchmark into an I/O one.
        snapshot_interval=int(t_slow),
        config_path=str(CONFIG_PATH),
        noise_config=nc,
    )


def measure_point(n_tau: int, t_slow: int, n_traj: int, noise: str,
                  device) -> dict:
    """Compile, warm up, then time ``N_REPEATS`` runs of one configuration."""
    nc = (NoiseConfig.all_off() if noise == "all_off" else NoiseConfig.all_on())

    with jax.default_device(device):
        # --- first call: trace + compile + execute, timed on its own ---------
        t0 = time.perf_counter()
        with _quiet():
            result = _call_solver(n_tau, t_slow, n_traj, nc)
        blocked = _block_until_ready(result)
        first_call_s = time.perf_counter() - t0
        out_bytes = _output_bytes(result)
        del result

        # --- warmup: a second call so nothing compiles inside the timed window
        with _quiet():
            result = _call_solver(n_tau, t_slow, n_traj, nc)
        blocked += _block_until_ready(result)
        del result

        # --- timed repetitions ----------------------------------------------
        samples: list[float] = []
        with _RSSSampler() as rss:
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                with _quiet():
                    result = _call_solver(n_tau, t_slow, n_traj, nc)
                blocked += _block_until_ready(result)
                samples.append(time.perf_counter() - t0)
                del result

        dev_mem_mb = device_memory_mb(device)

    wall_s = float(statistics.median(samples))
    total_mode_roundtrips = float(t_slow) * float(n_tau) * float(n_traj)
    return {
        "status": "ok",
        "first_call_s": first_call_s,
        "jit_compile_s": max(first_call_s - wall_s, 0.0),
        "wall_s": wall_s,
        "wall_min_s": float(min(samples)),
        "wall_max_s": float(max(samples)),
        "wall_samples_s": [float(x) for x in samples],
        "n_repeats": N_REPEATS,
        "ns_per_roundtrip_per_mode": wall_s * 1e9 / total_mode_roundtrips,
        "throughput_roundtrips_per_s": float(t_slow) / wall_s,
        "throughput_mode_roundtrips_per_s": total_mode_roundtrips / wall_s,
        "peak_rss_mb": rss.peak_bytes / 1024 ** 2,
        "rusage_maxrss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "peak_device_mem_mb": dev_mem_mb,
        "output_bytes": out_bytes,
        "output_mb": out_bytes / 1024 ** 2,
        "jax_arrays_blocked": int(blocked),
    }


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def fit_scaling_exponent(n_tau_values, wall_values) -> dict:
    """Log-log slope of TOTAL wall time vs n_tau, plus an n-log-n comparison.

    ``alpha_total`` is d log(T) / d log(n_tau). An FFT-limited step predicts
    ``alpha_total = 1 + 1/ln(n_tau)``, i.e. ~1.1 over this grid -- the
    "1 + log-correction". ``alpha_per_mode = alpha_total - 1`` is the slope of
    the normalized ns/round-trip/mode, which is the quantity the table shows.

    Also fits ``T = c*n*log2(n)`` (one free scale) and reports its R^2, because
    over six octaves a weak logarithm and a small power look alike, and the
    honest comparison is which model the data actually prefer.

    FIXED OVERHEAD IS WHY THE PLAIN SLOPE READS LOW. Each solver call carries a
    per-call cost that does not depend on n_tau (host-side config load and
    validation, noise-sequence setup, scan dispatch). At small grids that floor
    is most of the runtime, so a log-log fit across the whole grid measures the
    floor as much as the FFT. Measured here: the plain fit over
    256..8192 gives alpha_total ~ 0.80, while the compute-bound upper half
    (2048..8192) gives ~1.08 against a predicted ~1.14. Three extra numbers are
    therefore reported so the plain exponent can be read correctly rather than
    quoted bare:

    * ``fixed_overhead_s``            -- the intercept of ``T = a + c*n*log2(n)``;
    * ``overhead_frac_at_min_n_tau``  -- what fraction of the smallest point that
      intercept is (the contamination, in one number);
    * ``alpha_total_compute_bound``   -- the log-log slope over the LARGEST half
      of the n_tau points, where the floor no longer dominates. This is the one
      to compare against ``predicted_alpha_n_log_n``.
    """
    n = np.asarray(n_tau_values, dtype=np.float64)
    t = np.asarray(wall_values, dtype=np.float64)
    keep = np.isfinite(n) & np.isfinite(t) & (n > 0) & (t > 0)
    n, t = n[keep], t[keep]
    out = {
        "n_points": int(n.size),
        "alpha_total": None,
        "alpha_per_mode": None,
        "r_squared": None,
        "predicted_alpha_n_log_n": None,
        "n_log_n_r_squared": None,
        "fixed_overhead_s": None,
        "overhead_frac_at_min_n_tau": None,
        "alpha_total_compute_bound": None,
        "compute_bound_n_tau": None,
        "note": "",
    }
    if n.size < 2:
        out["note"] = ("fewer than 2 distinct n_tau points; no slope is "
                       "defined")
        return out

    log_n, log_t = np.log(n), np.log(t)
    slope, intercept = np.polyfit(log_n, log_t, 1)
    resid = log_t - (slope * log_n + intercept)
    ss_tot = float(np.sum((log_t - log_t.mean()) ** 2))
    out["alpha_total"] = float(slope)
    out["alpha_per_mode"] = float(slope - 1.0)
    out["r_squared"] = (float(1.0 - np.sum(resid ** 2) / ss_tot)
                        if ss_tot > 0 else None)
    # The FFT prediction is n-dependent; quote it at the geometric mean of the
    # grid, which is where a single power-law slope is anchored.
    n_ref = float(np.exp(log_n.mean()))
    out["predicted_alpha_n_log_n"] = float(1.0 + 1.0 / math.log(n_ref))
    out["reference_n_tau"] = n_ref

    model = n * np.log2(n)
    scale = float(np.sum(model * t) / np.sum(model * model))
    pred = scale * model
    ss_res = float(np.sum((t - pred) ** 2))
    ss_t = float(np.sum((t - t.mean()) ** 2))
    out["n_log_n_r_squared"] = float(1.0 - ss_res / ss_t) if ss_t > 0 else None
    out["n_log_n_scale_s_per_mode_roundtrip"] = scale

    # --- affine model T = a + c*n*log2(n): a is the n_tau-independent floor ---
    if n.size >= 3:
        design = np.stack([np.ones_like(model), model], axis=1)
        (a_fit, c_fit), *_ = np.linalg.lstsq(design, t, rcond=None)
        pred_affine = a_fit + c_fit * model
        ss_res_a = float(np.sum((t - pred_affine) ** 2))
        out["fixed_overhead_s"] = float(a_fit)
        out["n_log_n_affine_r_squared"] = (
            float(1.0 - ss_res_a / ss_t) if ss_t > 0 else None)
        t_min = float(t[int(np.argmin(n))])
        if t_min > 0:
            out["overhead_frac_at_min_n_tau"] = float(a_fit / t_min)

    # --- compute-bound slope: largest half of the grid, floor no longer ruling
    if n.size >= 4:
        cut = float(np.median(n))
        sel = n >= cut
        if sel.sum() >= 2:
            slope_cb, _ = np.polyfit(np.log(n[sel]), np.log(t[sel]), 1)
            out["alpha_total_compute_bound"] = float(slope_cb)
            out["compute_bound_n_tau"] = [int(v) for v in n[sel]]

    notes = []
    if n.size == 2:
        notes.append(
            "two n_tau points only: this is an exact two-point slope, not a "
            "fit -- r_squared is 1 by construction and carries no information")
    frac = out.get("overhead_frac_at_min_n_tau")
    if frac is not None and frac > 0.3:
        notes.append(
            f"the n_tau-independent per-call floor is {frac:.0%} of the "
            f"smallest point's runtime, so alpha_total UNDERSTATES the FFT "
            f"scaling; read alpha_total_compute_bound instead")
    elif n.size == 2:
        notes.append(
            "at this grid size the per-call floor (host config load, noise "
            "setup, scan dispatch) is a large share of the runtime, so this "
            "slope reflects that floor as much as the FFT")
    out["note"] = "; ".join(notes)
    return out


def noise_overhead_pct(points: list[dict]) -> dict:
    """all_on vs all_off wall-time overhead [%], matched point by point.

    Matched on (device, n_tau, t_slow, n_traj) so the comparison never straddles
    two different problem sizes. Includes the deterministic thermo-optic ODE,
    which ``NoiseConfig.all_on()`` switches on along with the noise channels.
    """
    by_key: dict[tuple, dict] = {}
    for p in points:
        if p.get("status") != "ok":
            continue
        key = (p["device"], p["n_tau"], p["t_slow"], p["n_traj"])
        by_key.setdefault(key, {})[p["noise"]] = p["wall_s"]

    rows, values = [], []
    for key, pair in sorted(by_key.items()):
        if "all_off" not in pair or "all_on" not in pair:
            continue
        off, on = pair["all_off"], pair["all_on"]
        if not (off > 0):
            continue
        pct = (on - off) / off * 100.0
        values.append(pct)
        rows.append({"device": key[0], "n_tau": key[1], "t_slow": key[2],
                     "n_traj": key[3], "wall_off_s": off, "wall_on_s": on,
                     "overhead_pct": pct})
    return {
        "per_configuration": rows,
        "median_pct": float(statistics.median(values)) if values else None,
        "min_pct": float(min(values)) if values else None,
        "max_pct": float(max(values)) if values else None,
        "n_pairs": len(values),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def _calibrated_projection(points_to_run: list[dict], calib: dict) -> float:
    """Rough total wall-clock seconds, from an n*log2(n)*t_slow*n_traj model."""
    if not calib:
        return float("nan")
    total = 0.0
    for p in points_to_run:
        work = (p["n_tau"] * math.log2(p["n_tau"]) * p["t_slow"] * p["n_traj"])
        per_run = calib["s_per_unit"] * work * (
            calib["noise_factor"] if p["noise"] == "all_on" else 1.0)
        total += per_run * (N_REPEATS + 2)  # compile + warmup + timed runs
    return total


def run_sweep(out_dir: Path, *, quick: bool, max_mem_gb: float,
              max_seconds: float | None, devices: list) -> dict:
    """Run the whole sweep and write both artifacts. Returns the report dict."""
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_taus = QUICK_N_TAU if quick else N_TAU_GRID
    t_slows = QUICK_T_SLOW if quick else T_SLOW_GRID
    n_trajs = QUICK_N_TRAJ if quick else N_TRAJ_GRID

    grid = [
        {"device": d.platform, "device_label": device_label(d), "_device": d,
         "n_tau": nt, "t_slow": ts, "n_traj": ntr, "noise": noise}
        for d in devices
        for nt in n_taus
        for ts in t_slows
        for ntr in n_trajs
        for noise in NOISE_SETS
    ]

    max_mem_bytes = float(max_mem_gb) * 1024 ** 3
    print("=" * 78)
    print("STOCHASTIC-LLE SOLVER RUNTIME SCALING")
    print("=" * 78)
    for d in devices:
        print(f"  backend {d.platform:<4} : {device_label(d)}")
    print(f"  grid         : n_tau={list(n_taus)} t_slow={list(t_slows)} "
          f"n_traj={list(n_trajs)} noise={list(NOISE_SETS)}")
    print(f"  points       : {len(grid)}  ({N_REPEATS} timed runs each, "
          f"after compile + warmup)")
    print(f"  memory budget: {max_mem_gb:g} GB"
          + (f"   time budget: {max_seconds:g} s/run" if max_seconds else
             "   time budget: disabled"))
    print("=" * 78)

    points: list[dict] = []
    calib: dict = {}
    for i, cell in enumerate(grid, start=1):
        device = cell.pop("_device")
        mem = project_memory_bytes(cell["n_tau"], cell["t_slow"],
                                   cell["n_traj"], cell["noise"] == "all_on")
        record = {**cell, "projected_memory": mem}

        if mem["total"] > max_mem_bytes:
            record.update({
                "status": f"skipped: projected {mem['total_gb']:.2f} GB",
                "skip_reason": (
                    f"projected peak {mem['total_gb']:.2f} GB exceeds the "
                    f"--max-mem-gb budget of {max_mem_gb:g} GB"),
            })
            points.append(record)
            print(f"[{i}/{len(grid)}] {_label(cell)} SKIPPED "
                  f"(projected {mem['total_gb']:.2f} GB)")
            continue

        if max_seconds is not None and calib:
            work = (cell["n_tau"] * math.log2(cell["n_tau"])
                    * cell["t_slow"] * cell["n_traj"])
            projected = calib["s_per_unit"] * work * (
                calib["noise_factor"] if cell["noise"] == "all_on" else 1.0)
            if projected > max_seconds:
                record.update({
                    "status": f"skipped: projected {projected:.0f} s per run",
                    "projected_run_s": projected,
                    "skip_reason": (
                        f"projected single-run wall time {projected:.0f} s "
                        f"exceeds the --max-seconds budget of {max_seconds:g} s"),
                })
                points.append(record)
                print(f"[{i}/{len(grid)}] {_label(cell)} SKIPPED "
                      f"(projected {projected:.0f} s/run)")
                continue

        try:
            measured = measure_point(cell["n_tau"], cell["t_slow"],
                                     cell["n_traj"], cell["noise"], device)
        except Exception as exc:  # noqa: BLE001 - one point must not kill the sweep
            record.update({"status": "failed", "error": str(exc)[:400]})
            points.append(record)
            print(f"[{i}/{len(grid)}] {_label(cell)} FAILED: {str(exc)[:90]}")
            continue

        record.update(measured)
        points.append(record)

        # Calibrate the projection off the first successful all_off point.
        work = (cell["n_tau"] * math.log2(cell["n_tau"])
                * cell["t_slow"] * cell["n_traj"])
        if cell["noise"] == "all_off":
            calib.setdefault("s_per_unit", measured["wall_s"] / work)
            calib.setdefault("noise_factor", 1.0)
        elif "s_per_unit" in calib:
            calib["noise_factor"] = max(
                measured["wall_s"] / (calib["s_per_unit"] * work), 1.0)

        print(f"[{i}/{len(grid)}] {_label(cell)} "
              f"wall={measured['wall_s']:.4f}s "
              f"compile={measured['jit_compile_s']:.2f}s "
              f"{measured['ns_per_roundtrip_per_mode']:.2f} ns/RT/mode "
              f"rss={measured['peak_rss_mb']:.0f}MB")

        if i == 1 and not quick:
            est = _calibrated_projection(
                [c for c in grid[1:]], calib)
            if math.isfinite(est):
                print(f"    [projection] remaining {len(grid) - 1} points "
                      f"~{est / 3600:.1f} h at this rate (n*log2(n) model)")

    report = _build_report(points, devices, quick, max_mem_gb, max_seconds,
                           time.time() - t_start)
    _write_json(out_dir / "runtime.json", report)
    (out_dir / "runtime_table.md").write_text(_render_markdown(report),
                                              encoding="utf-8")
    _print_gate(report, out_dir)
    return report


def _label(cell: dict) -> str:
    return (f"{cell['device']:<3} n_tau={cell['n_tau']:<5} "
            f"t_slow={cell['t_slow']:<7} n_traj={cell['n_traj']:<3} "
            f"{cell['noise']:<8}")


def _build_report(points, devices, quick, max_mem_gb, max_seconds,
                  wall_clock_s) -> dict:
    ok = [p for p in points if p.get("status") == "ok"]

    fits: dict[str, dict] = {}
    for device in {p["device"] for p in ok}:
        for noise in NOISE_SETS:
            # Fit on the (t_slow, n_traj) row with the WIDEST n_tau lever arm,
            # tie-broken towards the most compute-bound row. Selecting purely by
            # "largest t_slow/n_traj" would, once the guardrails prune the
            # expensive corners, hand the fit a truncated n_tau range -- and the
            # span of the abscissa is what a scaling exponent is made of.
            candidates = []
            for t_slow in sorted({p["t_slow"] for p in ok}):
                for n_traj in sorted({p["n_traj"] for p in ok}):
                    sel = [p for p in ok if p["device"] == device
                           and p["noise"] == noise and p["t_slow"] == t_slow
                           and p["n_traj"] == n_traj]
                    n_distinct = len({p["n_tau"] for p in sel})
                    if n_distinct >= 2:
                        candidates.append(
                            (n_distinct, t_slow * n_traj, t_slow, n_traj, sel))
            if not candidates:
                continue
            _n, _work, t_slow, n_traj, sel = max(
                candidates, key=lambda c: (c[0], c[1]))
            sel.sort(key=lambda p: p["n_tau"])
            fit = fit_scaling_exponent([p["n_tau"] for p in sel],
                                       [p["wall_s"] for p in sel])
            fit.update({"t_slow": t_slow, "n_traj": n_traj,
                        "n_tau": [p["n_tau"] for p in sel]})
            fits[f"{device}/{noise}"] = fit

    physical = _load_config(CONFIG_PATH)
    return {
        "provenance": provenance_stamp(
            "benchmarks/runtime_scaling.py", seed=RNG_SEED,
            physical_params=physical, quick=quick),
        "run": {
            "quick": quick,
            "max_mem_gb": float(max_mem_gb),
            "max_seconds": (None if max_seconds is None else float(max_seconds)),
            "n_repeats": N_REPEATS,
            "wall_clock_s": float(wall_clock_s),
            "n_points": len(points),
            "n_ok": len(ok),
            "n_skipped": sum(1 for p in points
                             if str(p.get("status", "")).startswith("skipped")),
            "n_failed": sum(1 for p in points if p.get("status") == "failed"),
            "grid": {
                "n_tau": list(QUICK_N_TAU if quick else N_TAU_GRID),
                "t_slow": list(QUICK_T_SLOW if quick else T_SLOW_GRID),
                "n_traj": list(QUICK_N_TRAJ if quick else N_TRAJ_GRID),
                "noise": list(NOISE_SETS),
            },
            "operating_point": {
                "pin_w": PIN_W, "kappa_rad_per_s": KAPPA,
                "kappa_c_rad_per_s": KAPPA_C, "beta2_s": BETA2,
                "dw_over_kappa": DW_OVER_KAPPA, "rng_seed": RNG_SEED,
            },
            "solver_numerics": {
                "n_substeps": 1, "dealias_two_thirds": False,
                "edge_absorber": False, "snapshot_interval": "t_slow",
                "note": ("solver defaults; analysis.dks_access.PRODUCTION_NUMERICS "
                         "(n_substeps=4 + dealias + absorber) costs ~4x more per "
                         "round trip and is what the physics drivers run"),
            },
            "memory_model": {
                "working_field_copies": WORKING_FIELD_COPIES,
                "n_rt_histories": N_RT_HISTORIES,
                "n_noise_sequences": N_NOISE_SEQUENCES,
                "note": ("projected peak = 2*(result) + working + noise + input; "
                         "see project_memory_bytes for the term-by-term model"),
            },
            "measurement_contract": {
                "block_until_ready": True,
                "note": ("solve_lle_ssfm_jax returns np.asarray of every output, "
                         "a blocking device-to-host transfer, so the timed call "
                         "cannot measure dispatch alone; jax_arrays_blocked "
                         "records how many outputs still needed an explicit "
                         "block (0 = the conversion had already synchronized). "
                         "wall_s is end-to-end and INCLUDES that transfer."),
            },
        },
        "hardware": {
            "devices": [{"platform": d.platform, "model": device_label(d),
                         "device_kind": getattr(d, "device_kind", None)}
                        for d in devices],
            "cpu_model": cpu_model(),
            "cpu_logical_cores": cpu_count_logical(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "points": points,
        "scaling_fits": fits,
        "noise_overhead": noise_overhead_pct(points),
    }


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
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_clean(payload), fh, indent=2, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _fmt(x, spec=".3g", dash="—"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return dash
    return format(x, spec)


def _render_markdown(report: dict) -> str:
    points = report["points"]
    ok = [p for p in points if p.get("status") == "ok"]
    devices = [d["platform"] for d in report["hardware"]["devices"]]
    caption_bits = [f"{d['platform'].upper()} {d['model']}"
                    for d in report["hardware"]["devices"]]
    missing = [b for b in ("cpu", "gpu") if b not in devices]

    lines = [
        "# Stochastic-LLE solver: runtime scaling",
        "",
        f"- generated: `{report['provenance']['generated_utc']}`",
        f"- git commit: `{report['provenance']['git_commit']}`",
        f"- quick mode: {report['run']['quick']}",
        f"- points: {report['run']['n_ok']} measured, "
        f"{report['run']['n_skipped']} skipped, {report['run']['n_failed']} failed",
        f"- timing: median of {report['run']['n_repeats']} runs after a separate "
        f"compile call and a warmup call; every output is materialized on the "
        f"host before the clock stops",
        "",
        "## ns per round-trip per mode vs `n_tau`",
        "",
        "**Hardware.** " + "; ".join(caption_bits)
        + f". {report['hardware']['cpu_logical_cores']} logical cores. "
        + f"JAX {report['hardware']['jax']}, NumPy {report['hardware']['numpy']}, "
        + f"Python {report['hardware']['python']}, "
        + f"{report['hardware']['platform']}.",
        "",
    ]
    if missing:
        lines += [
            f"> No **{'/'.join(m.upper() for m in missing)}** backend was "
            f"available on this machine, so those columns are absent rather "
            f"than empty. `jax.devices('gpu')` raises rather than returning an "
            f"empty list when no GPU backend is linked in.",
            "",
        ]

    lines += [
        "Cell = `ns_per_roundtrip_per_mode` = `wall_s * 1e9 / "
        "(t_slow * n_tau * n_traj)`, median over the swept `t_slow` and "
        "`n_traj` at that `n_tau`.",
        "",
    ]
    cols = [(d, n) for d in devices for n in NOISE_SETS]
    header = "| `n_tau` | " + " | ".join(f"{d.upper()} {n}" for d, n in cols) + " |"
    lines += [header, "|" + "---|" * (len(cols) + 1)]
    for n_tau in sorted({p["n_tau"] for p in ok}):
        row = [f"`{n_tau}`"]
        for dev, noise in cols:
            vals = [p["ns_per_roundtrip_per_mode"] for p in ok
                    if p["n_tau"] == n_tau and p["device"] == dev
                    and p["noise"] == noise]
            row.append(_fmt(statistics.median(vals), ".4g") if vals else "—")
        lines.append("| " + " | ".join(row) + " |")

    # --- scaling exponent ---------------------------------------------------
    lines += [
        "",
        "## Fitted scaling exponent",
        "",
        "`alpha_total` is the log-log slope of **total wall time** vs `n_tau`. "
        "An FFT-limited step costs O(n log n), predicting "
        "`alpha_total = 1 + 1/ln(n_tau)` (the *1 + log-correction*). "
        "`alpha_per_mode = alpha_total - 1` is the slope of the normalized "
        "ns/round-trip/mode shown above — that column divides one factor of "
        "`n_tau` back out, so a healthy FFT-bound solver shows a small positive "
        "slope there, not ~1.",
        "",
        "Each solver call also carries a **fixed cost independent of `n_tau`** "
        "(host config load and validation, noise-sequence setup, scan "
        "dispatch). At small grids that floor is most of the runtime and drags "
        "the plain log-log slope below the FFT prediction, so the floor is "
        "fitted explicitly as `T = a + c·n·log₂n` and reported: compare "
        "`alpha_compute_bound` (largest half of the grid) against `predicted`, "
        "not `alpha_total`, whenever `overhead@min` is large.",
        "",
        "| device / noise | `t_slow` | `n_traj` | points | `alpha_total` | "
        "`alpha_per_mode` | `alpha_compute_bound` | predicted (n log n) | R² | "
        "floor `a` [s] | overhead@min |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, fit in sorted(report["scaling_fits"].items()):
        frac = fit.get("overhead_frac_at_min_n_tau")
        lines.append(
            f"| {key} | {fit.get('t_slow')} | {fit.get('n_traj')} | "
            f"{fit['n_points']} | {_fmt(fit['alpha_total'], '.4g')} | "
            f"{_fmt(fit['alpha_per_mode'], '.4g')} | "
            f"{_fmt(fit.get('alpha_total_compute_bound'), '.4g')} | "
            f"{_fmt(fit['predicted_alpha_n_log_n'], '.4g')} | "
            f"{_fmt(fit['r_squared'], '.4g')} | "
            f"{_fmt(fit.get('fixed_overhead_s'), '.3g')} | "
            f"{'—' if frac is None else format(frac, '.0%')} |")
    notes = {f["note"] for f in report["scaling_fits"].values() if f.get("note")}
    for note in sorted(notes):
        lines += ["", f"> {note}"]

    # --- noise overhead -----------------------------------------------------
    ov = report["noise_overhead"]
    lines += [
        "",
        "## Noise-on overhead",
        "",
        "`(wall_on - wall_off) / wall_off`, matched per "
        "(device, `n_tau`, `t_slow`, `n_traj`). `NoiseConfig.all_on()` enables "
        "every stochastic channel **and** `thermal_feedback`, the deterministic "
        "thermo-optic ODE, so this is the cost of the whole switched-on stack.",
        "",
    ]
    if ov["n_pairs"]:
        lines += [
            f"**Median overhead: {ov['median_pct']:.1f}%** "
            f"(range {ov['min_pct']:.1f}% – {ov['max_pct']:.1f}% over "
            f"{ov['n_pairs']} matched pairs).",
            "",
            "| device | `n_tau` | `t_slow` | `n_traj` | off [s] | on [s] | overhead |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in ov["per_configuration"]:
            lines.append(
                f"| {r['device']} | {r['n_tau']} | {r['t_slow']} | "
                f"{r['n_traj']} | {_fmt(r['wall_off_s'], '.4g')} | "
                f"{_fmt(r['wall_on_s'], '.4g')} | {r['overhead_pct']:+.1f}% |")
    else:
        lines.append("No matched all_off/all_on pair completed.")

    # --- full per-point table ----------------------------------------------
    lines += [
        "",
        "## All measured points",
        "",
        "| device | `n_tau` | `t_slow` | `n_traj` | noise | compile [s] | "
        "wall [s] (min–max) | ns/RT/mode | RT/s | peak RSS [MB] | "
        "device mem [MB] | out [MB] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in points:
        if p.get("status") != "ok":
            continue
        lines.append(
            f"| {p['device']} | {p['n_tau']} | {p['t_slow']} | {p['n_traj']} | "
            f"{p['noise']} | {_fmt(p['jit_compile_s'], '.3g')} | "
            f"{_fmt(p['wall_s'], '.4g')} ({_fmt(p['wall_min_s'], '.3g')}–"
            f"{_fmt(p['wall_max_s'], '.3g')}) | "
            f"{_fmt(p['ns_per_roundtrip_per_mode'], '.4g')} | "
            f"{_fmt(p['throughput_roundtrips_per_s'], '.4g')} | "
            f"{_fmt(p['peak_rss_mb'], '.4g')} | "
            f"{_fmt(p['peak_device_mem_mb'], '.4g')} | "
            f"{_fmt(p['output_mb'], '.3g')} |")

    skipped = [p for p in points if str(p.get("status", "")).startswith("skipped")]
    if skipped:
        lines += [
            "",
            "## Skipped configurations",
            "",
            "| device | `n_tau` | `t_slow` | `n_traj` | noise | status | reason |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in skipped:
            lines.append(
                f"| {p['device']} | {p['n_tau']} | {p['t_slow']} | "
                f"{p['n_traj']} | {p['noise']} | {p['status']} | "
                f"{p.get('skip_reason', '')} |")

    lines += [
        "",
        "## Method notes",
        "",
        f"- **Blocking.** `solve_lle_ssfm_jax` returns `np.asarray` of every "
        f"output, which is a blocking device-to-host transfer, so a timed call "
        f"cannot measure dispatch alone. `_block_until_ready` additionally "
        f"walks the result for any remaining `jax.Array`; the count it finds is "
        f"stored per point as `jax_arrays_blocked` "
        f"({sum(p.get('jax_arrays_blocked', 0) for p in ok)} across all points "
        f"here — zero means the conversion had already synchronized).",
        "- **`wall_s` is end-to-end** and includes the host transfer of every "
        "returned history; `out [MB]` shows how much data that was.",
        "- **`jit_compile_s`** is `first_call_s - wall_s`: the first call must "
        "also execute, so compile time is only observable by difference.",
        "- **`peak RSS`** is sampled from `/proc/self/statm` on a thread during "
        "the timed runs, not taken from `ru_maxrss`, which is a "
        "process-lifetime high-water mark that every later configuration would "
        "inherit from the biggest earlier one.",
        f"- **Numerics**: solver defaults (`n_substeps=1`, no dealias, no edge "
        f"absorber). `PRODUCTION_NUMERICS` costs ~4x more per round trip.",
        f"- **Memory guardrail**: `--max-mem-gb {report['run']['max_mem_gb']:g}`, "
        f"projected by an explicit byte model whose per-point breakdown is in "
        f"`runtime.json`.",
    ]
    return "\n".join(lines) + "\n"


def _print_gate(report: dict, out_dir: Path) -> None:
    run = report["run"]
    print("\n" + "=" * 78)
    print("RUNTIME SCALING -- summary")
    print("=" * 78)
    for d in report["hardware"]["devices"]:
        print(f"  {d['platform']}: {d['model']}")
    print(f"  points: {run['n_ok']} ok, {run['n_skipped']} skipped, "
          f"{run['n_failed']} failed")
    for key, fit in sorted(report["scaling_fits"].items()):
        cb = fit.get("alpha_total_compute_bound")
        print(f"  alpha_total[{key}] = {_fmt(fit['alpha_total'], '.4g')} "
              f"(per-mode {_fmt(fit['alpha_per_mode'], '.4g')}"
              + (f"; compute-bound {_fmt(cb, '.4g')}" if cb is not None else "")
              + f"; n log n predicts "
              f"{_fmt(fit['predicted_alpha_n_log_n'], '.4g')})")
        if fit.get("note"):
            print(f"      note: {fit['note']}")
    ov = report["noise_overhead"]
    if ov["n_pairs"]:
        print(f"  noise-on overhead: median {ov['median_pct']:.1f}% "
              f"({ov['min_pct']:.1f}%..{ov['max_pct']:.1f}%, "
              f"{ov['n_pairs']} pairs)")
    blocked = sum(p.get("jax_arrays_blocked", 0)
                  for p in report["points"] if p.get("status") == "ok")
    print(f"  jax arrays needing an explicit block: {blocked} "
          f"(0 = solver's np.asarray already synchronized)")
    print(f"  wrote {out_dir / 'runtime.json'}")
    print(f"  wrote {out_dir / 'runtime_table.md'}")
    print(f"  wall clock: {run['wall_clock_s']:.1f} s")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Runtime and memory scaling of solve_lle_ssfm_jax.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help=f"only n_tau in {list(QUICK_N_TAU)}, "
                         f"t_slow={QUICK_T_SLOW[0]}, n_traj={QUICK_N_TRAJ[0]}")
    ap.add_argument("--max-mem-gb", type=float, default=DEFAULT_MAX_MEM_GB,
                    help=f"skip a configuration whose PROJECTED peak exceeds "
                         f"this (default {DEFAULT_MAX_MEM_GB:g})")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="optional: also skip a configuration whose projected "
                         "single-run wall time exceeds this. Default disabled, "
                         "so the memory budget is the only guardrail unless "
                         "asked for. Runtime, not memory, is what makes the "
                         "full grid expensive")
    ap.add_argument("--device", action="append", default=None,
                    choices=["cpu", "gpu", "tpu"],
                    help="restrict to this backend (repeatable); default is "
                         "every backend present")
    ap.add_argument("--out", type=Path, default=BENCH_DIR,
                    help="output directory (default benchmarks/)")
    args = ap.parse_args(argv)

    devices = available_devices(args.device)
    report = run_sweep(args.out, quick=args.quick, max_mem_gb=args.max_mem_gb,
                       max_seconds=args.max_seconds, devices=devices)
    return 0 if report["run"]["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
