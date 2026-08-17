#!/usr/bin/env python
"""ONE command that regenerates every manuscript figure from committed artifacts.

    python scripts/make_paper_figures.py --all

No manual steps, no notebook, no editing paths. Every figure is declared in the
:data:`FIGURES` registry as ``figure id -> FigureSpec(builder, inputs, outputs)``
and the driver does the rest: check inputs, build, save PDF + PNG, write a
provenance sidecar, print a summary table.

WHY REGISTRY-DRIVEN
-------------------
The figure ids here are the ids to use in the manuscript (``fig4_budget``, not
"the budget one"), so a referee asking "where did panel 4b come from" is
answered by one lookup rather than by archaeology. Adding a figure means adding
one :class:`FigureSpec`; the driver, the sidecars and the summary table need no
change.

READ ARTIFACTS, DO NOT RE-RUN THE SOLVER
----------------------------------------
Builders read committed ``.npz``/``.json`` artifacts. The exceptions are marked
``recompute=True`` and are cheap by construction:

* ``fig1_verification`` panel (a) re-runs the four deterministic parameter sets
  and differences them against the COMMITTED goldens in ``tests/data/golden/``.
  That is the measurement -- a ULP residual needs a fresh run to compare
  against -- and it costs ~40 s.
* ``fig2_numerics`` panel (b) runs four short vacuum-only solves (~15 s), and
  panel (c) integrates a scalar ODE (milliseconds).

Everything else is pure artifact reading and takes no solver time at all.

MISSING INPUTS NEVER CRASH
--------------------------
Three outcomes, and the summary table names which applies to every figure:

``built``       every input present, figure written.
``partial``     the figure has panels whose input is not committed. The
                available panels are drawn and each missing one becomes an
                annotated placeholder naming the exact command that would
                produce it -- a manuscript figure must not silently lose a
                panel, and it must not vanish because one panel's study was
                never run.
``skipped``     a REQUIRED input is absent. Reported with the missing path and
                what would generate it. Exit status is still 0: a skip is a
                stated fact about this checkout, not a failure.

Panels (b), (c) and (d) of ``fig1_verification`` are the standing ``partial``
case in this repository: ``validation/convergence.py`` and
``validation/analytic_cw.py`` write no JSON artifact, so there is nothing
committed to read, and their studies (a dt ladder; 256 realizations for the
weak order) are far past the 60 s recompute bar. Pass ``--recompute-heavy`` to
run them in-process anyway.

PROVENANCE
----------
Every figure gets ``<stem>.provenance.json`` listing each input artifact with
its sha256, byte size and mtime, plus the builder, the git commit and the
environment. A referee can therefore trace any panel back to the exact run that
produced its data, and a figure regenerated from changed inputs is detectable by
comparing hashes rather than by eyeballing the plot.

OUTPUTS
-------
Both formats, always: ``.pdf`` (vector, for the manuscript) and ``.png``
(300 dpi, for drafts and slides). ``analysis.plot_utils.apply_pub_style`` sets
the shared house style; note it pins ``savefig.dpi`` to 150, so the PNG save
overrides dpi explicitly rather than inheriting it.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.plot_utils import apply_pub_style  # noqa: E402

__all__ = [
    "FIGURES",
    "FigureSpec",
    "FigureResult",
    "PNG_DPI",
    "build_figure",
    "main",
]

#: PNG resolution. apply_pub_style pins savefig.dpi to 150 for the analysis
#: figures; manuscript rasters want 300, so every PNG save passes dpi explicitly.
PNG_DPI = 300

DEFAULT_OUT = REPO_ROOT / "paper" / "figures"

RESULTS = REPO_ROOT / "analysis" / "results"
VALIDATION_RESULTS = REPO_ROOT / "validation" / "results"
GOLDEN_DIR = REPO_ROOT / "tests" / "data" / "golden"
BENCH_DIR = REPO_ROOT / "benchmarks"
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FigureSpec:
    """One manuscript figure.

    Attributes:
        builder: ``(ctx) -> (matplotlib Figure, notes dict)``.
        inputs: REQUIRED artifacts. Any missing one skips the figure.
        optional_inputs: artifacts a builder uses when present. Missing ones
            degrade the figure to ``partial``, never to a crash.
        outputs: written stems (the driver appends .pdf and .png).
        recompute: True when the builder deliberately runs the solver because
            the figure IS a fresh-vs-committed comparison or is trivially cheap.
        recompute_budget_s: the cost that justified ``recompute``; printed by
            ``--list`` so the 60 s bar stays visible rather than remembered.
        heavy: builder can do more when ``--recompute-heavy`` is passed.
    """

    fig_id: str
    builder: Callable[["BuildContext"], tuple[Any, dict]]
    inputs: tuple[Path, ...] = ()
    optional_inputs: tuple[Path, ...] = ()
    stem: str = ""
    recompute: bool = False
    recompute_budget_s: float | None = None
    heavy: bool = False
    description: str = ""

    def outputs(self, out_dir: Path) -> tuple[Path, Path]:
        stem = self.stem or self.fig_id
        return (out_dir / f"{stem}.pdf", out_dir / f"{stem}.png")

    def sidecar(self, out_dir: Path) -> Path:
        stem = self.stem or self.fig_id
        return out_dir / f"{stem}.provenance.json"


@dataclass
class BuildContext:
    """What a builder is handed."""

    spec: "FigureSpec"
    out_dir: Path
    recompute_heavy: bool = False
    notes: dict = field(default_factory=dict)
    #: Panels the builder could not draw: label -> what would produce it.
    missing_panels: dict[str, str] = field(default_factory=dict)


@dataclass
class FigureResult:
    fig_id: str
    status: str                      # built | partial | skipped | failed
    outputs: list[str] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)
    message: str = ""
    elapsed_s: float = 0.0
    missing_panels: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise.

    ``Path.relative_to`` RAISES for anything outside the repository, so calling
    it bare would crash the whole driver the moment someone passes
    ``--out /tmp/figures`` or registers an artifact living elsewhere. A display
    helper must never be able to take the run down.
    """
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_record(path: Path) -> dict:
    """sha256 + size + mtime of one input, or a present=False stub."""
    path = Path(path)
    if not path.exists():
        return {"path": _rel(path), "present": False}
    stat = path.stat()
    return {
        "path": _rel(path),
        "present": True,
        "sha256": _sha256_file(path),
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _git_commit() -> str:
    try:
        from simulator.provenance import _git_commit as _gc

        return _gc(short=True)
    except Exception:  # noqa: BLE001
        return "unknown"


def _env() -> dict:
    import platform

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    try:
        import jax

        env["jax"] = jax.__version__
    except Exception:  # noqa: BLE001
        pass
    return env


def _write_sidecar(spec: FigureSpec, out_dir: Path, records: list[dict],
                   notes: dict, status: str, missing: dict) -> Path:
    path = spec.sidecar(out_dir)
    payload = {
        "figure_id": spec.fig_id,
        "status": status,
        "description": spec.description,
        "builder": f"{spec.builder.__module__}.{spec.builder.__name__}",
        "recompute": spec.recompute,
        "outputs": [_rel(p) for p in spec.outputs(out_dir)],
        "input_artifacts": records,
        "missing_panels": missing,
        "notes": notes,
        "git_commit": _git_commit(),
        "environment": _env(),
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/make_paper_figures.py",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_clean(payload), fh, indent=2, default=str)
        fh.write("\n")
    return path


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        yield buf


def _placeholder(ax, title: str, how: str) -> None:
    """Draw a panel that says what is missing and how to produce it.

    A blank panel in a manuscript draft is a bug report waiting to happen; a
    panel that names the command is a to-do list.
    """
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linestyle((0, (4, 4)))
        spine.set_color("0.6")
    ax.text(0.5, 0.5, "input artifact not committed\n\n" + how,
            ha="center", va="center", transform=ax.transAxes, fontsize=7.5,
            color="0.35", wrap=True,
            bbox=dict(boxstyle="round,pad=0.6", fc="0.96", ec="0.75", lw=0.8))


def _load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _panel_label(ax, letter: str) -> None:
    ax.text(-0.02, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------------------
# fig1 -- verification
# ---------------------------------------------------------------------------
def build_fig1_verification(ctx: BuildContext):
    """(a) all-off ULP residual, (b) analytic-CW residual, (c) order, (d) weak order."""
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6))
    notes: dict = {}

    # ---- (a) all-off ULP residual vs the committed goldens -----------------
    ax = axes[0, 0]
    _panel_label(ax, "a")
    from validation.noise_off_identity import ARRAY_NAMES, PARAM_SETS, run_deterministic

    labels, ulp_max, ulp_mean = [], [], []
    for pset in PARAM_SETS:
        golden_path = GOLDEN_DIR / f"{pset['name']}.npz"
        if not golden_path.exists():
            continue
        with _quiet():
            fresh = run_deterministic(pset, seed=0)
        golden = np.load(golden_path, allow_pickle=False)
        worst, mean_acc, n_acc = 0.0, 0.0, 0
        for name in ARRAY_NAMES:
            if name not in golden.files:
                continue
            a = np.asarray(fresh[name]).ravel()
            b = np.asarray(golden[name]).ravel()
            if a.shape != b.shape:
                continue
            if np.iscomplexobj(a) or np.iscomplexobj(b):
                a = np.concatenate([a.real, a.imag])
                b = np.concatenate([b.real, b.imag])
            a = a.astype(np.float64)
            b = b.astype(np.float64)
            # ULP distance: |a-b| divided by the spacing of the representable
            # doubles at b. Exact bit-identity is 0 ULP.
            spacing = np.spacing(np.abs(b))
            spacing[spacing == 0] = np.finfo(np.float64).tiny
            ulp = np.abs(a - b) / spacing
            worst = max(worst, float(np.max(ulp)))
            mean_acc += float(np.sum(ulp))
            n_acc += ulp.size
        labels.append(pset["name"])
        ulp_max.append(worst)
        ulp_mean.append(mean_acc / max(n_acc, 1))

    if labels:
        x = np.arange(len(labels))
        # A log axis cannot show 0, and an exact 0 is the RESULT here, so bars
        # are drawn at a display floor and the axis label says so -- otherwise
        # a bar sitting at 1e-2 reads as "0.01 ULP of drift", which is the
        # opposite of what was measured.
        floor = 1e-2
        ax.bar(x - 0.2, np.maximum(ulp_max, floor), 0.4, label="max ULP",
               color="C3")
        ax.bar(x + 0.2, np.maximum(ulp_mean, floor), 0.4, label="mean ULP",
               color="C0")
        for xi, value in zip(x, ulp_max):
            if value == 0.0:
                ax.text(xi, floor * 1.25, "0", ha="center", va="bottom",
                        fontsize=7, color="0.2", fontweight="bold")
        ax.axhline(1.0, color="0.4", ls="--", lw=0.9, label="1 ULP")
        ax.set_yscale("log")
        ax.set_ylim(floor * 0.7, 2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("ULP distance from committed golden\n"
                      f"(exact 0 drawn at the {floor:g} display floor)")
        ax.set_title("Noise-off determinism: fresh run vs golden")
        ax.legend(fontsize=7)
        bitwise = all(v == 0.0 for v in ulp_max)
        notes["fig1a_bitwise_identical"] = bool(bitwise)
        notes["fig1a_max_ulp"] = dict(zip(labels, ulp_max))
        if bitwise:
            ax.text(0.5, 0.9, "bit-identical (0 ULP on every array)",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    color="C2", fontweight="bold")
    else:
        _placeholder(ax, "Noise-off determinism",
                     "no goldens in tests/data/golden/\n"
                     "run: python -m validation.noise_off_identity --write-golden")
        ctx.missing_panels["fig1a"] = "tests/data/golden/*.npz absent"

    # ---- (b) analytic-CW residual vs detuning ------------------------------
    ax = axes[0, 1]
    _panel_label(ax, "b")
    if ctx.recompute_heavy:
        from validation.analytic_cw import verify

        with _quiet():
            res = verify()
        grid = np.asarray(res["delta_over_kappa"])
        drawn = 0
        for key in ("rel_continuum", "rel_discrete"):
            block = res.get(key)
            if block is None:
                continue
            arr = np.atleast_2d(np.asarray(block, dtype=np.float64))
            for row in arr:
                if row.size == grid.size:
                    ax.semilogy(grid, np.abs(row), lw=1.0,
                                label=key if drawn < 2 else None)
                    drawn += 1
        ax.axhline(1e-12, color="0.4", ls="--", lw=0.9, label="1e-12")
        ax.set_xlabel(r"detuning $\delta\omega/\kappa$")
        ax.set_ylabel("relative residual")
        ax.set_title("Analytic CW steady state: solver residual")
        ax.legend(fontsize=7)
        notes["fig1b_source"] = "recomputed via validation.analytic_cw.verify()"
    else:
        _placeholder(
            ax, "Analytic CW residual vs detuning",
            "validation/analytic_cw.py writes no JSON artifact.\n\n"
            "run:  python -m validation.figures.fig_analytic_cw\n"
            "or:   --recompute-heavy to build it here")
        ctx.missing_panels["fig1b"] = (
            "no committed artifact; validation.analytic_cw.verify() is not run "
            "by default (pass --recompute-heavy)")

    # ---- (c) observed order of accuracy, 3 curves --------------------------
    ax = axes[1, 0]
    _panel_label(ax, "c")
    # ---- (d) weak order, noise on ------------------------------------------
    ax_d = axes[1, 1]
    _panel_label(ax_d, "d")

    if ctx.recompute_heavy:
        from validation.convergence import deterministic_study, weak_study

        with _quiet():
            det = deterministic_study()
        for key, block in det.items():
            dts = np.asarray(block["dts"], dtype=np.float64)
            errs = np.asarray(block["errors"], dtype=np.float64)
            ax.loglog(dts, errs, "-o", ms=3.5, label=f"{key} "
                      f"(p={block.get('observed_order', float('nan')):.2f})")
        if det:
            ref = np.asarray(next(iter(det.values()))["dts"], dtype=np.float64)
            base = float(np.asarray(
                next(iter(det.values()))["errors"], dtype=np.float64)[0])
            ax.loglog(ref, base * (ref / ref[0]), "--", color="0.6", lw=0.9,
                      label=r"$dt^1$")
            ax.loglog(ref, base * (ref / ref[0]) ** 2, ":", color="0.6",
                      lw=0.9, label=r"$dt^2$")
        ax.set_xlabel("dt [s]")
        ax.set_ylabel("error")
        ax.set_title("Observed order of accuracy")
        ax.legend(fontsize=7)

        with _quiet():
            weak = weak_study()
        dts = np.asarray(weak["dts"], dtype=np.float64)
        errs = np.asarray(weak["errors"], dtype=np.float64)
        ax_d.loglog(dts, errs, "-o", ms=4, color="C1",
                    label=f"weak error (p={weak.get('observed_order', float('nan')):.2f})")
        ax_d.set_xlabel("dt [s]")
        ax_d.set_ylabel(r"$|E[|E|^2]_{dt} - E[|E|^2]_{ref}|$")
        ax_d.set_title("Weak order, quantum-vacuum channel on")
        ax_d.legend(fontsize=7)
        notes["fig1cd_source"] = "recomputed via validation.convergence"
    else:
        how = ("validation/convergence.py writes no JSON artifact, and the\n"
               "study (dt ladder; 256 realizations for the weak order) is far\n"
               "past the 60 s recompute bar.\n\n"
               "run:  python -m validation.figures.fig_convergence\n"
               "or:   --recompute-heavy to build it here")
        _placeholder(ax, "Observed order of accuracy (3 configurations)", how)
        _placeholder(ax_d, "Weak order, noise on", how)
        ctx.missing_panels["fig1c"] = (
            "no committed artifact; validation.convergence.deterministic_study() "
            "not run by default (pass --recompute-heavy)")
        ctx.missing_panels["fig1d"] = (
            "no committed artifact; validation.convergence.weak_study() "
            "not run by default (pass --recompute-heavy)")

    fig.suptitle("Figure 1 — Solver verification", fontweight="bold")
    return fig, notes


# ---------------------------------------------------------------------------
# fig2 -- numerics
# ---------------------------------------------------------------------------
def build_fig2_numerics(ctx: BuildContext):
    """(a) TRN AR(1) burn-in, (b) vacuum floor vs n_tau, (c) Euler vs exponential."""
    apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    notes: dict = {}

    # ---- (a) TRN AR(1) burn-in, before vs after ----------------------------
    ax = axes[0]
    _panel_label(ax, "a")
    burnin = np.load(RESULTS / "validation" / "trn_burnin.npz", allow_pickle=False)
    t_slow = np.asarray(burnin["t_slow"], dtype=np.float64)
    ax.semilogx(t_slow, burnin["legacy_ratio"], "-o", ms=4, color="C3",
                label="legacy (zero start)")
    ax.semilogx(t_slow, burnin["stationary_ratio"], "-s", ms=4, color="C0",
                label="stationary init")
    if "legacy_ratio_analytic" in burnin.files:
        ax.semilogx(t_slow, burnin["legacy_ratio_analytic"], "--", color="0.5",
                    lw=1.0, label="legacy, analytic")
    ax.axhline(1.0, color="0.4", ls=":", lw=0.9)
    ax.set_xlabel(r"record length $t_{slow}$ [round trips]")
    ax.set_ylabel(r"measured / target  $\sigma^2_{\Delta T}$")
    ax.set_title("TRN AR(1) burn-in")
    ax.legend(fontsize=7)
    notes["fig2a_t_slow"] = t_slow.tolist()

    # ---- (b) vacuum floor vs n_tau (cheap recompute) -----------------------
    ax = axes[1]
    _panel_label(ax, "b")
    try:
        floors = _vacuum_floor_vs_n_tau()
        n_taus = [f[0] for f in floors]
        occ = [f[1] for f in floors]
        ax.semilogx(n_taus, occ, "-o", ms=5, color="C0", base=2,
                    label=r"measured $\langle n_\mu\rangle$")
        ax.axhline(0.5, color="C3", ls="--", lw=1.1,
                   label=r"symmetric-ordered vacuum, $\frac{1}{2}$ photon/mode")
        ax.set_xlabel(r"$n_\tau$")
        ax.set_ylabel(r"$\langle n_\mu\rangle$  [photons/mode]")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Vacuum floor vs grid size")
        ax.legend(fontsize=7)
        notes["fig2b_occupancy"] = dict(zip(map(str, n_taus), occ))
    except Exception as exc:  # noqa: BLE001
        _placeholder(ax, "Vacuum floor vs $n_\\tau$",
                     f"vacuum recompute unavailable:\n{str(exc)[:120]}")
        ctx.missing_panels["fig2b"] = f"recompute failed: {str(exc)[:160]}"

    # ---- (c) Euler vs exponential thermal integrator at dt/tau = 2.1 -------
    ax = axes[2]
    _panel_label(ax, "c")
    ratio = 2.1
    n_steps = 24
    steps = np.arange(n_steps + 1)
    # dT' = -dT/tau + F, driven to a unit steady state, started away from it.
    # Euler amplification is (1 - dt/tau) = -1.1, |.| > 1 -> divergence with a
    # sign flip every step. The exponential update uses exp(-dt/tau), which is
    # unconditionally stable and lands on the same fixed point.
    dt_over_tau = ratio
    euler, expo = [2.0], [2.0]
    for _ in range(n_steps):
        euler.append(euler[-1] + dt_over_tau * (1.0 - euler[-1]))
        decay = math.exp(-dt_over_tau)
        expo.append(expo[-1] * decay + (1.0 - decay) * 1.0)
    ax.plot(steps, np.abs(np.asarray(euler)), "-o", ms=3, color="C3",
            label=f"Euler  |1-dt/$\\tau$| = {abs(1 - ratio):.1f}")
    ax.plot(steps, np.abs(np.asarray(expo)), "-s", ms=3, color="C0",
            label=f"exponential  $e^{{-dt/\\tau}}$ = {math.exp(-ratio):.3f}")
    ax.axhline(1.0, color="0.4", ls=":", lw=0.9, label="fixed point")
    ax.set_yscale("log")
    ax.set_xlabel("thermal step")
    ax.set_ylabel(r"$|\Delta T| / \Delta T_{ss}$")
    ax.set_title(rf"Thermal integrator at $dt/\tau_{{th}}$ = {ratio}")
    ax.legend(fontsize=7)
    notes["fig2c_euler_amplification"] = abs(1.0 - ratio)
    notes["fig2c_euler_diverges"] = bool(abs(1.0 - ratio) > 1.0)

    fig.suptitle("Figure 2 — Numerical choices that matter", fontweight="bold")
    return fig, notes


def _vacuum_floor_vs_n_tau(n_taus=(128, 256, 512, 1024), t_slow=400):
    """Undriven vacuum occupation per mode; expect 1/2 photon (symmetric order).

    Four short quantum-noise-only solves. Normalization is the solver's own
    convention: ``<n_mu> = |fft(E)_mu|^2 / (n_tau^2 * hbar*omega0)``.
    """
    import jax

    from simulator.lle_solver import (
        _load_config, hbar_omega0_from_config, solve_lle_ssfm_jax,
    )
    from simulator.noise_config import NoiseConfig

    hbar_omega0 = hbar_omega0_from_config(_load_config(CONFIG_PATH))
    out = []
    for n_tau in n_taus:
        with _quiet():
            sol = solve_lle_ssfm_jax(
                pin=0.0,
                delta_omega=np.zeros((1, int(t_slow)), dtype=np.float32),
                t_slow=int(t_slow), beta=[1.578e-18], kappa=1.519e8,
                kappa_c=1.215e8, rng_key=jax.random.PRNGKey(0),
                n_tau=int(n_tau), snapshot_interval=max(t_slow // 8, 1),
                config_path=str(CONFIG_PATH),
                noise_config=NoiseConfig.all_off(quantum_vacuum=True),
            )
        snaps = np.asarray(sol["E_snapshots"])[0]
        # Discard the first half: the cold start relaxes onto the floor.
        snaps = snaps[snaps.shape[0] // 2:]
        power = np.abs(np.fft.fft(snaps, axis=-1)) ** 2
        occupancy = float(np.mean(power) / (n_tau ** 2 * hbar_omega0))
        out.append((int(n_tau), occupancy))
    return out


# ---------------------------------------------------------------------------
# fig3 -- pyLLE cross-check
# ---------------------------------------------------------------------------
def build_fig3_pylle(ctx: BuildContext):
    """Spectrum + waveform overlay with a residual sub-panel."""
    apply_pub_style()
    fields = np.load(VALIDATION_RESULTS / "pylle_crosscheck_v2_fields.npz",
                     allow_pickle=False)
    notes: dict = {}

    ours_key = next((k for k in ("field_ours_n8", "field_ours_n4",
                                 "field_ours_n16", "field_ours_n1")
                     if k in fields.files), None)
    theirs_key = next((k for k in fields.files
                       if k.startswith("field_pylle_") and "tight" in k), None)
    if theirs_key is None:
        theirs_key = next((k for k in fields.files
                           if k.startswith("field_pylle_")), None)
    if ours_key is None or theirs_key is None:
        raise KeyError("pyLLE cross-check fields npz has no comparable pair")

    ours = np.asarray(fields[ours_key])
    theirs = np.asarray(fields[theirs_key])
    mu = np.asarray(fields["mu"]) if "mu" in fields.files else np.arange(ours.size)

    def _dbc(field):
        p = np.abs(np.fft.fftshift(np.fft.fft(field))) ** 2
        return 10.0 * np.log10(np.maximum(p / p.max(), 1e-300))

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.22)

    # --- spectra ---
    ax = fig.add_subplot(gs[0, 0])
    _panel_label(ax, "a")
    s_ours, s_theirs = _dbc(ours), _dbc(theirs)
    mu_shift = np.fft.fftshift(mu) if mu.size == ours.size else np.arange(ours.size)
    ax.plot(mu_shift, s_ours, lw=0.9, color="C0", label=f"ours ({ours_key})")
    ax.plot(mu_shift, s_theirs, lw=0.9, color="C3", ls="--",
            label=f"pyLLE ({theirs_key})")
    ax.set_ylabel("power [dBc]")
    ax.set_title("Optical spectrum")
    ax.legend(fontsize=7)
    ax.tick_params(labelbottom=False)

    axr = fig.add_subplot(gs[1, 0], sharex=ax)
    axr.plot(mu_shift, s_ours - s_theirs, lw=0.8, color="0.3")
    axr.axhline(0.0, color="C2", lw=0.9)
    axr.set_xlabel(r"mode index $\mu$")
    axr.set_ylabel("residual [dB]")

    # --- waveforms ---
    ax2 = fig.add_subplot(gs[0, 1])
    _panel_label(ax2, "b")
    tau = np.linspace(-0.5, 0.5, ours.size, endpoint=False)
    ax2.plot(tau, np.abs(ours) ** 2, lw=0.9, color="C0", label="ours")
    ax2.plot(tau, np.abs(theirs) ** 2, lw=0.9, color="C3", ls="--",
             label="pyLLE")
    ax2.set_ylabel(r"$|E|^2$ [J]")
    ax2.set_title("Intracavity waveform")
    ax2.legend(fontsize=7)
    ax2.tick_params(labelbottom=False)

    ax2r = fig.add_subplot(gs[1, 1], sharex=ax2)
    resid = np.abs(ours) ** 2 - np.abs(theirs) ** 2
    ax2r.plot(tau, resid, lw=0.8, color="0.3")
    ax2r.axhline(0.0, color="C2", lw=0.9)
    ax2r.set_xlabel(r"fast time $\tau / t_r$")
    ax2r.set_ylabel("residual [J]")

    denom = float(np.max(np.abs(theirs) ** 2))
    notes["fig3_pair"] = {"ours": ours_key, "pylle": theirs_key}
    notes["fig3_max_abs_spectral_residual_db"] = float(
        np.max(np.abs(s_ours - s_theirs)))
    notes["fig3_peak_rel_waveform_residual"] = (
        float(np.max(np.abs(resid)) / denom) if denom > 0 else None)

    fig.suptitle("Figure 3 — pyLLE cross-check", fontweight="bold")
    return fig, notes


# ---------------------------------------------------------------------------
# fig4 -- noise budget
# ---------------------------------------------------------------------------
def _budget_headline_point(cells: dict, observable: str,
                           statistic: str = "mean") -> str | None:
    """The point of a multi-point observable with the most PLOTTABLE cells.

    Plottable means finite and strictly positive, because this figure uses a log
    axis. Counting merely non-null values would happily select a point whose
    every cell is exactly 0 -- valid data, invisible on a log scale -- and the
    group would render empty for no stated reason.
    """
    counts: dict[str, int] = {}
    for obs_map in cells.values():
        cell = obs_map.get(observable)
        if not cell:
            continue
        for point, blob in cell.get("points", {}).items():
            stats = blob.get("statistics") or {}
            value = stats.get(statistic)
            ok = (value is not None and math.isfinite(float(value))
                  and float(value) > 0.0)
            counts[point] = counts.get(point, 0) + (1 if ok else 0)
    if not counts:
        return None
    best = max(counts.values())
    if best == 0:
        # Points exist but none is plottable. Returning the first one anyway
        # would draw an empty group with a confident axis label on it.
        return None
    return next(p for p, n in counts.items() if n == best)


def build_fig4_budget(ctx: BuildContext):
    """Grouped bars: 5 observables x channel sets, log axis, error bars."""
    apply_pub_style()
    budget = _load_json(RESULTS / "budget" / "budget.json")
    cells = budget["cells"]
    notes: dict = {"budget_quick": budget["run"].get("quick"),
                   "budget_seeds": budget["run"].get("seeds")}

    observables = list(budget.get("derived", {}).keys()) or [
        "u_int_rms_frac", "s_rep", "line_linewidth", "timing_jitter",
        "step_jitter"]
    set_names = [s for s in budget.get("channel_sets", {}) if s in cells]

    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    n_sets = max(len(set_names), 1)
    width = 0.8 / n_sets
    cmap = plt.get_cmap("tab20")
    drawn_any = False
    xticks, xlabels = [], []

    empty_groups: dict[str, str] = {}
    for gi, obs in enumerate(observables):
        stat = "std" if obs == "step_jitter" else "mean"
        point = _budget_headline_point(cells, obs, stat)
        if point is None:
            # Keep the group SLOT so the reader sees the observable was in the
            # budget, and say why nothing is drawn. An unexplained gap in a
            # manuscript figure reads as a plotting bug.
            any_cell = any(obs in obs_map for obs_map in cells.values())
            why = ("all cells are 0 (log axis cannot show zero)"
                   if any_cell else "not present in this budget")
            if any_cell:
                # Distinguish "0 everywhere" from "null everywhere".
                seen_zero = any(
                    (blob.get("statistics") or {}).get(stat) == 0
                    for obs_map in cells.values()
                    for blob in (obs_map.get(obs, {}).get("points", {})
                                 or {}).values())
                if not seen_zero:
                    why = "no resolved cell in this budget run"
            empty_groups[obs] = why
            xticks.append(gi)
            xlabels.append(f"{obs}\n[—]")
            ax.annotate(why, xy=(gi, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, -18), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5, color="0.45",
                        style="italic")
            continue
        unit = ""
        for si, set_name in enumerate(set_names):
            cell = cells.get(set_name, {}).get(obs)
            if not cell:
                continue
            blob = cell.get("points", {}).get(point, {})
            stats = blob.get("statistics") or {}
            value = stats.get(stat)
            unit = unit or cell.get("unit", "")
            if value is None or not math.isfinite(float(value)) or value <= 0:
                continue
            err = stats.get("sem")
            err = float(err) if err and math.isfinite(float(err)) else None
            ax.bar(gi + si * width - 0.4 + width / 2, float(value), width,
                   color=cmap(si % 20),
                   yerr=err, capsize=1.6, ecolor="0.25",
                   error_kw={"lw": 0.7},
                   label=set_name if gi == 0 else None)
            drawn_any = True
        xticks.append(gi)
        label = obs if point == "value" else f"{obs}\n({point})"
        xlabels.append(f"{label}\n[{unit}]")

    if not drawn_any:
        raise ValueError("no finite budget cell to plot")

    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylabel("value (log scale; note the differing units per group)")
    ax.set_title(
        "Figure 4 — Noise budget by channel set "
        f"({'QUICK smoke run' if budget['run'].get('quick') else 'production'}, "
        f"{budget['run'].get('seeds')} seeds, common random numbers)",
        fontweight="bold")
    ax.legend(fontsize=6.5, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    notes["fig4_observables"] = observables
    notes["fig4_channel_sets"] = set_names
    if empty_groups:
        notes["fig4_empty_groups"] = empty_groups
        for obs, why in empty_groups.items():
            ctx.missing_panels[f"fig4_{obs}"] = why
    return fig, notes


# ---------------------------------------------------------------------------
# fig5 -- S_rep overlay
# ---------------------------------------------------------------------------
def build_fig5_srep(ctx: BuildContext):
    """S_rep at the budget report frequencies, per channel, with analytic limits."""
    apply_pub_style()
    budget = _load_json(RESULTS / "budget" / "budget.json")
    cells = budget["cells"]
    notes: dict = {}

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cmap = plt.get_cmap("tab10")
    plotted = 0
    for i, set_name in enumerate(budget.get("channel_sets", {})):
        cell = cells.get(set_name, {}).get("s_rep")
        if not cell:
            continue
        freqs, vals, errs = [], [], []
        for point, blob in cell.get("points", {}).items():
            stats = blob.get("statistics") or {}
            value = stats.get("mean")
            if value is None or not math.isfinite(float(value)) or value <= 0:
                continue
            target = (blob.get("notes") or {}).get("target_hz")
            if target is None:
                continue
            freqs.append(float(target))
            vals.append(float(value))
            sem = stats.get("sem")
            errs.append(float(sem) if sem and math.isfinite(float(sem)) else 0.0)
        if not freqs:
            continue
        order = np.argsort(freqs)
        f = np.asarray(freqs)[order]
        v = np.asarray(vals)[order]
        e = np.asarray(errs)[order]
        style = "-o" if set_name == "all_on" else "--s"
        lw = 2.0 if set_name == "all_on" else 1.1
        ax.errorbar(f, v, yerr=e, fmt=style, ms=5, lw=lw,
                    color=cmap(i % 10), capsize=2.5, label=set_name)
        plotted += 1

    # --- analytic / measured limit lines from the validation campaign -------
    campaign_path = RESULTS / "validation" / "campaign_report.json"
    if campaign_path.exists():
        w4 = _load_json(campaign_path).get("workstream4_beatnote", {})
        trn_limit = w4.get("trn_limit_srep_at_1e6hz")
        quantum_floor = w4.get("srep_quantum_limited_floor_hz2_per_hz")
        if trn_limit:
            ax.axhline(float(trn_limit), color="C3", ls="-.", lw=1.3,
                       label="TRN (K-G) + FSR limit @ 1 MHz")
            notes["trn_limit_srep_at_1e6hz"] = float(trn_limit)
        if quantum_floor:
            ax.axhline(float(quantum_floor), color="0.35", ls=":", lw=1.3,
                       label="quantum-limited floor")
            notes["srep_quantum_limited_floor"] = float(quantum_floor)

    if plotted == 0:
        raise ValueError("no channel set has a finite S_rep point to plot")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offset frequency $f$ [Hz]")
    ax.set_ylabel(r"$S_{rep}(f)$  [Hz$^2$/Hz]")
    ax.set_title("Figure 5 — Repetition-rate noise by channel", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.text(0.02, 0.03,
            "Points are the budget's discrete report frequencies, not a "
            "continuous PSD:\nno full S_rep(f) spectrum is committed. "
            f"{'Budget is a QUICK run, so most points are unresolved.' if budget['run'].get('quick') else ''}",
            transform=ax.transAxes, fontsize=6.5, color="0.35", va="bottom")
    notes["fig5_sets_plotted"] = plotted
    return fig, notes


# ---------------------------------------------------------------------------
# fig6 -- runtime
# ---------------------------------------------------------------------------
def build_fig6_runtime(ctx: BuildContext):
    """ns/round-trip/mode vs n_tau, per backend, with an n log n reference."""
    apply_pub_style()
    report = _load_json(BENCH_DIR / "runtime.json")
    ok = [p for p in report["points"] if p.get("status") == "ok"]
    if not ok:
        raise ValueError("runtime.json has no successful points")
    notes: dict = {}

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    devices = sorted({p["device"] for p in ok})
    markers = {"all_off": "o", "all_on": "s"}
    # Colour by (device, noise), not by device alone: two curves in the same
    # blue distinguished only by dash pattern is not a readable figure.
    colors = {("cpu", "all_off"): "C0", ("cpu", "all_on"): "C9",
              ("gpu", "all_off"): "C3", ("gpu", "all_on"): "C1",
              ("tpu", "all_off"): "C2", ("tpu", "all_on"): "C8"}
    slices_used: dict[str, str] = {}
    for dev in devices:
        for noise in ("all_off", "all_on"):
            sel = [p for p in ok if p["device"] == dev and p["noise"] == noise]
            if not sel:
                continue
            # A scaling curve must hold the WORKLOAD fixed while n_tau varies.
            # Taking a median over whatever (t_slow, n_traj) survived the
            # guardrail at each n_tau silently compares different amounts of
            # work per point and puts kinks in the curve that are pure
            # aggregation artefact. Pick the single (t_slow, n_traj) slice with
            # the widest n_tau coverage and plot only that.
            combos: dict[tuple[int, int], list] = {}
            for p in sel:
                combos.setdefault((p["t_slow"], p["n_traj"]), []).append(p)
            (t_slow, n_traj), pts = max(
                combos.items(),
                key=lambda kv: (len({p["n_tau"] for p in kv[1]}),
                                kv[0][0] * kv[0][1]))
            pts.sort(key=lambda p: p["n_tau"])
            ax.loglog([p["n_tau"] for p in pts],
                      [p["ns_per_roundtrip_per_mode"] for p in pts],
                      marker=markers[noise], ms=5,
                      color=colors.get((dev, noise), "C4"),
                      ls="-" if noise == "all_off" else "--",
                      label=f"{dev.upper()} {noise}  "
                            f"($t_{{slow}}$={t_slow:g}, $n_{{traj}}$={n_traj})",
                      base=2)
            slices_used[f"{dev}/{noise}"] = f"t_slow={t_slow}, n_traj={n_traj}"
    notes["fig6_slices"] = slices_used

    # n log n reference: constant total work per mode-round-trip would be flat,
    # so the FFT's per-mode cost grows as log2(n) -- anchored on the leftmost
    # measured all_off point so the eye compares slopes, not offsets.
    anchor = [p for p in ok if p["noise"] == "all_off"]
    if anchor:
        n0 = min(p["n_tau"] for p in anchor)
        base_vals = [p["ns_per_roundtrip_per_mode"] for p in anchor
                     if p["n_tau"] == n0]
        y0 = float(np.min(base_vals))
        grid = np.array(sorted({p["n_tau"] for p in anchor}), dtype=float)
        ax.loglog(grid, y0 * np.log2(grid) / math.log2(n0), ":", color="0.45",
                  lw=1.4, base=2,
                  label=r"$n\log n$ reference ($\propto \log_2 n_\tau$)")
        notes["fig6_reference_anchor_n_tau"] = int(n0)

    fits = report.get("scaling_fits", {})
    if fits:
        bits = []
        for key, fit in sorted(fits.items()):
            cb = fit.get("alpha_total_compute_bound")
            bits.append(f"{key}: " + r"$\alpha$=" + f"{fit['alpha_total']:.2f}"
                        + (f" (compute-bound {cb:.2f})" if cb else ""))
        ax.text(0.02, 0.97, "\n".join(bits), transform=ax.transAxes,
                fontsize=6.8, va="top", color="0.25",
                bbox=dict(boxstyle="round,pad=0.4", fc="0.97", ec="0.8", lw=0.7))
        notes["fig6_scaling_fits"] = {k: v.get("alpha_total")
                                      for k, v in fits.items()}

    hw = report.get("hardware", {})
    models = "; ".join(f"{d['platform'].upper()} {d['model']}"
                       for d in hw.get("devices", []))
    ax.set_xlabel(r"$n_\tau$")
    ax.set_ylabel("ns per round-trip per mode")
    ax.set_title("Figure 6 — Solver runtime scaling", fontweight="bold")
    ax.legend(fontsize=7)
    fig.supxlabel(
        f"Hardware: {models}. Each curve holds ($t_{{slow}}$, $n_{{traj}}$) "
        "fixed at the slice with the widest $n_\\tau$ coverage (shown in the "
        "legend), so the abscissa is the only thing varying.",
        fontsize=6.8, color="0.3")
    notes["fig6_hardware"] = models
    missing_gpu = [b for b in ("cpu", "gpu") if b not in devices]
    if missing_gpu:
        notes["fig6_backends_absent"] = missing_gpu
        ctx.missing_panels["fig6_gpu"] = (
            "no GPU backend on the machine that produced benchmarks/runtime.json; "
            "re-run the benchmark on a GPU host to add that curve")
    return fig, notes


# ---------------------------------------------------------------------------
# fig7 -- experiment (optional)
# ---------------------------------------------------------------------------
def build_fig7_experiment(ctx: BuildContext):
    """Measured vs simulated soliton-step histogram. Needs measured data."""
    raise FileNotFoundError(
        "no measured soliton-step dataset is committed to this repository")


# ---------------------------------------------------------------------------
# fig8 -- quiet point
# ---------------------------------------------------------------------------
def build_fig8_quietpoint(ctx: BuildContext):
    """S_rep at a fixed offset vs detuning; the minimum is the quiet point."""
    apply_pub_style()
    report = _load_json(RESULTS / "noise_comparison_report.json")
    block = report.get("fig7_quiet_point")
    if not block:
        raise KeyError("noise_comparison_report.json has no fig7_quiet_point")
    notes: dict = {}

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    f_offset = block.get("f_offset_hz")
    for i, (label, series) in enumerate(
            (k, v) for k, v in block.items() if isinstance(v, dict)):
        dw = np.asarray(series.get("dw_over_kappa", []), dtype=np.float64)
        srep = np.asarray(series.get("S_rep_at_offset", []), dtype=np.float64)
        if dw.size == 0 or srep.size != dw.size:
            continue
        ax.semilogy(dw, srep, "-o", ms=4, color=f"C{i}", label=label)
        qp = series.get("quiet_point_dw_over_kappa")
        qs = series.get("quiet_point_S_rep")
        if qp is not None and qs is not None:
            ax.plot([qp], [qs], "*", ms=15, color=f"C{i}", mec="k", mew=0.6)
            depth = series.get("quiet_point_depth_ratio")
            ax.annotate(
                f"quiet point\n{qp:.2f}$\\kappa$"
                + (f"\ndepth {depth:.1f}x" if depth else ""),
                xy=(qp, qs), xytext=(8, 14), textcoords="offset points",
                fontsize=7, color=f"C{i}",
                arrowprops=dict(arrowstyle="->", color=f"C{i}", lw=0.8))
            notes[f"quiet_point[{label}]"] = {
                "dw_over_kappa": qp, "S_rep": qs, "depth_ratio": depth}

    ax.set_xlabel(r"detuning $\delta\omega/\kappa$")
    ax.set_ylabel(rf"$S_{{rep}}$ at {f_offset:.3g} Hz  [Hz$^2$/Hz]"
                  if f_offset else r"$S_{rep}$ at fixed offset  [Hz$^2$/Hz]")
    ax.set_title("Figure 8 — Predicted quiet point", fontweight="bold")
    ax.legend(fontsize=7)
    notes["f_offset_hz"] = f_offset
    return fig, notes


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
FIGURES: dict[str, FigureSpec] = {
    "fig1_verification": FigureSpec(
        fig_id="fig1_verification",
        builder=build_fig1_verification,
        inputs=tuple(GOLDEN_DIR / f"{n}.npz" for n in
                     ("s256_short", "s512_prod", "s512_sub4", "s1024_near")),
        recompute=True,
        recompute_budget_s=45.0,
        heavy=True,
        description=("(a) all-off ULP residual vs committed goldens "
                     "(b) analytic-CW residual vs detuning "
                     "(c) observed order of accuracy, 3 curves "
                     "(d) weak order, noise on"),
    ),
    "fig2_numerics": FigureSpec(
        fig_id="fig2_numerics",
        builder=build_fig2_numerics,
        inputs=(RESULTS / "validation" / "trn_burnin.npz",),
        recompute=True,
        recompute_budget_s=20.0,
        description=("(a) TRN AR(1) burn-in before/after "
                     "(b) vacuum floor vs n_tau "
                     "(c) Euler vs exponential thermal at dt/tau = 2.1"),
    ),
    "fig3_pylle": FigureSpec(
        fig_id="fig3_pylle",
        builder=build_fig3_pylle,
        inputs=(VALIDATION_RESULTS / "pylle_crosscheck_v2_fields.npz",),
        optional_inputs=(VALIDATION_RESULTS / "pylle_crosscheck_v2.json",),
        description="spectrum + waveform overlay with residual sub-panels",
    ),
    "fig4_budget": FigureSpec(
        fig_id="fig4_budget",
        builder=build_fig4_budget,
        inputs=(RESULTS / "budget" / "budget.json",),
        description="grouped bars, 5 observables x channel sets, log axis, error bars",
    ),
    "fig5_srep": FigureSpec(
        fig_id="fig5_srep",
        builder=build_fig5_srep,
        inputs=(RESULTS / "budget" / "budget.json",),
        optional_inputs=(RESULTS / "validation" / "campaign_report.json",),
        description="S_rep overlay per channel with the TRN(K-G)+FSR limit",
    ),
    "fig6_runtime": FigureSpec(
        fig_id="fig6_runtime",
        builder=build_fig6_runtime,
        inputs=(BENCH_DIR / "runtime.json",),
        description="ns/round-trip/mode vs n_tau, CPU + GPU, n log n reference",
    ),
    "fig7_experiment": FigureSpec(
        fig_id="fig7_experiment",
        builder=build_fig7_experiment,
        inputs=(REPO_ROOT / "data" / "measured" / "soliton_steps.npz",),
        description="measured vs simulated soliton-step histogram (optional)",
    ),
    "fig8_quietpoint": FigureSpec(
        fig_id="fig8_quietpoint",
        builder=build_fig8_quietpoint,
        inputs=(RESULTS / "noise_comparison_report.json",),
        description="predicted quiet point from the quiet-point sweep (stretch)",
    ),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_figure(spec: FigureSpec, out_dir: Path, *,
                 recompute_heavy: bool = False,
                 check_only: bool = False) -> FigureResult:
    """Check inputs, build, save PDF + PNG, write the sidecar. Never raises."""
    t0 = time.time()
    all_inputs = list(spec.inputs) + list(spec.optional_inputs)
    records = [_artifact_record(p) for p in all_inputs]
    missing_required = [p for p in spec.inputs if not Path(p).exists()]

    if missing_required:
        rel = ", ".join(_rel(p) for p in missing_required)
        return FigureResult(
            fig_id=spec.fig_id, status="skipped", inputs=records,
            message=f"required input(s) absent: {rel}",
            elapsed_s=time.time() - t0)

    if check_only:
        missing_opt = [_rel(p) for p in spec.optional_inputs
                       if not Path(p).exists()]
        return FigureResult(
            fig_id=spec.fig_id, status="ok",
            outputs=[str(p) for p in spec.outputs(out_dir)], inputs=records,
            message=("all inputs present" if not missing_opt
                     else f"required OK; optional absent: {', '.join(missing_opt)}"),
            elapsed_s=time.time() - t0)

    ctx = BuildContext(spec=spec, out_dir=out_dir,
                       recompute_heavy=recompute_heavy)
    fig = None
    try:
        fig, notes = spec.builder(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path, png_path = spec.outputs(out_dir)
        fig.savefig(pdf_path)                       # vector
        fig.savefig(png_path, dpi=PNG_DPI)          # 300 dpi raster
        status = "partial" if ctx.missing_panels else "built"
        _write_sidecar(spec, out_dir, records, notes, status,
                       ctx.missing_panels)
        return FigureResult(
            fig_id=spec.fig_id, status=status,
            outputs=[str(pdf_path), str(png_path)], inputs=records,
            message=("all panels drawn" if not ctx.missing_panels
                     else f"{len(ctx.missing_panels)} panel(s) unavailable"),
            elapsed_s=time.time() - t0, missing_panels=ctx.missing_panels)
    except FileNotFoundError as exc:
        return FigureResult(
            fig_id=spec.fig_id, status="skipped", inputs=records,
            message=str(exc)[:200], elapsed_s=time.time() - t0)
    except Exception as exc:  # noqa: BLE001 - one figure must not kill the run
        return FigureResult(
            fig_id=spec.fig_id, status="failed", inputs=records,
            message=f"{type(exc).__name__}: {str(exc)[:180]}",
            elapsed_s=time.time() - t0)
    finally:
        if fig is not None:
            plt.close(fig)


def _short(h: str | None) -> str:
    return h[:12] if h else "—"


def _print_summary(results: list[FigureResult], out_dir: Path) -> None:
    icons = {"built": "OK", "partial": "PARTIAL", "skipped": "SKIP",
             "failed": "FAIL", "ok": "OK"}
    print("\n" + "=" * 100)
    print("PAPER FIGURE SUMMARY")
    print("=" * 100)
    print(f"  {'figure id':<20} {'status':<9} output")
    print("  " + "-" * 96)
    for r in results:
        out = "—"
        if r.outputs:
            out = _rel(r.outputs[0]).replace(".pdf", ".{pdf,png}")
        present = [i for i in r.inputs if i.get("present")]
        print(f"  {r.fig_id:<20} {icons.get(r.status, r.status):<9} {out}")
        if r.message:
            print(f"  {'':<20} └─ {r.message}")
        for panel, why in r.missing_panels.items():
            print(f"  {'':<20}    [{panel}] {why}")
        for i, rec in enumerate(present):
            tag = f"inputs ({len(present)}/{len(r.inputs)}):" if i == 0 else ""
            print(f"  {'':<20} {tag:<18} {_short(rec['sha256'])}  {rec['path']}")
    built = sum(1 for r in results if r.status == "built")
    partial = sum(1 for r in results if r.status == "partial")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    print("  " + "-" * 96)
    print(f"  {built} built, {partial} partial, {skipped} skipped, {failed} failed"
          f"   ->  {out_dir}")
    print("=" * 100 + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate every manuscript figure from committed artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="build every figure")
    ap.add_argument("--figure", action="append", default=None, metavar="ID",
                    choices=sorted(FIGURES),
                    help="build only this figure (repeatable)")
    ap.add_argument("--list", action="store_true",
                    help="list the registry and exit")
    ap.add_argument("--check", action="store_true",
                    help="verify inputs exist without building anything")
    ap.add_argument("--recompute-heavy", action="store_true",
                    help="also run the expensive verification studies that "
                         "have no committed artifact (fig1 b/c/d)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'figure id':<20} {'recompute':<10} {'inputs':<7} description")
        print("-" * 100)
        for fig_id, spec in FIGURES.items():
            budget = (f"~{spec.recompute_budget_s:.0f}s"
                      if spec.recompute_budget_s else "no")
            n_present = sum(1 for p in spec.inputs if Path(p).exists())
            print(f"{fig_id:<20} {budget:<10} "
                  f"{n_present}/{len(spec.inputs):<5} {spec.description}")
        return 0

    if not args.all and not args.figure and not args.check:
        ap.error("nothing to do: pass --all, --figure ID, --check or --list")

    selected = list(args.figure) if args.figure else list(FIGURES)
    out_dir = Path(args.out)

    mode = "CHECK" if args.check else "BUILD"
    print("=" * 100)
    print(f"MANUSCRIPT FIGURES -- {mode}")
    print("=" * 100)
    print(f"  registry : {len(FIGURES)} figures, {len(selected)} selected")
    print(f"  output   : {out_dir}")
    if args.recompute_heavy:
        print("  heavy recompute ENABLED (fig1 b/c/d studies will be run)")
    print("=" * 100)

    results = []
    for fig_id in selected:
        spec = FIGURES[fig_id]
        print(f"[{fig_id}] ...", flush=True)
        result = build_figure(spec, out_dir,
                              recompute_heavy=args.recompute_heavy,
                              check_only=args.check)
        results.append(result)
        print(f"[{fig_id}] {result.status.upper()} ({result.elapsed_s:.1f}s)"
              + (f" -- {result.message}" if result.message else ""))

    _print_summary(results, out_dir)
    # A skip is a stated fact about this checkout, not a failure. Only a builder
    # that raised unexpectedly is an error.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
