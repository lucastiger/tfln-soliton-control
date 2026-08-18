"""Two-panel figure for the order-of-accuracy verification.

Panel (a) — deterministic dt refinement, log-log. The three configurations are
drawn twice: SOLID for the second-order path (``symmetric_drive=True``,
``thermal_coupling="strang"``) and DASHED for the production path (all flags
off), in matching colours. Reference slopes ``dt^1`` and ``dt^2`` are overlaid
as grey guides. The figure's whole point is the gap between the solid and
dashed families: every dashed curve runs parallel to the ``dt^1`` guide and
every solid one to ``dt^2``, which is the finding stated as a picture.

Panel (b) — the weak-order fit. E[|E|^2] over 256 realizations with the
quantum-vacuum channel on and common random numbers across dt, with the fitted
slope drawn through the resolved points and the noise-dominated floor marked.

Usage
-----
    python -m validation.figures.fig_convergence [--quick] [--out PATH]

See :mod:`validation.convergence` for the physics and the derivations.
Reference: Herr, Tikan & Kippenberg, arXiv:2604.05897v1 (7 Apr 2026).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.plot_utils import apply_pub_style                        # noqa: E402
from validation.convergence import (                                   # noqa: E402
    DEFAULT_HALVINGS,
    deterministic_study,
    thermal_testbed_config,
    weak_study,
)
from validation.mms import DEFAULT_N_TAU, manufactured_solution        # noqa: E402

__all__ = ["make_figure", "main"]

DEFAULT_OUT = Path(__file__).resolve().parent / "fig_convergence.png"

#: Colourblind-safe qualitative colours (Wong 2011), one per configuration.
_COLORS = {
    "a_field_only": "#0072B2",
    "b_thermal_euler": "#E69F00",
    "c_thermal_exponential": "#009E73",
}
_SHORT = {
    "a_field_only": "(a) field only",
    "b_thermal_euler": "(b) thermal, euler",
    "c_thermal_exponential": "(c) thermal, exponential",
}


def make_figure(
    production: dict[str, Any],
    fixed: dict[str, Any],
    weak: dict[str, Any],
    out_path: Path,
) -> Path:
    """Render the two-panel order-of-accuracy figure from pre-computed results.

    Parameters
    ----------
    production : dict
        Results for the SHIPPING scheme, from
        :func:`validation.convergence.deterministic_study` and friends.
    fixed : dict
        The same for the fixed scheme (symmetric drive, Strang thermal
        coupling, exponential integrator).
    weak : dict
        The weak-convergence study, from
        :func:`validation.convergence.weak_study`.
    out_path : pathlib.Path
        Destination image.

    Returns
    -------
    pathlib.Path
        ``out_path``, for chaining.

    Raises
    ------
    KeyError
        If a study result is missing a series the figure plots.
    OSError
        If ``out_path`` cannot be written.
    ImportError
        If matplotlib is unavailable.

    Notes
    -----
    Both schemes are plotted on the SAME axes with reference slopes, because
    the finding is the difference between them: the shipping scheme is first
    order and the fixed one second, and two separate plots would leave a reader
    to compare gradients by eye across figures.

    No ``Examples`` section: it takes completed refinement ladders and writes a
    file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_pub_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # ---------------- panel (a): deterministic orders ----------------
    for key, color in _COLORS.items():
        f = fixed[key]
        p = production[key]
        ax_a.loglog(f["dts"], f["errors"], "-o", ms=4.0, color=color, lw=1.4,
                    label=f"{_SHORT[key]}  $p$={f['order']:.2f}")
        ax_a.loglog(p["dts"], p["errors"], "--s", ms=3.4, mfc="none", color=color,
                    lw=1.1, alpha=0.85)

    # Reference slopes, anchored so they sit clear of the data.
    all_dt = np.concatenate([fixed[k]["dts"] for k in _COLORS])
    dt_ref = np.array([all_dt.min(), all_dt.max()])
    prod_anchor = production["a_field_only"]["errors"][0]
    fix_anchor = fixed["a_field_only"]["errors"][0]
    dt0 = production["a_field_only"]["dts"][0]
    ax_a.loglog(dt_ref, prod_anchor * 3.0 * (dt_ref / dt0) ** 1, ":", color="0.45",
                lw=1.3, label=r"reference $\propto dt^1$")
    ax_a.loglog(dt_ref, fix_anchor * 0.25 * (dt_ref / dt0) ** 2, "-.", color="0.45",
                lw=1.3, label=r"reference $\propto dt^2$")

    ax_a.set_xlabel(r"time step  $dt$  [s]   ($dt = t_R/M$)")
    ax_a.set_ylabel("Richardson successive difference (relative)")
    ax_a.set_title("(a)  deterministic $dt$ refinement", loc="left")
    ax_a.legend(loc="lower right", fontsize=7.0, framealpha=0.92)
    ax_a.annotate(
        "solid  = symmetric_drive + strang coupling\n"
        "dashed = production (all flags off)",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.92},
    )

    # ---------------- panel (b): weak order ----------------
    dts = weak["dts"]
    errs = weak["errors"]
    n_fit = int(weak["n_fit"])
    ax_b.loglog(dts[:n_fit], errs[:n_fit], "o", ms=5.0, color="#CC79A7",
                label="resolved (fitted)")
    if n_fit < len(errs):
        ax_b.loglog(dts[n_fit:], errs[n_fit:], "x", ms=6.0, color="0.55",
                    label="noise-floor limited (excluded)")
    fit = np.polyfit(np.log(dts[:n_fit]), np.log(errs[:n_fit]), 1)
    ax_b.loglog(dts, np.exp(np.polyval(fit, np.log(dts))), "-", color="#CC79A7",
                lw=1.3, label=f"fit: $p$ = {weak['order']:.3f}")
    ax_b.loglog(dts, errs[0] * (dts / dts[0]) ** 1, ":", color="0.45", lw=1.3,
                label=r"reference $\propto dt^1$")

    ax_b.set_xlabel(r"time step  $dt$  [s]")
    ax_b.set_ylabel(r"$|\,\mathbb{E}[|E|^2](dt) - \mathbb{E}[|E|^2](dt/2)\,|$ (relative)")
    ax_b.set_title("(b)  weak order, quantum vacuum on", loc="left")
    ax_b.legend(loc="lower right", fontsize=7.0, framealpha=0.92)
    ax_b.annotate(
        f"{weak['n_realizations']} realizations, common random numbers\n"
        f"noise-borne fraction of $\\mathbb{{E}}[|E|^2]$ = {weak['noise_fraction']:.1e}\n"
        f"(deterministic-dominated: this confirms the noise\n"
        f"channel does not DEGRADE the order)",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.92},
    )

    # Decade ticks only: matplotlib's default log minor labels collide badly
    # over the ~2 decades of dt spanned here.
    for ax in (ax_a, ax_b):
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        )

    fig.suptitle(
        "Order of accuracy — the shipping scheme is first order; both barriers "
        "are opt-in fixes", fontsize=9,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the convergence studies and render the figure from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments; ``None`` (default) reads ``sys.argv[1:]``.
        Supports ``--out``, ``--halvings``, ``--n-tau``, ``--t-slow``,
        ``--weak-realizations`` and ``--quick``.

    Returns
    -------
    int
        The process exit status: 0 on success.

    Raises
    ------
    SystemExit
        From ``argparse`` on a malformed command line or ``--help``.
    ImportError
        If sympy (for the manufactured source) or matplotlib is unavailable.

    Notes
    -----
    Runs the studies itself rather than reading a stored artifact, so the
    figure cannot fall out of date with the numbers. ``--quick`` shortens the
    ladders for a fast look.

    No ``Examples`` section: every path runs the full refinement ladders.
    """
    ap = argparse.ArgumentParser(
        prog="python -m validation.figures.fig_convergence",
        description="Render the order-of-accuracy figure.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--halvings", type=int, default=DEFAULT_HALVINGS)
    ap.add_argument("--n-tau", type=int, default=DEFAULT_N_TAU)
    ap.add_argument("--t-slow", type=int, default=200)
    ap.add_argument("--weak-realizations", type=int, default=256)
    ap.add_argument("--quick", action="store_true",
                    help="3 halvings, 100 round trips, 64 realizations")
    args = ap.parse_args(argv)

    halvings = 3 if args.quick else args.halvings
    t_slow = 100 if args.quick else args.t_slow
    realizations = 64 if args.quick else args.weak_realizations

    sol = manufactured_solution()
    testbed = thermal_testbed_config(
        sol.params, pin=sol.pin, delta_omega=sol.delta_omega
    )
    common = dict(halvings=halvings, n_tau=args.n_tau, t_slow=t_slow,
                  testbed_config=testbed)
    production = deterministic_study(
        sol, symmetric_drive=False, thermal_coupling="lagged", **common
    )
    fixed = deterministic_study(
        sol, symmetric_drive=True, thermal_coupling="strang", **common
    )
    weak = weak_study(sol, n_realizations=realizations, symmetric_drive=False)

    out = make_figure(production, fixed, weak, args.out)
    print(f"wrote {out}")
    for key in _COLORS:
        print(f"  {key:>24}: production {production[key]['order']:.4f}   "
              f"fixed {fixed[key]['order']:.4f}")
    print(f"  {'weak (production)':>24}: {weak['order']:.4f} "
          f"(fitted over {weak['n_fit']}/{len(weak['errors'])} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
