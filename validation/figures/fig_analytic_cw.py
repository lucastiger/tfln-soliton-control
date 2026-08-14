"""Two-panel figure for the analytic CW verification.

Panel (a) — the bistable CW S-curve. The analytic steady state is drawn as a
continuous PARAMETRIC curve rather than root-by-root: solving the steady state
for the detuning instead of the power,

    delta_omega(P) = gamma*P -/+ sqrt(kappa_c*P_in/P - (kappa/2)^2),
    P in (0, P_max],  P_max = kappa_c*P_in/(kappa/2)^2

gives two single-valued arcs that join at ``P_max``, so concatenating the minus
arc (P increasing) with the plus arc (P decreasing) traces the whole S in one
stroke with no root-sorting and no missing segments. The unstable middle branch
is overlaid dashed from the cubic's middle root, and the bistable detuning
window is shaded. Solver points sit on top: circles from an upward detuning
scan (high branch) and squares from a downward scan (low branch), which is what
makes the bistability visible as two *reachable* states rather than as an
abstract multivalued curve.

Panel (b) — the residual spectrum, log axis, with the 1e-12 guide line. Two
families are plotted per pump power:

  * ``rel_continuum`` (lines, upper cluster ~3e-3) — the solver against the
    continuum LLE cubic. This is the mean-field truncation of the operator
    splitting: first order in dt, magnitude ~kappa*t_r/2. It is NOT a
    convergence artifact and does not shrink with settle time.
  * ``rel_discrete`` (markers, lower cluster ~1e-15) — the solver against the
    exact fixed point of its own discrete map, which is the machine-precision
    claim and what the CLI gates on.

Putting both on one axis is the point of the panel: the ~13 decades between
them is the size of the mean-field approximation, measured rather than assumed.

Usage
-----
    python -m validation.figures.fig_analytic_cw [--quick] [--out PATH]

See :mod:`validation.analytic_cw` for the physics and the derivations.
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

from analysis.plot_utils import apply_pub_style                      # noqa: E402
from validation.analytic_cw import (                                 # noqa: E402
    CavityParams,
    DEFAULT_DETUNING_GRID,
    DEFAULT_PIN_MULTIPLIERS,
    DEFAULT_N_TAU,
    DEFAULT_T_SLOW,
    analytic_cw_roots,
    load_cavity_params,
    solver_cw_steady_state,
    verify,
)

__all__ = ["analytic_s_curve", "bistable_window", "make_figure", "main"]

DEFAULT_OUT = Path(__file__).resolve().parent / "fig_analytic_cw.png"

#: Pump power for panel (a), in units of P_th. Only 5 and 10 P_th are bistable
#: anywhere on the [-5, 15]*kappa grid; 10 gives the widest window
#: (delta_omega/kappa in [2, 5]) and therefore the clearest S.
SCURVE_PIN_MULT = 10.0

#: Colorblind-safe qualitative colors (Wong 2011), one per pump power.
_PIN_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")


def analytic_s_curve(
    pin: float, params: CavityParams, n_points: int = 4001
) -> tuple[np.ndarray, np.ndarray]:
    """The full CW steady-state curve, traced parametrically in P.

    Returns ``(delta_omega/kappa, P)`` along one continuous stroke covering
    both arcs, so the multivalued (bistable) region is drawn correctly without
    any branch bookkeeping.
    """
    p_max = params.kappa_c * float(pin) / (params.kappa / 2.0) ** 2
    # Log spacing: the low branch spans several decades in P while the fold
    # region needs resolution near P_max, and the sqrt below has infinite
    # slope there. Squeeze the last decade toward P_max with a cosine taper.
    frac = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_points)))    # 0 -> 1, dense at ends
    p_lo = p_max * 1e-9
    p = p_lo * (p_max / p_lo) ** frac
    p = np.clip(p, None, p_max)

    inner = np.maximum(params.kappa_c * float(pin) / p - (params.kappa / 2.0) ** 2, 0.0)
    offset = np.sqrt(inner)
    dw_minus = params.gamma * p - offset
    dw_plus = params.gamma * p + offset

    # minus arc with P increasing, then plus arc back down: one continuous S.
    dw = np.concatenate([dw_minus, dw_plus[::-1]])
    pp = np.concatenate([p, p[::-1]])
    return dw / params.kappa, pp


def bistable_window(
    pin: float, params: CavityParams, grid: Sequence[float] | np.ndarray
) -> tuple[float, float] | None:
    """Detuning span (in units of kappa) where the cubic has three real roots.

    Computed on a refined grid spanning ``grid`` so the shaded region is not
    quantized to the coarse verification grid.
    """
    fine = np.linspace(float(np.min(grid)), float(np.max(grid)), 4001)
    counts = np.array(
        [
            analytic_cw_roots(pin, d * params.kappa, params.kappa, params.kappa_c,
                              params.gamma).size
            for d in fine
        ]
    )
    tri = fine[counts == 3]
    if tri.size == 0:
        return None
    return float(tri.min()), float(tri.max())


def make_figure(
    result: dict[str, Any],
    out_path: Path,
    *,
    scurve_pin_mult: float = SCURVE_PIN_MULT,
    t_slow: int = DEFAULT_T_SLOW,
    n_tau: int = DEFAULT_N_TAU,
) -> Path:
    """Render the two panels from a :func:`validation.analytic_cw.verify` result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_pub_style()

    params: CavityParams = result["params"]
    grid = result["delta_over_kappa"]
    mults = result["pin_over_pth"]

    # Which swept pump to draw the S-curve for; fall back to the largest one
    # present if the requested multiplier was not part of the sweep.
    if scurve_pin_mult in set(float(m) for m in mults):
        i_s = int(np.argmin(np.abs(mults - scurve_pin_mult)))
    else:
        i_s = int(np.argmax(mults))
    mult_s = float(mults[i_s])
    pin_s = mult_s * params.p_th

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 4.1))

    # ---------------- panel (a): the S-curve ----------------
    window = bistable_window(pin_s, params, grid)
    if window is not None:
        ax_a.axvspan(
            window[0], window[1], color="0.85", zorder=0,
            label=f"bistable: $\\delta\\omega/\\kappa \\in$ "
                  f"[{window[0]:.2f}, {window[1]:.2f}]",
        )

    dw_curve, p_curve = analytic_s_curve(pin_s, params)
    inside = (dw_curve >= grid.min()) & (dw_curve <= grid.max())
    ax_a.plot(
        dw_curve[inside], p_curve[inside], "-", color="#333333", lw=1.4, zorder=2,
        label="analytic steady state (continuum cubic)",
    )

    # The unstable middle root, dashed, from the cubic itself.
    fine = np.linspace(grid.min(), grid.max(), 2001)
    mid_dw, mid_p = [], []
    for d in fine:
        roots = analytic_cw_roots(
            pin_s, d * params.kappa, params.kappa, params.kappa_c, params.gamma
        )
        if roots.size == 3:
            mid_dw.append(d)
            mid_p.append(roots[1])
    if mid_dw:
        ax_a.plot(
            mid_dw, mid_p, "--", color="#B00020", lw=1.4, zorder=3,
            label="unstable middle branch",
        )

    # Solver points on both reachable branches.
    ax_a.plot(
        grid, result["p_solver"][i_s], "o", ms=4.0, mfc="none", mew=1.0,
        color="#0072B2", zorder=4, label="solver (upward $\\delta\\omega$ scan)",
    )
    p_down = solver_cw_steady_state(
        pin_s, grid * params.kappa, params,
        t_slow=t_slow, n_tau=n_tau, scan_direction="down",
    )
    ax_a.plot(
        grid, p_down, "s", ms=3.6, mfc="none", mew=1.0,
        color="#009E73", zorder=4, label="solver (downward $\\delta\\omega$ scan)",
    )

    ax_a.set_yscale("log")
    ax_a.set_xlim(grid.min(), grid.max())
    # Headroom above the fold: the top of the S sits at P_max = kappa_c*P_in/
    # (kappa/2)^2, and the legend must not be allowed to cover it (the upper
    # branch is the whole point of the panel).
    p_top = params.kappa_c * pin_s / (params.kappa / 2.0) ** 2
    p_bot = min(float(np.min(result["p_solver"][i_s])), float(np.min(p_down)))
    ax_a.set_ylim(p_bot / 3.0, p_top * 30.0)
    ax_a.set_xlabel(r"detuning  $\delta\omega/\kappa$   ($\delta\omega=\omega_{res}-\omega_{p}$)")
    ax_a.set_ylabel(r"intracavity power  $|E|^2$  [J]")
    ax_a.set_title(
        f"(a)  CW S-curve,  $P_{{in}} = {mult_s:g}\\,P_{{th}}$ "
        f"({pin_s * 1e3:.1f} mW)",
        loc="left",
    )
    # Upper LEFT: the blue-detuned corner is empty at every pump power, whereas
    # upper right is exactly where the high branch runs.
    ax_a.legend(loc="upper left", framealpha=0.9, fontsize=7.2)

    # ---------------- panel (b): residual spectrum ----------------
    # Points where the solver reproduces the discrete-map fixed point to 0 ULP
    # have residual exactly 0 and cannot be drawn on a log axis. Drop them
    # rather than clipping to a floor — a fake row of dots on the bottom axis
    # would read as "worst case", the opposite of the truth — and report the
    # count instead.
    n_exact = int(np.sum(result["rel_discrete"] == 0.0))
    for i, mult in enumerate(mults):
        color = _PIN_COLORS[i % len(_PIN_COLORS)]
        ax_b.plot(
            grid, result["rel_continuum"][i], "-", lw=1.2, color=color,
            label=f"{mult:g}$\\,P_{{th}}$",
        )
        nonzero = result["rel_discrete"][i] > 0.0
        ax_b.plot(
            grid[nonzero], result["rel_discrete"][i][nonzero], ".", ms=3.0,
            color=color,
        )

    ax_b.axhline(
        result["rtol"], color="#B00020", ls="--", lw=1.2, zorder=5,
        label=f"gate: {result['rtol']:.0e}",
    )
    ax_b.set_yscale("log")
    ax_b.set_xlim(grid.min(), grid.max())
    ax_b.set_ylim(1e-17, 1e-1)
    ax_b.set_xlabel(r"detuning  $\delta\omega/\kappa$")
    ax_b.set_ylabel(r"$|$relative residual in $|E|^2|$")
    ax_b.set_title("(b)  solver vs the two exact references", loc="left")
    ax_b.legend(loc="center right", ncol=1, framealpha=0.9, title="$P_{in}$")

    ax_b.annotate(
        f"lines: vs CONTINUUM cubic\n"
        f"max {result['max_rel_continuum']:.2e}  "
        f"($\\approx \\kappa t_R/2 = {params.kappa * params.t_r / 2:.2e}$,\n"
        f"first order in $dt$ — mean-field truncation)\n"
        f"all {len(mults)} curves coincide: the error is $P_{{in}}$-independent",
        xy=(0.03, 0.80), xycoords="axes fraction", va="top", ha="left", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.92},
    )
    ax_b.annotate(
        f"dots: vs EXACT DISCRETE MAP\n"
        f"max {result['max_rel_discrete']:.2e}, median "
        f"{result['median_rel_discrete']:.1e}\n"
        f"({n_exact}/{result['n_points']} points agree to 0 ULP, not plottable)",
        xy=(0.03, 0.60), xycoords="axes fraction", va="top", ha="left", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.92},
    )

    fig.suptitle(
        f"Analytic CW verification — {result['n_points']} points, "
        f"$\\kappa t_R$ = {params.kappa * params.t_r:.3e}, "
        f"all noise off, $\\beta=[0,0]$, $dn/dT=0$",
        fontsize=9,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m validation.figures.fig_analytic_cw",
        description="Render the analytic-CW verification figure.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--t-slow", type=int, default=DEFAULT_T_SLOW)
    ap.add_argument("--n-tau", type=int, default=DEFAULT_N_TAU)
    ap.add_argument("--quick", action="store_true",
                    help="reduced grid (41 detunings, 2 pumps, t_slow=20000)")
    args = ap.parse_args(argv)

    params = load_cavity_params(args.config)
    if args.quick:
        grid = np.linspace(-5.0, 15.0, 41)
        mults: tuple[float, ...] = (1.0, 10.0)
        t_slow = 20_000
    else:
        grid = DEFAULT_DETUNING_GRID
        mults = DEFAULT_PIN_MULTIPLIERS
        t_slow = args.t_slow

    result = verify(
        params=params, detuning_grid=grid, pin_multipliers=mults,
        t_slow=t_slow, n_tau=args.n_tau,
    )
    out = make_figure(result, args.out, t_slow=t_slow, n_tau=args.n_tau)
    print(f"wrote {out}")
    print(
        f"  max rel residual vs discrete map: {result['max_rel_discrete']:.3e} "
        f"(gate {result['rtol']:.0e}, {'PASS' if result['passed'] else 'FAIL'})"
    )
    print(f"  max rel residual vs continuum cubic: {result['max_rel_continuum']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
