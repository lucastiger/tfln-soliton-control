"""Figure for the existence-edge discretization measurement.

Three panels, from ``validation/results/existence_convergence_ours.json``:

  (a) edge midpoint vs 1/n_substeps, with the bisection bracket half-width as
      the error bar, and pyLLE's v2 edge drawn as a horizontal band. The pyLLE
      band is a READ-ONLY v2 value at the dt-ladder finest level -- it was not
      re-measured here, and that asymmetry is the main caveat on the recomputed
      G7 band.
  (b) the bisection trace at each level: every evaluated detuning, alive or
      dead, with the classifier contrast on a log axis and the contrast = 5
      decision threshold marked. This is what makes a threshold flip visible.
  (c) peak-power spread over the final 10 round trips at every bisection
      midpoint. This is the panel that shows whether the edge states are
      marginal; pyLLE's 22.6% at its own v2 lower-edge midpoint is marked for
      scale.

Usage::

    python -m validation.figures.fig_existence_convergence
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "validation" / "results"
RESULT_JSON = RESULTS / "existence_convergence_ours.json"
V2_JSON = RESULTS / "pylle_crosscheck_v2.json"
OUT = Path(__file__).resolve().parent / "fig_existence_convergence.png"

# pyLLE's measured peak-power spread at its v2 lower-edge midpoint, for scale.
PYLLE_V2_SPREAD = 0.2258


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = json.loads(RESULT_JSON.read_text())
    v2ex = json.loads(V2_JSON.read_text())["existence"]
    levels = [lv for lv in r["levels"] if lv.get("status") == "OK"]
    ns = [lv["n_substeps"] for lv in levels]

    fig, ax = plt.subplots(3, 1, figsize=(11, 14))

    # ---- (a) edge midpoint vs 1/n ------------------------------------------
    # One axis per edge. The two edges are 32 kappa apart while the drift being
    # measured is 0.19 kappa, so a shared y-axis -- log, symlog or linear --
    # renders the entire result as a flat line.
    ax[0].axis("off")
    pos = ax[0].get_position()
    fig.text(pos.x0, pos.y1 - 0.002,
             "(a) Ours-side existence edge under substep refinement. Error bars "
             "are bisection bracket half-widths.\nThe pyLLE bands are READ-ONLY "
             "v2 values (dt ladder), not re-measured here.",
             fontsize=10, ha="left", va="top")
    sub = ax[0].get_subplotspec().subgridspec(
        2, 2, wspace=0.26, height_ratios=[0.42, 1.0], hspace=0.0)
    for j, (edge, colour) in enumerate((("lower", "tab:blue"),
                                        ("upper", "tab:green"))):
        a = fig.add_subplot(sub[1, j])
        x = [1.0 / n for n in ns]
        y = [lv[edge]["midpoint"] for lv in levels]
        e = [lv[edge]["halfwidth"] for lv in levels]
        a.errorbar(x, y, yerr=e, marker="o", ms=7, lw=1.4, capsize=5,
                   color=colour, label="ours", zorder=3)
        mid_p = v2ex[f"{edge}_midpoint_pylle"]
        hw_p = v2ex[f"{edge}_halfwidth_pylle"]
        a.axhspan(mid_p - hw_p, mid_p + hw_p, color="tab:orange", alpha=0.22,
                  label="pyLLE v2 bracket")
        a.axhline(mid_p, color="tab:orange", ls="--", lw=1.2)
        for xi, yi, n in zip(x, y, ns):
            a.annotate(f"n={n}", xy=(xi, yi), xytext=(0, -14),
                       textcoords="offset points", ha="center", fontsize=7)
        a.set_xscale("log")
        a.invert_xaxis()
        a.set_xlabel("1 / n_substeps  (fine -> coarse)")
        if j == 0:
            a.set_ylabel("existence edge, delta_omega / kappa")
        drift = abs(y[-1] - y[0]) if len(y) > 1 else 0.0
        a.set_title(f"{edge} edge   drift |e(1)-e(finest)| = {drift:.4f} kappa",
                    fontsize=9)
        a.legend(fontsize=8, loc="best")
        a.grid(alpha=0.3)

    # ---- (b) bisection traces ----------------------------------------------
    markers = {1: "o", 2: "s", 4: "^", 8: "D", 16: "v"}
    for lv in levels:
        n = lv["n_substeps"]
        for edge in ("lower", "upper"):
            for t in lv[edge]["trace"]:
                ax[1].scatter(t["dw_over_kappa"], max(t["contrast"], 1e-2),
                              marker=markers.get(n, "o"), s=52,
                              facecolor=("tab:green" if t["alive"] else "none"),
                              edgecolor=("tab:green" if t["alive"] else "tab:red"),
                              linewidths=1.4, zorder=3)
    ax[1].axhline(5.0, color="k", ls="--", lw=1.2,
                  label="classifier threshold, contrast = 5")
    # The decisive observation of this panel is what is NOT here: no evaluation
    # lands near the threshold. Shade the empty band so that reads as a result
    # rather than as absence of data.
    alive = [t["contrast"] for lv in levels for e in ("lower", "upper")
             for t in lv[e]["trace"] if t["alive"]]
    dead = [t["contrast"] for lv in levels for e in ("lower", "upper")
            for t in lv[e]["trace"] if not t["alive"]]
    if alive and dead:
        ax[1].axhspan(max(dead), min(alive), color="grey", alpha=0.16,
                      label=f"empty band: no evaluation between "
                            f"{max(dead):.2f} and {min(alive):.2f} "
                            f"({min(alive) / max(dead):.1f}x)")
    for edge, colour in (("lower", "tab:blue"), ("upper", "tab:green")):
        ax[1].axvline(v2ex[f"{edge}_midpoint_pylle"], color=colour, ls=":", lw=1.0,
                      label=f"pyLLE v2 {edge} edge")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("delta_omega / kappa")
    ax[1].set_ylabel("classifier contrast (peak / median power)")
    handles = [
        plt.Line2D([], [], marker=markers.get(n, "o"), ls="", color="k",
                   markerfacecolor="none", label=f"n_substeps = {n}")
        for n in ns]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color="tab:green", label="alive"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                   color="tab:red", label="dead"),
        plt.Line2D([], [], color="k", ls="--", label="contrast = 5"),
    ]
    ax[1].legend(handles=handles, fontsize=8, ncol=2, loc="center right")
    ax[1].set_title(
        "(b) Every bisection evaluation. The classifier is a THRESHOLD test, so a "
        "point near contrast = 5 would be\none where the verdict came from the "
        "classifier rather than the physics. NONE of them is: the shaded band is "
        "empty.",
        fontsize=10, loc="left")
    ax[1].grid(alpha=0.3)

    # ---- (c) peak-power spread ---------------------------------------------
    for lv in levels:
        n = lv["n_substeps"]
        xs, ys = [], []
        for edge in ("lower", "upper"):
            for t in lv[edge]["trace"]:
                if t.get("peak_power_spread_last10") is not None:
                    xs.append(t["dw_over_kappa"])
                    ys.append(100.0 * t["peak_power_spread_last10"])
        order = np.argsort(xs)
        ax[2].plot(np.asarray(xs)[order], np.asarray(ys)[order],
                   marker=markers.get(n, "o"), ms=6, lw=0.9,
                   label=f"ours, n_substeps = {n}")
    ax[2].axhline(100.0 * PYLLE_V2_SPREAD, color="tab:orange", ls="--", lw=1.3,
                  label=f"pyLLE v2 at its lower-edge midpoint "
                        f"({100 * PYLLE_V2_SPREAD:.1f}%)")
    ax[2].axhline(1.0, color="grey", ls=":", lw=1.0,
                  label="1% (v2 NON_STATIONARY threshold)")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("delta_omega / kappa")
    ax[2].set_ylabel("peak-power spread over the final 10 round trips (%)")
    ax[2].set_title(
        "(c) How much the state is still moving at each bisection midpoint. A "
        "state that is still\nevolving by tens of percent is marginal, and a "
        "threshold classifier applied to it is fragile.",
        fontsize=10, loc="left")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    h = r["hypothesis_outcome"]
    lo = r["edge_convergence"]["lower"]
    fig.suptitle(
        f"Existence-edge discretization uncertainty (our solver only)\n"
        f"lower-edge drift n=1..4: {lo['drift_1_to_4']:.4f} kappa   ->   "
        f"pre-committed outcome {h['outcome']}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0.005, 1, 0.965))
    fig.savefig(OUT, dpi=130,
                metadata={"Software": "validation.figures.fig_existence_convergence",
                          "Creation Time": None})
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
