"""Figure for the LLE discretization-uncertainty study.

Reads ``validation/results/convergence_lle_dw30k.{json,npz}`` and renders four
panels; run after ``python -m validation.convergence_lle``.

  1. |q(h) - q*| vs h, log-log, per observable, with reference slopes 1 and 2.
  2. Each observable against 1/n_substeps, with its Richardson limit as an hline.
  3. Spectra at the coarsest and finest level, their dB residual, the -60 dBc
     line and the grid edges.
  4. N60 against its threshold from -50 to -80 dBc, which displays the metric's
     conditioning directly (a steep curve means the count is set by where a flat
     wing crosses an arbitrary line).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "validation" / "results"
JSON_PATH = RESULTS / "convergence_lle_dw30k.json"
NPZ_PATH = RESULTS / "convergence_lle_dw30k_fields.npz"
OUT_PATH = Path(__file__).resolve().parent / "fig_convergence_lle.png"

# Observables worth plotting on the convergence panels: normalized quantities
# only, so several can share an axis.
PLOT_OBS = ("P_peak_w", "S3_modes", "mu_DW", "dw_power_dbc", "U_mean_w", "comb_frac")


def _spectrum_dbc(field):
    p = np.abs(np.fft.fftshift(np.fft.fft(field)) / field.size) ** 2
    return 10.0 * np.log10(np.maximum(p, 1e-300) / p.max())


def main() -> int:
    data = json.loads(JSON_PATH.read_text())
    npz = np.load(NPZ_PATH)
    mu = npz["mu"]
    conv = data["convergence"]
    levels = [lv for lv in data["levels"] if lv["status"] == "OK"]
    n_subs = [lv["n_substeps"] for lv in levels]

    fig, ax = plt.subplots(4, 1, figsize=(11, 17))
    op = data["operating_point"]
    fig.suptitle(
        "LLE discretization uncertainty at the DW operating point\n"
        f"{op['n_modes']} modes, {op['n_roundtrips']} round trips, "
        f"δω {op['dw_start_over_kappa']:g}κ→{op['dw_end_over_kappa']:g}κ, "
        f"Kerr phase {op['kerr_phase_per_roundtrip_rad']:.3f} rad/RT",
        fontsize=12)

    # -- panel 1: error against the Richardson limit, log-log ----------------
    for key in PLOT_OBS:
        c = conv.get(key, {})
        q = c.get("extrapolated")
        if q is None or not c.get("levels"):
            continue
        hs = np.asarray(c["h"], float)
        errs = np.abs(np.asarray(c["levels"], float) - q) / max(abs(q), 1e-300)
        m = errs > 0
        if m.sum() >= 2:
            ax[0].loglog(hs[m], errs[m], "o-", lw=1.1, ms=4,
                         label=f"{key}  (p={c['observed_order']:.2f})")
    hh = np.array([min(1.0 / max(n_subs), 0.5), 1.0])
    for slope, style in ((1, ":"), (2, "--")):
        ax[0].loglog(hh, 1e-3 * (hh / hh[-1]) ** slope, style, color="grey", lw=0.9,
                     label=f"slope {slope}")
    ax[0].set_xlabel("h = 1 / n_substeps")
    ax[0].set_ylabel("|q(h) − q*| / |q*|")
    ax[0].set_title("Convergence toward the Richardson limit", fontsize=10, loc="left")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7, ncol=2)

    # -- panel 2: each observable vs 1/n, normalized to its own limit --------
    for key in PLOT_OBS:
        c = conv.get(key, {})
        vals = c.get("levels")
        if not vals:
            continue
        q = c.get("extrapolated")
        ref = q if q not in (None, 0) else (vals[-1] or 1.0)
        ax[1].plot(c["h"], np.asarray(vals, float) / ref, "o-", lw=1.1, ms=4, label=key)
    ax[1].axhline(1.0, color="k", lw=0.7, ls="--", label="Richardson limit q*")
    ax[1].set_xlabel("h = 1 / n_substeps")
    ax[1].set_ylabel("observable / q*")
    ax[1].set_title("Each observable normalized to its extrapolated limit",
                    fontsize=10, loc="left")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, ncol=2)

    # -- panel 3: spectra, coarsest vs finest, plus residual -----------------
    k_c, k_f = f"field_nsub{min(n_subs)}", f"field_nsub{max(n_subs)}"
    if k_c in npz and k_f in npz:
        s_c, s_f = _spectrum_dbc(npz[k_c]), _spectrum_dbc(npz[k_f])
        ax[2].plot(mu, s_c, lw=0.7, label=f"n_substeps = {min(n_subs)}")
        ax[2].plot(mu, s_f, lw=0.7, ls="--", label=f"n_substeps = {max(n_subs)}")
        ax[2].plot(mu, s_c - s_f, lw=0.6, color="crimson", alpha=0.8,
                   label="residual (coarse − fine)")
        ax[2].axhline(-60, color="grey", lw=0.6, ls=":", label="−60 dBc")
        for edge in (mu[0], mu[-1]):
            ax[2].axvline(edge, color="k", lw=0.6, alpha=0.4)
        ax[2].text(mu[0], -132, f"  edge {s_f[0]:.1f} dBc", fontsize=8, va="bottom")
        ax[2].text(mu[-1], -132, f"edge {s_f[-1]:.1f} dBc  ", fontsize=8,
                   va="bottom", ha="right")
        ax[2].set_ylim(-140, 10)
        ax[2].set_xlabel("mode index μ")
        ax[2].set_ylabel("power (dBc)")
        ax[2].set_title("Spectrum, coarsest vs finest, with residual",
                        fontsize=10, loc="left")
        ax[2].grid(alpha=0.3)
        ax[2].legend(fontsize=8, loc="upper right")

        # -- panel 4: N60 conditioning ---------------------------------------
        thr = np.linspace(-80, -50, 121)
        for lbl, s in ((f"n = {min(n_subs)}", s_c), (f"n = {max(n_subs)}", s_f)):
            ax[3].plot(thr, [(s >= t).sum() for t in thr], lw=1.2, label=lbl)
        ax[3].axvline(-60, color="grey", lw=0.8, ls=":", label="−60 dBc gate")
        d = data["levels"][0]["observables"].get("dN60_ddB")
        ax[3].set_xlabel("threshold (dBc)")
        ax[3].set_ylabel("line count")
        ax[3].set_title(
            f"N60 conditioning — dN60/ddB = {d} lines per dB at the gate "
            "(diagnostic only, never an acceptance criterion)",
            fontsize=10, loc="left")
        ax[3].grid(alpha=0.3)
        ax[3].legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PATH, dpi=140)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
