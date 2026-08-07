#!/usr/bin/env python
"""Quantify the AR(1) start-up (burn-in) bias of the thermorefractive channel.

The legacy ``simulator.noise_models._ar1_samples`` initialises its ``lax.scan``
carry at exactly ``0.0``, so the generated record is NOT stationary: its
variance ramps as

    Var(x_n) = sigma**2 * (1 - alpha**(2n)),   alpha = exp(-t_r/tau_corr),

reaching the target only after ``n_burn = tau_corr/(2*t_r)`` round trips. For
the committed SiN config (``t_r = 40.65 ps``, ``tau_th = 5 us``) that is
``n_burn ~ 6.1e4`` round trips -- longer than most trajectories in this
repository -- so EVERY short run has suppressed TRN amplitude, and it is
suppressed most at the START of the record.

Pooled over a record of length ``N`` the expected mean square is

    E[mean(x**2)] / sigma**2
        = 1 - alpha**2 * (1 - alpha**(2N)) / (N * (1 - alpha**2))          (*)

which tends to ``N*t_r/tau_corr`` for ``N*t_r << tau_corr``. This script
measures that ratio for both states of the ``trn_ar1_stationary_init`` flag and
compares it against (*).

Estimator note
--------------
The reported statistic is the RAW second moment ``mean(x**2)``, not
``numpy.var``. The two differ enormously here: with ``tau_corr >> N*t_r`` an
entire record is essentially one slowly varying excursion, so subtracting the
sample mean would remove precisely the low-frequency power the burn-in
suppresses and would report ~0 for both flag states. The process is zero-mean
by construction, so the raw second moment is the correct variance estimator.

With ``n_real`` realizations and ``tau_corr >> N*t_r`` the samples inside one
record are nearly perfectly correlated, so the estimator has roughly ``n_real``
degrees of freedom and a relative 1-sigma spread of ``sqrt(2/n_real)``
(~17.7 % at the default 64). The plot therefore draws that band; the *legacy*
suppression is one to two ORDERS of magnitude, far outside it.

Outputs
-------
* ``analysis/results/validation/trn_burnin.json`` -- every measured number plus
  the provenance stamp.
* ``analysis/results/validation/trn_burnin.png`` -- realized/target variance vs
  record length, one curve per flag state.
* ``analysis/results/validation/trn_burnin.npz`` (+ ``.provenance.json``) --
  the raw arrays, written through :func:`simulator.provenance.save_artifact`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from analysis.plot_utils import apply_pub_style  # noqa: E402
from simulator.noise_models import _ar1_samples, _load_config  # noqa: E402
from simulator.provenance import provenance_stamp, save_artifact  # noqa: E402

RESULT_DIR = Path(__file__).resolve().parent / "results" / "validation"
ARTIFACT_STEM = RESULT_DIR / "trn_burnin"
JSON_PATH = RESULT_DIR / "trn_burnin.json"
PNG_PATH = RESULT_DIR / "trn_burnin.png"

#: Record lengths in round trips. Spans the burn-in scale of the committed
#: config (n_burn = tau_th/(2*t_r) ~ 6.1e4 RT) from far below it to a few times
#: above, so the legacy curve is seen to rise across the whole range.
DEFAULT_T_SLOW = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000)
DEFAULT_N_REAL = 64
DEFAULT_SEED = 20260807


def analytic_ratio(n: int, alpha: float) -> float:
    """Eq. (*) above: E[mean(x**2)]/sigma**2 for the zero-start AR(1).

    Uses ``expm1``/``log`` for ``alpha**(2N)`` so the ``N ~ 3e5``,
    ``alpha ~ 1 - 8e-6`` corner does not lose precision to cancellation.
    """
    a2 = alpha * alpha
    if a2 >= 1.0:
        return 0.0
    # 1 - alpha**(2N) evaluated stably.
    one_minus_a2n = -math.expm1(2.0 * n * math.log(alpha))
    return 1.0 - a2 * one_minus_a2n / (n * (1.0 - a2))


def measure(
    n: int,
    alpha_tau: float,
    t_r: float,
    n_real: int,
    seed: int,
    stationary_init: bool,
) -> tuple[float, np.ndarray]:
    """(pooled mean-square / target, per-realization mean-squares).

    ``sigma_physical = 1.0``, so the target variance is exactly 1 and the
    returned ratio is dimensionless. The realizations are vmapped over
    independent subkeys of ``seed``; both flag states are driven from the SAME
    subkeys, so the two curves are paired rather than independently noisy.
    """
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), int(n_real))
    draw = jax.vmap(
        lambda k: _ar1_samples(k, int(n), alpha_tau, 1.0, t_r, stationary_init)
    )
    x = np.asarray(draw(keys), dtype=np.float64)          # (n_real, n)
    per_real = np.mean(x**2, axis=1)                      # (n_real,)
    return float(np.mean(per_real)), per_real


def run_study(
    t_slow_values=DEFAULT_T_SLOW,
    n_real: int = DEFAULT_N_REAL,
    seed: int = DEFAULT_SEED,
    config_path=None,
) -> dict:
    """Measure the ratio for both flag states at every record length."""
    physical = _load_config(config_path)
    t_r = 1.0 / float(physical.get("fsr_hz", 2.0e11))
    tau_th = float(physical.get("tau_th_s", 5.0e-6))
    alpha = math.exp(-t_r / tau_th)
    n_burn = tau_th / (2.0 * t_r)

    t_slow_values = [int(n) for n in t_slow_values]
    rows = []
    per_real_legacy, per_real_stationary = [], []
    for n in t_slow_values:
        legacy, pr_l = measure(n, tau_th, t_r, n_real, seed, False)
        stationary, pr_s = measure(n, tau_th, t_r, n_real, seed, True)
        per_real_legacy.append(pr_l)
        per_real_stationary.append(pr_s)
        rows.append(
            {
                "t_slow": n,
                "n_over_tau": n * t_r / tau_th,
                "legacy_ratio": legacy,
                "stationary_ratio": stationary,
                "legacy_ratio_analytic": analytic_ratio(n, alpha),
                "stationary_ratio_analytic": 1.0,
                "legacy_suppression_factor": (
                    float("inf") if legacy == 0.0 else stationary / legacy
                ),
            }
        )

    return {
        "rows": rows,
        "per_realization": {
            "legacy": np.stack(per_real_legacy),          # (n_points, n_real)
            "stationary": np.stack(per_real_stationary),
        },
        "params": {
            "t_r_s": t_r,
            "tau_th_s": tau_th,
            "alpha": alpha,
            "burn_in_round_trips": n_burn,
            "n_realizations": int(n_real),
            "seed": int(seed),
            "sigma_physical": 1.0,
            "estimator": "raw second moment mean(x**2), pooled over realizations",
            "estimator_rel_sigma": math.sqrt(2.0 / int(n_real)),
        },
    }


def make_figure(study: dict, path: Path = PNG_PATH) -> Path:
    """Two curves: stationary flat at 1.0, legacy rising from the burn-in."""
    apply_pub_style()
    rows = study["rows"]
    params = study["params"]
    n = np.array([r["t_slow"] for r in rows], dtype=float)
    legacy = np.array([r["legacy_ratio"] for r in rows])
    stationary = np.array([r["stationary_ratio"] for r in rows])
    legacy_an = np.array([r["legacy_ratio_analytic"] for r in rows])
    rel_sigma = params["estimator_rel_sigma"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(1.0 - rel_sigma, 1.0 + rel_sigma, color="0.85", zorder=0,
               label=f"$\\pm1\\sigma$ estimator band ({rel_sigma:.0%})")
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":", zorder=1)
    ax.loglog(n, stationary, "o-", color="#1b7837", zorder=3,
              label="stationary_init=True (measured)")
    ax.loglog(n, legacy, "s-", color="#b2182b", zorder=3,
              label="stationary_init=False — legacy (measured)")
    ax.loglog(n, legacy_an, "--", color="#b2182b", alpha=0.6, zorder=2,
              label=r"legacy analytic $1-\alpha^2\frac{1-\alpha^{2N}}{N(1-\alpha^2)}$")
    ax.axvline(params["burn_in_round_trips"], color="0.5", lw=0.8, ls="-.",
               zorder=1)
    ax.annotate(
        f"$n_{{burn}}=\\tau_{{th}}/2t_r={params['burn_in_round_trips']:.2g}$ RT",
        xy=(params["burn_in_round_trips"], 0.30), xycoords=("data", "axes fraction"),
        xytext=(4, 0), textcoords="offset points", fontsize=7, color="0.35",
        rotation=90, va="bottom",
    )
    ax.set_xlabel("record length $N$ [round trips]")
    ax.set_ylabel(r"realized variance / target variance")
    ax.set_title(
        "AR(1) start-up bias of the TRN channel\n"
        f"$\\tau_{{th}}={params['tau_th_s']:.1e}$ s, "
        f"$t_r={params['t_r_s']:.2e}$ s, "
        f"{params['n_realizations']} realizations",
        fontsize=9,
    )
    ax.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-real", type=int, default=DEFAULT_N_REAL,
                        help="realizations per (length, flag) point")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--config", default=None, help="config YAML override")
    args = parser.parse_args(argv)

    study = run_study(n_real=args.n_real, seed=args.seed,
                      config_path=args.config)
    physical = _load_config(args.config)
    stamp = provenance_stamp(
        "analysis/trn_burnin_study.py",
        seed=args.seed,
        physical_params=physical,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    png = make_figure(study)

    rows = study["rows"]
    arrays = {
        "t_slow": np.array([r["t_slow"] for r in rows], dtype=np.int64),
        "legacy_ratio": np.array([r["legacy_ratio"] for r in rows]),
        "stationary_ratio": np.array([r["stationary_ratio"] for r in rows]),
        "legacy_ratio_analytic": np.array(
            [r["legacy_ratio_analytic"] for r in rows]
        ),
        "per_realization_legacy": study["per_realization"]["legacy"],
        "per_realization_stationary": study["per_realization"]["stationary"],
    }
    meta = {"params": study["params"], "figure": str(png)}
    npz = save_artifact(ARTIFACT_STEM, arrays, meta, provenance=stamp)

    payload = {
        "provenance": stamp,
        "params": study["params"],
        "rows": rows,
        "artifacts": {"npz": str(npz), "png": str(png)},
    }
    with JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")

    print(f"{'N [RT]':>10}{'N/tau_th':>12}{'legacy':>12}{'analytic':>12}"
          f"{'stationary':>12}{'suppression':>13}")
    for r in rows:
        print(f"{r['t_slow']:>10d}{r['n_over_tau']:>12.4f}"
              f"{r['legacy_ratio']:>12.5f}{r['legacy_ratio_analytic']:>12.5f}"
              f"{r['stationary_ratio']:>12.5f}"
              f"{r['legacy_suppression_factor']:>13.1f}x")
    print(f"\nwrote {JSON_PATH}\n      {png}\n      {npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
