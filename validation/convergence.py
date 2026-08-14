"""Order-of-accuracy verification for the LLE solver.

Four studies
------------
1. **Temporal (deterministic), three configurations.** dt is refined by six
   halvings at fixed final time and the observed order is measured for
   (a) field only, (b) field + thermal with the Euler thermal integrator,
   (c) field + thermal with the exponential integrator.
2. **Spatial.** n_tau is refined; the error must fall spectrally (geometrically,
   not algebraically) until it hits the temporal-error floor.
3. **MMS.** The exact-reference order from ``validation/mms.py``, which is the
   cleanest measurement available because it needs no Richardson extrapolation.
4. **Weak (noise on).** E[|E|^2] over 256 realizations with the quantum-vacuum
   channel enabled, fitted against dt.

Every study is run TWICE — once on the production path and once on the
second-order path — and both are reported. See below for why that is not
optional.

FINDING: the production scheme is first order, for two independent reasons
--------------------------------------------------------------------------
Measured, MMS exact reference, no extrapolation involved:

    production path         order 1.0000  (errors 1.60e-02 ... 9.98e-04)
    symmetric_drive=True    order 2.0000  (errors 4.73e-05 ... 1.85e-07)

**Barrier 1 — the un-split drive kick.** The legacy sub-step applies the whole
drive as one Euler kick before the Strang core:

    P(dt) . L(dt/2) . N(dt) . L(dt/2)

which is not palindromic, so the method is first order despite the second-order
core. Splitting the drive into two half kicks straddling the core restores the
palindrome and gives exactly 2. This is the same defect that leaves the CW
fixed point 3.08e-03 away from the analytic cubic in
``validation/analytic_cw.py``; here it is measured directly as an order.

**Barrier 2 — the lagged field<->thermal coupling.** This one is not visible
from the CW study at all. ``thermal_integrator="exponential"`` removes the
truncation error of the ΔT sub-problem *given* P_abs, and ``_thermal_step``'s
docstring says that stops the thermal update from capping the global order at
1. That is true of the ODE in isolation and false of the coupled system: the
field step uses ΔT from the START of the step while the thermal step uses
P_abs from the END of it. A one-sided coupling like that is first order however
exactly the sub-problem is solved. Measured, with the drive already
symmetrized:

    lagged  + euler        field 1.03   ΔT 1.03
    lagged  + exponential  field 1.03   ΔT 1.03      <- exponential buys nothing
    strang  + euler        field 1.03   ΔT 1.03      <- correct: Euler is order 1
    strang  + exponential  field 2.00   ΔT 2.00

So configuration (c) reaches its expected order 2 only with BOTH
``symmetric_drive=True`` and ``thermal_coupling="strang"``, and configuration
(b)'s expected order 1 is currently met for the wrong reason — the field's
error, not the Euler thermal update's.

Both fixes are opt-in flags on ``solve_lle_ssfm_jax``, default off and
bit-identical when off (pinned by ``tests/test_mms_convergence.py``). Changing
either default would alter every committed trajectory, so ``--report`` gates on
the fixed path and prints the production path beside it, unmissably, on every
run.

Why studies (b) and (c) need a derived config
---------------------------------------------
At the committed ``tau_th = 5 us`` the thermal term does not participate at all
on a tractable window: over 200 round trips ΔT reaches 4.0e-07 K and the
thermo-optic shift is 0.000*kappa, so (a), (b) and (c) come out numerically
IDENTICAL to five significant figures and the thermal integrator's order is
simply not being measured. Resolving it at the committed parameters would need
~123,000 round trips (one thermal time constant) per dt, times 64 for the
finest refinement.

:func:`thermal_testbed_config` therefore derives a config with ``tau_th = 2 ns``
(about four thermal time constants inside the window) and ``mode_volume_m3``
rescaled so the steady-state thermo-optic shift is ~1*kappa — i.e. the thermal
term is a leading-order part of the dynamics rather than a rounding effect.
This is a NUMERICS TESTBED, not a device model: it exists so the order of an
integrator can be measured where the term it integrates actually contributes.
``config/sin_params.yaml`` is never modified, and the derivation asserts
leaf-by-leaf that only the two intended values changed.

Method
------
The deterministic studies use Richardson successive differences,
``||X(dt) - X(dt/2)||``, not errors against the finest run. Using the finest run
as a reference contaminates the last few orders (its own error is the same size
as the differences being measured) and makes them drift upward — measured
1.02, 1.05, 1.10, 1.22 for what is cleanly a first-order scheme. Successive
differences scale as ``C*dt^p*(1 - 2^-p)``, so their ratio is ``2^p`` with no
reference solution needed.

CLI
---
``python -m validation.convergence --report`` prints the tables and exits
nonzero if the fixed path fails any gate: (a) < 1.9, (b) outside [0.8, 1.2],
(c) < 1.9, or weak order < 0.9.

Reference: Herr, Tikan & Kippenberg, arXiv:2604.05897v1 (7 Apr 2026).
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulator.lle_solver import _load_config                          # noqa: E402
from validation.analytic_cw import CavityParams, load_cavity_params    # noqa: E402
from validation.mms import (                                           # noqa: E402
    DEFAULT_N_TAU,
    DEFAULT_T_FINAL_ROUND_TRIPS,
    MMSSolution,
    make_source_fn,
    manufactured_solution,
    mms_error,
)

__all__ = [
    "DEFAULT_HALVINGS",
    "GATES",
    "deterministic_study",
    "mms_study",
    "observed_order",
    "spatial_study",
    "thermal_testbed_config",
    "weak_study",
]

DEFAULT_HALVINGS = 6
DEFAULT_T_SLOW = DEFAULT_T_FINAL_ROUND_TRIPS

#: Acceptance gates. Keyed by study id; each maps to (low, high) on the order.
GATES: dict[str, tuple[float, float]] = {
    "a_field_only": (1.9, math.inf),
    "b_thermal_euler": (0.8, 1.2),
    "c_thermal_exponential": (1.9, math.inf),
    "weak": (0.9, math.inf),
}

_CONFIG_LABELS = {
    "a_field_only": "(a) field only, thermal decoupled",
    "b_thermal_euler": "(b) thermal live, integrator=euler",
    "c_thermal_exponential": "(c) thermal live, integrator=exponential",
}


# ---------------------------------------------------------------------------
# Order estimation
# ---------------------------------------------------------------------------
def observed_order(
    errors: Sequence[float] | np.ndarray, dts: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Successive log-log slopes: ``log(e_i/e_{i+1}) / log(dt_i/dt_{i+1})``.

    Returns an array of length ``len(errors) - 1``. Entries where either error
    is non-positive (exact agreement, or round-off floor reached) come back as
    ``nan`` rather than raising, so a spectral study that bottoms out is still
    reportable.
    """
    e = np.asarray(errors, dtype=np.float64)
    h = np.asarray(dts, dtype=np.float64)
    if e.shape != h.shape:
        raise ValueError(f"errors {e.shape} and dts {h.shape} must have equal shape.")
    if e.size < 2:
        raise ValueError("need at least two points to estimate an order.")
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.log(e[:-1] / e[1:])
        den = np.log(h[:-1] / h[1:])
        out = num / den
    out[~np.isfinite(out)] = np.nan
    out[(e[:-1] <= 0.0) | (e[1:] <= 0.0)] = np.nan
    return out


def _asymptotic_order(orders: np.ndarray, take: int = 3) -> float:
    """The order in the asymptotic (finest) regime: mean of the last ``take``."""
    good = orders[np.isfinite(orders)]
    if good.size == 0:
        return float("nan")
    return float(np.mean(good[-min(take, good.size):]))


# ---------------------------------------------------------------------------
# The thermal numerics testbed config
# ---------------------------------------------------------------------------
def thermal_testbed_config(
    params: CavityParams,
    *,
    pin: float,
    delta_omega: float,
    tau_th: float = 2.0e-9,
    worst_shift_over_kappa: float = 10.0,
    source: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> str:
    """Derive a config in which the thermo-optic term actually participates.

    Two leaves change relative to the committed config: ``tau_th_s`` (shortened
    so several thermal time constants fit inside the integration window) and
    ``mode_volume_m3`` (rescaled so the steady-state thermo-optic shift is
    ``shift_over_kappa * kappa`` at the study's operating point). ``dn_dT_per_k``
    is left at its committed value, so the thermo-optic pathway is fully live.

    The shift is calibrated from the same steady-state expression the solver's
    own pre-flight check uses:

        U_cw   = kappa_c*pin / ((kappa/2)^2 + delta_omega^2)     [J]
        P_abs  = kappa_i * U_cw                                  [W]
        R_th   = tau_th*Gamma_th / (rho*Cp*V)                    [K/W]
        shift  = (omega0/n0)*dn_dT * R_th * P_abs                [rad/s]

    solved for V. The calibration targets the WORST CASE delta_omega = 0, which
    is exactly what the solver's ``thermal_shift_max_over_kappa = 15``
    pre-flight guard tests. Calibrating at the operating detuning instead would
    leave the worst case at ((kappa/2)^2 + delta_omega^2)/(kappa/2)^2 = 17x the
    target and trip that guard. At the default 10 the guard passes with margin
    and the shift AT THE OPERATING POINT (delta_omega = 2*kappa) is ~0.6*kappa
    -- leading order, which is what the study needs.

    See the module docstring: this is a numerics testbed, not a device model.
    """
    src = Path(source) if source is not None else _REPO_ROOT / "config" / "sin_params.yaml"
    with src.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    physical = cfg.get("physical_parameters")
    if not isinstance(physical, dict):
        raise ValueError(f"{src} has no physical_parameters block.")
    original = dict(physical)

    omega0 = 2.0 * math.pi * 299_792_458.0 / float(physical["pump_wavelength_m"])
    to_coeff = (omega0 / float(physical["n0"])) * float(physical["dn_dT_per_k"])
    # Worst case (delta_omega = 0) -- the quantity the solver's pre-flight guards.
    u_cw_worst = params.kappa_c * float(pin) / ((params.kappa / 2.0) ** 2)
    p_abs_worst = params.kappa_i * u_cw_worst
    if p_abs_worst <= 0.0:
        raise ValueError("operating point has zero absorbed power; cannot calibrate.")
    r_th_target = float(worst_shift_over_kappa) * params.kappa / (to_coeff * p_abs_worst)
    volume = (
        float(tau_th) * float(physical["Gamma_th"])
        / (float(physical["rho_kg_per_m3"]) * float(physical["Cp_j_per_kg_k"]) * r_th_target)
    )

    physical["tau_th_s"] = float(tau_th)
    physical["mode_volume_m3"] = float(volume)

    if out_dir is None:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", prefix="convergence_thermal_", delete=False, encoding="utf-8"
        )
        dest = Path(handle.name)
        yaml.safe_dump(cfg, handle, sort_keys=False)
        handle.close()
    else:
        dest = Path(out_dir) / "convergence_thermal_testbed.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)

    reloaded = _load_config(dest)
    changed = sorted(k for k in original if original[k] != reloaded[k])
    if changed != ["mode_volume_m3", "tau_th_s"]:
        raise AssertionError(
            f"thermal testbed config changed {changed}, expected exactly "
            f"['mode_volume_m3', 'tau_th_s']"
        )
    return str(dest)


# ---------------------------------------------------------------------------
# One solver run at a given dt
# ---------------------------------------------------------------------------
def _solve_state(
    dt: float,
    sol: MMSSolution,
    *,
    n_tau: int,
    t_slow: int,
    config_path: str,
    thermal_integrator: str,
    symmetric_drive: bool,
    thermal_coupling: str,
    with_source: bool,
) -> tuple[np.ndarray, float]:
    """Return ``(E_final, DeltaT_final)`` for one dt. dt must divide t_r exactly."""
    import jax
    from simulator.lle_solver import solve_lle_ssfm_jax
    from simulator.noise_config import NoiseConfig

    p = sol.params
    ratio = p.t_r / float(dt)
    fine_cadence_M = int(round(ratio))
    if fine_cadence_M < 1 or abs(ratio - fine_cadence_M) > 1e-9 * fine_cadence_M:
        raise ValueError(f"dt={dt:.6e} must divide t_r={p.t_r:.6e} an integer number of times.")

    grid = sol.tau_grid(n_tau)
    out = solve_lle_ssfm_jax(
        pin=sol.pin,
        delta_omega=sol.delta_omega,
        t_slow=int(t_slow),
        beta=list(sol.beta),
        kappa=p.kappa,
        kappa_c=p.kappa_c,
        rng_key=jax.random.PRNGKey(0),
        n_tau=int(n_tau),
        config_path=config_path,
        snapshot_interval=int(t_slow),
        e0_override=sol.field(grid, 0.0),
        fine_cadence_M=fine_cadence_M,
        n_substeps=1,
        source_fn=make_source_fn(sol, n_tau) if with_source else None,
        symmetric_drive=symmetric_drive,
        thermal_coupling=thermal_coupling,
        noise_config=NoiseConfig.all_off(thermal_integrator=thermal_integrator),
    )
    return np.asarray(out["e_final"])[0], float(np.asarray(out["DeltaT_history"])[0, -1])


# ---------------------------------------------------------------------------
# Study 1: temporal, deterministic
# ---------------------------------------------------------------------------
def deterministic_study(
    sol: MMSSolution | None = None,
    *,
    halvings: int = DEFAULT_HALVINGS,
    n_tau: int = DEFAULT_N_TAU,
    t_slow: int = DEFAULT_T_SLOW,
    symmetric_drive: bool = False,
    thermal_coupling: str = "lagged",
    testbed_config: str | None = None,
) -> dict[str, dict[str, Any]]:
    """dt refinement over ``halvings`` halvings, for the three configurations.

    Richardson successive differences (see the module docstring). Configuration
    (a) runs on the thermo-optic-free config with the manufactured forcing, so
    its differences are true discretization differences of the field alone;
    (b) and (c) run on the thermal testbed config with the forcing off, and the
    reported order is that of the coupled (field, ΔT) state.
    """
    if sol is None:
        sol = manufactured_solution()
    if testbed_config is None:
        testbed_config = thermal_testbed_config(
            sol.params, pin=sol.pin, delta_omega=sol.delta_omega
        )

    ms = [2 ** i for i in range(halvings + 1)]
    dts = np.array([sol.params.t_r / m for m in ms], dtype=np.float64)
    results: dict[str, dict[str, Any]] = {}

    specs = (
        ("a_field_only", sol.params.config_path, "euler", True),
        ("b_thermal_euler", testbed_config, "euler", False),
        ("c_thermal_exponential", testbed_config, "exponential", False),
    )
    for key, cfg_path, integrator, with_source in specs:
        fields, temps = [], []
        for dt in dts:
            e, dT = _solve_state(
                float(dt), sol, n_tau=n_tau, t_slow=t_slow, config_path=cfg_path,
                thermal_integrator=integrator, symmetric_drive=symmetric_drive,
                thermal_coupling=thermal_coupling, with_source=with_source,
            )
            fields.append(e)
            temps.append(dT)
        diffs, dts_diff = [], []
        for i in range(len(ms) - 1):
            d_field = np.linalg.norm(fields[i] - fields[i + 1]) / np.linalg.norm(fields[i + 1])
            if key == "a_field_only":
                d = d_field
            else:
                # Coupled state: combine the field and thermal differences on a
                # common relative scale so neither component can hide the other.
                denom = abs(temps[i + 1]) if temps[i + 1] != 0.0 else 1.0
                d_temp = abs(temps[i] - temps[i + 1]) / denom
                d = math.hypot(d_field, d_temp)
            diffs.append(d)
            dts_diff.append(dts[i])
        diffs = np.asarray(diffs)
        dts_diff = np.asarray(dts_diff)
        orders = observed_order(diffs, dts_diff)
        results[key] = {
            "label": _CONFIG_LABELS[key],
            "dts": dts_diff,
            "errors": diffs,
            "orders": orders,
            "order": _asymptotic_order(orders),
            "delta_T_final": float(temps[-1]),
            "config_path": cfg_path,
        }
    return results


# ---------------------------------------------------------------------------
# Study 2: spatial
# ---------------------------------------------------------------------------
def spatial_study(
    sol: MMSSolution | None = None,
    *,
    n_taus: Sequence[int] = (16, 20, 24, 28, 32, 40, 48, 64),
    t_slow: int = DEFAULT_T_SLOW,
    fine_cadence_M: int = 64,
    symmetric_drive: bool = True,
) -> dict[str, Any]:
    """n_tau refinement against the exact manufactured solution.

    The error must fall GEOMETRICALLY (spectral accuracy) until it reaches the
    temporal-error floor, then go flat. The flat tail is expected, not a
    failure — it is where the spatial error has stopped being the dominant
    term.

    The defaults deliberately push the temporal error out of the way
    (``symmetric_drive=True`` at ``dt = t_r/64`` gives ~1e-8, against a spatial
    error of ~4e-4 at n_tau = 16). That is not a thumb on the scale: a spatial
    refinement study can only see the spatial error while it dominates, and on
    the production scheme the temporal floor is 5.0e-04 — larger than the
    spatial error at EVERY n_tau on this grid, so the study would show a flat
    line and measure nothing. This is the one study that is scheme-independent
    by construction, so it is run once rather than once per scheme.
    """
    if sol is None:
        sol = manufactured_solution()
    errors = []
    for n in n_taus:
        errors.append(
            mms_error(
                sol.params.t_r / int(fine_cadence_M), int(n), sol=sol,
                t_final_round_trips=t_slow, symmetric_drive=symmetric_drive,
            )
        )
    errors = np.asarray(errors, dtype=np.float64)
    floor = float(np.min(errors))
    # Spectral decay is measured over the pre-floor points only: once the
    # temporal floor is reached the spatial error is no longer observable.
    pre_floor = errors > 10.0 * floor
    decades = (
        math.log10(errors[0] / errors[pre_floor][-1])
        if pre_floor.sum() >= 2 else 0.0
    )
    return {
        "n_taus": np.asarray(n_taus, dtype=np.int64),
        "errors": errors,
        "floor": floor,
        "decades_before_floor": decades,
        "n_pre_floor": int(pre_floor.sum()),
    }


# ---------------------------------------------------------------------------
# Study 3: MMS exact-reference order
# ---------------------------------------------------------------------------
def mms_study(
    sol: MMSSolution | None = None,
    *,
    halvings: int = DEFAULT_HALVINGS,
    n_tau: int = DEFAULT_N_TAU,
    t_slow: int = DEFAULT_T_SLOW,
    symmetric_drive: bool = False,
) -> dict[str, Any]:
    """Order against the EXACT manufactured solution — no extrapolation."""
    if sol is None:
        sol = manufactured_solution()
    ms = [2 ** i for i in range(halvings + 1)]
    dts = np.array([sol.params.t_r / m for m in ms], dtype=np.float64)
    errors = np.array(
        [
            mms_error(float(dt), n_tau, sol=sol, t_final_round_trips=t_slow,
                      symmetric_drive=symmetric_drive)
            for dt in dts
        ]
    )
    orders = observed_order(errors, dts)
    return {
        "label": "MMS (exact reference)",
        "dts": dts,
        "errors": errors,
        "orders": orders,
        "order": _asymptotic_order(orders),
    }


# ---------------------------------------------------------------------------
# Study 4: weak convergence with noise on
# ---------------------------------------------------------------------------
def weak_study(
    sol: MMSSolution | None = None,
    *,
    n_realizations: int = 256,
    halvings: int = 4,
    n_tau: int = 64,
    t_slow: int = 100,
    symmetric_drive: bool = False,
    seed: int = 20260814,
) -> dict[str, Any]:
    """Weak order of E[|E|^2] with the quantum-vacuum channel enabled.

    Common random numbers: the SAME ``n_realizations`` PRNG keys are used at
    every dt, so the estimator differences across dt are as correlated as the
    solver's internal per-step ``fold_in`` ladder allows. This is the variance
    reduction the study needs; with independent seeds per dt the Monte-Carlo
    error (~sigma/sqrt(N)) swamps the discretization error entirely and the fit
    measures sampling noise.

    Honest caveat, reported alongside the number: at these parameters the weak
    error is DETERMINISTIC-dominated. The vacuum contribution to |E|^2 is ~1e-5
    relative while the dt=t_r discretization error is ~1.6e-2, so the fitted
    weak order confirms that adding the noise channel does not DEGRADE the
    deterministic order — it does not isolate the order of the noise
    discretization on its own. ``noise_fraction`` in the returned dict is that
    ratio, so the caveat is a measured quantity rather than a claim.
    """
    import jax
    from simulator.lle_solver import solve_lle_ssfm_jax
    from simulator.noise_config import NoiseConfig

    if sol is None:
        sol = manufactured_solution()
    p = sol.params
    grid = sol.tau_grid(n_tau)
    e0 = np.repeat(sol.field(grid, 0.0)[None, :], int(n_realizations), axis=0)
    ms = [2 ** i for i in range(halvings + 1)]
    dts = np.array([p.t_r / m for m in ms], dtype=np.float64)

    # All n_realizations trajectories in ONE vmapped call: delta_omega is
    # broadcast to (n_realizations, t_slow) and the solver derives one
    # independent noise key per trajectory from the single rng_key. Passing the
    # SAME rng_key at every dt is the common-random-numbers coupling — the
    # per-trajectory key chain is then identical across dt, which is the
    # strongest correlation the solver's per-step fold_in ladder allows.
    dw = np.full((int(n_realizations), int(t_slow)), sol.delta_omega, dtype=np.float64)
    source = make_source_fn(sol, n_tau)

    def _run(m, quantum):
        return solve_lle_ssfm_jax(
            pin=sol.pin, delta_omega=dw, t_slow=int(t_slow),
            beta=list(sol.beta), kappa=p.kappa, kappa_c=p.kappa_c,
            rng_key=jax.random.PRNGKey(int(seed)), n_tau=int(n_tau),
            config_path=p.config_path, snapshot_interval=int(t_slow),
            e0_override=e0, fine_cadence_M=int(m), n_substeps=1,
            source_fn=source, symmetric_drive=symmetric_drive,
            quantum_noise_enabled=quantum,
            quantum_noise_seed_vacuum_init=False,
            noise_config=NoiseConfig.all_off(quantum_vacuum=quantum),
        )

    means, sems = [], []
    for m in ms:
        vals = np.asarray(_run(m, True)["U_int_history"])[:, -1] / p.t_r
        means.append(float(vals.mean()))
        sems.append(float(vals.std(ddof=1) / math.sqrt(vals.size)))

    means = np.asarray(means)
    sems = np.asarray(sems)
    ref = means[-1]
    # Successive differences, for the same reason as the deterministic studies:
    # differencing against the finest estimator contaminates the last orders
    # with that estimator's own error (measured 1.11, 1.23, 1.60 for what is a
    # first-order scheme). |mean(dt) - mean(dt/2)| scales as C*dt^p*(1-2^-p).
    errors = np.abs(means[:-1] - means[1:]) / abs(ref)
    orders = observed_order(errors, dts[:-1])

    # How much of the observable is noise-borne, at the coarsest dt.
    det_val = float(np.asarray(_run(1, False)["U_int_history"])[0, -1]) / p.t_r
    noise_fraction = abs(means[0] - det_val) / abs(det_val)

    # Fit only the RESOLVED range. Once the discretization difference drops
    # below the noise-induced difference the sequence stops decreasing and the
    # remaining points measure the estimator floor, not an order. On the
    # production scheme every point is resolved; with the drive symmetrized the
    # deterministic error is second order and crosses under the (first-order,
    # but ~5e-06-relative) noise contribution partway down the sweep, which
    # shows up here as the prefix ending early rather than as a garbage slope.
    decreasing = np.ones(errors.size, dtype=bool)
    for i in range(1, errors.size):
        if not (errors[i] < errors[i - 1]):
            decreasing[i:] = False
            break
    n_fit = int(decreasing.sum())
    if n_fit >= 2 and np.all(errors[:n_fit] > 0.0):
        fit_order = float(
            np.polyfit(np.log(dts[:n_fit]), np.log(errors[:n_fit]), 1)[0]
        )
    else:
        fit_order = float("nan")

    return {
        "label": "weak order, E[|E|^2], quantum vacuum on",
        "dts": dts[:-1],
        "means": means,
        "sems": sems,
        "errors": errors,
        "orders": orders,
        "order": fit_order,
        "n_fit": int(n_fit),
        "successive_order": _asymptotic_order(orders),
        "n_realizations": int(n_realizations),
        "noise_fraction": float(noise_fraction),
        "max_sem_over_error": float(np.max(sems[:-1] / np.maximum(errors * abs(ref), 1e-300))),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_series(title: str, dts, errors, orders, extra: str = "") -> None:
    print(f"\n{title}{extra}")
    print("-" * 78)
    print(f"{'dt [s]':>14} {'t_r/dt':>8} {'error':>16} {'order':>10}")
    t_r = None
    for i in range(len(dts)):
        if t_r is None:
            t_r = dts[0]
        ratio = dts[0] / dts[i]
        o = "" if i == 0 else f"{orders[i-1]:>10.4f}"
        print(f"{dts[i]:>14.6e} {ratio:>8.0f} {errors[i]:>16.6e} {o}")
    print("-" * 78)


def _run_all(
    sol: MMSSolution,
    *,
    symmetric_drive: bool,
    thermal_coupling: str,
    halvings: int,
    n_tau: int,
    t_slow: int,
    weak_realizations: int,
    testbed_config: str,
    skip_weak: bool,
) -> dict[str, Any]:
    det = deterministic_study(
        sol, halvings=halvings, n_tau=n_tau, t_slow=t_slow,
        symmetric_drive=symmetric_drive, thermal_coupling=thermal_coupling,
        testbed_config=testbed_config,
    )
    out: dict[str, Any] = dict(det)
    out["mms"] = mms_study(
        sol, halvings=halvings, n_tau=n_tau, t_slow=t_slow,
        symmetric_drive=symmetric_drive,
    )
    if not skip_weak:
        out["weak"] = weak_study(
            sol, n_realizations=weak_realizations, symmetric_drive=symmetric_drive,
        )
    return out


def _gate(result: dict[str, Any]) -> list[tuple[str, float, tuple[float, float], bool]]:
    rows = []
    for key, (lo, hi) in GATES.items():
        if key not in result:
            continue
        order = result[key]["order"]
        ok = bool(np.isfinite(order) and lo <= order <= hi)
        rows.append((key, order, (lo, hi), ok))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m validation.convergence",
        description="Order-of-accuracy verification for the LLE solver.",
    )
    ap.add_argument("--report", action="store_true",
                    help="print the full report and gate the exit code")
    ap.add_argument("--halvings", type=int, default=DEFAULT_HALVINGS)
    ap.add_argument("--n-tau", type=int, default=DEFAULT_N_TAU)
    ap.add_argument("--t-slow", type=int, default=DEFAULT_T_SLOW)
    ap.add_argument("--weak-realizations", type=int, default=256)
    ap.add_argument("--skip-weak", action="store_true",
                    help="skip the (slow) weak-convergence study")
    ap.add_argument("--production-only", action="store_true",
                    help="measure only the shipping scheme (all flags off)")
    args = ap.parse_args(argv)

    sol = manufactured_solution()
    p = sol.params
    testbed = thermal_testbed_config(p, pin=sol.pin, delta_omega=sol.delta_omega)

    print("=" * 78)
    print("ORDER-OF-ACCURACY VERIFICATION  (validation/convergence.py)")
    print("=" * 78)
    print(f"  kappa = {p.kappa:.6e} rad/s   gamma = {p.gamma:.6e} J^-1 s^-1")
    print(f"  t_r = {p.t_r:.6e} s   kappa*t_r = {p.kappa * p.t_r:.6e}")
    print(f"  manufactured solution: A = {sol.amplitude:.6e} sqrt(J), "
          f"gamma*A^2/kappa = {p.gamma * sol.amplitude ** 2 / p.kappa:.3f}")
    print(f"  m_k = {sol.mode_number}, omega_m = {sol.omega_m / p.kappa:.2f}*kappa, "
          f"w = t_r/{p.t_r / sol.width:.0f}, beta = {sol.beta}")
    print(f"  window: {args.t_slow} round trips = {args.t_slow * p.t_r:.4e} s; "
          f"{args.halvings} halvings (dt = t_r .. t_r/{2 ** args.halvings})")
    print(f"  thermal testbed config: {testbed}")

    schemes = [("PRODUCTION (all flags off - the shipping scheme)", False, "lagged")]
    if not args.production_only:
        schemes.append(
            ("FIXED (symmetric_drive=True, thermal_coupling='strang')", True, "strang")
        )

    results = {}
    for title, sym, coupling in schemes:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        res = _run_all(
            sol, symmetric_drive=sym, thermal_coupling=coupling,
            halvings=args.halvings, n_tau=args.n_tau, t_slow=args.t_slow,
            weak_realizations=args.weak_realizations, testbed_config=testbed,
            skip_weak=args.skip_weak,
        )
        results[title] = res

        for key in ("a_field_only", "b_thermal_euler", "c_thermal_exponential"):
            r = res[key]
            extra = "" if key == "a_field_only" else f"   [DeltaT_final = {r['delta_T_final']:.4e} K]"
            _print_series(r["label"] + "  (Richardson differences)", r["dts"],
                          r["errors"], r["orders"], extra)
            print(f"  asymptotic order = {r['order']:.4f}")

        m = res["mms"]
        _print_series(m["label"], m["dts"], m["errors"], m["orders"])
        print(f"  asymptotic order = {m['order']:.4f}")

        if "weak" in res:
            w = res["weak"]
            _print_series(w["label"], w["dts"], w["errors"], w["orders"],
                          f"   [{w['n_realizations']} realizations, common random numbers]")
            print(f"  asymptotic order = {w['order']:.4f}")
            print(f"  noise-borne fraction of E[|E|^2] = {w['noise_fraction']:.3e} "
                  f"(deterministic-dominated: see docstring)")

    sp = spatial_study(sol, t_slow=args.t_slow)
    print("\n" + "=" * 78)
    print("SPATIAL REFINEMENT (scheme-independent; run once)")
    print("=" * 78)
    print("  dt = t_r/64 with symmetric_drive=True, so the temporal error "
          f"({sp['floor']:.2e}) sits far below the spatial one.")
    print("-" * 78)
    print(f"{'n_tau':>8} {'samples/width':>15} {'MMS error':>16} {'ratio':>10}")
    prev = None
    for n, e in zip(sp["n_taus"], sp["errors"]):
        r = "" if prev is None else f"{prev / e:>10.2f}"
        print(f"{int(n):>8d} {n * sol.width / p.t_r:>15.2f} {e:>16.6e} {r}")
        prev = e
    print("-" * 78)
    print(f"  fell {sp['decades_before_floor']:.1f} decades over "
          f"{sp['n_pre_floor']} points before hitting the temporal floor "
          f"{sp['floor']:.3e}")
    print("  Spectral accuracy = geometric decay (ratios grow), then flat once "
          "the\n  temporal error dominates. Algebraic decay would show CONSTANT "
          "ratios.")

    print("\n" + "=" * 78)
    print("GATES")
    print("=" * 78)
    gate_title = schemes[-1][0]
    rows = _gate(results[gate_title])
    print(f"  gated on: {gate_title}")
    print(f"  {'study':>24} {'order':>10} {'required':>18} {'':>6}")
    all_ok = True
    for key, order, (lo, hi), ok in rows:
        req = f">= {lo}" if math.isinf(hi) else f"[{lo}, {hi}]"
        print(f"  {key:>24} {order:>10.4f} {req:>18} {'PASS' if ok else 'FAIL':>6}")
        all_ok = all_ok and ok
    if len(schemes) > 1:
        prod = results[schemes[0][0]]
        print("\n  FOR COMPARISON - the shipping scheme, ungated:")
        for key in ("a_field_only", "b_thermal_euler", "c_thermal_exponential"):
            print(f"  {key:>24} {prod[key]['order']:>10.4f}")
        print(f"  {'mms':>24} {prod['mms']['order']:>10.4f}")
        print(
            "\n  The shipping scheme is FIRST order: the drive kick is applied\n"
            "  once per sub-step before the linear half-step (non-palindromic),\n"
            "  and the field<->thermal coupling is lagged. Both fixes are opt-in\n"
            "  flags, default off, because enabling either changes every\n"
            "  committed trajectory. See the module docstring."
        )
    print("\n" + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
