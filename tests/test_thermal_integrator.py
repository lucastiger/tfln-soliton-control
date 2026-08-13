"""Thermo-optic ODE integrator: euler (legacy, default) vs exponential (exact).

``simulator.lle_solver._fine_step`` advances the single-pole thermal ODE

    dDeltaT/dt = -DeltaT/tau_th + Gamma_th*P_abs/(rho*Cp*V)

once per fine step. The field steps around it are exact exponentials inside a
Strang splitting (2nd order), so an explicit-Euler thermal step caps the GLOBAL
order at 1 whenever thermo-optic feedback is active. The ODE is LINEAR in
DeltaT, so for piecewise-constant ``P_abs`` -- which is what a fine step sees --
the update

    a = exp(-dt/tau_th),  DeltaT_next = a*DeltaT + (1-a)*R_th*P_abs

is EXACT, carries no truncation error at all, and is unconditionally stable.

These tests pin three things:

1. the default (``thermal_integrator = "euler"``) still reproduces the committed
   S5 golden trajectories byte for byte, and traces no extra ops;
2. the exponential branch reproduces the analytic relaxation to rtol 1e-12 at
   dt/tau ratios spanning four orders of magnitude, INCLUDING ratios where the
   Euler scheme is unstable -- the stability contrast at dt/tau = 2.1 is a
   paper figure;
3. the two schemes converge to each other as dt/tau -> 0, at the rate the
   truncation analysis predicts.

Reference: Herr, Tikan & Kippenberg, arXiv:2604.05897v1.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simulator.lle_solver import (
    _PER_TRAJ,
    _PER_TRAJ_THERMAL_EXP,
    _load_config,
    _thermal_step,
    d2_to_beta2_lle,
    load_dint_grid,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
)
from simulator.noise_config import NoiseConfig
from validation.noise_off_identity import (
    ARRAY_NAMES,
    CONFIG_PATH,
    DEFAULT_GOLDEN_DIR,
    DISPERSION_CSV,
    PARAM_SETS,
    REPO_ROOT,
    SNAPSHOT_INTERVAL,
    compare_to_golden,
)

GOLDEN_DIR = REPO_ROOT / DEFAULT_GOLDEN_DIR

#: Grid size above which a parameter set is hidden behind ``--runslow``, matching
#: tests/test_noise_off_identity.py so the two suites agree on what is expensive.
SLOW_N_TAU = 1024


def _cases():
    return [
        pytest.param(
            ps,
            id=str(ps["name"]),
            marks=[pytest.mark.slow] if int(ps["n_tau"]) >= SLOW_N_TAU else [],
        )
        for ps in PARAM_SETS
    ]


def _thermal_dict(config_path=None) -> dict:
    """The ``thermal`` mapping ``_thermal_step`` consumes, as jnp float64 scalars.

    Built from the committed config exactly as ``solve_lle_ssfm_jax`` builds it,
    so the tests drive the real coefficients rather than invented ones.
    """
    p = _load_config(config_path)
    return {
        "tau_th": jnp.array(float(p["tau_th_s"]), dtype=jnp.float64),
        "Gamma_th": jnp.array(float(p["Gamma_th"]), dtype=jnp.float64),
        "rho": jnp.array(float(p["rho_kg_per_m3"]), dtype=jnp.float64),
        "Cp": jnp.array(float(p["Cp_j_per_kg_k"]), dtype=jnp.float64),
        "V": jnp.array(float(p["mode_volume_m3"]), dtype=jnp.float64),
    }


def _steady_state(thermal: dict, p_abs: float) -> float:
    """DeltaT_ss = R_th*P_abs = tau_th*Gamma_th*P_abs/(rho*Cp*V)  [K]."""
    return float(
        thermal["tau_th"]
        * thermal["Gamma_th"]
        * p_abs
        / (thermal["rho"] * thermal["Cp"] * thermal["V"])
    )


def _integrate(thermal: dict, p_abs: float, dt: float, n_steps: int,
               exponential: bool, delta_t0: float = 0.0) -> np.ndarray:
    """Run ``_thermal_step`` for ``n_steps`` with P_abs HELD CONSTANT.

    Returns the (n_steps,) trajectory in float64. Constant ``p_abs`` is exactly
    the regime in which the exponential update is claimed to be exact, so this
    isolates the integrator from the field dynamics.
    """
    dt_arr = jnp.array(float(dt), dtype=jnp.float64)
    p_arr = jnp.array(float(p_abs), dtype=jnp.float64)
    x = jnp.array(float(delta_t0), dtype=jnp.float64)
    out = []
    for _ in range(int(n_steps)):
        x = _thermal_step(x, p_arr, dt_arr, thermal, exponential)
        out.append(float(x))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. The default is untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param_set", _cases())
def test_euler_default_bit_identical(param_set):
    """Default euler still reproduces the S5 goldens, byte for byte.

    Three separate claims, because the change had three ways to go wrong:

    (a) extracting the update into ``_thermal_step`` must not have perturbed the
        expression tree -- checked against the committed golden trajectories
        (``strict=True``, raw-byte comparison, 0 ULP);
    (b) naming ``thermal_integrator="euler"`` explicitly must be byte-identical
        to leaving it at the default;
    (c) the euler branch must trace no extra ops, and the exponential branch
        must genuinely differ -- otherwise (a) and (b) are vacuous.
    """
    name = str(param_set["name"])

    # (a) goldens, raw-byte
    report = compare_to_golden(param_set, strict=True, golden_dir=GOLDEN_DIR)
    assert report["ok"], (
        f"[{name}] euler default no longer matches its S5 golden: "
        f"{report['n_differences']} elements differ "
        f"(golden {report['golden_npz']}); "
        f"version_mismatch={report['version_mismatch']}"
    )

    # (b) explicit "euler" == default, byte for byte
    n_tau = int(param_set["n_tau"])
    physical = _load_config(str(CONFIG_PATH))
    _kappa_i, kappa_c, kappa = resolve_cavity_rates(str(CONFIG_PATH))
    beta2 = float(d2_to_beta2_lle(physical["d2_rad_per_s2"], physical["fsr_hz"]))

    def _solve(integrator):
        return solve_lle_ssfm_jax(
            pin=float(physical["pin_w"]),
            delta_omega=float(param_set["dw_over_kappa"]) * float(kappa),
            t_slow=int(param_set["t_slow"]),
            beta=[beta2],
            kappa=float(kappa),
            kappa_c=float(kappa_c),
            rng_key=jax.random.PRNGKey(0),
            n_tau=n_tau,
            config_path=str(CONFIG_PATH),
            snapshot_interval=SNAPSHOT_INTERVAL,
            d_int_grid=load_dint_grid(n_tau, csv_path=DISPERSION_CSV).grid,
            n_substeps=int(param_set["n_substeps"]),
            noise_config=NoiseConfig.all_off(thermal_integrator=integrator),
        )

    default = _solve("euler")
    explicit = solve_lle_ssfm_jax(
        pin=float(physical["pin_w"]),
        delta_omega=float(param_set["dw_over_kappa"]) * float(kappa),
        t_slow=int(param_set["t_slow"]),
        beta=[beta2],
        kappa=float(kappa),
        kappa_c=float(kappa_c),
        rng_key=jax.random.PRNGKey(0),
        n_tau=n_tau,
        config_path=str(CONFIG_PATH),
        snapshot_interval=SNAPSHOT_INTERVAL,
        d_int_grid=load_dint_grid(n_tau, csv_path=DISPERSION_CSV).grid,
        n_substeps=int(param_set["n_substeps"]),
        noise_config=NoiseConfig.all_off(),           # thermal_integrator unset
    )
    for key in ("U_int_history", "DeltaT_history", "e_final", "E_snapshots"):
        got = np.ascontiguousarray(np.asarray(default[key]))
        want = np.ascontiguousarray(np.asarray(explicit[key]))
        assert got.dtype == want.dtype and got.shape == want.shape, key
        assert got.tobytes() == want.tobytes(), (
            f"[{name}] thermal_integrator='euler' differs from the field "
            f"default in {key}"
        )

    # (c) the exponential branch is a different object AND a different answer
    assert _PER_TRAJ is not _PER_TRAJ_THERMAL_EXP
    exponential = _solve("exponential")
    assert not np.array_equal(
        np.asarray(exponential["DeltaT_history"]),
        np.asarray(default["DeltaT_history"]),
    ), (
        f"[{name}] the exponential integrator produced an identical DeltaT "
        f"history; the euler bit-identity above would then be vacuous."
    )


def test_euler_branch_traces_no_exponential_ops():
    """Structural: the disabled branch adds zero ops to the traced graph.

    ``thermal_exponential`` is a static Python bool, so this is checkable in the
    jaxpr rather than by timing: euler must emit neither ``exp`` nor ``expm1``,
    and the exponential branch must emit exactly the ``expm1`` that keeps
    ``1 - a`` cancellation-free.
    """
    thermal = _thermal_dict()
    dt = jnp.array(1.0e-11, dtype=jnp.float64)

    def _primitives(exponential):
        jpr = jax.make_jaxpr(
            lambda d, p: _thermal_step(d, p, dt, thermal, exponential)
        )(jnp.array(0.0), jnp.array(1.0e-3))
        return [eqn.primitive.name for eqn in jpr.jaxpr.eqns]

    euler = _primitives(False)
    assert "exp" not in euler and "expm1" not in euler, euler

    exp_branch = _primitives(True)
    assert "expm1" in exp_branch, exp_branch
    assert "exp" in exp_branch, exp_branch


# ---------------------------------------------------------------------------
# 2. The exponential branch is exact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt_over_tau", [0.01, 0.5, 2.0, 10.0])
def test_exponential_matches_analytic_relaxation(dt_over_tau):
    """With P_abs constant, DeltaT follows the exact exponential to rtol 1e-12.

    Analytic solution from DeltaT(0) = 0:

        DeltaT(n*dt) = DeltaT_ss * (1 - exp(-n*dt/tau_th)),
        DeltaT_ss    = tau_th*Gamma_th*P_abs/(rho*Cp*V).

    The reference is evaluated as ``-expm1(-n*x)`` -- a closed form in ``n*x``,
    not a power of the recurrence's own ``a`` -- so it is an independent check
    of the update rather than a restatement of it.

    Includes dt/tau = 2.0 and 10.0, where the Euler scheme is at or past its
    stability limit |1 - dt/tau| < 1 and the exponential scheme is still exact:
    ``a = exp(-dt/tau)`` lies in (0, 1) for every dt > 0.
    """
    thermal = _thermal_dict()
    tau = float(thermal["tau_th"])
    dt = dt_over_tau * tau
    p_abs = 5.0e-3                     # W, the soliton-operating-point scale
    n_steps = 200

    got = _integrate(thermal, p_abs, dt, n_steps, exponential=True)

    n = np.arange(1, n_steps + 1, dtype=np.float64)
    want = _steady_state(thermal, p_abs) * (-np.expm1(-n * dt_over_tau))

    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_exponential_is_stable_where_euler_diverges():
    """dt/tau = 2.1: Euler blows up, the exponential update stays exact.

    The Euler amplification factor is 1 - dt/tau, so |1 - 2.1| = 1.1 > 1 and the
    error grows by 10% per step without bound; the exponential factor is
    exp(-2.1) = 0.122, so it contracts. This contrast is a paper figure, so it
    is asserted explicitly rather than left implicit in the parametrization
    above.
    """
    thermal = _thermal_dict()
    tau = float(thermal["tau_th"])
    dt_over_tau = 2.1
    dt = dt_over_tau * tau
    p_abs = 5.0e-3
    n_steps = 200
    ss = _steady_state(thermal, p_abs)

    euler = _integrate(thermal, p_abs, dt, n_steps, exponential=False)
    exponential = _integrate(thermal, p_abs, dt, n_steps, exponential=True)

    # Euler: what diverges is the DEVIATION from the fixed point, not DeltaT
    # itself. With e_n = DeltaT_n - DeltaT_ss the map is exactly
    # e_{n+1} = (1 - dt/tau)*e_n, so e alternates in sign and grows by |1.1| per
    # step; DeltaT_n therefore oscillates AROUND DeltaT_ss with a growing
    # amplitude rather than growing monotonically.
    dev = euler - ss
    amplification = abs(1.0 - dt_over_tau)
    growth = np.abs(dev[-1]) / ss
    assert growth > 1.0e6, (
        f"Euler at dt/tau = {dt_over_tau} should diverge (amplification "
        f"|1 - dt/tau| = {amplification:.2f} per step); "
        f"|DeltaT_final - DeltaT_ss|/DeltaT_ss = {growth:.3e}"
    )
    assert np.all(np.abs(dev[1:]) > np.abs(dev[:-1])), (
        "the Euler deviation from DeltaT_ss must grow monotonically"
    )
    np.testing.assert_allclose(
        np.abs(dev[1:] / dev[:-1]), amplification, rtol=1e-10, atol=0.0
    )
    assert np.all(np.sign(dev[1:]) != np.sign(dev[:-1])), (
        "the diverging Euler mode at dt/tau > 2 alternates in sign"
    )

    # Exponential: bounded by the steady state, and still exact.
    assert np.all(exponential >= 0.0)
    assert np.all(exponential <= ss * (1.0 + 1e-12))
    n = np.arange(1, n_steps + 1, dtype=np.float64)
    np.testing.assert_allclose(
        exponential, ss * (-np.expm1(-n * dt_over_tau)), rtol=1e-12, atol=0.0
    )

    # Sanity on the stability boundary itself: dt/tau = 2 is marginal for Euler
    # (|1 - 2| = 1 exactly), so it oscillates forever without growing.
    marginal = _integrate(thermal, p_abs, 2.0 * tau, 20, exponential=False)
    assert np.max(np.abs(marginal)) <= 2.0 * ss * (1.0 + 1e-12)


# ---------------------------------------------------------------------------
# 3. The two schemes agree as dt/tau -> 0
# ---------------------------------------------------------------------------

def test_both_agree_in_small_dt_limit():
    """The schemes converge to each other as dt/tau -> 0, at the predicted rate.

    NOTE ON THE TOLERANCE. The task asked for rtol 1e-6 at dt/tau = 1e-4; that
    is arithmetically unreachable, and the shortfall is exactly a factor 50.
    With x = dt/tau and DeltaT(0) = 0 the two trajectories are

        Euler:       DeltaT_ss * (1 - (1-x)^n)
        exponential: DeltaT_ss * (1 - e^(-nx))

    and since (1-x)^n = e^(-nx) * e^(-n*x^2/2) * (1 + O(n*x^3)), their relative
    difference is

        |DeltaT_E - DeltaT_X| / DeltaT_X = (x/2) * u*e^(-u)/(1 - e^(-u)),
        u = n*x,

    whose supremum over n is x/2, attained as u -> 0 and independent of how far
    the trajectory is integrated. At x = 1e-4 that floor is 5e-5, so no choice
    of n or normalization brings a first-order scheme within 1e-6 of an exact
    one; rtol 1e-6 requires x < 2e-6 (x = 2e-6 sits exactly ON the tolerance).

    This test therefore keeps x = 1e-4 and asserts something STRICTER than a
    loose bound -- that the measured difference matches the predicted x/2 to
    within 5%, which pins the coefficients of the exponential update (a wrong
    factor of 2, a tau vs 2*tau slip, or 1-exp in place of -expm1 all move it) --
    and then repeats the comparison at x = 1e-6, where the literal rtol 1e-6
    does hold and is asserted.
    """
    thermal = _thermal_dict()
    tau = float(thermal["tau_th"])
    p_abs = 5.0e-3
    ss = _steady_state(thermal, p_abs)

    # --- x = 1e-4: agreement at the rate the truncation analysis predicts ----
    x = 1.0e-4
    n_steps = 20_000                                  # u = n*x up to 2.0
    euler = _integrate(thermal, p_abs, x * tau, n_steps, exponential=False)
    exponential = _integrate(thermal, p_abs, x * tau, n_steps, exponential=True)

    rel = np.abs(euler - exponential) / np.abs(exponential)
    assert np.max(rel) < 1.0e-4, np.max(rel)

    u = np.arange(1, n_steps + 1, dtype=np.float64) * x
    predicted = (x / 2.0) * u * np.exp(-u) / (-np.expm1(-u))
    np.testing.assert_allclose(rel, predicted, rtol=0.05, atol=1e-18)

    # The supremum of the prediction is x/2 as u -> 0; confirm the measured
    # difference actually sits there rather than merely being bounded by it.
    assert rel[0] == pytest.approx(x / 2.0, rel=0.05), (rel[0], x / 2.0)

    # Both must still converge to the SAME steady state (there the schemes are
    # identical: DeltaT_ss is the fixed point of both maps).
    assert euler[-1] == pytest.approx(ss * (-np.expm1(-2.0)), rel=1e-4)
    assert exponential[-1] == pytest.approx(ss * (-np.expm1(-2.0)), rel=1e-12)

    # --- x = 1e-6: the regime where rtol 1e-6 is achievable -----------------
    # (the separation floor is x/2 = 5e-7, half the tolerance)
    x_small = 1.0e-6
    n_small = 2000
    euler_s = _integrate(thermal, p_abs, x_small * tau, n_small,
                         exponential=False)
    exponential_s = _integrate(thermal, p_abs, x_small * tau, n_small,
                               exponential=True)
    np.testing.assert_allclose(euler_s, exponential_s, rtol=1.0e-6, atol=0.0)


def test_small_dt_agreement_covers_the_production_ratio():
    """At the production dt_fine/tau_th the two integrators are interchangeable.

    t_r/tau_th = 8.13e-6 on the committed config, so the predicted relative
    separation is 4.1e-6: switching integrator cannot move a production
    trajectory by more than a few parts per million through the thermal step
    alone. Worth pinning, because it is the claim that makes "euler" a safe
    default rather than merely a legacy one.
    """
    physical = _load_config(None)
    thermal = _thermal_dict()
    tau = float(thermal["tau_th"])
    t_r = 1.0 / float(physical["fsr_hz"])
    x = t_r / tau
    assert x == pytest.approx(8.13e-6, rel=1e-2), x

    p_abs = 5.0e-3
    n_steps = 5000
    euler = _integrate(thermal, p_abs, t_r, n_steps, exponential=False)
    exponential = _integrate(thermal, p_abs, t_r, n_steps, exponential=True)
    rel = np.max(np.abs(euler - exponential) / np.abs(exponential))
    assert rel < 1.05 * (x / 2.0), (rel, x / 2.0)
    assert rel > 0.0, "the two schemes must not be accidentally identical"
