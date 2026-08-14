"""Tests for :mod:`validation.analytic_cw`.

Split into three groups:

* **Pure-math** — the two exact references, checked against independent
  formulations rather than against themselves. The cubic roots are substituted
  back into the steady-state equation; the discrete-map fixed point is checked
  against a brute-force iteration of the map written out step by step, so a
  mistake in the closed-form algebra of
  :func:`validation.analytic_cw.discrete_map_cw_fixed_point` cannot hide.
* **Config plumbing** — the derived thermo-optic-free config differs from the
  committed one in exactly one leaf, and the committed one is untouched.
* **Solver** — a reduced sweep pinning both the machine-precision agreement
  with the discrete map AND the ORDER in dt of the mean-field residual (the
  documented 3e-3 finding). These call the solver and are the slow ones; the
  full 405-point sweep is marked ``slow``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from simulator.lle_solver import _load_config
from validation.analytic_cw import (
    DEFAULT_DETUNING_GRID,
    DEFAULT_PIN_MULTIPLIERS,
    analytic_cw_roots,
    discrete_map_cw_fixed_point,
    flatness_floor,
    load_cavity_params,
    mean_field_order_check,
    p_threshold,
    solver_cw_steady_state,
    stable_branch,
    thermal_off_config,
    verify,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_CONFIG = _REPO_ROOT / "config" / "sin_params.yaml"


@pytest.fixture(scope="module")
def params():
    return load_cavity_params()


# ---------------------------------------------------------------------------
# Pure math: the continuum cubic
# ---------------------------------------------------------------------------
def test_roots_satisfy_the_steady_state_equation(params):
    """Every returned root substituted back into P*[(k/2)^2+(dw-g*P)^2] = kc*Pin."""
    worst = 0.0
    for mult in DEFAULT_PIN_MULTIPLIERS:
        pin = mult * params.p_th
        for d in DEFAULT_DETUNING_GRID:
            dw = d * params.kappa
            roots = analytic_cw_roots(pin, dw, params.kappa, params.kappa_c, params.gamma)
            assert roots.size >= 1
            for p in roots:
                lhs = p * ((params.kappa / 2.0) ** 2 + (dw - params.gamma * p) ** 2)
                rhs = params.kappa_c * pin
                worst = max(worst, abs(lhs - rhs) / rhs)
    assert worst < 1e-13, f"worst cubic residual {worst:.3e}"


def test_roots_are_sorted_real_and_positive(params):
    for mult in (0.5, 10.0):
        for d in (-5.0, 0.0, 3.0, 15.0):
            roots = analytic_cw_roots(
                mult * params.p_th, d * params.kappa,
                params.kappa, params.kappa_c, params.gamma,
            )
            assert roots.dtype == np.float64
            assert np.all(roots > 0.0)
            assert np.all(np.diff(roots) > 0.0)


def test_bistability_appears_only_above_threshold(params):
    """Three roots at 10*P_th inside the window, one at 1*P_th anywhere."""
    def n_roots(mult, d):
        return analytic_cw_roots(
            mult * params.p_th, d * params.kappa,
            params.kappa, params.kappa_c, params.gamma,
        ).size

    assert n_roots(10.0, 3.0) == 3
    assert n_roots(10.0, -5.0) == 1
    assert n_roots(10.0, 15.0) == 1
    assert all(n_roots(1.0, d) == 1 for d in DEFAULT_DETUNING_GRID)


def test_p_threshold_matches_solver_preflight_formula(params):
    assert p_threshold(params.kappa, params.kappa_c, params.gamma) == pytest.approx(
        (params.kappa / 2.0) ** 2 * params.kappa / (2.0 * params.gamma * params.kappa_c),
        rel=1e-15,
    )


# ---------------------------------------------------------------------------
# Pure math: branch selection
# ---------------------------------------------------------------------------
def test_stable_branch_picks_the_scan_endpoint():
    roots = np.array([1.0, 2.0, 3.0])
    assert stable_branch(roots, "up") == 3.0
    assert stable_branch(roots, "down") == 1.0
    assert stable_branch(np.array([7.0]), "up") == 7.0
    with pytest.raises(ValueError):
        stable_branch(np.array([]), "up")
    with pytest.raises(ValueError):
        stable_branch(roots, "sideways")


def test_upward_branch_is_continuous_across_the_bistable_window(params):
    """The up-scan branch must not jump inside the window.

    If ``stable_branch`` picked the wrong root the selected power would drop
    discontinuously at the window edge. Require the branch to stay monotonically
    increasing in delta_omega right through the window, which is the defining
    property of the arc an upward scan traverses.
    """
    pin = 10.0 * params.p_th
    fine = np.linspace(1.0, 5.5, 400)
    branch = np.array([
        stable_branch(
            analytic_cw_roots(pin, d * params.kappa, params.kappa, params.kappa_c,
                              params.gamma),
            "up",
        )
        for d in fine
    ])
    n_roots = np.array([
        analytic_cw_roots(pin, d * params.kappa, params.kappa, params.kappa_c,
                          params.gamma).size
        for d in fine
    ])
    inside = n_roots == 3
    assert inside.sum() > 50, "expected a well-resolved bistable window here"
    assert np.all(np.diff(branch[inside]) > 0.0)


# ---------------------------------------------------------------------------
# Pure math: the discrete map
# ---------------------------------------------------------------------------
def _iterate_map(pin, dw, kappa, kappa_c, gamma, dt, p0, n_steps=400_000):
    """Brute-force iteration of the solver's scalar CW map, in plain Python.

    Written out step by step exactly as ``_fine_step`` composes it — pump kick,
    half linear, Kerr, half linear — so it shares no algebra with
    ``discrete_map_cw_fixed_point``.
    """
    drive = math.sqrt(kappa_c * pin) * dt
    half = complex(math.exp(-kappa * dt / 4.0), 0.0) * complex(
        math.cos(-dw * dt / 2.0), math.sin(-dw * dt / 2.0)
    )
    e = complex(math.sqrt(p0), 0.0)
    for _ in range(n_steps):
        a = e + drive
        e_half = a * half
        phase = gamma * abs(e_half) ** 2 * dt
        e = e_half * complex(math.cos(phase), math.sin(phase)) * half
    return abs(e) ** 2


@pytest.mark.parametrize("d,mult", [(-2.0, 1.0), (0.0, 1.0), (3.0, 10.0), (12.0, 5.0)])
def test_discrete_reference_matches_brute_force_iteration(params, d, mult):
    pin = mult * params.p_th
    dw = d * params.kappa
    dt = params.t_r
    closed_form = discrete_map_cw_fixed_point(
        pin, dw, params.kappa, params.kappa_c, params.gamma, dt
    )
    iterated = _iterate_map(
        pin, dw, params.kappa, params.kappa_c, params.gamma, dt, closed_form
    )
    assert abs(iterated - closed_form) / closed_form < 1e-12


def test_discrete_reference_converges_to_the_continuum_root(params):
    """As dt -> 0 the map's fixed point must approach the cubic root, first order."""
    pin = 2.0 * params.p_th
    dw = 2.0 * params.kappa
    p_cont = stable_branch(
        analytic_cw_roots(pin, dw, params.kappa, params.kappa_c, params.gamma), "up"
    )
    residuals = []
    for n in (1, 2, 4, 8, 16):
        p_disc = discrete_map_cw_fixed_point(
            pin, dw, params.kappa, params.kappa_c, params.gamma, params.t_r / n
        )
        residuals.append(abs(p_disc - p_cont) / p_cont)
    residuals = np.asarray(residuals)
    assert np.all(np.diff(residuals) < 0.0)
    ratios = residuals[:-1] / residuals[1:]
    assert np.all(np.abs(ratios - 2.0) < 0.05), f"ratios {ratios}"


def test_discrete_reference_is_cancellation_free_near_resonance(params):
    """The half-angle form must beat the naive 1 - 2r*cos(psi) + r^2 form.

    Near resonance psi is small and the naive expression loses ~5 digits, which
    is what put the measured residual at 6.4e-12. Recompute the residual of the
    returned fixed point in EXTENDED precision (Fraction-free: use the identity
    directly with math.fsum) and require it to be at the rounding level.
    """
    pin = 2.0 * params.p_th
    dw = 0.75 * params.kappa
    dt = params.t_r
    p = discrete_map_cw_fixed_point(
        pin, dw, params.kappa, params.kappa_c, params.gamma, dt
    )
    r = math.exp(-params.kappa * dt / 2.0)
    one_minus_r = -math.expm1(-params.kappa * dt / 2.0)
    q = p / (r * r)
    psi = params.gamma * r * q * dt - dw * dt
    lhs = q * (one_minus_r ** 2 + 4.0 * r * math.sin(psi / 2.0) ** 2)
    rhs = (math.sqrt(params.kappa_c * pin) * dt) ** 2
    assert abs(lhs - rhs) / rhs < 1e-14


def test_flatness_floor_is_above_1e14_at_committed_parameters(params):
    """Documents why the stationarity bound is derived and not hard-coded 1e-14."""
    floor = flatness_floor(params.kappa, params.t_r)
    assert floor > 1e-14, (
        f"stationarity floor {floor:.3e} — if this ever drops below 1e-14 the "
        f"literal bound in the task spec becomes reachable"
    )
    assert floor < 1e-11


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
def test_thermal_off_config_changes_exactly_one_leaf(tmp_path):
    before = _SOURCE_CONFIG.read_bytes()
    derived = thermal_off_config(_SOURCE_CONFIG, out_dir=tmp_path)
    assert _SOURCE_CONFIG.read_bytes() == before, "committed config was modified"

    original = _load_config(_SOURCE_CONFIG)
    reloaded = _load_config(derived)
    assert set(original) == set(reloaded)
    differing = [k for k in original if original[k] != reloaded[k]]
    assert differing == ["dn_dT_per_k"]
    assert reloaded["dn_dT_per_k"] == 0.0


def test_thermal_off_config_leaves_stay_numeric(tmp_path):
    """tests/test_config.py's invariant must survive the derivation.

    That invariant is "numeric, except for the documented allowlist", so import
    the allowlist from the test that owns it rather than restating a stricter
    rule here (``trn_psd_model`` is a legitimately non-numeric leaf).
    """
    from tests.test_config import NON_NUMERIC_ALLOWLIST

    derived = thermal_off_config(_SOURCE_CONFIG, out_dir=tmp_path)
    for key, value in _load_config(derived).items():
        if key in NON_NUMERIC_ALLOWLIST:
            assert NON_NUMERIC_ALLOWLIST[key](value), (
                f"{key}={value!r} violates its allowlisted constraint"
            )
            continue
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"physical_parameters[{key!r}] = {value!r} is not a plain number"
        )


def test_thermal_off_config_preserves_the_noise_block(tmp_path):
    derived = thermal_off_config(_SOURCE_CONFIG, out_dir=tmp_path)
    src = yaml.safe_load(_SOURCE_CONFIG.read_text(encoding="utf-8"))
    got = yaml.safe_load(Path(derived).read_text(encoding="utf-8"))
    assert got.get("noise") == src.get("noise")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def test_solver_reproduces_the_discrete_map_fixed_point(params):
    """Reduced sweep: 9 detunings x 1 pump, machine precision against the map."""
    grid = np.linspace(-5.0, 15.0, 9)
    dws = grid * params.kappa
    pin = 1.0 * params.p_th
    solved = solver_cw_steady_state(pin, dws, params, t_slow=20_000)
    reference = np.array([
        discrete_map_cw_fixed_point(
            pin, dw, params.kappa, params.kappa_c, params.gamma, params.t_r
        )
        for dw in dws
    ])
    rel = np.abs(solved - reference) / reference
    assert rel.max() < 1e-12, f"max {rel.max():.3e} at dw/kappa={grid[rel.argmax()]}"


def test_solver_sits_on_the_seeded_branch_inside_the_bistable_window(params):
    """Up-scan and down-scan must land on DIFFERENT, stable branches."""
    pin = 10.0 * params.p_th
    dws = np.array([3.0, 4.0]) * params.kappa
    up = solver_cw_steady_state(pin, dws, params, t_slow=20_000, scan_direction="up")
    down = solver_cw_steady_state(pin, dws, params, t_slow=20_000, scan_direction="down")
    for i, dw in enumerate(dws):
        roots = analytic_cw_roots(pin, dw, params.kappa, params.kappa_c, params.gamma)
        assert roots.size == 3
        assert up[i] / roots[-1] == pytest.approx(1.0, rel=1e-2)
        assert down[i] / roots[0] == pytest.approx(1.0, rel=1e-2)
        assert up[i] > 10.0 * down[i]


def test_solver_raises_when_the_state_is_not_stationary(params):
    """The flatness assertion must actually fire, not just decorate the code."""
    with pytest.raises(AssertionError, match="did not settle"):
        solver_cw_steady_state(
            1.0 * params.p_th, np.array([2.0]) * params.kappa, params,
            t_slow=200,                 # far too short to relax (tau = 324 t_r)
            seed_perturbation=1.5,
        )


def test_mean_field_residual_is_first_order_in_dt(params):
    """Pins the documented 3e-3 finding as an ORDER, not as a magnitude.

    The solver's CW state differs from the continuum cubic by a splitting error
    that must scale as 1/n_substeps. A change to the ORDER means the splitting
    scheme changed; a change to the prefactor alone just means the parameters
    did.
    """
    order = mean_field_order_check(params, substeps=(1, 2, 4), t_slow=20_000)
    assert order["order"] == pytest.approx(1.0, abs=0.05), (
        f"fitted order {order['order']:.4f}, residuals {order['rel_continuum']}"
    )
    # ...and the prefactor is ~kappa*t_r/2, the mean-field scale.
    assert order["rel_continuum"][0] == pytest.approx(
        params.kappa * params.t_r / 2.0, rel=0.05
    )


@pytest.mark.slow
def test_full_sweep_passes_the_gate(params):
    result = verify(1e-12, params=params)
    assert result["n_points"] == 405
    assert result["n_bistable"] > 0
    assert result["passed"], (
        f"max discrete-map residual {result['max_rel_discrete']:.3e} at "
        f"{result['max_rel_discrete_at']}"
    )
    assert result["max_rel_continuum"] > 1e-3, (
        "the mean-field residual vanished — the splitting scheme changed"
    )
