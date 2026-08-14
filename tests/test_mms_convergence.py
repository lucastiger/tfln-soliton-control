"""Tests for :mod:`validation.mms` and :mod:`validation.convergence`.

Four groups:

* **Bit-identity** — the three new solver options, left at their defaults, must
  route to the untouched module-level jit objects and produce byte-identical
  output. This is the guard that lets the order fixes ship default-off.
* **Pure math** — ``observed_order`` against synthetic power laws.
* **MMS correctness** — the symbolically derived forcing is cross-checked
  against an INDEPENDENT spectral evaluation of the same residual, which is
  what catches a dispersion sign-convention error. Without this the whole study
  could converge cleanly to the wrong PDE.
* **Orders** — the measured orders, pinned both for the production path (1) and
  the fixed path (2), so a change to either scheme is caught.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("sympy", reason="validation.mms derives its forcing with sympy")

import jax                                                             # noqa: E402

from simulator.lle_solver import (                                     # noqa: E402
    _PER_TRAJ,
    _PER_TRAJ_THERMAL_EXP,
    _load_config,
    _per_traj_variant,
    solve_lle_ssfm_jax,
)
from simulator.noise_config import NoiseConfig                         # noqa: E402
from validation.convergence import (                                   # noqa: E402
    observed_order,
    thermal_testbed_config,
)
from validation.mms import (                                           # noqa: E402
    make_source_fn,
    manufactured_solution,
    manufactured_source,
    mms_error,
    periodicity_defect,
    residual_check,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_CONFIG = _REPO_ROOT / "config" / "sin_params.yaml"


@pytest.fixture(scope="module")
def sol():
    return manufactured_solution()


def _solve(sol, n_tau=32, t_slow=20, config_path=None, **kw):
    return np.asarray(
        solve_lle_ssfm_jax(
            pin=sol.pin, delta_omega=sol.delta_omega, t_slow=t_slow,
            beta=list(sol.beta), kappa=sol.params.kappa, kappa_c=sol.params.kappa_c,
            rng_key=jax.random.PRNGKey(0), n_tau=n_tau,
            config_path=config_path or sol.params.config_path, snapshot_interval=t_slow,
            e0_override=sol.field(sol.tau_grid(n_tau), 0.0),
            noise_config=NoiseConfig.all_off(),
            **kw,
        )["e_final"]
    )


# ---------------------------------------------------------------------------
# Bit-identity of the default path
# ---------------------------------------------------------------------------
def test_defaults_route_to_the_untouched_jit_objects():
    """The legacy objects keep their identity, so their 32-positional public
    surface and compilation cache are unaffected."""
    assert _per_traj_variant(False, None, False, False) is not _PER_TRAJ
    assert _per_traj_variant(True, None, False, False) is not _PER_TRAJ_THERMAL_EXP
    # ...and the variant builder caches, so a repeated request is the same object.
    assert _per_traj_variant(False, None, True, False) is _per_traj_variant(
        False, None, True, False
    )


def test_explicit_defaults_are_bit_identical_to_omitting_them(sol):
    base = _solve(sol)
    explicit = _solve(sol, source_fn=None, symmetric_drive=False,
                      thermal_coupling="lagged")
    assert np.array_equal(base, explicit)
    assert base.tobytes() == explicit.tobytes()


def test_zero_source_is_bit_identical_to_no_source(sol):
    """A source that returns exactly zero must not perturb a single bit.

    This pins that the forcing is added in the right place: ``x + 0.0 == x``
    exactly in IEEE754, so any bit difference here would mean the hook changed
    the order of operations around it rather than just adding a term.
    """
    import jax.numpy as jnp

    n_tau = 32
    zero = jnp.zeros(n_tau, dtype=jnp.complex128)

    def zero_source(t):
        return zero

    base = _solve(sol, n_tau=n_tau)
    with_zero = _solve(sol, n_tau=n_tau, source_fn=zero_source)
    assert base.tobytes() == with_zero.tobytes()


def test_symmetric_drive_flag_changes_the_result(sol):
    """Guard against a flag that is plumbed but never read."""
    assert not np.array_equal(_solve(sol), _solve(sol, symmetric_drive=True))


def test_thermal_strang_changes_the_result_only_when_thermal_is_LIVE(sol, tmp_path):
    """thermal_coupling is a no-op when dn_dT = 0, and that is correct.

    On the thermo-optic-free config the shift is identically zero, so
    symmetrizing the coupling cannot change the field — asserting otherwise
    would be asserting a bug. The flag must bite on the testbed config, where
    the thermal term actually participates.
    """
    off = dict(config_path=sol.params.config_path)
    assert np.array_equal(
        _solve(sol, **off), _solve(sol, thermal_coupling="strang", **off)
    )

    testbed = dict(
        config_path=thermal_testbed_config(
            sol.params, pin=sol.pin, delta_omega=sol.delta_omega, out_dir=tmp_path
        )
    )
    assert not np.array_equal(
        _solve(sol, **testbed), _solve(sol, thermal_coupling="strang", **testbed)
    )


def test_invalid_thermal_coupling_rejected(sol):
    with pytest.raises(ValueError, match="thermal_coupling"):
        _solve(sol, thermal_coupling="leapfrog")


# ---------------------------------------------------------------------------
# observed_order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1.0, 2.0, 3.5])
def test_observed_order_recovers_a_synthetic_power_law(p):
    dts = np.array([1.0, 0.5, 0.25, 0.125])
    errors = 3.7 * dts ** p
    assert np.allclose(observed_order(errors, dts), p, atol=1e-12)


def test_observed_order_handles_a_zero_error():
    dts = np.array([1.0, 0.5, 0.25])
    errors = np.array([1.0, 0.0, 0.25])
    out = observed_order(errors, dts)
    assert np.isnan(out).all()


def test_observed_order_shape_and_validation():
    with pytest.raises(ValueError):
        observed_order([1.0, 0.5], [1.0])
    with pytest.raises(ValueError):
        observed_order([1.0], [1.0])


# ---------------------------------------------------------------------------
# MMS correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t_eval", [0.0, 1e-9, 4.2e-9])
def test_symbolic_source_matches_an_independent_spectral_derivation(sol, t_eval):
    """The test that makes the whole MMS trustworthy.

    :func:`manufactured_source` differentiates in tau symbolically and assumes
    ``omega -> -i d/dtau``; :func:`residual_check` instead applies
    ``disp(omega)`` in Fourier space exactly as ``_fine_step`` does. A sign or
    factor error in either would show up here at O(1).
    """
    assert residual_check(sol, t_eval=t_eval) < 1e-13


def test_manufactured_solution_is_periodic_and_resolved(sol):
    d = periodicity_defect(sol, n_tau=128)
    # Both defects must sit far below the ~1e-8 errors the studies resolve.
    assert d["envelope_at_edge"] < 1e-20
    assert d["spectral_tail"] < 1e-40
    assert d["samples_per_width"] >= 4.0


def test_phase_factor_is_exactly_periodic(sol):
    """k must be an integer multiple of 2*pi/t_r for exact periodicity."""
    turns = sol.k * sol.params.t_r / (2.0 * math.pi)
    assert turns == pytest.approx(round(turns), abs=1e-12)
    assert round(turns) == sol.mode_number


def test_source_is_finite_and_separable(sol):
    psi, drive = manufactured_source(sol, 64)
    assert psi.shape == (64,)
    assert np.all(np.isfinite(psi))
    assert drive == pytest.approx(math.sqrt(sol.params.kappa_c * sol.pin), rel=1e-12)


def test_source_fn_object_is_cached(sol):
    """The solver caches its jit on source_fn identity; a fresh closure per call
    would recompile at every dt of the study."""
    assert make_source_fn(sol, 64) is make_source_fn(sol, 64)


def test_source_fn_reproduces_psi_at_t_zero(sol):
    import jax.numpy as jnp

    psi, drive = manufactured_source(sol, 64)
    got = np.asarray(make_source_fn(sol, 64)(jnp.float64(0.0)))
    assert np.allclose(got, psi - drive, rtol=1e-14, atol=0.0)


# ---------------------------------------------------------------------------
# Thermal testbed config
# ---------------------------------------------------------------------------
def test_thermal_testbed_changes_exactly_two_leaves(sol, tmp_path):
    before = _SOURCE_CONFIG.read_bytes()
    derived = thermal_testbed_config(
        sol.params, pin=sol.pin, delta_omega=sol.delta_omega, out_dir=tmp_path
    )
    assert _SOURCE_CONFIG.read_bytes() == before, "committed config was modified"
    original = _load_config(_SOURCE_CONFIG)
    reloaded = _load_config(derived)
    changed = sorted(k for k in original if original[k] != reloaded[k])
    assert changed == ["mode_volume_m3", "tau_th_s"]
    assert reloaded["dn_dT_per_k"] == original["dn_dT_per_k"], (
        "the thermo-optic pathway must stay live in the testbed"
    )


def test_thermal_testbed_satisfies_the_solver_preflight_guard(sol, tmp_path):
    """The calibration targets delta_omega = 0, which is what the guard tests."""
    derived = thermal_testbed_config(
        sol.params, pin=sol.pin, delta_omega=sol.delta_omega, out_dir=tmp_path
    )
    phys = _load_config(derived)
    p = sol.params
    omega0 = 2.0 * math.pi * 299_792_458.0 / phys["pump_wavelength_m"]
    r_th = phys["tau_th_s"] * phys["Gamma_th"] / (
        phys["rho_kg_per_m3"] * phys["Cp_j_per_kg_k"] * phys["mode_volume_m3"]
    )
    to_coeff = (omega0 / phys["n0"]) * phys["dn_dT_per_k"]
    u_cw = p.kappa_c * sol.pin / ((p.kappa / 2.0) ** 2)
    worst = to_coeff * r_th * (p.kappa_i * u_cw)
    limit = phys.get("thermal_shift_max_over_kappa", 15.0)
    assert worst / p.kappa < limit
    assert worst / p.kappa > 1.0, "testbed shift is too weak to measure anything"


def test_thermal_testbed_leaves_stay_numeric(sol, tmp_path):
    from tests.test_config import NON_NUMERIC_ALLOWLIST

    derived = thermal_testbed_config(
        sol.params, pin=sol.pin, delta_omega=sol.delta_omega, out_dir=tmp_path
    )
    for key, value in _load_config(derived).items():
        if key in NON_NUMERIC_ALLOWLIST:
            assert NON_NUMERIC_ALLOWLIST[key](value)
            continue
        assert isinstance(value, (int, float)) and not isinstance(value, bool)


def test_thermal_testbed_preserves_the_noise_block(sol, tmp_path):
    derived = thermal_testbed_config(
        sol.params, pin=sol.pin, delta_omega=sol.delta_omega, out_dir=tmp_path
    )
    src = yaml.safe_load(_SOURCE_CONFIG.read_text(encoding="utf-8"))
    got = yaml.safe_load(Path(derived).read_text(encoding="utf-8"))
    assert got.get("noise") == src.get("noise")


# ---------------------------------------------------------------------------
# Measured orders
# ---------------------------------------------------------------------------
def _mms_orders(sol, symmetric_drive, ms=(1, 2, 4, 8), t_slow=100):
    dts = np.array([sol.params.t_r / m for m in ms])
    errors = np.array([
        mms_error(float(dt), 128, sol=sol, t_final_round_trips=t_slow,
                  symmetric_drive=symmetric_drive)
        for dt in dts
    ])
    return observed_order(errors, dts), errors


def test_production_scheme_is_first_order(sol):
    """The FINDING, pinned. The shipping scheme is first order because the
    drive kick sits outside the linear half-step."""
    orders, errors = _mms_orders(sol, False)
    assert np.all(np.abs(orders - 1.0) < 0.05), f"orders {orders}, errors {errors}"


def test_symmetric_drive_restores_second_order(sol):
    """...and splitting the drive fixes it exactly."""
    orders, errors = _mms_orders(sol, True)
    assert np.all(np.abs(orders - 2.0) < 0.05), f"orders {orders}, errors {errors}"


def test_symmetric_drive_is_substantially_more_accurate(sol):
    """Not just a better slope — a much smaller error at the SAME dt."""
    kw = dict(sol=sol, t_final_round_trips=100)
    prod = mms_error(sol.params.t_r, 128, symmetric_drive=False, **kw)
    fixed = mms_error(sol.params.t_r, 128, symmetric_drive=True, **kw)
    assert fixed < prod / 100.0, f"production {prod:.3e} vs fixed {fixed:.3e}"


@pytest.mark.slow
def test_report_gates_pass_on_the_fixed_scheme():
    from validation.convergence import main

    assert main(["--report", "--halvings", "4", "--t-slow", "150",
                 "--weak-realizations", "64"]) == 0
