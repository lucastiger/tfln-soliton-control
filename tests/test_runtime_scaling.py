"""Unit tests for the runtime-scaling benchmark (``benchmarks/runtime_scaling.py``).

Covers the parts that decide whether a timing NUMBER is trustworthy, without
running the solver: the memory projection behind the skip guardrail, the
scaling fit and its overhead diagnostics, the noise-overhead pairing, and the
backend probe. The end-to-end run is exercised by
``python benchmarks/runtime_scaling.py --quick``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from benchmarks.runtime_scaling import (
    N_NOISE_SEQUENCES,
    N_TAU_GRID,
    N_TRAJ_GRID,
    NOISE_SETS,
    T_SLOW_GRID,
    BenchmarkError,
    _clean,
    available_devices,
    cpu_model,
    fit_scaling_exponent,
    noise_overhead_pct,
    project_memory_bytes,
)


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
def test_grids_are_the_specified_ones():
    assert N_TAU_GRID == (256, 512, 1024, 2048, 4096, 8192)
    assert T_SLOW_GRID == (1_000, 10_000, 100_000)
    assert N_TRAJ_GRID == (1, 8, 64)
    assert NOISE_SETS == ("all_off", "all_on")


def test_quick_grid_is_the_specified_subset():
    from benchmarks.runtime_scaling import (
        QUICK_N_TAU, QUICK_N_TRAJ, QUICK_T_SLOW,
    )

    assert QUICK_N_TAU == (256, 512)
    assert QUICK_T_SLOW == (1_000,)
    assert QUICK_N_TRAJ == (1,)


# ---------------------------------------------------------------------------
# Memory projection
# ---------------------------------------------------------------------------
def test_projection_counts_the_host_copy_as_well_as_the_device_result():
    """np.asarray materializes a second full copy; both live at once."""
    m = project_memory_bytes(512, 1000, 1, noise_on=False)
    assert m["result_host"] == m["result_device"]
    assert m["total"] >= m["result_device"] + m["result_host"]


def test_projection_scales_linearly_in_n_traj():
    a = project_memory_bytes(512, 1000, 1, noise_on=False)["total"]
    b = project_memory_bytes(512, 1000, 8, noise_on=False)["total"]
    assert b == pytest.approx(8 * a, rel=1e-9)


def test_projection_histories_dominate_at_long_records():
    """(n_traj, t_slow) round-trip histories, not the single snapshot."""
    m = project_memory_bytes(256, 100_000, 64, noise_on=False)
    assert m["result_device"] > 10 * (64 * 1 * 256 * 16)


def test_noise_adds_sequences_and_noise_off_does_not():
    off = project_memory_bytes(512, 1000, 4, noise_on=False)
    on = project_memory_bytes(512, 1000, 4, noise_on=True)
    assert off["noise"] == 0
    assert on["noise"] == 4 * 1000 * 8 * N_NOISE_SEQUENCES
    assert on["total"] > off["total"]


def test_projection_reports_an_auditable_formula():
    """A guardrail whose arithmetic cannot be inspected is not a guardrail."""
    m = project_memory_bytes(512, 1000, 2, noise_on=True)
    assert "working" in m["formula"] and "histories" in m["formula"]
    assert m["total_gb"] == pytest.approx(m["total"] / 1024 ** 3)


def test_largest_grid_point_fits_under_the_default_budget():
    """The specified memory budget does NOT skip the biggest point.

    Recorded deliberately: runtime, not memory, is what makes the full grid
    expensive, so a reader must not assume --max-mem-gb protects them from it.
    """
    m = project_memory_bytes(8192, 100_000, 64, noise_on=True)
    assert m["total_gb"] < 8.0


def test_projection_is_monotonic_in_every_axis():
    base = project_memory_bytes(512, 1000, 4, noise_on=False)["total"]
    assert project_memory_bytes(1024, 1000, 4, noise_on=False)["total"] > base
    assert project_memory_bytes(512, 2000, 4, noise_on=False)["total"] > base
    assert project_memory_bytes(512, 1000, 5, noise_on=False)["total"] > base


# ---------------------------------------------------------------------------
# Scaling fit
# ---------------------------------------------------------------------------
def test_pure_power_law_recovers_its_exponent():
    n = np.array([256, 512, 1024, 2048, 4096, 8192], dtype=float)
    fit = fit_scaling_exponent(n, 1e-6 * n ** 1.0)
    assert fit["alpha_total"] == pytest.approx(1.0, abs=1e-6)
    assert fit["alpha_per_mode"] == pytest.approx(0.0, abs=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_n_log_n_data_gives_exponent_slightly_above_one():
    """The '1 + log-correction' the FFT predicts."""
    n = np.array([256, 512, 1024, 2048, 4096, 8192], dtype=float)
    fit = fit_scaling_exponent(n, 1e-9 * n * np.log2(n))
    assert 1.0 < fit["alpha_total"] < 1.25
    assert fit["n_log_n_r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert fit["predicted_alpha_n_log_n"] == pytest.approx(
        fit["alpha_total"], abs=0.02)


def test_fixed_overhead_is_recovered_and_flagged():
    """A per-call floor drags the plain slope down; the fit must say so."""
    n = np.array([256, 512, 1024, 2048, 4096, 8192], dtype=float)
    overhead = 0.12
    fit = fit_scaling_exponent(n, overhead + 1e-9 * n * np.log2(n))
    assert fit["fixed_overhead_s"] == pytest.approx(overhead, rel=1e-3)
    assert fit["overhead_frac_at_min_n_tau"] > 0.3
    assert "UNDERSTATES" in fit["note"]
    # The plain slope is dragged below the prediction...
    assert fit["alpha_total"] < fit["predicted_alpha_n_log_n"]
    # ...while the compute-bound half recovers it.
    assert fit["alpha_total_compute_bound"] > fit["alpha_total"]


def test_compute_bound_slope_uses_the_upper_half_of_the_grid():
    n = np.array([256, 512, 1024, 2048, 4096, 8192], dtype=float)
    fit = fit_scaling_exponent(n, 1e-9 * n * np.log2(n))
    assert fit["compute_bound_n_tau"] == [2048, 4096, 8192]


def test_two_point_fit_is_labelled_as_a_slope_not_a_fit():
    fit = fit_scaling_exponent([256, 512], [0.1, 0.2])
    assert fit["n_points"] == 2
    assert "two n_tau points only" in fit["note"]
    assert fit["alpha_total"] == pytest.approx(1.0, abs=1e-9)
    # Two points cannot support the 2-parameter affine model.
    assert fit["fixed_overhead_s"] is None


def test_single_point_yields_no_slope():
    fit = fit_scaling_exponent([256], [0.1])
    assert fit["alpha_total"] is None
    assert "fewer than 2" in fit["note"]


def test_fit_ignores_non_finite_and_non_positive_samples():
    n = [256, 512, 1024, 2048]
    t = [0.1, float("nan"), 0.0, 0.4]
    fit = fit_scaling_exponent(n, t)
    assert fit["n_points"] == 2


# ---------------------------------------------------------------------------
# Noise overhead
# ---------------------------------------------------------------------------
def _pt(noise, wall, n_tau=512, t_slow=1000, n_traj=1, device="cpu"):
    return {"status": "ok", "device": device, "n_tau": n_tau, "t_slow": t_slow,
            "n_traj": n_traj, "noise": noise, "wall_s": wall}


def test_noise_overhead_is_a_matched_percentage():
    ov = noise_overhead_pct([_pt("all_off", 0.10), _pt("all_on", 0.25)])
    assert ov["n_pairs"] == 1
    assert ov["median_pct"] == pytest.approx(150.0)


def test_noise_overhead_never_pairs_across_problem_sizes():
    """An unmatched pair would compare two different amounts of work."""
    ov = noise_overhead_pct([_pt("all_off", 0.10, n_tau=256),
                             _pt("all_on", 0.25, n_tau=512)])
    assert ov["n_pairs"] == 0
    assert ov["median_pct"] is None


def test_noise_overhead_skips_incomplete_and_failed_points():
    pts = [_pt("all_off", 0.10), _pt("all_on", 0.25),
           {"status": "failed", "device": "cpu", "n_tau": 1024,
            "t_slow": 1000, "n_traj": 1, "noise": "all_off"},
           _pt("all_off", 0.2, n_tau=2048)]  # no all_on partner
    ov = noise_overhead_pct(pts)
    assert ov["n_pairs"] == 1


def test_noise_overhead_reports_range_over_pairs():
    ov = noise_overhead_pct([
        _pt("all_off", 0.10, n_tau=256), _pt("all_on", 0.20, n_tau=256),
        _pt("all_off", 0.10, n_tau=512), _pt("all_on", 0.40, n_tau=512),
    ])
    assert ov["n_pairs"] == 2
    assert ov["min_pct"] == pytest.approx(100.0)
    assert ov["max_pct"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Environment probing
# ---------------------------------------------------------------------------
def test_cpu_model_is_a_non_empty_string():
    """The table caption has to name the exact part it was measured on."""
    assert isinstance(cpu_model(), str) and cpu_model().strip()


def test_available_devices_always_finds_cpu():
    devices = available_devices(["cpu"])
    assert devices and devices[0].platform == "cpu"


def test_missing_backend_is_skipped_not_fatal():
    """jax.devices('gpu') RAISES when no GPU backend is linked in."""
    devices = available_devices(["cpu", "gpu", "tpu"])
    assert any(d.platform == "cpu" for d in devices)


def test_no_backend_at_all_is_an_error():
    with pytest.raises(BenchmarkError, match="no JAX backend"):
        available_devices([])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def test_clean_makes_non_finite_json_safe():
    payload = _clean({"a": float("nan"), "b": float("inf"),
                      "c": [1.0, float("-inf")], "d": np.float64(2.5)})
    assert payload == {"a": None, "b": None, "c": [1.0, None], "d": 2.5}
    assert json.loads(json.dumps(payload))["a"] is None
