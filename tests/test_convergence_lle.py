"""Unit tests for the LLE discretization-uncertainty harness.

These test the ESTIMATORS, not the physics: each one pins down a property that
motivated redefining an observable in ``validation/convergence_lle.py``, so that
a future "simplification" back to the legacy definition fails here rather than
silently degrading the uncertainty budget.

No solver run is required; every test builds its own analytic field.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from validation.convergence_lle import (
    DW_BAND_DEFAULT,
    band_limited_peak,
    dw_centroid,
    line_count_diagnostics,
    richardson,
    spectrum_dbc,
    subbin_span_3db,
)

N_MODES = 6601
MU = np.arange(-(N_MODES - 1) // 2, (N_MODES - 1) // 2 + 1)
W_SOLITON = 2.4267e-3          # rad; the DW-operating-point soliton width
T_R = 4.089705051744962e-11    # s


def _sech(x):
    """Overflow-safe sech."""
    a = np.abs(x)
    return 2.0 * np.exp(-a) / (1.0 + np.exp(-2.0 * a))


def _theta(n=N_MODES):
    th = 2.0 * math.pi * np.arange(n) / n
    return (th + math.pi) % (2.0 * math.pi) - math.pi


def _fourier_shift(field, samples):
    """Circularly shift by a (possibly fractional) number of samples."""
    n = field.size
    k = np.fft.fftfreq(n) * n
    return np.fft.ifft(np.fft.fft(field) * np.exp(-2j * math.pi * k * samples / n))


# ---------------------------------------------------------------- Richardson
def test_richardson_recovers_known_order():
    """f(h) = 7 + 3h^2 must give p = 2 and q* = 7 to round-off."""
    hs = [1.0, 0.5, 0.25]
    vals = [7.0 + 3.0 * h ** 2 for h in hs]
    r = richardson(vals, hs)
    assert r["status"] == "OK"
    assert abs(r["observed_order"] - 2.0) < 1e-6
    assert abs(r["extrapolated"] - 7.0) < 1e-9


def test_richardson_flags_non_monotone():
    """A sequence that turns around cannot be Richardson-extrapolated."""
    r = richardson([1.0, 2.0, 1.5], [1.0, 0.5, 0.25])
    assert r["status"] == "NON_MONOTONE"
    assert r["extrapolated"] is None
    assert r["observed_order"] is None
    # a conservative band must still be produced
    assert r["U_obs"] is not None and r["U_obs"] > 0


def test_richardson_flags_order_out_of_range():
    """An absurd observed order is reported, not silently used."""
    r = richardson([1000.0, 10.0, 9.999], [1.0, 0.5, 0.25])
    assert r["status"] == "ORDER_OUT_OF_RANGE"
    assert r["observed_order"] is not None      # kept for inspection
    assert r["U_obs"] is not None


def test_richardson_insufficient_levels():
    r = richardson([1.0, 2.0], [1.0, 0.5])
    assert r["status"] == "INSUFFICIENT_LEVELS"


# ------------------------------------------------------------- D1 peak power
def test_bandlimited_peak_exact_on_analytic():
    """Half-sample offset: the raw peak under-reads ~4%, the band-limited one does not.

    This is the entire reason P_peak is redefined. With FWHM = 4.34 samples the
    sampling-phase bound is 1 - sech^2(dtheta/(2w)) = 4.02%.
    """
    th = _theta()
    on_grid = _sech(th / W_SOLITON).astype(np.complex128)
    true_max = float(np.abs(on_grid).max() ** 2)          # sech(0) = 1 exactly on grid
    half = _fourier_shift(on_grid, 0.5)

    raw = float(np.abs(half).max() ** 2)
    under = (true_max - raw) / true_max

    # Compare against the ANALYTIC sampling-phase bound for the w actually used
    # here, rather than a hard-coded band. NOTE: with W_SOLITON = 2.4267e-3 rad
    # (= 2.5495 samples at this N) the bound is 3.75%, not the 4.02% quoted in
    # the task brief; 4.02% corresponds to w = 2.46 samples = 2.3416e-3 rad. The
    # discrepancy is in the brief's two statements of w, not in the estimator --
    # see docs/CONVERGENCE_LLE.md.
    dtheta = 2.0 * math.pi / N_MODES
    expected = 1.0 - float(_sech(dtheta / (2.0 * W_SOLITON))) ** 2
    assert abs(under - expected) < 2e-3, (
        f"raw under-read {under:.4%} does not match the analytic bound {expected:.4%}")
    assert 0.03 <= under <= 0.05, f"raw under-read {under:.4%} outside sanity range"

    got = band_limited_peak(half, t_r=1.0)["P_peak_w"]
    assert abs(got - true_max) / true_max < 1e-4


def test_bandlimited_peak_reports_offset():
    res = band_limited_peak(_fourier_shift(
        _sech(_theta() / W_SOLITON).astype(np.complex128), 0.5), t_r=1.0)
    assert "peak_subsample_offset" in res
    assert abs(res["peak_subsample_offset"]) <= 1.0


def test_bandlimited_peak_rejects_even_grid():
    with pytest.raises(ValueError):
        band_limited_peak(np.ones(64, dtype=np.complex128), t_r=1.0)


# --------------------------------------------------------------- D2 3 dB span
def test_subbin_span_translation_invariant():
    """S3 must not move under a sub-sample translation; S3_int may jump by one."""
    th = _theta()
    base = (_sech(th / (40 * W_SOLITON)) + 0.01).astype(np.complex128)
    a = subbin_span_3db(base, MU)
    b = subbin_span_3db(_fourier_shift(base, 1.0), MU)
    c = subbin_span_3db(_fourier_shift(base, 0.5), MU)
    assert abs(b["S3_modes"] - a["S3_modes"]) < 1e-6
    assert abs(c["S3_modes"] - a["S3_modes"]) < 1e-6
    # the legacy integer metric is allowed to move; that is the point
    assert abs(c["S3_int_modes"] - a["S3_int_modes"]) <= 1


def test_subbin_span_reports_runs():
    th = _theta()
    field = (_sech(th / (40 * W_SOLITON)) + 0.01).astype(np.complex128)
    assert subbin_span_3db(field, MU)["S3_n_runs"] >= 1


# ------------------------------------------------------------- D3 DW centroid
def test_dw_centroid_fringe_insensitive():
    """A 12-mode fringe must not move the centroid, but does move a boxcar argmax.

    This documents why the DW estimator changed: the wing carries soliton/DW
    walk-off fringes of period ~10-13 modes, comparable to the 25-point boxcar
    previously used.
    """
    lo, hi = DW_BAND_DEFAULT
    centre = 0.5 * (lo + hi)
    lines = np.zeros(N_MODES, dtype=np.complex128)
    bump = np.exp(-0.5 * ((MU - centre) / 40.0) ** 2)
    lines[:] = 1e-6 + bump                                  # amplitude spectrum

    def _field(a):
        return np.fft.ifft(np.fft.ifftshift(a)) * N_MODES

    plain = _field(lines)
    fringe_db = 3.0 * np.sin(2 * math.pi * MU / 12.0)
    fringed = _field(lines * 10 ** (fringe_db / 20.0))

    c0 = dw_centroid(plain, MU)["mu_DW"]
    c1 = dw_centroid(fringed, MU)["mu_DW"]
    assert abs(c1 - c0) < 0.05, f"centroid moved {abs(c1 - c0):.3f} modes"

    def _boxcar_argmax(f):
        d = spectrum_dbc(f)
        sm = np.convolve(d, np.ones(25) / 25, mode="same")
        sel = (MU >= lo) & (MU <= hi)
        return float(MU[sel][int(np.argmax(sm[sel]))])

    moved = abs(_boxcar_argmax(fringed) - _boxcar_argmax(plain))
    assert moved >= 1.0, f"boxcar argmax only moved {moved} modes"


def test_dw_centroid_reports_power():
    """dw_power_dbc is band power relative to the CARRIER, so a pump line must
    dominate the spectrum for the ratio to be negative -- as it is in any real comb."""
    lo, hi = DW_BAND_DEFAULT
    lines = 1e-6 + np.exp(-0.5 * ((MU - 0.5 * (lo + hi)) / 40.0) ** 2)
    lines[MU == 0] = 1.0e4                                   # carrier
    field = np.fft.ifft(np.fft.ifftshift(lines.astype(np.complex128))) * N_MODES
    out = dw_centroid(field, MU)
    assert out["dw_power_dbc"] is not None and out["dw_power_dbc"] < 0.0


# ------------------------------------------------------------ D4 conditioning
def test_n60_conditioning_reported():
    th = _theta()
    field = (_sech(th / (40 * W_SOLITON)) + 1e-3).astype(np.complex128)
    d = line_count_diagnostics(field, MU)
    assert "dN60_ddB" in d and d["dN60_ddB"] > 0
    assert d["N60"] == d["N60_core"] + d["N60_mid"] + d["N60_edge"]
