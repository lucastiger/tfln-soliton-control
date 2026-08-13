"""Vacuum-floor normalization: n_tau invariance and normal-ordered subtraction.

The solver integrates the LLE in the truncated-Wigner c-number limit
(arXiv:2604.05897v1 Sec. V.B.2, Eq. 126), where the FFT convention is
``Etilde_mu = a_mu * n_tau * sqrt(hbar_omega0)``. Two consequences are pinned here.

**The per-mode photon number must not depend on n_tau.** The Langevin drive is
injected in the TIME domain, once per fine step, with per-quadrature std
``sqrt(hbar*omega0*kappa*n_tau*dt_fine/4)``; the ``n_tau`` inside that square root is
exactly what makes the per-mode occupation come out at 1/2 independent of the grid, via
Parseval. A missing or extra ``sqrt(n_tau)`` anywhere on that path leaves the vacuum
correct at ONE grid size and wrong at every other -- which no single-n_tau test can see.
``test_vacuum_floor_invariant_under_n_tau`` sweeps a factor of 32 in n_tau to catch it.

**A symmetric-ordered spectrum is not what an instrument reads.** Wigner moments are
symmetrically ordered, so a vacuum mode carries half a photon and the simulated spectrum
sits on a flat pedestal of ``n_tau^2 * hbar_omega0 / 2``. An OSA measures
``<a^dagger a>`` and reads zero for the vacuum, so any simulation-vs-measurement
comparison must subtract that pedestal first --
:func:`analysis.spectral_metrics.normal_ordered_spectrum`. The remaining two tests pin
that the subtraction removes the pedestal exactly (zero-mean residual on pure vacuum)
and leaves features far above it untouched (< 0.01 dB on the pump line and the soliton
peak).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from analysis.spectral_metrics import normal_ordered_spectrum
from simulator.lle_solver import (
    _load_config,
    d2_to_beta2_lle,
    hbar_omega0_from_config,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

_PHYS = _load_config()
_KAPPA_I, _KAPPA_C, _KAPPA = resolve_cavity_rates()
_T_R = 1.0 / float(_PHYS["fsr_hz"])
_BETA2 = float(d2_to_beta2_lle(_PHYS["d2_rad_per_s2"], _PHYS["fsr_hz"]))
_HBW = hbar_omega0_from_config(_PHYS)

#: Photon lifetime in round trips, 1/kappa. Sets how long "steady state" takes.
_LIFETIME_RT = 1.0 / (_KAPPA * _T_R)          # ~162 RT on the committed config


def _solve_quiet(**kw):
    """solve_lle_ssfm_jax with the expected pin=0 warnings suppressed.

    An undriven cavity is below the MI threshold by construction and sits under the
    pin-scaled labeler OFF floor; both warnings are expected here and irrelevant to
    the vacuum normalization under test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_lle_ssfm_jax(**kw)


def _tk0_sidecar(tmp_path: Path) -> Path:
    """Base config with T_k = 0, so the ONLY live channel is the quantum vacuum.

    T_k = 0 zeroes the thermodynamic temperature-fluctuation variance and hence every
    dT-derived channel (TRN, pyro-EO, FSR); TCCR is already zero on this
    centrosymmetric SiN config (r33 = 0), and the pump channels default off. The
    quantum channel is then switched on explicitly by the solver kwarg.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg.setdefault("physical_parameters", {})["T_k"] = 0.0
    out = tmp_path / "sin_params_vacuum_only.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return out


def _undriven_vacuum_snapshots(n_tau, sidecar, *, t_slow=3000, snap_interval=250,
                               seed=101, n_traj=2):
    """Undriven (pin = 0) vacuum run; returns the STEADY-STATE snapshot stack.

    ``t_slow = 3000`` RT is ~18 photon lifetimes, and only snapshots taken after the
    first half of the record are returned, so the initial transient has decayed by
    ``exp(-1500/162) ~ 1e-4``. The cold start already seeds at the vacuum level
    (``quantum_noise_seed_vacuum_init`` defaults on), so the run starts near
    equilibrium and merely has to stay there.

    Returns ``(n_traj, n_kept_snapshots, n_tau)`` complex128.
    """
    sol = _solve_quiet(
        pin=0.0,
        delta_omega=np.zeros((int(n_traj), int(t_slow))),
        t_slow=int(t_slow),
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(int(seed)),
        n_tau=int(n_tau),
        config_path=str(sidecar),
        snapshot_interval=int(snap_interval),
        n_substeps=1,
        dealias_two_thirds=False,     # every mode must thermalize, including the wings
        edge_absorber=False,
        dispersion_validity_mask=False,
        fine_cadence_M=1,
        quantum_noise_enabled=True,
    )
    snaps = np.asarray(sol["E_snapshots"])
    return snaps[:, snaps.shape[1] // 2:, :]


def _mean_photon_number(snaps, n_tau):
    """Per-mode photon number <n_mu> = |FFT(E)|^2 / (n_tau^2 * hbar*omega0)."""
    spec = np.abs(np.fft.fft(snaps, axis=-1)) ** 2
    return spec.mean(axis=(0, 1)) / (float(n_tau) ** 2 * _HBW)


# ---------------------------------------------------------------------------
# 1. The vacuum floor must not depend on the grid size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tau", [256, 512, 2048, 8192])
def test_vacuum_floor_invariant_under_n_tau(tmp_path_factory, n_tau):
    """Undriven vacuum, quantum channel ONLY: <n_mu> = 0.5 to 5% at every n_tau.

    This is the test that catches a missing or extra sqrt(n_tau) in the time-domain
    Langevin injection. The injected per-quadrature std carries an explicit factor
    sqrt(n_tau) (``_qnoise_increment``); by Parseval that factor is exactly what makes
    the per-MODE occupation grid-independent. Drop it, or square it, and the occupation
    scales as 1/n_tau or n_tau -- correct at whatever grid the constant was tuned on,
    wrong everywhere else. Sweeping 256 -> 8192 (a factor of 32) turns such an error
    into a factor of 32 miss against a 5% tolerance.

    The mu = 0 bin is excluded: with pin = 0 the analytic CW state is identically zero,
    so that bin carries the field's fast-time mean and no CW component, but it is also
    the one bin where any residual coherent leakage would land. Every other mode is
    pure vacuum.

    Statistics: the estimate averages n_traj * n_snapshots snapshots over all n_tau - 1
    non-pump modes. Modes are independent, so even at n_tau = 256 the effective sample
    count is in the thousands and the 5% band is many standard errors wide.
    """
    tmp = tmp_path_factory.mktemp("vac_norm")
    sidecar = _tk0_sidecar(tmp)
    snaps = _undriven_vacuum_snapshots(n_tau, sidecar)
    n_mu = _mean_photon_number(snaps, n_tau)

    mean_n = float(np.mean(np.delete(n_mu, 0)))
    assert abs(mean_n / 0.5 - 1.0) < 0.05, (
        f"n_tau={n_tau}: mean per-mode photon number {mean_n:.4f} != 0.5 within 5%. "
        f"A ratio near {0.5 * 256 / n_tau:.3g} or {0.5 * n_tau / 256:.3g} would "
        f"indicate a missing/extra sqrt(n_tau) in the time-domain injection."
    )

    # The floor must also be FLAT in mu -- a grid-size error that happened to preserve
    # the mean would still tilt or bin-concentrate the occupation.
    non_pump = np.delete(n_mu, 0)
    halves = np.array_split(non_pump, 4)
    for i, part in enumerate(halves):
        assert abs(float(np.mean(part)) / 0.5 - 1.0) < 0.10, (
            f"n_tau={n_tau}: mu-quartile {i} mean {float(np.mean(part)):.4f} "
            f"is not flat at 0.5"
        )


# ---------------------------------------------------------------------------
# 2. Subtracting the pedestal leaves a zero-mean residual on pure vacuum
# ---------------------------------------------------------------------------

def test_normal_ordered_undriven_vacuum_is_zero_mean(tmp_path_factory):
    """The normally-ordered undriven vacuum has mean consistent with 0.

    An instrument reads <a^dagger a> = 0 for the vacuum, so after subtracting the
    symmetric-ordered pedestal the residual must average to zero. It does NOT vanish
    pointwise: a single-snapshot vacuum mode is exponentially distributed with mean and
    std both equal to the pedestal, so individual residuals are order the pedestal and
    are negative about 63% of the time (P(X < mean) = 1 - 1/e for an exponential).
    ``clip=False`` is therefore mandatory here -- clipping would turn a zero-mean
    residual into a strictly positive one and the test would fail by construction,
    which is exactly the failure mode the clip=False path exists to prevent.

    Tolerance is 4 standard errors of the mean, computed from the measured scatter
    rather than assumed, so it stays honest if the snapshot correlation changes.
    """
    n_tau = 2048
    tmp = tmp_path_factory.mktemp("vac_norm_zero")
    sidecar = _tk0_sidecar(tmp)
    snaps = _undriven_vacuum_snapshots(n_tau, sidecar, seed=202)

    residual = normal_ordered_spectrum(snaps, n_tau, _HBW, clip=False)
    assert residual.shape == snaps.shape
    per_mode = residual.mean(axis=(0, 1))[1:]          # drop the mu=0 bin

    pedestal = n_tau**2 * _HBW / 2.0
    mean_residual = float(np.mean(per_mode))
    sem = float(np.std(per_mode, ddof=1)) / math.sqrt(per_mode.size)

    assert abs(mean_residual) < 4.0 * sem, (
        f"normally-ordered vacuum residual {mean_residual:.4e} is "
        f"{abs(mean_residual) / sem:.1f} standard errors from zero "
        f"(sem {sem:.4e}, pedestal {pedestal:.4e})"
    )
    # ...and it is small compared with the pedestal that was removed, i.e. the
    # subtraction actually did the work rather than the numbers being tiny anyway.
    assert abs(mean_residual) < 0.05 * pedestal, (mean_residual, pedestal)

    # The signed residual must genuinely straddle zero. If clip=True had been used
    # (or if the pedestal were wrong) this would fail.
    frac_negative = float(np.mean(residual < 0.0))
    assert 0.5 < frac_negative < 0.75, (
        f"fraction of negative single-snapshot residuals {frac_negative:.3f} is far "
        f"from the exponential-distribution expectation 1 - 1/e = 0.632"
    )

    # clip=True is display-only and must be a pure floor of the same array.
    clipped = normal_ordered_spectrum(snaps, n_tau, _HBW, clip=True)
    assert np.array_equal(clipped, np.maximum(residual, 0.0))
    assert float(np.mean(clipped)) > 0.0        # clipping does bias it positive


# ---------------------------------------------------------------------------
# 3. Features far above the floor are untouched
# ---------------------------------------------------------------------------

def test_normal_ordered_soliton_preserves_peak(tmp_path_factory):
    """Pump line and soliton peak change by < 0.01 dB after subtraction.

    Both sit tens of dB above the half-photon pedestal, so removing it is a
    negligible correction there -- which is the other half of the contract: the
    subtraction must fix the floor WITHOUT touching the comb features an experiment
    actually resolves. 0.01 dB corresponds to a 0.23% power change, so this is a tight
    bound, and it is checked on the real solver output rather than a synthetic
    spectrum.
    """
    n_tau = 1024
    tmp = tmp_path_factory.mktemp("vac_norm_soliton")
    sidecar = _tk0_sidecar(tmp)

    # Single soliton, seeded deterministically and held on the red-detuned side, with
    # the quantum channel ON so the spectrum genuinely carries the vacuum pedestal.
    tau = (np.arange(n_tau) - n_tau // 2) * (_T_R / n_tau)
    kappa_c_pin = math.sqrt(max(_KAPPA_C * float(_PHYS["pin_w"]), 0.0))
    dw = 10.0 * _KAPPA
    # Analytic DKS ansatz amplitude/width for this detuning (see analysis.dks_access);
    # the exact prefactor does not matter here -- the test only needs a bright,
    # localized pulse whose comb lines sit far above the vacuum floor.
    gamma = float(_PHYS["gamma_LLE_per_J_per_s"])
    amp = math.sqrt(2.0 * dw / gamma)
    width = math.sqrt(abs(_BETA2) / (2.0 * dw))
    # clip the sech argument: the far wings underflow to 0 anyway, but cosh() of a
    # few thousand overflows float64 and warns on the way there.
    e0 = (amp / np.cosh(np.clip(tau / width, -700.0, 700.0))).astype(np.complex128)
    e0 = e0 + kappa_c_pin / (_KAPPA / 2.0 + 1j * dw)          # + CW background

    sol = _solve_quiet(
        pin=float(_PHYS["pin_w"]),
        delta_omega=dw,
        t_slow=400,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(303),
        n_tau=n_tau,
        config_path=str(sidecar),
        snapshot_interval=100,
        e0_override=e0,
        n_substeps=1,
        fine_cadence_M=1,
        quantum_noise_enabled=True,
    )
    field = np.asarray(sol["E_snapshots"])[0, -1]              # last snapshot
    raw = np.abs(np.fft.fft(field)) ** 2
    corrected = normal_ordered_spectrum(raw, n_tau, _HBW, clip=False)

    pedestal = n_tau**2 * _HBW / 2.0
    shifted_raw = np.fft.fftshift(raw)
    mu = np.arange(n_tau) - n_tau // 2

    pump_bin = int(np.argmax(shifted_raw))
    assert mu[pump_bin] == 0, "the pump line must be the mu=0 bin"

    # Soliton peak: the strongest NON-pump comb line (the soliton's own envelope).
    non_pump = shifted_raw.copy()
    non_pump[pump_bin] = -np.inf
    soliton_bin = int(np.argmax(non_pump))
    assert mu[soliton_bin] != 0

    shifted_corr = np.fft.fftshift(corrected)
    for name, b in (("pump line", pump_bin), ("soliton peak", soliton_bin)):
        before = float(shifted_raw[b])
        after = float(shifted_corr[b])
        assert after > 0.0, (name, after)
        delta_db = abs(10.0 * math.log10(after / before))
        assert delta_db < 0.01, (
            f"{name} at mu={mu[b]} moved {delta_db:.4f} dB after subtracting the "
            f"vacuum pedestal (before {before:.4e}, after {after:.4e}, pedestal "
            f"{pedestal:.4e}); it should sit far enough above the floor to be "
            f"unaffected"
        )
        # Sanity that the bound is meaningful: the feature really is far above the
        # pedestal, so "< 0.01 dB" is a consequence of the physics, not of a tiny
        # pedestal.
        assert before > 100.0 * pedestal, (name, before / pedestal)


# ---------------------------------------------------------------------------
# 4. Unit-level contract of the helper itself
# ---------------------------------------------------------------------------

def test_normal_ordered_spectrum_contract():
    """Exact arithmetic, input forms, and the argument validation."""
    n_tau, hbw = 64, 2.0e-19
    pedestal = n_tau**2 * hbw / 2.0

    power = np.array([0.0, pedestal, 2.0 * pedestal, 10.0 * pedestal])
    np.testing.assert_allclose(
        normal_ordered_spectrum(power, n_tau, hbw, clip=False),
        power - pedestal, rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        normal_ordered_spectrum(power, n_tau, hbw, clip=True),
        np.maximum(power - pedestal, 0.0), rtol=0, atol=0,
    )

    # A complex field input is FFT'd along the last axis (not fftshifted).
    rng = np.random.default_rng(0)
    field = rng.standard_normal((3, n_tau)) + 1j * rng.standard_normal((3, n_tau))
    np.testing.assert_allclose(
        normal_ordered_spectrum(field, n_tau, hbw, clip=False),
        np.abs(np.fft.fft(field, axis=-1)) ** 2 - pedestal,
    )

    # n_tau is an ARGUMENT, not inferred from the array: a masked/truncated spectrum
    # must still subtract the pedestal of the grid it was computed on.
    sub = power[:2]
    np.testing.assert_allclose(
        normal_ordered_spectrum(sub, n_tau, hbw, clip=False), sub - pedestal
    )

    for bad in (0, -8):
        with pytest.raises(ValueError, match="n_tau"):
            normal_ordered_spectrum(power, bad, hbw)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="hbar_omega0"):
            normal_ordered_spectrum(power, n_tau, bad)
    with pytest.raises(ValueError, match="non-negative"):
        normal_ordered_spectrum(np.array([-1.0, 1.0]), n_tau, hbw)
    with pytest.raises(ValueError, match="non-finite"):
        normal_ordered_spectrum(np.array([np.nan, 1.0]), n_tau, hbw)
