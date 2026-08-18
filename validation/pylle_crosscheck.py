"""Cross-code validation of this repo's LLE solver against pyLLE (Julia backend).

WHAT THIS IS, AND WHAT IT IS WORTH
----------------------------------
``validation/analytic_cw.py`` checks the solver against *exact mathematics* to
1e-12, and ``validation/convergence.py`` / ``validation/mms.py`` establish the
observed order of accuracy. Those are stronger claims than cross-code agreement:
two codes agree only to the level at which they discretize the same PDE, and
agreement cannot distinguish "both right" from "both wrong in the same way".

What a cross-code check adds is the one thing exact mathematics cannot give: an
independent check on *convention and bookkeeping* — the sign of the detuning,
the direction of the dispersion, factors of 2*pi, the mode-index origin. Those
are the errors that silently produce plausible-looking solitons. That, and not
"agreement to N digits", is what this module is for.

Reference for the physics: Herr, Tikan & Kippenberg, arXiv:2604.05897v1
(7 Apr 2026). Equation/section numbers are v1 numbers.

THE CONVENTION AUDIT (established from pyLLE 4.1.2 source, then verified at
runtime by the assertions in this file)
------------------------------------------------------------------------------
pyLLE's kernel (``pyLLE/ComputeLLE.jl``) integrates, with time measured in round
trips (``dt = 1``) and ``alpha = kappa*t_r`` dimensionless::

    FFT_Lin(it) = -alpha/2 + 1im*(Dint_shift - domega_all[1][it])*tR   (:338)
    NL(uu)      = -1im*(gamma*L*abs(uu)^2)                             (:342)

This repo's solver (``simulator/lle_solver.py::_fine_step``, :896/:936) composes
``exp((-kappa/2 - 1j*D_int - 1j*delta_omega)*dt)`` and ``exp(+1j*gamma*|E|^2*dt)``.

**Finding 1 — D_int is in rad/s (angular), NOT Hz.** In the Julia kernel both
``Dint`` and ``domega`` are multiplied by ``tR`` to become dimensionless, while
``kappa_ext = omega0/Qc*tR`` (:88) is likewise dimensionless. So both are angular
frequencies in rad/s, exactly like our ``d_int_grid`` and ``delta_omega``.
**No 2*pi conversion anywhere.** ``assert_round_trip`` re-checks this dimensionally.

**Finding 2 — pyLLE's field is the COMPLEX CONJUGATE of ours, and its detuning
sign is OPPOSITE.** Conjugating pyLLE's equation and matching term by term
against ours gives the exact dictionary

    E_ours(theta)   =  conj(A_pyLLE(theta)) * sqrt(t_r)          [sqrt(J)]
    delta_omega_p   = -delta_omega_ours                          [rad/s]
    D_int_p         = +D_int_ours                                [rad/s]

Ours is ``delta_omega = omega_res - omega_pump`` (positive = red-detuned =
soliton side); pyLLE's is ``omega_pump - omega_res``. **This is the trap.**
Conjugation in the time domain is a MIRROR in the frequency domain, mu -> -mu.
Comparing DW peak mode indices with an exact-match requirement without applying
it would fail — or, worse, spuriously PASS on a dispersion profile symmetric
enough to hide the mirror. The measured dispersion used here is strongly
asymmetric (its two DWs sit at mu = -3056 and +3275), so the mirror is not
hideable: that asymmetry is what gives observable #1 its teeth.

**Finding 2b — the mirror also applies to the DISPERSION INPUT, not just the
output.** This is the half that is easy to miss, and it was caught here only by
running: with the conjugate field ``psi = conj(A)``, the Fourier relation is
``psi_hat(mu) = conj(A_hat(-mu))``, so the dispersion operator pyLLE applies at
bin ``mu`` lands on our bin ``-mu``. Matching term by term therefore requires

    D_int_pyLLE(mu)  =  D_int_ours(-mu)

i.e. **pyLLE must be handed the MIRRORED dispersion profile**, and its output
field is then un-mirrored by the same conjugation. Feeding pyLLE our
un-mirrored D_int (the obvious thing to do, and what the first run here did)
produces two codes that agree on everything symmetric and disagree only on the
asymmetric observables: the diagnostic signature was our DW at mu = -3082
against pyLLE's at mu = +3078 — the same wave, reflected. There is no
non-conjugate escape from this: pyLLE's Kerr term is ``-1i*gamma*L*|A|^2``
against our ``+1i*gamma*|E|^2``, and gamma > 0 on both sides, so the
conjugation is forced by the nonlinearity and the dispersion mirror comes with
it. ``build_pylle_dispfile`` is where the mirror is applied; the JSON records
``dispersion_mirror_applied: true``.

**Finding 3 — the amplitude normalization is a power/energy conversion.**
pyLLE's ``A`` is in sqrt(W) (``|A|^2`` = intracavity power), ours is in sqrt(J).
Equating the Kerr phase per unit time, with gamma_LLE = gamma_NLSE*L/t_r^2,
gives ``|A|^2 = |E|^2 / t_r`` — verified independently against the CW steady
state of both equations (both reduce to ``4*kappa_c*Pin/kappa^2``).

**Finding 4 — the drive phase is fixable exactly.** pyLLE drives with
``Fdrive = -1im*Ain``, ``Ain = sqrt(Pin)*exp(-1im*phi_pmp)`` (:270, :186-189),
whereas ours drives with a real positive ``sqrt(kappa_c*Pin)``. Setting
``phi_pmp = -pi/2`` makes pyLLE's drive real positive too, so the dictionary in
Finding 2 holds with NO residual global phase. (Every observable here depends on
|field| only, so this is belt-and-braces — but it makes the field overlay panel
in the figure meaningful rather than phase-scrambled.)

**Finding 5 — pyLLE 4.1.2 is already deterministic, and that is a problem.**
The kernel defines ``Noise()`` (:172) but **its call site is commented out** —
``return Force #.- 1im*Noise()`` (:274) — and ``DKS_init`` defaults to
``np.zeros`` (``_llesolver.py:479``). So nothing needs seeding *off*. But the
consequence is that pyLLE started from its own defaults can never form a
soliton: with a single-mode drive, an exactly-uniform field and no noise, the
state stays exactly CW and modulational instability has nothing to amplify.
(In practice it is seeded by FFT round-off — deterministic, but not physics.)
Both codes are therefore given the SAME explicit deterministic sech seed; see
``soliton_seed``.

**Finding 6 — pyLLE re-zeros D_int at the DOMAIN CENTER, not the pump mode.**
``Dint = Dint .- Dint[mu0]`` (ComputeLLE.jl:115) with ``mu0`` the domain
midpoint (:106). We keep the pump centred (``ind_pmp = 0``) so the two coincide;
``_worker_main`` asserts ``Dint[mu0] == 0`` before the re-zero so this can never
silently bite.

**Finding 7 — dispersion enters pyLLE as a resonance-frequency CSV that it
spline-refits**, so its D_int is NOT bit-identical to our loader's
(``_analyzedisp.py:63-64``, :100-106). Rather than leave that as an unquantified
floor under every tolerance, we *eliminate* it: pyLLE computes D_int from
``config/pyLLE_dispersion_w4400_h800.csv`` (already in its native
``mode, f_Hz`` format), and that exact array is fed back into our solver via
``d_int_grid``. Both codes then integrate literally the same dispersion. The
residual difference against our native loader is still measured and reported as
``dispersion_refit_max_rel_diff`` — a diagnostic, not a tolerance.

**Finding 8 — the nominal pump wavelength is NOT the mode the dispersion is
referenced to, and pyLLE locates the pump by frequency.** Our loader references
D_int to the CSV row ``mu = 0`` (``lle_solver.py:509-522``) and the solver drives
FFT bin 0, so in our solver the pump sits on CSV ``mu = 0`` by construction. That
resonance is at 1.548600 um, **7.11 FSR away** from the config's nominal
``pump_wavelength_m = 1.55 um``. pyLLE instead searches its dispfile for the mode
nearest ``sim['f_pmp']`` (``_analyzedisp.py:57-70``), so passing ``c/1.55um``
would have put the two codes on pump modes seven free spectral ranges apart —
with both still producing perfectly plausible solitons. We therefore pin
``sim['f_pmp']`` to the CSV ``mu = 0`` resonance and derive ``Qi``/``Qc`` from
that SAME ``omega0``, which matters because pyLLE reconstructs
``kappa = omega0/Q*t_r`` internally: using a mismatched ``omega0`` in the two
places would have silently rescaled both loss rates by 0.09%. With the pump
pinned, ``omega0`` cancels identically and ``kappa_0 = kappa_i*t_r``,
``kappa_ext = kappa_c*t_r`` exactly. Nothing in our deterministic dynamics
depends on ``omega0`` once ``dn_dT`` and the quantum channel are off, so this
changes no numbers on our side -- it only removes a 7-FSR trap on pyLLE's.

**Finding 9 — D1 sets the co-moving frame AND, in pyLLE, the round-trip time;
the repo's two FSRs differ by 0.6% and only one choice keeps the DWs where they
belong.** ``load_dint_grid`` builds ``D_int = omega - omega0 - D1*mu`` from a
degree-7 fit of the measured CSV, giving ``D1_csv/2pi = 24.4516 GHz``, while the
solver takes ``t_r = 1/fsr_hz`` from the config leaf ``24.6 GHz``. pyLLE cannot
reproduce that split: its kernel hardwires ``t_r = 2*pi/D1`` (ComputeLLE.jl:86),
so one D1 must serve both roles. Passing ``D1 = 2*pi*24.6 GHz`` tilts D_int by
``mu*(D1_config - D1_csv)``, which reaches 3.1e12 rad/s at the grid edge --
two orders of magnitude larger than D_int itself -- and moves every zero
crossing, i.e. it relocates the dispersive waves. That is not an acceptable
"frame choice". We therefore set ``D1_manual = D1_csv`` so pyLLE reproduces the
repo's native D_int exactly, and derive a config with ``fsr_hz = D1_csv/(2*pi)``
so our t_r matches pyLLE's. The committed config is untouched and every leaf
stays numeric. This is the one place where the cross-check runs the repo's
solver at a t_r 0.6% away from the committed leaf; it is recorded in the JSON as
``fsr_hz_used`` next to ``fsr_hz_config``, and it is forced by pyLLE's
tR-D1 coupling, not chosen.

MATCHED DISCRETIZATION
----------------------
pyLLE's kernel is hardwired to one step per round trip (``dt = 1``). We
therefore run with ``n_substeps=1`` and ``fine_cadence_M=1`` so both codes take
the SAME step. This is deliberate: a cross-code check must compare the same
discretization, not the best one each code can do. One consequence is recorded
in the JSON as ``max_abs_Dint_t_r``: the outermost modes of this measured
dispersion have |D_int|*t_r slightly above pi, i.e. their per-round-trip phase
is under-resolved. That is a shared property of the mean-field LLE at this step
size, identical in both codes, not a disagreement source.

Our thermo-optic ODE has no counterpart in pyLLE, so it is switched off the same
way ``validation/analytic_cw.py`` does it — by deriving a config with
``dn_dT_per_k = 0.0`` (see :func:`derived_config`, which also carries the
Finding 9 timebase override; the committed config is never touched and every
leaf it already had is preserved bit-for-bit, so tests/test_config.py is
unaffected). ``delta_omega_eff == delta_omega`` exactly, which the run asserts.

RUNNING IT
----------
pyLLE needs numpy<2 (it calls the removed ``np.string_``) and a Julia backend,
so it CANNOT share this repo's environment. It runs out-of-process::

    python -m validation.pylle_crosscheck \\
        --pylle-python /path/to/pylle-env/bin/python \\
        --julia-bin    /path/to/pylle-env/bin/julia

Both may also be supplied as ``PYLLE_PYTHON`` / ``JULIA_BIN`` environment
variables. The same file is re-executed by the orchestrator inside the pyLLE
interpreter with ``--worker`` — that is why every heavy import in this module is
function-local: the worker half must import cleanly under numpy 1.x with no jax.

Outputs ``validation/results/pylle_crosscheck.json`` and
``validation/results/pylle_crosscheck.png``. Exit status is nonzero if any
observable fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

C0 = 299_792_458.0
REPO_ROOT = Path(__file__).resolve().parents[1]
DISP_CSV = REPO_ROOT / "config" / "pyLLE_dispersion_w4400_h800.csv"
RESULTS_DIR = REPO_ROOT / "validation" / "results"

# --------------------------------------------------------------------------
# Protocol constants. Chosen in the module docstring's terms; see there.
# --------------------------------------------------------------------------
MU_HALF = 3300           # mode grid is mu in [-MU_HALF, +MU_HALF]  (6601 modes)
N_ROUNDTRIPS_MAIN = 10_000     # main ramp run, round trips
N_ROUNDTRIPS_SCAN = 2_000      # per-point hold for the existence scan
DW_CORE_EXCLUDE = 1_500  # |mu| below this is soliton core, not a dispersive wave
DW_CENTROID_HALFWIDTH = 150   # modes either side of the bump used for its centroid
DW_MIN_PROMINENCE_DB = 3.0    # a DW must stand this far above the local wing level

TOLERANCES = {
    "dw_peak_mode_indices": 0.0,        # EXACT integer match
    "spectral_span_3db": 0.02,
    "soliton_peak_power": 0.02,
    "existence_range_edges": 0.05,
    "comb_line_count_60dbc": 0.02,
}


# ==========================================================================
# PART 1 — the parameter translation table
# ==========================================================================
def load_device(config_path: Path | None = None) -> dict:
    """Read the committed device parameters that both codes are driven from.

    Parameters
    ----------
    config_path : pathlib.Path or None, optional
        Config to read; ``None`` (default) uses ``config/sin_params.yaml``.

    Returns
    -------
    dict
        ``fsr_hz`` [Hz], ``pump_wavelength_m`` [m], ``pin_w`` [W],
        ``kappa_i_rad_per_s`` and ``kappa_c_rad_per_s`` [rad/s], ``n0``
        [dimensionless], ``n2_m2_per_w`` [m**2/W],
        ``effective_mode_area_m2`` [m**2] and ``gamma_LLE_per_J_per_s``
        [J^-1 s^-1].

    Raises
    ------
    OSError
        If the config cannot be read.
    KeyError
        If ``physical_parameters`` or any of the nine keys is absent. Every one
        is required: a cross-code comparison cannot fall back to a default for
        a device parameter without silently comparing two different devices.

    Notes
    -----
    Only these nine leaves are read, and each is coerced to ``float``. Every
    leaf under ``physical_parameters`` is a plain number by repository
    convention, so nothing here needs to interpret a boolean or a string.

    Examples
    --------
    >>> device = load_device()
    >>> sorted(device)
    ['effective_mode_area_m2', 'fsr_hz', 'gamma_LLE_per_J_per_s', 'kappa_c_rad_per_s', 'kappa_i_rad_per_s', 'n0', 'n2_m2_per_w', 'pin_w', 'pump_wavelength_m']
    >>> all(isinstance(v, float) for v in device.values())
    True
    """
    import yaml

    src = Path(config_path) if config_path else REPO_ROOT / "config" / "sin_params.yaml"
    with src.open("r", encoding="utf-8") as fh:
        p = (yaml.safe_load(fh) or {})["physical_parameters"]
    return {
        "fsr_hz": float(p["fsr_hz"]),
        "pump_wavelength_m": float(p["pump_wavelength_m"]),
        "pin_w": float(p["pin_w"]),
        "kappa_i_rad_per_s": float(p["kappa_i_rad_per_s"]),
        "kappa_c_rad_per_s": float(p["kappa_c_rad_per_s"]),
        "n0": float(p["n0"]),
        "n2_m2_per_w": float(p["n2_m2_per_w"]),
        "effective_mode_area_m2": float(p["effective_mode_area_m2"]),
        "gamma_LLE_per_J_per_s": float(p["gamma_LLE_per_J_per_s"]),
    }


def derived_config(fsr_hz: float, out_dir: Path) -> str:
    """Config identical to the committed one but with two leaves overridden.

    ``dn_dT_per_k = 0.0``   -- kills the thermo-optic shift, which pyLLE has no
                               counterpart for (same device as
                               ``analytic_cw.thermal_off_config``; the ODE still
                               runs but is decoupled, so
                               ``delta_omega_eff == delta_omega`` exactly).
    ``fsr_hz = D1_csv/2pi`` -- makes our ``t_r`` equal pyLLE's hardwired
                               ``2*pi/D1``; see Finding 9.

    The committed ``config/sin_params.yaml`` is never modified. Both overrides
    are plain floats, so every leaf under ``physical_parameters`` still parses
    as a number and ``tests/test_config.py`` is unaffected.

    Parameters
    ----------
    fsr_hz : float
        FSR [Hz] to write, normally ``D1_csv/(2*pi)`` from the fitted
        dispersion rather than the config's nominal value.
    out_dir : pathlib.Path
        Directory the derived config is written into.

    Returns
    -------
    str
        Path of the written ``pylle_crosscheck_config.yaml``.

    Raises
    ------
    AssertionError
        If the derived config changed the ``physical_parameters`` key set, or
        if any leaf other than the two overridden ones failed to survive the
        YAML round trip bit-for-bit.
    OSError
        If the source cannot be read or the destination cannot be written.

    Notes
    -----
    Checking every leaf directly is stronger than re-deriving the config
    test's allowlist here, and it cannot drift out of sync with it. A float
    that did not survive ``safe_dump`` -> ``safe_load`` unchanged would
    silently perturb one code's run and not the other's, which is precisely the
    class of bug this whole module exists to catch.

    No ``Examples`` section: it writes a file into a caller-supplied directory
    and reads the committed config.
    """
    import yaml

    src = REPO_ROOT / "config" / "sin_params.yaml"
    with src.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    physical = cfg["physical_parameters"]
    original = dict(physical)
    overrides = {"dn_dT_per_k": 0.0, "fsr_hz": float(fsr_hz)}
    physical.update(overrides)

    dest = out_dir / "pylle_crosscheck_config.yaml"
    with dest.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    # The derived config must differ in EXACTLY the two overridden leaves, and
    # every other leaf must survive the YAML round trip bit-for-bit -- a float
    # that did not would silently perturb the run. Checking this directly is
    # stronger than re-deriving tests/test_config.py's allowlist here, and it
    # cannot drift out of sync with it.
    with dest.open("r", encoding="utf-8") as fh:
        reloaded = (yaml.safe_load(fh) or {})["physical_parameters"]
    if set(reloaded) != set(original):
        raise AssertionError("derived config changed the physical_parameters key set")
    for key, value in original.items():
        expected = overrides.get(key, value)
        got = reloaded[key]
        if isinstance(expected, float):
            if got != expected:
                raise AssertionError(
                    f"physical_parameters[{key!r}] did not survive the YAML round trip: "
                    f"{expected!r} -> {got!r}"
                )
        elif got != expected:
            raise AssertionError(
                f"derived config altered physical_parameters[{key!r}]: {expected!r} -> {got!r}"
            )
    return str(dest)


def ours_to_pylle(ours: dict, f_pmp_hz: float | None = None) -> dict:
    """Translate our SI device parameters into pyLLE's ``res``/``sim`` inputs.

    Every entry is annotated with its unit and the exact relation used. This is
    the table that the task warned is where cross-code checks go wrong, so it is
    written once, here, and nowhere else.

        ours                              ->  pyLLE
        --------------------------------------------------------------------
        fsr_hz                     [Hz]   ->  FSR (via D1_manual = 2*pi*fsr_hz)
        pump_wavelength_m          [m]    ->  sim['f_pmp'] = c/lambda      [Hz]
        pin_w                      [W]    ->  sim['Pin']                   [W]
        kappa_i_rad_per_s          [rad/s]->  res['Qi'] = omega0/kappa_i   [-]
        kappa_c_rad_per_s          [rad/s]->  res['Qc'] = omega0/kappa_c   [-]
        n2_m2_per_w, A_eff         [SI]   ->  res['gamma'] = gamma_NLSE  [1/W/m]
        gamma_LLE_per_J_per_s             ->  fixes n_g, hence R (see below)
        d_int_grid                 [rad/s]->  res['dispfile'] (mode, f_Hz CSV)
        delta_omega                [rad/s]->  sim['domega_*'] = -delta_omega

    ``n_g`` is not a config leaf, so it is DERIVED from leaves that are, using
    the repo's own documented chain gamma_LLE = gamma_NLSE * v_g * FSR with
    v_g = c/n_g. That makes the chain exactly invertible:

        n_g   := gamma_NLSE*c*FSR / gamma_LLE
        L     := c/(n_g*FSR)
        =>  gamma_NLSE*L/t_r^2  ==  gamma_LLE     (exactly, to fp round-off)

    which is the round-trip identity asserted in ``assert_round_trip``. R is
    then L/(2*pi); pyLLE's kernel only ever uses R through L = 2*pi*R in the
    Kerr term, so only the PRODUCT gamma_NLSE*L is physically constrained, and
    the identity above pins it exactly.

    Parameters
    ----------
    ours : dict
        Device parameters as returned by :func:`load_device`.
    f_pmp_hz : float or None, optional
        Pump frequency [Hz]. ``None`` (default) uses ``c/pump_wavelength_m``.
        The caller overrides it so the pump is pinned to the mode the
        dispersion grid is referenced to, which is NOT the nominal wavelength.

    Returns
    -------
    dict
        ``f_pmp_hz`` [Hz], ``omega0_rad_per_s`` [rad/s],
        ``pump_wavelength_m_nominal`` [m], ``t_r_s`` [s], ``Pin_w`` [W],
        ``Qi`` / ``Qc`` [dimensionless], ``gamma_nlse_per_w_per_m``
        [W^-1 m^-1], ``n_g`` [dimensionless], ``L_cav_m`` [m], ``R_m`` [m],
        ``D1_manual_rad_per_s`` [rad/s], ``phi_pmp_rad`` [rad], ``n0``
        [dimensionless], ``n2_m2_per_w`` [m**2/W] and ``Aeff_m2`` [m**2].

    Raises
    ------
    KeyError
        If ``ours`` is missing a device parameter.
    ZeroDivisionError
        If the pump wavelength, the FSR, the effective area or ``gamma_LLE``
        is zero.

    Examples
    --------
    >>> forward = ours_to_pylle(load_device())
    >>> sorted(forward)[:5]
    ['Aeff_m2', 'D1_manual_rad_per_s', 'L_cav_m', 'Pin_w', 'Qc']

    The identity the round trip rests on, ``gamma_NLSE*L/t_r**2 == gamma_LLE``:

    >>> device = load_device()
    >>> lhs = (forward["gamma_nlse_per_w_per_m"] * forward["L_cav_m"]
    ...        / forward["t_r_s"] ** 2)
    >>> bool(abs(lhs / device["gamma_LLE_per_J_per_s"] - 1) < 1e-12)
    True

    The pump phase is fixed at -pi/2 so pyLLE's drive comes out real and
    positive:

    >>> bool(abs(forward["phi_pmp_rad"] + 3.141592653589793 / 2) < 1e-15)
    True
    """
    lam = ours["pump_wavelength_m"]
    fsr = ours["fsr_hz"]
    # Finding 8 (below): the pump is pinned to the mode the dispersion grid is
    # referenced to, which is NOT c/lambda_nominal. omega0 follows f_pmp so that
    # Q = omega0/kappa inverts exactly inside pyLLE's own kappa = omega0/Q*t_r.
    f_pmp = C0 / lam if f_pmp_hz is None else float(f_pmp_hz)
    omega0 = 2.0 * math.pi * f_pmp                    # rad/s
    t_r = 1.0 / fsr                                   # s
    gamma_lle = ours["gamma_LLE_per_J_per_s"]         # J^-1 s^-1

    # gamma_NLSE = 2*pi*n2/(lambda*A_eff)   [W^-1 m^-1]
    gamma_nlse = 2.0 * math.pi * ours["n2_m2_per_w"] / (lam * ours["effective_mode_area_m2"])
    n_g = gamma_nlse * C0 * fsr / gamma_lle           # dimensionless
    l_cav = C0 / (n_g * fsr)                          # m (round-trip length)
    radius = l_cav / (2.0 * math.pi)                  # m

    return {
        "f_pmp_hz": f_pmp,
        "omega0_rad_per_s": omega0,
        "pump_wavelength_m_nominal": lam,
        "t_r_s": t_r,
        "Pin_w": ours["pin_w"],
        "Qi": omega0 / ours["kappa_i_rad_per_s"],
        "Qc": omega0 / ours["kappa_c_rad_per_s"],
        "gamma_nlse_per_w_per_m": gamma_nlse,
        "n_g": n_g,
        "L_cav_m": l_cav,
        "R_m": radius,
        # D1 sets BOTH the co-moving frame of D_int and (in pyLLE) t_r = 2*pi/D1.
        # Overridden by the caller to the CSV-fitted D1 -- see Finding 9.
        "D1_manual_rad_per_s": 2.0 * math.pi * fsr,
        "phi_pmp_rad": -math.pi / 2.0,   # Finding 4: makes pyLLE's drive real +ve
        "n0": ours["n0"],
        "n2_m2_per_w": ours["n2_m2_per_w"],
        "Aeff_m2": ours["effective_mode_area_m2"],
    }


def pylle_to_ours(q: dict) -> dict:
    """Invert :func:`ours_to_pylle` -- the round-trip leg of the translation check.

    Parameters
    ----------
    q : dict
        pyLLE-side parameters as produced by :func:`ours_to_pylle`.

    Returns
    -------
    dict
        The nine device parameters of :func:`load_device`, in SI:
        ``fsr_hz`` [Hz], ``pump_wavelength_m`` [m], ``pin_w`` [W],
        ``kappa_i_rad_per_s`` / ``kappa_c_rad_per_s`` [rad/s], ``n0``
        [dimensionless], ``n2_m2_per_w`` [m**2/W],
        ``effective_mode_area_m2`` [m**2] and ``gamma_LLE_per_J_per_s``
        [J^-1 s^-1].

    Raises
    ------
    KeyError
        If ``q`` is missing a key the inversion needs.
    ZeroDivisionError
        If ``t_r_s``, ``Qi`` or ``Qc`` is zero.

    Notes
    -----
    Every relation is an exact algebraic inversion, except
    ``gamma_LLE = gamma_NLSE * L / t_r**2``, which round-trips through the
    derived ``n_g`` and ``L`` -- those factors cancel identically, which is why
    :func:`assert_round_trip` can hold it to the same 1e-12 as the rest.

    Examples
    --------
    >>> device = load_device()
    >>> rebuilt = pylle_to_ours(ours_to_pylle(device))
    >>> sorted(rebuilt) == sorted(device)
    True
    >>> all(abs(rebuilt[k] - device[k]) <= 1e-12 * abs(device[k]) for k in device)
    True
    """
    omega0 = q["omega0_rad_per_s"]
    fsr = 1.0 / q["t_r_s"]
    return {
        "fsr_hz": fsr,
        "pump_wavelength_m": q["pump_wavelength_m_nominal"],
        "pin_w": q["Pin_w"],
        "kappa_i_rad_per_s": omega0 / q["Qi"],
        "kappa_c_rad_per_s": omega0 / q["Qc"],
        "n0": q["n0"],
        "n2_m2_per_w": q["n2_m2_per_w"],
        "effective_mode_area_m2": q["Aeff_m2"],
        # gamma_LLE = gamma_NLSE * L / t_r^2   (Finding 3)
        "gamma_LLE_per_J_per_s": q["gamma_nlse_per_w_per_m"] * q["L_cav_m"] / q["t_r_s"] ** 2,
    }


def assert_round_trip(ours: dict, rtol: float = 1e-12, f_pmp_hz: float | None = None) -> dict:
    """ours -> pyLLE -> ours must reproduce every scalar to ``rtol``.

    All of these relations are exact algebraic inversions (products, quotients
    and one reciprocal), so double precision reproduces them to a few ULP and
    1e-12 is a mathematically justified bound rather than a fudge. The one entry
    that is NOT a pure inversion is gamma_LLE, which round-trips through the
    derived n_g and L; the docstring of :func:`ours_to_pylle` shows the n_g and
    L factors cancel identically, so it is held to the same 1e-12.

    Parameters
    ----------
    ours : dict
        Device parameters as returned by :func:`load_device`.
    rtol : float, optional
        Relative bound [dimensionless] every scalar must round-trip within,
        default 1e-12.
    f_pmp_hz : float or None, optional
        Pump frequency [Hz] to translate at; ``None`` (default) derives it from
        the nominal wavelength.

    Returns
    -------
    dict
        One entry per parameter with ``ours``, ``round_trip`` and ``rel_diff``
        [dimensionless], plus ``_worst_rel_diff``.

    Raises
    ------
    AssertionError
        On the FIRST parameter that exceeds ``rtol``, naming it and both
        values. A translation bug is not something to average over.
    KeyError
        If ``ours`` is missing a parameter.

    Notes
    -----
    This is criterion H1, and it is HARD: it holds for any correct
    implementation of the translation, independently of what either solver
    computes. Anything above 1e-12 is a translation bug, not a physics
    disagreement.

    Examples
    --------
    >>> report = assert_round_trip(load_device())
    >>> bool(report["_worst_rel_diff"] <= 1e-12)
    True
    >>> sorted(report["fsr_hz"])
    ['ours', 'rel_diff', 'round_trip']
    """
    fwd = ours_to_pylle(ours, f_pmp_hz=f_pmp_hz)
    back = pylle_to_ours(fwd)
    report = {}
    worst = 0.0
    for key, original in ours.items():
        rebuilt = back[key]
        rel = abs(rebuilt - original) / abs(original) if original else abs(rebuilt)
        report[key] = {"ours": original, "round_trip": rebuilt, "rel_diff": rel}
        worst = max(worst, rel)
        if rel > rtol:
            raise AssertionError(
                f"Round-trip translation failed for {key!r}: {original!r} -> "
                f"{rebuilt!r}, relative difference {rel:.3e} > {rtol:.1e}."
            )
    report["_worst_rel_diff"] = worst
    return report


def print_translation_table(ours: dict, fwd: dict, rt: dict) -> None:
    """Print the parameter translation table and its round-trip residuals.

    Parameters
    ----------
    ours : dict
        Device parameters from :func:`load_device`.
    fwd : dict
        Their translation, from :func:`ours_to_pylle`.
    rt : dict
        The round-trip report from :func:`assert_round_trip`.

    Returns
    -------
    None
        Writes the table to stdout.

    Raises
    ------
    KeyError
        If any input is missing a parameter the table displays.

    Notes
    -----
    Printed in full on every run, deliberately. The parameter map is where
    cross-code comparisons go wrong, so it is written once, here, and shown
    every time rather than left as something a reader has to reconstruct from
    two config files.

    No ``Examples`` section: it prints a wide formatted table.
    """
    kappa = ours["kappa_i_rad_per_s"] + ours["kappa_c_rad_per_s"]
    print("=" * 78)
    print("PARAMETER TRANSLATION TABLE   (ours -> pyLLE)")
    print("=" * 78)
    rows = [
        ("fsr_hz",              f"{ours['fsr_hz']:.6e} Hz",     "D1_manual", f"{fwd['D1_manual_rad_per_s']:.6e} rad/s"),
        ("pump_wavelength_m",   f"{ours['pump_wavelength_m']:.6e} m", "sim['f_pmp']", f"{fwd['f_pmp_hz']:.6e} Hz"),
        ("pin_w",               f"{ours['pin_w']:.6e} W",       "sim['Pin']", f"{fwd['Pin_w']:.6e} W"),
        ("kappa_i_rad_per_s",   f"{ours['kappa_i_rad_per_s']:.6e} rad/s", "res['Qi']", f"{fwd['Qi']:.6e}"),
        ("kappa_c_rad_per_s",   f"{ours['kappa_c_rad_per_s']:.6e} rad/s", "res['Qc']", f"{fwd['Qc']:.6e}"),
        ("n2_m2_per_w",         f"{ours['n2_m2_per_w']:.6e} m2/W", "res['gamma']", f"{fwd['gamma_nlse_per_w_per_m']:.6e} 1/W/m"),
        ("effective_mode_area", f"{ours['effective_mode_area_m2']:.6e} m2", "(in res['gamma'])", "-"),
        ("gamma_LLE_per_J_per_s", f"{ours['gamma_LLE_per_J_per_s']:.6e} 1/J/s", "res['R'] (via n_g)", f"{fwd['R_m']:.6e} m"),
        ("n0",                  f"{ours['n0']:.6f}",            "res['n0'] (unused by kernel)", f"{fwd['n0']:.6f}"),
    ]
    print(f"{'ours':<24}{'value':<26}{'pyLLE':<26}{'value'}")
    print("-" * 78)
    for a, b, c, d in rows:
        print(f"{a:<24}{b:<26}{c:<26}{d}")
    print("-" * 78)
    print(f"derived n_g               = {fwd['n_g']:.9f}   (from gamma_LLE; not a config leaf)")
    print(f"derived L_cav             = {fwd['L_cav_m']:.9e} m")
    print(f"t_r                       = {fwd['t_r_s']:.9e} s     kappa*t_r = {kappa * fwd['t_r_s']:.6e}")
    print(f"phi_pmp                   = {fwd['phi_pmp_rad']:.9f} rad  (Finding 4: drive real +ve)")
    print(f"round-trip worst rel diff = {rt['_worst_rel_diff']:.3e}   (tolerance 1e-12)  PASS")
    print("=" * 78)
    print("CONVENTIONS APPLIED (from pyLLE 4.1.2 source; see module docstring):")
    print("  D_int  : rad/s on both sides, NO 2*pi factor                 [Finding 1]")
    print("  field  : E_ours = conj(A_pyLLE) * sqrt(t_r)                  [Findings 2,3]")
    print("  detune : delta_omega_pyLLE = -delta_omega_ours               [Finding 2]")
    print("  mu     : conj => spectrum mirrors mu -> -mu (handled in map) [Finding 2]")
    print("=" * 78)


# ==========================================================================
# Shared physics helpers (pure numpy: must import under numpy 1.x AND 2.x)
# ==========================================================================
def cw_lower_branch(delta_omega: float, kappa: float, kappa_c: float,
                    gamma: float, pin: float) -> complex:
    """Lowest-|E| root of the CW steady state, as the soliton's background.

    Solves  P*[(kappa/2)^2 + (delta_omega - gamma*P)^2] = kappa_c*Pin  for
    P = |E|^2 (J) and returns the complex amplitude on that branch. Solitons sit
    on the LOWER CW branch, which is why the smallest positive real root is
    taken (``analytic_cw.stable_branch`` takes the largest, for the upward-scan
    CW state -- a different question).

    Parameters
    ----------
    delta_omega : float
        Detuning ``omega_res - omega_pump`` [rad/s].
    kappa : float
        Total loss rate [rad/s].
    kappa_c : float
        Coupling rate [rad/s].
    gamma : float
        ``gamma_LLE`` [J^-1 s^-1].
    pin : float
        Pump power [W].

    Returns
    -------
    complex
        The CW amplitude ``E = sqrt(kappa_c*pin)/(kappa/2 + 1j*(dw - gamma*P))``
        [sqrt(J)], on the lower branch.

    Raises
    ------
    numpy.linalg.LinAlgError
        Propagated from ``numpy.roots`` for a degenerate cubic.

    Notes
    -----
    Returns the AMPLITUDE, not the power, because the soliton seed needs the
    background's phase as well as its magnitude: seeding a sech on a background
    of the wrong phase radiates a transient that the existence scan would then
    have to settle through.

    If no positive real root exists the largest root modulus is used as a
    fallback, so the seed builder degrades rather than raising at an operating
    point outside the CW solution set.

    Examples
    --------
    >>> e_cw = cw_lower_branch(2e9, 1.5e8, 1.2e8, 1.03e18, 0.214)
    >>> isinstance(e_cw, complex)
    True

    It satisfies the CW steady-state equation it was derived from:

    >>> p = abs(e_cw) ** 2
    >>> lhs = p * ((1.5e8 / 2) ** 2 + (2e9 - 1.03e18 * p) ** 2)
    >>> bool(abs(lhs / (1.2e8 * 0.214) - 1) < 1e-9)
    True
    """
    g, k2 = gamma, kappa / 2.0
    # gamma^2 P^3 - 2*gamma*dw P^2 + (k2^2 + dw^2) P - kappa_c*Pin = 0
    coeffs = [g * g, -2.0 * g * delta_omega, k2 * k2 + delta_omega ** 2, -kappa_c * pin]
    roots = np.roots(coeffs)
    real = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)) and r.real > 0)
    p = real[0] if real else float(np.max(np.abs(roots)).real)
    # E = sqrt(kappa_c*Pin) / (kappa/2 + i*(delta_omega - gamma*P))
    return math.sqrt(kappa_c * pin) / complex(k2, delta_omega - g * p)


def soliton_seed(n_modes: int, delta_omega: float, kappa: float, kappa_c: float,
                 gamma: float, pin: float, d2: float) -> np.ndarray:
    """Deterministic single-soliton seed in OUR convention, units sqrt(J).

    Standard DKS ansatz for  dE/dt = -(kappa/2)E - i*dw*E + i*(D2/2)d2E/dtheta^2
    + i*gamma|E|^2 E + F, on top of the lower CW branch::

        E(theta) = E_cw + A*sech(theta/w),  A = sqrt(2*dw/gamma), w = sqrt(D2/(2*dw))

    theta is the FFT fast-time grid theta_j = 2*pi*j/n_modes, wrapped to
    (-pi, pi] so the pulse sits at theta = 0. No randomness: the same array is
    handed to BOTH codes (pyLLE gets conj(E)/sqrt(t_r); Finding 5).

    Parameters
    ----------
    n_modes : int
        Number of modes and fast-time samples.
    delta_omega : float
        Detuning [rad/s]. Must be positive -- the red-detuned, soliton side.
    kappa : float
        Total loss rate [rad/s].
    kappa_c : float
        Coupling rate [rad/s].
    gamma : float
        ``gamma_LLE`` [J^-1 s^-1].
    pin : float
        Pump power [W].
    d2 : float
        Local second-order dispersion ``D_2`` [rad/s**2], which sets the
        soliton width.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_modes,)``, dtype ``complex128``, units sqrt(J).

    Raises
    ------
    ValueError
        If ``delta_omega <= 0``. A soliton ansatz has no meaning on the blue
        side, and silently returning a CW background would make an existence
        scan report a spurious lower edge.

    Notes
    -----
    The sech is evaluated in the OVERFLOW-SAFE form
    ``sech(x) = 2*exp(-|x|)/(1 + exp(-2|x|))``. At these detunings
    ``theta/width`` reaches about 1500, where a naive ``1/cosh`` overflows and
    returns EXACT zeros in the tails -- a hard-truncated seed whose
    discontinuity radiates. The safe form underflows smoothly to zero instead.

    No randomness anywhere: the identical array is handed to both codes, which
    is what makes a difference in their outputs attributable to the solvers.

    Examples
    --------
    >>> seed = soliton_seed(101, 2e9, 1.5e8, 1.2e8, 1.03e18, 0.214, 5e4)
    >>> print(seed.shape, seed.dtype)
    (101,) complex128

    The pulse sits at the start of the grid (theta = 0) and decays away from it:

    >>> import numpy as np
    >>> int(np.argmax(np.abs(seed)))
    0

    The blue side is refused, not silently accepted:

    >>> soliton_seed(101, -1.0, 1.5e8, 1.2e8, 1.03e18, 0.214, 5e4)
    Traceback (most recent call last):
        ...
    ValueError: soliton seed requires delta_omega > 0 (red-detuned side)
    """
    if delta_omega <= 0:
        raise ValueError("soliton seed requires delta_omega > 0 (red-detuned side)")
    theta = 2.0 * math.pi * np.arange(n_modes) / n_modes
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    amp = math.sqrt(2.0 * delta_omega / gamma)
    width = math.sqrt(abs(d2) / (2.0 * delta_omega))
    # Overflow-safe sech: at these detunings theta/width reaches ~1500, and a
    # naive 1/cosh overflows to give EXACT zeros in the tails, i.e. a hard-
    # truncated seed whose discontinuity radiates. sech(x) = 2e^-|x|/(1+e^-2|x|)
    # underflows smoothly to zero instead.
    x = np.abs(theta / width)
    sech = 2.0 * np.exp(-x) / (1.0 + np.exp(-2.0 * x))
    return cw_lower_branch(delta_omega, kappa, kappa_c, gamma, pin) + amp * sech


def build_pylle_dispfile(path: Path, mu: np.ndarray, d_int_ours: np.ndarray,
                         f_pmp_hz: float, d1: float, m0: int) -> None:
    """Write the ``mode, f_Hz`` CSV encoding the MIRRORED device (Finding 2b).

    pyLLE must integrate ``D_int_p(mu) = D_int_ours(-mu)``, so we synthesise
    resonance frequencies from our own D_int rather than reusing the raw
    measured CSV::

        omega_p(mu) = omega_pmp + D1*mu + D_int_ours(-mu)

    Column 0 is the ABSOLUTE azimuthal order ``m0 + mu``: pyLLE reads it as
    ``rM`` and uses it only for its neff/ng report, while D_int is rebuilt from
    column 1 and ``D1_manual`` (``_analyzedisp.py:100-106``), so the offset
    cannot perturb the physics. ``D_int_ours(0) == 0`` makes row ``mu = 0``
    land exactly on ``f_pmp``, which is how pyLLE locates the pump (Finding 8).

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV.
    mu : numpy.ndarray
        Mode numbers [dimensionless], symmetric about 0.
    d_int_ours : numpy.ndarray
        ``D_int(mu)`` [rad/s] in OUR convention, same shape as ``mu``.
    f_pmp_hz : float
        Pump frequency [Hz], which row ``mu = 0`` must land on exactly.
    d1 : float
        ``D_1`` [rad/s], setting the linear term of the synthesised
        resonances.
    m0 : int
        Absolute azimuthal order of the pump mode [dimensionless], so column 0
        is ``m0 + mu``.

    Returns
    -------
    None
        Writes the CSV; nothing is returned.

    Raises
    ------
    OSError
        If ``path`` cannot be written.
    ValueError
        If ``mu`` and ``d_int_ours`` have different lengths.

    Notes
    -----
    The mirror is applied HERE, by synthesising resonance frequencies from our
    own ``D_int`` reversed, rather than by reusing the raw measured CSV and
    flipping indices afterwards. That is what makes the sign convention a
    property of the file both codes read, instead of a correction applied to
    one code's output.

    No ``Examples`` section: it writes a file for the out-of-process pyLLE
    run to consume.
    """
    d_int_mirror = d_int_ours[::-1].copy()          # D_int(-mu) on a symmetric grid
    omega_p = 2.0 * math.pi * f_pmp_hz + d1 * mu + d_int_mirror
    with path.open("w") as fh:
        for m, w in zip(m0 + mu, omega_p):
            fh.write("%d,%.10f\n" % (int(m), float(w) / (2.0 * math.pi)))


def local_d2_from_dint(mu: np.ndarray, d_int: np.ndarray) -> float:
    """Fit the local second-order dispersion at the pump.

    Parameters
    ----------
    mu : numpy.ndarray
        Mode numbers [dimensionless].
    d_int : numpy.ndarray
        ``D_int(mu)`` [rad/s], same shape as ``mu``.

    Returns
    -------
    float
        ``D_2 = d**2(D_int)/dmu**2`` at ``mu = 0`` [rad/s**2], from a degree-6
        fit over ``5 < |mu| <= 600``.

    Raises
    ------
    ValueError
        Propagated from the polynomial fit if fewer points fall inside the
        window than the degree requires.

    Notes
    -----
    Uses the same defect-excluding window as
    :func:`~simulator.lle_solver.load_dint_grid`: the measured CSV carries a
    localized artefact for ``|mu| <= 5``, and a fit that includes it biases
    ``D_2`` -- which then sets the seed soliton's width, and so the initial
    condition both codes start from.

    Examples
    --------
    A pure quadratic recovers its own curvature exactly:

    >>> import numpy as np
    >>> mu = np.arange(-600, 601)
    >>> print(f"{local_d2_from_dint(mu, 0.5 * 5e4 * mu ** 2):.6e}")
    5.000000e+04
    """
    sel = (np.abs(mu) <= 600) & (np.abs(mu) > 5)
    poly = np.polynomial.Polynomial.fit(mu[sel].astype(float), d_int[sel], 6)
    return float(poly.deriv().deriv()(0.0))


def spectrum_dbc(field: np.ndarray) -> np.ndarray:
    """Compute the per-mode power spectrum in dB relative to the strongest line.

    Parameters
    ----------
    field : numpy.ndarray
        Intracavity field, shape ``(n_modes,)``, complex, units sqrt(J).

    Returns
    -------
    numpy.ndarray
        Shape ``(n_modes,)``, dtype ``float64``, units dBc, mu-centred. The
        strongest line is exactly 0 dBc.

    Raises
    ------
    ValueError
        Propagated from ``numpy.fft`` for an empty field.

    Notes
    -----
    ``fftshift(fft(x))/N``, so bin values are per-mode amplitudes and the mu
    axis is centred. The SAME estimator is applied to both codes' fields, which
    is what makes the comparison fair: each code's own plotting normalization
    is irrelevant because neither is used.

    Examples
    --------
    >>> import numpy as np
    >>> n = 401
    >>> tau = np.arange(n) - n // 2
    >>> dbc = spectrum_dbc((1.0 / np.cosh(tau / 8.0)).astype(complex) + 0.1)
    >>> print(dbc.shape, f"{dbc.max():.3f}")
    (401,) 0.000
    """
    lines = np.fft.fftshift(np.fft.fft(field)) / field.size
    power = np.abs(lines) ** 2
    return 10.0 * np.log10(np.maximum(power, 1e-300) / power.max())


def observables(field_ours: np.ndarray, t_r: float, mu: np.ndarray) -> dict:
    """Compute the single-run observables, identically for both codes.

    Parameters
    ----------
    field_ours : numpy.ndarray
        Final field in OUR convention, shape ``(n_modes,)``, complex, units
        sqrt(J). pyLLE's field must ALREADY have been mapped through
        ``conj(A)*sqrt(t_r)`` before arriving here.
    t_r : float
        Round-trip time [s].
    mu : numpy.ndarray
        Mode numbers [dimensionless], shape ``(n_modes,)``, mu-centred.

    Returns
    -------
    dict
        Dispersive-wave descriptors on each side --
        ``dw_peak_mu_red`` / ``dw_peak_mu_blue`` [mode number],
        ``dw_argmax_mu_*`` [mode number], ``dw_centroid_*`` [mode number],
        ``dw_peak_dbc_*`` [dBc] -- plus ``spectral_span_3db_modes`` [modes],
        ``soliton_peak_power_w`` [W] and ``comb_line_count_60dbc`` [count].

    Raises
    ------
    IndexError
        If ``mu`` and ``field_ours`` have different lengths.

    Notes
    -----
    The mu -> -mu mirror is applied BY CONSTRUCTION, upstream, rather than by
    fudging indices after the fact -- which is the difference between a
    convention that is stated once and one that has to be re-derived at every
    comparison.

    The 3 dB span excludes the pump bin. The pump stands far above every comb
    line, so a span measured against it collapses to the pump bin alone -- a
    span of 0 on both sides, and a vacuous pass. Excluding it measures the
    soliton's own envelope, which is both physically meaningful and a strictly
    harder test.

    Examples
    --------
    >>> import numpy as np
    >>> n = 401
    >>> mu = np.arange(n) - n // 2
    >>> tau = np.arange(n) - n // 2
    >>> obs = observables((1.0 / np.cosh(tau / 8.0)).astype(complex) + 0.1,
    ...                   t_r=4.0e-11, mu=mu)
    >>> "soliton_peak_power_w" in obs and "spectral_span_3db_modes" in obs
    True
    >>> bool(obs["soliton_peak_power_w"] > 0)
    True
    """
    dbc = spectrum_dbc(field_ours)

    # (1) DW peak mode indices. A dispersive wave is a genuine LOCAL MAXIMUM of
    # the envelope out in the wings, not simply the largest value inside the
    # search window -- taking a plain argmax over |mu| >= DW_CORE_EXCLUDE would
    # return the window's inner edge whenever no DW is excited, and then
    # "compare" two artefacts. So: smooth, require an interior local max, and
    # report None if the wings are featureless.
    win = 25
    kernel = np.ones(win) / win
    smooth = np.convolve(dbc, kernel, mode="same")
    guard = win + 1                       # convolution edge, not physics
    dw = {}
    for name, side in (("red", mu < 0), ("blue", mu > 0)):
        m = (np.abs(mu) >= DW_CORE_EXCLUDE) & side
        m[:guard] = False
        m[-guard:] = False
        idx_all = np.flatnonzero(m)
        # A real DW must (a) be a local maximum over a WINDOW, not a single
        # sample -- a bare i-1/i+1 test fires on the search window's own inner
        # boundary and on isolated wiggles in a monotonically decaying wing --
        # and (b) stand proud of its surroundings by DW_MIN_PROMINENCE_DB.
        # Without (a) and (b) a featureless wing yields a spurious "DW" at
        # exactly mu = +/-DW_CORE_EXCLUDE, which is an artefact of the search
        # window rather than a feature of the comb.
        floor = float(np.median(smooth[idx_all])) if idx_all.size else -np.inf
        peaks = [i for i in idx_all
                 if smooth[i] >= smooth[max(i - win, 0):i + win + 1].max()
                 and smooth[i] >= floor + DW_MIN_PROMINENCE_DB]
        if peaks:
            best = max(peaks, key=lambda i: smooth[i])
            # A DW is a BROAD resonance whose top is flat to a small fraction of
            # a dB over tens of modes, so a bare argmax is not a stable
            # estimator: two independently discretized codes agreeing to ~1e-3
            # dB can still put the argmax one mode apart. The power-weighted
            # centroid of the same bump is stable, and being the identical
            # functional applied to both spectra it keeps the exact-integer
            # requirement meaningful rather than loosening it. Both are
            # reported; the centroid is what the verdict uses.
            w = m & (np.abs(mu - mu[best]) <= DW_CENTROID_HALFWIDTH)
            lin = 10.0 ** (smooth[w] / 10.0)
            lin = lin - lin.min()                      # remove the local pedestal
            centroid = (float(np.sum(mu[w] * lin) / np.sum(lin))
                        if lin.sum() > 0 else float(mu[best]))
            dw[name] = {"mu": int(round(centroid)), "argmax_mu": int(mu[best]),
                        "centroid": centroid, "dbc": float(dbc[best])}
        else:
            dw[name] = {"mu": None, "argmax_mu": None, "centroid": None, "dbc": None}

    # (2) 3 dB spectral span of the COMB ENVELOPE, in mode numbers.
    # The pump line carries the CW background and stands far above every comb
    # line, so a 3 dB span measured against it collapses to the pump bin alone
    # (span 0 on both sides -- a vacuous pass). Excluding mu = 0 measures the
    # soliton's own spectral envelope, which is the physically meaningful width
    # and a strictly harder test.
    env = dbc.copy()
    env[mu == 0] = -np.inf
    above = np.flatnonzero(env >= env.max() - 3.0)
    span_3db = int(mu[above[-1]] - mu[above[0]]) if above.size else 0

    # (3) Soliton peak power in W (|E|^2/t_r converts sqrt(J) -> W; Finding 3).
    peak_power_w = float(np.max(np.abs(field_ours) ** 2) / t_r)

    # (5) Comb lines above -60 dBc.
    line_count = int(np.count_nonzero(dbc >= -60.0))

    return {
        "dw_peak_mu_red": dw["red"]["mu"],
        "dw_peak_mu_blue": dw["blue"]["mu"],
        "dw_argmax_mu_red": dw["red"]["argmax_mu"],
        "dw_argmax_mu_blue": dw["blue"]["argmax_mu"],
        "dw_centroid_red": dw["red"]["centroid"],
        "dw_centroid_blue": dw["blue"]["centroid"],
        "dw_peak_dbc_red": dw["red"]["dbc"],
        "dw_peak_dbc_blue": dw["blue"]["dbc"],
        "spectral_span_3db_modes": span_3db,
        "soliton_peak_power_w": peak_power_w,
        "comb_line_count_60dbc": line_count,
    }


def is_single_soliton(field_ours: np.ndarray, t_r: float, kappa: float,
                      gamma: float) -> tuple[bool, int, float]:
    """Classify a final state as a single localized soliton.

    Parameters
    ----------
    field_ours : numpy.ndarray
        Final field in OUR convention, shape ``(n_modes,)``, complex, units
        sqrt(J).
    t_r : float
        Round-trip time [s].
    kappa : float
        Total loss rate [rad/s]. Accepted for signature stability; the
        classification is threshold-free in ``kappa``.
    gamma : float
        ``gamma_LLE`` [J^-1 s^-1]. Likewise accepted and not used.

    Returns
    -------
    single : bool
        ``True`` iff exactly one connected component stands above the midpoint
        between background and peak.
    n_peaks : int
        The number of such components; 0 when the field is not localized at
        all.
    contrast : float
        ``max(power)/median(power)`` [dimensionless].

    Raises
    ------
    ValueError
        Propagated from ``numpy`` for an empty field.

    Notes
    -----
    "Localized" means the peak stands well above the field's OWN background
    floor -- the median power, not an absolute threshold -- so the test carries
    over unchanged between operating points and between codes. Counting
    connected components above the midpoint separates a single dissipative
    Kerr soliton from a multi-soliton state and from CW or chaos, without
    reference to either code's internals.

    The component count is CIRCULAR, matching the periodic fast-time domain: a
    pulse straddling the wrap point is one soliton, not two.

    Examples
    --------
    >>> import numpy as np
    >>> n = 401
    >>> tau = np.arange(n) - n // 2
    >>> single, n_peaks, contrast = is_single_soliton(
    ...     (1.0 / np.cosh(tau / 8.0)).astype(complex) + 1e-3,
    ...     t_r=4.0e-11, kappa=1.5e8, gamma=1.03e18)
    >>> single, n_peaks
    (True, 1)

    Two pulses are two, and a flat field is not localized at all:

    >>> two = ((1.0 / np.cosh((tau - 80) / 8.0))
    ...        + (1.0 / np.cosh((tau + 80) / 8.0))).astype(complex) + 1e-3
    >>> is_single_soliton(two, 4.0e-11, 1.5e8, 1.03e18)[:2]
    (False, 2)
    >>> is_single_soliton(np.ones(n, dtype=complex), 4.0e-11, 1.5e8, 1.03e18)[:2]
    (False, 0)
    """
    power = np.abs(field_ours) ** 2 / t_r                       # W
    peak, floor = float(power.max()), float(np.median(power))
    contrast = peak / max(floor, 1e-300)
    if contrast < 5.0:                                          # not localized
        return False, 0, contrast
    thresh = 0.5 * (peak + floor)
    above = power >= thresh
    # circular connected-component count
    n_peaks = int(np.count_nonzero(above & ~np.roll(above, 1)))
    return (n_peaks == 1), n_peaks, contrast


# ==========================================================================
# pyLLE worker — runs INSIDE the pyLLE interpreter (numpy<2, no jax)
# ==========================================================================
def _worker_main(argv: list[str]) -> int:
    import pyLLE

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--julia-bin", required=True)
    a = ap.parse_args(argv)

    spec = json.loads(Path(a.spec).read_text())
    mu_half = int(spec["mu_half"])
    n_modes = 2 * mu_half + 1
    t_r = float(spec["t_r_s"])

    # The dispfile is built by the orchestrator and already encodes the MIRRORED
    # dispersion required by Finding 2b.
    dispfile = spec["dispfile"]
    mu_csv = np.arange(-mu_half, mu_half + 1)

    res = {
        "R": spec["R_m"],
        "Qi": spec["Qi"],
        "Qc": spec["Qc"],
        "gamma": spec["gamma_nlse_per_w_per_m"],
        "dispfile": dispfile,
    }

    results = {}
    dint_saved = None
    for run in spec["runs"]:
        sim = {
            "Pin": [spec["Pin_w"]],
            "f_pmp": [spec["f_pmp_hz"]],
            "phi_pmp": [spec["phi_pmp_rad"]],
            "Tscan": float(run["n_roundtrips"]),
            "mu_sim": [-mu_half, mu_half],
            # mu_fit is PUMP-RELATIVE and pyLLE indexes its CSV as
            # ind_pmp + mu_fit[0] >= 0 (_analyzedisp.py:127), so the fit window
            # must stay inside the file on both sides of the pump mode.
            "mu_fit": [int(mu_csv.min()), int(mu_csv.max())],
            "D1_manual": spec["D1_manual_rad_per_s"],
            # Finding 2: pyLLE detuning is the NEGATIVE of ours.
            "domega_init": -float(run["delta_omega_start"]),
            "domega_end": -float(run["delta_omega_end"]),
            "num_probe": 200,
        }
        solver = pyLLE.LLEsolver(sim=sim, res=res)
        solver.Analyze(plot=False)

        d_int = np.asarray(solver._sim["Dint"], dtype=np.float64).ravel()
        if d_int.size != n_modes:
            raise RuntimeError(f"pyLLE Dint size {d_int.size} != expected {n_modes}")
        mu0 = (n_modes - 1) // 2
        # Finding 6: the kernel re-zeros at the domain centre; with ind_pmp = 0
        # that IS the pump mode, so the re-zero must be a no-op. Verify it.
        if abs(float(d_int[mu0])) > 1e-6 * max(1.0, float(np.max(np.abs(d_int)))):
            raise RuntimeError(
                f"pyLLE D_int is not zero at the domain centre (D_int[mu0]="
                f"{d_int[mu0]:.6e}); pump is not centred and Finding 6 would bite."
            )
        dint_saved = d_int

        # Finding 5 + Finding 2/3: identical deterministic seed, mapped into
        # pyLLE's convention  A = conj(E)/sqrt(t_r).
        seed_ours = np.asarray(run["seed_real"], dtype=np.float64) + 1j * np.asarray(
            run["seed_imag"], dtype=np.float64)
        solver._sim["DKS_init"] = np.conj(seed_ours) / math.sqrt(t_r)

        solver.Setup(verbose=False)
        solver.SolveTemporal(bin=a.julia_bin)
        solver.RetrieveData()

        u = solver._sol["u_probe"]           # (n_modes, num_probe), sqrt(W)
        final = np.asarray(u[:, -1], dtype=np.complex128)
        if not np.all(np.isfinite(final)):
            raise RuntimeError("pyLLE returned a non-finite field")
        results[run["name"]] = {
            "field_real": final.real.tolist(),
            "field_imag": final.imag.tolist(),
            "detuning_pylle_final": float(solver._sol["detuning"][-1]),
        }

    import scipy
    payload = {
        "d_int_rad_per_s": np.asarray(dint_saved).tolist(),
        "runs": results,
        "versions": {
            "pyLLE": getattr(pyLLE, "__version__", "4.1.2"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    Path(a.out).write_text(json.dumps(payload))
    print("WORKER_OK")
    return 0


# ==========================================================================
# Our solver
# ==========================================================================
def run_ours(delta_omega_ramp: np.ndarray, seed: np.ndarray, d_int_fftorder: np.ndarray,
             dev: dict, config_path: str, n_substeps: int = 1) -> np.ndarray:
    """Run this repository's solver over a detuning ramp from a given seed.

    Parameters
    ----------
    delta_omega_ramp : numpy.ndarray
        Detuning per round trip [rad/s], shape ``(t_slow,)``.
    seed : numpy.ndarray
        Initial field, shape ``(n_modes,)``, complex, units sqrt(J).
    d_int_fftorder : numpy.ndarray
        ``D_int(mu)`` [rad/s], shape ``(n_modes,)``, in FFT-bin order.
    dev : dict
        Device parameters from :func:`load_device`.
    config_path : str
        The DERIVED config from :func:`derived_config`.
    n_substeps : int, optional
        Strang sub-steps per round trip [dimensionless], default 1.

    Returns
    -------
    numpy.ndarray
        Final field, shape ``(n_modes,)``, complex, units sqrt(J).

    Raises
    ------
    AssertionError
        If ``delta_omega_eff`` deviates from the programmed ramp, meaning the
        thermo-optic shift is not off, or if the final field is non-finite.
    ValueError
        Propagated from the solver for a shape mismatch.

    Notes
    -----
    ``n_substeps = 1`` with ``fine_cadence_M = 1`` matches pyLLE's
    one-step-per-round-trip kernel EXACTLY. Matching the discretization is the
    point: two codes integrating the same equation with different step counts
    would differ for a reason that is nobody's error, and the resulting number
    would not test either implementation.

    No ``Examples`` section: it runs a multi-thousand-round-trip solve on a
    6601-mode grid.
    """
    import jax
    from simulator.lle_solver import solve_lle_ssfm_jax
    from simulator.noise_config import NoiseConfig

    kappa = dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]
    n_steps = int(delta_omega_ramp.size)
    out = solve_lle_ssfm_jax(
        pin=dev["pin_w"],
        delta_omega=np.asarray(delta_omega_ramp, dtype=float),
        t_slow=n_steps,
        beta=[0.0, 0.0],                      # unused: d_int_grid drives dispersion
        kappa=kappa,
        kappa_c=dev["kappa_c_rad_per_s"],
        rng_key=jax.random.PRNGKey(0),        # unused: every channel is off
        n_tau=int(seed.size),
        config_path=config_path,
        snapshot_interval=n_steps,
        e0_override=np.asarray(seed, dtype=np.complex128),
        d_int_grid=np.asarray(d_int_fftorder, dtype=float),
        n_substeps=int(n_substeps),
        fine_cadence_M=1,
        noise_config=NoiseConfig.all_off(),
    )
    eff = np.asarray(out["delta_omega_eff_history"], dtype=float).ravel()
    programmed = np.asarray(delta_omega_ramp, dtype=float).ravel()
    worst = float(np.max(np.abs(eff - programmed)))
    if worst > 1e-6 * max(1.0, float(np.max(np.abs(programmed)))):
        raise AssertionError(
            f"delta_omega_eff deviates from the programmed detuning by {worst:.3e} rad/s: "
            "the thermo-optic shift is not off, so the two codes are not at the same detuning."
        )
    return np.asarray(out["e_final"], dtype=np.complex128).reshape(-1)


# ==========================================================================
# Orchestration
# ==========================================================================
def _fmt_br(br, kappa: float) -> str:
    return "-" if br is None else f"[{br[0] / kappa:.3f},{br[1] / kappa:.3f}]"


def _br_tuple(br, kappa: float):
    return None if br is None else [br[0] / kappa, br[1] / kappa]


def _rel(a: float, b: float) -> float:
    """Symmetric relative difference, 0.0 when both values are 0.

    Parameters
    ----------
    a, b : float
        The two values, in any consistent unit.

    Returns
    -------
    float
        ``|a - b| / max(|a|, |b|)`` [dimensionless].

    Examples
    --------
    >>> _rel(2.0, 1.0), _rel(0.0, 0.0)
    (0.5, 0.0)
    """
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom if denom else 0.0


def _verdict(name: str, ours, pylle, tol: float, exact: bool = False) -> dict:
    """Compare one observable across the two codes and return a verdict record.

    Parameters
    ----------
    name : str
        Observable name, carried into the record.
    ours, pylle : float or None
        The two measured values, in the observable's own unit. ``None`` or NaN
        means the observable could not be measured on that side.
    tol : float
        Tolerance [dimensionless relative, or absolute when ``exact``].
    exact : bool, optional
        Compare absolutely rather than relatively, default ``False``.

    Returns
    -------
    dict
        ``observable``, both values, ``abs_diff``, ``rel_diff``, ``tolerance``
        and ``verdict`` (``'PASS'``, ``'FAIL'`` or ``'NOT_MEASURED'``).

    Notes
    -----
    An observable that could not be measured on one or both sides is
    ``NOT_MEASURED``, and that FAILS the overall run rather than passing
    silently. A comparison that did not happen is not a comparison that
    succeeded.

    Examples
    --------
    >>> _verdict("peak_power_w", 1.0, 1.0, 1e-9)["verdict"]
    'PASS'
    >>> _verdict("peak_power_w", 1.0, 2.0, 1e-9)["verdict"]
    'FAIL'
    >>> _verdict("peak_power_w", None, 2.0, 1e-9)["verdict"]
    'NOT_MEASURED'
    """
    if ours is None or pylle is None or (
            isinstance(ours, float) and math.isnan(ours)) or (
            isinstance(pylle, float) and math.isnan(pylle)):
        # Not silently a pass: an observable that could not be measured on one
        # or both sides is reported as such and fails the overall run.
        return {"observable": name, "ours": ours, "pylle": pylle,
                "abs_diff": None, "rel_diff": None, "tolerance": tol,
                "verdict": "NOT_MEASURED"}
    if exact:
        ok = int(ours) == int(pylle)
        rel = 0.0 if ok else 1.0
        absd = abs(int(ours) - int(pylle))
    else:
        rel = _rel(float(ours), float(pylle))
        absd = abs(float(ours) - float(pylle))
        ok = rel <= tol
    return {
        "observable": name, "ours": ours, "pylle": pylle,
        "abs_diff": absd, "rel_diff": rel, "tolerance": tol,
        "verdict": "PASS" if ok else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    """Run the v1 pyLLE cross-check from the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments; ``None`` (default) reads ``sys.argv[1:]``.
        ``--pylle-python`` and ``--julia-bin`` locate the pyLLE
        environment and default to the ``PYLLE_PYTHON`` and ``JULIA_BIN``
        environment variables.

    Returns
    -------
    int
        The process exit status: 0 when every observable agreed within its
        tolerance, non-zero otherwise.

    Raises
    ------
    SystemExit
        From ``argparse`` on a malformed command line or ``--help``.
    AssertionError
        From :func:`assert_round_trip` on a translation failure, or from
        :func:`run_ours` if the run left the programmed ramp.
    FileNotFoundError
        If the pyLLE interpreter, the Julia binary or the dispersion CSV
        cannot be found.

    Notes
    -----
    pyLLE runs OUT OF PROCESS, under its own interpreter: it pins
    ``numpy < 2`` and needs a Julia toolchain, so it cannot share an
    environment with this solver. ``--worker`` is intercepted in ``__main__``
    before this function is reached and is declared only so ``--help`` does not
    reject it.

    This is the v1 cross-check and is FROZEN; :mod:`validation.pylle_crosscheck_v2`
    supersedes it.

    No ``Examples`` section: it requires a provisioned pyLLE environment.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --worker is intercepted in __main__ before main() is reached; it is
    # declared here only so --help does not reject it.
    ap.add_argument("--pylle-python", default=os.environ.get("PYLLE_PYTHON"))
    ap.add_argument("--julia-bin", default=os.environ.get("JULIA_BIN"))
    ap.add_argument("--mu-half", type=int, default=MU_HALF)
    ap.add_argument("--roundtrips", type=int, default=N_ROUNDTRIPS_MAIN)
    ap.add_argument("--scan-roundtrips", type=int, default=N_ROUNDTRIPS_SCAN)
    ap.add_argument("--dw-start", type=float, default=1.0, help="ramp start, units of kappa")
    ap.add_argument("--dw-end", type=float, default=3.0, help="ramp end, units of kappa")
    ap.add_argument("--scan", type=str,
                    default="0.6,0.8,1.0,1.5,2.0,3.0,4.0,6.0,8.0,10.0,13.0,16.0",
                    help="existence-scan detunings, units of kappa")
    ap.add_argument("--converged-dw", type=float, default=8.0,
                    help="detuning (units of kappa) for the secondary comparison in "
                         "the regime where dt = 1 round trip is accurate")
    ap.add_argument("--convergence-levels", default="2,4",
                    help="extra n_substeps values for the Phase-7 convergence "
                         "attribution on our side (pyLLE cannot be refined); "
                         "empty string disables")
    ap.add_argument("--bisect-iters", type=int, default=5,
                    help="bisection steps per existence edge (per code)")
    ap.add_argument("--work-dir", default=None)
    args, _rest = ap.parse_known_args(argv)

    if not args.pylle_python or not args.julia_bin:
        print("ERROR: --pylle-python and --julia-bin (or PYLLE_PYTHON / JULIA_BIN) "
              "are required. pyLLE needs numpy<2 and a Julia backend and cannot "
              "share this repo's environment.", file=sys.stderr)
        return 2

    import tempfile
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="pylle_xcheck_"))
    work.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- PART 1: translation + round trip ----------------
    dev = load_device()
    # Finding 8: pin the pump to the mode the dispersion grid is referenced to.
    _csv = np.loadtxt(DISP_CSV, delimiter=",")
    _mu_csv = _csv[:, 0].astype(np.int64)
    f_pmp_disp = float(_csv[int(np.where(_mu_csv == 0)[0][0]), 1])
    pump_offset_fsr = (f_pmp_disp - C0 / dev["pump_wavelength_m"]) / dev["fsr_hz"]
    fwd = ours_to_pylle(dev, f_pmp_hz=f_pmp_disp)
    rt = assert_round_trip(dev, f_pmp_hz=f_pmp_disp)
    print_translation_table(dev, fwd, rt)
    print(f"pump pinned to CSV mu=0 resonance: f = {f_pmp_disp:.6e} Hz "
          f"(lambda = {C0 / f_pmp_disp:.9e} m), which is {pump_offset_fsr:+.2f} FSR from "
          f"the nominal c/pump_wavelength_m.  [Finding 8]")

    kappa = dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]
    t_r = fwd["t_r_s"]
    n_modes = 2 * args.mu_half + 1
    mu = np.arange(-args.mu_half, args.mu_half + 1)

    # ---------------- seeds and detuning trajectories ----------------
    # A first pyLLE call is needed to obtain ITS D_int (Finding 7) before the
    # seed width can be set from the same dispersion both codes will integrate.
    # Bootstrap the seed width from our own loader, then rebuild it from pyLLE's
    # D_int; the two agree to the refit tolerance reported below.
    from simulator.lle_solver import load_dint_grid
    ours_grid = load_dint_grid(n_modes, csv_path=DISP_CSV)
    d_int_ours_natural = np.fft.fftshift(np.asarray(ours_grid.grid, dtype=float))
    d2_local = local_d2_from_dint(mu, d_int_ours_natural)
    # Finding 9: pyLLE's t_r is 2*pi/D1, so D1 must be the CSV-fitted one that
    # load_dint_grid used, and our t_r must follow it.
    d1_csv = float(ours_grid.d1)
    fwd["D1_manual_rad_per_s"] = d1_csv
    fsr_used = d1_csv / (2.0 * math.pi)
    t_r = 1.0 / fsr_used
    fwd["t_r_s"] = t_r
    print(f"\nlocal D2 from measured dispersion = {d2_local:.6e} rad/s^2")
    print(f"D1 (CSV fit) = {d1_csv:.9e} rad/s -> FSR = {fsr_used / 1e9:.6f} GHz "
          f"(config leaf {dev['fsr_hz'] / 1e9:.6f} GHz); t_r = {t_r:.9e} s   [Finding 9]")

    dw_start = args.dw_start * kappa
    dw_end = args.dw_end * kappa
    scan_dws = [float(x) * kappa for x in args.scan.split(",")]

    seed_main = soliton_seed(n_modes, dw_start, kappa, dev["kappa_c_rad_per_s"],
                             dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)
    runs = [{
        "name": "main",
        "n_roundtrips": args.roundtrips,
        "delta_omega_start": dw_start,
        "delta_omega_end": dw_end,
        "seed_real": seed_main.real.tolist(),
        "seed_imag": seed_main.imag.tolist(),
    }]
    for dw in scan_dws:
        s = soliton_seed(n_modes, dw, kappa, dev["kappa_c_rad_per_s"],
                         dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)
        runs.append({
            "name": f"scan_{dw / kappa:.3f}",
            "n_roundtrips": args.scan_roundtrips,
            "delta_omega_start": dw, "delta_omega_end": dw,   # constant hold
            "seed_real": s.real.tolist(), "seed_imag": s.imag.tolist(),
        })

    # Finding 2b: pyLLE integrates the MIRRORED dispersion.
    dispfile = work / "pylle_dispfile_mirrored.csv"
    m0_abs = int(round(dev["n0"] * fwd["L_cav_m"] / dev["pump_wavelength_m"]))
    build_pylle_dispfile(dispfile, mu, d_int_ours_natural, f_pmp_disp, d1_csv, m0_abs)

    spec_base = {
        "mu_half": args.mu_half, "t_r_s": t_r, "dispfile": str(dispfile),
        "pump_wavelength_m": dev["pump_wavelength_m"], "n0": dev["n0"],
        **{k: fwd[k] for k in ("R_m", "Qi", "Qc", "gamma_nlse_per_w_per_m", "Pin_w",
                               "f_pmp_hz", "phi_pmp_rad", "D1_manual_rad_per_s", "L_cav_m")},
    }
    env = dict(os.environ, MPLBACKEND="Agg")

    def call_pylle(run_list: list[dict], tag: str) -> dict:
        """Run the pyLLE worker out-of-process on ``run_list`` (Julia backend)."""
        spec_local = dict(spec_base, runs=run_list)
        sp, op = work / f"spec_{tag}.json", work / f"pylle_out_{tag}.json"
        sp.write_text(json.dumps(spec_local))
        print(f"[pyLLE:{tag}] {len(run_list)} run(s), {n_modes} modes", flush=True)
        proc = subprocess.run(
            [args.pylle_python, str(Path(__file__).resolve()), "--worker",
             "--spec", str(sp), "--out", str(op), "--julia-bin", args.julia_bin],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0 or "WORKER_OK" not in (proc.stdout or ""):
            print("\n".join((proc.stdout or "").splitlines()[-25:]))
            print((proc.stderr or "")[-4000:], file=sys.stderr)
            raise RuntimeError(f"pyLLE worker failed (returncode {proc.returncode})")
        return json.loads(op.read_text())

    # ---------------- run pyLLE (Julia backend) ----------------
    pl = call_pylle(runs, "main")

    # ---------------- dispersion: use pyLLE's own D_int on both sides -------
    # pyLLE's D_int is the MIRROR of ours (Finding 2b); un-mirror it, and use
    # THAT as our solver's dispersion so both codes integrate one identical
    # array and the spline-refit difference (Finding 7) cancels exactly.
    d_int_pylle_natural = np.asarray(pl["d_int_rad_per_s"], dtype=float)
    d_int_unmirrored = d_int_pylle_natural[::-1].copy()
    scale = float(np.max(np.abs(d_int_ours_natural)))
    refit_rel = float(np.max(np.abs(d_int_unmirrored - d_int_ours_natural)) / scale)
    print(f"\n[dispersion] pyLLE refit (un-mirrored) vs our loader: max rel diff = "
          f"{refit_rel:.3e}  (pyLLE's own array is used by BOTH codes)")
    if refit_rel > 1e-6:
        raise AssertionError(
            f"pyLLE's D_int does not reproduce ours after un-mirroring "
            f"(max rel diff {refit_rel:.3e}). The dispersion mirror of Finding 2b "
            "is mis-applied, or D1/pump referencing disagrees."
        )
    d_int_fftorder = np.fft.ifftshift(d_int_unmirrored)
    max_dint_tr = float(np.max(np.abs(d_int_pylle_natural)) * t_r)

    # ---------------- run ours ----------------
    cfg = derived_config(fsr_used, work)
    print(f"[ours] derived config (dn_dT_per_k=0, fsr_hz={fsr_used:.6e}) -> {cfg}")

    ramp = np.linspace(dw_start, dw_end, args.roundtrips)
    print(f"[ours] main run: {args.roundtrips} round trips, "
          f"delta_omega {args.dw_start:.2f}k -> {args.dw_end:.2f}k")
    e_ours_main = run_ours(ramp, seed_main, d_int_fftorder, dev, cfg)

    def pylle_field_in_our_convention(name: str) -> np.ndarray:
        r = pl["runs"][name]
        a = np.asarray(r["field_real"], float) + 1j * np.asarray(r["field_imag"], float)
        return np.conj(a) * math.sqrt(t_r)         # Findings 2 & 3

    e_pylle_main = pylle_field_in_our_convention("main")

    obs_ours = observables(e_ours_main, t_r, mu)
    obs_pylle = observables(e_pylle_main, t_r, mu)
    print(f"[main] DW levels  ours: red {obs_ours['dw_peak_mu_red']}@"
          f"{obs_ours['dw_peak_dbc_red']} dBc, blue {obs_ours['dw_peak_mu_blue']}@"
          f"{obs_ours['dw_peak_dbc_blue']} dBc")
    print(f"[main] DW levels pylle: red {obs_pylle['dw_peak_mu_red']}@"
          f"{obs_pylle['dw_peak_dbc_red']} dBc, blue {obs_pylle['dw_peak_mu_blue']}@"
          f"{obs_pylle['dw_peak_dbc_blue']} dBc")

    # ---------------- convergence attribution (Phase 7) ----------------
    # pyLLE's step is NOT refinable: ComputeLLE.jl:123 hardwires dt = 1 round
    # trip, and :278 overrides the CLI nonlinear tolerance with 1e-2. So the
    # only convergence handle is ours. Re-running our side at n_substeps > 1
    # measures how far the SHARED one-step-per-round-trip discretization is
    # from converged; if that drift is comparable to the cross-code gap, the
    # gap is discretization, not translation.
    conv = {"note": "pyLLE cannot be refined: dt=1 round trip is hardwired "
                    "(ComputeLLE.jl:123) and its nonlinear-iteration tolerance is "
                    "fixed at 1e-2 (:278). Refinement is on our side only.",
            "pylle_refinable": False, "levels": {}}
    if args.convergence_levels:
        for nsub in [int(x) for x in args.convergence_levels.split(",") if int(x) > 1]:
            e_ref = run_ours(ramp, seed_main, d_int_fftorder, dev, cfg, n_substeps=nsub)
            o_ref = observables(e_ref, t_r, mu)
            conv["levels"][str(nsub)] = {
                "soliton_peak_power_w": o_ref["soliton_peak_power_w"],
                "comb_line_count_60dbc": o_ref["comb_line_count_60dbc"],
                "spectral_span_3db_modes": o_ref["spectral_span_3db_modes"],
                "dw_peak_mu_red": o_ref["dw_peak_mu_red"],
                "rel_drift_peak_power_vs_nsub1": _rel(
                    o_ref["soliton_peak_power_w"], obs_ours["soliton_peak_power_w"]),
                "rel_drift_line_count_vs_nsub1": _rel(
                    float(o_ref["comb_line_count_60dbc"]),
                    float(obs_ours["comb_line_count_60dbc"])),
            }
            print(f"[convergence] ours n_substeps={nsub}: peakP="
                  f"{o_ref['soliton_peak_power_w']:.3f} W (drift vs nsub=1 "
                  f"{conv['levels'][str(nsub)]['rel_drift_peak_power_vs_nsub1'] * 100:.2f}%), "
                  f"lines={o_ref['comb_line_count_60dbc']}", flush=True)

    # ---------------- existence range ----------------
    print(f"[ours] existence scan: {len(scan_dws)} detunings x {args.scan_roundtrips} round trips")
    surv_ours, surv_pylle = [], []
    scan_fields: dict[float, tuple] = {}
    for dw in scan_dws:
        name = f"scan_{dw / kappa:.3f}"
        s = soliton_seed(n_modes, dw, kappa, dev["kappa_c_rad_per_s"],
                         dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)
        e_o = run_ours(np.full(args.scan_roundtrips, dw), s, d_int_fftorder, dev, cfg)
        e_p = pylle_field_in_our_convention(name)
        ok_o, n_o, c_o = is_single_soliton(e_o, t_r, kappa, dev["gamma_LLE_per_J_per_s"])
        ok_p, n_p, c_p = is_single_soliton(e_p, t_r, kappa, dev["gamma_LLE_per_J_per_s"])
        surv_ours.append(ok_o)
        surv_pylle.append(ok_p)
        scan_fields[dw / kappa] = (e_o, e_p, ok_o and ok_p)
        print(f"   dw/k={dw / kappa:6.3f}  ours: single={ok_o} peaks={n_o} contrast={c_o:8.1f}"
              f"   pylle: single={ok_p} peaks={n_p} contrast={c_p:8.1f}")

    # -- refine both edges by bisection --------------------------------------
    # The coarse grid only locates each edge to within one scan interval, which
    # is far wider than the 5% tolerance; comparing coarse grid points would let
    # the two codes "agree" trivially. Bisect each edge for each code
    # INDEPENDENTLY (they need not bracket identically) and batch the pyLLE
    # midpoints, one worker call per iteration.
    def brackets(surv):
        idx = [i for i, s in enumerate(surv) if s]
        if not idx:
            return None, None
        i0, i1 = idx[0], idx[-1]
        low = (scan_dws[i0 - 1], scan_dws[i0]) if i0 > 0 else None      # (dead, alive)
        high = (scan_dws[i1], scan_dws[i1 + 1]) if i1 + 1 < len(scan_dws) else None
        return low, high                                                # (alive, dead)

    br_lo_o, br_hi_o = brackets(surv_ours)
    br_lo_p, br_hi_p = brackets(surv_pylle)

    def survives_ours(dw: float) -> bool:
        s = soliton_seed(n_modes, dw, kappa, dev["kappa_c_rad_per_s"],
                         dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)
        e = run_ours(np.full(args.scan_roundtrips, dw), s, d_int_fftorder, dev, cfg)
        return is_single_soliton(e, t_r, kappa, dev["gamma_LLE_per_J_per_s"])[0]

    for it in range(args.bisect_iters):
        mid_lo_p = 0.5 * (br_lo_p[0] + br_lo_p[1]) if br_lo_p else None
        mid_hi_p = 0.5 * (br_hi_p[0] + br_hi_p[1]) if br_hi_p else None
        batch, names = [], {}
        for key, dwm in (("lo", mid_lo_p), ("hi", mid_hi_p)):
            if dwm is None:
                continue
            nm = f"bis{it}_{key}"
            names[key] = nm
            s = soliton_seed(n_modes, dwm, kappa, dev["kappa_c_rad_per_s"],
                             dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)
            batch.append({"name": nm, "n_roundtrips": args.scan_roundtrips,
                          "delta_omega_start": dwm, "delta_omega_end": dwm,
                          "seed_real": s.real.tolist(), "seed_imag": s.imag.tolist()})
        if batch:
            got = call_pylle(batch, f"bisect{it}")
            for key, nm in names.items():
                r = got["runs"][nm]
                a = np.asarray(r["field_real"], float) + 1j * np.asarray(r["field_imag"], float)
                alive = is_single_soliton(np.conj(a) * math.sqrt(t_r), t_r, kappa,
                                          dev["gamma_LLE_per_J_per_s"])[0]
                if key == "lo":
                    br_lo_p = (br_lo_p[0], mid_lo_p) if alive else (mid_lo_p, br_lo_p[1])
                else:
                    br_hi_p = (mid_hi_p, br_hi_p[1]) if alive else (br_hi_p[0], mid_hi_p)
        if br_lo_o:
            m = 0.5 * (br_lo_o[0] + br_lo_o[1])
            br_lo_o = (br_lo_o[0], m) if survives_ours(m) else (m, br_lo_o[1])
        if br_hi_o:
            m = 0.5 * (br_hi_o[0] + br_hi_o[1])
            br_hi_o = (m, br_hi_o[1]) if survives_ours(m) else (br_hi_o[0], m)
        print(f"   [bisect {it}] ours lo={_fmt_br(br_lo_o, kappa)} hi={_fmt_br(br_hi_o, kappa)}"
              f" | pylle lo={_fmt_br(br_lo_p, kappa)} hi={_fmt_br(br_hi_p, kappa)}", flush=True)

    def edge_lo(br, surv):
        if br is not None:
            return br[1] / kappa                       # tightest surviving point
        idx = [i for i, s in enumerate(surv) if s]
        return scan_dws[idx[0]] / kappa if idx else float("nan")

    def edge_hi(br, surv):
        if br is not None:
            return br[0] / kappa
        idx = [i for i, s in enumerate(surv) if s]
        return scan_dws[idx[-1]] / kappa if idx else float("nan")

    lo_o, hi_o = edge_lo(br_lo_o, surv_ours), edge_hi(br_hi_o, surv_ours)
    lo_p, hi_p = edge_lo(br_lo_p, surv_pylle), edge_hi(br_hi_p, surv_pylle)
    edge_res = {
        "ours_lower_bracket_over_kappa": _br_tuple(br_lo_o, kappa),
        "ours_upper_bracket_over_kappa": _br_tuple(br_hi_o, kappa),
        "pylle_lower_bracket_over_kappa": _br_tuple(br_lo_p, kappa),
        "pylle_upper_bracket_over_kappa": _br_tuple(br_hi_p, kappa),
        "bisect_iters": args.bisect_iters,
    }

    # -- secondary comparison in the CONVERGED regime -------------------------
    # The primary point is chosen so the dispersive waves are excited, which on
    # this device needs a short soliton and therefore a Kerr phase of ~0.4 rad
    # per round trip -- where one step per round trip is NOT converged (see
    # convergence_attribution). To separate "the translation is wrong" from
    # "the shared step is coarse", the same observables are also compared at a
    # low-detuning point where the shared step IS accurate. These runs already
    # exist: they are existence-scan points, so this costs nothing extra.
    secondary = None
    if scan_fields:
        target = args.converged_dw
        pick = min(scan_fields, key=lambda k: abs(k - target))
        e_o2, e_p2, both_alive = scan_fields[pick]
        if both_alive:
            so2, sp2 = observables(e_o2, t_r, mu), observables(e_p2, t_r, mu)
            secondary = {
                "delta_omega_over_kappa": pick,
                "kerr_phase_per_roundtrip_rad": 2.0 * pick * kappa * t_r,
                "note": "same observables at a detuning where dt = 1 round trip is "
                        "accurate; DWs are not excited here, so the DW indices are "
                        "expected to be absent rather than matching",
                "ours": so2, "pylle": sp2,
                "rel_diff": {
                    "soliton_peak_power_w": _rel(so2["soliton_peak_power_w"],
                                                 sp2["soliton_peak_power_w"]),
                    "comb_line_count_60dbc": _rel(float(so2["comb_line_count_60dbc"]),
                                                  float(sp2["comb_line_count_60dbc"])),
                    "spectral_span_3db_modes": _rel(float(so2["spectral_span_3db_modes"]),
                                                    float(sp2["spectral_span_3db_modes"])),
                },
            }
            r = secondary["rel_diff"]
            print(f"\n[secondary @ dw={pick:.3f}k, Kerr phase "
                  f"{secondary['kerr_phase_per_roundtrip_rad']:.4f} rad/rt] "
                  f"peakP {r['soliton_peak_power_w'] * 100:.3f}%  "
                  f"lines {r['comb_line_count_60dbc'] * 100:.3f}%  "
                  f"span {r['spectral_span_3db_modes'] * 100:.3f}%")

    # ---------------- verdicts ----------------
    checks = [
        _verdict("dw_peak_mode_index_red", obs_ours["dw_peak_mu_red"],
                 obs_pylle["dw_peak_mu_red"], TOLERANCES["dw_peak_mode_indices"], exact=True),
        _verdict("dw_peak_mode_index_blue", obs_ours["dw_peak_mu_blue"],
                 obs_pylle["dw_peak_mu_blue"], TOLERANCES["dw_peak_mode_indices"], exact=True),
        _verdict("spectral_span_3db_modes", obs_ours["spectral_span_3db_modes"],
                 obs_pylle["spectral_span_3db_modes"], TOLERANCES["spectral_span_3db"]),
        _verdict("soliton_peak_power_w", obs_ours["soliton_peak_power_w"],
                 obs_pylle["soliton_peak_power_w"], TOLERANCES["soliton_peak_power"]),
        _verdict("existence_edge_lower_dw_over_kappa", lo_o, lo_p,
                 TOLERANCES["existence_range_edges"]),
        _verdict("existence_edge_upper_dw_over_kappa", hi_o, hi_p,
                 TOLERANCES["existence_range_edges"]),
        _verdict("comb_line_count_60dbc", obs_ours["comb_line_count_60dbc"],
                 obs_pylle["comb_line_count_60dbc"], TOLERANCES["comb_line_count_60dbc"]),
    ]
    overall = all(c["verdict"] == "PASS" for c in checks)

    print("\n" + "=" * 100)
    print("CROSS-CODE OBSERVABLE COMPARISON")
    print("=" * 100)
    print(f"{'observable':<40}{'ours':>16}{'pyLLE':>16}{'abs':>12}{'rel':>10}{'tol':>8}  verdict")
    print("-" * 100)
    def _cell(v, spec_fmt):
        return format(v, spec_fmt) if isinstance(v, (int, float)) else str(v)

    for c in checks:
        o = f"{c['ours']:.6g}" if isinstance(c["ours"], float) else str(c["ours"])
        p = f"{c['pylle']:.6g}" if isinstance(c["pylle"], float) else str(c["pylle"])
        print(f"{c['observable']:<40}{o:>16}{p:>16}"
              f"{_cell(c['abs_diff'], '>12.4g')}{_cell(c['rel_diff'], '>10.4g')}"
              f"{c['tolerance']:>8.3g}  {c['verdict']}")
    print("-" * 100)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 100)

    # ---------------- provenance JSON ----------------
    import platform
    import subprocess as _sp
    try:
        commit = _sp.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    import jax
    julia_ver = _sp.run([args.julia_bin, "--version"], capture_output=True, text=True).stdout.strip()

    payload = {
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "versions": {
            "ours": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "jax": jax.__version__},
            "pylle_env": pl["versions"],
            "julia": julia_ver,
        },
        "physical_parameters": dev,
        "translation": {k: v for k, v in fwd.items()},
        "translation_round_trip": rt,
        "conventions": {
            "d_int_units": "rad/s on both sides (Finding 1)",
            "field_map": "E_ours = conj(A_pyLLE)*sqrt(t_r)  (Findings 2,3)",
            "detuning_map": "delta_omega_pylle = -delta_omega_ours  (Finding 2)",
            "mode_index_map": "conjugation mirrors mu -> -mu; applied via the field map",
            "drive_phase": "phi_pmp = -pi/2 makes pyLLE's drive real positive (Finding 4)",
            "dispersion_source": "pyLLE's own refit D_int fed to BOTH codes (Finding 7)",
            "dispersion_mirror_applied": True,
            "dispersion_mirror_reason": "D_int_pyLLE(mu) = D_int_ours(-mu), forced by "
                                        "the conjugate field map (Finding 2b)",
            "dispersion_refit_max_rel_diff_vs_our_loader": refit_rel,
            "pump_reference": "sim['f_pmp'] pinned to the CSV mu=0 resonance (Finding 8)",
            "pump_offset_from_nominal_fsr": pump_offset_fsr,
            "f_pmp_hz_used": f_pmp_disp,
            "f_pmp_hz_nominal_c_over_lambda": C0 / dev["pump_wavelength_m"],
        },
        "frame_and_timebase": {
            "note": "Finding 9: pyLLE hardwires t_r = 2*pi/D1, so one D1 serves "
                    "both the co-moving frame and the timebase.",
            "D1_rad_per_s": d1_csv,
            "fsr_hz_used": fsr_used,
            "fsr_hz_config_leaf": dev["fsr_hz"],
            "fsr_rel_diff": abs(fsr_used - dev["fsr_hz"]) / dev["fsr_hz"],
            "t_r_s": t_r,
            "kappa_t_r": kappa * t_r,
        },
        "discretization": {
            "n_modes": n_modes, "mu_min": int(mu[0]), "mu_max": int(mu[-1]),
            "n_roundtrips_main": args.roundtrips,
            "n_roundtrips_scan": args.scan_roundtrips,
            "n_substeps": 1, "fine_cadence_M": 1,
            "step_note": "one step per round trip on both sides (pyLLE dt=1 is hardwired)",
            "max_abs_Dint_t_r_rad": max_dint_tr,
        },
        "initialization": {
            "seed": "deterministic sech DKS ansatz on the lower CW branch",
            "seed_amplitude_J": "A^2 = 2*delta_omega/gamma_LLE",
            "seed_width_theta": "w = sqrt(D2/(2*delta_omega))",
            "local_D2_rad_per_s2": d2_local,
            "randomness": "none on either side (Finding 5)",
            "noise": "NoiseConfig.all_off() + dn_dT_per_k=0; pyLLE 4.1.2 Noise() is "
                     "commented out at its call site",
        },
        "detuning": {
            "ramp_start_over_kappa": args.dw_start,
            "ramp_end_over_kappa": args.dw_end,
            "kappa_rad_per_s": kappa,
            "final_delta_omega_ours_rad_per_s": dw_end,
            "final_delta_omega_pylle_rad_per_s": -dw_end,
            "scan_over_kappa": [d / kappa for d in scan_dws],
            "survival_ours": surv_ours,
            "survival_pylle": surv_pylle,
            "edge_refinement": edge_res,
        },
        "observables": {"ours": obs_ours, "pylle": obs_pylle},
        "convergence_attribution": conv,
        "secondary_converged_regime": secondary,
        "checks": checks,
        "tolerances": TOLERANCES,
        "overall": "PASS" if overall else "FAIL",
    }
    json_path = RESULTS_DIR / "pylle_crosscheck.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {json_path}")

    # Raw final fields, so any disagreement can be re-diagnosed from data
    # without re-running the Julia backend.
    npz_path = RESULTS_DIR / "pylle_crosscheck_fields.npz"
    np.savez_compressed(
        npz_path, mu=mu, d_int_rad_per_s=d_int_unmirrored,
        field_ours=e_ours_main, field_pylle_mapped=e_pylle_main,
        spectrum_dbc_ours=spectrum_dbc(e_ours_main),
        spectrum_dbc_pylle=spectrum_dbc(e_pylle_main),
    )
    print(f"wrote {npz_path}")

    make_figure(mu, e_ours_main, e_pylle_main, t_r, dev, args, overall,
                obs_ours=obs_ours, obs_pylle=obs_pylle,
                kerr_phase=2.0 * dw_end * t_r)
    return 0 if overall else 1


def make_figure(mu, e_ours, e_pylle, t_r, dev, args, overall,
                obs_ours=None, obs_pylle=None, kerr_phase=None) -> None:
    """Draw the two-panel comparison figure and write it to disk.

    Parameters
    ----------
    mu : numpy.ndarray
        Mode numbers [dimensionless].
    e_ours, e_pylle : numpy.ndarray
        Final fields from each code, in OUR convention, complex, units sqrt(J).
    t_r : float
        Round-trip time [s].
    dev : dict
        Device parameters, for the annotation.
    args : argparse.Namespace
        Parsed command line, supplying the output path and run settings.
    overall : str
        The overall verdict, shown on the figure.
    obs_ours, obs_pylle : dict or None, optional
        Observable dicts from :func:`observables`, annotated when present.
    kerr_phase : float or None, optional
        Kerr phase per round trip [rad], annotated when present.

    Returns
    -------
    None
        Writes the figure to the path in ``args``.

    Raises
    ------
    OSError
        If the output path cannot be written.
    ImportError
        If matplotlib is unavailable.

    Notes
    -----
    Spectrum and waveform overlaid on the same axes, with the residual in its
    own sub-panel: a difference is only meaningful next to the thing it is a
    difference of, and two side-by-side panels make a small disagreement
    invisible.

    Uses the ``Agg`` backend explicitly so the figure renders identically
    headless.

    No ``Examples`` section: it needs both codes' fields and writes a file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_o, s_p = spectrum_dbc(e_ours), spectrum_dbc(e_pylle)
    n = e_ours.size
    theta = (2.0 * math.pi * np.arange(n) / n + math.pi) % (2.0 * math.pi) - math.pi
    order = np.argsort(theta)
    p_o = np.abs(e_ours) ** 2 / t_r
    p_p = np.abs(e_pylle) ** 2 / t_r

    fig, ax = plt.subplots(3, 1, figsize=(11, 11),
                           gridspec_kw={"height_ratios": [3, 3, 2]})
    kp = f"; Kerr phase {kerr_phase:.3f} rad/round trip" if kerr_phase else ""
    fig.suptitle(
        f"soliton-control vs pyLLE (Julia backend) — SiN, measured dispersion\n"
        f"pyLLE mapped through E = conj(A)·√t_r, δω_pyLLE = −δω_ours, "
        f"D_int mirrored μ→−μ\n"
        f"δω {args.dw_start:g}κ→{args.dw_end:g}κ{kp}   "
        f"[OVERALL {'PASS' if overall else 'FAIL'}]", fontsize=11)

    ax[0].plot(mu, s_o, lw=0.8, label="ours (soliton-control)")
    ax[0].plot(mu, s_p, lw=0.8, ls="--", label="pyLLE (conjugate-mapped)")
    ax[0].axhline(-60, color="grey", lw=0.6, ls=":", label="−60 dBc")
    for src, colour in ((obs_ours, "tab:blue"), (obs_pylle, "tab:orange")):
        if not src:
            continue
        for key in ("dw_peak_mu_red", "dw_peak_mu_blue"):
            if src.get(key) is not None:
                ax[0].axvline(src[key], color=colour, lw=0.7, alpha=0.5, ls="-.")
    if obs_ours and obs_ours.get("dw_peak_mu_red") is not None:
        ax[0].annotate("DW", xy=(obs_ours["dw_peak_mu_red"], -55),
                       xytext=(obs_ours["dw_peak_mu_red"], -30),
                       ha="center", fontsize=9,
                       arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0].set_ylabel("power (dBc)")
    ax[0].set_xlabel("mode index μ  (ours convention; pyLLE mirrored μ→−μ)")
    ax[0].set_ylim(-140, 5)
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title("Spectrum", fontsize=10, loc="left")

    # The soliton is ~1e-3 rad wide out of 2*pi, so the full round trip renders
    # as one spike; and the two codes park their soliton at DIFFERENT theta
    # (their composition order gives slightly different group velocity, so the
    # pulse drifts at a slightly different rate). Every observable compared
    # here is translation-invariant -- |spectrum|, peak power, line count -- so
    # that offset changes none of them; but plotting raw theta would show two
    # separated pulses and say nothing about SHAPE. Each trace is therefore
    # centred on its own peak and the measured offset is reported in the label.
    dtheta = 2 * math.pi / n
    i_o, i_p = int(np.argmax(p_o)), int(np.argmax(p_p))
    fwhm = max(float(np.count_nonzero(p_o >= 0.5 * p_o.max())) * dtheta, dtheta)
    offset = (theta[i_p] - theta[i_o] + math.pi) % (2 * math.pi) - math.pi
    rel = (np.arange(n) - n // 2) * dtheta
    ax[1].plot(rel, np.roll(p_o, n // 2 - i_o), lw=0.9, label="ours")
    ax[1].plot(rel, np.roll(p_p, n // 2 - i_p), lw=0.9, ls="--", label="pyLLE")
    ax[1].set_xlim(-12 * fwhm, 12 * fwhm)
    ax[1].set_ylabel("intracavity power (W)")
    ax[1].set_xlabel("fast time θ − θ_peak (rad), each trace centred on its own peak")
    ax[1].set_title(
        f"Waveform  —  the two solitons sit {offset:+.2e} rad apart in θ; "
        "every compared observable is translation-invariant",
        fontsize=9, loc="left")
    ax[1].legend(loc="upper right", fontsize=9)

    ax[2].plot(mu, s_o - s_p, lw=0.7, color="crimson")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("Δ spectrum (dB)")
    ax[2].set_xlabel("mode index μ")
    ax[2].set_title("Residual (ours − pyLLE)", fontsize=10, loc="left")

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RESULTS_DIR / "pylle_crosscheck.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        argv = [a for a in sys.argv[1:] if a != "--worker"]
        sys.exit(_worker_main(argv))
    sys.exit(main())
