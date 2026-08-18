# Validation

What is verified, to what tolerance, and which command proves it.

This page is the **entry point**. It gives the three tiers, the measured numbers and the
command per claim. Three companion documents carry the detail:

| Document | What it holds |
|---|---|
| [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md) | The single top-level answer to "how validated is this, and to what". Configuration-coverage table, every open item, when to revisit. **Read this before quoting any number.** |
| [`VALIDATION_METHODOLOGY.md`](VALIDATION_METHODOLOGY.md) | Why each check is constructed the way it is |
| [`PYLLE_STATUS_V2.md`](PYLLE_STATUS_V2.md) | The cross-code comparison in full, including its qualification |
| [`CONVERGENCE_LLE.md`](CONVERGENCE_LLE.md) | The discretization-uncertainty study at δω = 30 κ |

---

## The three tiers

The tiers answer three different questions, and no tier can substitute for another.

```
Tier 1  SELF-CONSISTENCY   "does the code compute what the code claims?"
        ├─ all-off bit-identity vs committed goldens        0 ULP
        └─ the exact fixed point of its own discrete map    1.9e-13

Tier 2  MATHEMATICAL       "does the discretization converge to the right thing?"
        ├─ observed order of accuracy (self-convergence)    1.00 shipping / 2.00 fixed
        ├─ manufactured solutions (MMS)                     1.00
        └─ weak order under noise                           3.05

Tier 3  CROSS-CODE         "are the conventions right at all?"
        └─ pyLLE 4.1.2, independent implementation          7/7 HARD checks pass
```

**Tier 1 and Tier 2 would both verify a consistently wrong convention perfectly.** An error
in the detuning sign, the dispersion direction, a factor of $2\pi$ or the mode-index origin
is invisible to a solver checked only against itself. Tier 3 is the only check in this
repository that can catch one — which is why changing the dispersion loader or the operator
splitting is a named trigger to re-run it.

---

## Tier 1 — self-consistency

### All channels off is bit-identical to the deterministic LLE

Every stochastic channel is opt-in, and every default reproduces the pre-noise solver
**bit-for-bit**. Four committed golden trajectories pin it.

```bash
python -m validation.noise_off_identity --check --strict
```

Measured on the reference environment (jax/jaxlib 0.10.2, numpy 2.4.6, Python 3.11.15, CPU):

```
4 param sets, mode=strict (0 ULP): 0 differences
all param sets reproduce the golden files
```

The four arrays compared per parameter set are `U_int_history`, `E_final`, `E_snapshots` and
`delta_T_history`. Parameter sets: `s256_short`, `s512_prod`, `s512_sub4`, `s1024_near`.

The same claim as a test, plus three further failure modes:

```bash
SOLITON_STRICT_ULP=1 pytest tests/test_noise_off_identity.py -q
```

| Test | Guards against |
|---|---|
| `test_all_off_bit_identical_to_golden` | the integrator drifting from the committed goldens |
| `test_all_off_equals_every_channel_individually_off` | naming a channel explicitly off changing anything — all 8 switch fields |
| `test_enabling_then_disabling_is_idempotent` | process-global contamination (mutated JAX config, a cache keyed on the wrong thing) |
| `test_golden_provenance_matches_current_config` | a config edit silently invalidating the goldens |

Strictness is environment-controlled: `SOLITON_STRICT_ULP=1` gives raw-byte equality,
otherwise `np.allclose(atol=1e-13, rtol=0)`. The loose default is deliberate — bit-identity
is a property of the *toolchain*, and demanding it unconditionally would turn every
dependency bump into a red suite that looks like a physics regression. See
[`LIMITATIONS.md`](LIMITATIONS.md) §1 for why this is a fixed-hardware claim.

### The exact CW fixed point of its own discrete map

```bash
python -m validation.analytic_cw
```

Runtime ≈ 1 m 52 s. Measured:

```
OK: the solver reproduces the exact fixed point of its own discrete map to
1.922e-13 at every one of 405 points.
```

against a gate of `rtol = 1e-12`.

The check compares against the fixed point of the **discrete map the solver actually
iterates**, not against the continuum LLE. The residual against the continuum cubic is
3.083e-03 — and that is *not* a convergence artifact: it is the mean-field
(Ikeda-map → LLE) truncation of the splitting, first order in $\Delta t$ and of size
$\kappa t_r/2 = 3.087\times10^{-3}$. The same run demonstrates that, showing the continuum
residual halving with `n_substeps` (fitted order 0.998).

---

## Tier 2 — mathematical

```bash
python -m validation.convergence --report
```

Runtime ≈ 4 m 38 s. Measured gates:

```
                   study      order           required
            a_field_only     2.0000             >= 1.9   PASS
         b_thermal_euler     0.9998         [0.8, 1.2]   PASS
   c_thermal_exponential     2.0000             >= 1.9   PASS
                    weak     3.0496             >= 0.9   PASS
```

Those are gated on the **fixed** scheme (`symmetric_drive=True`,
`thermal_coupling="strang"`). The shipping defaults measure, in the same run:

```
            a_field_only     1.0002
         b_thermal_euler     0.9999
   c_thermal_exponential     1.0005
                     mms     1.0000
```

**The shipping scheme is first order, by construction and on purpose.** The legacy sub-step
applies the whole drive as one Euler kick *before* L·N·L:

$$P(\Delta t)\cdot L(\tfrac{\Delta t}{2})\cdot N(\Delta t)\cdot L(\tfrac{\Delta t}{2})$$

which is not palindromic, so despite the Strang core the method is first order overall. With
`symmetric_drive=True` the drive splits into two half kicks straddling the core,

$$P(\tfrac{\Delta t}{2})\cdot L(\tfrac{\Delta t}{2})\cdot N(\Delta t)\cdot L(\tfrac{\Delta t}{2})\cdot P(\tfrac{\Delta t}{2}),$$

which is palindromic and measures 2.00. Both fixes are **opt-in, default off**, because
enabling either changes every committed trajectory.

Note `b_thermal_euler` stays at order 1 whichever way the field is integrated — that is
correct, and is what the bracketed gate `[0.8, 1.2]` asserts rather than an inequality.
Reaching second order in the coupled system needs the exponential integrator *and* Strang
coupling; the exponential integrator alone does not do it, because the field↔thermal
coupling is lagged. See [`LIMITATIONS.md`](LIMITATIONS.md) §6.

**Manufactured solutions.** `validation/mms.py` derives a forcing term $S(\tau, t)$
symbolically (sympy) and adds it to the field in the same place and the same way as the pump
kick — so the study measures the order of the scheme *as it actually runs*, not of an
idealized one.

**Discretization uncertainty** at the production operating point is quantified separately;
see [`CONVERGENCE_LLE.md`](CONVERGENCE_LLE.md).

---

## Tier 3 — cross-code

```bash
python -m validation.pylle_crosscheck \
    --pylle-python ./pylle-env/bin/python \
    --julia-bin    ./pylle-env/bin/julia
```

**This one cannot run in the solver's own environment.** pyLLE pins `numpy<2` and needs a
Julia toolchain, so `validation/` drives it out-of-process through `PYLLE_PYTHON`. Install
the extra with `pip install -e ".[pylle]"` into a *separate* environment; setup is in
[`PYLLE_STATUS_V2.md`](PYLLE_STATUS_V2.md).

**Status: FROZEN at verdict `FAIL (QUALIFIED)`, 7/7 HARD checks passing.** The verdict and
its qualification are the subject of `PYLLE_STATUS_V2.md` — read it before citing the
comparison either way.

The artifacts are hash-pinned, so the frozen result is verifiable *without* Julia:

```bash
pytest tests/test_validation_freeze.py tests/test_pylle_crosscheck_v2.py -q
```

Runtime ≈ 15 s. This checks that every committed artifact matches its recorded hash, that no
result artifact has been deleted, that the v1 artifacts are untouched, and that the
convention functions are still bit-identical to the v1 objects. Every artifact is listed in
[`FROZEN_MANIFEST.md`](../validation/results/FROZEN_MANIFEST.md).

### Re-run triggers

The cross-check is frozen, not abandoned. Five named changes require re-running it, because
each alters the **problem being solved** rather than only the solver solving it:

1. the dispersion loader `load_dint_grid()` — both codes are fed one array derived through it;
2. the D1/FSR reconciliation, the mode-index origin, or the CSV reference row;
3. the linear/nonlinear step composition — splitting order, drive-kick placement,
   `symmetric_drive`, half-step structure;
4. the detuning sign convention;
5. anything that moves the committed goldens.

---

## The noise channels themselves

Bit-identity proves the channels are *off* correctly. That they are *on* correctly is a
separate matter, covered per channel:

| Claim | Test |
|---|---|
| Injection variance matches the Eq. 126 prescription | `tests/test_quantum_noise.py::test_injection_variance_matches_prescription` |
| Undriven steady state is ½ photon per mode | `::test_vacuum_equilibrium_occupation` |
| Decay matches the cavity linewidth | `::test_vacuum_autocorrelation_linewidth` |
| MI sideband selection from vacuum matches Eq. 62 | `::test_mi_sideband_selection_from_vacuum` |
| Disabled path traces zero RNG ops in the scan body | `::test_disabled_path_adds_no_rng_to_scan_body` |
| Synthesized PSD tracks its target within 3 dB/octave over 3 decades | `tests/test_colored_noise.py::test_welch_matches_target_within_3db_per_octave` |
| K–G variance renormalizes to the Eq. 129 value | `::test_kg_variance_renormalized_to_eq129_gate1` |
| `single_pole` is bit-identical to the legacy AR(1) | `::test_single_pole_bit_identical_to_legacy_ar1` |
| Disabling one channel does not shift another's stream | `::test_disabling_trn_does_not_shift_tccr_stream` |
| Pump frequency-noise sign convention is exactly −2πδν | `tests/test_pump_noise.py::test_sign_convention_exact` |
| The linearized CW low-pass transfer at f_mod ∈ {κ/20, κ/2, 5κ}/2π | `::test_linear_response_transfer` |
| RIN energy balance | `::test_rin_energy_balance` |
| FSR mode-linear phase is exactly −μ·δD₁·t | `tests/test_noise_metrology.py::test_fsr_constant_dd1_exact_phase` |
| `T_k = 0` collapses every δT-derived channel | `::test_tk_zero_collapses_all_delta_t_channels` |

Run the lot:

```bash
pytest -q                    # fast suite, ~9 min on 4 cores
pytest -q --runslow          # + the large-grid cases
```

---

## Reproducible environment

For a number that must match a published one, use the pinned, hash-verified toolchain rather
than the floating one. `requirements.lock.txt` pins the exact `jaxlib` the goldens were
produced with, which is what makes the 0-ULP check meaningful:

```bash
docker build -t soliton-control . && docker run --rm soliton-control
```

or directly:

```bash
pip install --require-hashes -r requirements.lock.txt
pip install --no-deps --no-build-isolation -e .
SOLITON_STRICT_ULP=1 pytest tests/test_noise_off_identity.py -q
```

`environment.yml` is a conda convenience and is explicitly **not** the bit-identity
environment — conda-forge builds numpy/scipy against a different BLAS, and different BLAS
means different reduction order. The caveat is in the file's own header.
