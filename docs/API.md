# API

The callable surface, in the order you meet it. Every function listed here carries a full
NumPy-style docstring with units on every argument — `help(fn)` is authoritative; this page
is the map.

---

## `simulator.lle_solver`

### `solve_lle_ssfm_jax(...) -> dict[str, np.ndarray]`

The entry point. A batch-capable split-step Fourier integrator for the generalized LLE
coupled to a single-pole thermo-optic ODE.

**Required:**

| Argument | Units | Meaning |
|---|---|---|
| `pin` | W | pump power |
| `delta_omega` | rad/s | detuning sweep, shape `(n_traj, t_slow)`; a scalar or `(t_slow,)` is broadcast to one trajectory |
| `t_slow` | round trips | total integrated time is `t_slow * t_r` |
| `beta` | s^(k−1) | `[beta2, beta3, ...]` in the LLE convention $\beta_k = D_k/D_1^k$ |
| `kappa`, `kappa_c` | rad/s | total loss rate and coupling rate |
| `rng_key` | — | JAX PRNG key |

**Grid and numerics:**

| Argument | Default | Meaning |
|---|---|---|
| `n_tau` | `512` | fast-time points; resolves $\lvert\mu\rvert < n_\tau/2$ |
| `n_substeps` | `1` | Strang sub-steps per round trip; `1` is bit-identical to the legacy solver |
| `fine_cadence_M` | `1` | advance the whole evolution at $\Delta t = t_r/M$; `1` is bit-identical |
| `dealias_two_thirds` | `False` | zero $\lvert\mu\rvert > n_\tau/3$ after each nonlinear kick |
| `edge_absorber` / `edge_absorber_frac` | `False` / `0.12` | super-Gaussian damping at the grid edges |
| `dispersion_validity_mask` / `validity_phase_threshold` | `False` / `pi` | opt-in guard for coarse `n_substeps=1` runs |
| `d_int_grid` | `None` | measured $D_\mathrm{int}(\mu)$ in FFT-bin order; bypasses the Taylor `beta` path |

All numerics toggles default to the legacy path. `analysis.dks_access.PRODUCTION_NUMERICS`
bundles the quantitative-spectrum stack (`n_substeps=4`, `dealias_two_thirds=True`,
`edge_absorber=True`).

**Noise:**

| Argument | Meaning |
|---|---|
| `noise_config` | a `NoiseConfig`; the recommended way to set channels |
| `quantum_noise_enabled`, `pump_noise_enabled`, `fsr_noise_enabled` | deprecated per-channel kwargs; `None` = read the config |
| `pump_freq_noise_override`, `pump_rin_epsilon_override`, `fsr_delta_d1_override` | deterministic sequences that bypass the stochastic synthesis (used by the sign-convention and linear-response tests) |

**Order-of-accuracy options** (all default to the legacy behaviour, all static Python values,
so the default path traces the historical arithmetic token for token):

| Argument | Default | |
|---|---|---|
| `symmetric_drive` | `False` | `True` splits the drive into two half kicks straddling L·N·L → second order |
| `thermal_coupling` | `"lagged"` | `"strang"` sandwiches the ΔT update around the field update |
| `source_fn` | `None` | manufactured-solution forcing $S(\tau,t)$; pass a *stable* callable — it is part of the jit cache key |

**Diagnostics:**

| Argument | Meaning |
|---|---|
| `mode_probe_indices` | mode numbers $\mu$ whose complex FFT amplitudes are recorded every round trip (≤ 16); costs one extra FFT per round trip, only when enabled |
| `provenance` | attach a `"provenance"` entry recording git commit, config digests, seed, environment fingerprint. Off by default so the key set stays legacy-exact |

**Returns** a dict of numpy arrays, always containing `U_int_history` (J·s),
`P_trans_history` (W), `DeltaT_history` (K), `delta_omega_eff_history` (rad/s),
`E_snapshots` (complex128, $\sqrt{\mathrm J}$), `label_history` (int32), `e_final`,
`delta_t_final`. Active channels add `pump_freq_noise_history`,
`pump_rin_epsilon_history`, `fsr_delta_d1_history`; probes add `mode_probe_history`.

### Unit conversions — use these, do not hand-convert

```python
gamma_nlse_to_lle(gamma_nlse_per_w_per_m, fsr_hz, n_eff=2.2)  # W^-1 m^-1 -> J^-1 s^-1
d2_to_beta2_lle(d2_rad_per_s2, fsr_hz)                        # rad/s^2   -> s
d3_to_beta3_lle(d3_rad_per_s3, fsr_hz)                        # rad/s^3   -> s^2
```

These are the two classic ways to be wrong by orders of magnitude, so the solver asserts on
both ranges and each failure message names the function that fixes it.

### Config helpers

```python
resolve_cavity_rates(config_path=None) -> (kappa_i, kappa_c, kappa_total)
hbar_omega0_from_config(physical) -> float          # J; 0 or missing = compute from lambda_p
load_dint_grid(n_tau, csv_path=None) -> DintGrid    # .grid (rad/s, FFT-bin order), .d1
build_dispersion(omega, beta_list)                  # Taylor D_int on the frequency grid
```

`resolve_cavity_rates` is the single source of truth for $\kappa$ — every caller goes through
it, so "what was kappa?" has one answer per config. It warns if an explicit rate and its
quality factor disagree by more than 15 %.

`load_dint_grid` is a **validation re-run trigger**: both this solver and pyLLE are fed one
array derived through it, so changing it alters the problem, not just the solver. Results are
cached per `(n_tau, csv path)`.

---

## `simulator.noise_config`

### `NoiseConfig`

Immutable, hashable, frozen dataclass. Pure configuration — it imports nothing from the
solver.

```python
NoiseConfig.all_off(**overrides)   # every switch False; thermal_feedback left at its default
NoiseConfig.all_on(**overrides)    # every switch True (supports leave-one-out via overrides)
NoiseConfig.from_yaml(path)        # read a config file through the full precedence chain
nc.to_dict()                       # sorted, JSON-serialisable
nc.sha256()                        # digest of the whole config, for provenance
nc.enabled_channels                # tuple of the stochastic channels that are on
nc.describe()                      # one human-readable line per channel
```

Eight **switch** fields: `quantum_vacuum`, `trn`, `pyro_eo`, `tccr`, `pump_freq_noise`,
`pump_rin`, `fsr` (the seven stochastic channels) plus `thermal_feedback`. All default to
`False`, enforced by a test that enumerates the dataclass.

Parameter fields: `trn_psd_model` (`single_pole` | `kondratiev_gorodetsky` | `csv`),
`trn_ar1_stationary_init`, `thermal_integrator` (`euler` | `exponential`),
`quantum_injection_cadence` (0 | 1), `quantum_seed_vacuum_init`, `legacy_segment_noise`,
`noise_dtype` (`float32` | `float64` — see [`LIMITATIONS.md`](LIMITATIONS.md) §2), `seed`.

`thermal_feedback` is a switch but **not a noise channel**: it names the deterministic
thermo-optic ODE. `all_off()` deliberately does not force it.

---

## `simulator.noise_models`

Each class turns a `physical_parameters` mapping into (a) a closed-form one-sided PSD and
(b) a sampler that draws a realization of it, one sample per round trip at $f_s = 1/t_r$.
The PSDs are the validation targets.

| Class | Channel | Key methods |
|---|---|---|
| `TRNoise` | thermorefractive | `.sample(key, N)`, `.sample_delta_t(key, N)`, `.psd(f)`, `.c_pull`, `.var_delta_t` |
| `PyroEONoise` | pyro-electric/EO | `.sample`, `.psd`, `.pyro_coeff`, `.eps_r_eff` |
| `TCCRNoise` | thermal-carrier | `.sample`, `.psd`, `.sigma_tccr`, `.s0_tccr` |
| `TotalNoise` | their correlated sum | `.sample_with_delta_t(key, N)`, `.sample`, `.sample_full_with_delta_t` |
| `PumpNoise` | laser frequency + RIN | `.sample_freq`, `.sample_rin`, `.psd_freq`, `.psd_rin`, `.lorentzian_linewidth_hz` |

**Use `TotalNoise.sample_with_delta_t`, not the individual samplers, inside a solve.**
Sampling `PyroEONoise` standalone draws an *independent* temperature realization, which is
correct only for a per-channel PSD check. The physically correct coupling — one shared
$\delta T(t)$ behind both pull coefficients — comes from `TotalNoise`.

Every constructor takes `enabled: bool | None`. `None` (the default) reproduces the
historical implicit gating exactly; an explicit bool overrides it.

---

## `simulator.colored_noise`

```python
synthesize_from_psd(rng, n, psd, f_s, clamp_dc=True) -> np.ndarray   # (n,) float64
single_pole_psd(variance, tau) -> callable                            # Lorentzian
kondratiev_gorodetsky_psd(T_k, kappa_th, rho, cp, R, d_a, d_b,
                          mode_volume, f_max, f_lo=1.0) -> (callable, var_eq129)
csv_psd(csv_path) -> callable                                         # log-log interpolated
integrate_psd(psd, f_lo, f_hi, n_grid=20000) -> float
np_generator_from_key(key) -> np.random.Generator                     # JAX key -> numpy stream
```

Host-side numpy, float64 throughout, independent of the JAX x64 flag. The draw order inside
`synthesize_from_psd` (all real parts, then all imaginary parts) is part of the determinism
contract and must not be reordered.

---

## `simulator.state_labeler`

Seven-class classification of an intracavity-field snapshot: 0 off, 1 CW, 2 MI, 3 chaotic,
4 multi-soliton, 5 soliton crystal, 6 single soliton.

```python
make_threshold_params(kappa, kappa_c, pin, delta_omega_max, ...)  # the shared threshold dict
make_state_labeler(threshold_params)     # JAX-traceable, runs inside the solver's scan
label_soliton_state(E_tau, params)       # NumPy/SciPy path, for stored trajectories
label_trajectory(E_history, params)
assert_labelers_consistent(e_field, ...) # the two must agree
sech2_envelope_correlation(e_field)      # -> (pearson_corr, r2, fitted_mode_width)
physical_off_floor(kappa, kappa_c, pin, delta_omega_max, off_fraction=1e-3)
```

Both labelers are driven by the **same** threshold dict, so a disagreement is a genuine
mechanism drift rather than a config mismatch. Thresholds are derived from the physical
config rather than hard-coded — the OFF floor is a fraction of the dimmest CW state the
cavity can support over the sweep.

---

## `analysis` — the study layer

Command-line entry points (`python analysis/<name>.py --help`):

| Module | What it does |
|---|---|
| `noise_budget.py` | the per-channel budget: OAT, leave-one-out, interaction residual, with common random numbers. `--quick` for a smoke run |
| `noise_metrology.py` | per-line frequency-noise PSDs, elastic-tape decomposition, β-separation linewidths, timing jitter |
| `quantum_noise_report.py` | the quantum-channel validation suite and its figures |
| `pump_noise_report.py` | the pump-channel physics study, incl. dispersive-wave recoil |
| `noise_validation_campaign.py` | the W1–W5 campaign driver |
| `run_detuning_sweep.py` | warm-continuation step-and-hold detuning sweeps |
| `dks_access.py` | soliton access helpers — `load_cavity_params`, `attach_dispersion`, `sech_soliton_seed`, `access_by_seeding`, `PRODUCTION_NUMERICS` |
| `spectral_metrics.py` | `normal_ordered_spectrum()` — **required for any comparison to measurement** |

`analysis/dks_access.py` is the most useful of these interactively; `notebooks/` uses it
throughout.

---

## `validation` — the verification layer

| Module | Command |
|---|---|
| `noise_off_identity.py` | `python -m validation.noise_off_identity --check --strict` |
| `analytic_cw.py` | `python -m validation.analytic_cw` |
| `convergence.py` | `python -m validation.convergence --report` |
| `convergence_lle.py` | discretization uncertainty at the production point |
| `mms.py` | manufactured solutions (needs `sympy`, in the `dev` extra) |
| `pylle_crosscheck.py`, `pylle_crosscheck_v2.py` | cross-code; needs a separate pyLLE + Julia environment |
| `criteria.py` | the acceptance criteria the above are gated on |

See [`VALIDATION.md`](VALIDATION.md) for what each one proves.
