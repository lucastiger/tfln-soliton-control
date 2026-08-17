# soliton-control

[![CI](https://github.com/lucastiger/soliton-control/actions/workflows/ci.yml/badge.svg)](https://github.com/lucastiger/soliton-control/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`soliton-control` is a scientific computing project scaffold for simulating and controlling soliton dynamics in thin-film lithium niobate (TFLN) and silicon nitride microresonators. The repository is organized around four major workflows:

- **Simulation** of Lugiato–Lefever equation (LLE) dynamics with thermal effects and realistic noise models.

  The SSFM solver optionally includes physically normalized **quantum vacuum noise** (Herr, Tikan & Kippenberg, arXiv:2604.05897, Sec. V.B.2, Eq. 126): each cavity mode is driven by the vacuum Langevin input √κ·ξ̂\_μ(t) with ⟨ξ̂\_μ(t)ξ̂†\_μ′(t′)⟩ = δ(t−t′)δ\_μμ′ (both loss baths combined; coherent/vacuum baths only), implemented in the classical truncated-Wigner (symmetric-ordering) limit as an additive complex Gaussian injected in the fast-time domain once per fine step, with per-quadrature std √(ħω₀·κ·n\_tau·dt/4). Its undriven steady state is the symmetric-ordered vacuum occupation of **½ photon per mode** — the same convention used for the optional cold-start seed, which replaces the legacy ad-hoc 1e-3·|E\_cw| noise with a vacuum-scale draw (⟨n\_μ⟩ = ½ at t = 0). The channel is gated behind `quantum_noise_enabled` in `config/sin_params.yaml` (**off by default** — the flag-off solver is bit-identical to the legacy solver), with `quantum_noise_seed_vacuum_init` selecting the vacuum cold-start seed and `hbar_omega0_j` optionally overriding ħω₀ (0 = compute from `pump_wavelength_m`; the pump-mode ħω₀ is used for all comb modes, a <1 % approximation across the comb span). See `analysis/quantum_noise_report.py` for the validation suite (½-photon vacuum equilibrium, cavity-linewidth decay, MI sideband selection from vacuum against paper Eq. 62, and the single-soliton spectral floor at ħω₀/2).

  Config-encoding note: all quantum-noise booleans (and the cadence enum) are encoded as **0/1 integers** rather than YAML `true`/`false`/`null`, because the config regression tests pin every `physical_parameters` leaf to parse as a plain number; the solver validates them as boolean-valued. Two further knobs: `quantum_noise_injection_cadence` selects `0` = one injection per fine step (the exact prescription, default) or `1` = one injection per round trip with the variance rescaled to dt = t\_r — a CPU-performance option valid because κ·t\_r ≈ 6.2×10⁻³ ≪ 1 (steady occupation 0.5015 vs 0.5; bit-identical to the fine cadence when `fine_cadence_M = 1`). When — and only when — the channel is enabled, the state labeler activates its vacuum-floor parameters `labeler_vacuum_floor_margin` (default 10 = +10 dB) and `labeler_envelope_smooth_modes` (default 8): the single-soliton envelope gate smooths the linear spectrum and clips it at margin × n\_tau²ħω₀/2 (the per-mode vacuum level in raw |FFT|² units, whose single-snapshot fluctuation is ≈5.6 dB), and the OFF power floor is lifted to at least margin × n\_tau·ħω₀/2 so a vacuum-filled cavity labels OFF; with the channel disabled the labeler is bit-identical to the legacy one. The vacuum background's contribution to absorbed power is κ\_i·n\_tau·ħω₀/2 ≈ 1.6×10⁻⁸ W at n\_tau = 8192 — negligible for the thermal ODE.

  The solver also optionally includes **pump-laser noise** — frequency noise and relative intensity noise (RIN) — following the same paper (Herr, Tikan & Kippenberg, arXiv:2604.05897, Secs. V.B.4–V.B.5). Both channels are synthesized **host-side in float64** once per trajectory (`PumpNoise` in `simulator/noise_models.py`) and fed into the *existing* equations of motion, so the cavity's transfer function (low-pass filtering and quadrature rotation) and the thermal transduction pathway emerge from the solver itself — no transfer function is hand-implemented. **Frequency noise:** because the solver frame co-rotates with the pump, the instantaneous laser-frequency deviation δν\_p(t) is exactly a detuning noise; since δω ≡ ω\_res − ω\_p, a positive laser-frequency excursion *lowers* δω, so the contribution **−2π·δν\_p(t)** is summed into the per-round-trip detuning-noise sequence (no solver-scan change) and returned as `pump_freq_noise_history`. Its one-sided PSD is S\_δν(f) = h₀ + h₋₁/f on f ∈ [1/(t\_slow·t\_r), 1/(2t\_r)]: a white plateau h₀ carrying the intrinsic Lorentzian linewidth Δν\_L = π·h₀ (i.i.d. per round trip, variance h₀·f\_s/2) plus a flicker term h₋₁/f synthesized by Hermitian FFT (DC bin clamped to the first bin). **RIN:** P\_in(t) = P̄\_in·(1+ε(t)) from S\_ε(f) = 10^(floor/10) + 10^(excess/10)·(f\_c/f) below the corner f\_c (floor-only above), clipped so 1+ε ≥ 0 (a warning fires if >0.01 % of samples clip). The per-round-trip pump-power scale 1+ε is threaded as `pump_scale_sequence`, so the pump kick becomes √(max(κ\_c·P̄\_in·(1+ε), 0))·dt\_sub held constant across the fine steps (RIN bandwidth ≪ FSR ⇒ per-round-trip resolution is exact), and the absorbed-power/thermal pathway then transduces RIN → ΔT → detuning automatically (the paper's thermal-transfer mechanism); `pump_rin_epsilon_history` is returned for diagnostics. Both channels are gated behind `pump_noise_enabled` (0/1, **off by default** — the flag-off solver is bit-identical to the legacy solver, and the RIN-disabled path traces zero extra ops in the scan body via a static `None` sequence). The knobs are `pump_freq_noise_h0_hz2_per_hz` (ECDL ≈ 3×10³ ⇒ Δν\_L ≈ 10 kHz; fiber laser ≈ 30 ⇒ ≈ 100 Hz), `pump_freq_noise_hm1_hz3_per_hz` (flicker, representative ECDL ≈ 10¹⁰), and `pump_rin_floor_dbc_per_hz` / `pump_rin_excess_dbc_per_hz` / `pump_rin_corner_hz`; validation rejects negative h₀/h₋₁ and any RIN value above −80 dBc/Hz (a guard against accidental linear-vs-dB entry), and `enabled = 0` forces every channel inert regardless of the numbers. New per-trajectory PRNG subkeys are *appended* to the existing key chain, so enabling pump noise never perturbs a legacy stream. See `tests/test_pump_noise.py` for the validation suite (PSD fidelity within 3 dB/octave over 3 decades, exact −2πδν sign convention, the linearized CW low-pass transfer at f\_mod ∈ {κ/20, κ/2, 5κ}/2π, RIN energy balance, determinism, and config-validation triggers) and `analysis/pump_noise_report.py` for the physics study, including the **dispersive-wave-recoil** contrast (pump frequency noise is predominantly common-mode with Taylor-D₂-only dispersion, but couples more strongly into repetition-rate wander once the measured `d_int_grid` supplies the DW phase matching) and the RIN → ΔT transduction via R\_th ≈ 0.545 K/W.
- **Data generation** for large-scale supervised/physics-informed learning.
- **Model training** for a physics-informed recurrent network (PI-RNN).
- **Closed-loop control** using model predictive control (MPC) and hardware integration stubs.

## Noise models

Every stochastic channel of the simulator, its configuration keys (all under `physical_parameters` in `config/sin_params.yaml`), defaults, and the equation it implements from Herr, Tikan & Kippenberg, arXiv:2604.05897. **Every default reproduces the pre-colored-noise solver bit-for-bit**; each channel is opt-in.

| Channel | Physics / paper reference | Config keys (default) | Default state |
|---|---|---|---|
| Quantum vacuum | Langevin drive √κ·ξ̂\_μ(t), Eq. 126 (Sec. V.B.2); truncated-Wigner ½ photon/mode | `quantum_noise_enabled` (0), `quantum_noise_seed_vacuum_init` (1), `quantum_noise_injection_cadence` (0), `hbar_omega0_j` (0 = auto), `labeler_vacuum_floor_margin` (10), `labeler_envelope_smooth_modes` (8) | OFF |
| Thermorefractive (TRN) | δω(t) = C\_pull·δT(t); variance Eq. 129 k\_BT²/(ρC\_pV); spectrum selectable — `single_pole` (AR(1)/Lorentzian twin), `kondratiev_gorodetsky` (Eq. 130, variance renormalized to Eq. 129), `csv` (measured/FEM, Huang et al. 2019 style, log-log interpolated, flat-clamped) | `T_k` (300), `tau_th_s`, `dn_dT_per_k`, `rho_kg_per_m3`, `Cp_j_per_kg_k`, `mode_volume_m3`, `trn_psd_model` (`single_pole`), `trn_R_m`/`trn_da_m`/`trn_db_m` (null; required iff K-G, with d\_a ≥ 1.2·d\_b asserted), `trn_psd_csv_path` (null), `trn_csv_units` (`S_delta_T`) | ON (single-pole, as before) |
| Thermal expansion pull | "dimensional fluctuation" companion of TRN folded into the pull: C\_pull = (ω₀/n₀)(dn/dT + n₀·α\_L) | `alpha_L_per_k` (0.0) | OFF (0.0 = thermo-optic only) |
| Pyro-EO | χ⁽²⁾ pyroelectric-EO shift driven by the **same δT sequence** as TRN (partial cancellation, sign per z-cut TFLN) | `eo_r33_m_per_v` (0 for SiN), `pyroelectric_coeff_c_per_m2_k` (0), `eps_r_z`, stack thicknesses | OFF for SiN (r33 = 0) |
| TCCR | surface-carrier shot noise → EO shift, AR(1) with `tau_carrier_s` | `surface_state_density_per_m2`, `tau_carrier_s`, `eo_r33_m_per_v` | OFF for SiN (r33 = 0) |
| Pump frequency noise | S\_δν(f) = h₀ + h₋₁/f, Δν\_L = π·h₀ (Sec. V.B.4); enters as −2π·δν\_p(t) on the detuning axis | `pump_noise_enabled` (0), `pump_freq_noise_h0_hz2_per_hz` (0), `pump_freq_noise_hm1_hz3_per_hz` (0) | OFF |
| Pump RIN | S\_ε(f) floor + 1/f excess below `f_c` (Sec. V.B.5); pump-power scale 1+ε(t) | `pump_rin_floor_dbc_per_hz` (−300), `pump_rin_excess_dbc_per_hz` (−300), `pump_rin_corner_hz` (1e4) | OFF |
| FSR (repetition-rate) noise | δD₁(t) = (D₁/ω₀)·C\_pull·δT(t) from the **same δT sequence**; per-mode linear detuning μ·δD₁(t) in the linear operator — the TRN-limited f\_rep term (Sec. V.B.1 elastic tape) | `fsr_noise_enabled` (0) | OFF |
| Dataset segment cadence | `legacy_segment_noise` = 1 keeps the historical per-segment noise regeneration (bit-identical, with its documented boundary decorrelation transient); 0 = full-trajectory-then-slice continuity fix | `legacy_segment_noise` (1) | legacy |

> **DW-recoil linewidth validation note:** the ≈256σ measured-`D_int` curvature significance reported in the `fig6_linewidth_dwrecoil` block of `analysis/results/noise_comparison_report.json` is a *within-record* segment-resampling (Welch-segment bootstrap) estimate of how well that one record's parabola is resolved above its own noise — **not** a run-to-run reproducibility bound. The *physical* discriminator between genuine DW-recoil and a numerical artifact is the flat Taylor-D₂ control (a₂ = 0, S\_rep ratio ≥ 10⁶×), not the σ magnitude.

Infrastructure: `simulator/colored_noise.py` synthesizes any channel from a one-sided target PSD (exact recipe in its docstring: Hermitian rfft draw with c\_k = ζ\_k·√(S(f\_k)·f\_s·N/2), DC clamped to S(f₁); Var(x) = ∫₀^{f\_s/2}S df), host-side float64, seeded deterministically from JAX keys. `T_k = 0` (the `write_noise_off_config` sidecar) forces **every** δT-derived channel — TRN, Pyro-EO, and FSR noise, for every `trn_psd_model` including `csv` — identically to zero; quantum and pump noise carry their own switches.

**Noise metrology** (`analysis/noise_metrology.py`, paper Sec. V.B.1): the solver can record the complex FFT amplitudes of up to 16 probed modes every round trip (`solve_lle_ssfm_jax(mode_probe_indices=...)` → `mode_probe_history`, one extra FFT per round trip only when enabled). On those records the module computes per-line frequency-noise PSDs (Welch, detrended), the repetition-rate phase, the elastic-tape decomposition S\_μ(f) = S\_c + 2μS\_cr + μ²S\_rep with fix point μ\_fix = −S\_cr/S\_rep (least-squares across ≥5 probes per Fourier bin), β-separation-line effective linewidths (Di Domenico et al., Appl. Opt. 49, 4801 (2010)), timing jitter from the temporal peak trajectory, and a warm-continuation quiet-point sweep (Sec. V.B.5). `analysis/noise_comparison_report.py` regenerates the eight comparison/validation figures and their JSON metrics.

**Noise budget** (`analysis/noise_budget.py`): the per-channel budget table — how much of each observable's fluctuation each noise channel is responsible for, one-at-a-time (OAT), leave-one-out (LOO), and the interaction residual between them. Thirteen channel sets (nine named plus four generated LOO rows) × five observables (`u_int_rms_frac`, `s_rep`, `line_linewidth`, `timing_jitter`, `step_jitter`), each with a mean, SEM and a 2000-resample bootstrap 95 % CI over a seed list **shared across every channel set** — common random numbers, which the solver's unconditional PRNG splits make exact rather than approximate, so the OAT/LOO *differences* are not swamped by sampling noise. Three things the table records that a budget usually omits: (i) **which record length produced each cell** — the fast (t\_slow = 2×10⁵, 8.13 µs) and slow (2×10⁷, 813 µs) records reach different Fourier floors, and a target below a record's floor is reported as `null` with the record length that *would* observe it, never as an interpolation clamped to the nearest observed bin; (ii) the **Welch segmentation** (segment length, segment count, achieved bin spacing) behind every spectral number, since Welch resolves f\_s/nperseg rather than the record's Fourier floor; (iii) that `pyro_eo` and `fsr` are driven by the *same* δT(t) realization as `trn` and are therefore never run without it (alone they are identically zero). `thermal_feedback` is pinned on in every cell so OAT/LOO differences cannot pick up the deterministic thermo-optic ODE, and a built-in check asserts the `pyro_eo` row reproduces the `trn` row bit-for-bit (on centrosymmetric SiN, r₃₃ = 0 ⇒ pyro coefficient = 0), which is a live proof that common random numbers are in effect. Writes `analysis/results/budget/budget.json` (+ `budget.npz`/`.provenance.json`), `budget_table.tex` (booktabs, ready to `\input`) and `budget_table.md`; checkpoints after every solver unit and supports `--resume`, which refuses to reuse a store written under different run settings. Smoke test: `python analysis/noise_budget.py --quick`.

## Validation status

**Before quoting any number this simulator produced, read
[`docs/VALIDATION_STATUS.md`](docs/VALIDATION_STATUS.md).** It is the single
top-level answer to "how validated is this, and to what": the verification
hierarchy with its measured numbers, the **configuration-coverage table** (the
validated configuration is *not* the production configuration), every open item
with its impact on current claims, and when to revisit the pyLLE cross-check.

In short: the solver's CW fixed point is checked against exact mathematics to
~1e-14, its order of accuracy is measured against manufactured solutions, its
discretization uncertainty is quantified at δω = 30 κ, and its conventions are
cross-checked against an independent implementation (pyLLE 4.1.2) with 7/7 HARD
checks passing. That cross-check is **FROZEN** at verdict `FAIL (QUALIFIED)` —
see `docs/PYLLE_STATUS_V2.md`. Every result artifact is hash-pinned in
[`validation/results/FROZEN_MANIFEST.md`](validation/results/FROZEN_MANIFEST.md).

## Installation

Requires Python 3.10–3.12.

```bash
git clone https://github.com/lucastiger/soliton-control
cd soliton-control
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Install **editable** (`-e`). `simulator/lle_solver.py` resolves the default
config relative to the checkout, so a non-editable wheel imports but cannot find
`config/sin_params.yaml`.

That base install is everything needed to run the solver and the benchmark.
Extras are opt-in:

| Extra | Adds | For |
|---|---|---|
| `dev` | pytest, pytest-cov, ruff, jupyter, nbconvert, pip-tools, sympy | development and the full test suite |
| `ml` | torch, einops | the PI-RNN observer under `model/` and `data/dataloader.py` |
| `pylle` | pyLLE | the cross-check (also needs Julia; see `docs/PYLLE_STATUS_V2.md`) |

```bash
pip install -e ".[dev]"        # to run the tests
pip install -e ".[dev,ml]"     # + the learning stack
```

`torch` is **not** a base dependency: the PI-RNN is downstream of the benchmark,
and running an LLE should not require a ~1 GB deep-learning framework. Without
it, `tests/test_model.py` and `tests/test_ablation_encoders.py` are dropped from
collection (reported in the pytest header); everything else runs.

### Reproducible environment

For a number that must match a published one, use the pinned, hash-verified
toolchain rather than the floating one — `requirements.lock.txt` pins the exact
`jaxlib` the committed golden trajectories were produced with, which is what
makes the 0-ULP identity check meaningful:

```bash
docker build -t sc . && docker run --rm sc          # fast suite in the pinned image
```

or directly:

```bash
pip install --require-hashes -r requirements.lock.txt
pip install --no-deps --no-build-isolation -e .
SOLITON_STRICT_ULP=1 pytest tests/test_noise_off_identity.py -q
```

`environment.yml` is a conda convenience and is explicitly *not* the
bit-identity environment — see the caveat in its header.

> **Bit-identity needs fixed hardware, not just pinned versions.** On a CI
> runner carrying the goldens' exact toolchain (jax/jaxlib 0.10.2, numpy 2.4.6,
> python 3.11.15) the comparison still differs by `max_abs_diff = 6.2e-19`,
> because XLA reassociates reductions to suit the CPU it compiles for. The same
> command reproduces at 0 ULP on the machine that wrote the goldens. Treat
> `SOLITON_STRICT_ULP=1` as a fixed-hardware check, and quote the hardware
> alongside the versions in any reproducibility claim. Details in
> [`CONTRIBUTING.md`](CONTRIBUTING.md).

### Tests

```bash
pytest -q                      # fast suite (~9 min on 4 cores)
pytest -q --runslow            # + the large-grid cases
pytest -m pylle_full           # needs Julia and a pyLLE environment
```

## Citing

See [`CITATION.cff`](CITATION.cff). Changes that move numerical output follow
the bit-identity rule in [`CONTRIBUTING.md`](CONTRIBUTING.md); release history is
in [`CHANGELOG.md`](CHANGELOG.md).

## Usage Overview

- Place and tune physical constants in `config/tfln_params.yaml`.
- Implement solvers and noise models under `simulator/`.
- Generate datasets with scripts in `data/`.
- Develop and train PI-RNN models in `model/`.
- Integrate control loops and hardware APIs in `control/`.
- Run evaluation and visualization workflows from `analysis/`.
- Use `notebooks/exploration.ipynb` for exploratory experiments.
- Add validation coverage in `tests/` as modules are implemented.

## Project Layout

```text
tfln-soliton-control/
├── README.md
├── requirements.txt
├── config/
│   └── tfln_params.yaml
├── simulator/
├── data/
├── model/
├── control/
├── analysis/
├── notebooks/
└── tests/
```
