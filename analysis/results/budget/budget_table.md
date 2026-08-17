# Stochastic-LLE noise budget

- generated: `2026-08-17T07:38:45Z`
- git commit: `4880653e7966`
- seeds per cell: 2 (common random numbers: [100, 101])
- quick mode: True
- thermal_feedback pinned ON in every cell: True
- amplitude preset: {'name': 'ECDL flagship (analysis.noise_validation_campaign)', 'pump_freq_noise_h0_hz2_per_hz': 3000.0, 'pump_freq_noise_hm1_hz3_per_hz': 10000000000.0, 'pump_rin_floor_dbc_per_hz': -150.0, 'trn_psd_model': 'kondratiev_gorodetsky', 'trn_R_m': 0.0009298, 'trn_da_m': 2.2e-06, 'trn_db_m': 4e-07}

## Budget table

Cell = mean ± SEM over seeds, except `step_jitter`, whose budget statistic is the seed-to-seed **std** (the jitter itself).

| channel set | u_int_rms_frac [dimensionless] | s_rep (1e6 Hz) [Hz^2/Hz] | line_linewidth (mu=0) [Hz] | timing_jitter [fs] | step_jitter (N<=2) [kappa] |
|---|---|---|---|---|---|
| `all_off` | 1.395e-4 | — | 0 | — | 0 |
| `quantum` | 1.725e-4 ± 1.0e-6 | 4.898e5 ± 6.8e4 | 0 | — | 0 |
| `trn` | 1.765e-4 ± 9.9e-6 | — | 0 | — | 0 |
| `pyro_eo` | 1.765e-4 ± 9.9e-6 | — | 0 | — | 0 |
| `fsr` | 1.765e-4 ± 9.9e-6 | — | 0 | — | 0 |
| `dT_family` | 1.765e-4 ± 9.9e-6 | — | 0 | — | 0 |
| `pump_fn` | 0.02881 ± 0.0035 | — | 0 | — | 0 |
| `pump_rin` | 0.002014 ± 7.0e-5 | — | 0 | — | 0 |
| `all_on` | 0.02919 ± 0.0036 | 4.896e5 ± 6.8e4 | 0 | — | 0 |
| `loo_quantum` | 0.02918 ± 0.0036 | — | 0 | — | 0 |
| `loo_dT_family` | 0.02918 ± 0.0036 | 4.896e5 ± 6.8e4 | 0 | — | 0 |
| `loo_pump_fn` | 0.002021 ± 1.0e-4 | 4.898e5 ± 6.8e4 | 0 | — | 0 |
| `loo_pump_rin` | 0.02882 ± 0.0035 | 4.896e5 ± 6.8e4 | 0 | — | 0 |

## Record length per cell

| observable | point | record | t_slow [RT] | duration | Fourier floor |
|---|---|---|---|---|---|
| u_int_rms_frac | value | fast | 16000 | 0.6504 µs | 1.538e+06 Hz |
| s_rep | 1e6 Hz | slow | 45000 | 1.829 µs | 5.467e+05 Hz |
| line_linewidth | mu=0 | slow | 45000 | 1.829 µs | 5.467e+05 Hz |
| timing_jitter | value | slow | 45000 | 1.829 µs | 5.467e+05 Hz |
| step_jitter | N<=2 | staircase | 4950 | 0.2012 µs | n/a |

## u_int_rms_frac — contributions (mean)

| set | OAT: X[set]−X[all_off] | LOO: X[all_on]−X[all_on∖set] |
|---|---|---|
| `quantum` | 3.295e-5 ± 1.0e-6 | 6.875e-6 ± 0.0051 |
| `trn` | 3.703e-5 ± 9.9e-6 | — |
| `pyro_eo` | 3.703e-5 ± 9.9e-6 | — |
| `fsr` | 3.703e-5 ± 9.9e-6 | — |
| `dT_family` | 3.703e-5 ± 9.9e-6 | 4.669e-6 ± 0.0051 |
| `pump_fn` | 0.02867 ± 0.0035 | 0.02716 ± 0.0036 |
| `pump_rin` | 0.001875 ± 7.0e-5 | 3.631e-4 ± 0.005 |

**Interaction residual** (value): -0.00157 ± 0.005

Interaction residual for 'u_int_rms_frac': X[all_on] - X[all_off] - sum over {quantum, dT_family, pump_fn, pump_rin} of (X[set] - X[all_off]). A LARGE residual is expected, not anomalous. Three reasons (their relative size is not asserted here -- compare the OAT rows to see it): (1) The channels do not superpose. The LLE is nonlinear and the soliton is an attractor: two channels driving it together move the operating point in a way neither does alone, so the observables are not additive functionals of the drives. (2) trn, pyro_eo and fsr are ONE temperature realization delta_T(t) seen through three coupling coefficients, not three independent sources. That is exactly why the grouping uses the dT_family row rather than summing trn, pyro_eo and fsr separately -- summing them would triple-count one random variable. (3) all_on additionally carries tccr, which is NOT in the grouping. On this SiN device eo_r33_m_per_v = 0 so sigma_tccr = 0 and tccr contributes exactly zero, making this term vanish here -- but on a chi(2) platform (TFLN) it would not, and the grouping would need a tccr row before the residual could be read as pure interaction. Residual magnitude across the 1 resolved point(s) of this observable: min -0.00157, max -0.00157.

## s_rep — contributions (mean)

| set | OAT: X[set]−X[all_off] | LOO: X[all_on]−X[all_on∖set] |
|---|---|---|
| `quantum` | — | — |
| `trn` | — | — |
| `pyro_eo` | — | — |
| `fsr` | — | — |
| `dT_family` | — | -0.06735 ± 9.7e4 |
| `pump_fn` | — | -192.3 ± 9.7e4 |
| `pump_rin` | — | -5.856e-4 ± 9.7e4 |

**Interaction residual** (1e6 Hz): —

Interaction residual for 's_rep': X[all_on] - X[all_off] - sum over {quantum, dT_family, pump_fn, pump_rin} of (X[set] - X[all_off]). A LARGE residual is expected, not anomalous. Three reasons (their relative size is not asserted here -- compare the OAT rows to see it): (1) The channels do not superpose. The LLE is nonlinear and the soliton is an attractor: two channels driving it together move the operating point in a way neither does alone, so the observables are not additive functionals of the drives. (2) trn, pyro_eo and fsr are ONE temperature realization delta_T(t) seen through three coupling coefficients, not three independent sources. That is exactly why the grouping uses the dT_family row rather than summing trn, pyro_eo and fsr separately -- summing them would triple-count one random variable. (3) all_on additionally carries tccr, which is NOT in the grouping. On this SiN device eo_r33_m_per_v = 0 so sigma_tccr = 0 and tccr contributes exactly zero, making this term vanish here -- but on a chi(2) platform (TFLN) it would not, and the grouping would need a tccr row before the residual could be read as pure interaction.

## line_linewidth — contributions (mean)

| set | OAT: X[set]−X[all_off] | LOO: X[all_on]−X[all_on∖set] |
|---|---|---|
| `quantum` | 0 | 0 |
| `trn` | 0 | — |
| `pyro_eo` | 0 | — |
| `fsr` | 0 | — |
| `dT_family` | 0 | 0 |
| `pump_fn` | 0 | 0 |
| `pump_rin` | 0 | 0 |

**Interaction residual** (mu=0): 0

Interaction residual for 'line_linewidth': X[all_on] - X[all_off] - sum over {quantum, dT_family, pump_fn, pump_rin} of (X[set] - X[all_off]). A LARGE residual is expected, not anomalous. Three reasons (their relative size is not asserted here -- compare the OAT rows to see it): (1) The channels do not superpose. The LLE is nonlinear and the soliton is an attractor: two channels driving it together move the operating point in a way neither does alone, so the observables are not additive functionals of the drives. (2) trn, pyro_eo and fsr are ONE temperature realization delta_T(t) seen through three coupling coefficients, not three independent sources. That is exactly why the grouping uses the dT_family row rather than summing trn, pyro_eo and fsr separately -- summing them would triple-count one random variable. (3) all_on additionally carries tccr, which is NOT in the grouping. On this SiN device eo_r33_m_per_v = 0 so sigma_tccr = 0 and tccr contributes exactly zero, making this term vanish here -- but on a chi(2) platform (TFLN) it would not, and the grouping would need a tccr row before the residual could be read as pure interaction. Residual magnitude across the 5 resolved point(s) of this observable: min -1.27e+08, max 3.546e+06.

## timing_jitter — contributions (mean)

| set | OAT: X[set]−X[all_off] | LOO: X[all_on]−X[all_on∖set] |
|---|---|---|
| `quantum` | — | — |
| `trn` | — | — |
| `pyro_eo` | — | — |
| `fsr` | — | — |
| `dT_family` | — | — |
| `pump_fn` | — | — |
| `pump_rin` | — | — |

**Interaction residual** (value): —

Interaction residual for 'timing_jitter': X[all_on] - X[all_off] - sum over {quantum, dT_family, pump_fn, pump_rin} of (X[set] - X[all_off]). A LARGE residual is expected, not anomalous. Three reasons (their relative size is not asserted here -- compare the OAT rows to see it): (1) The channels do not superpose. The LLE is nonlinear and the soliton is an attractor: two channels driving it together move the operating point in a way neither does alone, so the observables are not additive functionals of the drives. (2) trn, pyro_eo and fsr are ONE temperature realization delta_T(t) seen through three coupling coefficients, not three independent sources. That is exactly why the grouping uses the dT_family row rather than summing trn, pyro_eo and fsr separately -- summing them would triple-count one random variable. (3) all_on additionally carries tccr, which is NOT in the grouping. On this SiN device eo_r33_m_per_v = 0 so sigma_tccr = 0 and tccr contributes exactly zero, making this term vanish here -- but on a chi(2) platform (TFLN) it would not, and the grouping would need a tccr row before the residual could be read as pure interaction.

## step_jitter — contributions (std)

| set | OAT: X[set]−X[all_off] | LOO: X[all_on]−X[all_on∖set] |
|---|---|---|
| `quantum` | 0 | 0 |
| `trn` | 0 | — |
| `pyro_eo` | 0 | — |
| `fsr` | 0 | — |
| `dT_family` | 0 | 0 |
| `pump_fn` | 0 | 0 |
| `pump_rin` | 0 | 0 |

**Interaction residual** (N<=2): 0

Interaction residual for 'step_jitter': X[all_on] - X[all_off] - sum over {quantum, dT_family, pump_fn, pump_rin} of (X[set] - X[all_off]). A LARGE residual is expected, not anomalous. Three reasons (their relative size is not asserted here -- compare the OAT rows to see it): (1) The channels do not superpose. The LLE is nonlinear and the soliton is an attractor: two channels driving it together move the operating point in a way neither does alone, so the observables are not additive functionals of the drives. (2) trn, pyro_eo and fsr are ONE temperature realization delta_T(t) seen through three coupling coefficients, not three independent sources. That is exactly why the grouping uses the dT_family row rather than summing trn, pyro_eo and fsr separately -- summing them would triple-count one random variable. (3) all_on additionally carries tccr, which is NOT in the grouping. On this SiN device eo_r33_m_per_v = 0 so sigma_tccr = 0 and tccr contributes exactly zero, making this term vanish here -- but on a chi(2) platform (TFLN) it would not, and the grouping would need a tccr row before the residual could be read as pure interaction. Residual magnitude across the 1 resolved point(s) of this observable: min 0, max 0.
