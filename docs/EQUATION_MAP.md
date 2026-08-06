# Equation map

Generated from `simulator/equation_map.py`. Reference: arXiv:2604.05897v1 (Herr, Tikan & Kippenberg, 7 Apr 2026); equation and section numbers are v1 numbers.

`--` means the repository does not record that cross-reference. It is left blank rather than guessed.

## Channel summary

| Channel | Paper eq. | Section | Enters as | Add/mult | Colour | Units | Code symbol | Shares source with | Validated by |
|---|---|---|---|---|---|---|---|---|---|
| quantum_vacuum | Eq. 126 | Sec. V.B.2 | additive field increment | additive | white | sqrt(J) (field increment; \|E\|^2 in J) | simulator.lle_solver._qnoise_increment | -- | test_injection_variance_matches_prescription, test_vacuum_equilibrium_occupation, test_vacuum_seed_occupancy_half_photon, test_disabled_path_adds_no_rng_to_scan_body, test_config_missing_keys_default_off |
| trn | Eqs. 129-130 | -- | detuning phase rotation | multiplicative | lorentzian(tau_th) \| kondratiev_gorodetsky \| csv | rad/s (detuning) | simulator.noise_models.TRNoise | pyro_eo, fsr | test_single_pole_bit_identical_to_legacy_ar1, test_trn_disabled_returns_exact_zeros, test_tk_zero_collapses_all_delta_t_channels, test_variance_conserved_across_psd_models |
| pyro_eo | -- | -- | detuning phase rotation | multiplicative | lorentzian(tau_th) \| kondratiev_gorodetsky \| csv | rad/s (detuning) | simulator.noise_models.PyroEONoise | trn, fsr | test_sample_with_delta_t_consistency, test_csv_delta_omega_units_share_delta_t_with_pyroeo, test_tk_zero_collapses_all_delta_t_channels |
| tccr | -- | -- | detuning phase rotation | multiplicative | lorentzian(tau_carrier) | rad/s (detuning) | simulator.noise_models.TCCRNoise | -- | test_disabling_trn_does_not_shift_tccr_stream, test_single_pole_bit_identical_to_legacy_ar1 |
| pump_freq_noise | -- | Sec. V.B.4 | detuning phase rotation | multiplicative | h0 + h-1/f | rad/s (detuning; sample_freq returns 2*pi*delta_nu) | simulator.noise_models.PumpNoise.sample_freq | -- | test_freq_noise_psd_fidelity, test_sign_convention_exact, test_linear_response_transfer, test_flag_off_bit_identical_to_legacy |
| pump_rin | -- | Sec. V.B.5 | drive amplitude scale | mixed | rin_floor + rin_excess*(f_c/f) below f_c | dimensionless (relative intensity epsilon) | simulator.noise_models.PumpNoise.sample_rin | -- | test_rin_psd_fidelity, test_rin_energy_balance, test_rin_clip_enforced_and_warns, test_flag_off_bit_identical_to_legacy |
| fsr | -- | Sec. V.B.1 | mode-linear detuning | multiplicative | lorentzian(tau_th) \| kondratiev_gorodetsky \| csv | rad/s (delta_D1; mode mu sees mu*delta_D1) | simulator.lle_solver._delta_t_sequences | trn, pyro_eo | test_fsr_constant_dd1_exact_phase, test_fsr_tk_zero_channel_identically_zero, test_fsr_without_trn_warns_and_is_zero |
| thermal_feedback | -- | -- | detuning phase rotation | multiplicative | deterministic (no stochastic source) | K (Delta T); enters as rad/s after the thermo-optic pull | simulator.lle_solver._thermal_params | -- | test_v3_thermal_sign, test_all_off_leaves_thermal_feedback_at_its_passed_value |

## Equations

### `quantum_vacuum` (Sec. V.B.2, Eq. 126)

Implemented by `simulator.lle_solver._qnoise_increment` in `simulator/lle_solver.py`.

Continuum:

```latex
\partial_t E = \dots + \sqrt{\kappa}\,\hat\xi_\mu(t),\quad \langle \hat\xi_\mu(t)\,\hat\xi^{\dagger}_{\mu'}(t')\rangle = \delta(t-t')\,\delta_{\mu\mu'}
```

Discretized (as implemented):

```latex
E_j \leftarrow E_j + \sigma_q\,(g_j + i\,h_j),\quad \sigma_q = \sqrt{\hbar\omega_0\,\kappa\,n_\tau\,\Delta t/4},\quad g_j,h_j \sim \mathcal{N}(0,1)\ \text{i.i.d. per fast-time sample},\ \Delta t = t_R/M\ \text{(or } t_R \text{ at roundtrip cadence)}
```

### `trn` (Eqs. 129-130)

Implemented by `simulator.noise_models.TRNoise` in `simulator/noise_models.py`.

Continuum:

```latex
\delta\omega_{\mathrm{TRN}}(t) = C_{\mathrm{pull}}\,\delta T(t),\quad C_{\mathrm{pull}} = \frac{\omega_0}{n_0}\left(\frac{dn}{dT} + n_0\alpha_L\right),\quad \langle \delta T^2\rangle = \frac{k_B T^2}{\rho C_p V},\quad S_{\delta T}(f) = \frac{4\,\langle\delta T^2\rangle\,\tau_{\mathrm{th}}}{1 + (2\pi f \tau_{\mathrm{th}})^2}
```

Discretized (as implemented):

```latex
\delta T_{k+1} = a\,\delta T_k + \sigma\sqrt{1-a^2}\;w_k,\quad a = e^{-t_R/\tau_{\mathrm{th}}},\ w_k \sim \mathcal{N}(0,1);\quad \delta\omega_k = C_{\mathrm{pull}}\,\delta T_k
```

### `pyro_eo`

Implemented by `simulator.noise_models.PyroEONoise` in `simulator/noise_models.py`.

Continuum:

```latex
\delta\omega_{\mathrm{pyro}}(t) = C_{\mathrm{pyro}}\,\delta T(t),\quad C_{\mathrm{pyro}} = \frac{\omega_0\,n_0^2\,r_{33}\,p}{2\,\varepsilon_0\,\varepsilon_{r,\mathrm{eff}}}
```

Discretized (as implemented):

```latex
\delta\omega^{\mathrm{tot}}_k = C_{\mathrm{pull}}\,\delta T_k - C_{\mathrm{pyro}}\,\delta T_k + \delta\omega^{\mathrm{TCCR}}_k,\quad \delta T_k\ \text{the identical realization used by TRN}
```

### `tccr`

Implemented by `simulator.noise_models.TCCRNoise` in `simulator/noise_models.py`.

Continuum:

```latex
\frac{d\omega}{dN_s} = -\frac{\omega_0 n_0^2 r_{33}}{2}\,\frac{e}{\varepsilon_0 \varepsilon_{r} A_{\mathrm{eff}}},\quad S_{\mathrm{TCCR}}(f) = \frac{S_0}{1+(2\pi f\tau_c)^2},\quad S_0 = \left(\frac{d\omega}{dN_s}\right)^2 N_{s,\mathrm{eq}}\,2\tau_c
```

Discretized (as implemented):

```latex
\delta\omega^{\mathrm{TCCR}}_{k+1} = a_c\,\delta\omega^{\mathrm{TCCR}}_k + \sigma_{\mathrm{TCCR}}\sqrt{1-a_c^2}\;w_k,\quad a_c = e^{-t_R/\tau_c},\ \tau_c = \tau_{\mathrm{carrier}}
```

### `pump_freq_noise` (Sec. V.B.4)

Implemented by `simulator.noise_models.PumpNoise.sample_freq` in `simulator/noise_models.py`.

Continuum:

```latex
S_{\delta\nu}(f) = h_0 + \frac{h_{-1}}{f}\ [\mathrm{Hz^2/Hz}],\quad \Delta\nu_L = \pi h_0,\quad \delta\omega \equiv \omega_{\mathrm{res}} - \omega_p \Rightarrow \delta\omega \mathrel{+}= -2\pi\,\delta\nu_p(t)
```

Discretized (as implemented):

```latex
\delta\nu_k = w_k\sqrt{h_0 f_s/2} + \mathrm{FFT}^{-1}\!\left[\sqrt{h_{-1}/\max(f,f_1)}\right]_k,\quad f_s = 1/t_R,\ f_1 = f_s/N;\quad \delta\omega_k \mathrel{+}= -2\pi\,\delta\nu_k
```

### `pump_rin` (Sec. V.B.5)

Implemented by `simulator.noise_models.PumpNoise.sample_rin` in `simulator/noise_models.py`.

Continuum:

```latex
P_{\mathrm{in}}(t) = \bar P_{\mathrm{in}}\,(1+\epsilon(t)),\quad S_\epsilon(f) = 10^{\mathrm{floor}/10} + 10^{\mathrm{excess}/10}\frac{f_c}{f}\ (f<f_c),\quad S_\epsilon(f) = 10^{\mathrm{floor}/10}\ (f \ge f_c)
```

Discretized (as implemented):

```latex
s_k = \max(1+\epsilon_k,\,0),\quad F_k = \sqrt{\max(\kappa_c \bar P_{\mathrm{in}} s_k,\,0)}\;\Delta t_{\mathrm{sub}},\quad E \leftarrow E + F_k\ \text{(pump kick, held over the round trip)}
```

### `fsr` (Sec. V.B.1)

Implemented by `simulator.lle_solver._delta_t_sequences` in `simulator/lle_solver.py`.

Continuum:

```latex
\delta D_1(t) = \frac{D_1}{\omega_0}\,C_{\mathrm{pull}}\,\delta T(t),\quad D_1 = 2\pi\,\mathrm{FSR};\quad \text{mode } \mu \text{ acquires the detuning } \mu\,\delta D_1(t)
```

Discretized (as implemented):

```latex
\mathcal{L}_\mu \leftarrow \mathcal{L}_\mu - i\,\mu\,\delta D_1(t_k)\,\Delta t_{\mathrm{sub}},\quad \delta D_1(t_k) = \frac{D_1}{\omega_0} C_{\mathrm{pull}}\,\delta T_k,\ \delta T_k\ \text{regenerated from the SAME noise keys as TRN}
```

### `thermal_feedback`

Implemented by `simulator.lle_solver._thermal_params` in `simulator/lle_solver.py`.

Continuum:

```latex
\frac{d\,\Delta T}{dt} = -\frac{\Delta T}{\tau_{\mathrm{th}}} + \frac{\Gamma_{\mathrm{th}}\,P_{\mathrm{abs}}}{\rho C_p V},\quad P_{\mathrm{abs}} = \kappa_i U_{\mathrm{int}}/t_R;\quad \delta\omega_{\mathrm{th}} = -\frac{\omega_0}{n_0}\frac{dn}{dT}\,\Delta T
```

Discretized (as implemented):

```latex
\Delta T_{k+1} = \Delta T_k + \Delta t_{\mathrm{fine}}\left(-\frac{\Delta T_k}{\tau_{\mathrm{th}}} + \frac{\Gamma_{\mathrm{th}} P_{\mathrm{abs},k}}{\rho C_p V}\right)\ \text{(forward Euler, once per fine step)}
```
