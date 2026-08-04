# Noise-channel inventory (read-only audit)

Audit date: 2026-08-04. Scope: `simulator/noise_models.py`, `simulator/colored_noise.py`,
`simulator/lle_solver.py`, `config/sin_params.yaml`,
`analysis/run_detuning_sweep.py::write_noise_off_config`, `simulator/state_labeler.py`.

Reference: Herr, Tikan & Kippenberg, **arXiv:2604.05897v1** (7 Apr 2026). Equation numbers
below are v1 numbers.

All line numbers are from the working tree at audit time. **No file was modified.**
JAX is not installed in this container, so nothing here was executed — every statement is
derived from source. Claims that rest on an inference rather than a literal source token are
marked inline.

Config defaults quoted are the values committed in `config/sin_params.yaml`
(a **SiN** device: `eo_r33_m_per_v = 0.0`, `pyroelectric_coeff_c_per_m2_k = 0.0`).
"absent → *x*" means the key is **not present in the config file** and the code's own
`cfg.get(..., x)` default applies.

---

## Channel table

| # | Channel | Generator (file:line) | Enters the EOM at (file:line, which term) | Add / mult on E | White / colored (τ_c or PSD shape) | Config keys → current default | How it is disabled | "Off" proven bit-identical? (test) | Shares a random source with |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Quantum vacuum** (Sec. V.B.2, Eq. 126) | `_qnoise_increment` `lle_solver.py:361-387`; per-quadrature scale `lle_solver.py:1372-1374`; cold-start vacuum seed `_vac_draw` `lle_solver.py:1609-1616` | `lle_solver.py:604-607` — `e_next = e_next + _qnoise_increment(...)`, i.e. the additive Langevin drive √κ·ξ_μ(t), injected after the Strang sub-step loop and **before** the absorber/validity masks. Second entry point: initial condition `lle_solver.py:1617` (`e0_init = e_cw_traj + vac`) | **Additive** (complex, time-domain, per fast-time sample) | **White.** i.i.d. complex Gaussian per fast-time sample per injection event; flat PSD. Per-quadrature std √(ħω₀·κ·n_tau·dt/4), `lle_solver.py:1372-1374`; dt = dt_fine or t_r per the cadence enum (`lle_solver.py:1371`) | `quantum_noise_enabled` → **0** (`sin_params.yaml:111`)<br>`quantum_noise_seed_vacuum_init` → **1** (`:116`)<br>`quantum_noise_injection_cadence` → **0** = fine (`:124`)<br>`hbar_omega0_j` → **0.0** = auto (`:130`)<br>`labeler_vacuum_floor_margin` → **10.0** (`:141`)<br>`labeler_envelope_smooth_modes` → **8** (`:146`)<br>kwargs: `lle_solver.py:937-939` | **Explicit boolean flag.** `quantum_noise_enabled` resolved at `lle_solver.py:1340-1342`, validated 0/1 by `_as_flag` (`:1334-1338`), passed as a **static** Python bool (`static_argnums` 27, `lle_solver.py:911`); the branch at `lle_solver.py:604` then traces zero extra ops. Off also forces `qnoise_scale=0.0` and the inactive labeler params (`:1424-1428`). `write_noise_off_config` sets it 0 (`run_detuning_sweep.py:405`) | **Yes, within the current tree.** `tests/test_quantum_noise.py::test_config_missing_keys_default_off` — `np.array_equal` of `E_snapshots` across {keys absent, key=0, kwarg False}. Structural: `tests/test_quantum_noise.py::test_disabled_path_adds_no_rng_to_scan_body` — zero RNG primitives in the traced scan body.<br>**Caveat:** no in-repo comparison against an archived pre-quantum solver; `tests/test_quantum_noise.py:684-686` states that check is "(out-of-repo)". `test_flag_off_legacy_seed_statistics` is statistical (5 % tol), not bit-identity | **Nothing.** Own chain `key_qnoise` from `_legacy_rng_chain` (`noise_models`-independent; `lle_solver.py:822-826`), split into `key_qnoise_inj` / `key_qnoise_seed` at `lle_solver.py:1433-1434`. Those two are siblings of each other only. Pinned by `tests/test_dataset_generator.py::test_key_isolation` |
| 2 | **TRN** (thermorefractive; Eqs. 129-130) | Solver-consumed sequence: `TotalNoise.sample_with_delta_t` `noise_models.py:425-460`, specifically `trn_noise = self.c_pull * temp_noise` **`noise_models.py:446`**. Underlying dT: `_ar1_samples` `noise_models.py:135-146` (`single_pole`) or `colored_noise.synthesize_from_psd:78-124` (colored), selected `noise_models.py:437-445`. Class `TRNoise` `noise_models.py:149-232`. Host entry `_detuning_noise_sequences` `lle_solver.py:829-858` | `lle_solver.py:640` `freq_noise = noise_sequence[step_idx]` → `lle_solver.py:566` `delta_omega_eff = dw_step + thermal_shift + freq_noise` → `lle_solver.py:571` `lin_exp = (-kappa/2 - 1j*disp - 1j*delta_omega_eff)*dt_sub`. **Term: the detuning term −i·δω·E in the linear half-step** | **Multiplicative** on E (enters as a phase factor exp(−i·δω_noise·dt) in the linear operator). It is *additive in the detuning*, multiplicative on the field | **Colored.** `single_pole` (default): Lorentzian, correlation time **τ_th = 5.0e-6 s**; S_δω(f) = C_pull²·4·var_dT·τ_th/(1+(2πfτ_th)²), `noise_models.py:224-229`, factory `colored_noise.py:147-164`.<br>`kondratiev_gorodetsky`: Eq. 130 shape ∝ ω^−1/2·[1+(ωτ_d)^3/4]^−2, renormalized to the Eq. 129 variance, `colored_noise.py:167-257`.<br>`csv`: tabulated, log-log interpolated, flat-clamped outside the span, `colored_noise.py:260-296` | `T_k` → **300.0** (`:62`)<br>`tau_th_s` → **5.0e-6** (`:63`)<br>`dn_dT_per_k` → **2.45e-5** (`:58`)<br>`rho_kg_per_m3` → **3.17e+3** (`:59`)<br>`Cp_j_per_kg_k` → **700.0** (`:60`)<br>`mode_volume_m3` → **1.954e-14** (`:46`)<br>`n0` → **1.87** (`:41`)<br>`pump_wavelength_m` → **1.55e-6** (`:39`)<br>`fsr_hz` → **2.46e+10** (`:9`, sets t_r and f_s)<br>`trn_psd_model` → **single_pole** (`:195`)<br>`trn_R_m`/`trn_da_m`/`trn_db_m` → **null** (`:201-203`)<br>`trn_psd_csv_path` → **null** (`:206`)<br>`trn_csv_units` → **S_delta_T** (`:211`)<br>`alpha_L_per_k` → **0.0** (`:216`)<br>`kappa_th_w_per_m_k` → **30.0** (`:61`, K-G model only)<br>`legacy_segment_noise` → **1** (`:236`, dataset-generator cadence only) | **NO explicit boolean flag.** Only implicit, via `T_k = 0`: `var_delta_t = k_B·T_k²/(ρ·Cp·V)` → 0 (`noise_models.py:182`) plus the explicit zero-PSD short circuit for every model at `noise_models.py:65-68`. That is exactly what `write_noise_off_config` does (`run_detuning_sweep.py:404`). There is no way to silence TRN alone — `T_k=0` also kills pyro-EO, the expansion pull and FSR | **Partly.** `tests/test_noise_metrology.py::test_tk_zero_collapses_all_delta_t_channels` asserts the sampled arrays are **exactly** `0.0` at T_k=0 for `single_pole` and `kondratiev_gorodetsky` (the `csv` model is **not** covered by that test, though `noise_models.py:65-68` short-circuits it too).<br>Stream bit-identity of the ON default vs the pre-colored-noise AR(1): `tests/test_colored_noise.py::test_single_pole_bit_identical_to_legacy_ar1` (`np.array_equal`, including the `_detuning_noise_sequences` solver surface).<br>**No test runs the full solver at T_k=0 and compares it to a noise-free solver**; the argument is "the added array is exactly +0.0", which is bit-neutral for finite operands | **Pyro-EO, thermal-expansion pull, and FSR — CONFIRMED.** One `temp_noise` is drawn at `noise_models.py:437-445` and consumed twice, at `:446` (TRN) and `:447` (pyro-EO). FSR: `lle_solver.py:1546` calls `_delta_t_sequences(noise_keys, ...)` with the **same** `noise_keys` created at `lle_solver.py:1328` and used by `_detuning_noise_sequences` at `lle_solver.py:1437`; `_delta_t_sequences` returns `sample_with_delta_t(k,N)[1]` (`lle_solver.py:881`, `:886`) — the identical `temp_noise`, regenerated from the identical `key_thermal` split (`noise_models.py:435`). **Independent of** TCCR (`key_tccr`, `noise_models.py:435/448`), quantum, and both pump channels |
| 3 | **Thermal-expansion pull** (the paper's "dimensional fluctuation" companion of TRN) | **Not a separate stochastic process** — a modification of the TRN pull coefficient: `noise_models.py:186-189` `c_pull = (omega_0/n0)*(dn_dT + n0*alpha_L)`. No independent generator, no independent draw | Wherever `c_pull` is used: TRN detuning `noise_models.py:446` → `lle_solver.py:640/566/571`; FSR amplitude `lle_solver.py:1548`. **Not** in the pyro-EO pull: `pyro_coeff` (`noise_models.py:270-273`) has no α_L term; `PyroEONoise` recomputes a local `_c_pull` at `noise_models.py:279-281` used **only** to map a `S_delta_omega` CSV into K² units (`noise_models.py:283`, `:108-116`) | **Multiplicative** on E (same detuning phase as TRN — it is a scalar rescale of the TRN amplitude) | **Same color as TRN** — it is the same dT(t) sequence, only the pull constant changes | `alpha_L_per_k` → **0.0** (`sin_params.yaml:216`); inherits every TRN key in row 2 | **NO boolean flag — value-gated.** `alpha_L_per_k = 0.0` makes `dn_dT + n0*0.0 == dn_dT` exactly in IEEE arithmetic (`noise_models.py:184-189`). Also killed by `T_k = 0` | **Yes.** `tests/test_colored_noise.py::test_alpha_l_zero_is_bitwise_neutral_and_scales_pull` asserts `TRNoise(cfg_alpha0).c_pull == trn0.c_pull` with `==` (exact), and that a non-zero α_L scales the pull as documented | **TRN, pyro-EO, FSR** — by construction it *is* the TRN dT realization (same `temp_noise`, `noise_models.py:437-447`) |
| 4 | **Pyro-EO** | Solver-consumed sequence: `pyroeo_noise = self.pyro_coeff * temp_noise` **`noise_models.py:447`**, combined with a **minus** sign at `noise_models.py:454`. Coefficient `noise_models.py:270-273`. Class `PyroEONoise` `noise_models.py:235-306`; its own `sample` (`:291-296`) uses an independently-keyed AR(1) and is **standalone/diagnostic only — not the solver path** | Identical to TRN: `lle_solver.py:640` → `:566` → `:571`. **Term: −i·δω·E** | **Multiplicative** on E (detuning phase) | **Colored — same dT spectrum as TRN** (it is the same sequence): default Lorentzian τ_th = 5.0e-6 s. Closed form S = pyro_coeff²·S_dT, `noise_models.py:298-306` | `eo_r33_m_per_v` → **0.0** (`:239`)<br>`pyroelectric_coeff_c_per_m2_k` → **0.0** (`:240`)<br>`eps_r_z` → absent → **28.0** (`noise_models.py:253`)<br>`t_ln_m` → absent → **4.0e-7** (`noise_models.py:257`)<br>`t_clad_top_m` → absent → **1.0e-6** (`:258`)<br>`t_clad_bot_m` → absent → **2.0e-6** (`:259`)<br>`eps_r_clad_top` → absent → **1.0** (`:260`)<br>`eps_r_clad_bot` → absent → **3.9** (`:261`)<br>plus every TRN thermal key (row 2) and `trn_psd_model` (`noise_models.py:277`)<br>*(see footnote [a] — the stack-geometry keys are commented out in the config at `:244-248` under different names)* | **NO boolean flag.** Zeroed implicitly by `eo_r33_m_per_v = 0` **or** `pyroelectric_coeff_c_per_m2_k = 0` → `pyro_coeff = 0` (`noise_models.py:270-273`); also killed by `T_k = 0` (`noise_models.py:65-68`, `:182`). Note the multiply at `noise_models.py:447` is still **traced and executed** — "off" is a zero value, not a removed op | **Weak.** No test asserts `pyro_coeff == 0` for the committed SiN config. Covered indirectly by:<br>`tests/test_colored_noise.py::test_single_pole_bit_identical_to_legacy_ar1` (the pyro term is part of the exact `np.array_equal` legacy reconstruction);<br>`tests/test_colored_noise.py::test_sample_with_delta_t_consistency` (asserts `combined ≈ (c_pull − pyro_coeff)·dT`, `rtol=2e-6` — **not** bit-exact);<br>`tests/test_noise_metrology.py::test_tk_zero_collapses_all_delta_t_channels` (T_k=0 path only).<br>**UNVERIFIED:** that `r33 = 0 ⇒ pyro contribution is exactly zero` is asserted by any test | **TRN, thermal-expansion pull, FSR — CONFIRMED**, same `temp_noise` object (`noise_models.py:437-447`). Independent of TCCR |
| 5 | **TCCR** (thermal carrier / surface-state) | `TCCRNoise` `noise_models.py:309-371`; sampled at `noise_models.py:448` `tccr_noise = self.tccr.sample(key_tccr, N)` → `_ar1_samples` `noise_models.py:368` | Summed into `combined` at `noise_models.py:454`, then identical to TRN: `lle_solver.py:640` → `:566` → `:571`. **Term: −i·δω·E** | **Multiplicative** on E (detuning phase) | **Colored.** Single-pole Lorentzian, **correlation time `tau_carrier`, default 1.0e-7 s** (`noise_models.py:313` — the key is absent from the config). S(f) = s0/(1+(2πf·τ_c)²), `noise_models.py:370-371`, with s0 = (dω/dN_s)²·N_s,eq·2τ_c (`noise_models.py:338`) | `tau_carrier_s` → absent → **1.0e-7** (`noise_models.py:313`)<br>`surface_state_density_per_m2` → absent → **1.0e16** (`noise_models.py:318`)<br>`eo_r33_m_per_v` → **0.0** (`:239`)<br>`eps_r_z` → absent → **28.0** (`noise_models.py:322`)<br>`effective_mode_area_m2` → **3.344e-12** (`:26`)<br>`n0` → **1.87** (`:41`)<br>`t_ln_m` → absent → **4.0e-7** (`noise_models.py:324`; **read but unused** in the formula — see the comment at `noise_models.py:333`)<br>`intrinsic_q` → **4.0e+7** (`:11`, warning threshold only, `noise_models.py:353-354`)<br>`pump_wavelength_m`, `fsr_hz`<br>`T_k` → **300.0** (`:62`) is stored at `noise_models.py:315` but **never enters** `s0_tccr`/`var_tccr` (`noise_models.py:334-339`) | **NO boolean flag.** Zeroed only via `eo_r33_m_per_v = 0.0` → `dw_dNs = 0` (`noise_models.py:335`) → `s0_tccr = 0` (`:338`) → `sigma_tccr = 0` (`:340`).<br>**NOT disableable via `T_k`** — the variance is T_k-independent (`noise_models.py:334-339`). `write_noise_off_config` sets only `T_k=0` (`run_detuning_sweep.py:404`); it leaves TCCR **active for any config with r33 ≠ 0**. The docstring acknowledges this in prose (`run_detuning_sweep.py:373-374`) but there is no code guard.<br>Even at σ=0 the AR(1) scan is still traced and a PRNG subkey is still consumed (`noise_models.py:435`, `:448`) | **Weak.** No test asserts the TCCR contribution is exactly zero for the committed config. `tests/test_colored_noise.py::test_single_pole_bit_identical_to_legacy_ar1` pins the TCCR AR(1) stream bit-exactly (`np.array_equal`) but that pins the *stream*, not that it is zero. `test_sample_with_delta_t_consistency` comments "SiN: TCCR is zero" and then asserts only `np.allclose(rtol=2e-6)`.<br>**UNVERIFIED:** any test proving `r33=0 ⇒ TCCR exactly 0`, and any test proving TCCR survives `T_k=0` (the behaviour that makes the noise-off sidecar non-deterministic on χ² platforms) | **Nothing.** Independent subkey `key_tccr` from `jax.random.split(key, 2)` at `noise_models.py:435`, consumed at `:448`. It shares only the *parent* per-trajectory key `noise_keys[i]` with the thermal channels — the two branches are independent streams |
| 6 | **Pump frequency noise** (Sec. V.B.4) | `PumpNoise.sample_freq` `noise_models.py:620-638` — white part `:631-632`, 1/f flicker via FFT synthesis `:633-637` → `_synthesize_from_onesided_psd` `noise_models.py:498-507` → `colored_noise.synthesize_from_psd:78-124` (`clamp_dc=False`). Host call site `lle_solver.py:1489-1493` | **Summed into the detuning array on the HOST** at `lle_solver.py:1497-1501` (`noise_sequences = noise_sequences + pump_freq_noise_history`), then `lle_solver.py:640` → `:566` → `:571`. **Term: −i·δω·E** — the same linear-operator detuning term as TRN. Sign −2π·δν_p applied at `lle_solver.py:1488` (override) / `:1492` (stochastic), per δω ≡ ω_res − ω_p | **Multiplicative** on E (detuning phase) | **Colored.** S_δν(f) = h₀ + h₋₁/f [Hz²/Hz], `noise_models.py:602-605`. White plateau h₀ (Δν_L = π·h₀, `noise_models.py:599`) + 1/f flicker with the DC bin clamped to f₁ = f_s/N (`noise_models.py:634-637`) | `pump_noise_enabled` → **0** (`:161`)<br>`pump_freq_noise_h0_hz2_per_hz` → **0.0** (`:167`)<br>`pump_freq_noise_hm1_hz3_per_hz` → **0.0** (`:169`)<br>`fsr_hz` (sets f_s, `noise_models.py:552-553`)<br>kwargs `pump_noise_enabled`, `pump_freq_noise_override` (`lle_solver.py:940-941`) | **Explicit boolean flag.** `pump_noise_enabled` validated 0/1 at `noise_models.py:554-560`; when off `_on = 0.0` zeroes `_h0`/`_hm1` (`noise_models.py:592-594`) and `sample_freq` early-returns exact zeros (`noise_models.py:627-628`). Host guard `lle_solver.py:1489` additionally requires `_h0 > 0 or _hm1 > 0`; the host add is skipped entirely unless `np.any(...)` (`lle_solver.py:1498`). Doubly inert here because the committed h₀ = h₋₁ = 0. `write_noise_off_config` forces 0 (`run_detuning_sweep.py:406`) | **Yes, within the current tree.** `tests/test_pump_noise.py::test_flag_off_bit_identical_to_legacy` — `np.array_equal` on `E_snapshots` and `U_int_history`, plus absence of the diagnostic keys. Supporting: `::test_disabled_forces_all_channels_inert`, `::test_disabled_solver_ignores_pump_numbers` | **Nothing.** Own subkey `key_pump_freq` from `jax.random.split(key_pump, 2)` at `lle_solver.py:1460`, split per trajectory at `:1490`. Sibling (independent) of `key_pump_rin`. The `key_pump` chain is *appended* to the legacy ladder (`lle_solver.py:1457-1459`) so it perturbs no legacy stream |
| 7 | **Pump RIN** (Sec. V.B.5) | `PumpNoise.sample_rin` `noise_models.py:640-678` — white floor `:650-653`, 1/f excess below f_c `:654-666`, clip at ε ≥ −1 `:678`. Host call `lle_solver.py:1508-1512`; scale array built `lle_solver.py:1518-1525` | `lle_solver.py:656` `scale_t = pump_scale_sequence[step_idx]` → `:657` `pin_t = pin*scale_t` → `:658-660` `kick_t = sqrt(max(kappa_c*pin_t,0))*dt_sub` → **`lle_solver.py:582` `e_pumped = e_sub + kick`** — the **pump/drive term**. Second (indirect) path: `pin_t` sets `p_trans` (`:680`) and the drive changes ⟨|E|²⟩ → P_abs (`:619`) → ΔT (`:624-628`) → `thermal_shift` (`:565`) → detuning | **Neither, directly.** It is **multiplicative on the drive amplitude F** (as √ of the power scale), and F enters E **additively** (`lle_solver.py:582`). Secondarily it reaches E multiplicatively through the thermal→detuning pathway | **Colored.** S_ε(f) = 10^(floor/10) + 10^(excess/10)·(f_c/f) for f < f_c, floor-only above f_c (`noise_models.py:607-617`). White floor + 1/f excess with corner f_c | `pump_noise_enabled` → **0** (`:161`)<br>`pump_rin_floor_dbc_per_hz` → **−300.0** (`:174`)<br>`pump_rin_excess_dbc_per_hz` → **−300.0** (`:175`)<br>`pump_rin_corner_hz` → **1.0e+4** (`:177`)<br>`fsr_hz` (sets f_s)<br>kwarg `pump_rin_epsilon_override` (`lle_solver.py:942`) | **Explicit boolean flag.** Off → `_on = 0.0` zeroes `_rin_floor_lin`/`_rin_excess_lin` (`noise_models.py:595-596`) and `sample_rin` early-returns zeros (`:646-647`); the host branch `lle_solver.py:1518-1525` then sets `pump_scale_sequence = None`, so the Python branch at `lle_solver.py:655-663` traces the legacy constant-kick path with zero extra ops.<br>*(footnote [b]: the enable predicate at `lle_solver.py:1519` keys on the **floor** only, not the excess)* | **Yes, within the current tree.** `tests/test_pump_noise.py::test_flag_off_bit_identical_to_legacy` (`np.array_equal`); supporting `::test_disabled_forces_all_channels_inert`, `::test_disabled_solver_ignores_pump_numbers` | **Nothing.** Own subkey `key_pump_rin` (`lle_solver.py:1460`), split per trajectory at `:1509`. Independent of `key_pump_freq` |
| 8 | **FSR / repetition-rate noise** (TRN-driven f_rep) | Built in the solver, not in `noise_models`: `lle_solver.py:1540-1549` `fsr_noise_sequences = (D1/omega0)*_trn.c_pull*_dt_seqs`, with dT from `_delta_t_sequences` `lle_solver.py:861-888` and `c_pull` from `TRNoise` (`noise_models.py:187-189`). Ultimate generator is therefore the TRN dT source (`_ar1_samples` `noise_models.py:135-146` / `colored_noise.py:78-124`) | `lle_solver.py:645` `delta_d1_t = fsr_noise_sequence[step_idx]` → `lle_solver.py:576` `lin_exp = lin_exp - 1j*(mu_grid*delta_d1)*dt_sub`, with `mu_grid` built at `lle_solver.py:544`. **Term: the per-mode linear detuning μ·δD₁(t) added to the linear operator** (a mode-number-proportional detuning, distinct from the uniform δω of rows 2-6) | **Multiplicative** on E (mode-dependent phase in the Fourier domain) | **Colored — identical spectrum to TRN**, since it is the same dT(t), scaled by (D₁/ω₀)·C_pull ≈ 1.27e-4·C_pull (`lle_solver.py:1548`, magnitude logged `:1556-1562`). Default correlation time τ_th = 5.0e-6 s | `fsr_noise_enabled` → **0** (`:225`)<br>`fsr_hz` → **2.46e+10** (`:9`, sets D₁ = 2π·FSR)<br>`pump_wavelength_m` → **1.55e-6** (`:39`, sets ω₀)<br>plus **every** TRN key of row 2 (`T_k`, `tau_th_s`, `dn_dT_per_k`, `alpha_L_per_k`, `n0`, `rho_kg_per_m3`, `Cp_j_per_kg_k`, `mode_volume_m3`, `trn_psd_model`, the K-G geometry, the CSV keys)<br>kwarg `fsr_delta_d1_override` (`lle_solver.py:944`) | **Explicit boolean flag.** `fsr_noise_enabled` resolved `lle_solver.py:1532-1533`, validated by `_as_flag` `:1534`; off → `fsr_noise_sequences = None` (`:1564`) → the three Python `is not None` branches at `lle_solver.py:541`, `:572`, `:644` trace zero extra ops (no `mu_grid`, no FMA, no gather). **Also** implicitly killed by `T_k = 0` even when the flag is on | **Partly.** `tests/test_noise_metrology.py::test_fsr_tk_zero_channel_identically_zero` — `np.array_equal` of `U_int_history` and `E_snapshots` between the channel-off run and `fsr_noise_enabled=True` at T_k=0, i.e. it proves the **enabled-but-zero** path is bit-identical to off. Source dT collapse: `::test_tk_zero_collapses_all_delta_t_channels`.<br>**UNVERIFIED:** no test asserts flag-off == pre-feature legacy for a `T_k ≠ 0` config; that rests on the `None`-branch construction, not on an assertion | **TRN, pyro-EO, thermal-expansion pull — CONFIRMED.** `lle_solver.py:1546` passes the **same** `noise_keys` (created `:1328`) that `_detuning_noise_sequences` consumes at `:1437`; `_delta_t_sequences` returns `TotalNoise.sample_with_delta_t(k,N)[1]` (`lle_solver.py:881` colored / `:886` legacy), which is the identical `temp_noise` drawn at `noise_models.py:437-445` from the identical `key_thermal` (`noise_models.py:435`). The dT is **regenerated**, not threaded — bit-consistency depends on that regeneration staying key-for-key identical (documented at `lle_solver.py:864-874`). Independent of TCCR, quantum, pump |

**Answer to the stated hypothesis (col. 9):** *TRN, pyro-EO and FSR all consume the same
`delta_T` realization* — **CONFIRMED**. TRN and pyro-EO share it by direct reuse of one array
(`noise_models.py:446-447`); FSR shares it by deterministic regeneration from the same
`noise_keys` (`lle_solver.py:1437` and `:1546` → `noise_models.py:435`). The
thermal-expansion pull is a fourth consumer of the same realization (it is a rescale of the
TRN pull, `noise_models.py:186-189`). **TCCR does *not*** share it — it takes the second
subkey of `jax.random.split(key, 2)` at `noise_models.py:435`.

### Table footnotes

- **[a]** `config/sin_params.yaml:244-248` comments out stack-geometry keys under names the
  code never reads (`t_SiN_m`), while the code reads `t_ln_m` (`noise_models.py:257`); the
  commented `eps_r_clad_top: 3.9` also disagrees with the code default `1.0`
  (`noise_models.py:260`). Numerically inert here only because `r33 = 0`.
- **[b]** `lle_solver.py:1518-1519`: the predicate that decides whether
  `pump_scale_sequence` is built is `pump_rin_epsilon_override is not None or (pump_on and
  _pump._rin_floor_lin > 0.0)` — it inspects `_rin_floor_lin` but **not** `_rin_excess_lin`.
  With the default floor of −300 dBc/Hz the linear floor is 1e-30 > 0, so in practice
  `pump_on` alone decides; an excess-only RIN config would be silently dropped only if the
  floor were exactly 0 (i.e. −inf dBc/Hz). Latent, not currently reachable.

---

## GAPS

### G1 — Channels with NO explicit boolean enable flag

Four of the eight channels are value-gated only. Only `quantum_noise_enabled`
(`sin_params.yaml:111`), `pump_noise_enabled` (`:161`) and `fsr_noise_enabled` (`:225`) exist
as switches.

- **TRN** — `simulator/noise_models.py:149` (`class TRNoise`). No `*_enabled` key is read
  anywhere in the class or in `TotalNoise` (`noise_models.py:374-485`). The only off switch
  is `T_k = 0` → `noise_models.py:182` (`var_delta_t`) and `noise_models.py:65-68` (explicit
  zero-PSD short circuit). `config/sin_params.yaml` has no TRN enable key — `trn_psd_model`
  (`:195`) selects a *spectrum*, never "off".
- **Pyro-EO** — `simulator/noise_models.py:235` (`class PyroEONoise`). Off only via
  `eo_r33_m_per_v = 0.0` (`sin_params.yaml:239`) or
  `pyroelectric_coeff_c_per_m2_k = 0.0` (`:240`) → `pyro_coeff = 0` at
  `noise_models.py:270-273`, or via `T_k = 0`. The multiply at `noise_models.py:447` and the
  subtraction at `:454` are executed regardless.
- **Thermal-expansion pull** — `simulator/noise_models.py:186-189`. Off only via
  `alpha_L_per_k = 0.0` (`sin_params.yaml:216`).
- **TCCR** — `simulator/noise_models.py:309` (`class TCCRNoise`). Off only via
  `eo_r33_m_per_v = 0.0` (`sin_params.yaml:239`) → `dw_dNs = 0` (`noise_models.py:335`) →
  `s0_tccr = 0` (`:338`) → `sigma_tccr = 0` (`:340`). **`T_k` does not gate it**: `T_k` is
  stored at `noise_models.py:315` but appears nowhere in `noise_models.py:334-339`.
  Consequence: `write_noise_off_config`, which sets only `T_k = 0` plus the quantum and pump
  flags (`analysis/run_detuning_sweep.py:404-406`), does **not** produce a deterministic run
  on any χ²/TFLN config with `r33 ≠ 0`. The docstring states this in prose
  (`run_detuning_sweep.py:373-374`) but nothing enforces it.

Related, same class of gap: with σ = 0 the TCCR AR(1) is still traced and still consumes a
PRNG subkey (`noise_models.py:435`, `:448`), and the pyro-EO product is still computed
(`noise_models.py:447`) — "off" for these channels is a zero *value*, never an elided op,
unlike the three flagged channels.

### G2 — Python-level (static, non-traced) branches that change the traced JAX graph on a noise flag

**Inside the traced solver body** (`_single_trajectory_solver`, `lle_solver.py:390-766`):

1. `lle_solver.py:541-544` — `if fsr_noise_sequence is not None:` builds `mu_grid`.
2. `lle_solver.py:572-576` — `if delta_d1 is not None:` adds `-1j*(mu_grid*delta_d1)*dt_sub`
   to `lin_exp`.
3. `lle_solver.py:604-607` — `if qnoise_enabled and (not qnoise_roundtrip or m == 0):` adds
   the Langevin increment. Three static predicates: `qnoise_enabled` and `qnoise_roundtrip`
   (`static_argnums` 27 and 28, `lle_solver.py:911`) and the Python loop index `m`
   (`lle_solver.py:670`).
4. `lle_solver.py:644-647` — `if fsr_noise_sequence is not None:` selects `delta_d1_t` vs
   `None`.
5. `lle_solver.py:655-663` — `if pump_scale_sequence is not None:` per-round-trip
   `sqrt`+gather kick vs the precomputed constant `pump_kick`.

(The same `None`-as-static-flag pattern is used by non-noise features at
`lle_solver.py:499-502`, `:519-540`, `:545-546`, `:708-709`; noted because it means a
`None`/array change silently forces a retrace.)

**Host-side branches that select which graph is traced or what is fed in:**

6. `lle_solver.py:848-858` — `if _noise_model.is_colored:` a Python loop with host numpy
   synthesis vs `jax.vmap` of the traced AR(1). The traced program depends on a **config
   string** (`trn_psd_model`).
7. `lle_solver.py:879-888` — the same branch in `_delta_t_sequences`.
8. `simulator/noise_models.py:216-219`, `:291-296`, `:436-445` — `is_colored` branches inside
   the samplers themselves (traced AR(1) vs host FFT synthesis).
9. `lle_solver.py:1364-1428` — `if qn_enabled:` … `else:` sets `qnoise_scale` and the labeler
   vacuum-floor parameters; the else branch (`:1424-1428`) pins them to inactive values.
   These flow into `_physical_state_labeler` (`lle_solver.py:1660-1668`), which is an
   `lru_cache`d **static** argument (`static_argnums` 13, `lle_solver.py:911`) — so the
   quantum flag changes the labeler object identity and forces a different traced labeler
   graph plus a full recompile.
10. `simulator/state_labeler.py:199-205` — `if ENV_SMOOTH_W > 1:` and
    `if VACUUM_FLOOR_LEVEL > 0.0:` are build-time Python branches inside the labeler closure,
    driven (via `make_threshold_params`, `state_labeler.py:94-95`) by the quantum flag.
11. `lle_solver.py:1589-1619` — `if qn_enabled and qn_seed_vacuum:` builds the vacuum
    cold-start field on the host and hands it in as a **warm** start, bypassing the traced
    cold-start seed at `lle_solver.py:722-739`; the traced `_is_cold` mask
    (`lle_solver.py:736`) then selects the warm branch. The initial-condition graph therefore
    differs on a noise flag.
12. `lle_solver.py:1498-1501` — `if np.any(pump_freq_noise_history):` — this one is
    **data-dependent, not flag-dependent**: whether the pump-frequency term is added at all
    is decided by the realized sample values. The same predicate governs the presence of the
    output key at `lle_solver.py:1721-1722`, so the result dict's key set depends on drawn
    data. (An all-zero deterministic override silently produces no
    `pump_freq_noise_history` key.)
13. `lle_solver.py:1518-1525` — decides `pump_scale_sequence = array` vs `None`, i.e. which
    of the two branches at `lle_solver.py:655-663` is traced; see footnote [b] on its
    predicate.
14. `lle_solver.py:1535-1565` — override/flag branch selecting `fsr_noise_sequences` as array
    vs `None`, i.e. which of branches 1/2/4 above are traced.

### G3 — dtype boundaries where a float32 array feeds the float64 solver

`jax_enable_x64` is forced at `lle_solver.py:26-27`, so the solver graph is float64 /
complex128. The classical detuning-noise chain is nevertheless generated in **float32**:

1. **`simulator/noise_models.py:139` and `:145`** — `_ar1_samples` draws
   `xi = jax.random.normal(..., dtype=jnp.float32)` and initialises the scan carry as
   `jnp.zeros((), dtype=jnp.float32)`. `alpha` (`:137`) and `sigma_step` (`:138`) are
   weakly-typed float64 scalars derived from Python floats, which do not promote a float32
   operand — and `jax.lax.scan` requires the carry dtype to be invariant, so the entire AR(1)
   recursion is float32. *(The non-promotion step is an inference from JAX's weak-typing
   rules; the carry-invariance requirement makes it the only self-consistent reading. Not
   executed — JAX is not installed here.)* This is the generator for **TRN, pyro-EO, TCCR and
   the legacy dT**.
2. **`simulator/noise_models.py:446-447`** — `c_pull * temp_noise` and
   `pyro_coeff * temp_noise`: Python float (weak f64) × f32 → **f32**. With
   `c_pull ≈ (ω₀/n₀)·dn_dT ≈ 1.6e10 rad/s/K`, the ~7-decimal-digit float32 mantissa sets the
   granularity of the detuning-noise amplitude.
3. **`simulator/noise_models.py:454`** — `combined = (trn - pyro + tccr).astype(jnp.float32)`
   — an **explicit downcast** of the summed sequence.
4. **`simulator/lle_solver.py:858`** — `jax.vmap(_gen_noise)(noise_keys).astype(jnp.float64)`
   — the f32 sequence is **upcast** into the f64 solver. The array carries only float32
   information from here on. It becomes `noise_sequence` (`lle_solver.py:640`),
   `delta_omega_eff` (`:566`) and the complex128 linear exponent (`:571`, `:577`).
5. **`simulator/noise_models.py:456-459`** — dT is upcast to float64 only
   `if jax.config.read("jax_enable_x64")`; the *content* is still the float32 AR(1). Paired
   with **`lle_solver.py:888`** `dts.astype(jnp.float64)` — this is the dT that scales the FSR
   channel at `lle_solver.py:1547-1549`, so δD₁(t) is float32-precision inside a float64
   array.
6. **`simulator/lle_solver.py:1500`** — `noise_sequences + jnp.asarray(pump_freq_noise_history,
   dtype=jnp.float64)` mixes a genuinely float64 host array with the float32-precision
   detuning array; the sum is float64-typed but inherits the coarser granularity of its TRN
   component.
7. **`simulator/state_labeler.py:149, 167, 213-214, 251-252`** —
   `entropy_max = jnp.log(jnp.array(n_tau, dtype=jnp.float32))`,
   `sign_changes = ...astype(jnp.float32)`, the two `mono_frac` reductions
   `...astype(jnp.float32)`, and `sentinel = jnp.float32(n_tau)` / the float32 `peak_locs`
   array. These are float32 reductions inside the labeler that is called on the complex128
   field at `lle_solver.py:690`. They affect the discrete label only, not the field.
8. **`simulator/state_labeler.py:516`** — `assert_labelers_consistent` casts the field to
   `jnp.complex64` before running the JAX labeler and compares against the NumPy labeler run
   in float64. Diagnostic helper, not the solver hot path, but it is a genuine
   complex64/float64 comparison boundary.

**Not** dtype boundaries (verified float64 end-to-end): the colored-noise engine and all PSD
factories (`simulator/colored_noise.py:78-124`, `:147-296`, explicit `dtype=np.float64`
throughout); both `PumpNoise` samplers (`simulator/noise_models.py:620-678`); and the quantum
Langevin increment (`lle_solver.py:383-387`, float64 draws → complex128).
