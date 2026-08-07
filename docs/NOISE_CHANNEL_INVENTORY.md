# Noise-channel inventory (read-only audit)

Audit date: 2026-08-07. Working tree: branch `claude/noise-channel-inventory-nom47l`,
`git rev-parse HEAD` = `13a84cd`, clean.

Scope (files read in full): `simulator/noise_models.py`, `simulator/colored_noise.py`,
`simulator/lle_solver.py`, `config/sin_params.yaml`,
`analysis/run_detuning_sweep.py::write_noise_off_config`, `simulator/state_labeler.py`.
Supporting reads used only to name switches/tests: `simulator/noise_config.py`,
`simulator/provenance.py`, `tests/`.

Reference: Herr, Tikan & Kippenberg, **arXiv:2604.05897v1** (7 Apr 2026); equation and
section numbers below are v1 numbers.

**No source file was modified.** Every line number is from the tree at the SHA above.
Runtime facts marked "(measured)" were obtained by importing the module and printing —
they change no file. Anything not established from a literal source token or a measured
value is marked **UNVERIFIED**.

Config defaults quoted are the values committed in `config/sin_params.yaml`, a **SiN**
device (`eo_r33_m_per_v = 0.0`, `pyroelectric_coeff_c_per_m2_k = 0.0`, `T_k = 300.0`).
"absent → *x*" means the key is **not present in the config file** and the code's own
`cfg.get(..., x)` default applies.

Measured at audit time from the committed config: `t_r = 4.065e-11 s`,
`tau_th = 5.0e-6 s`, `alpha = exp(-t_r/tau_th) = 0.99999187`,
`TotalNoise.trn.enabled = True`, `.pyroeo.enabled = True`, `.tccr.enabled = False`,
`sigma_tccr = 0.0`.

---

## Channel table

| # | Channel | 2. Generator class/function (file:line) | 3. Enters the EOM at (file:line, which term) | 4. Additive / multiplicative on E | 5. White or colored (τ_c or PSD shape) | 6. Config keys → current default | 7. How it is currently disabled | 8. "Off" proven bit-identical to legacy? (which test) | 9. Shares a random source with |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Quantum vacuum** (Sec. V.B.2, Eq. 126) | `_qnoise_increment` `lle_solver.py:577-603` (draw at `:599-602`); per-quadrature scale computed `lle_solver.py:1628-1630`; cold-start vacuum seed `_vac_draw` `lle_solver.py:1872-1879` | `lle_solver.py:820-823` — `e_next = e_next + _qnoise_increment(...)`: the **additive Langevin drive √κ·ξ_μ(t)**, injected after the Strang sub-step loop and before the absorber/validity masks. Second entry point: the initial condition, `lle_solver.py:1880` `e0_init = e_cw_traj[:,None] + vac` | **Additive** — a complex increment added directly to `E` per fast-time sample | **White.** i.i.d. complex Gaussian per fast-time sample per injection event ⇒ flat PSD. Per-quadrature std √(ħω₀·κ·n_tau·dt/4) (`lle_solver.py:1628-1630`), dt = `dt_fine` or `t_r` per the cadence enum (`lle_solver.py:1627`) | `quantum_noise_enabled` → **0** (`sin_params.yaml:168`)<br>`noise.quantum_vacuum` → **false** (`:44`)<br>`quantum_noise_seed_vacuum_init` → **1** (`:173`) / `noise.-` default `True` (`noise_config.py:201`)<br>`quantum_noise_injection_cadence` → **0** = fine (`:181`)<br>`hbar_omega0_j` → **0.0** = auto (`:187`)<br>`labeler_vacuum_floor_margin` → **10.0** (`:198`)<br>`labeler_envelope_smooth_modes` → **8** (`:203`)<br>kwargs `quantum_noise_enabled` / `_seed_vacuum_init` / `_injection_cadence` (`lle_solver.py:1167-1169`) | **Explicit boolean flag.** `NoiseConfig.quantum_vacuum` (`noise_config.py:187`), resolved once at `lle_solver.py:1603-1614`, validated 0/1 by `_as_flag` (`:1579-1583`), then passed as a **static** Python bool (`static_argnums` includes 27, `lle_solver.py:1141`). The branch at `lle_solver.py:820` therefore traces **zero extra ops**. Off also pins `qnoise_scale = 0.0` and the inactive labeler params (`lle_solver.py:1680-1684`). `write_noise_off_config` forces it off both ways (`run_detuning_sweep.py:416`, `:422`) | **Yes, within the current tree.** `tests/test_noise_off_identity.py::test_all_off_bit_identical_to_golden` (raw-byte under `SOLITON_STRICT_ULP=1`) and `::test_all_off_equals_every_channel_individually_off` (byte-exact, iterates every `SWITCH_FIELDS` entry incl. `quantum_vacuum`). Channel-local: `tests/test_quantum_noise.py::test_config_missing_keys_default_off` (`np.array_equal` over {keys absent, key=0, kwarg False}) and `::test_disabled_path_adds_no_rng_to_scan_body` (structural: zero RNG primitives in the traced scan body).<br>**Caveat:** no in-repo comparison against an *archived pre-quantum* solver — `tests/test_quantum_noise.py::test_flag_off_legacy_seed_statistics` is statistical (5 % tol), not bit-identity | **Nothing.** Its own chain: `key_qnoise` from `_legacy_rng_chain` (`lle_solver.py:1038-1039`), split into `key_qnoise_inj` / `key_qnoise_seed` at `lle_solver.py:1689`, per-trajectory at `:1690`. Isolation pinned by `tests/test_dataset_generator.py::test_key_isolation` |
| 2 | **TRN** (thermorefractive, Eqs. 129-130) | Solver-consumed sequence: `TotalNoise.sample_with_delta_t` `noise_models.py:538-584`, specifically `trn_noise = self.c_pull * temp_noise` at **`noise_models.py:570`**. Underlying dT(t): `_ar1_samples` `noise_models.py:180-191` (`single_pole`) or `colored_noise.synthesize_from_psd` `colored_noise.py:78-124` (colored), selected at `noise_models.py:553-568`. Class `TRNoise` `noise_models.py:194-295`; pull `c_pull` `:232-234`. Host entry `_detuning_noise_sequences` `lle_solver.py:1045-1081` | `lle_solver.py:856` `freq_noise = noise_sequence[step_idx]` → `lle_solver.py:782` `delta_omega_eff = dw_step + thermal_shift + freq_noise` → `lle_solver.py:787` `lin_exp = (-kappa/2 - 1j*disp - 1j*delta_omega_eff)*dt_sub`. **Term: the detuning term −i·δω·E of the linear half-step** | **Multiplicative on E** — it is *additive in the detuning* but reaches the field as the phase factor `exp(-1j·δω_noise·dt_sub)` (`lle_solver.py:787`, `:793`) | **Colored.** `single_pole` (default): Lorentzian with **correlation time τ_th = 5.0e-6 s** (≈ 1.23e5 round trips at the committed `fsr_hz`); `S_δω(f) = C_pull²·4·var_dT·τ_th/(1+(2πfτ_th)²)` (`noise_models.py:286-292`; factory `colored_noise.py:147-164`).<br>`kondratiev_gorodetsky`: Eq. 130 shape ∝ ω^(−1/2)·[1+(ω·τ_d)^(3/4)]^(−2), renormalized to the Eq. 129 variance (`colored_noise.py:167-257`).<br>`csv`: tabulated, log-log interpolated, flat-clamped outside the span (`colored_noise.py:260-296`) | `noise.trn` → **true** (`sin_params.yaml:45`)<br>`T_k` → **300.0** (`:119`)<br>`tau_th_s` → **5.0e-6** (`:120`)<br>`dn_dT_per_k` → **2.45e-5** (`:115`)<br>`rho_kg_per_m3` → **3.17e+3** (`:116`)<br>`Cp_j_per_kg_k` → **700.0** (`:117`)<br>`mode_volume_m3` → **1.954e-14** (`:103`)<br>`n0` → **1.87** (`:97`)<br>`pump_wavelength_m` → **1.55e-6** (`:96`)<br>`fsr_hz` → **2.46e+10** (`:66`, sets `t_r` and `f_s`)<br>`trn_psd_model` → **single_pole** (`:52` and `:252`)<br>`trn_ar1_stationary_init` → **false** (`:53`) — *declared but not consumed, see GAP G4*<br>`trn_R_m`/`trn_da_m`/`trn_db_m` → **null** (`:258-260`)<br>`trn_psd_csv_path` → **null** (`:263`)<br>`trn_csv_units` → **S_delta_T** (`:268`)<br>`alpha_L_per_k` → **0.0** (`:273`)<br>`kappa_th_w_per_m_k` → **30.0** (`:118`, K-G model only)<br>`noise_dtype` → **float32** (`:55`) — *declared but not consumed, see G4*<br>`legacy_segment_noise` → **true**/**1** (`:56`, `:293`; dataset-generator cadence) | **Explicit boolean flag (since commit `17362ae`), on top of the historical implicit gate.**<br>• Explicit: `NoiseConfig.trn` (`noise_config.py:188`) → `TotalNoise.__init__` `noise_models.py:501` → `TRNoise(cfg, enabled=...)` → `_resolve_enabled` `noise_models.py:36-55`, stored `noise_models.py:248`. Disabled ⇒ exact zeros with **no RNG draw** (`noise_models.py:275-276`, `:553-558`, `_zeros` `:58-70`).<br>• Implicit (still live when `enabled=None`): `T_k = 0` ⇒ `var_delta_t = k_B·T_k²/(ρ·Cp·V) = 0` (`noise_models.py:227`) plus the explicit zero-PSD short circuit for **every** PSD model (`noise_models.py:110-113`). `write_noise_off_config` writes both representations (`run_detuning_sweep.py:416`, `:421`).<br>• Note the disabled path still **executes the multiply** `0.0 * temp_noise` at `noise_models.py:570` (the coefficient, not the op, is elided) | **Yes, both directions.**<br>Off ≡ legacy: `tests/test_noise_off_identity.py::test_all_off_bit_identical_to_golden` and `::test_all_off_equals_every_channel_individually_off` (byte-exact, `trn` is in `SWITCH_FIELDS`). Sampler level: `tests/test_colored_noise.py::test_trn_disabled_returns_exact_zeros` (exact `== 0.0` for `single_pole` and `kondratiev_gorodetsky`; the `csv` model is **not** covered there, though `noise_models.py:110-113` short-circuits it too).<br>`enabled=None` ≡ pre-change implicit gating: `tests/test_colored_noise.py::test_enabled_none_reproduces_legacy_bitwise` (parametrized `T_k ∈ {300, 0}`, `np.array_equal` against an inline reimplementation of the old AR(1)).<br>ON default ≡ pre-colored-noise stream: `tests/test_colored_noise.py::test_single_pole_bit_identical_to_legacy_ar1` (`np.array_equal`, including the `_detuning_noise_sequences` solver surface).<br>T_k = 0 collapse: `tests/test_noise_metrology.py::test_tk_zero_collapses_all_delta_t_channels` | **Pyro-EO, thermal-expansion pull and FSR — CONFIRMED (see the hypothesis note below).** One `temp_noise` array is produced at `noise_models.py:553-568` and consumed twice, at `:570` (TRN) and `:571` (pyro-EO). FSR re-derives the *same* array: `lle_solver.py:1807` calls `_delta_t_sequences(noise_keys, ...)` with the **same** `noise_keys` created at `lle_solver.py:1573` and consumed by `_detuning_noise_sequences` at `:1694`; `_delta_t_sequences` returns `sample_with_delta_t(k, N)[1]` (`lle_solver.py:1111` colored / `:1116` legacy), i.e. the identical `temp_noise` regenerated from the identical `key_thermal` (`noise_models.py:552`). **Independent of** TCCR (`key_tccr`, `noise_models.py:552`/`:572`), quantum, and both pump channels |
| 3 | **Thermal-expansion pull** (the paper's "dimensional fluctuation" companion of TRN) | **Not a separate stochastic process** — a modification of the TRN pull coefficient: `noise_models.py:231-234` `c_pull = (omega_0/n0)*(dn_dT + n0*alpha_L)`. No independent generator, no independent RNG draw | Everywhere `c_pull` is used: TRN detuning `noise_models.py:570` → `lle_solver.py:856` → `:782` → `:787`; FSR amplitude `lle_solver.py:1811`. **Not** in the pyro-EO pull — `pyro_coeff` (`noise_models.py:333-336`) carries no α_L term; `PyroEONoise` recomputes a local `_c_pull` (`noise_models.py:342-344`) used **only** to convert an `S_delta_omega` CSV into K²/Hz (`noise_models.py:345-348` → `:153-161`) | **Multiplicative on E** — same detuning phase as TRN; it is a scalar rescale of the TRN amplitude | **Same color as TRN** — literally the same dT(t) realization; only the pull constant changes | `alpha_L_per_k` → **0.0** (`sin_params.yaml:273`); inherits every TRN key of row 2 | **NO boolean flag — value-gated only.** `alpha_L_per_k = 0.0` makes `dn_dT + n0*0.0 == dn_dT` exactly in IEEE arithmetic (`noise_models.py:231-234`, and `:342-344` for the CSV-conversion copy). Also killed by `T_k = 0` and by `noise.trn = false` (which zeroes `temp_noise` itself). There is **no** `NoiseConfig` field for it (`noise_config.py:187-204`) | **Yes.** `tests/test_colored_noise.py::test_alpha_l_zero_is_bitwise_neutral_and_scales_pull` asserts `TRNoise(cfg_alpha0).c_pull == trn0.c_pull` with `==` (exact), and that a non-zero α_L scales the pull as documented. It is additionally covered, as part of the whole trajectory, by `tests/test_noise_off_identity.py::test_all_off_bit_identical_to_golden` | **TRN, pyro-EO, FSR** — by construction it *is* the TRN dT realization (`noise_models.py:553-571`) |
| 4 | **Pyro-EO** (pyroelectric → Pockels detuning) | Solver-consumed sequence: `pyroeo_noise = self.pyro_coeff * temp_noise` at **`noise_models.py:571`**, combined with a **minus** sign at `noise_models.py:578`. Coefficient `noise_models.py:333-336`. Class `PyroEONoise` `noise_models.py:298-384`; its own `sample` (`:365-372`) uses an independently keyed AR(1) and is **standalone/diagnostic only — not the solver path** | Identical to TRN: `lle_solver.py:856` → `:782` → `:787`. **Term: −i·δω·E** | **Multiplicative on E** (detuning phase) | **Colored — the same dT spectrum as TRN** (same sequence): default Lorentzian, τ_th = 5.0e-6 s. Closed form `S = pyro_coeff²·S_dT` (`noise_models.py:374-384`) | `noise.pyro_eo` → **true** (`sin_params.yaml:46`)<br>`eo_r33_m_per_v` → **0.0** (`:296`)<br>`pyroelectric_coeff_c_per_m2_k` → **0.0** (`:297`)<br>`eps_r_z` → absent → **28.0** (`noise_models.py:316`)<br>`t_ln_m` → absent → **4.0e-7** (`noise_models.py:320`)<br>`t_clad_top_m` → absent → **1.0e-6** (`:321`)<br>`t_clad_bot_m` → absent → **2.0e-6** (`:322`)<br>`eps_r_clad_top` → absent → **1.0** (`:323`)<br>`eps_r_clad_bot` → absent → **3.9** (`:324`)<br>plus every TRN thermal key of row 2 and `trn_psd_model` (`noise_models.py:340`)<br>*(footnote [a]: the config's stack-geometry keys are commented out under names the code never reads)* | **Explicit boolean flag, plus two independent value gates.**<br>• Explicit: `NoiseConfig.pyro_eo` (`noise_config.py:189`) → `noise_models.py:502` → `PyroEONoise(cfg, enabled=...)` → `_resolve_enabled`, stored `noise_models.py:354`. Disabled ⇒ the coefficient in `TotalNoise` becomes the literal `0.0` (`noise_models.py:571`).<br>• Value gate 1: `eo_r33_m_per_v = 0` **or** `pyroelectric_coeff_c_per_m2_k = 0` ⇒ `pyro_coeff = 0` (`noise_models.py:333-336`). This is the committed SiN state and is deliberately **not** folded into the switch (`noise_models.py:349-353`).<br>• Value gate 2: `T_k = 0` (`noise_models.py:110-113`, `:314`).<br>• As with TRN the multiply at `noise_models.py:571` and the subtraction at `:578` still execute — "off" is a zero *value*, not an elided op | **Yes for the switch; weak for the value gate.**<br>Switch off ≡ legacy: `tests/test_noise_off_identity.py::test_all_off_equals_every_channel_individually_off` (byte-exact; `pyro_eo ∈ SWITCH_FIELDS`) and `::test_all_off_bit_identical_to_golden`. Sampler level: `tests/test_colored_noise.py::test_trn_disabled_returns_exact_zeros` (its second half runs `PyroEONoise(cfg2, enabled=False)` on a χ² config and asserts exact zeros).<br>`enabled=None` ≡ legacy implicit rule: `tests/test_colored_noise.py::test_enabled_none_reproduces_legacy_bitwise`.<br>**UNVERIFIED:** that `r33 = 0 ⇒ the pyro contribution is exactly zero` is asserted anywhere. The nearest checks are `tests/test_colored_noise.py::test_sample_with_delta_t_consistency` (`np.allclose`, `rtol=2e-6` — not bit-exact) and `::test_single_pole_bit_identical_to_legacy_ar1` (which pins the *combined* stream, in which the pyro term happens to be zero) | **TRN, thermal-expansion pull, FSR — CONFIRMED**; the same `temp_noise` object (`noise_models.py:553-571`). Independent of TCCR |
| 5 | **TCCR** (thermal carrier / surface-state) | `TCCRNoise` `noise_models.py:387-461`; sampled at `noise_models.py:572` `tccr_noise = self.tccr.sample(key_tccr, N)` → `_ar1_samples` `noise_models.py:456`. Segment-continuity variant: `single_pole_psd` + `synthesize_from_psd` at `noise_models.py:611-616` | Summed into `combined` at `noise_models.py:578`, then identical to TRN: `lle_solver.py:856` → `:782` → `:787`. **Term: −i·δω·E** | **Multiplicative on E** (detuning phase) | **Colored.** Single-pole Lorentzian with **correlation time `tau_carrier`, default 1.0e-7 s** (`noise_models.py:391`; the key is *absent* from the config). `S(f) = s0/(1+(2πf·τ_c)²)` (`noise_models.py:458-461`), `s0 = (dω/dN_s)²·N_s,eq·2τ_c` (`noise_models.py:416`) | `noise.tccr` → **false** (`sin_params.yaml:47`)<br>`tau_carrier_s` → absent → **1.0e-7** (`noise_models.py:391`)<br>`surface_state_density_per_m2` → absent → **1.0e16** (`noise_models.py:396`)<br>`eo_r33_m_per_v` → **0.0** (`:296`)<br>`eps_r_z` → absent → **28.0** (`noise_models.py:400`)<br>`effective_mode_area_m2` → **3.344e-12** (`:83`)<br>`n0` → **1.87** (`:97`)<br>`t_ln_m` → absent → **4.0e-7** (`noise_models.py:402`; **read but unused** in the formula — see the comment at `noise_models.py:411`)<br>`intrinsic_q` → **4.0e+7** (`:67`; warning threshold only, `noise_models.py:431-433`)<br>`pump_wavelength_m`, `fsr_hz`<br>`T_k` → **300.0** (`:119`) is stored at `noise_models.py:393` but **never enters** `s0_tccr`/`var_tccr` (`noise_models.py:412-418`) | **Explicit boolean flag, plus a value gate. NOT gated by `T_k`.**<br>• Explicit: `NoiseConfig.tccr` (`noise_config.py:190`) → `noise_models.py:503` → `TCCRNoise(cfg, enabled=...)`, stored `noise_models.py:449-451`. Disabled ⇒ exact zeros with **no RNG draw** (`noise_models.py:454-455`).<br>• Value gate: `eo_r33_m_per_v = 0.0` ⇒ `dw_dNs = 0` (`noise_models.py:413`) ⇒ `s0_tccr = 0` (`:416`) ⇒ `sigma_tccr = 0` (`:418`).<br>• **`T_k` does not gate it** — the variance carries no `T_k` factor (`noise_models.py:412-418`), which is exactly why the historical "`T_k = 0` means noise off" convention never silenced TCCR on χ² platforms. The *modern* `write_noise_off_config` does silence it, but only via the `noise:` block it now also writes (`run_detuning_sweep.py:416`); its legacy `physical_parameters` half (`:421-423`) still does not.<br>• When `enabled=True` but `sigma_tccr = 0` the AR(1) scan is still traced and a PRNG subkey is still consumed (`noise_models.py:552`, `:572`) | **Yes for the switch; weak for the value gate.**<br>Switch off ≡ legacy: `tests/test_noise_off_identity.py::test_all_off_equals_every_channel_individually_off` and `::test_all_off_bit_identical_to_golden`. Sampler level: `tests/test_colored_noise.py::test_trn_disabled_returns_exact_zeros` (χ² config, `TCCRNoise(cfg2, enabled=False)` ⇒ exact zeros). Stream isolation: `tests/test_colored_noise.py::test_disabling_trn_does_not_shift_tccr_stream` (`np.array_equal`).<br>`enabled=None` ≡ legacy rule: `tests/test_colored_noise.py::test_enabled_none_reproduces_legacy_bitwise` (asserts `tn.tccr.enabled is False` for SiN).<br>**UNVERIFIED:** any test proving `r33 = 0 ⇒ TCCR exactly 0` **as a value**, and any test proving TCCR survives `T_k = 0` on a χ² config (the behaviour that made the old sidecar non-deterministic there) | **Nothing.** Independent subkey `key_tccr` from the unconditional `jax.random.split(key, 2)` at `noise_models.py:552`, consumed at `:572` (and `:601`/`:611-616` on the segment-continuity path). It shares only the *parent* per-trajectory key `noise_keys[i]` with the thermal channels; the two branches are independent streams |
| 6 | **Pump frequency noise** (Sec. V.B.4) | `PumpNoise.sample_freq` `noise_models.py:752-770` — white part `:763-764`, 1/f flicker via FFT synthesis `:765-769` → `_synthesize_from_onesided_psd` `noise_models.py:630-639` → `colored_noise.synthesize_from_psd` `colored_noise.py:78-124` (`clamp_dc=False`). Class `PumpNoise` `noise_models.py:642-810`. Host call site `lle_solver.py:1752-1756` | **Summed into the detuning array on the HOST** at `lle_solver.py:1761-1764` (`noise_sequences = noise_sequences + pump_freq_noise_history`), then `lle_solver.py:856` → `:782` → `:787`. **Term: −i·δω·E**, the same linear-operator detuning term as TRN. Sign `−2π·δν_p` applied at `lle_solver.py:1751` (override) / `:1755` (stochastic), because δω ≡ ω_res − ω_p | **Multiplicative on E** (detuning phase) | **Colored.** `S_δν(f) = h₀ + h₋₁/f` [Hz²/Hz] (`noise_models.py:734-737`): a white plateau h₀ (Δν_L = π·h₀, `noise_models.py:731`) plus 1/f flicker whose DC bin is clamped to `f₁ = f_s/N` (`noise_models.py:766-769`) | `noise.pump_freq_noise` → **false** (`sin_params.yaml:48`)<br>`pump_noise_enabled` → **0** (`:218`)<br>`pump_freq_noise_h0_hz2_per_hz` → **0.0** (`:224`)<br>`pump_freq_noise_hm1_hz3_per_hz` → **0.0** (`:226`)<br>`fsr_hz` (sets f_s, `noise_models.py:684-685`)<br>kwargs `pump_noise_enabled`, `pump_freq_noise_override` (`lle_solver.py:1170-1171`) | **Explicit boolean flag (three layers, all inert here).** `NoiseConfig.pump_freq_noise` (`noise_config.py:191`) resolved at `lle_solver.py:1727`; the `PumpNoise` object is built live only if either pump channel is on (`lle_solver.py:1722-1725`), and its own `enabled` is validated 0/1 at `noise_models.py:686-692`. Off ⇒ `_on = 0.0` zeroes `_h0`/`_hm1` (`noise_models.py:724-726`) and `sample_freq` early-returns exact zeros (`noise_models.py:759-760`). The host guard at `lle_solver.py:1752` additionally requires `_h0 > 0 or _hm1 > 0`, and the host **add** is skipped entirely unless `np.any(...)` (`lle_solver.py:1761`). Doubly inert on this config because h₀ = h₋₁ = 0. `write_noise_off_config` forces it off both ways (`run_detuning_sweep.py:416`, `:423`) | **Yes, within the current tree.** `tests/test_pump_noise.py::test_flag_off_bit_identical_to_legacy` (`np.array_equal` on `E_snapshots` and `U_int_history`, plus absence of the diagnostic keys). Supporting: `::test_disabled_forces_all_channels_inert`, `::test_disabled_solver_ignores_pump_numbers`. Whole-trajectory: `tests/test_noise_off_identity.py::test_all_off_equals_every_channel_individually_off` | **Nothing.** Own subkey `key_pump_freq` from `jax.random.split(key_pump, 2)` at `lle_solver.py:1717`, split per trajectory at `:1753`. Sibling (independent) of `key_pump_rin`. The `key_pump` chain is *appended* to the legacy ladder (`lle_solver.py:1714-1716` reconstructs `_legacy_rng_chain` exactly, then splits once more), so it perturbs no legacy stream |
| 7 | **Pump RIN** (Sec. V.B.5) | `PumpNoise.sample_rin` `noise_models.py:772-810` — white floor `:782-784`, 1/f excess below f_c `:786-798`, clip at ε ≥ −1 `:810`. Host call `lle_solver.py:1771-1775`; the scale array `1+ε` is built at `lle_solver.py:1781-1788` | `lle_solver.py:872` `scale_t = pump_scale_sequence[step_idx]` → `:873` `pin_t = pin*scale_t` → `:874-876` `kick_t = sqrt(max(kappa_c*pin_t,0))*dt_sub` → **`lle_solver.py:798` `e_pumped = (e_sub + kick)`** — the **pump/drive term**. Second, indirect path: `pin_t` sets `p_trans` (`:896`) and the drive changes ⟨|E|²⟩ → `p_abs` (`:835`) → ΔT (`:840-844`) → `thermal_shift` (`:781`) → detuning | **Neither, directly.** It is **multiplicative on the drive amplitude F** (as the √ of the power scale), and F enters E **additively** (`lle_solver.py:798`). Secondarily it reaches E multiplicatively through the thermal → detuning pathway | **Colored.** `S_ε(f) = 10^(floor/10) + 10^(excess/10)·(f_c/f)` for f < f_c, floor-only above f_c (`noise_models.py:739-749`): white floor + 1/f excess with a corner frequency | `noise.pump_rin` → **false** (`sin_params.yaml:49`)<br>`pump_noise_enabled` → **0** (`:218`)<br>`pump_rin_floor_dbc_per_hz` → **−300.0** (`:231`)<br>`pump_rin_excess_dbc_per_hz` → **−300.0** (`:232`)<br>`pump_rin_corner_hz` → **1.0e+4** (`:234`)<br>`fsr_hz` (sets f_s)<br>kwarg `pump_rin_epsilon_override` (`lle_solver.py:1172`) | **Explicit boolean flag.** `NoiseConfig.pump_rin` (`noise_config.py:192`) resolved at `lle_solver.py:1728`. Off ⇒ `_on = 0.0` zeroes `_rin_floor_lin`/`_rin_excess_lin` (`noise_models.py:727-728`) and `sample_rin` early-returns zeros (`noise_models.py:778-779`); the host branch at `lle_solver.py:1781-1788` then sets `pump_scale_sequence = None`, so the Python branch at `lle_solver.py:871-879` traces the **legacy constant-kick path with zero extra ops**.<br>*(footnote [b]: the enable predicate at `lle_solver.py:1782` keys on the **floor** only, not the excess)* | **Yes, within the current tree.** `tests/test_pump_noise.py::test_flag_off_bit_identical_to_legacy` (`np.array_equal`); supporting `::test_disabled_forces_all_channels_inert`, `::test_disabled_solver_ignores_pump_numbers`. Whole-trajectory: `tests/test_noise_off_identity.py::test_all_off_equals_every_channel_individually_off` | **Nothing.** Own subkey `key_pump_rin` (`lle_solver.py:1717`), split per trajectory at `:1772`. Independent of `key_pump_freq` |
| 8 | **FSR / repetition-rate noise** (TRN-driven f_rep) | Built in the solver, not in `noise_models`: `lle_solver.py:1801-1812`, `fsr_noise_sequences = (D1/omega0)*_trn.c_pull*_dt_seqs`, with dT from `_delta_t_sequences` `lle_solver.py:1084-1118` and `c_pull` from a fresh `TRNoise(physical)` (`lle_solver.py:1804`, coefficient `noise_models.py:232-234`). The ultimate generator is therefore the TRN dT source: `_ar1_samples` `noise_models.py:180-191` or `colored_noise.py:78-124` | `lle_solver.py:861` `delta_d1_t = fsr_noise_sequence[step_idx]` → `lle_solver.py:792` `lin_exp = lin_exp - 1j*(mu_grid*delta_d1)*dt_sub`, with `mu_grid` built at `lle_solver.py:760`. **Term: the per-mode linear detuning μ·δD₁(t) inside the linear operator** — a mode-number-proportional detuning, distinct from the uniform δω of rows 2-7 | **Multiplicative on E** (mode-dependent phase in the Fourier domain) | **Colored — identical spectrum to TRN**, since it is the same dT(t), scaled by `(D₁/ω₀)·C_pull ≈ 1.27e-4·C_pull` (`lle_solver.py:1811`; magnitude logged `:1819-1825`). Default correlation time τ_th = 5.0e-6 s | `noise.fsr` → **false** (`sin_params.yaml:50`)<br>`fsr_noise_enabled` → **0** (`:282`)<br>`fsr_hz` → **2.46e+10** (`:66`, sets D₁ = 2π·FSR)<br>`pump_wavelength_m` → **1.55e-6** (`:96`, sets ω₀)<br>plus **every** TRN key of row 2 (`T_k`, `tau_th_s`, `dn_dT_per_k`, `alpha_L_per_k`, `n0`, `rho_kg_per_m3`, `Cp_j_per_kg_k`, `mode_volume_m3`, `trn_psd_model`, the K-G geometry keys, the CSV keys)<br>kwarg `fsr_delta_d1_override` (`lle_solver.py:1174`) | **Explicit boolean flag.** `NoiseConfig.fsr` (`noise_config.py:193`) resolved at `lle_solver.py:1795`; off ⇒ `fsr_noise_sequences = None` (`lle_solver.py:1827`) ⇒ the three Python `is not None` branches at `lle_solver.py:757`, `:788`, `:860` trace **zero extra ops** (no `mu_grid`, no fused multiply-add, no gather). **Also** implicitly killed by `T_k = 0`, by `noise.trn = false` (dT is zero — `TotalNoise.__init__` warns about exactly this at `noise_models.py:504-509`), and by `alpha_L`/`dn_dT` both zero | **Partly.**<br>`tests/test_noise_metrology.py::test_fsr_tk_zero_channel_identically_zero` — `np.array_equal` of `U_int_history` and `E_snapshots` between the channel-off run and `fsr_noise_enabled=True` at T_k = 0, i.e. it proves the **enabled-but-zero** path is bit-identical to off. Source collapse: `::test_tk_zero_collapses_all_delta_t_channels`. Whole-trajectory off: `tests/test_noise_off_identity.py::test_all_off_equals_every_channel_individually_off` (`fsr ∈ SWITCH_FIELDS`, byte-exact) and `::test_all_off_bit_identical_to_golden`.<br>**UNVERIFIED:** no test asserts flag-off ≡ *pre-feature* legacy for a `T_k ≠ 0` config; that rests on the `None`-branch construction, not on an assertion | **TRN, pyro-EO, thermal-expansion pull — CONFIRMED.** `lle_solver.py:1807` passes the **same** `noise_keys` (created `:1573`) that `_detuning_noise_sequences` consumes at `:1694`; `_delta_t_sequences` returns `TotalNoise.sample_with_delta_t(k, N)[1]` (`lle_solver.py:1111` colored / `:1116` legacy), which is the identical `temp_noise` drawn at `noise_models.py:553-568` from the identical `key_thermal` (`noise_models.py:552`). The dT is **regenerated, not threaded** — bit-consistency depends on that regeneration staying key-for-key identical (documented at `lle_solver.py:1088-1098`). Independent of TCCR, quantum and both pump channels |

### Answer to the stated hypothesis (column 9)

> *"I believe TRN, pyro-EO and FSR all consume the same `delta_T` realization."*

**CONFIRMED**, with line numbers:

* TRN and pyro-EO share it by **direct reuse of one array**: `temp_noise` is produced once at
  `simulator/noise_models.py:553-568` and multiplied at `:570` (`trn_noise`) and `:571`
  (`pyroeo_noise`), then combined with opposite signs at `:578`.
* FSR shares it by **deterministic regeneration from the same keys**:
  `simulator/lle_solver.py:1694` (`_detuning_noise_sequences(noise_keys, ...)`) and
  `simulator/lle_solver.py:1807` (`_delta_t_sequences(noise_keys, ...)`) receive the *same*
  `noise_keys` object created at `simulator/lle_solver.py:1573`; both funnel into
  `TotalNoise.sample_with_delta_t`, whose thermal subkey comes from the unconditional split at
  `simulator/noise_models.py:552`. `_delta_t_sequences` returns element `[1]` of that call
  (`lle_solver.py:1111` / `:1116`) — literally the same `temp_noise`.
* A **fourth** consumer of the same realization is the thermal-expansion pull, which is a
  rescale of the TRN pull coefficient rather than a channel of its own
  (`simulator/noise_models.py:231-234`).

**REFUTED for TCCR**: it takes the *second* subkey of `jax.random.split(key, 2)` at
`simulator/noise_models.py:552` and is drawn independently at `:572`, so it shares only the
parent per-trajectory key. Pinned by
`tests/test_colored_noise.py::test_disabling_trn_does_not_shift_tccr_stream`.

### Table footnotes

* **[a]** `config/sin_params.yaml:299-305` comments out stack-geometry keys under names the
  code never reads (`t_SiN_m`), while `PyroEONoise` reads `t_ln_m` (`noise_models.py:320`);
  the commented `eps_r_clad_top: 3.9` (`:304`) also disagrees with the code default `1.0`
  (`noise_models.py:323`). Numerically inert on this config only because `r33 = 0`.
* **[b]** `lle_solver.py:1781-1783`: the predicate deciding whether `pump_scale_sequence` is
  built is `pump_rin_epsilon_override is not None or (pump_rin_on and _pump._rin_floor_lin >
  0.0)` — it inspects `_rin_floor_lin` but **not** `_rin_excess_lin`. With the default floor
  of −300 dBc/Hz the linear floor is 1e-30 > 0, so in practice `pump_rin_on` alone decides; an
  excess-only RIN config would be silently dropped only if the floor were exactly 0
  (i.e. −inf dBc/Hz). Latent, not currently reachable.
* **[c]** The FSR path constructs `_TRNoise(physical)` at `lle_solver.py:1804` **without**
  passing the resolved `NoiseConfig`, so that object's `enabled` falls back to the legacy
  `T_k > 0` rule (`noise_models.py:248`). Harmless today because only `.c_pull` is read
  (`lle_solver.py:1811`) and `c_pull` does not depend on `enabled` — but it is a live
  inconsistency if `TRNoise.enabled` ever starts gating a coefficient.

---

## GAPS

### G1 — Channels with NO explicit boolean enable flag

Since commit `17362ae` ("simulator: explicit per-channel enables for TRN, pyro-EO and TCCR")
plus the `NoiseConfig` wiring in `2ff2881`, **seven of the eight** channels have a first-class
boolean in `simulator/noise_config.py:187-193` (`quantum_vacuum`, `trn`, `pyro_eo`, `tccr`,
`pump_freq_noise`, `pump_rin`, `fsr`), resolved in one place by
`simulator/lle_solver.py:152-205` (`_resolve_noise_flags`).

**One channel still has no boolean at all:**

* **Thermal-expansion pull** — `simulator/noise_models.py:231-234`
  (`c_pull = (omega_0/n0)*(dn_dT + n0*alpha_L)`), α_L read at `simulator/noise_models.py:231`
  and again at `simulator/noise_models.py:342` (the pyro-EO CSV-conversion copy). The only way
  to silence it is the material value `alpha_L_per_k = 0.0`
  (`config/sin_params.yaml:273`). There is **no** field for it in
  `simulator/noise_config.py:187-204`, so it appears in no `NoiseConfig`, in no
  `SWITCH_FIELDS` enumeration (`noise_config.py:81`), and in no ablation hash
  (`noise_config.py:378-386`). `NoiseConfig.all_off()` (`noise_config.py:240-256`) does
  **not** turn it off. `simulator/noise_config.py:19` already documents this as a known gap.

**Related residual gaps of the same class** (a switch exists, but "off" is still a zero
*value* rather than an elided operation, and the material gates remain unswitched):

* `simulator/noise_models.py:570-571` — with `trn`/`pyro_eo` disabled the coefficients become
  the Python literal `0.0` but the two multiplies and the subtraction at `:578` are still
  traced and executed.
* `simulator/noise_models.py:333-336` — pyro-EO is *additionally* value-gated by
  `eo_r33_m_per_v` (`config/sin_params.yaml:296`) and
  `pyroelectric_coeff_c_per_m2_k` (`:297`); those two have no boolean and are deliberately
  excluded from the switch (`simulator/noise_models.py:349-353`).
* `simulator/noise_models.py:412-418` — TCCR is *additionally* value-gated by
  `eo_r33_m_per_v`; and with `enabled=True, sigma_tccr=0` the AR(1) scan is still traced and
  a PRNG subkey is still consumed (`simulator/noise_models.py:552`, `:572`).
* `analysis/run_detuning_sweep.py:421-423` — the legacy half of `write_noise_off_config`
  still encodes "noise off" as `T_k = 0.0` plus two flags. `T_k` does not gate TCCR
  (`simulator/noise_models.py:412-418`), so that half alone is **not** a deterministic
  configuration on a χ² device. The function now also writes the authoritative `noise:` block
  (`analysis/run_detuning_sweep.py:416`), which does silence everything, and it emits a
  `DeprecationWarning` (`:403-411`).

### G2 — Python-level (static, non-traced) branches that change the traced JAX graph on a noise flag

**Inside the traced solver body** (`_single_trajectory_solver`,
`simulator/lle_solver.py:606-982`):

1. `lle_solver.py:757-760` — `if fsr_noise_sequence is not None:` builds `mu_grid`.
2. `lle_solver.py:788-792` — `if delta_d1 is not None:` adds
   `-1j*(mu_grid*delta_d1)*dt_sub` to `lin_exp`.
3. `lle_solver.py:820-823` — `if qnoise_enabled and (not qnoise_roundtrip or m == 0):` adds
   the quantum Langevin increment. Three static predicates: `qnoise_enabled` and
   `qnoise_roundtrip` (`static_argnums` 27 and 28, `lle_solver.py:1141`) and the Python loop
   index `m` from the unrolled fine-step loop (`lle_solver.py:886`).
4. `lle_solver.py:860-863` — `if fsr_noise_sequence is not None:` selects `delta_d1_t` vs
   `None`.
5. `lle_solver.py:871-879` — `if pump_scale_sequence is not None:` selects the per-round-trip
   `sqrt`+gather kick vs the precomputed constant `pump_kick`.

*(The same `None`-as-static-flag pattern is used by non-noise features at
`lle_solver.py:715-718`, `:735-756`, `:761-762`, `:924-925`; noted only because it means a
`None` → array change silently forces a full retrace.)*

**Host-side branches that select which graph is traced, or what is fed into it:**

6. `lle_solver.py:1071-1081` — `if _noise_model.is_colored:` a Python loop over host numpy
   synthesis vs `jax.vmap` of the traced AR(1). The traced program depends on a **config
   string** (`trn_psd_model`), not on a boolean.
7. `lle_solver.py:1109-1118` — the same branch inside `_delta_t_sequences`.
8. `simulator/noise_models.py:277-280`, `:368-372`, `:553-568`, `:602-616` — `is_colored` /
   `enabled` branches inside the samplers themselves (traced AR(1) vs host FFT synthesis vs
   `_zeros`). `noise_models.py:553-558` in particular means **the number of RNG draws taken
   depends on a flag** (safe only because the split at `:552` is unconditional — the code
   documents this at `:548-551`).
9. `lle_solver.py:1620-1684` — `if qn_enabled:` … `else:` sets `qnoise_scale` and the labeler
   vacuum-floor parameters (else branch `:1680-1684`). These flow into
   `_physical_state_labeler` (`lle_solver.py:1923-1931`), which is an `lru_cache`d **static**
   argument (`static_argnums` 13, `lle_solver.py:1141`) — so the quantum flag changes the
   labeler object identity and forces a different traced labeler graph plus a full recompile.
10. `simulator/state_labeler.py:199-205` — `if ENV_SMOOTH_W > 1:` and
    `if VACUUM_FLOOR_LEVEL > 0.0:` are build-time Python branches inside the labeler closure,
    driven (via `make_threshold_params`, `state_labeler.py:94-95`) by the quantum flag.
11. `lle_solver.py:1852-1882` — `if qn_enabled and qn_seed_vacuum:` builds the vacuum
    cold-start field on the host and hands it in as a **warm** start, bypassing the traced
    cold-start seed at `lle_solver.py:928-955`; the traced `_is_cold` mask
    (`lle_solver.py:952`) then selects the warm branch. The initial-condition graph therefore
    differs on a noise flag.
12. `lle_solver.py:1761-1764` — `if np.any(pump_freq_noise_history):` — this one is
    **data-dependent, not flag-dependent**: whether the pump-frequency term is added at all is
    decided by the *realized sample values*. The same predicate governs the presence of the
    output key at `lle_solver.py:1984-1985`, so the result dict's key set depends on drawn
    data. (An all-zero deterministic override silently produces no `pump_freq_noise_history`
    key.)
13. `lle_solver.py:1781-1788` — decides `pump_scale_sequence = array` vs `None`, i.e. which of
    the two branches at `lle_solver.py:871-879` is traced. See footnote [b] on its predicate.
14. `lle_solver.py:1796-1828` — the override/flag branch selecting `fsr_noise_sequences` as an
    array vs `None`, i.e. which of branches 1/2/4 above are traced. Also gates the output key
    at `lle_solver.py:1988-1989`.
15. `lle_solver.py:1986-1987` — `if pump_scale_sequence is not None:` gates the
    `pump_rin_epsilon_history` output key, so the returned dict's **key set** is a function of
    the noise flags (deliberate; `lle_solver.py:1990-1993` explains why the provenance stamp is
    also off by default for the same reason).
16. `lle_solver.py:1834-1846` → `probe_bins` (`static_argnums` 31): not a noise flag, but the
    same mechanism, and `mode_probe_indices` is the noise-metrology entry point.

### G3 — dtype boundaries where a float32 array feeds the float64 solver

`jax_enable_x64` is forced at `simulator/lle_solver.py:29-30`, so the solver graph is
float64 / complex128. The **classical detuning-noise chain is nevertheless generated in
float32** and upcast at the boundary. (Measured at audit time: `_ar1_samples(...)` returns
`float32`; `TotalNoise.sample_with_delta_t(...)` returns `(float32, float32)` when x64 is
off and `(float32, float64)` when x64 is on.)

1. **`simulator/noise_models.py:184` and `:190`** — `_ar1_samples` draws
   `xi = jax.random.normal(..., dtype=jnp.float32)` and initialises the `lax.scan` carry as
   `jnp.zeros((), dtype=jnp.float32)`. `alpha` (`:182`) and `sigma_step` (`:183`) are
   weakly-typed float64 scalars derived from Python floats, which do not promote a float32
   operand; `jax.lax.scan` additionally requires the carry dtype to be invariant, so the whole
   AR(1) recursion runs in float32. **Confirmed by execution** (returned dtype is `float32`).
   This is the generator for **TRN, pyro-EO, TCCR and the legacy dT**.
2. **`simulator/noise_models.py:570-571`** — `c_pull * temp_noise` and
   `pyro_coeff * temp_noise`: Python float (weak f64) × f32 → **f32**. With
   `c_pull ≈ (ω₀/n₀)·dn_dT ≈ 1.6e10 rad/s/K`, the ~7-decimal-digit float32 mantissa sets the
   granularity of the detuning-noise amplitude.
3. **`simulator/noise_models.py:578`** — `combined = (trn_noise - pyroeo_noise +
   tccr_noise).astype(jnp.float32)` — an **explicit downcast** of the summed sequence, and the
   point at which the TRN/pyro cancellation (`c_pull − pyro_coeff`) is rounded.
4. **`simulator/lle_solver.py:1081`** — `jax.vmap(_gen_noise)(noise_keys).astype(jnp.float64)`
   — the f32 sequence is **upcast into the f64 solver**. From here on the array is float64-typed
   but carries only float32 information. It becomes `noise_sequence` (`lle_solver.py:856`),
   `delta_omega_eff` (`:782`) and the complex128 linear exponent (`:787`, `:793`).
5. **`simulator/noise_models.py:582-583`** — dT is upcast to float64 only
   `if jax.config.read("jax_enable_x64")`; the *content* is still the float32 AR(1). Paired
   with **`simulator/lle_solver.py:1118`** (`dts.astype(jnp.float64)`). This is the dT that
   scales the FSR channel at `lle_solver.py:1810-1812`, so δD₁(t) is float32-precision inside
   a float64 array. Note the conditional makes the **returned dtype flag-dependent**
   (f32 without x64, f64 with) — a latent shape/dtype trap for any caller that imports
   `noise_models` without importing `lle_solver` first.
6. **`simulator/noise_models.py:558`** — the disabled path returns
   `_zeros(N, jnp.float64 if self.is_colored else jnp.float32)`; `_zeros`
   (`noise_models.py:58-70`) canonicalizes through `jax.dtypes.canonicalize_dtype`, so a
   requested float64 silently degrades to float32 when x64 is off. Deliberate (it matches the
   enabled path), but it is a dtype boundary that depends on process-global config.
7. **`simulator/lle_solver.py:1762-1763`** — `noise_sequences + jnp.asarray(
   pump_freq_noise_history, dtype=jnp.float64)` mixes a genuinely float64 host array with the
   float32-precision detuning array; the sum is float64-typed but inherits the coarser
   granularity of its TRN component.
8. **`simulator/state_labeler.py:149, 167, 213-214, 251-252, 265`** —
   `entropy_max = jnp.log(jnp.array(n_tau, dtype=jnp.float32))`,
   `sign_changes = ...astype(jnp.float32)`, the two `mono_frac` reductions
   `...astype(jnp.float32)`, `sentinel = jnp.float32(n_tau)` / the float32 `peak_locs`, and
   `valid_idx = jnp.arange(n_tau, dtype=jnp.float32)`. These are float32 reductions inside the
   labeler that is invoked on the complex128 field at `lle_solver.py:906`. They affect the
   discrete label only, never the field.
9. **`simulator/state_labeler.py:516`** — `assert_labelers_consistent` casts the field to
   `jnp.complex64` before running the JAX labeler and compares against the NumPy labeler run in
   float64. A diagnostic helper, not the solver hot path, but a genuine complex64/float64
   comparison boundary.

**Not** dtype boundaries (verified float64 end-to-end from the source):

* the colored-noise engine and every PSD factory — `simulator/colored_noise.py:78-124`,
  `:127-141`, `:147-296`, explicit `dtype=np.float64` throughout;
* both `PumpNoise` samplers and PSDs — `simulator/noise_models.py:734-810`;
* the quantum Langevin increment — `simulator/lle_solver.py:599-603` (float64 draws →
  complex128) and the vacuum cold-start seed `:1872-1879`;
* `TotalNoise.sample_full_with_delta_t` — `simulator/noise_models.py:589-617`, numpy float64
  throughout (this is the `legacy_segment_noise = false` path).

### G4 — Declared-but-unconsumed noise parameters (not requested, recorded because it changes how G1/G2 read)

Two fields exist in `NoiseConfig` and in the committed `noise:` block but are read by **no**
generator:

* `trn_ar1_stationary_init` — declared `simulator/noise_config.py:198`, documented
  `:168-170`, written `config/sin_params.yaml:53`. `grep` finds no consumer in
  `simulator/noise_models.py`; `_ar1_samples` (`simulator/noise_models.py:180-191`) takes no
  such parameter and unconditionally starts the scan carry at `jnp.zeros(())` (`:190`).
  Consequence: the AR(1) variance ramps as `sigma² · (1 − alpha^(2n))` rather than being
  stationary from sample 0. With the committed config (measured) `t_r = 4.065e-11 s`,
  `tau_th = 5.0e-6 s`, `alpha = 0.99999187`, the burn-in scale `tau_th/(2·t_r) ≈ 6.1e4`
  round trips.
  *(Addressed in the commit immediately following this audit: `_ar1_samples` gained a
  `stationary_init` argument, `TRNoise`/`TotalNoise` now thread the field, and
  `analysis/trn_burnin_study.py` measures the bias. The flag still defaults to `False`,
  so the description above remains the DEFAULT behaviour.)*
* `noise_dtype` — declared `simulator/noise_config.py:203`, written
  `config/sin_params.yaml:55`. No generator reads it; the working precision is hard-coded
  float32 at `simulator/noise_models.py:184`, `:190`, `:578` (see G3).

Both are classified as *parameters* rather than switches
(`simulator/noise_config.py:86-90`), so `NoiseConfig.all_off()` says nothing about them and
`tests/test_noise_config.py` only checks their defaults, not that they do anything.
