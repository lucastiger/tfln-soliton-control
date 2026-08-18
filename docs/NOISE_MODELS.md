# Noise models

The physics of all eight switchable channels, the equation each one implements, and the
config keys that control it.

Reference throughout: Herr, Tikan & Kippenberg, **arXiv:2604.05897v1** (7 Apr 2026).
Equation and section numbers are v1 numbers — later versions may renumber.

For the machine-checkable version of this table — continuum equation, discretized form,
implementing symbol and pinning test per channel, generated from
`simulator/equation_map.py` — see [`EQUATION_MAP.md`](EQUATION_MAP.md).

---

## Conventions

Every detuning-noise channel returns a contribution to $\delta\omega$ [rad/s] under the
repository's sign convention

$$\delta\omega = \omega_\mathrm{res} - \omega_\mathrm{pump},$$

matching the implemented term $-i\,\delta\omega\,E$. The field $E$ is in $\sqrt{\mathrm{J}}$.

Every sampler is deterministic in its JAX PRNG key. Colored models synthesize **host-side in
float64** through `simulator/colored_noise.py`; the historical `single_pole` path keeps the
traced AR(1) recursion, which is bit-identical to the pre-colored-noise code.

A disabled channel returns exact zeros **without drawing from its key**. That is safe
because JAX PRNG is functional — consuming a key has no side effect — but it makes the split
ladders load-bearing: they are unconditional, so toggling one channel can never shift
another channel's stream. That property is what makes the noise budget's common random
numbers exact rather than approximate.

---

## 1. Quantum vacuum — Sec. V.B.2, Eq. 126

Each cavity mode is driven by the vacuum Langevin input

$$\partial_t E = \dots + \sqrt{\kappa}\,\hat\xi_\mu(t), \qquad
\langle \hat\xi_\mu(t)\,\hat\xi^\dagger_{\mu'}(t')\rangle = \delta(t-t')\,\delta_{\mu\mu'},$$

with both loss baths ($\kappa_0$, $\kappa_\mathrm{ex}$) combined, since both are
coherent/vacuum baths — no squeezing. In the classical **truncated-Wigner**
(symmetric-ordering) c-number limit this is additive complex Gaussian noise with
$\langle \xi_\mu \xi^*_{\mu'}\rangle = \tfrac12\delta(t-t')\delta_{\mu\mu'}$, whose undriven
steady state is the symmetric-ordered vacuum occupation of **½ photon per mode**.

**Implementation.** Injected in the *time* domain, once per fine step
$\Delta t_\mathrm{fine} = t_r/M$ — deliberately avoiding extra FFTs. Every fast-time sample
receives an i.i.d. complex Gaussian of per-quadrature standard deviation

$$\sigma_q = \sqrt{\hbar\omega_0\,\kappa\,n_\tau\,\Delta t / 4}\,,$$

which by Parseval ($\tilde E_\mu = a_\mu n_\tau\sqrt{\hbar\omega_0}$,
$n_\mu = |\tilde E_\mu|^2/(n_\tau^2\hbar\omega_0)$) equals an independent photon-amplitude
increment of variance $(\kappa/2)\Delta t$ per mode. The increment is added **after** the
Strang sub-step loop and **before** the edge absorber and validity mask, so the numerical
masks damp rather than re-populate edge modes.

**Additive on $E$. White** — i.i.d. per fast-time sample, so a flat PSD.

| Config key | Default | Meaning |
|---|---|---|
| `noise.quantum_vacuum` | `false` | the switch |
| `quantum_noise_enabled` | `0` | deprecated legacy alias |
| `quantum_noise_seed_vacuum_init` | `1` | seed a cold start at ⟨n⟩ = ½ instead of the legacy `1e-3·|e_cw|` |
| `quantum_noise_injection_cadence` | `0` | 0 = per fine step (exact); 1 = per round trip |
| `hbar_omega0_j` | `0.0` | 0 = compute from `pump_wavelength_m` |
| `labeler_vacuum_floor_margin` | `10.0` | +10 dB margin for the labeler's vacuum floor |
| `labeler_envelope_smooth_modes` | `8` | smoothing width for the envelope gate |

The round-trip cadence is a CPU-performance knob, valid because
$\kappa t_r \approx 6.2\times10^{-3} \ll 1$ keeps even per-round-trip injection deep in the
continuum limit (steady occupation 0.5015 vs 0.5); with `fine_cadence_M = 1` the two
cadences are bit-identical.

When — and only when — this channel is enabled, the state labeler activates its vacuum-floor
parameters, so a vacuum-filled cavity labels OFF rather than CW. With the channel disabled
the labeler is bit-identical to the legacy one.

> **⚠ Symmetric ordering.** Spectra from this channel sit on a pedestal of
> $n_\tau^2\hbar\omega_0/2$ per mode. See [`LIMITATIONS.md`](LIMITATIONS.md) §4 before
> comparing to measurement.

---

## 2. Thermorefractive (TRN) — Eqs. 129–130

A thermodynamic temperature fluctuation $\delta T(t)$ [K] pulled onto the resonance:

$$\delta\omega(t) = C_\mathrm{pull}\,\delta T(t), \qquad
C_\mathrm{pull} = \frac{\omega_0}{n_0}\left(\frac{dn}{dT} + n_0\,\alpha_L\right),$$

with the Eq. 129 thermodynamic variance

$$\mathrm{Var}(\delta T) = \frac{k_B T^2}{\rho\,C_p\,V}.$$

**Multiplicative on $E$** — additive in the *detuning*, but it reaches the field as the
phase factor $\exp(-i\,\delta\omega\,\Delta t)$ in the linear half-step.

**Colored.** Three selectable spectra via `trn_psd_model`:

* **`single_pole`** (default) — the Lorentzian spectral twin of the historical AR(1)
  generator, correlation time $\tau_\mathrm{th}$:
  $$S_{\delta\omega}(f) = C_\mathrm{pull}^2 \cdot \frac{4\,\mathrm{Var}(\delta T)\,\tau_\mathrm{th}}{1 + (2\pi f \tau_\mathrm{th})^2}.$$
  The sampled stream is **byte-compatible** with the pre-colored-noise code.
* **`kondratiev_gorodetsky`** — the analytic whispering-gallery PSD of Eq. 130,
  $$S_{\delta T}(\omega) \propto \frac{k_B T^2}{\sqrt{\pi^3 \kappa_\mathrm{th}\rho C \omega}}
    \cdot \frac{1}{R\sqrt{d_a^2 - d_b^2}} \cdot \frac{1}{\left[1 + (\omega\tau_d)^{3/4}\right]^2},$$
  $\tau_d = (\pi/4)^{1/3}(\rho C/\kappa_\mathrm{th})d_b^2$. The analytic form is an
  *asymptotic matching* of the low- and high-frequency limits, so its absolute integral does
  not reproduce the thermodynamic variance; it is therefore **renormalized by a single
  constant** so that $\int S_{\delta T}\,df$ equals the Eq. 129 value. The curve supplies
  only the shape. This is documented behaviour, not a fudge — it is what makes the two TRN
  models carry the same total power and differ only in spectral shape.
* **`csv`** — a measured or FEM-tabulated PSD (Huang et al. 2019 style), interpolated
  linearly in log–log space and clamped flat outside the tabulated span.

| Config key | Default | |
|---|---|---|
| `noise.trn` | `true` | the switch |
| `T_k` | `300.0` | K, ambient |
| `tau_th_s` | `5.0e-6` | s, correlation time |
| `dn_dT_per_k` | `2.45e-5` | 1/K |
| `rho_kg_per_m3` / `Cp_j_per_kg_k` / `mode_volume_m3` | `3.17e3` / `700.0` / `1.954e-14` | Eq. 129 |
| `trn_psd_model` | `single_pole` | spectrum selector |
| `trn_R_m` / `trn_da_m` / `trn_db_m` | `null` | required (positive) iff K–G; asserts $d_a \ge 1.2\,d_b$ |
| `trn_psd_csv_path` / `trn_csv_units` | `null` / `S_delta_T` | required iff `csv` |
| `trn_ar1_stationary_init` | `false` | see the burn-in note below |

**AR(1) burn-in.** With `trn_ar1_stationary_init = false` (the historical default) the scan
carry starts at exactly 0, so the record is **not** stationary: its variance ramps as
$1 - \alpha^{2n}$ with $\alpha = e^{-t_r/\tau_\mathrm{th}}$, a burn-in scale of
$\tau_\mathrm{th}/(2t_r) \approx 6.15\times10^4$ round trips — longer than many
trajectories. A short run therefore has systematically suppressed amplitude. Setting the
flag `true` starts from the stationary distribution instead; it is off by default because it
changes every number.

---

## 3. Thermal-expansion pull

The paper's "dimensional fluctuation" companion of TRN, folded into the pull coefficient
rather than modelled as a separate process:

$$C_\mathrm{pull} = \frac{\omega_0}{n_0}\left(\frac{dn}{dT} + n_0\,\alpha_L\right).$$

**Not an independent stochastic channel** — no generator, no independent RNG draw, and no
boolean switch. It is controlled by the value `alpha_L_per_k` alone (default `0.0`, which
reproduces the historical thermo-optic-only pull *bit-for-bit*, since $x + n_0\cdot 0.0 = x$
exactly in IEEE arithmetic). It scales the TRN amplitude and, through it, the FSR channel.

It does **not** enter the pyro-EO pull, which carries its own coefficient.

---

## 4. Pyro-electric / electro-optic (Pyro-EO)

A temperature fluctuation releases pyro-electric surface charge, whose field shifts the
resonance through the electro-optic effect:

$$\delta\omega(t) = \frac{\omega_0\,n_0^2\,r_{33}\,p}{2\,\varepsilon_0\,\varepsilon_{r,\mathrm{eff}}}\;\delta T(t).$$

The screening factor is a 1-D approximation of the dielectric boundary conditions in the
thin-film stack — the field extending into the claddings dilutes $\varepsilon_{r,z}$:

$$\varepsilon_{r,\mathrm{eff}} = \varepsilon_{r,z}
  + \varepsilon_{r,\mathrm{top}}\frac{t_\mathrm{top}}{t_\mathrm{LN}}
  + \varepsilon_{r,\mathrm{bot}}\frac{t_\mathrm{bot}}{t_\mathrm{LN}}.$$

**Driven by the SAME $\delta T(t)$ realization as TRN** — one thermodynamic fluctuation seen
through two pull coefficients. They are therefore added **coherently**, with opposite signs
for z-cut TFLN with an air top cladding:

$$\delta\omega(t) = \left(C_\mathrm{pull} - p_\mathrm{coeff}\right)\delta T(t) + \delta\omega_\mathrm{TCCR}(t).$$

Adding two independent draws instead would get the total variance wrong.

**Multiplicative on $E$**, same detuning phase as TRN, same spectrum.

| Config key | Default (SiN) | |
|---|---|---|
| `noise.pyro_eo` | `true` | the switch |
| `eo_r33_m_per_v` | `0.0` | **zero on centrosymmetric SiN** |
| `pyroelectric_coeff_c_per_m2_k` | `0.0` | likewise |
| `eps_r_z`, `t_ln_m`, `t_clad_top_m`, `t_clad_bot_m`, `eps_r_clad_top`, `eps_r_clad_bot` | built-in defaults | stack screening |

On the committed SiN device $r_{33} = 0$, so this channel's coefficient is identically zero
even with the switch on. That is a **material** fact, not a switch, and is deliberately not
folded into `enabled`.

---

## 5. Thermal-carrier / surface-state (TCCR)

Surface-state carriers fluctuate by shot noise about their equilibrium number
$N_{s,\mathrm{eq}} = n_s A_\mathrm{eff}$. Each carrier contributes a field
$E_c = e/(\varepsilon_0\varepsilon_{r,\mathrm{eff}}A_\mathrm{eff})$, which the electro-optic
effect turns into a resonance shift

$$\frac{d\omega}{dN_s} = -\frac{\omega_0 n_0^2 r_{33} E_c}{2}.$$

With an exponential carrier autocorrelation of time $\tau_\mathrm{carrier}$ the spectrum is a
Lorentzian

$$S_{\delta\omega}(f) = \frac{s_0}{1 + (2\pi f \tau_\mathrm{carrier})^2}, \qquad
s_0 = \left(\frac{d\omega}{dN_s}\right)^2 N_{s,\mathrm{eq}}\,2\tau_\mathrm{carrier}.$$

The corner sits at $1/(2\pi\tau_\mathrm{carrier})$ — with the default 100 ns that is about
1.6 MHz, three decades above the thermorefractive corner, which is what makes the two
channels separable in a measured spectrum.

**Multiplicative on $E$** (detuning phase). **Colored**, Lorentzian, $\tau_\mathrm{carrier}$.

**Independent random stream.** Unlike the two thermal channels it has its own AR(1) fed by
the second subkey of an unconditional two-way split.

> **This channel is NOT gated by `T_k`** — its variance carries no $T^2$ factor. That is
> exactly why the historical "`T_k = 0` means noise off" convention never silenced it on
> χ⁽²⁾ platforms, and why the explicit switch exists.

| Config key | Default (SiN) | |
|---|---|---|
| `noise.tccr` | `false` | the switch |
| `tau_carrier_s` | `1e-7` | s |
| `surface_state_density_per_m2` | `1e16` | m⁻² |
| `eo_r33_m_per_v` | `0.0` | zero on SiN ⇒ $\sigma_\mathrm{TCCR} = 0$ |
| `effective_mode_area_m2` | `3.344e-12` | m² |

Construction warns if $\sigma_\mathrm{TCCR}$ is non-zero but outside $[10^4, 10^{11}]$ rad/s,
or if it exceeds the cavity linewidth — in which case the channel is non-perturbative and
would destabilize every soliton.

---

## 6. Pump frequency noise — Sec. V.B.4

The solver frame co-rotates with the pump, so the instantaneous laser-frequency deviation
$\delta\nu_p(t)$ *is* a detuning noise. One-sided PSD:

$$S_{\delta\nu}(f) = h_0 + \frac{h_{-1}}{f}\quad[\mathrm{Hz^2/Hz}],$$

a white plateau $h_0$ carrying the intrinsic Lorentzian linewidth
$\Delta\nu_L = \pi h_0$, plus a flicker term. The white part is drawn i.i.d. per round trip
with variance $h_0 f_s/2$; the flicker part is FFT-synthesized with the DC bin clamped to the
first bin $f_1 = f_s/N$, so the single DC bin carries $h_{-1}$ rather than diverging.

**Sign.** Because $\delta\omega = \omega_\mathrm{res} - \omega_p$, a positive laser-frequency
excursion **lowers** $\delta\omega$: the contribution $-2\pi\,\delta\nu_p(t)$ is summed into
the detuning-noise sequence **on the host**, so the solver scan is unchanged and the cavity
low-pass and quadrature rotation emerge from the equations of motion. No transfer function
is hand-implemented — the solver *is* the transfer function.

**Multiplicative on $E$** (detuning phase). **Colored**, $h_0 + h_{-1}/f$.

| Config key | Default | Representative |
|---|---|---|
| `noise.pump_freq_noise` | `false` | |
| `pump_freq_noise_h0_hz2_per_hz` | `0.0` | ECDL ≈ 3e3 ⇒ Δν_L ≈ 10 kHz; fibre laser ≈ 30 ⇒ ≈ 100 Hz |
| `pump_freq_noise_hm1_hz3_per_hz` | `0.0` | ECDL ≈ 1e10 |

---

## 7. Pump RIN — Sec. V.B.5

$P_\mathrm{in}(t) = \bar P_\mathrm{in}\,(1 + \varepsilon(t))$ with one-sided PSD

$$S_\varepsilon(f) = 10^{\mathrm{floor}/10} + 10^{\mathrm{excess}/10}\frac{f_c}{f}\ (f < f_c),
\qquad = 10^{\mathrm{floor}/10}\ (f \ge f_c).$$

$\varepsilon$ is clipped so $1 + \varepsilon \ge 0$; if more than 0.01 % of samples clip, a
warning reports the fraction rather than silently renormalizing.

**Neither purely additive nor purely multiplicative on $E$** — classified `mixed`. It is
multiplicative on the *drive amplitude*: the pump kick becomes
$\sqrt{\max(\kappa_c \bar P_\mathrm{in}(1+\varepsilon),\,0)}\,\Delta t$, held constant across
the round trip's fine and sub-steps (RIN bandwidth ≪ FSR, so per-round-trip resolution is
exact). The drive then enters $E$ *additively*. Secondarily it reaches $E$ multiplicatively
through the thermal pathway: RIN → $P_\mathrm{abs}$ → $\Delta T$ → detuning, transduced by
the existing thermo-optic ODE with no extra code.

| Config key | Default | |
|---|---|---|
| `noise.pump_rin` | `false` | |
| `pump_rin_floor_dbc_per_hz` | `-300.0` | validation rejects anything above −80 dBc/Hz |
| `pump_rin_excess_dbc_per_hz` | `-300.0` | (a guard against a linear density entered where dB is expected) |
| `pump_rin_corner_hz` | `1.0e4` | |

---

## 8. FSR / repetition-rate noise — the TRN-limited $f_\mathrm{rep}$ term

A temperature excursion changes the round-trip time as well as the resonance:

$$\delta D_1(t) = \frac{D_1}{\omega_0}\,C_\mathrm{pull}\,\delta T(t),$$

and each mode $\mu$ acquires the extra **mode-linear** detuning $\mu\,\delta D_1(t)$ inside
the linear operator. This is distinct from the uniform $\delta\omega$ of channels 2–7: it
tilts the comb rather than shifting it.

**Driven by the SAME $\delta T(t)$ realization as TRN and pyro-EO**, regenerated
deterministically from the same PRNG keys. Consequently `fsr = true` with `trn = false`
produces **identically zero** FSR noise, and the code warns about exactly that combination.

**Multiplicative on $E$** (mode-dependent phase in the Fourier domain). **Colored** —
identical spectrum to TRN, scaled by $(D_1/\omega_0)C_\mathrm{pull}$; on the committed device
$D_1/\omega_0 \approx 1.27\times10^{-4}$.

| Config key | Default | |
|---|---|---|
| `noise.fsr` | `false` | the switch |
| `fsr_noise_enabled` | `0` | deprecated legacy alias |

Magnitude: a $\delta\omega_0 \sim 10^5$ rad/s TRN excursion gives $\delta D_1 \sim 13$ rad/s,
i.e. $\sim1.3\times10^4$ rad/s at $|\mu| = 10^3$ — perturbative against $\kappa$, yet it is
*the* TRN-limited repetition-rate term.

---

## What shares a random source with what

This matters for any per-channel attribution:

| Channel | Shares its source with |
|---|---|
| TRN, thermal-expansion pull, Pyro-EO, FSR | **each other** — one $\delta T(t)$ realization |
| TCCR | nothing (independent subkey) |
| Quantum vacuum | nothing (own key chain) |
| Pump frequency noise | nothing (own subkey) |
| Pump RIN | nothing (own subkey, sibling of the above) |

The four members of the $\delta T$ family are one thermodynamic fluctuation seen four ways.
This is why `analysis/noise_budget.py` never runs `pyro_eo` or `fsr` alone — those rows
carry `trn`, and the physically meaningful grouped channel is `dT_family`. The budget
includes a built-in check that the `pyro_eo` row reproduces the `trn` row bit-for-bit on
centrosymmetric SiN, which is a live proof that common random numbers are in effect.

---

## Infrastructure

`simulator/colored_noise.py` synthesizes any channel from a one-sided target PSD. The exact
recipe: on the rfft bins $f_k = k f_s/N$, draw $\zeta_k$ as a standard complex normal
($E|\zeta_k|^2 = 1$) for $0 < k < N/2$ and a standard real normal at $k = 0$ and (even $N$)
$k = N/2$; set

$$c_k = \zeta_k\sqrt{S(f_k)\,f_s\,N/2}, \qquad x = \mathrm{irfft}(c, n=N).$$

Then $\mathrm{Var}(x) = \sum_k S(f_k)\Delta f \approx \int_0^{f_s/2} S\,df$ and the Welch PSD
of $x$ reproduces $S(f_k)$ bin by bin. The DC bin is clamped, $S(0) := S(f_1)$, so $1/f$-type
spectra cannot inject divergent DC power. The result is **stationary from sample 0** — no
AR(1)-style start-up transient.

Aliasing: these channels are sampled once per round trip at $f_s = 1/t_r \approx 24.6$ GHz,
far above any thermal band, so synthesis-band truncation is negligible.

Determinism is anchored to JAX PRNG keys via `np_generator_from_key`, which folds the full
key data into a `numpy.random.SeedSequence` — one seeding convention for every host-side
synthesis in the repository.
