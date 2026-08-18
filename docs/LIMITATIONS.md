# Limitations

Read this before quoting any number this repository produced. Everything below is a
property of the benchmark as it currently stands, not a to-do list — each item states what
the limitation is, why it exists, and what to do about it.

Reference throughout: Herr, Tikan & Kippenberg, **arXiv:2604.05897v1** (7 Apr 2026). The
version pin matters: equation and section numbers are v1 numbers.

---

## 1. Bit-identity is a CPU/float64 claim on a pinned toolchain, not a portable one

The "all channels off reproduces the deterministic LLE bit-for-bit" claim is asserted on
**CPU, in float64, with a pinned `jaxlib`** — the environment in `requirements.lock.txt`.
On that environment `python -m validation.noise_off_identity --check --strict` reports
0 ULP across all four committed parameter sets.

**On GPU the agreement is ~1e-12, not 0 ULP.** XLA is free to reassociate reductions and
fuse kernels differently for a different device, and does. The same is true across CPU
microarchitectures: on a CI runner carrying the goldens' exact library versions
(jax/jaxlib 0.10.2, numpy 2.4.6, Python 3.11.15) the comparison still differs by
`max_abs_diff = 6.2e-19` because XLA compiled for a different CPU.

Treat `SOLITON_STRICT_ULP=1` as a **fixed-hardware** check. Quote the hardware alongside
the library versions in any reproducibility claim. The loose default
(`np.allclose(atol=1e-13, rtol=0)`) is what runs in CI, deliberately: demanding 0 ULP
unconditionally would turn every dependency bump into a red suite that looks like a physics
regression.

## 2. The classical noise stream is generated in float32 by default

The historical AR(1) generator that produces the shared temperature realization
$\delta T(t)$ — and therefore the thermorefractive, pyro-electric/EO and thermal-carrier
detuning streams — draws its innovations in **float32** and carries a float32 scan carry.
The result is widened to float64 before it reaches the linear operator, so the solver
itself runs in double precision throughout; but the *information content* of that stream is
single precision.

This is deliberate: it is what keeps the sampled stream byte-compatible with the
pre-colored-noise code, which is what the golden trajectories pin. `NoiseConfig` carries a
`noise_dtype` field (`"float32"` default, `"float64"` accepted and validated) intended to
select this.

> **Status note.** `noise_dtype` is currently validated, hashed into the config digest and
> printed by `describe()`, but **no sampler reads it** — the float32 dtype is hard-coded in
> the AR(1) generator. Setting `noise_dtype: float64` therefore changes the config hash
> without changing any number. The colored PSD models (`kondratiev_gorodetsky`, `csv`) and
> both pump channels are unaffected: they synthesize host-side in float64 already.

## 3. Truncated Wigner requires ⟨n⟩ ≫ 1; we operate at ~1.2 × 10⁸ photons

The quantum-vacuum channel is implemented in the **truncated-Wigner** (c-number) limit: the
third-order derivative terms of the Wigner equation of motion are dropped, which is
controlled only when the mode occupation is large. At the committed operating point the
pumped mode carries of order **1.2 × 10⁸ photons**, comfortably in that regime.

The approximation degrades for any study that deliberately empties the cavity — an undriven
vacuum-equilibrium run sits at ⟨n⟩ = 1/2 per mode by construction. Those runs are valid as
*calibrations of the noise normalization* (that is exactly what they check) but must not be
read as quantum-optical predictions about a nearly-empty cavity.

## 4. Simulated spectra are symmetric-ordered; use `normal_ordered_spectrum()` for measurement

Wigner-representation moments are **symmetrically ordered**, so a mode in the vacuum state
carries $\langle a a^\dagger + a^\dagger a\rangle / 2 = 1/2$ of a photon and every simulated
spectrum sits on a flat, state-independent pedestal of exactly

$$n_\tau^2 \, \hbar\omega_0 / 2$$

per mode (the $n_\tau^2$ follows from this repository's FFT convention,
$\tilde E_\mu = a_\mu \, n_\tau \sqrt{\hbar\omega_0}$).

A photodiode or optical spectrum analyser measures $\langle a^\dagger a\rangle$ — **normally
ordered** — and reads exactly zero for the vacuum. Comparing a raw simulated
$|\tilde E_\mu|^2$ against a measured spectrum silently adds that pedestal to the simulation
alone. At the committed operating point it sits about **−82 dB relative to the pump**, so it
dominates every comb feature weaker than that.

Subtract it with `analysis.spectral_metrics.normal_ordered_spectrum()` before any comparison
to measurement. Two rules:

* **Do not** use it for energy accounting. `U_int`, $P_\mathrm{abs} = \kappa_i \langle|E|^2\rangle$,
  the thermo-optic drive and the intracavity energy budget are all statements about the
  symmetric-ordered field the solver actually integrates; subtracting there would
  double-count the vacuum and break the energy balance.
* Pass `clip=False` whenever you intend to average, fit, or attach an error bar. Subtraction
  removes the vacuum's **bias**, not its **variance**: a single-snapshot vacuum mode is
  exponentially distributed with mean and standard deviation both equal to the pedestal, so
  individual residuals are routinely negative. Clipping at zero turns a zero-mean residual
  into a positively biased one and destroys exactly the cancellation averaging relies on.
  `clip=True` is for display only.

`notebooks/01_quickstart.ipynb` demonstrates the whole effect in about 40 seconds.

## 5. 1/f channels are record-length dependent — every linewidth carries its observation time

Both pump-laser channels carry $1/f$ components: frequency noise as
$S_{\delta\nu}(f) = h_0 + h_{-1}/f$, and RIN as a flat floor plus an excess
$\propto f_c/f$ below the corner. A $1/f$ process has no stationary variance, so **any**
statistic drawn from it — an RMS, a linewidth, a jitter — depends on how long you watched.
There is no "the" linewidth of such a source.

Consequences that are enforced rather than merely documented:

* Every number in the noise budget carries **the record that produced it**. The production
  records are `fast` ($t_\mathrm{slow} = 2\times10^5$, 8.13 µs) and `slow`
  ($2\times10^7$, 813 µs), and they reach different Fourier floors.
* A target frequency below a record's floor is reported as **`null`, with the record length
  that would observe it** — never interpolated or clamped to the nearest observed bin.
* Spectral numbers additionally record their **Welch segmentation** (segment length, segment
  count, achieved bin spacing), because Welch resolves $f_s/\mathrm{nperseg}$ rather than
  the record's Fourier floor.
* Effective linewidths use the β-separation line (Di Domenico et al., *Appl. Opt.* **49**,
  4801 (2010)), which is itself an observation-time construction.

Never compare two cells taken from different records.

## 6. The default thermal integrator is explicit Euler — first order

`NoiseConfig.thermal_integrator` defaults to `"euler"`: the historical explicit-Euler update
of the single-pole thermo-optic ODE. It is **order 1** and only conditionally stable — the
amplification factor is $1 - \Delta t/\tau_\mathrm{th}$, so the scheme **diverges** once
$\Delta t/\tau_\mathrm{th} > 2$. At the production ratio
$t_r/\tau_\mathrm{th} \approx 8.1\times10^{-6}$ it is far from that limit, but the order-1
truncation is real.

`"exponential"` is **exact** for piecewise-constant $P_\mathrm{abs}$:
$\Delta T_{n+1} = a\,\Delta T_n + (1-a)\,R_\mathrm{th}P_\mathrm{abs}$ with
$a = \exp(-\Delta t/\tau_\mathrm{th})$, computed with `expm1` to avoid cancellation. It
carries no truncation error in $\Delta T$ at all and is unconditionally stable.

Euler remains the default because **every committed golden trajectory was produced with it**
and switching would change every number. Note also that the exponential integrator alone
does not buy you second order: the field↔thermal *coupling* is lagged, which caps the
coupled system at order 1 regardless. Reaching order 2 needs
`thermal_integrator="exponential"` **and** `thermal_coupling="strang"` **and**
`symmetric_drive=True` — measured 2.00; see `docs/VALIDATION.md`.

## 7. ħω₀ is evaluated at the pump for all comb modes

The photon energy used to normalize the quantum-vacuum channel is
$\hbar\omega_0 = \hbar\,2\pi c/\lambda_p$, evaluated at the **pump wavelength**, and applied
to **every** comb mode. The error for mode $\mu$ is $|\mu|\,\mathrm{FSR}/f_0$, which is
**below 1 %** across the comb span used here.

This is stated rather than hidden. `hbar_omega0_j` in the config overrides it if you need a
different normalization (0 means "compute from `pump_wavelength_m`"); the config spells
"auto" as `0` because every leaf under `physical_parameters` must parse as a plain number.

## 8. The PI-RNN and MPC code under `model/` and `control/` is EXPERIMENTAL

`model/` (the physics-informed RNN observer, its ablation encoders, training loop and loss)
and `control/` (the MPC controller and hardware-interface stubs) are **not part of the
validated benchmark**. They are downstream research code:

* they are not covered by the verification hierarchy in `docs/VALIDATION.md`;
* no golden trajectory, bit-identity claim or convergence result depends on them;
* their dependencies (`torch`, `einops`) are an opt-in extra (`pip install -e ".[ml]"`)
  precisely because running an LLE should not require a ~1 GB deep-learning framework;
* without `torch` installed, `tests/test_model.py` and `tests/test_ablation_encoders.py`
  are dropped from collection and reported in the pytest header.

Do not cite results from `model/` or `control/` as benchmark output.

---

## Other things worth knowing

**The validated configuration is not the production configuration.** The verification suite
runs at smaller grids and shorter records than a production campaign. See the
configuration-coverage table in [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md) before
assuming a check covers the settings you are running.

**The pyLLE cross-check is FROZEN at verdict `FAIL (QUALIFIED)`.** Seven of seven HARD
checks pass; the qualification and the five named triggers that require it to be re-run are
in [`PYLLE_STATUS_V2.md`](PYLLE_STATUS_V2.md). Every result artifact is hash-pinned in
[`validation/results/FROZEN_MANIFEST.md`](../validation/results/FROZEN_MANIFEST.md).

**The shipping scheme is first order overall.** The Strang core is second order, but the
drive kick is applied once per sub-step *before* the linear half-step, which makes the
composite map non-palindromic. Measured order 1.00. `symmetric_drive=True` restores the
palindrome and measures 2.00; it is opt-in because enabling it changes every trajectory.

**`thermal_feedback` is a bookkeeping switch, not a solver switch.** It appears in
`NoiseConfig`, in the config digest and in the budget's channel sets, but the thermo-optic
ODE in the solver is unconditional — no code path turns it off. It is documented as "not a
noise channel"; just do not expect `thermal_feedback: false` to change a number.
