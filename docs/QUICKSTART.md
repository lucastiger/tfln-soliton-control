# Quickstart

From a clean checkout to a soliton, then to a noise number. About ten minutes of reading,
under two minutes of compute.

If you would rather run it than read it, `notebooks/01_quickstart.ipynb` covers sections
2–4 below and executes in about 40 seconds.

---

## 1. Install

Requires Python 3.10–3.12. Install **editable** — `simulator/lle_solver.py` resolves the
default config relative to the checkout, so a non-editable wheel imports but cannot find
`config/sin_params.yaml`.

```bash
git clone https://github.com/Mengjie-Yu-Group/stochastic-lle
cd stochastic-lle
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Check the install:

```bash
python -c "
import jax
from simulator.lle_solver import resolve_cavity_rates
print('x64:', jax.config.read('jax_enable_x64'))
print('kappa_i, kappa_c, kappa =', resolve_cavity_rates())
"
```

`x64: True` is mandatory. `simulator.lle_solver` forces it at import time, and JAX bakes the
flag in at array-creation time — so that module must be imported before any `jax.numpy`
array is built. In complex64 the accumulated round-off over hundreds of thousands of round
trips pins the spectral floor near −70 dB and buries the sub-−70 dB structure this benchmark
exists to resolve.

## 2. Conventions you need before reading any number

| Quantity | Units | Note |
|---|---|---|
| Field `E` | $\sqrt{\mathrm{J}}$ | so $\lvert E\rvert^2$ is in joules |
| `U_int` | J·s | intracavity energy; circulating power is `U_int / t_r` |
| `gamma_LLE` | J⁻¹ s⁻¹ | **not** the fibre-optics W⁻¹ m⁻¹ — use `gamma_nlse_to_lle()` |
| `beta_k` | s^(k−1) | $\beta_k = D_k/D_1^k$ — **not** fibre GVD in s²/m; use `d2_to_beta2_lle()` |
| `delta_omega` | rad/s | $\delta\omega = \omega_\mathrm{res} - \omega_\mathrm{pump}$ |

**Detuning sign.** Positive `delta_omega` is a **red-detuned** pump (pump below resonance),
which is the soliton side. Solitons exist for roughly
$\kappa/2 < \delta\omega \lesssim 5\kappa$; for adiabatic access, sweep from negative to
positive.

The two unit traps above are the classic ways to be wrong by several orders of magnitude, so
the solver asserts on both: `gamma_LLE` must lie in $(10^{15}, 10^{25})$ and `beta[0]` in
$[10^{-20}, 10^{-12}]$, and each failure message names the conversion function that fixes it.

## 3. A deterministic single soliton

Every stochastic channel is **off by default** — all switch fields of `NoiseConfig` default
to `False`. This is the trajectory the golden files pin to 0 ULP.

```python
import warnings
from analysis.dks_access import (PRODUCTION_NUMERICS, access_by_seeding,
                                 attach_dispersion, load_cavity_params)
from simulator.noise_config import NoiseConfig

warnings.simplefilter("ignore")          # the sech ansatz overflows cosh in its far tails

cav = attach_dispersion(load_cavity_params(), n_tau=2048)   # measured D_int(mu)
res = access_by_seeding(
    8.0 * cav.kappa, cav, t_slow=20_000, n_tau=2048,
    noise_config=NoiseConfig.all_off(thermal_feedback=True),
    **PRODUCTION_NUMERICS,
)
m = res["metrics"]
print(res["is_single"], m["np_label"], f"{m['sech2_env_corr']:.3f}")
# -> True 6 0.940
```

What each piece is doing:

* **`load_cavity_params()`** reads `config/sin_params.yaml`; **`attach_dispersion()`**
  replaces the Taylor $\beta$ polynomial with the measured $D_\mathrm{int}(\mu)$ grid. The
  Taylor truncation is only as good as its order, and the pyLLE cross-check is run against
  the measured grid.
* **`PRODUCTION_NUMERICS`** is the quantitative-spectrum stack: `n_substeps=4`,
  `dealias_two_thirds=True`, `edge_absorber=True`. All three default to *off* at the public
  API so the legacy path stays bit-identical; production physical runs should set them.
* **`thermal_feedback=True`** is named explicitly because it is *not* a noise channel — it is
  the deterministic thermo-optic ODE, and `NoiseConfig.all_off()` deliberately leaves it at
  its own default rather than forcing it.
* **class label 6** is a single dissipative Kerr soliton. The seven classes are
  0 off, 1 CW, 2 MI, 3 chaotic, 4 multi-soliton, 5 soliton crystal, 6 single soliton.

`n_tau = 2048` is the smallest grid at which the single-DKS discriminator passes cleanly
here; the vacuum-floor measurement in the next section needs 4096 to get a comb-free band.

## 4. Turning on exactly one noise channel

Every channel is opt-in and independently switchable. The switch is a `NoiseConfig` field:

```python
from simulator.noise_config import NoiseConfig

nc = NoiseConfig.all_off(thermal_feedback=True, quantum_vacuum=True)
print(nc.enabled_channels)     # ('quantum_vacuum',)
print(nc.sha256()[:16])        # digest of the whole config, for provenance
print(nc.describe())           # one line per channel, on or off
```

Pass it as `solve_lle_ssfm_jax(..., noise_config=nc)`, or as `noise_config=` through the
`analysis.dks_access` helpers. Precedence, highest first — implemented once, in
`simulator.lle_solver._resolve_noise_flags`:

1. an explicit `solve_lle_ssfm_jax()` keyword argument;
2. a `NoiseConfig` passed as `noise_config=`;
3. an explicit deprecated switch in `physical_parameters`
   (`quantum_noise_enabled`, `pump_noise_enabled`, `fsr_noise_enabled`);
4. the top-level `noise:` block of the config file;
5. an implicit physical gate (`T_k > 0`, `eo_r33_m_per_v != 0`);
6. the `NoiseConfig` field default.

Levels 3 and 5 are both "legacy", split deliberately around the block: a key someone
actually *wrote* into `physical_parameters` is an instruction and outranks a block that may
have arrived through a config round-trip, whereas an implicit physical gate is not an
instruction at all.

> **Config rule.** Every leaf under `physical_parameters` must parse as a plain number —
> `tests/test_config.py` locks that in. That is why the legacy switches there are `0`/`1`
> integers and why the real switches live in the top-level `noise:` block, which is free to
> carry booleans and string enums.

## 5. Seeing the vacuum floor

With `quantum_vacuum` on, the spectral floor rises to the symmetric-ordered vacuum level
$n_\tau^2\hbar\omega_0/2$ — half a photon per mode. Measure it in the repository's own wing
band, $|\mu| \in [0.244\,n_\tau,\ 0.317\,n_\tau]$, which sits **inside** the 2/3 de-aliasing
window; beyond $n_\tau/3$ the modes are zeroed after every nonlinear kick and read
artificially low.

Measured at $n_\tau = 4096$ (see `notebooks/01_quickstart.ipynb`):

| | wing floor | ÷ pedestal |
|---|---|---|
| vacuum OFF | 5.75e−15 | 0.0053 |
| vacuum ON, raw | 1.10e−12 | **1.019** |
| vacuum ON, `normal_ordered_spectrum()` | 2.06e−14 | 0.019 |

The third row is the one that matters for any comparison to measurement — see
[`LIMITATIONS.md`](LIMITATIONS.md) §4.

## 6. Where to go next

| You want | Go to |
|---|---|
| The physics of all eight channels | [`NOISE_MODELS.md`](NOISE_MODELS.md) |
| Which line of code is which paper equation | [`EQUATION_MAP.md`](EQUATION_MAP.md) |
| What is verified, and to what tolerance | [`VALIDATION.md`](VALIDATION.md) |
| The callable surface | [`API.md`](API.md) |
| Regenerating the paper | [`REPRODUCING_THE_PAPER.md`](REPRODUCING_THE_PAPER.md) |
| **What not to trust** | [`LIMITATIONS.md`](LIMITATIONS.md) |
