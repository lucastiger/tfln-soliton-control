# stochastic-lle

[![CI](https://github.com/lucastiger/stochastic-lle/actions/workflows/ci.yml/badge.svg)](https://github.com/lucastiger/stochastic-lle/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-pending%20first%20release-lightgrey)](CITATION.cff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)

A validated stochastic **Lugiato–Lefever** benchmark for Kerr-soliton microcombs: a JAX
split-step Fourier solver with **eight independently switchable noise channels** — quantum
vacuum, thermorefractive, thermal-expansion pull, pyro-electric/electro-optic,
thermal-carrier, pump frequency noise, pump RIN and TRN-driven repetition-rate noise — each
physically normalized against Herr, Tikan & Kippenberg, arXiv:2604.05897v1. With every
channel off the solver is **bit-identical to the deterministic LLE**, reproducing committed
golden trajectories at 0 ULP, which is what makes per-channel noise attribution falsifiable
rather than merely plausible: the difference between two runs is the channel, and nothing
else. The verification suite is part of the deliverable — an exact-CW fixed point to 1.9e-13,
a measured order of accuracy, and a cross-check against an independent implementation.

## Install

Requires Python 3.10–3.12. Install **editable**: the solver resolves its default config
relative to the checkout, so a non-editable wheel imports but cannot find
`config/sin_params.yaml`.

```bash
# pip
git clone https://github.com/Mengjie-Yu-Group/stochastic-lle
cd stochastic-lle
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # add ".[dev]" for the tests and notebooks
```

```bash
# conda
conda env create -f environment.yml
conda activate stochastic-lle
pip install --no-deps -e .
```

```bash
# docker — the pinned, hash-verified toolchain; runs the fast suite on start
docker build -t stochastic-lle .
docker run --rm stochastic-lle
```

Extras are opt-in: `dev` (pytest, ruff, jupyter, nbconvert, sympy), `ml` (torch, einops — the
experimental PI-RNN only), `pylle` (the cross-check; needs a *separate* environment and a
Julia toolchain). `torch` is deliberately not a base dependency — running an LLE should not
require a ~1 GB deep-learning framework.

> `environment.yml` is a conda convenience and is **not** the bit-identity environment;
> `requirements.lock.txt` is. Conda-forge builds numpy/scipy against a different BLAS, and
> different BLAS means different reduction order.

## 60-second quickstart

A deterministic single soliton, in ten lines:

```python
import warnings
from analysis.dks_access import (PRODUCTION_NUMERICS, access_by_seeding,
                                 attach_dispersion, load_cavity_params)
from simulator.noise_config import NoiseConfig

warnings.simplefilter("ignore")                              # sech ansatz overflows cosh in its tails
cav = attach_dispersion(load_cavity_params(), n_tau=2048)    # measured D_int(mu)
res = access_by_seeding(8.0 * cav.kappa, cav, t_slow=20_000, n_tau=2048,
                        noise_config=NoiseConfig.all_off(thermal_feedback=True),
                        **PRODUCTION_NUMERICS)               # every stochastic channel OFF
print(res["is_single"], res["metrics"]["np_label"])          # -> True 6   (6 = single DKS)
```

About 20 seconds. `delta_omega = omega_res - omega_pump`, so positive is red-detuned — the
soliton side. Then turn on exactly one channel:

```python
nc = NoiseConfig.all_off(thermal_feedback=True, quantum_vacuum=True)
print(nc.enabled_channels)     # ('quantum_vacuum',)
```

`notebooks/01_quickstart.ipynb` runs this and watches the spectral floor rise to
ħω₀/2 per mode, then come back down under `normal_ordered_spectrum()` — 40 seconds,
headless.

## The noise channels

Generated from `simulator/equation_map.py`; the full render, with the continuum and
discretized form of every equation and the test that pins it, is in
[`docs/EQUATION_MAP.md`](docs/EQUATION_MAP.md). Physics and config keys:
[`docs/NOISE_MODELS.md`](docs/NOISE_MODELS.md).

| Channel | Paper eq. | Section | Enters as | Add/mult | Colour | Shares source with |
| --- | --- | --- | --- | --- | --- | --- |
| quantum_vacuum | Eq. 126 | Sec. V.B.2 | additive field increment | additive | white | -- |
| trn | Eqs. 129-130 | -- | detuning phase rotation | multiplicative | lorentzian(tau_th) \| kondratiev_gorodetsky \| csv | pyro_eo, fsr |
| pyro_eo | -- | -- | detuning phase rotation | multiplicative | lorentzian(tau_th) \| kondratiev_gorodetsky \| csv | trn, fsr |
| tccr | -- | -- | detuning phase rotation | multiplicative | lorentzian(tau_carrier) | -- |
| pump_freq_noise | -- | Sec. V.B.4 | detuning phase rotation | multiplicative | h0 + h-1/f | -- |
| pump_rin | -- | Sec. V.B.5 | drive amplitude scale | mixed | rin_floor + rin_excess*(f_c/f) below f_c | -- |
| fsr | -- | Sec. V.B.1 | mode-linear detuning | multiplicative | lorentzian(tau_th) \| kondratiev_gorodetsky \| csv | trn, pyro_eo |
| thermal_feedback | -- | -- | detuning phase rotation | multiplicative | deterministic (no stochastic source) | -- |

`--` means the repository does not record that cross-reference; it is left blank rather than
guessed. `thermal_feedback` is a switch but **not** a noise channel — it names the
deterministic thermo-optic ODE.

Every channel is off by default, and the four members of the δT family (`trn`, the
expansion pull, `pyro_eo`, `fsr`) share **one** temperature realization — they are one
thermodynamic fluctuation seen four ways, which is why the budget never runs `pyro_eo` or
`fsr` alone.

## Verification

Each claim, and the command that proves it. Runtimes are for a 4-core CPU.

| Claim | Command | Result |
|---|---|---|
| All channels off ⇒ **0 ULP** vs committed goldens | `python -m validation.noise_off_identity --check --strict` | `4 param sets, mode=strict (0 ULP): 0 differences` — 12 s |
| …as a test, plus 3 further failure modes | `SOLITON_STRICT_ULP=1 pytest tests/test_noise_off_identity.py -q` | `12 passed, 4 skipped` — 53 s |
| Exact CW fixed point, gate 1e-12 | `python -m validation.analytic_cw` | residual **1.922e-13** over 405 points — 1 m 52 s |
| Observed order of accuracy | `python -m validation.convergence --report` | `PASS`; 2.00 / 1.00 / 2.00, weak 3.05 — 4 m 38 s |
| pyLLE observable agreement | `python -m validation.pylle_crosscheck --pylle-python … --julia-bin …` | **FROZEN**, 7/7 HARD checks pass |
| …verify the frozen artifacts without Julia | `pytest tests/test_validation_freeze.py tests/test_pylle_crosscheck_v2.py -q` | `69 passed, 3 skipped` — 15 s |

Two caveats worth reading before quoting any of this:

* **0 ULP is a fixed-hardware claim.** XLA reassociates reductions for the CPU it compiles
  for; on a different runner with identical library versions the same check differs by
  ~6e-19, and GPU agrees to ~1e-12, not 0 ULP. CI therefore runs the loose form
  (`allclose(atol=1e-13)`) by default and `SOLITON_STRICT_ULP=1` is the pinned-environment
  contract.
* **The shipping scheme is first order overall**, by construction: the Strang core is second
  order but the drive kick is non-palindromic. `symmetric_drive=True` measures 2.00 and is
  opt-in, because enabling it changes every committed trajectory.

Before quoting any number, read [`docs/VALIDATION_STATUS.md`](docs/VALIDATION_STATUS.md) —
the configuration-coverage table in particular, because **the validated configuration is not
the production configuration**.

## Reproducing the paper

```bash
python scripts/make_paper_figures.py --check      # verify every input is present   (<1 s)
python scripts/make_paper_figures.py --all        # every figure from committed data (~25 s)
```

One command per figure:

| | Command |
|---|---|
| Fig. 1 verification | `python scripts/make_paper_figures.py --figure fig1_verification` |
| Fig. 2 numerics | `python scripts/make_paper_figures.py --figure fig2_numerics` |
| Fig. 3 pyLLE cross-check | `python scripts/make_paper_figures.py --figure fig3_pylle` |
| Fig. 4 noise budget | `python scripts/make_paper_figures.py --figure fig4_budget` |
| Fig. 5 S_rep | `python scripts/make_paper_figures.py --figure fig5_srep` |
| Fig. 6 runtime scaling | `python scripts/make_paper_figures.py --figure fig6_runtime` |
| Fig. 7 experiment | *skipped by design* — needs unpublished measured data |
| Fig. 8 quiet point | `python scripts/make_paper_figures.py --figure fig8_quietpoint` |
| Budget table | `python analysis/noise_budget.py --seeds 24` (`--quick` for a smoke run) |
| Runtime table | `python benchmarks/runtime_scaling.py` |

`4 built, 3 partial, 1 skipped` is the **expected** result on a clean checkout: heavy panels
are not recomputed by default (`--recompute-heavy` fills them in), and one figure needs
measured data that is not in the repository. A missing panel is always reported, never
silently substituted. Full detail, with expected outputs:
[`docs/REPRODUCING_THE_PAPER.md`](docs/REPRODUCING_THE_PAPER.md).

## Citing

See [`CITATION.cff`](CITATION.cff). The software and the paper are cited separately:

```bibtex
@software{stochastic_lle,
  title  = {stochastic-lle: a validated stochastic Lugiato-Lefever benchmark},
  author = {Wu, Lucas},
  year   = {2026},
  url    = {https://github.com/lucastiger/stochastic-lle},
  version = {1.0.0},
  license = {MIT}
}
```

The DOI is minted by Zenodo on the first archived release; until then there is no DOI to
cite and the badge above is a placeholder rather than a guess. The manuscript
(*A validated stochastic Lugiato-Lefever benchmark for Kerr-soliton microcombs*) is in
preparation — `preferred-citation` in `CITATION.cff` is deliberately incomplete rather than
speculative.

## Documentation

| | |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | install → soliton → one channel on, with the unit and sign conventions |
| [`docs/NOISE_MODELS.md`](docs/NOISE_MODELS.md) | the physics of all eight channels, with equations |
| [`docs/EQUATION_MAP.md`](docs/EQUATION_MAP.md) | generated: paper equation → implementing code → pinning test |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | the three verification tiers and what each proves |
| [`docs/API.md`](docs/API.md) | the callable surface |
| [`docs/REPRODUCING_THE_PAPER.md`](docs/REPRODUCING_THE_PAPER.md) | exact commands, runtimes, expected outputs |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | **read before quoting a number** |
| [`docs/VALIDATION_STATUS.md`](docs/VALIDATION_STATUS.md) | how validated this is, and to what |
| [`docs/PYLLE_STATUS_V2.md`](docs/PYLLE_STATUS_V2.md) | the frozen cross-check in full |
| [`notebooks/`](notebooks/) | `01_quickstart.ipynb`, `02_noise_budget.ipynb` — both run headless in under 15 min |

## Repository layout

```text
stochastic-lle/
├── config/sin_params.yaml     device config + the top-level `noise:` block
├── simulator/                 solver, noise models, colored-noise engine, labeler
├── analysis/                  budget, metrology, reports, soliton access
├── validation/                the verification suite and its frozen artifacts
├── benchmarks/                runtime scaling
├── scripts/                   paper-figure pipeline, artifact manifest
├── notebooks/                 01_quickstart, 02_noise_budget
├── docs/                      the pages listed above
├── tests/                     including the golden trajectories
├── model/ control/ data/      EXPERIMENTAL — not part of the validated benchmark
└── paper/figures/             generated figures + per-figure provenance
```

Contributing, and the bit-identity rule any numerics change must satisfy:
[`CONTRIBUTING.md`](CONTRIBUTING.md). Release history: [`CHANGELOG.md`](CHANGELOG.md).
