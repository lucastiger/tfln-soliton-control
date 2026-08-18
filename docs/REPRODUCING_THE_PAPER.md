# Reproducing the paper

Exact commands, expected runtimes and expected outputs. Every runtime below was measured on
one machine — 4-core x86-64 CPU, Python 3.11.15, jax/jaxlib 0.10.2, numpy 2.4.6, no GPU.
Scale accordingly; the numbers are there to tell you whether a command takes seconds or an
afternoon, not to be a benchmark.

Install the pinned toolchain first if the numbers have to match published ones:

```bash
pip install --require-hashes -r requirements.lock.txt
pip install --no-deps --no-build-isolation -e .
```

---

## 0. The one-command version

Every figure the manuscript uses is built from committed artifacts:

```bash
python scripts/make_paper_figures.py --all
```

**~25 s.** Writes `paper/figures/figN_*.{pdf,png}` plus a `.provenance.json` per figure
recording the input hashes it consumed. Expected on a clean checkout:

```
4 built, 3 partial, 1 skipped, 0 failed   ->  paper/figures
```

That is the correct outcome, not a failure — see §2 for what "partial" and "skipped" mean.

Inspect without building anything:

```bash
python scripts/make_paper_figures.py --list     # figure ids, recompute cost, input counts
python scripts/make_paper_figures.py --check    # verify every input is present and hashed
```

`--check` is **&lt;1 s** and is the fastest way to confirm a checkout is complete.

---

## 1. Verification claims — the numbers in the text

| Claim | Command | Runtime | Expected |
|---|---|---|---|
| All-off bit-identity, 0 ULP | `python -m validation.noise_off_identity --check --strict` | **12 s** | `4 param sets, mode=strict (0 ULP): 0 differences` |
| …as a test, with 3 further failure modes | `SOLITON_STRICT_ULP=1 pytest tests/test_noise_off_identity.py -q` | **53 s** | `12 passed, 4 skipped` |
| Exact CW fixed point | `python -m validation.analytic_cw` | **1 m 52 s** | `reproduces the exact fixed point … to 1.922e-13 at every one of 405 points` |
| Observed order of accuracy | `python -m validation.convergence --report` | **4 m 38 s** | final line `PASS`; 4 gates pass |
| Frozen cross-check artifacts | `pytest tests/test_validation_freeze.py tests/test_pylle_crosscheck_v2.py -q` | **15 s** | `69 passed, 3 skipped` |
| Cross-code, live | `python -m validation.pylle_crosscheck --pylle-python … --julia-bin …` | hours | needs a separate pyLLE + Julia env — see below |

The four skips in the identity suite are the large-grid parameter sets; add `--runslow` to
include them.

`validation.convergence --report` prints two blocks. The gated one is the **fixed** scheme
(`symmetric_drive=True`, `thermal_coupling="strang"`) at orders 2.00 / 1.00 / 2.00 / 3.05;
the ungated comparison block is the **shipping** scheme at 1.00 across the board. Both are
expected — see [`VALIDATION.md`](VALIDATION.md) Tier 2.

### The cross-check needs its own environment

pyLLE pins `numpy<2` and requires a Julia toolchain, so it cannot share this repository's
environment. `validation/` drives it out-of-process:

```bash
python -m venv pylle-env
./pylle-env/bin/pip install "pyLLE"          # + a Julia toolchain; see docs/PYLLE_STATUS_V2.md
python -m validation.pylle_crosscheck_v2 \
    --pylle-python ./pylle-env/bin/python \
    --julia-bin    ./pylle-env/bin/julia
```

The comparison is **FROZEN** at verdict `FAIL (QUALIFIED)` with 7/7 HARD checks passing.
Read [`PYLLE_STATUS_V2.md`](PYLLE_STATUS_V2.md) before citing it in either direction. You do
not need Julia to verify the frozen result — the artifact test above does that from the
hash-pinned files.

---

## 2. Figures, one command each

`--figure` may be repeated. All outputs land in `paper/figures/`.

| Figure | Command | Runtime | Content |
|---|---|---|---|
| 1 | `python scripts/make_paper_figures.py --figure fig1_verification` | 11 s | (a) all-off ULP residual vs goldens (b) analytic-CW residual vs detuning (c) observed order, 3 curves (d) weak order, noise on |
| 2 | `… --figure fig2_numerics` | 9 s | (a) TRN AR(1) burn-in before/after (b) vacuum floor vs `n_tau` (c) Euler vs exponential thermal at dt/τ = 2.1 |
| 3 | `… --figure fig3_pylle` | 1 s | spectrum + waveform overlay with residual sub-panels |
| 4 | `… --figure fig4_budget` | 1 s | grouped bars, 5 observables × channel sets, log axis, error bars |
| 5 | `… --figure fig5_srep` | 1 s | S_rep overlay per channel with the TRN(K–G) + FSR limit |
| 6 | `… --figure fig6_runtime` | 1 s | ns/round-trip/mode vs `n_tau`, CPU + GPU, n log n reference |
| 7 | `… --figure fig7_experiment` | — | **SKIPPED** by design: needs `data/measured/soliton_steps.npz`, which is not in the repository |
| 8 | `… --figure fig8_quietpoint` | 1 s | predicted quiet point from the quiet-point sweep |

### Why "partial" is the expected result

Three figures render with panels missing on a clean checkout, and each says why:

* **fig1**, panels b and c — `validation.analytic_cw.verify()` and
  `validation.convergence.deterministic_study()` are not run by default because they are
  minutes of compute. Add `--recompute-heavy` to fill them in (≈ 45 s for fig1).
* **fig4**, `timing_jitter` — no resolved cell in the committed `--quick` budget run: the
  quick record's Fourier floor sits above the target, so the cell is correctly reported
  `null` rather than interpolated. Rebuild the budget at production seeds to resolve it.
* **fig4**, `step_jitter` — all cells are exactly 0 in the quick run, and a log axis cannot
  render zero.
* **fig6** — the GPU curve is absent without GPU benchmark data.

The pipeline distinguishes *built* / *partial* / *skipped* / *failed* deliberately: a
missing panel is reported, never silently drawn from a fallback.

To render fig1 completely:

```bash
python scripts/make_paper_figures.py --figure fig1_verification --recompute-heavy
```

**~45 s.**

---

## 3. Tables

### The noise budget (Table: per-channel attribution)

```bash
python analysis/noise_budget.py --quick        # smoke run, 2 seeds
python analysis/noise_budget.py --seeds 24     # production
```

`--quick` is minutes; `--seeds 24` is hours — it is 13 channel sets × 5 observables × 24
seeds × two record lengths, and the slow record is 2 × 10⁷ round trips. It checkpoints after
every solver unit and supports `--resume`, which refuses to reuse a store written under
different run settings.

Writes `analysis/results/budget/`:

| File | |
|---|---|
| `budget.json` | every cell, per-seed values, bootstrap CIs, Welch segmentation, record length |
| `budget.npz` | the raw arrays |
| `budget_table.md` | the human-readable table |
| `budget_table.tex` | booktabs, ready to `\input` |
| `budget.provenance.json` | git commit, config digests, environment fingerprint |

The committed `budget.json` is a `--quick` run (2 seeds, `[100, 101]`). Two of its cells are
recomputed live, from `NoiseConfig` through the shipping engine functions, in
`notebooks/02_noise_budget.ipynb` — it reproduces them to `rel.diff = 0.00e+00` in about
15 s.

**Read the record-length column before comparing any two cells.** See
[`LIMITATIONS.md`](LIMITATIONS.md) §5.

### Runtime scaling (Table: cost per round trip per mode)

```bash
python benchmarks/runtime_scaling.py
```

Writes `benchmarks/runtime.json` and `benchmarks/runtime_table.md`, which feed fig6.

---

## 4. Campaigns

The W1–W5 validation campaign and the comparison-report figures are heavier drivers:

```bash
python analysis/noise_validation_campaign.py --help
python analysis/noise_comparison_report.py --help
python analysis/quantum_noise_report.py --help
python analysis/pump_noise_report.py --help
```

Regeneration procedure for the campaign, including which artifacts it overwrites, is in
[`REGENERATE_CAMPAIGN_W1_W5.md`](REGENERATE_CAMPAIGN_W1_W5.md).

> **Note on `analysis/results/noise_comparison_report.json`.** The ≈256σ measured-`D_int`
> curvature significance reported in its `fig6_linewidth_dwrecoil` block is a *within-record*
> Welch-segment bootstrap — how well that one record's parabola is resolved above its own
> noise. It is **not** a run-to-run reproducibility bound. The physical discriminator between
> genuine dispersive-wave recoil and a numerical artifact is the flat Taylor-D₂ control
> (a₂ = 0, S_rep ratio ≥ 10⁶×), not the σ magnitude.

---

## 5. Provenance

Every figure writes a `.provenance.json` next to it recording the SHA-256 of each input it
consumed. The solver itself can stamp a run:

```python
solution = solve_lle_ssfm_jax(..., provenance=True)
solution["provenance"]     # git commit, config digests, seed, env fingerprint, arXiv ref
```

It is off by default so the returned dict stays legacy-exact key-for-key: the golden
whole-dict hash in `tests/test_regression_figures.py` assumes the legacy key set, and the
stamp carries a wall-clock timestamp, so it is not reproducible by construction. It adds no
RNG calls and no arithmetic — the physics is bit-identical either way.

Artifact inventory and hashes:

```bash
python scripts/artifact_manifest.py
```

---

## 6. If a number does not reproduce

Work down this list before concluding the physics changed:

1. **Check the toolchain.** `pytest` prints `python= jax= jaxlib= numpy= scipy= h5py=` in its
   header. The goldens were produced with jax/jaxlib 0.10.2, numpy 2.4.6, Python 3.11.15.
2. **Check the hardware.** 0-ULP identity is a *fixed-hardware* claim. XLA reassociates
   reductions for the CPU it compiles for; a different runner can differ at 1e-19 while
   being entirely correct. GPU agrees to ~1e-12, not 0 ULP.
3. **Check the config digest.** `NoiseConfig.sha256()` and the golden
   `.provenance.json` sidecars record what the run was configured with.
   `tests/test_noise_off_identity.py::test_golden_provenance_matches_current_config` exists
   precisely so a config edit reads as "the config changed" rather than as a mystery ULP
   failure.
4. **Check the record length** for anything spectral. See [`LIMITATIONS.md`](LIMITATIONS.md) §5.
5. **Check the ordering convention** for anything compared to measurement. See
   [`LIMITATIONS.md`](LIMITATIONS.md) §4.
