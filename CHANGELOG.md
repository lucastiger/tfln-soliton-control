# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Numerical output is treated as part of the public interface. A release that
moves a trajectory bit-for-bit says so explicitly, under **Changed**, with the
regenerated fixtures named — see [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Changed

- `test_all_off_bit_identical_to_golden[s1024_near]` joins
  `conftest.HARDWARE_LOCKED_NODE_IDS`, so the weekly `slow` job skips it under
  `--skip-hardware-locked` like the other byte comparisons. The first scheduled
  run of that job — the first time the four `--runslow` cases had ever run on a
  shared runner — failed on this one case, and the reason is worth recording
  because it **refines the reproducibility claim rather than restating it**.

  The other hardware-locked comparisons miss by ~1e-19: a flat rounding
  difference, ~6 orders below the repo's ATOL of 1e-13. This one missed by
  `max_abs_diff = 1.2e-06` with `max_rel_diff = 1.06` on `E_snapshots` — fully
  decorrelated. `s1024_near` is the Δω = 2κ set, and labelling its own golden
  with `simulator.state_labeler.label_trajectory` puts it in CW for 40
  snapshots, modulation instability for 60, then **chaotic** for the last 100;
  the other three sets are CW throughout. Perturbing `delta_omega` by one ULP
  and re-running confirms the mechanism directly: max |δE| stays at ~1e-20 for
  all three CW sets and grows 11.6 decades, monotonically, to 1.3e-08 for
  `s1024_near` (λ = 1.29e-2 per round trip, an e-folding every 77, crossing
  ATOL at snapshot 85).

  So bit-identity for this case is hardware-locked in a stronger sense than the
  rest: **no tolerance both admits a different CPU and still detects a
  regression**, because the discrepancy saturates at the attractor's own scale.
  It is therefore skipped by node ID rather than by widening ATOL — widening
  would have had to reach order-1 relative error and would have asserted
  nothing. Nothing else changed: the other three parametrisations still compare
  against the goldens at ATOL in `fast`, `identity` and `slow`; `s1024_near`
  keeps its three same-process byte-identity tests in every job (those compare
  two runs on one machine, so chaos cannot affect them); and on the reference
  machine a plain `pytest --runslow` still asserts all four, at 0 ULP under
  `SOLITON_STRICT_ULP=1`. See
  [CONTRIBUTING.md](CONTRIBUTING.md#bit-identity-is-scoped-to-hardware-not-just-to-versions).

- The `fast` CI job now passes `--skip-hardware-locked`, and the `identity` job
  runs its 0-ULP comparison as an informational step with the tolerance
  comparison as the blocking one. Both follow from a measurement made by the
  first CI run: **the golden trajectories are reproducible bit-for-bit only on
  the hardware that produced them.** On a runner with the goldens' exact
  toolchain (`version_mismatch=None`) the comparison differed by
  `max_abs_diff = 6.2e-19` / `max_rel_diff = 1.8e-13`, because XLA reassociates
  reductions to fit the CPU it compiles for. That is ~6 orders of magnitude
  below the repo's own ATOL of 1e-13, so no physics claim is affected — but the
  *byte-level* claim is now known to be scoped to source × toolchain × hardware,
  where previously only the first two were documented. See
  [CONTRIBUTING.md](CONTRIBUTING.md#bit-identity-is-scoped-to-hardware-not-just-to-versions).

  `--skip-hardware-locked` is **off by default**: an ordinary `pytest` run on
  the reference machine still asserts every byte comparison.

- **The repository was renamed from `soliton-control` to `stochastic-lle`.**
  Development, issue tracking, releases and CI continue on the personal
  upstream, [`lucastiger/stochastic-lle`][upstream]; the install instructions in
  `README.md` and `docs/QUICKSTART.md` now clone the lab organization's
  public-facing fork, [`Mengjie-Yu-Group/stochastic-lle`][org-fork], which is
  maintained as a fast-forward mirror of that upstream.

  Three local identifiers changed with it and are not automatic on an existing
  checkout: the conda environment name in `environment.yml`, the documented
  Docker image tag, and the Python distribution name in `pyproject.toml` are all
  now `stochastic-lle` rather than `soliton-control`. Recreate the environment
  (`conda env create -f environment.yml`) or rebuild the image under the new
  tag; an environment or image created under the old name keeps working and is
  simply no longer what the documentation names.

  The distribution rename needs one further step. `simulator.__version__` is
  read from the installed distribution's metadata rather than duplicated in the
  source, so a checkout installed under the old name reports `"0.0.0+unknown"`
  until it is reinstalled with `pip install -e .`. The optional extra is now
  `pip install stochastic-lle[ml]`.

  Frozen validation artifacts, vendored `third_party/` sources and the
  historical provenance records under `analysis/results/` keep the old name on
  purpose: they are hash-pinned evidence of runs that happened under it, and
  rewriting them would be falsifying a record rather than updating a reference.

### Added

- `conftest.HARDWARE_LOCKED_NODE_IDS` — the seven test functions (ten node IDs)
  that compare solver output byte-for-byte against a committed artifact, with
  the evidence for why a shared runner cannot satisfy them. Guarded by
  `tests/test_packaging_metadata.py`, which checks each entry against what
  pytest actually collects, so neither a rename nor a dropped parameter can
  silently empty the list.

## [1.0.0] — 2026-08-17

First release under the project's current identity. `soliton-control` began as a
scaffold for closed-loop control of solitons in TFLN microresonators, with a
physics-informed RNN observer as the centrepiece. It is now a **validated
stochastic Lugiato–Lefever benchmark**: the object of study is the solver and the
evidence that its noise attribution is trustworthy, and the learning stack is a
downstream consumer rather than the point.

The pivot was not a rewrite. Nothing was deleted; what changed is which part
carries the claims, and how much of the repository exists to falsify them.

### Added

- **Stochastic channels**, each traceable to an equation in Herr, Tikan &
  Kippenberg, [arXiv:2604.05897v1](https://arxiv.org/abs/2604.05897), and each
  opt-in behind its own flag:
  quantum vacuum (Langevin drive √κ·ξ̂_μ, Eq. 126, truncated-Wigner ½ photon per
  mode); thermorefractive noise with selectable PSD model (`single_pole`,
  `kondratiev_gorodetsky` per Eq. 130, or a measured `csv`); the thermal
  expansion pull; pyroelectric–electro-optic and thermal-carrier (TCCR) shifts
  driven by the *same* δT realization as TRN; pump frequency noise
  (S_δν = h₀ + h₋₁/f) and pump RIN; and FSR/repetition-rate noise.
- **`simulator/colored_noise.py`** — synthesis of any channel from a one-sided
  target PSD, host-side in float64, seeded deterministically from JAX keys.
- **Verification suite**, in rough order of strength: exact-CW steady state to
  ~1e-14 (`validation/analytic_cw.py`); measured order of accuracy against
  manufactured solutions (`validation/mms.py`, which established that the
  shipping scheme is first order); a discretization-uncertainty harness at the
  dispersive-wave operating point; and a pyLLE cross-check, vendored down to its
  Julia kernel so pyLLE's own convergence could be measured rather than assumed.
- **Noise-off bit-identity goldens** (`tests/data/golden/`, hash-pinned with
  provenance sidecars) and `tests/test_noise_off_identity.py`, which compares at
  0 ULP under `SOLITON_STRICT_ULP=1` and at 1e-13 otherwise.
- **Noise budget engine** (`analysis/noise_budget.py`) — one-at-a-time,
  leave-one-out and interaction-residual attribution across thirteen channel
  sets and five observables, over a seed list shared by every set. Common random
  numbers make the differences exact rather than sampling-limited.
- **Noise metrology** (`analysis/noise_metrology.py`) — per-line frequency-noise
  PSDs, the elastic-tape decomposition with its fix point, β-separation-line
  effective linewidths, and timing jitter.
- **Machine-checkable equation map** (`simulator/equation_map.py`,
  `docs/EQUATION_MAP.md`) tying paper equations to the code that implements them.
- **Provenance stamping** (`simulator/provenance.py`) on solver output and every
  generated artifact.
- **Validation freeze** (`validation/results/FROZEN_MANIFEST.md` and
  `tests/test_validation_freeze.py`): artifacts are hash-pinned, the expected
  artifact set is pinned, and the test-suite split is checked in both directions
  so nothing can be quietly moved behind a marker.
- **Runtime and memory scaling benchmark** (`benchmarks/runtime_scaling.py`).
- **One-command manuscript figure pipeline** (`scripts/make_paper_figures.py`).
- **Packaging and reproducibility** (this release's final piece): `pyproject.toml`,
  a hash-pinned `requirements.lock.txt`, `Dockerfile`, `environment.yml`,
  `CITATION.cff`, `codemeta.json`, `.zenodo.json`, and a four-job GitHub Actions
  workflow — lint, fast (3.10/3.11/3.12), a pinned-toolchain 0-ULP identity job,
  and a weekly slow suite.

### Changed

- **`torch` and `einops` moved out of the base dependency set** into the `ml`
  extra. Running an LLE should not require a ~1 GB deep-learning framework; the
  PI-RNN is not on the path from a config file to a benchmark number. Install
  with `pip install -e ".[ml]"` for the model and dataloader stack.
  `requirements.txt` was narrowed to match, and `sympy` moved to the `dev` extra
  (it is lazily imported by `validation/mms.py`).

**No numerical output changed in this release.** The packaging work is additive
by construction: every golden trajectory in `tests/data/golden/` still compares
at 0 ULP under the pinned toolchain, and every default is unchanged.

### Fixed

- AR(1) burn-in transient in the colored-noise synthesis, behind a flag, with
  the residual bias quantified.
- Swapped `LossScaler` wiring between training phases 1 and 2.
- The state labeler's vacuum floor, normalized against `n_tau`.

### Known limitations

Read [`docs/VALIDATION_STATUS.md`](docs/VALIDATION_STATUS.md) before quoting any
number this simulator produced. In particular, as of this release:

- the pyLLE cross-check is frozen at **v2 (FAIL, qualified)** with a recorded
  stopping rule — see [`docs/PYLLE_STATUS_V2.md`](docs/PYLLE_STATUS_V2.md);
- the shipping scheme is **first order**, as measured, not second;
- the **validated configuration is not the production configuration**; the
  coverage table in `docs/VALIDATION_STATUS.md` states the gap;
- the DW-recoil linewidth significance is a within-record estimate, not a
  run-to-run reproducibility bound.

[Unreleased]: https://github.com/lucastiger/stochastic-lle/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/lucastiger/stochastic-lle/releases/tag/v1.0.0
[upstream]: https://github.com/lucastiger/stochastic-lle
[org-fork]: https://github.com/Mengjie-Yu-Group/stochastic-lle
