# Validation status

**How validated is this simulator, and to what?**

This is the top-level answer. It is written for someone who has never seen this
repository. Read it before quoting any number this simulator produced.

* **Status of the pyLLE cross-check: FROZEN as of 2026-08-17.** Its verdict is
  **FAIL (QUALIFIED)** and freezing does not change that — see §5. The freeze is
  a judgement that the qualification is adequately documented, not a claim that
  the comparison passed.
* **Artifacts:** every file under `validation/results/` is frozen and
  hash-pinned; see `validation/results/FROZEN_MANIFEST.md`.
* **Versions in play:** pyLLE 4.1.2, Julia 1.10.4 (conda-forge, FFTW 1.10.0 /
  HDF5 0.17.3), Python 3.11.15 both sides, NumPy 2.4.6 (ours) / 1.26.4 (pyLLE),
  SciPy 1.17.1, JAX 0.10.2.

---

## 1. What is validated, against what — in order of strength

Each layer below is a *different kind* of claim. They are not interchangeable,
and the strongest one covers the narrowest question.

### 1. `validation/analytic_cw.py` — against exact mathematics

The strongest check in the repository. The CW steady state has a closed form (a
cubic root, and the discrete map's fixed point), so the solver can be compared
against an *answer*, not against another approximation. Agreement is **~1e-14**,
gated at **1e-12**.

*Validates:* the CW fixed point of the implemented map — drive, loss, detuning
and Kerr phase, and their relative normalizations. It does **not** exercise
dispersion beyond a constant, and it does not exercise the soliton.

### 2. `validation/mms.py` and `validation/convergence.py` — observed order

Manufactured solutions and step halving, single-code, no external reference.
These establish that the discretization is what it claims to be.

**The production path is measured at order 1.0000** (errors 1.60e-02 …
9.98e-04); with `symmetric_drive=True` it is order 2.0000 (4.73e-05 …
1.85e-07). The first-order result is not a defect being hidden — it is a
measured property of the shipped composition, which is not palindromic. See
`validation/convergence.py`.

*Validates:* the order of accuracy of the scheme as it actually runs. It says
nothing about whether the *physics* is right, only that the numerics converge at
the stated rate.

### 3. `validation/convergence_lle.py` — discretization uncertainty

Quantifies how far the solver is from its own converged limit at a specific,
deliberately difficult operating point: **δω = 30 κ, n_tau = 6601, every mask
off, n_substeps 1…16**, plus a grid ladder at `mu_half` 3300/4400/5500.
Richardson-extrapolated limits with GCI bands. Observed orders: `P_peak` 2.31,
`mu_DW` 2.48, `dw_power_dbc` 1.96, `S3_modes` 1.56, `comb_frac` 1.10;
`U_mean_w` is **non-monotone and has no limit**.

*Validates:* the size of the numerical uncertainty on each observable at that
configuration. This is the source of every GATED tolerance downstream — no
tolerance in the suite is chosen, all are derived from this study and from
pyLLE's own equivalent.

### 4. `validation/pylle_crosscheck_v2.py` — conventions and envelope containment

An independent implementation (pyLLE 4.1.2, Julia backend) of the same PDE.
**7/7 HARD checks pass**: parameter round trip 1.23e-16, dispersion refit
2.32e-12, mirror applied, pump reference exact to 0 Hz, detuning endpoints
identical, thermo-optic off, seeds bit-identical. Both codes' extrapolated
`mu_DW` limits agree to **4.5e-4 modes out of 3075** on a strongly asymmetric
measured dispersion.

*Validates:* **conventions and bookkeeping** — detuning sign, dispersion
direction and mirror, factors of 2π, mode-index origin, which resonance the pump
sits on. These are the errors that silently produce plausible-looking solitons,
and no single-code test can catch them. It also answers whether the two codes,
each refined along its own ladder, converge to the same limit.

*Does not validate:* that either code is right. Two codes agree only to the
level at which they discretize the same PDE, and agreement cannot distinguish
"both right" from "both wrong in the same way". Full detail and every
qualification: `docs/PYLLE_STATUS_V2.md`.

---

## 2. Configuration coverage — the most important table here

Everything in §1 validates **one** configuration. Production uses others.

| context | δω | `n_tau` | 2/3 dealias | edge absorber | `n_substeps` | uncertainty budget? |
|---|---|---|---|---|---|---|
| **validation** (all four layers above) | **30 κ** | **6601** | off | off | 1…16 | **YES** |
| `analysis/dks_access.PRODUCTION_NUMERICS` | 10 κ (`OPERATING_DW_KAPPA`) | 8192 (`N_TAU`) | **ON** | **ON** | 4 | **NO** |
| `scripts/regenerate_dks_artifacts.py` settle | 10 κ | 16384 (`SETTLE_N_TAU`) | **ON** | **ON** | 4 | **NO** |
| `scripts/regenerate_dks_artifacts.py` scan | 10 κ | 8192 (`SCAN_N_TAU`) | **ON** | **ON** | 4 | **NO** |
| `data/dataset_generator.py` | varies (swept) | varies (default 512) | off | off | 1 | **NO** |

(`dispersion_validity_mask` is `False` everywhere, including in
`PRODUCTION_NUMERICS`, deliberately: the exact linear exponential is valid at any
phase and the mask amputates real soliton-tail and dispersive-wave spectrum.)

**The validated configuration is not the production configuration.** Turning the
masks off was the *correct* choice for the cross-code comparison — pyLLE has
neither a 2/3 dealias nor an edge absorber, so leaving ours on would have
compared two different problems and there would have been no way to attribute the
difference. It was correct for the convergence study too, for the same reason:
a mask changes what the scheme is, and the order of accuracy of the shipped
scheme is what that study set out to measure. But the consequence is that **the
quantities the production artifacts quote have no uncertainty budget**, and one
specific interaction is worrying enough to name here:

> At `n_tau = 8192` the 2/3 dealias zeroes every mode with `|mu| > 2730`. The
> dispersive-wave phase-matching crossings for this device sit at
> `|mu| ≈ 3038` and `3239`. **Both DWs are inside the region the production
> configuration zeroes.** `SETTLE_N_TAU = 16384` moves the cutoff to 5461 and
> does resolve them, which is why it exists — but the `SCAN_N_TAU = 8192` runs
> do not, and no convergence study has been run at production settings to say
> what that costs.

Closing this gap is the **next validation priority** (§6).

---

## 3. Stopping rule — is the pyLLE work done?

pyLLE validation is complete *for this project* when all eight hold. Status:

| # | criterion | met? | evidence |
|---|---|---|---|
| 1 | all seven HARD checks pass | **YES** | `pylle_crosscheck_v2.json` → `checks[]`, cids H1–H7 all `PASS`; `docs/PYLLE_STATUS_V2.md` §6 table |
| 2 | `mu_DW` and `dw_power_dbc` CONTAINED, with the grid-truncation caveat recorded | **YES** | `pylle_crosscheck_v2.json` → `containment`; caveat in `docs/PYLLE_STATUS_V2.md` §4 "Caveat on the two CONTAINED rows" |
| 3 | bit-reproducible, argv and digest recorded | **YES** | `provenance.argv`, `provenance.defaults_reproduce_this_run: true`, `numerical_digest` — two runs with identical argv gave identical digest, figure sha256 and npz sha256 |
| 4 | every falsified v1 interpretation documented with evidence, original text preserved | **YES** | `docs/PYLLE_STATUS_V2.md` §5 (a)–(i); `docs/PYLLE_STATUS.md` is unmodified apart from one prepended supersede line; `validation/results/v1/MANIFEST.md` |
| 5 | `comb_frac`'s grid-truncation hypothesis resolved from existing data | **YES** | `docs/PYLLE_STATUS_V2.md` §4: grid contribution 4.21e-6 vs a 1.342e-3 limit gap (319×), from `convergence_lle_dw30k.json` → `grid_levels`. **Falsified.** |
| 6 | the existence criterion carries a measured ours-side discretization term | **YES** | `existence_convergence_ours.json` → `edge_convergence`, `g7_recomputed`; outcome **H-C**, drift 0.1875 κ |
| 7 | remaining discrepancies documented with named falsification tests | **YES** | `docs/PYLLE_STATUS_V2.md` §8, one row per conclusion |
| 8 | no unresolved item materially affects a current claim | **YES, with one qualification** | §4 below gives the impact line for every open item. The qualification: this holds for claims about *conventions*. Claims about absolute DW position or power at production settings are **not** supported by this work — see §2. |

**All eight are met.** The pyLLE cross-check is frozen.

Freezing is not a pass. The overall verdict in `pylle_crosscheck_v2.json` is
`FAIL` with `overall_qualified: true`, and it stays that way. The judgement being
recorded here is that the *qualification* is adequate: every failing and
unresolved item is measured, attributed, and carries a falsification test.

---

## 4. Known open items, and what each one costs

Every item carries an explicit **impact on current claims**. Where the impact is
none, it says so.

### 4.1 Existence edges (G7) — both FAIL

Lower edge: ours 6.0781 κ (at `n_substeps = 4`) against pyLLE 6.1719 κ. Upper
edge: ours 37.9688 κ against pyLLE 37.0312 κ, gap 0.9375 κ.

Prompt E measured the ours-side discretization term: the lower edge drifts
**0.1875 κ** over `n_substeps` 1→4 — moving *toward* pyLLE, closing two thirds
of the original gap and bringing the brackets into (knife-edge) overlap. The
upper edge does not move at all. Pre-committed outcome **H-C** (intermediate:
0.05 κ < drift < 0.28 κ).

Two hypotheses were falsified along the way: settle time (4× and 2× holds move
no classification) and the threshold classifier (ALIVE contrast ≥ 19.56, DEAD
≤ 1.70, threshold 5 — an empty band spanning 11.5×, with no evaluation near it).

**Impact on current claims: none.** Nothing in the repository quotes a
single-soliton existence boundary from cross-code agreement. `OPERATING_DW_KAPPA
= 10 κ` sits far inside both codes' existence ranges and is chosen from the Hopf
boundary at ~9.3–9.4 κ, which is measured single-code, not from this comparison.

**Unresolved:** the upper-edge gap. pyLLE's own edge was never refined (that
needs Julia and was out of scope), so the recomputed band is asymmetric.

### 4.2 `comb_frac` SEPARATED at 1.13× the band

Ours 0.853371 ± 2.28e-4, pyLLE 0.854518 ± 5.47e-4; gap 1.342e-3 against a
combined band of 1.184e-3. Grid truncation is **falsified** as the cause
(4.21e-6 contribution, 319× too small). Two candidates remain: the nonlinear
step rule (pyLLE endpoint-trapezoidal, ours midpoint — leading), and both
ladders being non-asymptotic (observed orders 1.10 and 0.84, well below 2).

**Impact on current claims: none.** No claim in the repository rests on the
absolute comb energy fraction agreeing with an independent code. It is a 0.13%
disagreement in a quantity used here only as a convergence diagnostic.

### 4.3 Three GATED observables UNDERIVED / INDETERMINATE

`U_mean_w` (our ladder non-monotone), `P_peak_w` and `S3_modes` (pyLLE's tight
ladder non-monotone). Their tolerances could not be derived, so they were
**demoted to diagnostic and qualify the run** rather than being given a guessed
number.

**Impact on current claims: none, and this is the system working as designed.**
An UNDERIVED criterion asserts nothing; it does not silently pass. The
`overall_qualified: true` flag exists precisely so these cannot be mistaken for
clean passes.

### 4.4 `mu_DW` / `dw_power_dbc` CONTAINED on grid-truncated values

Both containment verdicts are at `mu_half = 3300`, where the comb is not
spectrally contained (grid edge −44 dBc, blue DW at μ ≈ +3239 only 61 modes
inside the boundary). Widening the grid moves `mu_DW` by **3.793 modes** and
`dw_power_dbc` by **7.673 dB**, both converged by `mu_half = 4400`.

**Impact on current claims: partial, and it matters.** The *conventions*
conclusion is unaffected — a mirror, sign or mode-index error would show up at
the scale of thousands of modes, not 4.5e-4, so the agreement remains strong
evidence that neither code has one. But these verdicts are **not** validation of
the physical DW position or amplitude. **Any artifact quoting an absolute DW
position or DW power must be checked against `mu_half`/`n_tau`, not against this
cross-check.** This is the same gap §2 names, seen from the other side.

### 4.5 pyLLE is non-stationary at the 1% level at 30 κ

Under a constant hold at the main operating point, pyLLE's peak power varies
1.2% over the final 10 round trips while ours varies 0.27%.

**Impact on current claims: none directly, but it bounds the comparison.** The
residual cross-code disagreements are themselves of order 1%, so a snapshot
comparison against pyLLE at 30 κ is comparing to a state still moving by about as
much as the quantity being compared. This is a reason not to chase the remaining
~1% energy-observable gap through more cross-code work — it is at the noise floor
of the reference.

### 4.6 The v1 → v2 record

The v1 run's argv was never recorded and its committed defaults do not reproduce
it; nine of its interpretations are corrected in `docs/PYLLE_STATUS_V2.md` §5.

**Impact on current claims: none.** v1's *conventions* work survived intact and
is what v2 builds on; only its interpretation of the residual disagreements was
wrong. The artifacts are preserved unmodified as the historical record.

---

## 5. When to revisit pyLLE — re-run triggers

Re-run `validation/pylle_crosscheck_v2.py` when **any** of these fires. Not
otherwise; the point of the freeze is to stop open-ended churn.

1. **A convention changes** — dispersion sign or mirror, mode-index origin,
   D1/FSR reconciliation, pump referencing, field normalization, or detuning
   sign. This is what the cross-check is *for*; it is the only test in the
   repository that can catch an error in any of them.
2. **The linear or nonlinear step composition changes** — `_fine_step`,
   `symmetric_drive`, or the splitting order. The HARD checks assume both codes
   integrate the same PDE the same way.
3. **`load_dint_grid` changes.** Both codes are fed one dispersion array derived
   through it; a change there changes the problem, not just the solver.
4. **A new device or dispersion profile is introduced.** The v2 report notes
   explicitly that agreement on *one* asymmetric profile could in principle be a
   coincidence of that profile.
5. **A manuscript claim comes to depend on absolute agreement with an
   independent code.** Today none does; if one did, the qualifications in §4
   would need to be closed rather than merely recorded.

Both `simulator/lle_solver.py::_fine_step` and `::load_dint_grid` carry a comment
block pointing here, so triggers 2 and 3 are visible to whoever next edits them.

**The cross-check remains runnable**, and freezing must never be allowed to mean
"unrunnable". The documented command, with the environment recipe in
`docs/PYLLE_STATUS_V2.md` §3:

```bash
python -m validation.pylle_crosscheck_v2 \
    --pylle-python ./pylle-env/bin/python \
    --julia-bin    ./pylle-env/bin/julia
```

Every default is the value the frozen run used, so that bare command reproduces
the frozen artifacts bit-for-bit.

---

## 6. Next validation priority, and why

**A convergence study at the production configuration.** Scaffolded, not run, at
`validation/production_config_convergence.py`.

The reason is §2: four layers of validation all sit at δω = 30 κ, `n_tau = 6601`,
masks off — and no production artifact uses that configuration. The two questions
that study must answer:

1. **Does the edge absorber measurably alter DW power at production settings,
   and by how much?** It is on in production and absent from every validated
   configuration, and it acts precisely in the wings where the DWs live.
2. **Are the artifacts that quote DW physics converged in `n_tau`?** At
   `n_tau = 8192` the 2/3 dealias zeroes `|mu| > 2730` while the DW crossings sit
   at `|mu| ≈ 3038` and `3239`. On its face the scan configuration cannot
   represent the dispersive waves at all; this needs measuring rather than
   assuming.

Until that study runs, the honest scope statement is: **the solver's conventions
and its convergence behaviour are validated; the production configuration's
dispersive-wave output is not.**

---

## 7. Where the documents live

| document | what it is |
|---|---|
| `docs/VALIDATION_STATUS.md` | this file — the top-level answer |
| `docs/PYLLE_STATUS_V2.md` | the cross-check in full: nine convention findings, ladder tables, containment, nine corrections to v1, open items, falsification tests |
| `docs/PYLLE_STATUS.md` | the v1 record. Superseded, preserved verbatim |
| `docs/VALIDATION_METHODOLOGY.md` | how the acceptance criteria and tolerances are derived, and why none was chosen |
| `docs/CONVERGENCE_LLE.md` | the discretization-uncertainty study in full |
| `validation/results/FROZEN_MANIFEST.md` | every artifact, its sha256, and what it is evidence of |
