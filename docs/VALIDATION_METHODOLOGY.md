# Validation methodology

How this repository decides whether a numerical result is acceptable, and why
each criterion is shaped the way it is.

The repository runs three kinds of check, and they are deliberately **not** held
to the same standard, because they are not the same kind of claim:

| kind | example | standard | rationale |
|---|---|---|---|
| mathematical verification | `validation/analytic_cw.py` | **1e-12**, unchanged | compares against exact mathematics (a cubic root, a discrete map's fixed point). Exactness is reachable, so it is demanded. |
| order of accuracy | `validation/mms.py`, `validation/convergence.py` | observed order in a stated window | compares the scheme against a manufactured solution. |
| cross-code agreement | `validation/criteria.py` (this document) | tolerance **derived** from measured convergence | two independent discretizations can only agree to the level at which they discretize the same PDE. |

**Nothing in this document loosens the first two.** The 1e-12 gate in
`analytic_cw.py` and the order-of-accuracy checks stay strict and untouched.
What follows applies only to cross-code agreement, where a fixed hand-picked
tolerance is not defensible.

---

## 1. Criterion classes

`validation/criteria.py` classifies every criterion as **data**, in the
`CRITERIA` table — not as a code path:

* **HARD** — convention, bookkeeping, exact algebraic identities. A failure
  invalidates the comparison outright: the two codes were not solving the same
  problem, so any agreement or disagreement downstream is meaningless.
* **GATED** — quantitative agreement on a well-conditioned functional. The
  tolerance is derived from both codes' measured numerical uncertainty. A
  failure is a real failure.
* **DIAGNOSTIC** — reported with a value and a conditioning number, and
  *structurally* unable to affect the verdict (the schema permits a DIAGNOSTIC
  criterion to carry only the `DIAGNOSTIC` verdict).

`overall = PASS` iff no HARD or GATED check is `FAIL` or `NOT_MEASURED`.
`NOT_MEASURED` on a HARD or GATED check is a **failure**, never a silent pass.

---

## 2. The criteria, and the defect each one replaces

| criterion | class | quantity measured | why the previous criterion was defective | measured evidence of the defect | new definition | tolerance source |
|---|---|---|---|---|---|---|
| H1 `parameter_round_trip_max_rel` | HARD | ours→pyLLE→ours round trip of every scalar | (unchanged, was already correct) | round trip closes to 1.2e-16 | ≤ 1e-12 | fixed threshold — exact algebraic inversion |
| H2 `dispersion_refit_max_rel` | HARD | pyLLE's spline-refit `D_int` vs ours after un-mirroring | (unchanged) | 2.3e-12 measured | ≤ 1e-6 | fixed threshold |
| H3 `dispersion_mirror_applied` | HARD | `D_int` handed to pyLLE is mirrored μ→−μ | not checked at all in v1 | omitting it put our DW at −3082 against pyLLE's +3078 — the same wave, reflected | boolean, must be true | n/a |
| H4 `pump_mode_reference_matches` | HARD | `\|f_pmp_used − csv_mu0_resonance\|` | not checked in v1 | the nominal `c/λ` is **7.11 FSR** from the CSV μ=0 resonance our `D_int` is referenced to | must be exactly 0 Hz | n/a |
| H5 `detuning_endpoint_match` | HARD | `\|δω_ours − (−δω_pylle)\|/\|δω_ours\|` | **computed by the worker and discarded by the orchestrator in v1** | upstream pyLLE returned the field at round trip 4976 of 5000 → 29.9328 κ vs our 30.0000 κ. The two codes were compared at *different detunings*. | ≤ 1e-12 | fixed threshold |
| H6 `delta_omega_eff_equals_programmed` | HARD | thermo-optic shift is off | (unchanged) | 0 measured | ≤ 1e-6 | fixed threshold |
| H7 `seed_arrays_identical` | HARD | sha256 of each code's seed after the conjugate map | not checked in v1 | — | boolean | n/a |
| G1 `intracavity_power_U_mean` | GATED | `Σ\|E_j\|²/N/t_r` (Parseval-exact) | not compared in v1 at all | translation- and sampling-invariant; codes agreed to 0.39% on the committed fields | relative | derived |
| G2 `comb_energy_fraction` | GATED | `(Σ\|E_μ\|² − \|E_0\|²)/Σ\|E_μ\|²` | not compared in v1 | agreed to 0.55% on the committed fields | relative | derived |
| G3 `band_limited_peak_power` | GATED | peak on a 32× zero-padded FFT | v1 used raw `max\|E_j\|²` on a grid where the soliton FWHM is 4.34 samples | sampling-phase bound **4.02%**; measured sub-sample offsets −0.281 and +0.094 account for ≈1.3 of the reported 4.23 pp | relative, on the band-limited peak | derived |
| G4 `subbin_3db_span` | GATED | 3 dB span by dB-linear level crossing | v1 applied a 2% relative tolerance to an **integer** extent | quantum is `1/span`: at span 443 that is 0.226%, so the v1 "0.23% pass" was *exactly one quantum* — the smallest non-zero value expressible. At span 64 the same quantum is 1.6% and identical quality fails. | relative, on the sub-bin span | derived |
| G5 `dw_centroid_fixed_band` | GATED | pedestal-subtracted power centroid over a fixed band | v1 demanded an **exact integer match** — a discontinuous criterion on a continuous quantity | the measured disagreement was **0.65 modes out of 3075 (0.02%)** and failed only because the values straddled `x.5`. Our own converged centroid is −3075.25, and the apparent "convergence toward pyLLE's −3074" at n=2,4 **reverses** at n=8,16. | **absolute**, in modes: `\|Δμ\| ≤ max(0.05, K·√(U_o²+U_p²))` | derived |
| G6 `dw_band_power_dbc` | GATED | DW band power relative to carrier | **measured by v1 and never compared** | moves **7.1 dB** across our own refinement ladder (−29.79 → −22.65 dBc) | relative | derived |
| G7 `existence_edges` | GATED | single-soliton existence bracket | v1 compared `br[1]`, the tightest *surviving* point, on both sides — a biased estimator | reported "5.94 vs 6.31 κ, 5.94%" from brackets `[5.75, 5.9375]` and `[6.125, 6.3125]` | brackets must **overlap**, or midpoints agree within `K·√(hw_o²+hw_p²+U_o²+U_p²)` | bracket half-widths |
| D1 `comb_line_count_60dbc` | DIAGNOSTIC | count ≥ −60 dBc, `dN60/ddB`, core/mid/edge | v1 **gated** on it at 2% | conditioning **85–121 lines per dB**; the core (\|μ\| ≤ 1500) is **identical** between codes (3001 = 3001) at every refinement level — the entire 8.83% disagreement lives in the wings | reported with its conditioning | never gates |
| D2 `integer_3db_span` | DIAGNOSTIC | legacy integer extent + quantum | superseded by G4 | quantum 0.226% at span 443 | reported | never gates |
| D3 `integer_dw_argmax` | DIAGNOSTIC | raw argmax on each side | superseded by G5 | — | reported | never gates |
| D4 `raw_peak_power` | DIAGNOSTIC | legacy `max\|E_j\|²` + sub-sample offset | superseded by G3 | offsets −0.281 / +0.094 samples | reported | never gates |
| D5 `spectral_residual_by_band` | DIAGNOSTIC | median \|dB\| residual per μ band | v1 reported a single global number | core 0.09 dB vs wings ~7 dB — a single number hid where the disagreement lives | 9 bands | never gates |
| D6 `spectral_edge_dbc` | DIAGNOSTIC | spectrum at μ = ±μ_half | not checked in v1 | committed run sits at **−52.2 dBc**: the comb is not spectrally contained and is aliasing | reported, warn above −100 dBc | never gates |
| D7 `dint_phase_budget` | DIAGNOSTIC | modes with `\|D_int\|·t_r > π` | not checked in v1 | **439 modes**, first at \|μ\| ≥ 2021 | reported | never gates |

---

## 3. How GATED tolerances are derived

```
tol(obs) = clip( K · sqrt( U_ours(obs)² + U_pylle(obs)² ), floor(obs), 0.25 )
```

* `K = COVERAGE_FACTOR = 2.0` — a two-sigma-style coverage factor, defined once.
* `U_ours` — from `validation/results/convergence_lle_dw30k.json`, at the
  discretization **actually used** by the comparison.
* `U_pylle` — from `validation/results/pylle_refinement_dw30k.json`.
* floors: 1e-6 (G1, G2), 1e-4 (G3, G4, G6), 0.05 modes (G5).

Both studies are consumed for **every** GATED criterion, so no criterion treats
either code as ground truth; the tolerance is symmetric in the two.

The `0.25` ceiling is a cap on *relative* agreement — a "tolerance" above 25% is
not an agreement test. It is deliberately **not** applied to G5, whose tolerance
is absolute in mode numbers: clipping there would silently *tighten* the
criterion below what the uncertainties support (0.53 modes → 0.25), which is the
opposite of a safety cap.

### UNDERIVED: when no defensible tolerance exists

If either uncertainty is unavailable — the underlying convergence study is
`NON_MONOTONE`, `ORDER_OUT_OF_RANGE`, or the observable is missing — the
tolerance is marked `UNDERIVED`, the check is **demoted to DIAGNOSTIC for that
run**, and `overall_qualified` is set with a reason string. It never falls back
to a hand-picked number. An undefendable tolerance is worse than no tolerance:
it looks like a test and certifies nothing.

Against the committed Prompt B / Prompt D artifacts this is not hypothetical.
With `ours = n_substeps 1`, `pylle = tight`:

| criterion | outcome |
|---|---|
| G1 `intracavity_power_U_mean` | **UNDERIVED** — our `U_mean_w` study is NON_MONOTONE |
| G2 `comb_energy_fraction` | derived, tol = 8.60e-3 |
| G3 `band_limited_peak_power` | **UNDERIVED** — pyLLE's `P_peak_w` ladder is NON_MONOTONE |
| G4 `subbin_3db_span` | **UNDERIVED** — pyLLE's `S3_modes` ladder is NON_MONOTONE |
| G5 `dw_centroid_fixed_band` | derived, tol = 0.534 modes |
| G6 `dw_band_power_dbc` | derived, tol = 0.25 (clipped at the ceiling) |

At `ours = n_substeps 8` the derived tolerances tighten to 1.34e-3 (G2),
0.05 modes (G5, at its floor) and 1.83e-2 (G6). That four of six GATED criteria
are currently underivable is itself a result: it says the two convergence
studies do not yet support a quantitative agreement claim on those observables,
and the honest response is to say so rather than to invent a number.

---

## 4. No tolerance here was chosen to make a result pass

This is the claim most worth being able to check, so here is how a reader
verifies it rather than trusting it.

1. **The tolerances are functions of data produced before this module existed.**
   `convergence_lle_dw30k.json` (Prompt B) and `pylle_refinement_dw30k.json`
   (Prompt D) were both committed *before* the criteria were written, and their
   sha256 hashes are recorded in every report's `criteria_provenance.sources`.
   Re-run `derive_tolerances()` against those files and you must get the same
   numbers; change either file and the tolerances move.

2. **The tolerances are fingerprinted before any comparison value is read.**
   `derive_tolerances()` takes a sha256 over the `(criterion → tolerance)`
   mapping at derivation time. `build_report()` recomputes it and **raises** if
   it differs. Adjusting a tolerance after seeing a result is therefore not a
   matter of discipline; it is mechanically prevented, and
   `tests/test_criteria.py::test_fingerprint_guard` proves the guard fires.

3. **No GATED tolerance is a literal in the source.**
   `test_no_gated_tolerance_literal_in_derivation_code` walks the AST of
   `derive_tolerances` and fails if any numeric constant other than trivial
   structural values appears. The only numbers written down anywhere in the
   module are `COVERAGE_FACTOR`, the floor tables, `MAX_TOL`, and the HARD
   thresholds `1e-12` / `1e-6`.

4. **The new criteria are, on the committed evidence, mostly *harder* than the
   ones they replace, not softer.** G5 replaces a criterion the committed run
   failed with one it also fails (0.65 modes against a 0.534-mode tolerance at
   n=1, and 0.05 at n=8). G6 introduces a comparison that v1 measured and never
   made. The one criterion that becomes structurally unable to fail — the −60 dBc
   line count — does so because its core is *identical* between the two codes at
   every refinement level, which is evidence that it was never measuring the
   comb.

5. **The superseded tolerances are preserved, not deleted.**
   `TOLERANCES_V1_HISTORICAL` keeps the v1 dict verbatim, and
   `test_v1_tolerances_preserved_but_unused` asserts by AST that no code path
   reads it.

---

## 5. An open disagreement this framework does not explain away

The lower existence edge is genuinely resolved as a **disagreement**: the
brackets `[5.75, 5.9375]` (ours) and `[6.125, 6.3125]` (pyLLE) do **not**
overlap, so G7 fails on real evidence rather than on a biased point estimator.

This one matters because the usual explanation does not apply to it. That edge
sits at a Kerr phase of about **0.075 rad per round trip** — the regime
`docs/PYLLE_STATUS.md` itself describes as converged, and roughly five times
gentler than the 0.373 rad/round trip at the DW operating point where the
step-size argument was made. Whatever separates the two codes at the lower
existence edge, it is not the shared step size.

This document does not resolve it. It records it as an open question with the
step-size explanation explicitly ruled out, so that a later investigation starts
from the right place.
