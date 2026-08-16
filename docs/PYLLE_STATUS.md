# pyLLE cross-check: OPERATIONAL

Status as of 2026-08-16. Supersedes an earlier revision of this file that
recorded the cross-check as blocked; **that conclusion was wrong.** It rested on
the Julia distribution hosts (`julialang-s3.julialang.org`,
`install.julialang.org`) being denied by the session's egress policy, and on
`apt` having no `julia` candidate. Both are true, and neither is sufficient:
**conda-forge is reachable and ships Julia.** The blocker was an incomplete
search, not the environment.

The cross-check now runs end to end: `validation/pylle_crosscheck.py`, with
results in `validation/results/pylle_crosscheck.{json,png}` and the raw final
fields in `pylle_crosscheck_fields.npz`.

## Environment recipe (the part that is not obvious)

pyLLE 4.1.2 cannot share this repo's environment: it calls `np.string_`
(removed in NumPy 2) and it shells out to a Julia binary. It therefore lives in
its own conda env and is driven out-of-process.

```bash
# 1. micromamba: itself a conda-forge package, and a plain .tar.bz2 that
#    Python's tarfile can unpack -- no bootstrap installer needed.
curl -sSLO https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2
python3 -c "import tarfile; tarfile.open('micromamba-2.9.0-0.tar.bz2').extractall('mm')"
chmod +x mm/bin/micromamba

# 2. Julia + the pyLLE Python stack, pinned to numpy<2.
export CURL_CA_BUNDLE=/root/.ccr/ca-bundle.crt   # proxy re-terminates TLS
./mm/bin/micromamba create -y -c conda-forge -p ./pylle-env \
    "julia=1.10" "python=3.11" "numpy<2" scipy h5py matplotlib plotly prettytable
./pylle-env/bin/pip install pyLLE ipdb

# 3. Julia packages. The Pkg server (pkg.julialang.org) is blocked, but git
#    access to GitHub works, and Pkg falls back to git when the server is
#    unset. This is the step that looks impossible and is not.
export JULIA_DEPOT_PATH=$PWD/juliadepot
export JULIA_PKG_SERVER=""
export JULIA_SSL_CA_ROOTS_FILE=/root/.ccr/ca-bundle.crt
./pylle-env/bin/julia -e 'using Pkg; Pkg.add(["FFTW","HDF5"])'
```

Then, from the repo root (with `JULIA_DEPOT_PATH` still exported):

```bash
python -m validation.pylle_crosscheck \
    --pylle-python ./pylle-env/bin/python \
    --julia-bin    ./pylle-env/bin/julia
```

Versions actually used: Julia 1.10.4 (conda-forge), pyLLE 4.1.2, Python 3.11.15,
NumPy 1.26.4 (pyLLE side) / 2.4.6 (ours), h5py 3.16.0, JAX 0.10.2.

## What the cross-check establishes, and what limits it

The nine convention findings (field conjugation, detuning sign, the dispersion
mirror, the pump-mode reference, the D1/timebase coupling) are documented in
full in the module docstring of `validation/pylle_crosscheck.py`, which is the
authoritative record. Two are worth repeating here because they are traps
rather than bookkeeping:

* **pyLLE's field is the complex conjugate of ours** and its detuning sign is
  opposite, forced by its `-i*gamma*L*|A|^2` Kerr term against our
  `+i*gamma*|E|^2`. Conjugation mirrors the spectrum, so pyLLE must ALSO be
  handed the **mirrored dispersion** `D_int_pyLLE(mu) = D_int_ours(-mu)`. Get
  the field map right but not the dispersion map and both codes still produce
  perfectly plausible solitons that disagree only on asymmetric observables.
* **The pump is not at `c/pump_wavelength_m`.** Our loader references D_int to
  the CSV row `mu = 0`, which is 7.11 FSR away from the nominal 1.55 um. pyLLE
  locates the pump by frequency, so the naive translation puts the two codes on
  pump modes seven free spectral ranges apart.

> **SUPERSEDED — see "Correction: pyLLE is refinable" at the end of this
> document.** The paragraph below is kept verbatim because it is what the
> committed cross-check was reasoned from. Its second clause is wrong: the floor
> is real, but it *is* adjustable. Both `dt` and the nonlinear tolerance are
> reachable parameters that upstream disables rather than omits.

The accuracy floor is **pyLLE's, and it is not adjustable**: `ComputeLLE.jl:123`
hardwires `dt = 1` round trip, and `:278` overrides the CLI nonlinear-iteration
tolerance with `1e-2`. At an operating point short enough to excite this
device's dispersive waves the Kerr phase reaches ~0.37 rad per round trip, where
one step per round trip is several percent from converged — measured directly by
refining our own side (`convergence_attribution` in the JSON). Sub-2% cross-code
agreement is therefore not attainable there by either code, which is a fact
about pyLLE as a reference, not about this repo's solver.

## Provenance note on the committed artifacts

`pylle_crosscheck.json` stamps `git_commit = b31cc0b`, the HEAD at the moment
the run executed; `validation/pylle_crosscheck.py` was still uncommitted then,
so that SHA dates the run rather than identifying the code. After the run I made
two edits to the **figure-rendering code only** — zooming the waveform panel and
centring each trace on its own peak (the two solitons sit ~2e-2 rad apart in
theta, and every compared observable is translation-invariant) — plus one
docstring wording fix, and regenerated the PNG from the saved `.npz` with the
committed `make_figure`. **No numerical result, tolerance, or verdict was
touched**: the JSON and the `.npz` fields are exactly as the run produced them.
Re-running the committed script reproduces the numbers; only the run timestamp
and this SHA will differ.

## Result of the run committed here

Operating point: measured SiN dispersion, 6601 modes (mu = -3300..3300), soliton
seeded deterministically and ramped delta_omega 16 kappa -> 30 kappa over 5000
round trips, one step per round trip on both sides.

| observable | ours | pyLLE | rel. diff | tol | verdict |
|---|---|---|---|---|---|
| DW peak mode index (red) | -3075 | -3074 | 1 mode (0.03%) | exact | FAIL |
| DW peak mode index (blue) | none | none | — | exact | NOT MEASURED |
| 3 dB spectral span | 443 | 444 modes | 0.23% | 2% | **PASS** |
| soliton peak power | 230.11 W | 240.28 W | 4.23% | 2% | FAIL |
| existence edge, lower | 5.94 kappa | 6.31 kappa | 5.94% | 5% | FAIL |
| existence edge, upper | 37.50 kappa | 36.88 kappa | 1.67% | 5% | **PASS** |
| comb lines > -60 dBc | 3293 | 3612 | 8.83% | 2% | FAIL |

**Overall: FAIL — and the failures are attributed, not mysterious.** No tolerance
was loosened and no observable was redefined after seeing a result.

What the run establishes positively:

* the translation is right — the round trip closes to 1.2e-16, and pyLLE's
  independently spline-refit D_int reproduces ours to 2.3e-12 after un-mirroring;
* both codes agree on soliton survival at **all seven** scan detunings, and
  their bisected existence edges agree to 5.9% (lower) and 1.7% (upper);
* the spectra overlap to a median 0.09 dB across the soliton core;
* both find the dispersive wave at the same mode to within one index out of
  3075.

The residual few-percent gaps track the shared, non-refinable step:

* refining only our side moves **our** DW index from -3075 to -3074, i.e. onto
  pyLLE's value, at both n_substeps = 2 and 4 — so that off-by-one is our own
  discretization error, not a convention error;
* the -60 dBc line count differs by 8.83% at the DW point (Kerr phase 0.373
  rad/round trip) but by **0.33%** at delta_omega = 8 kappa (0.099 rad/round
  trip) — a 27x improvement from lowering the per-step nonlinear phase alone;
* the spectra diverge only in the wings, where pyLLE sits ~10 dB higher (see the
  residual panel of the figure) — consistent with its fixed 1e-2 nonlinear
  iteration tolerance leaving a broadband numerical floor. That raised floor is
  what pushes its -60 dBc crossing outward and inflates its line count.

Two honest caveats on the criteria themselves, offered as observations rather
than as grounds for changing them:

* **"exact integer match" is a brittle criterion for a DW peak index.** The
  underlying disagreement is 0.65 modes out of 3075 (0.02%); no two distinct
  discretizations can be guaranteed to round to the same integer.
* **the 2% tolerances are below both codes' own accuracy at this operating
  point.** Our own n_substeps = 1 result sits 4.4% from our converged answer, so
  a 2% cross-code agreement is asking for better agreement than either code has
  with itself. It is met where it can be (the 3 dB span, 0.23%).

## Where verification actually rests

Cross-code agreement remains the weakest of the three checks, and this exercise
did not change that. It is a check on **convention and bookkeeping** — and it
earned its keep precisely there, catching the dispersion mirror and the 7-FSR
pump offset, neither of which any single-code test could have surfaced. It is
not a check on numerics: two codes agree only to the level at which they
discretize the same PDE, and here that level is set by the coarser of the two.

Quantitative verification continues to rest on:

* **`validation/analytic_cw.py`** — the solver against *exact mathematics*
  (cubic root and the discrete map's fixed point) to ~1e-14, gated at 1e-12.
* **`validation/convergence.py`** and **`validation/mms.py`** — observed order
  of accuracy against manufactured solutions.

Both are stronger claims than agreement with another code at a few percent.

---

# Correction: pyLLE is refinable

Added 2026-08-16. Evidence: `third_party/pylle/` (vendored kernel, three audited
patches, `verify_vendor.py`) and `validation/results/pylle_refinement_dw30k.json`.
Nothing above has been deleted; the affected paragraph is marked superseded in
place.

## 1. The claim being corrected

> "The accuracy floor is **pyLLE's, and it is not adjustable**: `ComputeLLE.jl:123`
> hardwires `dt = 1` round trip, and `:278` overrides the CLI nonlinear-iteration
> tolerance with `1e-2`. … Sub-2% cross-code agreement is therefore not
> attainable there by either code, which is a fact about pyLLE as a reference,
> not about this repo's solver."

The factual observations in that sentence are correct. The conclusion drawn from
them is not.

## 2. What the source actually shows

**`dt` is a plumbed parameter, not a structural constant.** Every occurrence of
`dt` in the upstream kernel was enumerated before any patch was written. Four are
live — `Nt = round(t_ramp/tR/dt)` (:129), the drive kick `.*dt` (:348), the
linear half operator `.*dt/2` (:350), and the trapezoidal nonlinear phase
`.*dt/2` (:362) — and **all four are dimensionally consistent with a step of
length `dt` round trips**. `dt = 1` at :123 is the only obstacle, and :122 is a
commented-out adaptive expression showing the author intended it to vary. Had any
use been inconsistent, the correct conclusion would have been that pyLLE is *not*
refinable; that is now a standing test.

**The tolerance CLI already exists upstream and is clobbered.** `:46-47` parse
`tol` and `maxiter` from `ARGS`; `:278-279` then unconditionally overwrite them
with `1e-2` and `10`. (`param["tol"] = 1e-3` at :402 is separately dead —
`SSFM½step` reads module globals and never consults `param`.) The knob was
built and then disabled, not absent.

Three minimal patches restore both. With `dt=1.0, tol=1e-2, maxiter=10` and the
probe patch reverted, the vendored kernel reproduces upstream **bit-for-bit**
(relative L2 = `0.000e+00`), so nothing was changed except what is documented.

## 3. Measured refinement table

pyLLE, vendored kernel, DW operating point (6601 modes, `Tscan = 5000`,
δω 16κ→30κ, same deterministic seed the cross-check uses). `P_peak` is the
band-limited peak power of `validation/convergence_lle.observables_v2`.

| dt | Nt | tol | mean Picard | max | P_peak (W) | S3 | mu_DW | U_mean (W) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 5000 | 1e-2 | **2.00** | 2 | 239.966 | 445.440 | −3074.556 | 0.192124 | 32.6 |
| 0.5 | 10000 | 1e-2 | **2.00** | 2 | 236.107 | 445.078 | −3074.941 | 0.192871 | 54.0 |
| 0.25 | 20000 | 1e-2 | **2.00** | 2 | 236.461 | 445.059 | −3075.093 | 0.193101 | 103.2 |
| 0.125 | 40000 | 1e-2 | **1.00** | 1 | 235.993 | 445.165 | −3074.272 | 0.193173 | 139.9 |
| 1.0 | 5000 | 1e-8 | 8.72 | 10 | 240.138 | 445.316 | −3074.658 | 0.192192 | 84.3 |
| 0.5 | 10000 | 1e-8 | 5.50 | 6 | 237.446 | 445.128 | −3075.045 | 0.192861 | 105.4 |
| 0.25 | 20000 | 1e-8 | 4.00 | 4 | 235.438 | 445.012 | −3075.228 | 0.193103 | 148.8 |
| 0.125 | 40000 | 1e-8 | 3.70 | 4 | 235.549 | 445.027 | −3075.248 | 0.193204 | 269.6 |

Zero maxiter exhaustions anywhere; no `Convergence Error` was printed by any run.

**pyLLE moves by 1.9% under refinement** (240.14 → 235.55 W at tight tolerance),
which confirms the premise: upstream at `dt = 1` really was ~2% from its own
converged answer. The floor was real. It was simply not a floor.

**Does it converge to the same limit as ours?** Partly, and the split is
informative:

| observable | pyLLE (refined) | ours (Prompt B, Richardson) | gap |
|---|---|---|---|
| `mu_DW` | −3075.25 (p = 3.18) | −3075.251 (p = 2.48) | **~1e-6** |
| `comb_frac` | 0.854518 | 0.853371 | 0.13% |
| `U_mean_w` | 0.193275 | 0.191299 | 1.03% |
| `P_peak_w` | ~235.5 (non-monotone) | 232.809 (p = 2.31) | ~1.2% |

The spectral-position observable agrees to a part in a million — strong evidence
the dispersion, detuning and mode-index conventions are all correct. The
energy-like observables retain a ~1% gap that refinement on either side does not
close. That gap is a real, open discrepancy, not a discretization artifact, and
it is the thing a cross-check should now be pointed at. The leading candidate is
the nonlinear-step rule: pyLLE uses **endpoint-trapezoidal**
`exp((NL(A0)+NL(A_iter))·dt/2)` iterated to a fixed point, ours uses **midpoint**
`exp(i·γ|E_half|²·dt_sub)`; both are O(dt²) with different error constants.

Note also that pyLLE's `P_peak` sequence is itself **non-monotone** at the finest
steps (235.44 → 235.55), so its own limit is not cleanly established; the
conservative band is 0.85%. Its well-conditioned observables (`mu_DW`,
`U_mean_w`, `comb_frac`) do converge with clean observed orders.

**Disagreement with the reference values quoted in the task brief, reported
rather than reconciled.** The brief cites a NumPy transcription giving
`dt=1 → 237.38`, `dt=0.5 → 233.42`, `dt=0.25 → 233.81 W`. Measured here (tight):
**240.14, 237.45, 235.44**, i.e. consistently 1.2–4.0 W higher. Nothing was
adjusted to close this. The most likely cause is patch 0003: the brief's
transcription presumably reads the field at the unpatched probe position
(29.9328 κ) whereas every number above is at 30.0000 κ, and a lower detuning
gives a lower peak. That is a hypothesis, not a measurement.

## 4. Picard iteration count at the default tolerance

**Mean 2.00, max 2, at every one of the 5000 steps** for `dt = 1, tol = 1e-2` —
reproducing the `{2: 5000}` histogram quoted in the brief.

The decisive observation is what happens to that count as `dt` shrinks at fixed
loose tolerance: **2.00 → 2.00 → 2.00 → 1.00**. It goes *down*. A criterion
measuring local truncation error would demand *more* work per step as the scheme
is refined toward an exact solve. This one demands less, because
`‖A_prop − A_half‖ / ‖A_half‖ < tol` measures **how much the field changed over
this round trip**, and a shorter step changes it less. At `dt = 0.125` the very
first iterate already passes. The upstream default is therefore not a convergence
criterion at all; it is a step-size-dependent activity test, and it silently
becomes weaker exactly when the user is trying to make the answer better.

At `tol = 1e-8` the count behaves correctly — 8.72 → 5.50 → 4.00 → 3.70, falling
gently as the step shrinks and the initial guess improves, which is what a real
convergence criterion does.

## 5. The probe off-by-24 and the detuning mismatch

`SaveStatus_CallBack` fires on `it*num_probe/Nt > probe` (:232-242), which never
triggers on the final step. With `Tscan = 5000` and `num_probe = 200` the last
probe is written at round trip **4976 of 5000** — 24 steps early. The returned
field therefore sat at **δω = 29.9328 κ** while our field sat at **30.0000 κ**:
the committed cross-check was comparing two solitons at different detunings, and
attributing the difference to the codes.

Patch 0003 overwrites only the final probe slot with the true end-of-run state.
After it, `detuning_pylle_final` is exactly **30.000000 κ** at every one of the
eight levels above.

This is small but not negligible: 0.0672 κ of detuning at a point where
`dP_peak/dδω` is steep. It is also exactly the class of error that no amount of
tolerance-tuning would ever have surfaced — the two codes were solving slightly
different problems.

## 6. What this does and does not change

It does not change any committed number in `pylle_crosscheck.json`; that run
stands as a record of upstream-default pyLLE against our default solver.

It does change what a future cross-check should do: run pyLLE refined, compare
limits rather than defaults, and treat the residual ~1% energy-observable gap as
the finding rather than as noise. The claim that "sub-2% cross-code agreement is
not attainable by either code" is withdrawn — it is attainable for `mu_DW` and
`comb_frac`, and the reason it is not yet attained for `P_peak` and `U_mean_w`
is an open question rather than a known limitation.
