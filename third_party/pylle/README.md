# Vendored pyLLE Julia kernel — made refinable

## Why this exists

pyLLE is useful to this repo as an **independent implementation** of the LLE, not
as a reference answer. An independent-implementation check is only meaningful at
an accuracy level *both* codes can actually reach, and upstream pyLLE is pinned
at exactly one integration step per round trip. At this repo's DW operating
point that is 1.5–2% from pyLLE's own converged answer, which is looser than any
tolerance worth quoting.

Restoring the step size lets us ask the only question that matters: **do the two
codes converge to the same limit?** If they do, the residual gaps at default
settings are discretization and nothing more. If they do not, there is a real
physics or implementation mismatch to find. Neither answer is available while
pyLLE cannot be refined.

**Nothing in `site-packages` is modified.** The vendored kernel is a separate
file; the driver invokes Julia on it directly.

## Pinned upstream

| | |
|---|---|
| pyLLE version | **4.1.2** (PyPI) |
| upstream `ComputeLLE.jl` sha256 | `fc84520cff40909cfae4e68c5cc6fb0812b947d4070a73f520c854dd426d12fe` |
| vendored `ComputeLLE.jl` sha256 | `f400979c29ec158ed26edc27030b93e8c40b0cb4a1b8712917b1ff5c2eb599cf` |
| upstream line count | 430 |
| vendored line count | 482 |

`ComputeLLE.jl.orig` is the pristine upstream file. `verify_vendor.py` re-derives
the vendored file by applying the three patches to `.orig` and asserts the result
is byte-identical, so the vendoring is auditable rather than trusted. It also
hashes the *installed* pyLLE kernel and fails loudly on version drift.

```bash
python third_party/pylle/verify_vendor.py --diff
```

`assert_vendor_integrity()` is the library entry point; call it before every run.

## Is the kernel actually refinable? (the premise, checked first)

Before writing any patch, every occurrence of `dt` in the upstream file was
listed and checked for dimensional consistency with a step of `dt` round trips.
There are six, two of which are dead:

| line | occurrence | verdict |
|---|---|---|
| 122 | `# dt=0.2/(sqrt(Ptot))` | commented out, dead |
| 123 | `dt = 1` | the definition — the only thing pinning the step |
| 129 | `Nt = round(t_ramp/tR/dt)` | consistent: step count scales as 1/dt |
| 348 | `A0 .+ Fdrive(it).*sqrt(κext).*dt` | consistent: drive kick over dt |
| 350 | `exp.(FFT_Lin(it) .* dt/2)` | consistent: linear half-step |
| 362 | `(NL½prop_0 .+ NL½prop_1) .* dt/2` | consistent: trapezoidal NL over dt |

All four live uses are consistent, so **the kernel is refinable as written** and
`dt = 1` is the only obstacle. Had any use been inconsistent, the correct
outcome would have been to report that pyLLE is *not* refinable rather than to
patch around it (task condition F3). `tests/test_pylle_vendor.py::
test_dt_is_dimensionally_consistent` keeps this checked.

## The three patches

Each is minimal, separately revertible, and carries an in-file header comment.

### `0001-restore-dt-cli` — 11 changed lines

Replaces `dt = 1` with `dt = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0`.

`ARGS[1..4]` are already `tmp_dir`, `tol`, `maxiter`, `step_factor` (upstream
:44-48), so `dt` is **appended as `ARGS[5]`** — no existing argument is
reordered. With the argument absent the value is `1.0` and behaviour is
bit-identical to upstream.

### `0002-restore-tol-maxiter-cli` — 33 changed lines

Upstream parses the CLI tolerance at :46-47 and then **unconditionally
overwrites it** at :278-279:

```julia
tol = parse(Float64,ARGS[2])   # :46
maxiter = parse(Int,ARGS[3])   # :47
...
tol = 1e-2                     # :278  <- clobbers the CLI value
maxiter = 10                   # :279
```

The patch deletes those two assignments so the ARGS values survive; :46-47 are
untouched. (`param["tol"] = 1e-3` at :402-403 is separately dead: `SSFM½step`
reads the module-level globals and never consults `param`.)

It also adds four counters (`picard_iter_total`, `picard_iter_max`,
`picard_calls`, `picard_fail_count`) and writes `S["picard_stats"]` into the
output HDF5. This is the point of restoring the knob: at `tol = 1e-2` the Picard
loop exits on a *change-per-round-trip* test rather than on local truncation
error, and only the iteration count reveals that. Measured at the DW operating
point: **mean 2.00, max 2 at `tol = 1e-2`**; **mean 8.72, max 10 at
`tol = 1e-8`**, zero maxiter exhaustions.

`RetrieveData` reads a fixed key list, so adding a key cannot break it.

### `0003-probe-final-step` — 14 changed lines

`SaveStatus_CallBack` fires on `it*num_probe/Nt > probe`, which never triggers on
the final step: with `Tscan = 5000` and `num_probe = 200` the last probe is
written at round trip **4976**, so the returned field sits at
`delta_omega = 29.9328 kappa` while the run was asked for `30.0000 kappa`. Any
cross-code comparison then silently compares two different detunings.

The patch overwrites only the **last** probe slot, after the main loop, with the
true end-of-run state. The intermediate probe cadence is deliberately unchanged.

## What is and is not preserved

**Preserved exactly.** With `dt = 1.0`, `tol = 1e-2`, `maxiter = 10` and patch
0003 reverted, the vendored kernel reproduces upstream **bit-for-bit**
(equivalence gate: relative L2 = `0.000e+00`, tolerance 1e-12). Patches 0001 and
0002 are behaviour-preserving at upstream's defaults by construction: one adds an
argument that defaults to the old constant, the other removes an assignment that
was overwriting a value which, at the default, equals what it was overwritten
with.

**Deliberately changed.** Patch 0003 changes the returned field, because
upstream's returned field is at the wrong detuning. This is a *fix*, not a
preservation, and it is the reason the gate reverts 0003 before comparing.

**Not changed, and not fixed here.** The drive kick is applied *before* the
L·N·L composition (`A0 += Fdrive*sqrt(κext)*dt` at :348), which makes the
composite map non-palindromic and therefore **first order overall** despite the
Strang-symmetric interior. This repo's own solver has the identical defect by
default (`symmetric_drive=False`). Both are left alone: fixing one and not the
other would destroy comparability, and fixing both is out of scope.

**A genuine scheme difference to keep in mind.** pyLLE's nonlinear step is an
**endpoint-trapezoidal** rule, `exp((NL(A0) + NL(A_iter))*dt/2)` with
`NL(u) = -1i*gamma_NLSE*L*|u|^2`, iterated to a fixed point. Ours is a
**midpoint** rule, `exp(1i*gamma_LLE*|E_half|^2*dt_sub)`. Both are O(dt²) in the
nonlinear step but with different error constants **and different signs of the
leading term** — which is the leading candidate explanation if the two codes
approach their common limit from opposite sides.

## How to revert

Any single patch:

```bash
cd third_party/pylle
patch -R -p0 ComputeLLE.jl < patches/0003-probe-final-step.patch
```

All the way back to upstream: use `ComputeLLE.jl.orig`, or
`verify_vendor.revert_to_upstream(dest)`. To run *upstream* through the same
driver, pass `jl_path` pointing at `ComputeLLE.jl.orig` and `pass_dt=False`
(that is exactly what the equivalence gate does).

## How the vendored kernel is driven

Upstream's `LLEsolver.SolveTemporal` builds its Julia command line inline with a
fixed four-argument list and offers no hook for a fifth, and `path_juliaScript`
is a module global resolved at import. Rather than subclass a large method or
monkeypatch a global, `run_refinement.py` uses pyLLE for `Analyze`/`Setup` only —
which is what writes `ParamLLEJulia.h5` — then invokes Julia itself:

```
julia <kernel.jl> <tmp_dir> <tol> <maxiter> <step_factor> <dt>
```

and reads `ResultsJulia.h5` directly. Reading it directly is also what makes the
`picard_stats` key visible: `RetrieveData` deletes both HDF5 files before
returning.

Note that pyLLE's `tmp_dir` is a filename **prefix**, not a directory — it builds
paths by plain concatenation (`tmp_dir * "ResultsJulia.h5"`), and so must any
caller.

## Provenance of the measurements

`validation/results/pylle_refinement_dw30k.json` records the pinned hashes, each
patch's hash and rationale, the Julia and Python package versions, the depot
path, the full `res`/`sim` dicts, the dispfile and seed hashes, and per level the
`dt`, `Nt`, `tol`, `maxiter`, mean/max Picard counts, wall clock and every
observable from `validation/convergence_lle.observables_v2`.
