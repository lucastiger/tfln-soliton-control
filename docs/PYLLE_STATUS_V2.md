# pyLLE cross-check v2: convention audit + convergence-envelope containment

Supersedes `docs/PYLLE_STATUS.md`, which is retained unchanged as the historical
record of the v1 run. The v1 artifacts are frozen at
`validation/results/v1/` (with a `MANIFEST.md`); the originals at
`validation/results/pylle_crosscheck.{json,png}` and
`pylle_crosscheck_fields.npz` are untouched and their sha256 are asserted on
every test run.

Code: `validation/pylle_crosscheck_v2.py`.
Results: `validation/results/pylle_crosscheck_v2.{json,png}` and
`pylle_crosscheck_v2_fields.npz`.

---

## 1. What a cross-code check can and cannot establish

It cannot establish that either code is right. Two independent implementations
of the same PDE agree only to the level at which they discretize it, and
agreement cannot distinguish "both right" from "both wrong in the same way".
What it *can* establish is two things exact mathematics cannot reach: first, the
**conventions and bookkeeping** — the sign of the detuning, the direction of the
dispersion, factors of 2π, the mode-index origin, which resonance the pump sits
on — which are the errors that silently produce plausible-looking solitons; and
second, **envelope containment**, whether the two codes, each refined along its
own ladder, extrapolate to the *same limit*. The second is a question about the
continuum solution both codes claim to approximate, and unlike a single-point
comparison it can be answered without either code standing in for the truth.
Everything below is one of those two claims. Nothing below is a claim that this
solver is correct because pyLLE agrees with it.

---

## 2. The nine convention findings — VERIFIED

Carried over verbatim from the module docstring of
`validation/pylle_crosscheck.py`, which remains the authoritative record. They
were established from pyLLE 4.1.2 source and are **imported, not re-derived**:
`tests/test_pylle_crosscheck_v2.py::test_convention_functions_are_the_v1_objects`
fails if v2 defines its own copy of any of them. Line numbers below were
re-checked against `third_party/pylle/ComputeLLE.jl.orig` (the pristine vendored
kernel, sha256 `fc84520c…`, bit-identical to the installed upstream) and against
pyLLE 4.1.2's Python sources for this document.

| # | Finding | Verified against |
|---|---|---|
| 1 | `D_int` is in **rad/s** (angular) on both sides; no 2π conversion anywhere. Both `Dint` and `domega` are multiplied by `tR` to become dimensionless, as is `κext`. | `ComputeLLE.jl:338` (`FFT_Lin`), `:88` (`κext = ω0[1]/Qc*tR`) |
| 2 | pyLLE's field is the **complex conjugate** of ours and its detuning sign is **opposite**: `E_ours = conj(A_pyLLE)·√t_r`, `δω_pyLLE = −δω_ours`. Forced by its `−1i·γL|A|²` Kerr term against our `+1i·γ|E|²`. | `ComputeLLE.jl:342` (`NL`), `:338`; our `simulator/lle_solver.py:896`/`:936` |
| 2b | The conjugation mirrors the spectrum, so the **dispersion input must also be mirrored**: `D_int_pyLLE(μ) = D_int_ours(−μ)`. Omitting it produces two plausible solitons that disagree only on asymmetric observables. | `validation/pylle_crosscheck.py::build_pylle_dispfile`; `test_dispersion_mirror_is_applied_to_the_input` |
| 3 | Amplitude normalization is a power/energy conversion: `\|A\|² = \|E\|²/t_r` (pyLLE's `A` is √W, ours is √J). | `ComputeLLE.jl:86` (`tR = 1/FSR`); both CW steady states reduce to `4κ_c·P_in/κ²` |
| 4 | The drive phase is fixable exactly: `φ_pmp = −π/2` makes pyLLE's `−1i·A_in` drive real positive. | `ComputeLLE.jl:70` (`φpmp = sim["phi_pmp"]`), `:189` (`Ain … exp(-1im*φpmp)`), `:272` (`Force .- 1im.*Ain`) |
| 5 | pyLLE 4.1.2 is **already deterministic**: `Noise()` is defined but its call site is commented out, and `DKS_init` defaults to zeros — so it can never form a soliton unaided. Both codes get the same explicit sech seed. | `ComputeLLE.jl:172` (definition), `:274` (`return Force #.- 1im*Noise()`), `_llesolver.py:479` (`np.zeros`) |
| 6 | pyLLE re-zeros `D_int` at the **domain centre**, not the pump mode. With the pump centred the two coincide; the worker asserts this rather than assuming it. | `ComputeLLE.jl:106` (`μ0`), `:115` (`Dint .- Dint[μ0]`) |
| 7 | Dispersion enters pyLLE as a resonance-frequency CSV that it **spline-refits**, so its `D_int` is not bit-identical to our loader's. Eliminated by feeding pyLLE's own refit array back into *both* codes. | `_analyzedisp.py:100-106` |
| 8 | The nominal pump wavelength is **not** the mode the dispersion is referenced to, and pyLLE locates the pump **by frequency**. Our loader references `D_int` to CSV `μ = 0`, which is **+7.11 FSR** from the config's nominal 1.55 µm. | `_analyzedisp.py:63-70` (`pmp_ind` search), `:127` (`mu_fit` assert) |
| 9 | `D1` sets the co-moving frame **and**, in pyLLE, the round-trip time `t_r = 2π/D1`. The repo's two FSRs differ by 0.6% and only the CSV-fitted `D1` keeps the dispersive waves where they belong. | `ComputeLLE.jl:86`; `simulator/lle_solver.py:509-522` |

**These are the part of the v1 work that has held up**, and they are what the
cross-check earns its keep for. Findings 2b and 8 in particular are traps no
single-code test could surface. Measured this run: the parameter round trip
closes to **1.23e-16** (H1), pyLLE's independently spline-refit `D_int`
reproduces ours to **2.32e-12** after un-mirroring (H2), and pyLLE's own pump
search lands **exactly** (0 Hz) on the CSV `μ = 0` resonance our `D_int` is
referenced to (H4).

---

## 3. Environment recipe

Unchanged from v1 (reproduced here so this document stands alone), plus the
vendored-kernel step.

pyLLE 4.1.2 cannot share this repo's environment: it calls `np.string_` (removed
in NumPy 2) and shells out to a Julia binary. It lives in its own conda env and
is driven out-of-process.

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

**Vendored kernel.** The refinable kernel lives at
`third_party/pylle/ComputeLLE.jl` (sha256 `f400979c…`) with the pristine
upstream beside it as `ComputeLLE.jl.orig` (sha256 `fc84520c…`) and three
audited patches in `third_party/pylle/patches/` (11, 33 and 14 lines):
`0001` restores the `dt` CLI, `0002` restores `tol`/`maxiter` and adds Picard
statistics, `0003` makes the last probe the true end-of-run state.
`third_party/pylle/verify_vendor.py::assert_vendor_integrity` re-derives the
vendored file from pristine + patches and is called at the start of every run.
**Nothing in `site-packages` is modified**: pyLLE is used for `Analyze`/`Setup`
and Julia is then invoked directly so `ARGS[5] = dt` can be passed.

Then, from the repo root:

```bash
python -m validation.pylle_crosscheck_v2 \
    --pylle-python ./pylle-env/bin/python \
    --julia-bin    ./pylle-env/bin/julia
```

**Every default is the value the committed run used**, so that bare command
reproduces the committed artifacts; the JSON records `argv` and
`defaults_reproduce_this_run: true` regardless.

**Reproducibility, verified.** The run was executed twice with identical argv.
Both produced `numerical_digest =
32f53054c2c8b33ea9a93162e2991001cd67d6eb271be310ae49431017efb9d3`, and the
figure and `.npz` sha256 were identical as well — i.e. every numerical field,
the plot and the raw fields are bit-for-bit reproducible. (The digest is a
sha256 over the whole report with only genuinely volatile fields removed —
wall-clock times, timestamp, hostname, work directory; the derived config is
recorded by content hash rather than by its ephemeral path so that a changing
temp directory cannot masquerade as a numerical difference.) The tolerance
fingerprint is `98ea9e2f…`, unchanged from Prompt A's derivation, which is the
mechanical proof that no tolerance was touched between derivation and
evaluation.

Versions used: Julia 1.10.4 (conda-forge), pyLLE 4.1.2, Python 3.11.15, NumPy
1.26.4 (pyLLE side) / 2.4.6 (ours), h5py 3.16.0, JAX 0.10.2. Wall clock ≈ 33
minutes.

---

## 4. The ladder table

Operating point: measured SiN dispersion, 6601 modes (μ = −3300…3300), the same
deterministic sech seed on both sides, δω ramped 16 κ → 30 κ over 5000 round
trips, thermo-optic off. Both codes integrate the *same* `D_int` array, the
*same* seed, and end at the *same* final detuning (H5 = 0 exactly).

Observables are `validation.convergence_lle.observables_v2` applied to fields
already mapped into our convention. `h` is the step in round trips
(`1/n_substeps` for us, `dt` for pyLLE).

### Ours (`n_substeps`)

| n | h | U_mean (W) | comb_frac | P_peak (W) | S3 (modes) | μ_DW | DW power (dBc) | wall (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.192856 | 0.857011 | 233.9266 | 444.7820 | −3074.9846 | −29.7889 | 14.2 |
| 2 | 1/2 | 0.191564 | 0.854963 | 233.9356 | 445.0197 | −3075.1195 | −24.0125 | 17.3 |
| 4 | 1/4 | 0.191314 | 0.854087 | 233.1375 | 445.0463 | −3075.2286 | −23.0275 | 26.3 |
| 8 | 1/8 | 0.191287 | 0.853705 | 232.8752 | 445.0389 | −3075.2469 | −22.7300 | 45.1 |
| 16 | 1/16 | 0.191299 | 0.853526 | 232.8222 | 445.0363 | −3075.2502 | −22.6535 | 81.9 |

### pyLLE (`dt`), shipped Picard pair (tol 1e-2, maxiter 10)

| dt | mean Picard | max | U_mean (W) | comb_frac | P_peak (W) | S3 | μ_DW | DW power (dBc) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **2.00** | 2 | 0.192124 | 0.856231 | 239.9660 | 445.4403 | −3074.5562 | −22.4712 | 22.8 |
| 1/2 | **2.00** | 2 | 0.192871 | 0.855685 | 236.1070 | 445.0780 | −3074.9413 | −21.8308 | 35.6 |
| 1/4 | **2.00** | 2 | 0.193101 | 0.855226 | 236.4606 | 445.0590 | −3075.0933 | −22.0377 | 63.0 |
| 1/8 | **1.00** | 1 | 0.193173 | 0.854954 | 235.9929 | 445.1654 | −3074.2722 | −25.1621 | 79.4 |

### pyLLE (`dt`), tight Picard pair (tol 1e-8, maxiter 60)

| dt | mean Picard | max | U_mean (W) | comb_frac | P_peak (W) | S3 | μ_DW | DW power (dBc) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 (`dt1_tight`) | 8.72 | 10 | 0.192192 | 0.856247 | 240.1384 | 445.3164 | −3074.6577 | −20.0540 | 49.6 |
| 1/2 | 5.50 | 6 | 0.192861 | 0.855709 | 237.4462 | 445.1278 | −3075.0451 | −21.6317 | 64.9 |
| 1/4 | 4.00 | 4 | 0.193103 | 0.855185 | 235.4384 | 445.0123 | −3075.2278 | −22.3882 | 94.2 |
| 1/8 | 3.70 | 4 | 0.193204 | 0.854892 | 235.5486 | 445.0271 | −3075.2479 | −22.5668 | 167.4 |

Zero `maxiter` exhaustions anywhere; no `Convergence Error` printed by any run.
Note the shipped-tolerance Picard count **falls** as `dt` shrinks (2 → 2 → 2 →
1): that criterion measures how much the field changed over the step, so a
shorter step passes on the first iterate. It is a step-size-dependent activity
test, not a convergence criterion. At `tol = 1e-8` the count behaves correctly
(8.72 → 5.50 → 4.00 → 3.70).

### Limits and containment

Each ladder is Richardson-extrapolated independently
(`validation.convergence_lle.richardson`, three finest levels, GCI with safety
factor 1.25, order window [0.5, 4.0]). Containment compares the two limits
against the combined band `K·√(U_o² + U_p²)` with `K = 2`. The headline pairing
is against pyLLE's **tight** ladder, because that is the ladder whose measured
uncertainty Prompt A derived every GATED tolerance from — a choice frozen and
fingerprinted before this run, not selected from its results.

| observable | ours limit ± U | pyLLE limit ± U | gap | band | verdict |
|---|---|---|---|---|---|
| `U_mean_w` | — (NON_MONOTONE) | 0.193275 ± 5.2e-4 | — | — | **INDETERMINATE** |
| `comb_frac` | 0.853371 ± 2.28e-4 | 0.854518 ± 5.47e-4 | 1.342e-3 | 1.184e-3 | **SEPARATED** |
| `P_peak_w` | 232.809 ± 2.28e-4 | — (NON_MONOTONE) | — | — | **INDETERMINATE** |
| `S3_modes` | 445.035 ± 5.67e-6 | — (NON_MONOTONE) | — | — | **INDETERMINATE** |
| `mu_DW` | −3075.2509 ± 3.3e-3 modes | −3075.2504 ± 2.0e-2 modes | **4.5e-4 modes** | 4.1e-2 modes | **CONTAINED** |
| `dw_power_dbc` | −22.6271 ± 3.4e-3 | −22.6221 ± 7.9e-3 | 2.2e-4 | 1.7e-2 | **CONTAINED** |

Observed orders: ours `P_peak` 2.31, `S3` 1.56, `μ_DW` 2.48, `dw_power` 1.96,
`comb_frac` 1.10; pyLLE (tight) `μ_DW` 3.18, `dw_power` 2.08, `U_mean` 1.26,
`comb_frac` 0.84.

**Caveat on the two CONTAINED rows: they are agreement on grid-truncated
values.** Both ladders run at `mu_half = 3300`, where the comb is not spectrally
contained (§6: the blue edge sits at −44 dBc and the blue DW predicted at
μ ≈ +3239 is unresolved, 61 modes inside the boundary). The `grid_levels` block
of `validation/results/convergence_lle_dw30k.json`, at `n_substeps = 8`, shows
what happens when the grid is widened until it *is* contained:

| `mu_half` | 3300 | 4400 | 5500 |
|---|---|---|---|
| `mu_DW` | −3075.2469 | −3079.0401 | −3079.0405 |
| `dw_power_dbc` | −22.7300 | −30.4028 | −30.4029 |

So the physical DW position moves by **3.793 modes** and its band power by
**7.673 dB** once the comb is contained — both enormous next to the 4.5e-4-mode
and 2.2e-4 containment gaps quoted above, and both converged by `mu_half = 4400`.

This does **not** weaken the conventions conclusion, which is what those rows are
for: a mirror, sign or mode-index-origin error would show up at the scale of
*thousands* of modes, not 4.5e-4, and both codes agreeing that precisely on a
strongly asymmetric dispersion remains strong evidence that neither has one. But
it does mean the CONTAINED verdicts are **not** validation of the physical
dispersive-wave position or amplitude. They establish that the two codes solve
the same truncated problem the same way; the truncated problem is not the
physical one.

### The one SEPARATED verdict

`comb_frac` — the fraction of intracavity energy outside the pump line — is the
single observable for which both ladders converged and the limits do **not**
overlap:

* ours: **0.853371**, observed order 1.10, U = 2.28e-4
* pyLLE: **0.854518**, observed order 0.84, U = 5.47e-4
* gap 1.342e-3 against a combined band of 1.184e-3, i.e. **1.13×** the band.

This is a marginal separation, and it should be read as marginal: a 13% excess
over a two-sigma-style band, with both observed orders below 1.2 (i.e. both
ladders converging slowly, which is exactly the regime in which a Richardson
band is least trustworthy). It is reported as the most informative outcome the
method produced, not as a proven defect in either code. Candidate causes, ranked
by the evidence:

1. **The nonlinear-step rule.** pyLLE uses endpoint-trapezoidal
   `exp((NL(A₀)+NL(A_iter))·dt/2)` iterated to a fixed point; ours uses midpoint
   `exp(i·γ|E_half|²·dt_sub)`. Both are O(dt²) with different error constants,
   and `comb_frac` is precisely an energy-partition observable, which is what a
   nonlinear-phase rule perturbs. This is the leading candidate because the two
   *spectral-position* observables (`μ_DW`, `dw_power_dbc`) are CONTAINED to a
   part in 10⁶ and 10⁵ respectively — the dispersion, detuning and mode-index
   conventions are demonstrably right, so what is left is the nonlinear step.
2. **Slow convergence invalidating the bands.** Observed orders 1.10 and 0.84
   are below the second order both schemes should exhibit, which suggests the
   ladders are not yet in their asymptotic regime at these steps. If so the
   Richardson bands understate the true uncertainty and the separation is
   spurious. Testable by extending both ladders (ours to n = 32/64, pyLLE to
   dt = 1/16/1/32). **This is now the only surviving alternative to (1).**
3. ~~**Grid truncation.**~~ **FALSIFIED — see below.** This was ranked third on
   the reasoning that the comb is not spectrally contained (§6) and `comb_frac`
   integrates the whole spectrum, so it should inherit whatever the wings do at
   the boundary. That reasoning was available to be checked against data already
   in the repository, and it does not survive the check.

**Correction: grid truncation is not the cause of the `comb_frac` separation.**
`validation/results/convergence_lle_dw30k.json`'s `grid_levels` block holds
`comb_frac` at three grid sizes, all at `n_substeps = 8`:

| `mu_half` | 3300 | 4400 | 5500 |
|---|---|---|---|
| `comb_frac` | 0.85370458 | 0.85370879 | 0.85370880 |

The entire grid contribution is **4.21e-6** (3300 → 4400, and 1.0e-8 thereafter,
i.e. converged), against a cross-code limit gap of **1.342e-3**. The gap is
**319× larger than the largest effect grid truncation can have on this
observable**. Widening the grid cannot close it, and the hypothesis is
withdrawn. The remaining ranking is therefore (1) the nonlinear-step rule,
leading, and (2) non-asymptotic ladders, which would make the separation
spurious rather than explain it.

Note that the *related* energy observable `U_mean_w` is INDETERMINATE because
**our** ladder is non-monotone on it (0.192856 → 0.191564 → 0.191314 →
0.191287 → 0.191299 — it turns around at n = 16). Its raw spread is 8.1e-4
relative, and pyLLE's limit 0.193275 sits about 1.0% above our finest value.
The ~1% energy-observable gap recorded in `docs/PYLLE_STATUS.md` is therefore
**still present and still unexplained**, and the honest statement is that this
run could not resolve it either way, because our own ladder does not converge on
it.

---

## 5. Corrections to v1

Each with its evidence. None of these was known when v1 ran; none of them is a
change of tolerance or of an observable after seeing a result.

### (a) pyLLE is refinable; the "not adjustable" claim was wrong

`docs/PYLLE_STATUS.md` recorded `pylle_refinable: false`. `dt` is a plumbed
parameter: `ComputeLLE.jl:123` sets `dt = 1`, and every live use of `dt`
(`Nt = t_ramp/tR/dt` at :129, the drive kick `.*dt` at :348, the linear half
operator `.*dt/2` at :350, the trapezoidal nonlinear phase `.*dt/2` at :362) is
dimensionally consistent with a step of `dt` round trips. The `tol`/`maxiter`
CLI plumbing **already exists upstream** at `:46-47` and is unconditionally
clobbered at `:278-279`. The knob was built and then disabled, not absent.
Evidence: `third_party/pylle/`, and with the patches reverted the vendored
kernel reproduces upstream bit-for-bit (relative L2 = 0.000e+00). This run
exercises four `dt` values × two Picard tolerances.

### (b) The v1 comparison was at two different final detunings

`SaveStatus_CallBack` fires on `it*num_probe/Nt > probe` (`ComputeLLE.jl:232-242`),
which never triggers on the final step. With `Tscan = 5000` and
`num_probe = 200` the last probe was written at round trip **4976 of 5000**, so
pyLLE's returned field sat at **δω = 29.9328 κ** while ours sat at
**30.0000 κ** — 0.0672 κ apart, at a point where `dP_peak/dδω` is steep. v1
attributed that difference to the codes. Patch 0003 overwrites only the final
probe slot; H5 now measures the endpoint match on every run and reads **exactly
0** at all eight pyLLE levels. Evidence: criterion H5 in the v2 JSON;
`tests/test_pylle_crosscheck_v2.py::test_detuning_endpoint_gate_catches_probe_offset`
feeds a synthetic 29.9328 κ result and asserts H5 FAILS.

### (c) The v1 DW convergence attribution reverses at n = 8 and 16

v1 concluded that "refining only our side moves **our** DW index from −3075 to
−3074, i.e. onto pyLLE's value, at both n_substeps = 2 and 4 — so that off-by-one
is our own discretization error". It refined only to n = 4. Applying **v1's own
observable** (`validation.pylle_crosscheck.observables`) to the committed fields
at every level of our ladder:

| n | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| v1 `dw_peak_mu_red` | −3075 | **−3074** | **−3074** | −3075 | −3075 |
| v1 smoothed centroid | −3074.5506 | −3074.1624 | −3074.4590 | −3074.6107 | −3074.6559 |
| fixed-band `mu_DW` | −3074.9846 | −3075.1195 | −3075.2286 | −3075.2469 | −3075.2502 |

The v1 estimator lands on −3074 at exactly the two levels v1 happened to run and
**reverts to −3075 at n = 8 and 16**; its underlying centroid wanders
non-monotonically over ~0.5 modes and does not converge. The fixed-band centroid
converges cleanly (observed order 2.48) to **−3075.2509**, moving *away* from
−3074, and both codes' limits agree to **4.5e-4 modes**. The attribution was an
artefact of a window-and-smoothing estimator, not a measurement.

### (d) The "4.4% from our converged answer" figure is not reproducible

v1 stated "our n_substeps = 1 result sits 4.4% from our converged answer". No
committed artifact yields 4.4%. Measured against our own Richardson limits: the
raw peak at n = 1 is 230.110 W against a limit of 228.241 W (**0.82%**); the
band-limited peak at n = 1 is 233.927 W against 232.809 W (**0.48%**). v1's own
`convergence_attribution` block records drifts of 1.64% (n = 2) and 0.22%
(n = 4) relative to n = 1. The nearest committed number to 4.4% is the *cross-code*
peak-power gap of 4.23%, which is a different quantity. The claim appears to
conflate the cross-code gap with self-convergence.

### (e) The raw peak-power observable carries a several-percent sampling-phase bound

v1 gated `soliton_peak_power` — `max|E_j|²/t_r` on the native grid — at 2%. The
field is band-limited and periodic, so its true maximum is not a grid sample.
Evaluating the exact trigonometric interpolant on a 64× grid and taking the
worst of the 64 possible native sampling phases gives a peak-power loss bound
of **5.08%** for the committed v1 `field_ours` and **6.03%** for the v1
`field_pylle_mapped` (soliton FWHM ≈ 4.0 and 3.8 native samples respectively).
The task brief's figure of **4.02%** is the sech formula evaluated at a scale
parameter of 2.461 samples; the committed field's actual scale parameter is
≈2.25 samples, which is why the direct measurement is larger. **Reported, not
reconciled** — the brief's number and mine differ and I have not resolved which
input is intended. Either way the bound is two to three times the tolerance v1
applied to the observable. The gated quantity is now the 32×-zero-padded
band-limited peak (`P_peak_w`), whose sub-sample offsets this run records as
−0.281 (ours) and −0.313 (pyLLE) native samples.

### (f) The −60 dBc line count's core is identical between codes; the entire disagreement is a wing crossing

v1 gated `comb_line_count_60dbc` at 2% and recorded 3293 vs 3612 (8.83%, FAIL).
Splitting the count by band, **`N60_core` (|μ| ≤ 1500) is exactly 3001 at every
one of the thirteen levels measured across both codes** — five of ours and eight
of pyLLE's, at every step size and both Picard tolerances. All variation lives
in the mid (1500 < |μ| ≤ 3000) and edge (|μ| > 3000) bands, where the spectrum
crosses −60 dBc almost tangentially: the conditioning `dN60/ddB` measured across
those thirteen levels ranges from **85 to 106 lines per dB**, so a 0.1 dB
difference in the wings moves the count by ~10 lines for reasons that have
nothing to do with comb width. At the finest levels this run gives 3324 (ours)
vs 3329 (pyLLE) — 0.15%. `N60` is now DIAGNOSTIC (D1) and can never gate.

### (g) The 3 dB span "0.23% pass" was exactly one mode quantum

v1 reported 443 vs 444 modes, 0.23%, PASS at 2%. The integer extent is quantized
in steps of one mode, so 0.23% *is* the quantum: the criterion could not have
resolved anything finer, and a "pass" at that level carries no information about
agreement. The gated quantity is now the sub-bin span `S3_modes` obtained by
dB-linear interpolation across the −3 dB level, which this run measures at
444.782 (ours, n = 1) and 445.027 (pyLLE, finest tight) — 0.055%. The integer
extent is retained as DIAGNOSTIC D2 with its quantum recorded alongside it.

### (h) The comb is not spectrally contained; the blue DW sits ~61 modes from the grid edge

At μ = +3300 the spectrum sits at **−44.0 dBc** (ours) and **−44.1 dBc**
(pyLLE) at the finest levels; the v1 committed fields give −52.2 and −46.3 dBc.
The containment threshold is −100 dBc, so the comb is running into the boundary
by more than 55 dB of margin. Phase matching `D_int(μ) = δω` has roots at
μ = −3038, −434, +404 and **+3239** — the blue dispersive wave predicted by
phase matching sits **61 modes inside the grid edge**, which is why v1 reported
it NOT MEASURED: there is no room for a peak plus its ±150-mode centroid window.
Every observable that integrates the wings (`N60`, `dw_power_dbc`, and
`comb_frac`) is therefore grid-limited at this `mu_half`. The run emits a
prominent `COMB NOT SPECTRALLY CONTAINED` diagnostic and records the edge levels
in the JSON. **The default `--mu-half` was not raised**; whether the blue DW
becomes resolved at a larger grid is an open measurement, not a claim.

### (i) The v1 run's argv was not recorded and its defaults do not reproduce it

`pylle_crosscheck.json` records no `argv`. Reconstructing from the fields it does
record, the run used `--roundtrips 5000 --dw-start 16 --dw-end 30 --scan
3,5,8,16,30,40,50 --bisect-iters 4` — **every one non-default**. The committed
defaults were 10000 round trips, a 1 κ → 3 κ ramp, a 12-point scan and 5
bisection iterations. The statement in `docs/PYLLE_STATUS.md` that "re-running
the committed script reproduces the numbers" is therefore false as written. v2
records the full `argv` and sets every default to the value the committed run
used, and asserts `defaults_reproduce_this_run` in the JSON and in
`tests/test_pylle_crosscheck_v2.py`.

---

## 6. Results of this run

**Overall: FAIL (QUALIFIED).** All seven HARD checks pass; three GATED criteria
are UNDERIVED (their tolerance could not be derived because a Richardson fit
failed on one side) and demote to diagnostic; both existence-edge criteria FAIL.

| cid | class | criterion | ours | pyLLE | diff | tol | verdict |
|---|---|---|---|---|---|---|---|
| H1 | HARD | parameter round trip | 1.23e-16 | — | — | 1e-12 | **PASS** |
| H2 | HARD | dispersion refit | 2.32e-12 | — | — | 1e-6 | **PASS** |
| H3 | HARD | dispersion mirror applied | true | — | — | — | **PASS** |
| H4 | HARD | pump mode reference | 0 Hz | — | — | 0 | **PASS** |
| H5 | HARD | detuning endpoint match | 0 | — | — | 1e-12 | **PASS** |
| H6 | HARD | δω_eff = programmed | 0 | — | — | 1e-6 | **PASS** |
| H7 | HARD | seed arrays identical | true | — | — | — | **PASS** |
| G1 | GATED | `U_mean_w` | 0.192856 | 0.193204 | 1.80e-3 | — | UNDERIVED |
| G2 | GATED | `comb_frac` | 0.857011 | 0.854892 | 2.47e-3 | 8.60e-3 | **PASS** |
| G3 | GATED | `P_peak_w` | 233.927 | 235.549 | 6.89e-3 | — | UNDERIVED |
| G4 | GATED | `S3_modes` | 444.782 | 445.027 | 5.51e-4 | — | UNDERIVED |
| G5 | GATED | `mu_DW` | −3074.98 | −3075.25 | 0.263 modes | 0.534 modes | **PASS** |
| G6 | GATED | `dw_power_dbc` | −29.789 | −22.567 | 0.242 | 0.25 | **PASS** |
| G7 | GATED | existence edge, lower | 5.8906 κ | 6.1719 κ | 0.281 | 0.133 | **FAIL** |
| G7 | GATED | existence edge, upper | 37.9688 κ | 37.0312 κ | 0.938 | 0.442 | **FAIL** |

Two of those "PASS" verdicts deserve to be read narrowly rather than banked:

* **G6 passes at the ceiling.** Its derived band was `2·√(0.3165² + 0.0079²)` =
  **63%** — dominated by our own 32% uncertainty at n = 1 — and `MAX_TOL` clipped
  it to 25%. The measured 24.2% squeaks under a bar that is *tighter* than the
  data supports, so the pass is real but almost uninformative. The containment
  verdict is where the evidence is: the two **limits** agree to 0.005 dB.
* **The GATED point comparison pairs ours at n = 1 with pyLLE at its finest
  tight level**, because that is the pairing Prompt A's frozen derivation
  attaches uncertainties to (`ours_level="numerical_uncertainty_at_n1"`,
  `pylle_tag="tight"`). It is deliberately not a like-for-like step comparison,
  and G6's 7.2 dB point gap against a 0.005 dB limit gap is the clearest
  illustration of why single-point cross-code comparison is the wrong
  instrument: essentially all of that 7.2 dB is our own n = 1 discretization
  error, which our ladder removes.

### The spectral residual, and the Picard-floor hypothesis falsified

Median (ours − pyLLE) per |μ| band, in dB:

| band | v1 committed fields (n=1 vs dt=1, **different detunings**) | this run, n=1 vs dt=1 shipped (**same detuning**) | this run, n=1 vs dt=1 **tight Picard** | this run, finest vs finest |
|---|---|---|---|---|
| [0, 50) | +0.09 | +0.07 | +0.07 | −0.05 |
| [50, 222) | +0.09 | +0.07 | +0.06 | −0.05 |
| [222, 500) | +0.02 | +0.00 | +0.00 | −0.05 |
| [500, 1000) | −0.26 | −0.28 | −0.28 | −0.05 |
| [1000, 1500) | −1.09 | −1.10 | −1.11 | −0.06 |
| [1500, 2000) | −2.81 | −2.68 | −2.70 | −0.07 |
| [2000, 2500) | −5.80 | −4.98 | −4.97 | −0.05 |
| [2500, 3000) | −9.33 | −5.63 | −5.55 | +0.16 |
| [3000, 3300] | −9.79 | −2.66 | −3.26 | +0.32 |

Three things follow.

1. **The v1 band table is reproduced exactly** from the frozen v1 fields
   (+0.09, +0.09, +0.02, −0.26, −1.09, −2.81, −5.80, −9.33, −9.79), so the
   comparison below is like-for-like.
2. **A large part of the v1 wing gap was the detuning mismatch of correction
   (b)**, not the codes: removing it (same n = 1, same dt = 1, but both at
   30.0000 κ) moves the outermost two bands from −9.33/−9.79 dB to
   −5.63/−2.66 dB.
3. **The gap is a discretization effect that refinement closes.** At each
   ladder's finest level the residual is −0.05 to −0.07 dB out to |μ| = 2500 and
   +0.16/+0.32 dB beyond it. Whatever the wings were doing at dt = 1, it is not
   a persistent disagreement between the codes.

**The standing hypothesis in `docs/PYLLE_STATUS.md` — that the wing gap is
pyLLE's 1e-2 nonlinear tolerance leaving a broadband numerical floor — is
FALSIFIED.** This was R7's designated falsification test: hold `dt = 1` and
tighten only the Picard pair to 1e-8/60, which raises the mean iteration count
from 2.00 to 8.72. If the hypothesis were right the wings would drop toward
ours. The |μ| ≥ 2000 median gap moves from **−4.977 dB to −4.974 dB** — a change
of **0.0024 dB**. The nonlinear tolerance is not what puts pyLLE's wings up; the
step size is.

Note also that the gap does **not** track `|D_int|·t_r`, as v1's framing
implied. The phase exceeds π only for 439 modes, first at |μ| = 2021, and peaks
at 3.303 — yet the finest-level residual is flat at −0.05 dB across bands whose
median phase ranges from 0.008 to 2.6, and turns *positive* in the outermost
band where the phase falls back to 0.68.

### Existence range

Both codes agree on soliton survival at **all seven** scan detunings
(3, 5, 8, 16, 30, 40, 50 κ → dead, dead, alive, alive, alive, dead, dead). The
disagreement appears only under bisection, and both edges now FAIL:

| edge | ours bracket | midpoint ± half-width | pyLLE bracket | midpoint ± half-width | overlap? |
|---|---|---|---|---|---|
| lower | [5.8438, 5.9375] | 5.8906 ± 0.0469 | [6.1250, 6.2188] | 6.1719 ± 0.0469 | **no** |
| upper | [37.8125, 38.1250] | 37.9688 ± 0.1563 | [36.8750, 37.1875] | 37.0312 ± 0.1563 | **no** |

The brackets are disjoint and the midpoints are 0.281 κ and 0.938 κ apart
against combined bands of 0.133 κ and 0.442 κ. v1 compared `br[1]` — the
tightest surviving point — on each side, reported 5.94 vs 6.31 κ (5.94%, FAIL at
5%) and 37.50 vs 36.88 κ (1.67%, **PASS**), and so recorded the upper edge as
agreement. Comparing intervals rather than biased endpoints shows both edges
disagree.

**Hold-time sensitivity (R8): the settle-time explanation is falsified.**
Re-classifying each edge midpoint at 1×, 2× and 4× the 2000-round-trip hold does
not move a single classification:

| edge / code | midpoint | 1× | 2× | 4× | moved? | peak-power spread |
|---|---|---|---|---|---|---|
| lower / ours | 5.8906 κ | dead | dead | dead | no | 2.7% |
| lower / pyLLE | 6.1719 κ | alive | alive | alive | no | 22.6% |
| upper / ours | 37.9688 κ | alive | alive | alive | no | 5.8% |
| upper / pyLLE | 37.0312 κ | dead | dead | dead | no | 6.7% |

So the lower-edge disagreement is **not** an artefact of insufficient settling,
which was the leading untested alternative. The classification is robust to a 4×
longer hold on both sides. (The peak-power spreads in the last column are large
— up to 23% — so the *states* at the edges are still evolving even though their
*labels* are stable. That is a caveat on the observable, not on the edge.)

**Stationarity (R8).** Classifying at the final round trip and at
final−1, −2, −5, −10 flags a point NON_STATIONARY if the label changes or the
peak power varies by more than 1%:

| δω/κ | Kerr phase (rad/rt) | ours | pyLLE |
|---|---|---|---|
| 3 | 0.037 | NON_STATIONARY (12.7%) | NON_STATIONARY (22.5%) |
| 5 | 0.062 | STATIONARY | STATIONARY |
| 8 | 0.099 | NON_STATIONARY (3.1%) | NON_STATIONARY (1.0%) |
| 16 | 0.199 | STATIONARY (0.29%) | NON_STATIONARY (1.0%) |
| **30** | **0.373** | STATIONARY (0.27%) | **NON_STATIONARY (1.2%)** |
| 40 | 0.497 | STATIONARY (0.16%) | NON_STATIONARY (1.6%) |
| 50 | 0.621 | STATIONARY (0.85%) | NON_STATIONARY (1.1%) |

pyLLE's state is not stationary at the 1% level at the main operating point
(30 κ) under a constant hold, while ours is (0.27%). That matters because the
residual cross-code disagreements are themselves of order 1%: at 30 κ a snapshot
comparison against pyLLE is comparing to a state that is still moving by about
as much as the quantity being compared.

---

## 7. Still unexplained

**The lower existence edge.** Ours is at 5.8906 ± 0.0469 κ, pyLLE's at
6.1719 ± 0.0469 κ; the brackets are disjoint with a clear gap between them. The
Kerr phase there is ≈ 0.073 rad per round trip — a twentieth of a radian — so
the step-size explanation that covers the wing residual and the DW index cannot
apply: at that phase one step per round trip is accurate, and indeed the two
codes' spectra at low detuning agree to a few hundredths of a dB. The leading
alternative, insufficient settle time, is now **falsified** by the hold-time
check above: 4× the hold moves neither code's classification.

I do not have an explanation for it. What can be said:

* it is not a convention error — every HARD check passes, and the two codes'
  `μ_DW` limits agree to 4.5e-4 modes on a strongly asymmetric dispersion;
* it is not the discretization at that operating point, by the Kerr-phase
  argument;
* it is not settle time, by direct measurement;
* the classifier itself (`is_single_soliton`: contrast ≥ 5 and exactly one
  connected component above the half-way level) is a threshold test, and near an
  existence boundary a threshold test can flip on a small difference in a state
  that is genuinely marginal. The contrast values recorded at every evaluation
  make such a flip visible, and at the edges the peak-power spread over the last
  ten round trips reaches 23% for pyLLE — which is consistent with, though not
  proof of, a marginal state being classified differently by the two codes for
  reasons that are not physical.

The most promising next measurement is to replace the binary classifier at the
edges with a continuous order parameter (peak-to-background contrast, or the
comb energy fraction) and locate the edge as a level crossing of that, which
would remove the threshold flip as a candidate. That is a change to an
observable and therefore out of scope for this task.

### Ours-side edge discretization, measured

*Added after the above was written.* The paragraph above stands as written —
the sentence "I do not have an explanation for it" was true of what was then
known, and one of its four bullets is now wrong. The second bullet claimed the
disagreement "is not the discretization at that operating point, by the
Kerr-phase argument". **That argument was too quick, and the measurement
contradicts it.**

The v2 existence bisection ran at `n_substeps = 1` only, so G7 was the one GATED
criterion whose band contained no discretization term at all. Refining it
(`validation/existence_convergence.py`, our solver only; pyLLE not run):

| `n_substeps` | 1 | 2 | 4 | drift 1→4 |
|---|---|---|---|---|
| lower edge (κ) | 5.890625 | 5.890625 | **6.078125** | **0.1875** |
| upper edge (κ) | 37.96875 | 37.96875 | 37.96875 | 0.0 |

Bracket half-widths are 0.046875 κ (lower) and 0.15625 κ (upper) at every level.
At `n_substeps = 1` this reproduces the v2 ours-side existence block **exactly** —
same survival vector, same brackets, same midpoints, same bisection trace — which
is the regression gate that makes the rest interpretable.

So the lower edge **does** move under refinement, by 0.1875 κ, and it moves
**toward pyLLE**: our estimate goes 5.8906 → 6.0781 κ against pyLLE's 6.1719 κ,
and the gap falls from 0.281 κ to 0.094 κ. The Kerr-phase argument — that at
0.073 rad per round trip one step per round trip must already be accurate — is a
statement about the *field*, and it does not transfer to the *edge*: the edge is
where a marginal state is classified by a threshold, and an arbitrarily small
change in a marginal state can move it. The upper edge, by contrast, does not
move at all across the ladder.

Discretization uncertainties, with `U_disc = max(drift_2→4, half-width(finest))`:
**lower 0.1875 κ** (from the drift), **upper 0.15625 κ** (floored at the
half-width, since the drift is zero). Richardson on the three-point sequence
returns `NON_MONOTONE` for both edges — a flat-then-jump sequence has no order to
fit — so the fallback is used and no fit is forced.

Recomputing G7 with the refined bracket and the measured term, decomposed so the
cause of any verdict change is visible:

| edge | variant | \|ours−pyLLE\| | band | overlap | verdict |
|---|---|---|---|---|---|
| lower | v2 as measured (n=1, no U_disc) | 0.28125 | 0.13258 | no | **FAIL** |
| lower | refined (n=4), no U_disc | 0.09375 | 0.13258 | **yes** | **PASS** |
| lower | refined (n=4) + U_disc | 0.09375 | 0.39775 | yes | **PASS** |
| upper | v2 as measured (n=1, no U_disc) | 0.9375 | 0.44194 | no | **FAIL** |
| upper | refined (n=4), no U_disc | 0.9375 | 0.44194 | no | **FAIL** |
| upper | refined (n=4) + U_disc | 0.9375 | 0.54127 | no | **FAIL** |

The lower edge's FAIL → PASS is attributable to **the refined measurement, not to
the widened band**: it passes at the *original* band once our own edge is
measured properly, and indeed the two brackets now overlap
(ours [6.03125, 6.125], pyLLE [6.125, 6.21875] — they meet exactly at 6.125 κ,
which is a knife-edge overlap and should be read as such). The band term is not
what rescued it. The upper edge fails under all three variants; its 0.9375 κ gap
is not a discretization effect, since the upper edge does not move at all.

Two honesty notes on the recomputed band. It is **asymmetric**: pyLLE's own edge
was not refined by this task, so it contributes no discretization term; a
pyLLE-side term could only widen the band further, which means a FAIL under this
band is conservative and a PASS is not. And for the upper edge `U_disc` is
floored at the bracket half-width, so that half-width enters the quadrature
twice — again making the band wider than strictly justified, and again meaning
the upper edge's FAIL is robust.

**Pre-committed outcome: H-C** (intermediate). The deciding number is the
largest drift across the two edges, 0.1875 κ, which falls between the H-B
threshold of 0.05 κ and the H-A threshold of 0.28 κ fixed before the run.

A secondary axis repeated the `n = 4` bisections at 2× the hold (4000 round
trips): both brackets came back identical, so at the refined step the edges are
insensitive to settle time as well.

### The classifier is not operating marginally — the fourth bullet is weakened too

The fourth bullet above named the threshold classifier as the leading remaining
suspect, on the reasoning that near a boundary a threshold test can flip on a
small difference in a marginal state. Recording the contrast at **every**
evaluation — which is what R8 of this task required, precisely so that a
threshold flip would be visible — shows that it is not happening here. Across
all 30 bisection evaluations of the three levels:

| | contrast range | n |
|---|---|---|
| classified ALIVE | 19.56 … 15074 | 17 |
| classified DEAD | 1.01 … 1.70 | 13 |

The decision threshold is **contrast = 5**, and it sits inside an **empty band
spanning a factor of 11.5×** (1.70 → 19.56). Not one of the 30 evaluations lands
anywhere near it. The transition is sharp and bimodal: either there is a soliton
(contrast ≥ 20, one peak) or the field is flat CW (contrast ≈ 1, zero peaks).

The decisive case is the lower edge at δω = 5.9375 κ, where refinement changes
the answer: `n = 1` gives contrast **61.33** with one peak (a clear soliton),
`n = 4` gives contrast **1.70** with zero peaks (no soliton at all). That is not
a classifier flipping on a borderline state — it is the soliton failing to form.
Whether a soliton survives at that detuning genuinely depends on the step size.

So the classifier is much less plausible as the explanation than it looked, and
the continuous-order-parameter measurement proposed above would probably
reproduce the same edges rather than dissolve the disagreement. It remains worth
doing, but as a confirmation rather than as the leading hypothesis.

**What is now known, and what is still not.** The lower-edge disagreement is
substantially — though not entirely — a resolution limit of our own measurement:
two thirds of the original 0.281 κ gap disappears on refining our side, and what
remains overlaps within the brackets. The upper-edge disagreement, which v1
reported as a PASS and v2 as a FAIL, survives refinement unchanged, is not a
classifier artefact, and is **still unexplained**. Two candidates remain for it,
neither tested: pyLLE's own edge may move under *its* refinement (not measured —
this task ran no Julia), or the two codes genuinely differ on where the soliton
ceases to exist at high detuning. Refining pyLLE's edge over its `dt` ladder is
the obvious next measurement and would also remove the asymmetry in the
recomputed band.

**The ~1% energy-observable gap.** `U_mean_w` remains INDETERMINATE because our
own ladder is non-monotone on it, and `comb_frac` is the one SEPARATED verdict.
Both are energy-partition quantities and both are consistent with the
nonlinear-step-rule hypothesis of §4, which remains untested.

---

## 8. What would falsify these conclusions

| conclusion | what would overturn it |
|---|---|
| The conventions are right (Findings 1–9) | Any HARD check failing on a re-run: a parameter round trip above 1e-12, a dispersion refit above 1e-6, a pump-frequency mismatch above 0 Hz, or a seed hash mismatch. Or: the two codes' `μ_DW` limits ceasing to agree when the dispersion is changed to a *different* asymmetric profile — agreement on one profile could in principle be a coincidence of that profile. |
| `μ_DW` and `dw_power_dbc` are CONTAINED | Extending either ladder (ours to n = 32/64, pyLLE to dt = 1/16) and finding the limits move outside the current bands, i.e. the observed orders 2.48 and 3.18 are not asymptotic. |
| `comb_frac` is SEPARATED | Either: (i) extending both ladders and finding the observed orders rise toward 2, which would shrink the bands' credibility and could merge the limits; or (ii) implementing the endpoint-trapezoidal nonlinear rule in our solver and finding it reproduces pyLLE's limit — which would also confirm the ranked cause. **Raising `--mu-half` will not do it**: the grid contribution to `comb_frac` is 4.21e-6 against a 1.342e-3 gap (§4), so that route is closed. |
| The wing gap is discretization, not a Picard floor | A run at `dt = 1` with tight Picard showing a wing-gap change materially larger than the 0.0024 dB measured. The test is cheap and is re-run every time. |
| The **lower** existence-edge disagreement is real | Largely already overturned: refining our own side moves the lower edge 0.1875 κ toward pyLLE and the brackets then overlap (§7). What remains would be settled by refining **pyLLE's** edge over its own `dt` ladder — not done here, which is why the recomputed band is asymmetric — or by raising `bisect_iters` so the two brackets no longer merely touch at 6.125 κ. |
| The **upper** existence-edge disagreement is real | Unmoved by everything tested so far: it survives `n_substeps` 1→4 (zero drift), 4× hold at n=1, and 2× hold at n=4. It would be overturned by refining pyLLE's upper edge and finding *it* moves by ≈0.9 κ. Replacing the binary classifier with a continuous order parameter is worth doing but is now unlikely to overturn it: no evaluation lands near the decision threshold (§7 — ALIVE contrast ≥ 19.6, DEAD ≤ 1.70, threshold 5). |
| The edges are resolution-limited rather than genuinely different | A drift measurement at `n_substeps` = 8 or 16 that moves either edge further. The ladder here stops at 4; the lower edge's jump occurs between 2 and 4 and has not been shown to have settled. |
| The v1 attribution of the DW index was an estimator artefact | Finding that `validation.pylle_crosscheck.observables` returns −3074 at n = 8 or 16 after all, i.e. that the table in correction (c) is wrong. It is regenerable from `validation/results/convergence_lle_dw30k_fields.npz` in seconds. |
| The run is reproducible | Two runs with the same argv producing different `numerical_digest` values. |

---

## 9. Where quantitative verification rests

In order of strength, unchanged by this exercise:

1. **`validation/analytic_cw.py`** — the solver against *exact mathematics* (the
   cubic root and the discrete map's fixed point), agreeing to ~1e-14 and gated
   at 1e-12. This is a claim about correctness against a closed-form answer and
   nothing here weakens or replaces it.
2. **`validation/mms.py` and `validation/convergence.py`** — observed order of
   accuracy against manufactured solutions. This is a claim about the
   discretization being what it says it is, verified without reference to any
   other code.
3. **This cross-check** — convention agreement and envelope containment. It is
   the weakest of the three and remains so. Its value is categorical rather than
   quantitative: it is the only one of the three that could have caught the
   dispersion mirror (Finding 2b) or the 7-FSR pump offset (Finding 8), because
   both are errors in the *setup* of a problem that every single-code test would
   then verify perfectly.

What v2 adds to that third item is that it now measures what it claims to
measure: tolerances derived from each code's own convergence data rather than
chosen, a containment test that compares limits rather than coarse snapshots,
and a set of HARD checks that make "the two codes were solving the same problem"
a gated assertion rather than an assumption. The v1 run's positive conclusions
about conventions survive; its interpretations of the residual disagreements do
not.
