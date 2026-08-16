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

The accuracy floor is **pyLLE's, and it is not adjustable**: `ComputeLLE.jl:123`
hardwires `dt = 1` round trip, and `:278` overrides the CLI nonlinear-iteration
tolerance with `1e-2`. At an operating point short enough to excite this
device's dispersive waves the Kerr phase reaches ~0.37 rad per round trip, where
one step per round trip is several percent from converged — measured directly by
refining our own side (`convergence_attribution` in the JSON). Sub-2% cross-code
agreement is therefore not attainable there by either code, which is a fact
about pyLLE as a reference, not about this repo's solver.

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
