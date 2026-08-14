# pyLLE cross-check: BLOCKED (no Julia toolchain reachable)

Attempt date: 2026-08-14. Working tree: branch `claude/stochastic-lle-benchmark-xgz7b7`,
`git rev-parse HEAD` = `ca7907ed25ac5b85566e1f2b068a4691b2e53595`, clean at the time of the
attempt.

**Outcome: `validation/pylle_crosscheck.py` was NOT written.** pyLLE cannot execute in this
environment, so a cross-check script would have no reference to check against. Writing one
that has never been run against a real pyLLE result would be worse than not writing it — it
would look like validation infrastructure while certifying nothing. This document records the
blocker instead, per the task's own fallback clause.

**No source file was modified.** Nothing was installed into the project environment: pyLLE and
its dependencies were installed into a throwaway virtualenv under the session scratchpad, which
is discarded with the container. `requirements.txt` is untouched.

Reference for the physics conventions discussed below: Herr, Tikan & Kippenberg,
**arXiv:2604.05897v1** (7 Apr 2026).

---

## 1. The blocker

pyLLE is a Python front-end over a **Julia** numerical kernel. It writes an HDF5 parameter
file to a temp dir, shells out to a `julia` binary to run `ComputeLLE.jl`, and reads an HDF5
result file back. Without a Julia binary there is no solver at all — the Python side computes
nothing numerical on its own.

Julia is not present in this image and **cannot be obtained**: the session's egress policy
denies the Julia distribution hosts at the proxy (HTTP 403 on CONNECT). Per
`/root/.ccr/README.md`, a 403/407 from the proxy is an organization policy denial that must be
reported rather than retried or routed around, so I stopped rather than looking for a mirror.

Recorded verbatim by the proxy's own status endpoint
(`curl -sS "$HTTPS_PROXY/__agentproxy/status"`, field `recentRelayFailures`):

```json
[
  {
    "ts": "2026-08-14T23:42:48.860Z",
    "kind": "connect_rejected",
    "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
    "host": "julialang-s3.julialang.org:443"
  },
  {
    "ts": "2026-08-14T23:43:20.331Z",
    "kind": "connect_rejected",
    "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
    "host": "julialang-s3.julialang.org:443"
  },
  {
    "ts": "2026-08-14T23:43:20.604Z",
    "kind": "connect_rejected",
    "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
    "host": "install.julialang.org:443"
  }
]
```

This is a hard environmental limit, not slow progress: the 3-hour timebox was never the
operative constraint. The attempt was abandoned after ~25 minutes because no additional
debugging time can produce a Julia binary.

### What was tried

| # | Attempt | Result |
|---|---------|--------|
| 1 | `julia --version` | `julia: command not found` |
| 2 | `find / -maxdepth 4 -name 'julia*'` | no hit; `/opt` holds maven/gradle/node/ruby/bun, no Julia |
| 3 | `apt-get install -s julia` | `E: Package 'julia' has no installation candidate` |
| 4 | `curl` official tarball, `julialang-s3.julialang.org` | `curl: (56) CONNECT tunnel failed, response 403` |
| 5 | `curl https://install.julialang.org` (juliaup) | `curl: (56) CONNECT tunnel failed, response 403` |
| 6 | `curl` GitHub release asset `JuliaLang/julia v1.10.9` | HTTP 404 (session's GitHub proxy is scoped to `lucastiger/soliton-control`) |
| 7 | `pip download pyLLE` | **succeeded** — PyPI is reachable; pyLLE 4.1.2 |

So the Python half is installable and the Julia half is not.

### Versions

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04.4 LTS, Linux 6.18.5-fc-v20 |
| Python | 3.11.15 |
| pyLLE | 4.1.2 (sdist `pyLLE-4.1.2.tar.gz`, from PyPI) |
| numpy | 1.26.4 (see §2 — pinned deliberately) |
| scipy | 1.17.1 |
| h5py | 3.16.0 |
| Julia | **absent, unobtainable** |

---

## 2. Second, independent blocker: pyLLE 4.1.2 is not NumPy-2 compatible

Found while driving the repro. It is separate from the Julia problem and would have to be
fixed even on a machine with Julia, so it is recorded here.

`pyLLE/_llesolver.py:697` (and `:721`) call `np.string_`, removed in NumPy 2.0:

```
  File ".../pyLLE/_llesolver.py", line 721, in SetupHDF5
    it = np.string_(it)
  File ".../numpy/__init__.py", line 778, in __getattr__
    raise AttributeError(
AttributeError: `np.string_` was removed in the NumPy 2.0 release. Use `np.bytes_` instead.
```

Workaround used: pin `numpy<2` (resolved to 1.26.4) **in the throwaway venv only**. This is a
dependency constraint, not a patch to site-packages. Note this is a genuine conflict to plan
around: pinning `numpy<2` for pyLLE inside the project environment would be a real constraint
on this repo, so any future cross-check should run pyLLE in a **separate** virtualenv and
exchange data through files rather than importing pyLLE alongside the JAX solver.

## 3. Exact terminal failure

With `numpy<2` pinned, pyLLE's Python front-end runs **end-to-end** at this repo's committed
SiN parameters — it parses the dispersion file, fits `Dint`, prints its parameter table, and
successfully writes `ParamLLEJulia.h5`. It then dies at the Julia launch:

```
-- Solving standard LLE --
	Simulation Parameters
		R = 929.80 µm
		Qi = 40.00 M
		Qc = 10.00 M
		γ = 0.29
	Simulation Parameters
		Pin[0] = 214.00 mW
		f_pmp[0] = 193.41 THz
		Tscan = 0.10 x1e6 Round Trip
		μ_sim = [-100.00,100.00]
		μ_fit = [-150.00,150.00]
		δω_init = -2.00 x2π GHz
		δω_end = 6.00 x2π GHz
		δω_stop = 6.00 x2π GHz
		ind_pump_sweep[0] = 0.00

HDF5 parameter file can be foud in: /tmp/tmp9dql21toParamLLEJulia.h5
=== SolveTemporal (reaches Julia) ===
----------------------------------------------------------------------
2026-08-14 23:46:02
[Errno 2] No such file or directory: '/opt/bin/julia'
Traceback (most recent call last):
  File ".../pyLLE/_llesolver.py", line 777, in LaunchJulia
    self.JuliaSolver = sub.Popen(julia, stdout=sub.PIPE, stderr=sub.PIPE)
  File "/usr/lib/python3.11/subprocess.py", line 1026, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
  File "/usr/lib/python3.11/subprocess.py", line 1955, in _execute_child
    raise child_exception_type(errno_num, err_msg, err_filename)
FileNotFoundError: [Errno 2] No such file or directory: '/opt/bin/julia'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File ".../pyLLE/_llesolver.py", line 888, in SolveTemporal
    JuliaLog, start_time = LaunchJulia(bin)
  File ".../pyLLE/_llesolver.py", line 780, in LaunchJulia
    raise ValueError(
ValueError: ('julia is not installed on the system path. ', 'Please add julia to the path or re-install ', 'and check the option to add to path.')
```

The hardcoded default path is `/opt/bin/julia` (`_llesolver.py:772`); it is overridable by
passing `bin=` to `SolveTemporal`, so no site-packages edit is needed once a binary exists.

---

## 4. Salvage: the convention audit, done from source

The task's Part 1 warned that the parameter translation is where this goes wrong, and asked
for the `D_int` unit and `mu` conventions to be **confirmed from the pyLLE source, not
assumed**. That audit does not require running pyLLE, so it was completed and is recorded here
to make a future attempt cheap. All line numbers are pyLLE 4.1.2.

Everything in this section is **derived from reading source, and NOT confirmed by execution**
— exactly the runtime check that could not be run. Treat it as a starting hypothesis to be
verified the moment a Julia binary is available, not as an established result.

pyLLE's kernel (`ComputeLLE.jl:338-344`) integrates, in units where time is the round trip
`tR` and `dt = 1`:

```
FFT_Lin(it) = -α/2 + 1im*(Dint_shift - δω_all[1][it])*tR
NL(uu)      = -1im*(γ*L*abs(uu)^2)
```

**Finding 1 — `D_int` is in rad/s (angular), not Hz.** `α = κ0 + κext` with
`κext = ω0/Qc*tR` (`ComputeLLE.jl:88`) is dimensionless, and `Dint` and `δω` are both
multiplied by `tR` to become dimensionless. Both are therefore angular frequencies in rad/s.
This matches our `d_int_grid` and `delta_omega` units directly — **no 2π conversion.**

**Finding 2 — pyLLE's field is the COMPLEX CONJUGATE of ours, and its detuning sign is
OPPOSITE.** Ours (per `validation/analytic_cw.py`): linear `exp((-kappa/2 - i*D_int -
i*delta_omega)*dt/2)`, Kerr `exp(+i*gamma*|E|^2*dt)`. pyLLE: `+i*Dint`, `-i*δω`,
`-i*γL|A|²`. Conjugating pyLLE's equation and matching term by term gives

```
A_pyLLE  =  conj(E_ours)      with    Dint_pyLLE = D_int_ours
                                      δω_pyLLE   = -delta_omega_ours
```

i.e. pyLLE's `δω` is `omega_pump - omega_res`, opposite to our
`delta_omega = omega_res - omega_pump`. **This is the trap, and it is precisely the one that
would corrupt observable #1.** Conjugation mirrors the spectrum, `μ -> -μ`. Comparing *DW peak
mode indices* with an EXACT-match requirement — as Part 2 specifies — would fail, or worse,
*spuriously pass* on a dispersion profile symmetric enough to hide the mirror. Any future
cross-check must apply the conjugation explicitly and assert the mirror direction on an
intentionally asymmetric `D_int`, or the "exact match" test certifies nothing.

**Finding 3 — the `mu` grid is pump-relative, and pyLLE re-zeros `D_int` at the DOMAIN
CENTER.** `_llesolver.py:470-472` builds `mu_sim_center = mu_sim + ind_pmp[0]`; the kernel
(`ComputeLLE.jl:106`) takes `μ0` as the midpoint of the domain and then forcibly applies
`Dint = Dint .- Dint[μ0]` (`:115`). Two consequences: any `D_int` offset we supply is
discarded, and if the pump is not at the domain centre the re-zeroing happens at the **wrong
mode**. Keep the pump centred (`ind_pmp = 0`) and supply `D_int` already zeroed at the pump.

**Finding 4 — pyLLE 4.1.2 is already fully deterministic, but for a reason that creates a
worse problem.** Part 2 asked whether pyLLE seeds a random background field. It does not:

* the kernel defines a `Noise()` function (`ComputeLLE.jl:172`) but **its call site is
  commented out** — `Fdrive` ends `return Force #.- 1im*Noise()` (`:274`). The drive carries no
  stochastic term.
* the initial field `DKS_init` defaults to `np.zeros(...)` (`_llesolver.py:479`), not to a
  random field.

So no seed needs fixing — but the consequence is that **pyLLE started from its own defaults
can never form a soliton.** With a single-mode drive, an exactly-zero initial field and no
noise, the state remains exactly CW: there is no seed to break the symmetry, so modulational
instability has nothing to amplify. A future cross-check must supply an explicit deterministic
`DKS_init` (e.g. a sech ansatz, or a fixed-seed field written once and committed) on the pyLLE
side, and seed our solver **identically**, or the two codes are not being asked the same
question. This is a substantive design constraint on Part 2 that was not anticipated in the
task description.

**Finding 5 — pyLLE takes dispersion as a resonance-frequency file, not a `D_int` array.**
`res['dispfile']` is a CSV of `azimuthal_mode_order, resonance_frequency_Hz`, from which
`AnalyzeDisp.GetDint` fits `D_int` by cubic spline (`_analyzedisp.py:63-64`). Our `d_int_grid`
must therefore be *inverted* into resonance frequencies before handing it over, and the spline
refit means the `D_int` pyLLE actually integrates is **not bit-identical** to ours. That refit
error is a floor on any agreement claim and must be measured (compare `Dint_sim` returned by
`Analyze` against our grid) rather than assumed negligible.

**Finding 6 — detuning ramp.** As the task anticipated, pyLLE sweeps internally: it builds a
linear ramp `δω_init -> δω_end` over `Nt = Tscan` round trips (`ComputeLLE.jl:157`) and
cannot be handed an arbitrary array. Matching therefore has to be done as the task specifies —
same final detuning after an equally-long linear ramp — with our `delta_omega` array
constructed to be the linear ramp pyLLE would have produced, **negated** per Finding 2.

A working repro that exercises all of the above up to the Julia call is small enough to
reconstruct from this document; it built a synthetic `dispfile` from `fsr_hz` and
`d2_rad_per_s2` and drove `Analyze -> Setup -> SolveTemporal`.

---

## 5. Where verification actually rests

With the cross-check blocked, the repo's verification of the deterministic solver rests on two
existing modules — and this is a **stronger** position than cross-code agreement, not a
weaker one:

* **`validation/analytic_cw.py`** — compares the solver's homogeneous steady state against
  *exact mathematics*: the cubic root of the continuum CW state, and the fixed point of the
  discrete map the solver actually iterates. The discrete-map target is matched to ~1e-14 and
  the CLI gates at 1e-12, over 405 sweep points.
* **`validation/convergence.py`** and **`validation/mms.py`** — establish the *observed order
  of accuracy* against manufactured solutions, i.e. that the discretization converges to the
  PDE it claims to solve, at the rate it claims.

Cross-code agreement is a weaker claim than either. Two codes agree only to the level at which
they discretize the same PDE — typically 1e-3 or worse, as `analytic_cw.py` notes in its own
docstring — and agreement between two independent codes cannot distinguish "both right" from
"both wrong in the same way", which is a live risk when both implement the same standard
split-step scheme. `analytic_cw.py` already demonstrates the value of the exact-mathematics
route concretely: it did not merely pass, it *quantified* a real 3e-3 first-order mean-field
splitting error that a 2%-tolerance cross-check against pyLLE would have sailed straight past.

What a pyLLE cross-check would genuinely add is narrower and worth stating honestly: an
independent check on **convention and bookkeeping** — sign of the detuning, direction of the
dispersion, factors of 2π — precisely the class of error that Findings 2, 3 and 5 above show is
easy to make. That is real value, but it is a check on translation, not on numerics, and the
audit in §4 captures most of it already.

## 6. To resume

1. Obtain a Julia binary — requires the egress policy to allow `julialang-s3.julialang.org`
   and `install.julialang.org` (or a pre-baked Julia in the image). **This is an
   infrastructure decision, not a code change.**
2. `julia InstallPkg.jl` (ships in the pyLLE sdist) to precompile `HDF5`, `FFTW`,
   `LinearAlgebra`. Expect several minutes on first run — and note this needs Julia's package
   registry (`pkg.julialang.org`) to be reachable too, which is a *second* egress requirement
   beyond the binary.
3. Run pyLLE in its own virtualenv with `numpy<2` (§2).
4. Implement the translation table from §4, and make Findings 2 and 4 explicit assertions
   rather than comments — the conjugation/mirror check and the shared `DKS_init` seed are the
   two things most likely to silently invalidate the comparison.
