# Discretization uncertainty of the LLE solver at the DW operating point

Produced by `validation/convergence_lle.py`; raw artifacts in
`validation/results/convergence_lle_dw30k.{json,npz}` and the figure in
`validation/figures/fig_convergence_lle.png`. Nothing in this study runs or
refers to pyLLE; no other code is treated as ground truth.

Reference for the physics: Herr, Tikan & Kippenberg, **arXiv:2604.05897v1**
(7 Apr 2026).

**A1 regression gate: PASSED, relative L2 = 0.0 (bit-identical).** The
`n_substeps = 1` level reproduces `pylle_crosscheck_fields.npz::field_ours`
exactly, so the harness is demonstrably driving the same solver on the same
problem as the committed cross-check.

**Determinism (T3): PASSED.** The study was run twice end to end and the two
JSONs were compared field by field: **zero numerical fields differ**. The only
differences are per-level `wall_s` / `wall_clock_total_s`, `timestamp_utc`, the
git SHA (a commit landed between the runs), the `mkdtemp` path of the derived
config, and `argv` (the second run necessarily carries `--out-tag`). Every
observable, order, limit and uncertainty is reproduced bitwise.

---

## 1. The operating point, and why it is the difficult one

Measured SiN dispersion (`config/pyLLE_dispersion_w4400_h800.csv`), mu in
[-3300, +3300] (6601 modes), 5000 round trips, delta_omega ramped
16 kappa -> 30 kappa, deterministic sech seed on the lower CW branch, one step
per round trip, every numerical aid off (`symmetric_drive`, `edge_absorber`,
`dealias_two_thirds`, `dispersion_validity_mask` all `False`).

| quantity | value | why it matters |
|---|---|---|
| Kerr phase per round trip | **0.3727 rad** | not a small angle for a first-order splitting |
| soliton width `w` | **2.461 samples** (FWHM 4.34) | `max\|E_j\|^2` is sampling phase as much as physics |
| `max\|D_int\|*t_r` | **3.3030 rad** | exceeds pi |
| modes with `\|D_int\|*t_r > pi` | **439**, first at `\|mu\| >= 2021` | splitting commutator error is O(1) there |
| spectral edge at n=1 | **-52.17 dBc** (blue), -85.65 dBc (red) | the comb is **not** spectrally contained |
| dispersion fully measured | **False** — 39 modes (mu -3300..-3262) extrapolated | CSV stops at mu = -3261 |
| DW phase-matching roots | **mu = -3038 and +3239** | +3239 is only 61 modes inside the grid |

The linear operator is applied *exactly* in Fourier space, so `|D_int|*t_r > pi`
is not by itself an error — but the splitting commutator is O(1) in that regime
and 439 modes of the grid sit there. Per the study's terms of reference, no
mask or absorber was enabled to "fix" this: doing so would change the reference
configuration.

---

## 2. Results

Substep ladder, `n_substeps` in (1, 2, 4, 8, 16); Richardson on the three finest
levels; `U_obs = max(GCI, |f1-f2|/|f1|)`; the two right-hand columns are the
relative distance of that level from the extrapolated limit `q*`.

| observable | n=1 | n=2 | n=4 | n=8 | n=16 | p | q* | U(n=1) | U(n=8) |
|---|---|---|---|---|---|---|---|---|---|
| `P_peak_w` | 233.927 | 233.936 | 233.138 | 232.875 | 232.822 | 2.308 | 232.809 | **0.4801%** | 0.0285% |
| `P_peak_raw_w` | 230.110 | 233.936 | 230.615 | 228.899 | 228.423 | 1.852 | 228.241 | 0.8187% | 0.2880% |
| `S3_modes` | 444.782 | 445.020 | 445.046 | 445.039 | 445.036 | 1.563 | 445.035 | 0.0568% | 0.0009% |
| `S3_int_modes` | 443 | 445 | 445 | 445 | 445 | — | — | — | — |
| `mu_DW` | -3074.98 | -3075.12 | -3075.23 | -3075.25 | -3075.25 | 2.484 | -3075.251 | 0.0087% | 0.0001% |
| `dw_power_dbc` | -29.789 | -24.013 | -23.028 | -22.730 | -22.654 | 1.960 | -22.627 | **31.65%** | 0.4548% |
| `U_mean_w` | 0.192856 | 0.191564 | 0.191314 | 0.191287 | 0.191299 | — (non-monotone) | — | 0.0139% | 0.0139% |
| `comb_frac` | 0.857011 | 0.854963 | 0.854087 | 0.853705 | 0.853526 | 1.102 | 0.853371 | 0.4266% | 0.0391% |
| `N60` | 3293 | 3357 | 3342 | 3330 | 3324 | — | — | — | — |
| `dN60_ddB` | 85 | 96 | 96 | 93 | 92 | — | — | — | — |
| `N60_core` | **3001** | **3001** | **3001** | **3001** | **3001** | — | — | — | — |
| `N60_mid` | 255 | 255 | 250 | 248 | 246 | — | — | — | — |
| `N60_edge` | 37 | 101 | 91 | 81 | 77 | — | — | — | — |
| `peak_subsample_offset` | -0.281 | 0.000 | -0.219 | -0.281 | -0.281 | — | — | — | — |
| `S3_n_runs` | 3 | 3 | 3 | 3 | 3 | — | — | — | — |

`N60_core` is **constant at 3001 across every level**: the entire variation in
the line count lives in the wings (`N60_edge` swings 37 -> 101 -> 77).

---

## 3. Did each observable converge?

* **`P_peak_w` — YES.** p = 2.31, cleanly in the second-order window. The
  band-limited definition is what makes this possible; see `P_peak_raw_w`.
* **`P_peak_raw_w` — yes, but on a contaminated quantity.** p = 1.85 and it
  converges to a *different* limit (228.24 W vs 232.81 W, 2.0% low) because the
  native-grid maximum systematically under-reads a 4.34-sample-FWHM pulse. Its
  level sequence is non-monotone in the raw values (230.11 -> 233.94 -> 230.61)
  and `peak_subsample_offset` shows why: the n=2 level happens to land the pulse
  exactly on a sample (offset 0.000) and the others do not.
* **`S3_modes` — YES.** p = 1.56, and the limit is stable to 9e-6 relative by
  n = 8.
* **`mu_DW` — YES.** p = 2.48, q* = -3075.251. Best-conditioned of all: 1e-6
  relative by n = 8.
* **`dw_power_dbc` — YES, but it is the slowest.** p = 1.96 and the value moves
  **7.14 dB** (-29.79 -> -22.65 dBc) across the ladder. At n = 1 it is 31.7%
  from its limit. It converges, but nowhere near converged at the committed
  discretization.
* **`U_mean_w` — NO, flagged `NON_MONOTONE`.** The sequence turns around between
  n = 8 (0.191287) and n = 16 (0.191299). The conservative fallback band is
  used: U = 0.0139%. This is a *tiny* non-monotonicity — the whole spread across
  the ladder is 0.8% and the last three levels agree to 1e-4 relative — so the
  practical uncertainty is small even though Richardson is inapplicable.
* **`comb_frac` — YES.** p = 1.10 (first order), q* = 0.853371.
* **`N60` and friends — not applicable by construction.** Integer-valued and
  disqualified as gates; see below.

---

## 4. Observables that are not fit for cross-code use

**Exceeds 2% numerical uncertainty at the committed discretization (n = 1):**

| observable | U at n=1 | comment |
|---|---|---|
| `dw_power_dbc` | **31.65%** | converges (p = 1.96) but is 7.1 dB from its limit at n = 1. Usable only at n >= 8, where U falls to 0.45%. |

Every other observable is below 1% at n = 1, and below 0.3% at n = 8.

**Quantized observables — the quantum is the error floor, regardless of
solution quality:**

| observable | quantum | relative size here | consequence |
|---|---|---|---|
| `S3_int_modes` | 1 mode | 1/443 = **0.226%** | the previously reported "0.23% agreement" is *exactly one quantum* — the smallest non-zero value the metric can express. At delta_omega = 8 kappa, S3 = 64 and the same quantum is 1.6%, so identical solution quality would "fail" a 2% test purely from quantization. Use `S3_modes`. |
| integer DW index | 1 mode | 1/3075 = **0.033%** | an exact-integer match criterion cannot be met reliably by two distinct discretizations. Our own index moves -3074.98 -> -3075.25 across the ladder, i.e. it crosses a rounding boundary under pure refinement. Use `mu_DW`. |
| `N60` | 1 line | — | conditioning is **`dN60_ddB` = 85-96 lines per dB**. A 0.1 dB wing difference moves the count by ~9 lines. `N60_core` is identical (3001) at every level, so all of the variation is a nearly-flat wing crossing an arbitrary threshold. **Diagnostic only; never a gate.** |

**Recommended for cross-code use**, in order of conditioning: `mu_DW`
(U = 1e-6 at n=8), `S3_modes` (9e-6), `U_mean_w` (1.4e-4), `P_peak_w` (2.9e-4),
`comb_frac` (3.9e-4).

---

## 5. Corrections to the previous convergence attribution

`docs/PYLLE_STATUS.md` states that "our own `n_substeps = 1` result sits 4.4%
from our converged answer". **That number is not reproducible from any committed
artifact, and this study measures 0.48%.**

* Measured here: `P_peak_w` at n = 1 is 233.927 W against a Richardson limit of
  232.809 W, i.e. **0.4801%** — a factor of nine smaller than the claim.
* The "4.4%" came from a scratch run at **1500** round trips (not the committed
  5000) comparing **raw** `max|E_j|^2` at n = 1 against n = 8. Both differences
  matter: it is a different operating point, and the raw peak is the
  sampling-phase-contaminated observable whose own level sequence is
  non-monotone. The `convergence_attribution` block inside
  `pylle_crosscheck.json` (n = 2 and 4 only) is likewise not a convergence
  study: two levels cannot yield an observed order.
* `docs/PYLLE_STATUS.md` is out of scope for this task and has been left
  unmodified; the correction is recorded here.

**The DW index reverses its apparent behaviour once the ladder is extended.**
From n = 1 to n = 4 the centroid moves -3074.98 -> -3075.23 and looks as though
it is heading past -3075.25; at n = 8 and n = 16 it settles at -3075.247 and
-3075.250, converging to q* = -3075.251 with p = 2.48. Any conclusion drawn from
n <= 4 about which integer the DW "belongs to" is premature: the value crosses
the -3075 rounding boundary between n = 1 and n = 2.

**Reproduction of the numbers quoted in the task brief.** Every value quoted
there is reproduced here independently and exactly:

| | brief | measured |
|---|---|---|
| peak n=1..16 | 233.9266 / 233.9356 / 233.1375 / 232.8752 / 232.8222 | identical to 4 d.p. |
| centroid n=1..16 | -3074.98 / -3075.12 / -3075.23 / -3075.25 / -3075.25 | identical |
| Richardson peak | p = 2.31, q* = 232.809 | p = 2.3077, q* = 232.809 |
| Richardson centroid | p = 2.48, q* = -3075.251 | p = 2.4839, q* = -3075.251 |
| n=1 peak error | 0.48% | 0.4801% |
| `dN60_ddB` (ours) | 85 | 85 |
| `N60_core` | 3001 at every level | 3001 at every level |
| DW power swing | -29.79 -> -22.65 dBc | -29.789 -> -22.654 dBc |
| S(+3300) at the committed run | -52.2 dBc | -52.17 dBc |

**One discrepancy in the brief, reported rather than reconciled.** The brief
states the sampling-phase bound as `1 - sech^2(dtheta/(2w)) = 4.02%` and, in the
test specification, gives `w = 2.4267e-3` rad. Those two are inconsistent: with
`w = 2.4267e-3` rad (= 2.5495 samples at N = 6601) the bound evaluates to
**3.75%**; 4.02% corresponds to `w = 2.46 samples = 2.3416e-3` rad, which is the
value this solver's actual seed uses (`soliton_w_samples` = 2.4612). The
harness's own measured `w` is 2.461 samples, so **4.02% is the correct figure for
the physical operating point** and 3.75% is what the test's stated `w` implies.
`tests/test_convergence_lle.py` therefore asserts against the analytic bound
recomputed from whatever `w` the test uses, rather than a hard-coded band.

---

## 6. Spectral containment and the blue dispersive wave

Grid ladder at fixed `n_substeps = 8`:

| mu_half | n_modes | edge red (dBc) | edge blue (dBc) | blue DW resolved | dispersion fully measured | `P_peak_w` | `N60` |
|---|---|---|---|---|---|---|---|
| 3300 | 6601 | -81.68 | **-44.00** | **False** | False | 232.875 | 3330 |
| 4400 | 8801 | -130.87 | **-127.48** | **True** | False | 232.122 | 3294 |
| 5500 | 11001 | -161.31 | **-160.95** | **True** | False | 232.126 | 3293 |

Two findings:

1. **The committed grid does not contain the comb.** At mu_half = 3300 the blue
   edge sits at -44 dBc (-52.17 dBc at n = 1), which is *above* the -60 dBc line
   the `N60` metric counts to. Energy is folding back at the Nyquist boundary.
   Enlarging to 8801 modes drops the edge by 83 dB to -127 dBc, and 11001 modes
   reaches -161 dBc; the comb is contained from 8801 on.
2. **The blue DW at mu = +3239 is not resolved on the committed grid and is
   resolved on the larger ones.** At mu_half = 3300 the phase-matching root lies
   only 61 modes inside the boundary and cannot be separated from the aliasing
   edge; at 4400 and 5500 it stands >= 3 dB above the local wing. The previously
   reported "no blue DW at this operating point" is a **grid artifact, not
   physics.**

**Spectral truncation dominates temporal discretization here.** `P_peak_w` moves
232.875 -> 232.126 W between mu_half = 3300 and 5500 — **0.32%** — which is an
order of magnitude larger than the substep uncertainty at the same n = 8
(0.0285%). Any uncertainty budget at the committed grid is set by the grid, not
by the step size.

`dispersion_fully_measured` is `False` at every grid level, because all of them
extend past the CSV's red edge at mu = -3261 and rely on the loader's linear
extrapolation there (39 modes at mu_half = 3300, 1139 at 4400, 2239 at 5500).
That extrapolation is C1-continuous by construction but is not measured data.

*Practical note:* the mandated grid sizes are pathological for the FFT —
8801 = 13 x **677** and 11001 = 3 x 19 x **193** carry large prime factors,
while 6601 = 7 x 23 x 41 is smooth. The larger grids cost several times more
than their mode count suggests.

---

## 7. What would falsify this

Each conclusion, and the specific measurement that would overturn it:

* **"`P_peak_w` converges at second order to 232.809 W."** Falsified by running
  n = 32 and 64 and finding the sequence departing from the extrapolated limit
  by more than the quoted GCI, or the observed order drifting outside
  [0.5, 4.0]. The npz stores every field, so this needs no re-derivation of the
  observables — only two more solver runs.
* **"The n = 1 peak error is 0.48%, not 4.4%."** Falsified by exhibiting a
  committed artifact from which 4.4% is reproducible at the documented operating
  point (6601 modes, 5000 round trips, band-limited peak). The claim here is
  specifically that no such artifact exists.
* **"`N60` disagreement is a wing effect."** Falsified by a refinement level or
  a grid size at which `N60_core` departs from 3001. It has been constant across
  five substep levels; a single counter-example would show the core is also
  moving.
* **"The blue DW is a grid artifact."** Falsified by resolving it at
  mu_half = 3300 with a *different* method that is not sensitive to the aliasing
  edge, or by showing that at mu_half = 4400 the bump at +3239 moves with the
  grid size (it does not: 4400 and 5500 agree).
* **"Spectral truncation dominates at the committed grid."** Falsified by a
  mu_half = 4400 run at n = 1 whose `P_peak_w` differs from the mu_half = 3300
  n = 1 value by much less than 0.32%.
* **"`U_mean_w` is non-monotone only at the 1e-4 level."** Falsified by a longer
  ladder in which the turnaround grows rather than staying at ~1e-4 — which
  would indicate a genuine failure to converge rather than round-off-level
  wobble near the limit.
* **The whole study.** Falsified if the A1 gate ever fails: that would mean the
  harness is not driving the same problem as the committed cross-check, and
  every number here would need re-deriving.
