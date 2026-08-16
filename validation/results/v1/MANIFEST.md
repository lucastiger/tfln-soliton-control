# Frozen v1 cross-check artifacts

These three files are **copies**. The originals remain in place at
`validation/results/` and are byte-identical to these; `git mv` was deliberately
not used, and `tests/test_pylle_crosscheck_v2.py::test_v1_artifacts_untouched`
asserts the originals' sha256 on every test run.

| frozen copy | original path | sha256 |
|---|---|---|
| `pylle_crosscheck.json` | `validation/results/pylle_crosscheck.json` | `2c404b9e2dbbfbdc892b40c5908be57281a718ec0798f8edb05d344ae8b7b017` |
| `pylle_crosscheck.png` | `validation/results/pylle_crosscheck.png` | `95869d7cabdb6813bc2ccf1eab98b62cd706ed40d37ff58e38e67326cc206b18` |
| `pylle_crosscheck_fields.npz` | `validation/results/pylle_crosscheck_fields.npz` | `1838653b6abd5c4012c1e501bb5ae40607c47670965a3fe1edeb9a35af98acfe` |

- **git commit stamped in the JSON**: `b31cc0b4b611266fd6d617fd39c457bdc709a002`
- **run timestamp (from the JSON)**: `2026-08-16T01:42:42.721271+00:00`
- **frozen on**: 2026-08-16
- **environment**: ours Python 3.11.15 / numpy 2.4.6 / jax 0.10.2; pyLLE 4.1.2 on
  Python 3.11.15 / numpy 1.26.4 / scipy 1.17.1; Julia 1.10.4 (conda-forge)
- **overall verdict recorded**: FAIL

## Reconstructed argv

The v1 run did **not** record its own argv, which is the defect that
`PROVENANCE` in the v2 task exists to close. The invocation was reconstructed
from the fields the JSON *does* record (`discretization.n_roundtrips_main`,
`n_roundtrips_scan`, `detuning.ramp_start_over_kappa`, `ramp_end_over_kappa`,
`detuning.scan_over_kappa`, `detuning.edge_refinement.bisect_iters`) and is:

```
python -m validation.pylle_crosscheck \
    --pylle-python <pylle-env>/bin/python \
    --julia-bin    <pylle-env>/bin/julia \
    --roundtrips 5000 \
    --dw-start 16 --dw-end 30 \
    --scan 3,5,8,16,30,40,50 \
    --bisect-iters 4
```

Every one of those flags is **non-default**. The committed defaults are 10000
round trips, a 1 kappa -> 3 kappa ramp, a 12-point scan
(`0.6,0.8,1.0,1.5,2.0,3.0,4.0,6.0,8.0,10.0,13.0,16.0`) and 5 bisection
iterations. Re-running `python -m validation.pylle_crosscheck` with no arguments
therefore does **not** reproduce the committed numbers, and the claim in
`docs/PYLLE_STATUS.md` that it does is false as written. The reconstruction
above is inferred, not recorded: it reproduces every parameter the JSON stores,
but nothing in the artifact set can prove no other flag was passed.

## What the v1 run established, and what is now known to be wrong about its interpretation

v1 established the part that has held up: the **convention translation**. Nine
findings — D_int in rad/s with no 2*pi, the conjugate field map
`E_ours = conj(A_pyLLE)*sqrt(t_r)`, the opposite detuning sign, the dispersion
mirror `D_int_pyLLE(mu) = D_int_ours(-mu)` that the conjugation forces, the
drive phase, pyLLE's domain-centre re-zero, its spline refit, the 7.11-FSR pump
reference trap, and the D1/t_r coupling — were derived from pyLLE 4.1.2 source
and then confirmed at runtime by a parameter round trip good to 1e-12 and a
dispersion refit agreeing to 2.3e-12. Two independent implementations placing
the same dispersive wave within one mode of each other, on a strongly
asymmetric measured dispersion, is real evidence that neither code has a sign,
factor-of-2*pi, or mode-index-origin error. That conclusion stands.

What is wrong is the **interpretation of the disagreements**. v1 compared
ours(n_substeps=1) against pyLLE(dt=1) and read the residual gaps — 4.2% in peak
power, 8.8% in the -60 dBc line count, one mode in the DW index, 5.9% at the
lower existence edge — as cross-code disagreement, against tolerances chosen
independently of either code's own discretization error. That comparison cannot
separate "both codes right, both coarse" from "one code wrong", because at
dt = 1 round trip neither code is near its own converged limit: our own
refinement moves the peak power by more than the cross-code gap. v1 also
asserted that pyLLE "cannot be refined" (its `convergence_attribution` note
records `pylle_refinable: false`); that is false — `dt` is a plumbed parameter
of the Julia kernel and the `tol`/`maxiter` CLI plumbing exists at
`ComputeLLE.jl:46-47` and is clobbered at `:278-279`. And the two codes were not
even compared at the same operating point: upstream's probe cadence returned
pyLLE's field at round trip 4976 of 5000, i.e. at 29.9328 kappa against our
30.0000 kappa.

The v2 run supersedes the verdicts, not the conventions. See
`docs/PYLLE_STATUS_V2.md`, section "Corrections to v1", for each correction with
its evidence.
