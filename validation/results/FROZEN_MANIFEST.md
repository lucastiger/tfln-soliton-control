# Frozen validation artifacts

Every file under `validation/results/` is **FROZEN**: it is evidence of a run
that happened, and it is never regenerated except by an explicit re-run trigger
from `docs/VALIDATION_STATUS.md`. `tests/test_validation_freeze.py` asserts every
sha256 below on each test run, so a regeneration cannot happen quietly.

There are currently **no LIVE artifacts**. The FROZEN / LIVE column exists so
that a future artifact which is *meant* to be regenerated (a nightly, say) can be
added without weakening the guarantee on these.

"Producing commit" is the commit that first added the file. For the v1 artifacts
that is not the commit that produced the *numbers*: the v1 run stamped
`git_commit = b31cc0b` in its own JSON, which dates the run rather than
identifying the code, because `validation/pylle_crosscheck.py` was still
uncommitted at the time. See `validation/results/v1/MANIFEST.md`.

| artifact | sha256 | producing commit | produced by | evidence of | state |
|---|---|---|---|---|---|
| `pylle_crosscheck.json` | `2c404b9e2dbbfbdc892b40c5908be57281a718ec0798f8edb05d344ae8b7b017` | `57adee8` | v1 run | the v1 cross-check verdicts (overall FAIL): the original 7-observable comparison at ours n=1 vs pyLLE dt=1 | FROZEN |
| `pylle_crosscheck.png` | `95869d7cabdb6813bc2ccf1eab98b62cd706ed40d37ff58e38e67326cc206b18` | `57adee8` | v1 run | the v1 figure; regenerated post-run from the npz with edited rendering code, as disclosed in docs/PYLLE_STATUS.md | FROZEN |
| `pylle_crosscheck_fields.npz` | `1838653b6abd5c4012c1e501bb5ae40607c47670965a3fe1edeb9a35af98acfe` | `57adee8` | v1 run | the v1 final fields, from which every v1 number is recomputable offline | FROZEN |
| `v1/MANIFEST.md` | `ebd17eb1e6c7b989cefefbb9655e6a804116eec82900ad7bb461059f51fbddbd` | `98e3c26` | Prompt C | the freeze record for the three v1 artifacts, incl. the reconstructed argv | FROZEN |
| `v1/pylle_crosscheck.json` | `2c404b9e2dbbfbdc892b40c5908be57281a718ec0798f8edb05d344ae8b7b017` | `98e3c26` | Prompt C | frozen copy of the v1 verdicts (byte-identical to the original) | FROZEN |
| `v1/pylle_crosscheck.png` | `95869d7cabdb6813bc2ccf1eab98b62cd706ed40d37ff58e38e67326cc206b18` | `98e3c26` | Prompt C | frozen copy of the v1 figure | FROZEN |
| `v1/pylle_crosscheck_fields.npz` | `1838653b6abd5c4012c1e501bb5ae40607c47670965a3fe1edeb9a35af98acfe` | `98e3c26` | Prompt C | frozen copy of the v1 fields | FROZEN |
| `convergence_lle_dw30k.json` | `9210cc74562d016696b627d255f11725f2fe7601ea070585ee05a8099e479f64` | `4130296` | Prompt B | OUR discretization uncertainty at the DW operating point: the substep ladder n=1..16, the grid ladder mu_half 3300/4400/5500, and the Richardson bands every GATED tolerance is derived from | FROZEN |
| `convergence_lle_dw30k_fields.npz` | `eecbaac88e3ab01dcd0c51ee9b956c4451ad13bb42c5dfd898618ad3c4c2686a` | `4130296` | Prompt B | the final field at every level of both ladders | FROZEN |
| `pylle_refinement_dw30k.json` | `a5609cdf94d780a37291164a40ce14557f5da90a946c81dad41391294cf904ad` | `36518e6` | Prompt D | pyLLE's OWN convergence: the dt ladder at both Picard tolerances, proving upstream is refinable and quantifying its uncertainty | FROZEN |
| `pylle_refinement_dw30k_fields.npz` | `06b786031898ddbcdfde9649037b5902009a7db30212c6e0d202c8ded344605d` | `36518e6` | Prompt D | pyLLE's final field at every dt level | FROZEN |
| `pylle_crosscheck_v2.json` | `c3338a2f107a641000e7ca557301d54bfd96ba47a15ab5de51f66b2b5b769309` | `98e3c26` | Prompt C | the v2 verdicts (overall FAIL, QUALIFIED): 7/7 HARD pass, one SEPARATED containment verdict, both existence edges FAIL | FROZEN |
| `pylle_crosscheck_v2.png` | `750ae3f4d83a429af119a4b7ce0f6caac7943465679916cfe73ed8745889f273` | `98e3c26` | Prompt C | the v2 four-panel figure, generated in-run | FROZEN |
| `pylle_crosscheck_v2_fields.npz` | `3526d0e90df08fbb21a3ded12662212fbfc9ece4839d7112fc98fa239c6ffce3` | `98e3c26` | Prompt C | the final field at every level of both codes' ladders | FROZEN |
| `existence_convergence_ours.json` | `7f0dd5b8f7f8766a1f8d9017b301b63f9bab6e8f69fabf09946b76504090c6d2` | `e41555d` | Prompt E | the ours-side existence-edge drift over n=1,2,4 and the resulting G7 discretization term (outcome H-C) | FROZEN |
| `existence_convergence_ours_fields.npz` | `cac0c0318a3d5715897a471812fc68adde873388163d4f2728418c2de515fe42` | `e41555d` | Prompt E | the final field at every existence-bisection evaluation | FROZEN |

## Regenerating any of these

Do not, unless a re-run trigger in `docs/VALIDATION_STATUS.md` has fired. If one
has:

```bash
# the cross-check (needs the pyLLE env + Julia; see docs/PYLLE_STATUS_V2.md §3)
python -m validation.pylle_crosscheck_v2 \
    --pylle-python ./pylle-env/bin/python --julia-bin ./pylle-env/bin/julia

# our discretization ladder (no Julia)
python -m validation.convergence_lle

# pyLLE's own refinement ladder (needs the pyLLE env + Julia)
python third_party/pylle/run_refinement.py \
    --pylle-python ./pylle-env/bin/python --julia-bin ./pylle-env/bin/julia

# the existence-edge ladder (no Julia)
python -m validation.existence_convergence --hold-block
```

Each writes a `numerical_digest` (or, for v1, nothing — that is one of the
defects v2 fixed). A regenerated artifact whose digest differs from the frozen
one is a **finding**, not a nuisance: something changed. Update this manifest in
the same commit that regenerates the artifact, and say in the commit message
which trigger fired.
