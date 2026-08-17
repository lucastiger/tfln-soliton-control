# Contributing

Thanks for looking at this. The project is a *benchmark*, so the bar is a little
different from a typical library: the numbers it produces are the deliverable,
and most of the rules below exist to keep those numbers meaningful.

## The bit-identity rule

**Any pull request that changes numerical output must regenerate the golden
fixtures and justify the change in the PR description.** This is the one rule
that is not negotiable.

Why it is stated this strongly: the repository's central claim is that when a
noise channel is switched off, the solver reproduces the deterministic reference
*exactly*. That is what makes per-channel attribution a statement about physics
rather than about accumulated numerical drift. A silent change to the last bits
of a trajectory does not announce itself — it shows up months later as a budget
row that cannot be reproduced, and by then nobody can tell whether the cause was
a physics fix, a refactor, or a dependency bump.

### What counts as changing numerical output

Anything that moves a bit. Reordering a sum, changing an `fft` normalization,
altering the PRNG key chain, replacing `x * 0.5` with `x / 2`, bumping jaxlib.
If you are unsure, run:

```bash
SOLITON_STRICT_ULP=1 pytest tests/test_noise_off_identity.py -q
```

on the pinned toolchain (see below). If it passes, you did not change output.
That test is the arbiter, not your judgement or mine.

### If you did change output, on purpose

1. **Say why, in the PR description.** Name the physics or the defect. "Cleaner"
   is not a reason to move a benchmark's reference numbers; "the previous form
   lost significance for κ·t_r ≪ 1, see the attached comparison" is.
2. **Regenerate the goldens** and commit them with the code change in the same
   PR, never separately:
   ```bash
   python -m validation.noise_off_identity --write-golden
   ```
3. **Show the size of the change.** The failure message from
   `test_noise_off_identity.py` already prints max abs diff, max rel diff, the
   first differing flat index and both sides' library versions. Paste it. A
   ULP-scale diff and a 1e-3 diff are entirely different events and the reviewer
   needs to see which one this is.
4. **Say what else moved.** If `validation/results/` artifacts change, they are
   hash-pinned in `FROZEN_MANIFEST.md` and `tests/test_validation_freeze.py`
   will tell you. Regenerate deliberately and record it.

### Dependency bumps are output changes

jaxlib ships XLA, and XLA may change reduction order and kernel fusion between
releases. Bumping the pin in `requirements.lock.txt` therefore follows the exact
procedure above — regenerate, measure, justify. It is not routine maintenance.
The header of that file says the same thing at more length.

## Everything else is opt-in

New behaviour goes behind a flag whose default reproduces current behaviour
exactly. This is how every noise channel in the repository was added, and it is
why `git log` contains no commit that quietly moved a number. Config booleans go
under `physical_parameters` in `config/sin_params.yaml` as `0`/`1` integers, not
YAML `true`/`false` — `tests/test_config.py` requires every leaf there to parse
as a plain number.

## Tests

Add them; do not rewrite them. If an existing test seems wrong, that is worth a
conversation before a commit — it may be encoding a physical constraint whose
reason is not obvious from the assertion. **Physics correctness outranks a
passing suite:** if a test encodes wrong physics, say so and stop rather than
adjusting the code to satisfy it.

Markers:

| Marker | Default | Run it with |
|---|---|---|
| *(none)* | runs | `pytest -q` |
| `slow` | skipped | `pytest -q --runslow` |
| `gpu` | runs if collected | `pytest -m gpu` |
| `pylle_full` | skipped | `pytest -m pylle_full` (needs Julia + a pyLLE env) |

## Environments

Three, for three different purposes.

```bash
# 1. Development. Floating versions, everything installed.
pip install -e ".[dev,ml]"

# 2. What a benchmark user gets. No torch; conftest.py drops the two
#    torch-only test modules from collection.
pip install -e .

# 3. The reference frame. Pinned, hash-verified, the toolchain the goldens
#    were produced with. Required for SOLITON_STRICT_ULP=1.
pip install --require-hashes -r requirements.lock.txt
pip install --no-deps --no-build-isolation -e .
```

Install **editable**. `simulator/lle_solver.py` resolves the default config
relative to the checkout, so a non-editable wheel imports but cannot find
`config/sin_params.yaml`.

`docker build -t sc . && docker run --rm sc` gives you (3) and runs the fast
suite in it. `environment.yml` is a conda convenience — it is explicitly *not*
the bit-identity environment, for the BLAS reason documented in its header.

To change a dependency, edit `requirements.in` (or `pyproject.toml`) and
recompile — never hand-edit the lock:

```bash
pip-compile --generate-hashes --strip-extras --output-file=requirements.lock.txt requirements.in
```

Then restore the `jaxlib IS THE ULP-STABILITY DEPENDENCY` note at the top of the
lock; `pip-compile` does not preserve it, and
`tests/test_packaging_metadata.py` fails if it goes missing.

## Style

`ruff check .` with `line-length = 100`. Beyond that, match the file you are
editing — comment density in this repository is high on purpose, and the useful
comments are the ones explaining *why a number is what it is*, not what the line
does.

## Provenance

Anything written to `analysis/results/` or `validation/results/` should carry a
`.provenance.json` sidecar (`simulator/provenance.py`). An artifact without one
cannot be traced to the config and commit that produced it, which makes it
unusable in the paper.
