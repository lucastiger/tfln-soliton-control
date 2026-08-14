# Regenerating W1 / W5 and `campaign_report.json`

The normally-ordered spectral reporting changes W1 and W5 outputs, so
`analysis/results/validation/campaign_report.json` must be regenerated on real
compute. W1 runs at `n_tau = 16384` with 24 seeds; that is a GPU job, not a
laptop job.

**W2, W3 and W4 are untouched by this change.** The campaign driver merges each
workstream into the existing JSON as it finishes, so running `--workstream 1,5`
preserves the committed W2/W3/W4 blocks. See the caveat at the bottom about the
provenance stamp if you take that route.

W5 reuses W1's ensemble when the two are run together (`w1_ens` is passed
straight through), so `--workstream 1,5` costs the same as W1 alone. Never run
them as two separate invocations — W5 would then build its own smaller ensemble
and the two blocks would describe different data.

---

## Google Colab

Pick a GPU runtime first: **Runtime → Change runtime type → T4 GPU** (A100 if
you have it; the run is ~3x faster and the 24-seed job fits comfortably).

### Cell 1 — environment

```python
!nvidia-smi -L
# JAX with CUDA. Colab ships a CUDA 12 image; this pulls the matching wheels.
!pip -q install --upgrade "jax[cuda12]" numpy scipy matplotlib pyyaml
import jax
print("jax", jax.__version__, "| backend:", jax.default_backend(), "|", jax.devices())
```

Stop here if `backend` prints `cpu` — the run will take days on CPU. Re-check the
runtime type.

### Cell 2 — clone the repo

```python
BRANCH = "claude/normal-ordered-spectrum"   # or "main" once this has merged

!git clone --depth 1 --branch {BRANCH} https://github.com/lucastiger/soliton-control.git
%cd /content/soliton-control
!git log --oneline -1
```

For a private repo, use a fine-grained PAT with `contents: read`:

```python
from getpass import getpass
TOKEN = getpass("GitHub token: ")
!git clone --depth 1 --branch {BRANCH} https://{TOKEN}@github.com/lucastiger/soliton-control.git
```

### Cell 3 — sanity check before spending GPU time

Confirms the vacuum normalization is intact on this machine and this JAX build.
Two minutes now beats discovering a broken toolchain after a two-hour run.

```python
!python -m pytest tests/test_vacuum_floor_normalization.py -q
```

### Cell 4 — the run

```python
import time
t0 = time.time()
!python analysis/noise_validation_campaign.py \
    --workstream 1,5 \
    --seeds 24 \
    --out analysis/results/validation
print(f"wall time: {(time.time()-t0)/60:.1f} min")
```

Expect roughly **1.5–3 h on a T4**, **35–60 min on an A100**, dominated by the
24 ON seeds at `n_tau = 16384`. Colab disconnects idle sessions, so keep the tab
open; if the runtime dies mid-run, the JSON on disk still holds whatever
workstreams completed and you can rerun.

Smoke it first if you want to see the whole pipeline in under a minute — this
writes to a scratch directory and does **not** touch the committed report:

```python
!python analysis/noise_validation_campaign.py \
    --workstream 1,5 --quick --seeds 2 --out /tmp/campaign_smoke
```

### Cell 5 — read off the numbers that changed

```python
import json
d = json.load(open("analysis/results/validation/campaign_report.json"))
w1, w5 = d["workstream1_dw_survival"], d["workstream5_vacuum_budget"]

print(f"pedestal (symmetric)        {w1['vacuum_floor_db_rel_pump']:+8.2f} dB rel pump")
print(f"ensemble s.e. (detect limit){w1['ensemble_mean_sem_median_db_rel_pump']:+8.2f} dB rel pump")
print(f"DW resolved / total         {w1['n_dw_peaks_resolved_normal_ordered']}"
      f"/{len(w1['dw_peaks'])}   (gate: {w1['dw_resolve_sigma']:g} sigma)")
for p in w1["dw_peaks"]:
    print(f"  {p['side']:>4} mu={p['mu_off']:>6}  OFF {p['level_db_off']:7.2f} dB"
          f" | ON sym {p['level_on_at_off_mode_db']:7.2f} dB"
          f" | sigma {p['residual_significance_sigma']:+6.2f}"
          f" | x{p['seed_factor_to_resolve_off_peak']:.1e} seeds to resolve")
print(f"comb fraction (normally ordered)  OFF {w1['comb_fraction_off_normal_ordered']:.4f}"
      f" | ON {w1['comb_fraction_on_normal_ordered']:.4f}")
print(f"3 dB span ON  sym {w1['three_db_span_ghz_on_mean']:.1f} GHz"
      f" | normally ordered {w1['three_db_span_ghz_on_normal_ordered']:.1f} GHz")
print(f"W5 far wing  sym {w5['far_wing_floor_multiple_of_half_hbar_omega0']:+.4f} x pedestal"
      f" | normally ordered {w5['far_wing_normal_ordered_multiple_of_pedestal']:+.4f}")
```

Two internal consistency checks worth eyeballing:

* `far_wing_normal_ordered_multiple_of_pedestal` must equal
  `far_wing_floor_multiple_of_half_hbar_omega0 - 1` **exactly** (it is the same
  median with the pedestal removed). If the symmetric multiple is ~1.00, the
  normally-ordered one must be ~0.00 — the far wing is pure vacuum, and an OSA
  reads zero there.
* `three_db_span_ghz_on_normal_ordered` should sit within a GHz or so of the
  symmetric-ordered ON mean. The 3 dB span is a comb-core metric whose own 60 dB
  floor mask already discards the pedestal-dominated wings, so it is nearly
  ordering-independent by construction. A large gap means the mask is engaging
  somewhere unexpected.

### Cell 6 — pull the artifacts back out

```python
!cd analysis/results/validation && zip -q -r /content/campaign_w1_w5.zip \
    campaign_report.json dw_survival_spectrum_off_on.png dw_peak_metrics.png \
    vacuum_floor_ensemble.png energy_fluctuation_budget.png
from google.colab import files
files.download("/content/campaign_w1_w5.zip")
```

Or commit straight back from Colab:

```python
!git config user.email "you@example.com" && git config user.name "Your Name"
!git add analysis/results/validation
!git commit -m "Regenerate W1/W5 with normally-ordered spectral reporting (24 seeds, GPU)"
!git push
```

---

## Local / cluster equivalent

```bash
python analysis/noise_validation_campaign.py --workstream 1,5 --seeds 24 \
    --out analysis/results/validation
```

Then verify the committed report still satisfies the campaign regression suite:

```bash
pytest tests/test_noise_validation_campaign.py -v
```

---

## Provenance caveat

`--workstream 1,5` rewrites the top-level `provenance` and `meta` blocks even
though W2/W3/W4 came from an earlier run, so the stamp will describe the partial
run rather than the whole report. That is the driver's pre-existing incremental
behaviour, not something this change introduced.

If you want a fully self-consistent report — every block produced by one
invocation, one provenance stamp, and the consolidated summary table printed at
the end (it only prints when all five workstreams are requested) — run:

```bash
python analysis/noise_validation_campaign.py --workstream all --seeds 24 \
    --out analysis/results/validation
```

That is several times the cost of `1,5`. For a paper table, run `all`; for
checking what this change did to the numbers, `1,5` is enough.
