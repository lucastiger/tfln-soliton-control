# Stochastic-LLE solver: runtime scaling

- generated: `2026-08-17T08:42:18Z`
- git commit: `938a552e4c54`
- quick mode: False
- points: 27 measured, 81 skipped, 0 failed
- timing: median of 3 runs after a separate compile call and a warmup call; every output is materialized on the host before the clock stops

## ns per round-trip per mode vs `n_tau`

**Hardware.** CPU Intel(R) Xeon(R) Processor @ 2.80GHz. 4 logical cores. JAX 0.10.2, NumPy 2.4.6, Python 3.11.15, Linux-6.18.5-fc-v20-x86_64-with-glibc2.39.

> No **GPU** backend was available on this machine, so those columns are absent rather than empty. `jax.devices('gpu')` raises rather than returning an empty list when no GPU backend is linked in.

Cell = `ns_per_roundtrip_per_mode` = `wall_s * 1e9 / (t_slow * n_tau * n_traj)`, median over the swept `t_slow` and `n_traj` at that `n_tau`.

| `n_tau` | CPU all_off | CPU all_on |
|---|---|---|
| `256` | 234 | 720.7 |
| `512` | 276.6 | 544.3 |
| `1024` | 253.1 | 407.7 |
| `2048` | 262.1 | 632.2 |
| `4096` | 274 | 498.5 |
| `8192` | 289.6 | 419.8 |

## Fitted scaling exponent

`alpha_total` is the log-log slope of **total wall time** vs `n_tau`. An FFT-limited step costs O(n log n), predicting `alpha_total = 1 + 1/ln(n_tau)` (the *1 + log-correction*). `alpha_per_mode = alpha_total - 1` is the slope of the normalized ns/round-trip/mode shown above — that column divides one factor of `n_tau` back out, so a healthy FFT-bound solver shows a small positive slope there, not ~1.

Each solver call also carries a **fixed cost independent of `n_tau`** (host config load and validation, noise-sequence setup, scan dispatch). At small grids that floor is most of the runtime and drags the plain log-log slope below the FFT prediction, so the floor is fitted explicitly as `T = a + c·n·log₂n` and reported: compare `alpha_compute_bound` (largest half of the grid) against `predicted`, not `alpha_total`, whenever `overhead@min` is large.

| device / noise | `t_slow` | `n_traj` | points | `alpha_total` | `alpha_per_mode` | `alpha_compute_bound` | predicted (n log n) | R² | floor `a` [s] | overhead@min |
|---|---|---|---|---|---|---|---|---|---|---|
| cpu/all_off | 1000 | 1 | 6 | 0.8192 | -0.1808 | 1.072 | 1.137 | 0.9779 | 0.0848 | 60% |
| cpu/all_on | 1000 | 1 | 6 | 0.5856 | -0.4144 | 0.7047 | 1.137 | 0.9838 | 0.524 | 113% |

> the n_tau-independent per-call floor is 113% of the smallest point's runtime, so alpha_total UNDERSTATES the FFT scaling; read alpha_total_compute_bound instead

> the n_tau-independent per-call floor is 60% of the smallest point's runtime, so alpha_total UNDERSTATES the FFT scaling; read alpha_total_compute_bound instead

## Noise-on overhead

`(wall_on - wall_off) / wall_off`, matched per (device, `n_tau`, `t_slow`, `n_traj`). `NoiseConfig.all_on()` enables every stochastic channel **and** `thermal_feedback`, the deterministic thermo-optic ODE, so this is the cost of the whole switched-on stack.

**Median overhead: 96.8%** (range 37.1% – 225.6% over 13 matched pairs).

| device | `n_tau` | `t_slow` | `n_traj` | off [s] | on [s] | overhead |
|---|---|---|---|---|---|---|
| cpu | 256 | 1000 | 1 | 0.142 | 0.4623 | +225.6% |
| cpu | 256 | 1000 | 8 | 0.4793 | 1.205 | +151.5% |
| cpu | 256 | 1000 | 64 | 3.431 | 4.705 | +37.1% |
| cpu | 256 | 10000 | 1 | 0.9713 | 2.183 | +124.8% |
| cpu | 512 | 1000 | 1 | 0.1952 | 0.5942 | +204.4% |
| cpu | 512 | 1000 | 8 | 0.9721 | 1.861 | +91.4% |
| cpu | 512 | 10000 | 1 | 1.416 | 2.787 | +96.8% |
| cpu | 1024 | 1000 | 1 | 0.3102 | 0.8087 | +160.7% |
| cpu | 1024 | 1000 | 8 | 2.074 | 2.958 | +42.6% |
| cpu | 1024 | 10000 | 1 | 2.472 | 4.175 | +68.9% |
| cpu | 2048 | 1000 | 1 | 0.5367 | 1.295 | +141.2% |
| cpu | 4096 | 1000 | 1 | 1.122 | 2.042 | +81.9% |
| cpu | 8192 | 1000 | 1 | 2.372 | 3.439 | +45.0% |

## All measured points

| device | `n_tau` | `t_slow` | `n_traj` | noise | compile [s] | wall [s] (min–max) | ns/RT/mode | RT/s | peak RSS [MB] | device mem [MB] | out [MB] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | 256 | 1000 | 1 | all_off | 2.52 | 0.142 (0.14–0.144) | 554.7 | 7042 | 438.1 | — | 0.0383 |
| cpu | 256 | 1000 | 1 | all_on | 2.3 | 0.4623 (0.451–0.485) | 1806 | 2163 | 550.8 | — | 0.0536 |
| cpu | 256 | 1000 | 8 | all_off | 1.72 | 0.4793 (0.475–0.492) | 234 | 2086 | 587 | — | 0.307 |
| cpu | 256 | 1000 | 8 | all_on | 2.72 | 1.205 (1.19–1.21) | 588.6 | 829.6 | 675 | — | 0.429 |
| cpu | 256 | 1000 | 64 | all_off | 1.66 | 3.431 (3.14–3.53) | 209.4 | 291.5 | 714.8 | — | 2.45 |
| cpu | 256 | 1000 | 64 | all_on | 2.67 | 4.705 (4.54–4.78) | 287.2 | 212.5 | 815.6 | — | 3.43 |
| cpu | 256 | 10000 | 1 | all_off | 1.29 | 0.9713 (0.965–1.02) | 379.4 | 1.029e+04 | 832.2 | — | 0.313 |
| cpu | 256 | 10000 | 1 | all_on | 2.07 | 2.183 (2.02–2.21) | 852.8 | 4580 | 893.2 | — | 0.466 |
| cpu | 256 | 10000 | 8 | all_off | 1.64 | 4.162 (4.08–4.17) | 203.2 | 2403 | 919.3 | — | 2.5 |
| cpu | 512 | 1000 | 1 | all_off | 1.05 | 0.1952 (0.193–0.196) | 381.3 | 5123 | 928.4 | — | 0.0462 |
| cpu | 512 | 1000 | 1 | all_on | 1.77 | 0.5942 (0.576–0.607) | 1160 | 1683 | 981 | — | 0.0614 |
| cpu | 512 | 1000 | 8 | all_off | 1.09 | 0.9721 (0.971–1) | 237.3 | 1029 | 993.8 | — | 0.369 |
| cpu | 512 | 1000 | 8 | all_on | 1.9 | 1.861 (1.83–1.98) | 454.4 | 537.3 | 1049 | — | 0.491 |
| cpu | 512 | 10000 | 1 | all_off | 0.883 | 1.416 (1.38–1.43) | 276.6 | 7061 | 1058 | — | 0.321 |
| cpu | 512 | 10000 | 1 | all_on | 1.21 | 2.787 (2.73–2.93) | 544.3 | 3589 | 1109 | — | 0.473 |
| cpu | 1024 | 1000 | 1 | all_off | 1.03 | 0.3102 (0.307–0.328) | 302.9 | 3224 | 1121 | — | 0.0618 |
| cpu | 1024 | 1000 | 1 | all_on | 1.77 | 0.8087 (0.778–0.829) | 789.7 | 1237 | 1173 | — | 0.077 |
| cpu | 1024 | 1000 | 8 | all_off | 1.14 | 2.074 (2.06–2.16) | 253.1 | 482.2 | 1186 | — | 0.494 |
| cpu | 1024 | 1000 | 8 | all_on | 2 | 2.958 (2.92–3.07) | 361.1 | 338.1 | 1245 | — | 0.616 |
| cpu | 1024 | 10000 | 1 | all_off | 0.819 | 2.472 (2.45–2.5) | 241.4 | 4045 | 1254 | — | 0.336 |
| cpu | 1024 | 10000 | 1 | all_on | 1.27 | 4.175 (4.16–4.2) | 407.7 | 2395 | 1304 | — | 0.489 |
| cpu | 2048 | 1000 | 1 | all_off | 1.11 | 0.5367 (0.528–0.538) | 262.1 | 1863 | 1315 | — | 0.093 |
| cpu | 2048 | 1000 | 1 | all_on | 1.78 | 1.295 (1.26–1.3) | 632.2 | 772.4 | 1369 | — | 0.108 |
| cpu | 4096 | 1000 | 1 | all_off | 1.01 | 1.122 (1.11–1.13) | 274 | 891 | 1377 | — | 0.156 |
| cpu | 4096 | 1000 | 1 | all_on | 1.6 | 2.042 (1.98–2.33) | 498.5 | 489.7 | 1431 | — | 0.171 |
| cpu | 8192 | 1000 | 1 | all_off | 1.02 | 2.372 (2.29–2.4) | 289.6 | 421.6 | 1444 | — | 0.281 |
| cpu | 8192 | 1000 | 1 | all_on | 1.66 | 3.439 (3.37–3.54) | 419.8 | 290.8 | 1491 | — | 0.296 |

## Skipped configurations

| device | `n_tau` | `t_slow` | `n_traj` | noise | status | reason |
|---|---|---|---|---|---|---|
| cpu | 256 | 10000 | 8 | all_on | skipped: projected 17 s per run | projected single-run wall time 17 s exceeds the --max-seconds budget of 12 s |
| cpu | 256 | 10000 | 64 | all_off | skipped: projected 91 s per run | projected single-run wall time 91 s exceeds the --max-seconds budget of 12 s |
| cpu | 256 | 10000 | 64 | all_on | skipped: projected 140 s per run | projected single-run wall time 140 s exceeds the --max-seconds budget of 12 s |
| cpu | 256 | 100000 | 1 | all_off | skipped: projected 14 s per run | projected single-run wall time 14 s exceeds the --max-seconds budget of 12 s |
| cpu | 256 | 100000 | 1 | all_on | skipped: projected 22 s per run | projected single-run wall time 22 s exceeds the --max-seconds budget of 12 s |
| cpu | 256 | 100000 | 8 | all_off | skipped: projected 114 s per run | projected single-run wall time 114 s exceeds the --max-seconds budget of 12 s |
| cpu | 256 | 100000 | 8 | all_on | skipped: projected 175 s per run | projected single-run wall time 175 s exceeds the --max-seconds budget of 12 s |
| cpu | 256 | 100000 | 64 | all_off | skipped: projected 909 s per run | projected single-run wall time 909 s exceeds the --max-seconds budget of 12 s |
| cpu | 256 | 100000 | 64 | all_on | skipped: projected 1397 s per run | projected single-run wall time 1397 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 1000 | 64 | all_off | skipped: projected 20 s per run | projected single-run wall time 20 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 1000 | 64 | all_on | skipped: projected 20 s per run | projected single-run wall time 20 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 10000 | 8 | all_off | skipped: projected 26 s per run | projected single-run wall time 26 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 10000 | 8 | all_on | skipped: projected 26 s per run | projected single-run wall time 26 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 10000 | 64 | all_off | skipped: projected 204 s per run | projected single-run wall time 204 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 10000 | 64 | all_on | skipped: projected 204 s per run | projected single-run wall time 204 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 100000 | 1 | all_off | skipped: projected 32 s per run | projected single-run wall time 32 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 100000 | 1 | all_on | skipped: projected 32 s per run | projected single-run wall time 32 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 100000 | 8 | all_off | skipped: projected 256 s per run | projected single-run wall time 256 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 100000 | 8 | all_on | skipped: projected 256 s per run | projected single-run wall time 256 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 100000 | 64 | all_off | skipped: projected 2045 s per run | projected single-run wall time 2045 s exceeds the --max-seconds budget of 12 s |
| cpu | 512 | 100000 | 64 | all_on | skipped: projected 2045 s per run | projected single-run wall time 2045 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 1000 | 64 | all_off | skipped: projected 45 s per run | projected single-run wall time 45 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 1000 | 64 | all_on | skipped: projected 45 s per run | projected single-run wall time 45 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 10000 | 8 | all_off | skipped: projected 57 s per run | projected single-run wall time 57 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 10000 | 8 | all_on | skipped: projected 57 s per run | projected single-run wall time 57 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 10000 | 64 | all_off | skipped: projected 454 s per run | projected single-run wall time 454 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 10000 | 64 | all_on | skipped: projected 454 s per run | projected single-run wall time 454 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 100000 | 1 | all_off | skipped: projected 71 s per run | projected single-run wall time 71 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 100000 | 1 | all_on | skipped: projected 71 s per run | projected single-run wall time 71 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 100000 | 8 | all_off | skipped: projected 568 s per run | projected single-run wall time 568 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 100000 | 8 | all_on | skipped: projected 568 s per run | projected single-run wall time 568 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 100000 | 64 | all_off | skipped: projected 4544 s per run | projected single-run wall time 4544 s exceeds the --max-seconds budget of 12 s |
| cpu | 1024 | 100000 | 64 | all_on | skipped: projected 4544 s per run | projected single-run wall time 4544 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 1000 | 8 | all_off | skipped: projected 12 s per run | projected single-run wall time 12 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 1000 | 8 | all_on | skipped: projected 12 s per run | projected single-run wall time 12 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 1000 | 64 | all_off | skipped: projected 100 s per run | projected single-run wall time 100 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 1000 | 64 | all_on | skipped: projected 100 s per run | projected single-run wall time 100 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 10000 | 1 | all_off | skipped: projected 16 s per run | projected single-run wall time 16 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 10000 | 1 | all_on | skipped: projected 16 s per run | projected single-run wall time 16 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 10000 | 8 | all_off | skipped: projected 125 s per run | projected single-run wall time 125 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 10000 | 8 | all_on | skipped: projected 125 s per run | projected single-run wall time 125 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 10000 | 64 | all_off | skipped: projected 1000 s per run | projected single-run wall time 1000 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 10000 | 64 | all_on | skipped: projected 1000 s per run | projected single-run wall time 1000 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 100000 | 1 | all_off | skipped: projected 156 s per run | projected single-run wall time 156 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 100000 | 1 | all_on | skipped: projected 156 s per run | projected single-run wall time 156 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 100000 | 8 | all_off | skipped: projected 1250 s per run | projected single-run wall time 1250 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 100000 | 8 | all_on | skipped: projected 1250 s per run | projected single-run wall time 1250 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 100000 | 64 | all_off | skipped: projected 9997 s per run | projected single-run wall time 9997 s exceeds the --max-seconds budget of 12 s |
| cpu | 2048 | 100000 | 64 | all_on | skipped: projected 9997 s per run | projected single-run wall time 9997 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 1000 | 8 | all_off | skipped: projected 27 s per run | projected single-run wall time 27 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 1000 | 8 | all_on | skipped: projected 27 s per run | projected single-run wall time 27 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 1000 | 64 | all_off | skipped: projected 218 s per run | projected single-run wall time 218 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 1000 | 64 | all_on | skipped: projected 218 s per run | projected single-run wall time 218 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 10000 | 1 | all_off | skipped: projected 34 s per run | projected single-run wall time 34 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 10000 | 1 | all_on | skipped: projected 34 s per run | projected single-run wall time 34 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 10000 | 8 | all_off | skipped: projected 273 s per run | projected single-run wall time 273 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 10000 | 8 | all_on | skipped: projected 273 s per run | projected single-run wall time 273 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 10000 | 64 | all_off | skipped: projected 2181 s per run | projected single-run wall time 2181 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 10000 | 64 | all_on | skipped: projected 2181 s per run | projected single-run wall time 2181 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 100000 | 1 | all_off | skipped: projected 341 s per run | projected single-run wall time 341 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 100000 | 1 | all_on | skipped: projected 341 s per run | projected single-run wall time 341 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 100000 | 8 | all_off | skipped: projected 2726 s per run | projected single-run wall time 2726 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 100000 | 8 | all_on | skipped: projected 2726 s per run | projected single-run wall time 2726 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 100000 | 64 | all_off | skipped: projected 21812 s per run | projected single-run wall time 21812 s exceeds the --max-seconds budget of 12 s |
| cpu | 4096 | 100000 | 64 | all_on | skipped: projected 21812 s per run | projected single-run wall time 21812 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 1000 | 8 | all_off | skipped: projected 59 s per run | projected single-run wall time 59 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 1000 | 8 | all_on | skipped: projected 59 s per run | projected single-run wall time 59 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 1000 | 64 | all_off | skipped: projected 473 s per run | projected single-run wall time 473 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 1000 | 64 | all_on | skipped: projected 473 s per run | projected single-run wall time 473 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 10000 | 1 | all_off | skipped: projected 74 s per run | projected single-run wall time 74 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 10000 | 1 | all_on | skipped: projected 74 s per run | projected single-run wall time 74 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 10000 | 8 | all_off | skipped: projected 591 s per run | projected single-run wall time 591 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 10000 | 8 | all_on | skipped: projected 591 s per run | projected single-run wall time 591 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 10000 | 64 | all_off | skipped: projected 4726 s per run | projected single-run wall time 4726 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 10000 | 64 | all_on | skipped: projected 4726 s per run | projected single-run wall time 4726 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 100000 | 1 | all_off | skipped: projected 738 s per run | projected single-run wall time 738 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 100000 | 1 | all_on | skipped: projected 738 s per run | projected single-run wall time 738 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 100000 | 8 | all_off | skipped: projected 5907 s per run | projected single-run wall time 5907 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 100000 | 8 | all_on | skipped: projected 5907 s per run | projected single-run wall time 5907 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 100000 | 64 | all_off | skipped: projected 47259 s per run | projected single-run wall time 47259 s exceeds the --max-seconds budget of 12 s |
| cpu | 8192 | 100000 | 64 | all_on | skipped: projected 47259 s per run | projected single-run wall time 47259 s exceeds the --max-seconds budget of 12 s |

## Method notes

- **Blocking.** `solve_lle_ssfm_jax` returns `np.asarray` of every output, which is a blocking device-to-host transfer, so a timed call cannot measure dispatch alone. `_block_until_ready` additionally walks the result for any remaining `jax.Array`; the count it finds is stored per point as `jax_arrays_blocked` (0 across all points here — zero means the conversion had already synchronized).
- **`wall_s` is end-to-end** and includes the host transfer of every returned history; `out [MB]` shows how much data that was.
- **`jit_compile_s`** is `first_call_s - wall_s`: the first call must also execute, so compile time is only observable by difference.
- **`peak RSS`** is sampled from `/proc/self/statm` on a thread during the timed runs, not taken from `ru_maxrss`, which is a process-lifetime high-water mark that every later configuration would inherit from the biggest earlier one.
- **Numerics**: solver defaults (`n_substeps=1`, no dealias, no edge absorber). `PRODUCTION_NUMERICS` costs ~4x more per round trip.
- **Memory guardrail**: `--max-mem-gb 8`, projected by an explicit byte model whose per-point breakdown is in `runtime.json`.
