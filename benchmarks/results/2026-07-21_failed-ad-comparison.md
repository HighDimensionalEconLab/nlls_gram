# Failed implicit-AD performance comparison

## Environment

- Host: `dh4.econ.ubc.ca`, Apple M4 Max (`arm64`), CPU backend
- Python: 3.13.2
- JAX: 0.10.1
- Baseline source: clean detached worktree at
  `bf22f01fb7ff91ead9e218c92217590f3e325be6`
- Post source: current failed-implicit-AD working tree on the same base commit
- Runtime method: 50 rounds, 100 blocked dispatches per round; medians below
  are per dispatch
- Regression gate: slowdown greater than `max(5%, 1 us)`

The original one-dispatch samples were too noisy for the 1 us gate. They are
retained as raw artifacts, but the gate uses the repeated-dispatch baseline and
post rerun:

- `2026-07-21_failed-ad-stable-baseline.json`
- `2026-07-21_failed-ad-stable-post-rerun.json`

## Successful implicit-AD runtime

| Full benchmark suffix | Baseline (us) | Post (us) | Delta (us) | Delta | Gate crossing |
| --- | ---: | ---: | ---: | ---: | --- |
| `jvp-direct` | 10.0225 | 10.0013 | -0.0212 | -0.21% | no |
| `jvp-direct_aux` | 8.7723 | 7.7760 | -0.9963 | -11.36% | no |
| `jvp-metric_factory` | 23.6000 | 23.5610 | -0.0390 | -0.17% | no |
| `jvp-vmapped` | 27.4027 | 26.4317 | -0.9710 | -3.54% | no |
| `vjp-direct` | 10.6850 | 9.8560 | -0.8290 | -7.76% | no |
| `vjp-direct_aux` | 9.0592 | 9.1277 | +0.0685 | +0.76% | no |
| `vjp-metric_factory` | 24.2642 | 23.0433 | -1.2208 | -5.03% | no |
| `vjp-vmapped` | 37.0979 | 36.2494 | -0.8485 | -2.29% | no |

No successful JVP/VJP runtime crosses the regression gate.

## Cold compilation

Cold compilation used five cache-cleared samples per case. These values are
recorded separately from the runtime gate.

| Case | Baseline (ms) | Post (ms) | Delta (ms) | Delta |
| --- | ---: | ---: | ---: | ---: |
| `jvp-direct` | 72.788 | 81.589 | +8.802 | +12.09% |
| `jvp-direct_aux` | 76.288 | 82.298 | +6.011 | +7.88% |
| `jvp-metric_factory` | 75.872 | 82.959 | +7.088 | +9.34% |
| `jvp-vmapped` | 94.400 | 98.925 | +4.525 | +4.79% |
| `vjp-direct` | 76.881 | 82.598 | +5.717 | +7.44% |
| `vjp-direct_aux` | 80.087 | 86.689 | +6.602 | +8.24% |
| `vjp-metric_factory` | 83.700 | 93.249 | +9.549 | +11.41% |
| `vjp-vmapped` | 95.396 | 103.481 | +8.085 | +8.47% |

The additional status selection and masking graph adds 4.5--9.5 ms to a cold
compile. It does not add a measurable hot-runtime regression in the cases
above.

## Full benchmark suite

`JAX_PLATFORMS=cpu uv run --group benchmark pytest benchmarks
--benchmark-only` completed with 70 passed and 93 skipped in 83.38 seconds.
The complete output is saved in
`2026-07-21_failed-ad-full-post.json`.
