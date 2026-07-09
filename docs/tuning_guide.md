# Tuning Guide

Decision-oriented heuristics for choosing solvers and hyperparameters —
written for humans and AI assistants alike. Contracts and formulas live in
the [main docs](index.md); the math is in
[Metric Gauss-Newton](gauss_newton.md). Throughout, `m` is the residual count
and `n` the parameter count; the package targets `m << n`.

## Starting Point

```python
solver = UnderdeterminedLevenbergMarquardt(residual_fn)
result = solver.solve(x0, args, max_steps=500, atol=..., gtol=...)
```

- **`linear_solver="cholesky"` (the default) is the best first choice for
  small-to-medium `m`** — it factors the small `m × m` Gram system, so `n`
  only enters through matvecs.
- **Geodesic acceleration is on by default** — it costs one extra
  directional derivative per step (plus one residual evaluation when the
  acceptance gate passes), the accept/reject test makes it safe, and on
  curved residuals it substantially cuts step counts. Near-linear problems
  gain little — `geodesic_acceleration=False` if the extra evaluation
  matters. With a custom metric it requires `metric.norm`.
- **The Jacobian cache is on by default** (rejected steps ~2x cheaper) at
  the cost of an `(n_params, n_residuals)` state buffer. Pass
  `cache_jacobian=False` for manual `update()` loops that swap `args`/`p`
  between steps (stale-cache hazard) or when the buffer strains GPU memory.
- Set `atol`/`gtol` rather than relying on `max_steps`: a converged solve
  that runs to `max_steps` wastes exactly the steps you didn't bound.

## Solver Selection

| situation | use |
| --- | --- |
| `m` up to a few thousand | `cholesky` (default) |
| `m × n` Jacobian too big to materialize, or very large `m` | `cg` |
| square-root-only metric, matrix-free | `lsmr` |
| ill-conditioned metric, moderate `m`, full-row-rank `J` | `qr` |

- **Avoid `qr` when massively overparameterized.** It does not use the Gram
  form: it factors the whitened `n × m` matrix, so cost scales with `n`
  (measured 8-16x slower than `cholesky` at `n=8192, m=1024`), and it
  requires full row rank — rank-deficient Jacobians produce non-finite steps.
  Its advantage is conditioning (it avoids squaring the condition number);
  reach for it only when that is the binding constraint.
- `cg` returns an *approximate* step under its iteration budget. That is
  usually fine — LM's accept/reject absorbs inexactness — but see the
  scheduling pattern below.
- `cholesky`/`cg` square the condition number (they factor `J P J'`). If the
  Gram system is ill-conditioned or implicit derivatives must be accurate,
  **enable `jax_enable_x64` first** — it fixes more numerical trouble than
  any damping adjustment.

## Damping

**Convergence is usually insensitive to the damping parameters — do not tune
them first.** The accept/reject loop self-corrects `init_damping` within a
few steps. Try them when you see specific signatures:

- Many early rejections → raise `init_damping` (start nearer gradient
  descent).
- Long rejection storms in float32 → set `max_damping` (~`1e6`) so damping
  cannot overflow.
- Accept/reject oscillation → bring `damping_decrease`/`damping_increase`
  closer to 1 (e.g. 0.7 / 2.0) for smoother adaptation.
- All steps accepted but progress is slow → lower `init_damping` or decrease
  faster (`damping_decrease=0.3`).

## Schedule Accuracy, Cheap → Exact

Inexact steps are cheap experiments early; near the solution, step quality
limits the convergence rate (and small damping makes the inner system harder
exactly then). Three patterns, in order of preference:

1. **Relative `iterative_tol`** (e.g. `1e-2`) with a generous
   `iterative_maxiter` cap: inner accuracy tightens automatically as the
   residual shrinks. No scheduling code.
2. **Grow the CG budget in a callback** when the loss crosses a threshold —
   single solve call, so implicit AD applies; see the
   [cookbook recipe](callbacks.md#scheduled-inner-solve-accuracy). All of
   `LMHyperparams` is resettable this way.
3. **Stage two solvers**: coarse `cg` solve, then a `cholesky` endgame
   warm-started with `result.x` and `result.lm_state`. The implicit
   derivative is unaffected (it is defined at the returned solution only).

Before scheduling accuracy, check whether a structural `dual_preconditioner`
removes the problem: when the dual operator's conditioning grows with problem
size (metric solves inject \(M^{-1}\) into it), a spectrally equivalent
preconditioner pins the required budget at a small constant — flat
`iterative_maxiter` around 2–20 — where the unpreconditioned budget grows with
refinement. See [Utilities](utilities.md#shermanmorrison-dual-preconditioner).

## What Is Free to Sweep

- **Free (traced, no recompile):** `max_steps`, `atol`/`gtol`/`xtol`, the
  array-valued `LMHyperparams` fields (same dtype; a knob compiled out as
  `None` cannot be switched on), and the *values* of `x0`/`args`/`p`.
- **Recompiles per value (static):** `linear_solver`,
  `geodesic_acceleration`, `cache_jacobian`, `has_aux`, the `Metric`
  callbacks, the callback function identity, and the solver instance itself.
  Construct solvers once at setup scope; an inline `lambda` callback at the
  call site recompiles every solve.

For crude hyperparameter search: sweep `init_damping` on a log scale by
replacing the damping in an `init()` state —
`dataclasses.replace(solver.init(x0, args), damping=jnp.asarray(d))`, traced
and recompile-free — and treat the static list as an outer loop of at most a
few compilations.

When sweeping `p` (or running continuation/homotopy), warm-start each solve
with the previous `result.x` — traced, recompile-free, and usually collapses
the step count.

## Failure Signatures

| symptom | likely cause | remedy |
| --- | --- | --- |
| `status == NONFINITE` at step 0 | bad initial point or data | check `residual_fn(x0, ...)` directly |
| `qr` gives non-finite steps; other solvers fine | rank-deficient Jacobian | use `cholesky`/`cg`/`lsmr` |
| `MAX_STEPS` but loss small and flat | converged without a stopping rule | set `gtol`/`xtol` |
| damping grows without bound (float32 `inf`) | rejection storm | `max_damping`, or check residual scaling |
| every `solve` call recompiles | new solver/callback object per call | construct once at setup scope |
| float32 problem crashes under x64 with `lsmr` | known Lineax dtype issue | use `cholesky`/`cg`/`qr` |
| implicit `jax.jvp`/`vjp` wrong or zero | `p` not in the residual signature, or perturbing `args` | move perturbed quantities into `p` |

## The Metric

In underdetermined problems the metric is not a preconditioner — it selects
*which* solution and *which* implicit derivative you get (minimum-`M`-norm).
For kernel parameterizations use `M = K` (coefficients) or `M = K^{-1}`
(function values); see the [kernel table](gauss_newton.md#choosing-the-metric-with-kernels).
If results look right but derivatives look wrong, check the metric before
anything else.
