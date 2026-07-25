# Tuning

## Picking a solver

Start with `RidgeLevenbergMarquardt` when the interpolant is the deliverable
and `LevenbergMarquardt` when the root is a means to an end. Defaults —
`Cholesky()`, `geodesic_acceleration=True`, `cache_jacobian=True` — are the
right first try for both.

Change the linear solver when a specific symptom appears:

| symptom | try |
|---|---|
| \(n\) in the thousands, dense \(J\) too big to form | `GramCG(precond)` when \(m \ll n\), else `CG(precond)` |
| step accuracy degrades as ridge or damping shrinks | `QR()` — it works at \(\operatorname{cond}(A)\), not \(\operatorname{cond}(A)^2\) |
| non-finite step on a rank-deficient Jacobian | `QR()` (rank-safe) or `Cholesky()` |
| tangent is NaN or wrong under implicit AD | `ad_solver=SVD()` |

`Cholesky()` picks the smaller of the \(m \times m\) dual and the \(n \times
n\) normal system by shape. Forcing `form=` is a cost decision only — for
\(\lambda > 0\) both give the same step.

## Inner-solve budget

The Krylov configs default to `tol=None` (a dtype default: `1e-10` in float64,
`1e-6` in float32). A cheap fixed budget early and an exact solve late is
usually better than either alone:

```python
CG(precond, tol=0.0, atol=0.0, maxiter=8)     # cheap, fixed
```

then raise `iterative_maxiter` from a callback as the loss falls — see
[Callbacks](callbacks.md). The budget is traced state, so changing it costs no
recompilation.

## Damping

`init_damping=1e-3` with `damping_decrease=0.5` / `damping_increase=4.0` is
the standard schedule. Reach for the others only on evidence:

- `min_damping` — lower it (e.g. `1e-12`) when the minimum-norm limit is the
  point and the default floor is truncating the endgame;
- `max_damping` — cap it when a bad region sends damping to infinity and the
  solver stalls instead of failing;
- `geodesic_acceptance_ratio` — lower it when the second-order correction is
  being accepted on steps where it overshoots.

## The ridge schedule

For `RidgeLevenbergMarquardt`, a fixed ridge leaves an \(O(\text{ridge})\)
bias. `AnnealRidge(ridge_floor=...)` anneals toward the interpolating
limit, advancing a level whenever the current one is stationary.

Calibrate `gtol` as roughly `1e-3 * ridge * sqrt(q(x*))` — the reported
`info.penalty_grad_norm` is exactly `sqrt(penalty_value)`, so a pilot run
gives you `q(x*)` to an order of magnitude. Choose `atol` **between** the
ridge-floor residual and the last intermediate level's residual (they differ
by roughly `1/decrease`), so the solve can only stop at the floor.

If the anneal freezes: widen `decrease` (e.g. `0.01`) first, since larger
jumps keep the per-level references generous; `stall_rtol ~ 0.99` is the
escape hatch when it does not.

## Free to sweep without recompiling

Traced, so changing them reuses the compiled loop: `atol`/`gtol`/`xtol`, the
damping and ridge values, `p`, `args`, `max_steps` (without `save_steps`), and
every `LMHyperparams` field.

Traced too: every metric and preconditioner **array** — the instances ride
in the carried state, so a fresh equal-config instance (and a callback
rebuild) reuses the compiled loop.

Static, so changing them compiles a new program: the residual, the
linear-solver config, `has_aux`, `cache_jacobian`, `geodesic_acceleration`,
the callback, any shape, and any instance **static field** (a metric's
`size`, a block-eigen family layout).

## Failure signatures

| you see | likely |
|---|---|
| `NONFINITE` at step 0 | `x0` is outside the residual's domain |
| status `MAX_STEPS`, loss flat | damping stuck high; check `info.damping` |
| `CONVERGED` but the residual is large | `atol` fired on the wrong scale, or `gtol` alone stopped a ridge level |
| tangent finite but wrong | the `ad_solver`'s operator is singular for this shape — see [Metric LM](metric_lm.md#rank-deficiency-and-implicit-ad) |
| every solve recompiles | a metric, preconditioner, or callback rebuilt per call |
