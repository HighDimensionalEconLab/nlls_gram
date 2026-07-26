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

- `geodesic_acceptance_ratio` — lower it when the second-order correction is
  being accepted on steps where it overshoots.

Damping is bounded only by an anti-underflow floor at
`jnp.finfo(residual.dtype).tiny`; there is no upper cap. Letting damping fall
freely is the point — the endgame wants it to vanish so the step approaches
Gauss–Newton and the minimum-norm limit. Clamp it in a callback on the rare
problem that needs a bound (see [Callbacks](callbacks.md)), and use
`RidgeLevenbergMarquardt` when what you actually want is regularization.

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

## GPU matmul precision in float32

On an NVIDIA GPU from Ampere onward, XLA serves float32 `dot_general` from
TF32 tensor cores by default: a 10-bit mantissa, so about \(10^{-3}\) relative
error rather than float32's \(10^{-7}\). That is a poor trade here. Forming a
Gram or normal matrix already squares the condition number, so paying TF32 on
top spends roughly three decimal digits before the factorization starts —
enough to move a converged step and to leave an implicit tangent visibly
wrong.

Every product **inside** this package is therefore pinned to
`jax.lax.Precision.HIGHEST`, unconditionally and with no opt-out: the Gram and
normal assembly, the matrix-free CG operator, the metric factor applications,
the preconditioner contractions, and the implicit-AD solves. A float32 solve
answers the same on GPU as on CPU, where the setting is a no-op.

That is necessary but **not sufficient**, and on a GPU you must do one more
thing. The package cannot reach the matmuls inside **your** residual, and the
Jacobian is differentiated through those — so a TF32 residual hands the solver
a Jacobian that is already wrong, and pinning everything downstream cannot
recover it. Since a residual here is usually a neural network or a kernel
evaluation, that is the normal case, not the exception. Measured on a dense
underdetermined float32 problem on an RTX 3090:

| | relative error |
|---|---|
| default | \(3.3 \times 10^{-4}\) |
| `jax_default_matmul_precision="highest"` | \(1.6 \times 10^{-7}\) |

So set it near the top of any script that will run on a GPU:

```python
import jax

jax.config.update("jax_default_matmul_precision", "highest")
```

or export `JAX_DEFAULT_MATMUL_PRECISION=highest`. TF32's speed only pays on
large well-conditioned matmuls, which is not what a least-squares solve is
doing, so there is no reason to leave it off.

The failure signature, if it is ever missed: results that agree between CPU
and GPU to about three digits and no further, with the gap concentrated in
whichever quantity passed through the most products — typically the tangent
before the primal.

CI has no GPU runner, so `tests/test_gpu.py` is skipped there. Run it on a
GPU box explicitly:

```bash
uv run --group gpu pytest -q
```
