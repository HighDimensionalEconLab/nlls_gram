# nlls_gram

Levenberg-Marquardt nonlinear least squares for JAX, built for
**underdetermined interpolation**: problems whose residual has many more
parameters than equations, so a zero-residual root is not unique and *which*
root you get is part of the contract.

Two solvers share one `init`/`update`/`solve` protocol, one linear-solver
menu, and one implicit-AD contract. Both accept `ridge=None` (unpenalized
nonlinear least squares) or `ridge=<positive scalar>` (Tikhonov-penalized
ridge objective):

| | `LevenbergMarquardt(..., ridge=...)` | `LevenbergMarquardt` | `Newton(..., ridge=...)` | `Newton` |
|---|---|---|---|---|
| minimizes | \(\|r(x)\|^2 + \lambda\,\|x_m\|_W^2\) | \(\|r(x)\|^2\) | \(\|r(x)\|^2 + \lambda\,\|x_m\|_W^2\) | \(\|r(x)\|^2\) |
| globalization | trust-region damping + Armijo backtracking | trust-region damping + Armijo backtracking | Armijo backtracking line search | Armijo backtracking line search |
| step direction | damped ridge minimum-\(W\)-norm | damped minimum-\(W\)-norm | ridge minimum-\(W\)-norm | minimum-\(W\)-norm \(-\bar F^{-1}(J \bar F^{-1})^+ r\) |
| use when | the interpolant is the deliverable | rank-deficient or curved paths need trust-region damping | RKHS / kernel ridge problems where full steps dominate | well-conditioned systems or with residual-coupled `damping > 0` + `geodesic_acceleration=True` |

## Install

```bash
uv add nlls-gram      # or: pip install nlls-gram
```

JAX is the only dependency.

## Residual interface

`residual_fn` takes `(x)`, `(x, args)`, or `(x, args, p)` — always in that
order — and returns a residual pytree, or `(residual, aux)` with
`has_aux=True`. `x` is any JAX pytree.

- `args` is **solver-inert** auxiliary data: sampled batches, module
  graphdefs, anything the solver should thread through untouched.
- `p` is everything you may want a **total derivative of the solution** with
  respect to. `solve(...).x` carries a custom implicit AD rule in `p`.

```python
import jax.numpy as jnp
from nlls_gram import LevenbergMarquardt


def residual(x, args, p):
    return args["design"] @ x - p["target"]


solver = LevenbergMarquardt(residual)
result = solver.solve(
    jnp.zeros(8), {"design": design}, p={"target": y}, max_steps=200, atol=1e-8
)
result.x, result.status, result.steps
```

## Minimum-norm interpolation

```python
import jax.numpy as jnp
from nlls_gram import AnnealRidge, LevenbergMarquardt, RepeatedFactorMetric

# W = blockdiag(K, K) for a kernel Gram matrix K: the RKHS seminorm over two
# coefficient blocks. The constructor takes the FACTOR.
metric = RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=2)

solver = LevenbergMarquardt(collocation_residual, metric=metric, ridge=1e-4)

# Anneal the ridge toward the interpolating limit on stationarity.
anneal = AnnealRidge(ridge_floor=1e-10)
result = solver.solve(
    x0, callback=anneal, user_state=anneal.init_state(), gtol=1e-8, atol=1e-8
)
```

The metric covers the leading `metric.size` coordinates; anything past that is
a **free block**, unpenalized by the ridge solver. Calibrate `gtol` as roughly
`1e-3 * ridge * sqrt(q(x*))`, using the reported
`info.penalty_grad_norm = sqrt(penalty_value)`.

## Linear solvers

The same typed configs serve both solvers (`LevenbergMarquardt` and `Newton`),
in both the forward and the implicit-AD role (`QR()` is forward-only). A knob
that exists for only one method is a field on that method, so it cannot be
passed with another.

| config | system | dimension | when |
|---|---|---|---|
| `Cholesky()` | dense, auto gram/normal | \(\min(m, n)\) | the default |
| `QR()` | damping-row QR, auto gram/normal | \(\min(m, n)\) | forward-only; tiny damping or ridge; rank-safe when damped (`LM` or `damping > 0`); float32 |
| `CG(precond)` | matrix-free normal | \(n\) | \(n \lesssim m\), no dense \(J\) |
| `GramCG(precond)` | matrix-free dual | \(m\) | \(m \ll n\), no dense \(J\) |
| `LU()` | dense solve of \(J\) or \(B\) | \(n\) | `Newton` forward and `ad_solver` for square systems |
| `SVD()` | pseudoinverse | — | `Newton` forward and `ad_solver` for rank-deficient systems |

For \(\lambda > 0\) the gram and normal forms compute the *same* step (the
push-through identity), so the choice is about cost, not semantics. A
`preconditioner` is required for the Krylov configs —
`IdentityPreconditioner()` is the explicit opt-out.

When `ridge=None`, `ad_solver=None` (the default) matches the forward family
where its undamped operator is invertible, and otherwise picks by shape: `LU()`
when \(m = n\), `SVD()` when not. When `ridge > 0`, `ad_solver=None` uses
matrix-free normal `CG` when the forward solver is `CG` and dense `Cholesky`
otherwise.

## Where to go next

- [Ridge LM](ridge_lm.md) — the objective, whitening, ridge continuation, stopping
- [Metric LM](metric_lm.md) — the damping geometry and its minimum-norm limit
- [Newton](newton.md) — undamped minimum-norm Newton / Gauss-Newton with Armijo backtracking
- [Metrics and preconditioners](metrics.md) — the two hook types
- [Callbacks](callbacks.md) — the solve loop and its cookbook
- [Implicit differentiation](implicit_ad.md) — the AD contract
- [Row-wise residuals](vmap_rows.md) — `vmap_rows` for simulated-path residuals
- [Tuning](tuning_guide.md) — picking a solver and a schedule
- [API](api.md)
