# nlls_gram

Levenberg-Marquardt nonlinear least squares for JAX, built for
**underdetermined interpolation**: problems whose residual has many more
parameters than equations, so a zero-residual root is not unique and *which*
root you get is part of the contract.

Two solvers differ in where that selection lives.

| | `RidgeLevenbergMarquardt` | `LevenbergMarquardt` |
|---|---|---|
| minimizes | \(\|r(x)\|^2 + \lambda\,\|x_m\|_W^2\) | \(\|r(x)\|^2\) |
| selection lives in | the **objective** | the **damping geometry** |
| converges to | the minimum-\(W\)-seminorm interpolant | the minimum-\(W\)-norm correction limit |
| use when | the interpolant is the deliverable | the root is a means to an end |

Both share one `init`/`update`/`solve` protocol, one `Metric`, one linear-solver
menu, and one implicit-AD contract.

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
result = solver.solve(jnp.zeros(8), {"design": design}, p={"target": y},
                      max_steps=200, atol=1e-8)
result.x, result.status, result.steps
```

## Minimum-norm interpolation

```python
import jax.numpy as jnp
from nlls_gram import AnnealRidge, RidgeLevenbergMarquardt, RepeatedFactorMetric

# W = blockdiag(K, K) for a kernel Gram matrix K: the RKHS seminorm over two
# coefficient blocks. The constructor takes the FACTOR.
metric = RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=2)

solver = RidgeLevenbergMarquardt(collocation_residual, metric=metric, ridge=1e-4)

# Anneal the ridge toward the interpolating limit on stationarity.
anneal = AnnealRidge(ridge_floor=1e-10)
result = solver.solve(x0, callback=anneal, user_state=anneal.init_state(),
                      gtol=1e-8, atol=1e-8)
```

The metric covers the leading `metric.size` coordinates; anything past that is
a **free block**, unpenalized by the ridge solver. Calibrate `gtol` as roughly
`1e-3 * ridge * sqrt(q(x*))`, using the reported
`info.penalty_grad_norm = sqrt(penalty_value)`.

## Linear solvers

The same typed configs serve both solvers, in both the forward and the
implicit-AD role. A knob that exists for only one method is a field on that
method, so it cannot be passed with another.

| config | system | dimension | when |
|---|---|---|---|
| `Cholesky()` | dense, auto gram/normal | \(\min(m, n)\) | the default |
| `QR()` | damping-row QR of \([\tilde J;\sqrt\lambda I]\) | \(n\) | tiny damping or ridge; rank-safe |
| `CG(precond)` | matrix-free normal | \(n\) | \(n \lesssim m\), no dense \(J\) |
| `GramCG(precond)` | matrix-free dual | \(m\) | \(m \ll n\), no dense \(J\) |
| `SVD()` | pseudoinverse | — | `ad_solver` only: rank-deficient tangents |

For \(\lambda > 0\) the gram and normal forms compute the *same* step (the
push-through identity), so the choice is about cost, not semantics. A
`preconditioner` is required for the Krylov configs —
`IdentityPreconditioner()` is the explicit opt-out.

`ad_solver=None` (the default) matches the forward family, falling back to
`Cholesky()` where the forward config's undamped operator would be singular.

## Where to go next

- [Ridge LM](ridge_lm.md) — the objective, whitening, ridge continuation, stopping
- [Metric LM](metric_lm.md) — the damping geometry and its minimum-norm limit
- [Metrics and preconditioners](metrics.md) — the two hook types
- [Callbacks](callbacks.md) — the solve loop and its cookbook
- [Implicit differentiation](implicit_ad.md) — the AD contract
- [Tuning](tuning_guide.md) — picking a solver and a schedule
- [API](api.md)
