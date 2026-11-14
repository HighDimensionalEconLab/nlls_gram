# nlls_gram

`nlls_gram` solves nonlinear equations, nonlinear least-squares problems, and generalized Tikhonov interpolation problems over JAX pytrees using Euclidean outer step geometry.

Two solvers share one `init`/`update`/`solve` protocol, one linear-solver menu, and one implicit differentiation rule:

| Solver | `ridge=None` | `ridge=<positive scalar>` | Globalization |
|---|---|---|---|
| `LevenbergMarquardt` | \(\frac{1}{2}\|r(\theta)\|_2^2\) | \(\frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta\) | Euclidean trust-region damping \(\mu I_n\) + Armijo backtracking |
| `Newton` | \(\frac{1}{2}\|r(\theta)\|_2^2\) | \(\frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta\) | Optional residual-coupled scalar damping + Armijo backtracking |

`info.loss` consistently reports \(2 F_\rho(\theta) = \|r(\theta)\|_2^2 + \rho\,\theta^\top Q \theta\).

## Install

```bash
uv add nlls-gram
```

## Residual interface

`residual_fn` accepts `(x)`, `(x, args)`, or `(x, args, p)` and returns a residual pytree, a `Residuals` / `Inequality` / `Complementarity` container, or `(residual, aux)` when `has_aux=True`.

- `args` carries solver-inert auxiliary data.
- `p` carries external parameters with respect to which `solve(...).x` is implicitly differentiable.

```python
import jax.numpy as jnp
from nlls_gram import LevenbergMarquardt


def residual(x, args, p):
    return args["design"] @ x - p["target"]


solver = LevenbergMarquardt(residual)
result = solver.solve(
    jnp.zeros(8), {"design": design}, p={"target": y}, max_steps=200, atol=1e-8
)
```

## Generalized Tikhonov interpolation

When `ridge` is positive, `ridge_metric` specifies a positive semidefinite quadratic operator \(Q = L^\top L\) in the objective while keeping trust-region damping strictly Euclidean (\(\mu \|d\|_2^2\)). Passing `ridge_metric=None` uses the Euclidean identity \(Q = I_n\). Supplying a non-`None` `ridge_metric` when `ridge=None` raises `ValueError`.

```python
import jax.numpy as jnp
from nlls_gram import AnnealRidge, LevenbergMarquardt, RepeatedFactorRidge

ridge_metric = RepeatedFactorRidge(jnp.linalg.cholesky(K, upper=True), repeats=2)
solver = LevenbergMarquardt(
    collocation_residual,
    ridge=1e-4,
    ridge_metric=ridge_metric,
)
anneal = AnnealRidge(ridge_floor=1e-10)
result = solver.solve(x0, callback=anneal, user_state=anneal.init_state(), gtol=1e-8)
```

## Linear solvers

Numerical preconditioners live on `linear_solver` and `ad_solver`, never on the outer solver constructor.

| Config | Space | Form | Notes |
|---|---|---|---|
| `Cholesky(form="auto", preconditioner=None)` | dense | `gram` or `normal` | Default dense solver with optional `DiagonalPreconditioner` |
| `QR(form="auto", preconditioner=None)` | dense | `gram` or `normal` | Avoids squaring \(\operatorname{cond}(J)\) in both forward and implicit AD |
| `CG(preconditioner, ...)` | parameter (\(n\)) | matrix-free normal | Supports matvec-only `RidgeMetric` |
| `GramCG(preconditioner, ...)` | residual (\(m\)) | matrix-free Gram | Unpenalized wide systems (\(m \ll n\)) |
| `LU()` | parameter (\(n\)) | square \(J\) | Undamped square systems (\(m = n\), `ridge=None`) |
| `SVD(rcond=None)` | parameter (\(n\)) | pseudoinverse | Rank-aware augmented or unpenalized least squares |

## Documentation guide

- [Tikhonov Regularization](ridge_lm.md)
- [Newton](newton.md)
- [Ridge Operators and Preconditioners](metrics.md)
- [Constraints](complementarity.md)
- [Callbacks](callbacks.md)
- [Multi-Start](multi_start.md)
- [Implicit Differentiation](implicit_ad.md)
- [Row-wise Residuals](vmap_rows.md)
- [Tuning](tuning_guide.md)
- [API](api.md)
