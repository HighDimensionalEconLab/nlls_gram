# Metrics

## The Metric Object

A custom metric is passed as a single `metric=Metric(...)` argument. The
`Metric` dataclass holds up to four callbacks that operate on the flattened
parameter vector produced internally by `ravel_pytree`. Let \(P=M^{-1}\), and
let \(S\) satisfy \(SS^\top=M^{-1}\).

| Field | Meaning |
| --- | --- |
| `solve(x)` | \(M^{-1}x = Px\) |
| `norm(x)` | \(\sqrt{x^\top Mx}\) |
| `inv_sqrt(x)` | \(Sx\) |
| `inv_sqrt_transpose(x)` | \(S^\top x\) |

Fields left as `None` (and `metric=None` itself) default to the identity
metric.

Shape requirements:

- `solve` must support vectors `(n_params,)` and matrices `(n_params, k)`.
- `inv_sqrt` must support vectors `(n_params,)`; matrices
  `(n_params, n_residuals)` are additionally required for implicit
  differentiation when the metric has no `solve`.
- `inv_sqrt_transpose` must support matrices `(n_params, n_residuals)` for QR.
- `norm` only needs to support vectors `(n_params,)`.

Validation rules:

- For `linear_solver in {"cholesky", "cg"}`, a custom metric requires
  `metric.solve`.
- For `linear_solver in {"qr", "lsmr"}`, a custom metric requires both
  `metric.inv_sqrt` and `metric.inv_sqrt_transpose`.
- If geodesic acceleration is enabled (the default) and a custom metric is
  supplied, `metric.norm` is required — supply it or pass
  `geodesic_acceleration=False`.
- The solver does not infer `norm`, `inv_sqrt`, or `inv_sqrt_transpose` from
  `solve`.

`norm` is separate because `solve` applies \(M^{-1}\), while the norm needs
\(M\):

$$
\|x\|_M = \sqrt{x^\top Mx}.
$$

Recovering that norm from a black-box \(M^{-1}\) solve would require another
inverse operation. Likewise, a square-root factor \(S\) is not generally
recoverable from an arbitrary solve callback.

## Cholesky Metric Helper

For a dense metric \(M=LL^\top\) with \(L\) lower triangular (the form
returned by `jnp.linalg.cholesky`), use:

```python
import jax.numpy as jnp

from nlls_gram import metric_from_cholesky

L = jnp.linalg.cholesky(metric_matrix)
metric = metric_from_cholesky(L)
```

The helper returns a `Metric` with all four callbacks filled in. Further
constructors — tridiagonal-precision, diagonal, and block-diagonal metrics,
plus a Sherman–Morrison dual preconditioner — are collected in
[Utilities](utilities.md).

## Metric Example

```python
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt, metric_from_cholesky


def residual_fn(theta, args):
    matrix, target = args
    return matrix @ theta - target


metric_matrix = jnp.array([[2.0, 0.2], [0.2, 1.0]])
L = jnp.linalg.cholesky(metric_matrix)

solver = UnderdeterminedLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    metric=metric_from_cholesky(L),
)
```

## Matrix-Free Metric Example

For CG, the metric only needs a solve callback. This can wrap an iterative solve
or any JAX-linear operation implementing \(M^{-1}x\), without materializing a
dense inverse:

```python
import jax.scipy.sparse.linalg as jsp_sparse_linalg

from nlls_gram import Metric, UnderdeterminedLevenbergMarquardt


def metric_matvec(x):
    return kernel_matvec(x) + ridge * x


def metric_solve(x):
    solution, _ = jsp_sparse_linalg.cg(metric_matvec, x, maxiter=32)
    return solution


solver = UnderdeterminedLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    linear_solver="cg",
    metric=Metric(solve=metric_solve),
)
```

This changes the LM damping metric. It is not a preconditioner for the inner CG
iteration; it changes the step being solved for.
