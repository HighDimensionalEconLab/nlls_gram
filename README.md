# nlls_gram

[![CI](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/nlls_gram/)
[![PyPI](https://img.shields.io/pypi/v/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![Python versions](https://img.shields.io/pypi/pyversions/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/nlls_gram)](https://github.com/HighDimensionalEconLab/nlls_gram/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Levenberg-Marquardt and Newton nonlinear least squares for JAX pytrees, aimed at
solving systems of equations. The core use case is **underdetermined** systems —
more parameters than residuals — where a zero-residual root is not unique and
something must select *which* interpolant is returned. Two solvers share a
unified `init`/`update`/`solve` protocol and support both unpenalized and
ridge-penalized modes:

- **`LevenbergMarquardt`** (trust-region damping) and **`Newton`** (Armijo
  backtracking line search) with **`ridge=<positive scalar>`** put the selection
  in the **objective**, minimizing \(\|r(x)\|^2 + \lambda\,\|x_m\|_W^2\) for a
  positive-definite metric \(W\) on the metric block of \(x = [x_m; x_f]\) (the
  free block stays unpenalized). Annealing \(\lambda\) toward zero tracks global
  (or local-basin) minimizers toward the minimum-seminorm — e.g.
  minimum-RKHS-norm — interpolant, by classical nonlinear Tikhonov
  regularization. Every inner problem is a well-posed NLLS.
- **`LevenbergMarquardt`** and **`Newton`** with **`ridge=None`** (the default)
  put the selection in the **step geometry**: minimizing \(\|r(x)\|^2\) where
  the trust region or Gauss-Newton pseudoinverse is measured in \(W\), so the
  step direction is the minimum-\(W\)-norm correction.

Both take a residual over `(x)`, `(x, args)`, or `(x, args, p)`, flatten any
pytree `x`, expose per-step `update(...)` and an internally jitted `solve(...)`
loop with callbacks and multi-start, and differentiate `solve(...).x` with
respect to `p` through a custom implicit rule — no unrolling.

## Install

```bash
uv add nlls-gram
```

For GPU use, install the JAX accelerator build that matches your hardware:

```bash
uv add nlls-gram "jax[cuda13]"
```

## Minimum-RKHS-norm interpolation

```python
import jax.numpy as jnp
from nlls_gram import AnnealRidge, LevenbergMarquardt, RepeatedFactorMetric

# W = blockdiag(K, K): the RKHS seminorm over two coefficient blocks. The
# constructor takes the FACTOR; shift a semidefinite K by epsilon*I first.
metric = RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=2)

solver = LevenbergMarquardt(collocation_residual, metric=metric, ridge=1e-4)

# Anneal the ridge toward the interpolating limit on stationarity.
anneal = AnnealRidge(ridge_floor=1e-10)
result = solver.solve(
    x0, callback=anneal, user_state=anneal.init_state(), gtol=1e-8, atol=1e-8
)
```

## General nonlinear least squares

```python
import jax, jax.numpy as jnp
from nlls_gram import LevenbergMarquardt


def residual(x, args, p):
    return args["design"] @ x - p["target"]


solver = LevenbergMarquardt(residual)
result = solver.solve(
    jnp.zeros(8), {"design": design}, p={"target": y}, max_steps=200, atol=1e-8
)

# The solution is differentiable in p, at a cost independent of max_steps.
sensitivity = jax.grad(
    lambda p: jnp.sum(
        solver.solve(jnp.zeros(8), {"design": design}, p=p, max_steps=200, atol=1e-8).x
        ** 2
    )
)({"target": y})
```

Pass `metric=` to weight the damping geometry.

## Linear solvers

The same typed configs serve both solvers: `Cholesky()` (the default,
auto-selecting the smaller of the dual and normal systems), `QR()`
(damping-row QR in the forward role — stable at tiny damping and rank-safe when
damped), `CG(precond)` and `GramCG(precond)` (matrix-free in parameter and
residual space), `LU()` for square systems, and `SVD()` for rank-deficient
steps or implicit-AD tangents. A knob that exists for only one method is a
field on that method, so it cannot be passed with another.

## Row-wise residuals

For an expensive per-state residual vmapped over a simulated path, replace
that `jax.vmap` with `vmap_rows`: the same values, with each row's gradient
computed once per linearization and reused across the Jacobian assembly
instead of one pass of the row function per row and Jacobian column (or
cotangent row).

```python
from nlls_gram import vmap_rows


@vmap_rows(in_axes=(0, None, None))
def euler_residual(state, policy, p): ...
```

## Docs

<https://highdimensionaleconlab.github.io/nlls_gram/>

## License

MIT.
