# nlls_gram

[![CI](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/nlls_gram/)
[![PyPI](https://img.shields.io/pypi/v/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![Python versions](https://img.shields.io/pypi/pyversions/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/nlls_gram)](https://github.com/HighDimensionalEconLab/nlls_gram/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Levenberg-Marquardt nonlinear least squares for JAX pytrees, aimed at solving
systems of equations. The core use case is **underdetermined** systems — more
parameters than residuals — where a zero-residual root is not unique and
something must select *which* interpolant is returned. Two solvers differ in
where that selection lives:

- **`RidgeLevenbergMarquardt`** puts it in the **objective**, minimizing
  \(\|r(x)\|^2 + \lambda\,\|x_m\|_W^2\) for a positive-definite metric
  \(W\) on the metric block of \(x = [x_m; x_f]\) (the free block stays
  unpenalized). Annealing \(\lambda\) toward zero converges to the
  minimum-seminorm — e.g. minimum-RKHS-norm — interpolant, by classical
  nonlinear Tikhonov regularization. Every inner problem is a well-posed NLLS.
- **`LevenbergMarquardt`** puts it in the **damping geometry**: standard
  damped LM whose trust region is measured in \(W\), so the small-damping
  Gauss-Newton limit is the minimum-\(W\)-norm correction.

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
from nlls_gram import AnnealRidge, RidgeLevenbergMarquardt, RepeatedFactorMetric

# W = blockdiag(K, K): the RKHS seminorm over two coefficient blocks. The
# constructor takes the FACTOR; shift a semidefinite K by epsilon*I first.
metric = RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=2)

solver = RidgeLevenbergMarquardt(collocation_residual, metric=metric, ridge=1e-4)

# Anneal the ridge toward the interpolating limit on stationarity.
anneal = AnnealRidge(ridge_floor=1e-10)
result = solver.solve(x0, callback=anneal, user_state=anneal.init_state(),
                      gtol=1e-8, atol=1e-8)
```

## General nonlinear least squares

```python
import jax, jax.numpy as jnp
from nlls_gram import LevenbergMarquardt

def residual(x, args, p):
    return args["design"] @ x - p["target"]

solver = LevenbergMarquardt(residual)
result = solver.solve(jnp.zeros(8), {"design": design}, p={"target": y},
                      max_steps=200, atol=1e-8)

# The solution is differentiable in p, at a cost independent of max_steps.
sensitivity = jax.grad(
    lambda p: jnp.sum(solver.solve(jnp.zeros(8), {"design": design}, p=p,
                                   max_steps=200, atol=1e-8).x ** 2)
)({"target": y})
```

Pass `metric=` to weight the damping geometry.

## Linear solvers

The same typed configs serve both solvers, forward and in the implicit-AD
role: `Cholesky()` (the default, auto-selecting the smaller of the dual and
normal systems), `QR()` (damping-row QR — stable at tiny damping and
rank-safe), `CG(precond)` and `GramCG(precond)` (matrix-free in parameter and
residual space), and `SVD()` for rank-deficient tangents. A knob that exists
for only one method is a field on that method, so it cannot be passed with
another.

## Docs

<https://highdimensionaleconlab.github.io/nlls_gram/>

## License

MIT.
