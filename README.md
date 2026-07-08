# nlls_gram

[![CI](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/nlls_gram/)
[![PyPI](https://img.shields.io/pypi/v/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![Python versions](https://img.shields.io/pypi/pyversions/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/nlls_gram)](https://github.com/HighDimensionalEconLab/nlls_gram/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Metric-aware Levenberg-Marquardt nonlinear least-squares for JAX pytrees, aimed
at underdetermined or interpolating problems where the number of parameters is
larger than the number of residuals.

`UnderdeterminedLevenbergMarquardt` minimizes a user residual taking
`(params)`, `(params, aux)`, or `(params, aux, p)`, always in that order.
Parameters may be any JAX pytree; internally they
are flattened with `jax.flatten_util.ravel_pytree`. The default dense solver
uses the residual-space Gram system, with QR, CG, and LSMR alternatives. Use
`update(...)` for a single LM step or `solve(...)` for an internally jitted loop.

## Problem

At each iteration the solver builds a step \(s\) from the metric-damped
linearized subproblem

$$
\min_s \frac12\|r + Js\|_2^2 + \frac{\lambda}{2}s^\top M s,
\qquad M \succ 0.
$$

The default is \(M=I\). For kernel/RKHS coefficient problems, if

$$
f_\alpha(x)=\sum_{j=1}^n \alpha_j K(x,x_j),
$$

then

$$
\|f_\alpha\|_{\mathcal H_K}^2 = \alpha^\top K\alpha,
$$

so the natural parameter metric is \(M=K\), not the Euclidean metric.

## Install

```bash
uv add nlls-gram
```

For GPU use, install the JAX accelerator build that matches your hardware, for
example:

```bash
uv add nlls-gram "jax[cuda13]"
```

## Minimal Example

```python
import jax
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt


def residual_fn(params, aux):
    x, y = aux
    return params["a"] * jnp.exp(params["b"] * x) - y


x = jnp.linspace(0.0, 2.0, 20)
y = 2.0 * jnp.exp(-1.0 * x)
params = {"a": 1.0, "b": 0.0}

solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
state = solver.init()


@jax.jit
def train_step(params, state):
    return solver.update(params, state, (x, y))


for _ in range(50):
    params, state, info = train_step(params, state)

print(params["a"], params["b"])  # approximately 2.0, -1.0
```

For a simple full solve loop:

```python
result = solver.solve(params, (x, y), max_steps=50, atol=1e-8)
params = result.params
```

`solve` stops on a residual-norm `atol`, gradient-norm `gtol`, or
accepted-step-norm `xtol` (each `0.0` disables), always enforces `max_steps`,
and takes a traceable callback for custom stopping, epoch-style data
resampling, and per-step history recording; the docs have a cookbook.

`solve(...).params` also supports custom implicit JVP/VJP with respect to `p`;
the docs give the metric-minimum-norm formula and a minimal `jax.jvp` /
`jax.vjp` example. The metric matters for underdetermined roots because it
selects which tangent is the minimum-norm solution. The per-step `update(...)`
interface does not define the implicit AD rule.

## Metric Example

For a dense SPD metric \(M = LL^\top\), use the Cholesky helper:

```python
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt, metric_from_cholesky

L = jnp.linalg.cholesky(metric_matrix)
solver = UnderdeterminedLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    metric=metric_from_cholesky(L),
)
```

The `Metric` callbacks act on the flattened parameter vector. The docs give
the exact callback contract, branch formulas, and validation rules.

## Solvers

- `linear_solver="cholesky"`: dense residual-space Gram solve, the default.
- `linear_solver="qr"`: dense QR solve of the whitened-step problem.
- `linear_solver="cg"`: matrix-free residual-space CG.
- `linear_solver="lsmr"`: matrix-free Lineax LSMR on the damped least-squares
  problem.

All four solve the same metric-damped linearized subproblem up to the accuracy
of the chosen linear solver.

## Docs and Alternatives

Full docs: https://highdimensionaleconlab.github.io/nlls_gram/

For a broader JAX nonlinear solver library, see
[Optimistix](https://github.com/patrick-kidger/optimistix). `nlls_gram` is more
specialized: it focuses on underdetermined nonlinear least-squares, residual
space Gram solves, and explicit parameter-space metrics.

## License

MIT
