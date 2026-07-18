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

`LevenbergMarquardt` minimizes a user residual taking
`(x)`, `(x, args)`, or `(x, args, p)`, always in that order. The unknown `x`
may be any JAX pytree; internally it
is flattened with `jax.flatten_util.ravel_pytree`. The default dense solver
uses the residual-space Gram system, with QR and CG alternatives. Use
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

Near the interpolation threshold the residuals are small, damping falls, and
LM becomes metric Gauss-Newton: each step is the minimum-\(M\)-norm correction
solving the linearized residual equations,

$$
s = -M^{-1}J^\top\left(JM^{-1}J^\top\right)^{-1}r
= \arg\min_s \|s\|_M
\;\;\text{s.t.}\;\;
r + Js = 0.
$$

With an RKHS metric this selects minimum-RKHS-norm corrections — kernel
methods let you control exactly which norm that is. The docs derive this and
the large-damping limit (metric gradient descent).

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

from nlls_gram import LevenbergMarquardt


def residual_fn(x, args):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


ts = jnp.linspace(0.0, 2.0, 20)
ys = 2.0 * jnp.exp(-1.0 * ts)
x = {"a": 1.0, "b": 0.0}

solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init(x, (ts, ys))


@jax.jit
def train_step(x, lm_state):
    return solver.update(x, lm_state, (ts, ys))


for _ in range(50):
    x, lm_state, info = train_step(x, lm_state)

print(x["a"], x["b"])  # approximately 2.0, -1.0
```

For a simple full solve loop:

```python
result = solver.solve(x, (ts, ys), max_steps=50, atol=1e-8)
x = result.x
```

`solve` stops on a residual-norm `atol`, gradient-norm `gtol`, or
accepted-step-norm `xtol` (each `0.0` disables), always enforces `max_steps`,
and takes a traceable callback for custom stopping, epoch-style data
resampling, and per-step history recording; the docs have a cookbook.

`solve(...).x` also supports custom implicit JVP/VJP with respect to `p`;
the docs give the metric-minimum-norm formula and a minimal `jax.jvp` /
`jax.vjp` example. With `linear_solver="cg"` the default implicit AD rule is
matrix-free; pass `implicit_solver="cholesky"` to restore the dense rule. The
metric matters for underdetermined roots because it selects which tangent is
the minimum-norm solution. The per-step `update(...)` interface does not
define the implicit AD rule.

## Metric Example

For a dense SPD metric \(M = LL^\top\), use the Cholesky helper:

```python
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt, metric_from_cholesky

L = jnp.linalg.cholesky(metric_matrix)
solver = LevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    metric=metric_from_cholesky(L),
)
```

The `Metric` callbacks act on the flattened parameter vector. The docs give
the exact callback contract, branch formulas, and validation rules, plus
structural constructors (`metric_from_tridiagonal_precision`,
`metric_from_state_space` and `metric_from_quasiseparable` for exact O(n)
Matérn/state-space kernel Grams, `metric_from_diagonal`,
`blockdiag_metric`) so common metrics need no callback plumbing.

## Solvers

- `linear_solver="cholesky"`: dense residual-space Gram solve, the default.
- `linear_solver="qr"`: dense QR solve of the whitened-step problem (requires
  a full-row-rank Jacobian).
- `linear_solver="augmented_qr"`: direct augmented QR in parameter space;
  robust to rank-deficient Jacobians when damping is positive and best suited
  to small systems.
- `linear_solver="cg"`: matrix-free residual-space CG. A `dual_preconditioner`
  is required (e.g. `sherman_morrison_preconditioner`, or the randomized
  `nystrom_preconditioner` for neural-network duals; pass
  `identity_preconditioner()` to run unpreconditioned CG explicitly);
  `implicit_solver="auto"` keeps `solve(...).x` matrix-free under AD and
  requires `implicit_preconditioner` the same way — at construction, even
  if the solve is never differentiated.

All four solve the same metric-damped linearized subproblem up to the accuracy
of the chosen linear solver.

## Docs and Alternatives

Full docs: https://highdimensionaleconlab.github.io/nlls_gram/

Working with an AI assistant? Point it at
[`docs/tuning_guide.md`](https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/)
if it doesn't pick it up automatically — solver selection, damping heuristics,
inner-solve scheduling, and failure signatures, written to be read by humans
and agents alike (also indexed via the site's `llms.txt`).

For a broader JAX nonlinear solver library, see
[Optimistix](https://github.com/patrick-kidger/optimistix). `nlls_gram` is more
specialized: it focuses on underdetermined nonlinear least-squares, residual
space Gram solves, and explicit parameter-space metrics.

## License

MIT
