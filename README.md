# nlls_gram

[![CI](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/nlls_gram/)
[![PyPI](https://img.shields.io/pypi/v/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![Python versions](https://img.shields.io/pypi/pyversions/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/nlls_gram)](https://github.com/HighDimensionalEconLab/nlls_gram/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Gram/dual-form Levenberg-Marquardt nonlinear least-squares solvers for JAX.

`GramLevenbergMarquardt` minimizes `||r(params)||^2` for a user-supplied
`residual_fn(params, batch)`, where `params` is any JAX pytree (a flat array, a
dict, `nnx.state(model, nnx.Param)`, ...). It follows an `init`/`update` protocol:
`update(params, state, batch)` returns the **new params pytree** (same structure),
the next state, and an `LMInfo`. For overparameterized systems (many more
parameters `p` than residual rows `n`) it factors the small `n x n` gram (dual)
system instead of the `p x p` normal equations.

The solver depends only on `jax` — it knows nothing about `flax`/`nnx`/`optax`.
Dtypes flow from your `params`/residual, and the damping state follows the
residual dtype; JAX decides `float32` vs `float64` via `jax_enable_x64`.
`init_damping` must be positive; use a small positive value for near
Gauss-Newton behavior. There is intentionally no normal-equation mode; use a
different LM implementation when the Gram system is not the right shape for your
problem.

## Install

```bash
uv add nlls-gram
```

For local development on an NVIDIA CUDA 13 machine, use the optional `gpu`
dependency group:

```bash
uv sync --group gpu
```

That group is for this repository's development and GPU tests; it is not a
published `nlls-gram[gpu]` extra. Users who want to run the optimizer on a GPU
should install the JAX accelerator build that matches their hardware alongside
`nlls-gram`, for example:

```bash
uv add nlls-gram "jax[cuda13]"
```

See the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
the current CUDA, ROCm, TPU, and CPU installation choices.

## Minimal example

Fit `y = a * exp(b * x)` to noise-free data generated from `(a, b) = (2, -1)`,
using a plain dict pytree of parameters. With JAX's default configuration, this
runs in float32:

```python
import jax
import jax.numpy as jnp

from nlls_gram import GramLevenbergMarquardt


# residual_fn(params, batch) -> 1-D residual array; the solver minimizes its SSQ.
def residual_fn(params, batch):
    x, y = batch
    return params["a"] * jnp.exp(params["b"] * x) - y


x = jnp.linspace(0.0, 2.0, 20)
y = 2.0 * jnp.exp(-1.0 * x)

params = {"a": 1.0, "b": 0.0}
solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init()


# The solver does not jit internally; wrap the train step yourself.
@jax.jit
def train_step(params, lm_state, batch):
    return solver.update(params, lm_state, batch)


for _ in range(50):
    params, lm_state, info = train_step(params, lm_state, (x, y))

print(params["a"], params["b"])  # ~2.0, ~-1.0
print(params["a"].dtype, info.loss.dtype)  # float32 float32
```

## Float64 example

Enable x64 before creating arrays, then initialize the data and parameters as
float64:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from nlls_gram import GramLevenbergMarquardt

dtype = jnp.float64


def residual_fn(params, batch):
    x, y = batch
    return params["a"] * jnp.exp(params["b"] * x) - y


x = jnp.linspace(0.0, 2.0, 20, dtype=dtype)
y = 2.0 * jnp.exp(-1.0 * x)

params = {
    "a": jnp.asarray(1.0, dtype=dtype),
    "b": jnp.asarray(0.0, dtype=dtype),
}

solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init()

for _ in range(50):
    params, lm_state, info = solver.update(params, lm_state, (x, y))

print(params["a"], params["b"])  # ~2.0, ~-1.0
print(params["a"].dtype, info.loss.dtype, info.damping.dtype)
```

## Fletcher regularization

The default `regularization="identity"` uses the classic LM damping matrix
`lambda * I`. If parameters are badly scaled, `regularization="fletcher"` can
help by damping each parameter direction in proportion to `diag(J.T @ J)`.
The diagonal is clipped before use, with defaults
`fletcher_min_diagonal=1e-6` and `fletcher_max_diagonal=1e6`, so nearly unused
or extremely sensitive parameter directions do not dominate the Gram solve.

```python
import jax.numpy as jnp

from nlls_gram import GramLevenbergMarquardt

x = jnp.linspace(0.0, 2.0, 50)
y = 2.0 * jnp.exp(-1.0 * x)
parameter_scale = 1e-3


def residual_fn(params, batch):
    x, y = batch
    b = parameter_scale * params["b_scaled"]
    return params["a"] * jnp.exp(b * x) - y


def iterations_to_threshold(regularization):
    params = {"a": 1.0, "b_scaled": 0.0}
    solver = GramLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        regularization=regularization,
        fletcher_min_diagonal=1e-6,
        fletcher_max_diagonal=1e6,
    )
    lm_state = solver.init()
    for iteration in range(1, 51):
        params, lm_state, info = solver.update(params, lm_state, (x, y))
        if float(info.loss) < 1e-8:
            return iteration
    return None


print(iterations_to_threshold("identity"))  # ~16
print(iterations_to_threshold("fletcher"))  # ~4
```

## Iterative solves

The default `linear_solver="cholesky"` materializes the dense Gram matrix and
uses a Cholesky factorization. For larger identity-regularized problems, use an
iterative solver with JAX JVP/VJP linearization instead of materializing `J`:

Iterative solvers default to a small fixed iteration budget:
`iterative_tol=0.0`, `iterative_atol=0.0`, and `iterative_maxiter=8`. This avoids
extra tolerance-driven convergence work and is intended for low-rank local
linear solves. Set a positive `iterative_tol` or `iterative_atol` if you want
early convergence checks instead.

For fixed-budget CG in residual space, use:

```python
solver = GramLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    linear_solver="cg",
    formulation="gram",
    iterative_tol=0.0,
    iterative_atol=0.0,
    iterative_maxiter=8,
)
```

For fixed-budget CG in parameter space, use:

```python
solver = GramLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    linear_solver="cg",
    formulation="normal",
    iterative_tol=0.0,
    iterative_atol=0.0,
    iterative_maxiter=8,
)
```

CG currently supports only `regularization="identity"`. `formulation="gram"`
solves in residual space, so the Krylov vectors have length equal to the number
of residuals. `formulation="normal"` solves in parameter space, so the Krylov
vectors have length equal to the number of parameters. Both use matrix-free JVPs
for `J @ v` and VJPs/linear transposes for `J.T @ u`; choose the formulation
based on which vector space is smaller and better conditioned.

`linear_solver="lsmr"` uses Lineax LSMR on the damped least-squares problem
directly:

```text
min_s ||J s + r||^2 + lambda ||s||^2
```

For fixed-budget LSMR, use:

```python
solver = GramLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    linear_solver="lsmr",
    iterative_tol=0.0,
    iterative_atol=0.0,
    iterative_maxiter=8,
    lsmr_conlim=float("inf"),
)
```

It uses the augmented operator `[J; sqrt(lambda) I]`, so matrix-vector products
call JAX JVPs for `J @ s` and transposed products call VJPs/linear transposes for
`J.T @ u`. LSMR does not use the Gram or normal formulation. Its default
`lsmr_conlim=float("inf")` prevents condition-limit early termination; Lineax
still computes LSMR's internal norm estimates each iteration. Iterative solvers
can reduce memory and factorization cost on larger dense GPU problems, but each
iteration performs matrix-free Jacobian-vector and transpose-vector products, so
the Cholesky default remains better for small residual dimensions.

## Geodesic acceleration

Geodesic acceleration is off by default. When enabled, the solver uses analytic
JAX forward-mode JVPs to build an accelerated candidate; it does not use finite
differences.

```python
solver = GramLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    geodesic_acceleration=True,
)
```

The accelerated candidate is used only when its acceleration ratio,
`2 * ||a|| / ||v||`, is at or below a positive `geodesic_acceptance_ratio` and
its loss is no worse than the plain LM velocity candidate. Otherwise the update
automatically falls back to the velocity step. Use `LMInfo.used_geodesic`,
`LMInfo.acceleration_ratio`, `LMInfo.loss_old`, `LMInfo.loss_candidate`, and
`LMInfo.damping_factor` to tune damping and geodesic behavior.

`params` can be any pytree. With Flax NNX, pass `nnx.state(model, nnx.Param)` as
`params` and write `residual_fn(state, batch)` using `nnx.merge`; the solver itself
stays NNX-agnostic.

## Filtering / freezing parameters

`update` optimizes exactly the `params` pytree you pass. For Flax NNX transfer
learning, construct or load the full module first, choose the trainable leaves
with an NNX filter, and pass only that trainable state to the solver. This mirrors
the `wrt` argument used by `nnx.Optimizer`: `wrt` means "differentiate and update
these leaves", while `...` captures the already-initialized frozen remainder.
Install Flax in your project to run this example.

```python
import jax
import jax.numpy as jnp
from flax import nnx

from nlls_gram import GramLevenbergMarquardt


class ExpModel(nnx.Module):
    def __init__(self):
        self.a = nnx.Param(jnp.asarray(1.0))
        self.b = nnx.Param(jnp.asarray(-1.0))

    def __call__(self, x):
        return self.a[...] * jnp.exp(self.b[...] * x)


x = jnp.linspace(0.0, 2.0, 20)
y = 2.0 * jnp.exp(-1.0 * x)

model = ExpModel()
wrt = nnx.PathContains("a")  # train "a"; keep all other initialized state fixed
graphdef, trainable, frozen = nnx.split(model, wrt, ...)


def residual_fn(trainable, batch):
    x, y = batch
    model = nnx.merge(graphdef, trainable, frozen)
    return model(x) - y


solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init()
for _ in range(50):
    trainable, lm_state, info = solver.update(trainable, lm_state, (x, y))

model = nnx.merge(graphdef, trainable, frozen)
print(model.a[...], model.b[...])  # ~2.0, -1.0
```

For built-in NNX layers, set both computation and parameter initialization dtypes
when you want an all-float64 model:

```python
layer = nnx.Linear(
    1,
    1,
    dtype=jnp.float64,
    param_dtype=jnp.float64,
    rngs=nnx.Rngs(0),
)
```

## Benchmarks

Optional pytest-benchmark checks live outside the normal test suite and do not run
in CI by default:

```bash
uv run --group benchmark pytest benchmarks --benchmark-only
```

For a larger RBF-style interpolation profile with CPU/GPU, Cholesky/CG/LSMR,
Gram/normal CG, and geodesic on/off variants:

```bash
uv run --group benchmark --group gpu pytest \
  benchmarks/test_large_interpolation_benchmark.py --benchmark-only
```

For a small classic geodesic-acceleration convergence benchmark based on the GSL
modified Rosenbrock example:

```bash
uv run --group benchmark pytest \
  benchmarks/test_classic_geodesic_benchmark.py --benchmark-only
```

On machines with a CUDA-enabled JAX install, the optional GPU test checks that a
jitted geodesic update runs on a GPU device:

```bash
uv run --group gpu pytest tests/test_gpu.py
```

## Documentation

Full docs: https://highdimensionaleconlab.github.io/nlls_gram/

## License

MIT
