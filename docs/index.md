# nlls_gram

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

## Install

```bash
uv add nlls-gram
```

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
        solve_method="normal",
        regularization=regularization,
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

## API reference

::: nlls_gram.GramLevenbergMarquardt

::: nlls_gram.LMState

::: nlls_gram.LMInfo
