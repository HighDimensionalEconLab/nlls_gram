# nlls_gram

Gram/dual-form Levenberg-Marquardt nonlinear least-squares solvers for JAX.

`GramLevenbergMarquardt` minimizes `||r(params)||^2` for a user-supplied
`residual_fn(params, batch)`, where `params` is any JAX pytree (a flat array, a
dict, `nnx.state(model, nnx.Param)`, ...). It follows an `init`/`update` protocol:
`update(params, state, batch)` returns the **new params pytree** (same structure),
the next state, and an `LMInfo`. For overparameterized systems (many more
parameters `p` than residual rows `n`) it factors the small `n x n` gram (dual)
system instead of the `p x p` normal equations.

The solver depends only on `jax` — it knows nothing about `flax`/`nnx`/`optax`. It
performs no float casts: dtypes flow from your `params`/residual and JAX decides
`float32` vs `float64` via `jax_enable_x64`.

## Install

```bash
pip install nlls-gram
```

## Minimal example

Fit `y = a * exp(b * x)` to noise-free data generated from `(a, b) = (2, -1)`,
using a plain dict pytree of parameters:

```python
import jax
import jax.numpy as jnp

from nlls_gram import GramLevenbergMarquardt

jax.config.update("jax_enable_x64", True)


# residual_fn(params, batch) -> 1-D residual array; the solver minimizes its SSQ.
def residual_fn(params, batch):
    x, y = batch
    return params["a"] * jnp.exp(params["b"] * x) - y


x = jnp.linspace(0.0, 2.0, 20)
y = 2.0 * jnp.exp(-1.0 * x)

params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init()


# The solver does not jit internally; wrap the train step yourself.
@jax.jit
def train_step(params, lm_state, batch):
    return solver.update(params, lm_state, batch)


for _ in range(50):
    params, lm_state, info = train_step(params, lm_state, (x, y))

print(params["a"], params["b"])  # ~2.0, ~-1.0
```

`params` can be any pytree. With Flax NNX, pass `nnx.state(model, nnx.Param)` as
`params` and write `residual_fn(state, batch)` using `nnx.merge`; the solver itself
stays NNX-agnostic.

## Filtering / freezing parameters

`update` optimizes exactly the `params` you pass, so freezing is just "pass fewer
params": keep the frozen values in `residual_fn`'s closure and hand the solver only
the trainable subset. Frozen leaves get no Jacobian column and never move — there
is no `wrt`/mask argument.

```python
# Optimize only "a"; "b" is frozen at its current value.
frozen = {"b": jnp.asarray(-1.0)}


def residual_fn(trainable, batch):
    x, y = batch
    params = {**frozen, **trainable}  # frozen from the closure, trainable optimized
    return params["a"] * jnp.exp(params["b"] * x) - y


trainable = {"a": jnp.asarray(1.0)}
solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init()
for _ in range(50):
    trainable, lm_state, info = solver.update(trainable, lm_state, (x, y))
# trainable["a"] -> ~2.0; "b" stayed -1.0
```

With Flax NNX, split the model into frozen and trainable states with a filter and
merge them back inside `residual_fn` (`freeze_filter` is any nnx `Filter` — a type,
path, or predicate — picking the params to hold fixed; `...` captures the rest):

```python
graphdef, frozen, trainable = nnx.split(model, freeze_filter, ...)


def residual_fn(trainable, batch):
    m = nnx.merge(graphdef, frozen, trainable)
    ...  # compute residuals from m


trainable, lm_state, info = solver.update(trainable, lm_state, batch)
new_model = nnx.merge(graphdef, frozen, trainable)
```

## API reference

::: nlls_gram.GramLevenbergMarquardt

::: nlls_gram.LMState

::: nlls_gram.LMInfo
