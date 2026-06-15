# nlls_gram

Gram/dual-form Levenberg-Marquardt nonlinear least-squares solvers for JAX/Flax
NNX models.

`GramLevenbergMarquardt` minimizes `||r(theta)||^2` for a residual defined over an
`nnx.Module`, following the optax/nnx `init`/`update` protocol so that steps apply
through `nnx.Optimizer(model, optax.identity(), wrt=...)`. For overparameterized
systems (many more parameters `p` than residual rows `n`) it factors the small
`n x n` gram (dual) system instead of the `p x p` normal equations.

## Install

```bash
pip install nlls-gram
```

## Minimal example

Fit `y = a * exp(b * x)` to noise-free data generated from `(a, b) = (2, -1)`:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nlls_gram import GramLevenbergMarquardt

jax.config.update("jax_enable_x64", True)


class ExpModel(nnx.Module):
    def __init__(self, a, b):
        self.a = nnx.Param(jnp.asarray(a))
        self.b = nnx.Param(jnp.asarray(b))

    def __call__(self, x):
        return self.a * jnp.exp(self.b * x)


x = jnp.linspace(0.0, 2.0, 20)
y = 2.0 * jnp.exp(-1.0 * x)

model = ExpModel(a=1.0, b=0.0)
solver = GramLevenbergMarquardt(lambda m, batch: m(batch[0]) - batch[1])
optimizer = nnx.Optimizer(model, optax.identity(), wrt=nnx.Param)
lm_state = solver.init()


@jax.jit
def train_step(graphdef, state, lm_state, batch):
    m, opt = nnx.merge(graphdef, state)
    updates, lm_state, info = solver.update(m, lm_state, batch)
    opt.update(m, updates)
    return lm_state, info, nnx.state((m, opt))


graphdef, state = nnx.split((model, optimizer))
for _ in range(50):
    lm_state, info, state = train_step(graphdef, state, lm_state, (x, y))
nnx.update((model, optimizer), state)

print(model.a[...], model.b[...])  # ~2.0, ~-1.0
```

## Documentation

Full docs: https://highdimensionaleconlab.github.io/nlls_gram/

## License

MIT
