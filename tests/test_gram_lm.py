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


def residual_fn(model, batch):
    x, y = batch
    return model(x) - y


def test_recovers_known_parameters():
    a_true, b_true = 2.0, -1.0
    x = jnp.linspace(0.0, 2.0, 20)
    y = a_true * jnp.exp(b_true * x)

    model = ExpModel(a=1.0, b=0.0)
    solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
    optimizer = nnx.Optimizer(model, optax.identity(), wrt=nnx.Param)
    lm_state = solver.init()

    @jax.jit
    def train_step(graphdef, state, lm_state, batch):
        m, opt = nnx.merge(graphdef, state)
        updates, lm_state, info = solver.update(m, lm_state, batch)
        opt.update(m, updates)
        return lm_state, info, nnx.state((m, opt))

    graphdef, state = nnx.split((model, optimizer))
    info = None
    for _ in range(50):
        lm_state, info, state = train_step(graphdef, state, lm_state, (x, y))
    nnx.update((model, optimizer), state)

    assert float(info.loss) < 1e-10
    assert jnp.allclose(model.a[...], a_true, atol=1e-4)
    assert jnp.allclose(model.b[...], b_true, atol=1e-4)
