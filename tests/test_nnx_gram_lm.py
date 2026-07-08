import jax
import jax.numpy as jnp
from flax import nnx

from nlls_gram import UnderdeterminedLevenbergMarquardt


class ExpModel(nnx.Module):
    def __init__(self, *, a0=1.0, b0=0.0):
        self.a = nnx.Param(jnp.asarray(a0))
        self.b = nnx.Param(jnp.asarray(b0))

    def __call__(self, x):
        return self.a[...] * jnp.exp(self.b[...] * x)


def test_nnx_state_params_recover_known_parameters():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    model = ExpModel()
    graphdef, params = nnx.split(model, nnx.Param)

    def residual(params, args, p):
        xx, yy = args
        model = nnx.merge(graphdef, params)
        return model(xx) - yy

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init(params, (x, y))

    @jax.jit
    def train_step(params, lm_state, args):
        return solver.update(params, lm_state, args)

    info = None
    for _ in range(50):
        params, lm_state, info = train_step(params, lm_state, (x, y))

    trained = nnx.merge(graphdef, params)
    assert float(info.loss) < 1e-8
    assert jnp.allclose(trained.a[...], 2.0, atol=1e-4)
    assert jnp.allclose(trained.b[...], -1.0, atol=1e-4)


def test_nnx_wrt_filter_freezes_unselected_initialized_params():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    model = ExpModel(b0=-1.0)

    wrt = nnx.PathContains("a")
    graphdef, trainable, frozen = nnx.split(model, wrt, ...)
    assert len(jax.tree.leaves(trainable)) == 1
    assert len(jax.tree.leaves(frozen)) == 1

    def residual(trainable, args, p):
        xx, yy = args
        model = nnx.merge(graphdef, trainable, frozen)
        return model(xx) - yy

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init(trainable, (x, y))

    @jax.jit
    def train_step(trainable, lm_state, args):
        return solver.update(trainable, lm_state, args)

    info = None
    for _ in range(50):
        trainable, lm_state, info = train_step(trainable, lm_state, (x, y))

    trained = nnx.merge(graphdef, trainable, frozen)
    assert float(info.loss) < 1e-8
    assert jnp.allclose(trained.a[...], 2.0, atol=1e-4)
    assert jnp.allclose(trained.b[...], -1.0, atol=1e-7)
