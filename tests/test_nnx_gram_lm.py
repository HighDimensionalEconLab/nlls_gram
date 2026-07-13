import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from nlls_gram import LMStatus, MultiStart, UnderdeterminedLevenbergMarquardt


class ExpModel(nnx.Module):
    def __init__(self, *, a0=1.0, b0=0.0):
        self.a = nnx.Param(jnp.asarray(a0))
        self.b = nnx.Param(jnp.asarray(b0))

    def __call__(self, x):
        return self.a[...] * jnp.exp(self.b[...] * x)


def test_nnx_state_params_recover_known_parameters():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    model = ExpModel()
    graphdef, x = nnx.split(model, nnx.Param)

    def residual(x, args, p):
        ts, ys = args
        model = nnx.merge(graphdef, x)
        return model(ts) - ys

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init(x, (ts, ys))

    @jax.jit
    def train_step(x, lm_state, args):
        return solver.update(x, lm_state, args)

    info = None
    for _ in range(50):
        x, lm_state, info = train_step(x, lm_state, (ts, ys))

    trained = nnx.merge(graphdef, x)
    assert float(info.loss) < 1e-8
    assert jnp.allclose(trained.a[...], 2.0, atol=1e-4)
    assert jnp.allclose(trained.b[...], -1.0, atol=1e-4)


def test_nnx_wrt_filter_freezes_unselected_initialized_params():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    model = ExpModel(b0=-1.0)

    wrt = nnx.PathContains("a")
    graphdef, trainable, frozen = nnx.split(model, wrt, ...)
    assert len(jax.tree.leaves(trainable)) == 1
    assert len(jax.tree.leaves(frozen)) == 1

    def residual(trainable, args, p):
        ts, ys = args
        model = nnx.merge(graphdef, trainable, frozen)
        return model(ts) - ys

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init(trainable, (ts, ys))

    @jax.jit
    def train_step(trainable, lm_state, args):
        return solver.update(trainable, lm_state, args)

    info = None
    for _ in range(50):
        trainable, lm_state, info = train_step(trainable, lm_state, (ts, ys))

    trained = nnx.merge(graphdef, trainable, frozen)
    assert float(info.loss) < 1e-8
    assert jnp.allclose(trained.a[...], 2.0, atol=1e-4)
    assert jnp.allclose(trained.b[...], -1.0, atol=1e-7)


class CurveMLP(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.hidden = nnx.Linear(1, 8, rngs=rngs)
        self.head = nnx.Linear(8, 1, rngs=rngs)

    def __call__(self, x):
        return self.head(nnx.tanh(self.hidden(x[:, None])))[:, 0]


CURVE_GRAPHDEF, _ = nnx.split(CurveMLP(rngs=nnx.Rngs(0)), nnx.Param)


def curve_residual(theta, args, p):
    ts, ys = args
    model = nnx.merge(CURVE_GRAPHDEF, theta)
    return model(ts) - ys


def draw_curve_params(key, x, args):
    from flax.nnx import Rngs

    _, theta = nnx.split(CurveMLP(rngs=Rngs(key)), nnx.Param)
    return theta, args


@pytest.mark.parametrize("parallel", [False, True])
def test_multi_start_nnx_redraw_recovers_from_bad_init(parallel):
    ts = jnp.linspace(-1.0, 1.0, 32)
    ys = jnp.sin(2.0 * ts)
    _, theta_good = nnx.split(CurveMLP(rngs=nnx.Rngs(1)), nnx.Param)
    theta_bad = jax.tree.map(lambda leaf: leaf * jnp.nan, theta_good)

    solver = UnderdeterminedLevenbergMarquardt(curve_residual, init_damping=1e-2)
    ms = MultiStart(
        key=jax.random.key(2), num_starts=4, draw=draw_curve_params, parallel=parallel
    )
    result = solver.solve(theta_bad, (ts, ys), max_steps=200, atol=1e-3, multi_start=ms)

    assert int(result.status) == LMStatus.CONVERGED
    assert int(result.multi_start.attempt) >= 1
    assert bool(result.multi_start.accepted)
    # The winner keeps the exact parameter pytree structure and dtypes.
    assert jax.tree_util.tree_structure(result.x) == jax.tree_util.tree_structure(
        theta_good
    )
    for got, want in zip(
        jax.tree.leaves(result.x), jax.tree.leaves(theta_good), strict=True
    ):
        assert got.shape == want.shape
        assert got.dtype == want.dtype
    trained = nnx.merge(CURVE_GRAPHDEF, result.x)
    assert float(jnp.max(jnp.abs(trained(ts) - ys))) < 0.05
