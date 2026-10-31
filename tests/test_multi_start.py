"""MultiStart and Flax NNX integration tests across sequential and parallel
drivers, hooks, warm-state handling, and implicit AD.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nlls_gram import (
    DrawNNXModule,
    LevenbergMarquardt,
    MultiStart,
    Newton,
    SolverAction,
    SolverStatus,
)

ALPHA = 0.1


def residual_linear(theta, args, p):
    return jnp.array([theta[0] + 2.0 * theta[1] - p])


def residual_nonlinear(theta, args, p):
    return jnp.array([theta[0] + 2.0 * theta[1] - p**2])


def residual_two_basin(theta, args, p):
    return jnp.array([theta[0] ** 2 - 1.0, ALPHA * (theta[0] - 1.0)])


def draw_zeros(key, x, args):
    return jnp.zeros_like(x)


def draw_uniform(key, x, args):
    return jax.random.uniform(key, x.shape, x.dtype, -3.0, 3.0)


def accept_deep_basin(key, result):
    return result.info.loss < 1e-3


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize("parallel", [False, True])
def test_multi_start_recovery_accept_hook_and_ad(solver_cls, parallel):
    # 1. Recovers from NaN start and differentiates through the winning start
    solver = solver_cls(residual_nonlinear)
    x0_nan = jnp.array([jnp.nan, jnp.nan])
    ms = MultiStart(
        key=jax.random.key(14), num_starts=3, draw=draw_zeros, parallel=parallel
    )

    def sum_x(p):
        res = solver.solve(x0_nan, p=p, max_steps=100, atol=1e-6, multi_start=ms)
        return jnp.sum(res.x)

    p0 = jnp.asarray(1.5)
    res = solver.solve(x0_nan, p=p0, max_steps=100, atol=1e-6, multi_start=ms)
    assert int(res.status) == int(SolverStatus.CONVERGED)
    assert int(res.multi_start.attempt) >= 1
    assert bool(res.multi_start.accepted)
    np.testing.assert_allclose(jax.grad(sum_x)(p0), 6.0 * p0 / 5.0, atol=1e-4)
    np.testing.assert_allclose(jax.hessian(sum_x)(p0), 6.0 / 5.0, atol=1e-3)

    # 2. Custom accept hook rejects shallow local minimum and finds global basin
    basin_solver = solver_cls(residual_two_basin)
    x0_shallow = jnp.array([-2.0])
    key = jax.random.key(5)
    deep_ms = MultiStart(
        key=key,
        num_starts=8,
        draw=draw_uniform,
        accept=accept_deep_basin,
        parallel=parallel,
    )
    deep = basin_solver.solve(x0_shallow, max_steps=100, gtol=1e-4, multi_start=deep_ms)
    assert bool(deep.multi_start.accepted)
    assert float(deep.multi_start.loss) < 1e-3


def test_warm_state_attempt_zero_reset_args_and_validation():
    # Attempt 0 honors a caller-supplied warm state; retries invalidate caches
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2, cache_jacobian=True)
    p = jnp.asarray(3.0)
    warm = solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).state
    ms = MultiStart(key=jax.random.key(24), num_starts=2, draw=draw_zeros)
    result = solver.solve(
        jnp.array([jnp.nan, jnp.nan]),
        p=p,
        state=warm,
        max_steps=80,
        atol=1e-6,
        save_steps=True,
        multi_start=ms,
    )
    assert int(result.multi_start.attempt) == 1
    assert result.x_history.shape == (81, 2)

    # reset_args resamples args on retry and composes with callback args mutation
    def reset_resample(key, args):
        data_key, carry_key = jax.random.split(key)
        return {
            "data": jax.random.normal(data_key, args["data"].shape, args["data"].dtype),
            "key": carry_key,
        }

    def cb_resample(ctx):
        def fresh(_):
            ck, dk = jax.random.split(ctx.args["key"])
            return {
                "data": jax.random.normal(
                    dk, ctx.args["data"].shape, ctx.args["data"].dtype
                ),
                "key": ck,
            }

        return SolverAction(
            args=jax.lax.cond(ctx.step == 1, fresh, lambda _: ctx.args, None)
        )

    res_solver = LevenbergMarquardt(lambda th, a, pv: th - a["data"] * pv)
    args0 = {"data": jnp.array([1.0, -2.0]), "key": jax.random.key(22)}
    ms_reset = MultiStart(
        key=jax.random.key(23),
        num_starts=3,
        draw=draw_zeros,
        reset_args=reset_resample,
    )
    res_r = res_solver.solve(
        jnp.array([jnp.nan, jnp.nan]),
        args0,
        p=jnp.asarray(2.0),
        max_steps=40,
        atol=1e-6,
        callback=cb_resample,
        multi_start=ms_reset,
    )
    assert int(res_r.status) == int(SolverStatus.CONVERGED)
    assert not jnp.allclose(res_r.args["data"], args0["data"])

    # Validation errors
    key = jax.random.key(19)
    with pytest.raises(ValueError, match="num_starts"):
        MultiStart(key=key, num_starts=0)
    with pytest.raises(ValueError, match="requires draw"):
        MultiStart(key=key, num_starts=2)
    with pytest.raises(TypeError, match="draw must be callable"):
        MultiStart(key=key, num_starts=2, draw=5)


class CurveMLP(nnx.Module):
    def __init__(self, *, scale=2.0, rngs: nnx.Rngs):
        self.scale = nnx.Variable(jnp.asarray(scale))
        self.hidden = nnx.Linear(1, 8, rngs=rngs)
        self.head = nnx.Linear(8, 1, rngs=rngs)

    def __call__(self, x):
        return self.scale[...] * self.head(nnx.tanh(self.hidden(x[:, None])))[:, 0]


@pytest.mark.parametrize("parallel", [False, True])
def test_nnx_module_and_draw_nnx_module(parallel):
    # Value semantics and wrt filter
    draw = DrawNNXModule(CurveMLP)
    assert draw == DrawNNXModule(CurveMLP)
    assert hash(draw) == hash(DrawNNXModule(CurveMLP))
    assert DrawNNXModule(CurveMLP, a=1, b=2) == DrawNNXModule(CurveMLP, b=2, a=1)
    assert DrawNNXModule(CurveMLP, 1) != DrawNNXModule(CurveMLP, 1.0)

    wrt = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("head")))
    draw_wrt = DrawNNXModule(CurveMLP, wrt=wrt)
    assert draw_wrt == DrawNNXModule(CurveMLP, wrt=wrt)
    assert draw_wrt != draw

    # Multi-start recovery on NNX module with non-Param Variable state
    graphdef, theta_good, nondiff = nnx.split(
        CurveMLP(rngs=nnx.Rngs(1)), nnx.Param, ...
    )
    ts = jnp.linspace(-1.0, 1.0, 24)
    ys = jnp.sin(2.0 * ts)
    theta_bad = jax.tree.map(lambda leaf: leaf * jnp.nan, theta_good)

    def residual(theta, args, p):
        t_in, y_in = args
        return nnx.merge(graphdef, theta, nondiff)(t_in) - y_in

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    ms = MultiStart(
        key=jax.random.key(2),
        num_starts=3,
        draw=draw,
        parallel=parallel,
    )
    result = solver.solve(theta_bad, (ts, ys), max_steps=150, atol=5e-3, multi_start=ms)
    assert int(result.status) == int(SolverStatus.CONVERGED)
    assert int(result.multi_start.attempt) >= 1
    assert bool(result.multi_start.accepted)
    trained = nnx.merge(graphdef, result.x, nondiff)
    assert float(jnp.max(jnp.abs(trained(ts) - ys))) < 0.06
