"""Opt-in successful-solve implicit-AD benchmarks."""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    SVD,
    LevenbergMarquardt,
    LMAction,
    LMStatus,
    Metric,
    register_pytree_dataclass,
)


def _direct_problem(*, has_aux):
    def residual(x, _, p):
        value = x - p
        if has_aux:
            return value, {"value": x**2 + p}
        return value

    solver = LevenbergMarquardt(
        residual,
        has_aux=has_aux,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(8)
    p = jnp.linspace(0.1, 0.8, 8)

    def solve(parameter):
        result = solver.solve(x0, p=parameter, max_steps=32, atol=1e-6)
        if has_aux:
            return result.x, result.aux["value"]
        return result.x

    def status(parameter):
        return solver.solve(x0, p=parameter, max_steps=32, atol=1e-6).status

    return solve, p, status


@dataclass(frozen=True, eq=False)
class TrackedDiagonalMetric(Metric):
    """An iterate-tracking metric: the weights are a carried leaf, rebuilt
    from the live iterate by the solve callback and frozen at the solution."""

    weights: jax.Array
    size: int = 8
    free_scale: float = 1.0

    def _scaled(self, v, factor):
        return v * factor.reshape(factor.shape + (1,) * (v.ndim - 1))

    def factor_apply(self, v, ctx):
        return self._scaled(v, jnp.sqrt(self.weights))

    def factor_solve(self, v, ctx):
        return self._scaled(v, 1.0 / jnp.sqrt(self.weights))

    def factor_solve_transpose(self, v, ctx):
        return self.factor_solve(v, ctx)


register_pytree_dataclass(
    TrackedDiagonalMetric, data_fields=("weights", "free_scale"), meta_fields=("size",)
)


def _tracked_metric_problem():
    design = jnp.reshape(jnp.linspace(-0.8, 1.0, 32), (4, 8))

    def residual(x, _, p):
        return design @ x - p, {"weight": 1.0 + 0.1 * x**2}

    x0 = jnp.zeros(8)
    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        metric=TrackedDiagonalMetric(1.0 + 0.1 * x0**2),
        ad_solver=SVD(),
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    p = jnp.linspace(-0.2, 0.3, 4)

    def track_iterate(ctx):
        return LMAction(
            lm_state=dataclasses.replace(
                ctx.lm_state, metric=TrackedDiagonalMetric(1.0 + 0.1 * ctx.x**2)
            )
        )

    def solve(parameter):
        return solver.solve(
            x0, p=parameter, max_steps=48, atol=1e-6, callback=track_iterate
        ).x

    def status(parameter):
        return solver.solve(
            x0, p=parameter, max_steps=48, atol=1e-6, callback=track_iterate
        ).status

    return solve, p, status


def _vmapped_problem():
    def residual(x, _, p):
        return x - p

    solver = LevenbergMarquardt(
        residual,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(4)
    p = jnp.reshape(jnp.linspace(0.1, 1.6, 64), (16, 4))

    def solve(parameters):
        return jax.vmap(
            lambda parameter: solver.solve(x0, p=parameter, max_steps=32, atol=1e-6).x
        )(parameters)

    def status(parameters):
        return jax.vmap(
            lambda parameter: (
                solver.solve(x0, p=parameter, max_steps=32, atol=1e-6).status
            )
        )(parameters)

    return solve, p, status


def _make_problem(case):
    if case == "direct":
        return _direct_problem(has_aux=False)
    if case == "direct_aux":
        return _direct_problem(has_aux=True)
    if case == "tracked_metric":
        return _tracked_metric_problem()
    return _vmapped_problem()


def _make_transformed(case, transform):
    solve, p, _ = _make_problem(case)
    tangent = jax.tree.map(jnp.ones_like, p)

    if transform == "jvp":

        @jax.jit
        def transformed(parameter):
            return jax.jvp(solve, (parameter,), (tangent,))

    else:

        @jax.jit
        def transformed(parameter):
            value, pullback = jax.vjp(solve, parameter)
            cotangent = jax.tree.map(jnp.ones_like, value)
            return value, pullback(cotangent)

    return transformed, p


@pytest.mark.parametrize("case", ["direct", "direct_aux", "tracked_metric", "vmapped"])
@pytest.mark.parametrize("transform", ["jvp", "vjp"])
def test_successful_implicit_ad(benchmark, case, transform):
    _, status_parameter, status = _make_problem(case)
    assert bool(jnp.all(status(status_parameter) == LMStatus.CONVERGED))

    transformed, p = _make_transformed(case, transform)
    warmup = transformed(p)
    jax.block_until_ready(warmup)

    def run():
        value = transformed(p)
        jax.block_until_ready(value)
        return value

    # These kernels are only a few microseconds. Averaging repeated blocked
    # dispatches within each round keeps scheduler noise below the 1 us gate.
    benchmark.pedantic(run, rounds=50, iterations=100)
