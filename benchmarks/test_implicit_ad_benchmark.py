"""Opt-in successful-solve implicit-AD benchmarks."""

import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    LMStatus,
    MetricFactory,
    metric_from_diagonal,
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


def _metric_factory_problem():
    design = jnp.reshape(jnp.linspace(-0.8, 1.0, 32), (4, 8))

    def residual(x, _, p):
        return design @ x - p, {"weight": 1.0 + 0.1 * x**2}

    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["weight"],
        build=metric_from_diagonal,
    )
    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        metric_factory=factory,
        ad_solver="svd",
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(8)
    p = jnp.linspace(-0.2, 0.3, 4)

    def solve(parameter):
        return solver.solve(x0, p=parameter, max_steps=48, atol=1e-6).x

    def status(parameter):
        return solver.solve(x0, p=parameter, max_steps=48, atol=1e-6).status

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
    if case == "metric_factory":
        return _metric_factory_problem()
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


@pytest.mark.parametrize("case", ["direct", "direct_aux", "metric_factory", "vmapped"])
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
