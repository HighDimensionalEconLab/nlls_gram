import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    SVD,
    DiagonalMetric,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMAction,
    LMStatus,
    Metric,
    MultiStart,
    register_pytree_dataclass,
)


@dataclass(frozen=True, eq=False)
class WeightedMetric(Metric):
    """A metric whose diagonal weight is a carried leaf: a callback rebuilds
    it from the live iterate, and the carried instance is frozen at the
    solution under implicit AD."""

    weight: jax.Array
    size: int = 1
    free_scale: float = 1.0

    def factor_apply(self, v, ctx):
        return jnp.sqrt(self.weight) * v

    def factor_solve(self, v, ctx):
        return v / jnp.sqrt(self.weight)

    def factor_solve_transpose(self, v, ctx):
        return self.factor_solve(v, ctx)


register_pytree_dataclass(
    WeightedMetric, data_fields=("weight", "free_scale"), meta_fields=("size",)
)


def test_failed_lane_uses_initial_point_under_vmap_jvp_and_vjp():
    def residual(x, _, p):
        return x - p

    solver = LevenbergMarquardt(
        residual,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(1)
    atols = jnp.asarray([1e-6, 0.0])

    def solved_x(parameters):
        def solve_one(parameter, atol):
            return solver.solve(
                x0,
                p=parameter,
                max_steps=1,
                max_steps_is_success=False,
                atol=atol,
            ).x[0]

        return jax.vmap(solve_one)(parameters, atols)

    parameters = jnp.asarray([0.0, 1.0])
    statuses = jax.vmap(
        lambda parameter, atol: (
            solver.solve(
                x0,
                p=parameter,
                max_steps=1,
                max_steps_is_success=False,
                atol=atol,
            ).status
        )
    )(parameters, atols)
    assert jnp.array_equal(
        statuses,
        jnp.asarray([LMStatus.CONVERGED, LMStatus.MAX_STEPS], dtype=jnp.int32),
    )

    _, tangent = jax.jvp(solved_x, (parameters,), (jnp.ones_like(parameters),))
    assert jnp.allclose(tangent, jnp.asarray([1.0, 0.0]), atol=1e-6)

    _, pullback = jax.vjp(solved_x, parameters)
    (cotangent,) = pullback(jnp.ones_like(parameters))
    assert jnp.all(jnp.isfinite(cotangent))
    assert jnp.allclose(cotangent, jnp.asarray([1.0, 0.0]), atol=1e-6)


def test_invalid_failed_result_uses_initial_instances_for_a_carried_metric():
    def residual(x, _, p):
        root = jnp.sqrt(x)
        return root - p, {"weight": 1.0 + root}

    x0 = jnp.ones(1)
    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        metric=WeightedMetric(1.0 + jnp.sqrt(jnp.abs(x0))),
        ad_solver=SVD(),
        cache_jacobian=False,
        geodesic_acceleration=False,
    )

    def invalidate(ctx):
        # Poison BOTH the iterate and the carried metric: the failed-lane
        # tangent program must read the pre-loop instances, or the NaN
        # weight reaches the whitening and the zero-tangent contract breaks.
        return LMAction(
            stop=True,
            status=LMStatus.NONFINITE,
            x=-jnp.ones(1),
            lm_state=dataclasses.replace(
                ctx.lm_state,
                metric=WeightedMetric(jnp.full((1,), jnp.nan, dtype=ctx.x.dtype)),
            ),
        )

    def failed_outputs(p):
        result = solver.solve(
            x0,
            p=p,
            max_steps=2,
            callback=invalidate,
        )
        return result.x[0], result.aux["weight"][0]

    p = jnp.asarray(1.0)
    result = solver.solve(x0, p=p, max_steps=2, callback=invalidate)
    assert result.status == LMStatus.NONFINITE
    assert not jnp.isfinite(result.aux["weight"][0])

    _, tangents = jax.jvp(failed_outputs, (p,), (jnp.asarray(1.0),))
    assert jnp.array_equal(jnp.asarray(tangents), jnp.zeros(2))

    _, pullback = jax.vjp(failed_outputs, p)
    (cotangent,) = pullback((jnp.asarray(1.0), jnp.asarray(1.0)))
    assert jnp.isfinite(cotangent)
    assert jnp.array_equal(cotangent, jnp.asarray(0.0))


def test_failed_gram_cg_rebuilds_aux_for_preconditioner_factory():
    def residual(x, _, p):
        value = jnp.asarray([jnp.sum(x) - p])
        return value, {"scale": 1.0 + 0.1 * jnp.sum(x**2)}

    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        linear_solver=GramCG(IdentityPreconditioner(), maxiter=8),
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(2)

    def failed_x(p):
        return solver.solve(x0, p=p, max_steps=1, max_steps_is_success=False).x

    p = jnp.asarray(1.0)
    result = solver.solve(x0, p=p, max_steps=1, max_steps_is_success=False)
    assert result.status == LMStatus.MAX_STEPS

    _, tangent = jax.jvp(failed_x, (p,), (jnp.asarray(1.0),))
    assert jnp.array_equal(tangent, jnp.zeros_like(x0))
    assert jnp.isfinite(tangent).all()

    _, pullback = jax.vjp(failed_x, p)
    (cotangent,) = pullback(jnp.ones_like(x0))
    assert jnp.isfinite(cotangent)
    assert jnp.array_equal(cotangent, jnp.asarray(0.0))


def test_failed_result_p_remains_a_pass_through():
    def residual(x, _, p):
        return x - p

    solver = LevenbergMarquardt(
        residual,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(1)

    def failed_outputs(p):
        result = solver.solve(x0, p=p, max_steps=1, max_steps_is_success=False)
        return result.x[0], result.p

    p = jnp.asarray(1.0)
    _, tangent = jax.jvp(failed_outputs, (p,), (jnp.asarray(2.0),))
    assert jnp.array_equal(jnp.asarray(tangent), jnp.asarray([0.0, 2.0]))

    _, pullback = jax.vjp(failed_outputs, p)
    (cotangent,) = pullback((jnp.asarray(3.0), jnp.asarray(4.0)))
    assert jnp.array_equal(cotangent, jnp.asarray(4.0))


@pytest.mark.parametrize("multi_start", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_failed_initial_x_and_args_are_differentiation_inert(multi_start, jit):
    def residual(x, args, p):
        return x + args - p

    solver = LevenbergMarquardt(
        residual,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    config = MultiStart(key=jax.random.key(0), num_starts=1) if multi_start else None

    def failed_x(x0, args, p):
        return solver.solve(
            x0,
            args,
            p=p,
            max_steps=1,
            max_steps_is_success=False,
            multi_start=config,
            jit=jit,
        ).x

    x0 = jnp.zeros(1)
    args = jnp.asarray(0.25)
    p = jnp.asarray(1.0)
    jacobians = jax.jacfwd(failed_x, argnums=(0, 1, 2))(x0, args, p)
    for jacobian in jacobians:
        assert jnp.array_equal(jacobian, jnp.zeros_like(jacobian))


def test_failed_pytree_tangent_and_cotangent_are_zero():
    def residual(x, _, p):
        return jnp.asarray([x["left"] + x["right"] - p])

    solver = LevenbergMarquardt(
        residual,
        ad_solver=SVD(),
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = {"left": jnp.asarray(0.0), "right": jnp.asarray(0.0)}

    def failed_x(p):
        return solver.solve(x0, p=p, max_steps=1, max_steps_is_success=False).x

    p = jnp.asarray(1.0)
    _, tangent = jax.jvp(failed_x, (p,), (jnp.asarray(1.0),))
    assert all(jnp.array_equal(leaf, 0.0) for leaf in jax.tree.leaves(tangent))

    _, pullback = jax.vjp(failed_x, p)
    (cotangent,) = pullback(jax.tree.map(jnp.ones_like, x0))
    assert jnp.isfinite(cotangent)
    assert jnp.array_equal(cotangent, jnp.asarray(0.0))


def test_failed_fixed_metric_tangent_and_cotangent_are_zero():
    def residual(x, _, p):
        return jnp.asarray([x[0] + x[1] - p])

    solver = LevenbergMarquardt(
        residual,
        metric=DiagonalMetric(jnp.asarray([2.0, 3.0])),
        ad_solver=SVD(),
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(2)

    def failed_x(p):
        return solver.solve(x0, p=p, max_steps=1, max_steps_is_success=False).x

    p = jnp.asarray(1.0)
    result = solver.solve(x0, p=p, max_steps=1, max_steps_is_success=False)
    assert result.status == LMStatus.MAX_STEPS

    _, tangent = jax.jvp(failed_x, (p,), (jnp.asarray(1.0),))
    _, pullback = jax.vjp(failed_x, p)
    (cotangent,) = pullback(jnp.ones_like(x0))
    assert jnp.array_equal(tangent, jnp.zeros_like(x0))
    assert jnp.array_equal(cotangent, jnp.asarray(0.0))


def test_successful_direct_aux_does_not_rebuild_aux_for_ad():
    calls = []

    def residual(x, _, p):
        jax.debug.callback(lambda _: calls.append(None), x, ordered=True)
        return x - p, {"value": x + p}

    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        metric=DiagonalMetric(jnp.ones(1)),
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(1)
    p = jnp.asarray([1.0])

    @jax.jit
    def primal(parameter):
        result = solver.solve(x0, p=parameter, max_steps=32, atol=1e-6)
        return result.x, result.aux

    @jax.jit
    def differentiated(parameter):
        return jax.jvp(primal, (parameter,), (jnp.ones_like(parameter),))

    jax.block_until_ready(primal(p))
    jax.block_until_ready(differentiated(p))

    calls.clear()
    jax.block_until_ready(primal(p))
    primal_calls = len(calls)
    calls.clear()
    jax.block_until_ready(differentiated(p))
    differentiated_calls = len(calls)

    assert differentiated_calls == primal_calls + 3


@pytest.mark.parametrize("max_steps_is_success", [True, False])
def test_max_steps_success_policy_controls_implicit_ad(max_steps_is_success):
    def residual(x, _, p):
        return x - p, {"value": x + p}

    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(1)

    def solved_outputs(p):
        result = solver.solve(
            x0,
            p=p,
            max_steps=1,
            max_steps_is_success=max_steps_is_success,
        )
        return result.x, result.aux["value"]

    p = jnp.ones(1)
    result = solver.solve(
        x0,
        p=p,
        max_steps=1,
        max_steps_is_success=max_steps_is_success,
    )
    assert result.status == LMStatus.MAX_STEPS

    _, tangent = jax.jvp(solved_outputs, (p,), (jnp.ones_like(p),))
    _, pullback = jax.vjp(solved_outputs, p)
    (cotangent,) = pullback((jnp.ones_like(p), jnp.ones_like(p)))
    expected_x = jnp.ones_like(p) if max_steps_is_success else jnp.zeros_like(p)
    expected_aux = 2.0 * expected_x
    expected_cotangent = 3.0 * expected_x
    assert jnp.allclose(tangent[0], expected_x, atol=1e-6)
    assert jnp.allclose(tangent[1], expected_aux, atol=1e-6)
    assert jnp.allclose(cotangent, expected_cotangent, atol=1e-6)


def test_custom_rejected_max_steps_remains_ad_successful_by_default():
    def residual(x, _, p):
        return x - p

    def reject(_, __):
        return False

    solver = LevenbergMarquardt(
        residual,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(1)
    multi_start = MultiStart(key=jax.random.key(1), num_starts=1, accept=reject)

    def solved_x(p):
        return solver.solve(x0, p=p, max_steps=1, multi_start=multi_start).x

    p = jnp.ones(1)
    result = solver.solve(x0, p=p, max_steps=1, multi_start=multi_start)
    assert result.status == LMStatus.MAX_STEPS
    assert not bool(result.multi_start.accepted)

    _, tangent = jax.jvp(solved_x, (p,), (jnp.ones_like(p),))
    _, pullback = jax.vjp(solved_x, p)
    (cotangent,) = pullback(jnp.ones_like(p))
    assert jnp.allclose(tangent, jnp.ones_like(p), atol=1e-6)
    assert jnp.allclose(cotangent, jnp.ones_like(p), atol=1e-6)
