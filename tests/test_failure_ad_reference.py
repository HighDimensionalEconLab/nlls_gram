import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    LMStatus,
    MetricFactory,
    MultiStart,
    metric_from_diagonal,
)


def sqrt_residual(x, domain, p):
    return x - jnp.sqrt(domain + p)


def safe_reference(x0, domain, p):
    return (
        x0,
        jnp.ones_like(domain),
        jnp.zeros_like(p),
    )


def test_failure_ad_reference_keeps_vmap_jvp_and_vjp_finite():
    solver = LevenbergMarquardt(
        sqrt_residual,
        geodesic_acceleration=False,
        cache_jacobian=False,
        linear_solver="normal_cholesky",
        ad_solver="svd",
    )
    x0 = jnp.zeros(1)
    domains = jnp.array([1.0, -2.0])

    def solved_x(parameters):
        def solve_one(domain, p):
            return solver.solve(
                x0,
                domain,
                p=p,
                max_steps=40,
                atol=1e-6,
                failure_ad_reference=safe_reference(x0, domain, p),
            ).x[0]

        return jax.vmap(solve_one)(domains, parameters)

    p = jnp.zeros(2)
    x, x_dot = jax.jvp(solved_x, (p,), (jnp.ones_like(p),))
    assert jnp.allclose(x[0], 1.0, atol=1e-5)
    assert jnp.allclose(x_dot, jnp.array([0.5, 0.0]), atol=1e-5)

    _, pullback = jax.vjp(solved_x, p)
    (p_bar,) = pullback(jnp.ones_like(x))
    assert jnp.all(jnp.isfinite(p_bar))
    assert jnp.allclose(p_bar, jnp.array([0.5, 0.0]), atol=1e-5)


@pytest.mark.parametrize("multi_start", [False, True])
def test_failure_ad_reference_does_not_change_converged_derivative(multi_start):
    solver = LevenbergMarquardt(
        sqrt_residual,
        geodesic_acceleration=False,
        cache_jacobian=False,
    )
    x0 = jnp.zeros(1)
    domain = jnp.asarray(4.0)
    config = MultiStart(key=jax.random.key(0), num_starts=1) if multi_start else None

    def solved_x(p, reference):
        return solver.solve(
            x0,
            domain,
            p=p,
            max_steps=40,
            atol=1e-6,
            multi_start=config,
            failure_ad_reference=(x0, jnp.asarray(1.0), reference),
        ).x[0]

    p = jnp.asarray(0.0)
    value, tangent = jax.jvp(
        lambda q: solved_x(q, jnp.asarray(0.0)),
        (p,),
        (jnp.asarray(1.0),),
    )
    assert jnp.allclose(value, 2.0, atol=1e-5)
    assert jnp.allclose(tangent, 0.25, atol=1e-5)

    # The reference is a coefficient of the custom rule, not another AD input.
    reference_grad = jax.grad(lambda ref: solved_x(p, ref))(jnp.asarray(0.0))
    assert jnp.allclose(reference_grad, 0.0)


def test_failure_ad_reference_rebuilds_aux_for_metric_factory():
    def residual(x, domain, p):
        root = jnp.sqrt(domain + p)
        return x - root, {"weight": jnp.ones_like(x) * (1.0 + root)}

    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["weight"],
        build=metric_from_diagonal,
    )
    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        metric_factory=factory,
        geodesic_acceleration=False,
        cache_jacobian=False,
        ad_solver="svd",
    )
    x0 = jnp.zeros(1)
    domain = jnp.asarray(-2.0)
    p = jnp.asarray(0.0)
    reference = safe_reference(x0, domain, p)

    result = solver.solve(
        x0,
        domain,
        p=p,
        max_steps=10,
        atol=1e-6,
        failure_ad_reference=reference,
    )
    assert result.status == LMStatus.NONFINITE

    def failed_outputs(parameter):
        failed = solver.solve(
            x0,
            domain,
            p=parameter,
            max_steps=10,
            atol=1e-6,
            failure_ad_reference=safe_reference(x0, domain, parameter),
        )
        return failed.x[0], failed.aux["weight"][0]

    _, tangents = jax.jvp(failed_outputs, (p,), (jnp.asarray(1.0),))
    assert jnp.allclose(jnp.asarray(tangents), 0.0)

    _, pullback = jax.vjp(failed_outputs, p)
    (p_bar,) = pullback((jnp.asarray(1.0), jnp.asarray(1.0)))
    assert jnp.isfinite(p_bar)
    assert jnp.allclose(p_bar, 0.0)


def test_failure_ad_reference_validates_structure_shape_and_dtype():
    solver = LevenbergMarquardt(sqrt_residual)
    x0 = jnp.zeros(1)
    domain = jnp.asarray(1.0)
    p = jnp.asarray(0.0)

    with pytest.raises(ValueError, match=r"tuple \(x_ref, args_ref, p_ref\)"):
        solver.solve(
            x0,
            domain,
            p=p,
            failure_ad_reference=(x0, domain),
        )

    with pytest.raises(ValueError, match="structure, shapes, and dtypes"):
        solver.solve(
            x0,
            domain,
            p=p,
            failure_ad_reference=(jnp.zeros(2), domain, p),
        )

    with pytest.raises(ValueError, match="structure, shapes, and dtypes"):
        solver.solve(
            x0,
            domain,
            p=p,
            failure_ad_reference=(x0, domain, jnp.asarray(0, dtype=jnp.int32)),
        )


def test_failure_ad_reference_allows_weak_type_mismatch():
    solver = LevenbergMarquardt(sqrt_residual)
    x0 = jnp.zeros(1)
    domain = jnp.zeros((), dtype=jnp.float32) + 1.0
    p = jnp.zeros((), dtype=jnp.float32)
    result = solver.solve(
        x0,
        domain,
        p=p,
        max_steps=40,
        atol=1e-6,
        failure_ad_reference=(x0, 1.0, 0.0),
    )
    assert result.status == LMStatus.CONVERGED
