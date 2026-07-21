import jax
import jax.numpy as jnp
import pytest

from nlls_gram import LevenbergMarquardt, Metric, identity_preconditioner

A_SQUARE = jnp.array([[2.0, 1.0], [-1.0, 3.0]])
P0 = jnp.array([1.0, -2.0])
P_DOT = jnp.array([0.3, -0.7])
X_BAR = jnp.array([0.4, -0.2])


def square_residual(x, _, p):
    return A_SQUARE @ x - p


def solved_x(solver, p):
    return solver.solve(jnp.zeros(2), p=p, max_steps=40, atol=1e-6).x


@pytest.mark.parametrize("ad_solver", ["auto", "direct"])
def test_square_direct_jvp_and_vjp_match_general_solve(ad_solver):
    solver = LevenbergMarquardt(square_residual, ad_solver=ad_solver)
    value, tangent = jax.jvp(
        lambda p: solved_x(solver, p),
        (P0,),
        (P_DOT,),
    )
    _, pullback = jax.vjp(lambda p: solved_x(solver, p), P0)

    assert solver._ad_solver_at(value, None, P0) == "direct"
    assert jnp.allclose(value, jnp.linalg.solve(A_SQUARE, P0), atol=1e-6)
    assert jnp.allclose(tangent, jnp.linalg.solve(A_SQUARE, P_DOT), atol=1e-6)
    assert jnp.allclose(
        pullback(X_BAR)[0],
        jnp.linalg.solve(A_SQUARE.T, X_BAR),
        atol=1e-6,
    )


def test_direct_supports_higher_order_ad():
    def residual(x, _, p):
        return x**2 - p

    solver = LevenbergMarquardt(residual, ad_solver="direct")

    def root(p):
        return solver.solve(jnp.ones(1), p=p, max_steps=40, atol=1e-6).x[0]

    p = jnp.asarray(4.0)
    assert jnp.allclose(jax.grad(root)(p), 0.25, atol=1e-6)
    assert jnp.allclose(jax.grad(jax.grad(root))(p), -0.03125, atol=1e-6)


def test_direct_rejects_non_square_system_at_trace_time():
    def residual(x, _, p):
        return jnp.array([x[0] - p, 2.0 * x[0] - p])

    solver = LevenbergMarquardt(residual, ad_solver="direct")
    with pytest.raises(ValueError, match="requires a square residual Jacobian"):
        jax.jvp(
            lambda p: solver.solve(jnp.zeros(1), p=p, max_steps=40).x,
            (jnp.asarray(1.0),),
            (jnp.asarray(1.0),),
        )


def test_direct_singular_square_system_fails_loudly():
    matrix = jnp.array([[1.0, 0.0], [0.0, 0.0]])

    def residual(x, _, p):
        return matrix @ x - p

    solver = LevenbergMarquardt(residual, ad_solver="direct")
    tangent = jax.jvp(
        lambda p: solver.solve(jnp.zeros(2), p=p, max_steps=1, atol=0.0).x,
        (jnp.zeros(2),),
        (jnp.ones(2),),
    )[1]
    assert not jnp.all(jnp.isfinite(tangent))


def test_auto_dispatches_by_shape_before_forward_solver():
    square_cg = LevenbergMarquardt(
        square_residual,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
    )
    assert square_cg._ad_solver_at(jnp.zeros(2), None, P0) == "direct"

    def tall_residual(x, _, p):
        return jnp.array([x[0] - p, 2.0 * x[0] - 2.0 * p])

    tall_dense = LevenbergMarquardt(tall_residual)
    tall_cg = LevenbergMarquardt(
        tall_residual,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        ad_solver_preconditioner=identity_preconditioner(),
    )
    assert tall_dense._ad_solver_at(jnp.zeros(1), None, jnp.asarray(1.0)) == "svd"
    assert tall_cg._ad_solver_at(jnp.zeros(1), None, jnp.asarray(1.0)) == "gram_cg"


def test_direct_does_not_require_metric_whitening_callbacks():
    metric = Metric(solve=lambda value: value)
    solver = LevenbergMarquardt(
        square_residual,
        metric=metric,
        linear_solver="gram_cholesky",
        geodesic_acceleration=False,
        ad_solver="direct",
    )
    tangent = jax.jvp(
        lambda p: solved_x(solver, p),
        (P0,),
        (P_DOT,),
    )[1]
    assert jnp.allclose(tangent, jnp.linalg.solve(A_SQUARE, P_DOT), atol=1e-6)


@pytest.mark.parametrize(
    ("ad_solver", "penalty"),
    [
        ("auto", 1e-6),
        ("direct", 1e-6),
        ("svd", 1e-6),
        ("qr", 0.0),
        ("gram_cg", 1e-6),
        ("normal_cg", 1e-6),
    ],
)
def test_penalty_does_not_select_an_ad_algorithm(ad_solver, penalty):
    with pytest.raises(ValueError, match="accepted only"):
        LevenbergMarquardt(
            square_residual,
            ad_solver=ad_solver,
            ad_solver_penalty=penalty,
        )


@pytest.mark.parametrize("ad_solver", ["augmented_qr", "regularized_normal_cg"])
@pytest.mark.parametrize("penalty", [None, 0.0, -1e-6])
def test_regularized_ad_algorithms_require_positive_penalty(ad_solver, penalty):
    with pytest.raises(ValueError, match="requires a positive"):
        LevenbergMarquardt(
            square_residual,
            ad_solver=ad_solver,
            ad_solver_penalty=penalty,
        )
