import jax
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt, LMStatus


def augmented_qr_solver(residual, *, has_aux=False):
    return LevenbergMarquardt(
        residual,
        linear_solver="augmented_qr",
        geodesic_acceleration=False,
        cache_jacobian=False,
        has_aux=has_aux,
    )


def test_nonlinear_algebraic_root_matches_closed_form():
    solver = augmented_qr_solver(lambda x, args, p: x**2 - p)
    result = solver.solve(jnp.array([1.0]), p=jnp.asarray(4.0), max_steps=50, atol=1e-6)

    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x, jnp.array([2.0]), atol=1e-5)
    assert float(jnp.sqrt(result.info.loss)) < 1e-6


def test_dae_style_pytree_root():
    def residual(z, args, p):
        y, t, theta = p
        return jnp.array([z["a"] + z["b"] - y * t, theta * z["a"] - z["b"]])

    solver = augmented_qr_solver(residual)
    y, t, theta = jnp.asarray(2.0), jnp.asarray(0.5), jnp.asarray(3.0)
    result = solver.solve(
        {"a": jnp.zeros(()), "b": jnp.zeros(())},
        p=(y, t, theta),
        max_steps=50,
        atol=1e-6,
    )
    a_expected = y * t / (1.0 + theta)

    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x["a"], a_expected, atol=1e-6)
    assert jnp.allclose(result.x["b"], theta * a_expected, atol=1e-6)


def test_warm_start_at_root_exits_in_zero_steps():
    solver = augmented_qr_solver(lambda x, args, p: x**2 - p)
    p = jnp.asarray(4.0)
    first = solver.solve(jnp.array([1.0]), p=p, max_steps=50, atol=1e-6)
    warm = solver.solve(first.x, p=p, max_steps=50, atol=1e-6)

    assert int(warm.status) == LMStatus.CONVERGED
    assert int(warm.steps) == 0
    assert jnp.allclose(warm.x, first.x)


def test_jit_python_and_vmap_solve_paths_agree():
    W = 0.1 * jax.random.normal(jax.random.key(64), (3, 3))

    def residual(z, _, p):
        return z + jnp.tanh(W @ z) - p

    solver = augmented_qr_solver(residual)
    ps = jax.random.normal(jax.random.key(65), (4, 3))
    z0 = jnp.zeros(3)

    compiled = solver.solve(z0, p=ps[0], max_steps=50, atol=1e-6)
    eager = solver.solve(z0, p=ps[0], max_steps=50, atol=1e-6, jit=False)
    batched = jax.vmap(lambda p: solver.solve(z0, p=p, max_steps=50, atol=1e-6))(ps)

    assert int(compiled.status) == int(eager.status)
    assert int(compiled.steps) == int(eager.steps)
    assert jnp.allclose(compiled.x, eager.x, atol=1e-6)
    for i in range(ps.shape[0]):
        lane = solver.solve(z0, p=ps[i], max_steps=50, atol=1e-6)
        assert int(batched.status[i]) == int(lane.status)
        assert jnp.allclose(batched.x[i], lane.x, atol=1e-5)


def test_linear_root_jvp_and_vjp_match_closed_form():
    A = jnp.array([[2.0, 0.5], [-0.3, 1.5]])
    B = jnp.array([[1.0, -0.5, 0.2], [0.3, 1.0, -0.7]])

    def residual(x, _, p):
        return A @ x - B @ p

    solver = augmented_qr_solver(residual)

    def solved_x(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=50, atol=1e-6).x

    p = jnp.array([1.0, -0.5, 0.25])
    p_dot = jnp.array([0.2, -0.1, 0.3])
    x_bar = jnp.array([0.7, -0.4])
    x, x_dot = jax.jvp(solved_x, (p,), (p_dot,))
    _, pullback = jax.vjp(solved_x, p)
    (p_bar,) = pullback(x_bar)

    assert jnp.allclose(x, jnp.linalg.solve(A, B @ p), atol=1e-5)
    assert jnp.allclose(x_dot, jnp.linalg.solve(A, B @ p_dot), atol=2e-5)
    assert jnp.allclose(p_bar, B.T @ jnp.linalg.solve(A.T, x_bar), atol=2e-5)
    assert jnp.allclose(x_dot @ x_bar, p_dot @ p_bar, atol=1e-6)


def test_second_order_derivatives_compose():
    solver = augmented_qr_solver(lambda x, args, p: x**2 - p)

    def scalar_root(p):
        return solver.solve(jnp.array([1.0]), p=p, max_steps=60, atol=1e-7).x[0]

    p = jnp.asarray(4.0)
    expected_second = -1.0 / (4.0 * p**1.5)

    assert jnp.allclose(jax.hessian(scalar_root)(p), expected_second, atol=1e-5)
    assert jnp.allclose(
        jax.jacrev(jax.jacfwd(scalar_root))(p), expected_second, atol=1e-5
    )


def test_has_aux_reports_and_differentiates_aux():
    def residual(x, args, p):
        return x**2 - p, {"scaled_root": x[0] * p}

    solver = augmented_qr_solver(residual, has_aux=True)

    def solved_aux(p):
        return solver.solve(jnp.array([1.0]), p=p, max_steps=50, atol=1e-6).aux[
            "scaled_root"
        ]

    p = jnp.asarray(4.0)
    value, tangent = jax.jvp(solved_aux, (p,), (jnp.asarray(1.0),))

    assert jnp.allclose(value, p * jnp.sqrt(p), atol=1e-4)
    assert jnp.allclose(tangent, 1.5 * jnp.sqrt(p), atol=2e-5)


def test_solve_composes_inside_outer_jit():
    solver = augmented_qr_solver(lambda x, args, p: x**2 - p)

    @jax.jit
    def stage(p):
        root = solver.solve(jnp.array([1.0]), p=p, max_steps=50, atol=1e-6).x[0]
        return root + p

    assert jnp.allclose(stage(jnp.asarray(4.0)), 6.0, atol=1e-5)
