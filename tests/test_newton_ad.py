import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    LU,
    QR,
    SVD,
    Cholesky,
    GramCG,
    IdentityPreconditioner,
    Newton,
    NewtonStatus,
    vmap_rows,
)


@pytest.mark.parametrize(
    "ad_solver",
    [
        None,
        SVD(),
        Cholesky(),
        Cholesky(form="gram"),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_wide_linear_jvp_and_vjp_transpose_identity(ad_solver):
    key = jax.random.key(10)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    m, n = 5, 11
    A = jax.random.normal(k1, (m, n), dtype=jnp.float32)
    p0 = jax.random.normal(k2, (m,), dtype=jnp.float32)
    v_p = jax.random.normal(k3, (m,), dtype=jnp.float32)
    u_x = jax.random.normal(k4, (n,), dtype=jnp.float32)
    x0 = jnp.zeros(n, dtype=jnp.float32)

    solver = Newton(lambda x, _, p: A @ x - p, ad_solver=ad_solver)

    def solve_x(p):
        return solver.solve(x0, p=p, atol=1e-6, max_steps=5).x

    _, jvp_x = jax.jvp(solve_x, (p0,), (v_p,))
    _, vjp_fn = jax.vjp(solve_x, p0)
    (vjp_p,) = vjp_fn(u_x)

    # Expected minimum-norm tangent: dx/dp = A^+ = A' (A A')^{-1}
    expected_jvp = A.T @ jnp.linalg.solve(A @ A.T, v_p)
    np.testing.assert_allclose(jvp_x, expected_jvp, rtol=1e-5, atol=1e-5)

    # Dot-product transpose identity: <u_x, JVP(v_p)> == <VJP(u_x), v_p>
    lhs = jnp.dot(u_x, jvp_x)
    rhs = jnp.dot(vjp_p, v_p)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "ad_solver",
    [
        None,
        LU(),
        SVD(),
        Cholesky(),
        CG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_square_nonlinear_sensitivity_matches_finite_differences(ad_solver):
    def residual_fn(x, _, p):
        return jnp.array(
            [
                x[0] ** 2 + x[1] - p[0],
                x[0] + x[1] ** 2 - p[1],
            ],
            dtype=jnp.float32,
        )

    x0 = jnp.array([1.2, 1.8], dtype=jnp.float32)
    p0 = jnp.array([3.0, 5.0], dtype=jnp.float32)
    solver = Newton(residual_fn, ad_solver=ad_solver)

    def solve_x(p):
        return solver.solve(x0, p=p, atol=1e-6, max_steps=15).x

    J_fwd = jax.jacfwd(solve_x)(p0)
    J_rev = jax.jacrev(solve_x)(p0)

    eps = 1e-3
    fd_cols = []
    for i in range(2):
        e = jnp.zeros(2, dtype=jnp.float32).at[i].set(eps)
        fd_cols.append((solve_x(p0 + e) - solve_x(p0 - e)) / (2.0 * eps))
    J_fd = jnp.stack(fd_cols, axis=-1)

    np.testing.assert_allclose(J_fwd, J_rev, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(J_fwd, J_fd, rtol=2e-3, atol=2e-3)


def test_rotating_row_space_counterexample_selected_tangent():
    def residual_fn(x, _, p):
        a = jnp.array([1.0 + p, 1.0 + 3.0 * p], dtype=jnp.float32)
        return jnp.array([jnp.dot(a, x) - 1.0], dtype=jnp.float32)

    # At p = 0: a(0) = (1, 1), ||a(0)||^2 = 2.
    # From x0 = (0.5, 0.0), 1 step gives x*(p) = x0 - ((a.x0 - 1)/||a||^2) a.
    # At p = 0: x*(0) = (0.5, 0) - (-0.5 / 2) (1, 1) = (0.75, 0.25).
    # Then dr/dp at (x*(0), 0) is [1, 3] . [0.75, 0.25] = 1.5, so the selected
    # minimum-norm tangent is -J^+ (dr/dp) = (-0.75, -0.75), whereas the branch
    # derivative dx*(p)/dp at p = 0 also includes the rotation of a(p).
    x0 = jnp.array([0.5, 0.0], dtype=jnp.float32)
    p0 = jnp.asarray(0.0, dtype=jnp.float32)
    solver = Newton(residual_fn)

    def solve_x(p):
        return solver.solve(x0, p=p, atol=1e-6, max_steps=5).x

    x_star, jvp_x = jax.jvp(solve_x, (p0,), (jnp.asarray(1.0, dtype=jnp.float32),))
    np.testing.assert_allclose(
        x_star, jnp.array([0.75, 0.25], dtype=jnp.float32), atol=1e-6
    )
    # Selected minimum-norm tangent: -J^+ (dr/dp) = -0.5 * (1, 1) * 1.5 = (-0.75, -0.75)
    np.testing.assert_allclose(
        jvp_x, jnp.array([-0.75, -0.75], dtype=jnp.float32), rtol=1e-5, atol=1e-5
    )


def test_svd_forward_square_rank_deficient_defaults_to_svd_ad():
    # Square 2x2 rank-deficient system where row 1 == row 0
    def residual_fn(x, _, p):
        r = x[0] + x[1] - p
        return jnp.array([r, r], dtype=jnp.float32)

    x0 = jnp.zeros(2, dtype=jnp.float32)
    p0 = jnp.asarray(2.0, dtype=jnp.float32)
    solver = Newton(residual_fn, linear_solver=SVD(), ad_solver=None)
    assert isinstance(solver._resolved_ad_solver(2, 2), SVD)

    def solve_x(p):
        return solver.solve(x0, p=p, atol=1e-6, max_steps=5).x

    x_sol, dx_dp = jax.jvp(solve_x, (p0,), (jnp.asarray(1.0, dtype=jnp.float32),))
    np.testing.assert_allclose(
        x_sol, jnp.array([1.0, 1.0], dtype=jnp.float32), atol=1e-5
    )
    np.testing.assert_allclose(
        dx_dp, jnp.array([0.5, 0.5], dtype=jnp.float32), atol=1e-5
    )


@pytest.mark.parametrize("chunk", [None, 1, 2, 10])
@pytest.mark.parametrize("mode", ["auto", "fwd", "rev"])
def test_jacobian_chunk_and_mode_agreement_and_second_order_ad(chunk, mode):
    def row_fn(x, idx, p):
        return jnp.sin(x[idx]) + x[idx + 2] - p[idx]

    batched = vmap_rows(row_fn, in_axes=(None, 0, None))
    indices = jnp.arange(3, dtype=jnp.int32)
    x0 = jnp.zeros(5, dtype=jnp.float32)
    p0 = jnp.array([0.2, -0.3, 0.4], dtype=jnp.float32)

    solver = Newton(
        lambda x, idx, p: batched(x, idx, p),
        jacobian_mode=mode,
        jacobian_chunk=chunk,
    )

    def loss_at_root(p):
        sol = solver.solve(x0, indices, p=p, atol=1e-6, max_steps=10)
        return jnp.sum(sol.x**2)

    grad_p = jax.grad(loss_at_root)(p0)
    hess_p = jax.hessian(loss_at_root)(p0)
    assert jnp.all(jnp.isfinite(grad_p))
    assert jnp.all(jnp.isfinite(hess_p))

    ref_solver = Newton(lambda x, idx, p: batched(x, idx, p))
    ref_grad = jax.grad(
        lambda p: jnp.sum(
            ref_solver.solve(x0, indices, p=p, atol=1e-6, max_steps=10).x ** 2
        )
    )(p0)
    np.testing.assert_allclose(grad_p, ref_grad, rtol=1e-4, atol=1e-4)


def test_aux_tangent_and_zero_bookkeeping_tangents():
    def residual_fn(x, _, p):
        r = jnp.array([2.0 * x[0] - p[0], 3.0 * x[1] - p[1]], dtype=jnp.float32)
        aux = x[0] * p[1] + p[0] ** 2
        return r, aux

    x0 = jnp.zeros(2, dtype=jnp.float32)
    p0 = jnp.array([4.0, 6.0], dtype=jnp.float32)
    v_p = jnp.array([1.0, -2.0], dtype=jnp.float32)
    solver = Newton(residual_fn, has_aux=True)

    primals, tangents = jax.jvp(
        lambda x_in, p_in: solver.solve(
            x_in, p=p_in, atol=1e-6, max_steps=5, save_steps=True
        ),
        (x0, p0),
        (jnp.ones_like(x0), v_p),
    )

    # x*(p) = (p0/2, p1/3), so dx/dp(v_p) = (0.5, -2/3)
    np.testing.assert_allclose(
        tangents.x, jnp.array([0.5, -2.0 / 3.0], dtype=jnp.float32), rtol=1e-5
    )
    # aux = x0 * p1 + p0^2 = (p0/2)*p1 + p0^2 -> d(aux) = dx0*p1 + x0*dp1 + 2*p0*dp0
    expected_aux_dot = 0.5 * 6.0 + 2.0 * (-2.0) + 2.0 * 4.0 * 1.0
    np.testing.assert_allclose(tangents.aux, expected_aux_dot, rtol=1e-5)
    np.testing.assert_allclose(tangents.p, v_p)

    # Bookkeeping fields have exact zero tangents
    assert float(tangents.info.loss) == 0.0
    assert float(tangents.info.stepsize) == 0.0
    assert float(tangents.info.grad_norm) == 0.0
    assert float(jnp.max(jnp.abs(tangents.x_history))) == 0.0
    assert float(jnp.max(jnp.abs(tangents.aux_history))) == 0.0


@pytest.mark.parametrize(
    "ad_solver",
    [
        LU(),
        SVD(),
        Cholesky(),
        CG(IdentityPreconditioner(), tol=1e-8, maxiter=16),
        GramCG(IdentityPreconditioner(), tol=1e-8, maxiter=16),
    ],
)
def test_failed_lanes_with_singular_initial_jacobian_produce_finite_zero_jvp_and_vjp(
    ad_solver,
):
    # At x0 = [0, 0], J(x0) = 0 (singular) and step_status = NON_DESCENT.
    # Failed AD lanes must sanitize the operator before linear solve so both
    # JVP and VJP are finite zero.
    def residual_fn(x, _, p):
        return jnp.array([x[0] ** 2 - p[0], x[1] ** 2 - p[1]], dtype=jnp.float32)

    p0 = jnp.array([1.0, 4.0], dtype=jnp.float32)
    v_p = jnp.array([1.0, 1.0], dtype=jnp.float32)
    u_x = jnp.array([2.0, -3.0], dtype=jnp.float32)
    solver = Newton(residual_fn, linear_solver=SVD(), ad_solver=ad_solver)

    # Batch of 2 initial guesses: lane 0 is [0, 0] (fails at singular J(x0)=0),
    # lane 1 is [1.5, 2.5] (converges to [1, 2]).
    x0_batch = jnp.array([[0.0, 0.0], [1.5, 2.5]], dtype=jnp.float32)

    def batched_solve(p):
        res = jax.vmap(
            lambda x_init: solver.solve(x_init, p=p, atol=1e-5, max_steps=10)
        )(x0_batch)
        return res.x, res.status

    (x_out, status_out), (jvp_x, _) = jax.jvp(batched_solve, (p0,), (v_p,))
    assert int(status_out[0]) == int(NewtonStatus.NON_DESCENT)
    assert int(status_out[1]) == int(NewtonStatus.CONVERGED)
    assert jnp.all(jnp.isfinite(jvp_x))
    np.testing.assert_allclose(jvp_x[0], jnp.zeros(2, dtype=jnp.float32))
    np.testing.assert_allclose(
        jvp_x[1], jnp.array([0.5, 0.25], dtype=jnp.float32), rtol=1e-4
    )

    _, vjp_fn = jax.vjp(lambda p: batched_solve(p)[0], p0)
    (vjp_p,) = vjp_fn(jnp.stack([u_x, u_x]))
    assert jnp.all(jnp.isfinite(vjp_p))
    np.testing.assert_allclose(
        vjp_p, jnp.array([0.5 * u_x[0], 0.25 * u_x[1]], dtype=jnp.float32), rtol=1e-4
    )


def test_compilation_sharing_and_static_key_identity():
    def residual_fn(x, _, p):
        return jnp.sin(x) - p

    s1 = Newton(residual_fn, c1=1e-4, decrease_factor=0.8, maxls=30, linear_solver=QR())
    s2 = Newton(residual_fn, c1=1e-4, decrease_factor=0.8, maxls=30, linear_solver=QR())
    s3 = Newton(
        residual_fn, c1=1e-4, decrease_factor=0.8, maxls=30, linear_solver=SVD()
    )

    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert s1 != s3

    with pytest.raises(AttributeError, match="frozen after construction"):
        s1.c1 = 0.5

    # Verify actual JIT compile-cache reuse across fresh solves, resumed state,
    # and updated hyper in newton_state (zero extra traces after the first call).
    trace_count = 0

    def counted_residual(x, scale, p):
        nonlocal trace_count
        trace_count += 1
        return scale * jnp.sin(x) - p

    solver = Newton(counted_residual, reg=0.01, geodesic_acceleration=True)
    x0 = jnp.array([0.1, -0.2], dtype=jnp.float32)
    p0 = jnp.array([0.05, -0.1], dtype=jnp.float32)

    res1 = solver.solve(x0, jnp.float32(1.0), p=p0, atol=1e-5, max_steps=10)
    traces_after_first = trace_count
    assert traces_after_first > 0

    # Second solve with different x0/args/p reuses the compiled loop
    res2 = solver.solve(x0 * 2.0, jnp.float32(1.5), p=p0 * 0.5, atol=1e-5, max_steps=10)
    assert trace_count == traces_after_first

    # Third solve passing res1.newton_state (and modified hyper) also reuses it
    import dataclasses

    tweaked_state = dataclasses.replace(
        res1.newton_state,
        hyper=dataclasses.replace(res1.newton_state.hyper, reg=jnp.float32(0.05)),
    )
    res3 = solver.solve(
        x0, jnp.float32(1.2), p=p0, newton_state=tweaked_state, atol=1e-5, max_steps=10
    )
    assert int(res2.status) == int(NewtonStatus.CONVERGED)
    assert int(res3.status) == int(NewtonStatus.CONVERGED)
    assert trace_count == traces_after_first
