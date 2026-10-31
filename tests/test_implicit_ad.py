"""Implicit differentiation tests for LevenbergMarquardt and Newton across
unpenalized (ridge=None) and Tikhonov-penalized (ridge > 0) objectives,
linear/AD solvers, has_aux, and failed-lane zero-tangent masking.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    LU,
    SVD,
    Cholesky,
    CholeskyMetric,
    DiagonalMetric,
    GramCG,
    IdentityMetric,
    IdentityPreconditioner,
    LevenbergMarquardt,
    MultiStart,
    Newton,
    RepeatedFactorMetric,
    SolverAction,
    SolverStatus,
    vmap_rows,
)

RNG = np.random.default_rng(17)
M, N = 4, 7
A_NP = RNG.normal(size=(M, N))
B_NP = RNG.normal(size=M)
W_ROOT = RNG.normal(size=(N, N))
W_NP = W_ROOT @ W_ROOT.T + N * np.eye(N)
A = jnp.asarray(A_NP, jnp.float32)
B = jnp.asarray(B_NP, jnp.float32)
L_W = jnp.asarray(np.linalg.cholesky(W_NP), jnp.float32)


def min_norm_solution(W, b):
    Winv = np.linalg.inv(W)
    return Winv @ A_NP.T @ np.linalg.solve(A_NP @ Winv @ A_NP.T, b)


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize(
    "ad_solver",
    [
        None,
        SVD(),
        Cholesky(),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=200),
    ],
)
def test_unpenalized_wide_jvp_and_vjp_transpose_identity(solver_cls, ad_solver):
    metric = CholeskyMetric(L_W)
    solver = solver_cls(
        lambda x, _, p: A @ x - p["b"],
        metric=metric,
        ad_solver=ad_solver,
    )
    p = {"b": B}
    p_dot = {"b": jnp.asarray(RNG.normal(size=M), jnp.float32)}
    cotangent = jnp.asarray(RNG.normal(size=N), jnp.float32)

    def run(pv):
        return solver.solve(jnp.zeros(N), p=pv, max_steps=150, atol=1e-6).x

    _, jvp_x = jax.jvp(run, (p,), (p_dot,))
    _, vjp_fn = jax.vjp(run, p)
    grad_b = vjp_fn(cotangent)[0]["b"]

    expected = min_norm_solution(W_NP, np.asarray(p_dot["b"], np.float64))
    np.testing.assert_allclose(np.asarray(jvp_x), expected, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(
        float(cotangent @ jvp_x), float(grad_b @ p_dot["b"]), rtol=2e-3, atol=1e-5
    )


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
def test_square_nonlinear_sensitivity_and_unwhitened_direct_lu(ad_solver):
    def residual_fn(x, _, p):
        return jnp.array(
            [x[0] ** 2 + x[1] - p[0], x[0] + x[1] ** 2 - p[1]], dtype=jnp.float32
        )

    x0 = jnp.array([1.2, 1.8], dtype=jnp.float32)
    p0 = jnp.array([3.0, 5.0], dtype=jnp.float32)
    solver = Newton(residual_fn, ad_solver=ad_solver)

    def solve_x(p):
        return solver.solve(x0, p=p, atol=1e-6, max_steps=15).x

    J_fwd = jax.jacfwd(solve_x)(p0)
    J_rev = jax.jacrev(solve_x)(p0)
    np.testing.assert_allclose(J_fwd, J_rev, rtol=1e-5, atol=1e-5)

    eps = 1e-3
    J_fd = jnp.stack(
        [
            (
                solve_x(p0 + jnp.zeros(2).at[i].set(eps))
                - solve_x(p0 - jnp.zeros(2).at[i].set(eps))
            )
            / (2.0 * eps)
            for i in range(2)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(J_fwd, J_fd, rtol=2e-3, atol=2e-3)


def test_ad_solver_resolution_and_selected_tangent():
    # Auto AD solver is LU when square (unless SVD/Krylov) and SVD when rectangular
    lm = LevenbergMarquardt(lambda x: x)
    assert isinstance(lm._resolved_ad_solver(8, 8), LU)
    assert isinstance(lm._resolved_ad_solver(4, 8), SVD)
    assert isinstance(lm._resolved_ad_solver(8, 4), SVD)
    assert isinstance(
        Newton(lambda x: x, linear_solver=SVD())._resolved_ad_solver(2, 2), SVD
    )

    with pytest.raises(ValueError, match="square"):
        LevenbergMarquardt(lambda x: x, ad_solver=LU())._resolved_ad_solver(4, 8)
    with pytest.raises(ValueError, match="m <= n"):
        LevenbergMarquardt(
            lambda x: x, ad_solver=GramCG(IdentityPreconditioner(), maxiter=8)
        )._resolved_ad_solver(9, 3)

    # Square direct LU tangent is invariant to metric
    A_sq = jnp.asarray(RNG.normal(size=(N, N)) + N * np.eye(N), jnp.float32)
    p0 = jnp.asarray(RNG.normal(size=N), jnp.float32)
    v_p = jnp.asarray(RNG.normal(size=N), jnp.float32)

    def sq_tangent(metric):
        s = LevenbergMarquardt(lambda x, _, p: A_sq @ x - p, metric=metric)
        return jax.jvp(
            lambda pv: s.solve(jnp.zeros(N), p=pv, max_steps=100, atol=1e-6).x,
            (p0,),
            (v_p,),
        )[1]

    np.testing.assert_allclose(
        sq_tangent(None), sq_tangent(CholeskyMetric(L_W)), rtol=1e-4, atol=1e-5
    )

    # Rotating row-space: returns selected minimum-norm tangent -J^+(dr/dp)
    def rot_res(x, _, p):
        a = jnp.array([1.0 + p, 1.0 + 3.0 * p], dtype=jnp.float32)
        return jnp.array([jnp.dot(a, x) - 1.0], dtype=jnp.float32)

    rot_solver = Newton(rot_res)
    _, jvp_rot = jax.jvp(
        lambda p: (
            rot_solver.solve(jnp.array([0.5, 0.0]), p=p, atol=1e-6, max_steps=5).x
        ),
        (jnp.float32(0.0),),
        (jnp.float32(1.0),),
    )
    np.testing.assert_allclose(jvp_rot, jnp.array([-0.75, -0.75]), rtol=1e-5, atol=1e-5)

    # Tall CG ad_solver (penalty=None): jacrev matches jacfwd
    A_tall = jnp.asarray(RNG.normal(size=(6, 3)), jnp.float32)
    p_tall = jnp.asarray(RNG.normal(size=6), jnp.float32)
    cg_tall_solver = LevenbergMarquardt(
        lambda x, _, p: A_tall @ x - p,
        linear_solver=CG(IdentityPreconditioner(), tol=1e-10, maxiter=64),
    )

    def solve_tall(p):
        return cg_tall_solver.solve(jnp.zeros(3, jnp.float32), p=p, atol=1e-6).x

    np.testing.assert_allclose(
        jax.jacfwd(solve_tall)(p_tall),
        jax.jacrev(solve_tall)(p_tall),
        rtol=1e-4,
        atol=1e-5,
    )

    # Cross-space preconditioner inheritance and form mismatch raise ValueError
    wide_fn = lambda x, _, p: A @ x - p  # noqa: E731
    with pytest.raises(ValueError, match="different space"):
        LevenbergMarquardt(
            wide_fn,
            linear_solver=GramCG(IdentityPreconditioner()),
            ad_solver=CG(None, penalty=1e-4),
        )
    with pytest.raises(ValueError, match="requires m <= n"):
        LevenbergMarquardt(
            lambda x, _, p: A_tall @ x - p, ad_solver=Cholesky(form="gram")
        )._ad_x_tangent_unpenalized(
            jnp.zeros(3, jnp.float32),
            None,
            p_tall,
            jnp.ones_like(p_tall),
            LevenbergMarquardt(lambda x, _, p: A_tall @ x - p).init(
                jnp.zeros(3, jnp.float32), p=p_tall
            ),
            jnp.asarray(True),
        )
    with pytest.raises(ValueError, match="requires n <= m"):
        LevenbergMarquardt(
            wide_fn, ad_solver=Cholesky(form="normal")
        )._ad_x_tangent_unpenalized(
            jnp.zeros(N, jnp.float32),
            None,
            B,
            jnp.ones_like(B),
            LevenbergMarquardt(wide_fn).init(jnp.zeros(N, jnp.float32), p=B),
            jnp.asarray(True),
        )


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize(
    "ad_solver",
    [None, Cholesky(), CG(IdentityPreconditioner(), tol=1e-10, maxiter=200)],
)
def test_ridge_implicit_ad_matches_closed_form(solver_cls, ad_solver):
    block, repeats = 3, 2  # N = 7 (6 metric + 1 free)
    root = RNG.normal(size=(block, block + 2))
    K_np = root @ root.T + 0.5 * np.eye(block)
    W_semi = np.zeros((N, N))
    for j in range(repeats):
        W_semi[j * block : (j + 1) * block, j * block : (j + 1) * block] = K_np
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(jnp.asarray(K_np, jnp.float32), upper=True),
        repeats=repeats,
    )
    ridge = 1e-2
    weights = jnp.asarray(RNG.normal(size=N), jnp.float32)
    solver = solver_cls(
        lambda th, _, p: A @ th - p,
        metric=metric,
        ridge=ridge,
        ad_solver=ad_solver,
    )

    def loss(p):
        res = solver.solve(jnp.zeros(N), p=p, max_steps=200, gtol=1e-5)
        return jnp.vdot(weights, res.x)

    grad = np.asarray(jax.grad(loss)(B))
    expected_grad = A_NP @ np.linalg.solve(
        A_NP.T @ A_NP + ridge * W_semi, np.asarray(weights, np.float64)
    )
    np.testing.assert_allclose(grad, expected_grad, rtol=2e-3, atol=2e-4)


def test_aux_tangents_second_order_ad_and_zero_bookkeeping():
    def row_fn(x, idx, p):
        return jnp.sin(x[idx]) + x[idx + 2] - p[idx]

    batched = vmap_rows(row_fn, in_axes=(None, 0, None))
    indices = jnp.arange(3, dtype=jnp.int32)
    x0 = jnp.zeros(5, dtype=jnp.float32)
    p0 = jnp.array([0.2, -0.3, 0.4], dtype=jnp.float32)
    solver = Newton(
        lambda x, idx, p: (batched(x, idx, p), x[0] * p[1] + p[0] ** 2),
        has_aux=True,
        jacobian_chunk=2,
    )

    primals, tangents = jax.jvp(
        lambda p_in: solver.solve(
            x0, indices, p=p_in, atol=1e-6, max_steps=10, save_steps=True
        ),
        (p0,),
        (jnp.ones_like(p0),),
    )
    assert jnp.all(jnp.isfinite(tangents.x))
    assert jnp.isfinite(tangents.aux)
    assert float(tangents.info.loss) == 0.0
    assert float(jnp.max(jnp.abs(tangents.x_history))) == 0.0

    hess = jax.hessian(
        lambda p_in: jnp.sum(
            solver.solve(x0, indices, p=p_in, atol=1e-6, max_steps=10).x ** 2
        )
    )(p0)
    assert jnp.all(jnp.isfinite(hess))

    # SVD implicit AD Hessian is finite and exact at repeated singular values
    rep_solver = LevenbergMarquardt(
        lambda x, _, p: (1.0 + p) * x[:2] - jnp.ones(2, dtype=jnp.float32),
        ad_solver=SVD(),
    )
    hess_rep = jax.hessian(
        lambda p_in: jnp.sum(
            rep_solver.solve(
                jnp.zeros(4, dtype=jnp.float32), p=p_in, atol=1e-6, max_steps=20
            ).x
        )
    )(jnp.float32(0.5))
    assert jnp.isfinite(hess_rep)
    np.testing.assert_allclose(hess_rep, 4.0 / (1.5**3), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
def test_failed_lanes_produce_finite_zero_tangents_and_cotangents(solver_cls):
    # At x0 = [0, 0], J(x0) = 0 (singular); lane 0 fails while lane 1 converges
    def residual_fn(x, _, p):
        return jnp.array([x[0] ** 2 - p[0], x[1] ** 2 - p[1]], dtype=jnp.float32)

    p0 = jnp.array([1.0, 4.0], dtype=jnp.float32)
    v_p = jnp.array([1.0, 1.0], dtype=jnp.float32)
    u_x = jnp.array([2.0, -3.0], dtype=jnp.float32)
    kwargs = {"linear_solver": SVD()} if solver_cls is Newton else {}
    solver = solver_cls(residual_fn, ad_solver=Cholesky(), **kwargs)
    x0_batch = jnp.array([[0.0, 0.0], [1.5, 2.5]], dtype=jnp.float32)

    def batched_solve(p):
        res = jax.vmap(
            lambda x_init: solver.solve(
                x_init,
                p=p,
                atol=1e-5,
                max_steps=15,
                max_steps_is_success=False,
            )
        )(x0_batch)
        return res.x, res.p

    (_, _), (jvp_x, jvp_p) = jax.jvp(batched_solve, (p0,), (v_p,))
    assert jnp.all(jnp.isfinite(jvp_x))
    np.testing.assert_allclose(jvp_x[0], jnp.zeros(2, dtype=jnp.float32))
    np.testing.assert_allclose(
        jvp_x[1], jnp.array([0.5, 0.25], dtype=jnp.float32), rtol=1e-4
    )
    np.testing.assert_allclose(jvp_p[0], v_p)

    _, vjp_fn = jax.vjp(lambda p: batched_solve(p)[0], p0)
    (vjp_p,) = vjp_fn(jnp.stack([u_x, u_x]))
    assert jnp.all(jnp.isfinite(vjp_p))
    np.testing.assert_allclose(
        vjp_p, jnp.array([0.5 * u_x[0], 0.25 * u_x[1]], dtype=jnp.float32), rtol=1e-4
    )

    # Ridge failed-lane AD with free block (n_f > 0) stays finite under vmap
    for ridge_ad in (Cholesky(), CG(IdentityPreconditioner(), tol=1e-8, maxiter=32)):
        ridge_solver = solver_cls(
            residual_fn,
            metric=IdentityMetric(1),
            ridge=1e-2,
            ad_solver=ridge_ad,
        )
        grad_ridge = jax.grad(
            lambda p, rs=ridge_solver: jnp.sum(
                jax.vmap(
                    lambda x_init: (
                        rs.solve(
                            x_init,
                            p=p,
                            gtol=1e-5,
                            atol=1e-1,
                            max_steps=25,
                            max_steps_is_success=False,
                        ).x
                    )
                )(x0_batch)
            )
        )(p0)
        assert jnp.all(jnp.isfinite(grad_ridge))

    # Poisoned carried metric on a failed callback stop falls back to initial metric
    bad_metric_solver = LevenbergMarquardt(
        lambda x, _, p: jnp.sqrt(x) - p,
        metric=DiagonalMetric(jnp.ones(1, dtype=jnp.float32)),
        ad_solver=SVD(),
    )

    def poison_cb(ctx):
        return SolverAction(
            stop=True,
            status=SolverStatus.NONFINITE,
            x=-jnp.ones(1, dtype=jnp.float32),
            state=dataclasses.replace(
                ctx.state,
                metric=DiagonalMetric(jnp.full((1,), jnp.nan, dtype=jnp.float32)),
            ),
        )

    def failed_poison(p):
        return bad_metric_solver.solve(
            jnp.ones(1, dtype=jnp.float32), p=p, max_steps=2, callback=poison_cb
        ).x

    _, tan_poison = jax.jvp(failed_poison, (jnp.float32(1.0),), (jnp.float32(1.0),))
    _, pb_poison = jax.vjp(failed_poison, jnp.float32(1.0))
    (cot_poison,) = pb_poison(jnp.ones(1, dtype=jnp.float32))
    assert float(tan_poison[0]) == 0.0
    assert float(cot_poison) == 0.0

    # Multi-start gradient flows through the winning start
    ms = MultiStart(
        key=jax.random.key(0),
        num_starts=2,
        draw=lambda k, x, a: jnp.array([1.5, 2.5], dtype=x.dtype),
    )
    grad_ms = jax.grad(
        lambda p: jnp.sum(
            solver.solve(
                jnp.zeros(2, dtype=jnp.float32),
                p=p,
                atol=1e-5,
                max_steps=15,
                max_steps_is_success=False,
                multi_start=ms,
            ).x
        )
    )(p0)
    np.testing.assert_allclose(grad_ms, jnp.array([0.5, 0.25]), rtol=1e-4)
