import dataclasses

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
    DiagonalPreconditioner,
    DiagonalRidge,
    FactorRidge,
    GramCG,
    IdentityPreconditioner,
    IdentityRidge,
    Inequality,
    InequalityPenalty,
    LevenbergMarquardt,
    MultiStart,
    Newton,
    RepeatedFactorRidge,
    Residuals,
    SolverAction,
    SolverContext,
    SolverStatus,
    vmap_rows,
)
from nlls_gram.linear_solvers import Subproblem
from nlls_gram.solver_base import _ADKrylovHyper

RNG = np.random.default_rng(17)
M, N = 4, 7
A_NP = RNG.normal(size=(M, N))
B_NP = RNG.normal(size=M)
A = jnp.asarray(A_NP, jnp.float32)
B = jnp.asarray(B_NP, jnp.float32)


def min_euclid_solution(b):
    return A_NP.T @ np.linalg.solve(A_NP @ A_NP.T, b)


def test_prepared_step_solve_residual_rhs_and_transpose_invariants():
    rng = np.random.default_rng(99)
    m, n = 4, 6
    J_np = rng.normal(size=(m, n))
    r_np = rng.normal(size=m)
    th_np = rng.normal(size=n)
    rhs_r_np = rng.normal(size=m)
    rhs_p_np = rng.normal(size=n)
    L_np = rng.normal(size=(3, 4))
    Q_np = np.zeros((n, n))
    Q_np[:4, :4] = L_np.T @ L_np
    rm = FactorRidge(jnp.asarray(L_np, jnp.float32))

    J = jnp.asarray(J_np, jnp.float32)
    Jt = J.T
    r = jnp.asarray(r_np, jnp.float32)
    th = jnp.asarray(th_np, jnp.float32)
    rhs_r = jnp.asarray(rhs_r_np, jnp.float32)
    rhs_p = jnp.asarray(rhs_p_np, jnp.float32)
    hyper = _ADKrylovHyper(
        iterative_tol=jnp.float32(1e-12),
        iterative_atol=jnp.float32(0.0),
        iterative_maxiter=256,
    )

    rho, mu = 0.2, 0.08
    H_ref = J_np.T @ J_np + rho * Q_np + mu * np.eye(n)
    g_ref = J_np.T @ r_np + rho * (Q_np @ th_np)
    v_ref = -np.linalg.solve(H_ref, g_ref)
    sr_ref = -np.linalg.solve(H_ref, J_np.T @ rhs_r_np)
    sr_t_ref = -(J_np @ np.linalg.solve(H_ref, rhs_p_np))

    for ls in (
        Cholesky(form="normal"),
        Cholesky(form="gram"),
        QR(form="normal"),
        QR(form="gram"),
        CG(IdentityPreconditioner()),
        SVD(),
    ):
        prep = ls.prepare(
            Subproblem(
                resid=r,
                theta=th,
                Jt=Jt if ls.materializes_jacobian else None,
                jvp_fn=(lambda v: J @ v) if not ls.materializes_jacobian else None,
                JT=(lambda u: Jt @ u) if not ls.materializes_jacobian else None,
                ridge=jnp.float32(rho),
                damping=jnp.float32(mu),
                ridge_metric=rm,
                penalty_grad=rm.matvec(th, SolverContext()),
                cache=None,
                cache_enabled=False,
                hyper=hyper,
                ctx=SolverContext(),
                penalized=True,
                undamped=False,
                active=jnp.asarray(True),
                preconditioner=getattr(ls, "preconditioner", None),
            )
        )
        assert bool(prep.solve_ok)
        np.testing.assert_allclose(prep.grad, g_ref, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(prep.velocity(), v_ref, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(
            prep.solve_residual_rhs(rhs_r), sr_ref, rtol=2e-4, atol=2e-5
        )
        _, vjp_fn = jax.vjp(prep.solve_residual_rhs, rhs_r)
        np.testing.assert_allclose(vjp_fn(rhs_p)[0], sr_t_ref, rtol=2e-4, atol=2e-5)

    G_wide = J_np @ J_np.T
    sr_wide_ref = -J_np.T @ np.linalg.solve(G_wide, rhs_r_np)
    sr_wide_t_ref = -np.linalg.solve(G_wide, J_np @ rhs_p_np)
    for ls in (
        Cholesky(form="gram"),
        QR(form="gram"),
        GramCG(IdentityPreconditioner()),
        SVD(),
    ):
        prep0 = ls.prepare(
            Subproblem(
                resid=r,
                theta=th,
                Jt=Jt if ls.materializes_jacobian else None,
                jvp_fn=(lambda v: J @ v) if not ls.materializes_jacobian else None,
                JT=(lambda u: Jt @ u) if not ls.materializes_jacobian else None,
                ridge=jnp.float32(0.0),
                damping=jnp.float32(0.0),
                ridge_metric=None,
                penalty_grad=jnp.zeros_like(th),
                cache=None,
                cache_enabled=False,
                hyper=hyper,
                ctx=SolverContext(),
                penalized=False,
                undamped=True,
                active=jnp.asarray(True),
                preconditioner=getattr(ls, "preconditioner", None),
            )
        )
        assert bool(prep0.solve_ok)
        np.testing.assert_allclose(
            prep0.solve_residual_rhs(rhs_r), sr_wide_ref, rtol=2e-4, atol=2e-5
        )
        _, vjp0 = jax.vjp(prep0.solve_residual_rhs, rhs_r)
        np.testing.assert_allclose(vjp0(rhs_p)[0], sr_wide_t_ref, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize(
    "ad_solver",
    [
        None,
        SVD(),
        Cholesky(form="gram"),
        QR(form="gram"),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=200),
    ],
)
def test_unpenalized_wide_jvp_and_vjp_transpose_identity(solver_cls, ad_solver):
    solver = solver_cls(
        lambda x, _, p: A @ x - p["b"],
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

    expected = min_euclid_solution(np.asarray(p_dot["b"], np.float64))
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
        QR(),
        CG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_square_nonlinear_sensitivity(ad_solver):
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


def test_ad_solver_resolution_and_validation():
    lm = LevenbergMarquardt(lambda x: x)
    assert isinstance(lm._resolved_ad_solver(8, 8, None), LU)
    assert isinstance(lm._resolved_ad_solver(4, 8, None), SVD)
    assert isinstance(lm._resolved_ad_solver(8, 4, None), SVD)
    assert isinstance(
        Newton(lambda x: x, linear_solver=SVD())._resolved_ad_solver(2, 2, None), SVD
    )

    with pytest.raises(ValueError, match="square"):
        LevenbergMarquardt(lambda x: x, ad_solver=LU())._resolved_ad_solver(4, 8, None)
    with pytest.raises(ValueError, match="m <= n"):
        LevenbergMarquardt(
            lambda x: x, ad_solver=GramCG(IdentityPreconditioner(), maxiter=8)
        )._resolved_ad_solver(9, 3, None)

    wide_fn = lambda x, _, p: A @ x - p  # noqa: E731
    with pytest.raises(ValueError, match="different space"):
        LevenbergMarquardt(
            wide_fn,
            linear_solver=GramCG(IdentityPreconditioner()),
            ad_solver=CG(None, penalty=1e-4),
        )


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize(
    "ad_solver",
    [
        None,
        Cholesky(form="normal"),
        Cholesky(form="gram"),
        Cholesky(form="normal", preconditioner=DiagonalPreconditioner(None)),
        QR(form="normal"),
        QR(form="gram"),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-10, maxiter=200),
    ],
)
def test_ridge_implicit_ad_matches_closed_form(solver_cls, ad_solver):
    block, repeats = 3, 2  # N = 7 (6 penalized + 1 free)
    root = RNG.normal(size=(block + 2, block))
    K_np = root.T @ root + 0.5 * np.eye(block)
    L_np = np.linalg.cholesky(K_np).T
    Q_semi = np.zeros((N, N))
    for j in range(repeats):
        Q_semi[j * block : (j + 1) * block, j * block : (j + 1) * block] = K_np
    rm = RepeatedFactorRidge(jnp.asarray(L_np, jnp.float32), repeats=repeats)
    ridge = 1e-2
    weights = jnp.asarray(RNG.normal(size=N), jnp.float32)
    p_dot = jnp.asarray(RNG.normal(size=M), jnp.float32)
    solver = solver_cls(
        lambda th, _, p: A @ th - p,
        ridge_metric=rm,
        ridge=ridge,
        ad_solver=ad_solver,
    )

    def run_x(p):
        return solver.solve(jnp.zeros(N), p=p, max_steps=200, gtol=1e-5).x

    _, jvp_x = jax.jvp(run_x, (B,), (p_dot,))
    _, vjp_fn = jax.vjp(run_x, B)
    (grad_p,) = vjp_fn(weights)

    H_inv_At = np.linalg.solve(A_NP.T @ A_NP + ridge * Q_semi, A_NP.T)
    expected_jvp = H_inv_At @ np.asarray(p_dot, np.float64)
    expected_vjp = H_inv_At.T @ np.asarray(weights, np.float64)
    np.testing.assert_allclose(np.asarray(jvp_x), expected_jvp, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(np.asarray(grad_p), expected_vjp, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(
        float(weights @ jvp_x), float(grad_p @ p_dot), rtol=2e-3, atol=1e-5
    )


def test_inequality_implicit_ad_active_and_inactive():
    # Minimize 0.5*(x0 - p0)^2 + 0.5*(x1 - p1)^2 + 0.5*w*min(x1 - 1.0, 0)^2
    # At p = (2.0, 0.0): dx0/dp0 = 1, x1* = (p1 + w)/(1 + w) (dx1/dp1 = 1/(1 + w))
    # At p = (2.0, 3.0): margin x1 - 1 > 0 inactive -> x1* = p1 (dx1/dp1 = 1)
    w = 4.0

    def res_fn(x, _, p):
        return Residuals(
            residual=x - p,
            inequality=Inequality(jnp.array([x[1] - 1.0], dtype=x.dtype)),
        )

    solver = LevenbergMarquardt(
        res_fn, inequality=InequalityPenalty(weight=w, smoothing=0.0)
    )
    J_active = jax.jacrev(
        lambda p: solver.solve(jnp.zeros(2, jnp.float32), p=p, gtol=1e-6).x
    )(jnp.array([2.0, 0.0], dtype=jnp.float32))
    np.testing.assert_allclose(J_active, np.diag([1.0, 1.0 / (1.0 + w)]), atol=1e-4)

    J_inactive = jax.jacrev(
        lambda p: (
            solver.solve(jnp.array([2.0, 3.0], dtype=jnp.float32), p=p, gtol=1e-6).x
        )
    )(jnp.array([2.0, 3.0], dtype=jnp.float32))
    np.testing.assert_allclose(J_inactive, np.eye(2), atol=1e-4)


def test_aux_tangents_second_order_ad_and_float0_leaves():
    def row_fn(x, idx, p):
        return jnp.sin(x[idx]) + x[idx + 2] - p["vec"][idx]

    batched = vmap_rows(row_fn, in_axes=(None, 0, None))
    indices = jnp.arange(3, dtype=jnp.int32)
    x0 = jnp.zeros(5, dtype=jnp.float32)
    p0 = {"vec": jnp.array([0.2, -0.3, 0.4], dtype=jnp.float32), "tag": jnp.int32(7)}
    solver = Newton(
        lambda x, idx, p: (batched(x, idx, p), x[0] * p["vec"][1] + p["vec"][0] ** 2),
        has_aux=True,
        jacobian_chunk=2,
    )

    p_dot = {
        "vec": jnp.ones(3, dtype=jnp.float32),
        "tag": jnp.zeros((), dtype=jax.dtypes.float0),
    }
    _, tangents = jax.jvp(
        lambda p_in: solver.solve(
            x0, indices, p=p_in, atol=1e-6, max_steps=10, save_steps=True
        ),
        (p0,),
        (p_dot,),
    )
    assert jnp.all(jnp.isfinite(tangents.x))
    assert jnp.isfinite(tangents.aux)
    assert float(tangents.info.loss) == 0.0
    assert float(jnp.max(jnp.abs(tangents.x_history))) == 0.0

    # VJP with integer leaf in p and integer leaf in args
    _, vjp_fn = jax.vjp(
        lambda p_in: solver.solve(x0, indices, p=p_in, atol=1e-6, max_steps=10).x,
        p0,
    )
    (cot_p,) = vjp_fn(jnp.ones_like(x0))
    assert jnp.all(jnp.isfinite(cot_p["vec"]))

    # Second-order jacfwd(jacrev(solve))
    def solve_vec(v):
        return solver.solve(
            x0, indices, p={"vec": v, "tag": jnp.int32(7)}, atol=1e-6, max_steps=10
        ).x

    H_mixed = jax.jacfwd(jax.jacrev(lambda v: jnp.sum(solve_vec(v) ** 2)))(p0["vec"])
    assert jnp.all(jnp.isfinite(H_mixed))

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
@pytest.mark.parametrize("ad_solver", [Cholesky(), QR(), LU(), SVD()])
def test_failed_lanes_produce_finite_zero_tangents_and_cotangents(
    solver_cls, ad_solver
):
    # At x0 = [0, 0], J(x0) = 0; lane 0 fails while lane 1 converges
    def residual_fn(x, _, p):
        return jnp.array([x[0] ** 2 - p[0], x[1] ** 2 - p[1]], dtype=jnp.float32)

    p0 = jnp.array([1.0, 4.0], dtype=jnp.float32)
    v_p = jnp.array([1.0, 1.0], dtype=jnp.float32)
    u_x = jnp.array([2.0, -3.0], dtype=jnp.float32)
    kwargs = {"linear_solver": SVD()} if solver_cls is Newton else {}
    solver = solver_cls(residual_fn, ad_solver=ad_solver, **kwargs)
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
    if not isinstance(ad_solver, LU):
        ridge_solver = solver_cls(
            residual_fn,
            ridge_metric=IdentityRidge(1),
            ridge=1e-2,
            ad_solver=ad_solver,
            **kwargs,
        )
        x0_ridge_batch = jnp.array([[100.0, 0.0], [1.0, 2.0]], dtype=jnp.float32)
        grad_ridge = jax.grad(
            lambda p, rs=ridge_solver: jnp.sum(
                jax.vmap(
                    lambda x_init: (
                        rs.solve(
                            x_init,
                            p=p,
                            gtol=1e-5,
                            max_steps=4,
                            max_steps_is_success=False,
                        ).x
                    )
                )(x0_ridge_batch)
            )
        )(p0)
        assert jnp.all(jnp.isfinite(grad_ridge))

    # Poisoned carried ridge_metric on failed callback falls back to initial
    if solver_cls is LevenbergMarquardt and isinstance(ad_solver, SVD):
        bad_rm_solver = LevenbergMarquardt(
            lambda x, _, p: jnp.sqrt(x) - p,
            ridge=1e-2,
            ridge_metric=DiagonalRidge(jnp.ones(1, dtype=jnp.float32)),
            ad_solver=SVD(),
        )

        def poison_cb(ctx):
            nan_rm = jax.tree.map(
                lambda leaf: jnp.full_like(leaf, jnp.nan), ctx.state.ridge_metric
            )
            return SolverAction(
                stop=True,
                status=SolverStatus.NONFINITE,
                x=-jnp.ones(1, dtype=jnp.float32),
                state=dataclasses.replace(ctx.state, ridge_metric=nan_rm),
            )

        def failed_poison(p):
            return bad_rm_solver.solve(
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


def test_batched_tangents_cotangents_and_ad_solve_failure_propagation():
    rng = np.random.default_rng(2026)
    m, n, d_p = 5, 8, 3
    A_w = jnp.asarray(rng.normal(size=(m, n)), jnp.float32)
    B_w = jnp.asarray(rng.normal(size=(m, d_p)), jnp.float32)
    R_k = jnp.linalg.cholesky(jnp.eye(3, dtype=jnp.float32) + 0.25, upper=True)
    rm = RepeatedFactorRidge(R_k, repeats=2)
    p0 = jnp.array([0.3, -0.2, 0.5], dtype=jnp.float32)

    def resid_aux(x, _, p):
        trunk = A_w @ x + B_w @ (p**2)
        aux = {
            "sum": jnp.sum(x * x) + jnp.sum(p),
            "tag": jnp.int32(7),
            "pos": x[0] > 0,
        }
        return trunk, aux

    ref_solver = LevenbergMarquardt(
        resid_aux, ridge=5e-2, ridge_metric=rm, linear_solver=SVD(), has_aux=True
    )

    def eval_pair(slv, p):
        res = slv.solve(jnp.zeros(n, jnp.float32), p=p, gtol=1e-6, max_steps=40)
        return res.x, res.aux["sum"]

    J_ref_fwd = jax.jacfwd(lambda p: eval_pair(ref_solver, p))(p0)
    J_ref_rev = jax.jacrev(lambda p: eval_pair(ref_solver, p))(p0)
    np.testing.assert_allclose(J_ref_fwd[0], J_ref_rev[0], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(J_ref_fwd[1], J_ref_rev[1], rtol=2e-4, atol=2e-5)

    for ls in (
        Cholesky(),
        QR(),
        QR(form="normal", preconditioner=DiagonalPreconditioner()),
        CG(IdentityPreconditioner(), tol=1e-6, maxiter=128),
    ):
        slv = LevenbergMarquardt(
            resid_aux, ridge=5e-2, ridge_metric=rm, linear_solver=ls, has_aux=True
        )
        fn = lambda p, s=slv: eval_pair(s, p)  # noqa: E731
        jf = jax.jacfwd(fn)(p0)
        jr = jax.jacrev(fn)(p0)
        _, f_jvp = jax.linearize(fn, p0)
        jl = jax.vmap(f_jvp, out_axes=(-1, -1))(jnp.eye(d_p, dtype=jnp.float32))
        _, f_vjp = jax.vjp(fn, p0)
        jv = jax.vmap(lambda ct, pb=f_vjp: pb((ct, jnp.float32(0.0)))[0])(
            jnp.eye(n, dtype=jnp.float32)
        )
        tol_ad = (1e-4, 1e-5) if isinstance(ls, CG) else (1e-5, 1e-6)
        np.testing.assert_allclose(jf[0], J_ref_fwd[0], rtol=5e-4, atol=5e-5)
        np.testing.assert_allclose(jr[0], J_ref_rev[0], rtol=5e-4, atol=5e-5)
        np.testing.assert_allclose(jl[0], jf[0], rtol=tol_ad[0], atol=tol_ad[1])
        np.testing.assert_allclose(jv, jr[0], rtol=tol_ad[0], atol=tol_ad[1])
        np.testing.assert_allclose(jf[1], J_ref_fwd[1], rtol=5e-4, atol=5e-5)
        np.testing.assert_allclose(jr[1], J_ref_rev[1], rtol=5e-4, atol=5e-5)

    # Singular explicit ad_solver=QR() propagates NaN in both JVP and VJP
    sing_solver = LevenbergMarquardt(
        lambda x, _, p: jnp.array([x[0] - p[0], 0.0 * x[1]]),
        ad_solver=QR(form="normal"),
    )

    def sing_fn(p):
        return sing_solver.solve(
            jnp.zeros(2, jnp.float32), p=p, atol=1e-6, max_steps=10
        ).x

    p_s = jnp.array([1.0], dtype=jnp.float32)
    _, jvp_sing = jax.jvp(sing_fn, (p_s,), (jnp.ones_like(p_s),))
    _, vjp_sing_fn = jax.vjp(sing_fn, p_s)
    (vjp_sing,) = vjp_sing_fn(jnp.ones(2, jnp.float32))
    assert bool(jnp.all(jnp.isnan(jvp_sing)))
    assert bool(jnp.all(jnp.isnan(vjp_sing)))

    # Non-converged Krylov AD (maxiter=1) propagates NaN in both JVP and VJP
    trunc_cg = LevenbergMarquardt(
        resid_aux,
        ridge=5e-2,
        ridge_metric=rm,
        linear_solver=Cholesky(),
        ad_solver=CG(IdentityPreconditioner(), tol=1e-6, maxiter=1),
        has_aux=True,
    )

    def trunc_fn(p):
        return trunc_cg.solve(jnp.zeros(n, jnp.float32), p=p, gtol=1e-6, max_steps=40).x

    _, jvp_trunc = jax.jvp(trunc_fn, (p0,), (jnp.ones_like(p0),))
    _, vjp_trunc_fn = jax.vjp(trunc_fn, p0)
    (vjp_trunc,) = vjp_trunc_fn(jnp.ones(n, jnp.float32))
    assert bool(jnp.all(jnp.isnan(jvp_trunc)))
    assert bool(jnp.all(jnp.isnan(vjp_trunc)))
