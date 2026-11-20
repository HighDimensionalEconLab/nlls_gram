import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    LU,
    PDAS,
    QR,
    SVD,
    AnnealRidge,
    Cholesky,
    CholeskyCache,
    Complementarity,
    DiagonalPreconditioner,
    DiagonalRidge,
    FactorRidge,
    FischerBurmeister,
    GramCG,
    IdentityPreconditioner,
    IdentityRidge,
    Inequality,
    InequalityPenalty,
    LevenbergMarquardt,
    LMState,
    MatrixRidge,
    Minimum,
    Newton,
    NewtonState,
    QRCache,
    RepeatedFactorRidge,
    Residuals,
    SolverAction,
    SolverStatus,
)

RNG = np.random.default_rng(11)
M, N = 4, 8
BLOCK, REPEATS, FREE = 3, 2, 2
N_M = REPEATS * BLOCK
A_NP = RNG.normal(size=(M, N))
B_NP = RNG.normal(size=M)
K_ROOT = RNG.normal(size=(BLOCK + 2, BLOCK))
K_NP = K_ROOT.T @ K_ROOT + 0.5 * np.eye(BLOCK)
L_K_NP = np.linalg.cholesky(K_NP).T
Q_SEMI_NP = np.zeros((N, N))
for _j in range(REPEATS):
    Q_SEMI_NP[_j * BLOCK : (_j + 1) * BLOCK, _j * BLOCK : (_j + 1) * BLOCK] = K_NP

A = jnp.asarray(A_NP, jnp.float32)
B = jnp.asarray(B_NP, jnp.float32)
L_K = jnp.asarray(L_K_NP, jnp.float32)
X0 = jnp.asarray(RNG.normal(size=N), jnp.float32)


def linear_residual(x):
    return jnp.matmul(A, x, precision=jax.lax.Precision.HIGHEST) - B


def nonlinear_residual(theta):
    return jnp.concatenate(
        [
            jnp.matmul(A, theta, precision=jax.lax.Precision.HIGHEST) - B,
            jnp.array([theta[0] * theta[1] - 0.4]),
        ]
    )


def make_repeated_ridge():
    return RepeatedFactorRidge(L_K, repeats=REPEATS)


def min_euclid_solution(b=B_NP):
    return A_NP.T @ np.linalg.solve(A_NP @ A_NP.T, b)


def ridge_minimizer(ridge, Q=Q_SEMI_NP):
    return np.linalg.solve(A_NP.T @ A_NP + ridge * Q, A_NP.T @ B_NP)


def min_seminorm_solution():
    kkt = np.block([[Q_SEMI_NP, A_NP.T], [A_NP, np.zeros((M, M))]])
    rhs = np.concatenate([np.zeros(N), B_NP])
    return np.linalg.solve(kkt, rhs)[:N]


FORWARD_SOLVERS = {
    "cholesky_gram": Cholesky(form="gram"),
    "cholesky_normal": Cholesky(form="normal"),
    "qr_gram": QR(form="gram"),
    "qr_normal": QR(form="normal"),
    "cg": CG(IdentityPreconditioner(), tol=1e-12, maxiter=400),
    "gram_cg": GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=400),
}


@pytest.mark.parametrize("name", list(FORWARD_SOLVERS))
def test_unpenalized_euclidean_step_and_minimum_norm_convergence(name):
    damping = 0.07
    solver = LevenbergMarquardt(
        linear_residual,
        linear_solver=FORWARD_SOLVERS[name],
        init_damping=damping,
        geodesic_acceleration=False,
    )
    x0_64 = np.asarray(X0, np.float64)
    r0 = A_NP @ x0_64 - B_NP
    expected_d = -np.linalg.solve(A_NP.T @ A_NP + damping * np.eye(N), A_NP.T @ r0)
    x1, _, info = solver.update(X0, solver.init(X0))
    assert bool(info.accepted)
    np.testing.assert_allclose(np.asarray(x1), x0_64 + expected_d, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(
        float(info.grad_norm), np.linalg.norm(A_NP.T @ r0), rtol=1e-5
    )

    result = solver.solve(jnp.zeros(N, jnp.float32), max_steps=200, atol=1e-6)
    assert int(result.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(
        np.asarray(result.x), min_euclid_solution(), rtol=2e-3, atol=2e-4
    )


def test_gram_qr_float32_conditioning_unpenalized():
    rng = np.random.default_rng(19)
    m_dec, n_dec = 24, 60
    U, _ = np.linalg.qr(rng.normal(size=(m_dec, m_dec)))
    V, _ = np.linalg.qr(rng.normal(size=(n_dec, m_dec)))
    A_dec = jnp.asarray(U @ np.diag(0.72 ** np.arange(m_dec)) @ V.T, jnp.float32)
    b_dec = jnp.asarray(rng.normal(size=m_dec), jnp.float32)
    x0_dec = jnp.zeros(n_dec, jnp.float32)

    def dec_residual(x):
        return jnp.matmul(A_dec, x, precision=jax.lax.Precision.HIGHEST) - b_dec

    chol_s = LevenbergMarquardt(
        dec_residual,
        init_damping=1e-12,
        linear_solver=Cholesky(form="gram"),
        geodesic_acceleration=False,
    )
    qr_s = LevenbergMarquardt(
        dec_residual,
        init_damping=1e-12,
        linear_solver=QR(form="gram"),
        geodesic_acceleration=False,
    )
    chol_loss = float(chol_s.update(x0_dec, chol_s.init(x0_dec))[2].loss)
    qr_loss = float(qr_s.update(x0_dec, qr_s.init(x0_dec))[2].loss)
    assert qr_loss < 1e-4 * chol_loss
    assert qr_loss < 1e-6


@pytest.mark.parametrize("name", list(FORWARD_SOLVERS))
def test_generalized_tikhonov_step_algebra_and_bordered_free_tail(name):
    rho = 0.08
    mu = 0.05
    rm = make_repeated_ridge()
    ls = FORWARD_SOLVERS[name]
    if isinstance(ls, GramCG):
        # GramCG requires full-coverage SPD penalized block (n_free == 0)
        rm_full = RepeatedFactorRidge(L_K, repeats=2)
        A_sub = A[:, :N_M]
        x0_sub = X0[:N_M]
        s_gram_cg = LevenbergMarquardt(
            lambda th: A_sub @ th - B,
            ridge=rho,
            ridge_metric=rm_full,
            linear_solver=ls,
            init_damping=mu,
            geodesic_acceleration=False,
        )
        x1_sub, _, info_sub = s_gram_cg.update(x0_sub, s_gram_cg.init(x0_sub))
        x0_64 = np.asarray(x0_sub, np.float64)
        A64 = np.asarray(A_sub, np.float64)
        Q64 = Q_SEMI_NP[:N_M, :N_M]
        r0 = A64 @ x0_64 - B_NP
        g_ref = A64.T @ r0 + rho * (Q64 @ x0_64)
        d_ref = -np.linalg.solve(A64.T @ A64 + rho * Q64 + mu * np.eye(N_M), g_ref)
        np.testing.assert_allclose(
            np.asarray(x1_sub), x0_64 + d_ref, rtol=2e-4, atol=2e-5
        )
        np.testing.assert_allclose(
            float(info_sub.grad_norm), np.linalg.norm(g_ref), rtol=1e-4
        )
        return

    solver = LevenbergMarquardt(
        linear_residual,
        ridge=rho,
        ridge_metric=rm,
        linear_solver=ls,
        init_damping=mu,
        geodesic_acceleration=False,
    )
    x1, _, info = solver.update(X0, solver.init(X0))
    x0_64 = np.asarray(X0, np.float64)
    r0 = A_NP @ x0_64 - B_NP
    g_ref = A_NP.T @ r0 + rho * (Q_SEMI_NP @ x0_64)
    H_ref = A_NP.T @ A_NP + rho * Q_SEMI_NP + mu * np.eye(N)
    d_ref = -np.linalg.solve(H_ref, g_ref)
    np.testing.assert_allclose(np.asarray(x1), x0_64 + d_ref, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(float(info.grad_norm), np.linalg.norm(g_ref), rtol=1e-4)
    np.testing.assert_allclose(
        float(info.loss),
        float(info.resid_loss) + rho * float(info.penalty_value),
        rtol=1e-5,
    )

    # Bordered free-tail stability at tiny damping (mu = 1e-7 in float32)
    s_tiny = LevenbergMarquardt(
        linear_residual,
        ridge=rho,
        ridge_metric=rm,
        linear_solver=ls,
        init_damping=1e-7,
        geodesic_acceleration=False,
    )
    x1_tiny, _, info_tiny = s_tiny.update(X0, s_tiny.init(X0))
    assert bool(jnp.all(jnp.isfinite(x1_tiny)))
    d_tiny_ref = -np.linalg.solve(
        A_NP.T @ A_NP + rho * Q_SEMI_NP + 1e-7 * np.eye(N), g_ref
    )
    np.testing.assert_allclose(
        np.asarray(x1_tiny), x0_64 + d_tiny_ref, rtol=1e-3, atol=1e-4
    )
    assert bool(jnp.isfinite(info_tiny.loss))


def test_ridge_completion_nullspace_decay_and_ridge_representations():
    rho = 0.25
    s_null = LevenbergMarquardt(
        lambda th: jnp.array([th[0] - 1.0], dtype=jnp.float32),
        ridge=rho,
        ridge_metric=IdentityRidge(),
    )
    res_null = s_null.solve(
        jnp.array([1.0, 3.0], dtype=jnp.float32), gtol=1e-6, max_steps=60
    )
    assert int(res_null.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(
        res_null.x, np.array([1.0 / (1.0 + rho), 0.0]), atol=1e-5
    )

    # Interior-zero DiagonalRidge, MatrixRidge, and FactorRidge match reference
    w_diag = jnp.array([1.5, 0.0, 2.5, 0.0, 0.8, 1.2], dtype=jnp.float32)
    Q_diag_np = np.diag(np.pad(np.asarray(w_diag, np.float64), (0, N - len(w_diag))))
    for ls in (
        Cholesky(form="normal"),
        Cholesky(form="gram"),
        QR(form="normal"),
        QR(form="gram"),
    ):
        res_d = LevenbergMarquardt(
            linear_residual,
            ridge=rho,
            ridge_metric=DiagonalRidge(w_diag),
            linear_solver=ls,
        ).solve(X0, gtol=1e-5, max_steps=80)
        assert int(res_d.status) == int(SolverStatus.CONVERGED)
        np.testing.assert_allclose(
            res_d.x, ridge_minimizer(rho, Q_diag_np), rtol=2e-3, atol=2e-4
        )

    rng = np.random.default_rng(5)
    L_fac = jnp.asarray(rng.normal(size=(4, 6)), jnp.float32)
    L_fac_np = np.asarray(L_fac, np.float64)
    Q_fac_np = np.zeros((N, N))
    Q_fac_np[:6, :6] = L_fac_np.T @ L_fac_np
    for rm in (
        MatrixRidge(jnp.asarray(Q_fac_np[:6, :6], jnp.float32)),
        FactorRidge(L_fac),
    ):
        for ls in (Cholesky(), QR()):
            res_f = LevenbergMarquardt(
                linear_residual, ridge=rho, ridge_metric=rm, linear_solver=ls
            ).solve(X0, gtol=1e-5, max_steps=80)
            assert int(res_f.status) == int(SolverStatus.CONVERGED)
            np.testing.assert_allclose(
                res_f.x, ridge_minimizer(rho, Q_fac_np), rtol=2e-3, atol=2e-4
            )


def test_dense_diagonal_preconditioning_and_capability_checks():
    # Multiscale tall system: column scaling improves conditioning without changing
    # Euclidean grad_norm, step_norm, or objective loss.
    scales = jnp.array([1e-3, 1.0, 1e3], dtype=jnp.float32)
    A_tall = (
        jnp.array(
            [[1.0, 0.2, -0.3], [0.4, 1.5, 0.1], [-0.2, 0.3, 1.2], [0.5, -0.4, 0.8]],
            dtype=jnp.float32,
        )
        * scales[None, :]
    )
    x_true = jnp.array([100.0, 0.5, -1e-3], dtype=jnp.float32)
    b_tall = A_tall @ x_true + jnp.array([0.01, -0.02, 0.01, -0.01], dtype=jnp.float32)
    x0_tall = jnp.zeros(3, dtype=jnp.float32)

    ref_solver = LevenbergMarquardt(
        lambda x: A_tall @ x - b_tall,
        init_damping=0.1,
        linear_solver=Cholesky(form="normal"),
        geodesic_acceleration=False,
    )
    x1_ref, _, info_ref = ref_solver.update(x0_tall, ref_solver.init(x0_tall))

    col_diag = jnp.sum(A_tall**2, axis=0)
    for pre in (DiagonalPreconditioner(None), DiagonalPreconditioner(col_diag)):
        for ls in (
            Cholesky(form="normal", preconditioner=pre),
            QR(form="normal", preconditioner=pre),
        ):
            s = LevenbergMarquardt(
                lambda x: A_tall @ x - b_tall,
                init_damping=0.1,
                linear_solver=ls,
                geodesic_acceleration=False,
            )
            x1, _, info = s.update(x0_tall, s.init(x0_tall))
            np.testing.assert_allclose(x1, x1_ref, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(info.grad_norm, info_ref.grad_norm, rtol=1e-5)
            np.testing.assert_allclose(info.step_norm, info_ref.step_norm, rtol=1e-4)
            np.testing.assert_allclose(info.loss, info_ref.loss, rtol=1e-5)

    # Multiscale wide system: Gram Cholesky and unpenalized Gram QR with equilibration
    row_scales = jnp.array([1e-2, 1.0, 1e2, 10.0], dtype=jnp.float32)
    A_wide = row_scales[:, None] * A
    b_wide = row_scales * B
    ref_wide = LevenbergMarquardt(
        lambda x: A_wide @ x - b_wide,
        init_damping=0.1,
        linear_solver=Cholesky(form="gram"),
        geodesic_acceleration=False,
    )
    xw_ref, _, infow_ref = ref_wide.update(X0, ref_wide.init(X0))
    row_diag = jnp.sum(A_wide**2, axis=1)
    for pre in (DiagonalPreconditioner(None), DiagonalPreconditioner(row_diag)):
        for ls in (
            Cholesky(form="gram", preconditioner=pre),
            QR(form="gram", preconditioner=pre),
        ):
            sw = LevenbergMarquardt(
                lambda x: A_wide @ x - b_wide,
                init_damping=0.1,
                linear_solver=ls,
                geodesic_acceleration=False,
            )
            xw, _, infow = sw.update(X0, sw.init(X0))
            np.testing.assert_allclose(xw, xw_ref, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(infow.grad_norm, infow_ref.grad_norm, rtol=1e-5)

    # Damped wide QR(form="normal") with column scaling preserves exact Euclidean step
    sw_qr_norm = LevenbergMarquardt(
        lambda x: A_wide @ x - b_wide,
        init_damping=0.1,
        linear_solver=QR(form="normal", preconditioner=DiagonalPreconditioner(None)),
        geodesic_acceleration=False,
    )
    xw_qn, _, _ = sw_qr_norm.update(X0, sw_qr_norm.init(X0))
    np.testing.assert_allclose(xw_qn, xw_ref, rtol=2e-4, atol=5e-5)
    with jax.enable_x64(True):
        A_w64 = jnp.asarray(A_wide, dtype=jnp.float64)
        b_w64 = jnp.asarray(b_wide, dtype=jnp.float64)
        X0_64 = jnp.asarray(X0, dtype=jnp.float64)
        xw_ref64, _, _ = LevenbergMarquardt(
            lambda x: A_w64 @ x - b_w64,
            init_damping=0.1,
            linear_solver=Cholesky(form="gram"),
            geodesic_acceleration=False,
        ).update(
            X0_64,
            LevenbergMarquardt(
                lambda x: A_w64 @ x - b_w64,
                init_damping=0.1,
                linear_solver=Cholesky(form="gram"),
                geodesic_acceleration=False,
            ).init(X0_64),
        )
        sw_qn64 = LevenbergMarquardt(
            lambda x: A_w64 @ x - b_w64,
            init_damping=0.1,
            linear_solver=QR(
                form="normal", preconditioner=DiagonalPreconditioner(None)
            ),
            geodesic_acceleration=False,
        )
        xw_qn64, _, _ = sw_qn64.update(X0_64, sw_qn64.init(X0_64))
        np.testing.assert_allclose(xw_qn64, xw_ref64, rtol=1e-12, atol=1e-12)

    # Capability rejections for unsupported QR configurations
    with pytest.raises(ValueError, match="m_aug >= n"):
        Newton(
            linear_residual,
            damping=0.0,
            linear_solver=QR(
                form="normal", preconditioner=DiagonalPreconditioner(None)
            ),
        ).init(X0)
    rk_def_n = Newton(
        lambda x: jnp.array([x[0] + x[1] + 1.0, x[0] + x[1] + 1.0], dtype=x.dtype),
        damping=0.0,
        linear_solver=QR(
            form="normal",
            preconditioner=DiagonalPreconditioner(jnp.array([4.0, 1.0], jnp.float32)),
        ),
    )
    assert int(rk_def_n.solve(jnp.zeros(2, jnp.float32), max_steps=5).status) == int(
        SolverStatus.LINEAR_SOLVE_FAILED
    )
    with pytest.raises(ValueError, match="does not support DiagonalPreconditioner"):
        LevenbergMarquardt(
            linear_residual,
            ridge=1e-2,
            linear_solver=QR(form="gram", preconditioner=DiagonalPreconditioner(None)),
        ).init(X0)


@pytest.mark.parametrize("name", list(FORWARD_SOLVERS))
def test_geodesic_correction_unpenalized_and_ridge_closed_form(name):
    def quad_res(x):
        return jnp.array([x[0] ** 2 - 2.0, x[0] * x[1] - 1.0], dtype=jnp.float32)

    damping = 0.2
    x0 = jnp.asarray([1.6, 0.7], jnp.float32)
    x = np.asarray(x0, np.float64)
    J = np.array([[2 * x[0], 0.0], [x[1], x[0]]])
    r = np.array([x[0] ** 2 - 2.0, x[0] * x[1] - 1.0])

    # Unpenalized closed-form velocity + 0.5 * acceleration
    solver = LevenbergMarquardt(
        quad_res,
        linear_solver=FORWARD_SOLVERS[name],
        init_damping=damping,
        geodesic_acceptance_ratio=1e9,
    )
    x1, _, info = solver.update(x0, solver.init(x0))
    G0 = J.T @ J + damping * np.eye(2)
    v0 = -np.linalg.solve(G0, J.T @ r)
    f_vv0 = np.array([2 * v0[0] ** 2, 2 * v0[0] * v0[1]])
    a0 = -np.linalg.solve(G0, J.T @ f_vv0)
    assert bool(info.used_geodesic)
    np.testing.assert_allclose(np.asarray(x1), x + v0 + 0.5 * a0, rtol=1e-4, atol=1e-5)

    rho = 0.3
    w_q = jnp.array([2.0, 0.5], dtype=jnp.float32)
    Q_mat = np.diag([2.0, 0.5])
    r_solver = LevenbergMarquardt(
        quad_res,
        ridge=rho,
        ridge_metric=DiagonalRidge(w_q),
        linear_solver=FORWARD_SOLVERS[name],
        init_damping=damping,
        geodesic_acceptance_ratio=1e9,
    )
    x1_r, _, info_r = r_solver.update(x0, r_solver.init(x0))
    H_r = J.T @ J + rho * Q_mat + damping * np.eye(2)
    v_r = -np.linalg.solve(H_r, J.T @ r + rho * (Q_mat @ x))
    f_vv_r = np.array([2 * v_r[0] ** 2, 2 * v_r[0] * v_r[1]])
    a_r = -np.linalg.solve(H_r, J.T @ f_vv_r)
    assert bool(info_r.used_geodesic)
    np.testing.assert_allclose(
        np.asarray(x1_r), x + v_r + 0.5 * a_r, rtol=1e-4, atol=1e-5
    )


def test_ridge_caches_rejected_step_reuse_and_anneal_continuation():
    rm = make_repeated_ridge()
    chol_solver = LevenbergMarquardt(
        nonlinear_residual,
        ridge_metric=rm,
        ridge=1e-3,
        linear_solver=Cholesky(form="normal"),
        geodesic_acceleration=False,
    )
    fresh = chol_solver.init(X0)
    x_clean, _, _ = chol_solver.update(X0, fresh)
    garbage = jnp.eye(N, dtype=fresh.solver_cache.G.dtype) * 123.0
    poisoned = dataclasses.replace(
        fresh, solver_cache=CholeskyCache(garbage, jnp.asarray(True), fresh.ridge)
    )
    assert not np.allclose(np.asarray(chol_solver.update(X0, poisoned)[0]), x_clean)
    stale = dataclasses.replace(
        fresh, solver_cache=CholeskyCache(garbage, jnp.asarray(True), fresh.ridge * 2.0)
    )
    np.testing.assert_allclose(
        np.asarray(chol_solver.update(X0, stale)[0]), x_clean, rtol=1e-6
    )

    qr_solver = LevenbergMarquardt(
        nonlinear_residual,
        ridge_metric=rm,
        ridge=1e-3,
        linear_solver=QR(form="normal"),
        geodesic_acceleration=False,
    )
    fresh_qr = qr_solver.init(X0)
    x_qr_clean, state_qr_clean, _ = qr_solver.update(X0, fresh_qr)
    r_garbage = jnp.ones_like(fresh_qr.solver_cache.R)
    poisoned_qr = dataclasses.replace(
        fresh_qr, solver_cache=QRCache(r_garbage, jnp.asarray(True), fresh_qr.ridge)
    )
    assert not np.allclose(np.asarray(qr_solver.update(X0, poisoned_qr)[0]), x_qr_clean)
    stale_qr = dataclasses.replace(
        fresh_qr,
        solver_cache=QRCache(r_garbage, jnp.asarray(True), fresh_qr.ridge * 2.0),
    )
    np.testing.assert_allclose(
        np.asarray(qr_solver.update(X0, stale_qr)[0]), x_qr_clean, rtol=1e-6
    )
    assert not bool(state_qr_clean.solver_cache.valid)

    # Rejected step reuses cached linearization
    calls = []

    def scalar_residual(theta):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return theta**2 - 1.0

    x = jnp.asarray([0.1], dtype=jnp.float32)
    solver_rej = LevenbergMarquardt(
        scalar_residual, init_damping=1e-8, maxls=1, geodesic_acceleration=False
    )
    state = solver_rej.init(x)
    calls.clear()
    x1, state, info = solver_rej.update(x, state)
    jax.effects_barrier()
    assert not bool(info.accepted)
    assert bool(state.jacobian_valid)
    assert len(calls) == 2

    calls.clear()
    solver_rej.update(x1, state)
    jax.effects_barrier()
    assert len(calls) == 1

    # AnnealRidge continuation converges toward KKT minimum-seminorm solution
    x_dagger = min_seminorm_solution()
    fixed = LevenbergMarquardt(linear_residual, ridge_metric=rm, ridge=1e-1).solve(
        jnp.zeros(N), max_steps=300, gtol=1e-5
    )
    fixed_err = np.linalg.norm(np.asarray(fixed.x) - x_dagger)

    anneal = AnnealRidge(ridge_floor=1e-4, decrease=0.1, grad_rtol=1e-3)
    solver_cont = LevenbergMarquardt(linear_residual, ridge_metric=rm, ridge=1e-1)
    cont = solver_cont.solve(
        jnp.zeros(N),
        max_steps=500,
        gtol=1e-6,
        atol=1e-4,
        callback=anneal,
        user_state=anneal.init_state(),
    )
    assert int(cont.status) == int(SolverStatus.CONVERGED)
    assert np.linalg.norm(np.asarray(cont.x) - x_dagger) < 0.05 * fixed_err
    assert float(cont.state.ridge) <= 1e-4 + 1e-9

    # Default atol=0 converges on gtol when ||r*|| > 0; explicit atol > 0 is conjunctive
    slv_pen = LevenbergMarquardt(linear_residual, ridge=1.0)
    nonzero_res = slv_pen.solve(jnp.zeros(N), max_steps=100, gtol=1e-5)
    assert int(nonzero_res.status) == int(SolverStatus.CONVERGED)
    assert float(nonzero_res.info.resid_loss) > 1e-3
    conj_blocked = slv_pen.solve(
        jnp.zeros(N),
        max_steps=20,
        gtol=1e-5,
        atol=1e-8,
        max_steps_is_success=False,
    )
    assert int(conj_blocked.status) == int(SolverStatus.MAX_STEPS)


def test_nonlinear_inequalities_kink_smoothing_diagnostics_and_callbacks():
    # 1. Exact kink derivative at g == 0 is strictly 0.0 (g < 0, never g <= 0)
    bare_ineq = LevenbergMarquardt(lambda x: Inequality(x))
    r_kink, _, _ = bare_ineq._eval_residual_and_aux(
        jnp.array([-2.0, 0.0, 3.0], dtype=jnp.float32), None, None
    )
    np.testing.assert_allclose(r_kink, jnp.array([-2.0, 0.0, 0.0], dtype=jnp.float32))
    J_fwd = jax.jacfwd(lambda z: bare_ineq._eval_residual_and_aux(z, None, None)[0])(
        jnp.array([-2.0, 0.0, 3.0], dtype=jnp.float32)
    )
    J_rev = jax.jacrev(lambda z: bare_ineq._eval_residual_and_aux(z, None, None)[0])(
        jnp.array([-2.0, 0.0, 3.0], dtype=jnp.float32)
    )
    expected_J = np.diag([1.0, 0.0, 0.0])
    np.testing.assert_allclose(J_fwd, expected_J)
    np.testing.assert_allclose(J_rev, expected_J)

    # 2. Smooth softplus overflow/NaN safety at extreme +/- margins
    smooth_ineq = LevenbergMarquardt(
        lambda x: Inequality(x),
        inequality=InequalityPenalty(weight=4.0, smoothing=0.1),
    )
    x_ext = jnp.array([-1e4, -0.05, 0.0, 1e4], dtype=jnp.float32)
    r_ext, _, _ = smooth_ineq._eval_residual_and_aux(x_ext, None, None)
    J_ext = jax.jacrev(lambda z: smooth_ineq._eval_residual_and_aux(z, None, None)[0])(
        x_ext
    )
    assert bool(jnp.all(jnp.isfinite(r_ext)))
    assert bool(jnp.all(jnp.isfinite(J_ext)))
    np.testing.assert_allclose(float(r_ext[0]), -2.0 * 1e4, rtol=1e-4)
    np.testing.assert_allclose(float(r_ext[3]), 0.0, atol=1e-7)

    # 3. Constrained projection: 0.5*(x0-0.5)^2 + 0.5*(x1+1)^2 + 0.5*w*min(x1-0.5, 0)^2
    # Exact minimizer for x0 = 0.5 and x1 = (-1.0 + 0.5*w)/(1 + w)
    w_pen = 9.0

    def mixed_fn(x):
        return Residuals(
            residual=jnp.array([x[0] - 0.5, x[1] + 1.0], dtype=x.dtype),
            inequality=Inequality({"margin": jnp.array([x[1] - 0.5], dtype=x.dtype)}),
        )

    x_init = jnp.array([-1.0, -2.0], dtype=jnp.float32)
    x_star_exact = np.array([0.5, (-1.0 + 0.5 * w_pen) / (1.0 + w_pen)])

    for solver_cls in (LevenbergMarquardt, Newton):
        res_exact = solver_cls(
            mixed_fn, inequality=InequalityPenalty(weight=w_pen, smoothing=0.0)
        ).solve(x_init, gtol=1e-6, max_steps=50)
        assert int(res_exact.status) == int(SolverStatus.CONVERGED)
        np.testing.assert_allclose(res_exact.x, x_star_exact, atol=1e-5)
        viol = max(0.5 - float(res_exact.x[1]), 0.0)
        np.testing.assert_allclose(
            float(res_exact.info.ineq_violation_norm), viol, rtol=1e-5
        )
        np.testing.assert_allclose(
            float(res_exact.info.ineq_max_violation), viol, rtol=1e-5
        )
        eq_expected = float((res_exact.x[0] - 0.5) ** 2 + (res_exact.x[1] + 1.0) ** 2)
        np.testing.assert_allclose(
            float(res_exact.info.eq_loss), eq_expected, rtol=1e-5
        )
        np.testing.assert_allclose(
            float(res_exact.info.loss), eq_expected + w_pen * viol**2, rtol=1e-5
        )

    # 4. Smooth solutions converge to exact solution as tau -> 0 and via smoothing_decay
    assert isinstance(LevenbergMarquardt(mixed_fn).linear_solver, QR)
    assert isinstance(Newton(mixed_fn).linear_solver, QR)
    errs = []
    for tau in (0.5, 0.05, 1e-3):
        sol_tau = LevenbergMarquardt(
            mixed_fn,
            linesearch_tol=1e-7,
            inequality=InequalityPenalty(weight=w_pen, smoothing=tau),
        ).solve(x_init, gtol=1e-5, max_steps=60)
        assert int(sol_tau.status) == int(SolverStatus.CONVERGED)
        errs.append(float(np.linalg.norm(np.asarray(sol_tau.x) - x_star_exact)))
    assert errs[2] < errs[1] < errs[0]
    assert errs[2] < 2e-3

    sol_decay = LevenbergMarquardt(
        mixed_fn,
        inequality=InequalityPenalty(
            weight=w_pen, smoothing=0.5, smoothing_min=1e-5, smoothing_decay=0.3
        ),
    ).solve(x_init, gtol=1e-5, max_steps=80)
    assert int(sol_decay.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(sol_decay.x, x_star_exact, atol=2e-4)

    # 5. Callback mutating state.ineq_weight suppresses premature convergence
    def grow_weight_cb(ctx):
        grow = ctx.step <= 3
        new_w = jnp.where(grow, ctx.state.ineq_weight * 2.0, ctx.state.ineq_weight)
        return SolverAction(state=dataclasses.replace(ctx.state, ineq_weight=new_w))

    s_cb = LevenbergMarquardt(
        mixed_fn, inequality=InequalityPenalty(weight=9.0 / 8.0, smoothing=0.0)
    )
    sol_cb = s_cb.solve(x_init, gtol=1e-5, max_steps=60, callback=grow_weight_cb)
    assert int(sol_cb.status) == int(SolverStatus.CONVERGED)
    assert int(sol_cb.steps) >= 4
    np.testing.assert_allclose(float(sol_cb.state.ineq_weight), 9.0, rtol=1e-6)
    np.testing.assert_allclose(sol_cb.x, x_star_exact, atol=1e-4)


def test_pytree_aux_histories_and_residuals_wrapper():
    def residual(x, args, p):
        r = A @ x["w"] - (args["b"] + p)
        return r, {"norm": jnp.linalg.norm(r)}

    x0 = {"w": jnp.zeros(N, jnp.float32)}
    args = {"b": B}
    p = jnp.ones(M, jnp.float32) * 0.1
    solver = LevenbergMarquardt(residual, has_aux=True)
    res_jit = solver.solve(x0, args, p=p, max_steps=30, atol=1e-5, save_steps=True)
    res_py = solver.solve(x0, args, p=p, max_steps=30, atol=1e-5, jit=False)
    assert int(res_jit.status) == int(SolverStatus.CONVERGED)
    assert int(res_jit.steps) == int(res_py.steps)
    np.testing.assert_allclose(res_jit.x["w"], res_py.x["w"], atol=1e-5)
    assert res_jit.x_history["w"].shape == (31, N)
    np.testing.assert_allclose(
        res_jit.x_history["w"][int(res_jit.steps)], res_jit.x["w"], rtol=1e-6
    )

    x0_flat = jnp.zeros(N, jnp.float32)
    ref = LevenbergMarquardt(lambda x, _, p: A @ x - (B + p)).solve(
        x0_flat, p=p, max_steps=30, atol=1e-6
    )
    for fn in (
        lambda x, _, p: Residuals(A @ x - (B + p)),
        lambda x, _, p: Residuals(
            {"top": (A @ x - (B + p))[:2], "bot": (A @ x - (B + p))[2:]}
        ),
        lambda x, _, p: {"top": (A @ x - (B + p))[:2], "bot": (A @ x - (B + p))[2:]},
    ):
        for solver_cls in (LevenbergMarquardt, Newton):
            res = solver_cls(fn).solve(x0_flat, p=p, max_steps=30, atol=1e-6)
            assert int(res.status) == int(SolverStatus.CONVERGED)
            np.testing.assert_allclose(res.x, ref.x, atol=1e-5)


def test_complementarity_euler_and_paired_validation():
    # Paired Complementarity shape/structure mismatch raises ValueError
    with pytest.raises(ValueError, match="shapes"):
        LevenbergMarquardt(lambda x: Complementarity(lhs=x[:2], rhs=x[:3])).init(
            jnp.zeros(4, jnp.float32)
        )
    with pytest.raises(ValueError, match="pytree"):
        LevenbergMarquardt(lambda x: Complementarity(lhs={"a": x}, rhs={"b": x})).init(
            jnp.zeros(2, jnp.float32)
        )

    beta = jnp.float32(0.95)
    r_gross = jnp.float32(1.05)
    kappa = 1.0 / (beta * r_gross)
    w_grid = jnp.linspace(0.5, 2.2, 10, dtype=jnp.float32)
    y0 = jnp.float32(1.1)
    a_exact = jnp.maximum(0.0, (w_grid - kappa * y0) / (1.0 + kappa * r_gross))
    a_init = jnp.full_like(w_grid, 0.25)

    def euler_comp_only(a, _, y):
        c1 = w_grid - a
        c2 = r_gross * a + y
        return Complementarity(lhs=a, rhs=1.0 / c1 - beta * r_gross / c2)

    configs = [
        None,
        Minimum(c_scale=0.5, w_comp=1.2),
        PDAS(c_scale=0.5, w_comp=1.2),
        PDAS(c_scale=0.5, w_comp=1.2, tau_init=0.05, tau_min=1e-4, tau_decay=0.4),
        FischerBurmeister(c_scale=0.5, w_comp=1.2),
        FischerBurmeister(
            c_scale=0.5, w_comp=1.2, mu_init=0.05, mu_min=1e-5, mu_decay=0.3
        ),
    ]
    for solver_cls in (LevenbergMarquardt, Newton):
        for cfg in configs:
            sol = solver_cls(euler_comp_only, complementarity=cfg).solve(
                a_init, p=y0, max_steps=60, atol=1e-5, gtol=1e-5
            )
            assert int(sol.status) == int(SolverStatus.CONVERGED)
            np.testing.assert_allclose(sol.x, a_exact, atol=2e-4)

    # Combined equality + inequality + complementarity system
    def euler_all_three(x, _, y):
        c1, a = x["c1"], x["a"]
        return Residuals(
            residual=c1 + a - w_grid,
            inequality=Inequality(c1 - 0.05),
            complementarity=Complementarity(
                lhs=a, rhs=1.0 / c1 - beta * r_gross / (r_gross * a + y)
            ),
        )

    sol_all = LevenbergMarquardt(euler_all_three, complementarity=PDAS()).solve(
        {"c1": w_grid - a_init, "a": a_init}, p=y0, max_steps=60, atol=1e-5
    )
    assert int(sol_all.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(sol_all.x["a"], a_exact, atol=2e-4)
    assert float(sol_all.info.ineq_max_violation) == 0.0

    # Float32 Fischer-Burmeister cancellation & dead-branch jacrev safety
    fb_solver = LevenbergMarquardt(
        lambda x: Complementarity(lhs=x[:2], rhs=x[2:]),
        complementarity=FischerBurmeister(),
    )
    x_extreme = jnp.array([1e4, -1.0, 1e-4, 0.0], dtype=jnp.float32)
    r_fb, _, _ = fb_solver._eval_residual_and_aux(x_extreme, None, None)
    J_fb = jax.jacrev(lambda z: fb_solver._eval_residual_and_aux(z, None, None)[0])(
        x_extreme
    )
    np.testing.assert_allclose(float(r_fb[0]), 1e-4, rtol=1e-3)
    assert bool(jnp.all(jnp.isfinite(J_fb)))


def test_lm_constructor_and_state_validation():
    rm = make_repeated_ridge()
    with pytest.raises(ValueError, match="ridge_metric requires a positive ridge"):
        LevenbergMarquardt(linear_residual, ridge=None, ridge_metric=rm)
    for bad_ridge in (0.0, -1e-3, True):
        with pytest.raises(ValueError):
            LevenbergMarquardt(linear_residual, ridge_metric=rm, ridge=bad_ridge)
    with pytest.raises(ValueError, match="linear_solver"):
        LevenbergMarquardt(linear_residual, linear_solver=LU())
    s_svd = LevenbergMarquardt(
        linear_residual, linear_solver=SVD(), geodesic_acceleration=False
    )
    s_chol = LevenbergMarquardt(
        linear_residual, linear_solver=Cholesky(), geodesic_acceleration=False
    )
    np.testing.assert_allclose(
        s_svd.update(X0, s_svd.init(X0))[0],
        s_chol.update(X0, s_chol.init(X0))[0],
        rtol=1e-4,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="exceeding"):
        LevenbergMarquardt(
            lambda theta: theta[:1], ridge_metric=IdentityRidge(3), ridge=1e-3
        ).init(jnp.zeros(2))

    bare = LevenbergMarquardt(
        linear_residual, ridge_metric=rm, ridge=1e-3, cache_jacobian=False
    )
    with pytest.raises(ValueError, match="ridge"):
        bare.update(jnp.zeros(N), LMState(jnp.asarray(1e-3), None))
    with pytest.raises(ValueError, match="strictly positive"):
        bare.solve(
            jnp.zeros(N),
            state=LMState(jnp.asarray(1e-3), jnp.asarray(0.0)),
            gtol=1e-5,
        )


def test_singular_ridge_damped_svd_and_terminal_callback_refresh():
    assert Cholesky().resolve_form(M, N, False, None) == "gram"
    assert QR().resolve_form(N, M, False, None) == "normal"

    R_sing = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    rm_sing = RepeatedFactorRidge(R_sing, repeats=2)
    s_sing = LevenbergMarquardt(
        lambda x: x[:2] - jnp.array([1.0, 2.0], jnp.float32),
        ridge=1e-2,
        ridge_metric=rm_sing,
        linear_solver=Cholesky(form="auto"),
    )
    res_sing = s_sing.solve(jnp.zeros(4, jnp.float32), gtol=1e-5, max_steps=15)
    assert int(res_sing.status) == int(SolverStatus.CONVERGED)
    with pytest.raises(ValueError, match="strictly positive definite"):
        LevenbergMarquardt(
            lambda x: x[:2],
            ridge=1e-2,
            ridge_metric=rm_sing,
            linear_solver=Cholesky(form="gram"),
        ).init(jnp.zeros(4, jnp.float32))

    J_tiny = jnp.diag(jnp.array([1.0, 1e-7], dtype=jnp.float32))
    s_svd = LevenbergMarquardt(
        lambda x: J_tiny @ x - jnp.array([0.0, 1.0], jnp.float32),
        linear_solver=SVD(),
        init_damping=1e-14,
        geodesic_acceleration=False,
    )
    x1_svd, _, _ = s_svd.update(
        jnp.zeros(2, jnp.float32), s_svd.init(jnp.zeros(2, jnp.float32))
    )
    assert float(x1_svd[1]) > 0.5

    s_ineq = LevenbergMarquardt(
        lambda x: Residuals(
            residual=x[:1] - 3.0,
            inequality=Inequality(x[1:] - 100.0),
            complementarity=Complementarity(x[:1], x[:1] + 2.0),
        ),
        complementarity=Minimum(),
        inequality=InequalityPenalty(weight=1.0, smoothing=1e-37, smoothing_decay=1.0),
    )
    info0 = s_ineq._initial_info(
        jnp.zeros(2, jnp.float32), s_ineq.init(jnp.zeros(2, jnp.float32)), None, None
    )
    assert bool(jnp.isfinite(info0.loss))
    assert float(info0.eq_loss) == pytest.approx(9.0, rel=1e-5)

    def stop_with_new_ridge(ctx):
        return SolverAction(
            stop=True, state=dataclasses.replace(ctx.state, ridge=jnp.float32(10.0))
        )

    s_cb = LevenbergMarquardt(
        lambda x: x - 1.0, ridge=0.1, ridge_metric=IdentityRidge()
    )
    res_cb = s_cb.solve(
        jnp.ones(2, jnp.float32), gtol=1e-6, max_steps=5, callback=stop_with_new_ridge
    )
    expected_loss = float(jnp.sum((res_cb.x - 1.0) ** 2) + 10.0 * jnp.sum(res_cb.x**2))
    assert float(res_cb.info.loss) == pytest.approx(expected_loss, rel=1e-5)


def test_penalized_newton_atol_bare_inequality_ad_and_zero_step_status():
    sol_n = Newton(lambda x: x - jnp.ones(1, jnp.float32), ridge=1.0).solve(
        jnp.zeros(1, jnp.float32), gtol=1e-5, atol=1e-3, max_steps=20
    )
    assert int(sol_n.status) != int(SolverStatus.CONVERGED)

    anneal = AnnealRidge(ridge_floor=1e-6, decrease=0.1, grad_rtol=1e-2)
    sol_na = Newton(lambda x: x - jnp.ones(1, jnp.float32), ridge=1.0).solve(
        jnp.zeros(1, jnp.float32),
        callback=anneal,
        user_state=anneal.init_state(),
        gtol=1e-5,
        atol=1e-3,
        max_steps=30,
    )
    assert int(sol_na.status) == int(SolverStatus.CONVERGED)
    assert float(jnp.abs(sol_na.x[0] - 1.0)) <= 1e-3

    g_iq = jax.grad(
        lambda p: jnp.sum(
            LevenbergMarquardt(lambda x, _, p_: Inequality(x - p_))
            .solve(jnp.array([2.0], jnp.float32), p=p, atol=1e-5)
            .x
        )
    )(jnp.array([1.0], jnp.float32))
    assert jnp.all(jnp.isfinite(g_iq)) and jnp.allclose(g_iq, 0.0)

    def trace_check(L):
        fr = FactorRidge(L)
        rfr = RepeatedFactorRidge(L, repeats=2)
        assert fr.penalized_is_spd is False
        assert rfr.penalized_is_spd is False
        return fr.L.sum() + rfr.L.sum()

    jax.eval_shape(trace_check, jnp.diag(jnp.array([1.0, 0.0], jnp.float32)))

    mr = MatrixRidge(jnp.array([[1.0, 0.0], [0.0, -1e-8]], dtype=jnp.float32))
    np.testing.assert_allclose(mr.Q, mr.L.T @ mr.L, atol=1e-12)

    for jit_flag in (True, False):
        r0 = LevenbergMarquardt(lambda x: x - 1.0).solve(
            jnp.array([0.0], jnp.float32), atol=1e-5, max_steps=0, jit=jit_flag
        )
        assert int(r0.status) == int(SolverStatus.MAX_STEPS)

    lm_gcg = LevenbergMarquardt(
        linear_residual,
        ridge=1e-2,
        ridge_metric=IdentityRidge(),
        linear_solver=GramCG(DiagonalPreconditioner(jnp.ones(M, jnp.float32))),
    )
    assert int(lm_gcg.solve(X0, gtol=1e-5, max_steps=40).status) == int(
        SolverStatus.CONVERGED
    )

    lm_cg = LevenbergMarquardt(
        lambda x: x - jnp.ones(3, jnp.float32), linear_solver=CG()
    )
    _, _, info_u = lm_cg.update(
        jnp.zeros(3, jnp.float32), LMState(jnp.float32(1e-3), hyper=lm_cg.hyperparams())
    )
    assert bool(info_u.accepted)
    nw_cg = Newton(lambda x: x - jnp.ones(3, jnp.float32), linear_solver=CG())
    _, _, info_nu = nw_cg.update(
        jnp.zeros(3, jnp.float32), NewtonState(hyper=nw_cg.hyperparams())
    )
    assert bool(info_nu.accepted)
