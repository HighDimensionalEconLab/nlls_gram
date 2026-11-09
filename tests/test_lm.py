"""Tests for LevenbergMarquardt across unpenalized (ridge=None) and Tikhonov-penalized
(ridge > 0) objectives, linear solvers, metrics, geodesic acceleration, caching,
and AnnealRidge continuation.
"""

import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
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
    CholeskyMetric,
    Complementarity,
    DiagonalMetric,
    Equality,
    FischerBurmeister,
    GramCG,
    IdentityMetric,
    IdentityPreconditioner,
    Inequality,
    LevenbergMarquardt,
    LMState,
    Minimum,
    Newton,
    QRCache,
    RepeatedFactorMetric,
    Residuals,
    SolverAction,
    SolverStatus,
)

RNG = np.random.default_rng(11)
M, N = 4, 8
BLOCK, REPEATS, FREE = 3, 2, 2  # REPEATS * BLOCK + FREE == N
N_M = REPEATS * BLOCK
A_NP = RNG.normal(size=(M, N))
B_NP = RNG.normal(size=M)
W_ROOT = RNG.normal(size=(N, N))
W_NP = W_ROOT @ W_ROOT.T + N * np.eye(N)
K_ROOT = RNG.normal(size=(BLOCK, BLOCK + 2))
K_NP = K_ROOT @ K_ROOT.T + 0.5 * np.eye(BLOCK)
W_SEMI_NP = np.zeros((N, N))
for _j in range(REPEATS):
    W_SEMI_NP[_j * BLOCK : (_j + 1) * BLOCK, _j * BLOCK : (_j + 1) * BLOCK] = K_NP

A = jnp.asarray(A_NP, jnp.float32)
B = jnp.asarray(B_NP, jnp.float32)
L_W = jnp.asarray(np.linalg.cholesky(W_NP), jnp.float32)
K = jnp.asarray(K_NP, jnp.float32)
X0 = jnp.asarray(RNG.normal(size=N), jnp.float32)


def linear_residual(x):
    return A @ x - B


def nonlinear_residual(theta):
    return jnp.concatenate([A @ theta - B, jnp.array([theta[0] * theta[1] - 0.4])])


def make_repeated_metric():
    return RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=REPEATS)


def min_norm_solution(W, b=B_NP):
    Winv = np.linalg.inv(W)
    return Winv @ A_NP.T @ np.linalg.solve(A_NP @ Winv @ A_NP.T, b)


def damped_step(W, x, damping):
    r = A_NP @ x - B_NP
    return -np.linalg.solve(A_NP.T @ A_NP + damping * W, A_NP.T @ r)


def ridge_minimizer(ridge, W=W_SEMI_NP):
    return np.linalg.solve(A_NP.T @ A_NP + ridge * W, A_NP.T @ B_NP)


def min_seminorm_solution():
    kkt = np.block([[W_SEMI_NP, A_NP.T], [A_NP, np.zeros((M, M))]])
    rhs = np.concatenate([np.zeros(N), B_NP])
    return np.linalg.solve(kkt, rhs)[:N]


FORWARD_SOLVERS = {
    "cholesky_auto": Cholesky(),
    "cholesky_gram": Cholesky(form="gram"),
    "cholesky_normal": Cholesky(form="normal"),
    "qr": QR(),
    "qr_gram": QR(form="gram"),
    "qr_normal": QR(form="normal"),
    "cg": CG(IdentityPreconditioner(), tol=1e-12, maxiter=400),
    "gram_cg": GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=400),
}


@pytest.mark.parametrize("name", list(FORWARD_SOLVERS))
def test_unpenalized_step_and_minimum_metric_norm_convergence(name):
    damping = 0.07
    metric = CholeskyMetric(L_W)
    solver = LevenbergMarquardt(
        linear_residual,
        metric=metric,
        linear_solver=FORWARD_SOLVERS[name],
        init_damping=damping,
        geodesic_acceleration=False,
    )
    x1, _, info = solver.update(X0, solver.init(X0))
    expected_step = np.asarray(X0, np.float64) + damped_step(
        W_NP, np.asarray(X0, np.float64), damping
    )
    assert bool(info.accepted)
    np.testing.assert_allclose(np.asarray(x1), expected_step, rtol=2e-4, atol=2e-5)

    result = solver.solve(jnp.zeros(N), max_steps=200, atol=1e-6)
    assert int(result.status) == int(SolverStatus.CONVERGED)
    expected_root = min_norm_solution(W_NP)
    np.testing.assert_allclose(
        np.asarray(result.x), expected_root, rtol=2e-3, atol=2e-4
    )


def test_metric_variants_and_gram_qr_float32_conditioning():
    # DiagonalMetric, RepeatedFactorMetric, and default Euclidean metric select
    # their respective minimum-norm roots.
    weights = jnp.asarray(RNG.uniform(0.5, 3.0, size=N), jnp.float32)
    res_diag = LevenbergMarquardt(
        linear_residual, metric=DiagonalMetric(weights)
    ).solve(jnp.zeros(N), max_steps=200, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(res_diag.x),
        min_norm_solution(np.diag(np.asarray(weights, np.float64))),
        rtol=2e-3,
        atol=2e-4,
    )

    res_euclid = LevenbergMarquardt(linear_residual).solve(
        jnp.zeros(N), max_steps=200, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(res_euclid.x), min_norm_solution(np.eye(N)), rtol=2e-3, atol=2e-4
    )

    # Decaying singular values in float32: QR(form="gram") reaches well below
    # the squared-conditioning floor of Cholesky(form="gram").
    m_dec, n_dec = 24, 60
    U, _ = np.linalg.qr(RNG.normal(size=(m_dec, m_dec)))
    V, _ = np.linalg.qr(RNG.normal(size=(n_dec, m_dec)))
    A_dec = jnp.asarray(U @ np.diag(0.65 ** np.arange(m_dec)) @ V.T, jnp.float32)
    b_dec = jnp.asarray(RNG.normal(size=m_dec), jnp.float32)
    x0_dec = jnp.zeros(n_dec, jnp.float32)

    chol_loss = float(
        LevenbergMarquardt(
            lambda x: A_dec @ x - b_dec,
            linear_solver=Cholesky(form="gram"),
            geodesic_acceleration=False,
        )
        .solve(x0_dec, max_steps=300)
        .info.loss
    )
    qr_loss = float(
        LevenbergMarquardt(
            lambda x: A_dec @ x - b_dec,
            linear_solver=QR(form="gram"),
            geodesic_acceleration=False,
        )
        .solve(x0_dec, max_steps=300)
        .info.loss
    )
    assert qr_loss < 1e-2 * chol_loss
    assert qr_loss < 1e-7


@pytest.mark.parametrize("name", list(FORWARD_SOLVERS))
def test_geodesic_correction_matches_closed_form(name):
    def quad_res(x):
        return jnp.array([x[0] ** 2 - 2.0, x[0] * x[1] - 1.0])

    damping = 0.2
    solver = LevenbergMarquardt(
        quad_res,
        linear_solver=FORWARD_SOLVERS[name],
        init_damping=damping,
        geodesic_acceptance_ratio=1e9,
    )
    x0 = jnp.asarray([1.6, 0.7], jnp.float32)
    x1, _, info = solver.update(x0, solver.init(x0))
    x = np.asarray(x0, np.float64)
    J = np.array([[2 * x[0], 0.0], [x[1], x[0]]])
    r = np.array([x[0] ** 2 - 2.0, x[0] * x[1] - 1.0])
    G = J.T @ J + damping * np.eye(2)
    velocity = -np.linalg.solve(G, J.T @ r)
    f_vv = np.array([2 * velocity[0] ** 2, 2 * velocity[0] * velocity[1]])
    acceleration = -np.linalg.solve(G, J.T @ f_vv)
    assert bool(info.used_geodesic)
    np.testing.assert_allclose(
        np.asarray(x1), x + velocity + 0.5 * acceleration, rtol=1e-4, atol=1e-5
    )


def test_fixed_ridge_default_identity_and_repeated_metric():
    ridge = 1e-2
    # metric=None with ridge > 0 defaults to Euclidean IdentityMetric(n)
    euclid_solver = LevenbergMarquardt(linear_residual, ridge=ridge)
    euclid_res = euclid_solver.solve(jnp.zeros(N), max_steps=200, gtol=1e-5)
    assert int(euclid_res.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(
        np.asarray(euclid_res.x),
        ridge_minimizer(ridge, np.eye(N)),
        rtol=2e-3,
        atol=2e-4,
    )

    # RepeatedFactorMetric with unpenalized free tail
    metric = make_repeated_metric()
    solver = LevenbergMarquardt(linear_residual, metric=metric, ridge=ridge)
    result = solver.solve(jnp.zeros(N), max_steps=250, gtol=1e-5)
    assert int(result.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(
        np.asarray(result.x), ridge_minimizer(ridge), rtol=2e-3, atol=2e-4
    )
    np.testing.assert_allclose(
        float(result.info.loss),
        float(result.info.resid_loss) + ridge * float(result.info.penalty_value),
        rtol=1e-5,
    )

    # Whitened stationarity norm matches dense reference
    _, _, info = solver.update(X0, solver.init(X0))
    F64 = np.linalg.cholesky(K_NP).T
    F_bar = jsp_linalg.block_diag(*([F64] * REPEATS), np.eye(FREE))
    grad_white = np.linalg.solve(
        F_bar.T, A_NP.T @ (A_NP @ np.asarray(X0, np.float64) - B_NP)
    ) + ridge * np.concatenate(
        [(F_bar @ np.asarray(X0, np.float64))[:N_M], np.zeros(FREE)]
    )
    np.testing.assert_allclose(
        float(info.grad_norm), np.linalg.norm(grad_white), rtol=1e-4
    )


def test_ridge_solvers_agree_and_caches_are_reused_and_ridge_keyed():
    metric = make_repeated_metric()
    steps = {}
    for name, ls in (
        ("cholesky", Cholesky()),
        ("qr", QR()),
        ("cg", CG(IdentityPreconditioner(), tol=1e-10, maxiter=None)),
    ):
        solver = LevenbergMarquardt(
            nonlinear_residual,
            metric=metric,
            ridge=1e-2,
            init_damping=1e-2,
            linear_solver=ls,
            geodesic_acceleration=False,
        )
        steps[name] = np.asarray(solver.update(X0, solver.init(X0))[0])
    np.testing.assert_allclose(steps["cholesky"], steps["qr"], atol=1e-3)
    np.testing.assert_allclose(steps["cholesky"], steps["cg"], atol=1e-3)

    # CholeskyCache & QRCache reuse on matching ridge and invalidate on ridge change
    chol_solver = LevenbergMarquardt(
        nonlinear_residual,
        metric=metric,
        ridge=1e-3,
        linear_solver=Cholesky(),
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
        metric=metric,
        ridge=1e-3,
        linear_solver=QR(),
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


def test_rejected_step_reuses_cached_linearization():
    calls = []

    def scalar_residual(theta):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return theta**2 - 1.0

    x = jnp.asarray([0.1], dtype=jnp.float32)
    bt_solver = LevenbergMarquardt(
        scalar_residual,
        init_damping=1e-8,
        geodesic_acceleration=False,
    )
    _, _, bt_info = bt_solver.update(x, bt_solver.init(x))
    assert bool(bt_info.accepted)
    assert float(bt_info.stepsize) == pytest.approx(0.25)
    assert float(bt_info.damping_factor) == pytest.approx(2.0)

    solver = LevenbergMarquardt(
        scalar_residual,
        init_damping=1e-8,
        maxls=1,
        geodesic_acceleration=False,
    )
    state = solver.init(x)
    calls.clear()
    x1, state, info = solver.update(x, state)
    jax.effects_barrier()
    assert not bool(info.accepted)
    assert bool(state.jacobian_valid)
    assert len(calls) == 2

    calls.clear()
    solver.update(x1, state)
    jax.effects_barrier()
    assert len(calls) == 1

    # Passing a rejected-step state into solve() resets jacobian_valid so a new
    # x0 is freshly linearized rather than reusing the old point's residual/Jt.
    res_warm = solver.solve(
        jnp.asarray([0.8], dtype=jnp.float32), state=state, atol=1e-5, max_steps=20
    )
    assert int(res_warm.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(
        res_warm.x, jnp.asarray([1.0], dtype=jnp.float32), atol=1e-4
    )

    # Tall QR(form="gram") and strict < on unpenalized atol boundary
    tall_qr = LevenbergMarquardt(
        lambda th: jnp.array([th[0] - 1.0, th[1] - 2.0, th[0] + th[1] - 3.0]),
        linear_solver=QR(form="gram"),
    ).solve(jnp.zeros(2, dtype=jnp.float32), atol=1e-5)
    assert int(tall_qr.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(tall_qr.x, jnp.array([1.0, 2.0]), atol=1e-4)

    boundary_solver = LevenbergMarquardt(lambda th: th - 1.0)
    assert (
        int(
            boundary_solver.solve(
                jnp.zeros(1, dtype=jnp.float32), atol=1.0, max_steps=1
            ).steps
        )
        == 1
    )


def test_anneal_ridge_continuation_and_conjunctive_atol():
    x_dagger = min_seminorm_solution()
    metric = make_repeated_metric()
    fixed = LevenbergMarquardt(linear_residual, metric=metric, ridge=1e-1).solve(
        jnp.zeros(N), max_steps=300, gtol=1e-5
    )
    fixed_err = np.linalg.norm(np.asarray(fixed.x) - x_dagger)

    anneal = AnnealRidge(ridge_floor=1e-4, decrease=0.1, grad_rtol=1e-3)
    solver = LevenbergMarquardt(linear_residual, metric=metric, ridge=1e-1)
    cont = solver.solve(
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

    # Conjunctive atol rejects atol alone and does not stop at step 0 on an
    # interpolating start with excess seminorm.
    with pytest.raises(ValueError, match="conjunctive"):
        solver.solve(jnp.zeros(N), atol=1e-8)

    # Always-shrinking callback suppresses convergence on every step
    always_shrink = lambda ctx: SolverAction(  # noqa: E731
        state=dataclasses.replace(ctx.state, ridge=ctx.state.ridge * 0.5)
    )
    shrunk = solver.solve(jnp.zeros(N), max_steps=15, gtol=1e3, callback=always_shrink)
    assert int(shrunk.status) == int(SolverStatus.MAX_STEPS)


def test_pytree_aux_histories_and_jit_parity():
    def residual(x, args, p):
        r = A @ x["w"] - (args["b"] + p)
        return r, {"norm": jnp.linalg.norm(r)}

    x0 = {"w": jnp.zeros(N)}
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
    np.testing.assert_allclose(
        float(res_jit.aux["norm"]),
        float(jnp.linalg.norm(A @ res_jit.x["w"] - (B + p))),
        atol=1e-6,
    )


def test_residuals_wrapper_and_pytree_residual():
    x0 = jnp.zeros(N, dtype=jnp.float32)
    p = jnp.ones(M, dtype=jnp.float32) * 0.1
    ref = LevenbergMarquardt(lambda x, _, p: A @ x - (B + p)).solve(
        x0, p=p, max_steps=30, atol=1e-6
    )

    def wrapped_array(x, _, p):
        r = A @ x - (B + p)
        return Residuals(r)

    def wrapped_pytree(x, _, p):
        r = A @ x - (B + p)
        return Residuals({"top": r[:2], "bottom": r[2:]})

    def bare_pytree(x, _, p):
        r = A @ x - (B + p)
        return {"top": r[:2], "bottom": r[2:]}

    def wrapped_with_aux(x, _, p):
        r = A @ x - (B + p)
        return Residuals({"top": r[:2], "bottom": r[2:]}), {"norm": jnp.linalg.norm(r)}

    for fn in (wrapped_array, wrapped_pytree, bare_pytree):
        for solver_cls in (LevenbergMarquardt, Newton):
            solver = solver_cls(fn)
            res = solver.solve(x0, p=p, max_steps=30, atol=1e-6)
            assert int(res.status) == int(SolverStatus.CONVERGED)
            np.testing.assert_allclose(res.x, ref.x, atol=1e-5)

    for bad_fn in (
        lambda x: Equality(x),
        lambda x: Inequality(x),
        lambda x: Residuals(x, equality=Equality(x)),
        lambda x: Residuals(x, inequality=Inequality(x)),
    ):
        with pytest.raises(NotImplementedError):
            LevenbergMarquardt(bad_fn).solve(x0)

    jac_ref = jax.jacobian(
        lambda p_: (
            LevenbergMarquardt(lambda x, _, p: A @ x - (B + p))
            .solve(x0, p=p_, max_steps=30, atol=1e-6)
            .x
        )
    )(p)
    jac_wrapped = jax.jacobian(
        lambda p_: (
            LevenbergMarquardt(wrapped_pytree)
            .solve(x0, p=p_, max_steps=30, atol=1e-6)
            .x
        )
    )(p)
    np.testing.assert_allclose(jac_wrapped, jac_ref, atol=1e-5)

    res_aux = LevenbergMarquardt(wrapped_with_aux, has_aux=True).solve(
        x0, p=p, max_steps=30, atol=1e-6
    )
    assert int(res_aux.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(res_aux.x, ref.x, atol=1e-5)
    np.testing.assert_allclose(float(res_aux.aux["norm"]), 0.0, atol=1e-5)


def test_complementarity_euler_consumption_smoothing():
    # Two-period consumption-savings Euler equation with borrowing constraint a >= 0:
    #   0 <= a_i  perp  (w_i - a_i)^{-1} - beta * R * (R * a_i + y)^{-1} >= 0
    # Exact solution: a*(w_i; y) = max(0, (w_i - y / (beta * R)) / (1 + 1 / beta)).
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
        euler_rhs = 1.0 / c1 - beta * r_gross / c2
        return Complementarity(lhs=a, rhs=euler_rhs)

    def euler_residuals_none(a, _, y):
        return Residuals(complementarity=euler_comp_only(a, None, y))

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
            for fn in (euler_comp_only, euler_residuals_none):
                solver = solver_cls(fn, complementarity=cfg)
                sol = solver.solve(a_init, p=y0, max_steps=60, atol=1e-5, gtol=1e-5)
                assert int(sol.status) == int(SolverStatus.CONVERGED)
                np.testing.assert_allclose(sol.x, a_exact, atol=2e-4)

    # Implicit AD w.r.t. income parameter y across active and inactive constraints.
    da_dy_exact = jnp.where(w_grid > kappa * y0, -kappa / (1.0 + kappa * r_gross), 0.0)
    for cfg in (PDAS(), FischerBurmeister(), Minimum()):
        solver = LevenbergMarquardt(euler_comp_only, complementarity=cfg)
        da_dy = jax.jacrev(
            lambda y, s=solver: s.solve(a_init, p=y, max_steps=60, atol=1e-6).x
        )(y0)
        np.testing.assert_allclose(da_dy, da_dy_exact, atol=2e-4)

    # Mixed equality + complementarity system: solve jointly for (c1, a)
    # with budget c1 + a - w = 0 and 0 <= a perp u'(c1) - beta R u'(c2) >= 0.
    def euler_mixed(x, _, y):
        c1, a = x["c1"], x["a"]
        budget_eq = c1 + a - w_grid
        euler_rhs = 1.0 / c1 - beta * r_gross / (r_gross * a + y)
        return Residuals(
            residual=budget_eq,
            complementarity=Complementarity(lhs=a, rhs=euler_rhs),
        )

    x_init_mixed = {"c1": w_grid - a_init, "a": a_init}
    for cfg in (
        PDAS(),
        FischerBurmeister(mu_init=0.02, mu_min=1e-5, mu_decay=0.3),
        Minimum(),
    ):
        sol_mixed = LevenbergMarquardt(euler_mixed, complementarity=cfg).solve(
            x_init_mixed, p=y0, max_steps=60, atol=1e-5
        )
        assert int(sol_mixed.status) == int(SolverStatus.CONVERGED)
        np.testing.assert_allclose(sol_mixed.x["a"], a_exact, atol=2e-4)
        np.testing.assert_allclose(sol_mixed.x["c1"], w_grid - a_exact, atol=2e-4)

    # Underdetermined Gram form (M < P): RBF policy interpolation of a(w) = Phi @ theta.
    centers = jnp.linspace(0.3, 2.4, 24, dtype=jnp.float32)
    phi = jnp.exp(-0.5 * ((w_grid[:, None] - centers[None, :]) / 0.25) ** 2)
    theta0 = jnp.zeros(24, dtype=jnp.float32)

    def euler_gram(theta, _, y):
        a = phi @ theta
        c1 = w_grid - a
        c2 = r_gross * a + y
        return Complementarity(lhs=a, rhs=1.0 / c1 - beta * r_gross / c2)

    for cfg in (PDAS(), FischerBurmeister(), Minimum()):
        sol_gram = LevenbergMarquardt(
            euler_gram,
            linear_solver=Cholesky(form="gram"),
            complementarity=cfg,
        ).solve(theta0, p=y0, max_steps=80, atol=1e-5)
        assert int(sol_gram.status) == int(SolverStatus.CONVERGED)
        np.testing.assert_allclose(phi @ sol_gram.x, a_exact, atol=3e-4)

    # Fischer-Burmeister float32 cancellation & dead-branch jacrev safety:
    # At (a, b) = (1e4, 1e-4) with mu=0, naive a + b - sqrt(a^2 + b^2) underflows to 0.0
    # in float32, whereas rationalized FB preserves ~2 * a * b / (a + b + s) ~= 1e-4.
    # At (a, b) = (-1.0, 0.0) with mu=0, a + b + s == 0 on the untaken branch, which
    # must not produce NaN in reverse-mode AD.
    fb_solver = LevenbergMarquardt(
        lambda x: Complementarity(lhs=x[:2], rhs=x[2:]),
        complementarity=FischerBurmeister(),
    )
    x_extreme = jnp.array([1e4, -1.0, 1e-4, 0.0], dtype=jnp.float32)
    r_fb, _ = fb_solver._residual_and_aux(x_extreme, None, None)
    J_fb = jax.jacrev(lambda z: fb_solver._residual_and_aux(z, None, None)[0])(
        x_extreme
    )
    np.testing.assert_allclose(float(r_fb[0]), 1e-4, rtol=1e-3)
    assert bool(jnp.all(jnp.isfinite(J_fb)))

    # Extreme-scale Fischer-Burmeister (lhs=1e20 in float32) avoids square overflow
    sol_large_fb = LevenbergMarquardt(
        lambda x: Complementarity(jnp.full_like(x, 1e20), x),
        complementarity=FischerBurmeister(),
    ).solve(jnp.array([1.0], dtype=jnp.float32), atol=1e-6)
    assert int(sol_large_fb.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(
        sol_large_fb.x, jnp.zeros(1, dtype=jnp.float32), atol=1e-6
    )

    # Positive-temperature PDAS (tau_decay=1.0) implicit AD and xtol convergence
    smooth_pdas = LevenbergMarquardt(
        lambda x, _, p: Complementarity(x, x - p),
        complementarity=PDAS(tau_init=1.0, tau_decay=1.0),
    )
    p1 = jnp.array([1.0], dtype=jnp.float32)
    sol_xtol = smooth_pdas.solve(
        jnp.array([0.5], dtype=jnp.float32), p=p1, max_steps=60, xtol=1e-6
    )
    assert int(sol_xtol.status) == int(SolverStatus.CONVERGED)
    jac_smooth = jax.jacrev(
        lambda p: (
            smooth_pdas.solve(
                jnp.array([0.5], dtype=jnp.float32), p=p, max_steps=60, atol=1e-6
            ).x
        )
    )(p1)
    s1 = jax.nn.sigmoid(1.0)
    np.testing.assert_allclose(
        float(jac_smooth[0, 0]), float(s1 + s1 * (1.0 - s1)), rtol=1e-4
    )

    # Integer lhs + float rhs promotes to float instead of truncating rhs to int
    sol_int_lhs = LevenbergMarquardt(
        lambda x: Complementarity(jnp.ones(1, dtype=jnp.int32), x - 0.5)
    ).solve(jnp.zeros(1, dtype=jnp.float32), atol=1e-6)
    np.testing.assert_allclose(
        sol_int_lhs.x, jnp.array([0.5], dtype=jnp.float32), atol=1e-5
    )

    # Cached PDAS under jit=False / disable_jit preserves base_mask on rejected steps
    with jax.disable_jit():
        sol_nojit = LevenbergMarquardt(
            euler_comp_only, init_damping=1e-6, cache_jacobian=True
        ).solve(a_init, p=y0, max_steps=60, atol=1e-5, jit=False)
        assert int(sol_nojit.status) == int(SolverStatus.CONVERGED)
        np.testing.assert_allclose(sol_nojit.x, a_exact, atol=2e-4)

    # Decoupled active-set trap: at x_trap = (-0.5, 5.0), lhs < rhs in both rows
    # and lhs depends only on x[0], giving a zero Jacobian column for x[1] under
    # hard active-set selection (PDAS(tau=0) and Minimum()), whereas
    # FischerBurmeister(mu=0) and PDAS(tau_init > 0) couple both branches and
    # converge to the unique root x* = (0.0, 1.0).
    def trap_ncp(x):
        lhs = jnp.array([x[0], x[0] + 1.0])
        rhs = jnp.array([x[1] - 1.0, x[1] - x[0] - 1.0])
        return Complementarity(lhs=lhs, rhs=rhs)

    x_trap = jnp.array([-0.5, 5.0], dtype=jnp.float32)
    x_trap_star = jnp.array([0.0, 1.0], dtype=jnp.float32)
    for stall_cfg in (PDAS(), Minimum()):
        sol_stall = LevenbergMarquardt(trap_ncp, complementarity=stall_cfg).solve(
            x_trap, max_steps=30, atol=1e-6
        )
        assert int(sol_stall.status) == int(SolverStatus.MAX_STEPS)
    for conv_cfg in (
        FischerBurmeister(),
        FischerBurmeister(mu_init=0.5, mu_min=0.0, mu_decay=0.5),
        PDAS(tau_init=1.0, tau_min=0.0, tau_decay=0.5),
    ):
        sol_conv = LevenbergMarquardt(trap_ncp, complementarity=conv_cfg).solve(
            x_trap, max_steps=40, atol=1e-6
        )
        assert int(sol_conv.status) == int(SolverStatus.CONVERGED)
        np.testing.assert_allclose(sol_conv.x, x_trap_star, atol=1e-5)


def test_lm_constructor_and_state_validation():
    metric = make_repeated_metric()
    for bad_ridge in (0.0, -1e-3, True):
        with pytest.raises(ValueError):
            LevenbergMarquardt(linear_residual, metric=metric, ridge=bad_ridge)
    for gram in (
        Cholesky(form="gram"),
        QR(form="gram"),
        GramCG(IdentityPreconditioner(), maxiter=8),
    ):
        with pytest.raises(ValueError, match="penalty rows"):
            LevenbergMarquardt(
                linear_residual, metric=metric, ridge=1e-3, linear_solver=gram
            )
    for bad_fwd in (SVD(), LU()):
        with pytest.raises(ValueError, match="linear_solver"):
            LevenbergMarquardt(linear_residual, linear_solver=bad_fwd)
    with pytest.raises(ValueError, match="ad_solver"):
        LevenbergMarquardt(linear_residual, ad_solver=QR())
    with pytest.raises(ValueError, match="free block"):
        LevenbergMarquardt(
            lambda theta: theta[:1], metric=IdentityMetric(3), ridge=1e-3
        ).init(jnp.zeros(2))

    bare = LevenbergMarquardt(
        linear_residual, metric=metric, ridge=1e-3, cache_jacobian=False
    )
    with pytest.raises(ValueError, match="ridge"):
        bare.update(jnp.zeros(N), LMState(jnp.asarray(1e-3), None))
    with pytest.raises(ValueError, match="strictly positive"):
        bare.solve(
            jnp.zeros(N),
            state=LMState(jnp.asarray(1e-3), jnp.asarray(0.0)),
            gtol=1e-5,
        )
