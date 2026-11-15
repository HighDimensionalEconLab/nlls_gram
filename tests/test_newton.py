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
    AnnealRidge,
    CallbackContext,
    Cholesky,
    DiagonalPreconditioner,
    FactorRidge,
    GramCG,
    IdentityPreconditioner,
    Inequality,
    InequalityPenalty,
    LevenbergMarquardt,
    Newton,
    Preconditioner,
    RepeatedFactorRidge,
    SolverAction,
    SolveResult,
    SolverStatus,
    register_pytree_dataclass,
)


@dataclasses.dataclass(frozen=True, eq=False)
class _DampingDividingPreconditioner(Preconditioner):
    requires_positive_damping = True

    def apply(self, v, damping, ctx):
        return v / damping


register_pytree_dataclass(
    _DampingDividingPreconditioner, data_fields=(), meta_fields=()
)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(form="gram"),
        Cholesky(form="gram", preconditioner=DiagonalPreconditioner(None)),
        QR(form="gram"),
        QR(form="gram", preconditioner=DiagonalPreconditioner(None)),
        SVD(),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_wide_linear_interpolation_one_step(linear_solver):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    m, n = 6, 14
    A = jax.random.normal(k1, (m, n), dtype=jnp.float32)
    b = jax.random.normal(k2, (m,), dtype=jnp.float32)
    x0 = jax.random.normal(k3, (n,), dtype=jnp.float32)

    solver = Newton(lambda x: A @ x - b, damping=0.0, linear_solver=linear_solver)
    x1, _, info = solver.update(x0, solver.init(x0))

    r0 = A @ x0 - b
    expected_x1 = x0 - A.T @ jnp.linalg.solve(A @ A.T, r0)
    np.testing.assert_allclose(x1, expected_x1, rtol=1e-5, atol=1e-5)
    assert bool(info.accepted)
    np.testing.assert_allclose(info.stepsize, 1.0, rtol=1e-6)
    assert int(info.num_linesearch_steps) == 1


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(form="normal"),
        Cholesky(form="gram"),
        QR(form="normal"),
        QR(form="gram"),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-12, maxiter=128),
    ],
)
def test_undamped_newton_ridge_bordered_free_tail_one_step(linear_solver):
    # At damping == 0.0 with RepeatedFactorRidge + free tail (n_f > 0), bordered
    # elimination solves H_0 d = -(J'r + rho Q x0) in one step with zero division by mu.
    rng = np.random.default_rng(42)
    m, block, repeats, free = 5, 3, 2, 2
    n = block * repeats + free
    A_np = rng.normal(size=(m, n))
    b_np = rng.normal(size=m)
    x0_np = rng.normal(size=n)
    root = rng.normal(size=(block + 1, block))
    K_np = root.T @ root + 0.5 * np.eye(block)
    L_np = np.linalg.cholesky(K_np).T
    Q_np = np.zeros((n, n))
    for j in range(repeats):
        Q_np[j * block : (j + 1) * block, j * block : (j + 1) * block] = K_np

    rho = 0.15
    expected_x = np.linalg.solve(A_np.T @ A_np + rho * Q_np, A_np.T @ b_np)

    A = jnp.asarray(A_np, jnp.float32)
    b = jnp.asarray(b_np, jnp.float32)
    x0 = jnp.asarray(x0_np, jnp.float32)
    rm = RepeatedFactorRidge(jnp.asarray(L_np, jnp.float32), repeats=repeats)
    solver = Newton(
        lambda x: A @ x - b,
        ridge=rho,
        ridge_metric=rm,
        damping=0.0,
        linear_solver=linear_solver,
        geodesic_acceleration=False,
    )
    x1, _, info = solver.update(x0, solver.init(x0))
    assert bool(info.accepted)
    np.testing.assert_allclose(np.asarray(x1), expected_x, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(),
        Cholesky(form="normal", preconditioner=DiagonalPreconditioner(None)),
        QR(),
        QR(form="normal", preconditioner=DiagonalPreconditioner(None)),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_tall_inconsistent_problem_direction_and_actual_slope(linear_solver):
    A = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=jnp.float32)
    b = jnp.array([1.0, 2.0, 5.0], dtype=jnp.float32)
    x0 = jnp.array([0.5, -0.5], dtype=jnp.float32)

    solver = Newton(lambda x: A @ x - b, damping=0.0, linear_solver=linear_solver)
    x1, _, info = solver.update(x0, solver.init(x0))

    r0 = A @ x0 - b
    grad = A.T @ r0
    expected_d = -jnp.linalg.solve(A.T @ A, grad)
    np.testing.assert_allclose(x1 - x0, expected_d, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        info.directional_derivative, jnp.dot(grad, expected_d), rtol=1e-5, atol=1e-5
    )
    res = solver.solve(x0, gtol=1e-5, max_steps=5)
    assert int(res.status) == int(SolverStatus.CONVERGED)


@pytest.mark.parametrize(
    "linear_solver",
    [
        LU(),
        QR(),
        Cholesky(),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_square_nonlinear_problem(linear_solver):
    def residual_fn(x):
        return jnp.array(
            [x[0] ** 2 + x[1] - 3.0, x[0] + x[1] ** 2 - 5.0], dtype=jnp.float32
        )

    x0 = jnp.array([1.2, 1.8], dtype=jnp.float32)
    res = Newton(residual_fn, linear_solver=linear_solver).solve(
        x0, atol=1e-5, max_steps=15
    )
    assert int(res.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(res.x, jnp.array([1.0, 2.0]), rtol=1e-5, atol=1e-5)


def test_armijo_backtracking_oracle_nonfinite_trials_and_jaxpr():
    x0 = jnp.array([3.0], dtype=jnp.float32)
    c1, beta = 1e-4, 0.5
    solver = Newton(
        jnp.arctan,
        c1=c1,
        decrease_factor=beta,
        maxls=30,
        damping=0.0,
        geodesic_acceleration=False,
    )
    x1, _, info = solver.update(x0, solver.init(x0))

    r0 = float(np.arctan(3.0))
    j0 = 1.0 / (1.0 + 3.0**2)
    d0 = -r0 / j0
    slope0 = (j0 * r0) * d0
    alpha, trials = 1.0, 0
    for _ in range(30):
        trials += 1
        if float(np.arctan(3.0 + alpha * d0) ** 2) <= r0**2 + 2.0 * c1 * alpha * slope0:
            break
        alpha *= beta
    assert int(info.num_linesearch_steps) == trials
    np.testing.assert_allclose(info.stepsize, alpha, rtol=1e-5)
    np.testing.assert_allclose(x1[0], 3.0 + alpha * d0, rtol=1e-5)

    log_solver = Newton(
        jnp.log,
        c1=1e-4,
        decrease_factor=0.8,
        maxls=30,
        damping=0.0,
        geodesic_acceleration=False,
    )
    xs = jnp.array([[1.5], [10.0]], dtype=jnp.float32)
    state = log_solver.init(xs[0])
    _, _, info_vmap = jax.vmap(lambda x: log_solver.update(x, state))(xs)
    np.testing.assert_array_equal(info_vmap.num_linesearch_steps, np.array([1, 5]))
    np.testing.assert_allclose(info_vmap.stepsize, np.array([1.0, 0.8**4]), rtol=1e-5)

    exp_solver = Newton(
        lambda x: jnp.exp(x) - 1.0,
        maxls=10,
        damping=0.0,
        geodesic_acceleration=False,
    )
    res_fail = exp_solver.solve(
        jnp.array([-10.0], dtype=jnp.float32), max_steps=5, max_steps_is_success=True
    )
    assert int(res_fail.status) == int(SolverStatus.LINE_SEARCH_FAILED)

    jaxpr = jax.make_jaxpr(lambda x, s: log_solver.update(x, s))(xs[0], state)
    while_eqns = []

    def sub_jaxprs(params):
        out = []
        for v in params.values():
            if hasattr(v, "jaxpr"):
                out.append(v.jaxpr)
            elif isinstance(v, (tuple, list)):
                out.extend(item.jaxpr for item in v if hasattr(item, "jaxpr"))
        return out

    def collect_while(jpr):
        for eqn in jpr.eqns:
            if eqn.primitive.name == "while":
                while_eqns.append(eqn)
            for sub in sub_jaxprs(eqn.params):
                collect_while(sub)

    collect_while(jaxpr.jaxpr)
    assert len(while_eqns) == 1
    body_prims = {
        eqn.primitive.name for eqn in while_eqns[0].params["body_jaxpr"].jaxpr.eqns
    }
    assert body_prims.isdisjoint(
        {"qr", "cholesky", "svd", "lu", "triangular_solve", "custom_linear_solve"}
    )


def test_failure_statuses_exact_root_and_rank_deficient_svd():
    nan_x0 = jnp.array([jnp.nan], dtype=jnp.float32)
    assert int(Newton(lambda x: x - 1.0).solve(nan_x0).status) == int(
        SolverStatus.NONFINITE
    )

    def dup_res(x):
        return jnp.array([x[0] + x[1] - 2.0, x[0] + x[1] - 2.0], dtype=jnp.float32)

    x0 = jnp.zeros(2, dtype=jnp.float32)
    for ls in (Cholesky(), QR()):
        assert int(Newton(dup_res, damping=0.0, linear_solver=ls).solve(x0).status) == (
            int(SolverStatus.LINEAR_SOLVE_FAILED)
        )
        res_damped = Newton(dup_res, linear_solver=ls).solve(x0, atol=1e-5)
        assert int(res_damped.status) == int(SolverStatus.CONVERGED)
        np.testing.assert_allclose(dup_res(res_damped.x), 0.0, atol=1e-5)
    res_svd = Newton(dup_res, damping=0.0, linear_solver=SVD()).solve(x0, atol=1e-5)
    assert int(res_svd.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(res_svd.x, jnp.array([1.0, 1.0]), atol=1e-5)

    # Semidefinite ridge with shared nullspace at damping=0:
    # Cholesky/QR fail cleanly while SVD selects the minimum-norm step.
    L_rank1 = jnp.array([[1.0, 1.0]], dtype=jnp.float32)
    for ls in (Cholesky(form="normal"), QR(form="normal")):
        st_sing = (
            Newton(
                dup_res,
                ridge=0.5,
                ridge_metric=FactorRidge(L_rank1),
                damping=0.0,
                linear_solver=ls,
            )
            .solve(x0)
            .status
        )
        assert int(st_sing) == int(SolverStatus.LINEAR_SOLVE_FAILED)

    res_svd_semi = Newton(
        dup_res,
        ridge=0.5,
        ridge_metric=FactorRidge(L_rank1),
        damping=0.0,
        linear_solver=SVD(),
    ).solve(jnp.array([2.0, -2.0], dtype=jnp.float32), gtol=1e-5, max_steps=10)
    assert int(res_svd_semi.status) == int(SolverStatus.CONVERGED)
    # Along [1, 1] minimizer is [0.8, 0.8]; nullspace [1, -1] stays at x0 projection.
    np.testing.assert_allclose(
        res_svd_semi.x[0] + res_svd_semi.x[1], 8.0 / 5.0, atol=1e-5
    )

    zero_grad = Newton(lambda x: x**2 + 1.0, linear_solver=SVD())
    assert int(zero_grad.solve(jnp.array([0.0]), gtol=0.0).status) == int(
        SolverStatus.NON_DESCENT
    )
    assert int(zero_grad.solve(jnp.array([0.0]), gtol=1e-4).status) == int(
        SolverStatus.CONVERGED
    )

    res_root = Newton(lambda x: x**2, damping=0.0, linear_solver=Cholesky()).solve(
        jnp.zeros(3, dtype=jnp.float32)
    )
    assert int(res_root.status) == int(SolverStatus.CONVERGED)
    assert int(res_root.steps) == 0


@pytest.mark.parametrize("shape_mode", ["wide", "tall"])
def test_residual_coupled_damping_and_geodesic_acceleration(shape_mode):
    k1, k2 = jax.random.split(jax.random.key(123))
    if shape_mode == "wide":
        A_half = 0.3 * jax.random.normal(k1, (3, 8), dtype=jnp.float32)
        A = jnp.concatenate([A_half, A_half], axis=0)
        n = 8
        solvers = [
            Cholesky(form="gram"),
            Cholesky(form="normal"),
            QR(form="gram"),
            SVD(),
            GramCG(IdentityPreconditioner(), tol=1e-8, maxiter=32),
            CG(IdentityPreconditioner(), tol=1e-8, maxiter=32),
        ]
    else:
        A_half = 0.3 * jax.random.normal(k1, (8, 2), dtype=jnp.float32)
        A = jnp.concatenate([A_half, A_half], axis=1)
        n = 4
        solvers = [
            Cholesky(form="normal"),
            Cholesky(form="gram"),
            QR(form="normal"),
            QR(form="gram"),
            SVD(),
            CG(IdentityPreconditioner(), tol=1e-8, maxiter=32),
            GramCG(IdentityPreconditioner(), tol=1e-8, maxiter=32),
        ]
    x_star = jax.random.normal(k2, (n,), dtype=jnp.float32)
    b = jnp.sinh(A @ x_star)

    def residual_fn(x):
        return jnp.sinh(A @ x) - b

    x0 = jnp.zeros(n, dtype=jnp.float32)
    for ls in solvers:
        solver = Newton(
            residual_fn,
            damping=0.1,
            damping_power=1.5,
            min_damping=1e-10,
            linear_solver=ls,
            geodesic_acceleration=True,
        )
        res = solver.solve(x0, atol=1e-5, max_steps=25)
        assert int(res.status) == int(SolverStatus.CONVERGED)
        assert bool(res.info.used_geodesic)
        np.testing.assert_allclose(residual_fn(res.x), 0.0, atol=1e-5)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(),
        QR(),
        CG(IdentityPreconditioner(), tol=1e-6, maxiter=64),
    ],
)
def test_newton_ridge_matches_lm_ridge_and_supports_anneal_ridge(linear_solver):
    k1, k2, k3 = jax.random.split(jax.random.key(888), 3)
    m, n_m, n_f = 5, 6, 2
    n = n_m + n_f
    A = jax.random.normal(k1, (m, n), dtype=jnp.float32)
    b = jax.random.normal(k2, (m,), dtype=jnp.float32)
    M_raw = jax.random.normal(k3, (n_m, n_m), dtype=jnp.float32)
    L = jnp.linalg.cholesky(
        M_raw @ M_raw.T + jnp.eye(n_m, dtype=jnp.float32), upper=True
    )
    rm = FactorRidge(L)
    ridge = 1e-1
    x0 = jnp.zeros(n, dtype=jnp.float32)

    euclid_n = Newton(lambda x: A @ x - b, ridge=ridge, linear_solver=linear_solver)
    res_euclid = euclid_n.solve(x0, gtol=1e-5, max_steps=20)
    assert int(res_euclid.status) == int(SolverStatus.CONVERGED)
    expected_euclid = jnp.linalg.solve(
        A.T @ A + ridge * jnp.eye(n, dtype=jnp.float32), A.T @ b
    )
    np.testing.assert_allclose(res_euclid.x, expected_euclid, rtol=1e-4, atol=1e-4)

    def residual_fn(x, _, p):
        return jnp.sinh(A @ x) - p * b

    p0 = jnp.asarray(0.5, dtype=jnp.float32)
    rn = Newton(
        residual_fn,
        ridge_metric=rm,
        ridge=ridge,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
    )
    rlm = LevenbergMarquardt(
        residual_fn, ridge_metric=rm, ridge=ridge, linear_solver=linear_solver
    )
    x1, state1, _ = rn.update(x0, rn.init(x0, None, p=p0), None, p=p0)
    _, _, step1_info = rn.update(x1, state1, None, p=p0)
    assert bool(step1_info.used_geodesic)
    res_rn = rn.solve(x0, None, p=p0, gtol=1e-5, max_steps=30)
    res_rlm = rlm.solve(x0, None, p=p0, gtol=1e-5, max_steps=50)
    assert int(res_rn.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(res_rn.x, res_rlm.x, rtol=1e-4, atol=1e-4)

    anneal = AnnealRidge(ridge_floor=1e-4, decrease=0.1, grad_rtol=1e-1)
    res_anneal = Newton(
        lambda x: A @ x - b, ridge_metric=rm, ridge=1e-1, linear_solver=linear_solver
    ).solve(
        x0,
        callback=anneal,
        user_state=anneal.init_state(jnp.float32),
        gtol=1e-5,
        max_steps=40,
    )
    assert int(res_anneal.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(res_anneal.state.ridge, 1e-4, rtol=1e-5)


def test_newton_callbacks_inequality_and_validation():
    def residual_fn(x, target):
        return jnp.exp(x) - target, {"norm_x": jnp.linalg.norm(x)}

    x0 = jnp.array([-10.0], dtype=jnp.float32)
    target = jnp.array([1.0], dtype=jnp.float32)
    solver = Newton(residual_fn, maxls=3, cache_jacobian=True, has_aux=True)

    def rescue_cb(ctx: CallbackContext):
        rescue = ctx.info.step_status == SolverStatus.LINE_SEARCH_FAILED
        new_maxls = jnp.where(rescue, jnp.int32(60), ctx.state.hyper.maxls)
        new_hyper = dataclasses.replace(ctx.state.hyper, maxls=new_maxls)
        return SolverAction(
            state=dataclasses.replace(ctx.state, hyper=new_hyper),
            user_state=ctx.user_state + ctx.info.num_linesearch_steps,
        )

    res = solver.solve(
        x0,
        target,
        atol=1e-5,
        max_steps=15,
        callback=rescue_cb,
        user_state=jnp.int32(0),
        save_steps=True,
    )
    assert isinstance(res, SolveResult)
    assert int(res.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(res.x, jnp.zeros_like(x0), atol=1e-5)

    # Newton bare Inequality solve
    ineq_n = Newton(
        lambda x: Inequality(x - jnp.array([2.0, -1.0], dtype=x.dtype)),
        inequality=InequalityPenalty(weight=1.0, smoothing=0.0),
    ).solve(jnp.array([0.5, 0.5], dtype=jnp.float32), atol=1e-6, max_steps=10)
    assert int(ineq_n.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(ineq_n.x, jnp.array([2.0, 0.5]), atol=1e-5)

    wide_res = lambda x: x[:2] + x[2:4]  # noqa: E731
    tall_res = lambda x: jnp.concatenate([x, x + 1.0])  # noqa: E731
    with pytest.raises(ValueError, match="full column support"):
        Newton(wide_res, damping=0.0, linear_solver=Cholesky(form="normal")).init(
            jnp.ones(4)
        )
    with pytest.raises(ValueError, match="requires m <= n"):
        Newton(tall_res, damping=0.0, linear_solver=Cholesky(form="gram")).init(
            jnp.ones(2)
        )
    with pytest.raises(ValueError, match="square"):
        Newton(wide_res, damping=0.0, linear_solver=LU()).init(jnp.ones(4))
    with pytest.raises(ValueError, match="positive damping"):
        Newton(
            tall_res, damping=0.0, linear_solver=CG(_DampingDividingPreconditioner())
        )
    with pytest.raises(ValueError, match="damping"):
        Newton(lambda x: x, damping=0.1, linear_solver=LU())
    with pytest.raises(TypeError, match="NewtonState"):
        Newton(wide_res).solve(
            jnp.ones(4), state=LevenbergMarquardt(wide_res).init(jnp.ones(4))
        )

    # Newton with semidefinite RepeatedFactorRidge in auto resolves normal
    R_sing = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    res_semi = Newton(
        lambda x: x[:2] - jnp.array([1.0, -1.0], jnp.float32),
        ridge=1e-2,
        ridge_metric=RepeatedFactorRidge(R_sing, repeats=1),
        damping=1e-3,
        linear_solver=QR(form="auto"),
    ).solve(jnp.zeros(2, jnp.float32), gtol=1e-5, max_steps=15)
    assert int(res_semi.status) == int(SolverStatus.CONVERGED)
