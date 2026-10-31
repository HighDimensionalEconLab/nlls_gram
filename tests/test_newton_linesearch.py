import jax
import jax.numpy as jnp
import numpy as np

from nlls_gram import (
    CG,
    QR,
    SVD,
    Cholesky,
    IdentityPreconditioner,
    Newton,
    NewtonStatus,
)


def test_full_step_acceptance_single_trial():
    def residual_fn(x):
        return 2.0 * x - 4.0

    x0 = jnp.array([0.0], dtype=jnp.float32)
    solver = Newton(residual_fn)
    x1, _, info = solver.update(x0, solver.init(x0))

    np.testing.assert_allclose(x1, jnp.array([2.0], dtype=jnp.float32))
    assert bool(info.accepted)
    np.testing.assert_allclose(info.stepsize, 1.0)
    np.testing.assert_allclose(info.trial_stepsize, 1.0)
    assert int(info.num_linesearch_steps) == 1
    assert int(info.step_status) == int(NewtonStatus.RUNNING)


def test_nonlinear_backtracking_matches_scalar_oracle_and_wide_embedding():
    # r(x) = arctan(x) from x0 = 3.0 overshoots violently on a full Newton step
    def scalar_res(x):
        return jnp.arctan(x)

    x0 = jnp.array([3.0], dtype=jnp.float32)
    c1 = 1e-4
    beta = 0.8
    solver = Newton(scalar_res, c1=c1, decrease_factor=beta, maxls=30)
    x1, _, info = solver.update(x0, solver.init(x0))

    # Independent scalar oracle
    r0 = float(np.arctan(3.0))
    j0 = 1.0 / (1.0 + 3.0**2)
    d0 = -r0 / j0
    slope0 = (j0 * r0) * d0
    loss0 = r0**2
    alpha = 1.0
    trials = 0
    for _ in range(30):
        trials += 1
        cand = 3.0 + alpha * d0
        cand_loss = float(np.arctan(cand) ** 2)
        if cand_loss <= loss0 + 2.0 * c1 * alpha * slope0:
            break
        alpha *= beta

    assert int(info.num_linesearch_steps) == trials
    assert trials > 1
    np.testing.assert_allclose(info.stepsize, alpha, rtol=1e-5)
    np.testing.assert_allclose(x1[0], 3.0 + alpha * d0, rtol=1e-5)

    # Embed in a wide problem where the first coordinate is x
    def wide_res(x):
        return jnp.array([jnp.arctan(x[0])], dtype=jnp.float32)

    x0_wide = jnp.array([3.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    wide_solver = Newton(wide_res, c1=c1, decrease_factor=beta, maxls=30)
    x1_wide, _, info_wide = wide_solver.update(x0_wide, wide_solver.init(x0_wide))
    assert int(info_wide.num_linesearch_steps) == trials
    np.testing.assert_allclose(x1_wide[0], x1[0], rtol=1e-5)
    np.testing.assert_allclose(x1_wide[1:], jnp.zeros(3, dtype=jnp.float32), atol=1e-7)


def test_nonfinite_log_trials_rejected_until_finite_candidate():
    # r(x) = log(x) from x = 10: d = -10 * log(10) ~ -23.02585
    # Trials for alpha = 1.0, 0.8, 0.64, 0.512 land at negative x (NaN log)
    # Trial 5 at alpha = 0.8**4 = 0.4096 lands at x ~ 0.5686 > 0 and is accepted.
    def residual_fn(x):
        return jnp.log(x)

    x0 = jnp.array([10.0], dtype=jnp.float32)
    solver = Newton(residual_fn, c1=1e-4, decrease_factor=0.8, maxls=30)
    x1, _, info = solver.update(x0, solver.init(x0))

    assert bool(info.accepted)
    assert int(info.num_linesearch_steps) == 5
    np.testing.assert_allclose(info.stepsize, 0.8**4, rtol=1e-6)
    np.testing.assert_allclose(info.trial_stepsize, 0.8**4, rtol=1e-6)
    assert float(x1[0]) > 0.0
    assert int(info.step_status) == int(NewtonStatus.RUNNING)


def test_exponential_overshoot_exhausts_30_trials_and_retains_base_point():
    # r(x) = exp(x) - 1 from x = -10: d ~ +22026.46, 30 trials exhausted
    def residual_fn(x):
        return jnp.exp(x) - 1.0

    x0 = jnp.array([-10.0], dtype=jnp.float32)
    solver = Newton(residual_fn, c1=1e-4, decrease_factor=0.8, maxls=30)
    x1, _, info = solver.update(x0, solver.init(x0))

    assert not bool(info.accepted)
    np.testing.assert_allclose(x1, x0)
    assert float(info.stepsize) == 0.0
    np.testing.assert_allclose(info.trial_stepsize, 0.8**29, rtol=1e-5)
    assert int(info.num_linesearch_steps) == 30
    assert int(info.step_status) == int(NewtonStatus.LINE_SEARCH_FAILED)
    np.testing.assert_allclose(info.loss, info.loss_old)

    res = solver.solve(x0, max_steps=10)
    assert int(res.status) == int(NewtonStatus.LINE_SEARCH_FAILED)
    assert int(res.steps) == 1
    np.testing.assert_allclose(res.x, x0)

    # Larger maxls succeeds in taking a finite accepted step
    solver_large = Newton(residual_fn, c1=1e-4, decrease_factor=0.8, maxls=55)
    x1_large, _, info_large = solver_large.update(x0, solver_large.init(x0))
    assert bool(info_large.accepted)
    assert int(info_large.step_status) == int(NewtonStatus.RUNNING)
    assert float(info_large.loss) < float(info_large.loss_old)


def test_mandatory_actual_slope_regression_tall_inconsistent():
    # r(x) = [x + 1, 100] at x = 0:
    # J = [[1], [0]], r(0) = [1, 100], ||r(0)||^2 = 10001
    # g = J'r = [1], d = -1, actual slope s = g'd = -1.
    # Candidate x = -1 has r(-1) = [0, 100], ||r(-1)||^2 = 10000.
    # True Armijo: 10000 <= 10001 + 2 * 1e-4 * 1 * (-1) = 10000.9998 (accepts alpha=1).
    # False shortcut (-||r||^2 = -10001): requires <= 9998.9998 (impossible).
    def residual_fn(x):
        return jnp.array([x[0] + 1.0, 100.0], dtype=jnp.float32)

    x0 = jnp.array([0.0], dtype=jnp.float32)
    solver = Newton(residual_fn)
    x1, _, info = solver.update(x0, solver.init(x0))

    assert bool(info.accepted)
    np.testing.assert_allclose(info.directional_derivative, -1.0, rtol=1e-6)
    np.testing.assert_allclose(info.stepsize, 1.0, rtol=1e-6)
    assert int(info.num_linesearch_steps) == 1
    np.testing.assert_allclose(x1, jnp.array([-1.0], dtype=jnp.float32), rtol=1e-6)


def test_actual_slope_for_truncated_svd_and_inexact_cg():
    # Rank-truncated SVD: second equation has zero Jacobian row and large
    # constant residual.
    def svd_res(x):
        return jnp.array([x[0] + x[1] - 2.0, 50.0], dtype=jnp.float32)

    x0 = jnp.zeros(2, dtype=jnp.float32)
    solver_svd = Newton(svd_res, linear_solver=SVD())
    x1, _, info_svd = solver_svd.update(x0, solver_svd.init(x0))
    assert bool(info_svd.accepted)
    np.testing.assert_allclose(info_svd.directional_derivative, -4.0, rtol=1e-5)
    np.testing.assert_allclose(info_svd.stepsize, 1.0, rtol=1e-5)
    np.testing.assert_allclose(x1, jnp.array([1.0, 1.0], dtype=jnp.float32), rtol=1e-5)

    # Inexact CG with maxiter=1 on a tall SPD normal system
    A = jnp.array([[2.0, 1.0], [1.0, 3.0], [0.5, -1.0]], dtype=jnp.float32)
    b = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    solver_cg = Newton(
        lambda x: A @ x - b,
        linear_solver=CG(IdentityPreconditioner(), tol=0.0, atol=0.0, maxiter=1),
    )
    x1_cg, _, info_cg = solver_cg.update(x0, solver_cg.init(x0))
    r0 = -b
    g0 = A.T @ r0
    d_cg = x1_cg / info_cg.stepsize
    np.testing.assert_allclose(
        info_cg.directional_derivative, jnp.dot(g0, d_cg), rtol=1e-5
    )


def test_non_descent_positive_or_zero_slope_and_gtol():
    # At x = 0, r(x) = [x^2 + 1] has J = [[0]], g = [0], d = [0], slope = 0
    def zero_grad_res(x):
        return x**2 + 1.0

    x0 = jnp.array([0.0], dtype=jnp.float32)
    solver = Newton(zero_grad_res, linear_solver=SVD())
    x1, _, info = solver.update(x0, solver.init(x0))
    assert not bool(info.accepted)
    assert int(info.num_linesearch_steps) == 0
    assert int(info.step_status) == int(NewtonStatus.NON_DESCENT)
    np.testing.assert_allclose(x1, x0)

    res_no_gtol = solver.solve(x0, gtol=0.0, max_steps=5)
    assert int(res_no_gtol.status) == int(NewtonStatus.NON_DESCENT)

    res_gtol = solver.solve(x0, gtol=1e-4, max_steps=5)
    assert int(res_gtol.status) == int(NewtonStatus.CONVERGED)


def test_failure_statuses_nonfinite_linear_solve_and_stagnation():
    # 1. Nonfinite base point
    solver = Newton(lambda x: x - 1.0)
    nan_x0 = jnp.array([jnp.nan], dtype=jnp.float32)
    res_nan = solver.solve(nan_x0)
    assert int(res_nan.status) == int(NewtonStatus.NONFINITE)
    assert int(res_nan.steps) == 0

    _, _, info_nan = solver.update(nan_x0, solver.init(nan_x0))
    assert int(info_nan.step_status) == int(NewtonStatus.NONFINITE)

    # 2. Singular Cholesky / QR factorization -> LINEAR_SOLVE_FAILED
    def singular_res(x):
        return jnp.array([x[0] + x[1] - 1.0, 0.0], dtype=jnp.float32)

    x0 = jnp.zeros(2, dtype=jnp.float32)
    s_chol = Newton(singular_res, linear_solver=Cholesky())
    res_chol = s_chol.solve(x0)
    assert int(res_chol.status) == int(NewtonStatus.LINEAR_SOLVE_FAILED)
    np.testing.assert_allclose(res_chol.x, x0)

    def dup_res(x):
        return jnp.array([x[0] + x[1] - 1.0, x[0] + x[1] - 1.0], dtype=jnp.float32)

    s_qr = Newton(dup_res, linear_solver=QR())
    res_qr = s_qr.solve(x0)
    assert int(res_qr.status) == int(NewtonStatus.LINEAR_SOLVE_FAILED)

    # 3. Representational stagnation: huge x0 with tiny step d so
    # x0 + alpha * d == x0 in float32.
    def tiny_step_res(x):
        return jnp.array([1e8 * (x[0] - 1.0) + 1.0], dtype=jnp.float32)

    x_stag = jnp.array([1.0], dtype=jnp.float32)
    s_stag = Newton(tiny_step_res, max_stepsize=1e-9)
    x1_stag, _, info_stag = s_stag.update(x_stag, s_stag.init(x_stag))
    assert not bool(info_stag.accepted)
    assert int(info_stag.step_status) == int(NewtonStatus.STAGNATION)
    np.testing.assert_allclose(x1_stag, x_stag)


def test_maxls_one_and_exact_last_trial_acceptance():
    def res_fn(x):
        return jnp.log(x)

    x0 = jnp.array([10.0], dtype=jnp.float32)
    # With maxls=1, trial 1 at alpha=1.0 is NaN and fails immediately
    s1 = Newton(res_fn, maxls=1)
    _, _, info1 = s1.update(x0, s1.init(x0))
    assert not bool(info1.accepted)
    assert int(info1.num_linesearch_steps) == 1
    np.testing.assert_allclose(info1.trial_stepsize, 1.0)
    assert int(info1.step_status) == int(NewtonStatus.LINE_SEARCH_FAILED)

    # With maxls=5, accepts on exactly the 5th (last allowed) trial
    s5 = Newton(res_fn, maxls=5, decrease_factor=0.8)
    _, _, info5 = s5.update(x0, s5.init(x0))
    assert bool(info5.accepted)
    assert int(info5.num_linesearch_steps) == 5
    np.testing.assert_allclose(info5.stepsize, 0.8**4, rtol=1e-6)


def test_tiny_nonzero_residual_underflow_not_treated_as_exact_root():
    # In float32, r = 1e-20 has r**2 == 0.0, but r != 0 entrywise
    def underflow_res(x):
        return x * 1e-20

    x0 = jnp.array([1.0], dtype=jnp.float32)
    solver = Newton(underflow_res)
    info0 = solver._initial_info(x0, solver.init(x0), None, None)
    assert float(info0.loss) == 0.0
    assert int(info0.step_status) == int(NewtonStatus.RUNNING)


def test_stepsize_resets_to_one_and_quadratic_local_convergence():
    # Start where step 1 backtracks (r(x) = log(x) from x = 10 takes alpha = 0.4096),
    # then subsequent steps reset alpha to 1.0 and converge rapidly to x = 1.0.
    solver = Newton(lambda x: jnp.log(x), c1=1e-4, decrease_factor=0.8, maxls=30)
    state = solver.init(jnp.array([10.0], dtype=jnp.float32))
    x = jnp.array([10.0], dtype=jnp.float32)

    x, state, info1 = solver.update(x, state)
    assert int(info1.num_linesearch_steps) == 5
    np.testing.assert_allclose(info1.stepsize, 0.4096, rtol=1e-5)

    x, state, info2 = solver.update(x, state)
    assert int(info2.num_linesearch_steps) == 1
    np.testing.assert_allclose(info2.stepsize, 1.0, rtol=1e-6)


def test_vmap_mixed_lane_linesearch_lengths():
    def residual_fn(x):
        return jnp.log(x)

    # x0 = 1.5 accepts full step (1 trial); x0 = 10.0 accepts on trial 5
    xs = jnp.array([[1.5], [10.0]], dtype=jnp.float32)
    solver = Newton(residual_fn, decrease_factor=0.8, maxls=30)
    state = solver.init(xs[0])

    x_out, _, info_out = jax.vmap(lambda x: solver.update(x, state))(xs)
    np.testing.assert_array_equal(info_out.num_linesearch_steps, np.array([1, 5]))
    np.testing.assert_allclose(info_out.stepsize, np.array([1.0, 0.8**4]), rtol=1e-5)
    assert bool(jnp.all(info_out.accepted))

    # Compare each lane against unbatched update
    for i in range(2):
        xi, _, info_i = solver.update(xs[i], state)
        np.testing.assert_allclose(x_out[i], xi, rtol=1e-6)
        np.testing.assert_allclose(info_out.stepsize[i], info_i.stepsize, rtol=1e-6)


def test_linesearch_jaxpr_has_no_factorizations_or_derivatives():
    def residual_fn(x):
        return jnp.sin(x[:3]) + x[3:6] - 1.0

    x0 = jnp.ones(6, dtype=jnp.float32)
    for ls in (Cholesky(), QR(), SVD()):
        solver = Newton(residual_fn, linear_solver=ls)
        state0 = solver.init(x0)
        jaxpr = jax.make_jaxpr(lambda x, s, solver=solver: solver.update(x, s))(
            x0, state0
        )

        # Find the while_loop equation inside the cond computation
        while_eqns = []

        def sub_jaxprs(params):
            out = []
            for v in params.values():
                if hasattr(v, "jaxpr"):
                    out.append(v.jaxpr)
                elif isinstance(v, (tuple, list)):
                    for item in v:
                        if hasattr(item, "jaxpr"):
                            out.append(item.jaxpr)
            return out

        def collect_while(jpr, acc=while_eqns):
            for eqn in jpr.eqns:
                if eqn.primitive.name == "while":
                    acc.append(eqn)
                for sub_jpr in sub_jaxprs(eqn.params):
                    collect_while(sub_jpr, acc)

        collect_while(jaxpr.jaxpr)
        assert len(while_eqns) == 1
        body_jaxpr = while_eqns[0].params["body_jaxpr"].jaxpr
        forbidden = {
            "qr",
            "cholesky",
            "svd",
            "lu",
            "triangular_solve",
            "custom_linear_solve",
        }

        def collect_primitives(jpr):
            names = set()
            for eqn in jpr.eqns:
                names.add(eqn.primitive.name)
                for sub_jpr in sub_jaxprs(eqn.params):
                    names |= collect_primitives(sub_jpr)
            return names

        body_prims = collect_primitives(body_jaxpr)
        assert forbidden.isdisjoint(body_prims)
