import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import pytest
from jax.flatten_util import ravel_pytree

from nlls_gram import (
    LevenbergMarquardt,
    LMSolveAction,
    LMState,
    LMStatus,
    Metric,
    blockdiag_metric,
    identity_preconditioner,
    matern_state_space,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_from_quasiseparable,
    metric_from_shifted_matvec,
    metric_from_state_space,
    metric_from_tridiagonal_precision,
    metric_with_compute_dtype,
    nystrom_preconditioner,
    pad_dual_preconditioner,
    repeated_blockdiag_metric,
    sherman_morrison_preconditioner,
    woodbury_preconditioner,
)


def residual_fn(x, args, p):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


REGRESSION_ATOL = 5e-5
REGRESSION_RTOL = 1e-5


def _solve_2x2(matrix, rhs):
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    return jnp.stack(
        (
            (matrix[1, 1] * rhs[0] - matrix[0, 1] * rhs[1]) / determinant,
            (-matrix[1, 0] * rhs[0] + matrix[0, 0] * rhs[1]) / determinant,
        )
    )


def test_recovers_known_parameters_with_jitted_step():
    a_true, b_true = 2.0, -1.0
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = a_true * jnp.exp(b_true * ts)

    x = {"a": 1.0, "b": 0.0}
    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    lm_state = solver.init(x, (ts, ys))

    @jax.jit
    def train_step(x, lm_state, args):
        return solver.update(x, lm_state, args)

    info = None
    for _ in range(50):
        x, lm_state, info = train_step(x, lm_state, (ts, ys))

    assert float(info.loss) < 1e-8
    assert jnp.allclose(x["a"], a_true, atol=1e-4)
    assert jnp.allclose(x["b"], b_true, atol=1e-4)


def test_default_metric_matches_explicit_identity_metric_solve():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    default_solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    identity_solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        metric=Metric(solve=lambda x: x, norm=jnp.linalg.norm),
    )

    default_x, default_state, default_info = default_solver.update(
        x, default_solver.init(x, (ts, ys)), (ts, ys)
    )
    identity_x, identity_state, identity_info = identity_solver.update(
        x, identity_solver.init(x, (ts, ys)), (ts, ys)
    )

    assert jnp.allclose(default_x["a"], identity_x["a"])
    assert jnp.allclose(default_x["b"], identity_x["b"])
    assert jnp.allclose(default_state.damping, identity_state.damping)
    assert jnp.allclose(default_info.loss, identity_info.loss)


def test_flat_array_x():
    def residual(theta, args, p):
        ts, ys = args
        return theta[0] * jnp.exp(theta[1] * ts) - ys

    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)

    x = jnp.array([1.0, 0.0])
    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init(x, (ts, ys))

    for _ in range(50):
        x, lm_state, _ = solver.update(x, lm_state, (ts, ys))

    assert x.shape == (2,)
    assert jnp.allclose(x, jnp.array([2.0, -1.0]), atol=1e-4)


def test_linear_problem_matches_closed_form_solution():
    def residual(x, args, p):
        ts, ys = args
        return x["a"] * ts - ys

    ts = jnp.array([1.0, 2.0, 3.0])
    ys = 2.0 * ts
    x = {"a": 0.0}
    init_damping = 1e-4

    solver = LevenbergMarquardt(residual, init_damping=init_damping)
    new_x, _, info = solver.update(x, solver.init(x, (ts, ys)), (ts, ys))
    expected_a = jnp.sum(ts * ys) / (jnp.sum(ts**2) + init_damping)
    expected_loss = jnp.sum((expected_a * ts - ys) ** 2)

    assert bool(info.accepted)
    assert jnp.allclose(new_x["a"], expected_a, atol=1e-6)
    assert jnp.allclose(info.loss, expected_loss, atol=1e-10)
    assert float(info.loss_old) == pytest.approx(56.0)
    assert jnp.allclose(info.loss_candidate, expected_loss, atol=1e-10)
    assert float(info.damping_factor) == pytest.approx(0.5)
    assert not bool(info.used_geodesic)
    assert float(info.acceleration_ratio) == pytest.approx(0.0)


def test_update_calls_residual_and_uses_values():
    calls = {"count": 0}

    def residual(x, args, p):
        calls["count"] += 1
        ts, ys = args
        return x["a"] * ts - ys

    ts = jnp.array([1.0, 2.0, 3.0])
    ys = 2.0 * ts
    x = {"a": 0.0}

    solver = LevenbergMarquardt(residual, init_damping=1e-4)
    new_x, _, info = solver.update(x, solver.init(x, (ts, ys)), (ts, ys))

    assert calls["count"] >= 2
    assert bool(info.accepted)
    assert jnp.allclose(new_x["a"], 2.0, atol=1e-4)


def test_defaults_enable_geodesic_and_jacobian_cache():
    solver = LevenbergMarquardt(lambda x: x)
    assert solver.geodesic_acceleration
    assert solver.cache_jacobian


def test_geodesic_off_uses_minimal_residual_evaluations():
    calls = {"count": 0}

    def residual(x, args, p):
        calls["count"] += 1
        ts, ys = args
        return x["a"] * ts - ys

    ts = jnp.array([1.0, 2.0, 3.0])
    ys = 2.0 * ts
    x = {"a": 0.0}

    solver = LevenbergMarquardt(
        residual, init_damping=1e-4, geodesic_acceleration=False, cache_jacobian=False
    )
    lm_state = solver.init(x, (ts, ys))
    calls["count"] = 0
    solver.update(x, lm_state, (ts, ys))

    assert calls["count"] == 2


def test_rejected_step_leaves_x_unchanged():
    ts = jnp.linspace(0.0, 2.0, 5)
    ys = jnp.ones_like(ts)
    x = {"a": 1.0, "b": 0.0}

    solver = LevenbergMarquardt(residual_fn)
    lm_state = solver.init(x, (ts, ys))
    new_x, new_lm_state, info = solver.update(x, lm_state, (ts, ys))

    assert not bool(info.accepted)
    assert jnp.allclose(new_x["a"], x["a"])
    assert jnp.allclose(new_x["b"], x["b"])
    assert float(new_lm_state.damping) > float(lm_state.damping)
    assert float(info.loss) == pytest.approx(0.0)
    assert float(info.loss_old) == pytest.approx(0.0)
    assert float(info.loss_candidate) == pytest.approx(0.0)
    assert float(info.damping_factor) == pytest.approx(4.0)
    assert not bool(info.used_geodesic)
    assert float(info.acceleration_ratio) == pytest.approx(0.0)


def test_unknown_linear_solver_raises():
    with pytest.raises(ValueError, match="unknown linear_solver"):
        LevenbergMarquardt(residual_fn, linear_solver="svd")


def test_init_damping_must_be_positive():
    with pytest.raises(ValueError, match="init_damping must be positive"):
        LevenbergMarquardt(residual_fn, init_damping=0.0)


def test_damping_update_factors_must_be_positive():
    with pytest.raises(ValueError, match="damping_decrease must be positive"):
        LevenbergMarquardt(residual_fn, damping_decrease=0.0)
    with pytest.raises(ValueError, match="damping_increase must be positive"):
        LevenbergMarquardt(residual_fn, damping_increase=0.0)


def test_iterative_options_must_be_valid():
    with pytest.raises(ValueError, match="iterative_tol must be nonnegative"):
        LevenbergMarquardt(residual_fn, linear_solver="cg", iterative_tol=-1.0)
    with pytest.raises(ValueError, match="iterative_atol must be nonnegative"):
        LevenbergMarquardt(residual_fn, linear_solver="cg", iterative_atol=-1.0)
    with pytest.raises(ValueError, match="iterative_maxiter must be positive or None"):
        LevenbergMarquardt(residual_fn, linear_solver="cg", iterative_maxiter=0)
    with pytest.raises(ValueError, match="iterative_maxiter must be set"):
        LevenbergMarquardt(
            residual_fn,
            linear_solver="cg",
            dual_preconditioner=identity_preconditioner(),
            implicit_preconditioner=identity_preconditioner(),
            iterative_tol=0.0,
            iterative_atol=0.0,
            iterative_maxiter=None,
        )


def test_implicit_solver_options_must_be_valid():
    with pytest.raises(ValueError, match="unknown implicit_solver"):
        LevenbergMarquardt(residual_fn, implicit_solver="lu")
    with pytest.raises(ValueError, match="implicit_tol must be nonnegative"):
        LevenbergMarquardt(residual_fn, implicit_tol=-1.0)
    with pytest.raises(ValueError, match="implicit_atol must be nonnegative"):
        LevenbergMarquardt(residual_fn, implicit_atol=-1.0)
    with pytest.raises(ValueError, match="implicit_maxiter must be positive or None"):
        LevenbergMarquardt(residual_fn, implicit_maxiter=0)
    with pytest.raises(ValueError, match="implicit_penalty must be nonnegative"):
        LevenbergMarquardt(residual_fn, implicit_penalty=-1e-8)
    with pytest.raises(ValueError, match="implicit_maxiter must be set"):
        LevenbergMarquardt(
            residual_fn,
            implicit_tol=0.0,
            implicit_atol=0.0,
            implicit_maxiter=None,
        )
    with pytest.raises(ValueError, match="implicit_preconditioner"):
        LevenbergMarquardt(residual_fn, implicit_preconditioner=lambda v: v)
    with pytest.raises(ValueError, match="implicit_preconditioner"):
        LevenbergMarquardt(
            residual_fn,
            linear_solver="qr",
            implicit_solver="auto",
            implicit_preconditioner=lambda v: v,
        )

    auto_cg = LevenbergMarquardt(
        residual_fn,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=lambda v: v,
    )
    explicit_cg = LevenbergMarquardt(
        residual_fn,
        implicit_solver="cg",
        implicit_preconditioner=lambda v: v,
    )
    assert auto_cg.implicit_solver == "auto"
    assert explicit_cg.implicit_solver == "cg"


def test_dual_solve_dtype_validation():
    with pytest.raises(ValueError, match="None or jnp.float64"):
        LevenbergMarquardt(residual_fn, dual_solve_dtype=jnp.float32)
    with pytest.raises(ValueError, match="dense cholesky paths"):
        LevenbergMarquardt(
            residual_fn,
            linear_solver="cg",
            iterative_tol=1e-7,
            iterative_maxiter=20,
            dual_preconditioner=identity_preconditioner(),
            implicit_preconditioner=identity_preconditioner(),
            dual_solve_dtype=jnp.float64,
        )
    # x64 is disabled in this test process, so float64 is unavailable and
    # requesting it raises rather than silently downcasting.
    with pytest.raises(ValueError, match="requires x64"):
        LevenbergMarquardt(residual_fn, dual_solve_dtype=jnp.float64)


def test_default_float32_x_keeps_float32_outputs():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    new_x, new_lm_state, info = solver.update(x, solver.init(x, (ts, ys)), (ts, ys))

    assert new_x["a"].dtype == jnp.float32
    assert new_x["b"].dtype == jnp.float32
    assert new_lm_state.damping.dtype == jnp.float32
    assert info.loss.dtype == jnp.float32
    assert info.loss_old.dtype == jnp.float32
    assert info.loss_candidate.dtype == jnp.float32
    assert info.damping.dtype == jnp.float32
    assert info.damping_factor.dtype == jnp.float32
    assert info.acceleration_ratio.dtype == jnp.float32
    assert info.grad_norm.dtype == jnp.float32
    assert info.step_norm.dtype == jnp.float32


def test_max_damping_below_init_damping_raises():
    with pytest.raises(ValueError, match="max_damping must be at least"):
        LevenbergMarquardt(residual_fn, init_damping=1e-2, max_damping=1e-3)


def test_metric_requirements_per_linear_solver():
    with pytest.raises(ValueError, match="metric.solve"):
        LevenbergMarquardt(residual_fn, metric=Metric(norm=jnp.linalg.norm))
    for linear_solver in ("qr", "augmented_qr"):
        with pytest.raises(ValueError, match="metric.inv_sqrt"):
            LevenbergMarquardt(
                residual_fn,
                linear_solver=linear_solver,
                metric=Metric(solve=lambda x: x),
            )
    with pytest.raises(ValueError, match="metric.norm"):
        LevenbergMarquardt(
            residual_fn,
            geodesic_acceleration=True,
            metric=Metric(solve=lambda x: x),
        )


def test_cg_step_matches_cholesky_identity_step():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    cholesky_solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )

    cholesky_x, cholesky_state, cholesky_info = cholesky_solver.update(
        x, cholesky_solver.init(x, (ts, ys)), (ts, ys)
    )
    cg_x, cg_state, cg_info = cg_solver.update(x, cg_solver.init(x, (ts, ys)), (ts, ys))

    assert bool(cg_info.accepted) == bool(cholesky_info.accepted)
    assert not bool(cg_info.used_geodesic)
    assert jnp.allclose(cg_x["a"], cholesky_x["a"], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(cg_x["b"], cholesky_x["b"], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(cg_state.damping, cholesky_state.damping)
    assert jnp.allclose(
        cg_info.loss,
        cholesky_info.loss,
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert cg_x["a"].dtype == jnp.float32
    assert cg_info.loss.dtype == jnp.float32


@pytest.mark.parametrize("linear_solver", ["qr", "augmented_qr"])
def test_qr_steps_match_cholesky_identity_step(linear_solver):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    cholesky_solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
    )
    qr_solver = LevenbergMarquardt(
        residual_fn, init_damping=1e-2, linear_solver=linear_solver
    )

    cholesky_x, cholesky_state, cholesky_info = cholesky_solver.update(
        x, cholesky_solver.init(x, (ts, ys)), (ts, ys)
    )
    qr_x, qr_state, qr_info = qr_solver.update(x, qr_solver.init(x, (ts, ys)), (ts, ys))

    assert bool(qr_info.accepted) == bool(cholesky_info.accepted)
    assert not bool(qr_info.used_geodesic)
    assert jnp.allclose(
        qr_x["a"],
        cholesky_x["a"],
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert jnp.allclose(
        qr_x["b"],
        cholesky_x["b"],
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert jnp.allclose(qr_state.damping, cholesky_state.damping)
    assert jnp.allclose(
        qr_info.loss,
        cholesky_info.loss,
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert qr_x["a"].dtype == jnp.float32
    assert qr_info.loss.dtype == jnp.float32


@pytest.mark.parametrize("linear_solver", ["qr", "augmented_qr"])
def test_qr_steps_match_closed_form_underdetermined_damped_solution(linear_solver):
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 2.0]])
    target = jnp.array([1.0, -2.0])
    theta0 = jnp.zeros(matrix.shape[1])
    init_damping = 0.1

    solver = LevenbergMarquardt(
        residual,
        init_damping=init_damping,
        linear_solver=linear_solver,
    )
    theta, lm_state, info = solver.update(
        theta0, solver.init(theta0, (matrix, target)), (matrix, target)
    )

    expected_step = jnp.linalg.solve(
        matrix.T @ matrix + init_damping * jnp.eye(matrix.shape[1]),
        matrix.T @ target,
    )
    expected_loss = jnp.sum((matrix @ expected_step - target) ** 2)

    assert bool(info.accepted)
    assert not bool(info.used_geodesic)
    assert jnp.allclose(theta, expected_step, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(info.loss, expected_loss, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(lm_state.damping, init_damping * 0.5)
    assert theta.dtype == jnp.float32
    assert info.loss.dtype == jnp.float32


@pytest.mark.parametrize(
    "matrix",
    [
        jnp.array([[1.0, 2.0], [0.5, -1.0]]),
        jnp.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]]),
    ],
)
def test_augmented_qr_matches_closed_form_for_square_and_overdetermined(matrix):
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    target = jnp.arange(1, matrix.shape[0] + 1, dtype=matrix.dtype)
    theta0 = jnp.zeros(matrix.shape[1])
    damping = 0.1
    solver = LevenbergMarquardt(
        residual,
        init_damping=damping,
        linear_solver="augmented_qr",
        geodesic_acceleration=False,
    )
    theta, _, info = solver.update(
        theta0,
        solver.init(theta0, (matrix, target)),
        (matrix, target),
    )
    expected = jnp.linalg.solve(
        matrix.T @ matrix + damping * jnp.eye(matrix.shape[1]),
        matrix.T @ target,
    )

    assert bool(info.accepted)
    assert jnp.allclose(theta, expected, rtol=1e-5, atol=1e-5)


def test_augmented_qr_rank_deficient_jacobian_has_finite_damped_step():
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 0.0], [2.0, 0.0]])
    target = jnp.array([1.0, 2.0])
    theta0 = jnp.zeros(2)
    damping = 1e-3
    solver = LevenbergMarquardt(
        residual,
        init_damping=damping,
        linear_solver="augmented_qr",
        geodesic_acceleration=False,
    )
    theta, _, info = solver.update(
        theta0,
        solver.init(theta0, (matrix, target)),
        (matrix, target),
    )
    expected = jnp.linalg.solve(
        matrix.T @ matrix + damping * jnp.eye(2), matrix.T @ target
    )

    assert bool(info.accepted)
    assert bool(jnp.all(jnp.isfinite(theta)))
    assert jnp.allclose(theta, expected, rtol=1e-5, atol=1e-5)


def test_qr_float32_handles_ill_conditioned_case_where_cholesky_fails():
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 0.0, 0.0, 0.0], [1.0, 1e-4, 0.0, 0.0]])
    theta_true = jnp.array([1.0, 1.0, 0.0, 0.0])
    target = matrix @ theta_true
    theta0 = jnp.zeros(matrix.shape[1])

    cholesky_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-12,
        linear_solver="cholesky",
    )
    qr_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-12,
        linear_solver="qr",
    )

    cholesky_theta, _, cholesky_info = cholesky_solver.update(
        theta0, cholesky_solver.init(theta0, (matrix, target)), (matrix, target)
    )
    qr_theta, _, qr_info = qr_solver.update(
        theta0, qr_solver.init(theta0, (matrix, target)), (matrix, target)
    )

    assert jnp.all(jnp.isfinite(qr_theta))
    assert jnp.isfinite(qr_info.loss_candidate)
    assert bool(qr_info.accepted)
    assert not bool(jnp.all(jnp.isfinite(cholesky_theta))) or not bool(
        jnp.isfinite(cholesky_info.loss_candidate)
    )
    assert qr_info.loss_candidate < 1e-12


def test_cg_update_jits():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )

    @jax.jit
    def train_step(x, lm_state):
        return solver.update(x, lm_state, (ts, ys))

    x, lm_state, info = train_step(x, solver.init(x, (ts, ys)))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(lm_state.damping)
    assert x["a"].dtype == jnp.float32
    assert x["b"].dtype == jnp.float32


@pytest.mark.parametrize("linear_solver", ["qr", "augmented_qr"])
def test_qr_updates_jit(linear_solver):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver=linear_solver,
    )

    @jax.jit
    def train_step(x, lm_state):
        return solver.update(x, lm_state, (ts, ys))

    x, lm_state, info = train_step(x, solver.init(x, (ts, ys)))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(lm_state.damping)
    assert x["a"].dtype == jnp.float32
    assert x["b"].dtype == jnp.float32


def test_geodesic_acceptance_ratio_zero_falls_back_to_velocity_step():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.0,
    )
    new_theta, _, info = solver.update(theta0, solver.init(theta0, target), target)

    jacobian = 2.0 * theta0[0]
    velocity = -jacobian * (theta0[0] ** 2 - target) / (jacobian**2 + 1e-6)
    expected_theta = theta0 + jnp.array([velocity])

    assert bool(info.accepted)
    assert not bool(info.used_geodesic)
    assert float(info.acceleration_ratio) > 0.0
    assert jnp.allclose(new_theta, expected_theta, rtol=1e-6, atol=1e-6)


def test_geodesic_acceleration_matches_closed_form_quadratic_step():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    new_theta, _, info = solver.update(theta0, solver.init(theta0, target), target)

    jacobian = 2.0 * theta0[0]
    linear_denominator = jacobian**2 + 1e-6
    velocity = -jacobian * (theta0[0] ** 2 - target) / linear_denominator
    f_vv = 2.0 * velocity**2
    acceleration = -jacobian * f_vv / linear_denominator
    expected_theta = theta0 + jnp.array([velocity + 0.5 * acceleration])
    expected_ratio = (
        2.0 * jnp.abs(acceleration) / (jnp.abs(velocity) + jnp.finfo(theta0.dtype).eps)
    )

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    assert new_theta.dtype == jnp.float32
    assert info.loss.dtype == jnp.float32
    assert info.loss_old.dtype == jnp.float32
    assert info.loss_candidate.dtype == jnp.float32
    assert info.damping.dtype == jnp.float32
    assert info.damping_factor.dtype == jnp.float32
    assert info.acceleration_ratio.dtype == jnp.float32
    assert jnp.allclose(new_theta, expected_theta, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(info.acceleration_ratio, expected_ratio, rtol=1e-6)


def test_cg_geodesic_acceleration_matches_cholesky():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    cholesky_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    cg_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=10,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    cholesky_theta, _, cholesky_info = cholesky_solver.update(
        theta0, cholesky_solver.init(theta0, target), target
    )
    cg_theta, _, cg_info = cg_solver.update(
        theta0, cg_solver.init(theta0, target), target
    )

    assert bool(cg_info.accepted)
    assert bool(cg_info.used_geodesic)
    assert jnp.allclose(cg_theta, cholesky_theta, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(
        cg_info.acceleration_ratio,
        cholesky_info.acceleration_ratio,
        rtol=1e-6,
        atol=1e-6,
    )


def test_qr_geodesic_acceleration_matches_cholesky():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    cholesky_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    qr_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="qr",
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    cholesky_theta, _, cholesky_info = cholesky_solver.update(
        theta0, cholesky_solver.init(theta0, target), target
    )
    qr_theta, _, qr_info = qr_solver.update(
        theta0, qr_solver.init(theta0, target), target
    )

    assert bool(qr_info.accepted)
    assert bool(qr_info.used_geodesic)
    assert jnp.allclose(qr_theta, cholesky_theta, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(
        qr_info.acceleration_ratio,
        cholesky_info.acceleration_ratio,
        rtol=1e-6,
        atol=1e-6,
    )


def test_geodesic_acceleration_jits():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    @jax.jit
    def step(theta, lm_state, target):
        return solver.update(theta, lm_state, target)

    theta, lm_state, info = step(
        jnp.array([1.9]), solver.init(jnp.array([1.9]), 4.0), 4.0
    )

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    assert jnp.isfinite(theta[0])
    assert jnp.isfinite(lm_state.damping)


def test_geodesic_acceleration_reduces_iterations_on_gsl_rosenbrock_example():
    def residual(theta, _, p):
        return jnp.array([100.0 * (theta[1] - theta[0] ** 2), 1.0 - theta[0]])

    def iterations_to_threshold(geodesic_acceleration):
        theta = jnp.array([-0.5, 1.75])
        solver = LevenbergMarquardt(
            residual,
            init_damping=1.0,
            geodesic_acceleration=geodesic_acceleration,
        )
        lm_state = solver.init(theta)
        used_geodesic = 0
        for iteration in range(1, 101):
            theta, lm_state, info = solver.update(theta, lm_state, None)
            used_geodesic += int(bool(info.used_geodesic))
            if float(info.loss) < 1e-12:
                return iteration, used_geodesic
        return 101, used_geodesic

    plain_iterations, plain_used_geodesic = iterations_to_threshold(False)
    geodesic_iterations, geodesic_used_geodesic = iterations_to_threshold(True)

    assert plain_used_geodesic == 0
    assert geodesic_used_geodesic > 0
    assert geodesic_iterations < plain_iterations / 2


def test_jitted_residual_with_jitted_geodesic_update():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = LevenbergMarquardt(
        jax.jit(residual_fn),
        init_damping=1e-2,
        geodesic_acceleration=True,
    )

    @jax.jit
    def train_step(x, lm_state):
        return solver.update(x, lm_state, (ts, ys))

    x, lm_state, info = train_step(x, solver.init(x, (ts, ys)))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(info.acceleration_ratio)
    assert jnp.isfinite(lm_state.damping)
    assert x["a"].dtype == jnp.float32
    assert x["b"].dtype == jnp.float32


def test_metric_from_cholesky_matches_dense_metric():
    L = jnp.array(
        [
            [2.0, 0.0, 0.0],
            [0.3, 1.5, 0.0],
            [-0.2, 0.4, 1.2],
        ]
    )
    metric_matrix = L @ L.T
    metric = metric_from_cholesky(L)
    vector = jnp.array([0.5, -1.0, 2.0])
    matrix = jnp.array([[1.0, -0.5], [0.2, 2.0], [-1.0, 0.25]])

    expected_solve_vector = jnp.linalg.solve(metric_matrix, vector)
    expected_solve_matrix = jnp.linalg.solve(metric_matrix, matrix)
    expected_norm = jnp.sqrt(vector @ metric_matrix @ vector)
    expected_inv_sqrt = jsp_linalg.solve_triangular(L.T, matrix, lower=False)
    expected_inv_sqrt_transpose = jsp_linalg.solve_triangular(L, matrix, lower=True)

    assert jnp.allclose(metric.solve(vector), expected_solve_vector)
    assert jnp.allclose(metric.solve(matrix), expected_solve_matrix)
    assert jnp.allclose(metric.norm(vector), expected_norm)
    assert jnp.allclose(metric.inv_sqrt(matrix), expected_inv_sqrt)
    assert jnp.allclose(
        metric.inv_sqrt_transpose(matrix),
        expected_inv_sqrt_transpose,
    )


def test_metric_selects_minimum_norm_interpolating_step():
    # r(theta) = theta_0 + theta_1 - 1 at theta = 0: every interpolating step
    # satisfies s_0 + s_1 = 1, and with tiny damping the update is the metric
    # Gauss-Newton step — the minimum-M-norm solution (docs worked example).
    def residual(theta, _, __):
        return jnp.array([theta[0] + theta[1] - 1.0])

    theta0 = jnp.zeros(2)

    identity_solver = LevenbergMarquardt(residual, init_damping=1e-9)
    x_identity, _, _ = identity_solver.update(theta0, identity_solver.init(theta0))
    assert jnp.allclose(x_identity, jnp.array([0.5, 0.5]), atol=1e-5)

    L = jnp.linalg.cholesky(jnp.diag(jnp.array([1.0, 4.0])))
    metric_solver = LevenbergMarquardt(
        residual, init_damping=1e-9, metric=metric_from_cholesky(L)
    )
    x_metric, _, _ = metric_solver.update(theta0, metric_solver.init(theta0))
    assert jnp.allclose(x_metric, jnp.array([0.8, 0.2]), atol=1e-5)


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "augmented_qr"])
def test_metric_step_matches_closed_form_solution(linear_solver):
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 2.0, -0.5], [0.3, -1.0, 1.5]])
    target = jnp.array([1.0, -2.0])
    theta0 = jnp.zeros(matrix.shape[1])
    init_damping = 0.1
    L = jnp.array(
        [
            [2.0, 0.0, 0.0],
            [0.3, 1.5, 0.0],
            [-0.2, 0.4, 1.2],
        ]
    )
    metric_matrix = L @ L.T
    solver_kwargs = {}
    if linear_solver == "cg":
        solver_kwargs = {
            "iterative_tol": 1e-7,
            "iterative_maxiter": 30,
            "dual_preconditioner": identity_preconditioner(),
            "implicit_preconditioner": identity_preconditioner(),
        }

    solver = LevenbergMarquardt(
        residual,
        init_damping=init_damping,
        linear_solver=linear_solver,
        metric=metric_from_cholesky(L),
        **solver_kwargs,
    )
    theta, lm_state, info = solver.update(
        theta0, solver.init(theta0, (matrix, target)), (matrix, target)
    )

    expected_step = jnp.linalg.solve(
        matrix.T @ matrix + init_damping * metric_matrix,
        matrix.T @ target,
    )
    expected_loss = jnp.sum((matrix @ expected_step - target) ** 2)

    assert bool(info.accepted)
    assert not bool(info.used_geodesic)
    assert jnp.allclose(theta, expected_step, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(info.loss, expected_loss, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(lm_state.damping, init_damping * 0.5)


def test_geodesic_step_matches_regression_values():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=10.0,
    )

    new_x, lm_state, info = solver.update(x, solver.init(x, (ts, ys)), (ts, ys))

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    assert jnp.allclose(
        new_x["a"],
        jnp.asarray(1.9073810577392578),
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert jnp.allclose(
        new_x["b"],
        jnp.asarray(-0.9168586730957031),
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert float(info.loss) == pytest.approx(0.029626082628965378, abs=REGRESSION_ATOL)
    assert float(info.loss_old) == pytest.approx(5.599210739135742, abs=REGRESSION_ATOL)
    assert float(info.loss_candidate) == pytest.approx(
        0.029626082628965378, abs=REGRESSION_ATOL
    )
    assert float(lm_state.damping) == pytest.approx(0.004999999888241291)
    assert float(info.damping_factor) == pytest.approx(0.5)
    assert float(info.acceleration_ratio) == pytest.approx(
        0.8667416572570801, abs=REGRESSION_ATOL
    )


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "augmented_qr"])
def test_metric_geodesic_acceleration_ratio_uses_metric_norm(linear_solver):
    def residual(theta, _, p):
        return jnp.array(
            [
                theta[0] ** 2 + theta[1] - 1.0,
                theta[0] + theta[1] ** 2 - 1.0,
            ]
        )

    theta0 = jnp.array([0.4, 1.2])
    init_damping = 0.1
    L = jnp.array([[3.0, 0.0], [0.2, 0.5]])
    metric_matrix = L @ L.T
    metric = metric_from_cholesky(L)
    solver_kwargs = {}
    if linear_solver == "cg":
        solver_kwargs = {
            "iterative_tol": 1e-7,
            "iterative_maxiter": 30,
            "dual_preconditioner": identity_preconditioner(),
            "implicit_preconditioner": identity_preconditioner(),
        }
    solver = LevenbergMarquardt(
        residual,
        init_damping=init_damping,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=100.0,
        metric=metric,
        **solver_kwargs,
    )

    _, _, info = solver.update(theta0, solver.init(theta0, None), None)

    resid = residual(theta0, None, None)
    jacobian = jnp.array([[2.0 * theta0[0], 1.0], [1.0, 2.0 * theta0[1]]])
    normal_matrix = jacobian.T @ jacobian + init_damping * metric_matrix
    velocity = -jnp.linalg.solve(normal_matrix, jacobian.T @ resid)
    f_vv = jnp.array([2.0 * velocity[0] ** 2, 2.0 * velocity[1] ** 2])
    acceleration = -jnp.linalg.solve(normal_matrix, jacobian.T @ f_vv)
    metric_ratio = (
        2.0
        * metric.norm(acceleration)
        / (metric.norm(velocity) + jnp.finfo(theta0.dtype).eps)
    )
    euclidean_ratio = (
        2.0
        * jnp.linalg.norm(acceleration)
        / (jnp.linalg.norm(velocity) + jnp.finfo(theta0.dtype).eps)
    )

    assert jnp.allclose(info.acceleration_ratio, metric_ratio, rtol=1e-5, atol=1e-5)
    assert not jnp.allclose(metric_ratio, euclidean_ratio, rtol=1e-3, atol=1e-3)


def test_custom_metric_update_jits_and_matches_closed_form_solution():
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 2.0, -0.5], [0.3, -1.0, 1.5]])
    target = jnp.array([1.0, -2.0])
    theta0 = jnp.zeros(matrix.shape[1])
    L = jnp.array(
        [
            [2.0, 0.0, 0.0],
            [0.3, 1.5, 0.0],
            [-0.2, 0.4, 1.2],
        ]
    )
    metric_matrix = L @ L.T
    init_damping = 0.1
    solver = LevenbergMarquardt(
        residual,
        init_damping=init_damping,
        metric=metric_from_cholesky(L),
    )

    @jax.jit
    def train_step(theta, lm_state):
        return solver.update(theta, lm_state, (matrix, target))

    theta, lm_state, info = train_step(theta0, solver.init(theta0, (matrix, target)))
    expected_step = jnp.linalg.solve(
        matrix.T @ matrix + init_damping * metric_matrix,
        matrix.T @ target,
    )

    assert bool(info.accepted)
    assert jnp.allclose(theta, expected_step, rtol=1e-5, atol=1e-5)
    assert jnp.isfinite(lm_state.damping)
    assert jnp.isfinite(info.loss)


def test_init_lm_state_matches_update_signature():
    # init() and update() must produce the same jit signature for `damping`
    # (strongly typed, matching dtype), or the second step recompiles.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)

    lm_state0 = solver.init(x, (ts, ys))
    _, lm_state1, _ = solver.update(x, lm_state0, (ts, ys))
    d0, d1 = lm_state0.damping, lm_state1.damping
    assert (d0.dtype, d0.weak_type, d0.shape) == (d1.dtype, d1.weak_type, d1.shape)
    assert d0.weak_type is False
    assert d0.dtype == jnp.result_type(float)


def test_solve_converges_with_args_and_p_jit_modes():
    def residual(theta, args, p):
        return theta - (args + p)

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    theta0 = jnp.array([0.0])
    args = jnp.array([1.25])
    p = jnp.array([0.75])

    jit_result = solver.solve(theta0, args, p=p, max_steps=40, atol=1e-6)
    python_result = solver.solve(theta0, args, p=p, max_steps=40, atol=1e-6, jit=False)

    assert int(jit_result.status) == LMStatus.CONVERGED
    assert int(python_result.status) == LMStatus.CONVERGED
    assert jnp.allclose(jit_result.x, jnp.array([2.0]), atol=1e-5)
    assert jnp.allclose(jit_result.x, python_result.x, atol=1e-6)
    assert jnp.allclose(jit_result.p, p)
    assert jit_result.steps <= 40


def test_solve_reports_max_steps_without_atol_convergence():
    def residual(theta, args, p):
        return theta - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(jnp.array([0.0]), jnp.array([1.0]), max_steps=3, atol=0.0)

    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.steps) == 3
    assert jnp.isfinite(result.info.loss)


def test_solve_callback_can_abort_on_nonfinite_candidate():
    def residual(theta, _, __):
        return jnp.where(theta[0] > 0.0, theta + 1.0, jnp.asarray([jnp.nan]))

    def callback(ctx):
        nonfinite = ~jnp.isfinite(ctx.info.loss_candidate)
        return LMSolveAction(stop=nonfinite, status=LMStatus.NONFINITE)

    solver = LevenbergMarquardt(residual, init_damping=1e-3)
    theta0 = jnp.array([0.1])
    result = solver.solve(theta0, max_steps=5, callback=callback)

    assert int(result.status) == LMStatus.NONFINITE
    assert int(result.steps) == 1
    assert jnp.allclose(result.x, theta0)
    assert jnp.isfinite(result.info.loss)
    assert not jnp.isfinite(result.info.loss_candidate)


def test_solve_callback_updates_args_and_user_state():
    def residual(theta, args, _):
        return theta - args

    def callback(ctx):
        next_args = jnp.where(ctx.step == 1, jnp.asarray([2.0]), ctx.args)
        return LMSolveAction(args=next_args, user_state=ctx.user_state + ctx.info.loss)

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        max_steps=2,
        callback=callback,
        user_state=jnp.asarray(0.0),
    )

    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.steps) == 2
    assert jnp.allclose(result.args, jnp.array([2.0]))
    assert result.x[0] > 1.0
    assert result.user_state > 0.0


def test_solve_save_steps_matches_manual_update_loop():
    a_true, b_true = 2.0, -1.0
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = a_true * jnp.exp(b_true * ts)
    x0 = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)

    result = solver.solve(x0, (ts, ys), max_steps=8, save_steps=True)
    assert int(result.steps) == 8
    assert result.aux_history is None
    # args never change: every args_history row repeats the initial pytree.
    assert result.args_history[0].shape == (9, *ts.shape)
    assert jnp.all(result.args_history[0] == ts)
    assert jnp.all(result.args_history[1] == ys)

    x, lm_state = x0, solver.init(x0, (ts, ys))
    iterates = [x0]
    for _ in range(8):
        x, lm_state, _ = solver.update(x, lm_state, (ts, ys))
        iterates.append(x)
    for s, expected in enumerate(iterates):
        assert jnp.allclose(result.x_history["a"][s], expected["a"])
        assert jnp.allclose(result.x_history["b"][s], expected["b"])
    assert jnp.allclose(result.x_history["a"][-1], result.x["a"])


@pytest.mark.parametrize("jit", [True, False])
def test_solve_save_steps_pads_rows_beyond_steps(jit):
    def residual(theta, args):
        return theta - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    theta0 = jnp.array([0.0])
    result = solver.solve(
        theta0, jnp.array([1.0]), max_steps=20, atol=1e-6, save_steps=True, jit=jit
    )

    steps = int(result.steps)
    assert int(result.status) == LMStatus.CONVERGED
    assert 0 < steps < 20
    assert result.x_history.shape == (21, 1)
    assert jnp.array_equal(result.x_history[0], theta0)
    assert jnp.allclose(result.x_history[steps], result.x)
    assert jnp.all(result.x_history[steps + 1 :] == 0.0)
    assert result.args_history.shape == (21, 1)
    assert jnp.all(result.args_history[: steps + 1] == 1.0)
    assert jnp.all(result.args_history[steps + 1 :] == 0.0)


def test_solve_save_steps_aux_history_aligns_with_iterates():
    def residual(theta, args):
        return theta - args, {"total": jnp.sum(theta), "sq": theta**2}

    solver = LevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)
    result = solver.solve(
        jnp.array([0.0, 0.5]),
        jnp.array([1.0, -1.0]),
        max_steps=15,
        atol=1e-6,
        save_steps=True,
    )

    steps = int(result.steps)
    for s in range(steps + 1):
        x_s = result.x_history[s]
        assert jnp.allclose(result.aux_history["total"][s], jnp.sum(x_s))
        assert jnp.allclose(result.aux_history["sq"][s], x_s**2)
    assert jnp.allclose(result.aux_history["total"][steps], result.aux["total"])


def test_solve_save_steps_composes_with_callback_and_user_state():
    def residual(theta, args):
        return theta - args

    def callback(ctx):
        replaced = jnp.where(ctx.step == 1, jnp.asarray([5.0]), ctx.x)
        return LMSolveAction(x=replaced, user_state=ctx.user_state + 1.0)

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        max_steps=3,
        callback=callback,
        user_state=jnp.asarray(0.0),
        save_steps=True,
    )

    # x_history records the kept post-action iterate.
    assert jnp.allclose(result.x_history[1], jnp.array([5.0]))
    assert int(result.user_state) == int(result.steps)


@pytest.mark.parametrize("jit", [True, False])
def test_solve_save_steps_records_args_replacement(jit):
    def residual(theta, args):
        return theta - args

    def callback(ctx):
        replaced = jnp.where(ctx.step == 2, jnp.asarray([3.0]), ctx.args)
        return LMSolveAction(args=replaced)

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        max_steps=4,
        callback=callback,
        save_steps=True,
        jit=jit,
    )

    assert int(result.steps) == 4
    # Row s holds the kept post-action args after step s: the original args
    # through step 1, the replacement from step 2 onward.
    assert jnp.array_equal(result.args_history[0], jnp.array([1.0]))
    assert jnp.array_equal(result.args_history[1], jnp.array([1.0]))
    for s in range(2, 5):
        assert jnp.array_equal(result.args_history[s], jnp.array([3.0]))
    assert jnp.array_equal(result.args_history[-1], result.args)


def test_solve_save_steps_args_history_none_without_args():
    def residual(theta):
        return theta - 1.0

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(jnp.array([0.0]), max_steps=5, atol=1e-6, save_steps=True)

    assert result.args_history is None
    assert result.x_history.shape == (6, 1)


def test_vmap_over_solve_save_steps_keeps_per_lane_padding():
    def residual(theta, _, p):
        return theta - p["target"]

    def callback(ctx):
        return LMSolveAction(stop=ctx.step >= ctx.p["stop_after"])

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    x0s = jnp.zeros((3, 1))
    p = {
        "target": jnp.arange(1.0, 4.0)[:, None],
        "stop_after": jnp.arange(1, 4, dtype=jnp.int32),
    }

    def solve_one(x0, p_one):
        return solver.solve(
            x0, p=p_one, max_steps=6, callback=callback, save_steps=True
        )

    batched = jax.vmap(solve_one)(x0s, p)
    assert batched.x_history.shape == (3, 7, 1)
    for i in range(3):
        single = solve_one(x0s[i], jax.tree.map(lambda leaf, i=i: leaf[i], p))
        assert int(batched.steps[i]) == int(single.steps)
        # Lanes that stop early must keep their rows frozen while other lanes
        # continue, so the padding beyond each lane's steps stays zero.
        assert jnp.allclose(batched.x_history[i], single.x_history, atol=1e-6)
        assert jnp.all(batched.x_history[i, int(batched.steps[i]) + 1 :] == 0.0)


def test_solve_save_steps_histories_are_differentiation_inert():
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), {"s": jnp.sum(theta)}

    solver = LevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)
    theta0 = jnp.zeros(2)

    def solved(p):
        result = solver.solve(theta0, p=p, max_steps=80, atol=1e-8, save_steps=True)
        return result.x, result.x_history, result.aux_history

    p, p_dot = jnp.asarray(3.0), jnp.asarray(0.7)
    (_, x_hist, _), (x_dot, x_hist_dot, aux_hist_dot) = jax.jvp(solved, (p,), (p_dot,))
    assert jnp.allclose(x_dot, jnp.array([p_dot / 5.0, 2.0 * p_dot / 5.0]), atol=1e-6)
    assert jnp.all(x_hist_dot == 0.0)
    assert jnp.all(aux_hist_dot["s"] == 0.0)

    # Reverse mode: cotangents on the histories pull back to zero, while the
    # solution cotangent still flows through the implicit rule.
    grad_p = jax.grad(lambda p_: jnp.sum(solved(p_)[0]) + jnp.sum(solved(p_)[1]))(p)
    assert jnp.allclose(grad_p, 3.0 / 5.0, atol=1e-6)


def test_vmap_over_solve_callback_stops_per_lane():
    def residual(theta, _, p):
        return theta - p["target"]

    def callback(ctx):
        stop = ctx.step >= ctx.p["stop_after"]
        return LMSolveAction(stop=stop, status=100 + ctx.p["stop_after"])

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    x0s = jnp.zeros((4, 1))
    p = {
        "target": jnp.arange(1.0, 5.0)[:, None],
        "stop_after": jnp.arange(1, 5, dtype=jnp.int32),
    }

    def solve_one(x0, p_one):
        return solver.solve(x0, p=p_one, max_steps=8, atol=0.0, callback=callback)

    batched = jax.vmap(solve_one)(x0s, p)
    loop = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[
            solve_one(x0s[i], jax.tree.map(lambda leaf, i=i: leaf[i], p))
            for i in range(x0s.shape[0])
        ],
    )

    assert jnp.array_equal(batched.steps, p["stop_after"])
    assert jnp.array_equal(batched.steps, loop.steps)
    assert jnp.array_equal(batched.status, loop.status)
    assert jnp.allclose(batched.x, loop.x, atol=1e-6)


@pytest.mark.parametrize("save_steps", [False, True])
def test_vmap_over_solve_heterogeneous_convergence_matches_sequential(save_steps):
    def residual(theta, _, p):
        return theta - p["target"]

    solver = LevenbergMarquardt(residual, init_damping=1.0)
    x0s = jnp.zeros((2, 1))
    p = {"target": jnp.array([[1e-4], [3.0]])}
    atol = 1e-5

    def solve_one(x0, p_one):
        return solver.solve(x0, p=p_one, max_steps=60, atol=atol, save_steps=save_steps)

    batched = jax.vmap(solve_one)(x0s, p)
    for i in range(2):
        single = solve_one(x0s[i], jax.tree.map(lambda leaf, i=i: leaf[i], p))
        assert int(batched.steps[i]) == int(single.steps)
        assert int(batched.status[i]) == int(single.status) == LMStatus.CONVERGED
        assert jnp.allclose(batched.x[i], single.x, atol=1e-6)
        assert jnp.linalg.norm(batched.x[i] - p["target"][i]) < atol
        if save_steps:
            assert jnp.allclose(batched.x_history[i], single.x_history, atol=1e-6)
            # A lane whose convergence fired must never keep writing rows while
            # the slower lane continues, so its padding stays exactly zero.
            assert jnp.all(batched.x_history[i, int(batched.steps[i]) + 1 :] == 0.0)
    assert int(batched.steps[0]) < int(batched.steps[1])


@pytest.mark.parametrize("save_steps", [False, True])
def test_vmap_over_solve_epoch_conditional_callback_early_stop(save_steps):
    steps_per_epoch = 3
    init_damping = 1.0

    def residual(theta, _, p):
        return theta - p["target"]

    solver = LevenbergMarquardt(residual, init_damping=init_damping)

    # Epoch-boundary early stopping: the expensive check runs only every
    # steps_per_epoch steps behind a lax.cond, which under vmap lowers to a
    # select that evaluates both branches for every lane; per-lane stops must
    # still bind exactly as in the sequential solves.
    def callback(ctx):
        def epoch_boundary(_):
            r = ctx.x - ctx.p["target"]
            stop = jnp.sum(r * r) < ctx.p["threshold"]
            status = jnp.where(stop, LMStatus.CONVERGED, LMStatus.RUNNING)
            return (
                stop,
                status.astype(jnp.int32),
                jnp.asarray(init_damping, ctx.lm_state.damping.dtype),
            )

        def mid_epoch(_):
            # RUNNING is the no-op status: solve only reads it when stop fires.
            return (
                jnp.asarray(False),
                jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
                ctx.lm_state.damping,
            )

        stop, status, damping = jax.lax.cond(
            ctx.step % steps_per_epoch == 0, epoch_boundary, mid_epoch, None
        )
        return LMSolveAction(
            stop=stop,
            status=status,
            lm_state=dataclasses.replace(ctx.lm_state, damping=damping),
        )

    x0s = jnp.zeros((2, 1))
    p = {
        "target": jnp.full((2, 1), 2.0),
        "threshold": jnp.array([1e-2, 1e-7]),
    }

    def solve_one(x0, p_one):
        return solver.solve(
            x0,
            p=p_one,
            max_steps=30,
            atol=0.0,
            callback=callback,
            save_steps=save_steps,
        )

    batched = jax.vmap(solve_one)(x0s, p)
    for i in range(2):
        single = solve_one(x0s[i], jax.tree.map(lambda leaf, i=i: leaf[i], p))
        assert int(batched.steps[i]) == int(single.steps)
        assert int(batched.status[i]) == int(single.status) == LMStatus.CONVERGED
        assert int(batched.steps[i]) % steps_per_epoch == 0
        assert jnp.allclose(batched.x[i], single.x, atol=1e-6)
        assert jnp.sum((batched.x[i] - p["target"][i]) ** 2) < p["threshold"][i]
        if save_steps:
            assert jnp.allclose(batched.x_history[i], single.x_history, atol=1e-6)
            assert jnp.all(batched.x_history[i, int(batched.steps[i]) + 1 :] == 0.0)
    assert int(batched.steps[0]) < int(batched.steps[1])


def test_vmap_over_solve_tolerances():
    def residual(theta, _, p):
        return theta - p

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    x0 = jnp.zeros(1)
    target = jnp.array([2.0])
    atols = jnp.array([5e-2, 1e-5])

    def solve_one(atol):
        return solver.solve(x0, p=target, max_steps=100, atol=atol)

    batched = jax.vmap(solve_one)(atols)
    for i in range(2):
        single = solve_one(atols[i])
        assert int(batched.steps[i]) == int(single.steps)
        assert int(batched.status[i]) == int(single.status) == LMStatus.CONVERGED
        assert jnp.allclose(batched.x[i], single.x, atol=1e-6)
    # The tight-tolerance lane keeps iterating after the loose lane stopped.
    assert int(batched.steps[1]) > int(batched.steps[0])


def test_dense_implicit_penalty_handles_singular_dual_from_redundant_rows():
    # Two identical residual rows: the undamped implicit dual J J' is singular
    # (rank 1) everywhere, but the system is consistent and x*(p) is smooth
    # with minimum-norm derivative d x* / d target = w / ||w||^2 from x0 = 0.
    # The default eps * trace ridge resolves it to that min-norm tangent.
    w = jnp.array([1.0, 2.0, 3.0])

    def duplicated_rows_residual(x, args, p):
        row = jnp.dot(w, x) - p["target"]
        return jnp.stack([row, row])

    x0 = jnp.zeros(3)
    p = {"target": 1.0}
    expected = jnp.sum(w) / jnp.dot(w, w)

    def sum_x_star(solver, p):
        return jnp.sum(solver.solve(x0, p=p, max_steps=50).x)

    regularized = LevenbergMarquardt(duplicated_rows_residual)
    vjp_grad = jax.jacobian(lambda p: sum_x_star(regularized, p))(p)["target"]
    assert jnp.allclose(vjp_grad, expected, rtol=1e-4)
    _, jvp_grad = jax.jvp(
        lambda t: sum_x_star(regularized, {"target": t}), (1.0,), (1.0,)
    )
    assert jnp.allclose(jvp_grad, expected, rtol=1e-4)

    # An explicit (larger) penalty is plumbed through and still resolves the
    # singular dual; the bias grows with the penalty but stays O(penalty * m).
    blunt = LevenbergMarquardt(duplicated_rows_residual, implicit_penalty=1e-4)
    blunt_grad = jax.jacobian(lambda p: sum_x_star(blunt, p))(p)["target"]
    assert jnp.isfinite(blunt_grad)
    assert jnp.allclose(blunt_grad, expected, rtol=1e-3)


@pytest.mark.parametrize("jit", [True, False])
def test_solve_implicit_jvp_and_vjp_wrt_p_match_underdetermined_root(jit):
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    theta0 = jnp.zeros(2)

    def solved_x(p):
        return solver.solve(theta0, p=p, max_steps=80, atol=1e-6, jit=jit).x

    p = jnp.asarray(3.0)
    p_dot = jnp.asarray(0.7)
    x, x_dot = jax.jvp(solved_x, (p,), (p_dot,))
    expected_x = jnp.array([3.0 / 5.0, 6.0 / 5.0])
    expected_x_dot = jnp.array([p_dot / 5.0, 2.0 * p_dot / 5.0])

    _, pullback = jax.vjp(solved_x, p)
    (p_cotangent,) = pullback(jnp.array([3.0, 4.0]))
    expected_p_cotangent = (3.0 + 2.0 * 4.0) / 5.0

    assert jnp.allclose(x, expected_x, atol=1e-5)
    assert jnp.allclose(x_dot, expected_x_dot, atol=1e-6)
    assert jnp.allclose(p_cotangent, expected_p_cotangent, atol=1e-6)


def test_implicit_cg_jvp_and_vjp_match_cholesky_with_metric():
    with jax.default_device(jax.devices("cpu")[0]):
        matrix = jnp.array([[1.0, 2.0, -0.5, 0.3], [0.2, -1.0, 1.5, 2.0]])
        target_matrix = jnp.array([[1.0, -0.5, 0.7], [0.3, 1.2, -1.0]])
        L = jnp.array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.3, 1.5, 0.0, 0.0],
                [-0.2, 0.4, 1.2, 0.0],
                [0.1, -0.1, 0.2, 1.7],
            ]
        )

        def residual(theta, _, p):
            return matrix @ theta - target_matrix @ p

        common = dict(
            init_damping=1e-2,
            linear_solver="cg",
            iterative_tol=1e-7,
            iterative_maxiter=30,
            dual_preconditioner=identity_preconditioner(),
            metric=metric_from_cholesky(L),
            geodesic_acceleration=False,
        )
        dense_implicit = LevenbergMarquardt(
            residual, implicit_solver="cholesky", **common
        )
        cg_implicit = LevenbergMarquardt(
            residual,
            implicit_solver="cg",
            implicit_tol=1e-7,
            implicit_preconditioner=identity_preconditioner(),
            **common,
        )
        theta0 = jnp.zeros(matrix.shape[1])

        def solved_x(solver, p):
            return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).x

        p = jnp.array([1.0, -0.5, 0.25])
        p_dot = jnp.array([0.2, -0.1, 0.3])
        _, dense_dot = jax.jvp(lambda q: solved_x(dense_implicit, q), (p,), (p_dot,))
        _, cg_dot = jax.jvp(lambda q: solved_x(cg_implicit, q), (p,), (p_dot,))

        theta_bar = jnp.array([0.4, -0.2, 0.7, 0.1])
        _, dense_pullback = jax.vjp(lambda q: solved_x(dense_implicit, q), p)
        _, cg_pullback = jax.vjp(lambda q: solved_x(cg_implicit, q), p)
        (dense_bar,) = dense_pullback(theta_bar)
        (cg_bar,) = cg_pullback(theta_bar)

        assert jnp.allclose(cg_dot, dense_dot, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(cg_bar, dense_bar, rtol=1e-5, atol=1e-5)


def test_implicit_cg_sign_and_transpose_match_closed_form():
    matrix = jnp.array([[1.0, 2.0, -0.5, 0.3], [0.2, -1.0, 1.5, 2.0]])
    target_matrix = jnp.array([[1.0, -0.5, 0.7], [0.3, 1.2, -1.0]])
    metric_weights = jnp.array([2.0, 0.5, 1.5, 3.0])
    metric_inverse = jnp.diag(1.0 / metric_weights)

    def residual(theta, _, p):
        return matrix @ theta - target_matrix @ p

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=30,
        implicit_solver="cg",
        implicit_tol=1e-7,
        metric=metric_from_diagonal(metric_weights),
        geodesic_acceleration=False,
    )
    theta0 = jnp.zeros(matrix.shape[1])

    def solved_x(p):
        return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).x

    p = jnp.array([0.7, -1.0, 0.4])
    p_dot = jnp.array([0.2, -0.1, 0.3])
    theta_bar = jnp.array([0.4, -0.2, 0.7, 0.1])
    gram = matrix @ metric_inverse @ matrix.T
    expected_dot = metric_inverse @ matrix.T @ _solve_2x2(gram, target_matrix @ p_dot)
    expected_bar = target_matrix.T @ _solve_2x2(
        gram, matrix @ metric_inverse @ theta_bar
    )

    _, theta_dot = jax.jvp(solved_x, (p,), (p_dot,))
    _, pullback = jax.vjp(solved_x, p)
    (p_bar,) = pullback(theta_bar)

    assert jnp.allclose(theta_dot, expected_dot, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(p_bar, expected_bar, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(theta_dot @ theta_bar, p_dot @ p_bar, atol=1e-5)


def test_implicit_cg_jvp_and_vjp_jit():
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
        implicit_solver="auto",
        implicit_tol=1e-7,
    )

    def solved_x(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x

    @jax.jit
    def jitted_jvp(p, p_dot):
        return jax.jvp(solved_x, (p,), (p_dot,))[1]

    @jax.jit
    def jitted_vjp(p, theta_bar):
        _, pullback = jax.vjp(solved_x, p)
        return pullback(theta_bar)[0]

    p = jnp.asarray(3.0)
    assert jnp.allclose(
        jitted_jvp(p, jnp.asarray(0.7)),
        jnp.array([0.7 / 5.0, 1.4 / 5.0]),
        atol=1e-6,
    )
    assert jnp.allclose(
        jitted_vjp(p, jnp.array([3.0, 4.0])),
        (3.0 + 2.0 * 4.0) / 5.0,
        atol=1e-6,
    )


def test_implicit_cg_jaxpr_does_not_materialize_dense_jacobian_transpose():
    n = 257
    grid = jnp.linspace(0.0, 1.0, n)
    row0 = jnp.sin(2.0 * jnp.pi * grid) + 1.5
    row1 = jnp.cos(3.0 * jnp.pi * grid) - 0.25
    row2 = grid + 0.1

    def residual(theta, _, p):
        return jnp.stack(
            (
                jnp.vdot(row0, theta) - p[0],
                jnp.vdot(row1, theta) - p[1],
                jnp.vdot(row2, theta) - p[2],
            )
        )

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-6,
        iterative_maxiter=5,
        implicit_solver="auto",
        implicit_tol=1e-6,
        implicit_maxiter=5,
        geodesic_acceleration=False,
    )

    def solved_x(p):
        return solver.solve(jnp.zeros(n), p=p, max_steps=1, atol=0.0).x

    jaxpr = str(
        jax.make_jaxpr(lambda p, p_dot: jax.jvp(solved_x, (p,), (p_dot,))[1])(
            jnp.array([1.0, -0.5, 0.25]),
            jnp.array([0.2, -0.1, 0.3]),
        )
    )

    assert f"f32[{n},3]" not in jaxpr
    assert f"f32[3,{n}]" not in jaxpr


def test_vmap_over_solve_matches_loop_and_implicit_ad():
    matrix = jnp.array([[1.0, 0.5, -0.2, 0.1], [0.3, -0.7, 0.4, 1.0]])
    gram = matrix @ matrix.T
    right_inverse = matrix.T @ jnp.linalg.inv(gram)

    def residual(theta, _, p):
        return matrix @ theta - p

    solver = LevenbergMarquardt(
        residual, init_damping=1e-2, geodesic_acceleration=False
    )
    ps = jnp.array([[1.0, -0.5], [0.25, 0.75], [-1.2, 0.2], [0.1, -1.4]])
    x0s = jnp.zeros((ps.shape[0], matrix.shape[1]))

    def solve_one(x0, p):
        return solver.solve(x0, p=p, max_steps=80, atol=1e-5)

    batched = jax.vmap(solve_one)(x0s, ps)
    loop = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[solve_one(x0s[i], ps[i]) for i in range(ps.shape[0])],
    )
    expected_x = jax.vmap(lambda p: right_inverse @ p)(ps)

    assert jnp.array_equal(batched.status, loop.status)
    assert jnp.array_equal(batched.steps, loop.steps)
    assert jnp.allclose(batched.x, loop.x, atol=1e-6)
    assert jnp.allclose(batched.x, expected_x, atol=1e-4)

    p_dot = jnp.array([[0.2, -0.1], [0.0, 0.3], [0.7, -0.4], [-0.2, 0.5]])

    def vmapped_x(p_batch):
        return jax.vmap(lambda p: solve_one(jnp.zeros(matrix.shape[1]), p).x)(p_batch)

    _, x_dot = jax.jvp(vmapped_x, (ps,), (p_dot,))
    expected_x_dot = jax.vmap(lambda dp: right_inverse @ dp)(p_dot)
    assert jnp.allclose(x_dot, expected_x_dot, atol=1e-5)

    cotangent = jnp.array(
        [
            [0.1, -0.2, 0.3, 0.0],
            [0.0, 0.4, -0.1, 0.2],
            [-0.3, 0.2, 0.1, 0.5],
            [0.7, -0.1, 0.0, -0.4],
        ]
    )
    _, pullback = jax.vjp(vmapped_x, ps)
    (p_bar,) = pullback(cotangent)
    expected_p_bar = jax.vmap(lambda c: jnp.linalg.solve(gram, matrix @ c))(cotangent)
    assert jnp.allclose(p_bar, expected_p_bar, atol=1e-5)


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr", "augmented_qr"])
def test_solve_implicit_jvp_wrt_p_uses_metric(linear_solver):
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    metric_matrix = jnp.array([[4.0, 0.0], [0.0, 1.0]])
    metric_inverse = jnp.linalg.inv(metric_matrix)
    jacobian = jnp.array([[1.0, 2.0]])
    full_metric = metric_from_cholesky(jnp.linalg.cholesky(metric_matrix))
    metric = (
        full_metric
        if linear_solver == "cholesky"
        else Metric(
            inv_sqrt=full_metric.inv_sqrt,
            inv_sqrt_transpose=full_metric.inv_sqrt_transpose,
        )
    )
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver=linear_solver,
        metric=metric,
        # The QR cases deliberately supply a square-root-only metric (no
        # norm), which the geodesic default would reject at construction.
        geodesic_acceleration=False,
    )

    def solved_x(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x

    p_dot = jnp.asarray(0.7)
    _, x_dot = jax.jvp(solved_x, (jnp.asarray(3.0),), (p_dot,))
    expected_x_dot = (
        metric_inverse
        @ jacobian.T
        @ jnp.linalg.solve(jacobian @ metric_inverse @ jacobian.T, jnp.array([p_dot]))
    ).ravel()

    assert jnp.allclose(x_dot, expected_x_dot, atol=1e-6)


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "augmented_qr"])
def test_grad_norm_and_step_norm_match_closed_form(linear_solver):
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 2.0]])
    target = jnp.array([1.0, -2.0])
    theta0 = jnp.zeros(matrix.shape[1])
    init_damping = 0.1
    solver_kwargs = {}
    if linear_solver == "cg":
        solver_kwargs = {
            "iterative_tol": 1e-7,
            "iterative_maxiter": 30,
            "dual_preconditioner": identity_preconditioner(),
            "implicit_preconditioner": identity_preconditioner(),
        }

    solver = LevenbergMarquardt(
        residual,
        init_damping=init_damping,
        linear_solver=linear_solver,
        **solver_kwargs,
    )
    _, _, info = solver.update(
        theta0, solver.init(theta0, (matrix, target)), (matrix, target)
    )

    expected_grad = matrix.T @ (matrix @ theta0 - target)
    expected_step = jnp.linalg.solve(
        matrix.T @ matrix + init_damping * jnp.eye(matrix.shape[1]),
        matrix.T @ target,
    )

    assert jnp.allclose(
        info.grad_norm, jnp.linalg.norm(expected_grad), rtol=1e-5, atol=1e-5
    )
    assert jnp.allclose(
        info.step_norm, jnp.linalg.norm(expected_step), rtol=1e-5, atol=1e-5
    )


def test_rejected_step_still_reports_step_norm():
    # The candidate step leaves theta = 0, producing a NaN residual, so every
    # step is rejected; the attempted step norm must still be reported.
    def residual(theta, _, __):
        return jnp.where(theta[0] == 0.0, theta + 1.0, jnp.full_like(theta, jnp.nan))

    solver = LevenbergMarquardt(residual, init_damping=1.0)
    _, _, info = solver.update(jnp.zeros(1), solver.init(jnp.zeros(1)))

    assert not bool(info.accepted)
    assert float(info.step_norm) == pytest.approx(0.5)
    assert float(info.grad_norm) == pytest.approx(1.0)


@pytest.mark.parametrize("jit", [True, False])
def test_solve_gtol_reports_converged(jit):
    def residual(theta, args, p):
        return theta - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]), jnp.array([1.0]), max_steps=50, gtol=1e-6, jit=jit
    )

    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.info.grad_norm) < 1e-6
    assert int(result.steps) < 50


@pytest.mark.parametrize("jit", [True, False])
def test_solve_xtol_reports_converged_on_accepted_step(jit):
    def residual(theta, args, p):
        return theta - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]), jnp.array([1.0]), max_steps=50, xtol=1e-6, jit=jit
    )

    assert int(result.status) == LMStatus.CONVERGED
    assert bool(result.info.accepted)
    assert float(result.info.step_norm) < 1e-6
    assert int(result.steps) < 50


def test_solve_xtol_ignores_rejected_steps():
    def residual(theta, _, __):
        return jnp.where(theta[0] == 0.0, theta + 1.0, jnp.full_like(theta, jnp.nan))

    solver = LevenbergMarquardt(residual, init_damping=1.0, max_damping=1e6)
    result = solver.solve(jnp.zeros(1), max_steps=30, xtol=10.0)

    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.steps) == 30
    assert not bool(result.info.accepted)


def test_max_damping_caps_growth_under_repeated_rejection():
    def residual(theta, _, __):
        return jnp.where(theta[0] == 0.0, theta + 1.0, jnp.full_like(theta, jnp.nan))

    capped = LevenbergMarquardt(residual, init_damping=1e-3, max_damping=1e4)
    result = capped.solve(jnp.zeros(1), max_steps=100)
    assert int(result.status) == LMStatus.MAX_STEPS
    assert float(result.lm_state.damping) == pytest.approx(1e4)

    uncapped = LevenbergMarquardt(residual, init_damping=1e-3)
    result = uncapped.solve(jnp.zeros(1), max_steps=100)
    assert not jnp.isfinite(result.lm_state.damping)


def test_solve_does_not_retrace_on_loop_control_changes():
    traces = {"count": 0}

    def residual(theta, args, p):
        traces["count"] += 1
        return theta - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2, cache_jacobian=False)
    solver.solve(jnp.array([0.0]), jnp.array([1.0]), max_steps=10, atol=1e-6)
    count_after_first = traces["count"]
    solver.solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        max_steps=25,
        atol=1e-8,
        gtol=1e-9,
        xtol=1e-9,
    )

    assert traces["count"] == count_after_first


def test_equal_settings_solvers_share_the_compiled_solve_loop():
    traces = {"count": 0}

    def residual(theta, args, p):
        traces["count"] += 1
        return theta - args

    def build(**overrides):
        settings = dict(init_damping=1e-2, cache_jacobian=False)
        settings.update(overrides)
        return LevenbergMarquardt(residual, **settings)

    a, b = build(), build()
    assert a == b
    assert hash(a) == hash(b)

    a.solve(jnp.array([0.0]), jnp.array([1.0]), max_steps=10, atol=1e-6)
    count_after_first = traces["count"]
    b.solve(jnp.array([0.0]), jnp.array([1.0]), max_steps=10, atol=1e-6)
    assert traces["count"] == count_after_first

    # Any static-setting change (or a different residual function) is a
    # different solver, so it cannot silently reuse the wrong compiled loop.
    assert a != build(init_damping=2e-2)
    assert a != build(geodesic_acceleration=False)
    assert a != LevenbergMarquardt(
        lambda theta, args, p: theta - args, init_damping=1e-2, cache_jacobian=False
    )


@pytest.mark.parametrize("cache_jacobian", [False, True])
def test_solve_callback_epoch_boundary_resamples_args_and_resets_damping(
    cache_jacobian,
):
    def residual(theta, args, _):
        return theta - args

    steps_per_epoch = 3

    def callback(ctx):
        boundary = ctx.step % steps_per_epoch == 0
        new_args = jnp.where(boundary, ctx.args + 1.0, ctx.args)
        new_lm_state = dataclasses.replace(
            ctx.lm_state,
            damping=jnp.where(
                boundary, ctx.initial_lm_state.damping, ctx.lm_state.damping
            ),
        )
        epochs = ctx.user_state + jnp.where(boundary, 1, 0)
        return LMSolveAction(args=new_args, lm_state=new_lm_state, user_state=epochs)

    solver = LevenbergMarquardt(
        residual, init_damping=1e-2, cache_jacobian=cache_jacobian
    )
    result = solver.solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        max_steps=7,
        callback=callback,
        user_state=jnp.asarray(0),
    )

    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.user_state) == 2
    assert jnp.allclose(result.args, jnp.array([3.0]))
    # Step 6 reset damping to init; the accepted step 7 halved it once.
    assert float(result.lm_state.damping) == pytest.approx(5e-3)


def test_callback_returning_args_invalidates_jacobian_cache():
    # theta stays at 0 (every candidate is NaN, so every step is rejected) and
    # the cache is therefore valid; when the callback swaps args at step 2,
    # step 3 must recompute the residual with the new args. A stale cache
    # would report loss_old = 1 (old args) instead of 4 (new args).
    def residual(theta, args, _):
        return jnp.where(theta[0] == 0.0, theta + args, jnp.full_like(theta, jnp.nan))

    def callback(ctx):
        new_args = jnp.where(ctx.step == 2, jnp.asarray([2.0]), ctx.args)
        return LMSolveAction(args=new_args)

    solver = LevenbergMarquardt(residual, init_damping=1.0, cache_jacobian=True)
    result = solver.solve(
        jnp.zeros(1), jnp.asarray([1.0]), max_steps=3, callback=callback
    )

    assert int(result.status) == LMStatus.MAX_STEPS
    assert float(result.info.loss_old) == pytest.approx(4.0)


def test_callback_returning_unchanged_args_keeps_jacobian_cache():
    # Every candidate is NaN, so every step is rejected and the cache stays
    # valid; a jit-style callback returning args with unchanged values must
    # not invalidate it.
    def residual(theta, args, _):
        return jnp.where(theta[0] == 0.0, theta + args, jnp.full_like(theta, jnp.nan))

    def callback(ctx):
        return LMSolveAction(args=ctx.args)

    solver = LevenbergMarquardt(residual, init_damping=1.0, cache_jacobian=True)
    result = solver.solve(
        jnp.zeros(1), jnp.asarray([1.0]), max_steps=3, callback=callback
    )

    assert int(result.status) == LMStatus.MAX_STEPS
    assert bool(result.lm_state.jacobian_valid)


@pytest.mark.parametrize("jit", [True, False])
def test_callback_changing_args_defers_stale_convergence(jit):
    # A callback that swaps args exactly when the old problem meets atol must
    # not let solve report CONVERGED for the swapped (x, args) pair; the
    # tolerances wait for a fresh update against the new args.
    def residual(theta, args, p):
        return theta - args

    def callback(ctx):
        swap = (jnp.sqrt(ctx.info.loss) < 1e-3) & (ctx.args[0] < 50.0)
        return LMSolveAction(args=jnp.where(swap, jnp.asarray([100.0]), ctx.args))

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.zeros(1),
        jnp.ones(1),
        max_steps=400,
        atol=1e-3,
        callback=callback,
        jit=jit,
    )

    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.args[0]) == pytest.approx(100.0)
    assert jnp.allclose(result.x, result.args, atol=1e-2)


@pytest.mark.parametrize("jit", [True, False])
def test_callback_echoing_nan_args_still_converges(jit):
    # An unchanged NaN sentinel in echoed args is not a change (equal_nan
    # comparison) and must not defer the tolerance checks.
    def residual(theta, _, __):
        return theta - 1.0

    def callback(ctx):
        return LMSolveAction(args=ctx.args)

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.zeros(1),
        jnp.asarray([jnp.nan]),
        max_steps=100,
        atol=1e-3,
        callback=callback,
        jit=jit,
    )

    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x, 1.0, atol=1e-2)


def test_callback_bare_lm_state_with_cache_raises_clear_error():
    def residual(theta, args, _):
        return theta - args

    def callback(ctx):
        return LMSolveAction(lm_state=LMState(ctx.lm_state.damping))

    solver = LevenbergMarquardt(residual, init_damping=1e-2, cache_jacobian=True)
    with pytest.raises(ValueError, match="Jacobian cache"):
        solver.solve(jnp.zeros(1), jnp.ones(1), max_steps=3, callback=callback)


def test_hyperparams_typing_and_solve_population():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)

    # init() stays lean for manual update() loops; solve() populates hyper.
    assert solver.init({"a": 1.0, "b": 0.0}, (ts, ys)).hyper is None
    hyper = solver.hyperparams(jnp.float32)
    assert hyper.damping_decrease.dtype == jnp.float32
    assert hyper.iterative_maxiter.dtype == jnp.int32
    assert hyper.max_damping is None
    result = solver.solve({"a": 1.0, "b": 0.0}, (ts, ys), max_steps=2)
    assert result.lm_state.hyper.damping_decrease.dtype == jnp.float32


def test_bare_lm_state_matches_hyper_lm_state_update():
    # hyper=None falls back to the constructor values, so both states must
    # produce the same step.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2, cache_jacobian=False)

    x_hyper, _, info_hyper = solver.update(x, solver.init(x, (ts, ys)), (ts, ys))
    x_bare, _, info_bare = solver.update(
        x, LMState(jnp.asarray(1e-2, dtype=jnp.float32)), (ts, ys)
    )

    assert jnp.allclose(x_hyper["a"], x_bare["a"])
    assert jnp.allclose(x_hyper["b"], x_bare["b"])
    assert jnp.allclose(info_hyper.loss, info_bare.loss)


@pytest.mark.parametrize("jit", [True, False])
def test_callback_grows_cg_budget_when_loss_small(jit):
    # The cookbook schedule: cheap CG steps far from the solution, accurate
    # ones near it, inside a single solve call.
    matrix = jnp.diag(jnp.logspace(0.0, 1.5, 8))
    target = jnp.linspace(1.0, 2.0, 8)

    def residual(theta, _, __):
        return matrix @ theta - target

    def grow_budget(ctx):
        grown = jnp.asarray(40, dtype=jnp.int32)
        new_maxiter = jnp.where(
            ctx.info.loss < 1.0, grown, ctx.lm_state.hyper.iterative_maxiter
        )
        new_hyper = dataclasses.replace(
            ctx.lm_state.hyper, iterative_maxiter=new_maxiter
        )
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, hyper=new_hyper)
        )

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-1,
        linear_solver="cg",
        iterative_maxiter=2,
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
    )
    theta0 = jnp.zeros(8)
    fixed = solver.solve(theta0, max_steps=60, jit=jit)
    scheduled = solver.solve(theta0, max_steps=60, callback=grow_budget, jit=jit)

    assert int(scheduled.lm_state.hyper.iterative_maxiter) == 40
    assert float(scheduled.info.loss) < 1e-2 * float(fixed.info.loss)


def test_callback_resets_max_damping_cap():
    # Every step rejects (NaN candidates), so damping grows; the callback
    # tightens the traced cap mid-solve.
    def residual(theta, args, _):
        return jnp.where(theta[0] == 0.0, theta + args, jnp.full_like(theta, jnp.nan))

    def cap_damping(ctx):
        new_hyper = dataclasses.replace(
            ctx.lm_state.hyper, max_damping=jnp.asarray(10.0, dtype=jnp.float32)
        )
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, hyper=new_hyper)
        )

    solver = LevenbergMarquardt(residual, init_damping=1.0, max_damping=1e6)
    result = solver.solve(
        jnp.zeros(1), jnp.asarray([1.0]), max_steps=20, callback=cap_damping
    )

    assert int(result.status) == LMStatus.MAX_STEPS
    assert float(result.lm_state.damping) <= 10.0


@pytest.mark.parametrize("jit", [True, False])
def test_callback_enabling_none_hyper_knob_raises_clear_error(jit):
    # max_damping=None is compiled out; flipping it on mid-solve must fail
    # identically with and without jit.
    def residual(theta, args, p):
        return theta - args

    def enable_cap(ctx):
        new_hyper = dataclasses.replace(
            ctx.lm_state.hyper, max_damping=jnp.asarray(10.0, dtype=jnp.float32)
        )
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, hyper=new_hyper)
        )

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    with pytest.raises(ValueError, match="enabled mid-solve"):
        solver.solve(
            jnp.zeros(1), jnp.ones(1), max_steps=3, callback=enable_cap, jit=jit
        )


def test_solve_callback_history_buffer_matches_update_loop():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    max_steps = 5

    def callback(ctx):
        history = {
            "loss": jax.lax.dynamic_update_slice(
                ctx.user_state["loss"], ctx.info.loss[None], (ctx.step - 1,)
            ),
            "damping": jax.lax.dynamic_update_slice(
                ctx.user_state["damping"], ctx.info.damping[None], (ctx.step - 1,)
            ),
        }
        return LMSolveAction(user_state=history)

    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    result = solver.solve(
        x,
        (ts, ys),
        max_steps=max_steps,
        callback=callback,
        user_state={
            "loss": jnp.zeros(max_steps),
            "damping": jnp.zeros(max_steps),
        },
    )

    loop_x, loop_lm_state = x, solver.init(x, (ts, ys))
    for i in range(max_steps):
        loop_x, loop_lm_state, info = solver.update(loop_x, loop_lm_state, (ts, ys))
        assert float(result.user_state["loss"][i]) == pytest.approx(
            float(info.loss), rel=1e-3, abs=1e-6
        )
        assert float(result.user_state["damping"][i]) == pytest.approx(
            float(loop_lm_state.damping), rel=1e-6
        )


@pytest.mark.parametrize("jit", [True, False])
def test_solve_callback_returning_none_leaves_loop_untouched(jit):
    # The cookbook logging pattern: observe the context, return None.
    def residual(theta, args, p):
        return theta - args

    def logging_callback(ctx):
        def log(_):
            jax.debug.print(
                "step {step}: loss={loss:.3e}", step=ctx.step, loss=ctx.info.loss
            )

        jax.lax.cond(ctx.step % 10 == 0, log, lambda _: None, operand=None)

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        max_steps=40,
        atol=1e-6,
        callback=logging_callback,
        jit=jit,
    )

    assert int(result.status) == LMStatus.CONVERGED


def test_one_arg_residual_closes_over_data():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)

    def residual(x):
        return x["a"] * jnp.exp(x["b"] * ts) - ys

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve({"a": 1.0, "b": 0.0}, max_steps=50, atol=1e-6)

    assert solver.residual_arity == 1
    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x["a"], 2.0, atol=1e-4)


def test_two_arg_residual_takes_args():
    def residual(x, args):
        ts, ys = args
        return x["a"] * jnp.exp(x["b"] * ts) - ys

    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    x0 = {"a": 1.0, "b": 0.0}
    x, lm_state, info = solver.update(x0, solver.init(x0, (ts, ys)), (ts, ys))

    assert solver.residual_arity == 2
    assert bool(info.accepted)


def test_residual_arity_matches_three_arg_solver():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    def one_arg(theta):
        return residual_fn(theta, (ts, ys), None)

    one_solver = LevenbergMarquardt(one_arg, init_damping=1e-2)
    three_solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    one_x, one_state, one_info = one_solver.update(x, one_solver.init(x))
    three_x, three_state, three_info = three_solver.update(
        x, three_solver.init(x, (ts, ys)), (ts, ys)
    )

    assert jnp.allclose(one_x["a"], three_x["a"])
    assert jnp.allclose(one_x["b"], three_x["b"])
    assert jnp.allclose(one_info.loss, three_info.loss)
    assert jnp.allclose(one_state.damping, three_state.damping)


def test_args_and_p_require_matching_residual_arity():
    def one_arg(theta):
        return theta

    def two_arg(theta, args):
        return theta - args

    one_solver = LevenbergMarquardt(one_arg)
    two_solver = LevenbergMarquardt(two_arg)
    theta0 = jnp.zeros(1)

    with pytest.raises(ValueError, match="takes only .x."):
        one_solver.update(theta0, one_solver.init(theta0), jnp.ones(1))
    with pytest.raises(ValueError, match="takes only .x."):
        one_solver.solve(theta0, jnp.ones(1))
    with pytest.raises(ValueError, match="takes no p argument"):
        two_solver.update(
            theta0, two_solver.init(theta0, jnp.ones(1)), jnp.ones(1), jnp.ones(1)
        )
    with pytest.raises(ValueError, match="takes no p argument"):
        two_solver.solve(theta0, jnp.ones(1), p=jnp.ones(1))


def test_zero_or_many_arg_residual_rejected_at_construction():
    with pytest.raises(ValueError, match="1 to 3 positional arguments"):
        LevenbergMarquardt(lambda: jnp.zeros(1))
    with pytest.raises(ValueError, match="1 to 3 positional arguments"):
        LevenbergMarquardt(lambda a, b, c, d: a)


def test_residual_with_default_args_counts_as_three_arg():
    def residual(theta, _=None, __=None):
        return theta - 1.0

    solver = LevenbergMarquardt(residual)
    assert solver.residual_arity == 3
    result = solver.solve(jnp.zeros(1), max_steps=30, atol=1e-6)
    assert int(result.status) == LMStatus.CONVERGED


def test_cache_jacobian_single_step_matches_fresh_solver_after_rejection():
    # After a rejection the cached solver reuses (resid, Jt); its next step
    # must match a fresh no-cache solver started at the identical (theta,
    # damping) up to float noise.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    args = (ts, ys)
    kw = {"init_damping": 1e-8, "damping_decrease": 0.01, "damping_increase": 100.0}
    kw["geodesic_acceleration"] = False  # float noise can flip its gate
    cached = LevenbergMarquardt(residual_fn, cache_jacobian=True, **kw)
    plain = LevenbergMarquardt(residual_fn, cache_jacobian=False, **kw)

    x = {"a": 1.0, "b": 3.0}
    lm_state = cached.init(x0=x, args=args)
    reuse_steps = 0
    for _ in range(12):
        x_prev, state_prev = x, lm_state
        x, lm_state, info = cached.update(x, lm_state, args)
        if bool(state_prev.jacobian_valid):
            reuse_steps += 1
            ref_x, _, ref_info = plain.update(x_prev, LMState(state_prev.damping), args)
            assert bool(ref_info.accepted) == bool(info.accepted)
            assert jnp.allclose(ref_x["a"], x["a"], rtol=1e-3, atol=1e-5)
            assert jnp.allclose(ref_x["b"], x["b"], rtol=1e-3, atol=1e-5)
    assert reuse_steps > 0


def test_cache_jacobian_solve_converges_and_matches_plain():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    cached = LevenbergMarquardt(residual_fn, init_damping=1e-2, cache_jacobian=True)
    plain = LevenbergMarquardt(residual_fn, init_damping=1e-2)

    cached_result = cached.solve(
        {"a": 1.0, "b": 0.0}, (ts, ys), max_steps=50, atol=1e-6
    )
    plain_result = plain.solve({"a": 1.0, "b": 0.0}, (ts, ys), max_steps=50, atol=1e-6)

    assert int(cached_result.status) == LMStatus.CONVERGED
    assert int(plain_result.status) == LMStatus.CONVERGED
    assert jnp.allclose(cached_result.x["a"], 2.0, atol=1e-4)
    assert jnp.allclose(cached_result.x["b"], -1.0, atol=1e-4)


def test_cache_jacobian_requires_sized_lm_state():
    solver = LevenbergMarquardt(residual_fn, cache_jacobian=True)
    with pytest.raises(ValueError, match="no Jacobian cache"):
        solver.update(
            {"a": 1.0, "b": 0.0},
            LMState(jnp.asarray(1e-3)),
            (jnp.ones(3), jnp.ones(3)),
        )


def test_cache_jacobian_is_inert_for_non_cholesky_solvers():
    solver = LevenbergMarquardt(
        residual_fn,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        cache_jacobian=True,
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )
    assert not solver.cache_jacobian
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    lm_state = solver.init(x, (ts, ys))
    assert lm_state.jacobian_valid is None
    _, _, info = solver.update(x, lm_state, (ts, ys))
    assert bool(info.accepted)


def test_init_infers_dtype_from_residual():
    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    lm_state = solver.init(x0={"a": 1.0, "b": 0.0}, args=(ts, ys))
    assert lm_state.damping.dtype == jnp.float32


def test_cache_jacobian_off_leaves_no_trace_in_the_jaxpr():
    # With caching (and geodesic) off there is no lax.cond in update at all,
    # so the disabled flag provably adds zero compiled overhead.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

    plain = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    cached = LevenbergMarquardt(residual_fn, init_damping=1e-2, cache_jacobian=True)
    plain_jaxpr = str(
        jax.make_jaxpr(lambda p, s: plain.update(p, s, (ts, ys)))(
            x, plain.init(x, (ts, ys))
        )
    )
    cached_jaxpr = str(
        jax.make_jaxpr(lambda p, s: cached.update(p, s, (ts, ys)))(
            x, cached.init(x0=x, args=(ts, ys))
        )
    )

    assert "cond" not in plain_jaxpr
    assert "cond" in cached_jaxpr


def aux_residual_fn(x, args):
    ts, ys = args
    r = x["a"] * jnp.exp(x["b"] * ts) - ys
    return r, {"mean_abs": jnp.mean(jnp.abs(r)), "max_abs": jnp.max(jnp.abs(r))}


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "augmented_qr"])
def test_has_aux_reports_aux_at_pre_step_x(linear_solver):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    solver_kwargs = {}
    if linear_solver == "cg":
        solver_kwargs = {
            "iterative_tol": 1e-7,
            "iterative_maxiter": 20,
            "dual_preconditioner": identity_preconditioner(),
            "implicit_preconditioner": identity_preconditioner(),
        }

    solver = LevenbergMarquardt(
        aux_residual_fn,
        init_damping=1e-2,
        has_aux=True,
        linear_solver=linear_solver,
        **solver_kwargs,
    )
    lm_state = solver.init(x, (ts, ys))
    _, _, info = solver.update(x, lm_state, (ts, ys))

    r, expected = aux_residual_fn(x, (ts, ys))
    assert float(info.aux["mean_abs"]) == pytest.approx(
        float(expected["mean_abs"]), rel=1e-6
    )
    assert float(info.aux["max_abs"]) == pytest.approx(
        float(expected["max_abs"]), rel=1e-6
    )
    assert float(info.loss) < float(info.loss_old)


def test_has_aux_flows_to_solve_callback_and_result():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)

    def callback(ctx):
        return LMSolveAction(stop=ctx.info.aux["max_abs"] < 1e-4)

    solver = LevenbergMarquardt(
        aux_residual_fn, init_damping=1e-2, has_aux=True, geodesic_acceleration=True
    )
    result = solver.solve(
        {"a": 1.0, "b": 0.0}, (ts, ys), max_steps=50, callback=callback
    )

    assert int(result.status) == LMStatus.CALLBACK_STOP
    assert float(result.info.aux["max_abs"]) < 1e-4


def test_has_aux_works_with_jacobian_cache():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    solver = LevenbergMarquardt(
        aux_residual_fn, init_damping=1e-2, has_aux=True, cache_jacobian=True
    )
    result = solver.solve(x, (ts, ys), max_steps=50, atol=1e-6)

    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.info.aux["mean_abs"]) < 1e-4


def test_has_aux_off_keeps_info_aux_none():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    _, _, info = solver.update(x, solver.init(x, (ts, ys)), (ts, ys))
    assert info.aux is None


@pytest.mark.parametrize("jit", [True, False])
def test_solve_returns_final_aux_at_returned_x(jit):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    solver = LevenbergMarquardt(aux_residual_fn, init_damping=1e-2, has_aux=True)
    result = solver.solve(
        {"a": 1.0, "b": 0.0}, (ts, ys), max_steps=50, atol=1e-6, jit=jit
    )

    assert int(result.status) == LMStatus.CONVERGED
    _, expected = aux_residual_fn(result.x, (ts, ys))
    assert float(result.aux["max_abs"]) == pytest.approx(
        float(expected["max_abs"]), rel=1e-3, abs=1e-7
    )
    # the final aux is at the solution, tighter than the pre-step info.aux
    assert float(result.aux["max_abs"]) <= float(result.info.aux["max_abs"])


def test_solve_returns_final_aux_without_convergence():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    solver = LevenbergMarquardt(aux_residual_fn, init_damping=1e-2, has_aux=True)
    result = solver.solve({"a": 1.0, "b": 0.0}, (ts, ys), max_steps=3)

    assert int(result.status) == LMStatus.MAX_STEPS
    _, expected = aux_residual_fn(result.x, (ts, ys))
    assert float(result.aux["max_abs"]) == pytest.approx(
        float(expected["max_abs"]), rel=1e-3, abs=1e-7
    )


def test_solve_result_aux_none_without_has_aux():
    def residual(theta, args):
        return theta - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(jnp.zeros(1), jnp.ones(1), max_steps=20, atol=1e-6)
    assert result.aux is None


def test_aux_non_numeric_leaf_raises():
    def residual(theta):
        r = theta**2 - 2.0
        return r, {"note": "not an array", "val": jnp.max(jnp.abs(r))}

    solver = LevenbergMarquardt(residual, has_aux=True)
    x0 = jnp.array([1.0])
    with pytest.raises(TypeError, match="aux leaves"):
        solver.solve(x0, atol=1e-5)
    with pytest.raises(TypeError, match="aux leaves"):
        solver.init(x0)


def test_solve_implicit_jvp_works_with_has_aux():
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), {"level": theta[0]}

    solver = LevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)

    def solved(p):
        result = solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6)
        return result.x, result.aux["level"]

    (x, level), (x_dot, level_dot) = jax.jvp(
        solved, (jnp.asarray(3.0),), (jnp.asarray(0.7),)
    )
    assert jnp.allclose(x_dot, jnp.array([0.7 / 5.0, 1.4 / 5.0]), atol=1e-6)
    # aux level = theta*[0] = p/5, so its tangent is p_dot/5.
    assert jnp.allclose(level_dot, 0.7 / 5.0, atol=1e-6)


@pytest.mark.parametrize("jit", [True, False])
def test_solve_implicit_jvp_of_aux_wrt_p(jit):
    # theta* = (p/5, 2p/5), so aux m = theta0*theta1 + p^2 has
    # dm/dp = theta1/5 + 2*theta0/5 + 2p = 4p/25 + 2p through both the
    # solution path and the direct p path. The int32 aux leaf must not
    # break the rule.
    def residual(theta, _, p):
        aux = {
            "m": theta[0] * theta[1] + p**2,
            "count": jnp.asarray(1, dtype=jnp.int32),
        }
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), aux

    solver = LevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)

    def solved_aux_m(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6, jit=jit).aux[
            "m"
        ]

    p, p_dot = jnp.asarray(3.0), jnp.asarray(0.7)
    _, m_dot = jax.jvp(solved_aux_m, (p,), (p_dot,))
    expected = (4.0 * p / 25.0 + 2.0 * p) * p_dot
    assert jnp.allclose(m_dot, expected, atol=1e-6)

    h = 1e-3
    fd = (solved_aux_m(p + h) - solved_aux_m(p - h)) / (2.0 * h)
    assert jnp.allclose(m_dot, fd * p_dot, atol=1e-3)

    # The int32 aux leaf gets a float0 tangent, not a zero float.
    def solved_aux(q):
        return solver.solve(jnp.zeros(2), p=q, max_steps=80, atol=1e-6, jit=jit).aux

    _, aux_dot = jax.jvp(solved_aux, (p,), (p_dot,))
    assert aux_dot["count"].dtype == jax.dtypes.float0
    assert jnp.allclose(aux_dot["m"], expected, atol=1e-6)


def test_solve_implicit_jvp_of_aux_with_pytree_p():
    # theta* = (s/5, 2s/5) with s = p["scale"]; aux vec = (theta0*s, theta1)
    # has d/ds = (theta0 + s/5, 2/5) = (2s/5, 2/5). The int32 p leaf takes a
    # float0 tangent and the int32 aux leaf produces one.
    def residual(theta, _, p):
        aux = {
            "vec": jnp.array([theta[0] * p["scale"], theta[1]]),
            "count": jnp.asarray(1, dtype=jnp.int32),
        }
        return jnp.array([theta[0] + 2.0 * theta[1] - p["scale"]]), aux

    solver = LevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)
    p = {"scale": jnp.asarray(3.0), "flag": jnp.asarray(2, dtype=jnp.int32)}
    p_dot = {
        "scale": jnp.asarray(1.0),
        "flag": jnp.zeros((), dtype=jax.dtypes.float0),
    }

    def solved(q):
        result = solver.solve(jnp.zeros(2), p=q, max_steps=80, atol=1e-6)
        return result.x, result.aux

    (x, aux), (x_dot, aux_dot) = jax.jvp(solved, (p,), (p_dot,))
    s = p["scale"]
    assert jnp.allclose(x_dot, jnp.array([1.0 / 5.0, 2.0 / 5.0]), atol=1e-6)
    assert jnp.allclose(
        aux_dot["vec"], jnp.array([2.0 * s / 5.0, 2.0 / 5.0]), atol=1e-5
    )
    assert aux_dot["count"].dtype == jax.dtypes.float0

    # VJP: pull back a cotangent on the vector aux leaf; the int32 p leaf
    # receives a float0 cotangent.
    def solved_aux_vec(q):
        return solver.solve(jnp.zeros(2), p=q, max_steps=80, atol=1e-6).aux["vec"]

    _, pullback = jax.vjp(solved_aux_vec, p)
    vec_bar = jnp.array([1.0, 10.0])
    (p_bar,) = pullback(vec_bar)
    assert jnp.allclose(p_bar["scale"], 2.0 * s / 5.0 + 10.0 * 2.0 / 5.0, atol=1e-5)
    assert p_bar["flag"].dtype == jax.dtypes.float0


def test_implicit_cg_works_with_has_aux_and_pytree_x_p():
    def residual(x, _, p):
        theta = jnp.array([x["left"], x["right"]["value"]])
        scale = p["scale"]
        r = jnp.array([theta[0] + 2.0 * theta[1] - scale])
        aux = {
            "m": theta[0] * theta[1] + scale**2,
            "count": jnp.asarray(1, dtype=jnp.int32),
        }
        return r, aux

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
        implicit_solver="auto",
        implicit_tol=1e-7,
        has_aux=True,
    )
    x0 = {"left": jnp.asarray(0.0), "right": {"value": jnp.asarray(0.0)}}
    p = {"scale": jnp.asarray(3.0), "flag": jnp.asarray(2, dtype=jnp.int32)}
    p_dot = {
        "scale": jnp.asarray(0.7),
        "flag": jnp.zeros((), dtype=jax.dtypes.float0),
    }

    def solved(q):
        result = solver.solve(x0, p=q, max_steps=80, atol=1e-6)
        return result.x, result.aux

    (x, aux), (x_dot, aux_dot) = jax.jvp(solved, (p,), (p_dot,))
    expected_m_dot = (4.0 * p["scale"] / 25.0 + 2.0 * p["scale"]) * p_dot["scale"]
    assert jnp.allclose(x["left"], p["scale"] / 5.0, atol=1e-5)
    assert jnp.allclose(x["right"]["value"], 2.0 * p["scale"] / 5.0, atol=1e-5)
    assert aux["count"].dtype == jnp.int32
    assert jnp.allclose(x_dot["left"], 0.7 / 5.0, atol=1e-6)
    assert jnp.allclose(x_dot["right"]["value"], 1.4 / 5.0, atol=1e-6)
    assert jnp.allclose(aux_dot["m"], expected_m_dot, atol=1e-5)
    assert aux_dot["count"].dtype == jax.dtypes.float0

    def solved_aux_m(q):
        return solver.solve(x0, p=q, max_steps=80, atol=1e-6).aux["m"]

    _, pullback = jax.vjp(solved_aux_m, p)
    (p_bar,) = pullback(jnp.asarray(1.3))
    assert jnp.allclose(
        p_bar["scale"],
        jnp.asarray(1.3) * (4.0 * p["scale"] / 25.0 + 2.0 * p["scale"]),
        atol=1e-5,
    )
    assert p_bar["flag"].dtype == jax.dtypes.float0


def test_solve_implicit_vjp_of_aux_wrt_p():
    def residual(theta, _, p):
        aux = {
            "m": theta[0] * theta[1] + p**2,
            "count": jnp.asarray(1, dtype=jnp.int32),
        }
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), aux

    solver = LevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)
    p = jnp.asarray(3.0)
    dm_dp = 4.0 * p / 25.0 + 2.0 * p

    def solved_aux_m(q):
        return solver.solve(jnp.zeros(2), p=q, max_steps=80, atol=1e-6).aux["m"]

    _, pullback = jax.vjp(solved_aux_m, p)
    m_bar = jnp.asarray(1.3)
    (p_bar,) = pullback(m_bar)
    assert jnp.allclose(p_bar, m_bar * dm_dp, atol=1e-6)

    # Joint cotangents on x and aux["m"] pull back additively.
    def solved_joint(q):
        result = solver.solve(jnp.zeros(2), p=q, max_steps=80, atol=1e-6)
        return result.x, result.aux["m"]

    _, joint_pullback = jax.vjp(solved_joint, p)
    x_bar = jnp.array([3.0, 4.0])
    (p_bar_joint,) = joint_pullback((x_bar, m_bar))
    expected_x_pullback = (x_bar[0] + 2.0 * x_bar[1]) / 5.0
    assert jnp.allclose(p_bar_joint, expected_x_pullback + m_bar * dm_dp, atol=1e-6)

    # grad-under-jit exercises the transposed rule inside an outer trace.
    jitted_grad = jax.jit(jax.grad(solved_aux_m))(p)
    assert jnp.allclose(jitted_grad, dm_dp, atol=1e-6)


def test_solve_callback_wall_clock_time_limit():
    # The cookbook recipe: read the host clock via io_callback with the start
    # time and budget carried in user_state.
    import time

    import numpy as np
    from jax.experimental import io_callback

    time_limit_status = 100

    def over_time_budget(start_and_budget, _step):
        start, budget = start_and_budget
        return np.bool_(time.perf_counter() - float(start) > float(budget))

    def time_limit_callback(ctx):
        timed_out = io_callback(
            over_time_budget,
            jax.ShapeDtypeStruct((), jnp.bool_),
            ctx.user_state,
            ctx.step,
        )
        return LMSolveAction(stop=timed_out, status=time_limit_status)

    def residual(x, args):
        return jnp.sin(x) - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    x0 = jnp.zeros(64)
    target = jnp.full(64, 0.5)

    # exhausted budget: stops immediately with the custom status
    result = solver.solve(
        x0,
        target,
        max_steps=50,
        callback=time_limit_callback,
        user_state=jnp.asarray([time.perf_counter(), 0.0]),
    )
    assert int(result.status) == time_limit_status
    assert int(result.steps) == 1

    # generous budget: runs to a normal stopping rule
    result = solver.solve(
        x0,
        target,
        max_steps=50,
        atol=1e-6,
        callback=time_limit_callback,
        user_state=jnp.asarray([time.perf_counter(), 1e9]),
    )
    assert int(result.status) == LMStatus.CONVERGED


def test_solve_implicit_ad_unaffected_by_cache_init_inside_trace():
    # With cache_jacobian=True, solve(lm_state=None) calls init() — one
    # residual evaluation outside the custom_jvp boundary. Its outputs are
    # shape-derived constants, so the implicit JVP/VJP wrt p must be
    # identical to the uncached solver's.
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    cached = LevenbergMarquardt(residual, init_damping=1e-2, cache_jacobian=True)
    plain = LevenbergMarquardt(residual, init_damping=1e-2)

    def solved_x(solver, p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x

    p, p_dot = jnp.asarray(3.0), jnp.asarray(0.7)
    _, cached_dot = jax.jvp(lambda q: solved_x(cached, q), (p,), (p_dot,))
    _, plain_dot = jax.jvp(lambda q: solved_x(plain, q), (p,), (p_dot,))
    assert jnp.allclose(cached_dot, jnp.array([0.7 / 5.0, 1.4 / 5.0]), atol=1e-6)
    assert jnp.allclose(cached_dot, plain_dot, atol=1e-7)

    _, pullback = jax.vjp(lambda q: solved_x(cached, q), p)
    (p_bar,) = pullback(jnp.array([3.0, 4.0]))
    assert jnp.allclose(p_bar, (3.0 + 2.0 * 4.0) / 5.0, atol=1e-6)


def test_solve_derivative_wrt_x0_is_zero_by_contract():
    # The implicit rule differentiates only wrt p; the initial guess is
    # treated as fixed, so tangents on x0 are zero rather than an error.
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    solver = LevenbergMarquardt(residual, init_damping=1e-2)

    def solved_x(x0):
        return solver.solve(x0, p=jnp.asarray(3.0), max_steps=80, atol=1e-6).x

    _, x0_dot = jax.jvp(solved_x, (jnp.zeros(2),), (jnp.ones(2),))
    assert jnp.allclose(x0_dot, jnp.zeros(2))


def test_dual_preconditioner_requires_cg():
    with pytest.raises(ValueError, match="dual_preconditioner"):
        LevenbergMarquardt(residual_fn, dual_preconditioner=lambda v, damping: v)
    with pytest.raises(ValueError, match="dual_preconditioner"):
        LevenbergMarquardt(
            residual_fn,
            linear_solver="qr",
            dual_preconditioner=lambda v, damping: v,
        )


def test_cg_requires_dual_preconditioner():
    with pytest.raises(ValueError, match="identity_preconditioner"):
        LevenbergMarquardt(
            residual_fn,
            linear_solver="cg",
            implicit_preconditioner=identity_preconditioner(),
        )
    # Both callbacks missing reports both remedies in one error.
    with pytest.raises(ValueError, match="dual_preconditioner, and"):
        LevenbergMarquardt(residual_fn, linear_solver="cg")


def test_implicit_cg_requires_preconditioner():
    with pytest.raises(ValueError, match="implicit_preconditioner"):
        LevenbergMarquardt(residual_fn, implicit_solver="cg")
    # implicit_solver="auto" resolves to cg when linear_solver="cg", so the
    # default auto still requires the implicit callback.
    with pytest.raises(ValueError, match="implicit_preconditioner"):
        LevenbergMarquardt(
            residual_fn,
            linear_solver="cg",
            dual_preconditioner=identity_preconditioner(),
        )
    # The dense implicit rule is the other escape hatch.
    LevenbergMarquardt(
        residual_fn,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_solver="cholesky",
    )


def test_implicit_cg_exact_preconditioner_matches_closed_form_with_tiny_budget():
    matrix = jnp.array([[1.0, 0.5, -0.2, 0.1], [0.3, -0.7, 0.4, 1.0]])
    target_matrix = jnp.array([[1.0, -0.5], [0.25, 0.75]])
    gram = matrix @ matrix.T

    def residual(theta, _, p):
        return matrix @ theta - target_matrix @ p

    def exact_implicit_preconditioner(v):
        return _solve_2x2(gram, v)

    preconditioned = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
        implicit_solver="cg",
        implicit_tol=0.0,
        implicit_maxiter=1,
        implicit_preconditioner=exact_implicit_preconditioner,
        geodesic_acceleration=False,
    )
    theta0 = jnp.zeros(matrix.shape[1])

    def solved_x(solver, p):
        return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).x

    p = jnp.array([0.2, -0.8])
    p_dot = jnp.array([0.7, -0.4])
    _, preconditioned_dot = jax.jvp(
        lambda q: solved_x(preconditioned, q), (p,), (p_dot,)
    )

    _, preconditioned_pullback = jax.vjp(lambda q: solved_x(preconditioned, q), p)
    theta_bar = jnp.array([0.1, -0.2, 0.3, 0.4])
    (preconditioned_bar,) = preconditioned_pullback(theta_bar)
    expected_dot = matrix.T @ _solve_2x2(gram, target_matrix @ p_dot)
    expected_bar = target_matrix.T @ _solve_2x2(gram, matrix @ theta_bar)

    assert jnp.allclose(preconditioned_dot, expected_dot, atol=1e-6)
    assert jnp.allclose(preconditioned_bar, expected_bar, atol=1e-6)


def test_implicit_cg_does_not_reuse_forward_dual_preconditioner():
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    # This callback is valid for the damped forward system, but would be
    # singular if the implicit rule silently called it with zero damping;
    # the implicit solve must use the separately supplied identity instead.
    def dual_preconditioner(v, damping):
        return v / damping

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
        dual_preconditioner=dual_preconditioner,
        implicit_solver="auto",
        implicit_tol=1e-7,
    )

    def solved_x(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x

    _, x_dot = jax.jvp(solved_x, (jnp.asarray(3.0),), (jnp.asarray(0.7),))

    assert jnp.all(jnp.isfinite(x_dot))
    assert jnp.allclose(x_dot, jnp.array([0.7 / 5.0, 1.4 / 5.0]), atol=1e-6)


def test_implicit_cg_can_explicitly_reuse_zero_damping_dual_preconditioner():
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    def dual_preconditioner(v, damping):
        return v / (5.0 + damping)

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        iterative_tol=1e-7,
        iterative_maxiter=20,
        dual_preconditioner=dual_preconditioner,
        implicit_solver="auto",
        implicit_tol=0.0,
        implicit_maxiter=1,
        implicit_preconditioner=lambda v: dual_preconditioner(
            v, jnp.asarray(0.0, v.dtype)
        ),
    )

    def solved_x(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x

    _, x_dot = jax.jvp(solved_x, (jnp.asarray(3.0),), (jnp.asarray(0.7),))
    _, pullback = jax.vjp(solved_x, jnp.asarray(3.0))
    (p_bar,) = pullback(jnp.array([3.0, 4.0]))

    assert jnp.allclose(x_dot, jnp.array([0.7 / 5.0, 1.4 / 5.0]), atol=1e-6)
    assert jnp.allclose(p_bar, (3.0 + 2.0 * 4.0) / 5.0, atol=1e-6)


def test_implicit_preconditioner_accepts_dual_signature():
    # A callable REQUIRING (v, damping) serves the implicit hook directly:
    # the solver calls it with an explicit ZERO damping. The dual here is
    # diag(1, 25), so v / (diag + damping) at a one-iteration budget is
    # exact only at damping == 0 — any other value (e.g. a leaked live
    # damping) misses the analytic tangent by ~1e-3, far outside the
    # tolerance.
    def residual(theta, _, p):
        return jnp.array([theta[0] - p, 5.0 * theta[1] - p])

    dual_eigenvalues = jnp.array([1.0, 25.0])

    def dual_preconditioner(v, damping):
        return v / (dual_eigenvalues + damping)

    common = dict(
        init_damping=1e-2,
        linear_solver="cg",
        iterative_tol=1e-7,
        iterative_maxiter=20,
        dual_preconditioner=dual_preconditioner,
        implicit_solver="auto",
        implicit_tol=0.0,
        implicit_maxiter=1,
        geodesic_acceleration=False,
    )
    direct = LevenbergMarquardt(
        residual, implicit_preconditioner=dual_preconditioner, **common
    )
    wrapped = LevenbergMarquardt(
        residual,
        implicit_preconditioner=lambda v: dual_preconditioner(
            v, jnp.asarray(0.0, v.dtype)
        ),
        **common,
    )

    def solved_x(solver, p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x

    p, p_dot = jnp.asarray(3.0), jnp.asarray(0.7)
    expected_dot = jnp.array([0.7, 0.7 / 5.0])
    _, direct_dot = jax.jvp(lambda q: solved_x(direct, q), (p,), (p_dot,))
    _, wrapped_dot = jax.jvp(lambda q: solved_x(wrapped, q), (p,), (p_dot,))
    _, pullback = jax.vjp(lambda q: solved_x(direct, q), p)
    (p_bar,) = pullback(jnp.array([3.0, 4.0]))

    assert jnp.allclose(direct_dot, expected_dot, atol=1e-6)
    assert jnp.allclose(direct_dot, wrapped_dot, atol=1e-7)
    assert jnp.allclose(p_bar, 3.0 + 4.0 / 5.0, atol=1e-6)


def test_implicit_preconditioner_arity_edge_cases():
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    common = dict(
        init_damping=1e-2,
        linear_solver="cg",
        iterative_tol=1e-7,
        iterative_maxiter=20,
        dual_preconditioner=identity_preconditioner(),
        implicit_tol=1e-7,
    )

    # A 1-arg-callable with a defaulted EXTRA argument passes through
    # unchanged: were it wrongly called with a zero second argument, the
    # preconditioner would return the zero vector and the tangent would be
    # garbage.
    def one_arg_with_default(v, scale=1.0):
        return v * scale

    def solved_x(solver, p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x

    solver = LevenbergMarquardt(
        residual, implicit_preconditioner=one_arg_with_default, **common
    )
    _, x_dot = jax.jvp(
        lambda q: solved_x(solver, q), (jnp.asarray(3.0),), (jnp.asarray(0.7),)
    )
    assert jnp.allclose(x_dot, jnp.array([0.7 / 5.0, 1.4 / 5.0]), atol=1e-6)

    # A jit-wrapped 1-arg callable is accepted unchanged.
    LevenbergMarquardt(residual, implicit_preconditioner=jax.jit(lambda v: v), **common)

    # Zero-argument callables are rejected at construction.
    with pytest.raises(ValueError, match="callable as .v."):
        LevenbergMarquardt(residual, implicit_preconditioner=lambda: 0, **common)

    # pad_dual_preconditioner divides by the live damping; the zero-damping
    # implicit hook rejects it at construction instead of dividing by zero.
    with pytest.raises(ValueError, match="undamped"):
        LevenbergMarquardt(
            residual,
            implicit_preconditioner=pad_dual_preconditioner(
                identity_preconditioner(), 1
            ),
            **common,
        )


def test_cg_preconditioned_step_matches_cholesky_identity_step():
    # A valid SPD preconditioner changes only the inner Krylov iteration, not
    # the step the inner solve converges to.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    weights = 1.0 + jnp.arange(20, dtype=jnp.float32) / 10.0

    cholesky_solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=40,
        dual_preconditioner=lambda v, damping: v / weights,
    )

    cholesky_x, cholesky_state, cholesky_info = cholesky_solver.update(
        x, cholesky_solver.init(x, (ts, ys)), (ts, ys)
    )
    cg_x, cg_state, cg_info = cg_solver.update(x, cg_solver.init(x, (ts, ys)), (ts, ys))

    assert bool(cg_info.accepted) == bool(cholesky_info.accepted)
    # float32 across BLAS/SIMD variants: CI runners land ~2e-5 off macOS.
    assert jnp.allclose(cg_x["a"], cholesky_x["a"], rtol=1e-4, atol=1e-4)
    assert jnp.allclose(cg_x["b"], cholesky_x["b"], rtol=1e-4, atol=1e-4)
    assert jnp.allclose(cg_state.damping, cholesky_state.damping)
    assert jnp.allclose(
        cg_info.loss,
        cholesky_info.loss,
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )


def test_cg_preconditioned_update_jits():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    weights = 1.0 + jnp.arange(20, dtype=jnp.float32) / 10.0
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=40,
        dual_preconditioner=lambda v, damping: v / weights,
    )

    @jax.jit
    def train_step(x, lm_state):
        return solver.update(x, lm_state, (ts, ys))

    x, lm_state, info = train_step(x, solver.init(x, (ts, ys)))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(lm_state.damping)


def test_cg_preconditioned_geodesic_matches_cholesky():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    cholesky_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    cg_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=10,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
        dual_preconditioner=lambda v, damping: v / (1.0 + damping),
    )

    cholesky_theta, _, cholesky_info = cholesky_solver.update(
        theta0, cholesky_solver.init(theta0, target), target
    )
    cg_theta, _, cg_info = cg_solver.update(
        theta0, cg_solver.init(theta0, target), target
    )

    assert bool(cg_info.accepted)
    assert bool(cg_info.used_geodesic)
    assert jnp.allclose(cg_theta, cholesky_theta, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(
        cg_info.acceleration_ratio,
        cholesky_info.acceleration_ratio,
        rtol=1e-6,
        atol=1e-6,
    )


def test_cg_dual_preconditioner_enables_ill_conditioned_convergence():
    # Kernel-collocation miniature: affine residual K x - b with metric M = K,
    # so the dual operator is K K^{-1} K = K itself -- as ill-conditioned as
    # the kernel (cond ~ 1e4 here). At a tight inner-iteration budget plain CG
    # stalls, while the exact preconditioner (a K-solve) recovers
    # Gauss-Newton-quality steps and converges immediately.
    n = 40
    rho = 0.98
    idx = jnp.arange(n)
    K = rho ** jnp.abs(idx[:, None] - idx[None, :])
    L = jnp.linalg.cholesky(K)
    x_true = jnp.sin(idx / 3.0)
    b = K @ x_true

    def residual(x):
        return K @ x - b

    def preconditioner(v, damping):
        y = jsp_linalg.solve_triangular(L, v, lower=True)
        return jsp_linalg.solve_triangular(L.T, y, lower=False)

    common = dict(
        init_damping=1e-6,
        linear_solver="cg",
        iterative_maxiter=3,
        implicit_preconditioner=identity_preconditioner(),
        metric=metric_from_cholesky(L),
    )
    plain = LevenbergMarquardt(
        residual, dual_preconditioner=identity_preconditioner(), **common
    )
    preconditioned = LevenbergMarquardt(
        residual, dual_preconditioner=preconditioner, **common
    )
    x0 = jnp.zeros(n)

    plain_result = plain.solve(x0, max_steps=20, atol=1e-3)
    preconditioned_result = preconditioned.solve(x0, max_steps=20, atol=1e-3)

    assert int(preconditioned_result.status) == LMStatus.CONVERGED
    assert int(plain_result.status) != LMStatus.CONVERGED


@pytest.mark.parametrize("parallel", [False, True])
def test_metric_from_tridiagonal_precision_matches_dense(parallel):
    # M = K with K_ij = rho^|i-j| (Matern-1/2 on a unit grid), whose precision
    # T = K^{-1} is exactly tridiagonal with closed-form AR(1) entries.
    n = 15
    rho = 0.6
    idx = jnp.arange(n)
    K = rho ** jnp.abs(idx[:, None] - idx[None, :])
    scale = 1.0 / (1.0 - rho**2)
    diag = scale * jnp.concatenate(
        [jnp.ones(1), (1.0 + rho**2) * jnp.ones(n - 2), jnp.ones(1)]
    )
    off_diag = -rho * scale * jnp.ones(n - 1)
    metric = metric_from_tridiagonal_precision(diag, off_diag, parallel=parallel)

    x = jax.random.normal(jax.random.PRNGKey(0), (n,))
    X = jax.random.normal(jax.random.PRNGKey(1), (n, 3))

    # The dense float32 references go through matmuls that Ampere GPUs run in
    # TF32 (~1e-3 precision) by default, while the tridiagonal callbacks are
    # elementwise-exact; pin full precision so the references are comparable.
    with jax.default_matmul_precision("highest"):
        T_dense = jnp.linalg.inv(K)

        assert jnp.allclose(metric.solve(x), T_dense @ x, rtol=1e-4, atol=1e-4)
        assert jnp.allclose(metric.solve(X), T_dense @ X, rtol=1e-4, atol=1e-4)
        assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ K @ x), rtol=1e-4, atol=1e-4)
        S = metric.inv_sqrt(jnp.eye(n))
        assert jnp.allclose(S @ S.T, T_dense, rtol=1e-3, atol=1e-4)
        assert jnp.allclose(
            metric.inv_sqrt_transpose(jnp.eye(n)), S.T, rtol=1e-4, atol=1e-4
        )


def test_metric_from_diagonal_matches_dense():
    weights = jnp.array([2.0, 0.5, 4.0, 1.5])
    metric = metric_from_diagonal(weights)
    M = jnp.diag(weights)
    x = jax.random.normal(jax.random.PRNGKey(2), (4,))
    X = jax.random.normal(jax.random.PRNGKey(3), (4, 3))

    assert jnp.allclose(metric.solve(x), jnp.linalg.solve(M, x))
    assert jnp.allclose(metric.solve(X), jnp.linalg.solve(M, X))
    assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ M @ x))
    S = metric.inv_sqrt(jnp.eye(4))
    assert jnp.allclose(S @ S.T, jnp.linalg.inv(M))
    assert jnp.allclose(metric.inv_sqrt_transpose(jnp.eye(4)), S.T)


def test_blockdiag_metric_matches_dense():
    # A dense 3x3 block composed with a diagonal 2-block.
    A = jnp.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.2], [0.0, 0.2, 1.5]])
    weights = jnp.array([4.0, 0.25])
    metric = blockdiag_metric(
        [
            (metric_from_cholesky(jnp.linalg.cholesky(A)), 3),
            (metric_from_diagonal(weights), 2),
        ]
    )
    M = jnp.block(
        [
            [A, jnp.zeros((3, 2))],
            [jnp.zeros((2, 3)), jnp.diag(weights)],
        ]
    )
    x = jax.random.normal(jax.random.PRNGKey(4), (5,))
    X = jax.random.normal(jax.random.PRNGKey(5), (5, 3))

    assert jnp.allclose(metric.solve(x), jnp.linalg.solve(M, x), atol=1e-5)
    assert jnp.allclose(metric.solve(X), jnp.linalg.solve(M, X), atol=1e-5)
    assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ M @ x), atol=1e-5)
    S = metric.inv_sqrt(jnp.eye(5))
    assert jnp.allclose(S @ S.T, jnp.linalg.inv(M), atol=1e-5)

    # A partially-specified block (here cg-style: solve and norm only) marks
    # the callbacks it lacks as missing on the composite rather than filling
    # identity, so the solver's validation still catches e.g. qr without
    # inv_sqrt — identity fill would break S S' = M^{-1} consistency.
    partial = blockdiag_metric(
        [
            (
                Metric(
                    solve=lambda x: x / 4.0, norm=lambda x: 2.0 * jnp.linalg.norm(x)
                ),
                3,
            ),
            (metric_from_diagonal(weights), 2),
        ]
    )
    assert partial.inv_sqrt is None
    assert partial.inv_sqrt_transpose is None
    x_partial = jax.random.normal(jax.random.PRNGKey(7), (5,))
    M_partial = jnp.block(
        [
            [4.0 * jnp.eye(3), jnp.zeros((3, 2))],
            [jnp.zeros((2, 3)), jnp.diag(weights)],
        ]
    )
    assert jnp.allclose(
        partial.solve(x_partial), jnp.linalg.solve(M_partial, x_partial), atol=1e-6
    )
    assert jnp.allclose(
        partial.norm(x_partial), jnp.sqrt(x_partial @ M_partial @ x_partial), atol=1e-5
    )
    with pytest.raises(ValueError, match="inv_sqrt"):
        LevenbergMarquardt(residual_fn, linear_solver="qr", metric=partial)


def test_sherman_morrison_preconditioner_matches_dense_inverse():
    n = 12
    idx = jnp.arange(n)
    A = 0.6 ** jnp.abs(idx[:, None] - idx[None, :])
    L = jnp.linalg.cholesky(A)
    u = jnp.ones(n)
    weight = 50.0

    preconditioner = sherman_morrison_preconditioner(
        metric_from_cholesky(L).solve, u, weight
    )
    P = A + weight * jnp.outer(u, u)
    v = jax.random.normal(jax.random.PRNGKey(6), (n,))

    assert jnp.allclose(
        preconditioner(v, 0.0), jnp.linalg.solve(P, v), rtol=1e-3, atol=1e-3
    )


def test_metric_from_tridiagonal_precision_float32_default_is_stable():
    # Long, stiff AR(1) grid (rho near 1) in float32: the parallel Mobius
    # scan's projective cancellation goes non-finite here, the sequential
    # recurrence does not. The dtype-aware default must never pick the
    # unstable path for float32 setups, on any backend.
    n, rho = 5000, 0.9999
    scale = 1.0 / (1.0 - rho**2)
    diag = (
        scale
        * jnp.concatenate([jnp.ones(1), (1.0 + rho**2) * jnp.ones(n - 2), jnp.ones(1)])
    ).astype(jnp.float32)
    off_diag = (-rho * scale * jnp.ones(n - 1)).astype(jnp.float32)

    metric = metric_from_tridiagonal_precision(diag, off_diag)
    assert bool(jnp.all(jnp.isfinite(metric.inv_sqrt(jnp.ones(n, jnp.float32)))))


def test_metric_from_tridiagonal_precision_single_point():
    metric = metric_from_tridiagonal_precision(jnp.array([4.0]), jnp.zeros(0))
    x = jnp.array([3.0])

    assert jnp.allclose(metric.solve(x), 4.0 * x)
    assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ x / 4.0))
    assert jnp.allclose(metric.inv_sqrt(x), 2.0 * x)


def test_blockdiag_metric_identity_block_defaults():
    # A bare Metric() block means identity on that block -- the other block's
    # weighting must survive.
    weights = jnp.array([4.0, 0.25])
    metric = blockdiag_metric([(Metric(), 3), (metric_from_diagonal(weights), 2)])
    M = jnp.block(
        [
            [jnp.eye(3), jnp.zeros((3, 2))],
            [jnp.zeros((2, 3)), jnp.diag(weights)],
        ]
    )
    x = jax.random.normal(jax.random.PRNGKey(9), (5,))

    assert metric.solve is not None
    assert jnp.allclose(metric.solve(x), jnp.linalg.solve(M, x), atol=1e-6)
    assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ M @ x), atol=1e-6)

    with pytest.raises(ValueError, match="at least one"):
        blockdiag_metric([])


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr", "augmented_qr"])
def test_structural_metrics_match_dense_metric_in_solver(linear_solver):
    # End-to-end: the O(n) tridiagonal metric drives the actual solver (and
    # its implicit derivative) to the same solution as the equivalent dense
    # metric — the constructors compose with every callback the solver uses.
    n = 8
    rho = 0.6
    idx = jnp.arange(n)
    K = rho ** jnp.abs(idx[:, None] - idx[None, :])
    scale = 1.0 / (1.0 - rho**2)
    diag = scale * jnp.concatenate(
        [jnp.ones(1), (1.0 + rho**2) * jnp.ones(n - 2), jnp.ones(1)]
    )
    off_diag = -rho * scale * jnp.ones(n - 1)
    tridiagonal = metric_from_tridiagonal_precision(diag, off_diag)
    dense = metric_from_cholesky(jnp.linalg.cholesky(K))

    # Underdetermined: 2 residuals, n parameters, external p for implicit AD.
    A = jnp.stack([jnp.ones(n), idx.astype(jnp.float32)])

    def residual(theta, _, p):
        return A @ theta - jnp.array([p, 0.5 * p])

    def solved_x(metric, p):
        solver = LevenbergMarquardt(
            residual, init_damping=1e-2, linear_solver=linear_solver, metric=metric
        )
        return solver.solve(jnp.zeros(n), p=p, max_steps=60, atol=1e-6).x

    p, p_dot = jnp.asarray(2.0), jnp.asarray(1.0)
    x_tri, x_tri_dot = jax.jvp(lambda q: solved_x(tridiagonal, q), (p,), (p_dot,))
    x_dense, x_dense_dot = jax.jvp(lambda q: solved_x(dense, q), (p,), (p_dot,))

    assert jnp.allclose(A @ x_tri, jnp.array([p, 0.5 * p]), atol=1e-4)
    assert jnp.allclose(x_tri, x_dense, atol=1e-4)
    assert jnp.allclose(x_tri_dot, x_dense_dot, atol=1e-4)

    x_bar = jnp.linspace(-1.0, 1.0, n)
    _, tri_pullback = jax.vjp(lambda q: solved_x(tridiagonal, q), p)
    _, dense_pullback = jax.vjp(lambda q: solved_x(dense, q), p)
    assert jnp.allclose(tri_pullback(x_bar)[0], dense_pullback(x_bar)[0], atol=1e-4)


def test_blockdiag_metric_implicit_ad_matches_dense_metric():
    # A blockdiag composite (dense kernel block + diagonal scalar block)
    # drives solve() and its implicit JVP/VJP to the same answers as the
    # equivalent single dense metric.
    idx = jnp.arange(3)
    K = 0.6 ** jnp.abs(idx[:, None] - idx[None, :])
    weights = jnp.array([4.0, 0.25])
    composite = blockdiag_metric(
        [
            (metric_from_cholesky(jnp.linalg.cholesky(K)), 3),
            (metric_from_diagonal(weights), 2),
        ]
    )
    M = jnp.block(
        [
            [K, jnp.zeros((3, 2))],
            [jnp.zeros((2, 3)), jnp.diag(weights)],
        ]
    )
    dense = metric_from_cholesky(jnp.linalg.cholesky(M))
    A = jnp.stack([jnp.ones(5), jnp.arange(5, dtype=jnp.float32)])

    def residual(theta, _, p):
        return A @ theta - jnp.array([p, 0.5 * p])

    def solved_x(metric, p):
        solver = LevenbergMarquardt(residual, init_damping=1e-2, metric=metric)
        return solver.solve(jnp.zeros(5), p=p, max_steps=60, atol=1e-6).x

    p, p_dot = jnp.asarray(2.0), jnp.asarray(1.0)
    x_block, x_block_dot = jax.jvp(lambda q: solved_x(composite, q), (p,), (p_dot,))
    x_dense, x_dense_dot = jax.jvp(lambda q: solved_x(dense, q), (p,), (p_dot,))
    assert jnp.allclose(x_block, x_dense, atol=1e-4)
    assert jnp.allclose(x_block_dot, x_dense_dot, atol=1e-4)

    x_bar = jnp.linspace(-1.0, 1.0, 5)
    _, block_pullback = jax.vjp(lambda q: solved_x(composite, q), p)
    _, dense_pullback = jax.vjp(lambda q: solved_x(dense, q), p)
    assert jnp.allclose(block_pullback(x_bar)[0], dense_pullback(x_bar)[0], atol=1e-4)


REPEATED_BLOCK = jnp.array(
    [
        [2.0, 0.2, 0.1, 0.0],
        [0.2, 1.8, 0.0, 0.1],
        [0.1, 0.0, 1.5, 0.2],
        [0.0, 0.1, 0.2, 1.2],
    ]
)


def test_repeated_blockdiag_metric_matches_blockdiag():
    # repeated_blockdiag_metric must equal the expanded blockdiag_metric of the
    # same blocks, callback for callback, and match a ground-truth dense metric.
    block_size, repeats = 4, 3
    weights = jnp.array([0.3, 0.7])
    block = metric_from_cholesky(jnp.linalg.cholesky(REPEATED_BLOCK))
    additional = metric_from_diagonal(weights)
    repeated = repeated_blockdiag_metric(
        block, block_size, repeats, additional=(additional, weights.shape[0])
    )
    reference = blockdiag_metric(
        [(block, block_size)] * repeats + [(additional, weights.shape[0])]
    )
    total = block_size * repeats + weights.shape[0]
    M = jsp_linalg.block_diag(*([REPEATED_BLOCK] * repeats), jnp.diag(weights))

    x = jax.random.normal(jax.random.PRNGKey(10), (total,))
    X = jax.random.normal(jax.random.PRNGKey(11), (total, 3))

    assert jnp.allclose(repeated.solve(x), reference.solve(x), atol=1e-5)
    assert jnp.allclose(repeated.solve(X), reference.solve(X), atol=1e-5)
    assert jnp.allclose(repeated.solve(x), jnp.linalg.solve(M, x), atol=1e-5)
    assert jnp.allclose(repeated.solve(X), jnp.linalg.solve(M, X), atol=1e-5)
    assert jnp.allclose(repeated.norm(x), reference.norm(x), atol=1e-5)
    assert jnp.allclose(repeated.norm(x), jnp.sqrt(x @ M @ x), atol=1e-5)

    S = repeated.inv_sqrt(jnp.eye(total))
    assert jnp.allclose(S, reference.inv_sqrt(jnp.eye(total)), atol=1e-5)
    assert jnp.allclose(S @ S.T, jnp.linalg.inv(M), atol=1e-5)
    assert jnp.allclose(
        repeated.inv_sqrt_transpose(jnp.eye(total)),
        reference.inv_sqrt_transpose(jnp.eye(total)),
        atol=1e-5,
    )
    assert jnp.allclose(repeated.inv_sqrt_transpose(jnp.eye(total)), S.T, atol=1e-5)


def test_repeated_blockdiag_metric_batches_one_call():
    # The base callback fires exactly once, on the whole repeated head reshaped
    # to (block_size, repeats * columns) -- the anti-degeneration guard against
    # regressing to one call per copy.
    block_size, repeats, rhs_columns = 4, 3, 5
    received_shapes = []

    def solve(x):
        received_shapes.append(x.shape)
        return x

    metric = repeated_blockdiag_metric(Metric(solve=solve), block_size, repeats)
    metric.solve(jnp.ones((block_size * repeats, rhs_columns)))

    assert received_shapes == [(block_size, repeats * rhs_columns)]


def test_repeated_blockdiag_metric_partial_and_identity():
    block_size, repeats = 4, 3
    total = block_size * repeats

    # A block defining only solve marks the other callbacks missing on the
    # composite (same contract as blockdiag_metric).
    partial = repeated_blockdiag_metric(Metric(solve=lambda x: x), block_size, repeats)
    assert partial.solve is not None
    assert partial.norm is None
    assert partial.inv_sqrt is None
    assert partial.inv_sqrt_transpose is None

    # A bare Metric() block is the identity on its span; the additional block's
    # weighting must survive.
    weights = jnp.array([0.3, 0.7])
    identity_repeated = repeated_blockdiag_metric(
        Metric(), block_size, repeats, additional=(metric_from_diagonal(weights), 2)
    )
    M = jsp_linalg.block_diag(jnp.eye(total), jnp.diag(weights))
    x = jax.random.normal(jax.random.PRNGKey(12), (total + 2,))
    assert identity_repeated.solve is not None
    assert jnp.allclose(identity_repeated.solve(x), jnp.linalg.solve(M, x), atol=1e-6)
    assert jnp.allclose(identity_repeated.norm(x), jnp.sqrt(x @ M @ x), atol=1e-5)

    with pytest.raises(ValueError, match="leading size"):
        identity_repeated.solve(jnp.ones(block_size))
    with pytest.raises(ValueError, match="positive integer"):
        repeated_blockdiag_metric(Metric(), block_size, 0)


def test_metric_with_compute_dtype_preserves_caller_dtype():
    # The wrapper restores the caller's dtype and preserves None callbacks; the
    # genuine wide-precision path is checked in the x64 subprocess suite.
    factor = jnp.linalg.cholesky(jnp.array([[2.0, 0.2], [0.2, 1.5]]))
    wrapped = metric_with_compute_dtype(metric_from_cholesky(factor), jnp.float64)
    value = jnp.array([1.0, -0.5], dtype=jnp.float32)

    solved = wrapped.solve(value)
    assert solved.dtype == value.dtype
    assert jnp.allclose(solved, metric_from_cholesky(factor).solve(value), atol=1e-6)

    partial = metric_with_compute_dtype(Metric(solve=lambda x: 2.0 * x), jnp.float64)
    assert partial.solve is not None
    assert partial.norm is None
    assert partial.inv_sqrt is None
    assert partial.inv_sqrt_transpose is None


def test_repeated_blockdiag_metric_norm_ad_matches_blockdiag():
    # AD through the vmapped block norm must match the expanded blockdiag norm
    # for both forward (jvp) and reverse (grad) modes, away from zero where
    # jnp.linalg.norm's gradient is defined.
    block_size, repeats = 4, 3
    weights = jnp.array([0.3, 0.7])
    block = metric_from_cholesky(jnp.linalg.cholesky(REPEATED_BLOCK))
    additional = metric_from_diagonal(weights)
    repeated = repeated_blockdiag_metric(
        block, block_size, repeats, additional=(additional, 2)
    )
    reference = blockdiag_metric([(block, block_size)] * repeats + [(additional, 2)])
    total = block_size * repeats + 2
    v = jax.random.normal(jax.random.PRNGKey(13), (total,))
    dv = jax.random.normal(jax.random.PRNGKey(14), (total,))

    val_rep, tangent_rep = jax.jvp(repeated.norm, (v,), (dv,))
    val_ref, tangent_ref = jax.jvp(reference.norm, (v,), (dv,))
    assert jnp.allclose(val_rep, val_ref, atol=1e-5)
    assert jnp.allclose(tangent_rep, tangent_ref, atol=1e-5)
    assert jnp.allclose(
        jax.grad(repeated.norm)(v), jax.grad(reference.norm)(v), atol=1e-5
    )


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr", "augmented_qr"])
@pytest.mark.parametrize("geodesic_acceleration", [False, True])
def test_repeated_blockdiag_metric_implicit_ad_matches_blockdiag(
    linear_solver, geodesic_acceleration
):
    # End-to-end: the repeated metric drives solve() and its implicit JVP/VJP to
    # the same answers as the equivalent expanded blockdiag metric, across the
    # dense solver steps, with geodesic acceleration exercising norm.
    idx = jnp.arange(3)
    K = 0.6 ** jnp.abs(idx[:, None] - idx[None, :])
    weights = jnp.array([4.0, 0.25])
    block = metric_from_cholesky(jnp.linalg.cholesky(K))
    additional = metric_from_diagonal(weights)
    repeated = repeated_blockdiag_metric(block, 3, 2, additional=(additional, 2))
    reference = blockdiag_metric([(block, 3), (block, 3), (additional, 2)])
    A = jnp.stack([jnp.ones(8), jnp.arange(8, dtype=jnp.float32)])

    def residual(theta, _, p):
        return A @ theta - jnp.array([p, 0.5 * p])

    def solved_x(metric, p):
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-2,
            linear_solver=linear_solver,
            geodesic_acceleration=geodesic_acceleration,
            metric=metric,
        )
        return solver.solve(jnp.zeros(8), p=p, max_steps=60, atol=1e-6).x

    p, p_dot = jnp.asarray(2.0), jnp.asarray(1.0)
    x_rep, x_rep_dot = jax.jvp(lambda q: solved_x(repeated, q), (p,), (p_dot,))
    x_ref, x_ref_dot = jax.jvp(lambda q: solved_x(reference, q), (p,), (p_dot,))
    assert jnp.allclose(x_rep, x_ref, atol=1e-4)
    assert jnp.allclose(x_rep_dot, x_ref_dot, atol=1e-4)

    x_bar = jnp.linspace(-1.0, 1.0, 8)
    _, rep_pullback = jax.vjp(lambda q: solved_x(repeated, q), p)
    _, ref_pullback = jax.vjp(lambda q: solved_x(reference, q), p)
    assert jnp.allclose(rep_pullback(x_bar)[0], ref_pullback(x_bar)[0], atol=1e-4)


def dense_matern_gram(t, sigma, ell, nu):
    tau = jnp.abs(t[:, None] - t[None, :])
    ft = jnp.sqrt(2.0 * nu) * tau / ell
    if nu == 0.5:
        corr = jnp.exp(-ft)
    elif nu == 1.5:
        corr = (1.0 + ft) * jnp.exp(-ft)
    else:
        corr = (1.0 + ft + ft**2 / 3.0) * jnp.exp(-ft)
    return sigma**2 * corr


def dense_from_quasiseparable_generators(d, p, q, A):
    # K[i, j] = p[i] @ A[i-1] ... A[j+1] @ q[j] for i > j, d on the
    # diagonal, symmetric — the documented generator convention.
    n, m = p.shape

    def entry(i, j):
        if i == j:
            return d[i]
        lo, hi = min(i, j), max(i, j)
        Phi = jnp.eye(m, dtype=d.dtype)
        for k in range(lo + 1, hi):
            Phi = A[k] @ Phi
        return p[hi] @ Phi @ q[lo]

    return jnp.array([[entry(i, j) for j in range(n)] for i in range(n)])


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
@pytest.mark.parametrize("parallel", [False, True])
def test_metric_from_state_space_matern_matches_dense(nu, parallel):
    # nu=0.5 is included even though metric_from_tridiagonal_precision is the
    # recommended constructor there — the helper claims support for it.
    n = 300
    sigma, ell = 1.3, 0.8
    nugget = 1e-8 * sigma**2
    uniform = jnp.arange(n) * 1.0
    nonuniform = jnp.cumsum(
        jax.random.uniform(jax.random.PRNGKey(0), (n,), minval=0.6, maxval=1.4)
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (n,))
    X = jax.random.normal(jax.random.PRNGKey(2), (n, 3))

    for t in (uniform, nonuniform):
        metric = metric_from_state_space(
            t, *matern_state_space(sigma, ell, nu), nugget=nugget, parallel=parallel
        )
        # Pin full matmul precision so the dense float32 references are
        # comparable to the scan-based callbacks (TF32 on Ampere GPUs).
        with jax.default_matmul_precision("highest"):
            K = dense_matern_gram(t, sigma, ell, nu) + nugget * jnp.eye(n)
            K_inv = jnp.linalg.inv(K)

            assert jnp.allclose(metric.solve(x), K_inv @ x, rtol=1e-4, atol=1e-4)
            assert jnp.allclose(metric.solve(X), K_inv @ X, rtol=1e-4, atol=1e-4)
            assert jnp.allclose(
                metric.norm(x), jnp.sqrt(x @ K @ x), rtol=1e-4, atol=1e-4
            )
            S = metric.inv_sqrt(jnp.eye(n))
            assert jnp.allclose(S @ S.T, K_inv, rtol=1e-3, atol=1e-4)
            assert jnp.allclose(
                metric.inv_sqrt_transpose(jnp.eye(n)), S.T, rtol=1e-4, atol=1e-4
            )
            # The implicit-AD fallback literally computes this identity.
            assert jnp.allclose(
                metric.solve(X),
                metric.inv_sqrt(metric.inv_sqrt_transpose(X)),
                rtol=1e-4,
                atol=1e-5,
            )


@pytest.mark.parametrize("parallel", [False, True])
def test_metric_from_quasiseparable_rank1_matches_dense(parallel):
    # Rank-1 exponential generators reproduce the AR(1) Gram rho^|i-j|.
    n, rho = 40, 0.7
    d = jnp.ones(n)
    p = jnp.full((n, 1), rho)
    q = jnp.ones((n, 1))
    A = jnp.full((n, 1, 1), rho)
    metric = metric_from_quasiseparable(d, p, q, A, parallel=parallel)

    idx = jnp.arange(n)
    K = rho ** jnp.abs(idx[:, None] - idx[None, :])
    x = jax.random.normal(jax.random.PRNGKey(3), (n,))
    X = jax.random.normal(jax.random.PRNGKey(4), (n, 3))

    with jax.default_matmul_precision("highest"):
        K_inv = jnp.linalg.inv(K)
        assert jnp.allclose(metric.solve(x), K_inv @ x, rtol=1e-4, atol=1e-4)
        assert jnp.allclose(metric.solve(X), K_inv @ X, rtol=1e-4, atol=1e-4)
        assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ K @ x), rtol=1e-4, atol=1e-4)
        S = metric.inv_sqrt(jnp.eye(n))
        assert jnp.allclose(S @ S.T, K_inv, rtol=1e-3, atol=1e-4)
        assert jnp.allclose(
            metric.inv_sqrt_transpose(jnp.eye(n)), S.T, rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_metric_from_quasiseparable_noncommuting_generators_match_dense(parallel):
    # m=1 generators commute and cannot catch product-order, A[k]-vs-A[k+1],
    # or transpose mistakes; hand-built noncommuting m=2 transitions can.
    n, m = 12, 2
    thetas = jnp.linspace(0.3, 2.4, n)
    scales = jnp.stack(
        [0.9 - 0.4 * jnp.linspace(0.0, 1.0, n), 0.5 + 0.3 * jnp.linspace(0.0, 1.0, n)],
        axis=-1,
    )
    cos, sin = jnp.cos(thetas), jnp.sin(thetas)
    rotations = jnp.stack(
        [jnp.stack([cos, -sin], axis=-1), jnp.stack([sin, cos], axis=-1)], axis=-2
    )
    A = rotations * scales[:, None, :]
    p = 0.4 * jax.random.normal(jax.random.PRNGKey(5), (n, m))
    q = 0.4 * jax.random.normal(jax.random.PRNGKey(6), (n, m))
    d = 5.0 * jnp.ones(n)

    K = dense_from_quasiseparable_generators(d, p, q, A)
    assert jnp.all(jnp.linalg.eigvalsh(K) > 0)
    metric = metric_from_quasiseparable(d, p, q, A, parallel=parallel)

    x = jax.random.normal(jax.random.PRNGKey(7), (n,))
    X = jax.random.normal(jax.random.PRNGKey(8), (n, 3))
    with jax.default_matmul_precision("highest"):
        K_inv = jnp.linalg.inv(K)
        assert jnp.allclose(metric.solve(x), K_inv @ x, rtol=1e-4, atol=1e-5)
        assert jnp.allclose(metric.solve(X), K_inv @ X, rtol=1e-4, atol=1e-5)
        assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ K @ x), rtol=1e-4, atol=1e-5)
        S = metric.inv_sqrt(jnp.eye(n))
        assert jnp.allclose(S @ S.T, K_inv, rtol=1e-3, atol=1e-5)
        assert jnp.allclose(
            metric.inv_sqrt_transpose(jnp.eye(n)), S.T, rtol=1e-4, atol=1e-5
        )


def test_metric_from_state_space_matern_edge_cases():
    # n=1: the metric is the scalar sigma^2 + nugget.
    metric = metric_from_state_space(
        jnp.array([0.5]), *matern_state_space(2.0, 1.0, 1.5)
    )
    x = jnp.array([3.0])
    assert jnp.allclose(metric.solve(x), x / 4.0)
    assert jnp.allclose(metric.norm(x), 6.0)
    assert jnp.allclose(metric.inv_sqrt(x), x / 2.0)

    # n=2, nugget=0 on a well-conditioned grid.
    t2 = jnp.array([0.0, 1.0])
    metric2 = metric_from_state_space(t2, *matern_state_space(1.3, 0.8, 2.5))
    K2 = dense_matern_gram(t2, 1.3, 0.8, 2.5)
    x2 = jnp.array([1.0, -2.0])
    assert jnp.allclose(metric2.solve(x2), jnp.linalg.solve(K2, x2), rtol=1e-5)
    assert jnp.allclose(metric2.norm(x2), jnp.sqrt(x2 @ K2 @ x2), rtol=1e-5)

    # nu is static and validated eagerly.
    with pytest.raises(ValueError, match="nu"):
        metric_from_state_space(t2, *matern_state_space(1.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="1-D"):
        metric_from_state_space(jnp.zeros((2, 2)), *matern_state_space(1.0, 1.0, 1.5))


def test_metric_from_quasiseparable_shape_validation():
    n, m = 4, 2
    d = jnp.ones(n)
    p = jnp.ones((n, m))
    with pytest.raises(ValueError, match="d must"):
        metric_from_quasiseparable(jnp.ones((n, 1)), p, p, jnp.ones((n, m, m)))
    with pytest.raises(ValueError, match="p must"):
        metric_from_quasiseparable(d, jnp.ones(n), p, jnp.ones((n, m, m)))
    with pytest.raises(ValueError, match="q must"):
        metric_from_quasiseparable(d, p, jnp.ones((n, m + 1)), jnp.ones((n, m, m)))


def test_metric_from_state_space_matern_float32_default_is_stable():
    # Long, stiff grid (spacing << ell) in float32: the parallel
    # substitutions propagate rank-1-corrected transitions with no
    # contraction guarantee, so the dtype-aware default must stay on the
    # sequential path for float32 setups, on any backend.
    n = 5000
    t = jnp.linspace(0.0, 5.0, n, dtype=jnp.float32)
    metric = metric_from_state_space(t, *matern_state_space(1.0, 1.0, 2.5), nugget=1e-4)
    assert bool(jnp.all(jnp.isfinite(metric.inv_sqrt(jnp.ones(n, jnp.float32)))))
    assert bool(jnp.all(jnp.isfinite(metric.solve(jnp.ones(n, jnp.float32)))))


def test_metric_from_state_space_matern_hyperparameter_grad_matches_dense():
    # The metric is constructed INSIDE jax.grad (and jax.jit(jax.grad)) from
    # traced (sigma, ell) — the downstream sweep pattern — and the gradient
    # must match the dense-metric gradient.
    n = 60
    t = jnp.cumsum(
        jax.random.uniform(jax.random.PRNGKey(9), (n,), minval=0.6, maxval=1.4)
    )
    v = jax.random.normal(jax.random.PRNGKey(10), (n,))
    nugget = 1e-6

    def loss_qsm(params):
        sigma, ell = params
        metric = metric_from_state_space(
            t, *matern_state_space(sigma, ell, 1.5), nugget=nugget
        )
        return v @ metric.solve(v)

    def loss_dense(params):
        sigma, ell = params
        K = dense_matern_gram(t, sigma, ell, 1.5) + nugget * jnp.eye(n)
        return v @ jnp.linalg.solve(K, v)

    params = jnp.array([1.3, 0.8])
    with jax.default_matmul_precision("highest"):
        grad_qsm = jax.grad(loss_qsm)(params)
        grad_jit = jax.jit(jax.grad(loss_qsm))(params)
        grad_dense = jax.grad(loss_dense)(params)
    assert jnp.allclose(grad_qsm, grad_dense, rtol=1e-2)
    assert jnp.allclose(grad_jit, grad_qsm, rtol=1e-4)


def test_metric_from_state_space_matern_small_pivot_grad_is_finite():
    # Nugget-free stiff grid drives the Schur pivots small; the gradient
    # through the Cholesky square roots must stay finite there (sqrt has an
    # AD blowup exactly at zero, so the pivots must not underflow — a truly
    # degenerate nugget-free grid NaNs, as documented).
    n = 30
    t = 0.3 * jnp.arange(n)

    def loss(ell):
        metric = metric_from_state_space(t, *matern_state_space(1.0, ell, 2.5))
        return jnp.sum(metric.inv_sqrt_transpose(jnp.ones(n)) ** 2)

    grad = jax.grad(loss)(jnp.asarray(1.0))
    assert bool(jnp.isfinite(grad))


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr", "augmented_qr"])
def test_metric_from_state_space_matern_matches_dense_metric_in_solver(linear_solver):
    # End-to-end: the O(n) Matern metric drives the actual solver (and its
    # implicit derivative) to the same solution as the equivalent dense
    # metric.
    n = 8
    t = jnp.arange(n) * 1.0
    sigma, ell, nugget = 1.3, 0.8, 1e-6
    K = dense_matern_gram(t, sigma, ell, 1.5) + nugget * jnp.eye(n)
    matern = metric_from_state_space(
        t, *matern_state_space(sigma, ell, 1.5), nugget=nugget
    )
    dense = metric_from_cholesky(jnp.linalg.cholesky(K))

    A = jnp.stack([jnp.ones(n), jnp.arange(n, dtype=jnp.float32)])

    def residual(theta, _, p):
        return A @ theta - jnp.array([p, 0.5 * p])

    solver_kwargs = {"init_damping": 1e-2, "linear_solver": linear_solver}

    def solved_x(metric, p):
        solver = LevenbergMarquardt(residual, metric=metric, **solver_kwargs)
        return solver.solve(jnp.zeros(n), p=p, max_steps=60, atol=1e-6).x

    p, p_dot = jnp.asarray(2.0), jnp.asarray(1.0)
    x_qsm, x_qsm_dot = jax.jvp(lambda s: solved_x(matern, s), (p,), (p_dot,))
    x_dense, x_dense_dot = jax.jvp(lambda s: solved_x(dense, s), (p,), (p_dot,))

    assert jnp.allclose(A @ x_qsm, jnp.array([p, 0.5 * p]), atol=1e-4)
    assert jnp.allclose(x_qsm, x_dense, atol=1e-4)
    assert jnp.allclose(x_qsm_dot, x_dense_dot, atol=1e-4)

    x_bar = jnp.linspace(-1.0, 1.0, n)
    _, qsm_pullback = jax.vjp(lambda s: solved_x(matern, s), p)
    _, dense_pullback = jax.vjp(lambda s: solved_x(dense, s), p)
    assert jnp.allclose(qsm_pullback(x_bar)[0], dense_pullback(x_bar)[0], atol=1e-4)


def test_metric_from_state_space_matern_cg_and_preconditioner_smoke():
    # cg exercises metric.solve under jax.linear_transpose, and the
    # Sherman-Morrison preconditioner built from metric.solve is the
    # downstream consumption shape.
    n = 8
    t = jnp.arange(n) * 1.0
    metric = metric_from_state_space(t, *matern_state_space(1.3, 0.8, 1.5), nugget=1e-6)

    A = jnp.stack([jnp.ones(n), jnp.arange(n, dtype=jnp.float32)])

    def residual(theta, _, p):
        return A @ theta - jnp.array([2.0, 1.0])

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-10,
        iterative_maxiter=50,
        metric=metric,
    )
    result = solver.solve(jnp.zeros(n), max_steps=60, atol=1e-5)
    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(A @ result.x, jnp.array([2.0, 1.0]), atol=1e-4)

    K = dense_matern_gram(t, 1.3, 0.8, 1.5) + 1e-6 * jnp.eye(n)
    u, weight = jnp.ones(n), 50.0
    preconditioner = sherman_morrison_preconditioner(metric.solve, u, weight)
    B = K + weight * jnp.outer(u, u)
    v = jax.random.normal(jax.random.PRNGKey(11), (n,))
    assert jnp.allclose(
        preconditioner(v, 0.0), jnp.linalg.solve(B, v), rtol=1e-3, atol=1e-3
    )


def shifted_composite_metric(kernel_block, n, k, eps):
    return blockdiag_metric(
        [(kernel_block, n), (metric_from_diagonal(eps * jnp.ones(k)), k)]
    )


def test_metric_from_shifted_matvec_matches_dense_metric():
    n, shift = 30, 0.5
    t = jnp.arange(n) * 1.0
    K = dense_matern_gram(t, 1.3, 0.8, 2.5)
    metric = metric_from_shifted_matvec(lambda x: K @ x, shift, tol=1e-6)
    x = jax.random.normal(jax.random.PRNGKey(20), (n,))
    X = jax.random.normal(jax.random.PRNGKey(21), (n, 3))

    with jax.default_matmul_precision("highest"):
        K_shifted = K + shift * jnp.eye(n)
        assert jnp.allclose(
            metric.solve(x), jnp.linalg.solve(K_shifted, x), rtol=1e-4, atol=1e-4
        )
        assert jnp.allclose(
            metric.solve(X), jnp.linalg.solve(K_shifted, X), rtol=1e-4, atol=1e-4
        )
        assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ K_shifted @ x), rtol=1e-4)
        # The default tolerance (sqrt of machine eps) also solves correctly,
        # just less tightly.
        default_metric = metric_from_shifted_matvec(lambda v: K @ v, shift)
        assert jnp.allclose(
            default_metric.solve(x), jnp.linalg.solve(K_shifted, x), rtol=1e-2
        )
    assert metric.inv_sqrt is None
    assert metric.inv_sqrt_transpose is None


def test_metric_from_shifted_matvec_solves_ill_conditioned_kernel():
    # Nugget-free fine-grid Matern-5/2 Gram: numerically singular K, but the
    # shift floors the spectrum so the iterative solve still matches the
    # dense shifted factorization -- the floor is the point.
    n, shift = 120, 1e-2
    t = 0.05 * jnp.arange(n)
    K = dense_matern_gram(t, 1.0, 0.8, 2.5)
    metric = metric_from_shifted_matvec(lambda x: K @ x, shift, tol=1e-6)
    x = jax.random.normal(jax.random.PRNGKey(22), (n,))

    with jax.default_matmul_precision("highest"):
        expected = jnp.linalg.solve(K + shift * jnp.eye(n), x)
        assert jnp.allclose(metric.solve(x), expected, rtol=1e-2, atol=1e-3)


def test_metric_from_shifted_matvec_validation_and_solver_requirements():
    for bad_shift in (0.0, -1.0):
        with pytest.raises(ValueError, match="shift"):
            metric_from_shifted_matvec(lambda x: x, bad_shift)
    with pytest.raises(ValueError, match="tol"):
        metric_from_shifted_matvec(lambda x: x, 1.0, tol=-1.0)
    with pytest.raises(ValueError, match="atol"):
        metric_from_shifted_matvec(lambda x: x, 1.0, atol=-1.0)
    with pytest.raises(ValueError, match="maxiter"):
        metric_from_shifted_matvec(lambda x: x, 1.0, maxiter=0)

    metric = metric_from_shifted_matvec(lambda x: 2.0 * x, 1.0)
    with pytest.raises(ValueError, match="inv_sqrt"):
        LevenbergMarquardt(residual_fn, linear_solver="qr", metric=metric)
    LevenbergMarquardt(residual_fn, metric=metric)
    LevenbergMarquardt(
        residual_fn,
        linear_solver="cg",
        metric=metric,
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
    )
    # norm is provided, so geodesic acceleration accepts this metric.
    LevenbergMarquardt(residual_fn, metric=metric, geodesic_acceleration=True)


@pytest.mark.parametrize("kernel_block", ["dense", "state_space", "matvec"])
def test_shifted_blockdiag_metric_matches_dense_metric(kernel_block):
    # The three representations of blockdiag(K, 0) + eps I are
    # interchangeable and match the dense factorization of the full M.
    n, k, eps = 24, 2, 1e-2
    t = jnp.arange(n) * 1.0
    sigma, ell = 1.3, 0.8
    K = dense_matern_gram(t, sigma, ell, 2.5)
    if kernel_block == "dense":
        block = metric_from_cholesky(jnp.linalg.cholesky(K + eps * jnp.eye(n)))
    elif kernel_block == "state_space":
        block = metric_from_state_space(
            t, *matern_state_space(sigma, ell, 2.5), nugget=eps
        )
    else:
        block = metric_from_shifted_matvec(lambda x: K @ x, eps, tol=1e-6)
    metric = shifted_composite_metric(block, n, k, eps)

    M = jnp.block(
        [
            [K + eps * jnp.eye(n), jnp.zeros((n, k))],
            [jnp.zeros((k, n)), eps * jnp.eye(k)],
        ]
    )
    x = jax.random.normal(jax.random.PRNGKey(23), (n + k,))
    X = jax.random.normal(jax.random.PRNGKey(24), (n + k, 3))
    with jax.default_matmul_precision("highest"):
        assert jnp.allclose(
            metric.solve(x), jnp.linalg.solve(M, x), rtol=1e-3, atol=1e-4
        )
        assert jnp.allclose(
            metric.solve(X), jnp.linalg.solve(M, X), rtol=1e-3, atol=1e-4
        )
        assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ M @ x), rtol=1e-4)
        if kernel_block == "matvec":
            assert metric.inv_sqrt is None
            assert metric.inv_sqrt_transpose is None
        else:
            S = metric.inv_sqrt(jnp.eye(n + k))
            assert jnp.allclose(S @ S.T, jnp.linalg.inv(M), rtol=1e-3, atol=1e-4)


SHIFTED_STEP_CASES = [
    ("dense", "cholesky"),
    ("dense", "qr"),
    ("dense", "cg"),
    ("state_space", "cholesky"),
    ("state_space", "qr"),
    ("state_space", "cg"),
    ("matvec", "cholesky"),
    ("matvec", "cg"),
]


@pytest.mark.parametrize("kernel_block,linear_solver", SHIFTED_STEP_CASES)
def test_shifted_metric_step_matches_closed_form_across_solvers(
    kernel_block, linear_solver
):
    n, k, eps = 12, 2, 1e-2
    t = jnp.arange(n) * 1.0
    sigma, ell = 1.3, 0.8
    K = dense_matern_gram(t, sigma, ell, 2.5)
    if kernel_block == "dense":
        block = metric_from_cholesky(jnp.linalg.cholesky(K + eps * jnp.eye(n)))
    elif kernel_block == "state_space":
        block = metric_from_state_space(
            t, *matern_state_space(sigma, ell, 2.5), nugget=eps
        )
    else:
        block = metric_from_shifted_matvec(lambda x: K @ x, eps, tol=1e-8)
    metric = shifted_composite_metric(block, n, k, eps)
    M = jnp.block(
        [
            [K + eps * jnp.eye(n), jnp.zeros((n, k))],
            [jnp.zeros((k, n)), eps * jnp.eye(k)],
        ]
    )

    A = jax.random.normal(jax.random.PRNGKey(25), (3, n + k))
    b = jnp.array([1.0, -0.5, 2.0])

    def residual(theta, _, p):
        return A @ theta - b

    damping = 1e-2
    preconditioner_kwargs = (
        {
            "dual_preconditioner": identity_preconditioner(),
            "implicit_preconditioner": identity_preconditioner(),
        }
        if linear_solver == "cg"
        else {}
    )
    solver = LevenbergMarquardt(
        residual,
        init_damping=damping,
        linear_solver=linear_solver,
        geodesic_acceleration=False,
        metric=metric,
        iterative_tol=1e-8,
        iterative_maxiter=500,
        **preconditioner_kwargs,
    )
    x0 = jnp.zeros(n + k)
    x1, _, info = solver.update(x0, solver.init(x0, None))
    assert bool(info.accepted)

    with jax.default_matmul_precision("highest"):
        step = jnp.linalg.solve(A.T @ A + damping * M, A.T @ b)
    assert jnp.allclose(x1, step, rtol=1e-3, atol=1e-4)


def test_shifted_metric_implicit_jvp_and_vjp_match_dense():
    # The matvec composite drives solve() and its implicit JVP/VJP to the
    # same answers as the dense factorization of the full M -- exercising
    # the inner CG on the (n, m) Jt inside the implicit rule and its
    # differentiation.
    n, k, eps = 10, 2, 1e-2
    t = jnp.arange(n) * 1.0
    K = dense_matern_gram(t, 1.3, 0.8, 2.5)
    composite = shifted_composite_metric(
        metric_from_shifted_matvec(lambda x: K @ x, eps, tol=1e-8), n, k, eps
    )
    M = jnp.block(
        [
            [K + eps * jnp.eye(n), jnp.zeros((n, k))],
            [jnp.zeros((k, n)), eps * jnp.eye(k)],
        ]
    )
    dense = metric_from_cholesky(jnp.linalg.cholesky(M))

    A = jax.random.normal(jax.random.PRNGKey(26), (3, n + k))

    def residual(theta, _, p):
        return A @ theta - jnp.array([p, 0.5 * p, -p])

    def solved_x(metric, linear_solver, p):
        preconditioner_kwargs = (
            {
                "dual_preconditioner": identity_preconditioner(),
                "implicit_preconditioner": identity_preconditioner(),
            }
            if linear_solver == "cg"
            else {}
        )
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-2,
            linear_solver=linear_solver,
            metric=metric,
            iterative_tol=1e-8,
            iterative_maxiter=500,
            **preconditioner_kwargs,
        )
        return solver.solve(jnp.zeros(n + k), p=p, max_steps=80, atol=1e-6).x

    p, p_dot = jnp.asarray(2.0), jnp.asarray(1.0)
    x_dense, x_dense_dot = jax.jvp(
        lambda q: solved_x(dense, "cholesky", q), (p,), (p_dot,)
    )
    for linear_solver in ("cholesky", "cg"):
        x_m, x_m_dot = jax.jvp(
            lambda q, ls=linear_solver: solved_x(composite, ls, q), (p,), (p_dot,)
        )
        assert jnp.allclose(x_m, x_dense, atol=1e-4)
        assert jnp.allclose(x_m_dot, x_dense_dot, atol=1e-4)

        x_bar = jnp.linspace(-1.0, 1.0, n + k)
        _, pull_m = jax.vjp(lambda q, ls=linear_solver: solved_x(composite, ls, q), p)
        _, pull_d = jax.vjp(lambda q: solved_x(dense, "cholesky", q), p)
        assert jnp.allclose(pull_m(x_bar)[0], pull_d(x_bar)[0], atol=1e-4)


def test_metric_from_shifted_matvec_preconditioner_passthrough():
    # An exact inverse as the inner preconditioner must give the same solve
    # (pins the M= plumbing into the inner CG).
    n, shift = 20, 0.5
    t = jnp.arange(n) * 1.0
    K = dense_matern_gram(t, 1.3, 0.8, 1.5)
    K_shifted_inv = jnp.linalg.inv(K + shift * jnp.eye(n))
    plain = metric_from_shifted_matvec(lambda x: K @ x, shift, tol=1e-6)
    preconditioned = metric_from_shifted_matvec(
        lambda x: K @ x, shift, tol=1e-6, preconditioner=lambda v: K_shifted_inv @ v
    )
    x = jax.random.normal(jax.random.PRNGKey(27), (n,))
    assert jnp.allclose(preconditioned.solve(x), plain.solve(x), rtol=1e-4, atol=1e-5)


def test_woodbury_preconditioner_matches_dense_inverse():
    n, k = 12, 2
    idx = jnp.arange(n)
    A = 0.6 ** jnp.abs(idx[:, None] - idx[None, :])
    solve = metric_from_cholesky(jnp.linalg.cholesky(A)).solve
    U = jax.random.normal(jax.random.PRNGKey(28), (n, k))
    weights = jnp.array([50.0, 20.0])

    preconditioner = woodbury_preconditioner(solve, U, weights)
    B = A + U @ jnp.diag(weights) @ U.T
    v = jax.random.normal(jax.random.PRNGKey(29), (n,))
    assert jnp.allclose(
        preconditioner(v, 0.0), jnp.linalg.solve(B, v), rtol=1e-3, atol=1e-3
    )

    # k=1 reduces exactly to Sherman-Morrison.
    rank1 = woodbury_preconditioner(solve, U[:, :1], weights[:1])
    sherman = sherman_morrison_preconditioner(solve, U[:, 0], weights[0])
    assert jnp.allclose(rank1(v, 0.0), sherman(v, 0.0), rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="U must"):
        woodbury_preconditioner(solve, U[:, 0], weights)


def test_cg_with_woodbury_spike_preconditioner_matches_cholesky_step():
    # Unified-eps metric: the scalar block injects the exactly known rank-k
    # spike (1/eps) J_beta J_beta' into the dual operator; preconditioning
    # the cg linear_solver with its Woodbury inverse never changes the
    # subproblem, so the step matches cholesky.
    n, k, eps = 16, 2, 1e-3
    t = jnp.arange(n) * 1.0
    K = dense_matern_gram(t, 1.3, 0.8, 1.5)
    K_shifted = K + eps * jnp.eye(n)
    metric = shifted_composite_metric(
        metric_from_cholesky(jnp.linalg.cholesky(K_shifted)), n, k, eps
    )

    m = 8
    J_alpha = jax.random.normal(jax.random.PRNGKey(30), (m, n))
    J_beta = jax.random.normal(jax.random.PRNGKey(31), (m, k))
    A = jnp.concatenate([J_alpha, J_beta], axis=1)
    b = jax.random.normal(jax.random.PRNGKey(32), (m,))

    def residual(theta, _, p):
        return A @ theta - b

    # Base = the kernel part of the dual operator, spike = the scalar block.
    base = J_alpha @ jnp.linalg.solve(K_shifted, J_alpha.T)
    base_solve = metric_from_cholesky(jnp.linalg.cholesky(base)).solve
    dual_preconditioner = woodbury_preconditioner(
        base_solve, J_beta, (1.0 / eps) * jnp.ones(k)
    )

    common = dict(init_damping=1e-2, geodesic_acceleration=False, metric=metric)
    cg_solver = LevenbergMarquardt(
        residual,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-8,
        iterative_maxiter=200,
        dual_preconditioner=dual_preconditioner,
        **common,
    )
    cholesky_solver = LevenbergMarquardt(residual, linear_solver="cholesky", **common)

    x0 = jnp.zeros(n + k)
    x_cg, _, info_cg = cg_solver.update(x0, cg_solver.init(x0, None))
    x_ch, _, info_ch = cholesky_solver.update(x0, cholesky_solver.init(x0, None))
    assert bool(info_cg.accepted) and bool(info_ch.accepted)
    # The 1/eps spike puts the dual condition number near the float32
    # attainable-residual floor, so the two paths agree only to ~1e-3 and
    # the exact gap varies across BLAS/SIMD variants.
    assert jnp.allclose(x_cg, x_ch, rtol=1e-2, atol=1e-3)


def test_nystrom_preconditioner_full_rank_matches_dense_solve():
    n = 12
    G = jax.random.normal(jax.random.PRNGKey(37), (n, n))
    A = G @ G.T + 0.5 * jnp.eye(n)
    preconditioner = nystrom_preconditioner(
        lambda X: A @ X, n, n, jax.random.PRNGKey(38)
    )
    v = jax.random.normal(jax.random.PRNGKey(39), (n,))

    # rank = n makes the Nystrom approximation exact, so the apply is the
    # exact damped inverse; damping=0 is valid because A is positive definite.
    for damping in (0.0, 0.5):
        expected = jnp.linalg.solve(A + damping * jnp.eye(n), v)
        assert jnp.allclose(preconditioner(v, damping), expected, rtol=1e-3, atol=1e-4)


def test_nystrom_preconditioner_exact_for_low_rank_operator():
    # A rank-r PSD operator sketched at rank >= r is recovered exactly, so
    # the apply matches the dense damped inverse.
    n, r = 14, 4
    W = jax.random.normal(jax.random.PRNGKey(40), (n, r))
    A = W @ W.T
    preconditioner = nystrom_preconditioner(
        lambda X: A @ X, n, 6, jax.random.PRNGKey(41)
    )
    v = jax.random.normal(jax.random.PRNGKey(42), (n,))
    damping = 0.3

    expected = jnp.linalg.solve(A + damping * jnp.eye(n), v)
    assert jnp.allclose(preconditioner(v, damping), expected, rtol=1e-3, atol=1e-4)


def test_nystrom_preconditioner_matches_ftu_formula():
    # Replicate the construction with the same key and assemble the FTU
    # apply densely: the unresolved complement must be balanced at
    # rho + damping (rho the smallest retained eigenvalue), not at the
    # bare damping.
    n, rank = 16, 6
    G = jax.random.normal(jax.random.PRNGKey(43), (n, n))
    A = G @ G.T / n + jnp.eye(n)
    key = jax.random.PRNGKey(44)
    preconditioner = nystrom_preconditioner(lambda X: A @ X, n, rank, key)

    Omega = jnp.linalg.qr(jax.random.normal(key, (n, rank)))[0]
    Y = A @ Omega
    finfo = jnp.finfo(Y.dtype)
    nu = jnp.maximum(finfo.eps * jnp.linalg.norm(Y), finfo.tiny / finfo.eps)
    Y_nu = Y + nu * Omega
    core = Omega.T @ Y_nu
    L = jnp.linalg.cholesky(0.5 * (core + core.T))
    B = jsp_linalg.solve_triangular(L, Y_nu.T, lower=True).T
    U, sigma, _ = jnp.linalg.svd(B, full_matrices=False)
    lam = jnp.maximum(sigma**2 - nu, 0.0)
    rho = lam[-1]
    damping = 0.25
    v = jax.random.normal(jax.random.PRNGKey(45), (n,))

    ftu = U @ ((U.T @ v) / (lam + damping)) + (v - U @ (U.T @ v)) / (rho + damping)
    assert jnp.allclose(preconditioner(v, damping), ftu, rtol=1e-5, atol=1e-6)
    # rho sits strictly inside the spectrum here (A is positive definite),
    # so the FTU complement genuinely differs from a sketch-and-solve
    # inverse that would divide the complement by the bare damping.
    assert float(rho) > damping
    naive = U @ ((U.T @ v) / (lam + damping)) + (v - U @ (U.T @ v)) / damping
    assert not jnp.allclose(preconditioner(v, damping), naive, rtol=1e-3)


def test_nystrom_preconditioner_symmetry_definiteness_and_key_determinism():
    n, rank = 10, 4
    G = jax.random.normal(jax.random.PRNGKey(46), (n, n))
    A = G @ G.T / n + 0.5 * jnp.eye(n)
    key = jax.random.PRNGKey(47)
    preconditioner = nystrom_preconditioner(lambda X: A @ X, n, rank, key)
    x = jax.random.normal(jax.random.PRNGKey(48), (n,))
    y = jax.random.normal(jax.random.PRNGKey(49), (n,))
    damping = 0.1

    assert jnp.allclose(
        x @ preconditioner(y, damping),
        y @ preconditioner(x, damping),
        rtol=1e-4,
        atol=1e-5,
    )
    assert float(x @ preconditioner(x, damping)) > 0.0
    same = nystrom_preconditioner(lambda X: A @ X, n, rank, key)
    assert jnp.allclose(preconditioner(x, damping), same(x, damping))
    different = nystrom_preconditioner(lambda X: A @ X, n, rank, jax.random.PRNGKey(50))
    assert not jnp.allclose(preconditioner(x, damping), different(x, damping))


def test_nystrom_preconditioner_validation_and_zero_operator():
    def matvec(X):
        return X

    key = jax.random.PRNGKey(51)
    with pytest.raises(ValueError, match="rank must"):
        nystrom_preconditioner(matvec, 4, 0, key)
    with pytest.raises(ValueError, match="rank must"):
        nystrom_preconditioner(matvec, 4, 5, key)
    with pytest.raises(ValueError, match="rank must"):
        nystrom_preconditioner(matvec, 4, 2.0, key)
    with pytest.raises(ValueError, match="n must"):
        nystrom_preconditioner(matvec, 0, 1, key)

    # The tiny floor on the stabilization shift keeps a zero operator's
    # build and apply finite.
    zero = nystrom_preconditioner(lambda X: 0.0 * X, 4, 2, key)
    assert bool(jnp.all(jnp.isfinite(zero(jnp.ones(4), 1.0))))


def test_cg_nystrom_preconditioner_enables_ill_conditioned_convergence():
    # Identity-metric mirror of the kernel-preconditioner test above: the
    # residual K x - b makes the dual operator K^2 + damping I,
    # ill-conditioned like cond(K)^2. At a tight inner budget
    # identity-preconditioned CG stalls; the Nystrom preconditioner built
    # from the K^2 matvec recovers Gauss-Newton-quality steps and converges.
    n = 40
    rho = 0.9
    idx = jnp.arange(n)
    K = rho ** jnp.abs(idx[:, None] - idx[None, :])
    x_true = jnp.sin(idx / 3.0)
    b = K @ x_true

    def residual(x):
        return K @ x - b

    # rank 16 of 40: the sketch resolves only the decaying head of the K^2
    # spectrum (the advertised low-rank regime), and the FTU complement
    # balance carries the rest.
    nystrom = nystrom_preconditioner(
        lambda V: K @ (K @ V), n, 16, jax.random.PRNGKey(52)
    )
    common = dict(
        init_damping=1e-6,
        linear_solver="cg",
        iterative_maxiter=3,
        implicit_preconditioner=identity_preconditioner(),
    )
    plain = LevenbergMarquardt(
        residual, dual_preconditioner=identity_preconditioner(), **common
    )
    preconditioned = LevenbergMarquardt(residual, dual_preconditioner=nystrom, **common)
    x0 = jnp.zeros(n)

    plain_result = plain.solve(x0, max_steps=20, atol=1e-3)
    preconditioned_result = preconditioned.solve(x0, max_steps=20, atol=1e-3)

    assert int(preconditioned_result.status) == LMStatus.CONVERGED
    assert int(plain_result.status) != LMStatus.CONVERGED


def test_cg_nystrom_mlp_ntk_example():
    # Copyable example: pure-jax MLP least squares under the identity
    # metric (n_params >> m collocation residuals), with the CG dual
    # preconditioner built by sketching the m x m empirical NTK Gram J J'
    # at the initial parameters via jax.linearize / jax.linear_transpose.
    m, width = 10, 16
    ts = jnp.linspace(-1.0, 1.0, m)
    targets = jnp.sin(jnp.pi * ts)

    keys = jax.random.split(jax.random.PRNGKey(53), 3)
    x0 = {
        "w1": jax.random.normal(keys[0], (1, width)),
        "b1": jnp.zeros(width),
        "w2": jax.random.normal(keys[1], (width, width)) / jnp.sqrt(width),
        "b2": jnp.zeros(width),
        "w3": jax.random.normal(keys[2], (width, 1)) / jnp.sqrt(width),
        "b3": jnp.zeros(1),
    }

    def mlp(params, t):
        h = jnp.tanh(t[:, None] @ params["w1"] + params["b1"])
        h = jnp.tanh(h @ params["w2"] + params["b2"])
        return (h @ params["w3"] + params["b3"]).ravel()

    def residual(params):
        return mlp(params, ts) - targets

    theta0, unravel = ravel_pytree(x0)
    _, jvp_fn = jax.linearize(lambda th: residual(unravel(th)), theta0)
    transpose_fn = jax.linear_transpose(jvp_fn, theta0)

    def ntk_matvec(V):
        # V is (m, k); apply J (J' v) column by column, frozen at x0.
        return jax.vmap(
            lambda col: jvp_fn(transpose_fn(col)[0]), in_axes=1, out_axes=1
        )(V)

    cg_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="cg",
        iterative_tol=1e-6,
        iterative_maxiter=20,
        # rank 6 of m=10: the NTK spectrum decays fast enough that a low-rank
        # sketch of its head preconditions the whole solve.
        dual_preconditioner=nystrom_preconditioner(
            ntk_matvec, m, 6, jax.random.PRNGKey(54)
        ),
        implicit_preconditioner=identity_preconditioner(),
    )
    cholesky_solver = LevenbergMarquardt(residual, init_damping=1e-2)

    cg_result = cg_solver.solve(x0, max_steps=100, atol=1e-3)
    cholesky_result = cholesky_solver.solve(x0, max_steps=100, atol=1e-3)

    assert int(cg_result.status) == LMStatus.CONVERGED
    assert int(cholesky_result.status) == LMStatus.CONVERGED
    # Compare model outputs, not raw parameters: an underdetermined network
    # has many interpolating parameter roots.
    assert jnp.allclose(mlp(cg_result.x, ts), targets, atol=2e-3)
    assert jnp.allclose(mlp(cg_result.x, ts), mlp(cholesky_result.x, ts), atol=2e-3)


def test_padded_zero_residual_cholesky_matches_unpadded():
    # Fixed-residual-shape pattern: appending exact-zero residual entries
    # (zero Jacobian rows) adds a decoupled damping*I block to the dual, so
    # the cholesky step is unchanged.
    m, n, pad = 12, 30, 4
    A = jax.random.normal(jax.random.PRNGKey(55), (m, n))
    b = jax.random.normal(jax.random.PRNGKey(56), (m,))
    init_damping = 1e-6

    def residual(theta):
        return A @ theta - b

    def residual_padded(theta):
        return jnp.concatenate((A @ theta - b, jnp.zeros(pad)))

    plain = LevenbergMarquardt(
        residual, init_damping=init_damping, geodesic_acceleration=False
    )
    padded = LevenbergMarquardt(
        residual_padded, init_damping=init_damping, geodesic_acceleration=False
    )
    theta0 = jnp.zeros(n)
    x_plain, _, info_plain = plain.update(theta0, plain.init(theta0))
    x_padded, _, info_padded = padded.update(theta0, padded.init(theta0))

    # Dual (residual-space) form of the damped step: well-conditioned in
    # float32, unlike the equivalent n x n primal solve.
    expected_step = A.T @ jnp.linalg.solve(A @ A.T + init_damping * jnp.eye(m), b)
    assert bool(info_plain.accepted) and bool(info_padded.accepted)
    assert jnp.allclose(x_padded, x_plain, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(x_padded, expected_step, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(
        info_padded.loss_candidate, info_plain.loss_candidate, rtol=1e-5
    )
    assert jnp.allclose(info_padded.grad_norm, info_plain.grad_norm, rtol=1e-5)


def test_padded_zero_residual_geodesic_matches_unpadded():
    # Nonlinear endgame with geodesic acceleration: the second-order solve
    # reuses the same factorization, so padding must not change the
    # accelerated step either.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.9, "b": -0.95}
    pad = 3

    def residual_padded(x, args, p):
        return jnp.concatenate((residual_fn(x, args, p), jnp.zeros(pad)))

    kwargs = dict(
        init_damping=1e-4,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    plain = LevenbergMarquardt(residual_fn, **kwargs)
    padded = LevenbergMarquardt(residual_padded, **kwargs)

    x_plain, _, info_plain = plain.update(x, plain.init(x, (ts, ys)), (ts, ys))
    x_padded, _, info_padded = padded.update(x, padded.init(x, (ts, ys)), (ts, ys))

    # The two dual factorizations have different sizes (m vs m + pad), so
    # float32 agreement is to solver noise, not bitwise.
    assert bool(info_plain.accepted) and bool(info_padded.accepted)
    assert bool(info_padded.used_geodesic) == bool(info_plain.used_geodesic)
    assert jnp.allclose(x_padded["a"], x_plain["a"], rtol=1e-4, atol=1e-5)
    assert jnp.allclose(x_padded["b"], x_plain["b"], rtol=1e-4, atol=1e-5)
    assert jnp.allclose(
        info_padded.acceleration_ratio,
        info_plain.acceleration_ratio,
        rtol=1e-3,
        atol=1e-4,
    )


def test_padded_zero_residual_cg_with_padded_preconditioner():
    # The dual operator of a padded problem is blockdiag(J J' + damping I,
    # damping I); pad_dual_preconditioner extends an exact base inverse with
    # the exact 1/damping pad block, so a one-iteration CG budget reproduces
    # the cholesky step.
    m, n, pad = 10, 24, 3
    A = jax.random.normal(jax.random.PRNGKey(57), (m, n))
    b = jax.random.normal(jax.random.PRNGKey(58), (m,))
    gram = A @ A.T
    init_damping = 1e-2

    def residual_padded(theta):
        return jnp.concatenate((A @ theta - b, jnp.zeros(pad)))

    def base_preconditioner(v, damping):
        return jnp.linalg.solve(gram + damping * jnp.eye(m), v)

    cg_solver = LevenbergMarquardt(
        residual_padded,
        init_damping=init_damping,
        linear_solver="cg",
        iterative_tol=0.0,
        iterative_maxiter=1,
        dual_preconditioner=pad_dual_preconditioner(base_preconditioner, m),
        implicit_preconditioner=identity_preconditioner(),
        geodesic_acceleration=False,
    )
    cholesky_solver = LevenbergMarquardt(
        residual_padded, init_damping=init_damping, geodesic_acceleration=False
    )
    theta0 = jnp.zeros(n)

    x_cg, _, info_cg = cg_solver.update(theta0, cg_solver.init(theta0))
    x_ch, _, info_ch = cholesky_solver.update(theta0, cholesky_solver.init(theta0))

    assert bool(info_cg.accepted) and bool(info_ch.accepted)
    assert jnp.allclose(x_cg, x_ch, rtol=1e-4, atol=1e-5)

    # A shape-fixed base preconditioner used unwrapped is invalid on the
    # padded residual space and fails at trace time.
    mismatched = LevenbergMarquardt(
        residual_padded,
        init_damping=init_damping,
        linear_solver="cg",
        iterative_tol=0.0,
        iterative_maxiter=1,
        dual_preconditioner=base_preconditioner,
        implicit_preconditioner=identity_preconditioner(),
        geodesic_acceleration=False,
    )
    with pytest.raises(ValueError, match="inconsistent size"):
        mismatched.update(theta0, mismatched.init(theta0))


def test_pad_dual_preconditioner_validation():
    with pytest.raises(ValueError, match="n_real must"):
        pad_dual_preconditioner(lambda v, damping: v, 0)
    with pytest.raises(ValueError, match="n_real must"):
        pad_dual_preconditioner(lambda v, damping: v, 2.0)
    # A vector shorter than n_real would silently clip through a
    # shape-generic base; the callback rejects it at trace time instead.
    padded = pad_dual_preconditioner(identity_preconditioner(), 4)
    with pytest.raises(ValueError, match="at least"):
        padded(jnp.ones(3), 0.1)
    assert jnp.allclose(padded(jnp.ones(6), 0.5), jnp.array([1.0] * 4 + [2.0] * 2))


def test_padded_zero_residual_qr_is_rank_deficient():
    # Padded zero rows make the Jacobian rank-deficient, which the qr
    # path's triangular solves cannot handle: the padded step is
    # non-finite where the unpadded one is fine.
    m, n, pad = 12, 30, 4
    A = jax.random.normal(jax.random.PRNGKey(60), (m, n))
    b = jax.random.normal(jax.random.PRNGKey(61), (m,))

    def residual_padded(theta):
        return jnp.concatenate((A @ theta - b, jnp.zeros(pad)))

    solver = LevenbergMarquardt(
        residual_padded,
        init_damping=1e-4,
        linear_solver="qr",
        geodesic_acceleration=False,
    )
    theta0 = jnp.zeros(n)
    x_padded, _, info = solver.update(theta0, solver.init(theta0))
    assert not bool(jnp.all(jnp.isfinite(x_padded))) or not bool(
        jnp.isfinite(info.loss_candidate)
    )


def test_padded_zero_residual_implicit_ad_is_singular_by_design():
    # Padded rows are zero Jacobian rows, so the UNDAMPED implicit dual
    # J P J' is singular. With implicit_penalty=0.0 the dense rule keeps the
    # loud unregularized contract (non-finite tangent); the default eps *
    # trace ridge resolves the padding -- a consistent singularity -- to the
    # minimum-metric-norm tangent of the unpadded formulation, with no
    # padding-aware special casing.
    A = jax.random.normal(jax.random.PRNGKey(59), (3, 8))

    def residual(theta, _, p):
        return A @ theta - p

    def residual_padded(theta, _, p):
        return jnp.concatenate((A @ theta - p, jnp.zeros(2)))

    plain = LevenbergMarquardt(residual, init_damping=1e-2, geodesic_acceleration=False)
    padded_loud = LevenbergMarquardt(
        residual_padded,
        init_damping=1e-2,
        geodesic_acceleration=False,
        implicit_penalty=0.0,
    )
    padded_default = LevenbergMarquardt(
        residual_padded, init_damping=1e-2, geodesic_acceleration=False
    )

    def solved_x(solver, p):
        return solver.solve(jnp.zeros(8), p=p, max_steps=40, atol=1e-5).x

    p0 = jnp.array([1.0, -0.5, 0.25])
    p_dot = jnp.ones(3)
    x_plain, x_plain_dot = jax.jvp(lambda p: solved_x(plain, p), (p0,), (p_dot,))
    x_padded, x_dot = jax.jvp(lambda p: solved_x(padded_loud, p), (p0,), (p_dot,))
    # The forward padded solve is healthy and matches the unpadded solution;
    # only the unregularized implicit tangent is non-finite.
    assert bool(jnp.all(jnp.isfinite(x_padded)))
    assert jnp.allclose(x_padded, x_plain, atol=1e-4)
    assert not bool(jnp.all(jnp.isfinite(x_dot)))
    # The default ridge recovers the unpadded implicit tangent.
    _, x_dot_default = jax.jvp(lambda p: solved_x(padded_default, p), (p0,), (p_dot,))
    assert bool(jnp.all(jnp.isfinite(x_dot_default)))
    assert jnp.allclose(x_dot_default, x_plain_dot, atol=1e-4)


def test_implicit_cg_with_shifted_matvec_metric_matches_dense():
    # End-to-end matrix-free implicit AD: the cg implicit rule applies
    # metric.solve to tangent-dependent data, and the VJP handles that
    # through the self-adjoint declaration -- the cotangent pass
    # re-EVALUATES the shifted-matvec metric's inner CG rather than
    # transposing it (which JAX cannot do). Pin both derivative modes
    # against the dense metric + dense implicit rule.
    n, k, eps = 10, 2, 1e-2
    t = jnp.arange(n) * 1.0
    K = dense_matern_gram(t, 1.3, 0.8, 2.5)
    composite = shifted_composite_metric(
        metric_from_shifted_matvec(lambda x: K @ x, eps, tol=1e-8), n, k, eps
    )
    M = jnp.block(
        [
            [K + eps * jnp.eye(n), jnp.zeros((n, k))],
            [jnp.zeros((k, n)), eps * jnp.eye(k)],
        ]
    )
    dense = metric_from_cholesky(jnp.linalg.cholesky(M))

    A = jax.random.normal(jax.random.PRNGKey(33), (3, n + k))

    def residual(theta, _, p):
        return A @ theta - jnp.array([p, 0.5 * p, -p])

    def solved_x(metric, linear_solver, implicit_solver, p):
        preconditioner_kwargs = {}
        if linear_solver == "cg":
            preconditioner_kwargs["dual_preconditioner"] = identity_preconditioner()
        if implicit_solver == "cg":
            preconditioner_kwargs["implicit_preconditioner"] = identity_preconditioner()
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-2,
            linear_solver=linear_solver,
            implicit_solver=implicit_solver,
            implicit_tol=1e-8,
            metric=metric,
            iterative_tol=1e-8,
            iterative_maxiter=500,
            **preconditioner_kwargs,
        )
        return solver.solve(jnp.zeros(n + k), p=p, max_steps=80, atol=1e-6).x

    p, p_dot = jnp.asarray(2.0), jnp.asarray(1.0)
    x_bar = jnp.linspace(-1.0, 1.0, n + k)
    _, dense_dot = jax.jvp(
        lambda q: solved_x(dense, "cholesky", "cholesky", q), (p,), (p_dot,)
    )
    _, dense_pull = jax.vjp(lambda q: solved_x(dense, "cholesky", "cholesky", q), p)

    # cg implicit rule with the matvec metric, under both forward solvers
    # (cholesky forward + cg implicit is the forced-"cg" combination).
    for linear_solver in ("cholesky", "cg"):
        _, cg_dot = jax.jvp(
            lambda q, ls=linear_solver: solved_x(composite, ls, "cg", q),
            (p,),
            (p_dot,),
        )
        _, cg_pull = jax.vjp(
            lambda q, ls=linear_solver: solved_x(composite, ls, "cg", q), p
        )
        assert jnp.allclose(cg_dot, dense_dot, atol=1e-4)
        assert jnp.allclose(cg_pull(x_bar)[0], dense_pull(x_bar)[0], atol=1e-4)


def test_implicit_cg_woodbury_preconditioner_with_shifted_metric():
    # Under the unified shifted metric, the scalar block injects the rank-k
    # spike (1/eps) J_beta J_beta' into the UNDAMPED implicit dual operator
    # too. An exact Woodbury preconditioner (passed directly; the implicit
    # hook calls it with zero damping) makes a
    # one-iteration implicit CG budget reproduce the dense-rule derivative.
    n, k, eps = 12, 2, 1e-3
    t = jnp.arange(n) * 1.0
    K = dense_matern_gram(t, 1.3, 0.8, 1.5)
    K_shifted = K + eps * jnp.eye(n)
    metric = shifted_composite_metric(
        metric_from_cholesky(jnp.linalg.cholesky(K_shifted)), n, k, eps
    )

    m = 6
    J_alpha = jax.random.normal(jax.random.PRNGKey(34), (m, n))
    J_beta = jax.random.normal(jax.random.PRNGKey(35), (m, k))
    A = jnp.concatenate([J_alpha, J_beta], axis=1)

    def residual(theta, _, p):
        return A @ theta - p * jnp.linspace(1.0, 2.0, m)

    base = J_alpha @ jnp.linalg.solve(K_shifted, J_alpha.T)
    base_solve = metric_from_cholesky(jnp.linalg.cholesky(base)).solve
    spike_preconditioner = woodbury_preconditioner(
        base_solve, J_beta, (1.0 / eps) * jnp.ones(k)
    )

    def solved_x(implicit_kwargs, p):
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-2,
            linear_solver="cg",
            dual_preconditioner=identity_preconditioner(),
            iterative_tol=1e-8,
            iterative_maxiter=200,
            metric=metric,
            geodesic_acceleration=False,
            **implicit_kwargs,
        )
        return solver.solve(jnp.zeros(n + k), p=p, max_steps=80, atol=1e-6).x

    p, p_dot = jnp.asarray(1.0), jnp.asarray(1.0)
    _, dense_dot = jax.jvp(
        # implicit_penalty=0.0: the spiked metric puts 1/eps into the dual
        # trace, so the trace-scaled default ridge would bias this exact
        # dense reference.
        lambda q: solved_x({"implicit_solver": "cholesky", "implicit_penalty": 0.0}, q),
        (p,),
        (p_dot,),
    )
    _, spike_dot = jax.jvp(
        lambda q: solved_x(
            {
                "implicit_solver": "cg",
                "implicit_tol": 0.0,
                "implicit_maxiter": 1,
                # A (v, damping) helper passes directly; the implicit hook
                # calls it with zero damping.
                "implicit_preconditioner": spike_preconditioner,
            },
            q,
        ),
        (p,),
        (p_dot,),
    )
    assert jnp.allclose(spike_dot, dense_dot, rtol=1e-3, atol=1e-4)


def test_implicit_cg_vmap_and_hessian_match_dense():
    # jax.vmap over differentiated solves and vmap-based second derivatives
    # (jax.hessian) must compose with the cg implicit rule -- the
    # self-adjoint metric-inverse declaration is built on
    # custom_linear_solve, which has a batching rule (linear_call does not).
    n, k, eps = 8, 2, 1e-2
    t = jnp.arange(n) * 1.0
    K = dense_matern_gram(t, 1.3, 0.8, 2.5)
    composite = shifted_composite_metric(
        metric_from_shifted_matvec(lambda x: K @ x, eps, tol=1e-8), n, k, eps
    )
    M = jnp.block(
        [
            [K + eps * jnp.eye(n), jnp.zeros((n, k))],
            [jnp.zeros((k, n)), eps * jnp.eye(k)],
        ]
    )
    dense = metric_from_cholesky(jnp.linalg.cholesky(M))

    A = jax.random.normal(jax.random.PRNGKey(36), (3, n + k))

    def residual(theta, _, p):
        return A @ theta - jnp.array([p, 0.5 * p, -p])

    def solved_x(metric, implicit_solver, p):
        preconditioner_kwargs = (
            {"implicit_preconditioner": identity_preconditioner()}
            if implicit_solver == "cg"
            else {}
        )
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-2,
            linear_solver="cg",
            dual_preconditioner=identity_preconditioner(),
            implicit_solver=implicit_solver,
            implicit_tol=1e-8,
            metric=metric,
            iterative_tol=1e-8,
            iterative_maxiter=300,
            **preconditioner_kwargs,
        )
        return solver.solve(jnp.zeros(n + k), p=p, max_steps=60, atol=1e-6).x

    ps = jnp.array([1.0, 2.0, 3.0])
    one = jnp.asarray(1.0)
    _, dots_cg = jax.vmap(
        lambda q: jax.jvp(lambda r: solved_x(composite, "cg", r), (q,), (one,))
    )(ps)
    _, dots_dense = jax.vmap(
        lambda q: jax.jvp(lambda r: solved_x(dense, "cholesky", r), (q,), (one,))
    )(ps)
    assert jnp.allclose(dots_cg, dots_dense, atol=1e-4)

    def loss_cg(q):
        return jnp.sum(solved_x(composite, "cg", q) ** 2)

    def loss_dense(q):
        return jnp.sum(solved_x(dense, "cholesky", q) ** 2)

    assert jnp.allclose(
        jax.vmap(jax.grad(loss_cg))(ps), jax.vmap(jax.grad(loss_dense))(ps), atol=1e-3
    )
    assert jnp.allclose(
        jax.hessian(loss_cg)(jnp.asarray(2.0)),
        jax.hessian(loss_dense)(jnp.asarray(2.0)),
        rtol=1e-3,
    )


def test_implicit_cg_rank_deficient_dual_fails_loudly_by_default():
    # J P J' singular with an INCONSISTENT tangent right-hand side: the
    # unregularized dense rule (implicit_penalty=0.0) NaNs through cho_solve,
    # and the cg rule's run-to-tolerance default diverges to non-finite as
    # well. Only a small bounded implicit_maxiter (the exact-preconditioner
    # budget mode) returns a finite -- and wrong -- derivative, which is why
    # that mode is reserved for exact preconditioners. (No ridge can make an
    # inconsistent dual meaningful; the default ridge would return a finite,
    # penalty-inflated tangent here, which is the documented trade-off.)
    A = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    def residual(theta, _, p):
        return A @ theta - p

    def x_dot(implicit_solver, implicit_maxiter):
        preconditioner_kwargs = (
            {"implicit_preconditioner": identity_preconditioner()}
            if implicit_solver == "cg"
            else {}
        )
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-2,
            linear_solver="cg",
            dual_preconditioner=identity_preconditioner(),
            implicit_solver=implicit_solver,
            implicit_tol=1e-8,
            implicit_maxiter=implicit_maxiter,
            implicit_penalty=0.0,
            iterative_tol=1e-8,
            iterative_maxiter=100,
            geodesic_acceleration=False,
            **preconditioner_kwargs,
        )

        def solved_x(p):
            return solver.solve(jnp.zeros(5), p=p, max_steps=40, atol=1e-5).x

        p0 = jnp.array([1.0, 2.0, 3.0])
        p_dot = jnp.array([1.0, 0.0, 0.0])
        return jax.jvp(solved_x, (p0,), (p_dot,))[1]

    assert not bool(jnp.all(jnp.isfinite(x_dot("cholesky", None))))
    assert not bool(jnp.all(jnp.isfinite(x_dot("cg", None))))
    assert bool(jnp.all(jnp.isfinite(x_dot("cg", 1))))


@pytest.mark.parametrize("jit", [True, False])
def test_callback_returns_bare_lmstatus_members_without_casts(jit):
    # The spooky-shaped epoch callback: a lax.cond whose branches return bare
    # LMStatus members / weak values -- no .astype or dtype= casts. The solver
    # canonicalizes stop to bool and status to int32 at the boundary.
    def residual(theta, args, p):
        return theta - args

    def epoch_callback(ctx):
        def check(_):
            stop = ctx.info.loss < 1e-8
            status = jnp.where(stop, LMStatus.CONVERGED, LMStatus.RUNNING)
            return stop, status

        def keep_running(_):
            return jnp.asarray(False), jnp.asarray(LMStatus.RUNNING)

        stop, status = jax.lax.cond(ctx.step % 2 == 0, check, keep_running, None)
        return LMSolveAction(stop=stop, status=status)

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        max_steps=50,
        callback=epoch_callback,
        jit=jit,
    )
    assert result.status.dtype == jnp.int32
    assert int(result.status) == LMStatus.CONVERGED


def test_lmstatus_intenum_semantics():
    assert int(LMStatus.CONVERGED) == 1
    assert LMStatus(2) is LMStatus.MAX_STEPS
    assert LMStatus.MAX_STEPS.name == "MAX_STEPS"
    labels = {LMStatus.CONVERGED: "early_stopping_met"}
    assert labels.get(LMStatus(int(jnp.asarray(1)))) == "early_stopping_met"
    assert bool(jnp.asarray(1, dtype=jnp.int32) == LMStatus.CONVERGED)
