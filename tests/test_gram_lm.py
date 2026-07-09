import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import pytest

from nlls_gram import (
    LMSolveAction,
    LMState,
    LMStatus,
    Metric,
    UnderdeterminedLevenbergMarquardt,
    metric_from_cholesky,
    metric_from_tridiagonal_precision,
)


def residual_fn(x, args, p):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


REGRESSION_ATOL = 5e-5
REGRESSION_RTOL = 1e-5


def test_recovers_known_parameters_with_jitted_step():
    a_true, b_true = 2.0, -1.0
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = a_true * jnp.exp(b_true * ts)

    x = {"a": 1.0, "b": 0.0}
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
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

    default_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    identity_solver = UnderdeterminedLevenbergMarquardt(
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
    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=init_damping)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-4)
    new_x, _, info = solver.update(x, solver.init(x, (ts, ys)), (ts, ys))

    assert calls["count"] >= 2
    assert bool(info.accepted)
    assert jnp.allclose(new_x["a"], 2.0, atol=1e-4)


def test_defaults_enable_geodesic_and_jacobian_cache():
    solver = UnderdeterminedLevenbergMarquardt(lambda x: x)
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

    solver = UnderdeterminedLevenbergMarquardt(
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

    solver = UnderdeterminedLevenbergMarquardt(residual_fn)
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
        UnderdeterminedLevenbergMarquardt(residual_fn, linear_solver="svd")


def test_init_damping_must_be_positive():
    with pytest.raises(ValueError, match="init_damping must be positive"):
        UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=0.0)


def test_damping_update_factors_must_be_positive():
    with pytest.raises(ValueError, match="damping_decrease must be positive"):
        UnderdeterminedLevenbergMarquardt(residual_fn, damping_decrease=0.0)
    with pytest.raises(ValueError, match="damping_increase must be positive"):
        UnderdeterminedLevenbergMarquardt(residual_fn, damping_increase=0.0)


def test_iterative_options_must_be_valid():
    with pytest.raises(ValueError, match="iterative_tol must be nonnegative"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn, linear_solver="cg", iterative_tol=-1.0
        )
    with pytest.raises(ValueError, match="iterative_atol must be nonnegative"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn, linear_solver="cg", iterative_atol=-1.0
        )
    with pytest.raises(ValueError, match="iterative_maxiter must be positive or None"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn, linear_solver="cg", iterative_maxiter=0
        )
    with pytest.raises(ValueError, match="iterative_maxiter must be set"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn,
            linear_solver="cg",
            iterative_tol=0.0,
            iterative_atol=0.0,
            iterative_maxiter=None,
        )
    with pytest.raises(ValueError, match="lsmr_conlim must be positive"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn, linear_solver="lsmr", lsmr_conlim=0.0
        )


def test_default_float32_x_keeps_float32_outputs():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
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
        UnderdeterminedLevenbergMarquardt(
            residual_fn, init_damping=1e-2, max_damping=1e-3
        )


def test_metric_requirements_per_linear_solver():
    with pytest.raises(ValueError, match="metric.solve"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn, metric=Metric(norm=jnp.linalg.norm)
        )
    with pytest.raises(ValueError, match="metric.inv_sqrt"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn, linear_solver="qr", metric=Metric(solve=lambda x: x)
        )
    with pytest.raises(ValueError, match="metric.norm"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn,
            geodesic_acceleration=True,
            metric=Metric(solve=lambda x: x),
        )


def test_cg_step_matches_cholesky_identity_step():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    cholesky_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
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


def test_qr_step_matches_cholesky_identity_step():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
    )
    qr_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn, init_damping=1e-2, linear_solver="qr"
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


def test_lsmr_step_matches_closed_form_damped_linear_solution():
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]])
    target = jnp.array([1.0, 2.0, -1.0])
    theta0 = jnp.array([0.0, 0.0])
    init_damping = 0.1

    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=init_damping,
        linear_solver="lsmr",
        iterative_tol=1e-7,
        iterative_maxiter=20,
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


def test_qr_step_matches_closed_form_underdetermined_damped_solution():
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 2.0]])
    target = jnp.array([1.0, -2.0])
    theta0 = jnp.zeros(matrix.shape[1])
    init_damping = 0.1

    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=init_damping,
        linear_solver="qr",
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


def test_qr_float32_handles_ill_conditioned_case_where_cholesky_fails():
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 0.0, 0.0, 0.0], [1.0, 1e-4, 0.0, 0.0]])
    theta_true = jnp.array([1.0, 1.0, 0.0, 0.0])
    target = matrix @ theta_true
    theta0 = jnp.zeros(matrix.shape[1])

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-12,
        linear_solver="cholesky",
    )
    qr_solver = UnderdeterminedLevenbergMarquardt(
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
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
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


def test_qr_update_jits():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="qr",
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


def test_lsmr_update_jits():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="lsmr",
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


def test_geodesic_acceptance_ratio_zero_falls_back_to_velocity_step():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    solver = UnderdeterminedLevenbergMarquardt(
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

    solver = UnderdeterminedLevenbergMarquardt(
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

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="cg",
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

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    qr_solver = UnderdeterminedLevenbergMarquardt(
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


def test_lsmr_geodesic_acceleration_matches_cholesky():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    lsmr_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="lsmr",
        iterative_tol=1e-7,
        iterative_maxiter=20,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    cholesky_theta, _, cholesky_info = cholesky_solver.update(
        theta0, cholesky_solver.init(theta0, target), target
    )
    lsmr_theta, _, lsmr_info = lsmr_solver.update(
        theta0, lsmr_solver.init(theta0, target), target
    )

    assert bool(lsmr_info.accepted)
    assert bool(lsmr_info.used_geodesic)
    assert jnp.allclose(lsmr_theta, cholesky_theta, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(
        lsmr_info.acceleration_ratio,
        cholesky_info.acceleration_ratio,
        rtol=1e-6,
        atol=1e-6,
    )


def test_geodesic_acceleration_jits():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    solver = UnderdeterminedLevenbergMarquardt(
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
        solver = UnderdeterminedLevenbergMarquardt(
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
    solver = UnderdeterminedLevenbergMarquardt(
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

    identity_solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-9)
    x_identity, _, _ = identity_solver.update(theta0, identity_solver.init(theta0))
    assert jnp.allclose(x_identity, jnp.array([0.5, 0.5]), atol=1e-5)

    L = jnp.linalg.cholesky(jnp.diag(jnp.array([1.0, 4.0])))
    metric_solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-9, metric=metric_from_cholesky(L)
    )
    x_metric, _, _ = metric_solver.update(theta0, metric_solver.init(theta0))
    assert jnp.allclose(x_metric, jnp.array([0.8, 0.2]), atol=1e-5)


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "lsmr"])
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
    if linear_solver in ("cg", "lsmr"):
        solver_kwargs = {"iterative_tol": 1e-7, "iterative_maxiter": 30}

    solver = UnderdeterminedLevenbergMarquardt(
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
    solver = UnderdeterminedLevenbergMarquardt(
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


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "lsmr"])
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
    if linear_solver in ("cg", "lsmr"):
        solver_kwargs = {"iterative_tol": 1e-7, "iterative_maxiter": 30}
    solver = UnderdeterminedLevenbergMarquardt(
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
    solver = UnderdeterminedLevenbergMarquardt(
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
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)

    lm_state0 = solver.init(x, (ts, ys))
    _, lm_state1, _ = solver.update(x, lm_state0, (ts, ys))
    d0, d1 = lm_state0.damping, lm_state1.damping
    assert (d0.dtype, d0.weak_type, d0.shape) == (d1.dtype, d1.weak_type, d1.shape)
    assert d0.weak_type is False
    assert d0.dtype == jnp.result_type(float)


def test_solve_converges_with_args_and_p_jit_modes():
    def residual(theta, args, p):
        return theta - (args + p)

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-3)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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


@pytest.mark.parametrize("jit", [True, False])
def test_solve_implicit_jvp_and_vjp_wrt_p_match_underdetermined_root(jit):
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr"])
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
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver=linear_solver,
        metric=metric,
        # The qr case deliberately supplies a square-root-only metric (no
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


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "lsmr"])
def test_grad_norm_and_step_norm_match_closed_form(linear_solver):
    def residual(theta, args, p):
        matrix, target = args
        return matrix @ theta - target

    matrix = jnp.array([[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 2.0]])
    target = jnp.array([1.0, -2.0])
    theta0 = jnp.zeros(matrix.shape[1])
    init_damping = 0.1
    solver_kwargs = {}
    if linear_solver in ("cg", "lsmr"):
        solver_kwargs = {"iterative_tol": 1e-7, "iterative_maxiter": 30}

    solver = UnderdeterminedLevenbergMarquardt(
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1.0)
    _, _, info = solver.update(jnp.zeros(1), solver.init(jnp.zeros(1)))

    assert not bool(info.accepted)
    assert float(info.step_norm) == pytest.approx(0.5)
    assert float(info.grad_norm) == pytest.approx(1.0)


@pytest.mark.parametrize("jit", [True, False])
def test_solve_gtol_reports_converged(jit):
    def residual(theta, args, p):
        return theta - args

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1.0, max_damping=1e6
    )
    result = solver.solve(jnp.zeros(1), max_steps=30, xtol=10.0)

    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.steps) == 30
    assert not bool(result.info.accepted)


def test_max_damping_caps_growth_under_repeated_rejection():
    def residual(theta, _, __):
        return jnp.where(theta[0] == 0.0, theta + 1.0, jnp.full_like(theta, jnp.nan))

    capped = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-3, max_damping=1e4
    )
    result = capped.solve(jnp.zeros(1), max_steps=100)
    assert int(result.status) == LMStatus.MAX_STEPS
    assert float(result.lm_state.damping) == pytest.approx(1e4)

    uncapped = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-3)
    result = uncapped.solve(jnp.zeros(1), max_steps=100)
    assert not jnp.isfinite(result.lm_state.damping)


def test_solve_does_not_retrace_on_loop_control_changes():
    traces = {"count": 0}

    def residual(theta, args, p):
        traces["count"] += 1
        return theta - args

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-2, cache_jacobian=False
    )
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

    solver = UnderdeterminedLevenbergMarquardt(
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

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1.0, cache_jacobian=True
    )
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

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1.0, cache_jacobian=True
    )
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-2, cache_jacobian=True
    )
    with pytest.raises(ValueError, match="Jacobian cache"):
        solver.solve(jnp.zeros(1), jnp.ones(1), max_steps=3, callback=callback)


def test_hyperparams_typing_and_solve_population():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)

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
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn, init_damping=1e-2, cache_jacobian=False
    )

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

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-1, linear_solver="cg", iterative_maxiter=2
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

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1.0, max_damping=1e6
    )
    result = solver.solve(
        jnp.zeros(1), jnp.asarray([1.0]), max_steps=20, callback=cap_damping
    )

    assert int(result.status) == LMStatus.MAX_STEPS
    assert float(result.lm_state.damping) <= 10.0


def test_callback_grows_lsmr_budget_when_loss_small():
    matrix = jnp.diag(jnp.logspace(0.0, 1.5, 8))
    target = jnp.linspace(1.0, 2.0, 8)

    def residual(theta, _, __):
        return matrix @ theta - target

    def grow_budget(ctx):
        new_maxiter = jnp.where(
            ctx.info.loss < 2.0,
            jnp.asarray(60, dtype=jnp.int32),
            ctx.lm_state.hyper.iterative_maxiter,
        )
        new_hyper = dataclasses.replace(
            ctx.lm_state.hyper, iterative_maxiter=new_maxiter
        )
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, hyper=new_hyper)
        )

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-1, linear_solver="lsmr", iterative_maxiter=1
    )
    theta0 = jnp.zeros(8)
    fixed = solver.solve(theta0, max_steps=60)
    scheduled = solver.solve(theta0, max_steps=60, callback=grow_budget)

    assert int(scheduled.lm_state.hyper.iterative_maxiter) == 60
    assert float(scheduled.info.loss) < 1e-2 * float(fixed.info.loss)


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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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
    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    one_solver = UnderdeterminedLevenbergMarquardt(one_arg, init_damping=1e-2)
    three_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
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

    one_solver = UnderdeterminedLevenbergMarquardt(one_arg)
    two_solver = UnderdeterminedLevenbergMarquardt(two_arg)
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
        UnderdeterminedLevenbergMarquardt(lambda: jnp.zeros(1))
    with pytest.raises(ValueError, match="1 to 3 positional arguments"):
        UnderdeterminedLevenbergMarquardt(lambda a, b, c, d: a)


def test_residual_with_default_args_counts_as_three_arg():
    def residual(theta, _=None, __=None):
        return theta - 1.0

    solver = UnderdeterminedLevenbergMarquardt(residual)
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
    cached = UnderdeterminedLevenbergMarquardt(residual_fn, cache_jacobian=True, **kw)
    plain = UnderdeterminedLevenbergMarquardt(residual_fn, cache_jacobian=False, **kw)

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
    cached = UnderdeterminedLevenbergMarquardt(
        residual_fn, init_damping=1e-2, cache_jacobian=True
    )
    plain = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)

    cached_result = cached.solve(
        {"a": 1.0, "b": 0.0}, (ts, ys), max_steps=50, atol=1e-6
    )
    plain_result = plain.solve({"a": 1.0, "b": 0.0}, (ts, ys), max_steps=50, atol=1e-6)

    assert int(cached_result.status) == LMStatus.CONVERGED
    assert int(plain_result.status) == LMStatus.CONVERGED
    assert jnp.allclose(cached_result.x["a"], 2.0, atol=1e-4)
    assert jnp.allclose(cached_result.x["b"], -1.0, atol=1e-4)


def test_cache_jacobian_requires_sized_lm_state():
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, cache_jacobian=True)
    with pytest.raises(ValueError, match="no Jacobian cache"):
        solver.update(
            {"a": 1.0, "b": 0.0},
            LMState(jnp.asarray(1e-3)),
            (jnp.ones(3), jnp.ones(3)),
        )


def test_cache_jacobian_is_inert_for_non_cholesky_solvers():
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver="cg",
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
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
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

    plain = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    cached = UnderdeterminedLevenbergMarquardt(
        residual_fn, init_damping=1e-2, cache_jacobian=True
    )
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


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "lsmr"])
def test_has_aux_reports_aux_at_pre_step_x(linear_solver):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    solver_kwargs = {}
    if linear_solver in ("cg", "lsmr"):
        solver_kwargs = {"iterative_tol": 1e-7, "iterative_maxiter": 20}

    solver = UnderdeterminedLevenbergMarquardt(
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

    solver = UnderdeterminedLevenbergMarquardt(
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

    solver = UnderdeterminedLevenbergMarquardt(
        aux_residual_fn, init_damping=1e-2, has_aux=True, cache_jacobian=True
    )
    result = solver.solve(x, (ts, ys), max_steps=50, atol=1e-6)

    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.info.aux["mean_abs"]) < 1e-4


def test_has_aux_off_keeps_info_aux_none():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    _, _, info = solver.update(x, solver.init(x, (ts, ys)), (ts, ys))
    assert info.aux is None


@pytest.mark.parametrize("jit", [True, False])
def test_solve_returns_final_aux_at_returned_x(jit):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    solver = UnderdeterminedLevenbergMarquardt(
        aux_residual_fn, init_damping=1e-2, has_aux=True
    )
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
    solver = UnderdeterminedLevenbergMarquardt(
        aux_residual_fn, init_damping=1e-2, has_aux=True
    )
    result = solver.solve({"a": 1.0, "b": 0.0}, (ts, ys), max_steps=3)

    assert int(result.status) == LMStatus.MAX_STEPS
    _, expected = aux_residual_fn(result.x, (ts, ys))
    assert float(result.aux["max_abs"]) == pytest.approx(
        float(expected["max_abs"]), rel=1e-3, abs=1e-7
    )


def test_solve_result_aux_none_without_has_aux():
    def residual(theta, args):
        return theta - args

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(jnp.zeros(1), jnp.ones(1), max_steps=20, atol=1e-6)
    assert result.aux is None


def test_solve_implicit_jvp_works_with_has_aux():
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), {"level": theta[0]}

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-2, has_aux=True
    )

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

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-2, has_aux=True
    )

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

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-2, has_aux=True
    )
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


def test_solve_implicit_vjp_of_aux_wrt_p():
    def residual(theta, _, p):
        aux = {
            "m": theta[0] * theta[1] + p**2,
            "count": jnp.asarray(1, dtype=jnp.int32),
        }
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), aux

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-2, has_aux=True
    )
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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

    cached = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-2, cache_jacobian=True
    )
    plain = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)

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

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)

    def solved_x(x0):
        return solver.solve(x0, p=jnp.asarray(3.0), max_steps=80, atol=1e-6).x

    _, x0_dot = jax.jvp(solved_x, (jnp.zeros(2),), (jnp.ones(2),))
    assert jnp.allclose(x0_dot, jnp.zeros(2))


def test_dual_preconditioner_requires_cg():
    with pytest.raises(ValueError, match="dual_preconditioner"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn, dual_preconditioner=lambda v, damping: v
        )
    with pytest.raises(ValueError, match="dual_preconditioner"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn,
            linear_solver="qr",
            dual_preconditioner=lambda v, damping: v,
        )


def test_cg_preconditioned_step_matches_cholesky_identity_step():
    # A valid SPD preconditioner changes only the inner Krylov iteration, not
    # the step the inner solve converges to.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    weights = 1.0 + jnp.arange(20, dtype=jnp.float32) / 10.0

    cholesky_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        iterative_tol=1e-7,
        iterative_maxiter=40,
        dual_preconditioner=lambda v, damping: v / weights,
    )

    cholesky_x, cholesky_state, cholesky_info = cholesky_solver.update(
        x, cholesky_solver.init(x, (ts, ys)), (ts, ys)
    )
    cg_x, cg_state, cg_info = cg_solver.update(x, cg_solver.init(x, (ts, ys)), (ts, ys))

    assert bool(cg_info.accepted) == bool(cholesky_info.accepted)
    assert jnp.allclose(cg_x["a"], cholesky_x["a"], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(cg_x["b"], cholesky_x["b"], rtol=1e-5, atol=1e-5)
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
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
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

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="cg",
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
        metric=metric_from_cholesky(L),
    )
    plain = UnderdeterminedLevenbergMarquardt(residual, **common)
    preconditioned = UnderdeterminedLevenbergMarquardt(
        residual, dual_preconditioner=preconditioner, **common
    )
    x0 = jnp.zeros(n)

    plain_result = plain.solve(x0, max_steps=20, atol=1e-3)
    preconditioned_result = preconditioned.solve(x0, max_steps=20, atol=1e-3)

    assert int(preconditioned_result.status) == LMStatus.CONVERGED
    assert int(plain_result.status) != LMStatus.CONVERGED


def test_metric_from_tridiagonal_precision_matches_dense():
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
    metric = metric_from_tridiagonal_precision(diag, off_diag)

    x = jax.random.normal(jax.random.PRNGKey(0), (n,))
    X = jax.random.normal(jax.random.PRNGKey(1), (n, 3))
    T_dense = jnp.linalg.inv(K)

    assert jnp.allclose(metric.solve(x), T_dense @ x, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(metric.solve(X), T_dense @ X, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(metric.norm(x), jnp.sqrt(x @ K @ x), rtol=1e-4, atol=1e-4)
    S = metric.inv_sqrt(jnp.eye(n))
    assert jnp.allclose(S @ S.T, T_dense, rtol=1e-3, atol=1e-4)
    assert jnp.allclose(
        metric.inv_sqrt_transpose(jnp.eye(n)), S.T, rtol=1e-4, atol=1e-4
    )
