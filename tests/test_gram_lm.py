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
)


def residual_fn(params, aux, p):
    x, y = aux
    return params["a"] * jnp.exp(params["b"] * x) - y


REGRESSION_ATOL = 5e-5
REGRESSION_RTOL = 1e-5


def test_recovers_known_parameters_with_jitted_step():
    a_true, b_true = 2.0, -1.0
    x = jnp.linspace(0.0, 2.0, 20)
    y = a_true * jnp.exp(b_true * x)

    params = {"a": 1.0, "b": 0.0}
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    lm_state = solver.init()

    @jax.jit
    def train_step(params, lm_state, aux):
        return solver.update(params, lm_state, aux)

    info = None
    for _ in range(50):
        params, lm_state, info = train_step(params, lm_state, (x, y))

    assert float(info.loss) < 1e-8
    assert jnp.allclose(params["a"], a_true, atol=1e-4)
    assert jnp.allclose(params["b"], b_true, atol=1e-4)


def test_default_metric_matches_explicit_identity_metric_solve():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}

    default_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    identity_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn, init_damping=1e-2, metric=Metric(solve=lambda x: x)
    )

    default_params, default_state, default_info = default_solver.update(
        params, default_solver.init(), (x, y)
    )
    identity_params, identity_state, identity_info = identity_solver.update(
        params, identity_solver.init(), (x, y)
    )

    assert jnp.allclose(default_params["a"], identity_params["a"])
    assert jnp.allclose(default_params["b"], identity_params["b"])
    assert jnp.allclose(default_state.damping, identity_state.damping)
    assert jnp.allclose(default_info.loss, identity_info.loss)


def test_flat_array_params():
    def residual(theta, aux, p):
        x, y = aux
        return theta[0] * jnp.exp(theta[1] * x) - y

    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)

    params = jnp.array([1.0, 0.0])
    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init()

    for _ in range(50):
        params, lm_state, _ = solver.update(params, lm_state, (x, y))

    assert params.shape == (2,)
    assert jnp.allclose(params, jnp.array([2.0, -1.0]), atol=1e-4)


def test_linear_problem_matches_closed_form_solution():
    def residual(params, aux, p):
        x, y = aux
        return params["a"] * x - y

    x = jnp.array([1.0, 2.0, 3.0])
    y = 2.0 * x
    params = {"a": 0.0}
    init_damping = 1e-4

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=init_damping)
    new_params, _, info = solver.update(params, solver.init(), (x, y))
    expected_a = jnp.sum(x * y) / (jnp.sum(x**2) + init_damping)
    expected_loss = jnp.sum((expected_a * x - y) ** 2)

    assert bool(info.accepted)
    assert jnp.allclose(new_params["a"], expected_a, atol=1e-6)
    assert jnp.allclose(info.loss, expected_loss, atol=1e-10)
    assert float(info.loss_old) == pytest.approx(56.0)
    assert jnp.allclose(info.loss_candidate, expected_loss, atol=1e-10)
    assert float(info.damping_factor) == pytest.approx(0.5)
    assert not bool(info.used_geodesic)
    assert float(info.acceleration_ratio) == pytest.approx(0.0)


def test_update_calls_residual_and_uses_values():
    calls = {"count": 0}

    def residual(params, aux, p):
        calls["count"] += 1
        x, y = aux
        return params["a"] * x - y

    x = jnp.array([1.0, 2.0, 3.0])
    y = 2.0 * x
    params = {"a": 0.0}

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-4)
    new_params, _, info = solver.update(params, solver.init(), (x, y))

    assert calls["count"] >= 2
    assert bool(info.accepted)
    assert jnp.allclose(new_params["a"], 2.0, atol=1e-4)


def test_default_disabled_geodesic_uses_minimal_residual_evaluations():
    calls = {"count": 0}

    def residual(params, aux, p):
        calls["count"] += 1
        x, y = aux
        return params["a"] * x - y

    x = jnp.array([1.0, 2.0, 3.0])
    y = 2.0 * x
    params = {"a": 0.0}

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-4)
    solver.update(params, solver.init(), (x, y))

    assert calls["count"] == 2


def test_rejected_step_leaves_params_unchanged():
    x = jnp.linspace(0.0, 2.0, 5)
    y = jnp.ones_like(x)
    params = {"a": 1.0, "b": 0.0}

    solver = UnderdeterminedLevenbergMarquardt(residual_fn)
    state = solver.init()
    new_params, new_state, info = solver.update(params, state, (x, y))

    assert not bool(info.accepted)
    assert jnp.allclose(new_params["a"], params["a"])
    assert jnp.allclose(new_params["b"], params["b"])
    assert float(new_state.damping) > float(state.damping)
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


def test_default_float32_params_keep_float32_outputs():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}

    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    new_params, new_state, info = solver.update(params, solver.init(), (x, y))

    assert new_params["a"].dtype == jnp.float32
    assert new_params["b"].dtype == jnp.float32
    assert new_state.damping.dtype == jnp.float32
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
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}

    cholesky_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )

    cholesky_params, cholesky_state, cholesky_info = cholesky_solver.update(
        params, cholesky_solver.init(), (x, y)
    )
    cg_params, cg_state, cg_info = cg_solver.update(params, cg_solver.init(), (x, y))

    assert bool(cg_info.accepted) == bool(cholesky_info.accepted)
    assert not bool(cg_info.used_geodesic)
    assert jnp.allclose(cg_params["a"], cholesky_params["a"], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(cg_params["b"], cholesky_params["b"], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(cg_state.damping, cholesky_state.damping)
    assert jnp.allclose(
        cg_info.loss,
        cholesky_info.loss,
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert cg_params["a"].dtype == jnp.float32
    assert cg_info.loss.dtype == jnp.float32


def test_qr_step_matches_cholesky_identity_step():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
    )
    qr_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn, init_damping=1e-2, linear_solver="qr"
    )

    cholesky_params, cholesky_state, cholesky_info = cholesky_solver.update(
        params, cholesky_solver.init(), (x, y)
    )
    qr_params, qr_state, qr_info = qr_solver.update(params, qr_solver.init(), (x, y))

    assert bool(qr_info.accepted) == bool(cholesky_info.accepted)
    assert not bool(qr_info.used_geodesic)
    assert jnp.allclose(
        qr_params["a"],
        cholesky_params["a"],
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert jnp.allclose(
        qr_params["b"],
        cholesky_params["b"],
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
    assert qr_params["a"].dtype == jnp.float32
    assert qr_info.loss.dtype == jnp.float32


def test_lsmr_step_matches_closed_form_damped_linear_solution():
    def residual(theta, aux, p):
        matrix, target = aux
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
    theta, state, info = solver.update(theta0, solver.init(), (matrix, target))

    expected_step = jnp.linalg.solve(
        matrix.T @ matrix + init_damping * jnp.eye(matrix.shape[1]),
        matrix.T @ target,
    )
    expected_loss = jnp.sum((matrix @ expected_step - target) ** 2)

    assert bool(info.accepted)
    assert not bool(info.used_geodesic)
    assert jnp.allclose(theta, expected_step, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(info.loss, expected_loss, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(state.damping, init_damping * 0.5)
    assert theta.dtype == jnp.float32
    assert info.loss.dtype == jnp.float32


def test_qr_step_matches_closed_form_underdetermined_damped_solution():
    def residual(theta, aux, p):
        matrix, target = aux
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
    theta, state, info = solver.update(theta0, solver.init(), (matrix, target))

    expected_step = jnp.linalg.solve(
        matrix.T @ matrix + init_damping * jnp.eye(matrix.shape[1]),
        matrix.T @ target,
    )
    expected_loss = jnp.sum((matrix @ expected_step - target) ** 2)

    assert bool(info.accepted)
    assert not bool(info.used_geodesic)
    assert jnp.allclose(theta, expected_step, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(info.loss, expected_loss, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(state.damping, init_damping * 0.5)
    assert theta.dtype == jnp.float32
    assert info.loss.dtype == jnp.float32


def test_qr_float32_handles_ill_conditioned_case_where_cholesky_fails():
    def residual(theta, aux, p):
        matrix, target = aux
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
        theta0, cholesky_solver.init(), (matrix, target)
    )
    qr_theta, _, qr_info = qr_solver.update(theta0, qr_solver.init(), (matrix, target))

    assert jnp.all(jnp.isfinite(qr_theta))
    assert jnp.isfinite(qr_info.loss_candidate)
    assert bool(qr_info.accepted)
    assert not bool(jnp.all(jnp.isfinite(cholesky_theta))) or not bool(
        jnp.isfinite(cholesky_info.loss_candidate)
    )
    assert qr_info.loss_candidate < 1e-12


def test_cg_update_jits():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )

    @jax.jit
    def train_step(params, state):
        return solver.update(params, state, (x, y))

    params, state, info = train_step(params, solver.init())

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(state.damping)
    assert params["a"].dtype == jnp.float32
    assert params["b"].dtype == jnp.float32


def test_qr_update_jits():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="qr",
    )

    @jax.jit
    def train_step(params, state):
        return solver.update(params, state, (x, y))

    params, state, info = train_step(params, solver.init())

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(state.damping)
    assert params["a"].dtype == jnp.float32
    assert params["b"].dtype == jnp.float32


def test_lsmr_update_jits():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="lsmr",
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )

    @jax.jit
    def train_step(params, state):
        return solver.update(params, state, (x, y))

    params, state, info = train_step(params, solver.init())

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(state.damping)
    assert params["a"].dtype == jnp.float32
    assert params["b"].dtype == jnp.float32


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
    new_theta, _, info = solver.update(theta0, solver.init(), target)

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
    new_theta, _, info = solver.update(theta0, solver.init(), target)

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
        theta0, cholesky_solver.init(), target
    )
    cg_theta, _, cg_info = cg_solver.update(theta0, cg_solver.init(), target)

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
        theta0, cholesky_solver.init(), target
    )
    qr_theta, _, qr_info = qr_solver.update(theta0, qr_solver.init(), target)

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
        theta0, cholesky_solver.init(), target
    )
    lsmr_theta, _, lsmr_info = lsmr_solver.update(theta0, lsmr_solver.init(), target)

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
    def step(theta, state, target):
        return solver.update(theta, state, target)

    theta, state, info = step(jnp.array([1.9]), solver.init(), 4.0)

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    assert jnp.isfinite(theta[0])
    assert jnp.isfinite(state.damping)


def test_geodesic_acceleration_reduces_iterations_on_gsl_rosenbrock_example():
    def residual(theta, aux, p):
        del aux
        return jnp.array([100.0 * (theta[1] - theta[0] ** 2), 1.0 - theta[0]])

    def iterations_to_threshold(geodesic_acceleration):
        theta = jnp.array([-0.5, 1.75])
        solver = UnderdeterminedLevenbergMarquardt(
            residual,
            init_damping=1.0,
            geodesic_acceleration=geodesic_acceleration,
        )
        state = solver.init()
        used_geodesic = 0
        for iteration in range(1, 101):
            theta, state, info = solver.update(theta, state, None)
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
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = UnderdeterminedLevenbergMarquardt(
        jax.jit(residual_fn),
        init_damping=1e-2,
        geodesic_acceleration=True,
    )

    @jax.jit
    def train_step(params, state):
        return solver.update(params, state, (x, y))

    params, state, info = train_step(params, solver.init())

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(info.acceleration_ratio)
    assert jnp.isfinite(state.damping)
    assert params["a"].dtype == jnp.float32
    assert params["b"].dtype == jnp.float32


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


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "lsmr"])
def test_metric_step_matches_closed_form_solution(linear_solver):
    def residual(theta, aux, p):
        matrix, target = aux
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
    theta, state, info = solver.update(theta0, solver.init(), (matrix, target))

    expected_step = jnp.linalg.solve(
        matrix.T @ matrix + init_damping * metric_matrix,
        matrix.T @ target,
    )
    expected_loss = jnp.sum((matrix @ expected_step - target) ** 2)

    assert bool(info.accepted)
    assert not bool(info.used_geodesic)
    assert jnp.allclose(theta, expected_step, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(info.loss, expected_loss, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(state.damping, init_damping * 0.5)


def test_geodesic_step_matches_regression_values():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=10.0,
    )

    new_params, state, info = solver.update(params, solver.init(), (x, y))

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    assert jnp.allclose(
        new_params["a"],
        jnp.asarray(1.9073810577392578),
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert jnp.allclose(
        new_params["b"],
        jnp.asarray(-0.9168586730957031),
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert float(info.loss) == pytest.approx(0.029626082628965378, abs=REGRESSION_ATOL)
    assert float(info.loss_old) == pytest.approx(5.599210739135742, abs=REGRESSION_ATOL)
    assert float(info.loss_candidate) == pytest.approx(
        0.029626082628965378, abs=REGRESSION_ATOL
    )
    assert float(state.damping) == pytest.approx(0.004999999888241291)
    assert float(info.damping_factor) == pytest.approx(0.5)
    assert float(info.acceleration_ratio) == pytest.approx(
        0.8667416572570801, abs=REGRESSION_ATOL
    )


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "lsmr"])
def test_metric_geodesic_acceleration_ratio_uses_metric_norm(linear_solver):
    def residual(theta, aux, p):
        del aux
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

    _, _, info = solver.update(theta0, solver.init(), None)

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
    def residual(theta, aux, p):
        matrix, target = aux
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
    def train_step(theta, state):
        return solver.update(theta, state, (matrix, target))

    theta, state, info = train_step(theta0, solver.init())
    expected_step = jnp.linalg.solve(
        matrix.T @ matrix + init_damping * metric_matrix,
        matrix.T @ target,
    )

    assert bool(info.accepted)
    assert jnp.allclose(theta, expected_step, rtol=1e-5, atol=1e-5)
    assert jnp.isfinite(state.damping)
    assert jnp.isfinite(info.loss)


def test_init_state_matches_update_signature():
    # init() and update() must produce the same jit signature for `damping`;
    # a weakly-typed init scalar forced a recompile on the second step.
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)

    state0 = solver.init()
    _, state1, _ = solver.update(params, state0, (x, y))
    d0, d1 = state0.damping, state1.damping
    assert (d0.dtype, d0.weak_type, d0.shape) == (d1.dtype, d1.weak_type, d1.shape)
    assert d0.weak_type is False
    assert d0.dtype == jnp.result_type(float)


def test_init_accepts_explicit_dtype():
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    assert solver.init(jnp.float32).damping.dtype == jnp.dtype("float32")


def test_solve_converges_with_aux_and_p_jit_modes():
    def residual(theta, aux, p):
        return theta - (aux + p)

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    theta0 = jnp.array([0.0])
    aux = jnp.array([1.25])
    p = jnp.array([0.75])

    jit_result = solver.solve(theta0, aux, p=p, max_steps=40, atol=1e-6)
    python_result = solver.solve(theta0, aux, p=p, max_steps=40, atol=1e-6, jit=False)

    assert int(jit_result.status) == LMStatus.CONVERGED
    assert int(python_result.status) == LMStatus.CONVERGED
    assert jnp.allclose(jit_result.params, jnp.array([2.0]), atol=1e-5)
    assert jnp.allclose(jit_result.params, python_result.params, atol=1e-6)
    assert jnp.allclose(jit_result.p, p)
    assert jit_result.steps <= 40


def test_solve_reports_max_steps_without_atol_convergence():
    def residual(theta, aux, p):
        return theta - aux

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(jnp.array([0.0]), jnp.array([1.0]), max_steps=3, atol=0.0)

    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.steps) == 3
    assert jnp.isfinite(result.info.loss)


def test_solve_callback_can_abort_on_nonfinite_candidate():
    def residual(theta, aux, p):
        del aux, p
        return jnp.where(theta[0] > 0.0, theta + 1.0, jnp.asarray([jnp.nan]))

    def callback(ctx):
        nonfinite = ~jnp.isfinite(ctx.info.loss_candidate)
        return LMSolveAction(stop=nonfinite, status=LMStatus.NONFINITE)

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-3)
    theta0 = jnp.array([0.1])
    result = solver.solve(theta0, max_steps=5, callback=callback)

    assert int(result.status) == LMStatus.NONFINITE
    assert int(result.steps) == 1
    assert jnp.allclose(result.params, theta0)
    assert jnp.isfinite(result.info.loss)
    assert not jnp.isfinite(result.info.loss_candidate)


def test_solve_callback_updates_aux_and_user_state():
    def residual(theta, aux, p):
        del p
        return theta - aux

    def callback(ctx):
        next_aux = jnp.where(ctx.step == 1, jnp.asarray([2.0]), ctx.aux)
        return LMSolveAction(aux=next_aux, user_state=ctx.user_state + ctx.info.loss)

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
    assert jnp.allclose(result.aux, jnp.array([2.0]))
    assert result.params[0] > 1.0
    assert result.user_state > 0.0


def test_solve_implicit_jvp_and_vjp_wrt_p_match_underdetermined_root():
    def residual(theta, aux, p):
        del aux
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    theta0 = jnp.zeros(2)

    def solved_params(p):
        return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).params

    p = jnp.asarray(3.0)
    p_dot = jnp.asarray(0.7)
    params, params_dot = jax.jvp(solved_params, (p,), (p_dot,))
    expected_params = jnp.array([3.0 / 5.0, 6.0 / 5.0])
    expected_params_dot = jnp.array([p_dot / 5.0, 2.0 * p_dot / 5.0])

    _, pullback = jax.vjp(solved_params, p)
    (p_cotangent,) = pullback(jnp.array([3.0, 4.0]))
    expected_p_cotangent = (3.0 + 2.0 * 4.0) / 5.0

    assert jnp.allclose(params, expected_params, atol=1e-5)
    assert jnp.allclose(params_dot, expected_params_dot, atol=1e-6)
    assert jnp.allclose(p_cotangent, expected_p_cotangent, atol=1e-6)


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr"])
def test_solve_implicit_jvp_wrt_p_uses_metric(linear_solver):
    def residual(theta, aux, p):
        del aux
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
    )

    def solved_params(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).params

    p_dot = jnp.asarray(0.7)
    _, params_dot = jax.jvp(solved_params, (jnp.asarray(3.0),), (p_dot,))
    expected_params_dot = (
        metric_inverse
        @ jacobian.T
        @ jnp.linalg.solve(jacobian @ metric_inverse @ jacobian.T, jnp.array([p_dot]))
    ).ravel()

    assert jnp.allclose(params_dot, expected_params_dot, atol=1e-6)


@pytest.mark.parametrize("linear_solver", ["cholesky", "cg", "qr", "lsmr"])
def test_grad_norm_and_step_norm_match_closed_form(linear_solver):
    def residual(theta, aux, p):
        matrix, target = aux
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
    _, _, info = solver.update(theta0, solver.init(), (matrix, target))

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
    def residual(theta, aux, p):
        del aux, p
        return jnp.where(theta[0] == 0.0, theta + 1.0, jnp.full_like(theta, jnp.nan))

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1.0)
    _, _, info = solver.update(jnp.zeros(1), solver.init())

    assert not bool(info.accepted)
    assert float(info.step_norm) == pytest.approx(0.5)
    assert float(info.grad_norm) == pytest.approx(1.0)


@pytest.mark.parametrize("jit", [True, False])
def test_solve_gtol_reports_converged(jit):
    def residual(theta, aux, p):
        return theta - aux

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]), jnp.array([1.0]), max_steps=50, gtol=1e-6, jit=jit
    )

    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.info.grad_norm) < 1e-6
    assert int(result.steps) < 50


@pytest.mark.parametrize("jit", [True, False])
def test_solve_xtol_reports_converged_on_accepted_step(jit):
    def residual(theta, aux, p):
        return theta - aux

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]), jnp.array([1.0]), max_steps=50, xtol=1e-6, jit=jit
    )

    assert int(result.status) == LMStatus.CONVERGED
    assert bool(result.info.accepted)
    assert float(result.info.step_norm) < 1e-6
    assert int(result.steps) < 50


def test_solve_xtol_ignores_rejected_steps():
    def residual(theta, aux, p):
        del aux, p
        return jnp.where(theta[0] == 0.0, theta + 1.0, jnp.full_like(theta, jnp.nan))

    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1.0, max_damping=1e6
    )
    result = solver.solve(jnp.zeros(1), max_steps=30, xtol=10.0)

    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.steps) == 30
    assert not bool(result.info.accepted)


def test_max_damping_caps_growth_under_repeated_rejection():
    def residual(theta, aux, p):
        del aux, p
        return jnp.where(theta[0] == 0.0, theta + 1.0, jnp.full_like(theta, jnp.nan))

    capped = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-3, max_damping=1e4
    )
    result = capped.solve(jnp.zeros(1), max_steps=100)
    assert int(result.status) == LMStatus.MAX_STEPS
    assert float(result.state.damping) == pytest.approx(1e4)

    uncapped = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-3)
    result = uncapped.solve(jnp.zeros(1), max_steps=100)
    assert not jnp.isfinite(result.state.damping)


def test_solve_does_not_retrace_on_loop_control_changes():
    traces = {"count": 0}

    def residual(theta, aux, p):
        traces["count"] += 1
        return theta - aux

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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


def test_solve_callback_epoch_boundary_resamples_aux_and_resets_damping():
    def residual(theta, aux, p):
        del p
        return theta - aux

    steps_per_epoch = 3

    def callback(ctx):
        boundary = ctx.step % steps_per_epoch == 0
        new_aux = jnp.where(boundary, ctx.aux + 1.0, ctx.aux)
        new_state = LMState(
            jnp.where(boundary, ctx.initial_state.damping, ctx.state.damping)
        )
        epochs = ctx.user_state + jnp.where(boundary, 1, 0)
        return LMSolveAction(aux=new_aux, state=new_state, user_state=epochs)

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    result = solver.solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        max_steps=7,
        callback=callback,
        user_state=jnp.asarray(0),
    )

    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.user_state) == 2
    assert jnp.allclose(result.aux, jnp.array([3.0]))
    # Step 6 reset damping to init; the accepted step 7 halved it once.
    assert float(result.state.damping) == pytest.approx(5e-3)


def test_solve_callback_history_buffer_matches_update_loop():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}
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
        params,
        (x, y),
        max_steps=max_steps,
        callback=callback,
        user_state={
            "loss": jnp.zeros(max_steps),
            "damping": jnp.zeros(max_steps),
        },
    )

    loop_params, loop_state = params, solver.init()
    for i in range(max_steps):
        loop_params, loop_state, info = solver.update(loop_params, loop_state, (x, y))
        assert float(result.user_state["loss"][i]) == pytest.approx(
            float(info.loss), rel=1e-4, abs=1e-8
        )
        assert float(result.user_state["damping"][i]) == pytest.approx(
            float(loop_state.damping), rel=1e-6
        )
