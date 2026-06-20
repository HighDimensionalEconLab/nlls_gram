import jax
import jax.numpy as jnp
import pytest

import nlls_gram
from nlls_gram import UnderdeterminedLevenbergMarquardt


def residual_fn(params, batch):
    x, y = batch
    return params["a"] * jnp.exp(params["b"] * x) - y


REGRESSION_ATOL = 5e-5
REGRESSION_RTOL = 1e-5


def test_old_solver_name_is_not_exported():
    assert not hasattr(nlls_gram, "GramLevenbergMarquardt")


def test_recovers_known_parameters_with_jitted_step():
    a_true, b_true = 2.0, -1.0
    x = jnp.linspace(0.0, 2.0, 20)
    y = a_true * jnp.exp(b_true * x)

    params = {"a": 1.0, "b": 0.0}
    solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    lm_state = solver.init()

    @jax.jit
    def train_step(params, lm_state, batch):
        return solver.update(params, lm_state, batch)

    info = None
    for _ in range(50):
        params, lm_state, info = train_step(params, lm_state, (x, y))

    assert float(info.loss) < 1e-8
    assert jnp.allclose(params["a"], a_true, atol=1e-4)
    assert jnp.allclose(params["b"], b_true, atol=1e-4)


def test_default_regularization_matches_explicit_identity():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}

    default_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    identity_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn, init_damping=1e-2, regularization="identity"
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
    def residual(theta, batch):
        x, y = batch
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
    def residual(params, batch):
        x, y = batch
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

    def residual(params, batch):
        calls["count"] += 1
        x, y = batch
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

    def residual(params, batch):
        calls["count"] += 1
        x, y = batch
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


def test_unknown_regularization_raises():
    with pytest.raises(ValueError, match="unknown regularization"):
        UnderdeterminedLevenbergMarquardt(residual_fn, regularization="marquardt")


def test_unknown_linear_solver_raises():
    with pytest.raises(ValueError, match="unknown linear_solver"):
        UnderdeterminedLevenbergMarquardt(residual_fn, linear_solver="svd")


def test_unknown_jac_raises():
    with pytest.raises(ValueError, match='only jac="vjp" is supported'):
        UnderdeterminedLevenbergMarquardt(residual_fn, jac="jvp")


def test_formulation_argument_is_removed():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        UnderdeterminedLevenbergMarquardt(residual_fn, formulation="gram")


def test_cg_requires_identity_regularization():
    with pytest.raises(ValueError, match='linear_solver="cg" only supports'):
        UnderdeterminedLevenbergMarquardt(
            residual_fn,
            linear_solver="cg",
            regularization="fletcher",
        )


def test_lsmr_requires_identity_regularization():
    with pytest.raises(ValueError, match='linear_solver="lsmr" only supports'):
        UnderdeterminedLevenbergMarquardt(
            residual_fn,
            linear_solver="lsmr",
            regularization="fletcher",
        )


def test_qr_requires_identity_regularization():
    with pytest.raises(ValueError, match='linear_solver="qr" only supports'):
        UnderdeterminedLevenbergMarquardt(
            residual_fn,
            linear_solver="qr",
            regularization="fletcher",
        )


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


def test_fletcher_diagonal_clip_bounds_must_be_valid():
    with pytest.raises(ValueError, match="fletcher_min_diagonal must be positive"):
        UnderdeterminedLevenbergMarquardt(
            residual_fn,
            regularization="fletcher",
            fletcher_min_diagonal=0.0,
        )
    with pytest.raises(
        ValueError,
        match="fletcher_max_diagonal must be greater than or equal",
    ):
        UnderdeterminedLevenbergMarquardt(
            residual_fn,
            regularization="fletcher",
            fletcher_min_diagonal=1.0,
            fletcher_max_diagonal=0.5,
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
    def residual(theta, batch):
        matrix, target = batch
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
    def residual(theta, batch):
        matrix, target = batch
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
    def residual(theta, batch):
        matrix, target = batch
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


def test_qr_update_is_differentiable():
    def residual(theta, batch):
        matrix, target = batch
        return jnp.sin(matrix @ theta) - target

    matrix = jnp.array([[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 2.0]])
    target = jnp.array([0.1, -0.2])
    theta0 = jnp.array([0.2, -0.1, 0.05, 0.3])
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="qr",
    )

    def one_step_loss(theta):
        _, _, info = solver.update(theta, solver.init(), (matrix, target))
        return info.loss

    grad = jax.grad(one_step_loss)(theta0)

    assert grad.shape == theta0.shape
    assert jnp.all(jnp.isfinite(grad))


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
    def residual(theta, target):
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
    def residual(theta, target):
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
    def residual(theta, target):
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
    def residual(theta, target):
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
    def residual(theta, target):
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
    def residual(theta, target):
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
    def residual(theta, batch):
        del batch
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


def test_fletcher_step_matches_regression_values():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        regularization="fletcher",
    )

    new_params, state, info = solver.update(params, solver.init(), (x, y))

    assert bool(info.accepted)
    assert not bool(info.used_geodesic)
    assert jnp.allclose(
        new_params["a"],
        jnp.asarray(1.6462632417678833),
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert jnp.allclose(
        new_params["b"],
        jnp.asarray(-0.7737699747085571),
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert float(info.loss) == pytest.approx(0.35342830419540405, abs=REGRESSION_ATOL)
    assert float(info.loss_old) == pytest.approx(5.599210739135742, abs=REGRESSION_ATOL)
    assert float(info.loss_candidate) == pytest.approx(
        0.35342830419540405, abs=REGRESSION_ATOL
    )
    assert float(state.damping) == pytest.approx(0.004999999888241291)
    assert float(info.damping_factor) == pytest.approx(0.5)
    assert float(info.acceleration_ratio) == pytest.approx(0.0)


def test_fletcher_clips_tiny_diagonal_from_below():
    def residual(theta, target):
        return jnp.array([theta[0] + 1e-4 * theta[1] - target])

    theta0 = jnp.array([0.0, 0.0])
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1.0,
        regularization="fletcher",
        fletcher_min_diagonal=1e-6,
        fletcher_max_diagonal=1e6,
    )
    raw_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1.0,
        regularization="fletcher",
        fletcher_min_diagonal=1e-12,
        fletcher_max_diagonal=1e6,
    )

    theta, _, info = solver.update(theta0, solver.init(), 1.0)
    raw_theta, _, _ = raw_solver.update(theta0, raw_solver.init(), 1.0)

    clipped_diagonal = 1e-6
    denominator = 1.0 + (1e-8 / clipped_diagonal) + 1.0
    expected_theta = jnp.array([1.0, 1e-4 / clipped_diagonal]) / denominator

    assert bool(info.accepted)
    assert jnp.allclose(theta, expected_theta, rtol=1e-5, atol=1e-5)
    assert jnp.abs(theta[1]) < 0.02 * jnp.abs(raw_theta[1])


def test_fletcher_clips_large_diagonal_from_above():
    def residual(theta, target):
        return jnp.array([1e4 * theta[0] - target])

    theta0 = jnp.array([0.0])
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1.0,
        regularization="fletcher",
        fletcher_min_diagonal=1e-6,
        fletcher_max_diagonal=1e6,
    )

    theta, _, info = solver.update(theta0, solver.init(), 1.0)

    clipped_diagonal = 1e6
    denominator = (1e8 / clipped_diagonal) + 1.0
    expected_theta = jnp.array([1e4 / clipped_diagonal / denominator])

    assert bool(info.accepted)
    assert jnp.allclose(theta, expected_theta, rtol=1e-6, atol=1e-8)


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


def test_fletcher_handles_unused_parameter():
    def residual(params, batch):
        x, y = batch
        return params["a"] * x - y

    x = jnp.array([1.0, 2.0, 3.0])
    y = 2.0 * x
    params = {"a": 0.0, "unused": 10.0}

    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-2,
        regularization="fletcher",
    )
    new_params, _, info = solver.update(params, solver.init(), (x, y))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.allclose(new_params["unused"], params["unused"])


def test_fletcher_geodesic_handles_unused_parameter():
    def residual(params, batch):
        x, y = batch
        return params["a"] * jnp.exp(x) - y

    x = jnp.array([0.0, 0.5, 1.0])
    y = 2.0 * jnp.exp(x)
    params = {"a": 1.0, "unused": 10.0}

    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-2,
        regularization="fletcher",
        geodesic_acceleration=True,
    )
    new_params, _, info = solver.update(params, solver.init(), (x, y))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(info.acceleration_ratio)
    assert jnp.allclose(new_params["unused"], params["unused"])


def test_fletcher_regularization_improves_scaled_parameter_case():
    x = jnp.linspace(0.0, 2.0, 50)
    y = 2.0 * jnp.exp(-1.0 * x)
    parameter_scale = 1e-3

    def residual(params, batch):
        x, y = batch
        b = parameter_scale * params["b_scaled"]
        return params["a"] * jnp.exp(b * x) - y

    def iterations_to_threshold(regularization):
        params = {"a": 1.0, "b_scaled": 0.0}
        solver = UnderdeterminedLevenbergMarquardt(
            residual,
            init_damping=1e-2,
            regularization=regularization,
        )
        state = solver.init()
        for iteration in range(1, 51):
            params, state, info = solver.update(params, state, (x, y))
            if float(info.loss) < 1e-8:
                return iteration
        return 51

    identity_iterations = iterations_to_threshold("identity")
    fletcher_iterations = iterations_to_threshold("fletcher")

    assert identity_iterations >= 10
    assert fletcher_iterations <= 6
    assert fletcher_iterations < identity_iterations


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
