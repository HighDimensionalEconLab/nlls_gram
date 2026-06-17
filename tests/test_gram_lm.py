import jax
import jax.numpy as jnp
import pytest

from nlls_gram import GramLevenbergMarquardt


def residual_fn(params, batch):
    x, y = batch
    return params["a"] * jnp.exp(params["b"] * x) - y


REGRESSION_ATOL = 5e-5
REGRESSION_RTOL = 1e-5


def test_recovers_known_parameters_with_jitted_step():
    a_true, b_true = 2.0, -1.0
    x = jnp.linspace(0.0, 2.0, 20)
    y = a_true * jnp.exp(b_true * x)

    params = {"a": 1.0, "b": 0.0}
    solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
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

    default_solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
    identity_solver = GramLevenbergMarquardt(
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
    solver = GramLevenbergMarquardt(residual, init_damping=1e-2)
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

    solver = GramLevenbergMarquardt(residual, init_damping=init_damping)
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

    solver = GramLevenbergMarquardt(residual, init_damping=1e-4)
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

    solver = GramLevenbergMarquardt(residual, init_damping=1e-4)
    solver.update(params, solver.init(), (x, y))

    assert calls["count"] == 2


def test_rejected_step_leaves_params_unchanged():
    x = jnp.linspace(0.0, 2.0, 5)
    y = jnp.ones_like(x)
    params = {"a": 1.0, "b": 0.0}

    solver = GramLevenbergMarquardt(residual_fn)
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
        GramLevenbergMarquardt(residual_fn, regularization="marquardt")


def test_init_damping_must_be_positive():
    with pytest.raises(ValueError, match="init_damping must be positive"):
        GramLevenbergMarquardt(residual_fn, init_damping=0.0)


def test_default_float32_params_keep_float32_outputs():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}

    solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
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


def test_geodesic_acceptance_ratio_zero_falls_back_to_velocity_step():
    def residual(theta, target):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    solver = GramLevenbergMarquardt(
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

    solver = GramLevenbergMarquardt(
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


def test_geodesic_acceleration_jits():
    def residual(theta, target):
        return jnp.array([theta[0] ** 2 - target])

    solver = GramLevenbergMarquardt(
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
        solver = GramLevenbergMarquardt(
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
    solver = GramLevenbergMarquardt(
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
    solver = GramLevenbergMarquardt(
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


def test_geodesic_step_matches_regression_values():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}
    solver = GramLevenbergMarquardt(
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

    solver = GramLevenbergMarquardt(
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

    solver = GramLevenbergMarquardt(
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
        solver = GramLevenbergMarquardt(
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
