import jax
import jax.numpy as jnp
import pytest

from nlls_gram import GramLevenbergMarquardt


def residual_fn(params, batch):
    x, y = batch
    return params["a"] * jnp.exp(params["b"] * x) - y


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


def test_normal_solve_method_recovers_parameters():
    a_true, b_true = 2.0, -1.0
    x = jnp.linspace(0.0, 2.0, 20)
    y = a_true * jnp.exp(b_true * x)

    params = {"a": 1.0, "b": 0.0}
    solver = GramLevenbergMarquardt(
        residual_fn, init_damping=1e-2, solve_method="normal"
    )
    lm_state = solver.init()

    info = None
    for _ in range(50):
        params, lm_state, info = solver.update(params, lm_state, (x, y))

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

    solver = GramLevenbergMarquardt(residual, init_damping=0.0, solve_method="normal")
    new_params, _, info = solver.update(params, solver.init(), (x, y))

    assert bool(info.accepted)
    assert jnp.allclose(new_params["a"], 2.0, atol=1e-6)
    assert float(info.loss) == pytest.approx(0.0, abs=1e-10)


def test_update_calls_residual_and_uses_values():
    calls = {"count": 0}

    def residual(params, batch):
        calls["count"] += 1
        x, y = batch
        return params["a"] * x - y

    x = jnp.array([1.0, 2.0, 3.0])
    y = 2.0 * x
    params = {"a": 0.0}

    solver = GramLevenbergMarquardt(residual, init_damping=0.0, solve_method="normal")
    new_params, _, info = solver.update(params, solver.init(), (x, y))

    assert calls["count"] >= 2
    assert bool(info.accepted)
    assert jnp.allclose(new_params["a"], 2.0, atol=1e-6)


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


def test_unknown_solve_method_raises():
    with pytest.raises(ValueError, match="unknown solve_method"):
        GramLevenbergMarquardt(residual_fn, solve_method="qr")


def test_unknown_regularization_raises():
    with pytest.raises(ValueError, match="unknown regularization"):
        GramLevenbergMarquardt(residual_fn, regularization="marquardt")


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
    assert info.damping.dtype == jnp.float32


def test_fletcher_normal_and_gram_steps_agree():
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": 1.0, "b": 0.0}

    normal_solver = GramLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        solve_method="normal",
        regularization="fletcher",
    )
    gram_solver = GramLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        solve_method="gram",
        regularization="fletcher",
    )

    normal_params, _, normal_info = normal_solver.update(
        params, normal_solver.init(), (x, y)
    )
    gram_params, _, gram_info = gram_solver.update(params, gram_solver.init(), (x, y))

    assert jnp.allclose(normal_params["a"], gram_params["a"], rtol=1e-5, atol=1e-6)
    assert jnp.allclose(normal_params["b"], gram_params["b"], rtol=1e-5, atol=1e-6)
    assert jnp.allclose(normal_info.loss, gram_info.loss, rtol=1e-5, atol=1e-6)


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
        solve_method="normal",
        regularization="fletcher",
    )
    new_params, _, info = solver.update(params, solver.init(), (x, y))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
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
            solve_method="normal",
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
