import jax
import jax.numpy as jnp
import pytest

from nlls_gram import GramLevenbergMarquardt

jax.config.update("jax_enable_x64", True)


# residual_fn(params, batch) -> residual array; params is a plain pytree (dict).
def residual_fn(params, batch):
    x, y = batch
    return params["a"] * jnp.exp(params["b"] * x) - y


def test_recovers_known_parameters():
    a_true, b_true = 2.0, -1.0
    x = jnp.linspace(0.0, 2.0, 20)
    y = a_true * jnp.exp(b_true * x)

    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
    lm_state = solver.init()

    @jax.jit
    def train_step(params, lm_state, batch):
        return solver.update(params, lm_state, batch)

    info = None
    for _ in range(50):
        params, lm_state, info = train_step(params, lm_state, (x, y))

    assert float(info.loss) < 1e-10
    assert jnp.allclose(params["a"], a_true, atol=1e-4)
    assert jnp.allclose(params["b"], b_true, atol=1e-4)


def test_normal_solve_method_recovers_parameters():
    a_true, b_true = 2.0, -1.0
    x = jnp.linspace(0.0, 2.0, 20)
    y = a_true * jnp.exp(b_true * x)

    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = GramLevenbergMarquardt(
        residual_fn, init_damping=1e-2, solve_method="normal"
    )
    lm_state = solver.init()

    info = None
    for _ in range(50):
        params, lm_state, info = solver.update(params, lm_state, (x, y))

    assert float(info.loss) < 1e-10
    assert jnp.allclose(params["a"], a_true, atol=1e-4)
    assert jnp.allclose(params["b"], b_true, atol=1e-4)


def test_flat_array_params():
    # params can be a bare array, not just a dict; structure is preserved out.
    def residual(theta, batch):
        x, y = batch
        return theta[0] * jnp.exp(theta[1] * x) - y

    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)

    params = jnp.asarray([1.0, 0.0])
    solver = GramLevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init()

    for _ in range(50):
        params, lm_state, _ = solver.update(params, lm_state, (x, y))

    assert params.shape == (2,)
    assert jnp.allclose(params, jnp.asarray([2.0, -1.0]), atol=1e-4)


def test_rejected_step_leaves_params_unchanged():
    # Starting at an exactly-zero-residual point (model == 1 everywhere) no step
    # can decrease the SSQ, so the step is rejected and params are returned as-is.
    x = jnp.linspace(0.0, 2.0, 5)
    y = jnp.ones_like(x)
    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

    solver = GramLevenbergMarquardt(residual_fn)
    state = solver.init()
    new_params, new_state, info = solver.update(params, state, (x, y))

    assert not bool(info.accepted)
    assert new_params["a"] == params["a"]
    assert new_params["b"] == params["b"]
    # damping is multiplied by damping_increase on rejection
    assert float(new_state.damping) > float(state.damping)


def test_freezes_held_out_params():
    # Freezing = pass only the trainable subset; the frozen leaves live in the
    # residual closure, get no Jacobian column, and never move. No wrt/mask needed.
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    frozen = {"b": jnp.asarray(-1.0)}

    def residual(trainable, batch):
        xx, yy = batch
        params = {**frozen, **trainable}
        return params["a"] * jnp.exp(params["b"] * xx) - yy

    trainable = {"a": jnp.asarray(1.0)}
    solver = GramLevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init()
    for _ in range(50):
        trainable, lm_state, _ = solver.update(trainable, lm_state, (x, y))

    # The solver only ever sees/returns the trainable subset; "b" stays put.
    assert set(trainable) == {"a"}
    assert jnp.allclose(trainable["a"], 2.0, atol=1e-4)
    assert frozen["b"] == -1.0


def test_unknown_solve_method_raises():
    with pytest.raises(ValueError, match="unknown solve_method"):
        GramLevenbergMarquardt(residual_fn, solve_method="qr")


def test_dtype_consistency_float64():
    # Under jax_enable_x64, float64 params/data must yield float64 out, no promotion.
    x = jnp.linspace(0.0, 2.0, 20)
    y = 2.0 * jnp.exp(-1.0 * x)
    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    assert params["a"].dtype == jnp.float64

    solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
    state = solver.init()
    new_params, new_state, info = solver.update(params, state, (x, y))

    assert new_params["a"].dtype == jnp.float64
    assert new_params["b"].dtype == jnp.float64
    assert info.loss.dtype == jnp.float64


def test_dtype_consistency_float32():
    # float32 params/data must stay float32 even with x64 enabled -- no cast/promote.
    x = jnp.linspace(0.0, 2.0, 20, dtype=jnp.float32)
    y = (2.0 * jnp.exp(-1.0 * x)).astype(jnp.float32)
    params = {"a": jnp.asarray(1.0, jnp.float32), "b": jnp.asarray(0.0, jnp.float32)}
    assert params["a"].dtype == jnp.float32

    solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
    state = solver.init()
    new_params, new_state, info = solver.update(params, state, (x, y))

    assert new_params["a"].dtype == jnp.float32
    assert new_params["b"].dtype == jnp.float32
    assert info.loss.dtype == jnp.float32
