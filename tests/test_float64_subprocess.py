import subprocess
import sys
import textwrap


def test_float64_plain_and_nnx_paths_do_not_use_float32():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx

from nlls_gram import UnderdeterminedLevenbergMarquardt


def assert_float64_tree(tree):
    leaves = jax.tree.leaves(tree)
    assert leaves
    for leaf in leaves:
        assert leaf.dtype == jnp.float64, (leaf.dtype, leaf)


def residual_fn(params, args, p):
    x, y = args
    return params["a"] * jnp.exp(params["b"] * x) - y


x = jnp.linspace(0.0, 2.0, 20, dtype=jnp.float64)
y = 2.0 * jnp.exp(-1.0 * x)
params = {
    "a": jnp.asarray(1.0, dtype=jnp.float64),
    "b": jnp.asarray(0.0, dtype=jnp.float64),
}
solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init(params, (x, y))
for _ in range(5):
    params, lm_state, info = solver.update(params, lm_state, (x, y))

assert_float64_tree(params)
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(jax.make_jaxpr(lambda p, s: solver.update(p, s, (x, y)))(params, lm_state))
assert "f32" not in jaxpr, jaxpr
solve_jaxpr = str(
    jax.make_jaxpr(
        lambda p: solver.solve(
            p, (x, y), max_steps=20, atol=1e-8, gtol=1e-10, xtol=1e-10
        ).x
    )(params)
)
assert "f32" not in solve_jaxpr, solve_jaxpr


def quadratic_residual(theta, target, p):
    return jnp.array([theta[0] ** 2 - target])


theta = jnp.asarray([1.9], dtype=jnp.float64)
target = jnp.asarray(4.0, dtype=jnp.float64)
solver = UnderdeterminedLevenbergMarquardt(
    quadratic_residual,
    init_damping=1e-12,
    geodesic_acceleration=True,
    geodesic_acceptance_ratio=1.0,
)
lm_state = solver.init(theta, target)
theta, lm_state, info = solver.update(theta, lm_state, target)

assert theta.dtype == jnp.float64
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(jax.make_jaxpr(lambda p, s: solver.update(p, s, target))(theta, lm_state))
assert "f32" not in jaxpr, jaxpr
solve_jaxpr = str(
    jax.make_jaxpr(
        lambda p: solver.solve(p, target, max_steps=20, atol=1e-10).x
    )(theta)
)
assert "f32" not in solve_jaxpr, solve_jaxpr


def linear_residual(theta, args, p):
    matrix, target = args
    return matrix @ theta - target


matrix = jnp.asarray([[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]], dtype=jnp.float64)
target = jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float64)
theta = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
solver = UnderdeterminedLevenbergMarquardt(
    linear_residual,
    init_damping=1e-2,
    linear_solver="lsmr",
    iterative_tol=1e-10,
    iterative_maxiter=20,
)
lm_state = solver.init(theta, (matrix, target))
theta, lm_state, info = solver.update(theta, lm_state, (matrix, target))

assert theta.dtype == jnp.float64
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(
    jax.make_jaxpr(lambda p, s: solver.update(p, s, (matrix, target)))(theta, lm_state)
)
assert "f32" not in jaxpr, jaxpr


matrix = jnp.asarray(
    [[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 2.0]],
    dtype=jnp.float64,
)
target = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
theta = jnp.zeros(matrix.shape[1], dtype=jnp.float64)
solver = UnderdeterminedLevenbergMarquardt(
    linear_residual,
    init_damping=1e-2,
    linear_solver="qr",
)
lm_state = solver.init(theta, (matrix, target))
theta, lm_state, info = solver.update(theta, lm_state, (matrix, target))

assert theta.dtype == jnp.float64
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(
    jax.make_jaxpr(lambda p, s: solver.update(p, s, (matrix, target)))(theta, lm_state)
)
assert "f32" not in jaxpr, jaxpr


class LinearModel(nnx.Module):
    def __init__(self):
        self.linear = nnx.Linear(
            1,
            1,
            use_bias=False,
            dtype=jnp.float64,
            param_dtype=jnp.float64,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x):
        return jnp.ravel(self.linear(x))


model = LinearModel()
graphdef, nnx_params = nnx.split(model, nnx.Param)
assert_float64_tree(nnx_params)

x_nnx = jnp.linspace(0.0, 2.0, 20, dtype=jnp.float64).reshape(-1, 1)
y_nnx = 2.0 * jnp.ravel(x_nnx)


def nnx_residual_fn(params, args, p):
    x, y = args
    model = nnx.merge(graphdef, params)
    return model(x) - y


solver = UnderdeterminedLevenbergMarquardt(nnx_residual_fn, init_damping=1e-12)
lm_state = solver.init(nnx_params, (x_nnx, y_nnx))
nnx_params, lm_state, info = solver.update(nnx_params, lm_state, (x_nnx, y_nnx))
trained = nnx.merge(graphdef, nnx_params)

assert_float64_tree(nnx_params)
assert trained.linear.kernel[...].dtype == jnp.float64
assert jnp.allclose(trained.linear.kernel[...], jnp.asarray([[2.0]], dtype=jnp.float64))
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(
    jax.make_jaxpr(lambda p, s: solver.update(p, s, (x_nnx, y_nnx)))(
        nnx_params, lm_state
    )
)
assert "f32" not in jaxpr, jaxpr
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_solve_with_float32_problem_under_x64_keeps_state_dtype_consistent():
    # Regression: solve(lm_state=None) used to build the damping in the default
    # float (float64 under x64) while update() returned it in the residual
    # dtype, mismatching the while_loop carry for float32 problems.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import LMStatus, UnderdeterminedLevenbergMarquardt


def residual(theta, args, p):
    return theta - args


solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
for jit in (True, False):
    result = solver.solve(
        jnp.zeros(1, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        max_steps=40,
        atol=1e-5,
        jit=jit,
    )
    assert int(result.status) == LMStatus.CONVERGED, int(result.status)
    assert result.lm_state.damping.dtype == jnp.float32, result.lm_state.damping.dtype
    assert result.x.dtype == jnp.float32

# All compute ops must stay float32; only call-boundary scalars (tolerances,
# default-dtype init damping) may arrive as f64 before being converted.
jaxpr = str(
    jax.make_jaxpr(
        lambda p, a: solver.solve(
            p, a, max_steps=40, atol=1e-5, gtol=1e-6, xtol=1e-6
        ).x
    )(jnp.zeros(1, dtype=jnp.float32), jnp.ones(1, dtype=jnp.float32))
)
for line in jaxpr.splitlines():
    stripped = line.strip()
    if " = " in stripped and ":f64[" in stripped.split(" = ")[0]:
        assert "convert_element_type" in stripped, stripped
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
