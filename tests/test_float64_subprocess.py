import subprocess
import sys
import textwrap


def test_float64_plain_and_nnx_paths_do_not_use_float32():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx

from nlls_gram import GramLevenbergMarquardt


def assert_float64_tree(tree):
    leaves = jax.tree.leaves(tree)
    assert leaves
    for leaf in leaves:
        assert leaf.dtype == jnp.float64, (leaf.dtype, leaf)


def residual_fn(params, batch):
    x, y = batch
    return params["a"] * jnp.exp(params["b"] * x) - y


x = jnp.linspace(0.0, 2.0, 20, dtype=jnp.float64)
y = 2.0 * jnp.exp(-1.0 * x)
params = {
    "a": jnp.asarray(1.0, dtype=jnp.float64),
    "b": jnp.asarray(0.0, dtype=jnp.float64),
}
solver = GramLevenbergMarquardt(residual_fn, init_damping=1e-2)
state = solver.init()
for _ in range(5):
    params, state, info = solver.update(params, state, (x, y))

assert_float64_tree(params)
assert state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
jaxpr = str(jax.make_jaxpr(lambda p, s: solver.update(p, s, (x, y)))(params, state))
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


def nnx_residual_fn(params, batch):
    x, y = batch
    model = nnx.merge(graphdef, params)
    return model(x) - y


solver = GramLevenbergMarquardt(
    nnx_residual_fn, init_damping=0.0, solve_method="normal"
)
state = solver.init()
nnx_params, state, info = solver.update(nnx_params, state, (x_nnx, y_nnx))
trained = nnx.merge(graphdef, nnx_params)

assert_float64_tree(nnx_params)
assert trained.linear.kernel[...].dtype == jnp.float64
assert jnp.allclose(trained.linear.kernel[...], jnp.asarray([[2.0]], dtype=jnp.float64))
assert state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
jaxpr = str(
    jax.make_jaxpr(lambda p, s: solver.update(p, s, (x_nnx, y_nnx)))(
        nnx_params, state
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
