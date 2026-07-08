import jax
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt


def _make_problem(*, n_residuals, n_params):
    grid = jnp.linspace(-1.0, 1.0, n_residuals * n_params)
    design = jnp.reshape(grid, (n_residuals, n_params)) / jnp.sqrt(n_params)
    theta_true = jnp.cos(jnp.linspace(0.0, 1.0, n_params))
    y = jnp.sin(design @ theta_true)
    params = jnp.zeros(n_params)

    def residual(theta, args, p):
        design, y = args
        return jnp.sin(design @ theta) - y

    solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
    lm_state = solver.init(params, (design, y))

    @jax.jit
    def step(params, lm_state):
        return solver.update(params, lm_state, (design, y))

    return params, lm_state, step


def _benchmark_update(benchmark, *, n_residuals, n_params):
    params, lm_state, step = _make_problem(n_residuals=n_residuals, n_params=n_params)

    warmup = step(params, lm_state)
    jax.block_until_ready(warmup)

    def run():
        out = step(params, lm_state)
        jax.block_until_ready(out)
        return out

    benchmark(run)


def test_gram_update_overparameterized(benchmark):
    _benchmark_update(benchmark, n_residuals=16, n_params=256)
