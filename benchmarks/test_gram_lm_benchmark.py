import jax
import jax.numpy as jnp

from nlls_gram import GramLevenbergMarquardt


def _make_problem(*, n_residuals, n_params, solve_method):
    grid = jnp.linspace(-1.0, 1.0, n_residuals * n_params)
    design = jnp.reshape(grid, (n_residuals, n_params)) / jnp.sqrt(n_params)
    theta_true = jnp.cos(jnp.linspace(0.0, 1.0, n_params))
    y = jnp.sin(design @ theta_true)
    params = jnp.zeros(n_params)

    def residual(theta, batch):
        design, y = batch
        return jnp.sin(design @ theta) - y

    solver = GramLevenbergMarquardt(
        residual, init_damping=1e-2, solve_method=solve_method
    )
    state = solver.init()

    @jax.jit
    def step(params, state):
        return solver.update(params, state, (design, y))

    return params, state, step


def _benchmark_update(benchmark, *, n_residuals, n_params, solve_method):
    params, state, step = _make_problem(
        n_residuals=n_residuals, n_params=n_params, solve_method=solve_method
    )

    warmup = step(params, state)
    jax.block_until_ready(warmup)

    def run():
        out = step(params, state)
        jax.block_until_ready(out)
        return out

    benchmark(run)


def test_gram_update_overparameterized(benchmark):
    _benchmark_update(benchmark, n_residuals=16, n_params=256, solve_method="gram")


def test_normal_update_low_parameter_count(benchmark):
    _benchmark_update(benchmark, n_residuals=256, n_params=16, solve_method="normal")
