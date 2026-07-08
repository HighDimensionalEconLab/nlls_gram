import jax
import jax.numpy as jnp
import pytest

from nlls_gram import UnderdeterminedLevenbergMarquardt

ITERATIVE_MAXITER = 8


def _devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _make_large_interpolation_problem(
    *, platform, linear_solver, geodesic_acceleration
):
    device = _devices(platform)[0]
    n_samples = 1024
    n_centers = 8192

    with jax.default_device(device):
        x = jnp.linspace(-1.0, 1.0, n_samples)
        centers = jnp.linspace(-1.2, 1.2, n_centers)
        scaled_distance = (x[:, None] - centers[None, :]) / 0.08
        features = jnp.exp(-0.5 * scaled_distance**2) / jnp.sqrt(n_centers)
        theta_true = jnp.cos(jnp.linspace(0.0, 12.0, n_centers))
        y = jnp.sin(features @ theta_true)
        params = 0.05 * jnp.sin(jnp.linspace(0.0, 8.0, n_centers))
        aux = (features, y)

    def residual(theta, aux, p):
        features, y = aux
        return jnp.sin(features @ theta) - y

    solver_kwargs = {
        "init_damping": 1e-2,
        "linear_solver": linear_solver,
    }
    if linear_solver in ("cg", "lsmr"):
        solver_kwargs.update(
            {
                "iterative_tol": 0.0,
                "iterative_atol": 0.0,
                "iterative_maxiter": ITERATIVE_MAXITER,
            }
        )

    base_solver = UnderdeterminedLevenbergMarquardt(residual, **solver_kwargs)
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        **solver_kwargs,
        geodesic_acceleration=geodesic_acceleration,
    )

    with jax.default_device(device):
        state = base_solver.init()
        state = type(state)(jnp.asarray(state.damping))

    @jax.jit
    def first_step(params, state):
        return base_solver.update(params, state, aux)

    params, state, _ = first_step(params, state)
    jax.block_until_ready((params, state))

    @jax.jit
    def step(params, state):
        return solver.update(params, state, aux)

    return params, state, step


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "linear_solver",
    [
        "cholesky",
        "qr",
        "cg",
        "lsmr",
    ],
)
@pytest.mark.parametrize("geodesic_acceleration", [False, True])
def test_large_rbf_interpolation_second_update(
    benchmark, platform, linear_solver, geodesic_acceleration
):
    if not _devices(platform):
        pytest.skip(f"JAX {platform!r} backend is not available")

    params, state, step = _make_large_interpolation_problem(
        platform=platform,
        linear_solver=linear_solver,
        geodesic_acceleration=geodesic_acceleration,
    )

    warmup = step(params, state)
    jax.block_until_ready(warmup)

    def run():
        out = step(params, state)
        jax.block_until_ready(out)
        return out

    benchmark(run)
