import jax
import jax.numpy as jnp
import pytest

from nlls_gram import LevenbergMarquardt, identity_preconditioner

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

    ts = jnp.linspace(-1.0, 1.0, n_samples)
    centers = jnp.linspace(-1.2, 1.2, n_centers)
    scaled_distance = (ts[:, None] - centers[None, :]) / 0.08
    features = jnp.exp(-0.5 * scaled_distance**2) / jnp.sqrt(n_centers)
    theta_true = jnp.cos(jnp.linspace(0.0, 12.0, n_centers))
    ys = jnp.sin(features @ theta_true)
    x = 0.05 * jnp.sin(jnp.linspace(0.0, 8.0, n_centers))
    # device_put commits the arrays so the jitted step stays on the target device.
    x = jax.device_put(x, device)
    args = jax.device_put((features, ys), device)

    def residual(theta, args, p):
        features, ys = args
        return jnp.sin(features @ theta) - ys

    solver_kwargs = {
        "init_damping": 1e-2,
        "linear_solver": linear_solver,
    }
    if linear_solver == "gram_cg":
        solver_kwargs.update(
            {
                "iterative_tol": 0.0,
                "iterative_atol": 0.0,
                "iterative_maxiter": ITERATIVE_MAXITER,
                "dual_preconditioner": identity_preconditioner(),
                "implicit_preconditioner": identity_preconditioner(),
            }
        )

    base_solver = LevenbergMarquardt(residual, **solver_kwargs)
    solver = LevenbergMarquardt(
        residual,
        **solver_kwargs,
        geodesic_acceleration=geodesic_acceleration,
    )

    lm_state = jax.device_put(base_solver.init(x, args), device)

    @jax.jit
    def first_step(x, lm_state):
        return base_solver.update(x, lm_state, args)

    x, lm_state, _ = first_step(x, lm_state)
    jax.block_until_ready((x, lm_state))

    @jax.jit
    def step(x, lm_state):
        return solver.update(x, lm_state, args)

    return x, lm_state, step


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "linear_solver",
    [
        "gram_cholesky",
        "qr",
        "gram_cg",
    ],
)
@pytest.mark.parametrize("geodesic_acceleration", [False, True])
def test_large_rbf_interpolation_second_update(
    benchmark, platform, linear_solver, geodesic_acceleration
):
    if not _devices(platform):
        pytest.skip(f"JAX {platform!r} backend is not available")

    x, lm_state, step = _make_large_interpolation_problem(
        platform=platform,
        linear_solver=linear_solver,
        geodesic_acceleration=geodesic_acceleration,
    )

    warmup = step(x, lm_state)
    jax.block_until_ready(warmup)

    def run():
        out = step(x, lm_state)
        jax.block_until_ready(out)
        return out

    benchmark(run)
