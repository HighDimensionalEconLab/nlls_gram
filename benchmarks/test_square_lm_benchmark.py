import jax
import jax.numpy as jnp
import pytest

from nlls_gram import SquareLevenbergMarquardt

# The DAE stage pattern from issue #14: 32 repeated warm-started square
# solves over slowly drifting targets. The direct-Newton fori_loop variant is
# the baseline the square solver is gated against (target: <= ~2x overhead
# for the small systems that dominate DAE stage solves).

N_SOLVES = 32
NEWTON_STEPS = 4


def _devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _make_problem(n, device):
    W = jax.device_put(0.1 * jax.random.normal(jax.random.PRNGKey(66), (n, n)), device)
    b0 = jax.device_put(jax.random.normal(jax.random.PRNGKey(67), (n,)), device)
    drift = jax.device_put(
        0.02 * jax.random.normal(jax.random.PRNGKey(68), (N_SOLVES, n)), device
    )
    targets = b0 + jnp.cumsum(drift, axis=0)

    def residual(z, args, p):
        return z + jnp.tanh(W @ z) - p

    return residual, targets


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("n", [1, 4, 8])
@pytest.mark.parametrize("method", ["square_lm", "direct_newton"])
def test_warm_started_stage_solves(benchmark, platform, n, method):
    if not _devices(platform):
        pytest.skip(f"JAX {platform!r} backend is not available")
    device = _devices(platform)[0]
    residual, targets = _make_problem(n, device)
    z0 = jax.device_put(jnp.zeros(n), device)

    if method == "square_lm":
        solver = SquareLevenbergMarquardt(residual)

        def stage(z, target):
            result = solver.solve(z, p=target, max_steps=8, atol=1e-5)
            return result.x, result.status

    else:

        def stage(z, target):
            def newton(_, z):
                J = jax.jacfwd(residual, argnums=0)(z, None, target)
                return z - jnp.linalg.solve(J, residual(z, None, target))

            z = jax.lax.fori_loop(0, NEWTON_STEPS, newton, z)
            return z, jnp.asarray(0, dtype=jnp.int32)

    @jax.jit
    def sweep(z):
        def body(z, target):
            z_next, status = stage(z, target)
            return z_next, status

        z_final, statuses = jax.lax.scan(body, z, targets)
        return z_final, statuses

    jax.block_until_ready(sweep(z0))

    def run():
        out = sweep(z0)
        jax.block_until_ready(out)
        return out

    benchmark(run)
