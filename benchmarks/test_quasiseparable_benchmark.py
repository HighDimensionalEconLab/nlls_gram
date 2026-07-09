import jax
import jax.numpy as jnp
import pytest

from nlls_gram import matern_state_space, metric_from_cholesky, metric_from_state_space
from nlls_gram.quasiseparable import cholesky, state_space_generators

SIGMA, ELL, NUGGET = 1.3, 0.8, 1e-6

DENSE_MAX_N = 1000


def _devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _grid(n, device):
    # Unit spacing keeps the Gram well-conditioned at every n.
    return jax.device_put(jnp.arange(n) * 1.0, device)


def _dense_metric(t, nu):
    tau = jnp.abs(t[:, None] - t[None, :])
    ft = jnp.sqrt(2.0 * nu) * tau / ELL
    if nu == 1.5:
        corr = (1.0 + ft) * jnp.exp(-ft)
    else:
        corr = (1.0 + ft + ft**2 / 3.0) * jnp.exp(-ft)
    K = SIGMA**2 * corr
    return metric_from_cholesky(jnp.linalg.cholesky(K + NUGGET * jnp.eye(t.shape[0])))


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("nu", [1.5, 2.5])
@pytest.mark.parametrize("variant", ["sequential", "parallel", "dense"])
def test_quasiseparable_apply(benchmark, platform, n, nu, variant):
    if not _devices(platform):
        pytest.skip(f"JAX {platform!r} backend is not available")
    if variant == "dense" and n > DENSE_MAX_N:
        pytest.skip("dense reference metric too large to materialize")

    device = _devices(platform)[0]
    t = _grid(n, device)
    if variant == "dense":
        metric = _dense_metric(t, nu)
    else:
        metric = metric_from_state_space(
            t,
            *matern_state_space(SIGMA, ELL, nu),
            nugget=NUGGET,
            parallel=variant == "parallel",
        )
    x = jax.device_put(jnp.sin(jnp.linspace(0.0, 20.0, n)), device)

    @jax.jit
    def apply(v):
        return metric.solve(v), metric.norm(v)

    jax.block_until_ready(apply(x))

    def run():
        out = apply(x)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("nu", [1.5, 2.5])
def test_quasiseparable_setup(benchmark, platform, n, nu):
    # The one-time sequential Cholesky setup — the hot path when the metric
    # is rebuilt from traced hyperparameters inside grad/vmap/sweeps.
    if not _devices(platform):
        pytest.skip(f"JAX {platform!r} backend is not available")

    device = _devices(platform)[0]
    t = _grid(n, device)

    @jax.jit
    def setup(points, sigma, ell):
        h, Pinf, transition = matern_state_space(sigma, ell, nu)
        d, p, q, A = state_space_generators(points, h, Pinf, transition)
        return cholesky(d + NUGGET, p, q, A)

    jax.block_until_ready(setup(t, SIGMA, ELL))

    def run():
        out = setup(t, SIGMA, ELL)
        jax.block_until_ready(out)
        return out

    benchmark(run)
