import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    matern_state_space,
    repeated_shifted_dense_metric,
    repeated_shifted_state_space_metric,
)

SIGMA = 1.0
ELL = 10.0
EPSILON = 1e-7


def _devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _matern_gram(t, nu):
    distance = jnp.abs(t[:, None] - t[None, :])
    scaled = jnp.sqrt(2.0 * nu) * distance / ELL
    if nu == 0.5:
        correlation = jnp.exp(-scaled)
    elif nu == 1.5:
        correlation = (1.0 + scaled) * jnp.exp(-scaled)
    else:
        correlation = (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)
    return SIGMA**2 * correlation


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize(
    ("n", "repeats", "zero_pad_size", "rhs_columns", "nu"),
    [
        (41, 3, 2, 123, 0.5),
        (81, 7, 5, 567, 0.5),
        (96, 5, 3, 480, 2.5),
        (96, 9, 5, 865, 2.5),
        (96, 201, 101, 1, 2.5),
    ],
)
@pytest.mark.parametrize("callback", ["solve", "inv_sqrt", "inv_sqrt_transpose"])
def test_repeated_shifted_dense_apply(
    benchmark,
    platform,
    n,
    repeats,
    zero_pad_size,
    rhs_columns,
    nu,
    callback,
):
    devices = _devices(platform)
    if not devices:
        pytest.skip(f"JAX {platform!r} backend is not available")
    device = devices[0]
    t = jax.device_put(jnp.linspace(0.0, 40.0, n), device)
    K = _matern_gram(t, nu)
    metric = repeated_shifted_dense_metric(
        K,
        repeats=repeats,
        zero_pad_size=zero_pad_size,
        epsilon=EPSILON,
    )
    total_size = repeats * n + zero_pad_size
    shape = (total_size,) if rhs_columns == 1 else (total_size, rhs_columns)
    x = jax.device_put(jax.random.normal(jax.random.key(0), shape), device)
    apply = jax.jit(getattr(metric, callback))
    jax.block_until_ready(apply(x))
    benchmark.group = f"repeated-shifted-dense-{callback}"

    def run():
        out = apply(x)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
@pytest.mark.parametrize("parallel", [False, True])
def test_repeated_shifted_state_space_apply(benchmark, platform, n, nu, parallel):
    devices = _devices(platform)
    if not devices:
        pytest.skip(f"JAX {platform!r} backend is not available")
    device = devices[0]
    t = jax.device_put(jnp.arange(n) * 1.0, device)
    metric = repeated_shifted_state_space_metric(
        t,
        *matern_state_space(SIGMA, ELL, nu),
        repeats=3,
        zero_pad_size=2,
        epsilon=EPSILON,
        parallel=parallel,
    )
    x = jax.device_put(jnp.sin(jnp.linspace(0.0, 20.0, 3 * n + 2)), device)

    @jax.jit
    def apply(value):
        return metric.solve(value), metric.norm(value)

    jax.block_until_ready(apply(x))
    benchmark.group = "repeated-shifted-state-space"

    def run():
        out = apply(x)
        jax.block_until_ready(out)
        return out

    benchmark(run)
