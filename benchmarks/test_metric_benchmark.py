import jax
import jax.numpy as jnp
import pytest

from nlls_gram import RepeatedFactorMetric
from nlls_gram.experimental import StateSpaceMetric, matern_state_space

SIGMA = 1.0
ELL = 10.0


def _epsilon(dtype):
    return 1e-5 if dtype == jnp.float32 else 1e-7


def _matern_gram(t, nu):
    distance = jnp.abs(t[:, None] - t[None, :])
    scaled = jnp.sqrt(jnp.asarray(2.0 * nu, dtype=t.dtype)) * distance / ELL
    if nu == 0.5:
        correlation = jnp.exp(-scaled)
    elif nu == 1.5:
        correlation = (1.0 + scaled) * jnp.exp(-scaled)
    else:
        correlation = (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)
    return jnp.asarray(SIGMA**2, dtype=t.dtype) * correlation


@pytest.mark.parametrize(
    ("n", "repeats", "rhs_columns", "nu"),
    [
        (96, 9, 865, 2.5),
        (96, 201, 1, 2.5),
    ],
)
@pytest.mark.parametrize(
    "callback", ["factor_apply", "factor_solve", "factor_solve_transpose"]
)
def test_repeated_factor_metric_apply(
    benchmark, device, dtype, n, repeats, rhs_columns, nu, callback
):
    eps = _epsilon(dtype)
    with jax.default_device(device):
        t = jnp.linspace(0.0, 40.0, n, dtype=dtype)
        K = _matern_gram(t, nu)
        shifted = K + jnp.asarray(eps, dtype=dtype) * jnp.eye(n, dtype=dtype)
        metric = RepeatedFactorMetric(
            jnp.linalg.cholesky(shifted, upper=True),
            repeats=repeats,
            free_scale=eps,
        )
        total_size = repeats * n
        shape = (total_size,) if rhs_columns == 1 else (total_size, rhs_columns)
        x = jax.random.normal(jax.random.key(0), shape, dtype=dtype)
        apply = jax.jit(lambda v: getattr(metric, callback)(v, None))
        jax.block_until_ready(apply(x))

    benchmark.group = f"repeated-factor-{callback}"

    def run():
        out = apply(x)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("n", [10_000, 100_000])
@pytest.mark.parametrize("nu", [1.5, 2.5])
@pytest.mark.parametrize("parallel", [False, True])
def test_state_space_metric_apply(benchmark, device, dtype, n, nu, parallel):
    eps = _epsilon(dtype)
    with jax.default_device(device):
        t = jnp.arange(n, dtype=dtype)
        metric = StateSpaceMetric(
            t,
            *matern_state_space(SIGMA, ELL, nu),
            repeats=3,
            epsilon=eps,
            parallel=parallel,
        )
        x = jnp.sin(jnp.linspace(0.0, 20.0, 3 * n, dtype=dtype))

        @jax.jit
        def apply(value):
            return metric.factor_solve(value, None), metric.norm(value, None)

        jax.block_until_ready(apply(x))

    benchmark.group = "state-space-metric"

    def run():
        out = apply(x)
        jax.block_until_ready(out)
        return out

    benchmark(run)
