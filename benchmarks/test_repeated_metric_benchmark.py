import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    blockdiag_metric,
    metric_from_cholesky,
    metric_from_diagonal,
    repeated_blockdiag_metric,
)

# Representative two-country sizing from the motivating issue: repeats copies of
# a block_size kernel block plus a small trailing block, many RHS columns.
BLOCK_SIZE = 96
REPEATS = 5
ADDITIONAL_SIZE = 3
RHS_COLUMNS = 480


def _devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _blocks(device):
    idx = jnp.arange(BLOCK_SIZE)
    K = jax.device_put(0.6 ** jnp.abs(idx[:, None] - idx[None, :]), device)
    block = metric_from_cholesky(jnp.linalg.cholesky(K))
    additional = metric_from_diagonal(
        jax.device_put(jnp.linspace(0.5, 2.0, ADDITIONAL_SIZE), device)
    )
    return block, additional


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("layout", ["repeated", "blockdiag"])
@pytest.mark.parametrize("callback", ["solve", "inv_sqrt", "inv_sqrt_transpose"])
def test_repeated_metric_apply(benchmark, platform, layout, callback):
    if not _devices(platform):
        pytest.skip(f"JAX {platform!r} backend is not available")

    device = _devices(platform)[0]
    block, additional = _blocks(device)
    if layout == "repeated":
        metric = repeated_blockdiag_metric(
            block, BLOCK_SIZE, REPEATS, additional=(additional, ADDITIONAL_SIZE)
        )
    else:
        metric = blockdiag_metric(
            [(block, BLOCK_SIZE)] * REPEATS + [(additional, ADDITIONAL_SIZE)]
        )

    total = BLOCK_SIZE * REPEATS + ADDITIONAL_SIZE
    X = jax.device_put(
        jax.random.normal(jax.random.PRNGKey(0), (total, RHS_COLUMNS)), device
    )
    apply = jax.jit(getattr(metric, callback))
    jax.block_until_ready(apply(X))

    # Group repeated vs blockdiag for the same callback so the ratio is visible.
    # No wall-clock assertion (timing is machine-dependent); the batching
    # guarantee is the fast unit test in tests/test_gram_lm.py.
    benchmark.group = f"repeated-metric-apply-{callback}"

    def run():
        out = apply(X)
        jax.block_until_ready(out)
        return out

    benchmark(run)
