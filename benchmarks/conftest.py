import os

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_default_matmul_precision", "highest")


def _active_device():
    return jax.devices()[0]


def _device_bytes_limit(device):
    stats = device.memory_stats() or {}
    return stats.get("bytes_limit", 0)


def _is_datacenter_gpu(device):
    return device.platform == "gpu" and _device_bytes_limit(device) >= 40 * 1024**3


def benchmark_batch_size(device=None):
    override = os.environ.get("NLLS_BENCH_BATCH")
    if override is not None:
        return int(override)
    dev = _active_device() if device is None else device
    if _is_datacenter_gpu(dev):
        return 32
    return 8


def benchmark_direction_count(device=None):
    override = os.environ.get("NLLS_BENCH_DIRECTIONS")
    if override is not None:
        return int(override)
    dev = _active_device() if device is None else device
    if _is_datacenter_gpu(dev):
        return 32
    return 16


def _benchmark_supports_float64():
    device = _active_device()
    if device.platform == "metal":
        return False
    env = os.environ.get("JAX_ENABLE_X64")
    if env in ("1", "true", "True"):
        return True
    if env in ("0", "false", "False"):
        return False
    if device.platform == "cpu":
        return True
    return _is_datacenter_gpu(device)


@pytest.fixture
def device():
    dev = _active_device()
    with jax.default_device(dev):
        yield dev


@pytest.fixture(
    params=[
        pytest.param(jnp.float32, id="float32"),
        pytest.param(
            jnp.float64,
            id="float64",
            marks=pytest.mark.skipif(
                not _benchmark_supports_float64(),
                reason=(
                    "float64 benchmarks run on CPU, datacenter GPUs (>=40 GiB),"
                    " or with JAX_ENABLE_X64=1"
                ),
            ),
        ),
    ]
)
def dtype(request):
    with jax.enable_x64(request.param == jnp.float64):
        yield request.param
