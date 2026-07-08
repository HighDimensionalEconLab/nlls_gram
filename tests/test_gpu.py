import jax
import jax.numpy as jnp
import pytest

from nlls_gram import UnderdeterminedLevenbergMarquardt


def _gpu_devices():
    try:
        devices = jax.devices()
    except RuntimeError:
        return []
    return [device for device in devices if device.platform == "gpu"]


pytestmark = pytest.mark.skipif(
    not _gpu_devices(),
    reason="JAX GPU backend is not available",
)


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr"])
def test_jitted_geodesic_update_runs_on_gpu(linear_solver):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    gpu = _gpu_devices()[0]
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    with jax.default_device(gpu):
        theta = jnp.asarray([1.9])
        target = jnp.asarray(4.0)
        state = solver.init(theta, target)

    @jax.jit
    def step(theta, state, target):
        return solver.update(theta, state, target)

    theta_new, state_new, info = step(theta, state, target)
    theta_new.block_until_ready()

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    assert next(iter(theta_new.devices())).platform == "gpu"
    assert next(iter(state_new.damping.devices())).platform == "gpu"
    assert next(iter(info.loss.devices())).platform == "gpu"
    assert jnp.isfinite(theta_new[0])
    assert jnp.isfinite(info.acceleration_ratio)


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr"])
def test_jitted_geodesic_update_does_not_transfer_to_host(linear_solver):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    gpu = _gpu_devices()[0]
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    with jax.default_device(gpu):
        theta = jnp.asarray([1.9])
        target = jnp.asarray(4.0)
        state = solver.init(theta, target)

    @jax.jit
    def step(theta, state, target):
        return solver.update(theta, state, target)

    jax.block_until_ready(step(theta, state, target))
    with jax.transfer_guard_device_to_host("disallow"):
        theta_new, state_new, info = step(theta, state, target)
        jax.block_until_ready((theta_new, state_new, info))

    for leaf in jax.tree.leaves((theta_new, state_new, info)):
        assert next(iter(leaf.devices())).platform == "gpu"
