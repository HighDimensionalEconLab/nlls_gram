import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    identity_preconditioner,
    matern_state_space,
    metric_from_state_space,
)


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


# Matrix-free params (cg with an explicit identity preconditioner, plain lsmr)
# ride alongside the dense ones; they auto-skip without a GPU like the rest.
_MATRIX_FREE_GPU_KWARGS = {
    "gram_cg": {
        "iterative_tol": 1e-7,
        "iterative_maxiter": 10,
        "dual_preconditioner": identity_preconditioner(),
        "implicit_preconditioner": identity_preconditioner(),
    },
    "lsmr": {"iterative_tol": 1e-8, "iterative_maxiter": 10},
}


@pytest.mark.parametrize(
    "linear_solver", ["gram_cholesky", "qr", "augmented_qr", "gram_cg", "lsmr"]
)
def test_jitted_geodesic_update_runs_on_gpu(linear_solver):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    gpu = _gpu_devices()[0]
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
        **_MATRIX_FREE_GPU_KWARGS.get(linear_solver, {}),
    )

    with jax.default_device(gpu):
        theta = jnp.asarray([1.9])
        target = jnp.asarray(4.0)
        lm_state = solver.init(theta, target)

    @jax.jit
    def step(theta, lm_state, target):
        return solver.update(theta, lm_state, target)

    theta_new, state_new, info = step(theta, lm_state, target)
    theta_new.block_until_ready()

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    assert next(iter(theta_new.devices())).platform == "gpu"
    assert next(iter(state_new.damping.devices())).platform == "gpu"
    assert next(iter(info.loss.devices())).platform == "gpu"
    assert jnp.isfinite(theta_new[0])
    assert jnp.isfinite(info.acceleration_ratio)


@pytest.mark.parametrize(
    "linear_solver", ["gram_cholesky", "qr", "augmented_qr", "gram_cg", "lsmr"]
)
def test_jitted_geodesic_update_does_not_transfer_to_host(linear_solver):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    gpu = _gpu_devices()[0]
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
        **_MATRIX_FREE_GPU_KWARGS.get(linear_solver, {}),
    )

    with jax.default_device(gpu):
        theta = jnp.asarray([1.9])
        target = jnp.asarray(4.0)
        lm_state = solver.init(theta, target)

    @jax.jit
    def step(theta, lm_state, target):
        return solver.update(theta, lm_state, target)

    jax.block_until_ready(step(theta, lm_state, target))
    with jax.transfer_guard_device_to_host("disallow"):
        theta_new, state_new, info = step(theta, lm_state, target)
        jax.block_until_ready((theta_new, state_new, info))

    for leaf in jax.tree.leaves((theta_new, state_new, info)):
        assert next(iter(leaf.devices())).platform == "gpu"


def test_quasiseparable_matern_metric_runs_on_gpu():
    # The parallel and sequential apply paths must both run and agree on
    # the GPU backend (float32 here, so the dtype-aware default picks the
    # sequential path; parallel=True forces the associative scans).
    gpu = _gpu_devices()[0]
    with jax.default_device(gpu):
        t = jnp.cumsum(jnp.ones(512))
        x = jnp.sin(jnp.linspace(0.0, 6.0, 512))
        model = matern_state_space(1.3, 0.8, 2.5)
        default = metric_from_state_space(t, *model, nugget=1e-6)
        parallel = metric_from_state_space(t, *model, nugget=1e-6, parallel=True)
        out = jax.jit(default.solve)(x)
        out_parallel = jax.jit(parallel.solve)(x)
        jax.block_until_ready((out, out_parallel))

    assert next(iter(out.devices())).platform == "gpu"
    assert bool(jnp.all(jnp.isfinite(out)))
    assert jnp.allclose(out, out_parallel, rtol=1e-4, atol=1e-5)
    assert jnp.allclose(default.norm(x), parallel.norm(x), rtol=1e-4)


def multi_start_residual(theta, args, p):
    return jnp.array([theta[0] ** 2 - p, 0.1 * theta[1]])


def multi_start_draw(key, x, args):
    return jax.random.uniform(key, x.shape, x.dtype, 0.5, 3.0), args


def test_parallel_multi_start_runs_on_gpu():
    from nlls_gram import LMStatus, MultiStart

    gpu = _gpu_devices()[0]
    solver = LevenbergMarquardt(multi_start_residual, init_damping=1e-2)

    with jax.default_device(gpu):
        x0 = jnp.array([jnp.nan, 0.0])  # lane 0 fails; drawn lanes converge
        p = jnp.asarray(4.0)
        ms = MultiStart(
            key=jax.random.key(0), num_starts=8, draw=multi_start_draw, parallel=True
        )
        result = solver.solve(x0, p=p, max_steps=80, atol=1e-6, multi_start=ms)
        jax.block_until_ready(result)

    for leaf in jax.tree.leaves(result):
        assert next(iter(leaf.devices())).platform == "gpu"
    assert int(result.status) == LMStatus.CONVERGED
    assert int(result.multi_start.attempt) >= 1
    assert jnp.allclose(result.x[0] ** 2, p, atol=1e-4)

    # Sequential mode compiles and runs on the GPU as well.
    with jax.default_device(gpu):
        seq = MultiStart(key=jax.random.key(1), num_starts=4, draw=multi_start_draw)
        seq_result = solver.solve(x0, p=p, max_steps=80, atol=1e-6, multi_start=seq)
        jax.block_until_ready(seq_result)
    assert int(seq_result.status) == LMStatus.CONVERGED
    assert next(iter(seq_result.x.devices())).platform == "gpu"
