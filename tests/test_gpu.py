"""GPU execution and no-host-transfer tests for LevenbergMarquardt and Newton."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    QR,
    SVD,
    AnnealRidge,
    BlockEigenPreconditioner,
    Cholesky,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    MultiStart,
    Newton,
    RepeatedFactorMetric,
    SolverStatus,
)
from nlls_gram.experimental import StateSpaceMetric, matern_state_space

GPU_PLATFORMS = ("gpu", "metal")


def _gpu_devices():
    try:
        devices = jax.devices()
    except RuntimeError:
        return []
    return [device for device in devices if device.platform in GPU_PLATFORMS]


def _gpu_supports_float64():
    devices = _gpu_devices()
    if not devices or devices[0].platform != "gpu":
        return False
    env = os.environ.get("JAX_ENABLE_X64")
    if env in ("1", "true", "True"):
        return True
    if env in ("0", "false", "False"):
        return False
    stats = devices[0].memory_stats() or {}
    return stats.get("bytes_limit", 0) >= 40 * 1024**3


pytestmark = pytest.mark.skipif(
    not _gpu_devices(),
    reason="JAX GPU backend is not available",
)


@pytest.fixture(
    params=[
        pytest.param(jnp.float32, id="float32"),
        pytest.param(
            jnp.float64,
            id="float64",
            marks=pytest.mark.skipif(
                not _gpu_supports_float64(),
                reason="float64 GPU tests require >=40 GiB or JAX_ENABLE_X64=1",
            ),
        ),
    ]
)
def dtype(request):
    with jax.enable_x64(request.param == jnp.float64):
        yield request.param


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(form="gram"),
        QR(),
        GramCG(IdentityPreconditioner(), tol=1e-7, maxiter=10),
    ],
)
def test_jitted_geodesic_update_runs_on_gpu_without_host_transfer(
    dtype, solver_cls, linear_solver
):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target], dtype=theta.dtype)

    gpu = _gpu_devices()[0]
    solver = solver_cls(
        residual,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    with jax.default_device(gpu):
        theta = jnp.asarray([1.9], dtype=dtype)
        target = jnp.asarray(4.0, dtype=dtype)
        state = solver.init(theta, target)

    @jax.jit
    def step(theta, state, target):
        return solver.update(theta, state, target)

    jax.block_until_ready(step(theta, state, target))
    with jax.transfer_guard_device_to_host("disallow"):
        theta_new, state_new, info = step(theta, state, target)
        jax.block_until_ready((theta_new, state_new, info))

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    for leaf in jax.tree.leaves((theta_new, state_new, info)):
        assert next(iter(leaf.devices())).platform in GPU_PLATFORMS


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize(
    ("linear_solver", "ad_solver"),
    [
        (Cholesky(form="gram"), SVD()),
        (QR(), Cholesky()),
        (GramCG(IdentityPreconditioner(), tol=1e-7, maxiter=100), None),
    ],
)
def test_unpenalized_solve_and_implicit_ad_on_gpu(
    dtype, solver_cls, linear_solver, ad_solver
):
    rng = np.random.default_rng(17)
    m, block, repeats, free = 3, 2, 2, 1
    n = block * repeats + free
    A_np = rng.normal(size=(m, n))
    b_np = rng.normal(size=m)
    b_dot_np = rng.normal(size=m)
    root = rng.normal(size=(block, block + 2))
    K_np = root @ root.T + 0.5 * np.eye(block)
    W_np = np.eye(n)
    for j in range(repeats):
        W_np[j * block : (j + 1) * block, j * block : (j + 1) * block] = K_np
    W_inv = np.linalg.inv(W_np)
    expected_x = W_inv @ A_np.T @ np.linalg.solve(A_np @ W_inv @ A_np.T, b_np)
    expected_dx = W_inv @ A_np.T @ np.linalg.solve(A_np @ W_inv @ A_np.T, b_dot_np)

    gpu = _gpu_devices()[0]
    atol = 1e-5 if dtype == jnp.float32 else 1e-10
    rtol_check = 3e-3 if dtype == jnp.float32 else 1e-6

    with jax.default_device(gpu):
        A = jnp.asarray(A_np, dtype=dtype)
        b = jnp.asarray(b_np, dtype=dtype)
        b_dot = jnp.asarray(b_dot_np, dtype=dtype)
        F = jnp.linalg.cholesky(jnp.asarray(K_np, dtype=dtype), upper=True)
        metric = RepeatedFactorMetric(F, repeats=repeats)
        x0 = jnp.zeros(n, dtype=dtype)
        solver = solver_cls(
            lambda x, _, p: A @ x - p,
            metric=metric,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
        )

        def run(p):
            return solver.solve(x0, p=p, max_steps=100, atol=atol).x

        @jax.jit
        def evaluate(p, p_dot):
            x, tangent = jax.jvp(run, (p,), (p_dot,))
            _, vjp_fn = jax.vjp(run, p)
            (cotangent,) = vjp_fn(jnp.ones(n, dtype=dtype))
            return x, tangent, cotangent

        x, tangent, cotangent = evaluate(b, b_dot)
        jax.block_until_ready((x, tangent, cotangent))

    for leaf in (x, tangent, cotangent):
        assert leaf.dtype == dtype
        assert next(iter(leaf.devices())).platform in GPU_PLATFORMS
    np.testing.assert_allclose(np.asarray(x), expected_x, rtol=rtol_check, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(tangent), expected_dx, rtol=rtol_check, atol=1e-4
    )


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize("mode", ["cholesky", "qr", "cg_block_eigen"])
def test_ridge_solve_continuation_and_implicit_ad_on_gpu(dtype, solver_cls, mode):
    rng = np.random.default_rng(31)
    m, block, repeats, free = 3, 2, 2, 1
    n_m = block * repeats
    n = n_m + free
    A_np = rng.normal(size=(m, n))
    b_np = rng.normal(size=m)
    b_dot_np = rng.normal(size=m)
    root = rng.normal(size=(block, block + 2))
    K_np = root @ root.T + 0.5 * np.eye(block)
    W_np = np.zeros((n, n))
    for j in range(repeats):
        W_np[j * block : (j + 1) * block, j * block : (j + 1) * block] = K_np

    ridge_floor = 1e-3 if dtype == jnp.float32 else 1e-6
    expected_x = np.linalg.solve(A_np.T @ A_np + ridge_floor * W_np, A_np.T @ b_np)
    expected_dx = np.linalg.solve(A_np.T @ A_np + ridge_floor * W_np, A_np.T @ b_dot_np)

    gpu = _gpu_devices()[0]
    gtol = 1e-5 if dtype == jnp.float32 else 1e-10
    rtol_check = 5e-3 if dtype == jnp.float32 else 1e-5

    with jax.default_device(gpu):
        A = jnp.asarray(A_np, dtype=dtype)
        b = jnp.asarray(b_np, dtype=dtype)
        b_dot = jnp.asarray(b_dot_np, dtype=dtype)
        F = jnp.linalg.cholesky(jnp.asarray(K_np, dtype=dtype), upper=True)
        metric = RepeatedFactorMetric(F, repeats=repeats)
        x0 = jnp.zeros(n, dtype=dtype)

        if mode == "cholesky":
            linear_solver = Cholesky()
            ad_solver = Cholesky()
        elif mode == "qr":
            linear_solver = QR()
            ad_solver = Cholesky()
        else:
            precond = BlockEigenPreconditioner(
                [
                    (jnp.eye(n_m, dtype=dtype)[None], 1.0),
                    (jnp.eye(free, dtype=dtype)[None], 0.0),
                ],
                jnp.arange(n),
            )
            linear_solver = CG(precond, tol=1e-7, maxiter=100)
            ad_solver = CG(None, tol=1e-7, maxiter=100)

        anneal = AnnealRidge(ridge_floor=ridge_floor, decrease=0.1)
        user_state0 = anneal.init_state(dtype=dtype)
        solver = solver_cls(
            lambda x, _, p: A @ x - p,
            metric=metric,
            ridge=ridge_floor * 10.0,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
        )

        def run(p):
            return solver.solve(
                x0,
                p=p,
                max_steps=120,
                gtol=gtol,
                callback=anneal,
                user_state=user_state0,
            ).x

        @jax.jit
        def evaluate(p, dp):
            x, tangent = jax.jvp(run, (p,), (dp,))
            grad = jax.grad(lambda q: jnp.sum(run(q)))(p)
            return x, tangent, grad

        x, tangent, grad = evaluate(b, b_dot)
        jax.block_until_ready((x, tangent, grad))

    for leaf in (x, tangent, grad):
        assert leaf.dtype == dtype
        assert next(iter(leaf.devices())).platform in GPU_PLATFORMS
    np.testing.assert_allclose(np.asarray(x), expected_x, rtol=rtol_check, atol=2e-4)
    np.testing.assert_allclose(
        np.asarray(tangent), expected_dx, rtol=rtol_check, atol=2e-4
    )


def test_state_space_metric_multi_start_and_failed_lane_ad_on_gpu(dtype):
    gpu = _gpu_devices()[0]
    rtol = 1e-4 if dtype == jnp.float32 else 1e-10
    atol = 1e-5 if dtype == jnp.float32 else 1e-10

    with jax.default_device(gpu):
        # 1. StateSpaceMetric sequential & parallel
        n, repeats = 128, 2
        t = jnp.cumsum(jnp.ones(n, dtype=dtype))
        x = jnp.sin(jnp.linspace(0.0, 6.0, repeats * n, dtype=dtype))
        model = matern_state_space(
            jnp.asarray(1.3, dtype=dtype), jnp.asarray(0.8, dtype=dtype), 2.5
        )
        seq = StateSpaceMetric(t, *model, repeats=repeats, epsilon=1e-5, parallel=False)
        par = StateSpaceMetric(t, *model, repeats=repeats, epsilon=1e-5, parallel=True)
        out_seq = jax.jit(lambda v: seq.factor_solve(v, None))(x)
        out_par = jax.jit(lambda v: par.factor_solve(v, None))(x)
        jax.block_until_ready((out_seq, out_par))
        assert jnp.allclose(out_seq, out_par, rtol=rtol, atol=atol)

        # 2. MultiStart + vmapped failed-lane AD
        solver = LevenbergMarquardt(
            lambda th, _, p: th - p, cache_jacobian=False, geodesic_acceleration=False
        )
        x0 = jnp.zeros(1, dtype=dtype)
        ps = jnp.asarray([0.0, 2.0], dtype=dtype)
        atols = jnp.asarray([1e-5, 0.0], dtype=dtype)

        def solved(values):
            return jax.vmap(
                lambda val, tol: solver.solve(
                    x0, p=val, max_steps=1, max_steps_is_success=False, atol=tol
                ).x[0]
            )(values, atols)

        _, tangent = jax.jit(lambda v: jax.jvp(solved, (v,), (jnp.ones_like(v),)))(ps)
        assert jnp.array_equal(tangent, jnp.asarray([1.0, 0.0], dtype=dtype))

        ms = MultiStart(
            key=jax.random.key(0),
            num_starts=3,
            draw=lambda k, x_in, a: jnp.ones_like(x_in),
            parallel=True,
        )
        res_ms = solver.solve(
            jnp.array([jnp.nan], dtype=dtype),
            p=jnp.asarray([2.0], dtype=dtype),
            max_steps=10,
            atol=atol,
            multi_start=ms,
        )
        assert int(res_ms.status) == int(SolverStatus.CONVERGED)
