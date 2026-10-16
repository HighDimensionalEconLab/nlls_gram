import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    LU,
    QR,
    SVD,
    AnnealRidge,
    BlockEigenPreconditioner,
    Cholesky,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMState,
    LMStatus,
    MultiStart,
    NystromPreconditioner,
    PaddedPreconditioner,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    ShermanMorrisonPreconditioner,
    SolverContext,
    WoodburyPreconditioner,
)
from nlls_gram.experimental import StateSpaceMetric, matern_state_space

GPU_PLATFORMS = ("gpu", "metal")


def _gpu_devices():
    try:
        devices = jax.devices()
    except RuntimeError:
        return []
    return [device for device in devices if device.platform in GPU_PLATFORMS]


def _gpu_bytes_limit():
    devices = _gpu_devices()
    if not devices:
        return 0
    stats = devices[0].memory_stats() or {}
    return stats.get("bytes_limit", 0)


def _gpu_supports_float64():
    devices = _gpu_devices()
    if not devices or devices[0].platform != "gpu":
        return False
    env = os.environ.get("JAX_ENABLE_X64")
    if env in ("1", "true", "True"):
        return True
    if env in ("0", "false", "False"):
        return False
    return _gpu_bytes_limit() >= 40 * 1024**3


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
                reason=(
                    "float64 GPU tests require a datacenter GPU (>=40 GiB)"
                    " or JAX_ENABLE_X64=1"
                ),
            ),
        ),
    ]
)
def dtype(request):
    with jax.enable_x64(request.param == jnp.float64):
        yield request.param


GPU_SOLVERS = [
    Cholesky(form="gram"),
    Cholesky(form="normal"),
    QR(),
    CG(IdentityPreconditioner(), tol=1e-7, maxiter=10),
    GramCG(IdentityPreconditioner(), tol=1e-7, maxiter=10),
]


@pytest.mark.parametrize("linear_solver", GPU_SOLVERS)
def test_jitted_geodesic_update_runs_on_gpu(dtype, linear_solver):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target], dtype=theta.dtype)

    gpu = _gpu_devices()[0]
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    with jax.default_device(gpu):
        theta = jnp.asarray([1.9], dtype=dtype)
        target = jnp.asarray(4.0, dtype=dtype)
        lm_state = solver.init(theta, target)

    @jax.jit
    def step(theta, lm_state, target):
        return solver.update(theta, lm_state, target)

    theta_new, state_new, info = step(theta, lm_state, target)
    theta_new.block_until_ready()

    assert bool(info.accepted)
    assert bool(info.used_geodesic)
    assert theta_new.dtype == dtype
    assert state_new.damping.dtype == dtype
    assert info.loss.dtype == dtype
    assert next(iter(theta_new.devices())).platform in GPU_PLATFORMS
    assert next(iter(state_new.damping.devices())).platform in GPU_PLATFORMS
    assert next(iter(info.loss.devices())).platform in GPU_PLATFORMS
    assert jnp.isfinite(theta_new[0])
    assert jnp.isfinite(info.acceleration_ratio)


@pytest.mark.parametrize("linear_solver", GPU_SOLVERS)
def test_jitted_geodesic_update_does_not_transfer_to_host(dtype, linear_solver):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target], dtype=theta.dtype)

    gpu = _gpu_devices()[0]
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    with jax.default_device(gpu):
        theta = jnp.asarray([1.9], dtype=dtype)
        target = jnp.asarray(4.0, dtype=dtype)
        lm_state = solver.init(theta, target)

    @jax.jit
    def step(theta, lm_state, target):
        return solver.update(theta, lm_state, target)

    jax.block_until_ready(step(theta, lm_state, target))
    with jax.transfer_guard_device_to_host("disallow"):
        theta_new, state_new, info = step(theta, lm_state, target)
        jax.block_until_ready((theta_new, state_new, info))

    for leaf in jax.tree.leaves((theta_new, state_new, info)):
        assert next(iter(leaf.devices())).platform in GPU_PLATFORMS


@pytest.mark.parametrize(
    ("name", "linear_solver", "ad_solver"),
    [
        ("chol_gram_svd", Cholesky(form="gram"), SVD()),
        ("chol_normal_chol", Cholesky(form="normal"), Cholesky()),
        ("qr_svd", QR(), SVD()),
        (
            "cg_gram_cg",
            CG(IdentityPreconditioner(), tol=1e-7, maxiter=100),
            GramCG(IdentityPreconditioner(), tol=1e-7, maxiter=100),
        ),
        (
            "gram_cg_auto",
            GramCG(IdentityPreconditioner(), tol=1e-7, maxiter=100),
            None,
        ),
    ],
)
def test_metric_lm_solve_and_implicit_ad_on_gpu(dtype, name, linear_solver, ad_solver):
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

        def residual(x, args, p):
            return A @ x - p

        solver = LevenbergMarquardt(
            residual,
            metric=metric,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
        )

        def run(p):
            return solver.solve(x0, p=p, max_steps=120, atol=atol).x

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
        assert bool(jnp.all(jnp.isfinite(leaf)))
    np.testing.assert_allclose(np.asarray(x), expected_x, rtol=rtol_check, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(tangent), expected_dx, rtol=rtol_check, atol=1e-4
    )
    np.testing.assert_allclose(
        float(jnp.dot(tangent, jnp.ones(n, dtype=dtype))),
        float(jnp.dot(cotangent, b_dot)),
        rtol=rtol_check,
        atol=1e-4,
    )


@pytest.mark.parametrize("ad_solver", [None, LU(), SVD()], ids=["auto", "lu", "svd"])
def test_square_implicit_ad_lu_and_svd_on_gpu(dtype, ad_solver):
    rng = np.random.default_rng(23)
    n = 5
    A_np = rng.normal(size=(n, n)) + n * np.eye(n)
    b_np = rng.normal(size=n)
    b_dot_np = rng.normal(size=n)
    expected_dx = np.linalg.solve(A_np, b_dot_np)

    gpu = _gpu_devices()[0]
    atol = 1e-5 if dtype == jnp.float32 else 1e-10
    rtol_check = 3e-3 if dtype == jnp.float32 else 1e-7

    with jax.default_device(gpu):
        A = jnp.asarray(A_np, dtype=dtype)
        b = jnp.asarray(b_np, dtype=dtype)
        b_dot = jnp.asarray(b_dot_np, dtype=dtype)
        x0 = jnp.zeros(n, dtype=dtype)
        solver = LevenbergMarquardt(
            lambda x, args, p: A @ x - p,
            ad_solver=ad_solver,
        )

        def run(p):
            return solver.solve(x0, p=p, max_steps=100, atol=atol).x

        x, tangent = jax.jit(lambda p, dp: jax.jvp(run, (p,), (dp,)))(b, b_dot)
        cotangent = jax.jit(jax.grad(lambda p: jnp.sum(run(p))))(b)
        jax.block_until_ready((x, tangent, cotangent))

    for leaf in (x, tangent, cotangent):
        assert leaf.dtype == dtype
        assert next(iter(leaf.devices())).platform in GPU_PLATFORMS
    np.testing.assert_allclose(
        np.asarray(tangent), expected_dx, rtol=rtol_check, atol=1e-4
    )


@pytest.mark.parametrize("mode", ["cholesky", "qr", "cg_block_eigen"])
def test_ridge_lm_solve_continuation_and_implicit_ad_on_gpu(dtype, mode):
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
        solver = RidgeLevenbergMarquardt(
            lambda x, args, p: A @ x - p,
            metric=metric,
            ridge=ridge_floor * 10.0,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
        )

        def run(p):
            return solver.solve(
                x0,
                p=p,
                max_steps=150,
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
        assert bool(jnp.all(jnp.isfinite(leaf)))
    np.testing.assert_allclose(np.asarray(x), expected_x, rtol=rtol_check, atol=2e-4)
    np.testing.assert_allclose(
        np.asarray(tangent), expected_dx, rtol=rtol_check, atol=2e-4
    )


def test_preconditioners_and_gram_cg_on_gpu(dtype):
    rng = np.random.default_rng(41)
    m = 4
    dual_np = rng.normal(size=(m, m))
    dual_np = dual_np @ dual_np.T + 0.5 * np.eye(m)
    gpu = _gpu_devices()[0]

    with jax.default_device(gpu):
        dual = jnp.asarray(dual_np, dtype=dtype)
        u = jnp.asarray(rng.normal(size=m), dtype=dtype)
        U = jnp.asarray(rng.normal(size=(m, 2)), dtype=dtype)
        weights = jnp.asarray([0.4, 0.7], dtype=dtype)
        v = jnp.asarray(rng.normal(size=m), dtype=dtype)
        damping = jnp.asarray(0.1, dtype=dtype)
        ctx = SolverContext(
            x=None, lm_state=LMState(damping=damping), args=None, p=None
        )

        def dual_solve(vec):
            return jnp.linalg.solve(dual, vec)

        def dual_matvec(mat):
            return dual @ mat

        cases = [
            IdentityPreconditioner(),
            ShermanMorrisonPreconditioner(dual_solve, u, jnp.asarray(0.3, dtype=dtype)),
            WoodburyPreconditioner(dual_solve, U, weights),
            NystromPreconditioner(
                dual_matvec, n=m, rank=2, key=jax.random.key(0), dtype=dtype
            ),
            PaddedPreconditioner(IdentityPreconditioner(), n_real=m - 1),
            BlockEigenPreconditioner(
                [(dual[None], 1.0)], jnp.arange(m, dtype=jnp.int32)
            ),
        ]
        outputs = [
            jax.jit(lambda vec, p=precond: p.apply(vec, damping, ctx))(v)
            for precond in cases
        ]
        jax.block_until_ready(outputs)

    for out in outputs:
        assert out.dtype == dtype
        assert next(iter(out.devices())).platform in GPU_PLATFORMS
        assert bool(jnp.all(jnp.isfinite(out)))
        assert float(jnp.dot(v, out)) > 0.0


def test_state_space_metric_runs_on_gpu(dtype):
    gpu = _gpu_devices()[0]
    rtol = 1e-4 if dtype == jnp.float32 else 1e-10
    atol = 1e-5 if dtype == jnp.float32 else 1e-11
    with jax.default_device(gpu):
        n, repeats = 256, 3
        t = jnp.cumsum(jnp.ones(n, dtype=dtype))
        x = jnp.sin(jnp.linspace(0.0, 6.0, repeats * n, dtype=dtype))
        model = matern_state_space(
            jnp.asarray(1.3, dtype=dtype),
            jnp.asarray(0.8, dtype=dtype),
            2.5,
        )
        sequential = StateSpaceMetric(
            t, *model, repeats=repeats, epsilon=1e-5, parallel=False
        )
        parallel = StateSpaceMetric(
            t, *model, repeats=repeats, epsilon=1e-5, parallel=True
        )
        out_seq = jax.jit(lambda v: sequential.factor_solve(v, None))(x)
        out_par = jax.jit(lambda v: parallel.factor_solve(v, None))(x)
        apply_seq = jax.jit(lambda v: sequential.factor_apply(v, None))(x)
        apply_par = jax.jit(lambda v: parallel.factor_apply(v, None))(x)
        jax.block_until_ready((out_seq, out_par, apply_seq, apply_par))

    for arr in (out_seq, out_par, apply_seq, apply_par):
        assert arr.dtype == dtype
        assert next(iter(arr.devices())).platform in GPU_PLATFORMS
        assert bool(jnp.all(jnp.isfinite(arr)))
    assert jnp.allclose(out_seq, out_par, rtol=rtol, atol=atol)
    assert jnp.allclose(apply_seq, apply_par, rtol=rtol, atol=atol)
    assert jnp.allclose(
        sequential.norm(x, None), parallel.norm(x, None), rtol=rtol, atol=atol
    )


def multi_start_residual(theta, args, p):
    return jnp.array([theta[0] ** 2 - p - args, 0.1 * theta[1]], dtype=theta.dtype)


def multi_start_draw(key, x, args):
    return jax.random.uniform(key, x.shape, x.dtype, 0.5, 3.0)


def multi_start_reset_args(key, args):
    return jnp.zeros_like(args)


def test_parallel_and_sequential_multi_start_runs_on_gpu(dtype):
    gpu = _gpu_devices()[0]
    solver = LevenbergMarquardt(multi_start_residual, init_damping=1e-2)
    atol = 1e-5 if dtype == jnp.float32 else 1e-10

    with jax.default_device(gpu):
        x0 = jnp.array([jnp.nan, 0.0], dtype=dtype)
        args0 = jnp.asarray(1.0, dtype=dtype)
        p = jnp.asarray(4.0, dtype=dtype)
        for parallel in (True, False):
            ms = MultiStart(
                key=jax.random.key(0),
                num_starts=6,
                draw=multi_start_draw,
                reset_args=multi_start_reset_args,
                parallel=parallel,
            )
            result = solver.solve(
                x0, args0, p=p, max_steps=80, atol=atol, multi_start=ms
            )
            grad = jax.jit(
                jax.grad(
                    lambda q, ms=ms: jnp.sum(
                        solver.solve(
                            x0, args0, p=q, max_steps=80, atol=atol, multi_start=ms
                        ).x
                    )
                )
            )(p)
            jax.block_until_ready((result, grad))

            for leaf in jax.tree.leaves((result.x, result.info.loss, grad)):
                assert leaf.dtype == dtype
                assert next(iter(leaf.devices())).platform in GPU_PLATFORMS
            assert int(result.status) == LMStatus.CONVERGED
            assert int(result.multi_start.attempt) >= 1
            assert jnp.allclose(result.x[0] ** 2, p, atol=1e-4)
            assert jnp.allclose(grad, 0.5 / jnp.sqrt(p), atol=1e-3)


def test_vmapped_solve_and_failed_lane_ad_on_gpu(dtype):
    gpu = _gpu_devices()[0]
    solver = LevenbergMarquardt(
        lambda x, args, p: x - p,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )

    with jax.default_device(gpu):
        x0 = jnp.zeros(1, dtype=dtype)
        ps = jnp.asarray([0.0, 2.0], dtype=dtype)
        atols = jnp.asarray([1e-5, 0.0], dtype=dtype)

        def solved(values):
            return jax.vmap(
                lambda val, tol: solver.solve(
                    x0,
                    p=val,
                    max_steps=1,
                    max_steps_is_success=False,
                    atol=tol,
                ).x[0]
            )(values, atols)

        @jax.jit
        def evaluate(values):
            _, tangent = jax.jvp(solved, (values,), (jnp.ones_like(values),))
            _, vjp_fn = jax.vjp(solved, values)
            (cotangent,) = vjp_fn(jnp.ones_like(values))
            return tangent, cotangent

        tangent, cotangent = evaluate(ps)
        jax.block_until_ready((tangent, cotangent))

    expected = jnp.asarray([1.0, 0.0], dtype=dtype)
    for arr in (tangent, cotangent):
        assert arr.dtype == dtype
        assert next(iter(arr.devices())).platform in GPU_PLATFORMS
        assert jnp.array_equal(arr, expected)
