import dataclasses
from dataclasses import dataclass

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
    CholeskyMetric,
    DiagonalMetric,
    DrawNNXModule,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    MultiStart,
    Newton,
    NewtonAction,
    NewtonContext,
    NewtonSolveResult,
    NewtonStatus,
    Preconditioner,
    RidgeLevenbergMarquardt,
    RidgeNewton,
    register_pytree_dataclass,
    vmap_rows,
)


@dataclass(frozen=True, eq=False)
class _DiagonalPreconditioner(Preconditioner):
    diag: jax.Array

    def apply(self, v, damping, ctx):
        return self.diag * v


register_pytree_dataclass(
    _DiagonalPreconditioner, data_fields=("diag",), meta_fields=()
)


@dataclass(frozen=True, eq=False)
class _DampingDividingPreconditioner(Preconditioner):
    requires_positive_damping = True

    def apply(self, v, damping, ctx):
        return v / damping


register_pytree_dataclass(
    _DampingDividingPreconditioner, data_fields=(), meta_fields=()
)


class _UnregisteredPreconditioner(Preconditioner):
    def apply(self, v, damping, ctx):
        return v


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(),
        Cholesky(form="gram"),
        QR(),
        QR(form="gram"),
        SVD(),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_wide_linear_interpolation_one_step(linear_solver):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    m, n = 6, 14
    A = jax.random.normal(k1, (m, n), dtype=jnp.float32)
    b = jax.random.normal(k2, (m,), dtype=jnp.float32)
    x0 = jax.random.normal(k3, (n,), dtype=jnp.float32)

    def residual_fn(x):
        return A @ x - b

    solver = Newton(residual_fn, linear_solver=linear_solver)
    state0 = solver.init(x0)
    x1, _, info = solver.update(x0, state0)

    r0 = A @ x0 - b
    expected_d = -A.T @ jnp.linalg.solve(A @ A.T, r0)
    expected_x1 = x0 + expected_d

    np.testing.assert_allclose(x1, expected_x1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(A @ (x1 - x0) + r0, jnp.zeros_like(r0), atol=1e-5)
    assert bool(info.accepted)
    np.testing.assert_allclose(info.stepsize, 1.0, rtol=1e-6)
    assert int(info.num_linesearch_steps) == 1
    assert int(info.step_status) == int(NewtonStatus.RUNNING)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(),
        Cholesky(form="normal"),
        QR(),
        QR(form="normal"),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_tall_consistent_problem(linear_solver):
    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, 3)
    m, n = 12, 5
    A = jax.random.normal(k1, (m, n), dtype=jnp.float32)
    x_true = jax.random.normal(k2, (n,), dtype=jnp.float32)
    b = A @ x_true
    x0 = jax.random.normal(k3, (n,), dtype=jnp.float32)

    solver = Newton(lambda x: A @ x - b, linear_solver=linear_solver)
    result = solver.solve(x0, atol=1e-5, max_steps=5)
    assert int(result.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(result.x, x_true, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(),
        QR(),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_tall_inconsistent_problem_direction_and_slope(linear_solver):
    A = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=jnp.float32)
    b = jnp.array([1.0, 2.0, 5.0], dtype=jnp.float32)
    x0 = jnp.array([0.5, -0.5], dtype=jnp.float32)

    solver = Newton(lambda x: A @ x - b, linear_solver=linear_solver)
    state0 = solver.init(x0)
    x1, _, info = solver.update(x0, state0)

    r0 = A @ x0 - b
    grad = A.T @ r0
    expected_d = -jnp.linalg.solve(A.T @ A, grad)
    expected_slope = jnp.dot(grad, expected_d)

    np.testing.assert_allclose(x1 - x0, expected_d, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        info.directional_derivative, expected_slope, rtol=1e-5, atol=1e-5
    )
    assert float(info.loss) > 0.1
    result = solver.solve(x0, gtol=1e-5, max_steps=5)
    assert int(result.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(result.x, x0 + expected_d, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "linear_solver",
    [
        LU(),
        QR(),
        Cholesky(),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
        GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=64),
    ],
)
def test_square_nonlinear_problem(linear_solver):
    def residual_fn(x):
        return jnp.array(
            [
                x[0] ** 2 + x[1] - 3.0,
                x[0] + x[1] ** 2 - 5.0,
            ],
            dtype=jnp.float32,
        )

    x0 = jnp.array([1.2, 1.8], dtype=jnp.float32)
    solver = Newton(residual_fn, linear_solver=linear_solver)
    result = solver.solve(x0, atol=1e-5, max_steps=15)
    assert int(result.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(result.x, jnp.array([1.0, 2.0]), rtol=1e-5, atol=1e-5)


def test_rank_deficient_svd_padded_and_duplicate_rows():
    key = jax.random.key(2)
    k1, k2, k3 = jax.random.split(key, 3)
    A = jax.random.normal(k1, (4, 8), dtype=jnp.float32)
    b = jax.random.normal(k2, (4,), dtype=jnp.float32)
    x0 = jax.random.normal(k3, (8,), dtype=jnp.float32)

    base_solver = Newton(lambda x: A @ x - b, linear_solver=SVD())
    x_base, _, _ = base_solver.update(x0, base_solver.init(x0))

    def padded_residual(x):
        r = A @ x - b
        return jnp.concatenate([r, jnp.zeros(3, dtype=r.dtype)])

    padded_solver = Newton(padded_residual, linear_solver=SVD())
    x_padded, _, info_padded = padded_solver.update(x0, padded_solver.init(x0))
    np.testing.assert_allclose(x_padded, x_base, rtol=1e-5, atol=1e-5)
    assert int(info_padded.step_status) == int(NewtonStatus.RUNNING)

    def dup_residual(x):
        r = A @ x - b
        return jnp.concatenate([r, r[:2]])

    dup_solver = Newton(dup_residual, linear_solver=SVD())
    res_dup = dup_solver.solve(x0, atol=1e-5, max_steps=5)
    assert int(res_dup.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(A @ res_dup.x, b, rtol=1e-5, atol=1e-5)


def test_all_zero_jacobian_and_exact_root():
    def const_nonzero_residual(x):
        return jnp.ones_like(x)

    x0 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    solver_svd = Newton(const_nonzero_residual, linear_solver=SVD())
    res_no_gtol = solver_svd.solve(x0, gtol=0.0, max_steps=5)
    assert int(res_no_gtol.status) == int(NewtonStatus.NON_DESCENT)
    assert int(res_no_gtol.steps) == 1
    np.testing.assert_allclose(res_no_gtol.x, x0)

    res_with_gtol = solver_svd.solve(x0, gtol=1e-5, max_steps=5)
    assert int(res_with_gtol.status) == int(NewtonStatus.CONVERGED)
    assert int(res_with_gtol.steps) == 1

    # Exact root at x0 bypasses singular Cholesky factorization even with all
    # tolerances set to 0.
    def singular_at_root(x):
        return x**2

    root_x0 = jnp.zeros(3, dtype=jnp.float32)
    solver_chol = Newton(singular_at_root, linear_solver=Cholesky())
    res_root = solver_chol.solve(root_x0, atol=0.0, gtol=0.0, xtol=0.0, max_steps=5)
    assert int(res_root.status) == int(NewtonStatus.CONVERGED)
    assert int(res_root.steps) == 0

    state0 = solver_chol.init(root_x0)
    x_u, _, info_u = solver_chol.update(root_x0, state0)
    np.testing.assert_allclose(x_u, root_x0)
    assert int(info_u.step_status) == int(NewtonStatus.CONVERGED)
    assert not bool(info_u.accepted)
    assert int(info_u.num_linesearch_steps) == 0


def test_shape_and_role_validation_errors():
    def wide_res(x):
        return x[:2] + x[2:4]

    def tall_res(x):
        return jnp.concatenate([x, x + 1.0])

    x_wide = jnp.ones(4, dtype=jnp.float32)
    x_tall = jnp.ones(2, dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires n <= m"):
        Newton(wide_res, linear_solver=Cholesky(form="normal")).init(x_wide)
    with pytest.raises(ValueError, match="requires n <= m"):
        Newton(wide_res, linear_solver=QR(form="normal")).init(x_wide)
    with pytest.raises(ValueError, match="requires m <= n"):
        Newton(tall_res, linear_solver=Cholesky(form="gram")).init(x_tall)
    with pytest.raises(ValueError, match="requires m <= n"):
        Newton(tall_res, linear_solver=QR(form="gram")).init(x_tall)
    with pytest.raises(ValueError, match="requires a square system"):
        Newton(wide_res, linear_solver=LU()).init(x_wide)
    with pytest.raises(ValueError, match="requires n <= m"):
        Newton(wide_res, linear_solver=CG(IdentityPreconditioner())).init(x_wide)
    with pytest.raises(ValueError, match="requires m <= n"):
        Newton(tall_res, linear_solver=GramCG(IdentityPreconditioner())).init(x_tall)

    with pytest.raises(ValueError, match="penalty"):
        Newton(
            tall_res,
            linear_solver=CG(IdentityPreconditioner(), penalty=1e-3),
        )
    with pytest.raises(ValueError, match="requires a preconditioner"):
        Newton(tall_res, linear_solver=CG(None))
    with pytest.raises(ValueError, match="divides by the live damping"):
        Newton(
            tall_res,
            linear_solver=CG(_DampingDividingPreconditioner()),
        )
    with pytest.raises(TypeError, match="registered pytree"):
        Newton(
            tall_res,
            linear_solver=CG(_UnregisteredPreconditioner()),
        )

    # LM still rejects LU and SVD in forward role
    with pytest.raises(ValueError, match="cannot serve as linear_solver"):
        LevenbergMarquardt(wide_res, linear_solver=LU())
    with pytest.raises(ValueError, match="cannot serve as linear_solver"):
        LevenbergMarquardt(wide_res, linear_solver=SVD())

    # Constructor hyperparameter validation
    with pytest.raises(ValueError, match="c1"):
        Newton(wide_res, c1=0.0)
    with pytest.raises(ValueError, match="c1"):
        Newton(wide_res, c1=1.0)
    with pytest.raises(ValueError, match="decrease_factor"):
        Newton(wide_res, decrease_factor=1.0)
    with pytest.raises(ValueError, match="maxls"):
        Newton(wide_res, maxls=0)
    with pytest.raises(ValueError, match="maxls"):
        Newton(wide_res, maxls=True)
    with pytest.raises(ValueError, match="max_stepsize"):
        Newton(wide_res, max_stepsize=0.0)
    with pytest.raises(ValueError, match="linesearch_tol"):
        Newton(wide_res, linesearch_tol=-1e-5)
    with pytest.raises(ValueError, match="reg must be nonnegative"):
        Newton(wide_res, reg=-0.1)
    with pytest.raises(ValueError, match="reg_power must be nonnegative"):
        Newton(wide_res, reg_power=-1.0)
    with pytest.raises(ValueError, match="min_damping must be positive"):
        Newton(wide_res, min_damping=0.0)
    with pytest.raises(ValueError, match="does not support reg > 0"):
        Newton(lambda x: x, reg=0.1, linear_solver=LU())

    # Rejects LMState in Newton.solve / Newton.update
    lm_state = LevenbergMarquardt(wide_res).init(x_wide)
    newton = Newton(wide_res)
    with pytest.raises(TypeError, match="NewtonState"):
        newton.update(x_wide, lm_state)
    with pytest.raises(TypeError, match="NewtonState"):
        newton.solve(x_wide, newton_state=lm_state)
    with pytest.raises(TypeError):
        newton.solve(x_wide, lm_state=lm_state)


def test_float32_conditioning_qr_and_svd_beat_gram_cholesky():
    rng = np.random.default_rng(42)
    m, n = 6, 12
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, m)))
    sigmas = np.geomspace(1.0, 1e-4, m)
    A64 = (U * sigmas) @ V.T
    x_true64 = V @ rng.standard_normal(m)
    b64 = A64 @ x_true64
    x0_64 = np.zeros(n, dtype=np.float64)
    oracle_step = np.linalg.lstsq(A64, b64, rcond=None)[0]

    A32 = jnp.asarray(A64, dtype=jnp.float32)
    b32 = jnp.asarray(b64, dtype=jnp.float32)
    x0_32 = jnp.asarray(x0_64, dtype=jnp.float32)

    def res_fn(x):
        return A32 @ x - b32

    s_qr = Newton(res_fn, linear_solver=QR(form="gram"))
    x_qr, _, info_qr = s_qr.update(x0_32, s_qr.init(x0_32))

    s_svd = Newton(res_fn, linear_solver=SVD())
    x_svd, _, info_svd = s_svd.update(x0_32, s_svd.init(x0_32))

    s_chol = Newton(res_fn, linear_solver=Cholesky(form="gram"))
    x_chol, _, _ = s_chol.update(x0_32, s_chol.init(x0_32))

    err_qr = np.linalg.norm(
        np.asarray(x_qr, dtype=np.float64) - oracle_step
    ) / np.linalg.norm(oracle_step)
    err_svd = np.linalg.norm(
        np.asarray(x_svd, dtype=np.float64) - oracle_step
    ) / np.linalg.norm(oracle_step)
    err_chol = np.linalg.norm(
        np.asarray(x_chol, dtype=np.float64) - oracle_step
    ) / np.linalg.norm(oracle_step)

    assert int(info_qr.step_status) == int(NewtonStatus.RUNNING)
    assert int(info_svd.step_status) == int(NewtonStatus.RUNNING)
    assert err_qr < 5e-3
    assert err_svd < 5e-3
    assert err_qr < err_chol


def test_krylov_preconditioner_refresh_and_block_eigen():
    key = jax.random.key(7)
    k1, k2, k3 = jax.random.split(key, 3)
    m, n = 5, 10
    A = jax.random.normal(k1, (m, n), dtype=jnp.float32)
    b = jax.random.normal(k2, (m,), dtype=jnp.float32)
    x0 = jax.random.normal(k3, (n,), dtype=jnp.float32)

    def residual_fn(x):
        return jnp.sinh(A @ x - b)

    precond0 = _DiagonalPreconditioner(jnp.ones(m, dtype=jnp.float32))
    solver = Newton(
        residual_fn,
        linear_solver=GramCG(precond0, tol=1e-6, maxiter=32),
    )

    def refresh_cb(ctx: NewtonContext):
        new_diag = jnp.full((m,), 2.0, dtype=jnp.float32)
        new_state = dataclasses.replace(
            ctx.state,
            preconditioner=_DiagonalPreconditioner(new_diag),
        )
        return NewtonAction(newton_state=new_state)

    result = solver.solve(x0, atol=1e-5, max_steps=10, callback=refresh_cb)
    assert int(result.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(
        result.newton_state.preconditioner.diag,
        jnp.full((m,), 2.0, dtype=jnp.float32),
    )

    # BlockEigenPreconditioner works with NewtonState (where state.ridge is None)
    gram = A @ A.T
    block_precond = BlockEigenPreconditioner(
        families=[(gram[None, :, :], 0)],
        permutation=jnp.arange(m, dtype=jnp.int32),
    )
    solver_be = Newton(
        residual_fn,
        linear_solver=GramCG(block_precond, tol=1e-6, maxiter=16),
    )
    res_be = solver_be.solve(x0, atol=1e-5, max_steps=10)
    assert int(res_be.status) == int(NewtonStatus.CONVERGED)


def test_manual_loop_and_jit_vs_python_parity():
    def residual_fn(x, scale):
        return jnp.array(
            [
                scale * jnp.sin(x[0]) + x[1] - 1.0,
                x[0] + scale * jnp.cos(x[1]) - 0.5,
            ],
            dtype=jnp.float32,
        )

    x0 = jnp.array([0.2, 0.3], dtype=jnp.float32)
    args = jnp.asarray(1.5, dtype=jnp.float32)
    solver = Newton(residual_fn)

    res_jit = solver.solve(x0, args, atol=1e-6, max_steps=10, jit=True)
    res_py = solver.solve(x0, args, atol=1e-6, max_steps=10, jit=False)

    assert isinstance(res_jit, NewtonSolveResult)
    assert int(res_jit.status) == int(NewtonStatus.CONVERGED)
    assert int(res_jit.steps) == int(res_py.steps)
    np.testing.assert_allclose(res_jit.x, res_py.x, rtol=1e-6, atol=1e-6)

    state = solver.init(x0, args)
    x = x0
    for _ in range(int(res_jit.steps)):
        x, state, _ = solver.update(x, state, args)
    np.testing.assert_allclose(x, res_jit.x, rtol=1e-6, atol=1e-6)


def test_callbacks_histories_and_aux():
    def residual_fn(x, offset):
        r = jnp.array([x[0] ** 2 - offset, x[1] - 2.0 * x[0]], dtype=jnp.float32)
        return r, {"norm_x": jnp.linalg.norm(x)}

    x0 = jnp.array([3.0, 0.5], dtype=jnp.float32)
    args0 = jnp.asarray(4.0, dtype=jnp.float32)
    solver = Newton(residual_fn, has_aux=True)

    def cb(ctx: NewtonContext):
        assert isinstance(ctx, NewtonContext)
        assert ctx.state is ctx.newton_state
        new_hyper = dataclasses.replace(
            ctx.newton_state.hyper,
            c1=jnp.asarray(1e-3, dtype=jnp.float32),
        )
        new_state = dataclasses.replace(ctx.newton_state, hyper=new_hyper)
        return NewtonAction(
            newton_state=new_state,
            user_state=ctx.user_state + ctx.info.num_linesearch_steps,
        )

    result = solver.solve(
        x0,
        args0,
        atol=1e-5,
        max_steps=10,
        callback=cb,
        user_state=jnp.asarray(0, dtype=jnp.int32),
        save_steps=True,
    )
    assert int(result.status) == int(NewtonStatus.CONVERGED)
    assert int(result.user_state) >= int(result.steps)
    steps = int(result.steps)
    np.testing.assert_allclose(result.x_history[0], x0)
    np.testing.assert_allclose(result.x_history[steps], result.x)
    np.testing.assert_allclose(
        result.aux["norm_x"], jnp.linalg.norm(result.x), rtol=1e-6
    )
    np.testing.assert_allclose(
        result.aux_history["norm_x"][0], jnp.linalg.norm(x0), rtol=1e-6
    )
    np.testing.assert_allclose(
        result.aux_history["norm_x"][steps], jnp.linalg.norm(result.x), rtol=1e-6
    )


def test_cache_jacobian_manual_retry_and_invalidation():
    def residual_fn(x, target):
        return jnp.exp(x) - target

    x0 = jnp.array([-10.0], dtype=jnp.float32)
    target = jnp.array([1.0], dtype=jnp.float32)
    solver = Newton(residual_fn, maxls=5, cache_jacobian=True)
    state0 = solver.init(x0, target)

    # First attempt with maxls=5 fails line search, leaving jacobian_valid=True
    x1, state1, info1 = solver.update(x0, state0, target)
    assert int(info1.step_status) == int(NewtonStatus.LINE_SEARCH_FAILED)
    assert bool(state1.jacobian_valid)
    assert bool(state1.solver_cache.valid)
    np.testing.assert_allclose(x1, x0)

    # Corrupt state1.Jt to verify that the retry genuinely reads from the cached Jt
    scaled_state = dataclasses.replace(
        state1,
        Jt=state1.Jt * 2.0,
        solver_cache=dataclasses.replace(
            state1.solver_cache, G=state1.solver_cache.G * 4.0
        ),
        hyper=dataclasses.replace(
            solver.hyperparams(jnp.float32), maxls=jnp.asarray(60, dtype=jnp.int32)
        ),
    )
    x2, state2, info2 = solver.update(x1, scaled_state, target)
    assert bool(info2.accepted)
    assert not bool(state2.jacobian_valid)
    assert not bool(state2.solver_cache.valid)

    # Fresh uncorrupted retry with maxls=60 accepts a step
    retry_state = dataclasses.replace(
        state1,
        hyper=dataclasses.replace(
            solver.hyperparams(jnp.float32), maxls=jnp.asarray(60, dtype=jnp.int32)
        ),
    )
    x3, _, info3 = solver.update(x1, retry_state, target)
    assert bool(info3.accepted)
    assert not np.allclose(x2, x3)


def test_multi_start_sequential_and_parallel():
    def residual_fn(x):
        return jnp.exp(x) - 1.0

    # Lane 0 starts at -10.0 and fails with maxls=30; drawn lane starts near 0.5
    # and converges.
    x0 = jnp.array([-10.0], dtype=jnp.float32)
    solver = Newton(residual_fn, maxls=30)

    def draw(key, x_prev, args_prev):
        return jax.random.uniform(
            key, shape=x_prev.shape, dtype=x_prev.dtype, minval=0.1, maxval=1.0
        )

    for parallel in (False, True):
        for jit in (True, False):
            ms = MultiStart(
                key=jax.random.key(99),
                num_starts=3,
                draw=draw,
                parallel=parallel,
            )
            res = solver.solve(x0, atol=1e-5, max_steps=10, multi_start=ms, jit=jit)
            assert int(res.status) == int(NewtonStatus.CONVERGED)
            assert bool(res.multi_start.accepted)
            assert int(res.multi_start.attempt) >= 1
            np.testing.assert_allclose(res.x, jnp.zeros_like(x0), atol=1e-5)


def test_terminal_search_failure_not_masked_by_max_steps_is_success():
    def residual_fn(x):
        return jnp.exp(x) - 1.0

    x0 = jnp.array([-10.0], dtype=jnp.float32)
    solver = Newton(residual_fn, maxls=5)
    res = solver.solve(x0, max_steps=1, max_steps_is_success=True)
    assert int(res.status) == int(NewtonStatus.LINE_SEARCH_FAILED)
    np.testing.assert_allclose(res.x, x0)


def test_nnx_module_and_vmap_rows_integration():
    pytest.importorskip("flax")
    from flax import nnx

    class TinyLinear(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.w = nnx.Param(
                jax.random.normal(rngs.params(), (4, 1), dtype=jnp.float32)
            )
            self.scale = nnx.Variable(jnp.asarray(2.0, dtype=jnp.float32))

        def __call__(self, x):
            return self.scale[...] * (x @ self.w[...]).squeeze(-1)

    model = TinyLinear(rngs=nnx.Rngs(0))
    graphdef, theta0, nondiff = nnx.split(model, nnx.Param, ...)
    X = jnp.eye(4, dtype=jnp.float32)
    y = jnp.array([2.0, -4.0, 6.0, -8.0], dtype=jnp.float32)

    def row_res(th, x_i, y_i):
        m = nnx.merge(graphdef, th, nondiff)
        return m(x_i[None, :])[0] - y_i

    batched_res = vmap_rows(row_res, in_axes=(None, 0, 0))
    solver = Newton(lambda th: batched_res(th, X, y))
    draw = DrawNNXModule(TinyLinear)
    ms = MultiStart(key=jax.random.key(3), num_starts=2, draw=draw)
    res = solver.solve(theta0, atol=1e-5, max_steps=5, multi_start=ms)
    assert int(res.status) == int(NewtonStatus.CONVERGED)
    merged = nnx.merge(graphdef, res.x, nondiff)
    np.testing.assert_allclose(merged(X), y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(form="gram"),
        QR(form="gram"),
        SVD(),
        GramCG(IdentityPreconditioner(), tol=1e-8, maxiter=32),
    ],
)
def test_residual_coupled_regularization_wide_rank_deficient(linear_solver):
    key = jax.random.key(123)
    k1, k2 = jax.random.split(key)
    # Rank-deficient wide Jacobian (duplicate rows) where undamped Cholesky/QR fail
    A_half = 0.3 * jax.random.normal(k1, (3, 10), dtype=jnp.float32)
    A = jnp.concatenate([A_half, A_half], axis=0)
    x_star = jax.random.normal(k2, (10,), dtype=jnp.float32)
    b = jnp.sinh(A @ x_star)

    def residual_fn(x):
        return jnp.sinh(A @ x) - b

    x0 = jnp.zeros(10, dtype=jnp.float32)
    if isinstance(linear_solver, (Cholesky, QR)):
        undamped = Newton(residual_fn, linear_solver=linear_solver)
        res_undamped = undamped.solve(x0, atol=1e-5, max_steps=15)
        assert int(res_undamped.status) == int(NewtonStatus.LINEAR_SOLVE_FAILED)

    damped = Newton(
        residual_fn,
        reg=0.1,
        reg_power=1.5,
        min_damping=1e-10,
        linear_solver=linear_solver,
    )
    res_damped = damped.solve(x0, atol=1e-5, max_steps=25)
    assert int(res_damped.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(residual_fn(res_damped.x), 0.0, atol=1e-5)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(form="normal"),
        QR(form="normal"),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-8, maxiter=32),
    ],
)
def test_residual_coupled_regularization_tall_rank_deficient(linear_solver):
    key = jax.random.key(456)
    k1, k2 = jax.random.split(key)
    # Rank-deficient tall Jacobian (duplicate columns) where undamped Cholesky/QR fail
    A_half = 0.3 * jax.random.normal(k1, (8, 2), dtype=jnp.float32)
    A = jnp.concatenate([A_half, A_half], axis=1)
    x_star = jax.random.normal(k2, (4,), dtype=jnp.float32)
    b = jnp.sinh(A @ x_star)

    def residual_fn(x):
        return jnp.sinh(A @ x) - b

    x0 = jnp.zeros(4, dtype=jnp.float32)
    if isinstance(linear_solver, (Cholesky, QR)):
        undamped = Newton(residual_fn, linear_solver=linear_solver)
        res_undamped = undamped.solve(x0, atol=1e-5, max_steps=15)
        assert int(res_undamped.status) == int(NewtonStatus.LINEAR_SOLVE_FAILED)

    damped = Newton(
        residual_fn,
        reg=0.1,
        reg_power=1.5,
        min_damping=1e-10,
        linear_solver=linear_solver,
    )
    res_damped = damped.solve(x0, atol=1e-5, max_steps=25)
    assert int(res_damped.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(residual_fn(res_damped.x), 0.0, atol=1e-5)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(),
        QR(),
        SVD(),
        GramCG(IdentityPreconditioner(), tol=1e-10, maxiter=32),
    ],
)
def test_newton_metric_minimum_w_norm_selection(linear_solver):
    key = jax.random.key(777)
    k1, k2 = jax.random.split(key)
    m, n = 4, 10
    A = jax.random.normal(k1, (m, n), dtype=jnp.float32)
    b = jax.random.normal(k2, (m,), dtype=jnp.float32)
    weights = jnp.linspace(1.0, 5.0, 6, dtype=jnp.float32)
    metric = DiagonalMetric(weights, free_scale=2.0)
    # Full diagonal factor F_bar = diag(sqrt(weights), sqrt(2.0) * I_4)
    f_bar_diag = jnp.concatenate(
        [jnp.sqrt(weights), jnp.full((n - 6,), jnp.sqrt(2.0), dtype=jnp.float32)]
    )
    B = A / f_bar_diag[None, :]
    u_oracle = jnp.linalg.pinv(B) @ b
    x_oracle = u_oracle / f_bar_diag

    solver = Newton(lambda x: A @ x - b, metric=metric, linear_solver=linear_solver)
    x0 = jnp.zeros(n, dtype=jnp.float32)
    res = solver.solve(x0, atol=1e-5, max_steps=5)
    assert int(res.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(res.x, x_oracle, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(),
        QR(),
        CG(IdentityPreconditioner(), tol=1e-10, maxiter=64),
    ],
)
def test_ridge_newton_matches_ridge_lm_and_supports_anneal_ridge(linear_solver):
    key = jax.random.key(888)
    k1, k2, k3 = jax.random.split(key, 3)
    m, n_m, n_f = 5, 8, 2
    n = n_m + n_f
    A = jax.random.normal(k1, (m, n), dtype=jnp.float32)
    b = jax.random.normal(k2, (m,), dtype=jnp.float32)
    M_raw = jax.random.normal(k3, (n_m, n_m), dtype=jnp.float32)
    K = M_raw @ M_raw.T + jnp.eye(n_m, dtype=jnp.float32)
    U = jnp.linalg.cholesky(K, upper=True)
    metric = CholeskyMetric(U)
    ridge = 1e-1

    def residual_fn(x, _, p):
        return jnp.sinh(A @ x) - p * b

    p0 = jnp.asarray(0.5, dtype=jnp.float32)
    x0 = jnp.zeros(n, dtype=jnp.float32)

    rn = RidgeNewton(
        residual_fn, metric=metric, ridge=ridge, linear_solver=linear_solver
    )
    rlm = RidgeLevenbergMarquardt(
        residual_fn, metric=metric, ridge=ridge, linear_solver=linear_solver
    )
    res_rn = rn.solve(x0, None, p=p0, gtol=1e-5, atol=1.0, max_steps=30)
    res_rlm = rlm.solve(x0, None, p=p0, gtol=1e-5, atol=1.0, max_steps=50)
    assert int(res_rn.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(res_rn.x, res_rlm.x, rtol=1e-4, atol=1e-4)

    # Implicit AD through RidgeNewton matches RidgeLevenbergMarquardt
    grad_rn = jax.grad(
        lambda p: jnp.sum(rn.solve(x0, None, p=p, gtol=1e-5, max_steps=30).x ** 2)
    )(p0)
    grad_rlm = jax.grad(
        lambda p: jnp.sum(rlm.solve(x0, None, p=p, gtol=1e-5, max_steps=50).x ** 2)
    )(p0)
    np.testing.assert_allclose(grad_rn, grad_rlm, rtol=1e-3, atol=1e-4)

    # AnnealRidge works with RidgeNewton
    anneal = AnnealRidge(ridge_floor=1e-4, decrease=0.1, grad_rtol=1e-1)
    rn_anneal = RidgeNewton(
        residual_fn, metric=metric, ridge=1e-2, linear_solver=linear_solver
    )
    res_anneal = rn_anneal.solve(
        x0,
        None,
        p=p0,
        callback=anneal,
        user_state=anneal.init_state(jnp.float32),
        gtol=1e-5,
        max_steps=40,
    )
    assert int(res_anneal.status) == int(NewtonStatus.CONVERGED)
    np.testing.assert_allclose(res_anneal.newton_state.ridge, 1e-4, rtol=1e-5)


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(),
        QR(),
        LU(),
        SVD(),
        CG(IdentityPreconditioner(), tol=1e-10, maxiter=32),
        GramCG(IdentityPreconditioner(), tol=1e-10, maxiter=32),
    ],
)
def test_newton_and_ridge_newton_geodesic_acceleration(linear_solver):
    key = jax.random.key(999)
    A = jax.random.normal(key, (4, 4), dtype=jnp.float32) + 2.0 * jnp.eye(
        4, dtype=jnp.float32
    )
    x_star = jnp.array([0.4, -0.3, 0.2, -0.5], dtype=jnp.float32)
    b = jnp.sinh(A @ x_star)

    def residual_fn(x):
        return jnp.sinh(A @ x) - b

    x0 = jnp.zeros(4, dtype=jnp.float32)
    solver = Newton(
        residual_fn,
        linear_solver=linear_solver,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    )
    res = solver.solve(x0, atol=1e-5, max_steps=20)
    assert int(res.status) == int(NewtonStatus.CONVERGED)
    assert bool(res.info.used_geodesic)
    np.testing.assert_allclose(res.x, x_star, rtol=1e-4, atol=1e-4)

    if linear_solver.supports_penalty:
        metric = DiagonalMetric(jnp.ones(4, dtype=jnp.float32))
        rn = RidgeNewton(
            residual_fn,
            metric=metric,
            ridge=1e-1,
            linear_solver=linear_solver,
            geodesic_acceleration=True,
        )
        res_rn = rn.solve(x0, gtol=1e-5, max_steps=20)
        assert int(res_rn.status) == int(NewtonStatus.CONVERGED)
        assert bool(res_rn.info.used_geodesic)


@pytest.mark.parametrize("shape_mode", ["wide", "tall"])
@pytest.mark.parametrize("reg", [0.0, 0.05])
def test_geodesic_correction_exact_reference_and_ratio_rejection(shape_mode, reg):
    key = jax.random.key(2026)
    k1, k2 = jax.random.split(key)
    m, n = (4, 8) if shape_mode == "wide" else (8, 4)
    A = 0.4 * jax.random.normal(k1, (m, n), dtype=jnp.float32)
    b = 0.3 * jax.random.normal(k2, (m,), dtype=jnp.float32)
    x0 = jnp.full((n,), 0.2, dtype=jnp.float32)

    def residual_fn(x):
        return jnp.exp(A @ x) - b

    r0 = residual_fn(x0)
    J0 = jax.jacfwd(residual_fn)(x0)
    if reg > 0.0:
        j_scale = jnp.sum(J0**2) / m
        mu = reg * j_scale * (jnp.linalg.norm(r0) ** 1.5)
        d_ref = -jnp.linalg.solve(
            J0.T @ J0 + mu * jnp.eye(n, dtype=jnp.float32), J0.T @ r0
        )
        f_vv = jax.jvp(
            lambda x: jax.jvp(residual_fn, (x,), (d_ref,))[1], (x0,), (d_ref,)
        )[1]
        a_ref = -jnp.linalg.solve(
            J0.T @ J0 + mu * jnp.eye(n, dtype=jnp.float32), J0.T @ f_vv
        )
    else:
        d_ref = -jnp.linalg.pinv(J0) @ r0
        f_vv = jax.jvp(
            lambda x: jax.jvp(residual_fn, (x,), (d_ref,))[1], (x0,), (d_ref,)
        )[1]
        a_ref = -jnp.linalg.pinv(J0) @ f_vv

    solvers = [Cholesky(), QR(), SVD()]
    if shape_mode == "wide":
        solvers.append(GramCG(IdentityPreconditioner(), tol=1e-10, maxiter=32))
    else:
        solvers.append(CG(IdentityPreconditioner(), tol=1e-10, maxiter=32))

    for ls in solvers:
        if reg > 0.0 and isinstance(ls, (CG, GramCG)):
            mu_ls = reg * (jnp.linalg.norm(r0) ** 1.5)
            d_ls = -jnp.linalg.solve(
                J0.T @ J0 + mu_ls * jnp.eye(n, dtype=jnp.float32), J0.T @ r0
            )
            f_vv_ls = jax.jvp(
                lambda x, d_in=d_ls: jax.jvp(residual_fn, (x,), (d_in,))[1],
                (x0,),
                (d_ls,),
            )[1]
            a_ls = -jnp.linalg.solve(
                J0.T @ J0 + mu_ls * jnp.eye(n, dtype=jnp.float32), J0.T @ f_vv_ls
            )
        else:
            d_ls, a_ls = d_ref, a_ref
        solver = Newton(
            residual_fn,
            reg=reg,
            linear_solver=ls,
            geodesic_acceleration=True,
            geodesic_acceptance_ratio=10.0,
        )
        x1, _, info1 = solver.update(x0, solver.init(x0))
        assert bool(info1.used_geodesic)
        np.testing.assert_allclose(x1, x0 + d_ls + 0.5 * a_ls, rtol=1e-4, atol=1e-4)

    # Rejects geodesic when ratio exceeds geodesic_acceptance_ratio
    strict_solver = Newton(
        residual_fn,
        reg=reg,
        linear_solver=QR(),
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1e-6,
    )
    x1_strict, _, info_strict = strict_solver.update(x0, strict_solver.init(x0))
    assert not bool(info_strict.used_geodesic)
    assert float(info_strict.acceleration_ratio) > 1e-6
    np.testing.assert_allclose(x1_strict, x0 + d_ref, rtol=1e-4, atol=1e-4)


def test_callback_rescues_failed_linesearch_and_anneals_stationary_ridge():
    def residual_fn(x):
        return jnp.exp(x) - 1.0

    x0 = jnp.array([-10.0], dtype=jnp.float32)
    solver = Newton(residual_fn, maxls=3, cache_jacobian=True)

    def rescue_cb(ctx: NewtonContext):
        rescue = ctx.info.step_status == NewtonStatus.LINE_SEARCH_FAILED
        new_maxls = jnp.where(rescue, jnp.int32(60), ctx.newton_state.hyper.maxls)
        new_hyper = dataclasses.replace(ctx.newton_state.hyper, maxls=new_maxls)
        return NewtonAction(
            newton_state=dataclasses.replace(ctx.newton_state, hyper=new_hyper)
        )

    for jit in (True, False):
        res = solver.solve(x0, atol=1e-5, max_steps=15, callback=rescue_cb, jit=jit)
        assert int(res.status) == int(NewtonStatus.CONVERGED)
        np.testing.assert_allclose(res.x, jnp.zeros_like(x0), atol=1e-5)

    # Linear ridge problem reaches exact stationarity at the first ridge in 1 step,
    # so step 2 before annealing has NON_DESCENT; AnnealRidge must rescue it.
    A = jnp.array([[1.0, 2.0, -1.0], [0.5, -1.0, 1.5]], dtype=jnp.float32)
    b = jnp.array([1.0, -0.5], dtype=jnp.float32)
    metric = DiagonalMetric(jnp.ones(3, dtype=jnp.float32))
    rn = RidgeNewton(lambda x: A @ x - b, metric=metric, ridge=1.0)
    anneal = AnnealRidge(ridge_floor=1e-3, decrease=0.1, grad_rtol=1e-2)
    for jit in (True, False):
        res_rn = rn.solve(
            jnp.zeros(3, dtype=jnp.float32),
            callback=anneal,
            user_state=anneal.init_state(jnp.float32),
            gtol=1e-5,
            max_steps=15,
            jit=jit,
        )
        assert int(res_rn.status) == int(NewtonStatus.CONVERGED)
        np.testing.assert_allclose(res_rn.newton_state.ridge, 1e-3, rtol=1e-5)


def test_mixed_precision_metric_and_tiny_reg_cholesky_floor():
    with jax.enable_x64(True):
        A = jnp.array([[1.0, 0.5, -0.2], [0.2, -1.0, 0.8]], dtype=jnp.float32)
        b = jnp.array([0.4, -0.6], dtype=jnp.float32)
        x0 = jnp.zeros(3, dtype=jnp.float32)
        metric64 = DiagonalMetric(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64))

        s_n = Newton(lambda x: A @ x - b, metric=metric64, reg=1e-3)
        res_n = s_n.solve(x0, atol=1e-5, max_steps=10)
        assert res_n.x.dtype == jnp.float32
        assert int(res_n.status) == int(NewtonStatus.CONVERGED)

        s_rn = RidgeNewton(lambda x: A @ x - b, metric=metric64, ridge=1e-1, reg=1e-3)
        res_rn = s_rn.solve(x0, gtol=1e-5, max_steps=10)
        assert res_rn.x.dtype == jnp.float32
        assert int(res_rn.status) == int(NewtonStatus.CONVERGED)

    # Tiny positive reg below float32 eps on a rank-deficient Jacobian still
    # regularizes Cholesky, QR, and SVD cleanly.
    A_Sing = jnp.array([[1.0, 2.0], [1.0, 2.0]], dtype=jnp.float32)
    b_sing = jnp.array([3.0, 3.0], dtype=jnp.float32)
    for ls in (Cholesky(), QR(), SVD()):
        s_tiny = Newton(
            lambda x: A_Sing @ x - b_sing,
            reg=1e-12,
            min_damping=1e-12,
            linear_solver=ls,
        )
        res_tiny = s_tiny.solve(
            jnp.zeros(2, dtype=jnp.float32), atol=1e-4, max_steps=10
        )
        assert int(res_tiny.status) == int(NewtonStatus.CONVERGED)
        np.testing.assert_allclose(A_Sing @ res_tiny.x, b_sing, atol=1e-3)

    # Undamped float64 duplicate-row rank-deficient Jacobian is rejected by
    # Cholesky and QR via the relative pivot cutoff, while SVD and GramCG
    # match the minimum-norm pseudoinverse step.
    with jax.enable_x64(True):
        A_top = jnp.array(
            [[1.0, 2.0, -1.0, 0.5], [0.5, -1.0, 1.5, -0.5]], dtype=jnp.float64
        )
        A64 = jnp.concatenate([A_top, A_top], axis=0)
        b64 = jnp.array([1.0, -0.5, 1.0, -0.5], dtype=jnp.float64)
        x0_64 = jnp.zeros(4, dtype=jnp.float64)
        expected_pinv = jnp.linalg.pinv(A64) @ b64

        for bad_ls in (Cholesky(), QR()):
            res_bad = Newton(lambda x: A64 @ x - b64, linear_solver=bad_ls).solve(
                x0_64, atol=1e-10, max_steps=5
            )
            assert int(res_bad.status) == int(NewtonStatus.LINEAR_SOLVE_FAILED)

        for good_ls in (SVD(), GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=16)):
            res_good = Newton(lambda x: A64 @ x - b64, linear_solver=good_ls).solve(
                x0_64, atol=1e-10, max_steps=5
            )
            assert int(res_good.status) == int(NewtonStatus.CONVERGED)
            np.testing.assert_allclose(
                res_good.x, expected_pinv, rtol=1e-10, atol=1e-10
            )
