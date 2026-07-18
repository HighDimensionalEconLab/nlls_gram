import types

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import pytest

from nlls_gram import (
    LMStatus,
    UnderdeterminedLevenbergMarquardt,
    gram_lm,
    identity_preconditioner,
    metric_from_cholesky,
    recycled_cg,
)


def residual_fn(x, args, p):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


REGRESSION_ATOL = 5e-5
REGRESSION_RTOL = 1e-5


@pytest.fixture
def use_recycled_cg(monkeypatch):
    # Swap the fork in for jax's cg at both gram_lm call sites (forward dual
    # solve and implicit tangent solve) without touching jax's own module.
    monkeypatch.setattr(
        gram_lm, "jsp_sparse_linalg", types.SimpleNamespace(cg=recycled_cg)
    )


def spd_test_system(n=30):
    W = jax.random.normal(jax.random.key(0), (n, n))
    A = W @ W.T + n * jnp.eye(n)
    b = jax.random.normal(jax.random.key(1), (n,))
    return A, b


# --- direct parity with jax.scipy.sparse.linalg.cg (identical graphs, so the
# --- results must match bitwise, not just to tolerance)


def test_recycled_cg_matches_jax_cg_bitwise():
    A, b = spd_test_system()

    def matvec(v):
        return A @ v

    expected, expected_info = jsp_sparse_linalg.cg(matvec, b, tol=1e-5, maxiter=100)
    got, info = recycled_cg(matvec, b, tol=1e-5, maxiter=100)

    assert info is expected_info is None
    assert jnp.array_equal(got, expected)


def test_recycled_cg_preconditioned_warm_start_matches_jax_cg():
    A, b = spd_test_system()
    weights = jnp.diag(A)
    x0 = 0.1 * jnp.ones_like(b)

    def matvec(v):
        return A @ v

    def preconditioner(v):
        return v / weights

    expected, _ = jsp_sparse_linalg.cg(
        matvec, b, x0=x0, tol=1e-5, atol=1e-8, maxiter=50, M=preconditioner
    )
    got, _ = recycled_cg(
        matvec, b, x0=x0, tol=1e-5, atol=1e-8, maxiter=50, M=preconditioner
    )

    assert jnp.array_equal(got, expected)


def test_recycled_cg_gradient_matches_jax_cg():
    # Exercises the imported _isolve/custom_linear_solve wrapper: derivatives
    # come from an implicit transpose solve, not from unrolling the loop.
    A, b = spd_test_system()

    def matvec(v):
        return A @ v

    def loss_fork(b):
        x, _ = recycled_cg(matvec, b, tol=1e-6, maxiter=200)
        return jnp.sum(x**2)

    def loss_jax(b):
        x, _ = jsp_sparse_linalg.cg(matvec, b, tol=1e-6, maxiter=200)
        return jnp.sum(x**2)

    assert jnp.array_equal(jax.grad(loss_fork)(b), jax.grad(loss_jax)(b))


def test_recycled_cg_jits():
    A, b = spd_test_system()

    @jax.jit
    def solve(b):
        return recycled_cg(lambda v: A @ v, b, tol=1e-5, maxiter=100)[0]

    eager, _ = recycled_cg(lambda v: A @ v, b, tol=1e-5, maxiter=100)

    assert jnp.allclose(solve(b), eager, rtol=1e-6, atol=1e-6)


def test_swap_fixture_reaches_forked_cg(monkeypatch):
    # Guard that the SimpleNamespace swap is not a silent no-op: count trace-
    # time calls through the patched attribute during a cg update.
    calls = []

    def counting_cg(*args, **kwargs):
        calls.append(1)
        return recycled_cg(*args, **kwargs)

    monkeypatch.setattr(
        gram_lm, "jsp_sparse_linalg", types.SimpleNamespace(cg=counting_cg)
    )
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )

    solver.update(x, solver.init(x, (ts, ys)), (ts, ys))

    assert calls


# --- copies of the basic CG tests from test_gram_lm.py, run with the fork
# --- swapped in for jax's cg


def test_cg_step_matches_cholesky_identity_step(use_recycled_cg):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}

    cholesky_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )

    cholesky_x, cholesky_state, cholesky_info = cholesky_solver.update(
        x, cholesky_solver.init(x, (ts, ys)), (ts, ys)
    )
    cg_x, cg_state, cg_info = cg_solver.update(x, cg_solver.init(x, (ts, ys)), (ts, ys))

    assert bool(cg_info.accepted) == bool(cholesky_info.accepted)
    assert not bool(cg_info.used_geodesic)
    assert jnp.allclose(cg_x["a"], cholesky_x["a"], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(cg_x["b"], cholesky_x["b"], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(cg_state.damping, cholesky_state.damping)
    assert jnp.allclose(
        cg_info.loss,
        cholesky_info.loss,
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )
    assert cg_x["a"].dtype == jnp.float32
    assert cg_info.loss.dtype == jnp.float32


def test_cg_update_jits(use_recycled_cg):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=20,
    )

    @jax.jit
    def train_step(x, lm_state):
        return solver.update(x, lm_state, (ts, ys))

    x, lm_state, info = train_step(x, solver.init(x, (ts, ys)))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(lm_state.damping)
    assert x["a"].dtype == jnp.float32


def test_cg_geodesic_acceleration_matches_cholesky(use_recycled_cg):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=10,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )

    cholesky_theta, _, cholesky_info = cholesky_solver.update(
        theta0, cholesky_solver.init(theta0, target), target
    )
    cg_theta, _, cg_info = cg_solver.update(
        theta0, cg_solver.init(theta0, target), target
    )

    assert bool(cg_info.accepted)
    assert bool(cg_info.used_geodesic)
    assert jnp.allclose(cg_theta, cholesky_theta, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(
        cg_info.acceleration_ratio,
        cholesky_info.acceleration_ratio,
        rtol=1e-6,
        atol=1e-6,
    )


def test_cg_preconditioned_step_matches_cholesky_identity_step(use_recycled_cg):
    # A valid SPD preconditioner changes only the inner Krylov iteration, not
    # the step the inner solve converges to.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    weights = 1.0 + jnp.arange(20, dtype=jnp.float32) / 10.0

    cholesky_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=40,
        dual_preconditioner=lambda v, damping: v / weights,
    )

    cholesky_x, cholesky_state, cholesky_info = cholesky_solver.update(
        x, cholesky_solver.init(x, (ts, ys)), (ts, ys)
    )
    cg_x, cg_state, cg_info = cg_solver.update(x, cg_solver.init(x, (ts, ys)), (ts, ys))

    assert bool(cg_info.accepted) == bool(cholesky_info.accepted)
    # float32 across BLAS/SIMD variants: CI runners land ~2e-5 off macOS.
    assert jnp.allclose(cg_x["a"], cholesky_x["a"], rtol=1e-4, atol=1e-4)
    assert jnp.allclose(cg_x["b"], cholesky_x["b"], rtol=1e-4, atol=1e-4)
    assert jnp.allclose(cg_state.damping, cholesky_state.damping)
    assert jnp.allclose(
        cg_info.loss,
        cholesky_info.loss,
        rtol=REGRESSION_RTOL,
        atol=REGRESSION_ATOL,
    )


def test_cg_preconditioned_update_jits(use_recycled_cg):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    weights = 1.0 + jnp.arange(20, dtype=jnp.float32) / 10.0
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=40,
        dual_preconditioner=lambda v, damping: v / weights,
    )

    @jax.jit
    def train_step(x, lm_state):
        return solver.update(x, lm_state, (ts, ys))

    x, lm_state, info = train_step(x, solver.init(x, (ts, ys)))

    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)
    assert jnp.isfinite(lm_state.damping)


def test_cg_preconditioned_geodesic_matches_cholesky(use_recycled_cg):
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0

    cholesky_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    cg_solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=10,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
        dual_preconditioner=lambda v, damping: v / (1.0 + damping),
    )

    cholesky_theta, _, cholesky_info = cholesky_solver.update(
        theta0, cholesky_solver.init(theta0, target), target
    )
    cg_theta, _, cg_info = cg_solver.update(
        theta0, cg_solver.init(theta0, target), target
    )

    assert bool(cg_info.accepted)
    assert bool(cg_info.used_geodesic)
    assert jnp.allclose(cg_theta, cholesky_theta, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(
        cg_info.acceleration_ratio,
        cholesky_info.acceleration_ratio,
        rtol=1e-6,
        atol=1e-6,
    )


def test_cg_dual_preconditioner_enables_ill_conditioned_convergence(use_recycled_cg):
    # Kernel-collocation miniature: affine residual K x - b with metric M = K,
    # so the dual operator is K K^{-1} K = K itself -- as ill-conditioned as
    # the kernel (cond ~ 1e4 here). At a tight inner-iteration budget plain CG
    # stalls, while the exact preconditioner (a K-solve) recovers
    # Gauss-Newton-quality steps and converges immediately.
    n = 40
    rho = 0.98
    idx = jnp.arange(n)
    K = rho ** jnp.abs(idx[:, None] - idx[None, :])
    L = jnp.linalg.cholesky(K)
    x_true = jnp.sin(idx / 3.0)
    b = K @ x_true

    def residual(x):
        return K @ x - b

    def preconditioner(v, damping):
        y = jsp_linalg.solve_triangular(L, v, lower=True)
        return jsp_linalg.solve_triangular(L.T, y, lower=False)

    common = dict(
        init_damping=1e-6,
        linear_solver="cg",
        iterative_maxiter=3,
        implicit_preconditioner=identity_preconditioner(),
        metric=metric_from_cholesky(L),
    )
    plain = UnderdeterminedLevenbergMarquardt(
        residual, dual_preconditioner=identity_preconditioner(), **common
    )
    preconditioned = UnderdeterminedLevenbergMarquardt(
        residual, dual_preconditioner=preconditioner, **common
    )
    x0 = jnp.zeros(n)

    plain_result = plain.solve(x0, max_steps=20, atol=1e-3)
    preconditioned_result = preconditioned.solve(x0, max_steps=20, atol=1e-3)

    assert int(preconditioned_result.status) == LMStatus.CONVERGED
    assert int(plain_result.status) != LMStatus.CONVERGED
