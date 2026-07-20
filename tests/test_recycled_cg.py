import dataclasses
import types

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    LMSolveAction,
    LMStatus,
    MultiStart,
    RecycleConfig,
    RecycleState,
    gram_lm,
    identity_preconditioner,
    metric_from_cholesky,
    recycled_cg,
)
from nlls_gram.recycled_cg import (
    HarvestState,
    build_coarse_operator,
    deflated_pcg,
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
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="gram_cg",
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

    cholesky_solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="gram_cg",
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
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="gram_cg",
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

    cholesky_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    cg_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="gram_cg",
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

    cholesky_solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
    cg_solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="gram_cg",
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
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="gram_cg",
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

    cholesky_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    cg_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="gram_cg",
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
        linear_solver="gram_cg",
        iterative_maxiter=3,
        implicit_preconditioner=identity_preconditioner(),
        metric=metric_from_cholesky(L),
    )
    plain = LevenbergMarquardt(
        residual, dual_preconditioner=identity_preconditioner(), **common
    )
    preconditioned = LevenbergMarquardt(
        residual, dual_preconditioner=preconditioner, **common
    )
    x0 = jnp.zeros(n)

    plain_result = plain.solve(x0, max_steps=20, atol=1e-3)
    preconditioned_result = preconditioned.solve(x0, max_steps=20, atol=1e-3)

    assert int(preconditioned_result.status) == LMStatus.CONVERGED
    assert int(plain_result.status) != LMStatus.CONVERGED


# --- deflated / recycled PCG ------------------------------------------------
#
# The additive two-level preconditioner M_defl = P + U E^-1 U' shifts each
# deflated eigenvalue lambda -> lambda + P|_u, so it CLUSTERS (and speeds CG)
# when the deflated modes are small outliers near 0 and the first-level P
# normalizes the bulk near 1. The fixtures below use that regime: a few tiny
# isolated eigenvalues plus a tight bulk near 1.


def clustered_spd(n, small_eigs, bulk=1.0, bulk_spread=1e-4, seed=0):
    key = jax.random.key(seed)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
    k = small_eigs.shape[0]
    bulk_eigs = bulk + bulk_spread * jax.random.uniform(
        jax.random.key(seed + 1), (n - k,)
    )
    eigs = jnp.concatenate([small_eigs, bulk_eigs])
    A = (Q * eigs) @ Q.T
    A = 0.5 * (A + A.T)
    return A, Q, eigs


def max_subspace_angle_deg(U, V):
    # Largest principal angle between range(U) and range(V) (both orthonormal).
    cos_angles = jnp.linalg.svd(U.T @ V, compute_uv=False)
    return float(jnp.degrees(jnp.arccos(jnp.clip(jnp.min(cos_angles), 0.0, 1.0))))


def test_deflated_pcg_cold_matches_recycled_cg_bitwise():
    # U=0 with a non-identity first-level P: the coarse correction and deflated
    # init vanish exactly (ridge floor keeps E finite while U'r=0), and the two
    # cond-functions both read ||r||^2, so the iterates are bitwise identical.
    n, k = 40, 4
    A, _, _ = clustered_spd(n, jnp.array([0.01, 0.02, 0.04, 0.08]))
    b = jax.random.normal(jax.random.key(1), (n,))
    weights = jnp.diag(A)

    def matvec(v):
        return A @ v

    def P(v):
        return v / weights

    U0 = jnp.zeros((n, k))
    _, E_factor = build_coarse_operator(matvec, U0)
    got, harvest = deflated_pcg(
        matvec,
        b,
        U=U0,
        E_factor=E_factor,
        M=P,
        tol=1e-5,
        atol=0.0,
        maxiter=300,
        window=3 * k,
        rank=k,
    )
    expected, _ = recycled_cg(matvec, b, tol=1e-5, atol=0.0, maxiter=300, M=P)

    assert isinstance(harvest, HarvestState)
    assert jnp.array_equal(got, expected)


def test_deflated_pcg_solves_spd_system():
    A, b = spd_test_system()
    n, k = b.shape[0], 3
    U0 = jnp.zeros((n, k))
    _, E_factor = build_coarse_operator(lambda v: A @ v, U0)
    got, _ = deflated_pcg(
        lambda v: A @ v,
        b,
        U=U0,
        E_factor=E_factor,
        tol=1e-7,
        atol=0.0,
        maxiter=200,
        window=2 * k,
        rank=k,
    )
    assert jnp.allclose(got, jnp.linalg.solve(A, b), rtol=1e-4, atol=1e-4)


def test_exact_basis_reduces_iterations():
    # A supplied basis spanning the k smallest eigenvectors clusters those modes
    # and cuts the iteration count to a fixed tolerance well below undeflated CG.
    n, k = 50, 4
    small = jnp.array([0.01, 0.02, 0.04, 0.08])
    A, Q, eigs = clustered_spd(n, small)
    b = jax.random.normal(jax.random.key(1), (n,))
    order = jnp.argsort(eigs)
    U_exact = Q[:, order[:k]]

    def matvec(v):
        return A @ v

    common = dict(tol=1e-6, atol=0.0, maxiter=300, window=3 * k, rank=k)
    _, cold = deflated_pcg(
        matvec,
        b,
        U=jnp.zeros((n, k)),
        E_factor=build_coarse_operator(matvec, jnp.zeros((n, k)))[1],
        **common,
    )
    _, defl = deflated_pcg(
        matvec,
        b,
        U=U_exact,
        E_factor=build_coarse_operator(matvec, U_exact)[1],
        **common,
    )
    assert int(defl.iterations) <= int(cold.iterations) // 2
    assert int(defl.iterations) <= 3


def test_harvest_approximates_smallest_eigenvectors():
    # One solve harvests a basis that approximates the true smallest eigenvectors
    # of the operator (subspace angle vs dense eigh) and is orthonormal.
    n, k = 50, 4
    small = jnp.array([0.01, 0.02, 0.04, 0.08])
    A, _, _ = clustered_spd(n, small)
    b = jax.random.normal(jax.random.key(1), (n,))
    _, evecs = jnp.linalg.eigh(A)
    true_small = evecs[:, :k]

    _, harvest = deflated_pcg(
        lambda v: A @ v,
        b,
        U=jnp.zeros((n, k)),
        E_factor=build_coarse_operator(lambda v: A @ v, jnp.zeros((n, k)))[1],
        tol=1e-6,
        atol=0.0,
        maxiter=n,
        window=3 * k,
        rank=k,
    )
    U = harvest.basis
    assert float(jnp.max(jnp.abs(U.T @ U - jnp.eye(k)))) < 1e-4
    assert max_subspace_angle_deg(U, true_small) < 5.0


def test_recycling_reduces_total_iterations():
    # A slowly drifting sequence A_j = A + t_j Delta: carrying the harvested basis
    # across solves drops the total iteration count well below the no-recycle run.
    n, k = 60, 4
    small = jnp.array([1e-3, 5e-3, 2e-2, 8e-2])
    A, _, _ = clustered_spd(n, small)
    D = jax.random.normal(jax.random.key(7), (n, n))
    Delta = (D @ D.T) / n
    b0 = jax.random.normal(jax.random.key(3), (n,))
    w = 2 * k

    def total_iterations(recycle):
        total = 0
        U = jnp.zeros((n, k))
        for j in range(8):
            Aj = A + (0.005 * j) * Delta
            Aj = 0.5 * (Aj + Aj.T)

            def matvec(v, Aj=Aj):
                return Aj @ v

            U_in = U if recycle else jnp.zeros((n, k))
            _, E_factor = build_coarse_operator(matvec, U_in)
            _, harvest = deflated_pcg(
                matvec,
                b0 + 0.02 * j,
                U=U_in,
                E_factor=E_factor,
                tol=1e-6,
                atol=0.0,
                maxiter=500,
                window=w,
                rank=k,
            )
            total += int(harvest.iterations)
            U = harvest.basis
        return total

    cold_total = total_iterations(recycle=False)
    recycled_total = total_iterations(recycle=True)
    assert recycled_total < 0.7 * cold_total


def test_deflated_pcg_gradient_matches_dense_solve():
    # Derivatives are implicit through the custom_linear_solve wrapper; the
    # deflation and harvest are stop_gradient'd, so the gradient matches the dense
    # linear solve regardless of the (populated) deflation basis.
    n, k = 30, 3
    A, _, _ = clustered_spd(n, jnp.array([0.02, 0.05, 0.1]))
    b = jax.random.normal(jax.random.key(1), (n,))
    A_inv = jnp.linalg.inv(A)
    w = 3 * k

    _, harvest = deflated_pcg(
        lambda v: A @ v,
        b,
        U=jnp.zeros((n, k)),
        E_factor=build_coarse_operator(lambda v: A @ v, jnp.zeros((n, k)))[1],
        tol=1e-7,
        atol=0.0,
        maxiter=200,
        window=w,
        rank=k,
    )
    U = harvest.basis

    def loss(rhs):
        _, E_factor = build_coarse_operator(lambda v: A @ v, U)
        x, _ = deflated_pcg(
            lambda v: A @ v,
            rhs,
            U=U,
            E_factor=E_factor,
            tol=1e-7,
            atol=0.0,
            maxiter=300,
            window=w,
            rank=k,
        )
        return jnp.sum(x**2)

    def loss_dense(rhs):
        return jnp.sum((A_inv @ rhs) ** 2)

    got = jax.grad(loss)(b)
    expected = jax.grad(loss_dense)(b)
    assert jnp.allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_deflated_pcg_jits():
    n, k = 30, 3
    A, _, _ = clustered_spd(n, jnp.array([0.02, 0.05, 0.1]))
    b = jax.random.normal(jax.random.key(1), (n,))
    w = 3 * k

    @jax.jit
    def solve(rhs):
        U0 = jnp.zeros((n, k))
        _, E_factor = build_coarse_operator(lambda v: A @ v, U0)
        x, harvest = deflated_pcg(
            lambda v: A @ v,
            rhs,
            U=U0,
            E_factor=E_factor,
            tol=1e-5,
            atol=0.0,
            maxiter=200,
            window=w,
            rank=k,
        )
        return x, harvest.iterations

    x, iters = solve(b)
    assert jnp.allclose(x, jnp.linalg.solve(A, b), rtol=1e-3, atol=1e-3)
    assert int(iters) >= 1


def test_reorthogonalize_false_also_solves():
    # The cheap coefficient-tridiagonal harvest still yields the correct solution
    # (the harvest quality only affects the NEXT solve, never correctness).
    n, k = 40, 4
    A, _, _ = clustered_spd(n, jnp.array([0.01, 0.02, 0.04, 0.08]))
    b = jax.random.normal(jax.random.key(1), (n,))
    U0 = jnp.zeros((n, k))
    _, E_factor = build_coarse_operator(lambda v: A @ v, U0)
    got, harvest = deflated_pcg(
        lambda v: A @ v,
        b,
        U=U0,
        E_factor=E_factor,
        tol=1e-6,
        atol=0.0,
        maxiter=200,
        window=2 * k,
        rank=k,
        reorthogonalize=False,
    )
    assert jnp.allclose(got, jnp.linalg.solve(A, b), rtol=1e-3, atol=1e-3)
    assert bool(jnp.all(jnp.isfinite(harvest.basis)))


def test_harvest_ring_wrap_indexing():
    # Force CG past the window so the ring buffer wraps (start = count - w > 0),
    # exercising the perm / off-boundary indexing. The robust reorthogonalize=True
    # harvest still resolves the smallest eigenvectors from the correct windowed
    # sub-block (a buggy perm would send the subspace angle to ~90 deg); the cheap
    # reorthogonalize=False route is polluted under wrap but must stay finite and
    # orthonormal with Rayleigh quotients inside the spectrum.
    n, k = 90, 3
    A, _, eigs = clustered_spd(n, jnp.array([1e-3, 1e-2, 1e-1]), seed=4)
    b = jax.random.normal(jax.random.key(1), (n,))
    _, evecs = jnp.linalg.eigh(A)
    true_small = evecs[:, :k]
    lo, hi = float(jnp.min(eigs)), float(jnp.max(eigs))
    w = 6

    _, robust = deflated_pcg(
        lambda v: A @ v,
        b,
        U=jnp.zeros((n, k)),
        E_factor=build_coarse_operator(lambda v: A @ v, jnp.zeros((n, k)))[1],
        tol=1e-8,
        atol=0.0,
        maxiter=n,
        window=w,
        rank=k,
        reorthogonalize=True,
    )
    assert int(robust.iterations) > w  # ring wrapped
    assert float(jnp.max(jnp.abs(robust.basis.T @ robust.basis - jnp.eye(k)))) < 1e-5
    assert max_subspace_angle_deg(robust.basis, true_small) < 30.0

    _, cheap = deflated_pcg(
        lambda v: A @ v,
        b,
        U=jnp.zeros((n, k)),
        E_factor=build_coarse_operator(lambda v: A @ v, jnp.zeros((n, k)))[1],
        tol=1e-8,
        atol=0.0,
        maxiter=n,
        window=w,
        rank=k,
        reorthogonalize=False,
    )
    assert bool(jnp.all(jnp.isfinite(cheap.basis)))
    assert float(jnp.max(jnp.abs(cheap.basis.T @ cheap.basis - jnp.eye(k)))) < 1e-5
    ritz = jnp.diag(cheap.basis.T @ (A @ cheap.basis))
    assert bool(jnp.all((ritz >= lo - 1e-6) & (ritz <= hi + 1e-6)))


def test_deflated_pcg_count_zero_warm_started():
    # Warm-starting at the exact solution with a cold U drives count == 0 (the
    # loop never runs). The all-invalid harvest must return a finite orthonormal
    # basis (via the sentinel finite fallback), and it must not poison the next
    # build_coarse_operator.
    n, k = 40, 4
    A, _, _ = clustered_spd(n, jnp.array([0.01, 0.02, 0.04, 0.08]))
    b = jax.random.normal(jax.random.key(1), (n,))
    x_star = jnp.linalg.solve(A, b)
    U0 = jnp.zeros((n, k))
    _, E_factor = build_coarse_operator(lambda v: A @ v, U0)
    # tol loose enough that the (float32) warm start is already converged, so the
    # loop never runs.
    y, harvest = deflated_pcg(
        lambda v: A @ v,
        b,
        U=U0,
        E_factor=E_factor,
        x0=x_star,
        tol=1e-3,
        atol=0.0,
        maxiter=200,
        window=3 * k,
        rank=k,
    )
    assert int(harvest.iterations) == 0
    assert bool(jnp.all(jnp.isfinite(harvest.basis)))
    # the returned solution is converged to tol (the differentiable pass solves
    # from zeros, independently of the warm-started harvest pass).
    assert float(jnp.linalg.norm(A @ y - b)) <= 1e-3 * float(jnp.linalg.norm(b))
    _, next_factor = build_coarse_operator(lambda v: A @ v, harvest.basis)
    assert bool(jnp.all(jnp.isfinite(next_factor[0])))


def test_harvest_false_returns_carried_basis():
    # harvest=False skips the Rayleigh-Ritz and returns the carried U verbatim; the
    # solution is unchanged from harvest=True (only the emitted basis differs).
    n, k = 40, 4
    A, Q, eigs = clustered_spd(n, jnp.array([0.01, 0.02, 0.04, 0.08]))
    b = jax.random.normal(jax.random.key(1), (n,))
    order = jnp.argsort(eigs)
    U = Q[:, order[:k]]
    _, E_factor = build_coarse_operator(lambda v: A @ v, U)
    common = dict(
        U=U,
        E_factor=E_factor,
        tol=1e-7,
        atol=0.0,
        maxiter=200,
        window=3 * k,
        rank=k,
    )
    y_off, off = deflated_pcg(lambda v: A @ v, b, harvest=False, **common)
    y_on, on = deflated_pcg(lambda v: A @ v, b, harvest=True, **common)
    # harvest=False returns the carried basis and skips the harvest pass (0
    # reported iterations), but solves the same system.
    assert jnp.array_equal(off.basis, U)
    assert int(off.iterations) == 0
    assert int(on.iterations) > 0
    assert jnp.allclose(y_off, y_on, rtol=1e-5, atol=1e-5)


def test_build_coarse_operator_cold_is_finite():
    # Zero U -> E = 0; the trace-scaled ridge's absolute floor keeps the Cholesky
    # factor finite and M_defl == P (since U'r = 0), so cold start is a no-op.
    n, k = 20, 3
    A, _, _ = clustered_spd(n, jnp.array([0.02, 0.05, 0.1]))
    U0 = jnp.zeros((n, k))
    W, E_factor = build_coarse_operator(lambda v: A @ v, U0)
    assert bool(jnp.all(jnp.isfinite(W)))
    assert bool(jnp.all(jnp.isfinite(E_factor[0])))
    r = jax.random.normal(jax.random.key(2), (n,))
    coarse = U0 @ jsp_linalg.cho_solve(E_factor, U0.T @ r)
    assert jnp.allclose(coarse, jnp.zeros(n))


def test_degenerate_basis_stays_finite():
    # A rank-deficient U (duplicate columns) must not produce NaN: the ridge keeps
    # E factorable and the solve still converges to the correct solution.
    n, k = 30, 3
    A, Q, eigs = clustered_spd(n, jnp.array([0.02, 0.05, 0.1]))
    b = jax.random.normal(jax.random.key(1), (n,))
    order = jnp.argsort(eigs)
    u = Q[:, order[0]]
    U_dup = jnp.stack([u, u, Q[:, order[1]]], axis=1)  # duplicate column
    _, E_factor = build_coarse_operator(lambda v: A @ v, U_dup)
    got, harvest = deflated_pcg(
        lambda v: A @ v,
        b,
        U=U_dup,
        E_factor=E_factor,
        tol=1e-6,
        atol=0.0,
        maxiter=200,
        window=2 * k,
        rank=k,
    )
    assert bool(jnp.all(jnp.isfinite(got)))
    assert bool(jnp.all(jnp.isfinite(harvest.basis)))
    assert jnp.allclose(got, jnp.linalg.solve(A, b), rtol=1e-3, atol=1e-3)


def test_deflated_pcg_nan_operator_propagates():
    # A NaN entering through the ITERATION path (not build_coarse_operator or the
    # init) propagates to non-finite output; it is never clamped to a quiet
    # pseudo-solution. E_factor is built from the FINITE operator; the matvec goes
    # NaN only on nonzero inputs, so the init matvec on zeros stays finite and the
    # NaN first appears at the first A(p) inside the loop.
    n, k = 20, 3
    A, _, _ = clustered_spd(n, jnp.array([0.02, 0.05, 0.1]))
    b = jax.random.normal(jax.random.key(1), (n,))
    U0 = jnp.zeros((n, k))
    _, E_factor = build_coarse_operator(lambda v: A @ v, U0)

    def nan_matvec(v):
        return jnp.where(jnp.any(v != 0), (A @ v) * jnp.nan, A @ v)

    got, _ = deflated_pcg(
        nan_matvec,
        b,
        U=U0,
        E_factor=E_factor,
        tol=1e-5,
        atol=0.0,
        maxiter=20,
        window=2 * k,
        rank=k,
    )
    assert bool(jnp.any(~jnp.isfinite(got)))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rank=0),
        dict(window=2, rank=5),
        dict(window=100, rank=3),
    ],
)
def test_deflated_pcg_rejects_bad_rank_window(kwargs):
    n = 20
    A, _, _ = clustered_spd(n, jnp.array([0.02, 0.05, 0.1]))
    b = jax.random.normal(jax.random.key(1), (n,))
    U0 = jnp.zeros((n, 3))
    _, E_factor = build_coarse_operator(lambda v: A @ v, U0)
    with pytest.raises(ValueError):
        deflated_pcg(
            lambda v: A @ v,
            b,
            U=U0,
            E_factor=E_factor,
            tol=1e-5,
            atol=0.0,
            maxiter=10,
            **kwargs,
        )


# --- LM-level recycling integration -----------------------------------------


def isolated_mode_problem(m=24, n=30, num_small=3):
    # A mildly nonlinear underdetermined least squares whose Gauss-Newton dual
    # operator J J' has `num_small` isolated small eigenvalues plus a bulk near 1
    # -- the regime where the additive deflation clusters the spectrum. The
    # nonlinearity forces several LM steps so a carried basis can pay off.
    Um, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(1), (m, m)))
    Vn, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(2), (n, n)))
    small = jnp.array([0.02, 0.05, 0.12])[:num_small]
    sv = jnp.concatenate([small, jnp.ones(m - num_small)])
    G = Um @ (sv[:, None] * Vn[:m, :])
    b = jax.random.normal(jax.random.key(3), (m,))

    def residual(x):
        linear = G @ x - b
        return linear + 0.03 * linear**2

    return residual, jnp.zeros(n)


def _reset_recycle(lm_state):
    recycle = lm_state.recycle
    return dataclasses.replace(
        lm_state,
        recycle=RecycleState(
            jnp.zeros_like(recycle.U),
            jnp.zeros_like(recycle.dual_velocity),
            jnp.zeros_like(recycle.dual_accel),
            jnp.zeros_like(recycle.valid),
            jnp.zeros_like(recycle.iterations),
            jnp.zeros_like(recycle.residual_norm),
        ),
    )


def test_recycle_requires_cg():
    with pytest.raises(ValueError, match="recycle requires"):
        LevenbergMarquardt(
            lambda x: x, linear_solver="gram_cholesky", recycle=RecycleConfig(rank=2)
        )


def test_recycle_config_hashing_shares_compilation():
    # Equal RecycleConfig -> equal solver static key -> shared jit cache; a
    # different rank is a different compiled program.
    residual, _ = isolated_mode_problem()
    common = dict(
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
    )
    a = LevenbergMarquardt(residual, recycle=RecycleConfig(rank=4), **common)
    b = LevenbergMarquardt(residual, recycle=RecycleConfig(rank=4), **common)
    c = LevenbergMarquardt(residual, recycle=RecycleConfig(rank=5), **common)
    assert a == b and hash(a) == hash(b)
    assert a != c


def test_init_allocates_cold_recycle_state():
    residual, x0 = isolated_mode_problem()
    solver = LevenbergMarquardt(
        residual,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        recycle=RecycleConfig(rank=4),
    )
    state = solver.init(x0)
    assert state.recycle is not None
    assert not bool(state.recycle.valid)
    assert state.recycle.U.shape == (24, 4)
    assert jnp.all(state.recycle.U == 0)
    assert int(state.recycle.iterations) == 0


def test_recycling_reduces_total_inner_iterations():
    # A manual update() sequence: carrying the harvested basis across steps cuts
    # the total inner (velocity) CG iterations vs the identical solver with the
    # basis cold-reset each step (the controlled no-recycle baseline).
    residual, x0 = isolated_mode_problem()
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-4,
        linear_solver="gram_cg",
        geodesic_acceleration=False,
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_maxiter=40,
        iterative_tol=1e-8,
        recycle=RecycleConfig(rank=3),
    )

    def total_inner(reset):
        x, state = x0, solver.init(x0)
        total = 0
        for _ in range(8):
            x, state, info = solver.update(x, state)
            total += int(state.recycle.iterations)
            if reset:
                state = _reset_recycle(state)
        return total, float(info.loss)

    recycled_total, recycled_loss = total_inner(reset=False)
    baseline_total, baseline_loss = total_inner(reset=True)
    assert recycled_loss < 1e-8 and baseline_loss < 1e-8
    assert recycled_total < 0.75 * baseline_total


def test_recycled_solve_converges_ill_conditioned():
    residual, x0 = isolated_mode_problem()
    common = dict(
        init_damping=1e-4,
        linear_solver="gram_cg",
        geodesic_acceleration=False,
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_maxiter=6,
        iterative_tol=1e-8,
    )
    recycled = LevenbergMarquardt(residual, recycle=RecycleConfig(rank=3), **common)
    result = recycled.solve(x0, max_steps=60, atol=1e-6)
    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.info.loss) < 1e-6


def test_recycled_update_reverse_ad_matches_cholesky():
    # update()'s recycled path stays reverse-differentiable and matches the dense
    # cholesky reference: the deflation/harvest are stop_gradient'd, so only the
    # (converged) step carries gradient. Differentiate w.r.t. the target data.
    ts = jnp.linspace(0.0, 2.0, 20)

    def residual_data(x, args, p):
        return x["a"] * jnp.exp(x["b"] * ts) - args

    x = {"a": 1.0, "b": 0.0}
    cholesky = LevenbergMarquardt(residual_data, init_damping=1e-2)
    recycled = LevenbergMarquardt(
        residual_data,
        init_damping=1e-2,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-9,
        iterative_maxiter=60,
        recycle=RecycleConfig(rank=4),
    )

    def loss_of(solver):
        def loss(ys):
            new_x, _, _ = solver.update(x, solver.init(x, ys), ys)
            return jnp.sum(new_x["a"] ** 2 + new_x["b"] ** 2)

        return loss

    ys = 2.0 * jnp.exp(-1.0 * ts)
    g_recycled = jax.grad(loss_of(recycled))(ys)
    g_cholesky = jax.grad(loss_of(cholesky))(ys)
    assert jnp.allclose(g_recycled, g_cholesky, rtol=1e-3, atol=1e-4)


def test_recycled_solve_implicit_p_derivative_matches_plain():
    # solve()'s p-derivative comes from the implicit rule at the converged root;
    # recycling only changes the forward path, so the jacobian matches plain cg.
    ts = jnp.linspace(0.0, 2.0, 12)

    def residual_p(x, args, p):
        return x * ts - p

    common = dict(
        init_damping=1e-3,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-10,
        iterative_maxiter=40,
    )
    plain = LevenbergMarquardt(residual_p, **common)
    recycled = LevenbergMarquardt(residual_p, recycle=RecycleConfig(rank=3), **common)
    x0 = jnp.zeros(())

    def solved(solver, p):
        return solver.solve(x0, p=p, max_steps=40, atol=1e-9).x

    p = jnp.asarray(1.7)
    j_plain = jax.jacobian(lambda q: solved(plain, q))(p)
    j_recycled = jax.jacobian(lambda q: solved(recycled, q))(p)
    assert jnp.allclose(j_recycled, j_plain, rtol=1e-4, atol=1e-5)


def test_recycled_multi_start_vmap():
    residual, x0 = isolated_mode_problem()
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-4,
        linear_solver="gram_cg",
        geodesic_acceleration=False,
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_maxiter=8,
        iterative_tol=1e-8,
        recycle=RecycleConfig(rank=3),
    )

    def draw(key, x, args):
        return x + 0.01 * jax.random.normal(key, x.shape), args

    for parallel in (False, True):
        ms = MultiStart(
            key=jax.random.key(0), num_starts=3, draw=draw, parallel=parallel
        )
        result = solver.solve(x0, max_steps=60, atol=1e-6, multi_start=ms)
        assert int(result.status) == LMStatus.CONVERGED
        assert float(result.info.loss) < 1e-6


def test_recycled_callback_maxiter_schedule_composes():
    # A callback that grows iterative_maxiter mid-solve composes with recycling:
    # rank/window are static (untouched), the traced maxiter rides hyper.
    residual, x0 = isolated_mode_problem()
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-4,
        linear_solver="gram_cg",
        geodesic_acceleration=False,
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_maxiter=2,
        iterative_tol=1e-8,
        recycle=RecycleConfig(rank=3),
    )

    def schedule(ctx):
        grown = dataclasses.replace(
            ctx.lm_state,
            hyper=dataclasses.replace(
                ctx.lm_state.hyper,
                iterative_maxiter=jnp.where(
                    ctx.info.loss < 1e-2, jnp.int32(8), jnp.int32(2)
                ),
            ),
        )
        return LMSolveAction(lm_state=grown)

    result = solver.solve(x0, max_steps=60, atol=1e-6, callback=schedule)
    assert int(result.status) == LMStatus.CONVERGED


def test_recycled_multi_start_cold_resets_basis():
    # _cold_lm_state must reset the recycle basis for drawn starts.
    residual, x0 = isolated_mode_problem()
    solver = LevenbergMarquardt(
        residual,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        recycle=RecycleConfig(rank=3),
    )
    state = solver.init(x0)
    # populate a basis, then cold-reset
    _, warmed, _ = solver.update(x0, state)
    assert bool(warmed.recycle.valid)
    cold = gram_lm._cold_lm_state(warmed)
    assert not bool(cold.recycle.valid)
    assert jnp.all(cold.recycle.U == 0)
    assert jnp.all(cold.recycle.dual_velocity == 0)


def test_recycled_update_nan_residual_rejects_step():
    # A residual that goes non-finite makes the recycled dual solve non-finite;
    # LM must reject the step (never accept a NaN candidate).
    def residual(x):
        return jnp.array(
            [
                x[0] ** 2 - 4.0,
                x[0] - 1.0,
                x[0] + 2.0,
                2.0 * x[0],
                3.0 * x[0],
                jnp.log(x[0]),
            ]
        )

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_maxiter=10,
        recycle=RecycleConfig(rank=1),
    )
    x0 = jnp.array([-1.0])  # log(-1) -> nan in the residual
    _, _, info = solver.update(x0, solver.init(x0))
    assert not bool(info.accepted)
