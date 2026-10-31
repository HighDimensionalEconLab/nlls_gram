"""Tests for Metric and Preconditioner classes, callback refresh, and StateSpace."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from nlls_gram import (
    CG,
    BlockEigenPreconditioner,
    GramCG,
    IdentityMetric,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMState,
    Newton,
    NystromPreconditioner,
    PaddedPreconditioner,
    RepeatedFactorMetric,
    ShermanMorrisonPreconditioner,
    SolverContext,
    SolverStatus,
    WoodburyPreconditioner,
    register_pytree_dataclass,
)
from nlls_gram.experimental import StateSpaceMetric, matern_state_space

REPEATS = 3
BLOCK = 4
N_M = REPEATS * BLOCK
CTX = SolverContext()


def make_factor(key, size):
    root = jax.random.normal(key, (size, size + 2))
    K = root @ root.T + 0.5 * jnp.eye(size)
    return jnp.linalg.cholesky(K, upper=True), K


def test_metrics_contract_and_live_context():
    F, K = make_factor(jax.random.key(0), BLOCK)
    metric = RepeatedFactorMetric(F, repeats=REPEATS)
    assert metric.size == N_M
    F_bar = jsp_linalg.block_diag(*([F] * REPEATS))
    W = jsp_linalg.block_diag(*([K] * REPEATS))
    v = jax.random.normal(jax.random.key(1), (N_M,))
    matrix = jax.random.normal(jax.random.key(2), (N_M, 3))

    applied = metric.factor_apply(v, CTX)
    np.testing.assert_allclose(applied, F_bar @ v, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        metric.factor_apply(matrix, CTX), F_bar @ matrix, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        metric.factor_solve(applied, CTX), v, rtol=2e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        metric.factor_solve_transpose(matrix, CTX),
        np.linalg.solve(np.asarray(F_bar, np.float64).T, np.asarray(matrix)),
        rtol=2e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        float(metric.norm(v, CTX)) ** 2, float(v @ (W @ v)), rtol=1e-4
    )

    # IdentityMetric is plain Euclidean
    ident = IdentityMetric(6)
    v6 = jax.random.normal(jax.random.key(3), (6,))
    np.testing.assert_array_equal(ident.factor_apply(v6, CTX), v6)
    np.testing.assert_allclose(float(ident.norm(v6, CTX)), float(jnp.linalg.norm(v6)))

    # Validation
    with pytest.raises(ValueError, match="square"):
        RepeatedFactorMetric(jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="repeats"):
        RepeatedFactorMetric(jnp.eye(2), repeats=0)

    # Live SolverContext reaches factor ops
    seen = []

    class Probe(IdentityMetric):
        def factor_apply(self, vec, ctx):
            seen.append(ctx.x is not None and ctx.state is not None)
            return vec

        factor_solve = factor_apply
        factor_solve_transpose = factor_apply

    register_pytree_dataclass(Probe, data_fields=("free_scale",), meta_fields=("size",))
    A = jnp.ones((2, 4), jnp.float32)
    s = LevenbergMarquardt(lambda th: A @ th - 1.0, metric=Probe(3), ridge=1e-3)
    s.update(jnp.zeros(4), s.init(jnp.zeros(4)))
    assert seen and all(seen)


def test_state_space_metric_sequential_and_parallel_agree():
    n, repeats = 64, 2
    t = jnp.cumsum(jnp.ones(n, dtype=jnp.float32))
    x = jnp.sin(jnp.linspace(0.0, 6.0, repeats * n, dtype=jnp.float32))
    model = matern_state_space(jnp.float32(1.3), jnp.float32(0.8), 2.5)
    seq = StateSpaceMetric(t, *model, repeats=repeats, epsilon=1e-5, parallel=False)
    par = StateSpaceMetric(t, *model, repeats=repeats, epsilon=1e-5, parallel=True)
    np.testing.assert_allclose(
        seq.factor_solve(x, None), par.factor_solve(x, None), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        seq.factor_apply(x, None), par.factor_apply(x, None), rtol=1e-4, atol=1e-5
    )


def test_block_eigen_and_dual_preconditioners():
    # 1. BlockEigenPreconditioner matches dense shifted inverse at damping=0 and >0
    keys = jax.random.split(jax.random.key(0), 3)

    def spd_blocks(key, groups, size):
        root = jax.random.normal(key, (groups, size, size))
        return jnp.einsum("gik,gjk->gij", root, root) + 0.5 * jnp.eye(size)

    fam_a = spd_blocks(keys[0], 2, 3)
    fam_free = spd_blocks(keys[1], 1, 2)
    perm = jnp.asarray(np.random.default_rng(0).permutation(8))
    precond = BlockEigenPreconditioner([(fam_a, 1.0), (fam_free, 0.0)], perm)
    dense = jsp_linalg.block_diag(fam_a[0], fam_a[1], fam_free[0])
    ridge_mask = jnp.concatenate([jnp.ones(6), jnp.zeros(2)])
    ridge = 0.05
    ctx = SolverContext(
        state=LMState(damping=jnp.asarray(1e-3), ridge=jnp.asarray(ridge))
    )
    v = jax.random.normal(keys[2], (8,))
    sel = jnp.eye(8)[perm]
    for damping in (0.0, 0.37):
        shifted = dense + jnp.diag(ridge_mask * ridge + damping)
        expected = jnp.linalg.solve(sel.T @ shifted @ sel, v)
        np.testing.assert_allclose(
            precond.apply(v, jnp.asarray(damping), ctx),
            expected,
            rtol=2e-4,
            atol=2e-5,
        )

    # 2. Callback refresh of BlockEigenPreconditioner drives next step and AD inherits
    p_dim, m_resid = 10, 14
    n_m, n_f = 8, 2
    F, _ = make_factor(jax.random.key(2), 4)
    metric = RepeatedFactorMetric(F, repeats=2)
    A = jax.random.normal(jax.random.key(3), (m_resid, p_dim)) / jnp.sqrt(p_dim)
    target = jax.random.normal(jax.random.key(4), (m_resid,))
    F_bar = jsp_linalg.block_diag(F, F, jnp.eye(n_f))
    J_w = jnp.linalg.solve(F_bar.T, A.T).T
    G = J_w.T @ J_w
    exact_p = BlockEigenPreconditioner(
        [(G[:n_m, :n_m][None], 1.0), (G[n_m:, n_m:][None], 0.0)], jnp.arange(p_dim)
    )
    scrambled_p = BlockEigenPreconditioner(
        [(100.0 * jnp.eye(n_m)[None], 1.0), (100.0 * jnp.eye(n_f)[None], 0.0)],
        jnp.arange(p_dim),
    )

    def res_fn(x, _, p):
        return A @ x - p * target

    s_scram = LevenbergMarquardt(
        res_fn,
        metric=metric,
        ridge=1e-3,
        linear_solver=CG(scrambled_p, tol=0.0, maxiter=3),
    )
    s_exact = LevenbergMarquardt(
        res_fn, metric=metric, ridge=1e-3, linear_solver=CG(exact_p, tol=0.0, maxiter=3)
    )
    x0 = jnp.zeros(p_dim)
    p0 = jnp.float32(1.0)
    swapped_state = dataclasses.replace(s_scram.init(x0, p=p0), preconditioner=exact_p)
    np.testing.assert_allclose(
        s_scram.update(x0, swapped_state, p=p0)[0],
        s_exact.update(x0, s_exact.init(x0, p=p0), p=p0)[0],
        rtol=1e-6,
        atol=1e-7,
    )

    # 3. Dual preconditioners (Identity, ShermanMorrison, Woodbury, Nystrom, Padded)
    rng = np.random.default_rng(5)
    m = 4
    A_dual = jnp.asarray(rng.normal(size=(m, 7)), jnp.float32)
    b_dual = jnp.asarray(rng.normal(size=m), jnp.float32)
    dual_mat = A_dual @ A_dual.T + 0.5 * jnp.eye(m, dtype=jnp.float32)
    u = jnp.asarray(rng.normal(size=m), jnp.float32)
    U = jnp.asarray(rng.normal(size=(m, 2)), jnp.float32)
    w = jnp.asarray([0.4, 0.7], jnp.float32)
    dual_solve = lambda vec: jnp.linalg.solve(dual_mat, vec)  # noqa: E731
    dual_mv = lambda mat: dual_mat @ mat  # noqa: E731

    for pre in (
        IdentityPreconditioner(),
        ShermanMorrisonPreconditioner(dual_solve, u, jnp.float32(0.3)),
        WoodburyPreconditioner(dual_solve, U, w),
        NystromPreconditioner(
            dual_mv, n=m, rank=2, key=jax.random.key(0), dtype=jnp.float32
        ),
        PaddedPreconditioner(IdentityPreconditioner(), n_real=m - 1),
    ):
        vec = jnp.asarray(rng.normal(size=m), jnp.float32)
        out = pre.apply(vec, jnp.float32(0.1), CTX)
        assert out.dtype == jnp.float32 and float(jnp.dot(vec, out)) > 0.0

    # GramCG with WoodburyPreconditioner converges on Newton
    n_sol = Newton(
        lambda x: A_dual @ x - b_dual,
        linear_solver=GramCG(
            WoodburyPreconditioner(dual_solve, U, w), tol=1e-7, maxiter=64
        ),
    ).solve(jnp.zeros(7, jnp.float32), atol=1e-5)
    assert int(n_sol.status) == int(SolverStatus.CONVERGED)

    # NystromPreconditioner avoids catastrophic cancellation on ill-conditioned
    # float32 operators and stays finite at zero damping even when rho == 0.
    diag_ill = jnp.array([1e8, 1.0, 1.0, 1.0], dtype=jnp.float32)
    nys_ill = NystromPreconditioner(
        lambda mat: diag_ill[:, None] * mat,
        n=4,
        rank=2,
        key=jax.random.key(7),
        dtype=jnp.float32,
    )
    v_perp = jnp.array([0.0, 1.0, -1.0, 0.5], dtype=jnp.float32)
    out_ill = nys_ill.apply(v_perp, jnp.float32(0.0), CTX)
    assert jnp.all(jnp.isfinite(out_ill))
    assert float(jnp.linalg.norm(out_ill)) > 0.1

    diag_rank1 = jnp.array([10.0, 0.0, 0.0], dtype=jnp.float32)
    nys_full = NystromPreconditioner(
        lambda mat: diag_rank1[:, None] * mat,
        n=3,
        rank=3,
        key=jax.random.key(9),
        dtype=jnp.float32,
    )
    assert jnp.all(
        jnp.isfinite(
            nys_full.apply(jnp.ones(3, dtype=jnp.float32), jnp.float32(0.0), CTX)
        )
    )
