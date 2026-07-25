# Float32 coverage for BlockEigenPreconditioner. The float64 primary coverage
# lives in test_float64_subprocess.py, matching the production default.
import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from nlls_gram import (
    CG,
    BlockEigenPreconditioner,
    Cholesky,
    LMAction,
    LMState,
    LMStatus,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    SolverContext,
)


def spd_blocks(key, groups, size):
    root = jax.random.normal(key, (groups, size, size))
    return jnp.einsum("gik,gjk->gij", root, root) + 0.5 * jnp.eye(size)


@pytest.mark.parametrize("damping", [0.0, 0.37])
def test_apply_matches_dense_inverse(damping):
    # Two ridge-flagged families (2 groups of 3, 1 group of 4) and one
    # damping-only free family (1 group of 2), under a nontrivial permutation
    # of the 12 coordinates.
    keys = jax.random.split(jax.random.key(0), 3)
    family_a = spd_blocks(keys[0], 2, 3)
    family_b = spd_blocks(keys[1], 1, 4)
    family_free = spd_blocks(keys[2], 1, 2)
    permutation = jnp.asarray(np.random.default_rng(0).permutation(12))
    families = [(family_a, 1.0), (family_b, 1.0), (family_free, 0.0)]
    dense_permuted = jsp_linalg.block_diag(
        family_a[0], family_a[1], family_b[0], family_free[0]
    )
    ridge_mask = jnp.concatenate([jnp.ones(10), jnp.zeros(2)])
    ridge = 0.05

    preconditioner = BlockEigenPreconditioner(families, permutation)
    ctx = SolverContext(
        lm_state=LMState(damping=jnp.asarray(1e-3), ridge=jnp.asarray(ridge))
    )
    v = jax.random.normal(jax.random.key(1), (12,))

    selection = jnp.eye(12)[permutation]
    shifted = dense_permuted + jnp.diag(ridge_mask * ridge + damping)
    expected = jnp.linalg.solve(selection.T @ shifted @ selection, v)

    result = preconditioner.apply(v, jnp.asarray(damping), ctx)
    np.testing.assert_allclose(result, expected, rtol=2e-4, atol=2e-5)


def test_constructor_rejects_mismatched_layouts():
    with pytest.raises(ValueError, match="permutation"):
        BlockEigenPreconditioner([(jnp.eye(2)[None], 1.0)], jnp.zeros(2))
    with pytest.raises(ValueError, match="groups, size, size"):
        BlockEigenPreconditioner([(jnp.eye(2), 1.0)], jnp.arange(2))
    with pytest.raises(ValueError, match="cover"):
        BlockEigenPreconditioner([(jnp.eye(2)[None], 1.0)], jnp.arange(3))


REPEATS = 2
BLOCK = 4
N_M = REPEATS * BLOCK
N_F = 2
P_DIM = N_M + N_F
M_RESID = 14
RIDGE = 1e-3
SOLVE_OPTIONS = dict(max_steps=60, gtol=1e-5, xtol=1e-7)


def build_problem():
    root = jax.random.normal(jax.random.key(2), (BLOCK, BLOCK + 2))
    K = root @ root.T + 0.5 * jnp.eye(BLOCK)
    F = jnp.linalg.cholesky(K, upper=True)
    metric = RepeatedFactorMetric(F, repeats=REPEATS)
    A = jax.random.normal(jax.random.key(3), (M_RESID, P_DIM)) / jnp.sqrt(P_DIM)
    target = jax.random.normal(jax.random.key(4), (M_RESID,))

    def residual(x, args, p):
        return A @ x - p["scale"] * target

    F_bar = jsp_linalg.block_diag(F, F, jnp.eye(N_F))
    J_whitened = jnp.linalg.solve(F_bar.T, A.T).T
    G = J_whitened.T @ J_whitened
    return metric, residual, G


def exact_preconditioner(G):
    # The exact whitened normal blocks -- one metric-block group, one free
    # group -- so CG converges in a handful of iterations.
    return BlockEigenPreconditioner(
        [(G[:N_M, :N_M][None], 1.0), (G[N_M:, N_M:][None], 0.0)],
        jnp.arange(P_DIM),
    )


def scrambled_preconditioner():
    # Same treedef as the exact one (one group of N_M, one of N_F), useless
    # values: a plain diagonal that ignores the whitened operator's structure.
    return BlockEigenPreconditioner(
        [(100.0 * jnp.eye(N_M)[None], 1.0), (100.0 * jnp.eye(N_F)[None], 0.0)],
        jnp.arange(P_DIM),
    )


def cholesky_reference(metric, residual):
    return RidgeLevenbergMarquardt(
        residual, metric=metric, ridge=RIDGE, linear_solver=Cholesky()
    )


def test_cg_solve_matches_cholesky():
    metric, residual, G = build_problem()
    p = {"scale": jnp.asarray(1.0)}
    x0 = jnp.zeros(P_DIM)
    args = {"data": jnp.asarray(1.0)}

    reference = cholesky_reference(metric, residual).solve(
        x0, args, p=p, **SOLVE_OPTIONS
    )
    assert int(reference.status) == int(LMStatus.CONVERGED)

    cg_solver = RidgeLevenbergMarquardt(
        residual,
        metric=metric,
        ridge=RIDGE,
        linear_solver=CG(exact_preconditioner(G), tol=1e-7, maxiter=200),
    )
    result = cg_solver.solve(x0, args, p=p, **SOLVE_OPTIONS)
    assert int(result.status) == int(LMStatus.CONVERGED)
    # Both solves stop on the same gtol, and a ridge-scaled stopping rule
    # leaves x-slack ~ gtol / ridge (1e-2 here), so the agreement of two
    # independently converged solves is a measured property rather than a
    # CG-tolerance bound. Kept an order below that slack: tight enough to
    # catch a wrong preconditioner (which breaks convergence outright).
    np.testing.assert_allclose(result.x, reference.x, rtol=2e-3, atol=2e-4)


def test_callback_refresh_reaches_the_next_inner_solve():
    # A callback-constructed instance replaces the carried one and drives the
    # very next CG solve: with a starved inner budget, a scrambled
    # preconditioner cannot reach gtol within the step cap, while refreshing
    # to the exact one at step 1 converges almost as fast as exact-from-start
    # -- and the swap is not a problem change, so convergence fires normally.
    metric, residual, G = build_problem()
    p = {"scale": jnp.asarray(1.0)}
    x0 = jnp.zeros(P_DIM)

    def build(callback=None):
        solver = RidgeLevenbergMarquardt(
            residual,
            metric=metric,
            ridge=RIDGE,
            linear_solver=CG(scrambled_preconditioner(), tol=0.0, maxiter=3),
        )
        return solver.solve(x0, p=p, max_steps=60, gtol=1e-5, callback=callback)

    def refresh(ctx):
        fresh = jax.lax.cond(
            ctx.step == 1,
            lambda: exact_preconditioner(G),
            lambda: ctx.lm_state.preconditioner,
        )
        return LMAction(
            lm_state=dataclasses.replace(ctx.lm_state, preconditioner=fresh)
        )

    stale = build()
    refreshed = build(refresh)
    assert int(stale.status) == int(LMStatus.MAX_STEPS)
    assert int(refreshed.status) == int(LMStatus.CONVERGED)
    assert int(refreshed.steps) <= 12
    # The carried instance in the result is the refreshed one.
    np.testing.assert_allclose(
        refreshed.lm_state.preconditioner.eigenvalues[0],
        exact_preconditioner(G).eigenvalues[0],
        rtol=1e-6,
    )


@pytest.mark.parametrize("ad_mode", ["explicit", "inherit", "inherit_knobs"])
def test_ad_tangent_matches_cholesky(ad_mode):
    # The zero-damping AD role applies the same typed preconditioner. With
    # ad_solver=None the forward CARRIED instance is inherited at the
    # AD-default tolerance and budget; CG(None, ...) inherits it while
    # pinning the knobs.
    metric, residual, G = build_problem()
    p = {"scale": jnp.asarray(1.0)}
    p_dot = {"scale": jnp.asarray(1.0)}
    x0 = jnp.zeros(P_DIM)
    args = {"data": jnp.asarray(1.0)}

    def solved_x(solver):
        def run(p_value):
            return solver.solve(x0, args, p=p_value, **SOLVE_OPTIONS).x

        return jax.jvp(run, (p,), (p_dot,))[1]

    forward = CG(exact_preconditioner(G), tol=1e-7, maxiter=200)
    ad_solver = {
        "explicit": CG(exact_preconditioner(G), tol=1e-7, maxiter=200),
        "inherit": None,
        "inherit_knobs": CG(None, tol=1e-7, maxiter=200),
    }[ad_mode]
    cg_solver = RidgeLevenbergMarquardt(
        residual, metric=metric, ridge=RIDGE, linear_solver=forward, ad_solver=ad_solver
    )
    if ad_mode == "inherit":
        assert cg_solver._ad_preconditioner_source == "carried"
        assert cg_solver.ad_solver_tol is None
    if ad_mode == "inherit_knobs":
        assert cg_solver._ad_preconditioner_source == "carried"
        assert cg_solver.ad_solver_tol == 1e-7
        assert cg_solver.ad_solver_maxiter == 200

    reference = solved_x(cholesky_reference(metric, residual))
    np.testing.assert_allclose(solved_x(cg_solver), reference, rtol=1e-3, atol=1e-4)
