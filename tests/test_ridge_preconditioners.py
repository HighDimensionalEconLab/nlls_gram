# Float32 coverage for BlockEigenPreconditioner. The float64 primary coverage
# lives in test_float64_subprocess.py, matching the production default.
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from nlls_gram import (
    CG,
    BlockEigenPreconditioner,
    Cholesky,
    LMState,
    LMStatus,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    SolverContext,
    block_eigen_state,
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

    preconditioner = BlockEigenPreconditioner(lambda theta, ctx: families, permutation)
    ctx = SolverContext(
        lm_state=LMState(damping=jnp.asarray(1e-3), ridge=jnp.asarray(ridge)),
        preconditioner_state=preconditioner.prepare(jnp.zeros(12), None),
    )
    v = jax.random.normal(jax.random.key(1), (12,))

    selection = jnp.eye(12)[permutation]
    shifted = dense_permuted + jnp.diag(ridge_mask * ridge + damping)
    expected = jnp.linalg.solve(selection.T @ shifted @ selection, v)

    result = preconditioner.apply(v, jnp.asarray(damping), ctx)
    np.testing.assert_allclose(result, expected, rtol=2e-4, atol=2e-5)


def test_block_eigen_state_rejects_mismatched_layouts():
    with pytest.raises(ValueError, match="permutation"):
        block_eigen_state([(jnp.eye(2)[None], 1.0)], jnp.zeros(2))
    with pytest.raises(ValueError, match="groups, size, size"):
        block_eigen_state([(jnp.eye(2), 1.0)], jnp.arange(2))
    with pytest.raises(ValueError, match="cover"):
        block_eigen_state([(jnp.eye(2)[None], 1.0)], jnp.arange(3))


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
    # group -- so CG converges in a handful of iterations. The residual is
    # linear, so the operator does not move with the iterate and blocks_fn
    # ignores theta.
    def blocks_fn(theta, ctx):
        return [(G[:N_M, :N_M][None], 1.0), (G[N_M:, N_M:][None], 0.0)]

    return BlockEigenPreconditioner(blocks_fn, jnp.arange(P_DIM))


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


def test_prepared_state_is_rebuilt_from_the_live_iterate():
    # blocks_fn sees the whitened iterate: a counter proves prepare runs
    # inside the loop, and the carried state tracks it.
    metric, residual, G = build_problem()
    seen = []

    def blocks_fn(theta, ctx):
        seen.append(theta)
        return [(G[:N_M, :N_M][None], 1.0), (G[N_M:, N_M:][None], 0.0)]

    solver = RidgeLevenbergMarquardt(
        residual,
        metric=metric,
        ridge=RIDGE,
        linear_solver=CG(
            BlockEigenPreconditioner(blocks_fn, jnp.arange(P_DIM)),
            tol=1e-7,
            maxiter=200,
        ),
    )
    x0 = jnp.zeros(P_DIM)
    state = solver.init(x0, {"data": jnp.asarray(1.0)}, p={"scale": jnp.asarray(1.0)})
    # init builds the state at x0 and marks it valid there.
    assert state.precond is not None
    assert bool(state.precond_valid)
    assert len(seen) == 1
    x1, state1, info = solver.update(
        x0, state, {"data": jnp.asarray(1.0)}, {"scale": jnp.asarray(1.0)}
    )
    # An accepted step moved x, so the carried state is marked for rebuild.
    assert bool(info.accepted)
    assert not bool(state1.precond_valid)


@pytest.mark.parametrize("explicit_ad_solver", [True, False])
def test_ad_tangent_matches_cholesky(explicit_ad_solver):
    # The zero-damping AD role applies the same typed preconditioner. With
    # ad_solver=None the forward config's preconditioner is inherited, at the
    # AD-default tolerance and budget.
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
    ad_solver = (
        CG(exact_preconditioner(G), tol=1e-7, maxiter=200)
        if explicit_ad_solver
        else None
    )
    cg_solver = RidgeLevenbergMarquardt(
        residual, metric=metric, ridge=RIDGE, linear_solver=forward, ad_solver=ad_solver
    )
    if not explicit_ad_solver:
        assert cg_solver.ad_solver_preconditioner is forward.preconditioner
        assert cg_solver.ad_solver_tol is None

    reference = solved_x(cholesky_reference(metric, residual))
    np.testing.assert_allclose(solved_x(cg_solver), reference, rtol=1e-3, atol=1e-4)
