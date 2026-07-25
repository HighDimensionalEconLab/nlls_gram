# Float32 coverage for BlockEigenPreconditioner (plus the precision-neutral
# contract tests: hashing, errors, retrace behavior). The float64 primary
# coverage lives in test_float64_subprocess.py, matching the production
# default.
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
    LMSolveAction,
    LMStatus,
    MetricContext,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    RidgeLMState,
    block_eigen_state,
)


def spd_blocks(key, groups, size):
    root = jax.random.normal(key, (groups, size, size))
    return jnp.einsum("gik,gjk->gij", root, root) + 0.5 * jnp.eye(size)


def packed_state(key):
    # Two ridge-flagged families (2 groups of 3, 1 group of 4) and one
    # damping-only free family (1 group of 2), under a nontrivial
    # permutation of the 12 coordinates.
    keys = jax.random.split(key, 3)
    family_a = spd_blocks(keys[0], 2, 3)
    family_b = spd_blocks(keys[1], 1, 4)
    family_free = spd_blocks(keys[2], 1, 2)
    permutation = jnp.asarray(np.random.default_rng(0).permutation(12))
    state = block_eigen_state(
        [(family_a, 1.0), (family_b, 1.0), (family_free, 0.0)], permutation
    )
    dense_permuted = jsp_linalg.block_diag(
        family_a[0], family_a[1], family_b[0], family_free[0]
    )
    ridge_mask = jnp.concatenate([jnp.ones(10), jnp.zeros(2)])
    return state, dense_permuted, ridge_mask, permutation


@pytest.mark.parametrize("damping", [0.0, 0.37])
def test_apply_matches_dense_inverse(damping):
    state, dense_permuted, ridge_mask, permutation = packed_state(jax.random.key(0))
    ridge = 0.05
    ctx = MetricContext(
        lm_state=RidgeLMState(damping=jnp.asarray(1e-3), ridge=jnp.asarray(ridge)),
        args={"preconditioner": state},
    )
    preconditioner = BlockEigenPreconditioner()
    v = jax.random.normal(jax.random.key(1), (12,))

    selection = jnp.eye(12)[permutation]
    shifted = dense_permuted + jnp.diag(ridge_mask * ridge + damping)
    dense_original = selection.T @ shifted @ selection
    expected = jnp.linalg.solve(dense_original, v)

    result = preconditioner.apply(v, jnp.asarray(damping), ctx)
    np.testing.assert_allclose(result, expected, rtol=2e-4, atol=2e-5)


def test_value_hashing_and_config_equality():
    assert BlockEigenPreconditioner() == BlockEigenPreconditioner("preconditioner")
    assert hash(BlockEigenPreconditioner("k")) == hash(BlockEigenPreconditioner("k"))
    assert BlockEigenPreconditioner("a") != BlockEigenPreconditioner("b")
    assert CG(BlockEigenPreconditioner(), maxiter=64) == CG(
        BlockEigenPreconditioner(), maxiter=64
    )


def test_missing_state_and_bad_builder_inputs():
    preconditioner = BlockEigenPreconditioner()
    ctx = MetricContext(
        lm_state=RidgeLMState(damping=jnp.asarray(1e-3), ridge=jnp.asarray(1e-3)),
        args={"other": 1.0},
    )
    with pytest.raises(ValueError, match="ctx.args"):
        preconditioner.apply(jnp.ones(3), jnp.asarray(0.1), ctx)
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


def exact_state(G):
    # The exact whitened normal blocks: one metric-block group, one free
    # group -- close enough to the operator that CG converges in a handful
    # of iterations.
    return block_eigen_state(
        [(G[:N_M, :N_M][None], 1.0), (G[N_M:, N_M:][None], 0.0)],
        jnp.arange(P_DIM),
    )


def test_cg_solve_matches_cholesky_with_callback_rebuild():
    metric, residual, G = build_problem()
    p = {"scale": jnp.asarray(1.0)}
    x0 = jnp.zeros(P_DIM)
    args = {"data": jnp.asarray(1.0), "preconditioner": exact_state(G)}
    solve_options = dict(max_steps=60, gtol=1e-5, xtol=1e-7)

    reference = RidgeLevenbergMarquardt(
        residual, metric=metric, ridge=RIDGE, linear_solver=Cholesky()
    ).solve(x0, args, p=p, **solve_options)
    assert int(reference.status) == int(LMStatus.CONVERGED)

    def rebuild_callback(ctx):
        # A real state swap mid-solve: scaled blocks at step 2 (values
        # change -> args replacement path exercised, convergence suppressed
        # for that step only).
        fresh = block_eigen_state(
            [(1.25 * G[:N_M, :N_M][None], 1.0), (1.25 * G[N_M:, N_M:][None], 0.0)],
            jnp.arange(P_DIM),
        )
        swap = ctx.step == 2
        new_state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(swap, new, old),
            ctx.args["preconditioner"],
            fresh,
        )
        return LMSolveAction(args={**ctx.args, "preconditioner": new_state})

    cg_solver = RidgeLevenbergMarquardt(
        residual,
        metric=metric,
        ridge=RIDGE,
        linear_solver=CG(BlockEigenPreconditioner(), tol=1e-7, maxiter=200),
    )
    result = cg_solver.solve(x0, args, p=p, callback=rebuild_callback, **solve_options)
    assert int(result.status) == int(LMStatus.CONVERGED)
    np.testing.assert_allclose(result.x, reference.x, rtol=2e-4, atol=2e-5)


def test_ad_role_zero_damping_matches_cholesky_tangent():
    # ad_solver is the EXPLICIT CG config: ad_solver=None resolves to the CG
    # family but with no preconditioner, which would leave the typed apply
    # untested in its zero-damping role.
    metric, residual, G = build_problem()
    p = {"scale": jnp.asarray(1.0)}
    p_dot = {"scale": jnp.asarray(1.0)}
    x0 = jnp.zeros(P_DIM)
    args = {"data": jnp.asarray(1.0), "preconditioner": exact_state(G)}
    solve_options = dict(max_steps=60, gtol=1e-5, xtol=1e-7)

    def solved_x(solver):
        def run(p_value):
            return solver.solve(x0, args, p=p_value, **solve_options).x

        return jax.jvp(run, (p,), (p_dot,))[1]

    reference = solved_x(
        RidgeLevenbergMarquardt(
            residual, metric=metric, ridge=RIDGE, linear_solver=Cholesky()
        )
    )
    cg_tangent = solved_x(
        RidgeLevenbergMarquardt(
            residual,
            metric=metric,
            ridge=RIDGE,
            linear_solver=CG(BlockEigenPreconditioner(), tol=1e-7, maxiter=200),
            ad_solver=CG(BlockEigenPreconditioner(), tol=1e-7, maxiter=200),
        )
    )
    np.testing.assert_allclose(cg_tangent, reference, rtol=1e-3, atol=1e-4)


def test_state_dataclass_survives_jit_boundary():
    # The instance is a static config field: equal instances must not
    # retrace when only the args-carried state values change.
    state, _, _, _ = packed_state(jax.random.key(5))
    preconditioner = BlockEigenPreconditioner()
    traces = []

    @jax.jit
    def apply(v, damping, ridge, state):
        traces.append(None)
        ctx = MetricContext(
            lm_state=RidgeLMState(damping=damping, ridge=ridge),
            args={"preconditioner": state},
        )
        return preconditioner.apply(v, damping, ctx)

    v = jax.random.normal(jax.random.key(6), (12,))
    first = apply(v, jnp.asarray(0.1), jnp.asarray(0.01), state)
    doubled = {
        **state,
        "families": tuple(
            {**family, "eigenvalues": 2.0 * family["eigenvalues"]}
            for family in state["families"]
        ),
    }
    second = apply(v, jnp.asarray(0.1), jnp.asarray(0.01), doubled)
    assert len(traces) == 1
    assert not jnp.allclose(first, second)


def test_dataclass_replace_keeps_identity():
    config = CG(BlockEigenPreconditioner(), maxiter=32)
    replaced = dataclasses.replace(config, maxiter=64)
    assert replaced.preconditioner == config.preconditioner
