import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from nlls_gram import (
    IdentityMetric,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    SolverContext,
)

REPEATS = 3
BLOCK = 5
N_M = REPEATS * BLOCK
CTX = SolverContext()


def make_factor(key, size):
    root = jax.random.normal(key, (size, size + 2))
    K = root @ root.T + 0.5 * jnp.eye(size)
    return jnp.linalg.cholesky(K, upper=True), K


def test_repeated_factor_metric_contract():
    # The solver contract in one place: batched shapes (vector and
    # leading-axis matrix), the three factor ops against the dense blockdiag
    # F_bar, and norm consistency with factor_apply -- the objective values
    # the solver compares are built from both forms.
    F, K = make_factor(jax.random.key(0), BLOCK)
    metric = RepeatedFactorMetric(F, repeats=REPEATS)
    assert metric.size == N_M
    F_bar = jsp_linalg.block_diag(*([F] * REPEATS))
    W = jsp_linalg.block_diag(*([K] * REPEATS))
    v = jax.random.normal(jax.random.key(1), (N_M,))
    matrix = jax.random.normal(jax.random.key(2), (N_M, 3))

    applied = metric.factor_apply(v, CTX)
    assert applied.shape == (N_M,)
    np.testing.assert_allclose(applied, F_bar @ v, rtol=1e-5, atol=1e-6)
    batched = metric.factor_apply(matrix, CTX)
    assert batched.shape == matrix.shape
    np.testing.assert_allclose(batched, F_bar @ matrix, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(
        metric.factor_solve(applied, CTX), v, rtol=2e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        metric.factor_solve_transpose(matrix, CTX),
        np.linalg.solve(np.asarray(F_bar, np.float64).T, np.asarray(matrix)),
        rtol=2e-4,
        atol=1e-5,
    )

    # norm(v) == ||factor_apply(v)||_2 == sqrt(v' W v).
    np.testing.assert_allclose(
        float(metric.norm(v, CTX)), float(jnp.linalg.norm(applied)), rtol=1e-6
    )
    np.testing.assert_allclose(
        float(metric.norm(v, CTX)) ** 2, float(v @ (W @ v)), rtol=1e-4
    )


def test_identity_metric_is_plain_ridge():
    metric = IdentityMetric(6)
    assert metric.size == 6
    v = jax.random.normal(jax.random.key(3), (6,))
    matrix = jax.random.normal(jax.random.key(4), (6, 2))
    for op in (
        metric.factor_apply,
        metric.factor_solve,
        metric.factor_solve_transpose,
    ):
        np.testing.assert_array_equal(op(v, CTX), v)
        np.testing.assert_array_equal(op(matrix, CTX), matrix)
    np.testing.assert_allclose(
        float(metric.norm(v, CTX)), float(jnp.linalg.norm(v)), rtol=1e-6
    )


def test_metrics_hash_by_identity():
    # eq=False frozen dataclasses: equal-config metrics are distinct compile
    # keys, one constructed metric equals itself.
    F, _ = make_factor(jax.random.key(5), BLOCK)
    a = RepeatedFactorMetric(F, repeats=2)
    b = RepeatedFactorMetric(F, repeats=2)
    assert a == a and hash(a) == hash(a)
    assert a != b
    assert IdentityMetric(3) != IdentityMetric(3)


def test_constructor_and_input_validation():
    with pytest.raises(ValueError, match="square"):
        RepeatedFactorMetric(jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="repeats"):
        RepeatedFactorMetric(jnp.eye(2), repeats=0)
    with pytest.raises(TypeError, match="floating"):
        RepeatedFactorMetric(jnp.eye(2, dtype=jnp.complex64))
    with pytest.raises(ValueError, match="positive"):
        IdentityMetric(0)
    with pytest.raises(ValueError, match="leading size"):
        RepeatedFactorMetric(jnp.eye(2)).factor_apply(jnp.zeros(3), CTX)
    with pytest.raises(ValueError, match="vector or matrix"):
        IdentityMetric(3).factor_apply(jnp.zeros((3, 2, 2)), CTX)
    with pytest.raises(ValueError, match="vector"):
        IdentityMetric(3).norm(jnp.zeros((3, 2)), CTX)


def test_solver_passes_live_context_to_the_factor_ops():
    # Every factor op receives a SolverContext carrying the flat iterate and
    # the live LMState (recorded at trace time -- the fields are
    # tracers, their presence and shapes are static).
    seen = []

    class Probe(IdentityMetric):
        def factor_apply(self, v, ctx):
            seen.append(
                ctx.x is not None
                and ctx.x.shape == (self.size + 1,)
                and ctx.lm_state is not None
            )
            return v

        factor_solve = factor_apply
        factor_solve_transpose = factor_apply

    A = jnp.asarray(np.random.default_rng(0).normal(size=(2, 4)), jnp.float32)

    def residual(theta):
        return A @ theta - 1.0

    # One free coordinate, so the extension wrappers slice before the ops.
    solver = RidgeLevenbergMarquardt(residual, metric=Probe(3), ridge=1e-3)
    x0 = jnp.zeros(4)
    solver.update(x0, solver.init(x0))
    assert seen and all(seen)
