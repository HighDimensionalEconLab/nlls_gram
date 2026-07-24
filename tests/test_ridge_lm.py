import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    RidgeLevenbergMarquardt,
    RidgeLMState,
    identity_penalty,
    repeated_dense_penalty,
)

# Analytic linear-Gaussian testbed: r(theta) = A theta - b with m < p, penalty
# M0 = blockdiag(K, K, 0_pad). The fixed-ridge minimizer and the constrained
# minimum-seminorm solution are computed in float64 numpy as references.
RNG = np.random.default_rng(7)
M_RESID, BLOCK, REPEATS, PAD = 4, 4, 2, 2
P_DIM = REPEATS * BLOCK + PAD
A_NP = RNG.normal(size=(M_RESID, P_DIM))
ROOT = RNG.normal(size=(BLOCK, BLOCK + 2))
K_NP = ROOT @ ROOT.T + 0.5 * np.eye(BLOCK)
M0_NP = np.zeros((P_DIM, P_DIM))
for j in range(REPEATS):
    M0_NP[j * BLOCK : (j + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = K_NP
B_NP = RNG.normal(size=M_RESID)

A = jnp.asarray(A_NP, dtype=jnp.float32)
B = jnp.asarray(B_NP, dtype=jnp.float32)
K = jnp.asarray(K_NP, dtype=jnp.float32)


def linear_residual(theta):
    return A @ theta - B


def make_penalty():
    return repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)


def ridge_minimizer(ridge):
    return np.linalg.solve(A_NP.T @ A_NP + ridge * M0_NP, A_NP.T @ B_NP)


def min_seminorm_solution():
    # KKT system for argmin x' M0 x s.t. A x = b; nonsingular because A has
    # full row rank and ker A ∩ ker M0 = {0} (generic random data).
    kkt = np.block([[M0_NP, A_NP.T], [A_NP, np.zeros((M_RESID, M_RESID))]])
    rhs = np.concatenate([np.zeros(P_DIM), B_NP])
    return np.linalg.solve(kkt, rhs)[:P_DIM]


def test_fixed_ridge_reaches_the_ridge_minimizer():
    ridge = 1e-2
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=ridge
    )
    result = solver.solve(jnp.zeros(P_DIM), max_steps=200, gtol=1e-5)
    assert int(result.status) == 1
    np.testing.assert_allclose(
        np.asarray(result.x), ridge_minimizer(ridge), rtol=2e-3, atol=2e-4
    )
    # info.loss is the ridge objective; resid_loss and penalty_value decompose it.
    np.testing.assert_allclose(
        float(result.info.loss),
        float(result.info.resid_loss) + ridge * float(result.info.penalty_value),
        rtol=1e-5,
    )


def test_ridge_bias_is_first_order_in_ridge():
    x_dagger = min_seminorm_solution()
    errors = {}
    for ridge in (1e-2, 1e-3):
        solver = RidgeLevenbergMarquardt(
            linear_residual, penalty=make_penalty(), ridge=ridge
        )
        result = solver.solve(jnp.zeros(P_DIM), max_steps=300, gtol=1e-5)
        assert int(result.status) == 1
        errors[ridge] = np.linalg.norm(np.asarray(result.x) - x_dagger)
        # The solver's error matches the analytic fixed-ridge bias.
        analytic_bias = np.linalg.norm(ridge_minimizer(ridge) - x_dagger)
        assert errors[ridge] < 2.0 * analytic_bias + 1e-3
    # O(ridge) bias: a 10x smaller ridge cuts the error by well over 3x.
    assert errors[1e-3] < errors[1e-2] / 3.0


def test_zero_padded_scalars_are_not_shrunk():
    # The tail coordinates carry no penalty rows, so the solver's tail solves
    # A_tail' r = 0 (no ridge shrinkage toward zero) and matches the
    # constrained solution's nonzero tail at O(ridge).
    ridge = 1e-3
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=ridge
    )
    result = solver.solve(jnp.zeros(P_DIM), max_steps=300, gtol=1e-5)
    assert int(result.status) == 1
    x_dagger = min_seminorm_solution()
    assert np.linalg.norm(x_dagger[-PAD:]) > 0.05
    resid = np.asarray(A_NP @ np.asarray(result.x, np.float64) - B_NP)
    tail_gradient = A_NP[:, -PAD:].T @ resid
    assert np.linalg.norm(tail_gradient) < 5e-5
    np.testing.assert_allclose(
        np.asarray(result.x)[-PAD:], x_dagger[-PAD:], rtol=0.05, atol=5e-3
    )


def nonlinear_residual(theta):
    head = theta[:3]
    return jnp.array(
        [
            head[0] * head[1] - 0.6,
            head[1] + head[2] ** 2 - 1.1,
            jnp.sum(theta**2) - 2.0,
        ]
    )


def test_objective_monotonicity_and_acceptance():
    penalty = identity_penalty(5)
    solver = RidgeLevenbergMarquardt(
        nonlinear_residual, penalty=penalty, ridge=1e-2, init_damping=1e-1
    )
    x = jnp.asarray([0.9, 0.8, 0.4, 0.1, -0.2], dtype=jnp.float32)
    lm_state = solver.init(x)
    previous_loss = None
    for _ in range(25):
        x_old = x
        x, lm_state, info = solver.update(x, lm_state)
        np.testing.assert_allclose(
            float(info.loss),
            min(float(info.loss_old), float(info.loss_candidate)),
            rtol=1e-6,
        )
        if bool(info.accepted):
            assert float(info.loss_candidate) < float(info.loss_old)
        else:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(x_old))
            assert float(info.loss) == float(info.loss_old)
        if previous_loss is not None:
            assert float(info.loss_old) <= previous_loss * (1 + 1e-6)
        previous_loss = float(info.loss)
    # The objective floor is ridge * q(x*), not zero; the equations themselves
    # are solved to float32 levels.
    assert float(info.resid_loss) < 1e-4


def test_grad_norm_is_the_ridge_stationarity_residual():
    ridge = 3e-2
    penalty = identity_penalty(5)
    solver = RidgeLevenbergMarquardt(nonlinear_residual, penalty=penalty, ridge=ridge)
    x = jnp.asarray([0.9, 0.8, 0.4, 0.1, -0.2], dtype=jnp.float32)
    lm_state = solver.init(x)
    _, _, info = solver.update(x, lm_state)
    jacobian = jax.jacobian(nonlinear_residual)(x)
    half_gradient = jacobian.T @ nonlinear_residual(x) + ridge * x
    np.testing.assert_allclose(
        float(info.grad_norm), float(jnp.linalg.norm(half_gradient)), rtol=1e-5
    )


def test_converged_solution_is_ridge_stationary():
    ridge = 1e-2
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=ridge
    )
    result = solver.solve(jnp.zeros(P_DIM), max_steps=200, gtol=1e-5)
    x = np.asarray(result.x, np.float64)
    stationarity = A_NP.T @ (A_NP @ x - B_NP) + ridge * (M0_NP @ x)
    assert np.linalg.norm(stationarity) < 1e-4


def test_atol_alone_is_rejected():
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )
    with pytest.raises(ValueError, match="conjunctive"):
        solver.solve(jnp.zeros(P_DIM), atol=1e-8)


def two_phase_residual(x):
    # Depends on x[0] only: after phase 1 (r ~ 0 at x[0] ~ 1) the entire
    # second coordinate is selection, resolved only by phase 2.
    return x[:1] - 1.0


def test_penalty_grad_norm_reports_the_gtol_calibration_scale():
    # info.penalty_grad_norm is ||L'L x|| at the pre-step x -- the scale the
    # gtol calibration recipe reads (gtol ~ 1e-3 * ridge * penalty_grad_norm).
    # Identity penalty: ||L'L x|| = ||x||; the init sentinel is +inf.
    ridge = 1e-3
    solver = RidgeLevenbergMarquardt(
        two_phase_residual, penalty=identity_penalty(2), ridge=ridge
    )
    x = jnp.array([0.0, 3.0])
    lm_state = solver.init(x)
    initial_info = solver._initial_info(x, lm_state, None, None)
    assert not jnp.isfinite(initial_info.penalty_grad_norm)
    _, _, info = solver.update(x, lm_state)
    np.testing.assert_allclose(
        float(info.penalty_grad_norm), float(jnp.linalg.norm(x)), rtol=1e-6
    )
    # The calibrated gtol resolves phase 2: from (0, 3) the residual floors
    # at x = (1, 3) -- a pure-residual test would stop 3 off in the second
    # coordinate -- while gtol ~ 1e-3 * ridge * ||x*|| holds the solve until
    # the iterate slides to the ridge minimizer (1/(1+ridge), 0).
    gtol = 1e-3 * ridge * 1.0
    result = solver.solve(x, max_steps=300, gtol=gtol, atol=1e-2)
    assert int(result.status) == 1
    np.testing.assert_allclose(
        np.asarray(result.x), [1.0 / (1.0 + ridge), 0.0], atol=2e-3
    )
    assert abs(float(result.x[1])) < 1e-3


def test_atol_is_conjunctive_an_interpolating_start_does_not_stop():
    # x0 interpolates (r = 0) but carries excess seminorm; a residual-only
    # atol would stop at step 0. The conjunctive contract keeps optimizing
    # until gtol fires, strictly reducing the penalty.
    x_particular = jnp.asarray(
        np.linalg.lstsq(A_NP, B_NP, rcond=None)[0], dtype=jnp.float32
    )
    null_basis = jnp.asarray(np.linalg.svd(A_NP)[2][M_RESID:].T, dtype=jnp.float32)
    x0 = x_particular + null_basis @ jnp.ones(P_DIM - M_RESID)
    assert float(jnp.linalg.norm(A @ x0 - B)) < 1e-4
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )
    initial_penalty = float(solver._quadratic(x0))
    result = solver.solve(x0, max_steps=300, gtol=1e-5, atol=1e-3)
    assert int(result.status) == 1
    assert int(result.steps) > 0
    assert float(result.info.penalty_value) < 0.9 * initial_penalty
    assert float(jnp.sqrt(result.info.resid_loss)) <= 1e-3


def test_ridge_none_resolves_to_dtype_default():
    solver = RidgeLevenbergMarquardt(linear_residual, penalty=make_penalty())
    lm_state = solver.init(jnp.zeros(P_DIM))
    expected = np.sqrt(np.finfo(np.float32).eps)
    np.testing.assert_allclose(float(lm_state.ridge), expected, rtol=1e-6)


def test_manual_state_without_ridge_raises():
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3, cache_jacobian=False
    )
    bad = RidgeLMState(jnp.asarray(1e-3), None)
    with pytest.raises(ValueError, match="ridge"):
        solver.update(jnp.zeros(P_DIM), bad)
    with pytest.raises(ValueError, match="ridge"):
        solver.solve(jnp.zeros(P_DIM), lm_state=bad, gtol=1e-5)


def test_constructor_validation():
    penalty = make_penalty()
    with pytest.raises(TypeError, match="RidgePenalty"):
        RidgeLevenbergMarquardt(linear_residual, penalty=object())
    with pytest.raises(ValueError, match="strictly positive"):
        RidgeLevenbergMarquardt(linear_residual, penalty=penalty, ridge=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        RidgeLevenbergMarquardt(linear_residual, penalty=penalty, ridge=-1e-3)
    # String solver names are gone: the typed configs are the only spelling.
    with pytest.raises(TypeError, match="solver config"):
        RidgeLevenbergMarquardt(
            linear_residual, penalty=penalty, linear_solver="cholesky"
        )
    with pytest.raises(TypeError, match="ad_solver must be None"):
        RidgeLevenbergMarquardt(linear_residual, penalty=penalty, ad_solver="auto")
    # Each config validates its own fields at construction.
    with pytest.raises(ValueError, match="maxiter"):
        CG(tol=0.0)
    with pytest.raises(TypeError, match="callable"):
        CG(preconditioner=object())
    with pytest.raises(NotImplementedError, match="penalty_factory"):
        RidgeLevenbergMarquardt(
            linear_residual, penalty=penalty, penalty_factory=object()
        )


def test_caller_state_with_nonpositive_ridge_rejected():
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), cache_jacobian=False
    )
    bad = RidgeLMState(jnp.asarray(1e-3), jnp.asarray(0.0))
    with pytest.raises(ValueError, match="strictly positive"):
        solver.solve(jnp.zeros(P_DIM), lm_state=bad, gtol=1e-5)


def test_update_with_ridge_replaced_by_hand_changes_the_subproblem():
    # dataclasses.replace on the state is the documented way to anneal ridge
    # between manual updates; the next step must honor the new value.
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-1
    )
    x0 = jnp.ones(P_DIM)
    lm_state = solver.init(x0)
    _, _, info_high = solver.update(x0, lm_state)
    lowered = dataclasses.replace(lm_state, ridge=jnp.asarray(1e-4))
    _, _, info_low = solver.update(x0, lowered)
    assert float(info_low.ridge) == pytest.approx(1e-4)
    assert float(info_low.loss_old) < float(info_high.loss_old)
