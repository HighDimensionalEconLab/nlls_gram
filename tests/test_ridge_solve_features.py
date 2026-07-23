import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    LMSolveAction,
    LMStatus,
    MultiStart,
    RidgeLevenbergMarquardt,
    identity_right_preconditioner,
    repeated_dense_penalty,
    ridge_continuation,
)

IDENTITY_RIGHT = identity_right_preconditioner()

RNG = np.random.default_rng(11)
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


def linear_residual_with_args(theta, args):
    return args["A"] @ theta - args["b"]


def make_penalty():
    return repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)


def min_seminorm_solution():
    kkt = np.block([[M0_NP, A_NP.T], [A_NP, np.zeros((M_RESID, M_RESID))]])
    rhs = np.concatenate([np.zeros(P_DIM), B_NP])
    return np.linalg.solve(kkt, rhs)[:P_DIM]


def test_ridge_continuation_beats_any_single_moderate_ridge():
    x_dagger = min_seminorm_solution()
    fixed = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-2
    )
    fixed_result = fixed.solve(jnp.zeros(P_DIM), max_steps=300, gtol=1e-5)
    fixed_error = np.linalg.norm(np.asarray(fixed_result.x) - x_dagger)

    callback, user_state0 = ridge_continuation(ridge_floor=1e-6, decrease=0.1)
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-2
    )
    result = solver.solve(
        jnp.zeros(P_DIM),
        max_steps=500,
        gtol=2e-5,
        atol=1e-3,
        callback=callback,
        user_state=user_state0,
    )
    assert int(result.status) == 1
    continuation_error = np.linalg.norm(np.asarray(result.x) - x_dagger)
    assert continuation_error < 0.05 * fixed_error
    # The continuation actually annealed the ridge down to (near) the floor.
    assert float(result.lm_state.ridge) < 1e-4


def test_callback_ridge_change_suppresses_convergence():
    # A callback that shrinks ridge EVERY step keeps changing the problem, so
    # a gtol that would otherwise fire never stops the loop.
    def always_shrink(ctx):
        return LMSolveAction(
            lm_state=dataclasses.replace(
                ctx.lm_state, ridge=ctx.lm_state.ridge * 0.5
            )
        )

    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-2
    )
    result = solver.solve(
        jnp.zeros(P_DIM), max_steps=25, gtol=1e3, callback=always_shrink
    )
    assert int(result.status) == int(LMStatus.MAX_STEPS)
    np.testing.assert_allclose(
        float(result.lm_state.ridge), 1e-2 * 0.5**25, rtol=1e-4
    )
    # Without the callback the huge gtol converges immediately after one step.
    plain = solver.solve(jnp.zeros(P_DIM), max_steps=25, gtol=1e3)
    assert int(plain.status) == int(LMStatus.CONVERGED)


def test_callback_weak_typed_ridge_is_recast():
    # A hand-rolled callback assigning a Python float must not change the
    # while_loop carry aval (weak-type float64 scalar under x64 disabled is
    # a plain float32 weak array; the solver recasts to the carried dtype).
    def clumsy(ctx):
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, ridge=5e-3)
        )

    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-2
    )
    result = solver.solve(jnp.zeros(P_DIM), max_steps=50, gtol=1e-4, callback=clumsy)
    assert result.lm_state.ridge.dtype == jnp.float32
    np.testing.assert_allclose(float(result.lm_state.ridge), 5e-3, rtol=1e-6)


def test_save_steps_histories():
    args = {"A": A, "b": B}
    solver = RidgeLevenbergMarquardt(
        linear_residual_with_args, penalty=make_penalty(), ridge=1e-3
    )
    x0 = jnp.zeros(P_DIM)
    max_steps = 40
    result = solver.solve(
        x0, args, max_steps=max_steps, gtol=1e-5, save_steps=True
    )
    assert int(result.status) == 1
    steps = int(result.steps)
    assert result.x_history.shape == (max_steps + 1, P_DIM)
    np.testing.assert_array_equal(np.asarray(result.x_history[0]), np.asarray(x0))
    np.testing.assert_allclose(
        np.asarray(result.x_history[steps]), np.asarray(result.x), rtol=1e-6
    )
    # Rows beyond the step count are zero padding.
    if steps + 1 < max_steps + 1:
        np.testing.assert_array_equal(
            np.asarray(result.x_history[steps + 1 :]),
            np.zeros((max_steps - steps, P_DIM)),
        )
    np.testing.assert_array_equal(
        np.asarray(result.args_history["A"][steps]), np.asarray(A)
    )


def draw_perturbed(key, x, args):
    return x + 0.1 * jax.random.normal(key, x.shape, dtype=x.dtype), args


@pytest.mark.parametrize("parallel", [False, True])
def test_multi_start_recovers_from_a_poisoned_start(parallel):
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )
    # NaN start: attempt 0 fails NONFINITE, a drawn start succeeds.
    x0 = jnp.full(P_DIM, jnp.nan, dtype=jnp.float32)
    ms = MultiStart(
        key=jax.random.key(3), num_starts=4, draw=draw_nan_recovery, parallel=parallel
    )
    result = solver.solve(
        x0, max_steps=200, gtol=1e-5, max_steps_is_success=False, multi_start=ms
    )
    assert int(result.status) == int(LMStatus.CONVERGED)
    assert int(result.multi_start.attempt) > 0
    assert bool(result.multi_start.accepted)
    # Cold lanes inherit the caller-supplied ridge, never a constructor reset.
    np.testing.assert_allclose(float(result.lm_state.ridge), 1e-3, rtol=1e-6)


def draw_nan_recovery(key, x, args):
    fresh = 0.5 * jax.random.normal(key, x.shape, dtype=x.dtype)
    return fresh, args


def test_multi_start_ranking_uses_the_ridge_objective():
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )
    ms = MultiStart(
        key=jax.random.key(5), num_starts=3, draw=draw_perturbed, parallel=True
    )
    result = solver.solve(jnp.zeros(P_DIM), max_steps=200, gtol=1e-5, multi_start=ms)
    assert bool(result.multi_start.accepted)
    expected = float(result.info.resid_loss) + float(result.lm_state.ridge) * float(
        result.info.penalty_value
    )
    np.testing.assert_allclose(float(result.multi_start.loss), expected, rtol=1e-5)


def residual_with_aux(theta):
    value = A @ theta - B
    return value, {"resid_norm": jnp.linalg.norm(value)}


def test_has_aux():
    solver = RidgeLevenbergMarquardt(
        residual_with_aux, penalty=make_penalty(), ridge=1e-3, has_aux=True
    )
    result = solver.solve(jnp.zeros(P_DIM), max_steps=200, gtol=1e-5)
    assert int(result.status) == 1
    np.testing.assert_allclose(
        float(result.aux["resid_norm"]),
        float(jnp.linalg.norm(A @ result.x - B)),
        rtol=1e-3,
        atol=1e-7,
    )
    assert result.info.aux is not None


def test_jit_false_parity():
    callback, user_state0 = ridge_continuation(ridge_floor=1e-5, decrease=0.1)
    results = {}
    for jit in (True, False):
        solver = RidgeLevenbergMarquardt(
            linear_residual, penalty=make_penalty(), ridge=1e-2
        )
        results[jit] = solver.solve(
            jnp.zeros(P_DIM),
            max_steps=300,
            gtol=2e-5,
            callback=callback,
            user_state=user_state0,
            jit=jit,
        )
    assert int(results[True].status) == int(results[False].status)
    assert int(results[True].steps) == int(results[False].steps)
    np.testing.assert_allclose(
        np.asarray(results[True].x), np.asarray(results[False].x), atol=1e-5
    )
    np.testing.assert_allclose(
        float(results[True].lm_state.ridge),
        float(results[False].lm_state.ridge),
        rtol=1e-6,
    )


def test_equal_settings_solvers_share_the_compiled_solve_loop():
    traces = {"count": 0}
    penalty = make_penalty()

    def residual(theta, args, p):
        traces["count"] += 1
        return A @ theta - args

    def build():
        return RidgeLevenbergMarquardt(
            residual, penalty=penalty, ridge=1e-3, cache_jacobian=False,
            linear_solver="lsmr", iterative_maxiter=40,
            lsmr_preconditioner=IDENTITY_RIGHT,
        )

    a, b = build(), build()
    assert a == b
    assert hash(a) == hash(b)
    a.solve(jnp.zeros(P_DIM), B, max_steps=10, gtol=1e-4)
    count_after_first = traces["count"]
    # Every ridge solve pays exactly ONE eager residual call (the
    # unconditional init that resolves ridge/dtype); an equal-config solver
    # must add nothing beyond that -- no retrace of the compiled loop.
    b.solve(jnp.zeros(P_DIM), B, max_steps=10, gtol=1e-4)
    assert traces["count"] == count_after_first + 1
    a.solve(jnp.zeros(P_DIM), B, max_steps=10, gtol=1e-4)
    assert traces["count"] == count_after_first + 2
    # Any static-setting change is a different solver.
    different = RidgeLevenbergMarquardt(
        residual, penalty=penalty, ridge=1e-4, cache_jacobian=False,
        linear_solver="lsmr", iterative_maxiter=40,
        lsmr_preconditioner=IDENTITY_RIGHT,
    )
    assert a != different
