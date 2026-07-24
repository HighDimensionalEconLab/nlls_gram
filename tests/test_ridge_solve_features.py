import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    IdentityMetric,
    IdentityPreconditioner,
    LMSolveAction,
    LMStatus,
    MultiStart,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    ridge_continuation,
)

CG_CONFIG = CG(IdentityPreconditioner(), maxiter=40)

RNG = np.random.default_rng(11)
M_RESID, BLOCK, REPEATS, FREE = 4, 4, 2, 2
P_DIM = REPEATS * BLOCK + FREE
A_NP = RNG.normal(size=(M_RESID, P_DIM))
ROOT = RNG.normal(size=(BLOCK, BLOCK + 2))
K_NP = ROOT @ ROOT.T + 0.5 * np.eye(BLOCK)
W_NP = np.zeros((P_DIM, P_DIM))
for j in range(REPEATS):
    W_NP[j * BLOCK : (j + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = K_NP
B_NP = RNG.normal(size=M_RESID)

A = jnp.asarray(A_NP, dtype=jnp.float32)
B = jnp.asarray(B_NP, dtype=jnp.float32)
K = jnp.asarray(K_NP, dtype=jnp.float32)


def linear_residual(theta):
    return A @ theta - B


def linear_residual_with_args(theta, args):
    return args["A"] @ theta - args["b"]


def make_metric():
    return RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=REPEATS)


def min_seminorm_solution():
    kkt = np.block([[W_NP, A_NP.T], [A_NP, np.zeros((M_RESID, M_RESID))]])
    rhs = np.concatenate([np.zeros(P_DIM), B_NP])
    return np.linalg.solve(kkt, rhs)[:P_DIM]


def test_ridge_continuation_beats_any_single_moderate_ridge():
    x_dagger = min_seminorm_solution()
    fixed = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-2)
    fixed_result = fixed.solve(jnp.zeros(P_DIM), max_steps=300, gtol=1e-5)
    fixed_error = np.linalg.norm(np.asarray(fixed_result.x) - x_dagger)

    callback, user_state0 = ridge_continuation(ridge_floor=1e-6, decrease=0.1)
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-2)
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
    # a gtol that would otherwise fire never stops the loop -- and a
    # hand-rolled callback assigning a weak-typed Python float must not
    # change the while_loop carry aval (the solver recasts it).
    def always_shrink(ctx):
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, ridge=ctx.lm_state.ridge * 0.5)
        )

    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-2)
    result = solver.solve(
        jnp.zeros(P_DIM), max_steps=25, gtol=1e3, callback=always_shrink
    )
    assert int(result.status) == int(LMStatus.MAX_STEPS)
    np.testing.assert_allclose(float(result.lm_state.ridge), 1e-2 * 0.5**25, rtol=1e-4)
    # Without the callback the huge gtol converges immediately after one step.
    plain = solver.solve(jnp.zeros(P_DIM), max_steps=25, gtol=1e3)
    assert int(plain.status) == int(LMStatus.CONVERGED)

    def clumsy(ctx):
        return LMSolveAction(lm_state=dataclasses.replace(ctx.lm_state, ridge=5e-3))

    result = solver.solve(jnp.zeros(P_DIM), max_steps=50, gtol=1e-4, callback=clumsy)
    assert result.lm_state.ridge.dtype == jnp.float32
    np.testing.assert_allclose(float(result.lm_state.ridge), 5e-3, rtol=1e-6)


def test_save_steps_histories():
    args = {"A": A, "b": B}
    solver = RidgeLevenbergMarquardt(
        linear_residual_with_args, metric=make_metric(), ridge=1e-3
    )
    x0 = jnp.zeros(P_DIM)
    max_steps = 40
    result = solver.solve(x0, args, max_steps=max_steps, gtol=1e-5, save_steps=True)
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


def draw_nan_recovery(key, x, args):
    fresh = 0.5 * jax.random.normal(key, x.shape, dtype=x.dtype)
    return fresh, args


@pytest.mark.parametrize("parallel", [False, True])
def test_multi_start_recovers_from_a_poisoned_start(parallel):
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-3)
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


def test_multi_start_ranking_uses_the_ridge_objective():
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-3)
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
        residual_with_aux, metric=make_metric(), ridge=1e-3, has_aux=True
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
            linear_residual, metric=make_metric(), ridge=1e-2
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
    # float32 slop only: XLA fuses the whitened triangular solves differently
    # under jit, and the anneal compounds the rounding over ~40 steps.
    np.testing.assert_allclose(
        np.asarray(results[True].x), np.asarray(results[False].x), atol=5e-5
    )
    np.testing.assert_allclose(
        float(results[True].lm_state.ridge),
        float(results[False].lm_state.ridge),
        rtol=1e-6,
    )


def test_equal_settings_solvers_share_the_compiled_solve_loop():
    traces = {"count": 0}
    metric = make_metric()

    def residual(theta, args, p):
        traces["count"] += 1
        return A @ theta - args

    def build():
        return RidgeLevenbergMarquardt(
            residual,
            metric=metric,
            ridge=1e-3,
            cache_jacobian=False,
            linear_solver=CG_CONFIG,
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
    # Any static-setting change is a different solver, including a change
    # buried inside the typed config. IdentityPreconditioner() is stateless
    # and value-equal, so a rebuilt equal-valued CG config SHARES the
    # compile; a rebuilt metric (identity hashing) does not.
    different_ridge = RidgeLevenbergMarquardt(
        residual,
        metric=metric,
        ridge=1e-4,
        cache_jacobian=False,
        linear_solver=CG_CONFIG,
    )
    assert a != different_ridge
    rebuilt_config = RidgeLevenbergMarquardt(
        residual,
        metric=metric,
        ridge=1e-3,
        cache_jacobian=False,
        linear_solver=CG(IdentityPreconditioner(), maxiter=40),
    )
    assert a == rebuilt_config
    rebuilt_metric = RidgeLevenbergMarquardt(
        residual,
        metric=make_metric(),
        ridge=1e-3,
        cache_jacobian=False,
        linear_solver=CG_CONFIG,
    )
    assert a != rebuilt_metric


def test_traced_changes_do_not_retrace_the_loop():
    # ridge and max_steps are traced state/data: replacing their VALUES
    # (directly, via a caller-supplied state, or via the continuation
    # callback) must reuse the compiled solve loop. The residual-body counter
    # increments only while tracing.
    traces = {"count": 0}

    def residual(theta):
        traces["count"] += 1
        return A @ theta - B

    solver = RidgeLevenbergMarquardt(residual, metric=IdentityMetric(P_DIM), ridge=1e-3)
    x0 = jnp.zeros(P_DIM)
    solver.solve(x0, max_steps=30, gtol=1e-4)
    baseline = traces["count"]

    # Different ridge VALUE through a caller-supplied state: +1 eager init
    # residual call, zero retraces.
    state = solver.init(x0)
    state = dataclasses.replace(state, ridge=jnp.asarray(3e-3))
    solver.solve(x0, lm_state=state, max_steps=30, gtol=1e-4)
    assert traces["count"] == baseline + 1

    # A different max_steps budget reuses the loop too.
    solver.solve(x0, max_steps=200, gtol=1e-4)
    assert traces["count"] == baseline + 2

    # The continuation callback (a NEW problem: callback identity keys the
    # compile) traces once, then repeat solves with the same callback reuse
    # the compiled loop.
    callback, us0 = ridge_continuation(ridge_floor=1e-6, decrease=0.1)
    solver.solve(x0, max_steps=60, gtol=1e-4, callback=callback, user_state=us0)
    with_callback = traces["count"]
    solver.solve(x0, max_steps=60, gtol=1e-4, callback=callback, user_state=us0)
    assert traces["count"] == with_callback + 1


def test_rejected_step_skips_assembly():
    # The cholesky path materializes J~' (factor_solve_transpose on a
    # MATRIX) only inside the cache-refresh branch, while the per-step
    # gradient pulls back vectors; counting matrix applications therefore
    # counts assemblies. ndim is static at trace time, so the Python branch
    # below never enters the compiled program. A ridge change at the same x
    # must force reassembly through the cache ridge key.
    counters = []

    class CountingMetric(IdentityMetric):
        def factor_solve_transpose(self, v, ctx):
            if v.ndim == 2:
                jax.debug.callback(lambda: counters.append(1), ordered=True)
            return v

    def residual(theta):
        # Nonlinear scalar tail engineered so the near-Newton step from a
        # small residual overshoots and gets rejected at tiny damping.
        return jnp.concatenate([A @ theta - B, jnp.array([theta[0] ** 2 - 1.0])])

    solver = RidgeLevenbergMarquardt(
        residual,
        metric=CountingMetric(P_DIM),
        ridge=1e-4,
        init_damping=1e-9,
        geodesic_acceleration=False,
    )
    x = jnp.zeros(P_DIM, dtype=jnp.float32).at[0].set(0.05)
    lm_state = solver.init(x)

    counters.clear()
    x1, state1, info1 = solver.update(x, lm_state)
    jax.effects_barrier()
    assert len(counters) == 1
    assert not bool(info1.accepted)
    assert bool(state1.solver_cache.valid)

    # Rejected step: same x, same ridge -- the assembly (including the J~'
    # materialization) is skipped, refactor only.
    counters.clear()
    x2, state2, info2 = solver.update(x1, state1)
    jax.effects_barrier()
    assert len(counters) == 0

    # A ridge change at the same x invalidates through the cache ridge key.
    counters.clear()
    lowered = dataclasses.replace(state2, ridge=state2.ridge * 0.1)
    solver.update(x2, lowered)
    jax.effects_barrier()
    assert len(counters) == 1
