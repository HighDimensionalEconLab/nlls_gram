import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from nlls_gram import (
    RidgeLevenbergMarquardt,
    RidgePenalty,
    ridge_continuation,
)

RNG = np.random.default_rng(41)
M_RESID, P_DIM = 5, 9
A = jnp.asarray(RNG.normal(size=(M_RESID, P_DIM)), dtype=jnp.float32)
B = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)


def counting_identity_penalty(counters):
    # An identity penalty whose add_scaled records EXECUTIONS (not traces):
    # jax.debug.callback fires only when the owning cond branch actually
    # runs, so the counter measures the normal-matrix assemblies the solver
    # really paid for.
    def sqrt_apply(x):
        return x

    def add_scaled(H, c):
        jax.debug.callback(lambda: counters.append(1), ordered=True)
        return H + jnp.asarray(c, dtype=H.dtype) * jnp.eye(
            H.shape[0], dtype=H.dtype
        )

    return RidgePenalty(
        sqrt_apply=sqrt_apply,
        sqrt_transpose_apply=sqrt_apply,
        num_rows=P_DIM,
        add_scaled=add_scaled,
    )


def test_rejected_step_skips_normal_matrix_assembly():
    # B1: a rejected step re-factors G + damping*I but must NOT re-run the
    # GEMM + add_scaled assembly of G = J'J + ridge*L'L; a callback ridge
    # change must force it again through the cache's ridge key.
    counters = []
    penalty = counting_identity_penalty(counters)

    def residual(theta):
        # Nonlinear scalar tail engineered so the near-Newton step from a
        # small residual overshoots and gets rejected at tiny damping.
        return jnp.concatenate([A @ theta - B, jnp.array([theta[0] ** 2 - 1.0])])

    solver = RidgeLevenbergMarquardt(
        residual,
        penalty=penalty,
        ridge=1e-4,
        init_damping=1e-9,
        geodesic_acceleration=False,
    )
    x = jnp.zeros(P_DIM).at[0].set(0.05)
    lm_state = solver.init(x)

    counters.clear()
    x1, state1, info1 = solver.update(x, lm_state)
    jax.effects_barrier()
    assert len(counters) == 1
    assert not bool(info1.accepted)
    assert bool(state1.solver_cache.valid)

    # Rejected step: same x, same ridge -- assembly skipped, refactor only.
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


def test_traced_ridge_change_does_not_retrace_the_loop():
    # ridge is traced state: replacing its VALUE (directly or via the
    # continuation callback) must reuse the compiled solve loop. The
    # residual-body counter increments only while tracing.
    traces = {"count": 0}

    def residual(theta):
        traces["count"] += 1
        return A @ theta - B

    from nlls_gram import identity_penalty

    solver = RidgeLevenbergMarquardt(
        residual, penalty=identity_penalty(P_DIM), ridge=1e-3
    )
    x0 = jnp.zeros(P_DIM)
    solver.solve(x0, max_steps=30, gtol=1e-4)
    baseline = traces["count"]

    # Different ridge VALUE through a caller-supplied state: +1 eager init
    # residual call, zero retraces.
    state = solver.init(x0)
    state = dataclasses.replace(state, ridge=jnp.asarray(3e-3))
    solver.solve(x0, lm_state=state, max_steps=30, gtol=1e-4)
    assert traces["count"] == baseline + 1

    # The continuation callback (a NEW problem: callback identity keys the
    # compile) traces once, then repeat solves with the same callback reuse
    # the compiled loop.
    callback, us0 = ridge_continuation(ridge_floor=1e-6, decrease=0.1)
    solver.solve(
        x0, max_steps=60, gtol=1e-4, callback=callback, user_state=us0
    )
    with_callback = traces["count"]
    solver.solve(
        x0, max_steps=60, gtol=1e-4, callback=callback, user_state=us0
    )
    assert traces["count"] == with_callback + 1


def test_max_steps_change_does_not_retrace():
    # max_steps is traced (no save_steps), so budget changes reuse the loop.
    traces = {"count": 0}

    def residual(theta):
        traces["count"] += 1
        return A @ theta - B

    from nlls_gram import identity_penalty

    solver = RidgeLevenbergMarquardt(
        residual, penalty=identity_penalty(P_DIM), ridge=1e-3
    )
    x0 = jnp.zeros(P_DIM)
    solver.solve(x0, max_steps=10, gtol=1e-4)
    baseline = traces["count"]
    solver.solve(x0, max_steps=200, gtol=1e-4)
    assert traces["count"] == baseline + 1
