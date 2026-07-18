import dataclasses

import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    LMSolveAction,
    LMStatus,
    MultiStart,
)

# Shared problems. The linear residual has the closed-form min-norm solution
# x*(p) = (p/5, 2p/5); the nonlinear variant x*(p) = (p^2/5, 2p^2/5) gives a
# nonzero Hessian for second-order checks. The saddle stalls at theta = 0
# (zero gradient, steps rejected forever -> MAX_STEPS). The two-basin problem
# has a global minimum at theta = 1 (loss 0) and a local one near theta = -1
# (loss ~ 4 * ALPHA^2); both report CONVERGED under gtol.
ALPHA = 0.1


def residual_linear(theta, args, p):
    return jnp.array([theta[0] + 2.0 * theta[1] - p])


def residual_nonlinear(theta, args, p):
    return jnp.array([theta[0] + 2.0 * theta[1] - p**2])


def residual_saddle(theta, args, p):
    return jnp.array([theta[0] ** 2 - p])


def residual_two_basin(theta, args, p):
    return jnp.array([theta[0] ** 2 - 1.0, ALPHA * (theta[0] - 1.0)])


def residual_inconsistent(theta, args, p):
    return jnp.array([theta[0] - p, theta[0] - p - 10.0])


def residual_nan(theta, args, p):
    return jnp.array([jnp.nan + theta[0]])


def draw_normal(key, x, args):
    return jax.random.normal(key, x.shape, x.dtype), args


def draw_chain(key, x, args):
    return 0.3 * x + jax.random.normal(key, x.shape, x.dtype), args


def draw_uniform(key, x, args):
    return jax.random.uniform(key, x.shape, x.dtype, -3.0, 3.0), args


def draw_zeros(key, x, args):
    return jnp.zeros_like(x), args


def draw_constant(key, x, args):
    return x, args


def draw_jump(key, x, args):
    return jax.random.uniform(key, x.shape, x.dtype, 1.5, 3.0), args


def accept_deep_basin(key, result):
    return result.info.loss < 1e-3


def accept_always(key, result):
    return jnp.asarray(True)


def accept_never(key, result):
    return jnp.asarray(False)


CALLS = {"draw": 0, "accept": 0}


def counting_draw(key, x, args):
    CALLS["draw"] += 1
    return jax.random.normal(key, x.shape, x.dtype), args


def counting_accept(key, result):
    CALLS["accept"] += 1
    return result.status == LMStatus.CONVERGED


RECORDED_ACCEPT_KEYS = []
RECORDED_DRAW_KEYS = []
RECORDED_DRAW_INPUTS = []


def recording_accept(key, result):
    RECORDED_ACCEPT_KEYS.append(key)
    return jnp.asarray(False)


def recording_draw(key, x, args):
    RECORDED_DRAW_KEYS.append(key)
    RECORDED_DRAW_INPUTS.append((x, args))
    return 0.5 * x - 1.0, args


def attempt_subkeys(key, k):
    draw_key, accept_key = jax.random.split(jax.random.fold_in(key, k))
    return draw_key, accept_key


def keys_equal(a, b):
    return bool(jnp.all(jax.random.key_data(a) == jax.random.key_data(b)))


def test_multi_start_none_is_inert():
    traces = {"count": 0}

    def residual(theta, args, p):
        traces["count"] += 1
        return theta - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2, cache_jacobian=False)
    result = solver.solve(jnp.array([0.0]), jnp.array([1.0]), max_steps=10, atol=1e-6)
    assert result.multi_start is None

    ms = MultiStart(key=jax.random.key(0), num_starts=2, draw=draw_normal)
    solver.solve(
        jnp.array([0.0]), jnp.array([1.0]), max_steps=10, atol=1e-6, multi_start=ms
    )
    count_after_multi = traces["count"]
    # The plain-solve cache is untouched: a later plain call with new values
    # adds no traces.
    solver.solve(jnp.array([0.5]), jnp.array([2.0]), max_steps=10, atol=1e-6)
    assert traces["count"] == count_after_multi


def test_sequential_first_attempt_success_matches_plain_solve():
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2)
    x0 = jnp.zeros(2)
    p = jnp.asarray(3.0)
    plain = solver.solve(x0, p=p, max_steps=80, atol=1e-6)
    ms = MultiStart(key=jax.random.key(1), num_starts=4, draw=draw_normal)
    result = solver.solve(x0, p=p, max_steps=80, atol=1e-6, multi_start=ms)

    assert jnp.allclose(result.x, plain.x, rtol=1e-7)
    assert int(result.steps) == int(plain.steps)
    assert int(result.status) == LMStatus.CONVERGED
    assert int(result.multi_start.attempt) == 0
    assert bool(result.multi_start.accepted)
    assert int(result.multi_start.attempts_run) == 1
    assert jnp.allclose(result.multi_start.loss, plain.info.loss)


def test_sequential_retries_until_converged():
    solver = LevenbergMarquardt(residual_saddle, init_damping=1e-2)
    p = jnp.asarray(4.0)
    ms = MultiStart(key=jax.random.key(2), num_starts=3, draw=draw_jump)
    result = solver.solve(
        jnp.array([0.0]), p=p, max_steps=60, atol=1e-6, multi_start=ms
    )

    assert int(result.status) == LMStatus.CONVERGED
    assert int(result.multi_start.attempt) >= 1
    assert int(result.multi_start.attempts_run) == int(result.multi_start.attempt) + 1
    assert bool(result.multi_start.accepted)
    assert jnp.allclose(result.x[0] ** 2, p, atol=1e-5)


def test_sequential_all_fail_returns_best_finite_loss():
    solver = LevenbergMarquardt(residual_inconsistent, init_damping=1.0)
    x0 = jnp.array([37.0])
    p = jnp.asarray(0.0)
    key = jax.random.key(3)
    num_starts = 4
    ms = MultiStart(key=key, num_starts=num_starts, draw=draw_chain)
    result = solver.solve(x0, p=p, max_steps=2, atol=1e-9, multi_start=ms)

    # Replicate the documented schedule with plain solves and chained draws.
    x_a = x0
    manual = []
    for k in range(num_starts):
        if k > 0:
            x_a, _ = draw_chain(attempt_subkeys(key, k)[0], x_a, None)
        manual.append(solver.solve(x_a, p=p, max_steps=2, atol=1e-9))
    losses = jnp.array([float(r.info.loss) for r in manual])
    expected = int(jnp.argmin(losses))

    assert int(result.multi_start.attempt) == expected
    assert not bool(result.multi_start.accepted)
    assert int(result.multi_start.attempts_run) == num_starts
    assert int(result.status) == LMStatus.MAX_STEPS
    assert jnp.allclose(result.x, manual[expected].x, rtol=1e-7)
    assert jnp.allclose(result.multi_start.loss, losses[expected], rtol=1e-7)


@pytest.mark.parametrize("jit", [True, False])
def test_sequential_none_finite_returns_last_attempt(jit):
    solver = LevenbergMarquardt(residual_nan, init_damping=1e-2)
    ms = MultiStart(key=jax.random.key(4), num_starts=3, draw=draw_normal)
    result = solver.solve(
        jnp.array([0.0]), max_steps=10, atol=1e-6, multi_start=ms, jit=jit
    )

    assert int(result.status) == LMStatus.NONFINITE
    assert int(result.multi_start.attempt) == 2
    assert int(result.multi_start.attempts_run) == 3
    assert not bool(result.multi_start.accepted)
    assert not jnp.isfinite(result.multi_start.loss)


def test_accept_hook_rejection_triggers_retry_and_acceptance():
    solver = LevenbergMarquardt(residual_two_basin, init_damping=1e-2)
    x0 = jnp.array([-2.0])
    key = jax.random.key(5)

    # Without accept, the shallow basin already reports CONVERGED at attempt 0.
    plain_ms = MultiStart(key=key, num_starts=8, draw=draw_uniform)
    shallow = solver.solve(x0, max_steps=100, gtol=1e-4, multi_start=plain_ms)
    assert int(shallow.multi_start.attempt) == 0
    assert float(shallow.info.loss) > 1e-3

    deep_ms = MultiStart(
        key=key, num_starts=8, draw=draw_uniform, accept=accept_deep_basin
    )
    deep = solver.solve(x0, max_steps=100, gtol=1e-4, multi_start=deep_ms)
    assert int(deep.multi_start.attempt) >= 1
    assert bool(deep.multi_start.accepted)
    assert float(deep.multi_start.loss) < 1e-3


@pytest.mark.parametrize("parallel", [False, True])
def test_accept_true_on_nonfinite_result_never_wins(parallel):
    solver = LevenbergMarquardt(residual_nan, init_damping=1e-2)
    ms = MultiStart(
        key=jax.random.key(6),
        num_starts=3,
        draw=draw_normal,
        accept=accept_always,
        parallel=parallel,
    )
    result = solver.solve(jnp.array([0.0]), max_steps=10, atol=1e-6, multi_start=ms)
    # accept=True cannot terminate or mark success while the loss is nonfinite.
    assert int(result.multi_start.attempts_run) == 3
    assert not bool(result.multi_start.accepted)


def test_accept_and_draw_receive_documented_key_schedule():
    RECORDED_ACCEPT_KEYS.clear()
    RECORDED_DRAW_KEYS.clear()
    RECORDED_DRAW_INPUTS.clear()
    solver = LevenbergMarquardt(residual_inconsistent, init_damping=1.0)
    key = jax.random.key(7)
    num_starts = 3
    ms = MultiStart(
        key=key, num_starts=num_starts, draw=recording_draw, accept=recording_accept
    )
    solver.solve(
        jnp.array([0.0]),
        p=jnp.asarray(0.0),
        max_steps=3,
        atol=1e-9,
        multi_start=ms,
        jit=False,
    )

    assert len(RECORDED_ACCEPT_KEYS) == num_starts
    assert len(RECORDED_DRAW_KEYS) == num_starts - 1
    for k in range(num_starts):
        assert keys_equal(RECORDED_ACCEPT_KEYS[k], attempt_subkeys(key, k)[1])
    for k in range(1, num_starts):
        assert keys_equal(RECORDED_DRAW_KEYS[k - 1], attempt_subkeys(key, k)[0])


def test_sequential_draw_receives_previous_initial_values():
    RECORDED_ACCEPT_KEYS.clear()
    RECORDED_DRAW_KEYS.clear()
    RECORDED_DRAW_INPUTS.clear()

    def residual(theta, args, p):
        return jnp.array([theta[0] - args, theta[0] - args - 10.0])

    def mutating_callback(ctx):
        return LMSolveAction(x=ctx.x + 100.0, args=ctx.args + 1.0)

    solver = LevenbergMarquardt(residual, init_damping=1.0)
    x0 = jnp.array([2.0])
    args0 = jnp.asarray(0.5)
    ms = MultiStart(key=jax.random.key(8), num_starts=3, draw=recording_draw)
    result = solver.solve(
        x0,
        args0,
        max_steps=3,
        atol=1e-9,
        callback=mutating_callback,
        multi_start=ms,
        jit=False,
    )

    # draw sees each attempt's INITIAL (x, args) -- the previous draw's output
    # chain -- never the callback-mutated finals.
    assert len(RECORDED_DRAW_INPUTS) == 2
    x_in_1, args_in_1 = RECORDED_DRAW_INPUTS[0]
    x_in_2, args_in_2 = RECORDED_DRAW_INPUTS[1]
    assert jnp.allclose(x_in_1, x0)
    assert jnp.allclose(args_in_1, args0)
    assert jnp.allclose(x_in_2, 0.5 * x0 - 1.0)
    assert jnp.allclose(args_in_2, args0)
    # The returned result carries the winner's final (mutated) args.
    assert float(result.args) > float(args0)


@pytest.mark.parametrize("parallel", [False, True])
def test_num_starts_one_matches_plain_solve(parallel):
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2)
    x0 = jnp.zeros(2)
    p = jnp.asarray(3.0)
    plain = solver.solve(x0, p=p, max_steps=80, atol=1e-6)
    ms = MultiStart(key=jax.random.key(9), num_starts=1, parallel=parallel)
    result = solver.solve(x0, p=p, max_steps=80, atol=1e-6, multi_start=ms)

    assert jnp.allclose(result.x, plain.x, rtol=1e-7)
    assert int(result.steps) == int(plain.steps)
    assert int(result.status) == int(plain.status)
    assert int(result.multi_start.attempt) == 0
    assert int(result.multi_start.attempts_run) == 1


def test_parallel_matches_manual_vmap_recipe():
    solver = LevenbergMarquardt(residual_two_basin, init_damping=1e-2)
    x0 = jnp.array([-2.0])
    key = jax.random.key(10)
    num_starts = 8
    ms = MultiStart(key=key, num_starts=num_starts, draw=draw_uniform, parallel=True)
    result = solver.solve(x0, max_steps=100, gtol=1e-4, multi_start=ms)

    lanes = [x0] + [
        draw_uniform(attempt_subkeys(key, k)[0], x0, None)[0]
        for k in range(1, num_starts)
    ]
    batched = jax.vmap(lambda x: solver.solve(x, max_steps=100, gtol=1e-4))(
        jnp.stack(lanes)
    )
    losses = jnp.where(jnp.isfinite(batched.info.loss), batched.info.loss, jnp.inf)
    successes = (batched.status == LMStatus.CONVERGED) & jnp.isfinite(losses)
    success_losses = jnp.where(successes, losses, jnp.inf)
    winner = int(
        jnp.argmin(success_losses) if bool(jnp.any(successes)) else jnp.argmin(losses)
    )

    assert int(result.multi_start.attempt) == winner
    assert jnp.allclose(result.x, batched.x[winner], rtol=1e-7)
    assert jnp.allclose(result.multi_start.loss, losses[winner], rtol=1e-7)
    # The deep basin (loss ~ 0) must beat the shallow one lane 0 lands in.
    assert float(result.multi_start.loss) < 1e-3


def test_parallel_exact_tie_breaks_to_lowest_lane():
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2)
    ms = MultiStart(
        key=jax.random.key(11), num_starts=4, draw=draw_constant, parallel=True
    )
    result = solver.solve(
        jnp.zeros(2), p=jnp.asarray(3.0), max_steps=80, atol=1e-6, multi_start=ms
    )
    assert int(result.multi_start.attempt) == 0


def test_parallel_all_fail_and_all_nonfinite_fallbacks():
    solver = LevenbergMarquardt(residual_inconsistent, init_damping=1.0)
    x0 = jnp.array([37.0])
    p = jnp.asarray(0.0)
    key = jax.random.key(12)
    num_starts = 4
    ms = MultiStart(key=key, num_starts=num_starts, draw=draw_chain, parallel=True)
    result = solver.solve(x0, p=p, max_steps=2, atol=1e-9, multi_start=ms)

    lanes = [x0] + [
        draw_chain(attempt_subkeys(key, k)[0], x0, None)[0]
        for k in range(1, num_starts)
    ]
    losses = jnp.array(
        [float(solver.solve(x, p=p, max_steps=2, atol=1e-9).info.loss) for x in lanes]
    )
    assert int(result.multi_start.attempt) == int(jnp.argmin(losses))
    assert not bool(result.multi_start.accepted)

    nan_solver = LevenbergMarquardt(residual_nan, init_damping=1e-2)
    nan_ms = MultiStart(key=key, num_starts=3, draw=draw_normal, parallel=True)
    nan_result = nan_solver.solve(
        jnp.array([0.0]), max_steps=5, atol=1e-6, multi_start=nan_ms
    )
    assert int(nan_result.multi_start.attempt) == 0
    assert not jnp.isfinite(nan_result.multi_start.loss)


def test_sequential_grad_through_winning_start_matches_closed_form():
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2)
    x0 = jnp.array([jnp.nan, jnp.nan])  # attempt 0 fails NONFINITE
    ms = MultiStart(key=jax.random.key(13), num_starts=2, draw=draw_zeros)

    def solved_x(p):
        return solver.solve(x0, p=p, max_steps=80, atol=1e-10, multi_start=ms).x

    p = jnp.asarray(3.0)
    p_dot = jnp.asarray(0.7)
    x, x_dot = jax.jvp(solved_x, (p,), (p_dot,))
    assert x.shape == (2,)
    assert x_dot.shape == (2,)
    assert jnp.allclose(x, jnp.array([3.0 / 5.0, 6.0 / 5.0]), atol=1e-5)
    assert jnp.allclose(x_dot, jnp.array([p_dot / 5.0, 2.0 * p_dot / 5.0]), atol=1e-6)

    _, pullback = jax.vjp(solved_x, p)
    (p_cotangent,) = pullback(jnp.array([3.0, 4.0]))
    assert p_cotangent.shape == ()
    assert jnp.allclose(p_cotangent, (3.0 + 2.0 * 4.0) / 5.0, atol=1e-6)


def test_parallel_grad_matches_closed_form_and_second_order():
    solver = LevenbergMarquardt(residual_nonlinear, init_damping=1e-2)
    x0 = jnp.array([jnp.nan, jnp.nan])
    ms = MultiStart(
        key=jax.random.key(14), num_starts=3, draw=draw_zeros, parallel=True
    )

    # x*(p) = (p^2/5, 2 p^2/5), so sum(x*) = 3 p^2 / 5.
    def sum_x(p):
        return jnp.sum(
            solver.solve(x0, p=p, max_steps=120, atol=1e-10, multi_start=ms).x
        )

    p = jnp.asarray(1.5)
    grad = jax.grad(sum_x)(p)
    assert grad.shape == ()
    assert jnp.allclose(grad, 6.0 * p / 5.0, atol=1e-4)
    hessian = jax.hessian(sum_x)(p)
    assert hessian.shape == ()
    assert jnp.allclose(hessian, 6.0 / 5.0, atol=1e-3)


def test_multi_start_does_not_retrace_on_value_changes():
    CALLS["draw"] = 0
    CALLS["accept"] = 0
    traces = {"count": 0}

    def residual(theta, args, p):
        traces["count"] += 1
        return theta - args

    solver = LevenbergMarquardt(residual, init_damping=1e-2, cache_jacobian=False)
    ms = MultiStart(
        key=jax.random.key(0), num_starts=3, draw=counting_draw, accept=counting_accept
    )
    solver.solve(
        jnp.array([0.0]), jnp.array([1.0]), max_steps=10, atol=1e-6, multi_start=ms
    )
    residual_traces = traces["count"]
    draw_calls = CALLS["draw"]
    accept_calls = CALLS["accept"]

    # New values everywhere (key, x0, args, loop controls, a *different*
    # num_starts > 1, a fresh-but-equal MultiStart object): zero new traces.
    # draw is invoked once more by the eval_shape structure check only.
    ms2 = MultiStart(
        key=jax.random.key(99),
        num_starts=5,
        draw=counting_draw,
        accept=counting_accept,
    )
    solver.solve(
        jnp.array([0.5]), jnp.array([2.0]), max_steps=25, atol=1e-8, multi_start=ms2
    )
    assert traces["count"] == residual_traces
    # At most one extra Python invocation of draw (the eval_shape structure
    # check; its trace may itself be cached) and none of accept.
    assert CALLS["draw"] <= draw_calls + 1
    assert CALLS["accept"] == accept_calls

    # Parallel: same num_starts, new key/x0 values -> zero new traces.
    pms = MultiStart(
        key=jax.random.key(1),
        num_starts=4,
        draw=counting_draw,
        accept=counting_accept,
        parallel=True,
    )
    solver.solve(
        jnp.array([0.0]), jnp.array([1.0]), max_steps=10, atol=1e-6, multi_start=pms
    )
    parallel_traces = traces["count"]
    pms2 = MultiStart(
        key=jax.random.key(2),
        num_starts=4,
        draw=counting_draw,
        accept=counting_accept,
        parallel=True,
    )
    solver.solve(
        jnp.array([0.25]), jnp.array([3.0]), max_steps=12, atol=1e-7, multi_start=pms2
    )
    assert traces["count"] == parallel_traces


def test_jit_false_matches_jit_true_and_draw_is_lazy():
    CALLS["draw"] = 0
    solver = LevenbergMarquardt(residual_two_basin, init_damping=1e-2)
    x0 = jnp.array([-2.0])
    ms = MultiStart(
        key=jax.random.key(15),
        num_starts=8,
        draw=counting_draw,
        accept=accept_deep_basin,
    )
    eager = solver.solve(x0, max_steps=100, gtol=1e-4, multi_start=ms, jit=False)
    # Lazy draws: exactly one call per retry that actually ran.
    assert CALLS["draw"] == int(eager.multi_start.attempt)

    jitted = solver.solve(x0, max_steps=100, gtol=1e-4, multi_start=ms)
    assert int(eager.multi_start.attempt) == int(jitted.multi_start.attempt)
    assert jnp.allclose(eager.x, jitted.x, rtol=1e-6)

    # A first-attempt success never draws.
    CALLS["draw"] = 0
    good = MultiStart(key=jax.random.key(16), num_starts=8, draw=counting_draw)
    solver.solve(
        jnp.array([2.0]), max_steps=100, gtol=1e-4, multi_start=good, jit=False
    )
    assert CALLS["draw"] == 0

    # Parallel evaluates every lane: num_starts - 1 draws.
    CALLS["draw"] = 0
    pms = MultiStart(
        key=jax.random.key(17), num_starts=4, draw=counting_draw, parallel=True
    )
    solver.solve(x0, max_steps=100, gtol=1e-4, multi_start=pms, jit=False)
    assert CALLS["draw"] == 3


def test_multi_start_inside_vmap_matches_loop():
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2)
    x0 = jnp.array([jnp.nan, jnp.nan])

    def solve_from(p, key):
        ms = MultiStart(key=key, num_starts=3, draw=draw_zeros)
        return solver.solve(x0, p=p, max_steps=80, atol=1e-10, multi_start=ms).x

    ps = jnp.array([1.0, 2.0, 3.0])
    keys = jax.random.split(jax.random.key(18), 3)
    batched = jax.vmap(solve_from)(ps, keys)
    looped = jnp.stack([solve_from(ps[i], keys[i]) for i in range(3)])
    assert jnp.allclose(batched, looped, rtol=1e-6)

    def solve_parallel(p, key):
        ms = MultiStart(key=key, num_starts=2, draw=draw_zeros, parallel=True)
        return solver.solve(x0, p=p, max_steps=80, atol=1e-10, multi_start=ms).x

    nested = jax.vmap(solve_parallel)(ps, keys)
    assert nested.shape == (3, 2)
    assert bool(jnp.all(jnp.isfinite(nested)))


def test_multi_start_validation_errors():
    key = jax.random.key(19)
    with pytest.raises(ValueError, match="num_starts"):
        MultiStart(key=key, num_starts=0)
    with pytest.raises(ValueError, match="num_starts"):
        MultiStart(key=key, num_starts=True)
    with pytest.raises(ValueError, match="requires draw"):
        MultiStart(key=key, num_starts=2)
    with pytest.raises(TypeError, match="draw must be callable"):
        MultiStart(key=key, num_starts=2, draw=5)
    with pytest.raises(TypeError, match="accept must be callable"):
        MultiStart(key=key, num_starts=1, accept=5)

    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2)
    x0 = jnp.zeros(2)
    with pytest.raises(TypeError, match="MultiStart"):
        solver.solve(x0, p=jnp.asarray(3.0), multi_start=42)

    def bad_draw(key, x, args):
        return x[:1], args

    bad = MultiStart(key=key, num_starts=2, draw=bad_draw)
    with pytest.raises(ValueError, match="draw must return"):
        solver.solve(x0, p=jnp.asarray(3.0), atol=1e-6, multi_start=bad)

    def vector_accept(key, result):
        return jnp.array([True, False])

    bad_accept = MultiStart(
        key=key, num_starts=2, draw=draw_normal, accept=vector_accept
    )
    with pytest.raises(ValueError, match="scalar"):
        solver.solve(x0, p=jnp.asarray(3.0), atol=1e-6, multi_start=bad_accept)


def test_save_steps_composes_with_multi_start():
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2)
    p = jnp.asarray(3.0)
    max_steps = 20
    ms = MultiStart(key=jax.random.key(20), num_starts=2, draw=draw_zeros)
    result = solver.solve(
        jnp.array([jnp.nan, jnp.nan]),
        p=p,
        max_steps=max_steps,
        atol=1e-10,
        save_steps=True,
        multi_start=ms,
    )
    plain = solver.solve(
        jnp.zeros(2), p=p, max_steps=max_steps, atol=1e-10, save_steps=True
    )

    assert int(result.multi_start.attempt) == 1
    assert result.x_history.shape == (max_steps + 1, 2)
    # The winner's history is the drawn attempt's, not attempt 0's NaN run.
    assert jnp.allclose(result.x_history, plain.x_history, rtol=1e-6, atol=1e-8)

    pms = MultiStart(
        key=jax.random.key(21), num_starts=3, draw=draw_zeros, parallel=True
    )
    parallel = solver.solve(
        jnp.array([jnp.nan, jnp.nan]),
        p=p,
        max_steps=max_steps,
        atol=1e-10,
        save_steps=True,
        multi_start=pms,
    )
    assert parallel.x_history.shape == (max_steps + 1, 2)


@pytest.mark.parametrize("parallel", [False, True])
def test_mv2020_style_draw_resamples_args(parallel):
    def residual(theta, args, p):
        return theta - args["data"] * p

    def draw_resample(key, x, args):
        data_key, carry_key = jax.random.split(key)
        data = jax.random.normal(data_key, args["data"].shape, args["data"].dtype)
        return jnp.zeros_like(x), {"data": data, "key": carry_key}

    def resample_callback(ctx):
        def fresh(_):
            carry_key, data_key = jax.random.split(ctx.args["key"])
            data = jax.random.normal(
                data_key, ctx.args["data"].shape, ctx.args["data"].dtype
            )
            return {"data": data, "key": carry_key}

        new_args = jax.lax.cond(ctx.step == 1, fresh, lambda _: ctx.args, None)
        return LMSolveAction(args=new_args)

    solver = LevenbergMarquardt(residual, init_damping=1e-2)
    args0 = {"data": jnp.array([1.0, -2.0, 0.5]), "key": jax.random.key(22)}
    x0 = jnp.array([jnp.nan, jnp.nan, jnp.nan])  # forces one retry
    p = jnp.asarray(2.0)
    ms = MultiStart(
        key=jax.random.key(23), num_starts=3, draw=draw_resample, parallel=parallel
    )

    def solved(p_value):
        return solver.solve(
            x0,
            args0,
            p=p_value,
            max_steps=60,
            atol=1e-8,
            callback=resample_callback,
            multi_start=ms,
        )

    result = solved(p)
    assert int(result.status) == LMStatus.CONVERGED
    assert int(result.multi_start.attempt) >= 1
    # The winner's args were resampled (by draw and by the callback).
    assert not jnp.allclose(result.args["data"], args0["data"])
    # With a callback present the ranking loss is recomputed at the returned
    # (x, args, p), where info.loss may be stale.
    recomputed = jnp.sum((result.x - result.args["data"] * p) ** 2)
    assert jnp.allclose(result.multi_start.loss, recomputed, atol=1e-12)

    # Implicit AD with a typed PRNG key living inside the selected args:
    # x*(p) = data * p at the frozen final args, so dx/dp = data.
    x_dot = jax.jvp(lambda pv: solved(pv).x, (p,), (jnp.asarray(1.0),))[1]
    assert jnp.allclose(x_dot, result.args["data"], atol=1e-5)
    _, pullback = jax.vjp(lambda pv: solved(pv).x, p)
    (p_cotangent,) = pullback(jnp.ones(3))
    assert jnp.allclose(p_cotangent, jnp.sum(result.args["data"]), atol=1e-5)


def test_warm_lm_state_applies_to_attempt_zero_only():
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2, cache_jacobian=True)
    p = jnp.asarray(3.0)
    warm = solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).lm_state
    x0 = jnp.array([jnp.nan, jnp.nan])
    ms = MultiStart(key=jax.random.key(24), num_starts=2, draw=draw_zeros)
    result = solver.solve(
        x0, p=p, lm_state=warm, max_steps=80, atol=1e-6, multi_start=ms
    )

    # The retry runs from the drawn start with the warm state's damping/hyper
    # but an invalidated Jacobian cache.
    cold = dataclasses.replace(warm, jacobian_valid=jnp.zeros_like(warm.jacobian_valid))
    manual = solver.solve(jnp.zeros(2), p=p, lm_state=cold, max_steps=80, atol=1e-6)
    assert int(result.multi_start.attempt) == 1
    assert int(result.steps) == int(manual.steps)
    assert jnp.allclose(result.x, manual.x, rtol=1e-7)

    # Attempt 0 converging under a warm cache keeps the plain-solve behavior.
    ok = solver.solve(
        jnp.zeros(2), p=p, lm_state=warm, max_steps=80, atol=1e-6, multi_start=ms
    )
    plain = solver.solve(jnp.zeros(2), p=p, lm_state=warm, max_steps=80, atol=1e-6)
    assert int(ok.multi_start.attempt) == 0
    assert jnp.allclose(ok.x, plain.x, rtol=1e-7)


def test_attempt_zero_honors_a_valid_jacobian_cache():
    # A fabricated VALID cache with a deliberately wrong Jacobian: attempt 0
    # must consume it exactly like a plain solve does (same trajectory), which
    # a mistaken _cold_lm_state on attempt 0 would break; the wrong first step
    # must also be observable against an invalidated-cache solve.
    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2, cache_jacobian=True)
    p = jnp.asarray(3.0)
    x0 = jnp.zeros(2)
    poisoned = dataclasses.replace(
        solver.init(x0, p=p),
        resid=jnp.array([-3.0]),
        Jt=2.0 * jnp.array([[1.0], [2.0]]),
        jacobian_valid=jnp.asarray(True),
    )
    honored = solver.solve(x0, p=p, lm_state=poisoned, max_steps=80, atol=1e-6)
    invalidated = solver.solve(
        x0,
        p=p,
        lm_state=dataclasses.replace(poisoned, jacobian_valid=jnp.asarray(False)),
        max_steps=80,
        atol=1e-6,
    )
    assert int(honored.steps) != int(invalidated.steps) or not jnp.allclose(
        honored.x, invalidated.x, rtol=1e-9, atol=0.0
    )

    ms = MultiStart(key=jax.random.key(27), num_starts=2, draw=draw_zeros)
    result = solver.solve(
        x0, p=p, lm_state=poisoned, max_steps=80, atol=1e-6, multi_start=ms
    )
    assert int(result.multi_start.attempt) == 0
    assert int(result.steps) == int(honored.steps)
    assert jnp.allclose(result.x, honored.x, rtol=1e-7)


@dataclasses.dataclass
class ScaledDraw:  # eq=True dataclass: instances are NOT hashable
    scale: float

    def __call__(self, key, x, args):
        return self.scale * jax.random.normal(key, x.shape, x.dtype), args


@dataclasses.dataclass
class ThresholdAccept:
    bound: float

    def __call__(self, key, result):
        return result.info.loss < self.bound


@pytest.mark.parametrize("parallel", [False, True])
def test_unhashable_callable_hooks_work_under_jit(parallel):
    draw = ScaledDraw(0.5)
    accept = ThresholdAccept(1e-8)
    with pytest.raises(TypeError):
        hash(draw)

    solver = LevenbergMarquardt(residual_linear, init_damping=1e-2)
    ms = MultiStart(
        key=jax.random.key(28),
        num_starts=3,
        draw=draw,
        accept=accept,
        parallel=parallel,
    )
    result = solver.solve(
        jnp.array([jnp.nan, jnp.nan]),
        p=jnp.asarray(3.0),
        max_steps=80,
        atol=1e-10,
        multi_start=ms,
    )
    assert int(result.status) == LMStatus.CONVERGED
    assert bool(result.multi_start.accepted)
    assert int(result.multi_start.attempt) >= 1


def test_args_only_redraw_invalidates_jacobian_cache():
    def residual(theta, args, p):
        return jnp.array([args[0] * theta[0] - 4.0])

    solver = LevenbergMarquardt(residual, init_damping=1e-2, cache_jacobian=True)

    def draw_args_only(key, x, args):
        return x, jnp.array([2.0])

    ms = MultiStart(key=jax.random.key(25), num_starts=2, draw=draw_args_only)
    result = solver.solve(
        jnp.array([1.0]),
        jnp.array([jnp.nan]),
        max_steps=60,
        atol=1e-8,
        multi_start=ms,
    )
    assert int(result.status) == LMStatus.CONVERGED
    assert int(result.multi_start.attempt) == 1
    assert jnp.allclose(result.x[0], 2.0, atol=1e-6)


@pytest.mark.parametrize("parallel", [False, True])
def test_multi_start_has_aux_tangents(parallel):
    def residual(theta, args, p):
        r = jnp.array([theta[0] + 2.0 * theta[1] - p])
        return r, {"scaled": p * theta[0]}

    solver = LevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)
    x0 = jnp.array([jnp.nan, jnp.nan])
    ms = MultiStart(
        key=jax.random.key(26), num_starts=2, draw=draw_zeros, parallel=parallel
    )

    def solved(p):
        return solver.solve(x0, p=p, max_steps=80, atol=1e-10, multi_start=ms)

    p = jnp.asarray(3.0)
    result, tangent = jax.jvp(solved, (p,), (jnp.asarray(1.0),))
    plain = solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-10)
    assert jnp.allclose(result.aux["scaled"], plain.aux["scaled"], rtol=1e-6)
    # aux = p * theta0*(p) = p^2/5, so d(aux)/dp = 2p/5.
    assert tangent.aux["scaled"].shape == ()
    assert jnp.allclose(tangent.aux["scaled"], 2.0 * p / 5.0, atol=1e-5)
