"""Multi-start configuration and the three drivers behind
``solve(multi_start=...)``: an unjitted Python loop, a jitted sequential
retry loop, and a jitted parallel ``vmap`` race.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from nlls_gram.lm_types import LMStatus
from nlls_gram.solve_loop import _solve_loop_impl, _solve_python_impl
from nlls_gram.utilities import _typed_key

__all__ = ["DrawNNXModule", "MultiStart", "MultiStartInfo"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MultiStartInfo:
    """Diagnostics attached to ``LMSolveResult.multi_start`` by a multi-start solve.

    ``attempt`` is the winning attempt/lane index (0 = the caller's
    ``(x0, args)``), ``accepted`` whether the winner passed the success test
    (``MultiStart.accept``, or the solve's ``max_steps_is_success`` policy), and
    ``attempts_run`` how many starts were solved (sequential mode stops at the
    first success; parallel mode always runs ``num_starts``). ``loss`` is the
    ranking objective used for selection -- the sum of squared residuals at the
    returned solution for ``LevenbergMarquardt``, the ridge objective for
    ``RidgeLevenbergMarquardt`` -- masked to ``+inf`` when nonfinite. Note
    ``accepted`` describes the multi-start success test, not ``LMInfo.accepted``
    (last-step acceptance).
    """

    attempt: jax.Array
    accepted: jax.Array
    attempts_run: jax.Array
    loss: jax.Array


@dataclass(frozen=True, eq=False)
class MultiStart:
    """Multi-start configuration for ``solve(multi_start=...)``.

    ``draw(key, x, args) -> (x_new, args_new)`` generates a fresh initial
    condition; it must be traceable and type-stable (returning the same pytree
    structure, shapes, and dtypes as its ``(x, args)`` inputs). ``accept(key,
    result) -> bool`` optionally overrides the success test (default:
    ``CONVERGED`` plus ``MAX_STEPS`` when the solve's
    ``max_steps_is_success=True``); it receives its own key so it can draw fresh
    validation data, and may return any scalar boolean-like value.
    Sequential mode (``parallel=False``) solves from ``(x0, args)`` and retries
    on failure, chaining each attempt's *initial* values into the next
    ``draw``; parallel mode solves all ``num_starts`` lanes under ``vmap``
    (lane 0 = the caller's ``(x0, args)``, the rest drawn from the originals)
    and selects the accepted lane with the lowest loss. The key schedule is
    ``draw_key, accept_key = jax.random.split(jax.random.fold_in(key, k))``
    for attempt ``k``.

    ``draw`` and ``accept`` enter the jit cache by identity (like
    ``callback``): define them once at setup scope, not inline per call.
    ``MultiStart`` is not a pytree -- ``solve`` unpacks it before tracing, with
    ``key`` the only traced field.
    """

    key: Any
    num_starts: int
    draw: Any = None
    accept: Any = None
    parallel: bool = False

    def __post_init__(self):
        if isinstance(self.num_starts, bool) or not isinstance(self.num_starts, int):
            raise ValueError("num_starts must be a Python int >= 1")
        if self.num_starts < 1:
            raise ValueError("num_starts must be a Python int >= 1")
        if self.num_starts > 1 and self.draw is None:
            raise ValueError(
                "num_starts > 1 requires draw; pass "
                "draw=(key, x, args) -> (x_new, args_new)"
            )
        if self.draw is not None and not callable(self.draw):
            raise TypeError("draw must be callable")
        if self.accept is not None and not callable(self.accept):
            raise TypeError("accept must be callable")


class DrawNNXModule:
    """Multi-start ``draw`` hook re-initializing a flax ``nnx.Module`` from a fresh key.

    Given a ``MultiStart`` retry key, builds
    ``module_cls(*args, rngs=nnx.Rngs(key), **kwargs)`` and returns its ``nnx.Param``
    state as the new solver start, passing ``args`` through unchanged. Use it instead
    of hand-rolling a re-init closure per driver::

        draw = DrawNNXModule(SequentialMLP, settings, dtype=dtype)
        ms = MultiStart(key=key, num_starts=5, draw=draw)

    The drawn parameter state must be type-stable against the solver's ``x0`` (same
    pytree structure, shapes, and dtypes) -- construct the module with a matching
    ``param_dtype``/``dtype`` (e.g. pass ``dtype=`` through). The paired
    ``nnx.GraphDef`` used by the residual's ``nnx.merge`` must come from the same
    ``module_cls(*args, **kwargs)`` spec.

    Value-hashable on ``(module_cls, args, kwargs)`` with jit's strict-type semantics
    (``1``, ``1.0``, and ``True`` key distinct compilations): equal specs compare equal
    and share one jit compilation instead of recompiling per instance (a fresh closure
    would not). ``args``/``kwargs`` must be hashable for that sharing, and their values
    must not be mutated after construction (a stale key would reuse the wrong compile);
    unhashable specs still work but recompile per instance. Requires ``flax`` installed
    (imported lazily on first draw).
    """

    def __init__(self, module_cls, *args, **kwargs):
        self.module_cls = module_cls
        self.args = args
        self.kwargs = tuple(sorted(kwargs.items()))

    def __call__(self, key, x_old, args_old):
        from flax import nnx

        module = self.module_cls(*self.args, rngs=nnx.Rngs(key), **dict(self.kwargs))
        _, theta = nnx.split(module, nnx.Param)
        return theta, args_old

    def __hash__(self):
        return hash((self.module_cls, _typed_key(self.args), _typed_key(self.kwargs)))

    def __eq__(self, other):
        return (
            isinstance(other, DrawNNXModule)
            and self.module_cls is other.module_cls
            and _typed_key(self.args) == _typed_key(other.args)
            and _typed_key(self.kwargs) == _typed_key(other.kwargs)
        )


def _accept_converged(_, result):
    return result.status == LMStatus.CONVERGED


def _accept_converged_or_max_steps(_, result):
    return (result.status == LMStatus.CONVERGED) | (result.status == LMStatus.MAX_STEPS)


def _attempt_success(accept_fn, accept_key, result, loss):
    value = jnp.asarray(accept_fn(accept_key, result))
    if value.shape != ():
        raise ValueError(
            f"multi_start.accept must return a scalar; got shape {value.shape}"
        )
    # An accepted-but-nonfinite result never wins: its masked loss is +inf.
    return value.astype(jnp.bool_) & jnp.isfinite(loss)


def _type_spec(tree):
    # weak_type is part of the spec: a weak/strong mismatch would break the
    # while_loop carry avals just like a dtype mismatch.
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    specs = []
    for leaf in leaves:
        if not (hasattr(leaf, "shape") and hasattr(leaf, "dtype")):
            leaf = jnp.asarray(leaf)
        specs.append((tuple(leaf.shape), leaf.dtype, getattr(leaf, "weak_type", False)))
    return treedef, specs


def _check_drawn_types(x, args, drawn):
    # Works on concrete draws and on jax.eval_shape outputs alike; a mismatch
    # would otherwise surface as an inscrutable while_loop/vmap error.
    if _type_spec(drawn) != _type_spec((x, args)):
        raise ValueError(
            "multi_start.draw must return (x, args) matching the structure, "
            f"shapes, and dtypes of its inputs; expected {_type_spec((x, args))}, "
            f"got {_type_spec(drawn)}"
        )


def _multi_start_python_impl(
    solver,
    x,
    lm_state,
    args,
    p,
    user_state,
    key,
    history_len,
    max_steps,
    atol,
    gtol,
    xtol,
    callback,
    num_starts,
    draw,
    accept,
    parallel,
):
    accept_fn = accept
    cold = solver._cold_state(lm_state)

    def run_attempt(x_a, lm_state_a, args_a, attempt):
        result = _solve_python_impl(
            solver,
            x_a,
            lm_state_a,
            args_a,
            p,
            user_state,
            history_len,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
        )
        accept_key = jax.random.split(jax.random.fold_in(key, attempt))[1]
        loss = solver._ranking_objective(result, p, callback)
        success = _attempt_success(accept_fn, accept_key, result, loss)
        return result, loss, bool(success)

    best = best_loss = best_attempt = None
    accepted = False
    if parallel:
        for lane in range(num_starts):
            if lane == 0:
                x_l, args_l = x, args
            else:
                draw_key = jax.random.split(jax.random.fold_in(key, lane))[0]
                x_l, args_l = draw(draw_key, x, args)
                _check_drawn_types(x, args, (x_l, args_l))
            result, loss, success = run_attempt(x_l, cold, args_l, lane)
            better = (
                best is None
                or (success and not accepted)
                or (success == accepted and bool(loss < best_loss))
            )
            if better:
                best, best_loss = result, loss
                best_attempt, accepted = lane, success
        attempts_run = num_starts
    else:
        x_a, args_a, lm_state_a = x, args, lm_state
        for attempt in range(num_starts):
            if attempt > 0:
                draw_key = jax.random.split(jax.random.fold_in(key, attempt))[0]
                x_a, args_a = draw(draw_key, x_a, args_a)
                _check_drawn_types(x, args, (x_a, args_a))
                lm_state_a = cold
            result, loss, success = run_attempt(x_a, lm_state_a, args_a, attempt)
            take = (
                best is None
                or success
                or bool(loss < best_loss)
                or not bool(jnp.isfinite(best_loss))
            )
            if take:
                best, best_loss = result, loss
                best_attempt, accepted = attempt, success
            if success:
                break
        attempts_run = attempt + 1
    info = MultiStartInfo(
        jnp.asarray(best_attempt, dtype=jnp.int32),
        jnp.asarray(accepted, dtype=jnp.bool_),
        jnp.asarray(attempts_run, dtype=jnp.int32),
        best_loss,
    )
    return dataclasses.replace(best, multi_start=info)


def _multi_start_sequential_impl(
    solver,
    x,
    lm_state,
    args,
    p,
    user_state,
    key,
    num_starts,
    history_len,
    max_steps,
    atol,
    gtol,
    xtol,
    callback,
    draw,
    accept,
):
    accept_fn = accept

    def run_attempt(x_a, lm_state_a, args_a, attempt):
        result = _solve_loop_impl(
            solver,
            x_a,
            lm_state_a,
            args_a,
            p,
            user_state,
            history_len,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
        )
        accept_key = jax.random.split(jax.random.fold_in(key, attempt))[1]
        loss = solver._ranking_objective(result, p, callback)
        success = _attempt_success(accept_fn, accept_key, result, loss)
        # p is loop-invariant: splice it out of the carried result and
        # reattach after selection.
        return dataclasses.replace(result, p=None), loss, success

    zero = jnp.asarray(0, dtype=jnp.int32)
    best, best_loss, done = run_attempt(x, lm_state, args, zero)
    if draw is None:
        info = MultiStartInfo(zero, done, jnp.asarray(1, dtype=jnp.int32), best_loss)
        return dataclasses.replace(best, p=p, multi_start=info)

    cold = solver._cold_state(lm_state)

    def cond(carry):
        attempt, _, _, _, _, _, done = carry
        return ~done & (attempt < num_starts)

    def body(carry):
        attempt, x_prev, args_prev, best, best_loss, best_attempt, _ = carry
        draw_key = jax.random.split(jax.random.fold_in(key, attempt))[0]
        x_next, args_next = draw(draw_key, x_prev, args_prev)
        result, loss, success = run_attempt(x_next, cold, args_next, attempt)
        # First success wins (the loop exits); among failures keep the lowest
        # masked loss, and an all-inf history always yields to the newest
        # attempt so the none-finite case returns the last one.
        take = success | (loss < best_loss) | ~jnp.isfinite(best_loss)
        best = jax.tree.map(lambda new, old: jnp.where(take, new, old), result, best)
        return (
            attempt + jnp.asarray(1, dtype=jnp.int32),
            x_next,
            args_next,
            best,
            jnp.where(take, loss, best_loss),
            jnp.where(take, attempt, best_attempt),
            success,
        )

    carry = jax.lax.while_loop(
        cond,
        body,
        (jnp.asarray(1, dtype=jnp.int32), x, args, best, best_loss, zero, done),
    )
    attempts_run, _, _, best, best_loss, best_attempt, accepted = carry
    info = MultiStartInfo(best_attempt, accepted, attempts_run, best_loss)
    return dataclasses.replace(best, p=p, multi_start=info)


def _multi_start_parallel_impl(
    solver,
    x,
    lm_state,
    args,
    p,
    user_state,
    key,
    history_len,
    max_steps,
    atol,
    gtol,
    xtol,
    callback,
    draw,
    accept,
    num_starts,
):
    accept_fn = accept
    lanes = jnp.arange(num_starts, dtype=jnp.int32)
    attempt_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(lanes)
    lane_keys = jax.vmap(jax.random.split)(attempt_keys)
    accept_keys = lane_keys[:, 1]
    draw_keys = lane_keys[1:, 0]
    xs_drawn, args_drawn = jax.vmap(lambda k: draw(k, x, args))(draw_keys)

    def prepend(first, rest):
        return jnp.concatenate([jnp.asarray(first)[None], rest], axis=0)

    xs = jax.tree.map(prepend, x, xs_drawn)
    args_lanes = None if args is None else jax.tree.map(prepend, args, args_drawn)
    # Under vmap the cache-reuse cond lowers to a select that evaluates both
    # branches, so a warm Jacobian cache cannot save work: drop it uniformly.
    cold = solver._cold_state(lm_state)

    def solve_lane(x_lane, args_lane, accept_key):
        result = _solve_loop_impl(
            solver,
            x_lane,
            cold,
            args_lane,
            p,
            user_state,
            history_len,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
        )
        loss = solver._ranking_objective(result, p, callback)
        success = _attempt_success(accept_fn, accept_key, result, loss)
        return dataclasses.replace(result, p=None), loss, success

    results, losses, successes = jax.vmap(
        solve_lane, in_axes=(0, None if args is None else 0, 0)
    )(xs, args_lanes, accept_keys)

    # Lowest masked loss among successful lanes; with none, lowest loss
    # overall (all-inf falls back to lane 0). argmin ties break low-index.
    success_losses = jnp.where(
        successes, losses, jnp.asarray(jnp.inf, dtype=losses.dtype)
    )
    winner = jnp.where(
        jnp.any(successes), jnp.argmin(success_losses), jnp.argmin(losses)
    ).astype(jnp.int32)
    best = jax.tree.map(lambda leaf: leaf[winner], results)
    info = MultiStartInfo(
        winner,
        successes[winner],
        jnp.asarray(num_starts, dtype=jnp.int32),
        losses[winner],
    )
    return dataclasses.replace(best, p=p, multi_start=info)


_multi_start_sequential_jit = jax.jit(
    _multi_start_sequential_impl, static_argnums=(0, 8, 13, 14, 15)
)
_multi_start_parallel_jit = jax.jit(
    _multi_start_parallel_impl, static_argnums=(0, 7, 12, 13, 14, 15)
)
