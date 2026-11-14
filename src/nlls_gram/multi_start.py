import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from nlls_gram.solve_loop import _solve_loop_impl, _solve_python_impl
from nlls_gram.types import SolverStatus
from nlls_gram.utilities import _typed_key

__all__ = ["DrawNNXModule", "MultiStart", "MultiStartInfo"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MultiStartInfo:
    """Diagnostics attached to ``SolveResult.multi_start`` by a multi-start solve."""

    attempt: jax.Array
    accepted: jax.Array
    attempts_run: jax.Array
    loss: jax.Array


@dataclass(frozen=True, eq=False)
class MultiStart:
    """Multi-start retry or parallel race configuration for ``solve(...)``."""

    key: Any
    num_starts: int
    draw: Any = None
    reset_args: Any = None
    accept: Any = None
    parallel: bool = False

    def __post_init__(self):
        if self.num_starts < 1:
            raise ValueError("num_starts must be >= 1")
        if self.num_starts > 1 and self.draw is None and self.reset_args is None:
            raise ValueError("num_starts > 1 requires draw and/or reset_args")


class DrawNNXModule:
    """Multi-start ``draw`` hook re-initializing a flax ``nnx.Module``."""

    def __init__(self, module_cls, *args, wrt=None, **kwargs):
        self.module_cls = module_cls
        self.args = args
        self.wrt = wrt
        self.kwargs = tuple(sorted(kwargs.items()))

    def __call__(self, key, x_old, args_old):
        from flax import nnx

        module = self.module_cls(*self.args, rngs=nnx.Rngs(key), **dict(self.kwargs))
        wrt = nnx.Param if self.wrt is None else self.wrt
        _, theta, _ = nnx.split(module, wrt, ...)
        return theta

    def __hash__(self):
        return hash(
            (self.module_cls, self.wrt, _typed_key(self.args), _typed_key(self.kwargs))
        )

    def __eq__(self, other):
        return (
            isinstance(other, DrawNNXModule)
            and self.module_cls is other.module_cls
            and self.wrt == other.wrt
            and _typed_key(self.args) == _typed_key(other.args)
            and _typed_key(self.kwargs) == _typed_key(other.kwargs)
        )


def _accept_converged(_, result):
    return result.status == SolverStatus.CONVERGED


def _accept_converged_or_max_steps(_, result):
    return (result.status == SolverStatus.CONVERGED) | (
        result.status == SolverStatus.MAX_STEPS
    )


def _attempt_success(accept_fn, accept_key, result, loss):
    value = jnp.asarray(accept_fn(accept_key, result))
    if value.shape != ():
        raise ValueError(
            f"multi_start.accept must return a scalar; got shape {value.shape}"
        )
    return value.astype(jnp.bool_) & jnp.isfinite(loss)


def _type_spec(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    specs = []
    for leaf in leaves:
        if not (hasattr(leaf, "shape") and hasattr(leaf, "dtype")):
            leaf = jnp.asarray(leaf)
        specs.append((tuple(leaf.shape), leaf.dtype, getattr(leaf, "weak_type", False)))
    return treedef, specs


def _check_hook_types(name, contract, expected, got):
    if _type_spec(got) != _type_spec(expected):
        raise ValueError(
            f"multi_start.{name} must return {contract} matching the structure, "
            f"shapes, and dtypes of its input; expected {_type_spec(expected)}, "
            f"got {_type_spec(got)}"
        )


def _attempt_keys(key, attempt):
    return jax.random.split(jax.random.fold_in(key, attempt), 3)


def _multi_start_python_impl(
    solver,
    x,
    state,
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
    reset_args,
    accept,
    parallel,
):
    accept_fn = accept
    cold = solver._cold_state(state)

    def run_attempt(x_a, state_a, args_a, attempt):
        result = _solve_python_impl(
            solver,
            x_a,
            state_a,
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
        accept_key = _attempt_keys(key, attempt)[1]
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
                keys = _attempt_keys(key, lane)
                x_l = x if draw is None else draw(keys[0], x, args)
                args_l = args if reset_args is None else reset_args(keys[2], args)
                if draw is not None:
                    _check_hook_types("draw", "x", x, x_l)
                if reset_args is not None:
                    _check_hook_types("reset_args", "args", args, args_l)
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
        x_a, args_a, state_a = x, args, state
        for attempt in range(num_starts):
            if attempt > 0:
                keys = _attempt_keys(key, attempt)
                x_next = x_a if draw is None else draw(keys[0], x_a, args_a)
                args_next = (
                    args_a if reset_args is None else reset_args(keys[2], args_a)
                )
                if draw is not None:
                    _check_hook_types("draw", "x", x, x_next)
                if reset_args is not None:
                    _check_hook_types("reset_args", "args", args, args_next)
                x_a, args_a = x_next, args_next
                state_a = cold
            result, loss, success = run_attempt(x_a, state_a, args_a, attempt)
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
    state,
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
    reset_args,
    accept,
):
    accept_fn = accept

    def run_attempt(x_a, state_a, args_a, attempt):
        result = _solve_loop_impl(
            solver,
            x_a,
            state_a,
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
        accept_key = _attempt_keys(key, attempt)[1]
        loss = solver._ranking_objective(result, p, callback)
        success = _attempt_success(accept_fn, accept_key, result, loss)
        return dataclasses.replace(result, p=None), loss, success

    zero = jnp.asarray(0, dtype=jnp.int32)
    best, best_loss, done = run_attempt(x, state, args, zero)
    if draw is None and reset_args is None:
        info = MultiStartInfo(zero, done, jnp.asarray(1, dtype=jnp.int32), best_loss)
        return dataclasses.replace(best, p=p, multi_start=info)

    cold = solver._cold_state(state)

    def cond(carry):
        attempt, _, _, _, _, _, done = carry
        return ~done & (attempt < num_starts)

    def body(carry):
        attempt, x_prev, args_prev, best, best_loss, best_attempt, _ = carry
        keys = _attempt_keys(key, attempt)
        x_next = x_prev if draw is None else draw(keys[0], x_prev, args_prev)
        args_next = args_prev if reset_args is None else reset_args(keys[2], args_prev)
        result, loss, success = run_attempt(x_next, cold, args_next, attempt)
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
    state,
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
    reset_args,
    accept,
    num_starts,
):
    accept_fn = accept
    lanes = jnp.arange(num_starts, dtype=jnp.int32)
    attempt_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(lanes)
    lane_keys = jax.vmap(lambda k: jax.random.split(k, 3))(attempt_keys)
    accept_keys = lane_keys[:, 1]
    draw_keys = lane_keys[1:, 0]
    reset_keys = lane_keys[1:, 2]

    def prepend(first, rest):
        return jnp.concatenate([jnp.asarray(first)[None], rest], axis=0)

    if draw is None:
        xs, x_axis = x, None
    else:
        xs_drawn = jax.vmap(lambda k: draw(k, x, args))(draw_keys)
        xs, x_axis = jax.tree.map(prepend, x, xs_drawn), 0
    if reset_args is None or args is None:
        args_lanes, args_axis = args, None
    else:
        args_reset = jax.vmap(lambda k: reset_args(k, args))(reset_keys)
        args_lanes, args_axis = jax.tree.map(prepend, args, args_reset), 0
    cold = solver._cold_state(state)

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

    results, losses, successes = jax.vmap(solve_lane, in_axes=(x_axis, args_axis, 0))(
        xs, args_lanes, accept_keys
    )

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
    _multi_start_sequential_impl, static_argnums=(0, 8, 13, 14, 15, 16)
)
_multi_start_parallel_jit = jax.jit(
    _multi_start_parallel_impl, static_argnums=(0, 7, 12, 13, 14, 15, 16)
)
