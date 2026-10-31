"""The solve loop: the jitted ``lax.while_loop`` driver, its Python mirror,
and the ``save_steps`` history buffers.

Both drivers consume an informal solver protocol -- ``update``,
``_apply_action``, ``_converged``, ``_step_failure_status``,
``_residual_and_aux``, ``_initial_info``, ``_cast_state``, ``has_aux`` --
implemented by ``SolverBase``.
"""

import jax
import jax.numpy as jnp

from nlls_gram.types import CallbackContext, SolveResult, SolverStatus


# save_steps bookkeeping shared by the jitted and Python solve loops: row `step` of
# x_history and args_history takes the kept post-action iterate and args; info.aux was
# evaluated at the pre-step x, so it lands one row earlier, and _finalize_history fills
# the last aux row from the final-solution evaluation. history_len is concrete (static
# under jit), so the buffers live entirely inside the loop implementations — no
# host-side allocation and no copy of a jit-input buffer before the in-place row
# updates. eval_shape gets the aux buffer shapes without paying for a residual
# evaluation.
def _history_buffer(tree, history_len):
    # Row 0 holds the initial value; tree.map over a None tree returns None.
    return jax.tree.map(
        lambda leaf: (
            jnp.zeros((history_len, *jnp.shape(leaf)), jnp.result_type(leaf))
            .at[0]
            .set(leaf)
        ),
        tree,
    )


def _init_history(solver, x0, args, p, history_len):
    if history_len is None:
        return None
    x_history = _history_buffer(x0, history_len)
    args_history = _history_buffer(args, history_len)
    aux_history = None
    if solver.has_aux:
        aux0 = jax.eval_shape(
            lambda x_, args_, p_: solver._residual_and_aux(x_, args_, p_)[1],
            x0,
            args,
            p,
        )
        aux_history = jax.tree.map(
            lambda leaf: jnp.zeros((history_len, *leaf.shape), leaf.dtype), aux0
        )
    return (x_history, aux_history, args_history)


def _record_history(history, step, x, info, args):
    if history is None:
        return None
    x_history, aux_history, args_history = history
    x_history = jax.tree.map(lambda buf, leaf: buf.at[step].set(leaf), x_history, x)
    args_history = jax.tree.map(
        lambda buf, leaf: buf.at[step].set(leaf), args_history, args
    )
    if aux_history is not None:
        aux_history = jax.tree.map(
            lambda buf, leaf: buf.at[step - 1].set(leaf), aux_history, info.aux
        )
    return (x_history, aux_history, args_history)


def _finalize_history(history, steps, final_aux):
    if history is None:
        return None, None, None
    x_history, aux_history, args_history = history
    if aux_history is not None:
        aux_history = jax.tree.map(
            lambda buf, leaf: buf.at[steps].set(leaf), aux_history, final_aux
        )
    return x_history, aux_history, args_history


def _solve_loop_impl(
    solver,
    x,
    state,
    args,
    p,
    user_state,
    history_len,
    max_steps,
    atol,
    gtol,
    xtol,
    callback,
):
    status_cls = SolverStatus
    context_cls = CallbackContext
    result_cls = SolveResult
    history = _init_history(solver, x, args, p, history_len)
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)
    info = solver._initial_info(x, state, args, p)
    # Recast the state's scalars (damping, hyperparameters, and any
    # solver-specific carried scalars) and the tolerances to the residual
    # dtype so the while_loop carry matches what update() returns.
    atol = jnp.asarray(atol, dtype=info.loss.dtype)
    gtol = jnp.asarray(gtol, dtype=info.loss.dtype)
    xtol = jnp.asarray(xtol, dtype=info.loss.dtype)
    state = solver._cast_state(state, info.loss.dtype)
    initial_state = state
    step = jnp.asarray(0, dtype=jnp.int32)
    initial_nonfinite = ~jnp.isfinite(info.loss)
    initial_converged = solver._converged(info, atol, gtol, xtol)
    stop = initial_nonfinite | initial_converged
    status = jnp.where(
        initial_nonfinite,
        jnp.asarray(status_cls.NONFINITE, dtype=jnp.int32),
        jnp.where(
            initial_converged,
            jnp.asarray(status_cls.CONVERGED, dtype=jnp.int32),
            jnp.asarray(status_cls.RUNNING, dtype=jnp.int32),
        ),
    )

    def cond(carry):
        _, _, _, _, _, _, step, _, stop = carry
        return (~stop) & (step < max_steps)

    def body(carry):
        x, state, args, user_state, history, _, step, _, _ = carry
        x_old, state_old = x, state
        x, state, info = solver.update(x, state, args, p)
        step = step + jnp.asarray(1, dtype=jnp.int32)
        step_failure = solver._step_failure_status(info)
        current_nonfinite = (~jnp.isfinite(info.loss)) | (
            step_failure == jnp.asarray(status_cls.NONFINITE, dtype=jnp.int32)
        )

        action = None
        if callback is not None:
            ctx = context_cls(
                step,
                x,
                x_old,
                state,
                state_old,
                initial_state,
                args,
                p,
                user_state,
                info,
            )
            action = callback(ctx)
        action, x, state, args, user_state, problem_changed = solver._apply_action(
            action, x, state, args, user_state
        )
        retry_changed = solver._action_rescued(
            action, state, state_old, problem_changed
        )
        history = _record_history(history, step, x, info, args)

        callback_stop = (
            jnp.asarray(False, dtype=jnp.bool_)
            if action.stop is None
            else jnp.asarray(action.stop, dtype=jnp.bool_)
        )
        callback_status = (
            jnp.asarray(status_cls.CALLBACK_STOP, dtype=jnp.int32)
            if action.status is None
            else jnp.asarray(action.status, dtype=jnp.int32)
        )
        # info describes the pre-action (x, args); if the action changed them,
        # the tolerances and any pre-action step failure wait for a fresh update.
        raw_converged = solver._converged(info, atol, gtol, xtol)
        converged = raw_converged & ~problem_changed
        step_failed = (
            (step_failure != jnp.asarray(status_cls.RUNNING, dtype=jnp.int32))
            & ~raw_converged
            & ~retry_changed
        )
        reached_max = step >= max_steps
        stop = current_nonfinite | callback_stop | converged | step_failed | reached_max
        status = jnp.where(
            current_nonfinite,
            jnp.asarray(status_cls.NONFINITE, dtype=jnp.int32),
            jnp.where(
                callback_stop,
                callback_status,
                jnp.where(
                    converged,
                    jnp.asarray(status_cls.CONVERGED, dtype=jnp.int32),
                    jnp.where(
                        step_failed,
                        step_failure,
                        jnp.where(
                            reached_max,
                            jnp.asarray(status_cls.MAX_STEPS, dtype=jnp.int32),
                            jnp.asarray(status_cls.RUNNING, dtype=jnp.int32),
                        ),
                    ),
                ),
            ),
        )
        return x, state, args, user_state, history, info, step, status, stop

    carry = jax.lax.while_loop(
        cond,
        body,
        (x, state, args, user_state, history, info, step, status, stop),
    )
    x, state, args, user_state, history, info, step, status, _ = carry
    final_aux = None
    if solver.has_aux:
        final_aux = solver._residual_and_aux(x, args, p)[1]
    x_history, aux_history, args_history = _finalize_history(history, step, final_aux)
    return result_cls(
        x,
        state,
        info,
        step,
        status,
        args,
        p,
        user_state,
        final_aux,
        x_history,
        aux_history,
        args_history,
    )


_solve_loop_jit = jax.jit(_solve_loop_impl, static_argnums=(0, 6, 11))


def _solve_python_impl(
    solver,
    x,
    state,
    args,
    p,
    user_state,
    history_len,
    max_steps,
    atol,
    gtol,
    xtol,
    callback,
):
    status_cls = SolverStatus
    context_cls = CallbackContext
    result_cls = SolveResult
    history = _init_history(solver, x, args, p, history_len)
    info = solver._initial_info(x, state, args, p)
    atol = jnp.asarray(atol, dtype=info.loss.dtype)
    gtol = jnp.asarray(gtol, dtype=info.loss.dtype)
    xtol = jnp.asarray(xtol, dtype=info.loss.dtype)
    state = solver._cast_state(state, info.loss.dtype)
    initial_state = state
    status = status_cls.RUNNING
    steps = 0
    if not bool(jnp.isfinite(info.loss)):
        status = status_cls.NONFINITE
    elif bool(solver._converged(info, atol, gtol, xtol)):
        status = status_cls.CONVERGED

    for steps in range(1, max_steps + 1):
        if status != status_cls.RUNNING:
            steps -= 1
            break
        x_old, state_old = x, state
        x, state, info = solver.update(x, state, args, p)
        step_failure = int(solver._step_failure_status(info))
        current_nonfinite = not bool(jnp.isfinite(info.loss)) or step_failure == int(
            status_cls.NONFINITE
        )
        action = None
        if callback is not None:
            ctx = context_cls(
                jnp.asarray(steps, dtype=jnp.int32),
                x,
                x_old,
                state,
                state_old,
                initial_state,
                args,
                p,
                user_state,
                info,
            )
            action = callback(ctx)
        action, x, state, args, user_state, problem_changed = solver._apply_action(
            action, x, state, args, user_state
        )
        retry_changed = solver._action_rescued(
            action, state, state_old, problem_changed
        )
        history = _record_history(history, steps, x, info, args)
        if current_nonfinite:
            status = status_cls.NONFINITE
            break
        if action.stop is not None and bool(action.stop):
            status = (
                status_cls.CALLBACK_STOP
                if action.status is None
                else int(action.status)
            )
            break
        # info describes the pre-action (x, args); if the action changed
        # them, the tolerances and any pre-action step failure wait for a fresh update.
        raw_converged = bool(solver._converged(info, atol, gtol, xtol))
        if raw_converged and not bool(problem_changed):
            status = status_cls.CONVERGED
            break
        if (
            step_failure != int(status_cls.RUNNING)
            and not raw_converged
            and not bool(retry_changed)
        ):
            status = step_failure
            break
    else:
        steps = max_steps

    if status == status_cls.RUNNING:
        status = status_cls.MAX_STEPS
    final_aux = None
    if solver.has_aux:
        final_aux = solver._residual_and_aux(x, args, p)[1]
    x_history, aux_history, args_history = _finalize_history(history, steps, final_aux)
    return result_cls(
        x,
        state,
        info,
        jnp.asarray(steps, dtype=jnp.int32),
        jnp.asarray(status, dtype=jnp.int32),
        args,
        p,
        user_state,
        final_aux,
        x_history,
        aux_history,
        args_history,
    )
