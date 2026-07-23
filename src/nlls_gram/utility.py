"""Shared solver machinery: status/hyperparameter/action/result types, the
jitted solve loop, multi-start drivers, and hashing/tree/history helpers.

Everything here is solver-agnostic: the loop and multi-start drivers consume
an informal solver protocol (``update``, ``_apply_action``, ``_converged``,
``_residual_and_aux``, ``_initial_info``, ``_cast_state``, ``_cold_state``,
``_ranking_objective``, ``has_aux``) implemented by both
:class:`~nlls_gram.LevenbergMarquardt` and
:class:`~nlls_gram.RidgeLevenbergMarquardt`. Public names are re-exported from
``nlls_gram`` (and ``nlls_gram.gram_lm`` for compatibility).
"""

import dataclasses
import enum
import inspect
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


class LMStatus(enum.IntEnum):
    """Integer status codes returned by ``solve``.

    Members are real ints (``IntEnum``): they work as dict keys, compare
    against status arrays, and ``LMStatus(int(result.status)).name`` recovers
    the label for logging. Callbacks may return bare members (or any weak
    integer value) as ``LMSolveAction.status`` -- the solver canonicalizes to
    int32 at the boundary, so no ``jnp.asarray(..., dtype=jnp.int32)`` casts
    are needed, under float32 or x64.
    """

    RUNNING = 0
    CONVERGED = 1
    MAX_STEPS = 2
    NONFINITE = 3
    CALLBACK_STOP = 4


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMHyperparams:
    """Per-step LM hyperparameters, carried in the solver state's ``hyper``.

    All fields are traced values, so a ``solve`` callback can reset them —
    e.g. grow the inner CG budget as the loss falls — via
    ``dataclasses.replace(ctx.lm_state, hyper=dataclasses.replace(
    ctx.lm_state.hyper, iterative_maxiter=...))``. A field constructed as
    ``None`` (uncapped ``max_damping``, backend-default ``iterative_maxiter``)
    is compiled out and stays ``None``. Static configuration (``linear_solver``,
    ``geodesic_acceleration``, ``cache_jacobian``, ``has_aux``, the metric or
    penalty) shapes the compiled program and lives on the solver, not here.
    """

    damping_decrease: jax.Array
    damping_increase: jax.Array
    min_damping: jax.Array
    max_damping: jax.Array | None
    geodesic_acceptance_ratio: jax.Array
    iterative_tol: jax.Array
    iterative_atol: jax.Array
    iterative_maxiter: jax.Array | None


def _damping_floor(min_damping, dtype):
    if dtype is None:
        seed = 0.0 if min_damping is None else min_damping
        dtype = jnp.asarray(seed).dtype
    dtype_floor = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    if min_damping is None:
        return dtype_floor
    return jnp.maximum(jnp.asarray(min_damping, dtype=dtype), dtype_floor)


def _cast_hyper(hyper, dtype):
    if hyper is None:
        return None
    return LMHyperparams(
        jnp.asarray(hyper.damping_decrease, dtype=dtype),
        jnp.asarray(hyper.damping_increase, dtype=dtype),
        _damping_floor(hyper.min_damping, dtype),
        None
        if hyper.max_damping is None
        else jnp.asarray(hyper.max_damping, dtype=dtype),
        jnp.asarray(hyper.geodesic_acceptance_ratio, dtype=dtype),
        jnp.asarray(hyper.iterative_tol, dtype=dtype),
        jnp.asarray(hyper.iterative_atol, dtype=dtype),
        None
        if hyper.iterative_maxiter is None
        else jnp.asarray(hyper.iterative_maxiter, dtype=jnp.int32),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveAction:
    """Optional callback action for ``solve``.

    A field left as ``None`` is unchanged. ``status`` is used only when ``stop``
    is true. ``stop`` and ``status`` are canonicalized by the solver (to bool
    and int32), so callbacks may return Python bools, bare ``LMStatus``
    members, or weak-typed arrays without explicit dtype casts.
    """

    stop: Any = None
    status: Any = None
    x: Any = None
    lm_state: Any = None
    args: Any = None
    user_state: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveContext:
    """Information passed to a ``solve`` callback after each LM update.

    ``lm_state`` and ``info`` hold the running solver's own state and info
    types (:class:`LMState`/:class:`LMInfo` for ``LevenbergMarquardt``,
    :class:`RidgeLMState`/:class:`RidgeLMInfo` for
    ``RidgeLevenbergMarquardt``).
    """

    step: jax.Array
    x: Any
    x_old: Any
    lm_state: Any
    lm_state_old: Any
    initial_lm_state: Any
    args: Any
    p: Any
    user_state: Any
    info: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveResult:
    """Final result returned by ``solve``.

    ``lm_state`` and ``info`` hold the solver's own state and info types
    (:class:`LMState`/:class:`LMInfo` for ``LevenbergMarquardt``,
    :class:`RidgeLMState`/:class:`RidgeLMInfo` for
    ``RidgeLevenbergMarquardt``).
    """

    x: Any
    lm_state: Any
    info: Any
    steps: jax.Array
    status: jax.Array
    args: Any
    p: Any
    user_state: Any
    # With has_aux=True: aux evaluated at the returned (x, args, p) — one extra
    # residual evaluation, well-defined for every status. Differentiable with
    # respect to p through the implicit rule (directly and through x*(p)).
    aux: Any = None
    # With save_steps=True: the iterate history as a pytree shaped like x with a
    # (max_steps + 1) leading axis — row 0 is x0, row s the kept iterate after
    # step s (post-callback-action), rows beyond ``steps`` are zero padding.
    # aux_history (has_aux only, else None) and args_history (None when args is
    # None) align row-for-row with x_history — args row s is the kept
    # post-action args after step s, the args consumed by step s + 1's update.
    # Differentiation-inert (zero tangents through the implicit rule).
    x_history: Any = None
    aux_history: Any = None
    args_history: Any = None
    # MultiStartInfo when solve ran with multi_start=...; None otherwise (an
    # empty pytree node, so the leaf count is unchanged when the feature is
    # off). Differentiation-inert (zero tangents through the implicit rule).
    multi_start: Any = None


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


def _typed_key(value):
    # Tag each hashable value/container with its type so the static key keeps 1, 1.0,
    # and True distinct -- raw == / hash collapse them (hash(1) == hash(True)), which
    # would silently reuse a mismatched compile. This mirrors jax's own strict-type
    # equality for static jit arguments; unhashable values still raise here (caught by
    # _hashable_hook, which degrades the spec to identity-hashing).
    if isinstance(value, tuple):
        return (tuple, tuple(_typed_key(v) for v in value))
    if isinstance(value, frozenset):
        return (frozenset, frozenset(_typed_key(v) for v in value))
    return (type(value), value)


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


def _tree_changed(new, old):
    new_leaves, new_treedef = jax.tree_util.tree_flatten(new)
    old_leaves, old_treedef = jax.tree_util.tree_flatten(old)
    if new_treedef != old_treedef:
        return jnp.asarray(True)
    changed = jnp.asarray(False)
    for new_leaf, old_leaf in zip(new_leaves, old_leaves, strict=True):
        # equal_nan: an unchanged NaN sentinel is not a change.
        changed = changed | ~jnp.array_equal(new_leaf, old_leaf, equal_nan=True)
    return changed


def _zero_tangent_leaf(leaf):
    if leaf is None:
        return None
    array = jnp.asarray(leaf)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return jnp.zeros(array.shape, dtype=jax.dtypes.float0)
    return jnp.zeros_like(leaf)


def _broadcast_leading_condition(condition, leaf):
    """Broadcast a scalar or leading-batch condition over an array leaf."""
    condition = jnp.asarray(condition, dtype=jnp.bool_)
    leaf_ndim = jnp.ndim(leaf)
    if condition.ndim < leaf_ndim:
        condition = jnp.reshape(
            condition, condition.shape + (1,) * (leaf_ndim - condition.ndim)
        )
    return condition


def _where_tree(condition, on_true, on_false):
    """Select matching pytrees, treating ``condition`` axes as leading axes."""

    def select(true_leaf, false_leaf):
        if true_leaf is None:
            return None
        return jnp.where(
            _broadcast_leading_condition(condition, true_leaf),
            true_leaf,
            false_leaf,
        )

    return jax.tree.map(select, on_true, on_false)


def _mask_tangent_tree(condition, tangent):
    """Keep tangent leaves where condition holds and zero them elsewhere."""

    def mask(leaf):
        if leaf is None:
            return None
        array = jnp.asarray(leaf)
        if array.dtype == jax.dtypes.float0:
            return leaf
        return jnp.where(
            _broadcast_leading_condition(condition, array),
            array,
            jnp.zeros_like(array),
        )

    return jax.tree.map(mask, tangent)


class _IdentityKey:
    """Static-key stand-in comparing by object identity (for unhashable values)."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return isinstance(other, _IdentityKey) and self.obj is other.obj

    def __hash__(self):
        return id(self.obj)


def _static_key_component(value):
    # Hashable settings (scalars, strings, functions, frozen metrics) key by
    # value; anything unhashable keys by identity so hashing never raises and
    # equality stays consistent with the hash.
    try:
        hash(value)
    except TypeError:
        return _IdentityKey(value)
    return value


class _IdentityCallable:
    """Hashable-by-identity pass-through for unhashable callables used as jit
    statics (e.g. an eq=True dataclass instance implementing ``__call__``).
    ``__weakref__`` is required: jax.eval_shape weak-references the callable.
    """

    __slots__ = ("fn", "__weakref__")

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)

    def __eq__(self, other):
        return isinstance(other, _IdentityCallable) and self.fn is other.fn

    def __hash__(self):
        return id(self.fn)


def _hashable_hook(fn):
    if fn is None:
        return None
    try:
        hash(fn)
    except TypeError:
        return _IdentityCallable(fn)
    return fn


def canonicalize_residual(residual_fn):
    """Wrap a residual taking ``(x)``, ``(x, args)``, or ``(x, args, p)`` --
    always in that order -- into the canonical 3-arg form, so the compiled
    code is identical for all three. Uninspectable signatures (or ``*args``)
    are assumed 3-arg. Returns ``(canonical_fn, arity)``.
    """
    try:
        signature = inspect.signature(residual_fn)
    except (TypeError, ValueError):
        residual_arity = 3
    else:
        residual_arity = 0
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                residual_arity += 1
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                residual_arity = 3
                break
        if residual_arity < 1 or residual_arity > 3:
            raise ValueError(
                "residual_fn must take 1 to 3 positional arguments: "
                "(x), (x, args), or (x, args, p)"
            )
    if residual_arity == 1:

        def canonical_residual(x, args, p):
            return residual_fn(x)

    elif residual_arity == 2:

        def canonical_residual(x, args, p):
            return residual_fn(x, args)

    else:
        canonical_residual = residual_fn
    return canonical_residual, residual_arity


def canonicalize_ad_preconditioner(ad_solver_preconditioner):
    """Normalize an ``ad_solver_preconditioner`` to the 1-arg form the AD
    solve calls. A callable already usable as ``(v)``, including helpers
    whose damping argument has a default, passes through unchanged. A
    callable REQUIRING a second argument (a ``(v, damping)`` helper such
    as Sherman-Morrison or Woodbury) is wrapped to be called with an
    explicit zero damping, the correct value for the undamped AD system.
    Helpers marked ``requires_positive_damping``
    (``pad_dual_preconditioner``) are rejected at construction: their
    zero-damping apply divides by zero. Uninspectable signatures pass
    through unchanged (the 1-arg contract).
    """
    if getattr(ad_solver_preconditioner, "requires_positive_damping", False):
        raise ValueError(
            "this preconditioner divides by the live damping and cannot "
            "serve as ad_solver_preconditioner (the AD system is undamped)"
        )
    try:
        signature = inspect.signature(ad_solver_preconditioner)
    except (TypeError, ValueError):
        return ad_solver_preconditioner
    try:
        signature.bind(object())
    except TypeError:
        pass
    else:
        return ad_solver_preconditioner
    try:
        signature.bind(object(), object())
    except TypeError:
        raise ValueError(
            "ad_solver_preconditioner must be callable as (v) or (v, damping)"
        ) from None

    def canonical_ad_preconditioner(v):
        return ad_solver_preconditioner(v, jnp.asarray(0.0, dtype=v.dtype))

    return canonical_ad_preconditioner


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
    lm_state,
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
    history = _init_history(solver, x, args, p, history_len)
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)
    info = solver._initial_info(x, lm_state, args, p)
    # Recast the state's scalars (damping, hyperparameters, and any
    # solver-specific carried scalars) and the tolerances to the residual
    # dtype so the while_loop carry matches what update() returns.
    atol = jnp.asarray(atol, dtype=info.loss.dtype)
    gtol = jnp.asarray(gtol, dtype=info.loss.dtype)
    xtol = jnp.asarray(xtol, dtype=info.loss.dtype)
    lm_state = solver._cast_state(lm_state, info.loss.dtype)
    initial_lm_state = lm_state
    step = jnp.asarray(0, dtype=jnp.int32)
    initial_nonfinite = ~jnp.isfinite(info.loss)
    initial_converged = solver._converged(info, atol, gtol, xtol)
    stop = initial_nonfinite | initial_converged
    status = jnp.where(
        initial_nonfinite,
        jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
        jnp.where(
            initial_converged,
            jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
            jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
        ),
    )

    def cond(carry):
        _, _, _, _, _, _, step, _, stop = carry
        return (~stop) & (step < max_steps)

    def body(carry):
        x, lm_state, args, user_state, history, _, step, _, _ = carry
        x_old, lm_state_old = x, lm_state
        x, lm_state, info = solver.update(x, lm_state, args, p)
        step = step + jnp.asarray(1, dtype=jnp.int32)
        current_nonfinite = ~jnp.isfinite(info.loss)

        action = None
        if callback is not None:
            ctx = LMSolveContext(
                step,
                x,
                x_old,
                lm_state,
                lm_state_old,
                initial_lm_state,
                args,
                p,
                user_state,
                info,
            )
            action = callback(ctx)
        action, x, lm_state, args, user_state, problem_changed = solver._apply_action(
            action, x, lm_state, args, user_state
        )
        history = _record_history(history, step, x, info, args)

        callback_stop = (
            jnp.asarray(False, dtype=jnp.bool_)
            if action.stop is None
            else jnp.asarray(action.stop, dtype=jnp.bool_)
        )
        callback_status = (
            jnp.asarray(LMStatus.CALLBACK_STOP, dtype=jnp.int32)
            if action.status is None
            else jnp.asarray(action.status, dtype=jnp.int32)
        )
        # info describes the pre-action (x, args); if the action changed them,
        # the tolerances must wait for a fresh update.
        converged = solver._converged(info, atol, gtol, xtol) & ~problem_changed
        reached_max = step >= max_steps
        stop = current_nonfinite | callback_stop | converged | reached_max
        status = jnp.where(
            current_nonfinite,
            jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                callback_stop,
                callback_status,
                jnp.where(
                    converged,
                    jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
                    jnp.where(
                        reached_max,
                        jnp.asarray(LMStatus.MAX_STEPS, dtype=jnp.int32),
                        jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
                    ),
                ),
            ),
        )
        return x, lm_state, args, user_state, history, info, step, status, stop

    carry = jax.lax.while_loop(
        cond,
        body,
        (x, lm_state, args, user_state, history, info, step, status, stop),
    )
    x, lm_state, args, user_state, history, info, step, status, _ = carry
    final_aux = None
    if solver.has_aux:
        final_aux = solver._residual_and_aux(x, args, p)[1]
    x_history, aux_history, args_history = _finalize_history(history, step, final_aux)
    return LMSolveResult(
        x,
        lm_state,
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
    lm_state,
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
    history = _init_history(solver, x, args, p, history_len)
    info = solver._initial_info(x, lm_state, args, p)
    lm_state = solver._cast_state(lm_state, info.loss.dtype)
    initial_lm_state = lm_state
    status = LMStatus.RUNNING
    steps = 0
    if not bool(jnp.isfinite(info.loss)):
        status = LMStatus.NONFINITE
    elif bool(solver._converged(info, atol, gtol, xtol)):
        status = LMStatus.CONVERGED

    for steps in range(1, max_steps + 1):
        if status != LMStatus.RUNNING:
            steps -= 1
            break
        x_old, lm_state_old = x, lm_state
        x, lm_state, info = solver.update(x, lm_state, args, p)
        if not bool(jnp.isfinite(info.loss)):
            status = LMStatus.NONFINITE
            history = _record_history(history, steps, x, info, args)
            break
        action = None
        if callback is not None:
            ctx = LMSolveContext(
                jnp.asarray(steps, dtype=jnp.int32),
                x,
                x_old,
                lm_state,
                lm_state_old,
                initial_lm_state,
                args,
                p,
                user_state,
                info,
            )
            action = callback(ctx)
        action, x, lm_state, args, user_state, problem_changed = solver._apply_action(
            action, x, lm_state, args, user_state
        )
        history = _record_history(history, steps, x, info, args)
        if action.stop is not None and bool(action.stop):
            status = (
                LMStatus.CALLBACK_STOP if action.status is None else int(action.status)
            )
            break
        # info describes the pre-action (x, args); if the action changed
        # them, the tolerances must wait for a fresh update.
        if bool(solver._converged(info, atol, gtol, xtol)) and not bool(
            problem_changed
        ):
            status = LMStatus.CONVERGED
            break
    else:
        steps = max_steps

    if status == LMStatus.RUNNING:
        status = LMStatus.MAX_STEPS
    final_aux = None
    if solver.has_aux:
        final_aux = solver._residual_and_aux(x, args, p)[1]
    x_history, aux_history, args_history = _finalize_history(history, steps, final_aux)
    return LMSolveResult(
        x,
        lm_state,
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
