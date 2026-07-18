import dataclasses
import enum
import inspect
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import numpy as np
from jax.flatten_util import ravel_pytree

from nlls_gram.lsmr import lsmr_solve
from nlls_gram.metrics import Metric
from nlls_gram.recycled_cg import (
    RecycleConfig,
    RecycleState,
    build_coarse_operator,
    deflated_pcg,
)

# init() -> lm_state, update(x, lm_state, args, p) -> (new_x, lm_state, info),
# plus a solve() convenience loop. x is ANY pytree; the solver only ravels and
# unravels it with ravel_pytree and knows nothing about flax/nnx/optax.
# update() does not jit internally; solve(jit=True) wraps the loop in jax.jit.
# Hyperparameters are static Python scalars; data-dependent control flow is
# traced (jnp.where), so a rejected step returns the unchanged x rather than
# branching. Dtypes flow from the residual; damping scalars are cast to match.


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
    """Per-step LM hyperparameters, carried in ``LMState.hyper``.

    All fields are traced values, so a ``solve`` callback can reset them —
    e.g. grow the inner CG budget as the loss falls — via
    ``dataclasses.replace(ctx.lm_state, hyper=dataclasses.replace(
    ctx.lm_state.hyper, iterative_maxiter=...))``. A field constructed as
    ``None`` (uncapped ``max_damping``, backend-default ``iterative_maxiter``)
    is compiled out and stays ``None``. Static configuration (``linear_solver``,
    ``geodesic_acceleration``, ``cache_jacobian``, ``has_aux``, the metric)
    shapes the compiled program and lives on the solver, not here.
    """

    damping_decrease: jax.Array
    damping_increase: jax.Array
    max_damping: jax.Array | None
    geodesic_acceptance_ratio: jax.Array
    iterative_tol: jax.Array
    iterative_atol: jax.Array
    iterative_maxiter: jax.Array | None


def _cast_hyper(hyper, dtype):
    if hyper is None:
        return None
    return LMHyperparams(
        jnp.asarray(hyper.damping_decrease, dtype=dtype),
        jnp.asarray(hyper.damping_increase, dtype=dtype),
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
class LMState:
    """Carried LM solver state threaded through ``init``/``update``/``solve``.

    Only ``damping`` is always live; the remaining fields are populated by the
    features that need them and stay ``None`` on the default path (compiled away
    at no cost). A ``solve`` callback that rebuilds ``lm_state`` must PRESERVE the
    fields it does not mean to change -- in particular the ``recycle`` basis and
    the ``precond``/``precond_valid`` factory state, which carry across steps.

    Attributes:
        damping: ``()`` current LM damping ``lambda``.
        resid: cached residual at the current ``x`` (``cache_jacobian=True`` only,
            else ``None``).
        Jt: cached transpose-Jacobian ``J'`` output at the current ``x``
            (``cache_jacobian=True`` only).
        jacobian_valid: ``()`` bool -- the cached ``resid``/``Jt`` are still
            current because the last step was rejected so ``x`` did not move
            (``cache_jacobian=True`` only).
        aux: residual aux pytree at the current ``x`` (``has_aux=True``).
        hyper: per-step :class:`LMHyperparams`, populated by ``solve``; ``None``
            (``init``'s default) falls back to the constructor values with
            identical compiled code and no extra per-call buffers in manual
            ``update`` loops.
        recycle: :class:`~nlls_gram.RecycleState` carrying the deflation basis and
            warm starts across steps (``recycle`` set only).
        precond: ``preconditioner_factory`` prepared state (the ``prepare``-built
            pytree) at the current ``x``; ``None`` on the default path.
        precond_valid: ``()`` bool -- the carried ``precond`` is still current
            because ``x`` has not moved since it was built (so it is reused, not
            rebuilt); ``None`` on the default path.
    """

    damping: jax.Array
    resid: jax.Array | None = None
    Jt: jax.Array | None = None
    jacobian_valid: jax.Array | None = None
    aux: Any = None
    hyper: LMHyperparams | None = None
    recycle: RecycleState | None = None
    precond: Any = None
    precond_valid: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMInfo:
    """Per-step diagnostics returned by ``update`` (and for each ``solve`` step).

    The loss/damping fields report the accept/reject outcome of the step, while
    ``grad_norm``/``step_norm``/``aux`` are evaluated at the PRE-step ``x`` (the
    iterate the step was computed from), so they describe the point entering the
    step, not the one it produced.

    Attributes:
        loss: ``min(loss_old, loss_candidate)`` sum of squared residuals (at the
            retained iterate).
        loss_old: sum of squared residuals at the pre-step ``x``.
        loss_candidate: sum of squared residuals at the trial point.
        accepted: ``()`` bool, whether the trial step was accepted.
        damping: ``()`` post-update damping ``lambda``.
        damping_factor: ``()`` multiplicative damping update applied this step.
        used_geodesic: ``()`` bool, whether the geodesic-acceleration correction
            entered the accepted step.
        acceleration_ratio: ``()`` geodesic acceleration-to-velocity norm ratio.
        grad_norm: ``()`` ``||J' r||`` at the pre-step ``x``.
        step_norm: ``()`` ``||candidate step||``, reported even when the step is
            rejected.
        aux: residual aux output at the pre-step ``x`` (``has_aux=True``, else
            ``None``).
    """

    loss: jax.Array
    loss_old: jax.Array
    loss_candidate: jax.Array
    accepted: jax.Array
    damping: jax.Array
    damping_factor: jax.Array
    used_geodesic: jax.Array
    acceleration_ratio: jax.Array
    grad_norm: jax.Array
    step_norm: jax.Array
    aux: Any = None


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
    """Information passed to a ``solve`` callback after each LM update."""

    step: jax.Array
    x: Any
    x_old: Any
    lm_state: LMState
    lm_state_old: LMState
    initial_lm_state: LMState
    args: Any
    p: Any
    user_state: Any
    info: LMInfo


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveResult:
    """Final result returned by ``solve``."""

    x: Any
    lm_state: LMState
    info: LMInfo
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
    (``MultiStart.accept``, or ``status == LMStatus.CONVERGED``), and
    ``attempts_run`` how many starts were solved (sequential mode stops at the
    first success; parallel mode always runs ``num_starts``). ``loss`` is the
    ranking loss selection used: the sum of squared residuals at the returned
    solution, masked to ``+inf`` when nonfinite. Note ``accepted`` describes
    the multi-start success test, not ``LMInfo.accepted`` (last-step
    acceptance).
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
    ``result.status == LMStatus.CONVERGED``); it receives its own key so it can
    draw fresh validation data, and may return any scalar boolean-like value.
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


class PreconditionerFactory:
    """θ-adaptive dual preconditioner: a value-hashable ``(prepare, apply)`` pair.

    For ``linear_solver="cg"``, supplies a dual preconditioner REBUILT from the
    current iterate every step, replacing the frozen ``dual_preconditioner``.
    Pass exactly one of ``dual_preconditioner`` or ``preconditioner_factory``.
    Use it when the dual operator ``J M^{-1} J' + damping I`` rotates enough as
    LM drifts ``x`` that a preconditioner frozen at ``x0`` decays into an
    ineffective (breakdown-inducing) approximation downstream, while one rebuilt
    from the live iterate keeps the inner CG converging::

        def prepare(x, args, p):
            # model-structured build from the CURRENT iterate x
            return diag  # any fixed-shape pytree of arrays

        def apply(state, v, damping):
            return v / (state + damping)  # SPD, linear in v

        solver = LevenbergMarquardt(
            residual_fn,
            linear_solver="cg",
            preconditioner_factory=PreconditionerFactory(prepare, apply),
            iterative_maxiter=...,
        )

    - ``prepare(x, args, p) -> state`` builds a fixed-shape pytree of arrays from
      the CURRENT solver iterate ``x`` (the user pytree, NOT the raveled flat
      ``theta`` — model-structured access is the point), the residual ``args``,
      and ``p``. Runs inside the jitted loop as traced ops (no recompile), once
      per accepted step: after a rejected step ``x`` did not move, so the carried
      state is reused and only the live ``damping`` changes.
    - ``apply(state, v, damping) -> vector`` is the per-iteration apply: an SPD,
      linear-in-``v`` approximation of ``(J M^{-1} J' + damping I)^{-1} v``. It
      must stay well-defined at ``damping = 0``, since the cg-resolved implicit
      derivative reuses it (undamped) at the converged solution unless an
      explicit ``implicit_preconditioner`` is given.

    Value-hashable on ``(prepare, apply)`` with jit's static-key semantics: equal
    pairs share one compiled solve loop (like ``DrawNNXModule`` and the frozen
    preconditioner identities). Define ``prepare``/``apply`` once at setup scope
    so their identities are stable; a fresh closure per call keys a new compile.
    ``prepare`` and ``apply`` must be hashable for that sharing; an unhashable
    pair still works but keys the solver by identity (recompiling per instance).
    """

    def __init__(self, prepare, apply):
        if not callable(prepare):
            raise TypeError("PreconditionerFactory.prepare must be callable")
        if not callable(apply):
            raise TypeError("PreconditionerFactory.apply must be callable")
        self.prepare = prepare
        self.apply = apply

    def __hash__(self):
        return hash((self.prepare, self.apply))

    def __eq__(self, other):
        return (
            isinstance(other, PreconditionerFactory)
            and self.prepare == other.prepare
            and self.apply == other.apply
        )


class WhitenedPreconditioner:
    """Parameter-space right-preconditioner for ``linear_solver="lsmr"``: a
    value-hashable pair ``(solve, solve_transpose)`` applying ``R^{-1}`` and
    ``R^{-T}``.

    LSMR then runs on the preconditioned operator ``B R^{-1}`` (``B = J S``): the
    iteration variable is ``z``, the operator is ``x -> B(solve(x, damping))``,
    the adjoint is ``w -> solve_transpose(Bᵀ w, damping)``, and the returned step
    un-preconditions the final iterate as ``u = R^{-1} z``. A well-chosen ``R``
    (a Schur-complement factor of the parameter-space normal operator is the
    canonical construction) clusters the spectrum of ``B R^{-1}`` and cuts the
    endgame iteration count by orders of magnitude::

        def solve(v, damping):
            return jsp_linalg.solve_triangular(R, v)              # R^{-1} v

        def solve_transpose(w, damping):
            return jsp_linalg.solve_triangular(R.T, w)            # R^{-T} w

        solver = LevenbergMarquardt(
            residual_fn, linear_solver="lsmr",
            whitened_preconditioner=WhitenedPreconditioner(solve, solve_transpose),
        )

    - ``solve(v, damping) -> vector`` applies ``R^{-1}`` on a parameter-space
      vector; ``solve_transpose(w, damping) -> vector`` applies ``R^{-T}``. Both
      receive the live ``damping`` (like ``dual_preconditioner(v, damping)``), so
      a ``damping``-analytic ``R`` folds ``lambda`` in exactly.
    - **Surrogate subproblem**: LSMR's scalar ``damp = sqrt(damping)`` applied to
      ``B R^{-1}`` regularizes in the ``RᵀR`` metric, so the computed step is
      ``u = -(BᵀB + damping RᵀR)^{-1} Bᵀ r`` -- a documented surrogate of the
      plain ``I``-metric-damped subproblem. This is admissible: LM acceptance
      guards on the true ``||r||``, and the ``damping -> 0`` selection limit is
      ``R``-invariant (it is the minimum-metric-norm step regardless of ``R``).
    - LSMR stopping (``iterative_tol``/``iterative_atol``) is measured on the
      preconditioned operator -- the well-conditioned ``z`` coordinates.

    ``None`` (the default) runs plain LSMR. Value-hashable on
    ``(solve, solve_transpose)`` with jit's static-key semantics: equal pairs
    share one compiled solve loop, so define the callables once at setup scope.
    """

    def __init__(self, solve, solve_transpose):
        if not callable(solve):
            raise TypeError("WhitenedPreconditioner.solve must be callable")
        if not callable(solve_transpose):
            raise TypeError("WhitenedPreconditioner.solve_transpose must be callable")
        self.solve = solve
        self.solve_transpose = solve_transpose

    def __hash__(self):
        return hash((self.solve, self.solve_transpose))

    def __eq__(self, other):
        return (
            isinstance(other, WhitenedPreconditioner)
            and self.solve == other.solve
            and self.solve_transpose == other.solve_transpose
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


def canonicalize_implicit_preconditioner(implicit_preconditioner):
    """Normalize an ``implicit_preconditioner`` to the 1-arg form the
    implicit solve calls. A callable already usable as ``(v)`` -- every
    pre-1.7 callback, and helpers whose damping argument has a default --
    passes through unchanged, so no existing behavior shifts. A callable
    REQUIRING a second argument (a ``(v, damping)`` dual helper such as
    Sherman-Morrison or Woodbury) is wrapped to be called with an explicit
    zero damping, the correct value for the undamped implicit dual.
    Helpers marked ``requires_positive_damping`` (``pad_dual_preconditioner``)
    are rejected at construction: their zero-damping apply divides by zero.
    Uninspectable signatures pass through unchanged (the historical 1-arg
    contract).
    """
    if getattr(implicit_preconditioner, "requires_positive_damping", False):
        raise ValueError(
            "this preconditioner divides by the live damping and cannot "
            "serve as implicit_preconditioner (the implicit dual system is "
            "undamped)"
        )
    try:
        signature = inspect.signature(implicit_preconditioner)
    except (TypeError, ValueError):
        return implicit_preconditioner
    try:
        signature.bind(object())
    except TypeError:
        pass
    else:
        return implicit_preconditioner
    try:
        signature.bind(object(), object())
    except TypeError:
        raise ValueError(
            "implicit_preconditioner must be callable as (v) or (v, damping)"
        ) from None

    def canonical_implicit_preconditioner(v):
        return implicit_preconditioner(v, jnp.asarray(0.0, dtype=v.dtype))

    return canonical_implicit_preconditioner


class LevenbergMarquardt:
    """Metric-damped Levenberg-Marquardt for ``min ||r(x, args, p)||^2`` over a
    JAX pytree ``x``, specialized to ``n_residuals << n_params``: the default
    path factors the small damped Gram system in residual space. With metric
    ``M`` and ``P = M^{-1}``, the dense dual step is
    ``step = -P J' (J P J' + damping I_m)^{-1} r``. Exposes per-step
    ``update``, a jitted ``solve`` loop with callbacks, and implicit
    differentiation of ``solve`` with respect to ``p``.

    ``linear_solver="augmented_qr"`` solves the whitened LM subproblem by a
    direct reduced QR factorization of ``[J S; sqrt(damping) I]``. Unlike the
    residual-dimension-reduced ``"qr"`` path, it remains well-defined for a
    rank-deficient Jacobian whenever damping is positive, but its factorization
    width is the flattened parameter count. It is therefore intended for small
    systems, including square algebraic roots, rather than the package's usual
    massively underdetermined regime.

    ``linear_solver="lsmr"`` is the matrix-free sibling of ``augmented_qr``: it
    solves the same whitened damped subproblem
    ``min_u ||r + B u||^2 + damping ||u||^2`` (``B = J S``, ``S = metric.inv_sqrt``,
    step ``s = S u``) by LSMR bidiagonalization from ``J``/``J'`` matvecs alone.
    Because it works on ``B`` -- whose condition number is the square root of the
    ``cg`` dual ``J P J' + damping I`` -- it keeps the step accurate at small
    damping where the squared dual solve hits its ``eps * cond`` floor. It needs
    the metric's ``inv_sqrt``/``inv_sqrt_transpose`` (the identity metric supplies
    them) and maps the same ``iterative_tol``/``iterative_atol``/
    ``iterative_maxiter`` hooks onto its normal-equations stopping test.
    ``dual_preconditioner``, ``preconditioner_factory``, and ``recycle`` are
    ``cg``-only. Its own preconditioner is ``whitened_preconditioner`` (a
    ``WhitenedPreconditioner``): a parameter-space right-preconditioner ``R^{-1}``
    running LSMR on ``B R^{-1}`` to cluster the spectrum, which cuts the endgame
    iteration count by orders of magnitude when ``B`` itself is ill-conditioned
    (a Schur-complement factor is canonical). Its ``damp`` then regularizes in the
    ``R'R`` metric -- the step ``u = -(B'B + damping R'R)^{-1} B' r`` is a
    documented surrogate whose ``damping -> 0`` selection limit is ``R``-invariant.
    Differentiating a forward ``lsmr`` solve uses the dense cholesky implicit rule
    under ``implicit_solver="auto"``.

    ``linear_solver="cg"`` requires ``dual_preconditioner(v, damping)``: a
    jit-traceable, linear, SPD approximation of
    ``(J P J' + damping I_m)^{-1} v`` used as the CG preconditioner (for the
    geodesic-acceleration solve as well); pass ``identity_preconditioner()``
    to run unpreconditioned CG. It never changes the subproblem: at
    inner convergence the step is identical, and a budget-truncated step still
    lies in ``range(P J')``, so the minimum-metric-norm selection for
    underdetermined residuals is unchanged — the preconditioner may be
    approximate even though ``metric.solve`` must stay exact.

    ``preconditioner_factory`` (a ``PreconditionerFactory``) is the θ-adaptive
    alternative to the frozen ``dual_preconditioner``: exactly one of the two is
    required for ``linear_solver="cg"``. Its ``prepare(x, args, p)`` rebuilds the
    preconditioner state from the CURRENT iterate inside the jitted loop (once
    per accepted step; a rejected step reuses the carried state since ``x`` did
    not move), and ``apply(state, v, damping)`` is the per-iteration apply. Reach
    for it when a preconditioner frozen at ``x0`` decays as LM drifts ``x`` (the
    dual operator rotates) where one rebuilt from the live iterate keeps CG
    converging. It composes with ``recycle`` (unchanged deflation on top of the
    rebuilt first level) and, when the implicit solve resolves to cg, seeds the
    implicit preconditioner from the state at the converged solution unless an
    explicit ``implicit_preconditioner`` overrides it.

    ``solve(...).x`` has a custom implicit AD rule with respect to ``p``. By
    default, ``implicit_solver="auto"`` uses a fully matrix-free CG implicit
    solve when the forward solver is ``linear_solver="cg"``, and otherwise uses
    the legacy dense Cholesky solve. Pass ``implicit_solver="cholesky"`` to
    force the dense implicit rule, or ``implicit_solver="cg"`` to force the
    matrix-free rule. A cg-resolved implicit solve likewise requires
    ``implicit_preconditioner``, an approximation of the UNDAMPED
    ``(J P J')^{-1} v``, taking either ``(v)`` or ``(v, damping)`` -- a
    callable REQUIRING the damping argument is called with an explicit
    zero, and one where it has a default passes through unchanged, so every
    shipped dual helper serves both hooks directly except
    ``pad_dual_preconditioner``, which divides by the live damping and is
    rejected here at construction.

    ``implicit_penalty`` regularizes the DENSE implicit rule only (the cg
    implicit rule ignores it): the implicit dual is factored as
    ``J P J' + implicit_penalty * trace(J P J') I_m``. Redundant residual
    rows at the returned solution -- e.g. a simulated trajectory settled onto
    its steady state -- make the undamped dual singular and the unregularized
    factorization non-finite; for such consistent systems the ridge returns
    the minimum-norm tangent with an O(``implicit_penalty * m``) relative
    bias. The default ``None`` resolves to ``1e-12`` for a float64 dual solve
    and ``1e-6`` for float32 (after any ``dual_solve_dtype`` promotion) --
    empirically orders of magnitude above the factorization noise floor of
    near-duplicate rows and below any visible tangent bias; pass ``0.0`` to
    restore the exact unregularized rule, whose non-finite tangents signal a
    singular dual loudly.

    ``dual_solve_dtype=jnp.float64`` promotes the dual (Gram) solve of the
    dense cholesky paths -- the forward ``linear_solver="cholesky"`` branch
    and the dense implicit rule -- to float64: ``J'`` is cast wide before the
    metric solve, the m x m assembly, factorization, and triangular solves
    run wide, and only the returned step/tangent is cast back, so the model,
    residual, and every output stay at the residual dtype. Forming
    ``J P J'`` squares the condition number of the whitened Jacobian
    ``J S`` (``S S' = P``), which is what makes the dense paths
    float32-fragile; the promotion buys full-x64 robustness for the dual
    solve at the cost of up to two wide n x m intermediates and roughly
    1.4x per cholesky update measured at m=100, n=2000 with a trivial
    residual (real residual/Jacobian costs dominate and stay float32).
    ``metric.solve`` receives the promoted dtype and must return it
    (jnp-composed callbacks promote automatically; an iterative callback
    like ``metric_from_shifted_matvec`` then runs its inner CG in float64
    with the float64 default tolerance, at callback-dependent cost).
    Requires x64 support to be enabled -- which by itself leaves explicitly
    float32 data in float32.
    """

    def __init__(
        self,
        residual_fn,
        *,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        max_damping=None,
        linear_solver="cholesky",
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=8,
        dual_preconditioner=None,
        preconditioner_factory=None,
        whitened_preconditioner=None,
        implicit_solver="auto",
        implicit_tol=None,
        implicit_atol=0.0,
        implicit_maxiter=None,
        implicit_preconditioner=None,
        implicit_penalty=None,
        dual_solve_dtype=None,
        metric=None,
        has_aux=False,
        cache_jacobian=True,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
        recycle=None,
    ):
        canonical_residual, residual_arity = canonicalize_residual(residual_fn)
        if linear_solver not in ("cholesky", "qr", "augmented_qr", "cg", "lsmr"):
            raise ValueError(f"unknown linear_solver: {linear_solver}")
        if init_damping <= 0:
            raise ValueError("init_damping must be positive")
        if damping_decrease <= 0:
            raise ValueError("damping_decrease must be positive")
        if damping_increase <= 0:
            raise ValueError("damping_increase must be positive")
        if max_damping is not None and max_damping < init_damping:
            raise ValueError("max_damping must be at least init_damping")
        if iterative_tol < 0:
            raise ValueError("iterative_tol must be nonnegative")
        if iterative_atol < 0:
            raise ValueError("iterative_atol must be nonnegative")
        if iterative_maxiter is not None and iterative_maxiter <= 0:
            raise ValueError("iterative_maxiter must be positive or None")
        if iterative_tol == 0 and iterative_atol == 0 and iterative_maxiter is None:
            raise ValueError(
                "iterative_maxiter must be set when both iterative tolerances are zero"
            )
        if dual_preconditioner is not None and linear_solver != "cg":
            raise ValueError('dual_preconditioner requires linear_solver="cg"')
        if preconditioner_factory is not None:
            if linear_solver != "cg":
                raise ValueError('preconditioner_factory requires linear_solver="cg"')
            if dual_preconditioner is not None:
                raise ValueError(
                    "pass exactly one of dual_preconditioner or "
                    "preconditioner_factory for a cg linear_solver, not both"
                )
        if whitened_preconditioner is not None and linear_solver != "lsmr":
            raise ValueError('whitened_preconditioner requires linear_solver="lsmr"')
        if recycle is not None:
            if linear_solver != "cg":
                raise ValueError('recycle requires linear_solver="cg"')
            if not isinstance(recycle, RecycleConfig):
                raise TypeError("recycle must be a RecycleConfig or None")
        if implicit_solver not in ("auto", "cholesky", "cg"):
            raise ValueError(f"unknown implicit_solver: {implicit_solver}")
        if implicit_tol is not None and implicit_tol < 0:
            raise ValueError("implicit_tol must be nonnegative or None")
        if implicit_atol < 0:
            raise ValueError("implicit_atol must be nonnegative")
        if implicit_maxiter is not None and implicit_maxiter <= 0:
            raise ValueError("implicit_maxiter must be positive or None")
        if implicit_penalty is not None and implicit_penalty < 0:
            raise ValueError("implicit_penalty must be nonnegative or None")
        if implicit_tol == 0 and implicit_atol == 0 and implicit_maxiter is None:
            raise ValueError(
                "implicit_maxiter must be set when both implicit tolerances are zero"
            )
        resolved_implicit_solver = (
            "cg"
            if implicit_solver == "cg"
            or (implicit_solver == "auto" and linear_solver == "cg")
            else "cholesky"
        )
        if implicit_preconditioner is not None and resolved_implicit_solver != "cg":
            raise ValueError(
                'implicit_preconditioner requires implicit_solver="cg" '
                'or implicit_solver="auto" with linear_solver="cg"'
            )
        missing_dual_preconditioner = (
            linear_solver == "cg"
            and dual_preconditioner is None
            and preconditioner_factory is None
        )
        # A factory serves the implicit hook too (undamped apply at the solution),
        # so it satisfies the cg implicit requirement like an explicit one.
        missing_implicit_preconditioner = (
            resolved_implicit_solver == "cg"
            and implicit_preconditioner is None
            and preconditioner_factory is None
        )
        if missing_dual_preconditioner and missing_implicit_preconditioner:
            raise ValueError(
                'linear_solver="cg" requires dual_preconditioner (or '
                "preconditioner_factory), and the cg-resolved implicit solver "
                '("cg", or "auto" alongside a cg forward solver) requires '
                "implicit_preconditioner; pass identity_preconditioner() for "
                'either to run unpreconditioned CG, or implicit_solver="cholesky" '
                "for the dense implicit rule"
            )
        if missing_dual_preconditioner:
            raise ValueError(
                'linear_solver="cg" requires dual_preconditioner or '
                "preconditioner_factory; pass identity_preconditioner() to run "
                "unpreconditioned CG"
            )
        if missing_implicit_preconditioner:
            raise ValueError(
                'implicit_solver="cg" requires implicit_preconditioner '
                '(implicit_solver="auto" resolves to cg when '
                'linear_solver="cg"); pass identity_preconditioner() to run '
                'unpreconditioned CG, or implicit_solver="cholesky" for the '
                "dense implicit rule"
            )
        if dual_solve_dtype is not None:
            if jnp.dtype(dual_solve_dtype) != jnp.dtype(jnp.float64):
                raise ValueError("dual_solve_dtype must be None or jnp.float64")
            if linear_solver != "cholesky" and resolved_implicit_solver != "cholesky":
                raise ValueError(
                    "dual_solve_dtype promotes only the dense cholesky paths; "
                    'it requires linear_solver="cholesky" or a '
                    "cholesky-resolved implicit solver"
                )
            if not jax.config.jax_enable_x64:
                raise ValueError(
                    "dual_solve_dtype=jnp.float64 requires x64 support; call "
                    'jax.config.update("jax_enable_x64", True) at startup '
                    "(explicitly float32 problem data stays float32)"
                )
        if metric is None:
            metric = Metric()
        has_custom_metric = any(
            cb is not None
            for cb in (
                metric.solve,
                metric.norm,
                metric.inv_sqrt,
                metric.inv_sqrt_transpose,
            )
        )
        if has_custom_metric and linear_solver in ("cholesky", "cg"):
            if metric.solve is None:
                raise ValueError(
                    f'linear_solver="{linear_solver}" with a custom metric requires '
                    "metric.solve"
                )
        if has_custom_metric and linear_solver in ("qr", "augmented_qr", "lsmr"):
            if metric.inv_sqrt is None or metric.inv_sqrt_transpose is None:
                raise ValueError(
                    f'linear_solver="{linear_solver}" with a custom metric requires '
                    "metric.inv_sqrt and metric.inv_sqrt_transpose"
                )
        if has_custom_metric and geodesic_acceleration and metric.norm is None:
            raise ValueError(
                "geodesic_acceleration (on by default) with a custom metric "
                "requires metric.norm; provide it or pass "
                "geodesic_acceleration=False"
            )
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.max_damping = max_damping
        self.linear_solver = linear_solver
        self.iterative_tol = iterative_tol
        self.iterative_atol = iterative_atol
        self.iterative_maxiter = iterative_maxiter
        self.dual_preconditioner = dual_preconditioner
        self.preconditioner_factory = preconditioner_factory
        self.whitened_preconditioner = whitened_preconditioner
        self.implicit_solver = implicit_solver
        self.implicit_tol = implicit_tol
        self.implicit_atol = implicit_atol
        self.implicit_maxiter = implicit_maxiter
        self.implicit_preconditioner = (
            None
            if implicit_preconditioner is None
            else canonicalize_implicit_preconditioner(implicit_preconditioner)
        )
        self.implicit_penalty = implicit_penalty
        self.dual_solve_dtype = (
            None if dual_solve_dtype is None else jnp.dtype(dual_solve_dtype)
        )
        self._resolved_implicit_solver = resolved_implicit_solver
        self.metric = metric
        # Only the dense cholesky path materializes J', so the flag is inert
        # for the other solvers.
        self.cache_jacobian = cache_jacobian and linear_solver == "cholesky"
        self.has_aux = has_aux
        self._has_custom_metric = has_custom_metric
        self._has_metric_solve = metric.solve is not None
        self.metric_solve = (lambda x: x) if metric.solve is None else metric.solve
        self.metric_norm = (
            (lambda x: jnp.linalg.norm(x)) if metric.norm is None else metric.norm
        )
        self.metric_inv_sqrt = (
            (lambda x: x) if metric.inv_sqrt is None else metric.inv_sqrt
        )
        self.metric_inv_sqrt_transpose = (
            (lambda x: x)
            if metric.inv_sqrt_transpose is None
            else metric.inv_sqrt_transpose
        )
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        self.recycle = recycle
        # Value-based identity: the jitted solve loop marks the solver itself
        # static, so equal-config solvers built around the same residual (and
        # metric/preconditioner objects) share the compiled loop across
        # instances instead of retracing once per construction. Keyed on the
        # constructor arguments -- every derived attribute is a function of them.
        self._static_key = tuple(
            _static_key_component(value)
            for value in (
                residual_fn,
                init_damping,
                damping_decrease,
                damping_increase,
                max_damping,
                linear_solver,
                iterative_tol,
                iterative_atol,
                iterative_maxiter,
                dual_preconditioner,
                preconditioner_factory,
                whitened_preconditioner,
                implicit_solver,
                implicit_tol,
                implicit_atol,
                implicit_maxiter,
                implicit_preconditioner,
                implicit_penalty,
                self.dual_solve_dtype,
                metric,
                has_aux,
                self.cache_jacobian,
                geodesic_acceleration,
                geodesic_acceptance_ratio,
                recycle,
            )
        )
        self._static_hash = hash(self._static_key)

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._static_key == other._static_key

    def __hash__(self):
        return self._static_hash

    def hyperparams(self, dtype=None):
        """``LMHyperparams`` built from the constructor values."""
        return LMHyperparams(
            jnp.asarray(self.damping_decrease, dtype=dtype),
            jnp.asarray(self.damping_increase, dtype=dtype),
            None
            if self.max_damping is None
            else jnp.asarray(self.max_damping, dtype=dtype),
            jnp.asarray(self.geodesic_acceptance_ratio, dtype=dtype),
            jnp.asarray(self.iterative_tol, dtype=dtype),
            jnp.asarray(self.iterative_atol, dtype=dtype),
            None
            if self.iterative_maxiter is None
            else jnp.asarray(self.iterative_maxiter, dtype=jnp.int32),
        )

    def init(self, x0, args=None, *, p=None):
        # One residual evaluation types the damping to match what update()
        # returns (keeping the jit signature and solve-loop carry stable) and
        # sizes the Jacobian cache buffers when cache_jacobian=True. hyper
        # stays None so manual update() loops carry no extra buffers; solve()
        # populates it for its callbacks.
        self._check_residual_args(args, p)
        residual, aux = self._residual_and_aux(x0, args, p)
        damping = jnp.asarray(self.init_damping, dtype=residual.dtype)
        recycle = self._init_recycle_state(residual)
        precond, precond_valid = self._init_precond(x0, args, p)
        if not self.cache_jacobian:
            return LMState(
                damping,
                recycle=recycle,
                precond=precond,
                precond_valid=precond_valid,
            )
        theta, _ = ravel_pytree(x0)
        return LMState(
            damping,
            jnp.zeros(residual.shape, dtype=residual.dtype),
            jnp.zeros((theta.size, residual.size), dtype=residual.dtype),
            jnp.asarray(False, dtype=jnp.bool_),
            jax.tree.map(jnp.zeros_like, aux),
            recycle=recycle,
            precond=precond,
            precond_valid=precond_valid,
        )

    def _init_precond(self, x0, args, p):
        # State built at x0 and valid there: the first update reuses it (x has
        # not moved yet), so init pays the one build and the first step does not
        # rebuild. None for the default path.
        if self.preconditioner_factory is None:
            return None, None
        state = jax.lax.stop_gradient(self.preconditioner_factory.prepare(x0, args, p))
        return state, jnp.asarray(True, dtype=jnp.bool_)

    def _init_recycle_state(self, residual):
        # Cold recycle state sized from the dual dimension m = residual.size:
        # zero basis and zero warm starts, valid=False. The zero-U invariant
        # makes the first deflated solve a pure P-only PCG (no branch).
        if self.recycle is None:
            return None
        m = residual.size
        k = self.recycle.rank
        w = self.recycle.resolved_window
        if k > m or w > m:
            raise ValueError(
                f"recycle rank ({k}) and window ({w}) must be <= the dual "
                f"dimension m ({m})"
            )
        dtype = residual.dtype
        return RecycleState(
            U=jnp.zeros((m, k), dtype=dtype),
            dual_velocity=jnp.zeros(m, dtype=dtype),
            dual_accel=jnp.zeros(m, dtype=dtype),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            iterations=jnp.zeros((), dtype=jnp.int32),
            residual_norm=jnp.zeros((), dtype=dtype),
        )

    def _residual_and_aux(self, x, args, p):
        if self.has_aux:
            value, aux = self.residual_fn(x, args, p)
            # aux rides through the jitted loop carry and the implicit-AD
            # zero-tangent map, so non-numeric leaves can never work; fail
            # here with a clear message instead of a dtype error deep in
            # the trace.
            for leaf in jax.tree.leaves(aux):
                if not isinstance(
                    leaf, (jax.Array, np.ndarray, np.generic, bool, int, float, complex)
                ):
                    raise TypeError(
                        "has_aux=True: aux leaves must be JAX numeric types "
                        f"(arrays or scalars); got {type(leaf).__name__}"
                    )
            return jnp.ravel(value), aux
        return jnp.ravel(self.residual_fn(x, args, p)), None

    def _initial_info(self, x, lm_state, args, p):
        # grad_norm is a +inf sentinel (computing it would cost a Jacobian
        # before the first step) and step_norm is zero; neither can satisfy
        # gtol/xtol before any update has run.
        residual, aux = self._residual_and_aux(x, args, p)
        loss = jnp.sum(residual**2)
        zero = jnp.zeros((), dtype=residual.dtype)
        one = jnp.ones((), dtype=residual.dtype)
        return LMInfo(
            loss,
            loss,
            loss,
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(lm_state.damping, dtype=residual.dtype),
            one,
            jnp.asarray(False, dtype=jnp.bool_),
            zero,
            jnp.asarray(jnp.inf, dtype=residual.dtype),
            zero,
            aux,
        )

    def update(self, x, lm_state, args=None, p=None):
        self._check_residual_args(args, p)
        # Linearize at x: flatten the pytree and view the residual over theta.
        theta, unravel = ravel_pytree(x)

        if self.has_aux:

            def residual_flat(th):
                value, aux = self.residual_fn(unravel(th), args, p)
                return jnp.ravel(value), aux

            def residual_value(th):
                return residual_flat(th)[0]

        else:

            def residual_flat(th):
                return jnp.ravel(self.residual_fn(unravel(th), args, p))

            residual_value = residual_flat

        # Build J': matrix-free JVP/VJP closures for cg and lsmr; m VJP passes for
        # the dense paths, reused from the cache after a rejected step.
        if self.linear_solver in ("cg", "lsmr"):
            if self.has_aux:
                resid, jvp_fn, aux = jax.linearize(residual_flat, theta, has_aux=True)
            else:
                resid, jvp_fn = jax.linearize(residual_flat, theta)
                aux = None
        elif self.cache_jacobian:
            if lm_state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True but the lm_state has no Jacobian cache; "
                    "create the lm_state with init(x, args, p=p)"
                )

            def compute_resid_and_jt(_):
                if self.has_aux:
                    resid, pullback, aux = jax.vjp(residual_flat, theta, has_aux=True)
                else:
                    resid, pullback = jax.vjp(residual_flat, theta)
                    aux = None
                residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
                Jt = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
                    residual_basis
                ).T
                return resid, Jt, aux

            def reuse_resid_and_jt(_):
                return lm_state.resid, lm_state.Jt, lm_state.aux

            resid, Jt, aux = jax.lax.cond(
                lm_state.jacobian_valid,
                reuse_resid_and_jt,
                compute_resid_and_jt,
                operand=None,
            )
        else:
            if self.has_aux:
                resid, pullback, aux = jax.vjp(residual_flat, theta, has_aux=True)
            else:
                resid, pullback = jax.vjp(residual_flat, theta)
                aux = None
        damping = jnp.asarray(lm_state.damping, dtype=resid.dtype)
        # Traced hyperparameters from the lm_state when present (resettable by
        # solve callbacks); the None fallback compiles to the same constants
        # as reading the constructor values directly.
        hyper = (
            lm_state.hyper
            if lm_state.hyper is not None
            else self.hyperparams(resid.dtype)
        )
        damping_decrease = jnp.asarray(hyper.damping_decrease, dtype=resid.dtype)
        damping_increase = jnp.asarray(hyper.damping_increase, dtype=resid.dtype)

        # Recycled dual solutions + harvest, captured in call order by the cg
        # branch's solve_step when recycling is active; consumed below to build
        # the next RecycleState. Stays None for every non-recycled path.
        recycle_solves = None
        if self.linear_solver == "cg":
            transpose_fn = jax.linear_transpose(jvp_fn, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

            grad = JT(resid)
            # Typed tolerances keep CG's scalars in the residual dtype under x64.
            cg_tol = jnp.asarray(hyper.iterative_tol, dtype=resid.dtype)
            cg_atol = jnp.asarray(hyper.iterative_atol, dtype=resid.dtype)

            def gram_matvec(cotangent):
                return jvp_fn(self.metric_solve(JT(cotangent))) + damping * cotangent

            # θ-adaptive preconditioner: rebuild the state from the pre-step x
            # (this step's dual linearization point), or reuse the carried state
            # when the previous step was rejected (x did not move) -- the
            # jacobian_valid lax.cond pattern, so the rebuild cost is skipped
            # exactly when nothing changed. apply reads the live damping, so a
            # reused state is still correct at the new damping.
            if self.preconditioner_factory is not None:
                if lm_state.precond is None:
                    raise ValueError(
                        "preconditioner_factory is set but the lm_state has no "
                        "preconditioner state; create the lm_state with "
                        "init(x, args, p=p)"
                    )
                precond_state = jax.lax.cond(
                    lm_state.precond_valid,
                    lambda _: lm_state.precond,
                    lambda _: jax.lax.stop_gradient(
                        self.preconditioner_factory.prepare(x, args, p)
                    ),
                    operand=None,
                )

                def cg_preconditioner(cotangent):
                    return self.preconditioner_factory.apply(
                        precond_state, cotangent, damping
                    )

            else:

                def cg_preconditioner(cotangent):
                    return self.dual_preconditioner(cotangent, damping)

            if self.recycle is not None and lm_state.recycle is not None:
                # Recycled/deflated dual solve: build the coarse operator once on
                # the current damped gram_matvec (reused across velocity and the
                # geodesic-acceleration RHS), harvest the next basis from the
                # velocity solve only, and capture the dual solutions (in call
                # order: velocity first, acceleration second) so the new
                # RecycleState can thread out through the returned lm_state.
                recycle = lm_state.recycle
                rank = self.recycle.rank
                window = self.recycle.resolved_window
                warm_start = self.recycle.warm_start
                reorthogonalize = self.recycle.reorthogonalize
                _, e_factor = build_coarse_operator(
                    gram_matvec, recycle.U, ridge=self.recycle.ridge
                )
                recycle_solves = []

                def solve_step(rhs):
                    is_velocity = len(recycle_solves) == 0
                    x0 = None
                    if warm_start:
                        x0 = (
                            recycle.dual_velocity if is_velocity else recycle.dual_accel
                        )
                    dual_solution, harvest = deflated_pcg(
                        gram_matvec,
                        rhs,
                        U=recycle.U,
                        E_factor=e_factor,
                        M=cg_preconditioner,
                        x0=x0,
                        tol=cg_tol,
                        atol=cg_atol,
                        maxiter=hyper.iterative_maxiter,
                        window=window,
                        rank=rank,
                        reorthogonalize=reorthogonalize,
                        harvest=is_velocity,
                    )
                    recycle_solves.append((dual_solution, harvest))
                    return -self.metric_solve(JT(dual_solution))

            else:

                def solve_step(rhs):
                    dual_solution, _ = jsp_sparse_linalg.cg(
                        gram_matvec,
                        rhs,
                        tol=cg_tol,
                        atol=cg_atol,
                        maxiter=hyper.iterative_maxiter,
                        M=cg_preconditioner,
                    )
                    return -self.metric_solve(JT(dual_solution))

        elif self.linear_solver == "lsmr":
            # Matrix-free whitened damped LS: min_u ||r + B u||^2 + damping ||u||^2
            # with B = J S (S = metric.inv_sqrt). Working on B (condition sqrt of
            # the cg dual's J M^{-1} J' + damping I) restores endgame accuracy in
            # the selection-critical slow directions where the squared dual solve
            # bottoms out at eps * cond. Step s = S u.
            transpose_fn = jax.linear_transpose(jvp_fn, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

            grad = JT(resid)
            # Hook -> Fong-Saunders/scipy LSMR name mapping (kept standard in
            # lsmr.py): iterative_tol becomes LSMR's atol (relative, scaled by
            # normar0 = ||A'b||, i.e. operator-scaled) and iterative_atol becomes
            # btol (absolute). Passed as lsmr_solve's atol/btol positionals below.
            lsmr_tol = jnp.asarray(hyper.iterative_tol, dtype=resid.dtype)
            lsmr_atol = jnp.asarray(hyper.iterative_atol, dtype=resid.dtype)
            sqrt_damping = jnp.sqrt(damping)
            m = resid.shape[0]
            n = theta.shape[0]
            # None (uncapped) has no meaning for a fixed-shape loop; a
            # tolerance-only stop still needs a hard cap. min(m, n) is the
            # bidiagonalization's exact-arithmetic termination bound.
            lsmr_maxiter = (
                hyper.iterative_maxiter
                if hyper.iterative_maxiter is not None
                else 4 * min(m, n)
            )

            # Parameter-space right-preconditioner R^{-1} (whitened_preconditioner):
            # run LSMR on the preconditioned operator A_r = B R^{-1} in the z
            # variable, then un-precondition u = R^{-1} z. A good R clusters the
            # spectrum of B R^{-1} and cuts iterations; damp then regularizes in the
            # R'R metric (a documented surrogate). None -> plain LSMR.
            if self.whitened_preconditioner is not None:

                def apply_Rinv(v):
                    return self.whitened_preconditioner.solve(v, damping)

                def apply_RinvT(w):
                    return self.whitened_preconditioner.solve_transpose(w, damping)

            else:

                def apply_Rinv(v):
                    return v

                def apply_RinvT(w):
                    return w

            def A_matvec(z):  # A_r = B R^{-1}
                return jvp_fn(self.metric_inv_sqrt(apply_Rinv(z)))

            def At_matvec(w):  # A_r' = R^{-T} B'
                return apply_RinvT(self.metric_inv_sqrt_transpose(JT(w)))

            # N = A_r'A_r + damping I: the SPD normal operator custom_linear_solve
            # differentiates through (the implicit rule for the preconditioned
            # damped LS solution in z-space).
            def N_matvec(z):
                return At_matvec(A_matvec(z)) + damping * z

            def solve_N(_, c):
                # Solve N z = c for arbitrary c via LSMR on the augmented operator
                # [A_r; sqrt(damping) I] with rhs [0; c / sqrt(damping)]: its normal
                # equations are exactly N z = c, at condition sqrt(cond(N)).
                # Stopping is measured on the preconditioned operator. The forward c
                # and every cotangent RHS route through here.
                def A_aug(zz):
                    return jnp.concatenate([A_matvec(zz), sqrt_damping * zz])

                def At_aug(ww):
                    return At_matvec(ww[:m]) + sqrt_damping * ww[m:]

                b_aug = jnp.concatenate([jnp.zeros(m, resid.dtype), c / sqrt_damping])
                z, _ = lsmr_solve(
                    A_aug,
                    At_aug,
                    b_aug,
                    jnp.zeros((), resid.dtype),
                    lsmr_tol,
                    lsmr_atol,
                    lsmr_maxiter,
                    n,
                )
                return z

            def solve_step(rhs):
                # z solves min ||A_r z + rhs||^2 + damping ||z||^2; c = A_r'(-rhs),
                # and the step un-preconditions the final iterate: u = R^{-1} z.
                c = At_matvec(-rhs)
                z = jax.lax.custom_linear_solve(
                    N_matvec, c, solve=solve_N, transpose_solve=solve_N, symmetric=True
                )
                return self.metric_inv_sqrt(apply_Rinv(z))

        else:
            if not self.cache_jacobian:
                residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
                Jt = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
                    residual_basis
                ).T
            grad = Jt @ resid

            if self.linear_solver == "augmented_qr":
                transformed_Jt = self.metric_inv_sqrt_transpose(Jt)
                n = transformed_Jt.shape[0]
                augmented_matrix = jnp.concatenate(
                    (
                        transformed_Jt.T,
                        jnp.sqrt(damping) * jnp.eye(n, dtype=resid.dtype),
                    ),
                    axis=0,
                )
                Q, R = jnp.linalg.qr(augmented_matrix, mode="reduced")

                def solve_step(rhs):
                    augmented_rhs = jnp.concatenate(
                        (-rhs, jnp.zeros(n, dtype=rhs.dtype))
                    )
                    transformed_step = jsp_linalg.solve_triangular(
                        R,
                        Q.T @ augmented_rhs,
                    )
                    return self.metric_inv_sqrt(transformed_step)

            elif self.linear_solver == "qr":
                # Whitened residual-dimension reduction followed by augmented QR.
                transformed_Jt = self.metric_inv_sqrt_transpose(Jt)
                if transformed_Jt.shape[0] >= transformed_Jt.shape[1]:
                    R = jnp.linalg.qr(transformed_Jt, mode="r")
                    basis_eye = jnp.eye(R.shape[0], dtype=resid.dtype)
                    augmented_matrix = jnp.concatenate(
                        (R.T, jnp.sqrt(damping) * basis_eye),
                        axis=0,
                    )
                    Qa, Ra = jnp.linalg.qr(augmented_matrix, mode="reduced")

                    def solve_step(rhs):
                        augmented_rhs = jnp.concatenate(
                            (-rhs, jnp.zeros(R.shape[0], dtype=rhs.dtype))
                        )
                        z = jsp_linalg.solve_triangular(
                            Ra,
                            Qa.T @ augmented_rhs,
                        )
                        y = jsp_linalg.solve_triangular(R, z)
                        return self.metric_inv_sqrt(transformed_Jt @ y)

                else:
                    Q, R = jnp.linalg.qr(transformed_Jt, mode="reduced")
                    basis_eye = jnp.eye(R.shape[0], dtype=resid.dtype)
                    augmented_matrix = jnp.concatenate(
                        (R.T, jnp.sqrt(damping) * basis_eye),
                        axis=0,
                    )
                    Qa, Ra = jnp.linalg.qr(augmented_matrix, mode="reduced")

                    def solve_step(rhs):
                        augmented_rhs = jnp.concatenate(
                            (-rhs, jnp.zeros(R.shape[0], dtype=rhs.dtype))
                        )
                        z = jsp_linalg.solve_triangular(
                            Ra,
                            Qa.T @ augmented_rhs,
                        )
                        return self.metric_inv_sqrt(Q @ z)

            else:
                # Damped Gram factorization: cholesky of J P J' + damping I.
                # With dual_solve_dtype the whole dual pipeline runs wide:
                # J' is promoted BEFORE the metric solve (a stiff metric's
                # 1/eps rows amplify float32 rounding of P J' into O(1) Gram
                # errors -- promoting only the assembly was measured ~4
                # digits worse), and the dual solution stays wide through the
                # final product. Only the returned step is cast back, so it
                # keeps the residual dtype. metric.solve therefore receives
                # the promoted dtype; jnp-composed callbacks promote
                # automatically.
                dual_dtype = (
                    resid.dtype
                    if self.dual_solve_dtype is None
                    else self.dual_solve_dtype
                )
                transposed_jacobian = Jt.astype(dual_dtype)
                gram_step_left = self.metric_solve(transposed_jacobian)
                linear_matrix = transposed_jacobian.T @ gram_step_left
                linear_matrix = linear_matrix + jnp.asarray(
                    damping, dtype=dual_dtype
                ) * jnp.eye(resid.shape[0], dtype=dual_dtype)

                linear_factor = jsp_linalg.cho_factor(linear_matrix)

                def solve_step(rhs):
                    dual_solution = jsp_linalg.cho_solve(
                        linear_factor, rhs.astype(dual_dtype)
                    )
                    step = -gram_step_left @ dual_solution
                    return step.astype(resid.dtype)

        # Dual solve for the first-order step (velocity).
        velocity = solve_step(resid)
        resid_velocity = residual_value(theta + velocity)
        loss_old = jnp.sum(resid**2)
        loss_velocity = jnp.sum(resid_velocity**2)
        zero = jnp.zeros((), dtype=resid.dtype)

        # Geodesic second-order correction, solved with the same factorization.
        if self.geodesic_acceleration:
            geodesic_acceptance_ratio = jnp.asarray(
                hyper.geodesic_acceptance_ratio, dtype=resid.dtype
            )

            def first_jvp(th):
                # [1] is the tangent with and without has_aux.
                return jax.jvp(residual_flat, (th,), (velocity,), has_aux=self.has_aux)[
                    1
                ]

            f_vv = jax.jvp(first_jvp, (theta,), (velocity,))[1]
            acceleration = solve_step(f_vv)
            accelerated_step = velocity + 0.5 * acceleration
            acceleration_ratio = (
                2.0
                * self.metric_norm(acceleration)
                / (self.metric_norm(velocity) + jnp.finfo(resid.dtype).eps)
            )
            ratio_accepted = (
                (geodesic_acceptance_ratio > zero)
                & (acceleration_ratio > zero)
                & (acceleration_ratio <= geodesic_acceptance_ratio)
            )

            def accelerated_loss(_):
                resid_accelerated = residual_value(theta + accelerated_step)
                return jnp.sum(resid_accelerated**2)

            loss_accelerated = jax.lax.cond(
                ratio_accepted,
                accelerated_loss,
                lambda _: jnp.asarray(jnp.inf, dtype=resid.dtype),
                operand=None,
            )
            used_geodesic = ratio_accepted & (loss_accelerated <= loss_velocity)
            step = jnp.where(used_geodesic, accelerated_step, velocity)
            loss_candidate = jnp.where(used_geodesic, loss_accelerated, loss_velocity)
        else:
            step = velocity
            loss_candidate = loss_velocity
            used_geodesic = jnp.asarray(False)
            acceleration_ratio = zero

        # Accept iff the sum of squared residuals decreases and is finite.
        improved = jnp.isfinite(loss_candidate) & (loss_candidate < loss_old)
        theta_new = jnp.where(improved, theta + step, theta)
        # Damping update: decrease on acceptance, increase on rejection.
        damping_factor = jnp.where(improved, damping_decrease, damping_increase)
        new_damping = damping * damping_factor
        if hyper.max_damping is not None:
            new_damping = jnp.minimum(
                new_damping, jnp.asarray(hyper.max_damping, dtype=resid.dtype)
            )
        loss = jnp.where(improved, loss_candidate, loss_old)
        # New recycle state: the velocity solve's harvested basis and the (stop-
        # gradient'd) dual solutions become warm starts for the next step. Threads
        # through unchanged (None) for every non-recycled path. dual_accel is the
        # acceleration dual when geodesic acceleration ran, else zeros -- a single
        # stable carry shape independent of the geodesic flag.
        new_recycle = lm_state.recycle
        if recycle_solves is not None:
            velocity_dual, velocity_harvest = recycle_solves[0]
            if len(recycle_solves) > 1:
                accel_dual = recycle_solves[1][0]
            else:
                accel_dual = jnp.zeros_like(velocity_dual)
            new_recycle = RecycleState(
                U=jax.lax.stop_gradient(velocity_harvest.basis),
                dual_velocity=jax.lax.stop_gradient(velocity_dual),
                dual_accel=jax.lax.stop_gradient(accel_dual),
                valid=jnp.asarray(True, dtype=jnp.bool_),
                iterations=jax.lax.stop_gradient(velocity_harvest.iterations),
                residual_norm=jax.lax.stop_gradient(velocity_harvest.residual_norm),
            )
        # Carry the state built at this step's pre-step x; precond_valid = ~improved
        # marks it reusable next step exactly when the step was rejected (x did
        # not move). On acceptance the carried state is stale but shape-stable --
        # the flag forces a rebuild at the new x before it is applied.
        new_precond = lm_state.precond
        new_precond_valid = lm_state.precond_valid
        if self.preconditioner_factory is not None:
            new_precond = precond_state
            new_precond_valid = ~improved
        # The input hyper (not the fallback) passes through so the loop carry
        # structure and dtypes are stable.
        if self.cache_jacobian:
            new_lm_state = LMState(
                new_damping,
                resid,
                Jt,
                ~improved,
                aux,
                lm_state.hyper,
                new_recycle,
                new_precond,
                new_precond_valid,
            )
        else:
            new_lm_state = LMState(
                new_damping,
                hyper=lm_state.hyper,
                recycle=new_recycle,
                precond=new_precond,
                precond_valid=new_precond_valid,
            )
        return (
            unravel(theta_new),
            new_lm_state,
            LMInfo(
                loss,
                loss_old,
                loss_candidate,
                improved,
                new_damping,
                damping_factor,
                used_geodesic,
                acceleration_ratio,
                jnp.linalg.norm(grad),
                jnp.linalg.norm(step),
                aux,
            ),
        )

    def solve(
        self,
        x0,
        args=None,
        *,
        p=None,
        lm_state=None,
        max_steps=256,
        atol=0.0,
        gtol=0.0,
        xtol=0.0,
        callback=None,
        user_state=None,
        save_steps=False,
        multi_start=None,
        jit=True,
    ):
        """Run repeated LM updates until a stopping rule fires.

        Parameters are the same as ``update`` plus loop controls. ``max_steps``
        is always enforced. ``atol`` stops when the residual norm is below the
        threshold, ``gtol`` when the gradient norm ``||J' r||`` is below the
        threshold, and ``xtol`` when an accepted step has norm below the
        threshold; each tolerance set to ``0`` disables that check, and all
        three report ``LMStatus.CONVERGED``. ``callback`` receives an
        ``LMSolveContext`` after each step and may return an ``LMSolveAction``
        to stop or to override x/lm_state/args/user_state. ``p`` is passed to
        the residual and callback but cannot be replaced by the action.
        ``save_steps=True`` records the full iterate history onto the result:
        ``x_history`` stacks x0 and every kept post-step iterate along a
        ``(max_steps + 1)`` leading axis (rows beyond ``steps`` are zero
        padding — slice with ``result.steps``), plus the row-aligned
        ``args_history`` (the kept post-action args, recorded even when no
        callback ever replaces them; ``None`` when ``args`` is ``None``) and,
        with ``has_aux``, the row-aligned ``aux_history``. The history buffers
        cost ``(max_steps + 1) x (size(x) + size(args) [+ size(aux)])``
        memory, are differentiation-inert, and (unlike the default) make the
        jitted loop retrace when ``max_steps`` changes, since the buffer shape
        depends on it.

        ``multi_start`` (a ``MultiStart``, default ``None``) retries or
        parallelizes the solve over fresh initial conditions drawn by
        ``multi_start.draw`` and returns the single best result, with
        diagnostics on ``result.multi_start`` (a ``MultiStartInfo``).
        Sequential mode retries only on failure and stops at the first
        success; ``parallel=True`` solves every start under ``vmap`` and
        selects the accepted lane with the lowest loss. Gradients with
        respect to ``p`` flow through the selected solution only, via the
        same implicit rule as a plain solve. With ``multi_start=None``
        nothing changes.
        """
        self._check_residual_args(args, p)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        # Tolerances are traced data inside the loop, so vmapped/traced values
        # skip the concrete-only sign validation.
        if not isinstance(atol, jax.core.Tracer) and atol < 0:
            raise ValueError("atol must be nonnegative")
        if not isinstance(gtol, jax.core.Tracer) and gtol < 0:
            raise ValueError("gtol must be nonnegative")
        if not isinstance(xtol, jax.core.Tracer) and xtol < 0:
            raise ValueError("xtol must be nonnegative")
        if lm_state is None:
            # The loop recasts the damping and hyperparameter dtypes itself; the
            # Jacobian cache (cache_jacobian), the recycle state (sized from the
            # residual), and the preconditioner state (built by prepare) need an
            # eager init() for their shapes.
            if (
                self.cache_jacobian
                or self.recycle is not None
                or self.preconditioner_factory is not None
            ):
                lm_state = self.init(x0, args, p=p)
            else:
                lm_state = LMState(jnp.asarray(self.init_damping))
        if lm_state.hyper is None:
            # Populate here (not in init) so manual update() loops stay lean;
            # inside the loop the extra scalars are loop-carried, not
            # re-dispatched per step.
            lm_state = dataclasses.replace(lm_state, hyper=self.hyperparams())
        # The history buffers need a concrete length, so it is fixed here (like
        # callback and jit, via closure) and the buffers are allocated inside
        # the loop implementations; with save_steps=False nothing changes, and
        # with save_steps=True the buffer shape retraces per max_steps anyway.
        history_len = max_steps + 1 if save_steps else None

        if multi_start is not None:
            if not isinstance(multi_start, MultiStart):
                raise TypeError("multi_start must be a MultiStart or None")
            num_starts = multi_start.num_starts
            # A single start never draws: normalizing to None keeps the jit
            # cache key independent of the (unused) draw identity. The hooks
            # are jit statics, so unhashable callables get an
            # identity-hashing wrapper (hash/eq by the wrapped function, so
            # repeat calls still share the compilation).
            draw = _hashable_hook(multi_start.draw if num_starts > 1 else None)
            accept = _hashable_hook(multi_start.accept)
            parallel = multi_start.parallel and num_starts > 1
            if draw is not None and jit:
                # Abstract trace only (no RNG, no FLOPs): fail loudly here
                # instead of deep inside the while_loop/vmap carry checks.
                # jit=False validates the first concrete draw instead, so a
                # successful first attempt never invokes draw.
                drawn = jax.eval_shape(draw, multi_start.key, x0, args)
                _check_drawn_types(x0, args, drawn)

            @jax.custom_jvp
            def solve_multi_start_with_implicit_p(
                x, lm_state, args, p, user_state, key, max_steps, atol, gtol, xtol
            ):
                return self._multi_start_impl(
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
                    jit,
                    num_starts,
                    draw,
                    accept,
                    parallel,
                )

            @solve_multi_start_with_implicit_p.defjvp
            def solve_multi_start_with_implicit_p_jvp(primals, tangents):
                p_dot = tangents[3]
                result = solve_multi_start_with_implicit_p(*primals)
                return result, self._implicit_result_tangent(result, p_dot)

            return solve_multi_start_with_implicit_p(
                x0,
                lm_state,
                args,
                p,
                user_state,
                multi_start.key,
                max_steps,
                atol,
                gtol,
                xtol,
            )

        @jax.custom_jvp
        def solve_with_implicit_p(
            x, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
        ):
            return self._solve_impl(
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
                jit,
            )

        @solve_with_implicit_p.defjvp
        def solve_with_implicit_p_jvp(primals, tangents):
            p_dot = tangents[3]
            result = solve_with_implicit_p(*primals)
            return result, self._implicit_result_tangent(result, p_dot)

        return solve_with_implicit_p(
            x0, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
        )

    def _solve_impl(
        self,
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
        jit,
    ):
        if jit:
            return _solve_loop_jit(
                self,
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
            )
        return self._solve_python(
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
        )

    def _multi_start_impl(
        self,
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
        jit,
        num_starts,
        draw,
        accept,
        parallel,
    ):
        if not jit:
            return self._multi_start_python(
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
            )
        if parallel:
            return _multi_start_parallel_jit(
                self,
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
            )
        return _multi_start_sequential_jit(
            self,
            x,
            lm_state,
            args,
            p,
            user_state,
            key,
            jnp.asarray(num_starts, dtype=jnp.int32),
            history_len,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
            draw,
            accept,
        )

    def _multi_start_python(
        self,
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
        accept_fn = _accept_converged if accept is None else accept
        cold = _cold_lm_state(lm_state)

        def run_attempt(x_a, lm_state_a, args_a, attempt):
            result = self._solve_python(
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
            loss = _ranking_loss(self, result, p, callback)
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

    def _implicit_result_tangent(self, result, p_dot):
        # The tangent is a pure function of the returned solution: relinearize
        # the residual at (result.x, result.args, result.p), never reusing the
        # forward iterations. Everything except x, p, and aux -- histories,
        # counters, multi-start diagnostics -- is bookkeeping with zero
        # tangents.
        x_dot = self._implicit_x_tangent_from_p(result.x, result.args, result.p, p_dot)
        zero_result = jax.tree.map(_zero_tangent_leaf, result)
        aux_dot = zero_result.aux
        if self.has_aux and result.p is not None:
            # aux depends on p directly and through the solution x*(p);
            # linearize the aux map at the returned solution with args
            # fixed (the same point where the primal result.aux is
            # evaluated) to account for both paths.
            def aux_at_solution(x_value, p_value):
                return self.residual_fn(x_value, result.args, p_value)[1]

            aux_dot = jax.jvp(aux_at_solution, (result.x, result.p), (x_dot, p_dot))[1]
        return dataclasses.replace(zero_result, x=x_dot, p=p_dot, aux=aux_dot)

    def _implicit_x_tangent_from_p(self, x, args, p, p_dot):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        if self._resolved_implicit_solver == "cg":
            return self._implicit_x_tangent_from_p_cg(x, args, p, p_dot)
        return self._implicit_x_tangent_from_p_cholesky(x, args, p, p_dot)

    def _implicit_x_tangent_from_p_cholesky(self, x, args, p, p_dot):
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)
        residual_basis = jnp.eye(residual.shape[0], dtype=residual.dtype)
        theta_transpose = jax.linear_transpose(theta_jvp, theta)
        Jt = jax.vmap(lambda cotangent: theta_transpose(cotangent)[0])(residual_basis).T

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        # The undamped implicit Gram has no + damping I floor, so it benefits
        # most from dual_solve_dtype promotion; same recipe as the forward
        # cholesky branch (J' promoted before the metric solve, wide
        # factor/solve and final product, only the tangent cast back to the
        # residual dtype).
        dual_dtype = (
            residual.dtype if self.dual_solve_dtype is None else self.dual_solve_dtype
        )
        transposed_jacobian = Jt.astype(dual_dtype)
        gram_step_left = self._metric_inverse(transposed_jacobian)
        gram = transposed_jacobian.T @ gram_step_left
        # Tikhonov ridge scaled by the trace: redundant residual rows at the
        # returned solution (e.g. a simulated trajectory settled onto its steady
        # state) make the undamped dual singular and the factorization non-finite;
        # for such consistent systems the ridge returns the minimum-norm tangent
        # with an O(implicit_penalty * m) relative bias. The defaults sit orders
        # of magnitude from both empirical edges (see the redundant-rows tests):
        # near-duplicate float64 rows need > ~1e-14 to factor while visible
        # tangent bias starts above ~1e-6, and the float32 value stays below the
        # library's float32 tangent tolerances while dominating float32 assembly
        # noise. implicit_penalty=0.0 disables (non-finite on a singular dual).
        penalty = self.implicit_penalty
        if penalty is None:
            penalty = 1e-12 if jnp.dtype(dual_dtype) == jnp.dtype(jnp.float64) else 1e-6
        gram = gram + penalty * jnp.trace(gram) * jnp.eye(
            gram.shape[0], dtype=dual_dtype
        )
        factor = jsp_linalg.cho_factor(gram)
        dual_solution = jsp_linalg.cho_solve(factor, residual_p_dot.astype(dual_dtype))
        theta_dot = -gram_step_left @ dual_solution
        return unravel(theta_dot.astype(residual.dtype))

    def _implicit_x_tangent_from_p_cg(self, x, args, p, p_dot):
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)
        theta_transpose = jax.linear_transpose(theta_jvp, theta)

        def JT(cotangent):
            return theta_transpose(cotangent)[0]

        def gram_matvec(cotangent):
            return theta_jvp(self._metric_inverse(JT(cotangent)))

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        cg_tol = self._implicit_cg_tol(residual.dtype)
        cg_atol = jnp.asarray(self.implicit_atol, dtype=residual.dtype)

        # An explicit implicit_preconditioner wins; otherwise a factory seeds the
        # implicit preconditioner from the state at the converged solution
        # (undamped, since the implicit dual has no damping floor). x/args/p are
        # the traced returned solution, so prepare() yields a traced state (never
        # a closure constant) and repeated solves at different p do not recompile;
        # stop_gradient because the preconditioner never moves the root.
        if self.implicit_preconditioner is not None:
            cg_preconditioner = self.implicit_preconditioner
        elif self.preconditioner_factory is not None:
            precond_state = jax.lax.stop_gradient(
                self.preconditioner_factory.prepare(x, args, p)
            )
            zero_damping = jnp.zeros((), dtype=residual.dtype)

            def cg_preconditioner(v):
                return self.preconditioner_factory.apply(precond_state, v, zero_damping)

        else:
            cg_preconditioner = None

        def solve(matvec, rhs):
            solution, _ = jsp_sparse_linalg.cg(
                matvec,
                rhs,
                tol=cg_tol,
                atol=cg_atol,
                maxiter=self.implicit_maxiter,
                M=cg_preconditioner,
            )
            return solution

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        dual_solution = jax.lax.custom_linear_solve(
            gram_matvec,
            residual_p_dot,
            solve,
            symmetric=True,
        )

        # The final metric inverse acts on tangent data, so VJP transposes
        # it. An iterative metric solve (e.g. metric_from_shifted_matvec)
        # is not transposable by JAX -- its CG captures tol*|b| inside the
        # linear solve's parameters -- but P is self-adjoint by contract,
        # so declare the application as its own transpose: every rule of
        # this custom_linear_solve routes through `solve` (the identity
        # matvec contributes nothing), and with symmetric=True the
        # cotangent pass just EVALUATES metric.solve. custom_linear_solve
        # is used rather than jax.custom_derivatives.linear_call because
        # linear_call has no batching rule, which would break jax.vmap
        # (and vmap-based second derivatives) over differentiated solves.
        theta_dot = -jax.lax.custom_linear_solve(
            lambda v: v,
            JT(dual_solution),
            lambda _, rhs: self._metric_inverse(rhs),
            symmetric=True,
        )
        return unravel(theta_dot)

    def _implicit_cg_tol(self, dtype):
        if self.implicit_tol is not None:
            return jnp.asarray(self.implicit_tol, dtype=dtype)
        default_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return jnp.asarray(default_tol, dtype=dtype)

    def _metric_inverse(self, x):
        if self._has_metric_solve:
            return self.metric_solve(x)
        return self.metric_inv_sqrt(self.metric_inv_sqrt_transpose(x))

    def _action_or_default(self, action):
        if action is None:
            return LMSolveAction()
        return action

    def _apply_action(self, action, x, lm_state, args, user_state):
        action = self._action_or_default(action)
        # The step's diagnostics and the cached Jacobian describe the
        # pre-action (x, args), so both are stale iff the action actually
        # changed the values — a traced comparison, so a jit-style callback
        # that returns the field every step with unchanged values (the
        # jnp.where recipe pattern) changes nothing.
        problem_changed = jnp.asarray(False)
        if action.x is not None:
            problem_changed = problem_changed | _tree_changed(action.x, x)
            x = action.x
        if action.lm_state is not None:
            previous_hyper = lm_state.hyper
            lm_state = action.lm_state
            if self.cache_jacobian and lm_state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True but the callback action returned an "
                    "lm_state without the Jacobian cache; use "
                    "dataclasses.replace(ctx.lm_state, ...) to preserve the "
                    "cache fields"
                )
            if self.recycle is not None and lm_state.recycle is None:
                raise ValueError(
                    "recycle is set but the callback action returned an lm_state "
                    "without the RecycleState; use dataclasses.replace("
                    "ctx.lm_state, ...) to preserve the recycle field (rank and "
                    "window are static and cannot change mid-solve)"
                )
            if self.preconditioner_factory is not None and lm_state.precond is None:
                raise ValueError(
                    "preconditioner_factory is set but the callback action "
                    "returned an lm_state without the preconditioner state; use "
                    "dataclasses.replace(ctx.lm_state, ...) to preserve the "
                    "precond and precond_valid fields"
                )
            # Trace-time guard so the hyper contract fails identically with
            # and without jit (jit would reject the carry mismatch anyway).
            if previous_hyper is not None and (
                lm_state.hyper is None
                or jax.tree_util.tree_structure(previous_hyper)
                != jax.tree_util.tree_structure(lm_state.hyper)
                or [leaf.dtype for leaf in jax.tree_util.tree_leaves(previous_hyper)]
                != [leaf.dtype for leaf in jax.tree_util.tree_leaves(lm_state.hyper)]
            ):
                raise ValueError(
                    "the callback action changed the structure or dtypes of "
                    "lm_state.hyper; reset values with "
                    "dataclasses.replace(ctx.lm_state.hyper, ...) using arrays "
                    "of the same dtype — a knob constructed as None cannot be "
                    "enabled mid-solve"
                )
        if action.args is not None:
            problem_changed = problem_changed | _tree_changed(action.args, args)
            args = action.args
        if action.user_state is not None:
            user_state = action.user_state
        if self.cache_jacobian and (action.x is not None or action.args is not None):
            lm_state = dataclasses.replace(
                lm_state, jacobian_valid=lm_state.jacobian_valid & ~problem_changed
            )
        return action, x, lm_state, args, user_state, problem_changed

    def _check_residual_args(self, args, p):
        # Silently dropping args/p a residual never sees would, in particular,
        # make the implicit derivative with respect to p a silent zero.
        if args is not None and self.residual_arity < 2:
            raise ValueError(
                "args was passed but residual_fn takes only (x); "
                "use residual_fn(x, args)"
            )
        if p is not None and self.residual_arity < 3:
            raise ValueError(
                "p was passed but residual_fn takes no p argument; "
                "use residual_fn(x, args, p)"
            )

    def _converged(self, info, atol, gtol, xtol):
        atol_met = (atol > 0) & (jnp.sqrt(info.loss) < atol)
        gtol_met = (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (xtol > 0) & info.accepted & (info.step_norm < xtol)
        return atol_met | gtol_met | xtol_met

    def _solve_python(
        self,
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
        history = _init_history(self, x, args, p, history_len)
        info = self._initial_info(x, lm_state, args, p)
        lm_state = dataclasses.replace(
            lm_state,
            damping=jnp.asarray(lm_state.damping, dtype=info.loss.dtype),
            hyper=_cast_hyper(lm_state.hyper, info.loss.dtype),
        )
        initial_lm_state = lm_state
        status = LMStatus.RUNNING
        steps = 0
        if not bool(jnp.isfinite(info.loss)):
            status = LMStatus.NONFINITE
        elif bool(self._converged(info, atol, gtol, xtol)):
            status = LMStatus.CONVERGED

        for steps in range(1, max_steps + 1):
            if status != LMStatus.RUNNING:
                steps -= 1
                break
            x_old, lm_state_old = x, lm_state
            x, lm_state, info = self.update(x, lm_state, args, p)
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
            action, x, lm_state, args, user_state, problem_changed = self._apply_action(
                action, x, lm_state, args, user_state
            )
            history = _record_history(history, steps, x, info, args)
            if action.stop is not None and bool(action.stop):
                status = (
                    LMStatus.CALLBACK_STOP
                    if action.status is None
                    else int(action.status)
                )
                break
            # info describes the pre-action (x, args); if the action changed
            # them, the tolerances must wait for a fresh update.
            if bool(self._converged(info, atol, gtol, xtol)) and not bool(
                problem_changed
            ):
                status = LMStatus.CONVERGED
                break
        else:
            steps = max_steps

        if status == LMStatus.RUNNING:
            status = LMStatus.MAX_STEPS
        final_aux = None
        if self.has_aux:
            final_aux = self._residual_and_aux(x, args, p)[1]
        x_history, aux_history, args_history = _finalize_history(
            history, steps, final_aux
        )
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
    # Recast damping, hyperparameters, and tolerances to the residual dtype so
    # the while_loop carry matches what update() returns.
    atol = jnp.asarray(atol, dtype=info.loss.dtype)
    gtol = jnp.asarray(gtol, dtype=info.loss.dtype)
    xtol = jnp.asarray(xtol, dtype=info.loss.dtype)
    lm_state = dataclasses.replace(
        lm_state,
        damping=jnp.asarray(lm_state.damping, dtype=info.loss.dtype),
        hyper=_cast_hyper(lm_state.hyper, info.loss.dtype),
    )
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


def _accept_converged(_, result):
    return result.status == LMStatus.CONVERGED


def _cold_lm_state(lm_state):
    # Drawn starts must not reuse a Jacobian cache, a deflation basis, or a
    # preconditioner state built at another (x, args); damping and
    # hyperparameters stay inherited from the caller's initial state. Never
    # materializes fields from None -- the carry structure must match the
    # attempt-0 result.
    updates = {}
    if lm_state.jacobian_valid is not None:
        updates["jacobian_valid"] = jnp.zeros_like(lm_state.jacobian_valid)
    if lm_state.precond is not None:
        # Zero the stale state and mark it invalid so the drawn start's first
        # update rebuilds prepare() at its own x before applying it.
        updates["precond"] = jax.tree.map(jnp.zeros_like, lm_state.precond)
        updates["precond_valid"] = jnp.zeros_like(lm_state.precond_valid)
    if lm_state.recycle is not None:
        recycle = lm_state.recycle
        updates["recycle"] = RecycleState(
            U=jnp.zeros_like(recycle.U),
            dual_velocity=jnp.zeros_like(recycle.dual_velocity),
            dual_accel=jnp.zeros_like(recycle.dual_accel),
            valid=jnp.zeros_like(recycle.valid),
            iterations=jnp.zeros_like(recycle.iterations),
            residual_norm=jnp.zeros_like(recycle.residual_norm),
        )
    if not updates:
        return lm_state
    return dataclasses.replace(lm_state, **updates)


def _ranking_loss(solver, result, p, callback):
    # A callback can replace x/args after the last update, leaving info.loss
    # stale relative to the returned solution, so any callback-bearing solve
    # pays one extra residual evaluation per attempt. Nonfinite losses mask
    # to +inf so selection prefers any finite attempt and comparisons never
    # propagate NaN.
    if callback is None:
        loss = result.info.loss
    else:
        residual = solver._residual_and_aux(result.x, result.args, p)[0]
        loss = jnp.sum(residual**2)
    return jnp.where(jnp.isfinite(loss), loss, jnp.asarray(jnp.inf, dtype=loss.dtype))


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
    accept_fn = _accept_converged if accept is None else accept

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
        loss = _ranking_loss(solver, result, p, callback)
        success = _attempt_success(accept_fn, accept_key, result, loss)
        # p is loop-invariant: splice it out of the carried result and
        # reattach after selection.
        return dataclasses.replace(result, p=None), loss, success

    zero = jnp.asarray(0, dtype=jnp.int32)
    best, best_loss, done = run_attempt(x, lm_state, args, zero)
    if draw is None:
        info = MultiStartInfo(zero, done, jnp.asarray(1, dtype=jnp.int32), best_loss)
        return dataclasses.replace(best, p=p, multi_start=info)

    cold = _cold_lm_state(lm_state)

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
    accept_fn = _accept_converged if accept is None else accept
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
    cold = _cold_lm_state(lm_state)

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
        loss = _ranking_loss(solver, result, p, callback)
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
