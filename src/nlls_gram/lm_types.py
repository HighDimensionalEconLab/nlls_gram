"""State, hyperparameter, action, and result types shared by both solvers.

One ``LMState`` and one ``LMInfo`` serve ``LevenbergMarquardt`` and
``RidgeLevenbergMarquardt``: fields a configuration does not use stay ``None``
and compile away as empty pytree subtrees. Nothing here imports the metric,
preconditioner, or linear-solver modules, so it sits at the bottom of the
import graph.
"""

import enum
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "LMAction",
    "LMContext",
    "LMHyperparams",
    "LMInfo",
    "LMSolveResult",
    "LMState",
    "LMStatus",
    "SolverContext",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverContext:
    """What the solver knows at a metric, preconditioner, or linear-solver
    call site -- the inner algebra's context, as opposed to
    :class:`LMContext`, which a per-step user callback receives.

    Fields are ``None`` where the call site has nothing to offer:

    - ``x``: the current FLATTENED iterate (the whole parameter vector, not
      just the metric block).
    - ``lm_state``: the live :class:`LMState` (damping, ridge, caches, and
      the carried metric/preconditioner instances). In the implicit-AD rule
      this is the returned state under ``stop_gradient`` -- inert
      conditioning data, like the ridge.
    - ``args`` / ``p``: the residual's auxiliary data and differentiation
      parameters as passed to ``solve``/``update``.
    """

    x: Any = None
    lm_state: Any = None
    args: Any = None
    p: Any = None


class LMStatus(enum.IntEnum):
    """Integer status codes returned by ``solve``.

    Members are real ints (``IntEnum``): they work as dict keys, compare
    against status arrays, and ``LMStatus(int(result.status)).name`` recovers
    the label for logging. Callbacks may return bare members (or any weak
    integer value) as ``LMAction.status`` -- the solver canonicalizes to
    int32 at the boundary, so no explicit dtype casts are needed.
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

    All fields are traced, so a ``solve`` callback can reset them -- e.g. grow
    the inner CG budget as the loss falls -- via
    ``dataclasses.replace(ctx.lm_state, hyper=dataclasses.replace(
    ctx.lm_state.hyper, iterative_maxiter=...))``. A field constructed as
    ``None`` (backend-default ``iterative_maxiter``) is compiled out and stays
    ``None``. Static configuration -- the linear solver, the metric,
    ``geodesic_acceleration``, ``cache_jacobian``, ``has_aux`` -- shapes the
    compiled program and lives on the solver.
    """

    damping_decrease: jax.Array
    damping_increase: jax.Array
    geodesic_acceptance_ratio: jax.Array
    iterative_tol: jax.Array
    iterative_atol: jax.Array
    iterative_maxiter: jax.Array | None


def _damping_floor(dtype):
    # Smallest positive normal. Damping is updated multiplicatively, so a value
    # in a backend's flush-to-zero range is absorbing: `0 * damping_increase`
    # stays 0 and the solver could never re-damp after a rejected step. Far
    # below the scale at which damping still perturbs the Gram diagonal, so it
    # is an anti-underflow backstop, not regularization -- use the ridge solver
    # for that. To hold a larger floor, clamp `damping` in a solve callback.
    if dtype is None:
        dtype = jnp.asarray(0.0).dtype
    return jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)


def _cast_hyper(hyper, dtype):
    if hyper is None:
        return None
    # NaN is the "resolve me against the problem dtype" sentinel for CG's
    # tol=None: solve() builds the hyperparameters before it has seen a
    # residual, and 1e-10 is unreachable in float32, so an unresolved default
    # would burn the full CG budget every inner solve.
    iterative_tol = jnp.asarray(hyper.iterative_tol, dtype=dtype)
    iterative_tol = jnp.where(
        jnp.isnan(iterative_tol),
        jnp.asarray(1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6, dtype=dtype),
        iterative_tol,
    )
    return LMHyperparams(
        jnp.asarray(hyper.damping_decrease, dtype=dtype),
        jnp.asarray(hyper.damping_increase, dtype=dtype),
        jnp.asarray(hyper.geodesic_acceptance_ratio, dtype=dtype),
        iterative_tol,
        jnp.asarray(hyper.iterative_atol, dtype=dtype),
        None
        if hyper.iterative_maxiter is None
        else jnp.asarray(hyper.iterative_maxiter, dtype=jnp.int32),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMState:
    """Carried solver state threaded through ``init``/``update``/``solve``.

    Only ``damping`` is always live; every other field is populated by the
    configuration that needs it and stays ``None`` otherwise. A ``solve``
    callback that rebuilds the state must PRESERVE the fields it does not mean
    to change -- use ``dataclasses.replace(ctx.lm_state, ...)``.

    The callback-owned fields are ``damping``, ``ridge``, ``metric``,
    ``preconditioner``, and the ``hyper`` group (replaceable as a unit);
    everything else is solver-owned bookkeeping.

    Attributes:
        damping: ``()`` current LM damping.
        ridge: ``()`` ridge weight, strictly positive, for
            ``RidgeLevenbergMarquardt``; ``None`` for the metric solver.
            Replacing it is the supported way to anneal mid-solve (see
            ``AnnealRidge``); the solver treats a ridge change as a problem
            change, suppressing that step's convergence test and
            invalidating the ridge-keyed caches.
        resid: cached residual at the current ``x`` (``cache_jacobian`` dense
            paths only).
        Jt: cached transpose-Jacobian ``J'`` at the current ``x``.
        jacobian_valid: ``()`` bool -- the cached ``resid``/``Jt`` are still
            current because the last step was rejected, so ``x`` did not move.
        aux: residual aux pytree at the current ``x`` (``has_aux=True``).
        hyper: per-step :class:`LMHyperparams`, populated by ``solve``;
            ``None`` (``init``'s default) falls back to the constructor values.
        solver_cache: the linear solver's own reject-step cache, whose pytree
            structure is fixed by the static ``linear_solver`` config.
        metric: the carried :class:`~nlls_gram.Metric` instance every factor
            op reads. A callback replaces it by constructing a new instance
            of the same type with matching leaf shapes and dtypes; the ridge
            solver treats a changed metric as a problem change (the metric
            defines its objective), the metric solver only invalidates the
            whitening-dependent solver caches.
        preconditioner: the carried :class:`~nlls_gram.Preconditioner`
            instance the Krylov configs read; ``None`` outside them. A
            callback refresh is never treated as a problem change --
            staleness only moves the CG iteration path.
    """

    damping: jax.Array
    ridge: jax.Array | None = None
    resid: jax.Array | None = None
    Jt: jax.Array | None = None
    jacobian_valid: jax.Array | None = None
    aux: Any = None
    hyper: LMHyperparams | None = None
    solver_cache: Any = None
    metric: Any = None
    preconditioner: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMInfo:
    """Per-step diagnostics returned by ``update`` and by each ``solve`` step.

    The loss/damping fields report the accept/reject outcome of the step, while
    ``grad_norm``/``step_norm``/``aux`` are evaluated at the PRE-step ``x`` --
    the iterate the step was computed from.

    ``loss`` is always the objective the running solver minimizes: the sum of
    squared residuals for ``LevenbergMarquardt``, and the RIDGE OBJECTIVE
    ``||r||^2 + ridge * ||x_m||_W^2`` (penalty included) for
    ``RidgeLevenbergMarquardt`` -- ridge code that means equation error must
    read ``resid_loss``.

    BOTH solvers run in the whitened variable ``y = F_bar x``, so
    ``grad_norm``, ``step_norm``, and ``penalty_grad_norm`` are Euclidean in
    ``y``: steps measured in the W-norm, gradients in the dual W^{-1}-norm.
    With the default Euclidean metric they are the plain quantities. Objective
    values are unaffected -- whitening is a linear bijection of the same
    objective.

    Attributes:
        loss: objective at the retained iterate, ``min(loss_old,
            loss_candidate)``.
        loss_old: objective at the pre-step ``x``.
        loss_candidate: objective at the trial point.
        accepted: ``()`` bool, whether the trial step was accepted.
        damping: ``()`` post-update damping.
        damping_factor: ``()`` multiplicative damping update applied this step.
        used_geodesic: ``()`` bool, whether the geodesic correction entered the
            accepted step.
        acceleration_ratio: ``()`` acceleration-to-velocity norm ratio.
        grad_norm: ``()`` whitened stationarity residual at the pre-step
            ``x``: ``||F_bar^{-T} J'r||``, plus ``ridge [y_m; 0]`` for the
            ridge solver.
        step_norm: ``()`` norm of the candidate step, reported even when the
            step is rejected.
        ridge: ``()`` the ridge weight used this step (ridge solver only).
        resid_loss: ``||r||^2`` at the retained iterate (ridge solver only).
        penalty_value: ``||x_m||_W^2 = ||y_m||^2`` at the retained iterate.
        penalty_grad_norm: ``()`` ``||[y_m; 0]|| = sqrt(penalty_value)`` at the
            pre-step ``x``, reported so ``gtol`` can be CALIBRATED rather than
            guessed: at a ridge minimizer the gradient is the cancellation of
            the residual pullback against ``ridge * [y_m; 0]``, so demanding
            ``grad_norm < c * ridge * penalty_grad_norm`` resolves the
            selection coordinates to ~``c`` relative accuracy. The recipe is
            ``gtol ~ 1e-3 * ridge * sqrt(q(x*))`` with ``q`` the solution's
            squared seminorm.
        aux: residual aux output at the pre-step ``x`` (``has_aux=True``).
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
    ridge: jax.Array | None = None
    resid_loss: jax.Array | None = None
    penalty_value: jax.Array | None = None
    penalty_grad_norm: jax.Array | None = None
    aux: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMAction:
    """Optional callback action for ``solve``.

    A field left as ``None`` is unchanged. ``status`` is used only when
    ``stop`` is true. ``stop`` and ``status`` are canonicalized by the solver
    (to bool and int32), so callbacks may return Python bools, bare
    ``LMStatus`` members, or weak-typed arrays without explicit casts.
    """

    stop: Any = None
    status: Any = None
    x: Any = None
    lm_state: Any = None
    args: Any = None
    user_state: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMContext:
    """Information passed to a ``solve`` callback after each LM update."""

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
    """Final result returned by ``solve``."""

    x: Any
    lm_state: Any
    info: Any
    steps: jax.Array
    status: jax.Array
    args: Any
    p: Any
    user_state: Any
    # With has_aux=True: aux evaluated at the returned (x, args, p) -- one extra
    # residual evaluation, well-defined for every status. Differentiable with
    # respect to p through the implicit rule (directly and through x*(p)).
    aux: Any = None
    # With save_steps=True: the iterate history as a pytree shaped like x with a
    # (max_steps + 1) leading axis -- row 0 is x0, row s the kept iterate after
    # step s (post-callback-action), rows beyond ``steps`` are zero padding.
    # aux_history (has_aux only) and args_history (None when args is None) align
    # row-for-row. Differentiation-inert (zero tangents through the implicit rule).
    x_history: Any = None
    aux_history: Any = None
    args_history: Any = None
    # MultiStartInfo when solve ran with multi_start=...; None otherwise (an
    # empty pytree node, so the leaf count is unchanged when the feature is off).
    multi_start: Any = None
