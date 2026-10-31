"""State, hyperparameter, action, diagnostic, status, and result types for
:class:`~nlls_gram.LevenbergMarquardt` and :class:`~nlls_gram.Newton`.

Both solvers share the outer-loop protocol types (:class:`SolverStatus`,
:class:`SolverContext`, :class:`CallbackContext`, :class:`SolverAction`,
:class:`SolveResult`) and carry their globalization-specific state and step
diagnostics in :class:`LMState` / :class:`LMInfo` or :class:`NewtonState` /
:class:`NewtonInfo`. Fields a configuration does not use stay ``None`` and
compile away as empty pytree subtrees. Nothing here imports the metric,
preconditioner, or linear-solver modules, so it sits at the bottom of the
import graph.
"""

import enum
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "CallbackContext",
    "LMHyperparams",
    "LMInfo",
    "LMState",
    "NewtonHyperparams",
    "NewtonInfo",
    "NewtonState",
    "SolveResult",
    "SolverAction",
    "SolverContext",
    "SolverStatus",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverContext:
    """What the solver knows at a metric, preconditioner, or linear-solver
    call site -- the inner algebra's context, as opposed to
    :class:`CallbackContext`, which a per-step user callback receives.

    Fields are ``None`` where the call site has nothing to offer:

    - ``x``: the current FLATTENED iterate (the whole parameter vector, not
      just the metric block).
    - ``state``: the live :class:`LMState` or :class:`NewtonState` (damping,
      ridge, caches, and the carried metric/preconditioner instances). In the
      implicit-AD rule this is the returned state under ``stop_gradient`` --
      inert conditioning data, like the ridge.
    - ``args`` / ``p``: the residual's auxiliary data and differentiation
      parameters as passed to ``solve``/``update``.
    """

    x: Any = None
    state: Any = None
    args: Any = None
    p: Any = None


class SolverStatus(enum.IntEnum):
    """Integer status codes returned by ``solve`` and reported per step in
    :attr:`NewtonInfo.step_status`.

    Members are real ints (``IntEnum``): they work as dict keys, compare
    against status arrays, and ``SolverStatus(int(result.status)).name``
    recovers the label for logging. Callbacks may return bare members (or any
    weak integer value) as ``SolverAction.status`` -- the solver canonicalizes
    to int32 at the boundary, so no explicit dtype casts are needed. Codes
    5--8 report terminal Newton/Armijo update failures.
    """

    RUNNING = 0
    CONVERGED = 1
    MAX_STEPS = 2
    NONFINITE = 3
    CALLBACK_STOP = 4
    LINE_SEARCH_FAILED = 5
    LINEAR_SOLVE_FAILED = 6
    NON_DESCENT = 7
    STAGNATION = 8


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverAction:
    """Optional callback action for ``solve``.

    A field left as ``None`` is unchanged. ``status`` is used only when
    ``stop`` is true. ``stop`` and ``status`` are canonicalized by the solver
    (to bool and int32), so callbacks may return Python bools, bare
    :class:`SolverStatus` members, or weak-typed arrays without explicit casts.
    """

    stop: Any = None
    status: Any = None
    x: Any = None
    state: Any = None
    args: Any = None
    user_state: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CallbackContext:
    """Information passed to a ``solve`` callback after each outer solver step."""

    step: jax.Array
    x: Any
    x_old: Any
    state: Any
    state_old: Any
    initial_state: Any
    args: Any
    p: Any
    user_state: Any
    info: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolveResult:
    """Final result returned by ``solve``."""

    x: Any
    state: Any
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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMHyperparams:
    """Per-step LM hyperparameters, carried in :attr:`LMState.hyper`.

    All fields are traced, so a ``solve`` callback can reset them -- e.g. grow
    the inner CG budget as the loss falls -- via
    ``dataclasses.replace(ctx.state, hyper=dataclasses.replace(
    ctx.state.hyper, iterative_maxiter=...))``. A field constructed as
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
    # is an anti-underflow backstop, not regularization -- pass `ridge=...`
    # for that. To hold a larger floor, clamp `damping` in a solve callback.
    if dtype is None:
        dtype = jnp.asarray(0.0).dtype
    return jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)


def _cast_lm_hyper(hyper, dtype):
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
    """Carried solver state threaded through :meth:`LevenbergMarquardt.init`,
    :meth:`LevenbergMarquardt.update`, and :meth:`LevenbergMarquardt.solve`.

    Only ``damping`` is always live; every other field is populated by the
    configuration that needs it and stays ``None`` otherwise. A ``solve``
    callback that rebuilds the state must PRESERVE the fields it does not mean
    to change -- use ``dataclasses.replace(ctx.state, ...)``.

    The callback-owned fields are ``damping``, ``ridge``, ``metric``,
    ``preconditioner``, and the ``hyper`` group (replaceable as a unit);
    everything else is solver-owned bookkeeping.

    Attributes:
        damping: ``()`` current LM damping.
        ridge: ``()`` ridge weight, strictly positive when ``ridge`` is active;
            ``None`` for unpenalized solves. Replacing it is the supported way
            to anneal mid-solve (see :class:`~nlls_gram.AnnealRidge`); the
            solver treats a ridge change as a problem change, suppressing that
            step's convergence test and invalidating the ridge-keyed caches.
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
            of the same type with matching leaf shapes and dtypes; when
            ``ridge`` is active the solver treats a changed metric as a problem
            change (the metric defines its objective), whereas unpenalized mode
            only invalidates the whitening-dependent solver caches.
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
    """Per-step diagnostics returned by :meth:`LevenbergMarquardt.update` and
    passed to callbacks in :attr:`CallbackContext.info`.

    The loss/damping fields report the accept/reject outcome of the step, while
    ``grad_norm``/``step_norm``/``aux`` are evaluated at the PRE-step ``x`` --
    the iterate the step was computed from.

    ``loss`` is always the objective the running solver minimizes: the sum of
    squared residuals when ``ridge=None``, and the RIDGE OBJECTIVE
    ``||r||^2 + ridge * ||x_m||_W^2`` (penalty included) when ``ridge`` is
    active -- ridge code that means equation error must read ``resid_loss``.

    The solver runs in the whitened variable ``y = F_bar x``, so ``grad_norm``,
    ``step_norm``, and ``penalty_grad_norm`` are Euclidean in ``y``: steps
    measured in the W-norm, gradients in the dual W^{-1}-norm. With the default
    Euclidean metric they are the plain quantities. Objective values are
    unaffected -- whitening is a linear bijection of the same objective.

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
            ``x``: ``||F_bar^{-T} J'r||``, plus ``ridge [y_m; 0]`` when
            ``ridge`` is active.
        step_norm: ``()`` norm of the candidate step, reported even when the
            step is rejected.
        ridge: ``()`` the ridge weight used this step (when ``ridge`` is
            active; ``None`` otherwise).
        resid_loss: ``||r||^2`` at the retained iterate (when ``ridge`` is
            active; ``None`` otherwise).
        penalty_value: ``||x_m||_W^2 = ||y_m||^2`` at the retained iterate
            (when ``ridge`` is active; ``None`` otherwise).
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
class NewtonHyperparams:
    """Per-step Newton and Armijo hyperparameters, carried in
    :attr:`NewtonState.hyper`.

    All fields are traced, so a ``solve`` callback can adjust line-search or
    inner Krylov settings without changing pytree structure or recompiling.
    A field constructed as ``None`` (backend-default ``iterative_maxiter``)
    stays ``None``.
    """

    c1: jax.Array
    decrease_factor: jax.Array
    maxls: jax.Array
    max_stepsize: jax.Array
    linesearch_tol: jax.Array
    iterative_tol: jax.Array
    iterative_atol: jax.Array
    iterative_maxiter: jax.Array | None
    damping: jax.Array = 1e-3
    damping_power: jax.Array = 1.5
    min_damping: jax.Array = 1e-10
    geodesic_acceptance_ratio: jax.Array = 0.75


def _cast_newton_hyper(hyper, dtype):
    if hyper is None:
        return None
    iterative_tol = jnp.asarray(hyper.iterative_tol, dtype=dtype)
    iterative_tol = jnp.where(
        jnp.isnan(iterative_tol),
        jnp.asarray(1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6, dtype=dtype),
        iterative_tol,
    )
    return NewtonHyperparams(
        c1=jnp.asarray(hyper.c1, dtype=dtype),
        decrease_factor=jnp.asarray(hyper.decrease_factor, dtype=dtype),
        maxls=jnp.asarray(hyper.maxls, dtype=jnp.int32),
        max_stepsize=jnp.asarray(hyper.max_stepsize, dtype=dtype),
        linesearch_tol=jnp.asarray(hyper.linesearch_tol, dtype=dtype),
        iterative_tol=iterative_tol,
        iterative_atol=jnp.asarray(hyper.iterative_atol, dtype=dtype),
        iterative_maxiter=None
        if hyper.iterative_maxiter is None
        else jnp.asarray(hyper.iterative_maxiter, dtype=jnp.int32),
        damping=jnp.asarray(hyper.damping, dtype=dtype),
        damping_power=jnp.asarray(hyper.damping_power, dtype=dtype),
        min_damping=jnp.asarray(hyper.min_damping, dtype=dtype),
        geodesic_acceptance_ratio=jnp.asarray(
            hyper.geodesic_acceptance_ratio, dtype=dtype
        ),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NewtonState:
    """Carried solver state threaded through :meth:`Newton.init`,
    :meth:`Newton.update`, and :meth:`Newton.solve`.

    With ``cache_jacobian=False`` (the default), only ``metric``,
    ``preconditioner``, ``ridge`` (when ``ridge`` is active), and (during
    ``solve``) ``hyper`` are populated; the Jacobian and linear-solver cache
    fields stay ``None`` and compile away as empty pytree subtrees. With
    ``cache_jacobian=True`` on a dense linear solver, a failed line search
    marks ``jacobian_valid`` and ``solver_cache.valid`` true so a retry at the
    unchanged base point reuses the linearization and assembled matrix or
    factorization.

    Attributes:
        resid: cached residual at the current ``x`` (``cache_jacobian=True``
            dense paths only).
        Jt: cached transpose-Jacobian ``J'`` at the current ``x``.
        jacobian_valid: ``()`` bool -- whether the cached ``resid``/``Jt``
            describe the current ``x``. Reset to ``False`` (or call ``init``)
            when changing ``p`` in a manual ``update`` loop.
        aux: residual aux pytree at the current ``x`` (``has_aux=True`` with
            ``cache_jacobian=True``).
        hyper: per-step :class:`NewtonHyperparams`, populated by ``solve``;
            ``None`` (``init``'s default) falls back to the constructor values.
        solver_cache: the linear solver's own reject-step cache when
            ``cache_jacobian=True``.
        ridge: ``()`` positive scalar ridge parameter when ``ridge`` is active;
            ``None`` for unpenalized solves.
        metric: the carried :class:`~nlls_gram.Metric` instance.
        preconditioner: the carried :class:`~nlls_gram.Preconditioner` instance
            for Krylov configs; ``None`` otherwise.
    """

    resid: jax.Array | None = None
    Jt: jax.Array | None = None
    jacobian_valid: jax.Array | None = None
    aux: Any = None
    hyper: NewtonHyperparams | None = None
    solver_cache: Any = None
    ridge: jax.Array | None = None
    metric: Any = None
    preconditioner: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NewtonInfo:
    """Per-step diagnostics returned by :meth:`Newton.update` and passed to
    callbacks in :attr:`CallbackContext.info`.

    When ``ridge=None``, ``loss`` is ``||r||_2^2``; when ``ridge`` is active,
    ``loss`` is the full ridge objective ``||r||_2^2 + ridge * ||x_m||_W^2``.
    ``directional_derivative`` is the directional derivative of the
    half-objective merit ``phi(x) = 0.5 * loss``: ``s = g' u``. In diagnostic
    ``loss`` units the Armijo condition is ``loss_candidate <= loss_old +
    2 * c1 * alpha * directional_derivative + 2 * linesearch_tol``.

    Attributes:
        loss: objective value at the retained iterate (before any callback
            action).
        loss_old: objective value at the base point of this update.
        loss_candidate: objective value at the last actually evaluated trial
            point (including an unsuccessful final trial).
        accepted: ``()`` bool, whether a line-search candidate was accepted.
        stepsize: ``()`` accepted step size ``alpha``; zero if no step was
            accepted.
        trial_stepsize: ``()`` last ``alpha`` actually tested in the line
            search (zero if no trial was evaluated).
        num_linesearch_steps: ``()`` int32 count of trial residual evaluations,
            including the initial full-step trial.
        directional_derivative: ``()`` actual Armijo slope ``s = g' u`` of
            ``0.5 * loss`` at the base point.
        grad_norm: ``()`` Euclidean norm of the whitened half-gradient ``g``
            at the base point (``||F_bar^{-T} J' r||_2`` when unpenalized, or
            ``||F_bar^{-T} J' r + ridge * [y_m; 0]||_2`` when ``ridge`` is
            active).
        direction_norm: ``()`` Euclidean norm of the whitened velocity
            direction ``u = F_bar d``.
        step_norm: ``()`` whitened norm of the accepted step (``||alpha * u||_2``
            or ``||alpha * u + 0.5 * alpha^2 * u_a||_2`` when geodesic
            acceleration is active); zero when the step is rejected.
        step_status: ``()`` int32 :class:`SolverStatus` for this update
            (``RUNNING`` on a normal accepted step, ``CONVERGED`` at an exact
            base-point root, or a specific failure code).
        used_geodesic: ``()`` bool, whether the geodesic second-order
            correction passed the acceptance ratio test and was included in an
            accepted step.
        acceleration_ratio: ``()`` ratio ``2 * ||u_a||_2 / ||u||_2`` in
            whitened coordinates (zero when ``geodesic_acceleration=False``).
        ridge: ``()`` active ridge parameter (when ``ridge`` is active;
            ``None`` otherwise).
        resid_loss: ``()`` unpenalized ``||r||_2^2`` (when ``ridge`` is active;
            ``None`` otherwise).
        penalty_value: ``()`` ``||x_m||_W^2 = ||y_m||_2^2`` (when ``ridge`` is
            active; ``None`` otherwise).
        penalty_grad_norm: ``()`` ``||y_m||_2`` (when ``ridge`` is active;
            ``None`` otherwise).
        aux: residual aux output at the base/pre-step ``x`` (``has_aux=True``).
    """

    loss: jax.Array
    loss_old: jax.Array
    loss_candidate: jax.Array
    accepted: jax.Array
    stepsize: jax.Array
    trial_stepsize: jax.Array
    num_linesearch_steps: jax.Array
    directional_derivative: jax.Array
    grad_norm: jax.Array
    direction_norm: jax.Array
    step_norm: jax.Array
    step_status: jax.Array
    used_geodesic: jax.Array | None = None
    acceleration_ratio: jax.Array | None = None
    ridge: jax.Array | None = None
    resid_loss: jax.Array | None = None
    penalty_value: jax.Array | None = None
    penalty_grad_norm: jax.Array | None = None
    aux: Any = None
