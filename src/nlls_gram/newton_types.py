"""State, hyperparameter, action, diagnostic, status, and result types for
:class:`~nlls_gram.Newton` and :class:`~nlls_gram.RidgeNewton`.

One :class:`NewtonState` and one :class:`NewtonInfo` serve both solvers: fields
a configuration does not use stay ``None`` and compile away as empty pytree
subtrees. Nothing here imports the metric, preconditioner, or linear-solver
modules, so it sits at the bottom of the import graph.
"""

import enum
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "NewtonAction",
    "NewtonContext",
    "NewtonHyperparams",
    "NewtonInfo",
    "NewtonSolveResult",
    "NewtonState",
    "NewtonStatus",
]


class NewtonStatus(enum.IntEnum):
    """Integer status codes returned by :meth:`Newton.solve` and reported per
    step in :attr:`NewtonInfo.step_status`.

    The first five codes match :class:`~nlls_gram.LMStatus`; codes 5--8 report
    terminal Newton/Armijo update failures.
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
    reg: jax.Array = 0.0
    reg_power: jax.Array = 1.5
    min_damping: jax.Array = 1e-12
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
        reg=jnp.asarray(hyper.reg, dtype=dtype),
        reg_power=jnp.asarray(hyper.reg_power, dtype=dtype),
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
    ``preconditioner``, ``ridge`` (for :class:`~nlls_gram.RidgeNewton`), and
    (during ``solve``) ``hyper`` are populated; the Jacobian and linear-solver
    cache fields stay ``None`` and compile away as empty pytree subtrees. With
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
        ridge: ``()`` positive scalar ridge parameter for
            :class:`~nlls_gram.RidgeNewton`; ``None`` for
            :class:`~nlls_gram.Newton`.
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
    callbacks in :attr:`NewtonContext.info`.

    For :class:`~nlls_gram.Newton`, ``loss`` is ``||r||_2^2``; for
    :class:`~nlls_gram.RidgeNewton`, ``loss`` is the full ridge objective
    ``||r||_2^2 + ridge * ||x_m||_W^2``. ``directional_derivative`` is the
    directional derivative of the half-objective merit ``phi(x) = 0.5 * loss``:
    ``s = g' u``. In diagnostic ``loss`` units the Armijo condition is
    ``loss_candidate <= loss_old + 2 * c1 * alpha * directional_derivative
    + 2 * linesearch_tol``.

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
            at the base point (``||F_bar^{-T} J' r||_2`` for :class:`Newton`,
            or ``||F_bar^{-T} J' r + ridge * [y_m; 0]||_2`` for
            :class:`RidgeNewton`).
        direction_norm: ``()`` Euclidean norm of the whitened velocity
            direction ``u = F_bar d``.
        step_norm: ``()`` whitened norm of the accepted step (``||alpha * u||_2``
            or ``||alpha * u + 0.5 * alpha^2 * u_a||_2`` when geodesic
            acceleration is active); zero when the step is rejected.
        step_status: ``()`` int32 :class:`NewtonStatus` for this update
            (``RUNNING`` on a normal accepted step, ``CONVERGED`` at an exact
            base-point root, or a specific failure code).
        used_geodesic: ``()`` bool, whether the geodesic second-order
            correction passed the acceptance ratio test and was included in an
            accepted step.
        acceleration_ratio: ``()`` ratio ``2 * ||u_a||_2 / ||u||_2`` in
            whitened coordinates (zero when ``geodesic_acceleration=False``).
        ridge: ``()`` active ridge parameter (:class:`RidgeNewton` only;
            ``None`` for :class:`Newton`).
        resid_loss: ``()`` unpenalized ``||r||_2^2`` (:class:`RidgeNewton`
            only; ``None`` for :class:`Newton`).
        penalty_value: ``()`` ``||x_m||_W^2 = ||y_m||_2^2`` (:class:`RidgeNewton`
            only; ``None`` for :class:`Newton`).
        penalty_grad_norm: ``()`` ``||y_m||_2`` (:class:`RidgeNewton` only;
            ``None`` for :class:`Newton`).
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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NewtonAction:
    """Optional callback action for :meth:`Newton.solve`.

    A field left as ``None`` is unchanged. ``status`` is used only when
    ``stop`` is true. ``stop`` and ``status`` are canonicalized to bool and
    int32 at the solver boundary.
    """

    stop: Any = None
    status: Any = None
    x: Any = None
    newton_state: Any = None
    args: Any = None
    user_state: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NewtonContext:
    """Information passed to a :meth:`Newton.solve` callback after each outer
    Newton update.
    """

    step: jax.Array
    x: Any
    x_old: Any
    newton_state: Any
    newton_state_old: Any
    initial_newton_state: Any
    args: Any
    p: Any
    user_state: Any
    info: Any

    @property
    def state(self):
        return self.newton_state


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NewtonSolveResult:
    """Final result returned by :meth:`Newton.solve`."""

    x: Any
    newton_state: Any
    info: Any
    steps: jax.Array
    status: jax.Array
    args: Any
    p: Any
    user_state: Any
    aux: Any = None
    x_history: Any = None
    aux_history: Any = None
    args_history: Any = None
    multi_start: Any = None
