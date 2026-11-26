import enum
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "PDAS",
    "BoxSolveResult",
    "CallbackContext",
    "CompileInfo",
    "Complementarity",
    "FischerBurmeister",
    "Inequality",
    "InequalityPenalty",
    "LMHyperparams",
    "LMInfo",
    "LMState",
    "LagrangeMultipliers",
    "Minimum",
    "NewtonHyperparams",
    "NewtonInfo",
    "NewtonState",
    "Residuals",
    "ScalarSolveResult",
    "SolveResult",
    "SolverAction",
    "SolverContext",
    "SolverStatus",
    "TrustRegionInfo",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Inequality:
    """Nonlinear inequality residual margins representing ``value >= 0``."""

    value: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Complementarity:
    """Paired complementarity margins representing ``0 <= lhs _|_ rhs >= 0``."""

    lhs: Any
    rhs: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Residuals:
    """Structured residual container for equality, inequality, and complementarity."""

    residual: Any = None
    inequality: Any = None
    complementarity: Any = None


@dataclass(frozen=True)
class InequalityPenalty:
    """Configuration for nonlinear inequality residuals ``g(x) >= 0``.

    ``weight`` scales the half-objective penalty ``0.5 * weight * ||psi(g)||_2^2``
    by multiplying the transformed inequality residual block by ``sqrt(weight)``.
    ``smoothing == 0`` applies the exact violation ``min(g, 0)`` with zero
    derivative for ``g >= 0``. Positive ``smoothing`` applies
    ``-smoothing * softplus(-g / smoothing)`` and decays toward ``smoothing_min``
    by ``smoothing_decay`` when ``smoothing_decay < 1``.
    """

    weight: Any = 1.0
    smoothing: float = 0.0
    smoothing_min: float = 0.0
    smoothing_decay: float = 1.0


@dataclass(frozen=True)
class Minimum:
    """Componentwise ``min`` reformulation for complementarity residuals."""

    c_scale: Any = 1.0
    w_comp: float = 1.0


@dataclass(frozen=True)
class PDAS:
    """Primal-dual active-set reformulation for complementarity residuals."""

    c_scale: Any = 1.0
    w_comp: float = 1.0
    tau_init: float = 0.0
    tau_min: float = 0.0
    tau_decay: float = 0.5


@dataclass(frozen=True)
class FischerBurmeister:
    """Scaled Fischer-Burmeister reformulation for complementarity residuals."""

    c_scale: Any = 1.0
    w_comp: float = 1.0
    mu_init: float = 0.0
    mu_min: float = 0.0
    mu_decay: float = 0.5
    eps_safe: float | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverContext:
    """Live solver context passed to ridge metrics and preconditioners."""

    x: Any = None
    state: Any = None
    args: Any = None
    p: Any = None
    ridge: Any = None
    damping: Any = None
    space: str | None = field(default=None, metadata=dict(static=True))
    role: str | None = field(default=None, metadata=dict(static=True))
    form: str | None = field(default=None, metadata=dict(static=True))


class SolverStatus(enum.IntEnum):
    """Integer status codes returned by ``solve`` and reported per Newton step."""

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
    """Optional per-step callback override for ``solve``."""

    stop: Any = None
    status: Any = None
    x: Any = None
    state: Any = None
    args: Any = None
    user_state: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CallbackContext:
    """Per-step context passed to a ``solve`` callback."""

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


class CompileInfo(NamedTuple):
    """Compilation wall time and XLA memory footprint returned by ``compile``."""

    compile_time: float
    peak_bytes: int


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolveResult:
    """Result returned by ``solve``."""

    x: Any
    state: Any
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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LagrangeMultipliers:
    """Nonnegative KKT shadow prices for active lower and upper box bounds."""

    lower: jax.Array
    upper: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoxSolveResult:
    """Result returned by ``BoxSecant.solve``."""

    x: jax.Array
    fun: jax.Array
    grad: jax.Array
    lm: LagrangeMultipliers
    bracket_width: jax.Array
    steps: jax.Array
    status: jax.Array
    lower: jax.Array
    upper: jax.Array
    args: Any = None
    p: Any = None
    aux: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrustRegionInfo:
    """Diagnostics carried in ``ScalarSolveResult.info`` under ``TrustRegion``."""

    radius: jax.Array
    step_norm: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarSolveResult:
    """Result returned by ``ScalarNewton.solve``."""

    x: jax.Array
    fun: jax.Array
    grad: jax.Array
    curv: jax.Array
    info: Any
    steps: jax.Array
    status: jax.Array
    args: Any = None
    p: Any = None
    aux: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMHyperparams:
    """Traced Levenberg-Marquardt hyperparameters carried in ``LMState.hyper``."""

    damping_decrease: jax.Array
    damping_increase: jax.Array
    geodesic_acceptance_ratio: jax.Array
    iterative_tol: jax.Array
    iterative_atol: jax.Array
    iterative_maxiter: jax.Array | None
    c1: jax.Array = 1e-4
    decrease_factor: jax.Array = 0.5
    maxls: jax.Array = 20
    max_stepsize: jax.Array = 1.0
    linesearch_tol: jax.Array = 0.0


def _damping_floor(dtype):
    if dtype is None:
        dtype = jnp.asarray(0.0).dtype
    return jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)


def _cast_lm_hyper(hyper, dtype):
    if hyper is None:
        return None
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
        c1=jnp.asarray(hyper.c1, dtype=dtype),
        decrease_factor=jnp.asarray(hyper.decrease_factor, dtype=dtype),
        maxls=jnp.asarray(hyper.maxls, dtype=jnp.int32),
        max_stepsize=jnp.asarray(hyper.max_stepsize, dtype=dtype),
        linesearch_tol=jnp.asarray(hyper.linesearch_tol, dtype=dtype),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMState:
    """Carried state threaded through ``init``, ``update``, and ``solve``."""

    damping: jax.Array
    ridge: jax.Array | None = None
    resid: jax.Array | None = None
    Jt: jax.Array | None = None
    jacobian_valid: jax.Array | None = None
    aux: Any = None
    hyper: LMHyperparams | None = None
    solver_cache: Any = None
    ridge_metric: Any = None
    preconditioner: Any = None
    homotopy: jax.Array | None = None
    ineq_weight: jax.Array | None = None
    ineq_smoothing: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMInfo:
    """Per-step diagnostics returned by ``LevenbergMarquardt.update``."""

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
    stepsize: jax.Array | None = None
    num_linesearch_steps: jax.Array | None = None
    homotopy: jax.Array | None = None
    complementarity_changed: jax.Array | None = None
    eq_loss: jax.Array | None = None
    ineq_violation_norm: jax.Array | None = None
    ineq_max_violation: jax.Array | None = None
    inequality_changed: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NewtonHyperparams:
    """Traced Newton and Armijo hyperparameters carried in ``NewtonState.hyper``."""

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
    """Carried state threaded through ``Newton.init``, ``update``, and ``solve``."""

    resid: jax.Array | None = None
    Jt: jax.Array | None = None
    jacobian_valid: jax.Array | None = None
    aux: Any = None
    hyper: NewtonHyperparams | None = None
    solver_cache: Any = None
    ridge: jax.Array | None = None
    ridge_metric: Any = None
    preconditioner: Any = None
    homotopy: jax.Array | None = None
    ineq_weight: jax.Array | None = None
    ineq_smoothing: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NewtonInfo:
    """Per-step diagnostics returned by ``Newton.update``."""

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
    homotopy: jax.Array | None = None
    complementarity_changed: jax.Array | None = None
    eq_loss: jax.Array | None = None
    ineq_violation_norm: jax.Array | None = None
    ineq_max_violation: jax.Array | None = None
    inequality_changed: jax.Array | None = None
