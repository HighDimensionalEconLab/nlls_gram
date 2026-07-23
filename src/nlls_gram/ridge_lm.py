"""Ridge-regularized Levenberg-Marquardt for underdetermined interpolation.

RidgeLevenbergMarquardt minimizes the ridge objective
``||r(x, args, p)||^2 + ridge * ||L x||^2`` for a user-supplied residual and a
positive-semidefinite penalty factor ``L`` (a :class:`~nlls_gram.RidgePenalty`),
with the ridge weight ``lambda`` carried as traced state that a ``solve``
callback may anneal toward zero (:func:`ridge_continuation`). Selection of the
minimum-seminorm interpolant lives in the OBJECTIVE -- classical nonlinear
Tikhonov regularization (Engl-Kunisch-Neubauer 1989; Engl-Hanke-Neubauer 1996
Ch. 10; the seminorm formulation goes back to Elden 1982) -- rather than in an
algorithmic implicit bias, and the per-accepted-step annealed limit is the
iteratively regularized Gauss-Newton method (Bakushinskii 1992;
Blaschke-Neubauer-Scherzer 1997; Kaltenbacher-Neubauer-Scherzer 2008).
Numerically the solver is stock Euclidean LM on the augmented residual
``[r; sqrt(ridge) L x]`` (Marquardt 1963; More 1978) with geodesic
acceleration (Transtrum-Sethna 2012); the ``qr`` path uses corrected
semi-normal equations (Bjorck 1987; Bjorck 1996 Sec. 6.6.5) and the ``lsmr``
path the Fong-Saunders (2011) bidiagonalization with a right preconditioner
(Bjorck 1996 Ch. 7).
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import numpy as np
from jax.flatten_util import ravel_pytree

from nlls_gram.lsmr import lsmr_solve
from nlls_gram.penalties import RidgePenalty
from nlls_gram.utility import (
    LMHyperparams,
    LMSolveAction,
    LMStatus,
    MultiStart,
    _accept_converged,
    _accept_converged_or_max_steps,
    _cast_hyper,
    _check_drawn_types,
    _damping_floor,
    _hashable_hook,
    _mask_tangent_tree,
    _multi_start_parallel_jit,
    _multi_start_python_impl,
    _multi_start_sequential_jit,
    _solve_loop_jit,
    _solve_python_impl,
    _static_key_component,
    _tree_changed,
    _where_tree,
    _zero_tangent_leaf,
    canonicalize_ad_preconditioner,
    canonicalize_residual,
)

__all__ = [
    "RidgeLevenbergMarquardt",
    "RidgeLMInfo",
    "RidgeLMState",
    "ridge_continuation",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RidgeLMState:
    """Carried solver state threaded through ``init``/``update``/``solve``.

    ``damping`` and ``ridge`` are always live; the remaining fields are
    populated by the configuration that needs them and stay ``None`` on the
    other paths (compiled away at no cost). A ``solve`` callback that rebuilds
    ``lm_state`` must PRESERVE the fields it does not mean to change -- use
    ``dataclasses.replace(ctx.lm_state, ...)``. Replacing ``ridge`` is the
    supported way to anneal the ridge weight mid-solve (see
    :func:`ridge_continuation`); the solver treats a ridge change as a problem
    change, suppressing that step's convergence test and invalidating the
    ridge-keyed caches.

    Attributes:
        damping: ``()`` current LM damping ``mu`` (the Euclidean trust-region
            parameter, decoupled from ``ridge``).
        ridge: ``()`` current ridge weight ``lambda``, strictly positive by
            contract. Set by ``init`` (constructor value or the dtype default)
            and replaced only by callbacks.
        resid: cached TRUE residual at the current ``x``
            (``cache_jacobian=True`` dense paths only, else ``None``).
        Jt: cached transpose-Jacobian ``J'`` of the TRUE residual at the
            current ``x`` (``cache_jacobian=True`` dense paths only).
        jacobian_valid: ``()`` bool -- the cached ``resid``/``Jt`` are still
            current because the last step was rejected so ``x`` did not move.
        aux: residual aux pytree at the current ``x`` (``has_aux=True``).
        hyper: per-step :class:`~nlls_gram.LMHyperparams`, populated by
            ``solve``; ``None`` (``init``'s default) falls back to the
            constructor values.
        G: cached assembled normal matrix ``J'J + ridge * L'L`` (pre-damping,
            in the linear-solve dtype) for the cholesky path, so a rejected
            step re-factors without re-assembling.
        G_valid: ``()`` bool -- ``G`` describes the current ``x``.
        G_ridge: ``()`` the ridge the cached ``G`` was assembled with; a
            callback ridge change invalidates the cache through this key.
        qr_R: cached ``R`` factor of ``[J; sqrt(ridge) L]`` for the qr path.
        qr_valid: ``()`` bool -- ``qr_R`` describes the current ``x``.
        qr_ridge: ``()`` the ridge the cached ``qr_R`` was built with.
        penalty_state: reserved for a future ``penalty_factory``; ``None``.
        penalty_valid: reserved for a future ``penalty_factory``; ``None``.
    """

    damping: jax.Array
    ridge: jax.Array
    resid: jax.Array | None = None
    Jt: jax.Array | None = None
    jacobian_valid: jax.Array | None = None
    aux: Any = None
    hyper: LMHyperparams | None = None
    G: jax.Array | None = None
    G_valid: jax.Array | None = None
    G_ridge: jax.Array | None = None
    qr_R: jax.Array | None = None
    qr_valid: jax.Array | None = None
    qr_ridge: jax.Array | None = None
    penalty_state: Any = None
    penalty_valid: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RidgeLMInfo:
    """Per-step diagnostics returned by ``update`` (and for each ``solve`` step).

    ``loss`` is the RIDGE OBJECTIVE ``||r||^2 + ridge * q(x)`` actually being
    minimized -- it includes the penalty term. Code that means equation error
    must read ``resid_loss`` (``||r||^2`` alone). The loss/damping fields
    report the accept/reject outcome of the step, while ``grad_norm``/
    ``step_norm``/``aux`` are evaluated at the PRE-step ``x``.

    Attributes:
        loss: ``min(loss_old, loss_candidate)`` ridge objective at the
            retained iterate.
        loss_old: ridge objective at the pre-step ``x``.
        loss_candidate: ridge objective at the trial point.
        resid_loss: ``||r||^2`` at the retained iterate.
        penalty_value: ``q(x) = ||L x||^2`` at the retained iterate.
        ridge: ``()`` the ridge weight ``lambda`` used this step.
        accepted: ``()`` bool, whether the trial step was accepted.
        damping: ``()`` post-update damping ``mu``.
        damping_factor: ``()`` multiplicative damping update applied this step.
        used_geodesic: ``()`` bool, whether the geodesic-acceleration
            correction entered the accepted step.
        acceleration_ratio: ``()`` geodesic acceleration-to-velocity Euclidean
            norm ratio.
        grad_norm: ``()`` ``||J'r + ridge * L'L x||`` at the pre-step ``x`` --
            the ridge stationarity residual, NOT ``||J'r||``.
        step_norm: ``()`` Euclidean ``||candidate step||``, reported even when
            the step is rejected.
        aux: residual aux output at the pre-step ``x`` (``has_aux=True``).
    """

    loss: jax.Array
    loss_old: jax.Array
    loss_candidate: jax.Array
    resid_loss: jax.Array
    penalty_value: jax.Array
    ridge: jax.Array
    accepted: jax.Array
    damping: jax.Array
    damping_factor: jax.Array
    used_geodesic: jax.Array
    acceleration_ratio: jax.Array
    grad_norm: jax.Array
    step_norm: jax.Array
    aux: Any = None


def ridge_continuation(*, decrease=0.1, ridge_floor, grad_rtol=1e-3, dtype=None):
    """Build ``(callback, user_state0)`` implementing ridge continuation.

    The callback multiplies ``lm_state.ridge`` by ``decrease`` whenever the
    inner fixed-ridge problem is approximately stationary -- ``info.grad_norm``
    below ``grad_rtol`` relative to its reference value at the current ridge
    level -- never below ``ridge_floor``. ``ridge_floor`` is REQUIRED and
    strictly positive (``ridge = 0`` is out of the solver's contract). The
    callback returns an :class:`~nlls_gram.LMSolveAction` replacing
    ``lm_state`` AND ``user_state`` (the per-level reference ``grad_norm``
    lives in ``user_state`` as a fixed-shape scalar, since a
    ``lax.while_loop`` carry cannot grow from ``None`` mid-loop -- hence the
    factory shape)::

        cb, us0 = ridge_continuation(ridge_floor=1e-10)
        result = solver.solve(x0, callback=cb, user_state=us0,
                              gtol=1e-8, atol=1e-8)

    The solved-out continuation path converges to the minimum-seminorm
    solution by nonlinear Tikhonov theory (Engl-Kunisch-Neubauer 1989;
    Engl-Hanke-Neubauer 1996), while annealing per accepted stationarity event
    rather than per fully solved level is the iteratively regularized
    Gauss-Newton method (Bakushinskii 1992; Blaschke-Neubauer-Scherzer 1997;
    Kaltenbacher-Neubauer-Scherzer 2008), whose theory wants exactly this kind
    of monotone, boundedly geometric schedule. ``dtype`` types the reference
    scalar in ``user_state0`` (default: the JAX default float; pass the
    problem dtype explicitly for a float32 program under enabled x64).
    """

    if not 0 < decrease < 1:
        raise ValueError("decrease must lie strictly between 0 and 1")
    concrete_floor = not isinstance(ridge_floor, (jax.Array, jax.core.Tracer))
    if concrete_floor and float(ridge_floor) <= 0:
        raise ValueError("ridge_floor must be strictly positive (ridge = 0 is "
                         "unsupported by RidgeLevenbergMarquardt)")
    if grad_rtol <= 0:
        raise ValueError("grad_rtol must be positive")
    if dtype is None:
        dtype = jnp.result_type(float)
    user_state0 = jnp.asarray(jnp.inf, dtype=dtype)

    def callback(ctx):
        ridge = ctx.lm_state.ridge
        grad_norm = jnp.asarray(ctx.info.grad_norm, dtype=ridge.dtype)
        reference = jnp.asarray(ctx.user_state, dtype=ridge.dtype)
        # +inf marks "no reference at this level yet": the first observation
        # after a ridge decrease (or the initial step) sets it.
        reference = jnp.where(jnp.isfinite(reference), reference, grad_norm)
        floor = jnp.asarray(ridge_floor, dtype=ridge.dtype)
        stationary = grad_norm <= jnp.asarray(grad_rtol, ridge.dtype) * reference
        new_ridge = jnp.where(
            stationary, jnp.maximum(ridge * jnp.asarray(decrease, ridge.dtype), floor),
            ridge,
        )
        # Reset the reference when the level actually changes; at the floor the
        # ridge is unchanged, so convergence is not suppressed and gtol/atol
        # can fire.
        new_reference = jnp.where(
            new_ridge < ridge, jnp.asarray(jnp.inf, ridge.dtype), reference
        )
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, ridge=new_ridge),
            user_state=new_reference.astype(jnp.asarray(ctx.user_state).dtype),
        )

    return callback, user_state0


class RidgeLevenbergMarquardt:
    """Levenberg-Marquardt for the ridge objective
    ``F(x) = ||r(x, args, p)||^2 + ridge * q(x)`` with ``q(x) = ||L x||^2``
    over a JAX pytree ``x``, built for underdetermined interpolation problems
    where the minimum-seminorm root ``argmin q s.t. r = 0`` is the target.

    The selection lives in the objective: under
    ``ker J ∩ ker L = {0}`` at the solution, each fixed-ridge problem has an
    isolated minimizer ``x_ridge`` with ``x_ridge -> x* = argmin q s.t. r = 0``
    at ``O(ridge)`` error as ``ridge -> 0`` (nonlinear Tikhonov theory:
    Engl-Kunisch-Neubauer 1989; Engl-Hanke-Neubauer 1996 Ch. 10; the seminorm
    formulation is Elden 1982). Plain Gauss-Newton on the underdetermined
    system converges to *some* root without selecting the minimal-norm one
    (Campbell-Kunkel-Bobinyec 2012; Pes-Rodriguez 2022), which is why the
    package's alternative is metric-damped
    :class:`~nlls_gram.LevenbergMarquardt`; this solver instead makes every
    inner problem a well-posed NLLS. Annealing ``ridge`` per stationarity
    event (:func:`ridge_continuation`) is the iteratively regularized
    Gauss-Newton method (Bakushinskii 1992; Kaltenbacher-Neubauer-Scherzer
    2008). For kernel penalties each inner step is a kernel ridge regression
    of the relinearized equations (Chen-Hosseini-Owhadi-Stuart 2021).

    Everywhere in the implementation the equivalent augmented-residual view is
    used: standard EUCLIDEAN LM on ``R(x) = [r(x); sqrt(ridge) L x]`` with
    ``A = [J; sqrt(ridge) L]`` and step
    ``(A'A + damping I) delta = -(J'r + ridge L'(L x))``. The penalty rows are
    affine in ``x``, so geodesic acceleration (Transtrum-Sethna 2012),
    accept/reject, and the implicit-AD rule reduce to the standard formulas
    with second-derivative contributions from the penalty identically zero.
    ``damping`` (``mu``) is a plain trust-region parameter, fully decoupled
    from ``ridge`` (``lambda``): ``mu`` moves every step by accept/reject,
    ``lambda`` only when a callback replaces ``lm_state.ridge``.

    ``ridge`` is STRICTLY POSITIVE by contract -- the constructor validates
    it, callbacks must keep it positive, and continuation floors are positive
    -- so ``J'J + ridge L'L`` is positive definite under the identification
    condition at every reachable state. ``ridge=None`` resolves at ``init``
    to ``sqrt(finfo(dtype).eps)`` of the residual dtype (float64 ~ 1.5e-8,
    float32 ~ 3.5e-4).

    ``linear_solver`` picks the algebra for
    ``(J'J + ridge L'L + damping I) delta = -g``; all three share one
    factorization between the velocity and geodesic-acceleration solves:

    - ``"auto"`` (= ``"cholesky"``, the default): dense normal equations.
      ``J'`` is materialized per ``jacobian_mode``, promoted to
      ``linear_solve_dtype`` BEFORE the Gram product, the penalty added via
      ``penalty.add_scaled``, and the assembled ``G = J'J + ridge L'L`` is
      cached across rejected steps (only the damping shift re-factors).
    - ``"qr"``: MINPACK-structured damping-row QR (More 1978), stable at
      small ``ridge``/``damping`` where forming ``G`` squares the condition
      number. The ``R`` factor of ``[J; sqrt(ridge) L]`` is cached per
      ``(x, ridge)``; each update stacks ``[R; sqrt(damping) I]``, re-factors,
      and solves by corrected semi-normal equations with one fixed
      iterative-refinement pass (Bjorck 1987; Bjorck 1996 Sec. 6.6.5).
    - ``"lsmr"``: matrix-free bidiagonalization (Fong-Saunders 2011) on the
      right-preconditioned augmented operator, requiring an explicit
      ``lsmr_preconditioner`` (a
      :class:`~nlls_gram.WhitenedPreconditioner`;
      ``identity_right_preconditioner()`` opts out). The augmented damping
      row is posed in the unpreconditioned variable, so every
      ``damping > 0`` subproblem is exactly the ``I``-damped problem for any
      right preconditioner (Bjorck 1996 Ch. 7).

    Stopping (``solve``): ``gtol`` bounds ``info.grad_norm = ||J'r + ridge
    L'L x||`` (ridge stationarity) and ``xtol`` the accepted Euclidean step
    norm -- they mean "done with the current fixed-ridge problem". ``atol``
    is a CONJUNCTIVE filter on the TRUE residual: convergence requires
    (``gtol`` or ``xtol`` fired) AND (``atol == 0`` or ``sqrt(resid_loss) <=
    atol``). atol alone never stops the solve -- a pure-residual test would
    stop at step 0 from any interpolating start before the seminorm is
    minimized -- so ``solve`` rejects ``atol > 0`` without a positive
    ``gtol`` or ``xtol``. In a continuation run the callback keeps lowering
    ``ridge``, so intermediate stationarity with a large residual correctly
    does not stop the loop.

    ``solve(...).x`` has a custom implicit AD rule with respect to ``p``:
    Gauss-Newton implicit differentiation of the ridge stationarity
    ``J'r + ridge M0 x = 0``, i.e.
    ``(J'J + ridge M0) x_dot = -J'(dr/dp) p_dot`` with ``ridge`` frozen
    (stop-gradient) at the returned state's value -- the continuation
    schedule's and multi-start selection's dependence on ``p`` is
    deliberately ignored, and no damping enters the AD matrix. Exact
    differentiation carries two extra terms (``sum_i r_i * d2r_i/dx2`` in the
    matrix and ``(dJ'/dp) r`` on the right); both are exactly zero for an
    affine residual and first order in ``||r||`` otherwise, so the rule is
    exact when the converged TRUE residual vanishes. The caveat to state
    plainly: the first-order-in-``||r||`` absolute error translates to a
    small RELATIVE tangent error only under conditioning assumptions
    (``J'J`` bounded below). On a genuinely underdetermined CURVED system the
    null-space block of the exact tangent carries the constraint-curvature
    term ``sum_i nu_i d2r_i`` with multiplier ``nu = r/ridge`` -- which does
    not vanish relative to ``ridge * M0`` as ``ridge -> 0`` -- so tangent
    components read off null-space directions retain an O(1)-in-``ridge``
    Gauss-Newton bias proportional to the residual curvature.
    ``ad_solver="cholesky"`` assembles and factors (wide under
    ``linear_solve_dtype``); ``"normal_cg"`` runs matrix-free CG on the same
    operator with the optional ``ad_solver_preconditioner`` hook (PD does not
    mean well-conditioned -- unpreconditioned CG degrades as ``ridge``
    shrinks). ``"auto"`` resolves to ``"cholesky"`` for the dense forwards
    and ``"normal_cg"`` for ``"lsmr"``. Failed statuses return exact zero
    tangents for ``result.x``/``result.aux`` and evaluate the masked tangent
    program at stop-gradient copies of the caller's original inputs and the
    INITIAL ridge (never a possibly-invalid callback-produced value).

    The init/update/solve protocol, callback contract
    (:class:`~nlls_gram.LMSolveContext` ->
    :class:`~nlls_gram.LMSolveAction`), ``multi_start``, ``save_steps``, and
    :class:`~nlls_gram.LMSolveResult` are shared with
    :class:`~nlls_gram.LevenbergMarquardt`; code written against that solver
    ports by changing the constructor (metric -> penalty) and reading
    ``info.resid_loss`` where it means equation error, since ``info.loss``
    here includes the penalty. Multi-start ranking uses the ridge objective
    at each lane's own final ridge -- comparable across lanes when they share
    a continuation schedule.
    """

    def __init__(
        self,
        residual_fn,
        *,
        penalty,
        penalty_factory=None,
        ridge=None,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        min_damping=None,
        max_damping=None,
        linear_solver="auto",
        jacobian_mode="auto",
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=8,
        lsmr_preconditioner=None,
        ad_solver="auto",
        ad_solver_tol=None,
        ad_solver_atol=0.0,
        ad_solver_maxiter=None,
        ad_solver_preconditioner=None,
        linear_solve_dtype=None,
        has_aux=False,
        cache_jacobian=True,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    ):
        canonical_residual, residual_arity = canonicalize_residual(residual_fn)
        if penalty_factory is not None:
            raise NotImplementedError(
                "penalty_factory is reserved for a future release (its contract "
                "must restrict the factor to differentiation-inert data); pass "
                "a fixed penalty"
            )
        if not isinstance(penalty, RidgePenalty):
            raise TypeError("penalty must be a RidgePenalty")
        if linear_solver not in ("auto", "cholesky", "qr", "lsmr"):
            raise ValueError(f"unknown linear_solver: {linear_solver}")
        if jacobian_mode not in ("auto", "fwd", "rev"):
            raise ValueError(f"unknown jacobian_mode: {jacobian_mode}")
        if ridge is not None:
            if jnp.ndim(ridge) != 0:
                raise ValueError("ridge must be a scalar or None")
            if (
                not isinstance(ridge, (jax.Array, jax.core.Tracer))
                and float(ridge) <= 0.0
            ):
                raise ValueError(
                    "ridge must be strictly positive (ridge = 0 is unsupported: "
                    "use ridge_continuation with a positive ridge_floor to "
                    "approach the ridgeless limit)"
                )
        if init_damping <= 0:
            raise ValueError("init_damping must be positive")
        if damping_decrease <= 0:
            raise ValueError("damping_decrease must be positive")
        if damping_increase <= 0:
            raise ValueError("damping_increase must be positive")
        if min_damping is not None and min_damping <= 0:
            raise ValueError("min_damping must be positive or None")
        if min_damping is not None and min_damping > init_damping:
            raise ValueError("min_damping must not exceed init_damping")
        if max_damping is not None and max_damping < init_damping:
            raise ValueError("max_damping must be at least init_damping")
        if iterative_tol < 0:
            raise ValueError("iterative_tol must be nonnegative")
        if iterative_atol < 0:
            raise ValueError("iterative_atol must be nonnegative")
        if iterative_maxiter is not None and iterative_maxiter <= 0:
            raise ValueError("iterative_maxiter must be positive or None")
        if lsmr_preconditioner is not None and linear_solver != "lsmr":
            raise ValueError('lsmr_preconditioner requires linear_solver="lsmr"')
        if linear_solver == "lsmr" and lsmr_preconditioner is None:
            raise ValueError(
                'linear_solver="lsmr" requires lsmr_preconditioner, a '
                "WhitenedPreconditioner applying a parameter-space right "
                "preconditioner; pass identity_right_preconditioner() to run "
                "unpreconditioned LSMR"
            )
        if ad_solver not in ("auto", "cholesky", "normal_cg"):
            raise ValueError(f"unknown ad_solver: {ad_solver}")
        if ad_solver_tol is not None and ad_solver_tol < 0:
            raise ValueError("ad_solver_tol must be nonnegative or None")
        if ad_solver_atol < 0:
            raise ValueError("ad_solver_atol must be nonnegative")
        if ad_solver_maxiter is not None and ad_solver_maxiter <= 0:
            raise ValueError("ad_solver_maxiter must be positive or None")
        if (
            ad_solver == "normal_cg"
            and ad_solver_tol == 0
            and ad_solver_atol == 0
            and ad_solver_maxiter is None
        ):
            raise ValueError(
                "ad_solver_maxiter must be set when both ad_solver tolerances are zero"
            )
        # jacobian_mode is consumed by the dense forward paths and the
        # cholesky AD method; an lsmr forward with a cg AD rule never
        # materializes J, so a forced mode is a construction error there.
        if (
            jacobian_mode != "auto"
            and linear_solver == "lsmr"
            and ad_solver != "cholesky"
        ):
            raise ValueError(
                "jacobian_mode controls dense Jacobian assembly, but neither "
                'linear_solver="lsmr" nor the '
                f'"{ad_solver}"-resolved ad_solver ever consumes it'
            )
        cg_resolved_possible = ad_solver == "normal_cg" or (
            ad_solver == "auto" and linear_solver == "lsmr"
        )
        if ad_solver_preconditioner is not None and not cg_resolved_possible:
            raise ValueError(
                "ad_solver_preconditioner requires a cg-resolved ad_solver: "
                'ad_solver="normal_cg", or "auto" alongside an lsmr forward '
                "solver"
            )
        if linear_solve_dtype is not None:
            if jnp.dtype(linear_solve_dtype) != jnp.dtype(jnp.float64):
                raise ValueError("linear_solve_dtype must be None or jnp.float64")
            if linear_solver == "lsmr" and ad_solver != "cholesky":
                raise ValueError(
                    "linear_solve_dtype promotes only the dense linear-solve "
                    "pipelines; it requires a cholesky, qr, or auto forward "
                    'solver, or ad_solver="cholesky"'
                )
            if not jax.config.jax_enable_x64:
                raise ValueError(
                    "linear_solve_dtype=jnp.float64 requires x64 support; call "
                    'jax.config.update("jax_enable_x64", True) at startup '
                    "(explicitly float32 problem data stays float32)"
                )
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.penalty = penalty
        self.ridge = ridge
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.linear_solver = linear_solver
        self.jacobian_mode = jacobian_mode
        self.iterative_tol = iterative_tol
        self.iterative_atol = iterative_atol
        self.iterative_maxiter = iterative_maxiter
        self.lsmr_preconditioner = lsmr_preconditioner
        self.ad_solver = ad_solver
        self.ad_solver_tol = ad_solver_tol
        self.ad_solver_atol = ad_solver_atol
        self.ad_solver_maxiter = ad_solver_maxiter
        self.ad_solver_preconditioner = (
            None
            if ad_solver_preconditioner is None
            else canonicalize_ad_preconditioner(ad_solver_preconditioner)
        )
        self.linear_solve_dtype = (
            None if linear_solve_dtype is None else jnp.dtype(linear_solve_dtype)
        )
        self.has_aux = has_aux
        # Only the dense paths materialize J' (and the cholesky/qr caches ride
        # on the same reject-reuse lifecycle), so the flag is inert for lsmr.
        self.cache_jacobian = cache_jacobian and linear_solver in (
            "auto",
            "cholesky",
            "qr",
        )
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        # Resolved penalty callbacks. The defaults derive from sqrt_apply, so
        # the objective's two penalty forms (||L x||^2 at the current iterate,
        # quadratic() at trial points) agree exactly.
        self._sqrt_apply = penalty.sqrt_apply
        self._sqrt_transpose_apply = penalty.sqrt_transpose_apply
        if penalty.quadratic is not None:
            self._quadratic = penalty.quadratic
        else:

            def default_quadratic(v):
                factor_apply = penalty.sqrt_apply(v)
                return jnp.sum(factor_apply**2)

            self._quadratic = default_quadratic
        if penalty.add_scaled is not None:
            self._add_scaled = penalty.add_scaled
        else:

            def default_add_scaled(H, c):
                rows = self._penalty_rows(H.shape[0], H.dtype).astype(H.dtype)
                return H + c * (rows.T @ rows)

            self._add_scaled = default_add_scaled
        # Value-based identity: the jitted solve loop marks the solver itself
        # static, so equal-config solvers built around the same residual and
        # penalty share the compiled loop across instances. Keyed on the
        # constructor arguments -- every derived attribute is a function of
        # them.
        self._static_key = tuple(
            _static_key_component(value)
            for value in (
                residual_fn,
                penalty,
                ridge,
                init_damping,
                damping_decrease,
                damping_increase,
                min_damping,
                max_damping,
                linear_solver,
                jacobian_mode,
                iterative_tol,
                iterative_atol,
                iterative_maxiter,
                lsmr_preconditioner,
                ad_solver,
                ad_solver_tol,
                ad_solver_atol,
                ad_solver_maxiter,
                ad_solver_preconditioner,
                self.linear_solve_dtype,
                has_aux,
                self.cache_jacobian,
                geodesic_acceleration,
                geodesic_acceptance_ratio,
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
            _damping_floor(self.min_damping, dtype),
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

    def _resolved_solver(self):
        return "cholesky" if self.linear_solver == "auto" else self.linear_solver

    def _penalty_rows(self, theta_size, dtype):
        # The dense (k, p) factor L: the penalty's own sqrt_rows when
        # provided, else assembled from sqrt_apply on an identity basis.
        if self.penalty.sqrt_rows is not None:
            return jnp.asarray(self.penalty.sqrt_rows())
        basis = jnp.eye(theta_size, dtype=dtype)
        return jax.vmap(self.penalty.sqrt_apply)(basis).T

    def _resolve_ridge(self, dtype):
        if self.ridge is None:
            return jnp.asarray(jnp.sqrt(jnp.finfo(dtype).eps), dtype=dtype)
        return jnp.asarray(self.ridge, dtype=dtype)

    def init(self, x0, args=None, *, p=None):
        """Build the initial :class:`RidgeLMState` at ``x0``.

        One residual evaluation types ``damping`` and resolves
        ``ridge=None`` to the dtype default, and sizes the Jacobian/normal/QR
        cache buffers for the configured dense path. ``hyper`` stays ``None``
        so manual ``update`` loops carry no extra buffers; ``solve``
        populates it for its callbacks.
        """
        self._check_residual_args(args, p)
        residual, aux = self._residual_and_aux(x0, args, p)
        dtype = residual.dtype
        min_damping = _damping_floor(self.min_damping, dtype)
        damping = jnp.maximum(
            jnp.asarray(self.init_damping, dtype=dtype), min_damping
        )
        ridge = self._resolve_ridge(dtype)
        if not self.cache_jacobian:
            return RidgeLMState(damping, ridge)
        theta, _ = ravel_pytree(x0)
        p_dim = theta.size
        m = residual.size
        dense_dtype = (
            dtype if self.linear_solve_dtype is None else self.linear_solve_dtype
        )
        invalid = jnp.asarray(False, dtype=jnp.bool_)
        common = dict(
            resid=jnp.zeros(residual.shape, dtype=dtype),
            Jt=jnp.zeros((p_dim, m), dtype=dtype),
            jacobian_valid=invalid,
            aux=jax.tree.map(jnp.zeros_like, aux),
        )
        if self._resolved_solver() == "cholesky":
            return RidgeLMState(
                damping,
                ridge,
                **common,
                G=jnp.zeros((p_dim, p_dim), dtype=dense_dtype),
                G_valid=invalid,
                G_ridge=jnp.zeros((), dtype=dtype),
            )
        r_rows = min(m + self.penalty.num_rows, p_dim)
        return RidgeLMState(
            damping,
            ridge,
            **common,
            qr_R=jnp.zeros((r_rows, p_dim), dtype=dense_dtype),
            qr_valid=invalid,
            qr_ridge=jnp.zeros((), dtype=dtype),
        )

    def _resolve_jacobian_mode(self, m, n):
        # Static (shape-driven) choice of dense Jacobian assembly on the TRUE
        # residual: "auto" takes n forward-mode columns when the system is
        # tall or square (n <= m) and m reverse-mode rows only when strictly
        # fat, so the identity basis being vmapped is always the small side.
        # The penalty rows are affine and never assembled through AD.
        if self.jacobian_mode != "auto":
            return self.jacobian_mode
        return "fwd" if n <= m else "rev"

    def _assemble_jt(self, jvp_fn, theta, resid):
        if self._resolve_jacobian_mode(resid.shape[0], theta.shape[0]) == "fwd":
            parameter_basis = jnp.eye(theta.shape[0], dtype=theta.dtype)
            return jax.vmap(jvp_fn)(parameter_basis)
        transpose_fn = jax.linear_transpose(jvp_fn, theta)
        residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
        return jax.vmap(lambda cotangent: transpose_fn(cotangent)[0])(residual_basis).T

    def _dense_resid_jt_aux(self, residual_flat, theta):
        if self.has_aux:
            resid, jvp_fn, aux = jax.linearize(residual_flat, theta, has_aux=True)
        else:
            resid, jvp_fn = jax.linearize(residual_flat, theta)
            aux = None
        return resid, self._assemble_jt(jvp_fn, theta, resid), aux

    def _residual_and_aux(self, x, args, p):
        if self.has_aux:
            value, aux = self.residual_fn(x, args, p)
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
        resid_loss = jnp.sum(residual**2)
        theta, _ = ravel_pytree(x)
        if lm_state.ridge is None:
            raise ValueError(
                "the lm_state has no ridge; create it with init(x, args, p=p)"
            )
        ridge = jnp.asarray(lm_state.ridge, dtype=residual.dtype)
        penalty_value = jnp.asarray(self._quadratic(theta), dtype=residual.dtype)
        loss = resid_loss + ridge * penalty_value
        zero = jnp.zeros((), dtype=residual.dtype)
        one = jnp.ones((), dtype=residual.dtype)
        return RidgeLMInfo(
            loss,
            loss,
            loss,
            resid_loss,
            penalty_value,
            ridge,
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
        """One LM step on the ridge objective: returns ``(x_new, state, info)``."""
        self._check_residual_args(args, p)
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

        # TRUE-residual linearization: matrix-free closures for lsmr, dense
        # J' (reused from the cache after a rejected step) otherwise.
        if self.linear_solver == "lsmr":
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
                return self._dense_resid_jt_aux(residual_flat, theta)

            def reuse_resid_and_jt(_):
                return lm_state.resid, lm_state.Jt, lm_state.aux

            resid, Jt, aux = jax.lax.cond(
                lm_state.jacobian_valid,
                reuse_resid_and_jt,
                compute_resid_and_jt,
                operand=None,
            )
        else:
            resid, Jt, aux = self._dense_resid_jt_aux(residual_flat, theta)

        hyper = (
            lm_state.hyper
            if lm_state.hyper is not None
            else self.hyperparams(resid.dtype)
        )
        damping_decrease = jnp.asarray(hyper.damping_decrease, dtype=resid.dtype)
        damping_increase = jnp.asarray(hyper.damping_increase, dtype=resid.dtype)
        min_damping = _damping_floor(hyper.min_damping, resid.dtype)
        damping = jnp.maximum(
            jnp.asarray(lm_state.damping, dtype=resid.dtype), min_damping
        )
        if lm_state.ridge is None:
            raise ValueError(
                "the lm_state has no ridge; create it with init(x, args, p=p)"
            )
        ridge = jnp.asarray(lm_state.ridge, dtype=resid.dtype)

        # Half-gradient g = J'r + ridge L'(L x); grad F = 2 g, the factor
        # cancels in the LM equations. Lx is reused for the pre-step penalty
        # value (identical to the resolved quadratic's default form). Penalty
        # callback outputs are pinned to the residual dtype so a wider-typed
        # penalty (e.g. float64 kernel data under a float32 residual) cannot
        # promote the gradient and break the loop-carry dtypes.
        factor_x = self._sqrt_apply(theta)
        penalty_value_old = jnp.asarray(jnp.sum(factor_x**2), dtype=resid.dtype)
        penalty_gradient = jnp.asarray(
            self._sqrt_transpose_apply(factor_x), dtype=resid.dtype
        )

        resolved_solver = self._resolved_solver()
        dense_dtype = (
            resid.dtype if self.linear_solve_dtype is None else self.linear_solve_dtype
        )

        if self.linear_solver == "lsmr":
            transpose_fn = jax.linear_transpose(jvp_fn, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

            grad = JT(resid) + ridge * penalty_gradient
            lsmr_tol = jnp.asarray(hyper.iterative_tol, dtype=resid.dtype)
            lsmr_atol = jnp.asarray(hyper.iterative_atol, dtype=resid.dtype)
            sqrt_damping = jnp.sqrt(damping)
            sqrt_ridge = jnp.sqrt(ridge)
            m = resid.shape[0]
            k = self.penalty.num_rows
            n = theta.shape[0]
            # None (uncapped) has no meaning for a fixed-shape loop; min(m+k, n)
            # is the augmented bidiagonalization's exact-arithmetic bound.
            lsmr_maxiter = (
                hyper.iterative_maxiter
                if hyper.iterative_maxiter is not None
                else 4 * min(m + k, n)
            )

            def apply_Rinv(v):
                return self.lsmr_preconditioner.solve(v, damping)

            def apply_RinvT(w):
                return self.lsmr_preconditioner.solve_transpose(w, damping)

            # Augmented operator A = [J; sqrt(ridge) L] with the penalty rows
            # applied through the penalty's own factor callbacks (outputs
            # pinned to the residual dtype).
            def A_matvec(u):
                penalty_rows_apply = jnp.asarray(
                    sqrt_ridge * self._sqrt_apply(u), dtype=resid.dtype
                )
                return jnp.concatenate([jvp_fn(u), penalty_rows_apply])

            def At_matvec(w):
                penalty_pullback = jnp.asarray(
                    sqrt_ridge * self._sqrt_transpose_apply(w[m:]),
                    dtype=resid.dtype,
                )
                return JT(w[:m]) + penalty_pullback

            # N = A'A + damping I, the R-free SPD operator that
            # custom_linear_solve differentiates through, posed on u (not z).
            def N_matvec(u):
                return At_matvec(A_matvec(u)) + damping * u

            def solve_N(_, c):
                # Solve (A'A + damping I) u = c by LSMR on the R-preconditioned
                # damped-augmented operator [A R^{-1}; sqrt(damping) R^{-1}] in
                # z = R u: its normal equations read
                # (A'A + damping I) u = A' b1 + sqrt(damping) b2, so
                # b_aug = [0_{m+k}; c / sqrt(damping)] targets c exactly, at
                # condition sqrt(cond(N)).
                def A_aug(zz):
                    u_zz = apply_Rinv(zz)
                    return jnp.concatenate([A_matvec(u_zz), sqrt_damping * u_zz])

                def At_aug(ww):
                    return apply_RinvT(
                        At_matvec(ww[: m + k]) + sqrt_damping * ww[m + k :]
                    )

                b_aug = jnp.concatenate(
                    [jnp.zeros(m + k, resid.dtype), c / sqrt_damping]
                )
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
                return apply_Rinv(z)

            def solve_step(rhs):
                return jax.lax.custom_linear_solve(
                    N_matvec, -rhs, solve=solve_N, transpose_solve=solve_N,
                    symmetric=True,
                )

            def accel_rhs(f_vv):
                return JT(f_vv)

        elif resolved_solver == "cholesky":
            grad = Jt @ resid + ridge * penalty_gradient
            # Wide pipeline: J' promoted BEFORE the Gram product and the
            # penalty added in the wide dtype (a narrow-rounded ridge*L'L
            # added into a wide H would defeat the promotion). The assembled
            # G = J'J + ridge L'L is cached across rejected steps -- only mu
            # changed, so the reject pays the p^3/3 refactor without the GEMM
            # or the penalty assembly; a callback ridge change invalidates
            # through the G_ridge key.
            transposed_jacobian = Jt.astype(dense_dtype)
            ridge_wide = jnp.asarray(ridge, dtype=dense_dtype)

            def assemble_normal(_):
                gram = transposed_jacobian @ transposed_jacobian.T
                return self._add_scaled(gram, ridge_wide)

            if self.cache_jacobian:
                if lm_state.G_valid is None:
                    raise ValueError(
                        "the lm_state has no normal-matrix cache for the "
                        "cholesky path; create the lm_state with "
                        "init(x, args, p=p)"
                    )
                normal_matrix = jax.lax.cond(
                    lm_state.G_valid & (lm_state.G_ridge == ridge),
                    lambda _: lm_state.G,
                    assemble_normal,
                    operand=None,
                )
            else:
                normal_matrix = assemble_normal(None)
            shifted = normal_matrix + jnp.asarray(
                damping, dtype=dense_dtype
            ) * jnp.eye(theta.shape[0], dtype=dense_dtype)
            factor = jsp_linalg.cho_factor(shifted)

            def solve_step(rhs):
                return -jsp_linalg.cho_solve(factor, rhs.astype(dense_dtype)).astype(
                    resid.dtype
                )

            def accel_rhs(f_vv):
                return Jt @ f_vv

        else:  # qr
            grad = Jt @ resid + ridge * penalty_gradient
            transposed_jacobian = Jt.astype(dense_dtype)
            ridge_wide = jnp.asarray(ridge, dtype=dense_dtype)
            damping_wide = jnp.asarray(damping, dtype=dense_dtype)
            penalty_rows = self._penalty_rows(theta.shape[0], resid.dtype).astype(
                dense_dtype
            )

            def assemble_r(_):
                stacked = jnp.concatenate(
                    [transposed_jacobian.T, jnp.sqrt(ridge_wide) * penalty_rows],
                    axis=0,
                )
                return jnp.linalg.qr(stacked, mode="r")

            if self.cache_jacobian:
                if lm_state.qr_valid is None:
                    raise ValueError(
                        "the lm_state has no QR cache for the qr path; create "
                        "the lm_state with init(x, args, p=p)"
                    )
                qr_R = jax.lax.cond(
                    lm_state.qr_valid & (lm_state.qr_ridge == ridge),
                    lambda _: lm_state.qr_R,
                    assemble_r,
                    operand=None,
                )
            else:
                qr_R = assemble_r(None)
            # Per-update damping-row refactor (More 1978's structure via a
            # plain stacked QR): R_mu'R_mu = R'R + damping I = A'A + damping I.
            # When m + k < p the cached R is upper trapezoidal and these
            # damping rows are what make the final system full rank.
            damped_stack = jnp.concatenate(
                [
                    qr_R,
                    jnp.sqrt(damping_wide)
                    * jnp.eye(theta.shape[0], dtype=dense_dtype),
                ],
                axis=0,
            )
            R_mu = jnp.linalg.qr(damped_stack, mode="r")

            def damped_normal_matvec(v):
                return (
                    transposed_jacobian @ (transposed_jacobian.T @ v)
                    + ridge_wide * (penalty_rows.T @ (penalty_rows @ v))
                    + damping_wide * v
                )

            def solve_step(rhs):
                # Corrected semi-normal equations (Bjorck 1987): triangular
                # solves against R_mu, then ONE fixed iterative-refinement
                # pass through matvecs (Bjorck 1996 Sec. 6.6.5).
                b = -rhs.astype(dense_dtype)
                half = jsp_linalg.solve_triangular(R_mu.T, b, lower=True)
                delta = jsp_linalg.solve_triangular(R_mu, half, lower=False)
                correction_rhs = b - damped_normal_matvec(delta)
                half = jsp_linalg.solve_triangular(R_mu.T, correction_rhs, lower=True)
                delta = delta + jsp_linalg.solve_triangular(R_mu, half, lower=False)
                return delta.astype(resid.dtype)

            def accel_rhs(f_vv):
                return Jt @ f_vv

        # First-order step (velocity) and its ridge objective.
        velocity = solve_step(grad)
        resid_velocity = residual_value(theta + velocity)
        resid_loss_old = jnp.sum(resid**2)
        loss_old = resid_loss_old + ridge * penalty_value_old
        resid_loss_velocity = jnp.sum(resid_velocity**2)
        penalty_velocity = jnp.asarray(
            self._quadratic(theta + velocity), dtype=resid.dtype
        )
        loss_velocity = resid_loss_velocity + ridge * penalty_velocity
        zero = jnp.zeros((), dtype=resid.dtype)

        # Geodesic second-order correction, sharing the factorization. The
        # penalty rows are affine, so their directional second derivative is
        # identically zero: f_vv comes from the TRUE residual only and the
        # correction RHS is J' f_vv.
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
            acceleration = solve_step(accel_rhs(f_vv))
            accelerated_step = velocity + 0.5 * acceleration
            acceleration_ratio = (
                2.0
                * jnp.linalg.norm(acceleration)
                / (jnp.linalg.norm(velocity) + jnp.finfo(resid.dtype).eps)
            )
            ratio_accepted = (
                (geodesic_acceptance_ratio > zero)
                & (acceleration_ratio > zero)
                & (acceleration_ratio <= geodesic_acceptance_ratio)
            )

            def accelerated_objective(_):
                resid_accelerated = residual_value(theta + accelerated_step)
                accel_resid_loss = jnp.sum(resid_accelerated**2)
                accel_penalty = jnp.asarray(
                    self._quadratic(theta + accelerated_step), dtype=resid.dtype
                )
                return accel_resid_loss, accel_penalty

            inf = jnp.asarray(jnp.inf, dtype=resid.dtype)
            resid_loss_accelerated, penalty_accelerated = jax.lax.cond(
                ratio_accepted,
                accelerated_objective,
                lambda _: (inf, inf),
                operand=None,
            )
            loss_accelerated = resid_loss_accelerated + ridge * penalty_accelerated
            used_geodesic = ratio_accepted & (loss_accelerated <= loss_velocity)
            step = jnp.where(used_geodesic, accelerated_step, velocity)
            loss_candidate = jnp.where(used_geodesic, loss_accelerated, loss_velocity)
            resid_loss_candidate = jnp.where(
                used_geodesic, resid_loss_accelerated, resid_loss_velocity
            )
            penalty_candidate = jnp.where(
                used_geodesic, penalty_accelerated, penalty_velocity
            )
        else:
            step = velocity
            loss_candidate = loss_velocity
            resid_loss_candidate = resid_loss_velocity
            penalty_candidate = penalty_velocity
            used_geodesic = jnp.asarray(False)
            acceleration_ratio = zero

        # Accept iff the ridge objective strictly decreases and is finite --
        # computed with the SAME ridge on both sides, so per-step monotonicity
        # is well-defined even when a callback anneals ridge between steps.
        improved = jnp.isfinite(loss_candidate) & (loss_candidate < loss_old)
        theta_new = jnp.where(improved, theta + step, theta)
        damping_factor = jnp.where(improved, damping_decrease, damping_increase)
        new_damping = damping * damping_factor
        if hyper.max_damping is not None:
            new_damping = jnp.minimum(
                new_damping,
                jnp.maximum(
                    jnp.asarray(hyper.max_damping, dtype=resid.dtype), min_damping
                ),
            )
        new_damping = jnp.maximum(new_damping, min_damping)
        loss = jnp.where(improved, loss_candidate, loss_old)
        resid_loss = jnp.where(improved, resid_loss_candidate, resid_loss_old)
        penalty_value = jnp.where(improved, penalty_candidate, penalty_value_old)
        # Thread the caches built at this step's pre-step (x, ridge):
        # valid = ~improved marks them reusable exactly when the step was
        # rejected (x did not move). ridge passes through unchanged -- only
        # init() and callbacks set it. The input hyper (not the fallback)
        # passes through so the loop carry structure is stable.
        if self.cache_jacobian:
            if resolved_solver == "cholesky":
                new_lm_state = RidgeLMState(
                    new_damping,
                    ridge,
                    resid,
                    Jt,
                    ~improved,
                    aux,
                    lm_state.hyper,
                    G=normal_matrix,
                    G_valid=~improved,
                    G_ridge=ridge,
                )
            else:
                new_lm_state = RidgeLMState(
                    new_damping,
                    ridge,
                    resid,
                    Jt,
                    ~improved,
                    aux,
                    lm_state.hyper,
                    qr_R=qr_R,
                    qr_valid=~improved,
                    qr_ridge=ridge,
                )
        else:
            new_lm_state = RidgeLMState(new_damping, ridge, hyper=lm_state.hyper)
        return (
            unravel(theta_new),
            new_lm_state,
            RidgeLMInfo(
                loss,
                loss_old,
                loss_candidate,
                resid_loss,
                penalty_value,
                ridge,
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
        max_steps_is_success=True,
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

        Parameters are the same as ``update`` plus loop controls, matching
        :meth:`LevenbergMarquardt.solve
        <nlls_gram.LevenbergMarquardt.solve>` (callbacks, ``save_steps``,
        ``multi_start``, ``max_steps_is_success``, ``jit``) with two
        differences. First, the tolerance semantics are conjunctive:
        ``gtol`` bounds the ridge stationarity ``||J'r + ridge L'L x||`` and
        ``xtol`` the accepted Euclidean step norm -- either fires "done with
        the current fixed-ridge problem" -- while ``atol > 0`` ADDITIONALLY
        requires ``sqrt(resid_loss) <= atol`` (the model equations actually
        solved, the ridgeless-endgame check) and never stops the solve alone;
        ``atol > 0`` therefore requires a positive ``gtol`` or ``xtol``
        (validated loudly). Second, ``lm_state=None`` always builds the state
        with :meth:`init` (resolving ``ridge=None`` needs the residual
        dtype); a caller-supplied ``lm_state`` must carry a positive
        ``ridge``.

        For ridge continuation pass the pair returned by
        :func:`ridge_continuation` as ``callback``/``user_state``. A callback
        ridge change is a problem change: that step's convergence test is
        suppressed and the ridge-keyed caches invalidate.
        """
        self._check_residual_args(args, p)
        if not isinstance(max_steps_is_success, bool):
            raise TypeError("max_steps_is_success must be a bool")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        # Tolerances are traced data inside the loop, so vmapped/traced values
        # skip the concrete-only validation.
        atol_concrete = not isinstance(atol, jax.core.Tracer)
        gtol_concrete = not isinstance(gtol, jax.core.Tracer)
        xtol_concrete = not isinstance(xtol, jax.core.Tracer)
        if atol_concrete and atol < 0:
            raise ValueError("atol must be nonnegative")
        if gtol_concrete and gtol < 0:
            raise ValueError("gtol must be nonnegative")
        if xtol_concrete and xtol < 0:
            raise ValueError("xtol must be nonnegative")
        if (
            atol_concrete
            and atol > 0
            and (gtol_concrete and gtol == 0)
            and (xtol_concrete and xtol == 0)
        ):
            raise ValueError(
                "atol > 0 requires a positive gtol or xtol: atol is a "
                "conjunctive filter on the TRUE residual, never a stopping "
                "rule by itself -- a residual-only test would stop at any "
                "interpolating iterate before the seminorm is minimized"
            )
        if lm_state is None:
            # Unconditional init (no minimal-state fast path): resolving
            # ridge=None needs the residual dtype, and the dense caches need
            # their shapes.
            lm_state = self.init(x0, args, p=p)
        else:
            if lm_state.ridge is None:
                raise ValueError(
                    "the caller-supplied lm_state has no ridge; create it "
                    "with init(x, args, p=p) or set a positive ridge"
                )
            if (
                not isinstance(lm_state.ridge, jax.core.Tracer)
                and jnp.ndim(lm_state.ridge) == 0
                and float(lm_state.ridge) <= 0.0
            ):
                raise ValueError(
                    "the caller-supplied lm_state.ridge must be strictly "
                    "positive (ridge = 0 is unsupported)"
                )
            # Recast a hand-replaced ridge to the carried scalar dtype: a
            # weak-typed `dataclasses.replace(state, ridge=1e-4)` would
            # otherwise change the jit input aval and retrace the loop.
            lm_state = dataclasses.replace(
                lm_state,
                ridge=jnp.asarray(
                    lm_state.ridge, dtype=jnp.asarray(lm_state.damping).dtype
                ),
            )
        if lm_state.hyper is None:
            lm_state = dataclasses.replace(lm_state, hyper=self.hyperparams())
        history_len = max_steps + 1 if save_steps else None

        if multi_start is not None:
            if not isinstance(multi_start, MultiStart):
                raise TypeError("multi_start must be a MultiStart or None")
            num_starts = multi_start.num_starts
            draw = _hashable_hook(multi_start.draw if num_starts > 1 else None)
            default_accept = (
                _accept_converged_or_max_steps
                if max_steps_is_success
                else _accept_converged
            )
            accept = _hashable_hook(
                default_accept if multi_start.accept is None else multi_start.accept
            )
            parallel = multi_start.parallel and num_starts > 1
            if draw is not None and jit:
                drawn = jax.eval_shape(draw, multi_start.key, x0, args)
                _check_drawn_types(x0, args, drawn)

            @jax.custom_jvp
            def solve_multi_start_with_ad_p(
                x,
                lm_state,
                args,
                p,
                user_state,
                key,
                max_steps,
                atol,
                gtol,
                xtol,
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

            @solve_multi_start_with_ad_p.defjvp
            def solve_multi_start_with_ad_p_jvp(primals, tangents):
                p_dot = tangents[3]
                result = solve_multi_start_with_ad_p(*primals)
                initial_ad_point = (
                    primals[0],
                    primals[2],
                    primals[3],
                    primals[1].ridge,
                )
                return result, self._ad_result_tangent(
                    result, p_dot, initial_ad_point, max_steps_is_success
                )

            return solve_multi_start_with_ad_p(
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
        def solve_with_ad_p(
            x,
            lm_state,
            args,
            p,
            user_state,
            max_steps,
            atol,
            gtol,
            xtol,
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

        @solve_with_ad_p.defjvp
        def solve_with_ad_p_jvp(primals, tangents):
            p_dot = tangents[3]
            result = solve_with_ad_p(*primals)
            initial_ad_point = (primals[0], primals[2], primals[3], primals[1].ridge)
            return result, self._ad_result_tangent(
                result, p_dot, initial_ad_point, max_steps_is_success
            )

        return solve_with_ad_p(
            x0,
            lm_state,
            args,
            p,
            user_state,
            max_steps,
            atol,
            gtol,
            xtol,
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
        return _solve_python_impl(
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
            return _multi_start_python_impl(
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

    def _action_or_default(self, action):
        if action is None:
            return LMSolveAction()
        return action

    def _apply_action(self, action, x, lm_state, args, user_state):
        action = self._action_or_default(action)
        # The step's diagnostics and every cache describe the pre-action
        # problem, so they are stale iff the action actually changed the
        # values -- a traced comparison, so a jit-style callback that returns
        # the field every step with unchanged values changes nothing. A ridge
        # change leaves the Jacobian cache VALID (J does not depend on ridge)
        # but invalidates the ridge-keyed normal/QR caches and suppresses the
        # convergence check (the diagnostics were computed at the old ridge).
        xargs_changed = jnp.asarray(False)
        ridge_changed = jnp.asarray(False)
        if action.x is not None:
            xargs_changed = xargs_changed | _tree_changed(action.x, x)
            x = action.x
        if action.lm_state is not None:
            previous_hyper = lm_state.hyper
            # Captured BEFORE replacing: ridge lives inside action.lm_state,
            # not as a top-level action field.
            previous_ridge = lm_state.ridge
            lm_state = action.lm_state
            if lm_state.ridge is None:
                raise ValueError(
                    "the callback action returned an lm_state without ridge; "
                    "use dataclasses.replace(ctx.lm_state, ...) to preserve it"
                )
            if self.cache_jacobian and lm_state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True but the callback action returned an "
                    "lm_state without the Jacobian cache; use "
                    "dataclasses.replace(ctx.lm_state, ...) to preserve the "
                    "cache fields"
                )
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
            # Recast a callback-provided ridge to the carried scalar's dtype:
            # a weak-typed Python float in the action must not change the
            # while_loop carry aval.
            new_ridge = jnp.asarray(lm_state.ridge, dtype=previous_ridge.dtype)
            ridge_changed = ridge_changed | ~jnp.array_equal(
                new_ridge, previous_ridge, equal_nan=True
            )
            lm_state = dataclasses.replace(lm_state, ridge=new_ridge)
        if action.args is not None:
            xargs_changed = xargs_changed | _tree_changed(action.args, args)
            args = action.args
        if action.user_state is not None:
            user_state = action.user_state
        problem_changed = xargs_changed | ridge_changed
        if self.cache_jacobian and (action.x is not None or action.args is not None):
            lm_state = dataclasses.replace(
                lm_state, jacobian_valid=lm_state.jacobian_valid & ~xargs_changed
            )
        touched = (
            action.x is not None
            or action.args is not None
            or action.lm_state is not None
        )
        if touched and lm_state.G_valid is not None:
            lm_state = dataclasses.replace(
                lm_state, G_valid=lm_state.G_valid & ~problem_changed
            )
        if touched and lm_state.qr_valid is not None:
            lm_state = dataclasses.replace(
                lm_state, qr_valid=lm_state.qr_valid & ~problem_changed
            )
        return action, x, lm_state, args, user_state, problem_changed

    def _check_residual_args(self, args, p):
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
        # gtol/xtol mean "done with the current fixed-ridge problem"; atol is
        # a CONJUNCTIVE filter on the TRUE residual, never sufficient alone.
        gtol_met = (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (xtol > 0) & info.accepted & (info.step_norm < xtol)
        residual_ok = (atol <= 0) | (jnp.sqrt(info.resid_loss) <= atol)
        return (gtol_met | xtol_met) & residual_ok

    def _cast_state(self, lm_state, dtype):
        updates = dict(
            damping=jnp.asarray(lm_state.damping, dtype=dtype),
            ridge=jnp.asarray(lm_state.ridge, dtype=dtype),
            hyper=_cast_hyper(lm_state.hyper, dtype),
        )
        if lm_state.G_ridge is not None:
            updates["G_ridge"] = jnp.asarray(lm_state.G_ridge, dtype=dtype)
        if lm_state.qr_ridge is not None:
            updates["qr_ridge"] = jnp.asarray(lm_state.qr_ridge, dtype=dtype)
        return dataclasses.replace(lm_state, **updates)

    def _cold_state(self, lm_state):
        # Drawn multi-start lanes must not reuse caches built at another
        # (x, args); damping, hyper, and ridge stay inherited from the
        # caller's initial state (never reset to constructor defaults --
        # parallel lane 0 uses the cold state while sequential attempt 0 uses
        # the original, and ridge=None cannot be resolved without the
        # residual dtype anyway).
        updates = {}
        if lm_state.jacobian_valid is not None:
            updates["jacobian_valid"] = jnp.zeros_like(lm_state.jacobian_valid)
        if lm_state.G is not None:
            updates["G"] = jnp.zeros_like(lm_state.G)
            updates["G_valid"] = jnp.zeros_like(lm_state.G_valid)
        if lm_state.qr_R is not None:
            updates["qr_R"] = jnp.zeros_like(lm_state.qr_R)
            updates["qr_valid"] = jnp.zeros_like(lm_state.qr_valid)
        if not updates:
            return lm_state
        return dataclasses.replace(lm_state, **updates)

    def _ranking_objective(self, result, p, callback):
        # Multi-start selection ranks by the ridge objective at each lane's
        # OWN final ridge (comparable across lanes when they share a
        # continuation schedule). Without a callback info.loss already reports
        # it at the retained iterate; a callback can replace x/args/ridge
        # after the last update, so recompute. Nonfinite masks to +inf.
        if callback is None:
            loss = result.info.loss
        else:
            residual = self._residual_and_aux(result.x, result.args, p)[0]
            theta, _ = ravel_pytree(result.x)
            ridge = jnp.asarray(result.lm_state.ridge, dtype=residual.dtype)
            loss = jnp.sum(residual**2) + ridge * jnp.asarray(
                self._quadratic(theta), dtype=residual.dtype
            )
        return jnp.where(
            jnp.isfinite(loss), loss, jnp.asarray(jnp.inf, dtype=loss.dtype)
        )

    def _ad_result_tangent(self, result, p_dot, initial_ad_point, max_steps_is_success):
        # A successful tangent relinearizes at the returned solution with the
        # winner's own final ridge (stop-gradient: lambda is inert
        # conditioning data). A failed tangent uses the differentiation-inert
        # original initial point and the pre-loop INITIAL ridge -- a failed
        # lane's callback may have left an invalid ridge behind. Everything
        # except x, p, and aux is bookkeeping with zero tangents.
        initial_x, initial_args, initial_p, initial_ridge = jax.tree.map(
            jax.lax.stop_gradient, initial_ad_point
        )
        ad_success = result.status == LMStatus.CONVERGED
        if max_steps_is_success:
            ad_success = ad_success | (result.status == LMStatus.MAX_STEPS)
        ad_x = _where_tree(ad_success, result.x, initial_x)
        ad_args = _where_tree(ad_success, result.args, initial_args)
        ad_p = _where_tree(ad_success, result.p, initial_p)
        ad_p_dot = _mask_tangent_tree(ad_success, p_dot)
        final_ridge = jax.lax.stop_gradient(result.lm_state.ridge)
        ad_ridge = jnp.where(
            ad_success, final_ridge, jnp.asarray(initial_ridge, final_ridge.dtype)
        )
        x_dot = self._ad_x_tangent_from_p(ad_x, ad_args, ad_p, ad_p_dot, ad_ridge)
        zero_result = jax.tree.map(_zero_tangent_leaf, result)
        x_dot = _where_tree(ad_success, x_dot, zero_result.x)
        aux_dot = zero_result.aux
        if self.has_aux and ad_p is not None:
            # aux depends on p directly and through the solution x*(p).
            def aux_at_solution(x_value, p_value):
                return self.residual_fn(x_value, ad_args, p_value)[1]

            aux_dot = jax.jvp(aux_at_solution, (ad_x, ad_p), (x_dot, ad_p_dot))[1]
            aux_dot = _where_tree(ad_success, aux_dot, zero_result.aux)
        return dataclasses.replace(zero_result, x=x_dot, p=p_dot, aux=aux_dot)

    def _resolved_ad_solver(self):
        if self.ad_solver != "auto":
            return self.ad_solver
        return "normal_cg" if self.linear_solver == "lsmr" else "cholesky"

    def _ad_x_tangent_from_p(self, x, args, p, p_dot, ridge):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        resolved = self._resolved_ad_solver()
        if (
            resolved == "normal_cg"
            and self.ad_solver_tol == 0
            and self.ad_solver_atol == 0
            and self.ad_solver_maxiter is None
        ):
            raise ValueError(
                "ad_solver_maxiter must be set when both ad_solver tolerances are zero"
            )
        if resolved == "cholesky":
            return self._ad_tangent_cholesky(x, args, p, p_dot, ridge)
        return self._ad_tangent_normal_cg(x, args, p, p_dot, ridge)

    def _ad_linearization(self, x, args, p, p_dot):
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        return theta, unravel, residual, theta_jvp, residual_p_dot

    def _ad_tangent_cholesky(self, x, args, p, p_dot, ridge):
        # (J'J + ridge M0) x_dot = -J'(dr/dp) p_dot, no damping: the matrix is
        # PD under ker J ∩ ker L = {0} because ridge > 0 by contract. Wide
        # pipeline under linear_solve_dtype, same recipe as the forward path.
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        Jt = self._assemble_jt(theta_jvp, theta, residual)
        ad_dtype = (
            residual.dtype
            if self.linear_solve_dtype is None
            else self.linear_solve_dtype
        )
        transposed_jacobian = Jt.astype(ad_dtype)
        ridge_wide = jnp.asarray(ridge, dtype=ad_dtype)
        normal_matrix = self._add_scaled(
            transposed_jacobian @ transposed_jacobian.T, ridge_wide
        )
        factor = jsp_linalg.cho_factor(normal_matrix)
        rhs = -(transposed_jacobian @ residual_p_dot.astype(ad_dtype))
        theta_dot = jsp_linalg.cho_solve(factor, rhs)
        return unravel(theta_dot.astype(residual.dtype))

    def _ad_tangent_normal_cg(self, x, args, p, p_dot, ridge):
        # Matrix-free CG on the same PD operator; matvec =
        # J'(J v) + ridge L'(L v).
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        theta_transpose = jax.linear_transpose(theta_jvp, theta)

        def JT(cotangent):
            return theta_transpose(cotangent)[0]

        ridge_typed = jnp.asarray(ridge, dtype=residual.dtype)

        def normal_matvec(u):
            penalty_apply = jnp.asarray(
                self._sqrt_transpose_apply(self._sqrt_apply(u)),
                dtype=residual.dtype,
            )
            return JT(theta_jvp(u)) + ridge_typed * penalty_apply

        cg_tol = self._ad_cg_tol(residual.dtype)
        cg_atol = jnp.asarray(self.ad_solver_atol, dtype=residual.dtype)

        def solve(matvec, rhs_value):
            solution, _ = jsp_sparse_linalg.cg(
                matvec,
                rhs_value,
                tol=cg_tol,
                atol=cg_atol,
                maxiter=self.ad_solver_maxiter,
                M=self.ad_solver_preconditioner,
            )
            return solution

        rhs = -JT(residual_p_dot)
        theta_dot = jax.lax.custom_linear_solve(
            normal_matvec,
            rhs,
            solve,
            symmetric=True,
        )
        return unravel(theta_dot)

    def _ad_cg_tol(self, dtype):
        if self.ad_solver_tol is not None:
            return jnp.asarray(self.ad_solver_tol, dtype=dtype)
        default_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return jnp.asarray(default_tol, dtype=dtype)
