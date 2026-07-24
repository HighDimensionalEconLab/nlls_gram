"""Ridge-regularized Levenberg-Marquardt for underdetermined interpolation.

RidgeLevenbergMarquardt minimizes the ridge objective
``||r(x, args, p)||^2 + ridge * ||x_m||_W^2`` for a user-supplied residual
and a positive-definite :class:`~nlls_gram.Metric` ``W`` on the metric block
``x_m`` (the free block ``x_f`` stays unpenalized), with the ridge weight
``lambda`` carried as traced state that a ``solve`` callback may anneal
toward zero (:func:`ridge_continuation`). Selection of the minimum-seminorm
interpolant lives in the OBJECTIVE -- classical nonlinear Tikhonov
regularization (Engl-Kunisch-Neubauer 1989; Engl-Hanke-Neubauer 1996 Ch. 10;
the seminorm formulation goes back to Elden 1982) -- rather than in an
algorithmic implicit bias, and the per-accepted-step annealed limit is the
iteratively regularized Gauss-Newton method (Bakushinskii 1992;
Blaschke-Neubauer-Scherzer 1997; Kaltenbacher-Neubauer-Scherzer 2008).
Numerically the solver runs entirely in the whitened variable
``y = F_bar x`` (``W = F'F``, ``F_bar = blockdiag(F, I)``): stock Euclidean
LM on the augmented residual ``[r; sqrt(ridge) y_m]`` (Marquardt 1963; More
1978) with geodesic acceleration (Transtrum-Sethna 2012); the ``qr`` path
uses corrected semi-normal equations (Bjorck 1987; Bjorck 1996 Sec. 6.6.5).
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

from nlls_gram.metrics import Metric, MetricContext
from nlls_gram.solver_config import CG, QR, Cholesky
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
    canonicalize_residual,
)

__all__ = [
    "CholeskyCache",
    "QRCache",
    "RidgeLevenbergMarquardt",
    "RidgeLMInfo",
    "RidgeLMState",
    "ridge_continuation",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CholeskyCache:
    """Per-``(x, ridge)`` cache carried by the ``Cholesky`` forward path.

    ``G`` is the assembled whitened normal matrix ``J~'J~ + ridge E``
    (``J~ = J F_bar^{-1}``, ``E`` the metric-block diagonal pad;
    pre-damping), so a rejected step re-factors without re-assembling;
    ``valid`` marks it current for the state's ``x``; ``ridge`` is the weight
    it was assembled with -- a callback ridge change invalidates through this
    key.
    """

    G: jax.Array
    valid: jax.Array
    ridge: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QRCache:
    """Per-``(x, ridge)`` cache carried by the ``QR`` forward path.

    ``R`` is the R factor of the augmented whitened stack
    ``[J~, sqrt(ridge) [I 0] | b~]`` -- the first ``p`` columns are the R
    factor of ``[J~; sqrt(ridge) [I 0]]`` and the last column carries the
    ``Q``-transformed residual, so the velocity solve is backward stable with
    no normal equations. ``valid``/``ridge`` have the :class:`CholeskyCache`
    semantics.
    """

    R: jax.Array
    valid: jax.Array
    ridge: jax.Array


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
        solver_cache: the configured forward solver's own reject-step cache
            -- a :class:`CholeskyCache` or :class:`QRCache` whose pytree
            structure is fixed by the static ``linear_solver`` config, or
            ``None`` when the path carries no cache (``CG``,
            ``cache_jacobian=False``).
        metric_state: reserved for the future ``metric_factory`` (adaptive
            metrics): traced data a factory's ``prepare``/``build`` pair
            would turn into a :class:`~nlls_gram.Metric` per step. Always
            ``None`` today.
        metric_valid: reserved alongside ``metric_state`` with the
            ``jacobian_valid`` reject-reuse semantics; a state change is a
            problem change (convergence suppressed, ridge-keyed caches
            invalidated). Always ``None`` today.
    """

    damping: jax.Array
    ridge: jax.Array
    resid: jax.Array | None = None
    Jt: jax.Array | None = None
    jacobian_valid: jax.Array | None = None
    aux: Any = None
    hyper: LMHyperparams | None = None
    solver_cache: Any = None
    metric_state: Any = None
    metric_valid: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RidgeLMInfo:
    """Per-step diagnostics returned by ``update`` (and for each ``solve`` step).

    ``loss`` is the RIDGE OBJECTIVE ``||r||^2 + ridge * ||x_m||_W^2``
    actually being minimized -- it includes the penalty term. Code that means
    equation error must read ``resid_loss`` (``||r||^2`` alone). The
    loss/damping fields report the accept/reject outcome of the step, while
    ``grad_norm``/``step_norm``/``aux`` are evaluated at the PRE-step ``x``.

    The solver runs in the whitened variable ``y = F_bar x``, so
    ``grad_norm``, ``step_norm``, and ``penalty_grad_norm`` are Euclidean in
    ``y`` -- equivalently, steps are measured in the W-norm and gradients in
    the dual W^{-1}-norm. Objective values
    (``loss``/``resid_loss``/``penalty_value``) are unaffected by the change
    of variables: whitening is a pure linear bijection of the same objective.

    Attributes:
        loss: ``min(loss_old, loss_candidate)`` ridge objective at the
            retained iterate.
        loss_old: ridge objective at the pre-step ``x``.
        loss_candidate: ridge objective at the trial point.
        resid_loss: ``||r||^2`` at the retained iterate.
        penalty_value: ``||x_m||_W^2 = ||y_m||^2`` at the retained iterate.
        ridge: ``()`` the ridge weight ``lambda`` used this step.
        accepted: ``()`` bool, whether the trial step was accepted.
        damping: ``()`` post-update damping ``mu``.
        damping_factor: ``()`` multiplicative damping update applied this step.
        used_geodesic: ``()`` bool, whether the geodesic-acceleration
            correction entered the accepted step.
        acceleration_ratio: ``()`` geodesic acceleration-to-velocity whitened
            norm ratio.
        grad_norm: ``()`` ``||F_bar^{-T} J'r + ridge [y_m; 0]||`` at the
            pre-step ``x`` -- the whitened ridge stationarity residual (the
            dual W^{-1}-norm of the half-gradient), NOT ``||J'r||``.
        penalty_grad_norm: ``()`` ``||[y_m; 0]|| = sqrt(penalty_value)`` at
            the pre-step ``x`` -- the whitened penalty-gradient scale,
            reported so ``gtol`` can be CALIBRATED instead of guessed: at a
            ridge minimizer the gradient is the cancellation of the residual
            pullback against ``ridge * [y_m; 0]``, so demanding ``grad_norm
            < c * ridge * penalty_grad_norm`` resolves the null-space
            (selection) coordinates to ~``c`` relative accuracy. The recipe
            is ``gtol ~ 1e-3 * ridge * sqrt(q(x*))`` with ``q`` the
            solution's squared seminorm, usually known to an order of
            magnitude before any pilot run.
        step_norm: ``()`` whitened ``||delta_y||`` of the candidate step --
            the W-norm of the x-space step -- reported even when the step is
            rejected (``xtol`` therefore bounds the whitened step).
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
    penalty_grad_norm: jax.Array
    step_norm: jax.Array
    aux: Any = None


def ridge_continuation(
    *, decrease=0.1, ridge_floor, grad_rtol=1e-2, stall_rtol=0.0, dtype=None
):
    """Build ``(callback, user_state0)`` implementing ridge continuation.

    The callback multiplies ``lm_state.ridge`` by ``decrease`` whenever the
    current level has yielded what it can, never below ``ridge_floor``:

    - **stationary**: ``info.grad_norm`` fell below ``grad_rtol`` relative to
      its reference value at the current ridge level (the level's first
      observed gradient), or
    - **stalled** (opt-in, ``stall_rtol > 0``): an ACCEPTED step improved
      the gradient by less than a factor ``stall_rtol``. The escape hatch
      for a frozen anneal: the per-level references COMPOUND -- with closely
      spaced levels each new reference is already small, the effective
      demand approaches ``grad_rtol ** levels`` times the initial gradient,
      and the anneal can freeze below the problem's noise floor while steps
      keep being accepted with negligible progress. Enable with
      ``stall_rtol ~ 0.99`` when that happens (advancing on stagnation is
      still more conservative than the cited iteratively regularized
      Gauss-Newton method, which anneals every step). Off by default
      because it cannot distinguish a converged level from an accepted
      micro-step under temporarily high damping early in a hard solve --
      where a false advance collapses the schedule prematurely; widening
      ``decrease`` (e.g. ``0.01``) is the alternative fix for a frozen
      anneal, since larger jumps keep the per-level references generous.

    ``ridge_floor`` is REQUIRED and strictly positive (``ridge = 0`` is out
    of the solver's contract). The callback returns an
    :class:`~nlls_gram.LMSolveAction` replacing ``lm_state`` AND
    ``user_state`` (the per-level reference and previous gradient live in
    ``user_state`` as fixed-shape scalars, since a ``lax.while_loop`` carry
    cannot grow from ``None`` mid-loop -- hence the factory shape)::

        cb, us0 = ridge_continuation(ridge_floor=1e-10)
        result = solver.solve(x0, callback=cb, user_state=us0,
                              gtol=1e-8, atol=1e-8)

    The solved-out continuation path converges to the minimum-seminorm
    solution by nonlinear Tikhonov theory (Engl-Kunisch-Neubauer 1989;
    Engl-Hanke-Neubauer 1996), while annealing per accepted stationarity event
    rather than per fully solved level is the iteratively regularized
    Gauss-Newton method (Bakushinskii 1992; Blaschke-Neubauer-Scherzer 1997;
    Kaltenbacher-Neubauer-Scherzer 2008), whose theory wants exactly this kind
    of monotone, boundedly geometric schedule. Pair the schedule with the
    conjunctive stopping rule: choose ``atol`` BETWEEN the ridge-floor
    residual and the last intermediate level's residual (they differ by
    roughly ``1 / decrease``), so the solve can only stop at the floor even
    when ``gtol`` must sit above a variant-dependent stationarity noise
    floor -- intermediate levels are stationary too, and ``atol`` is what
    rules them out. ``dtype`` types the ``user_state0`` scalars (default:
    the JAX
    default float; pass the problem dtype explicitly for a float32 program
    under enabled x64).
    """

    if not 0 < decrease < 1:
        raise ValueError("decrease must lie strictly between 0 and 1")
    concrete_floor = not isinstance(ridge_floor, (jax.Array, jax.core.Tracer))
    if concrete_floor and float(ridge_floor) <= 0:
        raise ValueError(
            "ridge_floor must be strictly positive (ridge = 0 is "
            "unsupported by RidgeLevenbergMarquardt)"
        )
    if grad_rtol <= 0:
        raise ValueError("grad_rtol must be positive")
    if not 0 <= stall_rtol < 1:
        raise ValueError("stall_rtol must lie in [0, 1)")
    if dtype is None:
        dtype = jnp.result_type(float)
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    user_state0 = {"reference": infinity, "previous": infinity}

    def callback(ctx):
        ridge = ctx.lm_state.ridge
        grad_norm = jnp.asarray(ctx.info.grad_norm, dtype=ridge.dtype)
        reference = jnp.asarray(ctx.user_state["reference"], dtype=ridge.dtype)
        previous = jnp.asarray(ctx.user_state["previous"], dtype=ridge.dtype)
        # +inf marks "no observation at this level yet": the first step after
        # a ridge decrease (or the initial step) sets the reference and can
        # never read as stalled.
        reference = jnp.where(jnp.isfinite(reference), reference, grad_norm)
        floor = jnp.asarray(ridge_floor, dtype=ridge.dtype)
        stationary = grad_norm <= jnp.asarray(grad_rtol, ridge.dtype) * reference
        if stall_rtol > 0:
            # ACCEPTED steps only: a rejected step leaves x (and so the
            # gradient) unchanged -- that is the trust region adapting, not
            # the level converging -- while an accepted step that improved
            # the gradient by less than the stall factor means the level has
            # yielded what it can.
            stalled = ctx.info.accepted & (
                grad_norm >= jnp.asarray(stall_rtol, ridge.dtype) * previous
            )
        else:
            stalled = jnp.asarray(False)
        new_ridge = jnp.where(
            stationary | stalled,
            jnp.maximum(ridge * jnp.asarray(decrease, ridge.dtype), floor),
            ridge,
        )
        # Reset the trackers when the level actually changes; at the floor the
        # ridge is unchanged, so convergence is not suppressed and gtol/atol
        # can fire.
        advanced = new_ridge < ridge
        fresh_level = jnp.asarray(jnp.inf, ridge.dtype)
        new_reference = jnp.where(advanced, fresh_level, reference)
        new_previous = jnp.where(advanced, fresh_level, grad_norm)
        state_dtype = jnp.asarray(ctx.user_state["reference"]).dtype
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, ridge=new_ridge),
            user_state={
                "reference": new_reference.astype(state_dtype),
                "previous": new_previous.astype(state_dtype),
            },
        )

    return callback, user_state0


class RidgeLevenbergMarquardt:
    """Levenberg-Marquardt for the ridge objective
    ``F(x) = ||r(x, args, p)||^2 + ridge * q(x)`` with
    ``q(x) = ||x_m||_W^2`` over a JAX pytree ``x``, built for underdetermined
    interpolation problems where the minimum-seminorm root
    ``argmin q s.t. r = 0`` is the target.

    The flattened parameter vector splits as ``x = [x_m; x_f]``: the METRIC
    BLOCK ``x_m`` (the leading ``metric.size`` coordinates, penalized by the
    positive-definite :class:`~nlls_gram.Metric` ``W``) and the FREE BLOCK
    ``x_f`` (the remaining ``n_f = len(x) - metric.size >= 0`` coordinates,
    unpenalized -- the full-space penalty is PSD by design). The metric is
    supplied through factor callbacks for ``W = F'F``
    (:class:`~nlls_gram.IdentityMetric` is plain ridge,
    :class:`~nlls_gram.RepeatedFactorMetric` the kernel workhorse), and the
    solver runs entirely in the whitened variable ``y = F_bar x`` with
    ``F_bar = blockdiag(F, I_{n_f})`` -- an exact, constant linear change of
    variables, never a matrix: only the factor ops are ever applied.

    The selection lives in the objective: under free-block identification at
    the solution, each fixed-ridge problem has an isolated minimizer
    ``x_ridge`` with ``x_ridge -> x* = argmin q s.t. r = 0`` at ``O(ridge)``
    error as ``ridge -> 0`` (nonlinear Tikhonov theory:
    Engl-Kunisch-Neubauer 1989; Engl-Hanke-Neubauer 1996 Ch. 10; the seminorm
    formulation is Elden 1982). Plain Gauss-Newton on the underdetermined
    system converges to *some* root without selecting the minimal-norm one
    (Campbell-Kunkel-Bobinyec 2012; Pes-Rodriguez 2022), which is why the
    package's alternative is metric-damped
    :class:`~nlls_gram.LevenbergMarquardt`; this solver instead makes every
    inner problem a well-posed NLLS. Annealing ``ridge`` per stationarity
    event (:func:`ridge_continuation`) is the iteratively regularized
    Gauss-Newton method (Bakushinskii 1992; Kaltenbacher-Neubauer-Scherzer
    2008). For kernel metrics each inner step is a kernel ridge regression
    of the relinearized equations (Chen-Hosseini-Owhadi-Stuart 2021).

    Everywhere in the implementation the equivalent augmented-residual view
    is used, posed in ``y``: standard EUCLIDEAN LM on
    ``R(y) = [r(x); sqrt(ridge) y_m]`` with ``J~ = J F_bar^{-1}``,
    ``A = [J~; sqrt(ridge) [I 0]]``, and step
    ``(J~'J~ + ridge E + mu I) delta_y = -(J~'r + ridge [y_m; 0])`` where
    ``E = blockdiag(I_{n_m}, 0)``; the x-space step is
    ``delta_x = F_bar^{-1} delta_y`` and the iterate is stored in ``x``. The
    penalty rows are affine (constant, even) in ``y``, so geodesic
    acceleration (Transtrum-Sethna 2012), accept/reject, and the implicit-AD
    rule reduce to the standard formulas with second-derivative contributions
    from the penalty identically zero; the trial penalty uses the linearity
    ``F_bar(x + delta_x) = y + delta_y``, so no second factor application is
    ever needed. ``damping`` (``mu``) is a plain trust-region parameter in
    the whitened geometry (``mu ||delta_y||^2 = mu ||delta_x_m||_W^2`` plus
    the Euclidean free block), fully decoupled from ``ridge`` (``lambda``):
    ``mu`` moves every step by accept/reject, ``lambda`` only when a callback
    replaces ``lm_state.ridge``. The whitened normal matrix
    ``J~'J~ + ridge E`` carries a clean spectral floor at ``ridge`` on the
    metric block regardless of ``W``'s conditioning -- the reason the default
    ``Cholesky()`` path stays accurate at deep ridge.

    ``ridge`` is STRICTLY POSITIVE by contract -- the constructor validates
    it, callbacks must keep it positive, and continuation floors are positive
    -- so ``J~'J~ + ridge E`` is positive definite under the identification
    condition at every reachable state. ``ridge=None`` resolves at ``init``
    to ``sqrt(finfo(dtype).eps)`` of the residual dtype (float64 ~ 1.5e-8,
    float32 ~ 3.5e-4). The metric's factorization never involves the ridge
    weight, so continuation composes unchanged.

    ``linear_solver`` picks the algebra for
    ``(J~'J~ + ridge E + damping I) delta_y = -g`` through a typed config
    (:class:`~nlls_gram.Cholesky` /
    :class:`~nlls_gram.QR` / :class:`~nlls_gram.CG` -- each method's
    knobs live on its own config); every path shares its factorization or
    inner-solve setup between the velocity and geodesic-acceleration solves:

    - ``Cholesky()`` (the default): dense normal equations. ``J'`` is
      materialized per ``jacobian_mode``, ``J~'`` follows by one batched
      ``factor_solve_transpose``, and the assembled
      ``G = J~'J~ + ridge E`` (``ridge`` added on the metric-block diagonal
      only) is cached across rejected steps (only the damping shift
      re-factors).
    - ``QR()``: MINPACK-structured damping-row QR (More 1978), stable at
      small ``ridge``/``damping`` where forming ``G`` squares the condition
      number. One QR of the augmented stack ``[J~, sqrt(ridge) [I 0] | b~]``
      with ``b~ = [r; sqrt(ridge) y_m]`` is cached per ``(x, ridge)`` (the
      extra column carries the ``Q``-transformed residual); each update
      re-factors ``[R; sqrt(damping) I]`` with its ``Q`` retained and solves
      the velocity as a backward-stable least-squares problem at ``cond(A)``,
      never ``cond(A)^2``. The geodesic-acceleration RHS reuses the damped
      factor through corrected semi-normal equations with one fixed
      iterative-refinement pass (Bjorck 1987; Bjorck 1996 Sec. 6.6.5) --
      the second-order correction tolerates the squared conditioning.
    - ``CG(preconditioner, ...)``: matrix-free preconditioned CG on the
      damped normal operator ``J~'J~ + ridge E + damping I`` itself -- the
      same SPD system the ``Cholesky()`` path factors, with products through
      ``jvp``/``vjp`` and the metric's factor callbacks instead of an
      assembled ``G``. The config requires a typed
      :class:`~nlls_gram.Preconditioner`
      (:class:`~nlls_gram.IdentityPreconditioner` opts out): its
      ``apply(v, damping, ctx)`` -- an SPD approximation of the damped
      inverse, handed the same :class:`~nlls_gram.MetricContext` as the
      metric ops -- sits in CG's ``M`` slot with the live damping. The
      operator carries the ``ridge`` spectral floor on the metric block, so
      the preconditioner only has to capture ``J~'J~``'s structure.

    Stopping (``solve``): a ridge solve has two phases -- the residual drops
    to its floor fast, then the iterate slides along the interpolation set
    resolving the null-space (selection) component while ``||r||`` stays
    essentially constant. A pure-residual test is blind to phase 2, so
    ``gtol`` bounds the whitened ridge stationarity ``info.grad_norm =
    ||J~'r + ridge [y_m; 0]||`` (the dual W^{-1}-norm of the half-gradient)
    and ``xtol`` the accepted whitened step norm ``||delta_y||`` (the W-norm
    of the step) -- they mean "done with the current fixed-ridge problem" --
    while ``atol`` is a CONJUNCTIVE filter on the TRUE residual: convergence
    requires (``gtol`` or ``xtol`` fired) AND (``atol == 0`` or
    ``sqrt(resid_loss) <= atol``). atol alone never stops the solve -- a
    pure-residual test would stop at step 0 from any interpolating start
    before the seminorm is minimized -- so ``solve`` rejects ``atol > 0``
    without a positive ``gtol`` or ``xtol``. Calibrating ``gtol`` is clean in
    the whitened geometry: ``info.penalty_grad_norm = sqrt(q(x))``, so
    ``gtol ~ 1e-3 * ridge * sqrt(q(x*))`` resolves the selection to ~1e-3
    relative accuracy, with ``sqrt(q(x*))`` (the solution's seminorm) usually
    known to an order of magnitude before any pilot run. In a continuation
    run the callback keeps lowering ``ridge``, so intermediate stationarity
    with a large residual correctly does not stop the loop.

    ``solve(...).x`` has a custom implicit AD rule with respect to ``p``:
    Gauss-Newton implicit differentiation of the ridge stationarity, posed on
    the whitened variable -- ``(J~'J~ + ridge E) y_dot = -J~'(dr/dp) p_dot``
    then ``x_dot = F_bar^{-1} y_dot`` -- with ``ridge`` frozen
    (stop-gradient) at the returned state's value; the continuation
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
    not vanish relative to the ``ridge``-scaled penalty as ``ridge -> 0`` --
    so tangent components read off null-space directions retain an
    O(1)-in-``ridge`` Gauss-Newton bias proportional to the residual
    curvature. ``ad_solver=Cholesky()`` assembles and factors;
    :class:`~nlls_gram.CG` runs matrix-free CG on the same operator with its
    ``preconditioner`` hook (PD does not mean well-conditioned --
    unpreconditioned CG degrades as ``ridge`` shrinks). ``ad_solver=None``
    (the default) matches the forward path: ``Cholesky()`` for the dense
    forwards, ``CG()`` for a matrix-free ``CG`` forward. Failed statuses
    return exact zero tangents for ``result.x``/``result.aux`` and evaluate
    the masked tangent program at stop-gradient copies of the caller's
    original inputs and the INITIAL ridge (never a possibly-invalid
    callback-produced value).

    The init/update/solve protocol, callback contract
    (:class:`~nlls_gram.LMSolveContext` ->
    :class:`~nlls_gram.LMSolveAction`), ``multi_start``, ``save_steps``, and
    :class:`~nlls_gram.LMSolveResult` are shared with
    :class:`~nlls_gram.LevenbergMarquardt`; code written against that solver
    ports by changing the constructor (its damping metric -> this metric)
    and reading ``info.resid_loss`` where it means equation error, since
    ``info.loss`` here includes the penalty. Multi-start ranking uses the
    ridge objective at each lane's own final ridge -- comparable across lanes
    when they share a continuation schedule.
    """

    def __init__(
        self,
        residual_fn,
        *,
        metric,
        metric_factory=None,
        ridge=None,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        min_damping=None,
        max_damping=None,
        linear_solver=Cholesky(),  # noqa: B008 -- frozen, immutable default
        jacobian_mode="auto",
        ad_solver=None,
        has_aux=False,
        cache_jacobian=True,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    ):
        canonical_residual, residual_arity = canonicalize_residual(residual_fn)
        if metric_factory is not None:
            raise NotImplementedError(
                "metric_factory is reserved for a future release: its "
                "documented contract is a prepare/build pair producing a "
                "Metric from traced, differentiation-inert metric_state "
                "(with metric_valid reject-step reuse, and a state change "
                "treated as a problem change -- the same "
                "convergence-suppression/cache-invalidation machinery as a "
                "callback ridge change); pass a fixed metric"
            )
        if not isinstance(metric, Metric):
            raise TypeError("metric must be a Metric")
        if not isinstance(linear_solver, (Cholesky, QR, CG)):
            raise TypeError(
                "linear_solver must be a solver config -- Cholesky(), QR(), "
                f"or CG(preconditioner, ...); got {linear_solver!r}"
            )
        if ad_solver is not None and not isinstance(ad_solver, (Cholesky, CG)):
            raise TypeError(
                "ad_solver must be None (match the forward path), Cholesky(), "
                f"or CG(...); got {ad_solver!r}"
            )
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
        # jacobian_mode is consumed by the dense forward paths and the
        # cholesky AD method; a matrix-free normal_cg forward with a cg AD
        # rule never materializes J, so a forced mode is a construction
        # error there.
        if (
            jacobian_mode != "auto"
            and isinstance(linear_solver, CG)
            and not isinstance(ad_solver, Cholesky)
        ):
            raise ValueError(
                "jacobian_mode controls dense Jacobian assembly, but neither "
                "a matrix-free CG forward solver nor its cg-resolved "
                "ad_solver ever consumes it"
            )
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.metric = metric
        self.ridge = ridge
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.linear_solver = linear_solver
        self.jacobian_mode = jacobian_mode
        self.ad_solver = ad_solver
        # The configs' numeric fields fold into the traced LMHyperparams carry
        # at init exactly as the flat constructor args used to; the config
        # instances themselves sit in the value-based static key, so
        # recompile-on-change behavior is unchanged.
        if isinstance(linear_solver, CG):
            # The config requires a typed Preconditioner applying an SPD
            # approximation of (J~'J~ + ridge E + damping I)^{-1}, posed on
            # the whitened variable. IdentityPreconditioner() opts out
            # explicitly.
            self.normal_cg_preconditioner = linear_solver.preconditioner
            # tol=None resolves per residual dtype in hyperparams(), the
            # _ad_cg_tol convention.
            self.iterative_tol = linear_solver.tol
            self.iterative_atol = linear_solver.atol
            self.iterative_maxiter = linear_solver.maxiter
        else:
            self.normal_cg_preconditioner = None
            self.iterative_tol = 0.0
            self.iterative_atol = 0.0
            self.iterative_maxiter = 8
        if isinstance(ad_solver, CG):
            if ad_solver.preconditioner.requires_positive_damping:
                raise ValueError(
                    "this preconditioner divides by the live damping and "
                    "cannot serve in ad_solver (the AD system is undamped)"
                )
            self.ad_solver_tol = ad_solver.tol
            self.ad_solver_atol = ad_solver.atol
            self.ad_solver_maxiter = ad_solver.maxiter
            self.ad_solver_preconditioner = ad_solver.preconditioner
        else:
            self.ad_solver_tol = None
            self.ad_solver_atol = 0.0
            self.ad_solver_maxiter = None
            self.ad_solver_preconditioner = None
        self.has_aux = has_aux
        # Only the dense paths materialize J' (and the cholesky/qr caches ride
        # on the same reject-reuse lifecycle), so the flag is inert for the
        # matrix-free normal_cg forward.
        self.cache_jacobian = cache_jacobian and not isinstance(linear_solver, CG)
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        # Value-based identity: the jitted solve loop marks the solver itself
        # static, so equal-config solvers built around the same residual and
        # metric share the compiled loop across instances. Keyed on the
        # constructor arguments -- every derived attribute is a function of
        # them. Metrics hash by identity, so a rebuilt equal-config metric
        # keys a fresh compilation.
        self._static_key = tuple(
            _static_key_component(value)
            for value in (
                residual_fn,
                metric,
                ridge,
                init_damping,
                damping_decrease,
                damping_increase,
                min_damping,
                max_damping,
                linear_solver,
                jacobian_mode,
                ad_solver,
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
        iterative_tol = self.iterative_tol
        if iterative_tol is None:
            # CG's tol=None: the _ad_cg_tol dtype-default convention.
            resolved = jnp.result_type(float) if dtype is None else dtype
            iterative_tol = 1e-10 if jnp.finfo(resolved).bits > 32 else 1e-6
        return LMHyperparams(
            jnp.asarray(self.damping_decrease, dtype=dtype),
            jnp.asarray(self.damping_increase, dtype=dtype),
            _damping_floor(self.min_damping, dtype),
            None
            if self.max_damping is None
            else jnp.asarray(self.max_damping, dtype=dtype),
            jnp.asarray(self.geodesic_acceptance_ratio, dtype=dtype),
            jnp.asarray(iterative_tol, dtype=dtype),
            jnp.asarray(self.iterative_atol, dtype=dtype),
            None
            if self.iterative_maxiter is None
            else jnp.asarray(self.iterative_maxiter, dtype=jnp.int32),
        )

    def _resolved_solver(self):
        if isinstance(self.linear_solver, Cholesky):
            return "cholesky"
        return "qr" if isinstance(self.linear_solver, QR) else "normal_cg"

    def _block_sizes(self, theta_size):
        # The free-block size is inferred from the flattened iterate: the
        # metric covers the leading metric.size coordinates, the rest is free.
        n_f = theta_size - self.metric.size
        if n_f < 0:
            raise ValueError(
                f"the metric covers {self.metric.size} leading coordinates "
                f"but x flattens to only {theta_size}; the free block is "
                "len(x) - metric.size and must be nonnegative"
            )
        return self.metric.size, n_f

    # The solver-internal identity extension F_bar = blockdiag(F, I_{n_f}):
    # the metric's factor op on the metric block, passthrough on the free
    # block. Applied to vectors or leading-axis-batched matrices; F_bar
    # itself is never materialized.
    def _extended_solve(self, v, ctx):
        n_m = self.metric.size
        return jnp.concatenate(
            [self.metric.factor_solve(v[:n_m], ctx), v[n_m:]], axis=0
        )

    def _extended_solve_transpose(self, v, ctx):
        n_m = self.metric.size
        return jnp.concatenate(
            [self.metric.factor_solve_transpose(v[:n_m], ctx), v[n_m:]], axis=0
        )

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
        theta, _ = ravel_pytree(x0)
        n_m, _ = self._block_sizes(theta.size)
        dtype = residual.dtype
        min_damping = _damping_floor(self.min_damping, dtype)
        damping = jnp.maximum(jnp.asarray(self.init_damping, dtype=dtype), min_damping)
        ridge = self._resolve_ridge(dtype)
        if not self.cache_jacobian:
            return RidgeLMState(damping, ridge)
        p_dim = theta.size
        m = residual.size
        invalid = jnp.asarray(False, dtype=jnp.bool_)
        common = dict(
            resid=jnp.zeros(residual.shape, dtype=dtype),
            Jt=jnp.zeros((p_dim, m), dtype=dtype),
            jacobian_valid=invalid,
            aux=jax.tree.map(jnp.zeros_like, aux),
        )
        if self._resolved_solver() == "cholesky":
            cache = CholeskyCache(
                G=jnp.zeros((p_dim, p_dim), dtype=dtype),
                valid=invalid,
                ridge=jnp.zeros((), dtype=dtype),
            )
        else:
            r_rows = min(m + n_m, p_dim + 1)
            cache = QRCache(
                R=jnp.zeros((r_rows, p_dim + 1), dtype=dtype),
                valid=invalid,
                ridge=jnp.zeros((), dtype=dtype),
            )
        return RidgeLMState(damping, ridge, **common, solver_cache=cache)

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
        # grad_norm and penalty_grad_norm are +inf sentinels (computing them
        # would cost a Jacobian before the first step) and step_norm is zero;
        # none can satisfy gtol/xtol before any update has run.
        residual, aux = self._residual_and_aux(x, args, p)
        resid_loss = jnp.sum(residual**2)
        theta, _ = ravel_pytree(x)
        if lm_state.ridge is None:
            raise ValueError(
                "the lm_state has no ridge; create it with init(x, args, p=p)"
            )
        ridge = jnp.asarray(lm_state.ridge, dtype=residual.dtype)
        n_m = self._block_sizes(theta.shape[0])[0]
        ctx = MetricContext(x=theta, lm_state=lm_state, args=args, p=p)
        y_m = jnp.asarray(
            self.metric.factor_apply(theta[:n_m], ctx), dtype=residual.dtype
        )
        penalty_value = jnp.sum(y_m**2)
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

        # TRUE-residual linearization: matrix-free closures for normal_cg,
        # dense J' (reused from the cache after a rejected step) otherwise.
        resolved_solver = self._resolved_solver()
        if resolved_solver == "normal_cg":
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

        # The subproblem is posed on the whitened variable y = F_bar x, where
        # the penalty rows are the constant [I_{n_m} | 0]: the half-gradient
        # is g = F_bar^{-T} J'r + ridge [y_m; 0] (grad F = 2 g; the factor
        # cancels in the LM equations), and every gradient/step quantity
        # below (including the reported norms) is the whitened one. y_m is
        # reused for the pre-step penalty value ||y_m||^2. Metric callback
        # outputs are pinned to the residual dtype so a wider-typed metric
        # (e.g. float64 kernel data under a float32 residual) cannot promote
        # the gradient and break the loop-carry dtypes.
        n_m, n_f = self._block_sizes(theta.shape[0])
        ctx = MetricContext(x=theta, lm_state=lm_state, args=args, p=p)
        y_m = jnp.asarray(self.metric.factor_apply(theta[:n_m], ctx), dtype=resid.dtype)
        penalty_value_old = jnp.sum(y_m**2)
        penalty_gradient = jnp.concatenate([y_m, jnp.zeros(n_f, dtype=resid.dtype)])

        if resolved_solver == "normal_cg":
            transpose_fn = jax.linear_transpose(jvp_fn, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

            # Whitened operator J~ = J F_bar^{-1}: products route through the
            # metric's factor callbacks.
            def J_sub(u):
                return jvp_fn(jnp.asarray(self._extended_solve(u, ctx), resid.dtype))

            def JT_sub(w):
                return jnp.asarray(
                    self._extended_solve_transpose(JT(w), ctx), dtype=resid.dtype
                )

            grad = JT_sub(resid) + ridge * penalty_gradient
            inner_tol = jnp.asarray(hyper.iterative_tol, dtype=resid.dtype)
            inner_atol = jnp.asarray(hyper.iterative_atol, dtype=resid.dtype)
            sqrt_ridge = jnp.sqrt(ridge)
            m = resid.shape[0]

            # Augmented operator A = [J~; sqrt(ridge) [I 0]]: the penalty
            # rows are the constant metric-block identity on y.
            def A_matvec(u):
                return jnp.concatenate([J_sub(u), sqrt_ridge * u[:n_m]])

            def At_matvec(w):
                penalty_pullback = sqrt_ridge * jnp.concatenate(
                    [w[m:], jnp.zeros(n_f, dtype=resid.dtype)]
                )
                return JT_sub(w[:m]) + penalty_pullback

            # N = A'A + damping I, the preconditioner-free SPD operator that
            # custom_linear_solve differentiates through, posed on u (not z).
            def N_matvec(u):
                return At_matvec(A_matvec(u)) + damping * u

            # Preconditioned CG on N itself -- the same damped SPD system
            # the cholesky path factors, applied matrix-free. The typed
            # Preconditioner sits in CG's M slot: an SPD approximation of
            # N^{-1} at the live damping, with the solver context along.
            def apply_M(v):
                return self.normal_cg_preconditioner.apply(v, damping, ctx)

            def solve_N(_, c):
                solution, _ = jsp_sparse_linalg.cg(
                    N_matvec,
                    c,
                    tol=inner_tol,
                    atol=inner_atol,
                    maxiter=hyper.iterative_maxiter,
                    M=apply_M,
                )
                return solution

            def solve_step(rhs):
                return jax.lax.custom_linear_solve(
                    N_matvec,
                    -rhs,
                    solve=solve_N,
                    transpose_solve=solve_N,
                    symmetric=True,
                )

            def accel_rhs(f_vv):
                return JT_sub(f_vv)

        elif resolved_solver == "cholesky":
            # The assembled G = J~'J~ + ridge E is cached across rejected
            # steps -- only mu changed, so the reject pays the p^3/3 refactor
            # without the GEMM; the J~' materialization (a batched factor
            # solve through the metric) is likewise skipped on reject, and
            # the per-step gradient and acceleration RHS only ever pull
            # vectors back through factor_solve_transpose. A callback ridge
            # change invalidates through the cache's ridge key.
            grad = (
                jnp.asarray(
                    self._extended_solve_transpose(Jt @ resid, ctx),
                    dtype=resid.dtype,
                )
                + ridge * penalty_gradient
            )

            def assemble_normal(_):
                Jt_sub = jnp.asarray(
                    self._extended_solve_transpose(Jt, ctx), dtype=resid.dtype
                )
                diag = jnp.arange(n_m)
                return (Jt_sub @ Jt_sub.T).at[diag, diag].add(ridge)

            if self.cache_jacobian:
                cache = lm_state.solver_cache
                if not isinstance(cache, CholeskyCache):
                    raise ValueError(
                        "the lm_state has no normal-matrix cache for the "
                        "cholesky path; create the lm_state with "
                        "init(x, args, p=p)"
                    )
                normal_matrix = jax.lax.cond(
                    cache.valid & (cache.ridge == ridge),
                    lambda _: cache.G,
                    assemble_normal,
                    operand=None,
                )
            else:
                normal_matrix = assemble_normal(None)
            shifted = normal_matrix + damping * jnp.eye(
                theta.shape[0], dtype=resid.dtype
            )
            factor = jsp_linalg.cho_factor(shifted)

            def solve_step(rhs):
                return -jsp_linalg.cho_solve(factor, rhs)

            def accel_rhs(f_vv):
                return jnp.asarray(
                    self._extended_solve_transpose(Jt @ f_vv, ctx),
                    dtype=resid.dtype,
                )

        else:  # qr
            # One QR of the AUGMENTED whitened stack
            # [J~; sqrt(ridge) [I 0] | b~] with b~ = [r; sqrt(ridge) y_m],
            # cached per (x, ridge): its first p columns are the stack's R
            # factor and its last column carries Q'b, so the velocity can be
            # solved as a backward-stable least-squares problem with NO
            # normal equations anywhere -- More 1978's actual damping-row
            # structure. (An extra residual-norm row appears when
            # m + n_m > p; it is a constant in the least-squares objective
            # and harmless.) The semi-normal route (R'R delta = -g) squares
            # the stack's condition number, which loses the Gauss-Newton
            # step accuracy exactly in the tiny-ridge regime this path
            # exists for. J~ is materialized only when the cache refreshes.
            grad = (
                jnp.asarray(
                    self._extended_solve_transpose(Jt @ resid, ctx),
                    dtype=resid.dtype,
                )
                + ridge * penalty_gradient
            )

            def assemble_r(_):
                Jt_sub = jnp.asarray(
                    self._extended_solve_transpose(Jt, ctx), dtype=resid.dtype
                )
                stacked = jnp.concatenate(
                    [
                        Jt_sub.T,
                        jnp.sqrt(ridge)
                        * jnp.eye(n_m, theta.shape[0], dtype=resid.dtype),
                    ],
                    axis=0,
                )
                b_stacked = jnp.concatenate([resid, jnp.sqrt(ridge) * y_m])
                augmented = jnp.concatenate([stacked, b_stacked[:, None]], axis=1)
                return jnp.linalg.qr(augmented, mode="r")

            if self.cache_jacobian:
                cache = lm_state.solver_cache
                if not isinstance(cache, QRCache):
                    raise ValueError(
                        "the lm_state has no QR cache for the qr path; create "
                        "the lm_state with init(x, args, p=p)"
                    )
                qr_R = jax.lax.cond(
                    cache.valid & (cache.ridge == ridge),
                    lambda _: cache.R,
                    assemble_r,
                    operand=None,
                )
            else:
                qr_R = assemble_r(None)
            r_factor = qr_R[:, :-1]
            transformed_rhs = qr_R[:, -1]
            # Per-update damping-row refactor: [R; sqrt(damping) I] = Q2 R2
            # with R2'R2 = A'A + damping I. When m + k < p the cached R is
            # upper trapezoidal and these damping rows are what make the
            # final system full rank. Q2 is retained to transform the
            # velocity RHS stably.
            damped_stack = jnp.concatenate(
                [
                    r_factor,
                    jnp.sqrt(damping) * jnp.eye(theta.shape[0], dtype=resid.dtype),
                ],
                axis=0,
            )
            Q_mu, R_mu = jnp.linalg.qr(damped_stack, mode="reduced")

            def damped_normal_matvec(v):
                gauss_newton = jnp.asarray(
                    self._extended_solve_transpose(
                        Jt
                        @ (
                            Jt.T
                            @ jnp.asarray(self._extended_solve(v, ctx), resid.dtype)
                        ),
                        ctx,
                    ),
                    dtype=resid.dtype,
                )
                metric_shift = jnp.concatenate(
                    [v[:n_m], jnp.zeros(n_f, dtype=resid.dtype)]
                )
                return gauss_newton + ridge * metric_shift + damping * v

            def solve_step(rhs):
                # Corrected semi-normal equations (Bjorck 1987) for the
                # geodesic-acceleration RHS: triangular solves against R_mu,
                # then ONE fixed iterative-refinement pass through matvecs
                # (Bjorck 1996 Sec. 6.6.5). The second-order correction
                # tolerates the squared conditioning; accept/reject guards it.
                b = -rhs
                half = jsp_linalg.solve_triangular(R_mu.T, b, lower=True)
                delta = jsp_linalg.solve_triangular(R_mu, half, lower=False)
                correction_rhs = b - damped_normal_matvec(delta)
                half = jsp_linalg.solve_triangular(R_mu.T, correction_rhs, lower=True)
                delta = delta + jsp_linalg.solve_triangular(R_mu, half, lower=False)
                return delta

            def solve_velocity():
                # min ||[R; sqrt(damping) I] delta + [Q'b; 0]||^2 solved
                # through Q2: exact and backward stable at cond(A), never
                # cond(A)^2.
                rhs_aug = jnp.concatenate(
                    [
                        transformed_rhs,
                        jnp.zeros(theta.shape[0], dtype=resid.dtype),
                    ]
                )
                return -jsp_linalg.solve_triangular(R_mu, Q_mu.T @ rhs_aug, lower=False)

            def accel_rhs(f_vv):
                return jnp.asarray(
                    self._extended_solve_transpose(Jt @ f_vv, ctx),
                    dtype=resid.dtype,
                )

        # First-order step (velocity) and its ridge objective. The solves
        # produce the whitened step delta_y: the x-space step maps back
        # through the factor solve, and the trial penalty uses the linearity
        # of the change of variables -- F_bar(theta + step) = y + delta_y, so
        # no second factor application is ever needed.
        if resolved_solver == "qr":
            velocity_sub = solve_velocity()
        else:
            velocity_sub = solve_step(grad)
        velocity = jnp.asarray(self._extended_solve(velocity_sub, ctx), resid.dtype)

        def trial_penalty(step_sub):
            return jnp.sum((y_m + step_sub[:n_m]) ** 2)

        resid_velocity = residual_value(theta + velocity)
        resid_loss_old = jnp.sum(resid**2)
        loss_old = resid_loss_old + ridge * penalty_value_old
        resid_loss_velocity = jnp.sum(resid_velocity**2)
        penalty_velocity = trial_penalty(velocity_sub)
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
            acceleration_sub = solve_step(accel_rhs(f_vv))
            acceleration = jnp.asarray(
                self._extended_solve(acceleration_sub, ctx), dtype=resid.dtype
            )
            accelerated_step = velocity + 0.5 * acceleration
            accelerated_step_sub = velocity_sub + 0.5 * acceleration_sub
            # The ratio criterion lives in the damping geometry's norm -- the
            # whitened (W-norm) one, matching the metric LM's metric_norm.
            acceleration_ratio = (
                2.0
                * jnp.linalg.norm(acceleration_sub)
                / (jnp.linalg.norm(velocity_sub) + jnp.finfo(resid.dtype).eps)
            )
            ratio_accepted = (
                (geodesic_acceptance_ratio > zero)
                & (acceleration_ratio > zero)
                & (acceleration_ratio <= geodesic_acceptance_ratio)
            )

            def accelerated_objective(_):
                resid_accelerated = residual_value(theta + accelerated_step)
                accel_resid_loss = jnp.sum(resid_accelerated**2)
                accel_penalty = trial_penalty(accelerated_step_sub)
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
            step_sub = jnp.where(used_geodesic, accelerated_step_sub, velocity_sub)
            loss_candidate = jnp.where(used_geodesic, loss_accelerated, loss_velocity)
            resid_loss_candidate = jnp.where(
                used_geodesic, resid_loss_accelerated, resid_loss_velocity
            )
            penalty_candidate = jnp.where(
                used_geodesic, penalty_accelerated, penalty_velocity
            )
        else:
            step = velocity
            step_sub = velocity_sub
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
                new_cache = CholeskyCache(normal_matrix, ~improved, ridge)
            else:
                new_cache = QRCache(qr_R, ~improved, ridge)
            new_lm_state = RidgeLMState(
                new_damping,
                ridge,
                resid,
                Jt,
                ~improved,
                aux,
                lm_state.hyper,
                solver_cache=new_cache,
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
                jnp.linalg.norm(penalty_gradient),
                jnp.linalg.norm(step_sub),
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
        ``gtol`` bounds the whitened ridge stationarity ``info.grad_norm =
        ||J~'r + ridge [y_m; 0]||`` (the dual W^{-1}-norm of the
        half-gradient) and ``xtol`` the accepted whitened step norm
        ``info.step_norm = ||delta_y||`` (the W-norm of the step) -- either
        fires "done with the current fixed-ridge problem". The calibration
        recipe reads ``gtol ~ 1e-3 * ridge * sqrt(q(x*))`` since
        ``penalty_grad_norm = sqrt(penalty_value)``. Meanwhile
        ``atol > 0`` ADDITIONALLY requires ``sqrt(resid_loss) <= atol``
        (the model equations actually solved, the ridgeless-endgame check)
        and never stops the solve alone; ``atol > 0`` therefore requires a
        positive ``gtol`` or ``xtol`` (validated loudly). Second,
        ``lm_state=None`` always builds the state with :meth:`init`
        (resolving ``ridge=None`` needs the residual dtype); a
        caller-supplied ``lm_state`` must carry a positive ``ridge``.

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
                "interpolating iterate before the seminorm is minimized. "
                "Calibrate gtol from a pilot run as roughly 1e-3 * ridge * "
                "info.penalty_grad_norm (the relative-stationarity recipe)"
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
        if touched and lm_state.solver_cache is not None:
            cache = lm_state.solver_cache
            lm_state = dataclasses.replace(
                lm_state,
                solver_cache=dataclasses.replace(
                    cache, valid=cache.valid & ~problem_changed
                ),
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
        if lm_state.solver_cache is not None:
            cache = lm_state.solver_cache
            updates["solver_cache"] = dataclasses.replace(
                cache, ridge=jnp.asarray(cache.ridge, dtype=dtype)
            )
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
        if lm_state.solver_cache is not None:
            updates["solver_cache"] = jax.tree.map(
                jnp.zeros_like, lm_state.solver_cache
            )
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
            n_m = self._block_sizes(theta.shape[0])[0]
            ctx = MetricContext(
                x=theta, lm_state=result.lm_state, args=result.args, p=p
            )
            ridge = jnp.asarray(result.lm_state.ridge, dtype=residual.dtype)
            y_m = jnp.asarray(
                self.metric.factor_apply(theta[:n_m], ctx), dtype=residual.dtype
            )
            loss = jnp.sum(residual**2) + ridge * jnp.sum(y_m**2)
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
        # The returned state rides along as inert MetricContext data for the
        # factor callbacks, like the frozen ridge.
        ad_lm_state = jax.lax.stop_gradient(result.lm_state)
        x_dot = self._ad_x_tangent_from_p(
            ad_x, ad_args, ad_p, ad_p_dot, ad_ridge, ad_lm_state
        )
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
        if self.ad_solver is None:
            # Matrix-free forward -> matrix-free AD.
            return "normal_cg" if self._resolved_solver() == "normal_cg" else "cholesky"
        return "cholesky" if isinstance(self.ad_solver, Cholesky) else "normal_cg"

    def _ad_x_tangent_from_p(self, x, args, p, p_dot, ridge, lm_state):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        if self._resolved_ad_solver() == "cholesky":
            return self._ad_tangent_cholesky(x, args, p, p_dot, ridge, lm_state)
        return self._ad_tangent_normal_cg(x, args, p, p_dot, ridge, lm_state)

    def _ad_linearization(self, x, args, p, p_dot):
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        return theta, unravel, residual, theta_jvp, residual_p_dot

    def _ad_tangent_cholesky(self, x, args, p, p_dot, ridge, lm_state):
        # The GN implicit rule posed on the whitened variable y = F_bar x:
        # (J~'J~ + ridge E) y_dot = -J~'(dr/dp) p_dot, then
        # x_dot = F_bar^{-1} y_dot -- no damping; the matrix is PD under the
        # identification condition because ridge > 0 by contract.
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        n_m = self._block_sizes(theta.shape[0])[0]
        ctx = MetricContext(x=theta, lm_state=lm_state, args=args, p=p)
        Jt = self._assemble_jt(theta_jvp, theta, residual)
        ridge_typed = jnp.asarray(ridge, dtype=residual.dtype)
        Jt_sub = jnp.asarray(
            self._extended_solve_transpose(Jt, ctx), dtype=residual.dtype
        )
        diag = jnp.arange(n_m)
        normal_matrix = (Jt_sub @ Jt_sub.T).at[diag, diag].add(ridge_typed)
        factor = jsp_linalg.cho_factor(normal_matrix)
        y_dot = jsp_linalg.cho_solve(factor, -(Jt_sub @ residual_p_dot))
        theta_dot = jnp.asarray(self._extended_solve(y_dot, ctx), residual.dtype)
        return unravel(theta_dot)

    def _ad_tangent_normal_cg(self, x, args, p, p_dot, ridge, lm_state):
        # Matrix-free CG on the same whitened PD operator (see
        # _ad_tangent_cholesky); matvec = J~'(J~ u) + ridge [u_m; 0].
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        n_m, n_f = self._block_sizes(theta.shape[0])
        ctx = MetricContext(x=theta, lm_state=lm_state, args=args, p=p)
        theta_transpose = jax.linear_transpose(theta_jvp, theta)

        def JT(cotangent):
            return theta_transpose(cotangent)[0]

        ridge_typed = jnp.asarray(ridge, dtype=residual.dtype)

        def normal_matvec(u):
            gauss_newton = jnp.asarray(
                self._extended_solve_transpose(
                    JT(
                        theta_jvp(
                            jnp.asarray(self._extended_solve(u, ctx), residual.dtype)
                        )
                    ),
                    ctx,
                ),
                dtype=residual.dtype,
            )
            metric_shift = jnp.concatenate(
                [u[:n_m], jnp.zeros(n_f, dtype=residual.dtype)]
            )
            return gauss_newton + ridge_typed * metric_shift

        cg_tol = self._ad_cg_tol(residual.dtype)
        cg_atol = jnp.asarray(self.ad_solver_atol, dtype=residual.dtype)
        if self.ad_solver_preconditioner is None:
            apply_M = None
        else:
            # The AD system is undamped, so the preconditioner sees zero
            # damping (requires_positive_damping subclasses were rejected at
            # construction).
            zero_damping = jnp.zeros((), dtype=residual.dtype)

            def apply_M(v):
                return self.ad_solver_preconditioner.apply(v, zero_damping, ctx)

        def solve(matvec, rhs_value):
            solution, _ = jsp_sparse_linalg.cg(
                matvec,
                rhs_value,
                tol=cg_tol,
                atol=cg_atol,
                maxiter=self.ad_solver_maxiter,
                M=apply_M,
            )
            return solution

        rhs = -jnp.asarray(
            self._extended_solve_transpose(JT(residual_p_dot), ctx),
            dtype=residual.dtype,
        )
        y_dot = jax.lax.custom_linear_solve(
            normal_matvec,
            rhs,
            solve,
            symmetric=True,
        )
        theta_dot = jnp.asarray(self._extended_solve(y_dot, ctx), residual.dtype)
        return unravel(theta_dot)

    def _ad_cg_tol(self, dtype):
        if self.ad_solver_tol is not None:
            return jnp.asarray(self.ad_solver_tol, dtype=dtype)
        default_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return jnp.asarray(default_tol, dtype=dtype)
