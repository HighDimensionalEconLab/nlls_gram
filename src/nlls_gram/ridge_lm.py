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
``y = F_bar x`` (``W = F'F``, ``F_bar = blockdiag(F, sqrt(free_scale) I)``):
stock Euclidean
LM on the augmented residual ``[r; sqrt(ridge) y_m]`` (Marquardt 1963; More
1978) with geodesic acceleration (Transtrum-Sethna 2012); the ``qr`` path
uses corrected semi-normal equations (Bjorck 1987; Bjorck 1996 Sec. 6.6.5).
"""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
from jax.flatten_util import ravel_pytree

from nlls_gram.linear_solvers import (
    CG,
    Cholesky,
    CholeskyCache,
    QRCache,
    Subproblem,
)
from nlls_gram.lm_core import LevenbergMarquardtBase
from nlls_gram.lm_types import (
    LMInfo,
    LMSolveAction,
    LMState,
    SolverContext,
    _cast_hyper,
    _damping_floor,
)
from nlls_gram.metrics import Metric
from nlls_gram.preconditioners import Preconditioner
from nlls_gram.utilities import (
    _static_key_component,
    _zero_tangent_leaf,
    canonicalize_residual,
)

__all__ = [
    "CholeskyCache",
    "RidgeContinuation",
    "QRCache",
    "RidgeLevenbergMarquardt",
    "ridge_continuation",
]


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

    schedule = RidgeContinuation(
        decrease=decrease,
        ridge_floor=ridge_floor,
        grad_rtol=grad_rtol,
        stall_rtol=stall_rtol,
    )
    infinity = jnp.asarray(
        jnp.inf, dtype=jnp.result_type(float) if dtype is None else dtype
    )
    return schedule, {"reference": infinity, "previous": infinity}


@dataclass(frozen=True)
class RidgeContinuation:
    """The callback :func:`ridge_continuation` builds; see it for the schedule.

    A frozen dataclass rather than a closure because ``solve`` marks the
    callback a jit STATIC argument: a fresh closure would key a fresh
    compilation of the whole solve loop on every construction. Two equal
    schedules compare equal and share one compiled loop. ``ridge_floor`` must
    be a concrete float for that sharing -- a traced value is unhashable and
    falls back to identity.
    """

    ridge_floor: float
    decrease: float = 0.1
    grad_rtol: float = 1e-2
    stall_rtol: float = 0.0

    def __post_init__(self):
        if not 0 < self.decrease < 1:
            raise ValueError("decrease must lie strictly between 0 and 1")
        if not isinstance(self.ridge_floor, (jax.Array, jax.core.Tracer)) and (
            float(self.ridge_floor) <= 0
        ):
            raise ValueError(
                "ridge_floor must be strictly positive (ridge = 0 is "
                "unsupported by RidgeLevenbergMarquardt)"
            )
        if self.grad_rtol <= 0:
            raise ValueError("grad_rtol must be positive")
        if not 0 <= self.stall_rtol < 1:
            raise ValueError("stall_rtol must lie in [0, 1)")

    def __call__(self, ctx):
        ridge = ctx.lm_state.ridge
        dtype = ridge.dtype
        grad_norm = jnp.asarray(ctx.info.grad_norm, dtype=dtype)
        reference = jnp.asarray(ctx.user_state["reference"], dtype=dtype)
        previous = jnp.asarray(ctx.user_state["previous"], dtype=dtype)
        # +inf marks "no observation at this level yet": the first step after a
        # ridge decrease (or the initial step) sets the reference and can never
        # read as stalled.
        reference = jnp.where(jnp.isfinite(reference), reference, grad_norm)
        stationary = grad_norm <= jnp.asarray(self.grad_rtol, dtype) * reference
        if self.stall_rtol > 0:
            # ACCEPTED steps only: a rejected step leaves x (and so the
            # gradient) unchanged -- that is the trust region adapting, not the
            # level converging -- while an accepted step that improved the
            # gradient by less than the stall factor means the level has
            # yielded what it can.
            stalled = ctx.info.accepted & (
                grad_norm >= jnp.asarray(self.stall_rtol, dtype) * previous
            )
        else:
            stalled = jnp.asarray(False)
        new_ridge = jnp.where(
            stationary | stalled,
            jnp.maximum(
                ridge * jnp.asarray(self.decrease, dtype),
                jnp.asarray(self.ridge_floor, dtype),
            ),
            ridge,
        )
        # Reset the trackers when the level actually changes; at the floor the
        # ridge is unchanged, so convergence is not suppressed and gtol/atol
        # can fire.
        advanced = new_ridge < ridge
        fresh_level = jnp.asarray(jnp.inf, dtype)
        state_dtype = jnp.asarray(ctx.user_state["reference"]).dtype
        return LMSolveAction(
            lm_state=dataclasses.replace(ctx.lm_state, ridge=new_ridge),
            user_state={
                "reference": jnp.where(advanced, fresh_level, reference).astype(
                    state_dtype
                ),
                "previous": jnp.where(advanced, fresh_level, grad_norm).astype(
                    state_dtype
                ),
            },
        )


class RidgeLevenbergMarquardt(LevenbergMarquardtBase):
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
      inverse, handed the same :class:`~nlls_gram.SolverContext` as the
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
    forwards, and a ``CG`` forward keeps its family AND its preconditioner
    -- the typed apply is damping-analytic, so the forward hook serves the
    undamped system at zero damping exactly (hooks marked
    ``requires_positive_damping`` fall back to unpreconditioned), while the
    AD tolerance and iteration budget stay at the AD defaults; pass
    ``ad_solver=CG(...)`` to pin those. Failed statuses
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
        if (
            ridge is not None
            and not isinstance(ridge, (jax.Array, jax.core.Tracer))
            and float(ridge) <= 0.0
        ):
            raise ValueError(
                "ridge must be strictly positive (ridge = 0 is unsupported: "
                "use ridge_continuation with a positive ridge_floor to "
                "approach the ridgeless limit)"
            )
        if init_damping <= 0 or damping_decrease <= 0 or damping_increase <= 0:
            raise ValueError(
                "init_damping, damping_decrease, and damping_increase must be positive"
            )
        if min_damping is not None and not 0 < min_damping <= init_damping:
            raise ValueError("min_damping must be positive and at most init_damping")
        if max_damping is not None and max_damping < init_damping:
            raise ValueError("max_damping must be at least init_damping")
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
        self._validate_configuration(linear_solver, ad_solver, penalized=True)
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
            # ad_solver=None matches the forward family, and a CG forward
            # also hands its preconditioner to the undamped implicit solve:
            # the AD operator IS the forward operator at zero damping, the
            # typed apply is damping-analytic there, and unpreconditioned
            # implicit CG degrades exactly like the forward as the ridge
            # shrinks. The AD tolerance and budget stay at the AD defaults
            # (run to tolerance); damping-dividing hooks fall back to
            # unpreconditioned.
            if (
                ad_solver is None
                and isinstance(linear_solver, CG)
                and not linear_solver.preconditioner.requires_positive_damping
            ):
                self.ad_solver_preconditioner = linear_solver.preconditioner
            else:
                self.ad_solver_preconditioner = None
        self.has_aux = has_aux
        # Only the dense paths materialize J' (and the cholesky/qr caches ride
        # on the same reject-reuse lifecycle), so the flag is inert for the
        # matrix-free normal_cg forward.
        self.cache_jacobian = cache_jacobian and not isinstance(linear_solver, CG)
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        # The forward preconditioner is the one whose prepared state is carried
        # (the AD role runs once, at the solution). Whether a hook is stateful
        # is a static property of its class, so the slots and their lax.cond
        # compile away entirely for the stateless default.
        self.preconditioner = self.normal_cg_preconditioner
        self._metric_prepares = type(metric).prepare is not Metric.prepare
        self._precond_prepares = self.preconditioner is not None and (
            type(self.preconditioner).prepare is not Preconditioner.prepare
        )
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

    def _resolve_ridge(self, dtype):
        if self.ridge is None:
            return jnp.asarray(jnp.sqrt(jnp.finfo(dtype).eps), dtype=dtype)
        return jnp.asarray(self.ridge, dtype=dtype)

    def init(self, x0, args=None, *, p=None):
        """Build the initial :class:`LMState` at ``x0``.

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
        hooks = self._init_hook_state(theta, LMState(damping, ridge), args, p)
        if not self.cache_jacobian:
            return LMState(damping, ridge, **hooks)
        p_dim = theta.size
        m = residual.size
        return LMState(
            damping,
            ridge,
            resid=jnp.zeros(residual.shape, dtype=dtype),
            Jt=jnp.zeros((p_dim, m), dtype=dtype),
            jacobian_valid=jnp.asarray(False, dtype=jnp.bool_),
            aux=jax.tree.map(jnp.zeros_like, aux),
            solver_cache=self.linear_solver.new_cache(m, p_dim, n_m, dtype, True),
            **hooks,
        )

    def _initial_info(self, x, lm_state, args, p):
        # grad_norm and penalty_grad_norm are +inf sentinels (computing them
        # would cost a Jacobian before the first step) and step_norm is zero;
        # none can satisfy gtol/xtol before any update has run.
        residual, aux = self._residual_and_aux(x, args, p)
        resid_loss = jnp.sum(residual**2)
        theta, _ = ravel_pytree(x)
        ridge = jnp.asarray(lm_state.ridge, dtype=residual.dtype)
        n_m = self._block_sizes(theta.shape[0])[0]
        ctx = self._carried_ctx(theta, lm_state, args, p)
        y_m = jnp.asarray(
            self.metric.factor_apply(theta[:n_m], ctx), dtype=residual.dtype
        )
        penalty_value = jnp.sum(y_m**2)
        loss = resid_loss + ridge * penalty_value
        zero = jnp.zeros((), dtype=residual.dtype)
        one = jnp.ones((), dtype=residual.dtype)
        infinity = jnp.asarray(jnp.inf, dtype=residual.dtype)
        return LMInfo(
            loss=loss,
            loss_old=loss,
            loss_candidate=loss,
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            damping=jnp.asarray(lm_state.damping, dtype=residual.dtype),
            damping_factor=one,
            used_geodesic=jnp.asarray(False, dtype=jnp.bool_),
            acceleration_ratio=zero,
            grad_norm=infinity,
            step_norm=zero,
            ridge=ridge,
            resid_loss=resid_loss,
            penalty_value=penalty_value,
            penalty_grad_norm=infinity,
            aux=aux,
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

        # TRUE-residual linearization: matrix-free closures when the linear
        # solver never materializes J, dense J' (reused from the cache after a
        # rejected step) otherwise.
        jvp_fn = JT = Jt = None
        if not self.linear_solver.materializes_jacobian:
            if self.has_aux:
                resid, jvp_fn, aux = jax.linearize(residual_flat, theta, has_aux=True)
            else:
                resid, jvp_fn = jax.linearize(residual_flat, theta)
                aux = None
            transpose_fn = jax.linear_transpose(jvp_fn, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

        elif self.cache_jacobian:
            resid, Jt, aux = jax.lax.cond(
                lm_state.jacobian_valid,
                lambda _: (lm_state.resid, lm_state.Jt, lm_state.aux),
                lambda _: self._dense_resid_jt_aux(residual_flat, theta),
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
            # A None ridge is a legal LMState (the metric solver leaves it
            # unset); this solver needs one.
            raise ValueError(
                "the lm_state has no ridge; create it with init(x, args, p=p)"
            )
        ridge = jnp.asarray(lm_state.ridge, dtype=resid.dtype)

        # The subproblem is posed on the whitened variable y = F_bar x, where
        # the penalty rows are the constant [I_{n_m} | 0]: the half-gradient is
        # g = F_bar^{-T} J'r + ridge [y_m; 0] (grad F = 2 g; the factor cancels
        # in the LM equations), and every gradient/step quantity below --
        # including the reported norms -- is the whitened one. y_m doubles as
        # the pre-step penalty value ||y_m||^2.
        n_m, n_f = self._block_sizes(theta.shape[0])
        metric_state, precond_state = self._hook_state(theta, lm_state, args, p)
        ctx = SolverContext(
            x=theta,
            lm_state=lm_state,
            args=args,
            p=p,
            metric_state=metric_state,
            preconditioner_state=precond_state,
        )
        y_m = jnp.asarray(self.metric.factor_apply(theta[:n_m], ctx), dtype=resid.dtype)
        penalty_value_old = jnp.sum(y_m**2)
        penalty_gradient = jnp.concatenate([y_m, jnp.zeros(n_f, dtype=resid.dtype)])

        step_solver = self.linear_solver.prepare(
            Subproblem(
                resid=resid,
                theta=theta,
                Jt=Jt,
                jvp_fn=jvp_fn,
                JT=JT,
                whiten=lambda v: self._extended_solve(v, ctx),
                whiten_transpose=lambda v: self._extended_solve_transpose(v, ctx),
                y_m=y_m,
                penalty_gradient=penalty_gradient,
                ridge=ridge,
                damping=damping,
                n_m=n_m,
                n_f=n_f,
                cache=lm_state.solver_cache,
                cache_enabled=self.cache_jacobian,
                hyper=hyper,
                ctx=ctx,
            )
        )
        grad = step_solver.grad

        # First-order step (velocity) and its ridge objective. The solves
        # produce the whitened step delta_y: the x-space step maps back through
        # the factor solve, and the trial penalty uses the linearity of the
        # change of variables -- F_bar(theta + step) = y + delta_y, so no
        # second factor application is ever needed.
        velocity_sub = step_solver.velocity()
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
            acceleration_sub = step_solver.correction(f_vv)
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
        # Thread the caches and prepared hook state built at this step's
        # pre-step (x, ridge): valid = ~improved marks them reusable exactly
        # when the step was rejected (x did not move). ridge passes through
        # unchanged -- only init() and callbacks set it. The input hyper (not
        # the fallback) passes through so the loop carry structure is stable.
        hooks = {}
        if self._metric_prepares:
            hooks["metric_state"] = metric_state
            hooks["metric_valid"] = ~improved
        if self._precond_prepares:
            hooks["precond"] = precond_state
            hooks["precond_valid"] = ~improved
        if self.cache_jacobian:
            new_lm_state = LMState(
                new_damping,
                ridge,
                resid,
                Jt,
                ~improved,
                aux,
                lm_state.hyper,
                solver_cache=step_solver.make_cache(~improved),
                **hooks,
            )
        else:
            new_lm_state = LMState(new_damping, ridge, hyper=lm_state.hyper, **hooks)
        return (
            unravel(theta_new),
            new_lm_state,
            LMInfo(
                loss=loss,
                loss_old=loss_old,
                loss_candidate=loss_candidate,
                accepted=improved,
                damping=new_damping,
                damping_factor=damping_factor,
                used_geodesic=used_geodesic,
                acceleration_ratio=acceleration_ratio,
                grad_norm=jnp.linalg.norm(grad),
                step_norm=jnp.linalg.norm(step_sub),
                ridge=ridge,
                resid_loss=resid_loss,
                penalty_value=penalty_value,
                penalty_grad_norm=jnp.linalg.norm(penalty_gradient),
                aux=aux,
            ),
        )

    def _validate_tolerances(self, atol, gtol, xtol):
        # atol is a CONJUNCTIVE filter on the true residual, never a stopping
        # rule alone: a residual-only test would stop at any interpolating
        # iterate, before the seminorm is minimized.
        concrete = [not isinstance(t, jax.core.Tracer) for t in (atol, gtol, xtol)]
        if (
            concrete[0]
            and atol > 0
            and (concrete[1] and gtol == 0)
            and (concrete[2] and xtol == 0)
        ):
            raise ValueError(
                "atol > 0 requires a positive gtol or xtol: atol is a "
                "conjunctive filter on the TRUE residual, never a stopping "
                "rule by itself. Calibrate gtol from a pilot run as roughly "
                "1e-3 * ridge * info.penalty_grad_norm"
            )

    def _solve_lm_state(self, x0, args, p, lm_state):
        if lm_state is None:
            # Unconditional init (no minimal-state fast path): resolving
            # ridge=None needs the residual dtype, and the dense caches need
            # their shapes.
            return self.init(x0, args, p=p)
        if lm_state.ridge is None:
            raise ValueError(
                "the caller-supplied lm_state has no ridge; create it with "
                "init(x, args, p=p) or set a positive ridge"
            )
        if (
            not isinstance(lm_state.ridge, jax.core.Tracer)
            and jnp.ndim(lm_state.ridge) == 0
            and float(lm_state.ridge) <= 0.0
        ):
            raise ValueError(
                "the caller-supplied lm_state.ridge must be strictly positive "
                "(ridge = 0 is unsupported)"
            )
        # Recast a hand-replaced ridge to the carried scalar dtype: a
        # weak-typed replace(state, ridge=1e-4) would change the jit input
        # aval and retrace the loop.
        return dataclasses.replace(
            lm_state,
            ridge=jnp.asarray(
                lm_state.ridge, dtype=jnp.asarray(lm_state.damping).dtype
            ),
        )

    def _initial_ad_point(self, x, lm_state, args, p):
        # The pre-loop ridge rides along: a failed lane's callback may have
        # left an invalid ridge behind, so the failed tangent uses this one.
        return (x, args, p, lm_state.ridge)

    def _check_action_state(self, lm_state):
        if lm_state.ridge is None:
            raise ValueError(
                "the callback action returned an lm_state without ridge; "
                "use dataclasses.replace(ctx.lm_state, ...) to preserve it"
            )

    def _apply_action_state(self, lm_state, previous):
        # A ridge change leaves the Jacobian cache VALID (J does not depend on
        # ridge) but invalidates the ridge-keyed normal/QR caches and
        # suppresses the convergence check, whose diagnostics were computed at
        # the old ridge. The recast keeps a weak-typed callback float from
        # changing the while_loop carry aval.
        new_ridge = jnp.asarray(lm_state.ridge, dtype=previous.ridge.dtype)
        changed = ~jnp.array_equal(new_ridge, previous.ridge, equal_nan=True)
        return dataclasses.replace(lm_state, ridge=new_ridge), changed

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
            ctx = self._carried_ctx(theta, result.lm_state, result.args, p)
            ridge = jnp.asarray(result.lm_state.ridge, dtype=residual.dtype)
            y_m = jnp.asarray(
                self.metric.factor_apply(theta[:n_m], ctx), dtype=residual.dtype
            )
            loss = jnp.sum(residual**2) + ridge * jnp.sum(y_m**2)
        return jnp.where(
            jnp.isfinite(loss), loss, jnp.asarray(jnp.inf, dtype=loss.dtype)
        )

    def _resolved_ad_solver(self):
        if self.ad_solver is None:
            # Matrix-free forward -> matrix-free AD.
            return "normal_cg" if isinstance(self.linear_solver, CG) else "cholesky"
        return "cholesky" if isinstance(self.ad_solver, Cholesky) else "normal_cg"

    def _ad_x_tangent(self, x, args, p, p_dot, result, ad_success, initial_ad_point):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        # A successful tangent uses the winner's own final ridge; a failed one
        # the pre-loop initial ridge. Both are stop-gradient'd -- lambda is
        # inert conditioning data, and the returned state rides along as
        # equally inert SolverContext data for the factor callbacks.
        final_ridge = jax.lax.stop_gradient(result.lm_state.ridge)
        initial_ridge = jax.lax.stop_gradient(initial_ad_point[3])
        ridge = jnp.where(
            ad_success, final_ridge, jnp.asarray(initial_ridge, final_ridge.dtype)
        )
        lm_state = jax.lax.stop_gradient(result.lm_state)
        if self._resolved_ad_solver() == "cholesky":
            return self._ad_tangent_cholesky(x, args, p, p_dot, ridge, lm_state)
        return self._ad_tangent_normal_cg(x, args, p, p_dot, ridge, lm_state)

    def _ad_tangent_cholesky(self, x, args, p, p_dot, ridge, lm_state):
        # The GN implicit rule posed on the whitened variable y = F_bar x:
        # (J~'J~ + ridge E) y_dot = -J~'(dr/dp) p_dot, then
        # x_dot = F_bar^{-1} y_dot -- no damping; the matrix is PD under the
        # identification condition because ridge > 0 by contract.
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        n_m = self._block_sizes(theta.shape[0])[0]
        ctx = self._frozen_ctx(theta, lm_state, args, p, self.ad_solver_preconditioner)
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
        ctx = self._frozen_ctx(theta, lm_state, args, p, self.ad_solver_preconditioner)
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
