"""Levenberg-Marquardt nonlinear least-squares solver for JAX, supporting both
unpenalized geometry-selected interpolation (``ridge=None``) and
Tikhonov-penalized ridge optimization (``ridge > 0``), plus the
:class:`AnnealRidge` continuation callback.
"""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from nlls_gram.linear_solvers import (
    Cholesky,
    CholeskyCache,
    QRCache,
    Subproblem,
)
from nlls_gram.solver_base import (
    SolverBase,
    _armijo_backtracking,
    _init_residual_and_aux,
    _validate_ridge_value,
)
from nlls_gram.types import (
    LMHyperparams,
    LMInfo,
    LMState,
    SolverAction,
    SolverContext,
    _cast_lm_hyper,
    _damping_floor,
)
from nlls_gram.utilities import mm

__all__ = [
    "AnnealRidge",
    "CholeskyCache",
    "LevenbergMarquardt",
    "QRCache",
]


@dataclass(frozen=True)
class AnnealRidge:
    """Ridge-continuation ``solve`` callback: multiply ``state.ridge`` by
    ``decrease`` whenever the current level has yielded what it can, never
    below ``ridge_floor``.

    Works identically with :class:`LevenbergMarquardt` and
    :class:`~nlls_gram.Newton` when constructed with a positive ``ridge``.

    A level has yielded what it can when it is

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
      ``stall_rtol ~ 0.99`` when that happens.

    ``ridge_floor`` is REQUIRED and strictly positive (``ridge = 0`` is out
    of the solver's contract). Usage::

        anneal = AnnealRidge(ridge_floor=1e-10)
        result = solver.solve(x0, callback=anneal,
                              user_state=anneal.init_state(),
                              gtol=1e-8, atol=1e-8)

    A wrapping callback calls the instance, inspects
    ``action.state.ridge < ctx.state.ridge`` for a level advance, and returns a
    ``dataclasses.replace`` of the action -- e.g. gating an expensive
    preconditioner rebuild on the advance with ``jax.lax.cond`` so the
    eigendecomposition is paid only when it fires::

        def driver_callback(ctx):
            action = anneal(ctx)
            advanced = action.state.ridge < ctx.state.ridge
            precond = jax.lax.cond(
                advanced,
                lambda: BlockEigenPreconditioner(families(ctx.x), PERM),
                lambda: ctx.state.preconditioner,
            )
            return dataclasses.replace(
                action,
                state=dataclasses.replace(
                    action.state, preconditioner=precond
                ),
            )
    """

    ridge_floor: float
    decrease: float = 0.1
    grad_rtol: float = 1e-2
    stall_rtol: float = 0.0

    def __post_init__(self):
        if not 0 < self.decrease < 1:
            raise ValueError("decrease must lie strictly between 0 and 1")
        _validate_ridge_value(self.ridge_floor, "ridge_floor")
        if self.grad_rtol <= 0:
            raise ValueError("grad_rtol must be positive")
        if not 0 <= self.stall_rtol < 1:
            raise ValueError("stall_rtol must lie in [0, 1)")

    def init_state(self, dtype=None):
        """The initial ``user_state``: both trackers at ``+inf``. ``dtype``
        defaults to the JAX default float; pass the problem dtype explicitly
        for a float32 program under enabled x64."""
        infinity = jnp.asarray(
            jnp.inf, dtype=jnp.result_type(float) if dtype is None else dtype
        )
        return {"reference": infinity, "previous": infinity}

    def __call__(self, ctx):
        state = ctx.state
        ridge = state.ridge
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
        advanced = new_ridge < ridge
        fresh_level = jnp.asarray(jnp.inf, dtype)
        state_dtype = jnp.asarray(ctx.user_state["reference"]).dtype
        new_state = dataclasses.replace(state, ridge=new_ridge)
        updated_previous = jnp.where(ctx.info.accepted, grad_norm, previous)
        new_user_state = {
            **ctx.user_state,
            "reference": jnp.where(advanced, fresh_level, reference).astype(
                state_dtype
            ),
            "previous": jnp.where(advanced, fresh_level, updated_previous).astype(
                state_dtype
            ),
        }
        return SolverAction(state=new_state, user_state=new_user_state)


class LevenbergMarquardt(SolverBase):
    """Levenberg-Marquardt nonlinear least-squares solver over a JAX pytree
    ``x``.

    Supports two root-selection regimes controlled by ``ridge``:

    - **Unpenalized (``ridge=None``, default)**: minimizes ``||r(x, args, p)||^2``
      with optional :class:`~nlls_gram.Metric` ``W = F'F`` shaping the
      trust-region geometry so the small-damping Gauss-Newton limit selects
      minimum-``W``-norm corrections. Stopping rules ``atol``, ``gtol``, and
      ``xtol`` combine disjunctively (``atol | gtol | xtol``).
    - **Penalized ridge (``ridge > 0``)**: minimizes the Tikhonov objective
      ``||r(x, args, p)||^2 + ridge * ||x_m||_W^2`` (defaulting ``metric`` to
      the Euclidean identity over all coordinates when ``metric=None``), with
      ``ridge`` carried in :attr:`LMState.ridge` for continuation schedules
      (:class:`AnnealRidge`). ``atol`` acts as a conjunctive filter on
      ``gtol | xtol``.

    ``linear_solver`` accepts :class:`~nlls_gram.Cholesky` (default),
    :class:`~nlls_gram.QR`, :class:`~nlls_gram.CG`, or
    :class:`~nlls_gram.GramCG` (unpenalized only). ``ad_solver`` configures the
    implicit-AD tangent system (``None`` inherits the forward family).
    """

    def __init__(
        self,
        residual_fn,
        *,
        metric=None,
        ridge=None,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        c1=1e-4,
        decrease_factor=0.5,
        maxls=20,
        max_stepsize=1.0,
        linesearch_tol=0.0,
        linear_solver=Cholesky(),  # noqa: B008 -- frozen, immutable default
        jacobian_mode="auto",
        jacobian_chunk=None,
        ad_solver=None,
        has_aux=False,
        cache_jacobian=True,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    ):
        if init_damping <= 0 or damping_decrease <= 0 or damping_increase <= 0:
            raise ValueError(
                "init_damping, damping_decrease, and damping_increase must be positive"
            )

        self._init_common(
            residual_fn,
            linear_solver=linear_solver,
            metric=metric,
            ridge=ridge,
            has_aux=has_aux,
            cache_jacobian=cache_jacobian,
            geodesic_acceleration=geodesic_acceleration,
            geodesic_acceptance_ratio=geodesic_acceptance_ratio,
            ad_solver=ad_solver,
            jacobian_mode=jacobian_mode,
            jacobian_chunk=jacobian_chunk,
        )
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.c1 = c1
        self.decrease_factor = decrease_factor
        self.maxls = maxls
        self.max_stepsize = max_stepsize
        self.linesearch_tol = linesearch_tol

        self._static_key = self._common_static_key()
        self._static_hash = hash(self._static_key)
        self._sealed = True

    def hyperparams(self, dtype=None):
        """``LMHyperparams`` built from the constructor values."""
        iterative_tol = self.iterative_tol
        if iterative_tol is None:
            if dtype is None:
                iterative_tol = jnp.nan
            else:
                iterative_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return LMHyperparams(
            jnp.asarray(self.damping_decrease, dtype=dtype),
            jnp.asarray(self.damping_increase, dtype=dtype),
            jnp.asarray(self.geodesic_acceptance_ratio, dtype=dtype),
            jnp.asarray(iterative_tol, dtype=dtype),
            jnp.asarray(self.iterative_atol, dtype=dtype),
            None
            if self.iterative_maxiter is None
            else jnp.asarray(self.iterative_maxiter, dtype=jnp.int32),
            c1=jnp.asarray(self.c1, dtype=dtype),
            decrease_factor=jnp.asarray(self.decrease_factor, dtype=dtype),
            maxls=jnp.asarray(self.maxls, dtype=jnp.int32),
            max_stepsize=jnp.asarray(self.max_stepsize, dtype=dtype),
            linesearch_tol=jnp.asarray(self.linesearch_tol, dtype=dtype),
        )

    def init(self, x0, args=None, *, p=None):
        """Build the initial :class:`LMState` at ``x0``.

        Uses ``jax.eval_shape`` to type ``damping`` and ``ridge`` and size the
        Jacobian and linear-solver cache buffers when ``cache_jacobian=True``
        on a dense linear solver without paying a device residual evaluation.
        ``hyper`` stays ``None`` so manual ``update`` loops carry no extra
        buffers; ``solve`` populates it.
        """
        self._check_residual_args(args, p)
        residual_shape, aux_shape = _init_residual_and_aux.eval_shape(self, x0, args, p)
        theta, _ = ravel_pytree(x0)
        p_dim = theta.size
        n_m, _ = self._block_sizes(p_dim)
        dtype = residual_shape.dtype
        damping = jnp.maximum(
            jnp.asarray(self.init_damping, dtype=dtype), _damping_floor(dtype)
        )
        ridge = self._resolve_ridge(dtype) if self.penalized else None
        instances = dict(
            metric=self.initial_metric, preconditioner=self.initial_preconditioner
        )
        if not self.cache_jacobian:
            return LMState(damping, ridge=ridge, **instances)
        m = residual_shape.size
        return LMState(
            damping,
            ridge,
            resid=jnp.zeros(residual_shape.shape, dtype=dtype),
            Jt=jnp.zeros((p_dim, m), dtype=dtype),
            jacobian_valid=jnp.asarray(False, dtype=jnp.bool_),
            aux=jax.tree.map(lambda s: jnp.zeros(s.shape, dtype=s.dtype), aux_shape),
            solver_cache=self.linear_solver.new_cache(
                m, p_dim, n_m, dtype, self.penalized
            ),
            **instances,
        )

    def _solve_state(self, x0, args, p, state):
        if state is None:
            if self.cache_jacobian or self.penalized:
                return self.init(x0, args, p=p)
            return LMState(
                jnp.asarray(self.init_damping),
                metric=self.initial_metric,
                preconditioner=self.initial_preconditioner,
            )
        if self.penalized:
            state = self._validate_supplied_ridge(
                state, jnp.asarray(state.damping).dtype
            )
        elif state.ridge is not None:
            raise ValueError(
                "the caller-supplied state has a non-None ridge, but this "
                "solver was constructed with ridge=None"
            )
        return self._cold_state(self._resolved_state(state))

    def _initial_info(self, x, state, args, p):
        residual, aux = self._residual_and_aux(x, args, p)
        dtype = residual.dtype
        resid_loss = jnp.sum(residual**2)
        zero = jnp.zeros((), dtype=dtype)
        one = jnp.ones((), dtype=dtype)
        infinity = jnp.asarray(jnp.inf, dtype=dtype)
        if self.penalized:
            theta, _ = ravel_pytree(x)
            ridge = jnp.asarray(state.ridge, dtype=dtype)
            n_m = self._block_sizes(theta.shape[0])[0]
            resolved = self._resolved_state(state)
            ctx = SolverContext(x=theta, state=resolved, args=args, p=p)
            y_m = self._factor_apply_metric_block(theta[:n_m], ctx, dtype)
            penalty_value = jnp.sum(y_m**2)
            loss = resid_loss + ridge * penalty_value
            ridge_fields = dict(
                ridge=ridge,
                resid_loss=resid_loss,
                penalty_value=penalty_value,
                penalty_grad_norm=infinity,
            )
        else:
            loss = resid_loss
            ridge_fields = {}
        return LMInfo(
            loss=loss,
            loss_old=loss,
            loss_candidate=loss,
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            damping=jnp.asarray(state.damping, dtype=dtype),
            damping_factor=one,
            used_geodesic=jnp.asarray(False, dtype=jnp.bool_),
            acceleration_ratio=zero,
            grad_norm=infinity,
            step_norm=zero,
            aux=aux,
            stepsize=zero,
            num_linesearch_steps=jnp.asarray(0, dtype=jnp.int32),
            **ridge_fields,
        )

    def _cast_state(self, state, dtype):
        updates = dict(
            damping=jnp.asarray(state.damping, dtype=dtype),
            hyper=_cast_lm_hyper(state.hyper, dtype),
        )
        if self.penalized:
            updates["ridge"] = jnp.asarray(state.ridge, dtype=dtype)
            if state.solver_cache is not None:
                cache = state.solver_cache
                updates["solver_cache"] = dataclasses.replace(
                    cache, ridge=jnp.asarray(cache.ridge, dtype=dtype)
                )
        return dataclasses.replace(state, **updates)

    def update(self, x, state, args=None, p=None):
        """One Levenberg-Marquardt step: returns ``(x_new, state, info)``."""
        self._check_residual_args(args, p)
        theta, unravel = ravel_pytree(x)
        residual_flat, residual_value, resid, Jt, jvp_fn, JT, aux = (
            self._linearize_step(theta, unravel, state, args, p)
        )
        dtype = resid.dtype
        hyper = state.hyper if state.hyper is not None else self.hyperparams(dtype)
        damping_decrease = jnp.asarray(hyper.damping_decrease, dtype=dtype)
        damping_increase = jnp.asarray(hyper.damping_increase, dtype=dtype)
        damping_floor = _damping_floor(dtype)
        damping = jnp.maximum(jnp.asarray(state.damping, dtype=dtype), damping_floor)

        n_m, n_f = self._block_sizes(theta.shape[0])
        resolved_state = self._resolved_state(state)
        ctx = SolverContext(x=theta, state=resolved_state, args=args, p=p)
        (
            ridge,
            y_m,
            penalty_value_old,
            penalty_gradient,
            resid_loss_old,
            loss_old,
        ) = self._base_objective(resid, theta, n_m, n_f, state, ctx)

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
                cache=state.solver_cache,
                cache_enabled=self.cache_jacobian,
                hyper=hyper,
                ctx=ctx,
                penalized=self.penalized,
            )
        )
        grad = step_solver.grad
        velocity_sub = step_solver.velocity()
        velocity = jnp.asarray(self._extended_solve(velocity_sub, ctx), dtype=dtype)
        zero = jnp.zeros((), dtype=dtype)
        one = jnp.ones((), dtype=dtype)
        slope = mm(grad, velocity_sub)
        descent_ok = (
            jnp.isfinite(loss_old)
            & jnp.all(jnp.isfinite(velocity_sub))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.isfinite(slope)
            & (slope < zero)
        )

        if self.geodesic_acceleration:
            geodesic_acceptance_ratio = jnp.asarray(
                hyper.geodesic_acceptance_ratio, dtype=dtype
            )
            (
                acceleration_sub,
                acceleration,
                acceleration_ratio,
                raw_ratio_accepted,
            ) = self._geodesic_correction(
                residual_flat,
                theta,
                velocity,
                velocity_sub,
                step_solver.correction,
                ctx,
                geodesic_acceptance_ratio,
                dtype,
            )
            ratio_accepted = (
                descent_ok
                & raw_ratio_accepted
                & jnp.all(jnp.isfinite(acceleration_sub))
                & jnp.all(jnp.isfinite(acceleration))
            )
            u_a = jnp.where(
                ratio_accepted, acceleration_sub, jnp.zeros_like(velocity_sub)
            )
            a = jnp.where(ratio_accepted, acceleration, jnp.zeros_like(velocity))
            u_a_m = u_a[:n_m] if self.penalized else None
        else:
            u_a = None
            a = None
            u_a_m = None
            acceleration_ratio = zero
            ratio_accepted = jnp.asarray(False, dtype=jnp.bool_)

        (
            theta_new,
            loss,
            resid_loss_new,
            penalty_new,
            loss_candidate,
            improved,
            stepsize,
            trial_stepsize,
            trials,
            _,
        ) = _armijo_backtracking(
            residual_value,
            theta,
            loss_old,
            velocity,
            slope,
            hyper,
            descent_ok,
            a=a,
            y_m=y_m if self.penalized else None,
            u_m=velocity_sub[:n_m] if self.penalized else None,
            u_a_m=u_a_m,
            ridge=ridge if self.penalized else None,
            resid_loss_old=resid_loss_old if self.penalized else None,
            penalty_old=penalty_value_old if self.penalized else None,
        )

        safe_alpha = jnp.where(improved, stepsize, one)
        damping_factor = jnp.where(
            improved,
            jnp.minimum(damping_increase, damping_decrease / safe_alpha),
            damping_increase,
        )
        new_damping = jnp.maximum(damping * damping_factor, damping_floor)
        used_geodesic = improved & ratio_accepted
        step_alpha = jnp.where(
            improved,
            stepsize,
            jnp.where(trial_stepsize > zero, trial_stepsize, one),
        )
        step_sub = (
            step_alpha * velocity_sub
            if u_a is None
            else step_alpha * velocity_sub + 0.5 * (step_alpha**2) * u_a
        )

        instances = dict(metric=state.metric, preconditioner=state.preconditioner)
        ridge_state = ridge if self.penalized else None
        if self.cache_jacobian and self.linear_solver.materializes_jacobian:
            new_state = LMState(
                new_damping,
                ridge=ridge_state,
                resid=resid,
                Jt=Jt,
                jacobian_valid=~improved,
                aux=aux,
                hyper=state.hyper,
                solver_cache=step_solver.make_cache(~improved),
                **instances,
            )
        else:
            new_state = LMState(
                new_damping,
                ridge=ridge_state,
                hyper=state.hyper,
                **instances,
            )

        if self.penalized:
            ridge_info = dict(
                ridge=ridge,
                resid_loss=resid_loss_new,
                penalty_value=penalty_new,
                penalty_grad_norm=jnp.linalg.norm(penalty_gradient),
            )
        else:
            ridge_info = {}

        return (
            unravel(theta_new),
            new_state,
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
                aux=aux,
                stepsize=stepsize,
                num_linesearch_steps=trials,
                **ridge_info,
            ),
        )
