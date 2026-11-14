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

__all__ = [
    "AnnealRidge",
    "CholeskyCache",
    "LevenbergMarquardt",
    "QRCache",
]


@dataclass(frozen=True)
class AnnealRidge:
    """Ridge-continuation callback reducing ``state.ridge`` down to ``ridge_floor``."""

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
        """Return initial ``user_state`` trackers at ``+inf``."""
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
    """Euclidean-step Levenberg-Marquardt solver for NLLS and generalized Tikhonov."""

    def __init__(
        self,
        residual_fn,
        *,
        ridge=None,
        ridge_metric=None,
        linear_solver=Cholesky(),  # noqa: B008
        ad_solver=None,
        jacobian_mode="auto",
        jacobian_chunk=None,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        c1=1e-4,
        decrease_factor=0.5,
        maxls=20,
        max_stepsize=1.0,
        linesearch_tol=0.0,
        has_aux=False,
        cache_jacobian=True,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
        complementarity=None,
        inequality=None,
    ):
        self._init_common(
            residual_fn,
            ridge=ridge,
            ridge_metric=ridge_metric,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
            jacobian_mode=jacobian_mode,
            jacobian_chunk=jacobian_chunk,
            has_aux=has_aux,
            cache_jacobian=cache_jacobian,
            geodesic_acceleration=geodesic_acceleration,
            geodesic_acceptance_ratio=geodesic_acceptance_ratio,
            complementarity=complementarity,
            inequality=inequality,
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
        """Return ``LMHyperparams`` built from constructor settings."""
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
        """Build initial ``LMState`` at ``x0`` without evaluating the residual."""
        self._check_residual_args(args, p)
        residual_shape, aux_shape, _ = _init_residual_and_aux.eval_shape(
            self, x0, args, p
        )
        theta, _ = ravel_pytree(x0)
        m, n = residual_shape.size, theta.size
        self.linear_solver.validate_shape(
            m,
            n,
            penalized=self.penalized,
            ridge_metric=self.initial_ridge_metric,
            damped=True,
            role="linear_solver",
        )
        dtype = residual_shape.dtype
        damping = jnp.maximum(
            jnp.asarray(self.init_damping, dtype=dtype), _damping_floor(dtype)
        )
        ridge = self._resolve_ridge(dtype) if self.penalized else None
        homotopy = self._initial_homotopy(dtype)
        ineq_weight, ineq_smoothing = self._initial_ineq_state(dtype)
        instances = dict(
            ridge_metric=self.initial_ridge_metric,
            preconditioner=self.initial_preconditioner,
            homotopy=homotopy,
            ineq_weight=ineq_weight,
            ineq_smoothing=ineq_smoothing,
        )
        if not self.cache_jacobian:
            return LMState(damping, ridge=ridge, **instances)
        return LMState(
            damping,
            ridge,
            resid=jnp.zeros(residual_shape.shape, dtype=dtype),
            Jt=jnp.zeros((n, m), dtype=dtype),
            jacobian_valid=jnp.asarray(False, dtype=jnp.bool_),
            aux=jax.tree.map(lambda s: jnp.zeros(s.shape, dtype=s.dtype), aux_shape),
            solver_cache=self.linear_solver.new_cache(
                m, n, self.initial_ridge_metric, dtype, self.penalized
            ),
            **instances,
        )

    def _solve_state(self, x0, args, p, state):
        if state is None:
            if self.cache_jacobian or self.penalized or self.inequality is not None:
                return self.init(x0, args, p=p)
            return LMState(
                jnp.asarray(self.init_damping),
                ridge_metric=None,
                preconditioner=self.initial_preconditioner,
                homotopy=self._initial_homotopy(),
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
        init_homotopy = self._initial_homotopy()
        if init_homotopy is not None and state.homotopy is None:
            state = dataclasses.replace(state, homotopy=init_homotopy)
        init_iw, init_is = self._initial_ineq_state()
        if init_iw is not None and state.ineq_weight is None:
            state = dataclasses.replace(state, ineq_weight=init_iw)
        if init_is is not None and state.ineq_smoothing is None:
            state = dataclasses.replace(state, ineq_smoothing=init_is)
        return self._cold_state(self._resolved_state(state))

    def _initial_info(self, x, state, args, p):
        residual, aux, (_, ineq_metrics) = self._eval_residual_and_aux(
            x,
            args,
            p,
            homotopy=state.homotopy,
            ineq_weight=state.ineq_weight,
            ineq_smoothing=state.ineq_smoothing,
        )
        theta, _ = ravel_pytree(x)
        resolved = self._resolved_state(state)
        self.linear_solver.validate_shape(
            residual.shape[0],
            theta.shape[0],
            penalized=self.penalized,
            ridge_metric=resolved.ridge_metric,
            damped=True,
            role="linear_solver",
        )
        dtype = residual.dtype
        homotopy = (
            None if state.homotopy is None else jnp.asarray(state.homotopy, dtype=dtype)
        )
        resid_loss = jnp.sum(residual**2)
        zero = jnp.zeros((), dtype=dtype)
        one = jnp.ones((), dtype=dtype)
        infinity = jnp.asarray(jnp.inf, dtype=dtype)
        if self.penalized:
            ridge = jnp.asarray(state.ridge, dtype=dtype)
            ctx = SolverContext(
                x=theta, state=resolved, args=args, p=p, ridge=ridge, damping=zero
            )
            penalty_value = jnp.asarray(
                resolved.ridge_metric.quad_form(theta, ctx), dtype=dtype
            )
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

        if ineq_metrics is not None:
            eq_loss, viol_norm, max_viol = ineq_metrics
            ineq_s = (
                None
                if state.ineq_smoothing is None
                else jnp.asarray(state.ineq_smoothing, dtype=dtype)
            )
            ineq_fields = dict(
                eq_loss=eq_loss,
                ineq_violation_norm=viol_norm,
                ineq_max_violation=max_viol,
                inequality_changed=self._ineq_smoothing_active(ineq_s),
            )
        else:
            ineq_fields = {}

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
            homotopy=homotopy,
            complementarity_changed=self._homotopy_active(homotopy),
            **ridge_fields,
            **ineq_fields,
        )

    def _cast_state(self, state, dtype):
        updates = dict(
            damping=jnp.asarray(state.damping, dtype=dtype),
            hyper=_cast_lm_hyper(state.hyper, dtype),
        )
        if state.homotopy is not None:
            updates["homotopy"] = jnp.asarray(state.homotopy, dtype=dtype)
        if state.ineq_weight is not None:
            updates["ineq_weight"] = jnp.asarray(state.ineq_weight, dtype=dtype)
        if state.ineq_smoothing is not None:
            updates["ineq_smoothing"] = jnp.asarray(state.ineq_smoothing, dtype=dtype)
        if self.penalized:
            updates["ridge"] = jnp.asarray(state.ridge, dtype=dtype)
            if state.solver_cache is not None:
                cache = state.solver_cache
                updates["solver_cache"] = dataclasses.replace(
                    cache, ridge=jnp.asarray(cache.ridge, dtype=dtype)
                )
        return dataclasses.replace(state, **updates)

    def update(self, x, state, args=None, p=None):
        """Execute one Levenberg-Marquardt step and return ``(x_new, state, info)``."""
        self._check_residual_args(args, p)
        theta, unravel = ravel_pytree(x)
        (
            residual_flat,
            residual_value,
            resid,
            Jt,
            jvp_fn,
            JT,
            aux,
            base_mask,
            base_ineq,
        ) = self._linearize_step(theta, unravel, state, args, p)
        dtype = resid.dtype
        m, n = resid.shape[0], theta.shape[0]
        resolved_state = self._resolved_state(state)
        self.linear_solver.validate_shape(
            m,
            n,
            penalized=self.penalized,
            ridge_metric=resolved_state.ridge_metric,
            damped=True,
            role="linear_solver",
        )

        hyper = state.hyper if state.hyper is not None else self.hyperparams(dtype)
        damping_decrease = jnp.asarray(hyper.damping_decrease, dtype=dtype)
        damping_increase = jnp.asarray(hyper.damping_increase, dtype=dtype)
        damping_floor = _damping_floor(dtype)
        damping = jnp.maximum(jnp.asarray(state.damping, dtype=dtype), damping_floor)

        space, form = self._solver_space_and_form(
            self.linear_solver, m, n, resolved_state.ridge_metric, damped=True
        )
        ridge_val = (
            jnp.asarray(state.ridge, dtype=dtype)
            if (self.penalized and state.ridge is not None)
            else None
        )
        ctx = SolverContext(
            x=theta,
            state=resolved_state,
            args=args,
            p=p,
            ridge=ridge_val,
            damping=damping,
            space=space,
            role="forward",
            form=form,
        )
        (
            ridge,
            penalty_value_old,
            penalty_grad,
            resid_loss_old,
            loss_old,
        ) = self._base_objective(resid, theta, state, ctx)

        step_solver = self.linear_solver.prepare(
            Subproblem(
                resid=resid,
                theta=theta,
                Jt=Jt,
                jvp_fn=jvp_fn,
                JT=JT,
                ridge=ridge,
                damping=damping,
                ridge_metric=resolved_state.ridge_metric,
                penalty_grad=penalty_grad,
                cache=state.solver_cache,
                cache_enabled=self.cache_jacobian,
                hyper=hyper,
                ctx=ctx,
                penalized=self.penalized,
                undamped=False,
                always_damped=True,
                active=True,
                preconditioner=resolved_state.preconditioner,
            )
        )
        grad = step_solver.grad
        velocity = jnp.asarray(step_solver.velocity(), dtype=dtype)
        zero = jnp.zeros((), dtype=dtype)
        one = jnp.ones((), dtype=dtype)
        slope = jnp.sum(grad * velocity)
        descent_ok = (
            step_solver.solve_ok
            & jnp.isfinite(loss_old)
            & jnp.all(jnp.isfinite(velocity))
            & jnp.isfinite(slope)
            & (slope < zero)
        )

        if self.geodesic_acceleration:
            geodesic_acceptance_ratio = jnp.asarray(
                hyper.geodesic_acceptance_ratio, dtype=dtype
            )
            (
                acceleration,
                acceleration_ratio,
                raw_ratio_accepted,
            ) = self._geodesic_correction(
                residual_flat,
                theta,
                velocity,
                step_solver.solve_residual_rhs,
                geodesic_acceptance_ratio,
                dtype,
            )
            ratio_accepted = (
                descent_ok & raw_ratio_accepted & jnp.all(jnp.isfinite(acceleration))
            )
            a = jnp.where(ratio_accepted, acceleration, jnp.zeros_like(velocity))
        else:
            a = None
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
            ridge=ridge if self.penalized else None,
            ridge_metric=resolved_state.ridge_metric if self.penalized else None,
            ctx=ctx,
            resid_loss_old=resid_loss_old if self.penalized else None,
            penalty_old=penalty_value_old if self.penalized else None,
        )

        grad_norm = jnp.linalg.norm(grad)
        (
            loss,
            resid_loss_new,
            new_homotopy,
            new_ineq_smoothing,
            comp_changed,
            ineq_changed,
            schedule_changed,
            out_ineq,
        ) = self._post_step_constraints(
            theta_new,
            unravel,
            args,
            p,
            state,
            improved,
            loss,
            resid_loss_new,
            penalty_new,
            ridge,
            grad_norm,
            base_mask,
            base_ineq,
        )

        safe_alpha = jnp.where(improved, stepsize, one)
        damping_factor = jnp.where(
            improved,
            jnp.minimum(damping_increase, damping_decrease / safe_alpha),
            jnp.where(schedule_changed, one, damping_increase),
        )
        new_damping = jnp.maximum(damping * damping_factor, damping_floor)
        used_geodesic = improved & ratio_accepted
        step_alpha = jnp.where(
            improved,
            stepsize,
            jnp.where(trial_stepsize > zero, trial_stepsize, one),
        )
        step_vec = (
            step_alpha * velocity
            if a is None
            else step_alpha * velocity + 0.5 * (step_alpha**2) * a
        )

        ineq_weight_out = (
            None
            if state.ineq_weight is None
            else jnp.asarray(state.ineq_weight, dtype=dtype)
        )
        instances = dict(
            ridge_metric=state.ridge_metric,
            preconditioner=state.preconditioner,
            homotopy=new_homotopy,
            ineq_weight=ineq_weight_out,
            ineq_smoothing=new_ineq_smoothing,
        )
        ridge_state = ridge if self.penalized else None
        cache_valid = (~improved) & (~schedule_changed) & step_solver.solve_ok
        if self.cache_jacobian and self.linear_solver.materializes_jacobian:
            new_state = LMState(
                new_damping,
                ridge=ridge_state,
                resid=resid,
                Jt=Jt,
                jacobian_valid=cache_valid,
                aux=aux,
                hyper=state.hyper,
                solver_cache=step_solver.make_cache(cache_valid),
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
                penalty_grad_norm=jnp.linalg.norm(penalty_grad),
            )
        else:
            ridge_info = {}

        if out_ineq is not None:
            eq_loss, viol_norm, max_viol = out_ineq
            ineq_info = dict(
                eq_loss=eq_loss,
                ineq_violation_norm=viol_norm,
                ineq_max_violation=max_viol,
                inequality_changed=ineq_changed,
            )
        else:
            ineq_info = {}

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
                grad_norm=grad_norm,
                step_norm=jnp.linalg.norm(step_vec),
                aux=aux,
                stepsize=stepsize,
                num_linesearch_steps=trials,
                homotopy=new_homotopy,
                complementarity_changed=comp_changed,
                **ridge_info,
                **ineq_info,
            ),
        )
