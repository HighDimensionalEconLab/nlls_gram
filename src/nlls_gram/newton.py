import dataclasses

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from nlls_gram.linear_solvers import (
    CG,
    LU,
    Cholesky,
    Subproblem,
)
from nlls_gram.solver_base import (
    SolverBase,
    _armijo_backtracking,
    _init_residual_and_aux,
)
from nlls_gram.types import (
    NewtonHyperparams,
    NewtonInfo,
    NewtonState,
    SolverContext,
    SolverStatus,
    _cast_newton_hyper,
)

__all__ = ["Newton"]


class Newton(SolverBase):
    """Euclidean-step Newton / Gauss-Newton solver with Armijo backtracking."""

    _forward_role_attr = "supports_newton"

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
        c1=1e-4,
        decrease_factor=0.5,
        maxls=30,
        max_stepsize=1.0,
        linesearch_tol=0.0,
        damping=None,
        damping_power=1.5,
        min_damping=1e-10,
        has_aux=False,
        cache_jacobian=False,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
        complementarity=None,
        inequality=None,
    ):
        if damping is None:
            damping = (
                0.0 if (ridge is not None or isinstance(linear_solver, LU)) else 1e-3
            )
        if damping > 0.0 and isinstance(linear_solver, LU):
            raise ValueError(
                "linear_solver=LU() does not support damping > 0; pass damping=0.0 "
                "or use Cholesky(), QR(), SVD(), CG(), or GramCG()"
            )
        precond = getattr(linear_solver, "preconditioner", None)
        if (
            damping == 0.0
            and precond is not None
            and getattr(precond, "requires_positive_damping", False)
        ):
            raise ValueError(
                "linear_solver.preconditioner requires positive damping and "
                "cannot serve an undamped Newton system"
            )
        if isinstance(linear_solver, CG) and linear_solver.penalty is not None:
            raise ValueError(
                "CG(penalty=...) is supported only in ad_solver=CG(...), not in "
                "Newton's forward linear_solver role"
            )

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
            geodesic_acceleration=bool(geodesic_acceleration),
            geodesic_acceptance_ratio=geodesic_acceptance_ratio,
            complementarity=complementarity,
            inequality=inequality,
        )
        self.c1 = c1
        self.decrease_factor = decrease_factor
        self.maxls = maxls
        self.max_stepsize = max_stepsize
        self.linesearch_tol = linesearch_tol
        self.damping = damping
        self.damping_power = damping_power
        self.min_damping = min_damping
        self._damped = bool(self.damping > 0)

        self._static_key = self._common_static_key() + (self._damped,)
        self._static_hash = hash(self._static_key)
        self._sealed = True

    def hyperparams(self, dtype=None):
        """Return ``NewtonHyperparams`` built from constructor settings."""
        eff_dtype = jnp.result_type(float) if dtype is None else dtype
        iterative_tol = self.iterative_tol
        if iterative_tol is None:
            if dtype is None:
                iterative_tol = jnp.nan
            else:
                iterative_tol = 1e-10 if jnp.finfo(eff_dtype).bits > 32 else 1e-6
        return NewtonHyperparams(
            jnp.asarray(self.c1, dtype=eff_dtype),
            jnp.asarray(self.decrease_factor, dtype=eff_dtype),
            jnp.asarray(self.maxls, dtype=jnp.int32),
            jnp.asarray(self.max_stepsize, dtype=eff_dtype),
            jnp.asarray(self.linesearch_tol, dtype=eff_dtype),
            jnp.asarray(iterative_tol, dtype=eff_dtype),
            jnp.asarray(self.iterative_atol, dtype=eff_dtype),
            None
            if self.iterative_maxiter is None
            else jnp.asarray(self.iterative_maxiter, dtype=jnp.int32),
            jnp.asarray(self.damping, dtype=eff_dtype),
            jnp.asarray(self.damping_power, dtype=eff_dtype),
            jnp.asarray(self.min_damping, dtype=eff_dtype),
            jnp.asarray(self.geodesic_acceptance_ratio, dtype=eff_dtype),
        )

    def init(self, x0, args=None, *, p=None):
        """Build initial ``NewtonState`` at ``x0`` without evaluating the residual."""
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
            damped=self._damped,
            role="linear_solver",
        )
        dtype = residual_shape.dtype
        ridge = self._resolve_ridge(dtype) if self.penalized else None
        homotopy = self._initial_homotopy(dtype)
        ineq_weight, ineq_smoothing = self._initial_ineq_state(dtype)
        instances = dict(
            ridge=ridge,
            ridge_metric=self.initial_ridge_metric,
            preconditioner=self.initial_preconditioner,
            homotopy=homotopy,
            ineq_weight=ineq_weight,
            ineq_smoothing=ineq_smoothing,
        )
        if not self.cache_jacobian:
            return NewtonState(**instances)
        return NewtonState(
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
        x_leaves = jax.tree.leaves(x0)
        x_dtype = jnp.result_type(*x_leaves) if x_leaves else jnp.result_type(float)
        if state is None:
            resolved = self.init(x0, args, p=p)
        else:
            if not isinstance(state, NewtonState):
                raise TypeError(
                    f"state must be a NewtonState; got {type(state).__name__}"
                )
            if self.penalized:
                state = self._validate_supplied_ridge(state, x_dtype)
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
            if self.cache_jacobian and state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True requires a NewtonState initialized with "
                    "solver.init(x0, args, p=p)"
                )
            resolved = self._cold_state(self._resolved_state(state))
        if resolved.hyper is None:
            resolved = dataclasses.replace(resolved, hyper=self.hyperparams(x_dtype))
        return self._cast_state(resolved, x_dtype)

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
            damped=self._damped,
            role="linear_solver",
        )
        dtype = residual.dtype
        homotopy = (
            None if state.homotopy is None else jnp.asarray(state.homotopy, dtype=dtype)
        )
        ineq_s = (
            None
            if state.ineq_smoothing is None
            else jnp.asarray(state.ineq_smoothing, dtype=dtype)
        )
        homotopy_active = self._homotopy_active(homotopy)
        ineq_active = self._ineq_smoothing_active(ineq_s)
        zero = jnp.zeros((), dtype=dtype)
        infinity = jnp.asarray(jnp.inf, dtype=dtype)
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        resid_loss = jnp.sum(residual**2)
        theta_finite = jnp.all(jnp.isfinite(theta))
        resid_finite = jnp.all(jnp.isfinite(residual))
        if self.penalized:
            ridge = jnp.asarray(state.ridge, dtype=dtype)
            ctx = SolverContext(
                x=theta, state=resolved, args=args, p=p, ridge=ridge, damping=zero
            )
            penalty_value = jnp.asarray(
                resolved.ridge_metric.quad_form(theta, ctx), dtype=dtype
            )
            raw_loss = resid_loss + ridge * penalty_value
            base_finite = theta_finite & resid_finite & jnp.isfinite(raw_loss)
            loss = jnp.where(base_finite, raw_loss, nan)
            step_status = jnp.where(
                ~base_finite,
                jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32),
                jnp.asarray(SolverStatus.RUNNING, dtype=jnp.int32),
            )
            grad_norm = infinity
            ridge_fields = dict(
                ridge=ridge,
                resid_loss=jnp.where(base_finite, resid_loss, nan),
                penalty_value=jnp.where(base_finite, penalty_value, nan),
                penalty_grad_norm=infinity,
            )
        else:
            base_finite = theta_finite & resid_finite & jnp.isfinite(resid_loss)
            loss = jnp.where(base_finite, resid_loss, nan)
            exact_root = (
                base_finite
                & (~homotopy_active)
                & (~ineq_active)
                & jnp.all(residual == 0)
            )
            step_status = jnp.where(
                ~base_finite,
                jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32),
                jnp.where(
                    exact_root,
                    jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32),
                    jnp.asarray(SolverStatus.RUNNING, dtype=jnp.int32),
                ),
            )
            grad_norm = jnp.where(exact_root, zero, infinity)
            ridge_fields = {}

        if ineq_metrics is not None:
            eq_loss, viol_norm, max_viol = ineq_metrics
            ineq_fields = dict(
                eq_loss=jnp.where(base_finite, eq_loss, nan),
                ineq_violation_norm=jnp.where(base_finite, viol_norm, nan),
                ineq_max_violation=jnp.where(base_finite, max_viol, nan),
                inequality_changed=ineq_active,
            )
        else:
            ineq_fields = {}

        return NewtonInfo(
            loss=loss,
            loss_old=loss,
            loss_candidate=loss,
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            stepsize=zero,
            trial_stepsize=zero,
            num_linesearch_steps=jnp.asarray(0, dtype=jnp.int32),
            grad_norm=grad_norm,
            direction_norm=zero,
            step_norm=zero,
            directional_derivative=zero,
            step_status=step_status,
            used_geodesic=jnp.asarray(False, dtype=jnp.bool_),
            acceleration_ratio=zero,
            aux=aux,
            homotopy=homotopy,
            complementarity_changed=homotopy_active,
            **ridge_fields,
            **ineq_fields,
        )

    def _cast_state(self, state, dtype):
        updates = dict(hyper=_cast_newton_hyper(state.hyper, dtype))
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
        """Execute one Newton + Armijo step and return ``(x_new, state, info)``."""
        self._check_residual_args(args, p)
        if not isinstance(state, NewtonState):
            raise TypeError(f"state must be a NewtonState; got {type(state).__name__}")
        if self.penalized and state.ridge is None:
            raise ValueError(
                "the state has no ridge; create it with init(x, args, p=p)"
            )
        if self.cache_jacobian and state.jacobian_valid is None:
            raise ValueError(
                "cache_jacobian=True requires a NewtonState initialized with "
                "solver.init(x0, args, p=p)"
            )
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
        m, n = resid.shape[0], theta.shape[0]
        resolved_state = self._resolved_state(state)
        self.linear_solver.validate_shape(
            m,
            n,
            penalized=self.penalized,
            ridge_metric=resolved_state.ridge_metric,
            damped=self._damped,
            role="linear_solver",
        )
        dtype = resid.dtype
        zero = jnp.zeros((), dtype=dtype)
        nan = jnp.asarray(jnp.nan, dtype=dtype)

        hyper = state.hyper if state.hyper is not None else self.hyperparams(dtype)
        resid_loss_raw = jnp.sum(resid**2)
        has_traced_damping = state.hyper is not None or self._damped
        if not has_traced_damping:
            damping = zero
        else:
            damping_scale = jnp.asarray(hyper.damping, dtype=dtype)
            damping_power = jnp.asarray(hyper.damping_power, dtype=dtype)
            min_damping = jnp.asarray(hyper.min_damping, dtype=dtype)
            r_norm = jnp.sqrt(resid_loss_raw)

            def _compute_damping(_):
                j_scale = (
                    jnp.sum(Jt**2) / m if Jt is not None else jnp.ones((), dtype=dtype)
                )
                floor = min_damping
                if isinstance(self.linear_solver, Cholesky):
                    floor = jnp.maximum(floor, 4.0 * jnp.finfo(dtype).eps * j_scale)
                return jnp.maximum(
                    floor, damping_scale * j_scale * (r_norm**damping_power)
                )

            damping = jax.lax.cond(
                damping_scale > zero,
                _compute_damping,
                lambda _: zero,
                operand=None,
            )

        space, form = self._solver_space_and_form(
            self.linear_solver,
            m,
            n,
            resolved_state.ridge_metric,
            damped=has_traced_damping,
        )
        ridge_val = jnp.asarray(state.ridge, dtype=dtype) if self.penalized else None
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
            raw_loss_old,
        ) = self._base_objective(resid, theta, state, ctx)
        if self.penalized:
            penalty_grad_norm = jnp.linalg.norm(penalty_grad)
            base_finite = (
                jnp.all(jnp.isfinite(theta))
                & jnp.all(jnp.isfinite(resid))
                & jnp.all(jnp.isfinite(penalty_grad))
                & jnp.isfinite(raw_loss_old)
            )
            exact_root = jnp.asarray(False, dtype=jnp.bool_)
        else:
            penalty_value_old = zero
            base_finite = (
                jnp.all(jnp.isfinite(theta))
                & jnp.all(jnp.isfinite(resid))
                & jnp.isfinite(raw_loss_old)
            )
            exact_root = (
                base_finite
                & (~self._homotopy_active(state.homotopy))
                & (~self._ineq_smoothing_active(state.ineq_smoothing))
                & jnp.all(resid == 0)
            )
        loss_old = jnp.where(base_finite, raw_loss_old, nan)

        sub = Subproblem(
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
            undamped=not has_traced_damping,
            active=True,
            preconditioner=resolved_state.preconditioner,
        )

        def _compute_step(_):
            newton_step = self.linear_solver.prepare(sub)
            grad = newton_step.grad
            d = jnp.asarray(newton_step.velocity(), dtype=dtype)
            grad_norm = jnp.linalg.norm(grad)
            direction_norm = jnp.linalg.norm(d)
            slope = jnp.sum(grad * d)
            solve_ok = (
                newton_step.solve_ok
                & jnp.all(jnp.isfinite(d))
                & jnp.isfinite(grad_norm)
                & jnp.isfinite(direction_norm)
                & jnp.isfinite(slope)
            )
            descent_ok = solve_ok & (slope < zero)
            if self.geodesic_acceleration:
                geodesic_acceptance_ratio = jnp.asarray(
                    hyper.geodesic_acceptance_ratio, dtype=dtype
                )
                (
                    a_raw,
                    acceleration_ratio,
                    raw_ratio_accepted,
                ) = self._geodesic_correction(
                    residual_flat,
                    theta,
                    d,
                    newton_step.solve_residual_rhs,
                    geodesic_acceptance_ratio,
                    dtype,
                )
                ratio_accepted = (
                    descent_ok & raw_ratio_accepted & jnp.all(jnp.isfinite(a_raw))
                )
                a = jnp.where(ratio_accepted, a_raw, jnp.zeros_like(d))
            else:
                a = None
                acceleration_ratio = zero
                ratio_accepted = jnp.asarray(False, dtype=jnp.bool_)

            def _run_ls(_):
                (
                    theta_ls,
                    loss_ls,
                    resid_loss_ls,
                    penalty_ls,
                    loss_cand_ls,
                    accepted_ls,
                    stepsize_ls,
                    trial_stepsize_ls,
                    trials_ls,
                    status_ls,
                ) = _armijo_backtracking(
                    residual_value,
                    theta,
                    loss_old,
                    d,
                    slope,
                    hyper,
                    descent_ok,
                    a=a,
                    ridge=ridge if self.penalized else None,
                    ridge_metric=resolved_state.ridge_metric
                    if self.penalized
                    else None,
                    ctx=ctx,
                    resid_loss_old=resid_loss_old if self.penalized else None,
                    penalty_old=penalty_value_old if self.penalized else None,
                    penalty_grad=penalty_grad if self.penalized else None,
                )
                if self.geodesic_acceleration:
                    step_d_norm = jnp.linalg.norm(
                        stepsize_ls * d + 0.5 * (stepsize_ls**2) * a
                    )
                else:
                    step_d_norm = stepsize_ls * direction_norm
                step_norm_ls = jnp.where(accepted_ls, step_d_norm, zero)
                return (
                    theta_ls,
                    loss_ls,
                    resid_loss_ls,
                    penalty_ls,
                    loss_cand_ls,
                    accepted_ls,
                    stepsize_ls,
                    trial_stepsize_ls,
                    trials_ls,
                    step_norm_ls,
                    status_ls,
                )

            def _skip_ls(_):
                fail_status = jnp.where(
                    solve_ok,
                    jnp.asarray(SolverStatus.NON_DESCENT, dtype=jnp.int32),
                    jnp.asarray(SolverStatus.LINEAR_SOLVE_FAILED, dtype=jnp.int32),
                )
                return (
                    theta,
                    loss_old,
                    resid_loss_old,
                    penalty_value_old,
                    loss_old,
                    jnp.asarray(False, dtype=jnp.bool_),
                    zero,
                    zero,
                    jnp.asarray(0, dtype=jnp.int32),
                    zero,
                    fail_status,
                )

            (
                theta_out,
                loss_out,
                resid_loss_out,
                penalty_out,
                loss_cand_out,
                accepted_out,
                stepsize_out,
                trial_stepsize_out,
                trials_out,
                step_norm_out,
                status_out,
            ) = jax.lax.cond(descent_ok, _run_ls, _skip_ls, operand=None)
            cache_valid = (~accepted_out) & solve_ok
            used_geodesic = accepted_out & ratio_accepted
            return (
                theta_out,
                loss_out,
                resid_loss_out,
                penalty_out,
                loss_cand_out,
                accepted_out,
                stepsize_out,
                trial_stepsize_out,
                trials_out,
                grad_norm,
                direction_norm,
                step_norm_out,
                slope,
                status_out,
                used_geodesic,
                acceleration_ratio,
                cache_valid,
                newton_step.make_cache(cache_valid) if self.cache_jacobian else None,
            )

        def _skip_step(_):
            status_out = jnp.where(
                exact_root,
                jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32),
                jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32),
            )
            scalar_val = jnp.where(exact_root, zero, nan)
            solver_cache = state.solver_cache
            if solver_cache is not None:
                solver_cache = dataclasses.replace(
                    solver_cache, valid=jnp.asarray(False, dtype=jnp.bool_)
                )
            return (
                theta,
                scalar_val,
                scalar_val,
                scalar_val,
                scalar_val,
                jnp.asarray(False, dtype=jnp.bool_),
                zero,
                zero,
                jnp.asarray(0, dtype=jnp.int32),
                scalar_val,
                zero,
                zero,
                scalar_val,
                status_out,
                jnp.asarray(False, dtype=jnp.bool_),
                zero,
                jnp.asarray(False, dtype=jnp.bool_),
                solver_cache,
            )

        (
            theta_new,
            loss,
            resid_loss,
            penalty_value,
            loss_candidate,
            accepted,
            stepsize,
            trial_stepsize,
            num_linesearch_steps,
            grad_norm,
            direction_norm,
            step_norm,
            directional_derivative,
            step_status,
            used_geodesic,
            acceleration_ratio,
            cache_valid,
            solver_cache,
        ) = jax.lax.cond(
            base_finite & (~exact_root),
            _compute_step,
            _skip_step,
            operand=None,
        )

        (
            loss,
            resid_loss,
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
            accepted,
            loss,
            resid_loss,
            penalty_value,
            ridge,
            grad_norm,
            base_mask,
            base_ineq,
        )
        step_status = jnp.where(
            schedule_changed
            & jnp.isfinite(loss)
            & (step_status != SolverStatus.LINEAR_SOLVE_FAILED),
            jnp.asarray(SolverStatus.RUNNING, dtype=jnp.int32),
            step_status,
        )
        cache_valid = cache_valid & (~schedule_changed)
        if solver_cache is not None:
            solver_cache = dataclasses.replace(
                solver_cache, valid=solver_cache.valid & (~schedule_changed)
            )

        ineq_weight_out = (
            None
            if state.ineq_weight is None
            else jnp.asarray(state.ineq_weight, dtype=dtype)
        )
        instances = dict(
            ridge=ridge if self.penalized else None,
            ridge_metric=state.ridge_metric,
            preconditioner=state.preconditioner,
            homotopy=new_homotopy,
            ineq_weight=ineq_weight_out,
            ineq_smoothing=new_ineq_smoothing,
        )
        if self.cache_jacobian:
            new_state = NewtonState(
                resid=resid,
                Jt=Jt,
                jacobian_valid=cache_valid,
                aux=aux,
                hyper=state.hyper,
                solver_cache=solver_cache,
                **instances,
            )
        else:
            new_state = NewtonState(hyper=state.hyper, **instances)

        if self.penalized:
            ridge_info = dict(
                ridge=ridge,
                resid_loss=resid_loss,
                penalty_value=penalty_value,
                penalty_grad_norm=jnp.where(base_finite, penalty_grad_norm, nan),
            )
        else:
            ridge_info = {}

        if out_ineq is not None:
            eq_loss, viol_norm, max_viol = out_ineq
            ineq_info = dict(
                eq_loss=jnp.where(base_finite, eq_loss, nan),
                ineq_violation_norm=jnp.where(base_finite, viol_norm, nan),
                ineq_max_violation=jnp.where(base_finite, max_viol, nan),
                inequality_changed=ineq_changed,
            )
        else:
            ineq_info = {}

        return (
            unravel(theta_new),
            new_state,
            NewtonInfo(
                loss=loss,
                loss_old=loss_old,
                loss_candidate=loss_candidate,
                accepted=accepted,
                stepsize=stepsize,
                trial_stepsize=trial_stepsize,
                num_linesearch_steps=num_linesearch_steps,
                grad_norm=grad_norm,
                direction_norm=direction_norm,
                step_norm=step_norm,
                directional_derivative=directional_derivative,
                step_status=step_status,
                used_geodesic=used_geodesic,
                acceleration_ratio=acceleration_ratio,
                aux=aux,
                homotopy=new_homotopy,
                complementarity_changed=comp_changed,
                **ridge_info,
                **ineq_info,
            ),
        )

    def _converged(self, info, atol, gtol, xtol):
        homotopy_active = self._homotopy_active(info.homotopy)
        ineq_decay_active = (
            jnp.asarray(False, dtype=jnp.bool_)
            if info.inequality_changed is None
            else info.inequality_changed
        )
        ready = (
            self._constraints_stable(info) & (~homotopy_active) & (~ineq_decay_active)
        )
        step_ok = info.step_status == SolverStatus.RUNNING
        stationarity_eligible = (info.step_status != SolverStatus.NONFINITE) & (
            info.step_status != SolverStatus.LINEAR_SOLVE_FAILED
        )
        gtol_met = ready & stationarity_eligible & (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (
            ready & step_ok & (xtol > 0) & info.accepted & (info.step_norm < xtol)
        )
        if self.penalized:
            return gtol_met | xtol_met
        exact_root = (
            (~homotopy_active)
            & (~ineq_decay_active)
            & (info.step_status == SolverStatus.CONVERGED)
        )
        atol_met = ready & step_ok & (atol > 0) & (jnp.sqrt(info.loss) < atol)
        return exact_root | atol_met | gtol_met | xtol_met

    def _step_failure_status(self, info):
        ok = (info.step_status == SolverStatus.RUNNING) | (
            info.step_status == SolverStatus.CONVERGED
        )
        return jnp.where(
            ~jnp.isfinite(info.loss),
            jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                ok,
                jnp.asarray(SolverStatus.RUNNING, dtype=jnp.int32),
                info.step_status,
            ),
        )
