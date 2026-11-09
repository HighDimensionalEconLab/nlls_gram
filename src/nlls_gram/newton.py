"""Undamped (or residual-coupled) Newton / Gauss-Newton with Armijo backtracking
for nonlinear least squares over a JAX pytree ``x``, supporting both unpenalized
geometry-selected interpolation (``ridge=None``) and Tikhonov-penalized ridge
optimization (``ridge > 0``).

At each outer step, computes the minimum-metric-norm Gauss-Newton direction
(or penalized ridge Gauss-Newton direction) once, optionally adds the
second-order geodesic correction along a parabolic trial curve, and searches
with a residual-only Armijo backtracking loop against the half-objective merit
function using the actual directional derivative ``s = g'u``.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from nlls_gram.linear_solvers import (
    CG,
    LU,
    Cholesky,
    GramCG,
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
from nlls_gram.utilities import mm

__all__ = ["Newton"]


class Newton(SolverBase):
    """Newton / Gauss-Newton with Armijo backtracking over a JAX pytree ``x``.

    Supports two root-selection regimes controlled by ``ridge``:

    - **Unpenalized (``ridge=None``, default)**: minimizes ``||r(x, args, p)||^2``
      with optional :class:`~nlls_gram.Metric` ``W = F'F`` selecting
      minimum-``W``-norm Gauss-Newton directions ``u = -B^+ r`` in whitened
      coordinates ``u = F_bar d`` (``B = J F_bar^{-1}``). Supports
      :class:`~nlls_gram.Cholesky` (default), :class:`~nlls_gram.QR`,
      :class:`~nlls_gram.SVD`, :class:`~nlls_gram.LU` (square systems),
      :class:`~nlls_gram.CG`, and :class:`~nlls_gram.GramCG`.
    - **Penalized ridge (``ridge > 0``)**: minimizes the Tikhonov objective
      ``||r(x, args, p)||^2 + ridge * ||x_m||_W^2`` (defaulting ``metric`` to
      the Euclidean identity over all coordinates when ``metric=None``), with
      ``ridge`` carried in :attr:`NewtonState.ridge` for continuation schedules
      (:class:`~nlls_gram.AnnealRidge`). Supports :class:`~nlls_gram.Cholesky`,
      :class:`~nlls_gram.QR`, and :class:`~nlls_gram.CG`.

    Line-search defaults: ``c1=1e-4``, ``decrease_factor=0.5``, ``maxls=30``,
    ``max_stepsize=1.0``, and ``linesearch_tol=0.0``. Every outer iteration
    resets the initial trial step size to ``min(1.0, max_stepsize)``.

    When ``damping > 0`` (default ``1e-3`` when ``ridge=None`` and ``0.0`` when
    ``ridge`` is active or ``linear_solver=LU()``), applies stateless
    parameter-scale-invariant residual-coupled step damping to the linear solve
    at each outer step before Armijo backtracking:
    ``max(min_damping, damping * (||B||_F^2 / m) * ||r||_2**damping_power)``
    for dense solvers (or ``max(min_damping, damping * ||r||_2**damping_power)``
    for matrix-free Krylov solvers).
    """

    _forward_role_attr = "supports_newton"

    def __init__(
        self,
        residual_fn,
        *,
        metric=None,
        ridge=None,
        c1=1e-4,
        decrease_factor=0.5,
        maxls=30,
        max_stepsize=1.0,
        linesearch_tol=0.0,
        damping=None,
        damping_power=1.5,
        min_damping=1e-10,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
        linear_solver=Cholesky(),  # noqa: B008 -- frozen, immutable default
        jacobian_mode="auto",
        jacobian_chunk=None,
        ad_solver=None,
        has_aux=False,
        cache_jacobian=False,
        complementarity=None,
    ):
        if damping is None:
            damping = (
                0.0 if (ridge is not None or isinstance(linear_solver, LU)) else 1e-3
            )
        if not (0.0 < c1 < 1.0):
            raise ValueError("c1 must satisfy 0 < c1 < 1")
        if not (0.0 < decrease_factor < 1.0):
            raise ValueError("decrease_factor must satisfy 0 < decrease_factor < 1")
        if isinstance(maxls, bool) or not isinstance(maxls, int) or maxls <= 0:
            raise ValueError("maxls must be a positive int")
        if max_stepsize <= 0.0:
            raise ValueError("max_stepsize must be positive")
        if linesearch_tol < 0.0:
            raise ValueError("linesearch_tol must be nonnegative")
        if damping < 0.0:
            raise ValueError("damping must be nonnegative")
        if damping_power < 0.0:
            raise ValueError("damping_power must be nonnegative")
        if min_damping <= 0.0:
            raise ValueError("min_damping must be positive")
        if damping > 0.0 and isinstance(linear_solver, LU):
            raise ValueError(
                "linear_solver=LU() does not support damping > 0; pass damping=0.0 "
                "or use Cholesky(), QR(), SVD(), CG(), or GramCG()"
            )
        if isinstance(linear_solver, (CG, GramCG)):
            if (
                linear_solver.preconditioner is not None
                and linear_solver.preconditioner.requires_positive_damping
            ):
                raise ValueError(
                    "this preconditioner divides by the live damping and "
                    "cannot serve in Newton (the Newton system is undamped)"
                )
            if getattr(linear_solver, "penalty", None) is not None:
                raise ValueError(
                    "CG(penalty=...) is not supported in Newton's forward "
                    "linear_solver role: the forward step must remain undamped. "
                    "Use penalty only in ad_solver=CG(...)"
                )

        self._init_common(
            residual_fn,
            linear_solver=linear_solver,
            metric=metric,
            ridge=ridge,
            has_aux=has_aux,
            cache_jacobian=cache_jacobian,
            geodesic_acceleration=bool(geodesic_acceleration),
            geodesic_acceptance_ratio=geodesic_acceptance_ratio,
            ad_solver=ad_solver,
            jacobian_mode=jacobian_mode,
            jacobian_chunk=jacobian_chunk,
            complementarity=complementarity,
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
        """``NewtonHyperparams`` built from the constructor values."""
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
        """Build the initial :class:`~nlls_gram.NewtonState` at ``x0``.

        Uses ``jax.eval_shape`` to validate shapes and size optional Jacobian
        cache buffers without paying an extra device residual evaluation.
        """
        self._check_residual_args(args, p)
        residual_shape, aux_shape = _init_residual_and_aux.eval_shape(self, x0, args, p)
        theta, _ = ravel_pytree(x0)
        m, n = residual_shape.size, theta.size
        n_m, _ = self._block_sizes(n)
        self.linear_solver._validate_newton_shape(
            m + n_m if self.penalized else m, n, damped=self._damped
        )
        dtype = residual_shape.dtype
        ridge = self._resolve_ridge(dtype) if self.penalized else None
        homotopy = self._initial_homotopy(dtype)
        instances = dict(
            ridge=ridge,
            metric=self.initial_metric,
            preconditioner=self.initial_preconditioner,
            homotopy=homotopy,
        )
        if not self.cache_jacobian:
            return NewtonState(**instances)
        return NewtonState(
            resid=jnp.zeros(residual_shape.shape, dtype=dtype),
            Jt=jnp.zeros((n, m), dtype=dtype),
            jacobian_valid=jnp.asarray(False, dtype=jnp.bool_),
            aux=jax.tree.map(lambda s: jnp.zeros(s.shape, dtype=s.dtype), aux_shape),
            solver_cache=self.linear_solver.new_cache(m, n, n_m, dtype, self.penalized),
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
        residual, aux = self._residual_and_aux(x, args, p, homotopy=state.homotopy)
        theta, _ = ravel_pytree(x)
        n_m, _ = self._block_sizes(theta.shape[0])
        self.linear_solver._validate_newton_shape(
            residual.shape[0] + n_m if self.penalized else residual.shape[0],
            theta.shape[0],
            damped=self._damped,
        )
        dtype = residual.dtype
        homotopy = (
            None if state.homotopy is None else jnp.asarray(state.homotopy, dtype=dtype)
        )
        homotopy_active = self._homotopy_active(homotopy)
        zero = jnp.zeros((), dtype=dtype)
        infinity = jnp.asarray(jnp.inf, dtype=dtype)
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        resid_loss = jnp.sum(residual**2)
        theta_finite = jnp.all(jnp.isfinite(theta))
        resid_finite = jnp.all(jnp.isfinite(residual))
        if self.penalized:
            ridge = jnp.asarray(state.ridge, dtype=dtype)
            resolved = self._resolved_state(state)
            ctx = SolverContext(x=theta, state=resolved, args=args, p=p)
            y_m = self._factor_apply_metric_block(theta[:n_m], ctx, dtype)
            penalty_value = jnp.sum(y_m**2)
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
            exact_root = base_finite & (~homotopy_active) & jnp.all(residual == 0)
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
        )

    def _cast_state(self, state, dtype):
        updates = dict(hyper=_cast_newton_hyper(state.hyper, dtype))
        if state.homotopy is not None:
            updates["homotopy"] = jnp.asarray(state.homotopy, dtype=dtype)
        if self.penalized:
            updates["ridge"] = jnp.asarray(state.ridge, dtype=dtype)
            if state.solver_cache is not None:
                cache = state.solver_cache
                updates["solver_cache"] = dataclasses.replace(
                    cache, ridge=jnp.asarray(cache.ridge, dtype=dtype)
                )
        return dataclasses.replace(state, **updates)

    def update(self, x, state, args=None, p=None):
        """One Newton + Armijo step: returns ``(x_new, state, info)``."""
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
        n_m, n_f = self._block_sizes(theta.shape[0])
        residual_flat, residual_value, resid, Jt, jvp_fn, JT, aux, base_mask = (
            self._linearize_step(theta, unravel, state, args, p)
        )
        self.linear_solver._validate_newton_shape(
            resid.shape[0] + n_m if self.penalized else resid.shape[0],
            theta.shape[0],
            damped=self._damped,
        )
        dtype = resid.dtype
        zero = jnp.zeros((), dtype=dtype)
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        resolved_state = self._resolved_state(state)
        ctx = SolverContext(x=theta, state=resolved_state, args=args, p=p)
        (
            ridge,
            y_m,
            penalty_value_old,
            penalty_gradient,
            resid_loss_old,
            raw_loss_old,
        ) = self._base_objective(resid, theta, n_m, n_f, state, ctx)
        if self.penalized:
            penalty_grad_norm = jnp.linalg.norm(penalty_gradient)
            base_finite = (
                jnp.all(jnp.isfinite(theta))
                & jnp.all(jnp.isfinite(resid))
                & jnp.all(jnp.isfinite(y_m))
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
                & jnp.all(resid == 0)
            )
        loss_old = jnp.where(base_finite, raw_loss_old, nan)

        hyper = state.hyper if state.hyper is not None else self.hyperparams(dtype)

        def whiten(v):
            return jnp.asarray(self._extended_solve(v, ctx), dtype=dtype)

        def whiten_transpose(v):
            return jnp.asarray(self._extended_solve_transpose(v, ctx), dtype=dtype)

        if state.hyper is None and not self._damped:
            damping = zero
        else:
            damping_scale = jnp.asarray(hyper.damping, dtype=dtype)
            damping_power = jnp.asarray(hyper.damping_power, dtype=dtype)
            min_damping = jnp.asarray(hyper.min_damping, dtype=dtype)
            r_norm = jnp.sqrt(resid_loss_old)

            def _compute_damping(_):
                j_scale = (
                    jnp.sum(whiten_transpose(Jt) ** 2) / resid.shape[0]
                    if Jt is not None
                    else jnp.ones((), dtype=dtype)
                )
                return jnp.maximum(
                    min_damping, damping_scale * j_scale * (r_norm**damping_power)
                )

            damping = jax.lax.cond(
                damping_scale > zero,
                _compute_damping,
                lambda _: zero,
                operand=None,
            )

        sub = Subproblem(
            resid=resid,
            theta=theta,
            Jt=Jt,
            jvp_fn=jvp_fn,
            JT=JT,
            whiten=whiten,
            whiten_transpose=whiten_transpose,
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

        def _compute_step(_):
            newton_step = self.linear_solver.prepare_newton(sub)
            grad = newton_step.grad
            u = newton_step.d
            d = whiten(u)
            grad_norm = jnp.linalg.norm(grad)
            direction_norm = jnp.linalg.norm(u)
            slope = mm(grad, u)
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
                    u_a_raw,
                    a_raw,
                    acceleration_ratio,
                    raw_ratio_accepted,
                ) = self._geodesic_correction(
                    residual_flat,
                    theta,
                    d,
                    u,
                    newton_step.correction,
                    ctx,
                    geodesic_acceptance_ratio,
                    dtype,
                )
                ratio_accepted = (
                    descent_ok
                    & raw_ratio_accepted
                    & jnp.all(jnp.isfinite(u_a_raw))
                    & jnp.all(jnp.isfinite(a_raw))
                )
                u_a = jnp.where(ratio_accepted, u_a_raw, jnp.zeros_like(u))
                a = jnp.where(ratio_accepted, a_raw, jnp.zeros_like(d))
                u_a_m = u_a[:n_m] if self.penalized else None
            else:
                u_a = None
                a = None
                u_a_m = None
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
                    y_m=y_m if self.penalized else None,
                    u_m=u[:n_m] if self.penalized else None,
                    u_a_m=u_a_m,
                    ridge=ridge if self.penalized else None,
                    resid_loss_old=resid_loss_old if self.penalized else None,
                    penalty_old=penalty_value_old if self.penalized else None,
                )
                if self.geodesic_acceleration:
                    step_u_norm = jnp.linalg.norm(
                        stepsize_ls * u + 0.5 * (stepsize_ls**2) * u_a
                    )
                else:
                    step_u_norm = stepsize_ls * direction_norm
                step_norm_ls = jnp.where(accepted_ls, step_u_norm, zero)
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
            comp_changed,
            homotopy_changed,
        ) = self._post_step_complementarity(
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
        )
        step_status = jnp.where(
            homotopy_changed
            & jnp.isfinite(loss)
            & (step_status != SolverStatus.LINEAR_SOLVE_FAILED),
            jnp.asarray(SolverStatus.RUNNING, dtype=jnp.int32),
            step_status,
        )
        cache_valid = cache_valid & (~homotopy_changed)
        if solver_cache is not None:
            solver_cache = dataclasses.replace(
                solver_cache, valid=solver_cache.valid & (~homotopy_changed)
            )

        instances = dict(
            ridge=ridge if self.penalized else None,
            metric=state.metric,
            preconditioner=state.preconditioner,
            homotopy=new_homotopy,
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
            ),
        )

    def _converged(self, info, atol, gtol, xtol):
        homotopy_active = self._homotopy_active(info.homotopy)
        stable = ~info.complementarity_changed
        step_ok = info.step_status == SolverStatus.RUNNING
        stationarity_eligible = (info.step_status != SolverStatus.NONFINITE) & (
            info.step_status != SolverStatus.LINEAR_SOLVE_FAILED
        )
        gtol_met = stable & stationarity_eligible & (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (
            stable & step_ok & (xtol > 0) & info.accepted & (info.step_norm < xtol)
        )
        if self.penalized:
            residual_ok = (~homotopy_active) & (
                (atol <= 0) | (jnp.sqrt(info.resid_loss) <= atol)
            )
            return (gtol_met | xtol_met) & residual_ok
        exact_root = (~homotopy_active) & (info.step_status == SolverStatus.CONVERGED)
        atol_met = (
            (~homotopy_active) & step_ok & (atol > 0) & (jnp.sqrt(info.loss) < atol)
        )
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
