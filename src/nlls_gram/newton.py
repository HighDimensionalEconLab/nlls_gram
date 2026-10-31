"""Undamped (or residual-coupled) Newton / Gauss-Newton with Armijo backtracking
for ``min ||r(x, args, p)||^2`` (:class:`Newton`) and the ridge objective
``min ||r(x, args, p)||^2 + ridge * ||x_m||_W^2`` (:class:`RidgeNewton`).

At each outer step, computes the minimum-metric-norm Gauss-Newton direction
(or penalized ridge Gauss-Newton direction) once and searches along
``x + alpha * d`` with a residual-only Armijo backtracking loop against the
half-objective merit function using the actual directional derivative
``s = g'u``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
from jax.flatten_util import ravel_pytree

from nlls_gram.linear_solvers import (
    CG,
    LU,
    SVD,
    Cholesky,
    GramCG,
    Subproblem,
    _config_static_key,
)
from nlls_gram.lm_core import LevenbergMarquardtBase, _init_residual_and_aux
from nlls_gram.lm_types import SolverContext
from nlls_gram.metric_lm import _EuclideanMetric
from nlls_gram.newton_types import (
    NewtonAction,
    NewtonContext,
    NewtonHyperparams,
    NewtonInfo,
    NewtonSolveResult,
    NewtonState,
    NewtonStatus,
    _cast_newton_hyper,
)
from nlls_gram.utilities import (
    _static_key_component,
    _where_tree,
    _zero_tangent_leaf,
    canonicalize_residual,
    mm,
)

__all__ = ["Newton", "RidgeNewton"]


def _armijo_backtracking(
    residual_value,
    theta,
    loss_old,
    d,
    slope,
    hyper,
    active,
    *,
    a=None,
    y_m=None,
    u_m=None,
    u_a_m=None,
    ridge=None,
    resid_loss_old=None,
    penalty_old=None,
):
    dtype = loss_old.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    c1 = jnp.asarray(hyper.c1, dtype=dtype)
    decrease_factor = jnp.asarray(hyper.decrease_factor, dtype=dtype)
    maxls = jnp.where(
        active,
        jnp.asarray(hyper.maxls, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    max_stepsize = jnp.asarray(hyper.max_stepsize, dtype=dtype)
    linesearch_tol = jnp.asarray(hyper.linesearch_tol, dtype=dtype)
    alpha0 = jnp.minimum(one, max_stepsize)
    penalized = ridge is not None
    if not penalized:
        resid_loss_old = loss_old
        penalty_old = zero

    def cond(carry):
        trial, _, accepted, stagnated, _, _, _, _, _, _ = carry
        return (~accepted) & (~stagnated) & (trial < maxls)

    def body(carry):
        (
            trial,
            alpha,
            _,
            _,
            theta_kept,
            loss_kept,
            resid_loss_kept,
            penalty_kept,
            _,
            _,
        ) = carry
        step_theta = alpha * d if a is None else alpha * d + 0.5 * (alpha**2) * a
        cand_theta = theta + step_theta
        cand_finite = jnp.all(jnp.isfinite(cand_theta))
        same_theta = cand_finite & jnp.all(cand_theta == theta)
        cand_resid = residual_value(cand_theta)
        cand_resid_loss = jnp.sum(cand_resid**2)
        if penalized:
            step_u_m = (
                alpha * u_m if u_a_m is None else alpha * u_m + 0.5 * (alpha**2) * u_a_m
            )
            cand_penalty = jnp.sum((y_m + step_u_m) ** 2)
            cand_loss = cand_resid_loss + ridge * cand_penalty
        else:
            cand_penalty = zero
            cand_loss = cand_resid_loss
        trial_finite = (
            cand_finite
            & jnp.all(jnp.isfinite(cand_resid))
            & jnp.isfinite(cand_resid_loss)
            & jnp.isfinite(cand_loss)
        )
        rhs = loss_old + 2.0 * c1 * alpha * slope + 2.0 * linesearch_tol
        ok = (~same_theta) & trial_finite & (cand_loss <= rhs)
        new_trial = trial + jnp.asarray(1, dtype=jnp.int32)
        new_theta = jnp.where(ok, cand_theta, theta_kept)
        new_loss = jnp.where(ok, cand_loss, loss_kept)
        new_resid_loss = jnp.where(ok, cand_resid_loss, resid_loss_kept)
        new_penalty = jnp.where(ok, cand_penalty, penalty_kept)
        next_alpha = jnp.where(
            ok | same_theta | (new_trial >= maxls),
            alpha,
            alpha * decrease_factor,
        )
        last_loss = jnp.where(same_theta, loss_old, cand_loss)
        return (
            new_trial,
            next_alpha,
            ok,
            same_theta,
            new_theta,
            new_loss,
            new_resid_loss,
            new_penalty,
            alpha,
            last_loss,
        )

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),
        alpha0,
        jnp.asarray(False, dtype=jnp.bool_),
        jnp.asarray(False, dtype=jnp.bool_),
        theta,
        loss_old,
        resid_loss_old,
        penalty_old,
        zero,
        loss_old,
    )
    (
        trials,
        _,
        accepted,
        stagnated,
        theta_new,
        loss_new,
        resid_loss_new,
        penalty_new,
        last_alpha,
        last_loss,
    ) = jax.lax.while_loop(cond, body, init_carry)
    stepsize = jnp.where(accepted, last_alpha, zero)
    step_status = jnp.where(
        accepted,
        jnp.asarray(NewtonStatus.RUNNING, dtype=jnp.int32),
        jnp.where(
            stagnated,
            jnp.asarray(NewtonStatus.STAGNATION, dtype=jnp.int32),
            jnp.asarray(NewtonStatus.LINE_SEARCH_FAILED, dtype=jnp.int32),
        ),
    )
    return (
        theta_new,
        loss_new,
        resid_loss_new,
        penalty_new,
        last_loss,
        accepted,
        stepsize,
        last_alpha,
        trials,
        step_status,
    )


class Newton(LevenbergMarquardtBase):
    """Newton / Gauss-Newton with Armijo backtracking over a JAX pytree ``x``.

    ``residual_fn`` takes ``(x)``, ``(x, args)``, or ``(x, args, p)`` and
    returns a residual pytree (or ``(residual, aux)`` with ``has_aux=True``).
    ``args`` is solver-inert auxiliary data. ``p`` is the parameter pytree with
    respect to which ``solve(...).x`` carries an implicit derivative.

    ``metric`` (default ``None`` -- Euclidean) is a
    :class:`~nlls_gram.Metric` covering the leading ``metric.size``
    coordinates of the flattened iterate, with the rest a free block. In
    whitened coordinates ``u = F_bar d`` (``B = J F_bar^{-1}``), each outer
    step computes the minimum-metric-norm Gauss-Newton direction ``u = -B^+ r``
    once, maps back ``d = F_bar^{-1} u``, and searches along ``x + alpha * d``
    with a residual-only Armijo backtracking loop against the merit function
    ``0.5 * ||r(x)||^2`` using the actual directional derivative
    ``s = g'u = (B'r)'u = r'J d``:

    - :class:`~nlls_gram.Cholesky` (the default, ``form="auto"``) factors the
      ``m x m`` Gram matrix ``B B'`` when ``n > m`` and the ``n x n`` normal
      matrix ``B'B`` when ``n <= m``.
    - :class:`~nlls_gram.QR` computes the minimum-norm or least-squares step
      from a single thin QR factorization per outer step without forming
      ``B B'`` or ``B'B``.
    - :class:`~nlls_gram.SVD` computes the truncated-SVD pseudoinverse step
      with threshold ``max(m, n) * eps * sigma_max`` for rank-deficient
      systems.
    - :class:`~nlls_gram.LU` factors a square Jacobian directly (``m == n``).
    - :class:`~nlls_gram.GramCG` and :class:`~nlls_gram.CG` apply matrix-free
      preconditioned conjugate gradients in residual space (``m <= n``) or
      parameter space (``n <= m``).

    Line-search defaults match ``jaxopt.BacktrackingLineSearch`` (JAXopt 0.8.5):
    ``c1=1e-4``, ``decrease_factor=0.8``, ``maxls=30``, ``max_stepsize=1.0``,
    and ``linesearch_tol=0.0``. Every outer iteration resets the initial trial
    step size to ``min(1.0, max_stepsize)``.

    When ``reg > 0`` (default ``0.0``), applies stateless parameter-scale-invariant
    residual-coupled Tikhonov regularization to the linear solve at each outer
    step before Armijo backtracking:
    ``damping = max(min_damping, reg * (||B||_F^2 / m) * ||r||_2**reg_power)``
    for dense solvers (or ``max(min_damping, reg * ||r||_2**reg_power)`` for
    matrix-free Krylov solvers). As ``||r||_2 -> 0``, the residual-coupled term
    decays superlinearly for ``reg_power > 1`` until it reaches ``min_damping``
    (with ``Cholesky`` additionally enforcing a relative diagonal floor
    ``eps * max(diag(G))``), keeping ill-conditioned neural or collocation
    Jacobians well-posed far from the root.
    """

    _action_cls = NewtonAction
    _context_cls = NewtonContext
    _result_cls = NewtonSolveResult
    _status_cls = NewtonStatus
    _state_field = "newton_state"
    _has_metric = True
    _metric_defines_objective = False
    _forward_role_attr = "supports_newton"

    def __init__(
        self,
        residual_fn,
        *,
        metric=None,
        c1=1e-4,
        decrease_factor=0.8,
        maxls=30,
        max_stepsize=1.0,
        linesearch_tol=0.0,
        reg=0.0,
        reg_power=1.5,
        min_damping=1e-12,
        geodesic_acceleration=False,
        geodesic_acceptance_ratio=0.75,
        linear_solver=Cholesky(),  # noqa: B008 -- frozen, immutable default
        jacobian_mode="auto",
        jacobian_chunk=None,
        ad_solver=None,
        has_aux=False,
        cache_jacobian=False,
    ):
        canonical_residual, residual_arity = canonicalize_residual(residual_fn)
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
        if reg < 0.0:
            raise ValueError("reg must be nonnegative")
        if reg_power < 0.0:
            raise ValueError("reg_power must be nonnegative")
        if min_damping <= 0.0:
            raise ValueError("min_damping must be positive")
        if geodesic_acceptance_ratio < 0.0:
            raise ValueError("geodesic_acceptance_ratio must be nonnegative")
        if reg > 0.0 and isinstance(linear_solver, LU):
            raise ValueError(
                "linear_solver=LU() does not support reg > 0; use Cholesky(), QR(), "
                "SVD(), CG(), or GramCG()"
            )
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.initial_metric = _EuclideanMetric() if metric is None else metric
        self._check_registered_instance(self.initial_metric, "metric")
        self.c1 = c1
        self.decrease_factor = decrease_factor
        self.maxls = maxls
        self.max_stepsize = max_stepsize
        self.linesearch_tol = linesearch_tol
        self.reg = reg
        self.reg_power = reg_power
        self.min_damping = min_damping
        self.geodesic_acceleration = bool(geodesic_acceleration)
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        self.linear_solver = linear_solver
        self.jacobian_mode = jacobian_mode
        self.jacobian_chunk = jacobian_chunk
        self.ad_solver = ad_solver
        self._validate_configuration(linear_solver, ad_solver, penalized=False)
        krylov = isinstance(linear_solver, (CG, GramCG))
        if krylov:
            if linear_solver.preconditioner is None:
                raise ValueError(
                    "the forward linear_solver requires a preconditioner; "
                    "IdentityPreconditioner() is the explicit opt-out "
                    "(preconditioner=None is legal only in the ad_solver role)"
                )
            if linear_solver.preconditioner.requires_positive_damping:
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
            self._check_registered_instance(
                linear_solver.preconditioner, "linear_solver.preconditioner"
            )
            self.initial_preconditioner = linear_solver.preconditioner
            self.iterative_tol = linear_solver.tol
            self.iterative_atol = linear_solver.atol
            self.iterative_maxiter = linear_solver.maxiter
        else:
            self.initial_preconditioner = None
            self.iterative_tol = 0.0
            self.iterative_atol = 0.0
            self.iterative_maxiter = 8
        if isinstance(ad_solver, (CG, GramCG)):
            self.ad_solver_tol = ad_solver.tol
            self.ad_solver_atol = ad_solver.atol
            self.ad_solver_maxiter = ad_solver.maxiter
            self.ad_solver_penalty = getattr(ad_solver, "penalty", None)
            if ad_solver.preconditioner is None:
                self.ad_solver_preconditioner = None
                if self.initial_preconditioner is None:
                    self._ad_preconditioner_source = "none"
                else:
                    self._ad_preconditioner_source = "carried"
            else:
                if ad_solver.preconditioner.requires_positive_damping:
                    raise ValueError(
                        "this preconditioner divides by the live damping and "
                        "cannot serve in ad_solver (the AD system is undamped)"
                    )
                self._check_registered_instance(
                    ad_solver.preconditioner, "ad_solver.preconditioner"
                )
                self.ad_solver_preconditioner = ad_solver.preconditioner
                self._ad_preconditioner_source = "explicit"
        else:
            self.ad_solver_tol = None
            self.ad_solver_atol = 0.0
            self.ad_solver_maxiter = None
            self.ad_solver_penalty = None
            self.ad_solver_preconditioner = None
            inherit = (
                ad_solver is None
                and krylov
                and not linear_solver.preconditioner.requires_positive_damping
            )
            self._ad_preconditioner_source = "carried" if inherit else "none"
        self.has_aux = has_aux
        self.cache_jacobian = cache_jacobian and linear_solver.materializes_jacobian
        self._static_key = tuple(
            _static_key_component(value)
            for value in (
                residual_fn,
                jax.tree_util.tree_structure(self.initial_metric),
                c1,
                decrease_factor,
                maxls,
                max_stepsize,
                linesearch_tol,
                reg,
                reg_power,
                min_damping,
                self.geodesic_acceleration,
                geodesic_acceptance_ratio,
                _config_static_key(linear_solver, baked=False),
                jacobian_mode,
                jacobian_chunk,
                _config_static_key(ad_solver, baked=True),
                has_aux,
                self.cache_jacobian,
            )
        )
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
            jnp.asarray(self.reg, dtype=eff_dtype),
            jnp.asarray(self.reg_power, dtype=eff_dtype),
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
        self.linear_solver._validate_newton_shape(m, n)
        instances = dict(
            metric=self.initial_metric, preconditioner=self.initial_preconditioner
        )
        if not self.cache_jacobian:
            return NewtonState(**instances)
        dtype = residual_shape.dtype
        return NewtonState(
            resid=jnp.zeros(residual_shape.shape, dtype=dtype),
            Jt=jnp.zeros((n, m), dtype=dtype),
            jacobian_valid=jnp.asarray(False, dtype=jnp.bool_),
            aux=jax.tree.map(lambda s: jnp.zeros(s.shape, dtype=s.dtype), aux_shape),
            solver_cache=self.linear_solver.new_cache(m, n, n_m, dtype, False),
            **instances,
        )

    def _solve_lm_state(self, x0, args, p, newton_state):
        x_leaves = jax.tree.leaves(x0)
        x_dtype = jnp.result_type(*x_leaves) if x_leaves else jnp.result_type(float)
        if newton_state is not None:
            if not isinstance(newton_state, NewtonState):
                raise TypeError(
                    "newton_state must be a NewtonState; got "
                    f"{type(newton_state).__name__}"
                )
            if self.cache_jacobian and newton_state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True requires a NewtonState initialized with "
                    "solver.init(x0, args, p=p)"
                )
            resolved = self._resolved_state(newton_state)
        else:
            resolved = self.init(x0, args, p=p)
        if resolved.hyper is None:
            return dataclasses.replace(resolved, hyper=self.hyperparams(x_dtype))
        return self._cast_state(resolved, x_dtype)

    def _initial_info(self, x, newton_state, args, p):
        residual, aux = self._residual_and_aux(x, args, p)
        theta, _ = ravel_pytree(x)
        self._block_sizes(theta.shape[0])
        self.linear_solver._validate_newton_shape(residual.shape[0], theta.shape[0])
        theta_finite = jnp.all(jnp.isfinite(theta))
        resid_finite = jnp.all(jnp.isfinite(residual))
        raw_loss = jnp.sum(residual**2)
        base_finite = theta_finite & resid_finite & jnp.isfinite(raw_loss)
        loss = jnp.where(
            base_finite, raw_loss, jnp.asarray(jnp.nan, dtype=residual.dtype)
        )
        exact_root = base_finite & jnp.all(residual == 0)
        zero = jnp.zeros((), dtype=residual.dtype)
        infinity = jnp.asarray(jnp.inf, dtype=residual.dtype)
        step_status = jnp.where(
            ~base_finite,
            jnp.asarray(NewtonStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                exact_root,
                jnp.asarray(NewtonStatus.CONVERGED, dtype=jnp.int32),
                jnp.asarray(NewtonStatus.RUNNING, dtype=jnp.int32),
            ),
        )
        return NewtonInfo(
            loss=loss,
            loss_old=loss,
            loss_candidate=loss,
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            stepsize=zero,
            trial_stepsize=zero,
            num_linesearch_steps=jnp.asarray(0, dtype=jnp.int32),
            grad_norm=jnp.where(exact_root, zero, infinity),
            direction_norm=zero,
            step_norm=zero,
            directional_derivative=zero,
            step_status=step_status,
            used_geodesic=jnp.asarray(False, dtype=jnp.bool_),
            acceleration_ratio=zero,
            aux=aux,
        )

    def update(self, x, newton_state, args=None, p=None):
        """One Newton + Armijo step: returns ``(x_new, newton_state, info)``."""
        self._check_residual_args(args, p)
        if not isinstance(newton_state, NewtonState):
            raise TypeError(
                f"newton_state must be a NewtonState; got {type(newton_state).__name__}"
            )
        if self.cache_jacobian and newton_state.jacobian_valid is None:
            raise ValueError(
                "cache_jacobian=True requires a NewtonState initialized with "
                "solver.init(x0, args, p=p)"
            )
        theta, unravel = ravel_pytree(x)
        n_m, n_f = self._block_sizes(theta.shape[0])

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
                newton_state.jacobian_valid,
                lambda _: (newton_state.resid, newton_state.Jt, newton_state.aux),
                lambda _: self._dense_resid_jt_aux(residual_flat, theta),
                operand=None,
            )
        else:
            resid, Jt, aux = self._dense_resid_jt_aux(residual_flat, theta)

        self.linear_solver._validate_newton_shape(resid.shape[0], theta.shape[0])
        dtype = resid.dtype
        zero = jnp.zeros((), dtype=dtype)
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        raw_loss_old = jnp.sum(resid**2)
        base_finite = (
            jnp.all(jnp.isfinite(theta))
            & jnp.all(jnp.isfinite(resid))
            & jnp.isfinite(raw_loss_old)
        )
        loss_old = jnp.where(base_finite, raw_loss_old, nan)
        exact_root = base_finite & jnp.all(resid == 0)

        hyper = (
            newton_state.hyper
            if newton_state.hyper is not None
            else self.hyperparams(dtype)
        )
        ctx = SolverContext(
            x=theta, lm_state=self._resolved_state(newton_state), args=args, p=p
        )

        def whiten(v):
            return jnp.asarray(self._extended_solve(v, ctx), dtype=dtype)

        def whiten_transpose(v):
            return jnp.asarray(self._extended_solve_transpose(v, ctx), dtype=dtype)

        if newton_state.hyper is None and self.reg == 0.0:
            damping = zero
        else:
            reg = jnp.asarray(hyper.reg, dtype=dtype)
            reg_power = jnp.asarray(hyper.reg_power, dtype=dtype)
            min_damping = jnp.asarray(hyper.min_damping, dtype=dtype)
            r_norm = jnp.sqrt(raw_loss_old)

            def _compute_damping(_):
                j_scale = (
                    jnp.sum(whiten_transpose(Jt) ** 2) / resid.shape[0]
                    if Jt is not None
                    else jnp.ones((), dtype=dtype)
                )
                return jnp.maximum(min_damping, reg * j_scale * (r_norm**reg_power))

            damping = jax.lax.cond(
                reg > zero, _compute_damping, lambda _: zero, operand=None
            )
        sub = Subproblem(
            resid=resid,
            theta=theta,
            Jt=Jt,
            jvp_fn=jvp_fn,
            JT=JT,
            whiten=whiten,
            whiten_transpose=whiten_transpose,
            y_m=jnp.zeros(n_m, dtype=dtype),
            penalty_gradient=jnp.zeros(theta.shape[0], dtype=dtype),
            ridge=zero,
            damping=damping,
            n_m=n_m,
            n_f=n_f,
            cache=newton_state.solver_cache,
            cache_enabled=self.cache_jacobian,
            hyper=hyper,
            ctx=ctx,
            penalized=False,
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

                def first_jvp(th):
                    return jax.jvp(residual_flat, (th,), (d,), has_aux=self.has_aux)[1]

                f_vv = jax.jvp(first_jvp, (theta,), (d,))[1]
                u_a_raw = newton_step.correction(f_vv)
                a_raw = whiten(u_a_raw)
                acceleration_ratio = (
                    2.0
                    * jnp.linalg.norm(u_a_raw)
                    / (direction_norm + jnp.finfo(dtype).eps)
                )
                ratio_accepted = (
                    descent_ok
                    & (geodesic_acceptance_ratio > zero)
                    & jnp.all(jnp.isfinite(u_a_raw))
                    & jnp.all(jnp.isfinite(a_raw))
                    & (acceleration_ratio > zero)
                    & (acceleration_ratio <= geodesic_acceptance_ratio)
                )
                u_a = jnp.where(ratio_accepted, u_a_raw, jnp.zeros_like(u))
                a = jnp.where(ratio_accepted, a_raw, jnp.zeros_like(d))
            else:
                u_a = None
                a = None
                acceleration_ratio = zero
                ratio_accepted = jnp.asarray(False, dtype=jnp.bool_)

            def _run_ls(_):
                (
                    theta_ls,
                    loss_ls,
                    _,
                    _,
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
                    jnp.asarray(NewtonStatus.NON_DESCENT, dtype=jnp.int32),
                    jnp.asarray(NewtonStatus.LINEAR_SOLVE_FAILED, dtype=jnp.int32),
                )
                return (
                    theta,
                    loss_old,
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
                jnp.asarray(NewtonStatus.CONVERGED, dtype=jnp.int32),
                jnp.asarray(NewtonStatus.NONFINITE, dtype=jnp.int32),
            )
            scalar_val = jnp.where(exact_root, zero, nan)
            solver_cache = newton_state.solver_cache
            if solver_cache is not None:
                solver_cache = dataclasses.replace(
                    solver_cache, valid=jnp.asarray(False, dtype=jnp.bool_)
                )
            return (
                theta,
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

        instances = dict(
            metric=newton_state.metric, preconditioner=newton_state.preconditioner
        )
        if self.cache_jacobian:
            new_state = NewtonState(
                resid=resid,
                Jt=Jt,
                jacobian_valid=cache_valid,
                aux=aux,
                hyper=newton_state.hyper,
                solver_cache=solver_cache,
                **instances,
            )
        else:
            new_state = NewtonState(hyper=newton_state.hyper, **instances)
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
            ),
        )

    def solve(
        self,
        x0,
        args=None,
        *,
        p=None,
        newton_state=None,
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
        """Run repeated Newton + Armijo updates until a stopping rule fires.

        Parameters match :meth:`~nlls_gram.LevenbergMarquardt.solve`, with
        ``newton_state`` in place of ``lm_state`` and
        :class:`~nlls_gram.NewtonContext` / :class:`~nlls_gram.NewtonAction` /
        :class:`~nlls_gram.NewtonSolveResult` / :class:`~nlls_gram.NewtonStatus`
        for callbacks and results.
        """
        return self._run_solve(
            x0,
            args,
            p=p,
            state=newton_state,
            max_steps=max_steps,
            max_steps_is_success=max_steps_is_success,
            atol=atol,
            gtol=gtol,
            xtol=xtol,
            callback=callback,
            user_state=user_state,
            save_steps=save_steps,
            multi_start=multi_start,
            jit=jit,
        )

    def _converged(self, info, atol, gtol, xtol):
        exact_root = info.step_status == NewtonStatus.CONVERGED
        step_ok = info.step_status == NewtonStatus.RUNNING
        stationarity_eligible = (info.step_status != NewtonStatus.NONFINITE) & (
            info.step_status != NewtonStatus.LINEAR_SOLVE_FAILED
        )
        atol_met = step_ok & (atol > 0) & (jnp.sqrt(info.loss) < atol)
        gtol_met = stationarity_eligible & (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = step_ok & (xtol > 0) & info.accepted & (info.step_norm < xtol)
        return exact_root | atol_met | gtol_met | xtol_met

    def _step_failure_status(self, info):
        ok = (info.step_status == NewtonStatus.RUNNING) | (
            info.step_status == NewtonStatus.CONVERGED
        )
        return jnp.where(
            ~jnp.isfinite(info.loss),
            jnp.asarray(NewtonStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                ok,
                jnp.asarray(NewtonStatus.RUNNING, dtype=jnp.int32),
                info.step_status,
            ),
        )

    def _cast_state(self, newton_state, dtype):
        return dataclasses.replace(
            newton_state,
            hyper=_cast_newton_hyper(newton_state.hyper, dtype),
        )

    def _ranking_objective(self, result, p, callback):
        if callback is None:
            loss = result.info.loss
        else:
            residual = self._residual_and_aux(result.x, result.args, p)[0]
            loss = jnp.sum(residual**2)
        return jnp.where(
            jnp.isfinite(loss), loss, jnp.asarray(jnp.inf, dtype=loss.dtype)
        )

    def _resolved_ad_solver(self, m, n):
        resolved = self.ad_solver
        if resolved is None:
            if isinstance(self.linear_solver, SVD):
                return SVD()
            if isinstance(self.linear_solver, GramCG) and m <= n:
                return self.linear_solver
            if isinstance(self.linear_solver, CG) and n <= m:
                return self.linear_solver
            if m == n:
                return LU()
            return SVD()
        if isinstance(resolved, LU) and m != n:
            raise ValueError(
                f"ad_solver=LU() needs a square system, but the residual is {m} "
                f"and x flattens to {n}: a rectangular tangent is selected by a "
                "minimum-metric-norm rule that a plain solve cannot express. "
                "Use SVD(), or Cholesky() when the small side has full rank"
            )
        if isinstance(resolved, GramCG) and m > n:
            raise ValueError(
                f"ad_solver=GramCG() needs m <= n, but the residual is {m} and "
                f"x flattens to {n}: the undamped dual J~J~' is singular there, "
                "so CG returns a wrong tangent rather than failing. Use SVD(), "
                "or CG(precond, penalty=...) to regularize"
            )
        if isinstance(resolved, CG) and n > m and resolved.penalty is None:
            raise ValueError(
                f"ad_solver=CG() needs n <= m, but the residual is {m} and x "
                f"flattens to {n}: the undamped normal J~'J~ is singular there, "
                "so CG returns a wrong tangent rather than failing. Use "
                "GramCG(precond), SVD(), or CG(precond, penalty=...)"
            )
        if isinstance(resolved, Cholesky):
            resolved._validate_newton_shape(m, n)
        return resolved

    def _ad_x_tangent(self, x, args, p, p_dot, result, ad_success, initial_ad_point):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        newton_state = jax.lax.stop_gradient(result.newton_state)
        initial_instances = jax.lax.stop_gradient(initial_ad_point[3:5])
        newton_state = dataclasses.replace(
            newton_state,
            metric=_where_tree(ad_success, newton_state.metric, initial_instances[0]),
            preconditioner=_where_tree(
                ad_success, newton_state.preconditioner, initial_instances[1]
            ),
        )
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        resolved = self._resolved_ad_solver(residual.shape[0], theta.shape[0])
        if isinstance(resolved, LU):
            Jt = self._assemble_jt(theta_jvp, theta, residual)
            n = Jt.shape[0]
            J = jnp.where(ad_success, Jt.T, jnp.eye(n, dtype=Jt.dtype))
            return unravel(-jnp.linalg.solve(J, residual_p_dot))
        ctx = SolverContext(x=theta, lm_state=newton_state, args=args, p=p)
        dtype = residual.dtype

        def whiten(v):
            return jnp.asarray(self._extended_solve(v, ctx), dtype=dtype)

        def whiten_transpose(v):
            return jnp.asarray(self._extended_solve_transpose(v, ctx), dtype=dtype)

        if isinstance(resolved, (CG, GramCG)):
            u = self._ad_tangent_krylov(
                resolved,
                theta,
                theta_jvp,
                residual_p_dot,
                whiten,
                whiten_transpose,
                ctx,
                ad_success,
            )
        else:
            Bt = whiten_transpose(self._assemble_jt(theta_jvp, theta, residual))
            if isinstance(resolved, SVD):
                U, sigma, Vt = jnp.linalg.svd(Bt.T, full_matrices=False)
                cutoff = max(Bt.shape) * jnp.finfo(Bt.dtype).eps * sigma[0]
                mask = sigma > cutoff
                safe_sigma = jnp.where(mask, sigma, jnp.ones_like(sigma))
                inverted = jnp.where(mask, 1.0 / safe_sigma, jnp.zeros_like(sigma))
                u = -mm(Vt.T, inverted * mm(U.T, residual_p_dot))
            else:
                n, m = Bt.shape
                if resolved._resolved_form(m, n, False) == "gram":
                    gram = mm(Bt.T, Bt)
                    gram = jnp.where(ad_success, gram, jnp.eye(m, dtype=Bt.dtype))
                    factor = jsp_linalg.cho_factor(gram)
                    u = -mm(Bt, jsp_linalg.cho_solve(factor, residual_p_dot))
                else:
                    normal = mm(Bt, Bt.T)
                    normal = jnp.where(ad_success, normal, jnp.eye(n, dtype=Bt.dtype))
                    factor = jsp_linalg.cho_factor(normal)
                    u = -jsp_linalg.cho_solve(factor, mm(Bt, residual_p_dot))
        theta_dot = jax.lax.custom_linear_solve(
            lambda v: v,
            u,
            lambda _, b: whiten(b),
            transpose_solve=lambda _, b: whiten_transpose(b),
        )
        return unravel(theta_dot)

    def _ad_tangent_krylov(
        self,
        config,
        theta,
        theta_jvp,
        residual_p_dot,
        whiten,
        whiten_transpose,
        ctx,
        ad_success=True,
    ):
        dtype = residual_p_dot.dtype
        transpose_fn = jax.linear_transpose(theta_jvp, theta)
        zero_damping = jnp.zeros((), dtype=dtype)

        def JT(cotangent):
            return transpose_fn(cotangent)[0]

        def B(u):
            return theta_jvp(whiten(u))

        def Bt(w):
            return whiten_transpose(JT(w))

        apply_M = None
        ad_preconditioner = self._ad_preconditioner(ctx.lm_state)
        if ad_preconditioner is not None:

            def apply_M(v):
                return jnp.where(
                    ad_success, ad_preconditioner.apply(v, zero_damping, ctx), v
                )

        def cg(matvec, rhs, preconditioner=apply_M):
            solution, _ = jsp_sparse_linalg.cg(
                matvec,
                rhs,
                tol=self._ad_cg_tol(dtype),
                atol=jnp.asarray(self.ad_solver_atol, dtype=dtype),
                maxiter=self.ad_solver_maxiter,
                M=preconditioner,
            )
            return solution

        if isinstance(config, GramCG):

            def dual_matvec(y):
                return jnp.where(ad_success, B(Bt(y)), y)

            y = jax.lax.custom_linear_solve(
                dual_matvec,
                residual_p_dot,
                lambda _, c: cg(dual_matvec, c),
                symmetric=True,
            )
            return -Bt(y)

        penalty = self.ad_solver_penalty

        def normal_matvec(u):
            value = Bt(B(u))
            if penalty is not None:
                value = value + jnp.asarray(penalty, dtype=dtype) * u
            return jnp.where(ad_success, value, u)

        if penalty is not None:
            transpose_solve = lambda _, c: cg(normal_matvec, c)  # noqa: E731
        else:

            def transpose_solve(_, c):
                return Bt(dual_solve(dual_solve(B(c))))

            def dual_solve(y):
                return cg(
                    lambda w: jnp.where(ad_success, B(Bt(w)), w),
                    y,
                    preconditioner=None,
                )

        rhs = -Bt(residual_p_dot)
        return jax.lax.custom_linear_solve(
            normal_matvec,
            rhs,
            lambda _, c: cg(normal_matvec, c),
            transpose_solve=transpose_solve,
        )


class RidgeNewton(Newton):
    """Newton / Gauss-Newton with Armijo backtracking for the ridge objective
    ``F(x) = ||r(x, args, p)||^2 + ridge * ||x_m||_W^2``.

    The Newton + Armijo counterpart to :class:`~nlls_gram.RidgeLevenbergMarquardt`.
    In whitened coordinates ``y = F_bar x`` (``J~ = J F_bar^{-1}``), each outer
    step solves the ridge Gauss-Newton system
    ``(J~'J~ + ridge * E + damping * I) u = -(J~'r + ridge * [y_m; 0])`` once,
    maps back ``d = F_bar^{-1} u``, and runs Armijo backtracking on the full
    ridge objective ``||r(x + alpha * d)||^2 + ridge * ||y_m + alpha * u_m||^2``
    against the directional derivative ``s = g'u``. Supports
    :class:`~nlls_gram.AnnealRidge`, conjunctive ``(gtol | xtol) & atol``
    stopping, and Gauss-Newton implicit differentiation of the ridge stationary
    point with respect to ``p`` (solving
    ``(J~'J~ + ridge * E) y_dot = -J~' (dr/dp) p_dot``, matching
    :class:`~nlls_gram.RidgeLevenbergMarquardt`).
    """

    _metric_defines_objective = True

    def __init__(
        self,
        residual_fn,
        *,
        metric,
        ridge=None,
        c1=1e-4,
        decrease_factor=0.8,
        maxls=30,
        max_stepsize=1.0,
        linesearch_tol=0.0,
        reg=0.0,
        reg_power=1.5,
        min_damping=1e-12,
        geodesic_acceleration=False,
        geodesic_acceptance_ratio=0.75,
        linear_solver=Cholesky(),  # noqa: B008 -- frozen, immutable default
        jacobian_mode="auto",
        jacobian_chunk=None,
        ad_solver=None,
        has_aux=False,
        cache_jacobian=False,
    ):
        canonical_residual, residual_arity = canonicalize_residual(residual_fn)
        if (
            ridge is not None
            and not isinstance(ridge, (jax.Array, jax.core.Tracer))
            and float(ridge) <= 0.0
        ):
            raise ValueError(
                "ridge must be strictly positive (ridge = 0 is unsupported: "
                "use AnnealRidge with a positive ridge_floor to "
                "approach the ridgeless limit)"
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
        if reg < 0.0:
            raise ValueError("reg must be nonnegative")
        if reg_power < 0.0:
            raise ValueError("reg_power must be nonnegative")
        if min_damping <= 0.0:
            raise ValueError("min_damping must be positive")
        if geodesic_acceptance_ratio < 0.0:
            raise ValueError("geodesic_acceptance_ratio must be nonnegative")
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.initial_metric = metric
        self._check_registered_instance(metric, "metric")
        self.ridge = ridge
        self.c1 = c1
        self.decrease_factor = decrease_factor
        self.maxls = maxls
        self.max_stepsize = max_stepsize
        self.linesearch_tol = linesearch_tol
        self.reg = reg
        self.reg_power = reg_power
        self.min_damping = min_damping
        self.geodesic_acceleration = bool(geodesic_acceleration)
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        self.linear_solver = linear_solver
        self.jacobian_mode = jacobian_mode
        self.jacobian_chunk = jacobian_chunk
        self.ad_solver = ad_solver
        self._validate_configuration(linear_solver, ad_solver, penalized=True)
        if isinstance(linear_solver, CG):
            if linear_solver.preconditioner is None:
                raise ValueError(
                    "the forward linear_solver requires a preconditioner; "
                    "IdentityPreconditioner() is the explicit opt-out "
                    "(preconditioner=None is legal only in the ad_solver role)"
                )
            if linear_solver.preconditioner.requires_positive_damping:
                raise ValueError(
                    "this preconditioner divides by the live damping and "
                    "cannot serve in RidgeNewton (the RidgeNewton system is undamped)"
                )
            if getattr(linear_solver, "penalty", None) is not None:
                raise ValueError(
                    "CG(penalty=...) is not supported in RidgeNewton's forward "
                    "linear_solver role: the ridge is already in the operator. "
                    "Use penalty only in ad_solver=CG(...)"
                )
            self._check_registered_instance(
                linear_solver.preconditioner, "linear_solver.preconditioner"
            )
            self.initial_preconditioner = linear_solver.preconditioner
            self.iterative_tol = linear_solver.tol
            self.iterative_atol = linear_solver.atol
            self.iterative_maxiter = linear_solver.maxiter
        else:
            self.initial_preconditioner = None
            self.iterative_tol = 0.0
            self.iterative_atol = 0.0
            self.iterative_maxiter = 8
        if isinstance(ad_solver, CG):
            self.ad_solver_tol = ad_solver.tol
            self.ad_solver_atol = ad_solver.atol
            self.ad_solver_maxiter = ad_solver.maxiter
            if ad_solver.preconditioner is None:
                self.ad_solver_preconditioner = None
                if self.initial_preconditioner is None:
                    self._ad_preconditioner_source = "none"
                else:
                    self._ad_preconditioner_source = "carried"
            else:
                if ad_solver.preconditioner.requires_positive_damping:
                    raise ValueError(
                        "this preconditioner divides by the live damping and "
                        "cannot serve in ad_solver (the AD system is undamped)"
                    )
                self._check_registered_instance(
                    ad_solver.preconditioner, "ad_solver.preconditioner"
                )
                self.ad_solver_preconditioner = ad_solver.preconditioner
                self._ad_preconditioner_source = "explicit"
        else:
            self.ad_solver_tol = None
            self.ad_solver_atol = 0.0
            self.ad_solver_maxiter = None
            self.ad_solver_preconditioner = None
            inherit = (
                ad_solver is None
                and isinstance(linear_solver, CG)
                and not linear_solver.preconditioner.requires_positive_damping
            )
            self._ad_preconditioner_source = "carried" if inherit else "none"
        self.has_aux = has_aux
        self.cache_jacobian = cache_jacobian and not isinstance(linear_solver, CG)
        self._static_key = tuple(
            _static_key_component(value)
            for value in (
                residual_fn,
                jax.tree_util.tree_structure(metric),
                ridge,
                c1,
                decrease_factor,
                maxls,
                max_stepsize,
                linesearch_tol,
                reg,
                reg_power,
                min_damping,
                self.geodesic_acceleration,
                geodesic_acceptance_ratio,
                _config_static_key(linear_solver, baked=False),
                jacobian_mode,
                jacobian_chunk,
                _config_static_key(ad_solver, baked=True),
                has_aux,
                self.cache_jacobian,
            )
        )
        self._static_hash = hash(self._static_key)
        self._sealed = True

    def _resolve_ridge(self, dtype):
        if self.ridge is None:
            return jnp.asarray(jnp.sqrt(jnp.finfo(dtype).eps), dtype=dtype)
        return jnp.asarray(self.ridge, dtype=dtype)

    def init(self, x0, args=None, *, p=None):
        """Build the initial :class:`~nlls_gram.NewtonState` at ``x0``."""
        self._check_residual_args(args, p)
        residual_shape, aux_shape = _init_residual_and_aux.eval_shape(self, x0, args, p)
        theta, _ = ravel_pytree(x0)
        m, n = residual_shape.size, theta.size
        n_m, _ = self._block_sizes(n)
        dtype = residual_shape.dtype
        ridge = self._resolve_ridge(dtype)
        instances = dict(
            ridge=ridge,
            metric=self.initial_metric,
            preconditioner=self.initial_preconditioner,
        )
        if not self.cache_jacobian:
            return NewtonState(**instances)
        return NewtonState(
            resid=jnp.zeros(residual_shape.shape, dtype=dtype),
            Jt=jnp.zeros((n, m), dtype=dtype),
            jacobian_valid=jnp.asarray(False, dtype=jnp.bool_),
            aux=jax.tree.map(lambda s: jnp.zeros(s.shape, dtype=s.dtype), aux_shape),
            solver_cache=self.linear_solver.new_cache(m, n, n_m, dtype, True),
            **instances,
        )

    def _solve_lm_state(self, x0, args, p, newton_state):
        x_leaves = jax.tree.leaves(x0)
        x_dtype = jnp.result_type(*x_leaves) if x_leaves else jnp.result_type(float)
        if newton_state is None:
            resolved = self.init(x0, args, p=p)
        else:
            if not isinstance(newton_state, NewtonState):
                raise TypeError(
                    "newton_state must be a NewtonState; got "
                    f"{type(newton_state).__name__}"
                )
            if newton_state.ridge is None:
                raise ValueError(
                    "the caller-supplied newton_state has no ridge; create it with "
                    "init(x, args, p=p) or set a positive ridge"
                )
            if (
                not isinstance(newton_state.ridge, jax.core.Tracer)
                and jnp.ndim(newton_state.ridge) == 0
                and float(newton_state.ridge) <= 0.0
            ):
                raise ValueError(
                    "the caller-supplied newton_state.ridge must be strictly positive "
                    "(ridge = 0 is unsupported)"
                )
            if self.cache_jacobian and newton_state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True requires a NewtonState initialized with "
                    "solver.init(x0, args, p=p)"
                )
            resolved = self._resolved_state(newton_state)
        if resolved.hyper is None:
            resolved = dataclasses.replace(resolved, hyper=self.hyperparams(x_dtype))
        return self._cast_state(resolved, x_dtype)

    def _initial_info(self, x, newton_state, args, p):
        residual, aux = self._residual_and_aux(x, args, p)
        theta, _ = ravel_pytree(x)
        n_m, _ = self._block_sizes(theta.shape[0])
        dtype = residual.dtype
        ridge = jnp.asarray(newton_state.ridge, dtype=dtype)
        resolved = self._resolved_state(newton_state)
        ctx = SolverContext(x=theta, lm_state=resolved, args=args, p=p)
        y_m = jnp.asarray(resolved.metric.factor_apply(theta[:n_m], ctx), dtype=dtype)
        penalty_value = jnp.sum(y_m**2)
        resid_loss = jnp.sum(residual**2)
        raw_loss = resid_loss + ridge * penalty_value
        base_finite = (
            jnp.all(jnp.isfinite(theta))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(raw_loss)
        )
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        loss = jnp.where(base_finite, raw_loss, nan)
        resid_loss_out = jnp.where(base_finite, resid_loss, nan)
        penalty_out = jnp.where(base_finite, penalty_value, nan)
        zero = jnp.zeros((), dtype=dtype)
        infinity = jnp.asarray(jnp.inf, dtype=dtype)
        step_status = jnp.where(
            ~base_finite,
            jnp.asarray(NewtonStatus.NONFINITE, dtype=jnp.int32),
            jnp.asarray(NewtonStatus.RUNNING, dtype=jnp.int32),
        )
        return NewtonInfo(
            loss=loss,
            loss_old=loss,
            loss_candidate=loss,
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            stepsize=zero,
            trial_stepsize=zero,
            num_linesearch_steps=jnp.asarray(0, dtype=jnp.int32),
            grad_norm=infinity,
            direction_norm=zero,
            step_norm=zero,
            directional_derivative=zero,
            step_status=step_status,
            used_geodesic=jnp.asarray(False, dtype=jnp.bool_),
            acceleration_ratio=zero,
            ridge=ridge,
            resid_loss=resid_loss_out,
            penalty_value=penalty_out,
            penalty_grad_norm=infinity,
            aux=aux,
        )

    def update(self, x, newton_state, args=None, p=None):
        """One Newton + Armijo step on the ridge objective."""
        self._check_residual_args(args, p)
        if not isinstance(newton_state, NewtonState):
            raise TypeError(
                f"newton_state must be a NewtonState; got {type(newton_state).__name__}"
            )
        if newton_state.ridge is None:
            raise ValueError(
                "the newton_state has no ridge; create it with init(x, args, p=p)"
            )
        if self.cache_jacobian and newton_state.jacobian_valid is None:
            raise ValueError(
                "cache_jacobian=True requires a NewtonState initialized with "
                "solver.init(x0, args, p=p)"
            )
        theta, unravel = ravel_pytree(x)
        n_m, n_f = self._block_sizes(theta.shape[0])

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
                newton_state.jacobian_valid,
                lambda _: (newton_state.resid, newton_state.Jt, newton_state.aux),
                lambda _: self._dense_resid_jt_aux(residual_flat, theta),
                operand=None,
            )
        else:
            resid, Jt, aux = self._dense_resid_jt_aux(residual_flat, theta)

        dtype = resid.dtype
        zero = jnp.zeros((), dtype=dtype)
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        ridge = jnp.asarray(newton_state.ridge, dtype=dtype)
        resolved_state = self._resolved_state(newton_state)
        ctx = SolverContext(x=theta, lm_state=resolved_state, args=args, p=p)
        y_m = jnp.asarray(
            resolved_state.metric.factor_apply(theta[:n_m], ctx), dtype=dtype
        )
        penalty_value_old = jnp.sum(y_m**2)
        penalty_gradient = jnp.concatenate([y_m, jnp.zeros(n_f, dtype=dtype)])
        penalty_grad_norm = jnp.linalg.norm(penalty_gradient)
        resid_loss_old = jnp.sum(resid**2)
        raw_loss_old = resid_loss_old + ridge * penalty_value_old
        base_finite = (
            jnp.all(jnp.isfinite(theta))
            & jnp.all(jnp.isfinite(resid))
            & jnp.all(jnp.isfinite(y_m))
            & jnp.isfinite(raw_loss_old)
        )
        loss_old = jnp.where(base_finite, raw_loss_old, nan)

        hyper = (
            newton_state.hyper
            if newton_state.hyper is not None
            else self.hyperparams(dtype)
        )

        def whiten(v):
            return jnp.asarray(self._extended_solve(v, ctx), dtype=dtype)

        def whiten_transpose(v):
            return jnp.asarray(self._extended_solve_transpose(v, ctx), dtype=dtype)

        if newton_state.hyper is None and self.reg == 0.0:
            damping = zero
        else:
            reg = jnp.asarray(hyper.reg, dtype=dtype)
            reg_power = jnp.asarray(hyper.reg_power, dtype=dtype)
            min_damping = jnp.asarray(hyper.min_damping, dtype=dtype)
            r_norm = jnp.sqrt(resid_loss_old)

            def _compute_damping(_):
                j_scale = (
                    jnp.sum(whiten_transpose(Jt) ** 2) / resid.shape[0]
                    if Jt is not None
                    else jnp.ones((), dtype=dtype)
                )
                return jnp.maximum(min_damping, reg * j_scale * (r_norm**reg_power))

            damping = jax.lax.cond(
                reg > zero, _compute_damping, lambda _: zero, operand=None
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
            cache=newton_state.solver_cache,
            cache_enabled=self.cache_jacobian,
            hyper=hyper,
            ctx=ctx,
            penalized=True,
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

                def first_jvp(th):
                    return jax.jvp(residual_flat, (th,), (d,), has_aux=self.has_aux)[1]

                f_vv = jax.jvp(first_jvp, (theta,), (d,))[1]
                u_a_raw = newton_step.correction(f_vv)
                a_raw = whiten(u_a_raw)
                acceleration_ratio = (
                    2.0
                    * jnp.linalg.norm(u_a_raw)
                    / (direction_norm + jnp.finfo(dtype).eps)
                )
                ratio_accepted = (
                    descent_ok
                    & (geodesic_acceptance_ratio > zero)
                    & jnp.all(jnp.isfinite(u_a_raw))
                    & jnp.all(jnp.isfinite(a_raw))
                    & (acceleration_ratio > zero)
                    & (acceleration_ratio <= geodesic_acceptance_ratio)
                )
                u_a = jnp.where(ratio_accepted, u_a_raw, jnp.zeros_like(u))
                a = jnp.where(ratio_accepted, a_raw, jnp.zeros_like(d))
                u_a_m = u_a[:n_m]
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
                    y_m=y_m,
                    u_m=u[:n_m],
                    u_a_m=u_a_m,
                    ridge=ridge,
                    resid_loss_old=resid_loss_old,
                    penalty_old=penalty_value_old,
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
                    jnp.asarray(NewtonStatus.NON_DESCENT, dtype=jnp.int32),
                    jnp.asarray(NewtonStatus.LINEAR_SOLVE_FAILED, dtype=jnp.int32),
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
            solver_cache = newton_state.solver_cache
            if solver_cache is not None:
                solver_cache = dataclasses.replace(
                    solver_cache, valid=jnp.asarray(False, dtype=jnp.bool_)
                )
            return (
                theta,
                nan,
                nan,
                nan,
                nan,
                jnp.asarray(False, dtype=jnp.bool_),
                zero,
                zero,
                jnp.asarray(0, dtype=jnp.int32),
                nan,
                zero,
                zero,
                nan,
                jnp.asarray(NewtonStatus.NONFINITE, dtype=jnp.int32),
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
            base_finite,
            _compute_step,
            _skip_step,
            operand=None,
        )

        instances = dict(
            ridge=ridge,
            metric=newton_state.metric,
            preconditioner=newton_state.preconditioner,
        )
        if self.cache_jacobian:
            new_state = NewtonState(
                resid=resid,
                Jt=Jt,
                jacobian_valid=cache_valid,
                aux=aux,
                hyper=newton_state.hyper,
                solver_cache=solver_cache,
                **instances,
            )
        else:
            new_state = NewtonState(hyper=newton_state.hyper, **instances)
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
                ridge=ridge,
                resid_loss=resid_loss,
                penalty_value=penalty_value,
                penalty_grad_norm=jnp.where(base_finite, penalty_grad_norm, nan),
                aux=aux,
            ),
        )

    def _validate_tolerances(self, atol, gtol, xtol):
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

    def _initial_ad_point(self, x, newton_state, args, p):
        return (
            x,
            args,
            p,
            newton_state.metric,
            newton_state.preconditioner,
            newton_state.ridge,
        )

    def _check_action_state(self, newton_state):
        if newton_state.ridge is None:
            raise ValueError(
                "the callback action returned a newton_state without ridge; "
                "use dataclasses.replace(ctx.newton_state, ...) to preserve it"
            )

    def _apply_action_state(self, newton_state, previous):
        new_ridge = jnp.asarray(newton_state.ridge, dtype=previous.ridge.dtype)
        changed = ~jnp.array_equal(new_ridge, previous.ridge, equal_nan=True)
        return dataclasses.replace(newton_state, ridge=new_ridge), changed

    def _converged(self, info, atol, gtol, xtol):
        stationarity_eligible = (info.step_status != NewtonStatus.NONFINITE) & (
            info.step_status != NewtonStatus.LINEAR_SOLVE_FAILED
        )
        gtol_met = stationarity_eligible & (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (
            (info.step_status == NewtonStatus.RUNNING)
            & (xtol > 0)
            & info.accepted
            & (info.step_norm < xtol)
        )
        residual_ok = (atol <= 0) | (jnp.sqrt(info.resid_loss) <= atol)
        return (gtol_met | xtol_met) & residual_ok

    def _cast_state(self, newton_state, dtype):
        updates = dict(
            ridge=jnp.asarray(newton_state.ridge, dtype=dtype),
            hyper=_cast_newton_hyper(newton_state.hyper, dtype),
        )
        if newton_state.solver_cache is not None:
            cache = newton_state.solver_cache
            updates["solver_cache"] = dataclasses.replace(
                cache, ridge=jnp.asarray(cache.ridge, dtype=dtype)
            )
        return dataclasses.replace(newton_state, **updates)

    def _ranking_objective(self, result, p, callback):
        if callback is None:
            loss = result.info.loss
        else:
            residual = self._residual_and_aux(result.x, result.args, p)[0]
            theta, _ = ravel_pytree(result.x)
            n_m = self._block_sizes(theta.shape[0])[0]
            resolved = self._resolved_state(result.newton_state)
            ctx = SolverContext(x=theta, lm_state=resolved, args=result.args, p=p)
            ridge = jnp.asarray(result.newton_state.ridge, dtype=residual.dtype)
            y_m = jnp.asarray(
                resolved.metric.factor_apply(theta[:n_m], ctx), dtype=residual.dtype
            )
            loss = jnp.sum(residual**2) + ridge * jnp.sum(y_m**2)
        return jnp.where(
            jnp.isfinite(loss), loss, jnp.asarray(jnp.inf, dtype=loss.dtype)
        )

    def _ad_x_tangent(self, x, args, p, p_dot, result, ad_success, initial_ad_point):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        final_ridge = jax.lax.stop_gradient(result.newton_state.ridge)
        initial_ridge = jax.lax.stop_gradient(initial_ad_point[5])
        ridge = jnp.where(
            ad_success, final_ridge, jnp.asarray(initial_ridge, final_ridge.dtype)
        )
        newton_state = jax.lax.stop_gradient(result.newton_state)
        initial_instances = jax.lax.stop_gradient(initial_ad_point[3:5])
        newton_state = dataclasses.replace(
            newton_state,
            metric=_where_tree(ad_success, newton_state.metric, initial_instances[0]),
            preconditioner=_where_tree(
                ad_success, newton_state.preconditioner, initial_instances[1]
            ),
        )
        use_cg = isinstance(self.ad_solver, CG) or (
            self.ad_solver is None and isinstance(self.linear_solver, CG)
        )
        if not use_cg:
            return self._ad_tangent_cholesky(x, args, p, p_dot, ridge, newton_state)
        return self._ad_tangent_normal_cg(x, args, p, p_dot, ridge, newton_state)

    def _ad_tangent_cholesky(self, x, args, p, p_dot, ridge, newton_state):
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        n_m, n_f = self._block_sizes(theta.shape[0])
        ctx = SolverContext(x=theta, lm_state=newton_state, args=args, p=p)
        Jt = self._assemble_jt(theta_jvp, theta, residual)
        ridge_typed = jnp.asarray(ridge, dtype=residual.dtype)
        Jt_sub = jnp.asarray(
            self._extended_solve_transpose(Jt, ctx), dtype=residual.dtype
        )
        if n_f == 0 and Jt_sub.shape[0] > Jt_sub.shape[1]:
            diag = jnp.arange(Jt_sub.shape[1])
            dual_matrix = mm(Jt_sub.T, Jt_sub).at[diag, diag].add(ridge_typed)
            factor = jsp_linalg.cho_factor(dual_matrix)
            y_dot = -mm(Jt_sub, jsp_linalg.cho_solve(factor, residual_p_dot))
        else:
            diag = jnp.arange(n_m)
            normal_matrix = mm(Jt_sub, Jt_sub.T).at[diag, diag].add(ridge_typed)
            factor = jsp_linalg.cho_factor(normal_matrix)
            y_dot = jsp_linalg.cho_solve(factor, -mm(Jt_sub, residual_p_dot))
        theta_dot = jnp.asarray(self._extended_solve(y_dot, ctx), residual.dtype)
        return unravel(theta_dot)

    def _ad_tangent_normal_cg(self, x, args, p, p_dot, ridge, newton_state):
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        n_m, n_f = self._block_sizes(theta.shape[0])
        ctx = SolverContext(x=theta, lm_state=newton_state, args=args, p=p)
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
        ad_preconditioner = self._ad_preconditioner(newton_state)
        if ad_preconditioner is None:
            apply_M = None
        else:
            zero_damping = jnp.zeros((), dtype=residual.dtype)

            def apply_M(v):
                return ad_preconditioner.apply(v, zero_damping, ctx)

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
