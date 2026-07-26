"""Metric-damped Levenberg-Marquardt for ``min ||r(x, args, p)||^2``.

Built for interpolation problems: zero-residual roots that stay rank-deficient
along some directions at every shape, so WHICH root the solver returns is part
of the contract rather than a tie-break. The optional
:class:`~nlls_gram.Metric` defines the damping geometry, and the small-damping
Gauss-Newton limit selects the minimum-metric-norm correction; the same
selection carries into the implicit derivative of ``solve(...).x`` with
respect to ``p``.

The ridge sibling, :class:`~nlls_gram.RidgeLevenbergMarquardt`, puts that
selection in the OBJECTIVE instead. Prefer it when the interpolant itself is
the deliverable; prefer this one when the root is a means to an end and
Euclidean damping is fine.
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
    LU,
    SVD,
    Cholesky,
    GramCG,
    Subproblem,
    _config_static_key,
)
from nlls_gram.lm_core import LevenbergMarquardtBase
from nlls_gram.lm_types import (
    LMInfo,
    LMState,
    SolverContext,
    _cast_hyper,
    _damping_floor,
)
from nlls_gram.metrics import Metric
from nlls_gram.utilities import (
    _static_key_component,
    _where_tree,
    _zero_tangent_leaf,
    canonicalize_residual,
    mm,
    register_pytree_dataclass,
)

__all__ = ["LevenbergMarquardt"]


@dataclass(frozen=True, eq=False)
class _EuclideanMetric(Metric):
    """The default metric: ``F = I`` over however many coordinates ``x``
    flattens to. ``size = 0`` puts everything in the free block, and the
    static unit ``free_scale`` folds the free-block scaling away."""

    size: int = 0
    free_scale: float = 1.0


register_pytree_dataclass(
    _EuclideanMetric, data_fields=(), meta_fields=("size", "free_scale")
)


class LevenbergMarquardt(LevenbergMarquardtBase):
    """Levenberg-Marquardt least squares over a JAX pytree ``x``.

    ``residual_fn`` takes ``(x)``, ``(x, args)``, or ``(x, args, p)`` and
    returns a residual pytree (or ``(residual, aux)`` with ``has_aux=True``).
    ``args`` is solver-inert auxiliary data; ``p`` is what ``solve(...).x``
    carries an implicit derivative with respect to.

    ``metric`` (default ``None`` -- Euclidean) is a
    :class:`~nlls_gram.Metric` covering the leading ``metric.size``
    coordinates of the flattened iterate, with the rest a free block. It
    defines the damping geometry: the subproblem is
    ``min ||r + J s||^2 + damping ||s||_W^2``, so the ``damping -> 0`` limit
    is the minimum-``W``-norm correction.

    ``linear_solver`` is a typed config -- :class:`~nlls_gram.Cholesky` (the
    default; ``form="auto"`` factors the smaller of the ``m x m`` dual and the
    ``n x n`` normal system), :class:`~nlls_gram.QR` (damping-row QR, stable
    at tiny damping and rank-safe), :class:`~nlls_gram.CG` (matrix-free in
    parameter space), or :class:`~nlls_gram.GramCG` (matrix-free in residual
    space, the ``m << n`` form). ``ad_solver`` takes ``None`` (match the
    forward family), ``Cholesky()``, :class:`~nlls_gram.SVD` (the
    pseudoinverse rule, for a rank-deficient undamped tangent), ``CG(...)``,
    or ``GramCG(...)``.

    ``init``/``update``/``solve``, the callback protocol, ``save_steps``,
    ``multi_start``, and implicit AD are shared with the ridge solver;
    ``info.loss`` here is the plain sum of squared residuals. Stopping is
    disjunctive: ``atol`` on the residual norm, ``gtol`` on the whitened
    stationarity ``info.grad_norm``, ``xtol`` on an accepted step's whitened
    norm; any one firing reports ``CONVERGED``.
    """

    def __init__(
        self,
        residual_fn,
        *,
        metric=None,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        linear_solver=Cholesky(),  # noqa: B008 -- frozen, immutable default
        jacobian_mode="auto",
        ad_solver=None,
        has_aux=False,
        cache_jacobian=True,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    ):
        canonical_residual, residual_arity = canonicalize_residual(residual_fn)
        if init_damping <= 0 or damping_decrease <= 0 or damping_increase <= 0:
            raise ValueError(
                "init_damping, damping_decrease, and damping_increase must be positive"
            )
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.initial_metric = _EuclideanMetric() if metric is None else metric
        self._check_registered_instance(self.initial_metric, "metric")
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.linear_solver = linear_solver
        self.jacobian_mode = jacobian_mode
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
                # preconditioner=None in the AD role inherits the CARRIED
                # forward instance at the solution (callback refreshes
                # included) while pinning the AD tolerance and budget.
                self.ad_solver_preconditioner = None
                if self.initial_preconditioner is None:
                    self._ad_preconditioner_source = "none"
                elif self.initial_preconditioner.requires_positive_damping:
                    raise ValueError(
                        "ad_solver preconditioner=None inherits the forward "
                        "preconditioner, but this one divides by the live "
                        "damping and cannot serve the undamped AD system"
                    )
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
            # ad_solver=None under a matrix-free forward hands the CARRIED
            # forward preconditioner to the undamped implicit solve: the AD
            # operator IS the forward operator at zero damping.
            # Damping-dividing hooks fall back to unpreconditioned.
            inherit = (
                ad_solver is None
                and krylov
                and not linear_solver.preconditioner.requires_positive_damping
            )
            self._ad_preconditioner_source = "carried" if inherit else "none"
            if inherit:
                self.ad_solver_penalty = getattr(linear_solver, "penalty", None)
        self.has_aux = has_aux
        # Only the dense paths materialize J', and the caches ride the same
        # reject-reuse lifecycle, so the flag is inert for the matrix-free forms.
        self.cache_jacobian = cache_jacobian and linear_solver.materializes_jacobian
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        # Metric and forward-preconditioner instances key by pytree structure:
        # their arrays are threaded through the carried state, so equal-config
        # fresh instances share one compiled loop. An explicit AD instance is
        # baked into the tangent program as constants, so it keys by identity.
        self._static_key = tuple(
            _static_key_component(value)
            for value in (
                residual_fn,
                jax.tree_util.tree_structure(self.initial_metric),
                init_damping,
                damping_decrease,
                damping_increase,
                _config_static_key(linear_solver, baked=False),
                jacobian_mode,
                _config_static_key(ad_solver, baked=True),
                has_aux,
                self.cache_jacobian,
                geodesic_acceleration,
                geodesic_acceptance_ratio,
            )
        )
        self._static_hash = hash(self._static_key)
        self._sealed = True

    def init(self, x0, args=None, *, p=None):
        """Build the initial :class:`~nlls_gram.LMState` at ``x0``.

        One residual evaluation types ``damping`` and sizes the Jacobian and
        linear-solver cache buffers. ``hyper`` stays ``None`` so manual
        ``update`` loops carry no extra buffers; ``solve`` populates it.
        """
        self._check_residual_args(args, p)
        residual, aux = self._residual_and_aux(x0, args, p)
        theta, _ = ravel_pytree(x0)
        n_m, _ = self._block_sizes(theta.size)
        dtype = residual.dtype
        damping = jnp.maximum(
            jnp.asarray(self.init_damping, dtype=dtype), _damping_floor(dtype)
        )
        instances = dict(
            metric=self.initial_metric, preconditioner=self.initial_preconditioner
        )
        if not self.cache_jacobian:
            return LMState(damping, **instances)
        return LMState(
            damping,
            resid=jnp.zeros(residual.shape, dtype=dtype),
            Jt=jnp.zeros((theta.size, residual.size), dtype=dtype),
            jacobian_valid=jnp.asarray(False, dtype=jnp.bool_),
            aux=jax.tree.map(jnp.zeros_like, aux),
            solver_cache=self.linear_solver.new_cache(
                residual.size, theta.size, n_m, dtype, False
            ),
            **instances,
        )

    def _solve_lm_state(self, x0, args, p, lm_state):
        if lm_state is not None:
            # A hand-built pre-seeding state enters the loop with the
            # constructor instances; the carry needs them present.
            return self._resolved_state(lm_state)
        if self.cache_jacobian:
            return self.init(x0, args, p=p)
        # Nothing needs sizing from a residual evaluation, so skip it: the
        # loop recasts the damping dtype itself.
        return LMState(
            jnp.asarray(self.init_damping),
            metric=self.initial_metric,
            preconditioner=self.initial_preconditioner,
        )

    def _initial_info(self, x, lm_state, args, p):
        # grad_norm is a +inf sentinel (computing it would cost a Jacobian
        # before the first step) and step_norm is zero; neither can satisfy
        # gtol/xtol before any update has run.
        residual, aux = self._residual_and_aux(x, args, p)
        loss = jnp.sum(residual**2)
        zero = jnp.zeros((), dtype=residual.dtype)
        return LMInfo(
            loss=loss,
            loss_old=loss,
            loss_candidate=loss,
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            damping=jnp.asarray(lm_state.damping, dtype=residual.dtype),
            damping_factor=jnp.ones((), dtype=residual.dtype),
            used_geodesic=jnp.asarray(False, dtype=jnp.bool_),
            acceleration_ratio=zero,
            grad_norm=jnp.asarray(jnp.inf, dtype=residual.dtype),
            step_norm=zero,
            aux=aux,
        )

    def update(self, x, lm_state, args=None, p=None):
        """One LM step: returns ``(x_new, lm_state, info)``."""
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
        damping_floor = _damping_floor(resid.dtype)
        damping = jnp.maximum(
            jnp.asarray(lm_state.damping, dtype=resid.dtype), damping_floor
        )

        n_m, n_f = self._block_sizes(theta.shape[0])
        ctx = SolverContext(
            x=theta, lm_state=self._resolved_state(lm_state), args=args, p=p
        )
        zero = jnp.zeros((), dtype=resid.dtype)
        step_solver = self.linear_solver.prepare(
            Subproblem(
                resid=resid,
                theta=theta,
                Jt=Jt,
                jvp_fn=jvp_fn,
                JT=JT,
                whiten=lambda v: self._extended_solve(v, ctx),
                whiten_transpose=lambda v: self._extended_solve_transpose(v, ctx),
                y_m=jnp.zeros(n_m, dtype=resid.dtype),
                penalty_gradient=jnp.zeros(theta.shape[0], dtype=resid.dtype),
                ridge=zero,
                damping=damping,
                n_m=n_m,
                n_f=n_f,
                cache=lm_state.solver_cache,
                cache_enabled=self.cache_jacobian,
                hyper=hyper,
                ctx=ctx,
                penalized=False,
            )
        )

        # The solves produce the whitened step; the x-space step maps back
        # through the factor solve.
        velocity_sub = step_solver.velocity()
        velocity = jnp.asarray(self._extended_solve(velocity_sub, ctx), resid.dtype)
        loss_old = jnp.sum(resid**2)
        resid_velocity = residual_value(theta + velocity)
        loss_velocity = jnp.sum(resid_velocity**2)

        # Geodesic second-order correction, sharing the factorization.
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
            # The ratio criterion lives in the damping geometry's norm -- the
            # whitened one.
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
            loss_accelerated = jax.lax.cond(
                ratio_accepted,
                lambda _: jnp.sum(residual_value(theta + accelerated_step) ** 2),
                lambda _: jnp.asarray(jnp.inf, dtype=resid.dtype),
                operand=None,
            )
            used_geodesic = ratio_accepted & (loss_accelerated <= loss_velocity)
            step = jnp.where(used_geodesic, accelerated_step, velocity)
            step_sub = jnp.where(
                used_geodesic, velocity_sub + 0.5 * acceleration_sub, velocity_sub
            )
            loss_candidate = jnp.where(used_geodesic, loss_accelerated, loss_velocity)
        else:
            step, step_sub = velocity, velocity_sub
            loss_candidate = loss_velocity
            used_geodesic = jnp.asarray(False)
            acceleration_ratio = zero

        improved = jnp.isfinite(loss_candidate) & (loss_candidate < loss_old)
        theta_new = jnp.where(improved, theta + step, theta)
        damping_factor = jnp.where(improved, damping_decrease, damping_increase)
        new_damping = jnp.maximum(damping * damping_factor, damping_floor)
        loss = jnp.where(improved, loss_candidate, loss_old)

        # The input state's instances pass through verbatim -- None stays
        # None, so a user's own loop around update keeps its carry structure.
        instances = dict(metric=lm_state.metric, preconditioner=lm_state.preconditioner)
        if self.cache_jacobian:
            new_lm_state = LMState(
                new_damping,
                resid=resid,
                Jt=Jt,
                jacobian_valid=~improved,
                aux=aux,
                hyper=lm_state.hyper,
                solver_cache=step_solver.make_cache(~improved),
                **instances,
            )
        else:
            new_lm_state = LMState(new_damping, hyper=lm_state.hyper, **instances)
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
                grad_norm=jnp.linalg.norm(step_solver.grad),
                step_norm=jnp.linalg.norm(step_sub),
                aux=aux,
            ),
        )

    def _converged(self, info, atol, gtol, xtol):
        atol_met = (atol > 0) & (jnp.sqrt(info.loss) < atol)
        gtol_met = (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (xtol > 0) & info.accepted & (info.step_norm < xtol)
        return atol_met | gtol_met | xtol_met

    def _cast_state(self, lm_state, dtype):
        return dataclasses.replace(
            lm_state,
            damping=jnp.asarray(lm_state.damping, dtype=dtype),
            hyper=_cast_hyper(lm_state.hyper, dtype),
        )

    def _ranking_objective(self, result, p, callback):
        # Without a callback info.loss already reports the objective at the
        # retained iterate; a callback can replace x/args after the last
        # update, so recompute. Nonfinite masks to +inf.
        if callback is None:
            loss = result.info.loss
        else:
            residual = self._residual_and_aux(result.x, result.args, p)[0]
            loss = jnp.sum(residual**2)
        return jnp.where(
            jnp.isfinite(loss), loss, jnp.asarray(jnp.inf, dtype=loss.dtype)
        )

    def _resolved_ad_solver(self, m, n):
        # The implicit-AD system is UNDAMPED, so the dual J~J~' is singular
        # whenever m > n and the normal J~'J~ whenever n > m. Each rule below
        # is only offered where its operator is invertible; SVD() covers the
        # cases where neither is (rank deficiency within the small side, as
        # padded zero residuals produce).
        resolved = self.ad_solver
        if resolved is None:
            # Match the forward family where its operator is invertible;
            # otherwise fall back to the assembled rule, which picks the
            # nonsingular side itself.
            if isinstance(self.linear_solver, GramCG) and m <= n:
                return self.linear_solver
            if isinstance(self.linear_solver, CG) and (
                n <= m or self.linear_solver.penalty is not None
            ):
                return self.linear_solver
            # Square: the tangent is a unique plain solve, so factor B itself
            # rather than squaring its condition number through B'B.
            if m == n:
                return LU()
            # Rectangular: the undamped system is singular whenever the small
            # side is rank deficient, which padded zero residuals produce by
            # construction, and the assembled Cholesky rules have no answer
            # there. SVD selects the minimum-metric-norm tangent at cond(B)
            # rather than cond(B)^2, so it is the safe default; Cholesky() is
            # the opt-in when the small side is known to have full rank.
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
        return resolved

    def _ad_x_tangent(self, x, args, p, p_dot, result, ad_success, initial_ad_point):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        # The carried instances are frozen conditioning data at the solution;
        # a failed lane reads the differentiation-inert pre-loop instances
        # instead (a callback may have left invalid arrays behind).
        lm_state = jax.lax.stop_gradient(result.lm_state)
        initial_instances = jax.lax.stop_gradient(initial_ad_point[3:5])
        lm_state = dataclasses.replace(
            lm_state,
            metric=_where_tree(ad_success, lm_state.metric, initial_instances[0]),
            preconditioner=_where_tree(
                ad_success, lm_state.preconditioner, initial_instances[1]
            ),
        )
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        resolved = self._resolved_ad_solver(residual.shape[0], theta.shape[0])
        if isinstance(resolved, LU):
            # Square: J theta_dot = -dr/dp p_dot has a unique solution, so the
            # metric selects nothing and the whitening round-trip below is
            # avoidable work -- solve the unwhitened system directly. This is
            # the same uniqueness that makes the plain factorization valid.
            Jt = self._assemble_jt(theta_jvp, theta, residual)
            return unravel(self._ad_tangent_lu(Jt, residual_p_dot))
        ctx = SolverContext(x=theta, lm_state=lm_state, args=args, p=p)
        n_m, n_f = self._block_sizes(theta.shape[0])
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
            )
        else:
            # B' = F_bar^{-T} J', shape (n, m).
            Bt = whiten_transpose(self._assemble_jt(theta_jvp, theta, residual))
            u = (
                self._ad_tangent_svd(Bt, residual_p_dot)
                if isinstance(resolved, SVD)
                else self._ad_tangent_dense(Bt, residual_p_dot)
            )
        # S = F_bar^{-1} is not self-adjoint, and a matrix-free factor may
        # be opaque to JAX's transpose machinery, so declare its transpose
        # explicitly: the identity matvec exposes nothing to AD and every rule
        # routes through the declared solves. That keeps reverse mode working
        # through a metric JAX could not transpose on its own.
        theta_dot = jax.lax.custom_linear_solve(
            lambda v: v,
            u,
            lambda _, b: whiten(b),
            transpose_solve=lambda _, b: whiten_transpose(b),
        )
        return unravel(theta_dot)

    def _ad_tangent_dense(self, Bt, residual_p_dot):
        # Undamped Gauss-Newton tangent: u = -B^+ (dr/dp) p_dot through the
        # smaller of the two normal systems. Requires full rank; a rank-
        # deficient B needs SVD(), which selects the minimum-norm tangent.
        n, m = Bt.shape
        if n > m:
            factor = jsp_linalg.cho_factor(mm(Bt.T, Bt))
            return -mm(Bt, jsp_linalg.cho_solve(factor, residual_p_dot))
        factor = jsp_linalg.cho_factor(mm(Bt, Bt.T))
        return -jsp_linalg.cho_solve(factor, mm(Bt, residual_p_dot))

    def _ad_tangent_lu(self, Bt, residual_p_dot):
        # Square B: u = -B^{-1} (dr/dp) p_dot, factored directly. The normal
        # form (B'B)^{-1}B' is algebraically the same map here but numerically
        # worse, at cond(B)^2. Reverse mode transposes this solve, which is the
        # same factorization applied to B'.
        return -jnp.linalg.solve(Bt.T, residual_p_dot)

    def _ad_tangent_svd(self, Bt, residual_p_dot):
        # Spectral filter: u = -B^+ (dr/dp) p_dot, the minimum-metric-norm
        # tangent. This is the rule for the singular undamped systems that
        # padded zero residuals produce by construction, where the dense and
        # QR rules have no answer to give. The factors are constants in the
        # tangent program, so the map stays linear in residual_p_dot.
        U, sigma, Vt = jnp.linalg.svd(Bt.T, full_matrices=False)
        cutoff = max(Bt.shape) * jnp.finfo(Bt.dtype).eps * sigma[0]
        inverted = jnp.where(sigma > cutoff, 1.0 / jnp.maximum(sigma, cutoff), 0.0)
        return -mm(Vt.T, inverted * mm(U.T, residual_p_dot))

    def _ad_tangent_krylov(
        self, config, theta, theta_jvp, residual_p_dot, whiten, whiten_transpose, ctx
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
            # The AD system is undamped, so the preconditioner sees zero
            # damping (requires_positive_damping hooks were rejected).
            def apply_M(v):
                return ad_preconditioner.apply(v, zero_damping, ctx)

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
            # Dual: (B B') y = (dr/dp) p_dot, then u = -B' y. Selection is
            # safe under any preconditioner here -- u = -B'y is invariant to
            # the null(B') component of y -- unlike the normal form, where the
            # preconditioner must preserve range(B').
            def dual_matvec(y):
                return B(Bt(y))

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
            return value

        if penalty is not None:
            # N = B'B + penalty I is SPD, so CG converges for any right-hand
            # side and the symmetric operator is its own transpose.
            transpose_solve = lambda _, c: cg(normal_matvec, c)  # noqa: E731
        else:
            # N = B'B is SINGULAR whenever B is (always, on the
            # underdetermined problems this solver targets). The forward
            # right-hand side lies in range(B') so CG converges there, but a
            # reverse-mode cotangent does not, and plain CG on it diverges.
            # Route the transpose through the push-through identity
            # N^+ = B' (BB')^{+2} B instead: each dual solve sees a right-hand
            # side in range(B), where BB' is invertible.
            def transpose_solve(_, c):
                return Bt(dual_solve(dual_solve(B(c))))

            def dual_solve(y):
                # UNPRECONDITIONED: this solve is posed on residual-space
                # m-vectors, while self.ad_solver_preconditioner is the
                # parameter-space one CG's own operator takes. Handing it an
                # m-vector is a shape error at best and a wrong tangent at
                # worst.
                return cg(lambda w: B(Bt(w)), y, preconditioner=None)

        rhs = -Bt(residual_p_dot)
        return jax.lax.custom_linear_solve(
            normal_matvec,
            rhs,
            lambda _, c: cg(normal_matvec, c),
            transpose_solve=transpose_solve,
        )
