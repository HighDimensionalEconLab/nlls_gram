import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from nlls_gram.linear_solvers import (
    CG,
    LU,
    QR,
    SVD,
    Cholesky,
    GramCG,
    Subproblem,
    _config_static_key,
)
from nlls_gram.metrics import IdentityRidge, RidgeMetric
from nlls_gram.multi_start import (
    MultiStart,
    _accept_converged,
    _accept_converged_or_max_steps,
    _check_hook_types,
    _multi_start_parallel_jit,
    _multi_start_python_impl,
    _multi_start_sequential_jit,
)
from nlls_gram.preconditioners import (
    DiagonalPreconditioner,
    Preconditioner,
)
from nlls_gram.solve_loop import _solve_loop_jit, _solve_python_impl
from nlls_gram.types import (
    PDAS,
    Complementarity,
    FischerBurmeister,
    Inequality,
    InequalityPenalty,
    Minimum,
    Residuals,
    SolverAction,
    SolverContext,
    SolverStatus,
)
from nlls_gram.utilities import (
    _hashable_hook,
    _mask_tangent_tree,
    _static_key_component,
    _tree_changed,
    _where_tree,
    _zero_tangent_leaf,
    canonicalize_residual,
)


class _ADKrylovHyper(NamedTuple):
    iterative_tol: jax.Array
    iterative_atol: jax.Array
    iterative_maxiter: int | None


@functools.partial(jax.jit, static_argnums=0)
def _init_residual_and_aux(solver, x, args, p):
    return solver._eval_residual_and_aux(x, args, p)


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
    ridge=None,
    ridge_metric=None,
    ctx=None,
    resid_loss_old=None,
    penalty_old=None,
    penalty_grad=None,
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
        p_c1 = p_c2 = p_c3 = p_c4 = zero
    else:
        d_safe = jnp.where(active, d, zero)
        pg_safe = jnp.where(active, penalty_grad, zero)
        p_c1 = 2.0 * jnp.sum(d_safe * pg_safe)
        if a is None:
            p_c2 = jnp.asarray(ridge_metric.quad_form(d_safe, ctx), dtype=dtype)
            p_c3 = p_c4 = zero
        elif ridge_metric.has_factor:
            a_safe = jnp.where(active, a, zero)
            u_d = jnp.asarray(ridge_metric.factor_apply(d_safe, ctx), dtype=dtype)
            u_a = jnp.asarray(ridge_metric.factor_apply(a_safe, ctx), dtype=dtype)
            p_c2 = jnp.sum(a_safe * pg_safe) + jnp.sum(u_d**2)
            p_c3 = jnp.sum(u_d * u_a)
            p_c4 = 0.25 * jnp.sum(u_a**2)
        else:
            a_safe = jnp.where(active, a, zero)
            qd = jnp.asarray(ridge_metric.matvec(d_safe, ctx), dtype=dtype)
            p_c2 = jnp.sum(a_safe * pg_safe) + jnp.sum(d_safe * qd)
            p_c3 = jnp.sum(a_safe * qd)
            p_c4 = 0.25 * jnp.asarray(ridge_metric.quad_form(a_safe, ctx), dtype=dtype)

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
        rhs_shifted = resid_loss_old + 2.0 * c1 * alpha * slope + 2.0 * linesearch_tol
        if penalized:
            alpha2 = alpha**2
            delta_penalty = (
                alpha * p_c1
                + alpha2 * p_c2
                + (alpha * alpha2) * p_c3
                + (alpha2**2) * p_c4
            )
            cand_penalty = jnp.maximum(zero, penalty_old + delta_penalty)
            cand_loss = cand_resid_loss + ridge * cand_penalty
            lhs_shifted = cand_resid_loss + ridge * delta_penalty
        else:
            cand_penalty = zero
            cand_loss = cand_resid_loss
            lhs_shifted = cand_resid_loss
        trial_finite = (
            cand_finite
            & jnp.all(jnp.isfinite(cand_resid))
            & jnp.isfinite(cand_resid_loss)
            & jnp.isfinite(cand_loss)
            & jnp.isfinite(lhs_shifted)
        )
        ok = (~same_theta) & trial_finite & (lhs_shifted <= rhs_shifted)
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
        jnp.asarray(SolverStatus.RUNNING, dtype=jnp.int32),
        jnp.where(
            stagnated,
            jnp.asarray(SolverStatus.STAGNATION, dtype=jnp.int32),
            jnp.asarray(SolverStatus.LINE_SEARCH_FAILED, dtype=jnp.int32),
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


def _validate_ridge_value(value, name="ridge"):
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive scalar float, not bool "
            f"({name} = 0 is unsupported)"
        )
    arr = jnp.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be a scalar (0-D), got shape {arr.shape}")
    if not jnp.issubdtype(arr.dtype, jnp.floating) and not jnp.issubdtype(
        arr.dtype, jnp.integer
    ):
        raise TypeError(f"{name} must be a real scalar, got dtype {arr.dtype}")
    if not isinstance(value, jax.core.Tracer) and not (
        np.isfinite(float(value)) and float(value) > 0.0
    ):
        raise ValueError(
            f"{name} must be strictly positive ({name} = 0 is unsupported)"
        )


def _validate_inequality_config(cfg):
    if cfg is None:
        return
    if not isinstance(cfg, InequalityPenalty):
        raise TypeError(
            f"inequality must be an InequalityPenalty or None, got {type(cfg).__name__}"
        )
    w = jnp.asarray(cfg.weight)
    if not isinstance(cfg.weight, jax.core.Tracer):
        w_np = np.asarray(w)
        if not np.all(np.isfinite(w_np)) or np.any(w_np < 0.0):
            raise ValueError("InequalityPenalty.weight must be finite and nonnegative")
    if cfg.smoothing < 0.0 or cfg.smoothing_min < 0.0:
        raise ValueError(
            "InequalityPenalty smoothing and smoothing_min must be nonnegative"
        )
    if not (0.0 < cfg.smoothing_decay <= 1.0):
        raise ValueError("InequalityPenalty.smoothing_decay must lie in (0, 1]")


class SolverBase:
    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._static_key == other._static_key

    def __hash__(self):
        return self._static_hash

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError(
                f"{type(self).__name__} is frozen after construction; build a "
                "new solver instead of assigning attributes"
            )
        object.__setattr__(self, name, value)

    penalized = False
    _forward_role_attr = "supports_forward"

    def _init_common(
        self,
        residual_fn,
        *,
        ridge,
        ridge_metric,
        linear_solver,
        ad_solver,
        jacobian_mode,
        jacobian_chunk,
        has_aux,
        cache_jacobian,
        geodesic_acceleration,
        geodesic_acceptance_ratio,
        complementarity=None,
        inequality=None,
    ):
        penalized = ridge is not None
        if not penalized and ridge_metric is not None:
            raise ValueError(
                "ridge_metric requires a positive ridge; outer steps use "
                "Euclidean geometry and do not accept a step metric"
            )
        if penalized:
            _validate_ridge_value(ridge, "ridge")
            if ridge_metric is None:
                resolved_rm = IdentityRidge()
            else:
                if not isinstance(ridge_metric, RidgeMetric):
                    raise TypeError(
                        "ridge_metric must be a RidgeMetric instance, got "
                        f"{type(ridge_metric).__name__}"
                    )
                self._check_registered_instance(ridge_metric, "ridge_metric")
                resolved_rm = ridge_metric
        else:
            resolved_rm = None

        _validate_inequality_config(inequality)

        self._raw_residual_fn = residual_fn
        self.residual_fn, self.residual_arity = canonicalize_residual(residual_fn)
        self.has_aux = has_aux
        self.linear_solver = linear_solver
        self.penalized = penalized
        self.initial_ridge = ridge
        self.initial_ridge_metric = resolved_rm
        self.cache_jacobian = cache_jacobian
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        self.ad_solver = ad_solver
        self.jacobian_mode = jacobian_mode
        self.jacobian_chunk = jacobian_chunk
        self.complementarity = complementarity
        self.inequality = inequality

        comp_cfg = PDAS() if complementarity is None else complementarity
        self._resolved_complementarity = comp_cfg
        self._uses_homotopy_decay = (
            isinstance(comp_cfg, PDAS)
            and comp_cfg.tau_init > 0
            and comp_cfg.tau_decay < 1.0
        ) or (
            isinstance(comp_cfg, FischerBurmeister)
            and comp_cfg.mu_init > 0
            and comp_cfg.mu_decay < 1.0
        )
        ineq_cfg = InequalityPenalty() if inequality is None else inequality
        self._resolved_inequality = ineq_cfg
        self._uses_ineq_decay = (
            inequality is not None
            and ineq_cfg.smoothing > 0.0
            and ineq_cfg.smoothing_decay < 1.0
        )

        self._validate_configuration(linear_solver, ad_solver, penalized=penalized)

        krylov = isinstance(linear_solver, (CG, GramCG))
        if krylov:
            if linear_solver.preconditioner is None:
                raise ValueError(
                    "the forward linear_solver requires a preconditioner; "
                    "pass IdentityPreconditioner() for unpreconditioned Krylov "
                    "(preconditioner=None is legal only in ad_solver)"
                )
            if not isinstance(linear_solver.preconditioner, Preconditioner):
                raise TypeError(
                    "linear_solver.preconditioner must be a Preconditioner instance"
                )
            if (
                isinstance(linear_solver.preconditioner, DiagonalPreconditioner)
                and linear_solver.preconditioner.diagonal is None
            ):
                raise ValueError(
                    "DiagonalPreconditioner(diagonal=None) automatic equilibration "
                    "requires a dense solver (Cholesky or QR); pass an explicit "
                    "positive diagonal array for CG or GramCG"
                )
            self._check_registered_instance(
                linear_solver.preconditioner, "linear_solver.preconditioner"
            )
            self.initial_preconditioner = linear_solver.preconditioner
            self.iterative_tol = linear_solver.tol
            self.iterative_atol = linear_solver.atol
            self.iterative_maxiter = linear_solver.maxiter
        else:
            dense_precond = getattr(linear_solver, "preconditioner", None)
            if dense_precond is not None:
                self._check_registered_instance(
                    dense_precond, "linear_solver.preconditioner"
                )
            self.initial_preconditioner = dense_precond
            self.iterative_tol = 0.0
            self.iterative_atol = 0.0
            self.iterative_maxiter = 8

        if isinstance(ad_solver, (CG, GramCG)):
            self.ad_solver_tol = ad_solver.tol
            self.ad_solver_atol = ad_solver.atol
            self.ad_solver_maxiter = ad_solver.maxiter
            if ad_solver.preconditioner is None:
                self.ad_solver_preconditioner = None
                if self.initial_preconditioner is None:
                    self._ad_preconditioner_source = "none"
                elif type(ad_solver) is not type(linear_solver):
                    raise ValueError(
                        "ad_solver preconditioner=None cannot inherit from "
                        f"linear_solver={type(linear_solver).__name__} because "
                        f"{type(ad_solver).__name__} operates in a different space; "
                        "pass an explicit preconditioner "
                        "(e.g. IdentityPreconditioner())"
                    )
                elif self.initial_preconditioner.requires_positive_damping:
                    raise ValueError(
                        "ad_solver preconditioner=None inherits the forward "
                        "preconditioner, which requires positive damping and "
                        "cannot serve the undamped AD system"
                    )
                else:
                    self._ad_preconditioner_source = "carried"
            else:
                if not isinstance(ad_solver.preconditioner, Preconditioner):
                    raise TypeError(
                        "ad_solver.preconditioner must be a Preconditioner instance"
                    )
                if (
                    isinstance(ad_solver.preconditioner, DiagonalPreconditioner)
                    and ad_solver.preconditioner.diagonal is None
                ):
                    raise ValueError(
                        "DiagonalPreconditioner(diagonal=None) requires Cholesky or QR"
                    )
                if ad_solver.preconditioner.requires_positive_damping:
                    raise ValueError(
                        "ad_solver.preconditioner requires positive damping and "
                        "cannot serve the undamped AD system"
                    )
                self._check_registered_instance(
                    ad_solver.preconditioner, "ad_solver.preconditioner"
                )
                self.ad_solver_preconditioner = ad_solver.preconditioner
                self._ad_preconditioner_source = "explicit"
        else:
            self.ad_solver_tol = (
                linear_solver.tol if (ad_solver is None and krylov) else None
            )
            self.ad_solver_atol = (
                linear_solver.atol if (ad_solver is None and krylov) else 0.0
            )
            self.ad_solver_maxiter = (
                linear_solver.maxiter if (ad_solver is None and krylov) else None
            )
            ad_precond = getattr(ad_solver, "preconditioner", None)
            if ad_precond is not None:
                if ad_precond.requires_positive_damping:
                    raise ValueError(
                        "ad_solver.preconditioner requires positive damping and "
                        "cannot serve the undamped AD system"
                    )
                self._check_registered_instance(ad_precond, "ad_solver.preconditioner")
                self.ad_solver_preconditioner = ad_precond
                self._ad_preconditioner_source = "explicit"
            else:
                self.ad_solver_preconditioner = None
                inherit = (
                    ad_solver is None
                    and self.initial_preconditioner is not None
                    and not self.initial_preconditioner.requires_positive_damping
                )
                self._ad_preconditioner_source = "carried" if inherit else "none"

        self.cache_jacobian = cache_jacobian and linear_solver.materializes_jacobian

    def _dataclass_static_key(self, cfg):
        if cfg is None:
            return None
        return (
            type(cfg),
            *(
                _static_key_component(getattr(cfg, f.name))
                for f in dataclasses.fields(cfg)
            ),
        )

    def _initial_homotopy(self, dtype=None):
        if isinstance(self.complementarity, PDAS):
            return jnp.asarray(self.complementarity.tau_init, dtype=dtype)
        if isinstance(self.complementarity, FischerBurmeister):
            return jnp.asarray(self.complementarity.mu_init, dtype=dtype)
        return None

    def _initial_ineq_state(self, dtype=None):
        if self.inequality is None:
            return None, None
        return (
            jnp.asarray(self.inequality.weight, dtype=dtype),
            jnp.asarray(self.inequality.smoothing, dtype=dtype),
        )

    def _common_static_key(self):
        return (
            _static_key_component(self._raw_residual_fn),
            self.residual_arity,
            self.has_aux,
            self.penalized,
            _config_static_key(self.linear_solver, baked=False),
            None
            if not self.penalized
            else jax.tree_util.tree_structure(self.initial_ridge_metric),
            self.cache_jacobian,
            self.geodesic_acceleration,
            _config_static_key(self.ad_solver, baked=True),
            self.jacobian_mode,
            self.jacobian_chunk,
            self._dataclass_static_key(self.complementarity),
            self._dataclass_static_key(self.inequality),
        )

    def _check_registered_instance(self, instance, keyword):
        if instance is None:
            return
        if jax.tree_util.all_leaves([instance]):
            raise TypeError(
                f"{keyword} must be a registered pytree ({type(instance).__name__} "
                "flattens as a leaf); register the class with "
                "nlls_gram.register_pytree_dataclass"
            )
        for leaf in jax.tree_util.tree_leaves(instance):
            if not isinstance(
                leaf, (jax.Array, np.ndarray, np.generic, int, float, complex, bool)
            ):
                raise TypeError(
                    f"{keyword} contains a non-array leaf of type "
                    f"{type(leaf).__name__}; register nested classes with "
                    "nlls_gram.register_pytree_dataclass"
                )

    def _validate_configuration(self, linear_solver, ad_solver, penalized):
        roles = [(linear_solver, "linear_solver", self._forward_role_attr)]
        if ad_solver is not None:
            roles.append((ad_solver, "ad_solver", "supports_ad"))
        for config, keyword, attribute in roles:
            if not getattr(config, attribute):
                raise ValueError(
                    f"{config!r} cannot serve as {keyword}: it does not "
                    "implement that role"
                )
            if penalized and not config.supports_penalty:
                raise ValueError(f"{config!r} cannot serve as {keyword} when ridge > 0")
        if self.jacobian_mode not in ("auto", "fwd", "rev"):
            raise ValueError(f"unknown jacobian_mode: {self.jacobian_mode}")
        if self.jacobian_chunk is not None and self.jacobian_chunk <= 0:
            raise ValueError("jacobian_chunk must be a positive int or None")

    def _resolve_ridge(self, dtype, ridge=None):
        source = self.initial_ridge if ridge is None else ridge
        if source is None:
            raise ValueError(
                "ridge was passed to init() on an unpenalized solver, or ridge is None"
            )
        _validate_ridge_value(source, "ridge")
        return jnp.asarray(source, dtype=dtype)

    def _cold_state(self, state):
        updates = {}
        if state.jacobian_valid is not None:
            updates["jacobian_valid"] = jnp.zeros_like(state.jacobian_valid)
        if state.solver_cache is not None:
            updates["solver_cache"] = jax.tree.map(jnp.zeros_like, state.solver_cache)
        return dataclasses.replace(state, **updates) if updates else state

    def _resolve_jacobian_mode(self, m, n):
        if self.jacobian_mode != "auto":
            return self.jacobian_mode
        return "fwd" if n <= m else "rev"

    def _assemble_jt(self, jvp_fn, theta, resid):
        def over_basis(fn, size, dtype):
            if self.jacobian_chunk is None or self.jacobian_chunk >= size:
                return jax.vmap(fn)(jnp.eye(size, dtype=dtype))

            def eval_idx(idx):
                basis_vec = (jnp.arange(size) == idx).astype(dtype)
                return fn(basis_vec)

            return jax.lax.map(
                eval_idx,
                jnp.arange(size, dtype=jnp.int32),
                batch_size=self.jacobian_chunk,
            )

        if self._resolve_jacobian_mode(resid.shape[0], theta.shape[0]) == "fwd":
            return over_basis(jvp_fn, theta.shape[0], theta.dtype)
        transpose_fn = jax.linear_transpose(jvp_fn, theta)
        return over_basis(
            lambda cotangent: transpose_fn(cotangent)[0],
            resid.shape[0],
            resid.dtype,
        ).T

    def _dense_resid_jt_aux(self, eval_base, theta):
        resid, jvp_fn, (aux, meta) = jax.linearize(eval_base, theta, has_aux=True)
        return resid, self._assemble_jt(jvp_fn, theta, resid), aux, meta

    def _eval_residual_and_aux(
        self,
        x,
        args,
        p,
        homotopy=None,
        ineq_weight=None,
        ineq_smoothing=None,
        frozen_mask=None,
        detach_mask=True,
    ):
        if self.has_aux:
            value, aux = self.residual_fn(x, args, p)
        else:
            value, aux = self.residual_fn(x, args, p), None

        if isinstance(value, Inequality):
            res_tree, ineq_tree, comp = None, value.value, None
        elif isinstance(value, Complementarity):
            res_tree, ineq_tree, comp = None, None, value
        elif isinstance(value, Residuals):
            res_tree = value.residual
            ineq_raw = value.inequality
            ineq_tree = ineq_raw.value if isinstance(ineq_raw, Inequality) else ineq_raw
            comp = value.complementarity
            if comp is not None and not isinstance(comp, Complementarity):
                raise TypeError(
                    "Residuals.complementarity must be a Complementarity(lhs, rhs) "
                    "instance"
                )
            if res_tree is None and ineq_tree is None and comp is None:
                raise ValueError(
                    "Residuals must provide at least one of residual, inequality, "
                    "or complementarity"
                )
        else:
            res_tree, ineq_tree, comp = value, None, None

        r_eq = None if res_tree is None else ravel_pytree(res_tree)[0]
        if ineq_tree is None and comp is None:
            return r_eq, aux, (None, None)

        dtypes = [float]
        if r_eq is not None:
            dtypes.append(r_eq.dtype)
        if comp is not None:
            lhs_leaves, lhs_def = jax.tree_util.tree_flatten(comp.lhs)
            rhs_leaves, rhs_def = jax.tree_util.tree_flatten(comp.rhs)
            if lhs_def != rhs_def or [jnp.shape(a) for a in lhs_leaves] != [
                jnp.shape(b) for b in rhs_leaves
            ]:
                raise ValueError(
                    "Complementarity lhs and rhs must have matching pytree "
                    "structure and shapes"
                )
            lhs = ravel_pytree(comp.lhs)[0]
            rhs = ravel_pytree(comp.rhs)[0]
            dtypes.extend([lhs.dtype, rhs.dtype])
        else:
            lhs = rhs = None
        if ineq_tree is not None:
            g_raw = ravel_pytree(ineq_tree)[0]
            dtypes.append(g_raw.dtype)
        else:
            g_raw = None

        dtype = jnp.result_type(*dtypes)
        blocks = []
        if r_eq is not None:
            r_eq = jnp.asarray(r_eq, dtype=dtype)
            blocks.append(r_eq)

        mask = None
        if comp is not None:
            lhs = jnp.asarray(lhs, dtype=dtype)
            rhs = jnp.asarray(rhs, dtype=dtype)
            cfg = self._resolved_complementarity
            c_scale = jnp.asarray(cfg.c_scale, dtype=dtype)
            w_comp = jnp.asarray(cfg.w_comp, dtype=dtype)
            a_tilde = lhs / c_scale
            b_tilde = rhs

            if isinstance(cfg, Minimum):
                r_comp = w_comp * jnp.where(a_tilde <= b_tilde, a_tilde, b_tilde)
            elif isinstance(cfg, PDAS):
                tau = (
                    jnp.asarray(cfg.tau_init, dtype=dtype)
                    if homotopy is None
                    else jnp.asarray(homotopy, dtype=dtype)
                )
                if frozen_mask is None:
                    diff = b_tilde - a_tilde
                    hard_mask = jax.lax.stop_gradient(
                        (a_tilde <= b_tilde).astype(dtype)
                    )
                    safe_tau = jnp.where(tau > 0, tau, jnp.ones_like(tau))
                    soft_mask = jax.nn.sigmoid(diff / safe_tau)
                    if detach_mask:
                        soft_mask = jax.lax.stop_gradient(soft_mask)
                    mask = jnp.where(tau > 0, soft_mask, hard_mask)
                else:
                    mask = jax.lax.stop_gradient(jnp.asarray(frozen_mask, dtype=dtype))
                hard_branch = jnp.where(mask >= 0.5, a_tilde, b_tilde)
                soft_branch = mask * a_tilde + (1.0 - mask) * b_tilde
                r_comp = w_comp * jnp.where(tau > 0, soft_branch, hard_branch)
            else:
                mu = (
                    jnp.asarray(cfg.mu_init, dtype=dtype)
                    if homotopy is None
                    else jnp.asarray(homotopy, dtype=dtype)
                )
                eps_safe = (
                    jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
                    if cfg.eps_safe is None
                    else jnp.asarray(cfg.eps_safe, dtype=dtype)
                )
                q = 2.0 * jnp.square(mu) + eps_safe
                scale = jax.lax.stop_gradient(
                    jnp.maximum(
                        jnp.maximum(jnp.abs(a_tilde), jnp.abs(b_tilde)), jnp.sqrt(q)
                    )
                )
                safe_scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
                a_hat = a_tilde / safe_scale
                b_hat = b_tilde / safe_scale
                q_hat = q / jnp.square(safe_scale)
                s_hat = jnp.sqrt(jnp.square(a_hat) + jnp.square(b_hat) + q_hat)
                apb_hat = a_hat + b_hat
                pos = apb_hat > 0
                safe_denom = jnp.where(pos, apb_hat + s_hat, jnp.ones_like(s_hat))
                fb_hat = jnp.where(
                    pos,
                    (2.0 * a_hat * b_hat - q_hat) / safe_denom,
                    apb_hat - s_hat,
                )
                r_comp = w_comp * safe_scale * fb_hat
            blocks.append(r_comp)

        ineq_metrics = None
        if g_raw is not None:
            g = jnp.asarray(g_raw, dtype=dtype)
            zero = jnp.zeros((), dtype=dtype)
            eq_loss = jnp.sum(r_eq**2) if r_eq is not None else zero
            viol = jnp.maximum(-g, jnp.zeros_like(g))
            viol_norm = jnp.linalg.norm(viol)
            max_viol = jnp.max(viol, initial=zero)
            ineq_metrics = (
                jax.lax.stop_gradient(eq_loss),
                jax.lax.stop_gradient(viol_norm),
                jax.lax.stop_gradient(max_viol),
            )

            icfg = self._resolved_inequality
            w = (
                jnp.asarray(icfg.weight, dtype=dtype)
                if ineq_weight is None
                else jnp.asarray(ineq_weight, dtype=dtype)
            )
            if w.ndim > 0 and w.shape != g.shape:
                raise ValueError(
                    f"InequalityPenalty weight shape {w.shape} does not match "
                    f"inequality margin shape {g.shape}"
                )
            tau_i = (
                jnp.asarray(icfg.smoothing, dtype=dtype)
                if ineq_smoothing is None
                else jnp.asarray(ineq_smoothing, dtype=dtype)
            )
            neg = g < 0
            psi_exact = jnp.where(neg, g, jnp.zeros_like(g))
            tau_pos = tau_i > 0
            safe_tau_i = jnp.where(tau_pos, tau_i, jnp.ones_like(tau_i))
            g_neg = jnp.where(tau_pos & neg, g, jnp.zeros_like(g))
            g_pos = jnp.where(tau_pos & (~neg), g, jnp.zeros_like(g))
            psi_smooth = jnp.where(
                neg,
                g_neg - safe_tau_i * jax.nn.softplus(g_neg / safe_tau_i),
                -safe_tau_i * jax.nn.softplus(-g_pos / safe_tau_i),
            )
            psi = jnp.where(tau_pos, psi_smooth, psi_exact)

            w_pos = w > 0
            sqrt_w = jnp.where(
                w_pos,
                jnp.sqrt(jnp.where(w_pos, w, jnp.ones_like(w))),
                jnp.zeros_like(w),
            )
            blocks.append(sqrt_w * psi)

        resid = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=0)
        return resid, aux, (mask, ineq_metrics)

    def _residual_and_aux(
        self,
        x,
        args,
        p,
        homotopy=None,
        ineq_weight=None,
        ineq_smoothing=None,
        frozen_mask=None,
        detach_mask=True,
    ):
        resid, aux, _ = self._eval_residual_and_aux(
            x,
            args,
            p,
            homotopy=homotopy,
            ineq_weight=ineq_weight,
            ineq_smoothing=ineq_smoothing,
            frozen_mask=frozen_mask,
            detach_mask=detach_mask,
        )
        return resid, aux

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

    def _linearize_step(self, theta, unravel, state, args, p):
        homotopy = state.homotopy
        ineq_weight = state.ineq_weight
        ineq_smoothing = state.ineq_smoothing

        def eval_base(th):
            resid, aux, meta = self._eval_residual_and_aux(
                unravel(th),
                args,
                p,
                homotopy=homotopy,
                ineq_weight=ineq_weight,
                ineq_smoothing=ineq_smoothing,
                frozen_mask=None,
            )
            return resid, (aux, meta)

        jvp_fn = JT = Jt = None
        if not self.linear_solver.materializes_jacobian:
            resid, jvp_fn, (aux, base_meta) = jax.linearize(
                eval_base, theta, has_aux=True
            )
            transpose_fn = jax.linear_transpose(jvp_fn, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

        elif self.cache_jacobian:
            if state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True requires a state initialized with "
                    "cache_jacobian=True; call solver.init(x, args, p=p)"
                )
            meta_shapes = jax.eval_shape(eval_base, theta)[1][1]
            need_meta = meta_shapes[0] is not None or meta_shapes[1] is not None

            def fresh_dense(_):
                return self._dense_resid_jt_aux(eval_base, theta)

            def cached_dense(_):
                meta = (
                    self._eval_residual_and_aux(
                        unravel(theta),
                        args,
                        p,
                        homotopy=homotopy,
                        ineq_weight=ineq_weight,
                        ineq_smoothing=ineq_smoothing,
                        frozen_mask=None,
                    )[2]
                    if need_meta
                    else (None, None)
                )
                return state.resid, state.Jt, state.aux, meta

            resid, Jt, aux, base_meta = jax.lax.cond(
                ~state.jacobian_valid,
                fresh_dense,
                cached_dense,
                operand=None,
            )
        else:
            resid, Jt, aux, base_meta = self._dense_resid_jt_aux(eval_base, theta)

        base_mask, base_ineq = base_meta
        if self.has_aux:

            def residual_flat(th):
                return self._residual_and_aux(
                    unravel(th),
                    args,
                    p,
                    homotopy=homotopy,
                    ineq_weight=ineq_weight,
                    ineq_smoothing=ineq_smoothing,
                    frozen_mask=base_mask,
                )

            def residual_value(th):
                return residual_flat(th)[0]

        else:

            def residual_flat(th):
                return self._residual_and_aux(
                    unravel(th),
                    args,
                    p,
                    homotopy=homotopy,
                    ineq_weight=ineq_weight,
                    ineq_smoothing=ineq_smoothing,
                    frozen_mask=base_mask,
                )[0]

            residual_value = residual_flat

        return (
            residual_flat,
            residual_value,
            resid,
            Jt,
            jvp_fn,
            JT,
            aux,
            base_mask,
            base_ineq,
        )

    def _homotopy_active(self, homotopy):
        if homotopy is None or not self._uses_homotopy_decay:
            return jnp.asarray(False, dtype=jnp.bool_)
        floor = jnp.asarray(
            self._resolved_complementarity.tau_min
            if isinstance(self._resolved_complementarity, PDAS)
            else self._resolved_complementarity.mu_min,
            dtype=homotopy.dtype,
        )
        return homotopy > floor

    def _ineq_smoothing_active(self, ineq_smoothing):
        if ineq_smoothing is None or not self._uses_ineq_decay:
            return jnp.asarray(False, dtype=jnp.bool_)
        floor = jnp.asarray(
            self._resolved_inequality.smoothing_min, dtype=ineq_smoothing.dtype
        )
        return ineq_smoothing > floor

    def _decay_scalar(self, val, decay, min_val, improved, subproblem_solved, dtype):
        zero = jnp.zeros((), dtype=dtype)
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        eps_sqrt = jnp.sqrt(eps)
        floor = jnp.maximum(
            jnp.asarray(min_val, dtype=dtype),
            eps_sqrt if min_val == 0.0 else eps,
        )
        should_decay = (improved | subproblem_solved) & (
            val > jnp.asarray(min_val, dtype=dtype)
        )
        decayed = val * jnp.asarray(decay, dtype=dtype)
        target = jnp.asarray(min_val, dtype=dtype) if min_val > 0.0 else zero
        next_val = jnp.where(decayed <= floor, target, decayed)
        new_val = jnp.where(should_decay, next_val, val)
        return new_val, new_val != val

    def _post_step_constraints(
        self,
        theta_new,
        unravel,
        args,
        p,
        state,
        improved,
        loss,
        resid_loss,
        penalty_value,
        ridge,
        grad_norm,
        base_mask,
        base_ineq,
    ):
        dtype = loss.dtype
        has_pdas = base_mask is not None
        has_homotopy = state.homotopy is not None
        has_ineq = base_ineq is not None
        has_ineq_state = state.ineq_smoothing is not None
        false_val = jnp.asarray(False, dtype=jnp.bool_)

        if not has_pdas and not has_homotopy and not has_ineq and not has_ineq_state:
            return (
                loss,
                resid_loss,
                None,
                None,
                false_val,
                None,
                false_val,
                None,
            )

        eps_sqrt = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
        if has_homotopy:
            cfg = self._resolved_complementarity
            if isinstance(cfg, PDAS):
                decay, min_val = cfg.tau_decay, cfg.tau_min
            else:
                decay, min_val = cfg.mu_decay, cfg.mu_min
            eta = jnp.asarray(state.homotopy, dtype=dtype)
            if decay < 1.0:
                tol = jnp.maximum(1e-2 * eta, eps_sqrt)
                solved = (grad_norm <= tol) | (jnp.sqrt(resid_loss) <= tol)
                new_homotopy, homotopy_changed = self._decay_scalar(
                    eta, decay, min_val, improved, solved, dtype
                )
            else:
                new_homotopy, homotopy_changed = eta, false_val
        else:
            new_homotopy, homotopy_changed = None, false_val

        if has_ineq_state:
            icfg = self._resolved_inequality
            tau_i = jnp.asarray(state.ineq_smoothing, dtype=dtype)
            if icfg.smoothing_decay < 1.0:
                tol_i = jnp.maximum(1e-2 * tau_i, eps_sqrt)
                solved_i = grad_norm <= tol_i
                new_ineq_smoothing, ineq_smoothing_changed = self._decay_scalar(
                    tau_i,
                    icfg.smoothing_decay,
                    icfg.smoothing_min,
                    improved,
                    solved_i,
                    dtype,
                )
            else:
                new_ineq_smoothing, ineq_smoothing_changed = tau_i, false_val
        else:
            new_ineq_smoothing, ineq_smoothing_changed = None, false_val

        schedule_changed = homotopy_changed | ineq_smoothing_changed
        need_refresh = improved | schedule_changed

        if has_pdas or has_homotopy or has_ineq:

            def refresh(_):
                fresh_resid, _, (fresh_mask, fresh_ineq) = self._eval_residual_and_aux(
                    unravel(theta_new),
                    args,
                    p,
                    homotopy=new_homotopy,
                    ineq_weight=state.ineq_weight,
                    ineq_smoothing=new_ineq_smoothing,
                    frozen_mask=None,
                )
                fresh_resid_loss = jnp.sum(fresh_resid**2)
                fresh_loss = (
                    fresh_resid_loss + ridge * penalty_value
                    if self.penalized
                    else fresh_resid_loss
                )
                mask_flipped = (
                    jnp.any((fresh_mask >= 0.5) != (base_mask >= 0.5))
                    if has_pdas
                    else false_val
                )
                return fresh_loss, fresh_resid_loss, mask_flipped, fresh_ineq

            def keep(_):
                return loss, resid_loss, false_val, base_ineq

            loss, resid_loss, mask_flipped, out_ineq = jax.lax.cond(
                need_refresh, refresh, keep, operand=None
            )
        else:
            mask_flipped = false_val
            out_ineq = None

        comp_changed = (
            homotopy_changed | mask_flipped | self._homotopy_active(new_homotopy)
        )
        ineq_changed = (
            ineq_smoothing_changed | self._ineq_smoothing_active(new_ineq_smoothing)
            if has_ineq
            else None
        )
        return (
            loss,
            resid_loss,
            new_homotopy,
            new_ineq_smoothing,
            comp_changed,
            ineq_changed,
            schedule_changed,
            out_ineq,
        )

    def _resolved_state(self, state):
        if not self.penalized and state.ridge_metric is not None:
            raise ValueError(
                "state.ridge_metric requires ridge > 0; outer steps use "
                "Euclidean geometry"
            )
        rm = (
            self.initial_ridge_metric
            if (self.penalized and state.ridge_metric is None)
            else state.ridge_metric
        )
        precond = (
            self.initial_preconditioner
            if state.preconditioner is None
            else state.preconditioner
        )
        if rm is state.ridge_metric and precond is state.preconditioner:
            return state
        return dataclasses.replace(state, ridge_metric=rm, preconditioner=precond)

    def _solver_space_and_form(self, solver_cfg, m, n, ridge_metric, *, damped):
        if isinstance(solver_cfg, (Cholesky, QR)):
            form = solver_cfg.resolve_form(
                m, n, self.penalized, ridge_metric, damped=damped
            )
            space = "residual" if form == "gram" else "parameter"
            return space, form
        if isinstance(solver_cfg, GramCG):
            return "residual", "gram"
        if isinstance(solver_cfg, CG):
            return "parameter", "normal"
        if isinstance(solver_cfg, LU):
            return "parameter", "lu"
        return "parameter", "svd"

    def _base_objective(self, resid, theta, state, ctx):
        dtype = resid.dtype
        resid_loss_old = jnp.sum(resid**2)
        if not self.penalized:
            return (
                jnp.zeros((), dtype=dtype),
                None,
                jnp.zeros_like(theta),
                resid_loss_old,
                resid_loss_old,
            )
        if state.ridge is None:
            raise ValueError(
                "the state has no ridge; create it with init(x, args, p=p)"
            )
        ridge = jnp.asarray(state.ridge, dtype=dtype)
        rm = ctx.state.ridge_metric
        penalty_grad = jnp.asarray(rm.matvec(theta, ctx), dtype=dtype)
        penalty_value_old = jnp.asarray(rm.quad_form(theta, ctx), dtype=dtype)
        loss_old = resid_loss_old + ridge * penalty_value_old
        return (
            ridge,
            penalty_value_old,
            penalty_grad,
            resid_loss_old,
            loss_old,
        )

    def _geodesic_correction(
        self,
        residual_flat,
        theta,
        velocity,
        solve_residual_rhs,
        geodesic_acceptance_ratio,
        dtype,
    ):
        v_detached = jax.lax.stop_gradient(velocity)

        def first_jvp(th):
            return jax.jvp(residual_flat, (th,), (v_detached,), has_aux=self.has_aux)[1]

        r_vv = jax.jvp(first_jvp, (theta,), (v_detached,))[1]
        acceleration = jnp.asarray(solve_residual_rhs(r_vv), dtype=dtype)
        zero = jnp.zeros((), dtype=dtype)
        acceleration_ratio = (
            2.0
            * jnp.linalg.norm(acceleration)
            / (jnp.linalg.norm(velocity) + jnp.finfo(dtype).eps)
        )
        ratio_accepted = (
            (geodesic_acceptance_ratio > zero)
            & (acceleration_ratio > zero)
            & (acceleration_ratio <= geodesic_acceptance_ratio)
        )
        return acceleration, acceleration_ratio, ratio_accepted

    def _ad_linearization(
        self, x, args, p, homotopy=None, ineq_weight=None, ineq_smoothing=None
    ):
        theta, unravel = ravel_pytree(x)

        def eval_for_ad(theta_value, p_value):
            return self._residual_and_aux(
                unravel(theta_value),
                args,
                p_value,
                homotopy=homotopy,
                ineq_weight=ineq_weight,
                ineq_smoothing=ineq_smoothing,
                detach_mask=False,
            )

        (residual, _), lin_fn = jax.linearize(eval_for_ad, theta, p)
        return theta, unravel, residual, lin_fn

    def _ad_cg_tol(self, dtype):
        if self.ad_solver_tol is not None:
            return jnp.asarray(self.ad_solver_tol, dtype=dtype)
        default_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return jnp.asarray(default_tol, dtype=dtype)

    def _ad_preconditioner(self, state):
        if self._ad_preconditioner_source == "carried":
            return state.preconditioner
        if self._ad_preconditioner_source == "explicit":
            return self.ad_solver_preconditioner
        return None

    def _step_failure_status(self, info):
        return jnp.where(
            ~jnp.isfinite(info.loss),
            jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32),
            jnp.asarray(SolverStatus.RUNNING, dtype=jnp.int32),
        )

    def _check_instance_structure(self, new, previous, name):
        def spec(tree):
            leaves, treedef = jax.tree_util.tree_flatten(tree)
            return treedef, [
                (
                    jnp.shape(leaf),
                    jnp.result_type(leaf),
                    getattr(jax.typeof(jnp.asarray(leaf)), "weak_type", False),
                )
                for leaf in leaves
            ]

        if spec(new) != spec(previous):
            raise ValueError(
                f"the callback action replaced state.{name} with a "
                "different type, structure, or leaf shape/dtype/weak-type; "
                "rebuild the same class with arrays matching the carried "
                "instance, and preserve untouched fields with "
                "dataclasses.replace(ctx.state, ...)"
            )

    def _apply_action(self, action, x, state, args, user_state):
        if action is None:
            action = SolverAction()
        action_state = action.state
        xargs_changed = jnp.asarray(False)
        state_changed = jnp.asarray(False)
        rm_changed = jnp.asarray(False)
        if action.x is not None:
            xargs_changed = xargs_changed | _tree_changed(action.x, x)
            x = action.x
        if action_state is not None:
            previous = state
            state = action_state
            if self.cache_jacobian and state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True but the callback action returned a "
                    "state without the Jacobian cache; use "
                    "dataclasses.replace(ctx.state, ...) to preserve cache fields"
                )
            self._check_action_state(state)
            previous_hyper = previous.hyper
            if previous_hyper is not None:
                if state.hyper is None or (
                    jax.tree_util.tree_structure(previous_hyper)
                    != jax.tree_util.tree_structure(state.hyper)
                ):
                    raise ValueError(
                        "the callback action changed the structure or dtypes of "
                        "state.hyper; reset values with "
                        "dataclasses.replace(ctx.state.hyper, ...)"
                    )
                for prev_leaf, new_leaf in zip(
                    jax.tree_util.tree_leaves(previous_hyper),
                    jax.tree_util.tree_leaves(state.hyper),
                    strict=True,
                ):
                    if isinstance(new_leaf, bool) or (
                        isinstance(new_leaf, jax.Array | np.ndarray | np.generic)
                        and (
                            new_leaf.shape != prev_leaf.shape
                            or new_leaf.dtype != prev_leaf.dtype
                        )
                    ):
                        raise ValueError(
                            "the callback action changed the structure or dtypes of "
                            "state.hyper; reset values with "
                            "dataclasses.replace(ctx.state.hyper, ...)"
                        )
                state = dataclasses.replace(
                    state,
                    hyper=jax.tree.map(
                        lambda prev_leaf, new_leaf: jnp.asarray(
                            new_leaf, dtype=prev_leaf.dtype
                        ),
                        previous_hyper,
                        state.hyper,
                    ),
                )
            if self.penalized:
                new_rm, prev_rm = state.ridge_metric, previous.ridge_metric
                if type(new_rm) is type(prev_rm) and any(
                    isinstance(leaf, jax.core.Tracer)
                    for leaf in jax.tree_util.tree_leaves(new_rm)
                ):
                    rm_fields = getattr(new_rm, "__dataclass_fields__", {})
                    if "penalized_is_spd" in rm_fields:
                        object.__setattr__(
                            new_rm, "penalized_is_spd", prev_rm.penalized_is_spd
                        )
                    if "pos_indices" in rm_fields:
                        object.__setattr__(new_rm, "pos_indices", prev_rm.pos_indices)
                self._check_instance_structure(new_rm, prev_rm, "ridge_metric")
                rm_changed = _tree_changed(new_rm, prev_rm)
            self._check_instance_structure(
                state.preconditioner, previous.preconditioner, "preconditioner"
            )
            state, state_changed = self._apply_action_state(state, previous)
            if self.penalized:
                state_changed = state_changed | rm_changed
        if action.args is not None:
            xargs_changed = xargs_changed | _tree_changed(action.args, args)
            args = action.args
        if action.user_state is not None:
            user_state = action.user_state
        problem_changed = xargs_changed | state_changed
        if action.x is not None or action.args is not None:
            if state.jacobian_valid is not None:
                state = dataclasses.replace(
                    state, jacobian_valid=state.jacobian_valid & ~xargs_changed
                )
        touched = (
            action.x is not None or action.args is not None or action_state is not None
        )
        state = self._invalidate_caches(state, touched, problem_changed | rm_changed)
        return action, x, state, args, user_state, problem_changed

    def _action_rescued(self, action, state, previous, problem_changed):
        if action is None or action.state is None:
            return problem_changed
        rm_changed = (
            _tree_changed(state.ridge_metric, previous.ridge_metric)
            if self.penalized
            else jnp.asarray(False)
        )
        hyper_changed = _tree_changed(state.hyper, previous.hyper)
        precond_changed = _tree_changed(state.preconditioner, previous.preconditioner)
        return problem_changed | rm_changed | hyper_changed | precond_changed

    def _check_action_state(self, state):
        if self.penalized and state.ridge is None:
            raise ValueError(
                "the callback action returned a state without ridge; "
                "use dataclasses.replace(ctx.state, ...) to preserve it"
            )
        if not self.penalized and state.ridge_metric is not None:
            raise ValueError(
                "the callback action set state.ridge_metric on an unpenalized solver"
            )

    def _apply_action_state(self, state, previous):
        changed = jnp.asarray(False)
        resid_transform_changed = jnp.asarray(False)
        if previous.homotopy is not None:
            new_homotopy = jnp.asarray(state.homotopy, dtype=previous.homotopy.dtype)
            homotopy_changed = ~jnp.array_equal(
                new_homotopy, previous.homotopy, equal_nan=True
            )
            state = dataclasses.replace(state, homotopy=new_homotopy)
            resid_transform_changed = resid_transform_changed | homotopy_changed
        if previous.ineq_weight is not None:
            new_iw = jnp.asarray(state.ineq_weight, dtype=previous.ineq_weight.dtype)
            if new_iw.shape != previous.ineq_weight.shape:
                raise ValueError(
                    "the callback action changed the shape of state.ineq_weight"
                )
            iw_changed = ~jnp.array_equal(new_iw, previous.ineq_weight, equal_nan=True)
            state = dataclasses.replace(state, ineq_weight=new_iw)
            resid_transform_changed = resid_transform_changed | iw_changed
        if previous.ineq_smoothing is not None:
            new_is = jnp.asarray(
                state.ineq_smoothing, dtype=previous.ineq_smoothing.dtype
            )
            is_changed = ~jnp.array_equal(
                new_is, previous.ineq_smoothing, equal_nan=True
            )
            state = dataclasses.replace(state, ineq_smoothing=new_is)
            resid_transform_changed = resid_transform_changed | is_changed
        if state.jacobian_valid is not None:
            state = dataclasses.replace(
                state,
                jacobian_valid=state.jacobian_valid & ~resid_transform_changed,
            )
        changed = changed | resid_transform_changed
        if not self.penalized:
            return state, changed
        _validate_ridge_value(state.ridge, "state.ridge")
        new_ridge = jnp.asarray(state.ridge, dtype=previous.ridge.dtype)
        ridge_changed = ~jnp.array_equal(new_ridge, previous.ridge, equal_nan=True)
        return dataclasses.replace(state, ridge=new_ridge), changed | ridge_changed

    def _invalidate_caches(self, state, touched, caches_stale):
        if touched and state.solver_cache is not None:
            cache = state.solver_cache
            state = dataclasses.replace(
                state,
                solver_cache=dataclasses.replace(
                    cache, valid=cache.valid & ~caches_stale
                ),
            )
        return state

    def solve(
        self,
        x0,
        args=None,
        *,
        p=None,
        state=None,
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
        """Run solver updates until a stopping rule fires and support AD in ``p``."""
        self._check_residual_args(args, p)
        self._validate_tolerances(atol, gtol, xtol)
        callback = _hashable_hook(callback)
        state = self._solve_state(x0, args, p, state)
        if state.hyper is None:
            state = dataclasses.replace(state, hyper=self.hyperparams())
        history_len = max_steps + 1 if save_steps else None

        if multi_start is not None:
            num_starts = multi_start.num_starts
            draw = _hashable_hook(multi_start.draw if num_starts > 1 else None)
            reset_args = _hashable_hook(
                multi_start.reset_args if num_starts > 1 else None
            )
            default_accept = (
                _accept_converged_or_max_steps
                if max_steps_is_success
                else _accept_converged
            )
            accept = _hashable_hook(
                default_accept if multi_start.accept is None else multi_start.accept
            )
            parallel = multi_start.parallel and num_starts > 1
            if jit:
                if draw is not None:
                    drawn = jax.eval_shape(draw, multi_start.key, x0, args)
                    _check_hook_types("draw", "x", x0, drawn)
                if reset_args is not None:
                    reset = jax.eval_shape(reset_args, multi_start.key, args)
                    _check_hook_types("reset_args", "args", args, reset)

            @jax.custom_jvp
            def solve_multi_start_with_ad_p(
                x, state, args, p, user_state, key, max_steps, atol, gtol, xtol
            ):
                return self._multi_start_impl(
                    x,
                    state,
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
                    reset_args,
                    accept,
                    parallel,
                )

            @solve_multi_start_with_ad_p.defjvp
            def solve_multi_start_with_ad_p_jvp(primals, tangents):
                p_dot = tangents[3]
                result = solve_multi_start_with_ad_p(*primals)
                point = self._initial_ad_point(
                    primals[0], primals[1], primals[2], primals[3]
                )
                return result, self._ad_result_tangent(
                    result, p_dot, point, max_steps_is_success
                )

            return solve_multi_start_with_ad_p(
                x0,
                state,
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
        def solve_with_ad_p(x, state, args, p, user_state, max_steps, atol, gtol, xtol):
            return self._solve_impl(
                x,
                state,
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
            point = self._initial_ad_point(
                primals[0], primals[1], primals[2], primals[3]
            )
            return result, self._ad_result_tangent(
                result, p_dot, point, max_steps_is_success
            )

        return solve_with_ad_p(
            x0, state, args, p, user_state, max_steps, atol, gtol, xtol
        )

    def _validate_tolerances(self, atol, gtol, xtol):
        if not self.penalized:
            return
        concrete = [not isinstance(t, jax.core.Tracer) for t in (atol, gtol, xtol)]
        if (
            concrete[0]
            and atol > 0
            and (concrete[1] and gtol == 0)
            and (concrete[2] and xtol == 0)
        ):
            raise ValueError(
                "ridge > 0 requires a positive gtol or xtol to test stationarity "
                "of the penalized objective; atol alone cannot detect a ridge minimizer"
            )

    def _validate_supplied_ridge(self, state, dtype):
        if state.ridge is None:
            raise ValueError(
                "the caller-supplied state has no ridge; create it with "
                "init(x, args, p=p) or set a positive ridge"
            )
        _validate_ridge_value(state.ridge, "state.ridge")
        return dataclasses.replace(state, ridge=jnp.asarray(state.ridge, dtype=dtype))

    def _initial_ad_point(self, x, state, args, p):
        resolved = self._resolved_state(state)
        base = (
            x,
            args,
            p,
            resolved.ridge_metric,
            resolved.preconditioner,
            resolved.homotopy,
            resolved.ineq_weight,
            resolved.ineq_smoothing,
        )
        if self.penalized:
            return (*base, resolved.ridge)
        return base

    def _constraints_stable(self, info):
        comp_stable = (
            jnp.asarray(True, dtype=jnp.bool_)
            if info.complementarity_changed is None
            else ~info.complementarity_changed
        )
        ineq_stable = (
            jnp.asarray(True, dtype=jnp.bool_)
            if info.inequality_changed is None
            else ~info.inequality_changed
        )
        return comp_stable & ineq_stable

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
        gtol_met = ready & (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = ready & (xtol > 0) & info.accepted & (info.step_norm < xtol)
        if self.penalized:
            residual_ok = (atol <= 0) | (jnp.sqrt(info.resid_loss) <= atol)
            return (gtol_met | xtol_met) & residual_ok
        atol_met = ready & (atol > 0) & (jnp.sqrt(info.loss) < atol)
        return atol_met | gtol_met | xtol_met

    def _refresh_final_info(self, info, x, state, args, p):
        residual, aux, (_, ineq_metrics) = self._eval_residual_and_aux(
            x,
            args,
            p,
            homotopy=state.homotopy,
            ineq_weight=state.ineq_weight,
            ineq_smoothing=state.ineq_smoothing,
        )
        dtype = info.loss.dtype
        resid_loss = jnp.asarray(jnp.sum(residual**2), dtype=dtype)
        updates = {"aux": aux}
        if state.homotopy is not None:
            updates["homotopy"] = jnp.asarray(state.homotopy, dtype=dtype)
        if self.penalized:
            theta, _ = ravel_pytree(x)
            resolved = self._resolved_state(state)
            ridge = jnp.asarray(state.ridge, dtype=dtype)
            ctx = SolverContext(
                x=theta,
                state=resolved,
                args=args,
                p=p,
                ridge=ridge,
                damping=jnp.zeros((), dtype=dtype),
            )
            penalty_grad = jnp.asarray(
                resolved.ridge_metric.matvec(theta, ctx), dtype=dtype
            )
            penalty_value = jnp.asarray(
                resolved.ridge_metric.quad_form(theta, ctx), dtype=dtype
            )
            updates["ridge"] = ridge
            updates["resid_loss"] = resid_loss
            updates["penalty_value"] = penalty_value
            updates["penalty_grad_norm"] = jnp.asarray(
                jnp.linalg.norm(penalty_grad), dtype=dtype
            )
            updates["loss"] = jnp.asarray(
                resid_loss + ridge * penalty_value, dtype=dtype
            )
        else:
            updates["loss"] = resid_loss
        if ineq_metrics is not None:
            eq_loss, viol_norm, max_viol = ineq_metrics
            updates["eq_loss"] = jnp.asarray(eq_loss, dtype=dtype)
            updates["ineq_violation_norm"] = jnp.asarray(viol_norm, dtype=dtype)
            updates["ineq_max_violation"] = jnp.asarray(max_viol, dtype=dtype)
        return dataclasses.replace(info, **updates)

    def _ranking_objective(self, result, p, callback):
        homotopy = result.state.homotopy
        ineq_smoothing = result.state.ineq_smoothing
        if (
            callback is None
            and not self._uses_homotopy_decay
            and not self._uses_ineq_decay
        ):
            loss = result.info.loss
        else:
            target_homotopy = (
                None
                if homotopy is None
                else jnp.where(
                    self._homotopy_active(homotopy),
                    jnp.asarray(
                        self._resolved_complementarity.tau_min
                        if isinstance(self._resolved_complementarity, PDAS)
                        else self._resolved_complementarity.mu_min,
                        dtype=homotopy.dtype,
                    ),
                    homotopy,
                )
            )
            target_ineq_smoothing = (
                None
                if ineq_smoothing is None
                else jnp.where(
                    self._ineq_smoothing_active(ineq_smoothing),
                    jnp.asarray(
                        self._resolved_inequality.smoothing_min,
                        dtype=ineq_smoothing.dtype,
                    ),
                    ineq_smoothing,
                )
            )
            residual = self._residual_and_aux(
                result.x,
                result.args,
                p,
                homotopy=target_homotopy,
                ineq_weight=result.state.ineq_weight,
                ineq_smoothing=target_ineq_smoothing,
            )[0]
            if not self.penalized:
                loss = jnp.sum(residual**2)
            else:
                theta, _ = ravel_pytree(result.x)
                resolved = self._resolved_state(result.state)
                ctx = SolverContext(x=theta, state=resolved, args=result.args, p=p)
                ridge = jnp.asarray(result.state.ridge, dtype=residual.dtype)
                pen = jnp.asarray(
                    resolved.ridge_metric.quad_form(theta, ctx), dtype=residual.dtype
                )
                loss = jnp.sum(residual**2) + ridge * pen
        return jnp.where(
            jnp.isfinite(loss), loss, jnp.asarray(jnp.inf, dtype=loss.dtype)
        )

    def _solve_impl(
        self,
        x,
        state,
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
        driver = _solve_loop_jit if jit else _solve_python_impl
        return driver(
            self,
            x,
            state,
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
        state,
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
        reset_args,
        accept,
        parallel,
    ):
        if not jit:
            return _multi_start_python_impl(
                self,
                x,
                state,
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
                reset_args,
                accept,
                parallel,
            )
        if parallel:
            return _multi_start_parallel_jit(
                self,
                x,
                state,
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
                reset_args,
                accept,
                num_starts,
            )
        return _multi_start_sequential_jit(
            self,
            x,
            state,
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
            reset_args,
            accept,
        )

    def _ad_result_tangent(self, result, p_dot, initial_ad_point, max_steps_is_success):
        zero_result = jax.tree.map(_zero_tangent_leaf, result)
        if result.p is None:
            return dataclasses.replace(zero_result, p=p_dot)
        initial_x, initial_args, initial_p = jax.tree.map(
            jax.lax.stop_gradient, initial_ad_point[:3]
        )
        ad_success = result.status == SolverStatus.CONVERGED
        if max_steps_is_success:
            ad_success = ad_success | (result.status == SolverStatus.MAX_STEPS)
        ad_x = _where_tree(ad_success, result.x, initial_x)
        ad_args = _where_tree(ad_success, result.args, initial_args)
        ad_p = _where_tree(ad_success, result.p, initial_p)
        ad_p_dot = _mask_tangent_tree(ad_success, p_dot)
        x_dot, aux_dot = self._ad_x_tangent(
            ad_x,
            ad_args,
            ad_p,
            ad_p_dot,
            result,
            ad_success,
            initial_ad_point,
            zero_result.aux,
        )
        return dataclasses.replace(zero_result, x=x_dot, p=p_dot, aux=aux_dot)

    def _resolved_ad_solver(self, m, n, ridge_metric, *, has_inequality=False):
        if self.ad_solver is not None:
            self.ad_solver.validate_shape(
                m,
                n,
                penalized=self.penalized,
                ridge_metric=ridge_metric,
                damped=False,
                role="ad_solver",
            )
            return self.ad_solver
        if not self.penalized:
            if isinstance(self.linear_solver, CG) and (
                n <= m or self.linear_solver.penalty is not None
            ):
                return self.linear_solver
            if isinstance(self.linear_solver, GramCG) and m <= n:
                return self.linear_solver
            if isinstance(self.linear_solver, SVD):
                return self.linear_solver
            return (
                LU()
                if (m == n and not has_inequality and self.inequality is None)
                else SVD()
            )

        if isinstance(self.linear_solver, SVD):
            return self.linear_solver
        if isinstance(self.linear_solver, CG) and (
            ridge_metric.n_free_for(n) <= m or self.linear_solver.penalty is not None
        ):
            return self.linear_solver
        if (
            isinstance(self.linear_solver, GramCG)
            and ridge_metric.has_shifted_solve
            and ridge_metric.n_free_for(n) == 0
            and ridge_metric.penalized_is_spd
        ):
            return self.linear_solver
        if ridge_metric.has_factor and (
            (m + ridge_metric.factor_rows_for(n) < n)
            or (not ridge_metric.penalized_is_spd and m < n)
            or (not ridge_metric.penalized_is_spd and ridge_metric.n_free_for(n) > m)
        ):
            return SVD()
        if isinstance(self.linear_solver, (QR, Cholesky)):
            try:
                self.linear_solver.validate_shape(
                    m, n, penalized=True, ridge_metric=ridge_metric, damped=False
                )
            except ValueError:
                pass
            else:
                return self.linear_solver
        fallback = Cholesky()
        fallback.validate_shape(
            m,
            n,
            penalized=True,
            ridge_metric=ridge_metric,
            damped=False,
            role="ad_solver",
        )
        return fallback

    def _ad_x_tangent(
        self, x, args, p, p_dot, result, ad_success, initial_ad_point, zero_aux
    ):
        state = self._resolved_state(jax.lax.stop_gradient(result.state))
        (
            init_rm,
            init_precond,
            init_homotopy,
            init_iw,
            init_is,
        ) = jax.lax.stop_gradient(initial_ad_point[3:8])

        def _select_opt(final_v, init_v):
            if final_v is None:
                return None
            return jnp.where(
                ad_success, final_v, jnp.asarray(init_v, dtype=final_v.dtype)
            )

        homotopy = _select_opt(state.homotopy, init_homotopy)
        ineq_weight = _select_opt(state.ineq_weight, init_iw)
        ineq_smoothing = _select_opt(state.ineq_smoothing, init_is)
        state = dataclasses.replace(
            state,
            ridge_metric=(
                _where_tree(ad_success, state.ridge_metric, init_rm)
                if self.penalized
                else None
            ),
            preconditioner=_where_tree(ad_success, state.preconditioner, init_precond),
            homotopy=homotopy,
            ineq_weight=ineq_weight,
            ineq_smoothing=ineq_smoothing,
        )

        theta, unravel, residual, lin_fn = self._ad_linearization(
            x,
            args,
            p,
            homotopy=homotopy,
            ineq_weight=ineq_weight,
            ineq_smoothing=ineq_smoothing,
        )
        zero_theta = jnp.zeros_like(theta)
        zero_p = jax.tree.map(_zero_tangent_leaf, p)

        def theta_jvp(v):
            return lin_fn(v, zero_p)[0]

        residual_p_dot, aux_p_dot = lin_fn(zero_theta, p_dot)
        n, m = theta.shape[0], residual.shape[0]
        dtype = residual.dtype
        zero = jnp.zeros((), dtype=dtype)
        if self.penalized:
            final_ridge = jax.lax.stop_gradient(result.state.ridge)
            init_ridge = jax.lax.stop_gradient(initial_ad_point[8])
            ridge = jnp.where(
                ad_success,
                jnp.asarray(final_ridge, dtype=dtype),
                jnp.asarray(init_ridge, dtype=dtype),
            )
            state = dataclasses.replace(state, ridge=ridge)
        else:
            ridge = zero

        resolved = self._resolved_ad_solver(
            m,
            n,
            state.ridge_metric,
            has_inequality=result.info.ineq_max_violation is not None,
        )
        space, form = self._solver_space_and_form(
            resolved, m, n, state.ridge_metric, damped=False
        )
        ctx = SolverContext(
            x=theta,
            state=state,
            args=args,
            p=p,
            ridge=ridge if self.penalized else None,
            damping=zero,
            space=space,
            role="ad",
            form=form,
        )

        Jt = jvp_fn = JT = None
        if resolved.materializes_jacobian:
            Jt = self._assemble_jt(theta_jvp, theta, residual)
        else:
            jvp_fn = theta_jvp
            transpose_fn = jax.linear_transpose(theta_jvp, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

        penalty_grad = (
            jnp.asarray(state.ridge_metric.matvec(theta, ctx), dtype=dtype)
            if self.penalized
            else zero_theta
        )
        ad_hyper = (
            _ADKrylovHyper(
                iterative_tol=self._ad_cg_tol(dtype),
                iterative_atol=jnp.asarray(self.ad_solver_atol, dtype=dtype),
                iterative_maxiter=self.ad_solver_maxiter,
            )
            if isinstance(resolved, (CG, GramCG))
            else None
        )
        ad_precond = (
            self._ad_preconditioner(state)
            if (
                isinstance(resolved, (CG, GramCG))
                or getattr(resolved, "preconditioner", None) is not None
            )
            else None
        )
        prepared = resolved.prepare(
            Subproblem(
                resid=residual,
                theta=theta,
                Jt=Jt,
                jvp_fn=jvp_fn,
                JT=JT,
                ridge=ridge,
                damping=zero,
                ridge_metric=state.ridge_metric,
                penalty_grad=penalty_grad,
                cache=None,
                cache_enabled=False,
                hyper=ad_hyper,
                ctx=ctx,
                penalized=self.penalized,
                undamped=True,
                active=ad_success,
                preconditioner=ad_precond,
            )
        )
        valid_scale = jnp.where(
            (~ad_success) | prepared.solve_ok,
            jnp.ones((), dtype=dtype),
            jnp.asarray(jnp.nan, dtype=dtype),
        )
        raw_x_dot_vec = valid_scale * prepared.solve_residual_rhs(residual_p_dot)
        x_dot_vec = jnp.where(ad_success, raw_x_dot_vec, zero_theta)
        x_dot = unravel(x_dot_vec)
        if not self.has_aux:
            return x_dot, zero_aux
        aux_x_dot = lin_fn(x_dot_vec, zero_p)[1]
        aux_dot = jax.tree.map(
            lambda a, b: (
                None
                if a is None
                else (a if getattr(a, "dtype", None) == jax.dtypes.float0 else a + b)
            ),
            aux_x_dot,
            aux_p_dot,
        )
        return x_dot, _mask_tangent_tree(ad_success, aux_dot)


__all__ = ["MultiStart", "SolverBase"]
