"""``SolverBase`` -- shared solver protocol, linearization, whitening, geodesic
correction, outer loop driver, and implicit differentiation for
:class:`~nlls_gram.LevenbergMarquardt` and :class:`~nlls_gram.Newton`.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import numpy as np
from jax.flatten_util import ravel_pytree

from nlls_gram.linear_solvers import (
    CG,
    LU,
    SVD,
    Cholesky,
    GramCG,
    _config_static_key,
)
from nlls_gram.metrics import Metric
from nlls_gram.multi_start import (
    MultiStart,
    _accept_converged,
    _accept_converged_or_max_steps,
    _check_hook_types,
    _multi_start_parallel_jit,
    _multi_start_python_impl,
    _multi_start_sequential_jit,
)
from nlls_gram.solve_loop import _solve_loop_jit, _solve_python_impl
from nlls_gram.types import (
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
    mm,
)


# The residual evaluation that sizes the initial state runs outside the jitted
# loop; keying it on the solver's value means a repeated solve with an equal
# configuration retraces nothing.
@functools.partial(jax.jit, static_argnums=0)
def _init_residual_and_aux(solver, x, args, p):
    return solver._residual_and_aux(x, args, p)


@jax.custom_jvp
def _svd_pinv(Bt):
    # Moore-Penrose pseudoinverse X = (Bt^T)^+ of shape (n, m), with the same
    # rank cutoff max(n, m) * eps * sigma_max as the forward SVD solver.
    U, sigma, Vt = jnp.linalg.svd(Bt.T, full_matrices=False)
    cutoff = max(Bt.shape) * jnp.finfo(Bt.dtype).eps * sigma[0]
    mask = sigma > cutoff
    safe_sigma = jnp.where(mask, sigma, jnp.ones_like(sigma))
    inverted = jnp.where(mask, 1.0 / safe_sigma, jnp.zeros_like(sigma))
    return mm(Vt.T * inverted, U.T)


@_svd_pinv.defjvp
def _svd_pinv_jvp(primals, tangents):
    (Bt,) = primals
    (dBt,) = tangents
    X = _svd_pinv(Bt)
    # Golub-Pereyra (1973) pseudoinverse derivative for B = Bt^T, dB = dBt^T:
    #   dX = -X dB X + X X^T dB^T (I_m - B X) + (I_n - X B) dB^T X^T X
    # Grouped through min(n, m) x min(n, m) products so no large projector is
    # formed and no 1 / (sigma_i^2 - sigma_j^2) SVD-vector gaps ever appear.
    if Bt.shape[0] >= Bt.shape[1]:
        term1 = -mm(X, mm(dBt.T, X))
        XXt_dBt = mm(X, mm(X.T, dBt))
        term2 = XXt_dBt - mm(XXt_dBt, mm(Bt.T, X))
        dBt_XtX = mm(dBt, mm(X.T, X))
        term3 = dBt_XtX - mm(X, mm(Bt.T, dBt_XtX))
    else:
        term1 = -mm(mm(X, dBt.T), X)
        XXt_dBt = mm(mm(X, X.T), dBt)
        term2 = XXt_dBt - mm(mm(XXt_dBt, Bt.T), X)
        dBt_XtX = mm(mm(dBt, X.T), X)
        term3 = dBt_XtX - mm(mm(X, Bt.T), dBt_XtX)
    return X, term1 + term2 + term3


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


class SolverBase:
    # Value-based identity: the jitted solve loop marks the solver itself
    # static, so equal-config solvers built around the same residual share the
    # compiled loop across instances. Subclasses set _static_key/_static_hash
    # in __init__ from their constructor arguments; metric/preconditioner
    # instances key by pytree STRUCTURE (their arrays are threaded through
    # the carried state, so equal-config fresh instances share one compile).
    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._static_key == other._static_key

    def __hash__(self):
        return self._static_hash

    # Sealed at the end of each subclass __init__: the configuration keys
    # the compiled solve loop, so mutate-by-assignment would keep the stale
    # key and silently reuse another configuration's compilation.
    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError(
                f"{type(self).__name__} is frozen after construction: its "
                "configuration keys the compiled solve loop. Build a new "
                "solver instead of assigning attributes"
            )
        object.__setattr__(self, name, value)

    _has_metric = False
    penalized = False
    _forward_role_attr = "supports_forward"

    def _init_common(
        self,
        residual_fn,
        *,
        linear_solver,
        metric,
        ridge,
        has_aux,
        cache_jacobian,
        geodesic_acceleration,
        geodesic_acceptance_ratio,
        ad_solver,
        jacobian_mode,
        jacobian_chunk,
    ):
        if not isinstance(has_aux, bool):
            raise TypeError("has_aux must be a bool")
        if not isinstance(cache_jacobian, bool):
            raise TypeError("cache_jacobian must be a bool")
        if not isinstance(geodesic_acceleration, bool):
            raise TypeError("geodesic_acceleration must be a bool")
        if (
            isinstance(geodesic_acceptance_ratio, bool)
            or not np.isfinite(float(geodesic_acceptance_ratio))
            or float(geodesic_acceptance_ratio) < 0.0
        ):
            raise ValueError("geodesic_acceptance_ratio must be nonnegative")

        if metric is not None:
            if not isinstance(metric, Metric):
                raise TypeError(
                    f"metric must be a Metric instance, got {type(metric).__name__}"
                )
            self._check_registered_instance(metric, "metric")

        penalized = ridge is not None
        if penalized:
            _validate_ridge_value(ridge, "ridge")

        self._raw_residual_fn = residual_fn
        self.residual_fn, self.residual_arity = canonicalize_residual(residual_fn)
        self.has_aux = has_aux
        self.linear_solver = linear_solver
        self.initial_metric = metric
        self._has_metric = metric is not None
        self.penalized = penalized
        self.initial_ridge = ridge
        self.cache_jacobian = cache_jacobian
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        self.ad_solver = ad_solver
        self.jacobian_mode = jacobian_mode
        self.jacobian_chunk = jacobian_chunk
        self._validate_configuration(linear_solver, ad_solver, penalized=penalized)

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
            self.ad_solver_penalty = (
                getattr(linear_solver, "penalty", None)
                if (ad_solver is None and krylov)
                else None
            )
            self.ad_solver_preconditioner = None
            inherit = (
                ad_solver is None
                and krylov
                and not linear_solver.preconditioner.requires_positive_damping
            )
            self._ad_preconditioner_source = "carried" if inherit else "none"

        self.cache_jacobian = cache_jacobian and linear_solver.materializes_jacobian

    def _common_static_key(self):
        return (
            _static_key_component(self._raw_residual_fn),
            self.residual_arity,
            self.has_aux,
            self.penalized,
            _config_static_key(self.linear_solver, baked=False),
            None
            if not self._has_metric
            else jax.tree_util.tree_structure(self.initial_metric),
            self.cache_jacobian,
            self.geodesic_acceleration,
            _config_static_key(self.ad_solver, baked=True),
            self.jacobian_mode,
            self.jacobian_chunk,
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
        # A nested unregistered object (e.g. PaddedPreconditioner over an
        # unregistered base) flattens as an opaque leaf and would fail deep
        # inside jit instead of here.
        for leaf in jax.tree_util.tree_leaves(instance):
            if not isinstance(
                leaf, (jax.Array, np.ndarray, np.generic, int, float, complex, bool)
            ):
                raise TypeError(
                    f"{keyword} contains a non-array leaf of type "
                    f"{type(leaf).__name__}; register nested "
                    "metric/preconditioner classes with "
                    "nlls_gram.register_pytree_dataclass"
                )

    def _validate_configuration(self, linear_solver, ad_solver, penalized):
        """Reject a config in a role it cannot fill, at construction.

        Without this an unsupported combination resolves to a DIFFERENT
        algorithm -- ``ad_solver=QR()`` quietly running dense Cholesky, or the
        ridge objective handing its penalized subproblem to a dual form that
        never sees the penalty rows.
        """
        role = [(linear_solver, "linear_solver", self._forward_role_attr)]
        if ad_solver is not None:
            role.append((ad_solver, "ad_solver", "supports_ad"))
        for config, keyword, attribute in role:
            if not getattr(config, attribute):
                raise ValueError(
                    f"{config!r} cannot serve as {keyword}: it does not "
                    f"implement that role"
                )
            if penalized and not config.supports_penalty:
                raise ValueError(
                    f"{config!r} cannot serve as {keyword} for the ridge "
                    "objective: its operator never sees the penalty rows"
                )
        if self.jacobian_mode not in ("auto", "fwd", "rev"):
            raise ValueError(f"unknown jacobian_mode: {self.jacobian_mode}")
        if self.jacobian_chunk is not None and (
            isinstance(self.jacobian_chunk, bool)
            or not isinstance(self.jacobian_chunk, int)
            or self.jacobian_chunk <= 0
        ):
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
        # Drawn multi-start lanes must not reuse caches built at another
        # (x, args); damping, ridge, hyper, and the carried instances stay
        # inherited from the caller's initial state.
        updates = {}
        if state.jacobian_valid is not None:
            updates["jacobian_valid"] = jnp.zeros_like(state.jacobian_valid)
        if state.solver_cache is not None:
            updates["solver_cache"] = jax.tree.map(jnp.zeros_like, state.solver_cache)
        return dataclasses.replace(state, **updates) if updates else state

    def _resolve_jacobian_mode(self, m, n):
        # "auto" vmaps the identity basis of the SMALL side: n forward-mode
        # columns when the system is tall or square (n <= m), m reverse-mode
        # rows only when strictly fat.
        if self.jacobian_mode != "auto":
            return self.jacobian_mode
        return "fwd" if n <= m else "rev"

    def _assemble_jt(self, jvp_fn, theta, resid):
        # The basis goes through the linearization in one batch, or in chunks
        # of jacobian_chunk so the batched pass's live intermediates scale
        # with the chunk instead of with m or n.
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
            return jnp.ravel(value), aux
        return jnp.ravel(self.residual_fn(x, args, p)), None

    def _check_residual_args(self, args, p):
        # Silently dropping args/p a residual never sees would, in particular,
        # make the implicit derivative with respect to p a silent zero.
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
            if state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True requires a state initialized with "
                    "cache_jacobian=True; call solver.init(x, args, p=p)"
                )
            resid, Jt, aux = jax.lax.cond(
                state.jacobian_valid,
                lambda _: (state.resid, state.Jt, state.aux),
                lambda _: self._dense_resid_jt_aux(residual_flat, theta),
                operand=None,
            )
        else:
            resid, Jt, aux = self._dense_resid_jt_aux(residual_flat, theta)

        return residual_flat, residual_value, resid, Jt, jvp_fn, JT, aux

    def _block_sizes(self, theta_size):
        # With an explicit Metric, the metric block covers the leading
        # metric.size coordinates and the rest is free. Without an explicit
        # Metric, unpenalized mode puts all coordinates in the free block
        # (n_m = 0), while ridge mode penalizes all coordinates under the
        # Euclidean identity metric (n_m = theta_size, n_f = 0).
        if not self._has_metric:
            return (theta_size, 0) if self.penalized else (0, theta_size)
        n_f = theta_size - self.initial_metric.size
        if n_f < 0:
            raise ValueError(
                f"the metric covers {self.initial_metric.size} leading "
                f"coordinates but x flattens to only {theta_size}; the free "
                "block is len(x) - metric.size and must be nonnegative"
            )
        return self.initial_metric.size, n_f

    def _resolved_state(self, state):
        # A hand-built pre-seeding state reads through the constructor
        # instances; states from init/solve carry their own. The resolved
        # view is ephemeral (ctx only) -- update passes the input fields
        # through, so a user's own while_loop carry keeps its structure.
        metric = self.initial_metric if state.metric is None else state.metric
        precond = (
            self.initial_preconditioner
            if state.preconditioner is None
            else state.preconditioner
        )
        if metric is state.metric and precond is state.preconditioner:
            return state
        return dataclasses.replace(state, metric=metric, preconditioner=precond)

    def _factor_apply_metric_block(self, v_m, ctx, dtype):
        metric = ctx.state.metric
        if metric is None:
            return jnp.asarray(v_m, dtype=dtype)
        return jnp.asarray(metric.factor_apply(v_m, ctx), dtype=dtype)

    # The solver-internal extension F_bar = blockdiag(F, sqrt(free_scale) I):
    # the CARRIED metric's factor op on the metric block, a scalar on the
    # free block. Applied to vectors or leading-axis-batched matrices; F_bar
    # itself is never materialized. When metric is None (Euclidean identity),
    # F_bar = I and the ops return v directly.
    def _scaled_free(self, v, metric):
        return v / jnp.sqrt(jnp.asarray(metric.free_scale, v.dtype))

    def _extended_solve(self, v, ctx):
        metric = ctx.state.metric
        if metric is None:
            return v
        n_m = metric.size
        if n_m == 0:
            return self._scaled_free(v, metric)
        if v.shape[0] == n_m:
            return metric.factor_solve(v, ctx)
        return jnp.concatenate(
            [metric.factor_solve(v[:n_m], ctx), self._scaled_free(v[n_m:], metric)],
            axis=0,
        )

    def _extended_solve_transpose(self, v, ctx):
        metric = ctx.state.metric
        if metric is None:
            return v
        n_m = metric.size
        if n_m == 0:
            return self._scaled_free(v, metric)
        if v.shape[0] == n_m:
            return metric.factor_solve_transpose(v, ctx)
        return jnp.concatenate(
            [
                metric.factor_solve_transpose(v[:n_m], ctx),
                self._scaled_free(v[n_m:], metric),
            ],
            axis=0,
        )

    def _base_objective(self, resid, theta, n_m, n_f, state, ctx):
        dtype = resid.dtype
        resid_loss_old = jnp.sum(resid**2)
        if not self.penalized:
            return (
                jnp.zeros((), dtype=dtype),
                jnp.zeros(n_m, dtype=dtype),
                None,
                jnp.zeros_like(theta, dtype=dtype),
                resid_loss_old,
                resid_loss_old,
            )
        if state.ridge is None:
            raise ValueError(
                "the state has no ridge; create it with init(x, args, p=p)"
            )
        ridge = jnp.asarray(state.ridge, dtype=dtype)
        y_m = self._factor_apply_metric_block(theta[:n_m], ctx, dtype)
        penalty_value_old = jnp.sum(y_m**2)
        penalty_gradient = (
            y_m if n_f == 0 else jnp.concatenate([y_m, jnp.zeros(n_f, dtype=dtype)])
        )
        loss_old = resid_loss_old + ridge * penalty_value_old
        return (
            ridge,
            y_m,
            penalty_value_old,
            penalty_gradient,
            resid_loss_old,
            loss_old,
        )

    def _geodesic_correction(
        self,
        residual_flat,
        theta,
        velocity,
        velocity_sub,
        correction_fn,
        ctx,
        geodesic_acceptance_ratio,
        dtype,
    ):
        def first_jvp(th):
            return jax.jvp(residual_flat, (th,), (velocity,), has_aux=self.has_aux)[1]

        f_vv = jax.jvp(first_jvp, (theta,), (velocity,))[1]
        acceleration_sub = correction_fn(f_vv)
        acceleration = jnp.asarray(
            self._extended_solve(acceleration_sub, ctx), dtype=dtype
        )
        zero = jnp.zeros((), dtype=dtype)
        acceleration_ratio = (
            2.0
            * jnp.linalg.norm(acceleration_sub)
            / (jnp.linalg.norm(velocity_sub) + jnp.finfo(dtype).eps)
        )
        ratio_accepted = (
            (geodesic_acceptance_ratio > zero)
            & (acceleration_ratio > zero)
            & (acceleration_ratio <= geodesic_acceptance_ratio)
        )
        return (
            acceleration_sub,
            acceleration,
            acceleration_ratio,
            ratio_accepted,
        )

    def _ad_linearization(self, x, args, p, p_dot):
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        return theta, unravel, residual, theta_jvp, residual_p_dot

    def _ad_cg_tol(self, dtype):
        if self.ad_solver_tol is not None:
            return jnp.asarray(self.ad_solver_tol, dtype=dtype)
        default_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return jnp.asarray(default_tol, dtype=dtype)

    def _ad_preconditioner(self, state):
        # "carried": the forward instance at the solution, callback refreshes
        # included; "explicit": the ad_solver's own baked instance.
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
        # Trace-time guard mirroring the hyper contract: a replaced instance
        # must be the same registered type with matching static fields and
        # leaf avals -- weak type included, since a weak/strong scalar swap
        # passes the while-loop carry's physical checks but retraces the body
        # under different promotion rules.
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
        # The step's diagnostics and every cache describe the pre-action
        # problem, so they are stale iff the action actually changed the
        # values -- a traced comparison, so a jit-style callback that returns
        # the field every step with unchanged values changes nothing.
        xargs_changed = jnp.asarray(False)
        state_changed = jnp.asarray(False)
        metric_changed = jnp.asarray(False)
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
                    "dataclasses.replace(ctx.state, ...) to preserve the "
                    "cache fields"
                )
            self._check_action_state(state)
            # Trace-time guard so the hyper contract fails identically with
            # and without jit (jit would reject the carry mismatch anyway).
            previous_hyper = previous.hyper
            if previous_hyper is not None:
                if state.hyper is None or (
                    jax.tree_util.tree_structure(previous_hyper)
                    != jax.tree_util.tree_structure(state.hyper)
                ):
                    raise ValueError(
                        "the callback action changed the structure or dtypes of "
                        "state.hyper; reset values with "
                        "dataclasses.replace(ctx.state.hyper, ...) using arrays of "
                        "the same dtype — a knob constructed as None cannot be "
                        "enabled mid-solve"
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
                            "dataclasses.replace(ctx.state.hyper, ...) using arrays of "
                            "the same dtype — a knob constructed as None cannot be "
                            "enabled mid-solve"
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
            if self._has_metric:
                self._check_instance_structure(state.metric, previous.metric, "metric")
                # A changed metric moves the ridge objective (suppressing this
                # step's convergence test) but only re-whitens the unpenalized
                # solver's; both stale the whitening-dependent solver caches,
                # neither the Jacobian cache -- J = dr/dx does not see the
                # metric. Preconditioner changes are deliberately not compared:
                # staleness only moves the CG iteration path.
                metric_changed = _tree_changed(state.metric, previous.metric)
            self._check_instance_structure(
                state.preconditioner, previous.preconditioner, "preconditioner"
            )
            state, state_changed = self._apply_action_state(state, previous)
            if self.penalized:
                state_changed = state_changed | metric_changed
        if action.args is not None:
            xargs_changed = xargs_changed | _tree_changed(action.args, args)
            args = action.args
        if action.user_state is not None:
            user_state = action.user_state
        problem_changed = xargs_changed | state_changed
        if action.x is not None or action.args is not None:
            # The Jacobian cache describes the pre-action (x, args) and is
            # stale once either moves.
            if state.jacobian_valid is not None:
                state = dataclasses.replace(
                    state, jacobian_valid=state.jacobian_valid & ~xargs_changed
                )
        touched = (
            action.x is not None or action.args is not None or action_state is not None
        )
        state = self._invalidate_caches(
            state, action, touched, problem_changed | metric_changed
        )
        return action, x, state, args, user_state, problem_changed

    def _action_rescued(self, action, state, previous, problem_changed):
        if action is None or action.state is None:
            return problem_changed
        metric_changed = (
            _tree_changed(state.metric, previous.metric)
            if self._has_metric
            else jnp.asarray(False)
        )
        hyper_changed = _tree_changed(state.hyper, previous.hyper)
        precond_changed = _tree_changed(state.preconditioner, previous.preconditioner)
        return problem_changed | metric_changed | hyper_changed | precond_changed

    def _check_action_state(self, state):
        if self.penalized and state.ridge is None:
            raise ValueError(
                "the callback action returned a state without ridge; "
                "use dataclasses.replace(ctx.state, ...) to preserve it"
            )

    def _apply_action_state(self, state, previous):
        if not self.penalized:
            return state, jnp.asarray(False)
        # A ridge change leaves the Jacobian cache VALID (J does not depend on
        # ridge) but invalidates the ridge-keyed normal/QR caches and
        # suppresses the convergence check, whose diagnostics were computed at
        # the old ridge. The recast keeps a weak-typed callback float from
        # changing the while_loop carry aval.
        _validate_ridge_value(state.ridge, "state.ridge")
        new_ridge = jnp.asarray(state.ridge, dtype=previous.ridge.dtype)
        changed = ~jnp.array_equal(new_ridge, previous.ridge, equal_nan=True)
        return dataclasses.replace(state, ridge=new_ridge), changed

    def _invalidate_caches(self, state, action, touched, caches_stale):
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
        """Run repeated solver updates until a stopping rule fires.

        Parameters are ``update``'s plus loop controls. ``max_steps`` is always
        enforced; ``max_steps_is_success=True`` (the default) treats
        :attr:`SolverStatus.MAX_STEPS` as usable for implicit AD and for the
        default multi-start acceptance, while keeping the status for
        diagnostics.

        ``atol``/``gtol``/``xtol`` bound the residual norm, the stationarity
        residual ``info.grad_norm``, and an accepted step's ``info.step_norm``;
        ``0`` disables a check and every firing rule reports
        :attr:`SolverStatus.CONVERGED`. When ``ridge=None``, the three rules combine
        disjunctively (``atol | gtol | xtol``); when ``ridge`` is active,
        ``atol`` acts as a conjunctive filter on ``gtol | xtol``.

        ``callback`` receives a :class:`CallbackContext` after each step and may
        return a :class:`SolverAction` to stop or to override ``x``/``state``/
        ``args``/``user_state``; ``p`` is passed through but cannot be
        replaced. A callback that installs an invalid ``x`` or ``args`` must
        also stop with a failed status.

        ``save_steps=True`` records the iterate history on the result:
        ``x_history`` stacks ``x0`` and every kept post-step iterate along a
        ``(max_steps + 1)`` leading axis (rows beyond ``steps`` are zero
        padding), plus the row-aligned ``args_history`` and, with ``has_aux``,
        ``aux_history``. The buffers are differentiation-inert and make the
        jitted loop retrace when ``max_steps`` changes.

        ``multi_start`` (a :class:`MultiStart`) retries or parallelizes over
        fresh initial conditions and returns the single best result, with
        diagnostics on ``result.multi_start``.

        Implicit AD uses ``CONVERGED``, and ``MAX_STEPS`` when
        ``max_steps_is_success=True``; every failed status receives zero
        tangents for ``result.x`` and ``result.aux``, with the failed lane's
        linear tangent program evaluated at differentiation-inert copies of the
        original ``(x0, args, p)``.

        ONLY ``result.x``, ``result.aux``, and ``result.p`` carry tangents.
        Everything else -- ``info`` (including ``info.loss``), ``state``,
        ``steps``, the histories, and the multi-start diagnostics -- is
        differentiation-inert and gets EXACT ZERO, since damping, step norms,
        and iteration counts are artifacts of the path rather than properties
        of the root. Differentiating a loss therefore means recomputing it
        from ``result.x``, not reading ``result.info.loss``::

            jax.grad(lambda p: objective(solver.solve(x0, p=p).x, p))(p)
        """
        self._check_residual_args(args, p)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        # Tolerances are traced data inside the loop, so vmapped/traced values
        # skip the concrete-only sign validation.
        if not isinstance(atol, jax.core.Tracer) and atol < 0:
            raise ValueError("atol must be nonnegative")
        if not isinstance(gtol, jax.core.Tracer) and gtol < 0:
            raise ValueError("gtol must be nonnegative")
        if not isinstance(xtol, jax.core.Tracer) and xtol < 0:
            raise ValueError("xtol must be nonnegative")
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

    def _validate_supplied_ridge(self, state, dtype):
        if state.ridge is None:
            raise ValueError(
                "the caller-supplied state has no ridge; create it with "
                "init(x, args, p=p) or set a positive ridge"
            )
        _validate_ridge_value(state.ridge, "state.ridge")
        return dataclasses.replace(state, ridge=jnp.asarray(state.ridge, dtype=dtype))

    def _initial_ad_point(self, x, state, args, p):
        # The pre-loop instances (and ridge when penalized) ride along: a
        # failed lane's callback may have left invalid metric/preconditioner/
        # ridge values behind, so the failed tangent program reads these.
        if self.penalized:
            return (x, args, p, state.metric, state.preconditioner, state.ridge)
        return (x, args, p, state.metric, state.preconditioner)

    def _converged(self, info, atol, gtol, xtol):
        if self.penalized:
            # gtol/xtol mean "done with the current fixed-ridge problem"; atol is
            # a CONJUNCTIVE filter on the TRUE residual, never sufficient alone.
            gtol_met = (gtol > 0) & (info.grad_norm < gtol)
            xtol_met = (xtol > 0) & info.accepted & (info.step_norm < xtol)
            residual_ok = (atol <= 0) | (jnp.sqrt(info.resid_loss) <= atol)
            return (gtol_met | xtol_met) & residual_ok
        atol_met = (atol > 0) & (jnp.sqrt(info.loss) < atol)
        gtol_met = (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (xtol > 0) & info.accepted & (info.step_norm < xtol)
        return atol_met | gtol_met | xtol_met

    def _ranking_objective(self, result, p, callback):
        # Without a callback, info.loss at step > 0 already reports the
        # objective at the retained iterate; a callback can replace x/args/
        # ridge after the last update, so recompute. Nonfinite masks to +inf.
        if callback is None:
            loss = result.info.loss
        elif not self.penalized:
            residual = self._residual_and_aux(result.x, result.args, p)[0]
            loss = jnp.sum(residual**2)
        else:
            residual = self._residual_and_aux(result.x, result.args, p)[0]
            theta, _ = ravel_pytree(result.x)
            n_m = self._block_sizes(theta.shape[0])[0]
            resolved = self._resolved_state(result.state)
            ctx = SolverContext(x=theta, state=resolved, args=result.args, p=p)
            ridge = jnp.asarray(result.state.ridge, dtype=residual.dtype)
            y_m = self._factor_apply_metric_block(theta[:n_m], ctx, residual.dtype)
            loss = jnp.sum(residual**2) + ridge * jnp.sum(y_m**2)
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
        # A successful tangent relinearizes at the returned solution. A failed
        # tangent uses the differentiation-inert original initial point, so the
        # linear tangent program stays finite under vmap. Everything except x,
        # p, and aux is bookkeeping with zero tangents.
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
        zero_result = jax.tree.map(_zero_tangent_leaf, result)
        x_dot = self._ad_x_tangent(
            ad_x, ad_args, ad_p, ad_p_dot, result, ad_success, initial_ad_point
        )
        x_dot = _where_tree(ad_success, x_dot, zero_result.x)
        aux_dot = zero_result.aux
        if self.has_aux and ad_p is not None:
            # aux depends on p directly and through the solution x*(p).
            def aux_at_solution(x_value, p_value):
                return self.residual_fn(x_value, ad_args, p_value)[1]

            aux_dot = jax.jvp(aux_at_solution, (ad_x, ad_p), (x_dot, ad_p_dot))[1]
            aux_dot = _where_tree(ad_success, aux_dot, zero_result.aux)
        return dataclasses.replace(zero_result, x=x_dot, p=p_dot, aux=aux_dot)

    def _resolved_ad_solver(self, m=None, n=None):
        if self.penalized:
            if self.ad_solver is None:
                return "normal_cg" if isinstance(self.linear_solver, CG) else "cholesky"
            return "cholesky" if isinstance(self.ad_solver, Cholesky) else "normal_cg"
        if self.ad_solver is not None:
            if m is not None and n is not None:
                if isinstance(self.ad_solver, LU) and n != m:
                    raise ValueError(
                        f"ad_solver=LU() requires a square system (m == n), "
                        f"got m={m} and n={n}"
                    )
                if (
                    isinstance(self.ad_solver, CG)
                    and n > m
                    and self.ad_solver.penalty is None
                ):
                    raise ValueError(
                        f"ad_solver=CG(...) requires n <= m (or a positive penalty), "
                        f"got m={m} and n={n}"
                    )
                if isinstance(self.ad_solver, GramCG) and m > n:
                    raise ValueError(
                        f"ad_solver=GramCG(...) requires m <= n, got m={m} and n={n}"
                    )
            return self.ad_solver
        if isinstance(self.linear_solver, CG) and (
            m is None or n is None or n <= m or self.linear_solver.penalty is not None
        ):
            return self.linear_solver
        if isinstance(self.linear_solver, GramCG) and (
            m is None or n is None or m <= n
        ):
            return self.linear_solver
        if isinstance(self.linear_solver, SVD):
            return self.linear_solver
        if m is not None and n is not None:
            return LU() if m == n else SVD()
        return SVD()

    def _ad_x_tangent(self, x, args, p, p_dot, result, ad_success, initial_ad_point):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        state = self._resolved_state(jax.lax.stop_gradient(result.state))
        initial_instances = jax.lax.stop_gradient(initial_ad_point[3:5])
        state = dataclasses.replace(
            state,
            metric=(
                _where_tree(ad_success, state.metric, initial_instances[0])
                if self._has_metric
                else state.metric
            ),
            preconditioner=_where_tree(
                ad_success, state.preconditioner, initial_instances[1]
            ),
        )
        if self.penalized:
            final_ridge = jax.lax.stop_gradient(result.state.ridge)
            initial_ridge = jax.lax.stop_gradient(initial_ad_point[5])
            ridge = jnp.where(
                ad_success, final_ridge, jnp.asarray(initial_ridge, final_ridge.dtype)
            )
            if self._resolved_ad_solver() == "cholesky":
                return self._ad_tangent_ridge_cholesky(
                    x, args, p, p_dot, ridge, state, ad_success
                )
            return self._ad_tangent_ridge_normal_cg(
                x, args, p, p_dot, ridge, state, ad_success
            )
        return self._ad_x_tangent_unpenalized(x, args, p, p_dot, state, ad_success)

    def _ad_whiten(self, v, ctx, dtype):
        # S = F_bar^{-1} is not self-adjoint, and a matrix-free factor may
        # be opaque to JAX's transpose machinery, so declare its transpose
        # explicitly: the identity matvec exposes nothing to AD and every rule
        # routes through the declared solves.
        if ctx.state.metric is None:
            return jnp.asarray(v, dtype=dtype)
        return jax.lax.custom_linear_solve(
            lambda u: u,
            jnp.asarray(v, dtype=dtype),
            lambda _, b: jnp.asarray(self._extended_solve(b, ctx), dtype=dtype),
            transpose_solve=lambda _, b: jnp.asarray(
                self._extended_solve_transpose(b, ctx), dtype=dtype
            ),
        )

    def _ad_whiten_transpose(self, v, ctx, dtype):
        if ctx.state.metric is None:
            return jnp.asarray(v, dtype=dtype)
        return jax.lax.custom_linear_solve(
            lambda u: u,
            jnp.asarray(v, dtype=dtype),
            lambda _, b: jnp.asarray(
                self._extended_solve_transpose(b, ctx), dtype=dtype
            ),
            transpose_solve=lambda _, b: jnp.asarray(
                self._extended_solve(b, ctx), dtype=dtype
            ),
        )

    def _ad_x_tangent_unpenalized(self, x, args, p, p_dot, state, ad_success):
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        n, m = theta.shape[0], residual.shape[0]
        dtype = residual.dtype
        resolved = self._resolved_ad_solver(m, n)
        ctx = SolverContext(x=theta, state=state, args=args, p=p)
        if isinstance(resolved, LU):
            Jt = self._assemble_jt(theta_jvp, theta, residual)
            return unravel(self._ad_tangent_lu(Jt, residual_p_dot, ad_success))

        self._block_sizes(n)

        def whiten(v):
            return self._ad_whiten(v, ctx, dtype)

        def whiten_transpose(v):
            return self._ad_whiten_transpose(v, ctx, dtype)

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
            # B' = F_bar^{-T} J', shape (n, m).
            Bt = whiten_transpose(self._assemble_jt(theta_jvp, theta, residual))
            u = (
                self._ad_tangent_svd(Bt, residual_p_dot)
                if isinstance(resolved, SVD)
                else self._ad_tangent_dense(
                    Bt, residual_p_dot, ad_success, form=resolved.form
                )
            )
        theta_dot = whiten(u)
        return unravel(theta_dot)

    def _ad_tangent_dense(self, Bt, residual_p_dot, ad_success=True, form="auto"):
        # Undamped Gauss-Newton tangent: u = -B^+ (dr/dp) p_dot through the
        # requested or smaller normal/gram system. Requires full rank; a rank-
        # deficient B needs SVD(), which selects the minimum-norm tangent.
        n, m = Bt.shape
        if form == "gram" and m > n:
            raise ValueError(
                f"ad_solver=Cholesky(form='gram') requires m <= n, got m={m} and n={n}"
            )
        if form == "normal" and n > m:
            raise ValueError(
                "ad_solver=Cholesky(form='normal') requires n <= m, "
                f"got m={m} and n={n}"
            )
        use_gram = form == "gram" or (form == "auto" and n > m)
        if use_gram:
            gram = mm(Bt.T, Bt)
            gram = jnp.where(ad_success, gram, jnp.eye(m, dtype=Bt.dtype))
            factor = jsp_linalg.cho_factor(gram)
            return -mm(Bt, jsp_linalg.cho_solve(factor, residual_p_dot))
        normal = mm(Bt, Bt.T)
        normal = jnp.where(ad_success, normal, jnp.eye(n, dtype=Bt.dtype))
        factor = jsp_linalg.cho_factor(normal)
        return -jsp_linalg.cho_solve(factor, mm(Bt, residual_p_dot))

    def _ad_tangent_lu(self, Bt, residual_p_dot, ad_success=True):
        # Square B: u = -B^{-1} (dr/dp) p_dot, factored directly.
        n = Bt.shape[0]
        B = jnp.where(ad_success, Bt.T, jnp.eye(n, dtype=Bt.dtype))
        return -jnp.linalg.solve(B, residual_p_dot)

    def _ad_tangent_svd(self, Bt, residual_p_dot):
        # Spectral filter: u = -B^+ (dr/dp) p_dot, the minimum-metric-norm
        # tangent.
        return -mm(_svd_pinv(Bt), residual_p_dot)

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
        ad_preconditioner = self._ad_preconditioner(ctx.state)
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

        rhs = -Bt(residual_p_dot)
        return jax.lax.custom_linear_solve(
            normal_matvec,
            rhs,
            lambda _, c: cg(normal_matvec, c),
            symmetric=True,
        )

    def _ad_tangent_ridge_cholesky(
        self, x, args, p, p_dot, ridge, state, ad_success=True
    ):
        # The GN implicit rule posed on the whitened variable y = F_bar x:
        # (J~'J~ + ridge E) y_dot = -J~'(dr/dp) p_dot, then
        # x_dot = F_bar^{-1} y_dot -- no damping; the matrix is PD under the
        # identification condition because ridge > 0 by contract.
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        n_m, n_f = self._block_sizes(theta.shape[0])
        ctx = SolverContext(x=theta, state=state, args=args, p=p)
        Jt = self._assemble_jt(theta_jvp, theta, residual)
        ridge_typed = jnp.asarray(ridge, dtype=residual.dtype)
        Jt_sub = jnp.asarray(
            self._extended_solve_transpose(Jt, ctx), dtype=residual.dtype
        )
        if n_f == 0 and Jt_sub.shape[0] > Jt_sub.shape[1]:
            m = Jt_sub.shape[1]
            diag = jnp.arange(m)
            dual_matrix = mm(Jt_sub.T, Jt_sub).at[diag, diag].add(ridge_typed)
            dual_matrix = jnp.where(
                ad_success, dual_matrix, jnp.eye(m, dtype=residual.dtype)
            )
            factor = jsp_linalg.cho_factor(dual_matrix)
            y_dot = -mm(Jt_sub, jsp_linalg.cho_solve(factor, residual_p_dot))
        else:
            n = theta.shape[0]
            diag = jnp.arange(n_m)
            normal_matrix = mm(Jt_sub, Jt_sub.T).at[diag, diag].add(ridge_typed)
            normal_matrix = jnp.where(
                ad_success, normal_matrix, jnp.eye(n, dtype=residual.dtype)
            )
            factor = jsp_linalg.cho_factor(normal_matrix)
            y_dot = jsp_linalg.cho_solve(factor, -mm(Jt_sub, residual_p_dot))
        theta_dot = self._ad_whiten(y_dot, ctx, residual.dtype)
        return unravel(theta_dot)

    def _ad_tangent_ridge_normal_cg(
        self, x, args, p, p_dot, ridge, state, ad_success=True
    ):
        # Matrix-free CG on the same whitened PD operator; matvec = J~'(J~ u) +
        # ridge [u_m; 0].
        theta, unravel, residual, theta_jvp, residual_p_dot = self._ad_linearization(
            x, args, p, p_dot
        )
        n_m, n_f = self._block_sizes(theta.shape[0])
        ctx = SolverContext(x=theta, state=state, args=args, p=p)
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
            metric_shift = (
                u
                if n_f == 0
                else jnp.concatenate([u[:n_m], jnp.zeros(n_f, dtype=residual.dtype)])
            )
            value = gauss_newton + ridge_typed * metric_shift
            return jnp.where(ad_success, value, u)

        cg_tol = self._ad_cg_tol(residual.dtype)
        cg_atol = jnp.asarray(self.ad_solver_atol, dtype=residual.dtype)
        ad_preconditioner = self._ad_preconditioner(state)
        if ad_preconditioner is None:
            apply_M = None
        else:
            zero_damping = jnp.zeros((), dtype=residual.dtype)

            def apply_M(v):
                return jnp.where(
                    ad_success, ad_preconditioner.apply(v, zero_damping, ctx), v
                )

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

        rhs = -self._ad_whiten_transpose(JT(residual_p_dot), ctx, residual.dtype)
        y_dot = jax.lax.custom_linear_solve(
            normal_matvec,
            rhs,
            solve,
            symmetric=True,
        )
        theta_dot = self._ad_whiten(y_dot, ctx, residual.dtype)
        return unravel(theta_dot)


__all__ = ["MultiStart", "SolverBase"]
