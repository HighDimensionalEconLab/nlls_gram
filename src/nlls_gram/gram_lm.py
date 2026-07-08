import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import lineax as lx
from jax.flatten_util import ravel_pytree

from nlls_gram.metrics import Metric

# General-JAX least-squares solver: a class exposing init() -> lm_state,
# update(x, lm_state, args, p) -> (new_x, lm_state, info), and a solve()
# convenience loop. x is ANY pytree
# (a flat array, a dict, nnx.state(model, nnx.Param), ...); the solver only ravels
# and unravels it with jax.flatten_util.ravel_pytree and calls the user's
# residual_fn, which may take (x), (x, args), or (x, args, p).
# It knows nothing about flax/nnx/optax. update()
# does not jit internally; solve(jit=True) wraps the loop in jax.jit. All
# hyperparameters are static Python scalars; all data-dependent control flow is
# traced (jnp.where), so a rejected step returns the unchanged x rather than
# branching. Dtypes flow from x/residual; damping scalars are converted to
# the residual dtype.


class LMStatus:
    """Integer status codes returned by ``solve``."""

    RUNNING = 0
    CONVERGED = 1
    MAX_STEPS = 2
    NONFINITE = 3
    CALLBACK_STOP = 4


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMState:
    damping: jax.Array
    # Jacobian cache, populated only when cache_jacobian=True: the residual
    # (and its aux output when has_aux=True) and J' at the current x,
    # and whether they are still valid (the last step was rejected, so x
    # did not move).
    resid: Any = None
    Jt: Any = None
    jacobian_valid: Any = None
    aux: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMInfo:
    loss: jax.Array  # min(old, new) sum of squared residuals
    loss_old: jax.Array
    loss_candidate: jax.Array
    accepted: jax.Array
    damping: jax.Array  # post-update damping
    damping_factor: jax.Array
    used_geodesic: jax.Array
    acceleration_ratio: jax.Array
    grad_norm: jax.Array  # ||J' r|| at the pre-step x
    step_norm: jax.Array  # ||candidate step||, reported even when rejected
    aux: Any = None  # residual aux output at the pre-step x (has_aux)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveAction:
    """Optional callback action for ``solve``.

    A field left as ``None`` is unchanged. ``status`` is used only when ``stop``
    is true.
    """

    stop: Any = None
    status: Any = None
    x: Any = None
    lm_state: Any = None
    args: Any = None
    user_state: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveContext:
    """Information passed to a ``solve`` callback after each LM update."""

    step: jax.Array
    x: Any
    x_old: Any
    lm_state: LMState
    lm_state_old: LMState
    initial_lm_state: LMState
    args: Any
    p: Any
    user_state: Any
    info: LMInfo


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveResult:
    """Final result returned by ``solve``."""

    x: Any
    lm_state: LMState
    info: LMInfo
    steps: jax.Array
    status: jax.Array
    args: Any
    p: Any
    user_state: Any
    # With has_aux=True: the residual's aux output evaluated at the returned
    # (x, args, p) — one extra residual evaluation, well-defined for
    # every status since x are always the last accepted iterate.
    aux: Any = None


def _zero_tangent_leaf(leaf):
    if leaf is None:
        return None
    array = jnp.asarray(leaf)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return jnp.zeros(array.shape, dtype=jax.dtypes.float0)
    return jnp.zeros_like(leaf)


# Classic Marquardt damping on min ||r(theta)||^2: accept the step iff the sum of
# squared residuals decreases, multiplying the damping by damping_decrease on
# acceptance and damping_increase on rejection. The default solver factors the
# small damped Gram system, which is the intended use case for n_residuals <<
# n_params. With metric M and P = M^{-1}, the dense dual step is:
#   step = -P J' (J P J' + damping I_m)^{-1} r
class UnderdeterminedLevenbergMarquardt:
    def __init__(
        self,
        residual_fn,
        *,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        max_damping=None,
        linear_solver="cholesky",
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=8,
        lsmr_conlim=float("inf"),
        metric=None,
        has_aux=False,
        cache_jacobian=False,
        geodesic_acceleration=False,
        geodesic_acceptance_ratio=0.75,
    ):
        # The residual may take (x), (x, args), or (x, args, p) —
        # always in that order. Arity is inspected once here and the function
        # is wrapped into the canonical 3-arg form, so the compiled code is
        # identical for all three. Callables whose signature cannot be
        # inspected (or that take *args) are assumed to be 3-arg.
        try:
            signature = inspect.signature(residual_fn)
        except (TypeError, ValueError):
            residual_arity = 3
        else:
            residual_arity = 0
            for parameter in signature.parameters.values():
                if parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    residual_arity += 1
                elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                    residual_arity = 3
                    break
            if residual_arity < 1 or residual_arity > 3:
                raise ValueError(
                    "residual_fn must take 1 to 3 positional arguments: "
                    "(x), (x, args), or (x, args, p)"
                )
        if residual_arity == 1:

            def canonical_residual(x, args, p):
                return residual_fn(x)

        elif residual_arity == 2:

            def canonical_residual(x, args, p):
                return residual_fn(x, args)

        else:
            canonical_residual = residual_fn
        if linear_solver not in ("cholesky", "qr", "cg", "lsmr"):
            raise ValueError(f"unknown linear_solver: {linear_solver}")
        if init_damping <= 0:
            raise ValueError("init_damping must be positive")
        if damping_decrease <= 0:
            raise ValueError("damping_decrease must be positive")
        if damping_increase <= 0:
            raise ValueError("damping_increase must be positive")
        if max_damping is not None and max_damping < init_damping:
            raise ValueError("max_damping must be at least init_damping")
        if iterative_tol < 0:
            raise ValueError("iterative_tol must be nonnegative")
        if iterative_atol < 0:
            raise ValueError("iterative_atol must be nonnegative")
        if iterative_maxiter is not None and iterative_maxiter <= 0:
            raise ValueError("iterative_maxiter must be positive or None")
        if iterative_tol == 0 and iterative_atol == 0 and iterative_maxiter is None:
            raise ValueError(
                "iterative_maxiter must be set when both iterative tolerances are zero"
            )
        if lsmr_conlim <= 0:
            raise ValueError("lsmr_conlim must be positive")
        if metric is None:
            metric = Metric()
        has_custom_metric = any(
            cb is not None
            for cb in (
                metric.solve,
                metric.norm,
                metric.inv_sqrt,
                metric.inv_sqrt_transpose,
            )
        )
        if has_custom_metric and linear_solver in ("cholesky", "cg"):
            if metric.solve is None:
                raise ValueError(
                    f'linear_solver="{linear_solver}" with a custom metric requires '
                    "metric.solve"
                )
        if has_custom_metric and linear_solver in ("qr", "lsmr"):
            if metric.inv_sqrt is None or metric.inv_sqrt_transpose is None:
                raise ValueError(
                    f'linear_solver="{linear_solver}" with a custom metric requires '
                    "metric.inv_sqrt and metric.inv_sqrt_transpose"
                )
        if has_custom_metric and geodesic_acceleration and metric.norm is None:
            raise ValueError(
                "geodesic_acceleration with a custom metric requires metric.norm"
            )
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.max_damping = max_damping
        self.linear_solver = linear_solver
        self.iterative_tol = iterative_tol
        self.iterative_atol = iterative_atol
        self.iterative_maxiter = iterative_maxiter
        self.lsmr_conlim = lsmr_conlim
        self.metric = metric
        # The cache only pays for the dense cholesky path, where rejected
        # steps otherwise redo the m VJP passes at an unchanged point; the
        # matrix-free solvers never materialize J, so there is nothing to
        # cache and the flag is inert for them (and for qr).
        self.cache_jacobian = cache_jacobian and linear_solver == "cholesky"
        self.has_aux = has_aux
        self._has_custom_metric = has_custom_metric
        self._has_metric_solve = metric.solve is not None
        self.metric_solve = (lambda x: x) if metric.solve is None else metric.solve
        self.metric_norm = (
            (lambda x: jnp.linalg.norm(x)) if metric.norm is None else metric.norm
        )
        self.metric_inv_sqrt = (
            (lambda x: x) if metric.inv_sqrt is None else metric.inv_sqrt
        )
        self.metric_inv_sqrt_transpose = (
            (lambda x: x)
            if metric.inv_sqrt_transpose is None
            else metric.inv_sqrt_transpose
        )
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio

    def init(self, x0, args=None, *, p=None):
        # One residual evaluation infers the problem dtype, so the damping is
        # strongly typed to match what update() returns (a mismatched or
        # weakly-typed scalar here would force a recompile on the second step
        # or break the solve loop carry). The same evaluation sizes the
        # Jacobian cache buffers when cache_jacobian=True.
        self._check_residual_args(args, p)
        residual, aux = self._residual_and_aux(x0, args, p)
        damping = jnp.asarray(self.init_damping, dtype=residual.dtype)
        if not self.cache_jacobian:
            return LMState(damping)
        theta, _ = ravel_pytree(x0)
        return LMState(
            damping,
            jnp.zeros(residual.shape, dtype=residual.dtype),
            jnp.zeros((theta.size, residual.size), dtype=residual.dtype),
            jnp.asarray(False, dtype=jnp.bool_),
            jax.tree.map(jnp.zeros_like, aux),
        )

    def _residual_and_aux(self, x, args, p):
        if self.has_aux:
            value, aux = self.residual_fn(x, args, p)
            return jnp.ravel(value), aux
        return jnp.ravel(self.residual_fn(x, args, p)), None

    def _initial_info(self, x, lm_state, args, p):
        # grad_norm is a +inf sentinel (computing it would cost a Jacobian
        # before the first step) and step_norm is zero; neither can satisfy
        # gtol/xtol before any update has run.
        residual, aux = self._residual_and_aux(x, args, p)
        loss = jnp.sum(residual**2)
        zero = jnp.zeros((), dtype=residual.dtype)
        one = jnp.ones((), dtype=residual.dtype)
        return LMInfo(
            loss,
            loss,
            loss,
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(lm_state.damping, dtype=residual.dtype),
            one,
            jnp.asarray(False, dtype=jnp.bool_),
            zero,
            jnp.asarray(jnp.inf, dtype=residual.dtype),
            zero,
            aux,
        )

    def update(self, x, lm_state, args=None, p=None):
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

        if self.linear_solver in ("cg", "lsmr"):
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
                if self.has_aux:
                    resid, pullback, aux = jax.vjp(residual_flat, theta, has_aux=True)
                else:
                    resid, pullback = jax.vjp(residual_flat, theta)
                    aux = None
                residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
                Jt = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
                    residual_basis
                ).T
                return resid, Jt, aux

            def reuse_resid_and_jt(_):
                return lm_state.resid, lm_state.Jt, lm_state.aux

            resid, Jt, aux = jax.lax.cond(
                lm_state.jacobian_valid,
                reuse_resid_and_jt,
                compute_resid_and_jt,
                operand=None,
            )
        else:
            if self.has_aux:
                resid, pullback, aux = jax.vjp(residual_flat, theta, has_aux=True)
            else:
                resid, pullback = jax.vjp(residual_flat, theta)
                aux = None
        damping = jnp.asarray(lm_state.damping, dtype=resid.dtype)
        damping_decrease = jnp.asarray(self.damping_decrease, dtype=resid.dtype)
        damping_increase = jnp.asarray(self.damping_increase, dtype=resid.dtype)

        if self.linear_solver == "cg":
            transpose_fn = jax.linear_transpose(jvp_fn, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

            grad = JT(resid)
            # Typed tolerances keep CG's internal scalar ops in the residual
            # dtype when x64 is enabled for a float32 problem.
            cg_tol = jnp.asarray(self.iterative_tol, dtype=resid.dtype)
            cg_atol = jnp.asarray(self.iterative_atol, dtype=resid.dtype)

            def gram_matvec(cotangent):
                return jvp_fn(self.metric_solve(JT(cotangent))) + damping * cotangent

            def solve_step(rhs):
                dual_solution, _ = jsp_sparse_linalg.cg(
                    gram_matvec,
                    rhs,
                    tol=cg_tol,
                    atol=cg_atol,
                    maxiter=self.iterative_maxiter,
                )
                return -self.metric_solve(JT(dual_solution))

        elif self.linear_solver == "lsmr":
            transpose_fn = jax.linear_transpose(jvp_fn, theta)
            grad = transpose_fn(resid)[0]
            sqrt_damping = jnp.sqrt(damping)
            zero_tangent = jnp.zeros_like(theta)

            def augmented_matvec(tangent):
                return jnp.concatenate(
                    (
                        jvp_fn(self.metric_inv_sqrt(tangent)),
                        sqrt_damping * tangent,
                    )
                )

            augmented_operator = lx.FunctionLinearOperator(
                augmented_matvec,
                jax.ShapeDtypeStruct(theta.shape, theta.dtype),
            )
            lsmr_solver = lx.LSMR(
                rtol=self.iterative_tol,
                atol=self.iterative_atol,
                max_steps=self.iterative_maxiter,
                conlim=self.lsmr_conlim,
            )

            def solve_step(rhs):
                augmented_rhs = jnp.concatenate((-rhs, zero_tangent))
                solution = lx.linear_solve(
                    augmented_operator,
                    augmented_rhs,
                    solver=lsmr_solver,
                    throw=False,
                )
                return self.metric_inv_sqrt(solution.value)

        else:
            if not self.cache_jacobian:
                residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
                Jt = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
                    residual_basis
                ).T
            grad = Jt @ resid

            if self.linear_solver == "qr":
                transformed_Jt = self.metric_inv_sqrt_transpose(Jt)
                if transformed_Jt.shape[0] >= transformed_Jt.shape[1]:
                    R = jnp.linalg.qr(transformed_Jt, mode="r")
                    basis_eye = jnp.eye(R.shape[0], dtype=resid.dtype)
                    augmented_matrix = jnp.concatenate(
                        (R.T, jnp.sqrt(damping) * basis_eye),
                        axis=0,
                    )
                    Qa, Ra = jnp.linalg.qr(augmented_matrix, mode="reduced")

                    def solve_step(rhs):
                        augmented_rhs = jnp.concatenate(
                            (-rhs, jnp.zeros(R.shape[0], dtype=rhs.dtype))
                        )
                        z = jsp_linalg.solve_triangular(
                            Ra,
                            Qa.T @ augmented_rhs,
                        )
                        y = jsp_linalg.solve_triangular(R, z)
                        return self.metric_inv_sqrt(transformed_Jt @ y)

                else:
                    Q, R = jnp.linalg.qr(transformed_Jt, mode="reduced")
                    basis_eye = jnp.eye(R.shape[0], dtype=resid.dtype)
                    augmented_matrix = jnp.concatenate(
                        (R.T, jnp.sqrt(damping) * basis_eye),
                        axis=0,
                    )
                    Qa, Ra = jnp.linalg.qr(augmented_matrix, mode="reduced")

                    def solve_step(rhs):
                        augmented_rhs = jnp.concatenate(
                            (-rhs, jnp.zeros(R.shape[0], dtype=rhs.dtype))
                        )
                        z = jsp_linalg.solve_triangular(
                            Ra,
                            Qa.T @ augmented_rhs,
                        )
                        return self.metric_inv_sqrt(Q @ z)

            else:
                gram_step_left = self.metric_solve(Jt)
                linear_matrix = Jt.T @ gram_step_left
                linear_matrix = linear_matrix + damping * jnp.eye(
                    resid.shape[0], dtype=resid.dtype
                )

                linear_factor = jsp_linalg.cho_factor(linear_matrix)

                def solve_step(rhs):
                    return -gram_step_left @ jsp_linalg.cho_solve(linear_factor, rhs)

        velocity = solve_step(resid)
        resid_velocity = residual_value(theta + velocity)
        loss_old = jnp.sum(resid**2)
        loss_velocity = jnp.sum(resid_velocity**2)
        zero = jnp.zeros((), dtype=resid.dtype)

        if self.geodesic_acceleration:
            geodesic_acceptance_ratio = jnp.asarray(
                self.geodesic_acceptance_ratio, dtype=resid.dtype
            )

            def first_jvp(th):
                # [1] is the tangent with and without has_aux.
                return jax.jvp(residual_flat, (th,), (velocity,), has_aux=self.has_aux)[
                    1
                ]

            f_vv = jax.jvp(first_jvp, (theta,), (velocity,))[1]
            acceleration = solve_step(f_vv)
            accelerated_step = velocity + 0.5 * acceleration
            acceleration_ratio = (
                2.0
                * self.metric_norm(acceleration)
                / (self.metric_norm(velocity) + jnp.finfo(resid.dtype).eps)
            )
            ratio_accepted = (
                (geodesic_acceptance_ratio > zero)
                & (acceleration_ratio > zero)
                & (acceleration_ratio <= geodesic_acceptance_ratio)
            )

            def accelerated_loss(_):
                resid_accelerated = residual_value(theta + accelerated_step)
                return jnp.sum(resid_accelerated**2)

            loss_accelerated = jax.lax.cond(
                ratio_accepted,
                accelerated_loss,
                lambda _: jnp.asarray(jnp.inf, dtype=resid.dtype),
                operand=None,
            )
            used_geodesic = ratio_accepted & (loss_accelerated <= loss_velocity)
            step = jnp.where(used_geodesic, accelerated_step, velocity)
            loss_candidate = jnp.where(used_geodesic, loss_accelerated, loss_velocity)
        else:
            step = velocity
            loss_candidate = loss_velocity
            used_geodesic = jnp.asarray(False)
            acceleration_ratio = zero

        improved = jnp.isfinite(loss_candidate) & (loss_candidate < loss_old)
        theta_new = jnp.where(improved, theta + step, theta)
        damping_factor = jnp.where(improved, damping_decrease, damping_increase)
        new_damping = damping * damping_factor
        if self.max_damping is not None:
            new_damping = jnp.minimum(
                new_damping, jnp.asarray(self.max_damping, dtype=resid.dtype)
            )
        loss = jnp.where(improved, loss_candidate, loss_old)
        if self.cache_jacobian:
            new_lm_state = LMState(new_damping, resid, Jt, ~improved, aux)
        else:
            new_lm_state = LMState(new_damping)
        return (
            unravel(theta_new),
            new_lm_state,
            LMInfo(
                loss,
                loss_old,
                loss_candidate,
                improved,
                new_damping,
                damping_factor,
                used_geodesic,
                acceleration_ratio,
                jnp.linalg.norm(grad),
                jnp.linalg.norm(step),
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
        atol=0.0,
        gtol=0.0,
        xtol=0.0,
        callback=None,
        user_state=None,
        jit=True,
    ):
        """Run repeated LM updates until a stopping rule fires.

        Parameters are the same as ``update`` plus loop controls. ``max_steps``
        is always enforced. ``atol`` stops when the residual norm is below the
        threshold, ``gtol`` when the gradient norm ``||J' r||`` is below the
        threshold, and ``xtol`` when an accepted step has norm below the
        threshold; each tolerance set to ``0`` disables that check, and all
        three report ``LMStatus.CONVERGED``. ``callback`` receives an
        ``LMSolveContext`` after each step and may return an ``LMSolveAction`` to
        stop, override x/lm_state/args, or update user lm_state. ``p`` is passed to
        the residual and callback but cannot be replaced by the action.
        """
        self._check_residual_args(args, p)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if atol < 0:
            raise ValueError("atol must be nonnegative")
        if gtol < 0:
            raise ValueError("gtol must be nonnegative")
        if xtol < 0:
            raise ValueError("xtol must be nonnegative")
        if lm_state is None:
            # init() would evaluate the residual eagerly just for the dtype,
            # which the loop's recast makes unnecessary; only the Jacobian
            # cache genuinely needs the shapes.
            if self.cache_jacobian:
                lm_state = self.init(x0, args, p=p)
            else:
                lm_state = LMState(jnp.asarray(self.init_damping))

        @jax.custom_jvp
        def solve_with_implicit_p(
            x, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
        ):
            return self._solve_impl(
                x,
                lm_state,
                args,
                p,
                user_state,
                max_steps,
                atol,
                gtol,
                xtol,
                callback,
                jit,
            )

        @solve_with_implicit_p.defjvp
        def solve_with_implicit_p_jvp(primals, tangents):
            x, lm_state, args, p, user_state, max_steps, atol, gtol, xtol = primals
            _, _, _, p_dot, _, _, _, _, _ = tangents
            result = solve_with_implicit_p(
                x, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
            )
            x_dot = self._implicit_x_tangent_from_p(
                result.x, result.args, result.p, p_dot
            )
            zero_result = jax.tree.map(_zero_tangent_leaf, result)
            return (
                result,
                LMSolveResult(
                    x_dot,
                    zero_result.lm_state,
                    zero_result.info,
                    zero_result.steps,
                    zero_result.status,
                    zero_result.args,
                    p_dot,
                    zero_result.user_state,
                    zero_result.aux,
                ),
            )

        return solve_with_implicit_p(
            x0, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
        )

    def _solve_impl(
        self,
        x,
        lm_state,
        args,
        p,
        user_state,
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
                max_steps,
                atol,
                gtol,
                xtol,
                callback,
            )
        return self._solve_python(
            x,
            lm_state,
            args,
            p,
            user_state,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
        )

    def _implicit_x_tangent_from_p(self, x, args, p, p_dot):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)

        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)
        residual_basis = jnp.eye(residual.shape[0], dtype=residual.dtype)
        theta_transpose = jax.linear_transpose(theta_jvp, theta)
        Jt = jax.vmap(lambda cotangent: theta_transpose(cotangent)[0])(residual_basis).T

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        gram_step_left = self._metric_inverse(Jt)
        gram = Jt.T @ gram_step_left
        factor = jsp_linalg.cho_factor(gram)
        theta_dot = -gram_step_left @ jsp_linalg.cho_solve(factor, residual_p_dot)
        return unravel(theta_dot)

    def _metric_inverse(self, x):
        if self._has_metric_solve:
            return self.metric_solve(x)
        return self.metric_inv_sqrt(self.metric_inv_sqrt_transpose(x))

    def _action_or_default(self, action):
        if action is None:
            return LMSolveAction()
        return action

    def _apply_action(self, action, x, lm_state, args, user_state):
        action = self._action_or_default(action)
        if action.x is not None:
            x = action.x
        if action.lm_state is not None:
            lm_state = action.lm_state
        if action.args is not None:
            args = action.args
        if action.user_state is not None:
            user_state = action.user_state
        # A callback that returns x or args may have changed the point the
        # cached Jacobian was computed at, so invalidate conservatively. The
        # check is structural (did the action include the field), which is why
        # it composes with jit: values are traced, presence is not.
        if self.cache_jacobian and (action.x is not None or action.args is not None):
            lm_state = dataclasses.replace(
                lm_state, jacobian_valid=jnp.asarray(False, dtype=jnp.bool_)
            )
        return action, x, lm_state, args, user_state

    def _check_residual_args(self, args, p):
        # A residual that never sees args/p must not have them silently
        # dropped — in particular the implicit derivative with respect to p
        # would be a silent zero.
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
        atol_met = (atol > 0) & (jnp.sqrt(info.loss) < atol)
        gtol_met = (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (xtol > 0) & info.accepted & (info.step_norm < xtol)
        return atol_met | gtol_met | xtol_met

    def _solve_python(
        self,
        x,
        lm_state,
        args,
        p,
        user_state,
        max_steps,
        atol,
        gtol,
        xtol,
        callback,
    ):
        info = self._initial_info(x, lm_state, args, p)
        lm_state = dataclasses.replace(
            lm_state, damping=jnp.asarray(lm_state.damping, dtype=info.loss.dtype)
        )
        initial_lm_state = lm_state
        status = LMStatus.RUNNING
        steps = 0
        if not bool(jnp.isfinite(info.loss)):
            status = LMStatus.NONFINITE
        elif bool(self._converged(info, atol, gtol, xtol)):
            status = LMStatus.CONVERGED

        for steps in range(1, max_steps + 1):
            if status != LMStatus.RUNNING:
                steps -= 1
                break
            x_old, lm_state_old = x, lm_state
            x, lm_state, info = self.update(x, lm_state, args, p)
            if not bool(jnp.isfinite(info.loss)):
                status = LMStatus.NONFINITE
                break
            action = None
            if callback is not None:
                ctx = LMSolveContext(
                    jnp.asarray(steps, dtype=jnp.int32),
                    x,
                    x_old,
                    lm_state,
                    lm_state_old,
                    initial_lm_state,
                    args,
                    p,
                    user_state,
                    info,
                )
                action = callback(ctx)
            action, x, lm_state, args, user_state = self._apply_action(
                action, x, lm_state, args, user_state
            )
            if action.stop is not None and bool(action.stop):
                status = (
                    LMStatus.CALLBACK_STOP
                    if action.status is None
                    else int(action.status)
                )
                break
            if bool(self._converged(info, atol, gtol, xtol)):
                status = LMStatus.CONVERGED
                break
        else:
            steps = max_steps

        if status == LMStatus.RUNNING:
            status = LMStatus.MAX_STEPS
        final_aux = None
        if self.has_aux:
            final_aux = self._residual_and_aux(x, args, p)[1]
        return LMSolveResult(
            x,
            lm_state,
            info,
            jnp.asarray(steps, dtype=jnp.int32),
            jnp.asarray(status, dtype=jnp.int32),
            args,
            p,
            user_state,
            final_aux,
        )


def _solve_loop_impl(
    solver,
    x,
    lm_state,
    args,
    p,
    user_state,
    max_steps,
    atol,
    gtol,
    xtol,
    callback,
):
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)
    info = solver._initial_info(x, lm_state, args, p)
    # Recast so the while_loop carry dtype matches what update() returns (the
    # residual dtype), which init()'s default float may disagree with; the
    # tolerances likewise so all comparisons run in the residual dtype.
    atol = jnp.asarray(atol, dtype=info.loss.dtype)
    gtol = jnp.asarray(gtol, dtype=info.loss.dtype)
    xtol = jnp.asarray(xtol, dtype=info.loss.dtype)
    lm_state = dataclasses.replace(
        lm_state, damping=jnp.asarray(lm_state.damping, dtype=info.loss.dtype)
    )
    initial_lm_state = lm_state
    step = jnp.asarray(0, dtype=jnp.int32)
    initial_nonfinite = ~jnp.isfinite(info.loss)
    initial_converged = solver._converged(info, atol, gtol, xtol)
    stop = initial_nonfinite | initial_converged
    status = jnp.where(
        initial_nonfinite,
        jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
        jnp.where(
            initial_converged,
            jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
            jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
        ),
    )

    def cond(carry):
        _, _, _, _, _, step, status, stop = carry
        del status
        return (~stop) & (step < max_steps)

    def body(carry):
        x, lm_state, args, user_state, info, step, status, stop = carry
        del info, status, stop
        x_old, lm_state_old = x, lm_state
        x, lm_state, info = solver.update(x, lm_state, args, p)
        step = step + jnp.asarray(1, dtype=jnp.int32)
        current_nonfinite = ~jnp.isfinite(info.loss)

        action = None
        if callback is not None:
            ctx = LMSolveContext(
                step,
                x,
                x_old,
                lm_state,
                lm_state_old,
                initial_lm_state,
                args,
                p,
                user_state,
                info,
            )
            action = callback(ctx)
        action, x, lm_state, args, user_state = solver._apply_action(
            action, x, lm_state, args, user_state
        )

        callback_stop = (
            jnp.asarray(False, dtype=jnp.bool_) if action.stop is None else action.stop
        )
        callback_status = (
            jnp.asarray(LMStatus.CALLBACK_STOP, dtype=jnp.int32)
            if action.status is None
            else jnp.asarray(action.status, dtype=jnp.int32)
        )
        converged = solver._converged(info, atol, gtol, xtol)
        reached_max = step >= max_steps
        stop = current_nonfinite | callback_stop | converged | reached_max
        status = jnp.where(
            current_nonfinite,
            jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                callback_stop,
                callback_status,
                jnp.where(
                    converged,
                    jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
                    jnp.where(
                        reached_max,
                        jnp.asarray(LMStatus.MAX_STEPS, dtype=jnp.int32),
                        jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
                    ),
                ),
            ),
        )
        return x, lm_state, args, user_state, info, step, status, stop

    carry = jax.lax.while_loop(
        cond,
        body,
        (x, lm_state, args, user_state, info, step, status, stop),
    )
    x, lm_state, args, user_state, info, step, status, _ = carry
    final_aux = None
    if solver.has_aux:
        final_aux = solver._residual_and_aux(x, args, p)[1]
    return LMSolveResult(
        x, lm_state, info, step, status, args, p, user_state, final_aux
    )


_solve_loop_jit = jax.jit(_solve_loop_impl, static_argnums=(0, 10))
