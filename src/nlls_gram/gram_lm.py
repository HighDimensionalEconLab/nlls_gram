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

# General-JAX least-squares solver: a class exposing init() -> state,
# update(params, state, aux, p) -> (new_params, state, info), and a solve()
# convenience loop. params is ANY pytree
# (a flat array, a dict, nnx.state(model, nnx.Param), ...); the solver only ravels
# and unravels it with jax.flatten_util.ravel_pytree and calls the user's
# residual_fn, which may take (params), (params, aux), or (params, aux, p).
# It knows nothing about flax/nnx/optax. update()
# does not jit internally; solve(jit=True) wraps the loop in jax.jit. All
# hyperparameters are static Python scalars; all data-dependent control flow is
# traced (jnp.where), so a rejected step returns the unchanged params rather than
# branching. Dtypes flow from params/residual; damping scalars are converted to
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
    grad_norm: jax.Array  # ||J' r|| at the pre-step params
    step_norm: jax.Array  # ||candidate step||, reported even when rejected


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveAction:
    """Optional callback action for ``solve``.

    A field left as ``None`` is unchanged. ``status`` is used only when ``stop``
    is true.
    """

    stop: Any = None
    status: Any = None
    params: Any = None
    state: Any = None
    aux: Any = None
    user_state: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveContext:
    """Information passed to a ``solve`` callback after each LM update."""

    step: jax.Array
    params: Any
    params_old: Any
    state: LMState
    state_old: LMState
    initial_state: LMState
    aux: Any
    p: Any
    user_state: Any
    info: LMInfo


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveResult:
    """Final result returned by ``solve``."""

    params: Any
    state: LMState
    info: LMInfo
    steps: jax.Array
    status: jax.Array
    aux: Any
    p: Any
    user_state: Any


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
        geodesic_acceleration=False,
        geodesic_acceptance_ratio=0.75,
    ):
        # The residual may take (params), (params, aux), or (params, aux, p) —
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
                    "(params), (params, aux), or (params, aux, p)"
                )
        if residual_arity == 1:

            def canonical_residual(params, aux, p):
                return residual_fn(params)

        elif residual_arity == 2:

            def canonical_residual(params, aux, p):
                return residual_fn(params, aux)

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

    def init(self, dtype=None):
        # Use a strongly-typed damping matching the problem dtype so init() and
        # update() produce the same jit signature (a weakly-typed scalar here
        # would force a recompile on the second step). dtype=None defers to JAX's
        # default float type (float32, or float64 when x64 is enabled); pass an
        # explicit dtype for problems that do not use the default float.
        if dtype is None:
            dtype = jnp.result_type(float)
        return LMState(jnp.asarray(self.init_damping, dtype=dtype))

    def _initial_info(self, params, state, aux, p):
        # grad_norm is a +inf sentinel (computing it would cost a Jacobian
        # before the first step) and step_norm is zero; neither can satisfy
        # gtol/xtol before any update has run.
        residual = jnp.ravel(self.residual_fn(params, aux, p))
        loss = jnp.sum(residual**2)
        zero = jnp.zeros((), dtype=residual.dtype)
        one = jnp.ones((), dtype=residual.dtype)
        return LMInfo(
            loss,
            loss,
            loss,
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(state.damping, dtype=residual.dtype),
            one,
            jnp.asarray(False, dtype=jnp.bool_),
            zero,
            jnp.asarray(jnp.inf, dtype=residual.dtype),
            zero,
        )

    def update(self, params, state, aux=None, p=None):
        self._check_residual_args(aux, p)
        theta, unravel = ravel_pytree(params)

        def residual_flat(th):
            return jnp.ravel(self.residual_fn(unravel(th), aux, p))

        if self.linear_solver in ("cg", "lsmr"):
            resid, jvp_fn = jax.linearize(residual_flat, theta)
        else:
            resid, pullback = jax.vjp(residual_flat, theta)
        damping = jnp.asarray(state.damping, dtype=resid.dtype)
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
            residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
            Jt = jax.vmap(lambda cotangent: pullback(cotangent)[0])(residual_basis).T
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
        resid_velocity = residual_flat(theta + velocity)
        loss_old = jnp.sum(resid**2)
        loss_velocity = jnp.sum(resid_velocity**2)
        zero = jnp.zeros((), dtype=resid.dtype)

        if self.geodesic_acceleration:
            geodesic_acceptance_ratio = jnp.asarray(
                self.geodesic_acceptance_ratio, dtype=resid.dtype
            )

            def first_jvp(th):
                return jax.jvp(residual_flat, (th,), (velocity,))[1]

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
                resid_accelerated = residual_flat(theta + accelerated_step)
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
        return (
            unravel(theta_new),
            LMState(new_damping),
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
            ),
        )

    def solve(
        self,
        params,
        aux=None,
        *,
        p=None,
        state=None,
        dtype=None,
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
        stop, override params/state/aux, or update user state. ``p`` is passed to
        the residual and callback but cannot be replaced by the action.
        """
        self._check_residual_args(aux, p)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if atol < 0:
            raise ValueError("atol must be nonnegative")
        if gtol < 0:
            raise ValueError("gtol must be nonnegative")
        if xtol < 0:
            raise ValueError("xtol must be nonnegative")
        if state is not None and dtype is not None:
            raise ValueError("dtype is only used when state is None")
        if state is None:
            state = self.init(dtype)

        @jax.custom_jvp
        def solve_with_implicit_p(
            params, state, aux, p, user_state, max_steps, atol, gtol, xtol
        ):
            return self._solve_impl(
                params,
                state,
                aux,
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
            params, state, aux, p, user_state, max_steps, atol, gtol, xtol = primals
            _, _, _, p_dot, _, _, _, _, _ = tangents
            result = solve_with_implicit_p(
                params, state, aux, p, user_state, max_steps, atol, gtol, xtol
            )
            params_dot = self._implicit_params_tangent_from_p(
                result.params, result.aux, result.p, p_dot
            )
            zero_result = jax.tree.map(_zero_tangent_leaf, result)
            return (
                result,
                LMSolveResult(
                    params_dot,
                    zero_result.state,
                    zero_result.info,
                    zero_result.steps,
                    zero_result.status,
                    zero_result.aux,
                    p_dot,
                    zero_result.user_state,
                ),
            )

        return solve_with_implicit_p(
            params, state, aux, p, user_state, max_steps, atol, gtol, xtol
        )

    def _solve_impl(
        self,
        params,
        state,
        aux,
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
                params,
                state,
                aux,
                p,
                user_state,
                max_steps,
                atol,
                gtol,
                xtol,
                callback,
            )
        return self._solve_python(
            params,
            state,
            aux,
            p,
            user_state,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
        )

    def _implicit_params_tangent_from_p(self, params, aux, p, p_dot):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, params)

        theta, unravel = ravel_pytree(params)

        def residual_from_theta(theta_value):
            return jnp.ravel(self.residual_fn(unravel(theta_value), aux, p))

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)
        residual_basis = jnp.eye(residual.shape[0], dtype=residual.dtype)
        theta_transpose = jax.linear_transpose(theta_jvp, theta)
        Jt = jax.vmap(lambda cotangent: theta_transpose(cotangent)[0])(residual_basis).T

        def residual_from_p(p_value):
            return jnp.ravel(self.residual_fn(params, aux, p_value))

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

    def _apply_action(self, action, params, state, aux, user_state):
        action = self._action_or_default(action)
        if action.params is not None:
            params = action.params
        if action.state is not None:
            state = action.state
        if action.aux is not None:
            aux = action.aux
        if action.user_state is not None:
            user_state = action.user_state
        return action, params, state, aux, user_state

    def _check_residual_args(self, aux, p):
        # A residual that never sees aux/p must not have them silently
        # dropped — in particular the implicit derivative with respect to p
        # would be a silent zero.
        if aux is not None and self.residual_arity < 2:
            raise ValueError(
                "aux was passed but residual_fn takes only (params); "
                "use residual_fn(params, aux)"
            )
        if p is not None and self.residual_arity < 3:
            raise ValueError(
                "p was passed but residual_fn takes no p argument; "
                "use residual_fn(params, aux, p)"
            )

    def _converged(self, info, atol, gtol, xtol):
        atol_met = (atol > 0) & (jnp.sqrt(info.loss) < atol)
        gtol_met = (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (xtol > 0) & info.accepted & (info.step_norm < xtol)
        return atol_met | gtol_met | xtol_met

    def _solve_python(
        self,
        params,
        state,
        aux,
        p,
        user_state,
        max_steps,
        atol,
        gtol,
        xtol,
        callback,
    ):
        info = self._initial_info(params, state, aux, p)
        state = LMState(jnp.asarray(state.damping, dtype=info.loss.dtype))
        initial_state = state
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
            params_old, state_old = params, state
            params, state, info = self.update(params, state, aux, p)
            if not bool(jnp.isfinite(info.loss)):
                status = LMStatus.NONFINITE
                break
            action = None
            if callback is not None:
                ctx = LMSolveContext(
                    jnp.asarray(steps, dtype=jnp.int32),
                    params,
                    params_old,
                    state,
                    state_old,
                    initial_state,
                    aux,
                    p,
                    user_state,
                    info,
                )
                action = callback(ctx)
            action, params, state, aux, user_state = self._apply_action(
                action, params, state, aux, user_state
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
        return LMSolveResult(
            params,
            state,
            info,
            jnp.asarray(steps, dtype=jnp.int32),
            jnp.asarray(status, dtype=jnp.int32),
            aux,
            p,
            user_state,
        )


def _solve_loop_impl(
    solver,
    params,
    state,
    aux,
    p,
    user_state,
    max_steps,
    atol,
    gtol,
    xtol,
    callback,
):
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)
    info = solver._initial_info(params, state, aux, p)
    # Recast so the while_loop carry dtype matches what update() returns (the
    # residual dtype), which init()'s default float may disagree with; the
    # tolerances likewise so all comparisons run in the residual dtype.
    atol = jnp.asarray(atol, dtype=info.loss.dtype)
    gtol = jnp.asarray(gtol, dtype=info.loss.dtype)
    xtol = jnp.asarray(xtol, dtype=info.loss.dtype)
    state = LMState(jnp.asarray(state.damping, dtype=info.loss.dtype))
    initial_state = state
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
        params, state, aux, user_state, info, step, status, stop = carry
        del info, status, stop
        params_old, state_old = params, state
        params, state, info = solver.update(params, state, aux, p)
        step = step + jnp.asarray(1, dtype=jnp.int32)
        current_nonfinite = ~jnp.isfinite(info.loss)

        action = None
        if callback is not None:
            ctx = LMSolveContext(
                step,
                params,
                params_old,
                state,
                state_old,
                initial_state,
                aux,
                p,
                user_state,
                info,
            )
            action = callback(ctx)
        action, params, state, aux, user_state = solver._apply_action(
            action, params, state, aux, user_state
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
        return params, state, aux, user_state, info, step, status, stop

    carry = jax.lax.while_loop(
        cond,
        body,
        (params, state, aux, user_state, info, step, status, stop),
    )
    params, state, aux, user_state, info, step, status, _ = carry
    return LMSolveResult(params, state, info, step, status, aux, p, user_state)


_solve_loop_jit = jax.jit(_solve_loop_impl, static_argnums=(0, 10))
