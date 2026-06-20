from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import lineax as lx
from jax.flatten_util import ravel_pytree

# General-JAX least-squares solver: a class exposing init() -> state and
# update(params, state, batch) -> (new_params, state, info). params is ANY pytree
# (a flat array, a dict, nnx.state(model, nnx.Param), ...); the solver only ravels
# and unravels it with jax.flatten_util.ravel_pytree and calls the user's
# residual_fn(params, batch). It knows nothing about flax/nnx/optax. The class does
# not jit internally -- wrap the caller's train step in jax.jit. All hyperparameters
# are static Python scalars; all data-dependent control flow is traced (jnp.where),
# so a rejected step returns the unchanged params rather than branching. Dtypes
# flow from params/residual; damping scalars are converted to the residual dtype.


class LMState(NamedTuple):
    damping: jax.Array


class LMInfo(NamedTuple):
    loss: jax.Array  # min(old, new) sum of squared residuals
    loss_old: jax.Array
    loss_candidate: jax.Array
    accepted: jax.Array
    damping: jax.Array  # post-update damping
    damping_factor: jax.Array
    used_geodesic: jax.Array
    acceleration_ratio: jax.Array


# Classic Marquardt damping on min ||r(theta)||^2: accept the step iff the sum of
# squared residuals decreases, multiplying the damping by damping_decrease on
# acceptance and damping_increase on rejection. The default solver factors the
# small damped Gram system, which is the intended use case for n_residuals <<
# n_params:
#   step = -J' (J J' + damping I_m)^{-1} r
# regularization controls the damping metric:
#   "identity": add damping * I
#   "fletcher": add damping * clipped_diag(J'J), using the equivalent dual formula
class UnderdeterminedLevenbergMarquardt:
    def __init__(
        self,
        residual_fn,
        *,
        jac="vjp",
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        regularization="identity",
        fletcher_min_diagonal=1e-6,
        fletcher_max_diagonal=1e6,
        linear_solver="cholesky",
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=8,
        lsmr_conlim=float("inf"),
        geodesic_acceleration=False,
        geodesic_acceptance_ratio=0.75,
    ):
        if jac != "vjp":
            raise ValueError(f'unknown jac: {jac}; only jac="vjp" is supported')
        if regularization not in ("identity", "fletcher"):
            raise ValueError(f"unknown regularization: {regularization}")
        if linear_solver not in ("cholesky", "qr", "cg", "lsmr"):
            raise ValueError(f"unknown linear_solver: {linear_solver}")
        if linear_solver in ("qr", "cg", "lsmr") and regularization != "identity":
            raise ValueError(
                f'linear_solver="{linear_solver}" only supports '
                'regularization="identity"'
            )
        if init_damping <= 0:
            raise ValueError("init_damping must be positive")
        if damping_decrease <= 0:
            raise ValueError("damping_decrease must be positive")
        if damping_increase <= 0:
            raise ValueError("damping_increase must be positive")
        if fletcher_min_diagonal <= 0:
            raise ValueError("fletcher_min_diagonal must be positive")
        if fletcher_max_diagonal < fletcher_min_diagonal:
            raise ValueError(
                "fletcher_max_diagonal must be greater than or equal to "
                "fletcher_min_diagonal"
            )
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
        self.residual_fn = residual_fn
        self.jac = jac
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.regularization = regularization
        self.fletcher_min_diagonal = fletcher_min_diagonal
        self.fletcher_max_diagonal = fletcher_max_diagonal
        self.linear_solver = linear_solver
        self.iterative_tol = iterative_tol
        self.iterative_atol = iterative_atol
        self.iterative_maxiter = iterative_maxiter
        self.lsmr_conlim = lsmr_conlim
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

    def update(self, params, state, batch):
        theta, unravel = ravel_pytree(params)

        def residual_flat(th):
            return jnp.ravel(self.residual_fn(unravel(th), batch))

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

            def gram_matvec(cotangent):
                return jvp_fn(JT(cotangent)) + damping * cotangent

            def solve_step(rhs):
                dual_solution, _ = jsp_sparse_linalg.cg(
                    gram_matvec,
                    rhs,
                    tol=self.iterative_tol,
                    atol=self.iterative_atol,
                    maxiter=self.iterative_maxiter,
                )
                return -JT(dual_solution)

        elif self.linear_solver == "lsmr":
            sqrt_damping = jnp.sqrt(damping)
            zero_tangent = jnp.zeros_like(theta)

            def augmented_matvec(tangent):
                return jnp.concatenate((jvp_fn(tangent), sqrt_damping * tangent))

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
                return solution.value

        else:
            residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
            Jt = jax.vmap(lambda cotangent: pullback(cotangent)[0])(residual_basis).T

            if self.linear_solver == "qr":
                if Jt.shape[0] >= Jt.shape[1]:
                    R = jnp.linalg.qr(Jt, mode="r")
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
                        return Jt @ y

                else:
                    Q, R = jnp.linalg.qr(Jt, mode="reduced")
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
                        return Q @ z

            else:
                if self.regularization == "fletcher":
                    fletcher_diagonal = jnp.clip(
                        jnp.sum(Jt**2, axis=1),
                        jnp.asarray(self.fletcher_min_diagonal, dtype=resid.dtype),
                        jnp.asarray(self.fletcher_max_diagonal, dtype=resid.dtype),
                    )

                if self.regularization == "identity":
                    gram_step_left = Jt
                    linear_matrix = Jt.T @ Jt
                else:
                    gram_step_left = Jt / fletcher_diagonal[:, None]
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
                * jnp.linalg.norm(acceleration)
                / (jnp.linalg.norm(velocity) + jnp.finfo(resid.dtype).eps)
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

        improved = loss_candidate < loss_old
        theta_new = jnp.where(improved, theta + step, theta)
        damping_factor = jnp.where(improved, damping_decrease, damping_increase)
        new_damping = damping * damping_factor
        loss = jnp.minimum(loss_candidate, loss_old)
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
            ),
        )
