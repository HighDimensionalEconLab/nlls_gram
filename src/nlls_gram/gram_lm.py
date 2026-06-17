from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
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
# acceptance and damping_increase on rejection. The solver factors the small
# damped Gram system, which is the intended use case for n_residuals << n_params:
#   step = -J' (J J' + damping I_n)^{-1} r
# regularization controls the damping metric:
#   "identity": add damping * I
#   "fletcher": add damping * clipped_diag(J'J), using the equivalent dual formula
class GramLevenbergMarquardt:
    def __init__(
        self,
        residual_fn,
        *,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        regularization="identity",
        fletcher_min_diagonal=1e-6,
        fletcher_max_diagonal=1e6,
        geodesic_acceleration=False,
        geodesic_acceptance_ratio=0.75,
    ):
        if regularization not in ("identity", "fletcher"):
            raise ValueError(f"unknown regularization: {regularization}")
        if init_damping <= 0:
            raise ValueError("init_damping must be positive")
        if fletcher_min_diagonal <= 0:
            raise ValueError("fletcher_min_diagonal must be positive")
        if fletcher_max_diagonal < fletcher_min_diagonal:
            raise ValueError(
                "fletcher_max_diagonal must be greater than or equal to "
                "fletcher_min_diagonal"
            )
        self.residual_fn = residual_fn
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.regularization = regularization
        self.fletcher_min_diagonal = fletcher_min_diagonal
        self.fletcher_max_diagonal = fletcher_max_diagonal
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio

    def init(self):
        return LMState(jnp.asarray(self.init_damping))

    def update(self, params, state, batch):
        theta, unravel = ravel_pytree(params)

        def residual_flat(th):
            return jnp.ravel(self.residual_fn(unravel(th), batch))

        resid, pullback = jax.vjp(residual_flat, theta)
        residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
        J = jax.vmap(lambda cotangent: pullback(cotangent)[0])(residual_basis)
        damping = jnp.asarray(state.damping, dtype=resid.dtype)
        damping_decrease = jnp.asarray(self.damping_decrease, dtype=resid.dtype)
        damping_increase = jnp.asarray(self.damping_increase, dtype=resid.dtype)
        if self.regularization == "fletcher":
            fletcher_diagonal = jnp.clip(
                jnp.sum(J**2, axis=0),
                jnp.asarray(self.fletcher_min_diagonal, dtype=resid.dtype),
                jnp.asarray(self.fletcher_max_diagonal, dtype=resid.dtype),
            )

        if self.regularization == "identity":
            gram_step_left = J.T
            linear_matrix = J @ J.T
        else:
            gram_step_left = J.T / fletcher_diagonal[:, None]
            linear_matrix = J @ gram_step_left
        linear_matrix = linear_matrix + damping * jnp.eye(J.shape[0], dtype=resid.dtype)
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
            resid_accelerated = residual_flat(theta + accelerated_step)
            loss_accelerated = jnp.sum(resid_accelerated**2)
            acceleration_ratio = (
                2.0
                * jnp.linalg.norm(acceleration)
                / (jnp.linalg.norm(velocity) + jnp.finfo(resid.dtype).eps)
            )
            used_geodesic = (
                (geodesic_acceptance_ratio > zero)
                & (acceleration_ratio > zero)
                & (acceleration_ratio <= geodesic_acceptance_ratio)
                & (loss_accelerated <= loss_velocity)
            )
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
