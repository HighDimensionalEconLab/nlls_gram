from typing import NamedTuple

import jax
import jax.numpy as jnp
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
    accepted: jax.Array
    damping: jax.Array  # post-update damping


# Classic Marquardt damping on min ||r(theta)||^2: accept the step iff the sum of
# squared residuals decreases, multiplying the damping by damping_decrease on
# acceptance and damping_increase on rejection. solve_method picks which linear
# system is factored for the identical step:
#   "gram":   step = -J' (J J' + damping I_n)^{-1} r   (n x n dual; right for n << p)
#   "normal": step = -(J'J + damping I_p)^{-1} J' r    (p x p; right for p <~ n)
# regularization controls the damping metric:
#   "identity": add damping * I
#   "fletcher": add damping * diag(J'J), using the equivalent dual formula
#               in gram mode
class GramLevenbergMarquardt:
    def __init__(
        self,
        residual_fn,
        *,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        solve_method="gram",
        regularization="identity",
    ):
        if solve_method not in ("gram", "normal"):
            raise ValueError(f"unknown solve_method: {solve_method}")
        if regularization not in ("identity", "fletcher"):
            raise ValueError(f"unknown regularization: {regularization}")
        self.residual_fn = residual_fn
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.solve_method = solve_method
        self.regularization = regularization

    def init(self):
        return LMState(jnp.asarray(self.init_damping))

    def update(self, params, state, batch):
        theta, unravel = ravel_pytree(params)

        def residual_flat(th):
            return jnp.ravel(self.residual_fn(unravel(th), batch))

        resid = residual_flat(theta)
        damping = jnp.asarray(state.damping, dtype=resid.dtype)
        damping_decrease = jnp.asarray(self.damping_decrease, dtype=resid.dtype)
        damping_increase = jnp.asarray(self.damping_increase, dtype=resid.dtype)
        J = jax.jacrev(residual_flat)(theta)
        if self.regularization == "fletcher":
            fletcher_diagonal = jnp.maximum(
                jnp.sum(J**2, axis=0), jnp.finfo(resid.dtype).eps
            )
        if self.solve_method == "gram":
            if self.regularization == "identity":
                step = -J.T @ jnp.linalg.solve(
                    J @ J.T + damping * jnp.eye(J.shape[0], dtype=resid.dtype),
                    resid,
                )
            else:
                weighted_JT = J.T / fletcher_diagonal[:, None]
                step = -weighted_JT @ jnp.linalg.solve(
                    J @ weighted_JT + damping * jnp.eye(J.shape[0], dtype=resid.dtype),
                    resid,
                )
        else:
            normal_matrix = J.T @ J
            if self.regularization == "identity":
                regularizer = jnp.eye(J.shape[1], dtype=resid.dtype)
            else:
                regularizer = jnp.diag(fletcher_diagonal)
            step = -jnp.linalg.solve(normal_matrix + damping * regularizer, J.T @ resid)
        resid_new = residual_flat(theta + step)
        loss_old = jnp.sum(resid**2)
        loss_new = jnp.sum(resid_new**2)
        improved = loss_new < loss_old
        theta_new = jnp.where(improved, theta + step, theta)
        new_damping = jnp.where(
            improved,
            damping * damping_decrease,
            damping * damping_increase,
        )
        loss = jnp.minimum(loss_new, loss_old)
        return (
            unravel(theta_new),
            LMState(new_damping),
            LMInfo(loss, improved, new_damping),
        )
