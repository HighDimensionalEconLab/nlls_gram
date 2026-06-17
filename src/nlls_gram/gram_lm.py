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
# so a rejected step returns the unchanged params rather than branching. No floats
# are cast anywhere on the step: dtypes flow from params/residual and jax decides
# float32 vs float64 via jax_enable_x64.


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
class GramLevenbergMarquardt:
    def __init__(
        self,
        residual_fn,
        *,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        solve_method="gram",
    ):
        if solve_method not in ("gram", "normal"):
            raise ValueError(f"unknown solve_method: {solve_method}")
        self.residual_fn = residual_fn
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.solve_method = solve_method

    def init(self):
        return LMState(jnp.asarray(self.init_damping))

    def update(self, params, state, batch):
        theta, unravel = ravel_pytree(params)

        def residual_flat(th):
            return jnp.ravel(self.residual_fn(unravel(th), batch))

        resid = residual_flat(theta)
        J = jax.jacrev(residual_flat)(theta)
        if self.solve_method == "gram":
            step = -J.T @ jnp.linalg.solve(
                J @ J.T + state.damping * jnp.eye(J.shape[0], dtype=resid.dtype), resid
            )
        else:
            step = -jnp.linalg.solve(
                J.T @ J + state.damping * jnp.eye(J.shape[1], dtype=resid.dtype),
                J.T @ resid,
            )
        resid_new = residual_flat(theta + step)
        improved = jnp.sum(resid_new**2) < jnp.sum(resid**2)
        theta_new = jnp.where(improved, theta + step, theta)
        damping = jnp.where(
            improved,
            state.damping * self.damping_decrease,
            state.damping * self.damping_increase,
        )
        loss = jnp.minimum(jnp.sum(resid_new**2), jnp.sum(resid**2))
        return unravel(theta_new), LMState(damping), LMInfo(loss, improved, damping)
