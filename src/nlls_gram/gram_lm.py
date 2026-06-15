from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.flatten_util import ravel_pytree

# optax/nnx-style least-squares and root-finding solvers: each class exposes
# init() -> state and update(model, state, batch) -> (updates, state, info), where
# updates are parameter *increments* structured like nnx.split(model, wrt, ...)[1],
# meant to be applied through nnx.Optimizer(model, optax.identity(), wrt=...).
# Classes do not jit internally -- wrap the caller's train step in jax.jit. All
# hyperparameters are static Python scalars; all data-dependent control flow is
# traced (jnp.where / lax.while_loop), so a rejected step returns zero updates
# rather than branching.


# Functionalize a model into a flat parameter vector and a residual function of it:
# residual_fn(model, batch) becomes residual_flat(theta) with theta the raveled
# wrt-filtered state (non-wrt state captured and passed through unchanged).
def flat_residual(model, wrt, residual_fn, batch):
    graphdef, diff_state, rest = nnx.split(model, wrt, ...)
    theta, unravel = ravel_pytree(diff_state)

    def residual_flat(th):
        m = nnx.merge(graphdef, unravel(th), rest)
        return jnp.ravel(residual_fn(m, batch))

    return theta, unravel, residual_flat


class LMState(NamedTuple):
    damping: jax.Array


class LMInfo(NamedTuple):
    loss: jax.Array  # min(old, new) sum of squared residuals
    accepted: jax.Array
    damping: jax.Array  # post-update damping


# Classic Marquardt damping on min ||R(theta)||^2: accept the step iff the sum of
# squared residuals decreases, multiplying the damping by damping_decrease on
# acceptance and damping_increase on rejection. solve_method picks which linear
# system is factored for the identical step:
#   "gram":   step = -J' (J J' + damping I_n)^{-1} R   (n x n dual; right for n << p)
#   "normal": step = -(J'J + damping I_p)^{-1} J' R    (p x p; right for p <~ n)
class GramLevenbergMarquardt:
    def __init__(
        self,
        residual_fn,
        *,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        solve_method="gram",
        wrt=nnx.Param,
    ):
        if solve_method not in ("gram", "normal"):
            raise ValueError(f"unknown solve_method: {solve_method}")
        self.residual_fn = residual_fn
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.solve_method = solve_method
        self.wrt = wrt

    def init(self):
        return LMState(jnp.asarray(self.init_damping))

    def update(self, model, state, batch):
        theta, unravel, residual_flat = flat_residual(
            model, self.wrt, self.residual_fn, batch
        )
        resid = residual_flat(theta)
        J = jax.jacrev(residual_flat)(theta)
        if self.solve_method == "gram":
            step = -J.T @ jnp.linalg.solve(
                J @ J.T + state.damping * jnp.eye(J.shape[0]), resid
            )
        else:
            step = -jnp.linalg.solve(
                J.T @ J + state.damping * jnp.eye(J.shape[1]), J.T @ resid
            )
        resid_new = residual_flat(theta + step)
        improved = jnp.sum(resid_new**2) < jnp.sum(resid**2)
        updates_flat = jnp.where(improved, step, jnp.zeros_like(step))
        damping = jnp.where(
            improved,
            state.damping * self.damping_decrease,
            state.damping * self.damping_increase,
        )
        loss = jnp.minimum(jnp.sum(resid_new**2), jnp.sum(resid**2))
        return unravel(updates_flat), LMState(damping), LMInfo(loss, improved, damping)
