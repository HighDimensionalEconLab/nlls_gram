"""Gram/dual-form Levenberg-Marquardt nonlinear least-squares for JAX.

GramLevenbergMarquardt minimizes ||r(params)||^2 for a user-supplied
residual_fn(params, batch), where params is any JAX pytree (a flat array, a dict,
nnx.state(model, nnx.Param), ...). It follows an init/update protocol:
update(params, state, batch) returns the new params pytree (same structure), the
next state, and an LMInfo. For overparameterized systems (p parameters >> n
residual rows) it factors the small n x n gram (dual) system instead of the p x p
normal equations. The solver depends only on jax/jax.numpy/jax.flatten_util.
"""

from nlls_gram.gram_lm import (
    GramLevenbergMarquardt,
    LMInfo,
    LMState,
)

__all__ = ["GramLevenbergMarquardt", "LMState", "LMInfo"]
