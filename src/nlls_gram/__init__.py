"""Underdetermined Levenberg-Marquardt nonlinear least-squares for JAX.

UnderdeterminedLevenbergMarquardt minimizes ||r(params)||^2 for a user-supplied
residual_fn(params, batch), where params is any JAX pytree (a flat array, a dict,
nnx.state(model, nnx.Param), ...). It follows an init/update protocol:
update(params, state, batch) returns the new params pytree (same structure), the
next state, and an LMInfo. For overparameterized systems (p parameters >> n
residual rows) the default solver factors the small residual-space gram (dual)
system instead of the p x p normal equations. The solver depends only on JAX plus
Lineax for LSMR.
"""

from nlls_gram.gram_lm import (
    LMInfo,
    LMState,
    UnderdeterminedLevenbergMarquardt,
)

__all__ = ["UnderdeterminedLevenbergMarquardt", "LMState", "LMInfo"]
