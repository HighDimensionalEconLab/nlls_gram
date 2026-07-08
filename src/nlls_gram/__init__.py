"""Metric-aware underdetermined Levenberg-Marquardt least-squares for JAX.

UnderdeterminedLevenbergMarquardt minimizes ||r(params, aux, p)||^2 for a
user-supplied residual function, where params is any JAX pytree (a flat array, a
dict, nnx.state(model, nnx.Param), ...). It follows an init/update protocol:
update(params, state, aux=None, p=None) returns the new params pytree (same
structure), the next state, and an LMInfo. solve(...) runs repeated LM steps with
optional callback control. Optional metric callbacks define a positive-definite
parameter-space metric for LM damping. The solver depends only on JAX plus
Lineax for LSMR.
"""

from nlls_gram.gram_lm import (
    LMInfo,
    LMSolveAction,
    LMSolveContext,
    LMSolveResult,
    LMState,
    LMStatus,
    UnderdeterminedLevenbergMarquardt,
)
from nlls_gram.metrics import metric_callbacks_from_cholesky

__all__ = [
    "UnderdeterminedLevenbergMarquardt",
    "LMState",
    "LMInfo",
    "LMStatus",
    "LMSolveAction",
    "LMSolveContext",
    "LMSolveResult",
    "metric_callbacks_from_cholesky",
]
