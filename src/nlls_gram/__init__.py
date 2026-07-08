"""Metric-aware underdetermined Levenberg-Marquardt least-squares for JAX.

UnderdeterminedLevenbergMarquardt minimizes ||r(params, args, p)||^2 for a
user-supplied residual function taking (params), (params, args), or
(params, args, p), where params is any JAX pytree (a flat array, a
dict, nnx.state(model, nnx.Param), ...). It follows an init/update protocol:
update(params, state, args=None, p=None) returns the new params pytree (same
structure), the next state, and an LMInfo. solve(...) runs repeated LM steps with
optional callback control. With has_aux=True the residual returns
(residual, aux) and the aux output is reported on LMInfo. An optional Metric
defines a positive-definite parameter-space metric for LM damping. The solver
depends only on JAX plus Lineax for LSMR.
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
from nlls_gram.metrics import Metric, metric_from_cholesky

__all__ = [
    "UnderdeterminedLevenbergMarquardt",
    "LMState",
    "LMInfo",
    "LMStatus",
    "LMSolveAction",
    "LMSolveContext",
    "LMSolveResult",
    "Metric",
    "metric_from_cholesky",
]
