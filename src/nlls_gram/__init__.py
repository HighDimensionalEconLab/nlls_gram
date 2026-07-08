"""Metric-aware underdetermined Levenberg-Marquardt least-squares for JAX.

UnderdeterminedLevenbergMarquardt minimizes ||r(x, args, p)||^2 for a
user-supplied residual function taking (x), (x, args), or
(x, args, p), where x is any JAX pytree (a flat array, a
dict, nnx.state(model, nnx.Param), ...). It follows an init/update protocol:
update(x, lm_state, args=None, p=None) returns the new x pytree (same
structure), the next lm_state, and an LMInfo. solve(...) runs repeated LM steps with
optional callback control. With has_aux=True the residual returns
(residual, aux) and the aux output is reported on LMInfo. An optional Metric
defines a positive-definite parameter-space metric for LM damping. The solver
depends only on JAX plus Lineax for LSMR.

Tuning heuristics (solver selection, damping, inner-solve scheduling):
https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/
"""

from nlls_gram.gram_lm import (
    LMHyperparams,
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
    "LMHyperparams",
    "LMInfo",
    "LMStatus",
    "LMSolveAction",
    "LMSolveContext",
    "LMSolveResult",
    "Metric",
    "metric_from_cholesky",
]
