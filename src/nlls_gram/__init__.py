"""Metric-aware underdetermined Levenberg-Marquardt least-squares for JAX.

UnderdeterminedLevenbergMarquardt minimizes ||r(x, args, p)||^2 for a
user-supplied residual function taking (x), (x, args), or
(x, args, p), where x is any JAX pytree (a flat array, a
dict, nnx.state(model, nnx.Param), ...). It follows an init/update protocol:
update(x, lm_state, args=None, p=None) returns the new x pytree (same
structure), the next lm_state, and an LMInfo. solve(...) runs repeated LM steps with
optional callback control; solve(multi_start=MultiStart(...)) retries failed
solves from fresh initial conditions or races them in parallel under vmap,
returning the single best result. With has_aux=True the residual returns
(residual, aux) and the aux output is reported on LMInfo. An optional Metric
defines a positive-definite parameter-space metric for LM damping.
SquareLevenbergMarquardt is a solve-only damped-Newton companion for square
nonsingular systems (DAE stage roots) with a direct dense step and implicit
differentiation. The solvers depend only on JAX.

Tuning heuristics (solver selection, damping, inner-solve scheduling):
https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/
"""

from nlls_gram.gram_lm import (
    DrawNNXModule,
    LMHyperparams,
    LMInfo,
    LMSolveAction,
    LMSolveContext,
    LMSolveResult,
    LMState,
    LMStatus,
    MultiStart,
    MultiStartInfo,
    UnderdeterminedLevenbergMarquardt,
)
from nlls_gram.metrics import (
    Metric,
    blockdiag_metric,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_from_quasiseparable,
    metric_from_shifted_matvec,
    metric_from_state_space,
    metric_from_tridiagonal_precision,
    metric_with_compute_dtype,
    repeated_blockdiag_metric,
)
from nlls_gram.preconditioners import (
    identity_preconditioner,
    nystrom_preconditioner,
    pad_dual_preconditioner,
    sherman_morrison_preconditioner,
    woodbury_preconditioner,
)
from nlls_gram.quasiseparable import matern_state_space
from nlls_gram.recycled_cg import recycled_cg
from nlls_gram.square_lm import SquareLevenbergMarquardt, SquareSolveResult

__all__ = [
    "UnderdeterminedLevenbergMarquardt",
    "SquareLevenbergMarquardt",
    "SquareSolveResult",
    "LMState",
    "LMHyperparams",
    "LMInfo",
    "LMStatus",
    "LMSolveAction",
    "LMSolveContext",
    "LMSolveResult",
    "Metric",
    "MultiStart",
    "MultiStartInfo",
    "DrawNNXModule",
    "blockdiag_metric",
    "repeated_blockdiag_metric",
    "identity_preconditioner",
    "matern_state_space",
    "metric_from_cholesky",
    "metric_from_diagonal",
    "metric_from_quasiseparable",
    "metric_from_shifted_matvec",
    "metric_from_state_space",
    "metric_from_tridiagonal_precision",
    "metric_with_compute_dtype",
    "nystrom_preconditioner",
    "pad_dual_preconditioner",
    "recycled_cg",
    "sherman_morrison_preconditioner",
    "woodbury_preconditioner",
]
