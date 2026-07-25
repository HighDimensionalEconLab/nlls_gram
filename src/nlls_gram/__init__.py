"""Levenberg-Marquardt nonlinear least-squares for JAX, built for
underdetermined interpolation with an explicit selection of which interpolant
is returned.

Two solvers share one ``init``/``update``/``solve`` protocol -- ``x`` is any
JAX pytree, the residual takes ``(x)``, ``(x, args)``, or ``(x, args, p)``,
``solve`` runs a jitted loop with callback control, ``save_steps`` histories,
and multi-start retries or parallel races, and ``solve(...).x`` carries a
custom implicit AD rule with respect to ``p``. They differ in where the
selection of the returned root lives:

- ``RidgeLevenbergMarquardt`` puts it in the OBJECTIVE, minimizing
  ``||r(x)||^2 + ridge * ||x_m||_W^2`` for a positive-definite ``Metric`` W on
  the metric block, with the ridge weight carried as traced state a callback
  can anneal (``ridge_continuation``). Classical nonlinear Tikhonov
  regularization; the minimum-seminorm interpolant is what it converges to.
- ``LevenbergMarquardt`` puts it in the DAMPING GEOMETRY, minimizing
  ``||r(x)||^2`` with the same ``Metric`` weighting the trust region, so the
  small-damping Gauss-Newton limit selects minimum-metric-norm corrections.

Both take the same typed linear-solver configs -- ``Cholesky()``, ``QR()``,
``CG(preconditioner)``, ``GramCG(preconditioner)`` -- and the same
``ad_solver`` menu, defaulting to the forward family.

The package depends only on JAX. Tuning heuristics:
https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/
"""

from nlls_gram.linear_solvers import (
    CG,
    QR,
    SVD,
    Cholesky,
    CholeskyCache,
    GramCG,
    QRCache,
)
from nlls_gram.lm_types import (
    LMHyperparams,
    LMInfo,
    LMSolveAction,
    LMSolveContext,
    LMSolveResult,
    LMState,
    LMStatus,
    SolverContext,
)
from nlls_gram.metric_lm import LevenbergMarquardt
from nlls_gram.metrics import (
    CholeskyMetric,
    DiagonalMetric,
    IdentityMetric,
    Metric,
    RepeatedFactorMetric,
)
from nlls_gram.multi_start import DrawNNXModule, MultiStart, MultiStartInfo
from nlls_gram.preconditioners import (
    BlockEigenPreconditioner,
    IdentityPreconditioner,
    NystromPreconditioner,
    PaddedPreconditioner,
    Preconditioner,
    ShermanMorrisonPreconditioner,
    WoodburyPreconditioner,
    block_eigen_state,
)
from nlls_gram.ridge_lm import RidgeLevenbergMarquardt, ridge_continuation

__all__ = [
    "CG",
    "QR",
    "SVD",
    "BlockEigenPreconditioner",
    "Cholesky",
    "CholeskyCache",
    "CholeskyMetric",
    "DiagonalMetric",
    "DrawNNXModule",
    "GramCG",
    "IdentityMetric",
    "IdentityPreconditioner",
    "LMHyperparams",
    "LMInfo",
    "LMSolveAction",
    "LMSolveContext",
    "LMSolveResult",
    "LMState",
    "LMStatus",
    "LevenbergMarquardt",
    "Metric",
    "MultiStart",
    "MultiStartInfo",
    "NystromPreconditioner",
    "PaddedPreconditioner",
    "Preconditioner",
    "QRCache",
    "RepeatedFactorMetric",
    "RidgeLevenbergMarquardt",
    "ShermanMorrisonPreconditioner",
    "SolverContext",
    "WoodburyPreconditioner",
    "block_eigen_state",
    "ridge_continuation",
]
