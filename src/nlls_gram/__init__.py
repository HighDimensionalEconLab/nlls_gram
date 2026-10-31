"""Levenberg-Marquardt and Newton/Gauss-Newton nonlinear least-squares for JAX,
built for underdetermined interpolation with an explicit selection of which
interpolant is returned.

Four solvers share one ``init``/``update``/``solve`` protocol -- ``x`` is any
JAX pytree, the residual takes ``(x)``, ``(x, args)``, or ``(x, args, p)``,
``solve`` runs a jitted loop with callback control, ``save_steps`` histories,
and multi-start retries or parallel races, and ``solve(...).x`` carries a
custom implicit AD rule with respect to ``p``. They differ in globalization
(trust-region damping vs Armijo backtracking) and in where the selection of the
returned root lives:

- ``RidgeLevenbergMarquardt`` and ``RidgeNewton`` put it in the OBJECTIVE,
  minimizing ``||r(x)||^2 + ridge * ||x_m||_W^2`` for a positive-definite
  ``Metric`` W on the metric block, with the ridge weight carried as traced
  state a callback can anneal (``AnnealRidge``). Classical nonlinear Tikhonov
  regularization; the minimum-seminorm interpolant is what it converges to.
- ``LevenbergMarquardt`` and ``Newton`` put it in the STEP GEOMETRY, minimizing
  ``||r(x)||^2`` with the optional ``Metric`` selecting minimum-metric-norm
  Gauss-Newton corrections.

All four take the same typed linear-solver configs -- ``Cholesky()``, ``QR()``,
``CG(preconditioner)``, ``GramCG(preconditioner)`` (plus ``LU()`` and ``SVD()``
for ``Newton`` and ``ad_solver``) -- and the same ``ad_solver`` menu,
defaulting to the forward family. ``vmap_rows`` is ``jax.vmap`` for a per-row
residual block, with a derivative rule that computes each row's gradient once
per linearization and reuses it across the Jacobian assembly.

The package depends only on JAX. Tuning heuristics:
https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/
"""

from nlls_gram.linear_solvers import (
    CG,
    LU,
    QR,
    SVD,
    Cholesky,
    CholeskyCache,
    GramCG,
    QRCache,
)
from nlls_gram.lm_types import (
    LMAction,
    LMContext,
    LMHyperparams,
    LMInfo,
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
from nlls_gram.newton import Newton, RidgeNewton
from nlls_gram.newton_types import (
    NewtonAction,
    NewtonContext,
    NewtonHyperparams,
    NewtonInfo,
    NewtonSolveResult,
    NewtonState,
    NewtonStatus,
)
from nlls_gram.preconditioners import (
    BlockEigenPreconditioner,
    IdentityPreconditioner,
    NystromPreconditioner,
    PaddedPreconditioner,
    Preconditioner,
    ShermanMorrisonPreconditioner,
    WoodburyPreconditioner,
)
from nlls_gram.ridge_lm import AnnealRidge, RidgeLevenbergMarquardt
from nlls_gram.utilities import register_pytree_dataclass
from nlls_gram.vmap_rows import vmap_rows

__all__ = [
    "CG",
    "LU",
    "QR",
    "SVD",
    "AnnealRidge",
    "BlockEigenPreconditioner",
    "Cholesky",
    "CholeskyCache",
    "CholeskyMetric",
    "DiagonalMetric",
    "DrawNNXModule",
    "GramCG",
    "IdentityMetric",
    "IdentityPreconditioner",
    "LMAction",
    "LMContext",
    "LMHyperparams",
    "LMInfo",
    "LMSolveResult",
    "LMState",
    "LMStatus",
    "LevenbergMarquardt",
    "Metric",
    "MultiStart",
    "MultiStartInfo",
    "Newton",
    "NewtonAction",
    "NewtonContext",
    "NewtonHyperparams",
    "NewtonInfo",
    "NewtonSolveResult",
    "NewtonState",
    "NewtonStatus",
    "NystromPreconditioner",
    "PaddedPreconditioner",
    "Preconditioner",
    "QRCache",
    "RepeatedFactorMetric",
    "RidgeLevenbergMarquardt",
    "RidgeNewton",
    "ShermanMorrisonPreconditioner",
    "SolverContext",
    "WoodburyPreconditioner",
    "register_pytree_dataclass",
    "vmap_rows",
]
