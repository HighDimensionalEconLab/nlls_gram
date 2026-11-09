"""Levenberg-Marquardt and Newton/Gauss-Newton nonlinear least-squares for JAX,
built for underdetermined interpolation with an explicit selection of which
interpolant is returned.

Two solvers share one ``init``/``update``/``solve`` protocol -- ``x`` is any
JAX pytree, the residual takes ``(x)``, ``(x, args)``, or ``(x, args, p)``,
``solve`` runs a jitted loop with callback control, ``save_steps`` histories,
and multi-start retries or parallel races, and ``solve(...).x`` carries a
custom implicit AD rule with respect to ``p``. They differ in globalization
(:class:`LevenbergMarquardt` trust-region damping vs :class:`Newton` Armijo
backtracking) and support two root-selection regimes controlled by ``ridge``:

- **Unpenalized (``ridge=None``, default)** puts root selection in the STEP
  GEOMETRY, minimizing ``||r(x)||^2`` with the optional :class:`Metric`
  selecting minimum-metric-norm Gauss-Newton corrections.
- **Penalized ridge (``ridge > 0``)** puts root selection in the OBJECTIVE,
  minimizing ``||r(x)||^2 + ridge * ||x_m||_W^2`` for a positive-definite
  :class:`Metric` ``W`` on the metric block (defaulting to the Euclidean
  identity over all coordinates when ``metric=None``), with the ridge weight
  carried in traced state that a callback can anneal (:class:`AnnealRidge`).

Both solvers take the same typed linear-solver configs -- ``Cholesky()``,
``QR()``, ``CG(preconditioner)``, ``GramCG(preconditioner)`` (plus ``LU()``
and ``SVD()`` for ``Newton`` and ``ad_solver``; ``QR()`` is forward-only) --
and ``ad_solver=None`` resolves to ``LU()`` (square), ``SVD()`` (rectangular
unpenalized), ``Cholesky()`` (penalized), or the forward Krylov solver when
shape-compatible. ``vmap_rows`` is ``jax.vmap`` for a
per-row residual block, with a derivative rule that computes each row's
gradient once per linearization and reuses it across the Jacobian assembly.

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
from nlls_gram.lm import AnnealRidge, LevenbergMarquardt
from nlls_gram.metrics import (
    CholeskyMetric,
    DiagonalMetric,
    IdentityMetric,
    Metric,
    RepeatedFactorMetric,
)
from nlls_gram.multi_start import DrawNNXModule, MultiStart, MultiStartInfo
from nlls_gram.newton import Newton
from nlls_gram.preconditioners import (
    BlockEigenPreconditioner,
    IdentityPreconditioner,
    NystromPreconditioner,
    PaddedPreconditioner,
    Preconditioner,
    ShermanMorrisonPreconditioner,
    WoodburyPreconditioner,
)
from nlls_gram.types import (
    PDAS,
    CallbackContext,
    Complementarity,
    Equality,
    FischerBurmeister,
    Inequality,
    LMHyperparams,
    LMInfo,
    LMState,
    Minimum,
    NewtonHyperparams,
    NewtonInfo,
    NewtonState,
    Residuals,
    SolverAction,
    SolverContext,
    SolveResult,
    SolverStatus,
)
from nlls_gram.utilities import register_pytree_dataclass
from nlls_gram.vmap_rows import vmap_rows

__all__ = [
    "CG",
    "LU",
    "PDAS",
    "QR",
    "SVD",
    "AnnealRidge",
    "BlockEigenPreconditioner",
    "CallbackContext",
    "Cholesky",
    "CholeskyCache",
    "CholeskyMetric",
    "Complementarity",
    "DiagonalMetric",
    "DrawNNXModule",
    "Equality",
    "FischerBurmeister",
    "GramCG",
    "IdentityMetric",
    "IdentityPreconditioner",
    "Inequality",
    "LMHyperparams",
    "LMInfo",
    "LMState",
    "LevenbergMarquardt",
    "Metric",
    "Minimum",
    "MultiStart",
    "MultiStartInfo",
    "Newton",
    "NewtonHyperparams",
    "NewtonInfo",
    "NewtonState",
    "NystromPreconditioner",
    "PaddedPreconditioner",
    "Preconditioner",
    "QRCache",
    "RepeatedFactorMetric",
    "Residuals",
    "ShermanMorrisonPreconditioner",
    "SolveResult",
    "SolverAction",
    "SolverContext",
    "SolverStatus",
    "WoodburyPreconditioner",
    "register_pytree_dataclass",
    "vmap_rows",
]
