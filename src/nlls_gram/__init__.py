"""Levenberg-Marquardt nonlinear least-squares for JAX, built for
underdetermined interpolation with an explicit selection of which
interpolant is returned.

Two solvers share one init/update/solve protocol (x is any JAX pytree; the
residual takes (x), (x, args), or (x, args, p); solve(...) runs a jitted
loop with callback control, save_steps histories, and multi-start retries or
parallel races; solve(...).x carries a custom implicit AD rule with respect
to p):

- RidgeLevenbergMarquardt minimizes the ridge objective
  ||r(x)||^2 + ridge * ||L x||^2 for a RidgePenalty factor L, with the
  ridge weight carried as traced state that a callback can anneal toward a
  positive floor (ridge_continuation) — the minimum-seminorm
  (min-RKHS-norm) selection lives in the OBJECTIVE, per classical nonlinear
  Tikhonov regularization. Linear solvers: dense cholesky (default, with a
  reject-step cache of the assembled normal matrix), a damping-row qr path
  for tiny ridge, and matrix-free lsmr. GN-implicit AD via cholesky or
  normal_cg.
- LevenbergMarquardt minimizes ||r(x)||^2 with an optional positive-definite
  parameter-space Metric (or iterate-aware MetricFactory) defining the
  damping geometry, so the small-damping Gauss-Newton limit selects
  minimum-metric-norm corrections. The default linear_solver="auto"
  resolves to the smaller dense factorization (gram_cholesky /
  normal_cholesky); QR, augmented QR, gram/normal CG, and LSMR variants
  cover direct and matrix-free regimes, with a swappable ad_solver menu
  (direct, svd, qr, augmented_qr, gram_cg, normal_cg,
  regularized_normal_cg).

The package depends only on JAX. Tuning heuristics (solver selection,
damping, inner-solve scheduling):
https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/
"""

from nlls_gram.gram_lm import (
    DrawNNXModule,
    LevenbergMarquardt,
    LMHyperparams,
    LMInfo,
    LMSolveAction,
    LMSolveContext,
    LMSolveResult,
    LMState,
    LMStatus,
    MetricFactory,
    MultiStart,
    MultiStartInfo,
    PreconditionerFactory,
    WhitenedPreconditioner,
)
from nlls_gram.lsmr import LSMRState, lsmr
from nlls_gram.metrics import (
    Metric,
    metric_from_cholesky,
    metric_from_diagonal,
    repeated_shifted_dense_metric,
    repeated_shifted_state_space_metric,
)
from nlls_gram.penalties import (
    RidgePenalty,
    identity_penalty,
    penalty_from_factor,
    repeated_dense_penalty,
)
from nlls_gram.preconditioners import (
    identity_preconditioner,
    identity_right_preconditioner,
    nystrom_preconditioner,
    pad_dual_preconditioner,
    sherman_morrison_preconditioner,
    woodbury_preconditioner,
)
from nlls_gram.quasiseparable import matern_state_space
from nlls_gram.recycled_cg import (
    HarvestState,
    RecycleConfig,
    RecycleState,
    build_coarse_operator,
    deflated_pcg,
    recycled_cg,
)
from nlls_gram.ridge_lm import (
    RidgeLevenbergMarquardt,
    RidgeLMInfo,
    RidgeLMState,
    ridge_continuation,
)

__all__ = [
    "LevenbergMarquardt",
    "LMState",
    "LMHyperparams",
    "LMInfo",
    "LMStatus",
    "LMSolveAction",
    "LMSolveContext",
    "LMSolveResult",
    "Metric",
    "MetricFactory",
    "MultiStart",
    "MultiStartInfo",
    "DrawNNXModule",
    "PreconditionerFactory",
    "WhitenedPreconditioner",
    "RecycleConfig",
    "RecycleState",
    "HarvestState",
    "build_coarse_operator",
    "deflated_pcg",
    "identity_penalty",
    "identity_preconditioner",
    "identity_right_preconditioner",
    "lsmr",
    "LSMRState",
    "penalty_from_factor",
    "repeated_dense_penalty",
    "ridge_continuation",
    "RidgeLevenbergMarquardt",
    "RidgeLMInfo",
    "RidgeLMState",
    "RidgePenalty",
    "matern_state_space",
    "metric_from_cholesky",
    "metric_from_diagonal",
    "nystrom_preconditioner",
    "pad_dual_preconditioner",
    "recycled_cg",
    "repeated_shifted_dense_metric",
    "repeated_shifted_state_space_metric",
    "sherman_morrison_preconditioner",
    "woodbury_preconditioner",
]
