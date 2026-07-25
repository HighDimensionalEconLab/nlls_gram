"""Levenberg-Marquardt nonlinear least-squares for JAX, built for
underdetermined interpolation with an explicit selection of which
interpolant is returned.

Two solvers share one init/update/solve protocol (x is any JAX pytree; the
residual takes (x), (x, args), or (x, args, p); solve(...) runs a jitted
loop with callback control, save_steps histories, and multi-start retries or
parallel races; solve(...).x carries a custom implicit AD rule with respect
to p):

- RidgeLevenbergMarquardt minimizes the ridge objective
  ||r(x)||^2 + ridge * ||x_m||_W^2 for a positive-definite Metric W on the
  metric block x_m of x = [x_m; x_f] (the free block x_f stays
  unpenalized), with the ridge weight carried as traced state that a
  callback can anneal toward a positive floor (ridge_continuation) — the
  minimum-seminorm (min-RKHS-norm) selection lives in the OBJECTIVE, per
  classical nonlinear Tikhonov regularization. The metric is supplied
  through factor callbacks for W = F'F (IdentityMetric is plain ridge,
  RepeatedFactorMetric the kernel workhorse) and the solver runs entirely
  in the whitened variable y = F_bar x with constant penalty rows [I 0] —
  a clean spectral floor at the ridge, so the default Cholesky() path
  stays accurate at deep ridge. Stopping is conjunctive gtol + atol with
  the whitened geometry (steps in the W-norm, gradients in the dual
  W^{-1}-norm); calibrate gtol ~ 1e-3 * ridge * sqrt(q(x*)) since
  info.penalty_grad_norm = sqrt(penalty_value). Linear solvers are typed
  configs — Cholesky() (the default: dense normal equations with a
  reject-step cache), QR() (damping-row QR for tiny ridge),
  CG(preconditioner, ...) (matrix-free preconditioned CG on the damped
  whitened normal operator) — and the AD side takes Cholesky() or CG(...),
  defaulting to the forward family.
- LevenbergMarquardt minimizes ||r(x)||^2 with an optional positive-definite
  parameter-space GramMetric (or iterate-aware MetricFactory) defining the
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
    LevenbergMarquardt,
    MetricFactory,
    PreconditionerFactory,
    WhitenedPreconditioner,
)
from nlls_gram.linear_solvers import CG, QR, Cholesky, CholeskyCache, QRCache
from nlls_gram.lm_types import (
    LMHyperparams,
    LMInfo,
    LMSolveAction,
    LMSolveContext,
    LMSolveResult,
    LMState,
    LMStatus,
)
from nlls_gram.lsmr import LSMRState, lsmr
from nlls_gram.metrics import (
    GramMetric,
    IdentityMetric,
    Metric,
    MetricContext,
    RepeatedFactorMetric,
    metric_from_cholesky,
    metric_from_diagonal,
    repeated_shifted_dense_metric,
    repeated_shifted_state_space_metric,
)
from nlls_gram.multi_start import DrawNNXModule, MultiStart, MultiStartInfo
from nlls_gram.preconditioners import (
    BlockEigenPreconditioner,
    IdentityPreconditioner,
    Preconditioner,
    block_eigen_state,
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
from nlls_gram.ridge_lm import RidgeLevenbergMarquardt, ridge_continuation

__all__ = [
    "BlockEigenPreconditioner",
    "Cholesky",
    "CholeskyCache",
    "LevenbergMarquardt",
    "LMState",
    "CG",
    "QR",
    "QRCache",
    "LMHyperparams",
    "LMInfo",
    "LMStatus",
    "LMSolveAction",
    "LMSolveContext",
    "LMSolveResult",
    "GramMetric",
    "IdentityMetric",
    "Metric",
    "MetricContext",
    "MetricFactory",
    "MultiStart",
    "IdentityPreconditioner",
    "Preconditioner",
    "RepeatedFactorMetric",
    "MultiStartInfo",
    "DrawNNXModule",
    "PreconditionerFactory",
    "WhitenedPreconditioner",
    "RecycleConfig",
    "RecycleState",
    "HarvestState",
    "block_eigen_state",
    "build_coarse_operator",
    "deflated_pcg",
    "identity_preconditioner",
    "identity_right_preconditioner",
    "lsmr",
    "LSMRState",
    "ridge_continuation",
    "RidgeLevenbergMarquardt",
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
