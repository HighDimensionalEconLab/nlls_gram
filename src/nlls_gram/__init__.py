"""Metric-aware Levenberg-Marquardt nonlinear least-squares for JAX.

LevenbergMarquardt minimizes ||r(x, args, p)||^2 for a
user-supplied residual function taking (x), (x, args), or
(x, args, p), where x is any JAX pytree (a flat array, a
dict, nnx.state(model, nnx.Param), ...). It follows an init/update protocol:
update(x, lm_state, args=None, p=None) returns the new x pytree (same
structure), the next lm_state, and an LMInfo. solve(...) runs repeated LM steps with
optional callback control; solve(multi_start=MultiStart(...)) retries failed
solves from fresh initial conditions or races them in parallel under vmap,
returning the single best result. With has_aux=True the residual returns
(residual, aux) and the aux output is reported on LMInfo. An optional Metric
defines a positive-definite parameter-space metric for LM damping; a
MetricFactory instead rebuilds that metric from the current iterate and
residual aux every accepted step.
The default linear_solver="auto" resolves at trace time to the smaller dense
factorization: residual-space Gram Cholesky (gram_cholesky) when n > m, else
whitened normal Cholesky (normal_cholesky) — the two compute the same step.
Reduced QR (full row rank) and direct augmented QR cover small direct
solves. Three matrix-free solvers use only J/J' products: CG on the
metric-damped residual-space dual (gram_cg, required dual preconditioner),
CG on the whitened normal system in parameter space (normal_cg, required
normal preconditioner), and LSMR on the whitened subproblem (optional
WhitenedPreconditioner right-preconditioner), the last staying accurate at
small damping where the squared Gram/normal solves degrade. The implicit
differentiation rule is independently swappable via explicit ad_solver methods:
direct, svd, qr, augmented_qr, gram_cg, normal_cg, and
regularized_normal_cg. The solver depends only on JAX.

Tuning heuristics (solver selection, damping, inner-solve scheduling):
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
from nlls_gram.preconditioners import (
    identity_preconditioner,
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
    "identity_preconditioner",
    "lsmr",
    "LSMRState",
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
