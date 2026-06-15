"""Gram/dual-form Levenberg-Marquardt nonlinear least-squares for JAX/Flax NNX models.

GramLevenbergMarquardt minimizes ||r(theta)||^2 for an nnx.Module-defined residual,
following the optax/nnx init/update protocol so steps apply through
nnx.Optimizer(model, optax.identity(), wrt=...). For overparameterized systems
(p parameters >> n residual rows) it factors the small n x n gram (dual) system.
"""

from nlls_gram.gram_lm import (
    GramLevenbergMarquardt,
    LMInfo,
    LMState,
    flat_residual,
)

__all__ = ["GramLevenbergMarquardt", "LMState", "LMInfo", "flat_residual"]
