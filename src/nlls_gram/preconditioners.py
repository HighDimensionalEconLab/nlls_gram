from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from nlls_gram.utilities import register_pytree_dataclass

__all__ = [
    "DiagonalPreconditioner",
    "IdentityPreconditioner",
    "Preconditioner",
]


class Preconditioner:
    """Numerical preconditioner protocol for Krylov and dense linear solvers.

    ``apply(v, damping, ctx)`` applies an SPD approximation of the damped
    operator inverse in original Euclidean coordinates (parameter space for
    ``CG``, residual space for ``GramCG``). Subclasses must register as JAX
    pytrees with ``register_pytree_dataclass``. Set ``requires_positive_damping = True``
    when ``apply`` divides by the live damping and cannot run at zero damping.
    """

    requires_positive_damping = False

    def apply(self, v, damping, ctx):
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityPreconditioner(Preconditioner):
    """Stateless identity preconditioner."""

    def apply(self, v, damping, ctx):
        return v


register_pytree_dataclass(IdentityPreconditioner, data_fields=())


@dataclass(frozen=True, eq=False)
class DiagonalPreconditioner(Preconditioner):
    """Diagonal Jacobi or equilibration preconditioner for Krylov and dense solvers.

    When ``diagonal`` is a positive 1-D array ``d``, Krylov solvers apply
    ``v / (d + damping)`` and dense solvers scale by ``1 / sqrt(d + damping)``.
    When ``diagonal is None``, dense ``Cholesky`` and ``QR`` solvers compute
    automatic diagonal equilibration from the assembled matrix diagonal or
    column/row norms without altering the target linear equation.
    """

    diagonal: jax.Array | None = None

    def __post_init__(self):
        if self.diagonal is None:
            return
        diag = jnp.asarray(self.diagonal)
        if diag.ndim != 1:
            raise ValueError("DiagonalPreconditioner diagonal must be a 1-D array")
        diag = diag.astype(jnp.result_type(diag, 1.0))
        if not isinstance(diag, jax.core.Tracer):
            d_np = np.asarray(diag)
            if not np.all(np.isfinite(d_np)) or np.any(d_np <= 0.0):
                raise ValueError(
                    "DiagonalPreconditioner diagonal entries must be finite "
                    "and strictly positive"
                )
        else:
            valid = jnp.all(jnp.isfinite(diag)) & jnp.all(diag > 0.0)
            diag = jnp.where(valid, diag, jnp.full_like(diag, jnp.nan))
        object.__setattr__(self, "diagonal", diag)

    def apply(self, v, damping, ctx):
        if self.diagonal is None:
            raise ValueError(
                "DiagonalPreconditioner(diagonal=None) automatic equilibration "
                "requires a dense solver (Cholesky or QR). Pass an explicit "
                "positive diagonal array for matrix-free CG or GramCG"
            )
        d = jnp.asarray(self.diagonal, dtype=v.dtype)
        shift = d + jnp.asarray(damping, dtype=v.dtype)
        floor = jnp.asarray(jnp.finfo(v.dtype).tiny, dtype=v.dtype)
        denom = jnp.maximum(shift, floor)
        shape = (denom.shape[0],) + (1,) * (v.ndim - 1)
        return v / denom.reshape(shape)

    def cholesky_scale(self, diag_H, damping, dtype):
        floor = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype))
        diag_H = jax.lax.stop_gradient(jnp.asarray(diag_H, dtype=dtype))
        mu = jax.lax.stop_gradient(jnp.asarray(damping, dtype=dtype))
        if self.diagonal is None:
            raw = jnp.sqrt(jnp.maximum(diag_H, 0.0))
            return jnp.where(raw > floor, raw, jnp.ones_like(raw))
        d = jax.lax.stop_gradient(jnp.asarray(self.diagonal, dtype=dtype))
        if d.shape != diag_H.shape:
            raise ValueError(
                f"DiagonalPreconditioner diagonal shape {d.shape} does not match "
                f"Cholesky system dimension {diag_H.shape}"
            )
        raw = jnp.sqrt(jnp.maximum(d + mu, 0.0))
        return jnp.maximum(raw, floor)

    def qr_column_scale(self, col_sq_norms, damping, dtype):
        floor = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype))
        col_sq = jax.lax.stop_gradient(jnp.asarray(col_sq_norms, dtype=dtype))
        mu = jax.lax.stop_gradient(jnp.asarray(damping, dtype=dtype))
        if self.diagonal is None:
            norms = jnp.sqrt(jnp.maximum(col_sq + mu, 0.0))
            return jnp.where(norms > floor, 1.0 / norms, jnp.ones_like(norms))
        d = jax.lax.stop_gradient(jnp.asarray(self.diagonal, dtype=dtype))
        if d.shape != col_sq.shape:
            raise ValueError(
                f"DiagonalPreconditioner diagonal shape {d.shape} does not match "
                f"QR column count {col_sq.shape}"
            )
        denom = jnp.maximum(jnp.sqrt(jnp.maximum(d + mu, 0.0)), floor)
        return 1.0 / denom


register_pytree_dataclass(DiagonalPreconditioner, data_fields=("diagonal",))
