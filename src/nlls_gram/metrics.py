"""The positive-definite metric both solvers take, given through factor
callbacks.

One contract serves two roles. For
:class:`~nlls_gram.RidgeLevenbergMarquardt` the metric ``W`` weights the
OBJECTIVE's penalty, ``ridge * ||x_m||_W^2``; for
:class:`~nlls_gram.LevenbergMarquardt` it is the damping geometry, so the
small-damping Gauss-Newton limit selects minimum-``W``-norm corrections.
Either way the solver runs in the whitened variable and never materializes
``W`` or its factor.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from nlls_gram.lm_types import SolverContext
from nlls_gram.utilities import register_pytree_dataclass

__all__ = [
    "CholeskyMetric",
    "DiagonalMetric",
    "IdentityMetric",
    "Metric",
    "RepeatedFactorMetric",
    "SolverContext",
]


class Metric:
    """Positive-definite metric ``W``, via factor callbacks.

    ``x = [x_m; x_f]`` splits into the metric block ``x_m`` -- the leading
    ``size`` coordinates, covered by ``W`` -- and a free block ``x_f``. The
    metric is supplied through callbacks for an invertible factor ``F`` with
    ``W = F'F``, the whitened variable being ``F x_m`` and
    ``||v||_W = ||F v||_2``. ``F`` should be upper triangular; the canonical
    example is ``F = jnp.linalg.cholesky(K, upper=True)``. The solver extends
    it to ``F_bar = blockdiag(F, sqrt(free_scale) I)`` over the whole vector
    and never materializes either.

    Subclasses implement the ops on metric-block vectors (or matrices whose
    LEADING axis is ``size``; columns are batched):

    - ``factor_apply(v, ctx)``: ``F v``
    - ``factor_solve(v, ctx)``: ``F^{-1} v``
    - ``factor_solve_transpose(v, ctx)``: ``F^{-T} v``
    - ``norm(v, ctx)``: ``||v||_W``, defaulted to ``||factor_apply(v)||_2``.
      Provided for callers; the solvers measure in the whitened variable and
      never call it.

    Every op receives a :class:`~nlls_gram.SolverContext` carrying the
    solver's live state, so an exotic metric can key off the iterate through
    ``ctx.x`` and ``ctx.lm_state``.

    Metric instances are JAX PYTREES: array fields are traced leaves, the
    type plus its static fields are structure. Every concrete class must be
    registered with
    :func:`~nlls_gram.register_pytree_dataclass` -- the solvers reject
    unregistered instances. The instance rides inside the solver state
    (``lm_state.metric``), so a ``solve`` callback replaces the metric by
    constructing a new instance of the same type (same static fields, same
    leaf shapes and dtypes) -- pure traced ops, no recompilation. Equal-config
    instances with fresh arrays share one compiled solve loop.
    ``dataclasses.replace`` works too: metric constructors only validate and
    derive shapes, so re-running them under trace is cheap.

    ``free_scale`` weights the free block in the whitened variable: ``1.0``
    (the default) leaves it Euclidean. The ridge solver never penalizes the
    free block whatever the scale -- ``free_scale`` only changes its
    trust-region geometry -- while for the metric solver it IS that block's
    damping weight. It is a traced leaf, canonicalized by each constructor to
    the factor's float dtype, so changing it never recompiles.

    Contracts: the factor must be EXACT. The solver hardcodes the identity
    penalty block in the whitened variable, so an approximate factor silently
    changes the objective (unlike a CG preconditioner, which may be sloppy).
    The ridge weight never enters the factorization, so ridge continuation
    composes unchanged. How a subclass fulfills the ops -- prefactorized
    storage, factorize-in-``__init__``, fully matrix-free -- is its
    constructor's business, and the constructor must be traceable when the
    metric is rebuilt inside a jitted callback.
    """

    size: int
    free_scale: float = 1.0

    def factor_apply(self, v, ctx):
        """``F v`` for a metric-block vector or leading-axis-batched matrix."""
        raise NotImplementedError

    def factor_solve(self, v, ctx):
        """``F^{-1} v`` for a metric-block vector or batched matrix."""
        raise NotImplementedError

    def factor_solve_transpose(self, v, ctx):
        """``F^{-T} v`` for a metric-block vector or batched matrix."""
        raise NotImplementedError

    def norm(self, v, ctx):
        """``||v||_W = ||F v||_2`` for a metric-block vector."""
        return jnp.linalg.norm(self.factor_apply(v, ctx))


def _canonical_free_scale(free_scale, dtype):
    # F_bar = blockdiag(F, sqrt(free_scale) I), so a non-positive scale makes
    # the whitening noninvertible or complex. Traced values skip the sign
    # check; the strong-typed cast keeps rebuilt instances aval-identical to
    # the originals inside lax.cond/while_loop.
    if not isinstance(free_scale, (jax.Array, jax.core.Tracer)) and free_scale <= 0:
        raise ValueError("free_scale must be positive")
    return jnp.asarray(free_scale, dtype=dtype)


def _check_leading_size(v, size):
    if v.shape[0] != size:
        raise ValueError(
            f"metric factor input leading size must be {size}, got {v.shape[0]}"
        )


@dataclass(frozen=True, eq=False)
class IdentityMetric(Metric):
    """The identity metric ``W = I`` on ``size`` coordinates -- plain ridge for
    the ridge solver, Euclidean damping for the metric solver.

    ``F = I``: every factor op is the identity and ``norm`` is the Euclidean
    norm, with no special-casing anywhere downstream.
    """

    size: int
    free_scale: float = 1.0

    def __post_init__(self):
        if self.size < 0:
            raise ValueError("size must be nonnegative")
        object.__setattr__(
            self,
            "free_scale",
            _canonical_free_scale(self.free_scale, jnp.result_type(float)),
        )

    def factor_apply(self, v, ctx):
        _check_leading_size(v, self.size)
        return v

    def factor_solve(self, v, ctx):
        _check_leading_size(v, self.size)
        return v

    def factor_solve_transpose(self, v, ctx):
        _check_leading_size(v, self.size)
        return v

    def norm(self, v, ctx):
        _check_leading_size(v, self.size)
        return jnp.linalg.norm(v)


register_pytree_dataclass(
    IdentityMetric, data_fields=("free_scale",), meta_fields=("size",)
)


@dataclass(frozen=True, eq=False)
class CholeskyMetric(Metric):
    """Dense metric ``W = L L'`` from its lower-triangular Cholesky factor.

    ``L`` is the factor as ``jnp.linalg.cholesky`` returns it, so the upper
    factor this class works with is ``F = L'``. Triangularity and positive
    definiteness are assumed, not validated -- the entries may be traced, and
    a singular factor propagates NaN loudly through the triangular solves.
    """

    L: jax.Array
    free_scale: float = 1.0
    size: int = field(init=False)

    def __post_init__(self):
        L = jnp.asarray(self.L)
        if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] == 0:
            raise ValueError("L must be a nonempty square matrix")
        object.__setattr__(self, "L", L)
        object.__setattr__(self, "size", L.shape[0])
        object.__setattr__(
            self,
            "free_scale",
            _canonical_free_scale(self.free_scale, jnp.result_type(L, 1.0)),
        )

    def factor_apply(self, v, ctx):
        _check_leading_size(v, self.size)
        return self.L.T @ v

    def factor_solve(self, v, ctx):
        _check_leading_size(v, self.size)
        return jsp_linalg.solve_triangular(self.L.T, v, lower=False)

    def factor_solve_transpose(self, v, ctx):
        _check_leading_size(v, self.size)
        return jsp_linalg.solve_triangular(self.L, v, lower=True)


register_pytree_dataclass(
    CholeskyMetric, data_fields=("L", "free_scale"), meta_fields=("size",)
)


@dataclass(frozen=True, eq=False)
class DiagonalMetric(Metric):
    """The diagonal metric ``W = diag(weights)``; ``F = diag(sqrt(weights))``.

    ``weights`` must be a positive 1-D array. Positivity is not validated
    because the values may be traced. Every op is elementwise.
    """

    weights: jax.Array
    free_scale: float = 1.0
    size: int = field(init=False)

    def __post_init__(self):
        weights = jnp.asarray(self.weights)
        if weights.ndim != 1:
            raise ValueError("weights must be 1-D")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "size", weights.shape[0])
        object.__setattr__(
            self,
            "free_scale",
            _canonical_free_scale(self.free_scale, jnp.result_type(weights, 1.0)),
        )

    def _scaled(self, v, factor):
        _check_leading_size(v, self.size)
        return v * factor.reshape(factor.shape + (1,) * (v.ndim - 1))

    def factor_apply(self, v, ctx):
        return self._scaled(v, jnp.sqrt(self.weights))

    def factor_solve(self, v, ctx):
        return self._scaled(v, 1.0 / jnp.sqrt(self.weights))

    def factor_solve_transpose(self, v, ctx):
        return self.factor_solve(v, ctx)


register_pytree_dataclass(
    DiagonalMetric, data_fields=("weights", "free_scale"), meta_fields=("size",)
)


@dataclass(frozen=True, eq=False)
class RepeatedFactorMetric(Metric):
    """``repeats`` copies of one block factor: ``W = blockdiag(F'F, ...)``.

    ``F`` is an upper-triangular invertible block factor -- e.g.
    ``jnp.linalg.cholesky(K, upper=True)`` for a positive-definite Gram matrix
    ``K``, giving the repeated kernel seminorm ``sum_j alpha_j' K alpha_j``
    over ``repeats`` coefficient blocks. The constructor takes the FACTOR, not
    ``K`` (callers typically already hold it); a positive-SEMIdefinite ``K``
    needs a shift first: ``jnp.linalg.cholesky(K + epsilon * I, upper=True)``.
    Triangularity and positive definiteness are assumed, not validated.

    All repeated blocks (and all batched columns) share a single triangular
    product or solve: the ops reshape the metric block into the columns of one
    ``(block, repeats * cols)`` matrix, so no repeated factor or full block
    diagonal is ever formed. ``size = repeats * F.shape[0]``.
    """

    F: jax.Array
    repeats: int = field(default=1, kw_only=True)
    free_scale: float = field(default=1.0, kw_only=True)
    size: int = field(init=False)

    def __post_init__(self):
        F = jnp.asarray(self.F)
        if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] == 0:
            raise ValueError("F must be a nonempty square matrix")
        dtype = jnp.result_type(F, 1.0)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("F must have a real floating-point dtype")
        if self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        object.__setattr__(self, "F", F.astype(dtype))
        object.__setattr__(self, "size", self.repeats * F.shape[0])
        object.__setattr__(
            self, "free_scale", _canonical_free_scale(self.free_scale, dtype)
        )

    def _map_blocks(self, block_op, v):
        _check_leading_size(v, self.size)
        block_size = self.F.shape[0]
        trailing_shape = v.shape[1:]
        packed = jnp.moveaxis(
            v.reshape((self.repeats, block_size) + trailing_shape), 0, 1
        ).reshape(block_size, -1)
        return jnp.moveaxis(
            block_op(packed).reshape((block_size, self.repeats) + trailing_shape),
            0,
            1,
        ).reshape((self.size,) + trailing_shape)

    def factor_apply(self, v, ctx):
        return self._map_blocks(lambda m: self.F @ m, v)

    def factor_solve(self, v, ctx):
        return self._map_blocks(
            lambda m: jsp_linalg.solve_triangular(self.F, m, lower=False), v
        )

    def factor_solve_transpose(self, v, ctx):
        return self._map_blocks(
            lambda m: jsp_linalg.solve_triangular(self.F.T, m, lower=True), v
        )


register_pytree_dataclass(
    RepeatedFactorMetric,
    data_fields=("F", "free_scale"),
    meta_fields=("repeats", "size"),
)
