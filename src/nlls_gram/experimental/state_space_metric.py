"""Repeated state-space kernel metric in linear storage (experimental).

``StateSpaceMetric`` is a :class:`~nlls_gram.Metric` whose factor is the
quasiseparable Cholesky of a stationary state-space kernel Gram matrix, never
formed densely: ``O(n)`` storage and ``O(n)`` work per apply for ``n``
coordinates, against ``O(n^2)``/``O(n^3)`` for the dense route. One structured
factor is shared by every repeated block, which are processed as batched
right-hand sides.
"""

from dataclasses import InitVar, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from nlls_gram.experimental import quasiseparable
from nlls_gram.experimental.quasiseparable import matern_state_space
from nlls_gram.metrics import Metric, _canonical_free_scale, _check_leading_size
from nlls_gram.utilities import register_pytree_dataclass

__all__ = ["StateSpaceMetric", "matern_state_space"]


@dataclass(frozen=True, eq=False)
class StateSpaceMetric(Metric):
    """``repeats`` copies of a shifted state-space kernel Gram matrix.

    ``t`` is a strictly increasing 1-D coordinate; ``h``, ``Pinf``, and
    ``transition`` define the stationary state-space kernel, as
    :func:`matern_state_space` supplies for Matern-1/2, -3/2, and -5/2.
    ``transition(dt)`` returns the transpose of the textbook state transition
    for each gap. Non-increasing coordinates propagate NaN rather than quietly
    defining a nonstationary factor. All four are consumed at construction;
    only the quasiseparable factor generators are stored (as traced leaves).

    The metric is ``blockdiag(K + epsilon I, ...)`` over ``repeats`` blocks;
    ``epsilon`` is added before the quasiseparable factorization. Anything
    beyond ``size = repeats * len(t)`` is the solver's free block, weighted by
    ``free_scale``.

    ``parallel`` selects sequential or associative scans; the default picks
    associative scans only for float64 off CPU. Pass it explicitly when arrays
    use nondefault device placement.
    """

    t: InitVar[Any]
    h: InitVar[Any]
    Pinf: InitVar[Any]
    transition: InitVar[Any]
    repeats: int = field(default=1, kw_only=True)
    epsilon: InitVar[Any] = field(default=1e-8, kw_only=True)
    free_scale: Any = field(default=1.0, kw_only=True)
    parallel: bool | None = field(default=None, kw_only=True)
    size: int = field(init=False)
    factor: tuple = field(init=False)

    def __post_init__(self, t, h, Pinf, transition, epsilon):
        t = jnp.asarray(t)
        if t.ndim != 1 or t.shape[0] == 0:
            raise ValueError("t must be a nonempty 1-D array")
        if self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        d, p, q, A = quasiseparable._state_space_generators(t, h, Pinf, transition)
        dtype = jnp.result_type(d, p, q, A, 1.0)
        epsilon = jnp.asarray(epsilon, dtype=dtype)
        # A non-increasing coordinate or a non-positive shift poisons the
        # factor rather than silently producing a wrong one.
        epsilon = jnp.where(
            (epsilon > 0.0) & jnp.all(jnp.diff(t) > 0.0), epsilon, jnp.nan
        )
        d = d.astype(dtype) + epsilon
        p, q, A = (v.astype(dtype) for v in (p, q, A))
        parallel = self.parallel
        if parallel is None:
            parallel = jax.default_backend() != "cpu" and dtype == jnp.float64
        c, w = quasiseparable._cholesky(d, p, q, A)
        object.__setattr__(self, "parallel", bool(parallel))
        object.__setattr__(self, "factor", (c, p, w, A))
        object.__setattr__(self, "size", self.repeats * t.shape[0])
        object.__setattr__(
            self, "free_scale", _canonical_free_scale(self.free_scale, dtype)
        )

    def _map_blocks(self, block_op, v):
        _check_leading_size(v, self.size)
        block = self.size // self.repeats
        trailing = v.shape[1:]
        packed = jnp.moveaxis(
            v.reshape((self.repeats, block) + trailing), 0, 1
        ).reshape(block, -1)
        return jnp.moveaxis(
            block_op(packed).reshape((block, self.repeats) + trailing), 0, 1
        ).reshape((self.size,) + trailing)

    def factor_apply(self, v, ctx):
        c, p, w, A = self.factor
        return self._map_blocks(
            lambda m: quasiseparable._cholesky_transpose_matvec(
                c, p, w, A, m, self.parallel
            ),
            v,
        )

    def factor_solve(self, v, ctx):
        c, p, w, A = self.factor
        return self._map_blocks(
            lambda m: quasiseparable._backward_substitution(
                c, p, w, A, m, self.parallel
            ),
            v,
        )

    def factor_solve_transpose(self, v, ctx):
        c, p, w, A = self.factor
        return self._map_blocks(
            lambda m: quasiseparable._forward_substitution(
                c, p, w, A, m, self.parallel
            ),
            v,
        )


register_pytree_dataclass(
    StateSpaceMetric,
    data_fields=("factor", "free_scale"),
    meta_fields=("repeats", "parallel", "size"),
)
