from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


@dataclass(frozen=True)
class Metric:
    """Positive-definite parameter-space metric ``M`` given through callbacks.

    All callbacks act on the flattened parameter vector. With ``P = M^{-1}``
    and ``S`` satisfying ``S S' = M^{-1}``:

    - ``solve(x)``: ``M^{-1} x``
    - ``norm(x)``: ``sqrt(x' M x)``
    - ``inv_sqrt(x)``: ``S x``
    - ``inv_sqrt_transpose(x)``: ``S' x``

    Fields left as ``None`` default to the identity metric. Which fields are
    required depends on the solver configuration; see
    ``UnderdeterminedLevenbergMarquardt``.
    """

    solve: Callable | None = None
    norm: Callable | None = None
    inv_sqrt: Callable | None = None
    inv_sqrt_transpose: Callable | None = None


def metric_from_cholesky(L):
    """Build a dense ``Metric`` from a lower-triangular Cholesky factor.

    ``L`` is the factor of the metric matrix ``M = L @ L.T``, as returned by
    ``jnp.linalg.cholesky``.
    """

    def solve(x):
        y = jsp_linalg.solve_triangular(L, x, lower=True)
        return jsp_linalg.solve_triangular(L.T, y, lower=False)

    def norm(x):
        return jnp.linalg.norm(L.T @ x)

    def inv_sqrt(x):
        return jsp_linalg.solve_triangular(L.T, x, lower=False)

    def inv_sqrt_transpose(x):
        return jsp_linalg.solve_triangular(L, x, lower=True)

    return Metric(
        solve=solve,
        norm=norm,
        inv_sqrt=inv_sqrt,
        inv_sqrt_transpose=inv_sqrt_transpose,
    )
