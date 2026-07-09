from collections.abc import Callable
from dataclasses import dataclass

import jax
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


def metric_from_tridiagonal_precision(diag, off_diag):
    """Build an O(n) ``Metric`` from a symmetric tridiagonal precision matrix.

    ``diag`` (length n) and ``off_diag`` (length n-1) give the main and
    off-diagonal of ``T = M^{-1}``, which must be positive definite. Suited to
    Markov kernels, where the Gram inverse is exactly tridiagonal (e.g.
    Matern-1/2 / Ornstein-Uhlenbeck on sorted points): every callback costs
    O(n) and nothing is factored densely.
    """

    diag = jnp.asarray(diag)
    off_diag = jnp.asarray(off_diag)
    if diag.ndim != 1 or off_diag.shape != (diag.shape[0] - 1,):
        raise ValueError(
            "diag must be 1-D and off_diag must have shape (len(diag) - 1,)"
        )
    zero = jnp.zeros((1,), dtype=diag.dtype)
    lower = jnp.concatenate([zero, off_diag])
    upper = jnp.concatenate([off_diag, zero])

    # Bidiagonal Cholesky T = C C': c_off[i] = off[i] / c_d[i],
    # c_d[i+1] = sqrt(diag[i+1] - c_off[i]^2).
    def chol_step(c_d_prev, inputs):
        off_i, diag_next = inputs
        c_off_i = off_i / c_d_prev
        c_d_i = jnp.sqrt(diag_next - c_off_i**2)
        return c_d_i, (c_off_i, c_d_i)

    c_d_0 = jnp.sqrt(diag[0])
    _, (c_off, c_d_rest) = jax.lax.scan(chol_step, c_d_0, (off_diag, diag[1:]))
    c_d = jnp.concatenate([c_d_0[None], c_d_rest])

    def expand(v, x):
        return v.reshape(v.shape + (1,) * (x.ndim - 1))

    def shift_up(x):
        return jnp.concatenate([x[1:], jnp.zeros_like(x[:1])])

    def shift_down(x):
        return jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]])

    def solve(x):
        # M^{-1} x = T x: a tridiagonal matvec along the leading axis.
        return (
            expand(diag, x) * x
            + expand(upper, x) * shift_up(x)
            + expand(lower, x) * shift_down(x)
        )

    def norm(x):
        # x' M x = x' T^{-1} x through one O(n) tridiagonal solve.
        y = jax.lax.linalg.tridiagonal_solve(lower, diag, upper, x[:, None])
        return jnp.sqrt(x @ y[:, 0])

    def inv_sqrt(x):
        # S = C (lower bidiagonal) with S S' = M^{-1}.
        return expand(c_d, x) * x + expand(
            jnp.concatenate([zero, c_off]), x
        ) * shift_down(x)

    def inv_sqrt_transpose(x):
        return expand(c_d, x) * x + expand(
            jnp.concatenate([c_off, zero]), x
        ) * shift_up(x)

    return Metric(
        solve=solve,
        norm=norm,
        inv_sqrt=inv_sqrt,
        inv_sqrt_transpose=inv_sqrt_transpose,
    )
