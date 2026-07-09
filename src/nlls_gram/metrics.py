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


def metric_from_tridiagonal_precision(diag, off_diag, parallel=None):
    """Build an O(n) ``Metric`` from a symmetric tridiagonal precision matrix.

    ``diag`` (length n) and ``off_diag`` (length n-1) give the main and
    off-diagonal of ``T = M^{-1}``, which must be positive definite (not
    validated, since inputs may be traced; non-PD input silently produces a
    non-metric). Suited to
    Markov kernels, where the Gram inverse is exactly tridiagonal (e.g.
    Matern-1/2 / Ornstein-Uhlenbeck on sorted points): every callback costs
    O(n) and nothing is factored densely.

    ``parallel`` picks how the one-time bidiagonal Cholesky setup runs:
    ``None`` (default) uses an associative O(log n)-depth scan off-CPU, where
    a sequential scan pays a kernel launch per step, and the sequential scan
    on CPU, where it is faster.
    """

    diag = jnp.asarray(diag)
    off_diag = jnp.asarray(off_diag)
    if diag.ndim != 1 or off_diag.shape != (diag.shape[0] - 1,):
        raise ValueError(
            "diag must be 1-D and off_diag must have shape (len(diag) - 1,)"
        )
    if parallel is None:
        parallel = jax.default_backend() != "cpu"
    zero = jnp.zeros((1,), dtype=diag.dtype)
    lower = jnp.concatenate([zero, off_diag])
    upper = jnp.concatenate([off_diag, zero])

    # Bidiagonal Cholesky T = C C': c_off[i] = off[i] / c_d[i],
    # c_d[i+1] = sqrt(diag[i+1] - c_off[i]^2), i.e. the pivot recurrence
    # delta_{i+1} = diag_{i+1} - off_i^2 / delta_i with delta = c_d^2.
    if off_diag.shape[0] == 0:
        c_d = jnp.sqrt(diag)
        c_off = off_diag
    elif parallel:
        # The pivot recurrence is a Mobius map delta -> (a delta + b)/delta,
        # so cumulative 2x2 matrix products compose it associatively; the
        # per-element normalization leaves the projective action unchanged
        # while keeping the entries O(1).
        mobius = jnp.stack(
            [
                jnp.stack([diag[1:], -(off_diag**2)], axis=-1),
                jnp.stack([jnp.ones_like(off_diag), jnp.zeros_like(off_diag)], axis=-1),
            ],
            axis=-2,
        )

        def combine(earlier, later):
            product = later @ earlier
            scale = jnp.max(jnp.abs(product), axis=(-2, -1), keepdims=True)
            return product / scale

        cumulative = jax.lax.associative_scan(combine, mobius)
        delta_rest = (cumulative[:, 0, 0] * diag[0] + cumulative[:, 0, 1]) / (
            cumulative[:, 1, 0] * diag[0] + cumulative[:, 1, 1]
        )
        delta = jnp.concatenate([diag[:1], delta_rest])
        c_d = jnp.sqrt(delta)
        c_off = off_diag / c_d[:-1]
    else:

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


def metric_from_diagonal(weights):
    """Build a ``Metric`` for the diagonal metric ``M = diag(weights)``.

    ``weights`` (length n) must be positive -- not validated, since inputs
    may be traced. Every callback is elementwise.
    """

    weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be 1-D")
    sqrt_weights = jnp.sqrt(weights)

    def expand(v, x):
        return v.reshape(v.shape + (1,) * (x.ndim - 1))

    def solve(x):
        return x / expand(weights, x)

    def norm(x):
        return jnp.sqrt(x @ (weights * x))

    def inv_sqrt(x):
        return x / expand(sqrt_weights, x)

    return Metric(
        solve=solve,
        norm=norm,
        inv_sqrt=inv_sqrt,
        inv_sqrt_transpose=inv_sqrt,
    )


def blockdiag_metric(blocks):
    """Compose per-block ``Metric``s into one block-diagonal ``Metric``.

    ``blocks`` is a sequence of ``(metric, size)`` pairs in the order the
    flattened parameter vector is laid out. ``solve``, ``inv_sqrt``, and
    ``inv_sqrt_transpose`` slice on the leading axis (vector and matrix
    inputs both work); ``norm`` combines the block norms in quadrature. A
    callback left ``None`` by a block defaults to the identity metric on that
    block, matching the bare ``Metric()`` convention.
    """

    if not blocks:
        raise ValueError("blocks must contain at least one (metric, size) pair")
    metrics = [metric for metric, _ in blocks]
    offsets = [0]
    for _, size in blocks:
        offsets.append(offsets[-1] + size)

    def split(x):
        return [
            x[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        ]

    def identity(x):
        return x

    def blockwise(callbacks):
        filled = [identity if callback is None else callback for callback in callbacks]

        def apply(x):
            return jnp.concatenate(
                [
                    callback(part)
                    for callback, part in zip(filled, split(x), strict=True)
                ]
            )

        return apply

    norms = [
        jnp.linalg.norm if metric.norm is None else metric.norm for metric in metrics
    ]

    def norm(x):
        parts = split(x)
        return jnp.sqrt(
            sum(
                callback(part) ** 2 for callback, part in zip(norms, parts, strict=True)
            )
        )

    return Metric(
        solve=blockwise([metric.solve for metric in metrics]),
        norm=norm,
        inv_sqrt=blockwise([metric.inv_sqrt for metric in metrics]),
        inv_sqrt_transpose=blockwise([metric.inv_sqrt_transpose for metric in metrics]),
    )
