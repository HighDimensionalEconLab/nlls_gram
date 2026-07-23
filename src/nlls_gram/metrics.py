from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from nlls_gram import quasiseparable


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
    required depends on the solver configuration; see ``LevenbergMarquardt``.
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

    L = jnp.asarray(L)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix")

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


def metric_from_diagonal(weights):
    """Build a ``Metric`` for the diagonal metric ``M = diag(weights)``.

    ``weights`` must be a one-dimensional array of positive values. Positivity
    is not validated because the values may be traced. Every callback is
    elementwise.
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


def _validate_repeated_shifted_layout(repeats, zero_pad_size):
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if (
        not isinstance(zero_pad_size, int)
        or isinstance(zero_pad_size, bool)
        or zero_pad_size < 0
    ):
        raise ValueError("zero_pad_size must be a nonnegative integer")


def _repeated_shifted_metric(block_metric, block_size, repeats, zero_pad_size, epsilon):
    repeated_size = repeats * block_size
    total_size = repeated_size + zero_pad_size
    sqrt_epsilon = jnp.sqrt(epsilon)

    def check_input(x, *, norm=False):
        expected_ndim = (1,) if norm else (1, 2)
        if x.ndim not in expected_ndim:
            kind = "a vector" if norm else "a vector or matrix"
            raise ValueError(f"metric callback requires {kind}")
        if x.shape[0] != total_size:
            raise ValueError(
                f"metric leading size must be {total_size}, got {x.shape[0]}"
            )

    def packed_head(x):
        trailing_shape = x.shape[1:]
        return jnp.moveaxis(
            x[:repeated_size].reshape((repeats, block_size) + trailing_shape),
            0,
            1,
        ).reshape(block_size, -1)

    def unpack_head(x, trailing_shape):
        return jnp.moveaxis(
            x.reshape((block_size, repeats) + trailing_shape), 0, 1
        ).reshape((repeated_size,) + trailing_shape)

    def apply(block_callback, tail_scale):
        def callback(x):
            check_input(x)
            trailing_shape = x.shape[1:]
            head = unpack_head(block_callback(packed_head(x)), trailing_shape)
            tail = x[repeated_size:] / tail_scale
            return jnp.concatenate([head, tail], axis=0)

        return callback

    def norm(x):
        check_input(x, norm=True)
        head_norm = block_metric.norm(packed_head(x))
        tail = x[repeated_size:]
        return jnp.sqrt(head_norm**2 + epsilon * jnp.vdot(tail, tail))

    return Metric(
        solve=apply(block_metric.solve, epsilon),
        norm=norm,
        inv_sqrt=apply(block_metric.inv_sqrt, sqrt_epsilon),
        inv_sqrt_transpose=apply(block_metric.inv_sqrt_transpose, sqrt_epsilon),
    )


def repeated_shifted_dense_metric(K, *, repeats: int, zero_pad_size: int, epsilon):
    """Build a repeated shifted dense metric without repeating its factor.

    The metric is ``blockdiag(K, ..., K, 0) + epsilon * I``, with ``repeats``
    copies of the square positive-semidefinite matrix ``K`` and a trailing zero
    block of size ``zero_pad_size``. ``epsilon`` must be a positive scalar.

    The constructor factors ``K + epsilon * I`` once, stores one dense
    Cholesky factor and the scalar shift, and batches all repeated blocks into
    the right-hand-side columns of each triangular solve. It never forms or
    stores the full block diagonal, repeated factors, or a padding vector.

    The flattened parameter layout is the repeated ``K`` blocks followed by
    the zero-padded coordinates. All four metric callbacks are provided;
    ``solve``, ``inv_sqrt``, and ``inv_sqrt_transpose`` accept vectors or
    matrices, while ``norm`` accepts a vector.
    """

    _validate_repeated_shifted_layout(repeats, zero_pad_size)
    K = jnp.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] == 0:
        raise ValueError("K must be a nonempty square matrix")

    original_epsilon = epsilon
    epsilon = jnp.asarray(epsilon)
    if epsilon.ndim != 0:
        raise ValueError("epsilon must be a scalar")
    dtype = jnp.result_type(K, epsilon, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError("K and epsilon must have a real floating-point dtype")
    if (
        not isinstance(original_epsilon, (jax.Array, jax.core.Tracer))
        and float(original_epsilon) <= 0.0
    ):
        raise ValueError("epsilon must be positive")
    K = K.astype(dtype)
    epsilon = epsilon.astype(dtype)
    epsilon = jnp.where(epsilon > 0.0, epsilon, jnp.nan)

    block_size = K.shape[0]
    shifted = K + epsilon * jnp.eye(block_size, dtype=K.dtype)
    block_metric = metric_from_cholesky(jnp.linalg.cholesky(shifted))
    return _repeated_shifted_metric(
        block_metric, block_size, repeats, zero_pad_size, epsilon
    )


def _metric_from_quasiseparable(d, p, q, A, *, epsilon, parallel):
    d = jnp.asarray(d) + epsilon
    p = jnp.asarray(p)
    q = jnp.asarray(q)
    A = jnp.asarray(A)
    n = d.shape[0]
    if p.ndim != 2 or p.shape[0] != n:
        raise ValueError("p must have shape (len(d), state_size)")
    state_size = p.shape[1]
    if q.shape != (n, state_size) or A.shape != (n, state_size, state_size):
        raise ValueError(
            "q must have shape (len(d), state_size) and A must have shape "
            "(len(d), state_size, state_size)"
        )
    if parallel is None:
        parallel = jax.default_backend() != "cpu" and d.dtype == jnp.float64
    c, w = quasiseparable._cholesky(d, p, q, A)

    def solve(x):
        y = quasiseparable._forward_substitution(c, p, w, A, x.reshape(n, -1), parallel)
        return quasiseparable._backward_substitution(c, p, w, A, y, parallel).reshape(
            x.shape
        )

    def norm(x):
        y = quasiseparable._cholesky_transpose_matvec(
            c, p, w, A, x.reshape(n, -1), parallel
        )
        return jnp.linalg.norm(y)

    def inv_sqrt(x):
        return quasiseparable._backward_substitution(
            c, p, w, A, x.reshape(n, -1), parallel
        ).reshape(x.shape)

    def inv_sqrt_transpose(x):
        return quasiseparable._forward_substitution(
            c, p, w, A, x.reshape(n, -1), parallel
        ).reshape(x.shape)

    return Metric(
        solve=solve,
        norm=norm,
        inv_sqrt=inv_sqrt,
        inv_sqrt_transpose=inv_sqrt_transpose,
    )


def repeated_shifted_state_space_metric(
    t,
    h,
    Pinf,
    transition,
    *,
    repeats: int,
    zero_pad_size: int,
    epsilon,
    parallel=None,
):
    """Build a repeated shifted state-space kernel metric in linear storage.

    ``t`` is a strictly increasing one-dimensional coordinate. ``h``,
    ``Pinf``, and ``transition`` define a stationary state-space kernel; the
    convenience function ``matern_state_space`` supplies these objects for the
    Matérn-1/2, Matérn-3/2, and Matérn-5/2 kernels. Non-increasing coordinates
    propagate ``NaN`` rather than silently defining a nonstationary factor.
    ``transition(dt)`` returns the transpose of the textbook state transition
    for each gap in ``dt``.

    The resulting metric is ``blockdiag(K, ..., K, 0) + epsilon * I``, with
    ``repeats`` copies of the implicit kernel Gram matrix ``K`` and a trailing
    zero block of size ``zero_pad_size``. ``epsilon`` is added before the
    quasiseparable Cholesky factorization. One structured factor is shared by
    every repeated block, and all blocks are processed as batched right-hand
    sides. No dense ``K``, repeated factor, full block diagonal, or padding
    vector is formed.

    ``parallel`` selects sequential or associative scans. The default uses the
    process backend and chooses associative scans only for float64 metrics off
    CPU; pass it explicitly when arrays use nondefault device placement. All
    four metric callbacks are provided; ``solve``, ``inv_sqrt``, and
    ``inv_sqrt_transpose`` accept vectors or matrices, while ``norm`` accepts a
    vector.
    """

    _validate_repeated_shifted_layout(repeats, zero_pad_size)
    t = jnp.asarray(t)
    if t.ndim != 1 or t.shape[0] == 0:
        raise ValueError("t must be a nonempty 1-D array")
    d, p, q, A = quasiseparable._state_space_generators(t, h, Pinf, transition)

    original_epsilon = epsilon
    epsilon = jnp.asarray(epsilon)
    if epsilon.ndim != 0:
        raise ValueError("epsilon must be a scalar")
    dtype = jnp.result_type(d, p, q, A, epsilon, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError(
            "state-space generators and epsilon must have a real floating-point dtype"
        )
    if (
        not isinstance(original_epsilon, (jax.Array, jax.core.Tracer))
        and float(original_epsilon) <= 0.0
    ):
        raise ValueError("epsilon must be positive")
    d = d.astype(dtype)
    p = p.astype(dtype)
    q = q.astype(dtype)
    A = A.astype(dtype)
    epsilon = epsilon.astype(dtype)
    valid_coordinate = jnp.all(jnp.diff(t) > 0.0)
    epsilon = jnp.where((epsilon > 0.0) & valid_coordinate, epsilon, jnp.nan)

    block_metric = _metric_from_quasiseparable(
        d, p, q, A, epsilon=epsilon, parallel=parallel
    )
    return _repeated_shifted_metric(
        block_metric, d.shape[0], repeats, zero_pad_size, epsilon
    )


def _metric_with_compute_dtype(metric: Metric, dtype) -> Metric:
    dtype = jnp.dtype(dtype)

    def wrap(callback):
        if callback is None:
            return None

        def apply(x):
            return callback(x.astype(dtype)).astype(x.dtype)

        return apply

    return Metric(
        solve=wrap(metric.solve),
        norm=wrap(metric.norm),
        inv_sqrt=wrap(metric.inv_sqrt),
        inv_sqrt_transpose=wrap(metric.inv_sqrt_transpose),
    )
