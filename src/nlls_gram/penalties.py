"""Penalty types and constructors for ``RidgeLevenbergMarquardt``.

A :class:`RidgePenalty` packages the positive-semidefinite seminorm
``q(x) = ||L x||^2`` (``M0 = L'L``, allowed singular) through factor
callbacks, so the solver never forms ``M0`` unless a dense path asks for it.
``repeated_dense_penalty`` is the kernel workhorse: ``repeats`` copies of a
Gram matrix ``K`` on the head coordinates with an unpenalized zero-padded
tail, factored once and applied with batched triangular products.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from nlls_gram.metrics import _validate_repeated_shifted_layout


@dataclass(frozen=True)
class RidgePenalty:
    """Positive-semidefinite penalty ``q(x) = ||L x||^2`` given through callbacks.

    All callbacks act on the flattened parameter vector. ``sqrt_apply`` may
    return a vector of any length ``k`` (the number of penalty rows);
    ``M0 = L'L`` is never formed unless ``add_scaled`` is omitted.

    - ``sqrt_apply(x)``: ``L x`` (required)
    - ``sqrt_transpose_apply(y)``: ``L' y`` (required)
    - ``num_rows``: static int ``k``, sizes the augmented system for the
      dense/qr/lsmr paths (required)
    - ``quadratic(x)``: ``x' L'L x``; ``None`` defaults to
      ``||sqrt_apply(x)||^2``. A custom callback must be numerically
      consistent with ``sqrt_apply`` -- the solver compares objective values
      built from both forms when it accepts or rejects a step.
    - ``add_scaled(H, c)``: ``H + c * L'L`` for a dense ``(p, p)`` ``H``;
      ``None`` falls back to materializing the ``(k, p)`` factor via
      ``sqrt_rows`` and paying a ``k x p x p`` product per solver update --
      provide it for anything large.
    - ``sqrt_rows()``: the dense ``(k, p)`` factor ``L``; ``None`` defaults to
      assembling it from ``sqrt_apply`` applied to an identity basis (a
      trace-time cost of ``p`` applications).

    Instances hash by their callback fields (Python function identity), so
    construct a penalty once at setup scope and reuse it -- rebuilding an
    equal-config penalty per call would key a fresh solver compilation.
    """

    sqrt_apply: Callable
    sqrt_transpose_apply: Callable
    num_rows: int
    quadratic: Callable | None = None
    add_scaled: Callable | None = None
    sqrt_rows: Callable | None = None

    def __post_init__(self):
        if not callable(self.sqrt_apply):
            raise TypeError("RidgePenalty.sqrt_apply must be callable")
        if not callable(self.sqrt_transpose_apply):
            raise TypeError("RidgePenalty.sqrt_transpose_apply must be callable")
        if (
            isinstance(self.num_rows, bool)
            or not isinstance(self.num_rows, int)
            or self.num_rows < 0
        ):
            raise ValueError("num_rows must be a nonnegative Python int")
        for name in ("quadratic", "add_scaled", "sqrt_rows"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"RidgePenalty.{name} must be callable or None")


def repeated_dense_penalty(K, *, repeats: int, zero_pad_size: int):
    """Build the repeated dense kernel penalty ``q(x) = sum_j alpha_j' K alpha_j``.

    The flattened parameter layout is ``repeats`` head blocks of kernel
    coefficients ``alpha_j`` (each sized like the square positive-semidefinite
    Gram matrix ``K``) followed by ``zero_pad_size`` unpenalized scalars, i.e.
    ``M0 = blockdiag(K, ..., K, 0)`` -- no epsilon shift on the tail. The
    constructor factors ``K = C C'`` once (``jnp.linalg.cholesky``; add jitter
    to ``K`` yourself if it is numerically semidefinite -- a NaN factor
    propagates loudly) and batches all repeated blocks into the columns of
    each triangular product; the zero-padded tail contributes no penalty rows
    (``num_rows = repeats * K.shape[0]``). ``add_scaled`` scatter-adds
    ``c * K`` into the head diagonal blocks without materializing the block
    diagonal.
    """

    _validate_repeated_shifted_layout(repeats, zero_pad_size)
    K = jnp.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] == 0:
        raise ValueError("K must be a nonempty square matrix")
    dtype = jnp.result_type(K, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError("K must have a real floating-point dtype")
    K = K.astype(dtype)
    block_size = K.shape[0]
    repeated_size = repeats * block_size
    total_size = repeated_size + zero_pad_size
    C = jnp.linalg.cholesky(K)

    def check_input(x, size, name):
        if x.ndim != 1:
            raise ValueError(f"penalty {name} requires a vector")
        if x.shape[0] != size:
            raise ValueError(f"penalty {name} size must be {size}, got {x.shape[0]}")

    # Head blocks batch as the columns of one (block_size, repeats) matrix, so
    # every block shares a single triangular product.
    def pack(v):
        return v.reshape(repeats, block_size).T

    def unpack(matrix):
        return matrix.T.reshape(repeated_size)

    def sqrt_apply(x):
        check_input(x, total_size, "sqrt_apply input")
        return unpack(C.T @ pack(x[:repeated_size]))

    def sqrt_transpose_apply(y):
        check_input(y, repeated_size, "sqrt_transpose_apply input")
        head = unpack(C @ pack(y))
        if zero_pad_size == 0:
            return head
        return jnp.concatenate([head, jnp.zeros(zero_pad_size, dtype=head.dtype)])

    def add_scaled(H, c):
        scaled = jnp.asarray(c, dtype=H.dtype) * K.astype(H.dtype)
        for j in range(repeats):
            start = j * block_size
            H = H.at[start : start + block_size, start : start + block_size].add(
                scaled
            )
        return H

    def sqrt_rows():
        rows = jsp_linalg.block_diag(*([C.T] * repeats))
        if zero_pad_size == 0:
            return rows
        return jnp.concatenate(
            [rows, jnp.zeros((repeated_size, zero_pad_size), dtype=rows.dtype)],
            axis=1,
        )

    return RidgePenalty(
        sqrt_apply=sqrt_apply,
        sqrt_transpose_apply=sqrt_transpose_apply,
        num_rows=repeated_size,
        add_scaled=add_scaled,
        sqrt_rows=sqrt_rows,
    )


def penalty_from_factor(L):
    """Build a :class:`RidgePenalty` from a dense ``(k, p)`` factor ``L``.

    ``q(x) = ||L x||^2`` with ``M0 = L'L`` precomputed once for
    ``add_scaled``. ``L`` may be rectangular in either direction and need not
    have full rank.
    """

    L = jnp.asarray(L)
    if L.ndim != 2:
        raise ValueError("L must be a matrix")
    dtype = jnp.result_type(L, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError("L must have a real floating-point dtype")
    L = L.astype(dtype)
    M0 = L.T @ L

    def sqrt_apply(x):
        return L @ x

    def sqrt_transpose_apply(y):
        return L.T @ y

    def add_scaled(H, c):
        return H + jnp.asarray(c, dtype=H.dtype) * M0.astype(H.dtype)

    def sqrt_rows():
        return L

    return RidgePenalty(
        sqrt_apply=sqrt_apply,
        sqrt_transpose_apply=sqrt_transpose_apply,
        num_rows=int(L.shape[0]),
        add_scaled=add_scaled,
        sqrt_rows=sqrt_rows,
    )


def identity_penalty(size: int):
    """Build the identity penalty ``q(x) = ||x||^2`` on ``size`` parameters.

    The minimum-Euclidean-norm choice: ``L = I``, every coordinate penalized
    equally.
    """

    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive Python int")

    def check_input(x):
        if x.ndim != 1 or x.shape[0] != size:
            raise ValueError(
                f"identity penalty requires a vector of size {size}, "
                f"got shape {x.shape}"
            )
        return x

    def sqrt_apply(x):
        return check_input(x)

    def sqrt_transpose_apply(y):
        return check_input(y)

    def quadratic(x):
        check_input(x)
        return jnp.sum(x**2)

    def add_scaled(H, c):
        return H + jnp.asarray(c, dtype=H.dtype) * jnp.eye(size, dtype=H.dtype)

    def sqrt_rows():
        return jnp.eye(size, dtype=jnp.result_type(float))

    return RidgePenalty(
        sqrt_apply=sqrt_apply,
        sqrt_transpose_apply=sqrt_transpose_apply,
        num_rows=size,
        quadratic=quadratic,
        add_scaled=add_scaled,
        sqrt_rows=sqrt_rows,
    )
