"""Penalty types and constructors for ``RidgeLevenbergMarquardt``.

A :class:`RidgePenalty` packages the positive-semidefinite seminorm
``q(x) = ||L x||^2`` (``M0 = L'L``, allowed singular) through factor
callbacks, so the solver never forms ``M0`` unless a dense path asks for it.
``repeated_dense_penalty`` is the kernel workhorse: ``repeats`` copies of a
Gram matrix ``K`` on the head coordinates with an unpenalized zero-padded
tail, factored once and applied with batched triangular products.

A :class:`Whitener` extends the penalty with an invertible square extension
``L_bar`` of the factor (identity on the unpenalized tail), letting the
solver run the whole subproblem in the whitened variable ``y = L_bar x``
where the penalty rows are the constant ``[I_k | 0]``.
``repeated_block_whitener`` is the kernel workhorse mirroring
``repeated_dense_penalty``; ``whitener_from_factor`` covers general dense
square factors.
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


@dataclass(frozen=True)
class Whitener(RidgePenalty):
    """A :class:`RidgePenalty` whose factor extends to an invertible square
    ``L_bar`` (identity on the unpenalized tail), enabling the solver's
    whitened change of variables ``y = L_bar x``.

    The base penalty fields stay fully populated and consistent, so a
    ``Whitener`` degrades gracefully to a plain penalty anywhere the whitened
    algebra is not implemented. The extra callbacks act on the flattened
    parameter vector and accept a vector or a matrix whose LEADING axis is
    the parameter dimension (columns are batched):

    - ``whiten(v)``: ``L_bar v`` -- its first ``num_rows`` entries must
      coincide with ``sqrt_apply(v)`` (the head rows ARE the penalty factor)
      and the tail must pass through unchanged
    - ``unwhiten(v)``: ``L_bar^{-1} v``
    - ``unwhiten_transpose(v)``: ``L_bar^{-T} v``

    Under a ``Whitener`` the y-space penalty rows are the constant
    ``[I_k | 0]`` (never materialized), so
    :class:`~nlls_gram.RidgeLevenbergMarquardt` damps and factors the
    whitened subproblem directly -- see the solver's whitening notes for the
    changed ``grad_norm``/``step_norm`` semantics. The factorization behind
    the callbacks depends on the penalty alone, never the ridge weight, so
    continuation/annealing composes unchanged.
    """

    whiten: Callable | None = None
    unwhiten: Callable | None = None
    unwhiten_transpose: Callable | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("whiten", "unwhiten", "unwhiten_transpose"):
            if not callable(getattr(self, name)):
                raise TypeError(f"Whitener.{name} must be callable")


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


def repeated_block_whitener(K, *, repeats: int, zero_pad_size: int):
    """Build the repeated dense kernel penalty as a :class:`Whitener`.

    Same penalty as :func:`repeated_dense_penalty` -- the flattened layout is
    ``repeats`` head blocks penalized by the square positive-definite Gram
    matrix ``K`` followed by ``zero_pad_size`` unpenalized scalars, and every
    base field matches -- plus the whitening callbacks for the square factor
    ``L_bar = blockdiag(C', ..., C', I_tail)`` with ``K = C C'`` factored
    once (``jnp.linalg.cholesky``; ``K`` must be numerically positive
    definite here, since the whitened solve inverts the factor -- add jitter
    yourself if needed, a NaN factor propagates loudly). All repeated blocks
    batch into the columns of one triangular product/solve, and the callbacks
    accept vectors or matrices with the parameter dimension leading. The
    ridge weight never enters the factorization, so ridge continuation
    composes unchanged.
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
        if x.ndim not in (1, 2):
            raise ValueError(f"penalty {name} requires a vector or matrix")
        if x.shape[0] != size:
            raise ValueError(
                f"penalty {name} leading size must be {size}, got {x.shape[0]}"
            )

    # Head blocks batch as the columns of one (block_size, repeats * cols)
    # matrix, so every block (and every batched column) shares a single
    # triangular product or solve.
    def pack_head(x):
        trailing_shape = x.shape[1:]
        return jnp.moveaxis(
            x[:repeated_size].reshape((repeats, block_size) + trailing_shape),
            0,
            1,
        ).reshape(block_size, -1)

    def unpack_head(matrix, trailing_shape):
        return jnp.moveaxis(
            matrix.reshape((block_size, repeats) + trailing_shape), 0, 1
        ).reshape((repeated_size,) + trailing_shape)

    def head_block_map(block_op, x):
        check_input(x, total_size, "whitener input")
        trailing_shape = x.shape[1:]
        head = unpack_head(block_op(pack_head(x)), trailing_shape)
        return jnp.concatenate([head, x[repeated_size:]], axis=0)

    def whiten(x):
        return head_block_map(lambda m: C.T @ m, x)

    def unwhiten(y):
        return head_block_map(
            lambda m: jsp_linalg.solve_triangular(C.T, m, lower=False), y
        )

    def unwhiten_transpose(y):
        return head_block_map(
            lambda m: jsp_linalg.solve_triangular(C, m, lower=True), y
        )

    def sqrt_apply(x):
        check_input(x, total_size, "sqrt_apply input")
        trailing_shape = x.shape[1:]
        return unpack_head(C.T @ pack_head(x), trailing_shape)

    def sqrt_transpose_apply(y):
        check_input(y, repeated_size, "sqrt_transpose_apply input")
        trailing_shape = y.shape[1:]
        head = unpack_head(C @ pack_head(y), trailing_shape)
        if zero_pad_size == 0:
            return head
        zeros = jnp.zeros((zero_pad_size,) + trailing_shape, dtype=head.dtype)
        return jnp.concatenate([head, zeros], axis=0)

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

    return Whitener(
        sqrt_apply=sqrt_apply,
        sqrt_transpose_apply=sqrt_transpose_apply,
        num_rows=repeated_size,
        add_scaled=add_scaled,
        sqrt_rows=sqrt_rows,
        whiten=whiten,
        unwhiten=unwhiten,
        unwhiten_transpose=unwhiten_transpose,
    )


def whitener_from_factor(L_bar, *, num_rows: int):
    """Build a :class:`Whitener` from a dense square invertible factor.

    ``L_bar`` is the full ``(p, p)`` whitening map ``y = L_bar x``; its first
    ``num_rows`` rows are the penalty factor ``L``, so
    ``q(x) = ||(L_bar x)[:num_rows]||^2``. Invertibility is not validated
    (the entries may be traced) -- a singular factor propagates NaN loudly
    through the triangular/linear solves. Intended for tests and non-kernel
    uses; the kernel workhorse is :func:`repeated_block_whitener`.
    """

    L_bar = jnp.asarray(L_bar)
    if L_bar.ndim != 2 or L_bar.shape[0] != L_bar.shape[1]:
        raise ValueError("L_bar must be a square matrix")
    dtype = jnp.result_type(L_bar, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError("L_bar must have a real floating-point dtype")
    L_bar = L_bar.astype(dtype)
    size = L_bar.shape[0]
    if (
        isinstance(num_rows, bool)
        or not isinstance(num_rows, int)
        or not 0 <= num_rows <= size
    ):
        raise ValueError(
            f"num_rows must be a Python int in [0, {size}], got {num_rows!r}"
        )
    L = L_bar[:num_rows]
    M0 = L.T @ L

    def sqrt_apply(x):
        return L @ x

    def sqrt_transpose_apply(y):
        return L.T @ y

    def add_scaled(H, c):
        return H + jnp.asarray(c, dtype=H.dtype) * M0.astype(H.dtype)

    def sqrt_rows():
        return L

    def whiten(v):
        return L_bar @ v

    def unwhiten(v):
        return jnp.linalg.solve(L_bar, v)

    def unwhiten_transpose(v):
        return jnp.linalg.solve(L_bar.T, v)

    return Whitener(
        sqrt_apply=sqrt_apply,
        sqrt_transpose_apply=sqrt_transpose_apply,
        num_rows=num_rows,
        add_scaled=add_scaled,
        sqrt_rows=sqrt_rows,
        whiten=whiten,
        unwhiten=unwhiten,
        unwhiten_transpose=unwhiten_transpose,
    )
