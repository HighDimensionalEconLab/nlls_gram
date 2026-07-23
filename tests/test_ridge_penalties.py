import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from nlls_gram import (
    RidgePenalty,
    identity_penalty,
    penalty_from_factor,
    repeated_dense_penalty,
)

REPEATS = 3
BLOCK = 5
PAD = 2
TOTAL = REPEATS * BLOCK + PAD


def make_gram(key, size):
    root = jax.random.normal(key, (size, size + 2))
    return root @ root.T + 0.5 * jnp.eye(size)


def materialized_m0(K, repeats, pad):
    return jsp_linalg.block_diag(*([K] * repeats), jnp.zeros((pad, pad), K.dtype))


def test_repeated_dense_penalty_matches_materialized_m0():
    K = make_gram(jax.random.key(0), BLOCK)
    penalty = repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)
    M0 = materialized_m0(K, REPEATS, PAD)
    x = jax.random.normal(jax.random.key(1), (TOTAL,))

    Lx = penalty.sqrt_apply(x)
    assert Lx.shape == (REPEATS * BLOCK,)
    assert penalty.num_rows == REPEATS * BLOCK
    np.testing.assert_allclose(jnp.sum(Lx**2), x @ (M0 @ x), rtol=1e-5)

    y = jax.random.normal(jax.random.key(2), (penalty.num_rows,))
    Lty = penalty.sqrt_transpose_apply(y)
    assert Lty.shape == (TOTAL,)
    # Adjoint identity <Lx, y> = <x, L'y> ties the two callbacks together.
    np.testing.assert_allclose(jnp.vdot(Lx, y), jnp.vdot(x, Lty), rtol=1e-5)
    # L'(Lx) = M0 x.
    np.testing.assert_allclose(
        penalty.sqrt_transpose_apply(Lx), M0 @ x, rtol=1e-4, atol=1e-4
    )


def test_repeated_dense_penalty_sqrt_rows_and_num_rows_consistency():
    K = make_gram(jax.random.key(3), BLOCK)
    penalty = repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)
    rows = penalty.sqrt_rows()
    assert rows.shape == (penalty.num_rows, TOTAL)
    x = jax.random.normal(jax.random.key(4), (TOTAL,))
    np.testing.assert_allclose(rows @ x, penalty.sqrt_apply(x), rtol=1e-5, atol=1e-6)
    y = jax.random.normal(jax.random.key(5), (penalty.num_rows,))
    np.testing.assert_allclose(
        rows.T @ y, penalty.sqrt_transpose_apply(y), rtol=1e-5, atol=1e-6
    )


def test_repeated_dense_penalty_add_scaled_matches_materialized():
    K = make_gram(jax.random.key(6), BLOCK)
    penalty = repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)
    M0 = materialized_m0(K, REPEATS, PAD)
    H = jax.random.normal(jax.random.key(7), (TOTAL, TOTAL))
    H = H + H.T
    c = 0.37
    np.testing.assert_allclose(
        penalty.add_scaled(H, c), H + c * M0, rtol=1e-5, atol=1e-5
    )


def test_repeated_dense_penalty_zero_pad_tail_is_unpenalized():
    K = make_gram(jax.random.key(8), BLOCK)
    penalty = repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)
    x = jax.random.normal(jax.random.key(9), (TOTAL,))
    bumped = x.at[REPEATS * BLOCK :].add(100.0)
    np.testing.assert_allclose(penalty.sqrt_apply(x), penalty.sqrt_apply(bumped))
    y = jax.random.normal(jax.random.key(10), (penalty.num_rows,))
    tail = penalty.sqrt_transpose_apply(y)[REPEATS * BLOCK :]
    np.testing.assert_allclose(tail, jnp.zeros(PAD))


def test_repeated_dense_penalty_no_pad_layout():
    K = make_gram(jax.random.key(11), BLOCK)
    penalty = repeated_dense_penalty(K, repeats=2, zero_pad_size=0)
    M0 = jsp_linalg.block_diag(K, K)
    x = jax.random.normal(jax.random.key(12), (2 * BLOCK,))
    np.testing.assert_allclose(
        jnp.sum(penalty.sqrt_apply(x) ** 2), x @ (M0 @ x), rtol=1e-5
    )
    np.testing.assert_allclose(
        penalty.sqrt_transpose_apply(penalty.sqrt_apply(x)),
        M0 @ x,
        rtol=1e-4,
        atol=1e-4,
    )


def test_penalty_from_factor_matches_dense_algebra():
    L = jax.random.normal(jax.random.key(13), (4, 7))
    penalty = penalty_from_factor(L)
    assert penalty.num_rows == 4
    x = jax.random.normal(jax.random.key(14), (7,))
    np.testing.assert_allclose(penalty.sqrt_apply(x), L @ x, rtol=1e-6)
    y = jax.random.normal(jax.random.key(15), (4,))
    np.testing.assert_allclose(penalty.sqrt_transpose_apply(y), L.T @ y, rtol=1e-6)
    np.testing.assert_allclose(penalty.sqrt_rows(), L)
    H = jnp.eye(7)
    np.testing.assert_allclose(
        penalty.add_scaled(H, 2.0), H + 2.0 * L.T @ L, rtol=1e-5, atol=1e-6
    )


def test_identity_penalty():
    penalty = identity_penalty(6)
    x = jax.random.normal(jax.random.key(16), (6,))
    np.testing.assert_allclose(penalty.sqrt_apply(x), x)
    np.testing.assert_allclose(penalty.sqrt_transpose_apply(x), x)
    np.testing.assert_allclose(penalty.quadratic(x), jnp.sum(x**2), rtol=1e-6)
    np.testing.assert_allclose(penalty.sqrt_rows(), jnp.eye(6))
    np.testing.assert_allclose(
        penalty.add_scaled(jnp.zeros((6, 6)), 3.0), 3.0 * jnp.eye(6)
    )
    assert penalty.num_rows == 6


def test_constructor_validation():
    with pytest.raises(ValueError, match="square"):
        repeated_dense_penalty(jnp.ones((2, 3)), repeats=1, zero_pad_size=0)
    with pytest.raises(ValueError, match="repeats"):
        repeated_dense_penalty(jnp.eye(2), repeats=0, zero_pad_size=0)
    with pytest.raises(ValueError, match="zero_pad_size"):
        repeated_dense_penalty(jnp.eye(2), repeats=1, zero_pad_size=-1)
    with pytest.raises(ValueError, match="matrix"):
        penalty_from_factor(jnp.ones(3))
    with pytest.raises(ValueError, match="positive"):
        identity_penalty(0)
    with pytest.raises(TypeError, match="sqrt_apply"):
        RidgePenalty(sqrt_apply=None, sqrt_transpose_apply=lambda y: y, num_rows=1)
    with pytest.raises(ValueError, match="num_rows"):
        RidgePenalty(
            sqrt_apply=lambda x: x, sqrt_transpose_apply=lambda y: y, num_rows=-1
        )


def test_penalty_shape_validation_raises_at_trace_time():
    K = make_gram(jax.random.key(17), BLOCK)
    penalty = repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)
    with pytest.raises(ValueError, match="size must be"):
        penalty.sqrt_apply(jnp.zeros(TOTAL + 1))
    with pytest.raises(ValueError, match="size must be"):
        penalty.sqrt_transpose_apply(jnp.zeros(penalty.num_rows + 1))
    with pytest.raises(ValueError, match="vector"):
        penalty.sqrt_apply(jnp.zeros((TOTAL, 2)))
