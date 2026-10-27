"""The gram form of QR: the dual step through a thin QR of the transpose.

Runs at JAX's default precision (x64 OFF). The float32 floor test is the
motivating case: a wide Jacobian whose singular values decay past the float32
square-root-of-epsilon line, where forming the dual Gram ``J J'`` buries the
tail modes and the gram Cholesky stalls, while the dual QR keeps them.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import QR, Cholesky, LevenbergMarquardt, QRCache

RNG = np.random.default_rng(11)
M, N = 24, 60


def decaying_jacobian(decay):
    # Singular values decay ** k, k = 0 .. M - 1, with random orthogonal factors.
    U, _ = np.linalg.qr(RNG.normal(size=(M, M)))
    V, _ = np.linalg.qr(RNG.normal(size=(N, M)))
    return U @ np.diag(decay ** np.arange(M)) @ V.T


A_NP = decaying_jacobian(0.65)
B_NP = RNG.normal(size=M)
A = jnp.asarray(A_NP, jnp.float32)
B = jnp.asarray(B_NP, jnp.float32)
X0 = jnp.zeros(N, jnp.float32)


def residual(x):
    return A @ x - B


def damped_step(x, damping):
    r = A_NP @ x - B_NP
    return -A_NP.T @ np.linalg.solve(A_NP @ A_NP.T + damping * np.eye(M), r)


def test_form_is_validated_and_resolved_by_shape():
    with pytest.raises(ValueError, match="QR.form"):
        QR(form="dual")
    assert QR()._resolved_form(M, N, penalized=False) == "gram"
    assert QR()._resolved_form(N, M, penalized=False) == "normal"
    assert QR()._resolved_form(M, N, penalized=True) == "normal"
    assert QR(form="normal")._resolved_form(M, N, penalized=False) == "normal"
    assert not QR(form="gram").supports_penalty


def test_gram_cache_holds_the_thin_qr_of_the_transpose():
    cache = QR().new_cache(M, N, N, jnp.float32, penalized=False)
    assert isinstance(cache, QRCache)
    assert cache.R.shape == (N + M, M)
    assert QR().new_cache(N, M, M, jnp.float32, penalized=False).R.shape == (
        min(N, M + 1),
        M + 1,
    )


@pytest.mark.parametrize("damping", [1e-1, 1e-3, 1e-6])
def test_gram_step_matches_the_closed_form_damped_solution(damping):
    solver = LevenbergMarquardt(
        residual,
        linear_solver=QR(form="gram"),
        init_damping=damping,
        geodesic_acceleration=False,
    )
    x0 = jnp.asarray(RNG.normal(size=N), jnp.float32)
    x1, _, info = solver.update(x0, solver.init(x0))
    expected = np.asarray(x0, np.float64) + damped_step(
        np.asarray(x0, np.float64), damping
    )
    assert bool(info.accepted)
    # GPU float32 QR at damping 1e-6 lands a few parts in 1e-3 from the CPU.
    np.testing.assert_allclose(np.asarray(x1), expected, rtol=5e-3, atol=1e-3)


def test_gram_and_normal_forms_take_the_same_geodesic_step():
    def quadratic_residual(x, args, p):
        return A @ x + 0.1 * (A @ x) ** 2 - B

    steps = {}
    for form in ("gram", "normal"):
        solver = LevenbergMarquardt(
            quadratic_residual,
            linear_solver=QR(form=form),
            init_damping=1e-2,
            geodesic_acceleration=True,
            geodesic_acceptance_ratio=1.0,
        )
        x1, _, info = solver.update(X0, solver.init(X0))
        assert bool(info.accepted)
        steps[form] = np.asarray(x1)
    np.testing.assert_allclose(steps["gram"], steps["normal"], rtol=1e-3, atol=1e-5)


def test_gram_qr_reaches_below_the_float32_gram_cholesky_floor():
    # The dual Gram squares the singular values 0.65 ** k, so in float32 the
    # modes past k ~ 18 are rounding noise and the Cholesky step stalls; the
    # dual QR solves at cond([A, sqrt(damping) I]) and keeps them.
    def loss(linear_solver):
        solver = LevenbergMarquardt(
            residual, linear_solver=linear_solver, geodesic_acceleration=False
        )
        result = solver.solve(X0, max_steps=300)
        assert result.x.dtype == jnp.float32
        return float(result.info.loss)

    cholesky_loss = loss(Cholesky(form="gram"))
    qr_loss = loss(QR(form="gram"))
    assert qr_loss < 1e-3 * cholesky_loss
    assert qr_loss < 1e-7
