import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    LSMR,
    MultiStart,
    NormalCG,
    RidgeLevenbergMarquardt,
    identity_penalty,
    identity_preconditioner,
    identity_right_preconditioner,
    repeated_dense_penalty,
)

RNG = np.random.default_rng(31)
M_RESID, BLOCK, REPEATS, PAD = 4, 3, 2, 1
P_DIM = REPEATS * BLOCK + PAD
A_NP = RNG.normal(size=(M_RESID, P_DIM))
ROOT = RNG.normal(size=(BLOCK, BLOCK + 2))
K_NP = ROOT @ ROOT.T + 0.5 * np.eye(BLOCK)

A = jnp.asarray(A_NP, dtype=jnp.float32)
K = jnp.asarray(K_NP, dtype=jnp.float32)
WEIGHTS = jnp.asarray(RNG.normal(size=P_DIM), dtype=jnp.float32)


def linear_residual(theta, args, p):
    return A @ theta - p


def make_penalty():
    return repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)


def solution_functional(solver, p):
    result = solver.solve(
        jnp.zeros(P_DIM), p=p, max_steps=300, gtol=1e-5
    )
    return jnp.vdot(WEIGHTS, result.x)


def central_difference(fn, p, step):
    grads = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        bump = np.zeros(p.shape[0], dtype=np.float32)
        bump[i] = step
        grads[i] = (
            float(fn(p + jnp.asarray(bump))) - float(fn(p - jnp.asarray(bump)))
        ) / (2 * step)
    return grads


def test_implicit_gradient_matches_finite_differences_linear():
    # Linear residual: J is p-independent and the residual Hessian vanishes,
    # so the GN implicit rule is EXACT at every ridge -- tight comparison.
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )
    p0 = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)

    def loss(p):
        return solution_functional(solver, p)

    grad = np.asarray(jax.grad(loss)(p0))
    fd = central_difference(loss, p0, 3e-2)
    np.testing.assert_allclose(grad, fd, rtol=2e-2, atol=2e-3)


def nonlinear_residual(theta, args, p):
    interactions = jnp.array(
        [
            theta[0] * theta[1] - p[0],
            theta[2] + theta[3] ** 2 - p[1],
            jnp.tanh(theta[4]) - 0.1 * p[2],
            theta[5] - p[0] * theta[6],
        ]
    )
    return interactions


def test_implicit_gradient_nonlinear_tight_at_small_ridge_biased_at_moderate():
    p0 = jnp.asarray([0.7, 0.9, 0.5], dtype=jnp.float32)
    x0 = 0.5 * jnp.ones(P_DIM)
    errors = {}
    for ridge in (1e-4, 1e-1):
        solver = RidgeLevenbergMarquardt(
            nonlinear_residual, penalty=identity_penalty(P_DIM), ridge=ridge
        )

        def loss(p, solver=solver):
            result = solver.solve(x0, p=p, max_steps=300, gtol=1e-5)
            return jnp.vdot(WEIGHTS, result.x)

        grad = np.asarray(jax.grad(loss)(p0))
        fd = central_difference(loss, p0, 3e-2)
        scale = np.linalg.norm(fd) + 1e-8
        errors[ridge] = np.linalg.norm(grad - fd) / scale
    # Documented GN contract, not equality: for a CURVED residual on an
    # underdetermined system the exact tangent's null-space block carries the
    # constraint-curvature term sum_i nu_i d2r_i (nu = r/ridge, which does not
    # vanish relative to ridge*M0 as ridge -> 0), and the GN rule drops it.
    # The functional reads null-space components, so a loose bound is the
    # honest assertion; exactness is pinned by the linear-residual test above,
    # where both dropped terms are identically zero.
    assert errors[1e-4] < 0.3
    assert errors[1e-1] < 0.5


def test_cg_and_cholesky_ad_agree():
    p0 = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)
    grads = {}
    for name, ad_solver in (
        ("cholesky", None),
        ("normal_cg", NormalCG(maxiter=200, tol=1e-8)),
    ):
        settings = {} if ad_solver is None else {"ad_solver": ad_solver}
        solver = RidgeLevenbergMarquardt(
            linear_residual,
            penalty=make_penalty(),
            ridge=1e-3,
            **settings,
        )

        def loss(p, solver=solver):
            return solution_functional(solver, p)

        grads[name] = np.asarray(jax.grad(loss)(p0))
    np.testing.assert_allclose(
        grads["cholesky"], grads["normal_cg"], rtol=1e-3, atol=1e-5
    )


def test_lsmr_forward_auto_resolves_to_matrix_free_ad():
    p0 = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)
    lsmr_solver = RidgeLevenbergMarquardt(
        linear_residual,
        penalty=make_penalty(),
        ridge=1e-3,
        linear_solver=LSMR(
            identity_right_preconditioner(), tol=1e-10, maxiter=None
        ),
    )
    assert lsmr_solver._resolved_ad_solver() == "normal_cg"
    dense_solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )
    assert dense_solver._resolved_ad_solver() == "cholesky"

    def lsmr_loss(p):
        return solution_functional(lsmr_solver, p)

    def dense_loss(p):
        return solution_functional(dense_solver, p)

    np.testing.assert_allclose(
        np.asarray(jax.grad(lsmr_loss)(p0)),
        np.asarray(jax.grad(dense_loss)(p0)),
        rtol=1e-2,
        atol=1e-4,
    )


def test_normal_cg_forward_auto_resolves_to_matrix_free_ad():
    p0 = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)
    cg_solver = RidgeLevenbergMarquardt(
        linear_residual,
        penalty=make_penalty(),
        ridge=1e-3,
        linear_solver=NormalCG(
            identity_preconditioner(), tol=1e-10, maxiter=None
        ),
    )
    assert cg_solver._resolved_ad_solver() == "normal_cg"
    dense_solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )

    def cg_loss(p):
        return solution_functional(cg_solver, p)

    def dense_loss(p):
        return solution_functional(dense_solver, p)

    np.testing.assert_allclose(
        np.asarray(jax.grad(cg_loss)(p0)),
        np.asarray(jax.grad(dense_loss)(p0)),
        rtol=1e-2,
        atol=1e-4,
    )


def test_failed_status_returns_zero_tangents():
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )
    p0 = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)

    def loss(p):
        result = solver.solve(
            jnp.zeros(P_DIM),
            p=p,
            max_steps=1,
            max_steps_is_success=False,
            gtol=1e-12,
        )
        return jnp.vdot(WEIGHTS, result.x)

    grad = np.asarray(jax.grad(loss)(p0))
    np.testing.assert_array_equal(grad, np.zeros(M_RESID))


def draw_perturbed(key, x, args):
    return x + 0.1 * jax.random.normal(key, x.shape, dtype=x.dtype), args


def test_multi_start_gradient_flows_through_the_winner():
    solver = RidgeLevenbergMarquardt(
        linear_residual, penalty=make_penalty(), ridge=1e-3
    )
    p0 = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)
    ms = MultiStart(key=jax.random.key(2), num_starts=3, draw=draw_perturbed)

    def multi_loss(p):
        result = solver.solve(
            jnp.zeros(P_DIM), p=p, max_steps=300, gtol=1e-5, multi_start=ms
        )
        return jnp.vdot(WEIGHTS, result.x)

    def plain_loss(p):
        return solution_functional(solver, p)

    np.testing.assert_allclose(
        np.asarray(jax.grad(multi_loss)(p0)),
        np.asarray(jax.grad(plain_loss)(p0)),
        rtol=1e-3,
        atol=1e-5,
    )


def test_ad_maxiter_required_when_cg_tolerances_zero():
    # An uncapped zero-tolerance CG loop has no stopping rule; the config
    # rejects it at construction rather than at differentiation time.
    with pytest.raises(ValueError, match="maxiter"):
        NormalCG(tol=0.0)
    NormalCG(tol=0.0, maxiter=50)
    NormalCG(tol=0.0, atol=1e-10)
