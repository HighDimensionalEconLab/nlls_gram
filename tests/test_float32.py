"""float32 coverage: dtype purity and accuracy across the config matrix.

This module runs at JAX's default precision (x64 OFF), so every array is
float32 unless something promotes it. The x64-ENABLED-with-float32-inputs
case -- where Python scalars default to f64 and can silently promote the whole
compute -- is the more dangerous one and lives in test_float64_subprocess.py,
which needs a subprocess to set the flag before JAX initializes.

Nothing here promotes a solve to float64: nlls-gram has no ad/metric solve
dtype knob, so float32 in must mean float32 out, at float32 accuracy.

float32 is also where matmul precision bites: XLA:GPU serves float32
``dot_general`` from TF32 tensor cores by default, at a 10-bit mantissa. The
package pins every product it owns to HIGHEST, which the last test here
asserts on the jaxpr -- device-independently, so CPU CI protects the GPU path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    LU,
    QR,
    SVD,
    BlockEigenPreconditioner,
    Cholesky,
    CholeskyMetric,
    DiagonalMetric,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMState,
    LMStatus,
    NystromPreconditioner,
    PaddedPreconditioner,
    ShermanMorrisonPreconditioner,
    SolverContext,
    WoodburyPreconditioner,
)

RNG = np.random.default_rng(5)
M, N = 4, 7  # underdetermined: the package's target regime
A_NP = RNG.normal(size=(M, N))
B_NP = RNG.normal(size=M)
A = jnp.asarray(A_NP, jnp.float32)
B = jnp.asarray(B_NP, jnp.float32)
X0 = jnp.zeros(N, jnp.float32)

A_SQ_NP = RNG.normal(size=(N, N)) + N * np.eye(N)  # square and well conditioned
A_SQ = jnp.asarray(A_SQ_NP, jnp.float32)
P_SQ = jnp.asarray(RNG.normal(size=N), jnp.float32)
X0_SQ = jnp.zeros(N, jnp.float32)


def fat_residual(x, args, p):
    return A @ x - p


def square_residual(x, args, p):
    return A_SQ @ x - p


def min_norm(b):
    return A_NP.T @ np.linalg.solve(A_NP @ A_NP.T, np.asarray(b, np.float64))


FORWARD = {
    "cholesky_auto": Cholesky(),
    "cholesky_gram": Cholesky(form="gram"),
    "cholesky_normal": Cholesky(form="normal"),
    "qr": QR(),
    "cg": CG(IdentityPreconditioner(), tol=1e-7, maxiter=200),
    "gram_cg": GramCG(IdentityPreconditioner(), tol=1e-7, maxiter=200),
}


@pytest.mark.parametrize("name", list(FORWARD))
def test_float32_forward_solvers_stay_float32_and_hit_the_min_norm_root(name):
    solver = LevenbergMarquardt(fat_residual, linear_solver=FORWARD[name])
    result = solver.solve(X0, None, p=B, max_steps=200, atol=1e-5)
    assert int(result.status) == int(LMStatus.CONVERGED)
    assert result.x.dtype == jnp.float32
    assert result.lm_state.damping.dtype == jnp.float32
    assert result.info.loss.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(result.x), min_norm(B), rtol=2e-3, atol=2e-4)


AD_RULES_FAT = {
    "auto": None,  # -> SVD for a rectangular system
    "svd": SVD(),
    "cholesky": Cholesky(),
    "gram_cg": GramCG(IdentityPreconditioner(), tol=1e-7, maxiter=200),
}


@pytest.mark.parametrize("name", list(AD_RULES_FAT))
def test_float32_implicit_tangent_is_float32_and_matches_the_min_norm_map(name):
    # x*(b) is linear in b here, so the tangent is the same map applied to b_dot.
    solver = LevenbergMarquardt(fat_residual, ad_solver=AD_RULES_FAT[name])
    b_dot = jnp.asarray(RNG.normal(size=M), jnp.float32)

    def run(p):
        return solver.solve(X0, None, p=p, max_steps=200, atol=1e-6).x

    tangent = jax.jvp(run, (B,), (b_dot,))[1]
    (cotangent,) = jax.vjp(run, B)[1](jnp.ones(N, jnp.float32))

    assert tangent.dtype == jnp.float32
    assert cotangent.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(tangent)) and jnp.all(jnp.isfinite(cotangent))
    np.testing.assert_allclose(
        np.asarray(tangent), min_norm(b_dot), rtol=3e-3, atol=3e-4
    )


AD_RULES_SQUARE = {"auto": None, "lu": LU(), "svd": SVD()}


@pytest.mark.parametrize("name", list(AD_RULES_SQUARE))
def test_float32_square_tangent_is_float32_and_matches_the_direct_solve(name):
    solver = LevenbergMarquardt(square_residual, ad_solver=AD_RULES_SQUARE[name])
    direction = jnp.asarray(RNG.normal(size=N), jnp.float32)

    def run(p):
        return solver.solve(X0_SQ, None, p=p, max_steps=200, atol=1e-6).x

    tangent = jax.jvp(run, (P_SQ,), (direction,))[1]
    expected = np.linalg.solve(A_SQ_NP, np.asarray(direction, np.float64))
    assert tangent.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(tangent), expected, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize(
    "metric",
    [
        None,
        DiagonalMetric(jnp.linspace(0.5, 2.0, N, dtype=jnp.float32)),
        CholeskyMetric(jnp.eye(N, dtype=jnp.float32) * 1.5),
    ],
    ids=["euclidean", "diagonal", "cholesky"],
)
def test_float32_metrics_keep_the_solve_and_its_tangent_float32(metric):
    solver = LevenbergMarquardt(fat_residual, metric=metric)

    def run(p):
        return solver.solve(X0, None, p=p, max_steps=200, atol=1e-5).x

    x = run(B)
    tangent = jax.jvp(run, (B,), (jnp.ones(M, jnp.float32),))[1]
    assert x.dtype == jnp.float32
    assert tangent.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(tangent))


# --- preconditioners -------------------------------------------------------
# Nystrom, Woodbury, Sherman-Morrison and Padded had no coverage at all; these
# check the apply contract in float32 against a dense reference.

DUAL = A_NP @ A_NP.T + 0.5 * np.eye(M)  # an SPD m x m dual operator


def dual_solve(v):
    """Static hashable A^-1 for the low-rank-update preconditioners."""
    return jnp.linalg.solve(jnp.asarray(DUAL, jnp.float32), v)


def dual_matvec(X):
    return jnp.asarray(DUAL, jnp.float32) @ X


def preconditioner_cases():
    u = jnp.asarray(RNG.normal(size=M), jnp.float32)
    U = jnp.asarray(RNG.normal(size=(M, 2)), jnp.float32)
    weights = jnp.asarray([0.4, 0.7], jnp.float32)
    families = [(jnp.asarray(np.eye(2)[None] * 2.0, jnp.float32), 1.0)]
    return {
        "identity": (IdentityPreconditioner(), M),
        "sherman_morrison": (
            ShermanMorrisonPreconditioner(dual_solve, u, jnp.asarray(0.3, jnp.float32)),
            M,
        ),
        "woodbury": (WoodburyPreconditioner(dual_solve, U, weights), M),
        "nystrom": (
            NystromPreconditioner(
                dual_matvec, n=M, rank=2, key=jax.random.key(0), dtype=jnp.float32
            ),
            M,
        ),
        "block_eigen": (
            BlockEigenPreconditioner(families, jnp.arange(2)),
            2,
        ),
        "padded": (PaddedPreconditioner(IdentityPreconditioner(), n_real=M - 1), M),
    }


@pytest.mark.parametrize("name", list(preconditioner_cases()))
def test_float32_preconditioner_apply_is_float32_finite_and_spd(name):
    preconditioner, size = preconditioner_cases()[name]
    damping = jnp.asarray(0.1, jnp.float32)
    # BlockEigen reads the carried ridge, so the state must be real; the metric
    # solver carries ridge=None, which is the LevenbergMarquardt case.
    ctx = SolverContext(x=None, lm_state=LMState(damping=damping), args=None, p=None)
    v = jnp.asarray(RNG.normal(size=size), jnp.float32)

    out = preconditioner.apply(v, damping, ctx)
    assert out.dtype == jnp.float32, out.dtype
    assert jnp.all(jnp.isfinite(out))

    # SPD: v'M v > 0, and the map is symmetric on a random pair.
    assert float(jnp.dot(v, out)) > 0.0
    w = jnp.asarray(RNG.normal(size=size), jnp.float32)
    left = float(jnp.dot(w, preconditioner.apply(v, damping, ctx)))
    right = float(jnp.dot(v, preconditioner.apply(w, damping, ctx)))
    np.testing.assert_allclose(left, right, rtol=2e-3, atol=2e-5)


DUAL_PRECONDITIONERS = ["identity", "sherman_morrison", "woodbury", "nystrom"]


@pytest.mark.parametrize("name", DUAL_PRECONDITIONERS)
def test_float32_gram_cg_converges_under_each_dual_preconditioner(name):
    # A preconditioner may only change the CG path, never the converged step.
    preconditioner, _ = preconditioner_cases()[name]
    solver = LevenbergMarquardt(
        fat_residual,
        linear_solver=GramCG(preconditioner, tol=1e-7, maxiter=200),
        ad_solver=SVD(),
    )
    result = solver.solve(X0, None, p=B, max_steps=200, atol=1e-5)
    assert int(result.status) == int(LMStatus.CONVERGED)
    assert result.x.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(result.x), min_norm(B), rtol=3e-3, atol=3e-4)


def test_internal_matmuls_are_pinned_to_highest_precision():
    # TF32 tensor cores serve float32 dot_general on Ampere by default, at a
    # 10-bit mantissa (~1e-3). Forming a Gram matrix already squares the
    # condition number, so every product the package owns is pinned. The
    # residual below has no matmul of its own, so every dot_general in the
    # trace belongs to the solver. Asserted on the jaxpr rather than on
    # numbers: that holds on CPU, where the setting is a no-op, and so guards
    # the GPU path from CI. tests/conftest.py sets the global default too, but
    # only for the tests' own reference matmuls -- this must pass without it.
    def elementwise_residual(x, args, p):
        return x - p

    solver = LevenbergMarquardt(
        elementwise_residual, cache_jacobian=False, geodesic_acceleration=False
    )
    p0 = jnp.linspace(0.5, 1.5, N, dtype=jnp.float32)

    def run(p):
        return solver.solve(jnp.zeros(N, jnp.float32), None, p=p, max_steps=4).x

    stack = [jax.make_jaxpr(run)(p0).jaxpr]
    dots = []
    while stack:
        current = stack.pop()
        for equation in current.eqns:
            if equation.primitive.name == "dot_general":
                dots.append(equation)
            for value in equation.params.values():
                if hasattr(value, "eqns"):
                    stack.append(value)
                elif hasattr(value, "jaxpr"):
                    inner = value.jaxpr
                    stack.append(inner.jaxpr if hasattr(inner, "jaxpr") else inner)

    assert dots, "no dot_general in the trace; the assertion below is vacuous"
    unpinned = [
        equation
        for equation in dots
        if equation.params.get("precision")
        != (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    ]
    assert not unpinned, "\n".join(str(equation)[:110] for equation in unpinned[:5])
