"""LevenbergMarquardt: the damped step, the metric's selection of the root,
and the implicit derivative of that selection.

Everything here checks against a closed form or an independently computed
reference. Loop mechanics shared with the ridge solver (callbacks, save_steps,
multi-start) are covered once, in test_ridge_solve_features.py and
test_multi_start.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    QR,
    SVD,
    Cholesky,
    CholeskyMetric,
    DiagonalMetric,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMStatus,
    RepeatedFactorMetric,
)

RNG = np.random.default_rng(11)
M, N = 4, 7  # underdetermined: which root comes back is a real choice
A_NP = RNG.normal(size=(M, N))
B_NP = RNG.normal(size=M)
W_ROOT = RNG.normal(size=(N, N))
W_NP = W_ROOT @ W_ROOT.T + N * np.eye(N)

A = jnp.asarray(A_NP, jnp.float32)
B = jnp.asarray(B_NP, jnp.float32)
L_W = jnp.asarray(np.linalg.cholesky(W_NP), jnp.float32)


def linear_residual(x):
    return A @ x - B


def nonlinear_residual(x, args, p):
    return jnp.concatenate([A @ x - p["scale"] * B, jnp.array([x[0] * x[1] - 0.3])])


def min_norm_solution(W, b=B_NP):
    """argmin ||x||_W subject to A x = b."""
    Winv = np.linalg.inv(W)
    return Winv @ A_NP.T @ np.linalg.solve(A_NP @ Winv @ A_NP.T, b)


def damped_step(W, x, damping):
    """The exact solution of min ||r + J s||^2 + damping ||s||_W^2."""
    r = A_NP @ x - B_NP
    return -np.linalg.solve(A_NP.T @ A_NP + damping * W, A_NP.T @ r)


FORWARD_SOLVERS = {
    "cholesky_auto": Cholesky(),
    "cholesky_gram": Cholesky(form="gram"),
    "cholesky_normal": Cholesky(form="normal"),
    "qr": QR(),
    "cg": CG(IdentityPreconditioner(), tol=1e-12, maxiter=400),
    "gram_cg": GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=400),
}


@pytest.mark.parametrize("name", list(FORWARD_SOLVERS))
def test_first_step_matches_the_closed_form_damped_solution(name):
    damping = 0.07
    metric = CholeskyMetric(L_W)
    solver = LevenbergMarquardt(
        linear_residual,
        metric=metric,
        linear_solver=FORWARD_SOLVERS[name],
        init_damping=damping,
        geodesic_acceleration=False,
    )
    x0 = jnp.asarray(RNG.normal(size=N), jnp.float32)
    x1, _, info = solver.update(x0, solver.init(x0))
    expected = np.asarray(x0, np.float64) + damped_step(
        W_NP, np.asarray(x0, np.float64), damping
    )
    assert bool(info.accepted)
    np.testing.assert_allclose(np.asarray(x1), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("name", list(FORWARD_SOLVERS))
def test_converges_to_the_minimum_metric_norm_root(name):
    # The damping -> 0 limit selects the minimum-W-norm interpolant, and the
    # identity metric selects a demonstrably different one.
    metric = CholeskyMetric(L_W)
    solver = LevenbergMarquardt(
        linear_residual,
        metric=metric,
        linear_solver=FORWARD_SOLVERS[name],
        min_damping=1e-12,
    )
    result = solver.solve(jnp.zeros(N), max_steps=200, atol=1e-6)
    assert int(result.status) == int(LMStatus.CONVERGED)
    expected = min_norm_solution(W_NP)
    np.testing.assert_allclose(np.asarray(result.x), expected, rtol=2e-3, atol=2e-4)
    # The selection is real: the Euclidean root differs well beyond tolerance.
    euclidean = min_norm_solution(np.eye(N))
    assert np.linalg.norm(expected - euclidean) > 0.1


def test_metric_forms_agree_at_positive_damping():
    # Push-through identity: the gram and normal factorizations of the same
    # damped subproblem give the same step.
    metric = CholeskyMetric(L_W)
    x0 = jnp.asarray(RNG.normal(size=N), jnp.float32)
    steps = {}
    for form in ("gram", "normal"):
        solver = LevenbergMarquardt(
            linear_residual,
            metric=metric,
            linear_solver=Cholesky(form=form),
            init_damping=0.3,
            geodesic_acceleration=False,
        )
        steps[form] = np.asarray(solver.update(x0, solver.init(x0))[0])
    np.testing.assert_allclose(steps["gram"], steps["normal"], rtol=1e-5, atol=1e-6)


def test_diagonal_and_dense_metrics_agree_on_the_same_geometry():
    weights = jnp.asarray(RNG.uniform(0.5, 3.0, size=N), jnp.float32)
    dense = CholeskyMetric(jnp.diag(jnp.sqrt(weights)))
    x0 = jnp.zeros(N)
    results = [
        LevenbergMarquardt(linear_residual, metric=metric, min_damping=1e-12).solve(
            x0, max_steps=200, atol=1e-6
        )
        for metric in (DiagonalMetric(weights), dense)
    ]
    np.testing.assert_allclose(
        np.asarray(results[0].x), np.asarray(results[1].x), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(results[0].x),
        min_norm_solution(np.diag(np.asarray(weights, np.float64))),
        rtol=2e-3,
        atol=2e-4,
    )


def test_repeated_factor_metric_matches_its_dense_block_diagonal():
    block, repeats = 3, 2
    root = RNG.normal(size=(block, block + 2))
    K = root @ root.T + 0.5 * np.eye(block)
    free = N - block * repeats
    dense = np.zeros((N, N))
    for j in range(repeats):
        dense[j * block : (j + 1) * block, j * block : (j + 1) * block] = K
    dense[block * repeats :, block * repeats :] = np.eye(free)

    repeated = RepeatedFactorMetric(
        jnp.asarray(np.linalg.cholesky(K).T, jnp.float32), repeats=repeats
    )
    solver = LevenbergMarquardt(linear_residual, metric=repeated, min_damping=1e-12)
    result = solver.solve(jnp.zeros(N), max_steps=200, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result.x), min_norm_solution(dense), rtol=3e-3, atol=3e-4
    )


def test_default_metric_is_euclidean():
    plain = LevenbergMarquardt(linear_residual, min_damping=1e-12).solve(
        jnp.zeros(N), max_steps=200, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(plain.x), min_norm_solution(np.eye(N)), rtol=2e-3, atol=2e-4
    )


def quadratic_residual(x):
    # r(x) = [x0^2 - 2, x0 x1 - 1]: a curved residual where the geodesic
    # correction has something to do.
    return jnp.array([x[0] ** 2 - 2.0, x[0] * x[1] - 1.0])


@pytest.mark.parametrize("name", list(FORWARD_SOLVERS))
def test_geodesic_correction_matches_its_closed_form(name):
    damping = 0.2
    solver = LevenbergMarquardt(
        quadratic_residual,
        linear_solver=FORWARD_SOLVERS[name],
        init_damping=damping,
        geodesic_acceptance_ratio=1e9,  # always accept, so the value is checked
    )
    x0 = jnp.asarray([1.6, 0.7], jnp.float32)
    x1, _, info = solver.update(x0, solver.init(x0))
    x = np.asarray(x0, np.float64)
    J = np.array([[2 * x[0], 0.0], [x[1], x[0]]])
    r = np.array([x[0] ** 2 - 2.0, x[0] * x[1] - 1.0])
    G = J.T @ J + damping * np.eye(2)
    velocity = -np.linalg.solve(G, J.T @ r)
    # Directional second derivative of r along the velocity.
    f_vv = np.array([2 * velocity[0] ** 2, 2 * velocity[0] * velocity[1]])
    acceleration = -np.linalg.solve(G, J.T @ f_vv)
    assert bool(info.used_geodesic)
    np.testing.assert_allclose(
        np.asarray(x1), x + velocity + 0.5 * acceleration, rtol=1e-4, atol=1e-5
    )


def test_rejected_step_leaves_x_and_reuses_the_cached_jacobian():
    calls = []

    def counting_residual(x):
        calls.append(None)
        return quadratic_residual(x)

    # A tiny max_damping forces the first step to overshoot and be rejected.
    solver = LevenbergMarquardt(
        counting_residual, init_damping=1e-8, geodesic_acceleration=False
    )
    x0 = jnp.asarray([8.0, 8.0], jnp.float32)
    state = solver.init(x0)
    x1, state1, info1 = solver.update(x0, state)
    if bool(info1.accepted):
        pytest.skip("fixture no longer produces a rejected first step")
    np.testing.assert_array_equal(np.asarray(x1), np.asarray(x0))
    assert bool(state1.jacobian_valid)
    before = len(calls)
    solver.update(x1, state1)
    # The reused cache costs one trial evaluation, not a fresh linearization.
    assert len(calls) - before <= 2


def test_solve_and_manual_update_loop_agree():
    solver = LevenbergMarquardt(linear_residual, min_damping=1e-12)
    x = jnp.zeros(N)
    state = solver.init(x)
    for _ in range(40):
        x, state, _ = solver.update(x, state)
    looped = LevenbergMarquardt(linear_residual, min_damping=1e-12).solve(
        jnp.zeros(N), max_steps=40, atol=0.0, gtol=0.0, xtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(looped.x), np.asarray(x), rtol=1e-5, atol=1e-6
    )


def test_pytree_x_and_args_round_trip():
    def residual(x, args, p):
        return jnp.concatenate(
            [A @ x["head"] - args["target"], x["tail"] - p["anchor"]]
        )

    x0 = {"head": jnp.zeros(N), "tail": jnp.zeros(2)}
    args = {"target": B}
    p = {"anchor": jnp.asarray([0.25, -0.5], jnp.float32)}
    result = LevenbergMarquardt(residual, min_damping=1e-12).solve(
        x0, args, p=p, max_steps=200, atol=1e-6
    )
    assert int(result.status) == int(LMStatus.CONVERGED)
    np.testing.assert_allclose(
        np.asarray(result.x["tail"]), np.asarray(p["anchor"]), rtol=1e-4, atol=1e-5
    )


# The undamped AD operator is singular on the side the problem is deficient
# in, so each Krylov rule is offered only where its operator is invertible:
# this fixture is underdetermined (m < n), which rules out plain CG.
AD_SOLVERS = {
    "default": None,
    "cholesky": Cholesky(),
    "svd": SVD(),
    "gram_cg": GramCG(IdentityPreconditioner(), tol=1e-12, maxiter=400),
}


@pytest.mark.parametrize("name", list(AD_SOLVERS))
def test_implicit_jvp_matches_the_analytic_min_norm_map(name):
    # x*(b) = W^-1 A' (A W^-1 A')^-1 b is linear in b, so the tangent is the
    # same map applied to b_dot.
    metric = CholeskyMetric(L_W)
    solver = LevenbergMarquardt(
        lambda x, args, p: A @ x - p["b"],
        metric=metric,
        ad_solver=AD_SOLVERS[name],
        min_damping=1e-12,
    )
    p = {"b": B}
    p_dot = {"b": jnp.asarray(RNG.normal(size=M), jnp.float32)}

    def run(p_value):
        return solver.solve(jnp.zeros(N), p=p_value, max_steps=200, atol=1e-6).x

    tangent = jax.jvp(run, (p,), (p_dot,))[1]
    expected = min_norm_solution(W_NP, np.asarray(p_dot["b"], np.float64))
    np.testing.assert_allclose(np.asarray(tangent), expected, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("name", list(AD_SOLVERS))
def test_implicit_vjp_is_the_transpose_of_the_jvp(name):
    solver = LevenbergMarquardt(
        lambda x, args, p: A @ x - p["b"],
        metric=CholeskyMetric(L_W),
        ad_solver=AD_SOLVERS[name],
        min_damping=1e-12,
    )
    p = {"b": B}
    p_dot = {"b": jnp.asarray(RNG.normal(size=M), jnp.float32)}
    cotangent = jnp.asarray(RNG.normal(size=N), jnp.float32)

    def run(p_value):
        return solver.solve(jnp.zeros(N), p=p_value, max_steps=200, atol=1e-6).x

    tangent = jax.jvp(run, (p,), (p_dot,))[1]
    gradient = jax.vjp(run, p)[1](cotangent)[0]["b"]
    np.testing.assert_allclose(
        float(cotangent @ tangent), float(gradient @ p_dot["b"]), rtol=2e-3, atol=1e-5
    )


def test_gram_cg_ad_rejects_the_overdetermined_shape():
    # The undamped dual is singular for m > n, where CG returns a wrong
    # tangent rather than failing, so the resolution rejects it outright.
    tall_A = jnp.asarray(RNG.normal(size=(9, 3)), jnp.float32)
    solver = LevenbergMarquardt(
        lambda x, args, p: tall_A @ x - p["b"],
        ad_solver=GramCG(IdentityPreconditioner(), maxiter=16),
    )
    p = {"b": jnp.asarray(RNG.normal(size=9), jnp.float32)}
    with pytest.raises(ValueError, match="m <= n"):
        jax.jvp(lambda pv: solver.solve(jnp.zeros(3), p=pv, max_steps=20).x, (p,), (p,))


def test_nonlinear_residual_solves_and_differentiates():
    solver = LevenbergMarquardt(nonlinear_residual, min_damping=1e-12)
    p = {"scale": jnp.asarray(1.0)}
    result = solver.solve(jnp.zeros(N), p=p, max_steps=300, atol=1e-5)
    assert int(result.status) == int(LMStatus.CONVERGED)
    np.testing.assert_allclose(
        np.asarray(nonlinear_residual(result.x, None, p)), 0.0, atol=1e-4
    )
    # Finite differences on a scalar summary of the solution.
    summary = lambda s: jnp.sum(  # noqa: E731
        solver.solve(jnp.zeros(N), p={"scale": s}, max_steps=300, atol=1e-5).x ** 2
    )
    analytic = float(jax.grad(summary)(jnp.asarray(1.0)))
    h = 1e-3
    numeric = float(
        (summary(jnp.asarray(1.0 + h)) - summary(jnp.asarray(1.0 - h))) / (2 * h)
    )
    np.testing.assert_allclose(analytic, numeric, rtol=2e-2, atol=2e-3)


def test_padded_zero_residual_matches_the_unpadded_solve():
    # The fixed-residual-shape pattern: identically-zero rows keep compiled
    # shapes stable and must not move the solution. The undamped dual is
    # singular there, which is what SVD() is for.
    pad = 3

    def padded(x, args, p):
        return jnp.concatenate([A @ x - p["b"], jnp.zeros(pad)])

    p = {"b": B}
    unpadded = LevenbergMarquardt(
        lambda x, args, p: A @ x - p["b"], min_damping=1e-12
    ).solve(jnp.zeros(N), p=p, max_steps=200, atol=1e-6)
    padded_result = LevenbergMarquardt(
        padded, ad_solver=SVD(), min_damping=1e-12
    ).solve(jnp.zeros(N), p=p, max_steps=200, atol=1e-6)
    # Two independently converged float32 solves, so the agreement is a
    # measured property rather than a tolerance bound.
    np.testing.assert_allclose(
        np.asarray(padded_result.x), np.asarray(unpadded.x), rtol=2e-3, atol=2e-4
    )


def test_has_aux_reports_pre_step_aux_and_a_final_value():
    def residual(x):
        return linear_residual(x), {"norm": jnp.sum(x**2)}

    solver = LevenbergMarquardt(residual, has_aux=True, min_damping=1e-12)
    x0 = jnp.ones(N)
    _, _, info = solver.update(x0, solver.init(x0))
    np.testing.assert_allclose(float(info.aux["norm"]), float(N), rtol=1e-6)
    result = solver.solve(x0, max_steps=200, atol=1e-6)
    np.testing.assert_allclose(
        float(result.aux["norm"]), float(jnp.sum(result.x**2)), rtol=1e-5
    )


def test_equal_configs_share_one_compilation():
    def build():
        return LevenbergMarquardt(
            linear_residual, linear_solver=Cholesky(), init_damping=1e-3
        )

    assert build() == build() and hash(build()) == hash(build())
    assert build() != LevenbergMarquardt(linear_residual, linear_solver=QR())
