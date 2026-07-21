# The dense AD rule on tall systems (issue #22) and the jacobian_mode
# assembly geometry it relies on (issue #23). The workhorse is a
# tall (m > n) linear least-squares residual A x - (b + p) with a nonzero
# residual at the solution, where the Gauss-Newton implicit tangent has the
# closed form x_dot = (A'A)^{-1} A' p_dot.

import jax
import jax.numpy as jnp
import numpy as np

from nlls_gram import (
    LevenbergMarquardt,
    LMStatus,
    identity_preconditioner,
    metric_from_diagonal,
)

M, N = 7, 3
_rng = np.random.default_rng(1234)
A_TALL = jnp.asarray(_rng.normal(size=(M, N)), dtype=jnp.float32)
B_TALL = jnp.asarray(_rng.normal(size=(M,)), dtype=jnp.float32)
P0 = jnp.asarray(_rng.normal(size=(M,)), dtype=jnp.float32)
P_DOT = jnp.asarray(_rng.normal(size=(M,)), dtype=jnp.float32)
X_BAR = jnp.asarray(_rng.normal(size=(N,)), dtype=jnp.float32)
X0 = jnp.zeros(N, dtype=jnp.float32)


def tall_residual(x, args, p):
    return A_TALL @ x - (B_TALL + p)


def analytic_tangent(A, p_dot):
    # x*(p) = argmin ||A x - (b + p)||^2 = (A'A)^{-1} A'(b + p), so
    # dx*/dp p_dot = (A'A)^{-1} A' p_dot -- metric-independent for a
    # full-column-rank tall system.
    return jnp.linalg.solve(A.T @ A, A.T @ p_dot)


def tall_solver(**overrides):
    settings = dict(
        linear_solver="augmented_qr",
        geodesic_acceleration=False,
        cache_jacobian=False,
    )
    settings.update(overrides)
    return LevenbergMarquardt(tall_residual, **settings)


def solved_x_fn(solver):
    def solved_x(p):
        return solver.solve(X0, p=p, max_steps=100, gtol=1e-5).x

    return solved_x


def test_tall_augmented_qr_jvp_matches_analytic_and_finite_differences():
    solver = tall_solver()
    result = solver.solve(X0, p=P0, max_steps=100, gtol=1e-5)
    assert int(result.status) == LMStatus.CONVERGED
    # The system is inconsistent: the residual at the solution is nonzero, so
    # this exercises the nonzero-residual Gauss-Newton implicit contract.
    assert float(result.info.loss) > 1e-2

    solved_x = solved_x_fn(solver)
    x, x_dot = jax.jvp(solved_x, (P0,), (P_DOT,))
    # Measured 1.8e-7 (x) and 2.2e-7 (x_dot) float32 under the SVD/QR rule.
    assert jnp.allclose(x, analytic_tangent(A_TALL, B_TALL + P0), atol=1e-4)
    assert jnp.allclose(x_dot, analytic_tangent(A_TALL, P_DOT), atol=2e-6)

    # Central finite differences: the problem is linear in p, so a large step
    # has zero truncation error and only solver noise divided by 2 * eps.
    eps = 0.5
    finite_difference = (solved_x(P0 + eps * P_DOT) - solved_x(P0 - eps * P_DOT)) / (
        2.0 * eps
    )
    assert jnp.allclose(x_dot, finite_difference, atol=1e-3)


def test_tall_vjp_jvp_dot_product_transpose_identity():
    solver = tall_solver()
    solved_x = solved_x_fn(solver)
    _, x_dot = jax.jvp(solved_x, (P0,), (P_DOT,))
    _, pullback = jax.vjp(solved_x, P0)
    (p_bar,) = pullback(X_BAR)
    # Measured 3.7e-6 relative float32.
    assert jnp.allclose(
        jnp.vdot(X_BAR, x_dot), jnp.vdot(p_bar, P_DOT), rtol=5e-5, atol=1e-6
    )


def test_metric_whitened_tall_tangent_is_metric_independent():
    # A nonidentity diagonal metric changes the whitened B = J S the normal
    # form factors, but for a full-column-rank tall system the tangent is the
    # unique (J'J)^{-1} J' p_dot regardless of the metric.
    metric = metric_from_diagonal(jnp.asarray([0.5, 2.0, 4.0], dtype=jnp.float32))
    solver = tall_solver(metric=metric)
    assert solver._resolved_ad_solver == "dense"
    _, x_dot = jax.jvp(solved_x_fn(solver), (P0,), (P_DOT,))
    # Measured 6e-8 float32.
    assert jnp.allclose(x_dot, analytic_tangent(A_TALL, P_DOT), atol=1e-6)


# Rank-deficient tall system: parameter column 2 duplicates column 1, so
# B = J is column-rank-deficient. The disjoint one-hot columns are exact in
# float arithmetic, so the undamped normal matrix is exactly singular and the
# ad_solver_penalty=0.0 rank guard poisons the tangent to NaN.
_A_RANK_DEFICIENT = jnp.zeros((M, N), dtype=jnp.float32)
_A_RANK_DEFICIENT = _A_RANK_DEFICIENT.at[0, 0].set(2.0)
_A_RANK_DEFICIENT = _A_RANK_DEFICIENT.at[1, 1].set(3.0)
_A_RANK_DEFICIENT = _A_RANK_DEFICIENT.at[1, 2].set(3.0)


def rank_deficient_residual(x, args, p):
    return _A_RANK_DEFICIENT @ x - (B_TALL + p)


def test_rank_deficient_tall_penalty_regularizes_and_zero_penalty_fails_loudly():
    def tangent(penalty):
        solver = LevenbergMarquardt(
            rank_deficient_residual,
            linear_solver="augmented_qr",
            geodesic_acceleration=False,
            cache_jacobian=False,
            ad_solver_penalty=penalty,
        )

        def solved_x(p):
            return solver.solve(X0, p=p, max_steps=100, gtol=1e-5).x

        return jax.jvp(solved_x, (P0,), (P_DOT,))[1]

    # Default None: the spectral-filter pseudoinverse returns the exact
    # minimum-norm tangent, which splits evenly across the duplicated
    # parameters -- no ridge, no bias.
    filtered = tangent(None)
    assert bool(jnp.all(jnp.isfinite(filtered)))
    assert jnp.allclose(filtered[1], filtered[2], atol=1e-5)

    # The opt-in ridge solves the augmented QR [B; sqrt(penalty * trace) I];
    # check it against the float64 analytic ridged solution. Resolving an
    # EXACTLY duplicated column pair against the ridge pivot is a cancellation
    # at eps32 * sigma_max^2 / (penalty * trace) regardless of factorization
    # (measured: 5.6e-5 relative at penalty 1e-4, 1.3e-3 at 1e-5, 4.8e-2 at
    # 1e-6), so this fixture floors near penalty 1e-5 in float32; the
    # cond(B)-not-cond(B)^2 accuracy of the path itself is pinned on a
    # generic fixture by test_ridged_dense_tangent_accuracy_scales_with_cond_b.
    A64 = np.asarray(_A_RANK_DEFICIENT, np.float64)
    G64 = A64.T @ A64
    p_dot64 = np.asarray(P_DOT, np.float64)

    def analytic_ridged(penalty):
        return np.linalg.solve(
            G64 + penalty * np.trace(G64) * np.eye(N), A64.T @ p_dot64
        )

    for penalty, tolerance in ((1e-4, 3e-4), (1e-5, 5e-3)):
        regularized = tangent(penalty)
        assert bool(jnp.all(jnp.isfinite(regularized)))
        reference = analytic_ridged(penalty)
        relative = np.linalg.norm(
            np.asarray(regularized, np.float64) - reference
        ) / np.linalg.norm(reference)
        assert relative < tolerance, (penalty, relative)

    unregularized = tangent(0.0)
    assert not bool(jnp.all(jnp.isfinite(unregularized)))


def test_ridged_dense_tangent_accuracy_scales_with_cond_b():
    # Regression for the cond(B)-not-cond(B)^2 contract of the ridged dense
    # rule: at cond(B) ~ 1e3 and penalty 1e-6 a normal-equations Cholesky
    # loses eps32 * cond^2 ~ 0.1 of the tangent, while the small-side
    # augmented QR [B; sqrt(penalty * trace) I] keeps the float32 error at
    # the eps32 * cond level -- measured 2.2e-6 relative against the float64
    # analytic ridge.
    m, n, penalty = 12, 4, 1e-6
    U, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (m, n)))
    V, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(1), (n, n)))
    A = ((U * jnp.logspace(0.0, -3.0, n)) @ V.T).astype(jnp.float32)
    b = jax.random.normal(jax.random.key(2), (m,), dtype=jnp.float32)
    p_dot = jax.random.normal(jax.random.key(3), (m,), dtype=jnp.float32)

    def residual(x, args, p):
        return A @ x - (b + p)

    solver = LevenbergMarquardt(
        residual,
        linear_solver="augmented_qr",
        geodesic_acceleration=False,
        cache_jacobian=False,
        ad_solver_penalty=penalty,
    )
    x_dot = jax.jvp(
        lambda p: (
            solver.solve(jnp.zeros(n, jnp.float32), p=p, max_steps=100, gtol=1e-5).x
        ),
        (jnp.zeros(m, jnp.float32),),
        (p_dot,),
    )[1]

    A64 = np.asarray(A, np.float64)
    G64 = A64.T @ A64
    expected = np.linalg.solve(
        G64 + penalty * np.trace(G64) * np.eye(n),
        A64.T @ np.asarray(p_dot, np.float64),
    )
    relative = np.linalg.norm(
        np.asarray(x_dot, np.float64) - expected
    ) / np.linalg.norm(expected)
    assert relative < 2e-5, relative


A_FAT = jnp.asarray(_rng.normal(size=(2, 4)), dtype=jnp.float32)
P_FAT = jnp.asarray(_rng.normal(size=(2,)), dtype=jnp.float32)
P_FAT_DOT = jnp.asarray(_rng.normal(size=(2,)), dtype=jnp.float32)
X0_FAT = jnp.zeros(4, dtype=jnp.float32)


def fat_residual(x, args, p):
    return A_FAT @ x - p


def test_auto_ad_solver_dispatch():
    cg_solver = LevenbergMarquardt(
        tall_residual,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        ad_solver_preconditioner=identity_preconditioner(),
    )
    assert cg_solver._resolved_ad_solver == "gram_cg"

    # Every non-cg forward uses the one shape-independent dense rule.
    for linear_solver in ("qr", "augmented_qr", "lsmr"):
        whitened = LevenbergMarquardt(tall_residual, linear_solver=linear_solver)
        assert whitened._resolved_ad_solver == "dense"

    for form, resolved in (
        ("gram_cholesky", "dense"),
        ("normal_cholesky", "dense"),
        ("normal_cg", "normal_cg"),
    ):
        kwargs = (
            {"normal_preconditioner": identity_preconditioner()}
            if form == "normal_cg"
            else {}
        )
        follows = LevenbergMarquardt(tall_residual, linear_solver=form, **kwargs)
        assert follows._resolved_ad_solver == resolved


def _iter_jaxprs(jaxpr):
    yield jaxpr
    for eqn in jaxpr.eqns:
        for value in eqn.params.values():
            yield from _iter_jaxprs_in_param(value)


def _iter_jaxprs_in_param(value):
    if hasattr(value, "jaxpr"):  # ClosedJaxpr
        yield from _iter_jaxprs(value.jaxpr)
    elif hasattr(value, "eqns"):  # raw Jaxpr
        yield from _iter_jaxprs(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_jaxprs_in_param(item)


def test_tall_jvp_jaxpr_materializes_no_m_by_m_array():
    # Issue #23's memory contract: differentiating the tall solve must never
    # build an (m, m) residual-space array -- neither the dense-assembly
    # identity basis nor a dual Gram matrix -- anywhere in the program.
    solver = tall_solver()
    solved_x = solved_x_fn(solver)
    closed = jax.make_jaxpr(lambda p: jax.jvp(solved_x, (p,), (P_DOT,)))(P0)
    offenders = []
    for jaxpr in _iter_jaxprs(closed.jaxpr):
        variables = list(jaxpr.invars) + list(jaxpr.constvars) + list(jaxpr.outvars)
        for eqn in jaxpr.eqns:
            variables.extend(eqn.invars)
            variables.extend(eqn.outvars)
        for var in variables:
            aval = getattr(var, "aval", None)
            if aval is not None and tuple(getattr(aval, "shape", ())) == (M, M):
                offenders.append(str(var))
    assert not offenders, f"(m, m) arrays found in the jvp jaxpr: {offenders}"


# Consistent tall system: a full-rank square system stacked twice, so m = 2n
# but the residual at the solution is ~0 and the dual J J' is rank n.
A_SQUARE = jnp.asarray(np.eye(3) + 0.3 * _rng.normal(size=(3, 3)), dtype=jnp.float32)
X_TRUE = jnp.asarray(_rng.normal(size=(3,)), dtype=jnp.float32)
P_STACKED = jnp.concatenate([A_SQUARE @ X_TRUE, A_SQUARE @ X_TRUE])
P_STACKED_DOT = jnp.asarray(_rng.normal(size=(6,)), dtype=jnp.float32)


def stacked_residual(x, args, p):
    top = A_SQUARE @ x
    return jnp.concatenate([top, top]) - p


def test_stacked_residual_ridged_dense_tangent_matches_analytic():
    # An explicit ad_solver_penalty is the trace-scaled ridge on B'B
    # (trace(J P J') = trace(B'B), so the ridge means the same thing in
    # either push-through composition of the dense rule); the O(penalty * m)
    # bias stays below the analytic tolerance.
    solver = LevenbergMarquardt(
        stacked_residual,
        linear_solver="augmented_qr",
        geodesic_acceleration=False,
        cache_jacobian=False,
        ad_solver_penalty=1e-4,
    )

    def solved_x(p):
        return solver.solve(
            jnp.zeros(3, dtype=jnp.float32), p=p, max_steps=100, atol=1e-5
        ).x

    ridged = jax.jvp(solved_x, (P_STACKED,), (P_STACKED_DOT,))[1]
    assert bool(jnp.all(jnp.isfinite(ridged)))
    # Agrees with the analytic minimum-norm tangent of the stacked system,
    # x_dot = pinv([A; A]) p_dot = (2 A'A)^{-1} A'(p_dot_top + p_dot_bottom),
    # up to the O(penalty * m) ridge bias.
    expected = jnp.linalg.solve(
        2.0 * A_SQUARE.T @ A_SQUARE,
        A_SQUARE.T @ (P_STACKED_DOT[:3] + P_STACKED_DOT[3:]),
    )
    assert jnp.allclose(ridged, expected, atol=5e-3)


def test_dense_default_tangent_on_inconsistent_tall_matches_analytic():
    # -S (B'B)^+ B' (r_p p_dot) = -S B^+ (r_p p_dot): the default
    # spectral-filter pseudoinverse computes the Gauss-Newton tangent even
    # when the residual at the solution is nonzero (the rhs B'(r_p p_dot)
    # always lies in range(B'B), so the filtered solve is exact).
    solver = tall_solver(ad_solver="dense")
    dense_tangent = jax.jvp(solved_x_fn(solver), (P0,), (P_DOT,))[1]
    # Measured 2.2e-7 float32.
    assert jnp.allclose(dense_tangent, analytic_tangent(A_TALL, P_DOT), atol=2e-6)


def test_lsmr_forward_with_auto_implicit_matches_analytic_tangent():
    solver = LevenbergMarquardt(
        tall_residual,
        linear_solver="lsmr",
        iterative_maxiter=200,
        iterative_tol=1e-8,
        geodesic_acceleration=False,
        cache_jacobian=False,
    )
    assert solver._resolved_ad_solver == "dense"

    def solved_x(p):
        return solver.solve(X0, p=p, max_steps=100, gtol=1e-5).x

    x, x_dot = jax.jvp(solved_x, (P0,), (P_DOT,))
    # Measured 3e-7 (x) and 2.2e-7 (x_dot) float32.
    assert jnp.allclose(x, analytic_tangent(A_TALL, B_TALL + P0), atol=1e-4)
    assert jnp.allclose(x_dot, analytic_tangent(A_TALL, P_DOT), atol=2e-6)
