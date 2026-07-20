"""Geometry-aware implicit solvers (issue #22) and the jacobian_mode assembly
geometry they rely on (issue #23).

The workhorse problem is a tall (m > n) linear least-squares residual
``A x - (b + p)`` with a nonzero residual at the solution, where the implicit
tangent has the closed form ``x_dot = (A'A)^{-1} A' p_dot``.
"""

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
    assert jnp.allclose(x, analytic_tangent(A_TALL, B_TALL + P0), atol=1e-3)
    assert jnp.allclose(x_dot, analytic_tangent(A_TALL, P_DOT), atol=1e-3)

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
    assert jnp.allclose(
        jnp.vdot(X_BAR, x_dot), jnp.vdot(p_bar, P_DOT), rtol=1e-3, atol=1e-4
    )


def test_metric_whitened_tall_tangent_is_metric_independent():
    # A nonidentity diagonal metric changes the whitened B = J S the primal
    # rule factors, but for a full-column-rank tall system the tangent is the
    # unique (J'J)^{-1} J' p_dot regardless of the metric.
    metric = metric_from_diagonal(jnp.asarray([0.5, 2.0, 4.0], dtype=jnp.float32))
    solver = tall_solver(metric=metric)
    assert solver._implicit_rule_at(X0, None, P0) == "primal_qr"
    _, x_dot = jax.jvp(solved_x_fn(solver), (P0,), (P_DOT,))
    assert jnp.allclose(x_dot, analytic_tangent(A_TALL, P_DOT), atol=1e-3)


# Rank-deficient tall system: parameter column 2 duplicates column 1, so
# B = J is column-rank-deficient. The disjoint one-hot columns are exact in
# float arithmetic, so the unregularized QR pivot is exactly zero and the
# implicit_penalty=0.0 failure is non-finite rather than merely huge.
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
            implicit_penalty=penalty,
        )

        def solved_x(p):
            return solver.solve(X0, p=p, max_steps=100, gtol=1e-5).x

        return jax.jvp(solved_x, (P0,), (P_DOT,))[1]

    regularized = tangent(1e-6)
    assert bool(jnp.all(jnp.isfinite(regularized)))
    # The trace-scaled ridge returns the minimum-norm tangent, which splits
    # evenly across the duplicated parameters.
    assert jnp.allclose(regularized[1], regularized[2], atol=1e-4)

    unregularized = tangent(0.0)
    assert not bool(jnp.all(jnp.isfinite(unregularized)))


A_FAT = jnp.asarray(_rng.normal(size=(2, 4)), dtype=jnp.float32)
P_FAT = jnp.asarray(_rng.normal(size=(2,)), dtype=jnp.float32)
P_FAT_DOT = jnp.asarray(_rng.normal(size=(2,)), dtype=jnp.float32)
X0_FAT = jnp.zeros(4, dtype=jnp.float32)


def fat_residual(x, args, p):
    return A_FAT @ x - p


def test_auto_implicit_solver_dispatch():
    cg_solver = LevenbergMarquardt(
        tall_residual,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
    )
    assert cg_solver._resolved_implicit_solver == "dual_cg"

    for linear_solver in ("qr", "augmented_qr", "lsmr"):
        whitened = LevenbergMarquardt(tall_residual, linear_solver=linear_solver)
        assert whitened._resolved_implicit_solver == "geometry"

    dense = LevenbergMarquardt(tall_residual, linear_solver="cholesky")
    assert dense._resolved_implicit_solver == "dual_cholesky"

    # The "geometry" selection resolves per problem shape at trace time.
    tall = tall_solver()
    assert tall._implicit_rule_at(X0, None, P0) == "primal_qr"
    fat = LevenbergMarquardt(
        fat_residual,
        linear_solver="augmented_qr",
        geodesic_acceleration=False,
        cache_jacobian=False,
    )
    assert fat._implicit_rule_at(X0_FAT, None, P_FAT) == "dual_cholesky"


def test_backward_compatible_implicit_solver_aliases():
    # "cholesky" is an alias of "dual_cholesky": identical construction and
    # bitwise-identical tangents.
    def dual_tangent(implicit_solver):
        solver = tall_solver(implicit_solver=implicit_solver)
        return jax.jvp(solved_x_fn(solver), (P0,), (P_DOT,))[1]

    alias = dual_tangent("cholesky")
    explicit = dual_tangent("dual_cholesky")
    assert bool(jnp.array_equal(alias, explicit))
    assert bool(jnp.all(jnp.isfinite(alias)))

    # "cg" still constructs and differentiates (on a fat system, where the
    # dual J J' is nonsingular); the tangent is the minimum-norm
    # A'(A A')^{-1} p_dot.
    cg_solver = LevenbergMarquardt(
        fat_residual,
        implicit_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        geodesic_acceleration=False,
    )

    def solved_fat_x(p):
        return cg_solver.solve(X0_FAT, p=p, max_steps=100, atol=1e-6).x

    _, x_dot = jax.jvp(solved_fat_x, (P_FAT,), (P_FAT_DOT,))
    expected = A_FAT.T @ jnp.linalg.solve(A_FAT @ A_FAT.T, P_FAT_DOT)
    assert jnp.allclose(x_dot, expected, atol=1e-3)


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


def test_near_zero_residual_dual_and_primal_tangents_agree():
    # The trace-scaled ridge means the same thing in both geometries; an
    # explicit 1e-4 penalty keeps the float32 factorization of the singular
    # dual (condition ~ 1/penalty) well above its noise floor while the
    # O(penalty * m) tangent bias stays below the analytic tolerance.
    def tangent(implicit_solver):
        solver = LevenbergMarquardt(
            stacked_residual,
            linear_solver="augmented_qr",
            geodesic_acceleration=False,
            cache_jacobian=False,
            implicit_solver=implicit_solver,
            implicit_penalty=1e-4,
        )

        def solved_x(p):
            return solver.solve(
                jnp.zeros(3, dtype=jnp.float32), p=p, max_steps=100, atol=1e-5
            ).x

        return jax.jvp(solved_x, (P_STACKED,), (P_STACKED_DOT,))[1]

    dual = tangent("dual_cholesky")
    primal = tangent("primal_qr")
    assert bool(jnp.all(jnp.isfinite(dual)))
    assert bool(jnp.all(jnp.isfinite(primal)))
    assert jnp.allclose(dual, primal, atol=1e-3)
    # Both agree with the analytic minimum-norm tangent of the stacked system,
    # x_dot = pinv([A; A]) p_dot = (2 A'A)^{-1} A'(p_dot_top + p_dot_bottom),
    # up to the O(penalty * m) ridge bias.
    expected = jnp.linalg.solve(
        2.0 * A_SQUARE.T @ A_SQUARE,
        A_SQUARE.T @ (P_STACKED_DOT[:3] + P_STACKED_DOT[3:]),
    )
    assert jnp.allclose(primal, expected, atol=5e-3)


def test_primal_cholesky_matches_primal_qr_on_tall_problem():
    def tangent(implicit_solver):
        solver = tall_solver(implicit_solver=implicit_solver)
        return jax.jvp(solved_x_fn(solver), (P0,), (P_DOT,))[1]

    qr_tangent = tangent("primal_qr")
    cholesky_tangent = tangent("primal_cholesky")
    assert jnp.allclose(qr_tangent, cholesky_tangent, atol=1e-4)
    assert jnp.allclose(qr_tangent, analytic_tangent(A_TALL, P_DOT), atol=1e-3)


def test_lsmr_forward_with_auto_implicit_matches_analytic_tangent():
    solver = LevenbergMarquardt(
        tall_residual,
        linear_solver="lsmr",
        iterative_maxiter=200,
        iterative_tol=1e-8,
        geodesic_acceleration=False,
        cache_jacobian=False,
    )
    assert solver._resolved_implicit_solver == "geometry"

    def solved_x(p):
        return solver.solve(X0, p=p, max_steps=100, gtol=1e-5).x

    x, x_dot = jax.jvp(solved_x, (P0,), (P_DOT,))
    assert jnp.allclose(x, analytic_tangent(A_TALL, B_TALL + P0), atol=1e-3)
    assert jnp.allclose(x_dot, analytic_tangent(A_TALL, P_DOT), atol=1e-3)
