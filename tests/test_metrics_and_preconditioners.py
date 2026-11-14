import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

import nlls_gram
from nlls_gram import (
    CG,
    QR,
    SVD,
    Cholesky,
    DiagonalPreconditioner,
    DiagonalRidge,
    FactorRidge,
    GramCG,
    IdentityPreconditioner,
    IdentityRidge,
    LevenbergMarquardt,
    MatrixRidge,
    Newton,
    Preconditioner,
    RepeatedFactorRidge,
    RidgeMetric,
    SolverContext,
    SolverStatus,
    register_pytree_dataclass,
)

CTX = SolverContext()


@dataclasses.dataclass(frozen=True, eq=False)
class MatvecOnlyRidge(RidgeMetric):
    weights: jax.Array

    def matvec(self, v, ctx):
        w = jnp.asarray(self.weights, dtype=v.dtype)
        shape = (w.shape[0],) + (1,) * (v.ndim - 1)
        return v * w.reshape(shape)


register_pytree_dataclass(MatvecOnlyRidge, data_fields=("weights",))


def test_removed_legacy_exports_absent():
    removed = (
        "Metric",
        "IdentityMetric",
        "DiagonalMetric",
        "CholeskyMetric",
        "RepeatedFactorMetric",
        "StateSpaceMetric",
        "BlockEigenPreconditioner",
        "NystromPreconditioner",
        "ShermanMorrisonPreconditioner",
        "WoodburyPreconditioner",
        "PaddedPreconditioner",
        "Equality",
    )
    for name in removed:
        assert not hasattr(nlls_gram, name), f"{name} must not be exported"


def test_ridge_operators_quadratic_forms_and_matrix_vs_factor_distinction():
    rng = np.random.default_rng(7)
    n = 8
    v = jnp.asarray(rng.normal(size=n), jnp.float32)
    mat = jnp.asarray(rng.normal(size=(n, 3)), jnp.float32)

    # IdentityRidge full and leading slice
    id_full = IdentityRidge()
    id_part = IdentityRidge(5)
    np.testing.assert_allclose(id_full.matvec(v, CTX), v)
    np.testing.assert_allclose(id_full.quad_form(v, CTX), jnp.sum(v**2), rtol=1e-6)
    np.testing.assert_allclose(
        id_part.matvec(v, CTX), jnp.concatenate([v[:5], jnp.zeros(3, v.dtype)])
    )
    np.testing.assert_allclose(
        id_part.quad_form(v, CTX), jnp.sum(v[:5] ** 2), rtol=1e-6
    )
    with pytest.raises(ValueError, match="exceeding"):
        id_part.matvec(jnp.zeros(4, jnp.float32), CTX)

    # DiagonalRidge with interior and trailing zeros + free tail
    w = jnp.array([2.0, 0.0, 3.5, 0.0, 1.5, 0.0], dtype=jnp.float32)
    diag_r = DiagonalRidge(w)
    assert diag_r.n_free_for(n) == n - 3
    p_idx, f_idx = diag_r.partition_indices(n)
    np.testing.assert_array_equal(np.asarray(p_idx), np.array([0, 2, 4]))
    np.testing.assert_array_equal(np.asarray(f_idx), np.array([1, 3, 5, 6, 7]))
    w_pad = np.pad(np.asarray(w, np.float64), (0, n - len(w)))
    np.testing.assert_allclose(
        diag_r.matvec(v, CTX), w_pad * np.asarray(v, np.float64), rtol=1e-6
    )
    np.testing.assert_allclose(
        diag_r.quad_form(v, CTX),
        np.sum(w_pad * np.asarray(v, np.float64) ** 2),
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        DiagonalRidge(jnp.array([1.0, -0.1]))

    # MatrixRidge(Q) vs FactorRidge(L): Q and L with Q = L'L are NOT interchangeable
    L_rect_np = rng.normal(size=(3, 5))
    Q_psd_np = L_rect_np.T @ L_rect_np
    mat_r = MatrixRidge(jnp.asarray(Q_psd_np, jnp.float32))
    fac_r = FactorRidge(jnp.asarray(L_rect_np, jnp.float32))
    assert not mat_r.penalized_is_spd
    assert not fac_r.penalized_is_spd
    np.testing.assert_allclose(
        mat_r.matvec(v, CTX), fac_r.matvec(v, CTX), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        mat_r.quad_form(v, CTX), fac_r.quad_form(v, CTX), rtol=1e-5, atol=1e-6
    )

    # MatrixRidge(M) gives Q = M, whereas FactorRidge(M) gives Q = M'M = M^2
    M_spd = jnp.array([[2.0, 0.5], [0.5, 3.0]], dtype=jnp.float32)
    v2 = jnp.array([1.0, -2.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        MatrixRidge(M_spd).matvec(v2, CTX), M_spd @ v2, rtol=1e-6
    )
    np.testing.assert_allclose(
        FactorRidge(M_spd).matvec(v2, CTX), (M_spd.T @ M_spd) @ v2, rtol=1e-6
    )
    assert not np.allclose(
        MatrixRidge(M_spd).matvec(v2, CTX), FactorRidge(M_spd).matvec(v2, CTX)
    )

    # MatrixRidge rejects asymmetric and indefinite matrices
    with pytest.raises(ValueError, match="symmetric"):
        MatrixRidge(jnp.array([[1.0, 2.0], [0.0, 1.0]], dtype=jnp.float32))
    with pytest.raises(ValueError, match="positive semidefinite"):
        MatrixRidge(jnp.array([[1.0, 0.0], [0.0, -0.1]], dtype=jnp.float32))

    # RepeatedFactorRidge packed multi-RHS GEMMs + unpenalized free tail
    L_blk = jnp.asarray(rng.normal(size=(4, 3)), jnp.float32)
    rep_r = RepeatedFactorRidge(L_blk, repeats=2)
    assert rep_r.size == 6
    assert rep_r.n_free_for(n) == 2
    L_bar = jsp_linalg.block_diag(L_blk, L_blk)
    K_bar = jsp_linalg.block_diag(L_blk.T @ L_blk, L_blk.T @ L_blk, jnp.zeros((2, 2)))
    np.testing.assert_allclose(
        rep_r.factor_apply(mat, CTX), L_bar @ mat[:6], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        rep_r.matvec(mat, CTX), K_bar @ mat, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        rep_r.quad_form(v, CTX), v @ (K_bar @ v), rtol=1e-5, atol=1e-6
    )


def test_matvec_only_ridge_metric_on_cg_and_solver_capability_checks():
    rng = np.random.default_rng(13)
    m, n = 5, 6
    A = jnp.asarray(rng.normal(size=(m, n)), jnp.float32)
    b = jnp.asarray(rng.normal(size=m), jnp.float32)
    w = jnp.asarray(rng.uniform(0.5, 2.5, size=n), jnp.float32)
    rho = 0.05
    mv_ridge = MatvecOnlyRidge(w)

    cg_solver = LevenbergMarquardt(
        lambda x: A @ x - b,
        ridge=rho,
        ridge_metric=mv_ridge,
        linear_solver=CG(IdentityPreconditioner(), tol=1e-10, maxiter=128),
        ad_solver=CG(IdentityPreconditioner(), tol=1e-10, maxiter=128),
    )
    res = cg_solver.solve(jnp.zeros(n, jnp.float32), gtol=1e-6, max_steps=50)
    assert int(res.status) == int(SolverStatus.CONVERGED)
    expected = np.linalg.solve(
        np.asarray(A, np.float64).T @ np.asarray(A, np.float64)
        + rho * np.diag(np.asarray(w, np.float64)),
        np.asarray(A, np.float64).T @ np.asarray(b, np.float64),
    )
    np.testing.assert_allclose(res.x, expected, rtol=1e-4, atol=1e-4)

    # QR, Cholesky, GramCG, and SVD reject matvec-only ridge metrics
    for bad_ls in (Cholesky(), QR(), GramCG(IdentityPreconditioner())):
        with pytest.raises(ValueError):
            LevenbergMarquardt(
                lambda x: A @ x - b,
                ridge=rho,
                ridge_metric=mv_ridge,
                linear_solver=bad_ls,
            ).init(jnp.zeros(n, jnp.float32))
    with pytest.raises(ValueError, match="factor"):
        Newton(
            lambda x: A @ x - b,
            ridge=rho,
            ridge_metric=mv_ridge,
            linear_solver=SVD(),
        ).init(jnp.zeros(n, jnp.float32))


def test_diagonal_preconditioner_and_live_solver_context():
    with pytest.raises(ValueError, match="strictly positive"):
        DiagonalPreconditioner(jnp.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="dense solver"):
        DiagonalPreconditioner(None).apply(jnp.ones(3), jnp.float32(0.1), CTX)

    d = jnp.array([2.0, 5.0, 10.0], dtype=jnp.float32)
    dp = DiagonalPreconditioner(d)
    v = jnp.array([4.0, 11.0, 21.0], dtype=jnp.float32)
    np.testing.assert_allclose(dp.apply(v, jnp.float32(0.5), CTX), v / (d + 0.5))

    seen_ridge = []
    seen_pre = []

    @dataclasses.dataclass(frozen=True, eq=False)
    class ProbeRidge(RidgeMetric):
        def matvec(self, vec, ctx):
            seen_ridge.append(
                (ctx.x is not None, ctx.ridge is not None, ctx.role, ctx.space)
            )
            return vec

    @dataclasses.dataclass(frozen=True, eq=False)
    class ProbePre(Preconditioner):
        scale: jax.Array

        def apply(self, vec, damping, ctx):
            seen_pre.append(
                (
                    ctx.x is not None,
                    ctx.ridge is not None,
                    ctx.damping is not None,
                    ctx.space,
                    ctx.role,
                    ctx.form,
                )
            )
            return vec / (self.scale + damping)

    register_pytree_dataclass(ProbeRidge, data_fields=())
    register_pytree_dataclass(ProbePre, data_fields=("scale",))

    A = jnp.array([[1.0, 0.5, 0.0], [0.0, 2.0, 1.0]], dtype=jnp.float32)
    b = jnp.array([1.0, -1.0], dtype=jnp.float32)
    s = LevenbergMarquardt(
        lambda th, _, p: A @ th - p,
        ridge=1e-2,
        ridge_metric=ProbeRidge(),
        linear_solver=CG(ProbePre(jnp.ones(3, jnp.float32)), tol=1e-8, maxiter=16),
    )
    x0 = jnp.zeros(3, jnp.float32)
    s.update(x0, s.init(x0, p=b), p=b)
    assert seen_ridge and all(ok_x and ok_r for ok_x, ok_r, _, _ in seen_ridge)
    assert seen_pre and all(
        ok_x and ok_r and ok_d and space == "parameter" and role == "forward"
        for ok_x, ok_r, ok_d, space, role, _ in seen_pre
    )

    # Callback swap of preconditioner takes effect on subsequent step
    s_bad = LevenbergMarquardt(
        lambda th: A @ th - b,
        linear_solver=CG(DiagonalPreconditioner(jnp.full(3, 1e4)), tol=0.0, maxiter=2),
    )
    diag_exact = jnp.diag(A.T @ A)
    s_good = LevenbergMarquardt(
        lambda th: A @ th - b,
        linear_solver=CG(DiagonalPreconditioner(diag_exact), tol=0.0, maxiter=2),
    )
    swapped = dataclasses.replace(
        s_bad.init(x0), preconditioner=DiagonalPreconditioner(diag_exact)
    )
    np.testing.assert_allclose(
        s_bad.update(x0, swapped)[0],
        s_good.update(x0, s_good.init(x0))[0],
        rtol=1e-6,
        atol=1e-7,
    )

    # MatrixRidge preserves small positive eigenvalues, AD wrt Q, and traced NaNs
    Q_wide_scale = jnp.diag(jnp.array([1e10, 1.0], dtype=jnp.float32))
    mr_scale = MatrixRidge(Q_wide_scale)
    v_unit = jnp.array([0.0, 1.0], dtype=jnp.float32)
    assert float(mr_scale.quad_form(v_unit, CTX)) == pytest.approx(1.0)
    assert float(jnp.sum(mr_scale.factor_apply(v_unit, CTX) ** 2)) == pytest.approx(1.0)
    grad_q = jax.grad(
        lambda q: MatrixRidge(q).quad_form(jnp.array([2.0, 3.0], jnp.float32), CTX)
    )(jnp.eye(2, dtype=jnp.float32))
    np.testing.assert_allclose(
        grad_q, jnp.array([[4.0, 6.0], [6.0, 9.0]], dtype=jnp.float32), rtol=1e-5
    )
    bad_q_val = jax.jit(
        lambda q: MatrixRidge(q).quad_form(jnp.ones(2, jnp.float32), CTX)
    )(jnp.diag(jnp.array([1.0, -1.0], dtype=jnp.float32)))
    assert bool(jnp.isnan(bad_q_val))
    bad_d_val = jax.jit(
        lambda d: DiagonalPreconditioner(d).apply(jnp.ones(2, jnp.float32), 0.1, CTX)
    )(jnp.array([1.0, -2.0], dtype=jnp.float32))
    assert bool(jnp.all(jnp.isnan(bad_d_val)))
