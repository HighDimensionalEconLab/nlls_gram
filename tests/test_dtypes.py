import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    PDAS,
    QR,
    SVD,
    AnnealRidge,
    BoxSecant,
    Cholesky,
    Complementarity,
    DiagonalPreconditioner,
    DiagonalRidge,
    FactorRidge,
    FischerBurmeister,
    GramCG,
    IdentityPreconditioner,
    Inequality,
    InequalityPenalty,
    LevenbergMarquardt,
    LMState,
    Minimum,
    MultiStart,
    Newton,
    RepeatedFactorRidge,
    ScalarNewton,
    SolverAction,
    SolverStatus,
)

RNG = np.random.default_rng(5)
M, N = 4, 7
A_NP = RNG.normal(size=(M, N))
B_NP = RNG.normal(size=M)
A = jnp.asarray(A_NP, jnp.float32)
B = jnp.asarray(B_NP, jnp.float32)
X0 = jnp.zeros(N, jnp.float32)


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
@pytest.mark.parametrize(
    "ls", [Cholesky(), QR(), GramCG(IdentityPreconditioner(), maxiter=64)]
)
def test_float32_purity_and_highest_matmul_precision(solver_cls, ls):
    def elem_res(x, _, p):
        return x[:M] + x[M : 2 * M] - p

    solver = solver_cls(
        elem_res,
        linear_solver=ls,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    p0 = jnp.linspace(0.5, 1.5, M, dtype=jnp.float32)
    x0 = jnp.zeros(2 * M, jnp.float32)

    res = solver.solve(x0, None, p=p0, max_steps=10, atol=1e-5)
    assert int(res.status) == int(SolverStatus.CONVERGED)
    assert res.x.dtype == jnp.float32
    assert res.info.loss.dtype == jnp.float32

    def run(p):
        return solver.solve(x0, None, p=p, max_steps=4).x

    tangent = jax.jvp(run, (p0,), (jnp.ones_like(p0),))[1]
    (cotangent,) = jax.vjp(run, p0)[1](jnp.ones_like(x0))
    assert tangent.dtype == jnp.float32 and cotangent.dtype == jnp.float32

    stack = [jax.make_jaxpr(run)(p0).jaxpr]
    dots = []
    while stack:
        current = stack.pop()
        for eqn in current.eqns:
            if eqn.primitive.name == "dot_general":
                dots.append(eqn)
            for val in eqn.params.values():
                if hasattr(val, "eqns"):
                    stack.append(val)
                elif hasattr(val, "jaxpr"):
                    inner = val.jaxpr
                    stack.append(inner.jaxpr if hasattr(inner, "jaxpr") else inner)
                elif isinstance(val, (tuple, list)):
                    stack.extend(item.jaxpr for item in val if hasattr(item, "jaxpr"))

    assert dots
    unpinned = [
        eqn
        for eqn in dots
        if eqn.params.get("precision")
        != (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    ]
    assert not unpinned


def _run_x64_checks():
    jax.config.update("jax_default_matmul_precision", "highest")
    assert jax.config.jax_enable_x64
    # 1. Five-way step agreement (Cholesky, QR, CG) to 1e-9 in float64
    m, block, repeats, free = 5, 4, 2, 2
    p_dim = repeats * block + free
    A64 = jnp.asarray(RNG.normal(size=(m, p_dim)), jnp.float64)
    b64 = jnp.asarray(RNG.normal(size=m), jnp.float64)
    root = RNG.normal(size=(block + 2, block))
    K_np = root.T @ root + 0.5 * np.eye(block)
    K64 = jnp.asarray(K_np, jnp.float64)
    x0_64 = jnp.asarray(RNG.normal(size=p_dim), jnp.float64)
    rm64 = RepeatedFactorRidge(jnp.linalg.cholesky(K64, upper=True), repeats=repeats)

    steps = {}
    for name, ls in (
        ("chol_normal", Cholesky(form="normal")),
        ("chol_gram", Cholesky(form="gram")),
        ("qr_normal", QR(form="normal")),
        ("qr_gram", QR(form="gram")),
        ("cg", CG(IdentityPreconditioner(), tol=1e-14, maxiter=None)),
    ):
        s = LevenbergMarquardt(
            lambda th: A64 @ th - b64,
            ridge_metric=rm64,
            ridge=1e-8,
            init_damping=1e-12,
            linear_solver=ls,
            geodesic_acceleration=False,
        )
        x1, state1, info1 = s.update(x0_64, s.init(x0_64))
        assert x1.dtype == jnp.float64 and state1.ridge.dtype == jnp.float64
        assert info1.loss.dtype == jnp.float64
        steps[name] = np.asarray(x1)
    for high_acc in ("chol_gram", "qr_gram"):
        np.testing.assert_allclose(
            steps["qr_normal"], steps[high_acc], rtol=1e-11, atol=1e-12
        )
    for normal_eq in ("chol_normal", "cg"):
        np.testing.assert_allclose(
            steps["qr_normal"], steps[normal_eq], rtol=1e-6, atol=5e-7
        )

    # 2. AnnealRidge continuation matches minimum-seminorm KKT solution to 1e-6
    M0 = np.zeros((p_dim, p_dim))
    for j in range(repeats):
        M0[j * block : (j + 1) * block, j * block : (j + 1) * block] = K_np
    A_np, b_np = np.asarray(A64), np.asarray(b64)
    kkt = np.block([[M0, A_np.T], [A_np, np.zeros((m, m))]])
    x_dagger = np.linalg.solve(kkt, np.concatenate([np.zeros(p_dim), b_np]))[:p_dim]

    anneal = AnnealRidge(ridge_floor=1e-8, decrease=0.1)
    ridge_res = LevenbergMarquardt(
        lambda th: A64 @ th - b64,
        ridge_metric=rm64,
        ridge=1e-4,
        linear_solver=QR(form="gram"),
    ).solve(
        jnp.zeros(p_dim, jnp.float64),
        max_steps=400,
        gtol=1e-14,
        callback=anneal,
        user_state=anneal.init_state(jnp.float64),
    )
    assert int(ridge_res.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(np.asarray(ridge_res.x), x_dagger, atol=1e-6)

    # 3. Square implicit tangent uses LU at cond(J) instead of Cholesky at cond(J)^2
    n_sq = 10
    u, _ = np.linalg.qr(RNG.standard_normal((n_sq, n_sq)))
    v, _ = np.linalg.qr(RNG.standard_normal((n_sq, n_sq)))
    A_ill_np = u @ np.diag(np.logspace(0, -8, n_sq)) @ v.T
    A_ill = jnp.asarray(A_ill_np, jnp.float64)
    p_ill = jnp.asarray(RNG.standard_normal(n_sq), jnp.float64)
    d_ill = jnp.asarray(RNG.standard_normal(n_sq), jnp.float64)
    expected_tan = np.linalg.solve(A_ill_np, np.asarray(d_ill))

    def tan_err(ad_s):
        s = LevenbergMarquardt(
            lambda x, _, p: A_ill @ x - p,
            ad_solver=ad_s,
            cache_jacobian=False,
            geodesic_acceleration=False,
        )
        got = jax.jvp(
            lambda pv: (
                s.solve(
                    jnp.zeros(n_sq, jnp.float64),
                    p=pv,
                    max_steps=100,
                    atol=1e-13,
                    max_steps_is_success=True,
                ).x
            ),
            (p_ill,),
            (d_ill,),
        )[1]
        return float(
            jnp.linalg.norm(got - expected_tan) / jnp.linalg.norm(expected_tan)
        )

    assert tan_err(None) < 1e-7
    assert tan_err(None) < tan_err(Cholesky()) / 1000.0

    # 4. Near-duplicate rows in float64: SVD AD recovers exact pseudoinverse tangent
    w = jnp.array([1.0, 2.0, 3.0], jnp.float64)
    wiggles = 1.0 + 1e-13 * jnp.arange(30.0, dtype=jnp.float64)
    s_dup = LevenbergMarquardt(
        lambda x, _, p: wiggles * (jnp.dot(w, x) - p), ad_solver=SVD()
    )
    grad_dup = jax.grad(
        lambda t: jnp.sum(
            s_dup.solve(jnp.zeros(3, jnp.float64), p=t, max_steps=50, gtol=1e-12).x
        )
    )(jnp.float64(1.0))
    np.testing.assert_allclose(grad_dup, jnp.sum(w) / jnp.dot(w, w), rtol=1e-8)

    # 5. Damping floor in float64 is finfo(float64).tiny
    s_floor = LevenbergMarquardt(
        lambda th: th, cache_jacobian=False, geodesic_acceleration=False
    )
    _, st_floor, _ = s_floor.update(jnp.ones(1, jnp.float64), LMState(jnp.float64(0.0)))
    assert st_floor.damping == jnp.finfo(jnp.float64).tiny

    # 6. Float32 problems under enabled x64 never promote
    A32 = jnp.asarray(A_NP, jnp.float32)
    b32 = jnp.asarray(B_NP, jnp.float32)
    x0_32 = jnp.zeros(N, jnp.float32)

    def promotions(text):
        bad = []
        for line in text.splitlines():
            s = line.strip()
            if (
                " = " in s
                and ":f64[" in s.split(" = ")[0]
                and "convert_element_type" not in s
            ):
                bad.append(s[:100])
        return bad

    for solver_cls in (LevenbergMarquardt, Newton):
        for kwargs in [
            dict(linear_solver=Cholesky(preconditioner=DiagonalPreconditioner(None))),
            dict(
                linear_solver=QR(
                    form="gram", preconditioner=DiagonalPreconditioner(None)
                )
            ),
            dict(
                linear_solver=GramCG(IdentityPreconditioner(), maxiter=8),
                ad_solver=SVD(),
            ),
            dict(
                ridge_metric=DiagonalRidge(jnp.ones(N, jnp.float32) * 2.0),
                ridge=1e-3,
            ),
            dict(
                ridge_metric=FactorRidge(jnp.eye(N, dtype=jnp.float32)),
                ridge=1e-3,
            ),
        ]:
            solver = solver_cls(lambda x, _, p: A32 @ x - p, **kwargs)
            tol_kw = dict(gtol=1e-4) if "ridge" in kwargs else dict(atol=1e-5)

            def run(p, solver=solver, tol_kw=tol_kw):
                return solver.solve(x0_32, None, p=p, max_steps=6, **tol_kw).x

            for mode, jpr in (
                ("primal", jax.make_jaxpr(run)(b32)),
                (
                    "jvp",
                    jax.make_jaxpr(
                        lambda p: jax.jvp(run, (p,), (jnp.ones_like(p),))[1]
                    )(b32),
                ),
                (
                    "vjp",
                    jax.make_jaxpr(
                        lambda p: jax.vjp(run, p)[1](jnp.ones(N, jnp.float32))[0]
                    )(b32),
                ),
            ):
                bad = promotions(str(jpr))
                assert not bad, f"{solver_cls.__name__} [{mode}] promoted to f64: {bad}"
            out = run(b32)
            assert out.dtype == jnp.float32

        def cb(ctx):
            stop = ctx.info.loss < 1e-10
            status = jnp.where(stop, SolverStatus.CONVERGED, SolverStatus.RUNNING)
            return SolverAction(stop=stop, status=status)

        ms = MultiStart(
            key=jax.random.key(1),
            num_starts=2,
            draw=lambda k, x, a: jnp.zeros_like(x),
            parallel=True,
        )
        res_ms = solver_cls(lambda x, _, p: A32 @ x - p).solve(
            x0_32,
            p=b32,
            max_steps=15,
            atol=1e-5,
            callback=cb,
            save_steps=True,
            multi_start=ms,
        )
        assert res_ms.x.dtype == jnp.float32
        assert res_ms.x_history.dtype == jnp.float32
        assert res_ms.multi_start.loss.dtype == jnp.float32
        assert res_ms.status.dtype == jnp.int32

    # 7. Inequality and Complementarity in float64 and float32 non-promotion
    s_ineq32 = LevenbergMarquardt(
        lambda x, _, p: Inequality(x - p),
        inequality=InequalityPenalty(weight=2.0, smoothing=0.1, smoothing_decay=0.5),
    )
    jpr_ineq32 = jax.make_jaxpr(
        lambda p: (
            s_ineq32.solve(jnp.zeros(3, jnp.float32), p=p, max_steps=6, gtol=1e-5).x
        )
    )(jnp.ones(3, jnp.float32))
    assert not promotions(str(jpr_ineq32))

    w64 = jnp.linspace(0.6, 2.0, 6, dtype=jnp.float64)
    beta64, r64, y64 = jnp.float64(0.95), jnp.float64(1.05), jnp.float64(1.1)
    kappa64 = 1.0 / (beta64 * r64)
    a_exact64 = jnp.maximum(0.0, (w64 - kappa64 * y64) / (1.0 + kappa64 * r64))

    def euler64(a, _, y):
        return Complementarity(
            lhs=a, rhs=1.0 / (w64 - a) - beta64 * r64 / (r64 * a + y)
        )

    for cfg in (
        PDAS(tau_init=0.05, tau_min=1e-8),
        FischerBurmeister(mu_init=0.05, mu_min=1e-10),
        Minimum(),
    ):
        sol64 = LevenbergMarquardt(euler64, complementarity=cfg).solve(
            jnp.full_like(w64, 0.2), p=y64, max_steps=60, atol=1e-12
        )
        assert sol64.x.dtype == jnp.float64
        np.testing.assert_allclose(sol64.x, a_exact64, atol=1e-11)

        s32 = LevenbergMarquardt(
            lambda x, _, p: Complementarity(lhs=x, rhs=x - p),
            complementarity=cfg,
        )
        jpr32 = jax.make_jaxpr(
            lambda p, s=s32: (
                s.solve(jnp.ones(3, jnp.float32), p=p, max_steps=6, atol=1e-5).x
            )
        )(jnp.ones(3, jnp.float32))
        bad32 = promotions(str(jpr32))
        assert not bad32, f"Complementarity {cfg} promoted to f64: {bad32}"

    # 8. BoxSecant (.lm) and ScalarNewton float64 precision and float32 non-promotion
    bs64 = BoxSecant(lambda x, _, p: 0.5 * (x - p) ** 2)
    r_bs64 = bs64.solve(jnp.float64(0.2), jnp.float64(1.0), p=jnp.float64(-0.3))
    assert r_bs64.x.dtype == jnp.float64 and r_bs64.lm.lower.dtype == jnp.float64
    np.testing.assert_allclose(r_bs64.lm.lower, 0.5, atol=1e-13)

    sn64 = ScalarNewton(lambda x, _, p: 0.5 * (x - p) ** 2 + 0.1 * jnp.cosh(x - p))
    r_sn64 = sn64.solve(jnp.float64(0.0), p=jnp.float64(0.7))
    assert r_sn64.x.dtype == jnp.float64 and r_sn64.curv.dtype == jnp.float64
    np.testing.assert_allclose(r_sn64.x, 0.7, atol=1e-13)
    assert abs(float(r_sn64.grad)) <= 1e-12

    p32 = jnp.float32(0.4)
    bs_f32 = lambda p: (  # noqa: E731
        bs64.solve(jnp.float32(0.5), jnp.float32(1.0), p=p).lm.lower
    )
    sn_f32 = lambda p: sn64.solve(jnp.float32(0.1), p=p).x  # noqa: E731
    for label, fn32 in (("BoxSecant.lm", bs_f32), ("ScalarNewton", sn_f32)):
        for mode, jpr in (
            ("primal", jax.make_jaxpr(fn32)(p32)),
            (
                "jvp",
                jax.make_jaxpr(lambda p, f=fn32: jax.jvp(f, (p,), (p32,))[1])(p32),
            ),
            ("vjp", jax.make_jaxpr(lambda p, f=fn32: jax.grad(f)(p))(p32)),
        ):
            bad_s = promotions(str(jpr))
            assert not bad_s, f"{label} [{mode}] promoted to f64: {bad_s}"


def test_float64_and_x64_nonpromotion_in_subprocess():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_dtypes import _run_x64_checks; _run_x64_checks()",
        ],
        env={
            **os.environ,
            "JAX_ENABLE_X64": "1",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        },
        check=True,
    )
