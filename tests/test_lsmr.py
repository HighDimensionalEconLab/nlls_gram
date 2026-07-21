import dataclasses
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    LMSolveAction,
    LMStatus,
    Metric,
    MultiStart,
    RecycleConfig,
    WhitenedPreconditioner,
    identity_preconditioner,
    metric_from_cholesky,
)
from nlls_gram.lsmr import lsmr


def _right_preconditioner(G, lam):
    # Parameter-space right-preconditioner R = chol(G'G + lam I)': A_r = G R^{-1}
    # has a clustered spectrum (a Schur-style factor). Returns the hook and R.
    R = jnp.linalg.cholesky(G.T @ G + lam * jnp.eye(G.shape[1])).T

    def solve(v, damping):
        return jsp_linalg.solve_triangular(R, v, lower=False)

    def solve_transpose(w, damping):
        return jsp_linalg.solve_triangular(R.T, w, lower=True)

    return WhitenedPreconditioner(solve, solve_transpose), R


# --- LSMR core correctness vs a dense reference ------------------------------


@pytest.mark.parametrize("shape", [(20, 8), (8, 20), (12, 12)])
@pytest.mark.parametrize("lam", [1.0, 1e-2])
def test_lsmr_core_matches_dense(shape, lam):
    # Well-conditioned so the float32 dense normal-equations reference is itself
    # trustworthy; LSMR solves min ||A x - b||^2 + lam ||x||^2.
    m, n = shape
    A = jax.random.normal(jax.random.key(0), (m, n))
    b = jax.random.normal(jax.random.key(1), (m,))
    x, state = lsmr(
        lambda z: A @ z,
        lambda y: A.T @ y,
        b,
        damp=jnp.sqrt(jnp.asarray(lam)),
        atol=1e-8,
        btol=0.0,
        maxiter=200,
    )
    xref = jnp.linalg.solve(A.T @ A + lam * jnp.eye(n), A.T @ b)
    assert jnp.allclose(x, xref, rtol=1e-3, atol=1e-4)
    assert int(state.iterations) >= 1
    assert float(state.normal_residual) < 1e-3


def test_lsmr_core_jits():
    m, n = 15, 6
    A = jax.random.normal(jax.random.key(0), (m, n))
    b = jax.random.normal(jax.random.key(1), (m,))

    @jax.jit
    def solve(b):
        return lsmr(lambda z: A @ z, lambda y: A.T @ y, b, atol=1e-8, maxiter=100)[0]

    xref = jnp.linalg.lstsq(A, b)[0]
    assert jnp.allclose(solve(b), xref, rtol=1e-3, atol=1e-4)


# --- step parity with the dense whitened / dual paths ------------------------


def residual_fn(x, args, p):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


def test_lsmr_step_matches_augmented_qr_and_cholesky():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    common = dict(init_damping=1e-2, geodesic_acceleration=False)
    cholesky = LevenbergMarquardt(residual_fn, **common)
    aug_qr = LevenbergMarquardt(residual_fn, linear_solver="augmented_qr", **common)
    lsmr_solver = LevenbergMarquardt(
        residual_fn,
        linear_solver="lsmr",
        iterative_tol=1e-10,
        iterative_maxiter=50,
        **common,
    )
    xc, _, ic = cholesky.update(x, cholesky.init(x, (ts, ys)), (ts, ys))
    xa, _, _ = aug_qr.update(x, aug_qr.init(x, (ts, ys)), (ts, ys))
    xl, _, il = lsmr_solver.update(x, lsmr_solver.init(x, (ts, ys)), (ts, ys))
    assert bool(il.accepted) == bool(ic.accepted)
    for key in ("a", "b"):
        assert jnp.allclose(xl[key], xc[key], rtol=1e-4, atol=1e-4)
        assert jnp.allclose(xl[key], xa[key], rtol=1e-4, atol=1e-4)


def _ill_conditioned_linear(m=12, n=12, cond=1e3):
    U, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (m, m)))
    V, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(1), (n, n)))
    k = min(m, n)
    sv = jnp.logspace(0.0, -jnp.log10(cond), k)
    G = (U[:, :k] * sv) @ V[:k, :]
    b = jax.random.normal(jax.random.key(2), (m,))

    def residual(x):
        return G @ x - b

    return residual, jnp.zeros(n), G, b


def test_lsmr_beats_cg_dual_step_on_ill_conditioned_operator():
    # THE motivating case: the cg dual operator J M^{-1} J' + lambda I has
    # condition ~ cond(G)^2, so at tiny lambda its float32 step bottoms out at
    # eps * cond and here degrades enough to be rejected; the whitened operator
    # (augmented_qr and lsmr) works at cond(G) and produces an accurate step.
    residual, x0, G, b = _ill_conditioned_linear(cond=1e3)
    lam = 1e-8
    common = dict(init_damping=lam, geodesic_acceleration=False)
    reference = LevenbergMarquardt(residual, linear_solver="augmented_qr", **common)
    cg = LevenbergMarquardt(
        residual,
        linear_solver="gram_cg",
        dual_preconditioner=identity_preconditioner(),
        ad_solver_preconditioner=identity_preconditioner(),
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=200,
        **common,
    )
    lsmr_solver = LevenbergMarquardt(
        residual,
        linear_solver="lsmr",
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=200,
        **common,
    )
    xr, _, _ = reference.update(x0, reference.init(x0))
    xc, _, _ = cg.update(x0, cg.init(x0))
    xl, _, _ = lsmr_solver.update(x0, lsmr_solver.init(x0))
    scale = float(jnp.linalg.norm(xr))
    err_cg = float(jnp.linalg.norm(xc - xr)) / scale
    err_lsmr = float(jnp.linalg.norm(xl - xr)) / scale
    # lsmr tracks the whitened reference; the squared cg step is far off.
    assert err_lsmr < 1e-3
    assert err_cg > 0.1
    assert err_lsmr < 0.01 * err_cg


# --- solve / jit / callbacks -------------------------------------------------


def test_lsmr_solve_converges():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="lsmr",
        iterative_tol=1e-10,
        iterative_maxiter=50,
    )
    result = solver.solve({"a": 1.0, "b": 0.0}, (ts, ys), max_steps=60, atol=1e-6)
    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.info.loss) < 1e-6


def test_lsmr_update_jits():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="lsmr",
        iterative_tol=1e-9,
        iterative_maxiter=40,
    )

    @jax.jit
    def step(x, lm_state):
        return solver.update(x, lm_state, (ts, ys))

    x, lm_state, info = step(x, solver.init(x, (ts, ys)))
    assert bool(info.accepted)
    assert jnp.isfinite(info.loss)


def test_lsmr_callback_maxiter_schedule_composes():
    # A callback that grows iterative_maxiter mid-solve reschedules the traced
    # LSMR cap; the loop retraces nothing (maxiter rides hyper) and still converges.
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=1e-2,
        linear_solver="lsmr",
        geodesic_acceleration=False,
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=2,
    )

    def schedule(ctx):
        grown = dataclasses.replace(
            ctx.lm_state,
            hyper=dataclasses.replace(
                ctx.lm_state.hyper,
                iterative_maxiter=jnp.where(
                    ctx.info.loss < 1e-1, jnp.int32(40), jnp.int32(2)
                ),
            ),
        )
        return LMSolveAction(lm_state=grown)

    result = solver.solve(
        {"a": 1.0, "b": 0.0}, (ts, ys), max_steps=80, atol=1e-6, callback=schedule
    )
    assert int(result.status) == LMStatus.CONVERGED


def test_lsmr_multi_start_vmap():
    residual, x0, _, _ = _ill_conditioned_linear(cond=1e2)
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-4,
        linear_solver="lsmr",
        geodesic_acceleration=False,
        iterative_tol=1e-9,
        iterative_maxiter=40,
    )

    def draw(key, x, args):
        return x + 0.01 * jax.random.normal(key, x.shape), args

    for parallel in (False, True):
        ms = MultiStart(
            key=jax.random.key(0), num_starts=3, draw=draw, parallel=parallel
        )
        result = solver.solve(x0, max_steps=80, atol=1e-6, multi_start=ms)
        assert int(result.status) == LMStatus.CONVERGED


# --- custom metric -----------------------------------------------------------


def test_lsmr_with_custom_metric_matches_cholesky():
    # A dense-cholesky metric supplies inv_sqrt / inv_sqrt_transpose; the whitened
    # operator B = J S uses them and the step matches the cholesky dual solve.
    n = 6
    W = jax.random.normal(jax.random.key(0), (n, n))
    Mmat = W @ W.T + n * jnp.eye(n)
    L = jnp.linalg.cholesky(Mmat)
    metric = metric_from_cholesky(L)
    G = jax.random.normal(jax.random.key(3), (10, n))
    b = jax.random.normal(jax.random.key(4), (10,))

    def residual(x):
        return G @ x - b

    common = dict(init_damping=1e-3, metric=metric, geodesic_acceleration=False)
    cholesky = LevenbergMarquardt(residual, **common)
    lsmr_solver = LevenbergMarquardt(
        residual,
        linear_solver="lsmr",
        iterative_tol=1e-10,
        iterative_maxiter=80,
        **common,
    )
    x0 = jnp.zeros(n)
    xc, _, _ = cholesky.update(x0, cholesky.init(x0))
    xl, _, _ = lsmr_solver.update(x0, lsmr_solver.init(x0))
    assert jnp.allclose(xl, xc, rtol=1e-3, atol=1e-4)


# --- differentiation ---------------------------------------------------------


def test_lsmr_update_reverse_ad_matches_cholesky():
    ts = jnp.linspace(0.0, 2.0, 20)

    def residual_data(x, args, p):
        return x["a"] * jnp.exp(x["b"] * ts) - args

    x = {"a": 1.0, "b": 0.0}
    cholesky = LevenbergMarquardt(
        residual_data, init_damping=1e-2, geodesic_acceleration=False
    )
    lsmr_solver = LevenbergMarquardt(
        residual_data,
        init_damping=1e-2,
        linear_solver="lsmr",
        geodesic_acceleration=False,
        iterative_tol=1e-10,
        iterative_maxiter=80,
    )

    def loss_of(solver):
        def loss(ys):
            new_x, _, _ = solver.update(x, solver.init(x, ys), ys)
            return jnp.sum(new_x["a"] ** 2 + new_x["b"] ** 2)

        return loss

    ys = 2.0 * jnp.exp(-1.0 * ts)
    g_lsmr = jax.grad(loss_of(lsmr_solver))(ys)
    g_cholesky = jax.grad(loss_of(cholesky))(ys)
    assert jnp.allclose(g_lsmr, g_cholesky, rtol=1e-3, atol=1e-4)


def test_lsmr_geodesic_matches_cholesky():
    def residual(theta, target, p):
        return jnp.array([theta[0] ** 2 - target])

    theta0 = jnp.array([1.9])
    target = 4.0
    cholesky = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    lsmr_solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        linear_solver="lsmr",
        iterative_tol=1e-10,
        iterative_maxiter=20,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=1.0,
    )
    ct, _, ci = cholesky.update(theta0, cholesky.init(theta0, target), target)
    lt, _, li = lsmr_solver.update(theta0, lsmr_solver.init(theta0, target), target)
    assert bool(li.used_geodesic)
    assert jnp.allclose(lt, ct, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(
        li.acceleration_ratio, ci.acceleration_ratio, rtol=1e-4, atol=1e-5
    )


def test_lsmr_implicit_p_derivative_matches_analytic():
    # auto + lsmr defers the implicit form to trace-time shapes; this system
    # is tall (m=12 > n=1), so shape_auto resolves to normal_cholesky and the
    # p-derivative matches the analytic least-squares sensitivity
    # sum(ts)/sum(ts^2) exactly. The gram_cholesky forward's default filter
    # computes the same tangent through the rank-1 12x12 dual.
    ts = jnp.linspace(0.0, 2.0, 12)

    def residual_p(x, args, p):
        return x * ts - p

    cholesky = LevenbergMarquardt(
        residual_p, init_damping=1e-3, linear_solver="gram_cholesky"
    )
    lsmr_solver = LevenbergMarquardt(
        residual_p,
        init_damping=1e-3,
        linear_solver="lsmr",
        iterative_tol=1e-10,
        iterative_maxiter=60,
    )
    p = jnp.asarray(1.7)
    assert lsmr_solver._resolved_ad_solver == "auto"
    assert lsmr_solver._ad_solver_at(jnp.zeros(()), None, p) == "svd"
    j_analytic = jnp.sum(ts) / jnp.sum(ts**2)

    def solved(solver, q):
        return solver.solve(jnp.zeros(()), p=q, max_steps=60, atol=1e-9).x

    j_cholesky = jax.jacobian(lambda q: solved(cholesky, q))(p)
    j_lsmr = jax.jacobian(lambda q: solved(lsmr_solver, q))(p)
    assert jnp.allclose(j_lsmr, j_analytic, rtol=1e-4, atol=1e-5)
    assert jnp.allclose(j_cholesky, j_analytic, rtol=1e-4, atol=1e-5)


# --- validation --------------------------------------------------------------


def test_lsmr_rejects_recycle_and_preconditioner_hooks():
    residual, x0, _, _ = _ill_conditioned_linear()
    with pytest.raises(ValueError, match="recycle requires"):
        LevenbergMarquardt(
            residual, linear_solver="lsmr", recycle=RecycleConfig(rank=2)
        )
    from nlls_gram import PreconditionerFactory

    with pytest.raises(ValueError, match="preconditioner_factory requires"):
        LevenbergMarquardt(
            residual,
            linear_solver="lsmr",
            preconditioner_factory=PreconditionerFactory(
                lambda *a: jnp.zeros(()), lambda *a: a[1]
            ),
        )
    with pytest.raises(ValueError, match="dual_preconditioner requires"):
        LevenbergMarquardt(
            residual,
            linear_solver="lsmr",
            dual_preconditioner=identity_preconditioner(),
        )


def test_lsmr_custom_metric_requires_inv_sqrt():
    with pytest.raises(ValueError, match="inv_sqrt"):
        LevenbergMarquardt(
            lambda x: x,
            linear_solver="lsmr",
            geodesic_acceleration=False,
            metric=Metric(solve=lambda x: x),
        )


def test_unknown_linear_solver_rejected():
    with pytest.raises(ValueError, match="unknown linear_solver"):
        LevenbergMarquardt(lambda x: x, linear_solver="lsqr")


# --- x64 in a clean subprocess -----------------------------------------------


def test_lsmr_float64_subprocess():
    script = textwrap.dedent(
        """
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from nlls_gram import LevenbergMarquardt, LMStatus

        # Ill-conditioned whitened operator: the float64 dense normal-equations
        # solve is limited by cond(G)^2 ~ 1e12 (~1e-4 relative), while lsmr works
        # at cond(G) ~ 1e6 and matches the whitened augmented_qr reference tightly.
        f64 = jnp.float64
        m, n = 12, 12
        U, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (m, m), dtype=f64))
        V, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(1), (n, n), dtype=f64))
        sv = jnp.logspace(0.0, -6.0, n, dtype=f64)
        G = (U[:, :n] * sv) @ V[:n, :]
        b = jax.random.normal(jax.random.key(2), (m,), dtype=f64)

        def residual(x):
            return G @ x - b

        x0 = jnp.zeros(n, dtype=f64)
        lam = 1e-10
        common = dict(init_damping=lam, geodesic_acceleration=False)
        ref = LevenbergMarquardt(residual, linear_solver="augmented_qr", **common)
        lsmr_solver = LevenbergMarquardt(
            residual, linear_solver="lsmr", iterative_tol=0.0,
            iterative_atol=0.0, iterative_maxiter=400, **common,
        )
        xr, _, _ = ref.update(x0, ref.init(x0))
        xl, lm_state, info = lsmr_solver.update(x0, lsmr_solver.init(x0))
        assert xl.dtype == f64
        assert lm_state.damping.dtype == f64
        rel = float(jnp.linalg.norm(xl - xr) / jnp.linalg.norm(xr))
        assert rel < 1e-5, rel
        upd = lambda x, s: lsmr_solver.update(x, s)
        jaxpr = str(jax.make_jaxpr(upd)(x0, lsmr_solver.init(x0)))
        assert "f32" not in jaxpr, jaxpr
        print("OK", rel)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout


# --- whitened_preconditioner (parameter-space right-preconditioner) ----------


def test_whitened_preconditioner_reduces_lsmr_iterations():
    # A good right-preconditioner clusters the spectrum of B R^{-1}, so LSMR reaches
    # the same relative tolerance in far fewer iterations than plain LSMR.
    residual, x0, G, b = _ill_conditioned_linear(m=40, n=40, cond=1e3)
    lam = 1e-6
    prec, R = _right_preconditioner(G, lam)
    sq = jnp.sqrt(jnp.asarray(lam))
    _, plain = lsmr(
        lambda z: G @ z, lambda y: G.T @ y, -b, damp=sq, atol=1e-8, maxiter=500
    )
    _, precond = lsmr(
        lambda z: G @ prec.solve(z, lam),
        lambda y: prec.solve_transpose(G.T @ y, lam),
        -b,
        damp=sq,
        atol=1e-8,
        maxiter=500,
    )
    assert int(precond.iterations) < int(plain.iterations) // 10


def test_whitened_preconditioner_converges_at_tight_budget():
    # THE motivating case: at a tight inner budget the right-preconditioner is what
    # lets the LM solve converge; plain LSMR stalls.
    residual, x0, G, b = _ill_conditioned_linear(m=40, n=40, cond=1e3)
    lam = 1e-6
    prec, _ = _right_preconditioner(G, lam)
    common = dict(
        init_damping=lam,
        linear_solver="lsmr",
        geodesic_acceleration=False,
        iterative_tol=1e-8,
        iterative_atol=0.0,
        iterative_maxiter=12,
    )
    plain = LevenbergMarquardt(residual, **common)
    preconditioned = LevenbergMarquardt(
        residual, whitened_preconditioner=prec, **common
    )
    plain_result = plain.solve(x0, max_steps=60, atol=1e-4)
    precond_result = preconditioned.solve(x0, max_steps=60, atol=1e-4)
    assert int(precond_result.status) == LMStatus.CONVERGED
    assert float(precond_result.info.loss) < 1e-6
    assert int(plain_result.status) != LMStatus.CONVERGED
    assert float(plain_result.info.loss) > 1e-2


def test_whitened_preconditioner_converged_solution_invariant():
    # The R'R-metric damping reweights the per-step subproblem but not the converged
    # selection: a generously-budgeted preconditioned solve reaches cholesky's x*.
    residual, x0, G, b = _ill_conditioned_linear(m=30, n=30, cond=1e2)
    lam = 1e-5
    prec, _ = _right_preconditioner(G, lam)
    cholesky = LevenbergMarquardt(
        residual, init_damping=lam, geodesic_acceleration=False
    )
    preconditioned = LevenbergMarquardt(
        residual,
        init_damping=lam,
        linear_solver="lsmr",
        geodesic_acceleration=False,
        whitened_preconditioner=prec,
        iterative_tol=1e-10,
        iterative_maxiter=200,
    )
    rc = cholesky.solve(x0, max_steps=80, atol=1e-6)
    rp = preconditioned.solve(x0, max_steps=80, atol=1e-6)
    assert jnp.allclose(rp.x, rc.x, rtol=1e-3, atol=1e-4)


def test_whitened_preconditioner_forward_step_is_identity_damped():
    # The augmented damping row is sqrt(lam) R^{-1} z = sqrt(lam) u, so the
    # preconditioned single-step update is EXACTLY the I-damped
    # u = (G'G + lam I)^{-1} G' b at x0=0 (resid = -b) for any R -- and
    # distinguishable from the old R'R-damped surrogate for this non-trivial R.
    residual, x0, G, b = _ill_conditioned_linear(m=10, n=10, cond=1e2)
    lam = 1e-2
    prec, R = _right_preconditioner(G, lam)
    solver = LevenbergMarquardt(
        residual,
        init_damping=lam,
        linear_solver="lsmr",
        geodesic_acceleration=False,
        whitened_preconditioner=prec,
        iterative_tol=1e-12,
        iterative_maxiter=200,
    )
    xl, _, info = solver.update(x0, solver.init(x0))
    n = G.shape[1]
    u_rtr = jnp.linalg.solve(G.T @ G + lam * (R.T @ R), G.T @ b)
    u_identity = jnp.linalg.solve(G.T @ G + lam * jnp.eye(n), G.T @ b)
    assert bool(info.accepted)
    assert jnp.allclose(xl, u_identity, rtol=1e-3, atol=1e-4)
    assert not jnp.allclose(xl, u_rtr, rtol=1e-2, atol=1e-3)


def test_whitened_preconditioner_reverse_ad_and_implicit_p():
    # Reverse-AD through a preconditioned update is finite (the custom_linear_solve
    # on the preconditioned normal operator differentiates), and the converged
    # p-derivative -- R-invariant, resolved by shape to the normal form on
    # this tall system -- matches the analytic sensitivity, as does the
    # gram_cholesky forward's default filter.
    ts = jnp.linspace(0.0, 2.0, 12)

    def residual_p(x, args, p):
        return x * ts - p

    # data-independent frozen R from a fixed matrix (J = diag(ts) here)
    M = jax.random.normal(jax.random.key(0), (1, 1))
    R = jnp.linalg.cholesky(M @ M.T + jnp.eye(1)).T

    def solve(v, damping):
        return jsp_linalg.solve_triangular(R, v, lower=False)

    def solve_transpose(w, damping):
        return jsp_linalg.solve_triangular(R.T, w, lower=True)

    prec = WhitenedPreconditioner(solve, solve_transpose)
    cholesky = LevenbergMarquardt(
        residual_p, init_damping=1e-3, linear_solver="gram_cholesky"
    )
    preconditioned = LevenbergMarquardt(
        residual_p,
        init_damping=1e-3,
        linear_solver="lsmr",
        whitened_preconditioner=prec,
        iterative_tol=1e-10,
        iterative_maxiter=60,
    )

    def update_loss(pp):
        nx, _, _ = preconditioned.update(
            jnp.zeros(()), preconditioned.init(jnp.zeros(()), p=pp), p=pp
        )
        return jnp.sum(nx**2)

    g = jax.grad(update_loss)(jnp.asarray(1.3))
    assert bool(jnp.isfinite(g))

    p = jnp.asarray(1.7)
    j_analytic = jnp.sum(ts) / jnp.sum(ts**2)

    def solved(solver, q):
        return solver.solve(jnp.zeros(()), p=q, max_steps=60, atol=1e-9).x

    j_cholesky = jax.jacobian(lambda q: solved(cholesky, q))(p)
    j_preconditioned = jax.jacobian(lambda q: solved(preconditioned, q))(p)
    assert jnp.allclose(j_preconditioned, j_analytic, rtol=1e-4, atol=1e-5)
    assert jnp.allclose(j_cholesky, j_analytic, rtol=1e-4, atol=1e-5)


def test_whitened_preconditioner_requires_lsmr():
    residual, x0, G, b = _ill_conditioned_linear()
    prec, _ = _right_preconditioner(G, 1e-6)
    with pytest.raises(ValueError, match="whitened_preconditioner requires"):
        LevenbergMarquardt(
            residual,
            linear_solver="gram_cg",
            dual_preconditioner=identity_preconditioner(),
            ad_solver_preconditioner=identity_preconditioner(),
            whitened_preconditioner=prec,
        )


def test_whitened_preconditioner_hashing_shares_compilation():
    residual, x0, G, b = _ill_conditioned_linear()
    prec, _ = _right_preconditioner(G, 1e-6)
    common = dict(linear_solver="lsmr", iterative_maxiter=8)
    a = LevenbergMarquardt(residual, whitened_preconditioner=prec, **common)
    b_solver = LevenbergMarquardt(residual, whitened_preconditioner=prec, **common)
    c = LevenbergMarquardt(residual, **common)  # plain
    assert a == b_solver and hash(a) == hash(b_solver)
    assert a != c


# --- chained two-phase AD contract -------------------------------------------


def test_chained_solve_derivative_is_final_phase_implicit_rule():
    # The two-phase pattern (fast solver to a plateau, then a certifying lsmr
    # solve warm-started from it) must differentiate as the implicit rule at
    # the FINAL converged point only: solve()'s custom JVP consumes the p
    # tangent alone, so warm-start (x0/lm_state) tangents from an unconverged
    # phase 1 are dropped by construction.
    n = 6
    A = jax.random.normal(jax.random.key(0), (n, n)) + 3.0 * jnp.eye(n)
    p0 = jnp.exp(0.2 * jax.random.normal(jax.random.key(1), (n,)))

    def residual(x, args, p):
        return A @ x + 0.1 * jnp.tanh(x) - p

    phase1 = LevenbergMarquardt(residual, init_damping=1e-2, ad_solver="svd")
    phase2 = LevenbergMarquardt(
        residual,
        linear_solver="lsmr",
        ad_solver="svd",
        init_damping=1e-10,
        iterative_tol=1e-13,
        iterative_atol=1e-13,
        iterative_maxiter=200,
        geodesic_acceleration=False,
    )
    reference = LevenbergMarquardt(residual, init_damping=1e-2, ad_solver="svd")

    def chained(p):
        plateau = phase1.solve(jnp.zeros(n), p=p, max_steps=2, atol=0.0)
        return phase2.solve(plateau.x, p=p, max_steps=20, atol=1e-12).x

    def direct(p):
        return reference.solve(jnp.zeros(n), p=p, max_steps=60, atol=1e-12).x

    tangent = jnp.linspace(-1.0, 1.0, n)
    x_chained, dx_chained = jax.jvp(chained, (p0,), (tangent,))
    x_direct, dx_direct = jax.jvp(direct, (p0,), (tangent,))
    # Well-determined system: unique root, so both paths converge to the same
    # x* and the chained derivative must equal the single-solve implicit rule.
    assert jnp.allclose(x_chained, x_direct, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(dx_chained, dx_direct, rtol=1e-4, atol=1e-5)

    # Reverse mode agrees with forward mode through the chain.
    cotangent = jnp.cos(jnp.arange(n, dtype=p0.dtype))
    _, pullback = jax.vjp(chained, p0)
    (p_bar,) = pullback(cotangent)
    assert jnp.allclose(p_bar @ tangent, cotangent @ dx_chained, rtol=1e-4, atol=1e-5)

    # The phase boundary carries no derivative: perturbing the warm start
    # leaves the chained solution's tangent at exactly zero.
    def from_start(x0):
        return phase2.solve(x0, p=p0, max_steps=20, atol=1e-12).x

    _, dx_start = jax.jvp(from_start, (0.1 * jnp.ones(n),), (jnp.ones(n),))
    assert jnp.allclose(dx_start, jnp.zeros(n), atol=1e-12)
