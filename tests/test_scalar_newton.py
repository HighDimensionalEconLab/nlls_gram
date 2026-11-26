import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    Cholesky,
    LevenbergMarquardt,
    ScalarGlobalization,
    ScalarNewton,
    ScalarSolveResult,
    SolverStatus,
    TrustRegion,
    TrustRegionInfo,
    vmap_rows,
)
from nlls_gram.scalar_newton import _solve_scalar_newton_jit

VAL_TOL_F32 = 1e-5
GRAD_TOL_F32 = 2e-4
VAL_TOL_F64 = 1e-10
GRAD_TOL_F64 = 1e-9


@pytest.mark.parametrize("loop", ["while", "fori"])
def test_arities_senses_and_warm_start_fast_exit(loop):
    s1 = ScalarNewton(
        lambda x: 0.5 * (x - 0.3) ** 2 + 0.1 * jnp.cosh(x - 0.3), loop=loop
    )
    r1 = s1.solve(jnp.float32(2.0))
    assert isinstance(r1, ScalarSolveResult)
    assert isinstance(r1.info, TrustRegionInfo)
    assert int(r1.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(float(r1.x), 0.3, atol=VAL_TOL_F32)
    np.testing.assert_allclose(float(r1.grad), 0.0, atol=VAL_TOL_F32)
    np.testing.assert_allclose(float(r1.curv), 1.1, rtol=1e-4)

    s2 = ScalarNewton(
        lambda x, a: -0.5 * a * (x + 1.4) ** 2 + 50.0, sense="max", loop=loop
    )
    r2 = s2.solve(jnp.float32(0.5), jnp.float32(3.0))
    assert int(r2.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(float(r2.x), -1.4, atol=VAL_TOL_F32)
    np.testing.assert_allclose(float(r2.fun), 50.0, atol=1e-5)
    np.testing.assert_allclose(float(r2.curv), -3.0, rtol=1e-5)

    # Exact warm start exits in 0 loop steps; nearby warm start converges in 1 step
    s3 = ScalarNewton(
        lambda x, _, p: 50.0 + 0.5 * (x - p) ** 2 + 0.05 * jnp.cos(x - p),
        loop=loop,
    )
    r_exact = s3.solve(jnp.float32(0.4), p=jnp.float32(0.4))
    assert int(r_exact.status) == int(SolverStatus.CONVERGED)
    assert int(r_exact.steps) == 0
    assert float(r_exact.info.step_norm) == 0.0

    r_near = s3.solve(jnp.float32(0.4 + 5e-4), p=jnp.float32(0.4))
    assert int(r_near.status) == int(SolverStatus.CONVERGED)
    assert int(r_near.steps) <= 2
    np.testing.assert_allclose(float(r_near.x), 0.4, atol=VAL_TOL_F32)


def test_escaping_local_maxima_nan_domains_and_large_scale():
    # Warm start dead on a strict concave peak (g == 0, curv < 0) escapes downhill
    s_peak = ScalarNewton(lambda x: jnp.cos(x) + 0.25 * x**2)
    r_peak = s_peak.solve(jnp.float32(0.0))
    assert int(r_peak.status) == int(SolverStatus.CONVERGED)
    assert float(r_peak.curv) > 0.0
    assert float(r_peak.fun) < float(s_peak._raw_fn(0.0)) - 0.2

    # Undefined domain wall (x <= 0 gives NaN/Inf in log): shrinks and recovers
    s_log = ScalarNewton(lambda x, _, p: -p * jnp.log(x) + x)
    r_log = s_log.solve(jnp.float32(1.0), p=jnp.float32(0.02))
    assert int(r_log.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(float(r_log.x), 0.02, rtol=1e-4, atol=1e-5)

    # Constant objective converges in 0 steps
    s_const = ScalarNewton(lambda x, _, p: 0.0 * x + p**3)
    r_const = s_const.solve(jnp.float32(1.5), p=jnp.float32(2.0))
    assert int(r_const.steps) == 0
    assert int(r_const.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(
        float(jax.grad(lambda p: s_const.solve(1.5, p=p).fun)(jnp.float32(2.0))),
        12.0,
        rtol=1e-5,
    )

    # Large-coordinate float32 scaling
    s_large = ScalarNewton(lambda x: 0.5 * (x - jnp.float32(15432.5)) ** 2)
    r_large = s_large.solve(jnp.float32(1000.0))
    assert int(r_large.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(float(r_large.x), 15432.5, rtol=1e-5)


@pytest.mark.parametrize("use_x64", [False, True])
def test_implicit_ad_first_and_second_order_and_inert_x0(use_x64):
    with jax.enable_x64(use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        rtol = GRAD_TOL_F64 if use_x64 else GRAD_TOL_F32
        atol = VAL_TOL_F64 if use_x64 else GRAD_TOL_F32

        # Closed-form solution: min_x 0.5*(1 + p0^2)*x^2 - p1*x + p2^3
        # x*(p) = p1 / (1 + p0^2), fun*(p) = -0.5 * p1^2 / (1 + p0^2) + p2^3
        def obj(x, a, p):
            return 0.5 * (a + p[0] ** 2) * x**2 - p[1] * x + p[2] ** 3

        solver = ScalarNewton(obj)
        a_val = jnp.asarray(1.0, dtype=dtype)
        x0 = jnp.asarray(0.1, dtype=dtype)
        p0 = jnp.array([0.6, 1.3, -0.5], dtype=dtype)

        def exact_pair(p):
            xs = p[1] / (1.0 + p[0] ** 2)
            fs = -0.5 * (p[1] ** 2) / (1.0 + p[0] ** 2) + p[2] ** 3
            return jnp.stack([xs, fs, jnp.zeros_like(xs)])

        def solve_vec(p):
            r = solver.solve(x0, a_val, p=p)
            return jnp.stack([r.x, r.fun, r.grad])

        jf = jax.jacfwd(solve_vec)(p0)
        jr = jax.jacrev(solve_vec)(p0)
        je = jax.jacrev(exact_pair)(p0)
        np.testing.assert_allclose(jf, je, rtol=rtol, atol=atol)
        np.testing.assert_allclose(jr, je, rtol=rtol, atol=atol)
        np.testing.assert_array_equal(np.asarray(jr[2]), np.zeros(3))

        # Second-order Hessians of both .x and .fun (Envelope Theorem check)
        hx = jax.hessian(lambda p: solver.solve(x0, a_val, p=p).x)(p0)
        hf = jax.hessian(lambda p: solver.solve(x0, a_val, p=p).fun)(p0)
        np.testing.assert_allclose(
            hx, jax.hessian(lambda p: exact_pair(p)[0])(p0), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            hf, jax.hessian(lambda p: exact_pair(p)[1])(p0), rtol=rtol, atol=atol
        )

        # Initial guess x0 is inert under AD
        gx0 = jax.grad(lambda xi: solver.solve(xi, a_val, p=p0).x)(x0)
        assert float(gx0) == 0.0


def test_has_aux_failed_lanes_vmap_and_outer_lm():
    def obj_aux(x, shift, p):
        val = 0.5 * (jnp.sqrt(x + shift) - p["t"]) ** 2 + 0.1 * (x - p["t"] ** 2) ** 2
        aux = {
            "val_sq": x**2 + 2.0 * p["s"],
            "tag": jnp.int32(3),
            "pos": x > 0.0,
            "none": None,
        }
        return val, aux

    solver = ScalarNewton(obj_aux, has_aux=True)
    p0 = {"t": jnp.float32(0.8), "s": jnp.float32(1.5)}

    # Lane 0 starts at sqrt(-4) -> NONFINITE; lane 1 converges cleanly
    shifts = jnp.array([-5.0, 0.0], dtype=jnp.float32)
    x_inits = jnp.array([1.0, 0.5], dtype=jnp.float32)

    def batch_run(p):
        return jax.vmap(lambda xi, sh: solver.solve(xi, sh, p=p))(x_inits, shifts)

    res, tan = jax.jvp(
        batch_run,
        (p0,),
        ({"t": jnp.float32(1.0), "s": jnp.float32(1.0)},),
    )
    assert int(res.status[0]) == int(SolverStatus.NONFINITE)
    assert int(res.status[1]) == int(SolverStatus.CONVERGED)
    assert float(tan.x[0]) == 0.0
    assert float(tan.aux["val_sq"][0]) == 0.0
    assert jnp.isfinite(tan.x[1]) and float(tan.x[1]) > 0.0

    grad_p = jax.grad(lambda p: jnp.sum(batch_run(p).x + batch_run(p).aux["val_sq"]))(
        p0
    )
    assert jnp.isfinite(grad_p["t"]) and jnp.isfinite(grad_p["s"])
    np.testing.assert_allclose(float(grad_p["s"]), 2.0, rtol=1e-5)

    # MAX_STEPS masking vs max_steps_is_success=True
    slow = ScalarNewton(lambda x, _, p: jnp.cosh(x - p) + 0.2 * (x - p) ** 4)
    r_ms = slow.solve(jnp.float32(4.0), p=jnp.float32(0.2), max_steps=1)
    assert int(r_ms.status) == int(SolverStatus.MAX_STEPS)
    g_mask = jax.grad(
        lambda p: slow.solve(4.0, p=p, max_steps=1, max_steps_is_success=False).x
    )(jnp.float32(0.2))
    g_unmask = jax.grad(
        lambda p: slow.solve(4.0, p=p, max_steps=1, max_steps_is_success=True).x
    )(jnp.float32(0.2))
    assert float(g_mask) == 0.0
    assert float(g_unmask) != 0.0

    # Nested 2D vmap with per-lane TrustRegion radii + outer LevenbergMarquardt
    inner = ScalarNewton(
        lambda x, w, th: 0.5 * (x - (w + 0.5 * th[0])) ** 2 + 0.1 * th[1] * x
    )
    ws = jnp.linspace(-0.5, 0.8, 8, dtype=jnp.float32)
    targets = ws + 0.5 * 0.6 - 0.1 * (-0.4)
    batched_rows = vmap_rows(
        lambda w, t, th: inner.solve(w, w, p=th).x - t, in_axes=(0, 0, None)
    )
    outer = LevenbergMarquardt(
        lambda th, args: batched_rows(args[0], args[1], th),
        linear_solver=Cholesky(form="normal"),
    )
    out_res = outer.solve(
        jnp.zeros(2, dtype=jnp.float32), (ws, targets), atol=1e-5, max_steps=20
    )
    assert int(out_res.status) == int(SolverStatus.CONVERGED)
    assert float(out_res.info.loss) < 1e-9

    radii = jnp.array([0.2, 1.0, 3.0], dtype=jnp.float32)
    gtols = jnp.full((3, 4), 1e-5, dtype=jnp.float32)
    with jax.checking_leaks():
        v_rad = jax.jit(
            jax.vmap(
                lambda r, gt_row: jax.vmap(
                    lambda x_i, gt: (
                        ScalarNewton(
                            lambda x, _, p: 0.5 * (x - p) ** 2,
                            globalization=TrustRegion(init_radius=r),
                        )
                        .solve(x_i, p=jnp.float32(0.7), gtol=gt)
                        .x
                    )
                )(jnp.zeros(4, dtype=jnp.float32), gt_row)
            )
        )(radii, gtols)
    assert v_rad.shape == (3, 4)
    np.testing.assert_allclose(v_rad, np.full((3, 4), 0.7), atol=VAL_TOL_F32)


def test_review_edge_cases_stagnation_budgets_elementwise_and_higher_ad():
    # 1. All-rejected trials due to precision loss report STAGNATION, not CONVERGED
    s_stag = ScalarNewton(
        lambda x: jnp.float32(1e8) + jnp.float32(5e-5) * (x - 5.0) ** 2
    )
    r_stag = s_stag.solve(jnp.float32(0.0))
    assert int(r_stag.status) == int(SolverStatus.STAGNATION)

    # 2. Small-magnitude objective does not accept uphill jumps under ULP polish,
    #    and max_radius caps initial step radius
    s_tiny = ScalarNewton(
        lambda x: (
            jnp.float32(1e-7) * (0.5 * (1000.0 * x - 1.0) ** 2 + (1000.0 * x) ** 6)
        ),
        globalization=TrustRegion(max_radius=0.1),
    )
    r_tiny = s_tiny.solve(jnp.float32(0.0))
    assert float(r_tiny.fun) <= float(s_tiny._raw_fn(jnp.float32(0.0)))
    assert float(r_tiny.info.radius) <= 0.1 + 1e-7

    # 3. Per-lane max_steps under vmap matches independent scalar solves
    s_cap = ScalarNewton(
        lambda x: 0.5 * (x - 2.0) ** 2, globalization=TrustRegion(init_radius=0.25)
    )
    caps = jnp.array([0, 1, 9], dtype=jnp.int32)
    v_cap = jax.vmap(lambda ms: s_cap.solve(jnp.float32(0.0), max_steps=ms))(caps)
    np.testing.assert_array_equal(np.asarray(v_cap.steps), np.array([0, 1, 4]))
    np.testing.assert_allclose(
        np.asarray(v_cap.x), np.array([0.0, 0.25, 2.0]), atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(v_cap.status),
        np.array(
            [SolverStatus.MAX_STEPS, SolverStatus.MAX_STEPS, SolverStatus.CONVERGED]
        ),
    )

    # 4. Elementwise batch (x0 shape (3,)) with scalar p supports JVP & VJP
    s_elem = ScalarNewton(lambda x, _, p: 0.5 * (x - p) ** 2)
    x0_vec = jnp.array([-1.0, 0.0, 2.0], dtype=jnp.float32)
    g_elem = jax.grad(lambda p: jnp.sum(s_elem.solve(x0_vec, p=p).x))(jnp.float32(0.4))
    np.testing.assert_allclose(float(g_elem), 3.0, atol=1e-5)

    # 5. Higher-order reverse AD (grad(grad) & jacrev(jacfwd)) with failed NaN lane
    s_nan = ScalarNewton(lambda x, a, p: 0.5 * (x - jnp.sqrt(a + p)) ** 2)
    a_lanes = jnp.array([-2.0, 1.0], dtype=jnp.float32)
    loss_x = lambda p: jnp.sum(  # noqa: E731
        jax.vmap(lambda a: s_nan.solve(jnp.float32(0.5), a, p=p).x)(a_lanes)
    )
    expected_h = -0.25 / (2.0**1.5)
    np.testing.assert_allclose(
        float(jax.grad(jax.grad(loss_x))(jnp.float32(1.0))), expected_h, rtol=1e-4
    )
    np.testing.assert_allclose(
        float(jax.jacrev(jax.jacfwd(loss_x))(jnp.float32(1.0))), expected_h, rtol=1e-4
    )


def test_compilation_sharing_and_immutability():
    assert issubclass(TrustRegion, ScalarGlobalization)
    _solve_scalar_newton_jit._clear_cache()
    try:
        fn = lambda x, a, p: 0.5 * a * (x - p) ** 2  # noqa: E731
        for r_init, target in ((0.5, 0.2), (2.0, 0.8), (10.0, -0.4)):
            slv = ScalarNewton(
                fn,
                globalization=TrustRegion(
                    init_radius=r_init, max_radius=100.0, eta_accept=1e-3
                ),
            )
            slv.solve(
                jnp.float32(0.0),
                jnp.float32(2.0),
                p=jnp.float32(target),
                gtol=1e-5,
                max_steps=16,
            )
        assert _solve_scalar_newton_jit._cache_size() == 1

        with pytest.raises(AttributeError, match="frozen after construction"):
            slv.sense = "max"
    finally:
        _solve_scalar_newton_jit._clear_cache()
