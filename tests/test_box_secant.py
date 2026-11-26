import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    BoxSecant,
    BoxSolveResult,
    Cholesky,
    LagrangeMultipliers,
    LevenbergMarquardt,
    SolverStatus,
    vmap_rows,
)
from nlls_gram.box_secant import _solve_box_secant_jit


@pytest.mark.parametrize("loop", ["while", "fori"])
@pytest.mark.parametrize("jit", [True, False])
def test_arities_senses_and_basic_regimes(loop, jit):
    s1 = BoxSecant(lambda x: 0.5 * (x - 0.3) ** 2, loop=loop)
    r1 = s1.solve(0.0, 1.0, jit=jit)
    assert isinstance(r1, BoxSolveResult)
    assert int(r1.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(float(r1.x), 0.3, atol=1e-5)
    np.testing.assert_allclose(float(r1.fun), 0.0, atol=1e-10)
    np.testing.assert_allclose(float(r1.grad), 0.0, atol=1e-5)

    s2 = BoxSecant(lambda x, a: -0.5 * a * (x - 0.7) ** 2 + 2.0, sense="max", loop=loop)
    r2 = s2.solve(0.0, 1.0, jnp.float32(3.0), jit=jit)
    assert int(r2.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(float(r2.x), 0.7, atol=1e-5)
    np.testing.assert_allclose(float(r2.fun), 2.0, atol=1e-6)
    np.testing.assert_allclose(float(r2.grad), 0.0, atol=1e-5)

    s3 = BoxSecant(
        lambda x, a, p: 0.5 * a * (x - p["target"]) ** 2, sense="min", loop=loop
    )
    r_lo = s3.solve(
        0.2, 1.0, jnp.float32(2.0), p={"target": jnp.float32(-0.5)}, jit=jit
    )
    assert int(r_lo.status) == int(SolverStatus.CONVERGED)
    assert float(r_lo.x) == pytest.approx(float(r_lo.lower), abs=0.0)
    np.testing.assert_allclose(float(r_lo.grad), 2.0 * (0.2 + 0.5), rtol=1e-5)

    r_hi = s3.solve(0.2, 1.0, jnp.float32(2.0), p={"target": jnp.float32(1.8)}, jit=jit)
    assert int(r_hi.status) == int(SolverStatus.CONVERGED)
    assert float(r_hi.x) == pytest.approx(float(r_hi.upper), abs=0.0)
    np.testing.assert_allclose(float(r_hi.grad), 2.0 * (1.0 - 1.8), rtol=1e-5)

    r_deg = s3.solve(
        0.5, 0.5, jnp.float32(2.0), p={"target": jnp.float32(0.1)}, jit=jit
    )
    assert int(r_deg.status) == int(SolverStatus.CONVERGED)
    assert int(r_deg.steps) == 0
    assert float(r_deg.x) == pytest.approx(0.5, abs=0.0)
    assert float(r_deg.bracket_width) == 0.0


def test_validation_and_immutability():
    with pytest.raises(ValueError, match="sense"):
        BoxSecant(lambda x: x**2, sense="saddle")
    with pytest.raises(ValueError, match="loop"):
        BoxSecant(lambda x: x**2, loop="scan")
    with pytest.raises(ValueError, match="1 to 3 positional"):
        BoxSecant(lambda: 0.0)

    s1 = BoxSecant(lambda x: x**2)
    with pytest.raises(ValueError, match="args was passed"):
        s1.solve(0.0, 1.0, 2.0)
    s2 = BoxSecant(lambda x, a: a * x**2)
    with pytest.raises(ValueError, match="p was passed"):
        s2.solve(0.0, 1.0, 2.0, p=1.0)
    with pytest.raises(ValueError, match="loop"):
        s2.solve(0.0, 1.0, 2.0, loop="bad")
    with pytest.raises(ValueError, match="num_starts"):
        s2.solve(0.0, 1.0, 2.0, num_starts=0)
    with pytest.raises(ValueError, match="num_starts"):
        s2.solve(0.0, 1.0, 2.0, num_starts=True)
    with pytest.raises(AttributeError, match="frozen after construction"):
        s2.sense = "max"


def test_linear_constant_concave_and_large_scale_objectives():
    # Purely linear objectives hit exact box endpoints in 1 step
    s_lin = BoxSecant(lambda x, _, p: p * x)
    r_pos = s_lin.solve(-2.0, 3.0, p=jnp.float32(4.0))
    assert int(r_pos.steps) == 1
    assert float(r_pos.x) == pytest.approx(-2.0, abs=0.0)
    r_neg = s_lin.solve(-2.0, 3.0, p=jnp.float32(-4.0))
    assert int(r_neg.steps) == 1
    assert float(r_neg.x) == pytest.approx(3.0, abs=0.0)

    # Constant objective converges in 0 steps with exact parameter derivative
    s_const = BoxSecant(lambda x, _, p: 0.0 * x + p**3)
    r_const = s_const.solve(0.0, 1.0, p=jnp.float32(2.0))
    assert int(r_const.steps) == 0
    assert int(r_const.status) == int(SolverStatus.CONVERGED)
    grad_const = jax.grad(lambda p: s_const.solve(0.0, 1.0, p=p).fun)(jnp.float32(2.0))
    np.testing.assert_allclose(float(grad_const), 12.0, rtol=1e-5)

    # Concave objectives avoid interior/boundary local maxima and hit global minimum
    s_conc = BoxSecant(lambda x, m: -((x - m) ** 2))
    for m, expected_x in [
        (0.4, 1.0),
        (0.6, 0.0),
        (0.625, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ]:
        res = s_conc.solve(0.0, 1.0, jnp.float32(m))
        assert int(res.status) == int(SolverStatus.CONVERGED)
        assert float(res.x) == pytest.approx(expected_x, abs=0.0)
        # Warm-starting right on the concave peak still escapes to a boundary minimum
        res_ws = s_conc.solve(0.0, 1.0, jnp.float32(m), x0=jnp.float32(m))
        assert int(res_ws.status) == int(SolverStatus.CONVERGED)
        assert float(res_ws.x) in (0.0, 1.0)
        assert float(res_ws.fun) <= -0.15

    # Large-coordinate float32 problem uses relative ULP scaling cleanly
    s_large = BoxSecant(lambda x: (x - jnp.float32(15432.5)) ** 2)
    r_large = s_large.solve(jnp.float32(10000.0), jnp.float32(20000.0))
    assert int(r_large.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(float(r_large.x), 15432.5, rtol=1e-5)


def test_warm_start_zero_step_convergence_and_clipping():
    solver = BoxSecant(lambda x, _, p: 0.5 * (x - p) ** 2 + 0.05 * jnp.cos(x - p))

    # Interior exact warm start exits in 0 loop steps
    r_int = solver.solve(0.0, 1.0, x0=jnp.float32(0.4), p=jnp.float32(0.4))
    assert int(r_int.status) == int(SolverStatus.CONVERGED)
    assert int(r_int.steps) == 0
    np.testing.assert_allclose(float(r_int.x), 0.4, atol=1e-6)

    # Boundary warm start (including out-of-box x0 clipped to lower) exits in 0 steps
    r_bnd = solver.solve(0.0, 1.0, x0=jnp.float32(-3.0), p=jnp.float32(-0.5))
    assert int(r_bnd.status) == int(SolverStatus.CONVERGED)
    assert int(r_bnd.steps) == 0
    assert float(r_bnd.x) == 0.0


@pytest.mark.parametrize("use_x64", [False, True])
def test_implicit_ad_first_and_second_order_and_moving_bounds(use_x64):
    with jax.enable_x64(use_x64):
        f32_res = BoxSecant(lambda x, _, p: 0.5 * (x - p) ** 2).solve(
            0.0, 1.0, p=jnp.float32(0.4)
        )
        assert f32_res.x.dtype == jnp.float32
        dtype = jnp.float64 if use_x64 else jnp.float32
        rtol = 1e-9 if use_x64 else 2e-4
        atol = 1e-10 if use_x64 else 2e-4

        def obj(x, a, p):
            return 0.5 * p["c"] * (x - p["target"]) ** 2 + a * jnp.sin(x * p["freq"])

        solver = BoxSecant(obj, sense="min")
        a_val = jnp.asarray(0.15, dtype=dtype)

        def solve_vec(v):
            p_d = {"c": v[0], "target": v[1], "freq": v[2], "tag": jnp.int32(5)}
            res = solver.solve(
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
                a_val,
                p=p_d,
            )
            return jnp.stack([res.x, res.fun, res.grad])

        v0 = jnp.array([3.0, 0.45, 1.2], dtype=dtype)
        jf = jax.jacfwd(solve_vec)(v0)
        jr = jax.jacrev(solve_vec)(v0)
        np.testing.assert_allclose(jf, jr, rtol=rtol, atol=atol)
        # At interior optimum, grad == 0 identically across p -> d(grad)/dp == 0
        np.testing.assert_allclose(jf[2], np.zeros(3), atol=atol)

        # Finite-difference check on x*(v) and fun(v)
        eps_fd = 1e-6 if use_x64 else 1e-3
        for k in range(3):
            e = jnp.zeros(3, dtype=dtype).at[k].set(eps_fd)
            fd = (solve_vec(v0 + e) - solve_vec(v0 - e)) / (2.0 * eps_fd)
            np.testing.assert_allclose(
                jf[:2, k], fd[:2], rtol=5e-3 if not use_x64 else 1e-6, atol=1e-3
            )

        # Second-order AD: closed-form x*(p) = p / (1 + p^2) on [-2, 2]
        quad_solver = BoxSecant(lambda x, _, p: 0.5 * (1.0 + p**2) * x**2 - p * x)
        p_s = jnp.asarray(0.4, dtype=dtype)
        h_x = jax.hessian(
            lambda p: (
                quad_solver.solve(
                    jnp.asarray(-2.0, dtype=dtype), jnp.asarray(2.0, dtype=dtype), p=p
                ).x
            )
        )(p_s)
        # d/dp [p/(1+p^2)] = (1 - p^2)/(1+p^2)^2; d^2/dp^2 = 2p(p^2 - 3)/(1+p^2)^3
        p_f = float(p_s)
        expected_h = 2.0 * p_f * (p_f**2 - 3.0) / ((1.0 + p_f**2) ** 3)
        np.testing.assert_allclose(float(h_x), expected_h, rtol=rtol, atol=atol)

        # Moving bounds: lower(q) = 0.2 + 2*q, upper(q) = 0.8 + 3*q
        b_solver = BoxSecant(lambda x, _, q: 0.5 * (x - q[0]) ** 2)

        def bound_map(q):
            lo = 0.2 + 2.0 * q[1]
            hi = 0.8 + 3.0 * q[2]
            r = b_solver.solve(lo, hi, p=q)
            return jnp.stack([r.x, r.fun, r.grad])

        # Active lower bound: x* = lo = 0.2, fun = 0.5*(lo - (-1))^2, grad = lo - (-1)
        q_lo = jnp.array([-1.0, 0.0, 0.0], dtype=dtype)
        j_lo = jax.jacrev(bound_map)(q_lo)
        np.testing.assert_allclose(j_lo[0], [0.0, 2.0, 0.0], atol=atol)
        np.testing.assert_allclose(j_lo[1], [-1.2, 2.4, 0.0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(j_lo[2], [-1.0, 2.0, 0.0], rtol=rtol, atol=atol)

        # Active upper bound: x* = hi = 0.8, fun = 0.5*(hi - 2)^2, grad = hi - 2
        q_hi = jnp.array([2.0, 0.0, 0.0], dtype=dtype)
        j_hi = jax.jacrev(bound_map)(q_hi)
        np.testing.assert_allclose(j_hi[0], [0.0, 0.0, 3.0], atol=atol)
        np.testing.assert_allclose(j_hi[1], [1.2, 0.0, -3.6], rtol=rtol, atol=atol)
        np.testing.assert_allclose(j_hi[2], [-1.0, 0.0, 3.0], rtol=rtol, atol=atol)


def test_has_aux_pytree_and_implicit_ad():
    def obj_aux(x, scale, p):
        val = 0.5 * scale * (x - p["t"]) ** 2
        aux = {
            "weighted": x**2 + 3.0 * p["s"],
            "tag": jnp.int32(9),
            "positive": x > 0.0,
            "empty": None,
        }
        return val, aux

    solver = BoxSecant(obj_aux, has_aux=True)
    p0 = {"t": jnp.float32(0.35), "s": jnp.float32(2.0)}

    res = solver.solve(0.0, 1.0, jnp.float32(4.0), p=p0)
    assert int(res.status) == int(SolverStatus.CONVERGED)
    assert int(res.aux["tag"]) == 9
    assert bool(res.aux["positive"]) is True
    assert res.aux["empty"] is None
    np.testing.assert_allclose(
        float(res.aux["weighted"]), 0.35**2 + 6.0, rtol=1e-5, atol=1e-5
    )

    def pair_fn(v):
        r = solver.solve(0.0, 1.0, jnp.float32(4.0), p={"t": v[0], "s": v[1]})
        return jnp.stack([r.x, r.aux["weighted"]])

    v0 = jnp.array([0.35, 2.0], dtype=jnp.float32)
    jf = jax.jacfwd(pair_fn)(v0)
    jr = jax.jacrev(pair_fn)(v0)
    expected = np.array([[1.0, 0.0], [2.0 * 0.35, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(jf, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(jr, expected, rtol=1e-4, atol=1e-5)

    # Batched has_aux under vmap
    scales = jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32)
    b_res = jax.vmap(lambda sc: solver.solve(0.0, 1.0, sc, p=p0))(scales)
    assert b_res.aux["weighted"].shape == (3,)
    np.testing.assert_allclose(b_res.x, np.full(3, 0.35), atol=1e-5)


def test_failed_lanes_mask_tangents_and_cotangents_under_vmap():
    # Lane 0 evaluates sqrt(-5) -> NONFINITE while lane 1 converges at x* = p^2
    solver = BoxSecant(lambda x, shift, p: 0.5 * (jnp.sqrt(x + shift) - p) ** 2)
    shifts = jnp.array([-5.0, 0.0], dtype=jnp.float32)
    p0 = jnp.float32(0.5)

    def run_batch(p):
        return jax.vmap(lambda s: solver.solve(0.0, 1.0, s, p=p))(shifts)

    res, tan = jax.jvp(run_batch, (p0,), (jnp.float32(1.0),))
    assert int(res.status[0]) == int(SolverStatus.NONFINITE)
    assert int(res.status[1]) == int(SolverStatus.CONVERGED)
    assert float(tan.x[0]) == 0.0
    np.testing.assert_allclose(float(tan.x[1]), 2.0 * 0.5, rtol=1e-4)

    grad_p = jax.grad(lambda p: jnp.sum(run_batch(p).x))(p0)
    assert jnp.isfinite(grad_p)
    np.testing.assert_allclose(float(grad_p), 1.0, rtol=1e-4)

    # MAX_STEPS masking: max_steps_is_success=False zeroes tangent, True differentiates
    slow_solver = BoxSecant(lambda x, _, p: jnp.cosh(x - p) + 0.2 * (x - p) ** 4)

    def run_trunc(p, ok):
        return slow_solver.solve(
            -5.0,
            5.0,
            p=p,
            xtol=0.0,
            rtol=0.0,
            gtol=0.0,
            max_steps=1,
            max_steps_is_success=ok,
        )

    r_trunc = run_trunc(jnp.float32(0.3), False)
    assert int(r_trunc.status) == int(SolverStatus.MAX_STEPS)
    g_masked = jax.grad(lambda p: run_trunc(p, False).x)(jnp.float32(0.3))
    g_unmasked = jax.grad(lambda p: run_trunc(p, True).x)(jnp.float32(0.3))
    assert float(g_masked) == 0.0
    assert float(g_unmasked) != 0.0


def test_nested_vmap_and_vmap_rows_inside_outer_lm():
    # 2D nested vmap: (M=4 firms) x (K=6 sub-boxes) over a multi-modal objective
    solver = BoxSecant(
        lambda x, a, p: jnp.cos(4.0 * x - a) - 0.3 * (x - p) ** 2, sense="max"
    )
    edges = jnp.linspace(0.0, 3.0, 7, dtype=jnp.float32)
    lo_sub, hi_sub = edges[:-1], edges[1:]
    a_firms = jnp.array([0.2, 0.8, 1.5, 2.4], dtype=jnp.float32)
    p0 = jnp.float32(1.1)

    solve_2d = jax.jit(
        jax.vmap(
            lambda a_i: jax.vmap(lambda lo, hi: solver.solve(lo, hi, a_i, p=p0))(
                lo_sub, hi_sub
            )
        )
    )
    res_2d = solve_2d(a_firms)
    assert res_2d.x.shape == (4, 6)
    assert jnp.all(res_2d.status == int(SolverStatus.CONVERGED))

    # Bilevel equilibrium: outer LevenbergMarquardt + vmap_rows wrapping inner BoxSecant
    inner = BoxSecant(
        lambda x, w, theta: 0.5 * (x - (w + 0.5 * theta[0])) ** 2 + 0.1 * theta[1] * x
    )
    ws = jnp.linspace(-0.2, 0.8, 8, dtype=jnp.float32)
    targets = jnp.clip(ws + 0.5 * 0.6 - 0.1 * (-0.4), 0.0, 1.0)

    def row_resid(w, target, theta):
        x_opt = inner.solve(0.0, 1.0, w, p=theta).x
        return x_opt - target

    batched_rows = vmap_rows(row_resid, in_axes=(0, 0, None))
    outer = LevenbergMarquardt(
        lambda th, args: batched_rows(args[0], args[1], th),
        linear_solver=Cholesky(form="normal"),
    )
    out_res = outer.solve(
        jnp.zeros(2, dtype=jnp.float32), (ws, targets), atol=1e-5, max_steps=25
    )
    assert int(out_res.status) == int(SolverStatus.CONVERGED)
    assert float(out_res.info.loss) < 1e-9

    # Closed-over outer BatchTracer (vmap / lax.map(batch_size=...)) + inner vmap + AD
    edges_sub = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)

    def closed_outer(shift, p):
        slv = BoxSecant(lambda x, _, p_in: 0.5 * (x - (shift + p_in)) ** 2)
        return jax.vmap(lambda lo, hi: slv.solve(lo, hi, p=p).x)(
            edges_sub[:-1], edges_sub[1:]
        )

    shifts = jnp.array([0.1, 0.4, 0.8], dtype=jnp.float32)
    x_closed = jax.jit(jax.vmap(closed_outer, in_axes=(0, None)))(
        shifts, jnp.float32(0.05)
    )
    assert x_closed.shape == (3, 4)
    g_closed = jax.jit(
        jax.grad(
            lambda p: jnp.sum(jax.vmap(closed_outer, in_axes=(0, None))(shifts, p))
        )
    )(jnp.float32(0.05))
    np.testing.assert_allclose(float(g_closed), 3.0, rtol=1e-4)


def test_equal_solvers_and_traced_tolerances_share_one_compilation():
    _solve_box_secant_jit._clear_cache()
    try:
        fn = lambda x, a, p: 0.5 * a * (x - p) ** 2  # noqa: E731
        for target in (0.2, 0.5, 0.8):
            BoxSecant(fn).solve(0.0, 1.0, jnp.float32(2.0), p=jnp.float32(target))
        assert _solve_box_secant_jit._cache_size() == 1

        solver = BoxSecant(fn)
        solver.solve(
            -1.0,
            2.0,
            jnp.float32(3.0),
            p=jnp.float32(0.4),
            xtol=1e-5,
            gtol=1e-5,
            max_steps=16,
        )
        solver.solve(
            0.1, 0.9, jnp.float32(1.5), p=jnp.float32(0.6), rtol=1e-6, max_steps=40
        )
        assert _solve_box_secant_jit._cache_size() == 1
    finally:
        _solve_box_secant_jit._clear_cache()


def test_multi_start_skiba_basins_vmap_and_implicit_ad():
    # Narrow corner maximum at x = 0 across a Skiba valley plus a wide peak near p
    def skiba_rhs(x, tilt, p):
        inaction_bump = 0.25 * jnp.exp(-((x / 0.005) ** 2))
        active_peak = (0.20 + tilt) * jnp.exp(-(((x - p) / 0.08) ** 2))
        return inaction_bump + active_peak - 0.30 * x

    solver = BoxSecant(skiba_rhs, sense="max")
    p0 = jnp.float32(0.16)

    # Single-start misses narrow x=0 bump and lands on active_peak
    r_single = solver.solve(0.0, 0.25, jnp.float32(0.0), p=p0, num_starts=1)
    assert float(r_single.x) > 0.10

    # num_starts=8 finds the global maximum at x=0 even when warm-started at p0
    r_multi = solver.solve(0.0, 0.25, jnp.float32(0.0), x0=p0, p=p0, num_starts=8)
    assert int(r_multi.status) == int(SolverStatus.CONVERGED)
    assert float(r_multi.x) == 0.0
    assert float(r_multi.fun) > float(r_single.fun)

    # At tilt = 0.18, the interior active peak beats x = 0
    tilts = jnp.array([-0.02, 0.18], dtype=jnp.float32)

    def batch_multi(lo, p):
        return jax.vmap(lambda t: solver.solve(lo, 0.25, t, p=p, num_starts=8))(tilts)

    b_res = jax.jit(batch_multi)(jnp.float32(0.0), p0)
    assert jnp.all(b_res.status == int(SolverStatus.CONVERGED))
    assert float(b_res.x[0]) == 0.0
    assert float(b_res.x[1]) > 0.12

    # Implicit AD flows through lower on lane 0 and interior p-peak on lane 1
    j_lo, j_p = jax.jacrev(lambda lo, p: batch_multi(lo, p).x, argnums=(0, 1))(
        jnp.float32(0.0), p0
    )
    np.testing.assert_allclose(j_lo, [1.0, 0.0], atol=1e-4)
    assert abs(float(j_p[0])) == 0.0
    np.testing.assert_allclose(float(j_p[1]), 1.0, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("use_x64", [False, True])
def test_lagrange_multipliers_values_ad_and_nonconvex_corner(use_x64):
    with jax.enable_x64(use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        rtol = 1e-9 if use_x64 else 2e-4
        atol = 1e-10 if use_x64 else 2e-4

        # Minimization and maximization corner multipliers (>= 0 on both corners)
        s_min = BoxSecant(
            lambda x, _, q: 0.5 * (q[2] ** 2) * (x - q[3]) ** 2 + q[4] * jnp.sin(x),
            sense="min",
        )
        s_max = BoxSecant(
            lambda x, _, q: -(0.5 * (q[2] ** 2) * (x - q[3]) ** 2 + q[4] * jnp.sin(x)),
            sense="max",
        )

        # Interior optimum: both multipliers and their derivatives are exact 0
        q_int = jnp.array([0.0, 1.0, 2.0, 0.5, 0.1], dtype=dtype)
        for slv in (s_min, s_max):
            r_int = slv.solve(q_int[0], q_int[1], p=q_int)
            assert isinstance(r_int.lm, LagrangeMultipliers)
            assert float(r_int.lm.lower) == 0.0
            assert float(r_int.lm.upper) == 0.0
            j_int = jax.jacrev(
                lambda q, sv=slv: jnp.stack(
                    [
                        sv.solve(q[0], q[1], p=q).lm.lower,
                        sv.solve(q[0], q[1], p=q).lm.upper,
                    ]
                )
            )(q_int)
            np.testing.assert_array_equal(np.asarray(j_int), np.zeros((2, 5)))

        # Active lower bound (x* = lo): lm.lower = c^2*(lo - t) + w*cos(lo) > 0
        q_lo = jnp.array([0.2, 0.9, 2.5, -0.4, 0.3], dtype=dtype)
        exact_lo = lambda q: (q[2] ** 2) * (q[0] - q[3]) + q[4] * jnp.cos(q[0])  # noqa: E731
        for slv in (s_min, s_max):
            r_lo = slv.solve(q_lo[0], q_lo[1], p=q_lo)
            np.testing.assert_allclose(
                float(r_lo.lm.lower), float(exact_lo(q_lo)), rtol=rtol, atol=atol
            )
            assert float(r_lo.lm.upper) == 0.0
            lm_lo_fn = lambda q, sv=slv: sv.solve(q[0], q[1], p=q).lm.lower  # noqa: E731
            np.testing.assert_allclose(
                jax.jacfwd(lm_lo_fn)(q_lo),
                jax.jacfwd(exact_lo)(q_lo),
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_allclose(
                jax.jacrev(lm_lo_fn)(q_lo),
                jax.jacrev(exact_lo)(q_lo),
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_allclose(
                jax.hessian(lm_lo_fn)(q_lo),
                jax.hessian(exact_lo)(q_lo),
                rtol=rtol,
                atol=atol,
            )

        # Active upper bound (x* = hi): lm.upper = -(c^2*(hi - t) + w*cos(hi)) > 0
        q_hi = jnp.array([0.2, 0.9, 2.5, 1.8, -0.2], dtype=dtype)
        exact_hi = lambda q: -((q[2] ** 2) * (q[1] - q[3]) + q[4] * jnp.cos(q[1]))  # noqa: E731
        for slv in (s_min, s_max):
            r_hi = slv.solve(q_hi[0], q_hi[1], p=q_hi)
            assert float(r_hi.lm.lower) == 0.0
            np.testing.assert_allclose(
                float(r_hi.lm.upper), float(exact_hi(q_hi)), rtol=rtol, atol=atol
            )
            lm_hi_fn = lambda q, sv=slv: sv.solve(q[0], q[1], p=q).lm.upper  # noqa: E731
            np.testing.assert_allclose(
                jax.jacrev(lm_hi_fn)(q_hi),
                jax.jacrev(exact_hi)(q_hi),
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_allclose(
                jax.hessian(lm_hi_fn)(q_hi),
                jax.hessian(exact_hi)(q_hi),
                rtol=rtol,
                atol=atol,
            )

        # Degenerate point interval lo == hi splits net multiplier cleanly by sign
        r_deg_pos = s_min.solve(
            jnp.asarray(0.5, dtype), jnp.asarray(0.5, dtype), p=q_lo
        )
        assert float(r_deg_pos.lm.lower) > 0.0 and float(r_deg_pos.lm.upper) == 0.0
        r_deg_neg = s_min.solve(
            jnp.asarray(0.5, dtype), jnp.asarray(0.5, dtype), p=q_hi
        )
        assert float(r_deg_neg.lm.lower) == 0.0 and float(r_deg_neg.lm.upper) > 0.0

        # Nonconvex-near-corner polynomial hits exact x* = 0 with exact LM gradient
        s_inf = BoxSecant(lambda x, _, p: p * x - x**2 + (5.0 / 3.0) * x**4)
        p_inf = jnp.asarray(0.5, dtype=dtype)
        lo0, hi1 = jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype)
        r_inf = s_inf.solve(lo0, hi1, p=p_inf)
        assert int(r_inf.status) == int(SolverStatus.CONVERGED)
        assert int(r_inf.steps) <= 3
        assert float(r_inf.x) == 0.0
        np.testing.assert_allclose(float(r_inf.lm.lower), 0.5, rtol=rtol, atol=atol)
        dx_dp = jax.grad(lambda p: s_inf.solve(lo0, hi1, p=p).x)(p_inf)
        dlm_dp = jax.grad(lambda p: s_inf.solve(lo0, hi1, p=p).lm.lower)(p_inf)
        assert float(dx_dp) == 0.0
        np.testing.assert_allclose(float(dlm_dp), 1.0, rtol=rtol, atol=atol)
        # Envelope Hessian on interior matches closed form with fun_dot = f_p_dot
        quad_s = BoxSecant(lambda x, _, p: 0.5 * (x - p) ** 2 + 0.25 * p**4)
        h_fun = jax.hessian(lambda p: quad_s.solve(lo0, hi1, p=p).fun)(
            jnp.asarray(0.4, dtype=dtype)
        )
        np.testing.assert_allclose(float(h_fun), 3.0 * (0.4**2), rtol=rtol, atol=atol)


def test_review_edge_cases_gtol_band_collapsed_box_and_elementwise_ad():
    # 1. Clamped multiplier inside gtol band has zero derivative (not -1 / +1)
    s_band = BoxSecant(lambda x, _, p: 0.5 * (x - p) ** 2)
    r_band = s_band.solve(0.0, 1.0, x0=jnp.float32(0.0), p=jnp.float32(1e-6))
    assert float(r_band.lm.lower) == 0.0
    assert (
        float(
            jax.grad(lambda p: s_band.solve(0.0, 1.0, x0=0.0, p=p).lm.lower)(
                jnp.float32(1e-6)
            )
        )
        == 0.0
    )

    # 2. Collapsed box opening upward (lo == hi == 0, target = 1 -> g_h < 0)
    s_open = BoxSecant(lambda x: 0.5 * (x - 1.0) ** 2)
    _, tan_open = jax.jvp(
        lambda u: s_open.solve(jnp.float32(0.0), u),
        (jnp.float32(0.0),),
        (jnp.float32(1.0),),
    )
    np.testing.assert_allclose(float(tan_open.x), 1.0, atol=1e-6)
    np.testing.assert_allclose(float(tan_open.lm.upper), -1.0, atol=1e-6)

    # 3. Infinite boundary f_xx with fixed bound (lo_dot == 0) avoids inf * 0 = NaN
    s_frac = BoxSecant(lambda x, _, p: x**1.5 + p * x)
    j_frac = jax.jacfwd(lambda p: s_frac.solve(0.0, 1.0, p=p).lm.lower)(
        jnp.float32(1.0)
    )
    np.testing.assert_allclose(float(j_frac), 1.0, atol=1e-6)

    # 4. Warm-start tie on large-offset linear objective picks KKT-valid corner
    s_tie = BoxSecant(lambda x: jnp.float32(1e6) + x)
    r_tie = s_tie.solve(0.0, 1.0, x0=jnp.float32(1.0 / 32.0))
    assert float(r_tie.x) == 0.0
    np.testing.assert_allclose(float(r_tie.lm.lower), 1.0, atol=1e-5)

    # 5. Elementwise batch (lower/upper shape (3,)) with scalar p supports JVP & VJP
    lo_vec = jnp.array([0.0, 0.0, 0.6], dtype=jnp.float32)
    hi_vec = jnp.array([0.2, 1.0, 1.0], dtype=jnp.float32)
    g_elem = jax.grad(lambda p: jnp.sum(s_band.solve(lo_vec, hi_vec, p=p).x))(
        jnp.float32(0.4)
    )
    np.testing.assert_allclose(float(g_elem), 1.0, atol=1e-5)

    # 6. Higher-order reverse AD (grad(grad)) stays finite with a failed NaN lane
    s_nan = BoxSecant(lambda x, a, p: 0.5 * (x - jnp.sqrt(a + p)) ** 2)
    a_lanes = jnp.array([-2.0, 1.0], dtype=jnp.float32)
    gg = jax.grad(
        jax.grad(
            lambda p: jnp.sum(
                jax.vmap(lambda a: s_nan.solve(0.0, 5.0, a, p=p).x)(a_lanes)
            )
        )
    )(jnp.float32(1.0))
    np.testing.assert_allclose(float(gg), -0.25 / (2.0**1.5), rtol=1e-4, atol=1e-5)
