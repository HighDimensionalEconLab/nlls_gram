import jax
import jax.numpy as jnp
import pytest

from nlls_gram import LMStatus, SquareLevenbergMarquardt


def test_scalar_root_matches_closed_form():
    solver = SquareLevenbergMarquardt(lambda x, args, p: x**2 - p)
    result = solver.solve(jnp.array([1.0]), p=jnp.asarray(4.0), max_steps=50)

    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x, jnp.array([2.0]), atol=1e-5)
    assert float(result.residual_norm) < 1e-6
    assert int(result.steps) < 50


def test_two_by_two_system_matches_analytic_root():
    # x0 + x1 = p, x0 * x1 = 1 with the root branch near (p, 1/p) for large p.
    def residual(x, _, p):
        return jnp.array([x[0] + x[1] - p, x[0] * x[1] - 1.0])

    p = jnp.asarray(3.0)
    solver = SquareLevenbergMarquardt(residual)
    result = solver.solve(jnp.array([2.5, 0.5]), p=p, max_steps=50)

    root = (p + jnp.sqrt(p**2 - 4.0)) / 2.0
    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x, jnp.array([root, 1.0 / root]), atol=1e-5)


def test_four_by_four_nonlinear_system():
    W = 0.1 * jax.random.normal(jax.random.PRNGKey(62), (4, 4))
    b = jax.random.normal(jax.random.PRNGKey(63), (4,))

    def residual(z, _, p):
        return z + jnp.tanh(W @ z) - p

    solver = SquareLevenbergMarquardt(residual)
    result = solver.solve(jnp.zeros(4), p=b, max_steps=50)

    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x + jnp.tanh(W @ result.x), b, rtol=1e-5, atol=1e-5)


def test_pytree_x_and_dae_style_pytree_p():
    # Semi-explicit index-1 DAE stage shape: g(y, z, t, args, params) = 0 with
    # p = (y, t, params); analytic root z0 = y * t / (1 + params),
    # z1 = params * z0.
    def residual(z, args, p):
        y, t, params = p
        return jnp.array([z["a"] + z["b"] - y * t, params * z["a"] - z["b"]])

    solver = SquareLevenbergMarquardt(residual)
    y, t, params = jnp.asarray(2.0), jnp.asarray(0.5), jnp.asarray(3.0)
    result = solver.solve(
        {"a": jnp.zeros(()), "b": jnp.zeros(())}, p=(y, t, params), max_steps=50
    )

    a_expected = y * t / (1.0 + params)
    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x["a"], a_expected, atol=1e-6)
    assert jnp.allclose(result.x["b"], params * a_expected, atol=1e-6)


def test_warm_start_at_root_exits_in_zero_steps():
    solver = SquareLevenbergMarquardt(lambda x, args, p: x**2 - p)
    p = jnp.asarray(4.0)
    first = solver.solve(jnp.array([1.0]), p=p, max_steps=50)
    warm = solver.solve(first.x, p=p, max_steps=50)

    assert int(warm.status) == LMStatus.CONVERGED
    assert int(warm.steps) == 0
    assert jnp.allclose(warm.x, first.x)


def test_atol_defaults_and_explicit_value():
    solver = SquareLevenbergMarquardt(lambda x: x - 1.0)
    # float32 default atol = 1e-6: one Newton step lands essentially exactly.
    result = solver.solve(jnp.zeros(1), max_steps=20)
    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.residual_norm) < 1e-6

    # A loose explicit atol converges immediately at the starting point.
    loose = solver.solve(jnp.array([0.5]), max_steps=20, atol=1.0)
    assert int(loose.status) == LMStatus.CONVERGED
    assert int(loose.steps) == 0


def test_gtol_stops_at_stationary_non_root():
    # r(x) = x^2 + 1 has no real root; x = 0 is stationary for the loss.
    # With defaults the solver reports MAX_STEPS; opting into gtol reports
    # CONVERGED at the stationary point -- which is why a DAE caller keeps
    # gtol and xtol at 0 and trusts the residual criterion.
    solver = SquareLevenbergMarquardt(lambda x: x**2 + 1.0)
    default = solver.solve(jnp.array([0.5]), max_steps=30)
    with_gtol = solver.solve(jnp.array([0.5]), max_steps=30, gtol=1e-3)

    assert int(default.status) == LMStatus.MAX_STEPS
    assert float(default.residual_norm) == pytest.approx(1.0, abs=1e-3)
    assert int(with_gtol.status) == LMStatus.CONVERGED
    assert float(with_gtol.residual_norm) == pytest.approx(1.0, abs=1e-2)


def test_non_square_residual_raises():
    with pytest.raises(ValueError, match="square system"):
        SquareLevenbergMarquardt(lambda x: jnp.concatenate((x, x))).solve(jnp.ones(2))
    with pytest.raises(ValueError, match="square system"):
        SquareLevenbergMarquardt(lambda x: x[:1]).solve(jnp.ones(2))


def test_mismatched_x_and_residual_dtype_raises():
    # The loop carry requires one dtype; a residual that promotes (or an x0
    # in a different dtype) is rejected at trace time in both loop modes.
    solver = SquareLevenbergMarquardt(lambda x: x.astype(jnp.bfloat16) - 1.0)
    with pytest.raises(ValueError, match="share a dtype"):
        solver.solve(jnp.zeros(1), max_steps=5)
    with pytest.raises(ValueError, match="share a dtype"):
        solver.solve(jnp.zeros(1), max_steps=5, jit=False)


def test_residual_arity_conventions():
    ts = jnp.array([2.0])

    one = SquareLevenbergMarquardt(lambda x: x - 1.0)
    two = SquareLevenbergMarquardt(lambda x, args: x - args)
    three = SquareLevenbergMarquardt(lambda x, args, p: x - args * p)

    assert int(one.solve(jnp.zeros(1), max_steps=20).status) == LMStatus.CONVERGED
    assert jnp.allclose(two.solve(jnp.zeros(1), ts, max_steps=20).x, ts)
    assert jnp.allclose(
        three.solve(jnp.zeros(1), ts, p=jnp.asarray(2.0), max_steps=20).x, 2.0 * ts
    )

    with pytest.raises(ValueError, match="takes only .x."):
        one.solve(jnp.zeros(1), ts)
    with pytest.raises(ValueError, match="takes no p argument"):
        two.solve(jnp.zeros(1), ts, p=jnp.asarray(1.0))
    with pytest.raises(ValueError, match="1 to 3 positional arguments"):
        SquareLevenbergMarquardt(lambda: jnp.zeros(1))


def test_solve_options_must_be_valid():
    solver = SquareLevenbergMarquardt(lambda x: x)
    with pytest.raises(ValueError, match="max_steps must be a positive int"):
        solver.solve(jnp.zeros(1), max_steps=0)
    with pytest.raises(ValueError, match="max_steps must be a positive int"):
        solver.solve(jnp.zeros(1), max_steps=1.5)
    with pytest.raises(ValueError, match="atol must be nonnegative"):
        solver.solve(jnp.zeros(1), atol=-1.0)
    with pytest.raises(ValueError, match="gtol must be nonnegative"):
        solver.solve(jnp.zeros(1), gtol=-1.0)
    with pytest.raises(ValueError, match="xtol must be nonnegative"):
        solver.solve(jnp.zeros(1), xtol=-1.0)
    with pytest.raises(ValueError, match="init_damping must be positive"):
        SquareLevenbergMarquardt(lambda x: x, init_damping=0.0)
    with pytest.raises(ValueError, match="damping_decrease must be positive"):
        SquareLevenbergMarquardt(lambda x: x, damping_decrease=0.0)
    with pytest.raises(ValueError, match="damping_increase must be positive"):
        SquareLevenbergMarquardt(lambda x: x, damping_increase=0.0)


def test_nonfinite_residual_statuses():
    # Nonfinite at x0 reports NONFINITE immediately.
    nan_at_start = SquareLevenbergMarquardt(lambda x: jnp.full_like(x, jnp.nan))
    result = nan_at_start.solve(jnp.zeros(1), max_steps=10)
    assert int(result.status) == LMStatus.NONFINITE
    assert int(result.steps) == 0

    # Every candidate away from x0 is NaN: steps are rejected (damping grows)
    # and the budget runs out without a host exception.
    def rejects_everything(x):
        return jnp.where(x[0] == 0.0, x + 1.0, jnp.full_like(x, jnp.nan))

    stuck = SquareLevenbergMarquardt(rejects_everything).solve(
        jnp.zeros(1), max_steps=10
    )
    assert int(stuck.status) == LMStatus.MAX_STEPS
    assert jnp.allclose(stuck.x, jnp.zeros(1))
    assert float(stuck.residual_norm) == pytest.approx(1.0)


def test_singular_at_root_jacobian_stays_finite():
    # x^3 = 0 has a root with singular Jacobian; the damped step degrades
    # gracefully (slow progress, no exception, finite output).
    solver = SquareLevenbergMarquardt(lambda x: x**3)
    result = solver.solve(jnp.array([1.0]), max_steps=30)

    assert bool(jnp.all(jnp.isfinite(result.x)))
    assert float(result.residual_norm) < 1.0


def test_max_steps_one_reports_max_steps():
    solver = SquareLevenbergMarquardt(lambda x: jnp.tanh(x) - 0.9)
    result = solver.solve(jnp.zeros(1), max_steps=1, atol=1e-7)
    assert int(result.status) == LMStatus.MAX_STEPS
    assert int(result.steps) == 1


def test_jit_and_python_loops_agree_and_vmap_matches_loop():
    W = 0.1 * jax.random.normal(jax.random.PRNGKey(64), (3, 3))

    def residual(z, _, p):
        return z + jnp.tanh(W @ z) - p

    solver = SquareLevenbergMarquardt(residual)
    ps = jax.random.normal(jax.random.PRNGKey(65), (4, 3))
    z0 = jnp.zeros(3)

    jit_result = solver.solve(z0, p=ps[0], max_steps=50)
    python_result = solver.solve(z0, p=ps[0], max_steps=50, jit=False)
    assert int(jit_result.status) == int(python_result.status)
    assert int(jit_result.steps) == int(python_result.steps)
    assert jnp.allclose(jit_result.x, python_result.x, atol=1e-6)

    batched = jax.vmap(lambda p: solver.solve(z0, p=p, max_steps=50))(ps)
    for i in range(ps.shape[0]):
        lane = solver.solve(z0, p=ps[i], max_steps=50)
        assert int(batched.status[i]) == int(lane.status)
        assert jnp.allclose(batched.x[i], lane.x, atol=1e-5)


def test_linear_system_jvp_and_vjp_match_closed_form():
    A = jnp.array([[2.0, 0.5], [-0.3, 1.5]])
    B = jnp.array([[1.0, -0.5, 0.2], [0.3, 1.0, -0.7]])

    def residual(x, _, p):
        return A @ x - B @ p

    solver = SquareLevenbergMarquardt(residual)

    def solved_x(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=50).x

    p = jnp.array([1.0, -0.5, 0.25])
    p_dot = jnp.array([0.2, -0.1, 0.3])
    x_bar = jnp.array([0.7, -0.4])

    x, x_dot = jax.jvp(solved_x, (p,), (p_dot,))
    _, pullback = jax.vjp(solved_x, p)
    (p_bar,) = pullback(x_bar)

    assert jnp.allclose(x, jnp.linalg.solve(A, B @ p), atol=1e-5)
    assert jnp.allclose(x_dot, jnp.linalg.solve(A, B @ p_dot), atol=1e-6)
    assert jnp.allclose(p_bar, B.T @ jnp.linalg.solve(A.T, x_bar), atol=1e-6)
    # JVP/VJP dot-pairing identity.
    assert jnp.allclose(x_dot @ x_bar, p_dot @ p_bar, atol=1e-6)


def test_dae_style_implicit_tangent_matches_analytic():
    # zdot = -g_z^{-1} (g_y ydot + g_t tdot + g_theta thetadot) for the
    # analytic root z = (y t / (1 + theta), theta y t / (1 + theta)).
    def residual(z, args, p):
        y, t, theta = p
        return jnp.array([z[0] + z[1] - y * t, theta * z[0] - z[1]])

    solver = SquareLevenbergMarquardt(residual)

    def solved_z(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=50).x

    def analytic_z(p):
        y, t, theta = p
        z0 = y * t / (1.0 + theta)
        return jnp.array([z0, theta * z0])

    p = (jnp.asarray(2.0), jnp.asarray(0.5), jnp.asarray(3.0))
    p_dot = (jnp.asarray(0.7), jnp.asarray(-0.2), jnp.asarray(0.4))

    z, z_dot = jax.jvp(solved_z, (p,), (p_dot,))
    z_ref, z_dot_ref = jax.jvp(analytic_z, (p,), (p_dot,))

    assert jnp.allclose(z, z_ref, atol=1e-5)
    assert jnp.allclose(z_dot, z_dot_ref, atol=1e-5)


def test_derivative_wrt_x0_and_args_is_zero_by_contract():
    solver = SquareLevenbergMarquardt(lambda x, args, p: x**2 - args * p)
    args = jnp.asarray(2.0)
    p = jnp.asarray(2.0)

    def solved_from_x0(x0):
        return solver.solve(x0, args, p=p, max_steps=50).x

    _, x0_dot = jax.jvp(solved_from_x0, (jnp.array([1.0]),), (jnp.ones(1),))
    assert jnp.allclose(x0_dot, jnp.zeros(1))

    # args is fixed data, never an AD target: zero tangent and zero cotangent
    # even though the root genuinely depends on its value.
    def solved_from_args(a):
        return solver.solve(jnp.array([1.0]), a, p=p, max_steps=50).x

    _, args_dot = jax.jvp(solved_from_args, (args,), (jnp.asarray(1.0),))
    assert jnp.allclose(args_dot, jnp.zeros(1))
    _, pullback = jax.vjp(solved_from_args, args)
    assert jnp.allclose(pullback(jnp.ones(1))[0], 0.0)


def test_derivative_with_p_none_is_zero_and_does_not_crash():
    solver = SquareLevenbergMarquardt(lambda x: x**2 - 4.0)

    def solved_x(x0):
        return solver.solve(x0, max_steps=50).x

    x, x0_dot = jax.jvp(solved_x, (jnp.array([1.0]),), (jnp.ones(1),))
    assert jnp.allclose(x, jnp.array([2.0]), atol=1e-5)
    assert jnp.allclose(x0_dot, jnp.zeros(1))


def test_xtol_stops_on_accepted_small_step():
    # atol=0 disables the residual criterion, so CONVERGED here comes from
    # the accepted step norm alone.
    solver = SquareLevenbergMarquardt(lambda x: x - 1.0)
    result = solver.solve(jnp.array([0.9]), max_steps=20, atol=0.0, xtol=0.5)

    assert int(result.status) == LMStatus.CONVERGED
    assert int(result.steps) == 1
    assert float(result.residual_norm) < 0.1


def test_second_order_derivatives_both_directions():
    # x*(p) = sqrt(p): dx/dp = 1/(2 sqrt(p)), d2x/dp2 = -1/(4 p^(3/2)).
    solver = SquareLevenbergMarquardt(lambda x, args, p: x**2 - p)

    def scalar_root(p):
        return solver.solve(jnp.array([1.0]), p=p, max_steps=60, atol=1e-7).x[0]

    p = jnp.asarray(4.0)
    expected_first = 1.0 / (2.0 * jnp.sqrt(p))
    expected_second = -1.0 / (4.0 * p**1.5)

    assert jnp.allclose(jax.grad(scalar_root)(p), expected_first, atol=1e-5)
    # forward-over-reverse and reverse-over-forward.
    assert jnp.allclose(jax.hessian(scalar_root)(p), expected_second, atol=1e-5)
    assert jnp.allclose(
        jax.jacrev(jax.jacfwd(scalar_root))(p), expected_second, atol=1e-5
    )

    # vmap over differentiated solves.
    ps = jnp.array([1.0, 4.0, 9.0])
    grads = jax.vmap(jax.grad(scalar_root))(ps)
    assert jnp.allclose(grads, 1.0 / (2.0 * jnp.sqrt(ps)), atol=1e-5)


def test_has_aux_reports_and_differentiates_aux():
    # aux m = x0 * p depends on p directly and through the root x*(p) = sqrt(p):
    # dm/dp = sqrt(p) + p/(2 sqrt(p)) = 1.5 sqrt(p). The int32 leaf gets a
    # float0 tangent.
    def residual(x, args, p):
        aux = {"m": x[0] * p, "count": jnp.asarray(1, dtype=jnp.int32)}
        return x**2 - p, aux

    solver = SquareLevenbergMarquardt(residual, has_aux=True)
    p = jnp.asarray(4.0)
    result = solver.solve(jnp.array([1.0]), p=p, max_steps=50)
    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.aux["m"], jnp.sqrt(p) * p, atol=1e-4)
    assert result.aux["count"].dtype == jnp.int32

    def solved_aux_m(q):
        return solver.solve(jnp.array([1.0]), p=q, max_steps=50).aux["m"]

    _, m_dot = jax.jvp(solved_aux_m, (p,), (jnp.asarray(1.0),))
    assert jnp.allclose(m_dot, 1.5 * jnp.sqrt(p), atol=1e-5)
    assert jnp.allclose(jax.grad(solved_aux_m)(p), 1.5 * jnp.sqrt(p), atol=1e-5)

    def solved_aux(q):
        return solver.solve(jnp.array([1.0]), p=q, max_steps=50).aux

    _, aux_dot = jax.jvp(solved_aux, (p,), (jnp.asarray(1.0),))
    assert aux_dot["count"].dtype == jax.dtypes.float0


def test_solve_inside_outer_jit_composes():
    # The DAE consumption shape: solve called inside an outer jitted stage.
    solver = SquareLevenbergMarquardt(lambda x, args, p: x**2 - p)

    @jax.jit
    def stage(p):
        return solver.solve(jnp.array([1.0]), p=p, max_steps=50).x[0] + p

    assert jnp.allclose(stage(jnp.asarray(4.0)), 6.0, atol=1e-5)
