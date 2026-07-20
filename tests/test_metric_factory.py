import dataclasses

import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    LMSolveAction,
    LMStatus,
    Metric,
    MetricFactory,
    MultiStart,
    blockdiag_metric,
    identity_preconditioner,
    metric_from_cholesky,
    metric_from_diagonal,
)

TS = jnp.linspace(0.0, 1.0, 5)


def exp_residual_aux(x, args, p):
    r = x["a"] * jnp.exp(x["b"] * TS) - p["target"]
    w = 1.0 + x["a"] ** 2 + jnp.arange(2.0)
    return r, {"w": w}


def exp_residual_plain(x, args, p):
    return x["a"] * jnp.exp(x["b"] * TS) - p["target"]


X0 = {
    "a": jnp.asarray(0.8, dtype=jnp.float32),
    "b": jnp.asarray(0.4, dtype=jnp.float32),
}
P = {"target": 1.3 * jnp.exp(0.9 * TS)}

DIAG_FACTORY = MetricFactory(
    prepare=lambda x, args, p, aux: aux["w"],
    build=metric_from_diagonal,
)


def solver_kwargs_for(linear_solver):
    if linear_solver == "cg":
        return {
            "iterative_tol": 1e-7,
            "iterative_maxiter": 30,
            "dual_preconditioner": identity_preconditioner(),
            "implicit_preconditioner": identity_preconditioner(),
        }
    if linear_solver == "lsmr":
        return {"iterative_tol": 1e-10, "iterative_maxiter": 50}
    return {}


@pytest.mark.parametrize(
    "linear_solver", ["cholesky", "cg", "qr", "augmented_qr", "lsmr"]
)
@pytest.mark.parametrize("geodesic_acceleration", [False, True])
def test_factory_update_matches_static_metric_at_same_point(
    linear_solver, geodesic_acceleration
):
    # One update with the factory must equal one update with a fixed Metric
    # built from the same values the factory prepares at the pre-step x --
    # exercising every solver family's call sites, and with geodesic
    # acceleration both norm applications sharing the pre-step state.
    w_at_x0 = 1.0 + X0["a"] ** 2 + jnp.arange(2.0)
    dynamic = LevenbergMarquardt(
        exp_residual_aux,
        has_aux=True,
        linear_solver=linear_solver,
        metric_factory=DIAG_FACTORY,
        geodesic_acceleration=geodesic_acceleration,
        **solver_kwargs_for(linear_solver),
    )
    static = LevenbergMarquardt(
        exp_residual_plain,
        linear_solver=linear_solver,
        metric=metric_from_diagonal(w_at_x0),
        geodesic_acceleration=geodesic_acceleration,
        **solver_kwargs_for(linear_solver),
    )

    x_dyn, _, info_dyn = dynamic.update(X0, dynamic.init(X0, p=P), None, P)
    x_st, _, info_st = static.update(X0, static.init(X0, p=P), None, P)

    assert jnp.allclose(x_dyn["a"], x_st["a"], rtol=1e-6, atol=1e-7)
    assert jnp.allclose(x_dyn["b"], x_st["b"], rtol=1e-6, atol=1e-7)
    assert jnp.allclose(info_dyn.loss, info_st.loss, rtol=1e-6)
    assert jnp.allclose(
        info_dyn.acceleration_ratio, info_st.acceleration_ratio, rtol=1e-5, atol=1e-7
    )


def test_factory_cholesky_step_matches_closed_form_underdetermined():
    # One dual step: step = -P J' (J P J' + damping)^{-1} r with P read from
    # the aux the residual just produced.
    weights = jnp.array([4.0, 1.0])

    def residual(theta, args, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), {"w": weights}

    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["w"],
        build=metric_from_diagonal,
    )
    init_damping = 1e-2
    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        init_damping=init_damping,
        metric_factory=factory,
        geodesic_acceleration=False,
    )
    theta0 = jnp.zeros(2)
    p = jnp.asarray(3.0)
    theta1, _, _ = solver.update(theta0, solver.init(theta0, p=p), None, p)

    jacobian = jnp.array([[1.0, 2.0]])
    P_inv = jnp.diag(1.0 / weights)
    dual = jnp.linalg.solve(
        jacobian @ P_inv @ jacobian.T + init_damping * jnp.eye(1),
        jnp.array([theta0[0] + 2.0 * theta0[1] - p]),
    )
    expected = theta0 - (P_inv @ jacobian.T @ dual).ravel()
    assert jnp.allclose(theta1, expected, atol=1e-6)


@pytest.mark.parametrize(
    "linear_solver", ["cholesky", "cg", "qr", "augmented_qr", "lsmr"]
)
def test_factory_full_solve_converges(linear_solver):
    solver = LevenbergMarquardt(
        exp_residual_aux,
        has_aux=True,
        linear_solver=linear_solver,
        metric_factory=DIAG_FACTORY,
        **solver_kwargs_for(linear_solver),
    )
    result = solver.solve(X0, p=P, atol=1e-5, max_steps=100)
    assert result.status == LMStatus.CONVERGED
    assert jnp.allclose(result.x["a"], 1.3, atol=1e-3)
    assert jnp.allclose(result.x["b"], 0.9, atol=1e-3)


def test_factory_solve_python_matches_jit():
    solver = LevenbergMarquardt(
        exp_residual_aux, has_aux=True, metric_factory=DIAG_FACTORY
    )
    jitted = solver.solve(X0, p=P, atol=1e-5, max_steps=100)
    python = solver.solve(X0, p=P, atol=1e-5, max_steps=100, jit=False)
    assert jitted.status == python.status
    assert jnp.allclose(jitted.x["a"], python.x["a"], rtol=1e-6)
    assert jnp.allclose(jitted.x["b"], python.x["b"], rtol=1e-6)


def scalar_residual_aux(theta, args, p):
    w = 1.0 + jnp.atleast_1d(theta[0]) ** 2
    return jnp.atleast_1d(theta[0] ** 2 - 1.0), {"w": w}


def make_scalar_solver():
    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["w"],
        build=metric_from_diagonal,
    )
    return LevenbergMarquardt(
        scalar_residual_aux,
        has_aux=True,
        init_damping=1e-4,
        metric_factory=factory,
        geodesic_acceleration=False,
    )


def test_metric_valid_tracks_rejection_and_acceptance():
    solver = make_scalar_solver()
    # theta = 0.1: the tiny-damping Gauss-Newton step overshoots wildly and
    # the update is rejected, so the carried state stays valid.
    theta_reject = jnp.array([0.1])
    _, state, info = solver.update(
        theta_reject, solver.init(theta_reject, p=None), None, None
    )
    assert not bool(info.accepted)
    assert bool(state.metric_valid)
    # theta = 2.0: the step improves the loss and is accepted, so the carried
    # state is stale at the new x and marked for rebuild.
    theta_accept = jnp.array([2.0])
    _, state, info = solver.update(
        theta_accept, solver.init(theta_accept, p=None), None, None
    )
    assert bool(info.accepted)
    assert not bool(state.metric_valid)


def test_metric_state_reused_when_valid_and_rebuilt_when_invalid():
    solver = make_scalar_solver()
    theta = jnp.array([2.0])
    fresh_state = solver.init(theta, p=None)
    wrong = jax.tree.map(lambda v: 10.0 * v, fresh_state.metric_state)
    x_fresh, _, _ = solver.update(theta, fresh_state, None, None)
    # valid=True: the carried (wrong) state must be used, changing the step.
    x_wrong, _, _ = solver.update(
        theta,
        dataclasses.replace(fresh_state, metric_state=wrong),
        None,
        None,
    )
    assert not jnp.allclose(x_wrong, x_fresh)
    # valid=False: prepare() runs at the current x, so the wrong carry is
    # ignored and the step matches the fresh one.
    x_rebuilt, _, _ = solver.update(
        theta,
        dataclasses.replace(
            fresh_state,
            metric_state=wrong,
            metric_valid=jnp.asarray(False),
        ),
        None,
        None,
    )
    assert jnp.allclose(x_rebuilt, x_fresh)


def test_callback_x_replacement_invalidates_metric_state():
    solver = make_scalar_solver()
    theta = jnp.array([2.0])
    lm_state = solver.init(theta, p=None)
    action = LMSolveAction(x=theta + 0.5)
    _, _, new_state, _, _, problem_changed = solver._apply_action(
        action, theta, lm_state, None, None
    )
    assert bool(problem_changed)
    assert not bool(new_state.metric_valid)
    # Returning the unchanged x is not a change and keeps the state valid.
    _, _, same_state, _, _, problem_changed = solver._apply_action(
        LMSolveAction(x=theta), theta, lm_state, None, None
    )
    assert not bool(problem_changed)
    assert bool(same_state.metric_valid)


def test_callback_lm_state_must_preserve_metric_state():
    solver = make_scalar_solver()
    theta = jnp.array([2.0])
    lm_state = solver.init(theta, p=None)
    action = LMSolveAction(lm_state=dataclasses.replace(lm_state, metric_state=None))
    with pytest.raises(ValueError, match="preserve the metric_state"):
        solver._apply_action(action, theta, lm_state, None, None)


def test_update_requires_init_built_lm_state():
    from nlls_gram import LMState

    solver = make_scalar_solver()
    with pytest.raises(ValueError, match="create the lm_state with init"):
        solver.update(jnp.array([2.0]), LMState(jnp.asarray(1e-4)), None, None)


def test_jacobian_assembly_does_not_differentiate_aux():
    # The aux map has a non-finite derivative at the evaluation point
    # (d sqrt at 0); linearize(has_aux=True) must keep aux primal, so the
    # Jacobian, the step, and the solve stay finite.
    def residual(x, args, p):
        r = x["a"] * jnp.exp(x["b"] * TS) - p["target"]
        spike = jnp.sqrt(jnp.sum(0.0 * x["a"] ** 2))
        w = 1.0 + x["a"] ** 2 + jnp.arange(2.0) + spike
        return r, {"w": w}

    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["w"],
        build=metric_from_diagonal,
    )
    solver = LevenbergMarquardt(residual, has_aux=True, metric_factory=factory)
    result = solver.solve(X0, p=P, atol=1e-5, max_steps=100)
    assert result.status == LMStatus.CONVERGED
    assert bool(jnp.isfinite(result.x["a"]))


@pytest.mark.parametrize("implicit_solver", ["cholesky", "cg"])
def test_implicit_jvp_freezes_metric_at_solution(implicit_solver):
    # Underdetermined linear residual with a p-dependent metric read from aux:
    # the solve returns the min-M(p)-norm solution, and the implicit tangent
    # must use the FROZEN metric at the solution -- the min-norm formula with
    # P* = P(w(p)) held constant, no dP/dp term.
    def residual(theta, args, p):
        w = jnp.array([1.0 + p**2, 1.0])
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), {"w": w}

    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["w"],
        build=metric_from_diagonal,
    )
    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        init_damping=1e-2,
        metric_factory=factory,
        geodesic_acceleration=False,
        implicit_solver=implicit_solver,
        implicit_preconditioner=(
            identity_preconditioner() if implicit_solver == "cg" else None
        ),
        implicit_maxiter=30 if implicit_solver == "cg" else None,
    )

    def solved_x(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x

    p0 = jnp.asarray(3.0)
    p_dot = jnp.asarray(0.7)
    x_star, x_dot = jax.jvp(solved_x, (p0,), (p_dot,))

    jacobian = jnp.array([[1.0, 2.0]])
    P_frozen = jnp.diag(1.0 / jnp.array([1.0 + p0**2, 1.0]))
    min_norm = (
        P_frozen
        @ jacobian.T
        @ jnp.linalg.solve(jacobian @ P_frozen @ jacobian.T, jnp.eye(1))
    ).ravel()
    assert jnp.allclose(x_star, min_norm * p0, atol=1e-5)
    frozen_tangent = min_norm * p_dot
    assert jnp.allclose(x_dot, frozen_tangent, atol=1e-5)

    # The full derivative of p -> P(p) J' (J P(p) J')^{-1} p has a dP/dp term;
    # the frozen contract must NOT include it.
    def full_map(p):
        P_of_p = jnp.diag(1.0 / jnp.array([1.0 + p**2, 1.0]))
        return (
            P_of_p
            @ jacobian.T
            @ jnp.linalg.solve(jacobian @ P_of_p @ jacobian.T, jnp.array([p]))
        ).ravel()

    _, full_tangent = jax.jvp(full_map, (p0,), (p_dot,))
    assert not jnp.allclose(x_dot, full_tangent, atol=1e-4)


def test_implicit_aux_dot_uses_frozen_metric_solution_path():
    # aux_dot composes the solution tangent; with a factory it must run at the
    # frozen solution metric without error and stay finite.
    def residual(theta, args, p):
        w = jnp.array([1.0 + p**2, 1.0])
        return jnp.array([theta[0] + 2.0 * theta[1] - p]), {"w": w}

    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["w"],
        build=metric_from_diagonal,
    )
    solver = LevenbergMarquardt(
        residual,
        has_aux=True,
        metric_factory=factory,
        geodesic_acceleration=False,
    )

    def solved_aux(p):
        return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).aux["w"]

    _, aux_dot = jax.jvp(solved_aux, (jnp.asarray(3.0),), (jnp.asarray(1.0),))
    assert bool(jnp.all(jnp.isfinite(aux_dot)))
    assert jnp.allclose(aux_dot, jnp.array([6.0, 0.0]), atol=1e-4)


def test_multi_start_parallel_does_not_leak_metric_state():
    def draw(key, x, args):
        return jax.tree.map(
            lambda v: v + 0.3 * jax.random.normal(key, jnp.shape(v)), x
        ), args

    solver = LevenbergMarquardt(
        exp_residual_aux, has_aux=True, metric_factory=DIAG_FACTORY
    )
    result = solver.solve(
        X0,
        p=P,
        atol=1e-5,
        max_steps=100,
        multi_start=MultiStart(
            draw=draw, num_starts=3, key=jax.random.key(0), parallel=True
        ),
    )
    assert result.status == LMStatus.CONVERGED
    assert jnp.allclose(result.x["a"], 1.3, atol=1e-3)


def test_factory_solves_do_not_retrace_on_new_aux_values():
    traces = {"count": 0}

    def counting_build(w):
        traces["count"] += 1
        return metric_from_diagonal(w)

    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["w"],
        build=counting_build,
    )
    solver = LevenbergMarquardt(exp_residual_aux, has_aux=True, metric_factory=factory)
    solver.solve(X0, p=P, atol=1e-5, max_steps=50)
    count_after_first = traces["count"]
    other_p = {"target": 1.1 * jnp.exp(0.7 * TS)}
    solver.solve(X0, p=other_p, atol=1e-5, max_steps=50)
    assert traces["count"] == count_after_first

    # Equal-config solvers built around the same factory share the loop.
    twin = LevenbergMarquardt(exp_residual_aux, has_aux=True, metric_factory=factory)
    assert twin == solver
    assert hash(twin) == hash(solver)
    twin.solve(X0, p=P, atol=1e-5, max_steps=50)
    assert traces["count"] == count_after_first


def test_factory_composes_with_blockdiag_and_cholesky_builders():
    def residual(x, args, p):
        r = x["a"] * jnp.exp(x["b"] * TS) - p["target"]
        L = jnp.linalg.cholesky(
            jnp.diag(1.0 + x["a"] ** 2 + jnp.zeros(1)) + jnp.zeros((1, 1))
        )
        return r, {"L": L}

    factory = MetricFactory(
        prepare=lambda x, args, p, aux: aux["L"],
        build=lambda L: blockdiag_metric(
            [(metric_from_cholesky(L), 1), (metric_from_diagonal(jnp.ones(1)), 1)]
        ),
    )
    solver = LevenbergMarquardt(residual, has_aux=True, metric_factory=factory)
    result = solver.solve(X0, p=P, atol=1e-5, max_steps=100)
    assert result.status == LMStatus.CONVERGED
    assert jnp.allclose(result.x["a"], 1.3, atol=1e-3)


def test_factory_allows_has_aux_false_with_none_aux():
    seen = {}

    def prepare(x, args, p, aux):
        seen["aux"] = aux
        return 1.0 + x["a"] ** 2 + jnp.arange(2.0)

    factory = MetricFactory(prepare=prepare, build=metric_from_diagonal)
    solver = LevenbergMarquardt(exp_residual_plain, metric_factory=factory)
    result = solver.solve(X0, p=P, atol=1e-5, max_steps=100)
    assert result.status == LMStatus.CONVERGED
    assert seen["aux"] is None


def test_constructor_rejects_metric_and_factory_together():
    with pytest.raises(ValueError, match="at most one of metric or metric_factory"):
        LevenbergMarquardt(
            exp_residual_aux,
            has_aux=True,
            metric=Metric(solve=lambda v: v),
            metric_factory=DIAG_FACTORY,
        )


def test_constructor_rejects_non_factory_and_non_callable_hooks():
    with pytest.raises(TypeError, match="MetricFactory or None"):
        LevenbergMarquardt(exp_residual_plain, metric_factory=object())
    with pytest.raises(TypeError, match="prepare must be callable"):
        MetricFactory(prepare=None, build=metric_from_diagonal)
    with pytest.raises(TypeError, match="build must be callable"):
        MetricFactory(prepare=lambda x, args, p, aux: aux, build=None)


@pytest.mark.parametrize(
    "linear_solver,build,match",
    [
        (
            "cholesky",
            lambda w: Metric(inv_sqrt=lambda v: v, inv_sqrt_transpose=lambda v: v),
            "requires metric.solve",
        ),
        (
            "lsmr",
            lambda w: Metric(solve=lambda v: v / w),
            "requires metric.inv_sqrt",
        ),
    ],
)
def test_build_output_validated_at_trace_time(linear_solver, build, match):
    factory = MetricFactory(prepare=lambda x, args, p, aux: aux["w"], build=build)
    solver = LevenbergMarquardt(
        exp_residual_aux,
        has_aux=True,
        linear_solver=linear_solver,
        metric_factory=factory,
        geodesic_acceleration=False,
        **solver_kwargs_for(linear_solver),
    )
    with pytest.raises(ValueError, match=match):
        solver.solve(X0, p=P, max_steps=2)


def test_build_requiring_norm_under_geodesic_and_non_metric_output():
    no_norm = MetricFactory(
        prepare=lambda x, args, p, aux: aux["w"],
        build=lambda w: Metric(solve=lambda v: v / w),
    )
    solver = LevenbergMarquardt(
        exp_residual_aux,
        has_aux=True,
        metric_factory=no_norm,
        geodesic_acceleration=True,
    )
    with pytest.raises(ValueError, match="requires metric.norm"):
        solver.solve(X0, p=P, max_steps=2)

    not_a_metric = MetricFactory(
        prepare=lambda x, args, p, aux: aux["w"], build=lambda w: w
    )
    solver = LevenbergMarquardt(
        exp_residual_aux, has_aux=True, metric_factory=not_a_metric
    )
    with pytest.raises(TypeError, match="must return a Metric"):
        solver.solve(X0, p=P, max_steps=2)
