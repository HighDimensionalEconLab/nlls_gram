import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from nlls_gram import (
    CG,
    QR,
    Cholesky,
    CholeskyCache,
    IdentityMetric,
    IdentityPreconditioner,
    LMState,
    Preconditioner,
    QRCache,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)

# Analytic linear-Gaussian testbed: r(theta) = A theta - b with m < p, the
# metric W = blockdiag(K, K) on the leading metric block and a free block of
# size FREE. The fixed-ridge minimizer and the constrained minimum-seminorm
# solution are computed in float64 numpy as references.
RNG = np.random.default_rng(7)
M_RESID, BLOCK, REPEATS, FREE = 4, 4, 2, 2
P_DIM = REPEATS * BLOCK + FREE
N_M = REPEATS * BLOCK
A_NP = RNG.normal(size=(M_RESID, P_DIM))
ROOT = RNG.normal(size=(BLOCK, BLOCK + 2))
K_NP = ROOT @ ROOT.T + 0.5 * np.eye(BLOCK)
W_NP = np.zeros((P_DIM, P_DIM))
for j in range(REPEATS):
    W_NP[j * BLOCK : (j + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = K_NP
B_NP = RNG.normal(size=M_RESID)

A = jnp.asarray(A_NP, dtype=jnp.float32)
B = jnp.asarray(B_NP, dtype=jnp.float32)
K = jnp.asarray(K_NP, dtype=jnp.float32)
X0 = jnp.asarray(RNG.normal(size=P_DIM), dtype=jnp.float32)


def linear_residual(theta):
    return A @ theta - B


def nonlinear_residual(theta):
    return jnp.concatenate([A @ theta - B, jnp.array([theta[0] * theta[1] - 0.4])])


def make_metric():
    return RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=REPEATS)


def ridge_minimizer(ridge):
    return np.linalg.solve(A_NP.T @ A_NP + ridge * W_NP, A_NP.T @ B_NP)


def min_seminorm_solution():
    # KKT system for argmin x' W_bar x s.t. A x = b; nonsingular because A has
    # full row rank and the free block is identified (generic random data).
    kkt = np.block([[W_NP, A_NP.T], [A_NP, np.zeros((M_RESID, M_RESID))]])
    rhs = np.concatenate([np.zeros(P_DIM), B_NP])
    return np.linalg.solve(kkt, rhs)[:P_DIM]


def solver_config(name, preconditioner=None):
    if name == "cholesky":
        return Cholesky()
    if name == "qr":
        return QR()
    return CG(preconditioner or IdentityPreconditioner(), tol=1e-10, maxiter=None)


def build(name, preconditioner=None, **kwargs):
    settings = dict(metric=make_metric(), geodesic_acceleration=False)
    settings.update(kwargs)
    return RidgeLevenbergMarquardt(
        nonlinear_residual,
        linear_solver=solver_config(name, preconditioner),
        **settings,
    )


def test_fixed_ridge_reaches_the_ridge_minimizer():
    ridge = 1e-2
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=ridge)
    result = solver.solve(jnp.zeros(P_DIM), max_steps=200, gtol=1e-5)
    assert int(result.status) == 1
    np.testing.assert_allclose(
        np.asarray(result.x), ridge_minimizer(ridge), rtol=2e-3, atol=2e-4
    )
    # info.loss is the ridge objective; resid_loss and penalty_value decompose it.
    np.testing.assert_allclose(
        float(result.info.loss),
        float(result.info.resid_loss) + ridge * float(result.info.penalty_value),
        rtol=1e-5,
    )


def test_ridge_bias_is_first_order_in_ridge():
    x_dagger = min_seminorm_solution()
    errors = {}
    for ridge in (1e-2, 1e-3):
        solver = RidgeLevenbergMarquardt(
            linear_residual, metric=make_metric(), ridge=ridge
        )
        result = solver.solve(jnp.zeros(P_DIM), max_steps=300, gtol=1e-5)
        assert int(result.status) == 1
        errors[ridge] = np.linalg.norm(np.asarray(result.x) - x_dagger)
        # The solver's error matches the analytic fixed-ridge bias.
        analytic_bias = np.linalg.norm(ridge_minimizer(ridge) - x_dagger)
        assert errors[ridge] < 2.0 * analytic_bias + 1e-3
    # O(ridge) bias: a 10x smaller ridge cuts the error by well over 3x.
    assert errors[1e-3] < errors[1e-2] / 3.0


def test_free_block_is_not_shrunk():
    # The free coordinates carry no penalty, so the solver's free block
    # solves A_f' r = 0 (no ridge shrinkage toward zero) and matches the
    # constrained solution's nonzero free block at O(ridge).
    ridge = 1e-3
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=ridge)
    result = solver.solve(jnp.zeros(P_DIM), max_steps=300, gtol=1e-5)
    assert int(result.status) == 1
    x_dagger = min_seminorm_solution()
    assert np.linalg.norm(x_dagger[-FREE:]) > 0.05
    resid = np.asarray(A_NP @ np.asarray(result.x, np.float64) - B_NP)
    free_gradient = A_NP[:, -FREE:].T @ resid
    assert np.linalg.norm(free_gradient) < 5e-5
    np.testing.assert_allclose(
        np.asarray(result.x)[-FREE:], x_dagger[-FREE:], rtol=0.05, atol=5e-3
    )


def five_dim_residual(theta):
    head = theta[:3]
    return jnp.array(
        [
            head[0] * head[1] - 0.6,
            head[1] + head[2] ** 2 - 1.1,
            jnp.sum(theta**2) - 2.0,
        ]
    )


def test_objective_monotonicity_and_acceptance_identity_metric():
    solver = RidgeLevenbergMarquardt(
        five_dim_residual, metric=IdentityMetric(5), ridge=1e-2, init_damping=1e-1
    )
    x = jnp.asarray([0.9, 0.8, 0.4, 0.1, -0.2], dtype=jnp.float32)
    lm_state = solver.init(x)
    previous_loss = None
    for _ in range(25):
        x_old = x
        x, lm_state, info = solver.update(x, lm_state)
        np.testing.assert_allclose(
            float(info.loss),
            min(float(info.loss_old), float(info.loss_candidate)),
            rtol=1e-6,
        )
        if bool(info.accepted):
            assert float(info.loss_candidate) < float(info.loss_old)
        else:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(x_old))
            assert float(info.loss) == float(info.loss_old)
        if previous_loss is not None:
            assert float(info.loss_old) <= previous_loss * (1 + 1e-6)
        previous_loss = float(info.loss)
    # The objective floor is ridge * q(x*), not zero; the equations themselves
    # are solved to float32 levels.
    assert float(info.resid_loss) < 1e-4


def test_grad_norm_is_the_whitened_ridge_stationarity():
    # IdentityMetric: the whitened stationarity coincides with the plain
    # ||J'r + ridge x||. RepeatedFactorMetric: grad_norm and
    # penalty_grad_norm are the whitened quantities, checked against dense
    # hand-built F_bar references.
    ridge = 3e-2
    solver = RidgeLevenbergMarquardt(
        five_dim_residual, metric=IdentityMetric(5), ridge=ridge
    )
    x = jnp.asarray([0.9, 0.8, 0.4, 0.1, -0.2], dtype=jnp.float32)
    _, _, info = solver.update(x, solver.init(x))
    jacobian = jax.jacobian(five_dim_residual)(x)
    half_gradient = jacobian.T @ five_dim_residual(x) + ridge * x
    np.testing.assert_allclose(
        float(info.grad_norm), float(jnp.linalg.norm(half_gradient)), rtol=1e-5
    )

    metric = make_metric()
    solver = RidgeLevenbergMarquardt(nonlinear_residual, metric=metric, ridge=ridge)
    _, _, info = solver.update(X0, solver.init(X0))
    q_x = float(jnp.sum(metric.factor_apply(X0[:N_M], None) ** 2))
    np.testing.assert_allclose(float(info.penalty_grad_norm), np.sqrt(q_x), rtol=1e-5)
    F64 = np.linalg.cholesky(np.asarray(K, np.float64)).T
    F_bar = jsp_linalg.block_diag(*([F64] * REPEATS), np.eye(FREE))
    jacobian = np.asarray(jax.jacobian(nonlinear_residual)(X0), np.float64)
    resid = np.asarray(nonlinear_residual(X0), np.float64)
    y = F_bar @ np.asarray(X0, np.float64)
    grad_white = np.linalg.solve(F_bar.T, jacobian.T @ resid) + ridge * np.concatenate(
        [y[:N_M], np.zeros(FREE)]
    )
    np.testing.assert_allclose(
        float(info.grad_norm), np.linalg.norm(grad_white), rtol=1e-4
    )


def two_phase_residual(x):
    # Depends on x[0] only: after phase 1 (r ~ 0 at x[0] ~ 1) the entire
    # second coordinate is selection, resolved only by phase 2.
    return x[:1] - 1.0


def test_gtol_calibration_resolves_selection():
    # The whitened calibration recipe: gtol ~ 1e-3 * ridge * sqrt(q) with q
    # the solution's squared seminorm. Toy: r = x0 - 1, F = diag(2, 3), so
    # q(x) = 4 x0^2 + 9 x1^2 and the ridge minimizer is (1/(1 + 4 ridge), 0)
    # with sqrt(q) ~ 2. From (0, 3) the residual floors at (1, 3) -- a
    # pure-residual test would stop 3 off in the second coordinate -- while
    # the calibrated gtol holds the solve until the selection coordinate is
    # resolved.
    ridge = 1e-3
    metric = RepeatedFactorMetric(jnp.diag(jnp.array([2.0, 3.0])))
    solver = RidgeLevenbergMarquardt(two_phase_residual, metric=metric, ridge=ridge)
    x = jnp.array([0.0, 3.0])
    gtol = 1e-3 * ridge * 2.0
    result = solver.solve(x, max_steps=300, gtol=gtol, atol=1e-2)
    assert int(result.status) == 1
    np.testing.assert_allclose(
        np.asarray(result.x), [1.0 / (1.0 + 4.0 * ridge), 0.0], atol=2e-3
    )
    assert abs(float(result.x[1])) < 1e-3


def test_atol_is_conjunctive_an_interpolating_start_does_not_stop():
    # x0 interpolates (r = 0) but carries excess seminorm; a residual-only
    # atol would stop at step 0. The conjunctive contract keeps optimizing
    # until gtol fires, strictly reducing the penalty -- and atol alone is
    # rejected loudly.
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-3)
    with pytest.raises(ValueError, match="conjunctive"):
        solver.solve(jnp.zeros(P_DIM), atol=1e-8)
    x_particular = jnp.asarray(
        np.linalg.lstsq(A_NP, B_NP, rcond=None)[0], dtype=jnp.float32
    )
    null_basis = jnp.asarray(np.linalg.svd(A_NP)[2][M_RESID:].T, dtype=jnp.float32)
    x0 = x_particular + null_basis @ jnp.ones(P_DIM - M_RESID)
    assert float(jnp.linalg.norm(A @ x0 - B)) < 1e-4
    initial_penalty = float(jnp.sum(make_metric().factor_apply(x0[:N_M], None) ** 2))
    result = solver.solve(x0, max_steps=300, gtol=1e-5, atol=1e-3)
    assert int(result.status) == 1
    assert int(result.steps) > 0
    assert float(result.info.penalty_value) < 0.9 * initial_penalty
    assert float(jnp.sqrt(result.info.resid_loss)) <= 1e-3


def test_constructor_and_state_validation():
    metric = make_metric()
    with pytest.raises(ValueError, match="strictly positive"):
        RidgeLevenbergMarquardt(linear_residual, metric=metric, ridge=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        RidgeLevenbergMarquardt(linear_residual, metric=metric, ridge=-1e-3)
    # An uncapped zero-tolerance CG loop has no stopping rule.
    with pytest.raises(ValueError, match="maxiter"):
        CG(IdentityPreconditioner(), tol=0.0)
    # The metric must cover no more than the flattened iterate.
    small = RidgeLevenbergMarquardt(
        lambda theta: theta[:1], metric=IdentityMetric(3), ridge=1e-3
    )
    with pytest.raises(ValueError, match="free block"):
        small.init(jnp.zeros(2))
    # ridge=None resolves to the dtype default at init.
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric())
    lm_state = solver.init(jnp.zeros(P_DIM))
    np.testing.assert_allclose(
        float(lm_state.ridge), np.sqrt(np.finfo(np.float32).eps), rtol=1e-6
    )
    # Manual states must carry a positive ridge.
    bare = RidgeLevenbergMarquardt(
        linear_residual, metric=make_metric(), ridge=1e-3, cache_jacobian=False
    )
    bad = LMState(jnp.asarray(1e-3), None)
    with pytest.raises(ValueError, match="ridge"):
        bare.update(jnp.zeros(P_DIM), bad)
    with pytest.raises(ValueError, match="ridge"):
        bare.solve(jnp.zeros(P_DIM), lm_state=bad, gtol=1e-5)
    zero_ridge = LMState(jnp.asarray(1e-3), jnp.asarray(0.0))
    with pytest.raises(ValueError, match="strictly positive"):
        bare.solve(jnp.zeros(P_DIM), lm_state=zero_ridge, gtol=1e-5)


def test_update_with_ridge_replaced_by_hand_changes_the_subproblem():
    # dataclasses.replace on the state is the documented way to anneal ridge
    # between manual updates; the next step must honor the new value.
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-1)
    x0 = jnp.ones(P_DIM)
    lm_state = solver.init(x0)
    _, _, info_high = solver.update(x0, lm_state)
    lowered = dataclasses.replace(lm_state, ridge=jnp.asarray(1e-4))
    _, _, info_low = solver.update(x0, lowered)
    assert float(info_low.ridge) == pytest.approx(1e-4)
    assert float(info_low.loss_old) < float(info_high.loss_old)


@pytest.mark.parametrize("ridge", [1e-2, 1e-4])
@pytest.mark.parametrize("init_damping", [1e-2, 1e-5])
def test_solvers_agree_on_a_step(ridge, init_damping):
    steps = {}
    for name in ("cholesky", "qr", "normal_cg"):
        solver = build(name, ridge=ridge, init_damping=init_damping)
        lm_state = solver.init(X0)
        x_new, _, info = solver.update(X0, lm_state)
        steps[name] = np.asarray(x_new)
        assert np.isfinite(steps[name]).all()
    # float32 tolerances: the whitened cholesky path's conditioning no longer
    # degrades with RIDGE, but at damping 1e-5 the free-block directions are
    # damped only by mu, so the squared normal conditioning
    # (~ ||J~||^2 / mu ~ 1e6) still costs the cholesky path ~1e-2 in float32
    # against the backward-stable qr; normal_cg iterates on that same squared
    # system. Tight (1e-9) agreement is asserted under float64 in
    # test_float64_subprocess.py.
    np.testing.assert_allclose(steps["cholesky"], steps["qr"], atol=2e-2)
    np.testing.assert_allclose(steps["cholesky"], steps["normal_cg"], atol=2e-2)


def test_full_solves_agree_across_solvers():
    solutions = {}
    for name in ("cholesky", "qr", "normal_cg"):
        solver = build(name, ridge=1e-3)
        result = solver.solve(X0, max_steps=300, gtol=1e-4)
        assert int(result.status) == 1, name
        solutions[name] = np.asarray(result.x)
    np.testing.assert_allclose(solutions["cholesky"], solutions["qr"], atol=1e-3)
    np.testing.assert_allclose(solutions["cholesky"], solutions["normal_cg"], atol=5e-3)


def test_cholesky_normal_cache_is_reused_and_ridge_keyed():
    # Poisoned-cache probe: a state claiming a valid cached G at the matching
    # ridge must be BELIEVED (garbage in, different step out), and the same
    # garbage under a mismatched cache ridge must be recomputed (step
    # identical to the clean one). This pins both the reject-step reuse and
    # the callback-ridge-change invalidation without instrumenting internals.
    solver = build("cholesky", ridge=1e-3)
    fresh = solver.init(X0)
    x_clean, _, _ = solver.update(X0, fresh)

    garbage = jnp.eye(P_DIM, dtype=fresh.solver_cache.G.dtype) * 123.0
    poisoned_matching = dataclasses.replace(
        fresh,
        solver_cache=CholeskyCache(garbage, jnp.asarray(True), fresh.ridge),
    )
    x_poisoned, _, _ = solver.update(X0, poisoned_matching)
    assert not np.allclose(np.asarray(x_poisoned), np.asarray(x_clean))

    poisoned_stale_ridge = dataclasses.replace(
        fresh,
        solver_cache=CholeskyCache(garbage, jnp.asarray(True), fresh.ridge * 2.0),
    )
    x_recomputed, _, _ = solver.update(X0, poisoned_stale_ridge)
    np.testing.assert_allclose(np.asarray(x_recomputed), np.asarray(x_clean), rtol=1e-6)


def test_qr_cache_is_reused_and_ridge_keyed():
    solver = build("qr", ridge=1e-3)
    fresh = solver.init(X0)
    x_clean, state_clean, _ = solver.update(X0, fresh)

    fresh_r = fresh.solver_cache.R
    garbage = jnp.ones_like(fresh_r) + jnp.eye(
        fresh_r.shape[0], fresh_r.shape[1], dtype=fresh_r.dtype
    )
    poisoned_matching = dataclasses.replace(
        fresh,
        solver_cache=QRCache(garbage, jnp.asarray(True), fresh.ridge),
    )
    x_poisoned, _, _ = solver.update(X0, poisoned_matching)
    assert not np.allclose(np.asarray(x_poisoned), np.asarray(x_clean))

    poisoned_stale_ridge = dataclasses.replace(
        fresh,
        solver_cache=QRCache(garbage, jnp.asarray(True), fresh.ridge * 2.0),
    )
    x_recomputed, _, _ = solver.update(X0, poisoned_stale_ridge)
    np.testing.assert_allclose(np.asarray(x_recomputed), np.asarray(x_clean), rtol=1e-6)
    # An update after an ACCEPTED step invalidates the cache (x moved).
    assert not bool(state_clean.solver_cache.valid)
    assert not bool(state_clean.jacobian_valid)


def test_rejected_step_reuses_residual_and_jacobian():
    # r = theta^2 - 1 from theta = 0.1 with tiny damping overshoots wildly, so
    # the step is rejected and the next update must reuse the cached
    # linearization: the residual body executes once (the trial point) instead
    # of twice. jax.debug.callback fires only on execution, not while tracing
    # the untaken cond branch.
    calls = []

    def record():
        calls.append(1)

    def scalar_residual(theta):
        jax.debug.callback(record, ordered=True)
        return theta**2 - 1.0

    solver = RidgeLevenbergMarquardt(
        scalar_residual,
        metric=IdentityMetric(1),
        ridge=1e-6,
        init_damping=1e-8,
        geodesic_acceleration=False,
    )
    x = jnp.asarray([0.1], dtype=jnp.float32)
    lm_state = solver.init(x)
    calls.clear()
    x1, lm_state, info = solver.update(x, lm_state)
    jax.effects_barrier()
    assert not bool(info.accepted)
    assert bool(lm_state.jacobian_valid)
    assert len(calls) == 2  # linearization + trial point

    calls.clear()
    solver.update(x1, lm_state)
    jax.effects_barrier()
    assert len(calls) == 1  # trial point only: cached resid/Jt reused


def test_normal_cg_preconditioner_changes_nothing():
    # M changes the CG iteration path, never the solved subproblem: a
    # Jacobi-style SPD Preconditioner subclass must reproduce the identity-M
    # step -- and it receives the live MetricContext.
    seen = []

    @dataclasses.dataclass(frozen=True, eq=False)
    class JacobiPreconditioner(Preconditioner):
        scale: jax.Array

        def apply(self, v, damping, ctx):
            seen.append(ctx is not None and ctx.lm_state is not None)
            return v / (self.scale + damping)

    scale = jnp.asarray(RNG.uniform(0.5, 2.0, size=P_DIM), dtype=jnp.float32)
    preconditioned = build(
        "normal_cg", ridge=1e-3, preconditioner=JacobiPreconditioner(scale)
    )
    identity = build("normal_cg", ridge=1e-3)
    x_p, _, _ = preconditioned.update(X0, preconditioned.init(X0))
    x_i, _, _ = identity.update(X0, identity.init(X0))
    np.testing.assert_allclose(np.asarray(x_p), np.asarray(x_i), atol=2e-3)
    assert seen and all(seen)
