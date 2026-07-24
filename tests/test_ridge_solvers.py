import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    LSMR,
    QR,
    Cholesky,
    CholeskyCache,
    NormalCG,
    QRCache,
    RidgeLevenbergMarquardt,
    WhitenedPreconditioner,
    identity_preconditioner,
    identity_right_preconditioner,
    repeated_dense_penalty,
)

RNG = np.random.default_rng(23)
M_RESID, BLOCK, REPEATS, PAD = 5, 4, 2, 2
P_DIM = REPEATS * BLOCK + PAD
A_NP = RNG.normal(size=(M_RESID, P_DIM))
ROOT = RNG.normal(size=(BLOCK, BLOCK + 2))
K_NP = ROOT @ ROOT.T + 0.5 * np.eye(BLOCK)
B_NP = RNG.normal(size=M_RESID)

A = jnp.asarray(A_NP, dtype=jnp.float32)
B = jnp.asarray(B_NP, dtype=jnp.float32)
K = jnp.asarray(K_NP, dtype=jnp.float32)
X0 = jnp.asarray(RNG.normal(size=P_DIM), dtype=jnp.float32)


def nonlinear_residual(theta):
    return jnp.concatenate(
        [A @ theta - B, jnp.array([theta[0] * theta[1] - 0.4])]
    )


def make_penalty():
    return repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)


def solver_config(name, preconditioner=None):
    if name == "cholesky":
        return Cholesky()
    if name == "qr":
        return QR()
    if name == "normal_cg":
        return NormalCG(
            preconditioner or identity_preconditioner(),
            tol=1e-10,
            maxiter=None,
        )
    return LSMR(
        preconditioner or identity_right_preconditioner(),
        tol=1e-10,
        maxiter=None,
    )


def build(name, preconditioner=None, **kwargs):
    settings = dict(penalty=make_penalty(), geodesic_acceleration=False)
    settings.update(kwargs)
    return RidgeLevenbergMarquardt(
        nonlinear_residual,
        linear_solver=solver_config(name, preconditioner),
        **settings,
    )


@pytest.mark.parametrize("ridge", [1e-2, 1e-4])
@pytest.mark.parametrize("init_damping", [1e-2, 1e-5])
def test_solvers_agree_on_a_step(ridge, init_damping):
    steps = {}
    for name in ("cholesky", "qr", "lsmr", "normal_cg"):
        solver = build(name, ridge=ridge, init_damping=init_damping)
        lm_state = solver.init(X0)
        x_new, _, info = solver.update(X0, lm_state)
        steps[name] = np.asarray(x_new)
        assert np.isfinite(steps[name]).all()
    # float32 tolerances: at small ridge/damping the squared normal system's
    # conditioning costs the cholesky path ~1e-3 while qr stays accurate --
    # the qr path's reason to exist; normal_cg iterates on the same squared
    # system. Tight (1e-9) agreement is asserted under float64 in
    # test_float64_subprocess.py.
    np.testing.assert_allclose(steps["cholesky"], steps["qr"], atol=5e-3)
    np.testing.assert_allclose(steps["cholesky"], steps["lsmr"], atol=5e-3)
    np.testing.assert_allclose(steps["cholesky"], steps["normal_cg"], atol=5e-3)


def test_full_solves_agree_across_solvers():
    solutions = {}
    for name in ("cholesky", "qr", "lsmr", "normal_cg"):
        solver = build(name, ridge=1e-3)
        result = solver.solve(X0, max_steps=300, gtol=1e-4)
        assert int(result.status) == 1, name
        solutions[name] = np.asarray(result.x)
    np.testing.assert_allclose(solutions["cholesky"], solutions["qr"], atol=1e-3)
    np.testing.assert_allclose(solutions["cholesky"], solutions["lsmr"], atol=5e-3)
    np.testing.assert_allclose(
        solutions["cholesky"], solutions["normal_cg"], atol=5e-3
    )


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
    np.testing.assert_allclose(
        np.asarray(x_recomputed), np.asarray(x_clean), rtol=1e-6
    )


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
    np.testing.assert_allclose(
        np.asarray(x_recomputed), np.asarray(x_clean), rtol=1e-6
    )
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

    from nlls_gram import identity_penalty

    solver = RidgeLevenbergMarquardt(
        scalar_residual,
        penalty=identity_penalty(1),
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
    first_update_calls = len(calls)
    assert first_update_calls == 2  # linearization + trial point

    calls.clear()
    solver.update(x1, lm_state)
    jax.effects_barrier()
    assert len(calls) == 1  # trial point only: cached resid/Jt reused


def test_lsmr_right_preconditioner_changes_nothing():
    scale = jnp.asarray(RNG.uniform(0.5, 2.0, size=P_DIM), dtype=jnp.float32)

    def solve(v, damping):
        return v / scale

    def solve_transpose(w, damping):
        return w / scale

    preconditioned = build(
        "lsmr",
        ridge=1e-3,
        preconditioner=WhitenedPreconditioner(solve, solve_transpose),
    )
    identity = build("lsmr", ridge=1e-3)
    state_p = preconditioned.init(X0)
    state_i = identity.init(X0)
    x_p, _, _ = preconditioned.update(X0, state_p)
    x_i, _, _ = identity.update(X0, state_i)
    np.testing.assert_allclose(np.asarray(x_p), np.asarray(x_i), atol=2e-3)


def test_forward_normal_cg_requires_a_preconditioner():
    # The AD role keeps preconditioner=None legal; the forward role must
    # reject it -- nobody runs forward Krylov without an explicit choice.
    with pytest.raises(TypeError, match="identity_preconditioner"):
        RidgeLevenbergMarquardt(
            nonlinear_residual,
            penalty=make_penalty(),
            linear_solver=NormalCG(maxiter=100),
        )
    # The same config stays valid in its AD role.
    RidgeLevenbergMarquardt(
        nonlinear_residual,
        penalty=make_penalty(),
        ad_solver=NormalCG(maxiter=100),
    )


def test_normal_cg_preconditioner_changes_nothing():
    # M changes the CG iteration path, never the solved subproblem: a
    # Jacobi-style SPD preconditioner must reproduce the identity-M step.
    scale = jnp.asarray(RNG.uniform(0.5, 2.0, size=P_DIM), dtype=jnp.float32)

    def preconditioner(v, damping):
        return v / (scale + damping)

    preconditioned = build("normal_cg", ridge=1e-3, preconditioner=preconditioner)
    identity = build("normal_cg", ridge=1e-3)
    state_p = preconditioned.init(X0)
    state_i = identity.init(X0)
    x_p, _, _ = preconditioned.update(X0, state_p)
    x_i, _, _ = identity.update(X0, state_i)
    np.testing.assert_allclose(np.asarray(x_p), np.asarray(x_i), atol=2e-3)
