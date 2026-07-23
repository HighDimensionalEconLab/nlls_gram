import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from nlls_gram import (
    LSMR,
    QR,
    Cholesky,
    CholeskyCache,
    QRCache,
    RidgeLevenbergMarquardt,
    Whitener,
    identity_right_preconditioner,
    repeated_block_whitener,
    repeated_dense_penalty,
    whitener_from_factor,
)

# Shared linear-Gaussian testbed matching test_ridge_lm.py: r(theta) =
# A theta - b with m < p, penalty M0 = blockdiag(K, K, 0_pad). Whitening is a
# pure linear bijection, so every fixed-ridge minimizer must match the
# unwhitened solver's digit for digit (up to float32 solve tolerances) while
# iterate paths and step counts may differ.
RNG = np.random.default_rng(7)
M_RESID, BLOCK, REPEATS, PAD = 4, 4, 2, 2
P_DIM = REPEATS * BLOCK + PAD
HEAD = REPEATS * BLOCK
A_NP = RNG.normal(size=(M_RESID, P_DIM))
ROOT = RNG.normal(size=(BLOCK, BLOCK + 2))
K_NP = ROOT @ ROOT.T + 0.5 * np.eye(BLOCK)
M0_NP = np.zeros((P_DIM, P_DIM))
for j in range(REPEATS):
    M0_NP[j * BLOCK : (j + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = K_NP
B_NP = RNG.normal(size=M_RESID)

A = jnp.asarray(A_NP, dtype=jnp.float32)
B = jnp.asarray(B_NP, dtype=jnp.float32)
K = jnp.asarray(K_NP, dtype=jnp.float32)
X0 = jnp.asarray(RNG.normal(size=P_DIM), dtype=jnp.float32)


def linear_residual(theta):
    return A @ theta - B


def nonlinear_residual(theta):
    return jnp.concatenate(
        [A @ theta - B, jnp.array([theta[0] * theta[1] - 0.4])]
    )


def make_whitener():
    return repeated_block_whitener(K, repeats=REPEATS, zero_pad_size=PAD)


def ridge_minimizer(ridge):
    return np.linalg.solve(A_NP.T @ A_NP + ridge * M0_NP, A_NP.T @ B_NP)


def solver_config(name):
    if name == "cholesky":
        return Cholesky()
    if name == "qr":
        return QR()
    return LSMR(identity_right_preconditioner(), tol=1e-10, maxiter=None)


def build(name, penalty=None, **kwargs):
    settings = dict(
        penalty=make_whitener() if penalty is None else penalty,
        geodesic_acceleration=False,
    )
    settings.update(kwargs)
    return RidgeLevenbergMarquardt(
        nonlinear_residual, linear_solver=solver_config(name), **settings
    )


def test_whitener_construction_validation():
    whitener = make_whitener()
    assert isinstance(whitener, Whitener)
    with pytest.raises(TypeError, match="whiten"):
        dataclasses.replace(whitener, whiten=None)
    with pytest.raises(TypeError, match="unwhiten_transpose"):
        dataclasses.replace(whitener, unwhiten_transpose=object())
    with pytest.raises(ValueError, match="square"):
        whitener_from_factor(jnp.ones((2, 3)), num_rows=1)
    with pytest.raises(ValueError, match="num_rows"):
        whitener_from_factor(jnp.eye(3), num_rows=4)
    with pytest.raises(ValueError, match="num_rows"):
        whitener_from_factor(jnp.eye(3), num_rows=True)


def test_repeated_block_whitener_matches_dense_factor():
    whitener = make_whitener()
    C = jnp.linalg.cholesky(K)
    L_bar = jsp_linalg.block_diag(
        *([C.T] * REPEATS), jnp.eye(PAD, dtype=K.dtype)
    )
    v = jnp.asarray(RNG.normal(size=P_DIM), dtype=jnp.float32)
    matrix = jnp.asarray(RNG.normal(size=(P_DIM, 3)), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(whitener.whiten(v)), np.asarray(L_bar @ v), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(whitener.whiten(matrix)), np.asarray(L_bar @ matrix), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(whitener.unwhiten(whitener.whiten(v))),
        np.asarray(v),
        rtol=2e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(whitener.unwhiten_transpose(matrix)),
        np.linalg.solve(np.asarray(L_bar, np.float64).T, np.asarray(matrix)),
        rtol=2e-4,
        atol=1e-5,
    )
    # The general dense constructor agrees with the structured one.
    general = whitener_from_factor(L_bar, num_rows=HEAD)
    np.testing.assert_allclose(
        np.asarray(general.whiten(v)), np.asarray(whitener.whiten(v)), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(general.unwhiten(v)),
        np.asarray(whitener.unwhiten(v)),
        rtol=2e-4,
        atol=1e-5,
    )
    # Base penalty fields coincide with repeated_dense_penalty's: same head
    # rows, same quadratic, same assembled M0.
    base = repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)
    assert whitener.num_rows == base.num_rows == HEAD
    np.testing.assert_allclose(
        np.asarray(whitener.sqrt_apply(v)), np.asarray(base.sqrt_apply(v)), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(whitener.sqrt_transpose_apply(v[:HEAD])),
        np.asarray(base.sqrt_transpose_apply(v[:HEAD])),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(jnp.sum(whitener.sqrt_apply(v) ** 2)),
        float(v @ (jnp.asarray(M0_NP, jnp.float32) @ v)),
        rtol=1e-4,
    )
    zero = jnp.asarray(whitener.add_scaled(jnp.zeros((P_DIM, P_DIM)), 1.0))
    np.testing.assert_allclose(np.asarray(zero), M0_NP, rtol=1e-5, atol=1e-6)


def test_whitened_solve_reaches_the_ridge_minimizer():
    ridge = 1e-2
    plain = repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)
    for penalty in (make_whitener(), plain):
        solver = RidgeLevenbergMarquardt(linear_residual, penalty=penalty, ridge=ridge)
        result = solver.solve(jnp.zeros(P_DIM), max_steps=200, gtol=1e-5)
        assert int(result.status) == 1
        np.testing.assert_allclose(
            np.asarray(result.x), ridge_minimizer(ridge), rtol=2e-3, atol=2e-4
        )
        # The objective decomposition is unchanged under whitening.
        np.testing.assert_allclose(
            float(result.info.loss),
            float(result.info.resid_loss) + ridge * float(result.info.penalty_value),
            rtol=1e-5,
        )


@pytest.mark.parametrize("ridge", [1e-2, 1e-4])
@pytest.mark.parametrize("init_damping", [1e-2, 1e-5])
def test_three_solvers_agree_on_a_whitened_step(ridge, init_damping):
    steps = {}
    for name in ("cholesky", "qr", "lsmr"):
        solver = build(name, ridge=ridge, init_damping=init_damping)
        lm_state = solver.init(X0)
        x_new, _, info = solver.update(X0, lm_state)
        steps[name] = np.asarray(x_new)
        assert np.isfinite(steps[name]).all()
    # float32 tolerances: the whitened cholesky path's conditioning no longer
    # degrades with RIDGE, but at damping 1e-5 the unpenalized tail
    # directions are damped only by mu, so the squared normal conditioning
    # (~ ||J~||^2 / mu ~ 1e6) still costs the cholesky path ~1e-2 in float32
    # against the backward-stable qr. Tight (1e-9) three-way agreement is
    # asserted under float64 in test_float64_subprocess.py.
    np.testing.assert_allclose(steps["cholesky"], steps["qr"], atol=2e-2)
    np.testing.assert_allclose(steps["cholesky"], steps["lsmr"], atol=2e-2)


def test_full_whitened_solves_agree_across_solvers():
    solutions = {}
    for name in ("cholesky", "qr", "lsmr"):
        solver = build(name, ridge=1e-3)
        result = solver.solve(X0, max_steps=300, gtol=1e-4)
        assert int(result.status) == 1, name
        solutions[name] = np.asarray(result.x)
    np.testing.assert_allclose(solutions["cholesky"], solutions["qr"], atol=1e-3)
    np.testing.assert_allclose(solutions["cholesky"], solutions["lsmr"], atol=5e-3)


def test_whitened_gradient_semantics_single_update():
    # From any pre-step x: penalty_grad_norm = ||[y_head; 0]|| =
    # sqrt(penalty_value at x), and grad_norm is the y-space stationarity
    # ||L_bar^{-T} J'r + ridge [y_head; 0]|| -- both checked against dense
    # hand-built references.
    ridge = 3e-2
    whitener = make_whitener()
    solver = RidgeLevenbergMarquardt(
        nonlinear_residual, penalty=whitener, ridge=ridge
    )
    lm_state = solver.init(X0)
    _, _, info = solver.update(X0, lm_state)
    q_x = float(jnp.sum(whitener.sqrt_apply(X0) ** 2))
    np.testing.assert_allclose(
        float(info.penalty_grad_norm), np.sqrt(q_x), rtol=1e-5
    )
    C = np.linalg.cholesky(np.asarray(K, np.float64))
    L_bar = np.zeros((P_DIM, P_DIM))
    for j in range(REPEATS):
        L_bar[j * BLOCK : (j + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = C.T
    L_bar[HEAD:, HEAD:] = np.eye(PAD)
    jacobian = np.asarray(jax.jacobian(nonlinear_residual)(X0), np.float64)
    resid = np.asarray(nonlinear_residual(X0), np.float64)
    y = L_bar @ np.asarray(X0, np.float64)
    grad_white = np.linalg.solve(L_bar.T, jacobian.T @ resid) + ridge * np.concatenate(
        [y[:HEAD], np.zeros(PAD)]
    )
    np.testing.assert_allclose(
        float(info.grad_norm), np.linalg.norm(grad_white), rtol=1e-4
    )


def two_phase_residual(x):
    return x[:1] - 1.0


def test_whitened_gtol_calibration_resolves_selection():
    # The cleaner whitened calibration recipe: gtol ~ 1e-3 * ridge * sqrt(q)
    # with q the solution's seminorm. Toy: r = x0 - 1, L_bar = diag(2, 3), so
    # q(x) = 4 x0^2 + 9 x1^2 and the ridge minimizer is
    # (1/(1 + 4 ridge), 0) with sqrt(q) ~ 2. From (0, 3) the residual floors
    # at (1, 3); the calibrated whitened gtol holds the solve through phase 2
    # until the selection coordinate x1 is resolved.
    ridge = 1e-3
    whitener = whitener_from_factor(jnp.diag(jnp.array([2.0, 3.0])), num_rows=2)
    solver = RidgeLevenbergMarquardt(
        two_phase_residual, penalty=whitener, ridge=ridge
    )
    x = jnp.array([0.0, 3.0])
    gtol = 1e-3 * ridge * 2.0
    result = solver.solve(x, max_steps=300, gtol=gtol, atol=1e-2)
    assert int(result.status) == 1
    np.testing.assert_allclose(
        np.asarray(result.x), [1.0 / (1.0 + 4.0 * ridge), 0.0], atol=2e-3
    )
    assert abs(float(result.x[1])) < 1e-3


def test_whitened_cholesky_cache_is_reused_and_ridge_keyed():
    # Poisoned-cache probe (pattern of test_ridge_solvers.py): a valid cached
    # whitened G~ at the matching ridge must be BELIEVED, and the same
    # garbage under a mismatched cache ridge must be recomputed.
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


def test_whitened_qr_cache_is_reused_and_ridge_keyed():
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
    assert not bool(state_clean.solver_cache.valid)
    assert not bool(state_clean.jacobian_valid)


def counting_whitener(matrix_counters):
    # The whitened cholesky path materializes J~' (unwhiten_transpose on a
    # MATRIX) only inside the cache-refresh branch, while the per-step
    # gradient pulls back vectors; counting matrix applications therefore
    # counts assemblies, exactly like add_scaled in
    # test_ridge_compile_cache.py. ndim is static at trace time, so the
    # Python branch below never enters the compiled program.
    base = whitener_from_factor(jnp.eye(P_DIM, dtype=jnp.float32), num_rows=P_DIM)

    def unwhiten_transpose(v):
        if v.ndim == 2:
            jax.debug.callback(lambda: matrix_counters.append(1), ordered=True)
        return base.unwhiten_transpose(v)

    return dataclasses.replace(base, unwhiten_transpose=unwhiten_transpose)


def test_whitened_rejected_step_skips_assembly():
    counters = []
    whitener = counting_whitener(counters)

    def residual(theta):
        # Nonlinear scalar tail engineered so the near-Newton step from a
        # small residual overshoots and gets rejected at tiny damping.
        return jnp.concatenate([A @ theta - B, jnp.array([theta[0] ** 2 - 1.0])])

    solver = RidgeLevenbergMarquardt(
        residual,
        penalty=whitener,
        ridge=1e-4,
        init_damping=1e-9,
        geodesic_acceleration=False,
    )
    x = jnp.zeros(P_DIM, dtype=jnp.float32).at[0].set(0.05)
    lm_state = solver.init(x)

    counters.clear()
    x1, state1, info1 = solver.update(x, lm_state)
    jax.effects_barrier()
    assert len(counters) == 1
    assert not bool(info1.accepted)
    assert bool(state1.solver_cache.valid)

    # Rejected step: same x, same ridge -- the whitened assembly (including
    # the J~' materialization) is skipped, refactor only.
    counters.clear()
    x2, state2, info2 = solver.update(x1, state1)
    jax.effects_barrier()
    assert len(counters) == 0

    # A ridge change at the same x invalidates through the cache ridge key.
    counters.clear()
    lowered = dataclasses.replace(state2, ridge=state2.ridge * 0.1)
    solver.update(x2, lowered)
    jax.effects_barrier()
    assert len(counters) == 1


WEIGHTS = jnp.asarray(RNG.normal(size=P_DIM), dtype=jnp.float32)


def linear_residual_p(theta, args, p):
    return A @ theta - p


def central_difference(fn, p, step):
    grads = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        bump = np.zeros(p.shape[0], dtype=np.float32)
        bump[i] = step
        grads[i] = (
            float(fn(p + jnp.asarray(bump))) - float(fn(p - jnp.asarray(bump)))
        ) / (2 * step)
    return grads


def test_whitened_implicit_ad_linear_matches_fd_and_unwhitened():
    # Linear residual: the GN implicit rule is exact, and the whitened rule
    # is the same rule through a linear bijection -- so the whitened solver's
    # gradient must match finite differences AND the unwhitened solver's.
    p0 = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)
    grads = {}
    for tag, penalty in (
        ("whitened", make_whitener()),
        ("plain", repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)),
    ):
        solver = RidgeLevenbergMarquardt(
            linear_residual_p, penalty=penalty, ridge=1e-3
        )

        def loss(p, solver=solver):
            result = solver.solve(jnp.zeros(P_DIM), p=p, max_steps=300, gtol=1e-5)
            return jnp.vdot(WEIGHTS, result.x)

        grads[tag] = np.asarray(jax.grad(loss)(p0))
        if tag == "whitened":
            fd = central_difference(loss, p0, 3e-2)
            np.testing.assert_allclose(grads[tag], fd, rtol=2e-2, atol=2e-3)
    # The two rules are algebraically identical (verified to 1e-13 in the
    # float64 subprocess test); in float32 the comparison is dominated by the
    # slightly different gtol=1e-5 converged iterates each solver
    # relinearizes at.
    np.testing.assert_allclose(grads["whitened"], grads["plain"], rtol=1e-2, atol=1e-4)


def nonlinear_residual_p(theta, args, p):
    return jnp.concatenate(
        [
            A @ theta - p,
            jnp.array([theta[0] * theta[1] - 0.3 * p[0]]),
        ]
    )


def test_whitened_implicit_ad_nonlinear_matches_unwhitened():
    p0 = jnp.asarray(RNG.normal(size=M_RESID), dtype=jnp.float32)
    x0 = 0.3 * jnp.ones(P_DIM)
    grads = {}
    for tag, penalty in (
        ("whitened", make_whitener()),
        ("plain", repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD)),
    ):
        solver = RidgeLevenbergMarquardt(
            nonlinear_residual_p, penalty=penalty, ridge=1e-3
        )

        def loss(p, solver=solver):
            result = solver.solve(x0, p=p, max_steps=300, gtol=1e-5)
            return jnp.vdot(WEIGHTS, result.x)

        grads[tag] = np.asarray(jax.grad(loss)(p0))
    np.testing.assert_allclose(grads["whitened"], grads["plain"], rtol=2e-2, atol=2e-4)


def test_equal_config_whitened_solvers_share_the_compiled_loop():
    traces = {"count": 0}

    def residual(theta):
        traces["count"] += 1
        return A @ theta - B

    whitener = make_whitener()
    first = RidgeLevenbergMarquardt(residual, penalty=whitener, ridge=1e-3)
    second = RidgeLevenbergMarquardt(residual, penalty=whitener, ridge=1e-3)
    assert first == second
    assert hash(first) == hash(second)
    # A plain penalty (or a freshly constructed whitener: new callback
    # identities) keys a different compilation.
    plain = RidgeLevenbergMarquardt(
        residual,
        penalty=repeated_dense_penalty(K, repeats=REPEATS, zero_pad_size=PAD),
        ridge=1e-3,
    )
    assert first != plain

    x0 = jnp.zeros(P_DIM)
    first.solve(x0, max_steps=30, gtol=1e-4)
    baseline = traces["count"]
    # +1 eager init residual call, zero retraces of the loop body.
    second.solve(x0, max_steps=30, gtol=1e-4)
    assert traces["count"] == baseline + 1
