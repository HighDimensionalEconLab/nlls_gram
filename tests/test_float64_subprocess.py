import subprocess
import sys
import textwrap


def test_failed_implicit_ad_float64_is_finite_and_exactly_zero():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import LMStatus, LevenbergMarquardt


def residual(x, args, p):
    return x + args - p


solver = LevenbergMarquardt(
    residual,
    cache_jacobian=False,
    geodesic_acceleration=False,
)
x0 = jnp.zeros(1, dtype=jnp.float64)
args = jnp.asarray(0.0, dtype=jnp.float64)
parameters = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
atols = jnp.asarray([1e-12, 0.0], dtype=jnp.float64)


def solved_x(values):
    return jax.vmap(
        lambda value, atol: solver.solve(
            x0,
            args,
            p=value,
            max_steps=1,
            max_steps_is_success=False,
            atol=atol,
        ).x[0]
    )(values, atols)


statuses = jax.vmap(
    lambda value, atol: solver.solve(
        x0,
        args,
        p=value,
        max_steps=1,
        max_steps_is_success=False,
        atol=atol,
    ).status
)(parameters, atols)
assert jnp.array_equal(
    statuses,
    jnp.asarray([LMStatus.CONVERGED, LMStatus.MAX_STEPS], dtype=jnp.int32),
)

_, tangent = jax.jvp(
    solved_x,
    (parameters,),
    (jnp.ones_like(parameters),),
)
_, pullback = jax.vjp(solved_x, parameters)
(cotangent,) = pullback(jnp.ones_like(parameters))

expected = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
assert tangent.dtype == jnp.float64
assert cotangent.dtype == jnp.float64
assert jnp.all(jnp.isfinite(tangent))
assert jnp.all(jnp.isfinite(cotangent))
assert jnp.array_equal(tangent, expected)
assert jnp.array_equal(cotangent, expected)

hessian = jax.hessian(lambda values: jnp.sum(solved_x(values)))(parameters)
assert hessian.dtype == jnp.float64
assert jnp.all(jnp.isfinite(hessian))
assert jnp.array_equal(hessian, jnp.zeros_like(hessian))
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_solve_with_float32_problem_under_x64_keeps_lm_state_dtype_consistent():
    # solve(lm_state=None) must carry the damping in the residual dtype, not
    # the default float, or the while_loop carry mismatches update()'s output
    # for float32 problems under x64.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import LMStatus, LevenbergMarquardt


def residual(theta, args, p):
    return theta - args


solver = LevenbergMarquardt(residual, init_damping=1e-2)
for jit in (True, False):
    result = solver.solve(
        jnp.zeros(1, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        max_steps=40,
        atol=1e-5,
        jit=jit,
    )
    assert int(result.status) == LMStatus.CONVERGED, int(result.status)
    assert result.lm_state.damping.dtype == jnp.float32, result.lm_state.damping.dtype
    assert result.x.dtype == jnp.float32

# All compute ops must stay float32; only call-boundary scalars (tolerances,
# default-dtype init damping) may arrive as f64 before being converted.
jaxpr = str(
    jax.make_jaxpr(
        lambda p, a: solver.solve(
            p, a, max_steps=40, atol=1e-5, gtol=1e-6, xtol=1e-6
        ).x
    )(jnp.zeros(1, dtype=jnp.float32), jnp.ones(1, dtype=jnp.float32))
)
for line in jaxpr.splitlines():
    stripped = line.strip()
    if " = " in stripped and ":f64[" in stripped.split(" = ")[0]:
        assert "convert_element_type" in stripped, stripped
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_float64_svd_ad_solver_near_duplicate_rows():
    # The growth-model pathology: a converged simulation duplicates its
    # late-horizon states to ~1e-13, so the float64 undamped implicit dual has
    # eigenvalues far below the factorization noise floor and the unregularized
    # unregularized factorization goes non-finite. The SVD pseudoinverse returns
    # the minimum-norm
    # tangent d sum(x*)/d target = sum(w)/||w||^2 (exact in the duplicate
    # limit) to high accuracy.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import SVD, LevenbergMarquardt

w = jnp.array([1.0, 2.0, 3.0])
wiggles = 1.0 + 1e-13 * jnp.arange(40.0)


def residual_fn(x, args, p):
    return wiggles * (jnp.dot(w, x) - p["target"])


x0 = jnp.zeros(3)
solver = LevenbergMarquardt(residual_fn, ad_solver=SVD())


def sum_x_star(target):
    return jnp.sum(solver.solve(x0, p={"target": target}, max_steps=50).x)


expected = jnp.sum(w) / jnp.dot(w, w)
_, jvp = jax.jvp(sum_x_star, (1.0,), (1.0,))
assert jnp.isfinite(jvp), jvp
assert jnp.allclose(jvp, expected, rtol=1e-8), (jvp, expected)
grad = jax.grad(sum_x_star)(1.0)
assert jnp.isfinite(grad), grad
assert jnp.allclose(grad, expected, rtol=1e-8), (grad, expected)
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_float64_multi_start_modes_and_float32_data_under_x64():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import (
    LMAction,
    LMStatus,
    MultiStart,
    LevenbergMarquardt,
)


def residual_fn(theta, args, p):
    return jnp.array([theta[0] + 2.0 * theta[1] - p])


# The case that used to force explicit int32 casts in user callbacks: under
# x64 both lax.cond branches return bare/weak LMStatus values and the solver
# coerces stop -> bool and status -> int32 at the boundary.
def epoch_callback(ctx):
    def check(_):
        stop = ctx.info.loss < 1e-16
        return stop, jnp.where(stop, LMStatus.CONVERGED, LMStatus.RUNNING)

    def keep_running(_):
        return jnp.asarray(False), jnp.asarray(LMStatus.RUNNING)

    stop, status = jax.lax.cond(ctx.step % 2 == 0, check, keep_running, None)
    return LMAction(stop=stop, status=status)


callback_solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
cb_result = callback_solver.solve(
    jnp.zeros(2, dtype=jnp.float64),
    p=jnp.asarray(3.0, dtype=jnp.float64),
    max_steps=50,
    callback=epoch_callback,
)
assert cb_result.status.dtype == jnp.int32, cb_result.status.dtype
assert int(cb_result.status) == LMStatus.CONVERGED, int(cb_result.status)


def draw_zeros(key, x, args):
    return jnp.zeros_like(x), args


solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
x0 = jnp.array([jnp.nan, jnp.nan], dtype=jnp.float64)
p = jnp.asarray(3.0, dtype=jnp.float64)
expected = jnp.sum(jnp.array([1.0, 2.0])) / 5.0

for parallel in (False, True):
    ms = MultiStart(
        key=jax.random.key(0), num_starts=3, draw=draw_zeros, parallel=parallel
    )

    def sum_x(pv, ms=ms):
        return jnp.sum(
            solver.solve(x0, p=pv, max_steps=80, atol=1e-12, multi_start=ms).x
        )

    result = solver.solve(x0, p=p, max_steps=80, atol=1e-12, multi_start=ms)
    assert int(result.status) == LMStatus.CONVERGED, int(result.status)
    assert result.x.dtype == jnp.float64
    assert result.multi_start.loss.dtype == jnp.float64
    assert result.multi_start.attempt.dtype == jnp.int32
    assert result.multi_start.attempts_run.dtype == jnp.int32
    grad = jax.grad(sum_x)(p)
    assert jnp.allclose(grad, expected, rtol=1e-8), (grad, expected)
    jaxpr = str(jax.make_jaxpr(sum_x)(p))
    assert "f32" not in jaxpr, jaxpr

# x64 enabled but float32 problem data: nothing widens to f64/i64 -- the inf
# sentinels, masked losses, and argmin winner index all stay narrow.
x0_f32 = jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32)
p_f32 = jnp.asarray(3.0, dtype=jnp.float32)
for parallel in (False, True):
    ms = MultiStart(
        key=jax.random.key(1), num_starts=3, draw=draw_zeros, parallel=parallel
    )
    result = solver.solve(x0_f32, p=p_f32, max_steps=80, atol=1e-6, multi_start=ms)
    assert result.x.dtype == jnp.float32, result.x.dtype
    assert result.info.loss.dtype == jnp.float32
    assert result.multi_start.loss.dtype == jnp.float32, result.multi_start.loss.dtype
    assert result.multi_start.attempt.dtype == jnp.int32
    assert result.multi_start.accepted.dtype == jnp.bool_
    history = solver.solve(
        x0_f32, p=p_f32, max_steps=20, atol=1e-6, save_steps=True, multi_start=ms
    ).x_history
    assert history.dtype == jnp.float32
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_float64_damping_floor_is_the_float64_normal_floor():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt, LMState


def residual(theta):
    return theta


solver = LevenbergMarquardt(
    residual,
    cache_jacobian=False,
    geodesic_acceleration=False,
)
state = LMState(jnp.asarray(0.0, dtype=jnp.float64))
x, state, info = solver.update(jnp.ones(1, dtype=jnp.float64), state)
floor = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)

assert info.accepted
assert jnp.all(jnp.isfinite(x))
assert state.damping.dtype == jnp.float64
assert state.damping == floor, (state.damping, floor)
assert info.damping == floor, (info.damping, floor)
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_ridge_lm_float64_solver_agreement_and_dtype_policy():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nlls_gram import (
    QR,
    Cholesky,
    CG,
    IdentityPreconditioner,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)

rng = np.random.default_rng(3)
m, block, repeats, free = 5, 4, 2, 2
p_dim = repeats * block + free
A = jnp.asarray(rng.normal(size=(m, p_dim)))
b = jnp.asarray(rng.normal(size=m))
root = rng.normal(size=(block, block + 2))
K_np = root @ root.T + 0.5 * np.eye(block)
K = jnp.asarray(K_np)
x0 = jnp.asarray(rng.normal(size=p_dim))


def residual(theta):
    return A @ theta - b


def make_metric(K_value):
    return RepeatedFactorMetric(
        jnp.linalg.cholesky(K_value, upper=True), repeats=repeats
    )


SOLVER_CONFIGS = {
    "cholesky": Cholesky(),
    "qr": QR(),
    "normal_cg": CG(IdentityPreconditioner(), tol=1e-14, maxiter=None),
}


def build(name, **kwargs):
    settings = dict(
        metric=make_metric(K),
        ridge=1e-8,
        init_damping=1e-6,
        geodesic_acceleration=False,
    )
    settings.update(kwargs)
    return RidgeLevenbergMarquardt(
        residual, linear_solver=SOLVER_CONFIGS[name], **settings
    )


# Tight three-way whitened step agreement at small ridge/damping: float64
# keeps even the squared cholesky path accurate enough to meet qr at 1e-9;
# the matrix-free path stops at its CG tolerance on this operator, not at
# direct-solve accuracy.
steps = {}
for name in ("cholesky", "qr", "normal_cg"):
    solver = build(name)
    lm_state = solver.init(x0)
    x1, new_state, info = solver.update(x0, lm_state)
    steps[name] = np.asarray(x1)
    assert x1.dtype == jnp.float64
    assert info.loss.dtype == jnp.float64
    assert info.resid_loss.dtype == jnp.float64
    assert info.penalty_value.dtype == jnp.float64
    assert info.grad_norm.dtype == jnp.float64
    assert info.ridge.dtype == jnp.float64
    assert new_state.ridge.dtype == jnp.float64
np.testing.assert_allclose(steps["cholesky"], steps["qr"], atol=1e-9)
np.testing.assert_allclose(steps["cholesky"], steps["normal_cg"], atol=1e-8)

# No float32 leaks anywhere in the compiled update or solve.
solver = build("cholesky")
lm_state = solver.init(x0)
jaxpr = str(jax.make_jaxpr(lambda t, s: solver.update(t, s))(x0, lm_state))
assert "f32" not in jaxpr, jaxpr
solve_jaxpr = str(
    jax.make_jaxpr(lambda t: solver.solve(t, max_steps=20, gtol=1e-10).x)(x0)
)
assert "f32" not in solve_jaxpr, solve_jaxpr

# A float32 problem under enabled x64 stays float32 end to end -- the ridge
# solver runs entirely at the residual dtype (the qr path is the in-dtype
# conditioning fix; there is no linear_solve_dtype promotion knob).
x0_f32 = x0.astype(jnp.float32)
A32, b32, K32 = A.astype(jnp.float32), b.astype(jnp.float32), K.astype(jnp.float32)


def residual32(theta):
    return A32 @ theta - b32


for name in ("cholesky", "qr"):
    solver32 = RidgeLevenbergMarquardt(
        residual32,
        linear_solver=SOLVER_CONFIGS[name],
        metric=make_metric(K32),
        ridge=1e-6,
    )
    state32 = solver32.init(x0_f32)
    assert state32.ridge.dtype == jnp.float32
    if name == "cholesky":
        assert state32.solver_cache.G.dtype == jnp.float32
    else:
        assert state32.solver_cache.R.dtype == jnp.float32
    assert state32.solver_cache.ridge.dtype == jnp.float32
    x1_32, new_state32, info32 = solver32.update(x0_f32, state32)
    assert x1_32.dtype == jnp.float32
    assert info32.loss.dtype == jnp.float32
    assert info32.grad_norm.dtype == jnp.float32
    result32 = solver32.solve(x0_f32, max_steps=50, gtol=1e-4)
    assert result32.x.dtype == jnp.float32
    assert result32.lm_state.ridge.dtype == jnp.float32

# The production scenario whitening exists for: at TINY ridge the whitened
# CHOLESKY solve matches the numpy normal-equations reference (itself only
# ~1e-5 accurate at 1/ridge conditioning, hence the loose rtol) at the
# cholesky path's per-step cost. gtol follows the whitened calibration
# recipe c * ridge * sqrt(q(x*)); c = 1e-5 still sits far above the
# whitened stationarity noise floor here.
W = np.zeros((p_dim, p_dim))
for j in range(repeats):
    W[j * block : (j + 1) * block, j * block : (j + 1) * block] = K_np
ridge = 1e-10
A_np = np.asarray(A)
x_ridge = np.linalg.solve(A_np.T @ A_np + ridge * W, A_np.T @ np.asarray(b))
sqrt_q = float(np.sqrt(x_ridge @ (W @ x_ridge)))
tiny = RidgeLevenbergMarquardt(residual, metric=make_metric(K), ridge=ridge)
result_tiny = tiny.solve(jnp.zeros(p_dim), max_steps=400, gtol=1e-5 * ridge * sqrt_q)
assert int(result_tiny.status) == 1
np.testing.assert_allclose(np.asarray(result_tiny.x), x_ridge, rtol=1e-4, atol=1e-7)
# Whitened gtol semantics: penalty_grad_norm reports sqrt(penalty_value) at
# the converged iterate's pre-step x (both evaluated at x* here).
np.testing.assert_allclose(
    float(result_tiny.info.penalty_grad_norm), sqrt_q, rtol=1e-4
)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_ridge_continuation_matches_metric_lm_min_seminorm_float64():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nlls_gram import (
    QR,
    AnnealRidge,
    LevenbergMarquardt,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)

rng = np.random.default_rng(9)
m, block, repeats, pad = 4, 4, 2, 2
p_dim = repeats * block + pad
A_np = rng.normal(size=(m, p_dim))
b_np = rng.normal(size=m)
root = rng.normal(size=(block, block + 2))
K_np = root @ root.T + 0.5 * np.eye(block)
M0 = np.zeros((p_dim, p_dim))
for j in range(repeats):
    M0[j * block : (j + 1) * block, j * block : (j + 1) * block] = K_np

A = jnp.asarray(A_np)
b = jnp.asarray(b_np)
K = jnp.asarray(K_np)


def residual(theta):
    return A @ theta - b


# The affine problem's exact minimum-seminorm interpolant from the KKT system.
kkt = np.block([[M0, A_np.T], [A_np, np.zeros((m, m))]])
x_dagger = np.linalg.solve(kkt, np.concatenate([np.zeros(p_dim), b_np]))[:p_dim]

# Metric-damped LM with a small epsilon shift selects x_dagger + O(eps).
# free_scale weights the zero-padded tail, the role epsilon used to play
# there; the metric block carries K + epsilon I.
metric = RepeatedFactorMetric(
    jnp.linalg.cholesky(K + 1e-8 * jnp.eye(K.shape[0], dtype=K.dtype), upper=True),
    repeats=repeats,
    free_scale=1e-8,
)
metric_solver = LevenbergMarquardt(residual, metric=metric)
metric_result = metric_solver.solve(jnp.zeros(p_dim), max_steps=200, atol=1e-12)
x_metric = np.asarray(metric_result.x)

# Ridge LM with continuation to a small floor selects x_dagger + O(floor).
# The qr path carries the endgame: below ridge ~ 1e-8 the squared normal
# system's conditioning (~ 1/ridge) costs the cholesky path float64 digits,
# while the QR of [J; sqrt(ridge) L] works at the square root of that.
# Floor choice: the selection is resolved to grad_norm / ridge, and the
# achievable float64 gradient is ~1e-15, so ridge_floor ~ 1e-8 balances the
# O(ridge) bias against the eps/ridge stationarity resolution -- pushing the
# floor lower makes the answer WORSE, not better. gtol must sit well below
# ridge times the target selection accuracy.
callback = AnnealRidge(ridge_floor=1e-8, decrease=0.1)
user_state0 = callback.init_state()
ridge_solver = RidgeLevenbergMarquardt(
    residual,
    metric=RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=repeats),
    ridge=1e-4,
    linear_solver=QR(),
)
ridge_result = ridge_solver.solve(
    jnp.zeros(p_dim),
    max_steps=500,
    gtol=1e-15,
    atol=1e-8,
    callback=callback,
    user_state=user_state0,
)
assert int(ridge_result.status) == 1
x_ridge = np.asarray(ridge_result.x)

np.testing.assert_allclose(x_ridge, x_dagger, atol=1e-6)
np.testing.assert_allclose(x_metric, x_dagger, atol=1e-6)
np.testing.assert_allclose(x_ridge, x_metric, atol=1e-6)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_float64_block_eigen_preconditioner_default_precision():
    script = r"""
import dataclasses

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from nlls_gram import (
    CG,
    BlockEigenPreconditioner,
    Cholesky,
    LMStatus,
    SolverContext,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    LMState,
)

# apply == dense inverse of the shifted block-diagonal approximation, at
# float64 accuracy, for both a positive and a zero damping (the AD role).
keys = jax.random.split(jax.random.key(0), 3)


def spd_blocks(key, groups, size):
    root = jax.random.normal(key, (groups, size, size))
    return jnp.einsum("gik,gjk->gij", root, root) + 0.5 * jnp.eye(size)


family_a = spd_blocks(keys[0], 2, 3)
family_free = spd_blocks(keys[1], 1, 2)
permutation = jnp.asarray(np.random.default_rng(0).permutation(8))
families = [(family_a, 1.0), (family_free, 0.0)]
preconditioner = BlockEigenPreconditioner(families, permutation)
for leaf in jax.tree.leaves(preconditioner):
    assert jnp.issubdtype(leaf.dtype, jnp.integer) or leaf.dtype == jnp.float64, (
        leaf.dtype
    )
ridge = 3e-9
ctx = SolverContext(
    lm_state=LMState(damping=jnp.asarray(1e-3), ridge=jnp.asarray(ridge)),
)
v = jax.random.normal(keys[2], (8,))
selection = jnp.eye(8)[permutation]
dense_permuted = jsp_linalg.block_diag(family_a[0], family_a[1], family_free[0])
ridge_mask = jnp.concatenate([jnp.ones(6), jnp.zeros(2)])
for damping in (0.0, 0.37):
    shifted = dense_permuted + jnp.diag(ridge_mask * ridge + damping)
    expected = jnp.linalg.solve(selection.T @ shifted @ selection, v)
    got = preconditioner.apply(v, jnp.asarray(damping), ctx)
    assert got.dtype == jnp.float64
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)

# End-to-end at the production precision: CG with the block-eigen state
# (including a mid-solve callback rebuild) matches the Cholesky solution,
# and the tangent through the zero-damping AD-role CG matches too.
REPEATS = 2
BLOCK = 4
N_M = REPEATS * BLOCK
N_F = 2
P_DIM = N_M + N_F
M_RESID = 14
RIDGE = 1e-8

root = jax.random.normal(jax.random.key(2), (BLOCK, BLOCK + 2))
K = root @ root.T + 0.5 * jnp.eye(BLOCK)
F = jnp.linalg.cholesky(K, upper=True)
metric = RepeatedFactorMetric(F, repeats=REPEATS)
A = jax.random.normal(jax.random.key(3), (M_RESID, P_DIM)) / jnp.sqrt(P_DIM)
target = jax.random.normal(jax.random.key(4), (M_RESID,))


def residual(x, args, p):
    return A @ x - p["scale"] * target


F_bar = jsp_linalg.block_diag(F, F, jnp.eye(N_F))
J_whitened = jnp.linalg.solve(F_bar.T, A.T).T
G = J_whitened.T @ J_whitened


def exact_preconditioner():
    return BlockEigenPreconditioner(
        [(G[:N_M, :N_M][None], 1.0), (G[N_M:, N_M:][None], 0.0)],
        jnp.arange(P_DIM),
    )


p_value = {"scale": jnp.asarray(1.0)}
p_dot = {"scale": jnp.asarray(1.0)}
x0 = jnp.zeros(P_DIM)
args = {"data": jnp.asarray(1.0)}
solve_options = dict(max_steps=80, gtol=1e-10, xtol=1e-14)

reference_solver = RidgeLevenbergMarquardt(
    residual, metric=metric, ridge=RIDGE, linear_solver=Cholesky()
)
reference = reference_solver.solve(x0, args, p=p_value, **solve_options)
assert int(reference.status) == int(LMStatus.CONVERGED)


cg_solver = RidgeLevenbergMarquardt(
    residual,
    metric=metric,
    ridge=RIDGE,
    linear_solver=CG(exact_preconditioner(), tol=1e-12, maxiter=400),
    ad_solver=CG(exact_preconditioner(), tol=1e-12, maxiter=400),
)
result = cg_solver.solve(x0, args, p=p_value, **solve_options)
assert int(result.status) == int(LMStatus.CONVERGED)
# Matched to the tangent comparison below. The two solves stop on the same
# gtol, and a ridge-scaled stopping rule leaves x-slack ~ gtol / ridge
# (1e-2 here) in the weakly curved directions, so agreement is a measured
# property and NOT bounded by the CG tolerance: platform BLAS ordering
# moves the last accepted step. Measured ~1.6e-10 absolute.
np.testing.assert_allclose(np.asarray(result.x), np.asarray(reference.x),
                           rtol=1e-7, atol=1e-9)


def tangent(solver):
    def run(p_in):
        return solver.solve(x0, args, p=p_in, **solve_options).x

    return jax.jvp(run, (p_value,), (p_dot,))[1]


np.testing.assert_allclose(
    np.asarray(tangent(cg_solver)),
    np.asarray(tangent(reference_solver)),
    rtol=1e-7,
    atol=1e-9,
)

# The whole CG solve traces float64-only.
solve_jaxpr = str(
    jax.make_jaxpr(
        lambda p_in: cg_solver.solve(x0, args, p=p_in, **solve_options).x
    )(p_value)
)
assert "f32" not in solve_jaxpr
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
