import subprocess
import sys
import textwrap


def test_float64_plain_and_nnx_paths_do_not_use_float32():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx

from nlls_gram import UnderdeterminedLevenbergMarquardt


def assert_float64_tree(tree):
    leaves = jax.tree.leaves(tree)
    assert leaves
    for leaf in leaves:
        assert leaf.dtype == jnp.float64, (leaf.dtype, leaf)


def residual_fn(x, args, p):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


ts = jnp.linspace(0.0, 2.0, 20, dtype=jnp.float64)
ys = 2.0 * jnp.exp(-1.0 * ts)
x = {
    "a": jnp.asarray(1.0, dtype=jnp.float64),
    "b": jnp.asarray(0.0, dtype=jnp.float64),
}
solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init(x, (ts, ys))
for _ in range(5):
    x, lm_state, info = solver.update(x, lm_state, (ts, ys))

assert_float64_tree(x)
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(jax.make_jaxpr(lambda p, s: solver.update(p, s, (ts, ys)))(x, lm_state))
assert "f32" not in jaxpr, jaxpr
solve_jaxpr = str(
    jax.make_jaxpr(
        lambda p: solver.solve(
            p, (ts, ys), max_steps=20, atol=1e-8, gtol=1e-10, xtol=1e-10
        ).x
    )(x)
)
assert "f32" not in solve_jaxpr, solve_jaxpr


def quadratic_residual(theta, target, p):
    return jnp.array([theta[0] ** 2 - target])


theta = jnp.asarray([1.9], dtype=jnp.float64)
target = jnp.asarray(4.0, dtype=jnp.float64)
solver = UnderdeterminedLevenbergMarquardt(
    quadratic_residual,
    init_damping=1e-12,
    geodesic_acceleration=True,
    geodesic_acceptance_ratio=1.0,
)
lm_state = solver.init(theta, target)
theta, lm_state, info = solver.update(theta, lm_state, target)

assert theta.dtype == jnp.float64
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(jax.make_jaxpr(lambda p, s: solver.update(p, s, target))(theta, lm_state))
assert "f32" not in jaxpr, jaxpr
solve_jaxpr = str(
    jax.make_jaxpr(
        lambda p: solver.solve(p, target, max_steps=20, atol=1e-10).x
    )(theta)
)
assert "f32" not in solve_jaxpr, solve_jaxpr


def linear_residual(theta, args, p):
    matrix, target = args
    return matrix @ theta - target


matrix = jnp.asarray([[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]], dtype=jnp.float64)
target = jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float64)
theta = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
solver = UnderdeterminedLevenbergMarquardt(
    linear_residual,
    init_damping=1e-2,
    linear_solver="cg",
    iterative_tol=1e-10,
    iterative_maxiter=20,
)
lm_state = solver.init(theta, (matrix, target))
theta, lm_state, info = solver.update(theta, lm_state, (matrix, target))

assert theta.dtype == jnp.float64
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(
    jax.make_jaxpr(lambda p, s: solver.update(p, s, (matrix, target)))(theta, lm_state)
)
assert "f32" not in jaxpr, jaxpr


matrix = jnp.asarray(
    [[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 2.0]],
    dtype=jnp.float64,
)
target = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
theta = jnp.zeros(matrix.shape[1], dtype=jnp.float64)
solver = UnderdeterminedLevenbergMarquardt(
    linear_residual,
    init_damping=1e-2,
    linear_solver="qr",
)
lm_state = solver.init(theta, (matrix, target))
theta, lm_state, info = solver.update(theta, lm_state, (matrix, target))

assert theta.dtype == jnp.float64
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(
    jax.make_jaxpr(lambda p, s: solver.update(p, s, (matrix, target)))(theta, lm_state)
)
assert "f32" not in jaxpr, jaxpr


class LinearModel(nnx.Module):
    def __init__(self):
        self.linear = nnx.Linear(
            1,
            1,
            use_bias=False,
            dtype=jnp.float64,
            param_dtype=jnp.float64,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x):
        return jnp.ravel(self.linear(x))


model = LinearModel()
graphdef, nnx_params = nnx.split(model, nnx.Param)
assert_float64_tree(nnx_params)

x_nnx = jnp.linspace(0.0, 2.0, 20, dtype=jnp.float64).reshape(-1, 1)
y_nnx = 2.0 * jnp.ravel(x_nnx)


def nnx_residual_fn(x, args, p):
    ts, ys = args
    model = nnx.merge(graphdef, x)
    return model(ts) - ys


solver = UnderdeterminedLevenbergMarquardt(nnx_residual_fn, init_damping=1e-12)
lm_state = solver.init(nnx_params, (x_nnx, y_nnx))
nnx_params, lm_state, info = solver.update(nnx_params, lm_state, (x_nnx, y_nnx))
trained = nnx.merge(graphdef, nnx_params)

assert_float64_tree(nnx_params)
assert trained.linear.kernel[...].dtype == jnp.float64
assert jnp.allclose(trained.linear.kernel[...], jnp.asarray([[2.0]], dtype=jnp.float64))
assert lm_state.damping.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
assert info.loss_old.dtype == jnp.float64
assert info.loss_candidate.dtype == jnp.float64
assert info.damping.dtype == jnp.float64
assert info.damping_factor.dtype == jnp.float64
assert info.acceleration_ratio.dtype == jnp.float64
assert info.grad_norm.dtype == jnp.float64
assert info.step_norm.dtype == jnp.float64
jaxpr = str(
    jax.make_jaxpr(lambda p, s: solver.update(p, s, (x_nnx, y_nnx)))(
        nnx_params, lm_state
    )
)
assert "f32" not in jaxpr, jaxpr
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_quasiseparable_float64_parallel_matches_sequential():
    # The evidence gating the parallel-apply default (off-CPU + float64):
    # sequential and associative-scan applies must agree, and the parallel
    # path must stay finite, on both a well-conditioned grid and a long,
    # stiff stress grid. Also the tight float64 hyperparameter-gradient
    # cross-check against a dense metric, with the metric constructed inside
    # jax.grad and jax.jit(jax.grad(...)).
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import matern_state_space, metric_from_state_space

n = 5000
sigma, ell = 1.3, 0.8
x = jax.random.normal(jax.random.PRNGKey(0), (n,))
X = jax.random.normal(jax.random.PRNGKey(1), (n, 3))
well = jnp.cumsum(
    jax.random.uniform(jax.random.PRNGKey(2), (n,), minval=0.6, maxval=1.4)
)
stiff = jnp.linspace(0.0, 5.0, n)
# On the well-conditioned grid the two scan orders agree to machine
# precision; on the stiff grid (nugget-floored pivots, cond ~1e9) the
# substitutions are condition-limited to ~cond * eps.
for (name, t), tol in ((("well", well), 1e-12), (("stiff", stiff), 1e-6)):
    for nu in (1.5, 2.5):
        model = matern_state_space(sigma, ell, nu)
        seq = metric_from_state_space(
            t, *model, nugget=1e-8 * sigma**2, parallel=False
        )
        par = metric_from_state_space(
            t, *model, nugget=1e-8 * sigma**2, parallel=True
        )
        for callback in ("solve", "norm", "inv_sqrt", "inv_sqrt_transpose"):
            a = getattr(seq, callback)(x)
            b = getattr(par, callback)(x)
            assert bool(jnp.all(jnp.isfinite(b))), (name, nu, callback)
            rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))
            assert rel < tol, (name, nu, callback, rel)
        rel = float(
            jnp.linalg.norm(seq.solve(X) - par.solve(X)) / jnp.linalg.norm(seq.solve(X))
        )
        assert rel < tol, (name, nu, "solve-matrix", rel)


def dense_matern_gram(t, sigma, ell, nu):
    tau = jnp.abs(t[:, None] - t[None, :])
    ft = jnp.sqrt(2.0 * nu) * tau / ell
    if nu == 1.5:
        corr = (1.0 + ft) * jnp.exp(-ft)
    else:
        corr = (1.0 + ft + ft**2 / 3.0) * jnp.exp(-ft)
    return sigma**2 * corr


n = 300
t = jnp.cumsum(jax.random.uniform(jax.random.PRNGKey(3), (n,), minval=0.6, maxval=1.4))
v = jax.random.normal(jax.random.PRNGKey(4), (n,))
for nu in (1.5, 2.5):

    def loss_qsm(params, nu=nu):
        s, l = params
        model = matern_state_space(s, l, nu)
        return v @ metric_from_state_space(t, *model, nugget=1e-8).solve(v)

    def loss_dense(params, nu=nu):
        s, l = params
        K = dense_matern_gram(t, s, l, nu) + 1e-8 * jnp.eye(n)
        return v @ jnp.linalg.solve(K, v)

    params = jnp.array([1.3, 0.8])
    grad_qsm = jax.grad(loss_qsm)(params)
    grad_jit = jax.jit(jax.grad(loss_qsm))(params)
    grad_dense = jax.grad(loss_dense)(params)
    rel = float(jnp.linalg.norm(grad_qsm - grad_dense) / jnp.linalg.norm(grad_dense))
    assert rel < 1e-9, (nu, rel)
    rel_jit = float(jnp.linalg.norm(grad_jit - grad_qsm) / jnp.linalg.norm(grad_qsm))
    assert rel_jit < 1e-12, (nu, rel_jit)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_shifted_metric_eps_limit_matches_kkt():
    # The unified shifted metric M = blockdiag(K, 0) + eps I: as eps -> 0
    # the minimum-M-norm interpolant converges at rate O(eps) to the
    # bordered-KKT solution of min alpha' K alpha s.t. J theta = b with
    # beta free -- under the uniqueness conditions (K PD, J full row rank,
    # J_beta full column rank) that make that solution unique. Also: the
    # matrix-free composite matches the dense composite, and the implicit
    # derivative error shrinks with the inner CG tolerance.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import (
    UnderdeterminedLevenbergMarquardt,
    blockdiag_metric,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_from_shifted_matvec,
)

n, k, m = 12, 2, 4
t = jnp.arange(n) * 1.0
ft = jnp.sqrt(3.0) * jnp.abs(t[:, None] - t[None, :]) / 0.8
K = 1.3**2 * (1.0 + ft) * jnp.exp(-ft) + 1e-8 * jnp.eye(n)  # PD
J_alpha = jax.random.normal(jax.random.PRNGKey(0), (m, n))
J_beta = jax.random.normal(jax.random.PRNGKey(1), (m, k))  # full column rank
J = jnp.concatenate([J_alpha, J_beta], axis=1)
b = jax.random.normal(jax.random.PRNGKey(2), (m,))

# Bordered KKT: [[2K, 0, J_a'], [0, 0, J_b'], [J_a, J_b, 0]] [a, B, -y] = [0,0,b]
kkt = jnp.block(
    [
        [2.0 * K, jnp.zeros((n, k)), J_alpha.T],
        [jnp.zeros((k, n)), jnp.zeros((k, k)), J_beta.T],
        [J_alpha, J_beta, jnp.zeros((m, m))],
    ]
)
kkt_solution = jnp.linalg.solve(kkt, jnp.concatenate([jnp.zeros(n + k), b]))
theta_kkt = kkt_solution[: n + k]
assert jnp.allclose(J @ theta_kkt, b, atol=1e-9)


def residual(theta, _, p):
    return J @ theta - p * b


def dense_composite(eps):
    return blockdiag_metric(
        [
            (metric_from_cholesky(jnp.linalg.cholesky(K + eps * jnp.eye(n))), n),
            (metric_from_diagonal(eps * jnp.ones(k)), k),
        ]
    )


def solved_x(metric, p):
    solver = UnderdeterminedLevenbergMarquardt(
        residual, init_damping=1e-6, metric=metric, geodesic_acceleration=False
    )
    return solver.solve(jnp.zeros(n + k), p=p, max_steps=200, atol=1e-12).x


errors = {}
for eps in (1e-3, 1e-5, 1e-7):
    theta_eps = solved_x(dense_composite(eps), jnp.asarray(1.0))
    assert jnp.allclose(J @ theta_eps, b, atol=1e-8), eps
    errors[eps] = float(jnp.linalg.norm(theta_eps - theta_kkt))

# Convergence to the KKT solution, at first order in eps.
assert errors[1e-7] < 1e-5, errors
rate = errors[1e-3] / errors[1e-5]
assert 30.0 < rate < 300.0, (errors, rate)

# The matrix-free composite matches the dense composite at eps = 1e-5.
eps = 1e-5
matvec_metric = blockdiag_metric(
    [
        (metric_from_shifted_matvec(lambda x: K @ x, eps, tol=1e-12), n),
        (metric_from_diagonal(eps * jnp.ones(k)), k),
    ]
)
theta_matvec = solved_x(matvec_metric, jnp.asarray(1.0))
theta_dense = solved_x(dense_composite(eps), jnp.asarray(1.0))
rel = float(jnp.linalg.norm(theta_matvec - theta_dense) / jnp.linalg.norm(theta_dense))
assert rel < 1e-9, rel

# Implicit-derivative error shrinks with the inner CG tolerance: the final
# metric solve has no accept/reject safeguard, so tol directly sets it.
p, p_dot = jnp.asarray(1.0), jnp.asarray(1.0)
_, dot_dense = jax.jvp(lambda q: solved_x(dense_composite(eps), q), (p,), (p_dot,))


def matvec_composite(tol):
    return blockdiag_metric(
        [
            (metric_from_shifted_matvec(lambda x: K @ x, eps, tol=tol), n),
            (metric_from_diagonal(eps * jnp.ones(k)), k),
        ]
    )


derivative_errors = {}
for tol in (1e-4, 1e-12):
    _, dot_matvec = jax.jvp(
        lambda q: solved_x(matvec_composite(tol), q), (p,), (p_dot,)
    )
    derivative_errors[tol] = float(jnp.linalg.norm(dot_matvec - dot_dense))
assert derivative_errors[1e-12] < derivative_errors[1e-4], derivative_errors
assert derivative_errors[1e-12] < 1e-8, derivative_errors

# The matrix-free cg implicit rule (metric tol at or below implicit_tol, per
# the nested-tolerance guidance) reproduces the dense-rule derivative in
# both directions.
def solved_x_cg_implicit(p_value):
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        metric=matvec_composite(1e-12),
        geodesic_acceleration=False,
        implicit_solver="cg",
        implicit_tol=1e-12,
    )
    return solver.solve(jnp.zeros(n + k), p=p_value, max_steps=200, atol=1e-12).x

_, dot_cg_implicit = jax.jvp(solved_x_cg_implicit, (p,), (p_dot,))
assert float(jnp.linalg.norm(dot_cg_implicit - dot_dense)) < 1e-8

x_bar = jnp.linspace(-1.0, 1.0, n + k)
_, pull_dense = jax.vjp(lambda q: solved_x(dense_composite(eps), q), p)
_, pull_cg_implicit = jax.vjp(solved_x_cg_implicit, p)
assert float(jnp.abs(pull_cg_implicit(x_bar)[0] - pull_dense(x_bar)[0])) < 1e-8
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

from nlls_gram import LMStatus, UnderdeterminedLevenbergMarquardt


def residual(theta, args, p):
    return theta - args


solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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
