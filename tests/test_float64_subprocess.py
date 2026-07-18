import subprocess
import sys
import textwrap


def test_float64_plain_and_nnx_paths_do_not_use_float32():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx

from nlls_gram import (
    UnderdeterminedLevenbergMarquardt,
    identity_preconditioner,
    nystrom_preconditioner,
)


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
    dual_preconditioner=identity_preconditioner(),
    implicit_preconditioner=identity_preconditioner(),
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


# Nystrom build + apply traced end to end: the sketch, Cholesky, and SVD must
# all stay float64.
n_dual = 6
G_psd = jax.random.normal(jax.random.PRNGKey(0), (n_dual, n_dual), dtype=jnp.float64)
A_psd = G_psd @ G_psd.T + jnp.eye(n_dual)


def build_and_apply(A, v):
    preconditioner = nystrom_preconditioner(
        lambda X: A @ X, n_dual, n_dual, jax.random.PRNGKey(1)
    )
    return preconditioner(v, jnp.asarray(1e-3, v.dtype))


v_dual = jnp.ones(n_dual, dtype=jnp.float64)
assert build_and_apply(A_psd, v_dual).dtype == jnp.float64
jaxpr = str(jax.make_jaxpr(build_and_apply)(A_psd, v_dual))
assert "f32" not in jaxpr, jaxpr

# An explicit float32 dtype stays float32 even with x64 enabled.
pre32 = nystrom_preconditioner(
    lambda X: A_psd.astype(jnp.float32) @ X,
    n_dual,
    3,
    jax.random.PRNGKey(2),
    dtype=jnp.float32,
)
assert pre32(jnp.ones(n_dual, jnp.float32), jnp.float32(0.5)).dtype == jnp.float32

# A cg solver whose dual preconditioner is a (float64) Nystrom sketch of
# J J' keeps the whole update float64.
solver = UnderdeterminedLevenbergMarquardt(
    linear_residual,
    init_damping=1e-2,
    linear_solver="cg",
    iterative_tol=1e-10,
    iterative_maxiter=20,
    dual_preconditioner=nystrom_preconditioner(
        lambda V: matrix @ (matrix.T @ V),
        matrix.shape[0],
        matrix.shape[0],
        jax.random.PRNGKey(3),
    ),
    implicit_preconditioner=identity_preconditioner(),
)
lm_state = solver.init(theta, (matrix, target))
theta_nystrom, lm_state, info = solver.update(theta, lm_state, (matrix, target))

assert theta_nystrom.dtype == jnp.float64
assert info.loss.dtype == jnp.float64
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


def test_dual_solve_dtype_promotes_dense_dual_solve():
    # A 1e-7 metric weight injects a 1/eps spike into the dual, driving
    # cond(J P J') ~ 1e7: the float32 cholesky paths lose the step and the
    # implicit derivative, while dual_solve_dtype=jnp.float64 recovers the
    # float64 reference on the SAME float32-representable data to ~1e-6,
    # with every output still float32.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt, metric_from_diagonal

n, m, eps = 12, 4, 1e-7
A32 = jax.random.normal(jax.random.PRNGKey(0), (m, n), dtype=jnp.float32)
b32 = jax.random.normal(jax.random.PRNGKey(1), (m,), dtype=jnp.float32)
w32 = jnp.concatenate([jnp.array([eps], jnp.float32), jnp.ones(n - 1, jnp.float32)])
# The reference solves the SAME problem (float32 values are exactly
# representable in float64), isolating solve error from data rounding.
A64, b64, w64 = (v.astype(jnp.float64) for v in (A32, b32, w32))


def make(matrix, target, weights, dual_dtype):
    def residual(theta, _, p):
        return matrix @ theta - p * target

    return UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-3,
        metric=metric_from_diagonal(weights),
        geodesic_acceleration=False,
        dual_solve_dtype=dual_dtype,
    )


plain32 = make(A32, b32, w32, None)
promoted = make(A32, b32, w32, jnp.float64)
reference = make(A64, b64, w64, None)
p32, p64 = jnp.float32(1.0), jnp.float64(1.0)
t032, t064 = jnp.zeros(n, jnp.float32), jnp.zeros(n, jnp.float64)


def rel(value, ref):
    difference = value.astype(jnp.float64) - ref
    return float(jnp.linalg.norm(difference) / jnp.linalg.norm(ref))


# Forward step.
x_plain = plain32.update(t032, plain32.init(t032, p=p32), p=p32)[0]
x_promoted = promoted.update(t032, promoted.init(t032, p=p32), p=p32)[0]
x_reference = reference.update(t064, reference.init(t064, p=p64), p=p64)[0]
assert x_promoted.dtype == jnp.float32
assert rel(x_promoted, x_reference) < 1e-6, rel(x_promoted, x_reference)
assert rel(x_plain, x_reference) > 1e-3, rel(x_plain, x_reference)


# Implicit JVP and VJP through the (auto-resolved) dense implicit rule.
def solved_x(solver, theta0, p_value):
    return solver.solve(theta0, p=p_value, max_steps=80, atol=0.0, gtol=1e-5).x


def tangent(solver, theta0, p_value):
    return jax.jvp(
        lambda q: solved_x(solver, theta0, q),
        (p_value,),
        (jnp.ones((), p_value.dtype),),
    )[1]


t_promoted = tangent(promoted, t032, p32)
t_reference = tangent(reference, t064, p64)
assert t_promoted.dtype == jnp.float32
assert rel(t_promoted, t_reference) < 1e-6, rel(t_promoted, t_reference)
assert rel(tangent(plain32, t032, p32), t_reference) > 1e-3


def summed_gradient(solver, theta0, p_value):
    return jax.grad(lambda q: jnp.sum(solved_x(solver, theta0, q)))(p_value)


g_promoted = float(summed_gradient(promoted, t032, p32))
g_reference = float(summed_gradient(reference, t064, p64))
g_plain = float(summed_gradient(plain32, t032, p32))
assert abs(g_promoted - g_reference) / abs(g_reference) < 1e-5
assert abs(g_plain - g_reference) / abs(g_reference) > 1e-2


# On a well-conditioned problem the flag changes nothing beyond float32
# rounding, and the qr-forward + dense-implicit consumer combination is
# accepted.
well = jnp.ones(n, jnp.float32)
plain_well = make(A32, b32, well, None)
promoted_well = make(A32, b32, well, jnp.float64)
x_plain_well = plain_well.update(t032, plain_well.init(t032, p=p32), p=p32)[0]
x_promoted_well = promoted_well.update(
    t032, promoted_well.init(t032, p=p32), p=p32
)[0]
assert jnp.allclose(x_promoted_well, x_plain_well, rtol=1e-5, atol=1e-6)

qr_dense_implicit = UnderdeterminedLevenbergMarquardt(
    lambda theta, _, p: A32 @ theta - p * b32,
    linear_solver="qr",
    geodesic_acceleration=False,
    dual_solve_dtype=jnp.float64,
)
qr_tangent = tangent(qr_dense_implicit, t032, p32)
assert qr_tangent.dtype == jnp.float32
assert bool(jnp.all(jnp.isfinite(qr_tangent)))


# Geodesic acceleration reuses the promoted solve_step: a nonlinear promoted
# update matches the float64 reference, and the diagnostics stay float32.
def make_geodesic(matrix, target, weights, dual_dtype):
    def residual(theta, _, p):
        linear = matrix @ theta
        return linear + 0.05 * linear**2 - p * target

    return UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-3,
        metric=metric_from_diagonal(weights),
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=10.0,
        dual_solve_dtype=dual_dtype,
    )


geo_promoted = make_geodesic(A32, b32, w32, jnp.float64)
geo_reference = make_geodesic(A64, b64, w64, None)
xg_promoted, _, info_promoted = geo_promoted.update(
    t032, geo_promoted.init(t032, p=p32), p=p32
)
xg_reference, _, info_reference = geo_reference.update(
    t064, geo_reference.init(t064, p=p64), p=p64
)
assert xg_promoted.dtype == jnp.float32
assert info_promoted.acceleration_ratio.dtype == jnp.float32
assert bool(info_promoted.used_geodesic) == bool(info_reference.used_geodesic)
assert rel(xg_promoted, xg_reference) < 1e-5, rel(xg_promoted, xg_reference)


# Direct differentiation THROUGH update (no implicit rule) on the promoted
# path stays valid in both modes and matches the float64 reference.
def update_x(solver, theta0, p_value):
    return solver.update(theta0, solver.init(theta0, p=p_value), p=p_value)[0]


_, u_dot_promoted = jax.jvp(
    lambda q: update_x(promoted, t032, q), (p32,), (jnp.float32(1.0),)
)
_, u_dot_reference = jax.jvp(
    lambda q: update_x(reference, t064, q), (p64,), (jnp.float64(1.0),)
)
assert u_dot_promoted.dtype == jnp.float32
assert rel(u_dot_promoted, u_dot_reference) < 1e-5, rel(u_dot_promoted, u_dot_reference)

_, pull_promoted = jax.vjp(lambda q: update_x(promoted, t032, q), p32)
_, pull_reference = jax.vjp(lambda q: update_x(reference, t064, q), p64)
cotangent_promoted = float(pull_promoted(jnp.ones(n, jnp.float32))[0])
cotangent_reference = float(pull_reference(jnp.ones(n, jnp.float64))[0])
assert abs(cotangent_promoted - cotangent_reference) / abs(cotangent_reference) < 1e-5
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_augmented_qr_float64_and_dtype_policy():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import LMStatus, UnderdeterminedLevenbergMarquardt

W = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3), dtype=jnp.float64)


def residual(z, args, p):
    return z + jnp.tanh(W @ z) - p


solver = UnderdeterminedLevenbergMarquardt(
    residual,
    linear_solver="augmented_qr",
    geodesic_acceleration=False,
    cache_jacobian=False,
)
p = jax.random.normal(jax.random.PRNGKey(1), (3,), dtype=jnp.float64)
result = solver.solve(
    jnp.zeros(3, dtype=jnp.float64), p=p, max_steps=50, atol=1e-10
)

assert int(result.status) == LMStatus.CONVERGED
assert result.x.dtype == jnp.float64
assert result.info.loss.dtype == jnp.float64
assert float(jnp.sqrt(result.info.loss)) < 1e-10

jaxpr = str(
    jax.make_jaxpr(
        lambda q: solver.solve(
            jnp.zeros(3, dtype=jnp.float64), p=q, max_steps=50, atol=1e-10
        ).x
    )(p)
)
assert "f32" not in jaxpr, jaxpr

jvp_jaxpr = str(
    jax.make_jaxpr(
        lambda q, q_dot: jax.jvp(
            lambda r: solver.solve(
                jnp.zeros(3, dtype=jnp.float64), p=r, max_steps=50, atol=1e-10
            ).x,
            (q,),
            (q_dot,),
        )[1]
    )(p, p)
)
assert "f32" not in jvp_jaxpr, jvp_jaxpr

x, x_dot = jax.jvp(
    lambda q: solver.solve(
        jnp.zeros(3, dtype=jnp.float64), p=q, max_steps=50, atol=1e-10
    ).x,
    (p,),
    (jnp.ones(3, dtype=jnp.float64),),
)
assert x.dtype == jnp.float64 and x_dot.dtype == jnp.float64
J = jax.jacfwd(lambda z: residual(z, None, p))(x)
assert jnp.allclose(x_dot, jnp.linalg.solve(J, jnp.ones(3)), atol=1e-10)

# A float32 problem under enabled x64 stays float32.
W32 = W.astype(jnp.float32)


def residual32(z, args, p):
    return z + jnp.tanh(W32 @ z) - p


solver32 = UnderdeterminedLevenbergMarquardt(
    residual32,
    linear_solver="augmented_qr",
    geodesic_acceleration=False,
    cache_jacobian=False,
)
result32 = solver32.solve(
    jnp.zeros(3, dtype=jnp.float32),
    p=p.astype(jnp.float32),
    max_steps=50,
    atol=1e-6,
)
assert int(result32.status) == LMStatus.CONVERGED
assert result32.x.dtype == jnp.float32
assert result32.info.loss.dtype == jnp.float32
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


def test_repeated_blockdiag_metric_float64_matches_blockdiag():
    # In genuine float64: the batched repeated metric matches the expanded
    # blockdiag metric to machine precision on every callback (vector and
    # matrix), fires the base callback once, preserves the compute-dtype
    # round-trip, and stays exact when nested inside blockdiag_metric (the
    # human_capital composition). Norm AD matches forward and reverse.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import (
    Metric,
    blockdiag_metric,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_with_compute_dtype,
    repeated_blockdiag_metric,
)


def rel(a, b):
    return float(jnp.linalg.norm(jnp.ravel(a - b)) / jnp.linalg.norm(jnp.ravel(a)))


block_size, repeats = 4, 3
A = jnp.array(
    [
        [2.0, 0.2, 0.1, 0.0],
        [0.2, 1.8, 0.0, 0.1],
        [0.1, 0.0, 1.5, 0.2],
        [0.0, 0.1, 0.2, 1.2],
    ]
)
weights = jnp.array([0.3, 0.7])
block = metric_from_cholesky(jnp.linalg.cholesky(A))
additional = metric_from_diagonal(weights)
repeated = repeated_blockdiag_metric(
    block, block_size, repeats, additional=(additional, weights.shape[0])
)
reference = blockdiag_metric(
    [(block, block_size)] * repeats + [(additional, weights.shape[0])]
)
total = block_size * repeats + weights.shape[0]
x = jax.random.normal(jax.random.PRNGKey(0), (total,))
X = jax.random.normal(jax.random.PRNGKey(1), (total, 5))

for callback in ("solve", "inv_sqrt", "inv_sqrt_transpose"):
    for arg in (x, X):
        a = getattr(reference, callback)(arg)
        b = getattr(repeated, callback)(arg)
        assert rel(a, b) < 1e-12, (callback, arg.ndim, rel(a, b))
assert rel(reference.norm(x), repeated.norm(x)) < 1e-12

# The base callback fires exactly once, on the combined (block_size, repeats*k).
received = []


def record(v):
    received.append(v.shape)
    return v


recorded = repeated_blockdiag_metric(Metric(solve=record), block_size, repeats)
recorded.solve(jnp.ones((block_size * repeats, 5)))
assert received == [(block_size, repeats * 5)], received

# Compute-dtype wrapper: exact in float64, and a float32 input round-trips.
wrapped = metric_with_compute_dtype(reference, jnp.float64)
for callback in ("solve", "inv_sqrt", "inv_sqrt_transpose", "norm"):
    arg = x if callback == "norm" else X
    a = getattr(reference, callback)(arg)
    b = getattr(wrapped, callback)(arg)
    assert rel(a, b) < 1e-12, (callback, rel(a, b))
x32 = x.astype(jnp.float32)
solved32 = wrapped.solve(x32)
assert solved32.dtype == jnp.float32, solved32.dtype
wide = reference.solve(x32.astype(jnp.float64))
assert rel(wide, solved32.astype(jnp.float64)) < 1e-6

# Nested composition (human_capital pattern): a repeated block with no
# additional term, composed inside blockdiag_metric with a heterogeneous
# trailing block, equals the fully-expanded blockdiag_metric.
other = metric_from_diagonal(jnp.array([1.5, 0.4, 2.0]))
repeated_no_additional = repeated_blockdiag_metric(block, block_size, 2)
nested = blockdiag_metric([(repeated_no_additional, 2 * block_size), (other, 3)])
expanded = blockdiag_metric([(block, block_size)] * 2 + [(other, 3)])
nested_total = 2 * block_size + 3
y = jax.random.normal(jax.random.PRNGKey(2), (nested_total,))
Y = jax.random.normal(jax.random.PRNGKey(3), (nested_total, 4))
for callback in ("solve", "inv_sqrt", "inv_sqrt_transpose"):
    for arg in (y, Y):
        a = getattr(expanded, callback)(arg)
        b = getattr(nested, callback)(arg)
        assert rel(a, b) < 1e-12, ("nested", callback, arg.ndim, rel(a, b))
assert rel(expanded.norm(y), nested.norm(y)) < 1e-12

# Norm AD (forward + reverse) through the nested composite matches the
# expanded reference to machine precision.
dy = jax.random.normal(jax.random.PRNGKey(4), (nested_total,))
_, tangent_nested = jax.jvp(nested.norm, (y,), (dy,))
_, tangent_expanded = jax.jvp(expanded.norm, (y,), (dy,))
assert rel(tangent_expanded, tangent_nested) < 1e-12
assert rel(jax.grad(expanded.norm)(y), jax.grad(nested.norm)(y)) < 1e-12
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
    identity_preconditioner,
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
    # implicit_penalty=0.0: the eps-shifted metric spikes the dual trace by
    # 1/eps, so a trace-scaled ridge would perturb this exact-limit check.
    solver = UnderdeterminedLevenbergMarquardt(
        residual,
        init_damping=1e-6,
        metric=metric,
        geodesic_acceleration=False,
        implicit_penalty=0.0,
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
        implicit_preconditioner=identity_preconditioner(),
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


def test_float64_dense_implicit_penalty_near_duplicate_rows():
    # The growth-model pathology: a converged simulation duplicates its
    # late-horizon states to ~1e-13, so the float64 undamped implicit dual has
    # eigenvalues far below the factorization noise floor and the unregularized
    # dense rule goes non-finite. The default ridge must return the min-norm
    # tangent d sum(x*)/d target = sum(w)/||w||^2 (exact in the duplicate
    # limit) to high accuracy.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt

w = jnp.array([1.0, 2.0, 3.0])
wiggles = 1.0 + 1e-13 * jnp.arange(40.0)


def residual_fn(x, args, p):
    return wiggles * (jnp.dot(w, x) - p["target"])


x0 = jnp.zeros(3)
solver = UnderdeterminedLevenbergMarquardt(residual_fn)


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


def test_float64_deflated_pcg_build_and_harvest():
    # In genuine float64: build_coarse_operator, deflated_pcg, and the eigCG
    # harvest stay float64 end to end (no f32 in the jaxpr); U=0 reproduces
    # recycled_cg bitwise; an exact-eigenvector basis cuts the iteration count;
    # the harvested basis matches the true smallest eigenvectors to a tight f64
    # bound; and the implicit gradient matches the dense linear solve.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram.recycled_cg import build_coarse_operator, deflated_pcg, recycled_cg

n, k = 50, 4
Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (n, n)))
small = jnp.array([0.01, 0.02, 0.04, 0.08])
bulk = 1.0 + 1e-4 * jax.random.uniform(jax.random.key(2), (n - k,))
eigs = jnp.concatenate([small, bulk])
A = (Q * eigs) @ Q.T
A = 0.5 * (A + A.T)
b = jax.random.normal(jax.random.key(1), (n,))
w = 3 * k


def matvec(v):
    return A @ v


# dtypes stay float64
U0 = jnp.zeros((n, k))
W, E_factor = build_coarse_operator(matvec, U0)
assert W.dtype == jnp.float64
assert E_factor[0].dtype == jnp.float64
y, harvest = deflated_pcg(
    matvec, b, U=U0, E_factor=E_factor, tol=1e-10, atol=0.0,
    maxiter=200, window=w, rank=k,
)
assert y.dtype == jnp.float64
assert harvest.basis.dtype == jnp.float64
assert harvest.residual_norm.dtype == jnp.float64
assert jnp.allclose(y, jnp.linalg.solve(A, b), rtol=1e-8, atol=1e-8)

# no f32 anywhere in the traced program
jaxpr = str(
    jax.make_jaxpr(
        lambda rhs: deflated_pcg(
            matvec, rhs, U=U0, E_factor=E_factor, tol=1e-10, atol=0.0,
            maxiter=200, window=w, rank=k,
        )[0]
    )(b)
)
assert "f32" not in jaxpr, jaxpr

# U=0 reproduces recycled_cg bitwise (non-identity first-level P)
weights = jnp.diag(A)


def P(v):
    return v / weights


yd, _ = deflated_pcg(
    matvec, b, U=U0, E_factor=E_factor, M=P,
    tol=1e-10, atol=0.0, maxiter=300, window=w, rank=k,
)
yr, _ = recycled_cg(matvec, b, tol=1e-10, atol=0.0, maxiter=300, M=P)
assert bool(jnp.array_equal(yd, yr))

# exact-eigenvector basis cuts iterations
order = jnp.argsort(eigs)
U_exact = Q[:, order[:k]]
_, cold = deflated_pcg(
    matvec, b, U=U0, E_factor=E_factor, tol=1e-10, atol=0.0,
    maxiter=300, window=w, rank=k,
)
_, defl = deflated_pcg(
    matvec, b, U=U_exact, E_factor=build_coarse_operator(matvec, U_exact)[1],
    tol=1e-10, atol=0.0, maxiter=300, window=w, rank=k,
)
assert int(defl.iterations) < int(cold.iterations)

# harvested basis matches the true smallest eigenvectors to a tight f64 bound
_, evecs = jnp.linalg.eigh(A)
cos_angles = jnp.linalg.svd(harvest.basis.T @ evecs[:, :k], compute_uv=False)
assert float(jnp.min(cos_angles)) > 1.0 - 1e-8, float(jnp.min(cos_angles))
assert float(jnp.max(jnp.abs(harvest.basis.T @ harvest.basis - jnp.eye(k)))) < 1e-12

# implicit gradient matches the dense solve to a tight f64 bound
A_inv = jnp.linalg.inv(A)


def loss(rhs):
    x, _ = deflated_pcg(
        matvec, rhs, U=U_exact,
        E_factor=build_coarse_operator(matvec, U_exact)[1],
        tol=1e-12, atol=0.0, maxiter=300, window=w, rank=k,
    )
    return jnp.sum(x**2)


got = jax.grad(loss)(b)
expected = jax.grad(lambda rhs: jnp.sum((A_inv @ rhs) ** 2))(b)
assert jnp.allclose(got, expected, rtol=1e-9, atol=1e-9), float(
    jnp.max(jnp.abs(got - expected))
)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_float64_multi_start_modes_and_float32_data_under_x64():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import (
    LMSolveAction,
    LMStatus,
    MultiStart,
    UnderdeterminedLevenbergMarquardt,
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
    return LMSolveAction(stop=stop, status=status)


callback_solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
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


solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
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
