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
    LevenbergMarquardt,
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
solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
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
solver = LevenbergMarquardt(
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
solver = LevenbergMarquardt(
    linear_residual,
    init_damping=1e-2,
    linear_solver="gram_cg",
    iterative_tol=1e-10,
    iterative_maxiter=20,
    dual_preconditioner=identity_preconditioner(),
    ad_solver_preconditioner=identity_preconditioner(),
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
solver = LevenbergMarquardt(
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
solver = LevenbergMarquardt(
    linear_residual,
    init_damping=1e-2,
    linear_solver="gram_cg",
    iterative_tol=1e-10,
    iterative_maxiter=20,
    dual_preconditioner=nystrom_preconditioner(
        lambda V: matrix @ (matrix.T @ V),
        matrix.shape[0],
        matrix.shape[0],
        jax.random.PRNGKey(3),
    ),
    ad_solver_preconditioner=identity_preconditioner(),
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


solver = LevenbergMarquardt(nnx_residual_fn, init_damping=1e-12)
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


def test_linear_solve_dtype_promotes_dense_dual_solve():
    # A 1e-7 metric weight injects a 1/eps spike into the dual, driving
    # cond(J P J') ~ 1e7: the float32 cholesky paths lose the step and the
    # implicit derivative, while linear_solve_dtype=jnp.float64 recovers the
    # float64 reference on the SAME float32-representable data to ~1e-6,
    # with every output still float32.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt, metric_from_diagonal

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

    return LevenbergMarquardt(
        residual,
        init_damping=1e-3,
        metric=metric_from_diagonal(weights),
        geodesic_acceleration=False,
        linear_solve_dtype=dual_dtype,
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
# Unlike the forward step (whose float32 cholesky forms B'B and pays the
# squared-assembly floor, > 1e-3 above), the SVD AD method factors B itself:
# the UNPROMOTED float32 tangent already tracks the float64 reference on
# this cond ~ 1e3 fixture (measured 3.9e-7), so promotion is a no-op here.
assert rel(tangent(plain32, t032, p32), t_reference) < 1e-5


def summed_gradient(solver, theta0, p_value):
    return jax.grad(lambda q: jnp.sum(solved_x(solver, theta0, q)))(p_value)


g_promoted = float(summed_gradient(promoted, t032, p32))
g_reference = float(summed_gradient(reference, t064, p64))
g_plain = float(summed_gradient(plain32, t032, p32))
assert abs(g_promoted - g_reference) / abs(g_reference) < 1e-5
# Same cond(B)-not-cond(B)^2 story in reverse mode (measured 1.1e-6).
assert abs(g_plain - g_reference) / abs(g_reference) < 1e-4


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

qr_dense_implicit = LevenbergMarquardt(
    lambda theta, _, p: A32 @ theta - p * b32,
    linear_solver="qr",
    geodesic_acceleration=False,
    linear_solve_dtype=jnp.float64,
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

    return LevenbergMarquardt(
        residual,
        init_damping=1e-3,
        metric=metric_from_diagonal(weights),
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=10.0,
        linear_solve_dtype=dual_dtype,
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


def test_linear_solve_dtype_normal_forms_and_metric_solve_dtype_jaxpr():
    # The normal-form twin of the promoted-dual test. A metric spike does NOT
    # stress this form (the graded B'B factors its huge pivot first, stably),
    # so the float32 fragility is driven the way it actually arises: columns
    # with cond(A) ~ 1e3 square to cond(B'B) ~ 1e6, and the solution's
    # small-singular-direction components lose eps32 * cond^2 ~ O(0.1) while
    # linear_solve_dtype=jnp.float64 recovers the float64 reference on the
    # SAME float32-representable data, with float32 outputs. Then the jaxpr
    # policy: float64 problems stay f32-free through the normal forward and
    # implicit paths, and metric_solve_dtype injects f64 casts into an
    # otherwise-f32 update.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import (
    LevenbergMarquardt,
    identity_preconditioner,
    metric_from_cholesky,
)

m, n = 12, 4
U, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (m, n)))
V, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (n, n)))
A32 = ((U * jnp.logspace(0.0, -3.0, n)) @ V.T).astype(jnp.float32)
b32 = (A32.astype(jnp.float64) @ jnp.arange(1.0, n + 1.0)).astype(jnp.float32)
A64, b64 = A32.astype(jnp.float64), b32.astype(jnp.float64)


def make(matrix, target, solve_dtype):
    def residual(theta, _, p):
        return matrix @ theta - p * target

    return LevenbergMarquardt(
        residual,
        init_damping=1e-8,
        linear_solver="normal_cholesky",
        geodesic_acceleration=False,
        linear_solve_dtype=solve_dtype,
    )


plain32 = make(A32, b32, None)
promoted = make(A32, b32, jnp.float64)
reference = make(A64, b64, None)
p32, p64 = jnp.float32(1.0), jnp.float64(1.0)
t032, t064 = jnp.zeros(n, jnp.float32), jnp.zeros(n, jnp.float64)


def rel(value, ref):
    difference = value.astype(jnp.float64) - ref
    return float(jnp.linalg.norm(difference) / jnp.linalg.norm(ref))


x_plain = plain32.update(t032, plain32.init(t032, p=p32), p=p32)[0]
x_promoted = promoted.update(t032, promoted.init(t032, p=p32), p=p32)[0]
x_reference = reference.update(t064, reference.init(t064, p=p64), p=p64)[0]
assert x_promoted.dtype == jnp.float32
assert rel(x_promoted, x_reference) < 1e-6, rel(x_promoted, x_reference)
assert rel(x_plain, x_reference) > 1e-3, rel(x_plain, x_reference)


def tangent(solver, theta0, p_value):
    return jax.jvp(
        lambda q: solver.solve(theta0, p=q, max_steps=80, atol=0.0, gtol=1e-5).x,
        (p_value,),
        (jnp.ones((), p_value.dtype),),
    )[1]


t_promoted = tangent(promoted, t032, p32)
t_reference = tangent(reference, t064, p64)
assert t_promoted.dtype == jnp.float32
assert rel(t_promoted, t_reference) < 1e-6, rel(t_promoted, t_reference)
# The default spectral-filter (eigh) implicit is sturdier in float32 than the
# forward Cholesky, so its degradation is milder -- measured ~6e-4 here.
assert rel(tangent(plain32, t032, p32), t_reference) > 1e-4


# jaxpr policy on float64 data: no f32 anywhere through the normal forward
# paths or the normal-resolved implicit tangents.
def residual64(theta, _, p):
    return A64 @ theta - p * b64


normal_dense = LevenbergMarquardt(
    residual64,
    init_damping=1e-3,
    linear_solver="normal_cholesky",
    geodesic_acceleration=False,
)
state64 = normal_dense.init(t064, p=p64)
jaxpr = str(
    jax.make_jaxpr(lambda th, s, q: normal_dense.update(th, s, p=q))(
        t064, state64, p64
    )
)
assert "f32" not in jaxpr, jaxpr

normal_cg = LevenbergMarquardt(
    residual64,
    init_damping=1e-3,
    linear_solver="normal_cg",
    normal_preconditioner=identity_preconditioner(),
    ad_solver_preconditioner=identity_preconditioner(),
    iterative_tol=1e-10,
    iterative_maxiter=30,
    geodesic_acceleration=False,
)
state_cg = normal_cg.init(t064, p=p64)
jaxpr = str(
    jax.make_jaxpr(lambda th, s, q: normal_cg.update(th, s, p=q))(
        t064, state_cg, p64
    )
)
assert "f32" not in jaxpr, jaxpr

for solver in (normal_dense, normal_cg):
    jaxpr = str(jax.make_jaxpr(lambda q: tangent(solver, t064, q))(p64))
    assert "f32" not in jaxpr, jaxpr

implicit_normal_cg = LevenbergMarquardt(
    residual64,
    init_damping=1e-3,
    linear_solver="normal_cholesky",
    ad_solver="normal_cg",
    ad_solver_preconditioner=identity_preconditioner(),
    ad_solver_maxiter=30,
    geodesic_acceleration=False,
)
jaxpr = str(jax.make_jaxpr(lambda q: tangent(implicit_normal_cg, t064, q))(p64))
assert "f32" not in jaxpr, jaxpr


# metric_solve_dtype on a float32 problem: the update jaxpr gains f64 casts
# around the metric callbacks (absent without the knob) while every output
# stays float32.
L32 = jnp.array([[1.3, 0.0], [0.5, 0.8]], dtype=jnp.float32)
A_t32 = jnp.array([[1.0, 0.5], [0.3, 2.0], [-1.0, 1.0]], dtype=jnp.float32)
b_t32 = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
t02 = jnp.zeros(2, jnp.float32)


def residual32(theta, _, p):
    return A_t32 @ theta - p * b_t32


def make_metric_solver(solve_dtype):
    return LevenbergMarquardt(
        residual32,
        init_damping=1e-3,
        linear_solver="normal_cholesky",
        metric=metric_from_cholesky(L32),
        metric_solve_dtype=solve_dtype,
        geodesic_acceleration=False,
    )


wide_metric = make_metric_solver(jnp.float64)
plain_metric = make_metric_solver(None)
state32 = wide_metric.init(t02, p=jnp.float32(1.0))
jaxpr_wide = str(
    jax.make_jaxpr(lambda th, s, q: wide_metric.update(th, s, p=q))(
        t02, state32, jnp.float32(1.0)
    )
)
jaxpr_plain = str(
    jax.make_jaxpr(lambda th, s, q: plain_metric.update(th, s, p=q))(
        t02, plain_metric.init(t02, p=jnp.float32(1.0)), jnp.float32(1.0)
    )
)
assert "f64" in jaxpr_wide, jaxpr_wide
assert "f64" not in jaxpr_plain, jaxpr_plain
x1, state1, info1 = wide_metric.update(t02, state32, p=jnp.float32(1.0))
assert x1.dtype == jnp.float32
assert state1.damping.dtype == jnp.float32
assert info1.loss.dtype == jnp.float32
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

from nlls_gram import LMStatus, LevenbergMarquardt

W = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3), dtype=jnp.float64)


def residual(z, args, p):
    return z + jnp.tanh(W @ z) - p


solver = LevenbergMarquardt(
    residual,
    linear_solver="augmented_qr",
    ad_solver="svd",
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


solver32 = LevenbergMarquardt(
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


def test_repeated_shifted_state_space_float64_parallel_matches_sequential():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from nlls_gram import (
    matern_state_space,
    repeated_shifted_state_space_metric,
)


def rel(a, b):
    return float(jnp.linalg.norm(jnp.ravel(a - b)) / jnp.linalg.norm(jnp.ravel(a)))


n, repeats, zero_pad_size = 3000, 3, 2
sigma, ell, epsilon = 1.3, 0.8, 1e-8
well = jnp.cumsum(
    jax.random.uniform(jax.random.PRNGKey(2), (n,), minval=0.6, maxval=1.4)
)
stiff = jnp.linspace(0.0, 5.0, n)
total = repeats * n + zero_pad_size
x = jax.random.normal(jax.random.PRNGKey(0), (total,))
X = jax.random.normal(jax.random.PRNGKey(1), (total, 3))

for (name, t), tol in ((("well", well), 1e-12), (("stiff", stiff), 1e-6)):
    for nu in (0.5, 1.5, 2.5):
        model = matern_state_space(sigma, ell, nu)
        seq = repeated_shifted_state_space_metric(
            t,
            *model,
            repeats=repeats,
            zero_pad_size=zero_pad_size,
            epsilon=epsilon,
            parallel=False,
        )
        par = repeated_shifted_state_space_metric(
            t,
            *model,
            repeats=repeats,
            zero_pad_size=zero_pad_size,
            epsilon=epsilon,
            parallel=True,
        )
        for callback in ("solve", "norm", "inv_sqrt", "inv_sqrt_transpose"):
            a = getattr(seq, callback)(x)
            b = getattr(par, callback)(x)
            assert bool(jnp.all(jnp.isfinite(b))), (name, nu, callback)
            assert rel(a, b) < tol, (name, nu, callback, rel(a, b))
        assert rel(seq.solve(X), par.solve(X)) < tol, (name, nu, "matrix")


def dense_matern_gram(t, sigma, ell):
    tau = jnp.abs(t[:, None] - t[None, :])
    ft = jnp.sqrt(3.0) * tau / ell
    return sigma**2 * (1.0 + ft) * jnp.exp(-ft)


n, repeats, zero_pad_size = 150, 2, 1
t = jnp.cumsum(
    jax.random.uniform(jax.random.PRNGKey(3), (n,), minval=0.6, maxval=1.4)
)
v = jax.random.normal(jax.random.PRNGKey(4), (repeats * n + zero_pad_size,))


def structured_loss(params):
    sigma, ell, epsilon = params
    metric = repeated_shifted_state_space_metric(
        t,
        *matern_state_space(sigma, ell, 1.5),
        repeats=repeats,
        zero_pad_size=zero_pad_size,
        epsilon=epsilon,
    )
    return v @ metric.solve(v) + metric.norm(v)


def dense_loss(params):
    sigma, ell, epsilon = params
    K = dense_matern_gram(t, sigma, ell)
    blocks = [K + epsilon * jnp.eye(n)] * repeats
    blocks.append(epsilon * jnp.eye(zero_pad_size))
    M = jsp_linalg.block_diag(*blocks)
    return v @ jnp.linalg.solve(M, v) + jnp.sqrt(v @ M @ v)


params = jnp.array([1.3, 0.8, 1e-4])
grad_structured = jax.jit(jax.grad(structured_loss))(params)
grad_dense = jax.grad(dense_loss)(params)
assert rel(grad_dense, grad_structured) < 1e-9

sigma, ell, epsilon = params
metric = repeated_shifted_state_space_metric(
    t,
    *matern_state_space(sigma, ell, 1.5),
    repeats=repeats,
    zero_pad_size=zero_pad_size,
    epsilon=epsilon,
    parallel=False,
)
K = dense_matern_gram(t, sigma, ell)
blocks = [K + epsilon * jnp.eye(n)] * repeats
blocks.append(epsilon * jnp.eye(zero_pad_size))
M = jsp_linalg.block_diag(*blocks)
total = M.shape[0]
x = jax.random.normal(jax.random.PRNGKey(5), (total,))
X = jax.random.normal(jax.random.PRNGKey(6), (total, 3))
assert rel(jnp.linalg.solve(M, x), metric.solve(x)) < 1e-9
assert rel(jnp.linalg.solve(M, X), metric.solve(X)) < 1e-9
assert rel(jnp.sqrt(x @ M @ x), metric.norm(x)) < 1e-10
S = metric.inv_sqrt(jnp.eye(total))
assert rel(jnp.linalg.inv(M), S @ S.T) < 1e-9
assert rel(S.T, metric.inv_sqrt_transpose(jnp.eye(total))) < 1e-10

for callback in ("solve", "norm", "inv_sqrt", "inv_sqrt_transpose"):
    value = x if callback == "norm" else X
    shapes = [
        constant.shape
        for constant in jax.make_jaxpr(getattr(metric, callback))(value).consts
    ]
    assert (n, n) not in shapes, (callback, shapes)
    assert (total, total) not in shapes, (callback, shapes)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_repeated_shifted_dense_metric_float64_matches_explicit_matrix():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from nlls_gram import repeated_shifted_dense_metric


def rel(a, b):
    denominator = jnp.maximum(jnp.linalg.norm(jnp.ravel(a)), 1.0)
    return float(jnp.linalg.norm(jnp.ravel(a - b)) / denominator)


n, repeats, zero_pad_size, epsilon = 4, 3, 2, 0.2
K = jnp.array(
    [
        [2.0, 0.2, 0.1, 0.0],
        [0.2, 1.8, 0.0, 0.1],
        [0.1, 0.0, 1.5, 0.2],
        [0.0, 0.1, 0.2, 1.2],
    ],
    dtype=jnp.float64,
)
metric = repeated_shifted_dense_metric(
    K,
    repeats=repeats,
    zero_pad_size=zero_pad_size,
    epsilon=epsilon,
)
blocks = [K + epsilon * jnp.eye(n)] * repeats
blocks.append(epsilon * jnp.eye(zero_pad_size))
M = jsp_linalg.block_diag(*blocks)
total = M.shape[0]
x = jax.random.normal(jax.random.PRNGKey(0), (total,))
X = jax.random.normal(jax.random.PRNGKey(1), (total, 5))

for callback in ("solve", "inv_sqrt", "inv_sqrt_transpose"):
    for value in (x, X):
        actual = jax.jit(getattr(metric, callback))(value)
        if callback == "solve":
            expected = jnp.linalg.solve(M, value)
            assert rel(expected, actual) < 1e-12
        assert actual.dtype == jnp.float64

assert rel(jnp.sqrt(x @ M @ x), metric.norm(x)) < 1e-12
S = metric.inv_sqrt(jnp.eye(total))
assert rel(jnp.linalg.inv(M), S @ S.T) < 1e-12
assert rel(S.T, metric.inv_sqrt_transpose(jnp.eye(total))) < 1e-12

dx = jax.random.normal(jax.random.PRNGKey(2), (total,))
_, tangent = jax.jvp(metric.norm, (x,), (dx,))
expected_tangent = (x @ M @ dx) / jnp.sqrt(x @ M @ x)
assert rel(expected_tangent, tangent) < 1e-12
assert rel(jax.grad(metric.norm)(x), M @ x / jnp.sqrt(x @ M @ x)) < 1e-12

for callback in ("solve", "norm", "inv_sqrt", "inv_sqrt_transpose"):
    value = x if callback == "norm" else X
    constants = jax.make_jaxpr(getattr(metric, callback))(value).consts
    shapes = [constant.shape for constant in constants]
    assert (n, n) in shapes, (callback, shapes)
    assert (total, total) not in shapes, (callback, shapes)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_repeated_shifted_dense_metric_eps_limit_matches_kkt():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt, repeated_shifted_dense_metric

n, k, m = 12, 2, 4
t = jnp.arange(n, dtype=jnp.float64)
ft = jnp.sqrt(3.0) * jnp.abs(t[:, None] - t[None, :]) / 0.8
K = 1.3**2 * (1.0 + ft) * jnp.exp(-ft) + 1e-8 * jnp.eye(n)
J_alpha = jax.random.normal(jax.random.PRNGKey(0), (m, n))
J_beta = jax.random.normal(jax.random.PRNGKey(1), (m, k))
J = jnp.concatenate([J_alpha, J_beta], axis=1)
b = jax.random.normal(jax.random.PRNGKey(2), (m,))

kkt = jnp.block(
    [
        [2.0 * K, jnp.zeros((n, k)), J_alpha.T],
        [jnp.zeros((k, n)), jnp.zeros((k, k)), J_beta.T],
        [J_alpha, J_beta, jnp.zeros((m, m))],
    ]
)
rhs = jnp.concatenate([jnp.zeros(n + k), b])
theta_kkt = jnp.linalg.solve(kkt, rhs)[: n + k]
assert jnp.allclose(J @ theta_kkt, b, atol=1e-9)


def residual(theta, _, p):
    return J @ theta - p * b


def solved_x(epsilon, p):
    metric = repeated_shifted_dense_metric(
        K, repeats=1, zero_pad_size=k, epsilon=epsilon
    )
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-6,
        metric=metric,
        geodesic_acceleration=False,
        ad_solver="qr",
    )
    return solver.solve(jnp.zeros(n + k), p=p, max_steps=200, atol=1e-12).x


errors = {}
for epsilon in (1e-3, 1e-5, 1e-7):
    theta = solved_x(epsilon, jnp.asarray(1.0))
    assert jnp.allclose(J @ theta, b, atol=1e-8), epsilon
    errors[epsilon] = float(jnp.linalg.norm(theta - theta_kkt))

assert errors[1e-7] < 1e-5, errors
rate = errors[1e-3] / errors[1e-5]
assert 30.0 < rate < 300.0, (errors, rate)

epsilon = 1e-5
p, p_dot = jnp.asarray(1.0), jnp.asarray(1.0)
theta, theta_dot = jax.jvp(lambda q: solved_x(epsilon, q), (p,), (p_dot,))
assert jnp.allclose(theta_dot, theta, atol=1e-8)

x_bar = jnp.linspace(-1.0, 1.0, n + k)
_, pullback = jax.vjp(lambda q: solved_x(epsilon, q), p)
assert jnp.allclose(pullback(x_bar)[0], x_bar @ theta, atol=1e-8)
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

from nlls_gram import LevenbergMarquardt

w = jnp.array([1.0, 2.0, 3.0])
wiggles = 1.0 + 1e-13 * jnp.arange(40.0)


def residual_fn(x, args, p):
    return wiggles * (jnp.dot(w, x) - p["target"])


x0 = jnp.zeros(3)
solver = LevenbergMarquardt(residual_fn, ad_solver="svd")


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
    return LMSolveAction(stop=stop, status=status)


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


def test_float64_recycled_solve_carry_dtypes():
    # Regression: under x64 the deflated-PCG loop counter defaulted to int64
    # while LMState carried RecycleState.iterations as int32, breaking the
    # solve-loop carry on every recycled cg solve.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import (
    LevenbergMarquardt,
    LMStatus,
    RecycleConfig,
    identity_preconditioner,
)

A = jax.random.normal(jax.random.key(0), (20, 6))
b = jax.random.normal(jax.random.key(1), (20,))


def residual(theta):
    return A @ theta - b + 0.1 * jnp.sum(theta**2)


solver = LevenbergMarquardt(
    residual,
    linear_solver="gram_cg",
    iterative_maxiter=30,
    iterative_tol=1e-10,
    dual_preconditioner=identity_preconditioner(),
    ad_solver_preconditioner=identity_preconditioner(),
    recycle=RecycleConfig(rank=3),
)
# gtol is set above the solver's attainable gradient floor for this
# cg/recycle problem (~2e-8 under the locked jaxlib) so CONVERGED is a
# platform-stable outcome; the regression this guards is the dtype carry
# below, not the exact endgame accuracy.
result = solver.solve(jnp.zeros(6), max_steps=60, gtol=1e-7)
assert int(result.status) == LMStatus.CONVERGED, int(result.status)
assert result.lm_state.recycle.iterations.dtype == jnp.int32
assert result.lm_state.recycle.residual_norm.dtype == jnp.float64
assert float(result.info.grad_norm) < 1e-6, float(result.info.grad_norm)
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_float64_default_min_damping_uses_float64_normal_floor():
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
