import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.flatten_util import ravel_pytree

from nlls_gram import (
    IdentityRidge,
    LevenbergMarquardt,
    SolverStatus,
    vmap_rows,
)

N = 8
WIDTH = 16
T = 8
Q = 3


def policy(params, x):
    h = jnp.tanh(x[:, None] * params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    return jnp.mean(h, axis=0) @ params["w3"] + params["b3"]


def simulate(x0, shocks, params, p):
    def step(x, shock):
        x_next = p["rho"] * x + policy(params, x) + shock
        return x_next, x_next

    return jax.lax.scan(step, x0, shocks)[1]


def euler_row(x, t, next_shocks, params, p):
    u = policy(params, x)
    x_next = p["rho"] * x[None, :] + u + next_shocks
    u_next = jax.vmap(policy, in_axes=(None, 0))(params, x_next)
    return u - p["beta"] * jnp.mean(u_next), jnp.mean(x_next) - p["targets"][t]


def make_residual(block):
    def residual(theta, args, p):
        x0, shocks, next_shocks, t_idx = args
        xs = simulate(x0, shocks, theta, p)
        euler, mean = block(xs, t_idx, next_shocks, theta, p)
        return jnp.concatenate([euler, mean])

    return residual


IN_AXES = (0, 0, None, None, None)
native = make_residual(jax.vmap(euler_row, in_axes=IN_AXES))
rows = make_residual(vmap_rows(in_axes=IN_AXES)(euler_row))


def problem(seed=0, T=T):
    keys = jax.random.split(jax.random.key(seed), 8)
    scale = 0.1
    theta = {
        "w1": scale * jax.random.normal(keys[0], (WIDTH,)),
        "b1": scale * jax.random.normal(keys[1], (WIDTH,)),
        "w2": scale * jax.random.normal(keys[2], (WIDTH, WIDTH)),
        "b2": scale * jax.random.normal(keys[3], (WIDTH,)),
        "w3": scale * jax.random.normal(keys[4], (WIDTH,)),
        "b3": jnp.asarray(0.0),
    }
    args = (
        1.0 + 0.1 * jax.random.normal(keys[5], (N,)),
        0.05 * jax.random.normal(keys[6], (T, N)),
        0.05 * jax.random.normal(keys[7], (Q, N)),
        jnp.arange(T),
    )
    x0, shocks, next_shocks, _ = args
    rho = 0.9
    xs = jax.lax.scan(lambda x, s: ((rho * x + s),) * 2, x0, shocks)[1]
    targets = rho * jnp.mean(xs, axis=1) + jnp.mean(next_shocks)
    p = {"rho": rho, "beta": 0.95, "targets": targets}
    return theta, args, p


def assemble(residual, theta, args, p):
    flat, unravel = ravel_pytree(theta)

    def residual_flat(th):
        return residual(unravel(th), args, p)

    resid, jvp = jax.linearize(residual_flat, flat)
    transpose = jax.linear_transpose(jvp, flat)
    J = jax.vmap(lambda c: transpose(c)[0])(jnp.eye(resid.size, dtype=resid.dtype))
    v = jnp.sin(jnp.arange(flat.size, dtype=flat.dtype))
    f_vv = jax.jvp(lambda th: jax.jvp(residual_flat, (th,), (v,))[1], (flat,), (v,))[1]
    return resid, J, jvp(v), f_vv


@pytest.mark.parametrize("chunk", [None, 1, 4])
def test_matches_vmap_to_second_order(chunk):
    theta, args, p = problem()
    reference = jax.jit(lambda th: assemble(native, th, args, p))(theta)
    fn = (
        rows
        if chunk is None
        else make_residual(vmap_rows(in_axes=IN_AXES, chunk_size=chunk)(euler_row))
    )
    candidate = jax.jit(lambda th: assemble(fn, th, args, p))(theta)
    assert reference[1].shape[0] < ravel_pytree(theta)[0].size
    tol = 1e3 * float(jnp.finfo(reference[0].dtype).eps)
    for a, b in zip(reference, candidate, strict=True):
        np.testing.assert_allclose(b, a, rtol=tol, atol=tol * float(jnp.abs(a).max()))


def test_matches_vmap_in_forward_assembly():
    theta, args, p = problem(T=200)
    flat, unravel = ravel_pytree(theta)

    def forward(residual):
        resid, jvp = jax.linearize(lambda th: residual(unravel(th), args, p), flat)
        return jax.vmap(jvp)(jnp.eye(flat.size, dtype=flat.dtype)).T

    reference = jax.jit(lambda: forward(native))()
    candidate = jax.jit(lambda: forward(rows))()
    assert reference.shape[0] > flat.size
    tol = 1e3 * float(jnp.finfo(reference.dtype).eps)
    np.testing.assert_allclose(
        candidate, reference, rtol=tol, atol=tol * float(jnp.abs(reference).max())
    )


def test_solver_takes_the_same_steps_and_composes_with_checkpoint():
    theta, args, p = problem()
    ckpt_rows = make_residual(vmap_rows(jax.checkpoint(euler_row), in_axes=IN_AXES))
    results = []
    for residual in (native, rows, ckpt_rows):
        solver = LevenbergMarquardt(residual, geodesic_acceleration=True)
        results.append(solver.solve(theta, args, p=p, max_steps=100, atol=1e-4))
    reference, candidate, ckpt_candidate = results
    assert candidate.status == SolverStatus.CONVERGED
    assert ckpt_candidate.status == SolverStatus.CONVERGED
    assert int(candidate.steps) == int(reference.steps)
    assert int(ckpt_candidate.steps) == int(reference.steps)
    np.testing.assert_allclose(
        ravel_pytree(candidate.x)[0], ravel_pytree(reference.x)[0], rtol=1e-3
    )
    np.testing.assert_allclose(
        ravel_pytree(ckpt_candidate.x)[0], ravel_pytree(reference.x)[0], rtol=1e-3
    )


def test_composes_with_jit_and_outer_vmap():
    theta, args, p = problem()
    batched_args = jax.tree.map(lambda a: jnp.stack([a, 2.0 * a]), args)
    batched_args = batched_args[:3] + (jnp.stack([args[3], args[3]]),)

    def jacobian(residual, args_b):
        return assemble(residual, theta, args_b, p)[1]

    reference = jax.jit(jax.vmap(lambda a: jacobian(native, a)))(batched_args)
    candidate = jax.jit(jax.vmap(lambda a: jacobian(rows, a)))(batched_args)
    tol = 1e3 * float(jnp.finfo(reference.dtype).eps)
    np.testing.assert_allclose(
        candidate, reference, rtol=tol, atol=tol * float(jnp.abs(reference).max())
    )


def test_jacobian_chunk_matches_one_batch():
    theta, args, p = problem()
    for kwargs in (
        {},
        {"ridge_metric": IdentityRidge(ravel_pytree(theta)[0].size), "ridge": 1e-4},
    ):
        jacobians = []
        for chunk in (None, 4):
            solver = LevenbergMarquardt(rows, jacobian_chunk=chunk, **kwargs)
            state = solver.init(theta, args, p=p)
            jacobians.append(solver.update(theta, state, args, p)[1].Jt)
        batch, chunked = jacobians
        tol = 1e3 * float(jnp.finfo(batch.dtype).eps)
        np.testing.assert_allclose(
            chunked, batch, rtol=tol, atol=tol * float(jnp.abs(batch).max())
        )


def test_default_in_axes():
    theta, args, p = problem()

    @vmap_rows
    def row_default(x, next_shocks, params, p):
        return euler_row(x, 0, next_shocks, params, p)[0]

    def residual_default(th):
        x0, shocks, next_shocks, _ = args
        xs = simulate(x0, shocks, th, p)
        return row_default(xs, next_shocks, th, p)

    def residual_vmap(th):
        x0, shocks, next_shocks, _ = args
        xs = simulate(x0, shocks, th, p)
        return jax.vmap(
            lambda x, ns, pr, p_: euler_row(x, 0, ns, pr, p_)[0],
            in_axes=(0, None, None, None),
        )(xs, next_shocks, th, p)

    J_default = jax.jit(jax.jacrev(residual_default))(theta)
    J_vmap = jax.jit(jax.jacrev(residual_vmap))(theta)
    tol = 1e3 * float(jnp.finfo(jax.tree.leaves(J_vmap)[0].dtype).eps)
    for a, b in zip(jax.tree.leaves(J_vmap), jax.tree.leaves(J_default), strict=True):
        np.testing.assert_allclose(b, a, rtol=tol, atol=tol * float(jnp.abs(a).max()))


def test_one_arg_signature_nonfloating_aux_transpose_and_multishape_jit():
    grid = jnp.linspace(0.5, 2.0, 5, dtype=jnp.float32)

    @vmap_rows(in_axes=(None,))
    def bad_static(_x):
        return jnp.float32(0.0)

    with pytest.raises(ValueError, match="at least one"):
        bad_static(grid)

    def one_arg_row(g, x):
        return g * x[0] + x[1] - 1.0

    wrapped_1arg = vmap_rows(in_axes=(0, None))(one_arg_row)
    res_1arg = LevenbergMarquardt(lambda x: wrapped_1arg(grid, x)).solve(
        jnp.zeros(2, jnp.float32), atol=1e-5, max_steps=15
    )
    assert int(res_1arg.status) == int(SolverStatus.CONVERGED)

    @vmap_rows(in_axes=(0, None))
    def row_with_meta(g, x):
        r = jnp.sin(g * x[0]) - x[1]
        return r, {
            "idx": jnp.int32(3),
            "pos": r > 0,
            "const": jnp.ones(3, dtype=jnp.float32),
        }

    x0 = jnp.array([0.4, 0.1], dtype=jnp.float32)
    (r_val, meta_val), lin_fn = jax.linearize(lambda x: row_with_meta(grid, x), x0)
    assert meta_val["idx"].dtype == jnp.int32 and meta_val["pos"].dtype == jnp.bool_
    dr, dmeta = lin_fn(jnp.ones_like(x0))
    assert dmeta["idx"].dtype == jax.dtypes.float0
    assert dmeta["const"].shape == (5, 3) and jnp.all(dmeta["const"] == 0.0)
    (xt,) = jax.linear_transpose(lambda v: lin_fn(v)[0], x0)(jnp.ones_like(r_val))
    J_rev = jax.jacrev(lambda x: row_with_meta(grid, x)[0])(x0)
    np.testing.assert_allclose(xt, jnp.sum(J_rev, axis=0), rtol=1e-5, atol=1e-6)

    @jax.jit
    def multi_shape_jac(g, x):
        return jax.jacrev(lambda th: wrapped_1arg(g, th))(x)

    j5 = multi_shape_jac(grid, x0)
    j9 = multi_shape_jac(jnp.linspace(0.5, 2.0, 9, dtype=jnp.float32), x0)
    assert j5.shape == (5, 2) and j9.shape == (9, 2)


class RowMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.scale = nnx.Variable(jnp.asarray(2.0, dtype=jnp.float32))
        self.hidden = nnx.Linear(1, 4, rngs=rngs)
        self.head = nnx.Linear(4, 1, rngs=rngs)

    def __call__(self, x):
        return self.scale[...] * self.head(nnx.tanh(self.hidden(x[None])))[0]


def test_custom_pytree_module_argument():
    xs = jnp.linspace(-1.0, 1.0, 7, dtype=jnp.float32)
    ys = jnp.sin(xs)
    graphdef, theta0, nondiff = nnx.split(RowMLP(nnx.Rngs(0)), nnx.Param, ...)

    def row_fn(xy, mod):
        x, y = xy
        return mod(x) - y

    xy = jnp.stack([xs, ys], axis=1)
    rows_fn = vmap_rows(row_fn, in_axes=(0, None), chunk_size=3)
    vmap_fn = jax.vmap(row_fn, in_axes=(0, None))

    def res_rows(th, data, _p):
        return rows_fn(data, nnx.merge(graphdef, th, nondiff))

    def res_vmap(th, data, _p):
        return vmap_fn(data, nnx.merge(graphdef, th, nondiff))

    ref = jax.jit(lambda th: assemble(res_vmap, th, xy, None))(theta0)
    cand = jax.jit(lambda th: assemble(res_rows, th, xy, None))(theta0)
    tol = 1e3 * float(jnp.finfo(jnp.float32).eps)
    for a, b in zip(ref, cand, strict=True):
        np.testing.assert_allclose(b, a, rtol=tol, atol=tol)

    batched_xy = jnp.stack([xy, 0.5 * xy])
    j_ref = jax.jit(jax.vmap(lambda d: assemble(res_vmap, theta0, d, None)[1]))(
        batched_xy
    )
    j_cand = jax.jit(jax.vmap(lambda d: assemble(res_rows, theta0, d, None)[1]))(
        batched_xy
    )
    np.testing.assert_allclose(j_cand, j_ref, rtol=tol, atol=tol)

    sol_ref = LevenbergMarquardt(res_vmap, geodesic_acceleration=True).solve(
        theta0, xy, max_steps=30, atol=1e-4
    )
    sol_cand = LevenbergMarquardt(res_rows, geodesic_acceleration=True).solve(
        theta0, xy, max_steps=30, atol=1e-4
    )
    assert int(sol_cand.steps) == int(sol_ref.steps)
    np.testing.assert_allclose(
        ravel_pytree(sol_cand.x)[0], ravel_pytree(sol_ref.x)[0], rtol=1e-4, atol=1e-5
    )
