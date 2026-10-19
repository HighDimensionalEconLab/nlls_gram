"""``vmap_rows`` against ``jax.vmap`` on a simulated-path residual: the same
values, Jacobian, tangents, and second directional derivative; the same
solver steps; and composition with ``jit`` and an outer ``vmap``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from nlls_gram import (
    IdentityMetric,
    LevenbergMarquardt,
    LMStatus,
    RidgeLevenbergMarquardt,
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
    # Row targets consistent with the zero policy, so an exact root exists.
    x0, shocks, next_shocks, _ = args
    rho = 0.9
    xs = jax.lax.scan(lambda x, s: ((rho * x + s),) * 2, x0, shocks)[1]
    targets = rho * jnp.mean(xs, axis=1) + jnp.mean(next_shocks)
    p = {"rho": rho, "beta": 0.95, "targets": targets}
    return theta, args, p


def assemble(residual, theta, args, p):
    # The solver's reverse-mode assembly: m unit cotangents through the
    # transposed linearization.
    flat, unravel = ravel_pytree(theta)

    def residual_flat(th):
        return residual(unravel(th), args, p)

    resid, jvp = jax.linearize(residual_flat, flat)
    transpose = jax.linear_transpose(jvp, flat)
    J = jax.vmap(lambda c: transpose(c)[0])(jnp.eye(resid.size, dtype=resid.dtype))
    v = jnp.sin(jnp.arange(flat.size, dtype=flat.dtype))
    f_vv = jax.jvp(lambda th: jax.jvp(residual_flat, (th,), (v,))[1], (flat,), (v,))[1]
    return resid, J, jvp(v), f_vv


def test_matches_vmap_to_second_order():
    theta, args, p = problem()
    reference = jax.jit(lambda th: assemble(native, th, args, p))(theta)
    candidate = jax.jit(lambda th: assemble(rows, th, args, p))(theta)
    assert reference[1].shape[0] < ravel_pytree(theta)[0].size
    tol = 1e3 * float(jnp.finfo(reference[0].dtype).eps)
    for a, b in zip(reference, candidate, strict=True):
        np.testing.assert_allclose(b, a, rtol=tol, atol=tol * float(jnp.abs(a).max()))


def test_matches_vmap_in_forward_assembly():
    # Tall system: the solver assembles J from n unit tangents instead.
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


def test_solver_takes_the_same_steps():
    theta, args, p = problem()
    results = []
    for residual in (native, rows):
        solver = LevenbergMarquardt(residual, geodesic_acceleration=True)
        results.append(solver.solve(theta, args, p=p, max_steps=100, atol=1e-4))
    reference, candidate = results
    assert candidate.status == LMStatus.CONVERGED
    assert int(candidate.steps) == int(reference.steps)
    np.testing.assert_allclose(
        ravel_pytree(candidate.x)[0], ravel_pytree(reference.x)[0], rtol=1e-3
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
    # The cached Jacobian after one step is the assembled matrix itself.
    theta, args, p = problem()
    for solver_cls, kwargs in (
        (LevenbergMarquardt, {}),
        (
            RidgeLevenbergMarquardt,
            {"metric": IdentityMetric(ravel_pytree(theta)[0].size), "ridge": 1e-4},
        ),
    ):
        jacobians = []
        for chunk in (None, 4):
            solver = solver_cls(rows, jacobian_chunk=chunk, **kwargs)
            lm_state = solver.init(theta, args, p=p)
            jacobians.append(solver.update(theta, lm_state, args, p)[1].Jt)
        batch, chunked = jacobians
        tol = 1e3 * float(jnp.finfo(batch.dtype).eps)
        np.testing.assert_allclose(
            chunked, batch, rtol=tol, atol=tol * float(jnp.abs(batch).max())
        )
