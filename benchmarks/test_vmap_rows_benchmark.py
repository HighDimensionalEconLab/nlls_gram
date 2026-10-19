"""Reverse-mode Jacobian assembly and one LM step for a simulated-path
residual, with the per-state block under ``jax.vmap`` and under
``vmap_rows``. ``extra_info["temp_mb"]`` records the compiled program's
temporary memory, the quantity ``vmap_rows`` exists to shrink.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from nlls_gram import LevenbergMarquardt, vmap_rows

WIDTH = 16
T = 64
Q = 3
IN_AXES = (0, None, None, None)


def policy(params, x):
    h = jnp.tanh(x[:, None] * params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    return jnp.mean(h, axis=0) @ params["w3"] + params["b3"]


def simulate(x0, shocks, params, p):
    def step(x, shock):
        x_next = p["rho"] * x + policy(params, x) + shock
        return x_next, x_next

    return jax.lax.scan(step, x0, shocks)[1]


def euler_row(x, next_shocks, params, p):
    u = policy(params, x)
    x_next = p["rho"] * x[None, :] + u + next_shocks
    u_next = jax.vmap(policy, in_axes=(None, 0))(params, x_next)
    return u - p["beta"] * jnp.mean(u_next)


BLOCKS = {
    "vmap": jax.vmap(euler_row, in_axes=IN_AXES),
    "vmap_rows": vmap_rows(euler_row, in_axes=IN_AXES),
}


def make_residual(block):
    def residual(theta, args, p):
        x0, shocks, next_shocks = args
        xs = simulate(x0, shocks, theta, p)
        r = block(xs, next_shocks, theta, p)
        return r / jnp.sqrt(r.size)

    return residual


def problem(device, dtype, n_firms):
    keys = jax.random.split(jax.random.key(0), 8)
    with jax.default_device(device):
        scale = jnp.asarray(0.1, dtype)
        theta = {
            "w1": scale * jax.random.normal(keys[0], (WIDTH,), dtype),
            "b1": scale * jax.random.normal(keys[1], (WIDTH,), dtype),
            "w2": scale * jax.random.normal(keys[2], (WIDTH, WIDTH), dtype),
            "b2": scale * jax.random.normal(keys[3], (WIDTH,), dtype),
            "w3": scale * jax.random.normal(keys[4], (WIDTH,), dtype),
            "b3": jnp.asarray(0.0, dtype),
        }
        args = (
            1.0 + 0.1 * jax.random.normal(keys[5], (n_firms,), dtype),
            0.05 * jax.random.normal(keys[6], (T, n_firms), dtype),
            0.05 * jax.random.normal(keys[7], (Q, n_firms), dtype),
        )
        p = {"rho": jnp.asarray(0.9, dtype), "beta": jnp.asarray(0.95, dtype)}
    return theta, args, p


def _temp_mb(lowered):
    analysis = lowered.compile().memory_analysis()
    return None if analysis is None else analysis.temp_size_in_bytes / 1e6


@pytest.mark.parametrize("n_firms", [128, 1024])
@pytest.mark.parametrize("block", ["vmap", "vmap_rows"])
def test_jacobian_assembly(benchmark, device, dtype, n_firms, block):
    theta, args, p = problem(device, dtype, n_firms)
    residual = make_residual(BLOCKS[block])
    flat, unravel = ravel_pytree(theta)

    # The solver's reverse-mode assembly: m unit cotangents through the
    # transposed linearization.
    @jax.jit
    def assemble(th):
        resid, jvp = jax.linearize(lambda t: residual(unravel(t), args, p), th)
        transpose = jax.linear_transpose(jvp, th)
        basis = jnp.eye(resid.size, dtype=resid.dtype)
        return resid, jax.vmap(lambda c: transpose(c)[0])(basis)

    with jax.default_device(device):
        benchmark.extra_info["temp_mb"] = _temp_mb(assemble.lower(flat))
        jax.block_until_ready(assemble(flat))

    benchmark.group = f"vmap-rows-jacobian-{n_firms}"

    def run():
        out = assemble(flat)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("n_firms", [128, 1024])
@pytest.mark.parametrize(
    ("block", "jacobian_chunk"),
    [("vmap", None), ("vmap_rows", None), ("vmap_rows", 16)],
    ids=["vmap", "vmap_rows", "vmap_rows-chunk16"],
)
def test_update_step(benchmark, device, dtype, n_firms, block, jacobian_chunk):
    theta, args, p = problem(device, dtype, n_firms)
    solver = LevenbergMarquardt(
        make_residual(BLOCKS[block]),
        geodesic_acceleration=True,
        cache_jacobian=False,
        jacobian_chunk=jacobian_chunk,
    )
    with jax.default_device(device):
        lm_state = solver.init(theta, args, p=p)

        @jax.jit
        def update(theta, lm_state):
            return solver.update(theta, lm_state, args, p)

        benchmark.extra_info["temp_mb"] = _temp_mb(update.lower(theta, lm_state))
        jax.block_until_ready(update(theta, lm_state))

    benchmark.group = f"vmap-rows-update-{n_firms}"

    def run():
        out = update(theta, lm_state)
        jax.block_until_ready(out)
        return out

    benchmark(run)
