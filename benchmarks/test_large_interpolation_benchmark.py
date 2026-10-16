import jax
import jax.numpy as jnp
import pytest

from benchmarks.conftest import benchmark_batch_size, benchmark_direction_count
from nlls_gram import (
    QR,
    Cholesky,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMStatus,
)

ITERATIVE_MAXITER = 8


def _make_large_interpolation_problem(
    *, device, dtype, linear_solver, geodesic_acceleration
):
    n_samples = 1024
    n_centers = 8192

    with jax.default_device(device):
        ts = jnp.linspace(-1.0, 1.0, n_samples, dtype=dtype)
        centers = jnp.linspace(-1.2, 1.2, n_centers, dtype=dtype)
        scaled_distance = (ts[:, None] - centers[None, :]) / jnp.asarray(
            0.08, dtype=dtype
        )
        features = jnp.exp(-0.5 * scaled_distance**2) / jnp.sqrt(
            jnp.asarray(n_centers, dtype=dtype)
        )
        theta_true = jnp.cos(jnp.linspace(0.0, 12.0, n_centers, dtype=dtype))
        ys = jnp.sin(features @ theta_true)
        x = jnp.asarray(0.05, dtype=dtype) * jnp.sin(
            jnp.linspace(0.0, 8.0, n_centers, dtype=dtype)
        )
        args = (features, ys)

        def residual(theta, args, p):
            features, ys = args
            return jnp.sin(features @ theta) - ys

        solver_kwargs = {"init_damping": 1e-2, "linear_solver": linear_solver}

        base_solver = LevenbergMarquardt(residual, **solver_kwargs)
        solver = LevenbergMarquardt(
            residual,
            **solver_kwargs,
            geodesic_acceleration=geodesic_acceleration,
        )

        lm_state = base_solver.init(x, args)

        @jax.jit
        def first_step(x, lm_state):
            return base_solver.update(x, lm_state, args)

        x, lm_state, _ = first_step(x, lm_state)
        jax.block_until_ready((x, lm_state))

        @jax.jit
        def step(x, lm_state):
            return solver.update(x, lm_state, args)

    return x, lm_state, step


@pytest.mark.parametrize(
    "linear_solver",
    [
        Cholesky(form="gram"),
        QR(),
        GramCG(IdentityPreconditioner(), maxiter=ITERATIVE_MAXITER),
    ],
    ids=["gram_cholesky", "qr", "gram_cg"],
)
@pytest.mark.parametrize("geodesic_acceleration", [False, True])
def test_large_rbf_interpolation_second_update(
    benchmark, device, dtype, linear_solver, geodesic_acceleration
):
    x, lm_state, step = _make_large_interpolation_problem(
        device=device,
        dtype=dtype,
        linear_solver=linear_solver,
        geodesic_acceleration=geodesic_acceleration,
    )

    warmup = step(x, lm_state)
    jax.block_until_ready(warmup)

    def run():
        out = step(x, lm_state)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize(
    "solver_kind",
    ["gram_cholesky", "gram_cg"],
)
@pytest.mark.parametrize(
    "transform",
    ["primal", "vmap_primal", "jvp", "vmap_jvp", "vjp", "vmap_vjp"],
)
def test_large_rbf_interpolation_solve_and_ad(
    benchmark, device, dtype, solver_kind, transform
):
    n_samples = 256
    n_centers = 2048
    batch = benchmark_batch_size(device)
    k_dir = benchmark_direction_count(device)
    tol = 1e-3 if dtype == jnp.float32 else 1e-7

    with jax.default_device(device):
        ts = jnp.linspace(-1.0, 1.0, n_samples, dtype=dtype)
        centers = jnp.linspace(-1.2, 1.2, n_centers, dtype=dtype)
        scaled_distance = (ts[:, None] - centers[None, :]) / jnp.asarray(
            0.12, dtype=dtype
        )
        features = jnp.exp(-0.5 * scaled_distance**2) / jnp.sqrt(
            jnp.asarray(n_centers, dtype=dtype)
        )
        features = features.at[:, :n_samples].add(2.0 * jnp.eye(n_samples, dtype=dtype))
        k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
        proj_in = jax.random.normal(k1, (n_samples, k_dir), dtype=dtype) / jnp.sqrt(
            jnp.asarray(k_dir, dtype=dtype)
        )
        proj_out = jax.random.normal(k2, (k_dir, n_centers), dtype=dtype) / jnp.sqrt(
            jnp.asarray(n_centers, dtype=dtype)
        )
        theta_true = jnp.cos(jnp.linspace(0.0, 6.0, n_centers, dtype=dtype)) * 0.1
        p = jax.random.normal(k3, (k_dir,), dtype=dtype) * 0.02
        p_batch = (
            p[None, :] + jax.random.normal(k3, (batch, k_dir), dtype=dtype) * 0.005
        )
        ys_base = jnp.sin(features @ theta_true)

        def residual(theta, _, param):
            return jnp.sin(features @ theta) - (ys_base + proj_in @ param)

        warmup_solver = LevenbergMarquardt(
            residual,
            init_damping=1e-2,
            linear_solver=Cholesky(form="gram"),
            cache_jacobian=False,
            geodesic_acceleration=False,
        )
        x_star = warmup_solver.solve(theta_true, p=p, max_steps=24, atol=tol).x
        jax.block_until_ready(x_star)

        if solver_kind == "gram_cholesky":
            linear_solver = Cholesky(form="gram")
            ad_solver = Cholesky()
        else:
            linear_solver = GramCG(IdentityPreconditioner(), tol=tol, maxiter=48)
            ad_solver = GramCG(IdentityPreconditioner(), tol=tol, maxiter=64)

        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-3,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
            cache_jacobian=False,
            geodesic_acceleration=False,
        )
        assert bool(
            solver.solve(x_star, p=p, max_steps=8, atol=tol).status
            == LMStatus.CONVERGED
        )

        def solve_obs(param):
            sol = solver.solve(x_star, p=param, max_steps=8, atol=tol)
            return proj_out @ sol.x

        if transform == "primal":
            fn, arg = jax.jit(solve_obs), p
        elif transform == "vmap_primal":
            fn, arg = jax.jit(jax.vmap(solve_obs)), p_batch
        elif transform == "jvp":
            tangent = jnp.ones_like(p)
            fn, arg = (
                jax.jit(lambda param: jax.jvp(solve_obs, (param,), (tangent,))),
                p,
            )
        elif transform == "vmap_jvp":
            fn, arg = jax.jit(jax.jacfwd(solve_obs)), p
        elif transform == "vjp":
            fn, arg = (
                jax.jit(jax.grad(lambda param: jnp.sum(solve_obs(param) ** 2))),
                p,
            )
        else:
            fn, arg = jax.jit(jax.jacrev(solve_obs)), p

        jax.block_until_ready(fn(arg))

        def run():
            out = fn(arg)
            jax.block_until_ready(out)
            return out

        rounds = 8 if device.platform in ("gpu", "metal") else 10
        iterations = 2 if device.platform in ("gpu", "metal") else 3
        benchmark.pedantic(run, rounds=rounds, iterations=iterations)
