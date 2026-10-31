"""Benchmarks for Newton + Armijo across wide linear/nonlinear interpolation,
tall least squares, vmap_rows trajectory residuals, and implicit AD.
"""

import jax
import jax.numpy as jnp
import pytest

from benchmarks.conftest import benchmark_batch_size, benchmark_direction_count
from nlls_gram import (
    CG,
    QR,
    Cholesky,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMStatus,
    Newton,
    NewtonStatus,
)


def _temp_mb(lowered):
    analysis = lowered.compile().memory_analysis()
    return None if analysis is None else analysis.temp_size_in_bytes / 1e6


def _make_wide_rbf_problem(device, dtype, n_samples=256, n_centers=2048):
    with jax.default_device(device):
        ts = jnp.linspace(-1.0, 1.0, n_samples, dtype=dtype)
        centers = jnp.linspace(-1.2, 1.2, n_centers, dtype=dtype)
        scaled_distance = (ts[:, None] - centers[None, :]) / jnp.asarray(
            0.12, dtype=dtype
        )
        features = jnp.exp(-0.5 * scaled_distance**2) / jnp.sqrt(
            jnp.asarray(n_centers, dtype=dtype)
        )
        features = features.at[:, :n_samples].add(1.5 * jnp.eye(n_samples, dtype=dtype))
        theta_true = jnp.cos(jnp.linspace(0.0, 6.0, n_centers, dtype=dtype)) * 0.1
        ys = jnp.sin(features @ theta_true)
        x0 = jnp.zeros(n_centers, dtype=dtype)
        args = (features, ys)

        def residual(theta, args, p):
            feats, target = args
            if p is not None:
                target = target + p
            return jnp.sin(feats @ theta) - target

    return residual, x0, args


@pytest.mark.parametrize(
    "solver_name",
    [
        "newton_cholesky",
        "newton_qr",
        "newton_gram_cg",
        "lm_cholesky_no_geo",
        "lm_cholesky_geo",
    ],
)
def test_wide_nonlinear_rbf_solve(benchmark, device, dtype, solver_name):
    residual, x0, args = _make_wide_rbf_problem(device, dtype)
    tol = 1e-4 if dtype == jnp.float32 else 1e-8

    with jax.default_device(device):
        if solver_name == "newton_cholesky":
            solver = Newton(residual, linear_solver=Cholesky(form="gram"))
        elif solver_name == "newton_qr":
            solver = Newton(residual, linear_solver=QR(form="gram"))
        elif solver_name == "newton_gram_cg":
            solver = Newton(
                residual,
                linear_solver=GramCG(IdentityPreconditioner(), tol=tol, maxiter=48),
            )
        elif solver_name == "lm_cholesky_no_geo":
            solver = LevenbergMarquardt(
                residual,
                init_damping=1e-3,
                linear_solver=Cholesky(form="gram"),
                cache_jacobian=False,
                geodesic_acceleration=False,
            )
        else:
            solver = LevenbergMarquardt(
                residual,
                init_damping=1e-3,
                linear_solver=Cholesky(form="gram"),
                cache_jacobian=False,
                geodesic_acceleration=True,
            )

        @jax.jit
        def run_solve(x_init, solver_args):
            return solver.solve(x_init, solver_args, atol=tol, max_steps=30)

        benchmark.extra_info["temp_mb"] = _temp_mb(run_solve.lower(x0, args))
        warm = run_solve(x0, args)
        jax.block_until_ready(warm)
        expected_status = (
            NewtonStatus.CONVERGED
            if solver_name.startswith("newton")
            else LMStatus.CONVERGED
        )
        assert int(warm.status) == int(expected_status)
        benchmark.extra_info["steps"] = int(warm.steps)
        benchmark.extra_info["final_loss"] = float(warm.info.loss)

        def run():
            out = run_solve(x0, args)
            jax.block_until_ready(out)
            return out

        benchmark.group = f"wide-rbf-solve-{dtype.__name__}"
        benchmark(run)


@pytest.mark.parametrize(
    "solver_name",
    ["newton_cholesky", "newton_qr", "newton_cg", "lm_cholesky_no_geo"],
)
def test_tall_nonlinear_least_squares_solve(benchmark, device, dtype, solver_name):
    m, n = 512, 64
    tol = 1e-4 if dtype == jnp.float32 else 1e-8
    with jax.default_device(device):
        k1, k2 = jax.random.split(jax.random.key(11))
        A = jax.random.normal(k1, (m, n), dtype=dtype) / jnp.sqrt(
            jnp.asarray(m, dtype=dtype)
        )
        x_true = jax.random.normal(k2, (n,), dtype=dtype) * 0.2
        b = jnp.sinh(A @ x_true)
        x0 = jnp.zeros(n, dtype=dtype)

        def residual(x):
            return jnp.sinh(A @ x) - b

        if solver_name == "newton_cholesky":
            solver = Newton(residual, linear_solver=Cholesky(form="normal"))
        elif solver_name == "newton_qr":
            solver = Newton(residual, linear_solver=QR(form="normal"))
        elif solver_name == "newton_cg":
            solver = Newton(
                residual,
                linear_solver=CG(IdentityPreconditioner(), tol=tol, maxiter=48),
            )
        else:
            solver = LevenbergMarquardt(
                residual,
                init_damping=1e-3,
                linear_solver=Cholesky(form="normal"),
                cache_jacobian=False,
                geodesic_acceleration=False,
            )

        @jax.jit
        def run_solve(x_init):
            return solver.solve(x_init, atol=tol, max_steps=25)

        warm = run_solve(x0)
        jax.block_until_ready(warm)
        benchmark.extra_info["steps"] = int(warm.steps)
        benchmark.extra_info["final_loss"] = float(warm.info.loss)

        def run():
            out = run_solve(x0)
            jax.block_until_ready(out)
            return out

        benchmark.group = f"tall-nlls-solve-{dtype.__name__}"
        benchmark(run)


@pytest.mark.parametrize("transform", ["primal", "vmap_primal", "jvp", "vjp"])
@pytest.mark.parametrize("solver_kind", ["newton", "lm_no_geo"])
def test_wide_rbf_solve_and_ad(benchmark, device, dtype, solver_kind, transform):
    n_samples, n_centers = 128, 1024
    batch = benchmark_batch_size(device)
    k_dir = benchmark_direction_count(device)
    tol = 1e-4 if dtype == jnp.float32 else 1e-8

    residual, x0, args = _make_wide_rbf_problem(
        device, dtype, n_samples=n_samples, n_centers=n_centers
    )
    with jax.default_device(device):
        k1, k2, k3 = jax.random.split(jax.random.key(21), 3)
        proj_in = jax.random.normal(k1, (n_samples, k_dir), dtype=dtype) / jnp.sqrt(
            jnp.asarray(k_dir, dtype=dtype)
        )
        proj_out = jax.random.normal(k2, (k_dir, n_centers), dtype=dtype) / jnp.sqrt(
            jnp.asarray(n_centers, dtype=dtype)
        )
        p = jax.random.normal(k3, (k_dir,), dtype=dtype) * 0.02
        p_batch = (
            p[None, :] + jax.random.normal(k3, (batch, k_dir), dtype=dtype) * 0.005
        )

        def param_residual(theta, solver_args, param):
            return residual(theta, solver_args, proj_in @ param)

        if solver_kind == "newton":
            solver = Newton(
                param_residual,
                linear_solver=Cholesky(form="gram"),
                ad_solver=Cholesky(form="gram"),
            )
        else:
            solver = LevenbergMarquardt(
                param_residual,
                init_damping=1e-3,
                linear_solver=Cholesky(form="gram"),
                ad_solver=Cholesky(form="gram"),
                cache_jacobian=False,
                geodesic_acceleration=False,
            )

        def solve_obs(param):
            sol = solver.solve(x0, args, p=param, max_steps=15, atol=tol)
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
        else:
            fn, arg = (
                jax.jit(jax.grad(lambda param: jnp.sum(solve_obs(param) ** 2))),
                p,
            )

        jax.block_until_ready(fn(arg))

        def run():
            out = fn(arg)
            jax.block_until_ready(out)
            return out

        benchmark.group = f"wide-rbf-ad-{transform}-{dtype.__name__}"
        benchmark(run)
