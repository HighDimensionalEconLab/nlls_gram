import jax
import jax.numpy as jnp
import pytest

from benchmarks.conftest import benchmark_batch_size, benchmark_direction_count
from nlls_gram import (
    CG,
    LU,
    SVD,
    Cholesky,
    GramCG,
    IdentityMetric,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMStatus,
    RidgeLevenbergMarquardt,
)

SOLVER_CASES = (
    "lu",
    "cholesky_gram",
    "svd",
    "gram_cg",
    "ridge_cholesky",
    "ridge_cg",
)
TRANSFORMS = (
    "primal",
    "vmap_primal",
    "jvp",
    "vmap_jvp",
    "vjp",
    "vmap_vjp",
    "vmap_grad",
)


def _make_large_ad_problem(solver_case, *, device=None, dtype=jnp.float32):
    m = 256
    n = 256 if solver_case == "lu" else 1024
    batch = benchmark_batch_size(device)
    k_dir = benchmark_direction_count(device)
    k_a, k_b, k_c, k_p = jax.random.split(jax.random.key(42), 4)
    raw = jax.random.normal(k_a, (m, n), dtype=dtype) / jnp.sqrt(
        jnp.asarray(n, dtype=dtype)
    )
    if solver_case == "lu":
        design = raw + 2.0 * jnp.eye(m, dtype=dtype)
    else:
        design = raw.at[:, :m].add(2.0 * jnp.eye(m, dtype=dtype))
    proj_in = jax.random.normal(k_b, (m, k_dir), dtype=dtype) / jnp.sqrt(
        jnp.asarray(k_dir, dtype=dtype)
    )
    proj_out = jax.random.normal(k_c, (k_dir, n), dtype=dtype) / jnp.sqrt(
        jnp.asarray(n, dtype=dtype)
    )
    p = jax.random.normal(k_p, (k_dir,), dtype=dtype) * 0.2
    p_batch = p[None, :] + jax.random.normal(k_p, (batch, k_dir), dtype=dtype) * 0.02
    tol = 1e-5 if dtype == jnp.float32 else 1e-8

    def residual(x, _, param):
        return design @ (x + 0.05 * x**2) - proj_in @ param

    warmup_solver = LevenbergMarquardt(
        residual,
        linear_solver=Cholesky(),
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x_star = warmup_solver.solve(
        jnp.zeros(n, dtype=dtype), p=p, max_steps=32, atol=tol
    ).x
    jax.block_until_ready(x_star)

    if solver_case in ("ridge_cholesky", "ridge_cg"):
        ad_solver = (
            Cholesky()
            if solver_case == "ridge_cholesky"
            else CG(IdentityPreconditioner(), tol=tol, maxiter=64)
        )
        linear_solver = (
            Cholesky()
            if solver_case == "ridge_cholesky"
            else CG(IdentityPreconditioner(), tol=tol, maxiter=64)
        )
        solver = RidgeLevenbergMarquardt(
            residual,
            metric=IdentityMetric(n),
            ridge=1e-2,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
            cache_jacobian=False,
            geodesic_acceleration=False,
        )
        x_star = solver.solve(x_star, p=p, max_steps=32, gtol=tol).x
        jax.block_until_ready(x_star)

        def solve_obs(param):
            sol = solver.solve(x_star, p=param, max_steps=5, gtol=tol)
            return proj_out @ sol.x

        def status(param):
            return solver.solve(x_star, p=param, max_steps=5, gtol=tol).status

    else:
        ad_solver = {
            "lu": LU(),
            "cholesky_gram": Cholesky(),
            "svd": SVD(),
            "gram_cg": GramCG(IdentityPreconditioner(), tol=tol, maxiter=64),
        }[solver_case]
        linear_solver = (
            GramCG(IdentityPreconditioner(), tol=tol, maxiter=64)
            if solver_case == "gram_cg"
            else Cholesky()
        )
        solver = LevenbergMarquardt(
            residual,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
            cache_jacobian=False,
            geodesic_acceleration=False,
        )

        def solve_obs(param):
            sol = solver.solve(x_star, p=param, max_steps=5, atol=tol)
            return proj_out @ sol.x

        def status(param):
            return solver.solve(x_star, p=param, max_steps=5, atol=tol).status

    return solve_obs, p, p_batch, status


def _make_transformed(solver_case, transform, *, device=None, dtype=jnp.float32):
    solve_obs, p, p_batch, _ = _make_large_ad_problem(
        solver_case, device=device, dtype=dtype
    )
    if transform == "primal":
        return jax.jit(solve_obs), p
    if transform == "vmap_primal":
        return jax.jit(jax.vmap(solve_obs)), p_batch
    if transform == "jvp":
        tangent = jnp.ones_like(p)
        return jax.jit(lambda param: jax.jvp(solve_obs, (param,), (tangent,))), p
    if transform == "vmap_jvp":
        return jax.jit(jax.jacfwd(solve_obs)), p
    if transform == "vjp":
        return jax.jit(jax.grad(lambda param: jnp.sum(solve_obs(param) ** 2))), p
    if transform == "vmap_vjp":
        return jax.jit(jax.jacrev(solve_obs)), p
    grad_one = jax.grad(lambda param: jnp.sum(solve_obs(param) ** 2))
    return jax.jit(jax.vmap(grad_one)), p_batch


@pytest.mark.parametrize("solver_case", SOLVER_CASES)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_large_implicit_ad(benchmark, device, dtype, solver_case, transform):
    with jax.default_device(device):
        _, p, _, status = _make_large_ad_problem(
            solver_case, device=device, dtype=dtype
        )
        assert bool(jnp.all(status(p) == LMStatus.CONVERGED))
        fn, arg = _make_transformed(solver_case, transform, device=device, dtype=dtype)
        warmup = fn(arg)
        jax.block_until_ready(warmup)

        def run():
            value = fn(arg)
            jax.block_until_ready(value)
            return value

        rounds = 10 if device.platform in ("gpu", "metal") else 15
        iterations = 3 if device.platform in ("gpu", "metal") else 5
        benchmark.pedantic(run, rounds=rounds, iterations=iterations)
