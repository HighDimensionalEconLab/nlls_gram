import dataclasses

import jax
import jax.numpy as jnp
import pytest

from benchmarks.conftest import benchmark_batch_size, benchmark_direction_count
from nlls_gram import (
    CG,
    QR,
    AnnealRidge,
    BlockEigenPreconditioner,
    Cholesky,
    IdentityPreconditioner,
    LevenbergMarquardt,
    LMStatus,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)

SIGMA = 1.0
ELL = 10.0

N_GRID, REPEATS, FREE_SIZE, M_RESID = 96, 5, 3, 480


def _epsilon(dtype):
    return 1e-2 if dtype == jnp.float32 else 1e-7


def _ridge(dtype):
    return 1e-3 if dtype == jnp.float32 else 1e-8


def _matern_gram(t):
    distance = jnp.abs(t[:, None] - t[None, :])
    scaled = jnp.sqrt(jnp.asarray(5.0, dtype=t.dtype)) * distance / ELL
    correlation = (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)
    return jnp.asarray(SIGMA**2, dtype=t.dtype) * correlation


def _problem(n, repeats, free_size, m_resid, device, dtype):
    with jax.default_device(device):
        t = jnp.linspace(0.0, 40.0, n, dtype=dtype)
        K = _matern_gram(t)
        p_dim = repeats * n + free_size
        A = jax.random.normal(
            jax.random.key(0), (m_resid, p_dim), dtype=dtype
        ) / jnp.sqrt(jnp.asarray(p_dim, dtype=dtype))
        b = jax.random.normal(jax.random.key(1), (m_resid,), dtype=dtype)
        x0 = jnp.asarray(0.1, dtype=dtype) * jax.random.normal(
            jax.random.key(2), (p_dim,), dtype=dtype
        )

        def residual(theta):
            return A @ theta - b

    return K, residual, x0


def _ridge_metric(K, repeats):
    eps = _epsilon(K.dtype)
    shifted = K + jnp.asarray(eps, dtype=K.dtype) * jnp.eye(K.shape[0], dtype=K.dtype)
    return RepeatedFactorMetric(
        jnp.linalg.cholesky(shifted, upper=True), repeats=repeats
    )


@pytest.mark.parametrize("configuration", ["cholesky", "qr", "cg", "metric"])
def test_update_step(benchmark, device, dtype, configuration):
    n, repeats, free_size, m_resid = N_GRID, REPEATS, FREE_SIZE, M_RESID
    eps = _epsilon(dtype)
    ridge = _ridge(dtype)
    K, residual, x0 = _problem(n, repeats, free_size, m_resid, device, dtype)
    with jax.default_device(device):
        if configuration == "metric":
            shifted = K + jnp.asarray(eps, dtype=dtype) * jnp.eye(n, dtype=dtype)
            metric = RepeatedFactorMetric(
                jnp.linalg.cholesky(shifted, upper=True),
                repeats=repeats,
                free_scale=eps,
            )
            solver = LevenbergMarquardt(residual, metric=metric)
        else:
            if configuration == "cg":
                config = CG(IdentityPreconditioner(), maxiter=32)
            elif configuration == "qr":
                config = QR()
            else:
                config = Cholesky()
            solver = RidgeLevenbergMarquardt(
                residual,
                metric=_ridge_metric(K, repeats),
                ridge=ridge,
                linear_solver=config,
            )
        lm_state = solver.init(x0)
        step = jax.jit(solver.update)
        jax.block_until_ready(step(x0, lm_state))

    benchmark.group = f"ridge-vs-metric-update-{n}x{repeats}+{free_size}"

    def run():
        out = step(x0, lm_state)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("configuration", ["cholesky", "qr"])
def test_rejected_update_step(benchmark, device, dtype, configuration):
    n, repeats, free_size, m_resid = N_GRID, REPEATS, FREE_SIZE, M_RESID
    ridge = _ridge(dtype)
    K, residual, x0 = _problem(n, repeats, free_size, m_resid, device, dtype)
    with jax.default_device(device):
        solver = RidgeLevenbergMarquardt(
            residual,
            metric=_ridge_metric(K, repeats),
            ridge=ridge,
            linear_solver=Cholesky() if configuration == "cholesky" else QR(),
        )
        lm_state = solver.init(x0)

        stepped = solver.update(x0, lm_state)[1]
        warm = dataclasses.replace(
            stepped,
            jacobian_valid=jnp.asarray(True),
            solver_cache=dataclasses.replace(
                stepped.solver_cache, valid=jnp.asarray(True)
            ),
        )
        step = jax.jit(solver.update)
        jax.block_until_ready(step(x0, warm))

    benchmark.group = "ridge-rejected-update-96x5+3"

    def run():
        out = step(x0, warm)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("configuration", ["cholesky", "qr", "cg_block_eigen"])
def test_ridge_continuation_solve(benchmark, device, dtype, configuration):
    n, repeats, free_size, m_resid = N_GRID, REPEATS, FREE_SIZE, M_RESID
    n_m = n * repeats
    p_dim = n_m + free_size
    ridge_floor = 1e-4 if dtype == jnp.float32 else 1e-7
    gtol = 1e-4 if dtype == jnp.float32 else 1e-8
    K, residual, x0 = _problem(n, repeats, free_size, m_resid, device, dtype)
    with jax.default_device(device):
        metric = _ridge_metric(K, repeats)
        if configuration == "cholesky":
            linear_solver = Cholesky()
        elif configuration == "qr":
            linear_solver = QR()
        else:
            precond = BlockEigenPreconditioner(
                [
                    (jnp.broadcast_to(jnp.eye(n, dtype=dtype), (repeats, n, n)), 1.0),
                    (jnp.eye(free_size, dtype=dtype)[None], 0.0),
                ],
                jnp.arange(p_dim, dtype=jnp.int32),
            )
            linear_solver = CG(precond, maxiter=32)
        anneal = AnnealRidge(ridge_floor=ridge_floor, decrease=0.1)
        user_state0 = anneal.init_state(dtype=dtype)
        solver = RidgeLevenbergMarquardt(
            residual,
            metric=metric,
            ridge=ridge_floor * 100.0,
            linear_solver=linear_solver,
        )
        solve = jax.jit(
            lambda x: (
                solver.solve(
                    x,
                    max_steps=40,
                    gtol=gtol,
                    callback=anneal,
                    user_state=user_state0,
                ).x
            )
        )
        jax.block_until_ready(solve(x0))

    benchmark.group = "ridge-continuation-solve-96x5+3"

    def run():
        out = solve(x0)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("configuration", ["cholesky", "cg_block_eigen", "metric"])
@pytest.mark.parametrize(
    "transform",
    ["primal", "vmap_primal", "jvp", "vmap_jvp", "vjp", "vmap_vjp"],
)
def test_ridge_solve_and_ad(benchmark, device, dtype, configuration, transform):
    n, repeats, free_size, m_resid = N_GRID, REPEATS, FREE_SIZE, M_RESID
    p_dim = repeats * n + free_size
    batch = benchmark_batch_size(device)
    k_dir = benchmark_direction_count(device)
    eps = 1e-4 if dtype == jnp.float32 else 1e-7
    ridge = 1e-2 if dtype == jnp.float32 else 1e-5
    tol = 1e-2 if dtype == jnp.float32 else 1e-8

    with jax.default_device(device):
        t = jnp.linspace(0.0, 40.0, n, dtype=dtype)
        K = _matern_gram(t)
        k1, k2, k3, k4 = jax.random.split(jax.random.key(11), 4)
        A = jax.random.normal(k1, (m_resid, p_dim), dtype=dtype) / jnp.sqrt(
            jnp.asarray(p_dim, dtype=dtype)
        )
        A = A.at[:, :m_resid].add(2.0 * jnp.eye(m_resid, dtype=dtype))
        proj_in = jax.random.normal(k2, (m_resid, k_dir), dtype=dtype) / jnp.sqrt(
            jnp.asarray(k_dir, dtype=dtype)
        )
        proj_out = jax.random.normal(k3, (k_dir, p_dim), dtype=dtype) / jnp.sqrt(
            jnp.asarray(p_dim, dtype=dtype)
        )
        p = jax.random.normal(k4, (k_dir,), dtype=dtype) * 0.2
        p_batch = (
            p[None, :]
            + jax.random.normal(k4, (batch, k_dir), dtype=dtype) * 0.02
        )

        def residual(theta, _, param):
            return A @ (theta + 0.02 * theta**2) - proj_in @ param

        metric = _ridge_metric(K, repeats)
        warmup_solver = RidgeLevenbergMarquardt(
            residual,
            metric=metric,
            ridge=ridge,
            linear_solver=Cholesky(),
            cache_jacobian=False,
            geodesic_acceleration=False,
        )
        x_star = warmup_solver.solve(
            jnp.zeros(p_dim, dtype=dtype), p=p, max_steps=32, gtol=tol
        ).x
        jax.block_until_ready(x_star)

        if configuration == "metric":
            shifted = K + jnp.asarray(eps, dtype=dtype) * jnp.eye(n, dtype=dtype)
            metric_lm = RepeatedFactorMetric(
                jnp.linalg.cholesky(shifted, upper=True),
                repeats=repeats,
                free_scale=eps,
            )
            solver = LevenbergMarquardt(
                residual,
                metric=metric_lm,
                linear_solver=Cholesky(),
                ad_solver=Cholesky(),
                cache_jacobian=False,
                geodesic_acceleration=False,
            )
            x_star = solver.solve(x_star, p=p, max_steps=32, atol=tol).x
            jax.block_until_ready(x_star)

            def solve_obs(param):
                sol = solver.solve(x_star, p=param, max_steps=5, atol=tol)
                return proj_out @ sol.x

            assert bool(
                solver.solve(x_star, p=p, max_steps=5, atol=tol).status
                == LMStatus.CONVERGED
            )
        else:
            if configuration == "cholesky":
                linear_solver = Cholesky()
                ad_solver = Cholesky()
            else:
                precond = BlockEigenPreconditioner(
                    [
                        (
                            jnp.broadcast_to(
                                jnp.eye(n, dtype=dtype), (repeats, n, n)
                            ),
                            1.0,
                        ),
                        (jnp.eye(free_size, dtype=dtype)[None], 0.0),
                    ],
                    jnp.arange(p_dim, dtype=jnp.int32),
                )
                linear_solver = CG(precond, tol=tol * 0.1, maxiter=128)
                ad_solver = CG(precond, tol=tol * 0.1, maxiter=128)
            solver = RidgeLevenbergMarquardt(
                residual,
                metric=metric,
                ridge=ridge,
                linear_solver=linear_solver,
                ad_solver=ad_solver,
                cache_jacobian=False,
                geodesic_acceleration=False,
            )

            def solve_obs(param):
                sol = solver.solve(x_star, p=param, max_steps=5, gtol=tol)
                return proj_out @ sol.x

            assert bool(
                solver.solve(x_star, p=p, max_steps=5, gtol=tol).status
                == LMStatus.CONVERGED
            )

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
