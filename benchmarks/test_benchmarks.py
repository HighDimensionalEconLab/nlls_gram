"""End-to-end performance and memory benchmarks for nlls-gram.

Covers the four workload families where runtime and temporary memory footprint
regressions matter in practice:
1. Wide nonlinear RBF interpolation (m=256, n=2048) across LM and Newton
   (Cholesky, QR, GramCG, and geodesic acceleration).
2. Wide nonlinear RBF solve + implicit reverse-mode differentiation (grad and
   vmapped grad).
3. Tall overdetermined nonlinear least squares (m=512, n=64).
4. Multi-block Matérn kernel ridge continuation (AnnealRidge) + implicit grad
   (n=483, m=480) across Cholesky, QR, and BlockEigen-preconditioned CG.
5. Simulated-trajectory Euler residual step under jax.vmap vs vmap_rows (and
   jacobian_chunk), recording compiled temporary memory (temp_mb).
"""

import jax
import jax.numpy as jnp
import pytest

from benchmarks.conftest import benchmark_batch_size, benchmark_direction_count
from nlls_gram import (
    CG,
    PDAS,
    QR,
    AnnealRidge,
    BlockEigenPreconditioner,
    Cholesky,
    Complementarity,
    FischerBurmeister,
    GramCG,
    IdentityPreconditioner,
    LevenbergMarquardt,
    Minimum,
    Newton,
    RepeatedFactorMetric,
    SolverStatus,
    vmap_rows,
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
        "lm_cholesky",
    ],
)
def test_wide_nonlinear_rbf_solve(benchmark, device, dtype, solver_name):
    residual, x0, args = _make_wide_rbf_problem(device, dtype)
    tol = 1e-4 if dtype == jnp.float32 else 1e-8

    with jax.default_device(device):
        if solver_name == "newton_cholesky":
            solver = Newton(
                residual,
                linear_solver=Cholesky(form="gram"),
                geodesic_acceleration=True,
            )
        elif solver_name == "newton_qr":
            solver = Newton(
                residual,
                linear_solver=QR(form="gram"),
                geodesic_acceleration=True,
            )
        elif solver_name == "newton_gram_cg":
            solver = Newton(
                residual,
                linear_solver=GramCG(IdentityPreconditioner(), tol=tol, maxiter=48),
                geodesic_acceleration=True,
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
        assert int(warm.status) == int(SolverStatus.CONVERGED)
        benchmark.extra_info["steps"] = int(warm.steps)
        benchmark.extra_info["final_loss"] = float(warm.info.loss)

        def run():
            out = run_solve(x0, args)
            jax.block_until_ready(out)
            return out

        benchmark.group = f"wide-rbf-solve-{dtype.__name__}"
        benchmark(run)


@pytest.mark.parametrize("transform", ["grad", "vmap_grad"])
@pytest.mark.parametrize("solver_kind", ["newton", "lm"])
def test_wide_rbf_solve_and_implicit_grad(
    benchmark, device, dtype, solver_kind, transform
):
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
                geodesic_acceleration=True,
            )
        else:
            solver = LevenbergMarquardt(
                param_residual,
                init_damping=1e-3,
                linear_solver=Cholesky(form="gram"),
                ad_solver=Cholesky(form="gram"),
                cache_jacobian=False,
                geodesic_acceleration=True,
            )

        def loss_fn(param):
            sol = solver.solve(x0, args, p=param, max_steps=15, atol=tol)
            return jnp.sum((proj_out @ sol.x) ** 2)

        grad_one = jax.grad(loss_fn)
        if transform == "grad":
            fn, arg = jax.jit(grad_one), p
        else:
            fn, arg = jax.jit(jax.vmap(grad_one)), p_batch

        jax.block_until_ready(fn(arg))

        def run():
            out = fn(arg)
            jax.block_until_ready(out)
            return out

        benchmark.group = f"wide-rbf-{transform}-{dtype.__name__}"
        benchmark(run)


@pytest.mark.parametrize(
    "solver_name",
    ["newton_cholesky", "newton_qr", "newton_cg", "lm_cholesky"],
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
            solver = Newton(
                residual,
                linear_solver=Cholesky(form="normal"),
                geodesic_acceleration=True,
            )
        elif solver_name == "newton_qr":
            solver = Newton(
                residual,
                linear_solver=QR(form="normal"),
                geodesic_acceleration=True,
            )
        elif solver_name == "newton_cg":
            solver = Newton(
                residual,
                linear_solver=CG(IdentityPreconditioner(), tol=tol, maxiter=48),
                geodesic_acceleration=True,
            )
        else:
            solver = LevenbergMarquardt(
                residual,
                init_damping=1e-3,
                linear_solver=Cholesky(form="normal"),
                cache_jacobian=False,
                geodesic_acceleration=True,
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


@pytest.mark.parametrize(
    "solver_name",
    [
        "lm_cholesky",
        "lm_qr",
        "lm_cg_block_eigen",
        "newton_cholesky",
        "newton_qr",
    ],
)
def test_ridge_continuation_solve_and_grad(benchmark, device, dtype, solver_name):
    n_grid, repeats, free_size, m_resid = 64, 4, 3, 256
    n_m = n_grid * repeats
    p_dim = n_m + free_size
    eps = 1e-2 if dtype == jnp.float32 else 1e-4
    ridge_floor = 1e-2 if dtype == jnp.float32 else 1e-5
    gtol = 1e-3 if dtype == jnp.float32 else 5e-5
    atol = 1e-1 if dtype == jnp.float32 else 1e-2

    with jax.default_device(device):
        t = jnp.linspace(0.0, 20.0, n_grid, dtype=dtype)
        dist = jnp.abs(t[:, None] - t[None, :])
        scaled = jnp.sqrt(jnp.asarray(5.0, dtype=dtype)) * dist / 10.0
        K = (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)
        shifted = K + jnp.asarray(eps, dtype=dtype) * jnp.eye(n_grid, dtype=dtype)
        F = jnp.linalg.cholesky(shifted, upper=True)
        metric = RepeatedFactorMetric(F, repeats=repeats)

        k1, k2, k3 = jax.random.split(jax.random.key(5), 3)
        A = (
            0.2
            * jax.random.normal(k1, (m_resid, p_dim), dtype=dtype)
            / jnp.sqrt(jnp.asarray(p_dim, dtype=dtype))
        )
        A = A.at[:, :n_m].add(jnp.eye(n_m, dtype=dtype))
        A = A.at[:free_size, n_m:].add(jnp.eye(free_size, dtype=dtype))
        b = jax.random.normal(k2, (m_resid,), dtype=dtype) * 0.2
        weights = jax.random.normal(k3, (p_dim,), dtype=dtype) / jnp.sqrt(
            jnp.asarray(p_dim, dtype=dtype)
        )
        x0 = jnp.zeros(p_dim, dtype=dtype)
        p0 = jnp.ones(4, dtype=dtype)

        def residual(theta, _, param):
            scale = jnp.mean(param)
            return A @ (theta + 0.02 * theta**2) - scale * b

        if solver_name.endswith("cholesky"):
            linear_solver = Cholesky()
            ad_solver = Cholesky()
        elif solver_name.endswith("qr"):
            linear_solver = QR()
            ad_solver = Cholesky()
        else:
            F_inv = jnp.linalg.inv(F)
            A_m = A[:, :n_m].reshape(m_resid, repeats, n_grid)
            A_w = jnp.einsum("mrg,gh->mrh", A_m, F_inv)
            gram_blocks = jnp.einsum("mrg,mrh->rgh", A_w, A_w)
            free_block = (
                A[:, n_m:].T @ A[:, n_m:] + 1e-2 * jnp.eye(free_size, dtype=dtype)
            )[None]
            precond = BlockEigenPreconditioner(
                [(gram_blocks, 1.0), (free_block, 0.0)],
                jnp.arange(p_dim, dtype=jnp.int32),
            )
            linear_solver = CG(precond, tol=gtol * 0.01, maxiter=128)
            ad_solver = CG(precond, tol=gtol * 0.01, maxiter=128)

        anneal = AnnealRidge(ridge_floor=ridge_floor, decrease=0.1, grad_rtol=0.1)
        user_state0 = anneal.init_state(dtype=dtype)
        solver_cls = Newton if solver_name.startswith("newton") else LevenbergMarquardt
        lm_kw = {"init_damping": 1e-6} if solver_cls is LevenbergMarquardt else {}
        solver = solver_cls(
            residual,
            metric=metric,
            ridge=ridge_floor * 100.0,
            linear_solver=linear_solver,
            ad_solver=ad_solver,
            cache_jacobian=False,
            geodesic_acceleration=True,
            **lm_kw,
        )

        def solve_and_functional(param):
            sol = solver.solve(
                x0,
                p=param,
                max_steps=40,
                gtol=gtol,
                atol=atol,
                callback=anneal,
                user_state=user_state0,
            )
            return jnp.vdot(weights, sol.x)

        warm_sol = solver.solve(
            x0,
            p=p0,
            max_steps=40,
            gtol=gtol,
            atol=atol,
            callback=anneal,
            user_state=user_state0,
        )
        jax.block_until_ready(warm_sol)
        assert int(warm_sol.status) == int(SolverStatus.CONVERGED)
        assert float(warm_sol.state.ridge) <= ridge_floor * 1.01
        benchmark.extra_info["steps"] = int(warm_sol.steps)

        fn = jax.jit(jax.value_and_grad(solve_and_functional))
        benchmark.extra_info["temp_mb"] = _temp_mb(fn.lower(p0))
        jax.block_until_ready(fn(p0))

        def run():
            out = fn(p0)
            jax.block_until_ready(out)
            return out

        benchmark.group = f"ridge-continuation-grad-{dtype.__name__}"
        benchmark(run)


WIDTH, T_HORIZON, Q_SHOCKS = 16, 64, 3
IN_AXES = (0, None, None, None)


def _policy(params, x):
    h = jnp.tanh(x[:, None] * params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    return jnp.mean(h, axis=0) @ params["w3"] + params["b3"]


def _simulate(x0, shocks, params, p):
    def step(x, shock):
        x_next = p["rho"] * x + _policy(params, x) + shock
        return x_next, x_next

    return jax.lax.scan(step, x0, shocks)[1]


def _euler_row(x, next_shocks, params, p):
    u = _policy(params, x)
    x_next = p["rho"] * x[None, :] + u + next_shocks
    u_next = jax.vmap(_policy, in_axes=(None, 0))(params, x_next)
    return u - p["beta"] * jnp.mean(u_next)


BLOCKS = {
    "vmap": jax.vmap(_euler_row, in_axes=IN_AXES),
    "vmap_rows": vmap_rows(_euler_row, in_axes=IN_AXES),
}


@pytest.mark.parametrize(
    ("block_name", "jacobian_chunk"),
    [("vmap", None), ("vmap_rows", None), ("vmap_rows", 16)],
    ids=["vmap", "vmap_rows", "vmap_rows-chunk16"],
)
def test_trajectory_vmap_rows_update(
    benchmark, device, dtype, block_name, jacobian_chunk
):
    n_firms = 512
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
            0.05 * jax.random.normal(keys[6], (T_HORIZON, n_firms), dtype),
            0.05 * jax.random.normal(keys[7], (Q_SHOCKS, n_firms), dtype),
        )
        p = {"rho": jnp.asarray(0.9, dtype), "beta": jnp.asarray(0.95, dtype)}
        block = BLOCKS[block_name]

        def residual(th, solver_args, param):
            x0, shocks, next_shocks = solver_args
            xs = _simulate(x0, shocks, th, param)
            r = block(xs, next_shocks, th, param)
            return r / jnp.sqrt(r.size)

        solver = LevenbergMarquardt(
            residual,
            geodesic_acceleration=True,
            cache_jacobian=False,
            jacobian_chunk=jacobian_chunk,
        )
        state = solver.init(theta, args, p=p)

        @jax.jit
        def step(th, s):
            return solver.update(th, s, args, p)

        benchmark.extra_info["temp_mb"] = _temp_mb(step.lower(theta, state))
        jax.block_until_ready(step(theta, state))

        def run():
            out = step(theta, state)
            jax.block_until_ready(out)
            return out

        benchmark.group = f"vmap-rows-update-{n_firms}-{dtype.__name__}"
        benchmark(run)


@pytest.mark.parametrize("comp_name", ["pdas", "fb", "minimum"])
def test_euler_complementarity_gram_solve(benchmark, device, dtype, comp_name):
    n_grid, n_centers = 64, 256
    tol = 1e-4 if dtype == jnp.float32 else 1e-8
    with jax.default_device(device):
        beta = jnp.asarray(0.95, dtype=dtype)
        r_gross = jnp.asarray(1.05, dtype=dtype)
        y = jnp.asarray(1.1, dtype=dtype)
        w_grid = jnp.linspace(0.5, 2.5, n_grid, dtype=dtype)
        centers = jnp.linspace(0.3, 2.7, n_centers, dtype=dtype)
        phi = jnp.exp(
            -0.5
            * ((w_grid[:, None] - centers[None, :]) / jnp.asarray(0.15, dtype)) ** 2
        ) / jnp.sqrt(jnp.asarray(n_centers, dtype))
        phi = phi.at[:, :n_grid].add(jnp.eye(n_grid, dtype=dtype))
        theta0 = jnp.zeros(n_centers, dtype=dtype)

        def residual(theta):
            a = phi @ theta
            c1 = w_grid - a
            c2 = r_gross * a + y
            return Complementarity(lhs=a, rhs=1.0 / c1 - beta * r_gross / c2)

        comp_cfg = {
            "pdas": PDAS(),
            "fb": FischerBurmeister(),
            "minimum": Minimum(),
        }[comp_name]
        solver = LevenbergMarquardt(
            residual,
            linear_solver=Cholesky(form="gram"),
            complementarity=comp_cfg,
            cache_jacobian=False,
            geodesic_acceleration=True,
        )

        @jax.jit
        def run_solve(th):
            return solver.solve(th, atol=tol, max_steps=40)

        benchmark.extra_info["temp_mb"] = _temp_mb(run_solve.lower(theta0))
        warm = run_solve(theta0)
        jax.block_until_ready(warm)
        assert int(warm.status) == int(SolverStatus.CONVERGED)
        benchmark.extra_info["steps"] = int(warm.steps)
        benchmark.extra_info["final_loss"] = float(warm.info.loss)

        def run():
            out = run_solve(theta0)
            jax.block_until_ready(out)
            return out

        benchmark.group = f"euler-comp-gram-solve-{dtype.__name__}"
        benchmark(run)
