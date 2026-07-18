import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    blockdiag_metric,
    identity_preconditioner,
    matern_state_space,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_from_shifted_matvec,
    metric_from_state_space,
    woodbury_preconditioner,
)
from nlls_gram.quasiseparable import matvec as quasiseparable_matvec
from nlls_gram.quasiseparable import state_space_generators

SIGMA, ELL = 1.3, 0.8

DENSE_MAX_N = 1_000


def _devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _grid(n, device):
    # Unit spacing keeps lambda_max = O(1), so eps is the floor scale.
    return jax.device_put(jnp.arange(n) * 1.0, device)


def _dense_gram(t):
    ft = jnp.sqrt(5.0) * jnp.abs(t[:, None] - t[None, :]) / ELL
    return SIGMA**2 * (1.0 + ft + ft**2 / 3.0) * jnp.exp(-ft)


def _kernel_matvec(t):
    # O(n) matrix-free Matern-5/2 matvec through the quasiseparable form.
    d, p, q, A = state_space_generators(t, *matern_state_space(SIGMA, ELL, 2.5))

    def matvec(x):
        return quasiseparable_matvec(
            d, p, q, A, x.reshape(t.shape[0], -1), False
        ).reshape(x.shape)

    return matvec


def _kernel_block(variant, t, eps):
    if variant == "dense_cholesky":
        K = _dense_gram(t)
        return metric_from_cholesky(jnp.linalg.cholesky(K + eps * jnp.eye(t.shape[0])))
    if variant == "state_space":
        return metric_from_state_space(
            t, *matern_state_space(SIGMA, ELL, 2.5), nugget=eps
        )
    return metric_from_shifted_matvec(_kernel_matvec(t), eps)


def _inner_cg_steps(t, eps, x):
    # Iteration count of the metric's ACTUAL inner solver (jax.scipy cg with
    # the default sqrt-machine-eps tolerance): doubling search for the
    # smallest maxiter whose residual meets the tolerance. Saturates at the
    # 10n budget when the tolerance is below CG's attainable floor
    # (~machine_eps * cond -- a real float32 hazard at small eps); that
    # saturation is itself the datum.
    matvec = _kernel_matvec(t)
    tol = float(jnp.finfo(x.dtype).eps) ** 0.5
    threshold = tol * float(jnp.linalg.norm(x))
    budget = 10 * x.shape[0]
    steps = 1
    while steps < budget:
        solution, _ = jsp_sparse_linalg.cg(
            lambda v: matvec(v) + eps * v, x, tol=tol, atol=0.0, maxiter=steps
        )
        residual = float(jnp.linalg.norm(matvec(solution) + eps * solution - x))
        if residual <= threshold:
            return steps
        steps *= 2
    return budget


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("n", [1_000, 10_000])
@pytest.mark.parametrize("variant", ["dense_cholesky", "state_space", "matvec_cg"])
@pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
def test_shifted_metric_apply(benchmark, platform, n, variant, eps):
    if not _devices(platform):
        pytest.skip(f"JAX {platform!r} backend is not available")
    if variant != "matvec_cg" and eps != 1e-4:
        pytest.skip("direct factorizations are eps-independent; one eps suffices")
    if variant == "dense_cholesky" and n > DENSE_MAX_N:
        pytest.skip("dense factorization too large to materialize")

    device = _devices(platform)[0]
    t = _grid(n, device)
    metric = _kernel_block(variant, t, eps)
    # A rough (full-spectrum) right-hand side: a smooth one converges fast
    # regardless of the condition number and would hide the eps-dependence.
    x = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (n,)), device)

    if variant == "matvec_cg":
        benchmark.extra_info["inner_cg_steps"] = _inner_cg_steps(t, eps, x)

    @jax.jit
    def apply(v):
        return metric.solve(v), metric.norm(v)

    jax.block_until_ready(apply(x))

    def run():
        out = apply(x)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("n", [1_000, 10_000])
@pytest.mark.parametrize(
    "variant",
    ["dense_cholesky", "state_space", "matvec_cg", "matvec_cg_woodbury"],
)
@pytest.mark.parametrize("linear_solver", ["cholesky", "cg"])
def test_shifted_metric_solver_step(benchmark, platform, n, variant, linear_solver):
    if not _devices(platform):
        pytest.skip(f"JAX {platform!r} backend is not available")
    if variant == "dense_cholesky" and n > DENSE_MAX_N:
        pytest.skip("dense factorization too large to materialize")
    if variant == "matvec_cg_woodbury" and linear_solver != "cg":
        pytest.skip("the Woodbury dual preconditioner applies to cg only")

    device = _devices(platform)[0]
    eps, m, k = 1e-4, 50, 2
    t = _grid(n, device)
    kernel_variant = "matvec_cg" if variant == "matvec_cg_woodbury" else variant
    metric = blockdiag_metric(
        [
            (_kernel_block(kernel_variant, t, eps), n),
            (metric_from_diagonal(eps * jnp.ones(k)), k),
        ]
    )

    key_alpha, key_beta, key_b = jax.random.split(jax.random.PRNGKey(0), 3)
    J_alpha = jax.device_put(jax.random.normal(key_alpha, (m, n)) / jnp.sqrt(n), device)
    J_beta = jax.device_put(jax.random.normal(key_beta, (m, k)), device)
    A = jnp.concatenate([J_alpha, J_beta], axis=1)
    b = jax.device_put(jax.random.normal(key_b, (m,)), device)

    def residual(theta, args, p):
        return A @ theta - b

    solver_kwargs = {
        "init_damping": 1e-2,
        "linear_solver": linear_solver,
        "geodesic_acceleration": False,
        "metric": metric,
    }
    if linear_solver == "cg":
        solver_kwargs.update(
            {
                "iterative_tol": 1e-8,
                "iterative_maxiter": 100,
                "dual_preconditioner": identity_preconditioner(),
                "implicit_preconditioner": identity_preconditioner(),
            }
        )
    if variant == "matvec_cg_woodbury":
        kernel_solve = _kernel_block(kernel_variant, t, eps).solve
        base = J_alpha @ kernel_solve(J_alpha.T)
        base_solve = metric_from_cholesky(jnp.linalg.cholesky(base)).solve
        solver_kwargs["dual_preconditioner"] = woodbury_preconditioner(
            base_solve, J_beta, (1.0 / eps) * jnp.ones(k)
        )

    solver = LevenbergMarquardt(residual, **solver_kwargs)
    x0 = jax.device_put(jnp.zeros(n + k), device)
    lm_state = solver.init(x0, None)

    @jax.jit
    def step(x, state):
        return solver.update(x, state, None)

    jax.block_until_ready(step(x0, lm_state))

    def run():
        out = step(x0, lm_state)
        jax.block_until_ready(out)
        return out

    benchmark(run)
