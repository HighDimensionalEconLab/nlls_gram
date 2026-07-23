"""Per-update step cost: ridge LM (cholesky/qr/lsmr) vs metric LM at the
kernel-driver problem sizes (p = repeats * n + pad parameters, m residuals).
The residual is a fixed random affine map so the benchmark isolates the
solver's linear algebra, matching test_repeated_shifted_metric_benchmark's
kernel geometry.
"""

import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    LSMR,
    QR,
    Cholesky,
    LevenbergMarquardt,
    RidgeLevenbergMarquardt,
    identity_right_preconditioner,
    repeated_block_whitener,
    repeated_dense_penalty,
    repeated_shifted_dense_metric,
)

SIGMA = 1.0
ELL = 10.0
EPSILON = 1e-7
RIDGE = 1e-8

SIZES = [
    # (n, repeats, zero_pad_size, m_resid) -- kernel-driver-like shapes.
    (41, 3, 2, 123),
    (96, 5, 3, 480),
]


def _devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _matern_gram(t):
    distance = jnp.abs(t[:, None] - t[None, :])
    scaled = jnp.sqrt(5.0) * distance / ELL
    correlation = (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)
    return SIGMA**2 * correlation


def _problem(n, repeats, zero_pad_size, m_resid, device):
    t = jax.device_put(jnp.linspace(0.0, 40.0, n), device)
    K = _matern_gram(t)
    p_dim = repeats * n + zero_pad_size
    A = jax.device_put(
        jax.random.normal(jax.random.key(0), (m_resid, p_dim)) / jnp.sqrt(p_dim),
        device,
    )
    b = jax.device_put(jax.random.normal(jax.random.key(1), (m_resid,)), device)
    x0 = jax.device_put(
        0.1 * jax.random.normal(jax.random.key(2), (p_dim,)), device
    )

    def residual(theta):
        return A @ theta - b

    return K, residual, x0


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize(("n", "repeats", "zero_pad_size", "m_resid"), SIZES)
@pytest.mark.parametrize(
    "configuration", ["cholesky", "qr", "lsmr", "metric", "whitened"]
)
def test_update_step(
    benchmark, platform, n, repeats, zero_pad_size, m_resid, configuration
):
    devices = _devices(platform)
    if not devices:
        pytest.skip(f"JAX {platform!r} backend is not available")
    device = devices[0]
    K, residual, x0 = _problem(n, repeats, zero_pad_size, m_resid, device)
    if configuration == "metric":
        metric = repeated_shifted_dense_metric(
            K, repeats=repeats, zero_pad_size=zero_pad_size, epsilon=EPSILON
        )
        solver = LevenbergMarquardt(residual, metric=metric)
    else:
        # "whitened" is the whitened-cholesky path: the Whitener penalty
        # with the default linear solver -- expect cholesky-class step cost.
        if configuration == "whitened":
            penalty = repeated_block_whitener(
                K, repeats=repeats, zero_pad_size=zero_pad_size
            )
        else:
            penalty = repeated_dense_penalty(
                K, repeats=repeats, zero_pad_size=zero_pad_size
            )
        if configuration == "lsmr":
            config = LSMR(identity_right_preconditioner(), maxiter=32)
        elif configuration == "qr":
            config = QR()
        else:
            config = Cholesky()
        solver = RidgeLevenbergMarquardt(
            residual,
            penalty=penalty,
            ridge=RIDGE,
            linear_solver=config,
        )
    lm_state = solver.init(x0)
    step = jax.jit(solver.update)
    jax.block_until_ready(step(x0, lm_state))
    benchmark.group = f"ridge-vs-metric-update-{n}x{repeats}+{zero_pad_size}"

    def run():
        out = step(x0, lm_state)
        jax.block_until_ready(out)
        return out

    benchmark(run)


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("configuration", ["cholesky", "qr"])
def test_rejected_update_step(benchmark, platform, configuration):
    # The reject-path amortization: jacobian_valid + the ridge-keyed
    # normal/QR cache let a rejected step skip the linearization AND the
    # assembly, paying only the damping refactor and the trial evaluation.
    devices = _devices(platform)
    if not devices:
        pytest.skip(f"JAX {platform!r} backend is not available")
    device = devices[0]
    n, repeats, zero_pad_size, m_resid = 96, 5, 3, 480
    K, residual, x0 = _problem(n, repeats, zero_pad_size, m_resid, device)
    penalty = repeated_dense_penalty(K, repeats=repeats, zero_pad_size=zero_pad_size)
    solver = RidgeLevenbergMarquardt(
        residual,
        penalty=penalty,
        ridge=RIDGE,
        linear_solver=Cholesky() if configuration == "cholesky" else QR(),
    )
    lm_state = solver.init(x0)
    import dataclasses

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
