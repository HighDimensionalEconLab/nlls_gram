import jax
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt


def _gsl_rosenbrock_residual(theta, aux, p):
    del aux
    # GSL's geodesic acceleration example uses this modified Rosenbrock canyon.
    return jnp.array([100.0 * (theta[1] - theta[0] ** 2), 1.0 - theta[0]])


def _make_gsl_rosenbrock_step(*, geodesic_acceleration):
    theta = jnp.array([-0.5, 1.75])
    solver = UnderdeterminedLevenbergMarquardt(
        _gsl_rosenbrock_residual,
        init_damping=1.0,
        geodesic_acceleration=geodesic_acceleration,
    )
    state = solver.init()

    @jax.jit
    def step(theta, state):
        return solver.update(theta, state, None)

    return theta, state, step


def _run_to_threshold(theta, state, step):
    used_geodesic = 0
    for iteration in range(1, 101):
        theta, state, info = step(theta, state)
        jax.block_until_ready((theta, state, info))
        used_geodesic += int(bool(info.used_geodesic))
        if float(info.loss) < 1e-12:
            return iteration, used_geodesic
    return 101, used_geodesic


def test_gsl_rosenbrock_convergence_plain(benchmark):
    theta, state, step = _make_gsl_rosenbrock_step(geodesic_acceleration=False)
    warmup = _run_to_threshold(theta, state, step)
    jax.block_until_ready(warmup)

    def run():
        return _run_to_threshold(theta, state, step)

    iterations, used_geodesic = benchmark(run)
    assert used_geodesic == 0
    assert iterations < 101


def test_gsl_rosenbrock_convergence_geodesic(benchmark):
    theta, state, step = _make_gsl_rosenbrock_step(geodesic_acceleration=True)
    warmup = _run_to_threshold(theta, state, step)
    jax.block_until_ready(warmup)

    def run():
        return _run_to_threshold(theta, state, step)

    iterations, used_geodesic = benchmark(run)
    assert used_geodesic > 0
    assert iterations < 101
