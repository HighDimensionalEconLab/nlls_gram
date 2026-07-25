"""Recompilation guards.

The jitted solve loop marks the SOLVER and the CALLBACK static, so anything
reaching the solver's static key by object identity recompiles the whole loop
per construction. Converting a closure to a class is not by itself enough --
a class that hashes on an array or on a user closure has the same behavior,
just relocated. These tests pin what must and must not recompile.

The instrument is the jitted loop's own compilation cache, so a "compilation"
here is exactly what JAX counts as one.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    CG,
    QR,
    Cholesky,
    CholeskyMetric,
    DiagonalMetric,
    IdentityPreconditioner,
    LevenbergMarquardt,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    ridge_continuation,
)
from nlls_gram.solve_loop import _solve_loop_jit

N, M = 5, 3
A = jnp.asarray(np.random.default_rng(3).normal(size=(M, N)), jnp.float32)
SOLVE = dict(max_steps=20, atol=1e-6)


def residual(x, args, p):
    return A @ x - p["b"]


def make_p(scale=1.0):
    return {"b": scale * jnp.ones(M, jnp.float32)}


@pytest.fixture(autouse=True)
def fresh_cache():
    _solve_loop_jit._clear_cache()
    yield
    _solve_loop_jit._clear_cache()


def compilations():
    return _solve_loop_jit._cache_size()


def test_equal_solvers_built_repeatedly_share_one_compilation():
    for scale in (1.0, 2.0, 3.0):
        LevenbergMarquardt(residual, linear_solver=Cholesky()).solve(
            jnp.zeros(N), p=make_p(scale), **SOLVE
        )
    assert compilations() == 1


def test_traced_values_and_loop_controls_do_not_recompile():
    solver = LevenbergMarquardt(residual)
    solver.solve(jnp.zeros(N), p=make_p(), **SOLVE)
    solver.solve(jnp.zeros(N), p=make_p(2.0), max_steps=20, atol=1e-8)
    solver.solve(jnp.zeros(N), p=make_p(3.0), max_steps=40, atol=1e-6, gtol=1e-9)
    solver.solve(jnp.zeros(N), p=make_p(4.0), max_steps=20, atol=1e-6, xtol=1e-9)
    assert compilations() == 1


def test_shape_change_compiles_one_extra_program_and_reuses_both():
    wide = jnp.asarray(np.random.default_rng(4).normal(size=(M, N + 2)), jnp.float32)

    def sized(x, args, p):
        return (A if x.shape[0] == N else wide) @ x - p["b"]

    solver = LevenbergMarquardt(sized)
    solver.solve(jnp.zeros(N), p=make_p(), **SOLVE)
    solver.solve(jnp.zeros(N + 2), p=make_p(), **SOLVE)
    assert compilations() == 2
    solver.solve(jnp.zeros(N), p=make_p(2.0), **SOLVE)
    solver.solve(jnp.zeros(N + 2), p=make_p(2.0), **SOLVE)
    assert compilations() == 2


# Metrics hold arrays, so they hash by identity and the documented contract is
# build-once-at-setup-scope. What must not happen is a recompilation when the
# same metric is reused and only the solver around it is rebuilt.
METRICS = {
    "cholesky": lambda: CholeskyMetric(jnp.eye(N, dtype=jnp.float32)),
    "diagonal": lambda: DiagonalMetric(jnp.ones(N, jnp.float32)),
    "repeated": lambda: RepeatedFactorMetric(jnp.eye(N, dtype=jnp.float32)),
}


@pytest.mark.parametrize("name", list(METRICS))
def test_reused_metric_survives_solver_rebuilds(name):
    metric = METRICS[name]()
    for scale in (1.0, 2.0, 3.0):
        LevenbergMarquardt(residual, metric=metric).solve(
            jnp.zeros(N), p=make_p(scale), **SOLVE
        )
    assert compilations() == 1


def test_stateless_preconditioner_is_value_equal():
    # IdentityPreconditioner holds nothing, so even a freshly constructed one
    # keys the same compilation -- the closure it replaced did not.
    for _ in range(3):
        LevenbergMarquardt(
            residual, linear_solver=CG(IdentityPreconditioner(), tol=1e-8, maxiter=32)
        ).solve(jnp.zeros(N), p=make_p(), **SOLVE)
    assert compilations() == 1


def ridge_solver():
    return RidgeLevenbergMarquardt(
        residual, metric=DiagonalMetric(jnp.ones(N, jnp.float32)), ridge=1e-3
    )


def run_continuation(solver, ridge_floor, scale=1.0):
    callback, user_state = ridge_continuation(ridge_floor=ridge_floor)
    return solver.solve(
        jnp.zeros(N),
        p=make_p(scale),
        callback=callback,
        user_state=user_state,
        max_steps=20,
        gtol=1e-6,
    )


def test_rebuilding_the_continuation_callback_does_not_recompile():
    # The callback is a jit STATIC argument, so the closure this used to
    # return recompiled the whole solve loop on every construction.
    solver = ridge_solver()
    for scale in (1.0, 2.0, 3.0):
        run_continuation(solver, 1e-8, scale)
    assert compilations() == 1


def test_a_different_continuation_schedule_is_a_different_program():
    solver = ridge_solver()
    run_continuation(solver, 1e-8)
    run_continuation(solver, 1e-9)
    assert compilations() == 2


def test_solver_rebuilt_inside_a_python_loop_compiles_once():
    # The realistic regression: a driver that constructs the solver each
    # iteration must not pay a compilation each iteration.
    metric = CholeskyMetric(jnp.eye(N, dtype=jnp.float32))
    for step in range(5):
        LevenbergMarquardt(
            residual, metric=metric, linear_solver=QR(), init_damping=1e-3
        ).solve(jnp.zeros(N), p=make_p(1.0 + step), **SOLVE)
    assert compilations() == 1


def test_a_changed_static_setting_is_a_different_program():
    metric = DiagonalMetric(jnp.ones(N, jnp.float32))

    def solve(**kwargs):
        LevenbergMarquardt(residual, metric=metric, **kwargs).solve(
            jnp.zeros(N), p=make_p(), **SOLVE
        )

    solve(linear_solver=Cholesky())
    solve(linear_solver=Cholesky())
    assert compilations() == 1
    solve(linear_solver=QR())  # a different algorithm
    assert compilations() == 2
    solve(linear_solver=QR(), damping_increase=8.0)  # a static scalar
    assert compilations() == 3


def test_matrix_free_path_never_factorizes():
    # The dense path's signature is a Cholesky factorization of an assembled
    # matrix; the matrix-free path must contain none.
    def jaxpr_text(config):
        solver = LevenbergMarquardt(residual, linear_solver=config)
        return str(
            jax.make_jaxpr(
                lambda pv: solver.solve(jnp.zeros(N), p=pv, **SOLVE).x  # noqa: B023
            )(make_p())
        )

    assert "cholesky" in jaxpr_text(Cholesky())
    assert "cholesky" not in jaxpr_text(
        CG(IdentityPreconditioner(), tol=1e-8, maxiter=16)
    )
