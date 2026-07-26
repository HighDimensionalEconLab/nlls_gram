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
    AnnealRidge,
    BlockEigenPreconditioner,
    Cholesky,
    CholeskyMetric,
    DiagonalMetric,
    IdentityPreconditioner,
    LevenbergMarquardt,
    MultiStart,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)
from nlls_gram.multi_start import (
    _multi_start_parallel_jit,
    _multi_start_sequential_jit,
)
from nlls_gram.solve_loop import _solve_loop_jit

N, M = 5, 3
A = jnp.asarray(np.random.default_rng(3).normal(size=(M, N)), jnp.float32)
SOLVE = dict(max_steps=20, atol=1e-6)


def residual(x, args, p):
    return A @ x - p["b"]


def make_p(scale=1.0):
    return {"b": scale * jnp.ones(M, jnp.float32)}


JITTED = (_solve_loop_jit, _multi_start_sequential_jit, _multi_start_parallel_jit)


@pytest.fixture(autouse=True)
def fresh_cache():
    for jitted in JITTED:
        jitted._clear_cache()
    yield
    for jitted in JITTED:
        jitted._clear_cache()


def compilations():
    # The multi-start drivers have their own caches, keyed on draw/accept and
    # num_starts as well; counting only the plain loop would miss them.
    return sum(jitted._cache_size() for jitted in JITTED)


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


# Metric instances key compilation by pytree STRUCTURE (type + static
# fields); their arrays are threaded through the carried state. A fresh
# equal-config metric per solve must therefore share one compiled loop --
# and, the other half of the guarantee, its VALUES must actually be used.
METRICS = {
    "cholesky": lambda scale=1.0: CholeskyMetric(scale * jnp.eye(N, dtype=jnp.float32)),
    "diagonal": lambda scale=1.0: DiagonalMetric(scale * jnp.ones(N, jnp.float32)),
    "repeated": lambda scale=1.0: RepeatedFactorMetric(
        scale * jnp.eye(N, dtype=jnp.float32)
    ),
}


@pytest.mark.parametrize("name", list(METRICS))
def test_fresh_equal_config_metric_compiles_once(name):
    for scale in (1.0, 2.0, 3.0):
        LevenbergMarquardt(residual, metric=METRICS[name]()).solve(
            jnp.zeros(N), p=make_p(scale), **SOLVE
        )
    assert compilations() == 1


def test_shared_compile_uses_each_metrics_own_values():
    # Two same-treedef, different-valued metrics share one compiled loop; a
    # leftover static read would silently reuse the first metric's factor
    # for the second solver. The damping geometry selects the returned root
    # of this underdetermined system, so different weights must move x.
    def solve_with(scale):
        return LevenbergMarquardt(
            residual,
            metric=DiagonalMetric(
                jnp.asarray([100.0, 1.0, 1.0, 1.0, 100.0], jnp.float32) ** scale
            ),
        ).solve(jnp.zeros(N), p=make_p(), max_steps=60, atol=1e-6)

    heavy_ends = solve_with(1.0)
    heavy_middle = solve_with(-1.0)
    assert compilations() == 1
    assert not np.allclose(
        np.asarray(heavy_ends.x), np.asarray(heavy_middle.x), atol=1e-4
    )


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
    callback = AnnealRidge(ridge_floor=ridge_floor)
    return solver.solve(
        jnp.zeros(N),
        p=make_p(scale),
        callback=callback,
        user_state=callback.init_state(),
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


def draw_shifted(key, x, args):
    return x + 0.1 * jax.random.normal(key, x.shape, x.dtype), args


def test_multi_start_reuses_its_driver_compilation():
    solver = LevenbergMarquardt(residual)
    for scale in (1.0, 2.0, 3.0):
        solver.solve(
            jnp.zeros(N),
            p=make_p(scale),
            multi_start=MultiStart(
                key=jax.random.key(0), num_starts=3, draw=draw_shifted
            ),
            **SOLVE,
        )
    # The sequential driver inlines the loop into its own jit, so there is
    # one program total -- and reusing an equal MultiStart adds none.
    assert compilations() == 1


def test_a_fresh_residual_closure_per_call_recompiles():
    # The commonest real-world trap, and one the package CANNOT fix: the
    # residual is a jit static, so a lambda rebuilt per iteration is a new
    # program every time. Pinned as a negative control so the guards above
    # cannot pass by accident.
    for scale in (1.0, 2.0, 3.0):
        LevenbergMarquardt(lambda x, args, p: A @ x - p["b"]).solve(
            jnp.zeros(N), p=make_p(scale), **SOLVE
        )
    assert compilations() == 3
    # A module-level residual reused across calls compiles once.
    for jitted in JITTED:
        jitted._clear_cache()
    for scale in (1.0, 2.0, 3.0):
        LevenbergMarquardt(residual).solve(jnp.zeros(N), p=make_p(scale), **SOLVE)
    assert compilations() == 1


def test_save_steps_makes_max_steps_static():
    # history_len is a jit static because the buffer shape depends on it --
    # documented, and pinned here so the cost is not a surprise.
    solver = LevenbergMarquardt(residual)
    for max_steps in (5, 6, 7):
        solver.solve(jnp.zeros(N), p=make_p(), max_steps=max_steps, atol=1e-6)
    assert compilations() == 1
    for jitted in JITTED:
        jitted._clear_cache()
    for max_steps in (5, 6, 7):
        solver.solve(
            jnp.zeros(N), p=make_p(), max_steps=max_steps, atol=1e-6, save_steps=True
        )
    assert compilations() == 3


def test_fresh_equal_config_preconditioner_compiles_once():
    # BlockEigenPreconditioner keys by structure too: a fresh equal-config
    # instance per solve (same family count, shapes, permutation length)
    # shares the compiled loop; its eigendecomposition arrays ride in the
    # carried state.
    for scale in (1.0, 2.0, 3.0):
        preconditioner = BlockEigenPreconditioner(
            [(scale * jnp.eye(N, dtype=jnp.float32)[None], 0.0)], jnp.arange(N)
        )
        LevenbergMarquardt(
            residual, linear_solver=CG(preconditioner, tol=1e-8, maxiter=16)
        ).solve(jnp.zeros(N), p=make_p(scale), **SOLVE)
    assert compilations() == 1


def test_callback_instance_swap_does_not_recompile():
    # A callback that rebuilds the carried metric/preconditioner inside the
    # loop constructs same-structure instances: no recompilation across
    # solves, and none from the swap itself.
    import dataclasses as dc

    from nlls_gram import LMAction

    metric0 = DiagonalMetric(jnp.ones(N, jnp.float32))

    def swap(ctx):
        fresh = jax.lax.cond(
            ctx.step == 2,
            lambda: DiagonalMetric(2.0 * jnp.ones(N, jnp.float32)),
            lambda: ctx.lm_state.metric,
        )
        return LMAction(lm_state=dc.replace(ctx.lm_state, metric=fresh))

    for scale in (1.0, 2.0):
        LevenbergMarquardt(residual, metric=metric0).solve(
            jnp.zeros(N), p=make_p(scale), callback=swap, **SOLVE
        )
    assert compilations() == 1


def test_a_different_instance_structure_is_a_different_program():
    LevenbergMarquardt(residual, metric=DiagonalMetric(jnp.ones(N, jnp.float32))).solve(
        jnp.zeros(N), p=make_p(), **SOLVE
    )
    LevenbergMarquardt(
        residual, metric=RepeatedFactorMetric(jnp.eye(N, dtype=jnp.float32))
    ).solve(jnp.zeros(N), p=make_p(), **SOLVE)
    assert compilations() == 2
