"""Compilation-cache and carried-instance state guards for LevenbergMarquardt
and Newton.
"""

import dataclasses

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
    GramCG,
    IdentityMetric,
    LevenbergMarquardt,
    Newton,
    Preconditioner,
    RepeatedFactorMetric,
    SolverAction,
    SolverStatus,
    register_pytree_dataclass,
)
from nlls_gram.multi_start import (
    _multi_start_parallel_jit,
    _multi_start_sequential_jit,
)
from nlls_gram.solve_loop import _solve_loop_jit

N, M = 5, 3
RNG = np.random.default_rng(3)
A = jnp.asarray(RNG.normal(size=(M, N)), jnp.float32)
SOLVE = dict(max_steps=20, atol=1e-6)
JITTED = (_solve_loop_jit, _multi_start_sequential_jit, _multi_start_parallel_jit)


def residual(x, args, p):
    return A @ x - p["b"]


def make_p(scale=1.0):
    return {"b": scale * jnp.ones(M, jnp.float32)}


@pytest.fixture(autouse=True)
def fresh_cache():
    for jitted in JITTED:
        jitted._clear_cache()
    yield
    for jitted in JITTED:
        jitted._clear_cache()


def compilations():
    return sum(jitted._cache_size() for jitted in JITTED)


@pytest.mark.parametrize("solver_cls", [LevenbergMarquardt, Newton])
def test_equal_solvers_and_traced_loop_controls_share_one_compilation(solver_cls):
    for scale in (1.0, 2.0, 3.0):
        solver_cls(residual, linear_solver=Cholesky()).solve(
            jnp.zeros(N), p=make_p(scale), **SOLVE
        )
    assert compilations() == 1

    solver = solver_cls(residual, linear_solver=Cholesky())
    solver.solve(jnp.zeros(N), p=make_p(2.0), max_steps=40, atol=1e-6, gtol=1e-9)
    solver.solve(jnp.zeros(N), p=make_p(3.0), max_steps=20, atol=1e-6, xtol=1e-9)
    assert compilations() == 1

    # Numeric constructor hyperparameters travel in state/hyper and share the
    # compiled loop across fresh solver instances (including leafless ad_solver
    # preconditioners).
    if solver_cls is LevenbergMarquardt:
        solver_cls(
            residual,
            init_damping=1e-1,
            damping_decrease=0.25,
            damping_increase=8.0,
            geodesic_acceptance_ratio=0.5,
            linear_solver=Cholesky(),
        ).solve(jnp.zeros(N), p=make_p(1.5), **SOLVE)
    else:
        solver_cls(
            residual,
            c1=1e-3,
            decrease_factor=0.5,
            maxls=15,
            max_stepsize=0.9,
            linesearch_tol=1e-12,
            geodesic_acceptance_ratio=0.5,
            linear_solver=Cholesky(),
        ).solve(jnp.zeros(N), p=make_p(1.5), **SOLVE)
    assert compilations() == 1

    # Frozen after construction
    with pytest.raises(AttributeError, match="frozen after construction"):
        solver.geodesic_acceleration = False


def test_metrics_preconditioners_and_callbacks_compile_once_and_use_live_values():
    # Two same-structure, different-valued metrics share one compiled loop and
    # use each metric's own factor values.
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
    assert not np.allclose(heavy_ends.x, heavy_middle.x, atol=1e-4)

    # Fresh BlockEigenPreconditioner and AnnealRidge callback share one compile
    for jitted in JITTED:
        jitted._clear_cache()
    for scale in (1.0, 2.0):
        precond = BlockEigenPreconditioner(
            [(scale * jnp.eye(N, dtype=jnp.float32)[None], 1.0)], jnp.arange(N)
        )
        anneal = AnnealRidge(ridge_floor=1e-8)
        LevenbergMarquardt(
            residual,
            metric=CholeskyMetric(scale * jnp.eye(N, dtype=jnp.float32)),
            ridge=1e-3,
            linear_solver=CG(precond, tol=1e-8, maxiter=16),
        ).solve(
            jnp.zeros(N),
            p=make_p(scale),
            callback=anneal,
            user_state=anneal.init_state(),
            max_steps=20,
            gtol=1e-6,
        )
    assert compilations() == 1

    # Changing a static setting compiles a distinct program
    LevenbergMarquardt(residual, linear_solver=QR()).solve(
        jnp.zeros(N), p=make_p(), **SOLVE
    )
    assert compilations() == 2


def test_carried_metric_swap_semantics_invalidation_and_implicit_ad():
    root = RNG.normal(size=(2, 4))
    K = jnp.asarray(root @ root.T + 0.5 * np.eye(2), jnp.float32)
    b_vec = jnp.ones(M, jnp.float32)

    def make_metric(scale=1.0):
        return RepeatedFactorMetric(
            scale * jnp.linalg.cholesky(K, upper=True), repeats=2
        )

    def swap_every_step(ctx):
        scale = 1.0 + 0.01 * ctx.step.astype(jnp.float32)
        return SolverAction(
            state=dataclasses.replace(ctx.state, metric=make_metric(scale))
        )

    # Ridge objective: metric swap changes the objective and suppresses convergence
    ridge_solver = LevenbergMarquardt(
        lambda x: A @ x - b_vec, metric=make_metric(), ridge=1e-3
    )
    assert int(
        ridge_solver.solve(
            jnp.zeros(N), max_steps=15, gtol=1e3, callback=swap_every_step
        ).status
    ) == int(SolverStatus.MAX_STEPS)

    # Unpenalized objective: metric swap re-whitens geometry without blocking
    unpen_solver = LevenbergMarquardt(lambda x: A @ x - b_vec, metric=make_metric())
    assert int(
        unpen_solver.solve(
            jnp.zeros(N), max_steps=15, gtol=1e3, callback=swap_every_step
        ).status
    ) == int(SolverStatus.CONVERGED)

    # Wrong structure or weak-type scalar swap raises naming state.metric
    x0 = jnp.zeros(N)
    state = ridge_solver.init(x0)
    with pytest.raises(ValueError, match="state.metric"):
        ridge_solver._apply_action(
            SolverAction(state=dataclasses.replace(state, metric=IdentityMetric(4))),
            x0,
            state,
            None,
            None,
        )
    leaves, treedef = jax.tree_util.tree_flatten(state.metric)
    doctored = jax.tree_util.tree_unflatten(
        treedef, [1.0 if jnp.ndim(leaf) == 0 else leaf for leaf in leaves]
    )
    with pytest.raises(ValueError, match="state.metric"):
        ridge_solver._apply_action(
            SolverAction(state=dataclasses.replace(state, metric=doctored)),
            x0,
            state,
            None,
            None,
        )

    # Implicit AD uses the carried metric at the solution after a mid-solve swap
    w_init = jnp.asarray(RNG.uniform(0.5, 2.0, size=N), jnp.float32)
    w_swap = jnp.asarray(RNG.uniform(4.0, 9.0, size=N), jnp.float32)

    def swap_at_2(ctx):
        fresh = jax.lax.cond(
            ctx.step == 2,
            lambda: DiagonalMetric(w_swap),
            lambda: ctx.state.metric,
        )
        return SolverAction(state=dataclasses.replace(ctx.state, metric=fresh))

    ad_solver = LevenbergMarquardt(
        lambda x, _, p: A @ x - p, metric=DiagonalMetric(w_init)
    )
    tangent = jax.jacfwd(
        lambda pv: (
            ad_solver.solve(
                jnp.zeros(N), p=pv, max_steps=60, atol=1e-6, callback=swap_at_2
            ).x
        )
    )(b_vec)
    W_inv = jnp.diag(1.0 / w_swap)
    analytic = W_inv @ A.T @ jnp.linalg.inv(A @ W_inv @ A.T)
    np.testing.assert_allclose(tangent, analytic, rtol=2e-4, atol=2e-5)


@dataclasses.dataclass(frozen=True, eq=False)
class DualJacobi(Preconditioner):
    diagonal: jax.Array

    def apply(self, v, damping, ctx):
        return v / (self.diagonal + damping)


register_pytree_dataclass(DualJacobi, data_fields=("diagonal",))


def test_preconditioner_and_hyper_schedule_in_callback():
    b_vec = jnp.ones(M, jnp.float32)
    dual_diag = jnp.diag(A @ A.T)
    solver = LevenbergMarquardt(
        lambda th: A @ th - b_vec,
        linear_solver=GramCG(DualJacobi(jnp.ones(M)), tol=0.0, maxiter=1),
    )

    def improve(ctx):
        fresh = jax.lax.cond(
            ctx.step == 2,
            lambda: DualJacobi(dual_diag),
            lambda: ctx.state.preconditioner,
        )
        grown = jnp.where(
            ctx.step >= 2,
            jnp.asarray(30, jnp.int32),
            ctx.state.hyper.iterative_maxiter,
        )
        return SolverAction(
            state=dataclasses.replace(
                ctx.state,
                preconditioner=fresh,
                hyper=dataclasses.replace(ctx.state.hyper, iterative_maxiter=grown),
            )
        )

    assert int(solver.solve(jnp.zeros(N), max_steps=30, atol=1e-5).status) == int(
        SolverStatus.MAX_STEPS
    )
    scheduled = solver.solve(jnp.zeros(N), max_steps=30, atol=1e-5, callback=improve)
    assert int(scheduled.status) == int(SolverStatus.CONVERGED)
    np.testing.assert_allclose(
        scheduled.state.preconditioner.diagonal, dual_diag, rtol=1e-6
    )
