# The carried-instance contract: metric/preconditioner instances ride in
# LMState, callbacks adapt them by constructing new instances, and the solver
# reacts (or deliberately does not) per its own semantics.
import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlls_gram import (
    DiagonalMetric,
    GramCG,
    IdentityMetric,
    LevenbergMarquardt,
    LMAction,
    LMStatus,
    Preconditioner,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    register_pytree_dataclass,
)

RNG = np.random.default_rng(21)
M_RESID, BLOCK, REPEATS, FREE = 4, 4, 2, 2
P_DIM = REPEATS * BLOCK + FREE
A = jnp.asarray(RNG.normal(size=(M_RESID, P_DIM)), jnp.float32)
B = jnp.asarray(RNG.normal(size=M_RESID), jnp.float32)
ROOT = RNG.normal(size=(BLOCK, BLOCK + 2))
K = jnp.asarray(ROOT @ ROOT.T + 0.5 * np.eye(BLOCK), jnp.float32)


def linear_residual(theta):
    return A @ theta - B


def make_metric(scale=1.0):
    return RepeatedFactorMetric(
        scale * jnp.linalg.cholesky(K, upper=True), repeats=REPEATS
    )


def swap_every_step(ctx):
    # A genuinely different metric every step: the scale tracks the step
    # counter, so the leaves never repeat.
    scale = 1.0 + 0.01 * ctx.step.astype(jnp.float32)
    return LMAction(
        lm_state=dataclasses.replace(ctx.lm_state, metric=make_metric(scale))
    )


def test_ridge_metric_swap_suppresses_convergence():
    # The metric defines the ridge objective: a callback that swaps it every
    # step keeps changing the problem, so a gtol that would otherwise fire
    # immediately never stops the loop.
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-3)
    swapped = solver.solve(
        jnp.zeros(P_DIM), max_steps=20, gtol=1e3, callback=swap_every_step
    )
    assert int(swapped.status) == int(LMStatus.MAX_STEPS)
    plain = solver.solve(jnp.zeros(P_DIM), max_steps=20, gtol=1e3)
    assert int(plain.status) == int(LMStatus.CONVERGED)


def test_metric_lm_metric_swap_does_not_suppress():
    # For the metric solver the metric is damping geometry only -- ||r||^2
    # did not move -- so the same every-step swap must not block convergence.
    solver = LevenbergMarquardt(linear_residual, metric=make_metric())
    swapped = solver.solve(
        jnp.zeros(P_DIM), max_steps=20, gtol=1e3, callback=swap_every_step
    )
    assert int(swapped.status) == int(LMStatus.CONVERGED)


def test_metric_swap_invalidation_semantics():
    # The reactive rules, pinned at the _apply_action seam: a metric swap
    # stales the whitening-dependent solver cache but never the Jacobian
    # cache (J = dr/dx does not see the metric); problem_changed is
    # ridge-solver-only. An untouched metric changes nothing.
    for solver_cls, expect_problem_changed in (
        (RidgeLevenbergMarquardt, True),
        (LevenbergMarquardt, False),
    ):
        kwargs = {"ridge": 1e-3} if solver_cls is RidgeLevenbergMarquardt else {}
        solver = solver_cls(linear_residual, metric=make_metric(), **kwargs)
        x0 = jnp.zeros(P_DIM)
        state = solver.init(x0)
        state = dataclasses.replace(
            state,
            jacobian_valid=jnp.asarray(True),
            solver_cache=dataclasses.replace(
                state.solver_cache, valid=jnp.asarray(True)
            ),
        )

        swap = LMAction(lm_state=dataclasses.replace(state, metric=make_metric(2.0)))
        _, _, out, _, _, problem_changed = solver._apply_action(
            swap, x0, state, None, None
        )
        assert bool(problem_changed) == expect_problem_changed
        assert bool(out.jacobian_valid)
        assert not bool(out.solver_cache.valid)

        untouched = LMAction(lm_state=dataclasses.replace(state, damping=state.damping))
        _, _, out, _, _, problem_changed = solver._apply_action(
            untouched, x0, state, None, None
        )
        assert not bool(problem_changed)
        assert bool(out.jacobian_valid)
        assert bool(out.solver_cache.valid)


def test_wrong_structure_swap_raises_naming_the_field():
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-3)
    x0 = jnp.zeros(P_DIM)
    state = solver.init(x0)
    wrong_type = LMAction(
        lm_state=dataclasses.replace(state, metric=IdentityMetric(REPEATS * BLOCK))
    )
    with pytest.raises(ValueError, match="lm_state.metric"):
        solver._apply_action(wrong_type, x0, state, None, None)
    wrong_shape = LMAction(
        lm_state=dataclasses.replace(
            state,
            metric=RepeatedFactorMetric(jnp.eye(2, dtype=jnp.float32), repeats=REPEATS),
        )
    )
    with pytest.raises(ValueError, match="lm_state.metric"):
        solver._apply_action(wrong_shape, x0, state, None, None)

    def swap_in_loop(ctx):
        return LMAction(
            lm_state=dataclasses.replace(
                ctx.lm_state, metric=IdentityMetric(REPEATS * BLOCK)
            )
        )

    with pytest.raises(ValueError, match="lm_state.metric"):
        solver.solve(x0, max_steps=5, callback=swap_in_loop)


def test_untouched_metric_costs_no_comparison_ops():
    # A callback that anneals the ridge via dataclasses.replace hands the
    # metric subtree back as the same tracers: the identity short-circuit
    # must emit NO comparison over the (BLOCK, BLOCK) factor leaf. The ridge
    # change detection still emits its scalar eq.
    solver = RidgeLevenbergMarquardt(linear_residual, metric=make_metric(), ridge=1e-3)
    x0 = jnp.zeros(P_DIM)
    state = solver.init(x0)

    def apply_anneal(state_in, x_in):
        action = LMAction(
            lm_state=dataclasses.replace(state_in, ridge=state_in.ridge * 0.5)
        )
        return solver._apply_action(action, x_in, state_in, None, None)[5]

    text = str(jax.make_jaxpr(apply_anneal)(state, x0))
    assert f"bool[{BLOCK},{BLOCK}]" not in text
    assert "eq" in text  # the scalar ridge compare is still there


@dataclass(frozen=True, eq=False)
class DualJacobi(Preconditioner):
    """Residual-space (equation-structure) preconditioner for GramCG: the
    inverse of diag(J J') + damping."""

    diagonal: jax.Array

    def apply(self, v, damping, ctx):
        return v / (self.diagonal + damping)


register_pytree_dataclass(DualJacobi, data_fields=("diagonal",))


def test_gram_cg_equation_space_preconditioner_and_budget_schedule():
    # The dual (residual-space) Krylov form takes the same Preconditioner
    # interface with v living in equation space. A starved inner budget
    # cannot reach gtol; a callback that BOTH refreshes the carried
    # preconditioner from live data and grows hyper.iterative_maxiter on a
    # schedule converges -- the budget is traced carry, the instance rides in
    # lm_state.
    def residual(theta):
        return A @ theta - B

    dual_diag = jnp.diag(A @ A.T)

    def build(callback=None, maxiter=1):
        solver = LevenbergMarquardt(
            residual,
            linear_solver=GramCG(
                DualJacobi(jnp.ones(M_RESID)), tol=0.0, maxiter=maxiter
            ),
        )
        return solver.solve(
            jnp.zeros(P_DIM), max_steps=40, atol=1e-5, callback=callback
        )

    def improve(ctx):
        fresh = jax.lax.cond(
            ctx.step == 3,
            lambda: DualJacobi(dual_diag),
            lambda: ctx.lm_state.preconditioner,
        )
        grown = jnp.where(
            ctx.step >= 3,
            jnp.asarray(30, jnp.int32),
            ctx.lm_state.hyper.iterative_maxiter,
        )
        return LMAction(
            lm_state=dataclasses.replace(
                ctx.lm_state,
                preconditioner=fresh,
                hyper=dataclasses.replace(ctx.lm_state.hyper, iterative_maxiter=grown),
            )
        )

    starved = build()
    scheduled = build(improve)
    assert int(starved.status) == int(LMStatus.MAX_STEPS)
    assert int(scheduled.status) == int(LMStatus.CONVERGED)
    np.testing.assert_allclose(
        scheduled.lm_state.preconditioner.diagonal, dual_diag, rtol=1e-6
    )


def test_ad_uses_the_carried_metric_at_the_solution():
    # The implicit tangent freezes the CARRIED instances: after a mid-solve
    # metric swap, the minimum-W-norm tangent selection must use the swapped
    # metric, not the construction-time one. Analytic reference for the
    # underdetermined linear system r = A x - p from x0 = 0:
    # dx/dp = W^{-1} A' (A W^{-1} A')^{-1}.
    n = 6
    A_fat = jnp.asarray(RNG.normal(size=(3, n)), jnp.float32)
    w_initial = jnp.asarray(RNG.uniform(0.5, 2.0, size=n), jnp.float32)
    w_swapped = jnp.asarray(RNG.uniform(4.0, 9.0, size=n), jnp.float32)

    def residual(x, args, p):
        return A_fat @ x - p

    def swap(ctx):
        fresh = jax.lax.cond(
            ctx.step == 2,
            lambda: DiagonalMetric(w_swapped),
            lambda: ctx.lm_state.metric,
        )
        return LMAction(lm_state=dataclasses.replace(ctx.lm_state, metric=fresh))

    solver = LevenbergMarquardt(residual, metric=DiagonalMetric(w_initial))
    p = jnp.asarray(RNG.normal(size=3), jnp.float32)

    def solved(p_value):
        return solver.solve(
            jnp.zeros(n), p=p_value, max_steps=60, atol=1e-6, callback=swap
        ).x

    assert int(
        solver.solve(jnp.zeros(n), p=p, max_steps=60, atol=1e-6, callback=swap).status
    ) == int(LMStatus.CONVERGED)
    tangent = jax.jacfwd(solved)(p)

    def analytic(weights):
        W_inv = jnp.diag(1.0 / weights)
        return W_inv @ A_fat.T @ jnp.linalg.inv(A_fat @ W_inv @ A_fat.T)

    np.testing.assert_allclose(tangent, analytic(w_swapped), rtol=2e-4, atol=2e-5)
    assert not np.allclose(tangent, analytic(w_initial), atol=1e-3)
