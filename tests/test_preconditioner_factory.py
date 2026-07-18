import dataclasses
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    LMStatus,
    MultiStart,
    PreconditionerFactory,
    RecycleConfig,
    identity_preconditioner,
)


def residual_fn(x, args, p):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


# A residual whose Jacobian rotates with x: with orthogonal A the dual operator
# J J' = D(x)^2 is exactly diagonal (D = diag(exp(a_i . x))), so the exact-current
# diagonal is a perfect inverse while the same diagonal frozen at x0 mismatches once
# the row scales drift. The canonical setting the factory is built for.
def rotating_problem(n=12):
    A, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (n, n)))
    x_true = 1.2 * jax.random.normal(jax.random.key(1), (n,))
    target = jnp.exp(A @ x_true)

    def residual(x):
        return jnp.exp(A @ x) - target

    def prepare(x, args, p):
        d = jnp.exp(A @ x)
        return d * d  # exact diag(J J') at the current x

    def apply(state, v, damping):
        return v / (state + damping)

    return residual, jnp.zeros(n), prepare, apply


# --- validation --------------------------------------------------------------


def test_factory_requires_cg():
    factory = PreconditionerFactory(lambda *a: jnp.zeros(()), lambda *a: a[1])
    with pytest.raises(ValueError, match="preconditioner_factory requires"):
        LevenbergMarquardt(
            lambda x: x, linear_solver="cholesky", preconditioner_factory=factory
        )


def test_factory_and_dual_preconditioner_mutually_exclusive():
    factory = PreconditionerFactory(lambda *a: jnp.zeros(()), lambda *a: a[1])
    with pytest.raises(ValueError, match="exactly one"):
        LevenbergMarquardt(
            lambda x: x,
            linear_solver="cg",
            dual_preconditioner=identity_preconditioner(),
            implicit_preconditioner=identity_preconditioner(),
            preconditioner_factory=factory,
        )


def test_factory_satisfies_cg_and_implicit_requirements():
    # With neither dual_preconditioner nor implicit_preconditioner, a factory alone
    # satisfies both the forward-cg and cg-implicit preconditioner requirements.
    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_maxiter=4,
    )
    assert solver.preconditioner_factory is not None


def test_factory_hashing_shares_compilation():
    # Equal (prepare, apply) identities -> equal solver static key -> shared jit
    # cache; a different apply is a different compiled program.
    residual, x0, prepare, apply = rotating_problem()
    other_apply = lambda state, v, damping: v / (state + damping)  # noqa: E731
    common = dict(linear_solver="cg", iterative_maxiter=4)
    a = LevenbergMarquardt(
        residual, preconditioner_factory=PreconditionerFactory(prepare, apply), **common
    )
    b = LevenbergMarquardt(
        residual, preconditioner_factory=PreconditionerFactory(prepare, apply), **common
    )
    c = LevenbergMarquardt(
        residual,
        preconditioner_factory=PreconditionerFactory(prepare, other_apply),
        **common,
    )
    assert a == b and hash(a) == hash(b)
    assert a != c


# --- init / state threading --------------------------------------------------


def test_init_builds_precond_state():
    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_maxiter=4,
    )
    state = solver.init(x0)
    assert state.precond is not None
    assert bool(state.precond_valid)
    assert jnp.array_equal(state.precond, prepare(x0, None, None))


def test_update_without_init_state_raises():
    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_maxiter=4,
    )
    from nlls_gram import LMState

    with pytest.raises(ValueError, match="no preconditioner state"):
        solver.update(x0, LMState(jnp.asarray(1e-3)))


# --- equivalence: prepare ignoring x reproduces the frozen preconditioner ----


def test_factory_matches_frozen_when_theta_independent():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    x = {"a": 1.0, "b": 0.0}
    weights = 1.0 + jnp.arange(20, dtype=jnp.float32) / 10.0

    def prepare(x, args, p):
        return weights  # ignores x -> a constant preconditioner

    def apply(state, v, damping):
        return v / (state + damping)

    def frozen(v, damping):
        return v / (weights + damping)

    common = dict(
        init_damping=1e-2,
        linear_solver="cg",
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-7,
        iterative_maxiter=40,
    )
    frozen_solver = LevenbergMarquardt(
        residual_fn, dual_preconditioner=frozen, **common
    )
    factory_solver = LevenbergMarquardt(
        residual_fn,
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        **common,
    )

    # Identical arithmetic graph -> bitwise identical step and full solve.
    xf, _, info_f = frozen_solver.update(x, frozen_solver.init(x, (ts, ys)), (ts, ys))
    xa, _, info_a = factory_solver.update(x, factory_solver.init(x, (ts, ys)), (ts, ys))
    assert jnp.array_equal(xf["a"], xa["a"])
    assert jnp.array_equal(xf["b"], xa["b"])
    assert jnp.array_equal(info_f.loss, info_a.loss)

    rf = frozen_solver.solve(x, (ts, ys), max_steps=30, atol=1e-6)
    ra = factory_solver.solve(x, (ts, ys), max_steps=30, atol=1e-6)
    assert jnp.array_equal(rf.x["a"], ra.x["a"])
    assert jnp.array_equal(rf.x["b"], ra.x["b"])


# --- the measured need: frozen stalls, factory converges ---------------------


def test_factory_converges_where_frozen_stalls():
    residual, x0, prepare, apply = rotating_problem()
    diag0 = prepare(x0, None, None)

    def frozen(v, damping):
        return v / (diag0 + damping)  # exact diagonal, frozen at x0

    common = dict(
        init_damping=1e-3,
        linear_solver="cg",
        geodesic_acceleration=False,
        implicit_preconditioner=identity_preconditioner(),
        iterative_maxiter=2,
        iterative_tol=1e-12,
    )
    frozen_solver = LevenbergMarquardt(residual, dual_preconditioner=frozen, **common)
    factory_solver = LevenbergMarquardt(
        residual, preconditioner_factory=PreconditionerFactory(prepare, apply), **common
    )

    frozen_result = frozen_solver.solve(x0, max_steps=80, atol=1e-5)
    factory_result = factory_solver.solve(x0, max_steps=80, atol=1e-5)

    assert int(factory_result.status) == LMStatus.CONVERGED
    assert float(factory_result.info.loss) < 1e-6
    assert int(frozen_result.status) != LMStatus.CONVERGED
    assert float(frozen_result.info.loss) > 1e-4


# --- rejected-step reuse vs accepted-step rebuild ----------------------------


def test_rejected_step_reuses_carried_precond_state():
    # precond_valid gates reuse-vs-rebuild. Injecting a deliberately wrong (but
    # SPD) carried state distinguishes the branches through the threaded-out state:
    # valid=True carries the wrong state through (reuse, no rebuild), valid=False
    # replaces it with the freshly built prepare(x) (rebuild).
    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-3,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_maxiter=4,
    )
    state = solver.init(x0)
    correct = prepare(x0, None, None)
    wrong = correct * 1000.0 + 1.0

    reuse_in = dataclasses.replace(
        state, precond=wrong, precond_valid=jnp.asarray(True)
    )
    _, reuse_out, reuse_info = solver.update(x0, reuse_in)
    assert jnp.array_equal(reuse_out.precond, wrong)  # reused, not rebuilt
    # precond_valid tracks acceptance: True iff the step was rejected (x fixed).
    assert bool(reuse_out.precond_valid) == (not bool(reuse_info.accepted))

    rebuild_in = dataclasses.replace(
        state, precond=wrong, precond_valid=jnp.asarray(False)
    )
    _, rebuild_out, _ = solver.update(x0, rebuild_in)
    assert jnp.array_equal(rebuild_out.precond, correct)  # rebuilt at x0
    assert not jnp.array_equal(rebuild_out.precond, wrong)


def test_update_jits():
    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-3,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_maxiter=4,
    )

    @jax.jit
    def step(x, lm_state):
        return solver.update(x, lm_state)

    x, lm_state, info = step(x0, solver.init(x0))
    assert jnp.isfinite(info.loss)
    assert lm_state.precond.shape == (12,)


# --- composition with recycle ------------------------------------------------


def test_factory_composes_with_recycle():
    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-3,
        linear_solver="cg",
        geodesic_acceleration=False,
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        recycle=RecycleConfig(rank=3),
        iterative_maxiter=4,
        iterative_tol=1e-10,
    )
    result = solver.solve(x0, max_steps=80, atol=1e-6)
    assert int(result.status) == LMStatus.CONVERGED
    assert float(result.info.loss) < 1e-6
    assert result.lm_state.precond is not None
    assert result.lm_state.recycle is not None


# --- multi_start -------------------------------------------------------------


def test_factory_multi_start_vmap():
    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-3,
        linear_solver="cg",
        geodesic_acceleration=False,
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_maxiter=4,
        iterative_tol=1e-10,
    )

    def draw(key, x, args):
        return x + 0.01 * jax.random.normal(key, x.shape), args

    for parallel in (False, True):
        ms = MultiStart(
            key=jax.random.key(0), num_starts=3, draw=draw, parallel=parallel
        )
        result = solver.solve(x0, max_steps=80, atol=1e-6, multi_start=ms)
        assert int(result.status) == LMStatus.CONVERGED
        assert float(result.info.loss) < 1e-6


def test_multi_start_cold_resets_precond():
    # _cold_lm_state must invalidate the carried preconditioner state so a drawn
    # start rebuilds prepare() at its own x rather than reusing another x's state.
    from nlls_gram import gram_lm

    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_maxiter=4,
    )
    state = solver.init(x0)
    _, warmed, _ = solver.update(x0, state)
    cold = gram_lm._cold_lm_state(warmed)
    assert not bool(cold.precond_valid)
    assert jnp.all(cold.precond == 0)


# --- differentiation ---------------------------------------------------------


def test_factory_update_reverse_ad_matches_cholesky():
    # update()'s factory path stays reverse-differentiable and matches the dense
    # cholesky reference: the preconditioner state is stop_gradient'd, so only the
    # (converged) step carries gradient. Differentiate w.r.t. the target data.
    ts = jnp.linspace(0.0, 2.0, 20)

    def residual_data(x, args, p):
        return x["a"] * jnp.exp(x["b"] * ts) - args

    def prepare(x, args, p):
        e = jnp.exp(x["b"] * ts)
        return e**2 + (x["a"] * ts * e) ** 2  # exact diag(J J') at x

    def apply(state, v, damping):
        return v / (state + damping)

    x = {"a": 1.0, "b": 0.0}
    cholesky = LevenbergMarquardt(residual_data, init_damping=1e-2)
    factory = LevenbergMarquardt(
        residual_data,
        init_damping=1e-2,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_tol=1e-9,
        iterative_maxiter=60,
    )

    def loss_of(solver):
        def loss(ys):
            new_x, _, _ = solver.update(x, solver.init(x, ys), ys)
            return jnp.sum(new_x["a"] ** 2 + new_x["b"] ** 2)

        return loss

    ys = 2.0 * jnp.exp(-1.0 * ts)
    g_factory = jax.grad(loss_of(factory))(ys)
    g_cholesky = jax.grad(loss_of(cholesky))(ys)
    assert jnp.allclose(g_factory, g_cholesky, rtol=1e-3, atol=1e-4)


# A full-rank underdetermined-style dual so the implicit derivative is
# preconditioner-independent: residual exp(A x) - p with orthogonal A gives
# J J' = D(x)^2 (full rank, all positive), the regime the package targets.
def target_problem(n=12):
    A, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (n, n)))

    def residual_p(x, args, p):
        return jnp.exp(A @ x) - p

    def prepare(x, args, p):
        d = jnp.exp(A @ x)
        return d * d

    def apply(state, v, damping):
        return v / (state + damping)

    p = jnp.exp(0.3 * jax.random.normal(jax.random.key(2), (n,)))
    return residual_p, jnp.zeros(n), prepare, apply, p


def test_factory_implicit_p_derivative_matches_cholesky():
    # solve()'s p-derivative comes from the implicit rule at the converged root.
    # With a factory and no explicit implicit_preconditioner, the implicit cg rule
    # seeds its preconditioner from prepare(result.x) at the solution; on a
    # full-rank dual the derivative is preconditioner-independent, so it must match
    # both the dense cholesky rule and an explicit identity implicit_preconditioner.
    residual_p, x0, prepare, apply, p = target_problem()
    common = dict(init_damping=1e-3, geodesic_acceleration=False)
    cholesky = LevenbergMarquardt(residual_p, **common)
    factory = LevenbergMarquardt(
        residual_p,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_tol=1e-12,
        iterative_maxiter=40,
        **common,
    )
    explicit = LevenbergMarquardt(
        residual_p,
        linear_solver="cg",
        dual_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-12,
        iterative_maxiter=40,
        **common,
    )

    def solved(solver, q):
        return solver.solve(x0, p=q, max_steps=80, atol=1e-8).x

    j_cholesky = jax.jacobian(lambda q: solved(cholesky, q))(p)
    j_factory = jax.jacobian(lambda q: solved(factory, q))(p)
    j_explicit = jax.jacobian(lambda q: solved(explicit, q))(p)
    assert jnp.allclose(j_factory, j_cholesky, rtol=1e-3, atol=1e-4)
    assert jnp.allclose(j_factory, j_explicit, rtol=1e-3, atol=1e-4)


def test_factory_implicit_p_derivative_no_recompile_across_p():
    # The implicit rule builds prepare(result.x) from the TRACED solution, so
    # differentiating at different p values shares one compilation instead of
    # baking a closure constant. Cache-size counting is fragile under full-suite
    # cache pressure (global eviction), so assert the key-stability property
    # directly: the traced jaxpr is identical across p values, i.e. p enters as
    # an input, never as an embedded constant.
    n = 8
    A, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (n, n)))
    p = jnp.exp(0.3 * jax.random.normal(jax.random.key(2), (n,)))

    def residual_p(x, args, q):
        return jnp.exp(A @ x) - q

    def prepare(x, args, q):
        d = jnp.exp(A @ x)
        return d * d

    def apply(state, v, damping):
        return v / (state + damping)

    solver = LevenbergMarquardt(
        residual_p,
        init_damping=1e-3,
        geodesic_acceleration=False,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_tol=1e-12,
        iterative_maxiter=30,
    )

    def jac_fn(q):
        return jax.jacobian(
            lambda qq: solver.solve(jnp.zeros(n), p=qq, max_steps=60, atol=1e-8).x
        )(q)

    jac = jax.jit(jac_fn)
    j1 = jac(p)
    j1.block_until_ready()
    j2 = jac(p * 1.3)
    j2.block_until_ready()
    assert bool(jnp.all(jnp.isfinite(j1))) and bool(jnp.all(jnp.isfinite(j2)))
    jaxpr_a = str(jax.make_jaxpr(jac_fn)(p))
    jaxpr_b = str(jax.make_jaxpr(jac_fn)(p * 1.3))
    assert jaxpr_a == jaxpr_b  # p is a traced input, not a baked constant


# --- callback action must preserve the precond fields ------------------------


def test_callback_dropping_precond_raises():
    residual, x0, prepare, apply = rotating_problem()
    solver = LevenbergMarquardt(
        residual,
        linear_solver="cg",
        preconditioner_factory=PreconditionerFactory(prepare, apply),
        iterative_maxiter=4,
    )
    from nlls_gram import LMSolveAction, LMState

    def callback(ctx):
        # A bare LMState lacks the precond fields -> loud failure.
        return LMSolveAction(lm_state=LMState(ctx.lm_state.damping))

    with pytest.raises(ValueError, match="without the preconditioner state"):
        solver.solve(x0, max_steps=3, callback=callback, jit=False)


# --- x64 in a clean subprocess -----------------------------------------------


def test_factory_float64_subprocess():
    script = textwrap.dedent(
        """
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from nlls_gram import LevenbergMarquardt, PreconditionerFactory, LMStatus

        n = 10
        A, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (n, n)))
        x_true = 1.2 * jax.random.normal(jax.random.key(1), (n,))
        target = jnp.exp(A @ x_true)

        def residual(x):
            return jnp.exp(A @ x) - target

        def prepare(x, args, p):
            d = jnp.exp(A @ x)
            return d * d

        def apply(state, v, damping):
            return v / (state + damping)

        x0 = jnp.zeros(n, dtype=jnp.float64)
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-3,
            linear_solver="cg",
            geodesic_acceleration=False,
            preconditioner_factory=PreconditionerFactory(prepare, apply),
            iterative_maxiter=2,
            iterative_tol=1e-14,
        )
        state = solver.init(x0)
        assert state.precond.dtype == jnp.float64
        result = solver.solve(x0, max_steps=80, atol=1e-10)
        assert int(result.status) == LMStatus.CONVERGED
        assert result.x.dtype == jnp.float64
        assert float(result.info.loss) < 1e-12
        jaxpr = str(jax.make_jaxpr(lambda x, s: solver.update(x, s))(x0, state))
        assert "f32" not in jaxpr, jaxpr
        print("OK")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout
