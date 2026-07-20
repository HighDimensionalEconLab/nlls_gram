import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import pytest

from nlls_gram import (
    LevenbergMarquardt,
    PreconditionerFactory,
    RecycleConfig,
    WhitenedPreconditioner,
    blockdiag_metric,
    identity_preconditioner,
    matern_state_space,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_from_quasiseparable,
    metric_from_state_space,
    metric_from_tridiagonal_precision,
    repeated_blockdiag_metric,
)

# Rank-2 tall interpolation fixture: row 3 duplicates row 1 AND column 3 is
# column 1 + column 2, so the problem is rank-deficient in both directions;
# b = A @ theta_true keeps it consistent (a zero-residual root exists).
A_RD = jnp.array(
    [
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 3.0],
    ]
)
THETA_TRUE = jnp.array([1.0, -1.0, 2.0])
B_RD = A_RD @ THETA_TRUE

A_TALL = jnp.array([[1.0, 0.5], [0.3, 2.0], [-1.0, 1.0]])
A_SQ = jnp.array([[2.0, 0.5], [-0.4, 1.5]])
A_WIDE = jnp.array([[1.0, 2.0, -0.5, 0.3], [0.2, -1.0, 1.5, 2.0]])

L2 = jnp.array([[1.3, 0.0], [0.5, 0.8]])
L3 = jnp.array([[1.5, 0.0, 0.0], [0.4, 1.2, 0.0], [-0.3, 0.2, 0.9]])
L4 = jnp.array(
    [
        [1.2, 0.0, 0.0, 0.0],
        [0.3, 1.5, 0.0, 0.0],
        [-0.2, 0.4, 0.9, 0.0],
        [0.1, -0.1, 0.2, 1.4],
    ]
)


def chol_S(L):
    # metric_from_cholesky(L) has S = L^{-T} with S S' = (L L')^{-1}.
    return jsp_linalg.solve_triangular(L.T, jnp.eye(L.shape[0]), lower=False)


def min_m_norm_root(A, b, S=None):
    if S is None:
        return jnp.linalg.pinv(A) @ b
    return S @ jnp.linalg.pinv(A @ S) @ b


def normal_cg_kwargs(maxiter=100, tol=1e-7):
    return dict(
        linear_solver="normal_cg",
        normal_preconditioner=identity_preconditioner(),
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=tol,
        iterative_maxiter=maxiter,
    )


def implicit_kwargs(implicit_solver):
    if implicit_solver in ("gram_cg", "normal_cg"):
        return {
            "implicit_preconditioner": identity_preconditioner(),
            "implicit_maxiter": 50,
        }
    return {}


IMPLICIT_FORMS = ["gram_cholesky", "normal_cholesky", "gram_cg", "normal_cg"]


# --- normal_cholesky closed-form steps ---------------------------------------


@pytest.mark.parametrize("A", [A_SQ, A_TALL], ids=["square", "tall"])
@pytest.mark.parametrize("use_metric", [False, True], ids=["identity", "cholesky"])
def test_normal_cholesky_step_matches_closed_form(A, use_metric):
    lam = 1e-2
    b = jnp.arange(1.0, A.shape[0] + 1.0)

    def residual(theta, args, p):
        return A @ theta - b

    solver = LevenbergMarquardt(
        residual,
        init_damping=lam,
        linear_solver="normal_cholesky",
        metric=metric_from_cholesky(L2) if use_metric else None,
        geodesic_acceleration=False,
    )
    theta0 = jnp.array([0.4, -0.3])
    theta1, _, info = solver.update(theta0, solver.init(theta0))

    S = chol_S(L2) if use_metric else jnp.eye(2)
    B = A @ S
    r0 = A @ theta0 - b
    u = jnp.linalg.solve(B.T @ B + lam * jnp.eye(2), -(B.T @ r0))
    assert bool(info.accepted)
    assert jnp.allclose(theta1, theta0 + S @ u, atol=1e-5)


# --- gram <-> normal push-through identity at lambda > 0 ---------------------


@pytest.mark.parametrize(
    "A", [A_RD, A_RD.T], ids=["tall_rank_deficient", "wide_rank_deficient"]
)
@pytest.mark.parametrize("use_metric", [False, True], ids=["identity", "cholesky"])
def test_gram_and_normal_steps_agree_at_positive_damping(A, use_metric):
    # Push-through identity: P J'(J P J' + lam I)^{-1} = S (B'B + lam I)^{-1} B'
    # with B = J S, exact for every lam > 0 regardless of rank or shape -- this
    # covers the gram form on a TALL residual and the normal form on a WIDE one.
    m, n = A.shape
    b = jnp.arange(1.0, m + 1.0)
    L = {3: L3, 4: L4}[n]

    def residual(theta, args, p):
        return A @ theta - b

    metric = metric_from_cholesky(L) if use_metric else None
    common = dict(init_damping=1e-2, metric=metric, geodesic_acceleration=False)
    gram = LevenbergMarquardt(residual, linear_solver="gram_cholesky", **common)
    normal = LevenbergMarquardt(residual, linear_solver="normal_cholesky", **common)

    theta0 = 0.1 * jnp.arange(1.0, n + 1.0)
    x_gram, _, info_gram = gram.update(theta0, gram.init(theta0))
    x_normal, _, info_normal = normal.update(theta0, normal.init(theta0))

    assert bool(info_gram.accepted) == bool(info_normal.accepted)
    # rtol 5e-4: the identity is exact (verified ~1e-13 at float64) but the two
    # factorizations round differently in float32 at this rank-deficient
    # conditioning, with measured rel diff ~1.2e-4 on the tall cases.
    assert jnp.allclose(x_gram, x_normal, rtol=5e-4, atol=1e-5)
    assert jnp.allclose(info_gram.loss, info_normal.loss, rtol=5e-4, atol=1e-6)


# --- normal_cg == normal_cholesky --------------------------------------------


def exp_residual(theta, args, p):
    ts, ys = args
    return theta[0] * jnp.exp(theta[1] * ts) - ys


@pytest.mark.parametrize("preconditioned", [False, True], ids=["identity", "exact"])
def test_normal_cg_step_matches_normal_cholesky(preconditioned):
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    theta0 = jnp.array([1.0, 0.0])
    common = dict(
        init_damping=1e-2,
        metric=metric_from_cholesky(L2),
        geodesic_acceleration=False,
    )
    dense = LevenbergMarquardt(exp_residual, linear_solver="normal_cholesky", **common)

    if preconditioned:
        # Exact n-space preconditioner (B'B + lam I)^{-1} frozen at theta0: SPD,
        # linear, and inert at inner convergence -- the step must not move.
        J0 = jax.jacobian(lambda th: exp_residual(th, (ts, ys), None))(theta0)
        B0 = J0 @ chol_S(L2)

        def normal_preconditioner(v, damping):
            return jnp.linalg.solve(B0.T @ B0 + damping * jnp.eye(2), v)

    else:
        normal_preconditioner = identity_preconditioner()

    cg = LevenbergMarquardt(
        exp_residual,
        linear_solver="normal_cg",
        normal_preconditioner=normal_preconditioner,
        implicit_preconditioner=identity_preconditioner(),
        iterative_tol=1e-8,
        iterative_maxiter=100,
        **common,
    )

    x_dense, _, info_dense = dense.update(
        theta0, dense.init(theta0, (ts, ys)), (ts, ys)
    )
    x_cg, _, info_cg = cg.update(theta0, cg.init(theta0, (ts, ys)), (ts, ys))

    assert bool(info_dense.accepted)
    assert bool(info_cg.accepted)
    assert jnp.allclose(x_cg, x_dense, rtol=1e-4, atol=1e-4)


# --- minimum-M-norm root selection -------------------------------------------


@pytest.mark.parametrize("form", ["normal_cholesky", "normal_cg"])
@pytest.mark.parametrize("use_metric", [False, True], ids=["identity", "cholesky"])
def test_normal_forms_converge_to_min_m_norm_root(form, use_metric):
    def residual(theta, args, p):
        return A_RD @ theta - B_RD

    metric = metric_from_cholesky(L3) if use_metric else None
    if form == "normal_cg":
        kwargs = normal_cg_kwargs(maxiter=100, tol=1e-7)
    else:
        kwargs = {"linear_solver": "normal_cholesky"}
    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-3,
        metric=metric,
        geodesic_acceleration=False,
        **kwargs,
    )
    result = solver.solve(jnp.zeros(3), atol=1e-5, max_steps=60)

    expected = min_m_norm_root(A_RD, B_RD, chol_S(L3) if use_metric else None)
    assert float(result.info.loss) < 1e-5
    assert jnp.allclose(result.x, expected, atol=5e-3)


def test_lambda_zero_selection_matches_weighted_pseudoinverse_float64():
    # One update at damping -> 0 from theta0 = 0 must select the M-weighted
    # pseudoinverse root S pinv(A S) b for every form -- including LSMR under a
    # right-preconditioner, which the damping fix makes exactly I-damped in u.
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from nlls_gram import (
    LevenbergMarquardt,
    WhitenedPreconditioner,
    identity_preconditioner,
    metric_from_cholesky,
)

A = jnp.array(
    [[1.0, 2.0, 3.0], [0.0, 1.0, 1.0], [1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]
)
b = A @ jnp.array([1.0, -1.0, 2.0])
L = jnp.array([[1.5, 0.0, 0.0], [0.4, 1.2, 0.0], [-0.3, 0.2, 0.9]])
S = jsp_linalg.solve_triangular(L.T, jnp.eye(3), lower=False)
expected = S @ jnp.linalg.pinv(A @ S) @ b


def residual(theta, args, p):
    return A @ theta - b


lam = 1e-10
theta0 = jnp.zeros(3)
common = dict(
    init_damping=lam, metric=metric_from_cholesky(L), geodesic_acceleration=False
)

for form in ("gram_cholesky", "normal_cholesky"):
    solver = LevenbergMarquardt(residual, linear_solver=form, **common)
    theta1, _, info = solver.update(theta0, solver.init(theta0))
    assert bool(info.accepted), form
    assert jnp.allclose(theta1, expected, atol=1e-7), (form, theta1, expected)

gram_cg = LevenbergMarquardt(
    residual,
    linear_solver="gram_cg",
    dual_preconditioner=identity_preconditioner(),
    implicit_preconditioner=identity_preconditioner(),
    iterative_tol=1e-12,
    iterative_maxiter=100,
    **common,
)
theta1, _, _ = gram_cg.update(theta0, gram_cg.init(theta0))
assert jnp.allclose(theta1, expected, atol=1e-6), ("gram_cg", theta1, expected)

normal_cg = LevenbergMarquardt(
    residual,
    linear_solver="normal_cg",
    normal_preconditioner=identity_preconditioner(),
    implicit_preconditioner=identity_preconditioner(),
    iterative_tol=1e-12,
    iterative_maxiter=100,
    **common,
)
theta1, _, _ = normal_cg.update(theta0, normal_cg.init(theta0))
assert jnp.allclose(theta1, expected, atol=1e-6), ("normal_cg", theta1, expected)

R = jnp.linalg.cholesky(A.T @ A + jnp.eye(3)).T
lsmr_common = dict(iterative_tol=1e-14, iterative_atol=0.0, iterative_maxiter=400)
plain = LevenbergMarquardt(residual, linear_solver="lsmr", **lsmr_common, **common)
preconditioned = LevenbergMarquardt(
    residual,
    linear_solver="lsmr",
    whitened_preconditioner=WhitenedPreconditioner(
        lambda v, damping: jsp_linalg.solve_triangular(R, v, lower=False),
        lambda w, damping: jsp_linalg.solve_triangular(R.T, w, lower=True),
    ),
    **lsmr_common,
    **common,
)
for name, solver in (("lsmr", plain), ("lsmr_preconditioned", preconditioned)):
    theta1, _, _ = solver.update(theta0, solver.init(theta0))
    assert jnp.allclose(theta1, expected, atol=1e-6), (name, theta1, expected)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


# --- auto resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    "A,explicit",
    [
        (A_TALL, "normal_cholesky"),
        (A_SQ, "normal_cholesky"),
        (A_WIDE, "gram_cholesky"),
    ],
    ids=["tall", "square", "wide"],
)
def test_auto_matches_explicit_form_per_shape(A, explicit):
    # auto resolves at trace time to gram_cholesky iff n_params > n_residuals,
    # else normal_cholesky -- the SAME branch, so the update is bitwise equal.
    m, n = A.shape
    b = jnp.arange(1.0, m + 1.0)

    def residual(theta, args, p):
        return A @ theta - b

    auto = LevenbergMarquardt(residual, init_damping=1e-2)
    explicit_solver = LevenbergMarquardt(
        residual, init_damping=1e-2, linear_solver=explicit
    )
    theta0 = 0.1 * jnp.arange(1.0, n + 1.0)
    x_auto, state_auto, info_auto = auto.update(theta0, auto.init(theta0))
    x_expl, state_expl, info_expl = explicit_solver.update(
        theta0, explicit_solver.init(theta0)
    )

    assert jnp.array_equal(x_auto, x_expl)
    assert jnp.array_equal(state_auto.damping, state_expl.damping)
    assert jnp.array_equal(info_auto.loss, info_expl.loss)


def test_auto_reuses_trace_within_shape_and_recompiles_on_shape_change():
    traces = {"count": 0}

    def residual(theta, args, p):
        traces["count"] += 1
        A, b = args
        return A @ theta - b

    solver = LevenbergMarquardt(
        residual, init_damping=1e-2, geodesic_acceleration=False
    )
    step = jax.jit(lambda x, s, args: solver.update(x, s, args))

    tall = (A_TALL, jnp.array([1.0, -0.5, 2.0]))
    theta = jnp.zeros(2)
    state = solver.init(theta, tall)
    step(theta, state, tall)
    count_after_trace = traces["count"]
    assert count_after_trace > 0

    # Same shapes, new values: the auto resolution is a trace-time constant, so
    # the compiled update is reused with no retrace.
    step(theta + 0.1, state, (A_TALL, jnp.array([0.3, 0.1, -1.0])))
    assert traces["count"] == count_after_trace

    # Shape flip (wide problem): a fresh trace resolves the other branch.
    wide = (A_WIDE, jnp.array([0.7, -0.2]))
    theta_wide = jnp.zeros(4)
    state_wide = solver.init(theta_wide, wide)
    count_before_wide_trace = traces["count"]
    step(theta_wide, state_wide, wide)
    assert traces["count"] > count_before_wide_trace


# --- constructor validation matrix -------------------------------------------


def test_old_solver_names_are_unknown():
    with pytest.raises(ValueError, match="unknown linear_solver"):
        LevenbergMarquardt(exp_residual, linear_solver="cholesky")
    with pytest.raises(ValueError, match="unknown linear_solver"):
        LevenbergMarquardt(exp_residual, linear_solver="cg")
    with pytest.raises(ValueError, match="unknown implicit_solver"):
        LevenbergMarquardt(exp_residual, implicit_solver="cholesky")
    with pytest.raises(ValueError, match="unknown implicit_solver"):
        LevenbergMarquardt(exp_residual, implicit_solver="cg")
    with pytest.raises(TypeError, match="dual_solve_dtype"):
        LevenbergMarquardt(exp_residual, dual_solve_dtype=jnp.float64)


def test_gram_cg_only_hooks_rejected_elsewhere():
    for kwargs in (
        {"dual_preconditioner": identity_preconditioner()},
        {
            "preconditioner_factory": PreconditionerFactory(
                prepare=lambda x, args, p, aux: jnp.ones(2),
                apply=lambda state, v, damping: v,
            )
        },
        {"recycle": RecycleConfig(rank=2)},
    ):
        (name,) = kwargs
        with pytest.raises(ValueError, match=name):
            LevenbergMarquardt(exp_residual, linear_solver="normal_cholesky", **kwargs)
        with pytest.raises(ValueError, match=name):
            LevenbergMarquardt(
                exp_residual,
                **normal_cg_kwargs(),
                **kwargs,
            )


def test_normal_preconditioner_required_by_and_exclusive_to_normal_cg():
    with pytest.raises(ValueError, match="normal_preconditioner"):
        LevenbergMarquardt(
            exp_residual,
            linear_solver="normal_cg",
            iterative_tol=1e-7,
            iterative_maxiter=30,
        )
    with pytest.raises(ValueError, match="normal_preconditioner"):
        LevenbergMarquardt(
            exp_residual,
            linear_solver="gram_cg",
            dual_preconditioner=identity_preconditioner(),
            implicit_preconditioner=identity_preconditioner(),
            iterative_maxiter=30,
            normal_preconditioner=identity_preconditioner(),
        )
    with pytest.raises(ValueError, match="normal_preconditioner"):
        LevenbergMarquardt(
            exp_residual,
            linear_solver="normal_cholesky",
            normal_preconditioner=identity_preconditioner(),
        )


def test_whitened_preconditioner_still_lsmr_only():
    hook = WhitenedPreconditioner(lambda v, damping: v, lambda w, damping: w)
    with pytest.raises(ValueError, match="whitened_preconditioner"):
        LevenbergMarquardt(
            exp_residual, linear_solver="normal_cholesky", whitened_preconditioner=hook
        )


def test_gram_cg_still_requires_dual_preconditioner():
    with pytest.raises(ValueError, match="dual_preconditioner"):
        LevenbergMarquardt(
            exp_residual,
            linear_solver="gram_cg",
            implicit_preconditioner=identity_preconditioner(),
            iterative_maxiter=30,
        )


def test_implicit_preconditioner_requires_cg_resolved_implicit():
    with pytest.raises(ValueError, match="implicit_preconditioner"):
        LevenbergMarquardt(
            exp_residual,
            linear_solver="normal_cholesky",
            implicit_preconditioner=identity_preconditioner(),
        )
    with pytest.raises(ValueError, match="implicit_preconditioner"):
        LevenbergMarquardt(
            exp_residual,
            linear_solver="normal_cholesky",
            implicit_solver="gram_cholesky",
            implicit_preconditioner=identity_preconditioner(),
        )


def test_dtype_knobs_require_x64_and_float64():
    # This suite runs with x64 disabled, so a legal target still trips the
    # x64 gate; non-float64 values are rejected outright.
    with pytest.raises(ValueError, match="x64"):
        LevenbergMarquardt(exp_residual, linear_solve_dtype=jnp.float64)
    with pytest.raises(ValueError, match="x64"):
        LevenbergMarquardt(
            exp_residual,
            metric=metric_from_cholesky(L2),
            metric_solve_dtype=jnp.float64,
        )
    with pytest.raises(ValueError, match="float64"):
        LevenbergMarquardt(exp_residual, linear_solve_dtype=jnp.float32)
    with pytest.raises(ValueError, match="float64"):
        LevenbergMarquardt(
            exp_residual,
            metric=metric_from_cholesky(L2),
            metric_solve_dtype=jnp.float32,
        )


def test_dtype_knob_legality_and_wiring_float64_subprocess():
    script = r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nlls_gram import (
    LevenbergMarquardt,
    MetricFactory,
    Metric,
    identity_preconditioner,
    metric_from_cholesky,
)

L = jnp.array([[1.3, 0.0], [0.5, 0.8]], dtype=jnp.float32)


def residual(theta, args, p):
    A, b = args
    return A @ theta - b


# Fully matrix-free pipeline (lsmr forward + normal_cg implicit): no dense
# linear-solve path exists for linear_solve_dtype to promote.
try:
    LevenbergMarquardt(
        residual,
        linear_solver="lsmr",
        implicit_solver="normal_cg",
        implicit_preconditioner=identity_preconditioner(),
        implicit_maxiter=30,
        iterative_maxiter=30,
        linear_solve_dtype=jnp.float64,
    )
except ValueError as error:
    assert "linear_solve_dtype" in str(error), error
else:
    raise AssertionError("matrix-free linear_solve_dtype must be rejected")

# metric_solve_dtype without a custom metric or factory has nothing to wrap.
try:
    LevenbergMarquardt(residual, metric_solve_dtype=jnp.float64)
except ValueError as error:
    assert "metric_solve_dtype" in str(error), error
else:
    raise AssertionError("metric_solve_dtype without a metric must be rejected")

# lsmr forward with implicit auto resolves densely by shape, a legal target.
LevenbergMarquardt(
    residual, linear_solver="lsmr", iterative_maxiter=30,
    linear_solve_dtype=jnp.float64,
)

A32 = jnp.array([[1.0, 0.5], [0.3, 2.0], [-1.0, 1.0]], dtype=jnp.float32)
b32 = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
theta0 = jnp.zeros(2, dtype=jnp.float32)

# linear_solve_dtype: float32 data in, float32 step out, wide solve inside.
promoted = LevenbergMarquardt(
    residual,
    linear_solver="normal_cholesky",
    linear_solve_dtype=jnp.float64,
    geodesic_acceleration=False,
)
theta1, state1, info1 = promoted.update(
    theta0, promoted.init(theta0, (A32, b32)), (A32, b32)
)
assert theta1.dtype == jnp.float32, theta1.dtype
assert bool(jnp.all(jnp.isfinite(theta1)))

# metric_solve_dtype wraps the resolved metric callbacks: they see float64
# inputs while the returned step stays at the residual dtype.
seen = {}
base = metric_from_cholesky(L)


def recording_inv_sqrt(v):
    seen["inv_sqrt"] = v.dtype
    return base.inv_sqrt(v)


def recording_inv_sqrt_transpose(v):
    seen["inv_sqrt_transpose"] = v.dtype
    return base.inv_sqrt_transpose(v)


recording = Metric(
    solve=base.solve,
    norm=base.norm,
    inv_sqrt=recording_inv_sqrt,
    inv_sqrt_transpose=recording_inv_sqrt_transpose,
)
wrapped = LevenbergMarquardt(
    residual,
    linear_solver="normal_cholesky",
    metric=recording,
    metric_solve_dtype=jnp.float64,
    geodesic_acceleration=False,
)
theta1, _, _ = wrapped.update(theta0, wrapped.init(theta0, (A32, b32)), (A32, b32))
assert theta1.dtype == jnp.float32, theta1.dtype
assert seen["inv_sqrt"] == jnp.float64, seen
assert seen["inv_sqrt_transpose"] == jnp.float64, seen

# A factory-built metric is wrapped the same way, after build.
factory = MetricFactory(
    prepare=lambda x, args, p, aux: jnp.zeros(()),
    build=lambda state: recording,
)
seen.clear()
factory_solver = LevenbergMarquardt(
    residual,
    linear_solver="normal_cholesky",
    metric_factory=factory,
    metric_solve_dtype=jnp.float64,
    geodesic_acceleration=False,
)
theta1, _, _ = factory_solver.update(
    theta0, factory_solver.init(theta0, (A32, b32)), (A32, b32)
)
assert seen["inv_sqrt"] == jnp.float64, seen
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


# --- S / S' adjoint consistency for every shipped builder --------------------


def shipped_inv_sqrt_metrics():
    w2 = jnp.array([2.0, 0.5])
    tridiag = metric_from_tridiagonal_precision(
        jnp.array([2.0, 2.5, 2.2, 1.8]), jnp.array([-0.7, 0.4, -0.5])
    )
    qs = metric_from_quasiseparable(
        2.0 * jnp.ones(4),
        0.5 * jnp.ones((4, 1)),
        0.5 * jnp.ones((4, 1)),
        0.9 * jnp.ones((4, 1, 1)),
    )
    state_space = metric_from_state_space(
        jnp.array([0.0, 0.3, 0.8, 1.1, 1.7]), *matern_state_space(1.2, 0.7, 1.5)
    )
    return [
        ("cholesky", metric_from_cholesky(L3), 3),
        ("diagonal", metric_from_diagonal(jnp.array([2.0, 0.5, 1.5])), 3),
        ("tridiagonal", tridiag, 4),
        ("quasiseparable", qs, 4),
        ("state_space", state_space, 5),
        (
            "blockdiag",
            blockdiag_metric(
                [(metric_from_cholesky(L2), 2), (metric_from_diagonal(w2), 2)]
            ),
            4,
        ),
        (
            "repeated_blockdiag",
            repeated_blockdiag_metric(metric_from_cholesky(L2), 2, 3),
            6,
        ),
    ]


@pytest.mark.parametrize(
    "metric,n",
    [case[1:] for case in shipped_inv_sqrt_metrics()],
    ids=[case[0] for case in shipped_inv_sqrt_metrics()],
)
def test_metric_builders_inv_sqrt_adjoint_consistency(metric, n):
    # The normal forms apply S and S' as a transpose PAIR: <S u, w> = <u, S' w>
    # and S S' = M^{-1} must hold for every shipped builder, or the normal-form
    # steps and the implicit S-transpose rule silently use a wrong adjoint.
    # (metric_from_shifted_matvec ships no inv_sqrt, so it has no pair to test.)
    u = jax.random.normal(jax.random.key(0), (n,))
    w = jax.random.normal(jax.random.key(1), (n,))
    assert jnp.allclose(
        metric.inv_sqrt(u) @ w, u @ metric.inv_sqrt_transpose(w), rtol=1e-4, atol=1e-5
    )
    assert jnp.allclose(
        metric.inv_sqrt(metric.inv_sqrt_transpose(w)),
        metric.solve(w),
        rtol=1e-4,
        atol=1e-5,
    )


# --- descent on an inconsistent residual -------------------------------------


@pytest.mark.parametrize("form", ["normal_cholesky", "normal_cg"])
def test_normal_forms_descend_on_inconsistent_residual(form):
    # Tall full-rank with b outside range(A): no root exists, the forward r is
    # never consistent, and the damped normal step must still be a descent
    # direction converging to the least-squares solution.
    b = jnp.array([1.0, -2.0, 0.5])

    def residual(theta, args, p):
        return A_TALL @ theta - b

    if form == "normal_cg":
        kwargs = normal_cg_kwargs(maxiter=100, tol=1e-7)
    else:
        kwargs = {"linear_solver": form}
    solver = LevenbergMarquardt(
        residual, init_damping=1e-2, geodesic_acceleration=False, **kwargs
    )
    theta = jnp.array([2.0, 1.0])
    state = solver.init(theta)
    _, _, first = solver.update(theta, state)
    assert bool(first.accepted)
    assert float(first.loss) < float(first.loss_old)

    for _ in range(40):
        theta, state, info = solver.update(theta, state)
    expected = jnp.linalg.lstsq(A_TALL, b)[0]
    assert jnp.allclose(theta, expected, atol=1e-3)


# --- range-violating preconditioner loses min-norm selection -----------------


def test_range_violating_normal_preconditioner_loses_min_norm_selection():
    # Pins DOCUMENTED behavior (the Codex counterexample): a budget-truncated
    # normal_cg iterate lies in the C-preconditioned Krylov space, so an SPD C
    # with C(range(B')) not within range(B') leaks nullspace components into
    # the step and minimum-M-norm selection is lost. The identity keeps every
    # iterate in range(B'). maxiter=2 < n=3 keeps CG genuinely truncated.
    def residual(theta, args, p):
        return A_RD @ theta - B_RD

    weights = jnp.array([1.0, 1.0, 25.0])
    range_projector = A_RD.T @ jnp.linalg.pinv(A_RD.T)
    steps = {}
    for name, hook in (
        ("identity", identity_preconditioner()),
        ("violating", lambda v, damping: v / weights),
    ):
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-3,
            linear_solver="normal_cg",
            normal_preconditioner=hook,
            implicit_preconditioner=identity_preconditioner(),
            iterative_tol=0.0,
            iterative_atol=0.0,
            iterative_maxiter=2,
            geodesic_acceleration=False,
        )
        theta0 = jnp.zeros(3)
        theta1, _, info = solver.update(theta0, solver.init(theta0))
        assert bool(info.accepted)
        steps[name] = theta1 - theta0

    out_of_range_identity = steps["identity"] - range_projector @ steps["identity"]
    out_of_range_violating = steps["violating"] - range_projector @ steps["violating"]
    assert float(jnp.linalg.norm(out_of_range_identity)) < 1e-4
    assert float(jnp.linalg.norm(out_of_range_violating)) > 1e-2


# --- LSMR damping fix --------------------------------------------------------


def test_lsmr_preconditioned_step_matches_plain_after_damping_fix():
    # The augmented operator damps sqrt(damping) * R^{-1} z = sqrt(damping) u,
    # so every lambda > 0 subproblem is exactly I-damped in u and the converged
    # step is R-invariant; a large lambda makes any leftover R'R-damping
    # surrogate visible immediately.
    def residual(theta, args, p):
        return A_RD @ theta - B_RD

    R = jnp.linalg.cholesky(A_RD.T @ A_RD + jnp.eye(3)).T
    hook = WhitenedPreconditioner(
        lambda v, damping: jsp_linalg.solve_triangular(R, v, lower=False),
        lambda w, damping: jsp_linalg.solve_triangular(R.T, w, lower=True),
    )
    common = dict(
        init_damping=0.1,
        geodesic_acceleration=False,
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=200,
    )
    plain = LevenbergMarquardt(residual, linear_solver="lsmr", **common)
    preconditioned = LevenbergMarquardt(
        residual, linear_solver="lsmr", whitened_preconditioner=hook, **common
    )
    theta0 = jnp.array([0.2, -0.1, 0.4])
    x_plain, _, _ = plain.update(theta0, plain.init(theta0))
    x_prec, _, _ = preconditioned.update(theta0, preconditioned.init(theta0))
    assert jnp.allclose(x_prec, x_plain, rtol=1e-4, atol=1e-4)


# --- reverse-mode grad through a normal_cg update ----------------------------


def test_reverse_grad_through_normal_cg_update_matches_normal_cholesky():
    # Non-diagonal metric: S = L^{-T} is NOT self-adjoint, so any code path
    # that transposes S as itself produces a wrong reverse-mode gradient here.
    w = jnp.array([0.3, -0.7, 1.1])

    def residual(theta, args, p):
        return A_RD @ theta - p * B_RD

    common = dict(
        init_damping=1e-2,
        metric=metric_from_cholesky(L3),
        geodesic_acceleration=False,
    )
    dense = LevenbergMarquardt(residual, linear_solver="normal_cholesky", **common)
    cg = LevenbergMarquardt(
        residual, **normal_cg_kwargs(maxiter=100, tol=1e-8), **common
    )

    def stepped(solver, p):
        theta0 = jnp.array([0.2, -0.1, 0.3])
        theta1, _, _ = solver.update(theta0, solver.init(theta0, p=p), None, p)
        return theta1 @ w

    p = jnp.asarray(1.3)
    grad_dense = jax.grad(lambda q: stepped(dense, q))(p)
    grad_cg = jax.grad(lambda q: stepped(cg, q))(p)
    assert jnp.allclose(grad_cg, grad_dense, rtol=1e-3, atol=1e-4)


# --- geodesic parity ---------------------------------------------------------


def test_geodesic_parity_across_forms():
    ts = jnp.linspace(0.0, 2.0, 20)
    ys = 2.0 * jnp.exp(-1.0 * ts)
    theta0 = jnp.array([1.0, 0.0])
    common = dict(init_damping=1e-2, metric=metric_from_cholesky(L2))
    gram = LevenbergMarquardt(exp_residual, linear_solver="gram_cholesky", **common)
    normal = LevenbergMarquardt(exp_residual, linear_solver="normal_cholesky", **common)
    cg = LevenbergMarquardt(
        exp_residual, **normal_cg_kwargs(maxiter=100, tol=1e-8), **common
    )

    results = {}
    for name, solver in (("gram", gram), ("normal", normal), ("normal_cg", cg)):
        x1, _, info = solver.update(theta0, solver.init(theta0, (ts, ys)), (ts, ys))
        results[name] = (x1, info)

    x_ref, info_ref = results["gram"]
    for name in ("normal", "normal_cg"):
        x1, info = results[name]
        assert bool(info.used_geodesic) == bool(info_ref.used_geodesic)
        assert jnp.allclose(x1, x_ref, rtol=1e-4, atol=1e-4)
        assert jnp.allclose(
            info.acceleration_ratio, info_ref.acceleration_ratio, rtol=1e-3, atol=1e-4
        )


# --- implicit AD: four forms x three regimes ---------------------------------


@pytest.mark.parametrize("implicit_solver", IMPLICIT_FORMS)
def test_implicit_forms_underdetermined_min_norm(implicit_solver):
    def residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        implicit_solver=implicit_solver,
        **implicit_kwargs(implicit_solver),
    )
    theta0 = jnp.zeros(2)

    def solved_x(p):
        return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).x

    p = jnp.asarray(3.0)
    p_dot = jnp.asarray(0.7)
    x, x_dot = jax.jvp(solved_x, (p,), (p_dot,))
    _, pullback = jax.vjp(solved_x, p)
    (p_bar,) = pullback(jnp.array([3.0, 4.0]))

    assert jnp.allclose(x, jnp.array([3.0 / 5.0, 6.0 / 5.0]), atol=1e-5)
    assert jnp.allclose(x_dot, jnp.array([p_dot / 5.0, 2.0 * p_dot / 5.0]), atol=1e-5)
    assert jnp.allclose(p_bar, (3.0 + 2.0 * 4.0) / 5.0, atol=1e-5)


@pytest.mark.parametrize("implicit_solver", IMPLICIT_FORMS)
def test_implicit_forms_square_full_rank_jvp_and_vjp(implicit_solver):
    # Square nonsingular Jacobian: the ridge -> 0 sensitivity is exactly
    # -J^{-1} J_p p_dot = A^{-1} c p_dot.
    c = jnp.array([1.0, -0.7])

    def residual(theta, _, p):
        return A_SQ @ theta - c * p

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-2,
        implicit_solver=implicit_solver,
        **implicit_kwargs(implicit_solver),
    )
    theta0 = jnp.zeros(2)

    def solved_x(p):
        return solver.solve(theta0, p=p, max_steps=80, atol=1e-7).x

    p = jnp.asarray(1.5)
    p_dot = jnp.asarray(0.6)
    dx_dp = jnp.linalg.solve(A_SQ, c)
    x, x_dot = jax.jvp(solved_x, (p,), (p_dot,))
    _, pullback = jax.vjp(solved_x, p)
    w = jnp.array([0.4, -0.2])
    (p_bar,) = pullback(w)

    assert jnp.allclose(x, dx_dp * p, atol=1e-4)
    assert jnp.allclose(x_dot, dx_dp * p_dot, atol=1e-4)
    assert jnp.allclose(p_bar, w @ dx_dp, atol=1e-4)


@pytest.mark.parametrize("implicit_solver", IMPLICIT_FORMS)
def test_implicit_forms_tall_rank_deficient_consistent_min_norm_tangent(
    implicit_solver,
):
    # Interpolation keeps the system consistent for every p (r_p p_dot lies in
    # range(B) by construction), so all four forms return the min-norm tangent
    # root * p_dot.
    def residual(theta, _, p):
        return A_RD @ theta - p * B_RD

    solver = LevenbergMarquardt(
        residual,
        init_damping=1e-3,
        implicit_solver=implicit_solver,
        geodesic_acceleration=False,
        **implicit_kwargs(implicit_solver),
    )
    theta0 = jnp.zeros(3)

    def solved_x(p):
        return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).x

    root = min_m_norm_root(A_RD, B_RD)
    p = jnp.asarray(1.2)
    p_dot = jnp.asarray(0.5)
    x, x_dot = jax.jvp(solved_x, (p,), (p_dot,))

    assert jnp.allclose(x, p * root, atol=2e-3)
    assert jnp.allclose(x_dot, p_dot * root, atol=2e-3)


def test_implicit_normal_cg_reverse_grad_nondiagonal_metric_matches_closed_form():
    # The implicit rule's final step applies S through a custom_linear_solve;
    # with a non-diagonal metric S is not self-adjoint, so reverse mode is only
    # correct if that solve declares its true transpose S'.
    a = jnp.array([1.0, 2.0, 0.5])
    w = jnp.array([0.4, -0.2, 0.7])

    def residual(theta, _, p):
        return jnp.array([a @ theta - p])

    def make_solver(implicit_solver):
        return LevenbergMarquardt(
            residual,
            init_damping=1e-2,
            metric=metric_from_cholesky(L3),
            geodesic_acceleration=False,
            implicit_solver=implicit_solver,
            **implicit_kwargs(implicit_solver),
        )

    def solved_dot_w(solver, p):
        return solver.solve(jnp.zeros(3), p=p, max_steps=80, atol=1e-6).x @ w

    P = jnp.linalg.inv(L3 @ L3.T)
    direction = P @ a / (a @ P @ a)
    p = jnp.asarray(2.0)
    grad_dense = jax.grad(lambda q: solved_dot_w(make_solver("normal_cholesky"), q))(p)
    grad_cg = jax.grad(lambda q: solved_dot_w(make_solver("normal_cg"), q))(p)

    assert jnp.allclose(grad_dense, w @ direction, atol=1e-4)
    assert jnp.allclose(grad_cg, w @ direction, atol=1e-4)


def test_implicit_shape_auto_resolves_by_solution_shape():
    # A matrix-free forward (lsmr) defers the implicit form to trace time:
    # eval_shape picks gram on the wide problem and normal on the tall one, and
    # both must produce the min-norm tangent.
    def wide_residual(theta, _, p):
        return jnp.array([theta[0] + 2.0 * theta[1] - p])

    def tall_residual(theta, _, p):
        return A_RD @ theta - p * B_RD

    def make_solver(residual):
        return LevenbergMarquardt(
            residual,
            init_damping=1e-3,
            linear_solver="lsmr",
            iterative_tol=1e-10,
            iterative_maxiter=100,
            geodesic_acceleration=False,
        )

    wide = make_solver(wide_residual)
    _, x_dot = jax.jvp(
        lambda p: wide.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).x,
        (jnp.asarray(3.0),),
        (jnp.asarray(1.0),),
    )
    assert jnp.allclose(x_dot, jnp.array([1.0 / 5.0, 2.0 / 5.0]), atol=1e-3)

    tall = make_solver(tall_residual)
    root = min_m_norm_root(A_RD, B_RD)
    _, x_dot = jax.jvp(
        lambda p: tall.solve(jnp.zeros(3), p=p, max_steps=80, atol=1e-6).x,
        (jnp.asarray(1.2),),
        (jnp.asarray(1.0),),
    )
    assert jnp.allclose(x_dot, root, atol=2e-3)


def test_implicit_penalty_zero_fails_loudly_on_rank_deficient_normal_cholesky():
    # implicit_penalty trio on the singular undamped N = B'B: the default None
    # takes the spectral-filter pseudoinverse (exact min-norm tangent), an
    # explicit positive value factors the trace-scaled ridge (finite, O(penalty)
    # bias), and 0.0 is the unridged solve whose rank guard poisons the tangent
    # to NaN rather than returning a quiet pseudo-solution.
    def residual(theta, _, p):
        return A_RD @ theta - p * B_RD

    def tangent(implicit_penalty):
        solver = LevenbergMarquardt(
            residual,
            init_damping=1e-3,
            implicit_solver="normal_cholesky",
            implicit_penalty=implicit_penalty,
            geodesic_acceleration=False,
        )
        return jax.jvp(
            lambda p: solver.solve(jnp.zeros(3), p=p, max_steps=80, atol=1e-6).x,
            (jnp.asarray(1.2),),
            (jnp.asarray(0.5),),
        )[1]

    expected = 0.5 * min_m_norm_root(A_RD, B_RD)
    default_tangent = tangent(None)
    assert bool(jnp.all(jnp.isfinite(default_tangent)))
    assert jnp.allclose(default_tangent, expected, atol=2e-3)
    ridged_tangent = tangent(1e-4)
    assert bool(jnp.all(jnp.isfinite(ridged_tangent)))
    assert jnp.allclose(ridged_tangent, expected, atol=5e-3)
    assert not bool(jnp.all(jnp.isfinite(tangent(0.0))))
