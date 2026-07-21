# Issue #23: jacobian_mode ("auto"/"fwd"/"rev") dense Jacobian assembly.

import jax
import jax.numpy as jnp
import pytest

from nlls_gram import LevenbergMarquardt, LMStatus, identity_preconditioner

TALL_M, TALL_N = 10, 3
FAT_M, FAT_N = 3, 10


def _linear_problem(m, n, seed):
    key_a, key_b = jax.random.split(jax.random.key(seed))
    A = jax.random.normal(key_a, (m, n)) + 0.1
    b = jax.random.normal(key_b, (m,))
    return A, b


def _pinv_solution(A, b):
    return jnp.linalg.pinv(A) @ b


# "auto" resolves to normal_cholesky on the tall shape and gram_cholesky on
# the fat one, so the shape parametrize covers both dense cholesky forms.
DENSE_CONFIGS = [
    ("qr", False),
    ("augmented_qr", False),
    ("auto", True),
    ("auto", False),
]


@pytest.mark.parametrize("shape", [(TALL_M, TALL_N), (FAT_M, FAT_N)])
@pytest.mark.parametrize("linear_solver,cache_jacobian", DENSE_CONFIGS)
def test_fwd_rev_auto_agree_and_match_closed_form(shape, linear_solver, cache_jacobian):
    m, n = shape
    A, b = _linear_problem(m, n, seed=7)
    expected = _pinv_solution(A, b)

    def residual(x, args, p):
        return A @ x - b

    def build(mode):
        return LevenbergMarquardt(
            residual,
            linear_solver=linear_solver,
            jacobian_mode=mode,
            cache_jacobian=cache_jacobian,
            geodesic_acceleration=False,
        )

    # The core of issue #23: fwd and rev assemble the SAME Jacobian two ways,
    # so a single deterministic update from x0 must agree tightly. This is the
    # platform-robust invariant -- unlike a full float32 LM solve, whose
    # accept/reject trajectory near the solution is rounding-sensitive and can
    # stop at slightly different points on different CPUs.
    steps = {}
    for mode in ("auto", "fwd", "rev"):
        solver = build(mode)
        x_next, _, _ = solver.update(jnp.zeros(n), solver.init(jnp.zeros(n)))
        steps[mode] = x_next
    assert jnp.allclose(steps["fwd"], steps["rev"], atol=1e-5)
    assert jnp.allclose(steps["auto"], steps["fwd"], atol=1e-5)
    assert jnp.allclose(steps["auto"], steps["rev"], atol=1e-5)

    # End to end each mode reaches the minimum-norm (pseudoinverse) solution;
    # from x0 = 0 every dense step stays in range(A'). The tolerance is loose
    # because a float32 LM solve stops at a platform-dependent point near the
    # solution -- the tight invariant is the per-update agreement above.
    for mode in ("auto", "fwd", "rev"):
        result = build(mode).solve(jnp.zeros(n), max_steps=200, gtol=1e-6)
        assert jnp.allclose(result.x, expected, atol=1e-3), (
            f"{linear_solver} mode={mode} missed the closed form"
        )


def _collect_shapes_from_jaxpr(jaxpr, shapes):
    for eqn in jaxpr.eqns:
        for var in list(eqn.invars) + list(eqn.outvars):
            aval = getattr(var, "aval", None)
            if aval is not None and hasattr(aval, "shape"):
                shapes.add(tuple(aval.shape))
        for value in eqn.params.values():
            _collect_shapes_from_param(value, shapes)


def _collect_shapes_from_param(value, shapes):
    if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        _collect_shapes_from_jaxpr(value.jaxpr, shapes)
    elif hasattr(value, "eqns"):
        _collect_shapes_from_jaxpr(value, shapes)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_shapes_from_param(item, shapes)


def _update_shapes(jacobian_mode, m, n):
    A, b = _linear_problem(m, n, seed=11)

    def residual(x, args, p):
        return A @ x - b

    solver = LevenbergMarquardt(
        residual,
        linear_solver="augmented_qr",
        jacobian_mode=jacobian_mode,
        cache_jacobian=False,
        geodesic_acceleration=False,
    )
    x0 = jnp.zeros(n)
    lm_state = solver.init(x0)
    closed = jax.make_jaxpr(lambda x, s: solver.update(x, s))(x0, lm_state)
    shapes = set()
    _collect_shapes_from_jaxpr(closed.jaxpr, shapes)
    return shapes


def test_auto_never_materializes_m_by_m_for_tall_systems():
    m, n = 13, 3
    for mode in ("auto", "fwd"):
        shapes = _update_shapes(mode, m, n)
        assert (m, m) not in shapes, (
            f"jacobian_mode={mode} materialized an (m, m) array on a tall system"
        )
    # Sanity check that the structural probe can see the blowup at all: the
    # forced reverse mode vmaps over an m x m residual identity basis.
    rev_shapes = _update_shapes("rev", m, n)
    assert (m, m) in rev_shapes


def test_auto_resolution_breaks_square_tie_to_fwd():
    # At n == m the pass counts are equal, so the tie goes to the cheaper
    # forward-mode JVP columns; reverse rows are kept only for strictly fat
    # systems (a shape probe cannot distinguish the modes at n == m, so the
    # resolution itself is asserted).
    solver = LevenbergMarquardt(lambda x, args, p: x - p)
    assert solver._resolve_jacobian_mode(5, 5) == "fwd"
    assert solver._resolve_jacobian_mode(6, 5) == "fwd"
    assert solver._resolve_jacobian_mode(5, 6) == "rev"


def test_has_aux_under_fwd_mode():
    A, b = _linear_problem(TALL_M, TALL_N, seed=3)

    def residual(x, args, p):
        r = A @ x - b
        return r, {"first_param": x[0], "sum_sq": jnp.sum(r**2)}

    solver = LevenbergMarquardt(
        residual,
        jacobian_mode="fwd",
        has_aux=True,
        geodesic_acceleration=False,
    )
    result = solver.solve(jnp.zeros(TALL_N), max_steps=100, gtol=1e-6)

    expected_x = _pinv_solution(A, b)
    expected_r = A @ result.x - b
    assert jnp.allclose(result.x, expected_x, atol=1e-4)
    # result.aux is the aux evaluated at the returned solution.
    assert jnp.allclose(result.aux["first_param"], result.x[0], atol=1e-6)
    assert jnp.allclose(result.aux["sum_sq"], jnp.sum(expected_r**2), atol=1e-4)
    # The per-step info aux has the same pytree structure.
    assert set(result.info.aux.keys()) == {"first_param", "sum_sq"}


def test_unknown_jacobian_mode_raises():
    with pytest.raises(ValueError, match="jacobian_mode"):
        LevenbergMarquardt(lambda x: x, jacobian_mode="bogus")


def test_fwd_mode_with_cg_solver_raises():
    with pytest.raises(ValueError, match="jacobian_mode"):
        LevenbergMarquardt(
            lambda x: x,
            linear_solver="gram_cg",
            jacobian_mode="fwd",
            dual_preconditioner=identity_preconditioner(),
            ad_solver_preconditioner=identity_preconditioner(),
        )


def test_fwd_mode_with_lsmr_and_dense_implicit_is_accepted():
    # The lsmr forward path is matrix-free, but the dense gram_cholesky
    # implicit rule consumes the jacobian_mode setting, so this is valid.
    solver = LevenbergMarquardt(
        lambda x: x,
        linear_solver="lsmr",
        jacobian_mode="fwd",
        ad_solver="dense",
    )
    assert solver.jacobian_mode == "fwd"


def test_fwd_mode_with_lsmr_and_dense_ad_is_accepted():
    # ad_solver="auto" under lsmr resolves to the dense AD rule --
    # jacobian_mode has a consumer.
    solver = LevenbergMarquardt(
        lambda x: x,
        linear_solver="lsmr",
        jacobian_mode="fwd",
    )
    assert solver.jacobian_mode == "fwd"


def test_geodesic_acceleration_under_fwd_mode():
    a_true, b_true = 2.0, -1.0
    ts = jnp.linspace(0.0, 2.0, 12)
    ys = a_true * jnp.exp(b_true * ts)

    def residual(x, args, p):
        return x["a"] * jnp.exp(x["b"] * ts) - ys

    # Default solver settings: auto (normal_cholesky on this tall problem,
    # m=12 > n=2), cache_jacobian=True, geodesic_acceleration=True, forced fwd.
    solver = LevenbergMarquardt(residual, jacobian_mode="fwd")
    assert solver.geodesic_acceleration

    result = solver.solve({"a": 1.0, "b": 0.0}, max_steps=100, atol=1e-5)

    assert int(result.status) == LMStatus.CONVERGED
    assert jnp.allclose(result.x["a"], a_true, atol=1e-4)
    assert jnp.allclose(result.x["b"], b_true, atol=1e-4)
