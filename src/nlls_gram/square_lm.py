from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.flatten_util import ravel_pytree

from nlls_gram.gram_lm import LMStatus, _zero_tangent_leaf, canonicalize_residual

# Solve-only damped-Newton (Levenberg-Marquardt) root solver for SQUARE
# nonsingular systems r(x, args, p) = 0, built for hot inner loops (DAE
# Runge-Kutta stage solves): a lean jitted while_loop with a direct dense
# augmented-QR step -- never the residual-space Gram J J' -- and a custom
# implicit JVP that solves J_x xdot = -J_p pdot directly at the returned root.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SquareSolveResult:
    """Result of ``SquareLevenbergMarquardt.solve``.

    ``residual_norm`` is the residual 2-norm at the returned ``x`` -- with the
    default tolerances, ``status == LMStatus.CONVERGED`` holds exactly when it
    is below ``atol``, and a caller enforcing its own root criterion can check
    it directly. ``aux`` is the residual's aux output at the returned ``x``
    (``has_aux=True`` only).
    """

    x: Any
    residual_norm: jax.Array
    steps: jax.Array
    status: jax.Array
    aux: Any = None


class SquareLevenbergMarquardt:
    """Damped-Newton (Levenberg-Marquardt) root solver for square systems
    ``r(x, args, p) = 0`` where the flattened residual and parameter sizes
    are equal and the Jacobian is nonsingular at the root.

    The step solves the LM subproblem
    ``min_s ||r + J s||^2 + damping ||s||^2`` through one reduced QR of the
    augmented matrix ``[J; sqrt(damping) I]`` -- a direct dense factorization
    that never forms ``J J'`` or ``J'J`` (no condition-number squaring), and
    whose augmented matrix has full column rank for any finite ``J`` when
    ``damping > 0`` -- a singular Jacobian still yields a well-defined damped
    step, vetted by the accept/reject test, rather than a host exception.
    Only ``solve`` is exposed (no ``init``/``update``); the
    loop is a lean ``lax.while_loop`` that evaluates the residual once per
    iteration plus once at ``x0`` (and once more at the returned ``x`` when
    ``has_aux=True``), recomputes the Jacobian only after an accepted step,
    and exits immediately on a warm start that already meets ``atol`` --
    without ever computing a Jacobian. ``x`` may be any pytree; the residual
    must return a single array (or ``(array, aux)`` with ``has_aux=True``)
    whose flattened size equals the flattened size of ``x``, in the same
    dtype. Construct the solver once at setup scope; a new instance per call
    retraces.

    ``solve(...).x`` has a custom implicit JVP with respect to ``p``: the
    tangent solves the square system ``J_x xdot = -J_p pdot`` with a direct
    dense solve at the returned solution, and VJPs are obtained by
    transposition (``jax.custom_jvp``), so forward, reverse, and second-order
    differentiation all compose. ``x0`` and ``args`` receive zero tangents by
    contract. Implicit derivatives are meaningful only when the returned
    point is a converged, nonsingular root -- check ``status`` and
    ``residual_norm``.
    """

    def __init__(
        self,
        residual_fn,
        *,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        has_aux=False,
    ):
        canonical_residual, residual_arity = canonicalize_residual(residual_fn)
        if init_damping <= 0:
            raise ValueError("init_damping must be positive")
        if damping_decrease <= 0:
            raise ValueError("damping_decrease must be positive")
        if damping_increase <= 0:
            raise ValueError("damping_increase must be positive")
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.has_aux = has_aux

    def solve(
        self,
        x0,
        args=None,
        *,
        p=None,
        max_steps=256,
        atol=None,
        gtol=0.0,
        xtol=0.0,
        jit=True,
    ):
        """Run damped-Newton steps from ``x0`` until a stopping rule fires.

        ``atol`` stops on the residual 2-norm; ``None`` uses a dtype-aware
        default (``1e-6`` in float32, ``1e-10`` in float64). ``gtol`` stops on
        the gradient norm ``||J' r||`` and ``xtol`` on an accepted step norm;
        both default to ``0`` (disabled), so ``LMStatus.CONVERGED`` means the
        residual criterion unless they are opted into. ``x0`` is a
        root-finding guess only; a warm start already at the root returns in
        zero steps. The loop controls (``max_steps``, ``atol``, ``gtol``,
        ``xtol``) are concrete Python scalars validated eagerly -- they are
        not traceable arguments.
        """
        # Silently dropping args/p a residual never sees would, in particular,
        # make the implicit derivative with respect to p a silent zero.
        if args is not None and self.residual_arity < 2:
            raise ValueError(
                "args was passed but residual_fn takes only (x); "
                "use residual_fn(x, args)"
            )
        if p is not None and self.residual_arity < 3:
            raise ValueError(
                "p was passed but residual_fn takes no p argument; "
                "use residual_fn(x, args, p)"
            )
        if not isinstance(max_steps, int) or isinstance(max_steps, bool):
            raise ValueError("max_steps must be a positive int")
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive int")
        if atol is not None and atol < 0:
            raise ValueError("atol must be nonnegative or None")
        if gtol < 0:
            raise ValueError("gtol must be nonnegative")
        if xtol < 0:
            raise ValueError("xtol must be nonnegative")

        @jax.custom_jvp
        def solve_with_implicit_p(x, args, p, max_steps, atol, gtol, xtol):
            if jit:
                return _square_solve_loop_jit(
                    self, x, args, p, max_steps, atol, gtol, xtol
                )
            return self._solve_python(x, args, p, max_steps, atol, gtol, xtol)

        @solve_with_implicit_p.defjvp
        def solve_with_implicit_p_jvp(primals, tangents):
            x, args, p, max_steps, atol, gtol, xtol = primals
            _, _, p_dot, _, _, _, _ = tangents
            result = solve_with_implicit_p(x, args, p, max_steps, atol, gtol, xtol)
            x_dot = self._implicit_x_tangent_from_p(result.x, args, p, p_dot)
            zero_result = jax.tree.map(_zero_tangent_leaf, result)
            aux_dot = zero_result.aux
            if self.has_aux and p is not None:
                # aux depends on p directly and through the root x*(p);
                # linearize the aux map at the returned solution to account
                # for both paths.
                def aux_at_solution(x_value, p_value):
                    return self.residual_fn(x_value, args, p_value)[1]

                aux_dot = jax.jvp(aux_at_solution, (result.x, p), (x_dot, p_dot))[1]
            return (
                result,
                SquareSolveResult(
                    x_dot,
                    zero_result.residual_norm,
                    zero_result.steps,
                    zero_result.status,
                    aux_dot,
                ),
            )

        return solve_with_implicit_p(x0, args, p, max_steps, atol, gtol, xtol)

    def _implicit_x_tangent_from_p(self, x, args, p, p_dot):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            value = self.residual_fn(unravel(theta_value), args, p)
            if self.has_aux:
                value = value[0]
            return jnp.ravel(value)

        def residual_from_p(p_value):
            value = self.residual_fn(x, args, p_value)
            if self.has_aux:
                value = value[0]
            return jnp.ravel(value)

        J_x = jax.jacfwd(residual_from_theta)(theta)
        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        theta_dot = -jnp.linalg.solve(J_x, residual_p_dot)
        return unravel(theta_dot)

    def _solve_python(self, x, args, p, max_steps, atol, gtol, xtol):
        setup = _square_solve_setup(self, x, args, p, atol, gtol, xtol)
        theta, unravel, residual_flat, resid, tolerances = setup
        atol, gtol, xtol = tolerances
        n = theta.size
        dtype = resid.dtype
        damping = jnp.asarray(self.init_damping, dtype=dtype)
        loss = jnp.sum(resid**2)
        status = LMStatus.RUNNING
        steps = 0
        J = None
        if not bool(jnp.isfinite(loss)):
            status = LMStatus.NONFINITE
        elif bool((atol > 0) & (jnp.sqrt(loss) < atol)):
            status = LMStatus.CONVERGED

        while status == LMStatus.RUNNING and steps < max_steps:
            if J is None:
                J = jax.jacfwd(residual_flat)(theta)
            grad_norm = jnp.linalg.norm(J.T @ resid)
            s = _square_damped_step(J, resid, damping, n, dtype)
            resid_candidate = residual_flat(theta + s)
            loss_candidate = jnp.sum(resid_candidate**2)
            improved = bool(jnp.isfinite(loss_candidate) & (loss_candidate < loss))
            if improved:
                theta = theta + s
                resid = resid_candidate
                loss = loss_candidate
                J = None
                damping = damping * self.damping_decrease
            else:
                damping = damping * self.damping_increase
            steps += 1
            if not bool(jnp.isfinite(loss)):
                status = LMStatus.NONFINITE
            elif bool(
                ((atol > 0) & (jnp.sqrt(loss) < atol))
                | ((gtol > 0) & (grad_norm < gtol))
                | ((xtol > 0) & improved & (jnp.linalg.norm(s) < xtol))
            ):
                status = LMStatus.CONVERGED

        if status == LMStatus.RUNNING:
            status = LMStatus.MAX_STEPS
        aux = None
        if self.has_aux:
            aux = self.residual_fn(unravel(theta), args, p)[1]
        return SquareSolveResult(
            unravel(theta),
            jnp.sqrt(jnp.sum(resid**2)),
            jnp.asarray(steps, dtype=jnp.int32),
            jnp.asarray(status, dtype=jnp.int32),
            aux,
        )


def _square_solve_setup(solver, x, args, p, atol, gtol, xtol):
    theta, unravel = ravel_pytree(x)

    def residual_flat(theta_value):
        value = solver.residual_fn(unravel(theta_value), args, p)
        if solver.has_aux:
            value = value[0]
        return jnp.ravel(value)

    resid = residual_flat(theta)
    if resid.size != theta.size:
        raise ValueError(
            f"SquareLevenbergMarquardt requires a square system: the residual "
            f"has {resid.size} entries but x has {theta.size}"
        )
    dtype = resid.dtype
    if theta.dtype != dtype:
        raise ValueError(
            f"x and the residual must share a dtype for the solve loop; got "
            f"x dtype {theta.dtype} and residual dtype {dtype}"
        )
    if atol is None:
        atol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
    tolerances = (
        jnp.asarray(atol, dtype=dtype),
        jnp.asarray(gtol, dtype=dtype),
        jnp.asarray(xtol, dtype=dtype),
    )
    return theta, unravel, residual_flat, resid, tolerances


def _square_damped_step(J, resid, damping, n, dtype):
    # min_s ||r + J s||^2 + damping ||s||^2 via one reduced QR of the
    # augmented matrix: its normal equations are (J'J + damping I) s = -J'r,
    # solved without forming J'J; full column rank for any J when damping > 0.
    augmented = jnp.concatenate((J, jnp.sqrt(damping) * jnp.eye(n, dtype=dtype)))
    rhs = jnp.concatenate((-resid, jnp.zeros(n, dtype=dtype)))
    Q, R = jnp.linalg.qr(augmented, mode="reduced")
    return jsp_linalg.solve_triangular(R, Q.T @ rhs)


def _square_solve_loop_impl(solver, x, args, p, max_steps, atol, gtol, xtol):
    setup = _square_solve_setup(solver, x, args, p, atol, gtol, xtol)
    theta, unravel, residual_flat, resid, tolerances = setup
    atol, gtol, xtol = tolerances
    n = theta.size
    dtype = resid.dtype
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)
    damping = jnp.asarray(solver.init_damping, dtype=dtype)
    damping_decrease = jnp.asarray(solver.damping_decrease, dtype=dtype)
    damping_increase = jnp.asarray(solver.damping_increase, dtype=dtype)

    loss0 = jnp.sum(resid**2)
    initial_nonfinite = ~jnp.isfinite(loss0)
    initial_converged = (atol > 0) & (jnp.sqrt(loss0) < atol)
    stop = initial_nonfinite | initial_converged
    status = jnp.where(
        initial_nonfinite,
        jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
        jnp.where(
            initial_converged,
            jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
            jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
        ),
    )
    # A warm start that already meets atol never enters the body and never
    # computes a Jacobian; the zeros are the loop-carry placeholder.
    carry = (
        theta,
        damping,
        resid,
        jnp.zeros((n, n), dtype=dtype),
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        status,
        stop,
    )

    def cond(carry):
        _, _, _, _, _, step, _, stop = carry
        return (~stop) & (step < max_steps)

    def body(carry):
        theta, damping, resid, J, jacobian_stale, step, _, _ = carry
        # Rejected steps leave theta unchanged, so the Jacobian is reused
        # and only accepted steps pay the recompute.
        J = jax.lax.cond(
            jacobian_stale,
            lambda: jax.jacfwd(residual_flat)(theta),
            lambda: J,
        )
        loss = jnp.sum(resid**2)
        grad_norm = jnp.linalg.norm(J.T @ resid)
        s = _square_damped_step(J, resid, damping, n, dtype)
        resid_candidate = residual_flat(theta + s)
        loss_candidate = jnp.sum(resid_candidate**2)
        improved = jnp.isfinite(loss_candidate) & (loss_candidate < loss)
        theta = jnp.where(improved, theta + s, theta)
        resid = jnp.where(improved, resid_candidate, resid)
        loss = jnp.where(improved, loss_candidate, loss)
        damping = damping * jnp.where(improved, damping_decrease, damping_increase)
        step = step + jnp.asarray(1, dtype=jnp.int32)

        current_nonfinite = ~jnp.isfinite(loss)
        converged = (
            ((atol > 0) & (jnp.sqrt(loss) < atol))
            | ((gtol > 0) & (grad_norm < gtol))
            | ((xtol > 0) & improved & (jnp.linalg.norm(s) < xtol))
        )
        reached_max = step >= max_steps
        stop = current_nonfinite | converged | reached_max
        status = jnp.where(
            current_nonfinite,
            jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                converged,
                jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
                jnp.where(
                    reached_max,
                    jnp.asarray(LMStatus.MAX_STEPS, dtype=jnp.int32),
                    jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
                ),
            ),
        )
        return (theta, damping, resid, J, improved, step, status, stop)

    theta, _, resid, _, _, step, status, _ = jax.lax.while_loop(cond, body, carry)
    aux = None
    if solver.has_aux:
        aux = solver.residual_fn(unravel(theta), args, p)[1]
    return SquareSolveResult(
        unravel(theta),
        jnp.sqrt(jnp.sum(resid**2)),
        step,
        status,
        aux,
    )


_square_solve_loop_jit = jax.jit(_square_solve_loop_impl, static_argnums=(0,))
