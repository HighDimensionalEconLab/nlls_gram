"""Typed linear-solver configs, and the contract they implement.

A config selects the algebra for the LM subproblem (``linear_solver``) or the
implicit-AD solve (``ad_solver``) and carries that method's own knobs as
fields, so an option that exists for only one method cannot be passed with
another. Configs compare and hash by value, so equal configs key the same
compiled solve loop; construct them inline freely.

Each config owns two things the solver used to branch on by name:

- ``new_cache(sub_shapes)`` builds its reject-step cache, so ``init`` never
  asks "which solver am I".
- ``prepare(sub)`` receives a :class:`Subproblem` and returns a
  :class:`StepSolver` -- the gradient, a velocity solve, a general solve for
  the geodesic correction, and a cache constructor. Dispatch happens once at
  trace time, so the compiled program is identical to a hand-written branch.

The menu:

- :class:`Cholesky` (the default): dense normal equations.
- :class:`QR`: MINPACK-structured damping-row QR, stable at tiny
  ridge/damping where forming a normal matrix squares the condition number.
- :class:`CG`: matrix-free preconditioned CG on the normal operator.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg

from nlls_gram.preconditioners import Preconditioner

__all__ = [
    "CG",
    "QR",
    "Cholesky",
    "CholeskyCache",
    "QRCache",
    "StepSolver",
    "Subproblem",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CholeskyCache:
    """Per-``(x, ridge)`` cache carried by the :class:`Cholesky` path.

    ``G`` is the assembled whitened normal matrix ``J~'J~ + ridge E``
    (``J~ = J F_bar^{-1}``, ``E`` the metric-block diagonal pad; pre-damping),
    so a rejected step re-factors without re-assembling. ``valid`` marks it
    current for the state's ``x``; ``ridge`` is the weight it was assembled
    with -- a callback ridge change invalidates through this key.
    """

    G: jax.Array
    valid: jax.Array
    ridge: jax.Array

    @property
    def payload(self):
        return self.G


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QRCache:
    """Per-``(x, ridge)`` cache carried by the :class:`QR` path.

    ``R`` is the R factor of the augmented whitened stack
    ``[J~; sqrt(ridge) [I 0] | b~]`` -- its first ``n`` columns are the R
    factor of the stack and its last column carries ``Q'b``, so the velocity
    solve is backward stable with no normal equations. ``valid``/``ridge``
    have the :class:`CholeskyCache` semantics.
    """

    R: jax.Array
    valid: jax.Array
    ridge: jax.Array

    @property
    def payload(self):
        return self.R


@dataclass(frozen=True)
class Subproblem:
    """One LM step's linearized data, posed in the whitened variable.

    The solver builds this once per ``update`` and hands it to
    ``linear_solver.prepare``. ``whiten``/``whiten_transpose`` apply
    ``F_bar^{-1}`` and ``F_bar^{-T}`` (vectors or leading-axis-batched
    matrices), already closed over the metric context; ``Jt`` is the dense
    ``J'`` for the direct paths and ``jvp_fn``/``JT`` the matrix-free pair.
    """

    resid: jax.Array
    theta: jax.Array
    Jt: Any
    jvp_fn: Any
    JT: Any
    whiten: Any
    whiten_transpose: Any
    y_m: jax.Array
    penalty_gradient: jax.Array
    ridge: jax.Array
    damping: jax.Array
    n_m: int
    n_f: int
    cache: Any
    cache_enabled: bool
    hyper: Any
    ctx: Any

    @property
    def dtype(self):
        return self.resid.dtype

    @property
    def n(self):
        return self.theta.shape[0]

    @property
    def m(self):
        return self.resid.shape[0]

    def whitened(self, v):
        """``F_bar^{-1} v`` pinned to the residual dtype.

        A wider-typed metric (float64 kernel data under a float32 residual)
        must not promote the gradient and break the loop-carry dtypes.
        """
        return jnp.asarray(self.whiten(v), dtype=self.dtype)

    def whitened_transpose(self, v):
        """``F_bar^{-T} v`` pinned to the residual dtype."""
        return jnp.asarray(self.whiten_transpose(v), dtype=self.dtype)

    def cached(self, assemble):
        """The cache payload when it is current for ``(x, ridge)``, else a
        fresh ``assemble()``."""
        if not self.cache_enabled:
            return assemble()
        cache = self.cache
        return jax.lax.cond(
            cache.valid & (cache.ridge == self.ridge),
            lambda _: cache.payload,
            lambda _: assemble(),
            operand=None,
        )


class StepSolver(NamedTuple):
    """What a linear solver returns for one LM step.

    ``grad`` is the whitened half-gradient; ``velocity()`` the first-order
    step in the whitened variable; ``solve(rhs)`` the same damped system
    against an arbitrary right-hand side (the geodesic correction);
    ``accel_rhs(f_vv)`` the correction's right-hand side; and
    ``make_cache(valid)`` the pytree to carry, or ``None``.
    """

    grad: jax.Array
    velocity: Any
    solve: Any
    accel_rhs: Any
    make_cache: Any


class LinearSolver:
    """Base class for the typed configs. ``materializes_jacobian`` drives the
    dense ``J'`` assembly and its reject-step reuse."""

    materializes_jacobian = True

    def new_cache(self, m, n, n_m, dtype):
        """The reject-step cache pytree at ``init``, or ``None``."""
        return None

    def prepare(self, sub):
        raise NotImplementedError


@dataclass(frozen=True)
class Cholesky(LinearSolver):
    """Dense normal equations.

    Assembles ``G = J~'J~ + ridge E`` -- cached across rejected steps, where
    only the damping changed, so a reject pays the ``n^3/3`` refactor without
    the GEMM and without re-materializing ``J~'`` -- and factors
    ``G + damping I`` per step. No knobs.
    """

    def new_cache(self, m, n, n_m, dtype):
        return CholeskyCache(
            G=jnp.zeros((n, n), dtype=dtype),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            ridge=jnp.zeros((), dtype=dtype),
        )

    def prepare(self, sub):
        n_m, ridge, dtype = sub.n_m, sub.ridge, sub.dtype
        grad = sub.whitened_transpose(sub.Jt @ sub.resid) + ridge * sub.penalty_gradient

        def assemble():
            Jt_sub = sub.whitened_transpose(sub.Jt)
            diagonal = jnp.arange(n_m)
            return (Jt_sub @ Jt_sub.T).at[diagonal, diagonal].add(ridge)

        normal_matrix = sub.cached(assemble)
        shifted = normal_matrix + sub.damping * jnp.eye(sub.n, dtype=dtype)
        factor = jsp_linalg.cho_factor(shifted)

        def solve(rhs):
            return -jsp_linalg.cho_solve(factor, rhs)

        return StepSolver(
            grad=grad,
            velocity=lambda: solve(grad),
            solve=solve,
            accel_rhs=lambda f_vv: sub.whitened_transpose(sub.Jt @ f_vv),
            make_cache=lambda valid: CholeskyCache(normal_matrix, valid, ridge),
        )


@dataclass(frozen=True)
class QR(LinearSolver):
    """Damping-row QR of the augmented whitened stack.

    One QR of ``[J~; sqrt(ridge) [I 0] | b~]`` with ``b~ = [r; sqrt(ridge)
    y_m]`` is cached per ``(x, ridge)``: its leading columns are the stack's R
    factor and its last column carries ``Q'b``, so the velocity is a
    backward-stable least-squares solve with NO normal equations -- More
    1978's damping-row structure, accurate at ``cond(A)`` rather than
    ``cond(A)^2``. Each step re-factors only the damping rows. No knobs.
    """

    def new_cache(self, m, n, n_m, dtype):
        return QRCache(
            R=jnp.zeros((min(m + n_m, n + 1), n + 1), dtype=dtype),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            ridge=jnp.zeros((), dtype=dtype),
        )

    def prepare(self, sub):
        n, n_m, ridge, dtype = sub.n, sub.n_m, sub.ridge, sub.dtype
        sqrt_ridge = jnp.sqrt(ridge)
        grad = sub.whitened_transpose(sub.Jt @ sub.resid) + ridge * sub.penalty_gradient

        def assemble():
            Jt_sub = sub.whitened_transpose(sub.Jt)
            stacked = jnp.concatenate(
                [Jt_sub.T, sqrt_ridge * jnp.eye(n_m, n, dtype=dtype)], axis=0
            )
            b_stacked = jnp.concatenate([sub.resid, sqrt_ridge * sub.y_m])
            return jnp.linalg.qr(
                jnp.concatenate([stacked, b_stacked[:, None]], axis=1), mode="r"
            )

        qr_R = sub.cached(assemble)
        r_factor, transformed_rhs = qr_R[:, :-1], qr_R[:, -1]
        # Per-step damping-row refactor: [R; sqrt(damping) I] = Q2 R2 with
        # R2'R2 = A'A + damping I. When m + n_m < n the cached R is upper
        # trapezoidal and these rows are what make the system full rank. Q2 is
        # retained to transform the velocity right-hand side stably.
        damped_stack = jnp.concatenate(
            [r_factor, jnp.sqrt(sub.damping) * jnp.eye(n, dtype=dtype)], axis=0
        )
        Q_mu, R_mu = jnp.linalg.qr(damped_stack, mode="reduced")

        def damped_normal_matvec(v):
            gauss_newton = sub.whitened_transpose(sub.Jt @ (sub.Jt.T @ sub.whitened(v)))
            metric_shift = jnp.concatenate([v[:n_m], jnp.zeros(sub.n_f, dtype=dtype)])
            return gauss_newton + ridge * metric_shift + sub.damping * v

        def solve(rhs):
            # Corrected semi-normal equations (Bjorck 1987) for the geodesic
            # right-hand side: triangular solves against R_mu, then ONE fixed
            # iterative-refinement pass through matvecs (Bjorck 1996 Sec.
            # 6.6.5). The second-order correction tolerates the squared
            # conditioning; accept/reject guards it.
            b = -rhs
            half = jsp_linalg.solve_triangular(R_mu.T, b, lower=True)
            delta = jsp_linalg.solve_triangular(R_mu, half, lower=False)
            correction_rhs = b - damped_normal_matvec(delta)
            half = jsp_linalg.solve_triangular(R_mu.T, correction_rhs, lower=True)
            return delta + jsp_linalg.solve_triangular(R_mu, half, lower=False)

        def velocity():
            # min ||[R; sqrt(damping) I] delta + [Q'b; 0]||^2 solved through
            # Q2: exact and backward stable at cond(A), never cond(A)^2.
            rhs = jnp.concatenate([transformed_rhs, jnp.zeros(n, dtype=dtype)])
            return -jsp_linalg.solve_triangular(R_mu, Q_mu.T @ rhs, lower=False)

        return StepSolver(
            grad=grad,
            velocity=velocity,
            solve=solve,
            accel_rhs=lambda f_vv: sub.whitened_transpose(sub.Jt @ f_vv),
            make_cache=lambda valid: QRCache(qr_R, valid, ridge),
        )


@dataclass(frozen=True)
class CG(LinearSolver):
    """Matrix-free preconditioned CG on the whitened normal operator.

    As ``linear_solver`` it solves the damped forward subproblem
    ``(J~'J~ + ridge E + damping I) delta_y = -g`` -- the same SPD system
    :class:`Cholesky` factors -- with the ``preconditioner`` in CG's ``M``
    slot at the live damping. As ``ad_solver`` it solves the undamped
    implicit-AD system ``J~'J~ + ridge E``, with the preconditioner applied at
    zero damping (subclasses marked ``requires_positive_damping`` are rejected
    for that role).

    ``preconditioner`` is REQUIRED in both roles -- nobody should run Krylov
    methods without a preconditioning decision, so
    :class:`~nlls_gram.IdentityPreconditioner` is the explicit opt-out and a
    custom one is a small subclass implementing ``apply(v, damping, ctx)``.
    ``tol=None`` resolves to a dtype default (``1e-10`` in float64, ``1e-6``
    in float32); ``maxiter`` must be set when both tolerances are explicitly
    zero, since an uncapped zero-tolerance CG loop has no stopping rule.
    """

    preconditioner: Preconditioner
    tol: float | None = None
    atol: float = 0.0
    maxiter: int | None = None

    materializes_jacobian = False

    def __post_init__(self):
        if self.tol is not None and self.tol < 0:
            raise ValueError("CG.tol must be nonnegative or None")
        if self.atol < 0:
            raise ValueError("CG.atol must be nonnegative")
        if self.maxiter is not None and self.maxiter <= 0:
            raise ValueError("CG.maxiter must be positive or None")
        if self.tol == 0 and self.atol == 0 and self.maxiter is None:
            raise ValueError("CG.maxiter must be set when both tolerances are zero")

    def prepare(self, sub):
        n_m, n_f, ridge, dtype = sub.n_m, sub.n_f, sub.ridge, sub.dtype
        m, damping, ctx = sub.m, sub.damping, sub.ctx
        sqrt_ridge = jnp.sqrt(ridge)

        # Whitened operator J~ = J F_bar^{-1}: products route through the
        # metric's factor callbacks.
        def J_sub(u):
            return sub.jvp_fn(sub.whitened(u))

        def JT_sub(w):
            return sub.whitened_transpose(sub.JT(w))

        grad = JT_sub(sub.resid) + ridge * sub.penalty_gradient

        # Augmented operator A = [J~; sqrt(ridge) [I 0]]: the penalty rows are
        # the constant metric-block identity on y.
        def A_matvec(u):
            return jnp.concatenate([J_sub(u), sqrt_ridge * u[:n_m]])

        def At_matvec(w):
            pullback = sqrt_ridge * jnp.concatenate(
                [w[m:], jnp.zeros(n_f, dtype=dtype)]
            )
            return JT_sub(w[:m]) + pullback

        # N = A'A + damping I, the preconditioner-free SPD operator that
        # custom_linear_solve differentiates through, posed on u.
        def N_matvec(u):
            return At_matvec(A_matvec(u)) + damping * u

        def apply_M(v):
            return self.preconditioner.apply(v, damping, ctx)

        def solve_N(_, c):
            solution, _ = jsp_sparse_linalg.cg(
                N_matvec,
                c,
                tol=jnp.asarray(sub.hyper.iterative_tol, dtype=dtype),
                atol=jnp.asarray(sub.hyper.iterative_atol, dtype=dtype),
                maxiter=sub.hyper.iterative_maxiter,
                M=apply_M,
            )
            return solution

        def solve(rhs):
            return jax.lax.custom_linear_solve(
                N_matvec, -rhs, solve=solve_N, transpose_solve=solve_N, symmetric=True
            )

        return StepSolver(
            grad=grad,
            velocity=lambda: solve(grad),
            solve=solve,
            accel_rhs=JT_sub,
            make_cache=lambda valid: None,
        )
