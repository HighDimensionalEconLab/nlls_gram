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
    # Static: the ridge solver carries penalty rows, the metric solver
    # does not. ridge is zero when unpenalized, so only the structural
    # choices (extra QR rows, the diagonal shift) key on this.
    penalized: bool = True

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

    ``grad`` is the whitened half-gradient (reported as ``info.grad_norm``);
    ``velocity()`` the first-order step in the whitened variable;
    ``correction(f_vv)`` the geodesic second-order correction from the
    directional second derivative, also whitened; and ``make_cache(valid)``
    the pytree to carry, or ``None``.

    The two solves are one method rather than a ``solve(rhs)`` the caller
    feeds: the Gram forms pose their right-hand side in residual space and the
    normal forms in parameter space, and nothing outside the config needs to
    know which.
    """

    grad: jax.Array
    velocity: Any
    correction: Any
    make_cache: Any


class LinearSolver:
    """Base class for the typed configs. ``materializes_jacobian`` drives the
    dense ``J'`` assembly and its reject-step reuse."""

    materializes_jacobian = True

    def new_cache(self, m, n, n_m, dtype, penalized):
        """The reject-step cache pytree at ``init``, or ``None``."""
        return None

    def prepare(self, sub):
        raise NotImplementedError


@dataclass(frozen=True)
class Cholesky(LinearSolver):
    """Dense factorization of the damped subproblem.

    ``form="normal"`` factors the ``n x n`` whitened normal system
    ``G = J~'J~ + ridge E`` (cached across rejected steps, where only the
    damping changed, so a reject pays the ``n^3/3`` refactor without the GEMM
    and without re-materializing ``J~'``) and solves
    ``(G + damping I) u = -g``.

    ``form="gram"`` factors the ``m x m`` dual ``D = J~ J~'`` instead and
    takes the step ``u = -J~'(D + damping I)^{-1} r``. For ``damping > 0`` the
    two produce the SAME step, by the push-through identity
    ``B'(BB' + lam I)^{-1} = (B'B + lam I)^{-1}B'``; they differ only in which
    dimension they factor in. ``form="auto"`` (the default) picks the smaller
    at trace time -- gram when ``n > m``, normal otherwise -- so it is a cost
    choice, not a semantics choice, and it keys on shape alone, never on
    numerical rank.
    """

    form: str = "auto"

    def _resolved_form(self, m, n, penalized):
        # The ridge solver's penalty rows have no dual analogue -- the dual
        # operator J~J~' never sees them -- so a penalized subproblem is
        # always the normal form. The ridge constructor rejects an explicit
        # form="gram" rather than silently ignoring it.
        if penalized:
            return "normal"
        if self.form != "auto":
            return self.form
        return "gram" if n > m else "normal"

    def new_cache(self, m, n, n_m, dtype, penalized):
        size = m if self._resolved_form(m, n, penalized) == "gram" else n
        return CholeskyCache(
            G=jnp.zeros((size, size), dtype=dtype),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            ridge=jnp.zeros((), dtype=dtype),
        )

    def prepare(self, sub):
        n_m, ridge, dtype = sub.n_m, sub.ridge, sub.dtype
        # B' = F_bar^{-T} J', shape (n, m). Every form below is built from it.
        grad = sub.whitened_transpose(sub.Jt @ sub.resid)
        if sub.penalized:
            grad = grad + ridge * sub.penalty_gradient
        gram = self._resolved_form(sub.m, sub.n, sub.penalized) == "gram"

        def assemble():
            Bt = sub.whitened_transpose(sub.Jt)
            if gram:
                return Bt.T @ Bt
            normal = Bt @ Bt.T
            if not sub.penalized:
                return normal
            diagonal = jnp.arange(n_m)
            return normal.at[diagonal, diagonal].add(ridge)

        matrix = sub.cached(assemble)
        size = sub.m if gram else sub.n
        factor = jsp_linalg.cho_factor(
            matrix + sub.damping * jnp.eye(size, dtype=dtype)
        )
        if gram:
            # u = -B'(D + damping I)^{-1} c on residual-space right-hand sides.
            def dual_step(c):
                return -sub.whitened_transpose(sub.Jt @ jsp_linalg.cho_solve(factor, c))

            velocity, correction = (lambda: dual_step(sub.resid)), dual_step
        else:

            def normal_step(c):
                return -jsp_linalg.cho_solve(factor, c)

            velocity = lambda: normal_step(grad)  # noqa: E731
            correction = lambda f_vv: normal_step(  # noqa: E731
                sub.whitened_transpose(sub.Jt @ f_vv)
            )
        return StepSolver(
            grad=grad,
            velocity=velocity,
            correction=correction,
            make_cache=lambda valid: CholeskyCache(matrix, valid, ridge),
        )


@dataclass(frozen=True)
class QR(LinearSolver):
    """Damping-row QR of the augmented whitened stack.

    One QR of ``[J~; sqrt(ridge) [I 0] | b~]`` (the penalty rows and the
    ``b~`` tail only for the ridge solver) is cached per ``(x, ridge)``: its
    leading columns are the stack's R factor and its last column carries
    ``Q'b``, so the velocity is a backward-stable least-squares solve with NO
    normal equations -- More 1978's damping-row structure, accurate at
    ``cond(A)`` rather than ``cond(A)^2``. Each step re-factors only the
    damping rows, and those rows keep the system full rank for any
    ``damping > 0``, so a rank-deficient Jacobian is handled rather than
    producing a non-finite step. No knobs.
    """

    def new_cache(self, m, n, n_m, dtype, penalized):
        rows = min(m + n_m, n + 1) if penalized else min(m, n + 1)
        return QRCache(
            R=jnp.zeros((rows, n + 1), dtype=dtype),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            ridge=jnp.zeros((), dtype=dtype),
        )

    def prepare(self, sub):
        n, n_m, ridge, dtype = sub.n, sub.n_m, sub.ridge, sub.dtype
        sqrt_ridge = jnp.sqrt(ridge)
        grad = sub.whitened_transpose(sub.Jt @ sub.resid)
        if sub.penalized:
            grad = grad + ridge * sub.penalty_gradient

        def assemble():
            Bt = sub.whitened_transpose(sub.Jt)
            rows, rhs = [Bt.T], [sub.resid]
            if sub.penalized:
                rows.append(sqrt_ridge * jnp.eye(n_m, n, dtype=dtype))
                rhs.append(sqrt_ridge * sub.y_m)
            stacked = jnp.concatenate(rows, axis=0)
            b_stacked = jnp.concatenate(rhs)
            return jnp.linalg.qr(
                jnp.concatenate([stacked, b_stacked[:, None]], axis=1), mode="r"
            )

        qr_R = sub.cached(assemble)
        r_factor, transformed_rhs = qr_R[:, :-1], qr_R[:, -1]
        # Per-step damping-row refactor: [R; sqrt(damping) I] = Q2 R2 with
        # R2'R2 = A'A + damping I. When the cached R is upper trapezoidal these
        # rows are what make the system full rank. Q2 is retained to transform
        # the velocity right-hand side stably.
        damped_stack = jnp.concatenate(
            [r_factor, jnp.sqrt(sub.damping) * jnp.eye(n, dtype=dtype)], axis=0
        )
        Q_mu, R_mu = jnp.linalg.qr(damped_stack, mode="reduced")

        def damped_normal_matvec(v):
            gauss_newton = sub.whitened_transpose(sub.Jt @ (sub.Jt.T @ sub.whitened(v)))
            shift = sub.damping * v
            if sub.penalized:
                shift = shift + ridge * jnp.concatenate(
                    [v[:n_m], jnp.zeros(sub.n_f, dtype=dtype)]
                )
            return gauss_newton + shift

        def correction(f_vv):
            # Corrected semi-normal equations (Bjorck 1987): triangular solves
            # against R_mu, then ONE fixed iterative-refinement pass through
            # matvecs (Bjorck 1996 Sec. 6.6.5). The second-order correction
            # tolerates the squared conditioning; accept/reject guards it.
            b = -sub.whitened_transpose(sub.Jt @ f_vv)
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
            correction=correction,
            make_cache=lambda valid: QRCache(qr_R, valid, ridge),
        )


class _KrylovConfig(LinearSolver):
    """Shared field validation for the matrix-free configs."""

    materializes_jacobian = False

    def __post_init__(self):
        if self.tol is not None and self.tol < 0:
            raise ValueError("tol must be nonnegative or None")
        if self.atol < 0:
            raise ValueError("atol must be nonnegative")
        if self.maxiter is not None and self.maxiter <= 0:
            raise ValueError("maxiter must be positive or None")
        if self.tol == 0 and self.atol == 0 and self.maxiter is None:
            raise ValueError("maxiter must be set when both tolerances are zero")

    def _cg(self, matvec, c, sub, apply_M):
        solution, _ = jsp_sparse_linalg.cg(
            matvec,
            c,
            tol=jnp.asarray(sub.hyper.iterative_tol, dtype=sub.dtype),
            atol=jnp.asarray(sub.hyper.iterative_atol, dtype=sub.dtype),
            maxiter=sub.hyper.iterative_maxiter,
            M=apply_M,
        )
        return solution


@dataclass(frozen=True)
class CG(_KrylovConfig):
    """Matrix-free preconditioned CG on the whitened NORMAL operator, in
    parameter space.

    As ``linear_solver`` it solves the damped forward subproblem
    ``(J~'J~ + ridge E + damping I) delta_y = -g`` -- the same SPD system
    :class:`Cholesky` factors -- with the ``preconditioner`` in CG's ``M`` slot
    at the live damping. As ``ad_solver`` it solves the undamped implicit-AD
    system, with the preconditioner applied at zero damping (subclasses marked
    ``requires_positive_damping`` are rejected for that role) and ``penalty``
    optionally adding a small ridge that stabilizes a rank-deficient tangent.

    ``preconditioner`` is REQUIRED -- nobody should run Krylov methods without
    a preconditioning decision, so :class:`~nlls_gram.IdentityPreconditioner`
    is the explicit opt-out and a custom one is a small subclass implementing
    ``apply(v, damping, ctx)``. On rank-deficient problems it must map
    ``range(B')`` into itself or the minimum-norm selection is silently lost;
    the identity, polynomials in the operator, and exact shifted inverses are
    safe, and on full-column-rank problems the condition is vacuous.

    ``tol=None`` resolves to a dtype default (``1e-10`` in float64, ``1e-6``
    in float32); ``maxiter`` must be set when both tolerances are explicitly
    zero, since an uncapped zero-tolerance CG loop has no stopping rule.
    """

    preconditioner: Preconditioner
    tol: float | None = None
    atol: float = 0.0
    maxiter: int | None = None
    penalty: float | None = None

    def prepare(self, sub):
        n_m, n_f, ridge, dtype = sub.n_m, sub.n_f, sub.ridge, sub.dtype
        damping, ctx = sub.damping, sub.ctx
        sqrt_ridge = jnp.sqrt(ridge)

        # Whitened operator J~ = J F_bar^{-1}: products route through the
        # metric's factor callbacks.
        def J_sub(u):
            return sub.jvp_fn(sub.whitened(u))

        def JT_sub(w):
            return sub.whitened_transpose(sub.JT(w))

        grad = JT_sub(sub.resid)
        if sub.penalized:
            grad = grad + ridge * sub.penalty_gradient

        # N = A'A + damping I for the augmented A = [J~; sqrt(ridge) [I 0]] --
        # the preconditioner-free SPD operator that custom_linear_solve
        # differentiates through, posed on u.
        def N_matvec(u):
            normal = JT_sub(J_sub(u))
            if sub.penalized:
                pullback = sqrt_ridge * jnp.concatenate(
                    [sqrt_ridge * u[:n_m], jnp.zeros(n_f, dtype=dtype)]
                )
                normal = normal + pullback
            return normal + damping * u

        def apply_M(v):
            return self.preconditioner.apply(v, damping, ctx)

        def solve_N(_, c):
            return self._cg(N_matvec, c, sub, apply_M)

        def solve(c):
            return jax.lax.custom_linear_solve(
                N_matvec, -c, solve=solve_N, transpose_solve=solve_N, symmetric=True
            )

        return StepSolver(
            grad=grad,
            velocity=lambda: solve(grad),
            correction=lambda f_vv: solve(JT_sub(f_vv)),
            make_cache=lambda valid: None,
        )


@dataclass(frozen=True)
class GramCG(_KrylovConfig):
    """Matrix-free preconditioned CG on the DUAL operator, in residual space.

    Applies ``y -> J~ J~' y + damping y`` on ``m``-vectors and takes the step
    ``u = -J~'y``, so the Krylov iteration lives in residual dimension -- the
    matrix-free form for the ``m << n`` regime this package targets, where
    :class:`CG`'s ``n``-dimensional iteration is the expensive one. At inner
    convergence the step matches :class:`CG`'s, and a budget-truncated step
    still lies in ``range(J~')``, so the minimum-metric-norm structure
    survives truncation.

    ``preconditioner`` acts on residual-space vectors -- an SPD approximation
    of ``(J~J~' + damping I)^{-1}`` -- which is the only difference from
    :class:`CG`'s contract, and the reason the two are separate configs rather
    than one with a flag. Only the ridge solver has penalty rows, and they
    have no dual analogue, so this config serves
    :class:`~nlls_gram.LevenbergMarquardt` alone.
    """

    preconditioner: Preconditioner
    tol: float | None = None
    atol: float = 0.0
    maxiter: int | None = None

    def prepare(self, sub):
        damping, ctx = sub.damping, sub.ctx

        def dual_matvec(y):
            return sub.jvp_fn(sub.whitened(sub.whitened_transpose(sub.JT(y)))) + (
                damping * y
            )

        def apply_M(v):
            return self.preconditioner.apply(v, damping, ctx)

        def solve_dual(_, c):
            return self._cg(dual_matvec, c, sub, apply_M)

        def step(c):
            y = jax.lax.custom_linear_solve(
                dual_matvec,
                c,
                solve=solve_dual,
                transpose_solve=solve_dual,
                symmetric=True,
            )
            return -sub.whitened_transpose(sub.JT(y))

        return StepSolver(
            grad=sub.whitened_transpose(sub.JT(sub.resid)),
            velocity=lambda: step(sub.resid),
            correction=step,
            make_cache=lambda valid: None,
        )


@dataclass(frozen=True)
class SVD(LinearSolver):
    """Spectral-filter pseudoinverse, for the ``ad_solver`` role only.

    The implicit-AD system is UNDAMPED, so it is singular whenever the
    whitened Jacobian is rank deficient -- padded zero residuals make it so by
    construction. This rule truncates at ``max(m, n) * eps * sigma_max`` and
    returns the minimum-metric-norm tangent, which is the right answer there
    rather than a NaN or a silent pseudo-solve. It assembles, so it is the
    dense fallback rather than the default.
    """

    def prepare(self, sub):
        raise NotImplementedError("SVD is an ad_solver, not a forward solver")
