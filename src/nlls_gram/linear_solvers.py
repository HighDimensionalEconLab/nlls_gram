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
from nlls_gram.utilities import _IdentityKey, mm

__all__ = [
    "CG",
    "LU",
    "QR",
    "SVD",
    "Cholesky",
    "CholeskyCache",
    "GramCG",
    "NewtonStep",
    "QRCache",
    "StepSolver",
    "Subproblem",
]


def _config_static_key(config, *, baked):
    # Compile-key component for a solver config. The Krylov configs hold a
    # preconditioner instance whose place in the key depends on how its
    # arrays reach the compiled program: threaded through the carried state
    # (forward role) -> key by pytree structure, so equal-config fresh
    # instances share a compile; baked in as closure constants (explicit
    # ad_solver role) -> key by identity, so different values never share
    # one. Everything else is a value-hashable dataclass and keys as itself.
    if config is None or not isinstance(config, (CG, GramCG)):
        return config
    precond = config.preconditioner
    if precond is None:
        precond_key = None
    elif baked:
        precond_key = _IdentityKey(precond)
    else:
        precond_key = jax.tree_util.tree_structure(precond)
    return (
        type(config),
        precond_key,
        config.tol,
        config.atol,
        config.maxiter,
        getattr(config, "penalty", None),
    )


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

    In the normal form ``R`` is the R factor of the augmented whitened stack
    ``[J~; sqrt(ridge) [I 0] | b~]`` -- its first ``n`` columns are the R
    factor of the stack and its last column carries ``Q'b``, so the velocity
    solve is backward stable with no normal equations. In the gram form it
    is the thin QR of ``J~'`` stacked as ``[Q_1; R_1]``, shape ``(n + m, m)``.
    ``valid``/``ridge`` have the :class:`CholeskyCache` semantics.
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


class NewtonStep(NamedTuple):
    """What a linear solver returns for one Newton step.

    ``grad`` is the whitened half-gradient ``B'r`` (reported as
    ``info.grad_norm``); ``d`` is the whitened step direction; ``solve_ok``
    is a scalar boolean indicating whether the factorization and solve
    succeeded with finite full-rank pivots; ``make_cache(valid)`` is the
    cache pytree to carry, or ``None``; and ``correction(f_vv)`` is the
    whitened geodesic second-order correction from the directional second
    derivative ``f_vv``.
    """

    grad: jax.Array
    d: jax.Array
    solve_ok: jax.Array
    make_cache: Any
    correction: Any = None


class LinearSolver:
    """Base class for the typed configs.

    The class attributes declare which roles a config can fill, so an
    unsupported combination is rejected at construction rather than silently
    resolving to a different algorithm:

    - ``materializes_jacobian`` drives the dense ``J'`` assembly and its
      reject-step reuse;
    - ``supports_forward`` / ``supports_newton`` / ``supports_ad``: the
      ``linear_solver`` (for LM and Newton) and ``ad_solver`` roles;
    - ``supports_penalty``: whether the config can carry
      :class:`~nlls_gram.RidgeLevenbergMarquardt`'s penalty rows. The dual
      operator never sees them, so the Gram forms cannot.
    """

    materializes_jacobian = True
    supports_forward = True
    supports_newton = True
    supports_ad = True
    supports_penalty = True

    def new_cache(self, m, n, n_m, dtype, penalized):
        """The reject-step cache pytree at ``init``, or ``None``."""
        return None

    def _validate_newton_shape(self, m, n):
        pass

    def prepare(self, sub):
        raise NotImplementedError

    def prepare_newton(self, sub):
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
    two produce the same step in exact arithmetic, by the push-through
    identity ``B'(BB' + lam I)^{-1} = (B'B + lam I)^{-1}B'``.

    In FLOATING POINT they part company when ``m > n``: the dual then carries
    ``m - n`` structural zero eigenvalues, so its condition number is
    ``sigma_max^2 / damping`` however well conditioned ``J~`` is, and the gram
    step loses digits as damping falls (measured ~1e-2 relative at
    ``damping=1e-14`` where the normal form holds 1e-15). ``form="auto"`` (the
    default) never picks gram there -- it takes gram only when ``n > m``, where
    the normal system is the singular one -- so the default is safe and the
    choice is a cost one. Forcing ``form="gram"`` on a tall problem is not.
    """

    form: str = "auto"

    def __post_init__(self):
        if self.form not in ("auto", "gram", "normal"):
            raise ValueError(
                f"Cholesky.form must be 'auto', 'gram', or 'normal'; got {self.form!r}"
            )

    @property
    def supports_penalty(self):
        # form="auto" resolves to normal under a penalized subproblem.
        return self.form != "gram"

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
        n_m, ridge = sub.n_m, sub.ridge
        # B' = F_bar^{-T} J', shape (n, m). Every form below is built from it.
        grad = sub.whitened_transpose(mm(sub.Jt, sub.resid))
        if sub.penalized:
            grad = grad + ridge * sub.penalty_gradient
        gram = self._resolved_form(sub.m, sub.n, sub.penalized) == "gram"

        def assemble():
            Bt = sub.whitened_transpose(sub.Jt)
            if gram:
                return mm(Bt.T, Bt)
            normal = mm(Bt, Bt.T)
            if not sub.penalized:
                return normal
            diagonal = jnp.arange(n_m)
            return normal.at[diagonal, diagonal].add(ridge)

        matrix = sub.cached(assemble)
        size = sub.m if gram else sub.n
        shift = jnp.arange(size)
        factor = jsp_linalg.cho_factor(matrix.at[shift, shift].add(sub.damping))
        if gram:
            # u = -B'(D + damping I)^{-1} c on residual-space right-hand sides.
            def dual_step(c):
                return -sub.whitened_transpose(
                    mm(sub.Jt, jsp_linalg.cho_solve(factor, c))
                )

            velocity, correction = (lambda: dual_step(sub.resid)), dual_step
        else:

            def normal_step(c):
                return -jsp_linalg.cho_solve(factor, c)

            velocity = lambda: normal_step(grad)  # noqa: E731
            correction = lambda f_vv: normal_step(  # noqa: E731
                sub.whitened_transpose(mm(sub.Jt, f_vv))
            )
        return StepSolver(
            grad=grad,
            velocity=velocity,
            correction=correction,
            make_cache=lambda valid: CholeskyCache(matrix, valid, ridge),
        )

    def _validate_newton_shape(self, m, n):
        if self.form == "gram" and m > n:
            raise ValueError(
                f"Cholesky(form='gram') in Newton requires m <= n, but got m={m}, "
                f"n={n}: the undamped dual J J' is structurally singular when m > n. "
                "Use Cholesky(form='auto'), Cholesky(form='normal'), QR(), or SVD()"
            )
        if self.form == "normal" and n > m:
            raise ValueError(
                f"Cholesky(form='normal') in Newton requires n <= m, but got m={m}, "
                f"n={n}: the undamped normal matrix J'J is structurally singular "
                "when n > m. Use Cholesky(form='auto'), Cholesky(form='gram'), "
                "QR(), or SVD()"
            )

    def prepare_newton(self, sub):
        if sub.penalized:
            n, n_m, ridge, dtype = sub.n, sub.n_m, sub.ridge, sub.dtype
            grad = (
                sub.whitened_transpose(mm(sub.Jt, sub.resid))
                + ridge * sub.penalty_gradient
            )

            def assemble_penalized():
                Bt = sub.whitened_transpose(sub.Jt)
                G = mm(Bt, Bt.T)
                return G.at[jnp.arange(n_m), jnp.arange(n_m)].add(ridge)

            matrix = sub.cached(assemble_penalized)
            tiny = jnp.finfo(dtype).tiny
            diag_max = jnp.maximum(jnp.max(jnp.diag(matrix)), tiny)
            eff_damping = jnp.maximum(sub.damping, jnp.finfo(dtype).eps * diag_max)
            idx = jnp.arange(n)
            shifted = jnp.where(
                sub.damping > 0,
                matrix.at[idx, idx].add(eff_damping),
                matrix,
            )
            factor = jsp_linalg.cho_factor(shifted)
            L_or_U = factor[0]
            factor_ok = jnp.all(jnp.isfinite(L_or_U)) & jnp.all(jnp.diag(L_or_U) > tiny)
            d = -jsp_linalg.cho_solve(factor, grad)
            solve_ok = (
                factor_ok & jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(d))
            )
            return NewtonStep(
                grad=grad,
                d=d,
                solve_ok=solve_ok,
                make_cache=lambda valid: CholeskyCache(matrix, valid, sub.ridge),
                correction=lambda f_vv: (
                    -jsp_linalg.cho_solve(
                        factor, sub.whitened_transpose(mm(sub.Jt, f_vv))
                    )
                ),
            )

        self._validate_newton_shape(sub.m, sub.n)
        grad = sub.whitened_transpose(mm(sub.Jt, sub.resid))
        gram = self._resolved_form(sub.m, sub.n, False) == "gram"

        def assemble():
            Bt = sub.whitened_transpose(sub.Jt)
            return mm(Bt.T, Bt) if gram else mm(Bt, Bt.T)

        matrix = sub.cached(assemble)
        k = sub.m if gram else sub.n
        tiny = jnp.finfo(sub.dtype).tiny
        diag_max = jnp.maximum(jnp.max(jnp.diag(matrix)), tiny)
        eff_damping = jnp.maximum(sub.damping, jnp.finfo(sub.dtype).eps * diag_max)
        idx = jnp.arange(k)
        shifted = jnp.where(
            sub.damping > 0,
            matrix.at[idx, idx].add(eff_damping),
            matrix,
        )
        factor = jsp_linalg.cho_factor(shifted)
        L_or_U = factor[0]
        diag_L = jnp.diag(L_or_U)
        cutoff = jnp.where(
            sub.damping > 0,
            tiny,
            jnp.maximum(
                tiny, max(sub.m, sub.n) * jnp.finfo(sub.dtype).eps * jnp.max(diag_L)
            ),
        )
        factor_ok = jnp.all(jnp.isfinite(L_or_U)) & jnp.all(diag_L > cutoff)
        if gram:
            d = -sub.whitened_transpose(
                mm(sub.Jt, jsp_linalg.cho_solve(factor, sub.resid))
            )

            def correction(f_vv):
                return -sub.whitened_transpose(
                    mm(sub.Jt, jsp_linalg.cho_solve(factor, f_vv))
                )

        else:
            d = -jsp_linalg.cho_solve(factor, grad)

            def correction(f_vv):
                return -jsp_linalg.cho_solve(
                    factor, sub.whitened_transpose(mm(sub.Jt, f_vv))
                )

        solve_ok = factor_ok & jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(d))
        return NewtonStep(
            grad=grad,
            d=d,
            solve_ok=solve_ok,
            make_cache=lambda valid: CholeskyCache(matrix, valid, sub.ridge),
            correction=correction,
        )


@dataclass(frozen=True)
class QR(LinearSolver):
    """Damping-row QR, accurate at ``cond(A)`` rather than ``cond(A)^2``.

    ``form="normal"`` is one QR of the augmented whitened stack
    ``[J~; sqrt(ridge) [I 0] | b~]`` (the penalty rows and the ``b~`` tail
    only for the ridge solver), cached per ``(x, ridge)``: its leading columns
    are the stack's R factor and its last column carries ``Q'b``, so the
    velocity is a backward-stable least-squares solve with NO normal
    equations -- More 1978's damping-row structure. Each step re-factors only
    the ``n`` damping rows, ``[R; sqrt(damping) I]``.

    ``form="gram"`` is the dual: the step ``u = -J~'(J~J~' + damping I)^{-1} c``
    is the leading ``n`` components of the minimum-norm solution of the wide
    system ``[J~, sqrt(damping) I] w = c``, which a thin QR of the transpose
    computes through ONE triangular solve, so it is accurate at
    ``cond([J~, sqrt(damping) I])`` where the gram Cholesky squares it. The
    thin QR ``J~' = Q_1 R_1`` is cached per ``(x, ridge)`` and each step
    re-factors only the ``2m x m`` stack ``[R_1; sqrt(damping) I]``, so the
    cost matches the gram Cholesky's rather than the ``n x n`` normal form's.

    ``form="auto"`` (the default) takes gram when ``n > m`` and the subproblem
    is unpenalized, as :class:`Cholesky` does; the penalty rows have no dual
    analogue. In either form the damping rows keep the system full rank for
    any ``damping > 0``, so a rank-deficient Jacobian is handled rather than
    producing a non-finite step.

    Forward only: the implicit-AD system is undamped, so the damping rows that
    make this path well posed vanish there. Use ``SVD()`` for a rank-deficient
    tangent.
    """

    form: str = "auto"

    supports_ad = False

    def __post_init__(self):
        if self.form not in ("auto", "gram", "normal"):
            raise ValueError(
                f"QR.form must be 'auto', 'gram', or 'normal'; got {self.form!r}"
            )

    @property
    def supports_penalty(self):
        return self.form != "gram"

    def _resolved_form(self, m, n, penalized):
        if penalized:
            return "normal"
        if self.form != "auto":
            return self.form
        return "gram" if n > m else "normal"

    def new_cache(self, m, n, n_m, dtype, penalized):
        if self._resolved_form(m, n, penalized) == "gram":
            shape = (n + m, m)
        else:
            rows = min(m + n_m, n + 1) if penalized else min(m, n + 1)
            shape = (rows, n + 1)
        return QRCache(
            R=jnp.zeros(shape, dtype=dtype),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            ridge=jnp.zeros((), dtype=dtype),
        )

    def prepare(self, sub):
        n, n_m, ridge, dtype = sub.n, sub.n_m, sub.ridge, sub.dtype
        sqrt_ridge = jnp.sqrt(ridge)
        grad = sub.whitened_transpose(mm(sub.Jt, sub.resid))
        if sub.penalized:
            grad = grad + ridge * sub.penalty_gradient
        if self._resolved_form(sub.m, sub.n, sub.penalized) == "gram":
            return self._prepare_gram(sub, grad)

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
            gauss_newton = sub.whitened_transpose(
                mm(sub.Jt, mm(sub.Jt.T, sub.whitened(v)))
            )
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
            b = -sub.whitened_transpose(mm(sub.Jt, f_vv))
            half = jsp_linalg.solve_triangular(R_mu.T, b, lower=True)
            delta = jsp_linalg.solve_triangular(R_mu, half, lower=False)
            correction_rhs = b - damped_normal_matvec(delta)
            half = jsp_linalg.solve_triangular(R_mu.T, correction_rhs, lower=True)
            return delta + jsp_linalg.solve_triangular(R_mu, half, lower=False)

        def velocity():
            # min ||[R; sqrt(damping) I] delta + [Q'b; 0]||^2 solved through
            # Q2: exact and backward stable at cond(A), never cond(A)^2.
            rhs = jnp.concatenate([transformed_rhs, jnp.zeros(n, dtype=dtype)])
            return -jsp_linalg.solve_triangular(R_mu, mm(Q_mu.T, rhs), lower=False)

        return StepSolver(
            grad=grad,
            velocity=velocity,
            correction=correction,
            make_cache=lambda valid: QRCache(qr_R, valid, ridge),
        )

    def _prepare_gram(self, sub, grad):
        n, m, dtype = sub.n, sub.m, sub.dtype

        def assemble():
            Q_1, R_1 = jnp.linalg.qr(sub.whitened_transpose(sub.Jt), mode="reduced")
            return jnp.concatenate([Q_1, R_1], axis=0)

        payload = sub.cached(assemble)
        Q_1, R_1 = payload[:n], payload[n:]
        # [R_1; sqrt(damping) I] = Q_mu R_mu gives the thin QR of the wide
        # system's transpose, [J~'; sqrt(damping) I] = blockdiag(Q_1, I) Q_mu
        # R_mu, so its minimum-norm solution for a residual-space c is
        # blockdiag(Q_1, I) Q_mu R_mu^{-T} c and the step is minus its leading
        # n components.
        damped_stack = jnp.concatenate(
            [R_1, jnp.sqrt(sub.damping) * jnp.eye(m, dtype=dtype)], axis=0
        )
        Q_mu, R_mu = jnp.linalg.qr(damped_stack, mode="reduced")

        def dual_step(c):
            half = jsp_linalg.solve_triangular(R_mu.T, c, lower=True)
            return -mm(Q_1, mm(Q_mu[:m], half))

        return StepSolver(
            grad=grad,
            velocity=lambda: dual_step(sub.resid),
            correction=dual_step,
            make_cache=lambda valid: QRCache(payload, valid, sub.ridge),
        )

    def _validate_newton_shape(self, m, n):
        if self.form == "gram" and m > n:
            raise ValueError(
                f"QR(form='gram') in Newton requires m <= n, but got m={m}, n={n}: "
                "the undamped wide QR of J' requires m <= n. Use QR(form='auto'), "
                "QR(form='normal'), or SVD()"
            )
        if self.form == "normal" and n > m:
            raise ValueError(
                f"QR(form='normal') in Newton requires n <= m, but got m={m}, n={n}: "
                "the undamped tall QR of J requires n <= m. Use QR(form='auto'), "
                "QR(form='gram'), or SVD()"
            )

    def prepare_newton(self, sub):
        if not sub.penalized:
            self._validate_newton_shape(sub.m, sub.n)
        grad = sub.whitened_transpose(mm(sub.Jt, sub.resid))
        if sub.penalized:
            grad = grad + sub.ridge * sub.penalty_gradient
        tiny = jnp.finfo(sub.dtype).tiny
        if self._resolved_form(sub.m, sub.n, sub.penalized) == "gram":
            n, m, dtype = sub.n, sub.m, sub.dtype

            def assemble():
                Q_1, R_1 = jnp.linalg.qr(sub.whitened_transpose(sub.Jt), mode="reduced")
                return jnp.concatenate([Q_1, R_1], axis=0)

            payload = sub.cached(assemble)
            Q_1, R_1 = payload[:n], payload[n:]

            def _damped_gram(_):
                damped_stack = jnp.concatenate(
                    [R_1, jnp.sqrt(sub.damping) * jnp.eye(m, dtype=dtype)], axis=0
                )
                Q_mu, R_mu = jnp.linalg.qr(damped_stack, mode="reduced")
                diag_abs = jnp.abs(jnp.diag(R_mu))
                r_ok = (
                    jnp.all(jnp.isfinite(payload))
                    & jnp.all(jnp.isfinite(R_mu))
                    & jnp.all(diag_abs > tiny)
                )
                Q_mu_top = Q_mu[:m]
                half = jsp_linalg.solve_triangular(R_mu.T, sub.resid, lower=True)
                d = -mm(Q_1, mm(Q_mu_top, half))
                return d, r_ok, Q_mu_top, R_mu

            def _undamped_gram(_):
                diag_abs = jnp.abs(jnp.diag(R_1))
                cutoff = jnp.maximum(
                    tiny, max(m, n) * jnp.finfo(dtype).eps * jnp.max(diag_abs)
                )
                r_ok = jnp.all(jnp.isfinite(payload)) & jnp.all(diag_abs > cutoff)
                half = jsp_linalg.solve_triangular(R_1.T, sub.resid, lower=True)
                d = -mm(Q_1, half)
                return d, r_ok, jnp.zeros((m, m), dtype=dtype), R_1

            d, r_ok, Q_mu_top, R_eff = jax.lax.cond(
                sub.damping > 0, _damped_gram, _undamped_gram, operand=None
            )

            def _gram_correction(f_vv):
                half = jsp_linalg.solve_triangular(R_eff.T, f_vv, lower=True)
                proj = jax.lax.cond(
                    sub.damping > 0,
                    lambda h: mm(Q_mu_top, h),
                    lambda h: h,
                    half,
                )
                return -mm(Q_1, proj)

            solve_ok = r_ok & jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(d))
            return NewtonStep(
                grad=grad,
                d=d,
                solve_ok=solve_ok,
                make_cache=lambda valid: QRCache(payload, valid, sub.ridge),
                correction=_gram_correction,
            )

        n, m, n_m, ridge, dtype = sub.n, sub.m, sub.n_m, sub.ridge, sub.dtype
        sqrt_ridge = jnp.sqrt(ridge)

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
        if qr_R.shape[0] < n:
            pad_rows = n - qr_R.shape[0]
            r_factor = jnp.pad(qr_R[:, :n], ((0, pad_rows), (0, 0)))
            transformed_rhs = jnp.pad(qr_R[:, -1], (0, pad_rows))
        else:
            r_factor = qr_R[:n, :n]
            transformed_rhs = qr_R[:n, -1]

        def _damped_normal(_):
            damped_stack = jnp.concatenate(
                [r_factor, jnp.sqrt(sub.damping) * jnp.eye(n, dtype=dtype)], axis=0
            )
            Q_mu, R_mu = jnp.linalg.qr(damped_stack, mode="reduced")
            diag_abs = jnp.abs(jnp.diag(R_mu))
            r_ok = (
                jnp.all(jnp.isfinite(qr_R))
                & jnp.all(jnp.isfinite(R_mu))
                & jnp.all(diag_abs > tiny)
            )
            rhs = jnp.concatenate([transformed_rhs, jnp.zeros(n, dtype=dtype)])
            d = -jsp_linalg.solve_triangular(R_mu, mm(Q_mu.T, rhs), lower=False)
            return d, r_ok, R_mu

        def _undamped_normal(_):
            diag_abs = jnp.abs(jnp.diag(r_factor))
            cutoff = jnp.maximum(
                tiny, max(m, n) * jnp.finfo(dtype).eps * jnp.max(diag_abs)
            )
            r_ok = jnp.all(jnp.isfinite(qr_R)) & jnp.all(diag_abs > cutoff)
            d = -jsp_linalg.solve_triangular(r_factor, transformed_rhs, lower=False)
            return d, r_ok, r_factor

        d, r_ok, R_eff = jax.lax.cond(
            sub.damping > 0, _damped_normal, _undamped_normal, operand=None
        )

        def damped_normal_matvec(v):
            gauss_newton = sub.whitened_transpose(
                mm(sub.Jt, mm(sub.Jt.T, sub.whitened(v)))
            )
            shift = sub.damping * v
            if sub.penalized:
                shift = shift + ridge * jnp.concatenate(
                    [v[:n_m], jnp.zeros(sub.n_f, dtype=dtype)]
                )
            return gauss_newton + shift

        def _normal_correction(f_vv):
            b = -sub.whitened_transpose(mm(sub.Jt, f_vv))
            half = jsp_linalg.solve_triangular(R_eff.T, b, lower=True)
            delta = jsp_linalg.solve_triangular(R_eff, half, lower=False)
            correction_rhs = b - damped_normal_matvec(delta)
            half = jsp_linalg.solve_triangular(R_eff.T, correction_rhs, lower=True)
            return delta + jsp_linalg.solve_triangular(R_eff, half, lower=False)

        solve_ok = r_ok & jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(d))
        return NewtonStep(
            grad=grad,
            d=d,
            solve_ok=solve_ok,
            make_cache=lambda valid: QRCache(qr_R, valid, sub.ridge),
            correction=_normal_correction,
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

    ``preconditioner`` is REQUIRED in the forward role -- nobody should run
    Krylov methods without a preconditioning decision, so
    :class:`~nlls_gram.IdentityPreconditioner` is the explicit opt-out and a
    custom one is a small registered dataclass implementing
    ``apply(v, damping, ctx)``. In the ``ad_solver`` role,
    ``preconditioner=None`` inherits the CARRIED forward instance at the
    solution while pinning the AD tolerance and budget. On rank-deficient
    problems the preconditioner must map ``range(B')`` into itself or the
    minimum-norm selection is silently lost; the identity, polynomials in the
    operator, and exact shifted inverses are safe, and on full-column-rank
    problems the condition is vacuous.

    ``tol=None`` resolves to a dtype default (``1e-10`` in float64, ``1e-6``
    in float32); ``maxiter`` must be set when both tolerances are explicitly
    zero, since an uncapped zero-tolerance CG loop has no stopping rule.
    """

    preconditioner: Preconditioner | None
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

        # The CARRIED instance, so a callback refresh reaches the very next
        # inner solve; the config's own field only seeds the initial state.
        precond = ctx.lm_state.preconditioner

        def apply_M(v):
            return precond.apply(v, damping, ctx)

        if precond is None:
            apply_M = None

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

    def _validate_newton_shape(self, m, n):
        if n > m:
            raise ValueError(
                f"linear_solver=CG(...) in Newton requires n <= m, but got m={m}, "
                f"n={n}: the undamped normal operator J'J is structurally singular "
                "when n > m. Use GramCG(...) for wide problems"
            )

    def prepare_newton(self, sub):
        if not sub.penalized:
            self._validate_newton_shape(sub.m, sub.n)
        step_solver = self.prepare(sub)
        grad = step_solver.grad
        d = step_solver.velocity()
        solve_ok = jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(d))
        return NewtonStep(
            grad=grad,
            d=d,
            solve_ok=solve_ok,
            make_cache=step_solver.make_cache,
            correction=step_solver.correction,
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
    than one with a flag. The ridge penalty rows have no dual analogue, so
    this config serves :class:`~nlls_gram.LevenbergMarquardt` and
    :class:`~nlls_gram.Newton` (not the ridge solvers).
    """

    supports_penalty = False

    preconditioner: Preconditioner | None
    tol: float | None = None
    atol: float = 0.0
    maxiter: int | None = None

    def prepare(self, sub):
        damping, ctx = sub.damping, sub.ctx

        def dual_matvec(y):
            return sub.jvp_fn(sub.whitened(sub.whitened_transpose(sub.JT(y)))) + (
                damping * y
            )

        # The CARRIED instance, so a callback refresh reaches the very next
        # inner solve; the config's own field only seeds the initial state.
        precond = ctx.lm_state.preconditioner

        def apply_M(v):
            return precond.apply(v, damping, ctx)

        if precond is None:
            apply_M = None

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

    def _validate_newton_shape(self, m, n):
        if m > n:
            raise ValueError(
                f"linear_solver=GramCG(...) in Newton requires m <= n, but got m={m}, "
                f"n={n}: the undamped dual operator J J' is structurally singular "
                "when m > n. Use CG(...) for tall problems"
            )

    def prepare_newton(self, sub):
        if not sub.penalized:
            self._validate_newton_shape(sub.m, sub.n)
        step_solver = self.prepare(sub)
        grad = step_solver.grad
        d = step_solver.velocity()
        solve_ok = jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(d))
        return NewtonStep(
            grad=grad,
            d=d,
            solve_ok=solve_ok,
            make_cache=step_solver.make_cache,
            correction=step_solver.correction,
        )


@dataclass(frozen=True)
class LU(LinearSolver):
    """Direct nonsymmetric solve of a SQUARE system (``Newton`` forward or
    ``ad_solver`` role).

    When the Jacobian is square the system ``J d = -r`` (or the implicit-AD
    system ``B u = -dr/dp``) has a unique solution, so there is no norm being
    minimized and a metric does not select among answers. That makes the plain
    factorization of ``J`` available, and it is strictly better than routing
    through a normal or dual operator: ``cond(J'J) = cond(J)^2``, so Cholesky
    loses half the significant digits for nothing. One factorization also
    serves both directions -- forward mode solves with ``J``, reverse mode
    with ``J'``.

    This is why ``ad_solver=None`` resolves here for a square dense problem.
    It requires squareness and says so: a rectangular system needs a selection
    rule (minimum-norm) that a plain solve cannot express, so :class:`SVD`,
    :class:`QR`, or the :class:`Cholesky` forms own that case.

    The LM forward subproblem is damped and therefore symmetric positive
    definite at every shape, so this config has no LM forward role.
    """

    supports_forward = False
    supports_penalty = False

    def prepare(self, sub):
        raise NotImplementedError("LU is not a damped LM forward solver")

    def _validate_newton_shape(self, m, n):
        if m != n:
            raise ValueError(
                f"linear_solver=LU() in Newton requires a square system (m == n), "
                f"but got m={m}, n={n}: use Cholesky(), QR(), or SVD() for "
                "rectangular systems"
            )

    def prepare_newton(self, sub):
        self._validate_newton_shape(sub.m, sub.n)
        Bt = sub.whitened_transpose(sub.Jt)
        grad = mm(Bt, sub.resid)
        lu, piv = jsp_linalg.lu_factor(Bt.T)
        tiny = jnp.finfo(sub.dtype).tiny
        diag_abs = jnp.abs(jnp.diag(lu))
        cutoff = jnp.maximum(
            tiny, max(sub.m, sub.n) * jnp.finfo(sub.dtype).eps * jnp.max(diag_abs)
        )
        lu_ok = jnp.all(jnp.isfinite(lu)) & jnp.all(diag_abs > cutoff)
        d = -jsp_linalg.lu_solve((lu, piv), sub.resid)
        solve_ok = lu_ok & jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(d))
        return NewtonStep(
            grad=grad,
            d=d,
            solve_ok=solve_ok,
            make_cache=lambda valid: None,
            correction=lambda f_vv: -jsp_linalg.lu_solve((lu, piv), f_vv),
        )


@dataclass(frozen=True)
class SVD(LinearSolver):
    """Spectral-filter pseudoinverse (``Newton`` forward or ``ad_solver`` role).

    Undamped Newton and implicit-AD systems are singular whenever the Jacobian
    is rank deficient -- padded zero residuals make it so by construction.
    This rule truncates singular values at ``max(m, n) * eps * sigma_max`` and
    returns the minimum-norm correction or tangent, which is well defined
    there rather than producing a NaN or a failed Cholesky/QR factorization.
    It assembles the dense Jacobian, so it is the rank-safe dense choice.
    """

    supports_forward = False
    supports_penalty = False

    def prepare(self, sub):
        raise NotImplementedError("SVD is not a damped LM forward solver")

    def prepare_newton(self, sub):
        Bt = sub.whitened_transpose(sub.Jt)
        grad = mm(Bt, sub.resid)
        U, sigma, Vt = jnp.linalg.svd(Bt.T, full_matrices=False)
        cutoff = max(sub.m, sub.n) * jnp.finfo(sub.dtype).eps * sigma[0]
        mask = sigma > cutoff
        safe_sigma = jnp.where(mask, sigma, jnp.ones_like(sigma))
        undamped_inv = jnp.where(mask, 1.0 / safe_sigma, jnp.zeros_like(sigma))
        damped_inv = jnp.where(
            mask, sigma / (sigma**2 + sub.damping), jnp.zeros_like(sigma)
        )
        inverted = jnp.where(sub.damping > 0, damped_inv, undamped_inv)
        d = -mm(Vt.T, inverted * mm(U.T, sub.resid))
        solve_ok = (
            jnp.all(jnp.isfinite(U))
            & jnp.all(jnp.isfinite(sigma))
            & jnp.all(jnp.isfinite(Vt))
            & jnp.all(jnp.isfinite(grad))
            & jnp.all(jnp.isfinite(d))
        )
        return NewtonStep(
            grad=grad,
            d=d,
            solve_ok=solve_ok,
            make_cache=lambda valid: None,
            correction=lambda f_vv: -mm(Vt.T, inverted * mm(U.T, f_vv)),
        )
