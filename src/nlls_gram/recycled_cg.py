# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forked conjugate-gradient core from ``jax.scipy.sparse.linalg``.

``recycled_cg`` is a drop-in replacement for ``jax.scipy.sparse.linalg.cg``.
Only the Krylov loop is vendored here (from the installed jax's
``jax/_src/scipy/sparse/linalg.py``) -- it is the extension point for
deflated/recycled CG across LM steps. The surrounding plumbing (``x0``/
``maxiter`` normalization and the ``lax.custom_linear_solve`` wrapper that
provides implicit derivatives) is imported from ``jax._src`` so upstream
improvements carry over; a jax release that moves those internals will fail
loudly at import time rather than silently diverge.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import lax
from jax._src.scipy.sparse.linalg import (
    _add,
    _identity,
    _isolve,
    _mul,
    _normalize_matvec,
    _sub,
    _vdot_real_tree,
)
from jax.tree_util import tree_leaves

_HIGHEST = lax.Precision.HIGHEST


def _apply_columns(matvec, X):
    # Apply the operator to every column of ``X`` (m, k). The package's dual
    # operators accept an ``(m, k)`` matrix directly (leading-axis batching) -- the
    # hot path, one operator application -- so probe that and validate the shape;
    # fall back to a per-column ``vmap`` for plain vector-only callables. The probe
    # is a trace-time capability check: any failure (a vector-only callable can
    # raise a shape/assert error of any type) just means "does not accept the
    # batched shape", so fall back. Shapes are static -- the choice is made once.
    m, k = X.shape
    try:
        out = matvec(X)
        batched = getattr(out, "shape", None) == (m, k)
    except Exception:
        batched = False
    if not batched:
        out = jax.vmap(matvec, in_axes=1, out_axes=1)(X)
    return out


def _sentinel(diag, valid, dtype):
    # A finite diagonal placeholder for masked-out (invalid) rows of the harvest
    # projection: larger than any valid Ritz value so they never rank among the k
    # smallest, yet finite even when NO column is valid (the count==0 cold-start
    # path, reachable by warm-starting at the solution) or when the *1e3 scaling
    # would overflow -- an all -inf reduction would otherwise poison eigh.
    max_valid = jnp.max(jnp.where(valid, diag, -jnp.inf))
    big = jnp.where(jnp.any(valid), max_valid * 1e3 + 1.0, jnp.ones((), dtype))
    return jnp.minimum(big, jnp.finfo(dtype).max)


def _recycled_cg_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):
    # Verbatim fork of jax's _cg_solve, modulo formatting and public-API
    # imports (jnp.result_type for dtypes.result_type); keep the structure
    # diffable against upstream. Recycling state (deflation basis, Lanczos
    # harvest buffers) will thread through this loop's carry.

    # tolerance handling uses the "non-legacy" behavior of
    # scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # preconditioned CG:
    # en.wikipedia.org/wiki/Conjugate_gradient_method (preconditioned variant)

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma.real if M is _identity else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1

    r0 = _sub(b, A(x0))
    p0 = z0 = M(r0)
    dtype = jnp.result_type(*tree_leaves(p0))
    gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
    initial_value = (x0, r0, gamma0, p0, 0)

    x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final


def recycled_cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
    """Drop-in fork of :func:`jax.scipy.sparse.linalg.cg`.

    Semantics are identical to the upstream solver: ``A`` is a hermitian
    positive-definite matvec callable (or matrix), convergence is
    ``norm(residual) <= max(tol * norm(b), atol)``, ``M`` approximates
    ``A^{-1}``, ``x0`` seeds the iteration, and derivatives are implicit
    (another CG solve through ``lax.custom_linear_solve``) rather than
    differentiated through the iterations. Returns ``(x, info)`` with
    ``info=None``, matching upstream.

    The Krylov loop is vendored in this package as the extension point for
    Krylov-subspace recycling across the LM solver's successive dual solves;
    with no recycling state it reproduces upstream CG exactly.
    """
    return _isolve(
        _recycled_cg_solve,
        A=A,
        b=b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        check_symmetric=True,
    )


# --- Deflated / recycled PCG -------------------------------------------------
#
# A two-level additive-coarse-space PCG with an eigCG-style thick-restart
# harvest, kept entirely separate from the verbatim ``recycled_cg`` parity path
# above. The first-level preconditioner ``P`` (the user's structured dual
# preconditioner) is composed with a deflation coarse solve on a carried basis
# ``U`` whose columns approximately span the smallest-eigenvalue subspace of the
# P-preconditioned operator. Each solve harvests the next basis from the CG
# Lanczos trace so it can be recycled into the following (slowly drifting) solve
# at zero rebuild cost.
#
# Callers must NOT differentiate through this directly: the deflation basis and
# harvest are ``stop_gradient``'d and the solution carries implicit derivatives
# via ``lax.custom_linear_solve`` (the same wrapper the parity path reuses).


class HarvestState(NamedTuple):
    """Diagnostics and the next deflation basis emitted by :func:`deflated_pcg`.

    Attributes:
        basis: ``(m, k)`` orthonormal deflation basis harvested from the solve,
            recycled into the next solve. ``stop_gradient``'d.
        iterations: number of PCG iterations run (``()`` integer).
        residual_norm: final ``||b - A x||`` (``()`` scalar).
    """

    basis: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array


def build_coarse_operator(A, U, *, ridge=None):
    """Precompute the deflation coarse operator ``W = A U`` and ``chol(U'A U)``.

    Built once per LM step and reused across every right-hand side (velocity and
    geodesic acceleration share one operator). ``E = U'A U`` is symmetrized and
    shifted by a trace-scaled ridge with a dtype-keyed absolute floor so the
    Cholesky factor stays finite even for a zero or rank-deficient ``U`` -- the
    ridge lives only inside a preconditioner and never moves the converged root.

    Args:
        A: hermitian positive-definite matvec callable (or square matrix).
        U: ``(m, k)`` deflation basis (``k`` columns).
        ridge: trace-scaled ridge fraction ``gamma``; ``None`` uses a dtype-keyed
            default (``1e-12`` float64, ``1e-6`` float32).

    Returns:
        ``(W, E_factor)`` where ``W = A U`` and ``E_factor`` is the
        ``jax.scipy.linalg.cho_factor`` of the ridged ``U'A U``.
    """
    Amv = _normalize_matvec(A)
    W = _apply_columns(Amv, U)
    E = jnp.matmul(U.T, W, precision=_HIGHEST)
    E = 0.5 * (E + E.T)
    dtype = E.dtype
    k = E.shape[0]
    finfo = jnp.finfo(dtype)
    frac = ridge
    if frac is None:
        frac = 1e-12 if jnp.dtype(dtype) == jnp.dtype(jnp.float64) else 1e-6
    floor = finfo.tiny / finfo.eps
    rho = jnp.maximum(jnp.asarray(frac, dtype) * jnp.trace(E) / k, floor)
    E_reg = E + rho * jnp.eye(k, dtype=dtype)
    return W, jsp_linalg.cho_factor(E_reg)


def _deflated_pcg_core(
    A, b, x0, M, U, *, maxiter, tol, atol, window, rank, reorthogonalize, harvest
):
    # Augmented PCG: the standard preconditioned recurrence (identical in the
    # x/r/gamma/p carry to the parity loop, so U=0 reproduces it bitwise), plus a
    # static (m, window) ring buffer of M-normalized Lanczos vectors and the
    # CG-scalar tridiagonal (used only by the cheap harvest route). The harvest
    # (deflation basis for the next solve) runs after the loop when ``harvest``.
    m = b.shape[0]
    w = window
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    r0 = _sub(b, A(x0))
    z0 = M(r0)
    p0 = z0
    dtype = jnp.result_type(*tree_leaves(z0))
    gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
    V0 = jnp.zeros((m, w), dtype)
    Tdiag0 = jnp.zeros((w,), dtype)
    Toff0 = jnp.zeros((w,), dtype)
    boa0 = jnp.zeros((), dtype)

    def cond_fun(value):
        _, r, _, _, _, k, _, _, _, _ = value
        return (_vdot_real_tree(r, r) > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, z, gamma, p, k, V, Tdiag, Toff, boa = value
        Ap = A(p)
        alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
        beta = gamma_ / gamma
        p_ = _add(z_, _mul(beta, p))
        # CG -> Lanczos of the P-preconditioned operator (Saad 6.7.3): the
        # tridiagonal entries come free from the CG scalars, the window column is
        # the M-normalized Lanczos vector z_k / sqrt(gamma_k). The window is a ring
        # (slot = k mod w) keeping the last w vectors -- near convergence these are
        # richest in the slow (small-eigenvalue) modes we want to deflate. Mapping
        # V G back to the original space recovers eigenvectors of the
        # preconditioned operator; Q'A Q equals the tridiagonal block exactly, so a
        # windowed Rayleigh-Ritz reads off the ring's principal sub-block.
        v_col = z / jnp.sqrt(gamma)
        diag_k = jnp.ones((), dtype) / alpha + boa
        # sqrt(beta) is real because beta = gamma_/gamma > 0 for an SPD
        # preconditioner M (gamma = <r, M r> > 0). A non-SPD M is a contract
        # violation and would surface here as a NaN rather than being masked.
        off_k = jnp.sqrt(beta) / alpha
        slot = k % w
        V_ = V.at[:, slot].set(v_col)
        Tdiag_ = Tdiag.at[slot].set(diag_k)
        Toff_ = Toff.at[slot].set(off_k)
        return x_, r_, z_, gamma_, p_, k + 1, V_, Tdiag_, Toff_, beta / alpha

    init = (x0, r0, z0, gamma0, p0, jnp.array(0), V0, Tdiag0, Toff0, boa0)
    x_f, r_f, _, _, _, count, V, Tdiag, Toff, _ = lax.while_loop(
        cond_fun, body_fun, init
    )
    resid_norm = jnp.sqrt(_vdot_real_tree(r_f, r_f))

    if not harvest:
        # Shared-operator RHS (e.g. geodesic acceleration): reuse the carried basis
        # unchanged and skip the Rayleigh-Ritz / QR / extra matvecs entirely.
        return x_f, count, resid_norm, U

    # Unroll the ring into chronological order over the last min(count, w) Lanczos
    # vectors.
    idx = jnp.arange(w)
    nvalid = jnp.minimum(count, w)
    start = jnp.maximum(count - w, 0)
    perm = (start + idx) % w
    V_ord = V[:, perm]

    if reorthogonalize:
        # Thick-restart / GCRO-DR harvest: Rayleigh-Ritz for A over the augmented
        # recycle space [U, window]. Including the carried basis lets U PERSIST and
        # refine even when deflation makes a solve converge in a few iterations
        # (too few to re-harvest a full basis from the window alone). The explicit
        # Q'A Q Rayleigh quotient on the reorthonormalized space is robust to the
        # Lanczos orthogonality drift that pollutes the coefficient tridiagonal's
        # near-converged Ritz vectors (design Hard 3), at k + w extra matvecs.
        # Zero columns (cold U, unfilled window) are masked by column norm and
        # sorted to the back before QR: a leading zero column would otherwise make
        # QR orthogonalize the real columns against an arbitrary completion
        # direction and corrupt their span. When fewer than rank columns are valid
        # (cold start converging in < rank iterations), the surplus selected
        # directions are QR completions -- non-Ritz but finite and orthonormal, and
        # the ridge in build_coarse_operator keeps the next E factorable.
        B = jnp.concatenate([U, V_ord], axis=1)
        colnorm = jnp.linalg.norm(B, axis=0)
        valid = colnorm > jnp.sqrt(jnp.finfo(dtype).eps) * jnp.max(colnorm)
        order = jnp.argsort(~valid)
        B = B[:, order]
        valid = valid[order]
        Q, _ = jnp.linalg.qr(B)
        AQ = _apply_columns(A, Q)
        H = jnp.matmul(Q.T, AQ, precision=_HIGHEST)
        H = 0.5 * (H + H.T)
        big = _sentinel(jnp.diag(H), valid, dtype)
        mask = valid[:, None] & valid[None, :]
        H = jnp.where(mask, H, 0.0) + jnp.diag(jnp.where(valid, 0.0, big))
        _, G = jnp.linalg.eigh(H)
        U_next = jnp.matmul(Q, G[:, :rank], precision=_HIGHEST)
    else:
        # Free coefficient-tridiagonal route (no extra matvecs): cheaper, but
        # window-only (no recycle-space augmentation, so it can starve when a
        # deflated solve converges before the window refills) and its
        # near-converged Ritz vectors can be polluted by orthogonality drift.
        valid = idx < nvalid
        d_ord = Tdiag[perm]
        off_ord = Toff[perm]
        big = _sentinel(d_ord, valid, dtype)
        d_h = jnp.where(valid, d_ord, big)
        off_h = jnp.where(idx < nvalid - 1, off_ord, jnp.zeros((), dtype))
        T = jnp.diag(d_h) + jnp.diag(off_h[: w - 1], 1) + jnp.diag(off_h[: w - 1], -1)
        _, G = jnp.linalg.eigh(T)
        U_raw = jnp.matmul(V_ord, G[:, :rank], precision=_HIGHEST)
        U_next, _ = jnp.linalg.qr(U_raw)
    return x_f, count, resid_norm, U_next


def deflated_pcg(
    A,
    b,
    *,
    U,
    E_factor,
    M=None,
    x0=None,
    tol=1e-5,
    atol=0.0,
    maxiter=None,
    window=None,
    rank=None,
    reorthogonalize=True,
    harvest=True,
):
    """Two-level deflated PCG with an eigCG-style harvest of the next basis.

    Solves ``A y = b`` with a two-level additive preconditioner
    ``M_defl(r) = P(r) + U (E^{-1} (U' r))`` (``P`` the first-level
    preconditioner ``M``, ``U`` the carried deflation basis, ``E = U'A U``
    supplied pre-factored as ``E_factor``) started from a deflated,
    warm-started initial guess, and harvests an orthonormal ``(m, rank)`` basis
    from the CG Lanczos trace for the next solve.

    With ``U = 0`` the coarse correction and deflated init vanish exactly (the
    ridge floor keeps ``E_factor`` finite while ``U' r = 0``), so the iterates
    reduce bitwise to plain PCG with ``P`` -- the parity path of
    :func:`recycled_cg`.

    Derivatives are implicit: the solution is wrapped in the same
    ``lax.custom_linear_solve`` the parity path reuses (warm-started at the
    harvested rough solution so the primal re-solve costs ~one matvec), and the
    basis/harvest are ``stop_gradient``'d. Callers must not differentiate through
    the harvest directly.

    Args:
        A: hermitian positive-definite matvec callable (or square matrix).
        b: right-hand side ``(m,)``.
        U: ``(m, rank)`` deflation basis (zeros for a cold start).
        E_factor: ``cho_factor`` of the ridged ``U'A U`` from
            :func:`build_coarse_operator`.
        M: first-level preconditioner ``P`` (callable); ``None`` is identity.
        x0: warm start ``(m,)`` (previous dual solution); ``None`` is zeros.
        tol: relative convergence tolerance on ``||b - A y||``.
        atol: absolute convergence tolerance.
        maxiter: iteration cap; ``None`` uses ``10 * m``.
        window: harvest window ``w`` (static); ``None`` uses
            ``max(2 * rank, rank + 4)``.
        rank: number of deflation vectors ``k`` (static); ``None`` uses
            ``U.shape[1]``.
        reorthogonalize: full QR re-orthonormalization of the window at harvest.
        harvest: when ``False`` (static), skip the Rayleigh-Ritz / QR / extra
            matvecs and return the carried ``U`` unchanged in the ``HarvestState``.
            Used for a right-hand side that shares the operator with an already
            harvested solve (e.g. the geodesic-acceleration correction).

    Returns:
        ``(y, harvest_state)`` with ``y`` the solution and ``harvest_state`` a
        :class:`HarvestState`.
    """
    Amv = _normalize_matvec(A)
    P = _identity if M is None else _normalize_matvec(M)
    dtype = b.dtype
    m = b.shape[0]
    k = U.shape[1] if rank is None else rank
    w = window if window is not None else max(2 * k, k + 4)
    if k <= 0:
        raise ValueError(f"rank must be positive, got {k}")
    if w < k:
        raise ValueError(f"window ({w}) must be >= rank ({k})")
    if k > m or w > m:
        raise ValueError(f"rank ({k}) and window ({w}) must be <= problem size m ({m})")
    if maxiter is None:
        maxiter = 10 * m
    tol = jnp.asarray(tol, dtype)
    atol = jnp.asarray(atol, dtype)

    def M_defl(r):
        c = jsp_linalg.cho_solve(E_factor, jnp.matmul(U.T, r, precision=_HIGHEST))
        return P(r) + jnp.matmul(U, c, precision=_HIGHEST)

    # Deflated + warm-started initial guess: removes the range(U) component of the
    # solution error before the first iteration (exact when U spans an invariant
    # subspace); reduces to x0 when U = 0.
    if x0 is None:
        resid0 = b
        warm = jnp.zeros(m, dtype)
    else:
        warm = x0
        resid0 = b - Amv(x0)
    x_start = warm + jnp.matmul(
        U,
        jsp_linalg.cho_solve(E_factor, jnp.matmul(U.T, resid0, precision=_HIGHEST)),
        precision=_HIGHEST,
    )

    x_rough, count, resid_norm, U_next = _deflated_pcg_core(
        Amv,
        b,
        x_start,
        M_defl,
        U,
        maxiter=maxiter,
        tol=tol,
        atol=atol,
        window=w,
        rank=k,
        reorthogonalize=reorthogonalize,
        harvest=harvest,
    )

    # Attach implicit derivatives via the same custom_linear_solve wrapper the
    # parity path reuses, warm-started at the rough solution: the primal solve is
    # already at tol so its while_loop exits immediately (~one matvec), while the
    # tangent/cotangent solves iterate deflation-accelerated within maxiter.
    x_ws = lax.stop_gradient(x_rough)

    def deflated_solve(matvec, rhs):
        return _recycled_cg_solve(
            matvec, rhs, x_ws, maxiter=maxiter, tol=tol, atol=atol, M=M_defl
        )

    y = lax.custom_linear_solve(
        Amv, b, solve=deflated_solve, transpose_solve=deflated_solve, symmetric=True
    )
    harvest = HarvestState(
        basis=lax.stop_gradient(U_next),
        iterations=lax.stop_gradient(count),
        residual_norm=lax.stop_gradient(resid_norm),
    )
    return y, harvest
