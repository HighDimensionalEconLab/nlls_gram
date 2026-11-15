from dataclasses import InitVar, dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from nlls_gram.types import SolverContext
from nlls_gram.utilities import mm, register_pytree_dataclass

__all__ = [
    "DiagonalRidge",
    "FactorRidge",
    "IdentityRidge",
    "MatrixRidge",
    "RepeatedFactorRidge",
    "RidgeMetric",
    "SolverContext",
]


class _ShiftedPenalizedBlock(NamedTuple):
    solve_Dp: Any
    solve_Gp: Any
    solve_Gp_T: Any
    solve_d0_qr: Any
    ok: jax.Array


def _check_covered_size(covered, n, label):
    if covered > n:
        raise ValueError(
            f"{label} covers {covered} coordinates, exceeding parameter dimension {n}"
        )


def _pad_tail(v_p, n_f, dtype):
    v_p = jnp.asarray(v_p, dtype=dtype)
    if n_f == 0:
        return v_p
    pad_shape = (n_f, *v_p.shape[1:])
    return jnp.concatenate([v_p, jnp.zeros(pad_shape, dtype=dtype)], axis=0)


class RidgeMetric:
    """Symmetric PSD objective operator ``Q`` for generalized Tikhonov regularization.

    Defines the objective penalty ``0.5 * ridge * x' Q x`` without altering the
    Euclidean step geometry. Custom matrix-free subclasses for ``CG`` need only
    implement ``matvec(v, ctx)`` and register with ``register_pytree_dataclass``.
    """

    has_factor = False
    has_normal_add = False
    has_shifted_solve = False
    fast_shifted_solve = False
    penalized_is_spd = False

    def n_free_for(self, n):
        return 0

    def partition_indices(self, n):
        return None

    def matvec(self, v, ctx):
        raise NotImplementedError

    def quad_form(self, v, ctx):
        qv = jnp.asarray(self.matvec(v, ctx), dtype=v.dtype)
        return jnp.sum(v * qv)


@dataclass(frozen=True, eq=False)
class IdentityRidge(RidgeMetric):
    """Identity penalty ``Q = I`` on leading ``size`` entries (all when ``None``)."""

    size: int | None = None

    has_factor = True
    has_normal_add = True
    has_shifted_solve = True
    fast_shifted_solve = True
    penalized_is_spd = True

    def __post_init__(self):
        if self.size is not None and self.size < 0:
            raise ValueError("IdentityRidge size must be nonnegative or None")

    def _penalized_dim(self, n):
        if self.size is None:
            return n
        _check_covered_size(self.size, n, "IdentityRidge")
        return self.size

    def n_free_for(self, n):
        return n - self._penalized_dim(n)

    def factor_rows_for(self, n):
        return self._penalized_dim(n)

    def matvec(self, v, ctx):
        n_p = self._penalized_dim(v.shape[0])
        return _pad_tail(v[:n_p], v.shape[0] - n_p, v.dtype)

    def quad_form(self, v, ctx):
        n_p = self._penalized_dim(v.shape[0])
        return jnp.sum(v[:n_p] ** 2)

    def factor_apply(self, v, ctx):
        n_p = self._penalized_dim(v.shape[0])
        return v[:n_p]

    def factor_matrix(self, n, dtype, ctx):
        n_p = self._penalized_dim(n)
        return jnp.eye(n_p, dtype=dtype)

    def add_scaled_to_normal(self, G, rho, ctx):
        n_p = self._penalized_dim(G.shape[0])
        if n_p == 0:
            return G
        idx = jnp.arange(n_p)
        return G.at[idx, idx].add(jnp.asarray(rho, dtype=G.dtype))

    def prepare_shifted_penalized(
        self, n, rho, mu, dtype, ctx, *, prefer_qr=False, active=True
    ):
        n_p = self._penalized_dim(n)
        shift = jnp.asarray(rho, dtype=dtype) + jnp.asarray(mu, dtype=dtype)
        tiny = jnp.finfo(dtype).tiny
        ok = jnp.isfinite(shift) & (shift > tiny)
        safe_shift = jnp.where(ok, shift, jnp.ones((), dtype=dtype))
        sqrt_shift = jnp.sqrt(safe_shift)

        def solve_Dp(rhs):
            return jnp.asarray(rhs, dtype=dtype) / safe_shift

        def solve_Gp(rhs):
            return jnp.asarray(rhs, dtype=dtype) / sqrt_shift

        def solve_d0_qr(theta_p):
            return -(jnp.asarray(rho, dtype=dtype) / safe_shift) * jnp.asarray(
                theta_p[:n_p], dtype=dtype
            )

        return _ShiftedPenalizedBlock(
            solve_Dp=solve_Dp,
            solve_Gp=solve_Gp,
            solve_Gp_T=solve_Gp,
            solve_d0_qr=solve_d0_qr,
            ok=ok,
        )


register_pytree_dataclass(IdentityRidge, data_fields=(), meta_fields=("size",))


@dataclass(frozen=True, eq=False)
class DiagonalRidge(RidgeMetric):
    """Nonnegative diagonal penalty ``Q = diag(weights)`` on leading coordinates."""

    weights: jax.Array
    pos_indices: tuple[int, ...] | None = field(init=False)
    penalized_is_spd: bool = field(init=False)

    has_factor = True
    has_normal_add = True
    has_shifted_solve = True
    fast_shifted_solve = True

    def __post_init__(self):
        w = jnp.asarray(self.weights)
        if w.ndim != 1:
            raise ValueError("DiagonalRidge weights must be a 1-D array")
        w = w.astype(jnp.result_type(w, 1.0))
        pos_indices = None
        is_spd = False
        if not isinstance(w, jax.core.Tracer):
            w_np = np.asarray(w)
            if not np.all(np.isfinite(w_np)) or np.any(w_np < 0.0):
                raise ValueError("DiagonalRidge weights must be finite and nonnegative")
            nz = np.flatnonzero(w_np > 0.0)
            if len(nz) < w_np.shape[0]:
                pos_indices = tuple(int(i) for i in nz)
            is_spd = True
        else:
            valid = jnp.all(jnp.isfinite(w)) & jnp.all(w >= 0.0)
            w = jnp.where(valid, w, jnp.full_like(w, jnp.nan))
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "pos_indices", pos_indices)
        object.__setattr__(self, "penalized_is_spd", is_spd)

    def _sqrt_weights(self, dtype):
        w = jnp.asarray(self.weights, dtype=dtype)
        pos = w > 0
        return jnp.where(pos, jnp.sqrt(jnp.where(pos, w, jnp.ones_like(w))), 0.0)

    def n_free_for(self, n):
        k = self.weights.shape[0]
        _check_covered_size(k, n, "DiagonalRidge")
        if self.pos_indices is not None:
            return n - len(self.pos_indices)
        return n - k

    def partition_indices(self, n):
        k = self.weights.shape[0]
        _check_covered_size(k, n, "DiagonalRidge")
        if self.pos_indices is None:
            return None
        if tuple(range(len(self.pos_indices))) == self.pos_indices:
            return None
        pos_set = set(self.pos_indices)
        free_idx = tuple(i for i in range(n) if i not in pos_set)
        return (
            jnp.asarray(self.pos_indices, dtype=jnp.int32),
            jnp.asarray(free_idx, dtype=jnp.int32),
        )

    def factor_rows_for(self, n):
        k = self.weights.shape[0]
        _check_covered_size(k, n, "DiagonalRidge")
        if self.pos_indices is not None:
            return len(self.pos_indices)
        return k

    def matvec(self, v, ctx):
        k = self.weights.shape[0]
        _check_covered_size(k, v.shape[0], "DiagonalRidge")
        w = jnp.asarray(self.weights, dtype=v.dtype)
        shape = (k,) + (1,) * (v.ndim - 1)
        return _pad_tail(v[:k] * w.reshape(shape), v.shape[0] - k, v.dtype)

    def quad_form(self, v, ctx):
        k = self.weights.shape[0]
        _check_covered_size(k, v.shape[0], "DiagonalRidge")
        w = jnp.asarray(self.weights, dtype=v.dtype)
        shape = (k,) + (1,) * (v.ndim - 1)
        return jnp.sum(w.reshape(shape) * (v[:k] ** 2))

    def factor_apply(self, v, ctx):
        k = self.weights.shape[0]
        _check_covered_size(k, v.shape[0], "DiagonalRidge")
        sw = self._sqrt_weights(v.dtype)
        if self.pos_indices is not None:
            m_pos = len(self.pos_indices)
            if m_pos == 0:
                return jnp.zeros((0, *v.shape[1:]), dtype=v.dtype)
            idx = jnp.asarray(self.pos_indices, dtype=jnp.int32)
            shape = (m_pos,) + (1,) * (v.ndim - 1)
            return v[idx] * sw[idx].reshape(shape)
        shape = (k,) + (1,) * (v.ndim - 1)
        return v[:k] * sw.reshape(shape)

    def factor_matrix(self, n, dtype, ctx):
        k = self.weights.shape[0]
        _check_covered_size(k, n, "DiagonalRidge")
        sw = self._sqrt_weights(dtype)
        if self.pos_indices is not None:
            if len(self.pos_indices) == 0:
                return jnp.zeros((0, 0), dtype=dtype)
            idx = jnp.asarray(self.pos_indices, dtype=jnp.int32)
            return jnp.diag(sw[idx])
        return jnp.diag(sw)

    def add_scaled_to_normal(self, G, rho, ctx):
        k = self.weights.shape[0]
        _check_covered_size(k, G.shape[0], "DiagonalRidge")
        if k == 0:
            return G
        idx = jnp.arange(k)
        return G.at[idx, idx].add(
            jnp.asarray(rho, dtype=G.dtype) * jnp.asarray(self.weights, dtype=G.dtype)
        )

    def prepare_shifted_penalized(
        self, n, rho, mu, dtype, ctx, *, prefer_qr=False, active=True
    ):
        k = self.weights.shape[0]
        _check_covered_size(k, n, "DiagonalRidge")
        w_all = jnp.asarray(self.weights, dtype=dtype)
        if self.pos_indices is not None:
            idx = jnp.asarray(self.pos_indices, dtype=jnp.int32)
            w_p = w_all[idx]
        else:
            w_p = w_all
        rho_d = jnp.asarray(rho, dtype=dtype)
        mu_d = jnp.asarray(mu, dtype=dtype)
        diag_d = rho_d * w_p + mu_d
        tiny = jnp.finfo(dtype).tiny
        ok = jnp.all(jnp.isfinite(diag_d)) & jnp.all(diag_d > tiny)
        safe_d = jnp.where(diag_d > tiny, diag_d, jnp.ones_like(diag_d))
        sqrt_d = jnp.sqrt(safe_d)

        def _scale(arr, denom):
            arr = jnp.asarray(arr, dtype=dtype)
            shape = (denom.shape[0],) + (1,) * (arr.ndim - 1)
            return arr / denom.reshape(shape)

        def solve_Dp(rhs):
            return _scale(rhs, safe_d)

        def solve_Gp(rhs):
            return _scale(rhs, sqrt_d)

        def solve_d0_qr(theta_p):
            th = jnp.asarray(theta_p, dtype=dtype)
            return -((rho_d * w_p) / safe_d) * th

        return _ShiftedPenalizedBlock(
            solve_Dp=solve_Dp,
            solve_Gp=solve_Gp,
            solve_Gp_T=solve_Gp,
            solve_d0_qr=solve_d0_qr,
            ok=ok,
        )


register_pytree_dataclass(
    DiagonalRidge,
    data_fields=("weights",),
    meta_fields=("pos_indices", "penalized_is_spd"),
)


@dataclass(frozen=True, eq=False)
class MatrixRidge(RidgeMetric):
    """Dense symmetric PSD penalty ``Q`` on the leading ``Q.shape[0]`` coordinates."""

    Q: jax.Array
    rtol: InitVar[float | None] = None
    L: jax.Array = field(init=False)
    size: int = field(init=False)
    penalized_is_spd: bool = field(init=False)

    has_factor = True
    has_normal_add = True
    has_shifted_solve = True
    fast_shifted_solve = False

    def __post_init__(self, rtol):
        q_arr = jnp.asarray(self.Q)
        if q_arr.ndim != 2 or q_arr.shape[0] != q_arr.shape[1]:
            raise ValueError("MatrixRidge Q must be a square 2-D matrix")
        dtype = jnp.result_type(q_arr, 1.0)
        q_arr = q_arr.astype(dtype)
        k = q_arr.shape[0]
        eps = float(jnp.finfo(dtype).eps)
        tiny = float(jnp.finfo(dtype).tiny)
        sym_tol = (max(k, 1) * 16.0 * eps) if rtol is None else float(rtol)
        if not isinstance(q_arr, jax.core.Tracer):
            q_np = np.asarray(q_arr)
            if not np.all(np.isfinite(q_np)):
                raise ValueError("MatrixRidge Q must contain finite entries")
            q_norm = float(np.linalg.norm(q_np))
            asym = float(np.linalg.norm(q_np - q_np.T))
            if asym > sym_tol * max(q_norm, tiny):
                raise ValueError(
                    f"MatrixRidge Q must be symmetric (||Q - Q'||_F = {asym:.3e}, "
                    f"||Q||_F = {q_norm:.3e})"
                )
            q_sym_np = 0.5 * (q_np + q_np.T)
            q_sym = jnp.asarray(q_sym_np, dtype=dtype)
            active = np.flatnonzero(np.any(q_sym_np != 0.0, axis=1))
            if len(active) == 0:
                L = jnp.zeros((0, k), dtype=dtype)
                is_spd = k == 0
            else:
                q_sub_np = q_sym_np[np.ix_(active, active)]
                evals_np, evecs_np = np.linalg.eigh(q_sub_np)
                sub_norm = float(np.linalg.norm(q_sub_np))
                min_eval = float(np.min(evals_np))
                if min_eval < -sym_tol * max(sub_norm, tiny):
                    raise ValueError(
                        f"MatrixRidge Q must be positive semidefinite "
                        f"(min eigenvalue = {min_eval:.3e})"
                    )
                max_eval = float(np.max(evals_np))
                eig_cutoff = max(len(active), 1) * eps * max(max_eval, 0.0)
                pos = evals_np > 0.0
                if not np.any(pos):
                    L = jnp.zeros((0, k), dtype=dtype)
                else:
                    L_sub_np = np.sqrt(evals_np[pos])[:, None] * evecs_np[:, pos].T
                    L_np = np.zeros((L_sub_np.shape[0], k), dtype=q_sym_np.dtype)
                    L_np[:, active] = L_sub_np
                    L = jnp.asarray(L_np, dtype=dtype)
                is_spd = len(active) == k and bool(np.all(evals_np > eig_cutoff))
        else:
            q_norm = jnp.linalg.norm(q_arr)
            asym = jnp.linalg.norm(q_arr - q_arr.T)
            q_sym_raw = 0.5 * (q_arr + q_arr.T)
            evals, evecs = jnp.linalg.eigh(q_sym_raw)
            scale = jnp.maximum(q_norm, jnp.finfo(dtype).tiny)
            valid = (
                jnp.all(jnp.isfinite(q_arr))
                & (asym <= sym_tol * scale)
                & (jnp.min(evals) >= -sym_tol * scale)
            )
            q_sym = jnp.where(valid, q_sym_raw, jnp.full_like(q_sym_raw, jnp.nan))
            pos = evals > 0.0
            sqrt_evals = jnp.where(
                pos, jnp.sqrt(jnp.where(pos, evals, jnp.ones_like(evals))), 0.0
            )
            L_raw = sqrt_evals[:, None] * evecs.T
            L = jnp.where(valid, L_raw, jnp.full_like(L_raw, jnp.nan))
            is_spd = False
        object.__setattr__(self, "Q", q_sym)
        object.__setattr__(self, "L", L)
        object.__setattr__(self, "size", k)
        object.__setattr__(self, "penalized_is_spd", is_spd)

    def n_free_for(self, n):
        _check_covered_size(self.size, n, "MatrixRidge")
        return n - self.size

    def factor_rows_for(self, n):
        _check_covered_size(self.size, n, "MatrixRidge")
        return self.L.shape[0]

    def factor_apply(self, v, ctx):
        _check_covered_size(self.size, v.shape[0], "MatrixRidge")
        L = jnp.asarray(self.L, dtype=v.dtype)
        return mm(L, v[: self.size])

    def factor_matrix(self, n, dtype, ctx):
        _check_covered_size(self.size, n, "MatrixRidge")
        return jnp.asarray(self.L, dtype=dtype)

    def matvec(self, v, ctx):
        _check_covered_size(self.size, v.shape[0], "MatrixRidge")
        q = jnp.asarray(self.Q, dtype=v.dtype)
        qp = mm(q, v[: self.size])
        return _pad_tail(qp, v.shape[0] - self.size, v.dtype)

    def quad_form(self, v, ctx):
        _check_covered_size(self.size, v.shape[0], "MatrixRidge")
        if self.size == 0:
            return jnp.zeros((), dtype=v.dtype)
        q = jnp.asarray(self.Q, dtype=v.dtype)
        vp = v[: self.size]
        return jnp.sum(vp * mm(q, vp))

    def add_scaled_to_normal(self, G, rho, ctx):
        _check_covered_size(self.size, G.shape[0], "MatrixRidge")
        if self.size == 0:
            return G
        q_block = jnp.asarray(rho, dtype=G.dtype) * jnp.asarray(self.Q, dtype=G.dtype)
        return G.at[: self.size, : self.size].add(q_block)

    def prepare_shifted_penalized(
        self, n, rho, mu, dtype, ctx, *, prefer_qr=False, active=True
    ):
        _check_covered_size(self.size, n, "MatrixRidge")
        return _dense_factor_shifted_block(
            jnp.asarray(self.L, dtype=dtype),
            self.size,
            rho,
            mu,
            dtype,
            prefer_qr=prefer_qr,
            active=active,
            Q_sym=jnp.asarray(self.Q, dtype=dtype),
        )


register_pytree_dataclass(
    MatrixRidge,
    data_fields=("Q", "L"),
    meta_fields=("size", "penalized_is_spd"),
)


def _dense_factor_shifted_block(
    L, k, rho, mu, dtype, *, prefer_qr=False, active=True, Q_sym=None
):
    rho_d = jnp.asarray(rho, dtype=dtype)
    mu_d = jnp.asarray(mu, dtype=dtype)
    sqrt_rho = jnp.sqrt(rho_d)
    tiny = jnp.finfo(dtype).tiny
    eps = jnp.finfo(dtype).eps
    if k == 0:
        ok = jnp.asarray(True, dtype=jnp.bool_)
        return _ShiftedPenalizedBlock(
            solve_Dp=lambda rhs: jnp.asarray(rhs, dtype=dtype),
            solve_Gp=lambda rhs: jnp.asarray(rhs, dtype=dtype),
            solve_Gp_T=lambda rhs: jnp.asarray(rhs, dtype=dtype),
            solve_d0_qr=lambda th: jnp.asarray(th[:0], dtype=dtype),
            ok=ok,
        )
    if prefer_qr:
        top = sqrt_rho * L
        stack = jnp.concatenate([top, jnp.sqrt(mu_d) * jnp.eye(k, dtype=dtype)], axis=0)
        if active is not True:
            stack = jnp.where(active, stack, jnp.eye(*stack.shape, dtype=dtype))
        Q_G, R_G = jnp.linalg.qr(stack, mode="reduced")
        diag_abs = jnp.abs(jnp.diag(R_G))
        cutoff = jnp.where(
            mu_d > 0,
            tiny,
            jnp.maximum(tiny, max(stack.shape) * eps * jnp.max(diag_abs)),
        )
        ok = jnp.all(jnp.isfinite(R_G)) & jnp.all(diag_abs > cutoff)
        R_work = (
            R_G if active is True else jnp.where(active, R_G, jnp.eye(k, dtype=dtype))
        )

        def solve_Gp(rhs):
            return jsp_linalg.solve_triangular(
                R_work, jnp.asarray(rhs, dtype=dtype), lower=False
            )

        def solve_Gp_T(rhs):
            return jsp_linalg.solve_triangular(
                R_work.T, jnp.asarray(rhs, dtype=dtype), lower=True
            )

        def solve_Dp(rhs):
            return solve_Gp(solve_Gp_T(rhs))

        def solve_d0_qr(theta_p):
            p_rows = top.shape[0]
            b_top = mm(top, jnp.asarray(theta_p, dtype=dtype))
            return -solve_Gp(mm(Q_G[:p_rows].T, b_top))

        return _ShiftedPenalizedBlock(
            solve_Dp=solve_Dp,
            solve_Gp=solve_Gp,
            solve_Gp_T=solve_Gp_T,
            solve_d0_qr=solve_d0_qr,
            ok=ok,
        )

    base_Q = mm(L.T, L) if Q_sym is None else Q_sym
    D_p = (rho_d * base_Q).at[jnp.arange(k), jnp.arange(k)].add(mu_d)
    if active is not True:
        D_p = jnp.where(active, D_p, jnp.eye(k, dtype=dtype))
    R_G = jsp_linalg.cho_factor(D_p, lower=False)[0]
    diag_R = jnp.diag(R_G)
    cutoff = jnp.where(
        mu_d > 0,
        tiny,
        jnp.maximum(tiny, jnp.sqrt(k * eps) * jnp.max(diag_R)),
    )
    ok = jnp.all(jnp.isfinite(R_G)) & jnp.all(diag_R > cutoff)
    R_work = R_G if active is True else jnp.where(active, R_G, jnp.eye(k, dtype=dtype))

    def solve_Gp(rhs):
        return jsp_linalg.solve_triangular(
            R_work, jnp.asarray(rhs, dtype=dtype), lower=False
        )

    def solve_Gp_T(rhs):
        return jsp_linalg.solve_triangular(
            R_work.T, jnp.asarray(rhs, dtype=dtype), lower=True
        )

    def solve_Dp(rhs):
        return jsp_linalg.cho_solve((R_work, False), jnp.asarray(rhs, dtype=dtype))

    def solve_d0_qr(theta_p):
        th = jnp.asarray(theta_p, dtype=dtype)
        q_th = mm(L.T, mm(L, th)) if Q_sym is None else mm(Q_sym, th)
        return -solve_Dp(rho_d * q_th)

    return _ShiftedPenalizedBlock(
        solve_Dp=solve_Dp,
        solve_Gp=solve_Gp,
        solve_Gp_T=solve_Gp_T,
        solve_d0_qr=solve_d0_qr,
        ok=ok,
    )


@dataclass(frozen=True, eq=False)
class FactorRidge(RidgeMetric):
    """Explicit factor ``L`` defining ``Q = L'L`` on leading ``L.shape[1]`` entries."""

    L: jax.Array
    size: int = field(init=False)
    penalized_is_spd: bool = field(init=False)

    has_factor = True
    has_normal_add = True
    has_shifted_solve = True
    fast_shifted_solve = False

    def __post_init__(self):
        L = jnp.asarray(self.L)
        if L.ndim != 2:
            raise ValueError("FactorRidge L must be a 2-D array")
        dtype = jnp.result_type(L, 1.0)
        L = L.astype(dtype)
        k = L.shape[1]
        if not isinstance(L, jax.core.Tracer):
            l_np = np.asarray(L)
            if k == 0:
                is_spd = bool(np.all(np.isfinite(l_np)))
            elif not np.all(np.isfinite(l_np)) or L.shape[0] < k:
                is_spd = False
            else:
                s = np.linalg.svd(l_np, compute_uv=False)
                cutoff = max(L.shape) * float(jnp.finfo(dtype).eps) * float(s[0])
                is_spd = bool(s[-1] > cutoff)
        else:
            is_spd = L.shape[0] >= k
        object.__setattr__(self, "L", L)
        object.__setattr__(self, "size", k)
        object.__setattr__(self, "penalized_is_spd", is_spd)

    def n_free_for(self, n):
        _check_covered_size(self.size, n, "FactorRidge")
        return n - self.size

    def factor_rows_for(self, n):
        _check_covered_size(self.size, n, "FactorRidge")
        return self.L.shape[0]

    def factor_apply(self, v, ctx):
        _check_covered_size(self.size, v.shape[0], "FactorRidge")
        L = jnp.asarray(self.L, dtype=v.dtype)
        return mm(L, v[: self.size])

    def factor_matrix(self, n, dtype, ctx):
        _check_covered_size(self.size, n, "FactorRidge")
        return jnp.asarray(self.L, dtype=dtype)

    def matvec(self, v, ctx):
        _check_covered_size(self.size, v.shape[0], "FactorRidge")
        L = jnp.asarray(self.L, dtype=v.dtype)
        qp = mm(L.T, mm(L, v[: self.size]))
        return _pad_tail(qp, v.shape[0] - self.size, v.dtype)

    def quad_form(self, v, ctx):
        lv = self.factor_apply(v, ctx)
        return jnp.sum(lv**2)

    def add_scaled_to_normal(self, G, rho, ctx):
        _check_covered_size(self.size, G.shape[0], "FactorRidge")
        if self.size == 0 or self.L.shape[0] == 0:
            return G
        L = jnp.asarray(self.L, dtype=G.dtype)
        q_block = jnp.asarray(rho, dtype=G.dtype) * mm(L.T, L)
        return G.at[: self.size, : self.size].add(q_block)

    def prepare_shifted_penalized(
        self, n, rho, mu, dtype, ctx, *, prefer_qr=False, active=True
    ):
        _check_covered_size(self.size, n, "FactorRidge")
        return _dense_factor_shifted_block(
            jnp.asarray(self.L, dtype=dtype),
            self.size,
            rho,
            mu,
            dtype,
            prefer_qr=prefer_qr,
            active=active,
        )


register_pytree_dataclass(
    FactorRidge,
    data_fields=("L",),
    meta_fields=("size", "penalized_is_spd"),
)


@dataclass(frozen=True, eq=False)
class RepeatedFactorRidge(RidgeMetric):
    """Block-diagonal RKHS seminorm ``Q = blockdiag(L'L, ..., L'L, 0)``."""

    L: jax.Array
    repeats: int = field(default=1, kw_only=True)
    K: jax.Array = field(init=False)
    size: int = field(init=False)
    penalized_is_spd: bool = field(init=False)

    has_factor = True
    has_normal_add = True
    has_shifted_solve = True
    fast_shifted_solve = True

    def __post_init__(self):
        L = jnp.asarray(self.L)
        if L.ndim != 2 or L.shape[1] == 0:
            raise ValueError(
                "RepeatedFactorRidge L must be a 2-D array with positive column count"
            )
        if self.repeats < 1:
            raise ValueError("RepeatedFactorRidge repeats must be >= 1")
        dtype = jnp.result_type(L, 1.0)
        L = L.astype(dtype)
        k = L.shape[1]
        if not isinstance(L, jax.core.Tracer):
            l_np = np.asarray(L)
            if not np.all(np.isfinite(l_np)) or L.shape[0] < k:
                is_spd = False
            else:
                s = np.linalg.svd(l_np, compute_uv=False)
                cutoff = max(L.shape) * float(jnp.finfo(dtype).eps) * float(s[0])
                is_spd = bool(s[-1] > cutoff)
        else:
            is_spd = L.shape[0] >= k
        object.__setattr__(self, "L", L)
        object.__setattr__(self, "K", mm(L.T, L))
        object.__setattr__(self, "size", self.repeats * k)
        object.__setattr__(self, "penalized_is_spd", is_spd)

    def _map_blocks(self, block_op, v_p, out_rows):
        k = self.L.shape[1]
        trailing = v_p.shape[1:]
        packed = jnp.moveaxis(v_p.reshape((self.repeats, k) + trailing), 0, 1).reshape(
            k, -1
        )
        out = block_op(packed)
        return jnp.moveaxis(
            out.reshape((out_rows, self.repeats) + trailing), 0, 1
        ).reshape((self.repeats * out_rows,) + trailing)

    def n_free_for(self, n):
        _check_covered_size(self.size, n, "RepeatedFactorRidge")
        return n - self.size

    def factor_rows_for(self, n):
        _check_covered_size(self.size, n, "RepeatedFactorRidge")
        return self.repeats * self.L.shape[0]

    def factor_apply(self, v, ctx):
        _check_covered_size(self.size, v.shape[0], "RepeatedFactorRidge")
        L = jnp.asarray(self.L, dtype=v.dtype)
        return self._map_blocks(lambda m: mm(L, m), v[: self.size], L.shape[0])

    def factor_matrix(self, n, dtype, ctx):
        _check_covered_size(self.size, n, "RepeatedFactorRidge")
        L = jnp.asarray(self.L, dtype=dtype)
        return jsp_linalg.block_diag(*([L] * self.repeats))

    def matvec(self, v, ctx):
        _check_covered_size(self.size, v.shape[0], "RepeatedFactorRidge")
        K_block = jnp.asarray(self.K, dtype=v.dtype)
        qp = self._map_blocks(lambda m: mm(K_block, m), v[: self.size], self.L.shape[1])
        return _pad_tail(qp, v.shape[0] - self.size, v.dtype)

    def quad_form(self, v, ctx):
        lv = self.factor_apply(v, ctx)
        return jnp.sum(lv**2)

    def add_scaled_to_normal(self, G, rho, ctx):
        _check_covered_size(self.size, G.shape[0], "RepeatedFactorRidge")
        if self.L.shape[0] == 0:
            return G
        k = self.L.shape[1]
        scaled_K = jnp.asarray(rho, dtype=G.dtype) * jnp.asarray(self.K, dtype=G.dtype)
        sub = G[: self.size, : self.size].reshape(self.repeats, k, self.repeats, k)
        idx = jnp.arange(self.repeats)
        sub = sub.at[idx, :, idx, :].add(scaled_K)
        return G.at[: self.size, : self.size].set(sub.reshape(self.size, self.size))

    def prepare_shifted_penalized(
        self, n, rho, mu, dtype, ctx, *, prefer_qr=False, active=True
    ):
        _check_covered_size(self.size, n, "RepeatedFactorRidge")
        L = jnp.asarray(self.L, dtype=dtype)
        K_block = jnp.asarray(self.K, dtype=dtype)
        k = L.shape[1]
        block = _dense_factor_shifted_block(
            L,
            k,
            rho,
            mu,
            dtype,
            prefer_qr=prefer_qr,
            active=active,
            Q_sym=K_block,
        )

        def solve_Dp(rhs):
            return self._map_blocks(block.solve_Dp, jnp.asarray(rhs, dtype=dtype), k)

        def solve_Gp(rhs):
            return self._map_blocks(block.solve_Gp, jnp.asarray(rhs, dtype=dtype), k)

        def solve_Gp_T(rhs):
            return self._map_blocks(block.solve_Gp_T, jnp.asarray(rhs, dtype=dtype), k)

        def solve_d0_qr(theta_p):
            return self._map_blocks(
                block.solve_d0_qr, jnp.asarray(theta_p, dtype=dtype), k
            )

        return _ShiftedPenalizedBlock(
            solve_Dp=solve_Dp,
            solve_Gp=solve_Gp,
            solve_Gp_T=solve_Gp_T,
            solve_d0_qr=solve_d0_qr,
            ok=block.ok,
        )


register_pytree_dataclass(
    RepeatedFactorRidge,
    data_fields=("L", "K"),
    meta_fields=("repeats", "size", "penalized_is_spd"),
)
