import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from nlls_gram.preconditioners import (
    DiagonalPreconditioner,
    IdentityPreconditioner,
    Preconditioner,
)
from nlls_gram.utilities import _IdentityKey, mm

__all__ = [
    "CG",
    "LU",
    "QR",
    "SVD",
    "Cholesky",
    "CholeskyCache",
    "GramCG",
    "PreparedStep",
    "QRCache",
    "Subproblem",
]


def _validate_dense_preconditioner(precond, cls_name):
    if precond is not None and not isinstance(
        precond, (IdentityPreconditioner, DiagonalPreconditioner)
    ):
        raise TypeError(
            f"{cls_name} only supports preconditioner=None, IdentityPreconditioner(), "
            f"or DiagonalPreconditioner(...), got {type(precond).__name__}. "
            "Use CG or GramCG for general matrix-free preconditioners"
        )


def _config_static_key(config, *, baked):
    if config is None:
        return None
    precond = getattr(config, "preconditioner", None)
    if precond is None:
        precond_key = None
    elif baked and jax.tree_util.tree_leaves(precond):
        precond_key = _IdentityKey(precond)
    else:
        precond_key = jax.tree_util.tree_structure(precond)
    if isinstance(config, (Cholesky, QR)):
        return (type(config), config.form, precond_key)
    if isinstance(config, (CG, GramCG)):
        if not baked:
            return (
                type(config),
                precond_key,
                config.maxiter is None,
                getattr(config, "penalty", None),
            )
        return (
            type(config),
            precond_key,
            config.tol,
            config.atol,
            config.maxiter,
            getattr(config, "penalty", None),
        )
    return config


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _svd_pinv(At, rcond=None):
    U, sigma, Vt = jnp.linalg.svd(At.T, full_matrices=False)
    eps = jnp.finfo(At.dtype).eps
    rel = max(At.shape) * eps if rcond is None else jnp.asarray(rcond, dtype=At.dtype)
    cutoff = rel * jnp.where(sigma.size > 0, sigma[0], jnp.zeros((), dtype=At.dtype))
    mask = sigma > cutoff
    safe_sigma = jnp.where(mask, sigma, jnp.ones_like(sigma))
    inverted = jnp.where(mask, 1.0 / safe_sigma, jnp.zeros_like(sigma))
    return mm(Vt.T * inverted, U.T)


@_svd_pinv.defjvp
def _svd_pinv_jvp(rcond, primals, tangents):
    (At,) = primals
    (dAt,) = tangents
    X = _svd_pinv(At, rcond)
    if At.shape[0] >= At.shape[1]:
        term1 = -mm(X, mm(dAt.T, X))
        XXt_dAt = mm(X, mm(X.T, dAt))
        term2 = XXt_dAt - mm(XXt_dAt, mm(At.T, X))
        dAt_XtX = mm(dAt, mm(X.T, X))
        term3 = dAt_XtX - mm(X, mm(At.T, dAt_XtX))
    else:
        term1 = -mm(mm(X, dAt.T), X)
        XXt_dAt = mm(mm(X, X.T), dAt)
        term2 = XXt_dAt - mm(mm(XXt_dAt, At.T), X)
        dAt_XtX = mm(mm(dAt, X.T), X)
        term3 = dAt_XtX - mm(mm(X, At.T), dAt_XtX)
    return X, term1 + term2 + term3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CholeskyCache:
    """Cached pre-damping symmetric matrix for ``Cholesky`` across rejected steps."""

    G: jax.Array
    valid: jax.Array
    ridge: jax.Array

    @property
    def payload(self):
        return self.G


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QRCache:
    """Cached pre-damping thin QR factors for ``QR`` across rejected steps."""

    R: jax.Array
    valid: jax.Array
    ridge: jax.Array

    @property
    def payload(self):
        return self.R


@dataclass(frozen=True)
class Subproblem:
    resid: jax.Array
    theta: jax.Array
    Jt: Any
    jvp_fn: Any
    JT: Any
    ridge: jax.Array
    damping: jax.Array
    ridge_metric: Any
    penalty_grad: jax.Array
    cache: Any
    cache_enabled: bool
    hyper: Any
    ctx: Any
    penalized: bool = False
    undamped: bool = False
    always_damped: bool = False
    active: Any = True
    preconditioner: Any = None

    @property
    def dtype(self):
        return self.resid.dtype

    @property
    def n(self):
        return self.theta.shape[0]

    @property
    def m(self):
        return self.resid.shape[0]

    def cached(self, assemble):
        if not self.cache_enabled or self.cache is None:
            return assemble()
        cache = self.cache
        return jax.lax.cond(
            cache.valid & (cache.ridge == self.ridge),
            lambda _: cache.payload,
            lambda _: assemble(),
            operand=None,
        )


class PreparedStep(NamedTuple):
    grad: jax.Array
    velocity: Any
    solve_residual_rhs: Any
    solve_ok: jax.Array
    make_cache: Any


def _active_matrix(mat, active):
    if active is True:
        return mat
    return jnp.where(active, mat, jnp.eye(mat.shape[0], dtype=mat.dtype))


def _active_rect(mat, active):
    if active is True:
        return mat
    return jnp.where(active, mat, jnp.eye(*mat.shape, dtype=mat.dtype))


def _cholesky_spd_solve(H, m_scale, damping, precond, active, *, spd_guaranteed=False):
    dtype = H.dtype
    k = H.shape[0]
    if k == 0:
        return (
            lambda rhs: jnp.asarray(rhs, dtype=dtype),
            jnp.asarray(True, dtype=jnp.bool_),
            None,
        )
    tiny = jnp.finfo(dtype).tiny
    eps = jnp.finfo(dtype).eps
    if isinstance(precond, DiagonalPreconditioner):
        c = precond.cholesky_scale(jnp.diag(H), damping, dtype)
        inv_c = 1.0 / c
        H_work = (inv_c[:, None] * H) * inv_c[None, :]
    else:
        inv_c = None
        H_work = H
    H_work = _active_matrix(H_work, active)
    factor = jsp_linalg.cho_factor(H_work)
    L_or_U = factor[0]
    diag_L = jnp.diag(L_or_U)
    diag_H = jnp.diag(H_work)
    scale_floor = m_scale * eps * jnp.max(diag_L)
    col_floor = jnp.sqrt(m_scale * eps * jnp.maximum(diag_H, tiny))
    cutoff = (
        tiny
        if spd_guaranteed
        else jnp.where(
            damping > 0,
            tiny,
            jnp.maximum(tiny, jnp.maximum(scale_floor, col_floor)),
        )
    )
    ok = jnp.all(jnp.isfinite(L_or_U)) & jnp.all(diag_L > cutoff)

    def solve(rhs):
        rhs = jnp.asarray(rhs, dtype=dtype)
        if inv_c is None:
            return jsp_linalg.cho_solve(factor, rhs)
        shape = (k,) + (1,) * (rhs.ndim - 1)
        ic = inv_c.reshape(shape)
        return ic * jsp_linalg.cho_solve(factor, ic * rhs)

    return solve, ok, factor


def _split_parameter_blocks(arr, n_p, part):
    if part is None:
        return arr[:n_p], arr[n_p:]
    p_idx, f_idx = part
    return arr[p_idx], arr[f_idx]


def _merge_parameter_blocks(arr_p, arr_f, n, part):
    if part is None:
        return jnp.concatenate([arr_p, arr_f], axis=0)
    p_idx, f_idx = part
    out = jnp.zeros((n, *arr_p.shape[1:]), dtype=arr_p.dtype)
    return out.at[p_idx].set(arr_p).at[f_idx].set(arr_f)


class LinearSolver:
    materializes_jacobian = True
    supports_forward = True
    supports_newton = True
    supports_ad = True
    supports_penalty = True

    def new_cache(self, m, n, ridge_metric, dtype, penalized):
        return None

    def validate_shape(
        self, m, n, *, penalized, ridge_metric, damped, role="linear_solver"
    ):
        pass

    def prepare(self, sub):
        raise NotImplementedError


@dataclass(frozen=True)
class Cholesky(LinearSolver):
    """Dense Cholesky solver on normal or Gram equations."""

    form: str = "auto"
    preconditioner: Preconditioner | None = None

    def __post_init__(self):
        if self.form not in ("auto", "gram", "normal"):
            raise ValueError(
                f"Cholesky.form must be 'auto', 'gram', or 'normal', got {self.form!r}"
            )
        _validate_dense_preconditioner(self.preconditioner, "Cholesky")

    def resolve_form(self, m, n, penalized, ridge_metric, damped=True):
        if self.form != "auto":
            return self.form
        if not penalized:
            return "gram" if n > m else "normal"
        if (
            n > m
            and ridge_metric.has_shifted_solve
            and ridge_metric.fast_shifted_solve
            and ridge_metric.penalized_is_spd
            and ridge_metric.n_free_for(n) <= m
        ):
            return "gram"
        return "normal"

    def validate_shape(
        self, m, n, *, penalized, ridge_metric, damped, role="linear_solver"
    ):
        form = self.resolve_form(m, n, penalized, ridge_metric, damped=damped)
        if penalized and form == "normal" and not ridge_metric.has_normal_add:
            raise ValueError(
                f"Cholesky(form='normal') in {role} requires a ridge_metric with "
                "dense normal-equation support (IdentityRidge, DiagonalRidge, "
                "MatrixRidge, FactorRidge, or RepeatedFactorRidge)"
            )
        if penalized and form == "gram":
            if not ridge_metric.has_shifted_solve:
                raise ValueError(
                    f"Cholesky(form='gram') in {role} requires a ridge_metric with "
                    "shifted block solves (IdentityRidge, DiagonalRidge, MatrixRidge, "
                    "FactorRidge, or RepeatedFactorRidge)"
                )
            if not ridge_metric.penalized_is_spd:
                raise ValueError(
                    f"Cholesky(form='gram') in {role} requires the penalized block "
                    "of ridge_metric to be strictly positive definite; use "
                    "Cholesky(form='normal'), Cholesky(form='auto'), CG(...), "
                    "or SVD(...)"
                )
            n_f = ridge_metric.n_free_for(n)
            if n_f > 0 and isinstance(self.preconditioner, DiagonalPreconditioner):
                raise ValueError(
                    "Cholesky(form='gram') with ridge > 0 and unpenalized free "
                    "coordinates does not support DiagonalPreconditioner; use "
                    "Cholesky(form='gram') without preconditioner or "
                    "Cholesky(form='normal', preconditioner=...)"
                )
            if not damped and n_f > m:
                raise ValueError(
                    f"Cholesky(form='gram') in {role} at zero damping requires "
                    f"n_free <= m, got n_free={n_f}, m={m}"
                )
            return
        if damped:
            return
        if not penalized and form == "gram" and m > n:
            raise ValueError(
                f"Cholesky(form='gram') in {role} requires m <= n at zero damping, "
                f"got m={m}, n={n}"
            )
        if form == "normal":
            eff_rows = (
                m + ridge_metric.factor_rows_for(n)
                if (penalized and ridge_metric.has_factor)
                else (m + n if penalized else m)
            )
            if n > eff_rows:
                raise ValueError(
                    f"Cholesky(form='normal') in {role} requires full column support "
                    f"at zero damping, got effective rows {eff_rows} < n={n}"
                )

    def new_cache(self, m, n, ridge_metric, dtype, penalized):
        form = self.resolve_form(m, n, penalized, ridge_metric, damped=True)
        if penalized and form == "gram":
            return None
        size = m if form == "gram" else n
        return CholeskyCache(
            G=jnp.zeros((size, size), dtype=dtype),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            ridge=jnp.zeros((), dtype=dtype),
        )

    def prepare(self, sub):
        m, n, dtype = sub.m, sub.n, sub.dtype
        rho, mu = sub.ridge, sub.damping
        rm = sub.ridge_metric
        grad = mm(sub.Jt, sub.resid)
        if sub.penalized:
            grad = grad + rho * sub.penalty_grad
        form = self.resolve_form(m, n, sub.penalized, rm, damped=not sub.undamped)
        m_scale = max(m, n, 1)
        precond = sub.preconditioner

        if form == "normal":

            def assemble():
                G = mm(sub.Jt, sub.Jt.T)
                if not sub.penalized:
                    return G
                return rm.add_scaled_to_normal(G, rho, sub.ctx)

            G = sub.cached(assemble)
            H_mu = G.at[jnp.arange(n), jnp.arange(n)].add(mu)
            solve_H, factor_ok, _ = _cholesky_spd_solve(
                H_mu, m_scale, mu, precond, sub.active
            )

            def velocity():
                return -solve_H(grad)

            def solve_residual_rhs(rhs_r):
                return -solve_H(mm(sub.Jt, jnp.asarray(rhs_r, dtype=dtype)))

            return PreparedStep(
                grad=grad,
                velocity=velocity,
                solve_residual_rhs=solve_residual_rhs,
                solve_ok=factor_ok & jnp.all(jnp.isfinite(grad)),
                make_cache=lambda valid: CholeskyCache(G, valid, rho),
            )

        if not sub.penalized:

            def assemble_gram():
                return mm(sub.Jt.T, sub.Jt)

            D_gram = sub.cached(assemble_gram)
            S_mu = D_gram.at[jnp.arange(m), jnp.arange(m)].add(mu)
            solve_S, factor_ok, _ = _cholesky_spd_solve(
                S_mu, m_scale, mu, precond, sub.active
            )

            def solve_residual_rhs(rhs_r):
                return -mm(sub.Jt, solve_S(jnp.asarray(rhs_r, dtype=dtype)))

            return PreparedStep(
                grad=grad,
                velocity=lambda: solve_residual_rhs(sub.resid),
                solve_residual_rhs=solve_residual_rhs,
                solve_ok=factor_ok & jnp.all(jnp.isfinite(grad)),
                make_cache=lambda valid: CholeskyCache(D_gram, valid, rho),
            )

        # Generalized ridge Gram with two-stage SPD bordered free-tail elimination
        tiny = jnp.finfo(dtype).tiny
        n_f = rm.n_free_for(n)
        n_p = n - n_f
        part = rm.partition_indices(n)
        block = rm.prepare_shifted_penalized(
            n, rho, mu, dtype, sub.ctx, prefer_qr=False, active=sub.active
        )
        Jp_t, Jf_t = _split_parameter_blocks(sub.Jt, n_p, part)
        theta_p, _ = _split_parameter_blocks(sub.theta, n_p, part)
        q_theta_p, _ = _split_parameter_blocks(sub.penalty_grad, n_p, part)
        grad_p, grad_f = _split_parameter_blocks(grad, n_p, part)

        def _penalized_d0():
            sol = block.solve_Dp(jnp.column_stack([theta_p, rho * q_theta_p, grad_p]))
            mu_dp_th = mu * sol[:, 0]
            d0_shift = -theta_p + mu_dp_th
            d0_rq = -sol[:, 1]
            d0_grad = -sol[:, 2]
            d0_split = jnp.where(
                jnp.linalg.norm(mu_dp_th) <= 0.5 * jnp.linalg.norm(theta_p),
                d0_shift,
                d0_rq,
            )
            use_grad = jnp.linalg.norm(d0_grad) < jnp.linalg.norm(d0_split)
            return jnp.where(use_grad, d0_grad, d0_split), use_grad

        if n_f == 0:
            Dp_inv_Jt = block.solve_Dp(Jp_t)
            S = jnp.eye(m, dtype=dtype) + mm(Jp_t.T, Dp_inv_Jt)
            if isinstance(precond, DiagonalPreconditioner):
                solve_S, s_ok, _ = _cholesky_spd_solve(
                    S,
                    m_scale,
                    jnp.zeros((), dtype=dtype),
                    precond,
                    sub.active,
                    spd_guaranteed=True,
                )
            else:
                S_work = _active_matrix(S, sub.active)
                L_S_raw = jnp.linalg.cholesky(S_work)

                def _qr_lower_s0(_):
                    Bp = block.solve_Gp_T(Jp_t)
                    stack = _active_rect(
                        jnp.concatenate([Bp, jnp.eye(m, dtype=dtype)], axis=0),
                        sub.active,
                    )
                    _, R_S = jnp.linalg.qr(stack, mode="reduced")
                    return R_S.T

                L_S = jax.lax.cond(
                    jnp.all(jnp.isfinite(L_S_raw)),
                    lambda _: L_S_raw,
                    _qr_lower_s0,
                    operand=None,
                )
                s_ok = jnp.all(jnp.isfinite(L_S)) & jnp.all(
                    jnp.abs(jnp.diag(L_S)) > tiny
                )

                def solve_S(rhs):
                    return jsp_linalg.cho_solve(
                        (L_S, True), jnp.asarray(rhs, dtype=dtype)
                    )

            ok = block.ok & s_ok & jnp.all(jnp.isfinite(grad))

            def velocity():
                d0, use_grad = _penalized_d0()
                jp_d0 = mm(Jp_t.T, d0)
                c = jnp.where(use_grad, jp_d0, sub.resid + jp_d0)
                return d0 - mm(Dp_inv_Jt, solve_S(c))

            def solve_residual_rhs(rhs_r):
                return -mm(Dp_inv_Jt, solve_S(jnp.asarray(rhs_r, dtype=dtype)))

            return PreparedStep(
                grad=grad,
                velocity=velocity,
                solve_residual_rhs=solve_residual_rhs,
                solve_ok=ok,
                make_cache=lambda valid: None,
            )

        # Two-stage bordered elimination when n_f > 0: zero 1/mu divisions
        Dp_inv_Jpt = block.solve_Dp(Jp_t) if n_p > 0 else jnp.zeros((0, m), dtype=dtype)
        S_p = (
            jnp.eye(m, dtype=dtype) + mm(Jp_t.T, Dp_inv_Jpt)
            if n_p > 0
            else jnp.eye(m, dtype=dtype)
        )
        S_p_work = _active_matrix(S_p, sub.active)
        L_S_raw = jnp.linalg.cholesky(S_p_work)
        if n_p > 0:

            def _qr_lower_sp(_):
                Bp = block.solve_Gp_T(Jp_t)
                stack = _active_rect(
                    jnp.concatenate([Bp, jnp.eye(m, dtype=dtype)], axis=0),
                    sub.active,
                )
                _, R_S = jnp.linalg.qr(stack, mode="reduced")
                return R_S.T

            L_S = jax.lax.cond(
                jnp.all(jnp.isfinite(L_S_raw)),
                lambda _: L_S_raw,
                _qr_lower_sp,
                operand=None,
            )
        else:
            L_S = L_S_raw
        s_ok = jnp.all(jnp.isfinite(L_S)) & jnp.all(jnp.abs(jnp.diag(L_S)) > tiny)
        W = jsp_linalg.solve_triangular(L_S, Jf_t.T, lower=True)
        M_f = mm(W.T, W).at[jnp.arange(n_f), jnp.arange(n_f)].add(mu)
        solve_Mf, mf_ok, _ = _cholesky_spd_solve(M_f, m_scale, mu, None, sub.active)
        ok = block.ok & s_ok & mf_ok & jnp.all(jnp.isfinite(grad))

        def _bordered_from_c(c, d0_p, extra_rhs_f=None):
            w_c = jsp_linalg.solve_triangular(L_S, c, lower=True)
            rhs_f = mm(W.T, w_c)
            if extra_rhs_f is not None:
                rhs_f = rhs_f + extra_rhs_f
            d_f = -solve_Mf(rhs_f)
            u = jsp_linalg.solve_triangular(L_S.T, w_c + mm(W, d_f), lower=False)
            d_p = d0_p - mm(Dp_inv_Jpt, u) if n_p > 0 else d0_p
            return _merge_parameter_blocks(d_p, d_f, n, part)

        def velocity():
            if n_p == 0:
                return _bordered_from_c(sub.resid, jnp.zeros((0,), dtype=dtype))
            d0_p, use_grad = _penalized_d0()
            jp_d0 = mm(Jp_t.T, d0_p)
            c = jnp.where(use_grad, jp_d0, sub.resid + jp_d0)
            extra_f = jnp.where(use_grad, grad_f, jnp.zeros_like(grad_f))
            return _bordered_from_c(c, d0_p, extra_f)

        def solve_residual_rhs(rhs_r):
            z_p = jnp.zeros((n_p,), dtype=dtype)
            return _bordered_from_c(jnp.asarray(rhs_r, dtype=dtype), z_p)

        return PreparedStep(
            grad=grad,
            velocity=velocity,
            solve_residual_rhs=solve_residual_rhs,
            solve_ok=ok,
            make_cache=lambda valid: None,
        )


@dataclass(frozen=True)
class QR(LinearSolver):
    """Augmented and square-root Gram QR solver without forming normal equations."""

    form: str = "auto"
    preconditioner: Preconditioner | None = None

    def __post_init__(self):
        if self.form not in ("auto", "gram", "normal"):
            raise ValueError(
                f"QR.form must be 'auto', 'gram', or 'normal', got {self.form!r}"
            )
        _validate_dense_preconditioner(self.preconditioner, "QR")

    def resolve_form(self, m, n, penalized, ridge_metric, damped=True):
        if self.form != "auto":
            return self.form
        if not penalized:
            return "gram" if n > m else "normal"
        if (
            n > m
            and ridge_metric.has_shifted_solve
            and ridge_metric.fast_shifted_solve
            and ridge_metric.penalized_is_spd
            and ridge_metric.n_free_for(n) <= m
        ):
            return "gram"
        return "normal"

    def validate_shape(
        self, m, n, *, penalized, ridge_metric, damped, role="linear_solver"
    ):
        form = self.resolve_form(m, n, penalized, ridge_metric, damped=damped)
        if penalized and form == "normal" and not ridge_metric.has_factor:
            raise ValueError(
                f"QR(form='normal') in {role} requires a factored ridge_metric "
                "(IdentityRidge, DiagonalRidge, MatrixRidge, FactorRidge, or "
                "RepeatedFactorRidge)"
            )
        if penalized and form == "gram":
            if not ridge_metric.has_shifted_solve:
                raise ValueError(
                    f"QR(form='gram') in {role} requires a ridge_metric with "
                    "shifted block solves"
                )
            if not ridge_metric.penalized_is_spd:
                raise ValueError(
                    f"QR(form='gram') in {role} requires a strictly positive-definite "
                    "penalized block; use QR(form='normal'), QR(form='auto'), "
                    "CG(...), or SVD(...)"
                )
            if isinstance(self.preconditioner, DiagonalPreconditioner):
                raise ValueError(
                    "QR(form='gram') with ridge > 0 does not support "
                    "DiagonalPreconditioner; use QR(form='gram') without "
                    "preconditioner, QR(form='normal', preconditioner=...), "
                    "or Cholesky(form='gram', preconditioner=...)"
                )
            if not damped:
                n_f = ridge_metric.n_free_for(n)
                if n_f > m:
                    raise ValueError(
                        f"QR(form='gram') in {role} at zero damping requires "
                        f"n_free <= m, got n_free={n_f}, m={m}"
                    )
            return
        if damped:
            return
        if not penalized and form == "gram" and m > n:
            raise ValueError(
                f"QR(form='gram') in {role} requires m <= n at zero damping, "
                f"got m={m}, n={n}"
            )
        if form == "normal":
            eff_rows = m + ridge_metric.factor_rows_for(n) if penalized else m
            if n > eff_rows:
                raise ValueError(
                    f"QR(form='normal') in {role} requires m_aug >= n at zero damping, "
                    f"got m_aug={eff_rows}, n={n}"
                )

    def new_cache(self, m, n, ridge_metric, dtype, penalized):
        form = self.resolve_form(m, n, penalized, ridge_metric, damped=True)
        if isinstance(self.preconditioner, DiagonalPreconditioner):
            return None
        if penalized and form == "gram":
            return None
        if form == "gram":
            k = min(n, m)
            shape = (n + k, m) if m <= n else (n, k + m)
        else:
            p_rows = ridge_metric.factor_rows_for(n) if penalized else 0
            m_aug = m + p_rows
            r = min(m_aug, n)
            shape = (m_aug + n, r)
        return QRCache(
            R=jnp.zeros(shape, dtype=dtype),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            ridge=jnp.zeros((), dtype=dtype),
        )

    def prepare(self, sub):
        m, n, dtype = sub.m, sub.n, sub.dtype
        rho, mu = sub.ridge, sub.damping
        rm = sub.ridge_metric
        grad = mm(sub.Jt, sub.resid)
        if sub.penalized:
            grad = grad + rho * sub.penalty_grad
        form = self.resolve_form(m, n, sub.penalized, rm, damped=not sub.undamped)
        tiny = jnp.finfo(dtype).tiny
        eps = jnp.finfo(dtype).eps
        precond = sub.preconditioner
        diag_precond = precond if isinstance(precond, DiagonalPreconditioner) else None

        if form == "normal":
            sqrt_rho = jnp.sqrt(rho)
            if sub.penalized:
                L_p = rm.factor_matrix(n, dtype, sub.ctx)
                p_rows, n_p = L_p.shape
                part = rm.partition_indices(n)
                if part is None:
                    L_full = L_p if n_p == n else jnp.pad(L_p, ((0, 0), (0, n - n_p)))
                else:
                    p_idx, _ = part
                    L_full = jnp.zeros((p_rows, n), dtype=dtype).at[:, p_idx].set(L_p)
                A0 = jnp.concatenate([sub.Jt.T, sqrt_rho * L_full], axis=0)
                b_pen = sqrt_rho * rm.factor_apply(sub.theta, sub.ctx)
                b_vel = jnp.concatenate([sub.resid, b_pen], axis=0)
            else:
                A0 = sub.Jt.T
                b_vel = sub.resid
            m_aug = A0.shape[0]

            if diag_precond is not None:
                col_sq = jnp.sum(A0**2, axis=0)
                t_scale = diag_precond.qr_column_scale(col_sq, mu, dtype)
                A_scaled = _active_rect(A0 * t_scale[None, :], sub.active)
                if sub.undamped:
                    Q_A, R_A = jnp.linalg.qr(A_scaled, mode="reduced")
                    Q_top = Q_A[:m_aug]
                elif sub.always_damped:
                    stack = jnp.concatenate(
                        [A_scaled, jnp.sqrt(mu) * jnp.diag(t_scale)], axis=0
                    )
                    Q_s, R_A = jnp.linalg.qr(stack, mode="reduced")
                    Q_top = Q_s[:m_aug]
                else:

                    def _damped(_):
                        stack = jnp.concatenate(
                            [A_scaled, jnp.sqrt(mu) * jnp.diag(t_scale)], axis=0
                        )
                        Q_s, R_s = jnp.linalg.qr(stack, mode="reduced")
                        return Q_s[:m_aug], R_s

                    def _undamped(_):
                        Q_u, R_u = jnp.linalg.qr(A_scaled, mode="reduced")
                        if R_u.shape[0] < n:
                            pad = n - R_u.shape[0]
                            Q_u = jnp.pad(Q_u, ((0, 0), (0, pad)))
                            R_u = jnp.pad(R_u, ((0, pad), (0, 0)))
                        return Q_u[:m_aug], R_u

                    Q_top, R_A = jax.lax.cond(mu > 0, _damped, _undamped, operand=None)
                diag_abs = jnp.abs(jnp.diag(R_A))
                cutoff = jnp.where(
                    mu > 0,
                    tiny,
                    jnp.maximum(tiny, max(m_aug, n, 1) * eps * jnp.max(diag_abs)),
                )
                r_ok = (
                    (R_A.shape[0] == n)
                    & jnp.all(jnp.isfinite(R_A))
                    & jnp.all(diag_abs > cutoff)
                )
                R_work = _active_matrix(R_A, sub.active)

                def _solve_aug(b_aug):
                    z = -jsp_linalg.solve_triangular(
                        R_work, mm(Q_top.T, b_aug), lower=False
                    )
                    return t_scale * z

                return PreparedStep(
                    grad=grad,
                    velocity=lambda: _solve_aug(b_vel),
                    solve_residual_rhs=lambda rhs_r: _solve_aug(
                        jnp.pad(
                            jnp.asarray(rhs_r, dtype=dtype),
                            (0, m_aug - m),
                        )
                    ),
                    solve_ok=r_ok & jnp.all(jnp.isfinite(grad)),
                    make_cache=lambda valid: None,
                )

            r = min(m_aug, n)

            def assemble_qr():
                Q_1, R_1 = jnp.linalg.qr(_active_rect(A0, sub.active), mode="reduced")
                return jnp.concatenate([Q_1, R_1.T], axis=0)

            payload = sub.cached(assemble_qr)
            Q_1, R_1 = payload[:m_aug], payload[m_aug:].T

            if sub.undamped:
                Q_mu_top = None
                R_eff = R_1
            elif sub.always_damped:
                stack = jnp.concatenate(
                    [R_1, jnp.sqrt(mu) * jnp.eye(n, dtype=dtype)], axis=0
                )
                Q_mu, R_eff = jnp.linalg.qr(stack, mode="reduced")
                Q_mu_top = Q_mu[:r]
            else:

                def _damped_qr(_):
                    stack = jnp.concatenate(
                        [R_1, jnp.sqrt(mu) * jnp.eye(n, dtype=dtype)], axis=0
                    )
                    Q_mu, R_mu = jnp.linalg.qr(stack, mode="reduced")
                    return Q_mu[:r], R_mu

                def _undamped_qr(_):
                    R_pad = jnp.pad(R_1, ((0, n - r), (0, 0))) if r < n else R_1
                    return jnp.eye(r, n, dtype=dtype), R_pad

                Q_mu_top, R_eff = jax.lax.cond(
                    mu > 0, _damped_qr, _undamped_qr, operand=None
                )

            diag_abs = jnp.abs(jnp.diag(R_eff))
            cutoff = jnp.where(
                mu > 0,
                tiny,
                jnp.maximum(tiny, max(m_aug, n, 1) * eps * jnp.max(diag_abs)),
            )
            r_ok = (
                (R_eff.shape[0] == n)
                & jnp.all(jnp.isfinite(payload))
                & jnp.all(jnp.isfinite(R_eff))
                & jnp.all(diag_abs > cutoff)
            )
            R_work = _active_matrix(R_eff, sub.active)

            def velocity():
                rhs_1 = mm(Q_1.T, b_vel)
                rhs = rhs_1 if Q_mu_top is None else mm(Q_mu_top.T, rhs_1)
                return -jsp_linalg.solve_triangular(R_work, rhs, lower=False)

            def solve_residual_rhs(rhs_r):
                b_r = jnp.asarray(rhs_r, dtype=dtype)
                rhs_1 = mm(Q_1[:m].T, b_r)
                rhs = rhs_1 if Q_mu_top is None else mm(Q_mu_top.T, rhs_1)
                return -jsp_linalg.solve_triangular(R_work, rhs, lower=False)

            return PreparedStep(
                grad=grad,
                velocity=velocity,
                solve_residual_rhs=solve_residual_rhs,
                solve_ok=r_ok & jnp.all(jnp.isfinite(grad)),
                make_cache=lambda valid: QRCache(payload, valid, rho),
            )

        # Gram QR path
        if not sub.penalized:
            if diag_precond is not None:
                row_sq = jnp.sum(sub.Jt**2, axis=0)
                s_scale = diag_precond.qr_column_scale(row_sq, mu, dtype)
                Jt_scaled = _active_rect(sub.Jt * s_scale[None, :], sub.active)
                if sub.undamped:
                    Q_M, R_M = jnp.linalg.qr(Jt_scaled, mode="reduced")
                    Q_top = Q_M[:n]
                elif sub.always_damped:
                    stack = jnp.concatenate(
                        [Jt_scaled, jnp.sqrt(mu) * jnp.diag(s_scale)], axis=0
                    )
                    Q_s, R_M = jnp.linalg.qr(stack, mode="reduced")
                    Q_top = Q_s[:n]
                else:

                    def _damped_g(_):
                        stack = jnp.concatenate(
                            [Jt_scaled, jnp.sqrt(mu) * jnp.diag(s_scale)], axis=0
                        )
                        Q_s, R_s = jnp.linalg.qr(stack, mode="reduced")
                        return Q_s[:n], R_s

                    def _undamped_g(_):
                        if m > n:
                            return (
                                jnp.zeros((n, m), dtype=dtype),
                                jnp.zeros((m, m), dtype=dtype),
                            )
                        Q_u, R_u = jnp.linalg.qr(Jt_scaled, mode="reduced")
                        return Q_u, R_u

                    Q_top, R_M = jax.lax.cond(
                        mu > 0, _damped_g, _undamped_g, operand=None
                    )
                diag_abs = jnp.abs(jnp.diag(R_M))
                cutoff = jnp.where(
                    mu > 0,
                    tiny,
                    jnp.maximum(tiny, max(m, n, 1) * eps * jnp.max(diag_abs)),
                )
                r_ok = (
                    (R_M.shape[0] == m)
                    & jnp.all(jnp.isfinite(R_M))
                    & jnp.all(diag_abs > cutoff)
                )
                R_work = _active_matrix(R_M, sub.active)

                def solve_residual_rhs(rhs_r):
                    sc = s_scale * jnp.asarray(rhs_r, dtype=dtype)
                    w = jsp_linalg.solve_triangular(R_work.T, sc, lower=True)
                    return -mm(Q_top, w)

                return PreparedStep(
                    grad=grad,
                    velocity=lambda: solve_residual_rhs(sub.resid),
                    solve_residual_rhs=solve_residual_rhs,
                    solve_ok=r_ok & jnp.all(jnp.isfinite(grad)),
                    make_cache=lambda valid: None,
                )

            k = min(n, m)
            concat_axis = 0 if m <= n else 1

            def assemble_gram_qr():
                Q_1, R_1 = jnp.linalg.qr(
                    _active_rect(sub.Jt, sub.active), mode="reduced"
                )
                return jnp.concatenate([Q_1, R_1], axis=concat_axis)

            payload = sub.cached(assemble_gram_qr)
            Q_1, R_1 = (
                (payload[:n], payload[n:])
                if m <= n
                else (payload[:, :k], payload[:, k:])
            )

            if sub.undamped:
                Q_mu_top = None
                R_eff = R_1
            elif sub.always_damped:
                stack = jnp.concatenate(
                    [R_1, jnp.sqrt(mu) * jnp.eye(m, dtype=dtype)], axis=0
                )
                Q_mu, R_eff = jnp.linalg.qr(stack, mode="reduced")
                Q_mu_top = Q_mu[:k]
            else:

                def _damped_gram(_):
                    stack = jnp.concatenate(
                        [R_1, jnp.sqrt(mu) * jnp.eye(m, dtype=dtype)], axis=0
                    )
                    Q_mu, R_mu = jnp.linalg.qr(stack, mode="reduced")
                    return Q_mu[:k], R_mu

                def _undamped_gram(_):
                    if m > n:
                        return (
                            jnp.zeros((k, m), dtype=dtype),
                            jnp.zeros((m, m), dtype=dtype),
                        )
                    return jnp.eye(k, m, dtype=dtype), R_1

                Q_mu_top, R_eff = jax.lax.cond(
                    mu > 0, _damped_gram, _undamped_gram, operand=None
                )

            diag_abs = jnp.abs(jnp.diag(R_eff))
            cutoff = jnp.where(
                mu > 0,
                tiny,
                jnp.maximum(tiny, max(m, n, 1) * eps * jnp.max(diag_abs)),
            )
            r_ok = (
                (R_eff.shape[0] == m)
                & jnp.all(jnp.isfinite(payload))
                & jnp.all(jnp.isfinite(R_eff))
                & jnp.all(diag_abs > cutoff)
            )
            R_work = _active_matrix(R_eff, sub.active)

            def solve_residual_rhs(rhs_r):
                half = jsp_linalg.solve_triangular(
                    R_work.T, jnp.asarray(rhs_r, dtype=dtype), lower=True
                )
                mid = half if Q_mu_top is None else mm(Q_mu_top, half)
                return -mm(Q_1, mid)

            return PreparedStep(
                grad=grad,
                velocity=lambda: solve_residual_rhs(sub.resid),
                solve_residual_rhs=solve_residual_rhs,
                solve_ok=r_ok & jnp.all(jnp.isfinite(grad)),
                make_cache=lambda valid: QRCache(payload, valid, rho),
            )

        # Penalized Square-Root Gram QR (with two-stage bordered free tail when n_f > 0)
        n_f = rm.n_free_for(n)
        n_p = n - n_f
        part = rm.partition_indices(n)
        block = rm.prepare_shifted_penalized(
            n, rho, mu, dtype, sub.ctx, prefer_qr=True, active=sub.active
        )
        Jp_t, Jf_t = _split_parameter_blocks(sub.Jt, n_p, part)
        theta_p, _ = _split_parameter_blocks(sub.theta, n_p, part)
        grad_p, grad_f = _split_parameter_blocks(grad, n_p, part)

        Bt_p = block.solve_Gp_T(Jp_t) if n_p > 0 else jnp.zeros((0, m), dtype=dtype)
        stack_S = jnp.concatenate([Bt_p, jnp.eye(m, dtype=dtype)], axis=0)
        Q_S, R_S = jnp.linalg.qr(stack_S, mode="reduced")
        Q_S_top = Q_S[:n_p]
        R_S_work = _active_matrix(R_S, sub.active)
        s_ok = jnp.all(jnp.isfinite(R_S)) & jnp.all(jnp.abs(jnp.diag(R_S)) > tiny)

        if n_f == 0:
            ok = block.ok & s_ok & jnp.all(jnp.isfinite(grad))

            def velocity():
                d0_split = block.solve_d0_qr(theta_p)
                d0_grad = -block.solve_Dp(grad_p)
                use_grad = jnp.linalg.norm(d0_grad) < jnp.linalg.norm(d0_split)
                d0 = jnp.where(use_grad, d0_grad, d0_split)
                jp_d0 = mm(Jp_t.T, d0)
                c = jnp.where(use_grad, jp_d0, sub.resid + jp_d0)
                w = jsp_linalg.solve_triangular(R_S_work.T, c, lower=True)
                return d0 - block.solve_Gp(mm(Q_S_top, w))

            def solve_residual_rhs(rhs_r):
                w = jsp_linalg.solve_triangular(
                    R_S_work.T, jnp.asarray(rhs_r, dtype=dtype), lower=True
                )
                return -block.solve_Gp(mm(Q_S_top, w))

            return PreparedStep(
                grad=grad,
                velocity=velocity,
                solve_residual_rhs=solve_residual_rhs,
                solve_ok=ok,
                make_cache=lambda valid: None,
            )

        W = _active_rect(
            jsp_linalg.solve_triangular(R_S_work.T, Jf_t.T, lower=True), sub.active
        )
        if sub.undamped:
            Q_M, R_M = jnp.linalg.qr(W, mode="reduced")
            Q_M_top = Q_M[:m]
        elif sub.always_damped:
            stack_m = jnp.concatenate(
                [W, jnp.sqrt(mu) * jnp.eye(n_f, dtype=dtype)], axis=0
            )
            Q_m, R_M = jnp.linalg.qr(stack_m, mode="reduced")
            Q_M_top = Q_m[:m]
        else:

            def _damped_mf(_):
                stack_m = jnp.concatenate(
                    [W, jnp.sqrt(mu) * jnp.eye(n_f, dtype=dtype)], axis=0
                )
                Q_m, R_m = jnp.linalg.qr(stack_m, mode="reduced")
                return Q_m[:m], R_m

            def _undamped_mf(_):
                if n_f > m:
                    return (
                        jnp.zeros((m, n_f), dtype=dtype),
                        jnp.zeros((n_f, n_f), dtype=dtype),
                    )
                Q_u, R_u = jnp.linalg.qr(W, mode="reduced")
                return Q_u, R_u

            Q_M_top, R_M = jax.lax.cond(mu > 0, _damped_mf, _undamped_mf, operand=None)

        diag_M = jnp.abs(jnp.diag(R_M))
        cutoff_M = jnp.where(
            mu > 0,
            tiny,
            jnp.maximum(tiny, max(m, n_f, 1) * eps * jnp.max(diag_M)),
        )
        mf_ok = (
            (R_M.shape[0] == n_f)
            & jnp.all(jnp.isfinite(R_M))
            & jnp.all(diag_M > cutoff_M)
        )
        R_M_work = _active_matrix(R_M, sub.active)
        ok = block.ok & s_ok & mf_ok & jnp.all(jnp.isfinite(grad))

        def _qr_bordered_from_c(c, d0_p, extra_rhs_f=None):
            w_c = jsp_linalg.solve_triangular(R_S_work.T, c, lower=True)
            rhs_r = mm(Q_M_top.T, w_c)
            if extra_rhs_f is not None:
                rhs_r = rhs_r + jsp_linalg.solve_triangular(
                    R_M_work.T, extra_rhs_f, lower=True
                )
            d_f = -jsp_linalg.solve_triangular(R_M_work, rhs_r, lower=False)
            d_p = (
                d0_p - block.solve_Gp(mm(Q_S_top, w_c + mm(W, d_f)))
                if n_p > 0
                else d0_p
            )
            return _merge_parameter_blocks(d_p, d_f, n, part)

        def velocity():
            if n_p == 0:
                return _qr_bordered_from_c(sub.resid, jnp.zeros((0,), dtype=dtype))
            d0_split = block.solve_d0_qr(theta_p)
            d0_grad = -block.solve_Dp(grad_p)
            use_grad = jnp.linalg.norm(d0_grad) < jnp.linalg.norm(d0_split)
            d0_p = jnp.where(use_grad, d0_grad, d0_split)
            jp_d0 = mm(Jp_t.T, d0_p)
            c = jnp.where(use_grad, jp_d0, sub.resid + jp_d0)
            extra_f = jnp.where(use_grad, grad_f, jnp.zeros_like(grad_f))
            return _qr_bordered_from_c(c, d0_p, extra_f)

        def solve_residual_rhs(rhs_r):
            return _qr_bordered_from_c(
                jnp.asarray(rhs_r, dtype=dtype), jnp.zeros((n_p,), dtype=dtype)
            )

        return PreparedStep(
            grad=grad,
            velocity=velocity,
            solve_residual_rhs=solve_residual_rhs,
            solve_ok=ok,
            make_cache=lambda valid: None,
        )


class _KrylovConfig(LinearSolver):
    materializes_jacobian = False

    def __post_init__(self):
        if self.tol == 0 and self.atol == 0 and self.maxiter is None:
            raise ValueError("maxiter must be set when both tolerances are zero")

    def _cg(self, matvec, c, sub, apply_M):
        dtype = sub.dtype
        c = jnp.asarray(c, dtype=dtype)
        n = c.shape[0]
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
        tol = jnp.asarray(sub.hyper.iterative_tol, dtype=dtype)
        atol = jnp.asarray(sub.hyper.iterative_atol, dtype=dtype)
        maxiter = (
            jnp.asarray(10 * max(n, 1), dtype=jnp.int32)
            if sub.hyper.iterative_maxiter is None
            else jnp.asarray(sub.hyper.iterative_maxiter, dtype=jnp.int32)
        )

        c_norm = jnp.sqrt(mm(c, c))
        nz = c_norm > 0
        safe_c_norm = jnp.where(nz, c_norm, jnp.ones((), dtype=dtype))
        b = jnp.where(nz, c / safe_c_norm, c)
        eff_tol = jnp.maximum(tol, atol / safe_c_norm)
        tol_sq = jnp.square(eff_tol)

        x0 = 0.0 * b
        r0 = b
        z0 = r0 if apply_M is None else jnp.asarray(apply_M(r0), dtype=dtype)
        p0 = z0
        rz0 = mm(r0, z0)
        rr0 = mm(r0, r0)
        op0 = jnp.zeros((), dtype=dtype)
        bad0 = jnp.asarray(False, dtype=jnp.bool_)

        def cond_fn(carry):
            k, _, _, _, _, _, rr, _, bad = carry
            return nz & (k < maxiter) & (rr > tol_sq) & (~bad)

        def body_fn(carry):
            k, x, r, z, p, rz, _, op, _ = carry
            Ap = jnp.asarray(matvec(p), dtype=dtype)
            p_norm = jnp.sqrt(mm(p, p))
            Ap_norm = jnp.sqrt(mm(Ap, Ap))
            op_new = jnp.maximum(op, Ap_norm / jnp.maximum(p_norm, tiny))
            pAp = mm(p, Ap)
            bad = (~jnp.isfinite(pAp)) | (~jnp.isfinite(rz)) | (rz <= 0) | (pAp <= 0)
            safe_pAp = jnp.where(bad, jnp.ones((), dtype=dtype), pAp)
            alpha = jnp.where(bad, jnp.zeros((), dtype=dtype), rz / safe_pAp)
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            rr_new = mm(r_new, r_new)
            z_new = (
                r_new if apply_M is None else jnp.asarray(apply_M(r_new), dtype=dtype)
            )
            rz_new = mm(r_new, z_new)
            safe_rz = jnp.where(bad | (rz <= 0), jnp.ones((), dtype=dtype), rz)
            beta = jnp.where(bad, jnp.zeros((), dtype=dtype), rz_new / safe_rz)
            p_new = z_new + beta * p
            return (
                k + jnp.int32(1),
                x_new,
                r_new,
                z_new,
                p_new,
                rz_new,
                rr_new,
                op_new,
                bad,
            )

        init = (jnp.int32(0), x0, r0, z0, p0, rz0, rr0, op0, bad0)
        _, x_unit, _, _, _, _, _, op_est, bad_final = jax.lax.while_loop(
            cond_fn, body_fn, init
        )
        solution = safe_c_norm * x_unit

        if sub.ctx is not None and sub.ctx.role == "ad":
            res_vec = jnp.asarray(matvec(solution), dtype=dtype) - c
            true_res = jnp.sqrt(mm(res_vec, res_vec))
            sol_norm = jnp.sqrt(mm(solution, solution))
            req_tol = jnp.maximum(atol, tol * c_norm)
            roundoff = max(n, 1) * eps * (op_est * sol_norm + c_norm)
            bound = 10.0 * (req_tol + roundoff)
            ok = jnp.all(jnp.isfinite(solution)) & (~bad_final) & (true_res <= bound)
            if sub.active is not True:
                ok = (~sub.active) | ok
            scale = jnp.where(
                ok,
                jnp.ones((), dtype=dtype),
                jnp.asarray(jnp.nan, dtype=dtype),
            )
            solution = scale * solution
        return solution


@dataclass(frozen=True)
class CG(_KrylovConfig):
    """Matrix-free PCG on the parameter-space operator ``J'J + rho Q + mu I``."""

    preconditioner: Preconditioner | None = IdentityPreconditioner()
    tol: float | None = None
    atol: float = 0.0
    maxiter: int | None = None
    penalty: float | None = None

    def validate_shape(
        self, m, n, *, penalized, ridge_metric, damped, role="linear_solver"
    ):
        if damped or self.penalty is not None:
            return
        if not penalized and n > m:
            raise ValueError(
                f"{role}=CG(...) requires n <= m at zero damping, got m={m}, n={n}. "
                "Use GramCG(...) for wide problems"
            )
        if penalized and ridge_metric.n_free_for(n) > m:
            raise ValueError(
                f"{role}=CG(...) at zero damping requires n_free <= m, "
                f"got n_free={ridge_metric.n_free_for(n)}, m={m}"
            )

    def prepare(self, sub):
        dtype = sub.dtype
        rho, mu = sub.ridge, sub.damping
        rm = sub.ridge_metric
        grad = sub.JT(sub.resid)
        if sub.penalized:
            grad = grad + rho * sub.penalty_grad
        extra_pen = (
            jnp.zeros((), dtype=dtype)
            if self.penalty is None or not sub.undamped
            else jnp.asarray(self.penalty, dtype=dtype)
        )
        total_shift = mu + extra_pen

        def H_matvec(v):
            v = jnp.asarray(v, dtype=dtype)
            out = sub.JT(sub.jvp_fn(v)) + total_shift * v
            if sub.penalized:
                out = out + rho * jnp.asarray(rm.matvec(v, sub.ctx), dtype=dtype)
            if sub.active is not True:
                out = jnp.where(sub.active, out, v)
            return out

        precond = sub.preconditioner
        if precond is None or isinstance(precond, IdentityPreconditioner):
            apply_M = None
        else:

            def apply_M(v):
                mv = precond.apply(v, total_shift, sub.ctx)
                if sub.active is not True:
                    mv = jnp.where(sub.active, mv, v)
                return mv

        def solve_H(_, rhs):
            return self._cg(H_matvec, rhs, sub, apply_M)

        def solve_parameter_rhs(rhs_p):
            return jax.lax.custom_linear_solve(
                H_matvec,
                jnp.asarray(rhs_p, dtype=dtype),
                solve=solve_H,
                transpose_solve=solve_H,
                symmetric=True,
            )

        def velocity():
            return solve_parameter_rhs(-grad)

        def solve_residual_rhs(rhs_r):
            return solve_parameter_rhs(-sub.JT(jnp.asarray(rhs_r, dtype=dtype)))

        return PreparedStep(
            grad=grad,
            velocity=velocity,
            solve_residual_rhs=solve_residual_rhs,
            solve_ok=jnp.all(jnp.isfinite(grad)),
            make_cache=lambda valid: None,
        )


@dataclass(frozen=True)
class GramCG(_KrylovConfig):
    """Matrix-free preconditioned conjugate gradients in residual space."""

    preconditioner: Preconditioner | None = IdentityPreconditioner()
    tol: float | None = None
    atol: float = 0.0
    maxiter: int | None = None

    def validate_shape(
        self, m, n, *, penalized, ridge_metric, damped, role="linear_solver"
    ):
        if penalized:
            if not ridge_metric.has_shifted_solve:
                raise ValueError(
                    f"{role}=GramCG(...) requires a ridge_metric with shifted "
                    "block solves"
                )
            if ridge_metric.n_free_for(n) > 0:
                raise ValueError(
                    f"{role}=GramCG(...) with ridge > 0 requires no unpenalized "
                    "free coordinates; use CG(...) when free coordinates are present"
                )
            if not ridge_metric.penalized_is_spd:
                raise ValueError(
                    f"{role}=GramCG(...) with ridge > 0 requires a strictly "
                    "positive-definite ridge_metric; use CG(...) for "
                    "semidefinite penalties"
                )
            return
        if not damped and m > n:
            raise ValueError(
                f"{role}=GramCG(...) requires m <= n at zero damping, "
                f"got m={m}, n={n}. Use CG(...) for tall problems"
            )

    def prepare(self, sub):
        n, dtype = sub.n, sub.dtype
        rho, mu = sub.ridge, sub.damping
        rm = sub.ridge_metric
        grad = sub.JT(sub.resid)
        if sub.penalized:
            grad = grad + rho * sub.penalty_grad
        precond = sub.preconditioner
        if precond is None or isinstance(precond, IdentityPreconditioner):
            apply_M = None
        else:

            def apply_M(v):
                mv = precond.apply(v, mu, sub.ctx)
                if sub.active is not True:
                    mv = jnp.where(sub.active, mv, v)
                return mv

        if not sub.penalized:

            def dual_matvec(y):
                y = jnp.asarray(y, dtype=dtype)
                out = sub.jvp_fn(sub.JT(y)) + mu * y
                if sub.active is not True:
                    out = jnp.where(sub.active, out, y)
                return out

            def solve_dual(_, rhs):
                return self._cg(dual_matvec, rhs, sub, apply_M)

            def solve_S(rhs_r):
                return jax.lax.custom_linear_solve(
                    dual_matvec,
                    jnp.asarray(rhs_r, dtype=dtype),
                    solve=solve_dual,
                    transpose_solve=solve_dual,
                    symmetric=True,
                )

            def solve_residual_rhs(rhs_r):
                return -sub.JT(solve_S(rhs_r))

            return PreparedStep(
                grad=grad,
                velocity=lambda: solve_residual_rhs(sub.resid),
                solve_residual_rhs=solve_residual_rhs,
                solve_ok=jnp.all(jnp.isfinite(grad)),
                make_cache=lambda valid: None,
            )

        block = rm.prepare_shifted_penalized(
            n, rho, mu, dtype, sub.ctx, prefer_qr=False, active=sub.active
        )

        def solve_D(v):
            return block.solve_Dp(jnp.asarray(v, dtype=dtype))

        def S_matvec(y):
            y = jnp.asarray(y, dtype=dtype)
            out = y + sub.jvp_fn(solve_D(sub.JT(y)))
            if sub.active is not True:
                out = jnp.where(sub.active, out, y)
            return out

        def solve_S_cg(_, rhs):
            return self._cg(S_matvec, rhs, sub, apply_M)

        def solve_S(rhs_r):
            return jax.lax.custom_linear_solve(
                S_matvec,
                jnp.asarray(rhs_r, dtype=dtype),
                solve=solve_S_cg,
                transpose_solve=solve_S_cg,
                symmetric=True,
            )

        def velocity():
            d0 = -solve_D(rho * sub.penalty_grad)
            c = sub.resid + sub.jvp_fn(d0)
            return d0 - solve_D(sub.JT(solve_S(c)))

        def solve_residual_rhs(rhs_r):
            return -solve_D(sub.JT(solve_S(rhs_r)))

        ok = block.ok & jnp.all(jnp.isfinite(grad))
        return PreparedStep(
            grad=grad,
            velocity=velocity,
            solve_residual_rhs=solve_residual_rhs,
            solve_ok=ok,
            make_cache=lambda valid: None,
        )


@dataclass(frozen=True)
class LU(LinearSolver):
    """Direct LU factorization for square unpenalized Newton and implicit-AD systems."""

    supports_forward = False
    supports_penalty = False

    def validate_shape(
        self, m, n, *, penalized, ridge_metric, damped, role="linear_solver"
    ):
        if m != n:
            raise ValueError(
                f"{role}=LU() requires a square system (m == n), got m={m}, n={n}"
            )

    def prepare(self, sub):
        m, n, dtype = sub.m, sub.n, sub.dtype
        grad = mm(sub.Jt, sub.resid)
        J_mat = _active_matrix(sub.Jt.T, sub.active)
        lu, piv = jsp_linalg.lu_factor(J_mat)
        tiny = jnp.finfo(dtype).tiny
        diag_abs = jnp.abs(jnp.diag(lu))
        cutoff = jnp.maximum(
            tiny, max(m, n, 1) * jnp.finfo(dtype).eps * jnp.max(diag_abs)
        )
        lu_ok = (
            (sub.damping == 0)
            & jnp.all(jnp.isfinite(lu))
            & jnp.all(diag_abs > cutoff)
            & jnp.all(jnp.isfinite(grad))
        )

        def solve_residual_rhs(rhs_r):
            return -jsp_linalg.lu_solve((lu, piv), jnp.asarray(rhs_r, dtype=dtype))

        return PreparedStep(
            grad=grad,
            velocity=lambda: solve_residual_rhs(sub.resid),
            solve_residual_rhs=solve_residual_rhs,
            solve_ok=lu_ok,
            make_cache=lambda valid: None,
        )


@dataclass(frozen=True)
class SVD(LinearSolver):
    """Rank-aware SVD solver selecting the minimum-norm augmented LS step."""

    rcond: float | None = None

    def validate_shape(
        self, m, n, *, penalized, ridge_metric, damped, role="linear_solver"
    ):
        if penalized and not ridge_metric.has_factor:
            raise ValueError(
                f"{role}=SVD(...) with ridge > 0 requires a factored ridge_metric"
            )

    def prepare(self, sub):
        m, n, dtype = sub.m, sub.n, sub.dtype
        rho, mu = sub.ridge, sub.damping
        rm = sub.ridge_metric
        grad = mm(sub.Jt, sub.resid)
        if sub.penalized:
            grad = grad + rho * sub.penalty_grad
            sqrt_rho = jnp.sqrt(rho)
            L_p = rm.factor_matrix(n, dtype, sub.ctx)
            p_rows, n_p = L_p.shape
            part = rm.partition_indices(n)
            if part is None:
                L_full = L_p if n_p == n else jnp.pad(L_p, ((0, 0), (0, n - n_p)))
            else:
                p_idx, _ = part
                L_full = jnp.zeros((p_rows, n), dtype=dtype).at[:, p_idx].set(L_p)
            At_aug = jnp.concatenate([sub.Jt, sqrt_rho * L_full.T], axis=1)
            b_pen = sqrt_rho * rm.factor_apply(sub.theta, sub.ctx)
            b_vel = jnp.concatenate([sub.resid, b_pen], axis=0)
        else:
            At_aug = sub.Jt
            b_vel = sub.resid

        if sub.undamped:
            pinv = _svd_pinv(At_aug, self.rcond)
            ok = jnp.all(jnp.isfinite(pinv)) & jnp.all(jnp.isfinite(grad))

            def velocity():
                return -mm(pinv, b_vel)

            def solve_residual_rhs(rhs_r):
                return -mm(pinv[:, :m], jnp.asarray(rhs_r, dtype=dtype))

            return PreparedStep(
                grad=grad,
                velocity=velocity,
                solve_residual_rhs=solve_residual_rhs,
                solve_ok=ok,
                make_cache=lambda valid: None,
            )

        U, sigma, Vt = jnp.linalg.svd(At_aug.T, full_matrices=False)
        damped_inv = sigma / (sigma**2 + mu)
        if sub.always_damped:
            inv_sigma = damped_inv
        else:
            eps = jnp.finfo(dtype).eps
            rel = (
                max(At_aug.shape) * eps
                if self.rcond is None
                else jnp.asarray(self.rcond, dtype=dtype)
            )
            cutoff = rel * jnp.where(
                sigma.size > 0, sigma[0], jnp.zeros((), dtype=dtype)
            )
            mask = sigma > cutoff
            safe_sigma = jnp.where(mask, sigma, jnp.ones_like(sigma))
            undamped_inv = jnp.where(mask, 1.0 / safe_sigma, jnp.zeros_like(sigma))
            inv_sigma = jnp.where(mu > 0, damped_inv, undamped_inv)
        ok = (
            jnp.all(jnp.isfinite(U))
            & jnp.all(jnp.isfinite(sigma))
            & jnp.all(jnp.isfinite(Vt))
            & jnp.all(jnp.isfinite(grad))
        )

        def _apply_pinv(b_aug):
            return -mm(Vt.T, inv_sigma * mm(U.T, b_aug))

        def solve_residual_rhs(rhs_r):
            b_r = jnp.asarray(rhs_r, dtype=dtype)
            return -mm(Vt.T, inv_sigma * mm(U[:m].T, b_r))

        return PreparedStep(
            grad=grad,
            velocity=lambda: _apply_pinv(b_vel),
            solve_residual_rhs=solve_residual_rhs,
            solve_ok=ok,
            make_cache=lambda valid: None,
        )
