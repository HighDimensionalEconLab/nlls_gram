import functools

import jax
import jax.numpy as jnp
import numpy as np

from nlls_gram.types import BoxSolveResult, LagrangeMultipliers, SolverStatus
from nlls_gram.utilities import (
    _broadcast_leading_condition,
    _static_key_component,
    _zero_tangent_leaf,
    canonicalize_residual,
)

__all__ = ["BoxSecant"]


def _default_xtol(dtype):
    return float(jnp.finfo(dtype).eps) ** 0.8


def _default_rtol(dtype):
    return 4.0 * float(jnp.finfo(dtype).eps)


def _default_gtol(dtype):
    return float(jnp.finfo(dtype).eps) ** 0.8


def _explicit_float_dtypes(tree):
    dtypes = []
    for leaf in jax.tree.leaves(tree):
        if (
            isinstance(leaf, (jax.Array, np.ndarray, np.generic))
            and jnp.issubdtype(leaf.dtype, jnp.floating)
            and not getattr(leaf, "weak_type", False)
        ):
            dtypes.append(leaf.dtype)
    return dtypes


def _infer_box_dtype(lower, upper, x0, args, p):
    dtypes = _explicit_float_dtypes((lower, upper, x0, args, p))
    if dtypes:
        return jnp.result_type(*dtypes)
    return jnp.result_type(float)


def _align_batched_leaf(is_b, leaf, coord_v):
    if is_b and jnp.ndim(leaf) < coord_v.ndim:
        leaf = jnp.reshape(
            leaf,
            (coord_v.shape[0],)
            + (1,) * (coord_v.ndim - jnp.ndim(leaf))
            + leaf.shape[1:],
        )
    return jnp.broadcast_to(leaf, coord_v.shape)


def _safe_primal_tree(ad_success, tree):
    def select(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        if not jnp.issubdtype(arr.dtype, jnp.inexact):
            return leaf
        if ad_success.ndim == 0 or (
            arr.ndim >= ad_success.ndim
            and arr.shape[: ad_success.ndim] == ad_success.shape
        ):
            cond = _broadcast_leading_condition(ad_success, arr)
        else:
            cond = jnp.any(ad_success)
        return jnp.where(cond, arr, jax.lax.stop_gradient(arr))

    return jax.tree.map(select, tree)


def _safe_tangent_tree(ad_success, tangent):
    def mask(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        if arr.dtype == jax.dtypes.float0:
            return leaf
        if ad_success.ndim == 0 or (
            arr.ndim >= ad_success.ndim
            and arr.shape[: ad_success.ndim] == ad_success.shape
        ):
            cond = _broadcast_leading_condition(ad_success, arr)
        else:
            cond = jnp.any(ad_success)
        return jnp.where(cond, arr, jnp.zeros_like(arr))

    return jax.tree.map(mask, tangent)


def _secant_curv(x, g, xp, gp):
    dx = x - xp
    safe_dx = jnp.where(dx != 0, dx, jnp.ones_like(dx))
    return jnp.where(dx != 0, (g - gp) / safe_dx, jnp.zeros_like(g))


def _kkt_converged(x, a0, b0, lo, hi, g, curv, dx_abs, xtol, rtol, gtol):
    scale = jnp.maximum(jnp.abs(x), 0.5 * (jnp.abs(lo) + jnp.abs(hi)))
    xtol_eff = xtol + rtol * scale
    at_lo_ok = (x <= a0) & ((g > gtol) | ((g >= -gtol) & (curv >= 0)))
    at_hi_ok = (x >= b0) & ((g < -gtol) | ((g <= gtol) & (curv >= 0)))
    grad_ok = (gtol > 0) & (jnp.abs(g) <= gtol) & (curv >= 0)
    bracket_ok = (xtol_eff > 0) & ((hi - lo) <= xtol_eff)
    step_ok = (
        (xtol_eff > 0)
        & (dx_abs <= xtol_eff)
        & ((gtol <= 0) | (jnp.abs(g) <= jnp.sqrt(gtol)))
        & (curv >= 0)
    )
    return at_lo_ok | at_hi_ok | grad_ok | bracket_ok | step_ok


def _safeguarded_step(x, a0, b0, lo, hi, g, curv, prev_span, tried_lo, tried_hi):
    lo_n = jnp.where(g < 0, jnp.maximum(lo, x), lo)
    hi_n = jnp.where(g > 0, jnp.minimum(hi, x), hi)
    mid = 0.5 * (lo_n + hi_n)
    quarter = lo_n + jnp.asarray(0.25, dtype=x.dtype) * (hi_n - lo_n)
    bis = jnp.where((mid == x) & (hi_n > lo_n), quarter, mid)

    pos_curv = curv > 0
    safe_curv = jnp.where(pos_curv, curv, jnp.ones_like(curv))
    raw_step = -g / safe_curv
    x_proj = jnp.clip(x + raw_step, lo_n, hi_n)

    strictly_in = (x_proj > lo_n) & (x_proj < hi_n)
    first_lo_hit = (x_proj == a0) & (x > a0) & (lo_n == a0) & (~tried_lo)
    first_hi_hit = (x_proj == b0) & (x < b0) & (hi_n == b0) & (~tried_hi)
    fast_enough = jnp.abs(x_proj - x) <= 0.5 * prev_span
    use_secant = pos_curv & ((strictly_in & fast_enough) | first_lo_hit | first_hi_hit)

    can_jump_lo = (~tried_lo) & (~pos_curv) & (g >= 0) & (lo_n == a0) & (x > a0)
    can_jump_hi = (~tried_hi) & (~pos_curv) & (g <= 0) & (hi_n == b0) & (x < b0)
    fallback = jnp.where(can_jump_lo, a0, jnp.where(can_jump_hi, b0, bis))

    x_next = jnp.where(use_secant, x_proj, fallback)
    return x_next, lo_n, hi_n, hi - lo


def _run_secant_loop(
    vg_fn, a0, b0, x0_in, xtol, rtol, gtol, max_steps, sign, use_fori, has_x0
):
    dtype = a0.dtype
    xtol_a = jnp.asarray(xtol, dtype=dtype)
    rtol_a = jnp.asarray(rtol, dtype=dtype)
    gtol_a = jnp.asarray(gtol, dtype=dtype)
    max_s_lane = jnp.broadcast_to(jnp.asarray(max_steps, dtype=jnp.int32), a0.shape)
    max_s = jnp.max(max_s_lane) if a0.size > 0 else jnp.asarray(0, dtype=jnp.int32)
    span0 = jnp.maximum(b0 - a0, jnp.zeros_like(a0))
    b_clip = jnp.maximum(a0, b0)

    if has_x0:
        one = jnp.ones_like(a0)
        x_init = jnp.clip(jnp.asarray(x0_in, dtype=dtype), a0, b_clip)
        f_init, g_init = vg_fn(x_init)
        delta = jnp.asarray(0.03125, dtype=dtype) * span0
        inward_dir = jnp.where(x_init - a0 <= b_clip - x_init, one, -one)
        desc_dir = jnp.where(g_init > 0, -one, jnp.where(g_init < 0, one, inward_dir))
        cand_second = x_init + desc_dir * delta
        out_of_box = (cand_second < a0) | (cand_second > b_clip)
        x_second = jnp.where(
            out_of_box, jnp.clip(x_init - desc_dir * delta, a0, b_clip), cand_second
        )
        f_second, g_second = vg_fn(x_second)
        second_end_kkt = ((x_second <= a0) & (g_second >= -gtol_a)) | (
            (x_second >= b_clip) & (g_second <= gtol_a)
        )
        pick_second = (f_second < f_init) | (
            (f_second == f_init) & second_end_kkt & (x_init > a0) & (x_init < b_clip)
        )
        x_start = jnp.where(pick_second, x_second, x_init)
        f_start = jnp.where(pick_second, f_second, f_init)
        g_start = jnp.where(pick_second, g_second, g_init)
        xp_start = jnp.where(pick_second, x_init, x_second)
        gp_start = jnp.where(pick_second, g_init, g_second)
    else:
        x_prev0 = a0 + jnp.asarray(0.375, dtype=dtype) * span0
        x0 = a0 + jnp.asarray(0.625, dtype=dtype) * span0
        f_prev0, g_prev0 = vg_fn(x_prev0)
        f0, g0 = vg_fn(x0)
        pick0 = f0 <= f_prev0
        x_start = jnp.where(pick0, x0, x_prev0)
        f_start = jnp.where(pick0, f0, f_prev0)
        g_start = jnp.where(pick0, g0, g_prev0)
        xp_start = jnp.where(pick0, x_prev0, x0)
        gp_start = jnp.where(pick0, g_prev0, g0)

    dx_inf = jnp.full_like(a0, jnp.inf)
    steps0 = jnp.zeros(a0.shape, dtype=jnp.int32)
    tried_lo0 = (x_start <= a0) | (xp_start <= a0)
    tried_hi0 = (x_start >= b0) | (xp_start >= b0)

    if use_fori:

        def body_fori(_, carry):
            (
                x,
                f_val,
                g,
                xp,
                gp,
                lo,
                hi,
                prev_span,
                dx_abs,
                f_best,
                x_best,
                g_best,
                tried_lo,
                tried_hi,
                steps,
            ) = carry
            curv = _secant_curv(x, g, xp, gp)
            conv = _kkt_converged(
                x, a0, b0, lo, hi, g, curv, dx_abs, xtol_a, rtol_a, gtol_a
            )
            active = (
                (~conv)
                & (steps < max_s_lane)
                & jnp.isfinite(f_val)
                & jnp.isfinite(g)
                & (a0 < b0)
            )
            x_cand, lo_n, hi_n, span_n = _safeguarded_step(
                x, a0, b0, lo, hi, g, curv, prev_span, tried_lo, tried_hi
            )
            x_eval = jnp.where(active, x_cand, x)
            f_n, g_n = vg_fn(x_eval)
            cand_fin = jnp.isfinite(f_n) & jnp.isfinite(g_n)
            hit_end = (x_cand <= a0) | (x_cand >= b0)
            accept_eval = active & (cand_fin | (~hit_end))
            better = accept_eval & cand_fin & (f_n <= f_best)
            return (
                jnp.where(accept_eval, x_cand, x),
                jnp.where(accept_eval, f_n, f_val),
                jnp.where(accept_eval, g_n, g),
                jnp.where(accept_eval, x, xp),
                jnp.where(accept_eval, g, gp),
                jnp.where(active, lo_n, lo),
                jnp.where(active, hi_n, hi),
                jnp.where(active, span_n, prev_span),
                jnp.where(accept_eval, jnp.abs(x_cand - x), dx_abs),
                jnp.where(better, f_n, f_best),
                jnp.where(better, x_eval, x_best),
                jnp.where(better, g_n, g_best),
                jnp.where(active, tried_lo | (x_cand <= a0), tried_lo),
                jnp.where(active, tried_hi | (x_cand >= b0), tried_hi),
                jnp.where(active, steps + 1, steps),
            )

        (
            x,
            f_val,
            g,
            xp,
            gp,
            lo,
            hi,
            _,
            dx_abs,
            f_best,
            x_best,
            g_best,
            _,
            _,
            steps,
        ) = jax.lax.fori_loop(
            0,
            max_s,
            body_fori,
            (
                x_start,
                f_start,
                g_start,
                xp_start,
                gp_start,
                a0,
                b0,
                span0,
                dx_inf,
                f_start,
                x_start,
                g_start,
                tried_lo0,
                tried_hi0,
                steps0,
            ),
        )
    else:

        def cond_while(carry):
            (
                x,
                f_val,
                g,
                xp,
                gp,
                lo,
                hi,
                _,
                dx_abs,
                _,
                _,
                _,
                _,
                _,
                steps,
                loop_s,
            ) = carry
            curv = _secant_curv(x, g, xp, gp)
            conv = _kkt_converged(
                x, a0, b0, lo, hi, g, curv, dx_abs, xtol_a, rtol_a, gtol_a
            )
            active = (
                (~conv)
                & (steps < max_s_lane)
                & jnp.isfinite(f_val)
                & jnp.isfinite(g)
                & (a0 < b0)
            )
            return (loop_s < max_s) & jnp.any(active)

        def body_while(carry):
            (
                x,
                f_val,
                g,
                xp,
                gp,
                lo,
                hi,
                prev_span,
                dx_abs,
                f_best,
                x_best,
                g_best,
                tried_lo,
                tried_hi,
                steps,
                loop_s,
            ) = carry
            curv = _secant_curv(x, g, xp, gp)
            conv = _kkt_converged(
                x, a0, b0, lo, hi, g, curv, dx_abs, xtol_a, rtol_a, gtol_a
            )
            active = (
                (~conv)
                & (steps < max_s_lane)
                & jnp.isfinite(f_val)
                & jnp.isfinite(g)
                & (a0 < b0)
            )
            x_cand, lo_n, hi_n, span_n = _safeguarded_step(
                x, a0, b0, lo, hi, g, curv, prev_span, tried_lo, tried_hi
            )
            x_eval = jnp.where(active, x_cand, x)
            f_n, g_n = vg_fn(x_eval)
            cand_fin = jnp.isfinite(f_n) & jnp.isfinite(g_n)
            hit_end = (x_cand <= a0) | (x_cand >= b0)
            accept_eval = active & (cand_fin | (~hit_end))
            better = accept_eval & cand_fin & (f_n <= f_best)
            return (
                jnp.where(accept_eval, x_cand, x),
                jnp.where(accept_eval, f_n, f_val),
                jnp.where(accept_eval, g_n, g),
                jnp.where(accept_eval, x, xp),
                jnp.where(accept_eval, g, gp),
                jnp.where(active, lo_n, lo),
                jnp.where(active, hi_n, hi),
                jnp.where(active, span_n, prev_span),
                jnp.where(accept_eval, jnp.abs(x_cand - x), dx_abs),
                jnp.where(better, f_n, f_best),
                jnp.where(better, x_eval, x_best),
                jnp.where(better, g_n, g_best),
                jnp.where(active, tried_lo | (x_cand <= a0), tried_lo),
                jnp.where(active, tried_hi | (x_cand >= b0), tried_hi),
                jnp.where(active, steps + 1, steps),
                loop_s + jnp.asarray(1, dtype=jnp.int32),
            )

        (
            x,
            f_val,
            g,
            xp,
            gp,
            lo,
            hi,
            _,
            dx_abs,
            f_best,
            x_best,
            g_best,
            _,
            _,
            steps,
            _,
        ) = jax.lax.while_loop(
            cond_while,
            body_while,
            (
                x_start,
                f_start,
                g_start,
                xp_start,
                gp_start,
                a0,
                b0,
                span0,
                dx_inf,
                f_start,
                x_start,
                g_start,
                tried_lo0,
                tried_hi0,
                steps0,
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )

    curv_end = _secant_curv(x, g, xp, gp)
    conv = _kkt_converged(
        x, a0, b0, lo, hi, g, curv_end, dx_abs, xtol_a, rtol_a, gtol_a
    ) | (a0 >= b0)
    deg = a0 >= b0
    # A converged finite last iterate is returned, so the status describes the
    # returned point. The best visited point is the fallback.
    take_curr = (conv & jnp.isfinite(f_val) & jnp.isfinite(g)) | (f_val <= f_best)
    x_out = jnp.where(deg, a0, jnp.where(take_curr, x, x_best))
    f_out = jnp.where(take_curr, f_val, f_best)
    g_out = jnp.where(take_curr, g, g_best)
    bw_out = jnp.where(
        deg, jnp.zeros_like(a0), jnp.maximum(hi - lo, jnp.zeros_like(a0))
    )
    finite = (
        jnp.isfinite(f_out) & jnp.isfinite(g_out) & jnp.isfinite(a0) & jnp.isfinite(b0)
    )
    status = jnp.where(
        ~finite,
        jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32),
        jnp.where(
            conv,
            jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32),
            jnp.asarray(SolverStatus.MAX_STEPS, dtype=jnp.int32),
        ),
    )
    return x_out, sign * f_out, sign * g_out, bw_out, steps, status


def _make_secant_primal(jet_fn, sign_float, use_fori, has_x0):
    @jax.custom_batching.custom_vmap
    def solve_level(lower, upper, x0, args, p, xtol, rtol, gtol, max_steps):
        sign = jnp.asarray(sign_float, dtype=lower.dtype)
        return _run_secant_loop(
            lambda x: jet_fn(x, args, p, sign),
            lower,
            upper,
            x0,
            xtol,
            rtol,
            gtol,
            max_steps,
            sign,
            use_fori,
            has_x0,
        )

    @solve_level.def_vmap
    def solve_level_vmap(
        axis_size,
        in_batched,
        lower,
        upper,
        x0,
        args,
        p,
        xtol,
        rtol,
        gtol,
        max_steps,
    ):
        b_lo, b_hi, b_x0, b_args, b_p, b_xtol, b_rtol, b_gtol, b_max = in_batched
        lo_v = (
            lower if b_lo else jnp.broadcast_to(lower, (axis_size, *jnp.shape(lower)))
        )
        hi_v = (
            upper if b_hi else jnp.broadcast_to(upper, (axis_size, *jnp.shape(upper)))
        )
        if has_x0:
            x0_v = x0 if b_x0 else jnp.broadcast_to(x0, (axis_size, *jnp.shape(x0)))
        else:
            x0_v = None
        xtol_v = _align_batched_leaf(b_xtol, xtol, lo_v)
        rtol_v = _align_batched_leaf(b_rtol, rtol, lo_v)
        gtol_v = _align_batched_leaf(b_gtol, gtol, lo_v)
        max_v = _align_batched_leaf(b_max, max_steps, lo_v)
        args_axes = jax.tree.map(lambda is_b: 0 if is_b else None, b_args)
        p_axes = jax.tree.map(lambda is_b: 0 if is_b else None, b_p)
        next_jet = jax.vmap(jet_fn, in_axes=(0, args_axes, p_axes, None))
        next_solver = _make_secant_primal(next_jet, sign_float, use_fori, has_x0)
        out = next_solver(lo_v, hi_v, x0_v, args, p, xtol_v, rtol_v, gtol_v, max_v)
        return out, (True, True, True, True, True, True)

    return solve_level


def _solve_box_secant(
    solver,
    use_fori,
    has_x0,
    max_steps_is_success,
    num_starts,
    lo,
    hi,
    x0,
    args,
    p,
    xtol,
    rtol,
    gtol,
    max_steps,
):
    fn_canon = solver._canonical_fn
    has_aux = solver.has_aux
    sign_float = solver._sign

    closed_fn, fn_out_struct = jax.make_jaxpr(fn_canon, return_shape=True)(lo, args, p)
    fn_out_tree = jax.tree.structure(fn_out_struct)
    fn_jaxpr = closed_fn.jaxpr
    fn_consts = tuple(closed_fn.consts)

    def fn_pure(z, a_val, c_vals, p_val):
        flat_in = jax.tree.leaves((z, a_val, p_val))
        flat_out = jax.core.eval_jaxpr(fn_jaxpr, c_vals, *flat_in)
        return jax.tree.unflatten(fn_out_tree, flat_out)

    def f_only(z, ac_val, p_val):
        a_val, c_vals = ac_val
        out = fn_pure(z, a_val, c_vals, p_val)
        val = out[0] if has_aux else out
        return jnp.asarray(val, dtype=z.dtype)

    def scalar_jet(x, ac_val, p_val, sign):
        return jax.jvp(
            lambda z: sign * f_only(z, ac_val, p_val), (x,), (jnp.ones_like(x),)
        )

    if num_starts == 1:
        primal_fn = _make_secant_primal(scalar_jet, sign_float, use_fori, has_x0)
    else:
        primal_sub_fn = _make_secant_primal(scalar_jet, sign_float, use_fori, False)

    @jax.custom_jvp
    def solve_ad(
        lo_in,
        hi_in,
        x0_in,
        args_in,
        consts_in,
        p_in,
        xtol_in,
        rtol_in,
        gtol_in,
        max_s_in,
    ):
        ac_in = (args_in, consts_in)
        if num_starts == 1:
            x_star, f_star, g_star, bw_star, steps, status = primal_fn(
                lo_in, hi_in, x0_in, ac_in, p_in, xtol_in, rtol_in, gtol_in, max_s_in
            )
        else:
            dtype = lo_in.dtype
            sign = jnp.asarray(sign_float, dtype=dtype)
            grid = jnp.linspace(
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
                num_starts + 1,
            )
            grid_exp = jnp.reshape(grid, (num_starts + 1,) + (1,) * lo_in.ndim)
            span = jnp.maximum(hi_in - lo_in, jnp.zeros_like(lo_in))
            edges = (
                (lo_in[None, ...] + grid_exp * span[None, ...])
                .at[0]
                .set(lo_in)
                .at[num_starts]
                .set(hi_in)
            )
            sub_x, sub_f, sub_g, sub_bw, sub_steps, sub_status = jax.vmap(
                lambda l_k, h_k: primal_sub_fn(
                    l_k, h_k, None, ac_in, p_in, xtol_in, rtol_in, gtol_in, max_s_in
                )
            )(edges[:-1], edges[1:])

            if has_x0:
                x0_clip = jnp.clip(x0_in, lo_in, hi_in)
                pts_extra = jnp.concatenate([x0_clip[None, ...], edges], axis=0)
            else:
                pts_extra = edges

            h_extra, g_h_extra = jax.vmap(lambda z: scalar_jet(z, ac_in, p_in, sign))(
                pts_extra
            )
            f_extra = sign * h_extra
            g_extra = sign * g_h_extra
            bw_extra = jnp.zeros_like(pts_extra)

            gtol_b = jnp.broadcast_to(gtol_in, lo_in.shape)[None, ...]
            lo_b = lo_in[None, ...]
            hi_b = hi_in[None, ...]
            deg_extra = lo_b >= hi_b
            lo_kkt = (pts_extra <= lo_b) & (g_h_extra >= -gtol_b)
            hi_kkt = (pts_extra >= hi_b) & (g_h_extra <= gtol_b)
            int_kkt = (gtol_b > 0) & (jnp.abs(g_h_extra) <= gtol_b)
            fin_extra = (
                jnp.isfinite(h_extra)
                & jnp.isfinite(g_h_extra)
                & jnp.isfinite(lo_b)
                & jnp.isfinite(hi_b)
            )
            conv_extra = deg_extra | lo_kkt | hi_kkt | int_kkt
            status_extra = jnp.where(
                ~fin_extra,
                jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32),
                jnp.where(
                    conv_extra,
                    jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32),
                    jnp.asarray(SolverStatus.MAX_STEPS, dtype=jnp.int32),
                ),
            )

            at_internal_seam = ((sub_x <= edges[:-1]) & (edges[:-1] > lo_b)) | (
                (sub_x >= edges[1:]) & (edges[1:] < hi_b)
            )
            seam_grad_ok = (gtol_b > 0) & (jnp.abs(sign * sub_g) <= gtol_b)
            conv_code = jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32)
            max_code = jnp.asarray(SolverStatus.MAX_STEPS, dtype=jnp.int32)
            sub_status_outer = jnp.where(
                at_internal_seam & (~seam_grad_ok) & (sub_status == conv_code),
                max_code,
                sub_status,
            )

            cand_x = jnp.concatenate([pts_extra, sub_x], axis=0)
            cand_h = jnp.concatenate([h_extra, sign * sub_f], axis=0)
            cand_f = jnp.concatenate([f_extra, sub_f], axis=0)
            cand_g = jnp.concatenate([g_extra, sub_g], axis=0)
            cand_bw = jnp.concatenate([bw_extra, sub_bw], axis=0)
            cand_status = jnp.concatenate([status_extra, sub_status_outer], axis=0)

            cand_fin = cand_status != jnp.asarray(
                SolverStatus.NONFINITE, dtype=jnp.int32
            )
            any_fin = jnp.any(cand_fin, axis=0, keepdims=True)
            rank_h = jnp.where(
                any_fin,
                jnp.where(cand_fin, cand_h, jnp.full_like(cand_h, jnp.inf)),
                jnp.zeros_like(cand_h),
            )
            j_best = jnp.argmin(rank_h, axis=0)[None, ...]
            x_star = jnp.take_along_axis(cand_x, j_best, axis=0)[0]
            f_star = jnp.take_along_axis(cand_f, j_best, axis=0)[0]
            g_star = jnp.take_along_axis(cand_g, j_best, axis=0)[0]
            bw_star = jnp.take_along_axis(cand_bw, j_best, axis=0)[0]
            status = jnp.take_along_axis(cand_status, j_best, axis=0)[0]
            steps = jnp.where(
                lo_in >= hi_in,
                jnp.zeros_like(sub_steps[0]),
                jnp.max(sub_steps, axis=0),
            )
        sign_arr = jnp.asarray(sign_float, dtype=x_star.dtype)
        g_h_star = sign_arr * g_star
        gtol_b_star = jnp.broadcast_to(gtol_in, x_star.shape)
        zero_x = jnp.zeros_like(x_star)
        fin_star = (
            status != jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32)
        ) & jnp.isfinite(g_h_star)
        deg_star = lo_in >= hi_in
        at_lo_star = (~deg_star) & (x_star <= lo_in) & (g_h_star >= -gtol_b_star)
        at_hi_star = (~deg_star) & (x_star >= hi_in) & (g_h_star <= gtol_b_star)
        lm_lo_act = at_lo_star | (deg_star & (g_h_star >= zero_x))
        lm_hi_act = at_hi_star | (deg_star & (g_h_star < zero_x))
        lm_star = LagrangeMultipliers(
            lower=jnp.where(
                fin_star & lm_lo_act, jnp.maximum(g_h_star, zero_x), zero_x
            ),
            upper=jnp.where(
                fin_star & lm_hi_act, jnp.maximum(-g_h_star, zero_x), zero_x
            ),
        )
        aux = fn_pure(x_star, args_in, consts_in, p_in)[1] if has_aux else None
        return BoxSolveResult(
            x=x_star,
            fun=f_star,
            grad=g_star,
            lm=lm_star,
            bracket_width=bw_star,
            steps=steps,
            status=status,
            lower=lo_in,
            upper=hi_in,
            args=args_in,
            p=p_in,
            aux=aux,
        )

    @solve_ad.defjvp
    def solve_ad_jvp(primals, tangents):
        lo_in, hi_in, _, args_in, consts_in, p_in, _, _, gtol_in, _ = primals
        lo_dot, hi_dot, _, _, _, p_dot, _, _, _, _ = tangents
        res = solve_ad(*primals)
        ac_in = (args_in, consts_in)

        ad_success = res.status == jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32)
        if max_steps_is_success:
            ad_success = ad_success | (
                res.status == jnp.asarray(SolverStatus.MAX_STEPS, dtype=jnp.int32)
            )

        mid_fallback = jax.lax.stop_gradient(
            jnp.where(
                jnp.isfinite(lo_in) & jnp.isfinite(hi_in),
                0.5 * (lo_in + hi_in),
                jnp.zeros_like(res.x),
            )
        )
        x_safe = jnp.where(ad_success & jnp.isfinite(res.x), res.x, mid_fallback)
        lo_safe = jnp.where(
            ad_success & jnp.isfinite(lo_in), lo_in, jnp.zeros_like(lo_in)
        )
        hi_safe = jnp.where(
            ad_success & jnp.isfinite(hi_in), hi_in, jnp.ones_like(hi_in)
        )
        p_safe = _safe_primal_tree(ad_success, p_in)

        ad_lo_dot = jnp.where(ad_success, lo_dot, jnp.zeros_like(lo_dot))
        ad_hi_dot = jnp.where(ad_success, hi_dot, jnp.zeros_like(hi_dot))
        ad_p_dot = _safe_tangent_tree(ad_success, p_dot)

        one_x = jnp.ones_like(x_safe)

        def val_and_slope(x_val, p_val):
            return jax.jvp(lambda z: f_only(z, ac_in, p_val), (x_val,), (one_x,))

        (_, g_f), (_, f_xx) = jax.jvp(
            lambda z: val_and_slope(z, p_safe), (x_safe,), (one_x,)
        )
        _, f_p_dot = jax.jvp(
            lambda pv: f_only(x_safe, ac_in, pv), (p_safe,), (ad_p_dot,)
        )
        _, f_xp_dot = jax.jvp(
            lambda pv: val_and_slope(x_safe, pv)[1], (p_safe,), (ad_p_dot,)
        )

        sign = jnp.asarray(sign_float, dtype=x_safe.dtype)
        g_h = sign * g_f
        gtol_safe = jnp.where(jnp.isfinite(gtol_in), gtol_in, jnp.zeros_like(gtol_in))
        zero_g = jnp.zeros_like(res.grad)

        deg = lo_safe >= hi_safe
        deg_hi = deg & (g_h < zero_g)
        deg_lo = deg & (~deg_hi)
        at_lower = (~deg) & (x_safe <= lo_safe) & (g_h >= -gtol_safe)
        at_upper = (~deg) & (x_safe >= hi_safe) & (g_h <= gtol_safe)
        on_bound = deg | at_lower | at_upper
        interior = (~on_bound) & ad_success
        nonzero_f_xx = interior & (f_xx != 0)
        safe_f_xx = jnp.where(nonzero_f_xx, f_xx, one_x)
        interior_dot = jnp.where(
            nonzero_f_xx, -f_xp_dot / safe_f_xx, jnp.zeros_like(x_safe)
        )

        bound_x_dot = jnp.where(
            deg_lo | at_lower,
            ad_lo_dot,
            jnp.where(deg_hi | at_upper, ad_hi_dot, jnp.zeros_like(x_safe)),
        )
        x_dot = jnp.where(
            ~ad_success,
            jnp.zeros_like(x_safe),
            jnp.where(on_bound, bound_x_dot, interior_dot),
        )
        safe_f_xx_bnd = jnp.where(jnp.isfinite(f_xx), f_xx, zero_g)
        if max_steps_is_success:
            fun_dot = jnp.where(
                ad_success, g_f * x_dot + f_p_dot, jnp.zeros_like(res.fun)
            )
            grad_dot = jnp.where(ad_success, safe_f_xx_bnd * x_dot + f_xp_dot, zero_g)
        else:
            fun_dot = jnp.where(
                ad_success,
                jnp.where(on_bound, g_f * bound_x_dot + f_p_dot, f_p_dot),
                jnp.zeros_like(res.fun),
            )
            bound_grad_dot = safe_f_xx_bnd * bound_x_dot + f_xp_dot
            grad_dot = jnp.where(ad_success & on_bound, bound_grad_dot, zero_g)
        bound_lm_dot = safe_f_xx_bnd * bound_x_dot + f_xp_dot
        lm_lo_active = (at_lower | deg_lo) & (g_h > zero_g)
        lm_hi_active = (at_upper | deg_hi) & (g_h < zero_g)
        lm_dot = LagrangeMultipliers(
            lower=jnp.where(ad_success & lm_lo_active, sign * bound_lm_dot, zero_g),
            upper=jnp.where(ad_success & lm_hi_active, -sign * bound_lm_dot, zero_g),
        )

        if has_aux:
            _, raw_aux_dot = jax.jvp(
                lambda x_val, p_val: fn_pure(x_val, args_in, consts_in, p_val)[1],
                (x_safe, p_safe),
                (x_dot, ad_p_dot),
            )
            aux_dot = _safe_tangent_tree(ad_success, raw_aux_dot)
        else:
            aux_dot = None

        tan_res = BoxSolveResult(
            x=x_dot,
            fun=fun_dot,
            grad=grad_dot,
            lm=lm_dot,
            bracket_width=jnp.zeros_like(res.bracket_width),
            steps=_zero_tangent_leaf(res.steps),
            status=_zero_tangent_leaf(res.status),
            lower=lo_dot,
            upper=hi_dot,
            args=jax.tree.map(_zero_tangent_leaf, res.args),
            p=p_dot,
            aux=aux_dot,
        )
        return res, tan_res

    return solve_ad(lo, hi, x0, args, fn_consts, p, xtol, rtol, gtol, max_steps)


_solve_box_secant_jit = functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))(
    _solve_box_secant
)


class BoxSecant:
    """Safeguarded projected secant-bisection solver for 1D box optimization.

    Minimizes or maximizes a scalar differentiable objective ``fn`` on a closed
    interval ``[lower, upper]`` using forward-mode directional derivatives,
    projected secant steps safeguarded by bracket bisection, fused
    ``custom_vmap`` batching, and box-aware implicit differentiation.
    """

    def __init__(
        self,
        fn,
        *,
        sense: str = "min",
        has_aux: bool = False,
        loop: str = "while",
    ):
        if sense not in ("min", "max"):
            raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
        if loop not in ("while", "fori"):
            raise ValueError(f"loop must be 'while' or 'fori', got {loop!r}")
        canonical_fn, fn_arity = canonicalize_residual(fn)
        self._raw_fn = fn
        self._canonical_fn = canonical_fn
        self.fn_arity = fn_arity
        self.sense = sense
        self.has_aux = bool(has_aux)
        self.loop = loop
        self._sign = 1.0 if sense == "min" else -1.0
        self._static_key = (
            _static_key_component(self._raw_fn),
            self.fn_arity,
            self.sense,
            self.has_aux,
            self.loop,
        )
        self._static_hash = hash(self._static_key)
        self._sealed = True

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._static_key == other._static_key

    def __hash__(self):
        return self._static_hash

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError(
                f"{type(self).__name__} is frozen after construction; build a "
                "new solver instead of assigning attributes"
            )
        object.__setattr__(self, name, value)

    def _check_fn_args(self, args, p):
        if args is not None and self.fn_arity < 2:
            raise ValueError("args was passed but fn takes only (x); use fn(x, args)")
        if p is not None and self.fn_arity < 3:
            raise ValueError(
                "p was passed but fn takes no p argument; use fn(x, args, p)"
            )

    def solve(
        self,
        lower,
        upper,
        args=None,
        *,
        x0=None,
        p=None,
        xtol=None,
        rtol=None,
        gtol=None,
        max_steps: int = 32,
        max_steps_is_success: bool = False,
        num_starts: int = 1,
        loop: str | None = None,
        jit: bool = True,
    ) -> BoxSolveResult:
        """Optimize ``fn`` over ``[lower, upper]`` with implicit AD.

        ``xtol`` and ``gtol`` default to ``eps ** 0.8`` and ``rtol`` defaults to
        ``4 * eps`` for the inferred floating dtype. ``num_starts > 1``
        partitions ``[lower, upper]`` into uniform sub-boxes (plus exact grid
        edges and optional ``x0``) in a single fused loop. Returned ``x``,
        ``fun``, ``grad``, ``lm``, and ``aux`` support implicit AD wrt
        ``(lower, upper, p)``.
        """
        self._check_fn_args(args, p)
        if isinstance(num_starts, bool) or int(num_starts) < 1:
            raise ValueError(f"num_starts must be an int >= 1, got {num_starts!r}")
        num_starts_int = int(num_starts)
        eff_loop = self.loop if loop is None else loop
        if eff_loop not in ("while", "fori"):
            raise ValueError(f"loop must be 'while' or 'fori', got {eff_loop!r}")
        dtype = _infer_box_dtype(lower, upper, x0, args, p)
        lo, hi = jnp.broadcast_arrays(
            jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype)
        )
        has_x0 = x0 is not None
        x0_arr = (
            None
            if not has_x0
            else jnp.broadcast_to(jnp.asarray(x0, dtype=dtype), lo.shape)
        )
        xtol_arr = jnp.asarray(
            _default_xtol(dtype) if xtol is None else xtol, dtype=dtype
        )
        rtol_arr = jnp.asarray(
            _default_rtol(dtype) if rtol is None else rtol, dtype=dtype
        )
        gtol_arr = jnp.asarray(
            _default_gtol(dtype) if gtol is None else gtol, dtype=dtype
        )
        max_s_arr = jnp.asarray(max_steps, dtype=jnp.int32)
        use_fori = eff_loop == "fori"
        driver = _solve_box_secant_jit if jit else _solve_box_secant
        return driver(
            self,
            use_fori,
            has_x0,
            bool(max_steps_is_success),
            num_starts_int,
            lo,
            hi,
            x0_arr,
            args,
            p,
            xtol_arr,
            rtol_arr,
            gtol_arr,
            max_s_arr,
        )
