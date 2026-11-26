import functools
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from nlls_gram.box_secant import (
    _align_batched_leaf,
    _default_gtol,
    _default_rtol,
    _default_xtol,
    _explicit_float_dtypes,
    _safe_primal_tree,
    _safe_tangent_tree,
)
from nlls_gram.types import ScalarSolveResult, SolverStatus, TrustRegionInfo
from nlls_gram.utilities import (
    _static_key_component,
    _zero_tangent_leaf,
    canonicalize_residual,
    register_pytree_dataclass,
)

__all__ = ["ScalarGlobalization", "ScalarNewton", "TrustRegion"]


class ScalarGlobalization:
    """Base class for 1D globalization strategies in ``ScalarNewton``."""


@dataclass(frozen=True, eq=False)
class TrustRegion(ScalarGlobalization):
    """Scalar trust-region globalization for ``ScalarNewton``.

    When ``init_radius`` is ``None``, the initial radius defaults to
    ``min(max(|x0|, 1), max_radius)``. When ``max_radius`` is ``None``, it
    defaults to ``1e4 * init_radius``.
    """

    init_radius: Any = None
    max_radius: Any = None
    eta_accept: Any = 1e-4
    eta_shrink: Any = 0.25
    eta_expand: Any = 0.75
    shrink_factor: Any = 0.25
    expand_factor: Any = 2.0


register_pytree_dataclass(
    TrustRegion,
    data_fields=(
        "init_radius",
        "max_radius",
        "eta_accept",
        "eta_shrink",
        "eta_expand",
        "shrink_factor",
        "expand_factor",
    ),
)


class _ScalarNewtonSpec:
    __slots__ = (
        "raw_fn",
        "canonical_fn",
        "fn_arity",
        "sense",
        "has_aux",
        "sign",
        "key",
        "h",
    )

    def __init__(self, raw_fn, canonical_fn, fn_arity, sense, has_aux, sign):
        self.raw_fn = raw_fn
        self.canonical_fn = canonical_fn
        self.fn_arity = fn_arity
        self.sense = sense
        self.has_aux = has_aux
        self.sign = sign
        self.key = (_static_key_component(raw_fn), fn_arity, sense, has_aux)
        self.h = hash(self.key)

    def __eq__(self, other):
        return isinstance(other, _ScalarNewtonSpec) and self.key == other.key

    def __hash__(self):
        return self.h


def _infer_scalar_dtype(x0, args, p, glob):
    dtypes = _explicit_float_dtypes((x0, args, p, glob))
    if dtypes:
        return jnp.result_type(*dtypes)
    return jnp.result_type(float)


def _scalar_converged(x, x_init, g, curv, xtol, rtol, gtol):
    zero = jnp.zeros_like(x)
    scale = jnp.maximum(jnp.abs(x), jnp.abs(x_init))
    xtol_eff = xtol + rtol * scale
    small_g = (gtol <= zero) | (jnp.abs(g) <= jnp.sqrt(jnp.maximum(gtol, zero)))
    grad_ok = (gtol > zero) & (jnp.abs(g) <= gtol) & (curv >= zero)
    step_ok = (
        (xtol_eff > zero) & small_g & (curv > zero) & (jnp.abs(g) <= curv * xtol_eff)
    )
    return grad_ok | step_ok


def _run_trust_region_loop(
    jet_fn, x0, glob: TrustRegion, xtol, rtol, gtol, max_steps, sign, use_fori
):
    dtype = x0.dtype
    zero = jnp.zeros_like(x0)
    one = jnp.ones_like(x0)
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    xtol_a = jnp.asarray(xtol, dtype=dtype)
    rtol_a = jnp.asarray(rtol, dtype=dtype)
    gtol_a = jnp.asarray(gtol, dtype=dtype)
    max_s_lane = jnp.broadcast_to(jnp.asarray(max_steps, dtype=jnp.int32), x0.shape)
    max_s = jnp.max(max_s_lane) if x0.size > 0 else jnp.asarray(0, dtype=jnp.int32)

    scale0 = jnp.maximum(jnp.abs(x0), one)
    rad0_raw = (
        scale0
        if glob.init_radius is None
        else jnp.broadcast_to(jnp.asarray(glob.init_radius, dtype=dtype), x0.shape)
    )
    if glob.max_radius is None:
        max_rad = jnp.asarray(1e4, dtype=dtype) * rad0_raw
        rad0 = rad0_raw
    else:
        max_rad = jnp.broadcast_to(jnp.asarray(glob.max_radius, dtype=dtype), x0.shape)
        rad0 = jnp.minimum(rad0_raw, max_rad)

    eta_accept = jnp.asarray(glob.eta_accept, dtype=dtype)
    eta_shrink = jnp.asarray(glob.eta_shrink, dtype=dtype)
    eta_expand = jnp.asarray(glob.eta_expand, dtype=dtype)
    shrink_factor = jnp.asarray(glob.shrink_factor, dtype=dtype)
    expand_factor = jnp.asarray(glob.expand_factor, dtype=dtype)

    h0, g0, c0 = jet_fn(x0)
    steps0 = jnp.zeros(x0.shape, dtype=jnp.int32)

    def tr_step(carry):
        x, h, g, c, rad, step_norm, steps = carry
        conv = _scalar_converged(x, x0, g, c, xtol_a, rtol_a, gtol_a)
        fin = jnp.isfinite(x) & jnp.isfinite(h) & jnp.isfinite(g) & jnp.isfinite(c)
        rad_floor = jnp.maximum(eps * jnp.maximum(jnp.abs(x), scale0), tiny)
        stag = (~conv) & (rad <= rad_floor)
        active = (~conv) & (~stag) & (steps < max_s_lane) & fin

        pos_c = c > zero
        safe_c = jnp.where(pos_c, c, one)
        newton_s = -g / safe_c
        desc_dir = jnp.where(g > zero, -one, one)
        s_raw = jnp.where(pos_c, jnp.clip(newton_s, -rad, rad), desc_dir * rad)
        x_eval = jnp.where(active, x + s_raw, x)
        h_n, g_n, c_n = jet_fn(x_eval)

        cand_fin = (
            jnp.isfinite(x_eval)
            & jnp.isfinite(h_n)
            & jnp.isfinite(g_n)
            & jnp.isfinite(c_n)
        )
        half = jnp.asarray(0.5, dtype=dtype)
        s_abs = jnp.abs(s_raw)
        pred = -(g * s_raw + half * c * s_raw * s_raw)
        ared = h - h_n
        rho = ared / jnp.maximum(pred, tiny)
        ulp_tol = jnp.asarray(4.0, dtype=dtype) * eps * jnp.maximum(jnp.abs(h), tiny)
        std_acc = (pred > zero) & (ared >= eta_accept * pred)
        polish_acc = (
            pos_c
            & (c_n > zero)
            & (pred <= ulp_tol)
            & (jnp.abs(g_n) <= half * jnp.abs(g))
            & (ared >= -ulp_tol)
        )
        accepted = active & cand_fin & (std_acc | polish_acc)

        rad_rej = shrink_factor * jnp.minimum(rad, jnp.maximum(s_abs, tiny))
        poor_fit = (~polish_acc) & (rho < eta_shrink)
        great_fit = (rho >= eta_expand) & (
            s_abs >= jnp.asarray(0.95, dtype=dtype) * rad
        )
        rad_acc = jnp.where(
            poor_fit,
            shrink_factor * s_abs,
            jnp.where(great_fit, jnp.minimum(expand_factor * rad, max_rad), rad),
        )
        rad_cand = jnp.where(accepted, rad_acc, rad_rej)

        return (
            jnp.where(accepted, x_eval, x),
            jnp.where(accepted, h_n, h),
            jnp.where(accepted, g_n, g),
            jnp.where(accepted, c_n, c),
            jnp.where(active, rad_cand, rad),
            jnp.where(accepted, s_abs, step_norm),
            jnp.where(active, steps + jnp.asarray(1, dtype=jnp.int32), steps),
        )

    init_carry = (x0, h0, g0, c0, rad0, zero, steps0)

    if use_fori:
        x, h, g, c, rad, step_norm, steps = jax.lax.fori_loop(
            0, max_s, lambda _, carry: tr_step(carry), init_carry
        )
    else:

        def cond_while(carry):
            x, h, g, c, rad, _, steps, loop_s = carry
            conv = _scalar_converged(x, x0, g, c, xtol_a, rtol_a, gtol_a)
            fin = jnp.isfinite(x) & jnp.isfinite(h) & jnp.isfinite(g) & jnp.isfinite(c)
            rad_floor = jnp.maximum(eps * jnp.maximum(jnp.abs(x), scale0), tiny)
            active = (~conv) & (rad > rad_floor) & (steps < max_s_lane) & fin
            return (loop_s < max_s) & jnp.any(active)

        def body_while(carry):
            *core, loop_s = carry
            return (*tr_step(core), loop_s + jnp.asarray(1, dtype=jnp.int32))

        x, h, g, c, rad, step_norm, steps, _ = jax.lax.while_loop(
            cond_while, body_while, (*init_carry, jnp.asarray(0, dtype=jnp.int32))
        )

    conv_end = _scalar_converged(x, x0, g, c, xtol_a, rtol_a, gtol_a)
    finite = jnp.isfinite(x) & jnp.isfinite(h) & jnp.isfinite(g) & jnp.isfinite(c)
    rad_floor_end = jnp.maximum(eps * jnp.maximum(jnp.abs(x), scale0), tiny)
    stag_end = (~conv_end) & (rad <= rad_floor_end) & (steps < max_s_lane)
    status = jnp.where(
        ~finite,
        jnp.asarray(SolverStatus.NONFINITE, dtype=jnp.int32),
        jnp.where(
            conv_end,
            jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32),
            jnp.where(
                stag_end,
                jnp.asarray(SolverStatus.STAGNATION, dtype=jnp.int32),
                jnp.asarray(SolverStatus.MAX_STEPS, dtype=jnp.int32),
            ),
        ),
    )
    info = TrustRegionInfo(radius=rad, step_norm=step_norm)
    return x, sign * h, sign * g, sign * c, info, steps, status


def _run_scalar_loop(jet_fn, x0, glob, xtol, rtol, gtol, max_steps, sign, use_fori):
    if isinstance(glob, TrustRegion):
        return _run_trust_region_loop(
            jet_fn, x0, glob, xtol, rtol, gtol, max_steps, sign, use_fori
        )
    raise TypeError(f"unsupported globalization strategy: {type(glob).__name__}")


def _make_newton_primal(jet_fn, sign_float, use_fori):
    @jax.custom_batching.custom_vmap
    def solve_level(x0, args, p, glob, xtol, rtol, gtol, max_steps):
        sign = jnp.asarray(sign_float, dtype=x0.dtype)
        return _run_scalar_loop(
            lambda x: jet_fn(x, args, p, sign),
            x0,
            glob,
            xtol,
            rtol,
            gtol,
            max_steps,
            sign,
            use_fori,
        )

    @solve_level.def_vmap
    def solve_level_vmap(
        axis_size, in_batched, x0, args, p, glob, xtol, rtol, gtol, max_steps
    ):
        b_x0, b_args, b_p, b_glob, b_xtol, b_rtol, b_gtol, b_max = in_batched
        x0_v = x0 if b_x0 else jnp.broadcast_to(x0, (axis_size, *jnp.shape(x0)))
        xtol_v = _align_batched_leaf(b_xtol, xtol, x0_v)
        rtol_v = _align_batched_leaf(b_rtol, rtol, x0_v)
        gtol_v = _align_batched_leaf(b_gtol, gtol, x0_v)
        max_v = _align_batched_leaf(b_max, max_steps, x0_v)
        glob_v = jax.tree.map(
            lambda is_b, leaf: _align_batched_leaf(is_b, leaf, x0_v),
            b_glob,
            glob,
        )
        args_axes = jax.tree.map(lambda is_b: 0 if is_b else None, b_args)
        p_axes = jax.tree.map(lambda is_b: 0 if is_b else None, b_p)
        next_jet = jax.vmap(jet_fn, in_axes=(0, args_axes, p_axes, None))
        next_solver = _make_newton_primal(next_jet, sign_float, use_fori)
        out = next_solver(x0_v, args, p, glob_v, xtol_v, rtol_v, gtol_v, max_v)
        out_batched = jax.tree.map(lambda _: True, out)
        return out, out_batched

    return solve_level


def _solve_scalar_newton(
    spec: _ScalarNewtonSpec,
    use_fori,
    max_steps_is_success,
    x0,
    args,
    p,
    glob,
    xtol,
    rtol,
    gtol,
    max_steps,
):
    fn_canon = spec.canonical_fn
    has_aux = spec.has_aux
    sign_float = spec.sign

    closed_fn, fn_out_struct = jax.make_jaxpr(fn_canon, return_shape=True)(x0, args, p)
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

    def newton_jet(x, ac_val, p_val, sign):
        one = jnp.ones_like(x)
        (h, g), (_, c) = jax.jvp(
            lambda z: jax.jvp(lambda w: sign * f_only(w, ac_val, p_val), (z,), (one,)),
            (x,),
            (one,),
        )
        return h, g, c

    primal_fn = _make_newton_primal(newton_jet, sign_float, use_fori)

    @jax.custom_jvp
    def solve_ad(
        x0_in,
        args_in,
        consts_in,
        p_in,
        glob_in,
        xtol_in,
        rtol_in,
        gtol_in,
        max_s_in,
    ):
        ac_in = (args_in, consts_in)
        x_star, f_star, g_star, c_star, info_star, steps, status = primal_fn(
            x0_in, ac_in, p_in, glob_in, xtol_in, rtol_in, gtol_in, max_s_in
        )
        aux = fn_pure(x_star, args_in, consts_in, p_in)[1] if has_aux else None
        return ScalarSolveResult(
            x=x_star,
            fun=f_star,
            grad=g_star,
            curv=c_star,
            info=info_star,
            steps=steps,
            status=status,
            args=args_in,
            p=p_in,
            aux=aux,
        )

    @solve_ad.defjvp
    def solve_ad_jvp(primals, tangents):
        x0_in, args_in, consts_in, p_in, _, _, _, _, _ = primals
        _, _, _, p_dot, _, _, _, _, _ = tangents
        res = solve_ad(*primals)
        ac_in = (args_in, consts_in)

        ad_success = res.status == jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32)
        if max_steps_is_success:
            ad_success = ad_success | (
                res.status == jnp.asarray(SolverStatus.MAX_STEPS, dtype=jnp.int32)
            )

        fallback_x = jax.lax.stop_gradient(
            jnp.where(jnp.isfinite(x0_in), x0_in, jnp.zeros_like(res.x))
        )
        x_safe = jnp.where(ad_success & jnp.isfinite(res.x), res.x, fallback_x)
        p_safe = _safe_primal_tree(ad_success, p_in)
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

        nonzero_f_xx = ad_success & (f_xx != 0)
        safe_f_xx = jnp.where(nonzero_f_xx, f_xx, one_x)
        x_dot = jnp.where(nonzero_f_xx, -f_xp_dot / safe_f_xx, jnp.zeros_like(x_safe))
        if max_steps_is_success:
            fun_dot = jnp.where(
                ad_success, g_f * x_dot + f_p_dot, jnp.zeros_like(res.fun)
            )
            grad_dot = jnp.where(
                ad_success, f_xx * x_dot + f_xp_dot, jnp.zeros_like(res.grad)
            )
        else:
            fun_dot = jnp.where(ad_success, f_p_dot, jnp.zeros_like(res.fun))
            grad_dot = jnp.zeros_like(res.grad)

        if has_aux:
            _, raw_aux_dot = jax.jvp(
                lambda x_val, p_val: fn_pure(x_val, args_in, consts_in, p_val)[1],
                (x_safe, p_safe),
                (x_dot, ad_p_dot),
            )
            aux_dot = _safe_tangent_tree(ad_success, raw_aux_dot)
        else:
            aux_dot = None

        tan_res = ScalarSolveResult(
            x=x_dot,
            fun=fun_dot,
            grad=grad_dot,
            curv=jnp.zeros_like(res.curv),
            info=jax.tree.map(_zero_tangent_leaf, res.info),
            steps=_zero_tangent_leaf(res.steps),
            status=_zero_tangent_leaf(res.status),
            args=jax.tree.map(_zero_tangent_leaf, res.args),
            p=p_dot,
            aux=aux_dot,
        )
        return res, tan_res

    return solve_ad(x0, args, fn_consts, p, glob, xtol, rtol, gtol, max_steps)


_solve_scalar_newton_jit = functools.partial(jax.jit, static_argnums=(0, 1, 2))(
    _solve_scalar_newton
)


class ScalarNewton:
    """Safeguarded second-order 1D unconstrained optimizer from an initial guess.

    Minimizes or maximizes a scalar twice-differentiable objective ``fn``
    starting from ``x0`` using second-order forward-mode scalar jets, pluggable
    flat-loop globalization (default ``TrustRegion()``), fused ``custom_vmap``
    batching, and implicit differentiation wrt ``p``.
    """

    def __init__(
        self,
        fn,
        *,
        sense: str = "min",
        globalization: ScalarGlobalization = TrustRegion(),  # noqa: B008
        has_aux: bool = False,
        loop: str = "while",
    ):
        canonical_fn, fn_arity = canonicalize_residual(fn)
        self._raw_fn = fn
        self._canonical_fn = canonical_fn
        self.fn_arity = fn_arity
        self.sense = sense
        self.globalization = globalization
        self.has_aux = bool(has_aux)
        self.loop = loop
        self._sign = 1.0 if sense == "min" else -1.0
        self._spec = _ScalarNewtonSpec(
            self._raw_fn,
            self._canonical_fn,
            self.fn_arity,
            self.sense,
            self.has_aux,
            self._sign,
        )
        self._static_key = (
            *self._spec.key,
            jax.tree.structure(self.globalization),
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

    def solve(
        self,
        x0,
        args=None,
        *,
        p=None,
        xtol=None,
        rtol=None,
        gtol=None,
        max_steps: int = 32,
        max_steps_is_success: bool = False,
        loop: str | None = None,
        jit: bool = True,
    ) -> ScalarSolveResult:
        """Optimize ``fn`` from ``x0`` with implicit AD wrt ``p``.

        ``xtol`` and ``gtol`` default to ``eps ** 0.8`` and ``rtol`` defaults to
        ``4 * eps`` for the inferred floating dtype. Returned ``x``, ``fun``,
        ``grad``, and ``aux`` support implicit AD wrt ``p`` at regular
        stationary points (``f_xx != 0``).
        """
        eff_loop = self.loop if loop is None else loop
        dtype = _infer_scalar_dtype(x0, args, p, self.globalization)
        x0_arr = jnp.asarray(x0, dtype=dtype)
        glob_typed = jax.tree.map(
            lambda leaf: jnp.asarray(leaf, dtype=dtype), self.globalization
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
        driver = _solve_scalar_newton_jit if jit else _solve_scalar_newton
        return driver(
            self._spec,
            use_fori,
            bool(max_steps_is_success),
            x0_arr,
            args,
            p,
            glob_typed,
            xtol_arr,
            rtol_arr,
            gtol_arr,
            max_s_arr,
        )
