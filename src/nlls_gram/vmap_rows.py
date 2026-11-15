import functools

import jax
import jax.experimental.hijax as hijax
import jax.numpy as jnp

from nlls_gram.utilities import HIGHEST

__all__ = ["vmap_rows"]


def _is_none(x):
    return x is None


def _freeze_tree_key(obj):
    if isinstance(obj, dict):
        return (dict, tuple((k, _freeze_tree_key(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return (list, tuple(_freeze_tree_key(v) for v in obj))
    if isinstance(obj, tuple):
        return (tuple, tuple(_freeze_tree_key(v) for v in obj))
    if isinstance(obj, hijax.ShapedArray):
        return (obj.shape, obj.dtype)
    try:
        hash(obj)
    except TypeError:
        return (type(obj), id(obj))
    return obj


def vmap_rows(f=None, *, in_axes=None, chunk_size=None):
    """Map ``f`` over residual rows with single per-row ``jacrev`` per linearization.

    ``in_axes`` defaults to ``(0, None, None, ...)`` when ``None`` and otherwise
    follows ``jax.vmap`` with outputs stacked along axis 0. Plain directional
    ``jax.jvp`` calls (including geodesic second directional derivatives) run
    ``jax.vmap(f, in_axes)`` directly. Under ``jax.linearize`` and reverse-mode
    autodiff, each row's Jacobian with respect to active floating inputs is
    evaluated once via ``jax.jacrev`` (batched with ``jax.vmap`` when
    ``chunk_size is None``, or in batches via ``jax.lax.map`` when ``chunk_size``
    is set) and contracted across tangent or cotangent directions.
    """
    if f is None:
        return functools.partial(vmap_rows, in_axes=in_axes, chunk_size=chunk_size)
    norm_axes = tuple(in_axes) if isinstance(in_axes, list) else in_axes
    fn_key = _freeze_tree_key(f)

    def axes_for(args):
        if norm_axes is None:
            return (0,) + (None,) * (len(args) - 1)
        return norm_axes if isinstance(norm_axes, tuple) else (norm_axes,) * len(args)

    def materialize_tangents(tangents, avals):
        def one(t, av):
            if isinstance(t, hijax.Zero):
                if jnp.issubdtype(av.dtype, jnp.floating):
                    return jnp.zeros(av.shape, av.dtype)
                return jnp.zeros(av.shape, jax.dtypes.float0)
            return t

        return jax.tree.map(
            one,
            tangents,
            avals,
            is_leaf=lambda x: isinstance(x, hijax.Zero),
        )

    def zero_out_tangent(av):
        if jnp.issubdtype(av.dtype, jnp.floating):
            return jnp.zeros(av.shape, av.dtype)
        return jnp.zeros(av.shape, jax.dtypes.float0)

    class RowsPrim(hijax.HiPrim):
        vjp_fwd, vjp_bwd_retval = hijax.vjp_from_lin

        def __init__(
            self,
            in_avals,
            out_aval,
            expand_fn,
            jvp_fn,
            lin_fn,
            linearized_fn,
            prim_key,
        ):
            self.in_avals = in_avals
            self.out_aval = out_aval
            self.expand_fn = expand_fn
            self.jvp_fn = jvp_fn
            self.lin_fn = lin_fn
            self.linearized_fn = linearized_fn
            self.params = {
                "_prim_key": (
                    prim_key,
                    _freeze_tree_key(in_avals),
                    _freeze_tree_key(out_aval),
                ),
                "chunk_size": chunk_size,
            }
            super().__init__()

        def expand(self, *args):
            return self.expand_fn(*args)

        def jvp(self, primals, tangents):
            return self.jvp_fn(primals, materialize_tangents(tangents, self.in_avals))

        def lin(self, nzs_in, *primals):
            return self.lin_fn(nzs_in, *primals)

        def linearized(self, jac, *tangents):
            if jac is None:
                return jax.tree.map(zero_out_tangent, self.out_aval)
            return self.linearized_fn(
                jac, *materialize_tangents(tangents, self.in_avals)
            )

        def transpose(self, out_ct, *maybe_accums):
            is_accum = lambda x: isinstance(x, hijax.GradAccum)  # noqa: E731
            acc_leaves, acc_tree = jax.tree.flatten(maybe_accums, is_leaf=is_accum)
            active_idx = [
                i
                for i, x in enumerate(acc_leaves)
                if isinstance(x, hijax.GradAccum) and not isinstance(x, hijax.NullAccum)
            ]
            if not active_idx:
                return None
            out_avals_flat, out_tdef = jax.tree.flatten(self.out_aval)
            float_idx = [
                j
                for j, av in enumerate(out_avals_flat)
                if jnp.issubdtype(av.dtype, jnp.floating)
            ]
            if not float_idx:
                for i in active_idx:
                    av = acc_leaves[i].aval
                    acc_leaves[i].accum(jnp.zeros(av.shape, av.dtype))
                return None

            dummies = [
                jax.ShapeDtypeStruct(acc_leaves[i].aval.shape, acc_leaves[i].aval.dtype)
                for i in active_idx
            ]
            active_pos = {idx: k for k, idx in enumerate(active_idx)}

            def linear_part(*active_vals):
                rebuilt = []
                for i, leaf in enumerate(acc_leaves):
                    if i in active_pos:
                        rebuilt.append(active_vals[active_pos[i]])
                    elif isinstance(leaf, hijax.NullAccum):
                        rebuilt.append(jnp.zeros(leaf.aval.shape, leaf.aval.dtype))
                    else:
                        rebuilt.append(leaf)
                y = self.expand_fn(*jax.tree.unflatten(acc_tree, rebuilt))
                y_flat = out_tdef.flatten_up_to(y)
                return tuple(y_flat[j] for j in float_idx)

            ct_inst = hijax.instantiate_zeros(out_ct)
            ct_flat = out_tdef.flatten_up_to(ct_inst)
            ct_float = tuple(ct_flat[j] for j in float_idx)
            in_cts = jax.linear_transpose(linear_part, *dummies)(ct_float)
            for i, ct in zip(active_idx, in_cts, strict=True):
                acc_leaves[i].accum(ct)
            return None

        def batch(self, axis_data, args, dims):
            if all(d is None for d in jax.tree.leaves(dims)):
                return self(*args), jax.tree.map(lambda _: None, self.out_aval)
            out_dims = jax.tree.map(lambda _: 0, self.out_aval)
            b_in_avals = jax.tree.map(
                lambda x: hijax.ShapedArray(jnp.shape(x), jnp.result_type(x)), args
            )
            b_out_aval = jax.tree.map(
                lambda av: hijax.ShapedArray((axis_data.size, *av.shape), av.dtype),
                self.out_aval,
            )
            b_expand = hijax.vmap_rule(axis_data, self.expand_fn, dims, out_dims)
            b_jvp = hijax.vmap_rule(
                axis_data, self.jvp_fn, (dims, dims), (out_dims, out_dims)
            )

            def b_lin(nzs_in, *primals):
                fn = functools.partial(self.lin_fn, nzs_in)
                sample = jax.eval_shape(
                    fn,
                    *jax.tree.map(
                        lambda av: jax.ShapeDtypeStruct(av.shape, av.dtype),
                        self.in_avals,
                    ),
                )
                res_dims = jax.tree.map(lambda _: 0, sample[1])
                return hijax.vmap_rule(axis_data, fn, dims, (out_dims, res_dims))(
                    *primals
                )

            def b_linearized(jac, *tangents):
                res_dims = jax.tree.map(lambda _: 0, jac)
                return hijax.vmap_rule(
                    axis_data, self.linearized_fn, (res_dims, *dims), out_dims
                )(jac, *tangents)

            b_prim = RowsPrim(
                b_in_avals,
                b_out_aval,
                b_expand,
                b_jvp,
                b_lin,
                b_linearized,
                (
                    self.params["_prim_key"],
                    "batch",
                    axis_data.size,
                    _freeze_tree_key(dims),
                ),
            )
            return b_prim(*args), out_dims

    def make_base_prim(in_avals, axes):
        out_shape = jax.eval_shape(
            lambda *a: jax.vmap(f, axes)(*a),
            *jax.tree.map(
                lambda av: jax.ShapeDtypeStruct(av.shape, av.dtype), in_avals
            ),
        )
        out_aval = jax.tree.map(
            lambda s: hijax.ShapedArray(s.shape, s.dtype), out_shape
        )
        out_avals_flat, out_tree = jax.tree.flatten(out_aval)
        float_out_indices = [
            i
            for i, av in enumerate(out_avals_flat)
            if jnp.issubdtype(av.dtype, jnp.floating)
        ]

        def active_mask_for(nzs_in):
            return jax.tree.map(
                lambda nz, av: bool(nz) and jnp.issubdtype(av.dtype, jnp.floating),
                nzs_in,
                in_avals,
            )

        def leaf_axes_for(primals):
            leaf_ax_list = []

            def collect(ax, sub):
                if ax is None:
                    for _ in jax.tree.leaves(sub):
                        leaf_ax_list.append(None)
                elif isinstance(ax, int):
                    for _ in jax.tree.leaves(sub):
                        leaf_ax_list.append(ax)
                else:
                    jax.tree.map(collect, ax, sub, is_leaf=_is_none)

            for ax, a in zip(axes, primals, strict=True):
                collect(ax, a)
            return leaf_ax_list

        def expand_fn(*args):
            return jax.vmap(f, axes)(*args)

        def jvp_fn(primals, tangents):
            return jax.jvp(expand_fn, primals, tangents)

        def lin_fn(nzs_in, *primals):
            active_mask = active_mask_for(nzs_in)
            if not float_out_indices or not any(jax.tree.leaves(active_mask)):
                return expand_fn(*primals), None

            def row(*primals_row):
                live_row = jax.tree.map(
                    lambda act, x: x if act else None,
                    active_mask,
                    primals_row,
                )

                def f_live(a):
                    merged = jax.tree.map(
                        lambda act, a_val, x_val: a_val if act else x_val,
                        active_mask,
                        a,
                        primals_row,
                    )
                    r_val = f(*merged)
                    r_flat = out_tree.flatten_up_to(r_val)
                    return tuple(r_flat[i] for i in float_out_indices), r_val

                return jax.jacrev(f_live, has_aux=True)(live_row)

            if chunk_size is None:
                jac, r = jax.vmap(row, in_axes=axes)(*primals)
            else:
                flat_primals, in_tree = jax.tree.flatten(primals)
                leaf_axes = leaf_axes_for(primals)
                mapped_indices = [i for i, ax in enumerate(leaf_axes) if ax is not None]
                mapped_leaves = tuple(
                    jnp.moveaxis(flat_primals[i], leaf_axes[i], 0)
                    if leaf_axes[i] != 0
                    else flat_primals[i]
                    for i in mapped_indices
                )

                def body(sliced_mapped):
                    row_flat = list(flat_primals)
                    for idx, val in zip(mapped_indices, sliced_mapped, strict=True):
                        row_flat[idx] = val
                    row_primals = jax.tree.unflatten(in_tree, row_flat)
                    return row(*row_primals)

                jac, r = jax.lax.map(
                    body,
                    mapped_leaves,
                    batch_size=None if chunk_size == 1 else chunk_size,
                )

            return r, jac

        def linearized_fn(jac, *tangents):
            first_jac = jac[0]

            active_tangents = jax.tree.map(
                lambda j_sub, t: None if j_sub is None else jnp.asarray(t),
                first_jac,
                tangents,
                is_leaf=_is_none,
            )

            def axis_for_active(ax, act_sub):
                if act_sub is None or ax is None:
                    return None
                if isinstance(ax, int):
                    return jax.tree.map(
                        lambda s: None if s is None else ax, act_sub, is_leaf=_is_none
                    )
                return jax.tree.map(axis_for_active, ax, act_sub, is_leaf=_is_none)

            act_axes = tuple(
                axis_for_active(ax, sub)
                for ax, sub in zip(axes, active_tangents, strict=True)
            )

            def contract(jac_row, tangents_row):
                def one_output(jac_o, row_aval):
                    terms = jax.tree.map(
                        lambda j, t: (
                            None
                            if j is None or t is None
                            else jnp.tensordot(j, t, axes=t.ndim, precision=HIGHEST)
                        ),
                        jac_o,
                        tangents_row,
                        is_leaf=_is_none,
                    )
                    valid_terms = [x for x in jax.tree.leaves(terms) if x is not None]
                    if not valid_terms:
                        return jnp.zeros(row_aval.shape[1:], dtype=row_aval.dtype)
                    return functools.reduce(jnp.add, valid_terms).astype(row_aval.dtype)

                return tuple(
                    one_output(jac_row[k], out_avals_flat[idx])
                    for k, idx in enumerate(float_out_indices)
                )

            contracted = jax.vmap(contract, in_axes=(0, act_axes))(jac, active_tangents)
            out_flat = [
                None
                if jnp.issubdtype(av.dtype, jnp.floating)
                else jnp.zeros(av.shape, jax.dtypes.float0)
                for av in out_avals_flat
            ]
            for k, idx in enumerate(float_out_indices):
                out_flat[idx] = contracted[k]
            return jax.tree.unflatten(out_tree, out_flat)

        return RowsPrim(
            in_avals,
            out_aval,
            expand_fn,
            jvp_fn,
            lin_fn,
            linearized_fn,
            (fn_key, _freeze_tree_key(axes)),
        )

    @functools.wraps(f)
    def call(*args):
        axes = axes_for(args)
        avals = jax.tree.map(
            lambda x: hijax.ShapedArray(jnp.shape(x), jnp.result_type(x)), args
        )
        return make_base_prim(avals, axes)(*args)

    return call
