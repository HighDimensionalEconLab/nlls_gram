"""``vmap_rows``: ``jax.vmap`` for a row-wise residual block, with a
derivative rule under which the Jacobian of the block costs one reverse pass
of the row function per row, taken once per linearization, instead of one
pass per row and Jacobian column or cotangent.
"""

import functools

import jax
import jax.numpy as jnp
from jax._src.interpreters.partial_eval import JaxprTracer
from jax.custom_derivatives import SymbolicZero

from nlls_gram.utilities import HIGHEST

__all__ = ["vmap_rows"]


def vmap_rows(f=None, *, in_axes=None, chunk_size=None):
    """``jax.vmap(f, in_axes)`` with a derivative rule built for the
    Jacobian assembly of :class:`~nlls_gram.LevenbergMarquardt`.

    ``f`` computes one residual row -- a scalar or a small pytree of real
    floating leaves -- from one row of each mapped positional argument and
    the whole of each unmapped one; ``in_axes`` defaults to
    ``(0, None, None, ...)`` when ``None`` and otherwise follows ``jax.vmap``,
    with outputs stacked along axis 0. The value and any plain directional
    ``jax.jvp`` (including the geodesic second directional derivative) run the
    plain ``vmap``. Under ``jax.linearize`` (and ``jax.vjp`` / ``jax.jacrev``)
    the rule evaluates every row's Jacobian with respect to every argument
    that carries a tangent, by ``jax.jacrev`` over the rows (in one ``vmap``
    when ``chunk_size is None``, or via ``jax.lax.map`` in batches of
    ``chunk_size``), and stores it; the tangent is its contraction with the
    input tangents. The solver's assembly then maps over that contraction:
    ``m`` products for the ``m`` unit cotangents of a fat system, ``n`` for
    the ``n`` unit tangents of a tall one, where the plain ``vmap`` pushes a
    pass of ``f`` through every row for each of them. The rule is exact --
    the Jacobian, its products, and the geodesic second directional
    derivative are unchanged to rounding -- and composes with ``jit``,
    ``scan``, and an outer ``vmap``.

    Use it on a row function that is expensive to differentiate (a network
    evaluated at a state and its successors) in a residual with many rows.
    Everything ``f`` differentiates must arrive as an argument (a
    closed-over tracer raises); integer and boolean arguments are fine and
    carry no derivative; the output shares the dtype of the differentiated
    inputs and should be small, since each output entry stores a copy of the
    row inputs. See [Row-wise residuals](vmap_rows.md).
    """
    if f is None:
        return functools.partial(vmap_rows, in_axes=in_axes, chunk_size=chunk_size)
    in_axes = tuple(in_axes) if isinstance(in_axes, list) else in_axes

    def axes_for(args):
        if in_axes is None:
            return (0,) + (None,) * (len(args) - 1)
        return in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)

    def vmapped(*args):
        return jax.vmap(f, axes_for(args))(*args)

    @jax.custom_jvp
    def rows(*args):
        return vmapped(*args)

    def is_symbolic(t):
        return isinstance(t, SymbolicZero)

    def is_none(x):
        return x is None

    def is_zero(t):
        return is_symbolic(t) or jnp.asarray(t).dtype == jax.dtypes.float0

    def zero_to_array(t, primal):
        def one(tl, pl):
            if is_symbolic(tl):
                pl = jnp.asarray(pl)
                if jnp.issubdtype(pl.dtype, jnp.floating):
                    return jnp.zeros_like(pl)
                return jnp.zeros(pl.shape, jax.dtypes.float0)
            return tl

        return jax.tree.map(one, t, primal, is_leaf=is_symbolic)

    def rows_jvp(primals, tangents):
        # Only arguments that carry a tangent are differentiated: closed-over
        # data arrives as a symbolic zero, integer and boolean leaves as
        # float0. Both become None here and get no stored Jacobian.
        active = jax.tree.map(
            lambda t: None if is_zero(t) else jnp.asarray(t),
            tangents,
            is_leaf=is_symbolic,
        )
        if not jax.tree.leaves(active):
            r = vmapped(*primals)
            return r, jax.tree.map(jnp.zeros_like, r)
        if not any(isinstance(t, JaxprTracer) for t in jax.tree.leaves(active)):
            return jax.jvp(vmapped, primals, zero_to_array(tangents, primals))
        axes = axes_for(primals)

        def row(*primals_row):
            live_row = jax.tree.map(
                lambda t, x: None if t is None else x,
                active,
                primals_row,
                is_leaf=is_none,
            )

            def f_live(a):
                merged = jax.tree.map(
                    lambda a, x: x if a is None else a,
                    a,
                    primals_row,
                    is_leaf=is_none,
                )
                r = f(*merged)
                return r, r

            return jax.jacrev(f_live, has_aux=True)(live_row)

        if chunk_size is None:
            jac, r = jax.vmap(row, in_axes=axes)(*primals)
        else:
            batched = tuple(
                a for a, ax in zip(primals, axes, strict=True) if ax is not None
            )

            def body(b):
                it = iter(b)
                row_args = [
                    next(it) if ax is not None else a
                    for a, ax in zip(primals, axes, strict=True)
                ]
                return row(*row_args)

            jac, r = jax.lax.map(
                body,
                batched,
                batch_size=None if chunk_size == 1 else chunk_size,
            )
        out_leaves, out_tree = jax.tree.flatten(r)
        out_dtypes = [leaf.dtype for leaf in out_leaves]

        def contract(jac_row, tangents_row):
            def one_output(jac_o, dtype):
                terms = jax.tree.map(
                    lambda j, t: jnp.tensordot(j, t, axes=t.ndim, precision=HIGHEST),
                    jac_o,
                    tangents_row,
                )
                return functools.reduce(jnp.add, jax.tree.leaves(terms)).astype(dtype)

            per_output = out_tree.flatten_up_to(jac_row)
            return jax.tree.unflatten(
                out_tree,
                [one_output(j, d) for j, d in zip(per_output, out_dtypes, strict=True)],
            )

        return r, jax.vmap(contract, in_axes=(0, axes))(jac, active)

    rows.defjvp(rows_jvp, symbolic_zeros=True)
    return rows
