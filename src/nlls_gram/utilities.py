"""Solver-agnostic helpers: pytree selection/masking, jit static-key hashing,
and residual-signature canonicalization.

No solver, metric, or linear-algebra code lives here -- only plumbing both
``LevenbergMarquardt`` and ``RidgeLevenbergMarquardt`` share.
"""

import inspect

import jax
import jax.numpy as jnp


def _tree_changed(new, old):
    new_leaves, new_treedef = jax.tree_util.tree_flatten(new)
    old_leaves, old_treedef = jax.tree_util.tree_flatten(old)
    if new_treedef != old_treedef:
        return jnp.asarray(True)
    changed = jnp.asarray(False)
    for new_leaf, old_leaf in zip(new_leaves, old_leaves, strict=True):
        # equal_nan: an unchanged NaN sentinel is not a change.
        changed = changed | ~jnp.array_equal(new_leaf, old_leaf, equal_nan=True)
    return changed


def _zero_tangent_leaf(leaf):
    if leaf is None:
        return None
    array = jnp.asarray(leaf)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return jnp.zeros(array.shape, dtype=jax.dtypes.float0)
    return jnp.zeros_like(leaf)


def _broadcast_leading_condition(condition, leaf):
    """Broadcast a scalar or leading-batch condition over an array leaf."""
    condition = jnp.asarray(condition, dtype=jnp.bool_)
    leaf_ndim = jnp.ndim(leaf)
    if condition.ndim < leaf_ndim:
        condition = jnp.reshape(
            condition, condition.shape + (1,) * (leaf_ndim - condition.ndim)
        )
    return condition


def _where_tree(condition, on_true, on_false):
    """Select matching pytrees, treating ``condition`` axes as leading axes."""

    def select(true_leaf, false_leaf):
        if true_leaf is None:
            return None
        return jnp.where(
            _broadcast_leading_condition(condition, true_leaf),
            true_leaf,
            false_leaf,
        )

    return jax.tree.map(select, on_true, on_false)


def _mask_tangent_tree(condition, tangent):
    """Keep tangent leaves where condition holds and zero them elsewhere."""

    def mask(leaf):
        if leaf is None:
            return None
        array = jnp.asarray(leaf)
        if array.dtype == jax.dtypes.float0:
            return leaf
        return jnp.where(
            _broadcast_leading_condition(condition, array),
            array,
            jnp.zeros_like(array),
        )

    return jax.tree.map(mask, tangent)


def _typed_key(value):
    # Tag each hashable value/container with its type so the static key keeps 1,
    # 1.0, and True distinct -- raw ==/hash collapse them (hash(1) == hash(True)),
    # which would silently reuse a mismatched compile. This mirrors jax's own
    # strict-type equality for static jit arguments. Unhashable values raise, and
    # _hashable_hook degrades those specs to identity hashing.
    if isinstance(value, tuple):
        return (tuple, tuple(_typed_key(v) for v in value))
    if isinstance(value, frozenset):
        return (frozenset, frozenset(_typed_key(v) for v in value))
    return (type(value), value)


class _IdentityKey:
    """Static-key stand-in comparing by object identity (for unhashable values)."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return isinstance(other, _IdentityKey) and self.obj is other.obj

    def __hash__(self):
        return id(self.obj)


def _static_key_component(value):
    # Hashable settings (scalars, strings, functions, frozen configs) key by
    # value; anything unhashable keys by identity so hashing never raises and
    # equality stays consistent with the hash.
    try:
        hash(value)
    except TypeError:
        return _IdentityKey(value)
    return value


class _IdentityCallable:
    """Hashable-by-identity pass-through for unhashable callables used as jit
    statics (e.g. an eq=True dataclass instance implementing ``__call__``).
    ``__weakref__`` is required: jax.eval_shape weak-references the callable.
    """

    __slots__ = ("fn", "__weakref__")

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)

    def __eq__(self, other):
        return isinstance(other, _IdentityCallable) and self.fn is other.fn

    def __hash__(self):
        return id(self.fn)


def _hashable_hook(fn):
    if fn is None:
        return None
    try:
        hash(fn)
    except TypeError:
        return _IdentityCallable(fn)
    return fn


def canonicalize_residual(residual_fn):
    """Wrap a residual taking ``(x)``, ``(x, args)``, or ``(x, args, p)`` --
    always in that order -- into the canonical 3-arg form, so the compiled
    code is identical for all three. Uninspectable signatures (or ``*args``)
    are assumed 3-arg. Returns ``(canonical_fn, arity)``.
    """
    try:
        signature = inspect.signature(residual_fn)
    except (TypeError, ValueError):
        residual_arity = 3
    else:
        residual_arity = 0
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                residual_arity += 1
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                residual_arity = 3
                break
        if residual_arity < 1 or residual_arity > 3:
            raise ValueError(
                "residual_fn must take 1 to 3 positional arguments: "
                "(x), (x, args), or (x, args, p)"
            )
    if residual_arity == 1:

        def canonical_residual(x, args, p):
            return residual_fn(x)

    elif residual_arity == 2:

        def canonical_residual(x, args, p):
            return residual_fn(x, args)

    else:
        canonical_residual = residual_fn
    return canonical_residual, residual_arity
