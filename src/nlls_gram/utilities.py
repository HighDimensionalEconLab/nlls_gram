import dataclasses
import inspect

import jax
import jax.numpy as jnp

# Pin internal matmuls to full IEEE precision so float32 GPU runs do not drop
# to TF32 tensor-core mantissa width when forming Gram or normal products.
HIGHEST = jax.lax.Precision.HIGHEST


def mm(a, b):
    return jnp.matmul(a, b, precision=HIGHEST)


def register_pytree_dataclass(cls, *, data_fields, meta_fields=()):
    """Register a frozen dataclass as a JAX pytree.

    ``data_fields`` become traced children and ``meta_fields`` become static
    hashable structure. Unflatten bypasses ``__init__`` and ``__post_init__``.
    Every dataclass field must appear in either ``data_fields`` or ``meta_fields``.
    """
    data_fields = tuple(data_fields)
    meta_fields = tuple(meta_fields)
    declared = {f.name for f in dataclasses.fields(cls)}
    listed = set(data_fields) | set(meta_fields)
    if len(data_fields) + len(meta_fields) != len(listed) or listed != declared:
        raise ValueError(
            f"register_pytree_dataclass({cls.__name__}): data_fields + "
            f"meta_fields must cover every dataclass field exactly once"
        )

    def static_aux(instance):
        return tuple(
            (_typed_key(value), value)
            for value in (getattr(instance, name) for name in meta_fields)
        )

    def flatten_with_keys(instance):
        children = [
            (jax.tree_util.GetAttrKey(name), getattr(instance, name))
            for name in data_fields
        ]
        return children, static_aux(instance)

    def flatten(instance):
        return [getattr(instance, name) for name in data_fields], static_aux(instance)

    def unflatten(aux, children):
        instance = object.__new__(cls)
        for name, value in zip(data_fields, children, strict=True):
            object.__setattr__(instance, name, value)
        for name, (_, value) in zip(meta_fields, aux, strict=True):
            object.__setattr__(instance, name, value)
        return instance

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)
    return cls


def _tree_changed(new, old):
    new_leaves, new_treedef = jax.tree_util.tree_flatten(new)
    old_leaves, old_treedef = jax.tree_util.tree_flatten(old)
    if new_treedef != old_treedef:
        return jnp.asarray(True)
    changed = jnp.asarray(False)
    for new_leaf, old_leaf in zip(new_leaves, old_leaves, strict=True):
        if new_leaf is old_leaf:
            continue
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
    condition = jnp.asarray(condition, dtype=jnp.bool_)
    leaf_ndim = jnp.ndim(leaf)
    if condition.ndim < leaf_ndim:
        condition = jnp.reshape(
            condition, condition.shape + (1,) * (leaf_ndim - condition.ndim)
        )
    return condition


def _where_tree(condition, on_true, on_false):
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
    if isinstance(value, tuple):
        return (tuple, tuple(_typed_key(v) for v in value))
    if isinstance(value, frozenset):
        return (frozenset, frozenset(_typed_key(v) for v in value))
    return (type(value), value)


class _IdentityKey:
    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return isinstance(other, _IdentityKey) and self.obj is other.obj

    def __hash__(self):
        return id(self.obj)


def _static_key_component(value):
    try:
        hash(value)
    except TypeError:
        return _IdentityKey(value)
    return value


class _IdentityCallable:
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

        def canonical_residual(x, _args, _p):
            return residual_fn(x)

    elif residual_arity == 2:

        def canonical_residual(x, args, _p):
            return residual_fn(x, args)

    else:
        canonical_residual = residual_fn
    return canonical_residual, residual_arity
