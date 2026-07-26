"""Solver-agnostic helpers: pytree selection/masking, jit static-key hashing,
residual-signature canonicalization, and the pinned-precision matrix product.

No solver or metric code lives here -- only plumbing both
``LevenbergMarquardt`` and ``RidgeLevenbergMarquardt`` share.
"""

import dataclasses
import inspect

import jax
import jax.numpy as jnp

# XLA:GPU serves float32 dot_general from TF32 tensor cores by default: a
# 10-bit mantissa, so ~1e-3 relative error. Forming a Gram or normal matrix
# already squares the condition number, and doing that at TF32 spends about
# three decimal digits before the factorization starts -- enough to move a
# converged step and to put an implicit tangent visibly off. Every product
# inside the package goes through `mm` (or passes `precision=HIGHEST` to
# einsum) so a float32 solve answers the same on GPU as on CPU, where the
# setting is a no-op. Callers should also set
# `jax.config.update("jax_default_matmul_precision", "highest")` near the top
# of a script: that covers matmuls in their own residual functions, which the
# package cannot reach from here.
HIGHEST = jax.lax.Precision.HIGHEST


def mm(a, b):
    return jnp.matmul(a, b, precision=HIGHEST)


def register_pytree_dataclass(cls, *, data_fields, meta_fields=()):
    """Register a frozen dataclass as a pytree whose unflatten bypasses
    ``__init__``/``__post_init__``.

    ``data_fields`` become traced leaves (subtrees, if a field holds a
    container); ``meta_fields`` become static structure and must hold
    hashable values. Unflatten rebuilds the instance with ``object.__new__``
    plus ``object.__setattr__``, so constructors are free to compute derived
    leaves and validate eagerly -- reconstruction inside jit, ``vmap``, or a
    loop carry restores the stored fields verbatim without re-running any of
    it. Static values are type-tagged, so treedefs compare and hash by value
    with jit's strict-type semantics (``1``, ``1.0``, and ``True`` stay
    distinct). Every dataclass field must appear in exactly one of the two
    lists. Returns ``cls``.

    Every concrete :class:`~nlls_gram.Metric` and
    :class:`~nlls_gram.Preconditioner` class must be registered this way
    (subclassing a registered base does not register the subclass); the
    solvers reject unregistered instances at construction.
    """
    data_fields = tuple(data_fields)
    meta_fields = tuple(meta_fields)
    declared = {f.name for f in dataclasses.fields(cls)}
    listed = set(data_fields) | set(meta_fields)
    if len(data_fields) + len(meta_fields) != len(listed) or listed != declared:
        raise ValueError(
            f"register_pytree_dataclass({cls.__name__}): data_fields + "
            f"meta_fields must cover every dataclass field exactly once; "
            f"declared {sorted(declared)}, listed {sorted(listed)}"
        )

    def static_aux(instance):
        # The typed tag makes equality/hash strict-typed; the raw value rides
        # alongside so unflatten can restore it verbatim.
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
        # The same tracer/array object is the same value: a callback that
        # passes a subtree through dataclasses.replace untouched costs no
        # comparison ops. equal_nan: an unchanged NaN sentinel is not a
        # change.
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
