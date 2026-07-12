import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import numpy as np
from jax.flatten_util import ravel_pytree

from nlls_gram.metrics import Metric

# init() -> lm_state, update(x, lm_state, args, p) -> (new_x, lm_state, info),
# plus a solve() convenience loop. x is ANY pytree; the solver only ravels and
# unravels it with ravel_pytree and knows nothing about flax/nnx/optax.
# update() does not jit internally; solve(jit=True) wraps the loop in jax.jit.
# Hyperparameters are static Python scalars; data-dependent control flow is
# traced (jnp.where), so a rejected step returns the unchanged x rather than
# branching. Dtypes flow from the residual; damping scalars are cast to match.


class LMStatus:
    """Integer status codes returned by ``solve``."""

    RUNNING = 0
    CONVERGED = 1
    MAX_STEPS = 2
    NONFINITE = 3
    CALLBACK_STOP = 4


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMHyperparams:
    """Per-step LM hyperparameters, carried in ``LMState.hyper``.

    All fields are traced values, so a ``solve`` callback can reset them —
    e.g. grow the inner CG budget as the loss falls — via
    ``dataclasses.replace(ctx.lm_state, hyper=dataclasses.replace(
    ctx.lm_state.hyper, iterative_maxiter=...))``. A field constructed as
    ``None`` (uncapped ``max_damping``, backend-default ``iterative_maxiter``)
    is compiled out and stays ``None``. Static configuration (``linear_solver``,
    ``geodesic_acceleration``, ``cache_jacobian``, ``has_aux``, the metric)
    shapes the compiled program and lives on the solver, not here.
    """

    damping_decrease: jax.Array
    damping_increase: jax.Array
    max_damping: jax.Array | None
    geodesic_acceptance_ratio: jax.Array
    iterative_tol: jax.Array
    iterative_atol: jax.Array
    iterative_maxiter: jax.Array | None


def _cast_hyper(hyper, dtype):
    if hyper is None:
        return None
    return LMHyperparams(
        jnp.asarray(hyper.damping_decrease, dtype=dtype),
        jnp.asarray(hyper.damping_increase, dtype=dtype),
        None
        if hyper.max_damping is None
        else jnp.asarray(hyper.max_damping, dtype=dtype),
        jnp.asarray(hyper.geodesic_acceptance_ratio, dtype=dtype),
        jnp.asarray(hyper.iterative_tol, dtype=dtype),
        jnp.asarray(hyper.iterative_atol, dtype=dtype),
        None
        if hyper.iterative_maxiter is None
        else jnp.asarray(hyper.iterative_maxiter, dtype=jnp.int32),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMState:
    damping: jax.Array
    # Jacobian cache (cache_jacobian=True only): residual, J', and aux at the
    # current x; jacobian_valid means the last step was rejected so x did not move.
    resid: jax.Array | None = None
    Jt: jax.Array | None = None
    jacobian_valid: jax.Array | None = None
    aux: Any = None  # arbitrary residual aux pytree
    # LMHyperparams, populated by solve(); None (init()'s default) falls back
    # to the constructor values with identical compiled code and no extra
    # per-call buffers in manual update() loops.
    hyper: LMHyperparams | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMInfo:
    loss: jax.Array  # min(old, new) sum of squared residuals
    loss_old: jax.Array
    loss_candidate: jax.Array
    accepted: jax.Array
    damping: jax.Array  # post-update damping
    damping_factor: jax.Array
    used_geodesic: jax.Array
    acceleration_ratio: jax.Array
    grad_norm: jax.Array  # ||J' r|| at the pre-step x
    step_norm: jax.Array  # ||candidate step||, reported even when rejected
    aux: Any = None  # residual aux output at the pre-step x (has_aux)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveAction:
    """Optional callback action for ``solve``.

    A field left as ``None`` is unchanged. ``status`` is used only when ``stop``
    is true.
    """

    stop: Any = None
    status: Any = None
    x: Any = None
    lm_state: Any = None
    args: Any = None
    user_state: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveContext:
    """Information passed to a ``solve`` callback after each LM update."""

    step: jax.Array
    x: Any
    x_old: Any
    lm_state: LMState
    lm_state_old: LMState
    initial_lm_state: LMState
    args: Any
    p: Any
    user_state: Any
    info: LMInfo


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMSolveResult:
    """Final result returned by ``solve``."""

    x: Any
    lm_state: LMState
    info: LMInfo
    steps: jax.Array
    status: jax.Array
    args: Any
    p: Any
    user_state: Any
    # With has_aux=True: aux evaluated at the returned (x, args, p) — one extra
    # residual evaluation, well-defined for every status. Differentiable with
    # respect to p through the implicit rule (directly and through x*(p)).
    aux: Any = None
    # With save_steps=True: the iterate history as a pytree shaped like x with a
    # (max_steps + 1) leading axis — row 0 is x0, row s the kept iterate after
    # step s (post-callback-action), rows beyond ``steps`` are zero padding.
    # aux_history (has_aux only, else None) and args_history (None when args is
    # None) align row-for-row with x_history — args row s is the kept
    # post-action args after step s, the args consumed by step s + 1's update.
    # Differentiation-inert (zero tangents through the implicit rule).
    x_history: Any = None
    aux_history: Any = None
    args_history: Any = None


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
    # Hashable settings (scalars, strings, functions, frozen metrics) key by
    # value; anything unhashable keys by identity so hashing never raises and
    # equality stays consistent with the hash.
    try:
        hash(value)
    except TypeError:
        return _IdentityKey(value)
    return value


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


def canonicalize_implicit_preconditioner(implicit_preconditioner):
    """Normalize an ``implicit_preconditioner`` to the 1-arg form the
    implicit solve calls. A callable already usable as ``(v)`` -- every
    pre-1.7 callback, and helpers whose damping argument has a default --
    passes through unchanged, so no existing behavior shifts. A callable
    REQUIRING a second argument (a ``(v, damping)`` dual helper such as
    Sherman-Morrison or Woodbury) is wrapped to be called with an explicit
    zero damping, the correct value for the undamped implicit dual.
    Helpers marked ``requires_positive_damping`` (``pad_dual_preconditioner``)
    are rejected at construction: their zero-damping apply divides by zero.
    Uninspectable signatures pass through unchanged (the historical 1-arg
    contract).
    """
    if getattr(implicit_preconditioner, "requires_positive_damping", False):
        raise ValueError(
            "this preconditioner divides by the live damping and cannot "
            "serve as implicit_preconditioner (the implicit dual system is "
            "undamped)"
        )
    try:
        signature = inspect.signature(implicit_preconditioner)
    except (TypeError, ValueError):
        return implicit_preconditioner
    try:
        signature.bind(object())
    except TypeError:
        pass
    else:
        return implicit_preconditioner
    try:
        signature.bind(object(), object())
    except TypeError:
        raise ValueError(
            "implicit_preconditioner must be callable as (v) or (v, damping)"
        ) from None

    def canonical_implicit_preconditioner(v):
        return implicit_preconditioner(v, jnp.asarray(0.0, dtype=v.dtype))

    return canonical_implicit_preconditioner


class UnderdeterminedLevenbergMarquardt:
    """Metric-damped Levenberg-Marquardt for ``min ||r(x, args, p)||^2`` over a
    JAX pytree ``x``, specialized to ``n_residuals << n_params``: the default
    path factors the small damped Gram system in residual space. With metric
    ``M`` and ``P = M^{-1}``, the dense dual step is
    ``step = -P J' (J P J' + damping I_m)^{-1} r``. Exposes per-step
    ``update``, a jitted ``solve`` loop with callbacks, and implicit
    differentiation of ``solve`` with respect to ``p``.

    ``linear_solver="cg"`` requires ``dual_preconditioner(v, damping)``: a
    jit-traceable, linear, SPD approximation of
    ``(J P J' + damping I_m)^{-1} v`` used as the CG preconditioner (for the
    geodesic-acceleration solve as well); pass ``identity_preconditioner()``
    to run unpreconditioned CG. It never changes the subproblem: at
    inner convergence the step is identical, and a budget-truncated step still
    lies in ``range(P J')``, so the minimum-metric-norm selection for
    underdetermined residuals is unchanged — the preconditioner may be
    approximate even though ``metric.solve`` must stay exact.

    ``solve(...).x`` has a custom implicit AD rule with respect to ``p``. By
    default, ``implicit_solver="auto"`` uses a fully matrix-free CG implicit
    solve when the forward solver is ``linear_solver="cg"``, and otherwise uses
    the legacy dense Cholesky solve. Pass ``implicit_solver="cholesky"`` to
    force the dense implicit rule, or ``implicit_solver="cg"`` to force the
    matrix-free rule. A cg-resolved implicit solve likewise requires
    ``implicit_preconditioner``, an approximation of the UNDAMPED
    ``(J P J')^{-1} v``, taking either ``(v)`` or ``(v, damping)`` -- a
    callable REQUIRING the damping argument is called with an explicit
    zero, and one where it has a default passes through unchanged, so every
    shipped dual helper serves both hooks directly except
    ``pad_dual_preconditioner``, which divides by the live damping and is
    rejected here at construction.

    ``implicit_penalty`` regularizes the DENSE implicit rule only (the cg
    implicit rule ignores it): the implicit dual is factored as
    ``J P J' + implicit_penalty * trace(J P J') I_m``. Redundant residual
    rows at the returned solution -- e.g. a simulated trajectory settled onto
    its steady state -- make the undamped dual singular and the unregularized
    factorization non-finite; for such consistent systems the ridge returns
    the minimum-norm tangent with an O(``implicit_penalty * m``) relative
    bias. The default ``None`` resolves to ``eps`` of the dual-solve dtype
    (~2.2e-16 in float64, ~1.2e-7 in float32, after any ``dual_solve_dtype``
    promotion) -- the classic semidefinite-jitter scale, negligible against
    any well-conditioned tangent; pass ``0.0`` to restore the exact
    unregularized rule, whose non-finite tangents signal a singular dual
    loudly.

    ``dual_solve_dtype=jnp.float64`` promotes the dual (Gram) solve of the
    dense cholesky paths -- the forward ``linear_solver="cholesky"`` branch
    and the dense implicit rule -- to float64: ``J'`` is cast wide before the
    metric solve, the m x m assembly, factorization, and triangular solves
    run wide, and only the returned step/tangent is cast back, so the model,
    residual, and every output stay at the residual dtype. Forming
    ``J P J'`` squares the condition number of the whitened Jacobian
    ``J S`` (``S S' = P``), which is what makes the dense paths
    float32-fragile; the promotion buys full-x64 robustness for the dual
    solve at the cost of up to two wide n x m intermediates and roughly
    1.4x per cholesky update measured at m=100, n=2000 with a trivial
    residual (real residual/Jacobian costs dominate and stay float32).
    ``metric.solve`` receives the promoted dtype and must return it
    (jnp-composed callbacks promote automatically; an iterative callback
    like ``metric_from_shifted_matvec`` then runs its inner CG in float64
    with the float64 default tolerance, at callback-dependent cost).
    Requires x64 support to be enabled -- which by itself leaves explicitly
    float32 data in float32.
    """

    def __init__(
        self,
        residual_fn,
        *,
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        max_damping=None,
        linear_solver="cholesky",
        iterative_tol=0.0,
        iterative_atol=0.0,
        iterative_maxiter=8,
        dual_preconditioner=None,
        implicit_solver="auto",
        implicit_tol=None,
        implicit_atol=0.0,
        implicit_maxiter=None,
        implicit_preconditioner=None,
        implicit_penalty=None,
        dual_solve_dtype=None,
        metric=None,
        has_aux=False,
        cache_jacobian=True,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    ):
        canonical_residual, residual_arity = canonicalize_residual(residual_fn)
        if linear_solver not in ("cholesky", "qr", "cg"):
            raise ValueError(f"unknown linear_solver: {linear_solver}")
        if init_damping <= 0:
            raise ValueError("init_damping must be positive")
        if damping_decrease <= 0:
            raise ValueError("damping_decrease must be positive")
        if damping_increase <= 0:
            raise ValueError("damping_increase must be positive")
        if max_damping is not None and max_damping < init_damping:
            raise ValueError("max_damping must be at least init_damping")
        if iterative_tol < 0:
            raise ValueError("iterative_tol must be nonnegative")
        if iterative_atol < 0:
            raise ValueError("iterative_atol must be nonnegative")
        if iterative_maxiter is not None and iterative_maxiter <= 0:
            raise ValueError("iterative_maxiter must be positive or None")
        if iterative_tol == 0 and iterative_atol == 0 and iterative_maxiter is None:
            raise ValueError(
                "iterative_maxiter must be set when both iterative tolerances are zero"
            )
        if dual_preconditioner is not None and linear_solver != "cg":
            raise ValueError('dual_preconditioner requires linear_solver="cg"')
        if implicit_solver not in ("auto", "cholesky", "cg"):
            raise ValueError(f"unknown implicit_solver: {implicit_solver}")
        if implicit_tol is not None and implicit_tol < 0:
            raise ValueError("implicit_tol must be nonnegative or None")
        if implicit_atol < 0:
            raise ValueError("implicit_atol must be nonnegative")
        if implicit_maxiter is not None and implicit_maxiter <= 0:
            raise ValueError("implicit_maxiter must be positive or None")
        if implicit_penalty is not None and implicit_penalty < 0:
            raise ValueError("implicit_penalty must be nonnegative or None")
        if implicit_tol == 0 and implicit_atol == 0 and implicit_maxiter is None:
            raise ValueError(
                "implicit_maxiter must be set when both implicit tolerances are zero"
            )
        resolved_implicit_solver = (
            "cg"
            if implicit_solver == "cg"
            or (implicit_solver == "auto" and linear_solver == "cg")
            else "cholesky"
        )
        if implicit_preconditioner is not None and resolved_implicit_solver != "cg":
            raise ValueError(
                'implicit_preconditioner requires implicit_solver="cg" '
                'or implicit_solver="auto" with linear_solver="cg"'
            )
        missing_dual_preconditioner = (
            linear_solver == "cg" and dual_preconditioner is None
        )
        missing_implicit_preconditioner = (
            resolved_implicit_solver == "cg" and implicit_preconditioner is None
        )
        if missing_dual_preconditioner and missing_implicit_preconditioner:
            raise ValueError(
                'linear_solver="cg" requires dual_preconditioner, and the '
                'cg-resolved implicit solver ("cg", or "auto" alongside a cg '
                "forward solver) requires implicit_preconditioner; pass "
                "identity_preconditioner() for either to run unpreconditioned "
                'CG, or implicit_solver="cholesky" for the dense implicit rule'
            )
        if missing_dual_preconditioner:
            raise ValueError(
                'linear_solver="cg" requires dual_preconditioner; pass '
                "identity_preconditioner() to run unpreconditioned CG"
            )
        if missing_implicit_preconditioner:
            raise ValueError(
                'implicit_solver="cg" requires implicit_preconditioner '
                '(implicit_solver="auto" resolves to cg when '
                'linear_solver="cg"); pass identity_preconditioner() to run '
                'unpreconditioned CG, or implicit_solver="cholesky" for the '
                "dense implicit rule"
            )
        if dual_solve_dtype is not None:
            if jnp.dtype(dual_solve_dtype) != jnp.dtype(jnp.float64):
                raise ValueError("dual_solve_dtype must be None or jnp.float64")
            if linear_solver != "cholesky" and resolved_implicit_solver != "cholesky":
                raise ValueError(
                    "dual_solve_dtype promotes only the dense cholesky paths; "
                    'it requires linear_solver="cholesky" or a '
                    "cholesky-resolved implicit solver"
                )
            if not jax.config.jax_enable_x64:
                raise ValueError(
                    "dual_solve_dtype=jnp.float64 requires x64 support; call "
                    'jax.config.update("jax_enable_x64", True) at startup '
                    "(explicitly float32 problem data stays float32)"
                )
        if metric is None:
            metric = Metric()
        has_custom_metric = any(
            cb is not None
            for cb in (
                metric.solve,
                metric.norm,
                metric.inv_sqrt,
                metric.inv_sqrt_transpose,
            )
        )
        if has_custom_metric and linear_solver in ("cholesky", "cg"):
            if metric.solve is None:
                raise ValueError(
                    f'linear_solver="{linear_solver}" with a custom metric requires '
                    "metric.solve"
                )
        if has_custom_metric and linear_solver == "qr":
            if metric.inv_sqrt is None or metric.inv_sqrt_transpose is None:
                raise ValueError(
                    'linear_solver="qr" with a custom metric requires '
                    "metric.inv_sqrt and metric.inv_sqrt_transpose"
                )
        if has_custom_metric and geodesic_acceleration and metric.norm is None:
            raise ValueError(
                "geodesic_acceleration (on by default) with a custom metric "
                "requires metric.norm; provide it or pass "
                "geodesic_acceleration=False"
            )
        self.residual_fn = canonical_residual
        self.residual_arity = residual_arity
        self.init_damping = init_damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.max_damping = max_damping
        self.linear_solver = linear_solver
        self.iterative_tol = iterative_tol
        self.iterative_atol = iterative_atol
        self.iterative_maxiter = iterative_maxiter
        self.dual_preconditioner = dual_preconditioner
        self.implicit_solver = implicit_solver
        self.implicit_tol = implicit_tol
        self.implicit_atol = implicit_atol
        self.implicit_maxiter = implicit_maxiter
        self.implicit_preconditioner = (
            None
            if implicit_preconditioner is None
            else canonicalize_implicit_preconditioner(implicit_preconditioner)
        )
        self.implicit_penalty = implicit_penalty
        self.dual_solve_dtype = (
            None if dual_solve_dtype is None else jnp.dtype(dual_solve_dtype)
        )
        self._resolved_implicit_solver = resolved_implicit_solver
        self.metric = metric
        # Only the dense cholesky path materializes J', so the flag is inert
        # for the other solvers.
        self.cache_jacobian = cache_jacobian and linear_solver == "cholesky"
        self.has_aux = has_aux
        self._has_custom_metric = has_custom_metric
        self._has_metric_solve = metric.solve is not None
        self.metric_solve = (lambda x: x) if metric.solve is None else metric.solve
        self.metric_norm = (
            (lambda x: jnp.linalg.norm(x)) if metric.norm is None else metric.norm
        )
        self.metric_inv_sqrt = (
            (lambda x: x) if metric.inv_sqrt is None else metric.inv_sqrt
        )
        self.metric_inv_sqrt_transpose = (
            (lambda x: x)
            if metric.inv_sqrt_transpose is None
            else metric.inv_sqrt_transpose
        )
        self.geodesic_acceleration = geodesic_acceleration
        self.geodesic_acceptance_ratio = geodesic_acceptance_ratio
        # Value-based identity: the jitted solve loop marks the solver itself
        # static, so equal-config solvers built around the same residual (and
        # metric/preconditioner objects) share the compiled loop across
        # instances instead of retracing once per construction. Keyed on the
        # constructor arguments -- every derived attribute is a function of them.
        self._static_key = tuple(
            _static_key_component(value)
            for value in (
                residual_fn,
                init_damping,
                damping_decrease,
                damping_increase,
                max_damping,
                linear_solver,
                iterative_tol,
                iterative_atol,
                iterative_maxiter,
                dual_preconditioner,
                implicit_solver,
                implicit_tol,
                implicit_atol,
                implicit_maxiter,
                implicit_preconditioner,
                implicit_penalty,
                self.dual_solve_dtype,
                metric,
                has_aux,
                self.cache_jacobian,
                geodesic_acceleration,
                geodesic_acceptance_ratio,
            )
        )
        self._static_hash = hash(self._static_key)

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._static_key == other._static_key

    def __hash__(self):
        return self._static_hash

    def hyperparams(self, dtype=None):
        """``LMHyperparams`` built from the constructor values."""
        return LMHyperparams(
            jnp.asarray(self.damping_decrease, dtype=dtype),
            jnp.asarray(self.damping_increase, dtype=dtype),
            None
            if self.max_damping is None
            else jnp.asarray(self.max_damping, dtype=dtype),
            jnp.asarray(self.geodesic_acceptance_ratio, dtype=dtype),
            jnp.asarray(self.iterative_tol, dtype=dtype),
            jnp.asarray(self.iterative_atol, dtype=dtype),
            None
            if self.iterative_maxiter is None
            else jnp.asarray(self.iterative_maxiter, dtype=jnp.int32),
        )

    def init(self, x0, args=None, *, p=None):
        # One residual evaluation types the damping to match what update()
        # returns (keeping the jit signature and solve-loop carry stable) and
        # sizes the Jacobian cache buffers when cache_jacobian=True. hyper
        # stays None so manual update() loops carry no extra buffers; solve()
        # populates it for its callbacks.
        self._check_residual_args(args, p)
        residual, aux = self._residual_and_aux(x0, args, p)
        damping = jnp.asarray(self.init_damping, dtype=residual.dtype)
        if not self.cache_jacobian:
            return LMState(damping)
        theta, _ = ravel_pytree(x0)
        return LMState(
            damping,
            jnp.zeros(residual.shape, dtype=residual.dtype),
            jnp.zeros((theta.size, residual.size), dtype=residual.dtype),
            jnp.asarray(False, dtype=jnp.bool_),
            jax.tree.map(jnp.zeros_like, aux),
        )

    def _residual_and_aux(self, x, args, p):
        if self.has_aux:
            value, aux = self.residual_fn(x, args, p)
            # aux rides through the jitted loop carry and the implicit-AD
            # zero-tangent map, so non-numeric leaves can never work; fail
            # here with a clear message instead of a dtype error deep in
            # the trace.
            for leaf in jax.tree.leaves(aux):
                if not isinstance(
                    leaf, (jax.Array, np.ndarray, np.generic, bool, int, float, complex)
                ):
                    raise TypeError(
                        "has_aux=True: aux leaves must be JAX numeric types "
                        f"(arrays or scalars); got {type(leaf).__name__}"
                    )
            return jnp.ravel(value), aux
        return jnp.ravel(self.residual_fn(x, args, p)), None

    def _initial_info(self, x, lm_state, args, p):
        # grad_norm is a +inf sentinel (computing it would cost a Jacobian
        # before the first step) and step_norm is zero; neither can satisfy
        # gtol/xtol before any update has run.
        residual, aux = self._residual_and_aux(x, args, p)
        loss = jnp.sum(residual**2)
        zero = jnp.zeros((), dtype=residual.dtype)
        one = jnp.ones((), dtype=residual.dtype)
        return LMInfo(
            loss,
            loss,
            loss,
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(lm_state.damping, dtype=residual.dtype),
            one,
            jnp.asarray(False, dtype=jnp.bool_),
            zero,
            jnp.asarray(jnp.inf, dtype=residual.dtype),
            zero,
            aux,
        )

    def update(self, x, lm_state, args=None, p=None):
        self._check_residual_args(args, p)
        # Linearize at x: flatten the pytree and view the residual over theta.
        theta, unravel = ravel_pytree(x)

        if self.has_aux:

            def residual_flat(th):
                value, aux = self.residual_fn(unravel(th), args, p)
                return jnp.ravel(value), aux

            def residual_value(th):
                return residual_flat(th)[0]

        else:

            def residual_flat(th):
                return jnp.ravel(self.residual_fn(unravel(th), args, p))

            residual_value = residual_flat

        # Build J': matrix-free JVP/VJP closures for cg; m VJP passes for the
        # dense paths, reused from the cache after a rejected step.
        if self.linear_solver == "cg":
            if self.has_aux:
                resid, jvp_fn, aux = jax.linearize(residual_flat, theta, has_aux=True)
            else:
                resid, jvp_fn = jax.linearize(residual_flat, theta)
                aux = None
        elif self.cache_jacobian:
            if lm_state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True but the lm_state has no Jacobian cache; "
                    "create the lm_state with init(x, args, p=p)"
                )

            def compute_resid_and_jt(_):
                if self.has_aux:
                    resid, pullback, aux = jax.vjp(residual_flat, theta, has_aux=True)
                else:
                    resid, pullback = jax.vjp(residual_flat, theta)
                    aux = None
                residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
                Jt = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
                    residual_basis
                ).T
                return resid, Jt, aux

            def reuse_resid_and_jt(_):
                return lm_state.resid, lm_state.Jt, lm_state.aux

            resid, Jt, aux = jax.lax.cond(
                lm_state.jacobian_valid,
                reuse_resid_and_jt,
                compute_resid_and_jt,
                operand=None,
            )
        else:
            if self.has_aux:
                resid, pullback, aux = jax.vjp(residual_flat, theta, has_aux=True)
            else:
                resid, pullback = jax.vjp(residual_flat, theta)
                aux = None
        damping = jnp.asarray(lm_state.damping, dtype=resid.dtype)
        # Traced hyperparameters from the lm_state when present (resettable by
        # solve callbacks); the None fallback compiles to the same constants
        # as reading the constructor values directly.
        hyper = (
            lm_state.hyper
            if lm_state.hyper is not None
            else self.hyperparams(resid.dtype)
        )
        damping_decrease = jnp.asarray(hyper.damping_decrease, dtype=resid.dtype)
        damping_increase = jnp.asarray(hyper.damping_increase, dtype=resid.dtype)

        if self.linear_solver == "cg":
            transpose_fn = jax.linear_transpose(jvp_fn, theta)

            def JT(cotangent):
                return transpose_fn(cotangent)[0]

            grad = JT(resid)
            # Typed tolerances keep CG's scalars in the residual dtype under x64.
            cg_tol = jnp.asarray(hyper.iterative_tol, dtype=resid.dtype)
            cg_atol = jnp.asarray(hyper.iterative_atol, dtype=resid.dtype)

            def gram_matvec(cotangent):
                return jvp_fn(self.metric_solve(JT(cotangent))) + damping * cotangent

            def cg_preconditioner(cotangent):
                return self.dual_preconditioner(cotangent, damping)

            def solve_step(rhs):
                dual_solution, _ = jsp_sparse_linalg.cg(
                    gram_matvec,
                    rhs,
                    tol=cg_tol,
                    atol=cg_atol,
                    maxiter=hyper.iterative_maxiter,
                    M=cg_preconditioner,
                )
                return -self.metric_solve(JT(dual_solution))

        else:
            if not self.cache_jacobian:
                residual_basis = jnp.eye(resid.shape[0], dtype=resid.dtype)
                Jt = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
                    residual_basis
                ).T
            grad = Jt @ resid

            if self.linear_solver == "qr":
                # Whitened factorization: augmented QR of S'J' with sqrt(damping).
                transformed_Jt = self.metric_inv_sqrt_transpose(Jt)
                if transformed_Jt.shape[0] >= transformed_Jt.shape[1]:
                    R = jnp.linalg.qr(transformed_Jt, mode="r")
                    basis_eye = jnp.eye(R.shape[0], dtype=resid.dtype)
                    augmented_matrix = jnp.concatenate(
                        (R.T, jnp.sqrt(damping) * basis_eye),
                        axis=0,
                    )
                    Qa, Ra = jnp.linalg.qr(augmented_matrix, mode="reduced")

                    def solve_step(rhs):
                        augmented_rhs = jnp.concatenate(
                            (-rhs, jnp.zeros(R.shape[0], dtype=rhs.dtype))
                        )
                        z = jsp_linalg.solve_triangular(
                            Ra,
                            Qa.T @ augmented_rhs,
                        )
                        y = jsp_linalg.solve_triangular(R, z)
                        return self.metric_inv_sqrt(transformed_Jt @ y)

                else:
                    Q, R = jnp.linalg.qr(transformed_Jt, mode="reduced")
                    basis_eye = jnp.eye(R.shape[0], dtype=resid.dtype)
                    augmented_matrix = jnp.concatenate(
                        (R.T, jnp.sqrt(damping) * basis_eye),
                        axis=0,
                    )
                    Qa, Ra = jnp.linalg.qr(augmented_matrix, mode="reduced")

                    def solve_step(rhs):
                        augmented_rhs = jnp.concatenate(
                            (-rhs, jnp.zeros(R.shape[0], dtype=rhs.dtype))
                        )
                        z = jsp_linalg.solve_triangular(
                            Ra,
                            Qa.T @ augmented_rhs,
                        )
                        return self.metric_inv_sqrt(Q @ z)

            else:
                # Damped Gram factorization: cholesky of J P J' + damping I.
                # With dual_solve_dtype the whole dual pipeline runs wide:
                # J' is promoted BEFORE the metric solve (a stiff metric's
                # 1/eps rows amplify float32 rounding of P J' into O(1) Gram
                # errors -- promoting only the assembly was measured ~4
                # digits worse), and the dual solution stays wide through the
                # final product. Only the returned step is cast back, so it
                # keeps the residual dtype. metric.solve therefore receives
                # the promoted dtype; jnp-composed callbacks promote
                # automatically.
                dual_dtype = (
                    resid.dtype
                    if self.dual_solve_dtype is None
                    else self.dual_solve_dtype
                )
                transposed_jacobian = Jt.astype(dual_dtype)
                gram_step_left = self.metric_solve(transposed_jacobian)
                linear_matrix = transposed_jacobian.T @ gram_step_left
                linear_matrix = linear_matrix + jnp.asarray(
                    damping, dtype=dual_dtype
                ) * jnp.eye(resid.shape[0], dtype=dual_dtype)

                linear_factor = jsp_linalg.cho_factor(linear_matrix)

                def solve_step(rhs):
                    dual_solution = jsp_linalg.cho_solve(
                        linear_factor, rhs.astype(dual_dtype)
                    )
                    step = -gram_step_left @ dual_solution
                    return step.astype(resid.dtype)

        # Dual solve for the first-order step (velocity).
        velocity = solve_step(resid)
        resid_velocity = residual_value(theta + velocity)
        loss_old = jnp.sum(resid**2)
        loss_velocity = jnp.sum(resid_velocity**2)
        zero = jnp.zeros((), dtype=resid.dtype)

        # Geodesic second-order correction, solved with the same factorization.
        if self.geodesic_acceleration:
            geodesic_acceptance_ratio = jnp.asarray(
                hyper.geodesic_acceptance_ratio, dtype=resid.dtype
            )

            def first_jvp(th):
                # [1] is the tangent with and without has_aux.
                return jax.jvp(residual_flat, (th,), (velocity,), has_aux=self.has_aux)[
                    1
                ]

            f_vv = jax.jvp(first_jvp, (theta,), (velocity,))[1]
            acceleration = solve_step(f_vv)
            accelerated_step = velocity + 0.5 * acceleration
            acceleration_ratio = (
                2.0
                * self.metric_norm(acceleration)
                / (self.metric_norm(velocity) + jnp.finfo(resid.dtype).eps)
            )
            ratio_accepted = (
                (geodesic_acceptance_ratio > zero)
                & (acceleration_ratio > zero)
                & (acceleration_ratio <= geodesic_acceptance_ratio)
            )

            def accelerated_loss(_):
                resid_accelerated = residual_value(theta + accelerated_step)
                return jnp.sum(resid_accelerated**2)

            loss_accelerated = jax.lax.cond(
                ratio_accepted,
                accelerated_loss,
                lambda _: jnp.asarray(jnp.inf, dtype=resid.dtype),
                operand=None,
            )
            used_geodesic = ratio_accepted & (loss_accelerated <= loss_velocity)
            step = jnp.where(used_geodesic, accelerated_step, velocity)
            loss_candidate = jnp.where(used_geodesic, loss_accelerated, loss_velocity)
        else:
            step = velocity
            loss_candidate = loss_velocity
            used_geodesic = jnp.asarray(False)
            acceleration_ratio = zero

        # Accept iff the sum of squared residuals decreases and is finite.
        improved = jnp.isfinite(loss_candidate) & (loss_candidate < loss_old)
        theta_new = jnp.where(improved, theta + step, theta)
        # Damping update: decrease on acceptance, increase on rejection.
        damping_factor = jnp.where(improved, damping_decrease, damping_increase)
        new_damping = damping * damping_factor
        if hyper.max_damping is not None:
            new_damping = jnp.minimum(
                new_damping, jnp.asarray(hyper.max_damping, dtype=resid.dtype)
            )
        loss = jnp.where(improved, loss_candidate, loss_old)
        # The input hyper (not the fallback) passes through so the loop carry
        # structure and dtypes are stable.
        if self.cache_jacobian:
            new_lm_state = LMState(
                new_damping, resid, Jt, ~improved, aux, lm_state.hyper
            )
        else:
            new_lm_state = LMState(new_damping, hyper=lm_state.hyper)
        return (
            unravel(theta_new),
            new_lm_state,
            LMInfo(
                loss,
                loss_old,
                loss_candidate,
                improved,
                new_damping,
                damping_factor,
                used_geodesic,
                acceleration_ratio,
                jnp.linalg.norm(grad),
                jnp.linalg.norm(step),
                aux,
            ),
        )

    def solve(
        self,
        x0,
        args=None,
        *,
        p=None,
        lm_state=None,
        max_steps=256,
        atol=0.0,
        gtol=0.0,
        xtol=0.0,
        callback=None,
        user_state=None,
        save_steps=False,
        jit=True,
    ):
        """Run repeated LM updates until a stopping rule fires.

        Parameters are the same as ``update`` plus loop controls. ``max_steps``
        is always enforced. ``atol`` stops when the residual norm is below the
        threshold, ``gtol`` when the gradient norm ``||J' r||`` is below the
        threshold, and ``xtol`` when an accepted step has norm below the
        threshold; each tolerance set to ``0`` disables that check, and all
        three report ``LMStatus.CONVERGED``. ``callback`` receives an
        ``LMSolveContext`` after each step and may return an ``LMSolveAction``
        to stop or to override x/lm_state/args/user_state. ``p`` is passed to
        the residual and callback but cannot be replaced by the action.
        ``save_steps=True`` records the full iterate history onto the result:
        ``x_history`` stacks x0 and every kept post-step iterate along a
        ``(max_steps + 1)`` leading axis (rows beyond ``steps`` are zero
        padding — slice with ``result.steps``), plus the row-aligned
        ``args_history`` (the kept post-action args, recorded even when no
        callback ever replaces them; ``None`` when ``args`` is ``None``) and,
        with ``has_aux``, the row-aligned ``aux_history``. The history buffers
        cost ``(max_steps + 1) x (size(x) + size(args) [+ size(aux)])``
        memory, are differentiation-inert, and (unlike the default) make the
        jitted loop retrace when ``max_steps`` changes, since the buffer shape
        depends on it.
        """
        self._check_residual_args(args, p)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        # Tolerances are traced data inside the loop, so vmapped/traced values
        # skip the concrete-only sign validation.
        if not isinstance(atol, jax.core.Tracer) and atol < 0:
            raise ValueError("atol must be nonnegative")
        if not isinstance(gtol, jax.core.Tracer) and gtol < 0:
            raise ValueError("gtol must be nonnegative")
        if not isinstance(xtol, jax.core.Tracer) and xtol < 0:
            raise ValueError("xtol must be nonnegative")
        if lm_state is None:
            # The loop recasts the damping and hyperparameter dtypes itself;
            # only the Jacobian cache needs an eager init() for the shapes.
            if self.cache_jacobian:
                lm_state = self.init(x0, args, p=p)
            else:
                lm_state = LMState(jnp.asarray(self.init_damping))
        if lm_state.hyper is None:
            # Populate here (not in init) so manual update() loops stay lean;
            # inside the loop the extra scalars are loop-carried, not
            # re-dispatched per step.
            lm_state = dataclasses.replace(lm_state, hyper=self.hyperparams())
        # The history buffers need a concrete length, so it is fixed here (like
        # callback and jit, via closure) and the buffers are allocated inside
        # the loop implementations; with save_steps=False nothing changes, and
        # with save_steps=True the buffer shape retraces per max_steps anyway.
        history_len = max_steps + 1 if save_steps else None

        @jax.custom_jvp
        def solve_with_implicit_p(
            x, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
        ):
            return self._solve_impl(
                x,
                lm_state,
                args,
                p,
                user_state,
                history_len,
                max_steps,
                atol,
                gtol,
                xtol,
                callback,
                jit,
            )

        @solve_with_implicit_p.defjvp
        def solve_with_implicit_p_jvp(primals, tangents):
            x, lm_state, args, p, user_state, max_steps, atol, gtol, xtol = primals
            _, _, _, p_dot, _, _, _, _, _ = tangents
            result = solve_with_implicit_p(
                x, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
            )
            x_dot = self._implicit_x_tangent_from_p(
                result.x, result.args, result.p, p_dot
            )
            zero_result = jax.tree.map(_zero_tangent_leaf, result)
            aux_dot = zero_result.aux
            if self.has_aux and p is not None:
                # aux depends on p directly and through the solution x*(p);
                # linearize the aux map at the returned solution with args
                # fixed (the same point where the primal result.aux is
                # evaluated) to account for both paths.
                def aux_at_solution(x_value, p_value):
                    return self.residual_fn(x_value, result.args, p_value)[1]

                aux_dot = jax.jvp(
                    aux_at_solution, (result.x, result.p), (x_dot, p_dot)
                )[1]
            # The iterate histories are training-trajectory bookkeeping, not
            # implicit functions of p: zero tangents.
            return (
                result,
                LMSolveResult(
                    x_dot,
                    zero_result.lm_state,
                    zero_result.info,
                    zero_result.steps,
                    zero_result.status,
                    zero_result.args,
                    p_dot,
                    zero_result.user_state,
                    aux_dot,
                    zero_result.x_history,
                    zero_result.aux_history,
                    zero_result.args_history,
                ),
            )

        return solve_with_implicit_p(
            x0, lm_state, args, p, user_state, max_steps, atol, gtol, xtol
        )

    def _solve_impl(
        self,
        x,
        lm_state,
        args,
        p,
        user_state,
        history_len,
        max_steps,
        atol,
        gtol,
        xtol,
        callback,
        jit,
    ):
        if jit:
            return _solve_loop_jit(
                self,
                x,
                lm_state,
                args,
                p,
                user_state,
                history_len,
                max_steps,
                atol,
                gtol,
                xtol,
                callback,
            )
        return self._solve_python(
            x,
            lm_state,
            args,
            p,
            user_state,
            history_len,
            max_steps,
            atol,
            gtol,
            xtol,
            callback,
        )

    def _implicit_x_tangent_from_p(self, x, args, p, p_dot):
        if p is None:
            return jax.tree.map(_zero_tangent_leaf, x)
        if self._resolved_implicit_solver == "cg":
            return self._implicit_x_tangent_from_p_cg(x, args, p, p_dot)
        return self._implicit_x_tangent_from_p_cholesky(x, args, p, p_dot)

    def _implicit_x_tangent_from_p_cholesky(self, x, args, p, p_dot):
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)
        residual_basis = jnp.eye(residual.shape[0], dtype=residual.dtype)
        theta_transpose = jax.linear_transpose(theta_jvp, theta)
        Jt = jax.vmap(lambda cotangent: theta_transpose(cotangent)[0])(residual_basis).T

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        # The undamped implicit Gram has no + damping I floor, so it benefits
        # most from dual_solve_dtype promotion; same recipe as the forward
        # cholesky branch (J' promoted before the metric solve, wide
        # factor/solve and final product, only the tangent cast back to the
        # residual dtype).
        dual_dtype = (
            residual.dtype if self.dual_solve_dtype is None else self.dual_solve_dtype
        )
        transposed_jacobian = Jt.astype(dual_dtype)
        gram_step_left = self._metric_inverse(transposed_jacobian)
        gram = transposed_jacobian.T @ gram_step_left
        # Tikhonov ridge scaled by the trace: redundant residual rows at the
        # returned solution (e.g. a simulated trajectory settled onto its steady
        # state) make the undamped dual singular and the factorization non-finite;
        # for such consistent systems the ridge returns the minimum-norm tangent
        # with an O(implicit_penalty * m) relative bias. The eps default is the
        # classic semidefinite-jitter scale eps * trace = m * eps * mean(eig),
        # invisible against well-conditioned tangents. implicit_penalty=0.0
        # disables (non-finite tangents on a singular dual).
        penalty = self.implicit_penalty
        if penalty is None:
            penalty = float(jnp.finfo(dual_dtype).eps)
        gram = gram + penalty * jnp.trace(gram) * jnp.eye(
            gram.shape[0], dtype=dual_dtype
        )
        factor = jsp_linalg.cho_factor(gram)
        dual_solution = jsp_linalg.cho_solve(factor, residual_p_dot.astype(dual_dtype))
        theta_dot = -gram_step_left @ dual_solution
        return unravel(theta_dot.astype(residual.dtype))

    def _implicit_x_tangent_from_p_cg(self, x, args, p, p_dot):
        theta, unravel = ravel_pytree(x)

        def residual_from_theta(theta_value):
            return self._residual_and_aux(unravel(theta_value), args, p)[0]

        residual, theta_jvp = jax.linearize(residual_from_theta, theta)
        theta_transpose = jax.linear_transpose(theta_jvp, theta)

        def JT(cotangent):
            return theta_transpose(cotangent)[0]

        def gram_matvec(cotangent):
            return theta_jvp(self._metric_inverse(JT(cotangent)))

        def residual_from_p(p_value):
            return self._residual_and_aux(x, args, p_value)[0]

        cg_tol = self._implicit_cg_tol(residual.dtype)
        cg_atol = jnp.asarray(self.implicit_atol, dtype=residual.dtype)

        cg_preconditioner = self.implicit_preconditioner

        def solve(matvec, rhs):
            solution, _ = jsp_sparse_linalg.cg(
                matvec,
                rhs,
                tol=cg_tol,
                atol=cg_atol,
                maxiter=self.implicit_maxiter,
                M=cg_preconditioner,
            )
            return solution

        residual_p_dot = jax.jvp(residual_from_p, (p,), (p_dot,))[1]
        dual_solution = jax.lax.custom_linear_solve(
            gram_matvec,
            residual_p_dot,
            solve,
            symmetric=True,
        )

        # The final metric inverse acts on tangent data, so VJP transposes
        # it. An iterative metric solve (e.g. metric_from_shifted_matvec)
        # is not transposable by JAX -- its CG captures tol*|b| inside the
        # linear solve's parameters -- but P is self-adjoint by contract,
        # so declare the application as its own transpose: every rule of
        # this custom_linear_solve routes through `solve` (the identity
        # matvec contributes nothing), and with symmetric=True the
        # cotangent pass just EVALUATES metric.solve. custom_linear_solve
        # is used rather than jax.custom_derivatives.linear_call because
        # linear_call has no batching rule, which would break jax.vmap
        # (and vmap-based second derivatives) over differentiated solves.
        theta_dot = -jax.lax.custom_linear_solve(
            lambda v: v,
            JT(dual_solution),
            lambda _, rhs: self._metric_inverse(rhs),
            symmetric=True,
        )
        return unravel(theta_dot)

    def _implicit_cg_tol(self, dtype):
        if self.implicit_tol is not None:
            return jnp.asarray(self.implicit_tol, dtype=dtype)
        default_tol = 1e-10 if jnp.finfo(dtype).bits > 32 else 1e-6
        return jnp.asarray(default_tol, dtype=dtype)

    def _metric_inverse(self, x):
        if self._has_metric_solve:
            return self.metric_solve(x)
        return self.metric_inv_sqrt(self.metric_inv_sqrt_transpose(x))

    def _action_or_default(self, action):
        if action is None:
            return LMSolveAction()
        return action

    def _apply_action(self, action, x, lm_state, args, user_state):
        action = self._action_or_default(action)
        # The step's diagnostics and the cached Jacobian describe the
        # pre-action (x, args), so both are stale iff the action actually
        # changed the values — a traced comparison, so a jit-style callback
        # that returns the field every step with unchanged values (the
        # jnp.where recipe pattern) changes nothing.
        problem_changed = jnp.asarray(False)
        if action.x is not None:
            problem_changed = problem_changed | _tree_changed(action.x, x)
            x = action.x
        if action.lm_state is not None:
            previous_hyper = lm_state.hyper
            lm_state = action.lm_state
            if self.cache_jacobian and lm_state.jacobian_valid is None:
                raise ValueError(
                    "cache_jacobian=True but the callback action returned an "
                    "lm_state without the Jacobian cache; use "
                    "dataclasses.replace(ctx.lm_state, ...) to preserve the "
                    "cache fields"
                )
            # Trace-time guard so the hyper contract fails identically with
            # and without jit (jit would reject the carry mismatch anyway).
            if previous_hyper is not None and (
                lm_state.hyper is None
                or jax.tree_util.tree_structure(previous_hyper)
                != jax.tree_util.tree_structure(lm_state.hyper)
                or [leaf.dtype for leaf in jax.tree_util.tree_leaves(previous_hyper)]
                != [leaf.dtype for leaf in jax.tree_util.tree_leaves(lm_state.hyper)]
            ):
                raise ValueError(
                    "the callback action changed the structure or dtypes of "
                    "lm_state.hyper; reset values with "
                    "dataclasses.replace(ctx.lm_state.hyper, ...) using arrays "
                    "of the same dtype — a knob constructed as None cannot be "
                    "enabled mid-solve"
                )
        if action.args is not None:
            problem_changed = problem_changed | _tree_changed(action.args, args)
            args = action.args
        if action.user_state is not None:
            user_state = action.user_state
        if self.cache_jacobian and (action.x is not None or action.args is not None):
            lm_state = dataclasses.replace(
                lm_state, jacobian_valid=lm_state.jacobian_valid & ~problem_changed
            )
        return action, x, lm_state, args, user_state, problem_changed

    def _check_residual_args(self, args, p):
        # Silently dropping args/p a residual never sees would, in particular,
        # make the implicit derivative with respect to p a silent zero.
        if args is not None and self.residual_arity < 2:
            raise ValueError(
                "args was passed but residual_fn takes only (x); "
                "use residual_fn(x, args)"
            )
        if p is not None and self.residual_arity < 3:
            raise ValueError(
                "p was passed but residual_fn takes no p argument; "
                "use residual_fn(x, args, p)"
            )

    def _converged(self, info, atol, gtol, xtol):
        atol_met = (atol > 0) & (jnp.sqrt(info.loss) < atol)
        gtol_met = (gtol > 0) & (info.grad_norm < gtol)
        xtol_met = (xtol > 0) & info.accepted & (info.step_norm < xtol)
        return atol_met | gtol_met | xtol_met

    def _solve_python(
        self,
        x,
        lm_state,
        args,
        p,
        user_state,
        history_len,
        max_steps,
        atol,
        gtol,
        xtol,
        callback,
    ):
        history = _init_history(self, x, args, p, history_len)
        info = self._initial_info(x, lm_state, args, p)
        lm_state = dataclasses.replace(
            lm_state,
            damping=jnp.asarray(lm_state.damping, dtype=info.loss.dtype),
            hyper=_cast_hyper(lm_state.hyper, info.loss.dtype),
        )
        initial_lm_state = lm_state
        status = LMStatus.RUNNING
        steps = 0
        if not bool(jnp.isfinite(info.loss)):
            status = LMStatus.NONFINITE
        elif bool(self._converged(info, atol, gtol, xtol)):
            status = LMStatus.CONVERGED

        for steps in range(1, max_steps + 1):
            if status != LMStatus.RUNNING:
                steps -= 1
                break
            x_old, lm_state_old = x, lm_state
            x, lm_state, info = self.update(x, lm_state, args, p)
            if not bool(jnp.isfinite(info.loss)):
                status = LMStatus.NONFINITE
                history = _record_history(history, steps, x, info, args)
                break
            action = None
            if callback is not None:
                ctx = LMSolveContext(
                    jnp.asarray(steps, dtype=jnp.int32),
                    x,
                    x_old,
                    lm_state,
                    lm_state_old,
                    initial_lm_state,
                    args,
                    p,
                    user_state,
                    info,
                )
                action = callback(ctx)
            action, x, lm_state, args, user_state, problem_changed = self._apply_action(
                action, x, lm_state, args, user_state
            )
            history = _record_history(history, steps, x, info, args)
            if action.stop is not None and bool(action.stop):
                status = (
                    LMStatus.CALLBACK_STOP
                    if action.status is None
                    else int(action.status)
                )
                break
            # info describes the pre-action (x, args); if the action changed
            # them, the tolerances must wait for a fresh update.
            if bool(self._converged(info, atol, gtol, xtol)) and not bool(
                problem_changed
            ):
                status = LMStatus.CONVERGED
                break
        else:
            steps = max_steps

        if status == LMStatus.RUNNING:
            status = LMStatus.MAX_STEPS
        final_aux = None
        if self.has_aux:
            final_aux = self._residual_and_aux(x, args, p)[1]
        x_history, aux_history, args_history = _finalize_history(
            history, steps, final_aux
        )
        return LMSolveResult(
            x,
            lm_state,
            info,
            jnp.asarray(steps, dtype=jnp.int32),
            jnp.asarray(status, dtype=jnp.int32),
            args,
            p,
            user_state,
            final_aux,
            x_history,
            aux_history,
            args_history,
        )


# save_steps bookkeeping shared by the jitted and Python solve loops: row `step` of
# x_history and args_history takes the kept post-action iterate and args; info.aux was
# evaluated at the pre-step x, so it lands one row earlier, and _finalize_history fills
# the last aux row from the final-solution evaluation. history_len is concrete (static
# under jit), so the buffers live entirely inside the loop implementations — no
# host-side allocation and no copy of a jit-input buffer before the in-place row
# updates. eval_shape gets the aux buffer shapes without paying for a residual
# evaluation.
def _history_buffer(tree, history_len):
    # Row 0 holds the initial value; tree.map over a None tree returns None.
    return jax.tree.map(
        lambda leaf: (
            jnp.zeros((history_len, *jnp.shape(leaf)), jnp.result_type(leaf))
            .at[0]
            .set(leaf)
        ),
        tree,
    )


def _init_history(solver, x0, args, p, history_len):
    if history_len is None:
        return None
    x_history = _history_buffer(x0, history_len)
    args_history = _history_buffer(args, history_len)
    aux_history = None
    if solver.has_aux:
        aux0 = jax.eval_shape(
            lambda x_, args_, p_: solver._residual_and_aux(x_, args_, p_)[1],
            x0,
            args,
            p,
        )
        aux_history = jax.tree.map(
            lambda leaf: jnp.zeros((history_len, *leaf.shape), leaf.dtype), aux0
        )
    return (x_history, aux_history, args_history)


def _record_history(history, step, x, info, args):
    if history is None:
        return None
    x_history, aux_history, args_history = history
    x_history = jax.tree.map(lambda buf, leaf: buf.at[step].set(leaf), x_history, x)
    args_history = jax.tree.map(
        lambda buf, leaf: buf.at[step].set(leaf), args_history, args
    )
    if aux_history is not None:
        aux_history = jax.tree.map(
            lambda buf, leaf: buf.at[step - 1].set(leaf), aux_history, info.aux
        )
    return (x_history, aux_history, args_history)


def _finalize_history(history, steps, final_aux):
    if history is None:
        return None, None, None
    x_history, aux_history, args_history = history
    if aux_history is not None:
        aux_history = jax.tree.map(
            lambda buf, leaf: buf.at[steps].set(leaf), aux_history, final_aux
        )
    return x_history, aux_history, args_history


def _solve_loop_impl(
    solver,
    x,
    lm_state,
    args,
    p,
    user_state,
    history_len,
    max_steps,
    atol,
    gtol,
    xtol,
    callback,
):
    history = _init_history(solver, x, args, p, history_len)
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)
    info = solver._initial_info(x, lm_state, args, p)
    # Recast damping, hyperparameters, and tolerances to the residual dtype so
    # the while_loop carry matches what update() returns.
    atol = jnp.asarray(atol, dtype=info.loss.dtype)
    gtol = jnp.asarray(gtol, dtype=info.loss.dtype)
    xtol = jnp.asarray(xtol, dtype=info.loss.dtype)
    lm_state = dataclasses.replace(
        lm_state,
        damping=jnp.asarray(lm_state.damping, dtype=info.loss.dtype),
        hyper=_cast_hyper(lm_state.hyper, info.loss.dtype),
    )
    initial_lm_state = lm_state
    step = jnp.asarray(0, dtype=jnp.int32)
    initial_nonfinite = ~jnp.isfinite(info.loss)
    initial_converged = solver._converged(info, atol, gtol, xtol)
    stop = initial_nonfinite | initial_converged
    status = jnp.where(
        initial_nonfinite,
        jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
        jnp.where(
            initial_converged,
            jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
            jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
        ),
    )

    def cond(carry):
        _, _, _, _, _, _, step, _, stop = carry
        return (~stop) & (step < max_steps)

    def body(carry):
        x, lm_state, args, user_state, history, _, step, _, _ = carry
        x_old, lm_state_old = x, lm_state
        x, lm_state, info = solver.update(x, lm_state, args, p)
        step = step + jnp.asarray(1, dtype=jnp.int32)
        current_nonfinite = ~jnp.isfinite(info.loss)

        action = None
        if callback is not None:
            ctx = LMSolveContext(
                step,
                x,
                x_old,
                lm_state,
                lm_state_old,
                initial_lm_state,
                args,
                p,
                user_state,
                info,
            )
            action = callback(ctx)
        action, x, lm_state, args, user_state, problem_changed = solver._apply_action(
            action, x, lm_state, args, user_state
        )
        history = _record_history(history, step, x, info, args)

        callback_stop = (
            jnp.asarray(False, dtype=jnp.bool_) if action.stop is None else action.stop
        )
        callback_status = (
            jnp.asarray(LMStatus.CALLBACK_STOP, dtype=jnp.int32)
            if action.status is None
            else jnp.asarray(action.status, dtype=jnp.int32)
        )
        # info describes the pre-action (x, args); if the action changed them,
        # the tolerances must wait for a fresh update.
        converged = solver._converged(info, atol, gtol, xtol) & ~problem_changed
        reached_max = step >= max_steps
        stop = current_nonfinite | callback_stop | converged | reached_max
        status = jnp.where(
            current_nonfinite,
            jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                callback_stop,
                callback_status,
                jnp.where(
                    converged,
                    jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
                    jnp.where(
                        reached_max,
                        jnp.asarray(LMStatus.MAX_STEPS, dtype=jnp.int32),
                        jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
                    ),
                ),
            ),
        )
        return x, lm_state, args, user_state, history, info, step, status, stop

    carry = jax.lax.while_loop(
        cond,
        body,
        (x, lm_state, args, user_state, history, info, step, status, stop),
    )
    x, lm_state, args, user_state, history, info, step, status, _ = carry
    final_aux = None
    if solver.has_aux:
        final_aux = solver._residual_and_aux(x, args, p)[1]
    x_history, aux_history, args_history = _finalize_history(history, step, final_aux)
    return LMSolveResult(
        x,
        lm_state,
        info,
        step,
        status,
        args,
        p,
        user_state,
        final_aux,
        x_history,
        aux_history,
        args_history,
    )


_solve_loop_jit = jax.jit(_solve_loop_impl, static_argnums=(0, 6, 11))
