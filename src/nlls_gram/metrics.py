from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg

from nlls_gram import quasiseparable


@dataclass(frozen=True)
class Metric:
    """Positive-definite parameter-space metric ``M`` given through callbacks.

    All callbacks act on the flattened parameter vector. With ``P = M^{-1}``
    and ``S`` satisfying ``S S' = M^{-1}``:

    - ``solve(x)``: ``M^{-1} x``
    - ``norm(x)``: ``sqrt(x' M x)``
    - ``inv_sqrt(x)``: ``S x``
    - ``inv_sqrt_transpose(x)``: ``S' x``

    Fields left as ``None`` default to the identity metric. Which fields are
    required depends on the solver configuration; see
    ``LevenbergMarquardt``.
    """

    solve: Callable | None = None
    norm: Callable | None = None
    inv_sqrt: Callable | None = None
    inv_sqrt_transpose: Callable | None = None


def metric_from_cholesky(L):
    """Build a dense ``Metric`` from a lower-triangular Cholesky factor.

    ``L`` is the factor of the metric matrix ``M = L @ L.T``, as returned by
    ``jnp.linalg.cholesky``.
    """

    def solve(x):
        y = jsp_linalg.solve_triangular(L, x, lower=True)
        return jsp_linalg.solve_triangular(L.T, y, lower=False)

    def norm(x):
        return jnp.linalg.norm(L.T @ x)

    def inv_sqrt(x):
        return jsp_linalg.solve_triangular(L.T, x, lower=False)

    def inv_sqrt_transpose(x):
        return jsp_linalg.solve_triangular(L, x, lower=True)

    return Metric(
        solve=solve,
        norm=norm,
        inv_sqrt=inv_sqrt,
        inv_sqrt_transpose=inv_sqrt_transpose,
    )


def metric_from_tridiagonal_precision(diag, off_diag, parallel=None):
    """Build an O(n) ``Metric`` from a symmetric tridiagonal precision matrix.

    ``diag`` (length n) and ``off_diag`` (length n-1) give the main and
    off-diagonal of ``T = M^{-1}``, which must be positive definite (not
    validated, since inputs may be traced; non-PD input silently produces a
    non-metric). Suited to
    Markov kernels, where the Gram inverse is exactly tridiagonal (e.g.
    Matern-1/2 / Ornstein-Uhlenbeck on sorted points): every callback costs
    O(n) and nothing is factored densely.

    ``parallel`` picks how the one-time bidiagonal Cholesky setup runs:
    ``None`` (default) uses an associative O(log n)-depth scan off-CPU in
    float64 — where a sequential scan pays a kernel launch per step — and
    the sequential scan otherwise. In float32 the default stays sequential
    even off-CPU: the parallel scan evaluates the pivot recurrence through
    projective 2x2 products whose cancellation can go non-finite on long,
    stiff grids (e.g. near-unit-correlation AR(1)), while the sequential
    recurrence is stable there. Pass ``parallel=True`` to override only when
    the setup is float64 or the grid is short and well-conditioned.
    """

    diag = jnp.asarray(diag)
    off_diag = jnp.asarray(off_diag)
    if diag.ndim != 1 or off_diag.shape != (diag.shape[0] - 1,):
        raise ValueError(
            "diag must be 1-D and off_diag must have shape (len(diag) - 1,)"
        )
    if parallel is None:
        parallel = jax.default_backend() != "cpu" and diag.dtype == jnp.float64
    zero = jnp.zeros((1,), dtype=diag.dtype)
    lower = jnp.concatenate([zero, off_diag])
    upper = jnp.concatenate([off_diag, zero])

    # Bidiagonal Cholesky T = C C': c_off[i] = off[i] / c_d[i],
    # c_d[i+1] = sqrt(diag[i+1] - c_off[i]^2), i.e. the pivot recurrence
    # delta_{i+1} = diag_{i+1} - off_i^2 / delta_i with delta = c_d^2.
    if off_diag.shape[0] == 0:
        c_d = jnp.sqrt(diag)
        c_off = off_diag
    elif parallel:
        # The pivot recurrence is a Mobius map delta -> (a delta + b)/delta,
        # so cumulative 2x2 matrix products compose it associatively; the
        # per-element normalization leaves the projective action unchanged
        # while keeping the entries O(1).
        mobius = jnp.stack(
            [
                jnp.stack([diag[1:], -(off_diag**2)], axis=-1),
                jnp.stack([jnp.ones_like(off_diag), jnp.zeros_like(off_diag)], axis=-1),
            ],
            axis=-2,
        )

        def combine(earlier, later):
            product = later @ earlier
            scale = jnp.max(jnp.abs(product), axis=(-2, -1), keepdims=True)
            return product / scale

        cumulative = jax.lax.associative_scan(combine, mobius)
        delta_rest = (cumulative[:, 0, 0] * diag[0] + cumulative[:, 0, 1]) / (
            cumulative[:, 1, 0] * diag[0] + cumulative[:, 1, 1]
        )
        delta = jnp.concatenate([diag[:1], delta_rest])
        c_d = jnp.sqrt(delta)
        c_off = off_diag / c_d[:-1]
    else:

        def chol_step(c_d_prev, inputs):
            off_i, diag_next = inputs
            c_off_i = off_i / c_d_prev
            c_d_i = jnp.sqrt(diag_next - c_off_i**2)
            return c_d_i, (c_off_i, c_d_i)

        c_d_0 = jnp.sqrt(diag[0])
        _, (c_off, c_d_rest) = jax.lax.scan(chol_step, c_d_0, (off_diag, diag[1:]))
        c_d = jnp.concatenate([c_d_0[None], c_d_rest])

    def expand(v, x):
        return v.reshape(v.shape + (1,) * (x.ndim - 1))

    def shift_up(x):
        return jnp.concatenate([x[1:], jnp.zeros_like(x[:1])])

    def shift_down(x):
        return jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]])

    def solve(x):
        # M^{-1} x = T x: a tridiagonal matvec along the leading axis.
        return (
            expand(diag, x) * x
            + expand(upper, x) * shift_up(x)
            + expand(lower, x) * shift_down(x)
        )

    def norm(x):
        # x' M x = x' T^{-1} x through one O(n) tridiagonal solve.
        y = jax.lax.linalg.tridiagonal_solve(lower, diag, upper, x[:, None])
        return jnp.sqrt(x @ y[:, 0])

    def inv_sqrt(x):
        # S = C (lower bidiagonal) with S S' = M^{-1}.
        return expand(c_d, x) * x + expand(
            jnp.concatenate([zero, c_off]), x
        ) * shift_down(x)

    def inv_sqrt_transpose(x):
        return expand(c_d, x) * x + expand(
            jnp.concatenate([c_off, zero]), x
        ) * shift_up(x)

    return Metric(
        solve=solve,
        norm=norm,
        inv_sqrt=inv_sqrt,
        inv_sqrt_transpose=inv_sqrt_transpose,
    )


def metric_from_quasiseparable(d, p, q, A, nugget=0.0, parallel=None):
    """Build an O(n) ``Metric`` from a symmetric quasiseparable SPD matrix.

    The metric is ``M = K + nugget * I`` with ``K`` given by rank-``m``
    quasiseparable generators ``d`` (n,), ``p`` (n, m), ``q`` (n, m), and
    ``A`` (n, m, m):

    ``K[i, j] = p[i] @ A[i-1] @ ... @ A[j+1] @ q[j]`` for ``i > j``, ``d``
    on the diagonal, and ``K[j, i] = K[i, j]``. ``A[k]`` is the transition
    INTO index ``k`` (``A[0]`` never enters the products; the state-space
    builders set it to the identity). ``nugget`` is an ABSOLUTE variance
    added to ``d`` before factorization and is part of the metric: all four
    callbacks act on ``M``, consistently. Every callback costs O(n m^2)
    through one quasiseparable Cholesky computed at construction;
    ``solve``, ``inv_sqrt``, and ``inv_sqrt_transpose`` accept vector and
    matrix inputs (``norm`` is vector-only, as everywhere in ``Metric``).

    ``M`` must be positive definite — not validated, since inputs may be
    traced; a non-PD or near-singular matrix silently produces NaN through
    the Cholesky square roots (same convention as
    ``metric_from_tridiagonal_precision``).

    ``parallel`` picks how the applies run, resolved ONCE at construction
    (a metric built on one backend keeps that scan choice for its
    lifetime): ``None`` (the default) uses associative O(log n)-depth scans
    off-CPU in float64 — where a sequential scan pays a kernel launch per
    step — and sequential scans otherwise. The
    parallel substitutions propagate rank-1-corrected transition matrices
    whose products are not contractive in general, so the float32 default
    stays sequential; pass ``parallel=True`` to override. The one-time
    Cholesky setup is always a sequential scan in this release.
    """

    d = jnp.asarray(d)
    p = jnp.asarray(p)
    q = jnp.asarray(q)
    A = jnp.asarray(A)
    if d.ndim != 1:
        raise ValueError("d must be 1-D")
    n = d.shape[0]
    if p.ndim != 2 or p.shape[0] != n:
        raise ValueError("p must have shape (len(d), m)")
    m = p.shape[1]
    if q.shape != (n, m) or A.shape != (n, m, m):
        raise ValueError("q must have shape (n, m) and A shape (n, m, m)")
    d = d + nugget
    if parallel is None:
        parallel = jax.default_backend() != "cpu" and d.dtype == jnp.float64
    c, w = quasiseparable.cholesky(d, p, q, A)

    def solve(x):
        # M^{-1} x = L^{-T} L^{-1} x: forward then backward substitution.
        y = quasiseparable.forward_substitution(c, p, w, A, x.reshape(n, -1), parallel)
        return quasiseparable.backward_substitution(c, p, w, A, y, parallel).reshape(
            x.shape
        )

    def norm(x):
        # x' M x through one quasiseparable matvec, no solve.
        y = quasiseparable.matvec(d, p, q, A, x.reshape(n, -1), parallel)
        return jnp.sqrt(x @ y.reshape(x.shape))

    def inv_sqrt(x):
        # S = L^{-T} with S S' = M^{-1}.
        return quasiseparable.backward_substitution(
            c, p, w, A, x.reshape(n, -1), parallel
        ).reshape(x.shape)

    def inv_sqrt_transpose(x):
        return quasiseparable.forward_substitution(
            c, p, w, A, x.reshape(n, -1), parallel
        ).reshape(x.shape)

    return Metric(
        solve=solve,
        norm=norm,
        inv_sqrt=inv_sqrt,
        inv_sqrt_transpose=inv_sqrt_transpose,
    )


def metric_from_state_space(points, h, Pinf, transition, nugget=0.0, parallel=None):
    """Build an O(n) ``Metric`` for a stationary state-space kernel Gram.

    A stationary Gaussian process has an exact O(n) Gram factorization
    precisely when it admits a finite-dimensional state-space (linear SDE)
    representation: an m-dimensional latent Gauss-Markov state observed
    through a row vector. Given the observation row ``h`` (m,), stationary
    state covariance ``Pinf`` (m, m), and ``transition(dt)`` mapping gaps
    (n,) to stacked transition matrices (n, m, m), the metric is
    ``M = K + nugget * I`` with

    ``K[i, j] = h @ Pinf @ A[i] @ A[i-1] @ ... @ A[j+1] @ h`` for ``i > j``,
    ``K[i, i] = h @ Pinf @ h``, symmetric, where
    ``A[k] = transition(points[k] - points[k-1])``. ``transition`` must
    return the TRANSPOSE of the textbook matrix exponential ``expm(F dt)``
    of the SDE drift (the tinygp orientation), and ``transition(0)`` must be
    the identity. The main application is the half-integer Matern family —
    ``matern_state_space(sigma, ell, nu)`` supplies the exact
    ``(h, Pinf, transition)`` mapping — but sums of exponentials and other
    CARMA/celerite-style kernels reduce to the same form.

    ``points`` must be 1-D and sorted STRICTLY increasing — not validated,
    since it may be traced; unsorted or repeated points silently produce a
    wrong or NaN metric. ``h``, ``Pinf``, and ``points`` may be traced (e.g.
    hyperparameter sweeps under ``jax.grad``), in which case the one-time
    sequential Cholesky setup lands on the hot path; see the tuning guide.
    The ABSOLUTE ``nugget`` folds into the diagonal before factorization and
    is part of the metric; ``parallel`` is forwarded to
    ``metric_from_quasiseparable``.
    """

    d, p, q, A = quasiseparable.state_space_generators(points, h, Pinf, transition)
    return metric_from_quasiseparable(d, p, q, A, nugget=nugget, parallel=parallel)


def metric_from_diagonal(weights):
    """Build a ``Metric`` for the diagonal metric ``M = diag(weights)``.

    ``weights`` (length n) must be positive -- not validated, since inputs
    may be traced. Every callback is elementwise.
    """

    weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be 1-D")
    sqrt_weights = jnp.sqrt(weights)

    def expand(v, x):
        return v.reshape(v.shape + (1,) * (x.ndim - 1))

    def solve(x):
        return x / expand(weights, x)

    def norm(x):
        return jnp.sqrt(x @ (weights * x))

    def inv_sqrt(x):
        return x / expand(sqrt_weights, x)

    return Metric(
        solve=solve,
        norm=norm,
        inv_sqrt=inv_sqrt,
        inv_sqrt_transpose=inv_sqrt,
    )


def metric_from_shifted_matvec(
    matvec, shift, *, tol=None, atol=0.0, maxiter=None, preconditioner=None
):
    """Build a matrix-free ``Metric`` for ``M = A + shift * I`` from a matvec.

    ``matvec(x)`` applies a symmetric positive-semidefinite ``A`` and must
    accept both ``(n,)`` vectors and ``(n, k)`` matrices on the leading axis
    (the same shape contract ``Metric.solve`` itself carries). ``shift``
    must be POSITIVE -- it is the spectral floor that makes an iterative
    metric solve viable at all: ``cond(A + shift I) <= (lambda_max + shift)
    / shift`` regardless of how singular ``A`` is, and ``solve`` costs
    ``~sqrt(lambda_max / shift) * log(1/tol)`` matvecs. A concrete
    ``shift <= 0`` raises; a traced ``shift`` is not validated and must be
    positive. Only ``solve`` and ``norm`` are provided (there is no
    matrix-free square root), so the solver accepts this metric for the
    ``cholesky`` and ``cg`` paths and rejects ``qr`` at construction.

    ``solve`` runs ``jax.scipy.sparse.linalg.cg`` on the shifted matvec, so
    it meets the library's exactness contract only in the tight-``tol``
    limit. Unlike ``dual_preconditioner`` error -- which never moves the
    converged root -- this solve's residual error perturbs the selected
    step, the converged solution, and the implicit derivatives at order
    ``tol``; the implicit derivative in particular is computed outside the
    accept/reject loop with no safeguard. ``tol`` is therefore an accuracy
    knob for the ANSWER, not a solver schedule. The default (``None`` ->
    the square root of the input dtype's machine epsilon, ~1.5e-8 in
    float64) keeps that perturbation at the level of typical outer
    tolerances; CG's attainable residual stagnates near
    ``machine_eps * (lambda_max + shift) / shift``, which bounds how far
    ``tol`` can usefully be tightened (~1e-10 is reachable in float64 only
    when that condition number is ~1e4 or better). ``maxiter=None`` runs to
    tolerance -- a truncated CG is not a linear function of its right-hand
    side, which breaks the linear-operator assumptions of the ``cg``
    linear_solver, so do not cap it as a cost control. ``preconditioner``
    is forwarded to the inner CG (its ``M`` argument). A matrix right-hand
    side is solved as one stacked CG call (same spectrum, batched matvecs).

    Because the inner CG is built on ``lax.custom_linear_solve`` with a
    symmetric operator, ``solve`` composes with ``jax.jvp``/``jax.vjp``/
    ``jax.grad`` — including both implicit-AD rules (dense and
    ``implicit_solver="cg"``) and differentiating ``update`` through the
    ``cg`` path — and with values the matvec closes over; ``matvec`` itself
    only needs to be linear. (Closure-value differentiation holds for
    direct ``Metric`` use and through ``update``; the LM ``solve()``
    entry point supports differentiation with respect to ``p`` only — its
    implicit rule is a ``jax.custom_jvp``, which cannot close over active
    tracers.) Raw ``jax.linear_transpose`` of ``solve`` is not supported by
    JAX's CG; the solver never applies it to ``metric.solve`` (the cg
    implicit rule declares the application self-adjoint instead of
    transposing it).
    """

    if not isinstance(shift, jax.core.Tracer) and float(shift) <= 0.0:
        raise ValueError("shift must be positive (it is the spectral floor)")
    if tol is not None and tol < 0:
        raise ValueError("tol must be nonnegative")
    if atol < 0:
        raise ValueError("atol must be nonnegative")
    if maxiter is not None and maxiter <= 0:
        raise ValueError("maxiter must be positive or None")

    def shifted_matvec(x):
        return matvec(x) + shift * x

    def solve(x):
        if tol is None:
            solve_tol = float(jnp.finfo(jnp.result_type(x)).eps) ** 0.5
        else:
            solve_tol = tol
        solution, _ = jsp_sparse_linalg.cg(
            shifted_matvec,
            x,
            tol=solve_tol,
            atol=atol,
            maxiter=maxiter,
            M=preconditioner,
        )
        return solution

    def norm(x):
        return jnp.sqrt(x @ shifted_matvec(x))

    return Metric(solve=solve, norm=norm)


def blockdiag_metric(blocks):
    """Compose per-block ``Metric``s into one block-diagonal ``Metric``.

    ``blocks`` is a sequence of ``(metric, size)`` pairs in the order the
    flattened parameter vector is laid out. ``solve``, ``inv_sqrt``, and
    ``inv_sqrt_transpose`` slice on the leading axis (vector and matrix
    inputs both work); ``norm`` combines the block norms in quadrature.

    A fully-default ``Metric()`` block means the identity metric on that
    block, and its callbacks are filled in accordingly. A block that defines
    some callbacks but leaves others ``None`` is treated as missing those
    callbacks: the composite field is ``None``, so the solver's
    construction-time validation applies exactly as it would to that block
    alone. Filling identity there instead would silently break the
    ``S S' = M^{-1}`` consistency between callbacks.
    """

    if not blocks:
        raise ValueError("blocks must contain at least one (metric, size) pair")
    metrics = [metric for metric, _ in blocks]
    offsets = [0]
    for _, size in blocks:
        offsets.append(offsets[-1] + size)

    identity_block = [
        metric.solve is None
        and metric.norm is None
        and metric.inv_sqrt is None
        and metric.inv_sqrt_transpose is None
        for metric in metrics
    ]

    def split(x):
        return [
            x[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        ]

    def identity(x):
        return x

    def blockwise(callbacks):
        if any(
            callback is None and not is_identity
            for callback, is_identity in zip(callbacks, identity_block, strict=True)
        ):
            return None
        filled = [identity if callback is None else callback for callback in callbacks]

        def apply(x):
            return jnp.concatenate(
                [
                    callback(part)
                    for callback, part in zip(filled, split(x), strict=True)
                ]
            )

        return apply

    def make_norm(callbacks):
        if any(
            callback is None and not is_identity
            for callback, is_identity in zip(callbacks, identity_block, strict=True)
        ):
            return None
        filled = [
            jnp.linalg.norm if callback is None else callback for callback in callbacks
        ]

        def norm(x):
            parts = split(x)
            return jnp.sqrt(
                sum(
                    callback(part) ** 2
                    for callback, part in zip(filled, parts, strict=True)
                )
            )

        return norm

    return Metric(
        solve=blockwise([metric.solve for metric in metrics]),
        norm=make_norm([metric.norm for metric in metrics]),
        inv_sqrt=blockwise([metric.inv_sqrt for metric in metrics]),
        inv_sqrt_transpose=blockwise([metric.inv_sqrt_transpose for metric in metrics]),
    )


def repeated_blockdiag_metric(
    block_metric: Metric,
    block_size: int,
    repeats: int,
    *,
    additional: tuple[Metric, int] | None = None,
) -> Metric:
    """Batch ``repeats`` identical block-diagonal blocks into one ``Metric``.

    Equivalent to ``blockdiag_metric([(block_metric, block_size)] * repeats)``
    (optionally with a single trailing block), but each callback fires **once**
    per apply instead of once per copy. The total leading size is
    ``repeats * block_size + additional_size`` and is *derived*, not passed, so
    a layout mismatch is caught rather than silently reshaped. The head of the
    parameter vector is ``repeats`` copies of ``block_metric``; the optional
    ``additional=(metric, size)`` is a single trailing block for the finite set
    of variables that sit outside the repeated structure (initial conditions,
    scalar parameters).

    ``solve``, ``inv_sqrt``, and ``inv_sqrt_transpose`` reshape and moveaxis the
    repeated head from ``(repeats * block_size, k)`` to
    ``(block_size, repeats * k)`` and invoke the base callback a single time — a
    dense Cholesky block does two triangular solves total, not two per copy —
    then reshape back; ``norm`` combines the per-copy block norms in quadrature.
    Both vector and ``(n, k)`` matrix inputs work (``norm`` is vector-only, as
    everywhere in ``Metric``).

    A fully-default ``Metric()`` block is the identity on its span. A block that
    defines some callbacks but leaves others ``None`` propagates the missing
    callbacks as ``None`` on the composite (the same contract as
    ``blockdiag_metric``, so the solver's construction-time validation applies).
    This returns a plain ``Metric`` honoring the ``(n,)``/``(n, k)`` contract,
    so it also composes inside ``blockdiag_metric`` for heterogeneous layouts.

    ``block_size``, ``repeats``, and any ``additional`` size must be positive
    integers (``ValueError`` otherwise). Unlike ``blockdiag_metric``'s
    permissive slicing, the reshape here assumes an exact
    ``(repeats, block_size)`` multiple, so a callback input whose leading size
    is not ``total_size`` — or whose ndim is not in ``{1, 2}`` — raises
    ``ValueError`` rather than silently consuming the wrong rows.
    """

    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if additional is None:
        additional_metric = None
        additional_size = 0
    else:
        additional_metric, additional_size = additional
        if (
            not isinstance(additional_size, int)
            or isinstance(additional_size, bool)
            or additional_size < 1
        ):
            raise ValueError("additional size must be a positive integer")

    repeated_size = repeats * block_size
    total_size = repeated_size + additional_size

    def is_identity(metric):
        return (
            metric.solve is None
            and metric.norm is None
            and metric.inv_sqrt is None
            and metric.inv_sqrt_transpose is None
        )

    block_is_identity = is_identity(block_metric)
    additional_is_identity = additional_metric is not None and is_identity(
        additional_metric
    )

    def check_leading_size(x):
        if x.ndim not in (1, 2):
            raise ValueError("metric callbacks require a vector or matrix")
        if x.shape[0] != total_size:
            raise ValueError(
                f"metric leading size must be {total_size}, got {x.shape[0]}"
            )

    def identity(x):
        return x

    def make_apply(block_callback, additional_callback):
        if block_callback is None and not block_is_identity:
            return None
        if (
            additional_metric is not None
            and additional_callback is None
            and not additional_is_identity
        ):
            return None
        apply_block = identity if block_callback is None else block_callback
        apply_additional = (
            identity if additional_callback is None else additional_callback
        )

        def apply(x):
            check_leading_size(x)
            trailing_shape = x.shape[1:]
            combined = jnp.moveaxis(
                x[:repeated_size].reshape((repeats, block_size) + trailing_shape),
                0,
                1,
            ).reshape(block_size, -1)
            transformed = apply_block(combined)
            repeated = jnp.moveaxis(
                transformed.reshape((block_size, repeats) + trailing_shape),
                0,
                1,
            ).reshape((repeated_size,) + trailing_shape)
            if additional_metric is None:
                return repeated
            return jnp.concatenate(
                [repeated, apply_additional(x[repeated_size:])], axis=0
            )

        return apply

    def make_norm(block_norm, additional_norm):
        if block_norm is None and not block_is_identity:
            return None
        if (
            additional_metric is not None
            and additional_norm is None
            and not additional_is_identity
        ):
            return None
        norm_block = jnp.linalg.norm if block_norm is None else block_norm
        norm_additional = (
            jnp.linalg.norm if additional_norm is None else additional_norm
        )

        def norm(x):
            check_leading_size(x)
            if x.ndim != 1:
                raise ValueError("metric norm requires a vector")
            blocks = x[:repeated_size].reshape(repeats, block_size)
            squared_norm = jnp.sum(jax.vmap(norm_block)(blocks) ** 2)
            if additional_metric is not None:
                squared_norm = squared_norm + norm_additional(x[repeated_size:]) ** 2
            return jnp.sqrt(squared_norm)

        return norm

    additional_solve = None if additional_metric is None else additional_metric.solve
    additional_norm = None if additional_metric is None else additional_metric.norm
    additional_inv_sqrt = (
        None if additional_metric is None else additional_metric.inv_sqrt
    )
    additional_inv_sqrt_transpose = (
        None if additional_metric is None else additional_metric.inv_sqrt_transpose
    )
    return Metric(
        solve=make_apply(block_metric.solve, additional_solve),
        norm=make_norm(block_metric.norm, additional_norm),
        inv_sqrt=make_apply(block_metric.inv_sqrt, additional_inv_sqrt),
        inv_sqrt_transpose=make_apply(
            block_metric.inv_sqrt_transpose,
            additional_inv_sqrt_transpose,
        ),
    )


def metric_with_compute_dtype(metric: Metric, dtype) -> Metric:
    """Wrap a ``Metric`` so every callback computes in ``dtype``.

    Each callback upcasts its input to ``dtype``, applies the wrapped metric,
    and restores the caller's dtype on output. This keeps an ill-conditioned
    factorization or solve in wide precision (float64) while the solver's
    residual/parameter dtype and loop-carried pytrees stay at the problem dtype
    — so JVP tangents and carried pytrees keep a stable dtype contract. The
    output round-trips to ``x.dtype``, a no-op when the solver has already
    promoted its duals to ``dtype``.

    ``None`` callbacks are preserved (a wrapped partial ``Metric`` stays
    partial), so the solver's construction-time validation is unchanged.
    """

    dtype = jnp.dtype(dtype)

    # The callback computes wide, then restores the solver's residual/parameter
    # dtype so JVP tangents and loop-carried pytrees keep a stable contract.
    def wrap(callback):
        if callback is None:
            return None

        def apply(x):
            return callback(x.astype(dtype)).astype(x.dtype)

        return apply

    return Metric(
        solve=wrap(metric.solve),
        norm=wrap(metric.norm),
        inv_sqrt=wrap(metric.inv_sqrt),
        inv_sqrt_transpose=wrap(metric.inv_sqrt_transpose),
    )
