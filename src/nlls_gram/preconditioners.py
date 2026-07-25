"""Preconditioners for the Krylov configs of both solvers.

``apply(v, damping, ctx)`` returns an SPD approximation of the damped
operator's inverse. Which space ``v`` lives in is named by the config that
consumes it: :class:`~nlls_gram.CG` is parameter space,
:class:`~nlls_gram.GramCG` residual space.

Unlike a :class:`~nlls_gram.Metric` -- which defines the converged root and
must stay exact -- a preconditioner only changes the CG iteration path, so
approximations and staleness are safe.
"""

from dataclasses import InitVar, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from nlls_gram.utilities import register_pytree_dataclass


class Preconditioner:
    """SPD preconditioner for the solvers' CG paths.

    ``apply(v, damping, ctx)`` returns an SPD approximation of the damped
    operator's inverse -- ``(J~'J~ + ridge E + damping I)^{-1}`` in parameter
    space under :class:`~nlls_gram.CG`, ``(J~J~' + damping I)^{-1}`` in
    residual space under :class:`~nlls_gram.GramCG`. In the forward role it
    sits in CG's ``M`` slot with the live damping; in the ``ad_solver`` role
    the implicit system is undamped and ``damping`` is zero. ``ctx`` is the
    same :class:`~nlls_gram.SolverContext` the metric factor ops receive.

    Preconditioner instances are JAX PYTREES: array fields are traced
    leaves, the type plus its static fields are structure. Every concrete
    class must be registered with
    :func:`~nlls_gram.register_pytree_dataclass` -- the solvers reject
    unregistered instances. The instance rides inside the solver state
    (``lm_state.preconditioner``), so a ``solve`` callback refreshes it by
    calling the CONSTRUCTOR again with fresh arrays -- same type, same leaf
    shapes and dtypes, pure traced ops, no recompilation. Never
    ``dataclasses.replace`` a preconditioner: ``replace`` re-runs
    ``__init__``, re-paying any eigendecomposition or sketch, and
    construction-time-only inputs are not stored to re-supply. A stale
    instance only changes the CG iteration path, never the converged step,
    so refreshing rarely (or never) is always safe.

    Implement a custom one as a small registered frozen dataclass::

        @dataclass(frozen=True, eq=False)
        class JacobiPreconditioner(Preconditioner):
            diagonal: jax.Array

            def apply(self, v, damping, ctx):
                return v / (self.diagonal + damping)

        register_pytree_dataclass(JacobiPreconditioner, data_fields=("diagonal",))

    Subclasses whose ``apply`` divides by the live damping must set
    ``requires_positive_damping = True``; the constructor rejects them for
    the AD role, where damping is zero.
    """

    requires_positive_damping = False

    def apply(self, v, damping, ctx):
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityPreconditioner(Preconditioner):
    """The identity map as an explicit "no preconditioner" choice for ``CG``.

    Nobody should run Krylov methods without thinking about preconditioning,
    so opting out is an explicit, greppable decision rather than a silent
    default. Stateless: every instance compares equal, so equal ``CG``
    configs share one compiled solve loop.
    """

    def apply(self, v, damping, ctx):
        return v


register_pytree_dataclass(IdentityPreconditioner, data_fields=())


@dataclass(frozen=True, eq=False)
class BlockEigenPreconditioner(Preconditioner):
    """Block-diagonal eigenbasis preconditioner over grouped whitened
    coordinates.

    The workhorse for structured whitened normal operators
    ``J~'J~ + ridge E + damping I`` built from repeated interacting blocks
    (multiple "agents" coupled through shared equations): approximate the
    operator by a block-diagonal matrix over a chosen grouping of the
    whitened coordinates, eigendecompose each block, and apply the exact
    inverse of the shifted approximation

        v  ->  V ((V' v) / (Lambda + ridge_weight * ridge + damping)) V'

    per block -- analytic in both the live ``damping`` (traced; it changes per
    LM step) and the live ``ridge`` (read from ``ctx.lm_state.ridge``, so
    ridge continuation composes with no refresh).

    ``families`` is a sequence of ``(blocks, ridge_weight)`` pairs:
    ``blocks`` has shape ``(groups, size, size)`` -- the stacked diagonal
    blocks of ``J~'J~`` restricted to that family's coordinate groups, in
    permuted order. Families in the metric block set ``ridge_weight = 1``
    (their diagonal carries the ``ridge`` spectral floor); free-block
    families set ``0`` (damping-only, so the zero-damping AD role applies
    their plain inverse -- positive definite whenever the free block is
    identified). Blocks are symmetrized and eigendecomposed HERE, once;
    positive semidefiniteness is assumed, not validated (entries may be
    traced). ``permutation`` is the 1-D integer array reordering the
    flattened whitened parameter vector into family-major order
    (``v_permuted = v[permutation]``); families are consumed in sequence and
    must cover it exactly. ``jnp.arange(n)`` serves when the natural layout
    already is family-major.

    The constructor is fully traceable, so a ``solve`` callback refreshes
    the preconditioner from the live iterate by constructing a new instance
    -- gate the rebuild with ``jax.lax.cond`` (e.g. on a ridge-continuation
    level advance) so the eigendecomposition is paid only when it fires; a
    ``jnp.where`` merge would pay it every step.
    """

    families: InitVar[Any]
    permutation: jax.Array
    eigenvectors: tuple = field(init=False)
    eigenvalues: tuple = field(init=False)
    ridge_weights: tuple = field(init=False)
    inverse_permutation: jax.Array = field(init=False)

    def __post_init__(self, families):
        permutation = jnp.asarray(self.permutation)
        if permutation.ndim != 1 or not jnp.issubdtype(permutation.dtype, jnp.integer):
            raise ValueError("permutation must be a 1-D integer array")
        eigenvectors, eigenvalues, ridge_weights = [], [], []
        covered = 0
        for blocks, ridge_weight in families:
            blocks = jnp.asarray(blocks)
            if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
                raise ValueError(
                    "each family's blocks must have shape (groups, size, size); "
                    f"got {blocks.shape}"
                )
            symmetrized = 0.5 * (blocks + jnp.swapaxes(blocks, 1, 2))
            values, vectors = jnp.linalg.eigh(symmetrized)
            # eigh of a numerically PSD block can return tiny negative
            # eigenvalues; clamped at zero the apply shift stays positive for
            # any positive ridge/damping (and the zero-damping AD role stays
            # SPD whenever the family itself is).
            eigenvalues.append(jnp.maximum(values, 0.0))
            eigenvectors.append(vectors)
            ridge_weights.append(jnp.asarray(ridge_weight, dtype=blocks.dtype))
            covered += blocks.shape[0] * blocks.shape[1]
        if covered != permutation.shape[0]:
            raise ValueError(
                f"families cover {covered} coordinates but the permutation "
                f"has {permutation.shape[0]}"
            )
        object.__setattr__(self, "permutation", permutation)
        object.__setattr__(self, "eigenvectors", tuple(eigenvectors))
        object.__setattr__(self, "eigenvalues", tuple(eigenvalues))
        object.__setattr__(self, "ridge_weights", tuple(ridge_weights))
        object.__setattr__(self, "inverse_permutation", jnp.argsort(permutation))

    def apply(self, v, damping, ctx):
        # LevenbergMarquardt carries no ridge, so its metric-block families
        # shift by the damping alone.
        carried = ctx.lm_state.ridge
        ridge = (
            jnp.zeros((), v.dtype)
            if carried is None
            else jnp.asarray(carried, dtype=v.dtype)
        )
        permuted = v[self.permutation]
        pieces = []
        offset = 0
        for V, values, ridge_weight in zip(
            self.eigenvectors, self.eigenvalues, self.ridge_weights, strict=True
        ):
            groups, size = V.shape[0], V.shape[1]
            segment = permuted[offset : offset + groups * size]
            offset += groups * size
            shift = ridge_weight * ridge + damping
            coefficients = jnp.einsum("gab,ga->gb", V, segment.reshape(groups, size))
            pieces.append(
                jnp.einsum("gab,gb->ga", V, coefficients / (values + shift)).reshape(-1)
            )
        return jnp.concatenate(pieces)[self.inverse_permutation].astype(v.dtype)


register_pytree_dataclass(
    BlockEigenPreconditioner,
    data_fields=(
        "permutation",
        "eigenvectors",
        "eigenvalues",
        "ridge_weights",
        "inverse_permutation",
    ),
)


@dataclass(frozen=True, eq=False)
class ShermanMorrisonPreconditioner(Preconditioner):
    """Dual preconditioner for ``B = A + weight * u u'`` from a solve with ``A``.

    Applies ``B^{-1} v = y - A^{-1}u (u' y) / (1/weight + u' A^{-1} u)`` with
    ``y = A^{-1} v`` by the Sherman-Morrison identity; ``A^{-1}u`` and the
    scalar denominator are precomputed at construction. This is the natural
    shape for kernel-collocation dual operators, where a metric weight ``m``
    on a scalar parameter injects an exactly known rank-1 spike
    ``(c^2/m) u u'`` into ``J M^{-1} J'``. The live ``damping`` is ignored --
    spectral closeness to the damped operator is all a preconditioner needs --
    which also makes it valid in the zero-damping ``ad_solver`` role.

    ``solve`` applies ``A^{-1}`` and is called on every ``apply``, so it is a
    STATIC field: a fixed hashable callable whose identity enters the
    instance's pytree structure, with anything it closes over entering the
    compiled program as constants. This class is a setup-scope object for a
    fixed dual operator, not a callback-refresh target.
    """

    solve: Any
    u: jax.Array
    weight: Any
    solve_u: jax.Array = field(init=False)
    denominator: jax.Array = field(init=False)

    def __post_init__(self):
        u = jnp.asarray(self.u)
        weight = jnp.asarray(self.weight, dtype=jnp.result_type(u, 1.0))
        solve_u = self.solve(u)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "solve_u", solve_u)
        object.__setattr__(self, "denominator", 1.0 / weight + u @ solve_u)

    def apply(self, v, damping, ctx):
        y = self.solve(v)
        return y - self.solve_u * ((self.u @ y) / self.denominator)


register_pytree_dataclass(
    ShermanMorrisonPreconditioner,
    data_fields=("u", "weight", "solve_u", "denominator"),
    meta_fields=("solve",),
)


@dataclass(frozen=True, eq=False)
class WoodburyPreconditioner(Preconditioner):
    """Dual preconditioner for ``B = A + U diag(weights) U'``.

    The rank-k generalization of :class:`ShermanMorrisonPreconditioner`:
    applies ``B^{-1} v = y - A^{-1}U C^{-1}(U' y)`` with ``y = A^{-1} v`` and
    capacitance ``C = diag(1/weights) + U' A^{-1} U``; ``A^{-1}U`` (one matrix
    solve) and the Cholesky factor of the k x k capacitance are precomputed.
    ``weights`` must be positive -- not validated, since inputs may be traced.
    Like Sherman-Morrison it ignores ``damping`` and so serves the AD role
    too, and its ``solve`` is the same STATIC always-called field.
    """

    solve: Any
    U: jax.Array
    weights: jax.Array
    solve_U: jax.Array = field(init=False)
    capacitance_factor: jax.Array = field(init=False)

    def __post_init__(self):
        U, weights = jnp.asarray(self.U), jnp.asarray(self.weights)
        if U.ndim != 2 or weights.shape != (U.shape[1],):
            raise ValueError("U must have shape (n, k) and weights shape (k,)")
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "weights", weights)
        solve_U = self.solve(U)
        object.__setattr__(self, "solve_U", solve_U)
        capacitance = jnp.diag(1.0 / weights) + U.T @ solve_U
        object.__setattr__(
            self, "capacitance_factor", jsp_linalg.cho_factor(capacitance)[0]
        )

    def apply(self, v, damping, ctx):
        y = self.solve(v)
        correction = jsp_linalg.cho_solve(
            (self.capacitance_factor, False), self.U.T @ y
        )
        return y - self.solve_U @ correction


register_pytree_dataclass(
    WoodburyPreconditioner,
    data_fields=("U", "weights", "solve_U", "capacitance_factor"),
    meta_fields=("solve",),
)


@dataclass(frozen=True, eq=False)
class PaddedPreconditioner(Preconditioner):
    """Extend a dual preconditioner to a residual padded with exact zeros.

    The fixed-residual-shape pattern appends identically-zero entries to an
    ``n_real``-entry residual so compiled shapes stay stable across problem
    instances. Padded rows have zero Jacobian rows, so the dual operator is
    exactly block diagonal::

        [ J P J' + damping I      0          ]
        [ 0                       damping I  ]

    and this applies ``base`` on the first ``n_real`` coordinates and the exact
    ``1/damping`` inverse on the padded block. That second block must NOT be
    zeroed -- that would make the preconditioner singular rather than SPD, even
    though zeros can appear to work when the padded coordinates are never
    excited. Because the padded block divides by the live damping, this serves
    only the damped forward solve; the undamped dual is singular there, which
    ``ad_solver=SVD()`` handles exactly.
    """

    base: Preconditioner
    n_real: int

    requires_positive_damping = True

    def apply(self, v, damping, ctx):
        # Static shapes, so this raises at trace time; without it a
        # shape-generic base would silently accept a too-short vector.
        if v.ndim != 1 or v.shape[0] < self.n_real:
            raise ValueError(
                f"padded residual vector must be 1-D with at least "
                f"n_real={self.n_real} entries; got shape {v.shape}"
            )
        return jnp.concatenate(
            (
                self.base.apply(v[: self.n_real], damping, ctx),
                v[self.n_real :] / damping,
            )
        )


register_pytree_dataclass(
    PaddedPreconditioner, data_fields=("base",), meta_fields=("n_real",)
)


@dataclass(frozen=True, eq=False)
class NystromPreconditioner(Preconditioner):
    """Randomized Nystrom preconditioner (Frangella-Tropp-Udell) for a PSD
    operator given only through ``matvec``.

    Sketches ``A`` with a rank-``rank`` Nystrom approximation
    ``A_hat = U diag(lam) U'`` -- a thin-QR'd Gaussian test matrix, one block
    application ``Y = A Omega``, and the shifted Cholesky/SVD recovery of
    arXiv:2110.02820 Algorithm 2.1 -- then applies their eq. 5.3::

        v  ->  U ((U'v) / (lam + damping)) + (v - U U'v) / (rho + damping)

    where ``rho`` is the smallest retained eigenvalue: directions the sketch
    resolved are inverted against the live shift, and the unresolved
    complement is treated as sitting at ``rho`` rather than at zero. That
    balance is what carries the FTU condition-number guarantee for
    fast-decaying spectra.

    The target use is neural-network least squares under the identity metric,
    where the dual operator is the ``m x m`` empirical NTK Gram ``J J'`` --
    fast spectral decay plus the LM damping shift is exactly the FTU regime.
    ``matvec`` must apply a symmetric PSD operator to ``(n, k)`` matrices; an
    indefinite one silently produces NaN through the Cholesky square root. It
    is consumed at construction -- the build costs ``rank`` operator
    applications plus an ``O(n rank^2)`` QR/SVD, and only the sketch is
    stored, so for a nonlinear problem the instance approximates the dual at
    the linearization point it was built from (staleness is safe; a callback
    refreshes by constructing a new instance from a fresh ``matvec``). Each
    apply is two ``(n, rank)`` matvecs.

    ``key`` is an explicit PRNG key; the same key reproduces the same
    preconditioner. ``dtype=None`` uses the JAX default float -- pass the
    operator dtype explicitly for a float32 problem under enabled x64.
    """

    matvec: InitVar[Any]
    n: int
    rank: int
    key: InitVar[Any]
    dtype: InitVar[Any] = None
    basis: jax.Array = field(init=False)
    eigenvalues: jax.Array = field(init=False)

    def __post_init__(self, matvec, key, dtype):
        if not 0 < self.rank <= self.n:
            raise ValueError("rank must be a positive int <= n")
        dtype = jnp.result_type(float) if dtype is None else dtype
        shape = (self.n, self.rank)
        Omega = jnp.linalg.qr(jax.random.normal(key, shape, dtype))[0]
        Y = matvec(Omega)
        # The floor keeps the shift usable for a (near-)zero operator, where
        # eps * ||Y||_F alone would leave the core singular; tiny/eps stays
        # clear of the subnormal range through the downstream products.
        finfo = jnp.finfo(dtype)
        nu = jnp.maximum(finfo.eps * jnp.linalg.norm(Y), finfo.tiny / finfo.eps)
        Y_nu = Y + nu * Omega
        core = Omega.T @ Y_nu
        L = jnp.linalg.cholesky(0.5 * (core + core.T))
        B = jsp_linalg.solve_triangular(L, Y_nu.T, lower=True).T
        U, sigma, _ = jnp.linalg.svd(B, full_matrices=False)
        object.__setattr__(self, "basis", U)
        object.__setattr__(self, "eigenvalues", jnp.maximum(sigma**2 - nu, 0.0))

    def apply(self, v, damping, ctx):
        # Regrouped so the apply is two (n, rank) matvecs instead of three.
        U, lam = self.basis, self.eigenvalues
        rho = lam[-1]
        Utv = U.T @ v
        return U @ (Utv / (lam + damping) - Utv / (rho + damping)) + v / (rho + damping)


register_pytree_dataclass(
    NystromPreconditioner,
    data_fields=("basis", "eigenvalues"),
    meta_fields=("n", "rank"),
)
