"""Preconditioner types and helpers for the package's Krylov paths.

:class:`Preconditioner` (with :class:`IdentityPreconditioner`) is the typed
hook of :class:`~nlls_gram.RidgeLevenbergMarquardt`'s ``CG`` config: an SPD
approximation of the damped whitened normal inverse, receiving the live
solver state through a :class:`~nlls_gram.SolverContext`.

The remaining helpers serve ``LevenbergMarquardt``'s string-named solver
menu: a ``dual_preconditioner(v, damping)`` callback supplies an
approximation of ``(J M^{-1} J' + damping I)^{-1} v`` on residual-space
vectors for ``linear_solver="gram_cg"``. Unlike ``metric.solve`` -- which
defines the converged root and must stay exact -- a preconditioner never
changes the subproblem being solved, so approximations are safe.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


class Preconditioner:
    """SPD preconditioner for ``RidgeLevenbergMarquardt``'s CG paths.

    ``apply(v, damping, ctx)`` returns an SPD approximation of
    ``(J~'J~ + ridge E + damping I)^{-1} v`` on whitened parameter-space
    vectors. In the forward role (``linear_solver=CG(...)``) it sits in CG's
    ``M`` slot with the live damping; in the AD role (``ad_solver=CG(...)``)
    the implicit-AD system is undamped and ``damping`` is zero. ``ctx`` is
    the same :class:`~nlls_gram.SolverContext` the metric factor ops receive
    (the flat iterate, the live ``LMState``, ``args``, ``p``), so a
    preconditioner can key off the solver state. A preconditioner changes
    the CG iteration path, never the subproblem being solved, so
    approximations are safe.

    Implement a custom preconditioner as a small dataclass (``eq=False``
    identity hashing when it holds arrays -- construct once at setup scope
    and reuse, since the instance enters the solver's compile-cache key)::

        @dataclass(frozen=True, eq=False)
        class JacobiPreconditioner(Preconditioner):
            diagonal: jax.Array

            def apply(self, v, damping, ctx):
                return v / (self.diagonal + damping)

    Subclasses whose ``apply`` divides by the live damping must set
    ``requires_positive_damping = True``; the constructor rejects them for
    the AD role, where damping is zero.
    """

    requires_positive_damping = False

    def prepare(self, theta, ctx):
        """Build this preconditioner's numeric state from the current iterate.

        The default is ``None`` -- a stateless preconditioner, whose state
        slot compiles away. Override to hold traced arrays that must track the
        iterate: the returned pytree rides on ``lm_state.precond`` and comes
        back as ``ctx.preconditioner_state`` in :meth:`apply`. It is rebuilt on
        accepted steps and reused across rejected ones (where ``x`` did not
        move), runs inside the jitted loop as traced ops, and is frozen at the
        solution under implicit AD. Its pytree structure must not change
        between rebuilds.

        Expensive setup that does NOT depend on the iterate belongs in
        ``__init__``, where it is paid once.
        """
        return None

    def rebuild(self, ctx):
        """Traced predicate gating a rebuild on an accepted step.

        The default rebuilds every accepted step. Return ``False`` to keep the
        carried state -- staleness only changes the CG iteration path, never
        the converged step, so declining is always safe and often much
        cheaper (e.g. rebuild only when ridge continuation advances a level).
        """
        return True

    def apply(self, v, damping, ctx):
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityPreconditioner(Preconditioner):
    """The identity map as an explicit "no preconditioner" choice for ``CG``.

    Nobody should run Krylov methods without thinking about preconditioning,
    so opting out is an explicit, greppable decision rather than a silent
    default. Stateless and value-equal: two instances compare equal, so
    equal ``CG`` configs share one compiled solve loop.
    """

    def apply(self, v, damping, ctx):
        return v


@dataclass(frozen=True, eq=False)
class BlockEigenPreconditioner(Preconditioner):
    """Block-diagonal eigenbasis preconditioner that owns its state.

    The workhorse for structured whitened normal operators
    ``J~'J~ + ridge E + damping I`` built from repeated interacting blocks
    (multiple "agents" coupled through shared equations): approximate the
    operator by a block-diagonal matrix over a chosen grouping of the
    whitened coordinates, eigendecompose each block, and apply the exact
    inverse of the shifted approximation

        v  ->  V ((V' v) / (Lambda + ridge_weight * ridge + damping)) V'

    per block -- analytic in both the live ``damping`` (traced; it changes per
    LM step) and the live ``ridge`` (read from ``ctx.lm_state.ridge``, so
    ridge continuation composes with no rebuild).

    ``blocks_fn(theta, ctx)`` returns the family list that
    :func:`block_eigen_state` packs: ``(blocks, ridge_weight)`` pairs whose
    ``blocks`` has shape ``(groups, size, size)`` -- the stacked diagonal
    blocks of ``J~'J~`` restricted to that family's coordinate groups, in
    permuted order. Families in the metric block set ``ridge_weight = 1``
    (their diagonal carries the ``ridge`` spectral floor); free-block families
    set ``0`` (damping-only, so the zero-damping AD role applies their plain
    inverse -- positive definite whenever the free block is identified).
    ``permutation`` reorders the flattened whitened vector into family-major
    order; ``jnp.arange(n)`` serves when the natural layout already is.

    The eigendecomposition runs in :meth:`prepare` from the live iterate, so
    it is rebuilt on accepted steps and reused across rejected ones. Override
    ``rebuild`` to decline -- a stale state only changes the CG iteration
    path, never the converged step, so refreshing only when ridge continuation
    advances a level is a pure saving::

        class OnLevelChange(BlockEigenPreconditioner):
            def rebuild(self, ctx):
                return ctx.lm_state.ridge < self.last_ridge
    """

    blocks_fn: Any
    permutation: jax.Array

    def prepare(self, theta, ctx):
        return block_eigen_state(self.blocks_fn(theta, ctx), self.permutation)

    def apply(self, v, damping, ctx):
        state = ctx.preconditioner_state
        ridge = jnp.asarray(ctx.lm_state.ridge, dtype=v.dtype)
        permuted = v[state["permutation"]]
        pieces = []
        offset = 0
        for family in state["families"]:
            V = family["eigenvectors"]
            eigenvalues = family["eigenvalues"]
            groups, size = V.shape[0], V.shape[1]
            segment = permuted[offset : offset + groups * size]
            offset += groups * size
            shift = family["ridge_weight"] * ridge + damping
            coefficients = jnp.einsum("gab,ga->gb", V, segment.reshape(groups, size))
            solved = coefficients / (eigenvalues + shift)
            pieces.append(jnp.einsum("gab,gb->ga", V, solved).reshape(-1))
        if offset != permuted.shape[0]:
            raise ValueError(
                f"block_eigen_state families cover {offset} coordinates but "
                f"the parameter vector has {permuted.shape[0]}"
            )
        return jnp.concatenate(pieces)[state["inverse_permutation"]].astype(v.dtype)


def block_eigen_state(families, permutation):
    """Pack stacked SPD diagonal blocks into a
    :class:`BlockEigenPreconditioner` state pytree.

    ``families`` is a sequence of ``(blocks, ridge_weight)`` pairs:
    ``blocks`` has shape ``(groups, size, size)`` -- the stacked diagonal
    blocks of the whitened normal operator ``J~'J~`` restricted to that
    family's coordinate groups, in permuted order -- and ``ridge_weight`` is
    ``1.0`` for metric-block families (the apply shift includes the live
    ridge) or ``0.0`` for free-block families (damping-only). Blocks are
    symmetrized and eigendecomposed here, once; positive semidefiniteness is
    assumed, not validated (entries may be traced).

    ``permutation`` is the 1-D integer array reordering the flattened
    whitened parameter vector into family-major order
    (``v_permuted = v[permutation]``); families are consumed in sequence
    and must cover it exactly. The identity ``jnp.arange(p)`` serves when
    the natural layout is already family-major.

    Fully traceable, so an adaptive rebuild can run inside a jitted solve
    callback; the state's pytree structure (family count and shapes) is
    static and must not change across rebuilds.
    """

    permutation = jnp.asarray(permutation)
    if permutation.ndim != 1 or not jnp.issubdtype(permutation.dtype, jnp.integer):
        raise ValueError("permutation must be a 1-D integer array")
    packed = []
    covered = 0
    for blocks, ridge_weight in families:
        blocks = jnp.asarray(blocks)
        if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
            raise ValueError(
                "each family's blocks must have shape (groups, size, size); "
                f"got {blocks.shape}"
            )
        symmetrized = 0.5 * (blocks + jnp.swapaxes(blocks, 1, 2))
        eigenvalues, eigenvectors = jnp.linalg.eigh(symmetrized)
        # eigh of a numerically PSD block can return tiny negative
        # eigenvalues; clamped at zero the apply shift stays positive for
        # any positive ridge/damping (and the zero-damping AD role stays
        # SPD whenever the family itself is).
        eigenvalues = jnp.maximum(eigenvalues, 0.0)
        packed.append(
            {
                "eigenvectors": eigenvectors,
                "eigenvalues": eigenvalues,
                "ridge_weight": jnp.asarray(ridge_weight, dtype=blocks.dtype),
            }
        )
        covered += blocks.shape[0] * blocks.shape[1]
    if covered != permutation.shape[0]:
        raise ValueError(
            f"families cover {covered} coordinates but the permutation has "
            f"{permutation.shape[0]}"
        )
    return {
        "permutation": permutation,
        "inverse_permutation": jnp.argsort(permutation),
        "families": tuple(packed),
    }


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
    """

    solve: object
    u: jax.Array
    weight: float
    _solve_u: jax.Array = field(init=False)
    _denominator: jax.Array = field(init=False)

    def __post_init__(self):
        solve_u = self.solve(self.u)
        object.__setattr__(self, "_solve_u", solve_u)
        object.__setattr__(self, "_denominator", 1.0 / self.weight + self.u @ solve_u)

    def apply(self, v, damping, ctx):
        y = self.solve(v)
        return y - self._solve_u * ((self.u @ y) / self._denominator)


@dataclass(frozen=True, eq=False)
class WoodburyPreconditioner(Preconditioner):
    """Dual preconditioner for ``B = A + U diag(weights) U'``.

    The rank-k generalization of :class:`ShermanMorrisonPreconditioner`:
    applies ``B^{-1} v = y - A^{-1}U C^{-1}(U' y)`` with ``y = A^{-1} v`` and
    capacitance ``C = diag(1/weights) + U' A^{-1} U``; ``A^{-1}U`` (one matrix
    solve) and the Cholesky factor of the k x k capacitance are precomputed.
    ``weights`` must be positive -- not validated, since inputs may be traced.
    Like Sherman-Morrison it ignores ``damping`` and so serves the AD role too.
    """

    solve: object
    U: jax.Array
    weights: jax.Array
    _solve_U: jax.Array = field(init=False)
    _factor: tuple = field(init=False)

    def __post_init__(self):
        U, weights = jnp.asarray(self.U), jnp.asarray(self.weights)
        if U.ndim != 2 or weights.shape != (U.shape[1],):
            raise ValueError("U must have shape (n, k) and weights shape (k,)")
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "weights", weights)
        solve_U = self.solve(U)
        object.__setattr__(self, "_solve_U", solve_U)
        capacitance = jnp.diag(1.0 / weights) + U.T @ solve_U
        object.__setattr__(self, "_factor", jsp_linalg.cho_factor(capacitance))

    def apply(self, v, damping, ctx):
        y = self.solve(v)
        return y - self._solve_U @ jsp_linalg.cho_solve(self._factor, self.U.T @ y)


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
    indefinite one silently produces NaN through the Cholesky square root. The
    build costs ``rank`` operator applications plus an ``O(n rank^2)``
    QR/SVD, paid once at construction, so for a nonlinear problem it
    approximates the dual at the linearization point it was built from
    (staleness is safe). Each apply is two ``(n, rank)`` matvecs.

    ``key`` is an explicit PRNG key; the same key reproduces the same
    preconditioner. ``dtype=None`` uses the JAX default float -- pass the
    operator dtype explicitly for a float32 problem under enabled x64.
    """

    matvec: object
    n: int
    rank: int
    key: jax.Array
    dtype: object = None
    _basis: jax.Array = field(init=False)
    _eigenvalues: jax.Array = field(init=False)

    def __post_init__(self):
        if not 0 < self.rank <= self.n:
            raise ValueError("rank must be a positive int <= n")
        dtype = jnp.result_type(float) if self.dtype is None else self.dtype
        shape = (self.n, self.rank)
        Omega = jnp.linalg.qr(jax.random.normal(self.key, shape, dtype))[0]
        Y = self.matvec(Omega)
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
        object.__setattr__(self, "_basis", U)
        object.__setattr__(self, "_eigenvalues", jnp.maximum(sigma**2 - nu, 0.0))

    def apply(self, v, damping, ctx):
        # Regrouped so the apply is two (n, rank) matvecs instead of three.
        U, lam = self._basis, self._eigenvalues
        rho = lam[-1]
        Utv = U.T @ v
        return U @ (Utv / (lam + damping) - Utv / (rho + damping)) + v / (rho + damping)
