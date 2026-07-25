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

from dataclasses import dataclass
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


def identity_preconditioner():
    """The identity map as an explicit "no preconditioner" choice for the
    ``LevenbergMarquardt`` hooks.

    ``linear_solver="gram_cg"`` requires ``dual_preconditioner``, and a
    ``gram_cg``-resolved AD solve requires ``ad_solver_preconditioner`` (under
    ``normal_cg`` the hook is optional) -- nobody should run Krylov methods
    without thinking about preconditioning, so opting out is an explicit,
    greppable decision rather than a silent default. The returned callable
    accepts every hook signature: ``dual_preconditioner(v, damping)`` and
    ``ad_solver_preconditioner(v)``. (The ridge solver's typed opt-out is
    :class:`IdentityPreconditioner`.)
    """

    def preconditioner(v, damping=None):
        return v

    return preconditioner


def identity_right_preconditioner():
    """The identity map as an explicit "no right-preconditioner" choice.

    Returns a :class:`WhitenedPreconditioner` whose ``solve`` and
    ``solve_transpose`` are both the identity, so the metric solver's
    ``linear_solver="lsmr"`` path runs unpreconditioned -- an explicit,
    greppable opt-out rather than a silent default.
    """

    def solve(v, damping):
        return v

    def solve_transpose(w, damping):
        return w

    return WhitenedPreconditioner(solve, solve_transpose)


class WhitenedPreconditioner:
    """Parameter-space right-preconditioner for ``linear_solver="lsmr"``: a
    value-hashable pair ``(solve, solve_transpose)`` applying ``R^{-1}`` and
    ``R^{-T}``.

    LSMR then runs in the preconditioned variable ``z = R u`` on the augmented
    operator ``[B R^{-1}; sqrt(damping) R^{-1}]`` (``B = J S`` for
    ``LevenbergMarquardt``), and the returned step un-preconditions the
    final iterate as ``u = R^{-1} z``. A well-chosen ``R`` (a Schur-complement
    factor of the parameter-space normal operator is the canonical
    construction) clusters the spectrum of ``B R^{-1}`` and cuts the endgame
    iteration count by orders of magnitude::

        def solve(v, damping):
            return jsp_linalg.solve_triangular(R, v)              # R^{-1} v

        def solve_transpose(w, damping):
            return jsp_linalg.solve_triangular(R.T, w)            # R^{-T} w

        solver = LevenbergMarquardt(
            residual_fn, linear_solver="lsmr",
            whitened_preconditioner=WhitenedPreconditioner(solve, solve_transpose),
        )

    - ``solve(v, damping) -> vector`` applies ``R^{-1}`` on a parameter-space
      vector; ``solve_transpose(w, damping) -> vector`` applies ``R^{-T}``. Both
      receive the live ``damping`` (like ``dual_preconditioner(v, damping)``), so
      a ``damping``-analytic ``R`` folds ``lambda`` in exactly.
    - **Exact subproblem for any R**: the augmented damping row is
      ``sqrt(damping) R^{-1} z = sqrt(damping) u``, so every ``damping > 0``
      subproblem is exactly the ``I``-damped
      ``min_u ||r + B u||^2 + damping ||u||^2`` -- the computed step is
      ``u = -(BᵀB + damping I)^{-1} Bᵀ r`` regardless of ``R``. The
      preconditioner changes the iteration path, never the subproblem, and the
      ``damping -> 0`` limit is the minimum-metric-norm step for ANY ``R``.
    - LSMR stopping (``iterative_tol``/``iterative_atol``) is measured on the
      preconditioned operator -- the well-conditioned ``z`` coordinates.

    ``None`` (the ``LevenbergMarquardt`` default) runs plain LSMR.
    Value-hashable on ``(solve, solve_transpose)`` with jit's static-key
    semantics: equal pairs share one compiled solve loop, so define the
    callables once at setup scope.
    """

    def __init__(self, solve, solve_transpose):
        if not callable(solve):
            raise TypeError("WhitenedPreconditioner.solve must be callable")
        if not callable(solve_transpose):
            raise TypeError("WhitenedPreconditioner.solve_transpose must be callable")
        self.solve = solve
        self.solve_transpose = solve_transpose

    def __hash__(self):
        return hash((self.solve, self.solve_transpose))

    def __eq__(self, other):
        return (
            isinstance(other, WhitenedPreconditioner)
            and self.solve == other.solve
            and self.solve_transpose == other.solve_transpose
        )


def sherman_morrison_preconditioner(solve, u, weight):
    """Preconditioner for ``B = A + weight * u u'`` from a solve with ``A``.

    Applies ``B^{-1} v = y - A^{-1}u (u' y) / (1/weight + u' A^{-1} u)`` with
    ``y = A^{-1} v`` by the Sherman-Morrison identity; ``A^{-1} u`` and the
    scalar denominator are precomputed. This is the natural shape for
    kernel-collocation dual operators, where a metric weight ``m`` on a scalar
    parameter injects an exactly known rank-1 spike ``(c^2/m) u u'`` into
    ``J M^{-1} J'``. The ``damping`` argument is accepted per the
    ``dual_preconditioner`` contract and ignored -- spectral closeness to the
    damped operator is all a preconditioner needs -- which also makes the
    helper directly valid as ``ad_solver_preconditioner`` (the solver calls
    two-argument callables with zero damping there).
    """

    solve_u = solve(u)
    denominator = 1.0 / weight + u @ solve_u

    def dual_preconditioner(v, damping):
        y = solve(v)
        return y - solve_u * ((u @ y) / denominator)

    return dual_preconditioner


def woodbury_preconditioner(solve, U, weights):
    """Preconditioner for ``B = A + U diag(weights) U'`` from a solve with ``A``.

    The rank-k generalization of ``sherman_morrison_preconditioner``:
    applies ``B^{-1} v = y - A^{-1}U C^{-1} (U' y)`` with ``y = A^{-1} v``
    and capacitance ``C = diag(1/weights) + U' A^{-1} U`` by the Woodbury
    identity; ``A^{-1} U`` (one matrix solve) and the Cholesky factor of the
    k x k capacitance are precomputed. This is the natural shape when a
    metric weight ``eps`` on a k-vector of scalar parameters injects the
    exactly known rank-k spike ``(c^2/eps) U U'`` into ``J M^{-1} J'``
    (``U`` the corresponding Jacobian columns up to sign and scale). With
    ``k = 1`` it reduces to ``sherman_morrison_preconditioner``. ``weights``
    must be positive -- not validated, since inputs may be traced. The
    ``damping`` argument is accepted per the ``dual_preconditioner``
    contract and ignored, so the helper is directly valid as
    ``ad_solver_preconditioner`` too.
    """

    U = jnp.asarray(U)
    weights = jnp.asarray(weights)
    if U.ndim != 2 or weights.shape != (U.shape[1],):
        raise ValueError("U must have shape (n, k) and weights shape (k,)")
    solve_U = solve(U)
    capacitance = jnp.diag(1.0 / weights) + U.T @ solve_U
    factor = jsp_linalg.cho_factor(capacitance)

    def dual_preconditioner(v, damping):
        y = solve(v)
        return y - solve_U @ jsp_linalg.cho_solve(factor, U.T @ y)

    return dual_preconditioner


def pad_dual_preconditioner(base_preconditioner, n_real):
    """Extend a dual preconditioner to a residual padded with exact zeros.

    The fixed-residual-shape pattern appends ``k`` identically-zero entries to
    an ``n_real``-entry residual so the compiled shapes stay stable across
    problem instances. The padded rows have zero Jacobian rows, so the dual
    operator becomes exactly block diagonal::

        [ J P J' + damping I      0          ]
        [ 0                       damping I  ]

    and the matching preconditioner applies ``base_preconditioner`` on the
    first ``n_real`` coordinates and the exact ``1 / damping`` inverse on the
    padded block -- the second block must NOT be zeroed (that would make the
    preconditioner singular rather than SPD, even though zeros can appear to
    work when the padded coordinates are never excited). Wrapping is needed
    for shape-fixed bases (dense solves, ``nystrom_preconditioner``,
    Sherman-Morrison/Woodbury built at the unpadded size); a shape-generic
    base like ``identity_preconditioner()`` stays valid unwrapped, it just
    forgoes the exact padded-block inverse. Like ``nystrom_preconditioner``
    this uses the live ``damping`` argument, and because the padded block
    divides by it, the returned callback serves only the damped forward
    solve -- never the ``ad_solver_preconditioner`` hook. Relatedly, padded
    rows make the undamped dual ``J P J'`` singular; ``ad_solver="svd"``
    handles this exactly (its spectral filter computes the minimum-metric-norm
    tangent, which equals the unpadded one), while ``ad_solver="qr"`` fails
    loudly there.
    """

    if not isinstance(n_real, int) or isinstance(n_real, bool) or n_real <= 0:
        raise ValueError("n_real must be a positive int")

    def dual_preconditioner(v, damping):
        # Static shapes, so this raises at trace time; without it a
        # shape-generic base would silently accept a too-short vector.
        if v.ndim != 1 or v.shape[0] < n_real:
            raise ValueError(
                f"padded residual vector must be 1-D with at least "
                f"n_real={n_real} entries; got shape {v.shape}"
            )
        return jnp.concatenate(
            (base_preconditioner(v[:n_real], damping), v[n_real:] / damping)
        )

    # The padded block divides by the live damping, so the zero-damping
    # implicit hook must reject this helper at construction.
    dual_preconditioner.requires_positive_damping = True
    return dual_preconditioner


def nystrom_preconditioner(matvec, n, rank, key, *, dtype=None):
    """Randomized Nystrom preconditioner (Frangella-Tropp-Udell) for a PSD
    operator given only through ``matvec``.

    Sketches ``A`` with a rank-``rank`` Nystrom approximation
    ``A_hat = U diag(lam) U'`` -- a thin-QR'd Gaussian test matrix, one
    block application ``Y = A Omega``, and the shifted Cholesky/SVD recovery
    of Frangella, Tropp, and Udell (arXiv:2110.02820, Algorithm 2.1); the
    stabilization shift ``nu ~ eps * ||Y||_F`` is removed from the recovered
    eigenvalues. The returned callback applies the FTU preconditioner
    (their eq. 5.3, up to the positive scalar ``rho + damping``, which CG
    ignores)::

        v  ->  U ((U'v) / (lam + damping)) + (v - U U'v) / (rho + damping)

    where ``rho`` is the smallest retained Nystrom eigenvalue: eigendirections
    the sketch resolved are inverted against the live shift, and the
    unresolved complement is treated as sitting at ``rho`` rather than at
    zero -- that balance is what carries the FTU condition-number guarantee
    for fast-decaying spectra. This is the one shipped base preconditioner
    that uses the live ``damping`` argument (Sherman-Morrison/Woodbury ignore
    it; the ``pad_dual_preconditioner`` wrapper also uses it): one
    construction serves every LM damping value, and passed as
    ``ad_solver_preconditioner`` it is called with zero damping and applies
    the undamped inverse (valid only when the retained spectrum is strictly
    positive).

    The target use is neural-network least squares under the identity
    metric, where the dual operator is the m x m empirical NTK Gram
    ``J J'`` -- fast spectral decay plus the LM damping shift is exactly the
    FTU regime. ``matvec`` must apply a symmetric PSD operator and accept
    ``(n, k)`` matrices (the same shape contract as ``Metric.solve``); an
    indefinite operator silently produces NaN through the Cholesky square
    root. The build costs ``rank`` operator applications plus an
    ``O(n rank^2)`` QR/SVD, done once at construction -- like every
    preconditioner it is frozen there, so for a nonlinear problem it
    approximates the dual at the linearization point it was built from
    (staleness is safe: preconditioner error never moves the converged
    root). Each apply is two ``(n, rank)`` matvecs.

    ``key`` is an explicit PRNG key; the same key reproduces the same
    preconditioner. ``dtype=None`` uses the JAX default float (respects
    x64) -- pass the operator dtype explicitly for a float32 problem under
    enabled x64. All operations are traceable; ``n`` and ``rank`` are static
    Python ints.
    """

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive int")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 < rank <= n:
        raise ValueError("rank must be a positive int <= n")
    if dtype is None:
        dtype = jnp.result_type(float)
    Omega = jnp.linalg.qr(jax.random.normal(key, (n, rank), dtype=dtype))[0]
    Y = matvec(Omega)
    # The floor keeps the shift usable for a (near-)zero operator, where
    # eps * ||Y||_F alone would leave the core singular; tiny/eps stays clear
    # of the subnormal range through the downstream products.
    finfo = jnp.finfo(dtype)
    nu = jnp.maximum(finfo.eps * jnp.linalg.norm(Y), finfo.tiny / finfo.eps)
    Y_nu = Y + nu * Omega
    core = Omega.T @ Y_nu
    L = jnp.linalg.cholesky(0.5 * (core + core.T))
    B = jsp_linalg.solve_triangular(L, Y_nu.T, lower=True).T
    U, sigma, _ = jnp.linalg.svd(B, full_matrices=False)
    lam = jnp.maximum(sigma**2 - nu, 0.0)
    rho = lam[-1]

    def preconditioner(v, damping=0.0):
        # U (U'v)/(lam+damping) + (v - U U'v)/(rho+damping), regrouped so the
        # apply is two (n, rank) matvecs instead of three.
        Utv = U.T @ v
        return U @ (Utv / (lam + damping) - Utv / (rho + damping)) + v / (rho + damping)

    return preconditioner
