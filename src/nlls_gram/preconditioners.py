"""Dual-preconditioner helpers for the ``linear_solver="gram_cg"`` path.

A ``dual_preconditioner(v, damping)`` callback supplies an approximation of
``(J M^{-1} J' + damping I)^{-1} v`` on residual-space vectors. Unlike
``metric.solve`` -- which defines the converged root and must stay exact -- a
preconditioner never changes the subproblem being solved, so approximations
are safe.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def identity_preconditioner():
    """The identity map as an explicit "no preconditioner" choice.

    ``linear_solver="gram_cg"`` requires ``dual_preconditioner``, and a
    cg-resolved
    implicit solve requires ``ad_solver_preconditioner`` -- nobody should run
    Krylov methods without thinking about preconditioning, so opting out is an
    explicit, greppable decision rather than a silent default. The returned
    callable accepts both hook signatures: ``dual_preconditioner(v, damping)``
    and ``ad_solver_preconditioner(v)``.
    """

    def preconditioner(v, damping=None):
        return v

    return preconditioner


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
    rows make the undamped dual ``J P J'`` singular; the default dense AD
    rule handles this exactly (its spectral filter computes the
    minimum-metric-norm tangent, which equals the unpadded one), while an
    explicit ``ad_solver_penalty=0.0`` fails loudly there.
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
    for fast-decaying spectra. This is the one shipped helper that uses the
    live ``damping`` argument (Sherman-Morrison/Woodbury ignore it): one
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
