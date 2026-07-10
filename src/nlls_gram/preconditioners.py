"""Dual-preconditioner helpers for the ``linear_solver="cg"`` path.

A ``dual_preconditioner(v, damping)`` callback supplies an approximation of
``(J M^{-1} J' + damping I)^{-1} v`` on residual-space vectors. Unlike
``metric.solve`` -- which defines the converged root and must stay exact -- a
preconditioner never changes the subproblem being solved, so approximations
are safe.
"""

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def identity_preconditioner():
    """The identity map as an explicit "no preconditioner" choice.

    ``linear_solver="cg"`` requires ``dual_preconditioner``, and a cg-resolved
    implicit solve requires ``implicit_preconditioner`` -- nobody should run
    Krylov methods without thinking about preconditioning, so opting out is an
    explicit, greppable decision rather than a silent default. The returned
    callable accepts both hook signatures: ``dual_preconditioner(v, damping)``
    and ``implicit_preconditioner(v)``.
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
    damped operator is all a preconditioner needs.
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
    contract and ignored.
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
