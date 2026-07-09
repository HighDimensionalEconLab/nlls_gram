"""Dual-preconditioner helpers for the ``linear_solver="cg"`` path.

A ``dual_preconditioner(v, damping)`` callback supplies an approximation of
``(J M^{-1} J' + damping I)^{-1} v`` on residual-space vectors. Unlike
``metric.solve`` -- which defines the converged root and must stay exact -- a
preconditioner only changes the inner CG iteration count, so approximations
are safe.
"""

from __future__ import annotations


def sherman_morrison_preconditioner(solve, u, weight):
    """Preconditioner for ``P = A + weight * u u'`` from a solve with ``A``.

    Applies ``P^{-1} v = y - A^{-1}u (u' y) / (1/weight + u' A^{-1} u)`` with
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
