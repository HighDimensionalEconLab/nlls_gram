"""Typed linear-solver configuration for ``RidgeLevenbergMarquardt``.

Each config is a small frozen dataclass selecting the algebra for the LM
subproblem (``linear_solver``) or the implicit-AD solve (``ad_solver``),
carrying that solver's own knobs as fields -- so an option that only exists
for one method cannot be passed with another. Instances compare and hash by
value (standard frozen-dataclass semantics), so equal configs key the same
compiled solve loop; construct them inline freely.

- ``Cholesky()`` (the default): dense normal equations (forward) or the
  assembled dense implicit-AD solve.
- ``QR()``: MINPACK-structured damping-row QR, stable at tiny ridge/damping.
- ``CG(preconditioner, ...)``: matrix-free preconditioned CG on the normal
  operator -- as ``linear_solver`` the damped forward subproblem, as
  ``ad_solver`` the undamped implicit-AD solve. ``preconditioner`` is
  required in both roles; ``identity_preconditioner()`` opts out.

``ad_solver=None`` (the default) matches the forward path's family:
``Cholesky`` for the dense forwards, ``CG`` under a ``CG`` forward.
``LevenbergMarquardt`` (the metric solver) keeps its string-named solver menu
for now; these types are the configuration surface the solvers are converging
on.
"""

from dataclasses import dataclass

from nlls_gram.preconditioners import Preconditioner

__all__ = ["Cholesky", "CG", "QR"]


@dataclass(frozen=True)
class Cholesky:
    """Dense normal-equations solve.

    Forward: assemble ``G = J'J + ridge L'L`` (cached across rejected steps)
    and factor ``G + damping I`` per step. AD: assemble and factor the
    undamped ``J'J + ridge L'L`` once. No knobs.
    """


@dataclass(frozen=True)
class QR:
    """Damping-row QR of the augmented stack ``[J; sqrt(ridge) L]``.

    Backward stable at ``cond(A)`` rather than ``cond(A)^2``, the choice for
    tiny ridge/damping where forming the normal matrix squares the condition
    number. One QR per ``(x, ridge)`` is cached; each step re-factors only the
    damping rows. No knobs.
    """


@dataclass(frozen=True)
class CG:
    """Matrix-free preconditioned CG on the whitened normal operator, in both
    roles.

    As ``linear_solver`` it solves the damped forward subproblem
    ``(J~'J~ + ridge E + damping I) delta_y = -g`` -- the same SPD system the
    :class:`Cholesky` path factors, matrix-free, with the ``ridge`` spectral
    floor on the metric block. The ``preconditioner`` is applied in CG's
    ``M`` slot with the live damping.

    As ``ad_solver`` it solves the undamped implicit-AD system
    ``J~'J~ + ridge E``; the preconditioner is applied with zero damping
    (subclasses marked ``requires_positive_damping`` are rejected for this
    role).

    ``preconditioner`` is REQUIRED in both roles and must be a
    :class:`~nlls_gram.Preconditioner` -- nobody should run Krylov methods
    without a preconditioning decision, so
    :class:`~nlls_gram.IdentityPreconditioner` is the explicit opt-out and a
    custom preconditioner is a small subclass implementing
    ``apply(v, damping, ctx)``. ``tol=None`` resolves to a dtype default
    (``1e-10`` in float64, ``1e-6`` in float32); ``maxiter`` must be set when
    both tolerances are explicitly zero, since an uncapped zero-tolerance CG
    loop has no stopping rule.
    """

    preconditioner: Preconditioner
    tol: float | None = None
    atol: float = 0.0
    maxiter: int | None = None

    def __post_init__(self):
        if not isinstance(self.preconditioner, Preconditioner):
            raise TypeError(
                "CG.preconditioner must be a Preconditioner subclass "
                "instance; pass IdentityPreconditioner() to run "
                "unpreconditioned CG"
            )
        if self.tol is not None and self.tol < 0:
            raise ValueError("CG.tol must be nonnegative or None")
        if self.atol < 0:
            raise ValueError("CG.atol must be nonnegative")
        if self.maxiter is not None and self.maxiter <= 0:
            raise ValueError("CG.maxiter must be positive or None")
        if self.tol == 0 and self.atol == 0 and self.maxiter is None:
            raise ValueError("CG.maxiter must be set when both tolerances are zero")
