"""Typed linear-solver configuration for ``RidgeLevenbergMarquardt``.

Each config is a small frozen dataclass selecting the algebra for the LM
subproblem (``linear_solver``) or the implicit-AD solve (``ad_solver``),
carrying that solver's own knobs as fields -- so an option that only exists
for one method cannot be passed with another. Instances compare and hash by
value (standard frozen-dataclass semantics), so equal configs key the same
compiled solve loop; construct them inline freely.

- ``Auto()``: resolve by context (forward: ``Cholesky``; AD: ``Cholesky``
  for the dense forwards, ``NormalCG`` under an ``LSMR`` forward).
- ``Cholesky()``: dense normal equations (forward) or the assembled dense
  implicit-AD solve.
- ``QR()``: MINPACK-structured damping-row QR, stable at tiny ridge/damping.
- ``LSMR(preconditioner, ...)``: matrix-free bidiagonalization; the right
  preconditioner is required (``identity_right_preconditioner()`` opts out).
- ``NormalCG(...)``: matrix-free CG on the undamped normal operator for the
  implicit-AD rule.

``LevenbergMarquardt`` (the metric solver) keeps its string-named solver menu
for now; these types are the configuration surface the solvers are converging
on.
"""

from dataclasses import dataclass

from nlls_gram.preconditioners import WhitenedPreconditioner

__all__ = ["Auto", "Cholesky", "LSMR", "NormalCG", "QR"]


@dataclass(frozen=True)
class Auto:
    """Context-resolved solver choice.

    As ``linear_solver`` it resolves to :class:`Cholesky`. As ``ad_solver``
    it resolves to :class:`Cholesky` when the forward solver is dense and to
    :class:`NormalCG` (with its defaults) under an :class:`LSMR` forward.
    """


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
class LSMR:
    """Matrix-free LSMR (Fong-Saunders 2011) on the augmented operator.

    ``preconditioner`` is REQUIRED: a
    :class:`~nlls_gram.WhitenedPreconditioner` applying a parameter-space
    right preconditioner (``identity_right_preconditioner()`` opts out
    explicitly -- nobody should run Krylov methods without thinking about
    preconditioning). ``tol``/``atol`` are the LSMR stopping tolerances
    (traced per-step data, adjustable through ``LMHyperparams``);
    ``maxiter=None`` caps the inner iteration at ``4 * min(m + k, p)``, the
    augmented bidiagonalization's exact-arithmetic bound.
    """

    preconditioner: WhitenedPreconditioner
    tol: float = 0.0
    atol: float = 0.0
    maxiter: int | None = 8

    def __post_init__(self):
        if not isinstance(self.preconditioner, WhitenedPreconditioner):
            raise TypeError(
                "LSMR requires preconditioner, a WhitenedPreconditioner "
                "applying a parameter-space right preconditioner; pass "
                "identity_right_preconditioner() to run unpreconditioned LSMR"
            )
        if self.tol < 0:
            raise ValueError("LSMR.tol must be nonnegative")
        if self.atol < 0:
            raise ValueError("LSMR.atol must be nonnegative")
        if self.maxiter is not None and self.maxiter <= 0:
            raise ValueError("LSMR.maxiter must be positive or None")


@dataclass(frozen=True)
class NormalCG:
    """Matrix-free CG for the implicit-AD solve on ``J'J + ridge L'L``.

    ``preconditioner`` is an optional hook called as ``(v)`` or
    ``(v, damping)`` (the AD system is undamped, so two-argument helpers are
    called with zero damping; helpers marked ``requires_positive_damping``
    are rejected). ``tol=None`` resolves to a dtype default (``1e-10`` in
    float64, ``1e-6`` in float32); ``maxiter`` must be set when both
    tolerances are explicitly zero, since an uncapped zero-tolerance CG loop
    has no stopping rule.
    """

    preconditioner: object = None
    tol: float | None = None
    atol: float = 0.0
    maxiter: int | None = None

    def __post_init__(self):
        if self.preconditioner is not None and not callable(self.preconditioner):
            raise TypeError("NormalCG.preconditioner must be callable or None")
        if self.tol is not None and self.tol < 0:
            raise ValueError("NormalCG.tol must be nonnegative or None")
        if self.atol < 0:
            raise ValueError("NormalCG.atol must be nonnegative")
        if self.maxiter is not None and self.maxiter <= 0:
            raise ValueError("NormalCG.maxiter must be positive or None")
        if self.tol == 0 and self.atol == 0 and self.maxiter is None:
            raise ValueError(
                "NormalCG.maxiter must be set when both tolerances are zero"
            )
