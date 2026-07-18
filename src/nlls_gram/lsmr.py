"""Matrix-free LSMR for the whitened damped least-squares LM subproblem.

LSMR (Fong & Saunders 2011) solves ``min_x ||A x - b||^2 + damp^2 ||x||^2`` given
only ``A`` and ``A'`` as matvecs, via Golub-Kahan bidiagonalization. It drives the
normal-equations residual ``||A'(b - A x) - damp^2 x||`` monotonically to zero.

The LM use is the whitened subproblem ``min_u ||r + B u||^2 + lambda ||u||^2`` with
``B = J S`` (``S = metric.inv_sqrt``, ``S S' = M^{-1}``) and step ``s = S u``. That
operator has condition number ``sqrt`` of the ``cg`` dual's ``J M^{-1} J' + lambda
I`` -- LSMR reaches the accuracy floor of the whitened operator, not its square, so
the selection-critical slow directions stay resolved at small ``lambda`` where the
squared dual solve degrades.

Stopping maps the package's iterative hooks: the normal-equations residual
``normar = |zetabar|`` (LSMR's exact monotone quantity) is driven below
``iterative_tol * normar_0 + iterative_atol`` where ``normar_0 = ||A'b||``, capped
by ``iterative_maxiter`` (all traced, so a solve callback can reschedule them).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

_HIGHEST = lax.Precision.HIGHEST


class LSMRState(NamedTuple):
    """Diagnostics emitted by :func:`lsmr` alongside the solution.

    Attributes:
        iterations: number of LSMR iterations run (``()`` int32).
        normal_residual: final ``||A'(b - A x) - damp^2 x||`` (``()`` scalar), the
            monotone normal-equations residual LSMR minimizes.
    """

    iterations: jax.Array
    normal_residual: jax.Array


def _norm(x):
    return jnp.sqrt(jnp.real(jnp.vdot(x, x, precision=_HIGHEST)))


def _safe_div(a, b):
    # Divisions by a rotation scalar that is only zero at exact convergence (the
    # loop has stopped); keep the carry finite so a post-stop step stays clean.
    return jnp.where(
        b == 0, jnp.zeros_like(a), a / jnp.where(b == 0, jnp.ones_like(b), b)
    )


def _sym_ortho(a, b):
    # Givens rotation [c s; -s c] [a; b] = [r; 0] with r = hypot(a, b) >= 0.
    r = jnp.hypot(a, b)
    c = _safe_div(a, r)
    s = _safe_div(b, r)
    return c, s, r


def lsmr_solve(A, At, b, damp, atol, btol, maxiter, n):
    """Core LSMR loop solving ``min ||A x - b||^2 + damp^2 ||x||^2``.

    ``A`` maps ``R^n -> R^m`` and ``At`` its transpose ``R^m -> R^n`` (matvecs);
    ``b`` is ``R^m``. ``damp >= 0`` is the Tikhonov weight (``sqrt(lambda)`` for LM).
    ``atol`` (relative) and ``btol`` (absolute) bound the normal-equations residual;
    ``maxiter`` (traced int) caps iterations, ``n`` is the static solution size.
    Returns ``(x, LSMRState)``. Not reverse-differentiable on its own (a raw
    ``while_loop``) -- wrap the solution in ``lax.custom_linear_solve`` for AD.
    """
    dtype = b.dtype
    zero = jnp.zeros((), dtype)
    one = jnp.ones((), dtype)
    damp = jnp.asarray(damp, dtype)
    atol = jnp.asarray(atol, dtype)
    btol = jnp.asarray(btol, dtype)

    beta = _norm(b)
    u = jnp.where(beta > 0, b / jnp.where(beta > 0, beta, one), b)
    v0 = At(u)
    alpha = _norm(v0)
    v = jnp.where(alpha > 0, v0 / jnp.where(alpha > 0, alpha, one), v0)

    normar0 = alpha * beta  # ||A' b||
    x0 = jnp.zeros((n,), dtype)
    hbar0 = jnp.zeros((n,), dtype)

    # (itn, u, v, alpha, beta, zetabar, alphabar, rho, rhobar, cbar, sbar, h, hbar,
    #  x, normar, stop)
    init = (
        jnp.zeros((), jnp.int32),
        u,
        v,
        alpha,
        beta,
        alpha * beta,  # zetabar
        alpha,  # alphabar
        one,  # rho
        one,  # rhobar
        one,  # cbar
        zero,  # sbar
        v,  # h
        hbar0,
        x0,
        normar0,  # normar (pre-loop estimate)
        (normar0 <= btol) | (maxiter <= 0),  # already converged / no iterations
    )

    def cond(carry):
        itn = carry[0]
        stop = carry[-1]
        return (~stop) & (itn < maxiter)

    def body(carry):
        (
            itn,
            u,
            v,
            alpha,
            beta,
            zetabar,
            alphabar,
            rho,
            rhobar,
            cbar,
            sbar,
            h,
            hbar,
            x,
            _,
            _,
        ) = carry
        itn = itn + 1

        # Continue the bidiagonalization: next beta, u and alpha, v.
        u = A(v) - alpha * u
        beta = _norm(u)
        u = jnp.where(beta > 0, u / jnp.where(beta > 0, beta, one), u)
        v_raw = At(u) - beta * v
        alpha = _norm(v_raw)
        v = jnp.where(alpha > 0, v_raw / jnp.where(alpha > 0, alpha, one), v_raw)

        # Damping rotation, then the two plane rotations of LSMR.
        chat, shat, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        thetabar = sbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, hbar, x.
        hbar = h - _safe_div(thetabar * rho, rhoold * rhobarold) * hbar
        x = x + _safe_div(zeta, rho * rhobar) * hbar
        h = v - _safe_div(thetanew, rho) * h

        normar = jnp.abs(zetabar)  # ||A' r_k|| exactly
        stop = (normar <= atol * normar0 + btol) | (itn >= maxiter)
        return (
            itn,
            u,
            v,
            alpha,
            beta,
            zetabar,
            alphabar,
            rho,
            rhobar,
            cbar,
            sbar,
            h,
            hbar,
            x,
            normar,
            stop,
        )

    carry = lax.while_loop(cond, body, init)
    itn = carry[0]
    x = carry[13]
    normar = carry[14]
    return x, LSMRState(iterations=itn, normal_residual=normar)


def lsmr(A, At, b, *, damp=0.0, atol=1e-6, btol=0.0, maxiter, n=None):
    """Solve ``min_x ||A x - b||^2 + damp^2 ||x||^2`` by LSMR (see :func:`lsmr_solve`).

    ``A``/``At`` are the operator and its transpose as matvec callables, ``b`` the
    right-hand side. ``n`` (the solution size) defaults to ``At(b).shape[0]``.
    Returns ``(x, LSMRState)``; not reverse-differentiable (raw loop).
    """
    if n is None:
        n = At(b).shape[0]
    return lsmr_solve(A, At, b, damp, atol, btol, maxiter, n)
