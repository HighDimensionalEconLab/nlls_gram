"""Quasiseparable (rank-m semiseparable) kernel algorithms.

The generator conventions and the Cholesky / substitution / matvec
recursions are transcribed from tinygp v0.3.1 (MIT License, Copyright (c)
2021-2024 Daniel Foreman-Mackey; ``tinygp/kernels/quasisep.py`` and
``tinygp/solvers/quasisep/core.py``), which builds on Eidelman & Gohberg
(1999) and Foreman-Mackey et al. (2017, celerite). The associative-scan
apply variants follow Sarkka & Garcia-Fernandez (2021).

A symmetric quasiseparable matrix is given by generators ``d`` (n,),
``p`` (n, m), ``q`` (n, m), ``A`` (n, m, m) as

    K[i, j] = p[i] @ A[i-1] @ ... @ A[j+1] @ q[j]   for i > j,
    K[i, i] = d[i],  K[j, i] = K[i, j],

where ``A[k]`` is the transition INTO index ``k`` (``A[0]`` is unused by the
products above and is the identity in the state-space construction). Its
Cholesky factor ``L`` (``L @ L.T = K``) has diagonal ``c`` and the same
strictly-lower generator structure with the same ``p`` and ``A`` and ``q``
replaced by ``w``.

All applies accept a matrix right-hand side ``y`` of shape (n, r); the
carries are m x m (or m x r) dense products, so every pass costs O(n m^2).
Each apply has a sequential ``lax.scan`` form and a parallel
``lax.associative_scan`` form over (matrix, offset) pairs composing as
``(M2 @ M1, M2 @ b1 + b2)``; the parallel substitutions eliminate the output
from the carry recursion, which makes it affine, and recover the outputs
from the exclusive-prefix carries in a vectorized post-map.
"""

import jax
import jax.numpy as jnp


def matern_state_space(sigma, ell, nu):
    """Return the ``(h, Pinf, transition)`` state-space model of a Matern kernel.

    The half-integer Matern kernels
    ``k(tau) = sigma^2 matern_nu(tau / ell)`` for ``nu`` in
    ``{0.5, 1.5, 2.5}`` (static) are exactly the covariances of
    continuous-time autoregressive (CAR(m)) state-space models with state
    dimension m = 1, 2, 3. This returns their observation row ``h`` (m,)
    (``sigma`` lives in ``h``, not ``Pinf``), stationary state covariance
    ``Pinf`` (m, m), and ``transition(dt)`` mapping gaps (n,) to stacked
    transition matrices (n, m, m) in the tinygp orientation (transpose of
    the textbook ``expm(F dt)``), for ``metric_from_state_space``.

    With ``f = sqrt(2 nu) / ell``: nu=0.5 has ``h = [sigma]``, ``Pinf = 1``,
    ``transition(dt) = exp(-f dt)``; nu=1.5 has ``h = [sigma, 0]``,
    ``Pinf = diag(1, f^2)``; nu=2.5 has ``h = [sigma, 0, 0]`` and the CAR(3)
    ``Pinf``/transition transcribed from tinygp v0.3.1.

    Nugget-free Matern-3/2 and 5/2 Grams on fine grids are extremely
    ill-conditioned (condition number ~1e21 at n = 5000 — a property of the
    matrix, not the solver); pass an ABSOLUTE nugget (e.g.
    ``1e-8 * sigma**2`` in float64) to ``metric_from_state_space``, which
    folds it into the metric exactly. For nu=0.5 the Gram inverse is exactly
    tridiagonal and ``metric_from_tridiagonal_precision`` is the specialized
    alternative whose applies are elementwise shifts.
    """

    if nu not in (0.5, 1.5, 2.5):
        raise ValueError("nu must be one of 0.5, 1.5, or 2.5")
    sigma = jnp.asarray(sigma)
    ell = jnp.asarray(ell)
    one = jnp.ones_like(ell)
    zero = jnp.zeros_like(ell)

    if nu == 0.5:
        h = sigma[None]
        Pinf = one[None, None]

        def transition(dt):
            return jnp.exp(-dt / ell)[:, None, None]

    elif nu == 1.5:
        f = jnp.sqrt(3.0) / ell
        h = jnp.stack([sigma, jnp.zeros_like(sigma)])
        Pinf = jnp.stack([jnp.stack([one, zero]), jnp.stack([zero, f**2])])

        def transition(dt):
            fd = f * dt
            return jnp.exp(-fd)[:, None, None] * jnp.stack(
                [
                    jnp.stack([1.0 + fd, -(f**2) * dt], axis=-1),
                    jnp.stack([dt, 1.0 - fd], axis=-1),
                ],
                axis=-2,
            )

    else:
        f = jnp.sqrt(5.0) / ell
        f2 = f**2
        h = jnp.stack([sigma, jnp.zeros_like(sigma), jnp.zeros_like(sigma)])
        Pinf = jnp.stack(
            [
                jnp.stack([one, zero, -f2 / 3.0]),
                jnp.stack([zero, f2 / 3.0, zero]),
                jnp.stack([-f2 / 3.0, zero, f2**2]),
            ]
        )

        def transition(dt):
            fd = f * dt
            d2 = dt**2
            return jnp.exp(-fd)[:, None, None] * jnp.stack(
                [
                    jnp.stack(
                        [
                            0.5 * f2 * d2 + fd + 1.0,
                            -0.5 * f * f2 * d2,
                            0.5 * f2 * f * dt * (fd - 2.0),
                        ],
                        axis=-1,
                    ),
                    jnp.stack(
                        [dt * (fd + 1.0), -f2 * d2 + fd + 1.0, f2 * dt * (fd - 3.0)],
                        axis=-1,
                    ),
                    jnp.stack(
                        [
                            0.5 * d2,
                            0.5 * dt * (2.0 - fd),
                            0.5 * f2 * d2 - 2.0 * fd + 1.0,
                        ],
                        axis=-1,
                    ),
                ],
                axis=-2,
            )

    return h, Pinf, transition


def state_space_generators(points, h, Pinf, transition):
    points = jnp.asarray(points)
    if points.ndim != 1:
        raise ValueError("points must be a 1-D array of sorted locations")
    n = points.shape[0]
    # The dt=0 shift trick makes A[0] = transition(0) = I.
    dt = points - jnp.concatenate([points[:1], points[:-1]])
    A = transition(dt)
    hP = h @ Pinf
    d = jnp.full((n,), hP @ h)
    q = jnp.broadcast_to(h, (n, h.shape[0]))
    p = jnp.einsum("i,kij->kj", hP, A)
    return d, p, q, A


def cholesky(d, p, q, A):
    # F_k = sum_{j<k} Phi_{k,j} w_j w_j' Phi_{k,j}', the Riccati-flow carry.
    m = p.shape[1]

    def step(F, inputs):
        d_k, p_k, q_k, A_k = inputs
        c_k = jnp.sqrt(d_k - p_k @ F @ p_k)
        tmp = F @ A_k.T
        w_k = (q_k - p_k @ tmp) / c_k
        F_next = A_k @ tmp + jnp.outer(w_k, w_k)
        return F_next, (c_k, w_k)

    F0 = jnp.zeros((m, m), dtype=d.dtype)
    _, (c, w) = jax.lax.scan(step, F0, (d, p, q, A))
    return c, w


def scan_affine(M, b, reverse=False):
    # Compose G_{k+1} = M_k @ G_k + b_k with G_0 = 0 and return the
    # exclusive-prefix carries G_k (reversed variant runs from the far end).
    def combine(earlier, later):
        M1, b1 = earlier
        M2, b2 = later
        return M2 @ M1, M2 @ b1 + b2

    if reverse:
        M = M[::-1]
        b = b[::-1]
    _, cumulative = jax.lax.associative_scan(combine, (M, b))
    carries = jnp.concatenate([jnp.zeros_like(cumulative[:1]), cumulative[:-1]])
    if reverse:
        carries = carries[::-1]
    return carries


def forward_substitution(c, p, w, A, y, parallel):
    # Solve L x = y with x[k] = (y[k] - p[k] @ G_k) / c[k],
    # G_{k+1} = A[k] @ G_k + outer(w[k], x[k]).
    if parallel:
        M = A - w[:, :, None] * (p / c[:, None])[:, None, :]
        b = w[:, :, None] * (y / c[:, None])[:, None, :]
        G = scan_affine(M, b)
        return (y - jnp.einsum("km,kmr->kr", p, G)) / c[:, None]

    def step(G, inputs):
        c_k, p_k, w_k, A_k, y_k = inputs
        x_k = (y_k - p_k @ G) / c_k
        return A_k @ G + jnp.outer(w_k, x_k), x_k

    G0 = jnp.zeros((p.shape[1], y.shape[1]), dtype=y.dtype)
    _, x = jax.lax.scan(step, G0, (c, p, w, A, y))
    return x


def backward_substitution(c, p, w, A, y, parallel):
    # Solve L' x = y with x[k] = (y[k] - w[k] @ G_k) / c[k],
    # G_{k-1} = A[k].T @ G_k + outer(p[k], x[k])  (p/w roles swap vs forward).
    if parallel:
        M = jnp.swapaxes(A, -1, -2) - p[:, :, None] * (w / c[:, None])[:, None, :]
        b = p[:, :, None] * (y / c[:, None])[:, None, :]
        G = scan_affine(M, b, reverse=True)
        return (y - jnp.einsum("km,kmr->kr", w, G)) / c[:, None]

    def step(G, inputs):
        c_k, p_k, w_k, A_k, y_k = inputs
        x_k = (y_k - w_k @ G) / c_k
        return A_k.T @ G + jnp.outer(p_k, x_k), x_k

    G0 = jnp.zeros((p.shape[1], y.shape[1]), dtype=y.dtype)
    _, x = jax.lax.scan(step, G0, (c, p, w, A, y), reverse=True)
    return x


def matvec(d, p, q, A, x, parallel):
    # y = d * x + lower + upper, each contribution read from the carry
    # BEFORE the update (exclusive prefix): lower_k = p[k] @ G_k with
    # G_{k+1} = A[k] @ G_k + outer(q[k], x[k]), and the reversed analogue
    # upper_k = q[k] @ H_k with H_{k-1} = A[k].T @ H_k + outer(p[k], x[k]).
    if parallel:
        G = scan_affine(A, q[:, :, None] * x[:, None, :])
        H = scan_affine(
            jnp.swapaxes(A, -1, -2), p[:, :, None] * x[:, None, :], reverse=True
        )
        lower = jnp.einsum("km,kmr->kr", p, G)
        upper = jnp.einsum("km,kmr->kr", q, H)
        return d[:, None] * x + lower + upper

    def lower_step(G, inputs):
        p_k, q_k, A_k, x_k = inputs
        return A_k @ G + jnp.outer(q_k, x_k), p_k @ G

    def upper_step(H, inputs):
        p_k, q_k, A_k, x_k = inputs
        return A_k.T @ H + jnp.outer(p_k, x_k), q_k @ H

    G0 = jnp.zeros((p.shape[1], x.shape[1]), dtype=x.dtype)
    _, lower = jax.lax.scan(lower_step, G0, (p, q, A, x))
    _, upper = jax.lax.scan(upper_step, G0, (p, q, A, x), reverse=True)
    return d[:, None] * x + lower + upper
