# Levenberg-Marquardt and Tikhonov Regularization

Let \(r(\theta) \in \mathbb{R}^m\) denote the flattened optimization residual after any inequality or complementarity transformations, \(J = D r(\theta) \in \mathbb{R}^{m \times n}\), \(\rho \ge 0\) denote `ridge`, \(\mu \ge 0\) denote scalar step damping, and \(Q = L^\top L \succeq 0\) denote the positive semidefinite penalty operator defined by `ridge_metric`.

## Objective and linear subproblem

The objective is

$$
F_\rho(\theta) = \frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta ,
$$

and `info.loss` reports \(L_\rho(\theta) = 2 F_\rho(\theta) = \|r(\theta)\|_2^2 + \rho\,\theta^\top Q \theta\). When `ridge=None`, \(\rho = 0\) and passing a non-`None` `ridge_metric` raises `ValueError`. When `ridge > 0` and `ridge_metric=None`, \(Q = I_n\).

Linearizing \(r(\theta + d) \approx r(\theta) + J d\) with step damping \(\frac{\mu}{2}\|d\|_2^2\) solves

$$
\min_{d \in \mathbb{R}^n} \; \frac{1}{2}\|r + J d\|_2^2 + \frac{\rho}{2}\|L(\theta + d)\|_2^2 + \frac{\mu}{2}\|d\|_2^2 ,
$$

giving the damped normal equations

$$
H_\mu d = -g, \qquad
H_\mu = J^\top J + \rho Q + \mu I_n, \qquad
g = J^\top r + \rho Q \theta .
$$

For \(\mu > 0\), \(H_\mu \succ 0\) even when \(Q\) is only positive semidefinite. At \(\mu = 0\) and \(\rho > 0\), \(H_0 \succ 0\) up to numerical rank thresholds if and only if \(\ker(J) \cap \ker(L) = \{0\}\). When \(\ker(J) \cap \ker(L) \ne \{0\}\) at \(\mu = 0\), `SVD()` computes the minimum-norm augmented least-squares step.

For linear interpolation \(A \theta = b\) with \(\ker(A) \cap \ker(Q) = \{0\}\), sending \(\rho \to 0^+\) approaches the minimum-seminorm interpolant satisfying

$$
\min_{\theta \in \mathbb{R}^n} \; \frac{1}{2}\theta^\top Q \theta \quad \text{subject to } A\theta = b, \qquad
\begin{bmatrix} Q & A^\top \\ A & 0 \end{bmatrix}
\begin{bmatrix} \theta \\ \lambda \end{bmatrix}
=
\begin{bmatrix} 0 \\ b \end{bmatrix} .
$$

Finite \(\rho > 0\) solves a regularized approximation rather than an exact interpolant.

## Globalization and damping update

`LevenbergMarquardt` does not compute a trust-region predicted reduction or gain ratio. Instead, each trial step is globalized directly by Armijo backtracking along the step ray or geodesic arc \(\theta(\alpha) = \theta_k + \alpha d + \frac{1}{2}\alpha^2 a\), testing \(\alpha_j = \min(1, \texttt{max\_stepsize}) \cdot \texttt{decrease\_factor}^{\,j}\) for \(j = 0, \dots, \texttt{maxls} - 1\). With directional slope \(s = g^\top d < 0\), the first finite step with \(\theta(\alpha) \ne \theta_k\) satisfying

$$
\|r(\theta(\alpha))\|_2^2 + \rho\,\Delta q(\alpha) \le \|r(\theta_k)\|_2^2 + 2 c_1 \alpha s + 2\,\texttt{linesearch\_tol}
$$

is accepted, where \(\Delta q(\alpha) = \theta(\alpha)^\top Q \theta(\alpha) - \theta_k^\top Q \theta_k\) is expanded analytically as a quartic polynomial in \(\alpha\) to avoid catastrophic cancellation at large \(\theta_k^\top Q \theta_k\):

$$
\Delta q(\alpha) = 2\alpha\, d^\top Q \theta_k + \alpha^2 \left(d^\top Q d + a^\top Q \theta_k\right) + \alpha^3\, d^\top Q a + \frac{1}{4}\alpha^4\, a^\top Q a .
$$

After the line search, `LevenbergMarquardt` updates \(\mu\) from the accepted step size \(\alpha_k\):

$$
\mu_{k+1} = \max\!\left(\varepsilon_{\text{tiny}},\, f_k \mu_k\right), \qquad
f_k =
\begin{cases}
\min\!\left(\texttt{damping\_increase},\, \texttt{damping\_decrease} / \alpha_k\right), & \text{if accepted}, \\
1, & \text{if rejected and a constraint schedule changed}, \\
\texttt{damping\_increase}, & \text{otherwise},
\end{cases}
$$

where \(\varepsilon_{\text{tiny}} = \texttt{jnp.finfo(dtype).tiny}\). Defaults are `init_damping=1e-4`, `damping_decrease=0.2`, `damping_increase=2.0`, `c1=1e-4`, `decrease_factor=0.5`, `maxls=20`, and `linesearch_tol=0.0`. On rejected steps with unchanged constraint schedules, `cache_jacobian=True` (the default for dense solvers) reuses `(resid, Jt)` and any valid pre-damping factorization without re-evaluating `J`.

## Geodesic acceleration

When `geodesic_acceleration=True` (default on `LevenbergMarquardt`, `False` on `Newton`), the solver evaluates the directional second derivative \(r_{dd} = D^2 r(\theta_k)[d, d]\) via nested forward-mode `jax.jvp` at detached velocity `d` and solves

$$
H_\mu a = -J^\top r_{dd}
$$

reusing the factorization of \(H_\mu\) prepared for \(d\). Because the augmented penalty map \(\theta \mapsto \sqrt{\rho} L \theta\) is linear, its second directional derivative vanishes and no \(\rho Q\) term enters the right-hand side. The correction \(a\) is included on \(\theta(\alpha) = \theta_k + \alpha d + \frac{1}{2}\alpha^2 a\) when \(g^\top d < 0\), \(a\) is finite, and

$$
0 < \frac{2\|a\|_2}{\|d\|_2 + \varepsilon_{\text{mach}}} \le \texttt{geodesic\_acceptance\_ratio}
$$

(default `0.75`). Otherwise \(a = 0\).

## RKHS seminorms and bordered Gram reductions

For collocation models with \(b\) kernel coefficient blocks \(\alpha_1, \dots, \alpha_b \in \mathbb{R}^{n_k}\) sharing a Gram matrix \(K = R^\top R \succ 0\) and trailing unpenalized coordinates \(\theta_f \in \mathbb{R}^{n_f}\),

$$
\theta = (\alpha_1, \dots, \alpha_b, \theta_f), \qquad
Q = \operatorname{blockdiag}(K, \dots, K, 0_{n_f \times n_f}) .
$$

When \(m < n\) and \(Q_{pp} \succ 0\) on the \(n_p\) penalized coordinates \(\theta_p\) with \(n_f = n - n_p\) unpenalized coordinates \(\theta_f\), `Cholesky(form="gram")` and `QR(form="gram")` partition \(J = [J_p \mid J_f]\), set \(D_p = \rho Q_{pp} + \mu I_{n_p} \succ 0\), and eliminate \(d_p\) without dividing unpenalized coordinates by \(\mu\):

1. Form the \(m \times m\) penalized Schur matrix and base shift
   $$
   S_p = I_m + J_p D_p^{-1} J_p^\top, \qquad
   d_{p,0} = -D_p^{-1}(\rho Q_{pp}\theta_p), \qquad
   \tilde{r} = r + J_p d_{p,0} .
   $$
2. Solve the \(n_f \times n_f\) reduced free-tail system
   $$
   \left(J_f^\top S_p^{-1} J_f + \mu I_{n_f}\right) d_f = -J_f^\top S_p^{-1} \tilde{r} ,
   $$
   then recover \(w = S_p^{-1}(\tilde{r} + J_f d_f)\) and \(d_p = d_{p,0} - D_p^{-1}(J_p^\top w)\).

With triangular factor \(D_p = G_p^\top G_p\), `QR(form="gram")` computes the same reduction through thin QR factorizations

$$
\begin{bmatrix} G_p^{-\top} J_p^\top \\ I_m \end{bmatrix} = \mathcal{Q}_p R_p, \qquad
\begin{bmatrix} R_p^{-\top} J_f \\ \sqrt{\mu}\, I_{n_f} \end{bmatrix} = \mathcal{Q}_f R_f ,
$$

avoiding explicit formation of normal products when \(n_f \le m\) and \(\operatorname{rank}(J_f) = n_f\) at \(\mu = 0\). When \(n_f = 0\), matrix-free `GramCG` solves \(S_p w = \tilde{r}\) by conjugate gradients and recovers \(d = d_{p,0} - D_p^{-1} J^\top w\).

## Ridge continuation (`AnnealRidge`)

`AnnealRidge` lowers `state.ridge` toward `ridge_floor` across outer iterations:

```python
import jax.numpy as jnp
from nlls_gram import AnnealRidge, LevenbergMarquardt, RepeatedFactorRidge

ridge_metric = RepeatedFactorRidge(jnp.linalg.cholesky(K, upper=True), repeats=b)
solver = LevenbergMarquardt(residual_fn, ridge=1e-4, ridge_metric=ridge_metric)
anneal = AnnealRidge(ridge_floor=1e-8, decrease=0.1, grad_rtol=1e-2)
result = solver.solve(
    x0, callback=anneal, user_state=anneal.init_state(), gtol=1e-8, max_steps=400
)
```

Let \(G_k = \texttt{info.grad\_norm}\) denote the base-point gradient norm reported by step \(k\), \(G_{\text{entry}}\) denote the first finite \(G_k\) observed at the current ridge level, and \(G_{\text{prev}}\) denote \(G_k\) from the previous accepted step at that level. `AnnealRidge` updates

$$
\rho_{\text{new}} = \max\!\left(\texttt{decrease} \cdot \rho,\, \texttt{ridge\_floor}\right)
$$

whenever \(G_k \le \texttt{grad\_rtol} \cdot G_{\text{entry}}\), or (when `stall_rtol > 0` and step \(k\) is accepted) \(G_k \ge \texttt{stall\_rtol} \cdot G_{\text{prev}}\). Whenever \(\rho\) strictly decreases, both \(G_{\text{entry}}\) and \(G_{\text{prev}}\) reset to \(+\infty\), convergence is suppressed on that step, and penalty-dependent caches are invalidated. See [Callbacks and Stopping](callbacks.md) for `atol`, `gtol`, and `xtol` rules on penalized problems.

## References

- Levenberg (1944), Marquardt (1963), Moré (1978), Madsen, Nielsen, and Tingleff (2004), and Armijo (1966) for damped least-squares and line-search globalization.
- Transtrum, Machta, and Sethna (2011) and Transtrum and Sethna (2012) for geodesic acceleration and the \(2\|a\|_2 / \|d\|_2\) acceptance gate.
- Tikhonov (1963), Eldén (1977), Björck (1996), Craig (1955), and Saunders (1995) for generalized Tikhonov regularization, augmented QR, and dual Schur/Craig reductions.
- Kimeldorf and Wahba (1971), Wahba (1990), and Schölkopf, Herbrich, and Smola (2001) for RKHS seminorms with unpenalized nullspace coordinates. Full bibliographic details appear in [References](references.md).
