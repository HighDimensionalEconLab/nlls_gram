# Tikhonov Regularization

Both `LevenbergMarquardt` and `Newton` minimize a single mathematical half-objective with Euclidean step geometry. Let \(r(\theta) \in \mathbb{R}^m\) denote the complete optimization residual after any inequality or complementarity transformations, \(J = D r(\theta) \in \mathbb{R}^{m \times n}\), \(\rho > 0\) denote `ridge`, \(\mu \ge 0\) denote scalar damping, and \(Q = L^\top L \succeq 0\) denote the fixed PSD operator defined by `ridge_metric`.

## Objective and Euclidean step equations

The half-objective is

$$
F_\rho(\theta) = \frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta ,
$$

and `info.loss` reports \(2 F_\rho(\theta) = \|r(\theta)\|_2^2 + \rho\,\theta^\top Q \theta\). When `ridge=None`, \(\rho = 0\) in all equations below and supplying a non-`None` `ridge_metric` raises `ValueError`.

Linearizing \(r(\theta + d) \approx r(\theta) + J d\) with Euclidean trust-region damping \(\frac{\mu}{2}\|d\|_2^2\) solves

$$
\min_{d \in \mathbb{R}^n} \; \frac{1}{2}\|r + J d\|_2^2 + \frac{\rho}{2}\|L(\theta + d)\|_2^2 + \frac{\mu}{2}\|d\|_2^2 ,
$$

whose normal equations are

$$
H_\mu d = -g, \qquad
H_\mu = J^\top J + \rho Q + \mu I_n, \qquad
g = J^\top r + \rho Q \theta .
$$

The \(\rho Q \theta\) gradient term penalizes parameter magnitude in the objective, whereas \(\mu I_n\) damps only the step \(d\) and never adds \(\mu \theta\) to \(g\). All reported step norms (`info.step_norm`), gradient norms (`info.grad_norm`), directional derivatives, geodesic ratios, and stopping tolerances (`atol`, `gtol`, `xtol`) are evaluated in original Euclidean coordinates.

For \(\mu > 0\), \(H_\mu\) is strictly positive definite even when \(Q\) is only positive semidefinite. At \(\mu = 0\) and \(\rho > 0\), \(H_0 = J^\top J + \rho Q\) is strictly positive definite if and only if \(\ker(J) \cap \ker(L) = \{0\}\). When \(\ker(J) \cap \ker(L) \ne \{0\}\) at \(\mu = 0\), `SVD()` selects the Euclidean minimum-norm augmented least-squares step while non-rank-revealing solvers report `LINEAR_SOLVE_FAILED`.

## RKHS seminorms with unpenalized coordinates

For collocation models with \(b\) kernel coefficient blocks \(\alpha_1, \dots, \alpha_b \in \mathbb{R}^{n_k}\) sharing a Gram matrix \(K = R^\top R \succ 0\) and trailing unpenalized structural or initial-value scalars \(\theta_f \in \mathbb{R}^{n_f}\),

$$
\theta = (\alpha_1, \dots, \alpha_b, \theta_f), \qquad
Q = \operatorname{blockdiag}(K, \dots, K, 0_{n_f \times n_f}) .
$$

Construct the operator from the upper Cholesky factor of \(K\):

```python
import jax.numpy as jnp
from nlls_gram import AnnealRidge, LevenbergMarquardt, RepeatedFactorRidge

L = jnp.linalg.cholesky(K, upper=True)
ridge_metric = RepeatedFactorRidge(L, repeats=b)
solver = LevenbergMarquardt(residual_fn, ridge=1e-4, ridge_metric=ridge_metric)
```

Trailing coordinates beyond `b * K.shape[0]` automatically carry zero objective penalty while still receiving Euclidean damping \(\mu I_{n_f}\) when \(\mu > 0\).

## Generalized ridge Gram and bordered free-tail elimination

When \(m < n\) and \(Q\) is block-invertible on the penalized coordinates, both `Cholesky(form="gram")` and `QR(form="gram")` reduce the \(n \times n\) step system to an \(m \times m\) dual system without dividing unpenalized coordinates by \(\mu\).

Partition \(\theta = (\theta_p, \theta_f)\) into \(n_p\) strictly penalized coordinates (\(Q_{pp} \succ 0\)) and \(n_f\) unpenalized coordinates (\(Q_{ff} = 0\)), with \(J = [J_p \mid J_f]\) and \(D_p = \rho Q_{pp} + \mu I_{n_p} \succ 0\):

1. Eliminate \(d_p\) via \(D_p\) to form the \(m \times m\) SPD penalized Schur matrix
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

`QR(form="gram")` computes the exact same reduction through thin QR factorizations of \(\begin{bmatrix} D_p^{-1/2} J_p^\top \\ I_m \end{bmatrix}\) and \(\begin{bmatrix} R_p^{-\top} J_f \\ \sqrt{\mu}\, I_{n_f} \end{bmatrix}\), avoiding squaring \(\operatorname{cond}(J)\) even at tiny \(\rho\) and \(\mu = 0\) (whenever \(n_f \le m\) and \(\operatorname{rank}(J_f) = n_f\)).

## Ridge continuation and stopping

`AnnealRidge` geometrically lowers `state.ridge` toward `ridge_floor` whenever the Euclidean stationarity norm `info.grad_norm` drops below `grad_rtol` times the entry gradient of the current ridge level:

```python
anneal = AnnealRidge(ridge_floor=1e-8, decrease=0.1, grad_rtol=1e-2)
result = solver.solve(
    x0,
    callback=anneal,
    user_state=anneal.init_state(),
    gtol=1e-8,
    max_steps=400,
)
```

Stopping checks operate directly on the Euclidean objective \(F_\rho(\theta)\):

- `gtol` tests \(\|J^\top r + \rho Q \theta\|_2 < \texttt{gtol}\).
- `xtol` tests \(\|d\|_2 < \texttt{xtol}\) on accepted steps.
- `atol` tests \(\|r\|_2 < \texttt{atol}\) on unpenalized problems (`ridge=None`) when enabled (`atol > 0`). When `ridge > 0`, default `atol=0` tests stationarity and step length alone (`gtol` and `xtol`) so regularized problems with \(\|r(\theta^*)\|_2 > 0\) converge cleanly, while an explicit `atol > 0` acts as a conjunctive residual ceiling \(\|r\|_2 \le \texttt{atol}\) (for example to rule out intermediate `AnnealRidge` levels) and cannot trigger convergence without `gtol` or `xtol`.
- Whenever a callback modifies `state.ridge` or `state.ridge_metric`, convergence is suppressed on that iteration and cached factorizations keyed to the previous penalty are invalidated.
