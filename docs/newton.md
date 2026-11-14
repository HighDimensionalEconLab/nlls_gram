# Newton with Armijo Backtracking

`Newton` solves nonlinear equations, nonlinear least-squares problems, and Tikhonov-regularized interpolation problems by computing a Euclidean Gauss-Newton (or regularized Gauss-Newton) step once per outer iteration and globalizing with an Armijo backtracking line search on \(F_\rho(\theta) = \frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta\).

## Step equations and Armijo condition

At iterate \(\theta\) with complete residual \(r \in \mathbb{R}^m\), Jacobian \(J = D r(\theta) \in \mathbb{R}^{m \times n}\), optional `ridge` \(\rho \ge 0\), and scalar step damping \(\mu \ge 0\), the half-gradient is

$$
g = J^\top r + \rho Q \theta ,
$$

and the step \(d\) solves

$$
\left(J^\top J + \rho Q + \mu I_n\right) d = -g .
$$

When `ridge=None` and `damping=0.0`, `Newton` requests the Euclidean minimum-norm least-squares step \(d = -J^+ r\). Rank-deficient problems at \(\mu = 0\) are supported with `linear_solver=SVD()`.

With directional slope \(s = g^\top d < 0\) and package loss \(L_\rho(\theta) = 2 F_\rho(\theta)\), a trial step size \(\alpha \in (0, 1]\) along the ray \(\theta + \alpha d\) (or along the geodesic arc \(\theta + \alpha d + \frac{1}{2}\alpha^2 a\) when `geodesic_acceleration=True` and \(2\|a\|_2 / \|d\|_2 \le \texttt{geodesic\_acceptance\_ratio}\)) is accepted when

$$
L_\rho(\theta(\alpha)) \le L_\rho(\theta) + 2 c_1 \alpha s + 2\,\texttt{linesearch\_tol} .
$$

## Residual-coupled scalar damping

When `damping` is `None` (the default), `Newton` sets `damping = 1e-3` when `ridge=None` (or `0.0` for `LU()` and when `ridge` is positive). For any positive `damping` coefficient \(c_d > 0\), `Newton` uses the stateless Euclidean residual-coupled scalar damping

$$
\mu_k = \max\!\left(\texttt{min\_damping},\; c_d \cdot \frac{\|J(\theta_k)\|_F^2}{m} \cdot \|r(\theta_k)\|_2^{\texttt{damping\_power}}\right)
$$

(replacing \(\|J\|_F^2 / m\) by \(1\) for matrix-free `CG` and `GramCG`). Pass `damping=0.0` for strictly undamped Gauss-Newton steps.

## Example

```python
import jax
import jax.numpy as jnp
from nlls_gram import Newton, QR, SolverStatus

key = jax.random.key(0)
A = jax.random.normal(key, (8, 24))
b = jnp.ones(8)
x0 = jnp.zeros(24)

solver = Newton(lambda x: A @ x - b, damping=0.0, linear_solver=QR())
result = solver.solve(x0, atol=1e-6, max_steps=10)
assert int(result.status) == int(SolverStatus.CONVERGED)
```

## Linear solver compatibility at `damping=0.0`

| Config | Unpenalized (`ridge=None`) | Penalized (`ridge > 0`) | Rank-deficient safe at \(\mu = 0\) |
|---|---|---|---|
| `Cholesky(form="auto")` | Full-rank (\(m \le n\) Gram, \(n \le m\) normal) | \(\ker(J) \cap \ker(L) = \{0\}\) (Gram when \(Q_{pp} \succ 0\) and \(n_f \le m\)) | No |
| `QR(form="auto")` | Full-rank (\(m \le n\) Gram, \(n \le m\) normal) | Factored `RidgeMetric` with \(m + p \ge n\), or Gram when \(Q_{pp} \succ 0\) and \(n_f \le m\) | No |
| `SVD(rcond=None)` | Any \(m, n\) | Any factored `RidgeMetric` | Yes (Euclidean minimum-norm) |
| `LU()` | Square \(m = n\) only | Rejected | No |
| `GramCG(precond)` | Wide \(m \le n\) | Rejected | Range-preserving preconditioner required |
| `CG(precond)` | Tall \(n \le m\) | \(\ker(J) \cap \ker(L) = \{0\}\) | Range-preserving preconditioner required |

## Failure statuses

`Newton` preserves the pre-step iterate and reports an explicit status rather than silently switching algorithms:

- `SolverStatus.LINEAR_SOLVE_FAILED` when a factorization encounters a singular or nonfinite pivot.
- `SolverStatus.NON_DESCENT` when \(g^\top d \ge 0\) at a non-stationary point.
- `SolverStatus.STAGNATION` when floating-point step addition underflows (`x + step == x`).
- `SolverStatus.LINE_SEARCH_FAILED` when all `maxls` backtracking trials are rejected.
