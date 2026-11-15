# Gauss-Newton with Armijo Backtracking

`Newton` minimizes the common objective \(F_\rho(\theta) = \frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta\) by solving one damped or undamped Gauss-Newton system \((J^\top J + \rho Q + \mu_k I_n) d = -(J^\top r + \rho Q \theta)\) per outer iteration and globalizing with the same Armijo backtracking line search and optional geodesic acceleration described in [Levenberg-Marquardt and Tikhonov](ridge_lm.md).

When `ridge=None` and \(r(\theta_k) = 0\) exactly, `Newton` skips linearization and immediately reports `SolverStatus.CONVERGED`. When `ridge=None` and `damping=0.0`, `Newton` computes the undamped Gauss-Newton step \(d = -J^+ r\) (using `SVD()` when rank-deficient).

## Residual-coupled scalar damping

| Constructor `damping` | Effective \(c_d\) |
|---|---|
| `None` with `ridge=None` and `linear_solver != LU()` | `1e-3` |
| `None` with `ridge > 0` or `linear_solver=LU()` | `0.0` |
| Explicit float `c_d >= 0.0` | `c_d` |

When \(c_d = 0\), \(\mu_k = 0\). For any positive coefficient \(c_d > 0\), `Newton` uses stateless residual-coupled damping:

$$
s_J =
\begin{cases}
\|J(\theta_k)\|_F^2 / m, & \text{dense solvers}, \\
1, & \text{matrix-free } \texttt{CG} \text{ or } \texttt{GramCG},
\end{cases}
$$

$$
\mu_k = \max\!\left(
\texttt{min\_damping},\;
c_d \, s_J \, \|r(\theta_k)\|_2^{\texttt{damping\_power}},\;
\mu_{\text{chol}}
\right) ,
$$

where \(\mu_{\text{chol}} = 4 \varepsilon_{\text{mach}} s_J\) when `linear_solver` is `Cholesky` and \(0\) otherwise. Defaults are `damping_power=1.5` and `min_damping=1e-12`.

## Example

```python
import jax
import jax.numpy as jnp
from nlls_gram import Newton, QR, SolverStatus

key = jax.random.key(0)
A = jax.random.normal(key, (8, 24))
b = jnp.ones(8)

solver = Newton(lambda x: A @ x - b, damping=0.0, linear_solver=QR())
result = solver.solve(jnp.zeros(24), atol=1e-6, max_steps=10)
assert int(result.status) == int(SolverStatus.CONVERGED)
```

See [Tuning and Capabilities](tuning_guide.md) for linear-solver compatibility at \(\mu = 0\), and [Callbacks and Stopping](callbacks.md) for stopping criteria and `Newton`-specific failure statuses (`LINEAR_SOLVE_FAILED`, `NON_DESCENT`, `STAGNATION`, and `LINE_SEARCH_FAILED`).

## References

- Yamashita and Fukushima (2001) and Fan and Yuan (2005) for residual-power Levenberg-Marquardt regularization \(\mu_k \propto \|r(\theta_k)\|_2^p\).
- Armijo (1966) and Nocedal and Wright (2006) for Gauss-Newton line-search globalization. Full citations appear in [References](references.md).
