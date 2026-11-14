# Tuning and Capability Matrix

## Solver selection

- Use `LevenbergMarquardt(..., ridge=<positive>, ridge_metric=...)` or `Newton(..., ridge=<positive>, ridge_metric=...)` when selecting a regularized or minimum-seminorm interpolant (such as an RKHS collocation solution).
- Use `LevenbergMarquardt(residual_fn)` (`ridge=None`) when solving curved or ill-conditioned nonlinear least-squares problems that benefit from adaptive Euclidean trust-region damping.
- Use `Newton(residual_fn)` (`ridge=None`) when residual-only Armijo line searches with stateless residual-coupled damping (`damping=1e-3` by default, or `damping=0.0` for undamped Gauss-Newton) are faster than trust-region re-factorizations.

## Linear solver capability matrix

Let \(m\) be the total optimization residual count, \(n\) be the parameter count, \(n_p\) be the penalized coordinate count, and \(n_f = n - n_p\) be the unpenalized coordinate count.

| Linear solver | `ridge=None`, \(\mu > 0\) | `ridge=None`, \(\mu = 0\) | `ridge > 0`, \(\mu > 0\) | `ridge > 0`, \(\mu = 0\) | Rank-deficient at \(\mu = 0\) |
|---|---|---|---|---|---|
| `Cholesky(form="normal")` | Yes | Full column rank (\(n \le m\)) | `has_normal_add` | \(\ker(J) \cap \ker(L) = \{0\}\) | No |
| `Cholesky(form="gram")` | Yes | Full row rank (\(m \le n\)) | `has_shifted_solve` | \(Q_{pp} \succ 0\), \(n_f \le m\), full rank | No |
| `QR(form="normal")` | Yes | Full column rank (\(n \le m\)) | Factored `RidgeMetric` | Factored `RidgeMetric`, \(m + p \ge n\), full rank | No |
| `QR(form="gram")` | Yes | Full row rank (\(m \le n\)) | `has_shifted_solve` (no `DiagonalPreconditioner`) | \(Q_{pp} \succ 0\), \(n_f \le m\), full rank | No |
| `CG(preconditioner)` | Yes | Full column rank or range-safe | Any `RidgeMetric` (including matvec-only) | \(\ker(J) \cap \ker(L) = \{0\}\) | Range-safe only |
| `GramCG(preconditioner)` | Yes | Full row rank (\(m \le n\)) | Rejected | Rejected | Range-safe only |
| `LU()` | Rejected | Square \(m = n\), nonsingular | Rejected | Rejected | No |
| `SVD(rcond=None)` | Yes | Any \(m, n\) (minimum-norm) | Factored `RidgeMetric` | Factored `RidgeMetric` (minimum-norm) | Yes |

## Dense and Krylov preconditioning

Preconditioners never alter the Euclidean step equation \(H_\mu d = -g\) where \(H_\mu = J^\top J + \rho Q + \mu I_n\) and \(g = J^\top r + \rho Q \theta\).

- **Dense equilibration**: pass `DiagonalPreconditioner()` (`diagonal=None`) to `Cholesky` or `QR(form="normal")` (or unpenalized `QR(form="gram")`) when parameter columns or residual rows span many orders of magnitude.
- **Custom parameter-space preconditioners for `CG`**: approximate \(H_\mu^{-1} v\) directly in unwhitened parameter coordinates. When porting a domain preconditioner (such as a block-arrow Woodbury solver over kernel coefficients \(\alpha\) with Gram matrix \(K\), integrated operator \(\tilde{K}\), and unpenalized initial values \(\theta_f\)), assemble the diagonal and border blocks from the unwhitened Basis matrices \((K, \tilde{K})\) rather than Cholesky-whitened factors \((K F^{-1}, \tilde{K} F^{-1})\), and shift each penalized block by \(\rho K + \mu I\) and each unpenalized block by \(\mu I\). Symmetric Jacobi equilibration \(D^{-1/2} B D^{-1/2}\) inside each local block Cholesky factorization keeps unwhitened block solves well-conditioned down to very small \(\rho\) and \(\mu\).

## Traced versus static settings

- **Traced (no recompile across calls)**: `atol`, `gtol`, `xtol`, `max_steps` (when `save_steps=False`), `p`, `args`, `ridge`, `damping`, `LMHyperparams`, `NewtonHyperparams`, and all array leaves inside `RidgeMetric` and `Preconditioner`.
- **Static (recompiles when changed)**: `residual_fn`, `linear_solver` class/form, `ad_solver` class/form, `jacobian_mode`, `jacobian_chunk`, `has_aux`, `cache_jacobian`, `geodesic_acceleration`, `callback`, shapes, and `meta_fields` on `RidgeMetric` or `Preconditioner`.

## GPU `float32` matmul precision

All internal matrix products in `nlls_gram` use `jax.lax.Precision.HIGHEST` so NVIDIA Ampere+ GPUs do not drop internal normal or Gram contractions to 10-bit TF32 mantissas. To keep `float32` matmuls inside your own `residual_fn` at full IEEE `float32` precision on CUDA GPUs as well, set:

```python
import jax

jax.config.update("jax_default_matmul_precision", "highest")
```
