# Tuning and Capability Matrix

## Solver selection

- Use `LevenbergMarquardt` or `Newton` with positive `ridge` and a `ridge_metric` when minimizing a Tikhonov-regularized objective or continuing toward a minimum-seminorm interpolant with `AnnealRidge`.
- Use `LevenbergMarquardt(residual_fn)` (`ridge=None`) when curved or ill-conditioned nonlinear least-squares problems benefit from adaptive step damping and rejected-step Jacobian reuse.
- Use `Newton(residual_fn)` (`ridge=None`) when stateless residual-coupled damping (`damping=1e-3` by default, or `damping=0.0` for undamped Gauss-Newton) avoids repeated trial factorizations.

## Linear solver capability matrix

Both `LevenbergMarquardt` and `Newton` default to `linear_solver=QR()`. Let \(m\) denote total residual rows, \(n\) denote parameter count, \(n_p\) denote penalized coordinates, \(n_f = n - n_p\) denote unpenalized coordinates, and \(p\) denote rows of `ridge_metric`'s factor \(L\).

| Linear solver | `ridge=None`, \(\mu > 0\) | `ridge=None`, \(\mu = 0\) | `ridge > 0`, \(\mu > 0\) | `ridge > 0`, \(\mu = 0\) | Rank-deficient at \(\mu = 0\) |
|---|---|---|---|---|---|
| `Cholesky(form="normal")` | Yes | Full column rank (\(n \le m\)) | `has_normal_add` | \(\ker(J) \cap \ker(L) = \{0\}\) | No |
| `Cholesky(form="gram")` | Yes | Full row rank (\(m \le n\)) | `has_shifted_solve`, \(Q_{pp} \succ 0\) (and \(n_f = 0\) if `DiagonalPreconditioner`) | \(Q_{pp} \succ 0\), \(n_f \le m\), full rank | No |
| `QR(form="normal")` | Yes | Full column rank (\(n \le m\)) | Factored `RidgeMetric` | Factored `RidgeMetric`, \(m + p \ge n\), full rank | No |
| `QR(form="gram")` | Yes | Full row rank (\(m \le n\)) | `has_shifted_solve`, \(Q_{pp} \succ 0\) (no `DiagonalPreconditioner`) | \(Q_{pp} \succ 0\), \(n_f \le m\), full rank | No |
| `CG(preconditioner)` | Yes | Full column rank (\(n \le m\)) or `penalty != None` | Any `RidgeMetric` (including matvec-only) | \(\ker(J) \cap \ker(L) = \{0\}\) (\(n_f \le m\) or `penalty != None` in AD) | Range-safe only |
| `GramCG(preconditioner)` | Yes | Full row rank (\(m \le n\)) | `has_shifted_solve`, \(Q \succ 0\), \(n_f = 0\) | `has_shifted_solve`, \(Q \succ 0\), \(n_f = 0\) | Range-safe only |
| `LU()` | Rejected | Square \(m = n\), nonsingular | Rejected | Rejected | No |
| `SVD(rcond=None)` | Yes | Any \(m, n\) (minimum-norm) | Factored `RidgeMetric` | Factored `RidgeMetric` (minimum-norm) | Yes |

### Automatic `form="auto"` selection

For `Cholesky(form="auto")` and `QR(form="auto")`:

- With `ridge=None`, `"gram"` is selected when \(n > m\), and `"normal"` otherwise.
- With `ridge > 0`, `"gram"` is selected when \(n > m\), `ridge_metric.fast_shifted_solve` is `True` (`IdentityRidge`, `DiagonalRidge`, or `RepeatedFactorRidge`), `ridge_metric.penalized_is_spd` is `True`, and \(n_f \le m\). Otherwise `"normal"` is selected. Note that `MatrixRidge` and `FactorRidge` always default to `"normal"` under `form="auto"` unless `form="gram"` is requested explicitly.

## Jacobian assembly and caching

- `jacobian_mode="auto"` selects forward-mode basis JVPs (`"fwd"`) when \(n \le m\) and reverse-mode basis VJPs (`"rev"`) when \(n > m\).
- `jacobian_chunk=c` batches those basis directions through `jax.lax.map(..., batch_size=c)` to cap peak memory without altering numerical values.
- `cache_jacobian=True` (default on `LevenbergMarquardt` for dense linear solvers, `False` on `Newton` and forced `False` for `CG`/`GramCG`) is pure rejected-step reuse rather than a quasi-Newton secant update. When an LM step is rejected without a constraint-schedule change, the unchanged `(resid, Jt)` and any valid pre-damping `CholeskyCache` or `QRCache` at the same `ridge` value are reused on the higher-\(\mu\) retry.

## Block preconditioning in parameter coordinates

Preconditioners never alter \(H_\mu d = -g\). Pass `DiagonalPreconditioner()` to `Cholesky` or `QR(form="normal")` (or unpenalized `QR(form="gram")`) when parameter columns or residual rows span several orders of magnitude. For structured block-kernel problems solved with `CG`, build the custom `Preconditioner` directly in parameter coordinates by shifting each penalized kernel block by \(\rho K + \mu I\) and each unpenalized block by \(\mu I\), applying symmetric Jacobi scaling \(D^{-1/2} B D^{-1/2}\) inside local block Cholesky factorizations when \(\rho\) and \(\mu\) are small.

## Traced versus static settings

- Traced across calls without recompilation: `atol`, `gtol`, `xtol`, `max_steps` (when `save_steps=False`), `p`, `args`, `ridge`, `damping`, `LMHyperparams`, `NewtonHyperparams`, and array leaves inside `RidgeMetric` and `Preconditioner` when static metadata such as `size` and `penalized_is_spd` remain unchanged. Construct `RidgeMetric` instances from concrete arrays on the host when `penalized_is_spd=True` is required for Gram routing.
- Static (recompiles when changed): `residual_fn`, `linear_solver` class/form, `ad_solver` class/form, `jacobian_mode`, `jacobian_chunk`, `has_aux`, `cache_jacobian`, `geodesic_acceleration`, `callback`, array shapes, and `meta_fields` on `RidgeMetric` or `Preconditioner`.

## GPU `float32` matmul precision

All internal matrix products in `nlls_gram` use `jax.lax.Precision.HIGHEST` so NVIDIA Ampere+ GPUs do not drop normal or Gram contractions to 10-bit TF32 mantissas. To keep `float32` matmuls inside `residual_fn` at full IEEE `float32` precision on CUDA GPUs as well, set:

```python
import jax

jax.config.update("jax_default_matmul_precision", "highest")
```
