# Ridge Operators and Preconditioners

`nlls_gram` separates objective regularization from numerical preconditioning into two independent interfaces:

- **`RidgeMetric`** defines the PSD quadratic penalty \(Q \succeq 0\) in the objective \(F_\rho(\theta) = \frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta\) when `ridge` is positive. It never alters step geometry when `ridge=None`.
- **`Preconditioner`** accelerates linear solves inside `linear_solver` and `ad_solver` without changing the mathematical objective, gradient, or converged step.

Both are registered as JAX pytrees via `register_pytree_dataclass` so their array leaves trace cleanly through `jit`, `vmap`, and callbacks without recompilation.

## `RidgeMetric` implementations

Every `RidgeMetric` implements `matvec(v, ctx)` (\(Q v\)) and `quad_form(v, ctx)` (\(v^\top Q v\), defaulting to `v @ matvec(v, ctx)`). Factored representations additionally expose square-root factor operations (\(Q = L^\top L\)) for `QR(form="normal")` and `SVD()`, and structured representations expose shifted block solves \((\rho Q_{pp} + \mu I_{n_p})^{-1}\) for `Cholesky(form="gram")` and `QR(form="gram")`. Never pass a Gram matrix \(Q\) where a square-root factor \(L\) is expected, or vice versa.

| Class | Constructor input | Penalty \(Q\) | Supported forward / AD solvers |
|---|---|---|---|
| `IdentityRidge(size=None)` | optional leading penalized size | \(\operatorname{blockdiag}(I_{\text{size}}, 0)\) | `Cholesky` (normal/gram), `QR` (normal/gram), `CG`, `SVD` |
| `DiagonalRidge(weights)` | 1D array \(w \ge 0\) (trailing or interior zeros allowed) | \(\operatorname{diag}(w, 0)\) | `Cholesky` (normal/gram), `QR` (normal/gram), `CG`, `SVD` |
| `MatrixRidge(Q)` | symmetric PSD matrix \(Q \in \mathbb{R}^{n_m \times n_m}\) | \(\operatorname{blockdiag}(Q, 0)\) | `Cholesky` (normal, and gram when \(Q \succ 0\)), `QR` (normal, and gram when \(Q \succ 0\)), `CG`, `SVD` |
| `FactorRidge(L)` | rectangular or triangular factor \(L \in \mathbb{R}^{p \times n_m}\) | \(\operatorname{blockdiag}(L^\top L, 0)\) | `Cholesky` (normal, and gram when \(\operatorname{rank}(L)=n_m\)), `QR` (normal, and gram when \(\operatorname{rank}(L)=n_m\)), `CG`, `SVD` |
| `RepeatedFactorRidge(L, repeats=b)` | block factor \(L \in \mathbb{R}^{p_k \times b_k}\) | \(\operatorname{blockdiag}(L^\top L, \dots, L^\top L, 0)\) | `Cholesky` (normal, and gram when \(\operatorname{rank}(L)=b_k\)), `QR` (normal, and gram when \(\operatorname{rank}(L)=b_k\)), `CG`, `SVD` |
| Custom `RidgeMetric` (`matvec` only) | user pytree dataclass | \(v \mapsto Q v\) | `CG` |

`MatrixRidge(Q)` validates symmetry and positive semidefiniteness up to roundoff tolerance via `jnp.linalg.eigh`, clamping tiny negative roundoff eigenvalues to zero when forming its internal PSD square root and recording whether \(Q\) is strictly positive definite (`penalized_is_spd`) without adding artificial diagonal jitter to \(Q\).

`RepeatedFactorRidge(L, repeats=b)` packs all `b` blocks and multiple right-hand sides into single batched matrix multiplications or Cholesky solves, avoiding materializing the \(b \cdot b_k \times b \cdot b_k\) block-diagonal matrix.

## `Preconditioner` implementations

Preconditioners are attached to `Cholesky`, `QR`, `CG`, or `GramCG`.

```python
class Preconditioner:
    requires_positive_damping: bool = False

    def apply(self, v, damping, ctx): ...
```

`ctx` is a `SolverContext` carrying `x`, `state`, `args`, `p`, `space` (`"parameter"` or `"residual"`), `role` (`"forward"` or `"ad"`), and `form` (`"normal"`, `"gram"`, `"lu"`, or `"svd"`).

| Class | Behavior | Supported linear solvers |
|---|---|---|
| `IdentityPreconditioner()` | Identity map \(v \mapsto v\) | `Cholesky`, `QR`, `CG`, `GramCG` |
| `DiagonalPreconditioner(diagonal=None)` | Explicit nonnegative diagonal (`1 / (diagonal + damping)`) or automatic Jacobi equilibration (`diagonal=None`) for dense solvers | `Cholesky` (normal/gram), `QR(form="normal")`, unpenalized `QR(form="gram")`, and `CG`/`GramCG` (when `diagonal` is an explicit array) |

### Dense diagonal scaling

When `DiagonalPreconditioner` is passed to a dense solver, it improves numerical conditioning while returning the exact unpreconditioned Euclidean solution:

- **`Cholesky(form="normal" | "gram", preconditioner=DiagonalPreconditioner(...))`**: applies symmetric diagonal congruence \(D^{-1/2} H_\mu D^{-1/2}\) (or \(D^{-1/2} S_\mu D^{-1/2}\) in Gram space) before Cholesky factorization.
- **`QR(form="normal", preconditioner=DiagonalPreconditioner(...))`**: scales columns of the augmented stack via \(T = \operatorname{diag}(t)\) while transforming the damping block consistently to \(\sqrt{\mu}\,T\), preserving \((J^\top J + \rho Q + \mu I_n) d = -g\). At \(\mu = 0\), column scaling preserves the least-squares solution when the augmented stack has full column rank and reports `LINEAR_SOLVE_FAILED` if rank-deficient.
- **`QR(form="gram", preconditioner=DiagonalPreconditioner(...))`**: supported on unpenalized wide systems (`ridge=None`), scaling rows of \(J\) at \(\mu > 0\) or full row rank at \(\mu = 0\). With `ridge > 0`, `QR(form="gram")` rejects `DiagonalPreconditioner` with `ValueError`.

### Custom preconditioners

Register custom preconditioners for `CG` or `GramCG` with `register_pytree_dataclass`. Because outer steps use unwhitened Euclidean coordinates, a parameter-space preconditioner for `CG` approximates \((J^\top J + \rho Q + \mu I_n)^{-1} v\) directly in original coordinates:

```python
from dataclasses import dataclass
import jax
from nlls_gram import Preconditioner, register_pytree_dataclass


@dataclass(frozen=True, eq=False)
class CustomJacobi(Preconditioner):
    jtj_diag: jax.Array
    q_diag: jax.Array

    def apply(self, v, damping, ctx):
        rho = ctx.state.ridge if ctx.state.ridge is not None else 0.0
        return v / (self.jtj_diag + rho * self.q_diag + damping)


register_pytree_dataclass(CustomJacobi, data_fields=("jtj_diag", "q_diag"))
```
