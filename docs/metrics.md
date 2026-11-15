# Ridge Operators and Preconditioners

`nlls_gram` separates objective regularization from linear-solver preconditioning:

- `RidgeMetric` defines the positive semidefinite penalty \(Q \succeq 0\) in \(F_\rho(\theta) = \frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta\) when `ridge > 0`.
- `Preconditioner` rescales linear solves inside `linear_solver` and `ad_solver` without changing the objective, gradient, or exact step equation.

Both are registered as JAX pytrees via `register_pytree_dataclass` so updating array leaves across calls with fixed shapes and metadata does not trigger recompilation.

## `RidgeMetric` implementations

Every `RidgeMetric` implements `matvec(v, ctx)` (\(Q v\)) and `quad_form(v, ctx)` (\(v^\top Q v\), defaulting to `v @ matvec(v, ctx)`). Factored representations additionally provide square-root operations (\(Q = L^\top L\)) for `QR(form="normal")` and `SVD()`, and structured representations provide shifted solves \((\rho Q_{pp} + \mu I_{n_p})^{-1}\) for `Cholesky(form="gram")`, `QR(form="gram")`, and `GramCG`.

| Class | Constructor input | Penalty \(Q\) | Capabilities |
|---|---|---|---|
| `IdentityRidge(size=None)` | optional leading penalized size | \(\operatorname{blockdiag}(I_{\text{size}}, 0)\) | Normal, factor, fast shifted solve |
| `DiagonalRidge(weights)` | 1D array \(w \ge 0\) | \(\operatorname{diag}(w, 0)\) | Normal, factor, fast shifted solve |
| `MatrixRidge(Q)` | symmetric PSD matrix \(Q \in \mathbb{R}^{k \times k}\) | \(\operatorname{blockdiag}(Q, 0)\) | Normal, factor, dense shifted solve |
| `FactorRidge(L)` | factor \(L \in \mathbb{R}^{r \times k}\) | \(\operatorname{blockdiag}(L^\top L, 0)\) | Normal, factor, dense shifted solve |
| `RepeatedFactorRidge(L, repeats=b)` | block factor \(L \in \mathbb{R}^{r_k \times b_k}\) | \(\operatorname{blockdiag}(L^\top L, \dots, L^\top L, 0)\) | Normal, factor, batched fast shifted solve |
| Custom `RidgeMetric` (`matvec` only) | user pytree dataclass | \(v \mapsto Q v\) | Matrix-free `CG` only |

`MatrixRidge(Q)` checks symmetry and positive semidefiniteness up to roundoff tolerance, clamps tiny negative roundoff eigenvalues to zero when forming its square-root factor \(L\), and stores the matching projected operator \(Q = L^\top L\). `RepeatedFactorRidge` evaluates block matrix-vector products and shifted solves in batched form without materializing the full block-diagonal matrix unless a consumer (`QR(form="normal")` or `SVD()`) explicitly requests `factor_matrix`. See [Tuning and Capabilities](tuning_guide.md) for the full solver compatibility matrix.

## `Preconditioner` implementations

Preconditioners attach to `Cholesky`, `QR`, `CG`, or `GramCG` and receive a `SolverContext` carrying `x`, `state`, `args`, `p`, `ridge`, `damping`, `space` (`"parameter"` or `"residual"`), `role` (`"forward"` or `"ad"`), and `form` (`"normal"`, `"gram"`, `"lu"`, or `"svd"`).

| Class | Input | Supported linear solvers |
|---|---|---|
| `IdentityPreconditioner()` | None | `Cholesky`, `QR`, `CG`, `GramCG` |
| `DiagonalPreconditioner(diagonal=None)` | `None` (automatic dense equilibration) or strictly positive 1D array `d > 0` | `Cholesky` (except penalized Gram with \(n_f > 0\)), `QR(form="normal")`, unpenalized `QR(form="gram")`, and `CG`/`GramCG` (explicit `d > 0` only) |

### Diagonal scaling formulas

Let \(\varepsilon_{\text{tiny}} = \texttt{jnp.finfo(dtype).tiny}\). Dense diagonal scaling preserves the exact unpreconditioned step:

| Solver path | Auto scale (`diagonal=None`) | Explicit scale (`diagonal = d > 0`) | Linear transformation |
|---|---|---|---|
| `Cholesky` normal or unpenalized Gram (`H`) | \(c_i = \sqrt{\max(H_{ii}, 0)}\) (unit if \(\le \sqrt{\varepsilon_{\text{tiny}}}\)) | \(c_i = \max(\sqrt{d_i + \mu}, \sqrt{\varepsilon_{\text{tiny}}})\) | Congruence \(C^{-1} H C^{-1}\), \(C = \operatorname{diag}(c)\) |
| `Cholesky` penalized Gram (\(n_f = 0\), \(S_p\)) | \(c_i = \sqrt{\max((S_p)_{ii}, 0)}\) | \(c_i = \max(\sqrt{d_i}, \sqrt{\varepsilon_{\text{tiny}}})\) | Congruence \(C^{-1} S_p C^{-1}\) |
| `QR(form="normal")`, \(A_0 = [J; \sqrt{\rho}L]\) | \(t_j = (\|(A_0)_{:,j}\|_2^2 + \mu)^{-1/2}\) | \(t_j = (d_j + \mu)^{-1/2}\) | Stack \([A_0 T; \sqrt{\mu} T]\), \(d = T z\) |
| `QR(form="gram")`, `ridge=None` | \(s_i = (\|J_{i,:}\|_2^2 + \mu)^{-1/2}\) | \(s_i = (d_i + \mu)^{-1/2}\) | Stack \([J^\top S; \sqrt{\mu} S]\) |
| `CG` / unpenalized `GramCG` | Rejected | \(M^{-1} v_i = v_i / \max(d_i + \text{shift}, \varepsilon_{\text{tiny}})\) | Shift is \(\mu\) (plus `CG.penalty` in AD) |
| Penalized `GramCG` (\(S = I_m + J D_p^{-1} J^\top\)) | Rejected | \(M^{-1} v_i = v_i / \max(d_i, \varepsilon_{\text{tiny}})\) | Preconditions \(S\) directly (zero additive shift) |

`Cholesky(form="gram")` rejects `DiagonalPreconditioner` when `ridge > 0` and \(n_f > 0\). `QR(form="gram")` rejects `DiagonalPreconditioner` whenever `ridge > 0`. `SVD(rcond=None)` uses the default singular-value cutoff \(\sigma_i > \max(m_{\text{aug}}, n)\,\varepsilon_{\text{mach}}\,\sigma_{\max}\).

### Custom preconditioners

Register custom preconditioners for `CG` or `GramCG` with `register_pytree_dataclass`. A parameter-space preconditioner for `CG` approximates \((J^\top J + \rho Q + \mu I_n)^{-1} v\) in parameter coordinates:

```python
from dataclasses import dataclass
import jax
from nlls_gram import Preconditioner, register_pytree_dataclass


@dataclass(frozen=True, eq=False)
class CustomJacobi(Preconditioner):
    jtj_diag: jax.Array
    q_diag: jax.Array

    def apply(self, v, damping, ctx):
        rho = ctx.ridge if ctx.ridge is not None else 0.0
        return v / (self.jtj_diag + rho * self.q_diag + damping)


register_pytree_dataclass(CustomJacobi, data_fields=("jtj_diag", "q_diag"))
```

## References

- van der Sluis (1969), Björck (1996), and Higham (2002) for diagonal equilibration and normal-versus-QR conditioning.
- Hestenes and Stiefel (1952), Craig (1955), Paige and Saunders (1982), and Saunders (1995) for preconditioned CG and dual Gram Krylov methods. Full citations appear in [References](references.md).
