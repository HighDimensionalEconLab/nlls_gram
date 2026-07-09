# Metric and Preconditioner Utilities

The library ships a small set of constructors and helpers so that models can
assemble metrics and CG preconditioners from structure they already know,
instead of hand-rolling callback plumbing. Everything here returns plain
callables or `Metric` objects; nothing is required — the solver only sees the
`metric=` and `dual_preconditioner=` arguments.

| Helper | Builds | Cost per apply |
| --- | --- | --- |
| `metric_from_cholesky(L)` | dense `Metric` from \(M = LL^\top\) | \(O(n^2)\) |
| `metric_from_tridiagonal_precision(diag, off_diag)` | `Metric` from a tridiagonal \(T = M^{-1}\) | \(O(n)\) |
| `metric_from_diagonal(weights)` | `Metric` from \(M = \operatorname{diag}(w)\) | \(O(n)\) |
| `blockdiag_metric(blocks)` | `Metric` over concatenated parameter blocks | sum of blocks |
| `sherman_morrison_preconditioner(solve, u, weight)` | `dual_preconditioner` for \(P = A + w\,uu^\top\) | one `solve` |

## Tridiagonal Precision Metric

For Markov kernels the Gram inverse is exactly tridiagonal — for the
Matérn-1/2 / Ornstein-Uhlenbeck kernel on sorted points \(t_1 < \dots < t_n\)
with \(\rho_i = e^{-(t_{i+1}-t_i)/\ell}\), the precision has closed-form
entries. Passing the two diagonals gives a `Metric` whose every callback is
\(O(n)\), with nothing factored densely:

```python
from nlls_gram import metric_from_tridiagonal_precision

metric = metric_from_tridiagonal_precision(diag, off_diag)
```

`parallel=None` (the default) runs the one-time bidiagonal Cholesky setup as
an associative \(O(\log n)\)-depth scan off-CPU — where a sequential scan
pays a kernel launch per step — and as the (faster) sequential scan on CPU.

## Diagonal and Block-Diagonal Metrics

Models with a kernel block plus a few scalar parameters can compose
per-block metrics instead of writing slice/concatenate glue. The blocks are
laid out in the order the solver flattens the parameter pytree
(`ravel_pytree` order):

```python
import jax.numpy as jnp

from nlls_gram import blockdiag_metric, metric_from_cholesky, metric_from_diagonal

metric = blockdiag_metric(
    [
        (metric_from_cholesky(jnp.linalg.cholesky(K)), n),
        (metric_from_diagonal(jnp.full(1, m_0)), 1),
    ]
)
```

`solve`, `inv_sqrt`, and `inv_sqrt_transpose` slice on the leading axis, so
vector and matrix inputs both work; `norm` combines the block norms in
quadrature. A fully-default `Metric()` block means the identity metric on
that block. A block that defines some callbacks but leaves others `None`
propagates the missing callbacks as `None` on the composite, so the solver's
construction-time validation applies exactly as it would to that block
alone.

## Sherman–Morrison Dual Preconditioner

With `linear_solver="cg"`, the `dual_preconditioner(v, damping)` argument
supplies an approximation of \((J M^{-1} J^\top + \lambda I)^{-1} v\) on
residual-space vectors. It never changes the subproblem being solved: at
inner convergence the step is identical, and a budget-truncated step still
lies in \(\operatorname{range}(M^{-1}J^\top)\), so the minimum-metric-norm
selection for underdetermined residuals is unchanged and an approximate
preconditioner is safe — even though `metric.solve` must stay exact.

A metric weight \(m\) on a scalar parameter injects an exactly known rank-1
spike into the dual operator. For the kernel-collocation family, a Jacobian
column \(-c\,u\) for that parameter contributes \((c^2/m)\,uu^\top\), and

```python
from nlls_gram import sherman_morrison_preconditioner

dual_preconditioner = sherman_morrison_preconditioner(
    alpha_metric.solve, jnp.ones(n), c**2 / m_0
)
```

builds \(P^{-1}\) for \(P = K + (c^2/m_0)\,\mathbf{1}\mathbf{1}^\top\) from
one kernel solve plus a rank-1 correction. Such structural preconditioners
can be spectrally equivalent to the dual operator uniformly in \(n\),
keeping the inner CG budget constant where the unpreconditioned budget grows
with refinement — see the [Tuning Guide](tuning_guide.md).

## API

::: nlls_gram.metric_from_tridiagonal_precision

::: nlls_gram.metric_from_diagonal

::: nlls_gram.blockdiag_metric

::: nlls_gram.sherman_morrison_preconditioner
