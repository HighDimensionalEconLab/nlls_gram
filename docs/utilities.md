# Metric and Preconditioner Utilities

The library ships a small set of constructors and helpers so that models can
assemble metrics and CG preconditioners from structure they already know,
instead of hand-rolling callback plumbing. Everything here returns plain
callables or `Metric` objects that the solver sees through the `metric=`,
`dual_preconditioner=`, and `implicit_preconditioner=` arguments. The CG
paths require explicit preconditioners; `identity_preconditioner()` is the
explicit opt-out.

| Helper | Builds | Cost per apply |
| --- | --- | --- |
| `metric_from_cholesky(L)` | dense `Metric` from \(M = LL^\top\) | \(O(n^2)\) |
| `metric_from_tridiagonal_precision(diag, off_diag)` | `Metric` from a tridiagonal \(T = M^{-1}\) | \(O(n)\) |
| `metric_from_state_space(points, h, Pinf, transition)` | `Metric` for a stationary state-space kernel Gram \(M = K + \eta I\) (Matérn via `matern_state_space`) | \(O(n m^2)\) |
| `metric_from_quasiseparable(d, p, q, A)` | `Metric` from quasiseparable generators | \(O(n m^2)\) |
| `metric_from_shifted_matvec(matvec, shift)` | matrix-free `Metric` for \(M = A + \varepsilon I\) via inner CG | \(O(\text{iters} \times \text{matvec})\) |
| `metric_from_diagonal(weights)` | `Metric` from \(M = \operatorname{diag}(w)\) | \(O(n)\) |
| `blockdiag_metric(blocks)` | `Metric` over concatenated parameter blocks | sum of blocks |
| `sherman_morrison_preconditioner(solve, u, weight)` | `dual_preconditioner` for \(B = A + w\,uu^\top\) | one `solve` |
| `woodbury_preconditioner(solve, U, weights)` | `dual_preconditioner` for \(B = A + U\operatorname{diag}(w)U^\top\) | one `solve` + \(k \times k\) |
| `identity_preconditioner()` | the explicit "no preconditioner" choice (both hook signatures) | free |

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
an associative \(O(\log n)\)-depth scan off-CPU in float64 — where a
sequential scan pays a kernel launch per step — and as the sequential scan
otherwise. In float32 the default stays sequential even off-CPU: the
parallel scan's projective \(2\times 2\) products can cancel to non-finite
values on long, stiff grids (near-unit-correlation AR(1)), while the
sequential recurrence is stable there.

## State-Space Kernel Metrics (Quasiseparable)

A stationary Gaussian process has an exact O(n) Gram factorization
precisely when it admits a finite-dimensional **state-space** (linear SDE)
representation: an \(m\)-dimensional latent Gauss-Markov state observed
through a row vector \(h\). With stationary state covariance
\(P_\infty\) and transition matrices \(A_k = \Phi(t_k - t_{k-1})^\top\)
(transposed matrix exponential of the SDE drift, the tinygp orientation),
the Gram on sorted points is

\[
K_{ij} = h^\top P_\infty A_i A_{i-1} \cdots A_{j+1}\, h \quad (i > j),
\]

\(h^\top P_\infty h\) on the diagonal, symmetric — a **quasiseparable**
(celerite-style, rank-\(m\) semiseparable) matrix whose Cholesky factor
shares the same structure, so every callback is one or two O(\(n m^2\))
scans.

The main application is the half-integer Matérn family, which is exactly
the CAR(\(m\)) state-space class with \(m = 1, 2, 3\) for
\(\nu = 1/2, 3/2, 5/2\); `matern_state_space(sigma, ell, nu)` supplies the
exact \((h, P_\infty, \Phi^\top)\) mapping (with \(f = \sqrt{2\nu}/\ell\):
\(h = [\sigma, 0, \ldots]\), e.g. \(P_\infty = \operatorname{diag}(1, f^2)\)
for \(\nu = 3/2\)). This route is necessary for exactness: only the
Matérn-1/2 value Gram has a sparse inverse. For 3/2 and 5/2 the sampled
value process is ARMA-like, not Markov — the Gram-inverse off-band entries
are \(\sim 10^{-2}\) *relative* (versus \(\sim 10^{-16}\) for 1/2), so a
truncated band would be an approximate metric, and the library's contract
requires `metric.solve` to be exact (approximations belong in
`dual_preconditioner`). Only the latent state is Markov; the state-space
form exploits exactly that.

```python
from nlls_gram import matern_state_space, metric_from_state_space

metric = metric_from_state_space(
    points, *matern_state_space(sigma, ell, nu=1.5), nugget=1e-8 * sigma**2
)
```

Other stationary state-space kernels (sums of exponentials, CARMA /
celerite-style terms) drop into the same constructor through their own
\((h, P_\infty, \Phi^\top)\).

`points` must be 1-D and sorted strictly increasing — **not validated**,
since it may be traced; unsorted or repeated points silently produce a
wrong or NaN metric. The metric is \(M = K + \eta I\): the absolute nugget
\(\eta\) folds into the diagonal generator before factorization, so it is
part of the metric — exact, not a solver fudge. Nugget-free Matérn-3/2 and
5/2 Grams on fine grids are extremely ill-conditioned (condition number
\(\sim 10^{21}\) at \(n = 5000\) — a property of the matrix, not the
solver); supply a nugget whenever the grid resolves the kernel. For
\(\nu = 1/2\) the constructor works but
`metric_from_tridiagonal_precision` is the specialized alternative whose
applies are elementwise shifts — strictly cheaper than scans on GPU.

`metric_from_quasiseparable(d, p, q, A, nugget=0.0, parallel=None)` is the
generator-level general API — any stationary state-space kernel (sums of
exponentials, celerite terms) reduces to it, and banded matrices are
themselves rank-\(p\) quasiseparable. \(A_k\) is the transition INTO index
\(k\) (\(A_0\) never enters the products; the state-space builders set it
to the identity). Positive definiteness is not validated (inputs may be
traced): a non-PD input silently produces NaN through the Cholesky square
roots, the same convention as the tridiagonal constructor.

`parallel=None` (the default) picks the apply implementation once at
construction, from the backend and dtype there: associative
O(\(\log n\))-depth scans off-CPU in float64 — where a sequential scan pays
a kernel launch per step — and sequential scans otherwise. Unlike the
tridiagonal constructor's setup scan, the parallel *substitutions* here
propagate rank-1-corrected transition matrices \(A_k - w_k p_k^\top / c_k\)
with no contraction guarantee, so the float32 default stays sequential on
every backend; the float64 default is backed by the stress-grid agreement
tests. Pass `parallel=True`/`False` to force either path. The one-time
Cholesky setup is a sequential scan in this release — cheap for fixed
metrics reused across solves, but on the hot path when the metric is
rebuilt from traced \(\sigma, \ell\) inside `jax.grad`/`vmap` sweeps; see
the [Tuning Guide](tuning_guide.md).

## Unified Shifted Block Metrics

For a kernel-coefficient block \(\alpha\) plus scalar parameters \(\beta\),
the unified shifted metric \(M = \operatorname{blockdiag}(K, 0) +
\varepsilon I = \operatorname{blockdiag}(K + \varepsilon I_n,
\varepsilon I_k)\) completes the RKHS seminorm \(\alpha^\top K \alpha\)
with a single spectral floor — the theory (spectrum, the
\(\varepsilon \to 0\) seminorm limit and its uniqueness conditions) is in
[Metric Gauss-Newton](gauss_newton.md#shifted-metrics-and-the-seminorm-limit).
It composes from existing constructors; the kernel block has three
interchangeable representations:

```python
scalar_block = metric_from_diagonal(eps * jnp.ones(k))
metric = blockdiag_metric([(kernel_block, n), (scalar_block, k)])
# kernel_block, by representation of K:
#   metric_from_cholesky(jnp.linalg.cholesky(K + eps * jnp.eye(n)))   # dense
#   metric_from_state_space(t, *matern_state_space(sigma, ell, nu), nugget=eps)
#   metric_from_shifted_matvec(kernel_matvec, eps)      # matvec only; cholesky/cg
```

One structural note: the Matérn-1/2 tridiagonal shortcut does not survive
the shift — `metric_from_tridiagonal_precision` parameterizes
\(T = M^{-1}\), and \((K + \varepsilon I)^{-1}\) is **not** tridiagonal —
so a shifted OU metric goes through
`metric_from_state_space(..., nu=0.5, nugget=eps)` instead.

`metric_from_shifted_matvec(matvec, shift, *, tol=None, atol=0.0,
maxiter=None, preconditioner=None)` needs only a matvec of a symmetric PSD
\(A\), accepting `(n,)` and `(n, k)` leading-axis inputs (the same shape
contract `Metric.solve` carries). The positive `shift` is what makes an
iterative metric solve viable: \(\kappa(A + \varepsilon I) \le
(\lambda_{\max} + \varepsilon)/\varepsilon\) regardless of how singular
\(A\) is — and in practice far better, since the shift *clusters* the
spectral tail at \(\approx \varepsilon\) and CG resolves a cluster in
about one iteration (measured: ~32 float64 iterations for a Matérn-5/2
Gram at n=1000, independent of \(\varepsilon\) from 1e-2 to 1e-8). It
provides `solve` and `norm` only (no matrix-free square root), so it works
with the `cholesky` and `cg` linear solvers and is rejected for
`qr`/`lsmr` at construction. Combined with
`implicit_solver="cg"` the whole pipeline — forward solve, JVP, and VJP —
runs matrix-free (see [Implicit AD](implicit_ad.md)). This is the one
constructor that meets the
`metric.solve` exactness contract in a limit rather than identically: its
inner CG tolerance is part of the answer, not of the schedule — the
residual error perturbs the selected solution and the implicit derivatives
at order `tol`, and the implicit derivative has no accept/reject
safeguard. The default tolerance (square root of the dtype's machine
epsilon) matches typical outer tolerances; never cap `maxiter` as a cost
control (a truncated CG is not a linear map, which breaks the `cg`
linear_solver's operator assumptions).

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

builds \(B^{-1}\) for \(B = K + (c^2/m_0)\,\mathbf{1}\mathbf{1}^\top\) from
one kernel solve plus a rank-1 correction (\(B\), not \(P\): the docs reserve
\(P\) for \(M^{-1}\)). Under the unified shifted metric the scalar-block
weight is \(\varepsilon\), so the spike weight is \(c^2/\varepsilon\) — it
grows as \(\varepsilon\) shrinks, making the preconditioner more
load-bearing, not less. For \(k\) scalar parameters at once,
`woodbury_preconditioner(solve, U, weights)` is the rank-\(k\)
generalization (\(B = A + U\operatorname{diag}(w)U^\top\), one matrix
solve plus a \(k \times k\) Cholesky, reducing exactly to Sherman-Morrison
at \(k = 1\)). Such structural preconditioners
can be spectrally equivalent to the dual operator uniformly in \(n\),
keeping the inner CG budget constant where the unpreconditioned budget grows
with refinement — see the [Tuning Guide](tuning_guide.md).

## Identity Preconditioner

`linear_solver="cg"` requires a `dual_preconditioner`, and a cg-resolved
implicit solve requires an `implicit_preconditioner` — running Krylov
methods unpreconditioned should be a decision, not a default.
`identity_preconditioner()` is that decision made explicit and greppable:

```python
from nlls_gram import identity_preconditioner

solver = UnderdeterminedLevenbergMarquardt(
    residual_fn,
    linear_solver="cg",
    dual_preconditioner=identity_preconditioner(),
    implicit_preconditioner=identity_preconditioner(),
)
```

The returned callable accepts both hook signatures —
`dual_preconditioner(v, damping)` and `implicit_preconditioner(v)` — so one
helper serves both arguments.

## API

::: nlls_gram.metric_from_tridiagonal_precision

::: nlls_gram.metric_from_state_space

::: nlls_gram.matern_state_space

::: nlls_gram.metric_from_quasiseparable

::: nlls_gram.metric_from_shifted_matvec

::: nlls_gram.metric_from_diagonal

::: nlls_gram.blockdiag_metric

::: nlls_gram.sherman_morrison_preconditioner

::: nlls_gram.woodbury_preconditioner

::: nlls_gram.identity_preconditioner
