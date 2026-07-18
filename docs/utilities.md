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
| `repeated_blockdiag_metric(block, block_size, repeats, *, additional)` | `Metric` batching `repeats` identical blocks + optional trailing block | one apply per callback (not per copy) |
| `metric_with_compute_dtype(metric, dtype)` | `Metric` computing in `dtype`, restoring the caller's dtype | same as wrapped metric |
| `sherman_morrison_preconditioner(solve, u, weight)` | `dual_preconditioner` for \(B = A + w\,uu^\top\) | one `solve` |
| `woodbury_preconditioner(solve, U, weights)` | `dual_preconditioner` for \(B = A + U\operatorname{diag}(w)U^\top\) | one `solve` + \(k \times k\) |
| `identity_preconditioner()` | the explicit "no preconditioner" choice (both hook signatures) | free |
| `nystrom_preconditioner(matvec, n, rank, key)` | randomized Nyström `dual_preconditioner` for a PSD operator (FTU) | two \((n, \text{rank})\) GEMVs |
| `pad_dual_preconditioner(base, n_real)` | extends a `dual_preconditioner` to a zero-padded residual | base + \(O(k)\) |

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
with the `cholesky` and `cg` linear solvers and is rejected for `qr`,
`augmented_qr`, and `lsmr` (which all require `inv_sqrt`) at construction. Combined with
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

### Repeated Block Metric

When many parameter blocks share the *same* metric — a multi-country model
whose per-country kernel-coefficient block \(K_\varepsilon\) repeats across
countries, plus a small block of finite-dimensional variables — the layout is
\(\operatorname{blockdiag}(K_\varepsilon \times 5,\ \varepsilon I_3)\).
`repeated_blockdiag_metric` batches the identical copies so each callback fires
**once**, not once per copy:

```python
from nlls_gram import (
    metric_from_cholesky,
    metric_from_diagonal,
    repeated_blockdiag_metric,
)

alpha_metric = metric_from_cholesky(jnp.linalg.cholesky(K + eps * jnp.eye(n)))
metric = repeated_blockdiag_metric(
    alpha_metric,
    block_size=n,
    repeats=5,
    additional=(metric_from_diagonal(eps * jnp.ones(3)), 3),
)
```

This equals `blockdiag_metric([(alpha_metric, n)] * 5 + [(scalar_block, 3)])`,
but instead of five separate solves the repeated head (leading size \(5n\)) is
reshaped to \((n, 5k)\) and the base callback runs once — a dense Cholesky
block does two triangular solves total, not two per copy. The total leading
size `repeats * block_size + additional_size` is derived, so a layout mismatch
raises rather than silently consuming the wrong rows. Because it returns a plain
`Metric` honoring the `(n,)`/`(n, k)` contract, it also composes *inside*
`blockdiag_metric` for heterogeneous layouts (a repeated country block next to a
separate aggregate block). Build the metric once at setup scope — like every
constructor here it returns fresh closures, so rebuilding it inside a
`jax.grad`/`vmap` sweep keys a new compilation each time.

### Compute-Dtype Wrapper

`metric_with_compute_dtype(metric, dtype)` wraps a metric so each callback
upcasts its input to `dtype`, applies the wrapped metric, and restores the
caller's dtype on output. This keeps an ill-conditioned factorization or solve
in wide precision (float64) while the solver's residual/parameter dtype and
loop-carried pytrees stay at the problem dtype — the output round-trips to
`x.dtype`, a no-op once the solver has already promoted its duals to `dtype`:

```python
from nlls_gram import metric_from_cholesky, metric_with_compute_dtype

metric = metric_with_compute_dtype(
    metric_from_cholesky(jnp.linalg.cholesky(K)), jnp.float64
)
```

`None` callbacks are preserved, so a wrapped partial `Metric` stays partial and
the solver's construction-time validation is unchanged.

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

solver = LevenbergMarquardt(
    residual_fn,
    linear_solver="cg",
    dual_preconditioner=identity_preconditioner(),
    implicit_preconditioner=identity_preconditioner(),
)
```

The returned callable accepts both hook signatures —
`dual_preconditioner(v, damping)` and `implicit_preconditioner(v)` — so one
helper serves both arguments.

## Nyström Preconditioner for Neural-Network Least Squares

When no structural preconditioner is available — typically neural-network
least squares under the identity metric, where the dual operator is the
\(m \times m\) empirical NTK Gram \(JJ^\top\) with fast spectral decay —
`nystrom_preconditioner(matvec, n, rank, key)` builds the randomized Nyström
preconditioner of
[Frangella, Tropp, and Udell](https://arxiv.org/abs/2110.02820): sketch the
PSD operator with a thin-QR'd Gaussian test matrix (`rank` operator
applications plus one \(O(n\,\text{rank}^2)\) factorization, once at
construction), recover the rank-`rank` approximation
\(\hat A = U\Lambda U^\top\), and apply

$$
v \mapsto U\frac{U^\top v}{\Lambda + \lambda}
+ \frac{v - UU^\top v}{\rho + \lambda},
$$

with \(\rho\) the smallest retained eigenvalue and \(\lambda\) the **live**
LM damping — this is the one shipped helper that uses the `damping`
argument, so a single construction serves every damping value. The
unresolved complement is balanced at \(\rho + \lambda\) rather than
\(\lambda\); that balance carries the FTU condition-number guarantee for
fast-decaying spectra. `matvec` must apply a symmetric PSD operator and
accept `(n, k)` matrices (the `Metric.solve` shape contract). Like every
preconditioner it is frozen at construction, so for a nonlinear residual it
approximates the dual at the linearization point it was built from —
staleness across LM steps is safe (preconditioner error never moves the
converged root), and refresh cadence is a tuning knob.

The NTK matvec assembles matrix-free from the residual at the initial
parameters via `jax.linearize` and `jax.linear_transpose` (this mirrors the
`test_cg_nystrom_mlp_ntk_example` test, which doubles as the runnable
example):

```python
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from nlls_gram import (
    LevenbergMarquardt,
    identity_preconditioner,
    nystrom_preconditioner,
)

# residual: m collocation residuals of a pure-jax MLP, n_params >> m
theta0, unravel = ravel_pytree(x0)
_, jvp_fn = jax.linearize(lambda th: residual(unravel(th)), theta0)
transpose_fn = jax.linear_transpose(jvp_fn, theta0)


def ntk_matvec(V):  # (m, k) -> J (J' V), frozen at x0
    return jax.vmap(
        lambda col: jvp_fn(transpose_fn(col)[0]), in_axes=1, out_axes=1
    )(V)


solver = LevenbergMarquardt(
    residual,
    linear_solver="cg",
    iterative_tol=1e-6,
    iterative_maxiter=20,
    dual_preconditioner=nystrom_preconditioner(
        ntk_matvec, m, rank, jax.random.PRNGKey(0)
    ),
    implicit_preconditioner=identity_preconditioner(),
)
```

Passed as `implicit_preconditioner` the helper applies its undamped
(zero-damping) inverse, which is valid only when the retained spectrum is
strictly positive.

## Iterate-Adaptive Preconditioner Factory

Every helper above is *frozen*: built once, at one linearization point. That is
safe when the dual operator \(J M^{-1} J^\top + \lambda I\) stays spectrally
close as LM drifts \(x\). When it does not — the Jacobian rotates enough that a
preconditioner built at \(x_0\) decays into an ineffective approximation once
\(x\) moves, and the inner CG stalls or breaks down — pass a
`PreconditionerFactory(prepare, apply)` instead of `dual_preconditioner` (pass
exactly one of the two for `linear_solver="cg"`). Its `prepare(x, args, p)`
rebuilds the preconditioner state from the **current** iterate, inside the
jitted loop as traced ops with no recompiles:

```python
from nlls_gram import LevenbergMarquardt, PreconditionerFactory

def prepare(x, args, p):
    # model-structured build from the CURRENT iterate x (the user pytree,
    # not the raveled theta); return any fixed-shape pytree of arrays
    d = jnp.exp(A @ x)
    return d * d                     # e.g. the exact current dual diagonal

def apply(state, v, damping):
    return v / (state + damping)     # SPD, linear in v

solver = LevenbergMarquardt(
    residual_fn,
    linear_solver="cg",
    preconditioner_factory=PreconditionerFactory(prepare, apply),
    iterative_maxiter=...,
)
```

- `prepare(x, args, p) -> state` receives the **user pytree** `x` (model
  structure intact), the residual `args`, and `p`, and returns a fixed-shape
  pytree of arrays.
- `apply(state, v, damping) -> vector` is the per-iteration apply: an SPD,
  linear-in-`v` approximation of \((J M^{-1} J^\top + \lambda I)^{-1} v\). It
  must stay well-defined at `damping = 0`, because the cg-resolved implicit
  derivative reuses it (undamped) at the converged solution unless an explicit
  `implicit_preconditioner` overrides it.

`prepare` runs **once per accepted step**: after a rejected step \(x\) did not
move, so the carried state is reused and only the live `damping` changes. That
makes the build cost proportional to progress, not to iteration count — but for
an expensive `prepare` it is still one build per accepted step, so keep it
cheap (a diagonal, a small factorization) or fold the heavy work into `args`.
The factory **composes with `recycle`**: deflation runs unchanged on top of the
rebuilt first level (`M_defl(r) = \text{apply}(r) + U E^{-1} U^\top r`). It is
value-hashable on `(prepare, apply)`, so equal pairs share one compiled solve
loop — define them once at setup scope.

## Padded Zero Residuals (Fixed Residual Shape)

Some JAX workflows keep a fixed residual shape across problem instances by
appending residual entries that are identically zero:

```python
def residual_padded(x):
    r = residual(x)
    return jnp.concatenate((r, jnp.zeros(pad, r.dtype)))
```

The padded rows have zero Jacobian rows, so the residual-space dual operator
becomes exactly block diagonal —
\(\operatorname{blockdiag}(J P J^\top + \lambda I,\; \lambda I)\) — and the
solvers behave as follows:

- **`cholesky`** is unchanged mathematically: the padded block decouples
  exactly, and the step matches the unpadded step (regression-tested for
  both the plain and geodesic-accelerated updates). The cost is the larger
  materialized residual dimension and dual factor; a few padded rows are
  harmless, large padding pays the dense residual-space price.
- **`cg`**: a shape-fixed `dual_preconditioner` (a dense solve,
  `nystrom_preconditioner`, or a Sherman-Morrison/Woodbury built at the
  unpadded size) fails on the padded residual space and must be wrapped;
  `pad_dual_preconditioner(base_preconditioner, n_real)` applies the base
  callback on the first `n_real` coordinates and the exact
  \(1/\lambda\) inverse on the padded block:

    ```python
    from nlls_gram import pad_dual_preconditioner

    dual_preconditioner = pad_dual_preconditioner(base_preconditioner, n_real)
    ```

  A shape-generic base (`identity_preconditioner()`) stays valid unwrapped —
  it just forgoes the exact padded-block inverse. Do **not** zero the padded
  block instead: that makes the preconditioner singular rather than SPD,
  even though it can appear to work when exactly-zero padding never excites
  those coordinates.
- **`qr` does not survive padding**: the padded zero rows make the Jacobian
  rank-deficient, which the QR path's triangular solves cannot handle — the
  step is non-finite. Use `cholesky` or `cg`, or the damped augmented
  `augmented_qr` / `lsmr` (full column rank for \(\lambda>0\) even when \(J\)
  is rank-deficient), for padded problems.
- **Implicit AD**: the padded rows make the *undamped* implicit dual
  \(J P J^\top\) singular, so the library's implicit rules (dense and cg)
  return a non-finite derivative of `solve(...).x` on padded problems — and
  for the same reason `pad_dual_preconditioner` divides the padded block by
  the live damping and is rejected at construction when passed as an
  `implicit_preconditioner`. The
  minimum-metric-norm derivative still exists mathematically (padding only
  appends redundant equations) and equals the unpadded derivative, so
  differentiate the unpadded formulation to compute it.

## Matrix-Free LSMR (Whitened Subproblem)

`linear_solver="lsmr"` solves the whitened damped LM subproblem
`min_u ||r + B u||² + damping ||u||²` (`B = J S`, `S = metric.inv_sqrt`,
`S Sᵀ = M⁻¹`, step `s = S u`) with [LSMR](https://web.stanford.edu/group/SOL/software/lsmr/)
Golub-Kahan bidiagonalization, using only `J`/`Jᵀ` matvecs — the matrix-free
counterpart of `augmented_qr`. It exists for the same reason `qr`/`augmented_qr`
do: the `cg` dual operator `J M⁻¹ Jᵀ + damping I` has the *square* of the
whitened operator's condition number, so at small damping its step bottoms out
at an `eps·cond` floor (which dense direct solves hit too) with the error
concentrated in the slow, selection-critical directions. LSMR works at
`cond(B) ~ sqrt`, restoring endgame accuracy. See the
[tuning guide](tuning_guide.md#solver-selection) for when to prefer it over `cg`.

- **Stopping** maps the standard hooks: the normal-equations residual
  `||Bᵀ(B u + r) + damping·u|| = |zetabar|` (LSMR's exact monotone quantity) is
  driven below `iterative_tol · ||Bᵀr|| + iterative_atol`, capped by
  `iterative_maxiter` (all traced, so a callback can reschedule them). With
  `iterative_maxiter=None` a `min(m, n)`-scaled fallback cap applies.
- **Metric** it needs `metric.inv_sqrt` and `metric.inv_sqrt_transpose` (the
  default identity metric supplies both); it does not use `metric.solve`.
- **Preconditioning** its hook is `whitened_preconditioner` (a
  `WhitenedPreconditioner`), a parameter-space **right-preconditioner** applying
  `R⁻¹`/`R⁻ᵀ` via `solve(v, damping)`/`solve_transpose(w, damping)`. The solver
  runs LSMR on the preconditioned operator `B R⁻¹` — operator `x → B(solve(x))`,
  adjoint `w → solve_transpose(Bᵀ w)`, and the step un-preconditions the final
  iterate `u = R⁻¹ z`. A good `R` (a Schur-complement factor of the
  parameter-space normal operator is canonical) clusters the spectrum of `B R⁻¹`
  and cuts the endgame iteration count by orders of magnitude — an ill-conditioned
  whitened operator can need thousands of plain iterations versus tens
  preconditioned. **Surrogate semantics**: LSMR's scalar `damp` on `B R⁻¹`
  regularizes in the `RᵀR` metric, so the computed step is
  `u = -(BᵀB + damping RᵀR)⁻¹ Bᵀ r` — a documented surrogate of the `I`-damped
  subproblem. It is admissible: LM acceptance guards on the true `||r||`, and the
  `damping → 0` selection limit is `R`-invariant (the minimum-metric-norm step
  regardless of `R`), so equivalence checks must use `R=None` or the `RᵀR`-damped
  dense reference. Stopping (`iterative_tol`/`iterative_atol`) is measured on the
  preconditioned operator. The half-solves receive the live `damping` like
  `dual_preconditioner(v, damping)`. `None` (default) runs plain LSMR.
  `dual_preconditioner`, `preconditioner_factory`, and `recycle` remain `cg`-only
  and are rejected loudly for `lsmr`.
- **Differentiation**: reverse-AD through `update` works (the whitened solution
  is wrapped in `lax.custom_linear_solve` on the SPD normal operator
  `BᵀB + damping I`). Differentiating a forward `solve(...).x` uses the dense
  cholesky implicit rule by default (`implicit_solver="auto"` → cholesky, since
  `lsmr` is not `cg`); set `implicit_solver="cg"` with an `implicit_preconditioner`
  for a fully matrix-free derivative at very large `m`.

The standalone [`lsmr`](#nlls_gram.lsmr) function (operator/transpose matvecs,
`b`, `damp`) is exposed too, returning `(x, LSMRState)` with iteration count and
final normal-equations residual.

## Krylov Recycling / Deflation

`recycle=RecycleConfig(rank=k)` on a `cg` solver carries a deflation basis
across LM steps: each step harvests an eigCG-style basis from its CG iterations
and recycles it into the next step's two-level additive preconditioner
`M_defl(r) = P(r) + U E^{-1}(U'r)` (first-level `P` = the frozen
`dual_preconditioner`, or the `preconditioner_factory`'s current iterate-built
`apply` when one is used) plus a deflated warm start. See the
[tuning guide](tuning_guide.md#recycling-and-deflation-across-steps) for when it
pays off and how to configure `rank`/`window`.

The building blocks are also exposed for standalone matrix-free solves:

- [`deflated_pcg`](#nlls_gram.deflated_pcg) — two-level deflated PCG with the
  harvest; returns `(solution, HarvestState)`.
- [`build_coarse_operator`](#nlls_gram.build_coarse_operator) — precompute
  `W = A U` and the ridged Cholesky factor of `E = U'A U` (built once per step,
  reused across right-hand sides).
- `RecycleState` rides `LMState.recycle`; its `iterations` / `residual_norm`
  fields report the last velocity solve's diagnostics.

The `E`-ridge is a trace-scaled shift with a dtype-keyed absolute floor
(`1e-12` float64, `1e-6` float32, floored at `tiny/eps`), so the coarse
Cholesky stays finite even for a zero or rank-deficient `U`; it lives only
inside a preconditioner and never moves the converged root. The harvest
reorthonormalizes its window (`reorthogonalize=True`, the robust default) so the
emitted basis is orthonormal; a genuinely non-finite operator still propagates
loudly rather than being clamped.

## API

::: nlls_gram.RecycleConfig

::: nlls_gram.RecycleState

::: nlls_gram.deflated_pcg

::: nlls_gram.HarvestState

::: nlls_gram.build_coarse_operator

::: nlls_gram.recycled_cg

::: nlls_gram.lsmr

::: nlls_gram.LSMRState

::: nlls_gram.WhitenedPreconditioner

::: nlls_gram.metric_from_tridiagonal_precision

::: nlls_gram.metric_from_state_space

::: nlls_gram.matern_state_space

::: nlls_gram.metric_from_quasiseparable

::: nlls_gram.metric_from_shifted_matvec

::: nlls_gram.metric_from_diagonal

::: nlls_gram.blockdiag_metric

::: nlls_gram.repeated_blockdiag_metric

::: nlls_gram.metric_with_compute_dtype

::: nlls_gram.sherman_morrison_preconditioner

::: nlls_gram.woodbury_preconditioner

::: nlls_gram.identity_preconditioner

::: nlls_gram.nystrom_preconditioner

::: nlls_gram.PreconditionerFactory

::: nlls_gram.pad_dual_preconditioner
