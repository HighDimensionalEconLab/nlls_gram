# Metric and Preconditioner Utilities

The library keeps its metric constructors focused on dense, diagonal, and the
repeated shifted kernel geometry used by kernel least-squares models. For any
other geometry, construct a [`Metric`](metrics.md) directly. Iterative solver
preconditioners remain separate: they can approximate an operator without
changing the nonlinear least-squares problem, whereas a metric defines the
problem itself.

| Helper | Builds | Storage |
| --- | --- | --- |
| `metric_from_cholesky(L)` | dense `Metric` from \(M = LL^\top\) | \(O(n^2)\) |
| `metric_from_diagonal(weights)` | `Metric` from \(M = \operatorname{diag}(w)\) | \(O(n)\) |
| `repeated_shifted_dense_metric(K, ...)` | repeated dense kernel blocks plus a common shift | \(O(n^2)+O(1)\) |
| `repeated_shifted_state_space_metric(t, ...)` | the same geometry for an implicit state-space kernel Gram | \(O(nq^2)+O(1)\) |
| `matern_state_space(sigma, ell, nu)` | state-space inputs for Matérn-1/2, 3/2, or 5/2 | \(q=1,2,3\) |
| `sherman_morrison_preconditioner(solve, u, weight)` | `dual_preconditioner` for \(B = A + w\,uu^\top\) | one `solve` |
| `woodbury_preconditioner(solve, U, weights)` | `dual_preconditioner` for \(B = A + U\operatorname{diag}(w)U^\top\) | one `solve` + \(k \times k\) |
| `identity_preconditioner()` | the explicit "no preconditioner" choice | free |
| `nystrom_preconditioner(matvec, n, rank, key)` | randomized Nyström `dual_preconditioner` | two \((n, \text{rank})\) GEMVs |
| `pad_dual_preconditioner(base, n_real)` | extends a `dual_preconditioner` to a zero-padded residual | base + \(O(k)\) |

## Dense and Diagonal Metrics

For a dense positive-definite metric \(M=LL^\top\), pass the lower Cholesky
factor. A diagonal metric takes its positive weights directly:

```python
import jax.numpy as jnp

from nlls_gram import metric_from_cholesky, metric_from_diagonal

dense_metric = metric_from_cholesky(jnp.linalg.cholesky(M))
diagonal_metric = metric_from_diagonal(jnp.array([2.0, 1.0, 0.5]))
```

Both constructors provide `solve`, `norm`, `inv_sqrt`, and
`inv_sqrt_transpose`, including matrix right-hand sides where the callback
contract permits them.

## Repeated Shifted Kernel Metrics

Both repeated constructors implement exactly

\[
M = \operatorname{blockdiag}(\underbrace{K,\ldots,K}_{r},0_s)
    + \varepsilon I
  = \operatorname{blockdiag}(K+\varepsilon I_n,\ldots,
      K+\varepsilon I_n,\varepsilon I_s).
\]

The flattened parameter vector must contain the `r` kernel-coefficient blocks
first and the `s` zero-block coordinates last. The keyword arguments
`repeats`, `zero_pad_size`, and `epsilon` are mandatory: `repeats` is a
positive integer, `zero_pad_size` is a nonnegative integer (use `0` when no
tail is present), and `epsilon` is a positive scalar. The shift is part of the
metric, including on the trailing zero block. A nonpositive Python scalar is
rejected eagerly; a nonpositive traced or device scalar is mapped to `NaN` so
the solve fails loudly without a host synchronization.

The constructors provide all four metric callbacks and therefore work with
both Gram and whitened linear solvers. They do not choose a representation
automatically: call the dense or state-space constructor explicitly.

### Dense Repeated Metric

```python
from nlls_gram import repeated_shifted_dense_metric

metric = repeated_shifted_dense_metric(
    K,
    repeats=5,
    zero_pad_size=3,
    epsilon=1e-8,
)
```

For a positive-semidefinite `K`, `repeated_shifted_dense_metric` factors
\(K+\varepsilon I_n\) once. It stores
one \(n\times n\) Cholesky factor and the scalar shift, not `repeats` copies,
a full block diagonal, or a padding vector. Each callback reshapes the
repeated blocks into columns and applies the shared factor once to that batch;
in particular, `solve` performs two triangular solves total rather than two
per block. Matrix inputs add their right-hand sides to the same packed column
dimension. The tail uses scalar division or scaling by `epsilon`.

Persistent storage is \(O(n^2)+O(1)\), independent of `repeats` and
`zero_pad_size`. Work still scales with the number of parameter coordinates,
as it must, but avoids factoring or loading a massive sparse-in-content dense
matrix. This is the preferred constructor when a dense `K` already exists and
for the small-to-moderate kernel grids where dense BLAS is fastest.

### State-Space Repeated Metric

For a stationary kernel with a finite-dimensional state-space representation,
the same geometry can be applied without constructing a dense \(K\):

```python
from nlls_gram import (
    matern_state_space,
    repeated_shifted_state_space_metric,
)

metric = repeated_shifted_state_space_metric(
    t,
    *matern_state_space(sigma=1.0, ell=10.0, nu=1.5),
    repeats=5,
    zero_pad_size=3,
    epsilon=1e-8,
)
```

Here `t` is the strictly increasing one-dimensional coordinate on which the
state-space kernel is evaluated; it need not represent calendar time. Ordering
is semantically required; non-increasing traced or device coordinates map the
metric shift to `NaN` so the solve fails loudly without a host synchronization.
The remaining positional inputs are the observation vector `h`, stationary
covariance `Pinf`, and callable `transition(dt)`. For Matérn-1/2, 3/2, and 5/2,
`matern_state_space` returns those objects with latent state dimension
\(q=1,2,3\), respectively.

The constructor folds `epsilon` into the diagonal before a structured
Cholesky factorization. It stores one quasiseparable factor and applies all
repeated blocks as batched right-hand sides through forward or reverse scans.
The metric norm evaluates \(\lVert L^\top x\rVert\) in one reverse scan. No
dense \(K\), repeated factor, full block diagonal, or padding vector is
formed. Persistent storage is \(O(nq^2)+O(1)\), independent of `repeats` and
`zero_pad_size`, and each apply costs \(O(nq^2b)\) for `b` packed right-hand
side columns (for a vector input, `b == repeats`).

`parallel=None` selects from the process default backend: associative scans
only for float64 off CPU and sequential scans otherwise. Pass `True` or
`False` when arrays use nondefault device placement or to force a path after
checking numerical agreement on the target grid. State-space structure does
not imply that it is faster at small `n`: benchmark the end-to-end solve, and
prefer `repeated_shifted_dense_metric` when a dense Gram is already needed by
the model. There is deliberately no automatic dense/state-space dispatch.

## Sherman–Morrison Dual Preconditioner

With `linear_solver="gram_cg"`, the `dual_preconditioner(v, damping)` argument
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

`linear_solver="gram_cg"` requires a `dual_preconditioner`,
`linear_solver="normal_cg"` a `normal_preconditioner`, and a
`gram_cg`-resolved AD solve an `ad_solver_preconditioner` — running
Krylov methods unpreconditioned should be a decision, not a default.
`identity_preconditioner()` is that decision made explicit and greppable:

```python
from nlls_gram import identity_preconditioner

solver = LevenbergMarquardt(
    residual_fn,
    linear_solver="gram_cg",
    dual_preconditioner=identity_preconditioner(),
    ad_solver_preconditioner=identity_preconditioner(),
)
```

The returned callable accepts both hook signatures — `(v, damping)` and
`(v)` — so one helper serves all three arguments (and it trivially satisfies
the `normal_preconditioner` range-preservation requirement below).

## The Normal-Space Preconditioner (`normal_cg`)

`linear_solver="normal_cg"` requires a `normal_preconditioner(v, damping)`:
a jit-traceable, linear, SPD **parameter-space** approximation of
\((B^\top B + \lambda I_n)^{-1} v\) with \(B = JS\)
(`identity_preconditioner()` is the explicit opt-out). Like every
preconditioner it may approximate freely as far as *convergence* is
concerned — at inner convergence the step is unchanged. But it carries one
structural requirement with no Gram-side analogue:

**Range preservation.** On rank-deficient problems the minimum-\(M\)-norm
selection rests on the CG iterates staying in
\(\operatorname{range}(B^\top)\): the right-hand side \(-B^\top r\) starts
there, the operator \(B^\top B + \lambda I\) maps the subspace to itself, so
unpreconditioned CG from zero never acquires a null-space component — the
converged step and every budget-truncated step stay selection-clean. A
preconditioner \(C\) enters the Krylov space through its images, so unless

$$
C\bigl(\operatorname{range}(B^\top)\bigr) \subseteq \operatorname{range}(B^\top),
$$

the iterates leak into the null space of \(B\) and the computed step
silently stops being the minimum-norm one — the CG residual still converges,
so nothing fails loudly. An arbitrary SPD approximation of the inverse does
**not** have this property, and rank deficiency is the package's home turf
(tall interpolation problems always carry redundant rows or collinear
columns). Safe constructions: the identity; polynomials in the operator
itself; an exact \((B^\top B + \tau I)^{-1}\) at a fixed shift
\(\tau > 0\); any \(C\)
that commutes with the orthogonal projector onto
\(\operatorname{range}(B^\top)\). On full-column-rank problems
(\(\operatorname{rank} B = n\)) the condition is vacuous and any SPD \(C\)
is safe.

`dual_preconditioner`, `preconditioner_factory`, and `recycle` remain
`gram_cg`-only — they live in residual space and cannot serve the
parameter-space system. A `normal_cg`-resolved *implicit* solve needs no
preconditioner at all (its right-hand side lies in
\(\operatorname{range}(B^\top)\), so unpreconditioned CG already selects
the minimum-norm tangent); an optional `ad_solver_preconditioner` supplied
there acts in the same parameter space under the same range-preservation
requirement at `damping = 0` — see
[Implicit AD](implicit_ad.md#the-ad-solver-preconditioner).

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
    linear_solver="gram_cg",
    iterative_tol=1e-6,
    iterative_maxiter=20,
    dual_preconditioner=nystrom_preconditioner(
        ntk_matvec, m, rank, jax.random.PRNGKey(0)
    ),
    ad_solver_preconditioner=identity_preconditioner(),
)
```

Passed as `ad_solver_preconditioner` the helper applies its undamped
(zero-damping) inverse, which is valid only when the retained spectrum is
strictly positive.

## Iterate-Adaptive Preconditioner Factory

Every helper above is *frozen*: built once, at one linearization point. That is
safe when the dual operator \(J M^{-1} J^\top + \lambda I\) stays spectrally
close as LM drifts \(x\). When it does not — the Jacobian rotates enough that a
preconditioner built at \(x_0\) decays into an ineffective approximation once
\(x\) moves, and the inner CG stalls or breaks down — pass a
`PreconditionerFactory(prepare, apply)` instead of `dual_preconditioner` (pass
exactly one of the two for `linear_solver="gram_cg"`; like both dual hooks it
is `gram_cg`-only). Its `prepare(x, args, p, aux)`
rebuilds the preconditioner state from the **current** iterate, inside the
jitted loop as traced ops with no recompiles:

```python
from nlls_gram import LevenbergMarquardt, PreconditionerFactory

def prepare(x, args, p, aux):
    # model-structured build from the CURRENT iterate x (the user pytree,
    # not the raveled theta); return any fixed-shape pytree of arrays
    d = jnp.exp(A @ x)
    return d * d                     # e.g. the exact current dual diagonal

def apply(state, v, damping):
    return v / (state + damping)     # SPD, linear in v

solver = LevenbergMarquardt(
    residual_fn,
    linear_solver="gram_cg",
    preconditioner_factory=PreconditionerFactory(prepare, apply),
    iterative_maxiter=...,
)
```

- `prepare(x, args, p, aux) -> state` receives the **user pytree** `x` (model
  structure intact), the residual `args`, `p`, and the residual aux evaluated
  at the same linearization point (`None` when `has_aux=False`) — the same
  signature as `MetricFactory.prepare` — and returns a fixed-shape pytree of
  arrays.
- `apply(state, v, damping) -> vector` is the per-iteration apply: an SPD,
  linear-in-`v` approximation of \((J M^{-1} J^\top + \lambda I)^{-1} v\). It
  must stay well-defined at `damping = 0`, because a `gram_cg`-resolved
  implicit derivative reuses it (undamped) at the converged solution unless
  an explicit `ad_solver_preconditioner` overrides it.

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

- **The dense forms are unchanged mathematically.** Under `gram_cholesky`
  the padded block decouples exactly and the step matches the unpadded step
  (regression-tested for both the plain and geodesic-accelerated updates);
  under the normal forms the zero rows contribute nothing to \(B^\top B\)
  or \(B^\top r\), so the system is literally identical to the unpadded
  one. One shape effect to know: padding raises \(m\), which can flip
  `auto`'s shape rule from `gram_cholesky` to `normal_cholesky` — the step
  is the same either way. Large padding costs the larger materialized
  residual dimension (and, for the Gram form, its dense dual factor).
- **`gram_cg`**: a shape-fixed `dual_preconditioner` (a dense solve,
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
  step is non-finite. Every other solver handles padding: the damped
  Gram/normal forms directly, `augmented_qr` / `lsmr` through the damping
  block that stays full column rank for \(\lambda>0\).
- **Implicit AD**: the padded rows make the *undamped* implicit systems
  singular *but consistent* — their `p`-derivative rows are identically
  zero too — which is exactly the singular-but-consistent regime of
  [the implicit rules](implicit_ad.md#rank-deficiency-and-the-ridge). The
  `svd` computes the minimum-metric-norm tangent through its spectral-filter
  pseudoinverse, and a `normal_cg` AD solve computes it with no ridge in exact
  arithmetic — on its default unridged path, with the inner CG run to
  convergence, and either unpreconditioned or with a range-preserving
  `ad_solver_preconditioner`. `regularized_normal_cg` is the separate biased
  method; the loud rank guard belongs to `qr`, not `normal_cg`. A
  `gram_cg`-resolved AD solve is the
  fragile choice here — run-to-tolerance CG on the singular padded dual —
  and `pad_dual_preconditioner` divides the padded block by the live
  damping, so it is rejected at construction when passed as an
  `ad_solver_preconditioner`. The minimum-metric-norm derivative equals the
  unpadded derivative (padding only appends redundant equations), so when
  in doubt differentiate the unpadded formulation.

## Matrix-Free LSMR (Whitened Subproblem)

`linear_solver="lsmr"` solves the whitened damped LM subproblem
`min_u ||r + B u||² + damping ||u||²` (`B = J S`, `S = metric.inv_sqrt`,
`S Sᵀ = M⁻¹`, step `s = S u`) with [LSMR](https://web.stanford.edu/group/SOL/software/lsmr/)
Golub-Kahan bidiagonalization, using only `J`/`Jᵀ` matvecs — the matrix-free
counterpart of `augmented_qr`. It exists for the same reason `qr`/`augmented_qr`
do: the Gram and normal operators (`J M⁻¹ Jᵀ + damping I`, `BᵀB + damping I`)
carry the *square* of the whitened operator's condition number, so at small
damping their steps bottom out at an `eps·cond` floor (which dense direct
solves hit too) with the error concentrated in the slow, selection-critical
directions. LSMR works at `cond(B) ~ sqrt`, restoring endgame accuracy. See
the [tuning guide](tuning_guide.md#solver-selection) for when to prefer it
over the CG forms.

- **Stopping** maps the standard hooks: the normal-equations residual
  `||Bᵀ(B u + r) + damping·u|| = |zetabar|` (LSMR's exact monotone quantity) is
  driven below `iterative_tol · ||Bᵀr|| + iterative_atol`, capped by
  `iterative_maxiter` (all traced, so a callback can reschedule them). With
  `iterative_maxiter=None` a `min(m, n)`-scaled fallback cap applies.
- **Metric** it needs `metric.inv_sqrt` and `metric.inv_sqrt_transpose` (the
  default identity metric supplies both); it does not use `metric.solve`.
- **Preconditioning** its hook is `whitened_preconditioner` (a
  `WhitenedPreconditioner`), a parameter-space **right-preconditioner**
  applying `R⁻¹`/`R⁻ᵀ` via `solve(v, damping)`/`solve_transpose(w, damping)`.
  The solver runs LSMR on the augmented preconditioned operator

        z → [B(solve(z)); sqrt(damping)·solve(z)]
        w → solve_transpose(Bᵀ w[:m]) + sqrt(damping)·solve_transpose(w[m:])

  and un-preconditions the final iterate, `u = R⁻¹ z`. Because the damping
  row is `sqrt(damping)·R⁻¹z` — not `sqrt(damping)·z` — the least-squares
  problem in `z` is exactly `min ||B u + r||² + damping ||u||²` over
  `u = R⁻¹ z`: **every `damping > 0` *posed* subproblem is the
  identity-damped whitened subproblem**, so at inner convergence the step
  is identical to plain LSMR's and the `damping → 0` selection limit is the
  minimum-metric-norm step for *any* `R`. `R` changes the iteration path,
  not the subproblem — a budget-truncated (unconverged) iterate can still
  depend on `R`. A good `R` (a Schur-complement factor of the parameter-space normal
  operator is canonical) clusters the spectrum of `B R⁻¹` and cuts the
  endgame iteration count by orders of magnitude — an ill-conditioned
  whitened operator can need thousands of plain iterations versus tens
  preconditioned. Stopping (`iterative_tol`/`iterative_atol`) is measured on
  the preconditioned operator. The half-solves receive the live `damping`
  like `dual_preconditioner(v, damping)`. `None` (default) runs plain LSMR.
  `dual_preconditioner`, `preconditioner_factory`, and `recycle` remain
  `gram_cg`-only and are rejected loudly for `lsmr`.
- **Differentiation**: reverse-AD through `update` works (the whitened
  solution is wrapped in `lax.custom_linear_solve` on the SPD preconditioned
  normal operator `R⁻ᵀ(BᵀB + damping I)R⁻¹`). Differentiating a forward
  `solve(...).x` uses `direct` for a square system and `svd` otherwise
  (`ad_solver="auto"`); set
  `ad_solver="normal_cg"` (no preconditioner needed) or
  `"gram_cg"` with an `ad_solver_preconditioner` for a fully matrix-free
  derivative.

The standalone [`lsmr`](#nlls_gram.lsmr) function (operator/transpose matvecs,
`b`, `damp`) is exposed too, returning `(x, LSMRState)` with iteration count and
final normal-equations residual.

## Krylov Recycling / Deflation

`recycle=RecycleConfig(rank=k)` on a `gram_cg` solver carries a deflation basis
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

::: nlls_gram.metric_from_cholesky

::: nlls_gram.metric_from_diagonal

::: nlls_gram.repeated_shifted_dense_metric

::: nlls_gram.repeated_shifted_state_space_metric

::: nlls_gram.matern_state_space

::: nlls_gram.sherman_morrison_preconditioner

::: nlls_gram.woodbury_preconditioner

::: nlls_gram.identity_preconditioner

::: nlls_gram.nystrom_preconditioner

::: nlls_gram.PreconditionerFactory

::: nlls_gram.pad_dual_preconditioner
