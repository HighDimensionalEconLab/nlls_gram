# Implicit Differentiation

`solve` has a custom implicit JVP/VJP with respect to `p` for the solved
parameters:

```python
solver.solve(x0, args, p=p).x
```

The custom rule is not defined on the per-step `update(...)` interface, and it
does not differentiate through the LM iterations. It differentiates the residual
equation at the returned solution. For implicit differentiation, use a fixed
`args` and read the differentiated value from `result.x`.

Here `p` means the external pytree argument passed to the residual function:

```python
residual_fn(x, args, p)
```

It does not mean LM hyperparameters such as `init_damping`, `max_steps`,
`atol`, callback choices, or metric callbacks. The custom rule treats `args` and
the initial guess `x0` as fixed for this derivative (their tangents are zero,
not an error).

There is no setup stage that AD must trace through: the whole iteration —
every update, callback, and the final aux evaluation — sits inside one
`jax.custom_jvp` boundary, so derivative information flows only through the
implicit rule at the returned solution. `init` is differentiation-inert (its
outputs are constants whose shapes and dtypes come from one residual
evaluation), so calling it by hand inside a differentiated function, or
implicitly via `cache_jacobian=True`, contributes exactly zero to any
derivative. `result.aux` also participates in the implicit rule — see
[Aux outputs](#aux-outputs) below. One
construction-time caveat: the solver (including any `Metric` callbacks) is a
static object — do not build it from traced values.

## Root Selection and the Metric

In underdetermined interpolation problems there may be many roots
\(\theta\) satisfying

$$
r(\theta, a, p)=0,
$$

where \(a\) denotes fixed auxiliary data from `args`. A perturbation \(\dot p\)
does not determine a unique parameter tangent when the parameter dimension
exceeds the residual dimension: the linearized root constraint is

$$
J_\theta \dot\theta + J_p \dot p = 0,
$$

and any null-space vector \(z\) with \(J_\theta z=0\) can be added to a solution.
The metric \(M \succ 0\) selects the tangent with minimum metric norm:

$$
\dot\theta
= \arg\min_u \frac12 u^\top M u
\quad\text{subject to}\quad
J_\theta u = -J_p\dot p.
$$

This is why the metric matters for implicit AD: in underdetermined problems the
norm is part of the definition of the derivative of the selected solution
branch. With \(M=I\) this is the Euclidean minimum-norm tangent; with an RKHS or
kernel coefficient metric, it is the minimum RKHS-norm tangent.

Let

$$
J_\theta =
\frac{\partial r}{\partial \theta}(\theta^\star, a, p)
\in \mathbb R^{m\times n},
\qquad
J_p \dot p =
\frac{\partial r}{\partial p}(\theta^\star, a, p)\dot p
\in \mathbb R^m,
\qquad
P = M^{-1}.
$$

For a pytree `p`, \(J_p\dot p\) means the JAX JVP of the residual with respect
to the `p` argument only, evaluated at fixed \(\theta^\star\) and `args`.

The Lagrange conditions for the minimum metric-norm problem give

$$
M\dot\theta + J_\theta^\top y = 0,
\qquad
J_\theta \dot\theta = -J_p\dot p,
$$

so

$$
\dot\theta
= -P J_\theta^\top
(J_\theta P J_\theta^\top)^{-1}
J_p\dot p.
$$

The dual system above is how the **Gram** implicit forms compute the
tangent. The **normal** forms compute the same minimum-\(M\)-norm tangent in
whitened variables: with \(B = J_\theta S\) (\(S S^\top = P\)) and
\(\dot\theta = S\dot u\),

$$
B^\top B\,\dot u = -B^\top J_p\dot p,
\qquad
\dot\theta = S\,\dot u,
$$

whose minimum-norm solution maps back to exactly the tangent above. In code,
the Gram forms apply \(P x\) with `metric.solve(x)` when available, else as
\(P x = S S^\top x\) through the square-root callbacks; the normal forms use
`metric.inv_sqrt`/`inv_sqrt_transpose` directly.

### Iterate-Dependent Metrics Are Frozen per Solve

With an iterate-dependent metric (`metric_factory=`), the metric is FROZEN at
the returned solution: `prepare`/`build` run once at
`(result.x, result.args, result.p, result.aux)` and the resulting \(P\) is
applied exactly as a fixed metric would be. The state-dependence of the metric
is deliberately not differentiated — the same contract as a fixed metric
closing over constants, matching the forward selection role the metric plays.
The built metric must stay self-adjoint and positive definite for that fixed
state.

The freeze is a **first-order** statement. Each first-order implicit solve
applies the metric frozen at *its* solution — the verified contract
(first-order forward and reverse mode, fixed or factory-built metric). The
tangent field \(p \mapsto \dot\theta(p)\) so defined uses, at every \(p\),
the metric rebuilt at that point's solution — but **higher-order AD through
a factory-built metric's state dependence is unsupported in the implicit
rules**: the metric wrappers inside the tangent solve cannot see the
solve-side parameters, so second derivatives do not account for the
metric's point dependence. Take higher-order derivatives of `solve` only
with a fixed metric, where the declared solves re-apply correctly (subject
to the spectral-filter caveat in
[the ridge section](#rank-deficiency-and-the-ridge)).

## The Four Implicit Forms

`implicit_solver` selects among
`{"auto", "gram_cholesky", "normal_cholesky", "gram_cg", "normal_cg"}` —
the same Gram/normal taxonomy as the forward solver, and always
independently swappable from it (an `lsmr` forward solve with
`implicit_solver="normal_cg"` is fully matrix-free end to end). The default
`"auto"` matches the forward form where one exists and otherwise falls back
to the dense shape rule:

| forward `linear_solver` | `implicit_solver="auto"` resolves to |
| --- | --- |
| `gram_cholesky` | `gram_cholesky` |
| `normal_cholesky` | `normal_cholesky` |
| `gram_cg` | `gram_cg` |
| `normal_cg` | `normal_cg` |
| `auto`, `qr`, `augmented_qr`, `lsmr` | `gram_cholesky` if \(n > m\), else `normal_cholesky` |

The shape fallback matters structurally: on a square-to-tall problem
(\(m \ge n\)) the \(m \times m\) dual is singular whenever \(J_\theta\) has
rank below \(m\) — which rank alone forces for \(m > n\) — so the normal
forms are the right implicit rules there, and that is what the fallback
picks.

The four forms:

- **`gram_cholesky`** materializes \(J_\theta^\top\), assembles the
  \(m \times m\) dual \(J_\theta P J_\theta^\top\), and factors it densely,
  regularized by the relative ridge described
  [below](#rank-deficiency-and-the-ridge).
- **`normal_cholesky`** assembles \(B^\top = S^\top J_\theta^\top\)
  (\(n \times m\)) and solves the \(n \times n\) normal system
  \(B^\top B\,\dot u = -B^\top J_p\dot p\), mapping back
  \(\dot\theta = S\dot u\). By default it applies the exact
  **pseudoinverse** of \(B^\top B\) through an `eigh` spectral filter —
  `implicit_penalty` selects among three behaviors,
  [described below](#rank-deficiency-and-the-ridge).
- **`gram_cg`** applies \(y \mapsto J_\theta P J_\theta^\top y\)
  matrix-free using JAX JVP/VJP closures, then solves with CG wrapped in a
  symmetric `jax.lax.custom_linear_solve` — both JVP and VJP of
  `solve(...).x` stay matrix-free.
- **`normal_cg`** applies \(\dot u \mapsto B^\top(B\dot u)\) matrix-free
  through the same JVP/VJP closures. Its right-hand side
  \(-B^\top(J_p\dot p)\) lies in \(\operatorname{range}(B^\top)\) by
  construction, so CG from zero converges to the minimum-norm \(\dot u\) —
  hence the minimum-\(M\)-norm tangent — with no ridge, in exact arithmetic
  even when \(B^\top B\) is singular. Reverse mode is *not* the same
  symmetric solve: a cotangent right-hand side \(S^\top\bar\theta\) has no
  reason to lie in \(\operatorname{range}(B^\top)\), and CG on the singular
  normal operator with an inconsistent right-hand side breaks down. The
  unridged rule therefore declares an explicit transpose through the
  push-through identity \(N^{+} = B^\top (BB^\top)^{+2} B\): two
  *consistent* \(m\)-space dual CG solves (unpreconditioned — the
  parameter-space hook does not apply there), with the trailing
  \(B^\top\) annihilating dual-null rounding noise. The ridged path (an
  explicitly positive `implicit_penalty`) is nonsingular and stays
  symmetric. The final \(\dot\theta = S\dot u\) map likewise declares its
  exact transpose (`inv_sqrt_transpose`), so non-self-adjoint square roots
  — a triangular Cholesky factor — differentiate correctly in reverse
  mode.

Both CG forms assume the inner solve **converges**.
`jax.lax.custom_linear_solve` transposes the declared solve as an exact
linear map, but a truncated or early-stopped CG (a small bounded
`implicit_maxiter`) is not a linear function of its right-hand side — its
declared transpose then no longer matches what was computed, and
derivatives are silently inaccurate on top of the truncation error itself.
Run the implicit CG solves to tolerance (the default
`implicit_maxiter=None`), and treat a bounded budget as valid only with a
preconditioner exact enough to converge within it.

The dense-resolved implicit rules inherit `linear_solve_dtype`: the undamped
implicit system is the most conditioning-sensitive solve in the library, so
it is never silently less precise than the damped forward solve (measured on
a \(10^{-7}\)-weight spike metric: a float32 implicit tangent wrong by ~5% —
and its VJP by ~40% — becomes accurate to ~\(10^{-7}\), with the returned
tangent still float32).

### The Implicit Preconditioner

A `gram_cg`-resolved implicit solve requires an `implicit_preconditioner`
at construction, even if `solve(...).x` is never differentiated — and the
default `implicit_solver="auto"` resolves to `gram_cg` whenever the forward
solver is `gram_cg`. There the hook approximates the undamped
residual-space \((J_\theta P J_\theta^\top)^{-1} v\) on \(m\)-vectors, and
a forward `preconditioner_factory`'s state at the solution serves as the
default when no explicit hook is given. Under `normal_cg` the hook is
**optional**: the tangent right-hand side lies in
\(\operatorname{range}(B^\top)\), so unpreconditioned CG already selects
the minimum-norm tangent. A hook supplied there acts in *parameter* space —
an approximation of the undamped \((B^\top B)^{-1} v\) on \(n\)-vectors —
and on rank-deficient problems it must preserve
\(\operatorname{range}(B^\top)\), the same structural requirement as the
forward
[`normal_preconditioner`](utilities.md#the-normal-space-preconditioner-normal_cg);
the dual-space factory can never serve this parameter-space system.

The callback may take `(v)` or `(v, damping)`: a callable *requiring* the
damping argument is called with an explicit zero damping (the implicit
system is undamped), and one whose damping has a default passes through
unchanged — so `identity_preconditioner()`,
`sherman_morrison_preconditioner`, `woodbury_preconditioner`, and
`nystrom_preconditioner` all serve the hook directly. The one exception
is `pad_dual_preconditioner`, which divides by the live damping and is
rejected at construction. Pass `identity_preconditioner()` to run the
implicit CG unpreconditioned, or a dense `implicit_solver` to use a dense
rule instead. The forward `dual_preconditioner` is never reused implicitly:
it approximates the damped operator, and the implicit system is undamped —
reusing one is an explicit choice (pass the same callable to both
arguments). Any `implicit_preconditioner` must be linear, self-adjoint, and
positive definite for the operator of its resolved space, and the metric
inverse \(P\) must be linear, self-adjoint, and positive definite in
parameter space.

## Rank Deficiency and the Ridge

The implicit system is intentionally not damped — damping would change the
minimum-\(M\)-norm derivative. What happens on a rank-deficient system
depends on consistency, and interpolation problems produce **consistent**
linearized systems by construction: differentiating the root identity
\(r(\theta^\star(p), a, p) = 0\) along the solution branch gives
\(J_\theta\dot\theta^\star + J_p\dot p = 0\), i.e.
\(J_p\dot p \in \operatorname{range}(J_\theta)\) — redundant rows and
collinear columns included. The four forms then behave as follows:

- **`gram_cholesky`** adds a relative Tikhonov ridge,
  `implicit_penalty * trace(J P J') I`, before factorization (default:
  `implicit_penalty=None`, resolving to `1e-12` for a float64 solve and
  `1e-6` for float32, after any `linear_solve_dtype` promotion). On a
  singular-but-consistent dual — redundant residual rows at the returned
  solution, e.g. a simulated trajectory that has settled onto its steady
  state — the ridge tangent converges to the minimum-norm tangent as the
  penalty shrinks, so the default returns the min-norm tangent at an
  O(`implicit_penalty` · \(m\)) relative bias instead of a NaN. A ridge is
  compatible with min-norm accuracy here because the composition ends in
  \(J_\theta^\top\), which annihilates dual-null factorization noise. The
  default constants are empirical: for near-duplicate float64 rows the
  factorization-noise floor sits below `1e-14` and visible tangent bias
  above `~1e-6`, so the default is orders of magnitude from both edges.
  The trade-off is visibility: on a genuinely *inconsistent* singular dual
  the ridge returns a finite, penalty-inflated tangent where an
  unregularized rule fails loudly. Pass `implicit_penalty=0.0` for the
  exact unridged factorization (non-finite on any singular dual).
- **`normal_cholesky`** treats `implicit_penalty` as a trio of behaviors,
  because no ridge can serve as *its* default: this rule ends in \(S\),
  with no \(J_\theta^\top\)-style cleanup, so a ridge \(\delta\) leaves an
  error floor of order \(\mathrm{eps}/\delta + \delta\) along
  \(\ker B\) — minimized at \(\sim\sqrt{\mathrm{eps}}\), provably short of
  min-norm-tangent accuracy. The default `None` therefore applies the exact
  **pseudoinverse** of \(B^\top B\) by an `eigh` spectral filter at the
  standard \(n \cdot \mathrm{eps} \cdot \lambda_{\max}\) rank cutoff:
  exact Gauss-Newton sensitivity at full column rank (square nonsingular:
  \(-J_\theta^{-1}J_p\dot p\)), the minimum-\(M\)-norm tangent for
  consistent rank-deficient systems, with no ridge bias in either. Genuine
  eigenvalues below the cutoff are treated as rank-deficient — standard
  pseudoinverse semantics. The filter sits inside a symmetric
  `custom_linear_solve`, so higher-order AD re-applies the solve to new
  right-hand sides and never differentiates `eigh` itself (whose
  derivative rule breaks on exactly the repeated-zero spectra this path
  exists for). That re-application drops the derivative of the range
  projector, which rotates when the active subspace moves with the
  differentiation point — higher-order derivatives through the filtered
  solve are exact only while the active subspace is locally constant. An explicitly *positive* `implicit_penalty` opts into the
  trace-scaled ridge `B'B + implicit_penalty * trace(B'B) I` —
  O(`implicit_penalty` · \(n\)) bias, smooth in the spectrum for
  higher-order AD on nearly-degenerate problems. An explicit `0.0` demands
  the unridged Cholesky, guarded by a deterministic pivot-based rank check
  that poisons the tangent to NaN on rank deficiency — without the guard
  an exactly-zero pivot can round into a finite answer silently shifted
  along \(\ker B\).
- **`gram_cg`** ignores `implicit_penalty` entirely: its run-to-tolerance
  default produces non-finite derivatives on a singular dual (loud), while
  a small bounded `implicit_maxiter` returns a finite — and wrong —
  derivative with no diagnostic, so reserve the bounded-budget mode for
  exact preconditioners.
- **`normal_cg`** is exact by default: no ridge, and its range-preserving
  Krylov iteration converges to the min-norm tangent on
  singular-but-consistent systems anyway. A matrix-free ridge is added only
  when `implicit_penalty` is passed explicitly positive (its scale set by
  a Rayleigh quotient of \(B^\top B\) over a fixed deterministic probe
  vector, so the tangent map stays linear in the seed) — an opt-in
  stabilizer for systems where floating-point drift, not rank, is the
  issue. The default stays exact rather than silently biased.

**Gauss–Newton semantics on inconsistent systems.** When the returned
point is a least-squares stationary point with nonzero residual rather than
an exact root, \(J_p\dot p\) need not lie in
\(\operatorname{range}(J_\theta)\). The normal equations
\(B^\top B\dot u = -B^\top J_p\dot p\) are consistent regardless (any
\(B^\top\)-image is), so the normal forms return a finite answer: the
minimum-\(M\)-norm **Gauss–Newton sensitivity of the linearized system**.
That is *not* the exact optimizer sensitivity — the exact derivative of a
nonzero-residual stationary point carries residual-weighted
second-derivative (curvature) terms that the frozen-Jacobian system drops.
A one-line counterexample: \(r(x, p) = (x - p,\ x^2 - 1)\) at the
stationary point \(x = 0.5\), \(p = -0.25\) has exact sensitivity
\(dx/dp = 2\), while the Gauss–Newton normal system gives \(0.5\). The
implicit rules are exact at interpolating (zero-residual) roots — the
package's target — and Gauss–Newton approximations away from them. The
Gram forms don't even get that far: their dual system is
singular-inconsistent there, giving a ridge-inflated finite tangent
(`gram_cholesky`) or a loud failure (`gram_cg`).

Self-adjointness of \(P\) is all the Gram forms need from `metric.solve`:
the final \(P J_\theta^\top y\) application acts on tangent data, and its
transpose in the VJP is declared to be \(P\) itself
(a symmetric `jax.lax.custom_linear_solve`, which also batches under
`jax.vmap`), so `metric.solve` is only ever
*evaluated*, never transposed. That is what lets an iterative metric solve —
[`metric_from_shifted_matvec`](utilities.md#unified-shifted-block-metrics),
or any hand-written CG-based `Metric.solve` — participate in both JVP and
VJP even though transposing through JAX's CG is unsupported. The normal
forms' final \(S\dot u\) map is *not* self-adjoint and declares the true
transpose pair (`inv_sqrt`/`inv_sqrt_transpose`) instead.

Accuracy of the CG forms is controlled separately from the forward
iterative solve:

- `implicit_tol=None` uses a dtype-aware default (`1e-6` in float32, `1e-10`
  in float64), chosen for derivative accuracy rather than forward-step speed.
  The same attainable-floor bound as any CG applies: the residual stagnates
  near `machine_eps` times the operator's condition number, so at small
  `eps` (spike weight \(c^2/\varepsilon\)) the float64 default is reachable
  only with the spike preconditioner below.
- `implicit_atol=0.0` and `implicit_maxiter=None` are passed to JAX CG.
  `None` leaves the iteration budget to JAX's CG policy.
- `implicit_preconditioner` is deliberately a separate argument from the
  forward preconditioner hooks — see
  [above](#the-implicit-preconditioner).

Two notes for the
[unified shifted metric](gauss_newton.md#shifted-metrics-and-the-seminorm-limit)
\(M = \operatorname{blockdiag}(K, 0) + \varepsilon I\):

- The scalar block injects its rank-\(k\) spike of weight \(c^2/\varepsilon\)
  into the *undamped* implicit dual operator too — there is no LM damping
  here to mask it — so at small \(\varepsilon\) a `gram_cg` implicit solve
  needs the same
  [Sherman–Morrison/Woodbury spike preconditioner](utilities.md#shermanmorrison-dual-preconditioner)
  as the forward solve — pass the helper directly; the implicit hook calls
  it with zero damping.
- With `metric_from_shifted_matvec` the `gram_cg` implicit solve nests an
  inner metric CG inside every operator application, and the outer solve
  cannot be more accurate than the inner one: for derivative-critical work
  set the metric's `tol` at or below `implicit_tol` (the metric's default,
  the square root of machine epsilon, is looser than the float64 implicit
  default of `1e-10`).

Example sketch for a large matrix-free residual:

```python
from nlls_gram import LevenbergMarquardt


def residual(theta, _, p):
    # `features(theta)` is evaluated by JVP/VJP closures; no dense Jacobian is
    # formed by the forward CG solve or the implicit CG derivative.
    return features(theta) - p


def implicit_preconditioner(v):
    return approximate_undamped_dual_solve(v)


solver = LevenbergMarquardt(
    residual,
    linear_solver="gram_cg",
    iterative_tol=1e-3,
    iterative_maxiter=20,
    dual_preconditioner=identity_preconditioner(),
    implicit_solver="auto",
    implicit_tol=None,
    implicit_preconditioner=implicit_preconditioner,
)
```

## VJP

The transpose of the same map gives the VJP. For a cotangent
\(\bar\theta\) on `result.x`, solve

$$
(J_\theta P J_\theta^\top)y = J_\theta P\bar\theta,
\qquad
\bar p = -J_p^\top y.
$$

For a pytree `p`, \(J_p^\top y\) means the JAX VJP of the residual with respect
to the `p` argument only. The normal forms compute the identical cotangent
by transposing the whitened map instead — for the unridged `normal_cg` rule
through its declared push-through transpose, since a cotangent right-hand
side need not lie in \(\operatorname{range}(B^\top)\). Rank-deficient
systems behave as in [the ridge section](#rank-deficiency-and-the-ridge):
the VJP applies the transpose of the same regularized, filtered, or exact
solve the JVP uses.

Example:

```python
import jax
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt


# p without args still uses the three-argument form; the second argument is
# simply ignored.
def residual(theta, _, p):
    return jnp.array([theta[0] + 2.0 * theta[1] - p])


solver = LevenbergMarquardt(residual, init_damping=1e-2)
theta0 = jnp.zeros(2)


def solved_x(p):
    return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).x


theta, theta_dot = jax.jvp(
    solved_x,
    (jnp.asarray(3.0),),
    (jnp.asarray(0.7),),
)

theta, pullback = jax.vjp(solved_x, jnp.asarray(3.0))
(p_bar,) = pullback(jnp.array([3.0, 4.0]))
```

Here \(J_\theta=[1,2]\), so the identity-metric tangent is
\(\dot\theta = [1,2]\dot p / 5\), and the VJP maps
\(\bar\theta\) to \((\bar\theta_0 + 2\bar\theta_1)/5\).

## Aux outputs

With `has_aux=True` the residual returns \((r, a)\); write the aux output map
as

$$
a = g(\theta, \text{args}, p),
\qquad
a = g(\theta^\star, \text{args}, p)
\text{ at the returned solution,}
$$

and define its two Jacobians at the solution (with \(k\) the flattened aux
dimension):

$$
G_\theta = \frac{\partial g}{\partial \theta}(\theta^\star, \text{args}, p)
\in \mathbb R^{k \times n},
\qquad
G_p = \frac{\partial g}{\partial p}(\theta^\star, \text{args}, p)
\in \mathbb R^{k \times \dim p}.
$$

For a pytree aux/`p`, \(G_\theta \dot\theta\) and \(G_p \dot p\) mean the JAX
JVP of the aux output with respect to that argument only (the same convention
as \(J_p \dot p\) above); transposes mean the corresponding VJP.

`p` moves the aux through both paths — directly, and through the solution
\(\theta^\star(p)\) — and the implicit rule accounts for both. **JVP**: with
\(\dot\theta\) the minimum-\(M\)-norm implicit tangent above,

$$
\dot a = G_\theta\,\dot\theta + G_p\,\dot p
= \bigl(-G_\theta P J_\theta^\top (J_\theta P J_\theta^\top)^{-1} J_p
  + G_p\bigr)\,\dot p.
$$

**VJP**: cotangents \(\bar\theta\) on `result.x` and \(\bar a\) on
`result.aux` combine — the aux cotangent pulls back through
\(G_\theta^\top\) into the solution cotangent and through \(G_p^\top\)
directly:

$$
(J_\theta P J_\theta^\top)\, y
= J_\theta P\,(\bar\theta + G_\theta^\top \bar a),
\qquad
\bar p = -J_p^\top y + G_p^\top \bar a.
$$

Setting \(\bar a = 0\) recovers the `x`-only VJP above; setting
\(\bar\theta = 0\) gives the pure aux pullback.

Example — the residual from above with an aux that depends on `p` both ways:

```python
def residual(theta, _, p):
    r = jnp.array([theta[0] + 2.0 * theta[1] - p])
    return r, {"m": theta[0] * theta[1] + p**2}


solver = LevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)


def solved_aux_m(p):
    return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).aux["m"]


da_dp = jax.grad(solved_aux_m)(jnp.asarray(3.0))
```

Here \(\theta^\star = (p/5, 2p/5)\), so the aux value is
\(2p^2/25 + p^2\) and its derivative is \(4p/25 + 2p = 6.48\) at \(p=3\):
the \(4p/25\) comes through the solution path \(G_\theta\dot\theta\) and the
\(2p\) through the direct path \(G_p\dot p\).
