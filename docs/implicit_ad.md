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

### Status policy and the initial AD point

Implicit AD is defined for `LMStatus.CONVERGED` and, by default,
`LMStatus.MAX_STEPS`. The forgiving default supports fixed-step solves: a
result that exhausts its budget relinearizes at the returned
`(result.x, result.args, result.p)`, while its diagnostic status remains
`MAX_STEPS`. Pass `max_steps_is_success=False` to make `MAX_STEPS` strict: it
then follows the same failed-AD path as `NONFINITE`, `CALLBACK_STOP`, or any
other non-`CONVERGED` status.

An AD-successful solve relinearizes at its returned values. A failed solve
keeps the primal result, status, aux, and diagnostics unchanged but returns
exactly zero tangents through `result.x` and `result.aux`. `result.p` remains
an independent identity pass-through.

The failed lane's linear tangent program is evaluated at stop-gradient copies
of the caller's original `(x0, args, p)`, with the `p` tangent masked to zero
before the implicit solve. This gives automatic transposition a finite linear
program even when the returned failed iterate is nonfinite, and keeps a failed
lane from poisoning successful lanes under `vmap`. These initial values are a
safety point, not a fallback solution: the returned failed iterate is never
replaced.

The original initial point must therefore be **JVP-safe**, not merely finite.
The residual and aux map must have valid derivatives there. A `MetricFactory`
must also be JVP-safe there when the resolved non-direct AD method consumes its
metric, and a `PreconditionerFactory` must be JVP-safe there when it supplies a
`gram_cg` AD preconditioner. A direct square AD solve ignores both factories,
so it does not rebuild aux for them. When a selected multi-start winner fails
the implicit-AD status policy, the safety point is the caller's original
`(x0, args, p)`, not one of the drawn starts; custom acceptance is independent.

Aux is reevaluated for factory construction only when it is actually consumed:
a failed `MetricFactory` lane under a non-direct AD method, or a failed
`gram_cg` lane whose `PreconditionerFactory` supplies the AD preconditioner.
Fixed metrics, explicit AD preconditioners, and direct square solves do not
trigger that factory-only rebuild. An AD-successful lane always reuses
`result.aux` for factory construction. Independently, when `has_aux=True`, the
aux map is linearized once at the selected point to compute the returned aux
tangent itself.

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
static object — do not build it from traced values. Constructing
`LevenbergMarquardt` *inside* a jitted function is supported: every
constructor argument is a static Python value (option strings, floats,
hooks) and the solver holds no arrays, so equal configurations share one
compilation. What must not happen is feeding a traced value into the
constructor, or rebuilding the *pieces* per call — an inline `lambda`
residual or a hook closed over fresh arrays keys a new compile by object
identity (see
[what is free to sweep](tuning_guide.md#what-is-free-to-sweep)).

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
(J_\theta P J_\theta^\top)^{+}
J_p\dot p.
$$

In whitened variables — \(B = J_\theta S\) with \(S S^\top = P\) and
\(\dot\theta = S\dot u\) — the same tangent is the minimum-norm solution of

$$
B^\top B\,\dot u = -B^\top J_p\dot p,
\qquad
\dot u = -B^{+} J_p\dot p,
\qquad
\dot\theta = S\,\dot u,
$$

using \((B^\top B)^{+}B^\top = B^{+}\). In code, the **dense** AD rule
computes exactly this: it materializes \(B\) and applies \(B^{+}\) from its
SVD. The **gram_cg** rule solves the \(m \times m\) dual system above
matrix-free, applying \(P x\) with `metric.solve(x)` when available, else as
\(P x = S S^\top x\) through the square-root callbacks; the **normal_cg**
rule solves the whitened normal system matrix-free through
`metric.inv_sqrt`/`inv_sqrt_transpose`.

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
the metric rebuilt at that point's solution. But **higher-order AD through
a factory-built metric's state dependence is unsupported in every AD rule**:
the metric-aware assembled and CG methods alike apply the frozen metric through
identity-matvec `custom_linear_solve` wrappers whose declared solves are
opaque to AD, so second derivatives do not account for the metric's point
dependence. Take higher-order derivatives of `solve` only with a fixed
metric (subject also to the spectral-filter caveat in
[the rank-deficiency section](#rank-deficiency-and-the-ridge)). Those same wrappers
declare each metric's transpose explicitly, so first-order reverse mode is
correct even when `inv_sqrt` is not reverse-differentiable but
`inv_sqrt_transpose` is supplied.

## The AD Solver

`ad_solver` selects how the implicit tangent and cotangent systems are
solved. Each method name selects exactly one algorithm:

- **`direct`** assembles the general square Jacobian and solves
  \(J_\theta\dot\theta=-J_p\dot p\) with `jnp.linalg.solve`. It is the
  inexpensive choice for a square, nonsingular root and gives the correct
  transpose solve in VJP even when \(J_\theta\) is nonsymmetric. A unique
  square root has no metric-dependent tangent selection, so `direct`
  intentionally ignores the metric. It rejects nonsquare systems.
- **`svd`** assembles the whitened Jacobian \(B=J_\theta S\) and applies its
  spectral pseudoinverse. It is the robust dense choice for rank-deficient or
  nearly rank-deficient problems and returns the minimum-\(M\)-norm tangent.
- **`qr`** factors the same \(B\) without regularization. It is cheaper than
  SVD at full numerical rank and fails loudly with a NaN tangent when the
  rank guard detects deficiency.
- **`augmented_qr`** factors
  \([B;\sqrt{\delta}I]\), where
  \(\delta=\texttt{ad\_solver\_penalty}\operatorname{tr}(B^\top B)\).
  This is the explicitly regularized dense method.
- **`gram_cg`** applies \(J_\theta P J_\theta^\top\) matrix-free and solves
  the residual-space system with CG.
- **`normal_cg`** applies \(B^\top B\) matrix-free. Its tangent right-hand
  side lies in \(\operatorname{range}(B^\top)\), so CG from zero selects the
  minimum-norm tangent without a ridge.
- **`regularized_normal_cg`** is the explicit matrix-free regularized method.
  It scales its ridge by a Rayleigh quotient of \(B^\top B\) over a fixed
  deterministic probe so the tangent remains linear in \(\dot p\).

The default `ad_solver="auto"` dispatches from traced shapes first:

| traced system / forward `linear_solver` | `auto` resolves to |
| --- | --- |
| square, for every forward solver | `direct` |
| nonsquare with `gram_cg` forward | `gram_cg` |
| nonsquare with `normal_cg` forward | `normal_cg` |
| any other nonsquare system | `svd` |

The shape-first rule matters: a square algebraic system gets the direct
solve even if its primal step used CG. The AD method is otherwise independent
of the forward solver, so an `lsmr` forward with explicit
`ad_solver="normal_cg"` remains matrix-free end to end.

The SVD and QR methods work from \(B\), never from a formed squared operator,
so their numerical accuracy is governed by \(\operatorname{cond}(B)\), not
\(\operatorname{cond}(B)^2\). They assemble the Jacobian from its small side;
see [`jacobian_mode`](tuning_guide.md#jacobian-assembly-jacobian_mode).
`direct`, `svd`, `qr`, and `augmented_qr` inherit `linear_solve_dtype`.

For `normal_cg`, reverse mode cannot simply reuse CG on a possibly singular
normal operator with an arbitrary cotangent. Its declared transpose uses the
push-through identity \(N^+=B^\top(BB^\top)^{+2}B\), which keeps both dual
solves consistent. The final metric-square-root application declares
`inv_sqrt_transpose` explicitly, so triangular and other non-self-adjoint
square roots transpose correctly.

Both CG forms assume the inner solve **converges**.
`jax.lax.custom_linear_solve` transposes the declared solve as an exact
linear map, but a truncated or early-stopped CG (a small bounded
`ad_solver_maxiter`) is not a linear function of its right-hand side — its
declared transpose then no longer matches what was computed, and
derivatives are silently inaccurate on top of the truncation error itself.
Run the AD CG solves to tolerance (the default
`ad_solver_maxiter=None`), and treat a bounded budget as valid only with a
preconditioner exact enough to converge within it.

The undamped implicit system is often the most conditioning-sensitive solve in
the library. `linear_solve_dtype=jnp.float64` promotes the assembled AD methods
while leaving model inputs and returned tangents at their original dtype.

### The AD Solver Preconditioner

An explicit `gram_cg` AD method requires an `ad_solver_preconditioner` at
construction. With `ad_solver="auto"`, the requirement is checked when a
differentiated nonsquare forward `gram_cg` solve is traced; a square system
has already resolved to `direct`. The hook approximates the undamped
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
the dual-space factory can never serve this parameter-space system. The
assembled methods use no preconditioner.

The callback may take `(v)` or `(v, damping)`: a callable *requiring* the
damping argument is called with an explicit zero damping (the implicit
system is undamped), and one whose damping has a default passes through
unchanged — so `identity_preconditioner()`,
`sherman_morrison_preconditioner`, `woodbury_preconditioner`, and
`nystrom_preconditioner` all serve the hook directly. The one exception
is `pad_dual_preconditioner`, which divides by the live damping and is
rejected at construction. Pass `identity_preconditioner()` to run the
AD CG unpreconditioned, or `ad_solver="svd"` to use the assembled
pseudoinverse instead. The forward `dual_preconditioner` is never reused for AD:
it approximates the damped operator, and the implicit system is undamped —
reusing one is an explicit choice (pass the same callable to both
arguments). Any `ad_solver_preconditioner` must be linear, self-adjoint, and
positive definite for the operator of its resolved space, and the metric
inverse \(P\) must be linear, self-adjoint, and positive definite in
parameter space.

## Rank Deficiency and the Ridge

One invariant holds package-wide: **no default applies a ridge to an implicit
tangent solve.** Regularization requires an explicit algorithm name and an
explicit positive penalty. The primal solve's only regularizer remains the LM
damping itself.

The implicit system is intentionally not damped — damping would change the
minimum-\(M\)-norm derivative. What happens on a rank-deficient system
depends on consistency, and interpolation problems produce **consistent**
linearized systems by construction: differentiating the root identity
\(r(\theta^\star(p), a, p) = 0\) along the solution branch gives
\(J_\theta\dot\theta^\star + J_p\dot p = 0\), i.e.
\(J_p\dot p \in \operatorname{range}(J_\theta)\) — redundant rows and
collinear columns included. The methods behave as follows:

- **`direct`** is exact for a nonsingular square Jacobian and fails through the
  underlying solve when the Jacobian is singular.
- **`svd`** applies \(B^+\) at the standard cutoff
  \(\max(m,n)\,\mathrm{eps}\,\sigma_{\max}\). This is the robust choice for
  a consistent rank-deficient system. The cutoff is numerical, so meaningful
  singular values below it are treated as null directions. Higher derivatives
  are exact only while the active singular subspace is locally constant.
- **`qr`** is exact at full numerical rank and returns NaNs when its rank guard
  detects deficiency.
- **`augmented_qr`** is smooth across rank changes but biased by
  O(`ad_solver_penalty` · dimension). Typical starting penalties are `1e-12`
  in float64 and `1e-6` in float32.
- **`gram_cg`** is unregularized and can fail on a singular dual even when the
  primal tangent equation is consistent. A small fixed iteration budget can
  instead return a finite but inaccurate answer, so run it to tolerance.
- **`normal_cg`** remains unregularized and converges to the minimum-norm
  tangent on singular-but-consistent systems in exact arithmetic.
- **`regularized_normal_cg`** supplies the explicit matrix-free ridge.

`ad_solver_penalty` is required and must be positive for `augmented_qr` and
`regularized_normal_cg`. It is rejected for every other method, including
`auto`; it never switches algorithms.

**Gauss–Newton semantics on inconsistent systems.** When the returned
point is a least-squares stationary point with nonzero residual rather than
an exact root, \(J_p\dot p\) need not lie in
\(\operatorname{range}(J_\theta)\). The default rules still return a finite
answer — the minimum-\(M\)-norm **Gauss–Newton sensitivity of the
linearized system**: the normal equations
\(B^\top B\dot u = -B^\top J_p\dot p\) are consistent regardless (any
\(B^\top\)-image is), and \(B^{+}\) applies their minimum-norm solution
directly. That tangent is *not* the exact optimizer
sensitivity — the exact derivative of a nonzero-residual stationary point
carries residual-weighted second-derivative (curvature) terms that the
frozen-Jacobian system drops. A one-line counterexample:
\(r(x, p) = (x - p,\ x^2 - 1)\) at the stationary point \(x = 0.5\),
\(p = -0.25\) has exact sensitivity \(dx/dp = 2\), while the Gauss–Newton
normal system gives \(0.5\). The implicit rules are exact at interpolating
(zero-residual) roots — the package's target — and Gauss–Newton
approximations away from them. The non-default modes differ: an opt-in
ridge returns a penalty-inflated finite tangent, and
`gram_cg` fails loudly (its Krylov solve faces the singular-inconsistent
dual).

Metric callback requirements follow the factorization each form uses.
`svd`, `qr`, `augmented_qr`, `normal_cg`, and `regularized_normal_cg` work in
whitened variables and need the
square-root pair `metric.inv_sqrt`/`inv_sqrt_transpose`; the final
\(S\dot u\) map is not self-adjoint and declares that true transpose pair,
so triangular Cholesky factors differentiate correctly in reverse mode.
`direct` does not use the metric. `gram_cg` needs only `metric.solve`, and only ever *evaluates* it: the
final \(P J_\theta^\top y\) application acts on tangent data, and its
transpose in the VJP is declared to be \(P\) itself (a symmetric
`jax.lax.custom_linear_solve`, which also batches under `jax.vmap`). That
is what lets a hand-written iterative, solve-only `Metric.solve` participate
in both JVP and VJP even though transposing through JAX's CG is unsupported:
pair such a metric with `ad_solver="gram_cg"`.

Accuracy of the CG forms is controlled separately from the forward
iterative solve:

- `ad_solver_tol=None` uses a dtype-aware default (`1e-6` in float32, `1e-10`
  in float64), chosen for derivative accuracy rather than forward-step speed.
  The same attainable-floor bound as any CG applies: the residual stagnates
  near `machine_eps` times the operator's condition number, so at small
  `eps` (spike weight \(c^2/\varepsilon\)) the float64 default is reachable
  only with the spike preconditioner below.
- `ad_solver_atol=0.0` and `ad_solver_maxiter=None` are passed to JAX CG.
  `None` leaves the iteration budget to JAX's CG policy.
- `ad_solver_preconditioner` is deliberately a separate argument from the
  forward preconditioner hooks — see
  [above](#the-ad-solver-preconditioner).

For the
[repeated shifted metric](gauss_newton.md#shifted-metrics-and-the-seminorm-limit)
\(M = \operatorname{blockdiag}(K, 0) + \varepsilon I\):

- The scalar block injects its rank-\(k\) spike of weight \(c^2/\varepsilon\)
  into the *undamped* implicit dual operator too — there is no LM damping
  here to mask it — so at small \(\varepsilon\) a `gram_cg` AD solve
  needs the same
  [Sherman–Morrison/Woodbury spike preconditioner](utilities.md#shermanmorrison-dual-preconditioner)
  as the forward solve — pass the helper directly; the AD hook calls
  it with zero damping.
Example sketch for a large matrix-free residual:

```python
from nlls_gram import LevenbergMarquardt


def residual(theta, _, p):
    # `features(theta)` is evaluated by JVP/VJP closures; no dense Jacobian is
    # formed by the forward CG solve or the CG-based AD rule.
    return features(theta) - p


def ad_solver_preconditioner(v):
    return approximate_undamped_dual_solve(v)


solver = LevenbergMarquardt(
    residual,
    linear_solver="gram_cg",
    iterative_tol=1e-3,
    iterative_maxiter=20,
    dual_preconditioner=identity_preconditioner(),
    ad_solver="auto",
    ad_solver_tol=None,
    ad_solver_preconditioner=ad_solver_preconditioner,
)
```

## VJP

JAX derives the VJP by automatically transposing this solver's linear custom
JVP rule; there is no separate custom VJP or cotangent solver. Writing
\(K=J_p\), \(G=J_\theta P J_\theta^\top\), and using the same pseudoinverse or
filtered solve as the JVP, the maps are

$$
\dot\theta=-P J_\theta^\top G^+ K\dot p,
\qquad
\bar p=-K^\top G^+ J_\theta P\bar\theta.
$$

Equivalently, for a cotangent \(\bar\theta\) on `result.x`, solve

$$
(J_\theta P J_\theta^\top)y = J_\theta P\bar\theta,
\qquad
\bar p = -J_p^\top y.
$$

For the square direct method, the JVP solves
\(J_\theta\dot\theta=-K\dot p\). Its automatically transposed VJP solves
\(J_\theta^\top\lambda=\bar\theta\) and returns
\(\bar p=-K^\top\lambda\).

For a pytree `p`, \(J_p^\top y\) means the JAX VJP of the residual with respect
to the `p` argument only. The whitened rules compute the identical cotangent
by transposing the whitened map instead — the `svd` rule through
\((B^{+})^\top\) from the same factorization, the unridged `normal_cg` rule
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
= \bigl(-G_\theta P J_\theta^\top (J_\theta P J_\theta^\top)^{+} J_p
  + G_p\bigr)\,\dot p.
$$

**VJP**: cotangents \(\bar\theta\) on `result.x` and \(\bar a\) on
`result.aux` combine — the aux cotangent pulls back through
\(G_\theta^\top\) into the solution cotangent and through \(G_p^\top\)
directly:

$$
y=(J_\theta P J_\theta^\top)^+
J_\theta P\,(\bar\theta + G_\theta^\top \bar a),
\qquad
\bar p = -J_p^\top y + G_p^\top \bar a.
$$

Setting \(\bar a = 0\) recovers the `x`-only VJP above; setting
\(\bar\theta = 0\) gives the pure aux pullback.

On a failed solve both \(\dot\theta\) and \(\dot a\) are masked to exact zero
after this tangent program is evaluated at the original JVP-safe initial
point. Automatic transposition therefore contributes zero implicit cotangents
from `result.x` and `result.aux`, while any cotangent on the independently
returned `result.p` still passes through unchanged.

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
