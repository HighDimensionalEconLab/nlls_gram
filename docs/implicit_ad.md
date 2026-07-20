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

In code, \(P x\) is applied with `metric.solve(x)` when available. If a QR
metric is supplied only through square-root callbacks, the same inverse metric is
applied as \(P x = S S^\top x\) using `metric.inv_sqrt` and
`metric.inv_sqrt_transpose`.

With an iterate-dependent metric (`metric_factory=`), the metric is FROZEN at
the returned solution: `prepare`/`build` run once at
`(result.x, result.args, result.p, result.aux)` and the resulting \(P\) is
applied exactly as a fixed metric would be. The state-dependence of the metric
is deliberately not differentiated — the same contract as a fixed metric
closing over constants, matching the forward selection role the metric plays.
The built metric must stay self-adjoint and positive definite for that fixed
state.

## Dense vs Matrix-Free Implicit Solve

The implicit derivative always uses the same undamped residual-space system:

$$
(J_\theta P J_\theta^\top)y = J_p\dot p,
\qquad
\dot\theta = -P J_\theta^\top y.
$$

The solver exposes two concrete implicit solvers plus an automatic selector:

- `implicit_solver="cholesky"` materializes \(J_\theta^\top\), assembles
  \(J_\theta P J_\theta^\top\), and uses a dense Cholesky solve. This is the
  historical rule and is still the explicit escape hatch. The implicit Gram
  has no `+ damping I` floor from the forward solve, but it is regularized by
  a small **relative ridge**, `+ implicit_penalty * trace(J P J') I`
  (default: `1e-12`/`1e-6` for a float64/float32 dual solve; see below) —
  still the most
  conditioning-sensitive solve in the library; `dual_solve_dtype=jnp.float64`
  runs it in float64
  for a float32 model (measured on a \(10^{-7}\)-weight spike metric: a
  float32 implicit tangent wrong by ~5% — and its VJP by ~40% — becomes
  accurate to ~\(10^{-7}\), with the returned tangent still float32).
- `implicit_solver="cg"` applies \(y \mapsto J_\theta P J_\theta^\top y\)
  matrix-free using JAX JVP/VJP closures, then solves with CG wrapped in
  `jax.lax.custom_linear_solve(..., symmetric=True)`. This makes both JVP and
  VJP of `solve(...).x` matrix-free.
- `implicit_solver="auto"` chooses the matrix-free CG rule only when
  `linear_solver="cg"`; otherwise it uses the dense Cholesky rule.

A cg-resolved implicit solve requires an explicit `implicit_preconditioner`
— an approximation of the undamped \((J_\theta P J_\theta^\top)^{-1} v\) —
at construction, even if `solve(...).x` is never differentiated. Note that
the default `implicit_solver="auto"` resolves to CG whenever
`linear_solver="cg"`, so a forward CG solver always needs both callbacks.
The callback may take `(v)` or `(v, damping)`: a callable *requiring* the
damping argument is called with an explicit zero damping (the implicit
system is undamped), and one whose damping has a default passes through
unchanged — so `identity_preconditioner()`,
`sherman_morrison_preconditioner`, `woodbury_preconditioner`, and
`nystrom_preconditioner` all serve both hooks directly. The one exception
is `pad_dual_preconditioner`, which divides by the live damping and is
rejected at construction (implicit AD on padded problems is singular
anyway). Pass `identity_preconditioner()` to run the implicit CG
unpreconditioned, or `implicit_solver="cholesky"` to use the dense rule
instead. The forward `dual_preconditioner` is never reused implicitly: it
approximates the damped operator, and the implicit system is undamped —
reusing one is an explicit choice (pass the same callable to both
arguments).

The matrix-free rule has the same mathematical contract as the dense rule:
\(J_\theta P J_\theta^\top\) must be nonsingular, so \(J_\theta\) must have
full row rank under the chosen metric. The operator is assumed symmetric
positive definite; the metric inverse \(P\) must be linear, self-adjoint, and
positive definite in parameter space, and any `implicit_preconditioner` must be
linear, self-adjoint, and positive definite for the residual-space operator.
The two rules treat rank deficiency differently. The **cg rule** is
intentionally not damped, because damping would change the
minimum-\(M\)-norm derivative: its run-to-tolerance default produces
non-finite derivatives on a singular system (loud), while a small bounded
`implicit_maxiter` returns a finite — and wrong — derivative with no
diagnostic, so reserve the bounded-budget mode for exact preconditioners.
The **dense rule** instead adds a relative Tikhonov ridge,
`implicit_penalty * trace(J P J') I`, before factorization. Redundant
residual rows at the returned solution — e.g. a simulated trajectory that
has settled onto its steady state, so late-horizon rows repeat — make the
undamped dual singular *but consistent*, and for such systems the ridge
tangent converges to the minimum-norm derivative as the penalty shrinks;
the default (`implicit_penalty=None`, resolving to `1e-12` for a float64
dual solve and `1e-6` for float32, after any `dual_solve_dtype` promotion)
therefore returns the min-norm tangent at an O(`implicit_penalty * m`)
relative bias instead of a NaN. The constants are empirical: for
near-duplicate float64 rows the factorization-noise floor sits below
`1e-14` and visible tangent bias above `~1e-6`, so the default is orders
of magnitude from both edges. The trade-off is visibility: on a genuinely *inconsistent* singular
dual the ridge returns a finite, penalty-inflated tangent where the
unregularized rule failed loudly. Pass `implicit_penalty=0.0` to restore
the exact unregularized contract (non-finite tangents on any singular
dual). The cg implicit rule ignores `implicit_penalty` entirely.

Self-adjointness of \(P\) is all the rule needs from `metric.solve`: the
final \(P J_\theta^\top y\) application acts on tangent data, and its
transpose in the VJP is declared to be \(P\) itself
(a symmetric `jax.lax.custom_linear_solve`, which also batches under
`jax.vmap`), so `metric.solve` is only ever
*evaluated*, never transposed. That is what lets an iterative metric solve —
[`metric_from_shifted_matvec`](utilities.md#unified-shifted-block-metrics),
or any hand-written CG-based `Metric.solve` — participate in both JVP and
VJP even though transposing through JAX's CG is unsupported.

Accuracy is controlled separately from the forward iterative solve:

- `implicit_tol=None` uses a dtype-aware default (`1e-6` in float32, `1e-10`
  in float64), chosen for derivative accuracy rather than forward-step speed.
  The same attainable-floor bound as any CG applies: the residual stagnates
  near `machine_eps` times the dual operator's condition number, so at small
  `eps` (spike weight \(c^2/\varepsilon\)) the float64 default is reachable
  only with the spike preconditioner below.
- `implicit_atol=0.0` and `implicit_maxiter=None` are passed to JAX CG.
  `None` leaves the iteration budget to JAX's CG policy.
- `implicit_preconditioner` is the preconditioner for the undamped implicit
  dual system, required whenever the implicit solver resolves to cg. It is
  deliberately a separate argument from `dual_preconditioner`, because the
  forward callback approximates the damped system and reuse should be a
  decision, not a default. Reusing one is now just passing the same
  callable to both arguments: a `(v, damping)` callable is called with an
  explicit zero damping on the implicit side, so no wrapper is needed
  (single-argument `(v)` callables remain valid).

Two notes for the
[unified shifted metric](gauss_newton.md#shifted-metrics-and-the-seminorm-limit)
\(M = \operatorname{blockdiag}(K, 0) + \varepsilon I\):

- The scalar block injects its rank-\(k\) spike of weight \(c^2/\varepsilon\)
  into the *undamped* implicit dual operator too — there is no LM damping
  here to mask it — so at small \(\varepsilon\) the implicit CG needs the
  same
  [Sherman–Morrison/Woodbury spike preconditioner](utilities.md#shermanmorrison-dual-preconditioner)
  as the forward solve — pass the helper directly; the implicit hook calls
  it with zero damping.
- With `metric_from_shifted_matvec` the implicit CG nests an inner metric CG
  inside every operator application, and the outer solve cannot be more
  accurate than the inner one: for derivative-critical work set the metric's
  `tol` at or below `implicit_tol` (the metric's default, the square root of
  machine epsilon, is looser than the float64 implicit default of `1e-10`).

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
    linear_solver="cg",
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
to the `p` argument only.

The residual-space system must be nonsingular; in the intended underdetermined
interpolation setting this means \(J_\theta\) has full row rank under the chosen
metric.

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
