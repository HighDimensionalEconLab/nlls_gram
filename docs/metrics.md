# Metrics

## The Metric Object

A custom metric is passed as a single `metric=Metric(...)` argument. The
`Metric` dataclass holds up to four callbacks that operate on the flattened
parameter vector produced internally by `ravel_pytree`. Let \(P=M^{-1}\), and
let \(S\) satisfy \(SS^\top=M^{-1}\).

| Field | Meaning |
| --- | --- |
| `solve(x)` | \(M^{-1}x = Px\) |
| `norm(x)` | \(\sqrt{x^\top Mx}\) |
| `inv_sqrt(x)` | \(Sx\) |
| `inv_sqrt_transpose(x)` | \(S^\top x\) |

Fields left as `None` (and `metric=None` itself) default to the identity
metric. A fixed `Metric` closes over its data once, at construction; for a
metric that depends on the current iterate or on residual aux outputs, see
[Iterate-Dependent Metrics](#iterate-dependent-metrics-metricfactory) below.

Shape requirements:

- `solve` must support vectors `(n_params,)` and matrices `(n_params, k)`.
- `inv_sqrt` must support vectors `(n_params,)`; matrices
  `(n_params, n_residuals)` are additionally required by the dense AD rule
  (it assembles the whitened Jacobian).
- `inv_sqrt_transpose` must support matrices `(n_params, n_residuals)` for
  the dense whitened forward forms (`normal_cholesky`, `qr`,
  `augmented_qr`) and the dense AD rule; the matrix-free forms
  (`normal_cg`, `lsmr`) apply it to vectors only.
- `norm` only needs to support vectors `(n_params,)`.

Validation rules — which callbacks a custom metric must supply depends on
the *form* the solver works in:

| form | requires |
| --- | --- |
| Gram forward: `gram_cholesky`, `gram_cg` | `solve` |
| whitened forward: `normal_cholesky`, `normal_cg`, `qr`, `augmented_qr`, `lsmr` | `inv_sqrt` + `inv_sqrt_transpose` |
| AD: `dense`, `normal_cg` | `inv_sqrt` + `inv_sqrt_transpose` |
| AD: `gram_cg` | `solve` |
| geodesic acceleration + custom metric | `norm` |

- Concrete forward solver names validate eagerly at construction; a forward
  `"auto"` defers its check to trace time, when the concrete shapes resolve
  the form — the same precedent as a `metric_factory`, whose built metric
  is validated when `build` first runs. The AD side has no shape rule, so
  its requirements are always static: a solve-only metric under any
  `dense`-resolved `ad_solver` fails eagerly — pair such a metric with
  `ad_solver="gram_cg"`.
- If geodesic acceleration is enabled (the default) and a custom metric is
  supplied, `metric.norm` is required — supply it or pass
  `geodesic_acceleration=False`.
- The solver does not infer `norm`, `inv_sqrt`, or `inv_sqrt_transpose` from
  `solve`.
- Every callback must accept and preserve the dtype it is handed. Normally
  that is the residual dtype; with `linear_solve_dtype=jnp.float64` the
  dense pipelines hand the callbacks float64 inputs and expect float64 back
  (jnp-composed callbacks satisfy this automatically through standard
  promotion). `metric_solve_dtype=jnp.float64` instead makes the solver
  wrap the resolved metric so its callbacks *compute* in float64 whatever
  dtype they are handed — see the
  [precision knobs](index.md#precision-knobs).

`norm` is separate because `solve` applies \(M^{-1}\), while the norm needs
\(M\):

$$
\|x\|_M = \sqrt{x^\top Mx}.
$$

Recovering that norm from a black-box \(M^{-1}\) solve would require another
inverse operation. Likewise, a square-root factor \(S\) is not generally
recoverable from an arbitrary solve callback.

## Cholesky Metric Helper

For a dense metric \(M=LL^\top\) with \(L\) lower triangular (the form
returned by `jnp.linalg.cholesky`), use:

```python
import jax.numpy as jnp

from nlls_gram import metric_from_cholesky

L = jnp.linalg.cholesky(metric_matrix)
metric = metric_from_cholesky(L)
```

The helper returns a `Metric` with all four callbacks filled in. Further
constructors — tridiagonal-precision, diagonal, and block-diagonal metrics,
plus a Sherman–Morrison dual preconditioner — are collected in
[Utilities](utilities.md).

## Metric Example

```python
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt, metric_from_cholesky


def residual_fn(theta, args):
    matrix, target = args
    return matrix @ theta - target


metric_matrix = jnp.array([[2.0, 0.2], [0.2, 1.0]])
L = jnp.linalg.cholesky(metric_matrix)

solver = LevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    metric=metric_from_cholesky(L),
)
```

## Matrix-Free Metric Example

For the Gram forms, the metric only needs `solve` (and `norm` for geodesic
acceleration). `metric_from_shifted_matvec` builds \(M = A + \varepsilon I\)
from a matvec alone, running an inner CG to a tight, dtype-aware tolerance:

```python
from nlls_gram import LevenbergMarquardt, metric_from_shifted_matvec

solver = LevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    linear_solver="gram_cg",
    metric=metric_from_shifted_matvec(kernel_matvec, eps),
    dual_preconditioner=identity_preconditioner(),
    ad_solver_preconditioner=identity_preconditioner(),
)
```

It has no matrix-free square root, so it serves the Gram forms only
(`gram_cholesky`, `gram_cg`) — on a square-to-tall problem pin one of them
explicitly, since the default `auto` would resolve to `normal_cholesky` and
reject the metric at trace time. The same constraint applies to the AD
side: pair it with `ad_solver="gram_cg"` (the dense AD rule needs the
square-root pair; a `gram_cholesky` forward with the default
`ad_solver="auto"` would resolve to `dense` and reject it).

This changes the LM damping metric. It is not a preconditioner for the inner CG
iteration; it changes the step being solved for — which is why the inner
solve must run to convergence (a truncated CG is not even a linear function
of its input; never cap `maxiter` as a cost control). See
[Utilities](utilities.md#unified-shifted-block-metrics) for the shift's
role and the exactness caveat.

## Iterate-Dependent Metrics (MetricFactory)

A fixed `Metric` freezes its data at construction. `MetricFactory` instead
rebuilds the metric from the live solve, through a value-hashable
`(prepare, build)` pair passed as `metric_factory=` (pass at most one of
`metric` or `metric_factory`):

- `prepare(x, args, p, aux) -> state` builds a fixed-shape pytree of arrays
  from the current iterate `x` (the user pytree, not the raveled vector),
  the residual `args`, `p`, and the residual aux evaluated at the same
  linearization point (`None` when `has_aux=False`). It runs once per
  accepted step inside the jitted loop; after a rejected step `x` did not
  move, so the carried state is reused. Expensive setup — Gram assembly, a
  dense Cholesky — belongs here, where it is cached.
- `build(state) -> Metric` assembles the metric from the prepared state,
  once per `update` and before the inner iterative loops, so builder-internal
  setup (e.g. the tridiagonal Cholesky scan) runs once per step rather than
  per cg/lsmr iteration. Any `Metric`-returning builder or composition works
  directly.

The canonical use is a factor the residual passes back through its aux
output — the residual computes it primally, the Jacobian never
differentiates it, and the metric consumes it:

```python
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt, MetricFactory, metric_from_cholesky


def residual(x, args, p):
    value = economic_residual(x, args, p)
    gram = kernel_gram(x["points"], p["kernel"])
    L = jnp.linalg.cholesky(gram + p["eps"] * jnp.eye(gram.shape[0]))
    return value, {"L": L}


solver = LevenbergMarquardt(
    residual,
    has_aux=True,
    metric_factory=MetricFactory(
        prepare=lambda x, args, p, aux: aux["L"],
        build=metric_from_cholesky,
    ),
)
```

`prepare` can equally compute from the iterate directly (no `has_aux`
needed), and `build` can compose the structural constructors — a
block-diagonal kernel-plus-ridge metric, a state-space Matérn metric on
moving points, or `metric_with_compute_dtype` for a float64 metric state
under a float32 residual:

```python
factory = MetricFactory(
    prepare=lambda x, args, p, aux: jnp.sort(x["points"]),
    build=lambda pts: metric_from_state_space(pts, h, Pinf, transition),
)
```

Rules and semantics:

- The built `Metric` obeys the same per-solver callback requirements and
  shape contract as a fixed custom metric; validation runs when `build`
  first executes (at trace time), not at construction.
- The metric defines the subproblem, so `build`'s callbacks must stay exact
  (unlike a preconditioner, which may approximate).
- Within one `update`, the velocity, geodesic-acceleration, and norm
  applications all use the same pre-step state.
- A `solve` callback that replaces `x` or `args` invalidates the carried
  state, and multi-start draws never inherit another start's state.
- With `has_aux=True`, `jax.linearize(..., has_aux=True)` keeps aux primal:
  an aux-only factorization costs one primal evaluation per linearization
  and contributes nothing to the Jacobian's JVP/VJP passes — no
  `stop_gradient` is needed.
- Under implicit differentiation of `solve` with respect to `p`, the metric
  is frozen at the returned solution: `prepare`/`build` run once at
  `(result.x, result.args, result.p, result.aux)` and the state-dependence
  is not differentiated. The freeze is a first-order statement — and
  higher-order AD through a factory-built metric's state dependence is
  unsupported in the implicit rules; take higher-order derivatives of
  `solve` only with a fixed metric. See
  [Implicit differentiation](implicit_ad.md#iterate-dependent-metrics-are-frozen-per-solve).
- Like every jit-static hook, define the `(prepare, build)` pair once at
  setup scope; a fresh closure per call keys a new compilation.
