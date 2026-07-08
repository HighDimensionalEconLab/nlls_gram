# nlls_gram

`nlls_gram` provides metric-aware Levenberg-Marquardt nonlinear least-squares for
JAX pytrees. It is designed for underdetermined or interpolating problems where
the parameter dimension is often larger than the residual dimension.

The solver is intentionally small: users provide `residual_fn(x, args, p)`,
and `UnderdeterminedLevenbergMarquardt` exposes `init()`, `update(...)`, and
`solve(...)`. It does not depend on Flax, NNX, Optax, or any model framework.

## Install

```bash
uv add nlls-gram
```

For accelerator use, install the JAX build matching your hardware alongside
`nlls-gram`, for example:

```bash
uv add nlls-gram "jax[cuda13]"
```

## Residual and Data Interface

The residual function takes one, two, or three positional arguments — always
in this order:

```python
residual_fn(x)          # closes over its data
residual_fn(x, args)
residual_fn(x, args, p)
```

- `x` is the pytree optimized by LM.
- `args` is arbitrary auxiliary data passed to `update(...)` or `solve(...)`.
- `p` is optional read-only data, useful for fixed deep parameters or outer
  perturbations; the implicit differentiation of `solve` is with respect to
  `p`, so that requires the three-argument form.

`args` and `p` may be any JAX pytree. The arity is inspected once at
construction and the function is wrapped into the canonical three-argument
form, so the compiled code is identical for all three; callables whose
signature cannot be inspected (or that take `*args`) are assumed to take all
three. The order is fixed: a two-argument residual always means
`(x, args)` — to use `p` without `args`, write the three-argument form and
ignore the second argument. Passing `args` or `p` to `update`/`solve` when the
residual does not accept it raises a `ValueError` rather than silently
dropping it.

`update(x, lm_state, args=None, p=None)` performs one LM step. The higher-level
`solve(x0, args=None, *, p=None, ...)` loop repeatedly calls `update` and
returns an `LMSolveResult`.

### Auxiliary Outputs (`has_aux`)

With `has_aux=True` (the same convention as Optimistix and `jax.grad`), the
residual returns a pair `(residual, aux)`, where `aux` is an arbitrary pytree
of extra outputs the optimizer ignores — per-block diagnostics, validation
metrics, anything already computed inside the residual:

```python
def residual_fn(x, args):
    r = model(x, args)
    return r, {"max_abs": jnp.max(jnp.abs(r))}


solver = UnderdeterminedLevenbergMarquardt(residual_fn, has_aux=True)
```

Each step's `LMInfo.aux` holds the aux from the residual evaluation at the
**pre-step** `x` (the linearization point — same convention as `loss_old`
and `grad_norm`), at no extra cost; callbacks read it as `ctx.info.aux`,
e.g. to early-stop on a diagnostic. The solve result additionally carries
`result.aux`: the aux evaluated at the returned `(result.x, result.args, result.p)`
with one extra residual evaluation after the loop. This is well-defined for
every status — the returned `x` is always the last accepted iterate — so it
holds the final diagnostics whether or not the solve converged, and it
participates in the implicit differentiation with respect to `p` (see
[Aux outputs](implicit_ad.md#aux-outputs)). With
`has_aux=False`, both are `None` and nothing is added to the compiled
program.

Callbacks passed to `solve` can replace `args` for later iterations, for example
to regenerate collocation points or refresh simulation data. They receive `p`
but cannot replace it; this keeps the optimized solution's dependence on
external parameters explicit for implicit differentiation.

## Mathematical Contract

At a parameter vector \(\theta\), let

- \(r \in \mathbb R^m\) be the flattened residual,
- \(J \in \mathbb R^{m \times n}\) be the Jacobian with respect to the flattened
  parameter vector,
- \(s \in \mathbb R^n\) be the proposed step,
- \(\lambda > 0\) be the current damping scalar,
- \(M \succ 0\) be the parameter-space metric.

Each linear solver targets the same metric-damped LM subproblem:

$$
\min_s \frac12\|r + Js\|_2^2 + \frac{\lambda}{2}s^\top M s.
$$

The normal equations are

$$
(J^\top J + \lambda M)s = -J^\top r.
$$

The default metric is \(M=I\), which recovers the usual Euclidean LM damping.
For kernel coefficient problems,

$$
f_\alpha(x)=\sum_{j=1}^n \alpha_j K(x,x_j)
$$

has RKHS norm

$$
\|f_\alpha\|_{\mathcal H_K}^2 = \alpha^\top K\alpha,
$$

so the natural parameter metric is \(M=K\).

The actual nonlinear step is accepted only if the unregularized residual sum of
squares decreases. On acceptance, damping is multiplied by `damping_decrease`;
on rejection, it is multiplied by `damping_increase`.

Near the interpolation threshold, small-damping LM becomes metric
Gauss-Newton, whose step is the minimum-\(M\)-norm solution of the linearized
residual equations; large damping is steepest descent in the \(M\)-metric.
[Metric Gauss-Newton and Minimum-Norm Steps](gauss_newton.md) derives both
limits, the spectral-filter view, and the kernel/RKHS metric choices.

## Metrics

A custom positive-definite parameter-space metric is passed as a single
`metric=Metric(...)` argument. See [Metrics](metrics.md) for the `Metric`
callback contract and validation rules, the `metric_from_cholesky` helper,
and dense and matrix-free examples.

## Linear Solver Formulas

### Cholesky

The default `linear_solver="cholesky"` materializes \(J^\top\), applies
`metric.solve`, and factors the residual-space dual system

$$
(J P J^\top + \lambda I)y = r,
\qquad s = -P J^\top y.
$$

This is usually the fastest dense path when \(m \ll n\).

### CG

`linear_solver="cg"` uses the same residual-space dual system as Cholesky, but
applies it matrix-free:

$$
u \mapsto J P J^\top u + \lambda u.
$$

It uses JAX linearization for JVPs/VJPs and `jax.scipy.sparse.linalg.cg`.

### QR

`linear_solver="qr"` uses the square-root metric form. With \(s = Sz\), solve

$$
\min_z \frac12\|r + JSz\|_2^2 + \frac{\lambda}{2}\|z\|_2^2.
$$

The implementation materializes \(S^\top J^\top\) via
`metric.inv_sqrt_transpose(J.T)` and solves the resulting augmented QR problem.
The returned parameter step is mapped back with `metric.inv_sqrt`.

The triangular solves require \(S^\top J^\top\) to have full column rank
(equivalently, \(J\) full row rank): a rank-deficient Jacobian produces a
non-finite step even though the damped subproblem remains well-posed. Use
`cholesky`, `cg`, or `lsmr` for rank-deficient problems.

### LSMR

`linear_solver="lsmr"` solves the same whitened-step problem as QR using
Lineax LSMR and a `lineax.FunctionLinearOperator`:

$$
z \mapsto
\begin{bmatrix}
JSz \\
\sqrt{\lambda}z
\end{bmatrix}.
$$

It returns \(s=Sz\). Iterative solvers default to a small fixed budget:
`iterative_tol=0.0`, `iterative_atol=0.0`, and `iterative_maxiter=8`.

## Minimal Example

```python
import jax
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt


def residual_fn(x, args):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


ts = jnp.linspace(0.0, 2.0, 20)
ys = 2.0 * jnp.exp(-1.0 * ts)
x = {"a": 1.0, "b": 0.0}

solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init(x, (ts, ys))


@jax.jit
def train_step(x, lm_state):
    return solver.update(x, lm_state, (ts, ys))


for _ in range(50):
    x, lm_state, info = train_step(x, lm_state)

print(x["a"], x["b"])
```

## Solve Loop Callbacks

`solve` accepts a traceable callback that can stop early, replace `args` and
`user_state` for later iterations, and reset damping and other per-step
hyperparameters mid-solve. See [Callbacks and Cookbook](callbacks.md) for the
callback contract, resettable hyperparameters, and recipes: host loops,
logging, divergence stops, epoch resampling, scheduled inner-solve accuracy,
validation early stopping, wall-clock time limits, and fixed-size history
recording.

## Performance Notes

- The solver instance and the callback are static arguments of the internal
  jitted loop, keyed by object identity. Construct the solver once and define
  callbacks at setup scope; an inline `lambda` at the call site recompiles
  every `solve`.
- `max_steps`, `atol`, `gtol`, and `xtol` are traced values: sweeping them
  does not recompile (concrete numbers, not tracers).
- Each `update` costs one residual linearization, one Jacobian
  materialization (`n_residuals` VJP passes in the dense paths), and one
  candidate residual evaluation; geodesic acceleration adds a
  forward-over-forward directional derivative and, only when the ratio gate
  passes, one more residual evaluation.
- Dtypes flow from `x` and the residual; keep everything in one dtype to
  avoid promotions.

### Jacobian Caching Across Rejected Steps

A rejected LM step leaves the parameters unchanged, so the next update's
residual and Jacobian are identical — only the damping changed. With
`cache_jacobian=True` (the default) the solver carries `(resid, Jt)` in
`LMState` and a
rejected step's successor skips the residual evaluation and the
`n_residuals` VJP passes, re-solving only the small damped system (roughly
2x faster per rejected step; more when the residual is expensive relative to
the Gram assembly). The flag only affects `linear_solver="cholesky"` — the
matrix-free solvers never materialize a Jacobian, so it is ignored for them.

```python
solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init(x0, args)  # x0 required to size the cache
```

Caveats:

- The cache is valid only while `x`, `args`, and `p` are all unchanged.
  Inside `solve`, invalidation is automatic: an accepted step invalidates it,
  and so does any callback action that actually changes the values of `x` or
  `args` — the comparison is by value, so the `jnp.where` recipe pattern that
  returns unchanged values every step keeps the cache, and epoch resampling
  needs no extra care. The one hazard is a **manual `update()` loop** that
  changes `args` or `p` between steps (minibatching): the solver cannot see
  the swap and a stale cache fails silently, with steps taken against the old
  Jacobian — leave the cache off there, or reset with
  `dataclasses.replace(lm_state, jacobian_valid=jnp.asarray(False))`.
- The cache adds an `(n_params, n_residuals)` buffer to `LMState` for the
  whole solve — relevant on GPU memory budgets.
- Callbacks that rebuild the lm_state must preserve the cache fields — use
  `dataclasses.replace(ctx.lm_state, damping=...)`, not a bare
  `LMState(damping)`.
- When rejections never happen the cache costs essentially nothing at run
  time, which is why it is on by default; pass `cache_jacobian=False` for
  manual minibatch loops (the hazard above) or when the buffer does not fit
  GPU memory.

## Implicit Differentiation

`solve` has a custom implicit JVP/VJP with respect to the external parameter
pytree `p`: derivatives of `result.x` (and, with `has_aux=True`, of
`result.aux`) are defined by the residual equation at the returned solution
rather than by differentiating through the LM iterations. See
[Implicit Differentiation](implicit_ad.md) for the math, the role of the
metric in selecting the minimum-norm tangent, and worked examples.

## Geodesic Acceleration

Geodesic acceleration is on by default (`geodesic_acceleration=False`
disables it). The solver computes the second-order residual directional term
with JAX forward-over-forward JVPs; the same metric-damped linear solve
computes the acceleration.

With a custom metric, the acceptance ratio uses the metric norm:

$$
\frac{2\|a\|_M}{\|v\|_M + \epsilon}.
$$

Therefore `metric.norm` is required whenever geodesic acceleration is combined
with a custom metric.

## Dtypes and Pytrees

Dtypes flow from `x` and the residual: every internal scalar (damping,
damping factors, tolerances, metric quantities) is cast to the residual dtype,
so a float32 problem computes purely in float32 and a float64 problem purely
in float64 — no example needs explicit dtypes beyond its data. Enable x64
before creating arrays if the problem should run in float64:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

`init(x0, args, p=...)` mirrors `update`'s data arguments: it evaluates the
residual once and types the lm_state (and any Jacobian cache buffers) from the
actual problem, so there is no dtype argument to get wrong — a float32
problem stays float32 even with x64 enabled. `solve` likewise recasts the
damping and tolerances to the residual dtype internally. One known exception:
`linear_solver="lsmr"` fails for float32 problems when x64 is enabled, due to
a dtype-promotion bug inside Lineax's LSMR; the other three solvers handle
that mixed configuration.

`update` optimizes exactly the `x` pytree you pass. With Flax NNX, pass only
the trainable state to the solver and merge it with frozen state inside
`residual_fn`; the solver itself remains NNX-agnostic.

## Benchmarks and GPU Checks

Optional pytest-benchmark checks live outside the default test suite:

```bash
uv run --group benchmark pytest benchmarks --benchmark-only
```

For the larger interpolation profile:

```bash
uv run --group benchmark --group gpu pytest \
  benchmarks/test_large_interpolation_benchmark.py --benchmark-only
```

On CUDA machines, the optional GPU test is:

```bash
uv run --group gpu pytest tests/test_gpu.py
```

## Optimistix

For a broader JAX nonlinear solver library, see
[Optimistix](https://github.com/patrick-kidger/optimistix). It provides general
least-squares, root-finding, and minimization abstractions. `nlls_gram` is
narrower and focuses on underdetermined LM with explicit parameter-space metrics.

## API Reference

::: nlls_gram.UnderdeterminedLevenbergMarquardt

::: nlls_gram.Metric

::: nlls_gram.metric_from_cholesky

::: nlls_gram.LMState

::: nlls_gram.LMHyperparams

::: nlls_gram.LMInfo

::: nlls_gram.LMStatus

::: nlls_gram.LMSolveAction

::: nlls_gram.LMSolveContext

::: nlls_gram.LMSolveResult
