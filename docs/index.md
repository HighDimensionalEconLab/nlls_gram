# nlls_gram

`nlls_gram` provides metric-aware Levenberg-Marquardt nonlinear least-squares for
JAX pytrees, aimed at solving systems of equations (i.e., interpolation). The
core use cases are underdetermined systems — more parameters than residuals —
where LM is solved in its Gram form, and square or redundant tall nonlinear
systems where regularization is needed to select among the interpolating
solutions, with the selection controlled by a user-chosen metric.

The solver is intentionally small: users provide `residual_fn(x, args, p)`,
and `LevenbergMarquardt` exposes `init()`, `update(...)`, and
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
returns an `LMSolveResult`. Pass `save_steps=True` to also record the full
iterate history on the result (`x_history`, plus the row-aligned
`args_history` and, with `has_aux=True`, `aux_history`) — see
[history recording](callbacks.md#fixed-size-history-recording) for the layout
and a plotting recipe. Pass `multi_start=MultiStart(...)` to retry failed
solves from fresh initial conditions or race several starts under `vmap`,
returning the single best result — see [multi-start](multi_start.md).

### Auxiliary Outputs (`has_aux`)

With `has_aux=True` (the same convention as Optimistix and `jax.grad`), the
residual returns a pair `(residual, aux)`, where `aux` is a pytree of extra
outputs the optimizer ignores — per-block diagnostics, validation metrics,
anything already computed inside the residual. Its structure is arbitrary,
but every leaf must be a JAX numeric type (an array, or a Python/NumPy
scalar), with fixed shape and dtype across iterations like any jitted value:
aux rides through the jitted solve loop and the implicit differentiation
rule, so a non-numeric leaf such as a string raises a `TypeError` at the
first evaluation. For example:

```python
def residual_fn(x, args):
    r = model(x, args)
    return r, {"max_abs": jnp.max(jnp.abs(r))}


solver = LevenbergMarquardt(residual_fn, has_aux=True)
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
and dense and matrix-free examples. For a metric that depends on the current
iterate or on residual aux outputs, pass a
`metric_factory=MetricFactory(prepare, build)` instead — the state is
rebuilt once per accepted step and `build` returns a plain `Metric`
([Iterate-Dependent Metrics](metrics.md#iterate-dependent-metrics-metricfactory)).

## Linear Solver Formulas

`linear_solver` selects among eight forms. Two families share the dense and
matrix-free slots: the **Gram** forms work in residual space on the
\(m \times m\) dual \(J P J^\top + \lambda I\), the **normal** forms in
parameter space on the \(n \times n\) whitened system
\(B^\top B + \lambda I\) with \(B = J S\), \(S S^\top = P\). For
\(\lambda > 0\) the two families produce the *same step* (the push-through
identity below); they differ in which dimension they factor or iterate in
and in which metric callbacks they need.

### Auto

`linear_solver="auto"` (the default) resolves to one of the two dense forms
at trace time, when the residual and parameter shapes are concrete:
`gram_cholesky` when \(n > m\) (strictly more parameters than residuals),
`normal_cholesky` otherwise. It factors the smaller of the two SPD systems.
The rule keys on shape only — it never inspects numerical rank — and since
both forms compute the same step, `auto` is a cost choice, not a semantics
choice. The resolution is a plain Python branch during tracing: one compiled
program per problem shape, no runtime branching.

### Gram Cholesky

`linear_solver="gram_cholesky"` materializes \(J^\top\), applies
`metric.solve`, and factors the residual-space dual system

$$
(J P J^\top + \lambda I_m)y = r,
\qquad s = -P J^\top y.
$$

This is the fastest dense path when \(m \ll n\): the factorization is
\(m \times m\) and \(n\) enters only through matvecs.

### Normal Cholesky

`linear_solver="normal_cholesky"` materializes
\(B^\top = S^\top J^\top\) via `metric.inv_sqrt_transpose` and factors the
whitened normal system in parameter space:

$$
(B^\top B + \lambda I_n)u = -B^\top r,
\qquad s = S u.
$$

The factorization is \(n \times n\), so this is the dense form for
square-to-tall problems (\(m \ge n\)). For \(\lambda > 0\) the step is
identical to the Gram form's by the push-through identity

$$
B^\top(B B^\top + \lambda I_m)^{-1} = (B^\top B + \lambda I_n)^{-1}B^\top,
$$

and its \(\lambda \to 0\) limit is the minimum-\(M\)-norm least-squares step
\(s = -S(JS)^{+}r\) at every shape and rank — tall problems that are still
rank-deficient along some directions (redundant rows, collinear columns)
keep the minimum-norm selection. See
[the pseudoinverse limit](gauss_newton.md#spectral-filter-view).

### Gram CG

`linear_solver="gram_cg"` applies the residual-space dual system matrix-free:

$$
y \mapsto J P J^\top y + \lambda y,
$$

using JAX linearization for JVPs/VJPs and `jax.scipy.sparse.linalg.cg`.
The iterative solve defaults to a small fixed budget: `iterative_tol=0.0`,
`iterative_atol=0.0`, and `iterative_maxiter=8`.
With the default `implicit_solver="auto"`, differentiating `solve(...).x`
also stays matrix-free for forward `gram_cg` solves. Pass
`implicit_solver="gram_cholesky"` (or `"normal_cholesky"`) to use a dense
implicit rule instead.

`linear_solver="gram_cg"` requires a `dual_preconditioner(v, damping)`
callback, applied as the CG preconditioner (for the geodesic-acceleration
solve as well). It must be a jit-traceable, linear, SPD approximation of
\((J P J^\top + \lambda I)^{-1} v\). It never changes the subproblem being
solved: at inner convergence the step is identical, and a budget-truncated
step still lies in \(\operatorname{range}(P J^\top)\), preserving the
minimum-metric-norm structure — so approximations are safe. Nobody should
run Krylov methods without thinking about preconditioning; to opt out
explicitly, pass `identity_preconditioner()` for unpreconditioned CG. When
the implicit solver resolves to `gram_cg` (the default
`implicit_solver="auto"` does exactly that when `linear_solver="gram_cg"`),
an `implicit_preconditioner` is required as well —
`identity_preconditioner()` works there too, or pass a dense
`implicit_solver` for a dense implicit rule. (A `normal_cg`-resolved
implicit needs no preconditioner; see
[Implicit AD](implicit_ad.md#the-implicit-preconditioner).) See
[Utilities](utilities.md) for structural constructors
(`sherman_morrison_preconditioner`, `woodbury_preconditioner`) and the
randomized `nystrom_preconditioner`. Like `metric`, the preconditioner
*callable* is static configuration: it is not carried in `LMState` and no
callback action can replace it — construct a new solver to change one. A
`preconditioner_factory` is the exception in the other direction: the callable
pair stays static, but its *prepared state* (`precond`/`precond_valid`, rebuilt
from the live iterate each accepted step) IS carried on `LMState`, so a `solve`
callback that rebuilds `lm_state` must preserve those two fields.
`dual_preconditioner`, `preconditioner_factory`, and `recycle` are
`gram_cg`-only hooks — they live in residual space.

### Normal CG

`linear_solver="normal_cg"` applies the whitened normal system matrix-free:

$$
u \mapsto B^\top(B u) + \lambda u,
\qquad B u = J(S u),\quad B^\top w = S^\top(J^\top w),
$$

with right-hand side \(-B^\top r\) and step \(s = S u\), through the same
JVP/VJP closures and `jax.scipy.sparse.linalg.cg` machinery as `gram_cg` —
but the Krylov iteration lives in the \(n\)-dimensional parameter space, so
it is the matrix-free form for square-to-tall problems. It requires the
metric's `inv_sqrt`/`inv_sqrt_transpose` and a
`normal_preconditioner(v, damping)` callback: a jit-traceable, linear, SPD
parameter-space approximation of \((B^\top B + \lambda I_n)^{-1}v\)
(`identity_preconditioner()` is the explicit opt-out). One structural
requirement has no Gram-side analogue: the preconditioner must map
\(\operatorname{range}(B^\top)\) into itself, or the minimum-norm selection
is silently lost on rank-deficient problems — see the
[normal-space preconditioner](utilities.md#the-normal-space-preconditioner-normal_cg)
for why, and for which constructions are safe.

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
non-finite step even though the damped subproblem remains well-posed. Every
other solver handles rank deficiency — the Gram/normal forms through their
damped SPD systems, `augmented_qr` / `lsmr` through the damping block that
keeps the augmented matrix full column rank for \(\lambda>0\).

### Augmented QR

`linear_solver="augmented_qr"` directly factors the whitened augmented system

$$
\begin{bmatrix}JS\\\sqrt{\lambda}I\end{bmatrix}z
\approx
\begin{bmatrix}-r\\0\end{bmatrix},
\qquad s=Sz.
$$

It never forms a Gram matrix and remains full column rank for finite \(J\) when
\(\lambda>0\), including when \(J\) is rank deficient. Its factorization width
is the parameter count \(n\), so it is intended for small systems such as DAE
algebraic roots. In the package's usual \(m\ll n\) regime, `qr` performs the
same damped solve after reducing to residual dimension and is substantially
cheaper when its full-row-rank assumption holds.

### LSMR

`linear_solver="lsmr"` solves that same whitened augmented system iteratively by
LSMR bidiagonalization, using only \(J\)/\(J^\top\) matvecs — the matrix-free
sibling of `augmented_qr`. Because it works on \(B = JS\) directly rather than
on a Gram/normal system whose condition number is the *square* of
\(\operatorname{cond}(B)\), it keeps the step accurate at small \(\lambda\)
where the squared solves hit their `eps·cond` floor. It requires the metric's
`inv_sqrt`/`inv_sqrt_transpose` and accepts an optional
`whitened_preconditioner` (a `WhitenedPreconditioner` parameter-space
right-preconditioner `R⁻¹`) that clusters the LSMR spectrum when the whitened
operator is itself ill-conditioned. The damping row rides inside the
preconditioned operator, so the *posed* subproblem is exactly the
identity-damped whitened subproblem in \(u = R^{-1}z\) for every
\(\lambda>0\): at inner convergence the step is independent of \(R\)
(budget-truncated iterates can still differ across \(R\) — the
preconditioner changes the iteration path, not the subproblem), and the
\(\lambda \to 0\) selection limit is minimum-\(M\)-norm for any \(R\). It is
detailed in the
[utilities guide](utilities.md#matrix-free-lsmr-whitened-subproblem).

## Minimal Example

```python
import jax
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt


def residual_fn(x, args):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


ts = jnp.linspace(0.0, 2.0, 20)
ys = 2.0 * jnp.exp(-1.0 * ts)
x = {"a": 1.0, "b": 0.0}

solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
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
- `jax.vmap(lambda x0: solver.solve(x0, ...))` works for independent
  multi-start and per-sample calibration solves. The batched loop runs until
  every lane has stopped, so `status` and `steps` are per-lane results but
  runtime is governed by the slowest lane. See
  [Callbacks and Cookbook](callbacks.md#batched-multi-start).
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
the Gram assembly). The flag only affects the dense Gram/normal forms —
`gram_cholesky`, `normal_cholesky`, and `auto`, which resolves to one of
them; the carried `(n_params, n_residuals)` \(J^\top\) buffer serves both
factorizations. The matrix-free solvers never materialize a Jacobian, so it
is ignored for them.

```python
solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
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
metric in selecting the minimum-norm tangent, matrix-free CG implicit solves,
and worked examples.

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
damping and tolerances to the residual dtype internally.

### Precision Knobs

Two constructor knobs promote a targeted slice of a float32 program to
float64 without touching the model. Both accept `None` (the default) or
`jnp.float64`, and both require x64 support to be enabled — they make
float64 available to one pipeline; they never demote explicitly-float64
data.

- **`linear_solve_dtype=jnp.float64`** promotes the dense linear-solve
  pipelines: the `gram_cholesky`/`normal_cholesky` forward factorizations
  *and* the dense implicit-AD rules (the implicit solve inherits the knob,
  so the undamped implicit system — the most conditioning-sensitive solve
  in the library — is never silently less precise than the damped forward
  one). The recipe: \(J^\top\) (or \(B^\top\)) is cast wide *before* the
  metric application, the assembly, factorization, and triangular solves
  run in float64, and only the returned step or tangent is cast back to the
  residual dtype. The model, residual, and Jacobian VJP passes stay
  float32. Forming a Gram or normal matrix squares
  \(\operatorname{cond}(JS)\), which is exactly what makes those paths
  float32-fragile for stiff systems (e.g. a metric weight injecting a
  \(1/\varepsilon\) spike into \(P\)) — this knob is the targeted fix. It
  requires a solver that has a dense pipeline to promote: forward
  `auto`/`gram_cholesky`/`normal_cholesky`, or a dense-resolved
  `implicit_solver`.
- **`metric_solve_dtype=jnp.float64`** sets the dtype the resolved metric
  callbacks *compute in*: the solver wraps the metric — a fixed `metric` or
  a `metric_factory`'s built one, after `build` — with the
  [`metric_with_compute_dtype`](utilities.md#compute-dtype-wrapper)
  mechanics, so each callback upcasts its input, computes wide, and
  restores the caller's dtype. `None` leaves the metric computing in
  whatever dtype the consuming solve hands it (float64 under
  `linear_solve_dtype`, the residual dtype otherwise). Kernel Gram
  factorizations are routinely the worst-conditioned piece of the whole
  pipeline, so this knob often earns `jnp.float64` even in
  otherwise-float32 programs — see the
  [Tuning Guide](tuning_guide.md#float64-a-la-carte). It requires a custom
  metric or metric factory to wrap.

See the [Tuning Guide](tuning_guide.md#float64-a-la-carte) for measured
costs and for choosing between the knobs and full x64.

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

For the quasiseparable state-space (Matérn) metric applies and setup (sequential vs
parallel vs dense, n up to 1e5):

```bash
uv run --group benchmark --group gpu pytest \
  benchmarks/test_quasiseparable_benchmark.py --benchmark-only
```

On CUDA machines, the optional GPU tests (including the quasiseparable
metric check) are:

```bash
uv run --group gpu pytest tests/test_gpu.py
```

## Optimistix

For a broader JAX nonlinear solver library, see
[Optimistix](https://github.com/patrick-kidger/optimistix). It provides general
least-squares, root-finding, and minimization abstractions. `nlls_gram` is
narrower and focuses on underdetermined LM with explicit parameter-space metrics.

## API Reference

::: nlls_gram.LevenbergMarquardt

::: nlls_gram.Metric

::: nlls_gram.MetricFactory

::: nlls_gram.metric_from_cholesky

::: nlls_gram.metric_from_tridiagonal_precision

::: nlls_gram.metric_from_state_space

::: nlls_gram.matern_state_space

::: nlls_gram.metric_from_quasiseparable

::: nlls_gram.metric_from_shifted_matvec

::: nlls_gram.metric_from_diagonal

::: nlls_gram.blockdiag_metric

::: nlls_gram.sherman_morrison_preconditioner

::: nlls_gram.woodbury_preconditioner

::: nlls_gram.LMState

::: nlls_gram.LMHyperparams

::: nlls_gram.LMInfo

::: nlls_gram.LMStatus

::: nlls_gram.LMSolveAction

::: nlls_gram.LMSolveContext

::: nlls_gram.LMSolveResult

::: nlls_gram.MultiStart

::: nlls_gram.MultiStartInfo
