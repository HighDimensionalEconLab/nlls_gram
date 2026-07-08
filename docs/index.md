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
holds the final diagnostics whether or not the solve converged. With
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
metric.

Shape requirements:

- `solve` must support vectors `(n_params,)` and matrices `(n_params, k)`.
- `inv_sqrt` must support vectors `(n_params,)`; matrices
  `(n_params, n_residuals)` are additionally required for implicit
  differentiation when the metric has no `solve`.
- `inv_sqrt_transpose` must support matrices `(n_params, n_residuals)` for QR.
- `norm` only needs to support vectors `(n_params,)`.

Validation rules:

- For `linear_solver in {"cholesky", "cg"}`, a custom metric requires
  `metric.solve`.
- For `linear_solver in {"qr", "lsmr"}`, a custom metric requires both
  `metric.inv_sqrt` and `metric.inv_sqrt_transpose`.
- If `geodesic_acceleration=True` and a custom metric is supplied,
  `metric.norm` is required.
- The solver does not infer `norm`, `inv_sqrt`, or `inv_sqrt_transpose` from
  `solve`.

`norm` is separate because `solve` applies \(M^{-1}\), while the norm needs
\(M\):

$$
\|x\|_M = \sqrt{x^\top Mx}.
$$

Recovering that norm from a black-box \(M^{-1}\) solve would require another
inverse operation. Likewise, a square-root factor \(S\) is not generally
recoverable from an arbitrary solve callback.

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

## Cholesky Metric Helper

For a dense metric \(M=LL^\top\) with \(L\) lower triangular (the form
returned by `jnp.linalg.cholesky`), use:

```python
import jax.numpy as jnp

from nlls_gram import metric_from_cholesky

L = jnp.linalg.cholesky(metric_matrix)
metric = metric_from_cholesky(L)
```

The helper returns a `Metric` with all four callbacks filled in.

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

## Solve Loop and Callbacks

`solve` runs repeated LM updates and returns an `LMSolveResult`:

```python
result = solver.solve(
    x,
    args=(ts, ys),
    p=None,
    max_steps=100,
    atol=1e-8,
)
x = result.x
```

`max_steps` is always enforced. Three optional tolerances stop with
`LMStatus.CONVERGED`; each defaults to `0.0`, which disables that check:

- `atol`: residual norm, \(\sqrt{\sum_i r_i^2} < \texttt{atol}\).
- `gtol`: gradient norm, \(\|J^\top r\|_2 < \texttt{gtol}\), evaluated at the
  pre-step parameters of the step just taken.
- `xtol`: step norm, \(\|s\|_2 < \texttt{xtol}\), checked only when the step
  was accepted (rejected steps say nothing about stationarity).

With all three at zero the solve runs until `max_steps` unless a callback
stops it. The tolerances and `max_steps` are traced values: changing them
between calls does not recompile the loop. They are validated in Python, so
pass concrete numbers, not tracers from an enclosing `jax.jit`.
`solve(jit=True)` is the default and jits the loop; use `jit=False` for
debugging Python callbacks.

The per-step diagnostics behind these rules are exposed on `LMInfo` as
`grad_norm` (\(\|J^\top r\|_2\), nearly free in the dense paths since
\(J^\top\) is already materialized) and `step_norm` (norm of the candidate
step, reported even when the step is rejected). Before the first update the
loop's `LMInfo` uses sentinels `grad_norm=inf` and `step_norm=0`, so `gtol`
and `xtol` cannot fire at step zero.

Repeated rejections multiply the damping by `damping_increase` without bound,
which can overflow in float32. The constructor's `max_damping` clamps the
damping from above; leave it `None` for uncapped classic behavior.

Status codes are integer constants:

| Status | Meaning |
| --- | --- |
| `LMStatus.CONVERGED` | A tolerance (`atol`, `gtol`, or `xtol`) was met. |
| `LMStatus.MAX_STEPS` | `max_steps` was reached. |
| `LMStatus.NONFINITE` | The current loss is nonfinite, or a callback chose this status. |
| `LMStatus.CALLBACK_STOP` | A callback stopped without a custom status. |
| `LMStatus.RUNNING` | Internal running state, not a final successful status. |

Callbacks receive an `LMSolveContext`:

| Field | Meaning |
| --- | --- |
| `step` | One-based step number just completed. |
| `x`, `lm_state`, `info` | Accepted/rejected step result and LM diagnostics. |
| `x_old`, `lm_state_old` | Values before that step. |
| `initial_lm_state` | State supplied to `solve` or created by `init`. |
| `args`, `p`, `user_state` | Current auxiliary data, read-only external data, and user state. |

A callback returns `None` or `LMSolveAction(...)`. Omitted action fields are left
unchanged. The callback may set `stop`, `status`, `x`, `lm_state`, `args`, or
`user_state`; it cannot replace `p`. When an action changes the values of `x`
or `args`, that step's tolerance checks are skipped — the diagnostics describe
the pre-action problem — and resume after the next update.

### Resettable Hyperparameters

`solve()` populates `lm_state.hyper` with an `LMHyperparams` of traced
per-step values: `damping_decrease`, `damping_increase`, `max_damping`,
`geodesic_acceptance_ratio`, `iterative_tol`, `iterative_atol`,
`iterative_maxiter`, and `lsmr_conlim`. Because they ride in the lm_state, a
callback can reset any of them mid-solve — exactly like a damping reset:

```python
new_hyper = dataclasses.replace(
    ctx.lm_state.hyper, iterative_maxiter=jnp.asarray(40, jnp.int32)
)
return LMSolveAction(lm_state=dataclasses.replace(ctx.lm_state, hyper=new_hyper))
```

Two contracts: knobs constructed as `None` (uncapped `max_damping`, unlimited
`iterative_maxiter`) are compiled out and stay `None` — a callback cannot turn
them on; and replacement values must be arrays of the same dtype (use
`jnp.asarray`/`jnp.where`), since they live in the jitted loop carry. Static
configuration — `linear_solver`, `geodesic_acceleration`, `cache_jacobian`,
`has_aux`, the metric — shapes the compiled program and stays on the solver.
`init()` leaves `hyper=None`, which falls back to the constructor values and
compiles to the same program with no extra per-call buffers — manual
`update()` loops pay nothing. To schedule hyperparameters in a manual loop,
opt in with `dataclasses.replace(lm_state, hyper=solver.hyperparams(dtype))`.
When chaining solves, a warm-started `lm_state` carries the *first* solver's
hyperparameters; pass `dataclasses.replace(lm_state, hyper=None)` to re-derive
them from the second solver's constructor.

Under `jit=True`, callbacks must be JAX-traceable and return the same pytree
structure on every iteration. Use `jnp.where` or `jax.lax.cond` for
data-dependent choices rather than Python `if` statements over arrays.

Example callback that aborts on a nonfinite candidate and otherwise stops at a
residual-norm threshold:

```python
import jax.numpy as jnp

from nlls_gram import LMSolveAction, LMStatus


def stopping_callback(ctx):
    nonfinite = ~jnp.isfinite(ctx.info.loss_candidate)
    converged = jnp.sqrt(ctx.info.loss) < 1e-8
    status = jnp.where(nonfinite, LMStatus.NONFINITE, LMStatus.CONVERGED)
    return LMSolveAction(stop=nonfinite | converged, status=status)


result = solver.solve(
    x,
    args=(ts, ys),
    max_steps=100,
    callback=stopping_callback,
)
```

The per-step `update(...)` API remains useful when you want to write the outer
loop yourself or manage host-side logging.

## Cookbook

Recipes for common training-loop patterns. The first is host-side; the rest
run entirely inside `solve(jit=True)`.

### Host Loop

Use `update()` with your own Python loop whenever you need things jit cannot
express: wall-clock time budgets, pandas/host-side logging, data whose shape
changes between epochs, or acceptance gates that restart from a fresh
initialization. Use `solve()` when the whole loop fits in jit.

```python
import time

train_step = jax.jit(solver.update)

lm_state = solver.init(x, args)
start = time.perf_counter()
for step in range(max_steps):
    x, lm_state, info = train_step(x, lm_state, args)
    if not bool(jnp.isfinite(info.loss_candidate)):
        break  # diverged
    if float(info.loss) < loss_threshold:
        break
    if time.perf_counter() - start > max_train_time:
        break
```

Reading `float(info.loss)` synchronizes with the device; for maximum
throughput read diagnostics once per epoch rather than once per step.

### Logging

The simplest callback logs and returns `None`, which means "no stop, no
overrides". Inside the jitted loop, printing goes through `jax.debug.print`:

```python
def logging_callback(ctx):
    jax.debug.print(
        "step {step}: loss={loss:.3e} damping={damping:.1e}",
        step=ctx.step,
        loss=ctx.info.loss,
        damping=ctx.lm_state.damping,
    )


result = solver.solve(x, args, atol=1e-8, callback=logging_callback)
```

To log every `k`-th step, gate the print with `jax.lax.cond` rather than a
Python `if`, since `ctx.step` is traced:

```python
def logging_callback(ctx):
    def log(_):
        jax.debug.print(
            "step {step}: loss={loss:.3e} damping={damping:.1e}",
            step=ctx.step,
            loss=ctx.info.loss,
            damping=ctx.lm_state.damping,
        )

    jax.lax.cond(ctx.step % 10 == 0, log, lambda _: None, operand=None)
```

With `solve(..., jit=False)` the same callback runs as ordinary Python, so
plain `print(float(ctx.info.loss))`, debuggers, and host-side libraries all
work.

### Divergence Stop

The default LM response to a nonfinite candidate is to reject the step and
increase damping, which often recovers. To instead treat it as terminal
divergence:

```python
def divergence_callback(ctx):
    nonfinite = ~jnp.isfinite(ctx.info.loss_candidate)
    return LMSolveAction(stop=nonfinite, status=LMStatus.NONFINITE)
```

### Epoch Resampling and Damping Reset

Hold a data lifetime fixed for `steps_per_epoch` LM steps, then resample the
collocation/simulation data and reset the damping. The PRNG key lives in
`user_state`. Note the contract: action fields are static in *structure* but
dynamic in *value* — a conditional override must always return the field,
selected with `jnp.where`, rather than sometimes returning `None`.

```python
steps_per_epoch = 16


def epoch_callback(ctx):
    boundary = ctx.step % steps_per_epoch == 0
    key, subkey = jax.random.split(ctx.user_state)
    resampled = sample_data(subkey)
    new_args = jax.tree.map(
        lambda new, old: jnp.where(boundary, new, old), resampled, ctx.args
    )
    new_lm_state = dataclasses.replace(
        ctx.lm_state,
        damping=jnp.where(boundary, ctx.initial_lm_state.damping, ctx.lm_state.damping),
    )
    new_key = jnp.where(boundary, key, ctx.user_state)
    return LMSolveAction(args=new_args, lm_state=new_lm_state, user_state=new_key)


result = solver.solve(
    x,
    sample_data(key0),
    max_steps=800,
    callback=epoch_callback,
    user_state=key1,
)
```

(`dataclasses` here is the standard-library module.) This recipe composes
with `cache_jacobian=True` without extra care: any action that changes the
values of `x` or `args` invalidates the Jacobian cache automatically.

### Scheduled Inner-Solve Accuracy

With `linear_solver="cg"`, cheap inexact steps are fine far from the solution
(the accept/reject test absorbs them), but near convergence step quality
limits the rate. Grow the CG budget once the loss crosses a threshold —
one solve call, so implicit differentiation still applies:

```python
solver = UnderdeterminedLevenbergMarquardt(
    residual_fn, linear_solver="cg", iterative_maxiter=2
)


def grow_budget(ctx):
    new_maxiter = jnp.where(
        ctx.info.loss < 1e-2,
        jnp.asarray(40, jnp.int32),
        ctx.lm_state.hyper.iterative_maxiter,
    )
    new_hyper = dataclasses.replace(ctx.lm_state.hyper, iterative_maxiter=new_maxiter)
    return LMSolveAction(lm_state=dataclasses.replace(ctx.lm_state, hyper=new_hyper))


result = solver.solve(x0, args, max_steps=200, atol=1e-8, callback=grow_budget)
```

Alternatively, a relative `iterative_tol` adapts the inner accuracy
automatically (CG's stopping test scales with the right-hand side, which is
the shrinking outer residual). For a dense endgame instead, chain two solves:
a coarse CG stage, then a Cholesky solver warm-started with `result.x` and
`result.lm_state` — the implicit derivative is unaffected since it is defined
at the returned solution only.

### Validation Early Stopping

Compute held-out metrics in the callback and stop when they jointly clear
their thresholds:

```python
def validation_callback(ctx):
    val_residual = validation_residual_fn(ctx.x, val_data)
    val_mse = jnp.mean(val_residual**2)
    return LMSolveAction(stop=val_mse < val_threshold)
```

Evaluating validation metrics every step costs a residual pass per step; gate
it on an epoch boundary with `jax.lax.cond` if it is expensive.

### Wall-Clock Time Limit

There is no traced clock in JAX, so a jitted loop reads the current time from
the host with `jax.experimental.io_callback`. The start time and budget ride
in `user_state` (traced values — a closure would bake the start time into the
compiled loop and force a recompile per solve), and passing `ctx.step` to the
host call keeps it from being hoisted out of the loop:

```python
import time

import numpy as np
from jax.experimental import io_callback

TIME_LIMIT_STATUS = 100  # any int outside the LMStatus constants


def over_time_budget(start_and_budget, _step):
    start, budget = start_and_budget
    return np.bool_(time.perf_counter() - float(start) > float(budget))


def time_limit_callback(ctx):
    timed_out = io_callback(
        over_time_budget,
        jax.ShapeDtypeStruct((), jnp.bool_),
        ctx.user_state,  # (start_time, budget_seconds)
        ctx.step,  # loop-varying arg so the call cannot be hoisted
    )
    return LMSolveAction(stop=timed_out, status=TIME_LIMIT_STATUS)


result = solver.solve(
    x0,
    args,
    max_steps=100_000,
    callback=time_limit_callback,
    user_state=jnp.asarray([time.perf_counter(), 15.0]),
)
timed_out = int(result.status) == TIME_LIMIT_STATUS
```

The host round-trip costs a fraction of a millisecond per step — negligible
for substantial problems, but prefer the host-loop pattern when steps are
microseconds. Under the default float32, storing `time.perf_counter()` in
`user_state` can round the start time by up to ~0.1s on long-uptime systems;
treat the budget as coarse.

### Fixed-Size History Recording

Record per-step diagnostics into preallocated buffers sized by `max_steps`,
then plot after the solve:

```python
max_steps = 200


def history_callback(ctx):
    history = {
        "loss": jax.lax.dynamic_update_slice(
            ctx.user_state["loss"], ctx.info.loss[None], (ctx.step - 1,)
        ),
        "damping": jax.lax.dynamic_update_slice(
            ctx.user_state["damping"], ctx.info.damping[None], (ctx.step - 1,)
        ),
    }
    return LMSolveAction(user_state=history)


result = solver.solve(
    x,
    args,
    max_steps=max_steps,
    callback=history_callback,
    user_state={
        "loss": jnp.zeros(max_steps),
        "damping": jnp.zeros(max_steps),
    },
)
losses = result.user_state["loss"][: int(result.steps)]
```

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
`cache_jacobian=True` the solver carries `(resid, Jt)` in `LMState` and a
rejected step's successor skips the residual evaluation and the
`n_residuals` VJP passes, re-solving only the small damped system (roughly
2x faster per rejected step; more when the residual is expensive relative to
the Gram assembly). The flag only affects `linear_solver="cholesky"` — the
matrix-free solvers never materialize a Jacobian, so it is ignored for them.

```python
solver = UnderdeterminedLevenbergMarquardt(
    residual_fn, init_damping=1e-2, cache_jacobian=True
)
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
  time, so for fixed-data problems it is safe to enable unconditionally; it
  is off by default only because of the manual-loop hazard.

## Implicit Differentiation

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
derivative. `result.aux` is likewise a non-differentiated output. One
construction-time caveat: the solver (including any `Metric` callbacks) is a
static object — do not build it from traced values.

### Root Selection and the Metric

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

In code, \(P x\) is applied with `metric.solve(x)` when available. If a QR/LSMR
metric is supplied only through square-root callbacks, the same inverse metric is
applied as \(P x = S S^\top x\) using `metric.inv_sqrt` and
`metric.inv_sqrt_transpose`.

### VJP

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

from nlls_gram import UnderdeterminedLevenbergMarquardt


# p without args still uses the three-argument form; the second argument is
# simply ignored.
def residual(theta, args, p):
    del args
    return jnp.array([theta[0] + 2.0 * theta[1] - p])


solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
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

## Metric Example

```python
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt, metric_from_cholesky


def residual_fn(theta, args):
    matrix, target = args
    return matrix @ theta - target


metric_matrix = jnp.array([[2.0, 0.2], [0.2, 1.0]])
L = jnp.linalg.cholesky(metric_matrix)

solver = UnderdeterminedLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    metric=metric_from_cholesky(L),
)
```

## Matrix-Free Metric Example

For CG, the metric only needs a solve callback. This can wrap an iterative solve
or any JAX-linear operation implementing \(M^{-1}x\), without materializing a
dense inverse:

```python
import jax.scipy.sparse.linalg as jsp_sparse_linalg

from nlls_gram import Metric, UnderdeterminedLevenbergMarquardt


def metric_matvec(x):
    return kernel_matvec(x) + ridge * x


def metric_solve(x):
    solution, _ = jsp_sparse_linalg.cg(metric_matvec, x, maxiter=32)
    return solution


solver = UnderdeterminedLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    linear_solver="cg",
    metric=Metric(solve=metric_solve),
)
```

This changes the LM damping metric. It is not a preconditioner for the inner CG
iteration; it changes the step being solved for.

## Geodesic Acceleration

Geodesic acceleration is off by default. When enabled, the solver computes the
second-order residual directional term with JAX forward-over-forward JVPs. The
same metric-damped linear solve computes the acceleration.

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
