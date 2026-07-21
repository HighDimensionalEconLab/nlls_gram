# Callbacks and Cookbook

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
stops it. The returned status remains `LMStatus.MAX_STEPS`, but the default
`max_steps_is_success=True` treats that finite result as usable for implicit AD
and default multi-start acceptance. Set it to `False` when budget exhaustion
must use the strict failed-solve path. `max_steps_is_success` is a static Python
boolean. The tolerances and `max_steps` are traced values: changing them between
calls does not recompile the loop. They are validated in Python, so pass
concrete numbers, not tracers from an enclosing `jax.jit`.
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
| `LMStatus.MAX_STEPS` | `max_steps` was reached; usable by default, strict when `max_steps_is_success=False`. |
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

A callback that replaces `x` or `args` with a point that is not valid for the
residual or implicit AD must also stop with `status=LMStatus.NONFINITE` (or
another failed status). In particular, a replacement on the final iteration is
otherwise reported as `MAX_STEPS`, which is intentionally AD-successful by
default; the solver does not pay for another unconditional residual check just
to second-guess a callback's explicit replacement.

### Resettable Hyperparameters

`solve()` populates `lm_state.hyper` with an `LMHyperparams` of traced
per-step values: `damping_decrease`, `damping_increase`, `max_damping`,
`geodesic_acceptance_ratio`, `iterative_tol`, `iterative_atol`, and
`iterative_maxiter`. Because they ride in the lm_state, a
callback can reset any of them mid-solve — exactly like a damping reset:

```python
new_hyper = dataclasses.replace(
    ctx.lm_state.hyper, iterative_maxiter=jnp.asarray(40, jnp.int32)
)
return LMSolveAction(lm_state=dataclasses.replace(ctx.lm_state, hyper=new_hyper))
```

Two contracts: knobs constructed as `None` (uncapped `max_damping`,
backend-default `iterative_maxiter`) are compiled out and stay `None` — a
callback cannot turn them on; and replacement values must be arrays of the same dtype (use
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

### Batched Multi-Start

For the common case — retry a failed solve from fresh starts, or race several
starts and keep the best — use the built-in
[`multi_start=MultiStart(...)`](multi_start.md) option on `solve`. It handles
the key schedule, success test, best-lane selection, and implicit
differentiation through the winner, and returns a single unbatched result:

```python
from nlls_gram import MultiStart

ms = MultiStart(key=jax.random.key(0), num_starts=8, draw=draw, parallel=True)
result = solver.solve(x0, args, max_steps=100, atol=1e-8, multi_start=ms)
```

The manual `jax.vmap` recipe below remains useful when you need **all** lanes
back (the feature deliberately returns only the winner). The result fields are
batched arrays and pytrees, so selecting the best converged lane is ordinary
JAX indexing:

```python
def solve_start(x0):
    return solver.solve(x0, args, max_steps=100, atol=1e-8)


results = jax.vmap(solve_start)(x0_batch)
converged = results.status == LMStatus.CONVERGED
loss = jnp.where(converged, results.info.loss, jnp.inf)
best = jnp.argmin(loss)
x_best = jax.tree.map(lambda leaf: leaf[best], results.x)
```

To batch per-sample calibration data, map over `p` as well:

```python
def solve_sample(x0, p):
    return solver.solve(x0, args, p=p, max_steps=100, atol=1e-8)


results = jax.vmap(solve_sample)(x0_batch, p_batch)
```

JAX batches the internal `while_loop` until every lane has stopped. Lanes that
finish early keep their result and report their own `steps` and `status`, but
the compiled call still pays for iterations until the slowest lane stops. Those
extra masked iterations are discarded work only: a stopped lane's `x`, `steps`,
and `status` stay frozen at its own stop, whether that stop came from a
tolerance or from a callback. Tolerances are traced data, so `atol`, `gtol`,
and `xtol` may themselves be vmapped to give each lane its own stopping
criterion. `save_steps` composes the same way — a lane that stops early keeps
its history rows frozen, with the padding beyond its own `steps` staying zero
while other lanes continue. Callbacks used under `vmap` must be traceable and
batch-safe; recipes based on `jax.experimental.io_callback`, such as the
wall-clock time limit below, are host callbacks and should stay outside
vmapped solves.

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
values of `x` or `args` invalidates the Jacobian cache automatically — and
likewise the `metric_factory` prepared state, which is rebuilt at the new
point on the next update.

### Scheduled Inner-Solve Accuracy

With `linear_solver="gram_cg"`, cheap inexact steps are fine far from the solution
(the accept/reject test absorbs them), but near convergence step quality
limits the rate. Grow the CG budget once the loss crosses a threshold —
one solve call, so implicit differentiation still applies:

```python
solver = LevenbergMarquardt(
    residual_fn,
    linear_solver="gram_cg",
    iterative_maxiter=2,
    dual_preconditioner=identity_preconditioner(),
    ad_solver_preconditioner=identity_preconditioner(),
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
a coarse CG stage, then a dense solver (`auto`) warm-started with `result.x` and
`result.lm_state` — the implicit derivative is unaffected since it is defined
at the returned solution only.

This schedule composes unchanged with Krylov recycling
([`recycle=RecycleConfig(...)`](tuning_guide.md#recycling-and-deflation-across-steps)):
`rank`/`window` are static shapes the callback must not touch, while the carried
deflation basis shrinks the budget each step needs and the schedule then grows
it toward the endgame. The callback contract is the same — preserve the recycle
state with `dataclasses.replace(ctx.lm_state, ...)` (a fresh `LMState` that drops
it is rejected, exactly like the Jacobian cache).

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

For the iterate history itself there is no need for a callback:
`solve(..., save_steps=True)` stacks x0 and every kept post-step iterate onto
`result.x_history` (a pytree shaped like `x` with a `max_steps + 1` leading
axis; rows beyond `result.steps` are zero padding), plus the row-aligned
`result.args_history` (the kept post-action args — recorded even when no
callback ever replaces them, so an args-resampling callback's history is
complete) and, with `has_aux=True`, `result.aux_history`. Slice with
`result.x_history[: int(result.steps) + 1]` (per leaf for a pytree `x`) — the
histories are differentiation-inert.

For other per-step diagnostics, record into preallocated buffers sized by
`max_steps`, then plot after the solve:

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
