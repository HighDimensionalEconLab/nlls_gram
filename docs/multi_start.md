# Multi-Start

`solve(multi_start=MultiStart(...))` retries a failed solve from fresh initial
conditions, or races several starts in parallel and keeps the best one. Either
way it returns a **single** `LMSolveResult` — the selected solution — with
diagnostics attached at `result.multi_start` (a `MultiStartInfo`). With
`multi_start=None` (the default) nothing changes: the solve takes exactly the
plain code path.

```python
import jax
import jax.numpy as jnp
from nlls_gram import MultiStart, UnderdeterminedLevenbergMarquardt

def residual(theta, args, p):
    return theta[0] ** 2 - p  # stalls when started at theta = 0

def draw(key, x, args):
    # Fresh initial condition; args may be redrawn too (see below).
    return jax.random.uniform(key, x.shape, x.dtype, 0.5, 3.0), args

solver = UnderdeterminedLevenbergMarquardt(residual)
ms = MultiStart(key=jax.random.key(0), num_starts=5, draw=draw)
result = solver.solve(jnp.zeros(1), p=jnp.asarray(4.0), atol=1e-8, multi_start=ms)
result.multi_start.attempt        # which start won (0 = your x0)
result.multi_start.attempts_run   # how many solves actually ran
```

`draw` and `accept` are jit **static** arguments, so they key the compile by
their `__hash__`/`__eq__`. Plain functions, lambdas, and closures hash **by
identity** — define them once at setup scope, never as fresh lambdas at the
call site, or each one recompiles. A value-hashable hook keys by value instead:
`DrawNNXModule` (below) shares one compile across equal specs. A `MultiStart`
object itself is cheap to rebuild per call (only the hooks' cache keys matter),
and its `key` is ordinary traced data — new
keys, new `x0`/`args` values, and (in sequential mode) a different
`num_starts` among values above one all reuse the compiled solve. Crossing
`num_starts = 1` to `N > 1` (or back) compiles once more: the single-start
form never draws, so it is a structurally different program.

## The draw contract

```python
def draw(key, x_old, args_old):
    ...
    return x_new, args_new
```

`draw` must be traceable and **type-stable**: the returned `(x, args)` must
match the input pytree structure, shapes, and dtypes exactly (checked up front
with an abstract `jax.eval_shape` trace under `jit=True`; the first concrete
draw is checked under `jit=False`). What it draws is up to you — reinitialize
a network, perturb the previous start, resample the data inside `args`, or any
combination. `p` is deliberately **not** an input and cannot change across
starts: it is the differentiation target, and the implicit gradient is taken
at a single fixed `p`.

In **sequential** mode `draw` receives the previous attempt's *initial*
values — the original `(x0, args)` for the first retry, then each drawn
`(x, args)` in turn — never the solver-mutated `result.x`/`result.args`. In
**parallel** mode every lane draws from the original `(x0, args)`.

A flax `nnx` reinitialization draw:

```python
def draw(key, x_old, args_old):
    _, theta = nnx.split(PolicyMLP(settings, rngs=nnx.Rngs(key)), nnx.Param)
    return theta, args_old
```

`DrawNNXModule` packages exactly this draw so you skip the per-driver closure:

```python
from nlls_gram import DrawNNXModule

draw = DrawNNXModule(PolicyMLP, settings, dtype=dtype)  # equal specs share one compile
```

It rebuilds `module_cls(*args, rngs=nnx.Rngs(key), **kwargs)` on each retry and returns its
`nnx.Param` state, passing `args` through unchanged. The drawn state must be type-stable against
`x0` (same structure, shapes, dtypes), so construct the module with a matching `param_dtype`/`dtype`
(e.g. thread `dtype=` through). Unlike a fresh closure it is **value-hashable** on
`(module_cls, args, kwargs)`, so equal specs share a single jit compilation instead of recompiling.

A data-resampling draw (mv2020 style), threading a key inside `args`:

```python
def draw(key, x_old, args_old):
    init_key, exo_key, carry_key = jax.random.split(key, 3)
    _, theta = nnx.split(PolicyMLP(settings, rngs=nnx.Rngs(init_key)), nnx.Param)
    args_new = args_old.replace(
        exo=simulate_markov_chain(exo_key, s_0, P_cumsum, train_T),
        key=carry_key,
        epoch=jnp.asarray(0, jnp.int32),
    )
    return theta, args_new
```

## The accept hook

By default an attempt succeeds when `result.status == LMStatus.CONVERGED`.
Pass `accept` to override the test — for example to require a fresh-data
validation metric rather than trusting the training tolerance:

```python
def accept(key, result):
    policy = nnx.merge(graphdef, result.x)
    test_exo = simulate_markov_chain(key, s_0_test, P_cumsum, test_T)
    return euler_mean_abs(policy, test_exo) < 5e-5

ms = MultiStart(key=key, num_starts=5, draw=draw, accept=accept)
```

`accept` receives its own key (see the schedule below) and must return a
scalar boolean-like value; the solver canonicalizes the dtype. Inside
`accept`, `result.multi_start` is still `None` (the diagnostics are attached
after selection). Note `result.info.loss` can be stale when a `callback`
replaced `x`/`args` after the last update — recompute anything you need at
`(result.x, result.args)`.

An accepted-but-nonfinite result never wins: effective success is
`accept(...) AND isfinite(loss)` in both modes.

## Sequential vs parallel

`parallel` changes the **selection semantics**, not just the execution
strategy:

| | sequential (default) | `parallel=True` |
|---|---|---|
| execution | one solve at a time, stops at the first success | all `num_starts` lanes under one `vmap` |
| winner | the **first** accepted attempt | the accepted lane with the **lowest** loss |
| all fail | lowest finite loss across attempts | lowest finite loss across lanes |
| none finite | the last attempt | lane 0 |
| cost | pays only for the attempts run | always pays for `num_starts` solves (but they run batched) |
| `num_starts` | traced — changes among values > 1 never retrace | static — changing it recompiles |

Parallel lanes share one vmapped `while_loop`, so every lane steps until the
slowest lane stops: the compiled cost is `num_starts x slowest lane`, not the
sum of each lane's own step count. Budget `max_steps` accordingly.

The ranking loss is the sum of squared residuals at the returned solution
(`result.info.loss`, or recomputed at `(result.x, result.args, p)` when a
`callback` is present), masked to `+inf` when nonfinite; ties break to the
lowest attempt index. `MultiStartInfo.loss` records the winner's value.

Identical `draw` keys are used in both modes (below), so a draw that ignores
`(x_old, args_old)` produces the same candidate starts sequentially and in
parallel; the modes still may pick different winners (first-accepted vs
best-of-batch).

## Key schedule

For attempt/lane `k`:

```python
draw_key, accept_key = jax.random.split(jax.random.fold_in(key, k))
```

Attempt 0 is always the caller's `(x0, args)` and never consumes its
`draw_key`. The schedule is a documented contract (pinned by tests), so runs
are reproducible and an attempt's draws do not depend on how many attempts ran
before it.

## Differentiation

Gradients with respect to `p` flow through the **selected solution only**,
via the same implicit rule as a plain solve (see
[implicit differentiation](implicit_ad.md)): the residual is relinearized at
the returned `(x, args, p)`, and everything else — the key, the initial
conditions, the losing attempts, the diagnostics — has zero tangents.

Because selection is discrete, the derivative is **piecewise**: it is the
chosen basin's implicit derivative, and it jumps when a different start wins
(ties, basin switches). The acceptance test and the argmin are not
differentiated. Derivatives are only meaningful when the winner actually
solved the problem — after an all-fail fallback the rule linearizes at a
non-solution.

## Interactions

- **`lm_state` warm starts** — the caller's `lm_state` applies to attempt 0;
  drawn attempts inherit its damping and hyperparameters but the Jacobian
  cache is invalidated (it described a different `(x, args)`). In parallel
  mode the cache is dropped on all lanes: under `vmap` the cache-reuse branch
  is a select that evaluates both sides, so a warm cache cannot save work.
- **`save_steps`** — composes; the winner's (unbatched) histories are
  returned. Parallel mode materializes `num_starts` history buffers during
  the solve, and sequential mode briefly holds two.
- **Outer `vmap`** — sequential multi-start composes with an outer `vmap`
  (e.g. one multi-start solve per sample); as with any vmapped `while_loop`,
  all lanes wait for the slowest sample's schedule. Parallel-inside-`vmap`
  nests two batch axes — fine for small `num_starts x batch`, memory-hungry
  beyond that.
- **`jit=False`** — both modes run eagerly for debugging; sequential mode
  then calls `draw` lazily, only when a retry actually happens.

## API

::: nlls_gram.MultiStart

::: nlls_gram.MultiStartInfo

::: nlls_gram.DrawNNXModule
