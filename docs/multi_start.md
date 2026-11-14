# Multi-Start

`solve(multi_start=MultiStart(...))` retries failed solves from fresh initial conditions sequentially, or evaluates multiple initial conditions in parallel under `jax.vmap` and returns the best candidate as a single unbatched `SolveResult` with `result.multi_start` (`MultiStartInfo`).

```python
import jax
import jax.numpy as jnp
from nlls_gram import LevenbergMarquardt, MultiStart


def residual(theta, args, p):
    return theta[0] ** 2 - p


def draw(key, x, args):
    return jax.random.uniform(key, x.shape, x.dtype, 0.5, 3.0)


solver = LevenbergMarquardt(residual)
ms = MultiStart(key=jax.random.key(0), num_starts=5, draw=draw)
result = solver.solve(
    jnp.zeros(1),
    p=jnp.asarray(4.0),
    atol=1e-8,
    max_steps_is_success=False,
    multi_start=ms,
)
```

## Hooks (`draw`, `reset_args`, `accept`)

- `draw(key, x_old, args_old) -> x_new` generates a fresh initial parameter pytree with the same structure, shapes, and dtypes as `x0`. `DrawNNXModule(module_cls, *args, **kwargs)` provides a value-hashable `draw` hook for Flax NNX modules.
- `reset_args(key, args_old) -> args_new` resets attempt-local solver `args` (such as epoch counters or simulation RNG streams) with the same pytree structure as `args`.
- `accept(key, result) -> bool` overrides the default acceptance check (`CONVERGED`, plus `MAX_STEPS` when `max_steps_is_success=True`). Effective acceptance is always `accept(key, result) & jnp.isfinite(loss)`.

At least one of `draw` or `reset_args` is required when `num_starts > 1`. Attempt `0` always evaluates the caller's original `(x0, args)` without calling `draw` or `reset_args`. For attempt `k`, keys follow the deterministic split:

```python
draw_key, accept_key, reset_key = jax.random.split(jax.random.fold_in(key, k), 3)
```

## Sequential versus parallel execution

| Property | `parallel=False` (default) | `parallel=True` |
|---|---|---|
| Execution | Sequential `while_loop` stopping at the first accepted attempt | Batched `vmap` across all `num_starts` lanes |
| Winner | First accepted attempt | Accepted lane with lowest objective loss |
| Fallback if none accepted | Lowest finite loss across attempts (or last attempt if all nonfinite) | Lowest finite loss across lanes (or lane `0` if all nonfinite) |
| `num_starts` compilation | Traced for `num_starts > 1` | Static |

## Implicit differentiation

Derivatives with respect to `p` pass through the selected winning solution only, using the solver's implicit differentiation rule at `(result.x, result.args, p)`. If the selected result fails the solver's AD success policy, `result.x` and `result.aux` return exact zero tangents.
