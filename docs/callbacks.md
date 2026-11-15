# Callbacks and Stopping

`solve` runs repeated `update` steps inside `jax.lax.while_loop` (when `jit=True`, the default) and returns a `SolveResult`.

## Stopping tolerances and statuses

Three optional scalar tolerances control convergence to `SolverStatus.CONVERGED` (each defaults to `0.0`, which disables that check) once all active constraint schedules and partitions have stabilized:

- `atol` tests the full stacked optimization residual norm \(\|r(\theta)\|_2 = \sqrt{\texttt{info.resid\_loss}}\) (including transformed inequality and complementarity rows, excluding the ridge penalty). With `ridge=None`, `atol > 0` stops when \(\|r(\theta)\|_2 < \texttt{atol}\). With `ridge > 0` (on both `LevenbergMarquardt` and `Newton`), default `atol=0` checks `gtol` and `xtol` alone, while `atol > 0` acts as a conjunctive residual ceiling \(\|r(\theta)\|_2 \le \texttt{atol}\) paired with `gtol` or `xtol` (passing `atol > 0` with `gtol=0` and `xtol=0` raises `ValueError`).
- `gtol` tests the pre-step gradient norm \(\|g(\theta_k)\|_2 = \|J(\theta_k)^\top r(\theta_k) + \rho Q \theta_k\|_2 < \texttt{gtol}\) and returns the updated iterate \(\theta_{k+1}\).
- `xtol` tests the accepted displacement norm \(\|\theta_{k+1} - \theta_k\|_2 = \|\alpha_k d_k + \frac{1}{2}\alpha_k^2 a_k\|_2 < \texttt{xtol}\).

`max_steps_is_success=True` (the default) treats `SolverStatus.MAX_STEPS` as usable for implicit AD and default multi-start acceptance. Set `max_steps_is_success=False` when budget exhaustion must follow the failed-solve path.

| Status | Meaning |
|---|---|
| `SolverStatus.CONVERGED` | Active stopping criteria were satisfied. |
| `SolverStatus.MAX_STEPS` | `max_steps` was reached without meeting stopping criteria (including `max_steps=0`). |
| `SolverStatus.NONFINITE` | Loss or state became nonfinite, or a callback set this status. |
| `SolverStatus.CALLBACK_STOP` | A callback returned `stop=True` without overriding `status`. |
| `SolverStatus.LINE_SEARCH_FAILED` | (`Newton` only) Armijo backtracking exhausted `maxls` trials. |
| `SolverStatus.LINEAR_SOLVE_FAILED` | (`Newton` only) Linear solve encountered a singular or nonfinite pivot. |
| `SolverStatus.NON_DESCENT` | (`Newton` only) Computed direction failed \(g^\top d < 0\). |
| `SolverStatus.STAGNATION` | (`Newton` only) Floating-point step addition underflowed (`x + step == x`). |

## Callback contract

Callbacks receive a `CallbackContext` after each step (`step`, `x`, `state`, `info`, `x_old`, `state_old`, `initial_state`, `args`, `p`, `user_state`) and return `None` or `SolverAction(stop=..., status=..., x=..., state=..., args=..., user_state=...)`.

- Changing `x` or `args` invalidates cached Jacobians and factorizations and suppresses convergence on that step.
- Changing `state.ridge`, `state.ridge_metric`, `state.ineq_weight`, or `state.ineq_smoothing` invalidates penalty-dependent caches and suppresses convergence on that step.
- Replacing `state.preconditioner` with an instance of the same pytree structure updates linear-solver conditioning without suppressing convergence or triggering recompilation.
- Traced numeric step parameters live in `state.hyper` (`LMHyperparams` or `NewtonHyperparams`) and can be updated inside callbacks via `dataclasses.replace`.

## Cookbook examples

### Host loop with `update`

```python
import time
import jax

step_fn = jax.jit(solver.update)
state = solver.init(x0, args)
x = x0
t0 = time.perf_counter()
for _ in range(max_steps):
    x, state, info = step_fn(x, state, args)
    if float(info.grad_norm) < 1e-8 or time.perf_counter() - t0 > 30.0:
        break
```

### Adaptive Krylov budget

```python
import dataclasses
import jax.numpy as jnp
from nlls_gram import GramCG, IdentityPreconditioner, LevenbergMarquardt, SolverAction

solver = LevenbergMarquardt(
    residual_fn,
    linear_solver=GramCG(IdentityPreconditioner(), maxiter=4),
)


def grow_budget(ctx):
    new_maxiter = jnp.where(
        ctx.info.loss < 1e-2,
        jnp.asarray(40, jnp.int32),
        ctx.state.hyper.iterative_maxiter,
    )
    new_hyper = dataclasses.replace(ctx.state.hyper, iterative_maxiter=new_maxiter)
    return SolverAction(state=dataclasses.replace(ctx.state, hyper=new_hyper))


result = solver.solve(x0, args, max_steps=200, atol=1e-8, callback=grow_budget)
```

### Adaptive inequality penalty continuation

```python
import dataclasses
import jax.numpy as jnp
from nlls_gram import SolverAction


def tighten_inequality(ctx):
    boost = (ctx.step % 10 == 0) & (ctx.info.ineq_max_violation > 1e-4)
    new_w = jnp.where(boost, ctx.state.ineq_weight * 10.0, ctx.state.ineq_weight)
    return SolverAction(state=dataclasses.replace(ctx.state, ineq_weight=new_w))
```
