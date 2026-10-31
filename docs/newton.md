# Newton with Armijo Backtracking

`Newton` solves nonlinear equations, least-squares problems, and ridge-regularized RKHS interpolation problems by computing a minimum-metric-norm (or penalized ridge) Gauss-Newton direction once per outer iteration and searching along that ray (or along the second-order parabolic geodesic arc when `geodesic_acceleration=True`) with a residual-only Armijo backtracking line search against \(\|r\|_2^2\) (`ridge=None`, the default) or \(\|r\|_2^2 + \text{ridge}\cdot\|x_m\|_W^2\) (`ridge=<positive scalar>`).

- `Newton(residual_fn, *, metric=None, ridge=None, ...)` is the Newton + Armijo counterpart to `LevenbergMarquardt(..., ridge=None)`: it minimizes \(\|r(x, \text{args}, p)\|_2^2\), using the optional `Metric` \(W = F^\top F\) to select the minimum-\(W\)-norm Gauss-Newton step \(d = -\bar{F}^{-1}(J \bar{F}^{-1})^+ r\).
- `Newton(residual_fn, *, metric=..., ridge=<positive scalar>, ...)` is the Newton + Armijo counterpart to `LevenbergMarquardt(..., ridge=<positive scalar>)`: it minimizes the penalized ridge objective \(\|r(x, \text{args}, p)\|_2^2 + \text{ridge} \cdot \|x_m\|_W^2\) (leaving the free block \(x_f\) unpenalized, or defaulting `metric` to the Euclidean identity over all parameters when `metric=None`) and supports `AnnealRidge`.

## Direction and line-search algebra

At iterate \(x\) with flattened residual \(r \in \mathbb{R}^m\) and Jacobian \(J = \partial r / \partial x \in \mathbb{R}^{m \times n}\), the Euclidean half-gradient is

$$
g = J^\top r
$$

and the minimum-Euclidean-norm Gauss-Newton direction is

$$
d = -J^+ r .
$$

For an underdetermined full-row-rank system (\(m \le n\)), \(d = -J^\top (J J^\top)^{-1} r\) is the shortest step satisfying the linearized equations \(J d = -r\). For an overdetermined full-column-rank system (\(n \le m\)), \(d = -(J^\top J)^{-1} J^\top r\) solves the linearized least-squares problem.

The merit function for backtracking is \(\phi(x) = \frac{1}{2}\|r(x)\|_2^2\), while `info.loss` reports the package-wide diagnostic \(L(x) = \|r(x)\|_2^2 = 2\phi(x)\). The directional derivative of \(\phi\) along \(d\) is the actual inner product

$$
s = g^\top d = r^\top J d .
$$

When \(d\) is computed by a truncated SVD, an inexact Krylov solve, or a tall least-squares projection where \(J d \ne -r\), \(s\) differs from \(-\|r\|_2^2\). `Newton` always uses the actual directional derivative \(s = g^\top d\) in the Armijo test. Expressed in diagnostic loss units, a trial step size \(\alpha > 0\) is accepted when \(x + \alpha d\) and \(r(x + \alpha d)\) are finite and

$$
\|r(x + \alpha d)\|_2^2 \le \|r(x)\|_2^2 + 2 c_1 \alpha s + 2\,\text{linesearch\_tol} .
$$

## Line-search and step-damping defaults

| Parameter | Default | Meaning |
|---|---|---|
| `c1` | `1e-4` | Armijo sufficient-decrease constant (\(0 < c_1 < 1\)) |
| `decrease_factor` | `0.5` | Geometric backtracking factor applied after a rejected trial |
| `maxls` | `30` | Maximum residual trials per outer step, including the initial step |
| `max_stepsize` | `1.0` | Initial trial step size \(\alpha_0 = \min(1.0, \text{max\_stepsize})\) at every outer step |
| `linesearch_tol` | `0.0` | Additive merit slack (multiplied by 2 in diagnostic loss units) |
| `damping` | `1e-3` (`0.0` with `ridge` or `LU()`) | Stateless residual-coupled step-damping coefficient (\(\text{damping} \ge 0\)) |
| `damping_power` | `1.5` | Residual norm exponent in residual-coupled step damping (\(\text{damping\_power} \ge 0\)) |
| `min_damping` | `1e-10` | Step-damping floor when \(\text{damping} > 0\) (\(\text{min\_damping} > 0\)) |
| `geodesic_acceleration` | `True` | Compute the second-order geodesic correction \(a\) reusing the step factorization and search along the parabolic arc \(x + \alpha d + \frac{1}{2}\alpha^2 a\) |
| `geodesic_acceptance_ratio` | `0.75` | Maximum allowed ratio \(2\|u_a\|_2 / \|u\|_2\) in whitened coordinates for accepting the geodesic correction |

Each outer step resets the initial trial step size to \(\alpha_0 = \min(1.0, \text{max\_stepsize})\).

When `damping > 0` (the default `1e-3`), `Newton` damps the outer linear system with a stateless, parameter-scale-invariant residual-coupled step-damping scalar before running Armijo backtracking:

$$
\lambda_k = \max\!\left(\text{min\_damping},\; \text{damping} \cdot \frac{\|B(x_k)\|_F^2}{m} \cdot \|r(x_k)\|_2^{\text{damping\_power}}\right)
$$

where \(B(x_k) = J(x_k)\bar{F}^{-1}\) is the whitened Jacobian (using unit Jacobian scale for matrix-free `CG` and `GramCG`). Unlike `ridge`, `damping` never enters the objective function or gradient and does not bias the converged root: on ill-conditioned or rank-deficient neural collocation problems where undamped steps explode along near-null singular directions, `damping=1e-3, damping_power=1.5` bounds the step far from the root and decays superlinearly as \(\|r(x_k)\|_2 \to 0\) until it hits the `min_damping` floor without carrying stateful trust-region damping across steps. Pass `damping=0.0` for strictly undamped Gauss-Newton steps.

When `geodesic_acceleration=True` (the default), each outer step also evaluates the directional second derivative \(f_{vv} = D^2 r(x_k)[d, d]\) via nested forward-mode `jax.jvp` at \(O(m + n)\) memory and solves for the second-order correction \(u_a\) by reusing the already-computed matrix factorization (a second triangular or Krylov solve on \(-f_{vv}\)). When \(2\|u_a\|_2 / \|u\|_2 \le \text{geodesic\_acceptance\_ratio}\), the Armijo line search evaluates candidates along the parabolic geodesic curve \(x_k + \alpha d + \frac{1}{2}\alpha^2 a\) instead of the straight ray \(x_k + \alpha d\).

Every trial inside the `jax.lax.while_loop` evaluates only the primal residual \(r(x(\alpha))\). The linearization, Jacobian assembly, and matrix factorization happen once per outer step before the loop. Because all backtracking trials reuse the same direction within a single `update` call, `cache_jacobian` defaults to `False` for `Newton` so the outer loop carry does not allocate or copy a dense Jacobian buffer across accepted steps. Passing `cache_jacobian=True` retains the base-point Jacobian and factorization when a line search fails, so a callback or manual `update` retry from the same iterate (for example, increasing `maxls` or `damping` in `state.hyper`) reuses both without re-differentiating or re-assembling.

To sweep numeric hyperparameters (`c1`, `maxls`, `damping`, etc.) or `ridge` across solves without triggering a JAX recompile, reuse a single solver instance and pass an updated `state` (via `dataclasses.replace(state, ridge=..., hyper=dataclasses.replace(state.hyper, damping=...))`) into `solver.solve(x0, state=...)`.

During a single outer `update`, `args`, `p`, and the carried `preconditioner` are held fixed across all backtracking trials so that every trial compares the same deterministic merit function.

## Penalized RKHS interpolation with `ridge`

Passing a positive `ridge` to `Newton` minimizes the penalized objective \(L(x) = \|r(x)\|_2^2 + \text{ridge} \cdot \|x_m\|_W^2\) using the normal-equations linear solvers that support penalty rows (`Cholesky()`, `QR()` which resolves to `form="normal"`, and `CG(precond)`). Like `LevenbergMarquardt(..., ridge=...)`, penalized `Newton` uses conjunctive stopping `(gtol | xtol) & atol` (so passing `atol > 0` alone raises `ValueError`), populates `info.ridge`, `info.resid_loss`, `info.penalty_value`, and `info.penalty_grad_norm`, and works directly with `AnnealRidge`:

```python
import jax.numpy as jnp
from nlls_gram import AnnealRidge, CholeskyMetric, Newton

metric = CholeskyMetric(jnp.linalg.cholesky(K))
solver = Newton(
    residual_fn,
    metric=metric,
    ridge=1e-2,
    geodesic_acceleration=True,
)
anneal = AnnealRidge(ridge_floor=1e-6, decrease=0.1, grad_rtol=1e-1)
result = solver.solve(
    x0,
    callback=anneal,
    user_state=anneal.init_state(),
    gtol=1e-6,
    max_steps=50,
)
```

## Minimal wide interpolation and manual stepping

```python
import jax
import jax.numpy as jnp
from nlls_gram import Newton, QR, SolverStatus

key = jax.random.key(0)
A = jax.random.normal(key, (8, 24))
b = jnp.ones(8)
x0 = jnp.zeros(24)

solver = Newton(lambda x: A @ x - b, linear_solver=QR())
result = solver.solve(x0, atol=1e-6, max_steps=10)
assert int(result.status) == int(SolverStatus.CONVERGED)

# Manual init/update loop
state = solver.init(x0)
x, state, info = solver.update(x0, state)
```

## Callbacks

Callbacks receive a `CallbackContext` after each outer update and return a `SolverAction`. A callback can log the accepted step size or adjust the line-search budget inside `state.hyper` without recomputing derivatives:

```python
import dataclasses
import jax.numpy as jnp
from nlls_gram import CallbackContext, Newton, SolverAction


def adapt_linesearch(ctx: CallbackContext):
    # Tighten maxls once the residual is small, and accumulate total trial count
    tighter_maxls = jnp.where(ctx.info.loss < 1e-2, jnp.int32(10), jnp.int32(30))
    new_hyper = dataclasses.replace(ctx.state.hyper, maxls=tighter_maxls)
    new_state = dataclasses.replace(ctx.state, hyper=new_hyper)
    return SolverAction(
        state=new_state,
        user_state=ctx.user_state + ctx.info.num_linesearch_steps,
    )


solver = Newton(residual_fn)
result = solver.solve(
    x0,
    atol=1e-6,
    callback=adapt_linesearch,
    user_state=jnp.int32(0),
)
```

## Linear solver compatibility

When `damping == 0` (`Newton` is undamped), each linear solver operates only on shapes where its undamped operator is well-posed:

| Config | Forward shapes | Rank-deficient safe | Notes |
|---|---|---|---|
| `Cholesky(form="auto")` | Any full-rank (\(m \le n\) Gram, \(n \le m\) normal) | No | Default. Squares \(\text{cond}(J)\) |
| `Cholesky(form="gram")` | \(m \le n\) only | No | Factors \(J J^\top \in \mathbb{R}^{m \times m}\) |
| `Cholesky(form="normal")` | \(n \le m\) only | No | Factors \(J^\top J \in \mathbb{R}^{n \times n}\) |
| `QR(form="auto")` | Any full-rank (\(m \le n\) Gram, \(n \le m\) normal) | No | Single thin QR per outer step at \(\text{cond}(J)\) |
| `QR(form="gram")` | \(m \le n\) only | No | Thin QR of \(J^\top\) |
| `QR(form="normal")` | \(n \le m\) only | No | Thin augmented QR of \([J \mid r]\) |
| `SVD()` | Any \(m, n\) | Yes | Truncates singular values at \(\max(m, n)\,\varepsilon\,\sigma_{\max}\) |
| `LU()` | Square \(m = n\) only | No | Direct nonsymmetric LU of \(J\) |
| `GramCG(precond)` | \(m \le n\) only | Needs range-safe right-hand side | Matrix-free CG on \(J J^\top\) in residual space |
| `CG(precond)` | \(n \le m\) only | No (`penalty` rejected in forward role) | Matrix-free CG on \(J^\top J\) in parameter space |

Preconditioners with `requires_positive_damping=True` are rejected because the Newton operator is undamped.

## Minimum-norm corrections versus minimum-norm roots

Each outer step \(d = -\bar{F}^{-1}(J(x_k)\bar{F}^{-1})^+ r(x_k)\) in unpenalized `Newton` is the minimum-\(W\)-norm correction to the current iterate \(x_k\). For a linear system \(r(x) = A x - b\), a single full step from \(x_0\) lands at \(x^* = x_0 - \bar{F}^{-1}(A\bar{F}^{-1})^+(A x_0 - b)\), which is the \(W\)-orthogonal projection of \(x_0\) onto the affine solution space (the minimum-\(W\)-norm root only when \(x_0 = 0\) or \(\bar{F} x_0 \in \text{range}((A\bar{F}^{-1})^\top)\)). For nonlinear underdetermined problems where \(J(x)\) rotates across iterations, a sequence of minimum-norm corrections is path-dependent and does not in general minimize \(\|x - x_0\|_W\) globally. Pass a positive `ridge` to `Newton` or `LevenbergMarquardt` (optionally with `AnnealRidge`) when global minimum-\(W\)-norm root selection is required.

## Failure statuses and alternatives

`Newton` never switches algorithms silently on failure. Instead it preserves the pre-step iterate `x` and reports an explicit `SolverStatus` (unless a `callback` modifies the iterate, problem parameters, or solver state to retry the step):

- `SolverStatus.LINEAR_SOLVE_FAILED`: a Cholesky, QR, or LU factorization encountered a nonfinite or rank-deficient pivot, or the linear solve returned nonfinite values. Switch to `QR()`, set `damping > 0`, or use `SVD()` / `LevenbergMarquardt` when the Jacobian is rank-deficient.
- `SolverStatus.NON_DESCENT`: the linear solve succeeded with finite values, but \(s = g^\top u \ge 0\) at a non-root iterate (for example, at a nonzero local minimum where the whitened gradient \(g = 0\)). When `gtol > 0` and \(\|g\|_2 < \text{gtol}\), `solve()` reports `CONVERGED` by stationarity. When `gtol == 0`, `solve()` terminates with `NON_DESCENT`.
- `SolverStatus.STAGNATION`: the scaled trial step underflowed so that `x + step == x` in working precision (common on nonzero-residual problems in float32 if `gtol` is set below the floating-point gradient noise floor).
- `SolverStatus.LINE_SEARCH_FAILED`: all `maxls` trial step sizes were rejected (or `Cholesky()` lost precision on an ill-conditioned Jacobian so the computed direction is not a descent direction for the nonlinear objective). Switch to `QR()` if `Cholesky()` lost precision, increase `maxls` or `damping`, use `MultiStart`, or switch to `LevenbergMarquardt`.

Under implicit differentiation (`jax.grad`, `jax.jvp`, `jax.vjp`), only `SolverStatus.CONVERGED` (and `SolverStatus.MAX_STEPS` when `max_steps_is_success=True`) carry nonzero tangents; all failure statuses (`LINE_SEARCH_FAILED`, `LINEAR_SOLVE_FAILED`, `NON_DESCENT`, `STAGNATION`, `NONFINITE`) return exact zero tangents for `result.x` and `result.aux`.
