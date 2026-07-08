# nlls_gram

`nlls_gram` provides metric-aware Levenberg-Marquardt nonlinear least-squares for
JAX pytrees. It is designed for underdetermined or interpolating problems where
the parameter dimension is often larger than the residual dimension.

The solver is intentionally small: users provide `residual_fn(params, aux, p)`,
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

The residual function is always called as:

```python
residual_fn(params, aux, p)
```

- `params` is the pytree optimized by LM.
- `aux` is arbitrary auxiliary data passed to `update(...)` or `solve(...)`.
- `p` is optional read-only data, useful for fixed deep parameters or outer
  perturbations.

`aux` and `p` may be any JAX pytree. Residual functions can ignore either one:

```python
def residual_fn(params, aux, p):
    x, y = aux
    del p
    return model(params, x) - y
```

`update(params, state, aux=None, p=None)` performs one LM step. The higher-level
`solve(params, aux=None, *, p=None, ...)` loop repeatedly calls `update` and
returns an `LMSolveResult`.

Callbacks passed to `solve` can replace `aux` for later iterations, for example
to regenerate collocation points or refresh simulation data. They receive `p`
but cannot replace it; this keeps the optimized solution's dependence on
external parameters explicit for future implicit-differentiation workflows.

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

## Metric Callbacks

Metric callbacks operate on the flattened parameter vector produced internally by
`ravel_pytree`. Let \(P=M^{-1}\), and let \(S\) satisfy \(SS^\top=M^{-1}\).

| Callback | Meaning |
| --- | --- |
| `metric_solve(x)` | \(M^{-1}x = Px\) |
| `metric_norm(x)` | \(\sqrt{x^\top Mx}\) |
| `metric_inv_sqrt(x)` | \(Sx\) |
| `metric_inv_sqrt_transpose(x)` | \(S^\top x\) |

Defaults are the identity metric:

```python
metric_solve = lambda x: x
metric_norm = lambda x: jnp.linalg.norm(x)
metric_inv_sqrt = lambda x: x
metric_inv_sqrt_transpose = lambda x: x
```

Shape requirements:

- `metric_solve` must support vectors `(n_params,)` and matrices
  `(n_params, k)`.
- `metric_inv_sqrt` must support vectors `(n_params,)`, and should support
  matrices when natural.
- `metric_inv_sqrt_transpose` must support matrices
  `(n_params, n_residuals)` for QR.
- `metric_norm` only needs to support vectors `(n_params,)`.

Validation rules:

- If all metric callbacks are omitted, the identity metric is used.
- For `linear_solver in {"cholesky", "cg"}`, a custom metric requires
  `metric_solve`.
- For `linear_solver in {"qr", "lsmr"}`, a custom metric requires both
  `metric_inv_sqrt` and `metric_inv_sqrt_transpose`.
- If `geodesic_acceleration=True` and a custom metric is supplied,
  `metric_norm` is required.
- The solver does not infer `metric_norm`, `metric_inv_sqrt`, or
  `metric_inv_sqrt_transpose` from `metric_solve`.

`metric_norm` is separate because `metric_solve` applies \(M^{-1}\), while the
norm needs \(M\):

$$
\|x\|_M = \sqrt{x^\top Mx}.
$$

Recovering that norm from a black-box \(M^{-1}\) solve would require another
inverse operation. Likewise, a square-root factor \(S\) is not generally
recoverable from an arbitrary solve callback.

## Linear Solver Formulas

### Cholesky

The default `linear_solver="cholesky"` materializes \(J^\top\), applies
`metric_solve`, and factors the residual-space dual system

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
`metric_inv_sqrt_transpose(J.T)` and solves the resulting augmented QR problem.
The returned parameter step is mapped back with `metric_inv_sqrt`.

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

For a dense metric \(M=LL^\top\), use:

```python
import jax.numpy as jnp

from nlls_gram import metric_callbacks_from_cholesky

L = jnp.linalg.cholesky(metric_matrix)
callbacks = metric_callbacks_from_cholesky(L)
```

The helper returns all four callbacks. It currently expects `lower=True`, the
default returned by `jnp.linalg.cholesky`.

## Minimal Example

```python
import jax
import jax.numpy as jnp

from nlls_gram import UnderdeterminedLevenbergMarquardt


def residual_fn(params, aux, p):
    x, y = aux
    del p
    return params["a"] * jnp.exp(params["b"] * x) - y


x = jnp.linspace(0.0, 2.0, 20)
y = 2.0 * jnp.exp(-1.0 * x)
params = {"a": 1.0, "b": 0.0}

solver = UnderdeterminedLevenbergMarquardt(residual_fn, init_damping=1e-2)
state = solver.init()


@jax.jit
def train_step(params, state):
    return solver.update(params, state, (x, y))


for _ in range(50):
    params, state, info = train_step(params, state)

print(params["a"], params["b"])
```

## Solve Loop and Callbacks

`solve` runs repeated LM updates and returns an `LMSolveResult`:

```python
result = solver.solve(
    params,
    aux=(x, y),
    p=None,
    max_steps=100,
    atol=1e-8,
)
params = result.params
```

`max_steps` is always enforced. `atol` is an optional residual-norm tolerance:
the solve converges when \(\sqrt{\sum_i r_i^2} < \texttt{atol}\). The default
`atol=0.0` disables tolerance stopping, so the solve runs until `max_steps`
unless a callback stops it. `solve(jit=True)` is the default and jits the loop;
use `jit=False` for debugging Python callbacks.

Status codes are integer constants:

| Status | Meaning |
| --- | --- |
| `LMStatus.CONVERGED` | Residual norm is below `atol`. |
| `LMStatus.MAX_STEPS` | `max_steps` was reached. |
| `LMStatus.NONFINITE` | The current loss is nonfinite, or a callback chose this status. |
| `LMStatus.CALLBACK_STOP` | A callback stopped without a custom status. |
| `LMStatus.RUNNING` | Internal running state, not a final successful status. |

Callbacks receive an `LMSolveContext`:

| Field | Meaning |
| --- | --- |
| `step` | One-based step number just completed. |
| `params`, `state`, `info` | Accepted/rejected step result and LM diagnostics. |
| `params_old`, `state_old` | Values before that step. |
| `initial_state` | State supplied to `solve` or created by `init`. |
| `aux`, `p`, `user_state` | Current auxiliary data, read-only external data, and user state. |

A callback returns `None` or `LMSolveAction(...)`. Omitted action fields are left
unchanged. The callback may set `stop`, `status`, `params`, `state`, `aux`, or
`user_state`; it cannot replace `p`.

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
    params,
    aux=(x, y),
    max_steps=100,
    callback=stopping_callback,
)
```

The per-step `update(...)` API remains useful when you want to write the outer
loop yourself or manage host-side logging.

## Implicit Differentiation

`solve` has a custom implicit JVP/VJP with respect to `p` for the solved
parameters:

```python
solver.solve(params0, aux, p=p).params
```

The custom rule is not defined on the per-step `update(...)` interface, and it
does not differentiate through the LM iterations. It differentiates the residual
equation at the returned solution. For implicit differentiation, use a fixed
`aux` and read the differentiated value from `result.params`.

Here `p` means the external pytree argument passed to the residual function:

```python
residual_fn(params, aux, p)
```

It does not mean LM hyperparameters such as `init_damping`, `max_steps`,
`atol`, callback choices, or metric callbacks. The custom rule treats `aux` and
the initial guess `params0` as fixed for this derivative.

### Root Selection and the Metric

In underdetermined interpolation problems there may be many roots
\(\theta\) satisfying

$$
r(\theta, a, p)=0,
$$

where \(a\) denotes fixed auxiliary data from `aux`. A perturbation \(\dot p\)
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
to the `p` argument only, evaluated at fixed \(\theta^\star\) and `aux`.

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

In code, \(P x\) is applied with `metric_solve(x)` when available. If a QR/LSMR
metric is supplied only through square-root callbacks, the same inverse metric is
applied as \(P x = S S^\top x\) using `metric_inv_sqrt` and
`metric_inv_sqrt_transpose`.

### VJP

The transpose of the same map gives the VJP. For a cotangent
\(\bar\theta\) on `result.params`, solve

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


def residual(theta, aux, p):
    del aux
    return jnp.array([theta[0] + 2.0 * theta[1] - p])


solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
theta0 = jnp.zeros(2)


def solved_params(p):
    return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).params


theta, theta_dot = jax.jvp(
    solved_params,
    (jnp.asarray(3.0),),
    (jnp.asarray(0.7),),
)

theta, pullback = jax.vjp(solved_params, jnp.asarray(3.0))
(p_bar,) = pullback(jnp.array([3.0, 4.0]))
```

Here \(J_\theta=[1,2]\), so the identity-metric tangent is
\(\dot\theta = [1,2]\dot p / 5\), and the VJP maps
\(\bar\theta\) to \((\bar\theta_0 + 2\bar\theta_1)/5\).

## Metric Example

```python
import jax.numpy as jnp

from nlls_gram import (
    UnderdeterminedLevenbergMarquardt,
    metric_callbacks_from_cholesky,
)


def residual_fn(theta, aux, p):
    matrix, target = aux
    del p
    return matrix @ theta - target


metric_matrix = jnp.array([[2.0, 0.2], [0.2, 1.0]])
L = jnp.linalg.cholesky(metric_matrix)

solver = UnderdeterminedLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    **metric_callbacks_from_cholesky(L),
)
```

## Matrix-Free Metric Example

For CG, the metric only needs a solve callback. This can wrap an iterative solve
or any JAX-linear operation implementing \(M^{-1}x\), without materializing a
dense inverse:

```python
import jax.scipy.sparse.linalg as jsp_sparse_linalg

from nlls_gram import UnderdeterminedLevenbergMarquardt


def metric_matvec(x):
    return kernel_matvec(x) + ridge * x


def metric_solve(x):
    solution, _ = jsp_sparse_linalg.cg(metric_matvec, x, maxiter=32)
    return solution


solver = UnderdeterminedLevenbergMarquardt(
    residual_fn,
    init_damping=1e-2,
    linear_solver="cg",
    metric_solve=metric_solve,
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

Therefore `metric_norm` is required whenever geodesic acceleration is combined
with a custom metric.

## Dtypes and Pytrees

Dtypes flow from `params` and the residual. `init(dtype=None)` uses JAX's default
floating dtype; pass an explicit dtype if needed. Enable x64 before creating
arrays if the problem should run in float64:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

`update` optimizes exactly the `params` pytree you pass. With Flax NNX, pass only
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

::: nlls_gram.metric_callbacks_from_cholesky

::: nlls_gram.LMState

::: nlls_gram.LMInfo

::: nlls_gram.LMStatus

::: nlls_gram.LMSolveAction

::: nlls_gram.LMSolveContext

::: nlls_gram.LMSolveResult
