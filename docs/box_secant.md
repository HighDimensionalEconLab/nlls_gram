# Univariate Box Optimization (`BoxSecant`)

`BoxSecant` minimizes (`sense="min"`) or maximizes (`sense="max"`) a scalar differentiable objective \(f(x, a, p)\) on a closed interval \([\ell, u] \subset \mathbb{R}\). Internally it minimizes \(h(x) = s\,f(x, a, p)\) with \(s = +1\) for `"min"` and \(s = -1\) for `"max"`, evaluating \((h(x), h'(x))\) in a single forward-mode `jax.jvp` per step and returning `result.fun` and `result.grad` in the caller's original orientation.

`fn` follows the same signature convention as `LevenbergMarquardt` and `Newton`, accepting `(x)`, `(x, args)`, or `(x, args, p)` and returning either `val` or `(val, aux)` when `has_aux=True`. Auxiliary outputs are evaluated once at \(x^*\) after the loop rather than carried inside loop state.

## Safeguarded projected secant-bisection step

Without a warm start (`x0=None`), iteration begins from symmetric interior probes \(\ell_0 + \frac{3}{8}(u_0 - \ell_0)\) and \(\ell_0 + \frac{5}{8}(u_0 - \ell_0)\), retaining the lower-\(h\) point as \(x_0\) and the other as \(x_{-1}\). When `x0` is supplied, \(x_{\text{init}} = \operatorname{clip}(x_0, \ell_0, u_0)\) is paired with a \(\frac{1}{32}(u_0 - \ell_0)\) probe step so already-optimal warm starts exit in `0` loop iterations under `loop="while"`.

At step \(k\) with bracket \([\ell_k, u_k]\), derivative \(g_k = h'(x_k)\), and secant curvature \(c_k = (g_k - g_{k-1}) / (x_k - x_{k-1})\):

1. Monotone bracket tightening sets \(\ell_{k+1} = \max(\ell_k, x_k)\) when \(g_k < 0\) and \(u_{k+1} = \min(u_k, x_k)\) when \(g_k > 0\).
2. When \(c_k > 0\), the projected secant trial \(x_{\text{proj}} = \operatorname{clip}(x_k - g_k / c_k, \ell_{k+1}, u_{k+1})\) is accepted if it lies strictly inside \((\ell_{k+1}, u_{k+1})\) with \(|x_{\text{proj}} - x_k| \le \frac{1}{2}(u_k - \ell_k)\) or hits an untouched outer bound \(\ell_0\) or \(u_0\) for the first time.
3. Otherwise the step falls back to bisection \(\frac{1}{2}(\ell_{k+1} + u_{k+1})\), except when \(c_k \le 0\) near an untouched outer bound, where jumping directly once to the downhill outer endpoint solves linear, concave, and inflection-near-boundary objectives without exhausting bisection steps (reverting cleanly to interior bisection if the endpoint itself evaluates non-finite).

## Machine-epsilon tolerances and second-order KKT stopping

When `xtol`, `rtol`, or `gtol` is `None`, defaults scale with \(\varepsilon = \texttt{jnp.finfo(dtype).eps}\):

| Parameter | Default formula | `float32` | `float64` |
|---|---|---|---|
| `xtol` | \(\varepsilon^{0.8}\) | \(\approx 2.95 \times 10^{-6}\) | \(\approx 3.95 \times 10^{-13}\) |
| `rtol` | \(4\varepsilon\) | \(\approx 4.77 \times 10^{-7}\) | \(\approx 8.88 \times 10^{-16}\) |
| `gtol` | \(\varepsilon^{0.8}\) | \(\approx 2.95 \times 10^{-6}\) | \(\approx 3.95 \times 10^{-13}\) |

Let \(\tau_x = \texttt{xtol} + \texttt{rtol}\,\max(|x_k|, \frac{1}{2}(|\ell_k| + |u_k|))\). Setting `xtol=0.0` and `rtol=0.0` disables bracket/step stopping, and setting `gtol=0.0` disables interior gradient stopping. A lane reports `SolverStatus.CONVERGED` when \(\ell_0 \ge u_0\), when active boundary KKT holds at \(\ell_0\) or \(u_0\), when \(u_k - \ell_k \le \tau_x\), or when interior stationarity holds with non-negative secant curvature (\(c_k \ge 0\)) so local maxima of \(h\) are never accepted as interior minima. On Ampere+ CUDA GPUs, set `jax.config.update("jax_default_matmul_precision", "highest")` when `fn` evaluates `float32` matrix products so 10-bit TF32 rounding (`~1e-3`) does not mask `eps ** 0.8` gradient convergence.

## Fused `vmap` batching and deterministic `num_starts`

`BoxSecant` implements recursive `jax.custom_batching.custom_vmap` rules (with automatic `jax.make_jaxpr` closure hoisting for functions that capture outer `vmap` or `jax.lax.map(..., batch_size=...)` tracers) so arbitrarily nested `vmap` calls execute inside a single fused loop:

- `loop="while"` (default) runs `jax.lax.while_loop` and exits as soon as every lane in the batch has converged, freezing converged lanes in place while stragglers finish.
- `loop="fori"` runs `jax.lax.fori_loop` for all `max_steps` iterations without a cross-lane `jnp.any(active)` reduction, which can reduce GPU launch overhead when `max_steps` is small (e.g. `6` to `8`).
- `num_starts > 1` deterministically partitions \([\ell, u]\) into `num_starts` uniform sub-boxes (and evaluates exact grid edges plus optional `x0`) inside one fused `vmap` without requiring a PRNG key, returning the candidate with the best objective value. When no converged candidate attains that value, a converged sub-box result within a relative `eps ** 0.8` of it is returned instead if it is an interior minimum of \([\ell, u]\) with nonnegative secant curvature or sits on a strictly active bound, and the slope at the best value points toward it.

## Corner Lagrange multipliers and implicit differentiation

`BoxSolveResult` exposes both `result.grad` (\(f_x(x^*, a, p)\)) and `result.lm = LagrangeMultipliers(lower=..., upper=...)`, which carries nonnegative KKT shadow prices (\(\lambda_\ell \ge 0, \lambda_u \ge 0\) in both `sense="min"` and `sense="max"`). With \(g_h = s\,f_x(x^*, a, p)\), active lower bounds set `lm.lower = max(g_h, 0)` and `lm.upper = 0`, active upper bounds set `lm.upper = max(-g_h, 0)` and `lm.lower = 0`, and strict interior optima set both multipliers to exact `0.0` (degenerate intervals \(\ell \ge u\) split the net multiplier \(g_h\) by sign).

`solve` attaches a custom JVP rule differentiating `(result.x, result.fun, result.grad, result.lm, result.aux)` with respect to `(lower, upper, p)` at the winning point \(x^*\) without unrolling loop iterations:

$$
\dot{x}^* =
\begin{cases}
\dot{\ell}, & \ell \ge u \;\text{or}\; (x^* \le \ell \;\text{and}\; s\,f_x \ge -\texttt{gtol}), \\
\dot{u}, & x^* \ge u \;\text{and}\; s\,f_x \le \texttt{gtol}, \\
-f_{xp}[\dot{p}] / f_{xx}, & \text{strict interior with } f_{xx} \ne 0,
\end{cases}
$$

with \(\dot{\texttt{fun}} = f_p[\dot{p}]\) on the interior and \(f_x \dot{x}^* + f_p[\dot{p}]\) on active bounds, \(\dot{\texttt{grad}} = 0\) on the interior and \(f_{xx} \dot{x}^* + f_{xp}[\dot{p}]\) on active bounds, \(\dot{\lambda}_\ell = s\,\dot{\texttt{grad}}\) on active lower bounds (`0` elsewhere), \(\dot{\lambda}_u = -s\,\dot{\texttt{grad}}\) on active upper bounds (`0` elsewhere), and \(\dot{\texttt{aux}} = D_x \text{aux}[\dot{x}^*] + D_p \text{aux}[\dot{p}]\) when `has_aux=True`. Failed lanes (`NONFINITE`, or `MAX_STEPS` with `max_steps_is_success=False`) mask tangents and reverse-mode cotangents to exact zero. Higher-order derivatives (`jax.hessian`, `jax.jacfwd(jax.jacrev(...))`) compose automatically.

## Example

```python
import jax
import jax.numpy as jnp
from nlls_gram import BoxSecant


def bellman_rhs(a, k, p):
    return k**0.6 - a - 0.5 * p["gamma"] * a**2 + p["beta"] * jnp.sqrt(0.9 * k + a)


solver = BoxSecant(bellman_rhs, sense="max")
ks = jnp.linspace(0.2, 2.0, 1024)
p0 = {"gamma": jnp.asarray(4.0), "beta": jnp.asarray(0.95)}

res = jax.vmap(lambda k: solver.solve(0.0, 1.0, k, p=p0))(ks)
val, grad_p = jax.value_and_grad(
    lambda p: jnp.mean(jax.vmap(lambda k: solver.solve(0.0, 1.0, k, p=p).fun)(ks))
)(p0)
```

## References

- Brent (1973) for safeguarded secant-bisection root-finding and 1D minimization on bounded intervals.
- Blondel et al. (2022) and Nocedal and Wright (2006) for active-set KKT sensitivity of box-constrained optima. Full citations appear in [References](references.md).
