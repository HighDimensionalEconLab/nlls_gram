# Unconstrained Scalar Optimization (`ScalarNewton`)

`ScalarNewton` minimizes (`sense="min"`) or maximizes (`sense="max"`) a scalar twice-differentiable objective \(f(x, a, p)\) over \(\mathbb{R}\) starting from an initial guess `x0`. Internally it minimizes \(h(x) = s\,f(x, a, p)\) with \(s = +1\) for `"min"` and \(s = -1\) for `"max"`, evaluating the second-order scalar jet \((h(x), h'(x), h''(x))\) in a single forward-over-forward `jax.jvp(jax.jvp(...))` pass with respect to `x` (without creating a reverse-mode activation tape for `p`). `ScalarSolveResult` returns `x`, `fun`, `grad`, and `curv` in the caller's original orientation alongside globalization diagnostics in `result.info`.

## Pluggable flat-loop globalization (`TrustRegion`)

Globalization is configured through `globalization=...` using subclasses of `ScalarGlobalization`. The default `TrustRegion()` strategy carries traced numeric fields (`init_radius`, `max_radius`, `eta_accept`, `eta_shrink`, `eta_expand`, `shrink_factor`, `expand_factor`) so varying numeric settings across calls reuses a single XLA compilation, returning `TrustRegionInfo(radius=..., step_norm=...)` in `result.info`.

At iterate \(x_k\) with \((h_k, g_k, c_k)\) and trust radius \(\Delta_k\) (defaulting to \(\Delta_0 = \min(\max(|x_0|, 1), \Delta_{\max})\) and \(\Delta_{\max} = 10^4 \Delta_0\)):

1. Under `loop="while"`, already-optimal warm starts (\(|g_0| \le \texttt{gtol}\) and \(c_0 \ge 0\)) exit immediately in `0` loop steps after a single jet evaluation.
2. Each active step minimizes the quadratic model \(m_k(s) = g_k s + \frac{1}{2} c_k s^2\) on \([-\Delta_k, \Delta_k]\) in closed form, proposing \(s_k = \operatorname{clip}(-g_k / c_k, -\Delta_k, \Delta_k)\) when \(c_k > 0\) and \(s_k = \pm \Delta_k\) downhill when \(c_k \le 0\) (stepping off exact concave stationary points \(g_k = 0, c_k < 0\)).
3. Non-finite trial evaluations (`NaN` or `Inf` from domain walls such as `log` or `sqrt`) or insufficient objective decrease reject the trial step in place and shrink \(\Delta_{k+1} = \texttt{shrink\_factor} \cdot \min(\Delta_k, |s_k|)\) within the same flat loop (`loop="while"` or `loop="fori"`), avoiding nested backtracking loops under `vmap`. Near stationary points in `float32`, steps with \(c_{\text{cand}} > 0\) that halve \(|g|\) within 4 ULPs of \(h_k\) are also accepted so quadratic polish is not blocked by floating-point rounding.

## Convergence criteria, `vmap`, and implicit differentiation

Default tolerances match [`BoxSecant`](box_secant.md) identically (`xtol = gtol = eps ** 0.8`, `rtol = 4 * eps`, `max_steps = 32`), reporting `SolverStatus.CONVERGED` when \(|g_k| \le \texttt{gtol}\) with \(c_k \ge 0\) or when step/radius localization reaches \(\texttt{xtol} + \texttt{rtol} \max(|x_k|, |x_0|)\) with \(c_k \ge 0\) and \(|g_k| \le \sqrt{\texttt{gtol}}\) (and `SolverStatus.STAGNATION` if \(\Delta_k\) collapses to machine precision without converging). On Ampere+ CUDA GPUs, set `jax.config.update("jax_default_matmul_precision", "highest")` when `fn` contains `float32` neural-network matmuls so 10-bit TF32 rounding (`~1e-3`) does not prevent reaching `float32` tolerances (`~3e-6`).

Recursive `jax.custom_batching.custom_vmap` rules fuse arbitrarily nested `vmap` calls into one loop, and `solve` attaches a `@jax.custom_jvp` rule differentiating `(x, fun, grad, aux)` with respect to `p` (`x0` and `args` are inert) at regular stationary points (\(f_{xx}(x^*, a, p) \ne 0\)):

$$
\dot{x}^* = -f_{xp}(x^*, a, p)[\dot{p}] / f_{xx}(x^*, a, p), \qquad
\dot{\texttt{fun}} = f_p(x^*, a, p)[\dot{p}], \qquad
\dot{\texttt{grad}} = 0 .
$$

Separating \(\dot{\texttt{fun}}\) from \(\dot{x}^*\) via the Envelope Theorem preserves exact second- and higher-order derivatives while allowing JAX dead-code elimination to skip mixed second-order backpropagation \(f_{xp}^\top\) through large parameter pytrees `p` whenever a loss depends only on `result.fun`.

## Example

```python
import jax
import jax.numpy as jnp
from nlls_gram import ScalarNewton, TrustRegion

jax.config.update("jax_default_matmul_precision", "highest")


def bellman_rhs(a, k, p):
    return k**0.6 - a - 1.5 * (a**2) / k + p["beta"] * jnp.sqrt(0.9 * k + a)


solver = ScalarNewton(bellman_rhs, sense="max", globalization=TrustRegion())
ks = jnp.linspace(0.4, 2.0, 1024)
a_init = jnp.full_like(ks, 0.1)
p0 = {"beta": jnp.asarray(0.95)}

res = jax.vmap(lambda a0, k: solver.solve(a0, k, p=p0))(a_init, ks)
val, grad_p = jax.value_and_grad(
    lambda p: jnp.mean(jax.vmap(lambda a0, k: solver.solve(a0, k, p=p).fun)(res.x, ks))
)(p0)
```
