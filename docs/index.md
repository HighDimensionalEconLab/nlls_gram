# nlls_gram

`nlls_gram` solves nonlinear equations, nonlinear least-squares problems, and generalized Tikhonov problems over JAX pytrees. Steps use scalar damping \(\mu I_n\) in parameter coordinates, and both outer solvers minimize

$$
F_\rho(\theta) = \frac{1}{2}\|r(\theta)\|_2^2 + \frac{\rho}{2}\theta^\top Q \theta ,
$$

reporting `info.loss` as \(2 F_\rho(\theta)\).

| Solver | Step damping \(\mu_k\) | Globalization |
|---|---|---|
| `LevenbergMarquardt` | Stateful scalar damping adapted from accepted Armijo step length | Armijo backtracking (optional geodesic acceleration) |
| `Newton` | Optional stateless residual-coupled damping \(\propto \|r(\theta_k)\|_2^p\) (or \(\mu_k = 0\)) | Armijo backtracking (optional geodesic acceleration) |

For univariate scalar optimization, [`BoxSecant`](box_secant.md) solves box-constrained problems over `[lower, upper]` (returning primal optima and corner `LagrangeMultipliers`), and [`ScalarNewton`](scalar_newton.md) solves unconstrained 1D problems from an initial guess `x0` using second-order forward jets and pluggable `TrustRegion` globalization. Both support fused `vmap` batching and implicit differentiation.

## Residual interface

`residual_fn` accepts `(x)`, `(x, args)`, or `(x, args, p)` and returns a residual pytree, a `Residuals` / `Inequality` / `Complementarity` container, or `(residual, aux)` when `has_aux=True`.

- `args` carries solver-inert auxiliary data.
- `p` carries external parameters with respect to which `solve(...).x` is implicitly differentiable.

```python
import jax
import jax.numpy as jnp
from nlls_gram import LevenbergMarquardt

design = jnp.arange(24.0).reshape(4, 6)
target = jnp.ones(4)


def residual(x, args, p):
    return args["design"] @ x - p["target"]


solver = LevenbergMarquardt(residual)
result = solver.solve(
    jnp.zeros(6), {"design": design}, p={"target": target}, max_steps=200, atol=1e-6
)
grad_p = jax.grad(
    lambda p: jnp.sum(
        solver.solve(jnp.zeros(6), {"design": design}, p=p, atol=1e-6).x ** 2
    )
)({"target": target})
```

## Next steps

- [Levenberg-Marquardt and Tikhonov](ridge_lm.md) for step equations, Armijo damping updates, geodesic acceleration, RKHS seminorms, and bordered Gram reductions.
- [Tuning and Capabilities](tuning_guide.md) for choosing linear solvers, preconditioning, Jacobian assembly, and `float32` GPU settings.
- [Implicit Differentiation](implicit_ad.md) for Gauss-Newton sensitivity equations and `ad_solver` routing.
