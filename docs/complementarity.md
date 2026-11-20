# Inequality and Complementarity Constraints

`LevenbergMarquardt` and `Newton` support nonlinear inequality residuals and nonlinear complementarity conditions alongside equality residuals. Every constraint block maps into the flattened optimization residual vector \(r(\theta)\), so all linear solvers, Tikhonov ridge operators, geodesic acceleration, multi-start, and implicit differentiation rules compose directly.

## Residual containers

Return constraints either directly or inside `Residuals`:

```python
from nlls_gram import Complementarity, Inequality, Residuals


def residual_fn(x, args, p):
    return Residuals(
        residual=eq_residual(x, args, p),
        inequality=Inequality(g_margins(x, args, p)),
        complementarity=Complementarity(lhs=a_lhs(x, args, p), rhs=b_rhs(x, args, p)),
    )
```

- `residual` targets \(r_{\text{eq}}(\theta) = 0\).
- `Inequality(value)` targets feasibility \(g(\theta) \ge 0\). Nonnegative margins are feasible and negative margins are violations.
- `Complementarity(lhs, rhs)` targets \(0 \le \text{lhs}_i(\theta) \perp \text{rhs}_i(\theta) \ge 0\). Both `lhs` and `rhs` must share identical pytree structure and leaf shapes.

At least one of `residual`, `inequality`, or `complementarity` must be non-`None`.

## Nonlinear inequalities (`Inequality` and `InequalityPenalty`)

Configure inequality handling via `inequality=InequalityPenalty(...)` on `LevenbergMarquardt` or `Newton` (defaulting to `InequalityPenalty()` when `residual_fn` returns an `Inequality` block):

```python
from nlls_gram import Inequality, InequalityPenalty, LevenbergMarquardt, Residuals

solver = LevenbergMarquardt(
    lambda x, _, p: Residuals(residual=x - p, inequality=Inequality(x - 1.0)),
    inequality=InequalityPenalty(
        weight=10.0,
        smoothing=1e-2,
        smoothing_min=0.0,
        smoothing_decay=0.5,
    ),
)
```

For scalar or per-constraint objective weight \(\rho_g = \texttt{weight} \ge 0\) and smoothing temperature \(\tau = \texttt{smoothing} \ge 0\), the transformed inequality rows are \(r_g = \sqrt{\rho_g}\,\psi_\tau(g(\theta))\), where

$$
\psi_0(g) = \min(g, 0), \qquad
\psi_\tau(g) = -\tau \log\!\left(1 + e^{-g/\tau}\right) \quad (\tau > 0) .
$$

At \(\tau = 0\), feasible components \(g_i \ge 0\) have zero residual and zero derivative (`g < 0` branch).

## Complementarity reformulations

Configure complementarity via `complementarity=...` on `LevenbergMarquardt` or `Newton` (defaulting to `Minimum()` when `Complementarity` is returned). Let \(\tilde{a} = \text{lhs} / c_{\text{scale}}\) and \(\tilde{b} = \text{rhs}\). Because `w_comp` scales the residual rows, its weight in \(F_\rho(\theta)\) is \(w_{\text{comp}}^2\).

| Config | Residual \(r_c(\tilde{a}, \tilde{b})\) | Homotopy |
|---|---|---|
| `PDAS()` | \(w_{\text{comp}} \big(m \odot \tilde{a} + (1 - m) \odot \tilde{b}\big)\), with detached gate \(m = \mathbf{1}_{\tilde{a} \le \tilde{b}}\) (\(\tau = 0\)) or \(m = \sigma((\tilde{b} - \tilde{a}) / \tau)\) (\(\tau > 0\)) | `tau_init` \(\downarrow\) `tau_min` |
| `FischerBurmeister()` | \(w_{\text{comp}}\left(\tilde{a} + \tilde{b} - \sqrt{\tilde{a}^2 + \tilde{b}^2 + 2\mu^2 + \varepsilon_{\text{safe}}}\right)\) | `mu_init` \(\downarrow\) `mu_min` |
| `Minimum()` | \(w_{\text{comp}} \min(\tilde{a}, \tilde{b})\) | None |

In `FischerBurmeister`, \(\varepsilon_{\text{safe}}\) defaults to `jnp.finfo(dtype).tiny`, and the residual is evaluated in scale-normalized rationalized form to prevent cancellation when \(\tilde{a} + \tilde{b} > 0\). Its zero locus satisfies \(\tilde{a}\tilde{b} = \mu^2 + \varepsilon_{\text{safe}} / 2\).

## Continuation schedules and diagnostics

- Within each outer `update` (including Jacobian assembly, geodesic acceleration, and Armijo trial losses), `state.ineq_weight`, `state.ineq_smoothing`, `state.homotopy`, and the `PDAS` gate \(m\) remain frozen at the base iterate.
- After each step with `decay < 1.0` and active parameter \(\eta > \eta_{\min}\), `PDAS` (\(\tau\)) and `FischerBurmeister` (\(\mu\)) multiply \(\eta\) by `decay` whenever the step is accepted or the current subproblem satisfies \(\|g\|_2 \le \text{tol}\) or \(\|r\|_2 \le \text{tol}\), where \(\text{tol} = \max(0.01 \eta, \sqrt{\varepsilon_{\text{mach}}})\). `InequalityPenalty` applies the same decay to `state.ineq_smoothing` when the step is accepted or \(\|g\|_2 \le \max(0.01 \tau, \sqrt{\varepsilon_{\text{mach}}})\). Candidate values at or below \(\max(\eta_{\min}, \sqrt{\varepsilon_{\text{mach}}})\) (when \(\eta_{\min} = 0\)) snap directly to `0.0`.
- Convergence is suppressed while homotopy or smoothing parameters remain above their target floors, when an automatic smoothing decay fires (`info.inequality_changed=True`), when the `PDAS` partition flips (`info.complementarity_changed=True`), or when a callback modifies `state.ineq_weight` or `state.ineq_smoothing`.
- `LMInfo` and `NewtonInfo` report `loss` (\(2 F_\rho\)), `eq_loss` (\(\|r_{\text{eq}}\|_2^2 + \|r_c\|_2^2\) excluding inequality and ridge penalties), `ineq_violation_norm` (\(\|\max(-g, 0)\|_2\)), and `ineq_max_violation` (\(\max_i \max(-g_i, 0)\)). See [Implicit Differentiation](implicit_ad.md) for constraint sensitivity conventions.

## References

- Hintermüller, Ito, and Kunisch (2002) for the primal-dual active-set semismooth Newton method.
- Fischer (1992), Kanzow (1996), Chen and Mangasarian (1996), and Facchinei and Pang (2003) for smoothed Fischer-Burmeister, softplus, and sigmoid reformulations. Full citations appear in [References](references.md).
