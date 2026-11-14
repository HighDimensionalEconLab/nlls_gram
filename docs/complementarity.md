# Inequality and Complementarity Constraints

`LevenbergMarquardt` and `Newton` support nonlinear inequality residuals and nonlinear complementarity conditions alongside ordinary equality residuals. Every constraint block maps into the flattened optimization residual vector \(r(\theta)\), so all linear solvers, Tikhonov ridge operators, geodesic acceleration, multi-start, and implicit differentiation rules compose directly.

## Residual containers

Return constraints either directly (for pure inequality or pure complementarity problems) or inside `Residuals`:

```python
from nlls_gram import Complementarity, Inequality, Residuals


def residual_fn(x, args, p):
    return Residuals(
        residual=eq_residual(x, args, p),
        inequality=Inequality(margins=g_margins(x, args, p)),
        complementarity=Complementarity(lhs=a_lhs(x, args, p), rhs=b_rhs(x, args, p)),
    )
```

- `residual` targets \(r_{\text{eq}}(\theta) = 0\).
- `Inequality(margins)` targets feasibility \(g(\theta) \ge 0\). Positive margins are feasible and negative margins are violations.
- `Complementarity(lhs, rhs)` targets \(0 \le \text{lhs}_i(\theta) \perp \text{rhs}_i(\theta) \ge 0\). Both `lhs` and `rhs` must share identical pytree structure and leaf shapes.

At least one of `residual`, `inequality`, or `complementarity` must be non-`None`.

## Nonlinear inequalities (`Inequality` and `InequalityPenalty`)

Configure inequality transformation via `inequality=InequalityPenalty(...)` on `LevenbergMarquardt` or `Newton` (defaulting to `InequalityPenalty()` when `residual_fn` returns an `Inequality` block):

```python
from nlls_gram import Inequality, InequalityPenalty, LevenbergMarquardt

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

For objective weight \(\rho_g = \texttt{weight} \ge 0\) and smoothing temperature \(\tau = \texttt{smoothing} \ge 0\), the effective inequality residual rows are \(\sqrt{\rho_g}\,\psi_\tau(g(\theta))\):

- **Exact mode (\(\tau = 0\))**: \(\psi_0(g) = \min(g, 0)\) implemented via strict `jnp.where(g < 0, g, 0.0)`, contributing \(\frac{1}{2}\rho_g \|\min(g(\theta), 0)\|_2^2\) to the half-objective \(F_\rho(\theta)\) with zero subgradient at feasible points \(g_i \ge 0\).
- **Smoothed mode (\(\tau > 0\))**: \(\psi_\tau(g) = -\tau \operatorname{softplus}(-g / \tau) = -\tau \log(1 + e^{-g/\tau})\) evaluated with overflow-safe double-`where` branching so values, Jacobians, second directional derivatives, and reverse-mode gradients remain finite across extreme positive or negative margins.

### Step consistency, continuation, and diagnostics

- Within each outer `update` (including Jacobian assembly, geodesic acceleration, and all LM or Armijo trial evaluations), `state.ineq_weight` and `state.ineq_smoothing` remain frozen so candidate steps and predicted reductions measure the exact subproblem that produced the step.
- After an accepted step, `state.ineq_smoothing` decays geometrically by `smoothing_decay` toward `smoothing_min` (snapping to `0.0` below \(\sqrt{\varepsilon_{\text{mach}}}\) when `smoothing_min == 0.0`).
- Callbacks may also adapt `state.ineq_weight` or `state.ineq_smoothing` (for example, increasing `ineq_weight` when `info.ineq_max_violation` exceeds a target). Any change to either scalar invalidates cached Jacobians and factorizations and sets `info.inequality_changed = True`, suppressing premature convergence on that step.
- `LMInfo` and `NewtonInfo` report `loss` (total \(2 F_\rho\)), `eq_loss` (\(\|r_{\text{eq}}\|_2^2\) excluding inequality and ridge penalties), `ineq_violation_norm` (\(\|\max(-g, 0)\|_2\)), and `ineq_max_violation` (\(\max_i \max(-g_i, 0)\)).

## Complementarity reformulations

Configure complementarity via `complementarity=...` on `LevenbergMarquardt` or `Newton` (defaulting to `PDAS()` when `Complementarity` is returned). With normalized \(\tilde{a} = \text{lhs} / c_{\text{scale}}\) and \(\tilde{b} = \text{rhs}\):

| Config | Residual \(w_{\text{comp}} R_{\text{comp}}(\tilde{a}, \tilde{b})\) | Homotopy |
|---|---|---|
| `PDAS()` | Detached active-set gate \(m^k \odot \tilde{a} + (1 - m^k) \odot \tilde{b}\) frozen across trial evaluations | `tau_init` \(\downarrow\) `tau_min` |
| `FischerBurmeister()` | Scale-normalized \(\tilde{a} + \tilde{b} - \sqrt{\tilde{a}^2 + \tilde{b}^2 + 2\mu^2 + \varepsilon_{\text{safe}}}\) | `mu_init` \(\downarrow\) `mu_min` |
| `Minimum()` | Dynamic `jnp.where(a_tilde <= b_tilde, a_tilde, b_tilde)` | None |

When an accepted step flips the `PDAS` active set or when the homotopy parameter remains above its target floor, `info.complementarity_changed` is `True` and convergence is suppressed until the updated partition is stationary.

## Implicit differentiation

Custom JVP and VJP rules differentiate the converged stationarity conditions at the frozen final penalty and homotopy state (`state.ineq_weight`, `state.ineq_smoothing`, `state.homotopy`). On square unpenalized problems with `inequality` blocks, `ad_solver=None` defaults to rank-safe `SVD()` rather than `LU()` because inactive exact inequality rows have zero Jacobian rows.
