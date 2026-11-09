# Complementarity Constraints

`LevenbergMarquardt` and `Newton` can solve nonlinear complementarity problems (NCPs) and mixed complementarity problems (MCPs) alongside standard nonlinear least-squares residuals, in both underdetermined interpolation (\(m < p\)) and overdetermined (\(m \ge p\)) regimes. Every complementarity algorithm maps the complementarity conditions into a flattened effective residual vector, so the full suite of linear solvers, metrics, ridge penalties, geodesic acceleration, multi-start, and implicit differentiation works unchanged.

## Problem statement

Given a parameter pytree \(x \in \mathbb{R}^p\), optional equality residuals \(r(x, a, p) \in \mathbb{R}^{n_{\text{eq}}}\), and a pair of complementarity functions \(\text{lhs}(x, a, p), \text{rhs}(x, a, p) \in \mathbb{R}^{n_{\text{comp}}}\) with matching pytree structure, the solver targets

$$
r(x, a, p) = 0, \qquad
0 \le \text{lhs}_i(x, a, p) \;\perp\; \text{rhs}_i(x, a, p) \ge 0
\quad \text{for } i = 1, \dots, n_{\text{comp}} .
$$

The complementarity condition \(0 \le a_i \perp b_i \ge 0\) means that at each coordinate \(i\), both \(a_i \ge 0\) and \(b_i \ge 0\) hold, and at least one is zero (\(a_i b_i = 0\)). In economic models with occasionally binding constraints, \(\text{lhs}\) is typically the distance to a primal bound (such as \(a' - a_{\min} \ge 0\)) and \(\text{rhs}\) is the corresponding Euler slack or shadow multiplier.

## Residual return interface

A residual function can declare complementarity conditions by returning a `Residuals` container or, for pure complementarity systems with no equality equations, a bare `Complementarity` container. Both forms compose with `has_aux=True` by returning `(Residuals(...), aux)` or `(Complementarity(...), aux)`:

- **Mixed equality and complementarity system**
  ```python
  from nlls_gram import Complementarity, Residuals

  def residual_fn(x, args, p):
      eq_res = ...
      lhs, rhs = ...
      return Residuals(
          residual=eq_res,
          complementarity=Complementarity(lhs=lhs, rhs=rhs),
      )
  ```
- **Pure complementarity system (`residual=None`)**
  ```python
  def residual_fn(x, args, p):
      lhs, rhs = ...
      return Residuals(complementarity=Complementarity(lhs=lhs, rhs=rhs))
  ```
  or directly
  ```python
  def residual_fn(x, args, p):
      lhs, rhs = ...
      return Complementarity(lhs=lhs, rhs=rhs)
  ```

`residual`, `lhs`, and `rhs` may each be any JAX pytree of arrays (including scalars or nested dictionaries and tuples). The solver flattens each block via `ravel_pytree`. The two leaves of `Complementarity(lhs, rhs)` must flatten to 1D vectors of the same length \(n_{\text{comp}}\).

`Residuals` also reserves `equality=Equality(...)` and `inequality=Inequality(...)` fields for future constrained extensions. Passing a non-`None` `equality` or `inequality` currently raises `NotImplementedError`.

## Scaling and block weighting

Primal bound distances (`lhs`) and optimality residuals (`rhs`) often carry different physical units. Comparing or squaring them inside \(J^\top J\) without normalization can degrade conditioning near the switching boundary. All three complementarity algorithms share two scaling parameters that normalize the inputs before forming the complementarity residual:

1. `c_scale` (\(c > 0\), scalar or array of shape `(n_comp,)`, default `1.0`) scales `lhs` to match the units of `rhs`:
   $$
   \tilde{a}(x, a, p) = \frac{\text{lhs}(x, a, p)}{c}, \qquad
   \tilde{b}(x, a, p) = \text{rhs}(x, a, p) .
   $$
2. `w_comp` (\(w > 0\), scalar float, default `1.0`) weights the complementarity residual block relative to the equality residual block in the stacked least-squares vector:
   $$
   R_{\text{total}}(x, a, p) = \begin{bmatrix}
   r(x, a, p) \\
   w \cdot R_{\text{comp}}\!\left(\tilde{a}(x, a, p),\, \tilde{b}(x, a, p)\right)
   \end{bmatrix} \in \mathbb{R}^{n_{\text{eq}} + n_{\text{comp}}} .
   $$

When `residual=None`, \(R_{\text{total}}\) consists solely of \(w \cdot R_{\text{comp}}(\tilde{a}, \tilde{b}) \in \mathbb{R}^{n_{\text{comp}}}\).

## Algorithms

Configure the complementarity reformulation via the `complementarity` argument on `LevenbergMarquardt` or `Newton`. When `complementarity=None` (the default) and `residual_fn` returns a `Complementarity` block, the solver defaults to `PDAS()`.

| Config | Reformulation \(R_{\text{comp}}(\tilde{a}, \tilde{b})\) | Line-search evaluation | Homotopy parameter |
|---|---|---|---|
| `PDAS()` (default) | \(m \odot \tilde{a} + (1 - m) \odot \tilde{b}\) with detached mask \(m\) | Mask \(m^k\) frozen at base iterate \(x^k\) | \(\tau \downarrow \tau_{\min}\) (`tau_init`, `tau_min`, `tau_decay`) |
| `FischerBurmeister()` | \(\tilde{a} + \tilde{b} - \sqrt{\tilde{a}^2 + \tilde{b}^2 + 2\mu^2 + \varepsilon_{\text{safe}}}\) | Smooth / semismooth at fixed \(\mu^k\) | \(\mu \downarrow \mu_{\min}\) (`mu_init`, `mu_min`, `mu_decay`) |
| `Minimum()` | \(\operatorname{where}(\tilde{a} \le \tilde{b},\, \tilde{a},\, \tilde{b})\) | Re-evaluated dynamically at trial \(x_{\text{trial}}\) | None |

### Primal-Dual Active Set (`PDAS`)

`PDAS(c_scale=1.0, w_comp=1.0, tau_init=0.0, tau_min=0.0, tau_decay=0.5)` implements a semismooth Primal-Dual Active Set strategy with an optional logistic homotopy temperature \(\tau \ge 0\).

At the start of step \(k\), the solver evaluates \(\tilde{a}^k = \tilde{a}(x^k)\) and \(\tilde{b}^k = \tilde{b}(x^k)\) and forms a detached gate \(m^k\) using `jax.lax.stop_gradient`:

- **Hard mode (\(\tau = 0\))**
  $$
  m^k = \operatorname{stop\_gradient}\!\left(\mathbb{I}(\tilde{a}^k - \tilde{b}^k \le 0)\right) \in \{0, 1\}^{n_{\text{comp}}}
  $$
- **Soft homotopy mode (\(\tau > 0\))**
  $$
  m_\tau^k = \operatorname{stop\_gradient}\!\left(\sigma\!\left(\frac{\tilde{b}^k - \tilde{a}^k}{\tau^k}\right)\right) \in (0, 1)^{n_{\text{comp}}}, \qquad
  \sigma(z) = \frac{1}{1 + e^{-z}} .
  $$

In both modes, the complementarity residual for step \(k\) is the gated combination

$$
R_{\text{comp}}(\tilde{a}, \tilde{b}; m^k) = m^k \odot \tilde{a} + (1 - m^k) \odot \tilde{b} .
$$

Because \(m^k\) is detached during linearization, the Jacobian rows select \(\nabla \tilde{a}_i\) on the active set (\(\tilde{a}_i \le \tilde{b}_i\)) and \(\nabla \tilde{b}_i\) on the inactive set (\(\tilde{a}_i > \tilde{b}_i\)).

**Frozen-mask globalization.** Throughout step \(k\), including Jacobian linearization, the geodesic acceleration directional second derivative, and every trial evaluation \(x_{\text{trial}} = x^k + \Delta x\) inside the Levenberg-Marquardt gain ratio or Armijo backtracking line search, the mask \(m^k\) is held **frozen** at its base-point value. Freezing \(m^k\) ensures that the trial objective measures progress on the exact smooth subproblem that generated \(\Delta x\), preventing line-search rejection or chattering when a step crosses an active-set boundary. Once a step is accepted (`step_norm > 0`), the mask is refreshed at \(x^{k+1}\) and any active temperature \(\tau\) decays via

$$
\tau^{k+1} = \begin{cases}
\max(\tau_{\min},\, \tau_{\text{decay}} \cdot \tau^k) & \text{if } \tau_{\text{decay}} \cdot \tau^k > \tau_{\min} \\
0 & \text{if } \tau_{\min} = 0 \text{ and } \tau_{\text{decay}} \cdot \tau^k \le \tau_{\text{snap}}
\end{cases}
$$

where \(\tau_{\text{snap}} = \sqrt{\varepsilon_{\text{mach}}}\) (\(\approx 1.49 \times 10^{-8}\) in float64 and \(3.45 \times 10^{-4}\) in float32).

### Smoothed Fischer-Burmeister (`FischerBurmeister`)

`FischerBurmeister(c_scale=1.0, w_comp=1.0, mu_init=0.0, mu_min=0.0, mu_decay=0.5, eps_safe=None)` reformulates each complementarity pair via the smoothed Fischer-Burmeister NCP function

$$
\phi_{\text{FB}}(\tilde{a}, \tilde{b}; \mu) = \tilde{a} + \tilde{b} - \sqrt{\tilde{a}^2 + \tilde{b}^2 + 2\mu^2 + \varepsilon_{\text{safe}}} ,
$$

whose zeros satisfy \(\tilde{a} \ge 0\), \(\tilde{b} \ge 0\), and \(\tilde{a}\tilde{b} = \mu^2\) as \(\varepsilon_{\text{safe}} \to 0\). Here `eps_safe` defaults to `jnp.finfo(dtype).tiny` (\(10^{-30}\) in float64 and \(10^{-12}\) in float32) so the square root remains differentiable at \((\tilde{a}, \tilde{b}, \mu) = (0, 0, 0)\).

To avoid catastrophic cancellation when \(\tilde{a} \gg \tilde{b} \ge 0\) or \(\tilde{b} \gg \tilde{a} \ge 0\) (especially in float32) and to prevent intermediate overflow or underflow when squaring large or tiny inputs, the implementation rescales each coordinate by the detached positive homogeneity factor

$$
s = \operatorname{stop\_gradient}\!\left(\max\!\left(|\tilde{a}|,\, |\tilde{b}|,\, \sqrt{2\mu^2 + \varepsilon_{\text{safe}}}\right)\right), \qquad
\hat{a} = \frac{\tilde{a}}{s}, \qquad
\hat{b} = \frac{\tilde{b}}{s}, \qquad
\hat{q} = \frac{2\mu^2 + \varepsilon_{\text{safe}}}{s^2} ,
$$

with \(s\) replaced by \(1\) when \(s = 0\), and evaluates the two-branch rationalized identity

$$
\phi_{\text{FB}}(\tilde{a}, \tilde{b}; \mu) = s \cdot \begin{cases}
\dfrac{2\hat{a}\hat{b} - \hat{q}}{\hat{a} + \hat{b} + \sqrt{\hat{a}^2 + \hat{b}^2 + \hat{q}}} & \text{if } \hat{a} + \hat{b} > 0 \\[10pt]
\hat{a} + \hat{b} - \sqrt{\hat{a}^2 + \hat{b}^2 + \hat{q}} & \text{if } \hat{a} + \hat{b} \le 0 .
\end{cases}
$$

Using `jnp.where` with safe denominators in the inactive branch keeps both value and reverse-mode derivatives finite everywhere. After each accepted step, the smoothing parameter \(\mu\) decays toward `mu_min` following the same geometric schedule as \(\tau\) in `PDAS`.

### Minimum residual (`Minimum`)

`Minimum(c_scale=1.0, w_comp=1.0)` applies the natural (minimum) NCP function directly:

$$
R_{\text{comp}}(\tilde{a}, \tilde{b}) = w_{\text{comp}} \cdot \operatorname{where}(\tilde{a} \le \tilde{b},\, \tilde{a},\, \tilde{b}) .
$$

Unlike `PDAS`, `Minimum` does not freeze the active-set branch during line search or carry a homotopy parameter. Trial steps re-evaluate \(\operatorname{where}(\tilde{a} \le \tilde{b}, \tilde{a}, \tilde{b})\) dynamically at \(x_{\text{trial}}\), and subgradients flow directly through `jnp.where` (selecting \(\nabla \tilde{a}_i\) on \(\tilde{a}_i \le \tilde{b}_i\) and \(\nabla \tilde{b}_i\) on \(\tilde{a}_i > \tilde{b}_i\)).

### Choosing an algorithm and tuning

- **Default to `PDAS()` or `Minimum()`** when `lhs` and `rhs` depend on the same policy parameters (such as a substituted Euler equation). Both take identical Gauss-Newton steps when full steps are accepted, run about 20–25% faster than `FischerBurmeister()`, and converge to full machine precision ($10^{-7}$ in `float32` and $10^{-15}$ in `float64`). `PDAS()` adds frozen-mask line-search globalization and suppresses premature convergence on steps where the active set flips.
- **Use `FischerBurmeister()` or `PDAS(tau_init=...)`** when `lhs` and `rhs` depend on different variable blocks (such as separate primal controls and shadow multipliers) and hard active-set selection risks zeroing out the Jacobian column of variables that only appear in the inactive branch at an initial guess. Because the Fischer-Burmeister partial derivatives \(1 - \tilde{a}/\sqrt{\tilde{a}^2 + \tilde{b}^2 + 2\mu^2}\) and \(1 - \tilde{b}/\sqrt{\tilde{a}^2 + \tilde{b}^2 + 2\mu^2}\) remain strictly positive whenever \(\tilde{a}, \tilde{b} \ne 0\), every row couples both branches even at \(\mu = 0\).
- **Keep homotopy off (`tau_init=0.0`, `mu_init=0.0`) by default.** Decaying \(\tau\) or \(\mu\) down to \(\sqrt{\varepsilon_{\text{mach}}}\) adds extra outer steps after the iterate is already near the root. Turn on `tau_init > 0` or `mu_init > 0` only when hard switching cycles across iterations or stalls from an infeasible start.
- **Prefer pure complementarity (`residual=None`) when an equality block is easy to substitute out.** In the underdetermined Gram form (`Cholesky(form="gram")`), substituting an accounting identity (such as a budget constraint or law of motion) into `Complementarity(lhs, rhs)` halves the number of residual rows \(m\), reducing the \(m \times m\) Gram build and Cholesky cost by roughly 2–3x on CPU and 1.5–2x on GPU.
- **Normalize mismatched units with `c_scale` and `w_comp`.** If `lhs` is measured in asset levels of order \(10^2\) while `rhs` is a marginal-utility Euler residual of order \(10^{-2}\), set `c_scale` to the typical ratio \(\|\text{lhs}\| / \|\text{rhs}\|\) so the active-set comparison \(\tilde{a} \le \tilde{b}\) and the Fischer-Burmeister sum \(\tilde{a}^2 + \tilde{b}^2\) weight both branches comparably.

## State, diagnostics, and stopping

Both `LMState` and `NewtonState` carry a scalar `homotopy` field (initialized to `tau_init` for `PDAS`, `mu_init` for `FischerBurmeister`, and `0.0` for `Minimum` or unconstrained problems). Each step's `LMInfo` and `NewtonInfo` report two complementarity fields:

- `info.homotopy` records the updated homotopy parameter after the step.
- `info.complementarity_changed` is a boolean scalar that is `True` whenever an accepted step flipped at least one active-set sign in `PDAS` (\(\mathbb{I}(\tilde{a}^{k+1} \le \tilde{b}^{k+1}) \ne \mathbb{I}(\tilde{a}^k \le \tilde{b}^k)\)) or whenever the homotopy parameter (\(\tau\) or \(\mu\)) is still strictly above its target floor (`tau_min` or `mu_min`).

The solver suppresses `CONVERGED` while `info.complementarity_changed` is `True`. This prevents `atol`, `gtol`, or `xtol` from declaring convergence on an intermediate homotopy level or on a step whose trial was accepted under a stale active-set partition before the refreshed partition has been tested for stationarity.

## Implicit differentiation

Custom JVP and VJP rules with respect to `p` work automatically for all three complementarity algorithms. At a converged solution \(x^*(p)\), the implicit differentiation rule linearizes the effective residual \(R_{\text{total}}(x^*, a, p) = 0\) at the final homotopy state `state.homotopy`:

- For `PDAS` at \(\tau^* = 0\) and for `Minimum`, the tangent system differentiates the active branch at each coordinate (\(\tilde{a}_i\) where \(\tilde{a}_i \le \tilde{b}_i\) and \(\tilde{b}_i\) where \(\tilde{a}_i > \tilde{b}_i\)), matching the classical sensitivity of an NCP at a strict complementarity solution.
- For `PDAS` terminated at a positive floor \(\tau^* = \tau_{\min} > 0\), the implicit linearization differentiates through the smooth sigmoid gate \(m_{\tau^*}(x^*, a, p)\) as well as \(\tilde{a}\) and \(\tilde{b}\), so the tangent matches the exact derivative of the smooth stationary point.
- For `FischerBurmeister`, the tangent system differentiates \(\phi_{\text{FB}}(\tilde{a}, \tilde{b}; \mu^*)\) with respect to both \(x\) and \(p\) at the converged smoothing parameter \(\mu^*\).

## Example: consumption-savings Euler equation

Consider a two-period consumption-savings problem on a grid of initial asset positions \(a_0 \in \mathbb{R}^m\) with income \(y\), gross interest rate \(R = 1 + r\), discount factor \(\beta\), CRRA utility \(u'(c) = c^{-\gamma}\), next-period consumption policy \(c_1(a')\), and a borrowing constraint \(a' \ge a_{\min}\). The optimality conditions at each grid point are

$$
c_0 + a' - (R a_0 + y) = 0, \qquad
0 \le a' - a_{\min} \;\perp\; c_0^{-\gamma} - \beta R\, c_1(a')^{-\gamma} \ge 0 .
$$

Substituting \(c_0 = R a_0 + y - a'\) eliminates the budget constraint and yields a pure complementarity system for an overdetermined or underdetermined basis expansion \(a'(a_0) = \Phi(a_0)\theta\). Alternatively, keeping \((c_0, a')\) as separate basis expansions gives a mixed equality and complementarity system:

```python
import jax
import jax.numpy as jnp
from nlls_gram import (
    Cholesky,
    Complementarity,
    FischerBurmeister,
    LevenbergMarquardt,
    PDAS,
    Residuals,
)

# 1. Pure complementarity formulation (budget constraint substituted in)
def euler_pure_comp(theta, args, p):
    a_next = args["basis"] @ theta
    c0 = p["R"] * args["a_grid"] + p["y"] - a_next
    c1 = (p["R"] - 1.0) * a_next + p["y"]
    lhs = a_next - p["a_min"]
    rhs = jnp.power(c0, -p["gamma"]) - p["beta"] * p["R"] * jnp.power(c1, -p["gamma"])
    return Complementarity(lhs=lhs, rhs=rhs)

solver = LevenbergMarquardt(
    euler_pure_comp,
    linear_solver=Cholesky(form="gram"),
    complementarity=PDAS(c_scale=1.0, w_comp=1.0),
)

# 2. Mixed equality + complementarity formulation
def euler_mixed(theta, args, p):
    theta_a, theta_c = jnp.split(theta, 2)
    a_next = args["basis"] @ theta_a
    c0 = args["basis"] @ theta_c
    c1 = (p["R"] - 1.0) * a_next + p["y"]
    budget_eq = c0 + a_next - (p["R"] * args["a_grid"] + p["y"])
    lhs = a_next - p["a_min"]
    rhs = jnp.power(c0, -p["gamma"]) - p["beta"] * p["R"] * jnp.power(c1, -p["gamma"])
    return Residuals(
        residual=budget_eq,
        complementarity=Complementarity(lhs=lhs, rhs=rhs),
    )

solver_fb = LevenbergMarquardt(
    euler_mixed,
    linear_solver=Cholesky(form="gram"),
    complementarity=FischerBurmeister(mu_init=1e-2, mu_min=0.0, mu_decay=0.2),
)
```

Because `result.x` carries the custom implicit differentiation rule with respect to `p`, `jax.jacrev(lambda y_val: solver.solve(theta0, args, p={**p, "y": y_val}).x)(p["y"])` differentiates the converged policy with respect to income \(y\) without unrolling the solver loop.

## API reference

::: nlls_gram.Residuals

::: nlls_gram.Complementarity

::: nlls_gram.PDAS

::: nlls_gram.FischerBurmeister

::: nlls_gram.Minimum

::: nlls_gram.Equality

::: nlls_gram.Inequality
