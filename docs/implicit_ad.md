# Implicit Differentiation

`solve` has a custom implicit JVP/VJP with respect to `p` for the solved
parameters:

```python
solver.solve(x0, args, p=p).x
```

The custom rule is not defined on the per-step `update(...)` interface, and it
does not differentiate through the LM iterations. It differentiates the residual
equation at the returned solution. For implicit differentiation, use a fixed
`args` and read the differentiated value from `result.x`.

Here `p` means the external pytree argument passed to the residual function:

```python
residual_fn(x, args, p)
```

It does not mean LM hyperparameters such as `init_damping`, `max_steps`,
`atol`, callback choices, or metric callbacks. The custom rule treats `args` and
the initial guess `x0` as fixed for this derivative (their tangents are zero,
not an error).

There is no setup stage that AD must trace through: the whole iteration —
every update, callback, and the final aux evaluation — sits inside one
`jax.custom_jvp` boundary, so derivative information flows only through the
implicit rule at the returned solution. `init` is differentiation-inert (its
outputs are constants whose shapes and dtypes come from one residual
evaluation), so calling it by hand inside a differentiated function, or
implicitly via `cache_jacobian=True`, contributes exactly zero to any
derivative. `result.aux` also participates in the implicit rule — see
[Aux outputs](#aux-outputs) below. One
construction-time caveat: the solver (including any `Metric` callbacks) is a
static object — do not build it from traced values.

## Root Selection and the Metric

In underdetermined interpolation problems there may be many roots
\(\theta\) satisfying

$$
r(\theta, a, p)=0,
$$

where \(a\) denotes fixed auxiliary data from `args`. A perturbation \(\dot p\)
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
to the `p` argument only, evaluated at fixed \(\theta^\star\) and `args`.

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

In code, \(P x\) is applied with `metric.solve(x)` when available. If a QR/LSMR
metric is supplied only through square-root callbacks, the same inverse metric is
applied as \(P x = S S^\top x\) using `metric.inv_sqrt` and
`metric.inv_sqrt_transpose`.

## VJP

The transpose of the same map gives the VJP. For a cotangent
\(\bar\theta\) on `result.x`, solve

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


# p without args still uses the three-argument form; the second argument is
# simply ignored.
def residual(theta, _, p):
    return jnp.array([theta[0] + 2.0 * theta[1] - p])


solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2)
theta0 = jnp.zeros(2)


def solved_x(p):
    return solver.solve(theta0, p=p, max_steps=80, atol=1e-6).x


theta, theta_dot = jax.jvp(
    solved_x,
    (jnp.asarray(3.0),),
    (jnp.asarray(0.7),),
)

theta, pullback = jax.vjp(solved_x, jnp.asarray(3.0))
(p_bar,) = pullback(jnp.array([3.0, 4.0]))
```

Here \(J_\theta=[1,2]\), so the identity-metric tangent is
\(\dot\theta = [1,2]\dot p / 5\), and the VJP maps
\(\bar\theta\) to \((\bar\theta_0 + 2\bar\theta_1)/5\).

## Aux outputs

With `has_aux=True` the residual returns \((r, a)\); write the aux output map
as

$$
a = g(\theta, \text{args}, p),
\qquad
a = g(\theta^\star, \text{args}, p)
\text{ at the returned solution,}
$$

and define its two Jacobians at the solution (with \(k\) the flattened aux
dimension):

$$
G_\theta = \frac{\partial g}{\partial \theta}(\theta^\star, \text{args}, p)
\in \mathbb R^{k \times n},
\qquad
G_p = \frac{\partial g}{\partial p}(\theta^\star, \text{args}, p)
\in \mathbb R^{k \times \dim p}.
$$

For a pytree aux/`p`, \(G_\theta \dot\theta\) and \(G_p \dot p\) mean the JAX
JVP of the aux output with respect to that argument only (the same convention
as \(J_p \dot p\) above); transposes mean the corresponding VJP.

`p` moves the aux through both paths — directly, and through the solution
\(\theta^\star(p)\) — and the implicit rule accounts for both. **JVP**: with
\(\dot\theta\) the minimum-\(M\)-norm implicit tangent above,

$$
\dot a = G_\theta\,\dot\theta + G_p\,\dot p
= \bigl(-G_\theta P J_\theta^\top (J_\theta P J_\theta^\top)^{-1} J_p
  + G_p\bigr)\,\dot p.
$$

**VJP**: cotangents \(\bar\theta\) on `result.x` and \(\bar a\) on
`result.aux` combine — the aux cotangent pulls back through
\(G_\theta^\top\) into the solution cotangent and through \(G_p^\top\)
directly:

$$
(J_\theta P J_\theta^\top)\, y
= J_\theta P\,(\bar\theta + G_\theta^\top \bar a),
\qquad
\bar p = -J_p^\top y + G_p^\top \bar a.
$$

Setting \(\bar a = 0\) recovers the `x`-only VJP above; setting
\(\bar\theta = 0\) gives the pure aux pullback.

Example — the residual from above with an aux that depends on `p` both ways:

```python
def residual(theta, _, p):
    r = jnp.array([theta[0] + 2.0 * theta[1] - p])
    return r, {"m": theta[0] * theta[1] + p**2}


solver = UnderdeterminedLevenbergMarquardt(residual, init_damping=1e-2, has_aux=True)


def solved_aux_m(p):
    return solver.solve(jnp.zeros(2), p=p, max_steps=80, atol=1e-6).aux["m"]


da_dp = jax.grad(solved_aux_m)(jnp.asarray(3.0))
```

Here \(\theta^\star = (p/5, 2p/5)\), so the aux value is
\(2p^2/25 + p^2\) and its derivative is \(4p/25 + 2p = 6.48\) at \(p=3\):
the \(4p/25\) comes through the solution path \(G_\theta\dot\theta\) and the
\(2p\) through the direct path \(G_p\dot p\).
