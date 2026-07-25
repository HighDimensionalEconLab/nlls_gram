# Implicit differentiation

`solve(...).x` carries a custom JVP rule with respect to `p`, so gradients do
**not** unroll the LM iterations. They differentiate the converged root
through the implicit function theorem, at a cost independent of `max_steps`.

## The rule

For a root \(r(x, a, p) = 0\) with \(J = r_x\) and \(K = r_p\), differentiating
gives \(J\,\dot x = -K\,\dot p\). That is underdetermined at an interpolating
root, and the metric picks the solution: with \(W = F^\top F\) and
\(B = JF^{-1}\),

$$
\dot x = -F^{-1}B^{+}K\dot p ,
$$

the **minimum-\(W\)-norm** tangent — the same selection the forward solve
makes, so the derivative is consistent with the solution it differentiates.
For `RidgeLevenbergMarquardt` the ridge makes the system positive definite and
the rule is the plain solve of \((\tilde J^\top\tilde J + \lambda E)\dot y =
-\tilde J^\top K\dot p\).

Reverse mode comes from JAX transposing that linear tangent program, so `grad`,
`vjp`, `hessian`, and `vmap` all work without a separate implementation.

## Choosing an `ad_solver`

`None`, the default, matches the forward family. Override it when the shape
or the rank says otherwise — see
[Metric LM](metric_lm.md#rank-deficiency-and-implicit-ad) for the table of
which rule is valid where. The short version:

- `SVD()` whenever the undamped system is singular by construction — padded
  zero residuals are the common case;
- `GramCG(precond)` for \(m \le n\) and `CG(precond)` for \(n \le m\) when the
  forward solve is matrix-free and you want the tangent to stay so;
- `Cholesky()` otherwise.

A Krylov rule used outside its valid shape raises rather than returning a
quietly wrong tangent.

## Frozen instances

Under differentiation the **carried** metric and preconditioner instances —
the geometry the solve actually converged under, callback refreshes included
— are frozen inputs to the implicit system: their leaves ride along as inert
conditioning data under `stop_gradient`, and the state-dependence of a
callback-refreshed instance is not differentiated. An iterate-tracking
metric that wants the tangent taken at the exact solution geometry refreshes
on every accepted step, which makes the carried instance current at
convergence.

## Failed solves

Implicit AD uses `CONVERGED`, and also `MAX_STEPS` when
`max_steps_is_success=True` (the default). Every other status returns **exact
zero tangents** for `result.x` and `result.aux`. To keep automatic VJP
transposition finite under `vmap`, a failed lane's linear tangent program is
evaluated at differentiation-inert copies of the original `(x0, args, p)`, so
those initial values must be valid for JVP evaluation of the residual. The
primal failed result is never replaced.

`x0` gets exactly zero tangent by contract: the returned root is a property of
the problem, not of where the search started.

## Aux outputs

With `has_aux=True`, `result.aux` is evaluated at the returned `(x, args, p)`
and is differentiable with respect to `p` both directly and through
\(x^*(p)\).

```python
def residual(x, args, p):
    value = model_residual(x, args, p)
    return value, {"fit": summary(x, p)}

solver = LevenbergMarquardt(residual, has_aux=True)
grad = jax.grad(lambda p: solver.solve(x0, args, p=p, atol=1e-8).aux["fit"])(p)
```

`args`, `user_state`, the histories, and the multi-start diagnostics are all
differentiation-inert.
