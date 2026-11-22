# Implicit Differentiation

`solve(...).x` carries a custom JVP rule that computes Gauss-Newton sensitivities with respect to `p` at the returned solution without unrolling solver iterations. Reverse-mode `jax.grad`, `jax.vjp`, `jax.jacrev`, and `jax.vmap` work automatically by JAX transposition of the linear tangent rule.

## Sensitivity equations

Let \(r(x, a, p) \in \mathbb{R}^m\) denote the flattened optimization residual at the returned iterate \(x^*\) (evaluated at detached final ridge, inequality, and complementarity parameters), with \(J = D_x r(x^*, a, p)\) and \(B = D_p r(x^*, a, p)\).

- With `ridge=None`, the tangent \(\dot{x}\) solves the undamped Gauss-Newton equation
  $$
  \dot{x} = -J^+ B \dot{p} .
  $$
- With `ridge` \(\rho > 0\), \(\dot{x}\) solves the undamped regularized stationarity system
  $$
  \left(J^\top J + \rho Q\right) \dot{x} = -J^\top B \dot{p} ,
  $$
  returning the minimum-norm augmented least-squares tangent when `ad_solver=SVD()` and \(\ker(J) \cap \ker(L) \ne \{0\}\). When `ad_solver=CG(..., penalty=\delta)` is used with `penalty` \(\delta > 0\), an extra \(\delta I_n\) shift is added to the AD operator.

For a locally unique stationary point of \(F_\rho\), the exact implicit-function-theorem system includes the residual-weighted curvature terms \(\sum_i r_i D^2_{xx} r_i\) and \((D_p J^\top) r\). The Gauss-Newton rule omits those second-order residual terms, so it matches the exact implicit derivative at nonsingular roots with \(r(x^*) = 0\) (or affine \(r\) with \(D_p J = 0\)) and provides a first-order approximation in \(\|r(x^*)\|_2\) on nonzero-residual problems. On underdetermined manifolds (`ridge=None`, \(m < n\)), \(\dot{x} = -J^+ B \dot{p}\) selects the minimum-norm tangent orthogonal to \(\ker(J)\), which need not coincide with the derivative of a path-dependent primal root selector along \(\ker(J)\).

## Auxiliary outputs and batched Jacobians

When `has_aux=True` for \((r(x, a, p), h(x, a, p))\), both outputs are jointly linearized once at \((x^*, a, p)\) and the auxiliary tangent is evaluated by the total derivative

$$
\dot{h} = D_x h(x^*, a, p)[\dot{x}] + D_p h(x^*, a, p)[\dot{p}] ,
$$

retaining `float0` tangents on integer or boolean leaves.

Within a single batched differentiation call (`jax.jacfwd`, `jax.jacrev`, `jax.linearize` + `jax.vmap`, or `jax.vjp` + `jax.vmap`), the primal solve, joint linearization, dense Jacobian assembly, and matrix factorization execute once as functions of the primals and are reused across all tangent or cotangent right-hand sides.

## Selecting `ad_solver`

Any linear solver valid at \(\mu = 0\) can serve as `ad_solver`. When `ad_solver=None` (the default):

- With `ridge=None`, `solve` reuses `GramCG` (\(m \le n\)), `CG` (\(n \le m\) or `penalty != None`), or `SVD` when that is the forward solver, selects `LU()` when \(m = n\) with no inequality block present, and falls back to `SVD()` otherwise.
- With `ridge > 0`, `solve` reuses `SVD`, compatible `CG` (\(n_f \le m\) or `penalty != None`), or compatible `GramCG` (\(Q \succ 0\), \(n_f = 0\), `has_shifted_solve`), selects `SVD()` on factored wide or rank-indefinite problems where undamped Cholesky/QR is invalid, reuses `QR` or `Cholesky` (preserving `form` and `preconditioner`) when valid at \(\mu = 0\), and validates a `Cholesky()` fallback otherwise.

## Constraints and failed solves

Carried `ridge`, `ridge_metric`, `preconditioner`, `homotopy`, `ineq_weight`, and `ineq_smoothing` values are detached via `jax.lax.stop_gradient`. Under `PDAS`, hard active sets (\(\tau = 0\), the default `tau_min=0` target) detach the indicator gate and differentiate the active branch, whereas positive terminal smoothing (\(\tau > 0\)) differentiates the full smoothed residual \(w_{\text{comp}}(m \tilde{a} + (1 - m)\tilde{b})\) including \(Dm\). Inactive exact inequalities contribute zero Jacobian rows, so automatic `ad_solver` routing uses rank-safe `SVD()` whenever an inequality block is present.

Implicit AD differentiates results with `SolverStatus.CONVERGED` (plus `SolverStatus.MAX_STEPS` when `max_steps_is_success=True`). Failed solves return exact zero tangents for `result.x` and `result.aux`. Under `jax.vmap`, failed batch lanes evaluate the masked linear tangent program at detached copies of the initial `(x0, args, p)` so reverse-mode transposition remains finite across all lanes. For 1D box-constrained KKT sensitivities wrt `(lower, upper, p)`, see [Univariate Box Optimization](box_secant.md).

## References

- Griewank and Walther (2008), Bell and Burke (2008), and Blondel et al. (2022) for implicit differentiation of nonlinear equations and optimization problems.
- Golub and Pereyra (1973) for differentiation and pseudoinverse structure of separable and rank-deficient least squares. Full citations appear in [References](references.md).
