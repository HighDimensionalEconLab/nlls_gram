# Implicit Differentiation

`solve(...).x` carries a custom JVP rule with respect to `p`, differentiating the converged solution via the implicit function theorem at a cost independent of `max_steps`. Reverse-mode `jax.grad`, `jax.vjp`, `jax.jacrev`, and `jax.vmap` work automatically by JAX transposition of the linear tangent rule.

## Mathematical rule

Let \(r(x, a, p) \in \mathbb{R}^m\) denote the complete optimization residual at the converged iterate \(x^*\) (evaluated at the frozen final ridge, inequality, and complementarity parameters), with \(J = D_x r(x^*, a, p)\) and \(K = D_p r(x^*, a, p)\).

- **Unpenalized (`ridge=None`)**: solves the undamped Gauss-Newton tangent equation
  $$
  \dot{x} = -J^+ K \dot{p} ,
  $$
  returning the Euclidean minimum-norm tangent on underdetermined interpolating manifolds and the linearized least-squares tangent on overdetermined systems.
- **Tikhonov-penalized (`ridge` \(\rho > 0\))**: solves the undamped regularized Gauss-Newton stationarity system
  $$
  \left(J^\top J + \rho Q\right) \dot{x} = -J^\top K \dot{p} ,
  $$
  returning the Euclidean minimum-norm augmented least-squares tangent when `ad_solver=SVD()` and \(\ker(J) \cap \ker(L) \ne \{0\}\).

On nonzero-residual least-squares problems (\(r(x^*) \ne 0\)), this rule uses the Gauss-Newton sensitivity equation, omitting the second-order curvature terms \(\sum_i r_i \nabla^2_{xx} r_i\) and \((D_p J^\top) r\). It is exact whenever \(r(x^*) = 0\), or when \(r\) is affine in \(x\) with \(D_p J = 0\), and first-order accurate in \(\|r(x^*)\|_2\) otherwise.

## Selecting `ad_solver`

Any linear solver valid at \(\mu = 0\) for the problem formulation (`Cholesky`, `QR`, `CG`, `GramCG`, `LU`, or `SVD`) can serve as `ad_solver`. When `ad_solver=None` (the default):

- With `ridge=None`: uses `GramCG` or `CG` when the forward solver is a compatible Krylov method (`m <= n` for `GramCG`, `n <= m` for `CG`), `LU()` when \(m = n\) without inequalities, and `SVD()` otherwise.
- With `ridge > 0`: uses `CG` or `GramCG` when the forward solver is a compatible Krylov method, `SVD()` when the forward solver is `SVD` or when the augmented operator lacks guaranteed full rank on a wide system, `QR` or `Cholesky` (preserving `form` and `preconditioner`) when the forward solver is valid at \(\mu = 0\), and `Cholesky()` otherwise.

## Frozen conditioning and failed solves

Carried `ridge`, `ridge_metric`, `preconditioner`, `homotopy`, `ineq_weight`, and `ineq_smoothing` values are treated as differentiation-inert inputs under `jax.lax.stop_gradient`.

Implicit AD differentiates results with `SolverStatus.CONVERGED`, plus `SolverStatus.MAX_STEPS` when `max_steps_is_success=True` (the default). All failure statuses return exact zero tangents for `result.x` and `result.aux`. Under `jax.vmap`, failed lanes evaluate the masked linear tangent program at stop-gradient copies of the initial `(x0, args, p)` so VJP transposition stays finite across every batch lane.

With `has_aux=True`, `(residual, aux)` is jointly linearized once at `(result.x, result.args, p)` so derivatives flow into `aux` both directly through `p` and indirectly through \(x^*(p)\) without redundant forward evaluations. Under `jax.jacfwd`, `jax.jacrev`, `jax.linearize` + `jax.vmap`, and `jax.vjp` + `jax.vmap`, the primal solve, joint residual linearization, Jacobian assembly, and dense AD matrix factorization execute once and are reused across all batched tangent or cotangent right-hand sides.
