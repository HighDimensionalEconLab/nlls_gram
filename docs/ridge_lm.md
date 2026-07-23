# Ridge Levenberg–Marquardt

`RidgeLevenbergMarquardt` solves underdetermined interpolation problems by
minimizing a **ridge objective** whose small-ridge limit is the
minimum-seminorm interpolant. Where
[`LevenbergMarquardt`](index.md) encodes the selection in a parameter-space
*metric* (the algorithm's damping geometry), this solver encodes it in the
*objective* — every inner problem is a well-posed nonlinear least squares
that any solver can attack with textbook guarantees.

## Problem statement

Given an underdetermined residual \(r(x) \in \mathbb R^m\) over parameters
\(x \in \mathbb R^p\) (\(m < p\)) and a positive-semidefinite penalty factor
\(L\) (\(M_0 = L^\top L\), allowed singular), the target is the
**minimum-seminorm interpolant**

$$
x^\dagger \;=\; \operatorname*{argmin}_x \; \|L x\|^2
\quad\text{s.t.}\quad r(x) = 0 ,
$$

identified under the condition \(\ker J \cap \ker L = \{0\}\) at the solution
(Eldén 1982). For kernel coefficient problems \(\|Lx\|^2\) is the squared
RKHS norm, so \(x^\dagger\) is the minimum-RKHS-norm interpolant.

## The ridge family and why the objective carries the selection

The solver minimizes, for a strictly positive weight \(\lambda\) (the
`ridge`),

$$
F_\lambda(x) \;=\; \|r(x)\|^2 + \lambda\,\|L x\|^2 .
$$

Under the identification condition (plus standard local regularity of an
isolated constrained minimizer) each fixed-\(\lambda\) problem has an
isolated minimizer \(x_\lambda\), and

$$
x_\lambda \;=\; x^\dagger + O(\lambda) \quad\text{as } \lambda \downarrow 0
$$

— classical nonlinear Tikhonov regularization (Engl–Kunisch–Neubauer 1989;
Engl–Hanke–Neubauer 1996, Ch. 10). The selection lives in the objective on
purpose: plain Gauss–Newton on an underdetermined system converges to *some*
interpolant, and characterizing **which** one requires either an explicit
null-space correction (Campbell–Kunkel–Bobinyec 2012, relaxed in
Pes–Rodriguez 2022) or an affine residual (Izmailov–Solodov 2026). With the
ridge objective no such characterization is needed — the minimizer itself
carries the selection, at any solver accuracy.

## The algorithm

Everywhere in the implementation the equivalent **augmented-residual** view
is used: standard *Euclidean* LM on

$$
R(x) = \begin{bmatrix} r(x) \\ \sqrt{\lambda}\, L x \end{bmatrix},
\qquad
A = \begin{bmatrix} J \\ \sqrt{\lambda}\, L \end{bmatrix},
$$

with the damped step

$$
\left(A^\top A + \mu I\right) \delta
= -\left(J^\top r + \lambda L^\top L x\right) .
$$

Three properties follow from the penalty rows being **affine** in \(x\):

- the gradient identity \(A^\top R = J^\top r + \lambda M_0 x\) is exact;
- the penalty rows have zero second directional derivative, so **geodesic
  acceleration** (Transtrum–Sethna 2012) uses the true residual only —
  \(R_{vv} = [\,f_{vv};\,0\,]\) — and the standard formulas apply unchanged;
- accept/reject compares the plain scalar objective \(F_\lambda\) (Euclidean
  trust region, Marquardt 1963; Moré 1978; Nocedal–Wright 2006, Ch. 10).

The two scalars are fully decoupled:

- \(\mu\) (`damping`) is the trust-region parameter — it moves every step by
  accept/reject exactly as in stock LM;
- \(\lambda\) (`ridge`) is the *selection weight* — carried as traced state,
  changed only by `init()` or a `solve` callback, annealed monotonically
  toward a positive floor.

\(\lambda = 0\) is out of contract everywhere (constructor validation,
callback contract, continuation floors), so
\(J^\top J + \lambda M_0\) is positive definite under the identification
condition at every reachable state.

### Ridge continuation

The homotopy \(\lambda \downarrow \lambda_{\min}\) is a **documented
callback recipe**, not solver machinery:

```python
from nlls_gram import ridge_continuation

cb, us0 = ridge_continuation(ridge_floor=1e-8, decrease=0.1)
result = solver.solve(x0, callback=cb, user_state=us0,
                      gtol=1e-8, atol=1e-8, max_steps=500)
```

The callback multiplies `lm_state.ridge` by `decrease` whenever the inner
fixed-\(\lambda\) problem is approximately stationary (`info.grad_norm`
below `grad_rtol` relative to its reference at the current level), never
below `ridge_floor`. Two pacing notes from production use: the per-level
references compound (each level's reference is the gradient it *entered*
with), so a schedule that freezes below the noise floor wants a wider
`decrease` (e.g. `0.01` — larger jumps keep the references generous) or the
opt-in `stall_rtol` stagnation advance; and with the conjunctive stopping
rule, choose `atol` **between** the ridge-floor residual and the last
intermediate level's residual (they differ by roughly `1 / decrease`), so
the solve can only stop at the floor even when `gtol` must sit above a
problem-dependent stationarity noise floor. Solving each level out and passing to the limit is the
nonlinear Tikhonov path above; annealing per stationarity event is the
**iteratively regularized Gauss–Newton method** (Bakushinskii 1992;
Blaschke–Neubauer–Scherzer 1997; Kaltenbacher–Neubauer–Scherzer 2008), whose
theory wants exactly this monotone, boundedly geometric schedule. A callback
ridge change is treated as a *problem change*: that step's convergence test
is suppressed (its diagnostics were computed at the old \(\lambda\)) and the
ridge-keyed factorization caches invalidate.

### Choosing the floor and the tolerances

Two practical rules, both consequences of the stationarity identity
\(P_{\ker J}(M_0 x) = -g_{\text{null}}/\lambda\):

1. **The selection is resolved to `grad_norm / ridge`.** Stopping with
   gradient norm \(g\) at level \(\lambda\) leaves a null-space
   (selection) error of order \(g/\lambda\) — so `gtol` must sit well below
   `ridge` times the selection accuracy you want.
2. **Pushing the floor lower makes the answer worse past a point.** The
   achievable gradient norm is bounded by rounding
   (\(\sim\!10^{-15}\) scale in float64), so the total error
   \(O(\lambda) + O(\varepsilon_{\text{mach}}/\lambda)\) is minimized near
   \(\lambda^* \sim \sqrt{\varepsilon_{\text{mach}}}\) — about `1e-8` in
   float64. Only push \(\lambda\) below that with `linear_solver="qr"`
   (see the solver table), and expect no gain from floors below the
   gradient noise floor divided by your target accuracy.

Stopping is **conjunctive**: `gtol` (on the ridge stationarity
\(\|J^\top r + \lambda M_0 x\|\)) or `xtol` (accepted Euclidean step norm)
mean "done with the current fixed-\(\lambda\) problem", while `atol > 0`
*additionally* requires \(\|r\| \le\) `atol` — the ridgeless-endgame check —
and never stops the solve alone. A residual-only test would stop at step 0
from any interpolating start before the seminorm is minimized, so
`solve` rejects `atol > 0` without a positive `gtol` or `xtol`.

## Kernel instantiation

For kernel collocation with \(J\) repeated coefficient blocks (one per
equation stack) and \(s\) unpenalized structural scalars, the penalty is

$$
M_0 = \operatorname{blockdiag}(K, \ldots, K, 0_s),
\qquad
q(x) = \sum_j \alpha_j^\top K \alpha_j ,
$$

built by `repeated_dense_penalty(K, repeats=J, zero_pad_size=s)` — one
Cholesky \(K = CC^\top\), batched triangular products over the repeated
blocks, **no epsilon shift on the scalars** (they carry no penalty rows at
all, unlike the metric formulation's \(\varepsilon I\)). Each inner LM step
is then a kernel ridge regression of the relinearized equations, exactly the
Gauss–Newton scheme of Chen–Hosseini–Owhadi–Stuart (2021) with the ridge
kept explicit. `penalty_from_factor(L)` and `identity_penalty(size)` cover
generic dense factors and the minimum-Euclidean-norm case. An \(O(N)\)
state-space (Matérn) penalty pairing with the `lsmr` path is planned for a
later release; at the problem sizes this package targets the dense penalty
serves all three linear solvers.

A `RidgePenalty` provides `sqrt_apply` (\(Lx\)), `sqrt_transpose_apply`
(\(L^\top y\)), `num_rows`, and optionally `quadratic`, `add_scaled`
(\(H + cL^\top L\), used by the cholesky path), and `sqrt_rows` (the dense
\((k, p)\) factor, used by the qr path). An `x`- or `p`-dependent penalty is
deliberately unsupported in this release: the gradient and the implicit-AD
rule treat \(L\) as constant, and a dependent factor would silently drop
derivative terms (the reserved `penalty_factory` keyword raises).

## Linear solvers

All three solve \((J^\top J + \lambda M_0 + \mu I)\,\delta = -g\) and share
one factorization between the velocity and geodesic-acceleration solves.

| `linear_solver` | Method | Cost per update | When |
| --- | --- | --- | --- |
| `"auto"` = `"cholesky"` | dense normal equations; \(G = J^\top J + \lambda M_0\) assembled via `add_scaled` and **cached across rejected steps** (a reject re-factors in \(p^3/3\) without the GEMM or penalty assembly) | \(mp^2\) GEMM (skipped on reject) + \(p^3/3\) | default; fine for \(\lambda \gtrsim 10^{-8}\) in float64 |
| `"qr"` | QR of \([J;\sqrt{\lambda}L]\) cached per \((x, \lambda)\); each damping update re-factors \([R;\sqrt{\mu}I]\) and solves by corrected semi-normal equations with one refinement pass (Björck 1987; Björck 1996, §6.6.5; the damping-row structure is Moré 1978's) | \((m{+}k)p^2\) QR (skipped on reject) + \(2p^3/3\)-ish refactor | small \(\lambda\)/\(\mu\), where forming \(G\) squares the condition number |
| `"lsmr"` | matrix-free bidiagonalization (Fong–Saunders 2011) on the right-preconditioned augmented operator; requires an explicit `lsmr_preconditioner` (`identity_right_preconditioner()` opts out); the damping row is posed in the unpreconditioned variable, so every \(\mu > 0\) subproblem is exactly the \(I\)-damped one for any right preconditioner (Björck 1996, Ch. 7) | iterations × (one J and one Jᵀ product + penalty factor products) | Jacobians too large to materialize |

`linear_solve_dtype=jnp.float64` promotes the dense pipelines (cholesky and
qr) from a float32 program: \(J^\top\) is cast wide *before* the Gram
product, the penalty is added in the wide dtype, and only the returned step
is cast back. The gradient stays at the residual dtype by design — at tiny
\(\lambda\) the float32 gradient noise \(\sim 10^{-7}/\lambda\) bounds the
selection regardless of the solve dtype (rule 1 above).

## Implicit differentiation

`solve(...).x` has a custom implicit rule with respect to `p`: Gauss–Newton
differentiation of the ridge stationarity \(J^\top r + \lambda M_0 x = 0\),

$$
\left(J^\top J + \lambda M_0\right) \dot x
= -J^\top \frac{\partial r}{\partial p}\,\dot p ,
$$

with \(\lambda\) frozen (stop-gradient) at the returned state's ridge — the
continuation schedule's and the multi-start selection's dependence on `p`
are deliberately ignored — and **no damping** in the AD matrix.
`ad_solver="cholesky"` assembles and factors (wide under
`linear_solve_dtype`); `"normal_cg"` is matrix-free CG on the same operator
with an optional `ad_solver_preconditioner` (positive definite does not mean
well conditioned — unpreconditioned CG degrades as \(\lambda\) shrinks).
`"auto"` picks cholesky for the dense forwards and normal_cg for `lsmr`.

The contract, stated plainly: exact differentiation carries two extra terms
(\(\sum_i r_i \nabla^2 r_i\) in the matrix, \((\partial J^\top/\partial p)r\)
on the right), both dropped. Both are exactly zero for an affine residual
and first order in \(\|r\|\) otherwise — so the rule is exact in the
interpolating limit **in absolute terms**. On a genuinely underdetermined
*curved* system, however, the null-space block of the exact tangent carries
the constraint-curvature term \(\sum_i \nu_i \nabla^2 r_i\) with multiplier
\(\nu = r/\lambda\), which does not vanish relative to \(\lambda M_0\) as
\(\lambda \downarrow 0\): tangent components read off null-space directions
retain a Gauss–Newton bias proportional to the residual curvature. This is
the same approximation the metric solver's frozen-projector AD rules make.
Failed statuses return exact zero tangents and evaluate the masked tangent
program at stop-gradient copies of the original inputs and the *initial*
ridge.

## Migration from the metric formulation

```python
# before (metric LM): selection via the algorithm's damping geometry,
# epsilon-shifted so the metric stays invertible
metric = repeated_shifted_dense_metric(K, repeats=3, zero_pad_size=d, epsilon=1e-7)
solver = LevenbergMarquardt(residual_fn, metric=metric,
                            linear_solve_dtype=jnp.float64,
                            metric_solve_dtype=jnp.float64)
result = solver.solve(theta_0, max_steps=400, atol=2e-8)

# after (ridge LM): selection in the objective, no epsilon anywhere
penalty = repeated_dense_penalty(K, repeats=3, zero_pad_size=d)
solver = RidgeLevenbergMarquardt(residual_fn, penalty=penalty,
                                 ridge=1e-8,          # or None for the dtype default
                                 linear_solve_dtype=jnp.float64)
result = solver.solve(theta_0, max_steps=400, gtol=1e-8, atol=2e-8)

# optional homotopy if a fixed ridge converges slowly
cb, us0 = ridge_continuation(ridge_floor=1e-8, decrease=0.1)
result = solver.solve(theta_0, max_steps=400, gtol=1e-8, atol=2e-8,
                      callback=cb, user_state=us0)
```

Porting notes:

- **`info.loss` includes the penalty.** Code that means equation error must
  read `info.resid_loss`.
- **Residual-only `atol` stopping must add a `gtol` or `xtol`** — the
  conjunctive contract makes this loud, not silent.
- `metric_solve_dtype` has no ridge analog: penalty callbacks follow their
  input dtypes. If a float32 program needs a float64 penalty pipeline, that
  is a future `penalty_compute_dtype` knob.
- LSMR users must pass `lsmr_preconditioner=` explicitly
  (`identity_right_preconditioner()` at minimum) — deliberately stricter
  than the metric solver's optional `whitened_preconditioner=`.
- Multi-start ranking uses the ridge objective at each lane's own final
  ridge — comparable across lanes when they share a continuation schedule.

## References

- Bakushinskii, A. B. (1992). "The problem of the convergence of the
  iteratively regularized Gauss–Newton method." *Comput. Math. Math. Phys.*
  32(9), 1353–1359.
- Björck, Å. (1987). "Stability analysis of the method of seminormal
  equations for linear least squares problems." *Linear Algebra Appl.*
  88–89, 31–48.
- Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.
- Blaschke (Kaltenbacher), B., A. Neubauer, and O. Scherzer (1997). "On
  convergence rates for the iteratively regularized Gauss–Newton method."
  *IMA J. Numer. Anal.* 17(3), 421–436.
- Campbell, S. L., P. Kunkel, and K. Bobinyec (2012). "A minimal norm
  corrected underdetermined Gauß–Newton procedure." *Appl. Numer. Math.*
  62(5), 592–605.
- Chen, Y., B. Hosseini, H. Owhadi, and A. M. Stuart (2021). "Solving and
  learning nonlinear PDEs with Gaussian processes." *J. Comput. Phys.* 447,
  110668.
- Eldén, L. (1982). "A weighted pseudoinverse, generalized singular values,
  and constrained least squares problems." *BIT* 22, 487–502.
- Engl, H. W., M. Hanke, and A. Neubauer (1996). *Regularization of Inverse
  Problems*. Kluwer.
- Engl, H. W., K. Kunisch, and A. Neubauer (1989). "Convergence rates for
  Tikhonov regularisation of non-linear ill-posed problems." *Inverse
  Problems* 5(4), 523–540.
- Fong, D. C.-L., and M. A. Saunders (2011). "LSMR: An iterative algorithm
  for sparse least-squares problems." *SIAM J. Sci. Comput.* 33(5),
  2950–2971.
- Izmailov, A. F., and M. V. Solodov (2026). "Local convergence of the
  Gauss–Newton methods for constrained nonlinear equations." *Comput.
  Optim. Appl.* (doi:10.1007/s10589-026-00801-4).
- Kaltenbacher, B., A. Neubauer, and O. Scherzer (2008). *Iterative
  Regularization Methods for Nonlinear Ill-Posed Problems*. de Gruyter.
- Marquardt, D. W. (1963). "An algorithm for least-squares estimation of
  nonlinear parameters." *J. SIAM* 11(2), 431–441.
- Moré, J. J. (1978). "The Levenberg–Marquardt algorithm: implementation
  and theory." In *Numerical Analysis (Dundee 1977)*, Lecture Notes in
  Math. 630, Springer, 105–116.
- Nocedal, J., and S. J. Wright (2006). *Numerical Optimization*, 2nd ed.
  Springer, Ch. 10.
- Pes, F., and G. Rodriguez (2022). "A doubly relaxed minimal-norm
  Gauss–Newton method for underdetermined nonlinear least-squares
  problems." *Appl. Numer. Math.* 171, 233–248.
- Transtrum, M. K., and J. P. Sethna (2012). "Improvements to the
  Levenberg–Marquardt algorithm for nonlinear least-squares minimization."
  arXiv:1201.5885.
