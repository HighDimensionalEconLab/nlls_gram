# Metric Gauss-Newton and Minimum-Norm Steps

This page explains the local behavior of the solver near the interpolation
threshold: with small damping, metric-aware Levenberg-Marquardt behaves like a
**metric Gauss-Newton** method, and the metric Gauss-Newton step is the
**minimum-\(M\)-norm** correction that solves the linearized residual
equations. Equivalently, ordinary Gauss-Newton in whitened coordinates is
metric Gauss-Newton in raw coordinates. For RKHS metrics this means the solver
selects minimum-RKHS-norm linearized corrections — and with kernel methods the
metric, hence the norm being minimized, can be carefully controlled.

Notation follows the [mathematical contract](index.md#mathematical-contract):
at the current flattened parameters \(\theta \in \mathbb R^n\), the residual is
\(r \in \mathbb R^m\), the Jacobian is \(J \in \mathbb R^{m \times n}\), the
step is \(s\), the damping is \(\lambda > 0\), and the metric is
\(M \succ 0\) with \(P = M^{-1}\) and
\(\|s\|_M = \sqrt{s^\top M s}\). Residual entries are always measured in the
Euclidean norm; \(M\) defines the geometry of *parameter* perturbations only.

## Why Gauss-Newton Is the Local Model Near Interpolation

The Hessian of \(\tfrac12\|r(\theta)\|_2^2\) decomposes as

$$
\nabla_\theta^2 \tfrac12\|r(\theta)\|_2^2
= J^\top J + \sum_{i=1}^m r_i \nabla_\theta^2 r_i.
$$

Near the interpolation threshold \(r_i \approx 0\), so the residual-weighted
curvature terms are negligible and \(J^\top J\) is the locally accurate model.
With a metric, the relevant local method is not Euclidean Gauss-Newton but its
metric version below.

## The Metric Gauss-Newton Step Is the Minimum-Norm Correction

Suppose the linearized equation \(J s = -r\) is feasible. With \(n > m\) it
typically has many solutions; metric Gauss-Newton selects the one with
minimum \(M\)-norm:

$$
s_{\mathrm{GN},M}
= \arg\min_{s} \tfrac12 \|s\|_M^2
\quad\text{subject to}\quad
J s = -r.
$$

The Lagrangian \(\tfrac12 s^\top M s + y^\top (J s + r)\) has first-order
conditions \(M s + J^\top y = 0\), so \(s = -P J^\top y\); imposing the
constraint gives \(J P J^\top y = r\) (invertible for full-row-rank \(J\);
the [rank-deficient case](#rank-deficiency) replaces it with a
pseudoinverse), and therefore

$$
s_{\mathrm{GN},M} = -P J^\top \left(J P J^\top\right)^{-1} r.
$$

For \(M = I\) this is the ordinary underdetermined Gauss-Newton /
pseudoinverse step \(s = -J^\top (J J^\top)^{-1} r\).

The \(m \times m\) matrix

$$
G_M = J P J^\top,
\qquad
(G_M)_{ij} = J_i P J_j^\top
$$

is the **metric Gram matrix**: the metric changes the induced inner product
between residual sensitivities \(J_i\), not the residual norm itself.

## Damping Interpolates Between Two Metric Methods

The solver's damped step (see the
[linear solver formulas](index.md#linear-solver-formulas)) is

$$
s_\lambda = -P J^\top \left(G_M + \lambda I_m\right)^{-1} r.
$$

**Small damping.** As \(\lambda \downarrow 0\) (with \(G_M\) nonsingular),
\(s_\lambda \to s_{\mathrm{GN},M}\): near interpolation, small-damping metric
LM is approximately the minimum-\(M\)-norm linearized residual correction.

**Large damping.** As \(\lambda \to \infty\),
\((G_M + \lambda I)^{-1} \approx \tfrac1\lambda I\), so

$$
s_\lambda \approx -\tfrac1\lambda P J^\top r
= -\tfrac1\lambda M^{-1} \nabla_\theta \tfrac12\|r(\theta)\|_2^2,
$$

which is steepest descent in the \(M\)-metric (a natural-gradient-style step),
not Euclidean gradient descent.

## Spectral Filter View

Let \(S\) satisfy \(S S^\top = M^{-1}\) and let the whitened Jacobian
\(J S\) have SVD \(J S = U \Sigma V^\top\). In whitened coordinates
\(s = S z\), the damped step is

$$
z_\lambda = -\sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}\, v_i (u_i^\top r),
\qquad
s_\lambda = S z_\lambda.
$$

The filter factor \(\sigma_i / (\sigma_i^2 + \lambda)\) acts direction by
direction: where \(\sigma_i^2 \gg \lambda\) it is \(\approx 1/\sigma_i\)
(Gauss-Newton-like), and where \(\sigma_i^2 \ll \lambda\) it is
\(\approx \sigma_i/\lambda\) (gradient-descent-like). LM is therefore
direction-wise between metric Gauss-Newton and metric gradient descent: as
damping falls near interpolation, accepted steps become Gauss-Newton-like on
the well-identified directions while poorly identified directions stay damped.
The effective number of active directions at damping \(\lambda\) is

$$
d_{\mathrm{eff}}(\lambda)
= \operatorname{tr}\!\left(G_M (G_M + \lambda I)^{-1}\right)
= \sum_i \frac{\sigma_i^2}{\sigma_i^2 + \lambda}.
$$

**The pseudoinverse limit.** As \(\lambda \downarrow 0\) each filter factor
converges to \(\sigma_i^{+}\) — \(1/\sigma_i\) where \(\sigma_i > 0\), and
exactly \(0\) where \(\sigma_i = 0\) — so

$$
z_\lambda \to -(JS)^{+} r,
\qquad
s_\lambda \to -S\,(JS)^{+} r,
$$

the minimum-\(M\)-norm least-squares correction of the
[rank-deficiency section](#rank-deficiency), with **no rank or shape
assumption**: redundant rows and collinear columns are filtered out
direction by direction, never inverted. The full-row-rank dual formula of
the earlier sections is the special case where the linearized least-squares
residual is zero. This primal limit is what every damped solver realizes as
damping falls near interpolation; how *accurately* a solver tracks it at
small \(\lambda\) is a conditioning question — the Gram and normal forms
work at the squared condition number of \(JS\), `lsmr` at
\(\operatorname{cond}(JS)\) itself.

## Whitened-Coordinate Equivalence

Metric Gauss-Newton in raw coordinates is ordinary Gauss-Newton in whitened
coordinates. With \(M = L L^\top\) and whitened coordinates
\(z = L^\top \theta\),

$$
\|s\|_M^2 = \|L^\top s\|_2^2,
\qquad
J_z = J L^{-\top},
$$

and the ordinary minimum-Euclidean-norm Gauss-Newton step
\(z\text{-step} = -J_z^\top (J_z J_z^\top)^{-1} r\) maps back to exactly
\(s_{\mathrm{GN},M}\). Passing whitened variables to an ordinary LM solver is
therefore equivalent to using the metric-aware solver in raw variables; the
metric-aware solver lets you stay in raw variables with the same
geometry. (This is precisely the substitution the whitened paths —
`normal_cholesky`, `normal_cg`, `qr`, `augmented_qr`, `lsmr` — make, with
\(S = L^{-\top}\).)

## Rank Deficiency

Without full row rank, replace the inverse with a pseudoinverse: writing
\(s = S z\) and \(A = J S\), the general metric Gauss-Newton step is

$$
s_{\mathrm{GN},M} = -S A^{+} r = -S (J S)^{+} r,
$$

the minimum-\(M\)-norm step among linearized least-squares minimizers. The
damped solvers remain well-posed for rank-deficient \(J\) (the `qr` path is
the exception — it [requires full row rank](index.md#qr)).

## Choosing the Metric with Kernels

The practical power of the metric is that kernel methods let you control it
exactly. Two standard parameterizations of a kernel function
\(f\) with Gram matrix \(K = [K(t_i, t_j)]_{ij}\):

**Kernel coefficients.** With \(f_\alpha(t) = \sum_j \alpha_j K(t, t_j)\), the
RKHS norm is \(\|f_\alpha\|_{\mathcal H_K}^2 = \alpha^\top K \alpha\), so the
parameter metric is \(M = K\) and

$$
s_{\mathrm{GN},M}
= -K^{-1} J^\top \left(J K^{-1} J^\top\right)^{-1} r
$$

is the correction that solves the linearized equations while minimizing the
RKHS norm of the *function* perturbation.

**Function values.** With parameters \(u_i = f(t_i)\), the minimum-norm
interpolant through \(u\) has \(\|f_u\|_{\mathcal H_K}^2 = u^\top K^{-1} u\),
so \(M = K^{-1}\) and \(P = K\):

$$
s_{\mathrm{GN},M}
= -K J^\top \left(J K J^\top\right)^{-1} r.
$$

| Parameterization | Function norm | Metric \(M\) | Inverse metric \(P\) |
| --- | --- | --- | --- |
| Kernel coefficients \(\alpha\) | \(\alpha^\top K \alpha\) | \(K\) | \(K^{-1}\) |
| Function values \(u = f(t)\) | \(u^\top K^{-1} u\) | \(K^{-1}\) | \(K\) |

The same choice governs [implicit differentiation](implicit_ad.md):
in underdetermined problems the metric is part of the definition of the
derivative, selecting the minimum-\(M\)-norm solution tangent.

## Shifted Metrics and the Seminorm Limit

Kernel models often carry a few extra scalar parameters \(\beta\) (level
constants, initial values) alongside the coefficients \(\alpha\), and the
natural objective is the RKHS *seminorm* \(\alpha^\top K \alpha\) with
\(\beta\) free — which is not a metric (\(M \succ 0\) fails on the
\(\beta\) block, and \(K\) itself is numerically singular on fine grids).
The **unified shifted metric**

$$
M_\varepsilon
= \begin{bmatrix} K & 0 \\ 0 & 0 \end{bmatrix} + \varepsilon I
= \begin{bmatrix} K + \varepsilon I_n & 0 \\ 0 & \varepsilon I_k \end{bmatrix}
$$

completes it with a single spectral floor: the eigenvalues are
\(\{\lambda_i(K) + \varepsilon\} \cup \{\varepsilon\}\), so
\(\|M_\varepsilon^{-1}\|_2 = 1/\varepsilon\) exactly — uniformly in \(n\)
and in how singular \(K\) is. The metric norm it minimizes is

$$
\|s\|_{M_\varepsilon}^2
= \alpha^\top K \alpha + \varepsilon \|s\|_2^2 ,
$$

the seminorm plus a flat Tikhonov ridge on the whole parameter vector.

**The \(\varepsilon \to 0\) limit.** When \(K \succ 0\), \(J\) has full row
rank, and the \(\beta\)-columns \(J_\beta\) have full column rank, the
seminorm-constrained problem \(\min_\theta \alpha^\top K \alpha\) s.t.
\(J\theta = b\) has a unique solution — the bordered KKT system

$$
\begin{bmatrix} 2K & 0 & J_\alpha^\top \\ 0 & 0 & J_\beta^\top \\
J_\alpha & J_\beta & 0 \end{bmatrix}
\begin{bmatrix} \alpha \\ \beta \\ -y \end{bmatrix}
= \begin{bmatrix} 0 \\ 0 \\ b \end{bmatrix}
$$

— and the minimum-\(M_\varepsilon\)-norm solution (and its implicit
derivative) converges to it at rate \(O(\varepsilon)\). For **singular**
\(K\) the limit is *lexicographic*: the minimum-Euclidean-norm element
among the seminorm minimizers (the Tikhonov tie-break), not a distinguished
"\(\beta\)-free" solution. (With \(K = 0\) and one constraint
\(\alpha + \beta = 1\), every feasible pair has zero seminorm;
\(M_\varepsilon\) selects \(\alpha = \beta = 1/2\).) State uniqueness
assumptions before claiming the \(O(\varepsilon)\) perturbation.

Compared to the two-knob block form \(\operatorname{blockdiag}(K +
\delta I, m_0 I_k)\), one \(\varepsilon\) is one dial: smaller
\(\varepsilon\) means less selection bias but a harder metric solve
(\(\kappa(K + \varepsilon I) = (\lambda_{\max} + \varepsilon)/\varepsilon\))
and a larger scalar-block spike \((c^2/\varepsilon)\,uu^\top\) in the dual
operator — see the [Tuning Guide](tuning_guide.md#the-metric) and
[Utilities](utilities.md#unified-shifted-block-metrics) for construction
and preconditioning.

## Worked Example

For the one-row residual \(r(\theta) = \theta_1 + \theta_2 - 1\) at
\(\theta = 0\): every interpolating step satisfies \(s_1 + s_2 = 1\). The
identity metric splits the correction evenly, \(s = (1/2, 1/2)\); the metric
\(M = \operatorname{diag}(1, 4)\) makes the second coordinate more expensive
and selects

$$
s = -P J^\top (J P J^\top)^{-1} r
= \frac{-r}{1 + 1/4}\begin{bmatrix}1\\[2pt]1/4\end{bmatrix}
= \begin{bmatrix}0.8\\[2pt]0.2\end{bmatrix},
\qquad r = -1.
$$

With tiny damping, one `update` reproduces both:

```python
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt, metric_from_cholesky


def residual(theta, _, __):
    return jnp.array([theta[0] + theta[1] - 1.0])


theta0 = jnp.zeros(2)

identity_solver = LevenbergMarquardt(residual, init_damping=1e-9)
x_identity, _, _ = identity_solver.update(theta0, identity_solver.init(theta0))
# x_identity ≈ [0.5, 0.5]

L = jnp.linalg.cholesky(jnp.diag(jnp.array([1.0, 4.0])))
metric_solver = LevenbergMarquardt(
    residual, init_damping=1e-9, metric=metric_from_cholesky(L)
)
x_metric, _, _ = metric_solver.update(theta0, metric_solver.init(theta0))
# x_metric ≈ [0.8, 0.2]
```

## Scope of the Claim

The minimum-norm statement is **local**: each small-damping step is the
minimum-\(M\)-norm correction to the *linearized* residual equations.
Nonlinear LM run to convergence does not globally solve
\(\min \|\theta\|_M\) subject to \(r(\theta) = 0\) — which root it reaches
depends on the initialization and the step history. The safe claims are that
near interpolation the steps are metric Gauss-Newton corrections, and that
the [implicit derivative](implicit_ad.md) at the returned
solution is exactly the minimum-\(M\)-norm tangent.
