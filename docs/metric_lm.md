# Metric LM

`LevenbergMarquardt` minimizes \(\|r(x)\|^2\) with the metric weighting the
**trust region** rather than the objective:

$$
\min_s \tfrac12\|r + Js\|^2 + \tfrac{\lambda}{2}\|s\|_W^2 ,
\qquad
s(\lambda) = -(J^\top J + \lambda W)^{-1}J^\top r .
$$

## The minimum-norm limit

At an interpolating root \(J\) is rank deficient, so \(J^\top J\) is singular
and the \(\lambda \to 0\) limit is a pseudoinverse rather than an inverse.
Writing \(W = F^\top F\) and \(B = JF^{-1}\), the step in the whitened
variable is \(u(\lambda) = -(B^\top B + \lambda I)^{-1}B^\top r\), whose limit
is \(-B^{+}r\) — the **minimum-Euclidean-norm** solution in \(u\), hence the
**minimum-\(W\)-norm** correction in \(s = F^{-1}u\):

$$
\lim_{\lambda \to 0} s(\lambda)
= -F^{-1}(JF^{-1})^{+}\,r
= \arg\min\{\|s\|_W : Js = -r\ \text{in the least-squares sense}\} .
$$

The limit holds at every shape and rank, so tall problems that are still
deficient along some directions (redundant rows, collinear columns) keep the
selection. Damping interpolates between two metric methods: large \(\lambda\)
gives a \(W^{-1}\)-scaled gradient step, small \(\lambda\) the metric
Gauss-Newton step above.

**This selects the correction, not the iterate.** The returned root is
\(x_0 + \sum_k s_k\), so it is the *path* that carries the geometry. When you
want the interpolant itself to be minimum-seminorm, the selection belongs in
the objective — use [`RidgeLevenbergMarquardt`](ridge_lm.md).

## Choosing the metric

Under a kernel model \(f(t) = \sum_j \alpha_j k(t, c_j)\), the RKHS norm is
\(\alpha^\top K \alpha\) with \(K\) the Gram matrix over centers, so
`W = K` and the factor is `jnp.linalg.cholesky(K, upper=True)`. A
positive-*semi*definite \(K\) needs a shift to be invertible:
`RepeatedFactorMetric(jnp.linalg.cholesky(K + 1e-8 * I, upper=True), repeats=r)`
factors \(K + \varepsilon I\) once and repeats it.

Parameters the metric should not weight (a scalar coefficient, a bias) go past
`metric.size` into the **free block**, whose damping weight is
`Metric.free_scale` (1.0 by default). The default `metric=None` is Euclidean
damping over the whole vector.

## Stopping

Disjunctive — any one rule fires `CONVERGED`:

- `atol`: \(\|r\| <\) atol, the equations are solved;
- `gtol`: `info.grad_norm` \(= \|F^{-\top}J^\top r\| <\) gtol, stationarity in
  the dual \(W^{-1}\)-norm;
- `xtol`: an accepted step's whitened norm \(\|u\| <\) xtol.

Setting one to `0` disables it. For an interpolation problem `atol` is usually
the one you want.

## Rank deficiency and implicit AD

The implicit rule differentiates the *undamped* Gauss-Newton system, which is
singular on exactly the side the problem is deficient in. Each config is
therefore offered only where its operator is invertible, and says so loudly
otherwise:

| `ad_solver` | needs | notes |
|---|---|---|
| `None` | — | matches the forward family, else `LU()` if \(m = n\) and `SVD()` if not |
| `LU()` | \(m = n\) | solves \(B\) itself at \(\text{cond}(B)\); rejects a rectangular system |
| `Cholesky()` | full rank in the small side | factors whichever of \(BB^\top\), \(B^\top B\) is smaller, at \(\text{cond}(B)^2\) |
| `SVD()` | nothing | spectral filter; the rule for padded zero residuals |
| `CG(precond)` | \(n \le m\) | `penalty=` regularizes it for \(n > m\), at an \(O(\text{penalty})\) bias |
| `GramCG(precond)` | \(m \le n\) | |

A **square** \(B\) is the case where the tangent is a plain nonsingular solve:
it is unique, so no norm is being minimized and the metric selects nothing.
That is what makes `LU()` valid there and why it is the default — and why the
square rule skips the whitening round-trip the rectangular rules require. It
also means one factorization serves both directions, forward mode solving with
\(B\) and reverse mode with \(B^\top\). `Cholesky()` computes the same map on
a square system but at \(\text{cond}(B)^2\), which costs roughly half the
significant digits of the tangent for nothing.

The **padded zero residual** pattern — appending identically-zero rows so
compiled shapes stay stable across problem instances — makes the undamped dual
singular by construction. `ad_solver=SVD()` computes the minimum-metric-norm
tangent there, which equals the unpadded one.

See [Implicit differentiation](implicit_ad.md) for the rule itself.
