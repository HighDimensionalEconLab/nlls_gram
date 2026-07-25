# Ridge Levenberg–Marquardt

`RidgeLevenbergMarquardt` solves underdetermined interpolation problems by
minimizing a **ridge objective** whose small-ridge limit is the
minimum-seminorm interpolant. Where
[`LevenbergMarquardt`](index.md) encodes the selection in a parameter-space
*metric* (the algorithm's damping geometry), this solver encodes it in the
*objective* — every inner problem is a well-posed nonlinear least squares
that any solver can attack with textbook guarantees.

## Problem statement

The flattened parameter vector splits into two blocks,
\(x = [x_m;\, x_f] \in \mathbb R^p\): the **metric block** \(x_m \in
\mathbb R^{n_m}\), covered by a user positive-definite metric \(W \succ 0\)
(a [`Metric`](#the-metric-interface)), and the **free block**
\(x_f \in \mathbb R^{n_f}\), unpenalized (\(n_f = p - n_m \ge 0\) is
inferred from the iterate at `init`). Given an underdetermined residual
\(r(x) \in \mathbb R^m\) (\(m < p\)), the target is the
**minimum-seminorm interpolant**

$$
x^\dagger \;=\; \operatorname*{argmin}_x \; \|x_m\|_W^2
\quad\text{s.t.}\quad r(x) = 0 ,
$$

with \(\|v\|_W = \sqrt{v^\top W v}\), identified under the condition that no
null direction of \(J\) lives entirely in the free block
(\(\ker J \cap (\{0\} \times \mathbb R^{n_f}) = \{0\}\) at the solution;
Eldén 1982 — the full-space penalty \(\mathrm{blockdiag}(W, 0)\) is PSD by
design). For kernel coefficient problems \(\|x_m\|_W^2\) is the squared
RKHS norm, so \(x^\dagger\) is the minimum-RKHS-norm interpolant.

## The ridge family and why the objective carries the selection

The solver minimizes, for a strictly positive weight \(\lambda\) (the
`ridge`),

$$
F_\lambda(x) \;=\; \|r(x)\|^2 + \lambda\,\|x_m\|_W^2 .
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

## The whitened change of variables

The metric enters through an invertible **factor** \(F\) with
\(W = F^\top F\); \(F\) should be upper triangular, and the canonical
example is the upper Cholesky factor,
`F = jnp.linalg.cholesky(K, upper=True)`. The solver extends it by the
identity on the free block and runs **entirely in the whitened variable**

$$
y = \bar F x, \qquad
\bar F = \operatorname{blockdiag}(F,\, I_{n_f}), \qquad
\tilde J = J \bar F^{-1}, \qquad
E = \operatorname{blockdiag}(I_{n_m},\, 0) ,
$$

where the penalty is simply \(\lambda \|y_m\|^2\). This is an exact,
constant linear change of variables — the chain rule is applied by hand
once, never by differentiating through a solve — and neither \(W\) nor
\(\bar F\) is ever materialized: the algorithm only applies the metric's
factor ops (\(F v\), \(F^{-1} v\), \(F^{-\top} v\)).

Everywhere in the implementation the equivalent **augmented-residual** view
is used: standard *Euclidean* LM on

$$
R(y) = \begin{bmatrix} r(x) \\ \sqrt{\lambda}\, y_m \end{bmatrix},
\qquad
A = \begin{bmatrix} \tilde J \\ \sqrt{\lambda}\, [\,I_{n_m} \mid 0\,] \end{bmatrix}
$$

— the rows of \(F\) are the appended penalty equations, and in \(y\) they
are the *constant* stack \([\,I \mid 0\,]\). Each LM step solves

$$
\left(\tilde J^\top \tilde J + \lambda E + \mu I\right) \delta_y
= -\left(\tilde J^\top r + \lambda\,[\,y_m;\, 0\,]\right),
\qquad
\delta_x = \bar F^{-1} \delta_y ,
$$

(equivalently, the least-squares problem with \(\sqrt{\mu}\,I\) damping rows
appended to \(A\)); the iterate is stored in \(x\), and trial penalties use
the linearity \(\bar F(x + \delta_x) = y + \delta_y\), so no second factor
application is ever needed.

Three properties follow from the penalty rows being **constant** in \(y\):

- the gradient identity
  \(A^\top R = \tilde J^\top r + \lambda [y_m; 0]\) is exact;
- the penalty rows have zero second directional derivative, so **geodesic
  acceleration** (Transtrum–Sethna 2012) uses the true residual only —
  \(R_{vv} = [\,f_{vv};\,0\,]\) — and the standard formulas apply unchanged;
- accept/reject compares the plain scalar objective \(F_\lambda\)
  (Marquardt 1963; Moré 1978; Nocedal–Wright 2006, Ch. 10), with the trust
  region posed in the whitened geometry:
  \(\mu\|\delta_y\|^2 = \mu\,(\|\delta_{x_m}\|_W^2 + \|\delta_{x_f}\|^2)\).

The conditioning payoff is structural: the whitened normal matrix
\(\tilde J^\top \tilde J + \lambda E\) has a clean **spectral floor at
\(\lambda\)** on the metric block regardless of \(W\)'s conditioning — no
interaction between the kernel's conditioning and the Gram product — which
is what keeps the default `Cholesky()` path accurate at deep ridge.
(Measured on the production DAE kernel drivers at
\(\lambda = 3{\times}10^{-12}\): 25–35% fewer LM steps than the unwhitened
x-space formulation at ~8× lower per-step cost than its QR fallback —
1.8–3× faster wall-clock with identical solution quality; that experiment
is why the whitened path is the only one.)

The two scalars are fully decoupled:

- \(\mu\) (`damping`) is the trust-region parameter — it moves every step by
  accept/reject exactly as in stock LM;
- \(\lambda\) (`ridge`) is the *selection weight* — carried as traced state,
  changed only by `init()` or a `solve` callback, annealed monotonically
  toward a positive floor. \(\lambda\) never enters the metric's
  factorization, so continuation composes unchanged.

\(\lambda = 0\) is out of contract everywhere (constructor validation,
callback contract, continuation floors), so
\(\tilde J^\top \tilde J + \lambda E\) is positive definite under the
identification condition at every reachable state.

### Ridge continuation

The homotopy \(\lambda \downarrow \lambda_{\min}\) is a **documented
callback recipe**, not solver machinery:

```python
from nlls_gram import ridge_continuation

cb, us0 = ridge_continuation(ridge_floor=1e-8, decrease=0.1)
result = solver.solve(x0, callback=cb, user_state=us0,
                      atol=1e-8, max_steps=500)
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
intermediate level's residual (they differ by roughly `1 / decrease`) — the
intermediate levels are stationary too, and `atol` is what rules them out,
so the solve can only stop at the floor. Solving each level out and passing
to the limit is the nonlinear Tikhonov path above; annealing per
stationarity event is the **iteratively regularized Gauss–Newton method**
(Bakushinskii 1992; Blaschke–Neubauer–Scherzer 1997;
Kaltenbacher–Neubauer–Scherzer 2008), whose theory wants exactly this
monotone, boundedly geometric schedule. A callback ridge change is treated
as a *problem change*: that step's convergence test is suppressed (its
diagnostics were computed at the old \(\lambda\)) and the ridge-keyed
factorization caches invalidate.

### Stopping: the two-phase picture

A ridge solve has **two phases**. Phase 1 drives the residual to its floor
— fast, a handful of Gauss–Newton-quality steps. Phase 2 slides the iterate
*along the interpolation set*, resolving the null-space (selection)
component while \(\|r\|\) stays essentially constant. A pure-residual test
is blind to phase 2. The toy problem \(r(x) = x_1 - 1\) with the identity
metric from \(x_0 = (0, 3)\) makes it concrete: the residual floors at
\((1, 3)\), and everything that happens afterwards — the slide to the ridge
minimizer \((1/(1{+}\lambda),\, 0)\) — is invisible to \(\|r\|\). Stopping
on the residual alone returns an answer whose second coordinate is wrong by
3, at machine-perfect residual.

The stopping geometry is the whitened one: `info.step_norm` (bounded by
`xtol`) is \(\|\delta_y\|\) — the **W-norm of the step** on the metric
block — and `info.grad_norm` (bounded by `gtol`) is the whitened
stationarity

$$
\left\| \tilde J^\top r + \lambda\,[\,y_m;\, 0\,] \right\|
$$

— the **dual \(W^{-1}\)-norm of the half-gradient** on the metric block.
Objective values (`loss`, `resid_loss`, `penalty_value`) are invariant
under the change of variables.

Phase 2 is what `gtol` exists to detect, and it should be **calibrated,
not guessed**. At a ridge minimizer the gradient vanishes as an exact
cancellation of its two terms,

$$
\tilde J^\top r \;=\; -\lambda\,[\,y_m;\, 0\,] ,
$$

each of magnitude \(\approx \lambda\,\|y_m\| = \lambda \sqrt{q(x)}\) with
\(q(x) = \|x_m\|_W^2\) the penalty value. The calibrated bound asks that
the computed sum be below \(c\) times that common magnitude — i.e. that the
two terms **cancel to relative accuracy** \(c\):

$$
\|\tilde J^\top r + \lambda [y_m; 0]\| \;<\; c \cdot \lambda\,\sqrt{q},
\qquad c \sim 10^{-3} .
$$

It is the standard relative-residual criterion of numerical linear algebra
(\(\|Ax - b\| \le \mathrm{tol}\,\|b\|\)) applied to the stationarity
equation, and the link to the *selection error* is first-order: along
null-space directions the whitened Hessian is \(\lambda E\), so a gradient
of norm \(g\) leaves a relative null-space displacement of exactly
\(g / (\lambda\sqrt q)\). A `gtol` at \(c\,\lambda\sqrt q\) resolves the
selection to \(\sim c\) relative accuracy; a `gtol` orders of magnitude
above it leaves it visibly unresolved (measured on the asset pricing
driver: a loose `gtol` left the free scalar \(p_0\) off by 310% at a
machine-perfect residual; the calibrated one, 0.1%). The solver reports
\(\sqrt q\) directly — `info.penalty_grad_norm` equals
`sqrt(info.penalty_value)` — and the solution's seminorm is usually known
to an order of magnitude before any pilot run, so the recipe is simply

```python
gtol = 1e-3 * ridge * jnp.sqrt(q)   # q = the solution's squared (RKHS) seminorm
result = solver.solve(x0, max_steps=400, gtol=gtol, atol=2e-8)
```

The same formula is a *scaled dual-feasibility* criterion in the
constrained-optimization sense — the ridge problem is the quadratic-penalty
form of \(\min \|x_m\|_W^2\) s.t. \(r = 0\) with multiplier estimate
\(\nu = r/\lambda\), and codes like IPOPT scale their KKT-residual tests by
multiplier magnitudes in exactly this way. Note the consequences: `gtol`
must be **re-calibrated** when the residual scaling, the metric scaling, or
the ridge level changes (all three move \(\lambda\sqrt q\)), and during
continuation a single absolute `gtol` can only be right at one level — pair
it with `atol` placement (above) so the solve stops only at the floor.

**Floors and precision.** The achievable `grad_norm` is bounded by
\(\varepsilon_{\text{mach}}\) and the conditioning of the augmented stack
(which grows like \(1/\sqrt{\lambda}\)); `gtol` must sit above that noise
floor or the solve dies loudly at `MAX_STEPS`. Two regimes follow. On
well-conditioned problems at \(\lambda \gtrsim \sqrt{\varepsilon}\), the
calibrated \(10^{-3}\lambda\sqrt q\) sits comfortably above the floor and
the recipe applies as stated (in float32 the same structure holds with
\(\varepsilon \sim 10^{-7}\); use the `ridge=None` default
\(\lambda = \sqrt{\varepsilon_{f32}}\) and a looser \(c\)). Deep
continuation floors on ill-scaled Jacobians (\(\lambda\) far below
\(\sqrt{\varepsilon}\)) can push the noise floor *above* the calibrated
value — there `gtol` becomes a measured constant instead: run once, watch
where `info.grad_norm` flattens, and set `gtol` just above it. And one
floor-choice rule: **pushing the ridge floor lower makes the answer worse
past a point** — the total error
\(O(\lambda) + O(\varepsilon_{\text{mach}}/\lambda)\) is minimized near
\(\lambda^* \sim \sqrt{\varepsilon_{\text{mach}}}\), about `1e-8` in
float64; only go below that with a measured `gtol`.

## The Metric interface

A `Metric` supplies the factor through four ops, each taking a metric-block
vector — or a matrix whose *leading* axis is `size` (columns batched) — and
a `MetricContext` carrying everything the solver knows at the call site
(the flat iterate `x`, the live `RidgeLMState`, `args`, `p`; the shipped
metrics ignore it, a custom metric may key off it):

- `factor_apply(v, ctx)` — \(F v\)
- `factor_solve(v, ctx)` — \(F^{-1} v\)
- `factor_solve_transpose(v, ctx)` — \(F^{-\top} v\) (the workhorse:
  \(\tilde J^\top\) assembly, gradient pullbacks, the AD rule)
- `norm(v, ctx)` — \(\|v\|_W\) (vectors only; must match
  \(\|F v\|_2\) to floating-point accuracy — the solver compares objective
  values built from both forms)

Two contracts matter. The factor must be **exact**: the identity penalty
block in \(y\) is hardcoded, so an approximate factor silently changes the
objective (unlike a `CG` preconditioner, which may be sloppy). And metrics
hash **by identity**: construct one at setup scope and reuse it — a rebuilt
equal-config metric keys a fresh compiled solve loop.

Shipped metrics:

- `IdentityMetric(size)` — \(F = I\): plain ridge
  \(\|r\|^2 + \lambda\|x_m\|^2\), every op a passthrough, with no
  special-casing anywhere in the solver.
- `RepeatedFactorMetric(F, repeats=J)` — `repeats` copies of one
  upper-triangular block factor on the metric block,
  \(W = \operatorname{blockdiag}(F^\top F, \ldots, F^\top F)\). The
  constructor takes the **factor**, not the Gram matrix (callers typically
  already hold it): `F = jnp.linalg.cholesky(K, upper=True)`. All repeated
  blocks and all batched columns share a single triangular product/solve.

An `x`- or `p`-dependent metric is deliberately unsupported in this
release: the gradient and the implicit-AD rule treat \(F\) as constant, and
a dependent factor would silently drop derivative terms (the reserved
`metric_factory` keyword raises; its documented contract is a
`prepare`/`build` pair producing a `Metric` from traced `metric_state`,
with `metric_valid` reject-step reuse and a state change treated as a
problem change).

### Kernel instantiation

For kernel collocation with \(J\) repeated coefficient blocks (one per
equation stack) and \(s\) unpenalized structural scalars in the free block,

$$
W = \operatorname{blockdiag}(K, \ldots, K),
\qquad
q(x) = \sum_j \alpha_j^\top K \alpha_j ,
$$

built by `RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True),
repeats=J)` — one Cholesky, batched triangular ops over the repeated
blocks, **no epsilon shift on the scalars** (they carry no penalty at all,
unlike the metric formulation's \(\varepsilon I\)). Each inner LM step is
then a kernel ridge regression of the relinearized equations, exactly the
Gauss–Newton scheme of Chen–Hosseini–Owhadi–Stuart (2021) with the ridge
kept explicit. An \(O(N)\) state-space (Matérn) factor pairing with the
matrix-free `CG` path is planned for a later release; at the problem sizes
this package targets the dense factor serves all three linear solvers.

**Choose the free block deliberately.** The identification condition —
whatever the residual does not pin, the metric must — is easy to violate by
moving too much into the free block, and it can hold *marginally* while the
selection still needs a block's seminorm in the objective. The experiments
that motivated this design produced a clear negative result: on the DAE
drivers, removing the penalty from coefficient blocks whose *levels* are
pinned by initial conditions left the residual converged but let the
unpenalized paths drift enormously between collocation points (seminorms
exploding by 2–80×, ground-truth checks failed). Keep every kernel
coefficient block under the metric; reserve the free block for genuinely
structural scalars, and verify the choice empirically — the free
coordinates must be reproducible across perturbed starts (multi-start
agreement is the practical test).

## Linear solvers

`linear_solver` takes a **typed config** — `Cholesky()`, `QR()`,
or `CG(preconditioner, tol=..., atol=..., maxiter=...)` — so each
method's knobs live on its own config and cannot be passed with another
(the configs hash by value: equal configs share one compiled solve loop).
All three solve
\((\tilde J^\top \tilde J + \lambda E + \mu I)\,\delta_y = -g\) and share
one factorization (or inner-solve setup) between the velocity and
geodesic-acceleration solves.

| `linear_solver` | Method | Cost per update | When |
| --- | --- | --- | --- |
| `Cholesky()` (default) | dense normal equations; \(\tilde J^\top = \bar F^{-\top} J^\top\) by one batched factor solve, then \(G = \tilde J^\top \tilde J + \lambda E\) (\(\lambda\) added on the metric-block diagonal only) **cached across rejected steps** (a reject re-factors in \(p^3/3\) without the GEMM or the \(\tilde J\) materialization) | \(mp^2\) GEMM + \(n_m\)-block triangular solve (both skipped on reject) + \(p^3/3\) | default; the \(\lambda\) spectral floor keeps it accurate at deep ridge |
| `QR()` | QR of the augmented stack \([\tilde J;\ \sqrt{\lambda}\,[I \mid 0]]\) cached per \((x, \lambda)\); each damping update re-factors \([R;\sqrt{\mu}I]\) and solves by corrected semi-normal equations with one refinement pass (Björck 1987; Björck 1996, §6.6.5; the damping-row structure is Moré 1978's) | \((m{+}n_m)p^2\) QR (skipped on reject) + \(2p^3/3\)-ish refactor | extreme tiny-\(\lambda\)/tiny-\(\mu\) regimes, where even the whitened normal equations square a large \(\|\tilde J\|^2/\mu\) |
| `CG(preconditioner, ...)` | matrix-free preconditioned CG on the damped whitened normal operator itself — the same SPD system `Cholesky()` factors, with products through jvp/vjp and the metric's factor ops; the required `preconditioner` (a typed [`Preconditioner`](#preconditioners); `IdentityPreconditioner()` opts out) sits in CG's `M` slot | iterations × (one \(\tilde J\) and one \(\tilde J^\top\) product + one preconditioner apply) | Jacobians too large to materialize, given a structured preconditioner; the \(\lambda\) floor on the metric block helps the spectrum |

Everything runs at the **residual dtype** — there is no promotion knob. The
selection resolution is bounded by the problem dtype either way (the
stationarity test reads the gradient, which lives at the residual dtype;
at tiny \(\lambda\) the float32 gradient noise \(\sim 10^{-7}/\lambda\)
bounds the selection regardless of any wider factorization), and `QR()` is
the in-dtype fix for extreme conditioning. A float32 program that
genuinely needs float64 selection should run the solve in float64.

### Preconditioners

The `CG` config requires a typed `Preconditioner` in both roles: a subclass
implementing `apply(v, damping, ctx)`, an SPD approximation of
\((\tilde J^\top \tilde J + \lambda E + \mu I)^{-1}\) applied with the live
damping (zero in the AD role), receiving the same `MetricContext` as the
metric ops. `IdentityPreconditioner()` is the explicit opt-out; a custom
one is a small dataclass:

```python
@dataclass(frozen=True, eq=False)
class JacobiPreconditioner(Preconditioner):
    diagonal: jax.Array

    def apply(self, v, damping, ctx):
        return v / (self.diagonal + damping)
```

A preconditioner changes the CG iteration path, never the subproblem —
approximations are safe (unlike the metric factor, which must be exact).

For repeated interacting blocks (multiple "agents" coupled through shared
equations), `BlockEigenPreconditioner` is the shipped workhorse: a
block-diagonal approximation over a chosen grouping of the whitened
coordinates, eigendecomposed once per build and applied with the
damping-analytic shift \(\Lambda + \texttt{ridge\_weight}\cdot\lambda +
\mu\) (metric-block families carry the live ridge, free-block families are
damping-only). The instance is stateless and value-hashable; its numeric
state — built by `block_eigen_state` from stacked SPD diagonal blocks and a
family-major permutation — rides in the residual `args` under a fixed key
and is read through `ctx.args` at apply time.

**Adaptive rebuilds through the solve callback.** Because the state lives
in `args`, a callback can rebuild it from the live iterate with no solver
support: staleness detection is a traced value comparison, so returning
identical values suppresses nothing, and a real swap suppresses only that
step's convergence check. The natural policy pairs rebuilds with
`ridge_continuation` — rebuild exactly when the anneal advances a level,
which already suppresses that step:

```python
continuation, user_state0 = ridge_continuation(ridge_floor=1e-11)

def callback(ctx):
    action = continuation(ctx)
    advanced = action.lm_state.ridge < ctx.lm_state.ridge
    fresh = build_state(ctx.x)          # model-side analytic assembly
    state = jax.tree_util.tree_map(
        lambda old, new: jnp.where(advanced, new, old),
        ctx.args["preconditioner"], fresh,
    )
    return LMSolveAction(
        lm_state=action.lm_state, user_state=action.user_state,
        args={**ctx.args, "preconditioner": state},
    )

solver = RidgeLevenbergMarquardt(
    residual_fn, metric=metric,
    linear_solver=CG(BlockEigenPreconditioner(), tol=1e-10, maxiter=2500),
)
result = solver.solve(x0, {"preconditioner": build_state(x0)},
                      callback=callback, user_state=user_state0,
                      max_steps=120, gtol=1e-11)
```

The AD-role CG reads the state from `result.args` at zero damping — a
callback-rebuilt state is exactly the near-solution build the tangent solve
wants; pass `ad_solver=CG(...)` EXPLICITLY when the tangent matters
(`ad_solver=None` matches the CG family but runs the tangent solve
unpreconditioned). One measured calibration note: with a family layout
whose blocks carry the same ridge floor as the operator, the CG iteration
count SATURATES as the ridge anneals down (it does not grow like
\(1/\sqrt{\lambda}\)) — budget `maxiter` for the saturated count, since a
truncated inner solve stalls the endgame. Two scope cautions for
args-carried state: `save_steps=True` records a full copy of `args` per
step (the whole eigenbasis, every step), and a parallel `multi_start`
vmaps the callback so a `where`-gated rebuild evaluates BOTH branches in
every lane — keep both out of preconditioned-CG production runs.

## Implicit differentiation

`solve(...).x` has a custom implicit rule with respect to `p`: Gauss–Newton
differentiation of the ridge stationarity, posed on the whitened variable —

$$
\left(\tilde J^\top \tilde J + \lambda E\right) \dot y
= -\tilde J^\top \frac{\partial r}{\partial p}\,\dot p ,
\qquad
\dot x = \bar F^{-1} \dot y ,
$$

with \(\lambda\) frozen (stop-gradient) at the returned state's ridge — the
continuation schedule's and the multi-start selection's dependence on `p`
are deliberately ignored — and **no damping** in the AD matrix.
`ad_solver=Cholesky()` assembles and factors;
`CG(preconditioner, tol=..., atol=..., maxiter=...)` is matrix-free CG on
the same operator (positive definite does not mean well conditioned —
unpreconditioned CG degrades as \(\lambda\) shrinks). `ad_solver=None`
(the default) matches the forward path: `Cholesky()` for the dense
forwards, `CG` for a `CG` forward.

The contract, stated plainly: exact differentiation carries two extra terms
(\(\sum_i r_i \nabla^2 r_i\) in the matrix, \((\partial J^\top/\partial p)r\)
on the right), both dropped. Both are exactly zero for an affine residual
and first order in \(\|r\|\) otherwise — so the rule is exact in the
interpolating limit **in absolute terms**. On a genuinely underdetermined
*curved* system, however, the null-space block of the exact tangent carries
the constraint-curvature term \(\sum_i \nu_i \nabla^2 r_i\) with multiplier
\(\nu = r/\lambda\), which does not vanish relative to the
\(\lambda\)-scaled penalty as \(\lambda \downarrow 0\): tangent components
read off null-space directions retain a Gauss–Newton bias proportional to
the residual curvature. This is the same approximation the metric solver's
frozen-projector AD rules make. Failed statuses return exact zero tangents
and evaluate the masked tangent program at stop-gradient copies of the
original inputs and the *initial* ridge.

## Migration from the metric formulation

```python
# before (metric LM): selection via the algorithm's damping geometry,
# epsilon-shifted so the metric stays invertible
metric = repeated_shifted_dense_metric(K, repeats=3, zero_pad_size=d, epsilon=1e-7)
solver = LevenbergMarquardt(residual_fn, metric=metric,
                            linear_solve_dtype=jnp.float64,
                            metric_solve_dtype=jnp.float64)
result = solver.solve(theta_0, max_steps=400, atol=2e-8)

# after (ridge LM): selection in the objective, no epsilon anywhere;
# the metric takes the factor directly (gtol ~ 1e-3 * ridge * sqrt(q))
metric = RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=3)
solver = RidgeLevenbergMarquardt(residual_fn, metric=metric,
                                 ridge=1e-8)  # or None for the dtype default
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
  conjunctive contract makes this loud, not silent; the whitened geometry
  makes the `gtol` choice a one-line calibration
  (`1e-3 * ridge * sqrt(q)`) instead of a guess.
- **No dtype-promotion knobs**: the solve runs at the residual dtype
  (`QR()` is the extreme-conditioning fix); the metric solver's
  `linear_solve_dtype`/`metric_solve_dtype` have no ridge analog.
- Forward `CG` users pass the preconditioner as a required config
  field — `CG(IdentityPreconditioner())` at minimum — deliberately
  stricter than the metric solver's optional preconditioner hooks.
- Multi-start ranking uses the ridge objective at each lane's own final
  ridge — comparable across lanes when they share a continuation schedule.

## API reference

::: nlls_gram.Metric

::: nlls_gram.MetricContext

::: nlls_gram.IdentityMetric

::: nlls_gram.RepeatedFactorMetric

::: nlls_gram.Preconditioner

::: nlls_gram.IdentityPreconditioner

::: nlls_gram.BlockEigenPreconditioner

::: nlls_gram.block_eigen_state

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
