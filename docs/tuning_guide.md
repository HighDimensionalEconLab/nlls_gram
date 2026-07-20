# Tuning Guide

Decision-oriented heuristics for choosing solvers and hyperparameters —
written for humans and AI assistants alike. Contracts and formulas live in
the [main docs](index.md); the math is in
[Metric Gauss-Newton](gauss_newton.md). Throughout, `m` is the residual count
and `n` the parameter count; the package targets `m << n`.

## Starting Point

```python
solver = LevenbergMarquardt(residual_fn)
result = solver.solve(x0, args, max_steps=500, atol=..., gtol=...)
```

- **`linear_solver="auto"` (the default) picks the smaller dense
  factorization from the problem shape** — the `m × m` residual-space Gram
  system when `n > m`, the `n × n` whitened normal system otherwise. The two
  produce the same step, so the default is a cost decision, never a
  semantics decision.
- **Geodesic acceleration is on by default** — it costs one extra
  directional derivative per step (plus one residual evaluation when the
  acceptance gate passes), the accept/reject test makes it safe, and on
  curved residuals it substantially cuts step counts. Near-linear problems
  gain little — `geodesic_acceleration=False` if the extra evaluation
  matters. With a custom metric it requires `metric.norm`.
- **The Jacobian cache is on by default** (rejected steps ~2x cheaper) at
  the cost of an `(n_params, n_residuals)` state buffer. Pass
  `cache_jacobian=False` for manual `update()` loops that swap `args`/`p`
  between steps (stale-cache hazard) or when the buffer strains GPU memory.
- Set `atol`/`gtol` rather than relying on `max_steps`: a converged solve
  that runs to `max_steps` wastes exactly the steps you didn't bound.

## Solver Selection

Shape first. The Gram and normal forms are the same method assembled in
different spaces — the `m × m` residual-space dual `J P J' + damping I`
versus the `n × n` whitened normal system `B'B + damping I` (`B = J S`) —
and they produce identical steps at any positive damping. So the first
question is only which of `m` and `n` is smaller, and `auto` (the default)
answers it: `gram_cholesky` when `n > m` strictly, else `normal_cholesky`,
resolved at trace time from the concrete shapes. The rule reads shapes, not
numerical rank — and it does not need to: rank-deficient problems
(redundant rows, collinear columns — tall interpolation problems always
have some) are handled by every form except `qr`, with the
minimum-`M`-norm small-damping selection intact.

| situation | use |
| --- | --- |
| dense factorization affordable on the smaller of `m`, `n` | `auto` (default) — `gram_cholesky` if `n > m`, else `normal_cholesky` |
| Jacobian too big to materialize, `n > m` | `gram_cg` |
| Jacobian too big to materialize, `m ≥ n` | `normal_cg` |
| matrix-free and ill-conditioned at small damping | `lsmr` |
| ill-conditioned metric, moderate `m`, full-row-rank `J` | `qr` |
| small system, rank-deficient `J`, direct factorization wanted | `augmented_qr` |

- **Cross-shape use is legitimate — the hooks decide.** The forms compute
  the same step, so picking against the shape rule is purely a cost/hook
  trade. `dual_preconditioner`, `preconditioner_factory`, and `recycle` are
  `gram_cg`-only (they live in residual space); a strong dual
  preconditioner or a carried deflation basis can justify `gram_cg` on a
  squarish problem. `normal_cg` is the way to precondition in *parameter*
  space, and the whitened forms swap `metric.solve` for
  `metric.inv_sqrt`/`inv_sqrt_transpose` — pick the form whose callbacks
  your metric can supply exactly.
- **`iterative_maxiter` budgets a different Krylov per form**: an
  `m`-dimensional space under `gram_cg`, an `n`-dimensional one under
  `normal_cg`. Re-tune it when switching forms; a budget tuned for a small
  dual can be far too small for a large parameter space.
- **`normal_cg`'s preconditioner has a structural requirement**: it must map
  `range(B')` into itself, or the minimum-norm selection is silently lost on
  rank-deficient problems — an arbitrary SPD approximation of the inverse
  does not qualify. `identity_preconditioner()` always does. See the
  [normal-space preconditioner](utilities.md#the-normal-space-preconditioner-normal_cg).
- **Avoid `qr` when massively overparameterized.** It does not use a
  Gram/normal form: it factors the whitened `n × m` matrix, so cost scales
  with `n` (measured 8-16x slower than `gram_cholesky` at `n=8192, m=1024`),
  and it requires full row rank — rank-deficient Jacobians produce
  non-finite steps. Its advantage is conditioning (it avoids squaring the
  condition number); reach for it only when that is the binding constraint.
- **Use `augmented_qr` for small systems when rank robustness matters.** It
  directly factors `[J S; sqrt(damping) I]`, whose damping block guarantees
  full column rank, but its width is the parameter count. That is attractive
  for DAE algebraic roots and expensive when `n` is large.
- **Use `lsmr` as the matrix-free sibling of `augmented_qr`.** It solves the
  same whitened damped subproblem `min_u ||r + B u||² + damping ||u||²`
  (`B = J S`, `S = metric.inv_sqrt`, step `s = S u`) by
  [LSMR](https://web.stanford.edu/group/SOL/software/lsmr/) bidiagonalization
  using only `J`/`Jᵀ` matvecs — no materialized Jacobian, no QR. Its payoff
  over the CG forms is conditioning: both Gram and normal operators carry
  the **square** of the whitened operator's condition number, so at small
  damping (`~1e10` in the motivating case) `eps·cond` puts an accuracy floor
  on the step that even dense direct solves hit, and CG truncation
  concentrates in the slow, selection-critical eigendirections. `lsmr` works
  at `cond(B) ~ sqrt` of that, restoring certifiable endgame accuracy. Reach
  for it when a matrix-free solve is required *and* the system is
  ill-conditioned near the solution; when it is well-conditioned (or a good
  preconditioner exists), the CG forms are cheaper per step. `lsmr` requires
  the metric's `inv_sqrt`/`inv_sqrt_transpose` (the identity metric supplies
  them). When the whitened operator itself is badly conditioned (`cond(B)`
  still large — e.g. `~1e8`, where plain LSMR needs thousands of endgame
  iterations), pass a `whitened_preconditioner` (a `WhitenedPreconditioner`):
  a parameter-space right-preconditioner `R⁻¹` running LSMR on `B R⁻¹` to
  cluster the spectrum and cut the endgame count to the tens (a
  Schur-complement factor is canonical). The damping row rides inside the
  preconditioned operator, so every subproblem is exactly the `I`-damped one
  in `u = R⁻¹ z`: `R` changes the iteration count, never the step, and the
  `damping → 0` selection limit is minimum-`M`-norm for any `R`. Stopping
  maps the same `iterative_tol`/`iterative_atol`/`iterative_maxiter` hooks
  (relative/absolute bound on the normal-equations residual, measured on the
  preconditioned operator, callback-schedulable). Differentiating a forward
  `lsmr` `solve(...).x` uses a dense implicit rule by default
  (`implicit_solver="auto"` applies the shape rule); pass
  `implicit_solver="gram_cg"` or `"normal_cg"` with an
  `implicit_preconditioner` for a fully matrix-free derivative.
- The CG forms return an *approximate* step under their iteration budget.
  That is usually fine — LM's accept/reject absorbs inexactness — but see
  the scheduling pattern below. With the default `implicit_solver="auto"`,
  differentiating a forward `gram_cg`/`normal_cg` `solve(...).x` uses the
  matching matrix-free CG rule instead of materializing \(J^\top\).

## Float64 à la Carte

The Gram and normal forms square the condition number (they factor
`J P J'` or `B'B`). If that system is ill-conditioned or implicit
derivatives must be accurate, reach for float64 — it fixes more numerical
trouble than any damping adjustment. Three grades, from narrowest to
widest:

- **`metric_solve_dtype=jnp.float64`** promotes only the resolved metric
  callbacks (fixed or factory-built), via the
  [`metric_with_compute_dtype`](utilities.md#compute-dtype-wrapper)
  mechanics. Kernel Gram factorizations are routinely the worst-conditioned
  piece of the whole pipeline, so this knob often earns float64 even in
  otherwise-float32 programs — it is the right first move when the metric,
  not the solver algebra, is the fragile part.
- **`linear_solve_dtype=jnp.float64`** promotes the dense linear-solve
  pipelines: the `gram_cholesky`/`normal_cholesky` forward factorizations
  and the dense implicit rules, while the model stays float32 — measured
  ~1.4x per `gram_cholesky` update at `m=100, n=2000` for a *trivial*
  residual (an upper bound: real residual and Jacobian costs dominate and
  stay float32), recovering the float64 answer to ~1e-6 on a 1e-7-spike
  metric where plain float32 is ~5% wrong.
- **`jax_enable_x64`** globally remains the full fix when the model itself
  needs it — and is still required for either knob: it is what makes
  float64 arrays available (explicitly float32 data stays float32).

Choosing between the grades: the knobs' win is proportional to how much of
the step is model evaluation (residual + the `m` VJP Jacobian passes, which
stay float32) versus promoted algebra; when the promoted algebra IS the
step — trivial residuals, dense-metric-dominated updates — the knobs cost
about the same wall time as full x64 and their remaining win is halved
model memory and the unchanged float32 contract. One cost surprise to know
about: a float64 metric with an iterative solve
(`metric_from_shifted_matvec`) runs its inner CG in float64 at the tighter
float64 default tolerance. Neither knob touches the CG solver paths — there
the remedies remain preconditioning and, when the attainable-residual floor
binds, full x64.

## Damping

**Convergence is usually insensitive to the damping parameters — do not tune
them first.** The accept/reject loop self-corrects `init_damping` within a
few steps. Try them when you see specific signatures:

- Many early rejections → raise `init_damping` (start nearer gradient
  descent).
- Long rejection storms in float32 → set `max_damping` (~`1e6`) so damping
  cannot overflow.
- Accept/reject oscillation → bring `damping_decrease`/`damping_increase`
  closer to 1 (e.g. 0.7 / 2.0) for smoother adaptation.
- All steps accepted but progress is slow → lower `init_damping` or decrease
  faster (`damping_decrease=0.3`).

## Schedule Accuracy, Cheap → Exact

Inexact steps are cheap experiments early; near the solution, step quality
limits the convergence rate (and small damping makes the inner system harder
exactly then). Three patterns, in order of preference:

1. **Relative `iterative_tol`** (e.g. `1e-2`) with a generous
   `iterative_maxiter` cap: inner accuracy tightens automatically as the
   residual shrinks. No scheduling code.
2. **Grow the CG budget in a callback** when the loss crosses a threshold —
   single solve call, so implicit AD applies; see the
   [cookbook recipe](callbacks.md#scheduled-inner-solve-accuracy). All of
   `LMHyperparams` is resettable this way.
3. **Stage two solvers**: a coarse CG-form solve, then a dense endgame
   (`auto`) warm-started with `result.x` and `result.lm_state`. The implicit
   derivative is unaffected (it is defined at the returned solution only).

Forward iterative tolerances and implicit AD tolerances are separate. The
implicit CG rules use `implicit_tol=None` by default, which means `1e-6` in
float32 and `1e-10` in float64; these defaults target derivative accuracy, not
cheap forward steps. Pass a dense `implicit_solver`
(`"gram_cholesky"`/`"normal_cholesky"`) when you want a dense implicit rule
under an iterative forward solver, or tune `implicit_tol`, `implicit_atol`,
`implicit_maxiter`, and `implicit_preconditioner(v)` for a matrix-free
derivative.

Before scheduling accuracy, check whether a structural `dual_preconditioner`
removes the problem: when the dual operator's conditioning grows with problem
size (metric solves inject \(M^{-1}\) into it), a spectrally equivalent
preconditioner can pin the required budget at a small constant (in a
kernel-collocation case study, a flat `iterative_maxiter` of 2–20 across two
orders of magnitude in problem size) where the unpreconditioned budget grows
with refinement. See [Utilities](utilities.md#shermanmorrison-dual-preconditioner).
When no structural preconditioner is available — identity-metric
neural-network duals (empirical NTK Grams) are the canonical case — reach
for the randomized
[`nystrom_preconditioner`](utilities.md#nystrom-preconditioner-for-neural-network-least-squares):
its sketch-and-shift construction targets exactly the fast-decaying spectra
those duals show, and it reads the live damping, so one construction serves
the whole solve.

Those helpers are all *frozen* at one linearization point. When the dual
operator rotates enough as LM drifts `x` that a preconditioner built at `x0`
decays into an ineffective approximation — the inner CG stalls or breaks down
several steps in, while rebuilding from the current iterate would keep it
converging — pass a
[`PreconditionerFactory(prepare, apply)`](utilities.md#iterate-adaptive-preconditioner-factory)
instead of `dual_preconditioner`. Its `prepare(x, args, p, aux)` rebuilds the
preconditioner state from the current iterate inside the jitted loop (once per
accepted step; a rejected step reuses the carried state), so keep `prepare`
cheap. It composes with recycling and seeds the implicit derivative's
preconditioner from the state at the solution.

## Recycling and Deflation Across Steps

When a frozen first-level `dual_preconditioner` plateaus above the accuracy bar
— it clusters most of the dual spectrum but leaves a handful of slow modes that
the fixed budget cannot resolve — carry a **deflation basis** across LM steps.
Pass `recycle=RecycleConfig(rank=k)` (requires `linear_solver="gram_cg"`). The
first-level `P` is unchanged; a second-level basis `U` is harvested from each
step's CG iterations (an eigCG-style thick restart) and recycled into the next
step's two-level preconditioner `M_defl(r) = P(r) + U E^{-1}(U'r)`, plus a
deflated, warm-started initial guess, at zero rebuild cost. Across a sequence of
slowly drifting shifted duals the carried basis adapts the *effective*
preconditioner every step, closing the terminal gap a frozen `P` cannot.

- The additive scheme lifts each deflated eigenvalue `λ → λ + 1`, so it
  *clusters* (and speeds CG) when the slow modes are small outliers near 0 and
  `P` normalizes the bulk near 1 — the classic deflation regime. It is a strict
  win precisely when a few isolated modes dominate the residual budget.
- **Prefer a damping-independent first-level `P`** (Sherman-Morrison, Woodbury,
  cholesky-metric — all ignore λ). Eigenvector deflation is shift-invariant for
  the unpreconditioned operator; under a λ-dependent `P` (nystrom, pad) the
  preconditioned Ritz vectors drift with λ, weakening cross-step reuse (still
  helpful, just approximate).
- `RecycleConfig.rank` (`k`) and `window` (`w`, default `max(2·rank, rank+4)`)
  are **static compile knobs** — one program per value. `window` is the primary
  memory knob (a transient `(m, w)` harvest buffer); keep it small.
- Recycling **composes** with the `iterative_maxiter` schedule above: the
  carried basis shrinks the budget each step needs, and the traced schedule
  (still resettable in a callback) then grows it toward the endgame. `warm_start`
  (on by default) reuses the previous dual solution as the initial guess.
- Recycling never changes the converged root or the implicit p-derivative (both
  are defined at the solution); it only accelerates the forward inner solves, and
  the harvest is `stop_gradient`'d.

## What Is Free to Sweep

- **Free (traced, no recompile):** `max_steps`, `atol`/`gtol`/`xtol`, the
  array-valued `LMHyperparams` fields (same dtype; a knob compiled out as
  `None` cannot be switched on), and the *values* of `x0`/`args`/`p`.
  The one exception is `max_steps` with `save_steps=True`: the history
  buffer's shape depends on it, so each distinct value then retraces.
- **Recompiles per value (static):** `linear_solver`,
  `implicit_solver`, the `implicit_*` accuracy knobs,
  `geodesic_acceleration`, `cache_jacobian`, `has_aux`, the `Metric`
  callbacks, `dual_preconditioner`, `preconditioner_factory`,
  `whitened_preconditioner`, `implicit_preconditioner`, `recycle` (the
  `RecycleConfig`, whose `rank`/`window` size the carried basis), and the
  callback function identity.
  Solvers themselves compare by configuration, so a freshly constructed
  solver with equal settings (around the same residual, metric, and
  preconditioner objects) reuses the compiled loop — rebuilding the solver
  per seed in an ensemble loop is free. What still forces a recompile is
  rebuilding the *pieces* per call: an inline `lambda` residual or callback
  at the call site, or a metric/preconditioner reconstructed around fresh
  arrays (unhashable objects key by identity). Define those once at setup
  scope.

For crude hyperparameter search: sweep `init_damping` on a log scale by
replacing the damping in an `init()` state —
`dataclasses.replace(solver.init(x0, args), damping=jnp.asarray(d))`, traced
and recompile-free — and treat the static list as an outer loop of at most a
few compilations.

When sweeping `p` (or running continuation/homotopy), warm-start each solve
with the previous `result.x` — traced, recompile-free, and usually collapses
the step count.

## Failure Signatures

| symptom | likely cause | remedy |
| --- | --- | --- |
| `status == NONFINITE` at step 0 | bad initial point or data | check `residual_fn(x0, ...)` directly |
| `qr` gives non-finite steps; other solvers fine | rank-deficient Jacobian | any other solver: the damped Gram/normal forms (`auto`), or `augmented_qr` / `lsmr` |
| `MAX_STEPS` but loss small and flat | converged without a stopping rule | set `gtol`/`xtol` |
| damping grows without bound (float32 `inf`) | rejection storm | `max_damping`, or check residual scaling |
| every `solve` call recompiles | residual/callback/metric object rebuilt per call (solvers compare by configuration, but their pieces key by identity) | define the pieces once at setup scope |
| implicit `jax.jvp`/`vjp` wrong or zero | `p` not in the residual signature, or perturbing `args` | move perturbed quantities into `p` |
| NaN or no progress with a quasiseparable Matérn metric | nugget-free Matérn-3/2/5/2 Gram conditioning wall (cond ~1e21 at n=5000) | add an absolute `nugget` — it folds into the metric exactly |

## The Metric

In underdetermined problems the metric is not a preconditioner — it selects
*which* solution and *which* implicit derivative you get (minimum-`M`-norm).
For kernel parameterizations use `M = K` (coefficients) or `M = K^{-1}`
(function values); see the [kernel table](gauss_newton.md#choosing-the-metric-with-kernels).
If results look right but derivatives look wrong, check the metric before
anything else.

For kernel blocks plus free scalar parameters, the unified shifted metric
`blockdiag(K, 0) + eps*I` (see
[Utilities](utilities.md#unified-shifted-block-metrics)) replaces the
two-knob `blockdiag(K + jitter*I, m_0*I)` form with one dial. Choosing
`eps`: the selected solution and its implicit derivative are biased O(eps)
away from the pure seminorm limit, while the metric inverse is bounded by
1/eps and the scalar-block dual spike carries weight c²/eps — so smaller
`eps` buys selection accuracy at the price of a harder dual solve (use the
Sherman-Morrison/Woodbury spike preconditioner — measured 3.7-4.8x per cg
step at n=1e3-1e4 with a matrix-free kernel block) and, for the
matrix-free representation, a harder inner solve. In practice the inner CG cost is
dominated by the smooth-kernel spectrum, not the worst-case bound: the
shift clusters the spectral tail, and measured float64 iteration counts
(~32 at n=1000 for Matérn-5/2) are flat in `eps` from 1e-2 to 1e-8. Two
budget notes for `linear_solver="gram_cg"` with a matrix-free metric: total
kernel matvecs = outer CG iterations x inner CG iterations, so the inner
tolerance is the dominant cost knob; and large LM damping hides metric
conditioning (the dual operator is G + lambda*I), so problems can look
easy early and harden near convergence. In float32 the inner CG's
attainable residual (~machine_eps x cond) can sit ABOVE the default
tolerance for small `eps` — the solve then silently burns its full
iteration budget; use float64 or a larger `eps`.

For Matérn value Grams on sorted 1-D points, pick the constructor by
structure, not habit:

| kernel / size | use |
| --- | --- |
| Matérn-1/2 (any `n`) | `metric_from_tridiagonal_precision` — applies are elementwise shifts |
| Matérn-3/2, 5/2, `n` below ~256 | `metric_from_cholesky` of the dense Gram — factorization is cheap and exact |
| Matérn-3/2, 5/2, larger `n` | `metric_from_state_space` with `matern_state_space` — exact O(n) quasiseparable callbacks |

On GPU the scan choice dominates everything (measured on an NVIDIA L40S,
n=1e5, float32): sequential applies take ~3.1–3.6 **seconds** per
solve+norm pair — a kernel launch per scan step — while the associative
(`parallel=True`) applies take ~0.5–0.9 **ms**, a ~3,000–7,000x gap. In
float64 the `parallel=None` default picks the parallel path off-CPU
automatically; in float32 it conservatively stays sequential (the parallel
substitutions have no contraction guarantee), so **on GPU in float32 pass
`parallel=True` explicitly** after checking finiteness on your grid — on
the L40S stress grids all four applies stayed finite and matched the
sequential path to ~1e-7 (well-conditioned) / ~5e-4 (stiff,
conditioning-amplified). On CPU the sequential default is right: at n=1e5
the applies cost ~2.5–4.3 ms and even beat the GPU for the sequential
variant. One caveat: the one-time Cholesky *setup* is a sequential scan —
~0.9 s at n=1e5 on the L40S versus ~2–3 ms on CPU — so when the metric is
rebuilt from traced `sigma`/`ell` inside `jax.grad`/`vmap` sweeps at large
`n` on GPU, setup dominates the step. Reuse a constructed metric across
solves whenever the hyperparameters are fixed; parallel setup is tracked
as a follow-up issue.
