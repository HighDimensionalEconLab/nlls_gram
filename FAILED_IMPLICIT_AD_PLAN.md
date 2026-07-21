# Failed Implicit-AD Simplification Plan

## Summary

Retain the existing stable `jax.custom_jvp` architecture. JAX will continue
deriving the VJP by transposing the linear tangent rule. Do not introduce
`custom_vjp`, HiJAX, separate tangent/cotangent solvers, or new AD algorithms.

Simplify failed-solve handling:

- Remove `failure_ad_reference` from `LevenbergMarquardt.solve`.
- Add the solve-level `max_steps_is_success=True` policy. Keep the diagnostic
  `MAX_STEPS` status, but treat it as usable for implicit AD and default
  multi-start acceptance unless the caller opts into strict handling.
- On every failed result, use the original stop-gradient `(x0, args, p)` as the
  finite AD evaluation point.
- Return zero tangents for `result.x` and `result.aux`; preserve the identity
  tangent through `result.p`.
- Require the initial point to be valid for JVP evaluation of the residual and
  any metric/preconditioner factories.
- Preserve all successful-solve mathematics, solver methods, tolerances, and
  performance.
- Finish nlls completely, then pause for review before touching tinydiffeq.

Keep version 2.4.0 and make this the final unreleased interface.

## Mathematical and Public Contract

For \(r(x,a,p)=0\), define \(J=r_x\), \(K=r_p\), \(P=M^{-1}\), and
\(G=JPJ^\top\). Preserve the implicit tangent

\[
\dot x=-PJ^\top G^+K\dot p,
\]

with the square direct special case

\[
J\dot x=-K\dot p.
\]

Document the automatically transposed cotangent explicitly:

\[
\bar p=-K^\top G^+JP\bar x,
\]

or, for the square direct method, solve \(J^\top\lambda=\bar x\) and return
\(\bar p=-K^\top\lambda\).

For `aux = h(x*(p), args, p)`, document

\[
\dot h=h_x\dot x+h_p\dot p,
\]

and explain that transposition combines an aux cotangent with the root
cotangent before applying the implicit pullback. The pass-through `result.p`
remains independently differentiable.

`LMStatus.CONVERGED` is always AD-successful. `LMStatus.MAX_STEPS` is also
AD-successful by default, preserving fixed-step implicit derivatives while the
returned status remains diagnostic. With `max_steps_is_success=False`, it is a
failure. For every failed status:

- Evaluate the linear tangent program at stop-gradient copies of the original
  `(x0, args, p)`.
- Mask `p_dot` to zero before the implicit solve.
- Return exactly zero `x` and `aux` tangents.
- Let automatic transposition produce zero implicit cotangents without
  encountering invalid failed iterates.
- Keep the primal failed result, status, diagnostics, and possibly nonfinite
  aux unchanged.

The initial point must be JVP-safe, not merely finite: the residual, aux map,
metric factory, and applicable preconditioner factory must have valid
derivatives there. Multi-start uses the caller's original initial point when
the selected winner fails the implicit-AD status policy; this is independent
of whether a custom acceptance hook accepted or rejected the winner.

## Phase 1: nlls Implementation

- Remove `failure_ad_reference` from the public signature, docstring,
  custom-JVP operands, validation helpers, README examples, and implicit-AD
  documentation. Do not add a compatibility alias.
- Pass the original `(x0, args, p)` from both ordinary and multi-start
  custom-JVP rules into the internal tangent construction.
- Add `max_steps_is_success=True` to `solve`; use the same policy for implicit
  AD and built-in multi-start acceptance, while a custom `MultiStart.accept`
  continues to override selection.
- Select returned solution data for converged lanes and stop-gradient initial
  data for failed lanes before evaluating any residual-derived AD operator.
- Preserve all existing `ad_solver*` controls and all direct, SVD, QR,
  augmented-QR, and CG behavior.
- Eliminate the unconditional failed-path aux rebuild:
  - Fixed metrics and direct square solves must not rebuild aux.
  - A `MetricFactory` rebuilds initial aux only when a non-direct AD method
    actually consumes it.
  - A `PreconditionerFactory` rebuilds initial aux only for `gram_cg` when it
    supplies the AD preconditioner.
  - Successful scalar solves reuse `result.aux`; do not add another residual
    evaluation.
  - The returned aux tangent is still computed at selected safe inputs and then
    masked to zero on failure.
- Preserve the existing factory-metric first-order and higher-order contracts.
- Update the README, implicit-AD guide, API docstrings, multi-start
  documentation, and `llms.txt`.

## nlls Tests and Performance Gate

Replace reference-option tests with behavior-focused tests covering mixed
successful/failed `vmap` lanes, deliberately invalid returned iterates,
`result.p` pass-through differentiation, direct and metric-aware methods,
factories, aux evaluation, multi-start, eager/jit execution, pytrees, float32,
float64, successful closed forms, higher-order fixed-metric AD, and zero
derivatives through the initial guess and fixed args.

Before solver edits, add successful-solve implicit-AD benchmarks for direct
square JVP/VJP, aux without a factory, metric-factory AD, and vmapped solves.
Save pre/post CPU JSON and Markdown comparisons. Any successful JVP/VJP
slowdown larger than `max(5%, 1 us)` is a regression to fix. Compare cold
compilation separately and record environment metadata.

Run:

```bash
uv run ruff check .
uv run pytest
uv run --group docs mkdocs build --strict
uv build
JAX_PLATFORMS=cpu uv run --group benchmark pytest benchmarks \
  --benchmark-only \
  --benchmark-json=benchmarks/results/<date>_failed-ad-post.json
```

## Mandatory nlls Review Checkpoint

After nlls code, docs, tests, builds, and benchmarks are complete, do not edit
tinydiffeq. Report the exact API and initial-point precondition, JVP/VJP
equations, aux evaluation behavior, validation results, timing comparison,
diff summary, dirty files, and narrowed behavior. Wait for explicit approval
before committing/pushing nlls or beginning downstream migration.

## Phase 2: tinydiffeq Migration After Approval

- Remove the obsolete nlls argument while retaining tinydiffeq's model-level
  reference for inactive field and aux evaluation.
- Substitute safe inputs before root calls that are already inactive under
  vmapped control flow.
- Require every newly attempted root's actual initial point to be JVP-safe.
- Preserve `LMRootSolver`'s single `ad_solver*` configuration and square-system
  `auto -> direct` behavior.
- Update DAE/SDAE/aux/API docs and tests, run all validation and performance
  suites, and commit/push each repository separately after approval.

## Explicit Non-Goals

- No HiJAX or low-level custom primitive.
- No explicit custom VJP.
- No separate tangent/cotangent solver options.
- No new AD algorithms or regularization changes.
- No augmented-QR redesign.
- No compatibility layer for the removed nlls argument.
- No tinydiffeq work before the nlls review checkpoint.
